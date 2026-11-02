"""Collect one continuous foreground replay under a controlled resource path.

This is a data collector, not a second prediction model.  Owned CPU/GPU replay
requests and owned allocation sizes are four exogenous control coordinates.
Total device utilization is separately observed; it is never renamed external
utilization or used to turn a requested replay into an occupancy percentage.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "profiling", ROOT / "mapping", ROOT / "modeling", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from state_calibration import run
from state_calibration.telemetry import DeviceTelemetrySampler, NativeClock
from resource_calibration.schedule import make_schedules, phase_for_interval, validate_schedule

SCHEMA = "resource-trajectory-campaign-0.1"
MAX_NATIVE_BLOCKS = 1000


def inventory_digest(bindings: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def prepare_campaign(out: Path, binary: Path, runtime_dirs: list[Path], *, cuda_runtime: Path | None = None,
                     phase_duration_ms: int = 1000, host_memory_bytes: int = 64 << 20,
                     device_memory_bytes: int = 64 << 20, seed: int = 80217,
                     repeats: int = 3, smoke: bool = False, mode: str = "mixed") -> dict[str, Any]:
    """Build fresh graphs and references; neither a profile nor CUDA is needed here."""
    schedules = make_schedules(phase_duration_ms=phase_duration_ms, host_memory_bytes=host_memory_bytes,
                              device_memory_bytes=device_memory_bytes, seed=seed, repeats=repeats, smoke=smoke)
    if mode not in ("isolated", "mixed"):
        raise ValueError("foreground mode must be isolated or mixed")
    if 16 * phase_duration_ms > 120_000:
        raise ValueError("total requested holds exceed bounded 120 seconds per trajectory")
    if device_memory_bytes and (cuda_runtime is None or not Path(cuda_runtime).is_file()):
        raise ValueError("device memory holds require an explicit existing CUDA runtime library")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from highlevel_accuracy.bridge import export_plan, reference_ports
    from highlevel_accuracy.cases import build_case
    from highlevel_accuracy.run import case_specs
    specification = next(row["kwargs"] for row in case_specs("representative")
                         if row["id"] == "h128-q16-history0-attention-ffn-cut128")
    compiled = build_case(out / "inputs", **specification).compile()
    sources = run.sources(binary, runtime_dirs)
    sources.update({str(Path(row["path"]).resolve()): row["sha256"] for row in compiled._compiled.sources})
    for path in Path(__file__).parent.glob("*.py"):
        sources[str(path.resolve())] = run.sha(path)
    if cuda_runtime is not None:
        sources[str(Path(cuda_runtime).resolve())] = run.sha(cuda_runtime)
    programs, jobs = {}, []
    for schedule in schedules:
        numeric_seed = 27101 + schedule["repeat"] if schedule["split"] == "train" else 100_000_001 + schedule["repeat"]
        program = export_plan(compiled, seed=numeric_seed)
        path, ref = out / "programs" / (schedule["id"] + ".json"), out / "references" / (schedule["id"] + ".json")
        run.save(path, program)
        run.save(ref, reference_ports(program))
        programs[schedule["id"]] = {"path": str(path), "sha256": run.sha(path), "reference": str(ref),
                                    "reference_sha256": run.sha(ref), "program_sha256": program["program_sha256"], "seed": numeric_seed}
        sources[str(path)] = run.sha(path)
        sources[str(ref)] = run.sha(ref)
        jobs.append({"id": schedule["id"], "split": schedule["split"], "repeat": schedule["repeat"], "seed": numeric_seed,
                     "mode": mode, "program_key": schedule["id"], "schedule": schedule})
    manifest = {"schema": SCHEMA, "protocol_id": SCHEMA + "-" + mode + "-legacy-f32-h128-q16",
                "created_utc": run.utc(), "binary": str(Path(binary).resolve()),
                "runtime_dirs": [str(Path(path).resolve()) for path in runtime_dirs],
                "cuda_runtime": str(Path(cuda_runtime).resolve()) if cuda_runtime is not None else None,
                "sources": sources, "source_inventory_sha256": inventory_digest(sources), "programs": programs, "jobs": jobs,
                "protocol": {"warmup": 20, "native_requested_blocks": MAX_NATIVE_BLOCKS, "minimum_block_ms": 200,
                             "telemetry_interval_ms": 50, "process_timeout_seconds": 180, "ready_timeout_seconds": 30,
                             "child_stop_timeout_seconds": 30, "main_affinity_mask": 3, "cpu_replay_affinity_mask": 3,
                             "gpu_replay_affinity_mask": 12, "execution_boundary": "legacy_double_sync",
                             "state_semantics": "offered_load_controlled", "numeric_dtype": "F32", "smoke": smoke,
                             "holds_begin_after_resource_readiness": True, "forecast_uses_total_telemetry": False,
                             "main_process_continuous_across_phases": True, "core_resource_settings_changed": False,
                             "finite_owned_work_only": True, "cost_fit_inside_collector": False,
                             "fit_config": {"regularization": "1/1000", "include_cpu_gpu_interaction": True,
                                            "min_training_runs": 1 if smoke else 3},
                             "memory_guard": "each owned allocation must not exceed one quarter of observed free physical/device memory"}}
    run.verify_sources(manifest)
    run.save(out / "manifest.json", manifest)
    return manifest


def _json_until_ready(process: Any, path: Path, clock: NativeClock, deadline_tick: int) -> dict[str, Any]:
    while clock.now() < deadline_tick:
        if path.exists():
            try:
                return run.read(path)
            except (OSError, json.JSONDecodeError):
                pass
        if process.poll() is not None:
            raise RuntimeError("owned child exited before complete readiness evidence")
        time.sleep(.01)
    raise TimeoutError("owned child readiness deadline expired")


def _assert_alive(children: list["OwnedChild"]) -> None:
    for child in children:
        if child.process is not None and child.process.poll() is not None:
            raise RuntimeError("owned child exited during offered load: " + child.name)


class OwnedChild:
    """One caller-owned finite process, with explicit graceful and forced cleanup."""
    def __init__(self, name: str, directory: Path, command: list[str], env: dict[str, str],
                 clock: NativeClock, *, affinity_mask: int, ready_timeout: float = 30,
                 stop_timeout: float = 30) -> None:
        self.name, self.directory, self.command, self.clock = name, directory, command, clock
        self.ready_path, self.stop_path = directory / "ready.json", directory / "stop.json"
        self.stop_timeout = stop_timeout
        self.process = None
        self.handles: list[Any] = []
        self.record: dict[str, Any] = {"name": name, "command": command, "launch_tick": clock.now(), "forced_termination": False}
        directory.mkdir(parents=True, exist_ok=False)
        try:
            self.process, stdout, stderr, affinity = run.launch(command, env, directory / "stdout.txt", directory / "stderr.txt", affinity_mask)
            self.handles.extend((stdout, stderr))
            self.record.update(pid=self.process.pid, affinity=affinity)
            ready = _json_until_ready(self.process, self.ready_path, clock, clock.now() + int(ready_timeout * clock.frequency_hz))
            if ready.get("process_id") != self.process.pid or any(ready.get("clock", {}).get(key) != clock.metadata()[key] for key in ("clock_id", "frequency_hz")):
                raise ValueError("owned child readiness identity/clock differs")
            self.record.update(ready=ready, readiness_observed_tick=clock.now())
        except BaseException:
            self.stop(cleanup=True)
            raise

    def stop(self, *, cleanup: bool = False) -> dict[str, Any]:
        if self.process is None:
            return self.record
        if "returncode" in self.record:
            return self.record
        self.record["stop_requested_tick"] = self.clock.now()
        if not self.stop_path.exists():
            run.save(self.stop_path, {"requested_tick": self.record["stop_requested_tick"], "cleanup": cleanup})
        try:
            self.record["returncode"] = self.process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.record["forced_termination"] = True
            try:
                self.record["returncode"] = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.record["killed_after_terminate"] = True
                self.record["returncode"] = self.process.wait(timeout=10)
        finally:
            self.record["exit_observed_tick"] = self.clock.now()
            self.record["process_exited_verified"] = self.process.poll() is not None
            for handle in self.handles:
                handle.close()
            self.handles.clear()
        run.save(self.directory / "process.json", self.record)
        return self.record


def _child_paths(directory: Path) -> tuple[str, str, str]:
    return str(directory / "ready.json"), str(directory / "stop.json"), str(directory / "result.json")


def start_memory_child(directory: Path, *, device: str, size: int, manifest: dict[str, Any], clock: NativeClock) -> OwnedChild:
    ready, stop, result = _child_paths(directory)
    command = [sys.executable, str(Path(__file__).resolve()), "memory-worker", "--device", device, "--bytes", str(size),
               "--ready", ready, "--stop", stop, "--result", result, "--timeout-seconds", "180"]
    if device == "gpu":
        if not manifest.get("cuda_runtime"):
            raise ValueError("explicit CUDA runtime required for owned device allocation")
        command.extend(("--cuda-runtime", manifest["cuda_runtime"]))
    return OwnedChild(device + "_memory", directory, command, dict(os.environ), clock, affinity_mask=12,
                      stop_timeout=manifest["protocol"]["child_stop_timeout_seconds"])


def start_replay_child(directory: Path, *, device: str, manifest: dict[str, Any], binding: dict[str, Any],
                       program: dict[str, Any], clock: NativeClock) -> OwnedChild:
    ready, stop, result = _child_paths(directory)
    env = run.child_environment(manifest["runtime_dirs"])
    env.pop("STATE_PROBE_FIXED_ITERATIONS", None)
    env.update(STATE_PROBE_READY_FILE=ready, STATE_PROBE_STOP_FILE=stop)
    target = run.choose_target(program, device)
    command = [manifest["binary"], binding["path"], "isolated", str(target), "20", str(MAX_NATIVE_BLOCKS), "200", "bare", result]
    return OwnedChild(device + "_replay", directory, command, env, clock,
                      affinity_mask=manifest["protocol"][device + "_replay_affinity_mask"],
                      stop_timeout=manifest["protocol"]["child_stop_timeout_seconds"])


def _validate_memory_result(child: OwnedChild, *, device: str, requested: int) -> None:
    result = run.read(child.directory / "result.json")
    if (child.record.get("returncode") != 0 or child.record.get("forced_termination")
            or result.get("status") != "released" or result.get("process_id") != child.process.pid
            or result.get("device") != device or result.get("requested_bytes") != requested
            or result.get("allocated_bytes") != requested or result.get("released") is not True
            or result.get("release_end_tick", 0) < result.get("allocation_end_tick", 0)):
        raise ValueError("owned memory allocation/release evidence invalid")


def verify_native_timeline(native: dict[str, Any], program: dict[str, Any], *, process_id: int,
                           target: int, mode: str, clock: dict[str, Any], stamped: bool = True) -> None:
    """Verify untruncated samples and an explicit final cooperative stop."""
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    if (native.get("format_version") != "state-calibration-native-v1" or native.get("program_sha256") != program["program_sha256"]
            or native.get("process_id") != process_id or native.get("mode") != mode or native.get("target_action_index") != target
            or native.get("actions") != program["actions"] or native.get("target", {}).get("action") != program["actions"][target]
            or native.get("observed_runtime_flags") != RUNTIME_FLAGS or native.get("execution_boundary") != "legacy_double_sync"):
        raise ValueError("native identity, actions, or runtime boundary differs")
    if any(native.get("clock", {}).get(key) != clock[key] for key in ("clock_id", "frequency_hz")):
        raise ValueError("native clock differs")
    if (native.get("iteration_policy") != "minimum_wall_time" or native.get("requested_iterations_per_block") is not None
            or native.get("full_fixed_count") is not None or native.get("graceful_stop_requested") is not True
            or native.get("requested_blocks") != MAX_NATIVE_BLOCKS or native.get("minimum_block_ms") != 200
            or not native.get("handshake", {}).get("ready_file_requested") or not native.get("handshake", {}).get("stop_file_requested")):
        raise ValueError("native continuous replay/cooperative stop differs")
    expected_instrumentation = "per_iteration_and_per_action_host_completed_work" if stamped else "per_iteration_host_completed_work_only"
    if native.get("instrumentation") != expected_instrumentation:
        raise ValueError("native instrumentation differs")
    blocks = native.get("blocks")
    if not isinstance(blocks, list) or not blocks or len(blocks) > MAX_NATIVE_BLOCKS:
        raise ValueError("native block envelope invalid")
    last = native["ready_tick"]
    indices = [target] if mode == "isolated" else list(range(len(program["actions"])))
    for bi, block in enumerate(blocks):
        begin, end = block["begin_tick"], block["end_tick"]
        if block.get("block_index") != bi or not last <= begin < end:
            raise ValueError("native block chronology invalid")
        wall = round((end - begin) * 1e9 / clock["frequency_hz"])
        if abs(wall - block.get("wall_elapsed_ns", -1)) > 1 or block.get("minimum_elapsed_satisfied") != (wall >= 200_000_000):
            raise ValueError("native block duration invalid")
        if bi + 1 < len(blocks) and not block["minimum_elapsed_satisfied"]:
            raise ValueError("only stopped final block may be short")
        if block.get("fixed_iteration_count_satisfied") is not None or block.get("target_diagnostics") != []:
            raise ValueError("native alternate sampling/diagnostics unexpected")
        iterations, actions = block["iterations"], block["action_intervals"]
        if not iterations or len(iterations) != block.get("iteration_count") or len(actions) != (len(iterations) * len(indices) if stamped else 0):
            raise ValueError("native sample/action coverage invalid")
        previous, completed = begin, 0
        for index, item in enumerate(iterations):
            left, right = item["begin_tick"], item["end_tick"]
            duration = round((right - left) * 1e9 / clock["frequency_hz"])
            if item.get("iteration") != index or not previous <= left < right <= end or abs(item["elapsed_ns"] - duration) > 1:
                raise ValueError("native sample chronology/duration invalid")
            completed += item["elapsed_ns"]
            previous = right
            cursor = left
            for ai, action in zip(indices, actions[index * len(indices):(index + 1) * len(indices)] if stamped else []):
                a, b = action["begin_tick"], action["end_tick"]
                if action.get("iteration") != index or action.get("action_index") != ai or not cursor <= a <= b <= right or abs(action["elapsed_ns"] - round((b - a) * 1e9 / clock["frequency_hz"])) > 1:
                    raise ValueError("native action chronology/duration invalid")
                cursor = b
        if completed != block.get("completed_work_ns"):
            raise ValueError("native completed work sum invalid")
        last = end
    if last > native.get("stop_observed_tick", -1):
        raise ValueError("native stop precedes last invocation")


def extract_observations(native: dict[str, Any], events: list[dict[str, Any]], *, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain every target and whole-invocation observation, including crossings."""
    rows = []
    target = native["target_action_index"]
    for block in native["blocks"]:
        for metric, items in (("iteration", block["iterations"]),
                              ("target", [a for a in block["action_intervals"] if a["action_index"] == target])):
            for item in items:
                rows.append({"id": f"{job['id']}-{metric}-{block['block_index']}-{item['iteration']}",
                             "run_id": job["id"], "split": job["split"], "seed": job["seed"], "metric": metric,
                             "block_index": block["block_index"], "iteration": item["iteration"],
                             "begin_tick": item["begin_tick"], "end_tick": item["end_tick"], "elapsed_ns": item["elapsed_ns"],
                             "phase": phase_for_interval(events, item["begin_tick"], item["end_tick"]),
                             "state_semantics": "offered_load_controlled", "total_telemetry_used_as_predictor": False})
    return rows


def _clipped_coverage(intervals: list[tuple[int, int]], begin: int, end: int) -> dict[str, Any]:
    if end <= begin:
        raise ValueError("coverage window must be positive")
    clipped = sorted((max(a, begin), min(b, end)) for a, b in intervals if a < end and begin < b)
    merged: list[list[int]] = []
    for left, right in clipped:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    covered = sum(b - a for a, b in merged)
    return {"covered_ticks": covered, "window_ticks": end - begin, "coverage_fraction": f"{covered}/{end-begin}",
            "overlapping_invocation_count": len(clipped), "scope": "union of completed native invocation brackets including waiting; not active utilization"}


def attach_resource_evidence(events: list[dict[str, Any]], children: list[OwnedChild]) -> None:
    """Audit offered worker lifetime and owned payload; no utilization invention."""
    by_pid = {child.process.pid: child for child in children}
    for event in events:
        evidence = {}
        for name, pid in event["active_resource_pids"].items():
            child = by_pid[pid]
            result = run.read(child.directory / "result.json")
            if name.endswith("replay"):
                intervals = [(item["begin_tick"], item["end_tick"]) for block in result["blocks"] for item in block["iterations"]]
                evidence[name] = {"pid": pid, "numeric_verified": True,
                                  **_clipped_coverage(intervals, event["steady_begin_tick"], event["steady_end_tick"])}
            else:
                held = result["allocation_end_tick"] <= event["steady_begin_tick"] <= event["steady_end_tick"] <= result["release_begin_tick"]
                evidence[name] = {"pid": pid, "requested_bytes": result["requested_bytes"], "allocated_bytes": result["allocated_bytes"],
                                  "allocation_lifetime_covers_phase": held, "resident_bytes_verified": False,
                                  "allocation_end_tick": result["allocation_end_tick"], "release_begin_tick": result["release_begin_tick"]}
                if not held:
                    raise ValueError("owned allocation lifetime does not cover requested phase")
        event["resource_evidence"] = evidence


def execute_timeline(out: Path, manifest: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """One process survives all phases; command intervals never masquerade as holds."""
    from resource_calibration.counters import ProcessCounterSampler
    schedule = validate_schedule(job["schedule"])
    directory = Path(out) / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=False)
    binding, protocol = manifest["programs"][job["program_key"]], manifest["protocol"]
    for name, digest in ((binding["path"], binding["sha256"]), (binding["reference"], binding["reference_sha256"])):
        if run.sha(name) != digest:
            raise ValueError("program/reference changed")
    program, reference = run.read(binding["path"]), run.read(binding["reference"])
    target, clock = run.choose_target(program), NativeClock()
    env = run.child_environment(manifest["runtime_dirs"])
    env.pop("STATE_PROBE_FIXED_ITERATIONS", None)
    ready, stop, result = _child_paths(directory / "main")
    env.update(STATE_PROBE_READY_FILE=ready, STATE_PROBE_STOP_FILE=stop)
    command = [manifest["binary"], binding["path"], job["mode"], str(target), str(protocol["warmup"]), str(MAX_NATIVE_BLOCKS), "200", "stamped", result]
    row: dict[str, Any] = {"schema": SCHEMA, "job": job, "started_utc": run.utc(), "launch_tick": clock.now(),
                           "source_inventory_sha256": manifest["source_inventory_sha256"], "program_sha256": binding["sha256"],
                           "native_clock": clock.metadata(), "summary": {}, "children": []}
    all_children: list[OwnedChild] = []
    active: dict[str, tuple[Any, OwnedChild]] = {}
    events, main, sampler, counters = [], None, None, None
    deadline = clock.now() + protocol["process_timeout_seconds"] * clock.frequency_hz
    try:
        sampler = DeviceTelemetrySampler(interval_ms=protocol["telemetry_interval_ms"]).start()
        main = OwnedChild("main", directory / "main", command, env, clock, affinity_mask=protocol["main_affinity_mask"],
                          stop_timeout=protocol["child_stop_timeout_seconds"])
        all_children.append(main)
        counters = ProcessCounterSampler(handles={"main": main.process._handle} if os.name == "nt" else {},
                                         interval_ms=protocol["telemetry_interval_ms"], clock=clock).start()
        for phase in schedule["phases"]:
            _assert_alive([main, *[child for _, child in active.values()]])
            requested = phase["resources"]
            event = {"phase_index": phase["index"], "requested_hold_ms": phase["hold_ms"],
                     "requested_resources": requested, "transition_begin_tick": clock.now(), "resource_commands": []}
            events.append(event)
            controls = {"cpu_replay": requested["cpu_replay"], "gpu_replay": requested["gpu_replay"],
                        "host_memory": requested["host_memory_bytes"], "device_memory": requested["device_memory_bytes"]}
            # Stop all removed/replaced resources before starting new resources.
            for name, (old_value, child) in list(active.items()):
                if controls[name] != old_value:
                    receipt = child.stop()
                    if receipt["returncode"] != 0 or receipt["forced_termination"]:
                        raise RuntimeError("owned child failed during phase transition")
                    event["resource_commands"].append({"resource": name, "operation": "stop", "receipt": receipt})
                    del active[name]
            for name, value in controls.items():
                if value and name not in active:
                    path = directory / "children" / f"phase-{phase['index']}-{name}"
                    if name.endswith("replay"):
                        child = start_replay_child(path, device=name.split("_")[0], manifest=manifest, binding=binding, program=program, clock=clock)
                    else:
                        child = start_memory_child(path, device="cpu" if name == "host_memory" else "gpu", size=value, manifest=manifest, clock=clock)
                    all_children.append(child)
                    active[name] = (value, child)
                    event["resource_commands"].append({"resource": name, "operation": "start", "requested_value": value,
                                                       "pid": child.process.pid, "ready": child.record["ready"],
                                                       "readiness_observed_tick": child.record["readiness_observed_tick"]})
            if os.name == "nt":
                counters.set_handles({"main": main.process._handle, **{name: child.process._handle for name, (_, child) in active.items()}})
            event["active_resource_pids"] = {name: child.process.pid for name, (_, child) in active.items()}
            event["steady_begin_tick"] = clock.now()
            hold_end = event["steady_begin_tick"] + phase["hold_ms"] * clock.frequency_hz // 1000
            while clock.now() < hold_end:
                if clock.now() >= deadline:
                    raise TimeoutError("bounded foreground timeline deadline expired")
                _assert_alive([main, *[child for _, child in active.values()]])
                time.sleep(min(.02, max(0, (hold_end - clock.now()) / clock.frequency_hz)))
            event["steady_end_tick"] = clock.now()
            event["transition_elapsed_ns"] = round((event["steady_begin_tick"] - event["transition_begin_tick"]) * 1e9 / clock.frequency_hz)
        main.stop()
        if main.record["returncode"] != 0 or main.record["forced_termination"]:
            raise RuntimeError("foreground failed or required forced termination")
        for _, child in active.values():
            child.stop()
        active.clear()
        native = run.read(result)
        verify_native_timeline(native, program, process_id=main.process.pid, target=target, mode=job["mode"], clock=clock.metadata())
        numeric = run.validate_numeric(native, reference)
        run.save(directory / "numerical.json", numeric)
        for child in all_children:
            if child is main:
                continue
            if child.name.endswith("_memory"):
                ready_row = child.record["ready"]
                _validate_memory_result(child, device=ready_row["device"], requested=ready_row["requested_bytes"])
            else:
                bg = run.read(child.directory / "result.json")
                device = child.name.split("_")[0]
                verify_native_timeline(bg, program, process_id=child.process.pid, target=run.choose_target(program, device),
                                       mode="isolated", clock=clock.metadata(), stamped=False)
                run.validate_numeric(bg, reference)
        attach_resource_evidence(events, all_children)
        observations = extract_observations(native, events, job=job)
        run.save(directory / "observations.json", observations)
        row.update(status="measured", process_id=main.process.pid,
                   summary={"native_numeric_passed": True, "phase_count": len(events), "observations": len(observations),
                            "steady_observations": sum(item["phase"]["kind"] == "steady" for item in observations),
                            "transition_or_outside_observations": sum(item["phase"]["kind"] != "steady" for item in observations),
                            "one_foreground_pid": True, "cost_fitted": False, "accuracy_acceptance_claimed": False})
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        cleanup_errors = []
        for child in reversed(all_children):
            try:
                child.stop(cleanup=True)
            except Exception as error:
                cleanup_errors.append({"child": child.name, "error": str(error)})
        for label, instance, raw_name, metadata_name in (("counters", counters, "counters.jsonl", "counter-metadata.json"),
                                                        ("telemetry", sampler, "telemetry.jsonl", "telemetry-metadata.json")):
            if instance is not None:
                # Independent attempts ensure a sampler write failure cannot
                # prevent the other sampler's shutdown or final raw receipts.
                for operation, action in (("stop", instance.stop),
                                           ("write", lambda instance=instance, raw_name=raw_name: instance.write_jsonl(directory / raw_name)),
                                           ("metadata", lambda instance=instance, metadata_name=metadata_name: run.save(directory / metadata_name, instance.metadata()))):
                    try:
                        action()
                    except Exception as error:
                        cleanup_errors.append({"child": label, "operation": operation, "error": str(error)})
        row["children"] = [child.record for child in all_children]
        row["cleanup_verified"] = bool(all_children) and all(child.record.get("process_exited_verified") for child in all_children) and not cleanup_errors
        row["cleanup_errors"] = cleanup_errors
        if cleanup_errors or not row["cleanup_verified"]:
            row.update(status="failed", cleanup_failure=True)
        run.save(directory / "timeline.json", {"schema": SCHEMA, "clock": clock.metadata(), "schedule": schedule, "events": events,
                                              "partial_events_retained": row.get("status") != "measured"})
        row.update(completed_utc=run.utc(), completed_tick=clock.now())
        row["artifacts_sha256"] = {str(path.relative_to(directory)).replace("\\", "/"): run.sha(path)
                                   for path in directory.rglob("*") if path.is_file()}
        run.save(directory / "execution.json", row)
    return row


def collect_campaign(out: Path, split: str, *, frozen_model: Path | None = None) -> list[dict[str, Any]]:
    if split not in ("train", "heldout"):
        raise ValueError("split must be train or heldout")
    out = Path(out).resolve()
    manifest = run.read(out / "manifest.json")
    manifest_hash = run.sha(out / "manifest.json")
    if manifest.get("schema") != SCHEMA or inventory_digest(manifest["sources"]) != manifest.get("source_inventory_sha256"):
        raise ValueError("campaign/source inventory invalid")
    run.verify_sources(manifest)
    barrier = None
    if split == "heldout":
        if frozen_model is None or not Path(frozen_model).is_file() or not (out / "train-rows.json").is_file():
            raise ValueError("heldout requires a frozen training-only model and completed training rows")
        training = run.read(out / "train-rows.json")
        if not training or any(row.get("status") != "measured" for row in training):
            raise ValueError("all training trajectories must succeed before heldout")
        from resource_calibration.evaluate import load_frozen
        _, model = load_frozen(out, frozen_model, verify_training=True)
        clock = NativeClock()
        if model.frozen_at_ns > clock.now() * 1_000_000_000 // clock.frequency_hz:
            raise ValueError("model freeze cutoff lies in the future")
        barrier = {"frozen_model": str(Path(frozen_model).resolve()), "model_sha256": run.sha(frozen_model),
                   "train_rows_sha256": run.sha(out / "train-rows.json"), "manifest_sha256": manifest_hash,
                   "created_utc": run.utc(), "frozen_before_any_heldout_launch": True}
        run.save(out / "heldout-barrier.json", barrier)
    rows = []
    with run.measurement_lock():
        for job in (job for job in manifest["jobs"] if job["split"] == split):
            run.verify_sources(manifest)
            if barrier is not None and run.sha(frozen_model) != barrier["model_sha256"]:
                raise ValueError("frozen model changed before heldout launch")
            row = execute_timeline(out, manifest, job)
            rows.append(row)
            print(json.dumps({"id": job["id"], "status": row["status"], "summary": row["summary"], "error": row.get("error")}), flush=True)
        run.verify_sources(manifest)
    if run.sha(out / "manifest.json") != manifest_hash:
        raise ValueError("manifest changed during measurement")
    if barrier is not None and (run.sha(frozen_model) != barrier["model_sha256"] or run.sha(out / "train-rows.json") != barrier["train_rows_sha256"]):
        raise ValueError("heldout altered frozen training artifacts")
    run.save(out / (split + "-rows.json"), rows)
    return rows


def _host_available_memory() -> int:
    from resource_calibration.counters import read_snapshot
    snapshot = read_snapshot()
    value = snapshot.get("host_memory", {}).get("values", {}).get("available_physical_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("host available physical memory is unknown; allocation rejected")
    return value


def _cuda_api(path: Path) -> Any:
    if not Path(path).is_file():
        raise ValueError("explicit CUDA runtime library does not exist")
    library = ctypes.WinDLL(str(Path(path).resolve())) if os.name == "nt" else ctypes.CDLL(str(Path(path).resolve()))
    signatures = {"cudaSetDevice": ([ctypes.c_int], ctypes.c_int), "cudaMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
                  "cudaMemset": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
                  "cudaFree": ([ctypes.c_void_p], ctypes.c_int), "cudaDeviceSynchronize": ([], ctypes.c_int),
                  "cudaMemGetInfo": ([ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int)}
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


def _cuda_check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA runtime status {code}")


def memory_worker(*, device: str, size: int, ready: Path, stop: Path, result: Path,
                  timeout_seconds: float = 180, cuda_runtime: Path | None = None) -> dict[str, Any]:
    """Hold and explicitly free one bounded allocation, never simulate compute load."""
    from resource_calibration.schedule import MAX_MEMORY_BYTES
    if device not in ("cpu", "gpu") or isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_MEMORY_BYTES:
        raise ValueError("invalid bounded owned allocation")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 180 or len({Path(path).resolve() for path in (ready, stop, result)}) != 3
            or any(Path(path).exists() for path in (ready, stop, result))):
        raise ValueError("memory worker requires fresh handshake paths and finite timeout")
    clock = NativeClock()
    row = {"schema": "owned-memory-hold-0.1", "process_id": os.getpid(), "clock": clock.metadata(), "device": device,
           "requested_bytes": size, "allocation_begin_tick": clock.now(), "allocated_bytes": 0,
           "released": False, "resident_bytes_verified": False, "status": "allocating"}
    allocation = None
    cuda = None
    pointer = ctypes.c_void_p()
    try:
        if device == "cpu":
            available = _host_available_memory()
            if size > available // 4:
                raise MemoryError("owned host allocation exceeds one quarter of observed free memory")
            allocation = bytearray(size)
            for index in range(0, size, 4096):
                allocation[index] = 1
            allocation[-1] = 1
            row.update(observed_available_before_bytes=available, allocation_api="bytearray with page touch", allocated_bytes=len(allocation))
        else:
            if cuda_runtime is None:
                raise ValueError("explicit CUDA runtime required")
            cuda = _cuda_api(cuda_runtime)
            _cuda_check(cuda.cudaSetDevice(0), "cudaSetDevice")
            # Establish this owned context before measuring free memory. Context
            # overhead remains a separate uncontrolled delta, not owned payload.
            _cuda_check(cuda.cudaFree(None), "cudaFree(context initialization)")
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            _cuda_check(cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
            if size > free.value // 4:
                raise MemoryError("owned device allocation exceeds one quarter of observed free memory")
            _cuda_check(cuda.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
            _cuda_check(cuda.cudaMemset(pointer, 0, size), "cudaMemset")
            _cuda_check(cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
            row.update(observed_available_before_bytes=free.value, observed_total_bytes=total.value,
                       allocation_api="explicit CUDA runtime cudaMalloc/cudaMemset", allocated_bytes=size,
                       cuda_runtime_sha256=run.sha(cuda_runtime), cuda_device=0,
                       context_and_allocator_overhead_bytes=None)
        row.update(status="holding", allocation_end_tick=clock.now(), ready_tick=clock.now())
        run.save(ready, row)
        deadline = clock.now() + int(timeout_seconds * clock.frequency_hz)
        while not Path(stop).exists():
            if clock.now() >= deadline:
                raise TimeoutError("owned memory holder finite deadline expired")
            time.sleep(.02)
        row["stop_observed_tick"] = clock.now()
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        row["release_begin_tick"] = clock.now()
        try:
            if cuda is not None and pointer.value:
                _cuda_check(cuda.cudaFree(pointer), "cudaFree")
                pointer = ctypes.c_void_p()
            allocation = None
            gc.collect()
            row.update(released=True, release_end_tick=clock.now())
            if row["status"] == "holding":
                row["status"] = "released"
        except Exception as error:
            row.update(status="failed", release_error=str(error))
        run.save(result, row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--binary", type=Path, required=True)
    prepare.add_argument("--runtime-dir", type=Path, action="append", required=True)
    prepare.add_argument("--cuda-runtime", type=Path)
    prepare.add_argument("--phase-ms", type=int, default=1000)
    prepare.add_argument("--host-memory-bytes", type=int, default=64 << 20)
    prepare.add_argument("--device-memory-bytes", type=int, default=64 << 20)
    prepare.add_argument("--seed", type=int, default=80217)
    prepare.add_argument("--repeats", type=int, default=3)
    prepare.add_argument("--mode", choices=("isolated", "mixed"), default="mixed")
    prepare.add_argument("--smoke", action="store_true")
    collect = sub.add_parser("collect")
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--split", choices=("train", "heldout"), required=True)
    collect.add_argument("--frozen-model", type=Path)
    worker = sub.add_parser("memory-worker")
    worker.add_argument("--device", choices=("cpu", "gpu"), required=True)
    worker.add_argument("--bytes", type=int, required=True)
    worker.add_argument("--ready", type=Path, required=True)
    worker.add_argument("--stop", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--timeout-seconds", type=float, default=180)
    worker.add_argument("--cuda-runtime", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare_campaign(args.out, args.binary, args.runtime_dir, cuda_runtime=args.cuda_runtime,
                                 phase_duration_ms=args.phase_ms, host_memory_bytes=args.host_memory_bytes,
                                 device_memory_bytes=args.device_memory_bytes, seed=args.seed, repeats=args.repeats,
                                 smoke=args.smoke, mode=args.mode)
        print(json.dumps({"prepared": True, "jobs": len(value["jobs"]), "hardware_executed": False}), flush=True)
        return 0
    if args.command == "collect":
        rows = collect_campaign(args.out, args.split, frozen_model=args.frozen_model)
        return 0 if rows and all(row["status"] == "measured" for row in rows) else 2
    value = memory_worker(device=args.device, size=args.bytes, ready=args.ready, stop=args.stop, result=args.result,
                          timeout_seconds=args.timeout_seconds, cuda_runtime=args.cuda_runtime)
    return 0 if value["status"] == "released" else 2


if __name__ == "__main__":
    raise SystemExit(main())
