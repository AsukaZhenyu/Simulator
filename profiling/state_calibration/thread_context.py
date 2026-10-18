"""Bounded native thread-configuration diagnostic using an existing program.

Only physical.threads and its dependent content identities change. This is not
a recompilation, cost fit, heldout accuracy claim, or isolated polling test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "profiling", ROOT / "mapping", ROOT / "modeling", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from state_calibration import run
from state_calibration.telemetry import DeviceTelemetrySampler, NativeClock, align_intervals

SCHEMA = "CPU-thread-context-diagnostic-0.1"
LIMITS = {
    "intervention": "CPU backend/threadpool requested thread count 1 versus 2, including all CPU-operation and surrounding runtime-context effects. Polling policy is unchanged and is not isolated.",
    "graph": "Exact existing compiled graph, actions, bindings, seed and tensor data. Only physical.threads and dependent identity hashes differ. No strategy recompilation or cost-table update is claimed.",
    "thread_evidence": "The bound native Executor reads physical.threads and configures the backend and persistent pool. Direct OS worker-count readback is not available in this native binary.",
    "statistics": "Minimum-wall-time sampling retains every invocation. Counts may differ. Process medians and paired ratios are descriptive; pooled equal-count calibration gates are not applied.",
    "causality": "Controlled requested-configuration comparison with matched affinity and numeric inputs. Device state and external applications remain observed, uncontrolled context; no pure polling or GPU-kernel causal attribution.",
    "scope": "No cost fitting, heldout claim, outlier removal, interpolation, universal slowdown multiplier or hardware-setting changes.",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_variant(program: dict[str, Any], threads: int) -> dict[str, Any]:
    """Change exactly one physical control plus its two dependent identities."""
    from highlevel_accuracy.bridge import sha as content_sha, validate_program
    if isinstance(threads, bool) or not isinstance(threads, int) or threads not in (1, 2):
        raise ValueError("diagnostic thread count must be exactly 1 or 2")
    validate_program(program)
    result = deepcopy(program)
    result["physical"]["threads"] = threads
    result["identity"]["physical"] = content_sha(result["physical"])
    result.pop("program_sha256")
    result["program_sha256"] = content_sha(result)
    validate_program(result)
    restored = deepcopy(result)
    restored["physical"]["threads"] = program["physical"]["threads"]
    restored["identity"]["physical"] = program["identity"]["physical"]
    restored["program_sha256"] = program["program_sha256"]
    if restored != program:
        raise ValueError("thread intervention changed an undeclared field")
    return result


def _ns(begin: int, end: int, frequency: int) -> int:
    value = Fraction((end - begin) * 1_000_000_000, frequency)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _integer(value: Any, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError("invalid " + name)
    return value


def verify_native(native: dict[str, Any], program: dict[str, Any], job: dict[str, Any], protocol: dict[str, Any], clock: dict[str, Any], process_id: int) -> None:
    """Verify minimum-wall-time sampling without the fixed-count verifier."""
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    target = run.choose_target(program)
    if (native.get("format_version") != "state-calibration-native-v1" or native.get("program_sha256") != program["program_sha256"]
            or native.get("process_id") != process_id or native.get("mode") != job["mode"]
            or native.get("target_action_index") != target or native.get("actions") != program["actions"]
            or native.get("target", {}).get("action") != program["actions"][target]):
        raise ValueError("native identity/action trace differs")
    if program["physical"]["threads"] != job["threads"]:
        raise ValueError("input requested thread configuration differs")
    if native.get("observed_runtime_flags") != RUNTIME_FLAGS or native.get("execution_boundary") != "legacy_double_sync":
        raise ValueError("runtime flags or execution boundary differs")
    if any(native.get("clock", {}).get(key) != clock[key] for key in ("clock_id", "frequency_hz")):
        raise ValueError("native clock differs")
    if (native.get("iteration_policy") != "minimum_wall_time" or native.get("requested_iterations_per_block") is not None
            or native.get("full_fixed_count") is not None or native.get("graceful_stop_requested")
            or native.get("requested_blocks") != protocol["blocks"] or native.get("warmup_iterations") != protocol["warmup"]
            or native.get("minimum_block_ms") != protocol["minimum_block_ms"] or len(native.get("blocks", [])) != protocol["blocks"]):
        raise ValueError("minimum-wall-time sampling envelope differs")
    if native.get("instrumentation") != "per_iteration_and_per_action_host_completed_work":
        raise ValueError("per-action instrumentation required")
    frequency = _integer(clock["frequency_hz"], "clock frequency", True)
    if not (native["diagnostic_begin_tick"] <= native["warmup_begin_tick"] <= native["warmup_end_tick"] <= native["ready_tick"]):
        raise ValueError("warmup chronology differs")
    last = native["ready_tick"]
    for block_index, block in enumerate(native["blocks"]):
        begin = _integer(block.get("begin_tick"), "block begin")
        end = _integer(block.get("end_tick"), "block end")
        if block.get("block_index") != block_index or not last <= begin < end:
            raise ValueError("block chronology differs")
        wall = _integer(block.get("wall_elapsed_ns"), "block wall", True)
        if abs(wall - _ns(begin, end, frequency)) > 1 or wall < protocol["minimum_block_ms"] * 1_000_000 or block.get("minimum_elapsed_satisfied") is not True:
            raise ValueError("block minimum duration differs")
        if block.get("fixed_iteration_count_satisfied") is not None or block.get("target_diagnostics"):
            raise ValueError("unexpected fixed-count or alternate-boundary record")
        samples = block.get("iterations")
        if not isinstance(samples, list) or not samples or block.get("iteration_count") != len(samples):
            raise ValueError("invocation coverage differs")
        actions = block.get("action_intervals")
        expected = [target] if job["mode"] == "isolated" else list(range(len(program["actions"])))
        if not isinstance(actions, list) or len(actions) != len(samples) * len(expected):
            raise ValueError("action coverage differs")
        previous, completed = begin, 0
        for index, sample in enumerate(samples):
            left, right = sample["begin_tick"], sample["end_tick"]
            if sample.get("iteration") != index or not previous <= left < right <= end:
                raise ValueError("invocation chronology differs")
            duration = _integer(sample.get("elapsed_ns"), "invocation elapsed", True)
            if abs(duration - _ns(left, right, frequency)) > 1:
                raise ValueError("invocation duration differs")
            completed += duration
            previous = right
            cursor = left
            for action_index, action in zip(expected, actions[index * len(expected):(index + 1) * len(expected)]):
                start, stop = action["begin_tick"], action["end_tick"]
                if action.get("iteration") != index or action.get("action_index") != action_index or not cursor <= start <= stop <= right:
                    raise ValueError("action identity/chronology differs")
                if abs(action["elapsed_ns"] - _ns(start, stop, frequency)) > 1:
                    raise ValueError("action duration differs")
                cursor = stop
        if completed != block.get("completed_work_ns"):
            raise ValueError("completed-work sum differs")
        last = end
    if last > native["diagnostic_end_tick"]:
        raise ValueError("diagnostic completion precedes final block")


def prepare(campaign: Path, out: Path) -> dict[str, Any]:
    from highlevel_accuracy.bridge import reference_ports
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    formal_path = campaign / "manifest.json"
    formal_digest = run.sha(formal_path)
    formal = run.read(formal_path)
    run.verify_sources(formal)
    sources = run.sources(formal["binary"], formal["runtime_dirs"])
    sources[str(formal_path)] = formal_digest
    programs, source_programs = {}, {}
    for repeat in range(3):
        original = formal["programs"]["h128-q16"][f"profile-{repeat}"]
        for path, digest in ((original["path"], original["sha256"]), (original["reference"], original["reference_sha256"])):
            if run.sha(path) != digest:
                raise ValueError("original program/reference bytes differ")
            sources[str(Path(path).resolve())] = digest
        source = run.read(original["path"])
        if (source["physical"]["config"]["hidden"] != 128 or source["physical"]["threads"] != 2
                or any(binding["shape"]["tokens"] != 16 for binding in source["bindings"])):
            raise ValueError("exact original H128 two-thread program required")
        reference = run.read(original["reference"])
        if reference_ports(source) != reference:
            raise ValueError("independent original reference differs")
        source_programs[str(repeat)] = {"program_sha256": source["program_sha256"], "file_sha256": original["sha256"], "seed": source["seed"]}
        for threads in (1, 2):
            program = thread_variant(source, threads)
            if reference_ports(program) != reference:
                raise ValueError("thread-only intervention changed independent numerical reference")
            name = f"repeat-{repeat}-threads-{threads}"
            path, ref = out / "programs" / (name + ".json"), out / "references" / (name + ".json")
            run.save(path, program)
            run.save(ref, reference)
            programs[name] = {"path": str(path), "sha256": run.sha(path), "reference": str(ref), "reference_sha256": run.sha(ref),
                              "program_sha256": program["program_sha256"], "seed": program["seed"], "threads": threads}
            sources[str(path)], sources[str(ref)] = run.sha(path), run.sha(ref)
    jobs = []
    for repeat in range(3):
        modes = ("isolated", "mixed") if repeat % 2 == 0 else ("mixed", "isolated")
        for mode_index, mode in enumerate(modes):
            configurations = (1, 2) if (repeat + mode_index) % 2 == 0 else (2, 1)
            for threads in configurations:
                key = f"repeat-{repeat}-threads-{threads}"
                jobs.append({"id": f"repeat-{repeat}-threads-{threads}-{mode}", "split": "diagnostic", "repeat": repeat,
                             "threads": threads, "mode": mode, "program_key": key, "seed": programs[key]["seed"], "condition": "baseline",
                             "instrumentation": "stamped", "telemetry": True, "execution_boundary": "legacy_double_sync"})
    protocol = {"blocks": 3, "minimum_block_ms": 200, "warmup": 20, "interval_ms": 50,
                "process_timeout_seconds": 180, "affinity_mask": 3, "iteration_policy": "minimum_wall_time",
                "fixed_iterations_environment_unset": True, "fresh_process_count": 12,
                "cost_fit": False, "heldout_accuracy_claimed": False, "equal_invocation_counts_claimed": False}
    manifest = {"schema": SCHEMA, "created_utc": utc(), "binary": formal["binary"], "runtime_dirs": formal["runtime_dirs"],
                "sources": sources, "programs": programs, "original_programs": source_programs, "jobs": jobs, "protocol": protocol, "limits": LIMITS}
    run.verify_sources(manifest)
    run.save(out / "manifest.json", manifest)
    return manifest


def _measure(out: Path, manifest: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    directory = out / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=False)
    binding, protocol = manifest["programs"][job["program_key"]], manifest["protocol"]
    program, reference = run.read(binding["path"]), run.read(binding["reference"])
    env = run.child_environment(manifest["runtime_dirs"])
    env.pop("STATE_PROBE_FIXED_ITERATIONS", None)
    env["STATE_PROBE_EXECUTION_BOUNDARY"] = "legacy_double_sync"
    target = run.choose_target(program)
    native_path = directory / "native.json"
    command = [manifest["binary"], binding["path"], job["mode"], str(target), str(protocol["warmup"]), str(protocol["blocks"]),
               str(protocol["minimum_block_ms"]), "stamped", str(native_path)]
    clock = NativeClock()
    row = {"job": job, "started_utc": utc(), "launch_tick": clock.now(), "command": command,
           "requested_threads": job["threads"], "program_sha256": binding["sha256"], "summary": {},
           "fixed_iterations_environment_unset": "STATE_PROBE_FIXED_ITERATIONS" not in env}
    sampler, process, handles = None, None, []
    try:
        sampler = DeviceTelemetrySampler(interval_ms=protocol["interval_ms"]).start()
        process, stdout, stderr, affinity = run.launch(command, env, directory / "native.stdout", directory / "native.stderr", protocol["affinity_mask"])
        handles.extend((stdout, stderr))
        row.update(process_id=process.pid, affinity=affinity)
        row["returncode"] = process.wait(timeout=protocol["process_timeout_seconds"])
        row["exit_tick"] = clock.now()
        sampler.stop()
        sampler.write_jsonl(directory / "telemetry.jsonl")
        run.save(directory / "telemetry-metadata.json", sampler.metadata())
        if row["returncode"] != 0:
            raise ValueError("native nonzero exit; raw stdout/stderr retained")
        native = run.read(native_path)
        verify_native(native, program, job, protocol, clock.metadata(), process.pid)
        numeric = run.validate_numeric(native, reference)
        run.save(directory / "numerical.json", numeric)
        alignment = align_intervals(native["blocks"], sampler.records, sampler.metadata()["clock"], interval_ms=protocol["interval_ms"], action_clock=native["clock"])
        run.save(directory / "alignment.json", alignment)
        targets = [item["elapsed_ns"] for block in native["blocks"] for item in block["action_intervals"] if item["action_index"] == target]
        iterations = [item["elapsed_ns"] for block in native["blocks"] for item in block["iterations"]]
        state = {key: statistics.median([block["observed_fields"][key]["median"] for block in alignment if key in block["observed_fields"]])
                 for key in sorted({key for block in alignment for key in block["observed_fields"]})}
        row["summary"] = {"target_median_ns": statistics.median(targets), "target_cv": run.cv(targets), "target_samples_ns": targets,
                          "iteration_median_ns": statistics.median(iterations), "iteration_cv": run.cv(iterations), "iteration_samples_ns": iterations,
                          "samples_per_block": [block["iteration_count"] for block in native["blocks"]],
                          "block_target_medians_ns": [statistics.median(item["elapsed_ns"] for item in block["action_intervals"] if item["action_index"] == target) for block in native["blocks"]],
                          "native_numeric_passed": True, "alignment_model_ready": all(block["quality"]["model_ready"] for block in alignment),
                          "device_state": state, "native_sha256": run.sha(native_path)}
        row["status"] = "measured"
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["summary"]["native_numeric_passed"] = False
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if sampler is not None and not sampler.metadata()["stopped"]:
            sampler.stop()
            sampler.write_jsonl(directory / "telemetry.jsonl")
            run.save(directory / "telemetry-metadata.json", sampler.metadata())
        for handle in handles:
            handle.close()
    row.update(completed_utc=utc(), completed_tick=clock.now(), native_clock=clock.metadata())
    row["artifacts_sha256"] = {path.name: run.sha(path) for path in directory.iterdir() if path.is_file()}
    run.save(directory / "execution.json", row)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups, pairs = [], []
    for mode in ("isolated", "mixed"):
        for threads in (1, 2):
            declared = [row for row in rows if row["job"]["mode"] == mode and row["job"]["threads"] == threads]
            valid = [row for row in declared if row["status"] == "measured"]
            values = [row["summary"]["target_median_ns"] for row in valid]
            groups.append({"mode": mode, "threads": threads, "declared_process_count": len(declared), "successful_process_count": len(valid),
                           "all_three_successful": len(declared) == len(valid) == 3,
                           "process_target_medians_ns": values,
                           "diagnostic_median_of_process_medians_ns": statistics.median(values) if len(valid) == len(declared) == 3 else None,
                           "between_process_median_cv": run.cv(values) if len(valid) == len(declared) == 3 else None,
                           "within_process_target_cv": [row["summary"]["target_cv"] for row in valid],
                           "failed_job_ids": [row["job"]["id"] for row in declared if row["status"] != "measured"]})
        for repeat in range(3):
            pair = [row for row in rows if row["job"]["mode"] == mode and row["job"]["repeat"] == repeat]
            valid = len(pair) == 2 and all(row["status"] == "measured" for row in pair) and {row["job"]["threads"] for row in pair} == {1, 2}
            values = {row["job"]["threads"]: row["summary"]["target_median_ns"] for row in pair} if valid else {}
            pairs.append({"mode": mode, "repeat": repeat, "both_successful": valid,
                          "threads_2_over_1_target_median_ratio": values[2] / values[1] if valid else None})
    return {"schema": SCHEMA, "groups": groups, "paired_comparisons": pairs, "limits": LIMITS,
            "cost_fitted": False, "heldout_accuracy_claimed": False, "equal_invocation_counts_claimed": False,
            "pure_polling_effect_proven": False, "performance_acceptance_implied": False,
            "all_twelve_successful": len(rows) == 12 and all(row["status"] == "measured" for row in rows)}


def collect(out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    manifest = run.read(out / "manifest.json")
    manifest_digest = run.sha(out / "manifest.json")
    if len(manifest["jobs"]) != 12 or manifest.get("schema") != SCHEMA:
        raise ValueError("exact twelve-process diagnostic manifest required")
    run.verify_sources(manifest)
    rows = []
    with run.measurement_lock():
        for job in manifest["jobs"]:
            run.verify_sources(manifest)
            row = _measure(out, manifest, job)
            rows.append(row)
            print(json.dumps({"id": job["id"], "status": row["status"], "target_median_ns": row["summary"].get("target_median_ns"), "error": row.get("error")}), flush=True)
        run.verify_sources(manifest)
    if run.sha(out / "manifest.json") != manifest_digest:
        raise ValueError("manifest changed during collection")
    run.save(out / "rows.json", rows)
    report = summarize(rows)
    report.update(manifest_sha256=manifest_digest, source_inventory_verified_unchanged=True,
                  rows_sha256=run.sha(out / "rows.json"), tool_sha256=run.sha(Path(__file__)))
    run.save(out / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze inputs without launching hardware work")
    args = parser.parse_args()
    manifest = prepare(args.campaign, args.out)
    print(json.dumps({"prepared": True, "jobs": len(manifest["jobs"]), "hardware_executed": False}), flush=True)
    if args.prepare_only:
        return 0
    return 0 if collect(args.out)["all_twelve_successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
