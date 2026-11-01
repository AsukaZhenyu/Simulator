"""Bounded first-pulse Attention diagnosis, preserving the owned actor plan."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from profiling.dynamic_validation import external as original
from profiling.dynamic_validation import idle_campaign as base
from profiling.dynamic_validation.pcie_telemetry import PCIeNVMLProvider
from profiling.state_calibration.telemetry import CPUBusyProvider, DeviceTelemetrySampler, NativeClock

ROOT = base.ROOT
read, write, sha, receipt, require = base.read, base.write, base.sha, base.receipt, base.require


def validate_attention_stages(stage, hz):
    ticks = [stage[k] for k in ("begin_tick", "graph_compute_enter_tick", "graph_compute_return_tick", "extra_sync_enter_tick", "extra_sync_return_tick", "end_tick")]
    require(all(a <= b for a, b in zip(ticks, ticks[1:])), "Attention host stage order differs")
    require(abs((ticks[-1] - ticks[0]) * 1e9 / hz - stage["host_wall_ns"]) <= 1, "Attention host wall differs")
    require(abs(stage["graph_compute_inclusive_ns"] + stage["extra_sync_ns"] + stage["wrapper_gaps_ns"] - stage["host_wall_ns"]) <= 3, "Attention wall partition does not close")
    require(stage["original_graph_compute_internal_sync_retained"] and not stage["device_span_added_to_host_wait"], "original synchronous operation or overlap contract differs")


def prepare(out, binary, *, cupti_dll):
    out, binary, cupti_dll = Path(out).resolve(), Path(binary).resolve(), Path(cupti_dll).resolve()
    require(not out.exists() and binary.is_file() and cupti_dll.is_file(), "new directory and explicit installed binary/CUPTI required")
    parent_path = ROOT / "build/dynamic-validation-20261026/external-v1/manifest.json"
    parent, runtime_parent = read(parent_path), read(ROOT / "build/dynamic-diagnosis-20261027/h2d-v3-round1/manifest.json")
    require(any(Path(x["path"]).resolve() == cupti_dll and x["sha256"] == sha(cupti_dll) for x in runtime_parent["runtime"]), "CUPTI binding differs")
    binding = parent["programs"]["gpu_dense-profile"]
    for name in ("program", "reference"):
        require(sha(binding[name]["path"]) == binding[name]["sha256"], "retained mathematical input changed")
    program = read(binding["program"]["path"])
    require(program["actions"][22]["kind"] == "COMPUTE" and program["actions"][22]["device"] == "gpu" and program["actions"][22]["target_id"].endswith(":attn"), "retained GPU Attention action differs")
    old = next(x for x in parent["jobs"] if x["id"] == "profile-r0-gpu_dense-cpu_gpu_staggered")
    paths = [Path(__file__), ROOT / "profiling/dynamic_validation/native/v5/external_probe.cpp",
             ROOT / "profiling/dynamic_validation/native/v5/CMakeLists.txt", ROOT / "profiling/dynamic_validation/native/v3/common.hpp",
             ROOT / "profiling/dynamic_validation/native/v3/thread_inventory.hpp", ROOT / "profiling/dynamic_validation/native/owned_timer.hpp",
             ROOT / "profiling/native/physical_stages/cupti_capture.hpp", ROOT / "profiling/native/steady_costs/probe.cpp",
             ROOT / "profiling/native/steady_costs/components.hpp", ROOT / "profiling/dynamic_validation/external.py",
             ROOT / "profiling/dynamic_validation/idle_campaign.py", ROOT / "profiling/dynamic_validation/pcie_telemetry.py",
             ROOT / "profiling/state_calibration/telemetry.py", ROOT / "profiling/state_calibration/run.py", ROOT / "profiling/dac_validation/e2a.py"]
    paths += list((ROOT / "profiling/native/steady_costs/build-release-v1").glob("*.inc"))
    out.mkdir(parents=True)
    sources = []
    for path in paths:
        archive = out / "executed_sources" / path.resolve().relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
        sources.append(dict(source=receipt(path), archived=receipt(archive)))
    archive = out / "executed_binary" / binary.name
    archive.parent.mkdir()
    shutil.copy2(binary, archive)
    jobs = []
    for policy, instruments in (("pinned_init", ("stamped", "bounded_stages")), ("wide_init", ("bounded_stages", "stamped"))):
        for instrument in instruments:
            request = copy.deepcopy(old["request"])
            request.update(schema="dynamic-external-stage-request-5", source_sha256=sha(paths[1]), instrumentation=instrument,
                           init_policy=policy, attention_action_index=22, capture_stop_after_ms=5000)
            for device in ("cpu", "gpu"):
                request[device + "_actor"]["init_policy"] = policy
            jobs.append(dict(id=f"diagnostic-{policy}-{instrument}", request=request))
    manifest = dict(schema="dynamic-external-stage-campaign-5", prepared_at_utc=base.e2a.utc(), parent_manifest=receipt(parent_path),
                    retained_actor_job=old, binary=receipt(archive), sources=sources, runtime=runtime_parent["runtime"],
                    runtime_dirs=runtime_parent["runtime_dirs"], hardware=parent["hardware"], program=binding, jobs=jobs,
                    protocol=dict(declared_processes=4, timeout_seconds=300, retries=0, continuous_replays=2048,
                                  report_block_replays=16, initial_reset_full_prime_only=True, target_main_mask=4,
                                  actor_masks=dict(cpu=4, gpu=65536), init_policies=["pinned_init", "wide_init"],
                                  capture_stop_after_ms=5000, capture_stop_rule="After the first completed action at/after target epoch+5s; not a kernel interruption or a target-latency-triggered actor plan",
                                  CUPTI_flush_gap_retained_separately=True, telemetry_interval_ms=50.,
                                  scope="diagnostic, no performance parameter fitting or accuracy validation; full observer may perturb target/actors"))
    write(out / "manifest.json", manifest)
    return dict(out=str(out), declared=4, hardware_executed=False)


def verify(native, job, binding, pid):
    request = job["request"]
    require(native["schema"] == "dynamic-external-stage-native-5" and native["request"] == request, "new diagnostic schema/request differs")
    require(native["init_policy"] == request["init_policy"], "target init policy differs")
    init_mask = 4 if request["init_policy"] == "pinned_init" else 4294967295
    require(native["initialization_affinity"]["observed_mask"] == init_mask and native["initialization_affinity"]["restored"], "target initialization extent differs")
    for device, mask in (("cpu", 4), ("gpu", 65536)):
        require(native["actors"][device]["initialization_affinity"]["observed_mask"] == (mask if request["init_policy"] == "pinned_init" else 4294967295), "actor initialization extent differs")
    # Reuse the original exact actor, action-ledger and numerical oracle checker.
    # This in-memory field adapter only supplies its old checker entry point;
    # actual raw data/schema remain new, and timing equivalence is never claimed.
    checked_native, checked_request = dict(native), dict(request)
    checked_native.update(schema="dynamic-external-native-1", main_thread_affinity=native["target_thread_affinity"], instrumentation="stamped")
    checked_request["instrumentation"] = "stamped"
    checked_native["request"] = checked_request
    checked = original.verify(checked_native, checked_request, binding, pid)
    observed = []
    for block in native["blocks"]:
        for action in block["actions"]:
            if action["attention_host_stages"] is not None:
                validate_attention_stages(action["attention_host_stages"], native["clock"]["frequency_hz"])
                require(action["action_index"] == 22 and action["external_id"] is not None, "unexpected observed action")
                observed.append(action["external_id"])
    if request["instrumentation"] == "bounded_stages":
        require(observed and native["cupti"]["all_requested_records_retained"], "bounded observer integrity failed")
        require(native["observer_flush_begin_tick"] <= native["observer_flush_end_tick"], "observer flush gap differs")
    else:
        require(not observed and native["cupti"] is None, "control unexpectedly observed")
    return dict(shared_exact_ledger_and_oracle=checked, observed_attention_actions=len(observed),
                raw_timing_equivalence_to_original_claimed=False, performance_parameters_fitted=False)


def collect(out, *, max_jobs=None):
    out = Path(out).resolve()
    manifest = read(out / "manifest.json")
    base.check(manifest)
    require(max_jobs is None or type(max_jobs) is int and max_jobs > 0, "positive diagnostic process limit required")
    if not (out / "diagnostic-started.json").exists():
        write(out / "diagnostic-started.json", dict(started_at_utc=base.e2a.utc(), performance_parameters_fitted=False))
    acquired = 0
    with base.run.measurement_lock():
        for job in manifest["jobs"]:
            folder = out / "jobs" / job["id"]
            if (folder / "execution.json").exists():
                continue
            if max_jobs is not None and acquired >= max_jobs:
                break
            for reference in (manifest["program"]["program"], manifest["program"]["reference"]):
                require(sha(reference["path"]) == reference["sha256"], "retained mathematical input changed before acquisition")
            folder.mkdir(parents=True, exist_ok=False)
            request, output = folder / "request.json", folder / "result.json"
            write(request, job["request"])
            command = [manifest["binary"]["path"], manifest["program"]["program"]["path"], str(request), str(output)]
            execution = dict(job=job, status="failed", started_at_utc=base.e2a.utc(), command=command)
            process, handles, sampler = None, [], None
            try:
                clock = NativeClock()
                sampler = DeviceTelemetrySampler(interval_ms=50., clock=clock, providers={"gpu": PCIeNVMLProvider(), "cpu": CPUBusyProvider(clock)})
                sampler.start()
                environment = base.run.child_environment(manifest["runtime_dirs"])
                environment["OPENBLAS_NUM_THREADS"] = "1"
                process, stdout, stderr, affinity = base.run.launch(command, environment, folder / "stdout.txt", folder / "stderr.txt", 4294967295)
                handles = [stdout, stderr]
                execution.update(process_id=process.pid, process_affinity=affinity)
                code = process.wait(timeout=manifest["protocol"]["timeout_seconds"])
                execution.update(returncode=code, process_exited_verified=True)
                require(code == 0, "owned native diagnostic failed; no retry/substitution")
                execution["verification"] = verify(read(output), job, manifest["program"], process.pid)
                execution.update(status="completed", native=receipt(output))
            except BaseException as error:
                execution["error"] = repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                if process is not None:
                    execution.update(returncode=process.returncode, process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:
                    handle.close()
                if sampler is not None:
                    try:
                        sampler.stop()
                        execution["telemetry"] = sampler.write_jsonl(folder / "telemetry.jsonl")
                        execution["telemetry_metadata"] = sampler.metadata()
                    except BaseException as error:
                        execution["telemetry_error"] = repr(error)
                execution["completed_at_utc"] = base.e2a.utc()
                write(folder / "execution.json", execution)
            acquired += 1
            print(json.dumps(dict(job=job["id"], status=execution["status"], error=execution.get("error"))), flush=True)
    entries = [read(out / "jobs" / job["id"] / "execution.json") for job in manifest["jobs"] if (out / "jobs" / job["id"] / "execution.json").exists()]
    target = out / ("executions.json" if len(entries) == len(manifest["jobs"]) else f"partial-executions-{len(entries)}.json")
    if not target.exists():
        write(target, entries)
    return dict(declared=len(manifest["jobs"]), acquired=len(entries), completed=sum(x["status"] == "completed" for x in entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "collect"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--binary")
    parser.add_argument("--cupti-dll")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    result = prepare(args.out, args.binary, cupti_dll=args.cupti_dll) if args.command == "prepare" else collect(args.out, max_jobs=args.max_jobs)
    print(json.dumps(result, ensure_ascii=False), flush=True)
