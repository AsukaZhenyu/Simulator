"""Two fresh-process ABA/BAB diagnostics; no response law is fitted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from profiling.dynamic_validation import idle_campaign as base
from profiling.dynamic_validation.pcie_telemetry import PCIeNVMLProvider
from profiling.state_calibration.telemetry import CPUBusyProvider, DeviceTelemetrySampler, NativeClock

read, write, sha, receipt, require = base.read, base.write, base.sha, base.receipt, base.require
ROOT = base.ROOT


def direct_associations(cupti, copies):
    """Exact external-correlation IDs only. Missing/ambiguous records stay unknown."""
    external = {}
    ambiguous = set()
    for row in cupti["records"]:
        if row["kind"] != "external_correlation":
            continue
        key, value = row["correlation_id"], row["external_id"]
        if key in external and external[key] != value:
            ambiguous.add(key)
        external[key] = value
    grouped = {}
    for row in cupti["records"]:
        if row["kind"] == "memcpy" and row["correlation_id"] in external and row["correlation_id"] not in ambiguous:
            grouped.setdefault(external[row["correlation_id"]], []).append(row)
    result = []
    for copy in copies:
        rows = grouped.get(copy["external_id"], [])
        reasons = []
        if len(rows) != 1:
            reasons.append(f"Expected one exact directly correlated memcpy; found {len(rows)}")
        elif not rows[0]["timestamp_valid"] or rows[0]["bytes"] != 2097152 or rows[0]["copy_kind"] != 1:
            reasons.append("Memcpy timestamp, actual 2MiB size or H2D kind differs")
        result.append(dict(external_id=copy["external_id"], supported=not reasons,
                           reasons=reasons, device_records=rows))
    return result


def prepare(out, binary, *, cupti_dll, telemetry_ms=50.):
    out, binary, cupti_dll = Path(out).resolve(), Path(binary).resolve(), Path(cupti_dll).resolve()
    require(not out.exists(), "new v4 diagnostic directory required")
    require(binary.is_file() and cupti_dll.is_file() and telemetry_ms > 0, "explicit installed binary/CUPTI/positive telemetry cadence required")
    parent_path = ROOT / "build/dynamic-diagnosis-20261027/h2d-v3-round1/manifest.json"
    parent = read(parent_path)
    require(any(Path(x["path"]).resolve() == cupti_dll and x["sha256"] == sha(cupti_dll) for x in parent["runtime"]),
            "explicit CUPTI DLL must match the retained runtime binding")
    binding = parent["program"]
    for name in ("program", "reference"):
        require(sha(binding[name]["path"]) == binding[name]["sha256"], "retained mathematical input changed")
    paths = [Path(__file__), ROOT / "profiling/dynamic_validation/pcie_telemetry.py",
             ROOT / "profiling/dynamic_validation/native/v4/h2d_probe.cpp",
             ROOT / "profiling/dynamic_validation/native/v4/memcpy_capture.hpp",
             ROOT / "profiling/dynamic_validation/native/v4/CMakeLists.txt",
             ROOT / "profiling/dynamic_validation/native/v3/common.hpp",
             ROOT / "profiling/dynamic_validation/native/v3/thread_inventory.hpp",
             ROOT / "profiling/dynamic_validation/native/owned_timer.hpp",
             ROOT / "profiling/native/physical_stages/cupti_capture.hpp",
             ROOT / "profiling/native/steady_costs/probe.cpp",
             ROOT / "profiling/native/steady_costs/components.hpp",
             ROOT / "profiling/state_calibration/telemetry.py",
             ROOT / "profiling/state_calibration/run.py",
             ROOT / "profiling/dynamic_validation/idle_campaign.py",
             ROOT / "profiling/dac_validation/e2a.py"]
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
    for index, modes in enumerate((("bare", "memcpy_only", "bare"), ("memcpy_only", "bare", "memcpy_only"))):
        sampling = base.e2a.request("action", trials=16, target=binding["target_action_index"])
        sampling["repeats"] = 16
        request = dict(schema="dynamic-h2d-memcpy-request-4", source_sha256=sha(paths[2]), sampling=sampling,
                       expected_process_mask=4294967295, memory_mode="registered", idle_ms=[0, 1, 5, 20],
                       orders=base.orders(0), focus_idle_ms=[0, 20], block_modes=list(modes))
        jobs.append(dict(id=f"diagnostic-registered-pinned-{'ABA' if index == 0 else 'BAB'}", request=request))
    manifest = dict(schema="dynamic-h2d-memcpy-campaign-4", prepared_at_utc=base.e2a.utc(),
                    parent_manifest=receipt(parent_path), binary=receipt(archive), sources=sources,
                    runtime=parent["runtime"], runtime_dirs=parent["runtime_dirs"], hardware=parent["hardware"],
                    program=binding, jobs=jobs,
                    protocol=dict(declared_processes=2, blocks_per_process=3, timeout_seconds=180, retries=0,
                                  trials_per_block=16, conditions_per_trial=4, copies_per_burst=16,
                                  telemetry_interval_ms=telemetry_ms, observer_discrepancy_limit=.05,
                                  focus_idle_ms=[0, 20], memory_mode="registered", init_policy="pinned_init",
                                  scope="diagnostic only; no calibration prices or hardware accuracy claim",
                                  association="exact direct CUPTI external correlation; missing or ambiguous is unsupported",
                                  crossover="ABA and BAB; sequence/time and persistent CUPTI lifecycle are retained, never subtracted as pure observer overhead",
                                  future_target_state_used_for_prediction=False, performance_parameters_fitted=False))
    write(out / "manifest.json", manifest)
    return dict(out=str(out), declared=2, hardware_executed=False)


def verify(native, job, binding, pid):
    require(native["schema"] == "dynamic-h2d-memcpy-native-4" and native["request"] == job["request"], "native request/schema differs")
    require(native["process_id"] == pid and native["source_sha256"] == job["request"]["source_sha256"], "native PID/source differs")
    require(native["program_sha256"] == binding["program_sha256"] and native["target_bytes"] == 2097152, "retained H2D semantics differ")
    require(native["main_thread_affinity"]["observed_mask"] == native["main_thread_affinity"]["final_mask"] == 4, "target affinity differs")
    require(native["arena_registration"]["during"]["registered_host"] and native["arena_registration"]["restored"], "actual registered memory/restore differs")
    for payload in (native["payload_before"], native["payload_after"]):
        require(payload["full_byte_equality"] and payload["bytes"] == 2097152, "copy-byte validation failed")
    require(native["native_repeat_consistency"]["passed"] and native["verification_native"]["copy_values_equal"], "native numerical check failed")
    import numpy as np
    oracle, ports = read(binding["reference"]["path"]), native["verification_native"]["ports"]
    require(set(oracle) == set(ports), "numerical port inventory differs")
    for name, expected in oracle.items():
        actual, expected = np.asarray(ports[name]), np.asarray(expected)
        require(actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=5e-5, atol=5e-5), "independent port mismatch " + name)
    hz, count, supports = native["clock"]["frequency_hz"], 0, []
    require(len(native["blocks"]) == 3, "all three crossover blocks required")
    for index, block in enumerate(native["blocks"]):
        mode = job["request"]["block_modes"][index]
        require(block["block"] == index and block["instrumentation"] == mode and len(block["sequences"]) == 64, "crossover block/conditioning count differs")
        focused = []
        last = 0
        for sequence, row in enumerate(block["sequences"]):
            require(row["trial"] == sequence // 4 and row["condition_slot"] == sequence % 4 and len(row["copies"]) == 16, "trial/condition count differs")
            require(last <= row["reset_begin_tick"] <= row["reset_end_tick"] <= row["prime_begin_tick"] <= row["prime_end_tick"] <= row["idle_begin_tick"] <= row["idle_end_tick"] <= row["burst_begin_tick"], "condition history clock differs")
            require(row["prior_state"]["query_end_tick"] <= row["reset_begin_tick"] and row["before_burst_state"]["query_end_tick"] <= row["burst_begin_tick"], "future state appears as a prior observation")
            previous = row["burst_begin_tick"]
            for ordinal, copy in enumerate(row["copies"]):
                require(copy["ordinal"] == ordinal and previous <= copy["begin_tick"] <= copy["end_tick"] <= row["burst_end_tick"], "copy interval order differs")
                require(abs((copy["end_tick"] - copy["begin_tick"]) * 1e9 / hz - copy["wall_ns"]) <= 1, "copy wall ledger differs")
                should_tag = mode == "memcpy_only" and row["requested_idle_ms"] in (0, 20)
                require((copy["external_id"] is not None) == should_tag, "observer scope differs")
                if should_tag:
                    focused.append(copy)
                count += 1
                previous = copy["end_tick"]
            last = row["burst_end_tick"]
        if mode == "memcpy_only":
            require(block["cupti"]["all_requested_records_retained"] and block["cupti"]["runtime_driver_kernel_activity_enabled"] is False, "selected-activity integrity failed")
            associations = direct_associations(block["cupti"], focused)
            supports.append(dict(block=index, required=len(associations), supported=sum(x["supported"] for x in associations), associations=associations))
        else:
            require(block["cupti"]["enabled"] is False, "bare block activity unexpectedly enabled")
    return dict(passed=True, copies=count, numerical_ports=True, physical_support=supports,
                all_device_copies_identified=all(x["supported"] == x["required"] for x in supports), performance_parameters_fitted=False)


def collect(out, *, max_jobs=None):
    out = Path(out).resolve()
    manifest = read(out / "manifest.json")
    base.check(manifest)
    require(max_jobs is None or type(max_jobs) is int and max_jobs > 0, "positive process limit required")
    if not (out / "diagnostic-started.json").exists():
        write(out / "diagnostic-started.json", dict(started_at_utc=base.e2a.utc(), model_fitting=False))
    acquired = 0
    with base.run.measurement_lock():
        for job in manifest["jobs"]:
            folder = out / "jobs" / job["id"]
            if (folder / "execution.json").exists():
                continue
            if max_jobs is not None and acquired >= max_jobs:
                break
            folder.mkdir(parents=True, exist_ok=False)
            request, output = folder / "request.json", folder / "result.json"
            write(request, job["request"])
            command = [manifest["binary"]["path"], manifest["program"]["program"]["path"], str(request), str(output)]
            execution = dict(job=job, status="failed", started_at_utc=base.e2a.utc(), command=command)
            process, handles, sampler = None, [], None
            try:
                clock = NativeClock()
                sampler = DeviceTelemetrySampler(interval_ms=manifest["protocol"]["telemetry_interval_ms"], clock=clock,
                                                 providers={"gpu": PCIeNVMLProvider(), "cpu": CPUBusyProvider(clock)})
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
    parser.add_argument("--telemetry-ms", type=float, default=50)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    result = (prepare(args.out, args.binary, cupti_dll=args.cupti_dll, telemetry_ms=args.telemetry_ms)
              if args.command == "prepare" else collect(args.out, max_jobs=args.max_jobs))
    print(json.dumps(result, ensure_ascii=False), flush=True)
