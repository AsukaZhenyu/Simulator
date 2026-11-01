"""Target-only runtime callback neutrality and CPU-core placement diagnosis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil
import subprocess

from profiling.dynamic_validation import idle_campaign as base
from profiling.dynamic_validation.pcie_telemetry import PCIeNVMLProvider
from profiling.state_calibration.telemetry import CPUBusyProvider, DeviceTelemetrySampler, NativeClock

ROOT = base.ROOT
read, write, sha, receipt, require = base.read, base.write, base.sha, base.receipt, base.require
API_COUNTS = {"cudaLaunchKernel_v7000": 205, "cudaMemcpyAsync_v3020": 129, "cudaStreamSynchronize_v3020": 2}


def physical_core_contract(topology):
    require(topology.get("known"), "verified physical CPU topology required; no logical-index inference")
    selected = {}
    for logical in (2, 4, 16):
        matching = [core for core in topology["all_core_records"]
                    if any(group["group"] == 0 and logical in group["logical_processors"] for group in core["group_masks"])]
        require(len(matching) == 1, "selected logical CPU has missing/ambiguous physical core")
        selected[str(logical)] = dict(logical_cpu=logical, mask=1 << logical,
                                      physical_core_record=matching[0]["core_record"], efficiency_class=matching[0]["efficiency_class"])
    require(selected["2"]["physical_core_record"] != selected["4"]["physical_core_record"], "CPU2 and CPU4 share a physical core; cross-core condition refused")
    return dict(provider=topology["provider"], selected=selected, CPU2_CPU4_physical_cores_disjoint=True,
                CPU2_CPU4_same_efficiency_class=selected["2"]["efficiency_class"] == selected["4"]["efficiency_class"],
                physical_core_count=topology["physical_core_record_count"], raw=topology)


def union_ticks(intervals):
    total, lower, upper = 0, None, None
    for begin, end in sorted(intervals):
        require(begin <= end, "reversed API callback interval")
        if lower is None:
            lower, upper = begin, end
        elif begin <= upper:
            upper = max(upper, end)
        else:
            total += upper - lower
            lower, upper = begin, end
    return total + (upper - lower if lower is not None else 0)


def callback_ledger(callbacks, phases, hz, *, expected_counts=API_COUNTS):
    """Host response partition only; never GPU work or CPU active time."""
    require(callbacks["all_selected_calls_retained"], "callback overflow/pairing/clock/return integrity failed")
    require(callbacks["activity_driver_kernel_collection_enabled"] is False, "activity observer unexpectedly enabled")
    require(callbacks["record_count"] == len(callbacks["callback_records"]), "callback record inventory differs")
    metadata, grouped = callbacks["metadata"], defaultdict(list)
    for index, row in enumerate(callbacks["callback_records"]):
        require(row["ordinal"] == index and row["phase_id"] in phases, "callback ordinal/phase differs")
        require(type(row["metadata_index"]) is int and 0 <= row["metadata_index"] < len(metadata), "callback metadata index invalid")
        phase = phases[row["phase_id"]]
        require(phase["begin_tick"] <= row["enter_tick"] <= row["exit_tick"] <= phase["end_tick"], "API callback escaped target action")
        require(row["exit_received"] and row["return_status"] == 0, "API callback lacks successful paired return")
        item = metadata[row["metadata_index"]]
        require(item["name"] in expected_counts, "unexpected callback API family")
        grouped[row["phase_id"]].append((row, item))
    if not callbacks["enabled"]:
        require(not grouped and not metadata, "unsubscribed control unexpectedly recorded callbacks")
        return dict(phases=0, records=0, phase_partitions=[], inclusive_response_only=True)
    require(set(grouped) == set(phases), "not every target phase has callback evidence")
    partitions = []
    for phase_id, phase in phases.items():
        records = grouped[phase_id]
        require(dict(Counter(meta["name"] for _, meta in records)) == expected_counts, "selected API counts differ from predeclared retained graph")
        measured = union_ticks((row["enter_tick"], row["exit_tick"]) for row, _ in records)
        wall = phase["end_tick"] - phase["begin_tick"]
        require(0 <= measured <= wall, "API union exceeds action wall")
        partitions.append(dict(phase_id=phase_id, selected_API_union_ns=measured * 1e9 / hz,
                               outside_selected_API_union_ns=(wall - measured) * 1e9 / hz,
                               action_wall_ns=wall * 1e9 / hz))
    return dict(phases=len(partitions), records=len(callbacks["callback_records"]), phase_partitions=partitions,
                inclusive_response_only=True, GPU_service_or_enqueue_acceptance_identified=False)


def prepare(out, binary, *, cupti_dll, cpu_actor_mask="both"):
    out, binary, cupti_dll = Path(out).resolve(), Path(binary).resolve(), Path(cupti_dll).resolve()
    require(not out.exists() and binary.is_file() and cupti_dll.is_file(), "new directory and explicit installed binary/CUPTI required")
    require(cpu_actor_mask in ("4", "16", "both"), "explicit CPU2/CPU4 condition required")
    from profiling.state_response.diagnose_cpu import topology
    topology_receipt = physical_core_contract(topology())
    parent_path = ROOT / "build/dynamic-validation-20261026/external-v1/manifest.json"
    parent, runtime_parent = read(parent_path), read(ROOT / "build/dynamic-diagnosis-20261027/h2d-v3-round1/manifest.json")
    require(any(Path(x["path"]).resolve() == cupti_dll and x["sha256"] == sha(cupti_dll) for x in runtime_parent["runtime"]), "CUPTI binding differs")
    binding = parent["programs"]["gpu_dense-profile"]
    for name in ("program", "reference"):
        require(sha(binding[name]["path"]) == binding[name]["sha256"], "retained mathematical input changed")
    program = read(binding["program"]["path"])
    require(program["actions"][22]["kind"] == "COMPUTE" and program["actions"][22]["device"] == "gpu" and program["actions"][22]["target_id"].endswith(":attn"), "retained GPU Attention action differs")
    old = next(x for x in parent["jobs"] if x["id"] == "profile-r0-gpu_dense-cpu_gpu_staggered")
    paths = [Path(__file__), ROOT / "profiling/dynamic_validation/native/v6/external_probe.cpp",
             ROOT / "profiling/dynamic_validation/native/v6/runtime_callbacks.hpp", ROOT / "profiling/dynamic_validation/native/v6/CMakeLists.txt",
             ROOT / "profiling/dynamic_validation/native/v3/common.hpp", ROOT / "profiling/dynamic_validation/native/v3/thread_inventory.hpp",
             ROOT / "profiling/dynamic_validation/native/owned_timer.hpp", ROOT / "profiling/native/physical_stages/cupti_capture.hpp",
             ROOT / "profiling/native/steady_costs/probe.cpp", ROOT / "profiling/native/steady_costs/components.hpp",
             ROOT / "profiling/dynamic_validation/idle_campaign.py", ROOT / "profiling/dynamic_validation/pcie_telemetry.py",
             ROOT / "profiling/state_calibration/telemetry.py", ROOT / "profiling/state_calibration/run.py",
             ROOT / "profiling/state_response/diagnose_cpu.py", ROOT / "profiling/dac_validation/e2a.py"]
    paths += list((ROOT / "profiling/native/steady_costs/build-release-v1").glob("*.inc"))
    sdk = cupti_dll.parent.parent / "include"
    paths += [sdk / name for name in ("cupti_callbacks.h", "cupti_runtime_cbid.h", "generated_cuda_runtime_api_meta.h")]
    out.mkdir(parents=True)
    sources = []
    for path in paths:
        relative = path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else Path("SDK") / path.name
        archive = out / "executed_sources" / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
        sources.append(dict(source=receipt(path), archived=receipt(archive)))
    archive = out / "executed_binary" / binary.name
    archive.parent.mkdir()
    shutil.copy2(binary, archive)
    jobs = []
    for mask, instruments in ((4, ("stamped", "runtime_callbacks")), (16, ("runtime_callbacks", "stamped"))):
        if cpu_actor_mask != "both" and int(cpu_actor_mask) != mask:
            continue
        for instrument in instruments:
            request = copy.deepcopy(old["request"])
            request.update(schema="dynamic-external-callback-request-6", source_sha256=sha(paths[1]), instrumentation=instrument,
                           callback_source_sha256=sha(paths[2]), init_policy="pinned_init", attention_action_index=22, callback_scope="attention_only_all2048", max_API_records=800000)
            for device in ("cpu", "gpu"):
                request[device + "_actor"]["init_policy"] = "pinned_init"
            request["cpu_actor"]["main_thread_mask"] = mask
            jobs.append(dict(id=f"diagnostic-cpu-mask{mask}-{instrument}", cpu_actor_mask=mask,
                             role="control" if instrument == "stamped" else "phase_profile", request=request))
    manifest = dict(schema="dynamic-external-callback-campaign-6", prepared_at_utc=base.e2a.utc(), parent_manifest=receipt(parent_path),
                    retained_actor_job=old, binary=receipt(archive), sources=sources, runtime=runtime_parent["runtime"], runtime_dirs=runtime_parent["runtime_dirs"],
                    hardware=parent["hardware"], topology=topology_receipt, program=binding, jobs=jobs,
                    protocol=dict(declared_processes=len(jobs), timeout_seconds=300, retries=0, continuous_replays=2048,
                                  init_policy="pinned_init", CPU_actor_masks=[j["cpu_actor_mask"] for j in jobs], GPU_actor_mask=65536,
                                  callback_scope="target Attention all2048; 3 selected runtime API families; other actor threads immediately filtered",
                                  expected_API_counts_per_attention=API_COUNTS, POD_capacity_same_for_control_and_observer=800000,
                                  record_materialization="strictly after target end and actor join; no mid-target flush", telemetry_interval_ms=50.,
                                  thread_state_helper="not enabled in initial neutrality campaign; no Ready/Wait inference",
                                  observer_quality=dict(max_total_wall_relative_difference=.05, max_complete_off_state_attention_mean_difference=.05,
                                                        raw_CV_not_dropped=True, transitions_retained=True),
                                  scope="diagnostic/phase-profile support only; no parameters automatically fitted or transferred"))
    write(out / "manifest.json", manifest)
    return dict(out=str(out), declared=len(jobs), hardware_executed=False)


def verify(native, job, binding, pid):
    request = job["request"]
    require(native["schema"] == "dynamic-external-callback-native-6" and native["request"] == request, "new callback request/schema differs")
    require(native["process_id"] == pid and native["source_sha256"] == request["source_sha256"], "native PID/source differs")
    require(native["callback_source_sha256"] == request["callback_source_sha256"], "compiled callback implementation differs")
    program = read(binding["program"]["path"])
    require(native["program_sha256"] == program["program_sha256"] and native["actions"] == program["actions"], "target program/action identity differs")
    require(native["init_policy"] == "pinned_init" and native["initialization_affinity"]["observed_mask"] == 4, "target initialization differs")
    for affinity in (native["target_thread_affinity"],):
        require(affinity["requested_mask"] == affinity["observed_mask"] == affinity["final_mask"] == 4 and affinity["restored"], "target execution affinity differs")
    process = native["process_affinity"]
    require(process["requested_mask"] == process["observed_mask"] == process["final_mask"] == 4294967295, "owned process mask differs")
    hz, epoch, begin, end = native["clock"]["frequency_hz"], native["target_expected_origin_tick"], native["target_begin_tick"], native["target_end_tick"]
    require(hz > 0 and epoch <= begin < end and abs((end - begin) * 1e9 / hz - native["target_wall_ns"]) <= 1, "target clock ledger differs")
    require(len(native["blocks"]) == 128 and native["continuous_replays"] == 2048 and not native["reset_prime_between_timed_blocks"], "continuous target contract differs")
    require(native["instrumentation"] == request["instrumentation"] and not native["actor_timeline_triggered_by_target_latency"] and not native["future_target_state_used"], "callback timing/causality differs")
    phases, previous = {}, begin
    for index, block in enumerate(native["blocks"]):
        require(block["block"] == index and block["replays"] == 16 and previous <= block["begin_tick"] < block["end_tick"] <= end, "block order differs")
        require(len(block["actions"]) == len(program["actions"]) * 16 and abs((block["end_tick"] - block["begin_tick"]) * 1e9 / hz - block["wall_ns"]) <= 1, "block action count/clock differs")
        last = block["begin_tick"]
        for number, action in enumerate(block["actions"]):
            repeat, action_index = divmod(number, len(program["actions"]))
            require(action["global_replay"] == index * 16 + repeat and action["action_index"] == action_index and last <= action["begin_tick"] <= action["end_tick"] <= block["end_tick"], "action identity/clock differs")
            require(abs((action["end_tick"] - action["begin_tick"]) * 1e9 / hz - action["wall_ns"]) <= 1, "action QPC ledger differs")
            if action_index == 22:
                require(action["callback_phase_id"] == action["global_replay"] + 1, "callback target phase identity differs")
                phases[action["callback_phase_id"]] = action
            else:
                require(action["callback_phase_id"] is None, "non-Attention action unexpectedly armed")
            last = action["end_tick"]
        previous = block["end_tick"]
    require(native["activity_capture_instantiated"] is False and native["observer_flush_gap_in_target_wall_ns"] == 0, "activity capture or target-time serialization present")
    for device, mask in (("cpu", job["cpu_actor_mask"]), ("gpu", 65536)):
        actor, specification = native["actors"][device], request[device + "_actor"]
        require(actor["thread_affinity"]["observed_mask"] == actor["thread_affinity"]["final_mask"] == actor["initialization_affinity"]["observed_mask"] == mask, "actor active/init affinity differs")
        require(actor["thread_affinity"]["restored"] and actor["repeated_output_equal"] and not actor["cancelled"], "actor completion/output differs")
        require(len(actor["intervals"]) == len(specification["intervals_ms"]), "actor interval inventory differs")
        for observed, planned in zip(actor["intervals"], specification["intervals_ms"]):
            require([observed["planned_begin_ms"], observed["planned_end_ms"]] == planned and observed["calls"] > 0, "actor plan/pulse overlap differs")
        require(actor["stopped_tick"] <= native["callback_finalize_begin_tick"], "callback records materialized before actor join")
    require(end <= native["callback_finalize_begin_tick"] <= native["callback_finalize_end_tick"], "callback materialization escaped posttarget boundary")
    callbacks = native["runtime_callbacks"]
    require(callbacks["enabled"] == (request["instrumentation"] == "runtime_callbacks") and callbacks["target_thread_id"] == native["main_thread_id"], "callback target/mode differs")
    require(callbacks["capacity"] == 800000, "balanced fixed POD allocation differs")
    checked = callback_ledger(callbacks, phases, hz)
    require(native["native_repeat_consistency"]["passed"] and native["verification_native"]["copy_values_equal"], "native numerical consistency failed")
    import numpy as np
    ports, oracle = native["verification_native"]["ports"], read(binding["reference"]["path"])
    require(set(ports) == set(oracle), "numerical port inventory differs")
    for name, expected in oracle.items():
        actual, expected = np.asarray(ports[name]), np.asarray(expected)
        require(actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=5e-5, atol=5e-5), "independent port mismatch " + name)
    return dict(passed=True, continuous_replays=2048, callback_phases=checked["phases"], callback_records=checked["records"],
                API_union_plus_complement_closes=True, numerical_oracle=True, performance_parameters_fitted=False)


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
                require(sha(reference["path"]) == reference["sha256"], "retained mathematical input changed")
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
                require(code == 0, "owned callback diagnostic failed; no retry/substitution")
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
    parser.add_argument("--cpu-actor-mask", choices=("4", "16", "both"), default="both")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    result = prepare(args.out, args.binary, cupti_dll=args.cupti_dll, cpu_actor_mask=args.cpu_actor_mask) if args.command == "prepare" else collect(args.out, max_jobs=args.max_jobs)
    print(json.dumps(result, ensure_ascii=False), flush=True)
