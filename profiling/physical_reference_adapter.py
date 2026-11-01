"""Bind collected physical-stage references to a causal diagnostic trajectory.

This offline adapter never launches hardware, refits a parameter, converts GPU
utilization to capacity, or treats the foreground duration as a supply sensor.
It checks saved identities/hashes and delegates activity attribution and causal
sample-hold semantics to their existing pure-data implementations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .available_service.calibration import digest, require
from .available_service.physical_records import interval_union, matmul_work, parse_physical_records
from .available_service.reference_supply import build_reference_supply
from .physical_job_index import effective_jobs

SCHEMA = "physical-reference-adaptation-1"
CPU_RESOURCE = "cpu_reference_thread_service"
GPU_RESOURCE = "gpu_reference_gemm_flops"


def _load(path, expected_sha256=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    require(expected_sha256 is None or checksum == expected_sha256, "file hash differs: " + str(path))
    value = json.loads(raw)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value, checksum


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be an integer >= " + str(minimum))
    return value


def _ticks_ns(ticks, frequency):
    _integer(ticks, "QPC ticks"); _integer(frequency, "QPC frequency", 1)
    return ticks // frequency * 1_000_000_000 + ticks % frequency * 1_000_000_000 // frequency


def _span(native, frequency):
    if native.get("schema") == "physical-stage-native-1":
        begin, end = native.get("sampling_begin_ns"), native.get("sampling_end_ns")
    else:
        require(native.get("format_version") == "state-response-native-v1", "unsupported actor native schema")
        begin = _ticks_ns(native.get("sampling_begin_tick"), frequency)
        end = _ticks_ns(native.get("sampling_end_tick"), frequency)
    _integer(begin, "sampling begin"); _integer(end, "sampling end")
    require(begin < end, "actor sampling interval must be positive")
    return dict(begin_ns=begin, end_ns=end)


def _numerical(receipt):
    flags = ("passed", "full_graph_equal", "action_trace_equal", "full_port_inventory_equal",
             "copy_routes_bytes_and_values_equal")
    require(isinstance(receipt, dict) and all(receipt.get(k) is True for k in flags),
            "passed independent numerical/port receipt required")
    _integer(receipt.get("independently_checked_ports"), "independently checked ports", 1)
    return True


def build_job_reference_supply(execution_path):
    """Return ``(supply_or_None, refsdata)`` for one completed campaign job.

    Integrity/identity errors raise ValueError. Unidentified physical quantities
    remain unknown in the returned reference-supply artifact. In particular the
    CPU cumulative-counter candidate is not qualified as pure physical service.
    AFTER references are drift audits only. Native numerical checks are bound
    receipts from collection; the independent numerical oracle is not rerun here.
    """
    execution_path = Path(execution_path).resolve()
    execution, execution_sha = _load(execution_path)
    campaign_root = execution_path.parents[2]
    manifest_path = campaign_root / "manifest.json"
    manifest, manifest_sha = _load(manifest_path)
    require(manifest.get("schema") == "physical-service-campaign-1" and
            execution.get("schema") == "physical-service-execution-1", "unsupported campaign/execution schema")
    require(execution.get("status") == "completed" and execution.get("cleanup_verified") is True,
            "only completed, cleanup-verified executions can supply references")
    require(manifest.get("source_inventory_sha256") == digest(manifest.get("sources")) ==
            execution.get("source_inventory_sha256"), "campaign source inventory binding differs")
    job = execution.get("job")
    require(isinstance(job, dict) and execution_path.parent.name == job.get("id"), "job path identity differs")
    if (campaign_root / "posttarget-job-index.json").exists():
        from .physical_posttarget_collection import effective_posttarget_jobs
        jobs = effective_posttarget_jobs(campaign_root, manifest)
    else:
        jobs = effective_jobs(campaign_root, manifest)
    matches = [j for j in jobs if j.get("id") == job["id"]]
    require(len(matches) == 1 and matches[0] == job, "execution job differs from prepared manifest")
    binding = manifest.get("programs", {}).get(job.get("program_key"))
    require(isinstance(binding, dict) and binding == execution.get("program"), "program preparation binding differs")
    program, program_bytes_sha = _load(binding["path"], binding["sha256"])
    _, oracle_sha = _load(binding["oracle"], binding["oracle_sha256"])
    require(program.get("program_sha256") == binding.get("program_sha256") ==
            digest({k: v for k, v in program.items() if k != "program_sha256"}), "program semantic digest differs")
    protocol = manifest.get("protocol")
    require(isinstance(protocol, dict), "prepared protocol required")
    max_hold = _integer(protocol.get("max_reference_hold_ns"), "predeclared reference hold", 1)
    require(type(job.get("references")) is bool, "explicit reference protocol required")
    expected = {"main"}
    if job["references"]:
        expected.update(("before-gpu", "before-cpu", "after-gpu", "after-cpu"))
    for kind in ("cpu", "gpu"):
        if _integer(job.get(kind + "_background"), kind + " background count"):
            expected.add(kind + "-worker")
    processes = execution.get("processes")
    require(isinstance(processes, list) and len({p.get("name") for p in processes}) == len(processes),
            "unique owned process inventory required")
    actors = {p.get("name"): p for p in processes}
    require(set(actors) == expected and set(execution.get("raw_files", {})) == expected,
            "owned process/raw actor inventory differs from prepared job")
    require(len({p.get("pid") for p in processes}) == len(processes), "duplicate owned process identities")
    natives, parsed, sources, spans = {}, {}, {}, {}
    common_clock = None
    for name in sorted(expected):
        process = actors[name]
        require(process.get("process_exited_verified") is True and process.get("forced_termination") is False and
                process.get("returncode") == 0, "actor did not exit normally: " + name)
        raw_file = execution["raw_files"][name]
        native_path = Path(raw_file["path"]).resolve()
        require(native_path.is_relative_to(execution_path.parent) and
                Path(process["command"][-1]).resolve() == native_path, "actor output path/command differs: " + name)
        native, raw_sha = _load(native_path, raw_file["sha256"])
        ready = process.get("ready")
        require(isinstance(ready, dict), "actor READY receipt missing: " + name)
        expected_binary = manifest["background_binary"] if name.endswith("-worker") else manifest["binary"]
        require(Path(process["command"][0]).resolve() == Path(expected_binary).resolve() and
                Path(process["command"][1]).resolve() == Path(binding["path"]).resolve(),
                "prepared actor binary/program command differs: " + name)
        pid = _integer(process.get("pid"), "owned PID", 1)
        require(native.get("process_id") == ready.get("process_id") == pid and
                native.get("main_thread_id") == ready.get("main_thread_id"), "native/READY process or thread differs: " + name)
        _integer(native.get("main_thread_id"), "native TID", 1)
        ready_clock = ready.get("clock", {})
        require(ready_clock.get("clock_id") == "windows_qpc", "actor READY QPC clock differs")
        frequency = _integer(ready_clock.get("frequency_hz"), "actor QPC frequency", 1)
        clock = native.get("clock", {})
        require(clock.get("frequency_hz") == frequency, "native/READY QPC frequency differs")
        if common_clock is None:
            common_clock = frequency
        require(frequency == common_clock, "actors do not share the same QPC frequency")
        require(native.get("program_sha256") == ready.get("program_sha256") == program["program_sha256"] and
                native.get("actions") == program["actions"],
                "native original program/action inventory differs: " + name)
        _numerical(execution.get("numerical", {}).get(name))
        span = _span(native, frequency)
        if name == "main" or name.startswith(("before-", "after-")):
            require(native.get("schema") == "physical-stage-native-1", "physical actor schema differs: " + name)
            require(native.get("clock", {}).get("clock_id") == "windows_qpc_floor_ns" and
                    native.get("clock", {}).get("clock_origin") == "absolute_qpc_zero", "physical clock differs")
            require(ready.get("physical_clock") == native["clock"], "READY/output physical clock differs")
            is_main = name == "main"
            target_kind = name.rsplit("-", 1)[-1]
            target = None if is_main else protocol["targets"]["reference_" + target_kind]
            count = job["samples"] if is_main else protocol["reference_" + target_kind + "_samples"]
            require(native.get("mode") == ("full" if is_main else "target") == ready.get("mode") and
                    native.get("target_action_index") == ready.get("target_action_index") == target and
                    native.get("instrumentation") == ready.get("instrumentation") == (job["instrumentation"] if is_main else "cupti") and
                    native.get("samples_requested") == native.get("samples_completed") == ready.get("samples_requested") == count,
                    "prepared native mode/target/count differs: " + name)
            completed = execution.get("completed", {}).get(name)
            require(isinstance(completed, dict) and completed.get("process_id") == pid and
                    completed.get("completed_samples") == count and completed.get("clock") == clock and
                    completed.get("sampling_begin_ns") == span["begin_ns"] and
                    completed.get("sampling_end_ns") == span["end_ns"], "sampling completion receipt differs: " + name)
            parsed[name] = parse_physical_records(native, program=program)
        else:
            require(clock.get("clock_id") == "windows_qpc", "background QPC clock differs")
        natives[name], spans[name] = native, span
        sources[name] = dict(path=str(native_path), sha256=raw_sha, process_id=pid,
                             main_thread_id=native["main_thread_id"], sampling_window=span)
        if name in parsed:
            attribution = parsed[name]
            sources[name].update(attribution_sha256=attribution["artifact_sha256"],
                capture_qualified=attribution["capture_qualified"], capture_issues=attribution["capture_issues"],
                orphan_record_count=len(attribution["orphan_records"]),
                invalid_time_record_count=len(attribution["invalid_time_records"]),
                conflicting_external_correlations=attribution["conflicting_external_correlations"])
    context = dict(hardware=manifest["hardware"], source_inventory_sha256=manifest["source_inventory_sha256"],
                   physical=program["physical"], timing_boundary="completed fixed reference batch; exact CUPTI kernel union or unqualified thread counter")
    horizon = max(s["end_ns"] for s in spans.values()) + 1
    warnings = []
    samples = []
    for name in ("before-gpu", "before-cpu", "after-gpu", "after-cpu") if job["references"] else ():
        native, attribution, span = natives[name], parsed[name], spans[name]
        sample = dict(sample_id=job["id"] + ":" + name,
            sample_role="before_target" if name.startswith("before-") else "after_target",
            begin_ns=span["begin_ns"], end_ns=span["end_ns"], sensor_window=span, context=context,
            clock_id="windows_qpc_floor_ns", clock_origin="absolute_qpc_zero", numerical_verified=True,
            run_id=job["id"], process_id=native["process_id"], evidence=[
                "execution-sha256:" + execution_sha, "native-sha256:" + sources[name]["sha256"],
                "program-sha256:" + program["program_sha256"], "oracle-bytes-sha256:" + oracle_sha,
                "independent numerical/port verification receipt bound in execution; not recomputed by adapter"])
        if name.endswith("gpu"):
            target = program["actions"][native["target_action_index"]]
            require(target.get("kind") == "COMPUTE" and target.get("device") == "gpu", "GPU reference is not a GPU COMPUTE")
            work = None
            try:
                work_description = matmul_work(program, target["target_id"])
                work = work_description["gemm_flops"] * native["samples_completed"]
            except (ValueError, KeyError, TypeError) as error:
                work_description = dict(identified=False, reason=str(error))
                warnings.append(name + ": reference GEMM work unknown: " + str(error))
            intervals = interval_union([i for a in attribution["actions"] for i in a["kernel_union"]])
            verified = attribution["capture_qualified"] is True and all(
                a["selected_action_attribution_qualified"] is True for a in attribution["actions"])
            if any(not span["begin_ns"] <= i["begin_ns"] < i["end_ns"] <= span["end_ns"] for i in intervals):
                sources[name]["unqualified_kernel_intervals"] = intervals
                intervals, verified = None, False
                warnings.append(name + ": correlated kernels outside sensor boundary retained as diagnostics; active interval unknown")
            if not verified:
                warnings.append(name + ": GPU capture unqualified; forward reference rate unknown")
            sample.update(resource=GPU_RESOURCE, work=work, work_unit="flop", timing_scope="device_compute_active",
                          kernel_intervals=intervals, gpu_intervals_verified=verified)
            sample["evidence"].extend(["known GEMM FLOPs only; all correlated reference-action kernels remain in denominator",
                "kernel envelopes establish activity duration, not SM-active or preemption-free service",
                "work-description-sha256:" + digest(work_description)])
            sources[name]["work_description"] = work_description
        else:
            target = program["actions"][native["target_action_index"]]
            require(target.get("kind") == "COMPUTE" and target.get("device") == "cpu", "CPU reference is not a CPU COMPUTE")
            cpu = native.get("batch_cpu", {}).get("main_thread_active_cpu_ns")
            require(cpu is None or type(cpu) is int and cpu >= 0, "invalid CPU cumulative counter")
            sample.update(resource=CPU_RESOURCE, work=cpu, cpu_active_ns=cpu, work_unit="thread_cpu_ns",
                          timing_scope="thread_active_service", physical_interval_verified=False)
            sample["evidence"].append("batch cumulative CPU counter includes query/reset/runtime/capture work and quantization; pure physical interval unverified")
            warnings.append(name + ": CPU counter candidate retained; physical_interval_verified=False; forward CPU service unknown")
        samples.append(sample)
    artifact = None if not samples else build_reference_supply(samples, context=context,
        clock_id="windows_qpc_floor_ns", clock_origin="absolute_qpc_zero", horizon_ns=horizon,
        max_hold_ns=max_hold, target_windows=[spans["main"]])
    refsdata = dict(schema=SCHEMA, execution=dict(path=str(execution_path), sha256=execution_sha),
        manifest=dict(path=str(manifest_path), sha256=manifest_sha), job=job,
        effective_job_inventory_sha256=digest(jobs),
        auxiliary_job_indices={name: dict(path=str(campaign_root / name),
            sha256=hashlib.sha256((campaign_root / name).read_bytes()).hexdigest())
            for name in ("job-replacements.json", "bare-transfer-protocol.json", "posttarget-job-index.json") if (campaign_root / name).exists()},
        program_bytes_sha256=program_bytes_sha, program_sha256=program["program_sha256"], oracle_sha256=oracle_sha,
        source_inventory_sha256=manifest["source_inventory_sha256"], context=context,
        actors=sources, horizon_ns=horizon, max_hold_ns=max_hold, target_window=spans["main"],
        main_attribution_sha256=parsed["main"]["artifact_sha256"],
        main_capture_qualified=parsed["main"]["capture_qualified"],
        reference_supply_sha256=None if artifact is None else artifact["artifact_sha256"], warnings=warnings,
        missing_requirements=[] if artifact is not None else ["job declared references=False; no reference supply fabricated"],
        native_numeric_recomputed_here=False, current_manifest_source_files_rehashed=False,
        execution_target_timing_used_to_derive_supply=False, after_reference_used_for_forward_supply=False,
        after_reference_state_scope=execution.get("posttarget_after_reference_scope", "original_offered_load_drift_audit"),
        after_reference_same_load_drift_valid=not execution.get("posttarget_cleanup_applied", False),
        physical_CPU_interval_verified=False, universal_available_capacity_identified=False,
        reference_sensor_neutrality_qualified=False, reference_to_target_transfer_qualified=False,
        hardware_accuracy_qualified=False, source_kind="observed_diagnostic")
    refsdata["artifact_sha256"] = digest(refsdata)
    return artifact, refsdata
