"""Read-only integrity and observer/lifecycle audit of physical campaigns.

This module never predicts target duration, fits parameters, or interprets
worker invocation occupancy as available CPU/GPU service. Reference probes are
finite, perturbing reference-workload observations, not neutral continuous
capacity sensors. Completed execution rows are immutable inputs; unfinished
jobs remain explicitly pending. All admission flags are independent of model
prediction errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


SCHEMA = "physical-service-campaign-audit-1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(data), _sha(data)


def _integer(value):
    return type(value) is int and value >= 0


def _span(begin, end):
    return _integer(begin) and _integer(end) and begin < end


def _union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if not _span(begin, end):
            raise ValueError("invalid interval")
        if not result or begin > result[-1][1]:
            result.append([begin, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def _length(intervals):
    return sum(end-begin for begin, end in _union(intervals))


def _overlap(intervals, begin, end):
    return _length([(max(left, begin), min(right, end)) for left, right in intervals
                    if max(left, begin) < min(right, end)])


def _gate(values, minimum=1):
    """Do not let all([]) turn absent evidence into a pass."""
    if any(value is False for value in values):
        return "fail"
    if len(values) < minimum or any(value is None for value in values):
        return "unknown"
    return "pass"


def _counter(raw):
    row = raw.get("batch_cpu", {})
    begin, end = row.get("begin_ns"), row.get("end_ns")
    wall = end-begin if _span(begin, end) else None
    active = row.get("main_thread_active_cpu_ns")
    ratio = active/wall if wall and _integer(active) else None
    return dict(begin_ns=begin, end_ns=end, wall_ns=wall, main_thread_active_cpu_ns=active,
                main_thread_active_wall_ratio=ratio,
                active_exceeds_wall=None if ratio is None else ratio > 1,
                counter_quantization_retained=row.get("counter_quantization_retained"),
                per_action_active_cpu_time_measured=False,
                available_cpu_service_inferred=False,
                qualification="cumulative counter, includes runtime/reset/capture; no clipping or per-stage service inference")


def _capture(raw):
    instrument = raw.get("instrumentation")
    cup = raw.get("cupti", {})
    if instrument == "bare":
        disabled = cup.get("enabled") is False
        return dict(status="not_instrumented", capture_integrity_qualified=None,
                    bare_capture_disabled=disabled, records=len(cup.get("records", [])))
    records = cup.get("records", [])
    timed = [r for r in records if r.get("kind") != "external_correlation"]
    invalid = sum(r.get("timestamp_valid") is not True or not _span(r.get("begin_ns"), r.get("end_ns")) for r in timed)
    clock_ok = (cup.get("clock_id") == raw.get("clock", {}).get("clock_id") == "windows_qpc_floor_ns"
                and cup.get("clock_origin") == raw.get("clock", {}).get("clock_origin") == "absolute_qpc_zero"
                and cup.get("qpc_frequency_hz") == raw.get("clock", {}).get("frequency_hz"))
    checks = [instrument == "cupti", cup.get("all_requested_records_retained") is True,
              cup.get("cupti_dropped_records") == 0, cup.get("local_record_overflow") == 0,
              cup.get("callback_errors") == [], cup.get("timestamp_callback_failed") is False,
              cup.get("timestamp_callback_registered_before_activity_enable") is True,
              bool(records), invalid == 0, clock_ok]
    return dict(status="recorded", capture_integrity_qualified=all(checks),
                dropped_records=cup.get("cupti_dropped_records"), local_record_overflow=cup.get("local_record_overflow"),
                callback_errors=cup.get("callback_errors"), timestamp_callback_failed=cup.get("timestamp_callback_failed"),
                invalid_timed_records=invalid, records=len(records), common_clock_verified=clock_ok,
                exact_action_attribution_audited=False,
                qualification="recording integrity only; use physical_records parser for exact attribution; kernel interval is device activity envelope, not guaranteed SM-active time")


def _physical(raw, expected_pid, program_sha):
    begin, end = raw.get("sampling_begin_ns"), raw.get("sampling_end_ns")
    rows = raw.get("iterations", [])
    spans = [(row.get("begin_ns"), row.get("end_ns")) for row in rows]
    valid = (_span(begin, end) and bool(rows) and all(_span(*s) and begin <= s[0] < s[1] <= end for s in spans)
             and all(left[1] <= right[0] for left, right in zip(spans, spans[1:])))
    durations = [right-left for left, right in spans] if valid else []
    clock = raw.get("clock", {})
    checks = [raw.get("schema") == "physical-stage-native-1", raw.get("process_id") == expected_pid,
              raw.get("program_sha256") == program_sha,
              clock.get("clock_id") == "windows_qpc_floor_ns", clock.get("clock_origin") == "absolute_qpc_zero",
              type(clock.get("frequency_hz")) is int and clock["frequency_hz"] > 0,
              raw.get("samples_completed") == raw.get("samples_requested") == len(rows), valid,
              raw.get("owned_Executor_destructor_returned") is True,
              all(raw.get(key, {}).get("passed") is True for key in
                  ("native_repeat_consistency", "warmup_repeat_consistency", "post_replay_consistency"))]
    return dict(identity_and_sample_inventory_qualified=all(checks), process_id=raw.get("process_id"), begin_ns=begin, end_ns=end,
                invocation_count=len(rows), invocation_median_ns=median(durations) if durations else None,
                invocation_min_ns=min(durations) if durations else None,
                invocation_max_ns=max(durations) if durations else None,
                clock=clock, capture=_capture(raw), cpu_counter=_counter(raw))


def _reference_audit(actors, required, max_hold, *, after_scope="backgrounds_requested_unchanged"):
    names = ["before-gpu", "before-cpu", "main", "after-gpu", "after-cpu"]
    result = dict(required=required, complete=all(name in actors for name in names) if required else None,
                  all_measured_windows_disjoint=None, order_verified=None, windows=[], per_resource={},
                  estimate_kind="reference_workload_effective_service_estimate",
                  neutral_continuous_availability_claimed=False,
                  after_reference_used_for_forward_rate=False,
                  after_reference_load_scope=after_scope,
                  after_reference_same_load_drift_claimed=False,
                  qualification="before references are finite sample-hold candidates; after references are drift audits only and include fresh-process validation/warmup/history")
    if not required:
        return result
    windows = [(name, actors[name]["begin_ns"], actors[name]["end_ns"]) for name in names if name in actors]
    result["windows"] = [dict(actor=name, begin_ns=begin, end_ns=end) for name, begin, end in windows]
    if not all(_span(begin, end) for _, begin, end in windows):
        result["all_measured_windows_disjoint"] = False
        return result
    ordered = sorted(windows, key=lambda row: row[1])
    result["all_measured_windows_disjoint"] = all(left[2] <= right[1] for left, right in zip(ordered, ordered[1:]))
    result["order_verified"] = [r[0] for r in ordered] == names if result["complete"] else None
    if "main" not in actors:
        return result
    main = actors["main"]
    for device in ("cpu", "gpu"):
        before, after = actors.get("before-"+device), actors.get("after-"+device)
        row = {}
        if before:
            age_begin, age_end = main["begin_ns"]-before["end_ns"], main["end_ns"]-before["end_ns"]
            row.update(before_end_to_main_begin_ns=age_begin, before_end_to_main_end_ns=age_end,
                       declared_max_hold_ns=max_hold,
                       before_reference_causal_and_within_declared_hold=(type(max_hold) is int and 0 <= age_begin <= age_end <= max_hold))
        if after:
            row["main_end_to_after_begin_ns"] = after["begin_ns"]-main["end_ns"]
            row["after_reference_is_future_audit_only"] = True
        result["per_resource"][device] = row
    return result


def _background(raw, expected_role, expected_pid, program_sha, windows, frequency, *, required_window_names=None):
    """Audit actual host calls and request separately; neither is service."""
    errors, calls = [], []
    clock, go = raw.get("clock", {}), raw.get("go_schedule", {})
    hz = clock.get("frequency_hz")
    clock_ok = (clock.get("clock_id") == "windows_qpc" and hz == frequency and type(hz) is int and hz > 0
                and go.get("clock", {}).get("frequency_hz") == hz and go.get("clock", {}).get("clock_id") == "windows_qpc")
    if not clock_ok:
        return dict(lifecycle_qualified=False, errors=["background/common QPC clock differs"], windows=[])
    to_ns = lambda value: value*1_000_000_000//hz
    resource = "cpu" if expected_role == "cpu-worker" else "gpu"
    go_begin, go_end = go.get("start_tick"), go.get("end_tick")
    go_valid = go.get("schema") == "state-response-go-v1" and _span(go_begin, go_end)
    if not go_valid:
        errors.append("invalid GO schedule")
    phase_spans = []
    for phase in go.get("phases", []):
        left, right = phase.get("begin_tick"), phase.get("end_tick")
        if not _span(left, right) or not go_valid or not go_begin <= left < right <= go_end:
            errors.append("invalid GO phase")
        elif phase.get(resource+"_replay") == 1:
            phase_spans.append((left, right))
    for block in raw.get("blocks", []):
        for row in block.get("iterations", []):
            left, right, check = row.get("begin_tick"), row.get("end_tick"), row.get("enabled_check_tick")
            if not _span(left, right) or not _integer(check) or check > left:
                errors.append("invalid worker invocation or enabled-check time")
                continue
            if not any(a <= check < b for a, b in phase_spans):
                errors.append("worker invocation not authorized by ON phase at enabled check")
            calls.append((to_ns(left), to_ns(right)))
    if any(left[1] > right[0] for left, right in zip(calls, calls[1:])):
        errors.append("worker calls overlap or inventory is unordered")
    sb, se = raw.get("sampling_begin_tick"), raw.get("sampling_end_tick")
    sampling = (to_ns(sb), to_ns(se)) if _span(sb, se) else None
    if sampling is None or any(a < sampling[0] or b > sampling[1] for a, b in calls):
        errors.append("worker invocation outside measured sampling lifecycle")
    identity = (raw.get("format_version") == "state-response-native-v1" and raw.get("role") == expected_role
                and raw.get("process_id") == expected_pid and raw.get("program_sha256") == program_sha
                and raw.get("owned_Executor_destructor_returned") is True)
    if not identity:
        errors.append("worker identity/native cleanup differs")
    on = [(to_ns(a), to_ns(b)) for a, b in phase_spans]
    rows = []
    required = {name for name, _, _ in windows} if required_window_names is None else set(required_window_names)
    for name, begin, end in windows:
        if not _span(begin, end):
            errors.append("invalid inspected main/reference window")
            continue
        requested = _overlap(on, begin, end)
        call_covered = _overlap(calls, begin, end)
        sampled = _overlap([sampling], begin, end) if sampling else 0
        rows.append(dict(actor=name, begin_ns=begin, end_ns=end, window_ns=end-begin,
                         required_for_constant_background_admission=name in required,
                         requested_on_coverage_ns=requested, requested_on_fraction=requested/(end-begin),
                         actual_sampling_lifecycle_coverage_ns=sampled,
                         actual_sampling_lifecycle_covers_window=sampled == end-begin,
                         worker_host_call_union_overlap_ns=call_covered,
                         worker_host_call_union_overlap_fraction=call_covered/(end-begin),
                         overlapping_host_calls=sum(a < end and b > begin for a, b in calls),
                         host_calls_completed_in_window=sum(begin < b <= end for _, b in calls)))
    required_rows = [row for row in rows if row["required_for_constant_background_admission"]]
    coverage = bool(required_rows) and {row["actor"] for row in required_rows} == required and all(
        r["requested_on_coverage_ns"] == r["window_ns"] and r["actual_sampling_lifecycle_covers_window"] for r in required_rows)
    return dict(lifecycle_qualified=not errors and coverage, errors=errors, role=resource,
                sampling_begin_ns=sampling[0] if sampling else None, sampling_end_ns=sampling[1] if sampling else None,
                invocation_count=len(calls), host_call_union_ns=_length(calls), windows=rows,
                required_window_names=sorted(required),
                host_call_observed_in_each_window=bool(rows) and all(r["overlapping_host_calls"] > 0 for r in rows),
                end_reason=raw.get("end_reason"),
                request_used_as_observed_activity=False, physical_service_or_utilization_inferred=False,
                qualification="GO is requested enablement; actual call intervals and sampling lifecycle are measured. Host-call occupancy includes waits and is not CPU/GPU busy time; between-call gaps are retained, not grounds for inventing continuous busy service")


def _process_receipts(directory, pids):
    receipts = []
    receipt_errors = []
    for receipt_path in sorted(directory.glob("*/process.json")):
        try:
            receipt, receipt_sha = _read(receipt_path)
            receipts.append(dict(path=str(receipt_path), sha256=receipt_sha, name=receipt.get("name"),
                pid=receipt.get("pid"), forced_termination=receipt.get("forced_termination"),
                returncode=receipt.get("returncode"), process_exited_verified=receipt.get("process_exited_verified"),
                launch_tick=receipt.get("launch_tick"), stop_requested_tick=receipt.get("stop_requested_tick"),
                exit_observed_tick=receipt.get("exit_observed_tick"),
                listed_in_execution=receipt.get("name") in pids and pids[receipt.get("name")] == receipt.get("pid")))
        except (OSError, ValueError, TypeError) as error:
            receipt_errors.append(dict(path=str(receipt_path), error=str(error)))
    return receipts, receipt_errors


def _posttarget_lifecycle(actors, process_receipts, backgrounds, execution, directory):
    """Check the operational amendment with observed timestamps, not its label."""
    result = dict(applied=execution.get("posttarget_cleanup_applied") is True,
                  after_reference_scope=execution.get("posttarget_after_reference_scope"),
                  qualified=False, errors=[], backgrounds=[], cleanup_receipt=None,
                  after_reference_same_load_drift_claimed=False)
    if not result["applied"] or result["after_reference_scope"] != "unloaded_drift_only":
        result["errors"].append("posttarget operational amendment not explicitly marked")
    binding = execution.get("posttarget_cleanup_receipt", {})
    receipt = {}
    try:
        receipt_path = Path(binding["path"])
        receipt, receipt_sha = _read(receipt_path)
        result["cleanup_receipt"] = dict(path=str(receipt_path), actual_sha256=receipt_sha,
                                         expected_sha256=binding.get("sha256"), content=receipt)
        if receipt_path.resolve() != (directory/"posttarget-cleanup.json").resolve() or receipt_sha != binding.get("sha256"):
            result["errors"].append("posttarget cleanup receipt identity/hash differs")
        index_path = directory.parent.parent/"posttarget-job-index.json"
        _, index_sha = _read(index_path)
        index_binding = receipt.get("index", {})
        if Path(index_binding.get("path", "")).resolve() != index_path.resolve() or index_binding.get("sha256") != index_sha:
            result["errors"].append("posttarget cleanup receipt binds a different operational index")
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["errors"].append("posttarget cleanup receipt unreadable: "+str(error))
    main = actors.get("main", {})
    main_end = main.get("end_ns")
    hz = main.get("clock", {}).get("frequency_hz")
    if not _integer(main_end) or type(hz) is not int or hz <= 0:
        result["errors"].append("main sampling completion clock unavailable")
        return result
    to_ns = lambda tick: tick*1_000_000_000//hz if _integer(tick) else None
    completion = receipt.get("main_sampling_complete", {})
    observed = to_ns(receipt.get("main_completion_observed_tick"))
    cleanup_begin = to_ns(receipt.get("background_stop_begin_tick"))
    cleanup_end = to_ns(receipt.get("background_stop_end_tick"))
    checks = dict(schema=receipt.get("schema") == "physical-posttarget-cleanup-1",
        job_id=receipt.get("job_id") == directory.name,
        scope=receipt.get("posttarget_after_reference_scope") == "unloaded_drift_only",
        clock=receipt.get("clock", {}).get("clock_id") == "windows_qpc" and receipt.get("clock", {}).get("frequency_hz") == hz,
        target_unchanged=receipt.get("target_sampling_altered") is False,
        before_reference_conditions_unchanged=receipt.get("before_reference_conditions_altered") is False,
        no_error_based_selection=receipt.get("numerical_or_prediction_selection") is False,
        main_completion=completion.get("schema") == "physical-stages-sampling-complete-1" and
            completion.get("process_id") == main.get("process_id") and
            completion.get("sampling_begin_ns") == main.get("begin_ns") and completion.get("sampling_end_ns") == main_end and
            completion.get("completed_samples") == main.get("invocation_count"),
        completion_observed_after_sampling=observed is not None and observed >= main_end,
        cleanup_after_completion=cleanup_begin is not None and observed is not None and cleanup_begin >= observed and
            cleanup_end is not None and cleanup_end >= cleanup_begin)
    result["receipt_checks"] = checks
    if not all(checks.values()):
        result["errors"].append("posttarget receipt clock/completion/scope/chronology differs")
    after = [row for row in process_receipts if row["name"] in {"after-gpu", "after-cpu"}]
    after_launches = [to_ns(row.get("launch_tick")) for row in after]
    if after and (any(value is None for value in after_launches) or min(after_launches) < main_end):
        result["errors"].append("after reference construction began before main completed or launch timestamp unknown")
    first_after_launch = min(after_launches) if after_launches and all(value is not None for value in after_launches) else None
    if after and (cleanup_end is None or first_after_launch is None or cleanup_end > first_after_launch):
        result["errors"].append("after reference construction precedes recorded background cleanup completion")
    result.update(main_sampling_end_ns=main_end, first_after_reference_launch_ns=first_after_launch)
    receipt_backgrounds = receipt.get("background_processes", [])
    expected_backgrounds = [row for row in process_receipts if row["name"].endswith("-worker")]
    receipt_keys = ("name", "pid", "stop_requested_tick", "exit_observed_tick", "returncode", "forced_termination", "process_exited_verified")
    if (receipt.get("background_count") != len(expected_backgrounds) or len(receipt_backgrounds) != len(expected_backgrounds) or
        {row.get("name") for row in receipt_backgrounds} != {row["name"] for row in expected_backgrounds} or
        any(not any(all(row.get(key) == expected.get(key) for key in receipt_keys) for row in receipt_backgrounds) for expected in expected_backgrounds)):
        result["errors"].append("posttarget receipt background inventory differs from all process receipts")
    for row in process_receipts:
        name = row["name"]
        if not name.endswith("-worker"):
            continue
        stop, exited = to_ns(row.get("stop_requested_tick")), to_ns(row.get("exit_observed_tick"))
        sample_end = backgrounds.get(name, {}).get("sampling_end_ns")
        checks = dict(stop_requested_after_main=stop is not None and stop >= main_end,
                      sampling_covers_main_end=sample_end is not None and sample_end >= main_end,
                      exit_not_before_stop=stop is not None and exited is not None and exited >= stop,
                      stop_exit_within_cleanup=stop is not None and exited is not None and cleanup_begin is not None and
                          cleanup_end is not None and cleanup_begin <= stop <= exited <= cleanup_end,
                      exit_before_after_construction=not after or (exited is not None and first_after_launch is not None and exited <= first_after_launch))
        result["backgrounds"].append(dict(actor=name, stop_requested_ns=stop, exit_observed_ns=exited,
                                          sampling_end_ns=sample_end, checks=checks))
        if not all(checks.values()):
            result["errors"].append(name+" stopped before main completed or remained during after construction")
    for name, background in backgrounds.items():
        after_rows = [row for row in background.get("windows", []) if row["actor"].startswith("after-")]
        if any(row["worker_host_call_union_overlap_ns"] != 0 or row["actual_sampling_lifecycle_coverage_ns"] != 0 for row in after_rows):
            result["errors"].append(name+" has observed activity/lifecycle during declared unloaded after reference")
    result["qualified"] = not result["errors"]
    return result


def _job(root, manifest, planned, *, posttarget=False):
    path = root/"jobs"/planned["id"]/"execution.json"
    if not path.exists():
        receipts, errors = _process_receipts(path.parent, {})
        return dict(job_id=planned["id"], split=planned["split"], state="pending", diagnostic_included=False,
                    exclusions=["no completed execution record"], actors={}, backgrounds={},
                    process_receipts=receipts, process_receipt_errors=errors)
    try:
        execution, execution_sha = _read(path)
    except (OSError, ValueError, TypeError) as error:
        receipts, receipt_errors = _process_receipts(path.parent, {})
        return dict(job_id=planned["id"], split=planned["split"], state="unreadable_execution", diagnostic_included=False,
                    exclusions=["execution record unreadable; may be incomplete during active collection"],
                    error=str(error), process_receipts=receipts, process_receipt_errors=receipt_errors, actors={}, backgrounds={})
    expected = {"main"}
    if planned["references"]:
        expected.update({"before-cpu", "before-gpu", "after-cpu", "after-gpu"})
    expected.update(device+"-worker" for device in ("cpu", "gpu") if planned[device+"_background"])
    processes = execution.get("processes", [])
    pids = {p.get("name"): p.get("pid") for p in processes}
    receipts, receipt_errors = _process_receipts(path.parent, pids)
    binding = manifest["programs"][planned["program_key"]]
    checks = dict(completed=execution.get("status") == "completed",
                  job_identity=execution.get("job") == planned,
                  program_binding=execution.get("program") == binding,
                  source_inventory=execution.get("source_inventory_sha256") == manifest["source_inventory_sha256"],
                  process_inventory=set(pids) == expected and len(processes) == len(expected),
                  owned_cleanup=execution.get("cleanup_verified") is True and all(
                      p.get("process_exited_verified") is True and p.get("returncode") == 0 and
                      p.get("forced_termination") is False for p in processes),
                  all_process_receipts_accounted=not receipt_errors and len(receipts) == len(expected) and
                      {p["name"] for p in receipts} == expected and all(p["listed_in_execution"] for p in receipts),
                  all_process_receipts_clean=bool(receipts) and not receipt_errors and all(
                      p["process_exited_verified"] is True and p["forced_termination"] is False and p["returncode"] == 0 for p in receipts),
                  numerical_inventory=set(execution.get("numerical", {})) == expected and all(
                      execution.get("numerical", {}).get(name, {}).get("passed") is True for name in expected),
                  raw_inventory=set(execution.get("raw_files", {})) == expected)
    actors, backgrounds, errors, raw_hashes, raw_backgrounds = {}, {}, [], {}, {}
    for name, record in execution.get("raw_files", {}).items():
        try:
            raw, actual_sha = _read(record["path"])
            raw_hashes[name] = dict(path=record["path"], expected_sha256=record.get("sha256"), actual_sha256=actual_sha,
                                    matches=actual_sha == record.get("sha256"))
            if name.endswith("-worker"):
                raw_backgrounds[name] = raw
            else:
                actors[name] = _physical(raw, pids.get(name), binding["program_sha256"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(actor=name, error=str(error)))
    checks["raw_hashes"] = bool(raw_hashes) and all(row["matches"] for row in raw_hashes.values())
    checks["raw_readable"] = not errors
    checks["native_samples_and_identity"] = bool(actors) and all(r["identity_and_sample_inventory_qualified"] for r in actors.values())
    checks["capture_integrity"] = all(r["capture"]["capture_integrity_qualified"] is True if
                                            r["capture"]["status"] == "recorded" else r["capture"]["bare_capture_disabled"]
                                            for r in actors.values()) and bool(actors)
    refs = _reference_audit(actors, planned["references"], manifest["protocol"].get("max_reference_hold_ns"),
                            after_scope="unloaded_drift_only" if posttarget else "backgrounds_requested_unchanged")
    checks["reference_windows"] = not planned["references"] or (refs["complete"] is True and
        refs["all_measured_windows_disjoint"] is True and refs["order_verified"] is True)
    windows = [(name, row["begin_ns"], row["end_ns"]) for name, row in actors.items()]
    frequency = actors.get("main", {}).get("clock", {}).get("frequency_hz")
    for name, raw in raw_backgrounds.items():
        try:
            backgrounds[name] = _background(raw, name, pids.get(name), binding["program_sha256"], windows, frequency,
                required_window_names={n for n in actors if not n.startswith("after-")} if posttarget else None)
        except (TypeError, ValueError, KeyError) as error:
            backgrounds[name] = dict(lifecycle_qualified=False, errors=[str(error)])
    checks["background_lifecycle"] = set(backgrounds) == {n for n in expected if n.endswith("-worker")} and all(
        row["lifecycle_qualified"] for row in backgrounds.values())
    posttarget_audit = _posttarget_lifecycle(actors, receipts, backgrounds, execution, path.parent) if posttarget else None
    checks["posttarget_operational_amendment"] = posttarget_audit["qualified"] if posttarget else execution.get("posttarget_cleanup_applied") is not True
    exclusions = [key for key, passed in checks.items() if not passed]
    hold_flags = [row.get("before_reference_causal_and_within_declared_hold") for row in refs["per_resource"].values()]
    reference_hold_status = _gate(hold_flags, 2) if planned["references"] else "not_applicable"
    return dict(job_id=planned["id"], split=planned["split"], state=execution.get("status"),
                program_key=planned["program_key"], instrumentation=planned["instrumentation"], references=planned["references"],
                execution_sha256=execution_sha, checks=checks, diagnostic_included=not exclusions,
                exclusions=exclusions, raw_files=raw_hashes, raw_errors=errors, actors=actors,
                process_receipts=receipts, process_receipt_errors=receipt_errors,
                execution_cleanup_verified_claim=execution.get("cleanup_verified"),
                references_audit=refs, backgrounds=backgrounds,
                posttarget_cleanup_audit=posttarget_audit,
                reference_staleness_gate=reference_hold_status,
                reference_conditioned_diagnostic_included=not exclusions and reference_hold_status == "pass",
                main_invocation_median_ns=actors.get("main", {}).get("invocation_median_ns"),
                hardware_accuracy_qualified=False)


def _effective_jobs(root, manifest, manifest_sha):
    """Verify additive index bytes; retain every superseded failed attempt."""
    jobs = list(manifest["jobs"])
    failed, bindings = [], []
    posttarget_metadata = dict(applied=False, changed_job_ids=[], after_reference_scope=None)
    replacement_path = root/"job-replacements.json"
    if replacement_path.exists():
        index, index_sha = _read(replacement_path)
        if index.get("parent_manifest_sha256") != manifest_sha:
            raise ValueError("replacement parent manifest differs")
        if index.get("schema") != "physical-job-replacements-1" or index.get("maximum_retries") != 1:
            raise ValueError("unsupported replacement protocol")
        bindings.append(dict(path=str(replacement_path), sha256=index_sha))
        for row in index["replacements"]:
            original = next((job for job in jobs if job["id"] == row["original_id"]), None)
            replacement = row["replacement_job"]
            if original is None or {k:v for k,v in original.items() if k != "id"} != {k:v for k,v in replacement.items() if k != "id"}:
                raise ValueError("retry changes workload or protocol")
            failure, failure_sha = _read(row["failed_execution_path"])
            expected_path = root/"jobs"/original["id"]/"execution.json"
            if (Path(row["failed_execution_path"]).resolve() != expected_path.resolve() or
                failure_sha != row["failed_execution_sha256"] or failure.get("status") != "failed" or
                "readiness deadline" not in failure.get("error", "")):
                raise ValueError("replacement requires retained matching readiness failure")
            attempt = _job(root, manifest, original)
            attempt.update(replacement_job_id=replacement["id"], selection_reason=row.get("selection_reason"),
                           effective_campaign_job=False, diagnostic_included=False)
            failed.append(attempt)
            jobs[jobs.index(original)] = replacement
    addition_path = root/"bare-transfer-protocol.json"
    if addition_path.exists():
        index, index_sha = _read(addition_path)
        if index.get("parent_manifest_sha256") != manifest_sha:
            raise ValueError("bare transfer parent manifest differs")
        bindings.append(dict(path=str(addition_path), sha256=index_sha))
        jobs.extend(index["jobs"])
    posttarget_path = root/"posttarget-job-index.json"
    if posttarget_path.exists():
        index, index_sha = _read(posttarget_path)
        if index.get("schema") != "physical-posttarget-job-index-1" or index.get("parent_manifest_sha256") != manifest_sha:
            raise ValueError("posttarget index schema/parent manifest differs")
        if index.get("posttarget_after_reference_scope") != "unloaded_drift_only":
            raise ValueError("posttarget after references must be explicitly unloaded drift only")
        frozen = index.get("frozen_file", {})
        frozen_bytes = Path(frozen["path"]).read_bytes()
        if _sha(frozen_bytes) != frozen.get("sha256"):
            raise ValueError("posttarget index frozen input differs")
        bindings.append(dict(path=str(posttarget_path), sha256=index_sha))
        for row in index.get("replacements", []):
            original = next((job for job in jobs if job["id"] == row["original_id"]), None)
            replacement = row["replacement_job"]
            if original is None or {k:v for k,v in original.items() if k != "id"} != {k:v for k,v in replacement.items() if k != "id"}:
                raise ValueError("posttarget retry changes workload or protocol")
            if original["split"] not in {"heldout", "heldout_bare"} or "-retry" in original["id"] or replacement["id"] != original["id"]+"-retry1":
                raise ValueError("posttarget retry is not one fresh heldout retry")
            failure, failure_sha = _read(row["failed_execution_path"])
            expected_path = root/"jobs"/original["id"]/"execution.json"
            if (Path(row["failed_execution_path"]).resolve() != expected_path.resolve() or
                failure_sha != row["failed_execution_sha256"] or failure.get("status") != "failed" or
                "readiness deadline" not in failure.get("error", "")):
                raise ValueError("posttarget retry requires retained matching readiness failure")
            attempt = _job(root, manifest, original)
            attempt.update(replacement_job_id=replacement["id"], selection_reason=row.get("selection_reason"),
                           effective_campaign_job=False, diagnostic_included=False)
            failed.append(attempt)
            jobs[jobs.index(original)] = replacement
        expected = {job["id"]: job for job in jobs}
        indexed = index.get("jobs", [])
        if len(indexed) != len(expected) or {job["id"]: job for job in indexed} != expected:
            raise ValueError("posttarget effective job inventory/configuration differs")
        changed = index.get("posttarget_cleanup_job_ids", [])
        original_done = index.get("original_completed_job_ids", [])
        if (len(set(changed)) != len(changed) or not set(changed) <= set(expected) or
            len(set(original_done)) != len(original_done) or not set(original_done) <= set(expected) or
            set(changed) & set(original_done)):
            raise ValueError("posttarget changed/completed job identities overlap or differ")
        selected = {job["id"] for job in jobs if job["split"] in {"heldout", "heldout_bare"}}
        if set(original_done) | set(changed) != selected:
            raise ValueError("posttarget changed/completed jobs are not a heldout partition")
        declared = datetime.fromisoformat(index["declared_at_utc"])
        for job_id in set(expected)-set(changed):
            execution, _ = _read(root/"jobs"/job_id/"execution.json")
            if execution.get("status") != "completed" or datetime.fromisoformat(execution["ended_at_utc"]) > declared:
                raise ValueError("unamended job was not already completed before index declaration")
        source_bindings = []
        sources = index.get("source_receipts")
        if not isinstance(sources, list) or not sources:
            raise ValueError("posttarget operational source receipts required")
        for source in sources:
            data = Path(source["path"]).read_bytes()
            actual_sha = _sha(data)
            if actual_sha != source["sha256"]:
                raise ValueError("posttarget operational source receipt changed")
            source_bindings.append(dict(path=source["path"], sha256=actual_sha))
        jobs = indexed
        posttarget_metadata = dict(applied=True, index_path=str(posttarget_path), index_sha256=index_sha,
            changed_job_ids=changed, original_completed_job_ids=original_done,
            after_reference_scope="unloaded_drift_only", same_load_after_drift_claimed=False,
            frozen_input=dict(path=frozen["path"], sha256=frozen["sha256"]),
            source_receipts=source_bindings)
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate effective job id")
    return jobs, failed, bindings, posttarget_metadata


def _observer(rows, protocol):
    groups = {}
    for row in rows:
        if row.get("split") == "observer" and row.get("program_key"):
            groups.setdefault(row["program_key"], {})[(row.get("instrumentation"), row.get("references"))] = row
    comparisons = []
    limit = protocol["observer_relative_error_max"]
    names = [("cupti_vs_bare_without_references", ("bare", False), ("cupti", False)),
             ("references_with_cupti_vs_cupti_without_references", ("cupti", False), ("cupti", True))]
    gates = {}
    for name, baseline_key, candidate_key in names:
        flags = []
        for program, group in sorted(groups.items()):
            baseline, candidate = group.get(baseline_key), group.get(candidate_key)
            qualified = bool(baseline and candidate and baseline["diagnostic_included"] and candidate["diagnostic_included"])
            left = baseline.get("main_invocation_median_ns") if baseline else None
            right = candidate.get("main_invocation_median_ns") if candidate else None
            delta = right/left-1 if qualified and left and right else None
            passed = abs(delta) <= limit if delta is not None else None
            flags.append(passed)
            comparisons.append(dict(contrast=name, program_key=program,
                baseline_job=baseline.get("job_id") if baseline else None, candidate_job=candidate.get("job_id") if candidate else None,
                baseline_median_ns=left, candidate_median_ns=right, signed_relative_change=delta,
                absolute_relative_change=None if delta is None else abs(delta), threshold=limit, passed=passed))
        gates[name] = dict(status=_gate(flags, protocol["minimum_independent_runs"]), independent_program_pairs=sum(f is not None for f in flags),
                           required_independent_pairs=protocol["minimum_independent_runs"])
    return dict(gates=gates, comparisons=comparisons, comparison_unit="fresh-process invocation medians paired by program/seed repeat; target invocations are not independent campaign replicates",
                tracer_correction_fitted=False, bare_accuracy_claim_permitted=gates[names[0][0]]["status"] == "pass",
                reference_neutrality_proven=False, bare_with_references_control_present=False,
                qualification="reference contrast is conditional on CUPTI; missing bare+references arm prevents full factorial interaction identification")


def _freeze(root, manifest_sha):
    started = root/"heldout-collection-started.json"
    if not started.exists():
        return dict(status="not_started", parameter_freeze_verified=None)
    try:
        row, _ = _read(started)
        binding = row.get("barrier") or {}
        barrier, actual_sha = _read(binding["path"])
        frozen, frozen_sha = _read(barrier["frozen_file"])
        del frozen
        precedes = datetime.fromisoformat(barrier["frozen_at_utc"]) <= datetime.fromisoformat(row["started_at_utc"])
        checks = dict(barrier_hash=actual_sha == binding.get("sha256"), manifest=barrier.get("manifest_sha256") == manifest_sha,
                      frozen_file_hash=frozen_sha == barrier.get("frozen_file_sha256"), frozen_before_heldout_launch=precedes)
        return dict(status="verified" if all(checks.values()) else "failed", parameter_freeze_verified=all(checks.values()),
                    checks=checks, barrier_path=binding["path"], barrier_sha256=actual_sha,
                    frozen_file=barrier["frozen_file"], frozen_file_sha256=frozen_sha,
                    qualification="audits saved barrier and current frozen bytes; collector separately checks before every launch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(status="failed", parameter_freeze_verified=False, error=str(error))


def audit_campaign(root):
    """Return JSON-ready audit without writing files or running any native work."""
    root = Path(root).resolve()
    manifest, manifest_sha = _read(root/"manifest.json")
    if manifest.get("schema") != "physical-service-campaign-1":
        raise ValueError("unsupported physical campaign schema")
    inventory_digest = _sha(json.dumps(manifest["sources"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    jobs, failed, index_bindings, posttarget_metadata = _effective_jobs(root, manifest, manifest_sha)
    rows = [_job(root, manifest, job, posttarget=job["id"] in posttarget_metadata["changed_job_ids"]) for job in jobs]
    inventory_ok = inventory_digest == manifest.get("source_inventory_sha256")
    if not inventory_ok:
        for row in rows+failed:
            row["diagnostic_included"] = False
            row["exclusions"].append("manifest_source_inventory")
            row["reference_conditioned_diagnostic_included"] = False
    known_ids = {row["job_id"] for row in rows+failed}
    unknown_attempts = []
    for directory in sorted((root/"jobs").glob("*")):
        if directory.is_dir() and directory.name not in known_ids:
            receipts, errors = _process_receipts(directory, {})
            unknown_attempts.append(dict(job_id=directory.name, process_receipts=receipts, process_receipt_errors=errors,
                                         diagnostic_included=False, reason="job directory absent from validated effective job index"))
    observer = _observer(rows, manifest["protocol"])
    completed = [row for row in rows if row["state"] != "pending"]
    freeze = _freeze(root, manifest_sha)
    return dict(schema=SCHEMA, campaign_root=str(root), manifest_sha256=manifest_sha,
                auditor_path=str(Path(__file__).resolve()), auditor_sha256=_sha(Path(__file__).read_bytes()),
                source_inventory_digest_matches=inventory_ok,
                source_files_rehashed=False, source_binding_scope="manifest semantic inventory and per-execution identity; source snapshot/launch verification remains collector responsibility",
                counts=dict(planned=len(rows), with_execution=len(completed), pending=len(rows)-len(completed),
                            diagnostic_included=sum(r["diagnostic_included"] for r in rows),
                            excluded_with_execution=sum(not r["diagnostic_included"] for r in completed)),
                jobs=rows, retained_failed_attempts=failed, unindexed_attempts=unknown_attempts, job_index_files=index_bindings,
                posttarget_operational_amendment=posttarget_metadata,
                observer=observer, heldout_freeze=freeze,
                qualifications=dict(diagnostic_inclusion_independent_of_prediction_error=True,
                    target_timing_used_to_derive_supply=False, background_enablement_equals_service=False,
                    universal_available_resource_measurement_claimed=False,
                    continuous_resource_trajectory_measured=False,
                    physical_action_attribution_audited=False,
                    hardware_accuracy_qualified=False,
                    actual_device_recovery_identified=False,
                    after_reference_same_load_drift_claimed=False,
                    calibration_or_prediction_performed=False),
                limitations=["Static background conditions per fresh process do not test a within-invocation resource transition.",
                    "Completed reference probes perturb resources and retain contexts/buffers; a finite workload rate is not neutral universal capacity.",
                    "After-reference delay includes process construction, validation and warmup, and cannot isolate device recovery.",
                    "Posttarget-amended jobs stop owned backgrounds after main completion; their after references are unloaded drift diagnostics, not same-load observations.",
                    "CPU cumulative counters may exceed wall time at their quantization scale; no clipping or per-action availability inference.",
                    "Device activity intervals may include preemption/stalls and do not prove active SM service.",
                    "A failed observer gate retains raw diagnostics but forbids claiming the traced campaign validates bare execution accuracy."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="optional new JSON file; never overwrites an existing result")
    args = parser.parse_args(argv)
    result = audit_campaign(args.root)
    content = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
