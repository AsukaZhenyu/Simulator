"""Causal reference-workload service estimates, never universal capacity.

Only before-target reference measurements update held rates. Target intervals
are used solely to reject sensor overlap; after-target measurements are drift
audits. A held reference rate remains an estimate for that reference workload,
not a neutral measurement of the LLM's counterfactual available hardware.
"""
from __future__ import annotations

from fractions import Fraction
import json

from .calibration import canonical, digest, encoded, exact, require

SCHEMA = "reference-workload-effective-service-estimate-1"
ESTIMATE_KIND = "reference_workload_effective_service_estimate"
UNITS = {"flop": "flop/s", "byte": "byte/s", "thread_cpu_ns": "thread_cpu_ns/s"}
SCOPES = {"flop": "device_compute_active", "byte": "payload_transfer_active",
          "thread_cpu_ns": "thread_active_service"}


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be an integer >= " + str(minimum))
    return value


def _text(value, label):
    require(isinstance(value, str) and value.strip(), label + " must be nonempty text")
    return value


def _interval(value, label):
    require(isinstance(value, dict) and set(value) == {"begin_ns", "end_ns"}, label + " requires begin_ns/end_ns")
    left, right = _integer(value["begin_ns"], label + " begin"), _integer(value["end_ns"], label + " end")
    require(left < right, label + " must have positive duration")
    return left, right


def _union_length(intervals):
    total, previous_end = 0, None
    for left, right in sorted(intervals):
        if previous_end is None or left >= previous_end:
            total += right - left
        elif right > previous_end:
            total += right - previous_end
        previous_end = right if previous_end is None else max(previous_end, right)
    return total


def _sample(value, context, clock_id, clock_origin):
    required = {"sample_id", "sample_role", "resource", "work", "work_unit", "begin_ns", "end_ns",
                "timing_scope", "sensor_window", "context", "clock_id", "clock_origin", "evidence"}
    optional = {"cpu_active_ns", "kernel_intervals", "copy_intervals", "gpu_intervals_verified",
                "physical_interval_verified", "numerical_verified", "run_id", "process_id"}
    require(isinstance(value, dict) and required <= set(value) and not set(value) - required - optional,
            "reference sample has missing or unexpected fields")
    result = json.loads(canonical(value))
    for key in ("sample_id", "resource"):
        _text(result[key], key)
    require(result["sample_role"] in {"before_target", "after_target"}, "explicit before/after reference role required")
    require(result["work_unit"] in UNITS, "unsupported physical reference work unit")
    require(result["timing_scope"] == SCOPES[result["work_unit"]],
            "reference timing scope must identify physical service, not inclusive host or stream wall time")
    require(result["context"] == context and (result["clock_id"], result["clock_origin"]) ==
            (clock_id, clock_origin), "reference context or clock differs")
    require(isinstance(result["evidence"], list) and result["evidence"] and
            all(isinstance(x, str) and x.strip() for x in result["evidence"]), "explicit source evidence required")
    begin, end = _interval({k: result[k] for k in ("begin_ns", "end_ns")}, "physical reference interval")
    sensor_begin, sensor_end = _interval(result["sensor_window"], "reference sensor window")
    require(sensor_begin <= begin < end <= sensor_end, "physical reference interval lies outside sensor window")
    for key in ("gpu_intervals_verified", "physical_interval_verified", "numerical_verified"):
        require(key not in result or result[key] is None or type(result[key]) is bool, key + " must be boolean or unknown")
    issues = []
    work = None if result["work"] is None else exact(result["work"])
    require(work is None or work >= 0, "reference work must be nonnegative or explicit unknown")
    if work is None:
        issues.append("reference_work_unknown")
    elif work == 0:
        issues.append("reference_work_zero_cannot_estimate_rate")
    if result.get("numerical_verified") is not True:
        issues.append("reference_numerical_verification_unestablished")
    duration = end - begin
    if result["work_unit"] == "thread_cpu_ns":
        cpu_active = result.get("cpu_active_ns")
        if cpu_active is None:
            issues.append("reference_cpu_active_service_unknown")
        else:
            _integer(cpu_active, "reference CPU active service")
            require(work is None or work == cpu_active, "thread service work differs from CPU counter evidence")
            if cpu_active > duration:
                issues.append("reference_cpu_service_exceeds_wall_interval")
        if result.get("physical_interval_verified") is not True:
            issues.append("reference_cpu_interval_unverified")
        interval_kind = "cpu_active_service_over_reference_wall_window"
    else:
        field = "kernel_intervals" if result["work_unit"] == "flop" else "copy_intervals"
        require(not result.get("copy_intervals" if field == "kernel_intervals" else "kernel_intervals"),
                "one reference rate cannot mix compute and copy activity")
        raw_intervals = result.get(field)
        if raw_intervals is None:
            issues.append("reference_correlated_GPU_activity_unknown")
            intervals = []
        else:
            require(isinstance(raw_intervals, list), "GPU activity intervals must be a list")
            intervals = [_interval(item, "correlated GPU activity") for item in raw_intervals]
            require(all(begin <= left < right <= end for left, right in intervals),
                    "correlated GPU activity is outside physical reference boundary")
        duration = _union_length(intervals)
        if not duration:
            issues.append("reference_correlated_GPU_activity_empty")
        if result.get("gpu_intervals_verified") is not True:
            issues.append("reference_GPU_activity_identity_unverified")
        interval_kind = "correlated_GPU_activity_union_not_host_or_stream_envelope"
    candidate = None if work is None or not duration else work * 1_000_000_000 / duration
    rate = candidate if not issues else None
    return result, dict(sample_id=result["sample_id"], sample_role=result["sample_role"], resource=result["resource"],
        unit=UNITS[result["work_unit"]], candidate_rate=encoded(candidate), rate=encoded(rate),
        rate_denominator_ns=duration, rate_denominator_scope=interval_kind,
        sensor_begin_ns=sensor_begin, available_after_ns=sensor_end,
        evidence_issues=issues, source_sample_sha256=digest(result),
        universal_capacity_identified=False, sensor_neutrality_qualified=False)


def build_reference_supply(reference_samples, *, context, clock_id, clock_origin, horizon_ns,
                           max_hold_ns, target_windows=()):
    """Hold only completed before-target reference estimates, with explicit gaps.

    ``max_hold_ns`` is a predeclared validity/cadence limit. No rate is available
    before the reference's sensor window ends, during any sensor window, after
    its expiry, or after a failed newer observation. All references must be
    disjoint from target windows and each other. After-target samples never
    update forward rates, even when they show a more favorable result.
    """
    require(isinstance(context, dict) and context, "nonempty exact context required")
    context = json.loads(canonical(context))
    _text(clock_id, "clock ID"); _text(clock_origin, "clock origin")
    _integer(horizon_ns, "reference-supply horizon", 1)
    _integer(max_hold_ns, "maximum hold/cadence", 1)
    require(isinstance(reference_samples, (list, tuple)) and reference_samples, "nonempty reference samples required")
    normalized, estimates = [], []
    for value in reference_samples:
        row, estimate = _sample(value, context, clock_id, clock_origin)
        require(estimate["available_after_ns"] <= horizon_ns, "reference sensor window exceeds supplied horizon")
        normalized.append(row); estimates.append(estimate)
    require(len({r["sample_id"] for r in normalized}) == len(normalized), "duplicate reference sample ID")
    sensors = sorted((e["sensor_begin_ns"], e["available_after_ns"], e["sample_id"]) for e in estimates)
    require(all(left[1] <= right[0] for left, right in zip(sensors, sensors[1:])), "reference sensor windows overlap")
    targets = []
    require(isinstance(target_windows, (tuple, list)), "target windows must be explicit intervals")
    for window in target_windows:
        begin, end = _interval(window, "target validation window")
        require(end <= horizon_ns, "target window exceeds supplied horizon")
        require(all(end <= left or right <= begin for left, right, _ in sensors), "reference sensor overlaps a target window")
        targets.append(dict(begin_ns=begin, end_ns=end))
    units = {}
    for estimate in estimates:
        name = estimate["resource"]
        require(name not in units or units[name] == estimate["unit"], "one resource mixes reference work units")
        units[name] = estimate["unit"]
    before = [e for e in estimates if e["sample_role"] == "before_target"]
    events = {0, horizon_ns}
    for left, right, _ in sensors:
        events.update((left, right))
    for estimate in before:
        events.add(min(horizon_ns, estimate["available_after_ns"] + max_hold_ns))
    segments = []
    for begin, end in zip(sorted(events), sorted(events)[1:]):
        if begin == end:
            continue
        sensor_ids = [sample for left, right, sample in sensors if left <= begin < right]
        rates, reasons, selected = {}, {}, {}
        for name in sorted(units):
            eligible = [e for e in before if e["resource"] == name and e["available_after_ns"] <= begin]
            latest = max(eligible, key=lambda e: e["available_after_ns"], default=None)
            issue = []
            if latest is None:
                issue.append("no_completed_before_target_reference")
            elif begin >= latest["available_after_ns"] + max_hold_ns:
                issue.append("reference_stale_cadence_limit_exceeded")
            elif latest["rate"] is None:
                issue.extend(latest["evidence_issues"])
            if sensor_ids:
                issue.append("reference_sensor_active")
            rates[name] = None if issue else latest["rate"]
            reasons[name] = issue
            selected[name] = None if latest is None else latest["sample_id"]
        segments.append(dict(begin_ns=begin, end_ns=end, rates=rates, unknown_reasons=reasons,
                             source_samples=selected, active_sensor_samples=sensor_ids))
    drift = []
    for after in [e for e in estimates if e["sample_role"] == "after_target"]:
        candidates = [e for e in before if e["resource"] == after["resource"] and
                      e["available_after_ns"] <= after["sensor_begin_ns"]]
        prior = max(candidates, key=lambda e: e["available_after_ns"], default=None)
        ratio = None if prior is None or after["rate"] is None or prior["rate"] is None else (
            exact(after["rate"]) / exact(prior["rate"]) if exact(prior["rate"]) else None)
        drift.append(dict(after_sample_id=after["sample_id"], prior_before_sample_id=None if prior is None else prior["sample_id"],
                          after_to_before_rate_ratio=encoded(ratio), used_for_forward_supply=False,
                          role="drift_audit_only"))
    artifact = dict(schema=SCHEMA, estimate_kind=ESTIMATE_KIND, context=context,
        clock_id=clock_id, clock_origin=clock_origin, horizon_ns=horizon_ns, max_hold_ns=max_hold_ns,
        reference_samples=normalized, sample_estimates=estimates, resources=units,
        target_windows_for_overlap_check_only=targets, segments=segments, after_reference_drift=drift,
        source_kind="observed_diagnostic", target_timing_used_to_compute_rates=False,
        after_reference_used_for_forward_supply=False, utilization_converted_to_rate=False,
        universal_capacity_identified=False, reference_sensor_neutrality_qualified=False,
        reference_to_target_transfer_qualified=False, hardware_accuracy_qualified=False,
        scope="Causal sample-hold effective service estimate for the measured reference workload; not universal actual resource capacity")
    artifact["artifact_sha256"] = digest(artifact)
    return artifact


def validate_reference_supply(document):
    require(isinstance(document, dict) and document.get("schema") == SCHEMA and
            document.get("artifact_sha256") == digest({k: v for k, v in document.items() if k != "artifact_sha256"}),
            "reference-supply schema/digest differs")
    rebuilt = build_reference_supply(document["reference_samples"], context=document["context"],
        clock_id=document["clock_id"], clock_origin=document["clock_origin"], horizon_ns=document["horizon_ns"],
        max_hold_ns=document["max_hold_ns"], target_windows=document["target_windows_for_overlap_check_only"])
    require(canonical(rebuilt) == canonical(document), "reference-supply trajectory does not reconstruct")
    return document


def materialize_reference_window(document, start_ns, end_ns, *, resources=None):
    """Expose a fully covered finite diagnostic window to the modeling API.

    Unknown/stale coverage is refused, not filled with the latest distant rate.
    The returned object remains observed_diagnostic and requires the modeling
    evaluator's explicit diagnostic replay permission.
    """
    from llm_infer_model.performance import AvailableResources, Parameter, ParameterBundle, ResourcePoint, ResourceTimeline
    validate_reference_supply(document)
    _integer(start_ns, "reference window start"); _integer(end_ns, "reference window end")
    require(start_ns < end_ns <= document["horizon_ns"], "reference window is outside the finite horizon")
    names = sorted(document["resources"]) if resources is None else list(resources)
    require(names and len(set(names)) == len(names) and set(names) <= set(document["resources"]),
            "explicit unique known reference resource names required")
    overlaps = [s for s in document["segments"] if s["begin_ns"] < end_ns and start_ns < s["end_ns"]]
    require(sum(min(end_ns, s["end_ns"]) - max(start_ns, s["begin_ns"]) for s in overlaps) == end_ns - start_ns,
            "reference window coverage incomplete")
    for segment in overlaps:
        for name in names:
            require(segment["rates"][name] is not None,
                    "unknown reference coverage for " + name + ": " + str(segment["unknown_reasons"][name]))
    evidence = ("reference-supply-sha256:" + document["artifact_sha256"],
                "reference workload effective service estimate; sensor neutrality and target transfer unqualified")
    parameters = tuple(Parameter(name, None, document["resources"][name], "profiled", evidence, document["scope"])
                       for name in names)
    points = tuple(ResourcePoint(max(start_ns, s["begin_ns"]),
        rates={name: exact(s["rates"][name]) for name in names}) for s in overlaps)
    return AvailableResources(ParameterBundle(parameters), ResourceTimeline(points), end_ns,
        document["clock_id"], document["clock_origin"], "observed_diagnostic", evidence, document["scope"])
