"""Profile-only external-request to owned-worker call response evidence.

This estimates a control-response proxy, not GPU occupancy or an execution-time
offset. No foreground duration is read. Prediction/integration belongs to the
model-owned RequestResponseDelays interface.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from .analysis import canonical, cv, encoded, exact, median

DIRECTIONS = ("cpu_start_ns", "cpu_stop_ns", "gpu_start_ns", "gpu_stop_ns")
DEFAULT_CONFIG = dict(enabled=True, minimum_edges_per_process=1,
    minimum_independent_processes=3, within_process_cv_max=.15,
    between_process_cv_max=.20, rounding="ceil_exact_median_ns")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _int(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name + " requires integer")
    return value


def _read_bound(record, key):
    path = record["raw_bindings"].get(key)
    if path is None:
        return None, None
    _require(path in record["source_files"], "worker/event raw binding has no source hash")
    raw = Path(path).read_bytes()
    expected = record["source_files"][path]
    _require(hashlib.sha256(raw).hexdigest() == expected, "worker/event raw source bytes changed")
    return json.loads(raw), dict(path=path, sha256=expected)


def control_edges(go, resource):
    """Explicit request flips, including the final scheduled request to idle."""
    _require(resource in {"cpu", "gpu"} and isinstance(go, dict), "resource and GO required")
    phases = go.get("phases")
    _require(isinstance(phases, list) and phases, "GO requires phases")
    start, end = _int(go["start_tick"], "GO start"), _int(go["end_tick"], "GO end", 1)
    cursor, previous, edges = start, 0, []
    for index, phase in enumerate(phases):
        _require(phase.get("phase_index") == index and phase["begin_tick"] == cursor, "GO phases reordered or not contiguous")
        cursor = _int(phase["end_tick"], "GO phase end", cursor + 1)
        current = phase[resource + "_replay"]
        _require(type(current) is int and current in (0, 1), "GO replay is numeric 0/1")
        if current != previous:
            edges.append(dict(edge_index=len(edges), phase_index=index, tick=phase["begin_tick"],
                              old=previous, new=current, direction=resource + ("_start_ns" if current else "_stop_ns")))
        previous = current
    _require(cursor == end, "GO final boundary differs")
    if previous:
        edges.append(dict(edge_index=len(edges), phase_index=len(phases), tick=end, old=1, new=0,
                          direction=resource + "_stop_ns", terminal_scheduled_idle=True))
    return edges


def _request_at(go, resource, tick):
    if tick < go["start_tick"] or tick >= go["end_tick"]:
        return 0
    return next(phase[resource + "_replay"] for phase in go["phases"]
                if phase["begin_tick"] <= tick < phase["end_tick"])


def _worker_intervals(worker, record, go, resource):
    _require(worker.get("role") == resource + "-worker" and type(worker.get("process_id")) is int,
             "worker role/PID differs")
    _require(worker.get("program_sha256") == record["program_sha256"], "worker program identity differs")
    _require(worker.get("actions") == record["action_inventory"], "worker action contract differs")
    _require(worker.get("go_schedule") == go, "worker GO differs from bound external controller GO")
    for key in ("clock_id", "frequency_hz"):
        _require(worker.get("clock", {}).get(key) == record["clock"][key], "worker clock differs")
    boundary = record["context"].get("execution_boundary")
    if boundary is not None:
        _require(worker.get("execution_boundary") == boundary, "worker execution boundary differs")
    _require(worker.get("diagnostic_only") is not True, "diagnostic worker cannot establish fitted response lag")
    begin = _int(worker["sampling_begin_tick"], "worker sampling begin")
    end = _int(worker["sampling_end_tick"], "worker sampling end", begin)
    _require(worker.get("ready_tick", go["start_tick"]) <= go["start_tick"], "worker not ready before GO")
    rows, previous = [], begin
    frequency = record["clock"]["frequency_hz"]
    for bi, block in enumerate(worker.get("blocks", [])):
        intervals = block["iterations"]
        _require(block.get("iteration_count", len(intervals)) == len(intervals), "worker iteration count differs")
        for ii, interval in enumerate(intervals):
            decision = _int(interval["enabled_check_tick"], "worker enabled-check tick")
            left = _int(interval["begin_tick"], "worker call begin", previous)
            right = _int(interval["end_tick"], "worker call end", left + 1)
            _require(begin <= decision <= left < right <= end, "worker invocation chronology differs")
            _require(_request_at(go, resource, decision) == 1, "worker began work without enabled external request")
            expected = round(Fraction((right - left) * 1_000_000_000, frequency))
            _require(abs(interval["elapsed_ns"] - expected) <= 1, "worker elapsed interval differs")
            rows.append(dict(block_index=bi, iteration=interval.get("iteration", ii),
                             enabled_check_tick=decision, begin_tick=left, end_tick=right))
            previous = right
    return rows, begin, end


def extract_worker_lags(record, resource):
    """Return every declared edge, including censored/unknown observations.

    Start: first invocation authorized in the new on request, after quiescence.
    Stop: completion of the last invocation authorized by the old on request.
    A zero stop delay requires completed old work plus a sampling clock spanning
    the off boundary. Absence of a completion/clock proof remains unknown.
    """
    _require(record.get("split") == "profile", "worker lag estimation is profile-only")
    external, event_source = _read_bound(record, "events")
    _require(isinstance(external, dict) and isinstance(external.get("go"), dict), "lag evidence requires bound external GO")
    go = external["go"]
    edges = control_edges(go, resource)
    worker, source = _read_bound(record, "worker_" + resource)
    if worker is None:
        return dict(resource=resource, process_id=None, source=None, event_source=event_source,
                    edges=[dict(edge, delay_ns=None, evidence_issue="worker_binding_missing") for edge in edges])
    intervals, sampling_begin, sampling_end = _worker_intervals(worker, record, go, resource)
    frequency = record["clock"]["frequency_hz"]
    result = []
    for index, edge in enumerate(edges):
        tick = edge["tick"]
        next_flip = edges[index + 1]["tick"] if index + 1 < len(edges) else None
        value, selected, issue, proof = None, None, None, None
        if edge["new"]:
            inflight = [row for row in intervals if row["enabled_check_tick"] < tick < row["end_tick"]]
            candidates = [row for row in intervals if row["enabled_check_tick"] >= tick and
                          (next_flip is None or row["enabled_check_tick"] < next_flip)]
            if inflight:
                issue = "on_edge_not_quiescent_prior_invocation_still_inflight"
            elif not candidates:
                issue = "on_request_had_no_identifiable_started_invocation_before_next_flip"
            else:
                selected = candidates[0]
                if next_flip is not None and selected["begin_tick"] >= next_flip:
                    issue = "authorized_invocation_started_after_next_request_flip_censored"
                else:
                    value = Fraction((selected["begin_tick"] - tick) * 1_000_000_000, frequency)
                    proof = "first worker call authorized after on edge; no earlier inflight call"
        else:
            previous_on = edges[index - 1]["tick"] if index else go["start_tick"]
            candidates = [row for row in intervals if previous_on <= row["enabled_check_tick"] < tick]
            if not sampling_begin <= tick <= sampling_end:
                issue = "worker_sampling_clock_does_not_cover_off_boundary"
            elif not candidates:
                issue = "off_edge_has_no_completed_old_request_invocation_proof"
            else:
                selected = candidates[-1]
                if next_flip is not None and selected["end_tick"] >= next_flip:
                    issue = "old_request_completion_not_observed_before_next_on_censored"
                else:
                    value = Fraction(max(0, selected["end_tick"] - tick) * 1_000_000_000, frequency)
                    proof = ("old-request call completed before off edge and sampling clock covers boundary; zero residual busy proxy"
                             if selected["end_tick"] <= tick else "last old-request call completes after off boundary")
        result.append(dict(edge, delay_ns=encoded(value), selected_worker_interval=selected,
                           evidence_issue=issue, proof=proof))
    return dict(resource=resource, process_id=worker["process_id"], source=source, event_source=event_source,
                sampling_begin_tick=sampling_begin, sampling_end_tick=sampling_end, edges=result)


def estimate_request_delays(records, *, config):
    from llm_infer_model.performance.conditioned_response import RequestResponseDelays
    _require(isinstance(config, dict) and config.get("enabled") is True, "predeclared enabled lag protocol required")
    _require(set(config) <= set(DEFAULT_CONFIG), "unknown lag protocol field")
    settings = dict(DEFAULT_CONFIG, **config)
    _int(settings["minimum_edges_per_process"], "minimum lag edges", 1)
    _int(settings["minimum_independent_processes"], "minimum independent lag processes", 3)
    _require(settings["rounding"] == "ceil_exact_median_ns", "lag rounding must be predeclared exact ceil")
    for key in ("within_process_cv_max", "between_process_cv_max"):
        _require(type(settings[key]) in (int, float) and math.isfinite(settings[key]) and settings[key] >= 0,
                 "invalid lag variability gate")
    records = tuple(records)
    _require(records and all(record["split"] == "profile" for record in records), "lag profile cohort required")
    extracted = []
    for record in records:
        extracted.append(dict(run_id=record["run_id"], foreground_process_id=record["process_id"], seed=record["seed"],
                              record_sha256=record["record_sha256"],
                              resources=[extract_worker_lags(record, resource) for resource in ("cpu", "gpu")]))
    cells, parameters = [], {}
    for direction in DIRECTIONS:
        processes, medians, issues = [], [], []
        for run in extracted:
            edges = [edge for resource in run["resources"] for edge in resource["edges"] if edge["direction"] == direction]
            if not edges:
                continue  # No declared edge is not a missing replicate of a declared edge.
            values = [exact(edge["delay_ns"]) for edge in edges if edge["delay_ns"] is not None]
            central, spread, process_issues = median(values), cv(values), []
            if any(edge["delay_ns"] is None for edge in edges):
                process_issues.append("one_or_more_declared_edges_unknown_or_censored")
            if len(values) < settings["minimum_edges_per_process"]:
                process_issues.append("insufficient_identifiable_edges")
            if spread is None or spread > settings["within_process_cv_max"]:
                process_issues.append("within_process_edge_cv_gate_failed_or_unknown")
            if central is not None:
                medians.append(central)
            processes.append(dict(run_id=run["run_id"], foreground_process_id=run["foreground_process_id"], seed=run["seed"],
                declared_edges=len(edges), identifiable_edges=len(values), all_edge_delay_ns=[edge["delay_ns"] for edge in edges],
                median_ns=encoded(central), cv=spread, evidence_issues=process_issues))
        if len(medians) < settings["minimum_independent_processes"]:
            issues.append("insufficient_independent_processes")
        if any(process["evidence_issues"] for process in processes):
            issues.append("one_or_more_processes_failed_lag_evidence_gates")
        between = cv(medians)
        if between is None or between > settings["between_process_cv_max"]:
            issues.append("between_process_lag_median_cv_gate_failed_or_unknown")
        central = median(medians)
        parameters[direction] = None if central is None else math.ceil(central)
        cells.append(dict(direction=direction, processes=processes, exact_median_ns=encoded(central),
            applied_integer_ns=parameters[direction], between_process_cv=between,
            evidence_issues=issues, qualified=not issues,
            estimator="exact median of per-process exact edge medians; equal process weight; ceil only at integer model boundary"))
    evidence = tuple(sorted({f"sha256:{resource['source']['sha256']}:{run['run_id']}:{resource['resource']}"
        for run in extracted for resource in run["resources"] if resource["source"] is not None}))
    model = RequestResponseDelays(**parameters, evidence=evidence or ("unknown: no bound profile worker evidence",),
                                  qualified=all(cell["qualified"] for cell in cells))
    report = dict(schema="profile-worker-request-response-lags-1", resolved_protocol=settings,
        cells=cells, profile_worker_evidence=extracted, profile_only=True, foreground_durations_used=False,
        heldout_worker_intervals_used=False, outliers_removed=0, unknown_or_censored_edges_retained=True,
        scope="owned worker invocation busy-interval response proxy; not pure device-active time, occupancy or an additive inference-time offset",
        identifiability_limit="four constant directional lags ignore other-state/history dependence; dynamic validation remains necessary")
    return model, report
