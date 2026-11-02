"""Preserve correlated completed-action measurements, without physical relabeling.

This is a diagnostic bridge for existing inclusive action records. It neither
infers external availability from offered workload nor supplies physical rates.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .calibration import canonical, digest, encoded, median, require, utc

SCHEMA = "coherent-completed-action-ensemble-1"


def _sample(record, invocation):
    """Validate one complete original trace and keep all disjoint token values."""
    begin, end = invocation["begin_ns"], invocation["end_ns"]
    require(type(begin) is int and type(end) is int and 0 <= begin < end, "invalid invocation bounds")
    actions = invocation["actions"]
    inventory = record["action_inventory"]
    require(len(actions) == len(inventory), "inclusive source requires all original actions")
    cursor, durations = begin, []
    for expected, action in enumerate(actions):
        require(isinstance(action, list) and len(action) == 3, "action interval must retain index/begin/end")
        index, left, right = action
        require(type(index) is int and index == expected and type(left) is int and type(right) is int and
                cursor <= left <= right <= end, "action indices/order/interval bounds differ")
        durations.extend((left-cursor, right-left))
        cursor = right
    durations.append(end-cursor)
    require(sum(durations) == end-begin, "disjoint intervals do not cover invocation")
    return dict(observation_id=invocation["observation_id"], begin_ns=begin, end_ns=end,
                measured_complete_call_ns=end-begin, durations_ns=durations)


def _cohort(record, invocation):
    matches = [s for s in record["calibration_segments"] if
               s["begin_ns"] <= invocation["begin_ns"] < invocation["end_ns"] <= s["end_ns"]]
    require(len(matches) <= 1, "overlapping calibration windows")
    if not matches or matches[0]["requests"] is None:
        return None
    segment = matches[0]
    return dict(offered_requests=segment["requests"], host_memory_bytes=segment["host_memory_bytes"],
                device_memory_bytes=segment["device_memory_bytes"])


def freeze_coherent_ensemble(records_path, expected_sha256, *, samples_per_run_state):
    """Retain actual invocation vectors, selected without duration-based fitting.

    ``None`` retains all eligible vectors. An explicit positive bound selects
    chronological equal-width ordinal bin centers independently for each run and
    offered-state cohort. Equal process weights are reported; no marginal price
    is advertised as a representative whole call. Raw observations remain at the
    immutable source path and are not copied wholesale into the compact export.
    """
    from profiling.service_response.evidence import token_inventory, validate_record
    from llm_infer_model.performance.serial_actions import SerialActionTiming, SerialTimingProgram

    require(samples_per_run_state is None or
            (type(samples_per_run_state) is int and samples_per_run_state > 0), "explicit positive sample bound or None required")
    path = Path(records_path).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64 and actual == expected_sha256,
            "coherent source record file SHA mismatch")
    records = json.loads(raw)
    require(isinstance(records, list) and records, "nonempty run-record list required")
    first = records[0]
    program = SerialTimingProgram(tuple(SerialActionTiming(i, a["kind"], f"action:{i}:duration", f"action:{i}:gap_before")
                                       for i, a in enumerate(first["action_inventory"])), "tail")
    sources, selected, cohorts, identities, lifetimes = [], [], [], set(), []
    all_count, eligible_count, context, contract = 0, 0, first["context"], first["contract_sha256"]
    for record in records:
        # Historical digest and record consistency only. Existing absolute source
        # code paths may have changed since collection; no raw reconstruction is
        # claimed here. A separate archived-source audit can accompany this file.
        validate_record(record, split="profile", verify_sources=False)
        require(record["instrumentation"] == "stamped", "inclusive ensemble needs complete stamped profiles")
        require(record["context"] == context and record["context_sha256"] == digest(context) and
                record["contract_sha256"] == contract and record["action_inventory"] == first["action_inventory"],
                "coherent records mix exact action/hardware contexts")
        require(record["token_inventory"] == token_inventory(record["action_inventory"]), "token inventory differs")
        identity = (record["run_id"], record["process_id"], record["seed"])
        require(identity not in identities and all(v[0] != identity[0] for v in identities), "duplicate run identity")
        require(all(v[1:] != identity[1:] for v in identities), "renamed PID/seed run is not independent")
        start, end = utc(record["started_at_utc"]), utc(record["ended_at_utc"])
        require(start <= end, "profile process lifetime is reversed")
        for other_pid, other_start, other_end in lifetimes:
            require(record["process_id"] != other_pid or not (max(start, other_start) < min(end, other_end) or
                    start == other_start), "overlapping PID lifetimes are not independent")
        identities.add(identity)
        lifetimes.append((record["process_id"], start, end))
        sources.append(dict(run_id=record["run_id"], process_id=record["process_id"], seed=record["seed"],
            record_sha256=record["record_sha256"], original_source_verified_receipt=record["source_verified"]))
        groups, previous, ids = defaultdict(list), None, set()
        for invocation in record["invocations"]:
            require(invocation["observation_id"] not in ids, "duplicate invocation observation ID")
            ids.add(invocation["observation_id"])
            sample = _sample(record, invocation)
            require(previous is None or previous <= sample["begin_ns"], "invocations overlap or are reordered")
            previous = sample["end_ns"]
            all_count += 1
            cohort = _cohort(record, invocation)
            if cohort is not None:
                groups[canonical(cohort).decode()].append(sample)
                eligible_count += 1
        for cohort_key, members in sorted(groups.items()):
            n = len(members)
            k = n if samples_per_run_state is None else min(n, samples_per_run_state)
            ordinals = list(range(n)) if k == n else [((2*j+1)*n)//(2*k) for j in range(k)]
            require(len(set(ordinals)) == k, "chronological selector duplicated a sample")
            cohort_id = digest(dict(run_id=record["run_id"], cohort=json.loads(cohort_key)))
            observed_median = median([Fraction(s["measured_complete_call_ns"]) for s in members])
            sum_marginal_medians = sum(median([Fraction(s["durations_ns"][i]) for s in members])
                                       for i in range(len(members[0]["durations_ns"])))
            cohorts.append(dict(cohort_id=cohort_id, run_id=record["run_id"], process_id=record["process_id"],
                offered_state=json.loads(cohort_key), candidate_count=n, selected_count=k,
                selected_chronological_ordinals=ordinals,
                candidate_observation_ids_sha256=digest([s["observation_id"] for s in members]),
                measured_whole_call_median_ns=encoded(observed_median),
                sum_marginal_token_medians_ns=encoded(sum_marginal_medians),
                marginal_sum_relative_to_whole_median=encoded(sum_marginal_medians/observed_median-1),
                statistics_role="audit_only; never a layer-duration model parameter"))
            for ordinal in ordinals:
                sample = dict(members[ordinal], sample_id=record["run_id"]+":"+members[ordinal]["observation_id"],
                    run_id=record["run_id"], process_id=record["process_id"], seed=record["seed"],
                    source_record_sha256=record["record_sha256"], cohort_id=cohort_id,
                    offered_state=json.loads(cohort_key), chronological_cohort_ordinal=ordinal)
                sample["tokens"] = [dict(token=token["token"], kind=token["kind"], duration_ns=duration)
                                    for token, duration in zip(record["token_inventory"], sample.pop("durations_ns"))]
                sample["sample_sha256"] = digest(sample)
                selected.append(sample)
    result = dict(schema=SCHEMA, source_records_path=str(path), source_records_sha256=actual,
        source_records_bytes=len(raw), source_validation="record file bytes, original record digest and interval consistency only",
        current_raw_source_reextraction_performed=False, original_source_receipts_retained=True,
        context=context, context_sha256=digest(context), contract_sha256=contract,
        serial_program=program.to_dict(), sources=sources, cohorts=cohorts, samples=selected,
        selector=dict(samples_per_run_state=samples_per_run_state,
            method="all eligible" if samples_per_run_state is None else "chronological equal-width ordinal-bin centers",
            uses_duration_for_selection=False),
        all_original_invocations_checked=all_count, eligible_complete_invocations=eligible_count,
        excluded_invocations_without_one_complete_known_calibration_window=all_count-eligible_count,
        all_original_intervals_retained_in_source=True, within_sample_correlations_preserved=True,
        independent_process_weight="equal per run; equal selected sample weight within each run/state",
        scope="inclusive completed actions and disjoint host gaps; offered state is not available service",
        external_available_service_inferred=False, physical_bindings=[],
        unmet_physical_requirements=["independently supplied available-service trajectory on the same clock",
            "physical stage work and exclusive active-service boundaries",
            "separate fixed submission/resume overhead and completion waiting",
            "established other-resource attribution or a multi-resource service model"],
        whole_layer_residual_fitted=False, heldout_used=False, hardware_accuracy_qualified=False,
        performance_predictions_computed_here=False)
    result["artifact_sha256"] = digest(result)
    return result


def validate_coherent_ensemble(document, *, verify_record_bytes=True):
    """Reconstruct compact selections, without claiming current raw re-extraction."""
    if isinstance(document, (str, Path)):
        document = json.loads(Path(document).read_text(encoding="utf-8"))
    document = json.loads(canonical(document))
    require(document.get("schema") == SCHEMA and document.get("artifact_sha256") ==
            digest({k: v for k, v in document.items() if k != "artifact_sha256"}), "coherent artifact digest differs")
    require(type(verify_record_bytes) is bool, "verify_record_bytes requires a boolean")
    require(document["within_sample_correlations_preserved"] is True and
            document["external_available_service_inferred"] is False and document["physical_bindings"] == [] and
            document["whole_layer_residual_fitted"] is False and document["heldout_used"] is False and
            document["performance_predictions_computed_here"] is False, "coherent diagnostic boundary violated")
    if verify_record_bytes:
        rebuilt = freeze_coherent_ensemble(document["source_records_path"], document["source_records_sha256"],
            samples_per_run_state=document["selector"]["samples_per_run_state"])
        require(canonical(rebuilt) == canonical(document), "coherent vectors do not reconstruct from record bytes")
    return document


def coherent_sample_parameters(document, sample_id, *, verify_record_bytes=True):
    """Supply one complete correlated vector to the EXISTING serial model.

    This is source replay and cannot be presented as independent accuracy,
    physical service, or prediction under another available-resource trajectory.
    """
    from llm_infer_model.performance import Parameter, ParameterBundle
    from llm_infer_model.performance.serial_actions import SerialTimingProgram

    document = validate_coherent_ensemble(document, verify_record_bytes=verify_record_bytes)
    matches = [s for s in document["samples"] if s["sample_id"] == sample_id]
    require(len(matches) == 1, "one exact coherent sample ID required")
    sample = matches[0]
    require(sample["sample_sha256"] == digest({k: v for k, v in sample.items() if k != "sample_sha256"}),
            "coherent sample digest differs")
    evidence = ("record-sha256:"+sample["source_record_sha256"], "sample:"+sample_id,
                "inclusive source replay only; not physical service or independent accuracy")
    scope = canonical(dict(context_sha256=document["context_sha256"], sample_id=sample_id,
                           timing_scope="inclusive_completed_action_and_disjoint_gaps")).decode()
    bundle = ParameterBundle(tuple(Parameter(t["token"], t["duration_ns"], "ns", "profiled", evidence, scope)
                                   for t in sample["tokens"]))
    return SerialTimingProgram.from_dict(document["serial_program"]), bundle
