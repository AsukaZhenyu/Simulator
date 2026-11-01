"""Physical work / independent available-service integrals, never wall fits."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

SCHEMA = "available-service-efficiencies-1"
SOURCE_SCHEMA = "available-service-stage-observations-1"
SCOPES = {
    "flop": "device_compute_active",
    "byte": "payload_transfer_active",
    "thread_cpu_ns": "thread_active_service",
}
FIELDS = {
    "observation_id", "target_parameter", "supply_parameter", "work", "work_unit",
    "begin_ns", "end_ns", "timing_scope", "clock_id", "clock_origin", "supply_scope",
    "context", "run_id", "process_id", "seed", "split", "started_at_utc", "ended_at_utc",
    "overheads_excluded", "other_resources_controlled", "numerical_verified", "evidence",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value):
    if type(value) is int:
        return Fraction(value)
    if type(value) is Fraction:
        return value
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        n, d = value["numerator"], value["denominator"]
        if type(n) is int and type(d) is int and d > 0:
            return Fraction(n, d)
    raise ValueError("work requires an integer or exact rational; floats are not evidence")


def encoded(value):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def utc(value):
    require(isinstance(value, str), "explicit UTC timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0,
            "timestamps must explicitly use UTC")
    return result


def median(values):
    values = sorted(values)
    if not values:
        return None
    i = len(values) // 2
    return values[i] if len(values) % 2 else (values[i - 1] + values[i]) / 2


def cv(values):
    values = list(values)
    if not values:
        return None
    average = sum(values) / len(values)
    return 0.0 if average == 0 and all(v == 0 for v in values) else (
        None if average == 0 else statistics.pstdev([float(v) for v in values]) / float(average))


def validate_row(row, context, frozen):
    require(isinstance(row, dict) and set(row) == FIELDS, "physical stage observation fields differ")
    for key in ("observation_id", "target_parameter", "supply_parameter", "clock_id", "clock_origin",
                "supply_scope", "run_id"):
        require(isinstance(row[key], str) and row[key].strip(), "nonempty " + key + " required")
    require(row["split"] == "profile", "heldout or diagnostic observations cannot estimate parameters")
    require(row["context"] == context, "exact physical stage context differs")
    require(type(row["process_id"]) is int and row["process_id"] > 0 and type(row["seed"]) is int,
            "positive process ID and integer seed required")
    require(type(row["begin_ns"]) is int and type(row["end_ns"]) is int and
            0 <= row["begin_ns"] < row["end_ns"], "nonempty physical stage interval required")
    require(row["work_unit"] in SCOPES and row["timing_scope"] == SCOPES[row["work_unit"]],
            "inclusive completed actions or waits cannot become physical efficiency")
    if row["work"] is not None:
        require(exact(row["work"]) > 0, "positive work or explicit unknown required")
    for key in ("overheads_excluded", "other_resources_controlled", "numerical_verified"):
        require(row[key] is None or type(row[key]) is bool, key + " must be boolean or unknown")
    require(isinstance(row["evidence"], list) and row["evidence"] and
            all(isinstance(v, str) and v.strip() for v in row["evidence"]), "explicit stage evidence required")
    require(utc(row["started_at_utc"]) <= utc(row["ended_at_utc"]) <= frozen,
            "stage run lifetime/freeze chronology differs")


def freeze_efficiencies(source_files, available_by_run, *, context, frozen_at_utc,
                        supply_source_files=None,
                        minimum_runs=3, minimum_samples_per_run=1,
                        within_process_cv_max=0.15, between_process_cv_max=0.20):
    """Freeze dimensionless ServiceBinding efficiencies from active stages.

    Each source maps a path to its expected SHA256 and uses SOURCE_SCHEMA.
    A scalar is work divided by modeling's exact available-service integral.
    Other bottlenecks and fixed overhead must be separately controlled/excluded;
    whole action wall intervals and offered-load labels are rejected as scopes.
    No heldout scalar, layer offset, duration predictor or source fallback exists.
    """
    from llm_infer_model.performance.available_resources import AvailableResources, ServiceBinding

    require(isinstance(source_files, dict) and source_files, "explicit source file SHA map required")
    require(isinstance(context, dict) and context, "exact nonempty context required")
    require(type(minimum_runs) is int and minimum_runs >= 3, "at least three independent runs required")
    require(type(minimum_samples_per_run) is int and minimum_samples_per_run > 0, "positive sample gate required")
    for value in (within_process_cv_max, between_process_cv_max):
        require(type(value) in (int, float) and 0 <= value < float("inf"), "finite nonnegative CV gates required")
    frozen = utc(frozen_at_utc)
    require(supply_source_files is None or isinstance(supply_source_files, dict), "supply receipts must map run IDs")
    supply_source_files = {} if supply_source_files is None else supply_source_files
    rows, ids, groups, metadata, source_receipts, used_supplies = [], set(), defaultdict(list), {}, [], {}
    used_supply_sources, supply_issues = {}, {}
    for filename, expected in sorted(source_files.items()):
        path = Path(filename).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(isinstance(expected, str) and len(expected) == 64 and actual == expected, "physical source SHA mismatch")
        document = json.loads(raw)
        require(set(document) == {"schema", "observations"} and document["schema"] == SOURCE_SCHEMA and
                isinstance(document["observations"], list), "physical source schema differs")
        source_receipts.append(dict(path=str(path), sha256=actual, bytes=len(raw)))
        for source_row in document["observations"]:
            row = json.loads(canonical(source_row))
            validate_row(row, context, frozen)
            require(row["observation_id"] not in ids, "duplicate physical observation ID")
            ids.add(row["observation_id"])
            run_id = row["run_id"]
            run = (row["process_id"], row["seed"], row["started_at_utc"], row["ended_at_utc"])
            require(run_id not in metadata or metadata[run_id] == run, "run metadata differs within one identity")
            metadata[run_id] = run
            available = available_by_run.get(run_id)
            require(isinstance(available, AvailableResources), "every run requires explicit AvailableResources")
            require((row["clock_id"], row["clock_origin"], row["supply_scope"]) ==
                    (available.clock_id, available.clock_origin, available.scope), "external supply clock/origin/scope differs")
            require(available.supply.get(row["supply_parameter"]).unit == row["work_unit"] + "/s",
                    "external service work unit differs")
            used_supplies[run_id] = available.to_dict()
            if run_id not in supply_issues:
                supply_issues[run_id] = []
                receipt = supply_source_files.get(run_id)
                if receipt is None:
                    used_supply_sources[run_id] = None
                    supply_issues[run_id].append("independent_supply_source_receipt_missing")
                else:
                    require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256"},
                            "supply source receipt requires path and SHA256")
                    supply_path = Path(receipt["path"]).resolve()
                    supply_raw = supply_path.read_bytes()
                    supply_sha = hashlib.sha256(supply_raw).hexdigest()
                    require(supply_sha == receipt["sha256"], "independent supply source SHA mismatch")
                    supply_document = json.loads(supply_raw)
                    require(set(supply_document) == {"schema", "available_resources", "measurement_scope",
                            "derived_from_target_timing", "independence_evidence"} and
                            supply_document["schema"] == "independent-available-resource-observation-1",
                            "independent supply observation schema differs")
                    require(supply_document["available_resources"] == available.to_dict(),
                            "independent supply source does not reconstruct supplied trajectory")
                    require(supply_document["measurement_scope"] == "external_resources_available_to_llm",
                            "supply must measure available resources, not background requests or utilization")
                    require(type(supply_document["derived_from_target_timing"]) is bool,
                            "supply target-timing derivation flag must be explicit")
                    independence = supply_document["independence_evidence"]
                    require(isinstance(independence, list) and all(isinstance(v, str) and v.strip() for v in independence),
                            "supply independence references must be explicit")
                    if supply_document["derived_from_target_timing"] or not independence:
                        supply_issues[run_id].append("supply_independence_unestablished_or_derived_from_target")
                    used_supply_sources[run_id] = dict(path=str(supply_path), sha256=supply_sha)
            issues, integral, candidate = [], None, None
            issues.extend(supply_issues[run_id])
            if available.source_kind != "independent_measurement":
                issues.append("external_supply_not_independently_measured")
            if row["work"] is None:
                issues.append("physical_work_unknown")
            if row["overheads_excluded"] is not True:
                issues.append("fixed_overhead_not_separately_excluded")
            if row["other_resources_controlled"] is not True:
                issues.append("other_resource_attribution_unestablished")
            if row["numerical_verified"] is not True:
                issues.append("numerical_verification_unestablished")
            try:
                integral = available.service_integral(row["supply_parameter"], row["begin_ns"], row["end_ns"])
            except ValueError as error:
                issues.append("available_service_unknown_or_unsupported: " + str(error))
            if integral is not None:
                if integral <= 0:
                    issues.append("nonpositive_available_service_integral")
                elif row["work"] is not None:
                    candidate = exact(row["work"]) / integral
                    if candidate > 1:
                        issues.append("work_exceeds_declared_available_service")
            estimate = dict(observation=row, source_path=str(path), source_sha256=actual,
                            available_service_integral=encoded(integral), candidate_efficiency=encoded(candidate),
                            evidence_issues=issues)
            rows.append(estimate)
            groups[(row["target_parameter"], row["supply_parameter"])].append(estimate)
    require(rows, "nonempty physical profile observations required")
    require(len(set(metadata.values())) == len(metadata), "renamed identical run metadata are not independent")
    # Same PID with overlapping lifetimes cannot manufacture an independent run.
    runs = list(metadata.items())
    for i, (_, (pid, _, start, end)) in enumerate(runs):
        for _, (other_pid, _, other_start, other_end) in runs[i + 1:]:
            require(pid != other_pid or not (max(utc(start), utc(other_start)) < min(utc(end), utc(other_end))
                                            or utc(start) == utc(other_start)), "overlapping PID lifetimes are not independent")
    cells, binding_parameters = [], []
    target_supplies = {}
    for (target, supply), members in sorted(groups.items()):
        require(target not in target_supplies or target_supplies[target] == supply, "one target mixes supply resources")
        target_supplies[target] = supply
        by_run, process_rows, medians, issues = defaultdict(list), [], [], []
        for item in members:
            by_run[item["observation"]["run_id"]].append(item)
        if len(by_run) < minimum_runs:
            issues.append("insufficient_independent_runs")
        for run_id, values in sorted(by_run.items()):
            candidates = [exact(v["candidate_efficiency"]) for v in values if v["candidate_efficiency"] is not None]
            center, variation = median(candidates), cv(candidates)
            run_issues = []
            if any(v["evidence_issues"] for v in values):
                run_issues.append("one_or_more_observations_failed_scope_or_evidence")
            if len(values) < minimum_samples_per_run:
                run_issues.append("minimum_sample_gate_failed")
            if variation is None or variation > within_process_cv_max:
                run_issues.append("within_process_cv_gate_failed")
            if center is not None:
                medians.append(center)
            process_rows.append(dict(run_id=run_id, process_id=metadata[run_id][0], seed=metadata[run_id][1],
                sample_count=len(values), observation_ids=[v["observation"]["observation_id"] for v in values],
                median_efficiency=encoded(center), within_process_cv=variation, evidence_issues=run_issues))
        if any(v["evidence_issues"] for v in process_rows):
            issues.append("one_or_more_runs_failed_evidence_gates")
        between = cv(medians)
        if between is None or between > between_process_cv_max:
            issues.append("between_process_cv_gate_failed")
        candidate = median(medians)
        accepted = not issues
        references = {"sha256:" + v["source_sha256"] + ":" + v["observation"]["observation_id"] for v in members}
        references.update("supply-sha256:"+used_supply_sources[v["observation"]["run_id"]]["sha256"]
                          for v in members if used_supply_sources[v["observation"]["run_id"]] is not None)
        evidence = tuple(sorted(references))
        scope = canonical(dict(context=context, timing_scope="physical_active_service",
                               overheads="separately_excluded", other_resources="controlled")).decode()
        binding = ServiceBinding(target, supply, candidate if accepted else None, "profiled", evidence, scope)
        binding_parameters.append(dict(target_parameter=target, supply_parameter=supply,
            efficiency=encoded(binding.efficiency), origin="profiled", evidence=list(evidence), scope=scope))
        cells.append(dict(target_parameter=target, supply_parameter=supply, processes=process_rows,
            candidate_efficiency=encoded(candidate), between_process_cv=between, accepted=accepted,
            evidence_issues=issues, parameter_kind="physical_stage_efficiency"))
    result = dict(schema=SCHEMA, frozen_at_utc=frozen_at_utc, context=context,
        source_files=source_receipts, available_by_run=used_supplies,
        supply_source_files=used_supply_sources, observation_estimates=rows,
        parameter_cells=cells, binding_parameters=binding_parameters,
        gates=dict(minimum_runs=minimum_runs, minimum_samples_per_run=minimum_samples_per_run,
                   within_process_cv_max=within_process_cv_max, between_process_cv_max=between_process_cv_max),
        aggregation="dimensionless physical stage efficiencies; exact median of equal-weight process medians",
        source_stage_values_reconstructed=True, external_supply_evidence_scope=
            "Independent supply source bytes, trajectory and explicit independence protocol receipts checked; no physical remeasurement or authenticity proof",
        all_observations_retained=True, outliers_removed=0, inclusive_wall_time_converted=False,
        whole_layer_residual_fitted=False, heldout_used_for_fit=False, performance_predictions_computed_here=False,
        hardware_accuracy_qualified=False)
    result["artifact_sha256"] = digest(result)
    return result


def load_efficiency_bindings(document, *, verify_sources=True):
    """Return model bindings; optionally reconstruct current raw source values.

    Unknown or failed cells remain ServiceBinding(efficiency=None). The caller
    cannot turn absent physical observations into calibrated rates by loading.
    """
    from llm_infer_model.performance.available_resources import AvailableResources, ServiceBinding

    if isinstance(document, (str, Path)):
        document = json.loads(Path(document).read_text(encoding="utf-8"))
    document = json.loads(canonical(document))
    require(document.get("schema") == SCHEMA and document.get("artifact_sha256") ==
            digest({k: v for k, v in document.items() if k != "artifact_sha256"}), "frozen efficiency digest differs")
    require(type(verify_sources) is bool, "verify_sources must be explicit boolean")
    if verify_sources:
        rebuilt = freeze_efficiencies({s["path"]: s["sha256"] for s in document["source_files"]},
            {k: AvailableResources.from_dict(v) for k, v in document["available_by_run"].items()},
            context=document["context"], frozen_at_utc=document["frozen_at_utc"],
            supply_source_files=document["supply_source_files"], **document["gates"])
        require(canonical(rebuilt) == canonical(document), "frozen efficiency evidence does not reconstruct")
    require(document["inclusive_wall_time_converted"] is False and document["whole_layer_residual_fitted"] is False and
            document["heldout_used_for_fit"] is False and document["performance_predictions_computed_here"] is False,
            "frozen efficiency violates profiling/modeling ownership")
    return tuple(ServiceBinding(v["target_parameter"], v["supply_parameter"],
        None if v["efficiency"] is None else exact(v["efficiency"]), v["origin"], tuple(v["evidence"]), v["scope"])
        for v in document["binding_parameters"])
