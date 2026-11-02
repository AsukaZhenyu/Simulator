"""Source-bound independent validation of model-owned available-service timing.

This module compares observations and reports evidence. It never estimates an
execution duration, resource supply, or efficiency from heldout target timing.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .calibration import canonical, digest, encoded, exact, median, require, utc, load_efficiency_bindings

REFERENCE_SCHEMA = "available-service-heldout-observations-1"
REFERENCE_FIELDS = {
    "observation_id", "run_id", "process_id", "seed", "split", "begin_ns", "end_ns",
    "clock_id", "clock_origin", "context", "graph_sha256", "signature_id",
    "started_at_utc", "ended_at_utc", "reference_timing_scope", "numerical_receipt",
    "state", "evidence",
}
DEFAULT_PROTOCOL = dict(minimum_runs=3, minimum_samples_per_run=10,
                        median_relative_error_max=0.20, p90_relative_error_max=0.20)


def _read(path, expected=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        require(actual == expected, "bound source SHA mismatch: " + str(path))
    return json.loads(raw), dict(path=str(path), sha256=actual, bytes=len(raw))


def _text(value, label):
    require(isinstance(value, str) and value.strip(), "nonempty " + label + " required")


def _protocol(value):
    value = dict(DEFAULT_PROTOCOL if value is None else value)
    require(set(value) == set(DEFAULT_PROTOCOL), "validation protocol fields differ")
    require(type(value["minimum_runs"]) is int and value["minimum_runs"] >= 3,
            "at least three independent heldout runs required")
    require(type(value["minimum_samples_per_run"]) is int and value["minimum_samples_per_run"] > 0,
            "positive minimum sample count required")
    for key in ("median_relative_error_max", "p90_relative_error_max"):
        require(type(value[key]) in (int, float) and 0 <= value[key] < float("inf"),
                "finite nonnegative error thresholds required")
    return value


def _profile_identities(frozen):
    runs = {}
    for item in frozen["observation_estimates"]:
        row = item["observation"]
        identity = {k: row[k] for k in ("run_id", "process_id", "seed", "started_at_utc", "ended_at_utc")}
        require(row["run_id"] not in runs or runs[row["run_id"]] == identity,
                "frozen profile run identity differs")
        runs[row["run_id"]] = identity
    return [runs[k] for k in sorted(runs)]


def make_barrier(frozen_file, *, graph, fixed_parameters, signature_id, created_at_utc,
                 protocol=None, verify_sources=True):
    """Bind the saved profile-only model and evaluation contract before launch.

    The caller must persist this returned object with a write-once operation
    before starting heldout processes. UTC metadata alone is not an independent
    audit of a collector's implementation or the operating-system clock.
    """
    from llm_infer_model.performance import Graph, ParameterBundle

    require(isinstance(graph, Graph) and isinstance(fixed_parameters, ParameterBundle),
            "barrier requires model Graph and fixed ParameterBundle")
    _text(signature_id, "signature ID")
    frozen, receipt = _read(frozen_file)
    load_efficiency_bindings(frozen, verify_sources=verify_sources)
    require(utc(frozen["frozen_at_utc"]) <= utc(created_at_utc), "barrier precedes profile freeze")
    protocol = _protocol(protocol)
    result = dict(schema="available-service-heldout-barrier-1", created_at_utc=created_at_utc,
        frozen_file=receipt, frozen_artifact_sha256=frozen["artifact_sha256"],
        graph_sha256=digest(graph.to_dict()), fixed_parameters_sha256=digest(fixed_parameters.to_dict()),
        signature_id=signature_id, context=frozen["context"], protocol=protocol,
        profile_runs=_profile_identities(frozen), profile_sources_verified=verify_sources,
        lifetime_scope="collector-declared UTC lifetimes; source file saved before heldout launch required")
    result["barrier_sha256"] = digest(result)
    return result


def _check_barrier(frozen, barrier, graph, fixed_parameters, verify_sources):
    require(barrier.get("schema") == "available-service-heldout-barrier-1" and
            barrier.get("barrier_sha256") == digest({k: v for k, v in barrier.items() if k != "barrier_sha256"}),
            "invalid available-service heldout barrier")
    actual_frozen, _ = _read(barrier["frozen_file"]["path"], barrier["frozen_file"]["sha256"])
    require(canonical(actual_frozen) == canonical(frozen), "supplied frozen model differs from barrier file")
    require(frozen["artifact_sha256"] == barrier["frozen_artifact_sha256"] and
            frozen["context"] == barrier["context"] and _profile_identities(frozen) == barrier["profile_runs"],
            "frozen profile differs from barrier")
    require(digest(graph.to_dict()) == barrier["graph_sha256"], "heldout graph differs from predeclared graph")
    require(digest(fixed_parameters.to_dict()) == barrier["fixed_parameters_sha256"],
            "fixed overhead parameters changed after barrier")
    _protocol(barrier["protocol"])
    return load_efficiency_bindings(frozen, verify_sources=verify_sources)


def _supply(run_id, available, source_files):
    from llm_infer_model.performance.available_resources import AvailableResources

    require(isinstance(available, AvailableResources), "every heldout run needs AvailableResources")
    receipt = source_files.get(run_id)
    require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256"},
            "every heldout run needs a separate supply source receipt")
    doc, actual = _read(receipt["path"], receipt["sha256"])
    require(set(doc) == {"schema", "available_resources", "measurement_scope",
                        "derived_from_target_timing", "independence_evidence"} and
            doc["schema"] == "independent-available-resource-observation-1", "heldout supply schema differs")
    require(doc["available_resources"] == available.to_dict(), "heldout supply does not reconstruct trajectory")
    require(doc["measurement_scope"] == "external_resources_available_to_llm",
            "heldout supply must describe available resources")
    require(type(doc["derived_from_target_timing"]) is bool, "explicit supply derivation flag required")
    refs = doc["independence_evidence"]
    require(isinstance(refs, list) and all(isinstance(v, str) and v.strip() for v in refs),
            "explicit supply independence evidence required")
    issues = []
    if available.source_kind != "independent_measurement":
        issues.append("external_supply_not_independently_measured")
    if doc["derived_from_target_timing"] or not refs:
        issues.append("external_supply_independence_unestablished_or_target_derived")
    return actual, issues


def _quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = Fraction(len(values) - 1) * probability
    lower = index.numerator // index.denominator
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def _summary(rows, run_ids, predicted_field="predicted_ns"):
    runs, all_errors = [], []
    for run_id in run_ids:
        selected = [r for r in rows if r["run_id"] == run_id]
        errors = [abs(Fraction(r[predicted_field]) - r["observed_ns"]) / r["observed_ns"]
                  for r in selected if r[predicted_field] is not None]
        all_errors.extend(errors)
        runs.append(dict(run_id=run_id, sample_count=len(selected), predictable_count=len(errors),
            median_absolute_relative_error=None if not errors else float(median(errors)),
            p90_absolute_relative_error=None if not errors else float(_quantile(errors, Fraction(9, 10))),
            worst_absolute_relative_error=None if not errors else float(max(errors))))
    centers = [r["median_absolute_relative_error"] for r in runs if r["median_absolute_relative_error"] is not None]
    return dict(sample_count=len(rows), predictable_count=len(all_errors),
        complete_prediction_coverage=len(all_errors) == len(rows),
        process_balanced_median_absolute_relative_error=None if not centers else float(median(centers)),
        pooled_p90_absolute_relative_error=None if not all_errors else float(_quantile(all_errors, Fraction(9, 10))),
        worst_absolute_relative_error=None if not all_errors else float(max(all_errors)), runs=runs)


def _fixed_qualification(graph, fixed_parameters):
    used = sorted({t.fixed_parameter for t in graph.tasks if t.fixed_parameter is not None})
    rows = []
    for name in used:
        p = fixed_parameters.get(name)
        rows.append(dict(parameter_name=name, value_known=p.value is not None, origin=p.origin,
            evidence=list(p.evidence), scope=p.scope, independently_qualified=False,
            issue="fixed-stage physical parameter qualification is outside the efficiency freezer"))
    return dict(parameters=rows, all_required_fixed_parameters_qualified=not rows,
                explicit_fixed_parameters_supplied=True)


def _crosses_supply_change(available, graph, begin_ns, end_ns):
    rates = {p.name: p.value for p in available.supply.parameters if p.unit != "byte"}
    capacities = {d.name: available.supply.get(d.capacity_parameter).value for d in graph.memory_domains}
    for point in available.timeline.points:
        changed = any(rates.get(k) != v for k, v in point.rates.items()) or any(
            capacities.get(k) != v for k, v in point.capacities.items())
        if begin_ns < point.at_ns < end_ns and changed:
            return True
        rates.update(point.rates)
        capacities.update(point.capacities)
        if point.at_ns >= end_ns:
            break
    return False


def _candidate_bindings(frozen, bindings):
    from llm_infer_model.performance.available_resources import ServiceBinding

    candidates = {(c["target_parameter"], c["supply_parameter"]): c["candidate_efficiency"]
                  for c in frozen["parameter_cells"]}
    result = []
    for binding in bindings:
        value = binding.efficiency
        if value is None:
            candidate = candidates.get((binding.target_parameter, binding.supply_parameter))
            value = None if candidate is None else exact(candidate)
            if value is not None and not 0 <= value <= 1:
                value = None
        result.append(ServiceBinding(binding.target_parameter, binding.supply_parameter, value,
            "profiled", binding.evidence + ("diagnostic profile-only candidate, not a qualified parameter",), binding.scope))
    return tuple(result)


def evaluate_available_service(frozen, heldout_source_files, available_by_run, supply_source_files,
                               graph, fixed_parameters, *, barrier, verify_sources=True,
                               diagnostic_candidates=False):
    """Compare fresh source-bound references using only modeling for timing.

    Primary predictions require independent non-target-derived supply and the
    qualified frozen bindings. Explicit diagnostic comparisons may use observed
    supply and profile-only candidate bindings, in separate output fields. They
    never fill primary unknowns or become accepted hardware evidence.
    """
    from llm_infer_model.performance.available_resources import evaluate_available_resources

    require(type(verify_sources) is bool and type(diagnostic_candidates) is bool, "explicit boolean options required")
    if isinstance(frozen, (str, Path)):
        frozen, _ = _read(frozen)
    all_bindings = _check_barrier(frozen, barrier, graph, fixed_parameters, verify_sources)
    required_targets = {d.parameter for task in graph.tasks for d in task.demands}
    bindings = tuple(b for b in all_bindings if b.target_parameter in required_targets)
    diagnostics = _candidate_bindings(frozen, bindings) if diagnostic_candidates else None
    require(isinstance(heldout_source_files, dict) and heldout_source_files, "source-bound heldout references required")
    profiles = barrier["profile_runs"]
    profile_runs = {p["run_id"] for p in profiles}
    profile_seeds = {p["seed"] for p in profiles}
    profile_pids = {p["process_id"] for p in profiles}
    records, source_receipts, seen, runs = [], [], set(), {}
    supply_receipts, supply_issues, receipts_cache = {}, {}, {}
    for path, sha256 in sorted(heldout_source_files.items()):
        document, source = _read(path, sha256)
        require(set(document) == {"schema", "observations"} and document["schema"] == REFERENCE_SCHEMA and
                isinstance(document["observations"], list), "heldout reference schema differs")
        source_receipts.append(source)
        for original in document["observations"]:
            row = json.loads(canonical(original))
            require(isinstance(row, dict) and set(row) == REFERENCE_FIELDS, "heldout reference fields differ")
            for key in ("observation_id", "run_id", "clock_id", "clock_origin", "signature_id"):
                _text(row[key], key)
            require(row["split"] == "heldout", "only heldout references may validate")
            require(row["context"] == frozen["context"] and row["graph_sha256"] == barrier["graph_sha256"] and
                    row["signature_id"] == barrier["signature_id"], "heldout exact context/graph/signature differs")
            require(row["reference_timing_scope"] == "whole_execution_completed", "reference boundary must cover whole execution")
            require(type(row["process_id"]) is int and row["process_id"] > 0 and type(row["seed"]) is int,
                    "positive heldout PID and integer seed required")
            require(type(row["begin_ns"]) is int and type(row["end_ns"]) is int and
                    0 <= row["begin_ns"] < row["end_ns"], "positive reference interval required")
            require(utc(barrier["created_at_utc"]) < utc(row["started_at_utc"]) <= utc(row["ended_at_utc"]),
                    "heldout process did not start after frozen barrier")
            require(row["run_id"] not in profile_runs and row["seed"] not in profile_seeds and
                    row["process_id"] not in profile_pids, "heldout identity reused from profile")
            require(row["observation_id"] not in seen, "duplicate heldout observation ID")
            seen.add(row["observation_id"])
            identity = tuple(row[k] for k in ("process_id", "seed", "started_at_utc", "ended_at_utc"))
            require(row["run_id"] not in runs or runs[row["run_id"]] == identity, "heldout run identity changed")
            runs[row["run_id"]] = identity
            require(row["state"] is None or isinstance(row["state"], dict), "state is a diagnostic object or unknown")
            require(isinstance(row["evidence"], list) and row["evidence"] and
                    all(isinstance(v, str) and v.strip() for v in row["evidence"]), "heldout evidence references required")
            numerical = row["numerical_receipt"]
            require(isinstance(numerical, dict) and set(numerical) == {"path", "sha256"}, "numerical receipt path/SHA required")
            numerical_key = (numerical["path"], numerical["sha256"])
            if numerical_key not in receipts_cache:
                numerical_document, actual_receipt = _read(*numerical_key)
                require(type(numerical_document.get("passed")) is bool, "numerical receipt needs explicit passed boolean")
                receipts_cache[numerical_key] = (numerical_document["passed"], actual_receipt)
            numerical_passed, numerical_source = receipts_cache[numerical_key]
            run_id = row["run_id"]
            available = available_by_run.get(run_id)
            if run_id not in supply_receipts:
                supply_receipts[run_id], supply_issues[run_id] = _supply(run_id, available, supply_source_files)
            require((row["clock_id"], row["clock_origin"]) == (available.clock_id, available.clock_origin),
                    "heldout reference and supply clocks/origins differ")
            records.append((row, available, source, numerical_passed, numerical_source))
    require(records, "nonempty heldout cohort required")
    require(len({r[0] for r in runs.values()}) == len(runs), "heldout processes must have distinct PIDs")
    require(len({r[1] for r in runs.values()}) == len(runs), "heldout processes must have distinct seeds")
    intervals = defaultdict(list)
    for row, *_ in records:
        intervals[row["run_id"]].append((row["begin_ns"], row["end_ns"]))
    for ordered in intervals.values():
        ordered.sort()
        require(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])),
                "reference invocations overlap within one run; model the joint graph explicitly")
    predictions = []
    for row, available, source, numerical_passed, numerical_source in records:
        predicted, issue, compared, compared_issue = None, None, None, None
        issues = list(supply_issues[row["run_id"]])
        if not numerical_passed:
            issues.append("numerical_verification_failed")
        if not issues:
            try:
                result = evaluate_available_resources(graph, fixed_parameters, available, bindings, start_ns=row["begin_ns"])
                predicted = result.duration_ns
            except ValueError as error:
                issue = str(error)
        else:
            issue = "; ".join(issues)
        if diagnostic_candidates:
            try:
                result = evaluate_available_resources(graph, fixed_parameters, available, diagnostics,
                    start_ns=row["begin_ns"], allow_observed_diagnostic=True)
                compared = result.duration_ns
            except ValueError as error:
                compared_issue = str(error)
        crossing = _crosses_supply_change(available, graph, row["begin_ns"], row["end_ns"])
        observed = row["end_ns"] - row["begin_ns"]
        predictions.append(dict(observation_id=row["observation_id"], run_id=row["run_id"], process_id=row["process_id"],
            begin_ns=row["begin_ns"], end_ns=row["end_ns"], observed_ns=observed,
            predicted_ns=predicted, model_problem=issue,
            absolute_relative_error=None if predicted is None else float(abs(Fraction(predicted)-observed)/observed),
            diagnostic_predicted_ns=compared, diagnostic_problem=compared_issue,
            diagnostic_absolute_relative_error=None if compared is None else float(abs(Fraction(compared)-observed)/observed),
            supply_crossing=crossing, state=row["state"], source=source, numerical_source=numerical_source,
            evidence_issues=issues))
    run_ids = sorted(runs)
    overall = _summary(predictions, run_ids)
    protocol = barrier["protocol"]
    parameter_issues = [{"target_parameter": c["target_parameter"], "issues": c["evidence_issues"]}
                        for c in frozen["parameter_cells"] if not c["accepted"] and c["target_parameter"] in required_targets]
    unused_parameter_issues = [{"target_parameter": c["target_parameter"], "issues": c["evidence_issues"]}
                        for c in frozen["parameter_cells"] if not c["accepted"] and c["target_parameter"] not in required_targets]
    fixed = _fixed_qualification(graph, fixed_parameters)
    qualification = []
    if len(runs) < protocol["minimum_runs"]:
        qualification.append("insufficient_independent_heldout_runs")
    if any(r["sample_count"] < protocol["minimum_samples_per_run"] for r in overall["runs"]):
        qualification.append("heldout_sample_count_gate_failed")
    if parameter_issues:
        qualification.append("profile_efficiency_evidence_failed")
    if not fixed["all_required_fixed_parameters_qualified"]:
        qualification.append("fixed_overhead_qualification_unestablished")
    if not verify_sources or not barrier["profile_sources_verified"]:
        qualification.append("profile_current_sources_not_verified")
    if any(p["evidence_issues"] for p in predictions):
        qualification.append("heldout_supply_or_numerical_evidence_failed")
    if not overall["complete_prediction_coverage"]:
        qualification.append("incomplete_prediction_coverage")
    for metric, threshold in (("process_balanced_median_absolute_relative_error", "median_relative_error_max"),
                              ("pooled_p90_absolute_relative_error", "p90_relative_error_max")):
        if overall[metric] is None or overall[metric] > protocol[threshold]:
            qualification.append(threshold + "_gate_failed")
    states = sorted({canonical(p["state"]).decode() for p in predictions})
    result = dict(schema="available-service-heldout-evaluation-1", barrier=barrier,
        parameter_artifact_sha256=frozen["artifact_sha256"], graph_sha256=barrier["graph_sha256"],
        heldout_sources=source_receipts, supply_sources=supply_receipts,
        numerical_receipts=[v[1] for v in receipts_cache.values()],
        prediction_provider="llm_infer_model.performance.evaluate_available_resources",
        conditional_on_observed_start=True, all_internal_boundaries_model_computed=True,
        scope="Whole execution wall duration conditional on independently observed external start and supplied resource history",
        primary_summary=overall, by_state=[dict(state=json.loads(state), **_summary([p for p in predictions if canonical(p["state"]).decode()==state], run_ids)) for state in states],
        by_supply_crossing={str(flag).lower(): _summary([p for p in predictions if p["supply_crossing"] is flag], run_ids) for flag in (False, True)},
        predictions=predictions, parameter_evidence_issues=parameter_issues,
        required_target_parameters=sorted(required_targets),
        unused_profile_parameter_evidence_issues=unused_parameter_issues,
        fixed_parameter_qualification=fixed, qualification_issues=qualification,
        accepted=not qualification, hardware_accuracy_qualified=not qualification,
        diagnostic_comparison=None if not diagnostic_candidates else dict(
            summary=_summary(predictions, run_ids, "diagnostic_predicted_ns"),
            candidate_parameters_from_profile_only=True, diagnostic_supply_replay_permitted=True,
            primary_unknowns_filled=False, accepted=False, hardware_accuracy_qualified=False),
        heldout_used_for_fit=False, whole_layer_duration_used_for_fit=False,
        all_observations_retained=True, external_supply_inferred_from_target=False,
        primary_target_derived_supply_allowed=False,
        diagnostic_target_derived_or_unestablished_supply_present=bool(diagnostic_candidates and any(
            "external_supply_independence_unestablished_or_target_derived" in values
            for values in supply_issues.values())),
        acceptance_scope="Declared measured graph/context and supplied external trajectory only; not arbitrary policy or hardware transfer",
        provenance_scope="Bound source bytes and declared measurement protocols, not physical remeasurement or producer authenticity proof")
    result["evaluation_sha256"] = digest(result)
    return result
