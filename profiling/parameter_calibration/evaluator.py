"""Independent heldout validation of model-owned predictions.

Only observations/statistics and residuals are calculated in profiling. All
execution costs, progress, dependency waits and capacity checks are computed by
llm_infer_model.performance.evaluate, including when parameters are unknown.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Iterable

from llm_infer_model import performance as modeling

from .observations import ParameterObservation, canonical, digest, rational, utc
from .parameters import cv, load_frozen_parameters, median, validate_run_intervals


def evaluate_calibration(graph: modeling.Graph, frozen: dict,
                         heldout: Iterable[ParameterObservation], *,
                         timeline: modeling.ResourceTimeline | None = None,
                         task_id: str | None = None, verify_sources: bool = True,
                         source_base: str | Path | None = None,
                         reference_timing_scope: str | None = None,
                         relative_error_max: float = 0.20) -> dict:
    """Report parameter evidence and model residuals as distinct diagnostics.

    No residual is automatically attributed to a missing mechanism. Even a
    large total residual can come from measurement boundary, input parameter,
    or composition errors; the cause remains unresolved without a control.
    """
    bundle = load_frozen_parameters(frozen, verify_sources=verify_sources, source_base=source_base)
    if isinstance(relative_error_max, bool) or not isinstance(relative_error_max, (int, float)) or not 0 <= relative_error_max < float("inf"):
        raise ValueError("relative_error_max must be finite and nonnegative")
    heldout = tuple(heldout)
    if not heldout:
        raise ValueError("heldout observations are required")
    required_scope = reference_timing_scope or ("whole_iteration_completed" if task_id is None else "whole_action_completed")
    if required_scope not in {"whole_iteration_completed", "whole_action_completed"}:
        raise ValueError("reference_timing_scope must identify a completed action or iteration")
    if task_id is not None and task_id not in {task.id for task in graph.tasks}:
        raise ValueError("task_id is absent from the model graph")
    graph_sha = digest(graph.to_dict())
    bound_graph = frozen["context"].get("graph_sha256")
    graph_applicability_verified = bound_graph == graph_sha
    if bound_graph is not None and not graph_applicability_verified:
        raise ValueError("requested graph does not match the context bound before parameter freeze")
    bound_scope = frozen["context"].get("reference_timing_scope")
    reference_boundary_verified = bound_scope == required_scope
    if bound_scope is not None and not reference_boundary_verified:
        raise ValueError("reference timing scope disagrees with frozen context")
    timeline_sha = None if timeline is None else digest(timeline.to_dict())
    bound_timeline = frozen["context"].get("timeline_sha256")
    timeline_applicability_verified = bound_timeline == timeline_sha
    if bound_timeline is not None and not timeline_applicability_verified:
        raise ValueError("resource timeline disagrees with frozen context")
    applicability_issues = []
    if not graph_applicability_verified:
        applicability_issues.append("graph_not_bound_in_profile_context")
    if not reference_boundary_verified:
        applicability_issues.append("reference_timing_scope_not_bound_in_profile_context")
    if not timeline_applicability_verified:
        applicability_issues.append("timeline_not_bound_in_profile_context")
    run_metadata, identity_runs, by_run, ids = {}, {}, defaultdict(list), set()
    for row in heldout:
        if not isinstance(row, ParameterObservation) or row.kind != "reference_duration" or row.timing_scope != required_scope:
            raise ValueError("heldout rows require a matching completed graph/action timing scope")
        if row.split != "heldout":
            raise ValueError("calibration evaluation is heldout-only")
        if row.context != frozen["context"]:
            raise ValueError("heldout exact context does not match the parameter context")
        if row.observation_id in ids:
            raise ValueError("duplicate heldout observation_id")
        ids.add(row.observation_id)
        if row.run_id in frozen["profile_run_ids"] or row.seed in frozen["profile_seeds"]:
            raise ValueError("heldout run IDs and seeds must be disjoint from profile evidence")
        if utc(row.started_at_utc) < utc(frozen["frozen_at_utc"]):
            raise ValueError("heldout collection began before parameter freeze")
        identity = (row.process_id, row.seed)
        if row.run_id in run_metadata and run_metadata[row.run_id] != identity:
            raise ValueError("inconsistent heldout process identity")
        if identity in identity_runs and identity_runs[identity] != row.run_id:
            raise ValueError("renamed heldout runs are not independent")
        run_metadata[row.run_id], identity_runs[identity] = identity, row.run_id
        by_run[row.run_id].append(row)
    validate_run_intervals(heldout)
    # Hash bound heldout raw sources too. Never accept JSON's source_verified
    # assertion as a substitute for the evaluator's current-byte recheck.
    if verify_sources:
        for row in heldout:
            path = Path(row.source_path)
            if not path.is_absolute():
                if source_base is None:
                    raise ValueError("relative heldout source requires source_base")
                path = Path(source_base) / path
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError("cannot reverify heldout raw source") from exc
            if actual != row.source_sha256:
                raise ValueError("heldout raw source hash mismatch")
    prediction, model_problem = None, None
    try:
        prediction = modeling.evaluate(graph, bundle, timeline=timeline)
    except modeling.InvalidModel as exc:
        model_problem = dict(type=type(exc).__name__, message=str(exc))
    predicted_ns = None if prediction is None else prediction.duration_ns
    if prediction is not None and task_id is not None:
        predicted_ns = next(task.duration_ns for task in prediction.tasks if task.task_id == task_id)
    runs, process_medians, measurement_issues = [], [], []
    gates = frozen["gates"]
    if len(by_run) < gates["minimum_independent_runs"]:
        measurement_issues.append("insufficient_independent_heldout_runs")
    for run_id, rows in sorted(by_run.items()):
        values = [row.value for row in rows if row.value is not None]
        value, spread, issues = median(values), cv(values), []
        if len(values) != len(rows):
            issues.append("unknown_observed_duration")
        if not all(row.numerical_passed is True for row in rows):
            issues.append("numerical_verification_not_passed")
        if not all(row.source_verified for row in rows) and not verify_sources:
            issues.append("heldout_source_bytes_not_verified")
        if spread is None or spread > gates["within_process_cv_max"]:
            issues.append("within_process_cv_gate_failed")
        if value is not None:
            process_medians.append(value)
        residual = None if predicted_ns is None or value is None else value - predicted_ns
        error = None if residual is None else abs(residual) / value
        runs.append(dict(run_id=run_id, process_id=rows[0].process_id, seed=rows[0].seed,
            sample_count=len(rows), median_observed_ns_exact=rational(value), cv=spread,
            evidence_issues=issues, residual_observed_minus_model_ns_exact=rational(residual),
            relative_error=None if error is None else float(error),
            observation_ids=[row.observation_id for row in rows]))
    between_cv, observed = cv(process_medians), median(process_medians)
    if between_cv is None or between_cv > gates["between_process_cv_max"]:
        measurement_issues.append("between_process_cv_gate_failed")
    if any(run["evidence_issues"] for run in runs):
        measurement_issues.append("one_or_more_heldout_runs_failed_evidence_gates")
    residual = None if predicted_ns is None or observed is None else observed - predicted_ns
    error = None if residual is None else abs(residual) / observed
    parameter_issues = [dict(parameter_name=cell["parameter_name"], issues=cell["evidence_issues"])
                        for cell in frozen["parameter_cells"] if cell["evidence_issues"]]
    model_root = Path(modeling.__file__).resolve().parent
    model_sources = {path.relative_to(model_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(model_root.rglob("*.py"))}
    request = dict(bundle=bundle.to_dict(), graph=graph.to_dict(),
                   timeline=None if timeline is None else timeline.to_dict())
    return dict(schema="physical-model-calibration-evaluation-1",
        parameter_artifact_sha256=frozen["artifact_sha256"],
        graph_sha256=graph_sha, graph=graph.to_dict(),
        graph_applicability_verified=graph_applicability_verified,
        reference_boundary_verified=reference_boundary_verified,
        timeline_applicability_verified=timeline_applicability_verified,
        applicability_issues=applicability_issues,
        parameter_evidence_issues=parameter_issues,
        measurement_evidence_issues=measurement_issues,
        current_raw_sources_verified=verify_sources,
        prediction_provider="llm_infer_model.performance.evaluate",
        model_source_files_sha256=model_sources,
        model_implementation_sha256=digest(model_sources),
        prediction_request_sha256=digest(request),
        freeze_chronology_scope="Declared UTC metadata checked; no independent collector barrier is established by these scalar rows.",
        prediction=None if prediction is None else prediction.to_dict(), model_problem=model_problem,
        comparison_scope="graph" if task_id is None else "task", task_id=task_id,
        reference_timing_scope=required_scope,
        predicted_ns=predicted_ns, observed_process_balanced_median_ns_exact=rational(observed),
        composition_residual_observed_minus_model_ns_exact=rational(residual),
        relative_error=None if error is None else float(error),
        worst_run_relative_error=max((run["relative_error"] for run in runs
                                      if run["relative_error"] is not None), default=None),
        residual_cause="unresolved", mechanism_missing_proven=False,
        runs=runs, between_process_cv=between_cv,
        heldout_observations=[row.to_dict() for row in heldout],
        heldout_used_for_estimation=False, outliers_removed=0,
        accepted=bool(not parameter_issues and not measurement_issues and not applicability_issues and error is not None
                      and error <= relative_error_max),
        acceptance_relative_error_max=relative_error_max)
