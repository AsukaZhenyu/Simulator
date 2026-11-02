"""Profile-only physical parameter estimates; no performance prediction formula."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics
from typing import Iterable

from .observations import ParameterObservation, canonical, digest, exact, rational, utc

SCHEMA = "frozen-physical-parameters-1"


def median(values: Iterable[Fraction]) -> Fraction | None:
    ordered = sorted(values)
    if not ordered:
        return None
    i = len(ordered) // 2
    return ordered[i] if len(ordered) % 2 else (ordered[i - 1] + ordered[i]) / 2


def cv(values: Iterable[Fraction]) -> float | None:
    values = list(values)
    if not values:
        return None
    average = sum(values) / len(values)
    if average == 0:
        return 0.0 if all(value == 0 for value in values) else None
    return statistics.pstdev([float(value) for value in values]) / float(average)


def validate_run_intervals(observations: Iterable[ParameterObservation]) -> None:
    """PID reuse is independent only when the recorded run lifetimes do not overlap."""
    runs = defaultdict(list)
    for row in observations:
        runs[row.run_id].append(row)
    intervals = []
    for run_id, rows in runs.items():
        intervals.append((rows[0].process_id, min(utc(row.started_at_utc) for row in rows),
                          max(utc(row.ended_at_utc) for row in rows), run_id))
    for i, (pid, start, end, run_id) in enumerate(intervals):
        for other_pid, other_start, other_end, other_id in intervals[i + 1:]:
            if pid == other_pid and (max(start, other_start) < min(end, other_end)
                                     or start == other_start):
                raise ValueError(f"same PID has overlapping run lifetimes: {run_id}, {other_id}; not independent processes")


def estimate_parameters(observations: Iterable[ParameterObservation], *, context: dict,
                        frozen_at_utc: str, minimum_runs: int = 3,
                        within_process_cv_max: float = 0.15,
                        between_process_cv_max: float = 0.20) -> dict:
    """Freeze median-of-process-medians, retaining every run and rejected sample.

    Different exact contexts cannot be silently combined. Samples within each
    run get equal weight; runs get equal weight regardless of sample count.
    Unstable/unknown/unverified evidence produces an unknown model parameter,
    never a fallback coefficient or a total-action throughput conversion.
    """
    from llm_infer_model.performance import Parameter, ParameterBundle

    frozen = utc(frozen_at_utc)
    if type(minimum_runs) is not int or minimum_runs < 3:
        raise ValueError("at least three independent runs are required")
    for threshold in (within_process_cv_max, between_process_cv_max):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold < float("inf"):
            raise ValueError("CV thresholds require finite nonnegative numbers")
    if not isinstance(context, dict) or not context:
        raise ValueError("context must be a nonempty exact context dictionary")
    context_json = canonical(context).decode()
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one profile observation is required")
    ids, run_metadata, process_seed_runs = set(), {}, {}
    groups = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, ParameterObservation):
            raise ValueError("use validated ParameterObservation instances")
        if observation.split != "profile":
            raise ValueError("heldout/diagnostic observations cannot estimate parameters")
        if observation.context_json != context_json:
            raise ValueError("exact context mismatch; select or measure the requested context explicitly")
        if utc(observation.ended_at_utc) > frozen:
            raise ValueError("profile evidence ended after the freeze timestamp")
        if observation.observation_id in ids:
            raise ValueError("duplicate observation_id")
        ids.add(observation.observation_id)
        metadata = (observation.process_id, observation.seed)
        if observation.run_id in run_metadata and run_metadata[observation.run_id] != metadata:
            raise ValueError("one run_id cannot claim multiple process IDs or seeds")
        run_metadata[observation.run_id] = metadata
        if metadata in process_seed_runs and process_seed_runs[metadata] != observation.run_id:
            raise ValueError("renamed run IDs with the same process_id and seed are not independent runs")
        process_seed_runs[metadata] = observation.run_id
        groups[observation.parameter_name].append(observation)
    validate_run_intervals(observations)
    cells, parameters = [], []
    for name, rows in sorted(groups.items()):
        signatures = {(row.kind, row.unit, row.timing_scope) for row in rows}
        if len(signatures) != 1:
            raise ValueError("one parameter cannot mix units, kinds, or timing scopes")
        kind, unit, timing_scope = next(iter(signatures))
        by_run = defaultdict(list)
        for row in rows:
            by_run[row.run_id].append(row)
        processes, issues, process_medians = [], [], []
        if len(by_run) < minimum_runs:
            issues.append("insufficient_independent_runs")
        for run_id, samples in sorted(by_run.items()):
            values = [row.value for row in samples if row.value is not None]
            run_issues = []
            if len(values) != len(samples):
                run_issues.append("unknown_observation_value")
            if not all(row.source_verified for row in samples):
                run_issues.append("source_bytes_not_verified")
            if kind != "capacity" and not all(row.numerical_passed is True for row in samples):
                run_issues.append("numerical_verification_not_passed")
            run_median, run_cv = median(values), cv(values)
            if run_cv is None or run_cv > within_process_cv_max:
                run_issues.append("within_process_cv_gate_failed")
            if run_median is not None:
                process_medians.append(run_median)
            processes.append(dict(run_id=run_id, process_id=samples[0].process_id,
                seed=samples[0].seed, sample_count=len(samples),
                observation_ids=[row.observation_id for row in samples],
                median_exact=rational(run_median), cv=run_cv, evidence_issues=run_issues))
        if any(process["evidence_issues"] for process in processes):
            issues.append("one_or_more_runs_failed_evidence_gates")
        between_cv = cv(process_medians)
        if between_cv is None or between_cv > between_process_cv_max:
            issues.append("between_process_cv_gate_failed")
        candidate = median(process_medians)
        reference_only = kind == "reference_duration"
        if reference_only:
            issues.append("reference_only_not_a_physical_model_parameter")
        accepted = not issues
        cell = dict(parameter_name=name, kind=kind, unit=unit, timing_scope=timing_scope,
                    context=json.loads(context_json), processes=processes,
                    process_balanced_median_exact=rational(candidate),
                    between_process_cv=between_cv, accepted=accepted,
                    evidence_issues=issues, all_samples_retained=True,
                    aggregation="exact median of per-process exact medians",
                    outliers_removed=0, reference_only=reference_only)
        cells.append(cell)
        if not reference_only:
            evidence = tuple(sorted({f"sha256:{row.source_sha256}:{row.observation_id}" for row in rows}))
            parameters.append(Parameter(name=name, value=candidate if accepted else None,
                unit=unit, origin="profiled", evidence=evidence,
                scope=canonical({"context": context, "timing_scope": timing_scope}).decode()))
    bundle = ParameterBundle(parameters=tuple(parameters))
    # The model's own serializer/validator is the only bundle contract.
    bundle_dict = bundle.to_dict()
    ParameterBundle.from_dict(bundle_dict)
    raw = [observation.to_dict() for observation in observations]
    result = dict(schema=SCHEMA, frozen_at_utc=frozen_at_utc, context=json.loads(context_json),
        bundle=bundle_dict, parameter_cells=cells,
        profile_observations=raw, profile_observations_sha256=digest(raw),
        profile_run_ids=sorted(run_metadata),
        profile_seeds=sorted({row.seed for row in observations}),
        profile_last_ended_at_utc=max(observations, key=lambda row: utc(row.ended_at_utc)).ended_at_utc,
        heldout_used_for_estimation=False, performance_predictions_computed_here=False,
        source_verification_scope="Bound file bytes at estimation; collector-provided scalar extraction is not independently recomputed here.",
        gates=dict(minimum_independent_runs=minimum_runs,
                   within_process_cv_max=within_process_cv_max,
                   between_process_cv_max=between_process_cv_max),
        explanation="Profiling estimates physical inputs; modeling computes execution time.")
    result["artifact_sha256"] = digest(result)
    return result


def load_frozen_parameters(document: dict | str | Path, *, verify_sources: bool = False,
                           source_base: str | Path | None = None):
    """Reconstruct receipts; optionally rehash current raw evidence as well.

    Digest integrity is not external authenticity. A source_verified receipt is
    historical; callers must request verify_sources to check current file bytes.
    """
    from llm_infer_model.performance import ParameterBundle
    if isinstance(document, (str, Path)):
        artifact_path = Path(document).resolve()
        source_base = artifact_path.parent if source_base is None else source_base
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    document = json.loads(canonical(document))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unsupported frozen parameter artifact")
    expected = document.pop("artifact_sha256", None)
    if expected != digest(document):
        raise ValueError("frozen parameter artifact digest mismatch")
    document["artifact_sha256"] = expected
    if type(verify_sources) is not bool:
        raise ValueError("verify_sources must be boolean")
    if verify_sources:
        import hashlib
        for row in document["profile_observations"]:
            source = Path(row["source_path"])
            if not source.is_absolute():
                if source_base is None:
                    raise ValueError("relative raw sources require source_base for current-byte verification")
                source = Path(source_base) / source
            try:
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f"cannot reverify frozen source: {source}") from exc
            if actual != row["source_sha256"]:
                raise ValueError(f"frozen source bytes changed: {source}")
    if document.get("heldout_used_for_estimation") is not False or document.get("performance_predictions_computed_here") is not False:
        raise ValueError("parameter artifact violates profiling/modeling boundary")
    if digest(document["profile_observations"]) != document["profile_observations_sha256"]:
        raise ValueError("profile observations digest mismatch")
    # Re-estimate all exact statistics and acceptance decisions, not just hashes.
    rows = [ParameterObservation.from_dict(row, source_verified=row.get("source_verified") is True)
            for row in document["profile_observations"]]
    gates = document["gates"]
    reconstructed = estimate_parameters(rows, context=document["context"],
        frozen_at_utc=document["frozen_at_utc"], minimum_runs=gates["minimum_independent_runs"],
        within_process_cv_max=gates["within_process_cv_max"],
        between_process_cv_max=gates["between_process_cv_max"])
    if canonical(reconstructed) != canonical(document):
        raise ValueError("frozen parameter artifact does not reconstruct from its profile evidence")
    return ParameterBundle.from_dict(document["bundle"])
