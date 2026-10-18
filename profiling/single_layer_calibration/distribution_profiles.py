"""Profile-only serial duration marginals; predictions are owned by modeling.

This optional adapter consumes an already frozen single-layer profile without
rewriting or re-signing it. Only context-trace interval observations become
empirical parameter supports. Complete invocation totals are retained solely
for diagnostic accounting and never become a model input or correction term.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from llm_infer_model.performance.serial_actions import SerialTimingProgram
from llm_infer_model.performance import serial_distributions as modeling

FORMAT = "single-layer-duration-distribution-model-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _encoded(value):
    value = Fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def _decoded(value):
    _require(isinstance(value, dict) and set(value) == {"numerator", "denominator"},
             "an exact fraction is required")
    numerator, denominator = value["numerator"], value["denominator"]
    _require(type(numerator) is int and type(denominator) is int and denominator > 0,
             "integer numerator and positive denominator required")
    return Fraction(numerator, denominator)


def _utc(value):
    _require(isinstance(value, str), "UTC process timestamp required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             "process timestamp must use UTC")
    return parsed


def _nonnegative_integer(value, label):
    _require(type(value) is int and value >= 0, f"{label} must be a known nonnegative integer")
    return value


def _mean(values):
    _require(bool(values), "nonempty observations required")
    return Fraction(sum(values), len(values))


def _variance(values):
    average = _mean(values)
    return _mean([value * value for value in values]) - average * average


def _median(values):
    values = sorted(values)
    _require(bool(values), "nonempty observations required")
    middle = len(values) // 2
    return Fraction(values[middle]) if len(values) % 2 else Fraction(values[middle - 1] + values[middle], 2)


def _verified_source(record):
    path = Path(record["source_path"])
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == record["source_sha256"],
             "context profile source bytes changed")
    _require(json.loads(raw) == record["native"], "context record differs from its raw source")


def _context_intervals(record, actions, timing_program):
    """Extract separately observed durations/gaps; no whole-total price exists."""
    native = record["native"]
    _require(record.get("split") == "profile" and record.get("mode") == "context-trace"
             and native.get("mode") == "context-trace", "only profile context observations may estimate marginals")
    _require(native.get("actions") == actions, "context action inventory differs")
    _require(native.get("components") == [], "context observations cannot mix isolated components")
    _require(record.get("numerical_passed") is True and record.get("source_verified") is True,
             "original numerical/source verification must have passed")
    iterations = native.get("iterations")
    _require(isinstance(iterations, list) and len(iterations) >= 3
             and type(native.get("samples")) is int and native["samples"] == len(iterations),
             "at least three retained context iterations required")
    totals = native.get("total_samples_ns")
    _require(isinstance(totals, list) and len(totals) == len(iterations), "context total accounting differs")
    references = [name for action in timing_program.actions
                  for name in (action.duration_parameter, action.gap_before_parameter)] + [timing_program.tail_parameter]
    _require(len(references) == len(set(references)),
             "context empirical marginals require a distinct parameter for each observed interval")
    series = {name: [] for name in references}
    for number, iteration in enumerate(iterations):
        _require(iteration.get("iteration") == number, "context iteration inventory differs")
        total = _nonnegative_integer(iteration.get("elapsed_ns"), "context elapsed ns")
        _require(total > 0 and totals[number] == total, "context total identity differs")
        intervals = iteration.get("action_intervals")
        _require(isinstance(intervals, list) and len(intervals) == len(actions), "context action intervals are incomplete")
        previous, accounting = 0, 0
        for index, (interval, action, price) in enumerate(zip(intervals, actions, timing_program.actions)):
            _require(interval.get("action_index") == index and interval.get("kind") == action["kind"],
                     "context interval index/kind differs")
            begin = _nonnegative_integer(interval.get("begin_offset_ns"), "context action begin")
            end = _nonnegative_integer(interval.get("end_offset_ns"), "context action end")
            _require(previous <= begin <= end <= total, "context intervals overlap or escape the invocation")
            duration, gap = end - begin, begin - previous
            series[price.duration_parameter].append(duration)
            series[price.gap_before_parameter].append(gap)
            accounting += duration + gap
            previous = end
        tail = total - previous
        series[timing_program.tail_parameter].append(tail)
        accounting += tail
        _require(accounting == total, "explicit interval accounting does not cover the context invocation")
    return series


def _correlation_diagnostic(process_id, series):
    arrays = list(series.values())
    _require(bool(arrays) and len({len(values) for values in arrays}) == 1,
             "diagnostics require matched retained interval counts")
    sums = [sum(values[i] for values in arrays) for i in range(len(arrays[0]))]
    variance_sum = sum((_variance(values) for values in arrays), Fraction())
    observed_variance = _variance(sums)
    mean_sum = sum((_mean(values) for values in arrays), Fraction())
    median_sum = sum((_median(values) for values in arrays), Fraction())
    observed_mean, observed_median = _mean(sums), _median(sums)
    _require(mean_sum == observed_mean, "linearity of the observed means failed accounting")
    return dict(process_id=process_id, retained_iterations=len(sums),
        sum_interval_population_variances_ns2=_encoded(variance_sum),
        observed_sum_population_variance_ns2=_encoded(observed_variance),
        twice_sum_cross_interval_covariances_ns2=_encoded(observed_variance - variance_sum),
        sum_interval_means_ns=_encoded(mean_sum), observed_sum_mean_ns=_encoded(observed_mean),
        observed_sum_mean_minus_sum_interval_means_ns=_encoded(observed_mean - mean_sum),
        sum_interval_medians_ns=_encoded(median_sum), observed_sum_median_ns=_encoded(observed_median),
        observed_sum_median_minus_sum_interval_medians_ns=_encoded(observed_median - median_sum),
        used_as_model_parameter=False,
        interpretation="descriptive covariance and median nonadditivity; not a missing-overhead attribution or fitted correction")


def _validate_distribution_artifact(frozen):
    _require(isinstance(frozen, dict), "frozen distribution artifact required")
    copied = deepcopy(frozen)
    signature = copied.pop("frozen_distribution_sha256", None)
    _require(isinstance(signature, str) and signature == _hash(copied), "frozen distribution artifact changed")
    _require(frozen.get("format_version") == FORMAT and frozen.get("heldout_used_for_estimation") is False,
             "unsupported or heldout-contaminated distribution artifact")
    _require(frozen.get("whole_iteration_totals_used_as_parameters") is False
             and frozen.get("whole_trace_resampling") is False
             and frozen.get("all_context_samples_retained") is True,
             "distribution estimation scope differs")


def predict_distribution(frozen_distribution):
    """Reconstruct explicit inputs and delegate every prediction to modeling."""
    _validate_distribution_artifact(frozen_distribution)
    program = SerialTimingProgram.from_dict(frozen_distribution["program"])
    program.validate_action_inventory(tuple(action["kind"] for action in frozen_distribution["actions"]))
    processes = modeling.distributions_from_dict(frozen_distribution["distribution_contract"])
    settings = frozen_distribution["monte_carlo_parameters"]
    return modeling.evaluate_serial_distributions(program, processes,
        trials=settings["trials"], seed=settings["seed"],
        cdf_failure_probability=_decoded(settings["cdf_failure_probability"])).to_dict()


def freeze_distributions(frozen_profile, *, trials, seed):
    """Estimate profile-only supports without changing the original artifact.

    The parent must be the original signed v1 freeze, already numerically and
    structurally verified by the single-layer calibration workflow. A later
    schema can wrap this immutable anchor plus this new model. This avoids
    self-referential signatures and never backdates the new model to an earlier
    heldout barrier. Prediction quality must be established with fresh bare runs.
    """
    _require(isinstance(frozen_profile, dict), "verified original frozen profile required")
    original_copy = deepcopy(frozen_profile)
    original_signature = original_copy.pop("frozen_sha256", None)
    _require(original_signature == _hash(original_copy), "original frozen profile changed")
    _require(frozen_profile.get("format_version") == "single-layer-calibration-frozen-v1"
             and frozen_profile.get("heldout_used") is False,
             "a profile-only original v1 anchor is required")
    actions = frozen_profile["actions"]
    timing = SerialTimingProgram.from_dict(frozen_profile["models"]["context_actions_and_gaps"]["program"])
    timing.validate_action_inventory(tuple(action["kind"] for action in actions))
    all_records = frozen_profile.get("profile_runs")
    _require(isinstance(all_records, list) and all_records and all(record.get("split") == "profile" for record in all_records),
             "original source inventory must contain only profiling runs")
    records = [record for record in all_records if record.get("mode") == "context-trace"]
    _require(len(records) >= frozen_profile["fixed_quality_gates"]["independent_processes_min"],
             "independent context process inventory is incomplete")
    ids = [record["run_id"] for record in records]
    _require(len(ids) == len(set(ids)), "context source run identities repeat")
    sources, processes, diagnostics = [], [], []
    for record in records:
        _require(record["contract_identity"] == frozen_profile["contract_identity"], "context exact program contract differs")
        _require(record["execution_scope"] == frozen_profile["execution_scope"]
                 and record["software_sha256"] == frozen_profile["software_sha256"],
                 "context execution/software contract differs")
        _require(_utc(record["started_at_utc"]) <= _utc(record["completed_at_utc"])
                 <= _utc(frozen_profile["frozen_at_utc"]), "context process completed after the original freeze")
        _verified_source(record)
        series = _context_intervals(record, actions, timing)
        marginals = {}
        for name, samples in series.items():
            counts = Counter(samples)
            support = tuple(sorted(counts))
            marginals[name] = modeling.DiscreteDuration(support, tuple(counts[value] for value in support))
            _require(marginals[name].total_count == len(samples), "profile histogram dropped observations")
        processes.append(modeling.ProcessDurationDistribution(record["run_id"], marginals))
        diagnostics.append(_correlation_diagnostic(record["run_id"], series))
        sources.append(dict(run_id=record["run_id"], process_id=record["process_id"],
            seed=record["seed"], source_path=record["source_path"], source_sha256=record["source_sha256"],
            started_at_utc=record["started_at_utc"], completed_at_utc=record["completed_at_utc"],
            retained_samples_per_interval=record["native"]["samples"],
            all_samples_retained=True, process_weight=_encoded(Fraction(1, len(records)))))
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            _require(first["source_path"] != second["source_path"]
                     and first["source_sha256"] != second["source_sha256"], "context raw process evidence reused")
            if first["process_id"] == second["process_id"]:
                overlap = max(_utc(first["started_at_utc"]), _utc(second["started_at_utc"])) <= min(
                    _utc(first["completed_at_utc"]), _utc(second["completed_at_utc"]))
                _require(not overlap, "same native PID has overlapping process lifetimes")
    contract = modeling.distributions_to_dict(tuple(processes))
    settings = dict(trials=trials, seed=seed, cdf_failure_probability=_encoded(Fraction(1, 100)))
    # The three diagnostic variance terms are descriptive, not additional
    # stochastic-model parameters. Between-process mean variation is retained
    # through the model's single equal-weight process choice per invocation.
    process_means = [_decoded(row["observed_sum_mean_ns"]) for row in diagnostics]
    between_mean_variance = _variance(process_means)
    conditional_independent_variance = _mean([
        _decoded(row["sum_interval_population_variances_ns2"]) for row in diagnostics])
    conditional_observed_variance = _mean([
        _decoded(row["observed_sum_population_variance_ns2"]) for row in diagnostics])
    frozen = dict(format_version=FORMAT, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        original_frozen_profile_sha256=original_signature,
        original_frozen_profile_time_utc=frozen_profile["frozen_at_utc"],
        original_pilot_artifact_modified=False,
        contract_identity=deepcopy(frozen_profile["contract_identity"]),
        actions=deepcopy(actions), execution_scope=deepcopy(frozen_profile["execution_scope"]),
        original_profile_software_sha256=frozen_profile["software_sha256"],
        fixed_quality_gates=deepcopy(frozen_profile["fixed_quality_gates"]),
        original_source_compute_copy_quality_passed=frozen_profile["models"]["context_actions_and_gaps"]["source_compute_copy_quality_passed"],
        original_instrumentation_control=deepcopy(frozen_profile["instrumentation_control"]),
        program=timing.to_dict(), distribution_contract=contract, monte_carlo_parameters=settings,
        profile_context_sources=sources, profile_context_sources_sha256=_hash(sources),
        profile_seed_inventory=sorted({record["seed"] for record in all_records}),
        original_profile_run_ids=[record["run_id"] for record in all_records],
        original_profile_source_paths=[record["source_path"] for record in all_records],
        original_profile_source_sha256s=[record["source_sha256"] for record in all_records],
        profile_processes_equal_weight=True, all_context_samples_retained=True,
        heldout_used_for_estimation=False, whole_iteration_totals_used_as_parameters=False,
        whole_trace_resampling=False,
        covariance_diagnostics=dict(per_process=diagnostics,
            equal_process_mean_conditional_independence_variance_ns2=_encoded(conditional_independent_variance),
            equal_process_mean_observed_conditional_variance_ns2=_encoded(conditional_observed_variance),
            variance_of_process_sum_means_ns2=_encoded(between_mean_variance),
            hierarchical_independent_sum_variance_ns2=_encoded(conditional_independent_variance + between_mean_variance),
            observed_hierarchical_sum_variance_ns2=_encoded(conditional_observed_variance + between_mean_variance),
            used_as_model_parameter=False),
        applicability="exact original compiled serial strategy/geometry and declared execution context; no unseen-strategy extrapolation",
        current_accuracy_claimed=False,
        provenance_scope="original verified profile anchor plus rehashed context raw observations; original binary-loader and measurement limits remain")
    # Seal inputs before using the same reconstruction path used by subsequent
    # predictions. Re-sealing a new artifact does not re-sign the original pilot.
    frozen["frozen_distribution_sha256"] = _hash(frozen)
    prediction = predict_distribution(frozen)
    frozen.pop("frozen_distribution_sha256")
    frozen["prediction"] = prediction
    frozen["frozen_distribution_sha256"] = _hash(frozen)
    _require(frozen_profile == {**original_copy, "frozen_sha256": original_signature},
             "original frozen profile was mutated")
    return frozen


def _cv(values):
    average = _mean(values)
    return 0.0 if average == 0 else math.sqrt(float(_variance(values))) / float(average)


def evaluate_distribution(frozen_distribution, base_report):
    """Evaluate the frozen diagnostic candidate without changing primary gates.

    The base report must be the source-verified v1 evaluation of the exact
    original profile anchor. Fresh bare observations must also postdate this
    new distribution freeze; an earlier pilot heldout cannot validate a model
    introduced afterward. Mean error and interval coverage are descriptive
    supplements, never replacements for the original median/CV/control gates.
    """
    from single_layer_calibration.analysis import GATES

    prediction = predict_distribution(frozen_distribution)
    _require(prediction == frozen_distribution.get("prediction"),
             "distribution implementation or cached prediction changed after freeze")
    _require(isinstance(base_report, dict)
             and base_report.get("format_version") == "single-layer-calibration-evaluation-v1",
             "a verified original single-layer evaluation is required")
    _require(base_report.get("raw_sources_verified") is True
             and base_report.get("heldout_used_for_fitting") is False
             and base_report.get("parameter_frozen_before_heldout") is True
             and base_report.get("all_samples_retained") is True,
             "base heldout evidence is unverified, contaminated or incomplete")
    _require(base_report.get("frozen_sha256") == frozen_distribution["original_frozen_profile_sha256"]
             and base_report.get("contract_identity") == frozen_distribution["contract_identity"],
             "distribution and base evaluation have different profile/program anchors")
    _require(base_report.get("fixed_quality_gates") == GATES
             and frozen_distribution["fixed_quality_gates"] == GATES,
             "original acceptance gates changed")
    _require(base_report.get("primary_model") == "context_actions_and_gaps", "original primary model differs")
    base_model = base_report["models"]["context_actions_and_gaps"]
    _require(base_report.get("accepted") == base_model["accepted"], "base primary acceptance is inconsistent")
    source_quality = frozen_distribution["original_source_compute_copy_quality_passed"]
    control = frozen_distribution["original_instrumentation_control"]["passed"]
    _require(type(source_quality) is bool and type(control) is bool
             and base_model["source_compute_copy_quality_passed"] == source_quality
             and base_model["instrumentation_control_passed"] == control,
             "original source-quality or instrumentation gate differs")
    records = base_report.get("heldout_runs")
    _require(isinstance(records, list) and len(records) >= GATES["independent_processes_min"],
             "fresh independent heldout process inventory is incomplete")
    run_ids, paths, hashes, process_rows, medians, means, pooled = set(), set(), set(), [], [], [], []
    distribution_freeze = _utc(frozen_distribution["frozen_at_utc"])
    lower, upper = prediction["p10_ns"], prediction["p90_ns"]
    for record in records:
        native = record["native"]
        _require(record.get("split") == "heldout" and record.get("mode") == "bare-replay"
                 and native.get("mode") == "bare-replay", "distribution validation requires fresh bare heldout runs")
        _require(record.get("source_verified") is True and record.get("numerical_passed") is True,
                 "heldout source/numerical evidence is unverified")
        _require(type(record.get("process_id")) is int and record["process_id"] > 0
                 and record["process_id"] == native.get("process_id")
                 and record.get("program_sha256") == native.get("program_sha256")
                 and type(record.get("seed")) is int,
                 "heldout record/native process or program identity differs")
        _require(record["contract_identity"] == frozen_distribution["contract_identity"]
                 and record["execution_scope"] == frozen_distribution["execution_scope"]
                 and record["software_sha256"] == frozen_distribution["original_profile_software_sha256"],
                 "heldout exact execution/program/software contract differs")
        _require(record["run_id"] not in run_ids and record["run_id"] not in frozen_distribution["original_profile_run_ids"]
                 and record["seed"] not in frozen_distribution["profile_seed_inventory"],
                 "heldout run or numerical seed was reused")
        _require(record["source_path"] not in paths and record["source_path"] not in frozen_distribution["original_profile_source_paths"]
                 and record["source_sha256"] not in hashes and record["source_sha256"] not in frozen_distribution["original_profile_source_sha256s"],
                 "heldout raw evidence was reused")
        _require(distribution_freeze <= _utc(record["started_at_utc"]) <= _utc(record["completed_at_utc"]),
                 "heldout began before the new distribution freeze")
        _require(native.get("actions") == frozen_distribution["actions"] and native.get("components") == [],
                 "heldout action inventory or observation mode differs")
        values = native.get("total_samples_ns")
        _require(isinstance(values, list) and len(values) >= 3 and native.get("samples") == len(values),
                 "heldout complete invocation samples are missing")
        _require(all(type(value) is int and value > 0 for value in values), "heldout durations must be positive known ns")
        iterations = native.get("iterations")
        _require(isinstance(iterations, list) and len(iterations) == len(values), "heldout invocation intervals are missing")
        for index, (iteration, value) in enumerate(zip(iterations, values)):
            _require(iteration.get("iteration") == index and iteration.get("elapsed_ns") == value
                     and iteration.get("action_intervals") == [], "heldout is instrumented or its invocation identity differs")
        _verified_source(record)
        run_ids.add(record["run_id"])
        paths.add(record["source_path"])
        hashes.add(record["source_sha256"])
        observed_median, observed_mean = _median(values), _mean(values)
        medians.append(observed_median)
        means.append(observed_mean)
        pooled.extend(values)
        process_rows.append(dict(run_id=record["run_id"], process_id=record["process_id"], samples=len(values),
            median_ns=_encoded(observed_median), mean_ns=_encoded(observed_mean), cv=_cv(values),
            p50_absolute_relative_error=float(abs(Fraction(prediction["p50_ns"]) - observed_median) / observed_median),
            expected_mean_absolute_relative_error=float(abs(_decoded(prediction["expected_duration_ns"]) - observed_mean) / observed_mean),
            p10_p90_empirical_coverage=float(Fraction(sum(lower <= value <= upper for value in values), len(values))),
            source_path=record["source_path"], source_sha256=record["source_sha256"]))
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            if first["process_id"] == second["process_id"]:
                overlap = max(_utc(first["started_at_utc"]), _utc(second["started_at_utc"])) <= min(
                    _utc(first["completed_at_utc"]), _utc(second["completed_at_utc"]))
                _require(not overlap, "heldout native PID lifetimes are not independent")
    observed_median = _median(pooled)
    observed_equal_process_mean = _mean(means)
    within_cvs = [row["cv"] for row in process_rows]
    between_cv = _cv(medians)
    baseline_stats = base_report["heldout_statistics"]
    _require([_decoded(value) for value in baseline_stats["process_medians_ns"]] == medians
             and _decoded(baseline_stats["pooled_median_ns"]) == observed_median
             and _decoded(baseline_stats["median_of_process_medians_ns"]) == _median(medians),
             "base heldout median statistics differ from rehashed raw observations")
    _require(len(baseline_stats["within_process_cv"]) == len(within_cvs)
             and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                     for a, b in zip(baseline_stats["within_process_cv"], within_cvs))
             and math.isclose(baseline_stats["between_process_median_cv"], between_cv,
                              rel_tol=1e-12, abs_tol=1e-15),
             "base heldout stability statistics differ from raw observations")
    within = all(value <= GATES["within_process_cv_max"] for value in within_cvs)
    between = between_cv <= GATES["between_process_median_cv_max"]
    _require(base_model["heldout_within_process_cv_passed"] == within
             and base_model["heldout_between_process_median_cv_passed"] == between,
             "original heldout CV gates differ")
    primary_duration = _nonnegative_integer(base_model["predicted_ns"], "original primary prediction")
    primary_error = abs(Fraction(primary_duration) - observed_median) / observed_median
    _require(math.isclose(base_model["absolute_relative_error"], float(primary_error), rel_tol=1e-12, abs_tol=1e-15)
             and base_model["accepted"] == (source_quality and within and between and control
                 and primary_error <= GATES["pooled_median_absolute_relative_error_max"]),
             "original primary median-error acceptance differs from its unchanged gates")
    signed = (Fraction(prediction["p50_ns"]) - observed_median) / observed_median
    mean_signed = (_decoded(prediction["expected_duration_ns"]) - observed_equal_process_mean) / observed_equal_process_mean
    coverage = _mean([Fraction(sum(lower <= value <= upper for value in row["native"]["total_samples_ns"]),
                               len(row["native"]["total_samples_ns"])) for row in records])
    candidate_passed = source_quality and within and between and control and abs(signed) <= GATES["pooled_median_absolute_relative_error_max"]
    return dict(format_version="single-layer-distribution-evaluation-v1",
        frozen_distribution_sha256=frozen_distribution["frozen_distribution_sha256"],
        original_frozen_profile_sha256=frozen_distribution["original_frozen_profile_sha256"],
        prediction=prediction, observed_heldout_pooled_median_ns=_encoded(observed_median),
        observed_heldout_equal_process_mean_ns=_encoded(observed_equal_process_mean),
        p50_signed_relative_error=float(signed), p50_absolute_relative_error=float(abs(signed)),
        expected_mean_signed_relative_error=float(mean_signed), expected_mean_absolute_relative_error=float(abs(mean_signed)),
        p10_p90_empirical_coverage=float(coverage), coverage_weight="equal heldout process weight",
        coverage_scope="descriptive sample coverage; correlated invocations and finite processes do not establish calibrated probability intervals",
        heldout_processes=process_rows, original_source_compute_copy_quality_passed=source_quality,
        heldout_within_process_cv_passed=within, heldout_between_process_median_cv_passed=between,
        instrumentation_control_passed=control, fixed_quality_gates=dict(GATES),
        candidate_original_gates_passed=candidate_passed,
        candidate_role="predeclared diagnostic conditional-independence p50 candidate; m3 remains primary",
        primary_model_changed=False, primary_acceptance_unchanged=base_report["accepted"],
        mean_or_coverage_used_to_change_acceptance=False,
        fresh_heldout_after_distribution_freeze=True, raw_sources_verified=True,
        heldout_used_for_fitting=False, all_samples_retained=True,
        residual_cause="unresolved", native_memory_capacity_accuracy_claimed=False)
