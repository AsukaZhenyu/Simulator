"""Create a compact portable aggregate of an already evaluated v2 campaign.

Standard library only. This tool verifies saved artifact signatures and complete
case inventories, then summarizes recorded values. It never predicts latency,
fits parameters, runs a simulator, launches hardware, or repeats numerical/raw
measurement verification. The output is a portable aggregate, not raw evidence.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

MODELS = ("isolated_components", "context_components", "context_actions_and_gaps")
GATES = {"within_process_cv_max": .15, "between_process_median_cv_max": .20,
         "pooled_median_absolute_relative_error_max": .20, "independent_processes_min": 3}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def signed(value, key, label):
    require(isinstance(value, dict) and value.get(key) == digest({name: item for name, item in value.items() if name != key}),
            label + " signature differs")


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def ratio(value):
    require(isinstance(value, dict) and set(value) == {"numerator", "denominator"} and
            type(value["numerator"]) is int and type(value["denominator"]) is int and value["denominator"] > 0,
            "invalid exact recorded fraction")
    return Fraction(value["numerator"], value["denominator"])


def milliseconds(value):
    return float((ratio(value) if isinstance(value, dict) else Fraction(value)) / 10**6)


def finite(value, label):
    require(type(value) in {int, float} and math.isfinite(value), label + " must be finite")
    return value


def model_row(value):
    require(type(value["accepted"]) is bool, "model accepted flag must be Boolean")
    return {"prediction_ms": milliseconds(value["predicted_ns"]),
            "absolute_relative_error": finite(value["absolute_relative_error"], "model error"),
            "signed_relative_error": finite(value["signed_relative_error"], "model signed error"),
            "accepted": value["accepted"],
            "per_process_absolute_relative_errors": value["per_process_absolute_relative_errors"],
            "gates": {name: value[name] for name in ("source_compute_copy_quality_passed",
                "heldout_within_process_cv_passed", "heldout_between_process_median_cv_passed", "instrumentation_control_passed")},
            "residual_cause": value.get("residual_cause", "unresolved"),
            "gap_quality_claimed": value["gap_quality_claimed"]}


def profile_summary(base, distribution):
    actions, stats = base["actions"], base["action_statistics"]
    failures = {}
    for context in ("isolated", "context"):
        selected = {index: row for index, row in stats[context].items()
                    if actions[int(index)]["kind"] in {"COMPUTE", "COPY"}}
        failures[context] = {"compute_copy_actions": len(selected),
            "actions_with_any_within_process_cv_failure": sum(not row["within_process_cv_passed"] for row in selected.values()),
            "actions_with_between_process_median_cv_failure": sum(not row["between_process_median_cv_passed"] for row in selected.values())}
    context = {index: row for index, row in stats["context"].items()
               if actions[int(index)]["kind"] in {"COMPUTE", "COPY"}}
    dominant = []
    for index, row in sorted(context.items(), key=lambda item: ratio(item[1]["value_ns"]), reverse=True)[:3]:
        observed, isolated = ratio(row["value_ns"]), ratio(stats["isolated"][index]["value_ns"])
        dominant.append({"action_index": int(index), "action": actions[int(index)],
            "context_median_ms": milliseconds(observed), "isolated_median_ms": milliseconds(isolated),
            "context_to_isolated_ratio": None if isolated == 0 else float(observed / isolated),
            "within_process_cv": [process["cv"] for process in row["processes"]],
            "between_process_median_cv": row["between_process_median_cv"],
            "process_medians_ms": [milliseconds(process["median_ns"]) for process in row["processes"]],
            "within_process_cv_passed": row["within_process_cv_passed"],
            "between_process_median_cv_passed": row["between_process_median_cv_passed"]})
    control = base["instrumentation_control"]
    context_median = ratio(control["context"]["median_of_process_medians_ns"])
    recorded_m3 = base["models"]["context_actions_and_gaps"]["prediction"]["duration_ns"]
    nonadditivity = {"context_total_median_ms": milliseconds(context_median),
        "recorded_sum_interval_medians_prediction_ms": milliseconds(recorded_m3),
        "recorded_prediction_minus_context_median_ms": milliseconds(Fraction(recorded_m3) - context_median),
        "relative_difference": None if context_median == 0 else float((recorded_m3 - context_median) / context_median),
        "interpretation": "statistical_diagnostic; median_not_additive; not_a_new_prediction_or_fitted_correction"}
    covariance = distribution.get("covariance_diagnostics", {})
    return {"original_compute_copy_source_failures": failures,
            "dominant_context_actions": dominant,
            "observed_gap_to_context_total_median_ratio": base["observed_gap_to_context_total_median_ratio"],
            "observed_gap_median_ms": milliseconds(base["observed_gap_median_ns"]),
            "gap_cv_scope": "diagnostic_not_original_source_criterion",
            "median_nonadditivity": nonadditivity,
            "instrumentation_control": {"relative_difference": control["relative_difference"],
                "maximum_relative_difference": control["maximum_relative_difference"], "passed": control["passed"],
                "context_process_medians_ms": [milliseconds(value) for value in control["context"]["process_medians_ns"]],
                "bare_process_medians_ms": [milliseconds(value) for value in control["bare"]["process_medians_ns"]]},
            "distribution_covariance_diagnostics": {name: value for name, value in covariance.items() if name != "per_process"},
            "residual_cause": "unresolved"}


def distribution_row(value):
    prediction = value["prediction"]
    return {"prediction_p50_ms": milliseconds(prediction["p50_ns"]),
            "prediction_expected_mean_ms": milliseconds(prediction["expected_duration_ns"]),
            "prediction_p10_ms": milliseconds(prediction["p10_ns"]),
            "prediction_p90_ms": milliseconds(prediction["p90_ns"]),
            "p50_absolute_relative_error": finite(value["p50_absolute_relative_error"], "distribution p50 error"),
            "expected_mean_absolute_relative_error": finite(value["expected_mean_absolute_relative_error"], "distribution mean error"),
            "p10_p90_empirical_coverage": finite(value["p10_p90_empirical_coverage"], "coverage"),
            "candidate_original_gates_passed": value["candidate_original_gates_passed"],
            "original_source_compute_copy_quality_passed": value["original_source_compute_copy_quality_passed"],
            "heldout_within_process_cv_passed": value["heldout_within_process_cv_passed"],
            "heldout_between_process_median_cv_passed": value["heldout_between_process_median_cv_passed"],
            "instrumentation_control_passed": value["instrumentation_control_passed"],
            "candidate_role": value["candidate_role"],
            "primary_model_changed": value["primary_model_changed"],
            "mean_or_coverage_used_to_change_acceptance": value["mean_or_coverage_used_to_change_acceptance"],
            "coverage_scope": value["coverage_scope"],
            "observed_heldout_equal_process_mean_ms": milliseconds(value["observed_heldout_equal_process_mean_ns"]),
            "processes": [{"process_id": row["process_id"], "samples": row["samples"],
                "median_ms": milliseconds(row["median_ns"]), "mean_ms": milliseconds(row["mean_ns"]),
                "cv": row["cv"], "p50_absolute_relative_error": row["p50_absolute_relative_error"],
                "expected_mean_absolute_relative_error": row["expected_mean_absolute_relative_error"],
                "p10_p90_empirical_coverage": row["p10_p90_empirical_coverage"]} for row in value["heldout_processes"]]}


def error_summary(rows, error_key, accepted_key):
    errors = [finite(row[error_key], error_key) for row in rows]
    require(bool(errors), "complete nonempty evaluation required")
    return {"cases": len(rows), "median_absolute_relative_error": statistics.median(errors),
            "worst_absolute_relative_error": max(errors), "error_at_most_20_percent": sum(value <= .20 for value in errors),
            "all_original_gates_passed": sum(row[accepted_key] is True for row in rows)}


def validate_recorded_metrics(base, report):
    """Check recorded residual arithmetic; never calculate a performance price."""
    stats = report["heldout_statistics"]
    observed = ratio(stats["pooled_median_ns"])
    require(observed > 0, "heldout median must be positive")
    within = all(finite(value, "heldout CV") <= .15 for value in stats["within_process_cv"])
    between = finite(stats["between_process_median_cv"], "between-process CV") <= .20
    control = base["instrumentation_control"]["passed"]
    for name in MODELS:
        value = report["models"][name]
        signed_error = float((Fraction(value["predicted_ns"]) - observed) / observed)
        require(math.isclose(value["signed_relative_error"], signed_error, rel_tol=1e-12, abs_tol=1e-15) and
                math.isclose(value["absolute_relative_error"], abs(signed_error), rel_tol=1e-12, abs_tol=1e-15),
                "recorded model residual arithmetic differs")
        source = base["models"][name]["source_compute_copy_quality_passed"]
        require(value["source_compute_copy_quality_passed"] == source and
                value["heldout_within_process_cv_passed"] == within and
                value["heldout_between_process_median_cv_passed"] == between and
                value["instrumentation_control_passed"] == control and
                value["accepted"] == (source and within and between and control and abs(signed_error) <= .20),
                "recorded original-gate acceptance differs")
    candidate = report["distribution_candidate"]
    error = float(abs(Fraction(candidate["prediction"]["p50_ns"]) - observed) / observed)
    require(math.isclose(candidate["p50_absolute_relative_error"], error, rel_tol=1e-12, abs_tol=1e-15),
            "recorded distribution residual arithmetic differs")
    require(0 <= finite(candidate["p10_p90_empirical_coverage"], "coverage") <= 1 and
            finite(candidate["expected_mean_absolute_relative_error"], "mean error") >= 0,
            "invalid recorded distribution coverage/error")
    source = base["models"][MODELS[2]]["source_compute_copy_quality_passed"]
    require(candidate["fixed_quality_gates"] == GATES and
            candidate["original_source_compute_copy_quality_passed"] == source and
            candidate["heldout_within_process_cv_passed"] == within and
            candidate["heldout_between_process_median_cv_passed"] == between and
            candidate["instrumentation_control_passed"] == control and
            candidate["candidate_original_gates_passed"] == (source and within and between and control and error <= .20),
            "recorded distribution original-gate acceptance differs")


def aggregate(campaign):
    campaign = Path(campaign).resolve()
    manifest, manifest_sha = read(campaign / "manifest.json")
    barrier, barrier_sha = read(campaign / "all-models-frozen.json")
    evaluation, evaluation_sha = read(campaign / "evaluation.json")
    signed(manifest, "manifest_sha256", "manifest")
    signed(barrier, "barrier_sha256", "barrier")
    require(manifest.get("schema") == "single-layer-context-campaign-2", "v2 campaign required")
    require(barrier.get("schema") == "single-layer-all-models-frozen-1" and
            barrier["manifest_sha256"] == manifest["manifest_sha256"] and
            barrier["software_sha256"] == manifest["software_sha256"] and
            barrier["heldout_used_for_estimation"] is False, "barrier campaign/software binding differs")
    require(evaluation.get("schema") == "single-layer-context-evaluation-2" and
            evaluation["manifest_sha256"] == manifest["manifest_sha256"] and
            evaluation["barrier_sha256"] == barrier["barrier_sha256"] and
            evaluation["heldout_used_for_estimation"] is False and evaluation["outliers_removed"] == 0,
            "evaluation campaign/barrier/protocol binding differs")
    expected = {(case["id"], poll) for case in manifest["cases"] for poll in manifest["protocol"]["polls"]}
    require(len(expected) == len(manifest["cases"]) * len(manifest["protocol"]["polls"]) and bool(expected), "duplicate/empty declared case inventory")
    keys = lambda rows: [(row["case_id"], row["poll"]) for row in rows]
    for name, rows in (("barrier", barrier["models"]), ("evaluation", evaluation["results"])):
        actual = keys(rows)
        require(len(actual) == len(set(actual)) and set(actual) == expected, name + " omits or duplicates a declared case/poll")
    reports = {(row["case_id"], row["poll"]): row["report"] for row in evaluation["results"]}
    sources = [{"path": name, "sha256": value} for name, value in
               (("manifest.json", manifest_sha), ("all-models-frozen.json", barrier_sha), ("evaluation.json", evaluation_sha))]
    cases = []
    for item in barrier["models"]:
        key = item["case_id"], item["poll"]
        model_path = campaign / "models" / (key[0] + "--poll-" + key[1] + ".json")
        require(Path(item["path"]).resolve() == model_path.resolve(), "frozen model path escapes its declared campaign file")
        combined, model_sha = read(model_path)
        require(model_sha == item["sha256"], "frozen model bytes changed")
        signed(combined, "combined_sha256", "combined model")
        require(combined.get("format_version") == "single-layer-combined-frozen-v2" and
                combined["heldout_used_for_estimation"] is False and combined["primary_model"] == MODELS[2], "combined model scope differs")
        base, distribution, report = combined["base"], combined["distribution"], reports[key]
        signed(base, "frozen_sha256", "base model")
        signed(distribution, "frozen_distribution_sha256", "distribution model")
        require(base.get("format_version") == "single-layer-calibration-frozen-v1" and base["heldout_used"] is False and
                distribution.get("format_version") == "single-layer-duration-distribution-model-v1" and
                distribution["heldout_used_for_estimation"] is False and
                distribution["original_frozen_profile_sha256"] == base["frozen_sha256"], "profile/distribution anchors differ")
        require(report.get("format_version") == "single-layer-calibration-evaluation-v1" and
                report["frozen_sha256"] == base["frozen_sha256"] and report["contract_identity"] == base["contract_identity"] and
                report["fixed_quality_gates"] == base["fixed_quality_gates"] == GATES and
                report["raw_sources_verified"] is True and report["heldout_used_for_fitting"] is False and
                report["parameter_frozen_before_heldout"] is True and report["all_samples_retained"] is True,
                "base evaluation verification/anchor differs")
        require(report["primary_model"] == MODELS[2] and report["accepted"] == report["models"][MODELS[2]]["accepted"], "primary acceptance differs")
        require(set(report["models"]) == set(MODELS), "evaluation model inventory differs")
        for name in MODELS:
            require(report["models"][name]["prediction"] == base["models"][name]["prediction"] and
                    report["models"][name]["predicted_ns"] == base["models"][name]["prediction"]["duration_ns"], "recorded prediction differs from frozen cache")
        candidate = report["distribution_candidate"]
        require(candidate.get("format_version") == "single-layer-distribution-evaluation-v1" and
                candidate["frozen_distribution_sha256"] == distribution["frozen_distribution_sha256"] and
                candidate["original_frozen_profile_sha256"] == base["frozen_sha256"] and
                candidate["prediction"] == distribution["prediction"] and
                candidate["raw_sources_verified"] is True and candidate["heldout_used_for_fitting"] is False and
                candidate["fresh_heldout_after_distribution_freeze"] is True and
                candidate["primary_model_changed"] is False and candidate["mean_or_coverage_used_to_change_acceptance"] is False,
                "distribution candidate cache/anchor/scope differs")
        validate_recorded_metrics(base, report)
        stats = report["heldout_statistics"]
        cases.append({"case_id": key[0], "poll": key[1],
            "models": {name: model_row(report["models"][name]) for name in MODELS},
            "heldout": {"pooled_median_ms": milliseconds(stats["pooled_median_ns"]),
                "median_of_process_medians_ms": milliseconds(stats["median_of_process_medians_ns"]),
                "process_medians_ms": [milliseconds(value) for value in stats["process_medians_ns"]],
                "within_process_cv": stats["within_process_cv"], "between_process_median_cv": stats["between_process_median_cv"],
                "raw_sources_verified_by_evaluator": report["raw_sources_verified"],
                "parameter_frozen_before_heldout_by_evaluator": report["parameter_frozen_before_heldout"]},
            "distribution_candidate": distribution_row(candidate),
            "profile": profile_summary(base, distribution), "accepted": report["accepted"]})
        sources.append({"path": model_path.relative_to(campaign).as_posix(), "sha256": model_sha,
                        "combined_sha256": combined["combined_sha256"], "base_sha256": base["frozen_sha256"],
                        "distribution_sha256": distribution["frozen_distribution_sha256"]})
    summary = {name: error_summary([row["models"][name] for row in cases], "absolute_relative_error", "accepted") for name in MODELS}
    candidates = [row["distribution_candidate"] for row in cases]
    summary["distribution_candidate_p50"] = error_summary(candidates, "p50_absolute_relative_error", "candidate_original_gates_passed")
    summary["distribution_candidate_expected_mean"] = error_summary(candidates, "expected_mean_absolute_relative_error", "candidate_original_gates_passed")
    summary["distribution_candidate_expected_mean"]["acceptance_scope"] = "candidate_original_gates_use_p50_not_mean_error"
    coverage = [row["p10_p90_empirical_coverage"] for row in candidates]
    return {"schema": "single-layer-calibration-portable-aggregate-v1", "campaign_name": campaign.name,
            "manifest_sha256": manifest["manifest_sha256"], "barrier_sha256": barrier["barrier_sha256"],
            "declared_case_poll_count": len(expected), "primary_model": MODELS[2], "fixed_quality_gates": GATES,
            "cases": cases, "summary": summary,
            "distribution_coverage_summary": {"median": statistics.median(coverage), "minimum": min(coverage), "maximum": max(coverage),
                "scope": "descriptive_recorded_p10_p90_coverage_not_calibrated_probability_guarantee"},
            "source_files": sources, "hardware_scope": evaluation["hardware_scope"],
            "verification_scope": "artifact_signatures_inventory_and_cached_prediction_bindings_only; no_raw_or_NumPy_reverification",
            "portable_aggregate_not_raw_evidence": True, "new_performance_predictions_calculated": False,
            "parameters_fitted": False, "raw_numerical_verification_repeated": False,
            "outliers_removed": 0, "source_results_changed": False,
            "residual_cause": "unresolved"}


def summarize(campaign, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("new output file required: " + str(output))
    report = aggregate(campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.campaign, args.out)
    print(json.dumps({"output": str(args.out.resolve()), "cases": report["declared_case_poll_count"],
                      "portable_aggregate_not_raw_evidence": True}))


if __name__ == "__main__":
    main()
