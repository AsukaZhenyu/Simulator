"""Small signed fixtures exercise portable aggregation without any simulator."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / "summarize_single_layer_calibration.py"
SPEC = importlib.util.spec_from_file_location("single_layer_summary", TOOL)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fraction(value):
    return {"numerator": value, "denominator": 1}


def seal(value, name):
    value.pop(name, None)
    value[name] = summary.digest(value)
    return value


class SummaryTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / "build/single-layer-summary-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.output = self.root / "portable.json"
        self.make_campaign()

    def write(self, name, value):
        path = self.campaign / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")
        return summary.read(path)[1]

    def make_campaign(self):
        manifest = seal({"schema": "single-layer-context-campaign-2", "software_sha256": "a" * 64,
            "cases": [{"id": "case-a"}, {"id": "case-b"}], "protocol": {"polls": ["default"]}}, "manifest_sha256")
        self.write("manifest.json", manifest)
        items, results = [], []
        for index, name in enumerate(("case-a", "case-b")):
            observed, source = (index + 1) * 1_000_000, index == 0
            models, reported = {}, {}
            predictions = (observed * 14 // 10, observed * 11 // 10, observed * 105 // 100)
            for key, prediction in zip(summary.MODELS, predictions):
                models[key] = {"prediction": {"duration_ns": prediction}, "source_compute_copy_quality_passed": source}
                error = (prediction - observed) / observed
                reported[key] = {"prediction": {"duration_ns": prediction}, "predicted_ns": prediction,
                    "absolute_relative_error": error, "signed_relative_error": error,
                    "accepted": source and error <= .20,
                    "per_process_absolute_relative_errors": [error] * 3,
                    "source_compute_copy_quality_passed": source, "heldout_within_process_cv_passed": True,
                    "heldout_between_process_median_cv_passed": True, "instrumentation_control_passed": True,
                    "residual_cause": "unresolved", "gap_quality_claimed": False}
            statistics = {}
            for number, cost in ((0, observed * 4 // 10), (1, observed * 6 // 10)):
                statistics[str(number)] = {"value_ns": fraction(cost), "within_process_cv_passed": source,
                    "between_process_median_cv_passed": True, "between_process_median_cv": .01,
                    "processes": [{"median_ns": fraction(cost), "cv": .01} for _ in range(3)]}
            context_totals = {"median_of_process_medians_ns": fraction(observed * 106 // 100),
                              "process_medians_ns": [fraction(observed * 106 // 100)] * 3}
            base = seal({"format_version": "single-layer-calibration-frozen-v1", "heldout_used": False,
                "contract_identity": {"graph": name}, "fixed_quality_gates": deepcopy(summary.GATES),
                "models": models, "actions": [{"kind": "COMPUTE", "target_id": "PV", "device": "gpu"},
                                             {"kind": "COPY", "target_id": "output", "src": "vram", "dst": "dram"}],
                "action_statistics": {"isolated": deepcopy(statistics), "context": deepcopy(statistics)},
                "instrumentation_control": {"context": context_totals,
                    "bare": {"process_medians_ns": [fraction(observed)] * 3},
                    "relative_difference": .06, "maximum_relative_difference": .10, "passed": True},
                "observed_gap_to_context_total_median_ratio": .001,
                "observed_gap_median_ns": fraction(1000)}, "frozen_sha256")
            distribution_prediction = {"p10_ns": observed * 8 // 10, "p50_ns": observed * 102 // 100,
                "p90_ns": observed * 15 // 10, "expected_duration_ns": fraction(observed * 11 // 10)}
            distribution = seal({"format_version": "single-layer-duration-distribution-model-v1",
                "original_frozen_profile_sha256": base["frozen_sha256"], "heldout_used_for_estimation": False,
                "prediction": distribution_prediction, "covariance_diagnostics": {"used_as_model_parameter": False}},
                "frozen_distribution_sha256")
            combined = seal({"format_version": "single-layer-combined-frozen-v2", "base": base,
                "distribution": distribution, "heldout_used_for_estimation": False,
                "primary_model": summary.MODELS[2]}, "combined_sha256")
            file = f"models/{name}--poll-default.json"
            model_sha = self.write(file, combined)
            items.append({"case_id": name, "poll": "default", "path": str(self.campaign / file), "sha256": model_sha})
            candidate = {"format_version": "single-layer-distribution-evaluation-v1",
                "frozen_distribution_sha256": distribution["frozen_distribution_sha256"],
                "original_frozen_profile_sha256": base["frozen_sha256"], "prediction": distribution_prediction,
                "raw_sources_verified": True, "heldout_used_for_fitting": False,
                "fresh_heldout_after_distribution_freeze": True, "primary_model_changed": False,
                "mean_or_coverage_used_to_change_acceptance": False,
                "p50_absolute_relative_error": .02, "expected_mean_absolute_relative_error": 0,
                "p10_p90_empirical_coverage": .8, "candidate_original_gates_passed": source,
                "original_source_compute_copy_quality_passed": source,
                "heldout_within_process_cv_passed": True, "heldout_between_process_median_cv_passed": True,
                "instrumentation_control_passed": True, "candidate_role": "predeclared_secondary_candidate",
                "coverage_scope": "descriptive_not_probability_guarantee", "fixed_quality_gates": deepcopy(summary.GATES),
                "observed_heldout_equal_process_mean_ns": fraction(observed * 11 // 10),
                "heldout_processes": [{"process_id": 100 + n, "samples": 101,
                    "median_ns": fraction(observed), "mean_ns": fraction(observed * 11 // 10), "cv": .01,
                    "p50_absolute_relative_error": .02, "expected_mean_absolute_relative_error": 0,
                    "p10_p90_empirical_coverage": .8} for n in range(3)]}
            report = {"format_version": "single-layer-calibration-evaluation-v1", "frozen_sha256": base["frozen_sha256"],
                "contract_identity": base["contract_identity"], "fixed_quality_gates": deepcopy(summary.GATES),
                "raw_sources_verified": True, "heldout_used_for_fitting": False,
                "parameter_frozen_before_heldout": True, "all_samples_retained": True,
                "primary_model": summary.MODELS[2], "accepted": source, "models": reported,
                "distribution_candidate": candidate,
                "heldout_statistics": {"pooled_median_ns": fraction(observed),
                    "median_of_process_medians_ns": fraction(observed), "process_medians_ns": [fraction(observed)] * 3,
                    "within_process_cv": [.01] * 3, "between_process_median_cv": .01}}
            results.append({"case_id": name, "poll": "default", "report": report})
        barrier = seal({"schema": "single-layer-all-models-frozen-1", "models": items,
            "manifest_sha256": manifest["manifest_sha256"], "software_sha256": manifest["software_sha256"],
            "heldout_used_for_estimation": False}, "barrier_sha256")
        self.write("all-models-frozen.json", barrier)
        self.write("evaluation.json", {"schema": "single-layer-context-evaluation-2", "results": results,
            "manifest_sha256": manifest["manifest_sha256"], "barrier_sha256": barrier["barrier_sha256"],
            "heldout_used_for_estimation": False, "outliers_removed": 0, "hardware_scope": "synthetic_fixture_only"})

    def mutate(self, file, callback):
        value, _ = summary.read(self.campaign / file)
        callback(value)
        self.write(file, value)

    def test_two_cases_aggregate_recorded_values_and_gates(self):
        report = summary.summarize(self.campaign, self.output)
        self.assertEqual(report["declared_case_poll_count"], 2)
        self.assertEqual(report["summary"][summary.MODELS[2]]["all_original_gates_passed"], 1)
        self.assertEqual(report["summary"][summary.MODELS[0]]["error_at_most_20_percent"], 0)
        self.assertAlmostEqual(report["summary"][summary.MODELS[1]]["median_absolute_relative_error"], .1)
        self.assertEqual(report["distribution_coverage_summary"]["median"], .8)
        self.assertEqual(report["cases"][0]["models"][summary.MODELS[2]]["prediction_ms"], 1.05)
        self.assertEqual(report["cases"][1]["profile"]["original_compute_copy_source_failures"]["context"]["actions_with_any_within_process_cv_failure"], 2)
        self.assertTrue(report["portable_aggregate_not_raw_evidence"])
        self.assertFalse(report["new_performance_predictions_calculated"])
        self.assertNotIn(str(self.root), self.output.read_text())
        self.assertEqual(len(report["source_files"]), 5)

    def test_output_must_be_new_before_any_campaign_read(self):
        self.output.write_text("preserve", encoding="utf-8")
        with patch.object(summary, "aggregate", side_effect=AssertionError("must not read")):
            with self.assertRaises(FileExistsError):
                summary.summarize(self.campaign, self.output)
        self.assertEqual(self.output.read_text(), "preserve")

    def test_missing_or_duplicate_evaluation_case_rejected(self):
        self.mutate("evaluation.json", lambda value: value["results"].pop())
        with self.assertRaisesRegex(ValueError, "evaluation omits or duplicates"):
            summary.summarize(self.campaign, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_barrier_model_rejected_even_if_resigned(self):
        barrier, _ = summary.read(self.campaign / "all-models-frozen.json")
        barrier["models"][-1] = deepcopy(barrier["models"][0])
        seal(barrier, "barrier_sha256")
        self.write("all-models-frozen.json", barrier)
        self.mutate("evaluation.json", lambda value: value.update(barrier_sha256=barrier["barrier_sha256"]))
        with self.assertRaisesRegex(ValueError, "barrier omits or duplicates"):
            summary.aggregate(self.campaign)

    def test_manifest_signature_change_rejected(self):
        self.mutate("manifest.json", lambda value: value["protocol"]["polls"].append("0"))
        with self.assertRaisesRegex(ValueError, "manifest signature"):
            summary.aggregate(self.campaign)

    def test_frozen_file_byte_tamper_rejected(self):
        self.mutate("models/case-a--poll-default.json", lambda value: value.update(primary_model="other"))
        with self.assertRaisesRegex(ValueError, "model bytes changed"):
            summary.aggregate(self.campaign)

    def test_stale_inner_signature_rejected_after_outer_reseal(self):
        model, _ = summary.read(self.campaign / "models/case-a--poll-default.json")
        model["base"]["observed_gap_to_context_total_median_ratio"] = .2
        seal(model, "combined_sha256")
        digest = self.write("models/case-a--poll-default.json", model)
        barrier, _ = summary.read(self.campaign / "all-models-frozen.json")
        barrier["models"][0]["sha256"] = digest
        seal(barrier, "barrier_sha256")
        self.write("all-models-frozen.json", barrier)
        self.mutate("evaluation.json", lambda value: value.update(barrier_sha256=barrier["barrier_sha256"]))
        with self.assertRaisesRegex(ValueError, "base model signature"):
            summary.aggregate(self.campaign)

    def test_model_path_cannot_redirect_to_other_file(self):
        barrier, _ = summary.read(self.campaign / "all-models-frozen.json")
        barrier["models"][0]["path"] = str(self.root / "elsewhere.json")
        seal(barrier, "barrier_sha256")
        self.write("all-models-frozen.json", barrier)
        self.mutate("evaluation.json", lambda value: value.update(barrier_sha256=barrier["barrier_sha256"]))
        with self.assertRaisesRegex(ValueError, "path escapes"):
            summary.aggregate(self.campaign)

    def test_evaluation_prediction_must_equal_frozen_cache(self):
        self.mutate("evaluation.json", lambda value: value["results"][0]["report"]["models"][summary.MODELS[0]].update(predicted_ns=123))
        with self.assertRaisesRegex(ValueError, "prediction differs"):
            summary.aggregate(self.campaign)

    def test_evaluation_residual_tamper_is_detected_without_predicting(self):
        self.mutate("evaluation.json", lambda value: value["results"][0]["report"]["models"][summary.MODELS[0]].update(absolute_relative_error=.01))
        with self.assertRaisesRegex(ValueError, "residual arithmetic"):
            summary.aggregate(self.campaign)

    def test_acceptance_cannot_be_relaxed_while_error_fails(self):
        self.mutate("evaluation.json", lambda value: value["results"][0]["report"]["models"][summary.MODELS[0]].update(accepted=True))
        with self.assertRaisesRegex(ValueError, "original-gate acceptance"):
            summary.aggregate(self.campaign)

    def test_distribution_original_anchor_is_required(self):
        self.mutate("evaluation.json", lambda value: value["results"][0]["report"]["distribution_candidate"].update(frozen_distribution_sha256="d" * 64))
        with self.assertRaisesRegex(ValueError, "candidate cache/anchor"):
            summary.aggregate(self.campaign)

    def test_missing_evaluation_cannot_emit_partial_report(self):
        (self.campaign / "evaluation.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summary.summarize(self.campaign, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
