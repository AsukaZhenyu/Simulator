"""Synthetic evidence accounting for the optional profile distribution adapter."""
from copy import deepcopy
from datetime import timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_infer_model.performance.serial_actions import SerialActionTiming, SerialTimingProgram
from single_layer_calibration import distribution_profiles as profile


def encoded(value):
    value = Fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def sign(value):
    value.pop("frozen_sha256", None)
    value["frozen_sha256"] = profile._hash(value)
    return value


class DistributionProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.actions = [dict(kind="COMPUTE", target_id="op", device="cpu"), dict(kind="ADVANCE")]
        self.program = SerialTimingProgram((SerialActionTiming(0, "COMPUTE", "duration0", "gap0"),
                                           SerialActionTiming(1, "ADVANCE", "duration1", "gap1")), "tail")
        self.records = [self.record(i) for i in range(3)]
        self.frozen = sign(dict(format_version="single-layer-calibration-frozen-v1",
            frozen_at_utc="2026-01-01T00:10:00+00:00", heldout_used=False,
            contract_identity=dict(geometry="synthetic-toy"), execution_scope=dict(clock="synthetic-test"),
            software_sha256="a" * 64, actions=self.actions,
            models=dict(context_actions_and_gaps=dict(program=self.program.to_dict(),
                        source_compute_copy_quality_passed=True)),
            instrumentation_control=dict(passed=True),
            profile_runs=self.records, fixed_quality_gates=dict(independent_processes_min=3,
                within_process_cv_max=0.15, between_process_median_cv_max=0.20,
                pooled_median_absolute_relative_error_max=0.20)))

    def record(self, index):
        # A and B are negatively correlated: their marginal medians are zero,
        # median(A+B)=10, Var(A)+Var(B)=48, Var(A+B)=16. Three constant ns from
        # gaps/tail change the mean/median but not either variance.
        a, b = [0, 0, 0, 10, 10], [10, 10, 0, 0, 0]
        iterations = []
        for number, (x, y) in enumerate(zip(a, b)):
            intervals = [dict(action_index=0, kind="COMPUTE", begin_offset_ns=1, end_offset_ns=1 + x),
                         dict(action_index=1, kind="ADVANCE", begin_offset_ns=2 + x, end_offset_ns=2 + x + y)]
            iterations.append(dict(iteration=number, elapsed_ns=3 + x + y, action_intervals=intervals))
        native = dict(mode="context-trace", process_id=100 + index, samples=5,
            actions=self.actions, components=[], iterations=iterations,
            total_samples_ns=[row["elapsed_ns"] for row in iterations])
        path = self.directory / f"process-{index}.json"
        path.write_text(json.dumps(native, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return dict(run_id=f"profile-{index}", split="profile", mode="context-trace",
            process_id=100 + index, seed=index, numerical_passed=True, source_verified=True,
            contract_identity=dict(geometry="synthetic-toy"), execution_scope=dict(clock="synthetic-test"),
            software_sha256="a" * 64,
            started_at_utc=f"2026-01-01T00:0{index}:00+00:00",
            completed_at_utc=f"2026-01-01T00:0{index}:01+00:00",
            source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), native=native)

    def freeze(self):
        return profile.freeze_distributions(self.frozen, trials=101, seed=31)

    def base_report(self, frozen_distribution):
        rows = []
        for index in range(3):
            native = dict(mode="bare-replay", process_id=200 + index, program_sha256="b" * 64,
                actions=self.actions, components=[], samples=5, total_samples_ns=[13] * 5,
                iterations=[dict(iteration=i, elapsed_ns=13, action_intervals=[]) for i in range(5)])
            path = self.directory / f"heldout-{index}.json"
            path.write_text(json.dumps(native) + "\n", encoding="utf-8", newline="\n")
            begin = profile._utc(frozen_distribution["frozen_at_utc"]) + timedelta(seconds=1 + 2 * index)
            rows.append(dict(run_id=f"heldout-{index}", split="heldout", mode="bare-replay",
                process_id=200 + index, program_sha256="b" * 64, seed=1000 + index,
                started_at_utc=begin.isoformat(), completed_at_utc=(begin + timedelta(seconds=1)).isoformat(),
                contract_identity=deepcopy(self.frozen["contract_identity"]),
                execution_scope=deepcopy(self.frozen["execution_scope"]), software_sha256="a" * 64,
                source_verified=True, numerical_passed=True, source_path=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), native=native))
        return dict(format_version="single-layer-calibration-evaluation-v1",
            frozen_sha256=self.frozen["frozen_sha256"], contract_identity=deepcopy(self.frozen["contract_identity"]),
            raw_sources_verified=True, heldout_used_for_fitting=False,
            parameter_frozen_before_heldout=True, all_samples_retained=True,
            fixed_quality_gates=deepcopy(self.frozen["fixed_quality_gates"]),
            primary_model="context_actions_and_gaps", accepted=False, heldout_runs=rows,
            heldout_statistics=dict(process_medians_ns=[encoded(13)] * 3, pooled_median_ns=encoded(13),
                median_of_process_medians_ns=encoded(13), within_process_cv=[0.0] * 3,
                between_process_median_cv=0.0),
            models=dict(context_actions_and_gaps=dict(predicted_ns=3, absolute_relative_error=10 / 13,
                source_compute_copy_quality_passed=True, instrumentation_control_passed=True,
                heldout_within_process_cv_passed=True, heldout_between_process_median_cv_passed=True,
                accepted=False)))

    def test_retains_all_observations_as_context_marginal_frequencies(self):
        frozen = self.freeze()
        processes = frozen["distribution_contract"]["processes"]
        self.assertEqual(len(processes), 3)
        for process in processes:
            self.assertEqual(process["durations"]["duration0"], dict(values_ns=[0, 10], counts=[3, 2]))
            self.assertEqual(process["durations"]["duration1"], dict(values_ns=[0, 10], counts=[3, 2]))
            self.assertEqual(process["durations"]["gap0"], dict(values_ns=[1], counts=[5]))
            self.assertEqual(process["durations"]["gap1"], dict(values_ns=[1], counts=[5]))
            self.assertEqual(process["durations"]["tail"], dict(values_ns=[1], counts=[5]))
        self.assertEqual(frozen["prediction"]["expected_duration_ns"], encoded(11))
        self.assertTrue(frozen["all_context_samples_retained"])

    def test_never_changes_or_resigns_original_pilot_artifact(self):
        original = deepcopy(self.frozen)
        result = self.freeze()
        self.assertEqual(self.frozen, original)
        self.assertEqual(result["original_frozen_profile_sha256"], original["frozen_sha256"])
        self.assertFalse(result["original_pilot_artifact_modified"])
        self.assertGreaterEqual(profile._utc(result["frozen_at_utc"]), profile._utc(original["frozen_at_utc"]))

    def test_covariance_and_median_diagnostics_are_exact_and_not_parameters(self):
        frozen = self.freeze()
        diagnostic = frozen["covariance_diagnostics"]
        for row in diagnostic["per_process"]:
            self.assertEqual(row["sum_interval_population_variances_ns2"], encoded(48))
            self.assertEqual(row["observed_sum_population_variance_ns2"], encoded(16))
            self.assertEqual(row["twice_sum_cross_interval_covariances_ns2"], encoded(-32))
            self.assertEqual(row["observed_sum_mean_minus_sum_interval_means_ns"], encoded(0))
            self.assertEqual(row["sum_interval_medians_ns"], encoded(3))
            self.assertEqual(row["observed_sum_median_ns"], encoded(13))
            self.assertEqual(row["observed_sum_median_minus_sum_interval_medians_ns"], encoded(10))
            self.assertFalse(row["used_as_model_parameter"])
        self.assertEqual(diagnostic["hierarchical_independent_sum_variance_ns2"], encoded(48))
        self.assertEqual(diagnostic["observed_hierarchical_sum_variance_ns2"], encoded(16))
        self.assertFalse(diagnostic["used_as_model_parameter"])
        self.assertFalse(frozen["whole_iteration_totals_used_as_parameters"])

    def test_only_interval_contract_is_passed_to_modeling(self):
        with mock.patch.object(profile.modeling, "evaluate_serial_distributions",
                               wraps=profile.modeling.evaluate_serial_distributions) as spy:
            result = self.freeze()
        spy.assert_called_once()
        program, processes = spy.call_args.args
        self.assertEqual(program, self.program)
        self.assertEqual(set(processes[0].durations), {"duration0", "duration1", "gap0", "gap1", "tail"})
        self.assertNotIn("total_samples_ns", result["distribution_contract"])
        self.assertFalse(result["whole_trace_resampling"])

    def test_predict_reconstructs_and_delegates_from_serialized_frozen_contract(self):
        result = self.freeze()
        restored = json.loads(json.dumps(result))
        with mock.patch.object(profile.modeling, "evaluate_serial_distributions",
                               wraps=profile.modeling.evaluate_serial_distributions) as spy:
            prediction = profile.predict_distribution(restored)
        spy.assert_called_once()
        self.assertEqual(prediction, result["prediction"])

    def test_process_sources_have_equal_weight_with_unchanged_source_hashes(self):
        before = {record["source_path"]: hashlib.sha256(Path(record["source_path"]).read_bytes()).hexdigest()
                  for record in self.records}
        result = self.freeze()
        for source in result["profile_context_sources"]:
            self.assertEqual(source["process_weight"], encoded(Fraction(1, 3)))
            self.assertEqual(source["retained_samples_per_interval"], 5)
            self.assertEqual(source["source_sha256"], before[source["source_path"]])
            self.assertEqual(hashlib.sha256(Path(source["source_path"]).read_bytes()).hexdigest(), before[source["source_path"]])

    def test_parent_signature_and_raw_bytes_are_both_checked(self):
        original = deepcopy(self.frozen)
        self.frozen["actions"][0]["target_id"] = "changed"
        with self.assertRaises(ValueError):
            self.freeze()
        self.frozen = original
        path = Path(self.records[0]["source_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.freeze()

    def test_heldout_record_and_unverified_numerics_are_rejected(self):
        self.frozen["profile_runs"][0]["split"] = "heldout"
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()
        self.frozen["profile_runs"][0]["split"] = "profile"
        self.frozen["profile_runs"][0]["numerical_passed"] = False
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_missing_interval_and_wrong_exact_context_are_rejected(self):
        self.frozen["profile_runs"][0]["execution_scope"] = dict(clock="different")
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()
        self.frozen["profile_runs"][0]["execution_scope"] = dict(clock="synthetic-test")
        native = self.frozen["profile_runs"][0]["native"]
        native["iterations"][0]["action_intervals"].pop()
        path = Path(self.frozen["profile_runs"][0]["source_path"])
        path.write_text(json.dumps(native) + "\n", encoding="utf-8", newline="\n")
        self.frozen["profile_runs"][0]["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_chronology_duplicate_sources_and_too_few_processes_rejected(self):
        self.frozen["profile_runs"][0]["completed_at_utc"] = "2026-01-01T00:11:00+00:00"
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()
        self.frozen["profile_runs"][0]["completed_at_utc"] = "2026-01-01T00:00:01+00:00"
        self.frozen["profile_runs"].pop()
        sign(self.frozen)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_distribution_artifact_tampering_is_rejected(self):
        frozen = self.freeze()
        frozen["distribution_contract"]["processes"][0]["durations"]["duration0"]["counts"][0] += 1
        with self.assertRaises(ValueError):
            profile.predict_distribution(frozen)

    def test_evaluation_reports_mean_and_coverage_without_changing_primary_acceptance(self):
        frozen = self.freeze()
        report = profile.evaluate_distribution(frozen, self.base_report(frozen))
        self.assertEqual(report["observed_heldout_equal_process_mean_ns"], encoded(13))
        self.assertAlmostEqual(report["expected_mean_absolute_relative_error"], 2 / 13)
        self.assertEqual(report["candidate_original_gates_passed"], report["p50_absolute_relative_error"] <= 0.20)
        self.assertFalse(report["primary_model_changed"])
        self.assertFalse(report["primary_acceptance_unchanged"])
        self.assertFalse(report["mean_or_coverage_used_to_change_acceptance"])
        self.assertEqual(report["p10_p90_empirical_coverage"], 1.0)
        self.assertTrue(report["raw_sources_verified"])

    def test_evaluation_rejects_wrong_anchor_or_unverified_base_report(self):
        frozen = self.freeze()
        report = self.base_report(frozen)
        report["frozen_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)
        report["frozen_sha256"] = self.frozen["frozen_sha256"]
        report["raw_sources_verified"] = False
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)

    def test_evaluation_rejects_old_pilot_heldout_or_reused_seed(self):
        frozen = self.freeze()
        report = self.base_report(frozen)
        report["heldout_runs"][0]["started_at_utc"] = (
            profile._utc(frozen["frozen_at_utc"]) - timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)
        report = self.base_report(frozen)
        report["heldout_runs"][0]["seed"] = self.records[0]["seed"]
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)

    def test_evaluation_rejects_changed_cached_model_even_if_artifact_is_resigned(self):
        frozen = self.freeze()
        frozen["prediction"]["p50_ns"] += 1
        frozen.pop("frozen_distribution_sha256")
        frozen["frozen_distribution_sha256"] = profile._hash(frozen)
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, self.base_report(frozen))

    def test_evaluation_rehashes_raw_and_checks_report_statistics(self):
        frozen = self.freeze()
        report = self.base_report(frozen)
        Path(report["heldout_runs"][0]["source_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)
        report = self.base_report(frozen)
        report["heldout_statistics"]["pooled_median_ns"] = encoded(12)
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)

    def test_evaluation_preserves_original_control_gate_and_threshold(self):
        frozen = self.freeze()
        report = self.base_report(frozen)
        report["fixed_quality_gates"]["pooled_median_absolute_relative_error_max"] = 0.30
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)
        report = self.base_report(frozen)
        report["models"]["context_actions_and_gaps"]["instrumentation_control_passed"] = False
        with self.assertRaises(ValueError):
            profile.evaluate_distribution(frozen, report)


if __name__ == "__main__":
    unittest.main()
