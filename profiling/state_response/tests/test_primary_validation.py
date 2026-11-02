"""Synthetic validation-gate fixtures, not hardware or raw-evidence receipts.

The helper consumes predictions already produced by modeling and observation
stability already verified by the existing evaluator. These tests exercise only
selection, coverage and gate aggregation. No fitting or time predictor lives here.
"""
from copy import deepcopy
import unittest

from profiling.state_response import analysis


BASE = dict(cpu_replay=0, gpu_replay=0, host_memory_bytes=0, device_memory_bytes=0)
CPU = dict(BASE, cpu_replay=1)


def fixture(*, dynamic_error=0, lagged_error=0, pids=(101, 102, 103)):
    records = [dict(run_id=f"r{i}", process_id=pid) for i, pid in enumerate(pids)]
    predictions = []
    for record in records:
        for label, state, observed in (("base", BASE, 1000), ("cpu", CPU, 2000), ("cross", None, 1500)):
            coverage = dict(category="known_state_crossing" if state is None else "steady", steady_state=state)
            predictions.append(dict(run_id=record["run_id"], process_id=record["process_id"], observation_id=label,
                observed_ns=observed, coverage=coverage,
                predicted_ns=round(observed * (1 + dynamic_error)), relative_error=dynamic_error,
                comparisons={"lagged_dynamic_state_integration": dict(
                    predicted_ns=round(observed * (1 + lagged_error)), relative_error=lagged_error)}))
    frozen = dict(protocol={"primary_prediction_model": "lagged_dynamic_state_integration"},
        gates=dict(minimum_independent_runs=3, within_process_cv_max=.15,
                   between_process_cv_max=.20, relative_error_max=.20),
        required_states=[BASE, CPU], source_qualified=True, request_response_delays={"qualified": True})
    stability = [dict(state=state, measurement_stability_qualified=True,
                      between_process_observed_median_cv=0) for state in (BASE, CPU)]
    return frozen, records, predictions, stability


def validate(parts, *, barrier=True, raw=True):
    frozen, records, predictions, stability = parts
    return analysis._primary_validation(frozen, records, predictions, barrier_verified=barrier,
                                        raw_verified=raw, state_stability=stability)


class PrimaryValidationTests(unittest.TestCase):
    def test_full_evaluator_adds_primary_scope_without_replacing_legacy_report(self):
        # Reuse the existing small, byte-bound synthetic raw fixture. This is
        # not an invocation of a native backend or a hardware accuracy claim.
        from profiling.state_response.tests import test_analysis as fixtures
        example = fixtures.ResponseAnalysisTests()
        example.setUp()
        self.addCleanup(example.doCleanups)
        program, _, frozen = example.profile()
        heldout = [example.record("heldout", bare=True)[1] for _ in range(3)]
        result = analysis.evaluate_heldout(program, frozen, heldout, barrier=example.barrier(frozen))
        self.assertEqual(result["legacy_validation_scope"]["model"], "dynamic_state_integration")
        self.assertFalse(result["legacy_validation_scope"]["primary_model_selection_applied_to_these_fields"])
        primary = result["primary_validation"]
        self.assertEqual(primary["chosen_model"], "dynamic_state_integration")
        self.assertEqual(primary["overall"], result["overall"])
        self.assertEqual(primary["buckets"], {name: result["buckets"][name] for name in primary["buckets"]})
        self.assertFalse(primary["accepted"])
        unsigned = {key: value for key, value in result.items() if key != "evaluation_sha256"}
        self.assertEqual(result["evaluation_sha256"], analysis.digest(unsigned))

    def test_good_legacy_error_cannot_accept_bad_lagged_primary(self):
        parts = fixture(dynamic_error=0, lagged_error=.5)
        before = deepcopy(parts)
        legacy = analysis._bucket_summary(parts[2], parts[1])
        result = validate(parts)
        self.assertEqual(legacy["process_balanced_median_absolute_relative_error"], 0)
        self.assertEqual(result["overall"]["process_balanced_median_absolute_relative_error"], .5)
        self.assertFalse(result["statistical_protocol_accepted"])
        self.assertEqual(parts, before)

    def test_good_primary_error_is_reported_separately_without_full_accuracy_claim(self):
        parts = fixture(dynamic_error=.5, lagged_error=0)
        result = validate(parts)
        self.assertTrue(result["steady_accepted"])
        self.assertTrue(result["known_crossing_accepted"])
        self.assertTrue(result["statistical_protocol_accepted"])
        self.assertFalse(result["full_accuracy_qualified"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["actual_CPU_GPU_available_service_transfer_identified"])
        self.assertIn("not independently established", result["instrumentation_control_status"])

    def test_unqualified_lag_source_blocks_perfect_primary_errors(self):
        parts = fixture()
        parts[0]["request_response_delays"]["qualified"] = False
        result = validate(parts)
        self.assertTrue(result["median_error_within_threshold"])
        self.assertTrue(result["original_compute_copy_profile_source_qualified"])
        self.assertFalse(result["lag_source_qualified"])
        self.assertFalse(result["required_profile_source_gates_passed"])
        self.assertFalse(result["statistical_protocol_accepted"])

    def test_original_profile_barrier_and_raw_guards_remain_required(self):
        for mode in ("source", "barrier", "raw"):
            with self.subTest(mode=mode):
                parts = fixture()
                if mode == "source":
                    parts[0]["source_qualified"] = False
                result = validate(parts, barrier=mode != "barrier", raw=mode != "raw")
                self.assertFalse(result["statistical_protocol_accepted"])

    def test_unknown_state_prediction_cannot_disappear_behind_good_median(self):
        parts = fixture()
        parts[2][1]["comparisons"]["lagged_dynamic_state_integration"].update(
            predicted_ns=None, relative_error=None)
        result = validate(parts)
        self.assertTrue(result["median_error_within_threshold"])
        cpu = next(row for row in result["by_steady_state"] if row["state"] == CPU)
        self.assertEqual((cpu["sample_count"], cpu["predictable_count"]), (3, 2))
        self.assertFalse(cpu["complete_prediction_coverage"])
        self.assertFalse(cpu["accuracy_passed"])
        self.assertFalse(result["steady_accepted"])
        self.assertFalse(result["statistical_protocol_accepted"])

    def test_required_state_without_samples_or_with_failed_cv_cannot_be_omitted(self):
        missing = fixture()
        missing[2][:] = [row for row in missing[2] if row["coverage"]["steady_state"] != CPU]
        failed_cv = fixture()
        failed_cv[3][1]["measurement_stability_qualified"] = False
        failed_cv[3][1]["between_process_observed_median_cv"] = .8
        for parts in (missing, failed_cv):
            with self.subTest(parts=parts[3]):
                self.assertFalse(validate(parts)["steady_accepted"])

    def test_crossing_requires_three_runs_three_pids_and_complete_prediction_coverage(self):
        two_runs = fixture()
        two_runs[2][:] = [row for row in two_runs[2]
                           if not (row["run_id"] == "r2" and row["observation_id"] == "cross")]
        two_pids = fixture(pids=(101, 102, 102))
        unknown = fixture()
        unknown[2][2]["comparisons"]["lagged_dynamic_state_integration"].update(
            predicted_ns=None, relative_error=None)
        for parts in (two_runs, two_pids, unknown):
            result = validate(parts)
            self.assertTrue(result["steady_accepted"])
            self.assertFalse(result["known_crossing_accepted"])
            self.assertFalse(result["statistical_protocol_accepted"])

    def test_fixed_primary_selection_changes_only_chosen_statistics(self):
        parts = fixture(dynamic_error=0, lagged_error=.5)
        raw_predictions = deepcopy(parts[2])
        gates = deepcopy(parts[0]["gates"])
        self.assertFalse(validate(parts)["statistical_protocol_accepted"])
        parts[0]["protocol"]["primary_prediction_model"] = "dynamic_state_integration"
        result = validate(parts)
        self.assertTrue(result["statistical_protocol_accepted"])
        self.assertEqual(result["chosen_model"], "dynamic_state_integration")
        self.assertFalse(result["lag_source_required"])
        self.assertIsNone(result["lag_source_qualified"])
        self.assertEqual(parts[0]["gates"], gates)
        self.assertEqual(parts[2], raw_predictions)

    def test_legacy_default_remains_explicit_and_does_not_require_optional_lag(self):
        parts = fixture()
        parts[0]["protocol"] = {}
        del parts[0]["request_response_delays"]
        result = validate(parts)
        self.assertEqual(result["chosen_model"], "dynamic_state_integration")
        self.assertEqual(result["selection_source"], "legacy dynamic default")
        self.assertTrue(result["statistical_protocol_accepted"])

    def test_missing_selected_model_or_unknown_error_pair_is_rejected(self):
        absent = fixture()
        absent[2][0]["comparisons"] = {}
        mismatched = fixture()
        mismatched[2][0]["comparisons"]["lagged_dynamic_state_integration"]["relative_error"] = None
        for parts in (absent, mismatched):
            with self.assertRaises(ValueError):
                validate(parts)

    def test_unknown_primary_model_cannot_fall_back_to_legacy_prices(self):
        parts = fixture()
        parts[0]["protocol"]["primary_prediction_model"] = "posthoc_best"
        with self.assertRaisesRegex(ValueError, "unknown primary validation model"):
            validate(parts)


if __name__ == "__main__":
    unittest.main()
