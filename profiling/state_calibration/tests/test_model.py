"""Synthetic source/heldout and unknown-scope checks; no hardware execution."""
import copy
from fractions import Fraction
import math
import statistics
import unittest

from profiling.state_calibration import model as m


def row(number, split="profile", condition="baseline", samples=None, **job_updates):
    samples = [100, 100, 100, 100] if samples is None else samples
    median = statistics.median(samples)
    cv = statistics.pstdev(samples) / statistics.mean(samples) if statistics.mean(samples) else 0
    job = {"id": f"{split}-{condition}-{number}", "split": split, "case_id": "exact-qk-H128-q16-hist0-f32",
           "seed": 1 if split == "profile" else 99, "repeat": number,
           "condition": condition, "mode": "isolated", "instrumentation": "stamped", "telemetry": True}
    job.update(job_updates)
    return {"job": job, "summary": {"target_median_ns": median, "iteration_median_ns": median,
                                    "target_samples_ns": list(samples), "iteration_samples_ns": list(samples),
                                    "target_cv": cv, "iteration_cv": cv,
                                    "native_numeric_passed": True, "condition_verified": True,
                                    "alignment_model_ready": True, "process_id": 12345 + number,
                                    "device_state": {"sm_clock_mhz": 1200}}}


def rows(split="profile", condition="baseline", value=100, count=3):
    return [row(number, split, condition, [value] * 4) for number in range(count)]


def fit(source=None):
    return m.fit_model(rows() if source is None else source, {"synthetic-native-source.cpp": "a" * 64}, "2026-10-18T00:00:00+00:00")


def price(model, condition="baseline", **options):
    return m.lookup(model, "exact-qk-H128-q16-hist0-f32", "isolated", "stamped", True, condition, **options)


class FitTests(unittest.TestCase):
    def test_exact_stable_sources_are_frozen_with_raw_samples(self):
        source = rows()
        result = fit(source)
        self.assertEqual(result["profile_row_count"], 3)
        self.assertEqual(result["profile_seeds"], [1])
        self.assertFalse(result["heldout_used_for_fit"])
        self.assertEqual(len(result["model_sha256"]), 64)
        self.assertEqual(price(result)["price_ns"], 100)
        self.assertTrue(price(result)["profile_accepted"])
        source[0]["summary"]["target_samples_ns"][0] = 99999
        self.assertEqual(price(result)["price_ns"], 100)
        cell = result["families"]["conditional"]["cells"][0]
        self.assertEqual(cell["processes"][0]["samples_ns"], [100] * 4)

    def test_heldout_cannot_enter_fit(self):
        with self.assertRaisesRegex(ValueError, "profile-only"):
            fit(rows("heldout"))
        with self.assertRaises(ValueError):
            fit(rows() + rows("heldout"))

    def test_nonlegacy_execution_boundaries_cannot_enter_profile_model(self):
        for boundary in ("async_one_sync", "async_events", None, "unknown"):
            source = rows()
            source[0]["job"]["execution_boundary"] = boundary
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "diagnostic-only"):
                fit(source)
        source = rows()
        for item in source:
            item["job"]["execution_boundary"] = "legacy_double_sync"
        self.assertTrue(price(fit(source))["profile_accepted"])

    def test_constant_pools_same_context_conditions_only(self):
        result = fit(rows(value=100) + rows(condition="gpu_replay_load", value=200))
        self.assertEqual(price(result, family="constant")["price_ns"], 150)
        self.assertEqual(price(result, "gpu_replay_load")["price_ns"], 200)
        self.assertEqual(price(result)["price_ns"], 100)
        self.assertFalse(price(result, family="constant")["profile_accepted"])

    def test_reference_rate_is_exact_context_rational(self):
        result = fit(rows(value=101) + rows(condition="gpu_replay_load", value=303))
        rate = price(result, "gpu_replay_load")["reference_service_rate"]
        self.assertEqual(Fraction(**rate), Fraction(1, 3))
        self.assertIn("exact measured operator", price(result)["rate_scope"])

    def test_unobserved_condition_is_unknown_even_for_constant(self):
        result = fit()
        for family in ("conditional", "constant"):
            with self.assertRaises(m.UnknownCost):
                price(result, "gpu_replay_load", family=family)
            with self.assertRaises(m.UnknownCost):
                price(result, "unobserved_state", family=family)

    def test_exact_context_axes_are_not_interpolated(self):
        result = fit()
        for changes in ({"case_id": "neighbor-shape"}, {"mode": "mixed"}, {"instrumentation": "bare"}, {"telemetry": False}):
            request = {"case_id": "exact-qk-H128-q16-hist0-f32", "mode": "isolated", "instrumentation": "stamped", "telemetry": True, "condition": "baseline"}
            request.update(changes)
            with self.assertRaises(m.UnknownCost):
                m.lookup(result, **request)

    def test_numerical_failure_poisoned_cell_is_not_dropped(self):
        source = rows(count=4)
        source[3]["summary"]["native_numeric_passed"] = False
        result = fit(source)
        with self.assertRaises(m.UnknownCost):
            price(result)
        cell = next(cell for cell in result["families"]["conditional"]["cells"] if cell["metric"] == "target")
        self.assertEqual(cell["process_count"], 4)
        self.assertFalse(cell["supported"])
        self.assertIsNone(cell["price_ns"])
        self.assertIn("native_numerical_verification_not_passed", cell["processes"][3]["validation_failures"])

    def test_missing_or_false_condition_is_not_assumed(self):
        for value in (None, False):
            source = rows()
            source[0]["summary"]["condition_verified"] = value
            with self.assertRaises(m.UnknownCost):
                price(fit(source))

    def test_alignment_failure_keeps_diagnostic_price_but_fails_acceptance(self):
        source = rows()
        source[1]["summary"]["alignment_model_ready"] = False
        result = price(fit(source))
        self.assertEqual(result["price_ns"], 100)
        self.assertFalse(result["profile_accepted"])
        self.assertIn("one_or_more_telemetry_alignments_not_ready", result["profile_gate_failures"])

    def test_missing_raw_samples_cannot_be_reconstructed_from_summary(self):
        source = rows()
        source[0]["summary"].pop("target_samples_ns")
        with self.assertRaises(m.UnknownCost):
            price(fit(source))

    def test_invalid_samples_zero_negative_bool_and_noninteger(self):
        for value in (0, -1, True, 1.5):
            source = rows()
            source[0]["summary"]["target_samples_ns"][0] = value
            with self.assertRaises(m.UnknownCost):
                price(fit(source))

    def test_nonfinite_anywhere_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            source = rows()
            source[0]["summary"]["device_state"]["sm_clock_mhz"] = value
            with self.assertRaises(ValueError):
                fit(source)

    def test_no_epsilon_or_near_zero_duration_adjustment(self):
        result = fit(rows(value=1))
        self.assertEqual(price(result)["price_ns"], 1)
        source = rows()
        source[0]["summary"]["target_samples_ns"] = [0, 0, 0, 0]
        with self.assertRaises(m.UnknownCost):
            price(fit(source))

    def test_reported_cv_mismatch_is_explicit(self):
        source = rows()
        source[0]["summary"]["target_cv"] = 0.123
        result = fit(source)
        cell = next(cell for cell in result["families"]["conditional"]["cells"] if cell["metric"] == "target")
        self.assertIn("declared_cv_disagrees_with_samples", cell["processes"][0]["validation_failures"])
        self.assertFalse(cell["supported"])

    def test_unstable_source_retains_price_without_acceptance(self):
        source = rows()
        source[0] = row(0, samples=[20, 100, 100, 300])
        value = price(fit(source))
        self.assertIsNotNone(value["price_ns"])
        self.assertFalse(value["profile_accepted"])
        self.assertIn("one_or_more_within_process_cv_gates_failed", value["profile_gate_failures"])

    def test_three_process_minimum_not_rescued_by_many_samples(self):
        source = [row(0, samples=[100] * 1000), row(1, samples=[100] * 1000)]
        value = price(fit(source))
        self.assertEqual(value["price_ns"], 100)
        self.assertFalse(value["profile_accepted"])
        self.assertIn("fewer_than_three_process_records", value["profile_gate_failures"])

    def test_equal_sample_counts_required_without_resampling(self):
        source = rows()
        source[1] = row(1, samples=[100] * 5)
        result = fit(source)
        cell = next(cell for cell in result["families"]["conditional"]["cells"] if cell["metric"] == "target")
        self.assertEqual([item["sample_count"] for item in cell["processes"]], [4, 5, 4])
        self.assertEqual(cell["price_estimator"], "diagnostic_median_of_process_medians")
        self.assertIsNone(cell["pooled_median_ns"])
        self.assertFalse(cell["accepted"])

    def test_exact_half_ns_price_is_preserved_and_half_up_rounded(self):
        result = fit([row(number, samples=[100, 101]) for number in range(3)])
        value = price(result)
        self.assertEqual(value["price_ns"], 101)
        self.assertEqual(Fraction(**value["price_ns_exact"]), Fraction(201, 2))

    def test_process_ids_required_but_pid_reuse_not_false_rejection(self):
        source = rows()
        for item in source:
            item["summary"]["process_id"] = 123
        self.assertTrue(price(fit(source))["profile_accepted"])
        source[0]["summary"].pop("process_id")
        with self.assertRaises(m.UnknownCost):
            price(fit(source))

    def test_duplicate_job_is_not_another_process(self):
        source = rows()
        source[1]["job"]["id"] = source[0]["job"]["id"]
        with self.assertRaises(ValueError):
            fit(source)

    def test_bindings_timestamp_and_model_mutation_rejected(self):
        for bindings in ({}, {"file": "guessed"}, {"file": "A" * 64}):
            with self.assertRaises(ValueError):
                m.fit_model(rows(), bindings)
        with self.assertRaises(ValueError):
            m.fit_model(rows(), {"file": "a" * 64}, "2026-10-18T00:00:00")
        result = fit()
        result["families"]["conditional"]["cells"][0]["price_ns"] = 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            price(result)


class EvaluationTests(unittest.TestCase):
    def test_conditional_beats_constant_without_heldout_fit(self):
        model = fit(rows(value=100) + rows(condition="gpu_replay_load", value=200))
        before = copy.deepcopy(model)
        heldout = rows("heldout", value=100) + rows("heldout", "gpu_replay_load", 200)
        result = m.evaluate(model, heldout)
        self.assertEqual(model, before)
        conditional, constant = result["families"]["conditional"], result["families"]["constant"]
        self.assertEqual(conditional["accepted_count"], 4)
        self.assertEqual(conditional["worst_relative_error_known_only"], 0)
        self.assertEqual(constant["accepted_count"], 0)
        self.assertEqual(constant["worst_relative_error_known_only"], 0.5)
        self.assertFalse(result["heldout_used_for_fit"])
        self.assertFalse(result["causal_device_slowdown_proven"])

    def test_profile_rows_or_leaked_numerical_seeds_rejected(self):
        model = fit()
        with self.assertRaisesRegex(ValueError, "heldout-only"):
            m.evaluate(model, rows())
        source_seed = rows("heldout")
        source_seed[0]["job"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed sets overlap"):
            m.evaluate(model, source_seed)

    def test_nonlegacy_execution_boundaries_cannot_enter_heldout_evaluation(self):
        fitted = fit()
        for boundary in ("async_one_sync", "async_events", None, "unknown"):
            heldout = rows("heldout")
            heldout[0]["job"]["execution_boundary"] = boundary
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "diagnostic-only"):
                m.evaluate(fitted, heldout)
        heldout = rows("heldout")
        for item in heldout:
            item["job"]["execution_boundary"] = "legacy_double_sync"
        result = m.evaluate(fitted, heldout)
        self.assertEqual(result["families"]["conditional"]["accepted_count"], 2)

    def test_heldout_unknown_condition_retained(self):
        result = m.evaluate(fit(), rows("heldout", "unseen"))
        for family in result["families"].values():
            self.assertEqual(family["accepted_count"], 0)
            unknown = [item for item in family["cells"] if item["condition"] == "unseen"]
            self.assertEqual(len(unknown), 2)
            self.assertTrue(all(item["unknown_reason"] for item in unknown))
            self.assertTrue(all(item["relative_error"] is None for item in unknown))

    def test_heldout_failure_keeps_all_processes_and_no_error(self):
        heldout = rows("heldout", count=4)
        heldout[3]["summary"]["native_numeric_passed"] = False
        result = m.evaluate(fit(), heldout)
        for cell in result["families"]["conditional"]["cells"]:
            self.assertEqual(cell["heldout"]["process_count"], 4)
            self.assertIsNone(cell["relative_error"])
            self.assertFalse(cell["accepted"])

    def test_large_heldout_error_not_recalibrated_away(self):
        result = m.evaluate(fit(), rows("heldout", value=200))
        for cell in result["families"]["conditional"]["cells"]:
            self.assertEqual(cell["prediction_ns"], 100)
            self.assertEqual(cell["relative_error"], 0.5)
            self.assertFalse(cell["accepted"])
            self.assertEqual([item["relative_error"] for item in cell["per_process_errors"]], [0.5] * 3)

    def test_missing_heldout_cell_retained_not_dropped(self):
        model = fit(rows() + rows(condition="gpu_replay_load"))
        result = m.evaluate(model, rows("heldout"))
        absent = [cell for cell in result["families"]["conditional"]["cells"] if cell["condition"] == "gpu_replay_load"]
        self.assertEqual(len(absent), 2)
        self.assertTrue(all(cell["heldout"]["process_count"] == 0 for cell in absent))
        self.assertTrue(all(not cell["accepted"] for cell in absent))

    def test_heldout_within_and_between_process_cv_gates(self):
        heldout = rows("heldout")
        heldout[0] = row(0, "heldout", samples=[20, 100, 100, 300])
        heldout[2] = row(2, "heldout", samples=[300] * 4)
        result = m.evaluate(fit(), heldout)
        for cell in result["families"]["conditional"]["cells"]:
            self.assertIn("heldout:one_or_more_within_process_cv_gates_failed", cell["gate_failures"])
            self.assertIn("heldout:between_process_cv_gate_failed", cell["gate_failures"])

    def test_heldout_sample_count_mismatch_fails_even_with_same_price(self):
        heldout = [row(number, "heldout", samples=[100] * 5) for number in range(3)]
        result = m.evaluate(fit(), heldout)
        for cell in result["families"]["conditional"]["cells"]:
            self.assertEqual(cell["relative_error"], 0)
            self.assertIn("profile_heldout_sample_counts_differ", cell["gate_failures"])

    def test_mixed_bare_target_unknown_iteration_still_valid(self):
        source = [row(number, mode="mixed", instrumentation="bare", telemetry=False) for number in range(3)]
        heldout = [row(number, "heldout", mode="mixed", instrumentation="bare", telemetry=False) for number in range(3)]
        for item in source + heldout:
            item["summary"]["target_samples_ns"] = None
            item["summary"]["target_median_ns"] = None
            item["summary"]["target_cv"] = None
            item["summary"]["alignment_model_ready"] = False
        result = m.evaluate(fit(source), heldout)
        cells = {cell["metric"]: cell for cell in result["families"]["conditional"]["cells"]}
        self.assertFalse(cells["target"]["accepted"])
        self.assertIsNone(cells["target"]["relative_error"])
        self.assertTrue(cells["iteration"]["accepted"])


if __name__ == "__main__":
    unittest.main()
