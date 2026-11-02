"""Dual affinity contract tests; no hardware or subprocess execution."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent)]
import run_dual as d
import run_native as original
import verify_dual as verify
from test_run_native import payload, suite


def dual_payload(case, protocol):
    result = payload(case, protocol)
    result.update(dispatch_cpu_id=2, affinity_variant="dual_logical_cpu_explicit",
                  observed_affinity={"process_mask": 5, "main_thread_mask": 1, "main_thread_group": 0,
                                     "dispatch_worker_created": case["family"] == "ffn",
                                     "dispatch_thread_mask": 4 if case["family"] == "ffn" else None,
                                     "dispatch_thread_group": 0 if case["family"] == "ffn" else None})
    return result


class DualContracts(unittest.TestCase):
    def test_ffn_actual_readback_masks_required(self):
        doc = suite()
        case = doc["cases"][0]
        d.validate_native({"case": case, "seed": 17}, dual_payload(case, doc["protocol"]), doc, expected_dispatch_cpu_id=2)

    def test_attention_has_no_parallel_dispatch_worker(self):
        doc = suite()
        case = doc["cases"][-1]
        native = dual_payload(case, doc["protocol"])
        d.validate_native({"case": case, "seed": 17}, native, doc, expected_dispatch_cpu_id=2)
        native["observed_affinity"]["dispatch_thread_mask"] = 4
        with self.assertRaisesRegex(ValueError, "affinity"):
            d.validate_native({"case": case, "seed": 17}, native, doc, expected_dispatch_cpu_id=2)

    def test_requested_cpu_is_not_enough_without_actual_masks(self):
        doc = suite()
        case = doc["cases"][0]
        for field, value in (("process_mask", 1), ("main_thread_mask", 5), ("dispatch_thread_mask", 1), ("dispatch_thread_group", 1)):
            native = dual_payload(case, doc["protocol"])
            native["observed_affinity"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "affinity"):
                d.validate_native({"case": case, "seed": 17}, native, doc, expected_dispatch_cpu_id=2)

    def test_native_cpu_must_match_observed_child_flag(self):
        doc = suite()
        case = doc["cases"][0]
        with self.assertRaisesRegex(ValueError, "observed child"):
            d.validate_native({"case": case, "seed": 17}, dual_payload(case, doc["protocol"]), doc, expected_dispatch_cpu_id=3)

    def test_explicit_distinct_dispatch_cpu_required(self):
        for value in (None, "", "guess", "-1", "64", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                d.dispatch_cpu_id({"GGML_CALIBRATION_GPU_DISPATCH_CPU_ID": value}, 0)
        self.assertEqual(d.dispatch_cpu_id({"GGML_CALIBRATION_GPU_DISPATCH_CPU_ID": "2"}, 0), 2)

    def test_statistical_recipes_unchanged(self):
        for case in suite()["cases"]:
            self.assertEqual(d.calibration_recipes(case), original.calibration_recipes(case))
            self.assertEqual(d.observed_aliases(case), original.observed_aliases(case))
        self.assertIs(verify.r, d)


if __name__ == "__main__":
    unittest.main()
