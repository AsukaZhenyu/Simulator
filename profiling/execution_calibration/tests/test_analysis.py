from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis as a


QUALITY = {"max_within_process_cv": .15, "max_between_process_cv": .2,
           "max_heldout_relative_error": .2}
BOOTSTRAP = {"iterations": 100, "seed": 7}
RECIPES = {"test": {"gpu/resident": {"sum": ["gpu/a", "copy"]},
                     "hybrid/resident": {"sum": [{"max": ["cpu/a", "gpu/b"]}, "copy", "sum"]}}}


def rows():
    result = []
    for split, seed in (("profile", 17), ("heldout", 29)):
        for repeat in range(2):
            values = {"gpu/a": 100, "cpu/a": 50, "gpu/b": 60, "copy": 10, "sum": 5,
                      "gpu/resident": 110, "hybrid/resident": 75}
            result.append({"id": f"{split}-{repeat}", "case": {"id": "test"},
                           "split": split, "seed": seed, "repeat": repeat,
                           "status": "measured", "metrics": {k: [v] * 5 for k, v in values.items()},
                           "raw_path": "raw.json", "raw_sha256": "abc",
                           "checks": {"reference": {"passed": True}}, "workspaces": {"gpu": None}})
    return result


def report(data):
    return a.evaluate(data, a.freeze_model(data, RECIPES), QUALITY, BOOTSTRAP,
                      scope="test", frozen_before_heldout=True)


class AnalysisTests(unittest.TestCase):
    def test_holdout_cannot_change_model(self):
        data = rows()
        before = a.freeze_model(data, RECIPES)
        for row in data:
            if row["split"] == "heldout":
                row["metrics"]["gpu/a"] = [999999] * 5
        self.assertEqual(before, a.freeze_model(data, RECIPES))

    def test_strategy_total_not_a_component_price(self):
        with self.assertRaisesRegex(ValueError, "strategy total"):
            a.freeze_model(rows(), {"test": {"gpu/resident": "gpu/resident"}})

    def test_joint_profile_changes_do_not_change_prediction(self):
        data = rows()
        old = a.freeze_model(data, RECIPES)
        for row in data:
            if row["split"] == "profile":
                row["metrics"]["gpu/resident"] = [20000] * 5
                row["metrics"]["hybrid/resident"] = [10000] * 5
        self.assertEqual(old, a.freeze_model(data, RECIPES))

    def test_actual_parallel_recipe_and_sum(self):
        result = report(rows())
        self.assertEqual([r["predicted_ns"] for r in result["results"]], [110, 75])
        self.assertTrue(result["all_declared_strategies_calibrated"])

    def test_heldout_relative_error_uses_heldout_joint(self):
        data = rows()
        for row in data:
            if row["split"] == "heldout":
                row["metrics"]["hybrid/resident"] = [150] * 5
        result = report(data)["results"][1]
        self.assertEqual(result["absolute_error_ns"], 75)
        self.assertEqual(result["absolute_relative_error"], .5)
        self.assertFalse(result["calibrated"])

    def test_between_process_drift_fails_small_pooled_error(self):
        data = rows()
        data[0]["metrics"]["gpu/a"] = [50] * 5
        data[1]["metrics"]["gpu/a"] = [150] * 5
        result = report(data)["results"][0]
        self.assertEqual(result["absolute_error_ns"], 0)
        self.assertFalse(result["gates"]["between_process_cv"])
        self.assertFalse(result["calibrated"])

    def test_within_process_noise_fails(self):
        data = rows()
        data[0]["metrics"]["copy"] = [1, 1, 10, 100, 100]
        self.assertFalse(report(data)["results"][0]["gates"]["within_process_cv"])

    def test_single_process_cannot_pass_between_gate(self):
        data = [r for r in rows() if r["repeat"] == 0]
        self.assertTrue(all(not r["calibrated"] for r in report(data)["results"]))

    def test_missing_component_unknown_not_zero(self):
        data = rows()
        for row in data:
            row["metrics"].pop("gpu/a")
        result = report(data)["results"][0]
        self.assertIsNone(result["predicted_ns"])
        self.assertEqual(result["missing_component_metrics"], ["gpu/a"])

    def test_missing_heldout_target_unknown(self):
        data = rows()
        for row in data:
            if row["split"] == "heldout":
                row["metrics"].pop("gpu/resident")
        result = report(data)["results"][0]
        self.assertEqual(result["unknown_reason"], "missing heldout strategy measurements")

    def test_failed_native_case_is_not_silently_ignored(self):
        data = rows()
        data[0]["status"] = "failed"
        result = report(data)
        self.assertFalse(result["numeric_execution_valid"])
        self.assertFalse(result["all_declared_strategies_calibrated"])
        self.assertTrue(all(not r["gates"]["numeric_and_process_coverage"] for r in result["results"]))

    def test_frozen_before_holdout_required(self):
        data = rows()
        result = a.evaluate(data, a.freeze_model(data, RECIPES), QUALITY, BOOTSTRAP,
                            scope="test", frozen_before_heldout=False)
        self.assertFalse(result["all_declared_strategies_calibrated"])

    def test_ranking_and_selection_regret(self):
        result = report(rows())
        ranking = result["rankings"][0]
        self.assertEqual(ranking["correct_decisive_pairs"], 1)
        self.assertEqual(ranking["selection_regret_relative"], 0)
        self.assertTrue(ranking["latency_quality_accepted"])

    def test_no_ranking_across_output_boundary(self):
        data = rows()
        result = a.evaluate(data, a.freeze_model(data, RECIPES), QUALITY, BOOTSTRAP,
                            scope="test", frozen_before_heldout=True,
                            comparison_groups={"test": {"gpu/resident": "vram", "hybrid/resident": "dram"}})
        self.assertEqual(result["rankings"], [])

    def test_outputs_preserve_all_raw_samples_and_unknown_gpu(self):
        data = rows()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            a.write_outputs(output, report(data), a.freeze_model(data, RECIPES), data)
            text = (output / "raw-timing-samples.csv").read_text(encoding="utf-8")
            self.assertEqual(len(text.splitlines()), 1 + 4 * 7 * 5)
            self.assertIn('"gpu": null', (output / "numerical-and-workspace-evidence.json").read_text(encoding="utf-8"))
            for path in output.iterdir():
                self.assertNotIn(b"\r\n", path.read_bytes())
            with self.assertRaises(ValueError):
                a.write_outputs(output, report(data), a.freeze_model(data, RECIPES), data)

    def test_invalid_timing_and_expression_refused(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                a.stats([10, 10, value])
        for expression in ({"guess": ["a"]}, {"max": []}, ["a"], {}):
            with self.assertRaises(ValueError):
                a.terms(expression)


if __name__ == "__main__":
    unittest.main()
