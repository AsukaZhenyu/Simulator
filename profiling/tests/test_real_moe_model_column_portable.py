"""The model column's portable inputs and its condition mean, without hardware."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from profiling.real_moe_validation import condition_mean, prepare_portable


class ShippedInputs(unittest.TestCase):
    def test_pinned_model_is_the_published_one(self):
        model = prepare_portable.pinned_model("olmoe-1b-7b")
        self.assertEqual(model["bytes"], 4213513024)
        self.assertEqual(model["sha256"], "3bd9ec48045f526508956ef663c5c11dbb3ef791a06f4e10a6be88379c40e1fe")

    def test_guards_derive_from_the_shipped_policy(self):
        policy = json.loads(prepare_portable.SHIPPED_POLICY.read_text(encoding="utf-8"))
        guards = prepare_portable.budget(policy)
        # The published cohort froze these three numbers; they follow from the
        # model bytes and the policy's own declared budget, not from a machine.
        self.assertEqual(guards["native_guard_bytes"], 5824125760)
        self.assertEqual(guards["parent_preflight_bytes"], 6076332864)
        self.assertEqual(guards["resident_peak_limit_bytes"], 5287254848)

    def test_shipped_suite_and_template_are_present(self):
        suite = json.loads(prepare_portable.SHIPPED_SUITE.read_text(encoding="utf-8"))
        self.assertTrue(suite["requests"])
        template = json.loads(prepare_portable.SHIPPED_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(template["model_bytes"], 4213513024)

    def test_template_rewrites_only_the_device_fields(self):
        template = json.loads(prepare_portable.SHIPPED_TEMPLATE.read_text(encoding="utf-8"))
        model = dict(path="D:/models/olmoe.gguf", sha256="a" * 64, bytes=4213513024)
        out = prepare_portable.template_for(model, template)
        self.assertEqual(out["model_path"], "D:/models/olmoe.gguf")
        self.assertEqual(out["model_sha256"], "a" * 64)
        for key in template:
            if key not in ("model_path", "model_sha256", "model_bytes"):
                self.assertEqual(out[key], template[key], key)

    def test_a_model_of_the_wrong_size_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "olmoe.gguf"
            path.write_bytes(b"not the model")
            with self.assertRaises(Exception):
                prepare_portable.verify_model(path, prepare_portable.pinned_model("olmoe-1b-7b"))


class ConditionMean(unittest.TestCase):
    """The aggregation that turns an evaluation into one row per condition."""

    def evaluation(self):
        def row(placement, request, repeat, measured, predicted, status="evaluated"):
            return dict(placement=placement, request_id=request, repeat=repeat, status=status,
                        measured_ns=dict(total=measured, prefill=measured // 2,
                                         decode_with_sampling=measured - measured // 2),
                        predicted_ns=dict(total=predicted, prefill=predicted // 2,
                                          decode_with_sampling=predicted - predicted // 2),
                        APE=dict(total=abs(predicted / measured - 1),
                                 prefill=abs((predicted // 2) / (measured // 2) - 1),
                                 decode_with_sampling=abs((predicted - predicted // 2)
                                                          / (measured - measured // 2) - 1)),
                        extrapolated_windows=0)
        rows = [row("cpu", "a", 0, 1000, 900), row("cpu", "a", 1, 2000, 900),
                row("cpu", "b", 0, 1000, 1000), row("cpu", "b", 1, 1000, 1000),
                # One repeat of this condition failed its endpoint: excluded, not averaged.
                row("hybrid", "c", 0, 1000, 800, status="workload_or_numerical_mismatch"),
                row("hybrid", "c", 1, 1000, 800)]
        return dict(rows=rows, target_APE=0.1, parameters_refitted=False)

    def test_condition_error_is_scored_against_the_mean_not_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "INDEPENDENT_EVALUATION.json").write_text(
                json.dumps(self.evaluation()), encoding="utf-8")
            # Two repeats at 1000 and 2000 ns against a 900 ns prediction: the
            # condition mean is 1500, so the error is 40 %, not the 35 % average
            # of the two per-launch errors.
            condition_mean.main(["--out", str(out)])
            result = json.loads((out / "CONDITION_MEAN_EVALUATION.json").read_text(encoding="utf-8"))
        conditions = {(c["placement"], c["request_id"]): c for c in result["conditions"]}
        self.assertAlmostEqual(conditions[("cpu", "a")]["APE"]["total"], 0.4, places=12)
        self.assertAlmostEqual(conditions[("cpu", "b")]["APE"]["total"], 0.0, places=12)
        self.assertEqual(conditions[("hybrid", "c")]["status"], "incomplete")
        self.assertEqual(result["declared_conditions"], 3)
        self.assertEqual(result["complete_endpoint_conditions"], 2)
        self.assertAlmostEqual(result["mean_APE"]["total"], 0.2, places=12)
        self.assertAlmostEqual(result["worst_APE"]["total"], 0.4, places=12)


if __name__ == "__main__":
    unittest.main()
