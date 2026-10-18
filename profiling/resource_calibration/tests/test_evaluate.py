"""Frozen-model integrity checks without native launches or device access."""
from dataclasses import replace
from fractions import Fraction
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling.resource_calibration import evaluate as workflow
from profiling.resource_calibration.functional import CalibrationRow, FitConfig, Hardware, Operation, ResourceState


class FrozenWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.source = "a" * 64
        self.hardware = Hardware(self.source, 1000, 1000)
        self.config = FitConfig(regularization=Fraction(1, 1000), min_training_runs=3)
        self.op = Operation("pv", "gpu", 100, (("shape", "8x8"),))
        self.rows = tuple(CalibrationRow(f"r{repeat}-{i}", "train", f"run{repeat}", repeat, i * 1000,
                                        Fraction(100 + 100 * c + 200 * g + d + v), i * 1000 + 900,
                                        self.op, ResourceState(c, g, d, v), self.source, "fixture-v1",
                                        "windows_qpc", "offered_load_controlled", True)
                          for repeat in range(3) for i, (c, g, d, v) in enumerate(itertools.product((0, 1), (0, 1), (0, 100), (0, 100))))
        self.coverage = {"fixture": True}
        workflow.run.save(self.root / "manifest.json", {"source_inventory_sha256": self.source,
                                                       "protocol": {"fit_config": self.config.to_dict()}})
        workflow.run.save(self.root / "train-rows.json", [{"fixture": True}])
        self.path = self.root / "model.json"

    def freeze(self):
        with patch.object(workflow, "calibration_rows", return_value=(self.rows, self.coverage, self.hardware)):
            return workflow.freeze(self.root, self.path)

    def test_freeze_and_independent_training_reconstruction(self):
        artifact = self.freeze()
        with patch.object(workflow, "calibration_rows", return_value=(self.rows, self.coverage, self.hardware)):
            loaded, model = workflow.load_frozen(self.root, self.path)
        self.assertEqual(artifact, loaded)
        self.assertEqual(len(model.training_run_ids), 3)
        self.assertFalse(loaded["hardware_accuracy_accepted"])

    def test_arbitrary_model_file_is_rejected(self):
        workflow.run.save(self.path, {})
        with self.assertRaisesRegex(ValueError, "not a frozen"):
            workflow.load_frozen(self.root, self.path)

    def test_training_file_change_is_rejected(self):
        self.freeze()
        with (self.root / "train-rows.json").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "bindings"):
            workflow.load_frozen(self.root, self.path, verify_training=False)

    def test_changed_phase_summary_is_rejected(self):
        self.freeze()
        with patch.object(workflow, "calibration_rows", return_value=(self.rows, {"changed": True}, self.hardware)):
            with self.assertRaisesRegex(ValueError, "reconstructed"):
                workflow.load_frozen(self.root, self.path)

    def test_self_consistent_coefficient_tamper_is_rejected_by_refit(self):
        artifact = self.freeze()
        from profiling.resource_calibration.functional import ResourceFunctional
        model = ResourceFunctional.from_dict(artifact["model"])
        surface = model.surfaces[0]
        bad = replace(model, surfaces=(replace(surface, coefficients_ns=(surface.coefficients_ns[0] + 1, *surface.coefficients_ns[1:])),))
        artifact["model"] = bad.to_dict()
        artifact["model_sha256"] = bad.model_sha256
        self.path.write_text(json.dumps(artifact), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "training-only reconstruction"):
            workflow.load_frozen(self.root, self.path, verify_training=False)

    def test_no_fit_after_any_heldout_barrier(self):
        workflow.run.save(self.root / "heldout-barrier.json", {})
        with self.assertRaisesRegex(ValueError, "cannot fit/refit"):
            workflow.freeze(self.root, self.path)

    def test_no_fit_after_heldout_rows(self):
        workflow.run.save(self.root / "heldout-rows.json", [])
        with self.assertRaisesRegex(ValueError, "cannot fit/refit"):
            workflow.freeze(self.root, self.path)

    def test_artifact_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            workflow._verified_artifacts(self.root, {"artifacts_sha256": {"../not-owned": "b" * 64}})

    def test_exact_half_nanosecond_aggregation_and_clock_conversion(self):
        self.assertEqual(workflow.median_exact([100, 101]), Fraction(201, 2))
        self.assertEqual(workflow.tick_ns(123, {"frequency_hz": 10_000_000}), 12300)
        with self.assertRaises(ValueError):
            workflow.tick_ns(True, {"frequency_hz": 10_000_000})

    def test_capacity_must_be_observed_and_consistent(self):
        for values in ([], [1000, 2000], [True], [0]):
            with self.assertRaises(ValueError):
                workflow._capacity(values, "fixture")


if __name__ == "__main__":
    unittest.main()
