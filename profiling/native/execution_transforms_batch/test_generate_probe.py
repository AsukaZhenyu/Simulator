"""Generator contract checks; these never execute a hardware probe."""
from pathlib import Path
import tempfile
import unittest

from generate_probe import HERE, generate


class BatchProbeGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.context = HERE.parent / "execution_transforms_dual/compute_context.cpp"
        self.helper = HERE.parent / "matched_mechanisms/compute.cpp"

    def test_deterministic_generation_has_raw_batch_units(self):
        first, manifest = generate(self.context, self.helper)
        second, second_manifest = generate(self.context, self.helper)
        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        source = first.decode("utf-8")
        self.assertIn('"completed_work_batch_total', source)
        self.assertIn('batch_wall_ns', source)
        self.assertNotIn(r'\"wall_ns\":{', source)
        self.assertEqual(source.count("return measure(o,"), 1)
        self.assertEqual(source.count("=measure_batched(o,"), 20)
        self.assertIn('times["hybrid/preload_"+std::string(name)+"_h2d"]', source)
        self.assertFalse(manifest["raw_times_divided"])
        self.assertFalse(manifest["graph_reference_precision_or_tolerance_changed"])

    def test_context_drift_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = Path(temporary) / "context.cpp"
            context.write_bytes(self.context.read_bytes() + b"// upstream drift\n")
            with self.assertRaisesRegex(ValueError, "source identity drift"):
                generate(context, self.helper)

    def test_helper_drift_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "helper.cpp"
            helper.write_bytes(self.helper.read_bytes() + b"// upstream drift\n")
            with self.assertRaisesRegex(ValueError, "source identity drift"):
                generate(self.context, helper)


if __name__ == "__main__":
    unittest.main()
