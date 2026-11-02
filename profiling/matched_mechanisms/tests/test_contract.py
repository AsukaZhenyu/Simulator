import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import experiment as e


def suite():
    return e.document(HERE / "suite.json")


def rows():
    c = suite()["cases"][0]
    result = []
    for split in ("profile", "heldout"):
        for repeat in range(2):
            metrics = {"cpu/" + x: [100, 100, 100] for x in ("gate", "up", "activation", "down")}
            metrics["cpu/serial"] = [410, 410, 410]
            result.append({"id": split + str(repeat), "case": c, "split": split,
                           "status": "measured", "metrics": metrics})
    return result


class Contracts(unittest.TestCase):
    def test_valid_explicit_suite_detached(self):
        original = suite(); parsed = e.validate_suite(original)
        parsed["cases"][0]["hidden"] = 1
        self.assertEqual(original["cases"][0]["hidden"], 32)

    def test_every_suite_field_required(self):
        for key in suite():
            with self.subTest(key=key):
                doc = suite(); doc.pop(key)
                with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_unknown_fields_refused(self):
        doc = suite(); doc["guess"] = 1
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_no_boolean_shape_or_protocol(self):
        for target, key in (("case", "hidden"), ("protocol", "samples"), ("quality", "max_within_process_cv")):
            doc = suite(); obj = doc["cases"][0] if target == "case" else doc["compute_protocol"] if target == "protocol" else doc["quality"]
            obj[key] = True
            with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_profile_and_heldout_disjoint(self):
        doc = suite(); doc["heldout_seeds"] = doc["profile_seeds"]
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_no_dtype_substitution(self):
        doc = suite(); doc["cases"][0]["dtype"] = "q4_k"
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_attention_hybrid_refused(self):
        doc = suite(); doc["cases"][3]["backend"] = "hybrid"
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_gqa_geometry_refused(self):
        doc = suite(); doc["cases"][3]["kv_heads"] = 3
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_cpu_mask_and_thread_contract_not_silently_changed(self):
        doc = suite(); doc["compute_protocol"]["threads"] = 2
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_no_cuda_graph_default(self):
        doc = suite(); doc["compute_protocol"]["cuda_graphs"] = "auto"
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_safe_unique_case_ids(self):
        for value in ("../outside", "", suite()["cases"][1]["id"]):
            doc = suite(); doc["cases"][0]["id"] = value
            with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_transfer_engine_explicit(self):
        doc = suite(); doc["transfers"][0]["mode"] = "automatic"
        with self.assertRaises(ValueError): e.validate_suite(doc)

    def test_zero_or_float_sample_refused(self):
        for sample in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): e.stats([1, 2, sample])

    def test_fixture_is_exact_f32_storage(self):
        payload = {"source_values": {"w": [.1, -.2]}}
        result = e.fixture_hashes(payload)["w"]
        self.assertEqual(result["storage_bytes"], 8)
        self.assertEqual(result, e.fixture_hashes(copy.deepcopy(payload))["w"])

    def test_nonfinite_fixture_refused(self):
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError): e.fixture_hashes({"source_values": {"w": [value]}})

    def test_fit_uses_only_profile(self):
        data = rows(); old = e.fit(data)
        for row in data:
            if row["split"] == "heldout": row["metrics"]["cpu/serial"] = [100000] * 3
        self.assertEqual(old, e.fit(data))

    def test_component_model_includes_activation_and_all_three_matrices(self):
        model = e.fit(rows()); serial = next(x for x in model["entries"] if x["metric"] == "cpu/serial")
        self.assertEqual(serial["component_sum_prediction_ns"], 400)
        self.assertEqual(serial["serial_minus_component_sum_ns"], 10)
        self.assertFalse(model["control_overhead_separately_identified"])

    def test_path_model_includes_both_input_and_return(self):
        self.assertEqual(e.components("hybrid/cpu_rows", "ffn"), ["hybrid/d2h/input", "cpu/gate", "cpu/up", "cpu/activation", "cpu/down", "hybrid/h2d/output"])

    def test_path_weights_all_three_included(self):
        self.assertEqual(e.components("hybrid/gpu_weights", "ffn")[:3], ["hybrid/h2d/gate_weight", "hybrid/h2d/up_weight", "hybrid/h2d/down_weight"])

    def test_good_declared_exact_model_passes(self):
        report = e.evaluate(rows(), e.fit(rows()), suite())
        self.assertTrue(report["all_declared_metrics_calibrated"])
        self.assertFalse(report["gpu_capacity_proved"])

    def test_large_between_process_drift_fails_even_equal_pooled_median(self):
        data = rows(); data[0]["metrics"]["cpu/serial"] = [100]*3; data[1]["metrics"]["cpu/serial"] = [720]*3
        report = e.evaluate(data, e.fit(data), suite())
        serial = next(x for x in report["results"] if x["metric"] == "cpu/serial")
        self.assertFalse(serial["gates"]["between_process"])
        self.assertFalse(serial["calibrated"])

    def test_heldout_error_gate_preserved(self):
        data = rows()
        for row in data:
            if row["split"] == "heldout": row["metrics"]["cpu/serial"] = [1000]*3
        serial = next(x for x in e.evaluate(data, e.fit(data), suite())["results"] if x["metric"] == "cpu/serial")
        self.assertFalse(serial["gates"]["heldout_error"])
        self.assertGreater(serial["heldout_relative_error"], .5)

    def test_failed_native_stays_failure(self):
        data = rows(); data[0]["status"] = "failed"
        report = e.evaluate(data, e.fit(data), suite())
        self.assertFalse(report["numeric_execution_valid"])
        self.assertFalse(report["all_declared_metrics_calibrated"])

    def test_missing_heldout_no_calibration_claim(self):
        data = rows()[:2]
        report = e.evaluate(data, e.fit(data), suite())
        self.assertTrue(all(not r["calibrated"] for r in report["results"]))

    def test_command_every_geometry_and_protocol_explicit(self):
        doc = suite(); cmd = e.compute_command("probe", doc["cases"][0], doc["compute_protocol"], 17)
        for flag in ("--hidden", "--intermediate", "--history", "--dtype", "--cpu-id", "--cuda-graphs", "--seed"):
            self.assertIn(flag, cmd)


if __name__ == "__main__":
    unittest.main()
