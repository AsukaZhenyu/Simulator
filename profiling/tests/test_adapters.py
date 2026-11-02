"""Check graph-derived shapes and profile queries without running hardware."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_suite_from_graph import extract_suite
import make_suite_from_graph as suite_generator
from query_profile import query


def graph_document(dtype_a="q4_K", dtype_b="q4_K"):
    return {"graphs": [{"id": "g1", "tensors": [
        {"id": "w1", "role": "parameter", "shape": [256, 512], "ggml_type": dtype_a,
         "name": "blk.3.ffn_gate.weight", "is_view": False},
        {"id": "w2", "role": "parameter", "shape": [256, 512], "ggml_type": dtype_b,
         "name": "blk.3.ffn_up.weight", "is_view": False}], "nodes": [
        {"id": "n1", "op": "MUL_MAT", "layer_id": 3, "input_tensor_ids": ["w1", "x"]},
        {"id": "n2", "op": "MUL_MAT", "layer_id": 3, "input_tensor_ids": ["w2", "x"]}]}]}


class GraphSuiteTests(unittest.TestCase):
    def test_identical_shapes_share_measurement_but_retain_both_tensor_names(self):
        suite = extract_suite(graph_document(), graph_id="g1", layer_ids={3}, tokens=[1, 32])
        self.assertEqual(len(suite["compute_cases"]), 2)
        one = suite["compute_cases"][0]
        self.assertEqual((one["k"], one["m"]), (256, 512))
        self.assertEqual(len(one["source"]["tensor_names"]), 2)

    def test_mixed_quantization_does_not_collapse_to_model_file_label(self):
        suite = extract_suite(graph_document(dtype_b="q6_K"), graph_id="g1", layer_ids={3}, tokens=[1])
        self.assertEqual({case["dtype"] for case in suite["compute_cases"]}, {"q4_k", "q6_k"})

    def test_f32_override_is_explicit_and_retains_original_precision(self):
        suite = extract_suite(graph_document(dtype_b="q6_K"), graph_id="g1", layer_ids={3}, tokens=[1], dtype_mode="f32")
        self.assertEqual(len(suite["compute_cases"]), 1)
        case = suite["compute_cases"][0]
        self.assertEqual(case["dtype"], "f32")
        self.assertEqual(set(case["source"]["original_dtypes"]), {"q4_k", "q6_k"})

    def test_unsupported_quantization_is_recorded_rather_than_substituted(self):
        suite = extract_suite(graph_document(dtype_b="iq2_xs"), graph_id="g1", layer_ids={3}, tokens=[1])
        self.assertEqual(len(suite["compute_cases"]), 1)
        self.assertIn("unsupported", suite["skipped"][0]["reason"])

    def test_layer_and_graph_selection_cannot_silently_fall_back(self):
        for graph_id, layers in (("missing", {3}), ("g1", {4})):
            with self.subTest(graph_id=graph_id, layers=layers), self.assertRaises(ValueError):
                extract_suite(graph_document(), graph_id=graph_id, layer_ids=layers, tokens=[1])

    def test_generated_suite_preserves_exact_token_grid_and_lf_bytes(self):
        with tempfile.TemporaryDirectory(prefix="profiling-graph-suite-") as folder:
            graph_path = Path(folder) / "graph.json"
            graph_path.write_bytes(json.dumps(graph_document()).encode("utf-8"))
            output = Path(folder) / "cases.json"
            with mock.patch.object(sys, "argv", ["make_suite_from_graph", "--graph", str(graph_path),
                "--graph-id", "g1", "--layer-id", "3", "--tokens", "1,3", "--output", str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(suite_generator.main(), 0)
            contents = output.read_bytes()
            self.assertNotIn(b"\r\n", contents)
            self.assertTrue(contents.endswith(b"\n"))
            suite = json.loads(contents)
            self.assertEqual({case["n"] for case in suite["compute_cases"]}, {1, 3})


class QueryTests(unittest.TestCase):
    def write_profile(self, path, status="measured"):
        (path / "manifest.json").write_text(json.dumps({"profile_id": "test", "environment": {}, "software": {}}))
        (path / "measurements.json").write_text(json.dumps([{
            "case_id": "projection__cuda0", "status": status, "signature": {"weight_dtype": "q4_k"},
            "raw_output": "raw/001.json", "native": {"wall_ns": [100, 101], "workspace_bytes": None,
                "workspace_source": "unknown_backend_private_allocations"}}]))

    def test_unknown_workspace_and_rounding_survive_query(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_profile(path)
            result = query(path, "projection__cuda0")
            self.assertEqual(result["duration_ns"], 101)
            self.assertIsNone(result["workspace_bytes"])
            self.assertFalse(result["capacity_verified"])

    def test_failed_or_missing_case_cannot_be_used_as_a_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.write_profile(path, status="failed")
            for case in ("projection__cuda0", "other"):
                with self.subTest(case=case), self.assertRaises(ValueError):
                    query(path, case)


if __name__ == "__main__":
    unittest.main()
