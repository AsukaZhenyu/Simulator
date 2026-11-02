"""Measured MoE compatibility delegates raw work without eager profiling imports."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from llm_infer_model.m2 import moe_ffn_contract, moe_measured
from llm_infer_profile import moe_measurements
from llm_infer_profile.statistics import statistics_ns

ROOT = Path(__file__).resolve().parents[2]


class MeasuredMoeOwnershipTests(unittest.TestCase):
    def test_model_import_does_not_load_profiling(self):
        code = ("import sys; sys.path.insert(0,sys.argv[1]); "
                "import llm_infer_model.m2; "
                "assert not any(n=='llm_infer_profile' or n.startswith('llm_infer_profile.') for n in sys.modules)")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "modeling")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_statistics_wrapper_delegates_to_profiling(self):
        with mock.patch("llm_infer_profile.statistics.statistics_ns", return_value={"owned": "profiling"}) as spy:
            self.assertEqual(moe_ffn_contract.statistics_ns([1, 2, 3]), {"owned": "profiling"})
        spy.assert_called_once_with([1, 2, 3])

    def test_compile_wrapper_preserves_explicit_unstable_option(self):
        with mock.patch.object(moe_measurements, "compile_measured_ffn", return_value="owned") as spy:
            self.assertEqual(moe_measured.compile_measured_ffn({"raw": 1}, {"hardware": 2}, allow_unstable=True), "owned")
        spy.assert_called_once_with({"raw": 1}, {"hardware": 2}, allow_unstable=True)

    def test_private_port_compatibility_delegates(self):
        port = dict(ne=[2, 3, 1, 1], category="weight", dtype="f32", bytes=24)
        with mock.patch.object(moe_measurements, "_physical_tensor", wraps=moe_measurements._physical_tensor) as spy:
            tensor = moe_measured._physical_tensor("w", port, initial=True)
        spy.assert_called_once_with("w", port, initial=True)
        self.assertEqual(tensor["storage_bytes"], 24)

    def test_existing_raw_conversion_matches_compatibility_result(self):
        path = ROOT / "mapping" / "tests" / "test_moe_measured.py"
        specification = importlib.util.spec_from_file_location("_ownership_moe_fixture", path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        direct = moe_measurements.compile_measured_ffn(module.protocol_fixture(), module.hardware_fixture())
        compatibility = moe_measured.compile_measured_ffn(module.protocol_fixture(), module.hardware_fixture())
        self.assertEqual(direct.scenario_document, compatibility.scenario_document)
        self.assertEqual(direct.bindings, compatibility.bindings)
        self.assertEqual(moe_measured.replay_measured_ffn(direct).status, "valid")

    def test_model_contract_has_no_statistics_import_or_estimation_body(self):
        tree = ast.parse(Path(moe_ffn_contract.__file__).read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "statistics" for alias in node.names)
                             for node in ast.walk(tree)))
        wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "statistics_ns")
        self.assertEqual(len(wrapper.body), 3)  # docstring, lazy import, forwarding return


class TimingSummaryCompatibilityTests(unittest.TestCase):
    def test_odd_and_even_medians_preserve_historical_schema(self):
        self.assertEqual(statistics_ns([100, 101, 102])["median_ns"], 101)
        self.assertEqual(statistics_ns([100, 101, 102, 103])["median_ns"], 101.5)

    def test_outliers_retained_and_input_unchanged(self):
        values = [1, 1, 1000]
        before = values[:]
        result = statistics_ns(values)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["max_ns"], 1000)
        self.assertEqual(values, before)

    def test_invalid_or_missing_samples_rejected(self):
        for values in ([1, 2], [1, 2, True], [1, 2, 3.0], [0, 1, 2], (1, 2, 3), None):
            with self.subTest(values=values), self.assertRaises(ValueError):
                statistics_ns(values)


if __name__ == "__main__":
    unittest.main()
