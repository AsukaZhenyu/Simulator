"""Unit semantics and M2's independence from the legacy cost bridge."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from llm_infer_model.tensor.spec import InvalidInput
from llm_infer_model.units import seconds_to_ns


class UnitConversionTests(unittest.TestCase):
    def test_rounding_and_existing_numeric_coercion_are_preserved(self):
        for value, expected in ((2e-3, 2_000_000), (1.5e-9, 2), (2.5e-9, 2),
                                ("0.000000001", 1), (True, 1_000_000_000)):
            with self.subTest(value=value):
                self.assertEqual(seconds_to_ns(value, "duration"), expected)

    def test_invalid_and_unrepresentable_durations_keep_error_context(self):
        cases = ((None, "不是数值"), ("unknown", "不是数值"),
                 (float("nan"), "必须是有限值"), (float("inf"), "必须是有限值"),
                 (0, "必须为正"), (-1, "必须为正"),
                 (4e-10, "无法表达"), (5e-10, "无法表达"))
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(InvalidInput) as caught:
                    seconds_to_ns(value, "stage.duration_s")
                self.assertIn("stage.duration_s:", str(caught.exception))
                self.assertIn(message, str(caught.exception))

    def test_legacy_import_reexports_the_same_function(self):
        from llm_infer_model.tensor.layer_costs import seconds_to_ns as legacy
        self.assertIs(legacy, seconds_to_ns)

    def test_m2_costs_imports_when_legacy_bridge_is_blocked(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class BlockLegacyBridge(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "llm_infer_model.tensor.layer_costs":
            raise AssertionError("M2 must not import the legacy layer-cost bridge")
sys.meta_path.insert(0, BlockLegacyBridge())
from llm_infer_model.m2.costs import seconds_to_ns as m2_conversion
from llm_infer_model.units import seconds_to_ns
assert m2_conversion is seconds_to_ns
assert seconds_to_ns(1.5e-9, "duration") == 2
assert "llm_infer_model.tensor.layer_costs" not in sys.modules
print(json.dumps({"bridge_loaded": False, "conversion_module": seconds_to_ns.__module__}))
'''
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout),
                         {"bridge_loaded": False, "conversion_module": "llm_infer_model.units"})


if __name__ == "__main__":
    unittest.main()
