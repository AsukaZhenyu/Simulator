"""Legacy prediction ownership and independent arithmetic; no hardware or fitting."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

from llm_infer_model.performance import legacy_composition, legacy_expressions, legacy_trace

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


execution = load("_ownership_execution_analysis", ROOT / "profiling/execution_calibration/analysis.py")
composition = load("_ownership_composition_fit", ROOT / "profiling/composition_calibration/model.py")
trace = load("_ownership_trace_extraction", ROOT / "profiling/accuracy_suite/trace_model.py")
shape = load("_ownership_shape_folds", ROOT / "profiling/accuracy_suite/shape_models.py")


def ffn_case():
    return dict(id="synthetic", family="ffn", backend="hybrid", dtype="f32", hidden=8,
                intermediate=16, cpu_intermediate=4, tokens=1, history=0,
                heads=0, kv_heads=0, head_dim=0)


def frozen_trace():
    costs = {"main:dispatch:gap_before": 0, "main:dispatch:duration": 5,
             "main:cpu:gap_before": 5, "main:cpu:duration": 20,
             "main:join:gap_before": 5, "main:join:completion_overhead": 10,
             "main:copy:gap_before": 5, "main:copy:duration": 10,
             "main:tail": 5, "worker_start_after_dispatch_begin": 3,
             "dispatch:gpu:duration": 77}
    return dict(schema=legacy_trace.SCHEMA, trained_on="profile_only",
                heldout_used_for_fitting=False, timing_unit="host_completed_work_nanoseconds",
                instrumentation_overhead_subtracted=False,
                paths=[dict(case_id="synthetic", strategy="fork_join",
                            program=dict(main=["dispatch", "cpu", "join", "copy"], dispatch=["gpu"]),
                            costs={name: {"median_ns": value} for name, value in costs.items()})])


class LegacyPredictionOwnershipTests(unittest.TestCase):
    def test_execution_sum_max_is_model_owned_with_independent_result(self):
        expression = {"sum": ["prepare", {"max": ["cpu", "gpu"]}, "finish"]}
        prices = dict(prepare=3, cpu=7, gpu=11, finish=5)
        self.assertEqual(legacy_expressions.predict(expression, prices), 19)
        self.assertEqual(execution.predict(expression, prices), 19)
        with mock.patch.object(legacy_expressions, "predict", return_value=123) as spy:
            self.assertEqual(execution.predict(expression, prices), 123)
        spy.assert_called_once_with(expression, prices)

    def test_expression_validation_also_delegates(self):
        with mock.patch.object(legacy_expressions, "terms", return_value={"owned"}) as spy:
            self.assertEqual(execution.terms("x"), {"owned"})
        spy.assert_called_once_with("x")
        with self.assertRaises(ValueError):
            legacy_expressions.predict({"min": ["x"]}, {"x": 1})

    def test_composition_hand_calculation_and_model_owned_features(self):
        case = ffn_case()
        prices = {name: 1 for name in legacy_composition.component_metrics(case)}
        frozen = dict(schema=legacy_composition.SCHEMA,
            feature_recipe_version=legacy_composition.FEATURE_RECIPE_VERSION,
            feature_names=list(legacy_composition.FEATURE_NAMES),
            coefficients=dict(zip(legacy_composition.FEATURE_NAMES, (2, .25, 7))))
        # Four CPU/GPU stages, copy and merge: max(4,4)+1+1=6 us.
        # Correction: five boundaries*2 + critical 4*.25 = 11 us.
        prediction = legacy_composition.predict(case, "hybrid/resident", prices, frozen)
        self.assertEqual(prediction["baseline_us"], 6)
        self.assertEqual(prediction["correction_us"], 11)
        self.assertEqual(prediction["predicted_ns"], 17_000)
        self.assertIs(composition.recipe, legacy_composition.recipe)
        self.assertIs(composition._features, legacy_composition._features)
        self.assertIs(composition._case, legacy_composition._case)
        self.assertIs(composition._evaluate, legacy_composition._evaluate)

    def test_composition_predict_wrapper_preserves_call_and_patch_semantics(self):
        arguments = (ffn_case(), "gpu/resident", {}, {})
        with mock.patch.object(legacy_composition, "predict", return_value={"owned": True}) as spy:
            self.assertEqual(composition.predict(*arguments), {"owned": True})
        spy.assert_called_once_with(*arguments)

    def test_dynamic_shape_loader_uses_model_helpers_and_profile_fit(self):
        self.assertIs(shape.shared.recipe, legacy_composition.recipe)
        self.assertIs(shape.shared._features, legacy_composition._features)
        self.assertNotEqual(shape.shared.fit.__module__, legacy_composition.__name__)
        with mock.patch.object(legacy_composition, "predict", return_value={"owned": True}) as spy:
            self.assertEqual(shape.shared.predict(ffn_case(), "gpu/resident", {}, {}), {"owned": True})
        spy.assert_called_once()

    def test_fork_join_hand_calculation_does_not_double_count_wait(self):
        frozen = frozen_trace()
        prediction = legacy_trace.predict(frozen, "synthetic", "fork_join")
        # GPU ends at 3+77=80; CPU reaches join at35. Join ends90,
        # then copy gap5+copy10+tail5 =>110, not an additional55 wait.
        self.assertEqual(prediction["predicted_ns"], 110)
        join = next(event for event in prediction["modeled_host_events"] if event["name"] == "join")
        self.assertEqual((join["begin_ns"], join["end_ns"]), (35, 90))
        with mock.patch.object(legacy_trace, "predict", return_value={"owned": True}) as spy:
            self.assertEqual(trace.predict(frozen, "synthetic", "fork_join"), {"owned": True})
        spy.assert_called_once_with(frozen, "synthetic", "fork_join")

    def test_raw_statistics_and_fitting_are_absent_from_model_modules(self):
        for module in (legacy_expressions, legacy_composition, legacy_trace):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names]
            self.assertNotIn("statistics", imports)
            functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            self.assertFalse(functions & {"fit", "freeze", "_nnls", "_summary", "stats", "_measure", "_profile_rows"})
        self.assertTrue(callable(composition.fit))
        self.assertTrue(callable(composition.pooled_component_prices))
        self.assertTrue(callable(trace.freeze))


if __name__ == "__main__":
    unittest.main()
