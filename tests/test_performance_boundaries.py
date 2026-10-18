"""Architecture boundary checks, independent of native measurements and CUDA."""
from __future__ import annotations

import ast
from fractions import Fraction
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


class PerformanceOwnershipTests(unittest.TestCase):
    def test_analytical_convenience_compute_delegates_to_modeling(self):
        from llm_simulator import AnalyticalCosts, Device
        public = importlib.import_module("llm_simulator.analytical")
        model = importlib.import_module("llm_infer_model.performance")
        self.assertIs(public.analytical_compute_duration_ns,
                      model.analytical_compute_duration_ns)
        device = Device(1000, 2000, launch_overhead_ns=3)
        with patch.object(public, "analytical_compute_duration_ns", return_value=137) as calculate:
            observed = AnalyticalCosts().compute_duration_ns(device, flops=10, traffic_bytes=20)
        self.assertEqual(observed, 137)
        calculate.assert_called_once_with(flops=10, traffic_bytes=20,
            flops_per_second=1000, memory_bytes_per_second=2000, launch_overhead_ns=3)

    def test_analytical_convenience_transfer_delegates_to_modeling(self):
        from llm_simulator import AnalyticalCosts, Device, Hardware
        public = importlib.import_module("llm_simulator.analytical")
        model = importlib.import_module("llm_infer_model.performance")
        self.assertIs(public.analytical_transfer_duration_ns,
                      model.analytical_transfer_duration_ns)
        device = Device(1000, 2000)
        hardware = Hardware(cpu=device, gpu=device, dram_capacity_bytes=1024,
                            vram_capacity_bytes=1024,
                            transfer_bandwidth_bytes_per_second=4000, transfer_latency_ns=7)
        with patch.object(public, "analytical_transfer_duration_ns", return_value=139) as calculate:
            observed = AnalyticalCosts().transfer_duration_ns(
                hardware, route="dram_to_vram", storage_bytes=20)
        self.assertEqual(observed, 139)
        calculate.assert_called_once_with(storage_bytes=20, bytes_per_second=4000, latency_ns=7)

    def test_explainable_graph_uses_critical_path_and_does_not_sum_overlap(self):
        from llm_infer_model.performance import (
            Graph, Parameter, ParameterBundle, Task, WorkDemand, evaluate,
        )
        bundle = ParameterBundle(tuple(
            Parameter(name, value, unit, "declared", (), "hand-computable example")
            for name, value, unit in (("host_submit", 3, "ns"),
                                      ("host_finish", 2, "ns"),
                                      ("gpu_rate", 1_000_000_000, "flop/s"),
                                      ("cpu_rate", 1_000_000_000, "flop/s"))
        ))
        graph = Graph((
            Task("submit", "host", fixed_parameter="host_submit"),
            Task("gpu", "gpu", (WorkDemand("gpu_rate", 10, "flop"),), ("submit",)),
            Task("cpu", "cpu", (WorkDemand("cpu_rate", 6, "flop"),)),
            Task("finish", "host", depends_on=("gpu", "cpu"), fixed_parameter="host_finish"),
        ))
        result = evaluate(graph, bundle)
        self.assertEqual(result.duration_ns, 15)
        times = {task.task_id: (task.start_ns, task.end_ns) for task in result.tasks}
        self.assertEqual(times, {"submit": (0, 3), "gpu": (3, 13),
                                 "cpu": (0, 6), "finish": (13, 15)})
        self.assertEqual(sum(task.duration_ns for task in result.tasks), 21)
        # A task's dependency wait is encoded by the DAG. It is not added a
        # second time as an unexplained completed-action correction.
        self.assertNotEqual(result.duration_ns, 21)

    def test_public_performance_facade_is_the_same_modeling_api(self):
        public = importlib.import_module("llm_simulator.performance")
        model = importlib.import_module("llm_infer_model.performance")
        for name in ("Parameter", "ParameterBundle", "WorkDemand", "Task", "Graph",
                     "MemoryDomain", "MemoryReservation", "ResourcePoint",
                     "ResourceTimeline", "evaluate", "InvalidModel", "UnknownParameter"):
            with self.subTest(name=name):
                self.assertIs(getattr(public, name), getattr(model, name))

    def test_installed_modeling_imports_without_optional_profiling_or_numeric_stack(self):
        # A clean interpreter cannot inherit imported optional modules from the
        # rest of the test suite. Block them even when installed on this machine.
        script = """
import importlib.abc
import sys

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'profiling', 'llm_infer_profile', 'resource_calibration',
                'state_calibration', 'numpy', 'scipy', 'sklearn', 'torch'}:
            raise AssertionError('modeling imported optional dependency: ' + fullname)

sys.meta_path.insert(0, RejectOptional())
from llm_infer_model.performance import evaluate, Parameter, ParameterBundle
assert callable(evaluate)
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_modeling_has_no_observation_statistics_or_estimator_dependencies(self):
        module = importlib.import_module("llm_infer_model.performance")
        path = Path(inspect.getfile(module))
        sources = sorted(path.parent.rglob("*.py")) if path.name == "__init__.py" else [path]
        self.assertTrue(sources)
        forbidden = {"profiling", "llm_infer_profile", "resource_calibration", "state_calibration", "statistics",
                     "numpy", "scipy", "sklearn", "torch"}
        for source in sources:
            with self.subTest(source=source.name):
                tree = ast.parse(source.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    else:
                        continue
                    self.assertFalse({name.split(".")[0] for name in names} & forbidden,
                                     f"{source.name}:{node.lineno} imports {names}")


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "profiling/parameter_calibration").is_dir(),
                     "cross-boundary calibration check requires the optional source-checkout tools")
class ProfilingOwnershipTests(unittest.TestCase):
    def test_heldout_evaluator_calls_modeling_instead_of_a_profiling_predictor(self):
        from llm_infer_model import performance as model
        from profiling.parameter_calibration.observations import ParameterObservation
        from profiling.parameter_calibration.parameters import estimate_parameters
        from profiling.parameter_calibration.evaluator import evaluate_calibration

        context = {"fixture": "independent architecture boundary test"}

        def observation(index, *, heldout=False):
            # Deliberately synthetic observations: this checks API ownership,
            # not hardware evidence or current raw-file verification.
            return ParameterObservation(
                observation_id=f"observation-{index}", parameter_name="reference" if heldout else "rate",
                kind="reference_duration" if heldout else "compute_throughput",
                value=Fraction(1 if heldout else 1_000_000_000),
                unit="ns" if heldout else "flop/s",
                timing_scope="whole_iteration_completed" if heldout else "device_compute_active",
                context_json=json.dumps(context), run_id=f"run-{index}", process_id=index + 100,
                seed=index, split="heldout" if heldout else "profile",
                started_at_utc="2026-10-18T00:00:06Z" if heldout else "2026-10-18T00:00:00Z",
                ended_at_utc="2026-10-18T00:00:07Z" if heldout else "2026-10-18T00:00:01Z",
                source_path="synthetic-test-fixture", source_sha256="0" * 64,
                evidence=("synthetic-test-fixture",), source_verified=True, numerical_passed=True)

        artifact = estimate_parameters(tuple(observation(i) for i in (1, 2, 3)),
            context=context, frozen_at_utc="2026-10-18T00:00:05Z")
        graph = model.Graph((model.Task("one", "gpu", (model.WorkDemand("rate", 1, "flop"),)),))
        with patch.object(model, "evaluate", wraps=model.evaluate) as calculate:
            report = evaluate_calibration(graph, artifact,
                tuple(observation(i, heldout=True) for i in (11, 12, 13)), verify_sources=False)
        calculate.assert_called_once()
        self.assertIs(calculate.call_args.args[0], graph)
        self.assertIsInstance(calculate.call_args.args[1], model.ParameterBundle)
        self.assertEqual(calculate.call_args.kwargs, {"timeline": None})
        self.assertEqual(report["predicted_ns"], 1)
        self.assertEqual(report["prediction_provider"], "llm_infer_model.performance.evaluate")
        self.assertFalse(report["heldout_used_for_estimation"])
        self.assertFalse(report["current_raw_sources_verified"])


if __name__ == "__main__":
    unittest.main()
