"""Profiling-only diagnostic retention never changes execution semantics."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling.highlevel_accuracy import bridge
from profiling.highlevel_accuracy.multilayer import build_multilayer_case
from tensor_mapping.causal import compile_command
from tensor_mapping.causal_cli import verify_sources
from tensor_mapping.python_frontend import _PlanPolicy
from tensor_mapping.unified_strategy import create_session
from llm_infer_model.tensor.runtime import dispatch_command


RECIPES = (("cpu", "cpu", "cpu"), ("gpu", "gpu", "gpu"), ("gpu", "cpu", "gpu"))


class ProfilingSnapshotRetentionTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2]/"build/test-profiling-snapshot-retention"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, name="case", *, layers=1, strategy=2):
        return build_multilayer_case(self.folder/name, layers=layers, hidden=8,
            intermediate=16, heads=2, kv_heads=1, head_dim=4, tokens=3,
            history=0, threads=1, devices=RECIPES[strategy]).compile()

    def drive(self, plan, *, retain_diagnostics):
        compiled = plan._compiled
        verify_sources(compiled.sources)
        session = create_session(compiled)
        runtime = session._session._runtime
        sink = None if retain_diagnostics else bridge._install_profiling_snapshot_sink(session)
        policy = _PlanPolicy(compiled, plan._bindings, plan._runtime_spec, plan._preparations)
        actions, observations = [], []
        with patch.object(runtime, "_snapshot", wraps=runtime._snapshot) as snapshots:
            for index in range(compiled.document["max_decisions"]):
                observation = session.observe()
                observations.append(observation)
                if observation.complete or observation.error_code:
                    break
                decision = policy(observation)
                self.assertIsNotNone(decision)
                command = compile_command(decision, observation)
                step = dispatch_command(session, command, source_index=index)
                self.assertTrue(step.ok, step.error_message)
                actions.append(asdict(command.action))
            final = session.observe()
            policy.observe_final(final)
            result = session.result()
            calls = snapshots.call_count
        self.assertTrue(final.complete)
        self.assertEqual(result.evaluation.status, "valid")
        if retain_diagnostics:
            self.assertEqual(len(result.evaluation.states), calls + 1)
        else:
            self.assertEqual(len(result.evaluation.states), 1)
            self.assertEqual(sink.append_count, calls)
        normalized = replace(result, evaluation=replace(result.evaluation, states=()))
        return actions, observations, final, normalized, calls

    def test_six_placement_depth_cases_preserve_every_public_and_execution_field(self):
        for layers in (1, 2):
            for strategy in range(3):
                with self.subTest(layers=layers, strategy=strategy):
                    plan = self.plan(f"l{layers}-s{strategy}", layers=layers, strategy=strategy)
                    original = self.drive(plan, retain_diagnostics=True)
                    compact = self.drive(plan, retain_diagnostics=False)
                    self.assertEqual(compact, original)

    def test_actual_export_and_independent_reference_are_unchanged(self):
        plan = self.plan(layers=2)
        with patch.object(bridge, "_install_profiling_snapshot_sink", return_value=None):
            original = bridge.export_plan(plan, seed=321, numeric_arrays=True, lightweight_actions=True)
        with patch.object(bridge, "_install_profiling_snapshot_sink",
                          wraps=bridge._install_profiling_snapshot_sink) as installer:
            compact = bridge.export_plan(plan, seed=321, numeric_arrays=True, lightweight_actions=True)
            self.assertEqual(installer.call_count, 1)
        self.assertEqual(bridge.sha(compact), bridge.sha(original))
        self.assertEqual(bridge.reference_ports(compact), bridge.reference_ports(original))

    def test_default_export_does_not_install_the_profiling_sink(self):
        with patch.object(bridge, "_install_profiling_snapshot_sink",
                          side_effect=AssertionError("default recorder changed")):
            program = bridge.export_plan(self.plan(), seed=321)
        self.assertNotIn("export_trace_recording", program)

    def test_sink_rejects_unknown_or_already_started_runtime(self):
        with self.assertRaisesRegex(bridge.BridgeError, "known causal"):
            bridge._install_profiling_snapshot_sink(object())
        session = create_session(self.plan()._compiled)
        runtime = session._session._runtime
        runtime._events.append(object())
        with self.assertRaisesRegex(bridge.BridgeError, "before any action"):
            bridge._install_profiling_snapshot_sink(session)


if __name__ == "__main__":
    unittest.main()
