"""Two lookup optimizations preserve exact causal observations and execution."""
from contextlib import contextmanager
from dataclasses import asdict
import inspect
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from llm_infer_model.m2 import causal
from llm_infer_model.tensor.spec import M1Mapspace
from llm_infer_model.tensor.runtime import dispatch_command
from profiling.highlevel_accuracy.bridge import export_plan, reference_ports, sha
from profiling.highlevel_accuracy.multilayer import build_multilayer_case
from tensor_mapping.causal import compile_command
from tensor_mapping.causal_cli import verify_sources
from tensor_mapping.python_frontend import _PlanPolicy
from tensor_mapping.unified_strategy import create_session


def legacy_allows(self, route, tensor_id):
    return tensor_id in self.allowed_transfers.get(route, ())


def legacy_observe():
    source = textwrap.dedent(inspect.getsource(causal.CausalM2Session.observe))
    declaration = "    visible_set = set(visible)\n"
    if source.count(declaration) != 1 or source.count("tensor in visible_set") != 1:
        raise AssertionError("known two-line visibility optimization seam changed")
    source = source.replace(declaration, "").replace("tensor in visible_set", "tensor in visible")
    namespace = dict(causal.__dict__)
    exec(compile(source, "<legacy-visible-tuple-observation>", "exec"), namespace)
    return namespace["observe"]


@contextmanager
def legacy_lookups():
    with patch.object(M1Mapspace, "allows", legacy_allows), \
         patch.object(causal.CausalM2Session, "observe", legacy_observe()):
        yield


RECIPES = (("cpu", "cpu", "cpu"), ("gpu", "gpu", "gpu"), ("gpu", "cpu", "gpu"))


class TransferMembershipEquivalenceTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2]/"build/test-transfer-membership-equivalence"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, name, *, layers, strategy, tokens=3, history=0):
        return build_multilayer_case(self.folder/name, layers=layers, hidden=8,
            intermediate=16, heads=2, kv_heads=1, head_dim=4, tokens=tokens,
            history=history, threads=1, devices=RECIPES[strategy]).compile()

    def drive(self, plan):
        compiled = plan._compiled
        verify_sources(compiled.sources)
        session = create_session(compiled)
        policy = _PlanPolicy(compiled, plan._bindings, plan._runtime_spec, plan._preparations)
        actions, observations = [], []
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
        self.assertTrue(final.complete)
        self.assertEqual(result.evaluation.status, "valid")
        return actions, observations, final, result

    def test_twelve_cases_preserve_observations_actions_full_result_and_order(self):
        for phase, tokens, history in (("prefill", 3, 0), ("decode", 1, 4)):
            for layers in (1, 2):
                for strategy in range(3):
                    with self.subTest(phase=phase, layers=layers, strategy=strategy):
                        plan = self.plan(f"{phase}-l{layers}-s{strategy}", layers=layers,
                                         strategy=strategy, tokens=tokens, history=history)
                        with legacy_lookups():
                            original = self.drive(plan)
                        optimized = self.drive(plan)
                        self.assertEqual(optimized, original)

    def test_actual_export_and_numeric_reference_are_identical(self):
        plan = self.plan("numeric", layers=2, strategy=2)
        with legacy_lookups():
            original = export_plan(plan, seed=123, numeric_arrays=True, lightweight_actions=True)
        optimized = export_plan(plan, seed=123, numeric_arrays=True, lightweight_actions=True)
        self.assertEqual(sha(optimized), sha(original))
        self.assertEqual(reference_ports(optimized), reference_ports(original))


if __name__ == "__main__":
    unittest.main()
