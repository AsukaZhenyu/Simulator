"""Opt-in module boundaries use real ports; all prices here are synthetic.

The B2 MoE fixture is a tiny generated mechanism test, not a physical fused
batch kernel or a performance measurement. Legacy scope barriers stay default.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from llm_infer_model.m2.global_execution import compile_global_execution_file, GlobalM2Session
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.tensor.engine import Action, ACTION_ADVANCE, STATUS_VALID
from llm_infer_model.tensor.spec import COPY_READY, OP_DONE, OP_NOT_STARTED, InvalidInput, Unsupported
from tensor_mapping.composition_observation import CompositionSession
from tensor_mapping.composition_policy import CompositionPolicy
from tensor_mapping.unified_strategy import (
    checkpoint_document, compile_unified_strategy_file, run_unified_strategy, strategy_fingerprint,
)


FIXTURES = Path(__file__).parent / "fixtures/global_execution"
EXAMPLES = Path(__file__).resolve().parents[2] / "mapping/examples/composition"


def write_document(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def execute_until(session, compiled, predicate):
    for action in plan_moe(compiled.aggregate):
        step = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
        if not step.ok:
            raise AssertionError(step.error_message)
        if predicate(session):
            return
    raise AssertionError("execution did not reach the requested boundary")


def two_request_model(folder):
    """Declare every B2 price explicitly; no measured-price extrapolation."""
    for path in FIXTURES.glob("*.json"):
        shutil.copyfile(path, folder / path.name)
    for layer in ("moeL0", "moeL1"):
        path = folder / f"{layer}.prefill.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["requests"].append(dict(document["requests"][0], request_id="r1"))
        document["state_in"]["requests"].append(dict(document["state_in"]["requests"][0], request_id="r1"))
        document["route_trace"] += [dict(row, request_id="r1") for row in document["route_trace"]]
        write_document(path, document)
    path = folder / "moe.costs.json"
    costs = json.loads(path.read_text(encoding="utf-8"))
    costs["entries"] = [dict(kind=kind, device=device, phase="prefill", batch=2,
        tokens=tokens, visible_history=0, duration_ns=duration, workspace_bytes=16,
        provenance="synthetic_layer", accounting=accounting)
        for device in ("cpu", "gpu")
        for kind, tokens, duration, accounting in (
            ("projection", 2, 100, "per_request"),
            ("attention", 2, 200, "per_request"),
            ("router", 2, 100, "per_request"),
            ("combine", 2, 100, "per_request"),
            ("expert_ffn", 2, 200, "per_expert_group"),
            ("expert_ffn", 4, 400, "per_expert_group"))]
    costs["source"] = "explicit_synthetic_B2_port_dependency_test"
    write_document(path, costs)
    path = folder / "moe.global.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["rounds"] = document["rounds"][:1]
    write_document(path, document)
    return path


class GlobalDependencyVisibilityTests(unittest.TestCase):
    def models(self):
        temporary = tempfile.TemporaryDirectory(prefix="llm-port-visibility-")
        self.addCleanup(temporary.cleanup)
        path = two_request_model(Path(temporary.name))
        return tuple(compile_global_execution_file(path, visibility_mode=mode)
                     for mode in ("scope_barriers", "data_dependencies"))

    def prefix(self, session, compiled):
        output = compiled.invocation_scopes[0].hidden_outputs[0]
        producer = compiled.scenario.workload.producer_of[output]
        execute_until(session, compiled, lambda current: current._runtime.state.op(producer) == OP_DONE)

    def test_downstream_request_can_run_before_previous_scope_finishes(self):
        for compiled in self.models():
            session = GlobalM2Session(compiled)
            self.prefix(session, compiled)
            first, second = compiled.invocation_scopes
            self.assertFalse(session._visibility.closed(first, session._runtime.state))
            downstream = [task for task in compiled.aggregate.tasks
                if task.kind == "projection" and task.operation_id in second.operation_ids]
            r0, r1 = (next(task for task in downstream if task.request_id == request)
                      for request in ("r0", "r1"))
            if compiled.visibility_mode == "data_dependencies":
                self.assertTrue(session.submit(Action("COMPUTE", r0.operation_id, device="cpu")).ok)
                self.assertTrue(session.advance().ok)
                self.assertEqual(session._runtime.state.op(r0.operation_id), OP_DONE)
                # r1's source value still does not exist, even though its
                # downstream task is public and the CPU is free now.
                rejected = session.submit(Action("COMPUTE", r1.operation_id, device="cpu"))
                self.assertFalse(rejected.ok)
                self.assertIn("input", rejected.error_message.lower())
            else:
                self.assertFalse(session.submit(Action("COMPUTE", r0.operation_id, device="cpu")).ok)

    def test_visible_future_module_is_not_active_from_weight_readiness(self):
        _, compiled = self.models()
        raw = GlobalM2Session(compiled)
        session = CompositionSession(raw, compiled, "global_m2")
        self.assertEqual({op.invocation_id for op in session.observe().operations},
                         {"moeL0-prefill", "moeL1-prefill"})
        self.assertEqual(session.observe().active_invocations, (("moeL0-prefill", "moeL0"),))
        self.assertTrue(raw.submit(Action("COPY", "w:moeL1:W_proj", src="dram", dst="vram")).ok)
        self.assertTrue(raw.advance().ok)
        self.assertEqual(session.observe().active_invocations, (("moeL0-prefill", "moeL0"),))
        self.prefix(raw, compiled)
        self.assertEqual(session.observe().active_invocations,
                         (("moeL0-prefill", "moeL0"), ("moeL1-prefill", "moeL1")))

    def test_cross_round_gate_and_persistent_state_are_preserved(self):
        compiled = compile_global_execution_file(FIXTURES / "dense.global.json",
                                                 visibility_mode="data_dependencies")
        session = GlobalM2Session(compiled)
        self.assertEqual({op.invocation_id for op in session.observe().operations},
                         {"denseL0-prefill", "denseL1-prefill"})
        execute_until(session, compiled, lambda current: len(current.observe().completed_invocations) == 2)
        gate = compiled.gate_operations[0][0]
        self.assertIn(gate, {op.id for op in session.observe().operations})
        self.assertFalse(any(op.kind != "state_update" and op.phase == "decode"
                             for op in session.observe().operations))
        self.assertTrue(session.submit(Action("COMPUTE", gate, device="cpu")).ok)
        self.assertTrue(session.advance().ok)
        self.assertTrue(any(op.kind != "state_update" and op.phase == "decode"
                            for op in session.observe().operations))
        full = GlobalM2Session(compiled)
        execute_until(full, compiled, lambda current: current.observe().complete)
        self.assertEqual(full.result().evaluation.status, STATUS_VALID)
        self.assertEqual(full.result().evaluation.makespan_ns, 44_007)
        state = full.persistent_state_document()
        self.assertEqual(len(state["kv_ids"]), 8)
        self.assertEqual(state["t_ns"], 44_007)
        self.assertEqual(len(state["completed_invocations"]), 4)
        for layer in ("denseL0", "denseL1"):
            for position in range(4):
                self.assertEqual(full._runtime.state.copy_at(f"kv:{layer}:r0:{position}", "dram"), COPY_READY)
        self.assertEqual(full.execution_document()["visibility_mode"], "data_dependencies")

    def test_modes_are_explicit_and_legacy_remains_default(self):
        compiled = compile_global_execution_file(FIXTURES / "dense.global.json")
        self.assertEqual(compiled.visibility_mode, "scope_barriers")
        with self.assertRaises(Unsupported):
            compile_global_execution_file(FIXTURES / "dense.global.json", visibility_mode="automatic")
        with self.assertRaisesRegex(InvalidInput, "must match"):
            GlobalM2Session(compiled, visibility_mode="data_dependencies")

    def test_future_routes_stay_private_when_static_modules_are_visible(self):
        temporary = tempfile.TemporaryDirectory(prefix="llm-port-causality-")
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        for path in FIXTURES.glob("*.json"):
            shutil.copyfile(path, folder / path.name)
        path = folder / "moeL1.prefill.json"
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["route_trace"][0]["expert_ids"] = [2, 3]
        write_document(path, changed)
        original = compile_global_execution_file(FIXTURES / "moe.global.json",
                                                 visibility_mode="data_dependencies")
        alternative = compile_global_execution_file(folder / "moe.global.json",
                                                    visibility_mode="data_dependencies")
        left, right = GlobalM2Session(original), GlobalM2Session(alternative)
        last_router = original.invocation_scopes[1].router_operation_ids[-1]
        for action in plan_moe(original.aggregate):
            self.assertEqual(left.observe(), right.observe())
            for session in (left, right):
                step = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok, step.error_message)
            if left._runtime.state.op(last_router) == OP_DONE:
                break
        else:
            self.fail("did not reach the second layer's router")
        self.assertNotEqual(left.observe().route_history, right.observe().route_history)


class UnifiedFactoryBindingTests(unittest.TestCase):
    def test_factory_is_fresh_on_full_partial_and_replay_runs(self):
        compiled = compile_unified_strategy_file(EXAMPLES / "dense.strategy.json")
        created = []

        def factory(source):
            self.assertIs(source, compiled)
            policy = CompositionPolicy(source.composition, history=source.history)
            created.append(policy)
            return policy

        full = run_unified_strategy(compiled, policy_factory=factory)
        partial = run_unified_strategy(compiled, max_decisions=1, policy_factory=factory)
        checkpoint = checkpoint_document(compiled, partial)
        replay = run_unified_strategy(compiled, checkpoint=checkpoint, policy_factory=factory)
        self.assertEqual(len({id(policy) for policy in created}), 3)
        self.assertEqual(full[0].evaluation.status, STATUS_VALID)
        self.assertEqual(full[0].evaluation.actions, replay[0].evaluation.actions)
        self.assertEqual(full[1].observe(), replay[1].observe())
        self.assertEqual(checkpoint["source_sha256"], strategy_fingerprint(compiled))
        self.assertEqual(replay[3], len(partial[0].decisions))

    def test_checkpoint_binds_document_mode_and_frontend_semantics(self):
        compiled = compile_unified_strategy_file(EXAMPLES / "dense.strategy.json")
        partial = run_unified_strategy(compiled, max_decisions=1)
        checkpoint = checkpoint_document(compiled, partial)
        changed = deepcopy(compiled.document)
        changed["id"] += "-changed"
        for candidate in (replace(compiled, document=changed),
                          replace(compiled, frontend_sha256="f" * 64)):
            with self.assertRaisesRegex(InvalidInput, "sources or runtime changed"):
                run_unified_strategy(candidate, checkpoint=checkpoint)
        self.assertNotEqual(strategy_fingerprint(compiled),
                            strategy_fingerprint(replace(compiled, visibility_mode="data_dependencies")))
        with self.assertRaisesRegex(InvalidInput, "policy_factory"):
            run_unified_strategy(compiled, policy_factory=1)


if __name__ == "__main__":
    unittest.main()
