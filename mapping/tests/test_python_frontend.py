"""Behavioral acceptance of modular plans on the original modeling runtime.

All timing here is synthetic. These tests establish strategy/causality and
bookkeeping contracts, not measured speedups or hardware prediction accuracy.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported, COPY_READY, COPY_ABSENT
from tensor_mapping.causal import Decision
from tensor_mapping.python_frontend import (
    Cache, DEFER, ModelPlan, PlanModule, Prefetch, RuntimePolicy, Scheduling,
    Select, Transport,
)
from tensor_mapping.unified_strategy import (
    compile_unified_strategy_file, create_session, run_unified_strategy, _runtime_signature,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "mapping/examples"


def dense_plan():
    return ModelPlan(id="python-modular-dense", model_kind="global_m2",
        model_file=ROOT / "modeling/tests/fixtures/global_execution/dense.global.json",
        environment_file=EXAMPLES / "causal/no-disturbance.environment.json",
        scheduling=Scheduling(placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
            cache=Cache(rule="lru", budget_bytes=200000),
            prefetch=Prefetch(next_layers=1, max_count=8, include_dense_weights=True)))


class Layers(PlanModule):
    def build(self, plan):
        for layer in ("denseL0", "denseL1"):
            with plan.module(layer, layer_id=layer):
                plan.place("dense_ffn", "cpu" if layer == "denseL0" else "gpu")


class PublicHistoryAudit(RuntimePolicy):
    def __init__(self, label="audit"):
        self.label = label
        self.observed = set()
        self.calls = 0

    def decide(self, observation):
        self.calls += 1
        for route in observation.route_history:
            self.observed.add((route.layer_id, route.request_id, route.absolute_position))
        return DEFER

    def state_document(self):
        return dict(label=self.label, calls=self.calls, observed=sorted(self.observed))


class InvalidPolicy(RuntimePolicy):
    def decide(self, observation):
        return {"kind": "COMPUTE", "target_id": "private-future-task"}


class UnrevealedPolicy(RuntimePolicy):
    def decide(self, observation):
        return Decision("COMPUTE", "not-public", device="gpu")


class WrongEngine(RuntimePolicy):
    def decide(self, observation):
        for tensor in observation.tensors:
            if (tensor.role == "weight" and tensor.status_at("dram") == COPY_READY and
                    tensor.status_at("vram") == COPY_ABSENT):
                return Decision("COPY", tensor.id, src="dram", dst="vram", transfer_engine="sm")
        return DEFER


class CPUProjection(RuntimePolicy):
    def decide(self, observation):
        if observation.running:
            return DEFER
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        for operation in observation.operations:
            if (operation.kind == "projection" and operation.status == "NOT_STARTED" and
                    all(tensors[tid].status_at("dram") == COPY_READY for tid in operation.inputs)):
                return Decision("COMPUTE", operation.id, device="cpu")
        return DEFER


class WeightLoad(RuntimePolicy):
    def decide(self, observation):
        for tensor in observation.tensors:
            if tensor.role == "weight" and tensor.status_at("dram") == COPY_READY and tensor.status_at("vram") == COPY_ABSENT:
                return Decision("COPY", tensor.id, src="dram", dst="vram")
        return DEFER


class EvictLiveWeight(RuntimePolicy):
    def decide(self, observation):
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        for operation in observation.operations:
            if operation.status == "RUNNING":
                for tid in operation.inputs:
                    if tensors[tid].role == "weight" and tensors[tid].status_at("vram") == COPY_READY:
                        return Decision("EVICT", tid, location="vram")
        return DEFER


class SwitchCommittedExpert(RuntimePolicy):
    def decide(self, observation):
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        for operation in observation.operations:
            if (operation.kind != "expert_up" or operation.status != "NOT_STARTED" or
                    operation.phase != "decode" or operation.layer_id != "fragmentL0"):
                continue
            gates = [gate for gate in observation.operations if gate.kind == "expert_gate" and
                     gate.region_id == operation.region_id and gate.invocation_id == operation.invocation_id and
                     gate.status == "DONE"]
            if any(all(tensors[tid].status_at("dram") == COPY_READY and
                       tensors[tid].status_at("vram") == COPY_ABSENT for tid in gate.outputs) for gate in gates):
                return Decision("COMPUTE", operation.id, device="gpu")
        return DEFER


class StartOneExpertOnCPU(RuntimePolicy):
    """Prepare one revealed gate, execute it, then let the base rule continue."""
    def __init__(self):
        self.target = None
        self.region = None
        self.finished = False

    def decide(self, observation):
        if self.finished:
            return DEFER
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        if self.target is None:
            candidates = [operation for operation in observation.operations
                if operation.kind == "expert_gate" and operation.phase == "prefill" and
                operation.layer_id == "fragmentL0" and operation.status == "NOT_STARTED" and
                all(any(status == COPY_READY for _, status in tensors[tid].copies) for tid in operation.inputs)]
            if not candidates:
                return DEFER
            self.target = candidates[0].id
            self.region = candidates[0].region_id
        if observation.running:
            return Decision("ADVANCE")
        operation = next(operation for operation in observation.operations if operation.id == self.target)
        for tid in operation.inputs:
            tensor = tensors[tid]
            if tensor.status_at("dram") != COPY_READY:
                if tensor.status_at("ssd") == COPY_READY:
                    return Decision("READ_SSD", tid)
                if tensor.status_at("vram") != COPY_READY:
                    raise AssertionError("public input has no prepared source")
                return Decision("COPY", tid, src="vram", dst="dram", transfer_engine="sm")
        self.finished = True
        return Decision("COMPUTE", operation.id, device="cpu")

    def state_document(self):
        return dict(target=self.target, region=self.region, finished=self.finished)


class ModularPlanTests(unittest.TestCase):
    def test_flat_and_nested_grouping_have_identical_actions_and_timeline(self):
        flat, nested = dense_plan(), dense_plan()
        for layer in ("denseL0", "denseL1"):
            flat.place("dense_ffn", "cpu" if layer == "denseL0" else "gpu", select=Select(layer_id=layer))
        Layers()(nested, name="decoder")
        a, b = flat.compile(), nested.compile()
        ar, br = a.run(), b.run()
        self.assertEqual(ar[0].termination_reason, "goal_reached", ar[0].compilation_error)
        self.assertEqual(br[0].termination_reason, "goal_reached", br[0].compilation_error)
        self.assertEqual(ar[0].evaluation.actions, br[0].evaluation.actions)
        self.assertEqual(ar[0].evaluation.events, br[0].evaluation.events)
        self.assertEqual(_runtime_signature(ar[1]), _runtime_signature(br[1]))
        self.assertEqual(a.ir_document()["modules"], [])
        self.assertEqual(len(b.ir_document()["modules"]), 3)
        self.assertNotEqual(a.ir_document(), b.ir_document())
        operations = {operation.id: operation for operation in br[1].observe().operations}
        ffn = [(operations[row.action.target_id].layer_id, row.action.device)
               for row in br[0].decisions if row.action.kind == "COMPUTE" and
               operations[row.action.target_id].kind == "dense_ffn"]
        self.assertEqual(ffn, [("denseL0", "cpu"), ("denseL1", "gpu")] * 2)

    def test_imported_strategies_preserve_default_execution(self):
        for name in ("dense", "moe"):
            path = EXAMPLES / f"composition/{name}.strategy.json"
            direct = run_unified_strategy(compile_unified_strategy_file(path))
            imported = ModelPlan.from_strategy(path).compile().run()
            self.assertEqual(imported[0].evaluation.actions, direct[0].evaluation.actions)
            self.assertEqual(imported[0].evaluation.events, direct[0].evaluation.events)
            self.assertEqual(_runtime_signature(imported[1]), _runtime_signature(direct[1]))

    def test_refinement_and_prefetch_use_one_shared_runtime(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/dense.strategy.json")
        with plan.module("model"):
            with plan.module("next-layer", layer_id="denseL1"):
                plan.place("dense_ffn", "gpu")
        compiled = plan.compile()
        result = compiled.run()
        run, session, policy, _ = result
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        self.assertEqual(len(session.persistent_state_document()["kv_ids"]), 8)
        self.assertEqual(len(compiled.ir_document()["backend"]["execution"]), 4)
        rows = policy.audit_document()["decisions"]
        prefetch = [row for row in rows if row["reason"] == "cross_layer_public_history_prefetch"]
        self.assertTrue(prefetch)
        events = run.evaluation.events
        self.assertTrue(any(event.kind == "COPY" and any(
            compute.kind == "COMPUTE" and compute.start_ns <= event.start_ns < compute.end_ns
            for compute in events) for event in events if any(
                row["evidence"].get("tensor") == event.target_id for row in prefetch)))
        budget = compiled.ir_document()["backend"]["cache"]["budget_bytes"]
        for record in run.decisions:
            resident = sum(tensor.storage_bytes for tensor in record.observation.tensors
                           if tensor.role == "weight" and tensor.status_at("vram") != COPY_ABSENT)
            self.assertLessEqual(resident, budget)

    def test_precise_refinement_cannot_be_silently_replaced(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/dense.strategy.json")
        row = plan._document["execution"][0]
        with self.assertRaisesRegex(InvalidInput, "already bound"):
            plan.refine(row["scope_id"], row["spec"])
        plan.refine(row["scope_id"], row["spec"], replace_existing=True)
        self.assertEqual(len(plan.compile().ir_document()["backend"]["execution"]), 4)

    def test_single_layer_selector_uses_the_actual_invocation_identity(self):
        plan = ModelPlan(id="single", model_kind="m2",
            model_file=ROOT / "modeling/tests/fixtures/global_execution/denseL0.prefill.json",
            environment_file=EXAMPLES / "causal/no-disturbance.environment.json",
            scheduling=Scheduling(placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"}))
        plan.place("dense_ffn", "cpu", select=Select(invocation_id="denseL0-prefill"))
        run, session, _, _ = plan.compile().run()
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        operations = {operation.id: operation for operation in session.observe().operations}
        devices = [record.action.device for record in run.decisions if record.action.kind == "COMPUTE" and
                   operations[record.action.target_id].kind == "dense_ffn"]
        self.assertEqual(devices, ["cpu"])

    def test_selector_conflicts_and_unmatched_labels_are_rejected(self):
        plan = dense_plan()
        plan.place("dense_ffn", "cpu", select=Select(layer_id="missing"))
        with self.assertRaisesRegex(InvalidInput, "matches no task"):
            plan.compile()
        plan = dense_plan()
        plan.place("dense_ffn", "cpu", select=Select(layer_id="denseL0"))
        plan.place("dense_ffn", "gpu", select=Select(request_id="r0"))
        with self.assertRaisesRegex(InvalidInput, "ambiguous"):
            plan.compile()
        plan.place("dense_ffn", "cpu", select=Select(layer_id="denseL0"), priority=1)
        self.assertEqual(plan.compile().run()[0].termination_reason, "goal_reached")
        with self.assertRaisesRegex(InvalidInput, "contradictory"):
            with dense_plan().module("outer", layer_id="L0") as plan:
                with plan.module("inner", layer_id="L1"):
                    pass

    def test_missing_cost_and_unsupported_domains_are_not_approximated(self):
        with self.assertRaises(Unsupported):
            dense_plan().place("dense_ffn", "gpu1")
        with self.assertRaises(Unsupported):
            dense_plan().place(None, "gpu")
        with self.assertRaises(Unsupported):
            Select(kind="invented_fused_kernel")
        path = EXAMPLES / "composition/block.strategy.json"
        self.assertEqual(ModelPlan.from_strategy(path).compile().run()[0].termination_reason, "goal_reached")
        with self.assertRaisesRegex(Unsupported, "block traversal"):
            ModelPlan.from_strategy(path, visibility_mode="data_dependencies").compile()


class RuntimeFrontendTests(unittest.TestCase):
    def test_custom_first_stage_and_default_scheduler_share_region_binding(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/moe.strategy.json")
        result = plan.runtime_policy(StartOneExpertOnCPU, allow_device_override=True).compile().run()
        run, session, policy, _ = result
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        state = policy.audit_document()["python_state"]
        self.assertTrue(state["finished"])
        operations = {operation.id: operation for operation in session.observe().operations}
        devices = [record.action.device for record in run.decisions if record.action.kind == "COMPUTE" and
                   operations[record.action.target_id].region_id == state["region"] and
                   operations[record.action.target_id].kind in {"expert_gate", "expert_up", "expert_swiglu", "expert_down"}]
        self.assertEqual(devices, ["cpu"] * 4)

    def test_callback_cannot_rebind_a_prepared_expert_region(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/moe.strategy.json")
        result = plan.runtime_policy(SwitchCommittedExpert, allow_device_override=True).compile().run()
        self.assertEqual(result[0].termination_reason, "decision_rejected")
        self.assertIn("already committed", result[0].compilation_error)
        self.assertTrue(any(record.action.kind == "COMPUTE" and record.action.device == "cpu"
                            for record in result[0].decisions))

    def test_custom_actions_keep_shared_capacity_and_reader_locks(self):
        plan = ModelPlan(id="budget", model_kind="global_m2",
            model_file=ROOT / "modeling/tests/fixtures/global_execution/dense.global.json",
            environment_file=EXAMPLES / "causal/no-disturbance.environment.json",
            scheduling=Scheduling(placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
                                  cache=Cache(budget_bytes=1)))
        result = plan.runtime_policy(WeightLoad).compile().run()
        self.assertEqual(result[0].termination_reason, "decision_rejected")
        self.assertIn("shared weight cache budget", result[0].compilation_error)
        result = dense_plan().runtime_policy(EvictLiveWeight).compile().run()
        self.assertEqual(result[0].termination_reason, "runtime_failure")
        self.assertFalse(result[0].decisions[-1].accepted)

    def test_deferred_callback_keeps_reconfiguration_protocol_and_costs(self):
        path = EXAMPLES / "reconfiguration/global.strategy.json"
        expected = run_unified_strategy(compile_unified_strategy_file(path))
        actual = ModelPlan.from_strategy(path).runtime_policy(PublicHistoryAudit).compile().run()
        self.assertEqual(actual[0].termination_reason, "goal_reached", actual[0].compilation_error)
        self.assertEqual(_runtime_signature(actual[1]), _runtime_signature(expected[1]))
        self.assertEqual(actual[0].evaluation.actions, expected[0].evaluation.actions)
    def test_python_history_callback_and_replay_have_fresh_state(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/moe.strategy.json")
        plan.runtime_policy(PublicHistoryAudit, parameters={"label": "causal"})
        compiled = plan.compile()
        full = compiled.run()
        partial = compiled.run(max_decisions=5)
        resumed = compiled.run(checkpoint=compiled.checkpoint(partial))
        self.assertEqual(full[0].termination_reason, "goal_reached")
        self.assertEqual(_runtime_signature(full[1]), _runtime_signature(resumed[1]))
        self.assertEqual(full[2].audit_document(), resumed[2].audit_document())
        self.assertTrue(full[2].audit_document()["python_state"]["observed"])
        self.assertEqual(full[0].decisions[0].observation.route_history, ())
        self.assertNotIn("routes", compiled.ir_document()["runtime_policy"])

    def test_invalid_and_unrevealed_callback_decisions_are_rejected(self):
        for factory in (InvalidPolicy, UnrevealedPolicy):
            result = dense_plan().runtime_policy(factory).compile().run()
            self.assertEqual(result[0].termination_reason, "decision_rejected")
            self.assertTrue(result[0].compilation_error)
            self.assertFalse(result[0].decisions)

    def test_callback_cannot_silently_override_transport(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/dense.strategy.json")
        rejected = plan.runtime_policy(WrongEngine).compile().run(max_decisions=1)
        self.assertEqual(rejected[0].termination_reason, "decision_rejected")
        self.assertIn("allow_transport_override", rejected[0].compilation_error)
        accepted = plan.runtime_policy(WrongEngine, allow_transport_override=True).compile().run(max_decisions=1)
        self.assertTrue(accepted[0].decisions[0].accepted)
        self.assertEqual(accepted[0].decisions[0].decision.transfer_engine, "sm")

    def test_callback_device_delegation_preserves_fixed_scope_bindings(self):
        plan = dense_plan().runtime_policy(CPUProjection)
        result = plan.compile().run(max_decisions=1)
        self.assertEqual(result[0].termination_reason, "decision_rejected")
        self.assertIn("allow_device_override", result[0].compilation_error)
        plan.runtime_policy(CPUProjection, allow_device_override=True)
        result = plan.compile().run(max_decisions=1)
        self.assertTrue(result[0].decisions[0].accepted)
        self.assertEqual(result[0].decisions[0].decision.device, "cpu")
        plan.place("projection", "gpu", select=Select(layer_id="denseL0"))
        result = plan.compile().run(max_decisions=1)
        self.assertEqual(result[0].termination_reason, "decision_rejected")
        self.assertIn("fixed scoped", result[0].compilation_error)

    def test_checkpoint_binds_frontend_parameters_and_source_content(self):
        a = dense_plan().runtime_policy(PublicHistoryAudit, parameters={"label": "a"}).compile()
        checkpoint = a.checkpoint(a.run(max_decisions=1))
        b = dense_plan().runtime_policy(PublicHistoryAudit, parameters={"label": "b"}).compile()
        with self.assertRaisesRegex(InvalidInput, "changed"):
            b.run(checkpoint=checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "author.py"
            source.write_text("# original\n", encoding="utf-8", newline="\n")
            compiled = dense_plan().compile(source_files=(source,))
            source.write_text("# changed\n", encoding="utf-8", newline="\n")
            with self.assertRaises(InvalidInput):
                compiled.run()

    def test_import_compile_source_race_is_rejected(self):
        path = EXAMPLES / "composition/dense.strategy.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for key in ("services_file", "environment_file"):
            document[key] = str((path.parent / document[key]).resolve())
        document["model"]["file"] = str((path.parent / document["model"]["file"]).resolve())
        for row in document["execution"]:
            row["spec"]["invocation_files"] = [str((path.parent / p).resolve()) for p in row["spec"]["invocation_files"]]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "strategy.json"
            source.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            plan = ModelPlan.from_strategy(source)
            document["id"] = "changed-after-import"
            source.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "changed since"):
                plan.compile()


if __name__ == "__main__":
    unittest.main()
