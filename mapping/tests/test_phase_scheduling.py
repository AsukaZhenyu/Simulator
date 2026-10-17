"""Preparation/phase contracts; fixture times are synthetic, not calibration."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.causal import ControlDecision, Decision
from tensor_mapping.phase_scheduling import DataPreparation, PhasePlacement
from tensor_mapping.python_frontend import Cache, DEFER, ModelPlan, RuntimePolicy, Scheduling
from tensor_mapping.reconfiguration import ReconfigurationSpec
from tensor_mapping.unified_strategy import _runtime_signature


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "mapping/examples"


def dense_plan(*, device="gpu", cache_budget=200000, model_file=None,
               environment_file=None, reconfiguration=None):
    return ModelPlan(id="phase-placement-synthetic", model_kind="global_m2",
        model_file=model_file or ROOT / "modeling/tests/fixtures/global_execution/dense.global.json",
        environment_file=environment_file or EXAMPLES / "causal/no-disturbance.environment.json",
        scheduling=Scheduling(placements={"projection": device, "attention": device, "dense_ffn": device},
            cache=Cache(rule="lru", budget_bytes=cache_budget), reconfiguration=reconfiguration))


class CustomControlProbe(RuntimePolicy):
    """A saved callback that visibly chooses DECLINE rather than built-in ACK."""
    def __init__(self):
        self.calls = 0
        self.declines = 0

    def decide(self, observation):
        self.calls += 1
        if observation.control_running is not None:
            return Decision("ADVANCE")
        if observation.pending_reclaim is not None:
            self.declines += 1
            return ControlDecision("DECLINE_RECLAIM", request_id=observation.pending_reclaim.id,
                                   reason="test_custom_controller")
        return DEFER

    def state_document(self):
        return dict(calls=self.calls, declines=self.declines)


class DataPreparationTests(unittest.TestCase):
    def test_document_roundtrip_has_exact_explicit_schema(self):
        declaration = DataPreparation(layer_id="layer0", stage="gate",
            tensor_ids=("w:layer0:gate",), expert_id=0, destination="dram",
            trigger="startup", max_count=1)
        document = declaration.document()
        self.assertEqual(DataPreparation.from_document(document), declaration)
        self.assertEqual(document["tensor_ids"], ["w:layer0:gate"])
        with self.assertRaises(FrozenInstanceError):
            declaration.trigger = "spare"
        for bad in (document | {"future_routes": []}, {k: v for k, v in document.items() if k != "trigger"}):
            with self.assertRaises(InvalidInput):
                DataPreparation.from_document(bad)

    def test_selector_is_conjunction_over_static_public_catalogue(self):
        observation = SimpleNamespace(weight_catalogue=(
            ("w:z", "L1", "gate", 1), ("w:a", "L1", "gate", 1),
            ("w:b", "L1", "up", 1), ("w:c", "L0", "gate", 1),
            ("w:d", "L1", "gate", 2)))
        selected = DataPreparation(layer_id="L1", stage="gate", expert_id=1)
        self.assertEqual(selected.matching_tensor_ids(observation), ("w:a", "w:z"))
        self.assertEqual(DataPreparation(layer_id="L1", stage="gate", expert_id=1,
            max_count=1).matching_tensor_ids(observation), ("w:a",))
        self.assertEqual(DataPreparation(tensor_ids=("w:b", "w:a")).matching_tensor_ids(observation),
                         ("w:a", "w:b"))

    def test_selector_does_not_read_future_or_revealed_route_data(self):
        class PublicOnly:
            weight_catalogue = (("w:e7", "L1", "gate", 7),)

            @property
            def route_history(self):
                raise AssertionError("preparation must not inspect router activations")

        self.assertEqual(DataPreparation(layer_id="L1", expert_id=7).matching_tensor_ids(PublicOnly()),
                         ("w:e7",))

    def test_contradictory_or_unknown_weight_selection_is_rejected(self):
        observation = SimpleNamespace(weight_catalogue=(("w:a", "L1", "gate", 0),))
        for declaration in (DataPreparation(layer_id="absent"),
                            DataPreparation(tensor_ids=("future-hidden-activation",)),
                            DataPreparation(layer_id="L0", tensor_ids=("w:a",))):
            with self.assertRaises(InvalidInput):
                declaration.matching_tensor_ids(observation)
        with self.assertRaises(Unsupported):
            DataPreparation(layer_id="L1").matching_tensor_ids(SimpleNamespace())

    def test_invalid_selectors_fail_without_coercion(self):
        for fields in ({}, {"layer_id": "L1", "max_count": 0},
                       {"layer_id": "L1", "max_count": True},
                       {"layer_id": "L1", "expert_id": False},
                       {"layer_id": "L1", "tensor_ids": ["w:a"]},
                       {"tensor_ids": ("w:a", "w:a")}, {"layer_id": "bad\nname"}):
            with self.subTest(fields=fields), self.assertRaises(InvalidInput):
                DataPreparation(**fields)
        for fields in ({"layer_id": "L1", "trigger": "future_route"},
                       {"layer_id": "L1", "destination": "ssd"}):
            with self.assertRaises(Unsupported):
                DataPreparation(**fields)


class PhasePlacementTests(unittest.TestCase):
    def test_phase_bindings_copy_inputs_and_apply_as_scoped_rules(self):
        prefill = {"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"}
        policy = PhasePlacement(prefill=prefill, decode={kind: "cpu" for kind in prefill})
        prefill["attention"] = "cpu"
        self.assertEqual(policy.prefill["attention"], "gpu")
        with self.assertRaises(TypeError):
            policy.decode["attention"] = "gpu"
        plan = dense_plan()
        self.assertIs(policy.apply(plan), plan)
        placements = plan.compile().ir_document()["placements"]
        self.assertEqual(len(placements), 6)
        self.assertEqual({row["selector"]["phase"] for row in placements}, {"prefill", "decode"})
        self.assertEqual(policy.document()["resource_scope"], "shared_cpu_gpu_phase_placement")

    def test_pd_uses_actual_kv_copies_one_clock_and_persistent_ids(self):
        plan = dense_plan()
        kinds = ("projection", "attention", "dense_ffn")
        PhasePlacement(prefill={kind: "gpu" for kind in kinds},
                       decode={kind: "cpu" for kind in kinds}).apply(plan)
        compiled = plan.compile()
        run, session, _, _ = compiled.run()
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        self.assertEqual(run.evaluation.status, "valid", run.evaluation.reason)
        compute = [record for record in run.decisions if record.action.kind == "COMPUTE"]
        for record in compute:
            operation = next(op for op in record.observation.operations if op.id == record.action.target_id)
            if operation.kind != "state_update":
                self.assertEqual(record.action.device, "gpu" if operation.phase == "prefill" else "cpu")
        events = run.evaluation.events
        copies = [event for event in events if event.kind == "COPY" and
                  event.target_id.startswith("kv:") and event.src == "vram" and event.dst == "dram"]
        self.assertGreaterEqual(len(copies), 6)
        self.assertTrue(all(event.bytes_transferred > 0 and event.end_ns > event.start_ns for event in copies))
        final = session.persistent_state_document()
        self.assertEqual(final["t_ns"], run.evaluation.makespan_ns)
        self.assertEqual(len(final["kv_ids"]), 8)
        self.assertEqual(len(final["kv_ids"]), len(set(final["kv_ids"])))
        self.assertEqual(len(final["completed_invocations"]), 4)

    def test_invalid_phase_contract_fails_before_mutating_plan(self):
        for bindings in ({}, {"unknown": "gpu"}, {"attention": "gpu1"}):
            error = InvalidInput if not bindings else Unsupported
            with self.assertRaises(error):
                PhasePlacement(prefill=bindings, decode={"attention": "cpu"})
        phase = PhasePlacement(prefill={"attention": "gpu"}, decode={"attention": "cpu"})
        with self.assertRaises(InvalidInput):
            phase.apply(object())
        with self.assertRaises(InvalidInput):
            phase.apply(dense_plan(), priority=True)


class PreparationRuntimeTests(unittest.TestCase):
    def test_startup_rejects_custom_control_override_explicitly(self):
        plan = dense_plan(device="cpu")
        plan.prepare(DataPreparation(layer_id="denseL1", trigger="startup", max_count=1))
        plan.runtime_policy(CustomControlProbe, allow_control_override=True)
        with self.assertRaisesRegex(Unsupported, "startup preparation requires the built-in resource controller"):
            plan.compile()

    def test_spare_keeps_custom_control_authority_and_defers_normal_work(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "control.environment.json"
            environment.write_text(json.dumps(dict(format_version="environment-trace-0.2", events=[],
                requests=[dict(id="public-request", domain="vram", t_notice_ns=0,
                               target_external_bytes=64, deadline_ns=1000)],
                control_costs=dict(ack_ns=3, decline_ns=7, decision_ns=2,
                    source="synthetic_control_test", timing_scope="dedicated_serial_control_resource"))),
                encoding="utf-8", newline="\n")
            plan = dense_plan(device="cpu", environment_file=environment,
                reconfiguration=ReconfigurationSpec("release_clean_weights", "lru", "fixed"))
            plan.prepare(DataPreparation(layer_id="denseL1", trigger="spare", max_count=1))
            plan.runtime_policy(CustomControlProbe, allow_control_override=True)
            run, _, policy, _ = plan.compile().run()
            self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
            self.assertEqual(run.evaluation.status, "valid", run.evaluation.reason)
            first = run.decisions[0]
            self.assertIsInstance(first.decision, ControlDecision)
            self.assertEqual(first.decision.kind, "DECLINE_RECLAIM")
            self.assertEqual(first.decision.reason, "test_custom_controller")
            self.assertTrue(first.accepted)
            audit = policy.audit_document()
            self.assertEqual(audit["python_state"]["declines"], 1)
            self.assertTrue(any(row["decision"]["kind"] == "DECLINE_RECLAIM" for row in audit["python_decisions"]))
            self.assertTrue(audit["explicit_data_preparations"])

    def test_startup_is_charged_before_any_compute_and_resolves_max_count_once(self):
        plan = dense_plan(device="cpu")
        rule = DataPreparation(layer_id="denseL1", trigger="startup", max_count=1)
        plan.prepare(rule)
        compiled = plan.compile()
        self.assertEqual(compiled.ir_document()["data_preparations"], [rule.document()])
        run, session, policy, _ = compiled.run()
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        records = policy.audit_document()["explicit_data_preparations"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trigger"], "startup")
        target = records[0]["tensor_id"]
        copies = [event for event in run.evaluation.events if event.kind == "COPY" and event.target_id == target]
        self.assertEqual(len(copies), 1)
        first_compute = next(event for event in run.evaluation.events if event.kind == "COMPUTE")
        self.assertGreater(copies[0].bytes_transferred, 0)
        self.assertGreater(copies[0].end_ns, copies[0].start_ns)
        self.assertGreaterEqual(first_compute.start_ns, copies[0].end_ns)
        self.assertEqual(session.persistent_state_document()["t_ns"], run.evaluation.makespan_ns)

    def test_spare_preparation_uses_only_spare_budget_without_eviction(self):
        for budget in (200000, 1):
            with self.subTest(cache_budget=budget):
                plan = dense_plan(device="cpu", cache_budget=budget)
                plan.prepare(DataPreparation(layer_id="denseL1", trigger="spare", max_count=1))
                run, _, policy, _ = plan.compile().run()
                self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
                records = policy.audit_document()["explicit_data_preparations"]
                self.assertEqual(len(records), 1 if budget > 1 else 0)
                self.assertTrue(all(row["action"]["kind"] == "COPY" for row in records))
                # Normal boundary cleanup can still release dead activations.
                # Speculation must not evict a weight to obtain cache space.
                self.assertFalse(any(event.kind == "EVICT" and event.target_id.startswith("w:")
                                     for event in run.evaluation.events))

    def test_startup_budget_failure_is_loud_before_compute(self):
        plan = dense_plan(device="cpu", cache_budget=1)
        plan.prepare(DataPreparation(tensor_ids=("w:denseL0:W_down",), trigger="startup"))
        run, _, _, _ = plan.compile().run()
        self.assertEqual(run.termination_reason, "decision_rejected")
        self.assertIn("cache budget", run.compilation_error)
        self.assertFalse(any(event.kind == "COMPUTE" for event in run.evaluation.events))

    def test_startup_physical_capacity_failure_is_loud_before_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = Path(directory) / "fixtures"
            shutil.copytree(ROOT / "modeling/tests/fixtures/global_execution", fixtures)
            for path in fixtures.glob("denseL*.prefill.json"):
                document = json.loads(path.read_text(encoding="utf-8"))
                document["hardware"]["vram_capacity_bytes"] = 1024
                path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            for path in fixtures.glob("denseL*.decode.json"):
                document = json.loads(path.read_text(encoding="utf-8"))
                document["hardware"]["vram_capacity_bytes"] = 1024
                path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            plan = dense_plan(device="cpu", model_file=fixtures / "dense.global.json")
            plan.prepare(DataPreparation(tensor_ids=("w:denseL0:W_down",), trigger="startup"))
            run, _, _, _ = plan.compile().run()
            self.assertEqual(run.termination_reason, "decision_rejected")
            self.assertIn("physical capacity", run.compilation_error)
            self.assertFalse(any(event.kind == "COMPUTE" for event in run.evaluation.events))

    def test_ssd_startup_uses_two_charged_hops_before_compute(self):
        # This existing MoE execution manifest declares an immutable SSD backing
        # copy with exact prices; the Dense model schema itself starts in RAM.
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/moe.strategy.json")
        target = "w:fragmentL1:E3:v1:gate"
        plan.prepare(DataPreparation(tensor_ids=(target,), trigger="startup"))
        run, _, policy, _ = plan.compile().run(max_decisions=20)
        self.assertFalse(run.compilation_error)
        hops = [row["action"]["kind"] for row in policy.audit_document()["explicit_data_preparations"]]
        self.assertEqual(hops, ["READ_SSD", "COPY"])
        events = [event for event in run.evaluation.events if event.kind in {"READ_SSD", "COPY"} and
                  event.target_id == target]
        self.assertEqual([event.kind for event in events], ["READ_SSD", "COPY"])
        self.assertTrue(all(event.bytes_transferred == 128 and event.end_ns > event.start_ns for event in events))
        self.assertGreaterEqual(run.evaluation.ssd_read_bytes, 128)
        self.assertGreaterEqual(run.evaluation.h2d_bytes, 128)
        first_compute = next(event for event in run.evaluation.events if event.kind == "COMPUTE")
        self.assertGreaterEqual(first_compute.start_ns, events[-1].end_ns)

    def test_inactive_expert_can_be_prepared_from_static_catalogue_before_router(self):
        plan = ModelPlan.from_strategy(EXAMPLES / "composition/moe.strategy.json")
        plan.prepare(DataPreparation(layer_id="fragmentL1", stage="gate", expert_id=3, trigger="startup"))
        run, _, policy, _ = plan.compile().run(max_decisions=2)
        self.assertFalse(run.compilation_error)
        self.assertEqual(run.decisions[0].observation.route_history, ())
        self.assertFalse(any(op.kind.startswith("expert_") for op in run.decisions[0].observation.operations))
        first = run.decisions[0]
        self.assertEqual(first.action.kind, "READ_SSD")
        self.assertEqual(first.action.target_id, "w:fragmentL1:E3:v1:gate")
        self.assertEqual(policy.audit_document()["explicit_data_preparations"][0]["trigger"], "startup")

    def test_preparation_is_fingerprint_bound_and_checkpoint_replays_same_ledger(self):
        plan = dense_plan(device="cpu")
        plan.prepare(DataPreparation(layer_id="denseL1", trigger="startup", max_count=1))
        compiled = plan.compile()
        full = compiled.run()
        partial = compiled.run(max_decisions=1)
        checkpoint = compiled.checkpoint(partial)
        resumed = compiled.run(checkpoint=checkpoint)
        self.assertEqual(full[0].termination_reason, "goal_reached")
        self.assertEqual(_runtime_signature(full[1]), _runtime_signature(resumed[1]))
        self.assertEqual(full[2].audit_document(), resumed[2].audit_document())
        other = dense_plan(device="cpu")
        other.prepare(DataPreparation(layer_id="denseL1", trigger="startup", max_count=2))
        with self.assertRaisesRegex(InvalidInput, "changed"):
            other.compile().run(checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
