"""Causal visibility, explicit engines and the common runtime ledger together."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json
from pathlib import Path
import unittest

from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.expand import load_and_compile
from llm_infer_model.m2.row_execution import parse_execution_spec
from llm_infer_model.tensor.engine import Action, STATUS_VALID
from llm_infer_model.tensor.environment import EnvironmentEvent, CODE_INFORMATION_NOT_KNOWN
from llm_infer_model.tensor.runtime import ExecutionCommand, dispatch_command
from llm_infer_model.tensor.services import CODE_SLOT_OCCUPIED, parse_services
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported
from tensor_mapping.causal import Decision, SerialDemandPolicy, compile_command, compile_decision, run_causal
from .test_causal_mapping import compiled, alternative


def service_contract(*, slots=()):
    return parse_services({"format_version": "tensor-services-0.1", "pcie": {
        "bandwidth_bytes_per_second": 10**9, "ce_bytes_per_second": 10**9,
        "sm_bytes_per_second": 10**9, "arbitration": "max_min_fair",
        "timing_scope": "payload_only_no_launch_latency"}, "slots": list(slots)})


def transfer(tid, engine):
    return Decision("COPY", tid, src="dram", dst="vram", transfer_engine=engine)


def execute(session, decision, index=0):
    return dispatch_command(session, compile_command(decision, session.observe()), source_index=index)


class UnifiedExecutionTests(unittest.TestCase):
    def test_no_engine_inference_and_no_legacy_metadata_loss(self):
        session = CausalM2Session(compiled(), services=service_contract())
        with self.assertRaisesRegex(InvalidInput, "explicit transfer_engine"):
            compile_command(transfer("w:moe-small:W_proj", None), session.observe())
        with self.assertRaisesRegex(Unsupported, "compile_command"):
            compile_decision(transfer("w:moe-small:W_proj", "ce"), session.observe())
        with self.assertRaisesRegex(Unsupported, "services-enabled"):
            compile_command(transfer("w:moe-small:W_proj", "ce"), CausalM2Session(compiled()).observe())
        with self.assertRaises(InvalidInput):
            compile_command(Decision("COMPUTE", session.observe().operations[0].id,
                device="cpu", transfer_engine="ce"), session.observe())
        self.assertEqual(session.result().evaluation.actions, ())

    def test_engine_free_legacy_baseline_is_unchanged(self):
        session = CausalM2Session(compiled())
        run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=100)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.evaluation.makespan_ns, 34_000_000)
        self.assertIsNone(session.observe().services)
        self.assertEqual(tuple(range(len(run.decisions))), run.runtime_result.action_sources)

    def test_common_envelope_runs_gpu_baseline_with_explicit_engines(self):
        session = CausalM2Session(compiled(), services=service_contract())
        baseline = SerialDemandPolicy("gpu")
        def policy(observation):
            decision = baseline(observation)
            return replace(decision, transfer_engine="ce") if decision.kind == "COPY" else decision
        run = run_causal(policy, session, max_decisions=180)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.termination_reason, "goal_reached")
        choices = session.service_document()["transfer_engines"]
        self.assertTrue(choices)
        for item in choices:
            self.assertEqual(item["engine"], "ce")
            record = run.decisions[item["source_index"]]
            self.assertEqual(record.action.target_id, item["tensor_id"])
            self.assertEqual(record.decision.transfer_engine, item["engine"])
        self.assertEqual(session.state_out().requests[0].next_position, 2)

    def test_shared_flow_progress_and_environment_use_same_clock(self):
        session = CausalM2Session(compiled(), services=service_contract(),
            environment_events=(EnvironmentEvent(16, cpu_available=False, vram_external_bytes=32),))
        self.assertTrue(execute(session, transfer("w:moe-small:W_proj", "ce"), 7).ok)
        self.assertTrue(execute(session, transfer("w:moe-small:W_router", "sm"), 8).ok)
        self.assertEqual(execute(session, Decision("ADVANCE"), 9).t_ns, 16)
        observed = session.observe()
        self.assertFalse(observed.cpu_available)
        self.assertEqual(observed.route_history, ())
        flows = {flow.engine: flow for flow in observed.services.flows}
        self.assertEqual(Fraction(flows["ce"].remaining_bytes_numerator,
                                  flows["ce"].remaining_bytes_denominator), 24)
        self.assertEqual(Fraction(flows["sm"].remaining_bytes_numerator,
                                  flows["sm"].remaining_bytes_denominator), 8)
        self.assertEqual(execute(session, Decision("ADVANCE"), 10).t_ns, 32)
        self.assertEqual(execute(session, Decision("ADVANCE"), 11).t_ns, 48)
        self.assertTrue(all(next(t for t in session.observe().tensors if t.id == tid).status_at("vram")
            == COPY_READY for tid in ("w:moe-small:W_proj", "w:moe-small:W_router")))
        self.assertEqual(session.result().evaluation.h2d_bytes, 48)
        self.assertEqual(session.result().action_sources, (7, 8, 9, 10, 11))
        self.assertEqual(session.observe().memory[1].external_bytes, 32)
        self.assertEqual(session.observe().route_history, ())

    def test_slot_contract_is_visible_and_rejects_overwrite(self):
        slot = dict(id="weight-slot", domain="vram", capacity_bytes=32,
                    members=["w:moe-small:W_proj", "w:moe-small:W_router"])
        session = CausalM2Session(compiled(), services=service_contract(slots=[slot]))
        observed = session.observe()
        self.assertEqual(observed.services.slots[0].owner, None)
        self.assertEqual(observed.memory[1].runtime_reserved_bytes, 32)
        self.assertTrue(execute(session, transfer("w:moe-small:W_proj", "ce")).ok)
        self.assertEqual(session.observe().services.slots[0].owner, "w:moe-small:W_proj")
        rejected = execute(session, transfer("w:moe-small:W_router", "sm"))
        self.assertFalse(rejected.ok)
        self.assertEqual(rejected.error_code, CODE_SLOT_OCCUPIED)

    def test_dynamic_slot_members_and_future_routes_have_identical_public_prefix(self):
        sources = (compiled(), alternative(compiled()))
        sessions = []
        for source in sources:
            hidden = next(task.output_tensor for task in source.tasks if task.kind == "expert_ffn")
            slot = dict(id="dynamic-output-slot", domain="vram", capacity_bytes=32, members=[hidden])
            sessions.append(CausalM2Session(source, services=service_contract(slots=[slot]),
                environment_events=(EnvironmentEvent(10**9, cpu_available=False),)))
        self.assertEqual(sessions[0].observe(), sessions[1].observe())
        self.assertEqual(sessions[0].observe().services.slots[0].members, ())
        self.assertEqual(sessions[0].observe().services.slots[0].owner, None)
        baseline = SerialDemandPolicy("cpu")
        for _ in range(6):
            observed = tuple(session.observe() for session in sessions)
            self.assertEqual(observed[0], observed[1])
            for session, view in zip(sessions, observed):
                self.assertTrue(execute(session, baseline(view)).ok)
        self.assertNotEqual(sessions[0].observe().route_history, sessions[1].observe().route_history)

    def test_runtime_does_not_reveal_unseen_action_via_service_slot(self):
        session = CausalM2Session(compiled(), services=service_contract())
        private = next(task for task in compiled().tasks if task.kind == "expert_ffn")
        with self.assertRaisesRegex(Unsupported, "not been revealed"):
            compile_command(transfer(private.output_tensor, "ce"), session.observe())
        step = session.submit_command(ExecutionCommand(Action("COPY", private.output_tensor,
            src="dram", dst="vram"), "ce"))
        self.assertEqual(step.error_code, CODE_INFORMATION_NOT_KNOWN)
        self.assertEqual(session.service_document()["transfer_engines"], [])

    def test_service_prices_are_explicit_isolated_estimates_and_views_are_frozen(self):
        observed = CausalM2Session(compiled(), services=service_contract()).observe()
        tensor = next(t for t in observed.tensors if t.id == "w:moe-small:W_proj")
        price = next(p for p in tensor.transfers if p.route == "dram_to_vram")
        self.assertIsNone(price.duration_ns)
        self.assertEqual(price.engine_durations_ns, (("ce", 32), ("sm", 32)))
        self.assertEqual(price.timing_scope, "isolated_payload_service_subject_to_contention")
        self.assertEqual(next(p for p in tensor.transfers if p.route == "ssd_to_dram").duration_ns, 3_000_000)
        with self.assertRaises(FrozenInstanceError):
            observed.services.bandwidth_bytes_per_second = 0
        self.assertTrue(all(op.layer_id == "moe-small" for op in observed.operations))
        self.assertTrue(all(op.invocation_id == compiled().invocation.id for op in observed.operations))

    def test_subweight_catalogue_and_persistent_inventory_never_recreate_parent(self):
        fixture = Path(__file__).resolve().parents[2] / "modeling/tests/fixtures/subweights"
        source = load_and_compile(fixture / "invocation.json")
        spec = parse_execution_spec(json.loads((fixture / "execution.json").read_text(encoding="utf-8")))
        session = CausalM2Session(source, execution_spec=spec, services=service_contract())
        view = session.observe()
        self.assertTrue(all(expert.weight_tensor_id is None for expert in view.experts))
        self.assertTrue(all(len(expert.weight_tensor_ids) == 3 for expert in view.experts))
        public = {tensor.id for tensor in view.tensors}
        self.assertTrue(all(set(expert.weight_tensor_ids) <= public for expert in view.experts))
        self.assertFalse(any(task.kind.startswith("expert_") for task in view.operations))
        with self.assertRaisesRegex(Unsupported, "not been revealed"):
            compile_command(Decision("READ_SSD", "w:moe-small:E0"), view)
        def policy(observation):
            if observation.running:
                return Decision("ADVANCE")
            tensors = {tensor.id: tensor for tensor in observation.tensors}
            pending = next((op for op in observation.operations if op.status == "NOT_STARTED"
                and all(any(status == COPY_READY for _, status in tensors[tid].copies)
                        for tid in op.inputs)), None)
            device = "gpu" if pending is not None and pending.kind in ("expert_gate", "expert_up") else "cpu"
            decision = SerialDemandPolicy(device)(observation)
            return replace(decision, transfer_engine="ce") if decision.kind == "COPY" else decision
        run = run_causal(policy, session, max_decisions=250)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        with self.assertRaisesRegex(Unsupported, "independently resident fragments"):
            session.state_out()
        inventory = session.persistent_state_document()
        self.assertEqual(inventory["format_version"], "causal-persistent-state-0.1")
        self.assertEqual(len(inventory["weight_fragments"]), 12)
        entries = {row["tensor_id"]: row for row in inventory["weight_fragments"]}
        for index in range(3):
            self.assertIn("vram", entries[f"w:moe-small:E{index}:v1:gate"]["locations"])
            self.assertNotIn("vram", entries[f"w:moe-small:E{index}:v1:down"]["locations"])
        for stage in ("gate", "up", "down"):
            self.assertEqual(entries[f"w:moe-small:E3:v1:{stage}"]["locations"], ["ssd"])
        self.assertEqual(sum(row["storage_bytes"] for row in entries.values()), 4*384)
        self.assertEqual({row["tensor_id"] for row in inventory["whole_weights"]},
                         {"w:moe-small:W_proj", "w:moe-small:W_router"})
        self.assertEqual(sum(row["storage_bytes"] for row in inventory["whole_weights"]), 48)


if __name__ == "__main__":
    unittest.main()
