"""Causal policy noninterference and model-owned MoE route revelation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import unittest

from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.expand import load_and_compile, compile_invocation
from llm_infer_model.m2.costs import MacroShape, MacroCost
from llm_infer_model.m2.moe_spec import RouteTraceEntry
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_READ_SSD, Action,
    STATUS_VALID,
)
from llm_infer_model.tensor.environment import EnvironmentEvent, CODE_INFORMATION_NOT_KNOWN
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported
from tensor_mapping.causal import (
    Decision, SerialDemandPolicy, compile_decision, observation_document,
    observation_hash, run_causal,
)


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples/m2"


def compiled():
    return load_and_compile(_EXAMPLES / "moe-interface-synthetic-prefill-2.json")


def alternative(compilation):
    invocation = replace(compilation.invocation, route_trace=(
        RouteTraceEntry("r0", "moe-small", 0, (2, 3)),
        RouteTraceEntry("r0", "moe-small", 1, (3, 0)),
    ))
    return compile_invocation(compilation.layer, invocation, compilation.cost_table)


def two_request_compiled(*, change_second=False):
    source = compiled()
    requests = (source.invocation.requests[0], replace(source.invocation.requests[0], request_id="r1"))
    history = replace(source.state_in, requests=(source.state_in.requests[0],
                                                replace(source.state_in.requests[0], request_id="r1")))
    routes = (
        RouteTraceEntry("r0", "moe-small", 0, (0, 1)),
        RouteTraceEntry("r0", "moe-small", 1, (0, 1)),
        RouteTraceEntry("r1", "moe-small", 0, (0, 2) if change_second else (2, 3)),
        RouteTraceEntry("r1", "moe-small", 1, (1, 3) if change_second else (2, 3)),
    )
    invocation = replace(source.invocation, requests=requests, state_in=history, route_trace=routes)
    # This test supplies explicit synthetic batch-2 costs. No interpolation or
    # claim that batch-1 measurements apply to physical batch-2 kernels.
    entries = {(kind, device, replace(shape, batch=2)): price
               for (kind, device, shape), price in source.cost_table.entries.items()}
    for device, duration in (("cpu", 15_000_000), ("gpu", 7_500_000)):
        entries[("expert_ffn", device, MacroShape("prefill", 2, 3, 0))] = MacroCost(
            duration, "synthetic_layer", 16, "per_expert_group")
    table = replace(source.cost_table, entries=entries)
    # A three-token grouped output is 48 bytes: this is an exact explicit toy
    # transfer entry, necessary only after the second request's router reveals.
    transfers = dict(table.transfers)
    for route in ("dram_to_vram", "vram_to_dram", "dram_to_ssd", "ssd_to_dram"):
        transfers[(route, 48)] = transfers[(route, 32)]
    table = replace(table, transfers=transfers)
    return compile_invocation(source.layer, invocation, table)


class CausalMappingTests(unittest.TestCase):
    def test_static_catalog_includes_inactive_experts_without_dynamic_graph(self):
        observation = CausalM2Session(compiled()).observe()
        self.assertEqual({op.kind for op in observation.operations}, {"projection", "attention", "router"})
        self.assertEqual(observation.route_history, ())
        self.assertEqual(tuple((expert.expert_id, expert.weight_tensor_id) for expert in observation.experts),
                         tuple((index, f"w:moe-small:E{index}") for index in range(4)))
        self.assertTrue(all(not operation.work_items and operation.expert_id is None
                            for operation in observation.operations))
        self.assertEqual({tensor.id for tensor in observation.tensors if tensor.role == "weight"},
                         {f"w:moe-small:{name}" for name in ("W_proj", "W_router", "E0", "E1", "E2", "E3")})
        self.assertFalse(any("expert." in tensor.id or "combine" in tensor.id for tensor in observation.tensors))
        for tensor in observation.tensors:
            if tensor.role == "weight":
                self.assertEqual(len(tensor.transfers), 4)
        self.assertFalse(hasattr(observation, "scenario"))
        self.assertFalse(hasattr(observation, "invocation"))
        self.assertFalse(hasattr(observation, "route_trace"))

    def test_same_revealed_prefix_different_future_routes_has_identical_decisions(self):
        source = compiled()
        sessions = (CausalM2Session(source), CausalM2Session(alternative(source)))
        policy = SerialDemandPolicy("cpu")
        for _ in range(6):
            observations = tuple(session.observe() for session in sessions)
            self.assertEqual(observations[0], observations[1])
            self.assertEqual(observation_hash(observations[0]), observation_hash(observations[1]))
            self.assertEqual(observations[0].route_history, ())
            self.assertFalse(any(op.kind in ("expert_ffn", "combine") for op in observations[0].operations))
            decisions = tuple(policy(observation) for observation in observations)
            self.assertEqual(decisions[0], decisions[1])
            for session, observation, decision in zip(sessions, observations, decisions):
                action = compile_decision(decision, observation)
                step = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok)
        revealed = tuple(session.observe() for session in sessions)
        self.assertEqual(revealed[0].t_ns, 4_000_000)
        self.assertNotEqual(revealed[0].route_history, revealed[1].route_history)
        self.assertEqual(len(revealed[0].route_history), 2)
        self.assertTrue(any(op.kind == "combine" for op in revealed[0].operations))
        groups = {op.expert_id: op for op in revealed[0].operations if op.kind == "expert_ffn"}
        self.assertEqual(groups[1].work_items, (("r0", 0), ("r0", 1)))
        self.assertEqual(groups[1].token_count, 2)
        self.assertEqual(groups[1].phase, "prefill")

    def test_partial_request_routes_reveal_without_leaking_other_request_groups(self):
        sessions = (CausalM2Session(two_request_compiled()),
                    CausalM2Session(two_request_compiled(change_second=True)))
        policy = SerialDemandPolicy("cpu")
        while not sessions[0].observe().route_history:
            observations = tuple(session.observe() for session in sessions)
            self.assertEqual(observations[0], observations[1])
            decisions = tuple(policy(observation) for observation in observations)
            self.assertEqual(decisions[0], decisions[1])
            for session, observation, decision in zip(sessions, observations, decisions):
                action = compile_decision(decision, observation)
                step = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok)
        observations = tuple(session.observe() for session in sessions)
        self.assertEqual(observations[0], observations[1])
        self.assertEqual({row.request_id for row in observations[0].route_history}, {"r0"})
        self.assertFalse(any(op.kind in ("combine", "expert_ffn") for op in observations[0].operations))
        self.assertFalse(any("expert." in tensor.id for tensor in observations[0].tensors))

    def test_immutable_observation_and_no_private_future_fields(self):
        session = CausalM2Session(compiled(), environment_events=(
            EnvironmentEvent(100_000_000, cpu_available=False),))
        observation = session.observe()
        with self.assertRaises(FrozenInstanceError):
            observation.t_ns = 9
        with self.assertRaises(FrozenInstanceError):
            observation.operations[0].status = "DONE"
        document = observation_document(observation)
        text = json.dumps(document)
        for private in ("remaining_ns", "nominal_ends", "environment_events", "route_trace", "scenario", "expert_work_items"):
            self.assertNotIn(private, text)
        self.assertNotIn("100000000", text)

    def test_compiler_and_runtime_refuse_unknown_expert_compute_before_reveal(self):
        session = CausalM2Session(compiled())
        action = Decision(ACTION_COMPUTE, "op:moe-small:__moe_batch__:expert.0", device="cpu")
        with self.assertRaisesRegex(Unsupported, "has not been revealed"):
            compile_decision(action, session.observe())
        refused = session.submit(Action(ACTION_COMPUTE, action.target_id, device="cpu"))
        self.assertFalse(refused.ok)
        self.assertEqual(refused.error_code, CODE_INFORMATION_NOT_KNOWN)

    def test_history_prediction_can_load_finally_inactive_expert_and_pays_cost(self):
        session = CausalM2Session(compiled())
        baseline = run_causal(SerialDemandPolicy("cpu"), CausalM2Session(compiled()), max_decisions=100)
        # E3 is in the static catalog but never in this invocation's actual trace.
        for decision in (Decision(ACTION_READ_SSD, "w:moe-small:E3"), Decision(ACTION_ADVANCE),
                         Decision(ACTION_COPY, "w:moe-small:E3", src="dram", dst="vram"),
                         Decision(ACTION_ADVANCE)):
            action = compile_decision(decision, session.observe())
            step = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
            self.assertTrue(step.ok, step.error_message)
        early = session.observe()
        self.assertEqual(early.route_history, ())
        self.assertEqual(early.t_ns, 3_100_000)
        expert = next(tensor for tensor in early.tensors if tensor.id == "w:moe-small:E3")
        self.assertEqual(expert.status_at("dram"), COPY_READY)
        self.assertEqual(expert.status_at("vram"), COPY_READY)
        run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=100)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.evaluation.makespan_ns, baseline.evaluation.makespan_ns + 3_100_000)
        self.assertEqual(run.evaluation.ssd_read_bytes, baseline.evaluation.ssd_read_bytes + 64)
        self.assertEqual(run.evaluation.h2d_bytes, baseline.evaluation.h2d_bytes + 64)
        self.assertEqual(run.evaluation.peak_vram_bytes, 64)
        self.assertFalse(any(3 in row.expert_ids for row in session.observe().route_history))
        cache = next(weight for weight in session.state_out().weight_copies if weight.id == "E3")
        self.assertEqual(set(cache.locations), {"dram", "vram", "ssd"})

    def test_fixed_cpu_baseline_completes_and_records_public_evidence(self):
        arguments = []
        baseline = SerialDemandPolicy("cpu")
        def policy(observation):
            arguments.append(observation)
            self.assertFalse(hasattr(observation, "_compiled"))
            self.assertFalse(hasattr(observation, "scenario"))
            return baseline(observation)
        session = CausalM2Session(compiled())
        run = run_causal(policy, session, max_decisions=100)
        self.assertEqual(run.termination_reason, "goal_reached")
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.evaluation.makespan_ns, 34_000_000)
        self.assertEqual(len(arguments), len(run.decisions))
        self.assertEqual(run.runtime_result.action_sources, tuple(range(len(run.decisions))))
        for record in run.decisions:
            self.assertEqual(record.observation_sha256, observation_hash(record.observation))
            self.assertEqual(record.action, compile_decision(record.decision, record.observation))
            self.assertTrue(record.accepted)
        self.assertEqual(session.state_out().requests[0].next_position, 2)

    def test_future_environment_timelines_are_not_visible_at_same_prefix(self):
        source = compiled()
        sessions = (
            CausalM2Session(source, environment_events=(EnvironmentEvent(500_000, cpu_available=False),
                                                       EnvironmentEvent(2_500_000, cpu_available=True))),
            CausalM2Session(source, environment_events=(EnvironmentEvent(500_000, cpu_available=False),
                                                       EnvironmentEvent(3_500_000, cpu_available=True))),
        )
        policy = SerialDemandPolicy("cpu")
        self.assertEqual(sessions[0].observe(), sessions[1].observe())
        for session in sessions:
            decision = policy(session.observe())
            self.assertTrue(session.submit(compile_decision(decision, session.observe())).ok)
            self.assertTrue(session.advance().ok)
        self.assertEqual(sessions[0].observe(), sessions[1].observe())
        self.assertFalse(sessions[0].observe().cpu_available)
        decisions = tuple(policy(session.observe()) for session in sessions)
        self.assertEqual(decisions[0], decisions[1])
        for session in sessions:
            self.assertTrue(session.advance().ok)
        self.assertNotEqual(sessions[0].observe().t_ns, sessions[1].observe().t_ns)

    def test_running_cpu_blackout_delays_completion_without_reset(self):
        session = CausalM2Session(compiled(), environment_events=(
            EnvironmentEvent(500_000, cpu_available=False, dram_external_bytes=128),
            EnvironmentEvent(2_500_000, cpu_available=True, dram_external_bytes=0),
        ))
        run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=100)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.evaluation.makespan_ns, 36_000_000)
        blackouts = [row.observation for row in run.decisions if not row.observation.cpu_available]
        self.assertTrue(blackouts)
        self.assertEqual(blackouts[0].t_ns, 500_000)
        self.assertEqual(blackouts[0].route_history, ())
        self.assertEqual(blackouts[0].memory[0].external_bytes, 128)
        self.assertTrue(blackouts[0].running)
        self.assertFalse(hasattr(blackouts[0].running[0], "remaining_ns"))

    def test_gpu_baseline_uses_explicit_routes_and_returns_actual_kv_state(self):
        session = CausalM2Session(compiled())
        run = run_causal(SerialDemandPolicy("gpu"), session, max_decisions=150)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertGreater(run.evaluation.ssd_read_bytes, 0)
        self.assertGreater(run.evaluation.h2d_bytes, 0)
        self.assertGreater(run.evaluation.d2h_bytes, 0)
        state = session.state_out()
        self.assertEqual(state.requests[0].next_position, 2)
        self.assertTrue(all(block.locations == ("dram", "vram") for block in state.requests[0].blocks))

    def test_cost_for_inactive_expert_must_be_exact_not_guessed(self):
        original = compiled()
        # Remove a required nominal 64-byte route from the table;
        # prior offline expansion still exists but causal adapter rebinds costs.
        transfers = dict(original.cost_table.transfers)
        transfers.pop(("ssd_to_dram", 64))
        broken = replace(original, cost_table=replace(original.cost_table, transfers=transfers))
        with self.assertRaises(Unsupported):
            CausalM2Session(broken)

    def test_explicit_budget_and_malformed_decisions_are_refused(self):
        session = CausalM2Session(compiled())
        for budget in (0, True, None):
            with self.assertRaises(InvalidInput):
                run_causal(SerialDemandPolicy("cpu"), session, max_decisions=budget)
        with self.assertRaises(InvalidInput):
            compile_decision(Decision(ACTION_ADVANCE, "hidden-future-task"), session.observe())
        with self.assertRaises(Unsupported):
            compile_decision(Decision("PREDICT_FUTURE_ROUTE"), session.observe())
        with self.assertRaises(InvalidInput):
            compile_decision(Decision([]), session.observe())

    def test_rejected_compilation_preserves_observation_and_decision(self):
        decision = Decision(ACTION_COMPUTE, "unrevealed-expert", device="cpu")
        run = run_causal(lambda observation: decision, CausalM2Session(compiled()), max_decisions=10)
        self.assertEqual(run.termination_reason, "decision_rejected")
        self.assertIsNotNone(run.compilation_error)
        self.assertEqual(len(run.decisions), 1)
        self.assertEqual(run.decisions[0].decision, decision)
        self.assertIsNone(run.decisions[0].action)
        self.assertFalse(run.decisions[0].accepted)
        self.assertEqual(run.decisions[0].error_code, "unsupported_decision")
        self.assertEqual(run.decisions[0].observation_sha256,
                         observation_hash(run.decisions[0].observation))

    def test_policy_stop_and_budget_exhaustion_are_not_success(self):
        run = run_causal(lambda observation: None, CausalM2Session(compiled()), max_decisions=10)
        self.assertEqual(run.termination_reason, "policy_stopped")
        self.assertNotEqual(run.evaluation.status, STATUS_VALID)
        run = run_causal(SerialDemandPolicy("cpu"), CausalM2Session(compiled()), max_decisions=1)
        self.assertEqual(run.termination_reason, "decision_budget_exhausted")
        self.assertNotEqual(run.evaluation.status, STATUS_VALID)


if __name__ == "__main__":
    unittest.main()
