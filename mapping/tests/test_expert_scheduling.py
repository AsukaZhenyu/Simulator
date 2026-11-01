"""Public expert scheduling semantics, current-rate ranking and causal guards."""
from copy import deepcopy
from dataclasses import replace
import unittest

from llm_infer_model.m2.causal import CausalM2Session, CausalObservation, DevicePrice, PublicOperation, RevealedRoute
from llm_infer_model.tensor.engine import ACTION_ADVANCE
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.causal import compile_decision, run_causal
from tensor_mapping.expert_scheduling import ExpertSchedulingSpec, order_expert_operations, rank_expert_devices
from tensor_mapping.policy_ir import parse_policy, policy_document
from tensor_mapping.policy_executor import DeclarativePolicy
from .test_causal_mapping import compiled, alternative
from .test_policy_ir import policy_document_fixture


def observation(*, staged=False):
    prices = (DevicePrice("cpu", 20, 0, 0), DevicePrice("gpu", 80, 0, 0))
    router = PublicOperation("router", "router", "DONE", (), (), prices)
    combine = PublicOperation("combine", "combine", "NOT_STARTED", (), (), prices)
    operations = [router]
    for expert, rows in ((0, (("r0", 0),)), (1, (("r1", 0), ("r1", 1)))):
        for stage in (("expert_gate", "expert_up", "expert_swiglu", "expert_down") if staged else ("expert_ffn",)):
            operations.append(PublicOperation(f"e{expert}.{stage}", stage, "NOT_STARTED", (), (), prices,
                expert_id=expert, work_items=rows, token_count=len(rows), phase="prefill", request_id="__batch__",
                region_id=f"expert:{expert}", execution_stage=stage, layer_id="L0", invocation_id="round0"))
    operations.append(combine)
    routes = (RevealedRoute("r0", "L0", 0, (0,), None),
              RevealedRoute("r1", "L0", 0, (1,), None), RevealedRoute("r1", "L0", 1, (1,), None))
    return CausalObservation(100, tuple(operations), (), (), (), True, routes, (), False, None)


def paths():
    return [dict(device=device, compute_ns=duration, weight_transfer_ns=10 if device == "cpu" else 0,
                 selected_input_transfer_ns=0, output_return_ns=0,
                 budget_feasible=True, physical_feasible=True)
            for device, duration in (("cpu", 20), ("gpu", 80))]


class ExpertSchedulingTests(unittest.TestCase):
    def test_version_three_roundtrip_and_legacy_identity(self):
        old = policy_document_fixture()
        self.assertEqual(policy_document(parse_policy(old)), old)
        newer = deepcopy(old)
        newer.update(format_version="causal-policy-0.3", request_placements={})
        newer["experts"]["order_rule"] = "token_count_desc"
        parsed = parse_policy(newer)
        self.assertEqual(parsed.expert_order_rule, "token_count_desc")
        self.assertEqual(policy_document(parsed), newer)
        old["experts"]["order_rule"] = "token_count_desc"
        with self.assertRaises(InvalidInput):
            parse_policy(old)
        with self.assertRaises(InvalidInput):
            policy_document(replace(parsed, format_version="causal-policy-0.2"))
        newer["experts"]["order_rule"] = "future_router_score"
        with self.assertRaises(Unsupported):
            parse_policy(newer)

    def test_token_priority_preserves_nonexpert_slots_and_stage_order(self):
        for staged in (False, True):
            seen = observation(staged=staged)
            self.assertIs(order_expert_operations(seen, ExpertSchedulingSpec()), seen.operations)
            ranked = order_expert_operations(seen, ExpertSchedulingSpec(order_rule="token_count_desc"))
            self.assertEqual(ranked[0], seen.operations[0])
            self.assertEqual(ranked[-1], seen.operations[-1])
            self.assertEqual([op.expert_id for op in ranked[1:-1]], [1] * (4 if staged else 1) + [0] * (4 if staged else 1))
            for expert in (0, 1):
                self.assertEqual([op.id for op in ranked if op.expert_id == expert],
                                 [op.id for op in seen.operations if op.expert_id == expert])
            self.assertEqual(set(op.id for op in ranked), set(op.id for op in seen.operations))

    def test_frequency_uses_deduplicated_revealed_history_and_stable_ties(self):
        seen = observation()
        spec = ExpertSchedulingSpec(order_rule="history_frequency_desc")
        self.assertEqual(order_expert_operations(seen, spec)[1].expert_id, 1)
        known_past = (RevealedRoute("past", "L0", 0, (0,), None),
                      RevealedRoute("past", "L0", 1, (0,), None))
        self.assertEqual(order_expert_operations(seen, spec, history=known_past + seen.route_history)[1].expert_id, 0)
        tie = (RevealedRoute("past", "L0", 0, (0,), None),)
        self.assertEqual(order_expert_operations(seen, spec, history=tie)[1].expert_id, 0)
        # A different layer's frequently used expert is a different identity.
        foreign = tuple(RevealedRoute("past", "L1", i, (0,), None) for i in range(5))
        self.assertEqual(order_expert_operations(seen, spec, history=foreign)[1].expert_id, 1)

    def test_unrevealed_rows_and_inconsistent_regional_geometry_are_rejected(self):
        seen = observation(staged=True)
        spec = ExpertSchedulingSpec(order_rule="token_count_desc")
        with self.assertRaisesRegex(InvalidInput, "unrevealed"):
            order_expert_operations(replace(seen, route_history=()), spec)
        bad_rows = (replace(seen.route_history[0], expert_ids=(1,)), *seen.route_history[1:])
        with self.assertRaisesRegex(InvalidInput, "unselected"):
            order_expert_operations(replace(seen, route_history=bad_rows), spec)
        bad = list(seen.operations)
        bad[2] = replace(bad[2], work_items=(("r1", 0),), expert_id=1)
        with self.assertRaisesRegex(InvalidInput, "consistent expert"):
            order_expert_operations(replace(seen, operations=tuple(bad)), spec)
        conflicting = (replace(seen.route_history[0], expert_ids=(1,)),)
        with self.assertRaisesRegex(InvalidInput, "contradicts"):
            order_expert_operations(seen, spec, history=conflicting)

    def test_device_ranking_changes_with_current_service_and_declared_feasibility(self):
        seen = observation()
        spec = ExpertSchedulingSpec(device_rule="current_cost")
        self.assertEqual(rank_expert_devices(seen, spec, paths(), preferred_device="gpu"), ("cpu", "gpu"))
        slow = replace(seen, cpu_service_rate_numerator=1, cpu_service_rate_denominator=10)
        self.assertEqual(rank_expert_devices(slow, spec, paths(), preferred_device="gpu"), ("gpu", "cpu"))
        blackout = replace(slow, gpu_service_rate_numerator=0)
        self.assertEqual(rank_expert_devices(blackout, spec, paths(), preferred_device="gpu"), ("cpu",))
        self.assertEqual(rank_expert_devices(replace(blackout, cpu_available=False), spec, paths(), preferred_device="gpu"), ())
        infeasible = paths()
        infeasible[0]["physical_feasible"] = False
        self.assertEqual(rank_expert_devices(seen, spec, infeasible, preferred_device="cpu"), ("gpu",))
        infeasible[1]["budget_feasible"] = False
        self.assertEqual(rank_expert_devices(seen, spec, infeasible, preferred_device="gpu"), ())

    def test_fixed_resident_and_missing_path_contracts_are_explicit(self):
        seen = observation()
        self.assertEqual(rank_expert_devices(seen, ExpertSchedulingSpec(), paths(), preferred_device="gpu"), ("gpu",))
        self.assertEqual(rank_expert_devices(seen, ExpertSchedulingSpec(device_rule="resident_first"), paths(),
                                            preferred_device="cpu", cached_gpu=True), ("gpu", "cpu"))
        for field in ("compute_ns", "weight_transfer_ns", "physical_feasible"):
            missing = paths()
            del missing[0][field]
            with self.subTest(field=field), self.assertRaises(InvalidInput):
                rank_expert_devices(seen, ExpertSchedulingSpec(device_rule="current_cost"), missing, preferred_device="cpu")
        with self.assertRaisesRegex(Unsupported, "existing explicit dispatcher"):
            rank_expert_devices(seen, ExpertSchedulingSpec(device_rule="hybrimoe_queues"), paths(), preferred_device="cpu")

    def test_real_moe_observations_hide_future_rows_and_keep_equal_prefix(self):
        source = compiled()
        sessions = (CausalM2Session(source), CausalM2Session(alternative(source)))
        declarations = policy_document_fixture()
        policies = (DeclarativePolicy(parse_policy(declarations)), DeclarativePolicy(parse_policy(declarations)))
        spec = ExpertSchedulingSpec(order_rule="token_count_desc")
        # Advance exactly the pre-router public prefix. The helper has no access
        # to the private graphs whose future expert choices differ.
        for _ in range(6):
            observations = tuple(session.observe() for session in sessions)
            self.assertEqual(observations[0], observations[1])
            ordered = tuple(order_expert_operations(seen, spec) for seen in observations)
            self.assertEqual(ordered[0], ordered[1])
            self.assertFalse(any(op.kind == "expert_ffn" for op in ordered[0]))
            decisions = tuple(policy(seen) for policy, seen in zip(policies, observations))
            self.assertEqual(decisions[0], decisions[1])
            for session, seen, decision in zip(sessions, observations, decisions):
                action = compile_decision(decision, seen)
                result = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
                self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
