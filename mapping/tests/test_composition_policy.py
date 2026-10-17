"""Mechanism composition is explicit, causal, and region/group scoped."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import json
import unittest

from llm_infer_model.m2.causal import DevicePrice, PublicExpert, PublicOperation, CausalObservation
from llm_infer_model.tensor.environment import RunningObservation
from llm_infer_model.tensor.services import PublicServices, PublicFixedSlot
from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, InvalidInput, Unsupported
from tensor_mapping.policy_ir import parse_history
from tensor_mapping.composition_policy import (
    parse_composition_policy, composition_policy_document, CompositionPolicy,
)
from .test_policy_executor import tensor, small_observation
from .test_policy_ir import policy_document_fixture, history_fixture


@dataclass(frozen=True)
class CompositionObservation(CausalObservation):
    layer_order: tuple = ()
    active_invocations: tuple = ()
    weight_catalogue: tuple = ()
    recoverable_release_candidates: tuple = ()


def inline_policy(*, rule="fixed", cache="none", unit="tensor", budget=None,
                  prefetch=False, ranking="history_frequency", services=False):
    legacy = policy_document_fixture()
    placements = legacy["placements"]
    placements.update(gather_rows="gpu", scatter_rows="gpu")
    engines = [dict(role=role, stage=None, src=src, dst=dst, engine="ce" if services else None)
        for role in ("weight", "input", "intermediate", "output")
        for src, dst in (("dram", "vram"), ("vram", "dram"))]
    return {"placements": placements, "request_placements": {},
        "experts": {"device_rule": rule, "selection_unit": "expert_region"},
        "release": {"rule": "none"},
        "cache": {"rule": cache, "unit": unit, "budget_bytes": budget,
                  "alpha": 0.5 if cache == "mrs" else None, "top_p": 1 if cache == "mrs" else None},
        "prefetch": {"enabled": prefetch, "scope": "next_layer" if prefetch else "none",
                     "lookahead_layers": 1 if prefetch else 0, "max_count": 4 if prefetch else 0,
                     "ranking": ranking, "include_dense_weights": False},
        "transport": {"rules": engines, "slot_reuse": "release_unlocked_backed_copy",
                      "boundary_release": "disabled"}}


def observe(base, **fields):
    return CompositionObservation(**base.__dict__, **fields)


def region_observation(*, cached=False, future=False, free=4096):
    base = small_observation(free_vram=free)
    tensors = [tensor("compact", 16, ("vram",), transfer_ns=10),
        tensor("wg", 32, ("dram", "ssd", "vram") if cached else ("dram", "ssd"), "weight", ("ssd",), "ssd"),
        tensor("wu", 32, ("dram", "ssd", "vram") if cached else ("dram", "ssd"), "weight", ("ssd",), "ssd"),
        tensor("wd", 32, ("dram", "ssd", "vram") if cached else ("dram", "ssd"), "weight", ("ssd",), "ssd"),
        tensor("g", 32), tensor("u", 32), tensor("a", 32), tensor("z", 16, transfer_ns=10), tensor("contribution", 16)]
    prices = (DevicePrice("cpu", 50, 0, 0), DevicePrice("gpu", 10, 0, 0))
    rows = (("r", 0),)
    def operation(name, kind, inputs, outputs, status="NOT_STARTED"):
        return PublicOperation(name, kind, status, inputs, outputs, prices, expert_id=0, work_items=rows,
            token_count=1, phase="prefill", request_id="r", region_id="expert:0", execution_stage=kind,
            execution_variant="swiglu_staged", layer_id="L0", invocation_id="r0L0")
    operations = (operation("gather", "gather_rows", (), ("compact",), "DONE"),
        operation("gate", "expert_gate", ("wg", "compact"), ("g",)),
        operation("up", "expert_up", ("wu", "compact"), ("u",)),
        operation("swiglu", "expert_swiglu", ("g", "u"), ("a",)),
        operation("down", "expert_down", ("wd", "a"), ("z",)),
        operation("scatter", "scatter_rows", ("z",), ("contribution",)))
    experts = (PublicExpert(0, None, "L0", ("wg", "wu", "wd")),)
    catalogue = (("wg", "L0", "gate", 0), ("wu", "L0", "up", 0), ("wd", "L0", "down", 0))
    if future:
        tensors += [tensor("L1wg", 32, ("dram", "ssd"), "weight", ("ssd",), "ssd")]
        experts += (PublicExpert(0, "L1wg", "L1", ("L1wg",)),)
        catalogue += (("L1wg", "L1", "gate", 0),)
    return observe(replace(base, operations=operations, tensors=tuple(tensors), experts=experts),
                   layer_order=("L0", "L1"), active_invocations=(("r0L0", "L0"),), weight_catalogue=catalogue)


def with_services(observation, *, slot=None):
    tensors = tuple(replace(item, transfers=tuple(replace(price, duration_ns=None,
        engine_durations_ns=(("ce", item.storage_bytes), ("sm", item.storage_bytes * 2)))
        if {price.src, price.dst} == {"dram", "vram"} else price for price in item.transfers)) for item in observation.tensors)
    services = PublicServices(10**9, 10**9, 10**9, "max_min_fair", "payload_only_no_launch_latency",
                              (slot,) if slot else (), ())
    return replace(observation, tensors=tensors, services=services)


class CompositionSchemaTests(unittest.TestCase):
    def test_round_trip_keeps_all_inline_mechanisms(self):
        for unit in ("tensor", "expert_group"):
            document = inline_policy(rule="fiddler_paths", cache="mrs", unit=unit, budget=128,
                                     prefetch=True, ranking="mrs", services=True)
            self.assertEqual(composition_policy_document(parse_composition_policy(document)), document)

    def test_unknown_fields_bad_group_or_implicit_parameters_are_rejected(self):
        documents = []
        for section, field, value in (("experts", "selection_unit", "stage"),
                                      ("cache", "unit", "whole_parent_guess"),
                                      ("cache", "alpha", 0.5), ("cache", "budget_bytes", True),
                                      ("prefetch", "lookahead_layers", 1)):
            doc = inline_policy(); doc[section][field] = value; documents.append(doc)
        doc = inline_policy(); doc["future_routes"] = []; documents.append(doc)
        doc = inline_policy(); doc["transport"]["rules"].append(deepcopy(doc["transport"]["rules"][0])); documents.append(doc)
        for doc in documents:
            with self.subTest(doc=doc), self.assertRaises((InvalidInput, Unsupported)):
                parse_composition_policy(doc)

    def test_execution_binding_roundtrips_and_requires_unique_pure_attention_device(self):
        document = inline_policy()
        document["placements"]["attention_core"] = "execution_binding"
        spec = parse_composition_policy(document)
        self.assertEqual(composition_policy_document(spec), document)
        policy = CompositionPolicy(spec)
        audit = policy.audit_document()
        self.assertNotIn("policy", audit)
        self.assertEqual(audit["format_version"], "composition-policy-audit-0.1")
        self.assertEqual(audit["authoritative_strategy_field"], "composition")
        self.assertEqual(audit["composition"], document)
        self.assertEqual(audit["composition"]["placements"]["attention_core"], "execution_binding")
        operation = PublicOperation("core", "attention_core", "NOT_STARTED", (), (),
                                   (DevicePrice("gpu", 10, 0, 0),), execution_variant="pure_attention")
        self.assertEqual(policy._placement(operation), "gpu")
        self.assertEqual(policy._placement(replace(operation, prices=(DevicePrice("cpu", 11, 0, 0),))), "cpu")
        with self.assertRaisesRegex(Unsupported, "one declared"):
            policy._placement(replace(operation, prices=operation.prices + (DevicePrice("cpu", 11, 0, 0),)))
        with self.assertRaisesRegex(Unsupported, "pure_attention"):
            policy._placement(replace(operation, execution_variant=""))


class CompleteExpertRegionTests(unittest.TestCase):
    def test_fiddler_compares_whole_staged_ffn_not_the_first_matmul(self):
        observed = region_observation()
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="fiddler_paths")))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.src, decision.dst),
                         ("COPY", "compact", "vram", "dram"))
        binding = policy.audit_document()["region_bindings"][0]
        self.assertEqual(binding["device"], "cpu")
        paths = {row["device"]: row for row in binding["paths"]}
        self.assertEqual(paths["cpu"]["compute_ns"], 200)
        self.assertEqual(paths["gpu"]["compute_ns"], 40)
        self.assertEqual(paths["gpu"]["weight_transfer_ns"], 300)
        self.assertEqual(paths["cpu"]["total_ns"], 220)
        self.assertEqual(set(paths["gpu"]["unique_weights"]), {"wg", "wu", "wd"})
        self.assertFalse(policy.audit_document()["claims"]["paper_strategy_reproduced"])

    def test_committed_device_is_identical_for_gate_up_swiglu_down(self):
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="fiddler_paths")))
        observed = region_observation(cached=True)
        policy(observed)
        estimated = policy._estimate_observation(observed)
        tensors = {item.id: item for item in estimated.tensors}
        devices = [policy._devices(op, estimated, tensors) for op in estimated.operations
                   if op.kind.startswith("expert_")]
        self.assertEqual(devices, [("gpu",)] * 4)

    def test_fixed_rule_retains_heterogeneous_per_stage_bindings(self):
        document = inline_policy()
        document["placements"].update(expert_gate="gpu", expert_up="gpu", expert_swiglu="cpu", expert_down="cpu")
        policy = CompositionPolicy(parse_composition_policy(document))
        observed = region_observation(cached=True)
        policy._public_catalogue(observed)
        tensors = {item.id: item for item in observed.tensors}
        self.assertEqual([policy._devices(op, observed, tensors) for op in observed.operations
                          if op.kind.startswith("expert_")], [("gpu",), ("gpu",), ("cpu",), ("cpu",)])

    def test_incomplete_stages_are_refused_instead_of_priced_as_whole_expert(self):
        observed = region_observation()
        observed = replace(observed, operations=tuple(op for op in observed.operations if op.kind != "expert_up"))
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="fiddler_paths")))
        with self.assertRaisesRegex(Unsupported, "all four"):
            policy(observed)

    def test_hybrimoe_queue_commits_entire_cpu_region(self):
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="hybrimoe_queues")))
        observed = region_observation()
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("COPY", "compact"))
        self.assertEqual(policy.audit_document()["region_bindings"][0]["device"], "cpu")
        observed = replace(observed, tensors=tuple(replace(t, copies=tuple((loc, COPY_READY if loc in ("dram", "vram") else status)
            for loc, status in t.copies)) if t.id == "compact" else t for t in observed.tensors))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.device), ("COMPUTE", "gate", "cpu"))
        self.assertEqual(len(policy.audit_document()["region_bindings"]), 1)

    def test_hybrimoe_cached_cpu_fallback_when_gpu_busy_and_no_uncached_queue(self):
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="hybrimoe_queues")))
        observed = replace(region_observation(cached=True),
                           running=(RunningObservation("gpu_compute", "other", "gpu", None),))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("COPY", "compact"))
        self.assertEqual(policy.audit_document()["region_bindings"][0]["device"], "cpu")
        self.assertEqual(policy.audit_document()["decisions"][-1]["evidence"]["queue_priority"]["rule"],
                         "cpu_cached_asc_when_uncached_queue_empty")


class CompositionCacheAndPrefetchTests(unittest.TestCase):
    def test_partial_group_residency_reserves_all_fragment_bytes(self):
        observed = region_observation()
        observed = replace(observed, tensors=tuple(replace(item, copies=tuple((location, COPY_READY if location == "vram" else status)
            for location, status in item.copies)) if item.id == "wg" else item for item in observed.tensors))
        policy = CompositionPolicy(parse_composition_policy(inline_policy(unit="expert_group", budget=96)))
        policy._public_catalogue(observed)
        self.assertEqual(policy._budget_for_additions({t.id: t for t in observed.tensors}), 96)
        policy = CompositionPolicy(parse_composition_policy(inline_policy(unit="tensor", budget=32)))
        policy._public_catalogue(observed)
        self.assertEqual(policy._budget_for_additions({t.id: t for t in observed.tensors}), 32)

    def test_all_gpu_demand_respects_weight_budget_even_without_prefetch(self):
        doc = inline_policy(unit="expert_group", budget=95)
        doc["placements"].update(expert_gate="gpu", expert_up="gpu", expert_down="gpu")
        policy = CompositionPolicy(parse_composition_policy(doc))
        with self.assertRaisesRegex(Unsupported, "exceeds its explicit weight budget"):
            policy(region_observation())

    def test_full_region_gpu_budget_failure_uses_declared_cpu_path(self):
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="fiddler_paths", unit="expert_group", budget=95)))
        policy(region_observation())
        binding = policy.audit_document()["region_bindings"][0]
        self.assertEqual(binding["device"], "cpu")
        self.assertFalse(next(p for p in binding["paths"] if p["device"] == "gpu")["budget_feasible"])

    def test_group_eviction_keeps_priority_until_all_members_are_absent(self):
        observed = region_observation(cached=True)
        observed = replace(observed, operations=())
        policy = CompositionPolicy(parse_composition_policy(inline_policy(cache="lru", unit="expert_group", budget=0)))
        decisions = []
        budgets = []
        for _ in range(3):
            decision = policy(observed)
            decisions.append(decision.target_id)
            observed = replace(observed, tensors=tuple(replace(t, copies=tuple((loc, COPY_ABSENT if loc == "vram" else status)
                for loc, status in t.copies)) if t.id == decision.target_id else t for t in observed.tensors))
            budgets.append(policy._budget_for_additions({t.id: t for t in observed.tensors}))
        self.assertEqual(set(decisions), {"wg", "wu", "wd"})
        self.assertEqual(budgets, [96, 96, 0])

    def test_history_prefetch_can_run_next_layer_while_current_compute_is_busy(self):
        observed = region_observation(future=True)
        observed = replace(observed, operations=(), running=(RunningObservation("gpu_compute", "current", "gpu", None),))
        doc = inline_policy(cache="lru", unit="expert_group", budget=128, prefetch=True)
        history = parse_history(history_fixture((0,), layer="L1"))
        policy = CompositionPolicy(parse_composition_policy(doc), history=history)
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("COPY", "L1wg"))
        evidence = policy.audit_document()["decisions"][-1]["evidence"]
        self.assertEqual(evidence["information_scope"], "static_next_layer_catalogue_and_revealed_history_only")
        self.assertEqual(evidence["public_priority"], 1)

    def test_no_history_does_not_use_current_future_catalogue_as_actual_router(self):
        observed = replace(region_observation(future=True), operations=())
        policy = CompositionPolicy(parse_composition_policy(inline_policy(cache="lru", budget=128, prefetch=True)))
        self.assertEqual(policy(observed).kind, "ADVANCE")
        self.assertEqual(policy.audit_document()["predictions"], [])

    def test_group_prefetch_budget_refuses_one_fragment_if_full_group_will_not_fit(self):
        observed = region_observation(future=True)
        # A future selected-row group exposes three known immutable fragments.
        extra = (tensor("L1wu", 32, ("dram", "ssd"), "weight", ("ssd",), "ssd"),
                 tensor("L1wd", 32, ("dram", "ssd"), "weight", ("ssd",), "ssd"))
        observed = replace(observed, operations=(), tensors=observed.tensors + extra,
            experts=observed.experts[:-1] + (PublicExpert(0, None, "L1", ("L1wg", "L1wu", "L1wd")),),
            weight_catalogue=observed.weight_catalogue + (("L1wu", "L1", "up", 0), ("L1wd", "L1", "down", 0)))
        policy = CompositionPolicy(parse_composition_policy(inline_policy(cache="lru", unit="expert_group", budget=95, prefetch=True)),
            history=parse_history(history_fixture((0,), layer="L1")))
        self.assertEqual(policy(observed).kind, "ADVANCE")

    def test_mrs_next_layer_prefetch_uses_only_public_full_router_scores(self):
        observed = replace(region_observation(future=True), operations=())
        observed = replace(observed, tensors=observed.tensors + (
            tensor("L1e1", 32, ("dram", "ssd"), "weight", ("ssd",), "ssd"),),
            experts=observed.experts + (PublicExpert(1, "L1e1", "L1", ("L1e1",)),),
            weight_catalogue=observed.weight_catalogue + (("L1e1", "L1", "gate", 1),))
        history = parse_history({"format_version": "causal-history-0.2", "entries": [{
            "request_id": "past", "layer_id": "L1", "absolute_position": 0,
            "expert_ids": [1], "scores": [0.9], "router_scores_scope": "all_experts",
            "router_scores": [{"expert_id": 0, "score": 0.1}, {"expert_id": 1, "score": 0.9}]}]})
        spec = parse_composition_policy(inline_policy(cache="mrs", unit="expert_group", budget=128,
                                                     prefetch=True, ranking="mrs"))
        policy = CompositionPolicy(spec, history=history)
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("COPY", "L1e1"))
        self.assertEqual(policy.audit_document()["mrs_priorities"], [
            {"layer_id": "L1", "expert_id": 0, "priority": 0.0},
            {"layer_id": "L1", "expert_id": 1, "priority": 0.45}])
        insufficient = CompositionPolicy(spec, history=parse_history(history_fixture((1,), layer="L1")))
        with self.assertRaisesRegex(Unsupported, "all-expert"):
            insufficient(observed)

    def test_same_public_prefix_with_different_future_models_has_same_prefetch(self):
        from llm_infer_model.m2.causal import CausalM2Session
        from tensor_mapping.causal import compile_command
        from llm_infer_model.tensor.runtime import dispatch_command
        from .test_causal_mapping import compiled, alternative
        source = compiled()
        sessions = (CausalM2Session(source), CausalM2Session(alternative(source)))
        spec = parse_composition_policy(inline_policy())
        policies = (CompositionPolicy(spec), CompositionPolicy(spec))
        for _ in range(6):
            observations = tuple(session.observe() for session in sessions)
            self.assertEqual(observations[0], observations[1])
            decisions = tuple(policy(observation) for policy, observation in zip(policies, observations))
            self.assertEqual(decisions[0], decisions[1])
            self.assertEqual(policies[0].audit_document(), policies[1].audit_document())
            for session, observation, decision in zip(sessions, observations, decisions):
                self.assertTrue(dispatch_command(session, compile_command(decision, observation)).ok)

    def test_prefetch_count_and_hop_limit_reset_for_the_next_source_invocation(self):
        observed = replace(region_observation(future=True), operations=())
        doc = inline_policy(cache="lru", budget=128, prefetch=True)
        doc["prefetch"]["max_count"] = 1
        policy = CompositionPolicy(parse_composition_policy(doc), history=parse_history(history_fixture((0,), layer="L1")))
        self.assertEqual(policy(observed).target_id, "L1wg")
        # A second traversal of the same public layer order can prefetch that
        # immutable weight again after it has been evicted. Limits are per
        # source invocation, rather than an accidental lifetime quota.
        later = replace(observed, active_invocations=(("r1L0", "L0"),))
        self.assertEqual(policy(later).target_id, "L1wg")
        evidence = policy.audit_document()["decisions"][-1]["evidence"]
        self.assertEqual(evidence["source_invocation_id"], "r1L0")
        self.assertEqual(policy.audit_document()["prefetch_max_count_scope"], "per_active_source_invocation")


class CompositionTransportTests(unittest.TestCase):
    def test_stage_and_direction_specific_rules_override_role_fallback(self):
        document = inline_policy(services=True)
        document["placements"].update(expert_gate="gpu", expert_up="gpu", expert_down="gpu")
        document["transport"]["rules"].append(dict(role="weight", stage="gate", src="dram", dst="vram", engine="sm"))
        policy = CompositionPolicy(parse_composition_policy(document))
        observed = with_services(region_observation())
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.transfer_engine), ("COPY", "wg", "sm"))
        estimated = policy._estimate_observation(observed)
        prices = {t.id: {p.route: p.duration_ns for p in t.transfers} for t in estimated.tensors}
        self.assertEqual(prices["wg"]["dram_to_vram"], 64)
        self.assertEqual(prices["wg"]["vram_to_dram"], 32)
        self.assertEqual(prices["wu"]["dram_to_vram"], 32)

    def test_slot_guard_does_not_consume_prefetch_intent(self):
        observed = replace(region_observation(future=True), operations=())
        observed = replace(observed, tensors=observed.tensors + (tensor("old", 32, ("dram", "vram"), "weight", ("dram",)),),
            weight_catalogue=observed.weight_catalogue + (("old", "L0", "gate", 1),),
            experts=observed.experts + (PublicExpert(1, "old", "L0", ("old",)),))
        slot = PublicFixedSlot("shared", "vram", 32, ("old", "L1wg"), "old")
        observed = with_services(observed, slot=slot)
        doc = inline_policy(services=True, cache="lru", budget=128, prefetch=True)
        policy = CompositionPolicy(parse_composition_policy(doc), history=parse_history(history_fixture((0,), layer="L1")))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("EVICT", "old"))
        self.assertEqual(policy.audit_document()["predictions"], [])
        self.assertEqual(policy._prefetch_units, set())
        observed = replace(observed, services=replace(observed.services, slots=(replace(slot, owner=None),)),
            tensors=tuple(replace(t, copies=tuple((loc, COPY_ABSENT if loc == "vram" else status) for loc, status in t.copies))
                if t.id == "old" else t for t in observed.tensors))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.transfer_engine), ("COPY", "L1wg", "ce"))

    def test_recoverable_kv_copy_releases_without_deleting_home_goal(self):
        observed = replace(region_observation(), operations=())
        kv = tensor("kv", 16, ("ssd", "dram", "vram"), "output", goal="ssd")
        observed = replace(observed, tensors=observed.tensors + (kv,), recoverable_release_candidates=(("kv", "vram"),))
        doc = inline_policy(); doc["transport"]["boundary_release"] = "declared_dead_intermediates"
        policy = CompositionPolicy(parse_composition_policy(doc))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.location), ("EVICT", "kv", "vram"))
        record = policy.audit_document()["cache_decisions"][-1]
        self.assertEqual(record["reason"], "release_recoverable_copy_keep_home")
        self.assertEqual(record["goal_location_preserved"], "ssd")

    def test_pending_consumer_blocks_recoverable_kv_release(self):
        observed = replace(region_observation(), operations=())
        kv = tensor("kv", 16, ("ssd", "dram", "vram"), "output", goal="ssd")
        reader = PublicOperation("reader", "attention", "NOT_STARTED", ("kv",), (), (DevicePrice("gpu", 5, 0, 0),))
        observed = replace(observed, tensors=observed.tensors + (kv,), operations=(reader,),
                           recoverable_release_candidates=(("kv", "vram"),))
        doc = inline_policy(); doc["transport"]["boundary_release"] = "declared_dead_intermediates"
        doc["placements"]["attention"] = "gpu"
        policy = CompositionPolicy(parse_composition_policy(doc))
        self.assertEqual(policy(observed).kind, "COMPUTE")
        self.assertEqual(policy.audit_document()["cache_decisions"], [])

    def test_slot_guard_rolls_back_complete_region_binding_and_append_only_audit(self):
        observed = region_observation()
        old = tensor("old", 32, ("dram", "vram"), "weight", ("dram",))
        observed = replace(observed, tensors=observed.tensors + (old,), cpu_available=False,
            experts=observed.experts + (PublicExpert(1, "old", "L0", ("old",)),),
            weight_catalogue=observed.weight_catalogue + (("old", "L0", "gate", 1),))
        observed = with_services(observed, slot=PublicFixedSlot("shared", "vram", 32, ("old", "wg"), "old"))
        policy = CompositionPolicy(parse_composition_policy(inline_policy(rule="fiddler_paths", services=True)))
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id), ("EVICT", "old"))
        self.assertEqual(policy.audit_document()["region_bindings"], [])
        self.assertEqual(policy.audit_document()["decisions"], [])
        self.assertEqual(policy.audit_document()["mrs_updates"], [])
        self.assertEqual(policy._clock, 0)
        self.assertEqual(policy.audit_document()["transport_decisions"][0]["reason"], "explicit_slot_owner_release")


class CompositionStagedIntegrationTests(unittest.TestCase):
    def test_dynamic_selectors_execute_each_materialized_stage_once_with_one_region_device(self):
        from llm_infer_model.m2.causal import CausalM2Session
        from llm_infer_model.m2.expand import load_and_compile
        from llm_infer_model.m2.row_execution import parse_execution_spec
        from tensor_mapping.causal import run_causal
        fixture = Path(__file__).resolve().parents[2] / "modeling/tests/fixtures/subweights"
        execution = parse_execution_spec(json.loads((fixture / "execution.json").read_text(encoding="utf-8")))
        for rule in ("fiddler_paths", "hybrimoe_queues"):
            session = CausalM2Session(load_and_compile(fixture / "invocation.json"), execution_spec=execution)
            document = inline_policy(rule=rule, cache="lru", unit="expert_group", budget=1536)
            document["release"]["rule"] = "last_use"
            policy = CompositionPolicy(parse_composition_policy(document))
            run = run_causal(policy, session, max_decisions=500)
            self.assertEqual(run.evaluation.status, "valid", (rule, run.compilation_error, run.evaluation.reason))
            compute = [record for record in run.decisions if record.decision.kind == "COMPUTE"]
            self.assertEqual(len(compute), len({record.decision.target_id for record in compute}))
            regions = {}
            for record in compute:
                operation = next(op for op in record.observation.operations if op.id == record.decision.target_id)
                if operation.kind.startswith("expert_"):
                    regions.setdefault(operation.region_id, []).append((operation.kind, record.decision.device))
            self.assertEqual(len(regions), 3)
            for stages in regions.values():
                self.assertEqual({kind for kind, _ in stages}, {"expert_gate", "expert_up", "expert_swiglu", "expert_down"})
                self.assertEqual(len({device for _, device in stages}), 1)
            self.assertTrue(run.evaluation.ssd_read_bytes)


if __name__ == "__main__":
    unittest.main()
