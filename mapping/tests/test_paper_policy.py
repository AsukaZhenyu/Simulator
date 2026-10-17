"""Mechanism-level paper contracts; synthetic prices, no performance claim."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from llm_infer_model.m2.causal import (
    CausalObservation, CausalM2Session, DevicePrice, PublicExpert, PublicOperation, RevealedRoute,
)
from llm_infer_model.m2.row_execution import parse_execution_spec
from llm_infer_model.tensor.environment import MemoryUsage, RunningObservation
from llm_infer_model.tensor.spec import COPY_READY, OP_NOT_STARTED, InvalidInput, Unsupported
from tensor_mapping.causal import compile_decision, run_causal
from tensor_mapping.policy_executor import DeclarativePolicy
from tensor_mapping.policy_ir import parse_policy, policy_document, parse_history, history_document
from .test_policy_ir import policy_document_fixture
from .test_policy_executor import tensor, small_observation, compiled, staged_execution
from .test_causal_mapping import alternative


def paper_document(rule="fixed", cache="none"):
    value = policy_document_fixture()
    value.update(format_version="causal-policy-0.2", request_placements={})
    value["experts"]["device_rule"] = rule
    value["cache"] = {"rule": cache}
    if cache == "mrs":
        value["cache"].update(alpha=0.5, top_p=3)
    return value


def prices(cpu=10, gpu=15):
    return (DevicePrice("cpu", cpu, 0, 0), DevicePrice("gpu", gpu, 0, 0))


def fiddler_observation():
    x = tensor("group", 16, ("vram",), transfer_ns=5)
    weight = tensor("weight", 64, ("dram",), "weight", ("dram",), "dram", transfer_ns=1)
    output = tensor("expert-result", 16, transfer_ns=100)
    final = tensor("contribution", 16)
    gather = PublicOperation("gather", "gather_rows", "DONE", (), (x.id,), prices(),
                             request_id="r0", execution_variant="opaque")
    expert = PublicOperation("expert", "expert_ffn", OP_NOT_STARTED, (weight.id, x.id), (output.id,),
                             prices(), expert_id=0, token_count=1, work_items=(("r0", 0),), region_id="expert:0",
                             execution_variant="opaque")
    scatter = PublicOperation("scatter", "scatter_rows", OP_NOT_STARTED, (output.id,), (final.id,),
                              prices(), request_id="r0", execution_variant="opaque")
    memory = (MemoryUsage("dram", 4096, 0, 0, 64, 0, 64),
              MemoryUsage("vram", 4096, 0, 0, 16, 0, 16))
    return CausalObservation(0, (gather, expert, scatter), (x, weight, output, final), (), memory,
                             True, (), (), False, None, experts=(PublicExpert(0, weight.id, "layer"),),
                             allow_copy_compute_overlap=True, allow_cpu_gpu_overlap=True,
                             allow_eviction=True, task_graph_complete=True)


def fiddler_policy(rule="fiddler_paths"):
    document = paper_document(rule)
    document["placements"]["gather_rows"] = "gpu"
    document["placements"]["scatter_rows"] = "gpu"
    return DeclarativePolicy(parse_policy(document))


def queue_observation(specifications=((0, 3, True), (1, 8, True), (2, 1, False), (3, 5, False))):
    operations, tensors, experts = [], [], []
    for expert, tokens, cached in specifications:
        weight = tensor(f"w{expert}", 32, ("dram", "vram") if cached else ("dram",),
                        "weight", ("dram",), "dram")
        source = tensor(f"x{expert}", tokens * 16, ("dram", "vram"))
        output = tensor(f"y{expert}", tokens * 16)
        operations.append(PublicOperation(f"e{expert}", "expert_ffn", OP_NOT_STARTED,
                           (weight.id, source.id), (output.id,), prices(), expert_id=expert,
                           token_count=tokens, work_items=tuple(("r0", position) for position in range(tokens)),
                           region_id=f"expert:{expert}", execution_variant="opaque"))
        tensors.extend((weight, source, output))
        experts.append(PublicExpert(expert, weight.id, "layer"))
    memory = tuple(MemoryUsage(domain, 4096, 0, 0,
                               sum(dict(item.allocation_bytes)[domain] for item in tensors
                                   if item.status_at(domain) == COPY_READY), 0,
                               sum(dict(item.allocation_bytes)[domain] for item in tensors
                                   if item.status_at(domain) == COPY_READY)) for domain in ("dram", "vram"))
    return CausalObservation(0, tuple(operations), tuple(tensors), (), memory, True, (), (), False, None,
                             experts=tuple(experts), allow_copy_compute_overlap=True,
                             allow_cpu_gpu_overlap=True, allow_eviction=True, task_graph_complete=True)


def score_history(rows=(("old", 1, (0, 1), (0.5, 0.3, 0.19, 0.01)),)):
    return {"format_version": "causal-history-0.2", "entries": [
        {"request_id": request, "layer_id": "layer", "absolute_position": position,
         "expert_ids": list(active), "scores": [values[expert] for expert in active],
         "router_scores_scope": "all_experts",
         "router_scores": [{"expert_id": expert, "score": value} for expert, value in enumerate(values)]}
        for request, position, active, values in rows]}


class PaperPolicyTests(unittest.TestCase):
    def test_version_two_round_trip_and_version_one_rejects_new_fields(self):
        document = paper_document("hybrimoe_queues", "mrs")
        document["request_placements"] = {"r0": {"attention_core": "cpu"}}
        self.assertEqual(policy_document(parse_policy(document)), document)
        legacy = deepcopy(document)
        legacy["format_version"] = "causal-policy-0.1"
        with self.assertRaises(InvalidInput):
            parse_policy(legacy)
        for alpha in (True, 0, -1, 1.1, float("nan")):
            bad = deepcopy(document)
            bad["cache"]["alpha"] = alpha
            with self.subTest(alpha=alpha), self.assertRaises(InvalidInput):
                parse_policy(bad)

    def test_request_binding_overrides_default_without_compiler_choice(self):
        document = paper_document()
        document["placements"]["attention_core"] = "gpu"
        document["request_placements"] = {"r0": {"attention_core": "cpu"}}
        observation = small_observation(free_vram=512)
        operation = PublicOperation("attention", "attention_core", OP_NOT_STARTED, ("x",), ("y",),
                                    prices(cpu=5, gpu=4), request_id="r0")
        observation = replace(observation, operations=(operation,), experts=())
        decision = DeclarativePolicy(parse_policy(document))(observation)
        self.assertEqual((decision.kind, decision.device), ("COMPUTE", "cpu"))
        self.assertEqual(compile_decision(decision, observation).device, "cpu")

    def test_fiddler_complete_return_path_changes_choice_from_current_cost(self):
        observation = fiddler_observation()
        simple = fiddler_policy("current_cost")(observation)
        self.assertEqual((simple.kind, simple.src, simple.dst), ("COPY", "vram", "dram"))
        policy = fiddler_policy()
        decision = policy(observation)
        self.assertEqual((decision.kind, decision.target_id, decision.dst), ("COPY", "weight", "vram"))
        paths = policy.audit_document()["decisions"][-1]["evidence"]["fiddler_complete_paths"]
        self.assertEqual({path["device"]: path["total_ns"] for path in paths}, {"cpu": 115, "gpu": 16})
        self.assertEqual(next(path for path in paths if path["device"] == "cpu")["output_return_ns"], 100)

    def test_fiddler_cached_gpu_priority_and_locked_choice(self):
        observation = fiddler_observation()
        weight = replace(observation.tensors[1], copies=(("dram", COPY_READY), ("vram", COPY_READY), ("ssd", "ABSENT")))
        observation = replace(observation, tensors=(observation.tensors[0], weight) + observation.tensors[2:])
        policy = fiddler_policy()
        decision = policy(observation)
        self.assertEqual((decision.kind, decision.device), ("COMPUTE", "gpu"))
        self.assertEqual(compile_decision(decision, observation).device, "gpu")
        uncached = fiddler_observation()
        policy = fiddler_policy()
        policy(uncached)
        # Preparing inputs does not silently recompute a different device plan.
        dram_group = replace(uncached.tensors[0], copies=(("dram", COPY_READY), ("vram", COPY_READY), ("ssd", "ABSENT")))
        policy(replace(uncached, tensors=(dram_group,) + uncached.tensors[1:]))
        self.assertEqual(policy.audit_document()["decisions"][-1]["evidence"]["fiddler_chosen_device"], "gpu")

    def test_fiddler_refuses_unpriced_output_or_wrong_packaging(self):
        observation = fiddler_observation()
        no_return = replace(observation.tensors[2], transfers=())
        with self.assertRaisesRegex(Unsupported, "output return"):
            fiddler_policy()(replace(observation, tensors=observation.tensors[:2] + (no_return,) + observation.tensors[3:]))
        with self.assertRaisesRegex(Unsupported, "selected_rows"):
            fiddler_policy()(replace(observation, operations=(observation.operations[0],
                replace(observation.operations[1], execution_variant="whole_request"), observation.operations[2])))

    def test_hybrimoe_three_queues_use_revealed_load_order(self):
        document = paper_document("hybrimoe_queues")
        observation = queue_observation()
        gpu = DeclarativePolicy(parse_policy(document))(observation)
        self.assertEqual((gpu.kind, gpu.target_id, gpu.device), ("COMPUTE", "e1", "gpu"))
        busy_gpu = replace(observation, running=(RunningObservation("gpu_compute", "other", "gpu", None),))
        cpu = DeclarativePolicy(parse_policy(document))(busy_gpu)
        self.assertEqual((cpu.kind, cpu.target_id, cpu.device), ("COMPUTE", "e2", "cpu"))
        busy = replace(busy_gpu, running=busy_gpu.running + (RunningObservation("cpu_compute", "other2", "cpu", None),))
        policy = DeclarativePolicy(parse_policy(document))
        transfer = policy(busy)
        self.assertEqual((transfer.kind, transfer.target_id, transfer.dst), ("COPY", "w3", "vram"))
        evidence = policy.audit_document()["decisions"][-1]["evidence"]
        self.assertEqual(evidence["queue_priority"]["rule"], "transfer_uncached_desc")
        self.assertEqual(evidence["revealed_token_count"], 5)

    def test_hybrimoe_cpu_can_take_low_load_cached_when_uncached_queue_empty(self):
        observation = queue_observation(((0, 3, True), (1, 8, True)))
        observation = replace(observation, running=(RunningObservation("gpu_compute", "other", "gpu", None),))
        policy = DeclarativePolicy(parse_policy(paper_document("hybrimoe_queues")))
        decision = policy(observation)
        self.assertEqual((decision.kind, decision.target_id, decision.device), ("COMPUTE", "e0", "cpu"))
        self.assertIn("cached_asc", policy.audit_document()["decisions"][-1]["evidence"]["queue_priority"]["rule"])

    def test_hybrimoe_transfer_claim_prevents_duplicate_cpu_start(self):
        observation = queue_observation(((3, 5, False),))
        policy = DeclarativePolicy(parse_policy(paper_document("hybrimoe_queues")))
        transfer = policy(replace(observation, cpu_available=False))
        self.assertEqual(transfer.kind, "COPY")
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", "RESERVED_COPY"), ("ssd", "ABSENT")))
                        if item.id == "w3" else item for item in observation.tensors)
        during = replace(observation, tensors=tensors,
                         running=(RunningObservation("pcie_copy", "w3", None, "dram_to_vram"),))
        self.assertEqual(policy(during).kind, "ADVANCE")

    def test_mrs_scores_include_unactivated_experts_and_deduplicate_revelations(self):
        observation = queue_observation()
        history = parse_history(score_history((
            ("z-earlier", 1, (0, 1), (0.5, 0.3, 0.19, 0.01)),
            ("a-later", 2, (3, 2), (0.1, 0.2, 0.3, 0.4)),
        )))
        policy = DeclarativePolicy(parse_policy(paper_document(cache="mrs")), history=history)
        policy.observe_final(observation)
        policy.observe_final(observation)
        audit = policy.audit_document()
        self.assertEqual(len(audit["mrs_updates"]), 2)
        scores = {row["expert_id"]: row["priority"] for row in audit["mrs_priorities"]}
        for expert, expected in enumerate((0.125, 0.175, 0.1975, 0.2)):
            self.assertAlmostEqual(scores[expert], expected)
        self.assertEqual(parse_history(policy.history_document()), history)
        self.assertEqual(history_document(history), score_history((
            ("z-earlier", 1, (0, 1), (0.5, 0.3, 0.19, 0.01)),
            ("a-later", 2, (3, 2), (0.1, 0.2, 0.3, 0.4)),
        )))

    def test_mrs_refuses_selected_scores_or_incomplete_all_expert_evidence(self):
        observation = queue_observation()
        route = RevealedRoute("r0", "layer", 1, (0, 1), (0.5, 0.3))
        with self.assertRaisesRegex(Unsupported, "all-expert"):
            DeclarativePolicy(parse_policy(paper_document(cache="mrs"))).observe_final(
                replace(observation, route_history=(route,)))
        bad = score_history()
        bad["entries"][0]["router_scores"].pop()
        with self.assertRaisesRegex(InvalidInput, "probabilities"):
            parse_history(bad)

    def test_paper_policies_complete_real_selected_row_graph_without_hidden_fields(self):
        execution_path = Path(__file__).resolve().parents[1] / "examples/causal-strategy/rows.execution.json"
        execution = parse_execution_spec(json.loads(execution_path.read_text()))
        for rule in ("fiddler_paths", "hybrimoe_queues"):
            document = paper_document(rule)
            for kind in ("projection", "attention", "router", "gather_rows", "scatter_rows", "combine_rows"):
                document["placements"][kind] = "gpu"
            document["release"]["rule"] = "last_use"
            session = CausalM2Session(compiled(), execution_spec=execution)
            policy = DeclarativePolicy(parse_policy(document))
            run = run_causal(policy, session, max_decisions=500)
            with self.subTest(rule=rule):
                self.assertEqual(run.evaluation.status, "valid")
                for record in run.decisions:
                    self.assertEqual(record.action, compile_decision(record.decision, record.observation))
                    self.assertFalse(hasattr(record.observation, "route_trace"))
                self.assertFalse(policy.audit_document()["claims"]["paper_strategy_reproduced"])

    def test_paper_rules_reject_staged_experts_instead_of_silent_fixed_fallback(self):
        for rule in ("fiddler_paths", "hybrimoe_queues"):
            session = CausalM2Session(compiled(), execution_spec=staged_execution())
            policy = DeclarativePolicy(parse_policy(paper_document(rule)))
            with self.subTest(rule=rule), self.assertRaisesRegex(Unsupported, "opaque"):
                run_causal(policy, session, max_decisions=500)

    def test_paper_rules_have_identical_prefix_for_different_hidden_future(self):
        execution_path = Path(__file__).resolve().parents[1] / "examples/causal-strategy/rows.execution.json"
        execution = parse_execution_spec(json.loads(execution_path.read_text()))
        source = compiled()
        for rule in ("fiddler_paths", "hybrimoe_queues"):
            document = paper_document(rule)
            for kind in ("projection", "attention", "router", "gather_rows", "scatter_rows", "combine_rows"):
                document["placements"][kind] = "gpu"
            sessions = (CausalM2Session(source, execution_spec=execution),
                        CausalM2Session(alternative(source), execution_spec=execution))
            policies = (DeclarativePolicy(parse_policy(document)), DeclarativePolicy(parse_policy(document)))
            count = 0
            while not sessions[0].observe().route_history:
                observations = tuple(session.observe() for session in sessions)
                self.assertEqual(observations[0], observations[1])
                decisions = tuple(policy(observation) for policy, observation in zip(policies, observations))
                self.assertEqual(decisions[0], decisions[1])
                self.assertEqual(policies[0].audit_document(), policies[1].audit_document())
                for session, observation, decision in zip(sessions, observations, decisions):
                    action = compile_decision(decision, observation)
                    step = session.advance() if action.kind == "ADVANCE" else session.submit(action)
                    self.assertTrue(step.ok)
                count += 1
                self.assertLess(count, 100)
            self.assertGreater(count, 0)

    def test_mrs_eviction_uses_nonactivated_scores_not_frequency(self):
        observation = small_observation()
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", "ABSENT"), ("ssd", COPY_READY)))
                        if item.id == "w0" else item for item in observation.tensors)
        tensors += (tensor("w1", 32, ("ssd",), "weight", ("ssd",), "ssd"),
                    tensor("w2", 32, ("ssd", "vram"), "weight", ("ssd",), "ssd"))
        memory = tuple(replace(item, capacity_bytes=item.capacity_bytes + 32, tensor_bytes=item.tensor_bytes + 32,
                               used_bytes=item.used_bytes + 32) if item.domain == "vram" else item
                       for item in observation.memory)
        observation = replace(observation, tensors=tensors, memory=memory,
                              experts=observation.experts + (PublicExpert(1, "w1", "layer"), PublicExpert(2, "w2", "layer")))
        document = paper_document(cache="mrs")
        document["placements"]["expert_ffn"] = "gpu"
        policy = DeclarativePolicy(parse_policy(document), history=parse_history(score_history()))
        decision = policy(observation)
        self.assertEqual((decision.kind, decision.target_id), ("EVICT", "w3"))
        frequency = deepcopy(document)
        frequency["cache"] = {"rule": "history_frequency"}
        baseline = DeclarativePolicy(parse_policy(frequency), history=parse_history(score_history()))
        self.assertEqual(baseline(observation).target_id, "w2")
