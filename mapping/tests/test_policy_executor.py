from dataclasses import replace
import unittest

from llm_infer_model.m2.causal import (
    CausalM2Session, CausalObservation, DevicePrice, PublicExpert, PublicOperation,
    PublicTensor, TransferPrice,
)
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.row_execution import parse_execution_spec, NEW_KINDS
from llm_infer_model.tensor.environment import MemoryUsage, RunningObservation, EnvironmentEvent
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_EVICT, ACTION_READ_SSD, STATUS_VALID,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT, COPY_READY, OP_NOT_STARTED, InvalidInput, Unsupported,
)
from tensor_mapping.causal import compile_decision, run_causal
from tensor_mapping.policy_executor import DeclarativePolicy
from tensor_mapping.policy_ir import parse_policy, parse_history
from .test_policy_ir import policy_document_fixture, history_fixture
from .test_causal_mapping import compiled, alternative


def make_policy(rule="fixed", cache="none", prefetch=False, device="cpu", history=(), release="none"):
    document = policy_document_fixture(rule, cache, prefetch)
    document["placements"] = {kind: device for kind in document["placements"]}
    document["release"]["rule"] = release
    return DeclarativePolicy(parse_policy(document), history=history)


def tensor(name, size, ready=(), role="intermediate", initial=(), goal=None, transfer_ns=100):
    transfers = tuple(TransferPrice(f"{src}_to_{dst}", src, dst, transfer_ns)
                      for src, dst in (("dram", "vram"), ("vram", "dram"),
                                       ("ssd", "dram"), ("dram", "ssd")))
    return PublicTensor(name, role, size,
                        tuple((location, COPY_READY if location in ready else COPY_ABSENT)
                              for location in ("dram", "vram", "ssd")), transfers,
                        allocation_bytes=(("dram", size), ("vram", size)),
                        initial_locations=initial, goal_location=goal)


def small_observation(*, free_vram=16, running=(), cpu_available=True):
    tensors = (tensor("x", 16, ("dram", "vram")),
               tensor("w0", 32, ("ssd",), "weight", ("ssd",), "ssd"),
               tensor("w3", 32, ("ssd", "vram"), "weight", ("ssd",), "ssd"),
               tensor("y", 16))
    op = PublicOperation("work", "expert_ffn", OP_NOT_STARTED, ("x", "w0"), ("y",),
                         (DevicePrice("cpu", 50, 0, 0), DevicePrice("gpu", 10, 0, 0)), expert_id=0)
    memory = (MemoryUsage("dram", 1024, 0, 0, 16, 0, 16),
              MemoryUsage("vram", 48 + free_vram, 0, 0, 48, 0, 48))
    return CausalObservation(0, (op,), tensors, running, memory, cpu_available, (), (), False, None,
                             experts=(PublicExpert(0, "w0", "layer"), PublicExpert(3, "w3", "layer")),
                             allow_copy_compute_overlap=True, allow_cpu_gpu_overlap=True, allow_eviction=True)


def staged_execution():
    # Independent explicit toy stage prices; never divide the original macro.
    document = {"format_version": "moe-execution-0.1", "variant": "selected_rows",
                "expert_variant": "swiglu_staged", "provenance": "synthetic_execution",
                "index_storage_contract": "stage_workspace_includes_row_index_temporaries",
                "intermediate_bytes_per_token": 32, "entries": [], "transfer_entries": []}
    for kind in sorted(NEW_KINDS):
        for device in ("cpu", "gpu"):
            for count in (1, 2):
                document["entries"].append({"kind": kind, "device": device, "phase": "prefill", "batch": 1,
                    "token_count": count, "hidden_bytes_per_token": 16, "intermediate_bytes_per_token": 32,
                    "duration_ns": 500 * count, "workspace_bytes": 4})
    return parse_execution_spec(document)


class PolicyExecutorTests(unittest.TestCase):
    def test_fixed_cpu_and_gpu_are_valid_and_explicit(self):
        for device in ("cpu", "gpu"):
            policy = make_policy(device=device)
            session = CausalM2Session(compiled())
            run = run_causal(policy, session, max_decisions=300)
            self.assertEqual(run.evaluation.status, STATUS_VALID)
            self.assertEqual({record.decision.device for record in run.decisions
                              if record.decision.kind == ACTION_COMPUTE}, {device})
            policy.observe_final(session.observe())
            self.assertEqual(len(policy.history_document()["entries"]), 2)
            self.assertEqual(len(policy.audit_document()["decisions"]), len(run.decisions))

    def test_same_public_prefix_different_future_keeps_identical_decisions(self):
        source = compiled()
        sessions = (CausalM2Session(source), CausalM2Session(alternative(source)))
        policies = (make_policy("current_cost", "lru", release="last_use"),
                    make_policy("current_cost", "lru", release="last_use"))
        for _ in range(6):
            observations = tuple(session.observe() for session in sessions)
            self.assertEqual(observations[0], observations[1])
            decisions = tuple(policy(observation) for policy, observation in zip(policies, observations))
            self.assertEqual(decisions[0], decisions[1])
            self.assertEqual(policies[0].audit_document(), policies[1].audit_document())
            for session, observation, decision in zip(sessions, observations, decisions):
                action = compile_decision(decision, observation)
                result = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
                self.assertTrue(result.ok)

    def test_resident_expert_rule_uses_cpu_and_gpu(self):
        source = compiled()
        weights = tuple(replace(weight, locations=("ssd", "vram")) if weight.id == "E1" else weight
                        for weight in source.state_in.weight_copies)
        state = replace(source.state_in, weight_copies=weights)
        invocation = replace(source.invocation, state_in=state)
        session = CausalM2Session(compile_invocation(source.layer, invocation, source.cost_table))
        run = run_causal(make_policy("resident_first"), session, max_decisions=300)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual({record.decision.device for record in run.decisions
                          if record.decision.kind == ACTION_COMPUTE}, {"cpu", "gpu"})
        gpu_expert = next(record for record in run.decisions if record.decision.kind == ACTION_COMPUTE
                          and record.decision.device == "gpu")
        self.assertEqual(next(op for op in gpu_expert.observation.operations
                              if op.id == gpu_expert.decision.target_id).expert_id, 1)

    def test_staged_fixed_bindings_execute_heterogeneous_ffn(self):
        document = policy_document_fixture()
        for kind in ("expert_gate", "expert_up"):
            document["placements"][kind] = "gpu"
        policy = DeclarativePolicy(parse_policy(document))
        session = CausalM2Session(compiled(), execution_spec=staged_execution())
        run = run_causal(policy, session, max_decisions=500)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        stages = {record.decision.target_id: record.decision.device for record in run.decisions
                  if record.decision.kind == ACTION_COMPUTE}
        operations = {op.id: op for op in session.observe().operations}
        self.assertTrue(any(operations[oid].kind == "expert_gate" and device == "gpu"
                            for oid, device in stages.items()))
        self.assertTrue(any(operations[oid].kind == "expert_down" and device == "cpu"
                            for oid, device in stages.items()))
        self.assertTrue(any(record.action.kind == "COPY" and record.action.src == "vram"
                            and record.action.dst == "dram" for record in run.decisions))
        self.assertTrue(all(record.accepted for record in run.decisions))

    def test_last_use_release_reduces_staged_peak_without_changing_work(self):
        document = policy_document_fixture()
        for kind in ("expert_gate", "expert_up"):
            document["placements"][kind] = "gpu"
        runs = []
        for rule in ("none", "last_use"):
            document["release"]["rule"] = rule
            session = CausalM2Session(compiled(), execution_spec=staged_execution())
            policy = DeclarativePolicy(parse_policy(document))
            run = run_causal(policy, session, max_decisions=500)
            self.assertEqual(run.evaluation.status, STATUS_VALID)
            runs.append(run)
            if rule == "last_use":
                releases = [record for record in run.decisions if record.action.kind == ACTION_EVICT]
                self.assertTrue(releases)
                self.assertTrue(all(record.observation.task_graph_complete for record in releases))
                self.assertTrue(all(next(tensor for tensor in record.observation.tensors
                                         if tensor.id == record.action.target_id).role == "intermediate"
                                    for record in releases))
        self.assertLess(runs[1].evaluation.peak_dram_bytes, runs[0].evaluation.peak_dram_bytes)
        self.assertLess(runs[1].evaluation.peak_vram_bytes, runs[0].evaluation.peak_vram_bytes)
        self.assertEqual(runs[1].evaluation.makespan_ns, runs[0].evaluation.makespan_ns)

    def test_last_use_waits_for_complete_graph_and_preserves_inflight_copies(self):
        base = small_observation(free_vram=512)
        dead = tensor("dead", 16, ("dram", "vram"))
        consumer = replace(base.operations[0], id="dead-consumer", status="DONE", inputs=("dead",), outputs=())
        base = replace(base, tensors=base.tensors + (dead,), operations=base.operations + (consumer,),
                       task_graph_complete=True)
        policy = make_policy(release="last_use")
        decision = policy(base)
        self.assertEqual((decision.kind, decision.target_id, decision.location),
                         (ACTION_EVICT, "dead", "dram"))
        blocked = [replace(base, task_graph_complete=False),
                   replace(base, running=(RunningObservation("pcie_copy", "dead", None, "dram_to_vram"),)),
                   replace(base, tensors=base.tensors[:-1] +
                           (replace(dead, copies=(("dram", COPY_READY), ("vram", "RESERVED_COPY"),
                                                  ("ssd", COPY_ABSENT))),)),
                   replace(base, tensors=base.tensors[:-1] + (replace(dead, initial_locations=("dram",)),)),
                   replace(base, tensors=base.tensors[:-1] + (replace(dead, goal_location="vram"),))]
        for observation in blocked:
            with self.subTest(observation=observation):
                decision = make_policy(release="last_use")(observation)
                self.assertFalse(decision.kind == ACTION_EVICT and decision.target_id == "dead")

    def test_current_cost_accounts_for_public_transfer_prices(self):
        observation = small_observation(free_vram=512)
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", COPY_ABSENT),
                                             ("ssd", COPY_READY))) if item.id == "w0" else item
                        for item in observation.tensors)
        observation = replace(observation, tensors=tensors)
        decision = make_policy("current_cost", device="gpu")(observation)
        self.assertEqual((decision.kind, decision.device), (ACTION_COMPUTE, "cpu"))

    def test_lru_eviction_preserves_immutable_backup(self):
        observation = small_observation()
        policy = make_policy(cache="lru", device="gpu")
        # Initial SSD read fits DRAM. VRAM headroom is checked when issuing H2D.
        decision = policy(observation)
        self.assertEqual(decision.kind, ACTION_READ_SSD)
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", COPY_ABSENT),
                                             ("ssd", COPY_READY))) if item.id == "w0" else item
                        for item in observation.tensors)
        decision = policy(replace(observation, tensors=tensors))
        self.assertEqual((decision.kind, decision.target_id, decision.location), (ACTION_EVICT, "w3", "vram"))
        self.assertEqual(policy.audit_document()["decisions"][-1]["reason"], "demand_capacity_eviction")

    def test_no_eviction_of_goal_only_copy_or_in_use_expert_copy(self):
        base = small_observation()
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", COPY_ABSENT),
                                             ("ssd", COPY_READY))) if item.id == "w0" else item
                        for item in base.tensors)
        base = replace(base, tensors=tensors)
        variants = []
        for field, value in (("goal_location", "vram"),
                             ("copies", (("dram", COPY_ABSENT), ("vram", COPY_READY), ("ssd", COPY_ABSENT)))):
            variants.append(replace(base, tensors=tuple(replace(item, **{field: value}) if item.id == "w3"
                                                        else item for item in base.tensors)))
        variants.append(replace(base, running=(RunningObservation("pcie_copy", "w3", None, "vram_to_dram"),)))
        for observation in variants:
            with self.subTest(observation=observation):
                decision = make_policy(cache="lru", device="gpu")(observation)
                self.assertEqual(decision.kind, ACTION_ADVANCE)

    def test_inherited_vram_cache_remains_evictable_with_ssd_backup(self):
        observation = small_observation()
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", COPY_ABSENT),
                                             ("ssd", COPY_READY))) if item.id == "w0" else
                        replace(item, initial_locations=("ssd", "vram")) if item.id == "w3" else item
                        for item in observation.tensors)
        decision = make_policy(cache="lru", device="gpu")(replace(observation, tensors=tensors))
        self.assertEqual((decision.kind, decision.target_id, decision.location), (ACTION_EVICT, "w3", "vram"))

    def test_history_frequency_evicts_less_frequent_revealed_expert(self):
        observation = small_observation()
        tensors = tuple(replace(item, copies=(("dram", COPY_READY), ("vram", COPY_ABSENT),
                                             ("ssd", COPY_READY))) if item.id == "w0" else item
                        for item in observation.tensors)
        tensors += (tensor("w1", 32, ("ssd", "vram"), "weight", ("ssd", "vram"), "ssd"),)
        history = history_fixture((3,), layer="layer")
        history["entries"].append({"request_id": "older-request", "layer_id": "layer",
                                    "absolute_position": 9, "expert_ids": [3], "scores": None})
        observation = replace(observation, tensors=tensors,
                              experts=observation.experts + (PublicExpert(1, "w1", "layer"),))
        decision = make_policy(cache="history_frequency", device="gpu",
                               history=parse_history(history))(observation)
        self.assertEqual((decision.kind, decision.target_id), (ACTION_EVICT, "w1"))

    def test_compute_reserves_aligned_outputs_and_workspace_before_launch(self):
        observation = small_observation(free_vram=16)
        tensors = tuple(replace(item, copies=(("dram", COPY_ABSENT), ("vram", COPY_READY),
                                             ("ssd", COPY_READY))) if item.id == "w0" else item
                        for item in observation.tensors)
        prices = (DevicePrice("gpu", 10, 1, 32),)
        observation = replace(observation, tensors=tensors,
                              operations=(replace(observation.operations[0], prices=prices),))
        decision = make_policy(cache="lru", device="gpu")(observation)
        self.assertEqual((decision.kind, decision.target_id), (ACTION_EVICT, "w3"))
        self.assertEqual(make_policy(cache="none", device="gpu")(observation).kind, ACTION_ADVANCE)

    def test_prefetch_does_not_reissue_same_speculative_hop_after_eviction(self):
        policy = make_policy(prefetch=True, history=parse_history(history_fixture(layer="layer")))
        observation = replace(small_observation(free_vram=512), operations=(),
                              running=(RunningObservation("cpu_compute", "unexposed-current", "cpu", None),),
                              tensors=tuple(replace(item, copies=(("dram", COPY_ABSENT),
                                                                  ("vram", COPY_ABSENT), ("ssd", COPY_READY)))
                                            if item.id == "w3" else item
                                            for item in small_observation(free_vram=512).tensors))
        first = policy(observation)
        self.assertEqual((first.kind, first.target_id), (ACTION_READ_SSD, "w3"))
        second = policy(observation)
        self.assertEqual(second.kind, ACTION_ADVANCE)

    def test_history_prefetch_can_miss_and_still_pays_transfers(self):
        policy = make_policy(cache="history_frequency", prefetch=True,
                             history=parse_history(history_fixture()))
        session = CausalM2Session(compiled())
        run = run_causal(policy, session, max_decisions=300)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        policy.observe_final(session.observe())
        self.assertEqual(policy.audit_document()["predictions"],
                         [{"tensor_id": "w:moe-small:E3", "outcome": "miss"}])
        actions = [record.action for record in run.decisions if record.action.target_id == "w:moe-small:E3"]
        self.assertEqual(len(actions), 2)
        self.assertEqual({action.kind for action in actions}, {"READ_SSD", "COPY"})
        self.assertTrue(all(item.status_at("vram") == COPY_READY for item in session.observe().tensors
                            if item.id == "w:moe-small:E3"))

    def test_prefetch_requires_same_layer_history_and_no_history_means_no_guess(self):
        for history in ((), parse_history(history_fixture(layer="another-layer"))):
            policy = make_policy(prefetch=True, history=history)
            session = CausalM2Session(compiled())
            run = run_causal(policy, session, max_decisions=300)
            self.assertEqual(run.evaluation.status, STATUS_VALID)
            # Current revealed history may populate a correct prediction, but
            # the initially inactive E3 cannot be guessed from another layer.
            self.assertFalse(any(record.action.target_id == "w:moe-small:E3" for record in run.decisions))

    def test_history_is_deduplicated_and_conflicts_fail(self):
        session = CausalM2Session(compiled())
        policy = make_policy()
        run_causal(policy, session, max_decisions=300)
        final = session.observe()
        for _ in range(3):
            policy.observe_final(final)
        self.assertEqual(len(policy.history_document()["entries"]), 2)
        changed = replace(final.route_history[0], expert_ids=(3,))
        with self.assertRaises(InvalidInput):
            policy.observe_final(replace(final, route_history=(changed,)))

    def test_cpu_blackout_waits_without_future_environment_information(self):
        policy = make_policy()
        session = CausalM2Session(compiled(), environment_events=(EnvironmentEvent(0, cpu_available=False),
                                                                EnvironmentEvent(2_000_000, cpu_available=True)))
        run = run_causal(policy, session, max_decisions=300)
        self.assertEqual(run.evaluation.status, STATUS_VALID)
        self.assertEqual(run.decisions[0].decision.kind, ACTION_ADVANCE)
        self.assertFalse(hasattr(run.decisions[0].observation, "environment_events"))

    def test_unknown_kind_missing_alignment_and_budget_are_explicit(self):
        observation = small_observation(free_vram=512)
        policy = make_policy(device="gpu")
        operation = replace(observation.operations[0], kind="not_declared")
        with self.assertRaises(Unsupported):
            policy(replace(observation, operations=(operation,)))
        prices = (DevicePrice("gpu", 10, 0),)
        with self.assertRaises(Unsupported):
            policy(replace(observation, operations=(replace(observation.operations[0], prices=prices),)))
        session = CausalM2Session(compiled())
        run = run_causal(make_policy(), session, max_decisions=1)
        self.assertEqual(run.termination_reason, "decision_budget_exhausted")
        self.assertEqual(len(run.decisions), 1)
        self.assertFalse(make_policy().audit_document()["claims"]["optimality"])
