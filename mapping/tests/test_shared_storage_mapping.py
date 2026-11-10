"""Shared physical storage through the same causal strategy compiler.

These are semantic acceptance tests with explicit fictional costs, not hardware
accuracy measurements. Independent COPY and zero-copy SHARE remain distinct.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llm_simulator import (Cache, DataPreparation, Hardware, PhaseHandoff,
                           Prefetch, RequestPipeline, Scheduling, Simulator)
from llm_simulator.demo import make_demo
from llm_infer_model.m2.causal import PublicTensor, TransferPrice
from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, Unsupported
from tensor_mapping.causal import Decision, compile_command
from tensor_mapping.policy_executor import DeclarativePolicy
from tensor_mapping.unified_strategy import create_session


def shared_simulator(kind="dense"):
    demo, _ = make_demo(kind)
    hardware = Hardware.shared_memory(cpu=demo.hardware.cpu, gpu=demo.hardware.gpu,
        capacity_bytes=64 * 1024 * 1024,
        ssd_read_bandwidth_bytes_per_second=1e9, ssd_write_bandwidth_bytes_per_second=5e8)
    return Simulator(model=demo.model, hardware=hardware, workload=demo.workload, cost_model=demo.cost_model)


def schedule(**changes):
    result = Scheduling(default_device="gpu", release="last_use",
                        cache=Cache(rule="lru", budget_bytes=1024 * 1024))
    return replace(result, **changes)


class SharedPublicStorageTests(unittest.TestCase):
    def tensor(self):
        return PublicTensor("w", "weight", 64, (("dram", COPY_READY), ("vram", COPY_ABSENT)),
            (TransferPrice("dram_to_vram", "dram", "vram", 0, timing_scope="shared_access_only"),),
            allocation_bytes=(("dram", 64), ("vram", 64)),
            memory_pools=(("dram", "shared"), ("vram", "shared")))

    def test_preparation_requires_explicit_share_without_physical_allocation(self):
        tensor = self.tensor()
        decision, price, resource = DeclarativePolicy._load_step(tensor, "vram")
        self.assertEqual(decision.kind, "SHARE")
        self.assertEqual((price.duration_ns, resource), (0, "shared_view"))
        self.assertEqual(DeclarativePolicy._allocation(tensor, "vram"), 0)
        self.assertEqual(DeclarativePolicy._transfer_estimate(tensor, "vram"), 0)
        self.assertTrue(tensor.can_access("vram"))
        self.assertEqual(tensor.status_at("vram"), COPY_ABSENT)

    def test_alias_is_neither_independent_backup_nor_physical_reclamation(self):
        tensor = replace(self.tensor(), copies=(("dram", COPY_READY), ("vram", COPY_READY)),
                         allocation_refs=(("dram", "a0"), ("vram", "a0")),
                         physical_release_bytes=(("dram", 0), ("vram", 0)),
                         independent_backups=(("dram", False), ("vram", False)))
        self.assertEqual(tensor.released_bytes("vram"), 0)
        self.assertFalse(tensor.has_independent_backup("vram"))
        self.assertIsNone(tensor.share_source("vram"))

    def test_separate_copy_cost_is_unchanged(self):
        tensor = replace(self.tensor(), memory_pools=(),
            transfers=(TransferPrice("dram_to_vram", "dram", "vram", 17),))
        decision, price, resource = DeclarativePolicy._load_step(tensor, "vram")
        self.assertEqual((decision.kind, price.duration_ns, resource), ("COPY", 17, "pcie_copy"))
        self.assertEqual(DeclarativePolicy._allocation(tensor, "vram"), 64)

    def test_physical_shared_shortfall_can_release_other_device_view(self):
        class Probe(DeclarativePolicy):
            def __init__(self):
                self.seen = []
            def _eviction(self, observation, tensors, location, protected, *, require_physical_release=False):
                self.seen.append((location, require_physical_release))
                return Decision("EVICT", "w", location="dram") if location == "dram" else None
        observation = SimpleNamespace(memory=(SimpleNamespace(domain="vram", available_bytes=0),))
        policy = Probe()
        fits, action = policy._fit_or_evict(observation, {"w": self.tensor()}, "vram", 64, set())
        self.assertFalse(fits)
        self.assertEqual(action.location, "dram")
        self.assertEqual(policy.seen, [("vram", True), ("dram", True)])
        separate = replace(self.tensor(), memory_pools=())
        policy = Probe()
        self.assertEqual(policy._fit_or_evict(observation, {"w": separate}, "vram", 64, set()), (False, None))
        self.assertEqual(policy.seen, [("vram", True)])


class SharedMappingIntegrationTests(unittest.TestCase):
    def test_offline_ir_share_is_distinct_from_explicit_physical_transfer(self):
        from .test_strategy_compiler import documents, strategy_document
        from llm_infer_model.tensor.spec import build_scenario
        from tensor_mapping.strategy_ir import parse_strategy_ir, strategy_to_document
        from tensor_mapping.strategy_compiler import compile_strategy, run_strategy
        graph, scene = documents()
        scene["architecture"].update(memory_mode="shared", shared_capacity_bytes=1024)
        raw = strategy_document()
        raw["program"][1]["kind"] = "share"
        scenario = build_scenario(graph, scene)
        ir = parse_strategy_ir(raw)
        self.assertEqual(strategy_to_document(ir), raw)
        shared = run_strategy(compile_strategy(ir, scenario)).evaluation
        self.assertEqual(shared.status, "valid")
        self.assertEqual(shared.h2d_bytes, 0)
        self.assertTrue(any(event.kind == "SHARE" for event in shared.events))
        physical = run_strategy(compile_strategy(parse_strategy_ir(strategy_document()), scenario)).evaluation
        self.assertEqual(physical.status, "valid")
        self.assertEqual(physical.h2d_bytes, 64)
        scene["costs"]["transfer"]["dram_to_vram"] = {}
        scenario = build_scenario(graph, scene)
        self.assertEqual(run_strategy(compile_strategy(ir, scenario)).evaluation.status, "valid")
        with self.assertRaisesRegex(Unsupported, "missing exact transfer"):
            compile_strategy(parse_strategy_ir(strategy_document()), scenario)

    def test_shared_only_route_cannot_compile_into_physical_copy(self):
        simulator = shared_simulator()
        with simulator.compile(scheduling=schedule()) as compiled:
            session = create_session(compiled.plan._compiled)
            observation = session.observe()
            tensor = next(t for t in observation.tensors if t.share_source("vram"))
            share = Decision("SHARE", tensor.id, src="dram", dst="vram")
            self.assertEqual(compile_command(share, observation).action.kind, "SHARE")
            with self.assertRaisesRegex(Unsupported, "physical COPY"):
                compile_command(Decision("COPY", tensor.id, src="dram", dst="vram"), observation)

    def test_share_cannot_compile_where_there_is_no_shared_pool(self):
        # The same rule in the other direction: SHARE aliases two views of *one*
        # physical pool, so a machine with DRAM and VRAM as separate pools has no
        # such action to take. A strategy that asks for one is refused rather
        # than priced at zero -- which is what it would cost if the compiler
        # trusted the name of the action instead of the host's pools.
        demo, _ = make_demo("dense")
        simulator = Simulator(model=demo.model, hardware=demo.hardware,
                              workload=demo.workload, cost_model=demo.cost_model)
        with simulator.compile(scheduling=schedule()) as compiled:
            session = create_session(compiled.plan._compiled)
            observation = session.observe()
            tensor = next(t for t in observation.tensors
                          if any(price.route == "dram_to_vram" for price in t.transfers))
            share = Decision("SHARE", tensor.id, src="dram", dst="vram")
            with self.assertRaisesRegex(Unsupported, "shared physical memory pool"):
                compile_command(share, observation)

    def test_startup_preload_and_cross_layer_prefetch_use_shared_views(self):
        simulator = shared_simulator()
        settings = schedule(prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True))
        def configure(plan):
            plan.prepare(DataPreparation(layer_id="layer0", destination="vram", trigger="startup"))
        with simulator.compile(scheduling=settings, configure=configure) as compiled:
            result = compiled.run(max_decisions=2000)
            self.assertTrue(result.success, result.summary())
            audit = result.raw[2].audit_document()
            self.assertEqual(audit["cache_budget_scope"], "explicit_gpu_access_view_payload_quota")
            self.assertFalse(audit["cache_budget_is_physical_capacity"])
            self.assertFalse(any(event.kind == "COPY" for event in result.events))
            self.assertTrue(any(row.decision.kind == "SHARE" for row in result.raw[0].decisions))
            prepared = audit.get("explicit_data_preparations", [])
            self.assertTrue(prepared)

    def test_dynamic_expert_paths_use_zero_shared_boundary_costs(self):
        simulator = shared_simulator("moe")
        with simulator.compile(scheduling=schedule(expert_rule="current_cost")) as compiled:
            result = compiled.run(max_decisions=3000)
            self.assertTrue(result.success, result.summary())
            paths = [path for row in result.raw[2].audit_document()["region_bindings"] for path in row["paths"]]
            self.assertTrue(paths)
            for path in paths:
                self.assertEqual(path["weight_transfer_ns"], 0)
                self.assertEqual(path["selected_input_transfer_ns"], 0)
                self.assertEqual(path["output_return_ns"], 0)

    def test_shared_checkpoint_replay_preserves_physical_allocation_identity(self):
        simulator = shared_simulator("moe")
        with simulator.compile(scheduling=schedule()) as compiled:
            partial = compiled.plan.run(max_decisions=13)
            checkpoint = compiled.plan.checkpoint(partial)
            resumed = compiled.plan.run(checkpoint=checkpoint)
            full = compiled.plan.run()
            self.assertEqual(resumed[0].termination_reason, "goal_reached")
            self.assertEqual(resumed[0].evaluation.events, full[0].evaluation.events)
            self.assertEqual(resumed[0].runtime_result.final_state.storage_allocations,
                             full[0].runtime_result.final_state.storage_allocations)
            self.assertEqual(resumed[1].persistent_state_document()["runtime_state"],
                             full[1].persistent_state_document()["runtime_state"])

    def test_single_layer_inventory_json_roundtrip_keeps_alias_identity(self):
        from llm_infer_model.m2.expand import load_and_compile
        from llm_infer_model.m2.causal import CausalM2Session
        from llm_infer_model.tensor.engine import used_physical_bytes
        from tensor_mapping.causal import SerialDemandPolicy, run_causal
        simulator = shared_simulator()
        with simulator.compile(scheduling=schedule()) as compiled:
            leaf = load_and_compile(compiled.directory / "layer0-prefill.json")
            session = CausalM2Session(leaf)
            result = run_causal(SerialDemandPolicy("gpu"), session, max_decisions=2000)
            self.assertEqual(result.termination_reason, "goal_reached")
            with self.assertRaisesRegex(Unsupported, "allocation identity"):
                session.state_out()
            restored = json.loads(json.dumps(session.persistent_state_document()))
            raw = restored["runtime_state"]
            original = session._runtime.state
            rebuilt = replace(original, storage_allocations=tuple(tuple(row) for row in raw["storage_allocations"]),
                              next_allocation_id=raw["next_allocation_id"])
            self.assertEqual(rebuilt.storage_allocations, original.storage_allocations)
            self.assertEqual(used_physical_bytes(session._runtime.scenario, rebuilt),
                             used_physical_bytes(session._runtime.scenario, original))
            self.assertEqual(raw["next_allocation_id"], original.next_allocation_id)
            self.assertFalse(restored["resumable_checkpoint"])

    def test_block_inventory_json_roundtrip_keeps_alias_identity(self):
        from tensor_mapping.composition_block import load_block_model_file, create_session as block_session
        from tensor_mapping.causal import SerialDemandPolicy, run_causal
        from llm_infer_model.tensor.engine import used_physical_bytes
        path = Path(__file__).resolve().parents[1] / "examples/papers/flexgen-layer-major.blocks.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["hardware"].update(memory_mode="shared", shared_capacity_bytes=65536)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "block.json"
            target.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            session = block_session(load_block_model_file(target))
            result = run_causal(SerialDemandPolicy("gpu"), session, max_decisions=2000)
            self.assertEqual(result.termination_reason, "goal_reached")
            restored = json.loads(json.dumps(session.persistent_state_document()))
            original = session._runtime.state
            raw = restored["runtime_state"]
            rebuilt = replace(original, storage_allocations=tuple(tuple(row) for row in raw["storage_allocations"]),
                              next_allocation_id=raw["next_allocation_id"])
            self.assertEqual(rebuilt.storage_allocations, original.storage_allocations)
            self.assertEqual(used_physical_bytes(session._runtime.scenario, rebuilt),
                             used_physical_bytes(session._runtime.scenario, original))
            self.assertFalse(restored["resumable_checkpoint"])

    def test_pd_handoff_in_shared_pool_does_not_force_decode_device(self):
        from examples.library.request_pipeline import make_simulator
        original = make_simulator()
        hardware = Hardware.shared_memory(cpu=original.hardware.cpu, gpu=original.hardware.gpu,
            capacity_bytes=1024 * 1024, ssd_read_bandwidth_bytes_per_second=1e9,
            ssd_write_bandwidth_bytes_per_second=1e9)
        simulator = Simulator(model=original.model, hardware=hardware, workload=original.workload,
                              cost_model=original.cost_model)
        def configure(plan):
            plan.pipeline(RequestPipeline(ready_rule="decode_first"))
            plan.handoff(PhaseHandoff(destination="dram"))
        with simulator.compile(scheduling=schedule(), configure=configure) as compiled:
            result = compiled.run(max_decisions=3000)
            self.assertTrue(result.success, result.summary())
            operations = {op.id: op for op in result.raw[1].observe().operations}
            gates = [e for e in result.events if e.kind == "COMPUTE" and
                     operations[e.target_id].execution_stage == "prefill_decode_handoff"]
            decode = [e for e in result.events if e.kind == "COMPUTE" and
                      operations[e.target_id].phase == "decode" and operations[e.target_id].kind == "attention"]
            self.assertTrue(gates)
            self.assertTrue(decode)
            self.assertTrue(all(e.device == "gpu" for e in decode))
            for gate in gates:
                for tid in operations[gate.target_id].inputs:
                    producer = next((e for e in result.events if e.kind == "COMPUTE" and
                                     tid in operations[e.target_id].outputs), None)
                    if producer:
                        self.assertGreaterEqual(gate.start_ns, producer.end_ns)

    def test_shared_moe_request_pipeline_preserves_route_revelation(self):
        from examples.library.request_pipeline import make_simulator, configure
        original = make_simulator("moe")
        hardware = Hardware.shared_memory(cpu=original.hardware.cpu, gpu=original.hardware.gpu,
            capacity_bytes=1024 * 1024, ssd_read_bandwidth_bytes_per_second=1e9,
            ssd_write_bandwidth_bytes_per_second=1e9)
        simulator = Simulator(model=original.model, hardware=hardware, workload=original.workload,
                              cost_model=original.cost_model)
        settings = schedule(expert_rule="current_cost", expert_order="token_count_desc",
                            prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True))
        with simulator.compile(scheduling=settings, configure=lambda plan: configure(plan, "moe")) as compiled:
            result = compiled.run(max_decisions=5000)
            self.assertTrue(result.success, result.summary())
            self.assertEqual((result.summary()["h2d_bytes"], result.summary()["d2h_bytes"]), (0, 0))
            self.assertEqual(set(result.raw[1].observe().retired_requests), {"interactive", "background"})
            saw_expert = False
            for record in result.raw[0].decisions:
                expert_operations = [op for op in record.observation.operations if op.kind == "expert_ffn"]
                for expert in expert_operations:
                    saw_expert = True
                    routers = [op for op in record.observation.operations
                               if op.kind == "router" and op.invocation_id == expert.invocation_id]
                    self.assertTrue(routers)
                    self.assertTrue(all(op.status == "DONE" for op in routers))
            self.assertTrue(saw_expert)


if __name__ == "__main__":
    unittest.main()
