"""Synthetic native-protocol fixtures for the measured MoE compiler.

These tests do not execute GGML, measure hardware, authenticate published data,
or establish numerical correctness. They verify the simulator contract and
quality/capacity gates using deliberately constructed native evidence.
"""
from __future__ import annotations
import copy
import hashlib
import unittest

from llm_infer_model.m2.moe_ffn_contract import (
    CONTRACT, NATIVE_SCHEMA, TIMING_SCOPE, contract_hash, groups,
    physical_signature, stage_memory, stage_names, stage_ports,
)
from llm_infer_model.m2.moe_measured import (
    compile_measured_ffn, plan_measured_ffn, replay_measured_ffn,
)
from llm_infer_model.tensor.engine import (
    Action, ACTION_COMPUTE, ACTION_ADVANCE, evaluate_mapping,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def protocol_fixture():
    """Explicit fabricated timings/digests; never label this actual measurement."""
    case = dict(id="synthetic-native-protocol", backend="CPU", threads=1,
                config=dict(hidden=4, intermediate=8, experts=4, top_k=2, tokens=2),
                routes=[[0, 1], [1, 2]], phase="prefill",
                work_items=[dict(request_id="r0", absolute_position=i) for i in range(2)],
                comment="SYNTHETIC UNIT-TEST PROTOCOL FIXTURE; NOT A HARDWARE MEASUREMENT")
    native = dict(schema_version=NATIVE_SCHEMA, contract=CONTRACT, backend="CPU",
                  routing_mode="prescribed_trace_replay", scope="resident_moe_ffn_subgraph",
                  random_weights_not_model_weights=True, router_discovery_used_for_grouping=False,
                  dtype="f32", config=copy.deepcopy(case["config"]), routes=copy.deepcopy(case["routes"]),
                  discovered_routes=copy.deepcopy(case["routes"]), threads_requested=1, threads_applied=1,
                  timing_scope=TIMING_SCOPE, port_digest_algorithm="fnv1a64_native_tensor_bytes",
                  warmup=1, samples=3, seed=7, stages={})
    for index, stage in enumerate(stage_names(case)):
        ports = stage_ports(case, stage)
        for name, port in ports.items():
            port["digest"] = hashlib.sha256(name.encode()).hexdigest()[:16]
        memory = stage_memory(case, stage)
        logical = memory["resident_logical_bytes"]
        internal, plan = 64 + index * 16, 32
        native["stages"][stage] = dict(
            samples_ns=[(index + 1) * 100] * 3 if stage != "joint" else [999999] * 3,
            workspace_bytes=internal + plan,
            workspace_source="ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
            memory_accounting=dict(resident_logical_bytes=logical,
                resident_allocated_bytes=((logical + 63) // 64) * 64,
                internal_buffer_bytes=internal, cpu_graph_plan_work_bytes=plan,
                metadata_bytes_included=False), all_outputs_finite=True,
            reference_check=dict(passed=True, max_abs_error=0.0, max_rel_error=0.0, atol=0.0001, rtol=0.0001),
            ports=ports)
    signature = physical_signature(case, "a" * 64)
    return dict(case_id=case["id"], case=case, status="measured", native=native,
                quality_flags=[], validation_only=False, physical_signature=signature,
                physical_contract_sha256=contract_hash(signature))


def hardware_fixture():
    return dict(dram_capacity_bytes=1 << 20, vram_capacity_bytes=4096,
                dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
                dram_alignment_bytes=1, vram_alignment_bytes=1)


class MeasuredMoEFFNTests(unittest.TestCase):
    def compile(self, record=None, hardware=None, **kwargs):
        return compile_measured_ffn(record or protocol_fixture(), hardware or hardware_fixture(), **kwargs)

    def test_physical_ports_match_f32_i32_geometry(self):
        compiled = self.compile()
        ports = stage_ports(compiled.case, "joint")
        tensors = {tensor["id"]: tensor for tensor in compiled.workload_document["tensors"]}
        self.assertEqual(set(tensors), set(ports))
        for name, port in ports.items():
            with self.subTest(port=name):
                tensor = tensors[name]
                a, b, c, _ = port["ne"]
                self.assertEqual(tensor["ne"], port["ne"])
                self.assertEqual(tensor["nb"], [4, 4*a, 4*a*b, 4*a*b*c])
                self.assertEqual(tensor["storage_bytes"], port["bytes"])
                self.assertEqual(tensor["dtype"], "GGML_TYPE_I32" if port["dtype"] == "i32" else "GGML_TYPE_F32")
                self.assertEqual(tensor["initial_locations"], ["dram"] if port["category"] in ("input", "weight") else [])

    def test_active_group_once_and_inactive_expert_absent(self):
        compiled = self.compile()
        operations = compiled.workload_document["operations"]
        self.assertEqual([operation["id"] for operation in operations], ["router", "expert.0", "expert.1", "expert.2", "combine"])
        self.assertEqual(compiled.bindings["groups"], {"0": 1, "1": 2, "2": 1})
        self.assertNotIn("gate.3", {tensor["id"] for tensor in compiled.workload_document["tensors"]})

    def test_workspace_contains_only_internal_and_plan(self):
        record = protocol_fixture()
        compiled = self.compile(record)
        for stage, costs in compiled.scenario_document["costs"]["compute"].items():
            native = record["native"]["stages"][stage]
            memory = native["memory_accounting"]
            self.assertEqual(costs["cpu"]["workspace_bytes"], memory["internal_buffer_bytes"] + memory["cpu_graph_plan_work_bytes"])
            self.assertNotEqual(costs["cpu"]["workspace_bytes"], native["workspace_bytes"] + memory["resident_allocated_bytes"])
        self.assertFalse(compiled.bindings["physical_capacity_verified"])

    def test_joint_duration_does_not_become_a_price(self):
        compiled = self.compile()
        self.assertNotIn("joint", compiled.scenario_document["costs"]["compute"])
        result = replay_measured_ffn(compiled)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.makespan_ns, 1500)
        self.assertNotEqual(result.makespan_ns, 999999)

    def test_joint_variation_does_not_change_stage_prices(self):
        record = protocol_fixture()
        record["native"]["stages"]["joint"]["samples_ns"] = [1, 1, 999999]
        compiled = self.compile(record)
        self.assertEqual(replay_measured_ffn(compiled).makespan_ns, 1500)

    def test_high_cv_default_rejects_even_forged_record_summary(self):
        record = protocol_fixture()
        record["native"]["stages"]["expert.1"]["samples_ns"] = [100, 100, 10000]
        record.update(summary={"expert.1": {"cv": 0.0}}, quality_flags=[], validation_only=False)
        with self.assertRaises(Unsupported):
            self.compile(record)

    def test_high_cv_explicit_opt_in_retains_validation_only(self):
        record = protocol_fixture()
        record["native"]["stages"]["expert.1"]["samples_ns"] = [100, 100, 10000]
        compiled = self.compile(record, allow_unstable=True)
        self.assertTrue(compiled.validation_only)
        self.assertTrue(compiled.bindings["validation_only"])
        self.assertIn("expert.1", compiled.bindings["unstable_cost_stages"])
        self.assertIn("validation_only=true", compiled.scenario_document["costs"]["source"])

    def test_allow_unstable_must_be_bool(self):
        for value in (1, 0, "true", None):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                self.compile(allow_unstable=value)

    def test_route_change_without_new_signature_rejected(self):
        record = protocol_fixture()
        record["case"]["routes"] = [[1, 0], [1, 2]]
        record["native"]["routes"] = [[1, 0], [1, 2]]
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_native_route_change_rejected(self):
        record = protocol_fixture()
        record["native"]["routes"][0] = [0, 2]
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_missing_or_extra_stage_rejected(self):
        for missing in (True, False):
            record = protocol_fixture()
            if missing:
                del record["native"]["stages"]["expert.1"]
            else:
                record["native"]["stages"]["expert.3"] = copy.deepcopy(record["native"]["stages"]["expert.2"])
            with self.subTest(missing=missing), self.assertRaises(InvalidInput):
                self.compile(record)

    def test_unknown_workspace_rejected(self):
        record = protocol_fixture()
        record["native"]["stages"]["router"]["workspace_bytes"] = None
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_resident_workspace_double_count_rejected(self):
        record = protocol_fixture()
        stage = record["native"]["stages"]["router"]
        stage["workspace_bytes"] += stage["memory_accounting"]["resident_allocated_bytes"]
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_cpu_only_device_mapping(self):
        compiled = self.compile()
        self.assertTrue(all(devices == ["cpu"] for devices in compiled.scenario_document["mapspace"]["allowed_devices"].values()))
        self.assertTrue(all(action.device == "cpu" for action in plan_measured_ffn(compiled) if action.kind == ACTION_COMPUTE))
        actions = (Action(ACTION_COMPUTE, "router", device="gpu"), Action(ACTION_ADVANCE))
        self.assertEqual(evaluate_mapping(compiled.scenario, actions).status, "invalid_mapping")

    def test_cuda_record_rejected_without_fallback(self):
        record = protocol_fixture()
        record["case"]["backend"] = "CUDA0"
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_expert_cannot_run_before_router(self):
        compiled = self.compile()
        result = evaluate_mapping(compiled.scenario, (Action(ACTION_COMPUTE, "expert.0", device="cpu"),))
        self.assertEqual(result.status, "invalid_mapping")

    def test_combine_requires_all_expert_results(self):
        compiled = self.compile()
        actions = tuple(action for action in plan_measured_ffn(compiled)
                        if not (action.kind == ACTION_COMPUTE and action.target_id == "expert.1"))
        self.assertEqual(evaluate_mapping(compiled.scenario, actions).status, "invalid_mapping")

    def test_exact_declared_dram_peak_passes_minus_one_fails(self):
        baseline = replay_measured_ffn(self.compile())
        self.assertEqual(baseline.status, "valid")
        hardware = hardware_fixture()
        hardware["dram_capacity_bytes"] = baseline.peak_dram_bytes
        exact = replay_measured_ffn(self.compile(hardware=hardware))
        self.assertEqual(exact.status, "valid")
        hardware["dram_capacity_bytes"] -= 1
        limited = replay_measured_ffn(self.compile(hardware=hardware))
        self.assertEqual(limited.status, "invalid_mapping")
        self.assertIsNone(limited.makespan_ns)

    def test_boolean_hardware_rejected(self):
        for key in hardware_fixture():
            hardware = hardware_fixture()
            hardware[key] = True
            with self.subTest(key=key), self.assertRaises(InvalidInput):
                self.compile(hardware=hardware)

    def test_boolean_native_scalar_rejected(self):
        for field in ("warmup", "samples", "seed", "threads_applied"):
            record = protocol_fixture()
            record["native"][field] = True
            with self.subTest(field=field), self.assertRaises(InvalidInput):
                self.compile(record)

    def test_boolean_port_dimensions_rejected(self):
        record = protocol_fixture()
        record["native"]["stages"]["router"]["ports"]["selected_scores"]["ne"][2] = True
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_failed_numeric_or_lineage_rejected(self):
        for mutation in ("numeric", "lineage"):
            record = protocol_fixture()
            if mutation == "numeric":
                record["native"]["stages"]["combine"]["reference_check"]["passed"] = False
            else:
                record["native"]["stages"]["combine"]["ports"]["expert_output.1"]["digest"] = "f" * 16
            with self.subTest(mutation=mutation), self.assertRaises(InvalidInput):
                self.compile(record)

    def test_boolean_signature_port_geometry_rejected(self):
        record = protocol_fixture()
        record["physical_signature"]["stage_ports"]["router"]["selected_scores"]["ne"][2] = True
        record["physical_contract_sha256"] = contract_hash(record["physical_signature"])
        with self.assertRaises(InvalidInput):
            self.compile(record)

    def test_malformed_binary_hash_reports_invalid_input(self):
        record = protocol_fixture()
        record["physical_signature"]["native_binary_sha256"] = "not_sha256"
        with self.assertRaises(InvalidInput):
            self.compile(record)


    def test_failed_unsupported_record_status_rejected(self):
        for status in ("failed", "unsupported", None):
            record = protocol_fixture()
            record["status"] = status
            with self.subTest(status=status), self.assertRaises(InvalidInput):
                self.compile(record)


def shared_protocol_fixture():
    """Fabricated v2 shared-arena evidence; no hardware or numeric proof."""
    record = protocol_fixture()
    case, native = record["case"], record["native"]
    case["sampling_mode"] = "interleaved_joint_context"
    native.update(schema_version="resident_moe_ffn_native_v2",
                  contract="ggml_cpu_resident_moe_ffn_f32_v2",
                  sampling_mode="interleaved_joint_context",
                  timing_scope="interleaved_materialized_stages_and_joint_on_shared_resident_graph")
    shared_logical = stage_memory(case, "joint")["resident_logical_bytes"]
    shared_allocated = ((shared_logical + 63) // 64) * 64
    for index, value in enumerate(native["stages"].values()):
        value["workspace_bytes"] = 320
        value["workspace_source"] = "shared_ggml_gallocr_internal_buffer_plus_cpu_plan_arena"
        value["memory_accounting"].update(
            resident_allocated_bytes=shared_allocated,
            shared_resident_logical_bytes=shared_logical,
            shared_resident_allocated_bytes=shared_allocated,
            internal_buffer_bytes=256, cpu_graph_plan_work_bytes=64,
            stage_cpu_graph_plan_required_bytes=32 + 4 * index,
            shared_allocator=True)
    record["physical_signature"] = physical_signature(case, "a" * 64)
    record["physical_contract_sha256"] = contract_hash(record["physical_signature"])
    return record


class SharedMeasuredMoEFFNTests(unittest.TestCase):
    def test_shared_resident_is_not_added_again_to_each_workspace(self):
        record = shared_protocol_fixture()
        compiled = compile_measured_ffn(record, hardware_fixture())
        self.assertEqual(compiled.bindings["contract"], "ggml_cpu_resident_moe_ffn_f32_v2")
        self.assertTrue(compiled.bindings["context_sensitive_prices"])
        self.assertFalse(compiled.bindings["physical_capacity_verified"])
        for stage, devices in compiled.scenario_document["costs"]["compute"].items():
            memory = record["native"]["stages"][stage]["memory_accounting"]
            self.assertEqual(devices["cpu"]["workspace_bytes"], 320)
            self.assertEqual(devices["cpu"]["workspace_bytes"],
                             memory["internal_buffer_bytes"] + memory["cpu_graph_plan_work_bytes"])
            self.assertNotEqual(devices["cpu"]["workspace_bytes"],
                                320 + memory["shared_resident_allocated_bytes"])
        self.assertEqual(sum(tensor["storage_bytes"] for tensor in compiled.workload_document["tensors"]),
                         stage_memory(record["case"], "joint")["resident_logical_bytes"])

    def test_v2_signature_binds_mode_scope_and_shared_context(self):
        record = shared_protocol_fixture()
        signature = record["physical_signature"]
        self.assertEqual(signature["sampling_mode"], "interleaved_joint_context")
        self.assertEqual(signature["workspace_context"], "shared_joint_allocator_and_common_CPU_plan_arena")
        self.assertNotEqual(record["physical_contract_sha256"], protocol_fixture()["physical_contract_sha256"])
        for field, value in (("sampling_mode", "isolated_warm"),
                             ("timing_scope", TIMING_SCOPE), ("contract", CONTRACT)):
            damaged = copy.deepcopy(record)
            damaged["native"][field] = value
            with self.subTest(field=field), self.assertRaises(InvalidInput):
                compile_measured_ffn(damaged, hardware_fixture())

    def test_v2_shared_integer_and_boolean_guards_are_strict(self):
        for field, values in (("shared_allocator", (1, 0, "true", None)),
                              ("shared_resident_logical_bytes", (True, None, -1)),
                              ("shared_resident_allocated_bytes", (True, None, -1)),
                              ("stage_cpu_graph_plan_required_bytes", (True, None, -1))):
            for value in values:
                record = shared_protocol_fixture()
                record["native"]["stages"]["expert.0"]["memory_accounting"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(InvalidInput):
                    compile_measured_ffn(record, hardware_fixture())

    def test_shared_context_must_be_same_arena_and_large_enough_for_each_plan(self):
        record = shared_protocol_fixture()
        stage = record["native"]["stages"]["expert.0"]
        stage["memory_accounting"]["internal_buffer_bytes"] += 16
        stage["workspace_bytes"] += 16
        with self.assertRaises(InvalidInput):
            compile_measured_ffn(record, hardware_fixture())
        record = shared_protocol_fixture()
        record["native"]["stages"]["expert.0"]["memory_accounting"]["stage_cpu_graph_plan_required_bytes"] = 65
        with self.assertRaises(InvalidInput):
            compile_measured_ffn(record, hardware_fixture())

    def test_v2_locks_measured_expert_order_while_v1_keeps_parallel_dag(self):
        actions = tuple(action for stage in ("router", "expert.1", "expert.0", "expert.2", "combine")
                        for action in (Action(ACTION_COMPUTE, stage, device="cpu"), Action(ACTION_ADVANCE)))
        shared = compile_measured_ffn(shared_protocol_fixture(), hardware_fixture())
        self.assertEqual(shared.bindings["cost_execution_order"], "fixed_serial_slice_order")
        self.assertEqual([op["id"] for op in shared.workload_document["operations"]],
                         ["router", "expert.0", "expert.1", "expert.2", "combine"])
        self.assertFalse(shared.bindings["control_edges_are_numeric_port_reads"])
        result = evaluate_mapping(shared.scenario, actions)
        self.assertEqual(result.status, "invalid_mapping")
        self.assertEqual(replay_measured_ffn(shared).status, "valid")
        isolated = compile_measured_ffn(protocol_fixture(), hardware_fixture())
        reordered = evaluate_mapping(isolated.scenario, actions)
        self.assertEqual(reordered.status, "valid")
        self.assertEqual(reordered.makespan_ns, 1500)


if __name__ == "__main__":
    unittest.main()
