"""Causal expert transfers reuse every M1 route and leave resident copies alone."""
from __future__ import annotations

import dataclasses
import unittest

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_READ_SSD,
    ACTION_WRITE_SSD, Action, MappingError,
    CODE_TRANSFER_DEPENDENCY_NOT_READY, evaluate_mapping,
    initial_state, legal_actions, transition, used_bytes,
)
from llm_infer_model.tensor.spec import (
    COPY_READY, OP_DONE, OP_RUNNING, InvalidInput, Unsupported,
    build_scenario, scenario_fingerprint,
)
from llm_infer_model.tensor.spec import _canonical_scenario_m1
from tests.support import m1_scenario


ROUTES = ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")


def documents(*, route="dram_to_vram", provenance="synthetic_layer", gates=True,
              resident=False):
    source = route.split("_to_")[0]
    locations = [source]
    if resident:
        locations = ["dram", "vram"]
    workload = {
        "schema_version": "0.2", "id": "causal-expert-transfer",
        "origin": {"kind": provenance},
        "tensors": [
            {"id": "W", "name": "expert.weight", "role": "weight",
             "dtype": "GGML_TYPE_F32", "ne": [4, 4, 1, 1],
             "nb": [4, 16, 64, 64], "storage_bytes": 64,
             "initial_locations": locations},
            {"id": "x", "name": "input", "role": "input",
             "dtype": "GGML_TYPE_F32", "ne": [4, 1, 1, 1],
             "nb": [4, 16, 16, 16], "storage_bytes": 16,
             "initial_locations": ["dram"]},
            {"id": "r", "name": "route", "role": "intermediate",
             "dtype": "GGML_TYPE_F32", "ne": [4, 1, 1, 1],
             "nb": [4, 16, 16, 16], "storage_bytes": 16},
            {"id": "y", "name": "result", "role": "output",
             "dtype": "GGML_TYPE_F32", "ne": [4, 1, 1, 1],
             "nb": [4, 16, 16, 16], "storage_bytes": 16},
        ],
        "operations": [
            {"id": "router", "semantic_op": "RELU",
             "ggml_op": "GGML_OP_UNARY", "unary_op": "GGML_UNARY_OP_RELU",
             "op_params": [], "inputs": ["x"], "output": "r"},
            {"id": "expert", "semantic_op": "MUL_MAT",
             "ggml_op": "GGML_OP_MUL_MAT", "unary_op": None,
             "op_params": [], "inputs": ["W", "r"], "output": "y"},
        ], "outputs": ["y"],
    }
    scenario = {
        "schema_version": "0.2", "id": "causal-expert-transfer",
        "architecture": {"dram_capacity_bytes": 256, "vram_capacity_bytes": 256,
                         "dram_runtime_reserved_bytes": 0,
                         "vram_runtime_reserved_bytes": 0,
                         "dram_alignment_bytes": 1, "vram_alignment_bytes": 1,
                         "cpu_compute_slots": 1, "gpu_compute_slots": 1,
                         "pcie_copy_slots": 1, "ssd_io_slots": 1},
        "goal": {"output_locations": {"y": "dram"}},
        "costs": {"source": "synthetic_fixed",
                  "compute": {"router": {"cpu": {"duration_ns": 10}},
                              "expert": {"cpu": {"duration_ns": 20},
                                         "gpu": {"duration_ns": 15}}},
                  "transfer": {route: {"W": {"duration_ns": 3}}}},
        "mapspace": {"allowed_devices": {"router": ["cpu"],
                                          "expert": ["cpu", "gpu"]},
                     "allowed_transfers": {route: ["W"]},
                     "allow_eviction": True, "allow_copy_compute_overlap": True,
                     "allow_cpu_gpu_overlap": True, "allow_recomputation": False},
        "mapper": {"algorithm": "uniform_cost", "max_expanded_states": 0,
                   "wall_time_limit_s": 0},
    }
    if gates:
        scenario["mapspace"]["transfer_dependencies"] = {"W": ["router"]}
    return workload, scenario


def make_scenario(**kwargs):
    return build_scenario(*documents(**kwargs))


def move(route):
    if route == "ssd_to_dram":
        return Action(ACTION_READ_SSD, "W")
    if route == "dram_to_ssd":
        return Action(ACTION_WRITE_SSD, "W")
    source, destination = route.split("_to_")
    return Action(ACTION_COPY, "W", src=source, dst=destination)


def finish_router(scenario):
    state = transition(scenario, initial_state(scenario),
                       Action(ACTION_COMPUTE, "router", device="cpu"))
    return transition(scenario, state, Action(ACTION_ADVANCE))


class TransferDependencyGates(unittest.TestCase):
    def assert_waits(self, scenario, state, action):
        with self.assertRaises(MappingError) as raised:
            transition(scenario, state, action)
        self.assertEqual(raised.exception.code, CODE_TRANSFER_DEPENDENCY_NOT_READY)
        self.assertEqual(raised.exception.operation_id, "router")
        self.assertEqual(raised.exception.tensor_id, "W")
        self.assertNotIn(action, legal_actions(scenario, state))

    def test_all_routes_wait_before_router_starts(self):
        for route in ROUTES:
            with self.subTest(route=route):
                scenario = make_scenario(route=route)
                self.assert_waits(scenario, initial_state(scenario), move(route))

    def test_all_routes_wait_while_router_is_running(self):
        for route in ROUTES:
            with self.subTest(route=route):
                scenario = make_scenario(route=route)
                state = transition(scenario, initial_state(scenario),
                                   Action(ACTION_COMPUTE, "router", device="cpu"))
                self.assertEqual(state.op("router"), OP_RUNNING)
                self.assert_waits(scenario, state, move(route))

    def test_all_routes_unlock_only_after_router_completion(self):
        for route in ROUTES:
            with self.subTest(route=route):
                scenario = make_scenario(route=route)
                state = finish_router(scenario)
                self.assertEqual(state.op("router"), OP_DONE)
                self.assertIn(move(route), legal_actions(scenario, state))
                transferred = transition(scenario, state, move(route))
                self.assertEqual(transferred.t, 10)
                self.assertEqual(transferred.running[0].remaining_ns, 3)

    def test_ssd_read_reserves_dram_only_when_started(self):
        scenario = make_scenario(route="ssd_to_dram")
        state = finish_router(scenario)
        before = used_bytes(scenario, state, "dram")
        state = transition(scenario, state, move("ssd_to_dram"))
        self.assertEqual(used_bytes(scenario, state, "dram") - before, 64)
        state = transition(scenario, state, Action(ACTION_ADVANCE))
        self.assertEqual(state.copy_at("W", "dram"), COPY_READY)

    def test_already_resident_expert_needs_no_transfer(self):
        scenario = make_scenario(resident=True)
        state = finish_router(scenario)
        action = Action(ACTION_COMPUTE, "expert", device="cpu")
        self.assertIn(action, legal_actions(scenario, state))
        state = transition(scenario, state, action)
        state = transition(scenario, state, Action(ACTION_ADVANCE))
        self.assertEqual(state.op("expert"), OP_DONE)

    def test_evaluation_records_early_transfer_refusal(self):
        scenario = make_scenario()
        result = evaluate_mapping(scenario, [move("dram_to_vram")])
        self.assertEqual(result.error_code, CODE_TRANSFER_DEPENDENCY_NOT_READY)
        self.assertEqual(result.error_action_index, 0)
        self.assertEqual(result.h2d_bytes, 0)

    def test_empty_gates_preserve_prefetch_before_router(self):
        scenario = make_scenario(gates=False)
        self.assertIn(move("dram_to_vram"), legal_actions(scenario, initial_state(scenario)))

    def test_every_declared_router_dependency_must_finish(self):
        workload, document = documents()
        routed = dict(workload["tensors"][2], id="r2", name="second.route")
        workload["tensors"].insert(3, routed)
        second = dict(workload["operations"][0], id="router2", inputs=["r"], output="r2")
        workload["operations"].insert(1, second)
        workload["operations"][-1]["inputs"] = ["W", "r2"]
        document["mapspace"]["allowed_devices"]["router2"] = ["cpu"]
        document["costs"]["compute"]["router2"] = {"cpu": {"duration_ns": 7}}
        document["mapspace"]["transfer_dependencies"]["W"] = ["router", "router2"]
        scenario = build_scenario(workload, document)
        state = finish_router(scenario)
        with self.assertRaises(MappingError) as raised:
            transition(scenario, state, move("dram_to_vram"))
        self.assertEqual(raised.exception.code, CODE_TRANSFER_DEPENDENCY_NOT_READY)
        self.assertEqual(raised.exception.operation_id, "router2")
        state = transition(scenario, state, Action(ACTION_COMPUTE, "router2", device="cpu"))
        with self.assertRaises(MappingError) as raised:
            transition(scenario, state, move("dram_to_vram"))
        self.assertEqual(raised.exception.code, CODE_TRANSFER_DEPENDENCY_NOT_READY)
        state = transition(scenario, state, Action(ACTION_ADVANCE))
        self.assertIn(move("dram_to_vram"), legal_actions(scenario, state))

    def test_measured_macro_provenance_allows_gates(self):
        scenario = make_scenario(provenance="measured_macro")
        self.assertEqual(scenario.mapspace.transfer_dependencies, {"W": ("router",)})


class TransferDependencyValidation(unittest.TestCase):
    def test_unknown_tensor_is_rejected(self):
        workload, document = documents()
        document["mapspace"]["transfer_dependencies"] = {"missing": ["router"]}
        with self.assertRaisesRegex(InvalidInput, "unknown tensor"):
            build_scenario(workload, document)

    def test_unknown_operation_is_rejected(self):
        workload, document = documents()
        document["mapspace"]["transfer_dependencies"] = {"W": ["missing"]}
        with self.assertRaisesRegex(InvalidInput, "unknown operation"):
            build_scenario(workload, document)

    def test_duplicate_operations_are_rejected(self):
        workload, document = documents()
        document["mapspace"]["transfer_dependencies"] = {"W": ["router", "router"]}
        with self.assertRaisesRegex(InvalidInput, "twice"):
            build_scenario(workload, document)

    def test_json_nonobject_dependencies_are_rejected(self):
        for invalid in (None, [], "router"):
            with self.subTest(invalid=invalid):
                workload, document = documents()
                document["mapspace"]["transfer_dependencies"] = invalid
                with self.assertRaises(InvalidInput):
                    build_scenario(workload, document)

    def test_json_dependencies_must_be_lists(self):
        for invalid in (None, "router", {"router": True}):
            with self.subTest(invalid=invalid):
                workload, document = documents()
                document["mapspace"]["transfer_dependencies"] = {"W": invalid}
                with self.assertRaises(InvalidInput):
                    build_scenario(workload, document)

    def test_json_operation_ids_must_be_strings(self):
        workload, document = documents()
        document["mapspace"]["transfer_dependencies"] = {"W": [False]}
        with self.assertRaises(InvalidInput):
            build_scenario(workload, document)

    def test_ordinary_m1_workloads_cannot_gain_route_gates(self):
        for provenance in ("synthetic_fixture", "ggml"):
            with self.subTest(provenance=provenance):
                with self.assertRaisesRegex(Unsupported, "M2"):
                    make_scenario(provenance=provenance)

    def test_direct_mapspace_must_use_dict_of_tuples(self):
        scenario = make_scenario()
        for invalid in (None, [], {"W": ["router"]}, {"W": ("router", "router")},
                        {"W": (False,)}, {False: ("router",)}):
            with self.subTest(invalid=invalid), self.assertRaises(InvalidInput):
                dataclasses.replace(scenario.mapspace, transfer_dependencies=invalid)

    def test_direct_scenario_checks_references(self):
        scenario = make_scenario()
        for dependencies in ({"missing": ("router",)}, {"W": ("missing",)}):
            with self.subTest(dependencies=dependencies), self.assertRaises(InvalidInput):
                dataclasses.replace(scenario, mapspace=dataclasses.replace(
                    scenario.mapspace, transfer_dependencies=dependencies))

    def test_direct_scenario_checks_provenance_and_version(self):
        scenario = make_scenario()
        with self.assertRaises(Unsupported):
            dataclasses.replace(scenario, schema_version="0.1")
        workload = dataclasses.replace(scenario.workload, origin=dataclasses.replace(
            scenario.workload.origin, kind="synthetic_fixture"))
        with self.assertRaises(Unsupported):
            dataclasses.replace(scenario, workload=workload)


class TransferDependencyFingerprints(unittest.TestCase):
    def test_optional_empty_field_keeps_existing_fingerprint(self):
        workload, document = documents(gates=False)
        implicit = build_scenario(workload, document)
        document["mapspace"]["transfer_dependencies"] = {}
        explicit = build_scenario(workload, document)
        self.assertEqual(scenario_fingerprint(implicit), scenario_fingerprint(explicit))
        self.assertNotIn("transfer_dependencies", _canonical_scenario_m1(explicit)["mapspace"])

    def test_old_m1_canonical_payload_and_hash_have_no_new_field(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as scenario:
            self.assertNotIn("transfer_dependencies", _canonical_scenario_m1(scenario)["mapspace"])
            self.assertEqual(scenario_fingerprint(scenario), "0c27a927960f9c613cc97d169f370a013a4566b37c16fa2361c33e94086d6c04")

    def test_nonempty_gate_changes_scenario_fingerprint(self):
        gated = make_scenario()
        ungated = make_scenario(gates=False)
        self.assertNotEqual(scenario_fingerprint(gated), scenario_fingerprint(ungated))

    def test_dependency_order_does_not_change_fingerprint(self):
        scenario = make_scenario()
        one = dataclasses.replace(scenario, mapspace=dataclasses.replace(
            scenario.mapspace, transfer_dependencies={"W": ("router", "expert")}))
        two = dataclasses.replace(scenario, mapspace=dataclasses.replace(
            scenario.mapspace, transfer_dependencies={"W": ("expert", "router")}))
        self.assertEqual(scenario_fingerprint(one), scenario_fingerprint(two))


if __name__ == "__main__":
    unittest.main()
