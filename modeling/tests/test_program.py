"""Hand-calculated program replay and explicit refusal boundaries (synthetic)."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from llm_infer_model.tensor.engine import (
    Action, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION, CODE_INPUT_NOT_READY,
    CODE_RESOURCE_BUSY, CODE_TENSOR_IN_USE, REASON_ACTIONS_REMAIN_AFTER_GOAL,
    STATUS_INCOMPLETE_MAPPING, STATUS_INFEASIBLE, STATUS_INVALID_MAPPING,
    STATUS_VALID, evaluate_mapping, initial_state, transition,
)
from llm_infer_model.tensor.program import (
    ActionStep, DeviceStateStep, WaitStep, execute_program, validate_device_state,
)
from llm_infer_model.tensor.spec import (
    COPY_READY, COPY_RESERVED_COPY, InvalidInput, M1Costs, Unsupported,
    build_scenario, scenario_fingerprint,
)


def documents(*, fork=False, m0=False):
    version = "0.1" if m0 else "0.2"
    tensors = []
    for tensor_id, role, initial in (
        ("x", "input", ["dram"]), ("h", "intermediate", []), ("y", "output", []),
    ):
        tensors.append({
            "id": tensor_id, "name": tensor_id, "role": role,
            "dtype": "GGML_TYPE_F32", "ne": [4, 1, 1, 1],
            "nb": [4, 16, 16, 16], "storage_bytes": 16,
            "initial_locations": initial,
        })
    workload = {
        "schema_version": version, "id": "program-chain",
        "origin": {"kind": "synthetic_fixture", "ggml_version": None, "sample": None},
        "tensors": tensors,
        "operations": [
            {"id": "c1", "semantic_op": "RELU", "ggml_op": "GGML_OP_UNARY",
             "unary_op": "GGML_UNARY_OP_RELU", "op_params": [],
             "inputs": ["x"], "output": "h"},
            {"id": "c2", "semantic_op": "RELU", "ggml_op": "GGML_OP_UNARY",
             "unary_op": "GGML_UNARY_OP_RELU", "op_params": [],
             "inputs": ["x" if fork else "h"], "output": "y"},
        ],
        "outputs": ["h", "y"] if fork else ["y"],
    }
    mapper = {"algorithm": "uniform_cost", "max_expanded_states": 0, "wall_time_limit_s": 0}
    if m0:
        return workload, {
            "schema_version": version, "id": "program-scene",
            "architecture": {"dram_capacity_bytes": None, "vram_capacity_bytes": 128,
                             "runtime_reserved_bytes": 0, "allocation_alignment_bytes": 1,
                             "gpu_compute_slots": 1, "h2d_copy_slots": 1},
            "costs": {"source": "synthetic_program", "compute": {
                "c1": {"duration_ns": 6, "workspace_bytes": 0},
                "c2": {"duration_ns": 8, "workspace_bytes": 0}},
                "h2d": {"x": {"duration_ns": 2}}},
            "mapspace": {"compute_device": "gpu", "copy_tensor_ids": ["x"],
                         "allow_copy_compute_overlap": True, "allow_eviction": True,
                         "allow_recomputation": False}, "mapper": mapper,
        }
    allowed = {"dram_to_vram": ["x", "h", "y"],
               "vram_to_dram": ["x", "h", "y"], "dram_to_ssd": ["y"]}
    scene = {
        "schema_version": version, "id": "program-scene",
        "architecture": {"dram_capacity_bytes": 128, "vram_capacity_bytes": 128,
                         "dram_runtime_reserved_bytes": 0, "vram_runtime_reserved_bytes": 0,
                         "dram_alignment_bytes": 1, "vram_alignment_bytes": 1,
                         "cpu_compute_slots": 1, "gpu_compute_slots": 1,
                         "pcie_copy_slots": 1, "ssd_io_slots": 1},
        "goal": {"output_locations": {tid: "dram" for tid in workload["outputs"]}},
        "costs": {"source": "synthetic_program", "compute": {
            "c1": {"cpu": {"duration_ns": 10, "workspace_bytes": 0},
                   "gpu": {"duration_ns": 6, "workspace_bytes": 0}},
            "c2": {"cpu": {"duration_ns": 20, "workspace_bytes": 0},
                   "gpu": {"duration_ns": 8, "workspace_bytes": 0}}},
            "transfer": {route: {tid: {"duration_ns": duration} for tid in ids}
                         for (route, ids), duration in zip(allowed.items(), (2, 3, 4))}},
        "mapspace": {"allowed_devices": {"c1": ["cpu", "gpu"], "c2": ["cpu", "gpu"]},
                     "allowed_transfers": allowed, "allow_eviction": True,
                     "allow_copy_compute_overlap": True, "allow_cpu_gpu_overlap": True,
                     "allow_recomputation": False}, "mapper": mapper,
    }
    return workload, scene


def scenario(**kwargs):
    return build_scenario(*documents(**kwargs))


def compute(operation, device="cpu"):
    return Action("COMPUTE", operation, device=device)


class StaticProgramEquivalence(unittest.TestCase):
    def test_m1_action_only_success_failure_incomplete_and_dead_tail_match(self):
        scene = scenario()
        valid = (compute("c1"), Action("ADVANCE"), compute("c2"), Action("ADVANCE"))
        for actions in (valid, valid + (Action("ADVANCE"),), valid[:2],
                        (compute("c2"),), (Action("ADVANCE"),)):
            with self.subTest(actions=actions):
                actual = execute_program(scene, tuple(ActionStep(a, i + 7) for i, a in enumerate(actions)))
                self.assertEqual(actual.evaluation, evaluate_mapping(scene, actions))
                self.assertEqual(actual.action_sources, tuple(range(7, 7 + len(actions))))
                self.assertEqual(actual.action_profiles, (scenario_fingerprint(scene),) * len(actions))
                state = initial_state(scene)
                for action in actions[:actual.evaluation.action_count]:
                    state = transition(scene, state, action)
                self.assertEqual(actual.final_state, state)

    def test_m0_actions_and_explicit_copy_wait_keep_historical_result(self):
        scene = scenario(m0=True)
        actions = (Action("COPY_H2D", "x"), Action("ADVANCE"),
                   Action("COMPUTE", "c1"), Action("ADVANCE"),
                   Action("COMPUTE", "c2"), Action("ADVANCE"))
        steps = (ActionStep(actions[0], 0), WaitStep(copies=(("x", "vram"),), source_index=1),
                 ActionStep(actions[2], 2), WaitStep(operations=("c1",), source_index=3),
                 ActionStep(actions[4], 4), WaitStep(drain=True, source_index=5))
        actual = execute_program(scene, steps)
        self.assertEqual(actual.evaluation, evaluate_mapping(scene, actions))
        self.assertEqual(actual.evaluation.makespan_ns, 16)

    def test_initial_overflow_matches_evaluator_without_fabricating_state(self):
        workload, document = documents()
        document["architecture"]["dram_capacity_bytes"] = 8
        scene = build_scenario(workload, document)
        actions = (compute("c1"),)
        actual = execute_program(scene, (ActionStep(actions[0], 4),))
        self.assertEqual(actual.evaluation, evaluate_mapping(scene, actions))
        self.assertEqual(actual.evaluation.status, STATUS_INFEASIBLE)
        self.assertIsNone(actual.final_state)
        self.assertEqual(actual.action_sources, ())

    def test_supplied_start_retains_absolute_clock_and_inflight_state(self):
        scene = scenario()
        start = replace(transition(scene, initial_state(scene), compute("c1")), t=100)
        result = execute_program(scene, (WaitStep(operations=("c1",), source_index=4),
                                        ActionStep(compute("c2"), 5), WaitStep(drain=True, source_index=6)),
                                 start=start)
        self.assertEqual(result.evaluation.makespan_ns, 130)
        self.assertEqual(result.evaluation.states[0].t, 100)
        self.assertEqual(result.final_state.copy_at("x", "dram"), COPY_READY)


class ExplicitWaits(unittest.TestCase):
    def test_wait_for_longer_task_expands_two_nearest_completions_with_same_source(self):
        scene = scenario(fork=True)
        start = transition(scene, initial_state(scene), Action("COPY", "x", src="dram", dst="vram"))
        start = transition(scene, start, Action("ADVANCE"))
        result = execute_program(scene, (
            ActionStep(compute("c1", "gpu"), 3), ActionStep(compute("c2"), 4),
            WaitStep(operations=("c2",), source_index=5)), start)
        self.assertEqual(result.final_state.t, 22)
        self.assertEqual(tuple(event.end_ns for event in result.evaluation.events[-2:]), (8, 22))
        self.assertEqual(result.action_sources, (3, 4, 5, 5))
        self.assertEqual(result.evaluation, evaluate_mapping(scene, result.evaluation.actions, start))

    def test_specific_wait_advances_nearest_events_without_draining_unrelated_gpu(self):
        scene = scenario(fork=True)
        start = transition(scene, initial_state(scene), Action("COPY", "x", src="dram", dst="vram"))
        start = transition(scene, start, Action("ADVANCE"))
        steps = (ActionStep(compute("c1", "gpu"), 3), ActionStep(compute("c2"), 4),
                 WaitStep(operations=("c1",), source_index=5))
        actual = execute_program(scene, steps, start=start)
        self.assertEqual(actual.evaluation.status, STATUS_INCOMPLETE_MAPPING)
        self.assertEqual(actual.final_state.t, 8)
        self.assertEqual(actual.final_state.op("c1"), "DONE")
        self.assertEqual(actual.final_state.op("c2"), "RUNNING")
        self.assertEqual(actual.action_sources, (3, 4, 5))
        self.assertEqual(actual.final_state.running[0].remaining_ns, 14)

    def test_wait_for_copy_can_finish_compute_output_but_not_other_location(self):
        scene = scenario()
        okay = execute_program(scene, (ActionStep(compute("c1"), 0),
                                       WaitStep(copies=(("h", "dram"),), source_index=1)))
        self.assertEqual(okay.final_state.t, 10)
        self.assertEqual(okay.final_state.copy_at("h", "dram"), COPY_READY)
        failed = execute_program(scene, (ActionStep(compute("c1"), 0),
                                         WaitStep(copies=(("h", "vram"),), source_index=12)))
        self.assertEqual(failed.evaluation.error_code, CODE_INPUT_NOT_READY)
        self.assertEqual(failed.final_state.t, 0)
        self.assertEqual(failed.error_source_index, 12)
        self.assertEqual(failed.evaluation.action_count, 1)

    def test_wait_before_start_unknown_targets_and_absent_copy_are_refused(self):
        scene = scenario()
        for wait, code in (
            (WaitStep(operations=("c1",), source_index=8), CODE_INPUT_NOT_READY),
            (WaitStep(operations=("missing",), source_index=8), CODE_ILLEGAL_ACTION),
            (WaitStep(copies=(("h", "dram"),), source_index=8), CODE_INPUT_NOT_READY),
            (WaitStep(copies=(("missing", "dram"),), source_index=8), CODE_ILLEGAL_ACTION),
        ):
            with self.subTest(wait=wait):
                result = execute_program(scene, (wait,))
                self.assertEqual(result.evaluation.error_code, code)
                self.assertEqual(result.evaluation.actions, ())
                self.assertEqual(result.final_state.t, 0)
                self.assertEqual(result.error_source_index, 8)

    def test_all_wait_targets_are_checked_before_advancing(self):
        scene = scenario()
        result = execute_program(scene, (ActionStep(compute("c1"), 1),
                                        WaitStep(operations=("c1", "c2"), source_index=2)))
        self.assertEqual(result.evaluation.error_code, CODE_INPUT_NOT_READY)
        self.assertEqual(result.final_state.t, 0)
        self.assertEqual(result.final_state.op("c1"), "RUNNING")

    def test_copy_reservation_without_running_producer_is_refused(self):
        scene = scenario()
        start = initial_state(scene)
        statuses = dict(start.copy_status)
        statuses["x"] = COPY_RESERVED_COPY
        start = replace(start, copy_status=tuple(sorted(statuses.items())))
        result = execute_program(scene, (WaitStep(copies=(("x", "vram"),), source_index=9),), start)
        self.assertEqual(result.evaluation.error_code, CODE_INPUT_NOT_READY)

    def test_ssd_wait_tracks_write_completion_and_exact_bytes(self):
        workload, document = documents()
        document["goal"]["output_locations"]["y"] = "ssd"
        scene = build_scenario(workload, document)
        steps = (ActionStep(compute("c1"), 0), WaitStep(drain=True, source_index=1),
                 ActionStep(compute("c2"), 2), WaitStep(drain=True, source_index=3),
                 ActionStep(Action("WRITE_SSD", "y"), 4),
                 WaitStep(copies=(("y", "ssd"),), source_index=5))
        result = execute_program(scene, steps)
        self.assertEqual(result.evaluation.status, STATUS_VALID)
        self.assertEqual(result.evaluation.makespan_ns, 34)
        self.assertEqual(result.evaluation.ssd_write_bytes, 16)
        self.assertEqual(result.final_state.copy_at("y", "ssd"), COPY_READY)

    def test_goal_rejects_any_extra_action_wait_or_environment(self):
        scene = scenario()
        prefix = (ActionStep(compute("c1"), 0), WaitStep(drain=True, source_index=1),
                  ActionStep(compute("c2"), 2), WaitStep(drain=True, source_index=3))
        for tail in (ActionStep(Action("ADVANCE"), 9), WaitStep(drain=True, source_index=9),
                     DeviceStateStep("same", scene, 9)):
            with self.subTest(tail=tail):
                result = execute_program(scene, prefix + (tail,))
                self.assertEqual(result.evaluation.status, STATUS_INVALID_MAPPING)
                self.assertTrue(result.evaluation.reason.startswith(REASON_ACTIONS_REMAIN_AFTER_GOAL))
                self.assertEqual(result.error_source_index, 9)
                self.assertEqual(result.final_state.t, 30)
                self.assertEqual(result.environment_events, ())


class EnvironmentBoundaries(unittest.TestCase):
    def test_switched_transfer_price_is_used_for_the_next_actual_copy(self):
        workload, document = documents()
        scene = build_scenario(workload, document)
        changed_document = copy.deepcopy(document)
        changed_document["costs"]["source"] = "synthetic_slow_pcie"
        changed_document["costs"]["transfer"]["dram_to_vram"]["h"]["duration_ns"] = 11
        changed = build_scenario(workload, changed_document)
        result = execute_program(scene, (
            ActionStep(compute("c1"), 0), WaitStep(drain=True, source_index=1),
            DeviceStateStep("slow_pcie", changed, 2),
            ActionStep(Action("COPY", "h", src="dram", dst="vram"), 3),
            WaitStep(copies=(("h", "vram"),), source_index=4),
            ActionStep(compute("c2", "gpu"), 5), WaitStep(drain=True, source_index=6),
            ActionStep(Action("COPY", "y", src="vram", dst="dram"), 7),
            WaitStep(drain=True, source_index=8),
        ))
        self.assertEqual(result.evaluation.status, STATUS_VALID)
        self.assertEqual(result.evaluation.makespan_ns, 32)  # 10 + 11 + 8 + 3
        self.assertEqual((result.evaluation.h2d_bytes, result.evaluation.d2h_bytes), (16, 16))
        copy_event = result.evaluation.events[2]
        self.assertEqual((copy_event.start_ns, copy_event.end_ns), (10, 21))
        self.assertEqual(result.action_profiles[2], scenario_fingerprint(changed))
        self.assertEqual(result.final_state.copy_at("h", "dram"), COPY_READY)
        self.assertEqual(result.final_state.copy_at("h", "vram"), COPY_READY)

    def test_new_complete_cost_profile_preserves_copies_clock_and_action_provenance(self):
        workload, document = documents()
        scene = build_scenario(workload, document)
        updated = copy.deepcopy(document)
        updated["costs"]["source"] = "synthetic_slow_cpu"
        updated["costs"]["compute"]["c2"]["cpu"]["duration_ns"] = 50
        updated["architecture"]["dram_runtime_reserved_bytes"] = 7
        changed = build_scenario(workload, updated)
        steps = (ActionStep(compute("c1"), 0), WaitStep(operations=("c1",), source_index=1),
                 DeviceStateStep("slow", changed, 2), ActionStep(compute("c2"), 3),
                 WaitStep(drain=True, source_index=4))
        result = execute_program(scene, steps)
        self.assertEqual(result.evaluation.status, STATUS_VALID)
        self.assertEqual(result.evaluation.makespan_ns, 60)
        self.assertEqual(result.final_state.copy_at("h", "dram"), COPY_READY)
        self.assertEqual(result.final_state.copy_at("x", "dram"), COPY_READY)
        self.assertEqual(result.action_sources, (0, 1, 3, 4))
        self.assertEqual(result.action_profiles, (scenario_fingerprint(scene),) * 2
                         + (scenario_fingerprint(changed),) * 2)
        event, = result.environment_events
        self.assertEqual((event.t_ns, event.used_dram_bytes, event.used_vram_bytes), (10, 39, 0))
        self.assertEqual((event.previous_profile, event.profile),
                         (scenario_fingerprint(scene), scenario_fingerprint(changed)))
        self.assertEqual(result.evaluation.peak_dram_bytes, 55)
        self.assertEqual(result.evaluation.events[2].start_ns, 10)
        self.assertEqual(result.evaluation.events[2].end_ns, 60)
        self.assertEqual(len(result.evaluation.states), result.evaluation.action_count + 1)

    def test_busy_switch_refuses_without_repricing_or_resetting_task(self):
        scene = scenario()
        changed = replace(scene, architecture=replace(scene.architecture, dram_capacity_bytes=256))
        result = execute_program(scene, (ActionStep(compute("c1"), 0),
                                        DeviceStateStep("large", changed, 7)))
        self.assertEqual(result.evaluation.error_code, CODE_RESOURCE_BUSY)
        self.assertEqual(result.final_state.running[0].remaining_ns, 10)
        self.assertEqual(result.final_state.t, 0)
        self.assertEqual(result.environment_events, ())

    def test_capacity_shrink_uses_actual_inventory_and_never_implicit_eviction(self):
        scene = scenario()
        changed = replace(scene, architecture=replace(scene.architecture, dram_capacity_bytes=31))
        validate_device_state(scene, changed)
        prefix = (ActionStep(compute("c1"), 0), WaitStep(drain=True, source_index=1))
        result = execute_program(scene, prefix + (DeviceStateStep("too_small", changed, 8),))
        self.assertEqual(result.evaluation.error_code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(result.final_state.t, 10)
        self.assertEqual(result.final_state.copy_at("h", "dram"), COPY_READY)
        self.assertEqual(result.final_state.copy_at("x", "dram"), COPY_READY)
        self.assertEqual(result.environment_events, ())
        exact = replace(scene, architecture=replace(scene.architecture, dram_capacity_bytes=32))
        accepted = execute_program(scene, prefix + (DeviceStateStep("exact", exact, 8),))
        self.assertEqual(len(accepted.environment_events), 1)
        self.assertEqual(accepted.environment_events[0].used_dram_bytes, 32)

    def test_actual_eviction_allows_profile_smaller_than_original_initial_layout(self):
        # At a later boundary the original input can genuinely be gone. Static
        # profile validation must not re-run initial_capacity on that old layout.
        scene = scenario()
        prefix = (ActionStep(compute("c1"), 0), WaitStep(drain=True, source_index=1),
                  ActionStep(Action("EVICT", "x", location="dram"), 2))
        changed = replace(scene, architecture=replace(scene.architecture, dram_capacity_bytes=16))
        result = execute_program(scene, prefix + (DeviceStateStep("trimmed", changed, 3),))
        self.assertEqual(result.environment_events[0].used_dram_bytes, 16)
        self.assertEqual(result.final_state.copy_at("x", "dram"), "ABSENT")

    def test_vram_shrink_and_runtime_reserve_overflow_are_checked(self):
        scene = scenario()
        prefix = (ActionStep(Action("COPY", "x", src="dram", dst="vram"), 0),
                  WaitStep(drain=True, source_index=1))
        for arch in (replace(scene.architecture, vram_capacity_bytes=15),
                     replace(scene.architecture, vram_capacity_bytes=16, vram_runtime_reserved_bytes=1)):
            result = execute_program(scene, prefix + (DeviceStateStep("small", replace(scene, architecture=arch), 2),))
            self.assertEqual(result.evaluation.error_code, CODE_CAPACITY_EXCEEDED)
            self.assertEqual(result.final_state.copy_at("x", "vram"), COPY_READY)
            self.assertEqual(result.final_state.t, 2)

    def test_switch_retains_read_lock_capacity_and_workspace_rules_from_engine(self):
        scene = scenario()
        workload, doc = documents()
        doc["costs"]["compute"]["c1"]["cpu"]["workspace_bytes"] = 16
        doc["architecture"]["dram_capacity_bytes"] = 48
        changed = build_scenario(workload, doc)
        result = execute_program(scene, (DeviceStateStep("scratch", changed, 0),
                                        ActionStep(compute("c1"), 1),
                                        ActionStep(Action("EVICT", "x", location="dram"), 2)))
        self.assertEqual(result.evaluation.error_code, CODE_TENSOR_IN_USE)
        self.assertEqual(result.evaluation.peak_dram_bytes, 48)
        self.assertEqual(result.evaluation.events[0].workspace_bytes, 16)
        self.assertEqual(result.evaluation.events[0].end_ns, 10)

    def test_profile_graph_goal_mapspace_layout_slots_format_and_mapper_drift_refuse(self):
        scene = scenario()
        candidates = [
            replace(scene, id="another"),
            replace(scene, goal=replace(scene.goal, output_locations={"y": "vram"})),
            replace(scene, mapspace=replace(scene.mapspace, allow_copy_compute_overlap=False)),
            replace(scene, architecture=replace(scene.architecture, dram_alignment_bytes=2)),
            replace(scene, architecture=replace(scene.architecture, allocation_alignment_bytes=2)),
            replace(scene, mapper=replace(scene.mapper, max_expanded_states=1)),
        ]
        workload, document = documents()
        workload["tensors"][0]["name"] = "different-input"
        candidates.append(build_scenario(workload, document))
        candidates.append(scenario(m0=True))
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with self.assertRaises((InvalidInput, Unsupported)):
                    validate_device_state(scene, candidate)
        with self.assertRaises(Unsupported):
            replace(scene.architecture, pcie_copy_slots=2)

    def test_profile_cost_completeness_and_bool_capacity_are_not_guessed(self):
        scene = scenario()
        missing = replace(scene, costs=M1Costs("missing", {"c1": scene.costs.compute["c1"]},
                                              scene.costs.transfer))
        with self.assertRaises(InvalidInput):
            validate_device_state(scene, missing)
        boolean = replace(scene, architecture=replace(scene.architecture, dram_capacity_bytes=True))
        with self.assertRaises(InvalidInput):
            validate_device_state(scene, boolean)
        bad_device = replace(scene, costs=M1Costs("extra", {
            **scene.costs.compute, "c1": {**scene.costs.compute["c1"], "npu": scene.costs.compute["c1"]["cpu"]}},
            scene.costs.transfer))
        with self.assertRaises(InvalidInput):
            validate_device_state(scene, bad_device)
        # A dict that resembles the JSON price entry is not a runtime Cost:
        # accepting it here would validate the serialized profile but later
        # crash the engine on cost.duration_ns.
        raw_cost = replace(scene, costs=M1Costs("wrong-runtime-type", {
            **scene.costs.compute, "c1": {**scene.costs.compute["c1"],
                                         "cpu": {"duration_ns": 10, "workspace_bytes": 0}}},
            scene.costs.transfer))
        with self.assertRaises(InvalidInput):
            validate_device_state(scene, raw_cost)
        bad_mapspace = replace(scene, mapspace=replace(scene.mapspace, allowed_devices={
            "c1": None, "c2": ("cpu", "gpu")}))
        with self.assertRaises(InvalidInput):
            validate_device_state(scene, bad_mapspace)

    def test_malformed_wait_and_source_indices_are_rejected_early(self):
        for make in (
            lambda: ActionStep(compute("c1"), True), lambda: WaitStep(),
            lambda: WaitStep(drain=1), lambda: WaitStep(operations=("c1", "c1")),
            lambda: WaitStep(copies=(("x", "unknown"),)),
            lambda: DeviceStateStep("", scenario()),
        ):
            with self.subTest(make=make):
                with self.assertRaises(InvalidInput):
                    make()


if __name__ == "__main__":
    unittest.main()
