"""Hand-calculated interference traces and current-information boundaries."""

from dataclasses import asdict, replace
import json
import unittest

from llm_infer_model.tensor.engine import (
    Action, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION,
    CODE_RESOURCE_BUSY, CODE_TENSOR_IN_USE,
    advance_elapsed, evaluate_mapping, initial_state, transition,
)
from llm_infer_model.tensor.environment import (
    CODE_ENVIRONMENT_DEADLOCK, CODE_EXTERNAL_MEMORY_CONFLICT, CODE_INFORMATION_NOT_KNOWN, CODE_NO_PROGRESS,
    EnvironmentEvent, SimulationSession,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT, COPY_RESERVED_OUTPUT, InvalidInput, OP_DONE, OP_RUNNING, RESOURCE_CPU_COMPUTE,
    build_scenario,
)


def scene(*, gpu_duration=6, capacity=128, workspace=16):
    tensors = [{"id": tid, "name": tid, "role": role, "dtype": "GGML_TYPE_F32",
                "ne": [4, 1, 1, 1], "nb": [4, 16, 16, 16], "storage_bytes": 16,
                "initial_locations": locations}
               for tid, role, locations in (("x", "input", ["dram", "vram"]),
                                            ("h", "output", []), ("y", "output", []))]
    workload = {"schema_version": "0.2", "id": "environment-fork",
                "origin": {"kind": "synthetic_fixture", "ggml_version": None, "sample": None},
                "tensors": tensors,
                "operations": [{"id": oid, "semantic_op": "RELU", "ggml_op": "GGML_OP_UNARY",
                                "unary_op": "GGML_UNARY_OP_RELU", "op_params": [],
                                "inputs": ["x"], "output": output}
                               for oid, output in (("cpu-op", "h"), ("gpu-op", "y"))],
                "outputs": ["h", "y"]}
    document = {"schema_version": "0.2", "id": "environment-scene",
                "architecture": {"dram_capacity_bytes": capacity, "vram_capacity_bytes": capacity,
                                 "dram_runtime_reserved_bytes": 0, "vram_runtime_reserved_bytes": 0,
                                 "dram_alignment_bytes": 1, "vram_alignment_bytes": 1,
                                 "cpu_compute_slots": 1, "gpu_compute_slots": 1,
                                 "pcie_copy_slots": 1, "ssd_io_slots": 1},
                "costs": {"source": "synthetic_environment", "compute": {
                    "cpu-op": {"cpu": {"duration_ns": 10, "workspace_bytes": workspace}},
                    "gpu-op": {"gpu": {"duration_ns": gpu_duration, "workspace_bytes": 0}}},
                    "transfer": {}},
                "mapspace": {"allowed_devices": {"cpu-op": ["cpu"], "gpu-op": ["gpu"]},
                             "allowed_transfers": {}, "allow_eviction": True,
                             "allow_cpu_gpu_overlap": True, "allow_copy_compute_overlap": True,
                             "allow_recomputation": False},
                "goal": {"output_locations": {"h": "dram", "y": "vram"}},
                "mapper": {"algorithm": "uniform_cost", "max_expanded_states": 0,
                           "wall_time_limit_s": 0}}
    return build_scenario(workload, document)


CPU = Action("COMPUTE", "cpu-op", device="cpu")
GPU = Action("COMPUTE", "gpu-op", device="gpu")


class SharedProgressTests(unittest.TestCase):
    def test_partial_progress_pauses_cpu_and_uses_existing_completion_rules(self):
        scenario = scene()
        state = transition(scenario, initial_state(scenario), CPU)
        state = transition(scenario, state, GPU)
        at_three = advance_elapsed(scenario, state, 3)
        self.assertEqual(at_three.task_on("cpu_compute").remaining_ns, 7)
        at_six = advance_elapsed(scenario, at_three, 3, (RESOURCE_CPU_COMPUTE,))
        self.assertEqual(at_six.op("gpu-op"), OP_DONE)
        self.assertEqual(at_six.task_on("cpu_compute").remaining_ns, 7)
        self.assertEqual(at_six.copy_at("h", "dram"), COPY_RESERVED_OUTPUT)

    def test_elapsed_cannot_skip_completion_and_rejects_unknown_resource_or_bool(self):
        scenario = scene()
        state = transition(scenario, initial_state(scenario), GPU)
        for elapsed, paused in ((7, ()), (True, ()), (-1, ()), (1, ("unknown",))):
            with self.subTest(elapsed=elapsed, paused=paused), self.assertRaises(InvalidInput):
                advance_elapsed(scenario, state, elapsed, paused)


class EnvironmentSessionTests(unittest.TestCase):
    def test_cpu_ten_service_units_with_three_to_seven_blackout_finishes_at_fourteen(self):
        runtime = SimulationSession(scene(), environment_events=(
            EnvironmentEvent(3, cpu_available=False), EnvironmentEvent(7, cpu_available=True)))
        self.assertTrue(runtime.submit(CPU).ok)
        self.assertTrue(runtime.submit(GPU).ok)
        self.assertIsNone(runtime.result().evaluation.events[0].end_ns)
        self.assertEqual(runtime.result().nominal_ends[0], 10)
        self.assertEqual(runtime.advance().t_ns, 3)
        self.assertFalse(runtime.observe().cpu_available)
        self.assertEqual(runtime.state.task_on("cpu_compute").remaining_ns, 7)
        dram = next(m for m in runtime.observe().memory if m.domain == "dram")
        self.assertEqual((dram.tensor_bytes, dram.workspace_bytes, dram.used_bytes), (32, 16, 48))
        self.assertEqual(runtime.advance().t_ns, 6)
        self.assertEqual(runtime.state.op("gpu-op"), OP_DONE)
        self.assertEqual(runtime.state.op("cpu-op"), OP_RUNNING)
        self.assertEqual(runtime.advance().t_ns, 7)
        self.assertTrue(runtime.observe().cpu_available)
        self.assertEqual(runtime.advance().t_ns, 14)
        result = runtime.result()
        self.assertEqual(result.evaluation.status, "valid")
        self.assertEqual(result.evaluation.events[0].end_ns, 14)
        self.assertEqual(result.evaluation.events[1].end_ns, 6)
        cpu_intervals = [(item.start_ns, item.end_ns) for item in result.service_intervals
                         if item.resource == "cpu_compute"]
        self.assertEqual(cpu_intervals, [(0, 3), (7, 14)])

    def test_paused_input_remains_read_locked(self):
        runtime = SimulationSession(scene(), environment_events=(EnvironmentEvent(3, cpu_available=False),))
        runtime.submit(CPU)
        runtime.advance()
        failed = runtime.submit(Action("EVICT", "x", location="dram"))
        self.assertEqual(failed.error_code, CODE_TENSOR_IN_USE)
        self.assertEqual(runtime.state.copy_at("h", "dram"), COPY_RESERVED_OUTPUT)

    def test_middle_memory_occupancy_applies_at_its_time_and_release_restores_budget(self):
        runtime = SimulationSession(scene(), environment_events=(
            EnvironmentEvent(3, dram_external_bytes=60), EnvironmentEvent(7, dram_external_bytes=4)))
        runtime.submit(CPU)
        runtime.submit(GPU)
        self.assertEqual(runtime.advance().t_ns, 3)
        usage = next(m for m in runtime.observe().memory if m.domain == "dram")
        self.assertEqual((usage.runtime_reserved_bytes, usage.external_bytes, usage.used_bytes), (0, 60, 108))
        self.assertEqual(runtime.state.task_on("cpu_compute").remaining_ns, 7)
        runtime.advance()  # GPU completion at 6
        self.assertEqual(runtime.advance().t_ns, 7)
        self.assertEqual(next(m for m in runtime.observe().memory if m.domain == "dram").external_bytes, 4)
        runtime.advance()
        self.assertEqual(runtime.result().evaluation.peak_dram_bytes, 108)
        self.assertEqual(runtime.result().evaluation.makespan_ns, 10)

    def test_external_memory_adds_to_existing_runtime_reserve(self):
        scenario = scene()
        scenario = replace(scenario, architecture=replace(scenario.architecture,
                           dram_runtime_reserved_bytes=10))
        runtime = SimulationSession(scenario, environment_events=(EnvironmentEvent(0, dram_external_bytes=20),))
        runtime.submit(CPU)
        usage = next(m for m in runtime.observe().memory if m.domain == "dram")
        self.assertEqual((usage.runtime_reserved_bytes, usage.external_bytes, usage.tensor_bytes,
                          usage.workspace_bytes, usage.used_bytes), (10, 20, 32, 16, 78))

    def test_ssd_transfer_keeps_progressing_during_cpu_blackout(self):
        from llm_infer_model.tensor.spec import TransferCost
        scenario = scene()
        scenario = replace(scenario,
                           mapspace=replace(scenario.mapspace, allowed_transfers={"dram_to_ssd": ("x",)}),
                           costs=replace(scenario.costs, transfer={"dram_to_ssd": {"x": TransferCost(5)}}))
        runtime = SimulationSession(scenario, environment_events=(
            EnvironmentEvent(3, cpu_available=False), EnvironmentEvent(7, cpu_available=True)))
        runtime.submit(CPU)
        runtime.submit(Action("WRITE_SSD", "x"))
        runtime.advance()
        step = runtime.advance()
        self.assertEqual(step.t_ns, 5)
        self.assertIn(("x", "ssd"), step.completed_copies)
        self.assertEqual(runtime.state.task_on("cpu_compute").remaining_ns, 7)
        self.assertEqual(runtime.result().evaluation.ssd_write_bytes, 16)
        self.assertEqual(runtime.result().evaluation.events[1].end_ns, 5)

    def test_overcommit_halts_at_event_preserving_tensor_inventory_and_rejected_request(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(
            EnvironmentEvent(3, dram_external_bytes=17),))
        runtime.submit(CPU)
        failed = runtime.advance()
        self.assertEqual((failed.error_code, failed.t_ns), (CODE_EXTERNAL_MEMORY_CONFLICT, 3))
        self.assertEqual(runtime.state.task_on("cpu_compute").remaining_ns, 7)
        self.assertEqual(runtime.state.copy_at("h", "dram"), COPY_RESERVED_OUTPUT)
        record = runtime.result().environment_events[0]
        self.assertFalse(record.accepted)
        self.assertEqual(next(m for m in record.memory if m.domain == "dram").used_bytes, 65)
        self.assertEqual(next(m for m in runtime.observe().memory if m.domain == "dram").external_bytes, 0)
        self.assertIsNone(runtime.result().evaluation.makespan_ns)

    def test_environment_conflict_identifies_the_advance_that_revealed_it(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(
            EnvironmentEvent(3, dram_external_bytes=17),))
        runtime.submit(CPU, source_index=11)
        runtime.advance(source_index=17)
        result = runtime.result()
        self.assertEqual(result.evaluation.error_action_index, 1)
        self.assertEqual(result.evaluation.actions[1], Action("ADVANCE"))
        self.assertEqual(result.action_sources[1], 17)
        self.assertEqual(len(result.evaluation.actions), len(result.action_sources))
        self.assertEqual(len(result.evaluation.events), len(result.nominal_ends))
        self.assertEqual(result.evaluation.events[1].t_ns, 3)

    def test_initial_environment_failure_has_no_action_attribution(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(
            EnvironmentEvent(0, dram_external_bytes=49),))
        result = runtime.result()
        self.assertEqual(result.evaluation.error_code, CODE_EXTERNAL_MEMORY_CONFLICT)
        self.assertIsNone(result.evaluation.error_action_index)
        self.assertEqual(result.evaluation.actions, ())
        self.assertEqual(result.evaluation.events, ())
        self.assertEqual(result.action_sources, ())
        self.assertFalse(result.environment_events[0].accepted)

    def test_completion_releases_workspace_before_same_time_atomic_environment_group(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(
            EnvironmentEvent(10, dram_external_bytes=32), EnvironmentEvent(10, cpu_available=False)))
        runtime.submit(CPU)
        runtime.submit(GPU)
        runtime.advance()  # GPU at 6
        step = runtime.advance()
        self.assertTrue(step.ok)
        self.assertEqual(step.t_ns, 10)
        usage = next(m for m in runtime.observe().memory if m.domain == "dram")
        self.assertEqual((usage.workspace_bytes, usage.external_bytes, usage.used_bytes), (0, 32, 64))
        self.assertEqual(len(runtime.result().environment_events), 1)

    def test_no_future_unblock_is_an_explicit_deadlock(self):
        runtime = SimulationSession(scene(), environment_events=(EnvironmentEvent(3, cpu_available=False),))
        runtime.submit(CPU)
        runtime.advance()
        self.assertEqual(runtime.advance().error_code, CODE_ENVIRONMENT_DEADLOCK)
        self.assertEqual(runtime.state.t, 3)

    def test_rejected_deadlock_advance_preserves_action_and_decision_source(self):
        runtime = SimulationSession(scene(), environment_events=(EnvironmentEvent(3, cpu_available=False),))
        runtime.submit(CPU, source_index=11)
        runtime.advance(source_index=12)
        runtime.advance(source_index=13)
        result = runtime.result()
        self.assertEqual(result.evaluation.error_action_index, 2)
        self.assertEqual(result.evaluation.actions[2], Action("ADVANCE"))
        self.assertEqual(result.action_sources, (11, 12, 13))
        self.assertEqual(len(result.evaluation.actions), 3)
        # Rejected attempts never create successful events or service time.
        self.assertEqual(len(result.evaluation.events), 2)
        self.assertEqual(len(result.nominal_ends), 2)
        self.assertEqual(result.evaluation.action_count, 2)
        self.assertEqual(result.final_state.t, 3)

    def test_rejected_empty_advance_is_recorded_without_success_event(self):
        runtime = SimulationSession(scene())
        step = runtime.advance(source_index=23)
        self.assertEqual(step.error_code, CODE_NO_PROGRESS)
        result = runtime.result()
        self.assertEqual(result.evaluation.error_action_index, 0)
        self.assertEqual(result.evaluation.actions, (Action("ADVANCE"),))
        self.assertEqual(result.action_sources, (23,))
        self.assertEqual(result.evaluation.events, ())
        self.assertEqual(result.nominal_ends, ())

    def test_empty_advance_keeps_legacy_fixed_evaluator_error_code(self):
        # Only the stateful environment session gains the richer diagnosis.
        legacy = evaluate_mapping(scene(), (Action("ADVANCE"),))
        self.assertEqual(legacy.error_code, CODE_ILLEGAL_ACTION)

    def test_no_progress_does_not_replace_malformed_tensor_command_codes(self):
        scenario = scene()
        # A repeated output computation remains an illegal tensor action.
        runtime = SimulationSession(scenario)
        runtime.submit(CPU)
        runtime.advance()
        self.assertEqual(runtime.submit(CPU).error_code, CODE_ILLEGAL_ACTION)
        # A known tensor without a ready source retains its readiness error.
        runtime = SimulationSession(scenario)
        self.assertEqual(runtime.submit(Action("COMPUTE", "missing", device="cpu")).error_code,
                         CODE_ILLEGAL_ACTION)

    def test_t_zero_updates_are_applied_before_policy_and_cpu_start_is_rejected(self):
        runtime = SimulationSession(scene(), environment_events=(
            EnvironmentEvent(0, cpu_available=False), EnvironmentEvent(0, dram_external_bytes=9)))
        self.assertFalse(runtime.observe().cpu_available)
        self.assertEqual(runtime.observe().memory[0].external_bytes, 9)
        self.assertEqual(runtime.submit(CPU).error_code, CODE_RESOURCE_BUSY)

    def test_external_reservation_prevents_new_compute_without_committing_output(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(
            EnvironmentEvent(0, dram_external_bytes=20),))
        self.assertEqual(runtime.submit(CPU).error_code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(runtime.state.op("cpu-op"), "NOT_STARTED")
        self.assertEqual(runtime.state.copy_at("h", "dram"), COPY_ABSENT)

    def test_empty_environment_complete_replay_exactly_matches_fixed_evaluator(self):
        scenario = scene()
        actions = (CPU, GPU, Action("ADVANCE"), Action("ADVANCE"))
        runtime = SimulationSession(scenario)
        for action in actions:
            self.assertTrue(runtime.submit(action).ok)
        self.assertEqual(runtime.result().evaluation, evaluate_mapping(scenario, actions))

    def test_goal_does_not_consume_future_environment(self):
        runtime = SimulationSession(scene(), environment_events=(EnvironmentEvent(100, dram_external_bytes=200),))
        for action in (CPU, GPU, Action("ADVANCE"), Action("ADVANCE")):
            runtime.submit(action)
        self.assertTrue(runtime.finished)
        self.assertEqual(runtime.result().environment_events, ())
        self.assertFalse(runtime.advance().ok)
        self.assertEqual(runtime.state.t, 10)
        self.assertEqual(runtime.result().environment_events, ())

    def test_observation_contains_no_future_time_remaining_duration_or_private_queue(self):
        runtime = SimulationSession(scene(), environment_events=(EnvironmentEvent(999, cpu_available=False),))
        runtime.submit(CPU)
        serialized = json.dumps(asdict(runtime.observe()))
        for private in ("999", "remaining", "nominal", "queue", "scenario", "duration"):
            self.assertNotIn(private, serialized)

    def test_visibility_filters_current_ledger_and_rejects_hidden_targets(self):
        class Mask:
            def visible_operation_ids(self, state):
                return ("cpu-op",)
            def visible_tensor_ids(self, state):
                return ("x", "h")
            def allow_action(self, action, state):
                return action.target_id in {"cpu-op", "x", "h"}
        runtime = SimulationSession(scene(), visibility=Mask())
        self.assertEqual(dict(runtime.observe().op_status), {"cpu-op": "NOT_STARTED"})
        self.assertNotIn("y", dict(runtime.observe().copy_status))
        self.assertEqual(runtime.submit(GPU).error_code, CODE_INFORMATION_NOT_KNOWN)

    def test_event_inputs_are_strict_and_same_time_conflicts_are_not_order_dependent(self):
        for event in (lambda: EnvironmentEvent(True, cpu_available=True),
                      lambda: EnvironmentEvent(0, dram_external_bytes=-1),
                      lambda: EnvironmentEvent(0, cpu_available=1),
                      lambda: EnvironmentEvent(0)):
            with self.assertRaises(InvalidInput):
                event()
        for events in ((EnvironmentEvent(2, cpu_available=False), EnvironmentEvent(1, cpu_available=True)),
                       (EnvironmentEvent(0, cpu_available=False), EnvironmentEvent(0, cpu_available=True))):
            with self.assertRaises(InvalidInput):
                SimulationSession(scene(), environment_events=events)

    def test_direct_scenario_with_bogus_workspace_is_rejected_before_execution(self):
        from copy import deepcopy
        scenario = scene()
        compute = deepcopy(scenario.costs.compute)
        compute["cpu-op"]["cpu"] = {"duration_ns": 10, "workspace_bytes": None}
        malformed = replace(scenario, costs=replace(scenario.costs, compute=compute))
        with self.assertRaises(InvalidInput):
            SimulationSession(malformed)


if __name__ == "__main__":
    unittest.main()
