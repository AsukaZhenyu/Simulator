"""Hand calculations for composite responses; no hardware or fitted targets."""
from fractions import Fraction
import json
import unittest
from unittest import mock

from llm_infer_model.performance import (INSTANTANEOUS_RATE, BlockedExecution,
    Graph, InstantaneousRate, InvalidModel, Parameter, ParameterBundle,
    ResourcePoint, ResourceTimeline, Task, UnknownParameter, WorkDemand, evaluate)
from llm_infer_model.performance import conditioned_response as response


BASE = response.ResponseState(0, 0, 0, 0)
LOAD = response.ResponseState(1, 1, 0, 0)


def cell(action, state, duration, qualified=True):
    return response.ResponseCell(action, state, duration, ("synthetic hand-calculated fixture",), qualified)


def table(base=(10, 4), loaded=(20, 8)):
    actions = tuple(f"interval:{i}" for i in range(len(base)))
    return response.ResponseTable("a" * 64, actions, tuple(
        cell(action, state, duration) for state, durations in ((BASE, base), (LOAD, loaded))
        for action, duration in zip(actions, durations)))


def parameter(name, value, unit="invocation/s"):
    return Parameter(name, value, unit, "declared", (), "synthetic explicit service")


class ConditionedResponseTests(unittest.TestCase):
    def test_constant_responses_preserve_original_serial_interval_semantics(self):
        from llm_infer_model.performance.serial_actions import (
            SerialActionTiming, SerialTimingProgram, build_serial_action_graph,
            evaluate_serial_actions)
        program = SerialTimingProgram((
            SerialActionTiming(0, "ADVANCE", "action0", "gap0"),
            SerialActionTiming(1, "COMPUTE", "action1", "gap1")), "tail")
        values = {"gap0": Fraction(1, 2), "action0": 0, "gap1": 0,
                  "action1": Fraction(3, 2), "tail": 3}
        bundle = ParameterBundle(tuple(parameter(key, value, "ns")
                                        for key, value in values.items()))
        original = evaluate_serial_actions(program, bundle)
        tasks = build_serial_action_graph(program).tasks
        prices = response.ResponseTable("a" * 64, tuple(t.id for t in tasks),
            tuple(cell(t.id, BASE, values[t.fixed_parameter]) for t in tasks))
        conditioned = response.evaluate_response(prices, response.ResponseTrajectory(BASE))
        self.assertEqual(conditioned.duration_ns, original.duration_ns)
        self.assertEqual([(t.task_id, t.start_ns, t.end_ns) for t in conditioned.execution.tasks],
                         [(t.task_id, t.start_ns, t.end_ns) for t in original.execution.tasks])

    def test_constant_state_serial_sequence_uses_same_rule(self):
        baseline = response.evaluate_response(table(), response.ResponseTrajectory(BASE))
        loaded = response.evaluate_response(table(), response.ResponseTrajectory(LOAD))
        self.assertEqual((baseline.duration_ns, loaded.duration_ns), (14, 28))
        self.assertEqual([(t.start_ns, t.end_ns) for t in loaded.execution.tasks], [(0, 20), (20, 28)])

    def test_mid_action_state_change_retains_completed_work(self):
        trajectory = response.ResponseTrajectory(BASE, (response.ResponsePoint(5, LOAD),))
        result = response.evaluate_response(table(), trajectory)
        # First half of A takes 5 ns, remaining half 10 ns; B takes 8 ns.
        self.assertEqual(result.duration_ns, 23)
        self.assertEqual(result.execution.tasks[0].end_ns, 15)

    def test_load_then_recovery_is_one_integral_with_disclosed_integer_completion(self):
        trajectory = response.ResponseTrajectory(BASE, (
            response.ResponsePoint(5, LOAD), response.ResponsePoint(10, BASE)))
        result = response.evaluate_response(table(), trajectory)
        # A progress = .5 + .25; last .25 needs 2.5 ns, rounded to 3.
        self.assertEqual(result.duration_ns, 17)
        self.assertEqual(result.execution.tasks[0].end_ns, 13)

    def test_actual_external_entry_timestamp_selects_state_not_heldout_internal_stamps(self):
        trajectory = response.ResponseTrajectory(BASE, (response.ResponsePoint(5, LOAD),))
        result = response.evaluate_response(table(), trajectory, start_ns=10)
        self.assertEqual(result.duration_ns, 28)
        self.assertEqual(result.end_ns, 38)
        self.assertEqual(result.to_dict()["absolute_tasks"][0]["start_ns"], 10)
        self.assertEqual(response.evaluate_response(table(), trajectory, start_ns=0).duration_ns, 23)

    def test_large_integer_external_clock_origin_is_exact(self):
        origin = 10**18
        trajectory = response.ResponseTrajectory(BASE, (response.ResponsePoint(origin + 5, LOAD),))
        result = response.evaluate_response(table(), trajectory, start_ns=origin)
        self.assertEqual((result.duration_ns, result.end_ns), (23, origin + 23))

    def test_explicit_composite_service_blackout_pauses_and_recovers(self):
        trajectory = response.ResponseTrajectory(BASE, (
            response.ResponsePoint(3, BASE, False), response.ResponsePoint(8, BASE, True)))
        self.assertEqual(response.evaluate_response(table(), trajectory).duration_ns, 19)

    def test_unrecoverable_blackout_is_not_unknown_or_instant(self):
        with self.assertRaises(BlockedExecution):
            response.evaluate_response(table(), response.ResponseTrajectory(BASE, initial_available=False))

    def test_mixed_zero_and_positive_state_cost_completes_same_instant(self):
        prices = table(base=(10, 4), loaded=(0, 0))
        trajectory = response.ResponseTrajectory(BASE, (response.ResponsePoint(4, LOAD),))
        result = response.evaluate_response(prices, trajectory)
        self.assertEqual(result.duration_ns, 4)
        self.assertEqual([(t.start_ns, t.end_ns) for t in result.execution.tasks], [(0, 4), (4, 4)])
        self.assertTrue(result.to_dict()["zero_duration_approximation_used"])

    def test_zero_state_while_blacked_out_waits_for_explicit_recovery(self):
        trajectory = response.ResponseTrajectory(LOAD, (
            response.ResponsePoint(8, LOAD, True),), initial_available=False)
        self.assertEqual(response.evaluate_response(table(loaded=(0, 0)), trajectory).duration_ns, 8)

    def test_all_zero_controls_and_empty_work_do_not_loop(self):
        self.assertEqual(response.evaluate_response(table(base=(0, 0)), response.ResponseTrajectory(BASE)).duration_ns, 0)
        costs = ParameterBundle((parameter("stopped", 0), parameter("zero", 0, "ns")))
        graph = Graph((Task("zero_work", "host", (WorkDemand("stopped", 0, "invocation"),)),
                       Task("zero_fixed", "host", depends_on=("zero_work",), fixed_parameter="zero")))
        self.assertEqual(evaluate(graph, costs).duration_ns, 0)

    def test_only_completed_response_scope_is_claimed(self):
        result = response.evaluate_response(table(), response.ResponseTrajectory(BASE)).to_dict()
        self.assertFalse(result["physical_gpu_throughput_claimed"])
        self.assertFalse(result["host_device_overlap_decomposed"])
        self.assertFalse(result["physical_memory_capacity_claimed"])
        self.assertFalse(result["independently_validated_accuracy_claimed"])
        self.assertFalse(result["parameter_estimation_or_measurement_performed_here"])
        self.assertFalse(result["cpu_gpu_controls_are_measured_utilization"])

    def test_qualification_is_reported_without_inventing_missing_duration(self):
        prices = response.ResponseTable("a" * 64, ("a",), (cell("a", BASE, 7, False),))
        result = response.evaluate_response(prices, response.ResponseTrajectory(BASE))
        self.assertEqual(result.duration_ns, 7)
        self.assertFalse(result.response_support_qualified)
        unknown = response.ResponseTable("a" * 64, ("a",), (cell("a", BASE, None, False),))
        with self.assertRaises(UnknownParameter):
            response.evaluate_response(unknown, response.ResponseTrajectory(BASE))

    def test_missing_state_or_interval_rejected_without_interpolation(self):
        with self.assertRaises(UnknownParameter):
            response.evaluate_response(table(), response.ResponseTrajectory(response.ResponseState(1, 0, 0, 0)))
        with self.assertRaises(UnknownParameter):
            response.evaluate_response(table(), response.ResponseTrajectory(response.ResponseState(0, 0, 64, 0)))
        missing = response.ResponseTable("a" * 64, ("a", "missing"), (cell("a", BASE, 1),))
        with self.assertRaises(UnknownParameter):
            response.evaluate_response(missing, response.ResponseTrajectory(BASE))

    def test_invocation_fork_or_physical_demand_misuse_rejected(self):
        with self.assertRaises(InvalidModel):
            response.ResponseTable("a" * 64, ("a",), (cell("a", BASE, 1),), "parallel_gpu")
        with self.assertRaises(InvalidModel):
            Task("bad", "gpu", (WorkDemand("response", 1, "invocation"), WorkDemand("gpu", 10, "flop")))

    def test_common_evaluator_is_used_with_invocation_units(self):
        with mock.patch.object(response, "evaluate", wraps=evaluate) as spy:
            result = response.evaluate_response(table(), response.ResponseTrajectory(BASE))
        spy.assert_called_once()
        graph, bundle, timeline = spy.call_args.args
        self.assertTrue(all(t.demands[0].unit == "invocation" for t in graph.tasks))
        self.assertTrue(all(p.unit == "invocation/s" for p in bundle.parameters))
        self.assertEqual({t.resource for t in graph.tasks}, {"completed_action_response"})
        self.assertEqual(evaluate(graph, bundle, timeline).duration_ns, result.duration_ns)

    def test_table_and_trajectory_json_preserve_exact_fraction_and_scope(self):
        prices = table(base=(Fraction(3, 2), 0))
        trajectory = response.ResponseTrajectory(BASE, (response.ResponsePoint(12, LOAD, False),))
        restored = response.ResponseTable.from_dict(json.loads(json.dumps(prices.to_dict())))
        restored_trajectory = response.ResponseTrajectory.from_dict(json.loads(json.dumps(trajectory.to_dict())))
        self.assertEqual(restored, prices)
        self.assertEqual(restored_trajectory, trajectory)
        self.assertEqual(response.evaluate_response(restored, restored_trajectory).duration_ns, 2)

    def test_states_context_order_and_unknown_json_fields_are_strict(self):
        for state in ((True, 0, 0, 0), (2, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, -1, 0)):
            with self.subTest(state=state), self.assertRaises(InvalidModel):
                response.ResponseState(*state)
        with self.assertRaises(InvalidModel):
            response.ResponseTable("unbound", ("a",), (cell("a", BASE, 1),))
        with self.assertRaises(InvalidModel):
            response.ResponseTable("a" * 64, ("a",), (cell("a", BASE, 1), cell("a", BASE, 2)))
        with self.assertRaises(InvalidModel):
            response.ResponseTrajectory(BASE, (response.ResponsePoint(2, BASE), response.ResponsePoint(1, LOAD)))
        document = table().to_dict()
        document["device_flops"] = 1
        with self.assertRaises(InvalidModel):
            response.ResponseTable.from_dict(document)

    def test_exact_lookup_cache_is_read_only_and_not_part_of_serialized_contract(self):
        prices = table()
        with self.assertRaises(TypeError):
            prices._by_key[("interval:0", BASE)] = cell("interval:0", BASE, 999)
        self.assertNotIn("_by_key", prices.to_dict())
        self.assertEqual(prices.lookup("interval:0", BASE).duration_ns, 10)
        self.assertEqual(response.ResponseTable.from_dict(prices.to_dict()), prices)


class ExplicitInstantaneousRateTests(unittest.TestCase):
    def test_sentinel_roundtrips_separately_from_numeric_pause(self):
        costs = ParameterBundle((parameter("instant", INSTANTANEOUS_RATE), parameter("paused", 0)))
        document = costs.to_dict()
        self.assertEqual(document["parameters"][0]["value"], {"instantaneous": True})
        self.assertEqual(ParameterBundle.from_dict(json.loads(json.dumps(document))), costs)
        timeline = ResourceTimeline((ResourcePoint(0, rates={"paused": INSTANTANEOUS_RATE}),))
        self.assertEqual(ResourceTimeline.from_dict(json.loads(json.dumps(timeline.to_dict()))), timeline)

    def test_instant_rates_cannot_describe_physical_flops_bytes_or_fixed_latency(self):
        for unit in ("flop/s", "byte/s", "ns", "byte"):
            with self.subTest(unit=unit), self.assertRaises(InvalidModel):
                parameter("bad", INSTANTANEOUS_RATE, unit)
        graph = Graph((Task("gpu", "gpu", (WorkDemand("flops", 10, "flop"),)),))
        with self.assertRaises(InvalidModel):
            evaluate(graph, ParameterBundle((parameter("flops", 1, "flop/s"),)),
                     ResourceTimeline((ResourcePoint(0, rates={"flops": INSTANTANEOUS_RATE}),)))

    def test_mid_task_numeric_pause_then_instant_recovery_keeps_same_engine(self):
        graph = Graph((Task("action", "response", (WorkDemand("r", 1, "invocation"),)),))
        timeline = ResourceTimeline((ResourcePoint(2, rates={"r": 0}),
                                     ResourcePoint(7, rates={"r": INSTANTANEOUS_RATE})))
        result = evaluate(graph, ParameterBundle((parameter("r", 100_000_000),)), timeline)
        self.assertEqual(result.duration_ns, 7)

    def test_zero_fixed_host_and_physical_gpu_join_use_overlap_without_adding_wait(self):
        costs = ParameterBundle((parameter("submit", 2, "ns"), parameter("host", 5, "ns"),
            parameter("gpu", 1_000_000_000, "flop/s"), parameter("join", 1, "ns")))
        graph = Graph((Task("submit", "host", fixed_parameter="submit"),
            Task("device", "gpu", (WorkDemand("gpu", 10, "flop"),), depends_on=("submit",)),
            Task("host_suffix", "host", depends_on=("submit",), fixed_parameter="host"),
            Task("resume", "host", depends_on=("device", "host_suffix"), fixed_parameter="join")))
        self.assertEqual(evaluate(graph, costs).duration_ns, 13)


if __name__ == "__main__":
    unittest.main()
