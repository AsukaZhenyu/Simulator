"""Hand-calculated serial boundaries, without measurements or local hardware."""
from fractions import Fraction
import json
import unittest
from unittest import mock

from llm_infer_model.performance import (InvalidModel, Parameter, ParameterBundle,
                                        UnknownParameter, evaluate)
from llm_infer_model.performance import serial_actions as serial


def parameter(name, value, unit="ns"):
    return Parameter(name, value, unit, "declared", (),
                     "synthetic nonoverlapping completed host interval")


def program():
    return serial.SerialTimingProgram((
        serial.SerialActionTiming(0, "COPY", "copy", "gap0"),
        serial.SerialActionTiming(1, "ADVANCE", "advance", "gap1"),
        serial.SerialActionTiming(2, "COMPUTE", "compute", "gap2"),
        serial.SerialActionTiming(3, "EVICT", "evict", "gap3")), "tail")


def bundle():
    return ParameterBundle(tuple(parameter(k, v) for k, v in (
        ("copy", 3), ("advance", 0), ("compute", 10), ("evict", 1),
        ("gap0", 2), ("gap1", 0), ("gap2", 4), ("gap3", 1), ("tail", 5))))


class SerialTimingTests(unittest.TestCase):
    def test_complete_actions_explicit_gaps_and_tail_hand_calculated(self):
        result = serial.evaluate_serial_actions(program(), bundle())
        self.assertEqual(result.duration_ns, 26)
        self.assertEqual([(a.begin_ns, a.end_ns) for a in result.actions],
                         [(2, 5), (5, 5), (9, 19), (20, 21)])
        self.assertEqual([a.gap_before_ns for a in result.actions], [2, 0, 4, 1])
        self.assertEqual(result.tail_ns, 5)
        self.assertEqual(result.unrounded_interval_sum_ns, 26)

    def test_calls_common_graph_evaluator_and_has_no_duplicate_timing_path(self):
        expected_costs = bundle()
        with mock.patch.object(serial, "evaluate", wraps=evaluate) as spy:
            result = serial.evaluate_serial_actions(program(), expected_costs)
        spy.assert_called_once()
        graph, costs = spy.call_args.args
        self.assertIs(costs, expected_costs)
        self.assertEqual(len(graph.tasks), 9)
        self.assertTrue(all(t.fixed_parameter is not None for t in graph.tasks))
        self.assertEqual(result.duration_ns, evaluate(graph, costs).duration_ns)

    def test_resource_names_do_not_invent_parallelism_with_completed_host_prices(self):
        graph = serial.build_serial_action_graph(program())
        self.assertEqual({t.resource for t in graph.tasks}, {"serial_host_completed"})
        for previous, current in zip(graph.tasks, graph.tasks[1:]):
            self.assertEqual(current.depends_on, (previous.id,))

    def test_all_contributions_partition_total_and_never_count_join_twice(self):
        document = serial.evaluate_serial_actions(program(), bundle()).to_dict()
        self.assertEqual(document["interval_contributions_ns"],
                         dict(COMPUTE=10, COPY=3, ADVANCE=0, EVICT=1, gap_before=7, tail=5))
        self.assertEqual(sum(document["interval_contributions_ns"].values()),
                         document["duration_ns"])
        self.assertFalse(document["physical_kernel_timeline_claimed"])
        self.assertFalse(document["physical_capacity_claimed"])
        self.assertFalse(document["instrumentation_overhead_subtracted"])
        self.assertFalse(document["independently_validated_accuracy_claimed"])
        self.assertFalse(document["profile_statistics_calculated_here"])

    def test_explicit_measured_zero_is_supported_for_all_control_intervals(self):
        costs = ParameterBundle(tuple(parameter(p.name, 0) for p in bundle().parameters))
        result = serial.evaluate_serial_actions(program(), costs)
        self.assertEqual(result.duration_ns, 0)
        self.assertEqual(len(result.actions), 4)
        self.assertTrue(all(a.duration_ns == a.gap_before_ns == 0 for a in result.actions))

    def test_fractional_observations_keep_exact_sum_and_report_integer_rounding(self):
        p = serial.SerialTimingProgram((serial.SerialActionTiming(0, "COMPUTE", "cost", "gap"),), "tail")
        costs = ParameterBundle((parameter("cost", Fraction(3, 2)),
                                 parameter("gap", Fraction(1, 2)), parameter("tail", 0)))
        result = serial.evaluate_serial_actions(p, costs)
        self.assertEqual(result.duration_ns, 3)
        self.assertEqual(result.unrounded_interval_sum_ns, 2)
        self.assertEqual(result.to_dict()["integer_stage_rounding_ns"],
                         {"numerator": 1, "denominator": 1})

    def test_missing_gap_and_unknown_action_price_are_never_assumed_zero(self):
        missing = ParameterBundle(tuple(p for p in bundle().parameters if p.name != "gap0"))
        with self.assertRaises(UnknownParameter):
            serial.evaluate_serial_actions(program(), missing)
        unknown = ParameterBundle(tuple(parameter(p.name, None if p.name == "compute" else p.value)
                                        for p in bundle().parameters))
        with self.assertRaises(UnknownParameter):
            serial.evaluate_serial_actions(program(), unknown)

    def test_rate_is_not_accepted_as_completed_action_time(self):
        wrong = ParameterBundle(tuple(parameter(p.name, p.value,
            "flop/s" if p.name == "compute" else "ns") for p in bundle().parameters))
        with self.assertRaises(InvalidModel):
            serial.evaluate_serial_actions(program(), wrong)

    def test_original_control_actions_and_order_must_match_inventory(self):
        p = program()
        p.validate_action_inventory(("COPY", "ADVANCE", "COMPUTE", "EVICT"))
        for inventory in (("COPY", "COMPUTE", "EVICT"),
                          ("COMPUTE", "ADVANCE", "COPY", "EVICT"),
                          ("COPY", "ADVANCE", "COMPUTE", "EVICT", "ADVANCE")):
            with self.subTest(inventory=inventory), self.assertRaises(InvalidModel):
                p.validate_action_inventory(inventory)

    def test_indices_cannot_duplicate_skip_or_reorder(self):
        for indices in ((0, 0), (0, 2), (1, 0), (1,)):
            with self.subTest(indices=indices), self.assertRaises(InvalidModel):
                serial.SerialTimingProgram(tuple(serial.SerialActionTiming(i, "COMPUTE",
                    f"duration{n}", f"gap{n}") for n, i in enumerate(indices)), "tail")

    def test_inclusive_action_duration_cannot_also_charge_gap_or_tail(self):
        with self.assertRaises(InvalidModel):
            serial.SerialActionTiming(0, "COMPUTE", "complete", "complete")
        actions = (serial.SerialActionTiming(0, "COMPUTE", "complete", "gap"),)
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram(actions, "complete")
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram(actions +
                (serial.SerialActionTiming(1, "COPY", "copy", "complete"),), "tail")

    def test_explicit_repeated_action_price_is_legal_without_becoming_one_action(self):
        p = serial.SerialTimingProgram((
            serial.SerialActionTiming(0, "COMPUTE", "same_duration", "zero_gap"),
            serial.SerialActionTiming(1, "COMPUTE", "same_duration", "zero_gap")), "zero_gap")
        result = serial.evaluate_serial_actions(p, ParameterBundle((
            parameter("same_duration", 7), parameter("zero_gap", 0))))
        self.assertEqual(result.duration_ns, 14)
        self.assertEqual(len(result.actions), 2)

    def test_strict_json_roundtrip_and_extra_internal_phase_rejection(self):
        restored = serial.SerialTimingProgram.from_dict(json.loads(json.dumps(program().to_dict())))
        self.assertEqual(restored, program())
        document = program().to_dict()
        document["actions"][0]["device_active_parameter"] = "already_inside_complete"
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram.from_dict(document)
        document = program().to_dict()
        del document["tail_parameter"]
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram.from_dict(document)

    def test_empty_and_malformed_contracts_rejected(self):
        for kwargs in (dict(action_index=True, kind="COPY", duration_parameter="a", gap_before_parameter="b"),
                       dict(action_index=0, kind="KERNEL", duration_parameter="a", gap_before_parameter="b"),
                       dict(action_index=0, kind="COMPUTE", duration_parameter="", gap_before_parameter="b")):
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidModel):
                serial.SerialActionTiming(**kwargs)
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram((), "tail")
        with self.assertRaises(InvalidModel):
            serial.SerialTimingProgram(program().actions, "tail", "unknown")
        with self.assertRaises(InvalidModel):
            serial.build_serial_action_graph({})
        with self.assertRaises(InvalidModel):
            serial.evaluate_serial_actions(program(), {})


if __name__ == "__main__":
    unittest.main()
