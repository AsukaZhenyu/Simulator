"""Hand-derived ISR/DPC service scenarios, not hardware calibration results."""
from dataclasses import replace
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance import (
    CpuInterruption, CpuInterruptionTimebase, CpuInterruptionTrace,
    Graph, INSTANTANEOUS_RATE, InvalidModel, MemoryDomain, Parameter,
    ParameterBundle, ResourcePoint, ResourceTimeline, Task, ThreadCpuBinding,
    UnknownParameter, WorkDemand, compose_cpu_interruptions, evaluate,
)

MS = 1_000_000
FULL = 1_000_000_000
CLOCK = CpuInterruptionTimebase("declared-clock", "scenario-zero", FULL,
                               "scenario-host/group0", (0, 1))
BINDINGS = (ThreadCpuBinding("host_supply", 0),)


def parameter(name="host_supply", value=FULL, unit="thread_cpu_ns/s"):
    return Parameter(name, value, unit, "declared", (),
                     "hand-derived scenario; base supply excludes supplied ISR/DPC intervals")


def interruption(start=3, end=7, *, cpu=0, kind="dpc", origin="predeclared_scenario"):
    return CpuInterruption(cpu, start * MS, end * MS, kind, origin,
                           ("hand-derived interruption scenario",))


def task(name="host", *, rate="host_supply", work=10 * MS, depends_on=()):
    return Task(name, "thread:" + name,
                (WorkDemand(rate, work, "thread_cpu_ns"),), depends_on)


def overlay(intervals=(), *, bundle=None, bindings=BINDINGS,
            baseline=None, clock=CLOCK, baseline_clock=CLOCK, observed=False):
    return compose_cpu_interruptions(
        bundle if bundle is not None else ParameterBundle((parameter(),)),
        bindings, CpuInterruptionTrace(clock, intervals),
        baseline_timebase=baseline_clock, baseline_timeline=baseline,
        allow_observed_diagnostic=observed)


class CpuInterruptionTests(unittest.TestCase):
    def test_overlapping_dpc_isr_union_adds_six_ms_not_eight(self):
        composition = overlay((interruption(3, 7), interruption(5, 9, kind="isr")))
        result = evaluate(Graph((task(),)), ParameterBundle((parameter(),)), composition.timeline)
        self.assertEqual(result.duration_ns, 16 * MS)
        self.assertEqual([(item.start_ns, item.end_ns) for item in composition.unions],
                         [(3 * MS, 9 * MS)])
        self.assertEqual(composition.unions[0].input_indices, (0, 1))
        self.assertEqual([(row.start_ns, row.end_ns, row.reasons) for row in result.waits],
                         [(3 * MS, 9 * MS, ("service_unavailable:host_supply",))])

    def test_nested_and_duplicate_intervals_are_not_double_charged(self):
        composition = overlay((interruption(5, 6, kind="isr"),
                               interruption(3, 9), interruption(3, 9)))
        self.assertEqual(len(composition.unions), 1)
        self.assertEqual(set(composition.unions[0].input_indices), {0, 1, 2})
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 16 * MS)

    def test_adjacent_intervals_do_not_create_an_instant_of_service(self):
        composition = overlay((interruption(3, 5), interruption(5, 9, kind="isr")))
        self.assertEqual([point.at_ns for point in composition.timeline.points], [3 * MS, 9 * MS])
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 16 * MS)

    def test_other_logical_cpu_is_unaffected(self):
        composition = overlay((interruption(3, 9, cpu=1),))
        self.assertEqual(composition.timeline.points, ())
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 10 * MS)
        self.assertEqual(composition.unions[0].logical_cpu_id, 1)

    def test_two_threads_on_different_cpus_only_bound_interrupted_thread_pauses(self):
        bundle = ParameterBundle((parameter(), parameter("second_supply")))
        bindings = BINDINGS + (ThreadCpuBinding("second_supply", 1),)
        composition = overlay((interruption(3, 9),), bundle=bundle, bindings=bindings)
        graph = Graph((task(), task("second", rate="second_supply")))
        result = evaluate(graph, bundle, composition.timeline)
        self.assertEqual({row.task_id: row.end_ns for row in result.tasks},
                         {"host": 16 * MS, "second": 10 * MS})

    def test_gpu_continues_and_memory_updates_are_preserved(self):
        bundle = ParameterBundle((parameter(), parameter("gpu_rate", FULL, "flop/s"),
                                  parameter("dram_capacity", 30, "byte")))
        baseline = ResourceTimeline((ResourcePoint(6 * MS, {"gpu_rate": FULL // 2},
                                                   {"dram": 20}, {"dram": 4}),))
        composition = overlay((interruption(3, 9),), bundle=bundle, baseline=baseline)
        graph = Graph((task(), Task("gpu", "device", (WorkDemand("gpu_rate", 12 * MS, "flop"),))),
                      (MemoryDomain("dram", "dram_capacity", 0, 0),))
        result = evaluate(graph, bundle, composition.timeline)
        # GPU does 6 ms work by t=6, then the remaining 6 at half rate: t=18.
        self.assertEqual({row.task_id: row.end_ns for row in result.tasks},
                         {"host": 16 * MS, "gpu": 18 * MS})
        middle = next(point for point in composition.timeline.points if point.at_ns == 6 * MS)
        self.assertEqual(dict(middle.rates), {"gpu_rate": FULL // 2})
        self.assertEqual(dict(middle.capacities), {"dram": 20})
        self.assertEqual(dict(middle.external_memory), {"dram": 4})

    def test_latest_base_rate_change_during_interrupt_is_restored(self):
        baseline = ResourceTimeline((ResourcePoint(4 * MS, {"host_supply": FULL // 4}),
                                     ResourcePoint(8 * MS, {"host_supply": FULL // 2})))
        composition = overlay((interruption(3, 9),), baseline=baseline)
        values = {point.at_ns: dict(point.rates) for point in composition.timeline.points}
        self.assertEqual(values[4 * MS], {"host_supply": 0})
        self.assertEqual(values[8 * MS], {"host_supply": 0})
        self.assertEqual(values[9 * MS], {"host_supply": FULL // 2})
        # 3 ms at full supply, 6 ms without service, 7 ms work at half supply.
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 23 * MS)

    def test_base_update_at_union_end_is_applied_atomically(self):
        baseline = ResourceTimeline((ResourcePoint(3 * MS, {"host_supply": FULL // 2}),
                                     ResourcePoint(9 * MS, {"host_supply": FULL})))
        composition = overlay((interruption(3, 9),), baseline=baseline)
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 16 * MS)

    def test_interruption_at_zero_applies_before_work_starts(self):
        composition = overlay((interruption(0, 5),))
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 15 * MS)

    def test_interruption_after_task_completion_is_not_a_fixed_layer_penalty(self):
        composition = overlay((interruption(12, 30),))
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 10 * MS)

    def test_fixed_wall_latency_is_not_paused(self):
        bundle = ParameterBundle((parameter(), parameter("wall", 10 * MS, "ns")))
        composition = overlay((interruption(3, 9),), bundle=bundle)
        graph = Graph((Task("wall", "wall-clock", fixed_parameter="wall"),))
        self.assertEqual(evaluate(graph, bundle, composition.timeline).duration_ns, 10 * MS)

    def test_unknown_or_wrong_unit_bound_rate_is_rejected(self):
        for bundle in (ParameterBundle(()), ParameterBundle((parameter(value=None),))):
            with self.subTest(bundle=bundle), self.assertRaises(UnknownParameter):
                overlay(bundle=bundle)
        with self.assertRaisesRegex(InvalidModel, "expected thread_cpu_ns/s"):
            overlay(bundle=ParameterBundle((parameter(unit="flop/s"),)))

    def test_invalid_hidden_rate_update_cannot_be_masked_into_valid_zero(self):
        for value in (FULL + 1, INSTANTANEOUS_RATE):
            with self.subTest(value=value), self.assertRaises(InvalidModel):
                overlay((interruption(3, 9),),
                        baseline=ResourceTimeline((ResourcePoint(5 * MS, {"host_supply": value}),)))

    def test_unknown_unbound_rate_or_non_rate_update_is_rejected(self):
        with self.assertRaises(UnknownParameter):
            overlay(baseline=ResourceTimeline((ResourcePoint(1, {"unknown": 1}),)))
        with self.assertRaisesRegex(InvalidModel, "service rates"):
            overlay(bundle=ParameterBundle((parameter(), parameter("delay", 10, "ns"))),
                    baseline=ResourceTimeline((ResourcePoint(1, {"delay": 1}),)))

    def test_unknown_cpu_ids_are_rejected_at_trace_and_binding_boundaries(self):
        with self.assertRaisesRegex(InvalidModel, "unknown logical CPU"):
            CpuInterruptionTrace(CLOCK, (interruption(cpu=2),))
        with self.assertRaisesRegex(InvalidModel, "unknown logical CPU"):
            overlay(bindings=(ThreadCpuBinding("host_supply", 2),))
        for value in (None, -1, True, "0"):
            with self.subTest(cpu=value), self.assertRaises(InvalidModel):
                ThreadCpuBinding("host_supply", value)

    def test_bad_clock_origin_frequency_or_namespace_is_rejected(self):
        for change in (dict(clock_id="other"), dict(origin_id="other"),
                       dict(frequency_hz=10_000_000), dict(logical_cpu_namespace="other"),
                       dict(logical_cpu_ids=(0,))):
            with self.subTest(change=change), self.assertRaisesRegex(InvalidModel, "must match exactly"):
                overlay(baseline_clock=replace(CLOCK, **change))
        with self.assertRaises(InvalidModel): overlay(baseline_clock=None)

    def test_bad_clock_metadata_and_unknown_cpu_inventory_are_rejected(self):
        for change in (dict(clock_id=""), dict(origin_id=""), dict(frequency_hz=0),
                       dict(frequency_hz=True), dict(logical_cpu_namespace=""),
                       dict(logical_cpu_ids=()), dict(logical_cpu_ids=(0, 0)),
                       dict(logical_cpu_ids=(None,)), dict(logical_cpu_ids=[0])):
            with self.subTest(change=change), self.assertRaises(InvalidModel):
                replace(CLOCK, **change)

    def test_invalid_interval_bounds_kind_origin_and_evidence_are_rejected(self):
        valid = interruption()
        for change in (dict(start_ns=-1), dict(end_ns=valid.start_ns),
                       dict(end_ns=valid.start_ns - 1), dict(start_ns=Fraction(1, 2)),
                       dict(kind="unknown"), dict(kind=[]), dict(origin="unknown"),
                       dict(evidence=()), dict(evidence=("",)), dict(evidence=["source"])):
            with self.subTest(change=change), self.assertRaises(InvalidModel):
                replace(valid, **change)

    def test_observed_diagnostic_schedule_cannot_be_prediction_input_by_default(self):
        with self.assertRaisesRegex(InvalidModel, "explicit diagnostic replay"):
            overlay((interruption(origin="observed_diagnostic"),))

    def test_explicit_observed_replay_retains_provenance_and_is_unqualified(self):
        source = interruption(3, 9, origin="observed_diagnostic")
        composition = overlay((source,), observed=True)
        self.assertTrue(composition.diagnostic_replay)
        self.assertFalse(composition.accuracy_qualified)
        encoded = json.loads(json.dumps(composition.to_dict()))
        self.assertEqual(encoded["trace"]["intervals"][0], source.to_dict())
        self.assertTrue(encoded["diagnostic_replay"])
        self.assertFalse(encoded["accuracy_qualified"])
        self.assertEqual(evaluate(Graph((task(),)), ParameterBundle((parameter(),)),
                                  composition.timeline).duration_ns, 16 * MS)

    def test_declared_and_external_scenarios_do_not_claim_empirical_accuracy(self):
        composition = overlay((interruption(), interruption(8, 9, origin="external_scenario")))
        self.assertFalse(composition.diagnostic_replay)
        self.assertFalse(composition.accuracy_qualified)

    def test_empty_trace_preserves_baseline_exactly(self):
        baseline = ResourceTimeline((ResourcePoint(3 * MS, {"host_supply": FULL // 2}),))
        self.assertEqual(overlay(baseline=baseline).timeline, baseline)

    def test_bad_container_types_and_duplicate_bindings_fail(self):
        for bindings in ((), [BINDINGS[0]], BINDINGS + BINDINGS, ("host",)):
            with self.subTest(bindings=bindings), self.assertRaises(InvalidModel):
                overlay(bindings=bindings)
        with self.assertRaises(InvalidModel): overlay(observed="true")
        with self.assertRaises(InvalidModel): overlay(baseline={})
        with self.assertRaises(InvalidModel): CpuInterruptionTrace(CLOCK, [])
        with self.assertRaises(InvalidModel): CpuInterruptionTrace(CLOCK, ("irq",))


if __name__ == "__main__":
    unittest.main()
