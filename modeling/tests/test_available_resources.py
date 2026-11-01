"""Hand-derived checks for absolute external resource supply, not OS simulation."""
from dataclasses import replace
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance.available_resources import (
    AvailableResources, ServiceBinding, evaluate_available_resources,
)
from llm_infer_model.performance.evaluator import CapacityExceeded
from llm_infer_model.performance.graph import (
    Graph, MemoryDomain, MemoryReservation, ResourcePoint, ResourceTimeline, Task, WorkDemand,
)
from llm_infer_model.performance.parameters import (
    INSTANTANEOUS_RATE, InvalidModel, Parameter, ParameterBundle, UnknownParameter,
)

FULL = 1_000_000_000
EVIDENCE = ("hand-derived physical resource declaration",)


def parameter(name, value, unit):
    return Parameter(name, value, unit, "declared", EVIDENCE, "resource available to this LLM")


def resources(*, points=(), horizon=100, values=None, source="declared_scenario"):
    values = values if values is not None else (("cpu_supply", FULL, "thread_cpu_ns/s"),
        ("gpu_supply", FULL, "flop/s"), ("link_supply", FULL, "byte/s"), ("memory_budget", 32, "byte"))
    return AvailableResources(ParameterBundle(tuple(parameter(*p) for p in values)),
        ResourceTimeline(tuple(points)), horizon, "scenario_ns", "scenario_start", source,
        EVIDENCE, "external available service; no utilization conversion")


def binding(target, source, efficiency=1):
    return ServiceBinding(target, source, efficiency, "declared", EVIDENCE, "explicit physical stage context")


def compute(name="cpu", amount=10, *, target="cpu_rate", unit="thread_cpu_ns", lane="cpu", deps=(), extra=(), memory=()):
    return Task(name, lane, (WorkDemand(target, amount, unit),), deps,
                extra_resources=extra, memory=memory)


def run(graph, available=None, bindings=None, fixed=(), **kwargs):
    return evaluate_available_resources(graph, ParameterBundle(tuple(fixed)),
        available if available is not None else resources(),
        bindings if bindings is not None else (binding("cpu_rate", "cpu_supply"),), **kwargs)


class AvailableResourcesTests(unittest.TestCase):
    def test_cpu_gpu_copy_same_law_with_independent_physical_units(self):
        graph = Graph((compute(amount=4),
            compute("gpu", 6, target="gpu_rate", unit="flop", lane="gpu"),
            compute("copy", 8, target="copy_rate", unit="byte", lane="copy")))
        result = run(graph, bindings=(binding("cpu_rate", "cpu_supply", Fraction(1, 2)),
            binding("gpu_rate", "gpu_supply", Fraction(1, 2)), binding("copy_rate", "link_supply", Fraction(1, 2))))
        self.assertEqual({task.task_id: task.end_ns for task in result.tasks}, {"cpu": 8, "gpu": 12, "copy": 16})
        self.assertFalse(result.accuracy_qualified)
        self.assertFalse(result.to_dict()["external_background_applied_again"])

    def test_zero_supply_retains_work_until_external_recovery(self):
        available = resources(points=(ResourcePoint(2, {"cpu_supply": 0}),
            ResourcePoint(5, {"cpu_supply": FULL // 2})))
        result = run(Graph((compute(amount=6),)), available)
        self.assertEqual(result.duration_ns, 13)  # 2 work before pause, 4 / .5 after recovery.
        self.assertTrue(any("service_unavailable:cpu_rate" in interval.reasons for interval in result.waits))

    def test_predicted_dependency_boundary_selects_later_supply(self):
        available = resources(points=(ResourcePoint(2, {"cpu_supply": FULL // 2}),
            ResourcePoint(10, {"gpu_supply": FULL // 4})))
        graph = Graph((compute(amount=10), compute("gpu", 8, target="gpu_rate", unit="flop", lane="gpu", deps=("cpu",))))
        result = run(graph, available, (binding("cpu_rate", "cpu_supply"), binding("gpu_rate", "gpu_supply")))
        executions = {task.task_id: task for task in result.tasks}
        self.assertEqual(executions["cpu"].end_ns, 18)
        self.assertEqual((executions["gpu"].start_ns, executions["gpu"].end_ns), (18, 50))

    def test_nonzero_entry_uses_absolute_external_history_once(self):
        available = resources(points=(ResourcePoint(10, {"cpu_supply": FULL // 2}),))
        result = run(Graph((compute(amount=4),)), available, start_ns=12)
        self.assertEqual(result.start_ns, 12)
        self.assertEqual((result.tasks[0].start_ns, result.tasks[0].end_ns), (0, 8))
        self.assertEqual(result.parameters.require("cpu_rate", "thread_cpu_ns/s"), FULL // 2)

    def test_fixed_overhead_is_not_scaled_and_gate_is_not_second_gpu_work(self):
        fixed = (parameter("launch_wall", 2, "ns"),)
        graph = Graph((Task("launch", "cpu", fixed_parameter="launch_wall"),
            compute("gpu", 10, target="gpu_rate", unit="flop", lane="gpu", deps=("launch",)),
            Task("wait", "blocked_caller", depends_on=("launch",), waits_for=("gpu",)),
            compute("other_host", 4, deps=("launch",)),
            compute("resume", 3, deps=("wait",))))
        result = run(graph, bindings=(binding("cpu_rate", "cpu_supply"), binding("gpu_rate", "gpu_supply")), fixed=fixed)
        by_id = {task.task_id: task for task in result.tasks}
        self.assertEqual((by_id["wait"].start_ns, by_id["wait"].end_ns), (2, 12))
        self.assertEqual((by_id["other_host"].start_ns, by_id["other_host"].end_ns), (2, 6))
        self.assertEqual((by_id["resume"].start_ns, result.duration_ns), (12, 15))

    def test_physical_runtime_overhead_pauses_with_thread_service(self):
        available = resources(points=(ResourcePoint(0, {"cpu_supply": FULL // 2}),))
        result = run(Graph((compute("runtime", 3),)), available)
        self.assertEqual(result.duration_ns, 6)

    def test_memory_budget_is_available_not_total_minus_external_again(self):
        graph = Graph((compute(amount=10, memory=(MemoryReservation("ram", 4),)),),
                      (MemoryDomain("ram", "memory_budget", 8, 0),))
        result = run(graph)
        self.assertEqual(result.parameters.require("memory_budget", "byte"), 32)
        with self.assertRaisesRegex(InvalidModel, "exclude external"):
            run(replace(graph, memory_domains=(MemoryDomain("ram", "memory_budget", 8, 1),)))
        with self.assertRaisesRegex(InvalidModel, "subtract it twice"):
            resources(points=(ResourcePoint(1, external_memory={"ram": 1}),))

    def test_memory_shrink_during_execution_fails_explicitly(self):
        graph = Graph((compute(amount=10, memory=(MemoryReservation("ram", 4),)),),
                      (MemoryDomain("ram", "memory_budget", 4, 0),))
        with self.assertRaisesRegex(CapacityExceeded, "at 5 ns"):
            run(graph, resources(points=(ResourcePoint(5, capacities={"ram": 7}),)))

    def test_capacity_recovery_defers_task_and_changes_following_state(self):
        graph = Graph((compute(amount=2, memory=(MemoryReservation("ram", 6),)),),
                      (MemoryDomain("ram", "memory_budget", 4, 0),))
        available = resources(values=(("cpu_supply", FULL, "thread_cpu_ns/s"), ("memory_budget", 8, "byte")),
                              points=(ResourcePoint(5, capacities={"ram": 10}),))
        result = run(graph, available)
        self.assertEqual((result.tasks[0].start_ns, result.duration_ns), (5, 7))

    def test_shared_supply_cannot_be_copied_to_concurrent_aliases(self):
        graph = Graph((compute("one", lane="lane1"), compute("two", target="other", lane="lane2")))
        bindings = (binding("cpu_rate", "cpu_supply"), binding("other", "cpu_supply"))
        with self.assertRaisesRegex(InvalidModel, "common physical resource lock"):
            run(graph, bindings=bindings)
        locked = Graph(tuple(replace(task, extra_resources=("shared_cpu",)) for task in graph.tasks))
        self.assertEqual(run(locked, bindings=bindings).duration_ns, 20)

    def test_dependency_ordered_aliases_are_safe_without_extra_lock(self):
        graph = Graph((compute("one", amount=2, lane="lane1"),
            compute("two", amount=3, target="other", lane="lane2", deps=("one",))))
        result = run(graph, bindings=(binding("cpu_rate", "cpu_supply"), binding("other", "cpu_supply")))
        self.assertEqual(result.duration_ns, 5)

    def test_long_reverse_ordered_chain_does_not_require_python_recursion(self):
        count = 1100
        graph = Graph(tuple(compute("stage:" + str(index), amount=1,
            lane="lane:" + str(index), deps=("stage:" + str(index - 1),) if index else ())
            for index in reversed(range(count))))
        result = run(graph, resources(horizon=count))
        self.assertEqual(result.duration_ns, count)
        by_id = {task.task_id: task for task in result.tasks}
        self.assertEqual((by_id["stage:0"].start_ns, by_id["stage:1099"].end_ns), (0, count))

    def test_one_task_cannot_double_spend_one_supply(self):
        graph = Graph((Task("dup", "cpu", (WorkDemand("a", 2, "thread_cpu_ns"), WorkDemand("b", 2, "thread_cpu_ns"))),))
        with self.assertRaisesRegex(InvalidModel, "duplicate a shared supply"):
            run(graph, bindings=(binding("a", "cpu_supply"), binding("b", "cpu_supply")))

    def test_memory_domains_cannot_duplicate_one_available_budget(self):
        graph = Graph((), (MemoryDomain("one", "memory_budget", 1, 0), MemoryDomain("two", "memory_budget", 1, 0)))
        with self.assertRaisesRegex(InvalidModel, "shared memory"):
            run(graph, bindings=())

    def test_unknown_is_not_zero_and_efficiency_is_not_silently_clipped(self):
        with self.assertRaises(UnknownParameter):
            run(Graph((compute(),)), bindings=(binding("cpu_rate", "cpu_supply", None),))
        with self.assertRaises(UnknownParameter):
            run(Graph((compute(),)), resources(values=(("cpu_supply", None, "thread_cpu_ns/s"),)))
        with self.assertRaises(UnknownParameter):
            run(Graph((compute(),)), bindings=(binding("cpu_rate", "cpu_supply", 0),))
        for invalid in (Fraction(11, 10), -1, .5):
            with self.assertRaises(InvalidModel):
                binding("cpu_rate", "cpu_supply", invalid)

    def test_no_extrapolation_after_finite_horizon(self):
        with self.assertRaisesRegex(UnknownParameter, "extrapolation rejected"):
            run(Graph((compute(amount=11),)), resources(horizon=10))
        self.assertEqual(run(Graph((compute(amount=10),)), resources(horizon=10)).duration_ns, 10)
        with self.assertRaises(UnknownParameter):
            run(Graph((compute(amount=1),)), resources(horizon=10), start_ns=11)

    def test_explicit_window_does_not_use_later_recovery(self):
        with self.assertRaisesRegex(UnknownParameter, "known available-resource horizon"):
            run(Graph((compute(amount=1),)), resources(points=(ResourcePoint(0, {"cpu_supply": 0}),
                ResourcePoint(20, {"cpu_supply": FULL}))), end_ns=10)

    def test_supply_integral_is_exact_and_uses_same_clock(self):
        available = resources(points=(ResourcePoint(2, {"cpu_supply": 0}), ResourcePoint(5, {"cpu_supply": FULL // 2})))
        self.assertEqual(available.service_integral("cpu_supply", 0, 13), 6)
        self.assertEqual(available.service_integral("cpu_supply", 1, 6), Fraction(3, 2))
        with self.assertRaises(UnknownParameter):
            available.service_integral("cpu_supply", 0, 101)
        with self.assertRaises(InvalidModel):
            available.service_integral("memory_budget", 0, 1)

    def test_external_physical_supply_rejects_inclusive_or_instantaneous_rates(self):
        with self.assertRaises(InvalidModel):
            resources(values=(("inclusive", 100, "invocation/s"),))
        with self.assertRaises(InvalidModel):
            resources(values=(("instant", INSTANTANEOUS_RATE, "invocation/s"),))
        with self.assertRaises(InvalidModel):
            run(Graph((Task("inclusive", "cpu", (WorkDemand("response", 1, "invocation"),)),)), bindings=())

    def test_unknown_target_unit_mismatch_and_duplicate_target_are_rejected(self):
        with self.assertRaises(UnknownParameter):
            run(Graph((compute(),)), bindings=())
        with self.assertRaises(InvalidModel):
            run(Graph((compute(unit="flop"),)))
        with self.assertRaises(InvalidModel):
            run(Graph((compute(),)), bindings=(binding("cpu_rate", "cpu_supply"), binding("cpu_rate", "cpu_supply")))

    def test_diagnostic_sources_need_explicit_replay_and_never_qualify_accuracy(self):
        available = resources(source="observed_diagnostic")
        with self.assertRaisesRegex(InvalidModel, "explicit replay"):
            run(Graph((compute(),)), available)
        result = run(Graph((compute(),)), available, allow_observed_diagnostic=True)
        self.assertTrue(result.diagnostic_replay)
        self.assertFalse(result.accuracy_qualified)

    def test_serialization_retains_exact_values_unknowns_and_no_os_claim(self):
        available = resources(points=(ResourcePoint(2, {"cpu_supply": Fraction(FULL, 3)}),))
        restored = AvailableResources.from_dict(json.loads(json.dumps(available.to_dict())))
        self.assertEqual(restored.to_dict(), available.to_dict())
        item = binding("cpu_rate", "cpu_supply", Fraction(2, 3))
        self.assertEqual(ServiceBinding.from_dict(json.loads(json.dumps(item.to_dict()))), item)
        unknown = binding("cpu_rate", "cpu_supply", None)
        self.assertEqual(ServiceBinding.from_dict(unknown.to_dict()), unknown)
        result = run(Graph((compute(amount=2),)), restored, (item,))
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["accuracy_qualified"], False)


if __name__ == "__main__":
    unittest.main()
