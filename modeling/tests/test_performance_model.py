"""Hand-calculated physical invariants; no local GPU or calibration required."""
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance import (
    BlockedExecution, CapacityExceeded, Graph, InvalidModel, MemoryDomain,
    MemoryReservation, Parameter, ParameterBundle, ResourcePoint, ResourceTimeline,
    Task, UnknownParameter, WorkDemand, analytical_compute_duration_ns,
    analytical_transfer_duration_ns, evaluate,
)


def parameter(name, value, unit):
    return Parameter(name, value, unit, "declared", (), "test hardware/backend/shape")


def bundle():
    return ParameterBundle((parameter("cpu", 1_000_000_000, "flop/s"),
        parameter("gpu", 2_000_000_000, "flop/s"),
        parameter("dram", 4_000_000_000, "byte/s"),
        parameter("pcie", 1_000_000_000, "byte/s"),
        parameter("submit", 5, "ns"), parameter("tiny", Fraction(1, 2), "ns"),
        parameter("vram", 100, "byte")))


def compute(name, resource, amount, dependencies=(), memory=(), extra=()):
    return Task(name, resource, (WorkDemand(resource, amount, "flop"),),
                dependencies, extra_resources=extra, memory=memory)


class ParameterContractTests(unittest.TestCase):
    def test_rational_json_roundtrip_preserves_scope_and_evidence(self):
        source = ParameterBundle((Parameter("gpu", Fraction(5, 3), "flop/s",
            "profiled", ("sha256:example:exclusive-device-work",), "F32 matrix family"),))
        restored = ParameterBundle.from_dict(json.loads(json.dumps(source.to_dict())))
        self.assertEqual(restored, source)
        self.assertEqual(restored.require("gpu", "flop/s"), Fraction(5, 3))

    def test_unknown_is_not_zero(self):
        unknown = ParameterBundle((parameter("gpu", None, "flop/s"),))
        with self.assertRaises(UnknownParameter):
            unknown.require("gpu", "flop/s")
        with self.assertRaises(UnknownParameter):
            bundle().require("missing", "ns")
        self.assertIsNone(unknown.to_dict()["parameters"][0]["value"])

    def test_wrong_units_fail_instead_of_silent_conversion(self):
        with self.assertRaises(InvalidModel):
            bundle().require("gpu", "byte/s")
        with self.assertRaises(InvalidModel):
            evaluate(Graph((Task("copy", "pcie", (WorkDemand("gpu", 1, "byte"),)),)), bundle())

    def test_profiled_evidence_is_required(self):
        with self.assertRaises(InvalidModel):
            Parameter("gpu", 1, "flop/s", "profiled", (), "test")

    def test_bad_values_and_unsupported_units_rejected(self):
        for value in (-1, True, 1.2, float("nan"), "1"):
            with self.subTest(value=value), self.assertRaises(InvalidModel):
                parameter("gpu", value, "flop/s")
        with self.assertRaises(InvalidModel):
            parameter("capacity", Fraction(1, 2), "byte")
        with self.assertRaises(InvalidModel):
            parameter("utilization", 1, "percent")

    def test_duplicate_version_and_schema_rejected(self):
        with self.assertRaises(InvalidModel):
            ParameterBundle((parameter("x", 1, "ns"), parameter("x", 2, "ns")))
        doc = bundle().to_dict()
        doc["schema_version"] = "unknown"
        with self.assertRaises(InvalidModel):
            ParameterBundle.from_dict(doc)

    def test_malformed_json_types_raise_model_errors(self):
        for field in ("unit", "origin"):
            doc = bundle().to_dict()
            doc["parameters"][0][field] = []
            with self.subTest(field=field), self.assertRaises(InvalidModel):
                ParameterBundle.from_dict(doc)
        with self.assertRaises(InvalidModel):
            WorkDemand("gpu", 1, [])
        doc = bundle().to_dict()
        doc["extra"] = "ignored?"
        with self.assertRaises(InvalidModel):
            ParameterBundle.from_dict(doc)
        doc = bundle().to_dict()
        doc["parameters"][0]["value"]["denominator"] = 0
        with self.assertRaises(InvalidModel):
            ParameterBundle.from_dict(doc)


class PhysicalExecutionTests(unittest.TestCase):
    def test_roofline_work_finishes_at_slowest_demand(self):
        task = Task("gpu", "gpu", (WorkDemand("gpu", 20, "flop"),
                                   WorkDemand("dram", 80, "byte")))
        self.assertEqual(evaluate(Graph((task,)), bundle()).duration_ns, 20)

    def test_independent_resources_overlap_and_join_uses_max(self):
        graph = Graph((compute("cpu", "cpu", 30), compute("gpu", "gpu", 40),
                       Task("join", "host", depends_on=("cpu", "gpu"), fixed_parameter="submit")))
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 35)
        self.assertEqual([t.start_ns for t in result.tasks], [0, 0, 30])
        self.assertEqual(result.tasks[2].ready_ns, 30)

    def test_fixed_submit_does_not_scale_with_gpu_availability(self):
        graph = Graph((Task("submit", "host", fixed_parameter="submit"),
                       compute("gpu", "gpu", 20, ("submit",))))
        slow = ResourceTimeline((ResourcePoint(0, {"gpu": 1_000_000_000}),))
        result = evaluate(graph, bundle(), slow)
        self.assertEqual(result.duration_ns, 25)
        self.assertEqual(result.tasks[0].duration_ns, 5)
        self.assertEqual(result.tasks[1].duration_ns, 20)

    def test_mid_operation_rate_change_retains_completed_work(self):
        timeline = ResourceTimeline((ResourcePoint(5, {"gpu": 1_000_000_000}),))
        result = evaluate(Graph((compute("gpu", "gpu", 20),)), bundle(), timeline)
        self.assertEqual(result.duration_ns, 15)  # 10 FLOP before 5 ns, 10 afterwards.

    def test_zero_rate_pauses_then_recovers(self):
        timeline = ResourceTimeline((ResourcePoint(5, {"gpu": 0}),
                                     ResourcePoint(20, {"gpu": 2_000_000_000})))
        self.assertEqual(evaluate(Graph((compute("gpu", "gpu", 20),)),
                                  bundle(), timeline).duration_ns, 25)

    def test_no_recovery_is_explicit_blocked(self):
        timeline = ResourceTimeline((ResourcePoint(0, {"gpu": 0}),))
        with self.assertRaises(BlockedExecution):
            evaluate(Graph((compute("gpu", "gpu", 20),)), bundle(), timeline)

    def test_fractional_service_survives_many_state_boundaries(self):
        one = ParameterBundle((parameter("gpu", Fraction(1_000_000_000, 3), "flop/s"),))
        timeline = ResourceTimeline(tuple(ResourcePoint(i, {"gpu": Fraction(1_000_000_000, 3)})
                                          for i in range(1, 9)))
        self.assertEqual(evaluate(Graph((compute("g", "gpu", 3),)), one, timeline).duration_ns, 9)

    def test_compute_and_copy_rates_change_independently(self):
        graph = Graph((compute("gpu", "gpu", 20),
                       Task("copy", "pcie", (WorkDemand("pcie", 20, "byte"),))))
        timeline = ResourceTimeline((ResourcePoint(0, {"gpu": 0}),
                                     ResourcePoint(30, {"gpu": 2_000_000_000})))
        result = evaluate(graph, bundle(), timeline)
        self.assertEqual([t.end_ns for t in result.tasks], [40, 20])

    def test_serial_resource_queues_and_records_wait_without_adding_it_twice(self):
        graph = Graph((compute("a", "cpu", 10), compute("b", "cpu", 20)))
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 30)
        self.assertEqual(result.tasks[1].waiting_ns, 10)
        self.assertEqual(result.waits[0].reasons, ("resource_busy",))
        self.assertEqual(sum(t.duration_ns for t in result.tasks), 30)

    def test_explicit_shared_resource_serializes_otherwise_independent_engines(self):
        graph = Graph((compute("cpu", "cpu", 10, extra=("memory_bus",)),
                       compute("gpu", "gpu", 20, extra=("memory_bus",))))
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 20)
        self.assertEqual([t.start_ns for t in result.tasks], [0, 10])

    def test_dependent_serial_components_are_added_once(self):
        graph = Graph((Task("copy", "pcie", (WorkDemand("pcie", 10, "byte"),)),
                       compute("cpu", "cpu", 20, ("copy",)),
                       compute("gpu", "gpu", 40, ("cpu",))))
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 50)
        self.assertEqual([t.ready_ns for t in result.tasks], [0, 10, 30])

    def test_zero_work_does_not_need_service_time_but_parameter_remains_required(self):
        self.assertEqual(evaluate(Graph((compute("g", "gpu", 0),)), bundle()).duration_ns, 0)
        with self.assertRaises(UnknownParameter):
            evaluate(Graph((compute("g", "missing", 0),)), bundle())

    def test_fixed_fractional_duration_rounds_once_to_integer_ns(self):
        result = evaluate(Graph((Task("tiny", "host", fixed_parameter="tiny"),)), bundle())
        self.assertEqual(result.duration_ns, 1)

    def test_empty_graph_has_zero_time(self):
        self.assertEqual(evaluate(Graph(()), bundle()).duration_ns, 0)


class MemoryTests(unittest.TestCase):
    def domains(self, resident=30, external=0):
        return (MemoryDomain("vram", "vram", resident, external),)

    def test_resident_and_workspace_capacity_is_separate_from_rate(self):
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 70),)),),
                      self.domains())
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 10)

    def test_capacity_waits_for_external_memory_release(self):
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 70),)),),
                      self.domains(external=1))
        timeline = ResourceTimeline((ResourcePoint(7, external_memory={"vram": 0}),))
        result = evaluate(graph, bundle(), timeline)
        self.assertEqual(result.duration_ns, 17)
        self.assertEqual(result.tasks[0].waiting_ns, 7)
        self.assertEqual(result.waits[0].reasons, ("memory_capacity",))

    def test_capacity_drop_during_execution_is_not_silent_eviction(self):
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 70),)),),
                      self.domains())
        with self.assertRaises(CapacityExceeded):
            evaluate(graph, bundle(), ResourceTimeline((ResourcePoint(5, capacities={"vram": 99}),)))

    def test_workspace_released_before_capacity_drop_at_exact_completion(self):
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 70),)),
                       Task("end", "host", depends_on=("g",), fixed_parameter="submit")), self.domains())
        result = evaluate(graph, bundle(), ResourceTimeline((ResourcePoint(10, capacities={"vram": 30}),)))
        self.assertEqual(result.duration_ns, 15)

    def test_parallel_workspace_contention_waits_for_completion(self):
        graph = Graph((compute("c", "cpu", 10, memory=(MemoryReservation("vram", 60),)),
                       compute("g", "gpu", 20, memory=(MemoryReservation("vram", 60),))), self.domains())
        result = evaluate(graph, bundle())
        self.assertEqual(result.duration_ns, 20)
        self.assertEqual(result.tasks[1].start_ns, 10)

    def test_resident_impossible_and_unrecoverable_workspace_fail(self):
        with self.assertRaises(CapacityExceeded):
            evaluate(Graph((), self.domains(resident=101)), bundle())
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 71),)),), self.domains())
        with self.assertRaises(BlockedExecution):
            evaluate(graph, bundle())

    def test_extra_capacity_occupancy_does_not_automatically_slow_compute(self):
        graph = Graph((compute("g", "gpu", 20),), self.domains())
        timeline = ResourceTimeline((ResourcePoint(0, external_memory={"vram": 50}),))
        self.assertEqual(evaluate(graph, bundle(), timeline).duration_ns, 10)


class InputAndLegacyTests(unittest.TestCase):
    def test_graph_and_timeline_json_roundtrip(self):
        graph = Graph((compute("g", "gpu", 20, memory=(MemoryReservation("vram", 10),)),),
                      (MemoryDomain("vram", "vram", 20, 0),))
        timeline = ResourceTimeline((ResourcePoint(5, {"gpu": Fraction(1_000_000_000, 3)},
                                                             {"vram": 90}, {"vram": 10}),))
        self.assertEqual(Graph.from_dict(json.loads(json.dumps(graph.to_dict()))), graph)
        restored = ResourceTimeline.from_dict(json.loads(json.dumps(timeline.to_dict())))
        self.assertEqual(restored, timeline)

    def test_cycle_missing_dependency_and_unknown_memory_rejected(self):
        for tasks in ((compute("a", "cpu", 1, ("a",)),),
                      (compute("a", "cpu", 1, ("b",)),),
                      (compute("a", "cpu", 1, memory=(MemoryReservation("x", 1),)),)):
            with self.subTest(tasks=tasks), self.assertRaises(InvalidModel):
                Graph(tasks)

    def test_timeline_cannot_override_fixed_latency_or_unknown_domain(self):
        for point in (ResourcePoint(0, {"submit": 1}), ResourcePoint(0, capacities={"unknown": 1})):
            with self.subTest(point=point), self.assertRaises(InvalidModel):
                evaluate(Graph((compute("g", "gpu", 1),)), bundle(), ResourceTimeline((point,)))

    def test_timeline_is_strict_and_immutable(self):
        source = {"gpu": 1}
        point = ResourcePoint(0, source)
        source["gpu"] = 100
        self.assertEqual(point.rates["gpu"], 1)
        with self.assertRaises(TypeError):
            point.rates["gpu"] = 2
        with self.assertRaises(InvalidModel):
            ResourceTimeline((point, point))
        with self.assertRaises(InvalidModel):
            ResourcePoint(0, {"gpu": .5})

    def test_task_cannot_mix_fixed_duration_and_physical_demands(self):
        with self.assertRaises(InvalidModel):
            Task("bad", "gpu", (WorkDemand("gpu", 1, "flop"),), fixed_parameter="submit")
        with self.assertRaises(InvalidModel):
            Task("bad", "gpu")

    def test_legacy_compute_rounding_and_launch_cost(self):
        self.assertEqual(analytical_compute_duration_ns(flops=1, traffic_bytes=3,
            flops_per_second=2_000_000_000, memory_bytes_per_second=2_000_000_000,
            launch_overhead_ns=5), 7)
        self.assertEqual(analytical_compute_duration_ns(flops=0, traffic_bytes=0,
            flops_per_second=1, memory_bytes_per_second=1, launch_overhead_ns=0), 1)

    def test_legacy_copy_rounding_and_startup(self):
        self.assertEqual(analytical_transfer_duration_ns(storage_bytes=3,
            bytes_per_second=2_000_000_000, latency_ns=5), 7)
        for rate in (None, 0, True, float("inf"), -1):
            with self.subTest(rate=rate), self.assertRaises(InvalidModel):
                analytical_transfer_duration_ns(storage_bytes=1, bytes_per_second=rate, latency_ns=0)


if __name__ == "__main__":
    unittest.main()
