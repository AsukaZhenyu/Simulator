"""Hand-calculated completion synchronization; no measured duration fitting."""
import json
import unittest

from llm_infer_model.performance import (
    BlockedExecution, CapacityExceeded, Graph, InvalidModel, MemoryDomain,
    MemoryReservation, Parameter, ParameterBundle, ResourcePoint, ResourceTimeline,
    Task, UnknownParameter, WorkDemand, evaluate,
)


def parameter(name, value, unit):
    return Parameter(name, value, unit, "declared", (), "illustrative completion synchronization")


def costs():
    return ParameterBundle((parameter("cpu", 1_000_000_000, "flop/s"),
        parameter("gpu", 1_000_000_000, "flop/s"), parameter("submit", 2, "ns"),
        parameter("five", 5, "ns"), parameter("capacity", 100, "byte")))


def work(name, resource, amount, *, depends_on=(), extra_resources=(), memory=()):
    rate = "gpu" if resource.startswith("gpu") else "cpu"
    return Task(name, resource, (WorkDemand(rate, amount, "flop"),), depends_on,
                extra_resources=extra_resources, memory=memory)


def pipeline(wait_resource="cpu", *, independent_host=False):
    tasks = [Task("submit", "cpu", fixed_parameter="submit"),
        Task("synchronize", wait_resource, depends_on=("submit",), waits_for=("device",)),
        work("device", "gpu", 10, depends_on=("submit",))]
    if independent_host:
        tasks.append(work("other_host", "cpu", 5, depends_on=("submit",)))
    tasks.append(work("resume", "cpu", 1, depends_on=("synchronize",)))
    return Graph(tuple(tasks))


def executions(result):
    return {row.task_id: row for row in result.tasks}


class CompletionWaitTests(unittest.TestCase):
    def test_polling_reservation_blocks_other_host_work_until_device_completion(self):
        result = evaluate(pipeline(independent_host=True), costs())
        rows = executions(result)
        self.assertEqual((rows["synchronize"].start_ns, rows["synchronize"].end_ns), (2, 12))
        self.assertEqual((rows["other_host"].ready_ns, rows["other_host"].start_ns, rows["other_host"].end_ns), (2, 12, 17))
        self.assertEqual(result.duration_ns, 18)
        self.assertTrue(any(row.task_id == "other_host" and row.reasons == ("resource_busy",)
                            for row in result.waits))
        self.assertTrue(any(row.task_id == "synchronize" and row.start_ns == 2 and row.end_ns == 12
                            and row.reasons == ("await_completion:device",) for row in result.waits))

    def test_blocking_uses_explicit_virtual_resource_releasing_cpu_for_other_work(self):
        result = evaluate(pipeline("host_wait", independent_host=True), costs())
        rows = executions(result)
        self.assertEqual((rows["other_host"].start_ns, rows["other_host"].end_ns), (2, 7))
        self.assertEqual((rows["synchronize"].start_ns, rows["synchronize"].end_ns), (2, 12))
        self.assertEqual((rows["resume"].start_ns, rows["resume"].end_ns), (12, 13))
        self.assertEqual(result.duration_ns, 13)

    def test_wait_overlaps_device_work_without_adding_same_gpu_duration_again(self):
        result = evaluate(pipeline(), costs())
        self.assertEqual(result.duration_ns, 2 + 10 + 1)
        self.assertEqual(executions(result)["synchronize"].duration_ns, 10)

    def test_gate_completion_is_fixed_point_independent_of_active_insertion_order(self):
        graph = Graph((Task("outer", "cpu", waits_for=("inner",)),
            Task("inner", "wait", waits_for=("device",)), work("device", "gpu", 10)))
        result = evaluate(graph, costs())
        self.assertEqual(result.duration_ns, 10)
        self.assertEqual({row.end_ns for row in result.tasks}, {10})
        self.assertEqual([event[2] for event in result.events if event[1] == "finish"], ["device", "inner", "outer"])

    def test_zero_device_work_releases_chained_gates_without_one_ns_advance(self):
        graph = Graph((Task("outer", "cpu", waits_for=("inner",)),
            Task("inner", "wait", waits_for=("device",)), work("device", "gpu", 0)))
        result = evaluate(graph, costs())
        self.assertEqual(result.duration_ns, 0)
        self.assertTrue(all(row.start_ns == row.end_ns == 0 for row in result.tasks))
        self.assertEqual(result.waits, ())

    def test_already_completed_target_does_not_delay_later_launched_gate(self):
        graph = Graph((work("device", "gpu", 0), Task("launch", "cpu", fixed_parameter="five"),
            Task("synchronize", "cpu", depends_on=("launch",), waits_for=("device",)),
            work("resume", "cpu", 1, depends_on=("synchronize",))))
        result = evaluate(graph, costs())
        row = executions(result)["synchronize"]
        self.assertEqual((row.ready_ns, row.start_ns, row.end_ns), (5, 5, 5))
        self.assertEqual(result.duration_ns, 6)

    def test_multiple_completion_targets_use_last_completion_and_named_wait_reasons(self):
        graph = Graph((Task("join", "cpu", waits_for=("device_a", "device_b")),
            work("device_a", "gpu", 10), work("device_b", "gpu2", 20)))
        result = evaluate(graph, costs())
        self.assertEqual(result.duration_ns, 20)
        waits = [row for row in result.waits if row.task_id == "join"]
        self.assertEqual([(row.start_ns, row.end_ns, row.reasons) for row in waits], [
            (0, 10, ("await_completion:device_a", "await_completion:device_b")),
            (10, 20, ("await_completion:device_b",))])

    def test_gpu_state_change_retains_progress_and_extends_synchronization(self):
        timeline = ResourceTimeline((ResourcePoint(7, {"gpu": 500_000_000}),))
        result = evaluate(pipeline(), costs(), timeline)
        rows = executions(result)
        self.assertEqual((rows["device"].start_ns, rows["device"].end_ns), (2, 17))
        self.assertEqual(rows["synchronize"].end_ns, 17)
        self.assertEqual(result.duration_ns, 18)

    def test_cpu_pause_after_join_retains_host_work_without_extending_finished_gpu_wait(self):
        graph = pipeline()
        graph = Graph(graph.tasks[:-1] + (work("resume", "cpu", 5, depends_on=("synchronize",)),))
        timeline = ResourceTimeline((ResourcePoint(13, {"cpu": 0}), ResourcePoint(20, {"cpu": 1_000_000_000})))
        result = evaluate(graph, costs(), timeline)
        rows = executions(result)
        self.assertEqual(rows["synchronize"].end_ns, 12)
        self.assertEqual((rows["resume"].start_ns, rows["resume"].end_ns), (12, 24))

    def test_cpu_submission_service_pause_moves_device_and_wait_launch(self):
        graph = Graph((work("submit", "cpu", 2),
            Task("synchronize", "cpu", depends_on=("submit",), waits_for=("device",)),
            work("device", "gpu", 10, depends_on=("submit",)),
            work("resume", "cpu", 1, depends_on=("synchronize",))))
        timeline = ResourceTimeline((ResourcePoint(0, {"cpu": 0}), ResourcePoint(5, {"cpu": 1_000_000_000})))
        result = evaluate(graph, costs(), timeline)
        self.assertEqual(executions(result)["synchronize"].start_ns, 7)
        self.assertEqual(result.duration_ns, 18)

    def test_extra_reservations_control_occupancy_without_resource_name_inference(self):
        graph = Graph((Task("wait", "arbitrary_virtual_name", extra_resources=("cpu",), waits_for=("device",)),
            work("device", "gpu", 10), work("other_host", "cpu", 5)))
        result = evaluate(graph, costs())
        self.assertEqual(executions(result)["other_host"].start_ns, 10)
        self.assertEqual(result.duration_ns, 15)

    def test_resource_deadlock_is_rejected_without_an_invented_gate_completion_clock(self):
        graph = Graph((Task("wait", "cpu", waits_for=("device",)),
            work("device", "gpu", 10, extra_resources=("cpu",))))
        with self.assertRaises(BlockedExecution):
            evaluate(graph, costs())

    def test_zero_device_service_without_recovery_cannot_complete_its_waiter(self):
        with self.assertRaises(BlockedExecution):
            evaluate(pipeline(), costs(), ResourceTimeline((ResourcePoint(0, {"gpu": 0}),)))

    def test_gate_releases_memory_to_closure_before_same_time_capacity_update(self):
        graph = Graph((Task("wait", "cpu", memory=(MemoryReservation("vram", 60),), waits_for=("device",)),
            work("device", "gpu", 10)), (MemoryDomain("vram", "capacity", 0, 0),))
        result = evaluate(graph, costs(), ResourceTimeline((ResourcePoint(10, external_memory={"vram": 50}),)))
        self.assertEqual(result.duration_ns, 10)
        with self.assertRaises(CapacityExceeded):
            evaluate(graph, costs(), ResourceTimeline((ResourcePoint(9, external_memory={"vram": 50}),)))

    def test_gate_requires_no_fabricated_latency_or_cpu_price_but_real_host_work_does(self):
        graph = Graph((Task("wait", "cpu", waits_for=("device",)), work("device", "gpu", 10)))
        only_device = ParameterBundle((parameter("gpu", 1_000_000_000, "flop/s"),))
        self.assertEqual(evaluate(graph, only_device).duration_ns, 10)
        with self.assertRaises(UnknownParameter):
            evaluate(Graph(graph.tasks + (work("resume", "cpu", 1, depends_on=("wait",)),)), only_device)


class CompletionWaitContractTests(unittest.TestCase):
    def test_legacy_task_document_and_positional_arguments_remain_unchanged(self):
        legacy = Task("task", "cpu", (WorkDemand("cpu", 1, "flop"),), (), None, (), ())
        expected = dict(id="task", resource="cpu", demands=[dict(parameter="cpu", amount=1, unit="flop")],
            depends_on=[], fixed_parameter=None, extra_resources=[], memory=[])
        self.assertEqual(legacy.to_dict(), expected)
        self.assertNotIn("waits_for", legacy.to_dict())
        self.assertEqual(Graph.from_dict(Graph((legacy,)).to_dict()), Graph((legacy,)))

    def test_wait_gate_json_roundtrip_preserves_separate_launch_and_completion_edges(self):
        graph = pipeline("wait")
        restored = Graph.from_dict(json.loads(json.dumps(graph.to_dict())))
        self.assertEqual(restored, graph)
        self.assertEqual(restored.tasks[1].depends_on, ("submit",))
        self.assertEqual(restored.tasks[1].waits_for, ("device",))

    def test_gate_cannot_hide_work_or_an_additional_measured_wait_price(self):
        for values in (dict(demands=(WorkDemand("cpu", 0, "flop"),)), dict(fixed_parameter="submit")):
            with self.subTest(values=values), self.assertRaises(InvalidModel):
                Task("wait", "cpu", waits_for=("device",), **values)
        with self.assertRaises(InvalidModel):
            Task("empty", "cpu")

    def test_unknown_self_and_combined_completion_launch_cycles_are_rejected(self):
        bad_graphs = (
            (Task("wait", "cpu", waits_for=("missing",)),),
            (Task("wait", "cpu", waits_for=("wait",)),),
            (Task("a", "cpu", waits_for=("b",)), Task("b", "wait", waits_for=("a",))),
            (Task("wait", "cpu", waits_for=("device",)), work("device", "gpu", 1, depends_on=("wait",))),
        )
        for tasks in bad_graphs:
            with self.subTest(tasks=tasks), self.assertRaises(InvalidModel):
                Graph(tasks)

    def test_wait_ids_and_json_types_are_strict(self):
        for ids in (("a", "a"), ["a"], (True,), ("",), "a", None):
            with self.subTest(ids=ids), self.assertRaises(InvalidModel):
                Task("wait", "cpu", waits_for=ids)
        document = pipeline().to_dict()
        document["tasks"][1]["waits_for"] = "device"
        with self.assertRaises(InvalidModel):
            Graph.from_dict(document)


if __name__ == "__main__":
    unittest.main()
