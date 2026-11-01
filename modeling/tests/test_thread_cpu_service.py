"""Hand-derived per-thread CPU service, not measured latency fitting."""
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance import (
    BlockedExecution, Graph, INSTANTANEOUS_RATE, InvalidModel, Parameter,
    ParameterBundle, ResourcePoint, ResourceTimeline, Task, UnknownParameter,
    WorkDemand, evaluate,
)

MS = 1_000_000
FULL = 1_000_000_000


def parameter(name, value, unit="thread_cpu_ns/s"):
    return Parameter(name, value, unit, "declared", (), "hand-derived one-thread supply; no hardware accuracy claim")


def cpu_task(name="host", work=10 * MS, rate="host_supply", *, depends_on=()):
    return Task(name, "thread:" + name, (WorkDemand(rate, work, "thread_cpu_ns"),), depends_on)


def pause(rate="host_supply"):
    return ResourceTimeline((ResourcePoint(2 * MS, {rate: 0}), ResourcePoint(7 * MS, {rate: FULL})))


class ThreadCpuServiceTests(unittest.TestCase):
    def test_ten_ms_work_paused_at_two_until_seven_completes_at_fifteen(self):
        result = evaluate(Graph((cpu_task(),)), ParameterBundle((parameter("host_supply", FULL),)), pause())
        self.assertEqual(result.duration_ns, 15 * MS)
        self.assertEqual([(row.task_id,row.start_ns,row.end_ns,row.reasons) for row in result.waits],
            [("host",2 * MS,7 * MS,("service_unavailable:host_supply",))])
        self.assertEqual(result.tasks[0].waiting_ns, 0)  # Audit pause is during execution, not before launch.

    def test_half_supply_requires_twenty_ms_without_inventing_ready_queue_delay(self):
        result = evaluate(Graph((cpu_task(),)), ParameterBundle((parameter("host_supply", FULL // 2),)))
        self.assertEqual(result.duration_ns,20 * MS)
        self.assertEqual(result.waits,())

    def test_fractional_supply_keeps_exact_work_progress(self):
        result=evaluate(Graph((cpu_task(work=1),)),ParameterBundle((parameter("host_supply",Fraction(FULL,3)),)))
        self.assertEqual(result.duration_ns,3)

    def test_independent_gpu_does_not_pause_when_cpu_supply_disappears(self):
        graph=Graph((cpu_task(),Task("device","gpu",(WorkDemand("gpu_rate",12 * MS,"flop"),))))
        costs=ParameterBundle((parameter("host_supply",FULL),parameter("gpu_rate",FULL,"flop/s")))
        result=evaluate(graph,costs,pause())
        rows={row.task_id:row for row in result.tasks}
        self.assertEqual(rows["device"].end_ns,12 * MS)
        self.assertEqual(rows["host"].end_ns,15 * MS)

    def test_completion_gate_and_resumption_cpu_service_are_separate(self):
        graph=Graph((Task("device","gpu",(WorkDemand("gpu_rate",5 * MS,"flop"),)),
            Task("wait","virtual-wait",waits_for=("device",)),
            cpu_task("resume",3 * MS,depends_on=("wait",))))
        costs=ParameterBundle((parameter("host_supply",0),parameter("gpu_rate",FULL,"flop/s")))
        result=evaluate(graph,costs,ResourceTimeline((ResourcePoint(9 * MS,{"host_supply":FULL}),)))
        rows={row.task_id:row for row in result.tasks}
        self.assertEqual(rows["device"].end_ns,5 * MS)
        self.assertEqual(rows["wait"].end_ns,5 * MS)
        self.assertEqual((rows["resume"].start_ns,rows["resume"].end_ns),(5 * MS,12 * MS))
        self.assertIn((5 * MS,9 * MS,("service_unavailable:host_supply",)),
            [(row.start_ns,row.end_ns,row.reasons) for row in result.waits if row.task_id=="resume"])

    def test_cpu_submission_delay_moves_device_launch_without_fixed_total_offset(self):
        graph=Graph((cpu_task("submit",4 * MS),
            Task("device","gpu",(WorkDemand("gpu_rate",3 * MS,"flop"),),depends_on=("submit",))))
        costs=ParameterBundle((parameter("host_supply",FULL),parameter("gpu_rate",FULL,"flop/s")))
        result=evaluate(graph,costs,pause())
        rows={row.task_id:row for row in result.tasks}
        self.assertEqual(rows["submit"].end_ns,9 * MS)
        self.assertEqual((rows["device"].start_ns,rows["device"].end_ns),(9 * MS,12 * MS))

    def test_fixed_wall_latency_is_not_cpu_work_and_does_not_pause(self):
        graph=Graph((Task("wall","wall-timer",fixed_parameter="latency"),))
        costs=ParameterBundle((parameter("latency",10 * MS,"ns"),parameter("host_supply",FULL)))
        result=evaluate(graph,costs,pause())
        self.assertEqual(result.duration_ns,10 * MS)
        self.assertEqual(result.waits,())

    def test_separate_threads_require_explicit_supplies_without_automatic_global_sharing(self):
        graph=Graph((cpu_task("first",rate="first_supply"),cpu_task("second",rate="second_supply")))
        both=ParameterBundle((parameter("first_supply",FULL),parameter("second_supply",FULL)))
        self.assertEqual(evaluate(graph,both).duration_ns,10 * MS)
        half=ParameterBundle((parameter("first_supply",FULL // 2),parameter("second_supply",FULL // 2)))
        self.assertEqual(evaluate(graph,half).duration_ns,20 * MS)

    def test_new_unit_cannot_be_an_unbounded_aggregate_core_rate(self):
        for value in (FULL + 1,Fraction(2 * FULL)):
            with self.assertRaisesRegex(InvalidModel,"one thread"):
                parameter("host_supply",value)
        with self.assertRaisesRegex(InvalidModel,"one thread"):
            evaluate(Graph((cpu_task(),)),ParameterBundle((parameter("host_supply",FULL),)),
                ResourceTimeline((ResourcePoint(0,{"host_supply":FULL + 1}),)))

    def test_missing_or_unknown_cpu_supply_is_rejected(self):
        for costs in (ParameterBundle(()),ParameterBundle((parameter("host_supply",None),))):
            with self.assertRaises(UnknownParameter): evaluate(Graph((cpu_task(),)),costs)

    def test_unknown_unit_aliases_or_wall_ns_cannot_substitute_for_cpu_service(self):
        with self.assertRaises(InvalidModel): WorkDemand("cpu",1,"cpu_ns")
        with self.assertRaises(InvalidModel): parameter("cpu",FULL,"cpu_ns/s")
        with self.assertRaises(InvalidModel):
            evaluate(Graph((cpu_task(),)),ParameterBundle((parameter("host_supply",10,"ns"),)))

    def test_instantaneous_rate_cannot_create_physical_thread_service(self):
        with self.assertRaises(InvalidModel): parameter("host_supply",INSTANTANEOUS_RATE)
        with self.assertRaises(InvalidModel):
            evaluate(Graph((cpu_task(),)),ParameterBundle((parameter("host_supply",FULL),)),
                ResourceTimeline((ResourcePoint(0,{"host_supply":INSTANTANEOUS_RATE}),)))

    def test_zero_service_without_declared_recovery_blocks_instead_of_completing(self):
        with self.assertRaises(BlockedExecution):
            evaluate(Graph((cpu_task(),)),ParameterBundle((parameter("host_supply",0),)))

    def test_zero_work_finishes_without_a_spurious_service_unavailable_interval(self):
        result=evaluate(Graph((cpu_task(work=0),)),ParameterBundle((parameter("host_supply",0),)))
        self.assertEqual(result.duration_ns,0);self.assertEqual(result.waits,())

    def test_legacy_flop_pause_wait_inventory_remains_unchanged(self):
        graph=Graph((Task("legacy","cpu",(WorkDemand("legacy_rate",10 * MS,"flop"),)),))
        result=evaluate(graph,ParameterBundle((parameter("legacy_rate",FULL,"flop/s"),)),pause("legacy_rate"))
        self.assertEqual(result.duration_ns,15 * MS);self.assertEqual(result.waits,())

    def test_cpu_service_graph_parameters_and_timeline_round_trip_exactly(self):
        graph=Graph((cpu_task(),));costs=ParameterBundle((parameter("host_supply",Fraction(FULL,2)),));timeline=pause()
        actual=evaluate(Graph.from_dict(json.loads(json.dumps(graph.to_dict()))),
            ParameterBundle.from_dict(json.loads(json.dumps(costs.to_dict()))),
            ResourceTimeline.from_dict(json.loads(json.dumps(timeline.to_dict()))))
        self.assertEqual(actual,evaluate(graph,costs,timeline))


if __name__=="__main__": unittest.main()
