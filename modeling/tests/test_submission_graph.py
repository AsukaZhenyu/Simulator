"""Hand-derived asynchronous submission dependency tests; no hardware."""
from dataclasses import replace
import unittest

from llm_infer_model.performance.evaluator import evaluate
from llm_infer_model.performance.graph import Graph, ResourcePoint, ResourceTimeline, Task, WorkDemand
from llm_infer_model.performance.parameters import InvalidModel, Parameter, ParameterBundle, UnknownParameter
from llm_infer_model.performance.submission_graph import (
    CompletionWait, DeviceCommand, HostSubmission, SubmissionProgram,
    ZERO_EVENT_PARAMETER, build_submission_graph,
)


EVIDENCE = ("hand-derived asynchronous microprogram",)


def cpu(n):
    return (WorkDemand("cpu", n, "thread_cpu_ns"),)


def command(name="kernel", amount=5, *, stream="stream0", resource="gpu", deps=()):
    return DeviceCommand(name, stream, resource, (WorkDemand("gpu", amount, "flop"),), deps)


def launch(name="launch", prefix=2, suffix=8, device=None, acceptance="after_prefix"):
    return HostSubmission(name, device or command(), cpu(prefix), cpu(suffix), acceptance)


def program(*steps):
    return SubmissionProgram("host_cpu", steps, EVIDENCE, "explicit CPU submission and GPU stream")


def parameters(*, unknown_cpu=False):
    return ParameterBundle((Parameter("cpu", None if unknown_cpu else 10**9, "thread_cpu_ns/s", "declared", EVIDENCE, "one thread"),
                            Parameter("gpu", 10**9, "flop/s", "declared", EVIDENCE, "declared device work")))


class SubmissionGraphTests(unittest.TestCase):
    def test_acceptance_before_return_allows_overlap(self):
        compiled = build_submission_graph(program(launch()))
        result = compiled.evaluate(parameters())
        rows = {r.task_id:r for r in result.tasks}
        boundary = compiled.boundaries[0]
        self.assertEqual(rows[boundary.acceptance_task].end_ns, 2)
        self.assertEqual((rows[boundary.device_task].start_ns, rows[boundary.device_task].end_ns), (2,7))
        self.assertEqual(rows[boundary.return_task].end_ns, 10)
        self.assertEqual(result.duration_ns, 10)
        self.assertFalse(compiled.hardware_accuracy_qualified)

    def test_cpu_blackout_after_enqueue_does_not_pause_gpu(self):
        compiled = build_submission_graph(program(launch()))
        timeline = ResourceTimeline((ResourcePoint(3,{"cpu":0}),ResourcePoint(20,{"cpu":10**9})))
        result = compiled.evaluate(parameters(), timeline)
        rows = {r.task_id:r for r in result.tasks}
        self.assertEqual(rows['device:kernel'].end_ns,7)
        self.assertEqual(rows['host:launch:return'].end_ns,27)
        self.assertTrue(any('service_unavailable:cpu' in w.reasons for w in result.waits))

    def test_device_blackout_does_not_pause_host_return(self):
        compiled = build_submission_graph(program(launch()))
        timeline = ResourceTimeline((ResourcePoint(3,{"gpu":0}),ResourcePoint(20,{"gpu":10**9})))
        result = compiled.evaluate(parameters(),timeline)
        rows = {r.task_id:r for r in result.tasks}
        self.assertEqual(rows['host:launch:return'].end_ns,10)
        self.assertEqual(rows['device:kernel'].end_ns,24)

    def test_stream_order_is_explicit_even_across_device_lanes(self):
        p = program(launch('a',1,0,command('first',5,resource='gpu0')),
                    launch('b',1,0,command('second',3,resource='gpu1')))
        rows={r.task_id:r for r in build_submission_graph(p).evaluate(parameters()).tasks}
        self.assertEqual((rows['device:first'].start_ns,rows['device:first'].end_ns),(1,6))
        self.assertEqual((rows['device:second'].start_ns,rows['device:second'].end_ns),(6,9))

    def test_different_streams_can_overlap_on_declared_independent_lanes(self):
        p=program(launch('a',1,0,command('first',5,stream='s0',resource='gpu0')),
                  launch('b',1,0,command('second',3,stream='s1',resource='gpu1')))
        rows={r.task_id:r for r in build_submission_graph(p).evaluate(parameters()).tasks}
        self.assertEqual((rows['device:second'].start_ns,rows['device:second'].end_ns),(2,5))

    def test_cross_stream_dependency_is_explicit(self):
        p=program(launch('a',1,0,command('first',5,stream='s0',resource='gpu0')),
                  launch('b',1,0,command('second',3,stream='s1',resource='gpu1',deps=('first',))))
        rows={r.task_id:r for r in build_submission_graph(p).evaluate(parameters()).tasks}
        self.assertEqual(rows['device:second'].start_ns,6)

    def test_blocked_wait_releases_cpu_and_resumes_after_device(self):
        p=program(launch(prefix=2,suffix=0,device=command(amount=10)),
                  CompletionWait('sync',('kernel',),cpu(1),cpu(2)))
        compiled=build_submission_graph(p)
        extra=Task('other_cpu','host_cpu',cpu(4),('host:sync:wait_entry',))
        graph=Graph(compiled.graph.tasks+(extra,))
        bundle=ParameterBundle(parameters().parameters+compiled.structural_parameters.parameters)
        rows={r.task_id:r for r in evaluate(graph,bundle).tasks}
        self.assertEqual((rows['other_cpu'].start_ns,rows['other_cpu'].end_ns),(3,7))
        self.assertEqual((rows['host:sync:completion_gate'].start_ns,rows['host:sync:completion_gate'].end_ns),(3,12))
        self.assertEqual((rows['host:sync:continuation'].start_ns,rows['host:sync:continuation'].end_ns),(12,14))

    def test_explicit_spin_occupancy_holds_cpu_without_second_gpu_charge(self):
        p=program(launch(prefix=2,suffix=0,device=command(amount=10)),
                  CompletionWait('sync',('kernel',),cpu(1),cpu(2),'spin_occupancy'))
        compiled=build_submission_graph(p)
        extra=Task('other_cpu','host_cpu',cpu(4),('host:sync:wait_entry',))
        bundle=ParameterBundle(parameters().parameters+compiled.structural_parameters.parameters)
        rows={r.task_id:r for r in evaluate(Graph(compiled.graph.tasks+(extra,)),bundle).tasks}
        self.assertGreaterEqual(rows['other_cpu'].start_ns,12)

    def test_entry_return_boundary_scenarios_are_explicit(self):
        p=program(launch())
        entry=build_submission_graph(p,acceptance_override='api_entry').evaluate(parameters())
        returned=build_submission_graph(p,acceptance_override='api_return').evaluate(parameters())
        self.assertEqual(next(t for t in entry.tasks if t.task_id=='device:kernel').start_ns,0)
        self.assertEqual(next(t for t in returned.tasks if t.task_id=='device:kernel').start_ns,10)
        self.assertEqual((entry.duration_ns,returned.duration_ns),(10,15))

    def test_unknown_profile_does_not_become_zero(self):
        with self.assertRaises(UnknownParameter):
            build_submission_graph(program(launch())).evaluate(parameters(unknown_cpu=True))

    def test_inclusive_host_response_cannot_double_charge_device(self):
        with self.assertRaisesRegex(InvalidModel,'inclusive host'):
            HostSubmission('bad',command(),(WorkDemand('price',1,'invocation'),),cpu(1),'api_return')

    def test_no_guessed_empty_prefix_or_acceptance(self):
        with self.assertRaises(InvalidModel):HostSubmission('bad',command(),(),cpu(1),'after_prefix')
        with self.assertRaises(InvalidModel):replace(launch(),acceptance_boundary=None)
        with self.assertRaises(TypeError):HostSubmission('bad',command(),cpu(1),cpu(1))

    def test_unsubmitted_wait_and_duplicate_command_rejected(self):
        with self.assertRaises(InvalidModel):program(CompletionWait('sync',('missing',),cpu(0),cpu(0)))
        with self.assertRaises(InvalidModel):program(launch('a'),launch('b'))
        with self.assertRaises(InvalidModel):program(launch(device=command(deps=('future',))))

    def test_device_cannot_lock_host_or_invalid_wait_mode(self):
        with self.assertRaises(InvalidModel):program(launch(device=command(resource='host_cpu')))
        with self.assertRaises(InvalidModel):CompletionWait('bad',('kernel',),cpu(1),cpu(1),'implicit')

    def test_caller_cannot_override_structural_zero(self):
        bundle=ParameterBundle(parameters().parameters+(Parameter(ZERO_EVENT_PARAMETER,10,'ns','declared',EVIDENCE,'bad'),))
        with self.assertRaises(InvalidModel):build_submission_graph(program(launch())).evaluate(bundle)


if __name__=='__main__':unittest.main()
