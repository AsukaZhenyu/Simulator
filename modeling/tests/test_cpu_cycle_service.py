"""Raw-cycle conservation with explicit service, never cycles-to-time fitting."""
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance import (
    BlockedExecution, Graph, INSTANTANEOUS_RATE, InvalidModel, Parameter,
    ParameterBundle, ResourcePoint, ResourceTimeline, Task, UnknownParameter,
    WorkDemand, evaluate,
)
from llm_infer_model.performance.available_resources import AvailableResources, ServiceBinding


def rate(name='cpu',value=1_000_000_000,unit='cpu_cycle/s'):
    return Parameter(name,value,unit,'declared',(),'scoped raw counter work; no universal capacity claim')


def cpu(amount=10):
    return Task('cpu-work','host',(WorkDemand('cpu',amount,'cpu_cycle'),))


class CpuCycleServiceTests(unittest.TestCase):
    def test_cycle_progress_is_conserved_across_pause_and_new_rate(self):
        timeline=ResourceTimeline((ResourcePoint(2,{'cpu':0}),ResourcePoint(7,{'cpu':2_000_000_000})))
        result=evaluate(Graph((cpu(),)),ParameterBundle((rate(),)),timeline)
        # 2 cycles before the pause, then 8 cycles at 2 cycles/ns.
        self.assertEqual(result.duration_ns,11)
        self.assertEqual([(w.start_ns,w.end_ns,w.reasons) for w in result.waits],[(2,7,('service_unavailable:cpu',))])

    def test_fractional_cycle_rate_is_exact_and_not_limited_to_one_ghz(self):
        self.assertEqual(evaluate(Graph((cpu(3),)),ParameterBundle((rate(value=Fraction(3_000_000_000,2)),))).duration_ns,2)
        self.assertEqual(evaluate(Graph((cpu(9),)),ParameterBundle((rate(value=9_000_000_000),))).duration_ns,1)

    def test_cpu_blackout_does_not_stop_already_submitted_device_work(self):
        graph=Graph((Task('submit','host',(WorkDemand('cpu',2,'cpu_cycle'),)),
            Task('device','gpu',(WorkDemand('gpu',6,'flop'),),depends_on=('submit',)),
            Task('host-runtime','host',(WorkDemand('cpu',3,'cpu_cycle'),),depends_on=('submit',)),
            Task('join','completion',depends_on=('host-runtime',),waits_for=('device',)),
            Task('notify','host',(WorkDemand('cpu',1,'cpu_cycle'),),depends_on=('join',))))
        timeline=ResourceTimeline((ResourcePoint(3,{'cpu':0}),ResourcePoint(9,{'cpu':1_000_000_000})))
        result=evaluate(graph,ParameterBundle((rate(),rate('gpu',1_000_000_000,'flop/s'))),timeline)
        rows={r.task_id:r for r in result.tasks}
        self.assertEqual((rows['device'].start_ns,rows['device'].end_ns),(2,8))
        self.assertEqual(rows['host-runtime'].end_ns,11)
        self.assertEqual(result.duration_ns,12)

    def test_cycle_work_cannot_consume_flop_or_thread_time_rates(self):
        for unit in ('flop/s','thread_cpu_ns/s','ns'):
            with self.subTest(unit=unit),self.assertRaises(InvalidModel):
                evaluate(Graph((cpu(),)),ParameterBundle((rate(unit=unit),)))
        with self.assertRaises(InvalidModel):WorkDemand('cpu',1,'cycles')
        with self.assertRaises(InvalidModel):rate(unit='cycle/s')

    def test_unknown_and_instantaneous_cycle_supply_are_not_replaced(self):
        for bundle in (ParameterBundle(()),ParameterBundle((rate(value=None),))):
            with self.assertRaises(UnknownParameter):evaluate(Graph((cpu(),)),bundle)
        with self.assertRaises(InvalidModel):rate(value=INSTANTANEOUS_RATE)
        with self.assertRaises(BlockedExecution):evaluate(Graph((cpu(),)),ParameterBundle((rate(value=0),)))

    def test_external_cycle_integral_and_efficiency_use_common_evaluator(self):
        available=AvailableResources(ParameterBundle((rate('external',2_000_000_000),)),
            ResourceTimeline((ResourcePoint(2,{'external':0}),ResourcePoint(7,{'external':4_000_000_000}))),
            100,'hand-derived','case-zero','declared_scenario',(),'raw counter supply; not elapsed CPU ns')
        self.assertEqual(available.service_integral('external',0,11),20)
        binding=ServiceBinding('cpu','external',Fraction(1,2),'declared',(),'explicit same-counter cycle efficiency')
        result=available.evaluate(Graph((cpu(),)),ParameterBundle(()),(binding,))
        self.assertEqual(result.duration_ns,11)

    def test_cycle_parameters_and_work_round_trip_without_conversion(self):
        graph=Graph((cpu(),));bundle=ParameterBundle((rate(value=Fraction(3_000_000_000,2)),))
        rebuilt_graph=Graph.from_dict(json.loads(json.dumps(graph.to_dict())))
        rebuilt_bundle=ParameterBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
        self.assertEqual(rebuilt_graph.tasks[0].demands[0].unit,'cpu_cycle')
        self.assertEqual(rebuilt_bundle.get('cpu').unit,'cpu_cycle/s')
        self.assertEqual(evaluate(graph,bundle),evaluate(rebuilt_graph,rebuilt_bundle))


if __name__=='__main__':unittest.main()
