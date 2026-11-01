"""Hand-derived external periodic-service integration, without OS inference."""
from dataclasses import replace
from fractions import Fraction
import unittest

from llm_infer_model.performance.evaluator import BlockedExecution
from llm_infer_model.performance.graph import Graph, ResourcePoint, ResourceTimeline, Task, WorkDemand
from llm_infer_model.performance.parameters import InvalidModel, Parameter, ParameterBundle, UnknownParameter
from llm_infer_model.performance.service_calendar import PeriodicServiceCalendar, compile_service_calendars


EVIDENCE=('declared external finite-service test',)


def bundle(unknown=False):
    return ParameterBundle((Parameter('cpu',None if unknown else 10**9,'thread_cpu_ns/s','declared',EVIDENCE,'one thread'),
                            Parameter('gpu',10**9,'flop/s','declared',EVIDENCE,'device'),
                            Parameter('other',10**9,'byte/s','declared',EVIDENCE,'link')))


def calendar(**kwargs):
    fields=dict(parameter='cpu',period_ns=10,blackout_ns=3,phase_ns=2,valid_until_ns=50,
                source_kind='declared_scenario',evidence=EVIDENCE,scope='explicit availability; no OS scheduler',
                clock_id='scenario_ns',clock_origin='explicit_test_zero')
    fields.update(kwargs)
    return PeriodicServiceCalendar(**fields)


def graph(n=12,name='cpu',unit='thread_cpu_ns'):
    return Graph((Task('work',name,(WorkDemand(name,n,unit),)),))


class ServiceCalendarTests(unittest.TestCase):
    def test_integrates_repeated_blackouts(self):
        result=compile_service_calendars(bundle(),(calendar(),))
        self.assertEqual(result.evaluate(graph(),bundle()).duration_ns,18)
        self.assertFalse(result.hardware_accuracy_qualified)

    def test_previous_period_can_cover_time_zero(self):
        c=calendar(phase_ns=8,blackout_ns=4)
        result=compile_service_calendars(bundle(),(c,))
        self.assertEqual(result.timeline.points[0].rates['cpu'],0)
        self.assertEqual(result.evaluate(graph(8),bundle()).duration_ns,14)

    def test_gpu_and_cpu_use_same_calendar_rule(self):
        cpu=compile_service_calendars(bundle(),(calendar(),)).evaluate(graph(),bundle())
        gpu=compile_service_calendars(bundle(),(calendar(parameter='gpu'),)).evaluate(graph(name='gpu',unit='flop'),bundle())
        self.assertEqual(cpu.duration_ns,gpu.duration_ns)

    def test_baseline_rate_change_inside_blackout_restores_latest_rate(self):
        baseline=ResourceTimeline((ResourcePoint(4,{'cpu':500_000_000,'other':10}),))
        result=compile_service_calendars(bundle(),(calendar(period_ns=20,blackout_ns=4),),baseline_timeline=baseline)
        bytime={p.at_ns:p for p in result.timeline.points}
        self.assertEqual(bytime[4].rates['cpu'],0)
        self.assertEqual(bytime[6].rates['cpu'],Fraction(500_000_000))
        self.assertEqual(bytime[4].rates['other'],10)
        self.assertEqual(result.evaluate(graph(6),bundle()).duration_ns,14)

    def test_unknown_phase_period_or_width_never_guessed(self):
        for field in ('phase_ns','period_ns','blackout_ns'):
            with self.subTest(field=field),self.assertRaises(UnknownParameter):
                compile_service_calendars(bundle(),(calendar(**{field:None}),))

    def test_unknown_base_service_never_zero_filled(self):
        with self.assertRaises(UnknownParameter):compile_service_calendars(bundle(True),(calendar(),))

    def test_explicit_zero_width_and_full_blackout_are_distinct(self):
        no_pause=compile_service_calendars(bundle(),(calendar(blackout_ns=0),))
        self.assertEqual(no_pause.evaluate(graph(),bundle()).duration_ns,12)
        full_pause=compile_service_calendars(bundle(),(calendar(blackout_ns=10),))
        with self.assertRaises(BlockedExecution):full_pause.evaluate(graph(),bundle())

    def test_horizon_rejects_extrapolated_prediction(self):
        c=calendar(valid_until_ns=10,blackout_ns=0)
        result=compile_service_calendars(bundle(),(c,))
        with self.assertRaises(UnknownParameter):result.evaluate(graph(12),bundle())
        with self.assertRaises(UnknownParameter):c.blackout_at(11)

    def test_observed_target_replay_explicit_and_unqualified(self):
        c=calendar(source_kind='observed_diagnostic')
        with self.assertRaises(InvalidModel):compile_service_calendars(bundle(),(c,))
        result=compile_service_calendars(bundle(),(c,),allow_observed_diagnostic=True)
        self.assertTrue(result.diagnostic_replay)
        self.assertFalse(result.hardware_accuracy_qualified)

    def test_duplicate_calendar_rejected(self):
        with self.assertRaises(InvalidModel):compile_service_calendars(bundle(),(calendar(),calendar()))

    def test_invalid_period_width_phase_or_source(self):
        for fields in ({'period_ns':0},{'blackout_ns':11},{'phase_ns':10},{'phase_ns':-1},{'source_kind':'assumed'}):
            with self.subTest(fields=fields),self.assertRaises(InvalidModel):calendar(**fields)

    def test_baseline_outside_horizon_rejected(self):
        with self.assertRaises(InvalidModel):
            compile_service_calendars(bundle(),(calendar(),),baseline_timeline=ResourceTimeline((ResourcePoint(51,{'cpu':1}),)))

    def test_event_limit_is_bounded(self):
        with self.assertRaises(InvalidModel):compile_service_calendars(bundle(),(calendar(period_ns=2,phase_ns=0,blackout_ns=1,valid_until_ns=1_000_000),))

    def test_mixed_clock_or_origin_is_rejected(self):
        for fields in ({'clock_id':'foreign_clock'},{'clock_origin':'foreign_zero'}):
            with self.subTest(fields=fields),self.assertRaises(InvalidModel):
                compile_service_calendars(bundle(),(calendar(),calendar(parameter='gpu',**fields)))

    def test_invalid_cpu_rate_update_is_rejected_before_evaluation(self):
        with self.assertRaises(InvalidModel):
            compile_service_calendars(bundle(),(calendar(),),baseline_timeline=ResourceTimeline((ResourcePoint(4,{'cpu':2_000_000_000}),)))


if __name__=='__main__':unittest.main()
