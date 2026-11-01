"""Hand-calculated host-cycle and completion-response hypotheses."""
from dataclasses import replace
from fractions import Fraction
import unittest

from llm_infer_model.performance.cycle_response import (
    CycleResponseStage, CycleResponseProgram, compile_cycle_response, compile_cycle_calendar,
)
from llm_infer_model.performance.graph import ResourcePoint,ResourceTimeline
from llm_infer_model.performance.parameters import InvalidModel,Parameter,ParameterBundle,UnknownParameter
from llm_infer_model.performance.service_calendar import PeriodicServiceCalendar


EVIDENCE=('hand-derived coarse critical-response scenario',)


def parameters(value=10**9,unit='cpu_cycle/s'):
    return ParameterBundle((Parameter('host_cycle_service',value,unit,'declared',EVIDENCE,'scoped raw counter service'),))


def program(stages=None,head=0,tail=0):
    return CycleResponseProgram((CycleResponseStage(0,'COMPUTE',4,10,2),) if stages is None else stages,
                                head,tail,EVIDENCE,'coarse response hypothesis, not physical CUDA lowering')


def calendar(**kwargs):
    fields=dict(parameter='host_cycle_service',period_ns=100,blackout_ns=15,phase_ns=5,valid_until_ns=100,
                source_kind='declared_scenario',evidence=EVIDENCE,scope='explicit cycle service scenario',
                clock_id='scenario_ns',clock_origin='scenario_zero')
    fields.update(kwargs)
    return PeriodicServiceCalendar(**fields)


class CycleResponseTests(unittest.TestCase):
    def test_after_host_and_overlap_are_distinct_declared_hypotheses(self):
        after=compile_cycle_response(program(),completion_phase='after_host').evaluate(parameters())
        overlap=compile_cycle_response(program(),completion_phase='overlap_host').evaluate(parameters())
        self.assertEqual((after.duration_ns,overlap.duration_ns),(16,12))

    def test_cpu_blackout_does_not_pause_released_completion(self):
        compiled=compile_cycle_response(program(),completion_phase='overlap_host')
        trace=ResourceTimeline((ResourcePoint(5,{'host_cycle_service':0}),ResourcePoint(20,{'host_cycle_service':10**9})))
        result=compiled.evaluate(parameters(),trace)
        rows={r.task_id:r for r in result.tasks}
        self.assertEqual(rows['action:0:completion_response'].end_ns,12)
        self.assertEqual(rows['action:0:host_cycles'].end_ns,21)
        self.assertEqual(result.duration_ns,21)

    def test_after_host_response_starts_after_cpu_recovery(self):
        compiled=compile_cycle_response(program())
        result=compile_cycle_calendar(parameters(),calendar()).evaluate(compiled,parameters())
        self.assertEqual(result.duration_ns,31)

    def test_subsequent_action_waits_for_both_components(self):
        stages=(CycleResponseStage(0,'COMPUTE',4,10,2),CycleResponseStage(1,'COPY',3,1,1))
        compiled=compile_cycle_response(program(stages,head=2,tail=3),completion_phase='overlap_host')
        result=compiled.evaluate(parameters())
        self.assertEqual(result.duration_ns,21)
        rows={r.task_id:r for r in result.tasks}
        self.assertEqual(rows['action:1:gap_cycles'].start_ns,14)

    def test_absolute_start_alignment_selects_existing_external_state(self):
        compiled=compile_cycle_response(program((CycleResponseStage(0,'COMPUTE',2,0,0),)))
        trace=ResourceTimeline((ResourcePoint(5,{'host_cycle_service':0}),ResourcePoint(20,{'host_cycle_service':10**9})))
        self.assertEqual(compiled.evaluate(parameters(),trace,start_ns=10).duration_ns,12)

    def test_raw_cycles_never_interpreted_as_active_nanoseconds(self):
        with self.assertRaises(InvalidModel):compile_cycle_response(program()).evaluate(parameters(unit='thread_cpu_ns/s'))
        self.assertEqual(compile_cycle_response(program()).evaluate(parameters(2*10**9)).duration_ns,13)

    def test_all_unknown_fields_fail_loudly(self):
        for field in ('cpu_cycles','completion_latency_ns','gap_cycles'):
            stage=replace(program().stages[0],**{field:None})
            with self.subTest(field=field),self.assertRaises(UnknownParameter):compile_cycle_response(program((stage,)))
        with self.assertRaises(UnknownParameter):compile_cycle_response(program(head=None))
        with self.assertRaises(UnknownParameter):compile_cycle_response(program(tail=None))
        with self.assertRaises(UnknownParameter):compile_cycle_response(program()).evaluate(parameters(None))

    def test_explicit_zero_and_fractional_completion_are_preserved(self):
        compiled=compile_cycle_response(program((CycleResponseStage(0,'ADVANCE',0,Fraction(3,2),0),)))
        self.assertEqual(compiled.evaluate(parameters()).duration_ns,2)
        self.assertFalse(compiled.hardware_accuracy_qualified)
        self.assertFalse(compiled.to_dict()['completion_is_GPU_active_service'])

    def test_calendar_unknown_phase_and_diagnostic_replay_not_guessed(self):
        with self.assertRaises(UnknownParameter):compile_cycle_calendar(parameters(),calendar(phase_ns=None))
        c=calendar(source_kind='observed_diagnostic')
        with self.assertRaises(InvalidModel):compile_cycle_calendar(parameters(),c)
        self.assertTrue(compile_cycle_calendar(parameters(),c,allow_observed_diagnostic=True).diagnostic_replay)

    def test_calendar_finite_horizon_rejects_extrapolation(self):
        compiled=compile_cycle_response(program())
        c=compile_cycle_calendar(parameters(),calendar(valid_until_ns=10,blackout_ns=0))
        with self.assertRaises(UnknownParameter):c.evaluate(compiled,parameters())

    def test_bad_action_order_assumption_or_negative_latency_rejected(self):
        with self.assertRaises(InvalidModel):program((CycleResponseStage(1,'COMPUTE',1,1,1),))
        with self.assertRaises(InvalidModel):CycleResponseStage(0,'COMPUTE',1,-1,1)
        with self.assertRaises(InvalidModel):compile_cycle_response(program(),completion_phase='guess')


if __name__=='__main__':unittest.main()
