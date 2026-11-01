from copy import deepcopy
from fractions import Fraction
import unittest

from profiling.event_stage_fit import (
    _cycles, _event, _pack, before_wall_inputs, derive_before_phase_anchor,
    fit_nonnegative_affine, reference_profile_summary,
)


CLOCK=dict(clock_id='windows_qpc_floor_ns',clock_origin='absolute_qpc_zero',frequency_hz=10_000_000)


def action(index,begin,duration,cycles=1000):
    return dict(action_index=index,begin_ns=begin+2,end_ns=begin+2+duration,
        thread_cycles=dict(enabled=True,unit='raw_QueryThreadCycleTime_cycles',
            before_query_begin_ns=begin,before_query_end_ns=begin+1,
            after_query_begin_ns=begin+duration+3,after_query_end_ns=begin+duration+4,
            before_cycles=10_000,after_cycles=10_000+cycles,delta_cycles=cycles,monotonic=True,
            cycles_converted_to_time=False,usable_as_heldout_resource_input=False))


def reference(target,pid,with_pause=False):
    iterations=[];cursor=100
    for index in range(3):
        duration=1000+(32_000_000 if with_pause and index==2 else 0)
        a=action(target,cursor,duration)
        iterations.append(dict(iteration=index,begin_ns=cursor,end_ns=cursor+duration+4,actions=[a]))
        cursor+=duration+10
    return dict(schema='physical-stage-native-1',process_id=pid,clock=deepcopy(CLOCK),mode='target',
        target_action_index=target,samples_requested=3,samples_completed=3,
        sampling_begin_ns=100,sampling_end_ns=cursor,iterations=iterations)


def frozen():
    return dict(baseline_cpu_reference_work_cycles=dict(numerator=1000,denominator=1),
        clock=deepcopy(CLOCK),reference_targets=dict(cpu=39,gpu=52),CPU_blackout_hypothesis=None,
        declared_protocol=dict(minimum_excess_ns=15_000_000))


class EventStageFitTests(unittest.TestCase):
    def test_exact_nonnegative_affine_identified(self):
        result=fit_nonnegative_affine([0,1,2],[2,5,8],regularization=Fraction(0))
        self.assertTrue(result['identified'])
        self.assertEqual(result['intercept'],2)
        self.assertEqual(result['slope'],3)

    def test_negative_slope_constraint_is_explicit(self):
        result=fit_nonnegative_affine([0,1,2],[8,5,2],regularization=Fraction(0))
        self.assertEqual(result['intercept'],5)
        self.assertEqual(result['slope'],0)

    def test_unvarying_feature_stays_unknown(self):
        result=fit_nonnegative_affine([0,0,0],[2,5,8],regularization=Fraction(1,1000))
        self.assertFalse(result['identified'])
        self.assertIsNone(result['slope'])

    def test_profile_reference_cycles_use_full_iteration_wall(self):
        cpu=reference(39,10)
        for invocation in cpu['iterations']:
            invocation['begin_ns']-=10
            invocation['end_ns']+=20
        result=reference_profile_summary(cpu)
        self.assertEqual(result['work_cycles'],1000)
        self.assertEqual(result['outer_query_wall_ns'],1004)
        self.assertEqual(result['full_iteration_wall_ns'],1034)
        self.assertEqual(result['iteration_outside_query_wall_statistics']['median'],30)
        self.assertEqual(result['empirical_cycle_rate_per_s'],Fraction(1000*1_000_000_000,1034))
        self.assertEqual(result['cycle_rate_denominator'],'full_reference_iteration_QPC_wall_ns')
        self.assertFalse(result['cycles_converted_with_assumed_GHz'])

    def test_profile_and_before_reference_rate_denominators_match(self):
        cpu=reference(39,10)
        for invocation in cpu['iterations']:
            invocation['begin_ns']-=10
            invocation['end_ns']+=20
        profile=reference_profile_summary(cpu)
        inputs=before_wall_inputs(frozen(),cpu,reference(52,11),cutoff_ns=1_000_000,
            target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])
        self.assertEqual(inputs['cpu_cycle_rate_per_s'],_pack(profile['empirical_cycle_rate_per_s']))

    def test_heldout_input_ignores_every_raw_cycle_field(self):
        cpu=reference(39,10);gpu=reference(52,11)
        arguments=dict(cutoff_ns=1_000_000,target_process_id=99,cpu_evidence=['cpu:raw'],gpu_evidence=['gpu:raw'])
        old=before_wall_inputs(frozen(),cpu,gpu,**arguments)
        for raw in (cpu,gpu):
            for invocation in raw['iterations']:
                for item in invocation['actions']:item['thread_cycles']={'deliberately':'invalid raw counters'}
        new=before_wall_inputs(frozen(),cpu,gpu,**arguments)
        self.assertEqual(old,new)
        self.assertFalse(new['measured_heldout_cycles_used'])

    def test_heldout_capacity_uses_iteration_wall_with_query_cost(self):
        result=before_wall_inputs(frozen(),reference(39,10),reference(52,11),cutoff_ns=1_000_000,
            target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])
        self.assertEqual(result['cpu_cycle_rate_per_s'],_pack(Fraction(1000*1_000_000_000,1004)))

    def test_reference_target_identity_checked(self):
        with self.assertRaisesRegex(ValueError,'target identity'):
            before_wall_inputs(frozen(),reference(52,10),reference(52,11),cutoff_ns=1_000_000,
                target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])

    def test_reference_clock_checked(self):
        cpu=reference(39,10);cpu['clock']['clock_origin']='renamed'
        with self.assertRaisesRegex(ValueError,'clock differs'):
            before_wall_inputs(frozen(),cpu,reference(52,11),cutoff_ns=1_000_000,
                target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])

    def test_target_cannot_be_own_reference(self):
        with self.assertRaisesRegex(ValueError,'source is target'):
            before_wall_inputs(frozen(),reference(39,99),reference(52,11),cutoff_ns=1_000_000,
                target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])

    def test_future_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'crosses cutoff'):
            before_wall_inputs(frozen(),reference(39,10),reference(52,11),cutoff_ns=500,
                target_process_id=99,cpu_evidence=['cpu'],gpu_evidence=['gpu'])

    def test_unknown_profile_cycle_counter_not_replaced_with_zero(self):
        a=action(39,100,1000);a['thread_cycles'].update(before_cycles=None,delta_cycles=None,monotonic=None)
        value,_=_cycles(a)
        self.assertIsNone(value)

    def test_bad_profile_delta_rejected(self):
        a=action(39,100,1000);a['thread_cycles']['delta_cycles']=9
        with self.assertRaisesRegex(ValueError,'delta differs'):_cycles(a)

    def test_periodic_file_poll_is_rejected(self):
        event=dict(schema='physical-event-parking-1',mode='parent_owned_Windows_events',go_event='go',stop_event='stop',
            post_READY_periodic_file_poll=True,post_sampling_periodic_wakeup=False)
        with self.assertRaisesRegex(ValueError,'parking contract'):
            _event(dict(event_parking=event),dict(ready=dict(event_parking=event),event_controller=dict(names=dict(go='go',stop='stop'))))

    def test_before_anchor_has_separate_pause_end_bounds(self):
        hypothesis=dict(period_candidate_ns=65_000_000,signature_sha256='signature',context=dict(
            clock_id=CLOCK['clock_id'],clock_origin=CLOCK['clock_origin'],resource_id='cpu:host-mask:4',competitor_context=dict(actors=['cpu'])))
        cpu=reference(39,10,with_pause=True)
        result=derive_before_phase_anchor(cpu,hypothesis,cutoff_ns=40_000_000,target_process_id=99,evidence=['before'],minimum_excess_ns=15_000_000)
        self.assertTrue(result['known'])
        end=result['pause_end_window_ns'];start=result['pause_start_window_ns']
        self.assertGreater(end['minimum'],start['maximum'])
        self.assertFalse(result['phase_anchor']['raw_cycle_counters_used'])
        self.assertLessEqual(end['maximum'],cpu['sampling_end_ns'])


if __name__=='__main__':unittest.main()
