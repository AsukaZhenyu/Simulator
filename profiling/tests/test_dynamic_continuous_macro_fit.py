"""Continuous calibration source/weighting tests; no hardware observations."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from profiling.dynamic_validation import continuous_macro_fit as fit


def native_fixture(count=4,response_ns=250000000,gap_ns=0,state=(0,0)):
    start=1000000;cursor=start;actions=[];gaps=[]
    for ordinal in range(count):
        gaps.append(dict(begin_tick=cursor,end_tick=cursor+gap_ns,wall_ns=gap_ns));cursor+=gap_ns
        actions.append(dict(ordinal=ordinal,action_index=22,begin_tick=cursor,
                            end_tick=cursor+response_ns,wall_ns=response_ns));cursor+=response_ns
    gaps.append(dict(begin_tick=cursor,end_tick=cursor+gap_ns,wall_ns=gap_ns));cursor+=gap_ns
    return dict(clock=dict(frequency_hz=10**9),instrumentation='stamped',callback_stage_prices_used=False,
        foreground_lifecycle='continuous_invocation',reset_prime_between_timed_blocks=False,
        request=dict(offered_state=list(state)),under_state_warmup=dict(action_index=22,begin_tick=0,end_tick=start,wall_ns=start),
        continuous_profile=dict(begin_tick=start,end_tick=cursor,actions=actions,gaps=gaps,
            min_duration_ns=1000000000,max_calls=512,minimum_duration_reached=True,stop_reason='minimum_duration',accuracy_dependent_stop=False))


class ContinuousContextTests(unittest.TestCase):
    def test_component_and_all_gaps_close_exact_clock_without_price_compensation(self):
        native=native_fixture(gap_ns=1000);result=fit.continuous_context(native)
        self.assertEqual(result['invocation_response_mean_ns'],250000000)
        self.assertEqual(result['outside_action_gap_ns'],5000)
        self.assertEqual(result['timed_window_ns'],1000005000)
        self.assertEqual(sum(result['invocation_responses_ns'])+sum(result['gap_responses_ns']),result['timed_window_ns'])
        self.assertFalse(result['outside_action_gap_added_to_component_price'])

    def test_separate_under_state_warm_response_is_not_included_in_steady_mean(self):
        native=native_fixture();result=fit.continuous_context(native)
        self.assertEqual(result['under_state_warmup_response_ns'],1000000)
        self.assertFalse(result['warm_response_used_as_steady_price'])
        self.assertEqual(result['invocation_response_mean_ns'],250000000)

    def test_safety_cap_before_minimum_duration_remains_unsupported(self):
        native=native_fixture(count=512,response_ns=1)
        native['continuous_profile'].update(minimum_duration_reached=False,stop_reason='max_calls')
        with self.assertRaises(ValueError):fit.continuous_context(native)
        native['continuous_profile']['minimum_duration_reached']=True
        with self.assertRaises(ValueError):fit.continuous_context(native)

    def test_one_postwarm_invocation_reaching_one_second_is_valid_without_variance_claim(self):
        result=fit.continuous_context(native_fixture(count=1,response_ns=1000000000))
        self.assertEqual(result['invocations'],1)
        self.assertTrue(result['single_long_postwarm_invocation'])
        self.assertIsNone(result['within_context_invocation_CV'])
        self.assertFalse(result['minimum_duration_or_sample_count_claims_low_variance'])

    def test_missing_gap_or_fabricated_QPC_partition_is_rejected(self):
        native=native_fixture();native['continuous_profile']['gaps'].pop()
        with self.assertRaises(ValueError):fit.continuous_context(native)
        native=native_fixture();native['continuous_profile']['gaps'][1]['end_tick']+=1
        with self.assertRaises(ValueError):fit.continuous_context(native)


class ContinuousCellTests(unittest.TestCase):
    def contexts(self,root):
        jobs=[];records={}
        for state_index,state in enumerate(fit.STATES):
            for index,count in enumerate((2,4,8)):
                number=state_index*3+index;name=f'state{state_index}-r{index}'
                job=dict(id=name,split='profile',request=dict(offered_state=list(state)),offered_state=list(state));jobs.append(job)
                path=root/(name+'.json');path.write_text(name,encoding='utf-8',newline='\n')
                execution=dict(status='completed',process_id=100+number,process_creation_time_100ns=1000+number,
                    native=dict(path=str(root/(name+'-native.json')),sha256=f'{number:064x}'))
                records[name]=(native_fixture(count=count,response_ns=1000000000//count,state=state),execution,path)
        return dict(jobs=jobs,scope=dict(context=dict(foreground_lifecycle='continuous_invocation'))),records

    def test_equal_context_means_do_not_overweight_faster_processes_with_more_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,records=self.contexts(root)
            cells,evidence=fit._calibrated_cells(root,manifest,lambda out,j,m:records[j['id']])
            expected=(Fraction(500000000)+250000000+125000000)/3
            self.assertTrue(all(c['mean_ns']==expected for c in cells))
            self.assertNotEqual(expected,Fraction(3000000000,14))
            self.assertEqual(len(evidence),12)
            self.assertTrue(all(c['fresh_contexts']==3 for c in cells))

    def test_old_single_call_or_heldout_context_is_not_importable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,records=self.contexts(root)
            manifest['jobs'][0]['imported']=True
            with self.assertRaises(ValueError):fit._calibrated_cells(root,manifest,lambda out,j,m:records[j['id']])

    def test_lifecycle_scope_and_distinct_processes_are_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,records=self.contexts(root)
            manifest['scope']['context']['foreground_lifecycle']='single_invocation'
            with self.assertRaises(ValueError):fit._calibrated_cells(root,manifest,lambda out,j,m:records[j['id']])
            manifest['scope']['context']['foreground_lifecycle']='continuous_invocation'
            records[manifest['jobs'][1]['id']][1].update(process_id=100,process_creation_time_100ns=1000)
            with self.assertRaises(ValueError):fit._calibrated_cells(root,manifest,lambda out,j,m:records[j['id']])


if __name__=='__main__':unittest.main()
