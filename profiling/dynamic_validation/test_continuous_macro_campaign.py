import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import continuous_macro_campaign as campaign


class ContinuousCampaignTests(unittest.TestCase):
    def fixture(self,folder):
        oracle=folder/'oracle.json';oracle.write_text('{"o":[1.0,2.0]}\n',encoding='utf-8',newline='\n')
        actions=[dict(kind='control',i=i) for i in range(22)]+[dict(kind='COMPUTE',device='gpu',target_id='layer:attn')]
        program=folder/'program.json';program.write_text(json.dumps(dict(actions=actions))+'\n',encoding='utf-8',newline='\n')
        request=dict(split='profile',instrumentation='stamped',offered_state=[1,0],source_sha256='a'*64,actor_source_sha256='b'*64,
            cpu_actor=dict(intervals_ms=[[0,30000]]),gpu_actor=dict(intervals_ms=[]))
        epoch=1000000000;entry=2025000000;begin=entry+6000000;end=begin+1200000000
        rows=[dict(ordinal=i,global_replay=i,action_index=22,begin_tick=begin+i*600000000,end_tick=begin+(i+1)*600000000,wall_ns=600000000) for i in range(2)]
        gaps=[dict(begin_tick=begin,end_tick=begin,wall_ns=0),dict(begin_tick=begin+600000000,end_tick=begin+600000000,wall_ns=0),dict(begin_tick=end,end_tick=end,wall_ns=0)]
        actor=lambda mask,intervals:dict(thread_affinity=dict(observed_mask=mask,final_mask=mask),ready_tick=epoch-1,repeated_output_equal=True,cancelled=False,intervals=intervals)
        n=dict(schema='dynamic-continuous-native-10',request=request,process_id=123,source_sha256='a'*64,actor_source_sha256='b'*64,
            program_sha256='d'*64,attention_action_index=22,actions=actions,target_action=actions[22],instrumentation='stamped',foreground_lifecycle='continuous_invocation',
            callback_stage_prices_used=False,profiling_CUPTI_initialized=False,profiling_NVML_initialized=False,profiling_extra_CUDA_events_created=False,
            internal_GPU_work_added=False,API_response_is_CPU_active=False,future_target_state_used=False,actor_timeline_triggered_by_target_latency=False,performance_parameters_fitted=False,
            complete_LLM_strategy_replayed=False,reset_prime_between_timed_blocks=False,complete_attention_graph_only=True,timed_result_checked_before_reset=True,
            process_affinity=dict(observed_mask=4294967295,final_mask=4294967295),initialization_affinity=dict(observed_mask=4,final_mask=4),target_thread_affinity=dict(observed_mask=4,final_mask=4),
            native_repeat_consistency=dict(passed=True),verification_native=dict(copy_values_equal=True,ports={'o':[1.,2.]}),clock=dict(clock_id='windows_qpc',frequency_hz=1000000000),
            target_begin_tick=begin,target_end_tick=end,target_expected_origin_tick=epoch,calibration_begin_tick=entry,entry_observed_tick=entry,planned_target_entry_tick=2000000000,
            initial_start_lateness_ns=25000000,target_entry_controller_passed=False,target_wall_ns=end-begin,
            blocks=[dict(block=0,begin_tick=begin,end_tick=end,replays=2,wall_ns=end-begin,actions=rows)],continuous_replays=2,
            under_state_warmup=dict(action_index=22,included_in_price=False,begin_tick=entry,end_tick=begin,wall_ns=begin-entry),
            continuous_profile=dict(actions=rows,begin_tick=begin,end_tick=end,min_duration_ns=1000000000,max_calls=512,accuracy_dependent_stop=False,
                minimum_duration_reached=True,stop_reason='minimum_duration',gaps=gaps),profile_window_contained=True,steady_window_begin_tick=epoch,steady_window_end_tick=epoch+30000000000,
            actors=dict(cpu=actor(4,[dict(planned_begin_ms=0,planned_end_ms=30000,calls=100,actual_first_tick=epoch,actual_last_completion_tick=epoch+30000000000)]),gpu=actor(65536,[])))
        return n,request,dict(program_sha256='d'*64,program=dict(path=str(program)),reference=dict(path=str(oracle)))

    def test_profile_records_continuous_window_without_pretending_late_arrival_passed(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));before=copy.deepcopy(n);v=campaign.verify(n,r,b,123)
            self.assertEqual(v['complete_Attention_replays'],2);self.assertFalse(v['original_entry_deadline_passed']);self.assertEqual(n,before)

    def test_actual_actor_must_bracket_warmup_as_well_as_timed_calls(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['actors']['cpu']['intervals'][0]['actual_first_tick']=n['target_begin_tick']-1
            with self.assertRaisesRegex(ValueError,r'warm\+timed'):campaign.verify(n,r,b,123)

    def test_gap_partition_cannot_drop_a_host_interval(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['continuous_profile']['gaps'][1]['begin_tick']-=1
            with self.assertRaisesRegex(ValueError,'gap partition'):campaign.verify(n,r,b,123)

    def test_maximum_call_safety_stop_cannot_masquerade_as_one_second_support(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['continuous_profile']['minimum_duration_reached']=False;n['continuous_profile']['stop_reason']='max_calls'
            with self.assertRaisesRegex(ValueError,'minimum duration'):campaign.verify(n,r,b,123)

    def test_new_plans_use_all_four_fresh_states_and_different_held_schedules(self):
        template=dict(cpu_actor=dict(main_thread_mask=4,init_policy='pinned_init'),gpu_actor=dict(main_thread_mask=65536,init_policy='pinned_init'))
        jobs=campaign.plans(template,'a'*64,'b'*64);self.assertEqual(len(jobs),18)
        for state in campaign.STATES:self.assertEqual(sum(j['split']=='profile' and tuple(j['offered_state'])==state for j in jobs),3)
        old=campaign.previous.previous.native_campaign.PULSES
        self.assertFalse(any(plan==previous for plan in campaign.PULSES.values() for previous in old.values()))
        self.assertTrue(all(not j['imported'] for j in jobs))
        self.assertFalse(campaign.protocol()['newly_unseen_mathematical_seed_claimed'])
        self.assertFalse(campaign.protocol()['previous_heldout_times_used_for_prices'])

    def test_heldout_deadline_remains_ten_ms(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));r['split']='heldout';n['entry_observed_tick']=n['target_begin_tick'];n['initial_start_lateness_ns']=n['target_begin_tick']-n['planned_target_entry_tick']
            with self.assertRaisesRegex(ValueError,'heldout entry late'):campaign.verify(n,r,b,123)


if __name__=='__main__':unittest.main()
