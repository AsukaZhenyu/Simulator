from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import bare_macro_v2_campaign as campaign


class ConstantStateValidationTests(unittest.TestCase):
    def fixture(self,folder):
        oracle=folder/'oracle.json';oracle.write_text('{"o":[1.0,2.0]}\n',encoding='utf-8',newline='\n')
        request=dict(split='profile',instrumentation='stamped',offered_state=[1,0],replays=1,block_replays=1,
            source_sha256='a'*64,callback_source_sha256='b'*64,actor_source_sha256='c'*64,
            cpu_actor=dict(intervals_ms=[[0,30000]]),gpu_actor=dict(intervals_ms=[]))
        epoch=1000000000;begin=2025000000;end=begin+6000000
        actor=lambda mask,intervals:dict(thread_affinity=dict(observed_mask=mask,final_mask=mask),ready_tick=epoch-1,
            repeated_output_equal=True,cancelled=False,intervals=intervals)
        native=dict(schema='dynamic-external-stage-native-8',request=request,process_id=123,
            source_sha256='a'*64,callback_source_sha256='b'*64,actor_source_sha256='c'*64,program_sha256='d'*64,
            attention_action_index=22,instrumentation='stamped',complete_attention_graph_only=True,
            complete_LLM_strategy_replayed=False,timed_result_checked_before_reset=True,
            internal_GPU_work_added=False,API_response_is_CPU_active=False,activity_capture_instantiated=False,
            future_target_state_used=False,actor_timeline_triggered_by_target_latency=False,performance_parameters_fitted=False,
            process_affinity=dict(observed_mask=4294967295,final_mask=4294967295),
            target_thread_affinity=dict(observed_mask=4,final_mask=4),initialization_affinity=dict(observed_mask=4,final_mask=4),
            native_repeat_consistency=dict(passed=True),verification_native=dict(copy_values_equal=True,ports={'o':[1.,2.]}),
            clock=dict(clock_id='windows_qpc',frequency_hz=1000000000),target_expected_origin_tick=epoch,
            target_begin_tick=begin,target_end_tick=end,steady_window_begin_tick=epoch,steady_window_end_tick=epoch+30000000000,
            profile_window_contained=True,planned_target_entry_tick=epoch+1000000000,initial_start_lateness_ns=25000000,
            target_entry_controller_passed=False,target_wall_ns=end-begin,continuous_replays=1,
            blocks=[dict(block=0,replays=1,begin_tick=begin,end_tick=end,wall_ns=end-begin,
                actions=[dict(action_index=22,global_replay=0,callback_phase_id=1,begin_tick=begin,end_tick=end,wall_ns=end-begin)])],
            actors=dict(cpu=actor(4,[dict(planned_begin_ms=0,planned_end_ms=30000,calls=1,actual_first_tick=epoch,actual_last_completion_tick=epoch+30000000000)]),gpu=actor(65536,[])),
            runtime_callbacks=dict(enabled=False,record_count=0,callback_records=[],metadata=[],all_selected_calls_retained=True,activity_driver_kernel_collection_enabled=False))
        return native,request,dict(program_sha256='d'*64,reference=dict(path=str(oracle)))

    def test_late_constant_component_is_valid_without_rewriting_original_flag(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));before=copy.deepcopy(n)
            proof=campaign.verify_constant_profile(n,r,b,123)
            self.assertTrue(proof['constant_actual_state_brackets_entire_component'])
            self.assertFalse(proof['original_native_entry_deadline_passed'])
            self.assertEqual(proof['initial_start_lateness_ns'],25000000)
            self.assertEqual(n,before)

    def test_planned_window_alone_does_not_replace_actual_actor_bracketing(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['actors']['cpu']['intervals'][0]['actual_first_tick']=n['target_begin_tick']+1
            with self.assertRaisesRegex(ValueError,'did not bracket'):campaign.verify_constant_profile(n,r,b,123)

    def test_complete_numerical_oracle_is_still_required(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['verification_native']['ports']['o'][1]=2.2
            with self.assertRaisesRegex(ValueError,'numerical mismatch'):campaign.verify_constant_profile(n,r,b,123)

    def test_rewriting_deadline_flag_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['target_entry_controller_passed']=True
            with self.assertRaisesRegex(ValueError,'deadline flag'):campaign.verify_constant_profile(n,r,b,123)

    def test_source_identity_and_core_affinity_are_not_relaxed(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));n['source_sha256']='e'*64
            with self.assertRaisesRegex(ValueError,'binary/source'):campaign.verify_constant_profile(n,r,b,123)
            n,r,b=self.fixture(Path(d));n['target_thread_affinity']['final_mask']=16
            with self.assertRaisesRegex(ValueError,'target/init affinity'):campaign.verify_constant_profile(n,r,b,123)

    def test_dynamic_heldout_keeps_original_ten_ms_deadline(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            n,r,b=self.fixture(Path(d));r['split']='heldout'
            with self.assertRaisesRegex(ValueError,'entry late'):campaign.native_campaign.verify(n,r,b,123)

    def test_original_failed_execution_and_native_flag_remain_immutable(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            folder=Path(d);n,r,b=self.fixture(folder);result=folder/'result.json';result.write_text(json.dumps(n)+'\n',encoding='utf-8',newline='\n')
            execution=folder/'execution.json';original=dict(status='failed',error=campaign.OLD_ERROR,returncode=0,process_exited_verified=True,
                command=['unchanged.exe','unchanged-program','unchanged-request',str(result)],process_id=123,process_creation_time_100ns=456)
            execution.write_text(json.dumps(original)+'\n',encoding='utf-8',newline='\n');before=execution.read_bytes()
            job=dict(id=campaign.LATE_JOB,split='profile',imported=True,request=r,imported_execution=campaign.receipt(execution),imported_native=campaign.receipt(result))
            native,view,_=campaign.verified_job(folder,job,dict(programs={'profile':b}))
            self.assertEqual(view['status'],'failed');self.assertEqual(view['native'],campaign.receipt(result));self.assertFalse(native['target_entry_controller_passed'])
            self.assertEqual(execution.read_bytes(),before);self.assertNotIn('native',json.loads(before))
            other=folder/'other.json';other.write_bytes(result.read_bytes());job['imported_native']=campaign.receipt(other)
            with self.assertRaisesRegex(ValueError,'exact unmodified original command output'):campaign.verified_job(folder,job,dict(programs={'profile':b}))

    def test_protocol_keeps_dynamic_and_accuracy_thresholds(self):
        p=campaign.protocol();self.assertEqual(p['dynamic_heldout_start_deadline_ns'],10000000)
        self.assertEqual(p['acceptance']['process_mean_APE_max'],.10);self.assertEqual(p['acceptance']['within_process_block_APE_P90_max'],.20)
        self.assertFalse(p['native_records_or_flags_rewritten'])


if __name__=='__main__':unittest.main()
