"""Pure protocol checks; these tests never initialize a compute backend."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from profiling.dynamic_validation.external import boundaries,component,plans,state_at,verify


def fixture(instrumentation='bare'):
    actions=[dict(kind='BEGIN'),dict(kind='COMPUTE'),dict(kind='END')]
    program=dict(program_sha256='b'*64,actions=actions)
    request=dict(source_sha256='a'*64,instrumentation=instrumentation,
        cpu_actor=dict(device='cpu',component=component('cpu'),intervals_ms=[[-1,3]]),
        gpu_actor=dict(device='gpu',component=component('gpu'),intervals_ms=[]))
    epoch=100000;begin=epoch+5;blocks=[];previous=begin
    for block in range(128):
        measured=[];b0=previous+1;cursor=b0
        if instrumentation=='stamped':
            for replay in range(16):
                for index in range(3):
                    measured.append(dict(global_replay=block*16+replay,action_index=index,
                        begin_tick=cursor,end_tick=cursor+1,wall_ns=100))
                    cursor+=2
        end=b0+100
        blocks.append(dict(block=block,replays=16,begin_tick=b0,end_tick=end,wall_ns=10000,actions=measured));previous=end
    actors={}
    for device,mask in (('cpu',4),('gpu',65536)):
        rows=[dict(planned_begin_ms=-1,planned_end_ms=3,calls=10,actual_first_tick=epoch-9999,
            actual_last_completion_tick=epoch+30001,start_lateness_ns=100)] if device=='cpu' else []
        actors[device]=dict(device=device,cancelled=False,repeated_output_equal=True,
            thread_affinity=dict(requested_mask=mask,observed_mask=mask,final_mask=mask,restored=True),
            ready_tick=epoch-22000,stopped_tick=epoch+31000,component=request[device+'_actor']['component'],
            offered_work_is_available_service=False,intervals=rows,
            controller=dict(kind='owned-high-resolution-waitable-timer-v1'),idle_actor_busy_polling=False)
    native=dict(schema='dynamic-external-native-1',request=request,process_id=123,source_sha256='a'*64,
        program_sha256='b'*64,actions=actions,main_thread_affinity=dict(requested_mask=4,observed_mask=4,final_mask=4,restored=True),
        process_affinity=dict(requested_mask=4294967295,observed_mask=4294967295,final_mask=4294967295,applied_before_backend_initialization=True),
        clock=dict(frequency_hz=10000000),target_expected_origin_tick=epoch,target_begin_tick=begin,target_end_tick=previous+1,
        target_wall_ns=(previous+1-begin)*100,initial_start_lateness_ns=500,blocks=blocks,continuous_replays=2048,
        reset_prime_between_timed_blocks=False,instrumentation=instrumentation,actor_timeline_triggered_by_target_latency=False,
        future_target_state_used=False,actors=actors,native_repeat_consistency=dict(passed=True),
        controller=dict(kind='owned-high-resolution-waitable-timer-v1',busy_spin=False,global_timeBeginPeriod_called=False),
        verification_native=dict(copy_values_equal=True,ports=dict(port=[1.,-1.])))
    binding=dict(program=dict(path='program.json'),reference=dict(path='reference.json'))
    return native,request,binding,program


class DynamicExternalTests(TestCase):
    def verified(self,native,request,binding,program):
        with patch('profiling.dynamic_validation.external.read',side_effect=lambda p:program if p=='program.json' else dict(port=[1.,-1.])):
            return verify(native,request,binding,123)

    def test_complete_bare_clock_and_actor_receipts(self):
        args=fixture();result=self.verified(*args)
        self.assertTrue(result['passed']);self.assertEqual(result['continuous_replays'],2048)
        self.assertEqual(result['profile_action_intervals'],0)

    def test_stamped_profile_keeps_every_action(self):
        result=self.verified(*fixture('stamped'))
        self.assertEqual(result['profile_action_intervals'],2048*3)

    def test_actual_start_lateness_not_hidden_by_reanchoring(self):
        native,request,binding,program=fixture();native['initial_start_lateness_ns']=0
        with self.assertRaisesRegex(ValueError,'origin lateness'):self.verified(native,request,binding,program)

    def test_missing_actor_pulse_rejected(self):
        native,request,binding,program=fixture();native['actors']['cpu']['intervals'][0]['calls']=0
        with self.assertRaisesRegex(ValueError,'pulse was not executed'):self.verified(native,request,binding,program)

    def test_actor_natural_work_completion_after_stop_is_permitted(self):
        native,request,binding,program=fixture()
        self.assertGreater(native['actors']['cpu']['intervals'][0]['actual_last_completion_tick'],130000)
        self.assertTrue(self.verified(native,request,binding,program)['passed'])

    def test_nonfinite_numerical_truth_rejected(self):
        native,request,binding,program=fixture();native['verification_native']['ports']['port'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'port mismatch'):self.verified(native,request,binding,program)

    def test_planned_edges_are_half_open_and_independent_of_target(self):
        cpu,gpu=plans('heldout','cpu_gpu_staggered',0)
        request=dict(cpu_actor=dict(intervals_ms=cpu),gpu_actor=dict(intervals_ms=gpu))
        self.assertEqual(state_at(request,3000000000-1),(0,0))
        self.assertEqual(state_at(request,3000000000),(1,0))
        self.assertEqual(state_at(request,9000000000),(0,1))
        self.assertEqual(state_at(request,15000000000),(0,0))
        self.assertEqual(boundaries(request),[3000000000,6000000000,9000000000,15000000000])

    def test_profile_and_heldout_have_distinct_predeclared_waveforms(self):
        for scenario in ('cpu_on_off','gpu_on_off','cpu_gpu_staggered'):
            self.assertNotEqual(plans('profile',scenario,0),plans('heldout',scenario,0))
            self.assertEqual(plans('heldout',scenario,0),plans('heldout',scenario,2))
        self.assertEqual(plans('heldout','baseline',0),([],[]))

