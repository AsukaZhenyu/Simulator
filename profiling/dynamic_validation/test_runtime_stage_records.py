"""Stage semantics only; no real callbacks or scheduler collection."""
import copy
import unittest

from .runtime_stage_records import coordinate, coordinate_sha256, ordered_stages, snapshot_evidence


class RuntimeStageTests(unittest.TestCase):
    def fixture(self):
        metadata=[dict(name='cudaMemcpyAsync_v3020',parameters_available=True,bytes=1024,copy_kind_cudaMemcpyKind=3,stream_handle_opaque_process_local=123),
                  dict(name='cudaStreamSynchronize_v3020',parameters_available=True,stream_handle_opaque_process_local=123)]
        rows=[dict(ordinal=i,phase_id=1,enter_tick=a,exit_tick=b,exit_received=True,return_status=0,metadata_index=i,correlation_id=100+i)
              for i,(a,b) in enumerate(((20,40),(70,90)))]
        return dict(all_selected_calls_retained=True,activity_driver_kernel_collection_enabled=False,metadata=metadata,callback_records=rows)

    def test_full_API_and_gap_serial_partition(self):
        r=ordered_stages(self.fixture(),dict(phase_id=1,begin_tick=10,end_tick=100),10**9,stream_roles={123:'target_compute'})
        self.assertEqual([x['response_ns'] for x in r['stages']],[10,20,30,20,10])
        self.assertEqual([x['role'] for x in r['stages']],['gap','api','gap','api','gap'])
        self.assertTrue(r['host_QPC_partition_closes'])
        self.assertFalse(r['API_response_is_active_CPU'])
        self.assertFalse(r['internal_GPU_work_added'])

    def test_opaque_handle_does_not_define_cross_process_coordinate(self):
        a=self.fixture()['metadata'][0];b=copy.deepcopy(a);b['stream_handle_opaque_process_local']=456
        self.assertEqual(coordinate_sha256(coordinate(a,{123:'target_compute'})),coordinate_sha256(coordinate(b,{456:'target_compute'})))
        with self.assertRaises(ValueError):coordinate(a,{})

    def test_overlap_and_incomplete_returns_reject_serialization(self):
        a=self.fixture();a['callback_records'][1]['enter_tick']=30
        with self.assertRaises(ValueError):ordered_stages(a,dict(phase_id=1,begin_tick=10,end_tick=100),10**9,stream_roles={123:'target_compute'})
        a=self.fixture();a['callback_records'][1]['return_status']=None
        with self.assertRaises(ValueError):ordered_stages(a,dict(phase_id=1,begin_tick=10,end_tick=100),10**9,stream_roles={123:'target_compute'})

    def test_gap_coordinate_contains_prev_and_next_API_semantics(self):
        r=ordered_stages(self.fixture(),dict(phase_id=1,begin_tick=10,end_tick=100),10**9,stream_roles={123:'target_compute'})
        gap=r['stages'][2]['coordinate']
        self.assertEqual(gap['previous']['name'],'cudaMemcpyAsync_v3020')
        self.assertEqual(gap['next']['name'],'cudaStreamSynchronize_v3020')
        self.assertNotIn('strategy_id',str(gap))

    def test_snapshot_counts_never_become_time_fractions(self):
        class Index:
            def contained_states(self,a,b):
                return [dict(actor=dict(actor_id='target'),thread=dict(thread_id=8,thread_state_provider='native',state_code=5,state_name='Wait',native_wait_reason_code=13,native_wait_reason_name='UserRequest'))]
        r=snapshot_evidence(Index(),10,100,actor_id='target',target_thread_id=8)
        self.assertEqual(r['snapshot_count'],1)
        self.assertEqual(r['raw_wait_reason_counts'][0]['native_code'],13)
        self.assertFalse(r['snapshot_counts_are_time_fractions'])
        self.assertFalse(r['driver_blocking_cause_identified'])


if __name__=='__main__':unittest.main()
