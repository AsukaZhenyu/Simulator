"""Software-only actual-input, full-vector and authorization gate checks."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from . import numerical_collect as c
from .numerical_v6 import ROOT
from .dataset import write

def temporary():
    root=(ROOT/'build/real-moe-validation-20261027/numerical-collector-test-fixtures').resolve()
    if ROOT.resolve() not in root.parents:raise ValueError('fixture outside workspace')
    root.mkdir(parents=True,exist_ok=True);return tempfile.TemporaryDirectory(prefix='case-',dir=root)

class NumericalCollectorTests(unittest.TestCase):
    def fixture(self,p):
        path=p/'scores.f32';payload=struct.pack('<2f',1.,2.);path.write_bytes(payload);h=hashlib.sha256(payload).hexdigest()
        source=dict(id='SSLS-00-S',origin_case_id='real-source-short',context_reset_index=0,prompt_token_ids=[1],max_output_tokens=8,logit_artifact_path=str(path))
        req=dict(schema='real-moe-numerical-request-6',source_sha256='b'*64,model_sha256='c'*64,model_contract=dict(layers=16),expected_runtime_environment={},mode='generate',threadpool_policy='explicit_persistent_owned',expected_process_mask=1,requests=[source])
        artifact=dict(path=str(path),sha256=h,bytes=8,rows=1,columns=2,format='little-endian IEEE754 F32 row-major',finite=True,existing_host_copy_used=True,new_GPU_reads=0,export_in_target_interval=False)
        actual=[dict(token=1,position=0,n_seq_id=1,seq_id=0,logits=True)]
        batch=dict(forward_index=0,rows=actual,actual_input_rows_sha256=hashlib.sha256(struct.pack('<5i',1,0,1,0,1)).hexdigest())
        row=dict(id=source['id'],origin_case_id=source['origin_case_id'],context_reset_index=0,memory_clear_data_requested=True,memory_clear_call_count_for_this_request=1,prompt_token_ids=[1],output_count=1,output_token_ids=[7],sampled_greedy_token_ids=[7],logit_observations=[dict(output_index=0,full_f32_logits_sha256=h,vocabulary_size=2)],forward_count=1,decode_forward_count=0,actual_batch_inputs=[batch],graph_observations=[{}],timing_usable_as_bare_reference=False,observer_extra_flush_ns=0,gpu_capture_error_code=0,full_logits_artifact=artifact)
        n=dict(schema='real-moe-numerical-native-6',request=req,process_id=10,source_sha256='b'*64,model_sha256_verified_by_parent='c'*64,observed_model_contract=req['model_contract'],raw_runtime_environment={},scheduler_callback_enabled=False,trace_callback_enabled=False,threadpool_policy=req['threadpool_policy'],process_affinity=dict(observed_mask=1),gpu_tap_hook_available=True,cpu_consumer_tap_enabled=False,results=[row],model_weights_loaded=True,actual_placement={},graph_or_allocator_modified=False)
        spec=dict(expected_request_schema=req['schema'],expected_result_schema=n['schema'],expected_source_sha256='b'*64)
        return n,req,spec

    def test_valid_actual_input_and_full_vector_acquisition_only(self):
        with temporary() as td:
            n,r,s=self.fixture(Path(td));out=c.verify_native(n,r,10,s)
            self.assertTrue(out['passed']);self.assertFalse(out['whole_model_accuracy_certified']);self.assertTrue(out['numerical_equality_not_assumed'])

    def test_actual_token_position_mask_mismatches_rejected(self):
        for field,value in [('token',2),('position',1),('logits',False),('seq_id',1)]:
            with temporary() as td:
                n,r,s=self.fixture(Path(td));n['results'][0]['actual_batch_inputs'][0]['rows'][0][field]=value
                with self.assertRaises(ValueError):c.verify_native(n,r,10,s)

    def test_reset_origin_and_new_source_bound_per_job(self):
        for mutate in ('origin','reset','source','schema'):
            with temporary() as td:
                n,r,s=self.fixture(Path(td))
                if mutate=='origin':n['results'][0]['origin_case_id']='different'
                elif mutate=='reset':n['results'][0]['context_reset_index']=1
                elif mutate=='source':n['source_sha256']='a'*64
                else:n['schema']='real-moe-tap-native-5'
                with self.assertRaises(ValueError):c.verify_native(n,r,10,s)

    def test_snapshot_cannot_add_GPU_read_or_enter_timing_interval(self):
        for key,value in [('new_GPU_reads',1),('export_in_target_interval',True)]:
            with temporary() as td:
                n,r,s=self.fixture(Path(td));n['results'][0]['full_logits_artifact'][key]=value
                with self.assertRaises(ValueError):c.verify_native(n,r,10,s)

    def test_wrong_raw_environment_and_gpu_errors_rejected(self):
        for condition in ('environment','GPUerror'):
            with temporary() as td:
                n,r,s=self.fixture(Path(td))
                if condition=='environment':n['raw_runtime_environment']={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                else:n['results'][0]['gpu_capture_error_code']=10
                with self.assertRaises(ValueError):c.verify_native(n,r,10,s)

    def test_sidecar_sha_failure_is_not_hidden_by_equal_top_logits(self):
        with temporary() as td:
            n,r,s=self.fixture(Path(td));Path(n['results'][0]['full_logits_artifact']['path']).write_bytes(struct.pack('<2f',1.,3.))
            with self.assertRaises(ValueError):c.verify_native(n,r,10,s)

    def test_collect_requires_root_receipt_before_model_or_launch(self):
        with temporary() as td,patch.object(c.common,'launch') as launch,patch.object(c,'file_sha') as modelhash:
            out=Path(td);write(out/'FREEZE_BARRIER.json',dict(manifest=dict(sha256='a'*64)))
            with self.assertRaises(FileNotFoundError):c.collect(out,'normal-SSLS')
            launch.assert_not_called();modelhash.assert_not_called();self.assertFalse((out/'jobs').exists())

if __name__=='__main__':unittest.main()
