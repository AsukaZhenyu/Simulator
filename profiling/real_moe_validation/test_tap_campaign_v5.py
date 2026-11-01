"""Software-only identity, request-derivation and no-launch authorization gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import tap_campaign_v5 as c
from .dataset import read,write

def temporary_fixture():
    # Windows sandbox system TEMP may permit directory creation but deny writes.
    # This tool-owned cleanup target is always below the checked workspace root.
    root=(c.ROOT/'build/real-moe-validation-20261027/software-v5-collector-fixtures').resolve()
    if c.ROOT.resolve() not in root.parents:raise ValueError('test fixture outside workspace')
    root.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='case-',dir=root)

class CampaignV5Tests(unittest.TestCase):
    def fixture(self,base):
        old=base/'old';build=base/'build';old.mkdir();build.mkdir();(old/'requests').mkdir()
        oldexe=base/'v4.exe';newexe=base/'v5.exe';oldexe.write_bytes(b'frozen v4 exe');newexe.write_bytes(b'private v5 exe')
        rows=[dict(id='short',prompt_token_ids=[1,2],max_output_tokens=8),dict(id='long',prompt_token_ids=[1,2,3],max_output_tokens=8)]
        req=dict(schema='real-moe-tap-request-4',source_sha256='a'*64,mode='generate',requests=rows,placement='cpu',minimum_free_host_bytes=5824125760)
        for placement in ('cpu','hybrid'):write(old/'requests'/f'{placement}-original-normal.json',dict(req,placement=placement))
        write(old/'manifest.json',dict(model=dict(path='not-loaded.gguf',sha256='c'*64,bytes=4213513024),model_contract=dict(layers=16),capacity_policy=dict(path='frozen-policy.json',sha256='d'*64),native_guard_bytes=5824125760,parent_preflight_bytes=6076332864,resident_peak_limit_bytes=5287254848,jobs=[dict(job_id=f'{p}-original-normal',request=c.receipt(old/'requests'/f'{p}-original-normal.json')) for p in ('cpu','hybrid')],bindings={'original_cpu_dll':dict(binary=c.receipt(oldexe),runtime=[],runtime_dirs=['unchanged-runtime'])}))
        write(build/'PREPARED.json',dict(native_source_sha256='b'*64));write(build/'GPU_OBSERVER_REVIEW.json',dict(software_only=True))
        write(build/'BUILD_COMPLETE.json',dict(schema='real-moe-gpu-consumer-build-complete-5',binaries=[c.receipt(newexe)]))
        return old,build

    def test_fixed_order_is_eight_serial_jobs(self):
        self.assertEqual(c.ORDER,('cpu-original-normal','cpu-private-normal','cpu-replay','cpu-capture','hybrid-original-normal','hybrid-private-normal','hybrid-replay','hybrid-capture'))

    def test_pipeline_stops_at_first_failure_without_retry_or_later_job(self):
        with temporary_fixture() as td:
            out=Path(td);write(out/'jobs/cpu-private-normal/execution.json',dict(status='failed'))
            with patch.object(c,'authorization'),patch.object(c,'collect',side_effect=[dict(status='completed'),dict(status='failed')]) as collect,patch.object(c,'make_pairs') as pairs:
                result=c.collect_all(out)
            self.assertFalse(result['completed']);self.assertEqual(collect.call_count,2);pairs.assert_not_called()
            self.assertEqual(read(out/'ACQUISITION_STOPPED.json')['remaining_not_run'],list(c.ORDER[2:]));self.assertFalse((out/'ACQUISITION_COMPLETE.json').exists())

    def test_prepare_binds_old_executable_and_schema_per_job_without_launch(self):
        with temporary_fixture() as td,patch.object(c.common,'launch') as launch:
            p=Path(td);old,build=self.fixture(p);out=p/'out';c.prepare(out,build,old);m=read(out/'manifest.json')
            original=read(out/'requests/cpu-original-normal.json');private=read(out/'requests/cpu-private-normal.json')
            self.assertEqual(original['schema'],'real-moe-tap-request-4');self.assertEqual(private['schema'],'real-moe-tap-request-5')
            self.assertEqual(c.semantic_request(original),c.semantic_request(private));self.assertNotEqual(original['source_sha256'],private['source_sha256'])
            self.assertEqual(Path(m['bindings']['original_v4']['binary']['path']).read_bytes(),b'frozen v4 exe')
            self.assertEqual(m['native_guard_bytes'],5824125760);self.assertEqual(m['parent_preflight_bytes'],6076332864);self.assertEqual(m['resident_peak_limit_bytes'],5287254848)
            self.assertEqual([j['job_id'] for j in m['jobs']],list(c.ORDER));self.assertFalse(m['timings_usable_as_bare_reference']);launch.assert_not_called()

    def test_prepare_refuses_existing_directory(self):
        with temporary_fixture() as td:
            p=Path(td);old,build=self.fixture(p);out=p/'out';out.mkdir()
            with self.assertRaises(ValueError):c.prepare(out,build,old)

    def test_freeze_does_not_authorize_collection(self):
        with temporary_fixture() as td,patch.object(c.common,'launch') as launch:
            p=Path(td);old,build=self.fixture(p);out=p/'out';c.prepare(out,build,old);tests=p/'tests.json';write(tests,dict(returncode=0,hardware_executed=False))
            c.freeze(out,tests);self.assertFalse(read(out/'FREEZE_BARRIER.json')['collection_authorized'])
            with self.assertRaises(FileNotFoundError):c.collect(out,'cpu-original-normal')
            self.assertFalse((out/'jobs').exists());launch.assert_not_called()

    def test_authorization_rejects_wrong_manifest_hash_before_launch(self):
        with temporary_fixture() as td,patch.object(c.common,'launch') as launch:
            p=Path(td);old,build=self.fixture(p);out=p/'out';c.prepare(out,build,old);tests=p/'tests.json';write(tests,dict(returncode=0,hardware_executed=False));c.freeze(out,tests)
            write(out/'ROOT_AUTHORIZATION_RECEIPT.json',dict(collection_authorized=True,manifest_sha256='0'*64,freeze_barrier_sha256=c.sha(out/'FREEZE_BARRIER.json')))
            with self.assertRaises(ValueError):c.collect(out,'cpu-original-normal')
            launch.assert_not_called()

    def test_pair_derivation_changes_only_mode_and_actual_output_ids(self):
        with temporary_fixture() as td:
            p=Path(td);old,build=self.fixture(p);out=p/'out';c.prepare(out,build,old);req=read(out/'requests/cpu-private-normal.json');folder=out/'jobs/cpu-private-normal'
            write(folder/'execution.json',dict(status='completed'));write(folder/'result.json',dict(request=req,results=[dict(id=r['id'],output_token_ids=[7,8]) for r in req['requests']]))
            plans=c.make_pairs(out,'cpu');self.assertEqual(len(plans),2)
            for plan in plans:
                paired=read(plan['request']['path']);expected=copy.deepcopy(req);expected['mode']=paired['mode']
                for row in expected['requests']:row['output_token_ids']=[7,8]
                self.assertEqual(paired,expected);self.assertEqual(paired['schema'],'real-moe-tap-request-5')

    def native(self):
        req=dict(schema='real-moe-tap-request-5',source_sha256='b'*64,model_sha256='c'*64,model_contract=dict(layers=16),threadpool_policy='explicit_persistent_owned',expected_process_mask=3,mode='generate',requests=[dict(id='q',prompt_token_ids=[1])])
        binding=dict(request_schema=req['schema'],result_schema='real-moe-tap-native-5',source_sha256=req['source_sha256']);spec=dict(expected_request_schema=req['schema'],expected_result_schema=binding['result_schema'],expected_source_sha256=req['source_sha256'])
        row=dict(id='q',prompt_token_ids=[1],output_count=1,sampled_greedy_token_ids=[2],output_token_ids=[2],decode_forward_count=0,logit_observations=[{}],graph_observations=[{}],forward_count=1,timing_usable_as_bare_reference=False,routes=[],gpu_capture_error_code=0,gpu_observer_copy_count=0,gpu_copy_inventory=[],observer_extra_flush_ns=0,observer_completion_boundary='existing llama_synchronize only')
        n=dict(schema=binding['result_schema'],request=req,process_id=11,source_sha256=req['source_sha256'],model_sha256_verified_by_parent=req['model_sha256'],scheduler_callback_enabled=False,trace_callback_enabled=False,cpu_consumer_tap_enabled=False,observed_model_contract=req['model_contract'],threadpool_policy=req['threadpool_policy'],process_affinity=dict(observed_mask=3),gpu_tap_hook_available=True,results=[row])
        return n,req,spec,binding

    def test_per_job_schema_cannot_be_normalized_to_another_version(self):
        n,r,s,b=self.native();self.assertTrue(c.verify_native(n,r,11,s,b)['acquisition_only']);n['schema']='real-moe-tap-native-4'
        with self.assertRaises(ValueError):c.verify_native(n,r,11,s,b)

    def test_wrong_native_source_binding_rejected(self):
        n,r,s,b=self.native();n['source_sha256']='e'*64
        with self.assertRaises(ValueError):c.verify_native(n,r,11,s,b)

    def test_GPU_unsupported_is_not_promoted_to_completed_verification(self):
        n,r,s,b=self.native();n['results'][0]['gpu_capture_error_code']=10
        with self.assertRaises(ValueError):c.verify_native(n,r,11,s,b)

    def test_extra_observer_flush_and_bare_promotion_rejected(self):
        for field,value in [('observer_extra_flush_ns',1),('timing_usable_as_bare_reference',True)]:
            n,r,s,b=self.native();n['results'][0][field]=value
            with self.assertRaises(ValueError):c.verify_native(n,r,11,s,b)

if __name__=='__main__':unittest.main()
