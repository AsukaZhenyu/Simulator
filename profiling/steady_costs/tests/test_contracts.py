from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from profiling.steady_costs.contracts import (
    component_reference,digest,mix32,representative_components,semantic_program,uniform,validate_component,verify_float_output,
)
from profiling.steady_costs.campaign import fraction,make_request,mean,pack,read,receipt,stats,utc,verify_native,write_new
from profiling.steady_costs.calibration import action_key,errors,evaluate,freeze,restore_table
from profiling import host
from llm_infer_model.performance.steady_action_costs import SteadyActionCost,SteadyActionKey,SteadyActionTable,compile_steady_actions


def program(kind='matmul',**shape):
    p=dict(format_version='steady-component-program-1',kind=kind,device='gpu',threads=1,cuda_device=0,
        seed=123,shape=shape or dict(k=3,m=2,n=2),allocation_budget_bytes=1<<20)
    p['program_sha256']=digest(p)
    return p


class ContractsTests(unittest.TestCase):
    def test_index_generator_matches_scalar_modular_equations(self):
        def scalar(seed,index,ordinal):
            z=(seed+(index+1)*0x9e3779b9+(ordinal+1)*0x85ebca6b)&0xffffffff
            z^=z>>16;z=z*0x7feb352d&0xffffffff;z^=z>>15;z=z*0x846ca68b&0xffffffff;z^=z>>16
            return z
        for seed in (0,123,0xffffffff):
            for ordinal in (0,1):self.assertEqual(mix32(seed,20,ordinal).tolist(),[scalar(seed,i,ordinal) for i in range(20)])

    def test_gemm_oracle_has_correct_layout_and_nontrivial_values(self):
        p=program();out=np.asarray(component_reference(p)['values']).reshape(2,2)
        w=uniform(123,6,0,1/np.sqrt(3)).reshape(2,3);x=uniform(123,6,1).reshape(2,3)
        wanted=np.asarray([[sum(float(x[i,k])*float(w[j,k]) for k in range(3)) for j in range(2)] for i in range(2)])
        np.testing.assert_allclose(out,wanted,rtol=1e-12,atol=1e-12)
        self.assertTrue(np.any(out!=0))

    def test_pv_merged_layout_is_query_then_head_then_dimension(self):
        p=program('attention_pv',heads=4,kv_heads=2,head_dim=3,query_tokens=2,kv_tokens=5)
        expected=np.asarray(component_reference(p)['values']).reshape(2,4,3)
        v=uniform(123,3*5*2,0).reshape(2,5,3)
        weights=((mix32(123,5*2*4,1)>>np.uint32(8)).astype(float)+1)/(1<<24)
        weights=weights.reshape(4,2,5);weights=(weights/weights.sum(-1,keepdims=True)).astype(np.float32)
        for query in range(2):
            for head in range(4):
                for dimension in range(3):
                    wanted=sum(float(weights[head,query,k])*float(v[head//2,k,dimension]) for k in range(5))
                    self.assertAlmostEqual(expected[query,head,dimension],wanted,places=14)
        self.assertFalse(np.array_equal(expected.ravel(),expected.transpose(1,0,2).ravel()))

    def test_copy_oracle_checks_all_bytes(self):
        p=program('transfer',bytes=31,src='dram',dst='vram')
        oracle=component_reference(p)
        data=bytes(int(v)&255 for v in mix32(123,31,0))
        self.assertEqual(oracle['sha256'],hashlib.sha256(data).hexdigest())

    def test_shape_and_budget_fail_without_downgrading(self):
        for p in (program(k=1000,m=1000,n=1000),program('attention_pv',heads=3,kv_heads=2,head_dim=3,query_tokens=2,kv_tokens=5)):
            with self.assertRaises(ValueError):validate_component(p)
        p=program();p['shape']['k']=4
        with self.assertRaisesRegex(ValueError,'hash'):validate_component(p)

    def test_numeric_seed_is_not_performance_coordinate(self):
        p=program();q=dict(p,seed=124);q['program_sha256']=digest({k:v for k,v in q.items() if k!='program_sha256'})
        self.assertEqual(semantic_program(p),semantic_program(q))
        self.assertNotEqual(p['program_sha256'],q['program_sha256'])

    def test_all_representative_cases_are_bounded_explicit(self):
        cases=representative_components();self.assertEqual(len(cases),8)
        for c in cases:
            p=dict(c['program'],seed=7);p['program_sha256']=digest(p)
            validate_component(p)

    def test_unknown_numeric_result_is_not_a_success(self):
        with self.assertRaises(ValueError):verify_float_output([1,2],[1,float('nan')])
        with self.assertRaises(ValueError):verify_float_output([1,2],[1,3])

    def test_exact_means_not_sum_of_medians_and_explicit_zero(self):
        s=stats([1,1,100]);self.assertEqual(fraction(s['mean']),34)
        self.assertEqual(fraction(s['median']),1)
        k=SteadyActionKey('a','gpu','shape','boundary')
        cost=SteadyActionCost(k,Fraction(3,2),('source',),2,(1,2))
        table=restore_table(SteadyActionTable((cost,)).to_dict())
        result=compile_steady_actions((k,k),table).evaluate()
        self.assertEqual(result.duration_ns,4) # each actual completion is integer-ns rounded

    def test_cost_scope_retains_repetition_and_backend_identity(self):
        manifest=dict(binary='exe',sources={'exe':'sha'},hardware={},inherited_backend_environment={})
        binding=dict(signature='shape');p=program();request=make_request('action',component=True)
        first=action_key(manifest,binding,p,request)
        second=action_key(manifest,binding,p,dict(request,repeats=8))
        self.assertNotEqual(first,second)

    def test_write_once_lf_and_tail_errors_kept(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            p=Path(folder)/'out.json';write_new(p,{'a':1});self.assertNotIn(b'\r\n',p.read_bytes())
            with self.assertRaises(FileExistsError):write_new(p,{'a':2})
        result=errors(3,[3,3,35]);self.assertGreater(result['p90_ape'],.9)

    def test_generator_means_and_nonpositive_observations(self):
        self.assertEqual(mean(Fraction(v,2) for v in (1,2,3)),1)
        for values in ([],[0],[1,0],[-1]):
            with self.assertRaises(ValueError):errors(3,values)

    def test_native_protocol_rejects_counter_affinity_and_timer_changes(self):
        p=program();q=make_request('action',component=True);oracle=component_reference(p)
        native=dict(schema='steady-cost-native-1',program_sha256=p['program_sha256'],program_spec=p,request=q,process_id=1,
            clock=dict(clock_id=host.measurement_clock_id(),frequency_hz=1_000_000_000),
            collected_cpu_cycles=False,per_action_CPU_counter_queries=False,
            observed_runtime_flags=dict(GGML_CUDA_DISABLE_GRAPHS='1',NVIDIA_TF32_OVERRIDE='0',matmul_precision='GGML_PREC_F32'),
            main_thread_affinity=dict(schema='steady-owned-main-thread-affinity-1',requested_mask=4,observed_mask=4,final_mask=4,
                set_succeeded=True,applied_before_backend_allocation_and_warmup=True,restored=True),
            warmup=dict(actual_iterations=20,wall_ns=500_000_000,begin_tick=0,end_tick=500_000_000,completed_outside_all_trial_timers=True),
            native_repeat_consistency=dict(passed=True),
            component_output=dict(kind='matmul',values=oracle['values'],independent_oracle_applied_in_native=False),
            trials=[dict(trial=i,repeats=q['repeats'],wall_ns=100,begin_tick=500_000_000+100*i,end_tick=500_000_100+100*i)
                for i in range(q['trials'])])
        self.assertTrue(verify_native(p,q,native,oracle)['passed'])
        for field,value in (('collected_cpu_cycles',True),('clock',dict(clock_id='unknown',frequency_hz=1000))):
            bad=deepcopy(native);bad[field]=value
            with self.assertRaises(ValueError):verify_native(p,q,bad,oracle)
        for section,field,value in (('main_thread_affinity','observed_mask',8),('warmup','wall_ns',499_999_999)):
            bad=deepcopy(native);bad[section][field]=value
            with self.assertRaises(ValueError):verify_native(p,q,bad,oracle)
        bad=deepcopy(native);bad['trials'][0]['wall_ns']=98
        with self.assertRaises(ValueError):verify_native(p,q,bad,oracle)

    def test_fresh_process_pipeline_freezes_only_profile_then_rejects_scope_change(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder);source=root/'source.py';source.write_text('# fixture\n',encoding='utf-8',newline='\n')
            programs={};sources={str(source):receipt(source)['sha256']};jobs=[]
            for split,seed in (('profile',123),('heldout',999)):
                p=program();p['seed']=seed;p['program_sha256']=digest({k:v for k,v in p.items() if k!='program_sha256'})
                path=root/(split+'.json');write_new(path,p)
                programs[split]=dict(program=receipt(path),seed=seed,signature=digest(semantic_program(p)),kind='component')
                sources[str(path)]=receipt(path)['sha256']
                for repeat in range(3):jobs.append(dict(id=f'{split}-{repeat}',case_id='tiny',split=split,program_key=split,
                    repeat=repeat,request=make_request('action',component=True,split=split),role='complete_component'))
            manifest=dict(schema='steady-cost-campaign-1',binary=str(source),sources=sources,source_inventory_sha256=digest(sources),
                programs=programs,jobs=jobs,cases=[dict(id='tiny',kind='component')],hardware={},
                inherited_backend_environment={},protocol=dict(quality=dict(max_process_mean_ape=.1,
                    max_trial_average_p90_ape=.2,max_observer_relative_mean_difference=.05,max_process_trial_cv=.1)))
            write_new(root/'manifest.json',manifest)
            def record(job,pid,wall):
                directory=root/'jobs'/job['id'];directory.mkdir(parents=True)
                native=dict(process_id=pid,program_sha256=read(programs[job['program_key']]['program']['path'])['program_sha256'],request=job['request'],
                    trials=[dict(wall_ns=wall*job['request']['repeats'],repeats=job['request']['repeats']) for i in range(job['request']['trials'])])
                write_new(directory/'result.json',native)
                execution=dict(job=job,status='completed',returncode=0,forced_termination=False,process_exited_verified=True,
                    numerical=dict(passed=True),resource_state_used_as_model_input=False,program=programs[job['program_key']],
                    native=receipt(directory/'result.json'),process_id=pid,started_at_utc=utc())
                write_new(directory/'execution.json',execution)
            for job in jobs[:3]:record(job,1+job['repeat'],100)
            result=freeze(root);self.assertEqual(result['entries'],1)
            frozen=read(root/'profile-frozen.json');self.assertFalse(frozen['heldout_measurements_read'])
            write_new(root/'heldout-started.json',dict(barrier=read(root/'frozen-barrier.json')))
            for job in jobs[3:]:record(job,11+job['repeat'],101)
            self.assertEqual(evaluate(root)['passed_cases'],1)
            with self.assertRaisesRegex(ValueError,'heldout acquisition'):freeze(root)
            bad=deepcopy(manifest);bad['jobs'][3]['request']['repeats']=8
            path=root/'manifest.json';path.unlink();write_new(path,bad)
            # The whole campaign identity changes before any mismatched prices
            # can be applied, including a repeat-boundary change.
            with self.assertRaisesRegex(ValueError,'freeze bytes'):evaluate(root)


if __name__=='__main__':unittest.main()
