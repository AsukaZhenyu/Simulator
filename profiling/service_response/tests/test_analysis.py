"""Offline source, isolation, causal-history and independent-validation tests."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
for folder in (ROOT,ROOT/'modeling'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))

from profiling.service_response import analysis as a
from profiling.service_response import evidence as e
from profiling.service_response.calibration import exact,worker_proxies


def protocol():
    return dict(schema='service-response-protocol-1',model_name='signed_inclusive_response_reservoir_v1',
        competitors=[dict(name='cpu',request_key='cpu_replay',actor_role='cpu-worker'),dict(name='gpu',request_key='gpu_replay',actor_role='gpu-worker')],
        capacity=dict(numerator=1,denominator=1),activity_weight=dict(numerator=1,denominator=1),backlog_weight=dict(numerator=1,denominator=1),
        coefficient_domain='signed_inclusive_response',profile_settle_ns=20,minimum_independent_runs=3,
        minimum_cell_samples_per_process=1,within_process_cv_max=.15,between_process_cv_max=.20,relative_error_max=.20,
        max_step_ns=1,max_points=100000,initial_work={n:dict(numerator=0,denominator=1) for n in ('cpu','gpu')},
        initial_history_evidence=['synthetic known initial normalized-work zero; no physical quiescence claim'],
        control_overhead_policy='baseline_fixed_ns_unqualified',proxy_mapping='arrival=drain+capacity/fill_median; drain=capacity/off_tail_median',
        combined_state_role='validation_only',model_selection='predeclared_before_profile_and_heldout',heldout_refit=False)


class ServiceProfilingTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'build').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='service-profile-tests-',dir=ROOT/'build');self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.counter=0

    def record(self,split='profile',*,cpu=20,gpu=30,combined=40,historical=False):
        self.counter+=1;pid=100+self.counter;seed=1000+self.counter+(10000 if split=='heldout' else 0)
        requests=[(0,100,0,0),(100,200,1,0),(200,300,0,0),(300,400,0,1),(400,500,0,0),(500,600,1,1),(600,900,0,0)]
        segments=[dict(begin_ns=left,end_ns=right,requests=dict(cpu=c,gpu=g),host_memory_bytes=0,device_memory_bytes=0) for left,right,c,g in requests]
        entries=[(10,10),(120,cpu),(210,10),(320,gpu),(410,10),(520,combined),(650,10)]
        if split=='heldout':entries=[(10,10),(145,20),(201,10),(250,10),(345,30),(401,10),(450,10),(650,10)]
        inv=[]
        for index,(begin,duration) in enumerate(entries):
            inv.append(dict(observation_id='i:'+str(index),begin_ns=begin,end_ns=begin+duration+10,
                actions=[] if split=='heldout' else [[0,begin+1,begin+1+duration],[1,begin+duration+3,begin+duration+5]]))
        program_actions=[dict(kind='COMPUTE',device='gpu',target_id='A'),dict(kind='ADVANCE')]
        workers={}
        for name,on,tail in (('cpu',100,5),('gpu',300,7)):
            calls=[[on,on,on+10],[on+10,on+10,on+20],[on+90,on+90,on+100+tail],
                   [500,500,510],[590,590,600+tail]]
            workers[name]=dict(actor_role=name+'-worker',process_id=pid+1000+(name=='gpu'),target_action_index=0,
                work_scope='single target',ready_ns=-20,sampling_begin_ns=0,sampling_end_ns=900,
                authorized_calls=calls,intervals_are_hardware_busy_time=False)
        source=self.path/(str(pid)+'.json');source.write_text(json.dumps(dict(invocations=inv,segments=segments,workers=workers)),encoding='utf-8')
        result=dict(schema=e.SCHEMA,run_id=split+'-'+str(pid),split=split,seed=seed,process_id=pid,
            started_at_utc='2026-10-22T02:00:00Z' if split=='heldout' else '2026-10-22T00:00:00Z',
            ended_at_utc='2026-10-22T02:01:00Z' if split=='heldout' else '2026-10-22T00:01:00Z',
            context={'synthetic':'same'},context_sha256=e.digest({'synthetic':'same'}),contract_sha256='a'*64,manifest_sha256='b'*64,
            service_protocol_sha256=None if historical else e.digest(protocol()),source_inventory_sha256='c'*64,
            original_normalized_sha256='d'*64,competitors=protocol()['competitors'],token_inventory=e.token_inventory(program_actions),
            action_inventory=program_actions,clock={'clock_id':'synthetic-ns','frequency_hz':1000000000},origin_tick=0,
            instrumentation='bare' if split=='heldout' else 'stamped',segments=segments,calibration_segments=deepcopy(segments),
            invocations=inv,workers=workers,numerical_verified=True,source_files={str(source):{'sha256':e.SourceAudit().hash(source),'bytes':source.stat().st_size}},
            source_verified=True,all_samples_retained=True,outliers_removed=0)
        result['record_sha256']=e.digest(result);return result

    def resign(self,record):
        record.pop('record_sha256',None);record['record_sha256']=e.digest(record);return record

    def frozen(self,**kwargs):
        rows=[self.record(**kwargs) for _ in range(3)]
        return rows,a.freeze_profile(rows,protocol=protocol(),frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)

    def barrier(self,frozen):
        return a.make_barrier(frozen,created_at_utc='2026-10-22T01:30:00Z',frozen_file_sha256='e'*64,manifest_sha256='b'*64,verify_sources=False)

    def test_isolated_coefficients_and_baseline_only_controls(self):
        rows,frozen=self.frozen()
        coefficients={v['name']:exact(v['value']) for v in frozen['service_model']['parameters'] if v['name'].startswith('interference:')}
        self.assertEqual(list(coefficients.values()),[Fraction(1,2),Fraction(1)])
        self.assertFalse(frozen['whole_layer_duration_used_for_fit']);self.assertFalse(frozen['combined_state_used_for_fit'])
        self.assertFalse(frozen['controls_state_independence_qualified'])
        self.assertFalse(any('invocations' in row for row in frozen['profile_sources']))

    def test_combined_actions_do_not_change_any_parameter_values(self):
        _,one=self.frozen(combined=40);_,two=self.frozen(combined=70)
        pick=lambda f:{p['name']:p['value'] for p in f['service_model']['parameters']}
        self.assertEqual(pick(one),pick(two))
        self.assertEqual({p['name']:p['value'] for p in one['base_bundle']['parameters']},
                         {p['name']:p['value'] for p in two['base_bundle']['parameters']})

    def test_signed_facilitation_is_not_clipped(self):
        _,frozen=self.frozen(cpu=5)
        self.assertEqual(exact(frozen['negative_coefficients'][0]['coefficient']),Fraction(-1,4))
        self.assertFalse(frozen['physical_mechanism_identified'])

    def test_nonpositive_denominator_is_unsupported(self):
        _,frozen=self.frozen(cpu=5,gpu=5)
        self.assertTrue(any('denominator_nonpositive' in issue for issue in frozen['unsupported_model_issues']))
        held=[self.record('heldout') for _ in range(3)]
        result=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        self.assertEqual(result['primary_summary']['predictable_count'],0)
        self.assertEqual(result['primary_summary']['sample_count'],24)

    def test_worker_proxy_edges_and_unknown_are_retained(self):
        record=self.record();proxy=worker_proxies(record,'cpu')
        self.assertEqual(proxy['off_tail_ns'],[5])
        self.assertEqual([edge['isolated_profile_fit_eligible'] for edge in proxy['off_edges']],[True,False])
        record['workers']['cpu']['sampling_end_ns']=199
        proxy=worker_proxies(record,'cpu')
        self.assertEqual(len(proxy['off_edges']),2);self.assertTrue(all(v['tail_ns'] is None for v in proxy['off_edges']))

    def test_zero_unknown_proxy_does_not_become_infinite_rate(self):
        rows=[self.record() for _ in range(3)]
        for r in rows:
            for worker in r['workers'].values():
                worker['authorized_calls']=[v for v in worker['authorized_calls'] if v[0] not in (190,390,590)]
            self.resign(r)
        frozen=a.freeze_profile(rows,protocol=protocol(),frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)
        self.assertFalse(frozen['model_parameters_available'])
        self.assertTrue(any(p['value'] is None for p in frozen['service_model']['parameters']))

    def test_causal_history_is_shared_and_not_reset_per_invocation(self):
        _,frozen=self.frozen();held=[self.record('heldout') for _ in range(3)]
        result=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        rows=[r for r in result['predictions'] if r['run_id']==held[0]['run_id']]
        self.assertGreater(rows[2]['predicted_ns'],rows[3]['predicted_ns'])
        self.assertEqual(rows[0]['predicted_ns'],20)
        self.assertFalse(result['work_history_reset_per_invocation']);self.assertFalse(result['accepted'])
        self.assertEqual(len(result['g_trajectories']),3)

    def test_observed_end_cannot_change_predicted_duration(self):
        _,frozen=self.frozen();held=[self.record('heldout') for _ in range(3)]
        before=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        held[0]['invocations'][0]['end_ns']+=3;self.resign(held[0])
        after=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        self.assertEqual(before['predictions'][0]['predicted_ns'],after['predictions'][0]['predicted_ns'])
        self.assertNotEqual(before['predictions'][0]['absolute_relative_error'],after['predictions'][0]['absolute_relative_error'])

    def test_independence_barrier_and_protocol_leak_guards(self):
        profiles,frozen=self.frozen();held=[self.record('heldout') for _ in range(3)]
        held[0]['seed']=profiles[0]['seed'];self.resign(held[0])
        with self.assertRaisesRegex(ValueError,'profile identity reused'):a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        bad=deepcopy(frozen);bad['protocol']['heldout_refit']=True
        with self.assertRaises(ValueError):a.make_barrier(bad,created_at_utc='2026-10-22T01:30:00Z',frozen_file_sha256='e'*64,manifest_sha256='b'*64)
        with self.assertRaisesRegex(ValueError,'not bound before collection'):self.frozen(historical=True)

    def test_changed_raw_bytes_are_rejected(self):
        rows=[self.record() for _ in range(3)]
        Path(next(iter(rows[0]['source_files']))).write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):a.freeze_profile(rows,protocol=protocol(),frozen_at_utc='2026-10-22T01:00:00Z')

    def test_all_rows_and_error_coverage_categories_are_retained(self):
        _,frozen=self.frozen();held=[self.record('heldout') for _ in range(3)]
        for r in held:
            r['invocations'].append(dict(observation_id='cross',begin_ns=195,end_ns=215,actions=[]));self.resign(r)
        result=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        self.assertEqual(len(result['predictions']),27)
        self.assertEqual(result['buckets']['request_crossing']['sample_count'],3)
        self.assertEqual(sum(v['sample_count'] for v in result['buckets'].values()),27)
        self.assertGreater(result['recovery']['sample_count'],0)

    def test_streamed_producer_reader_exactly_matches_native_objects(self):
        path=self.path/'native.json'
        native=dict(format_version='state-response-native-v1',role='main',actions=[dict(kind='COMPUTE')],
            clock=dict(clock_id='synthetic',frequency_hz=1000000000),
            blocks=[dict(action_intervals=[dict(iteration=0,action_index=0,begin_tick=10,end_tick=20,elapsed_ns=10,kind='COMPUTE')],
                         iterations=[dict(iteration=0,begin_tick=9,end_tick=21,elapsed_ns=12,enabled_check_tick=8)])],
            ports={'unused':[1,2,3]})
        path.write_text(json.dumps(native,indent=2)+'\n',encoding='utf-8')
        meta,calls,actions=e.read_native(path)
        self.assertEqual(calls,[(0,0,9,21,12,8)])
        self.assertEqual(actions,[(0,0,0,10,20,10,'COMPUTE')])
        self.assertNotIn('ports',meta)

    def test_tail_binding_stream_handles_large_intermediate_payload(self):
        path=self.path/'normalized.json';data={'large':['x'*100 for _ in range(5000)],'source_files':{'owned':'a'*64},'raw_bindings':{'program':'owned'}}
        path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        self.assertEqual(e.tail_field(path,'source_files'),data['source_files'])
        self.assertEqual(e.tail_field(path,'raw_bindings'),data['raw_bindings'])

    def test_saturation_assumption_is_gated(self):
        rows=[self.record() for _ in range(3)];p=protocol();p['profile_settle_ns']=1
        for row in rows:row['service_protocol_sha256']=e.digest(p);self.resign(row)
        frozen=a.freeze_profile(rows,protocol=p,frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)
        self.assertFalse(frozen['model_parameters_available'])
        self.assertTrue(any('assumed_saturation' in issue for issue in frozen['unsupported_model_issues']))

    def test_baseline_drain_plateau_assumption_is_gated(self):
        rows=[self.record() for _ in range(3)];p=protocol();p['profile_settle_ns']=12
        for row in rows:
            row['service_protocol_sha256']=e.digest(p)
            row['workers']['gpu']['authorized_calls'][2][2]=417
            self.resign(row)
        frozen=a.freeze_profile(rows,protocol=p,frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)
        self.assertIn('gpu:declared_settle_shorter_than_assumed_drain_time',frozen['unsupported_model_issues'])

    def recipes(self,rows):
        lookup={}
        for row in rows:
            path=self.path/(row['run_id']+'-execution.json');path.write_text(json.dumps({'run_id':row['run_id']}),encoding='utf-8')
            row['extraction_inputs']=dict(manifest_path=str(self.path/'synthetic-manifest.json'),execution_path=str(path))
            row['source_files'][str(path)]={'sha256':e.SourceAudit().hash(path),'bytes':path.stat().st_size}
            self.resign(row);lookup[row['run_id']]=row
        return lambda manifest,execution,**kwargs:deepcopy(lookup[execution['run_id']])

    def test_raw_recipe_derivation_guard_rejects_resigned_derived_change(self):
        row=self.record();rebuild=self.recipes([row]);original=deepcopy(row)
        with patch('profiling.service_response.evidence.extract_job',side_effect=rebuild):e.validate_record(original)
        original['invocations'][0]['actions'][0][2]+=1;self.resign(original)
        with patch('profiling.service_response.evidence.extract_job',side_effect=rebuild):
            with self.assertRaisesRegex(ValueError,'does not reconstruct'):e.validate_record(original)

    def test_frozen_parameters_rebuild_from_profile_recipe(self):
        rows=[self.record() for _ in range(3)];rebuild=self.recipes(rows)
        frozen=a.freeze_profile(rows,protocol=protocol(),frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)
        with patch('profiling.service_response.evidence.extract_job',side_effect=rebuild):a.validate_frozen(frozen)
        frozen['base_bundle']['parameters'][0]['value']['numerator']+=1
        frozen.pop('artifact_sha256');frozen['artifact_sha256']=e.digest(frozen)
        with patch('profiling.service_response.evidence.extract_job',side_effect=rebuild):
            with self.assertRaisesRegex(ValueError,'do not reconstruct'):a.validate_frozen(frozen)

    def test_missing_control_baseline_is_explicitly_unsupported(self):
        rows=[self.record() for _ in range(3)]
        for row in rows:
            row['token_inventory'].append(dict(token='missing-control',kind='ADVANCE',action_index=2,transformed=False));self.resign(row)
        frozen=a.freeze_profile(rows,protocol=protocol(),frozen_at_utc='2026-10-22T01:00:00Z',verify_sources=False)
        self.assertFalse(frozen['model_parameters_available'])
        self.assertIn('missing-control:unknown_baseline_response',frozen['unsupported_model_issues'])

    def test_numerical_step_comparison_never_selects_or_refits_primary(self):
        _,frozen=self.frozen();held=[self.record('heldout') for _ in range(3)];saved=deepcopy(frozen)
        original=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False)
        compared=a.evaluate_frozen(frozen,held,barrier=self.barrier(frozen),verify_sources=False,numerical_comparison_max_step_ns=2)
        self.assertEqual([r['predicted_ns'] for r in original['predictions']],[r['predicted_ns'] for r in compared['predictions']])
        self.assertEqual(compared['numerical_step_sensitivity']['paired_prediction_count'],24)
        self.assertEqual(frozen,saved)


if __name__=='__main__':unittest.main()
