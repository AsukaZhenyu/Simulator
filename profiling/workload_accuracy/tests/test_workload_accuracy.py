"""Software-only contract and real public compiler tests. No native execution."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from profiling.workload_accuracy.contract import (QUALITY, digest, replay_memory, stage_keys,
                                                  validate_case, validate_route_event_order, workload_key)
from profiling.workload_accuracy.examples import case, suite
from profiling.workload_accuracy.public_audit import export_audit
from profiling.workload_accuracy.run import (derive_case, evaluate, freeze, hardware_lock,
                                           collect, validate_suite, verify, write)


def memory(budget=32):
    events=[]
    for action,ident,component,requested,reserved,current in (
        ('allocate','a','resident_weights',12,16,16),('allocate','b','graph_buffers',15,16,32),
        ('release','b','graph_buffers',15,16,16),('release','a','resident_weights',12,16,0)):
        events.append(dict(event=action,id=ident,component=component,requested_bytes=requested,
                           reserved_bytes=reserved,live_reserved_bytes=current,round=0,layer=0))
    return dict(budget_bytes=budget,events=events,physical_capacity_verified=False,
                process_peak_bytes=None,cuda_private_workspace_bytes=None,
                cuda_private_workspace_bound_known=False,peak_tracked_reserved_bytes=32,
                final_tracked_reserved_bytes=0,tracked_budget_pass=True)


def record(split,seed,scale=1):
    c=derive_case(case(),seed);keys=stage_keys(c);samples=[]
    for si in range(5):
        stages=[dict(round=k['round'],layer=k['layer'],kind=k['kind'],compute_ns=int(10000*scale)) for k in keys]
        compute=sum(x['compute_ns'] for x in stages);gap=int(100000*scale)
        samples.append(dict(stages=stages,timing=dict(full_workload_wall_ns=compute+gap,
                           host_setup_materialization_gap_ns=gap)))
    return dict(split=split,case=c,native={'sample_timings':samples},raw_sha256=digest([split,seed]))


class WorkloadContractTests(unittest.TestCase):
    def test_examples_are_actual_multiple_layers_and_changing_rounds(self):
        for kind in ('dense','moe'):
            c=validate_case(case(kind))
            self.assertEqual(len(c['layers']),2)
            self.assertEqual([len(rd['hidden_input']) for rd in c['rounds']],[2,1,1,1])
            self.assertNotEqual(c['rounds'][1]['hidden_input'],c['rounds'][2]['hidden_input'])
        c=case('moe')
        self.assertNotEqual(c['rounds'][0]['routes'],c['rounds'][1]['routes'])
        self.assertEqual(c['rounds'][0]['routes'][0][0],c['rounds'][2]['routes'][0][0])

    def test_rejects_invalid_round_capacity_and_routes(self):
        mutations=[lambda c:c.update(max_context=4),
                   lambda c:c['rounds'][1]['routes'][0][0].__setitem__(1,2),
                   lambda c:c['rounds'][1]['route_scores'][0][0].__setitem__(1,float('nan')),
                   lambda c:c['rounds'][1]['hidden_input'][0].pop()]
        for mutate in mutations:
            c=case('moe');mutate(c)
            with self.assertRaises(ValueError):validate_case(c)

    def test_stage_key_uses_current_geometry_no_future_routes(self):
        c=case('moe');before=stage_keys(c)
        c['rounds'][3]['routes'][0][0]=[0,3]
        after=stage_keys(c)
        self.assertEqual(before[:6],after[:6])
        self.assertNotEqual(workload_key(case('moe')),workload_key(c))
        c=case();changed=copy.deepcopy(c);changed['rounds'][0]['hidden_input'][0][0]+=.1;changed['seed']+=1
        self.assertEqual(stage_keys(c),stage_keys(changed))
        self.assertEqual(workload_key(c),workload_key(changed))

    def test_route_barrier_precedes_residency_and_expert_execution(self):
        c=case('moe');events=[]
        for ri,rd in enumerate(c['rounds']):
            for kind in ('projection_complete','attention_complete','route_known','load',
                         'route_visible','expert_ffn_begin','expert_ffn_complete'):
                event=dict(event=kind,round=ri,sequence=len(events))
                if kind=='route_known':
                    event.update(routes=rd['routes'][0],selected_scores=rd['route_scores'][0],learned_router_executed=False)
                events.append(event)
        self.assertEqual(len(validate_route_event_order(c,0,events)),4)
        early=copy.deepcopy(events);early[2],early[3]=early[3],early[2]
        for index,event in enumerate(early):event['sequence']=index
        with self.assertRaisesRegex(ValueError,'before route_known'):validate_route_event_order(c,0,early)
        missing=[event for event in events if event['event']!='route_known']
        with self.assertRaisesRegex(ValueError,'route_known'):validate_route_event_order(c,0,missing)
        future=copy.deepcopy(events);future[2]['routes']=c['rounds'][1]['routes'][0]
        with self.assertRaisesRegex(ValueError,'current IDs'):validate_route_event_order(c,0,future)

    def test_budget_boundary_peak_and_release(self):
        self.assertEqual(replay_memory(memory())['tracked_peak_bytes'],32)
        with self.assertRaisesRegex(ValueError,'budget exceeded'):replay_memory(memory(31))
        self.assertEqual(replay_memory(memory(33))['tracked_peak_bytes'],32)
        bad=memory();bad['events'][2]['id']='a'
        with self.assertRaises(ValueError):replay_memory(bad)
        bad=memory();bad['physical_capacity_verified']=True
        with self.assertRaises(ValueError):replay_memory(bad)
        bad=memory();bad['cuda_private_workspace_bytes']=0
        with self.assertRaises(ValueError):replay_memory(bad)

    def test_disjoint_seed_process_coverage_original_quality(self):
        validate_suite(suite())
        bad=suite();bad['heldout_seeds'][0]=bad['profile_seeds'][0]
        with self.assertRaises(ValueError):validate_suite(bad)
        bad=suite();bad['quality']['max_heldout_relative_error']=.5
        with self.assertRaises(ValueError):validate_suite(bad)

    def test_profile_only_freeze_and_heldout_error_gate(self):
        profile=[record('profile',seed) for seed in (7101,7102,7103)]
        model=freeze(profile,{'test':'synthetic'},suite())
        good=evaluate([record('heldout',seed) for seed in (8101,8102,8103)],model,expected_processes=3)
        self.assertTrue(good['accepted'])
        bad=evaluate([record('heldout',seed,2) for seed in (8101,8102,8103)],model,expected_processes=3)
        self.assertFalse(bad['accepted'])
        self.assertTrue(all(not x['gates']['heldout_relative_error'] for x in bad['rows']))
        with self.assertRaisesRegex(ValueError,'heldout'):freeze([record('heldout',8101)],{},suite())

    def test_missing_stage_and_missing_process_are_rejected(self):
        model=freeze([record('profile',seed) for seed in (7101,7102,7103)],{},suite())
        del model['stages'][next(iter(model['stages']))]
        report=evaluate([record('heldout',8101)],model,expected_processes=3)
        self.assertFalse(report['accepted'])
        self.assertFalse(report['rows'][0]['gates']['exact_stage_and_host_gap_coverage'])
        self.assertFalse(report['rows'][0]['gates']['heldout_process_coverage'])

    def test_hardware_lock_exclusive_and_released(self):
        parent=ROOT/'profiling/workload_accuracy/_test_tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            lock=Path(directory)/'lock'
            with hardware_lock(lock):
                self.assertTrue(lock.exists())
                with self.assertRaises(FileExistsError):
                    with hardware_lock(lock):pass
            self.assertFalse(lock.exists())

    def test_timeout_nonzero_launch_and_invalid_json_retain_raw(self):
        parent=ROOT/'profiling/workload_accuracy/_test_tmp';parent.mkdir(exist_ok=True)
        binding={'native':{'runtime':{'synthetic.dll':'0'*64}}}
        fixtures=[subprocess.TimeoutExpired('synthetic',300,output=b'partial stdout',stderr=b'partial stderr'),
                  OSError('synthetic launch failure'),
                  subprocess.CompletedProcess(['synthetic'],7,'partial stdout','nonzero failure'),
                  subprocess.CompletedProcess(['synthetic'],0,'invalid JSON','diagnostic')]
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            for fixture in fixtures:
                kw={'side_effect':fixture} if isinstance(fixture,Exception) else {'return_value':fixture}
                with patch('profiling.workload_accuracy.run.subprocess.run',**kw):
                    raw,native=collect('synthetic',case(),Path(directory)/'case.json',samples=3,warmup=0,binding=binding)
                    self.assertIsNone(native)
                    self.assertIn('stdout',raw)
                    self.assertIn('stderr',raw)
                    self.assertEqual(raw['case'],case())

    def test_public_dense_moe_compile_and_runtime_match_state_shapes(self):
        parent=ROOT/'profiling/workload_accuracy/_test_tmp';parent.mkdir(exist_ok=True)
        for kind in ('dense','moe'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=parent) as directory:
                report=export_audit(case(kind),Path(directory)/'model')
                self.assertTrue(report['real_public_ModelPlan_runtime_success'])
                self.assertEqual(len(report['round_layer_checks']),8)
                self.assertEqual([x['history_after'] for x in report['round_layer_checks']],[2,2,3,3,4,4,5,5])
                self.assertFalse(report['exact_native_physical_profile_bind_allowed'])
                self.assertFalse(report['memory_capacity_equivalent'])

    def test_public_facade_refuses_different_variant(self):
        c=case();c['attention_gate']='sigmoid'
        with self.assertRaisesRegex(ValueError,'differs from explicit public Model'):export_audit(c,'unused')

    def test_mixed_actual_modelplan_supported_single_token_layout(self):
        c=case('moe');c['layers'][0]={'kind':'dense'}
        for rd in c['rounds']:
            rd['hidden_input']=rd['hidden_input'][:1]
            rd['routes'][0]=None;rd['route_scores'][0]=None
            rd['routes'][1]=rd['routes'][1][:1];rd['route_scores'][1]=rd['route_scores'][1][:1]
        parent=ROOT/'profiling/workload_accuracy/_test_tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            report=export_audit(c,Path(directory)/'model')
            self.assertTrue(report['real_public_ModelPlan_runtime_success'])
            self.assertEqual(report['model']['kind'],'mixed')
            self.assertEqual(report['round_layer_checks'][0]['task_kinds'],['projection','attention','dense_ffn'])
            self.assertIn('expert_ffn',report['round_layer_checks'][1]['task_kinds'])


if __name__=='__main__':unittest.main()
