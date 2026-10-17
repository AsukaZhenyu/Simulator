"""End-to-end synthetic MoE routing, expert grouping and cache continuation."""
from __future__ import annotations
import copy
import dataclasses
import unittest
from llm_infer_model.m2.costs import MacroShape,parse_cost_table
from llm_infer_model.m2.expand import compile_invocation,plan_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import parse_layer_document,parse_invocation_document,state_in_from_state_out,weight_id,to_state_document,parse_state_document
from llm_infer_model.tensor.engine import ACTION_READ_SSD,ACTION_COPY,ACTION_COMPUTE,ACTION_ADVANCE,Action
from llm_infer_model.tensor.spec import InvalidInput,Unsupported


def moe_layer_document(*,shared=False):
    weights=[dict(id='W_proj',storage_bytes=32,initial_location='dram'),dict(id='W_router',storage_bytes=16,initial_location='dram')]
    weights += [dict(id='E'+str(e),storage_bytes=64,initial_location='ssd') for e in range(4)]
    params=dict(experts=4,top_k=2,expert_weight_bytes=64,expert_weights=['E0','E1','E2','E3'])
    if shared:
        weights.append(dict(id='shared',storage_bytes=64,initial_location='ssd'))
        params.update(shared_expert_bytes=64,shared_expert_weight='shared')
    return dict(format_version='m2-layer-0.6',id='moe-small',provenance='synthetic_layer',hidden_size=4,
        hidden_bytes_per_token=16,kv_bytes_per_token=16,attention='full_causal',ffn='moe',ffn_params=params,
        weights=weights,tasks=[dict(id='proj',kind='projection',depends_on=[],weight='W_proj'),
        dict(id='attn',kind='attention',depends_on=['proj']),dict(id='router',kind='router',depends_on=['attn'],weight='W_router'),
        dict(id='expert',kind='expert_ffn',depends_on=['router']),dict(id='combine',kind='combine',depends_on=['expert'])])


def moe_invocation_document(layer,*,devices=('cpu',),phase='prefill',requests=None,mode='causal_replay'):
    requests=requests or [dict(request_id='r0',position_start=0,tokens=2 if phase=='prefill' else 1,input_location='dram',output_location='dram')]
    states=[dict(request_id=r['request_id'],next_position=0,version=1,state_location='dram',blocks=[]) for r in requests]
    routes=[]
    for r in requests:
        for pos in range(r['position_start'],r['position_start']+r['tokens']):
            idx=len(routes)%3; selected=([0,1],[1,2],[0,2])[idx]
            routes.append(dict(request_id=r['request_id'],layer_id=layer.id,absolute_position=pos,expert_ids=selected))
    return dict(format_version='m2-layer-0.6',id='moe-'+phase,layer=layer.id,layer_file='moe-interface-synthetic.layer.json',
        costs_file='moe-interface-synthetic.costs.json',phase=phase,devices=list(devices),route_trace=routes,
        route_source='synthetic_exact_top_k_fixture',information_mode=mode,
        hardware=dict(dram_capacity_bytes=4096,vram_capacity_bytes=4096,dram_alignment_bytes=1,vram_alignment_bytes=1,
                      dram_runtime_reserved_bytes=0,vram_runtime_reserved_bytes=0),
        requests=requests,state_in=dict(requests=states,weight_copies=[dict(id=w.id,version=1,
            storage_bytes=w.storage_bytes,locations=[w.initial_location]) for w in layer.weights]))


def moe_cost_document(layer,invocation):
    shapes={}
    for r in invocation.requests:
        hist=invocation.state_in.for_request(r.request_id).next_position
        shape=MacroShape(invocation.phase,invocation.batch,r.tokens,hist)
        for kind in ('projection','attention','router','combine'): shapes[(kind,shape)]=None
    counts={}
    for row in invocation.route_trace:
        for expert in row.expert_ids: counts[expert]=counts.get(expert,0)+1
    for count in counts.values(): shapes[('expert_ffn',MacroShape(invocation.phase,invocation.batch,count,0))]=None
    if layer.ffn_params.get('shared_expert_weight'):
        shapes[('shared_expert_ffn',MacroShape(invocation.phase,invocation.batch,sum(r.tokens for r in invocation.requests),0))]=None
    entries=[]
    for kind,shape in shapes:
        group=kind in ('expert_ffn','shared_expert_ffn')
        duration=5_000_000*shape.tokens if group else 2_000_000 if kind=='router' else 1_000_000
        for dev in invocation.devices:
            entries.append(dict(kind=kind,device=dev,phase=shape.phase,batch=shape.batch,tokens=shape.tokens,
                visible_history=shape.visible_history,duration_ns=duration if dev=='cpu' else duration//2,
                workspace_bytes=16,provenance='synthetic_layer',accounting='per_expert_group' if group else 'per_request'))
    sizes={w.storage_bytes for w in layer.weights}|{layer.kv_bytes_per_token}
    sizes.update(b.storage_bytes for s in invocation.state_in.requests for b in s.blocks)
    for r in invocation.requests:
        sizes.update({r.tokens*layer.hidden_bytes_per_token,4*layer.ffn_params['top_k']*r.tokens})
    sizes.update(count*layer.hidden_bytes_per_token for count in counts.values())
    if layer.ffn_params.get('shared_expert_weight'): sizes.add(sum(r.tokens for r in invocation.requests)*layer.hidden_bytes_per_token)
    transfers=[dict(route=route,storage_bytes=n,duration_ns=3_000_000 if route=='ssd_to_dram' else 100_000,
                    provenance='synthetic_transfer') for route in ('dram_to_vram','vram_to_dram','ssd_to_dram','dram_to_ssd') for n in sorted(sizes)]
    return dict(format_version='m2-layer-0.6',source='moe-synthetic-interface',entries=entries,transfer_entries=transfers)


def moe_compiled(*,devices=('cpu',),shared=False,phase='prefill',state=None,requests=None,mode='causal_replay'):
    layer=parse_layer_document(moe_layer_document(shared=shared))
    raw=moe_invocation_document(layer,devices=devices,phase=phase,requests=requests,mode=mode)
    if state is not None:
        for r in raw['requests']:
            r['position_start']=state.for_request(r['request_id']).next_position
        raw['route_trace']=[dict(request_id=r['request_id'],layer_id=layer.id,absolute_position=p,expert_ids=[0,1])
            for r in raw['requests'] for p in range(r['position_start'],r['position_start']+r['tokens'])]
        # Parse the declared zero-state then supply the exact prior snapshot;
        # validate_invocation checks it together with positions and the trace.
    invocation=parse_invocation_document(raw)
    if state is not None: invocation=dataclasses.replace(invocation,state_in=state)
    return compile_invocation(layer,invocation,parse_cost_table(moe_cost_document(layer,invocation)))


class MoEReplayTests(unittest.TestCase):
    def test_routed_experts_execute_once_with_exact_token_counts(self):
        c=moe_compiled(); result=run_invocation(c)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        experts=[t for t in c.tasks if t.kind=='expert_ffn']
        self.assertEqual([(t.expert_id,t.shape.tokens) for t in experts],[(0,1),(1,2),(2,1)])
        self.assertEqual(result.makespan_ns,34_000_000)
        self.assertEqual(result.state_out.requests[0].next_position,2)
        self.assertEqual(len(result.state_out.requests[0].blocks),2)
        reads=[a.target_id for a in result.evaluation.actions if a.kind==ACTION_READ_SSD]
        self.assertEqual(reads,[weight_id(c.layer.id,'E0'),weight_id(c.layer.id,'E1'),weight_id(c.layer.id,'E2')])
        self.assertNotIn(weight_id(c.layer.id,'E3'),reads)

    def test_two_decode_requests_share_one_group_and_weight_copy(self):
        requests=[dict(request_id=r,position_start=0,tokens=1,input_location='dram',output_location='dram') for r in ('A','B')]
        c=moe_compiled(phase='decode',requests=requests);result=run_invocation(c)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        task=next(t for t in c.tasks if t.expert_id==1)
        self.assertEqual(task.shape.tokens,2)
        self.assertEqual(set(task.work_items),{('A',0),('B',0)})
        self.assertEqual(sum(a.kind==ACTION_READ_SSD and a.target_id==weight_id(c.layer.id,'E1') for a in result.evaluation.actions),1)
        self.assertEqual(len(result.state_out.requests),2)

    def test_shared_expert_runs_on_all_tokens_outside_top_k(self):
        c=moe_compiled(shared=True);result=run_invocation(c)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        shared=next(t for t in c.tasks if t.kind=='shared_expert_ffn')
        self.assertEqual(shared.shape.tokens,2)
        self.assertEqual(set(c.expert_work_items['shared']),{('r0',0),('r0',1)})
        self.assertEqual(len([t for t in c.tasks if t.kind=='expert_ffn']),3)

    def test_router_gate_blocks_early_selective_ssd_read(self):
        c=moe_compiled();plan=list(plan_for(c)); idx=next(i for i,a in enumerate(plan) if a.kind==ACTION_READ_SSD)
        early=[plan[idx],Action(ACTION_ADVANCE),*plan[:idx],*plan[idx+2:]]
        result=run_invocation(c,actions=early)
        self.assertEqual(result.status,'invalid_mapping')
        self.assertEqual(result.evaluation.error_code,'TRANSFER_DEPENDENCY_NOT_READY')
        self.assertIsNone(result.state_out)

    def test_offline_oracle_can_prefetch_before_router_but_is_labeled(self):
        c=moe_compiled(mode='offline_oracle');plan=list(plan_for(c));idx=next(i for i,a in enumerate(plan) if a.kind==ACTION_READ_SSD)
        early=[plan[idx],Action(ACTION_ADVANCE),*plan[:idx],*plan[idx+2:]]
        result=run_invocation(c,actions=early)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        self.assertEqual(c.physical_boundary['replay_information_mode'],'offline_oracle')

    def test_prefill_decode_reuses_expert_dram_copy_and_versions(self):
        first=moe_compiled();prefill=run_invocation(first)
        state=state_in_from_state_out(prefill.state_out)
        second=moe_compiled(phase='decode',state=state);decode=run_invocation(second)
        self.assertEqual(decode.status,'valid',decode.evaluation.reason)
        self.assertFalse(any(a.kind==ACTION_READ_SSD and a.target_id.startswith('w:') for a in decode.evaluation.actions))
        self.assertEqual(decode.state_out.requests[0].next_position,3)
        self.assertEqual(len(decode.state_out.requests[0].blocks),3)
        by_id={w.id:w for w in decode.state_out.weight_copies}
        self.assertEqual(by_id['E0'].locations,('dram','ssd'))
        self.assertEqual(by_id['E3'].locations,('ssd',))
        self.assertEqual(by_id['E0'].version,1)
        doc=to_state_document(decode.state_out)
        self.assertEqual(doc['format_version'],'m2-state-0.2')
        self.assertEqual(parse_state_document(doc).as_state_in(),state_in_from_state_out(decode.state_out))

    def test_ragged_prefill_then_decode_keeps_request_histories_separate(self):
        requests=[dict(request_id='A',position_start=0,tokens=2,input_location='dram',output_location='dram'),
                  dict(request_id='B',position_start=0,tokens=1,input_location='dram',output_location='dram')]
        first=moe_compiled(requests=requests);prefill=run_invocation(first)
        self.assertEqual(prefill.status,'valid',prefill.evaluation.reason)
        histories={r.request_id:r for r in prefill.state_out.requests}
        self.assertEqual({k:r.next_position for k,r in histories.items()},{'A':2,'B':1})
        self.assertTrue(set(b.id for b in histories['A'].blocks).isdisjoint(b.id for b in histories['B'].blocks))
        requests=[dict(request_id=r,position_start=0,tokens=1,input_location='dram',output_location='dram') for r in ('A','B')]
        second=moe_compiled(phase='decode',state=state_in_from_state_out(prefill.state_out),requests=requests)
        decode=run_invocation(second)
        self.assertEqual(decode.status,'valid',decode.evaluation.reason)
        self.assertEqual({r.request_id:r.next_position for r in decode.state_out.requests},{'A':3,'B':2})
        self.assertFalse(any(a.kind==ACTION_READ_SSD and a.target_id.startswith('w:') for a in decode.evaluation.actions))
        self.assertEqual(set(second.expert_work_items['0']),{('A',2),('B',1)})

    def test_input_cache_snapshot_overrides_layer_initial_location(self):
        first=moe_compiled();prefill=run_invocation(first)
        state=state_in_from_state_out(prefill.state_out)
        state=dataclasses.replace(state,weight_copies=tuple(dataclasses.replace(w,version=7,
            locations=('dram',)) if w.id=='E0' else w for w in state.weight_copies))
        second=moe_compiled(phase='decode',state=state);decode=run_invocation(second)
        self.assertEqual(decode.status,'valid',decode.evaluation.reason)
        self.assertEqual(second.scenario.workload.tensor_by_id[weight_id(second.layer.id,'E0')].initial_locations,('dram',))
        cached=next(w for w in decode.state_out.weight_copies if w.id=='E0')
        self.assertEqual(cached.locations,('dram',))
        self.assertEqual(cached.version,7)
        self.assertFalse(any(a.kind==ACTION_READ_SSD and a.target_id==weight_id(second.layer.id,'E0') for a in decode.evaluation.actions))

    def test_missing_combine_expert_dependency_is_not_silently_accepted(self):
        c=moe_compiled();plan=list(plan_for(c))
        idx=next(i for i,a in enumerate(plan) if a.kind==ACTION_COMPUTE and a.target_id.endswith('expert.2'))
        del plan[idx:idx+2]
        result=run_invocation(c,actions=plan)
        self.assertEqual(result.status,'invalid_mapping')
        self.assertIsNone(result.state_out)

    def test_gpu_expert_copies_use_ssd_dram_then_vram(self):
        c=moe_compiled(devices=('cpu','gpu'))
        selected={t.operation_id:('gpu' if t.kind=='expert_ffn' else 'cpu') for t in c.tasks}
        result=run_invocation(c,placements=selected)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        actions=result.evaluation.actions
        for e in ('E0','E1','E2'):
            name=weight_id(c.layer.id,e)
            read=next(i for i,a in enumerate(actions) if a.kind==ACTION_READ_SSD and a.target_id==name)
            copy_index=next(i for i,a in enumerate(actions) if a.kind==ACTION_COPY and a.target_id==name and a.dst=='vram')
            self.assertLess(read,copy_index)

    def test_capacity_peak_and_peak_minus_one_are_distinguished(self):
        c=moe_compiled();result=run_invocation(c);peak=result.evaluation.peak_dram_bytes
        for capacity,status in ((peak,'valid'),(peak-1,'invalid_mapping')):
            inv=dataclasses.replace(c.invocation,hardware=dataclasses.replace(c.invocation.hardware,dram_capacity_bytes=capacity))
            limited=compile_invocation(c.layer,inv,c.cost_table)
            self.assertEqual(run_invocation(limited).status,status)

    def test_kv_can_be_written_to_ssd_and_restored_on_decode(self):
        c=moe_compiled();request=c.invocation.state_in.requests[0]
        history=dataclasses.replace(c.invocation.state_in,requests=(dataclasses.replace(request,state_location='ssd'),))
        inv=dataclasses.replace(c.invocation,state_in=history)
        first=compile_invocation(c.layer,inv,c.cost_table);result=run_invocation(first)
        self.assertEqual(result.status,'valid',result.evaluation.reason)
        # Keep only the explicitly delivered SSD copies to exercise restoration.
        state=state_in_from_state_out(result.state_out);r=state.requests[0]
        state=dataclasses.replace(state,requests=(dataclasses.replace(r,blocks=tuple(dataclasses.replace(b,locations=('ssd',)) for b in r.blocks)),))
        second=moe_compiled(phase='decode',state=state);decode=run_invocation(second)
        self.assertEqual(decode.status,'valid',decode.evaluation.reason)
        restored={a.target_id for a in decode.evaluation.actions if a.kind==ACTION_READ_SSD}
        self.assertTrue({b.id for b in r.blocks} <= restored)

if __name__=='__main__': unittest.main()
