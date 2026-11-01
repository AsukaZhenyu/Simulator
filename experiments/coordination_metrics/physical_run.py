"""Run frozen paired edits through public compiler, then independent audit."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
for folder in (ROOT,ROOT/'src',ROOT/'mapping',ROOT/'modeling'):sys.path.insert(0,str(folder))
from experiments.coordination_metrics.physical_protocol import OUT,write
from experiments.coordination_metrics.physical_ledger import replay
from experiments import dac_validation_e1 as e1
from experiments.dac_validation_e1_oracle import declared_dag,check_graph
from experiments.dac_validation_e1_numeric import check_numerical,NUMERICAL_CONTRACT
from experiments.serving_acceptance.oracles import expert_visibility
from llm_simulator import Cache,DataPreparation,Device,Hardware,Model,PhaseHandoff,PhasePlacement,Request,RequestFlow,RequestPipeline,Scheduling,Simulator

FIELDS=('h2d_bytes','d2h_bytes','peak_dram_bytes','peak_vram_bytes')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def snapshot():
    files=[]
    for source in ('src/llm_simulator','mapping/tensor_mapping','modeling/llm_infer_model'):
        files.extend((ROOT/source).rglob('*.py'))
    files.extend((ROOT/'experiments/coordination_metrics').glob('physical_*.py'))
    files.extend((ROOT/'experiments').glob('dac_validation_e1*.py'))
    files.append(ROOT/'experiments/serving_acceptance/oracles.py')
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def device_for(row,expected,oid):
    i=row['intent'];op=expected['operations'][oid];kind=op['semantic_op']
    if kind=='state_update':return 'cpu'
    if row['family']=='PDhandoff':return i['prefill_device'] if expected['contexts'][oid]['step']==0 else i['decode_device']
    if kind.startswith('ffn_'):
        return 'cpu' if '.host.' in oid else 'gpu'
    if kind in {'attention_qk','attention_softmax'}:return i['attention_front_device']
    if kind=='projection':return i['projection_device']
    if kind=='attention':return i['attention_device']
    if kind=='expert_ffn':return i['expert_device']
    return 'gpu'

def build(row,expected,folder):
    s=row['workload'];i=row['intent'];hw=row['hardware']
    hardware=Hardware(Device(1e9,1e9),Device(2e9,2e9),hw['dram_capacity_bytes'],hw['vram_capacity_bytes'],1e9,1e9,1e9)
    parameters=dict(layers=s['layers'],hidden_size=s['hidden'],intermediate_size=s['intermediate'],heads=2,kv_heads=1)
    model=Model.moe(experts=3,top_k=1,**parameters) if s['kind']=='moe' else Model.dense(**parameters)
    workload=RequestFlow(requests=tuple(Request(f'request{r}',s['q'],s['decode_steps'],r*2000) for r in range(s['requests'])),
        routes=e1.routes_for(s),route_source='private immutable model trace; only current router publishes demand' if s['kind']=='moe' else None)
    costs=e1.WitnessCosts();object.__setattr__(costs,'workspace_bytes',16)
    simulator=Simulator(model=model,hardware=hardware,workload=workload,cost_model=costs)
    def configure(plan):
        if s['requests']>1:plan.pipeline(RequestPipeline(ready_rule=i['pipeline']))
        if row['family']=='PDhandoff':
            kinds=('projection','attention','dense_ffn')
            PhasePlacement(prefill={k:i['prefill_device'] for k in kinds},decode={k:i['decode_device'] for k in kinds}).apply(plan)
        else:
            plan.place('projection',i['projection_device'])
            if not s['attention_refined']:plan.place('attention',i['attention_device'])
            if s['kind']=='moe':plan.place('expert_ffn',i['expert_device'])
        if s['handoff']:plan.handoff(PhaseHandoff(destination='dram'))
        for r in range(s['requests']):
            for step in range(s['decode_steps']+1):
                label='prefill' if step==0 else f'decode{step-1}'
                for l in range(s['layers']):
                    invocation=f'request{r}-{label}-layer{l}'
                    if s['attention_refined']:plan.transform(invocation,e1.attention_spec(s))
                    if s['ffn_refined']:plan.transform(invocation,e1.ffn_spec(s,shared=i['ffn_storage']=='shared_parent_slice_views'))
        if s['attention_refined']:
            for kind in ('attention_qk','attention_softmax'):plan.place(kind,i['attention_front_device'])
        if i['preparation']=='spare':
            weights=sorted({tid for oid,op in expected['operations'].items() if device_for(row,expected,oid)=='gpu'
                for tid in op['inputs'] if expected['tensors'][tid]['role']=='weight' and ':E' not in tid})
            if weights:plan.prepare(DataPreparation(tensor_ids=tuple(weights),trigger='spare'))
    cache=Cache(rule='lru',budget_bytes=hw['cache_budget_bytes']) if i['cache_rule']=='lru' else Cache()
    return simulator.compile(directory=folder/'inputs',configure=configure,scheduling=Scheduling(default_device='gpu',
        release='last_use',cache=cache,expert_rule='fixed',expert_order='graph_order'))

def capacity_reference(row,expected,*,negative=False):
    total=sum(t['storage_bytes'] for t in expected['tensors'].values())
    if negative:
        witnesses=[dict(operation=oid,tensor=t,bytes=expected['tensors'][t]['storage_bytes'])
            for oid,op in expected['operations'].items() if device_for(row,expected,oid)=='gpu'
            for t in [*op['inputs'],op['output']] if expected['tensors'][t]['storage_bytes']>1]
        if not witnesses:raise AssertionError('no mandatory GPU operand/output lower bound for impossible control')
        return dict(feasible=False,proof='mandatory GPU physical operand/output exceeds 1 byte',witness=witnesses[0])
    bound=total+16*len(expected['operations'])
    if bound>=min(row['hardware']['dram_capacity_bytes'],row['hardware']['vram_capacity_bytes']):raise AssertionError('declared conservative feasible witness does not fit')
    return dict(feasible=True,proof='Serial legal schedule with every declared allocation retained fits either pool; weights fit cache when enabled',
        all_declared_tensor_bytes=total,conservative_each_pool_upper_bound=bound)

def execute(row,folder,*,negative=False):
    folder.mkdir(parents=True,exist_ok=True);write(folder/'intent.json',row)
    expected=declared_dag(row['workload']);write(folder/'independent-workload.json',expected)
    reference=capacity_reference(row,expected,negative=negative)
    outcome=dict(id=row['id'],pair=row['pair'],family=row['family'],split=row['split'],variant=row['variant'],
        negative=negative,capacity_reference=reference,declared_operations=len(expected['operations']),declared_tensors=len(expected['tensors']))
    started=time.perf_counter_ns();compiled=None
    try:
        compiled=build(row,expected,folder);outcome['compile_wall_ns']=time.perf_counter_ns()-started
        write(folder/'compiled-ir.json',compiled.ir_document())
        started=time.perf_counter_ns();result=compiled.run();outcome['evaluate_wall_ns']=time.perf_counter_ns()-started
        summary=result.summary();write(folder/'production-summary.json',summary)
        outcome.update(accepted=result.success,production_status=summary)
        events=[asdict(e) for e in result.events];graph=result.raw[1].execution_document()
        write(folder/'events.json',events);write(folder/'execution-graph.json',graph)
        if negative:
            outcome['status']='false_accept' if result.success else 'rejected'
            outcome['rejection_kind']='capacity' if 'capacity' in str(summary).lower() or 'memory' in str(summary).lower() else 'runtime_unclassified'
            return outcome
        if not result.success:
            outcome['status']='false_reject_or_execution_failure';return outcome
        outcome['graph']=check_graph(row['workload'],expected,graph,events)
        for e in events:
            if e['kind']=='COMPUTE' and e['device']!=device_for(row,expected,e['target_id']):
                raise AssertionError('actual device differs from high-level policy '+e['target_id'])
        ledger=replay(row,expected,events,result.makespan_ns);write(folder/'independent-ledger.json',ledger)
        outcome['independent']={k:ledger[k] for k in FIELDS}
        outcome['observed']={k:summary[k] for k in FIELDS}
        outcome['absolute_byte_error']={k:outcome['observed'][k]-outcome['independent'][k] for k in FIELDS}
        outcome['event_counts']=ledger['event_counts']
        outcome['makespan_ns']=result.makespan_ns
        outcome['action_signature']=digest([{k:v for k,v in e.items() if k not in {'start_ns','end_ns','t_ns'}} for e in events])
        numerical=check_numerical(row['workload'],expected,graph,events)
        write(folder/'numerical-outputs.json',numerical.pop('outputs'));outcome['numerical']=numerical
        if row['workload']['kind']=='moe':outcome['causal_checks']=expert_visibility(result)
        outcome['status']='pass' if all(v==0 for v in outcome['absolute_byte_error'].values()) else 'physical_mismatch'
        return outcome
    except Exception as exc:
        outcome.update(status='unsupported' if type(exc).__name__=='Unsupported' else 'error',accepted=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if compiled is None:outcome['compile_wall_ns']=time.perf_counter_ns()-started
        return outcome
    finally:
        if compiled is not None:compiled.close()

def campaign(out=OUT,only=None):
    out=Path(out);protocol=json.loads((out/'physical_PROTOCOL.json').read_text(encoding='utf-8'))
    before=snapshot();write(out/'source-before.json',before)
    rows=[]
    for condition in protocol['conditions']:
        if only and condition['id'] not in only:continue
        row=execute(condition,out/'runs'/condition['id']);rows.append(row);write(out/'runs'/condition['id']/'result.json',row)
        print(condition['id'],row['status'],flush=True)
    if only:return
    negatives=[]
    for condition in protocol['conditions']:
        if condition['split']!='development' or condition['variant']!='B':continue
        case=deepcopy(condition);case['hardware']['vram_capacity_bytes']=1;case['id']+='-vram1'
        row=execute(case,out/'negative-controls'/case['id'],negative=True);negatives.append(row)
        write(out/'negative-controls'/case['id']/'result.json',row);print(case['id'],row['status'],flush=True)
    contrasts=[]
    for split in ('development','holdout'):
        for pair in sorted({r['pair'] for r in rows}):
            a=next(r for r in rows if r['pair']==pair and r['split']==split and r['variant']=='A')
            b=next(r for r in rows if r['pair']==pair and r['split']==split and r['variant']=='B')
            result=dict(pair=pair,split=split,family=a['family'],A=a['id'],B=b['id'])
            if all('independent' in r and 'observed' in r for r in (a,b)):
                result.update(reference_delta={k:b['independent'][k]-a['independent'][k] for k in FIELDS},
                    observed_delta={k:b['observed'][k]-a['observed'][k] for k in FIELDS})
                result['delta_error_bytes']={k:result['observed_delta'][k]-result['reference_delta'][k] for k in FIELDS}
                result['status']='pass' if all(v==0 for v in result['delta_error_bytes'].values()) else 'mismatch'
            else:result['status']='not_evaluated'
            contrasts.append(result)
    after=snapshot();write(out/'source-after.json',after)
    report=dict(schema='coordination-physical-results-1',protocol_sha256=hashlib.sha256((out/'physical_PROTOCOL.json').read_bytes()).hexdigest(),
        source_integrity=before==after,conditions=rows,contrasts=contrasts,negative_controls=negatives,
        numerical_contract=NUMERICAL_CONTRACT,
        counts=dict(nominal_conditions=len(rows),passing_conditions=sum(r['status']=='pass' for r in rows),
            paired_contrasts=len(contrasts),passing_contrasts=sum(r['status']=='pass' for r in contrasts),
            negative_controls=len(negatives),false_accepts=sum(r.get('accepted',False) for r in negatives),
            feasible_conditions_not_completed=sum(not r.get('accepted',False) for r in rows),
            unsupported=sum(r['status']=='unsupported' for r in rows),
            unique_observed_action_signatures=len({r['action_signature'] for r in rows if 'action_signature' in r})),
        interpretation='Finite paired edit preservation and independently replayed physical accounting, not hardware timing accuracy, strategy optimality, universal expressiveness or novelty proof. Timing is single-run implementation cost on this host.')
    write(out/'physical_RESULTS.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args();campaign(only=args.only)
