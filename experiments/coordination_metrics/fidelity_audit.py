"""Independent finite translation-fidelity audit; no hardware measurement.

Expected work is handwritten workload algebra plus public strategy declarations.
Compiler graph/events are observed evidence, never the expected side.  A fresh
public-API MoE example separately exercises cache pressure and causal prefixes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'mapping',ROOT/'modeling'):
    sys.path.insert(0,str(p))
OUT=ROOT/'build/coordination-metrics-approved-20261030/fidelity'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


class Audit:
    def __init__(self,identity): self.id=identity;self.checked=Counter();self.violations=[];self.witnesses={}
    def require(self,category,condition,target,detail):
        self.checked[category]+=1
        if not condition:self.violations.append(dict(category=category,target=target,detail=detail))
    def result(self):
        counts=Counter(x['category'] for x in self.violations)
        return dict(id=self.id,checks=dict(self.checked),violation_counts=dict(counts),
            total_violations=len(self.violations),violations=self.violations,witnesses=self.witnesses)


def declared_devices(declaration,dag,choices=None):
    """Written from public choices, without the production placement resolver."""
    result={};intent=declaration.get('intent',{})
    for oid,op in dag['operations'].items():
        k=op['semantic_op'];ctx=dag['contexts'][oid]
        if k=='state_update':device='cpu'
        elif choices is not None:
            key=k if declaration['workload']['kind']=='moe' else f"r{ctx['r']}:{'prefill' if ctx['step']==0 else 'decode'}:{k}"
            device=choices[key]
        elif declaration['family']=='PDhandoff':device=intent['prefill_device'] if ctx['step']==0 else intent['decode_device']
        elif k.startswith('ffn_'):device='cpu' if '.host.' in oid else 'gpu'
        elif k in ('attention_qk','attention_softmax'):device=intent['attention_front_device']
        elif k=='projection':device=intent['projection_device']
        elif k=='attention':device=intent['attention_device']
        elif k=='expert_ffn':device=intent['expert_device']
        else:device='gpu'
        result[oid]=device
    return result


def audit_trace(identity,declaration,dag,graph,events,*,choices=None,preparation='demand'):
    """Check finite semantic obligations using independent expected operands."""
    a=Audit(identity);ops=dag['operations'];tensors=dag['tensors'];devices=declared_devices(declaration,dag,choices)
    observed_ops={x['id']:x for x in graph['workload']['operations']}
    observed_tensors={x['id']:x for x in graph['workload']['tensors']}
    a.require('operation_inventory',set(ops)==set(observed_ops),identity,'exact expected operation IDs')
    a.require('tensor_inventory',set(tensors)==set(observed_tensors),identity,'exact expected tensor IDs')
    for tid,t in tensors.items():
        seen=observed_tensors.get(tid,{})
        a.require('tensor_inventory',all(seen.get(k)==t[k] for k in ('ne','dtype','storage_bytes')),tid,'independent shape/type/physical bytes')
    for oid,o in ops.items():
        seen=observed_ops.get(oid,{})
        a.require('required_dependencies',all(seen.get(k,[] if k=='extra_outputs' else None)==o[k] for k in ('inputs','output','extra_outputs','semantic_op')),oid,'exact ordered operands/outputs and semantic kind')
        if o['semantic_op'].startswith('ffn_'):
            a.require('partition',seen.get('op_params')==o['op_params'],oid,'declared FFN slice bounds')
    computes=Counter(e['target_id'] for e in events if e['kind']=='COMPUTE')
    for oid in ops:a.require('execution_multiplicity',computes[oid]==1,oid,'exactly one execution of declared operation')
    a.require('execution_multiplicity',set(computes)<=set(ops),identity,'no undeclared computations')
    producers={t:oid for oid,o in ops.items() for t in [o['output'],*o['extra_outputs']]}
    users=defaultdict(list)
    for oid,o in ops.items():
        for tid in o['inputs']:users[tid].append(oid)
    allocated={(tid,loc) for tid,t in tensors.items() for loc in t['initial_locations']}
    ready=set(allocated);running=[];completed=set();early=[];peak_weights=0;weight_evictions=[]
    now=0
    def finish(t):
        nonlocal running,now
        a.require('event_order',t>=now,identity,'monotonic issue time')
        keep=[]
        for task in running:
            if task['end']<=t:
                ready.update(task['outputs'])
                if task['op']:completed.add(task['op'])
            else:keep.append(task)
        running=keep;now=t
    for index,e in enumerate(events):
        finish(e['t_ns']);kind=e['kind'];tid=e.get('target_id')
        if kind=='ADVANCE':continue
        if kind=='EVICT':
            key=(tid,e['location'])
            a.require('release_safety',key in ready,tid,'release requires ready physical allocation')
            a.require('release_safety',all(key not in t['inputs'] for t in running),tid,'active readers finish before release')
            ready.discard(key);allocated.discard(key)
            if tid in tensors and tensors[tid]['role']=='weight':weight_evictions.append(dict(tensor=tid,t_ns=e['t_ns']))
            continue
        if kind not in ('COPY','COMPUTE'):
            a.require('action_inventory',False,tid,'unsupported action in separate-pool audit');continue
        if kind=='COPY':
            a.require('copy_guards',tid in tensors,tid,'COPY refers to declared tensor')
            if tid not in tensors:continue
            inputs=[(tid,e['src'])];outputs=[(tid,e['dst'])];operation=None
            a.require('copy_payload',e['bytes_transferred']==tensors[tid]['storage_bytes'],tid,'COPY transfers declared full backing size')
            a.require('copy_guards',inputs[0] in ready,tid,'COPY source complete at declared endpoint')
            if tensors[tid]['role']=='weight':
                candidates=[oid for oid in users[tid] if ('dram' if devices[oid]=='cpu' else 'vram')==e['dst']]
                current=[oid for oid in candidates if all(producers.get(x) in completed or x not in producers for x in ops[oid]['inputs'])]
                if preparation=='demand':a.require('preparation',bool(current),tid,'demand weight load requires a consumer whose producers have completed')
                elif not current:early.append(dict(tensor=tid,t_ns=e['t_ns']))
                if ':E' in tid:
                    a.require('causal_route',bool(current),tid,'expert weight demand requires current router completion')
        else:
            operation=tid
            if tid not in ops:continue
            a.require('placement',e['device']==devices[tid],tid,'executing device equals public strategy')
            location='dram' if e['device']=='cpu' else 'vram';o=ops[tid]
            inputs=[(t,location) for t in o['inputs']];outputs=[(t,location) for t in [o['output'],*o['extra_outputs']]]
            for p in o['inputs']:
                if p in producers:a.require('required_dependencies',producers[p] in completed,tid,'logical producer completes before consumer')
            if o['semantic_op']=='expert_ffn':
                router=[producers[t] for t in o['inputs'] if t in producers and ops[producers[t]]['semantic_op']=='router']
                a.require('causal_route',bool(router) and all(t in completed for t in router),tid,'expert compute after its revealed route')
            if o['semantic_op']=='state_update' and not o['inputs']:
                a.require('request_arrival',e['start_ns']>=2000*dag['contexts'][tid]['r'],tid,'request input appears after declared arrival')
            if o['semantic_op']=='state_update' and declaration['workload'].get('handoff') and dag['contexts'][tid]['step']==1:
                for kv in [x for x in o['inputs'] if x.startswith('kv:')]:
                    a.require('pd_handoff',(kv,'dram') in ready,kv,'all declared Prefill KV delivered to DRAM before first Decode gate')
        for key in inputs:a.require('copy_guards',key in ready,str(key),'all operands ready at the executing endpoint')
        for key in outputs:
            a.require('copy_guards',key not in allocated,str(key),'no overwrite of live or pending destination')
            allocated.add(key)
        running.append(dict(end=e['end_ns'],inputs=inputs,outputs=outputs,op=operation))
        weight_bytes=sum(tensors[t]['storage_bytes'] for t,loc in allocated if loc=='vram' and tensors[t]['role']=='weight')
        peak_weights=max(peak_weights,weight_bytes)
        budget=declaration.get('hardware',{}).get('cache_budget_bytes')
        if budget is not None:a.require('cache_capacity',weight_bytes<=budget,identity,'including pending GPU weight reservations')
    finish(max((e['end_ns'] for e in events),default=0))
    for tid in dag['terminal_hidden']:a.require('terminal_delivery',(tid,'dram') in ready,tid,'public request output is delivered to DRAM')
    a.witnesses=dict(early_static_preparations=early,peak_gpu_weight_bytes=peak_weights,
        weight_evictions=weight_evictions,
        lru_victim_validation='not_exercised: existing workloads have no declared cache-conflict witness',
        request_priority_validation='not_exercised: no policy-choice collision witness is asserted')
    return a.result()


def audit_lru_sequence(actions,routes,*,capacity=2):
    """Independent sequential LRU spec. No production cache code is called."""
    a=Audit('fresh-lru-victim');resident=set();last={};step=0;clock=0;executed=set();victims=[]
    for index,e in enumerate(actions):
        k=e['kind'];expert=e.get('expert')
        if k=='LOAD':
            a.require('causal_route',expert==routes[step],str(index),'demand LOAD is for current revealed expert')
            a.require('cache_capacity',expert not in resident and len(resident)<capacity,str(index),'vacant slot before LOAD')
            resident.add(expert)
        elif k=='RELEASE':
            expected=min(resident,key=lambda x:(last.get(x,-1),x)) if resident else None
            a.require('lru_victim',expert==expected,str(index),f'independent least recent completed use requires expert {expected}')
            a.require('lru_victim',len(resident)==capacity and routes[step] not in resident,str(index),'eviction required by current miss at capacity')
            victims.append(dict(step=step,expected=expected,observed=expert));resident.discard(expert)
        elif k=='EXPERT':
            a.require('placement',e.get('device')=='gpu',str(index),'fixed GPU expert execution')
            a.require('causal_route',e.get('step')==step and expert==routes[step],str(index),'expert equals current revealed route')
            a.require('execution_multiplicity',step not in executed,str(index),'one selected expert execution per call')
            a.require('copy_guards',expert in resident,str(index),'GPU expert weight resident')
            executed.add(step);clock+=1;last[expert]=clock
        elif k=='COMBINE':
            a.require('required_dependencies',step in executed and e['step']==step,str(index),'combine after selected expert')
            step+=1
    a.require('execution_multiplicity',step==len(routes),str(step),'all declared routed calls complete')
    a.witnesses['victims']=victims
    return a.result()


def audit_public_records(records,sequence,events,identity):
    """Route history is observed, not trusted: each row needs its router end."""
    a=Audit(identity)
    router_ends={e['target_id']:e['end_ns'] for e in events if e['kind']=='COMPUTE' and e['target_id'].endswith(':router')}
    for index,rec in enumerate(records):
        obs=rec['observation'];known={}
        for route in obs['route_history']:
            p=route['absolute_position'];label='prefill' if p==0 else 'decode'+str(p-1)
            router=f'g:request0-{label}-layer0:op:layer0:request0:router'
            legal=(route['request_id']=='request0' and route['layer_id']=='layer0' and 0<=p<len(sequence)
                and tuple(route['expert_ids'])==(sequence[p],) and router in router_ends and router_ends[router]<=obs['t_ns'])
            a.require('public_route_revelation',legal,str(index),'only declared route rows with independently located completed routers may be public')
            known[route['request_id'],route['layer_id'],p]=route['expert_ids']
        for op in obs['operations']:
            if op['kind']!='expert_ffn':continue
            for request,position in op['work_items']:
                a.require('public_visibility',op['expert_id'] in known.get((request,op['layer_id'],position),()),op['id'],'expert exposed only after matching public router revelation')
    return a.result()


def fresh_lru(out):
    """Public high-level Simulator first; scoped public Scheduling bridge also checked."""
    from llm_simulator import Cache,Device,Hardware,Model,Request,RequestFlow,Route,Scheduling,Simulator
    from experiments.dac_validation_e1 import WitnessCosts
    from profiling.moe_policy_validation.bridge import compile_expert_program
    rows=[];prefixes=[];bridge_outputs=[]
    for label,sequence in [('base',[0,1,0,2]),('changed-future',[0,1,0,1])]:
        folder=out/'fresh'/label
        declaration=dict(id=label,hidden=8,intermediate=16,expert_weight_bytes=1536,
            weight_budget_bytes=3072,routes=sequence,placement='all nonexpert CPU; all expert GPU',
            cache='LRU; touch at completed use; victim expert1 on miss2 after 0,1,0',
            seed=20261030,hardware_measurement=False)
        save(folder/'DECLARATION.json',declaration)
        flow=RequestFlow(requests=(Request('request0',1,3),),
            routes=tuple(Route('request0','layer0',p,(x,)) for p,x in enumerate(sequence)),route_source='private declared sequence')
        simulator=Simulator(model=Model.moe(layers=1,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1,experts=3,top_k=1),
            hardware=Hardware(Device(1e9,1e9),Device(2e9,2e9),1<<20,1<<20,1e9,1e9,1e9),workload=flow,cost_model=WitnessCosts())
        def configure(plan):
            for kind in ('projection','attention','router','combine'):plan.place(kind,'cpu')
            plan.place('expert_ffn','gpu')
        scheduling=Scheduling(default_device='cpu',release='last_use',cache=Cache(rule='lru',budget_bytes=3072,unit='expert_group'),expert_rule='fixed')
        with simulator.compile(directory=folder/'inputs',configure=configure,scheduling=scheduling) as compiled:
            save(folder/'compiled-ir.json',compiled.ir_document());result=compiled.run()
            events=[asdict(e) for e in result.events];save(folder/'events.json',events)
            save(folder/'summary.json',result.summary());records=[dict(observation=asdict(r.observation),decision=asdict(r.decision) if r.decision else None) for r in result.raw[0].decisions]
            save(folder/'public-records.json',records)
            a=Audit('public-full-layer-'+label);a.require('completion',result.success,label,'public full-layer compiler completes')
            routes_seen=[];compact=[];call=0
            for e in events:
                tid=e.get('target_id','')
                if e['kind'] in ('COPY','EVICT') and tid.startswith('w:layer0:E') and (e.get('dst')=='vram' or e.get('location')=='vram'):
                    compact.append(dict(kind='LOAD' if e['kind']=='COPY' else 'RELEASE',expert=int(tid.split(':E')[-1])))
                if e['kind']=='COMPUTE' and ':__moe_batch__:expert.' in tid:
                    compact.append(dict(kind='EXPERT',step=call,expert=int(tid.rsplit('.',1)[-1]),device=e['device']))
                if e['kind']=='COMPUTE' and tid.endswith(':combine'):
                    compact.append(dict(kind='COMBINE',step=call));call+=1
            lr=audit_lru_sequence(compact,sequence);a.checked.update(lr['checks']);a.violations+=lr['violations'];a.witnesses.update(lr['witnesses'])
            visibility=audit_public_records(records,sequence,events,label)
            a.checked.update(visibility['checks']);a.violations+=visibility['violations']
            prefix=[]
            for rec in records:
                if any(r['absolute_position']>=3 for r in rec['observation']['route_history']):break
                prefix.append(rec)
            prefixes.append(prefix);a.witnesses['public_prefix_length']=len(prefix)
            save(folder/'compact-expert-actions.json',compact);save(folder/'AUDIT.json',a.result());rows.append(a.result())
        # Existing common policy compiler under its honest serial projection.
        config=dict(hidden=8,intermediate=16,experts=3,gpu_slots=2,threads=1,cuda_device=0)
        price=dict(schema='moe-expert-policy-calibration-1',config=config,boundary='serial_host_expert_response_v1',prices_ns={
            'LOAD':20,'RELEASE':1,'EXPERT:cpu:q=1':100,'EXPERT:gpu:q=1':20,'COMBINE:q=1:topk=1':5})
        bridge=compile_expert_program(config=config,steps=[dict(input=[[.1]*8],routes=[[e]],scores=[[1.]]) for e in sequence],
            seed=20261030,recipe='gpu_demand_lru',prices=price,resource_guards=dict(host_reserve_bytes=0,gpu_reserve_bytes=0))
        save(folder/'scoped-bridge.json',bridge);br=audit_lru_sequence(bridge['program']['actions'],sequence);br['id']='serial-bridge-'+label
        save(folder/'bridge-AUDIT.json',br);rows.append(br);bridge_outputs.append(bridge)
    causal=Audit('future-prefix-pair');causal.require('causal_prefix',bool(prefixes[0]) and prefixes[0]==prefixes[1],'full-layer','private suffix change preserves complete public observations and decisions before revelation')
    bprefix=[[r for r in b['compilation_evidence']['records'] if r['step']<3] for b in bridge_outputs]
    causal.require('causal_prefix',bprefix[0]==bprefix[1],'serial-bridge','private suffix preserves serial policy prefix')
    rows.append(causal.result());save(out/'fresh/CAUSAL_PAIR.json',causal.result())
    return rows,bridge_outputs[0]


def mutation_controls(source,dag,graph,events,bridge,public_records,public_events):
    """Deliberately wrong observed evidence; the independent expectation is fixed."""
    results=[]
    def trace_mutation(name,mutate,category):
        ge=deepcopy(graph);ev=deepcopy(events);mutate(ge,ev)
        result=audit_trace(name,source,dag,ge,ev,preparation=source['intent']['preparation'])
        results.append(dict(id=name,expected_detector=category,caught=result['violation_counts'].get(category,0)>0,audit=result))
    trace_mutation('wrong-placement',lambda g,e:next(x for x in e if x['kind']=='COMPUTE').update(device='gpu'),'placement')
    trace_mutation('duplicate-compute',lambda g,e:e.append(deepcopy(next(x for x in e if x['kind']=='COMPUTE'))),'execution_multiplicity')
    trace_mutation('missing-compute',lambda g,e:e.remove(next(x for x in e if x['kind']=='COMPUTE')),'execution_multiplicity')
    trace_mutation('wrong-copy-bytes',lambda g,e:next(x for x in e if x['kind']=='COPY').update(bytes_transferred=1),'copy_payload')
    trace_mutation('missing-required-input',lambda g,e:next(x for x in g['workload']['operations'] if x['inputs'])['inputs'].pop(),'required_dependencies')
    for name,field,value,category in [('wrong-lru-victim','expert',0,'lru_victim'),('illegal-future-load','expert',2,'causal_route')]:
        acts=deepcopy(bridge['program']['actions'])
        next(e for e in acts if e['kind']==('RELEASE' if name=='wrong-lru-victim' else 'LOAD'))[field]=value
        result=audit_lru_sequence(acts,[0,1,0,2]);results.append(dict(id=name,expected_detector=category,caught=result['violation_counts'].get(category,0)>0,audit=result))
    records=deepcopy(public_records)
    future=deepcopy(next(route for rec in records for route in rec['observation']['route_history'] if route['absolute_position']==3))
    records[0]['observation']['route_history']=[future]
    result=audit_public_records(records,[0,1,0,2],public_events,'illegal-future-revelation')
    results.append(dict(id='illegal-future-revelation',expected_detector='public_route_revelation',
        caught=result['violation_counts'].get('public_route_revelation',0)>0,audit=result))
    return results


def main(out=OUT):
    from experiments.dac_validation_e1_oracle import declared_dag
    from experiments.ir_value import campaign as ir
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    source_hashes={str(p):sha(p) for p in [Path(__file__),Path(ir.__file__),ROOT/'experiments/dac_validation_e1_oracle.py',
        ROOT/'experiments/dac_validation_e1.py',ROOT/'profiling/moe_policy_validation/bridge.py']}
    core_hashes={str(p):sha(p) for folder in ('src/llm_simulator','mapping/tensor_mapping','modeling/llm_infer_model') for p in (ROOT/folder).rglob('*.py')}
    save(out/'PROTOCOL.json',dict(schema='strategy-translation-fidelity-1',created=datetime.now(timezone.utc).isoformat(),
        expected='Independent handwritten DAG plus public high-level placement/slice/preparation/handoff declarations; never compiler output',
        reuse='48 paired variants and 800 admitted frozen analytical candidates; actual new checking, no new hardware timings',
        rules=['exact declared operations and tensors; exactly one compute','placement from public choices','exact ordered operands and FFN slice bounds',
            'producer completion and endpoint readiness','source copy readiness; declared backing bytes; no premature read/release',
            'terminal request output in DRAM; all Prefill KV in DRAM at first Decode gate when handoff declared',
            'demand only with a current consumer whose input producers finished; spare permits early preparation but does not require a speedup',
            'fresh LRU capacity2, route0/1/0/2 requires eviction1; suffix0/1/0/1 cannot affect prior public prefix'],
        nonclaims=['no universal expression completeness','no performance prediction accuracy','untriggered cache and request-priority options not counted as strong witnesses'],
        source_hashes=source_hashes,core_source_hashes=core_hashes,hardware_measurement=False))
    rows=[];physical=ROOT/'build/coordination-metrics-20261030/physical';protocol=read(physical/'physical_PROTOCOL.json')
    inputs={str(physical/'physical_PROTOCOL.json'):sha(physical/'physical_PROTOCOL.json')}
    mutation_source=None
    for c in protocol['conditions']:
        folder=physical/'runs'/c['id'];paths=[folder/'execution-graph.json',folder/'events.json']
        graph,events=map(read,paths);dag=declared_dag(c['workload'])
        inputs.update({str(p):sha(p) for p in paths});result=audit_trace(c['id'],c,dag,graph,events,preparation=c['intent']['preparation'])
        result['cohort']='paired48';rows.append(result);save(out/'cases'/(c['id']+'.json'),result)
        if c['id']=='D01-development-A':mutation_source=(c,dag,graph,events)
    reportpath=ROOT/'build/ir-value-20261029-final-evidence/REPORT.json';old=read(reportpath);inputs[str(reportpath)]=sha(reportpath)
    descriptors={s[0]:s for s in ir.SCENARIOS}
    for row in old['rows']:
        if row['status']!='pass':continue
        s=ir.scenario(*descriptors[row['scenario']],row['regime']);dag=declared_dag(s)
        folder=Path(row['artifact_directory']);paths=[folder/'graph.json',folder/'events.json'];graph,events=map(read,paths)
        inputs.update({str(p):sha(p) for p in paths});declaration=dict(workload=s)
        result=audit_trace(row['id'],declaration,dag,graph,events,choices=row['choices'],preparation=row['storage'])
        result['cohort']='frozen800';rows.append(result);save(out/'cases'/(row['id']+'.json'),result)
    fresh,bridge=fresh_lru(out)
    for row in fresh:row['cohort']='fresh';rows.append(row)
    mutations=mutation_controls(*mutation_source,bridge,read(out/'fresh/base/public-records.json'),read(out/'fresh/base/events.json'))
    save(out/'MUTATION_CONTROLS.json',mutations)
    save(out/'RAW_INPUT_HASHES.json',inputs)
    totals=Counter();violations=Counter()
    for row in rows:totals.update(row['checks']);violations.update(row['violation_counts'])
    result=dict(schema='strategy-fidelity-summary-1',cases=len(rows),cohort_counts=dict(Counter(r['cohort'] for r in rows)),
        checks=dict(totals),violations=dict(violations),total_violations=sum(violations.values()),
        cases_with_violations=[r['id'] for r in rows if r['total_violations']],
        category_case_counts={k:dict(evaluated=sum(k in r['checks'] for r in rows),violated=sum(r['violation_counts'].get(k,0)>0 for r in rows)) for k in totals},
        rows=rows,mutation_controls_count=len(mutations),mutation_controls_caught=sum(r['caught'] for r in mutations),
        input_hashes_unchanged=all(sha(p)==h for p,h in inputs.items()),
        source_hashes_unchanged=all(sha(p)==h for p,h in {**source_hashes,**core_hashes}.items()),
        scope='Reexecuted independent finite contract checks on retained traces plus fresh public API/compiler LRU and causality microcases; no native hardware results')
    save(out/'SUMMARY.json',result)
    with (out/'COUNTS.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','cohort','category','checks','violations']);writer.writeheader()
        for r in rows:
            for k,n in r['checks'].items():writer.writerow(dict(case=r['id'],cohort=r['cohort'],category=k,checks=n,violations=r['violation_counts'].get(k,0)))
    save(out/'OUTPUT_HASHES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();main(args.out)
