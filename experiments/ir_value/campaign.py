"""Quantify strategy composition and E/S choices, not Python line count.

Analytical experiment. No hardware, profiling fit or production edits.
The independent reference DAG originates in workload algebra, not compilation.
The reference scheduler is an explicitly bounded priority family; its best cost
is not called a global optimum or a proof of abstraction completeness.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'src',ROOT/'mapping',ROOT/'modeling'):
    sys.path.insert(0,str(path))
from llm_simulator import (Cache,DataPreparation,Device,Hardware,Model,Request,
    RequestFlow,RequestPipeline,PhaseHandoff,Scheduling,Select,Simulator)
from experiments import dac_validation_e1 as e1
from experiments.dac_validation_e1_oracle import declared_dag,check_graph,replay_physical
from experiments.dac_validation_e1_numeric import check_numerical

PRIORITIES=tuple(itertools.product(('id','reverse_id','short','long'),repeat=2))
BYTE_NS={'compute':1,'transfer_constrained':50}
SCENARIOS=(
    ('dense','A01',False,0,1),
    ('attention','A02',True,0,1),
    ('pipeline','A05',False,0,2),
    ('pd','A10',False,1,1),
    ('moe','A06',False,0,1),
)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scenario(name,pattern,attention,decode,requests,regime):
    s=e1.spec(pattern,8,1)
    s.update(id=f'{name}-{regime}',layers=1,decode_steps=decode,requests=requests,
        attention_refined=attention,ffn_refined=False,prefetch=False,cache=True,
        handoff=name=='pd',regime=regime,physical_capacity_bytes=1048576,
        cache_budget_bytes=262144)
    return s

def group(s,ctx,op):
    if s['kind']=='moe':return op['semantic_op']
    phase='prefill' if ctx['step']==0 else 'decode'
    return f"r{ctx['r']}:{phase}:{op['semantic_op']}"

def groups_for(s,d):
    return sorted({group(s,d['contexts'][oid],op) for oid,op in d['operations'].items()
                   if op['semantic_op']!='state_update'})

def placement(s,d,choices):
    return {oid:('cpu' if op['semantic_op']=='state_update' else choices[group(s,d['contexts'][oid],op)])
            for oid,op in d['operations'].items()}

def op_time(s,ctx,kind,device):
    # RequestFlow declares one hidden-vector write plus oneI32 position/layer
    # for either Dense or MoE, even though opaqueMoE numerical ports do not
    # expose that position separately (request_flow.py gate_costs contract).
    if kind=='state_update':return 36
    if s['attention_refined'] and kind.startswith('attention_'):
        return {'attention_qk':11000,'attention_softmax':13000,'attention_pv':17000,'attention_output':19000}[kind]
    result={'projection':30000,'attention':50000,'dense_ffn':60000,'router':9000,'expert_ffn':60000,'combine':12000}[kind]
    if device=='cpu' and kind in {'dense_ffn','expert_ffn'}:result*=2
    return result

def reference_nodes(s,d,devices,storage):
    """Construct candidate-independent action inventory from declared operands.

    Unlimited declared capacity within this microbenchmark; no eviction or
    duplicate-replica optimization is part of the declared reference family.
    Static weights can copy before their consumers. Expert copies wait for the
    current router, despite this private benchmark file containing all routes.
    """
    ops=d['operations'];tensors=d['tensors'];producer={t:oid for oid,o in ops.items() for t in [o['output'],*o['extra_outputs']]}
    nodes={};copies={}
    def addcopy(t,dst,consumer):
        src='cpu' if tensors[t]['initial_locations'] else devices[producer[t]]
        if src==dst:return None
        identity=f'copy:{t}:{src}>{dst}'
        if identity not in nodes:
            dep=[] if tensors[t]['initial_locations'] else [producer[t]]
            if tensors[t]['role']=='weight' and ':E' in t:
                # Current demand's public router is an additional obligation.
                route=[producer[x] for x in ops[consumer]['inputs'] if x in producer and ops[producer[x]]['semantic_op']=='router']
                dep.extend(route)
            nodes[identity]=dict(id=identity,kind='copy',machine='pcie',
                deps=dep,duration=tensors[t]['storage_bytes']*BYTE_NS[s['regime']],
                release=0,bytes=tensors[t]['storage_bytes'],tensor=t,src=src,dst=dst)
            copies[identity]=[]
        if consumer is not None:copies[identity].append(consumer)
        return identity
    for oid,op in ops.items():
        nodes[oid]=dict(id=oid,kind='compute',machine=devices[oid],deps=[],
            duration=op_time(s,d['contexts'][oid],op['semantic_op'],devices[oid]),
            release=2000*d['contexts'][oid]['r'] if op['semantic_op']=='state_update' and not op['inputs'] else 0,
            bytes=0)
    for oid,op in ops.items():
        for t in op['inputs']:
            cp=addcopy(t,devices[oid],oid)
            dep=cp or producer.get(t)
            if dep:nodes[oid]['deps'].append(dep)
    # The public request contract requires final hidden outputs in DRAM.
    # Persistent KV may remain at its valid producer location except where the
    # explicitly declared P/D gate consumes it at the host handoff endpoint.
    for t in d['terminal_hidden']:addcopy(t,'cpu',None)
    if storage=='demand':
        for identity,users in copies.items():
            # Demand only when the earliest topological consumer's other
            # producers finish. This is a declared bounded storage rule.
            if not users:continue
            alternatives=[sorted({producer[t] for t in ops[consumer]['inputs'] if t in producer}) for consumer in users]
            if len(alternatives)==1:nodes[identity]['deps'].extend(alternatives[0])
            else:
                # A shared weight may be demanded by any currentlyready user.
                # Dict-order request0 is not a semantic requirement.
                nodes[identity]['dependency_alternatives']=alternatives
    elif storage=='startup':
        startup=[identity for identity,n in nodes.items() if n['kind']=='copy' and tensors[n['tensor']]['role']=='weight' and ':E' not in n['tensor']]
        for oid in ops:nodes[oid]['deps'].extend(startup)
    for n in nodes.values():n['deps']=sorted(set(n['deps']))
    return nodes

def schedule(nodes,priority):
    """Independent non-delay scheduler, 16 compute/copy tie-rule pairs."""
    remaining=set(nodes);done=set();running=[];now=0;trace=[]
    def rank(identity):
        n=nodes[identity];rule=priority[0 if n['kind']=='compute' else 1]
        if rule=='id':return (0,identity)
        if rule=='reverse_id':return (0,tuple(-ord(c) for c in identity))
        return ((1 if rule=='short' else -1)*n['duration'],identity)
    for _ in range(5*len(nodes)+100):
        ended=[r for r in running if r[0]<=now]
        done.update(r[1] for r in ended);running=[r for r in running if r[0]>now]
        if not remaining and not running:return dict(duration_ns=now,events=trace)
        used={nodes[r[1]]['machine'] for r in running}
        ready=[n for n in remaining if nodes[n]['release']<=now and set(nodes[n]['deps'])<=done
            and any(set(a)<=done for a in nodes[n].get('dependency_alternatives',[[]]))
            and nodes[n]['machine'] not in used]
        if ready:
            # Use a canonical machine scan and an independently variable
            # within-machine tie rule; changing machine scan alone would not
            # constitute a substantive schedule search on independent engines.
            machine=min(nodes[n]['machine'] for n in ready)
            identity=min((n for n in ready if nodes[n]['machine']==machine),key=rank)
            n=nodes[identity];remaining.remove(identity)
            trace.append(dict(id=identity,machine=n['machine'],start_ns=now,end_ns=now+n['duration']))
            if n['duration']==0:done.add(identity)
            else:running.append((now+n['duration'],identity))
            continue
        future=[r[0] for r in running if r[0]>now]+[nodes[n]['release'] for n in remaining if nodes[n]['release']>now]
        if not future:raise ValueError('Independent reference cyclic or deadlocked')
        now=min(future)
    raise ValueError('Reference step bound')

def build(s,d,choices,storage,folder):
    devices=placement(s,d,choices)
    hw=Hardware(cpu=Device(1e9,1e9),gpu=Device(2e9,2e9),dram_capacity_bytes=1048576,vram_capacity_bytes=1048576,
        transfer_bandwidth_bytes_per_second=1e9/BYTE_NS[s['regime']],
        ssd_read_bandwidth_bytes_per_second=1e9,ssd_write_bandwidth_bytes_per_second=1e9)
    params=dict(layers=1,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1)
    model=Model.moe(experts=3,top_k=1,**params) if s['kind']=='moe' else Model.dense(**params)
    flow=RequestFlow(requests=tuple(Request(f'request{r}',1,s['decode_steps'],2000*r) for r in range(s['requests'])),
        routes=e1.routes_for(s),route_source='private fixed benchmark trace' if s['kind']=='moe' else None)
    simulator=Simulator(model=model,hardware=hw,workload=flow,cost_model=e1.WitnessCosts(s['regime']))
    def configure(plan):
        if s['requests']>1:plan.pipeline(RequestPipeline(ready_rule='decode_first'))
        if s['handoff']:plan.handoff(PhaseHandoff(destination='dram'))
        if s['attention_refined']:
            transform=e1.attention_spec(s)
            for row in transform['transfer_entries']:row['duration_ns']=row['storage_bytes']*BYTE_NS[s['regime']]
            plan.transform('request0-prefill-layer0',transform)
        for name,device in choices.items():
            if s['kind']=='moe':plan.place(name,device)
            else:
                r,phase,kind=name.split(':')
                plan.place(kind,device,select=Select(request_id='request'+r[1:],phase=phase))
        if storage in {'startup','spare'}:
            selected=set()
            for oid,op in d['operations'].items():
                if devices[oid]=='gpu':
                    selected.update(t for t in op['inputs'] if d['tensors'][t]['role']=='weight' and ':E' not in t)
            if selected:plan.prepare(DataPreparation(tensor_ids=tuple(sorted(selected)),trigger=storage))
    return simulator.compile(directory=folder/'inputs',configure=configure,scheduling=Scheduling(default_device='gpu',
        release='last_use',cache=Cache(rule='lru',budget_bytes=262144)))

def campaign(out,limit=None,only=None):
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema='ir-value-bounded-1',utc=datetime.now(timezone.utc).isoformat(),
        scenarios=[r for r in SCENARIOS if only is None or r[0]==only],regimes=['compute','transfer_constrained'],
        transfer_ns_per_byte=BYTE_NS,
        geometry=dict(H=8,I=16,heads=2,kv_heads=1,F32=True),
        execution='all CPU/GPU assignments to declared groups; no undeclared expert oracle',
        storage=['demand','spare-required-static-weights'],
        reference='Hand-authored workload DAG; independently generated transfer inventory; 16 compute/copy tie-order pairs (id/reverse-id/shortest/longest). Not exhaustive low-level scheduling, no global optimum claim.',
        scope='Analytical fixed prices and unbounded-within-1MiB residency; no native hardware accuracy or finite-cache replacement evidence.',
        ablations='all-CPU fixed E; all-GPU fixed E; demand-only S; spare-only S, each minimizes remaining candidate dimensions',
        no_backend_edits=True,code_sha256=sha(__file__),limit=limit,
        correction_note='MoE-only rerun corrects independent RequestFlow input-preparation price32to36ns per declared hidden+I32 materialization; prior assertion failures retained. No production/profile changes.' if only=='moe' else None)
    write(out/'PROTOCOL.json',protocol)
    frozen={p.relative_to(ROOT).as_posix():sha(p) for directory in
        (ROOT/'src/llm_simulator',ROOT/'mapping/tensor_mapping',ROOT/'modeling/llm_infer_model')
        for p in directory.rglob('*.py')}
    write(out/'source-manifest-before.json',frozen)
    rows=[];start=time.time()
    for name,pattern,attention,decode,requests in SCENARIOS:
      if only is not None and name!=only:continue
      for regime in protocol['regimes']:
        s=scenario(name,pattern,attention,decode,requests,regime);d=declared_dag(s)
        section=out/s['id'];write(section/'declared-dag.json',d)
        gs=groups_for(s,d)
        for index,bits in enumerate(itertools.product(('cpu','gpu'),repeat=len(gs))):
          if limit is not None and index>=limit:break
          choices=dict(zip(gs,bits));devices=placement(s,d,choices)
          for storage in ('demand','spare'):
            folder=section/f'{index:03d}-{storage}';folder.mkdir(parents=True)
            row=dict(scenario=name,regime=regime,id=f'{s["id"]}-{index:03d}-{storage}',choices=choices,storage=storage,status='failed')
            try:
              nodes=reference_nodes(s,d,devices,storage)
              refs=[schedule(nodes,p) for p in PRIORITIES]
              best_index=min(range(len(refs)),key=lambda j:refs[j]['duration_ns'])
              write(folder/'independent-reference.json',dict(nodes=nodes,best_priority=PRIORITIES[best_index],
                  schedule_durations_ns=[r['duration_ns'] for r in refs],best=refs[best_index]))
              with build(s,d,choices,storage,folder) as compiled:
                write(folder/'compiled-ir.json',compiled.ir_document())
                result=compiled.run();result.save(folder/'execution')
                if not result.success:raise AssertionError(str(result.summary()))
                graph=result.raw[1].execution_document();events=[asdict(e) for e in result.events]
                write(folder/'graph.json',graph);write(folder/'events.json',events)
                graph_check=check_graph(s,d,graph,events)
                physical=replay_physical(s,d,graph,events,result.makespan_ns)
                numeric=check_numerical(s,d,graph,events);numeric.pop('outputs',None)
                actual_device={e['target_id']:e['device'] for e in events if e['kind']=='COMPUTE'}
                for oid,op in d['operations'].items():
                    if op['semantic_op']!='state_update' and actual_device[oid]!=devices[oid]:
                        raise AssertionError('declared placement not preserved '+oid)
                for event in events:
                    if event['kind']=='COMPUTE':
                        oid=event['target_id'];expected_time=op_time(s,d['contexts'][oid],d['operations'][oid]['semantic_op'],event['device'])
                        if event['end_ns']-event['start_ns']!=expected_time:raise AssertionError('independent compute price mismatch '+oid)
                    elif event['kind']=='COPY':
                        expected_time=event['bytes_transferred']*BYTE_NS[regime]
                        if event['end_ns']-event['start_ns']!=expected_time:raise AssertionError('independent copy price mismatch')
                observed_copies=Counter((e['target_id'],'cpu' if e['src']=='dram' else 'gpu',
                    'cpu' if e['dst']=='dram' else 'gpu') for e in events if e['kind']=='COPY')
                reference_copies=Counter((n['tensor'],n['src'],n['dst']) for n in nodes.values() if n['kind']=='copy')
                if observed_copies!=reference_copies:
                    raise AssertionError('independent transfer inventory mismatch '+str(observed_copies-reference_copies)+' / '+str(reference_copies-observed_copies))
                if s['kind']=='moe':
                    from experiments.serving_acceptance.oracles import expert_visibility
                    graph_check['causal']=expert_visibility(result)
                row.update(status='pass',duration_ns=result.makespan_ns,reference_family_best_ns=refs[best_index]['duration_ns'],
                    reference_gap_fraction=result.makespan_ns/refs[best_index]['duration_ns']-1,
                    reference_distinct_durations=len({r['duration_ns'] for r in refs}),
                    graph=graph_check,physical=physical,numerical=numeric,
                    automatically_derived_copies=sum(e['kind']=='COPY' for e in events),
                    payload_bytes=sum(e['bytes_transferred'] for e in events if e['kind']=='COPY'),
                    explicit_user_copy_commands=0,backend_edits=0)
            except Exception as error:
              row.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            rows.append(row);write(folder/'result.json',row)
        print(f"{s['id']}: {sum(r['status']=='pass' for r in rows if r['scenario']==name and r['regime']==regime)} rows, elapsed {time.time()-start:.1f}s",flush=True)
        write(out/'progress.json',rows)
    summary=[]
    for name,_,_,_,_ in SCENARIOS:
      for regime in protocol['regimes']:
        current=[r for r in rows if r['scenario']==name and r['regime']==regime and r['status']=='pass']
        if not current:continue
        def best(sub):return min(sub,key=lambda x:x['duration_ns']) if sub else None
        joint=best(current);refbest=min(r['reference_family_best_ns'] for r in current)
        constraints={'cpu_only':lambda r:set(r['choices'].values())=={'cpu'},
            'gpu_only':lambda r:set(r['choices'].values())=={'gpu'},
            'demand_only':lambda r:r['storage']=='demand',
            'spare_only':lambda r:r['storage']=='spare'}
        ablations={}
        for key,f in constraints.items():
            b=best([r for r in current if f(r)])
            ablations[key]=None if b is None else dict(id=b['id'],duration_ns=b['duration_ns'],gap_fraction=b['duration_ns']/joint['duration_ns']-1)
        summary.append(dict(scenario=name,regime=regime,conditions=len(current),best=joint['id'],
            best_ns=joint['duration_ns'],reference_family_best_ns=refbest,reference_gap_fraction=joint['duration_ns']/refbest-1,
            ablations=ablations))
    imported=[]
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in {'llm_simulator','tensor_mapping','llm_infer_model'}:continue
        path=Path(module.__file__).resolve();relative=path.relative_to(ROOT).as_posix()
        imported.append(dict(module=name,path=relative,sha256=sha(path),unchanged=frozen.get(relative)==sha(path)))
    write(out/'imported-source-verification.json',imported)
    report=dict(protocol_sha256=sha(out/'PROTOCOL.json'),elapsed_seconds=time.time()-start,conditions=len(rows),
        passed=sum(r['status']=='pass' for r in rows),failed=sum(r['status']!='pass' for r in rows),
        summary=summary,rows=rows,code_sha256_unchanged=sha(__file__)==protocol['code_sha256'],
        imported_production_sources_unchanged=all(r['unchanged'] for r in imported),
        reference_gap_scope='Signed gap to best independently generated bounded non-delay schedule family; not pure expression loss or global optimality.')
    write(out/'REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int);parser.add_argument('--only',choices=[r[0] for r in SCENARIOS])
    args=parser.parse_args();campaign(args.out,args.limit,args.only)
