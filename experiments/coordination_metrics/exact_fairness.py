"""Per-policy certified scheduling gaps; never substitute a priority sweep for T*.

The reference optimizer preserves the fixed independently authored DAG, device
placements, one-copy persistent inventory, eligible preparation scope and shared
CPU/GPU/PCIe resources. It optimizes legal lower-level order, not policy choice.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import csv
from datetime import datetime,timezone
from dataclasses import asdict
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'mapping',ROOT/'modeling'):sys.path.insert(0,str(p))
from experiments.ir_value import campaign as base
from experiments.dac_validation_e1_oracle import declared_dag,check_graph,replay_physical

OUT=ROOT/'build/coordination-metrics-approved-20261030/exact'
OLD=ROOT/'build/ir-value-20261029-final-v2'
MOE=ROOT/'build/ir-value-20261029-moe-price-contract-v3'

def write(path,value):base.write(path,value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dependencies(node):
    return node.get('dependency_alternatives',[[]])

def verify(nodes,events):
    """Recheck the feasible integer witness incl. shared-copy OR readiness."""
    trace={e['id']:e for e in events}
    assert len(trace)==len(events)==len(nodes),'missing/duplicate executable node'
    for identity,n in nodes.items():
        e=trace[identity];assert e['machine']==n['machine']
        assert type(e['start_ns']) is int and type(e['end_ns']) is int
        assert e['end_ns']-e['start_ns']==n['duration'] and e['start_ns']>=n['release']
        assert all(trace[p]['end_ns']<=e['start_ns'] for p in n['deps']),('AND readiness',identity)
        assert any(all(trace[p]['end_ns']<=e['start_ns'] for p in alternative) for alternative in dependencies(n)),('OR readiness',identity)
    groups=defaultdict(list)
    for e in events:
        if e['end_ns']>e['start_ns']:groups[e['machine']].append(e)
    for machine,group in groups.items():
        group.sort(key=lambda e:e['start_ns'])
        assert all(a['end_ns']<=b['start_ns'] for a,b in zip(group,group[1:])),('resource collision',machine)
    return max(e['end_ns'] for e in events)

def lower_bound(nodes):
    """Monotone AND/OR earliest-finish relaxation plus per-resource work."""
    finish={k:n['release']+n['duration'] for k,n in nodes.items()}
    for _ in range(len(nodes)+1):
        new={k:max(n['release'],max((finish[p] for p in n['deps']),default=0),
            min(max((finish[p] for p in a),default=0) for a in dependencies(n)))+n['duration'] for k,n in nodes.items()}
        if new==finish:break
        finish=new
    else:raise AssertionError('AND/OR earliest-finish relaxation failed to converge')
    workload=defaultdict(int)
    for n in nodes.values():workload[n['machine']]+=n['duration']
    return dict(lower_bound_ns=max(max(finish.values()),max(workload.values())),
        precedence_bound_ns=max(finish.values()),resource_work_ns=dict(workload),earliest_finish_ns=finish)

def strengthened_lower_bound(nodes):
    """Resource work between two mandatory precedence endpoints is unavoidable.

    If all jobs J precede v and follow a, their total work on one exclusive
    engine lies inside [finish(a), start(v)]. This valid energetic bound also
    captures a join of several weight copies, unlike a single longest path.
    It does not impose an order on the jobs and does not aggregate their actions.
    """
    ancestors={k:set(v['deps']) for k,v in nodes.items()}
    for _ in nodes:
        following={k:set().union(a,*(ancestors[p] for p in a)) if a else set() for k,a in ancestors.items()}
        if following==ancestors:break
        ancestors=following
    energetic={}
    for target,parents in ancestors.items():
        constraints=[]
        for anchor in (None,*sorted(parents)):
            work=defaultdict(int)
            for job in parents:
                if anchor is None or anchor in ancestors[job]:work[nodes[job]['machine']]+=nodes[job]['duration']
            if work:constraints.append((anchor,max(work.values())))
        energetic[target]=constraints
    finish={k:v['release']+v['duration'] for k,v in nodes.items()}
    for _ in range(2*len(nodes)+1):
        following={k:max(v['release'],max((finish[p] for p in v['deps']),default=0),
            min(max((finish[p] for p in branch),default=0) for branch in dependencies(v)),
            max(((0 if a is None else finish[a])+w for a,w in energetic[k]),default=0))+v['duration'] for k,v in nodes.items()}
        if following==finish:break
        finish=following
    else:raise AssertionError('energetic mandatory-precedence bound failed to converge')
    result=lower_bound(nodes)
    result.update(lower_bound_ns=max(result['lower_bound_ns'],max(finish.values())),
        energetic_precedence_bound_ns=max(finish.values()),energetic_earliest_finish_ns=finish,
        proof='Each exclusive-resource subset mandatorily after anchor and before target must fit its total processing work between those endpoints. AND-only ancestors are used, so optional OR branches never create false mandatory work.')
    return result

def faithful_nodes(s,d,devices,storage):
    nodes=base.reference_nodes(s,d,devices,storage)
    demand=base.reference_nodes(s,d,devices,'demand')
    # Spare preparation declares only selected static weights, not arbitrary
    # early activation or unknown expert traffic. Preserve demand gates there.
    if storage=='spare':
        for key,n in nodes.items():
            if n['kind']!='copy':continue
            static=d['tensors'][n['tensor']]['role']=='weight' and ':E' not in n['tensor']
            if not static:
                n['deps']=demand[key]['deps']
                if 'dependency_alternatives' in demand[key]:n['dependency_alternatives']=demand[key]['dependency_alternatives']
    return nodes

def event_trace(events):
    trace=[]
    for e in events:
        if e['kind']=='COMPUTE':
            trace.append(dict(id=e['target_id'],machine=e['device'],start_ns=e['start_ns'],end_ns=e['end_ns']))
        elif e['kind']=='COPY':
            src='cpu' if e['src']=='dram' else 'gpu';dst='cpu' if e['dst']=='dram' else 'gpu'
            trace.append(dict(id=f"copy:{e['target_id']}:{src}>{dst}",machine='pcie',start_ns=e['start_ns'],end_ns=e['end_ns']))
    return trace

def selected_rows():
    rows=[]
    for directory in (OLD,MOE):
        for row in read(directory/'REPORT.json')['rows']:
            if row['status']!='pass' or (directory==OLD and row['scenario']=='moe'):continue
            prefix=f"{row['scenario']}-{row['regime']}-"
            folder=directory/f"{row['scenario']}-{row['regime']}"/row['id'][len(prefix):]
            rows.append((row,folder))
    return rows

def audit(row,folder):
    name=row['scenario'];scenario_tuple=next(x for x in base.SCENARIOS if x[0]==name)
    s=base.scenario(*scenario_tuple,row['regime']);d=declared_dag(s)
    assert d==read(folder.parent/'declared-dag.json'),'fresh handwritten algebra differs from frozen declaration'
    events=read(folder/'events.json');graph=read(folder/'graph.json');ir=read(folder/'compiled-ir.json')
    devices=base.placement(s,d,row['choices']);nodes=faithful_nodes(s,d,devices,row['storage'])
    graph_result=check_graph(s,d,graph,events)
    for e in events:
        if e['kind']=='COMPUTE':
            oid=e['target_id'];assert e['device']==devices[oid],('placement differs',oid)
            assert e['end_ns']-e['start_ns']==base.op_time(s,d['contexts'][oid],d['operations'][oid]['semantic_op'],e['device'])
        elif e['kind']=='COPY':
            assert e['bytes_transferred']==d['tensors'][e['target_id']]['storage_bytes']
            assert e['end_ns']-e['start_ns']==e['bytes_transferred']*base.BYTE_NS[row['regime']]
            assert e['resource']=='pcie_copy','expected shared nonpreemptive PCIe resource'
    actual=event_trace(events);tc=verify(nodes,actual);assert tc==row['duration_ns']
    physical=replay_physical(s,d,graph,events,tc)
    # Read the IR only to audit declared scope, not to create the independent DAG.
    expected_static={n['tensor'] for n in nodes.values() if n['kind']=='copy' and n['dst']=='gpu'
        and d['tensors'][n['tensor']]['role']=='weight' and ':E' not in n['tensor']}
    declared_prepared={t for p in ir['data_preparations'] for t in p['tensor_ids']}
    assert declared_prepared==(expected_static if row['storage']=='spare' else set()),'preparation selector mismatch'
    assert all(p['trigger']=='spare' and p['destination']=='vram' for p in ir['data_preparations'])
    assert ir['backend']['cache']['rule']=='lru' and ir['backend']['cache']['budget_bytes']==262144
    assert ir['backend']['policy']['experts']['device_rule']=='fixed'
    assert ir['backend']['policy']['release']['rule']=='last_use'
    # Capacity never binds: even retaining all declared allocations twice and
    # every declared workspace simultaneously fits each 1MiB physical pool.
    capacity_bound=2*sum(t['storage_bytes'] for t in d['tensors'].values())+4*len(d['operations'])
    assert capacity_bound<1048576
    feasible=[dict(kind='actual_compiler',duration_ns=tc,events=actual)]
    for p in base.PRIORITIES:
        candidate=base.schedule(nodes,p);verify(nodes,candidate['events'])
        feasible.append(dict(kind='independent_feasible_priority_'+str(p),**candidate))
    upper=min(feasible,key=lambda x:x['duration_ns'])
    result=dict(id=row['id'],scenario=name,regime=row['regime'],storage=row['storage'],choices=row['choices'],
        T_C_ns=tc,node_count=len(nodes),compute_nodes=sum(n['kind']=='compute' for n in nodes.values()),
        copy_nodes=sum(n['kind']=='copy' for n in nodes.values()),OR_groups=sum('dependency_alternatives' in n for n in nodes.values()),
        capacity_proof_each_pool_upper_bound_bytes=capacity_bound,
        source_directory=str(folder),source_files={p.name:sha(p) for p in (folder/'result.json',folder/'events.json',folder/'graph.json',folder/'compiled-ir.json',folder.parent/'declared-dag.json')},
        graph_audit=graph_result,physical_peak_audit=physical['peak_physical_bytes'],
        fidelity=dict(placement=True,complete_tasks=True,dependencies=True,unique_COPY_inventory=True,
            COPY_payload_and_price=True,shared_CPU_GPU_PCIe_resources=True,explicit_preparation_scope=True,
            demand_shared_COPY_OR_readiness=True,terminal_DRAM_delivery=True,
            request_decode_priority='not_triggered: pipeline has Prefill only; PD one request',
            expert_order='not_triggered: top1 with one-token Prefill gives one expert per invocation',
            finite_cache_replacement='not_triggered: conservative full-retention capacity proof',
            future_routing='MoE private known test trace; only current router may release expert copies, no future route prefetch in this fixed inventory'),
        lower_bound=lower_bound(nodes),feasible_upper_bound_ns=upper['duration_ns'])
    return result,nodes,upper

_dll_handles=[]
def solver_dependencies():
    try:import scipy
    except ImportError:
        # Optional fallback: read-only use of a scipy already installed in another
        # interpreter of the same CPython ABI. Set LLM_INFER_SCIPY_ROOT to that
        # environment; it must contain Lib/site-packages and Library/bin.
        root=os.environ.get('LLM_INFER_SCIPY_ROOT')
        if not root:raise
        package=Path(root)/'Lib'/'site-packages'
        sys.path.append(str(package))
        _dll_handles.append(os.add_dll_directory(str(Path(root)/'Library'/'bin')))
        import scipy
    import numpy
    return dict(python=sys.executable,python_version=sys.version,scipy=scipy.__version__,scipy_file=scipy.__file__,numpy=numpy.__version__)

def solve_exact(nodes,upper,time_limit=3):
    """Disjunctive MILP with a selected complete prerequisite branch per OR."""
    solver_dependencies()
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint,milp
    from scipy.sparse import lil_matrix
    identities=list(nodes);index={k:i for i,k in enumerate(identities)};n=len(nodes)
    groups=defaultdict(list)
    for k,v in nodes.items():groups[v['machine']].append(k)
    ancestors={k:set(v['deps']) for k,v in nodes.items()}
    for _ in nodes:
        new={k:set().union(v,*(ancestors[p] for p in v)) if v else set() for k,v in ancestors.items()}
        if new==ancestors:break
        ancestors=new
    pairs=[(a,b) for g in groups.values() for a,b in itertools.combinations(g,2)
        if nodes[a]['duration'] and nodes[b]['duration'] and a not in ancestors[b] and b not in ancestors[a]]
    or_groups={k:dependencies(v) for k,v in nodes.items() if 'dependency_alternatives' in v}
    alternatives=[(k,j) for k,group in or_groups.items() for j in range(len(group))]
    selected={pair:n+1+len(pairs)+i for i,pair in enumerate(alternatives)}
    variables=n+1+len(pairs)+len(alternatives)
    horizon=upper['duration_ns']+max(v['duration'] for v in nodes.values())+1
    constraints=[];low=[];high=[]
    def add(v,lo=-np.inf,hi=np.inf):constraints.append(v);low.append(lo);high.append(hi)
    for k,v in nodes.items():
        add({n:1,index[k]:-1},lo=v['duration'])
        for parent in v['deps']:add({index[k]:1,index[parent]:-1},lo=nodes[parent]['duration'])
    for j,(a,b) in enumerate(pairs):
        binary=n+1+j
        add({index[a]:1,index[b]:-1,binary:horizon},hi=horizon-nodes[a]['duration'])
        add({index[b]:1,index[a]:-1,binary:-horizon},hi=-nodes[b]['duration'])
    for k,group in or_groups.items():
        add({selected[k,j]:1 for j in range(len(group))},lo=1,hi=1)
        for j,branch in enumerate(group):
            for parent in branch:
                add({index[k]:1,index[parent]:-1,selected[k,j]:-horizon},lo=nodes[parent]['duration']-horizon)
    # Valid energetic precedence cuts. They change neither policy nor inventory.
    strong=strengthened_lower_bound(nodes)
    for target,parents in ancestors.items():
        for anchor in sorted(parents):
            work=defaultdict(int)
            for job in parents:
                if anchor in ancestors[job]:work[nodes[job]['machine']]+=nodes[job]['duration']
            if work:add({index[target]:1,index[anchor]:-1},lo=nodes[anchor]['duration']+max(work.values()))
    matrix=lil_matrix((len(constraints),variables),dtype=float)
    for row,entries in enumerate(constraints):
        for col,value in entries.items():matrix[row,col]=value
    objective=np.zeros(variables);objective[n]=1
    lower=np.array([max(nodes[k]['release'],strong['energetic_earliest_finish_ns'][k]-nodes[k]['duration']) for k in identities]+[strong['lower_bound_ns']]+[0]*(variables-n-1),dtype=float)
    upperbounds=np.array([upper['duration_ns']]*(n+1)+[1]*(variables-n-1),dtype=float)
    before=time.perf_counter()
    result=milp(objective,integrality=np.array([0]*(n+1)+[1]*(variables-n-1)),
        bounds=Bounds(lower,upperbounds),constraints=LinearConstraint(matrix.tocsr(),np.array(low),np.array(high)),
        options=dict(time_limit=time_limit,mip_rel_gap=0,threads=1))
    doc=dict(status=int(result.status),message=result.message,elapsed_seconds=time.perf_counter()-before,
        variables=variables,resource_binary_pairs=len(pairs),OR_selector_variables=len(alternatives),constraints=len(constraints),
        objective_ns=None if result.fun is None else float(result.fun),dual_bound_ns=getattr(result,'mip_dual_bound',None),
        mip_gap=getattr(result,'mip_gap',None),integer_tick_witness_valid=False)
    if result.x is not None:
        trace=[dict(id=k,machine=nodes[k]['machine'],start_ns=int(round(result.x[index[k]])),
            end_ns=int(round(result.x[index[k]]))+nodes[k]['duration']) for k in identities]
        try:
            makespan=verify(nodes,trace)
            doc.update(integer_tick_witness_valid=True,verified_makespan_ns=makespan,events=trace)
        except AssertionError as exc:doc['witness_failure']=str(exc)
    return doc

def certify(row,nodes,upper,limit):
    lower=row['lower_bound']['lower_bound_ns'];bound=upper['duration_ns'];witness=upper
    result=dict(method='matching_integer_bounds',lower_bound_ns=lower,upper_bound_ns=bound,witness_kind=upper['kind'])
    if lower<bound and limit:
        solved=solve_exact(nodes,upper,time_limit=limit);result['milp']=solved
        if solved.get('integer_tick_witness_valid') and solved['verified_makespan_ns']<bound:
            bound=solved['verified_makespan_ns'];witness=dict(kind='MILP_integer_rechecked',duration_ns=bound,events=solved['events'])
        dual=solved.get('dual_bound_ns')
        if dual is not None and math.isfinite(dual):lower=max(lower,math.ceil(dual-1e-4))
        result.update(method='AND_OR_disjunctive_MILP',lower_bound_ns=lower,upper_bound_ns=bound,witness_kind=witness['kind'])
    assert lower<=bound,(row['id'],'invalid bounds',lower,bound)
    assert row['T_C_ns']>=bound,(row['id'],'negative gap before certification')
    result.update(certified=lower==bound,T_star_ns=bound if lower==bound else None,
        g=(row['T_C_ns']-bound)/bound if lower==bound else None,
        g_interval=[(row['T_C_ns']-bound)/bound,(row['T_C_ns']-lower)/lower],
        feasible_optimal_or_upper_witness=witness)
    return result

def campaign(out=OUT,mode='audit',limit=3):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if mode=='audit':
        if (out/'PROTOCOL.json').exists():raise ValueError('Frozen output exists')
        protocol=dict(schema='per-policy-exact-fairness-1',frozen_utc=datetime.now(timezone.utc).isoformat(),
            primary_metric='g(p)=(T_C(p)-T*(p))/T*(p), reported for each fixed policy; no best-of-space substitution',
            corpus='800 successful frozen points: Dense32, Attention256, Prefill pipeline256, legal DRAM-handoff PD128, causal MoE128. The other128PD unsupported choices remain excluded, not treated as infeasible physics.',
            physical_scope='H8/I16/q1, analytical fixed costs, nonpreemptive CPU and GPU, one shared PCIe engine; unique tensor COPY inventory; persistent copies legal under1MiB capacity proof; graph-order/release details are compiler decisions, not imposed optimality constraints.',
            policy_scope='Execution placements fixed. Demand COPY waits for any full consumer-producer readiness branch. Spare only relaxes readiness for explicitly selected static weights. Expert copies wait for their router. Final hidden delivery is DRAM.',
            OR_fix='Unlike the old exact_oracle.py solver, all dependency_alternatives are represented by branch-selection binary variables and independently checked in integer witnesses.',
            lower_bound='AND/OR earliest finish and resource totals, optionally tightened by MILP dual bound; known feasible integer witness upper bound. Exact only when integer lower==upper.',
            time_limit_seconds=limit,source_sha256=sha(__file__),
            source_reports={str(p):sha(p) for p in (OLD/'REPORT.json',MOE/'REPORT.json')},
            runtime=sys.executable,source_reference=sha(ROOT/'experiments/ir_value/campaign.py'),
            legacy_exact_oracle=sha(ROOT/'experiments/ir_value/exact_oracle.py'),
            selection='First audit all800. Solve allDense remaining cases up to10s each, then bounded representative non-Dense policies (CPU-only,GPU-only,first/middle/last mixed placements per family/storage/regime); also retain exact lower=upper results for any other corpus point. Every unresolved case remains an interval, no exact max over all800 claim unless bounded.')
        write(out/'PROTOCOL.json',protocol)
        rows=[]
        for original,folder in selected_rows():
            try:
                row,nodes,upper=audit(original,folder);row['audit_status']='pass'
                certificate=certify(row,nodes,upper,0);row['certificate']=certificate
                write(out/'cases'/row['id']/'nodes.json',nodes)
                write(out/'cases'/row['id']/'audit.json',row)
            except Exception as exc:
                row=dict(id=original['id'],scenario=original['scenario'],regime=original['regime'],storage=original['storage'],
                    choices=original['choices'],audit_status='fail',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
                write(out/'cases'/row['id']/'audit.json',row)
            rows.append(row)
        write(out/'AUDIT.json',dict(rows=rows,counts=dict(Counter(r['audit_status'] for r in rows))))
        print('AUDIT',dict(Counter(r['audit_status'] for r in rows)),flush=True)
    elif mode=='solve':
        solver_info=solver_dependencies();write(out/'SOLVER_ENVIRONMENT.json',solver_info)
        rows=read(out/'AUDIT.json')['rows'];selected=set()
        for scenario in sorted({r['scenario'] for r in rows}):
            for regime in ('compute','transfer_constrained'):
                for storage in ('demand','spare'):
                    current=[r for r in rows if r['audit_status']=='pass' and r['scenario']==scenario and r['regime']==regime and r['storage']==storage]
                    if scenario=='dense':selected.update(r['id'] for r in current);continue
                    mixed=[r for r in current if set(r['choices'].values())=={'cpu','gpu'}]
                    selected.update(r['id'] for r in current if len(set(r['choices'].values()))==1)
                    if mixed:selected.update(mixed[i]['id'] for i in {0,len(mixed)//2,len(mixed)-1})
        write(out/'SOLVE_SELECTION.json',dict(selected_policy_ids=sorted(selected),time_limit_seconds=limit,dense_time_limit_seconds=10,
            criterion='Selection fixed by family,regime,storage,placement index, never observed gap. Bound-matched certificates retained for all800.'))
        for row in rows:
            if row['audit_status']!='pass' or row['id'] not in selected or row['certificate']['certified']:continue
            nodes=read(out/'cases'/row['id']/'nodes.json');upper=row['certificate']['feasible_optimal_or_upper_witness']
            result=certify(row,nodes,upper,10 if row['scenario']=='dense' else limit);row['certificate']=result
            write(out/'cases'/row['id']/'certificate.json',result)
            print(row['id'],'certified',result['certified'],'bounds',result['lower_bound_ns'],result['upper_bound_ns'],'g',result['g'],flush=True)
        write(out/'RESULTS.json',dict(rows=rows))
    elif mode=='strengthen':
        original=read(out/'RESULTS.json');write(out/'RESULTS.stage1.json',original)
        rows=original['rows']
        for row in rows:
            if row['audit_status']!='pass':continue
            nodes=read(out/'cases'/row['id']/'nodes.json');bound=strengthened_lower_bound(nodes)
            row['strengthened_lower_bound']=bound;c=row['certificate']
            lower=max(c['lower_bound_ns'],bound['lower_bound_ns']);upper=c['upper_bound_ns']
            assert lower<=upper,(row['id'],lower,upper)
            verify(nodes,c['feasible_optimal_or_upper_witness']['events'])
            c.update(lower_bound_ns=lower,certified=lower==upper,T_star_ns=upper if lower==upper else None,
                g=(row['T_C_ns']-upper)/upper if lower==upper else None,
                g_interval=[(row['T_C_ns']-upper)/upper,(row['T_C_ns']-lower)/lower],
                strengthened_with_mandatory_interval_resource_work=True)
        write(out/'RESULTS.json',dict(rows=rows))
        print('strengthened',Counter((r['scenario'],r['certificate']['certified']) for r in rows if r['audit_status']=='pass'),flush=True)
    elif mode=='deepen':
        original=read(out/'RESULTS.json');write(out/'RESULTS.before-deepen.json',original)
        valid=[r for r in original['rows'] if r['audit_status']=='pass']
        wanted={r['id'] for r in valid if r['scenario']=='dense' and not r['certificate']['certified']}
        wanted.add(max(valid,key=lambda r:r['certificate']['g_interval'][0])['id'])
        write(out/'DEEPEN_ADDENDUM.json',dict(ids=sorted(wanted),time_limit_seconds=30,
            reason='Resolve remaining Dense point and the candidate proving the largest current corpus-wide gap lower bound; same frozen metric/inventory, stronger valid MILP cuts.',
            source_sha256=sha(__file__),old_results_sha256=sha(out/'RESULTS.before-deepen.json')))
        for r in valid:
            if r['id'] not in wanted:continue
            nodes=read(out/'cases'/r['id']/'nodes.json');upper=r['certificate']['feasible_optimal_or_upper_witness']
            r['lower_bound']=r['strengthened_lower_bound']
            r['certificate']=certify(r,nodes,upper,30)
            write(out/'cases'/r['id']/'certificate-deepened.json',r['certificate'])
            print(r['id'],r['certificate']['certified'],r['certificate']['g_interval'],flush=True)
        write(out/'RESULTS.json',original)
    elif mode=='closemax':
        document=read(out/'RESULTS.json');write(out/'RESULTS.before-closemax.json',document)
        rows=[r for r in document['rows'] if r['audit_status']=='pass']
        certified_max=max(r['certificate']['g'] for r in rows if r['certificate']['certified'])
        selected=[r for r in rows if not r['certificate']['certified'] and r['certificate']['g_interval'][1]>certified_max]
        write(out/'MAX_BOUND_ADDENDUM.json',dict(frozen_candidate_ids=[r['id'] for r in selected],time_limit_seconds=10,
            existing_certified_max_g=certified_max,reason='Tighten the whole-corpus maximum-gap interval. All per-policy rows retained; max certificate does not imply individual optima for the unselected unresolved policies.'))
        for row in selected:
            nodes=read(out/'cases'/row['id']/'nodes.json');old=row['certificate'];upper=old['feasible_optimal_or_upper_witness']
            row['lower_bound']=row['strengthened_lower_bound']
            certificate=certify(row,nodes,upper,10)
            if certificate['lower_bound_ns']<old['lower_bound_ns']:certificate['lower_bound_ns']=old['lower_bound_ns']
            lb,ub=certificate['lower_bound_ns'],certificate['upper_bound_ns']
            certificate.update(certified=lb==ub,T_star_ns=ub if lb==ub else None,g=(row['T_C_ns']-ub)/ub if lb==ub else None,
                g_interval=[(row['T_C_ns']-ub)/ub,(row['T_C_ns']-lb)/lb])
            row['certificate']=certificate
            write(out/'cases'/row['id']/'certificate-max-bound.json',certificate)
            write(out/'RESULTS.json',document)
            print('max-bound',row['id'],certificate['certified'],certificate['g_interval'],flush=True)
    elif mode=='fresh':fresh_checks(out)
    elif mode=='ffn':ffn_checks(out)
    elif mode=='report':report(out)
    else:raise ValueError(mode)

def report(out=OUT):
    out=Path(out);rows=read(out/'RESULTS.json')['rows'];valid=[r for r in rows if r['audit_status']=='pass']
    exact=[r for r in valid if r['certificate']['certified']]
    unresolved=[r for r in valid if not r['certificate']['certified']]
    worst=max(exact,key=lambda r:r['certificate']['g'])
    summary=[]
    for family in sorted({r['scenario'] for r in valid}):
        group=[r for r in valid if r['scenario']==family];cert=[r for r in group if r['certificate']['certified']]
        summary.append(dict(scenario=family,policies=len(group),certified=len(cert),unresolved=len(group)-len(cert),
            max_certified_g=max((r['certificate']['g'] for r in cert),default=None),
            max_g_lower_bound=max(r['certificate']['g_interval'][0] for r in group),
            max_g_upper_bound=max(r['certificate']['g_interval'][1] for r in group)))
    maximum_interval=[max(r['certificate']['g_interval'][0] for r in valid),max(r['certificate']['g_interval'][1] for r in valid)]
    result=dict(policies=len(rows),independent_fidelity_passed=len(valid),certified_individual_optima=len(exact),
        unresolved_individual_optima=len(unresolved),negative_gap_count=sum(r['certificate']['g_interval'][0]<0 for r in valid),
        worst_certified_policy=worst['id'],max_certified_g=worst['certificate']['g'],
        maximum_over_entire_corpus_interval=maximum_interval,
        entire_corpus_max_gap_certified=maximum_interval[0]==maximum_interval[1],
        families=summary,unresolved_policy_ids=[r['id'] for r in unresolved],
        source_sha256=sha(__file__),test_source_sha256=sha(Path(__file__).with_name('test_exact_fairness.py')),
        limitations=['Bounded analytical kernels and fixed single CPU/GPU/shared PCIe, no native accuracy conclusion.',
            'Shared-copy OR and selected static-prefetch eligibility explicitly constrained; solver changes order only.',
            'Most timing traces are retained20261029 compiler version. Source manifests retained; no claim that every result describes current compiler.',
            'Unresolved rows retain g intervals and no T* scalar. Whole-corpus maximum is certified only if max of all per-policy lower bounds equals max of all per-policy upper bounds.',
            'T* is offline for the complete declared workload, permits deliberate idle time and knows request arrival times. It is not the optimum of unknown-future, work-conserving online dispatch. Expert COPY remains router-gated.',
            'Requestpipeline benchmark has two Prefill requests, no mixedPD queue policy trigger; MoE top1 q1 has one current expert, no ordering competition.',
            'Large memory prevents finite-cache replacement and alias-sharing obligations from being exercised.'])
    nodes=read(out/'cases'/worst['id']/'nodes.json')
    priority_candidates=[dict(priority=p,**base.schedule(nodes,p)) for p in base.PRIORITIES]
    best_ready=min(priority_candidates,key=lambda r:r['duration_ns']);verify(nodes,best_ready['events'])
    result['ready_only_priority_witness']=dict(policy=worst['id'],priority=best_ready['priority'],duration_ns=best_ready['duration_ns'],
        observed_gap_to_this_feasible_witness=(worst['T_C_ns']-best_ready['duration_ns'])/best_ready['duration_ns'],
        scope='Non-delay fixed priority over currently ready jobs; does not choose a deliberate idle interval to wait for an unarrived request. Feasible witness only, not an online-optimum certificate.')
    write(out/'READY_ONLY_COUNTEREXAMPLE.json',dict(policy_id=worst['id'],**best_ready))
    weight_proofs=[]
    for folder in sorted({Path(r['source_directory']).parent for r in valid}):
        declared=read(folder/'declared-dag.json');weight_bytes=sum(t['storage_bytes'] for t in declared['tensors'].values() if t['role']=='weight')
        assert weight_bytes<=262144
        weight_proofs.append(dict(scenario_directory=str(folder),all_declared_weights_bytes=weight_bytes,cache_budget_bytes=262144))
    result['optimal_persistent_inventory_cache_feasibility']=weight_proofs
    write(out/'SUMMARY.json',result)
    columns=['id','scenario','regime','storage','T_C_ns','T_star_ns','lower_bound_ns','upper_bound_ns','g','g_lower','g_upper','certified','node_count','copy_nodes','OR_groups']
    with (out/'PER_POLICY.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for r in valid:
            c=r['certificate'];writer.writerow({**{k:r[k] for k in ('id','scenario','regime','storage','T_C_ns','node_count','copy_nodes','OR_groups')},
                **{k:c[k] for k in ('T_star_ns','lower_bound_ns','upper_bound_ns','g','certified')},'g_lower':c['g_interval'][0],'g_upper':c['g_interval'][1]})
    actual=event_trace(read(Path(worst['source_directory'])/'events.json'))
    write(out/'WORST_CERTIFIED_COUNTEREXAMPLE.json',dict(policy=worst,
        current_compiler_trace=actual,optimal_trace=worst['certificate']['feasible_optimal_or_upper_witness']['events'],
        nodes=read(out/'cases'/worst['id']/'nodes.json'),
        conclusion='Same declared execution placement and preparation scope; difference is legal ordering under the fixed independent inventory. This is a genuine counterexample to an already-negligible compiler scheduling gap.'))
    commands=[f'"{sys.executable}" experiments/coordination_metrics/exact_fairness.py --mode {m} --limit 3' for m in ('audit','solve','strengthen','deepen','closemax','fresh','ffn','report')]
    write(out/'RUN_COMMANDS.json',dict(cwd=str(ROOT),commands=commands,tests=f'"{sys.executable}" -m unittest experiments.coordination_metrics.test_exact_fairness -v',
        environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),solver_package=read(out/'SOLVER_ENVIRONMENT.json')))
    fresh=read(out/'FRESH_RESULTS.json')['rows'] if (out/'FRESH_RESULTS.json').exists() else []
    priorfresh=read(out/'FRESH_RESULTS.json.stage1')['rows'] if (out/'FRESH_RESULTS.json.stage1').exists() else []
    ffn=read(out/'FFN_RESULTS.json')['rows'] if (out/'FFN_RESULTS.json').exists() else []
    family_rows=''.join('<tr><td>'+r['scenario']+'</td><td>'+str(r['policies'])+'</td><td>'+str(r['certified'])+'</td><td>'+str(r['unresolved'])+'</td><td>'+f"{100*r['max_certified_g']:.4f}%"+'</td><td>'+f"[{100*r['max_g_lower_bound']:.4f}%, {100*r['max_g_upper_bound']:.4f}%]"+'</td></tr>' for r in summary)
    fresh_rows=''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+str(r['retained_T_C_ns'])+'</td><td>'+str(r['T_C_ns'])+'</td><td>'+str(r['certificate']['T_star_ns'])+'</td><td>'+f"{100*r['certificate']['g']:.4f}%"+'</td></tr>' for r in fresh)
    ffntxt='; '.join(f"{r['storage']}: T_C={r['T_C_ns']}ns，T*={r['certificate']['T_star_ns']}ns，g={100*r['certificate']['g']:.4f}%" for r in ffn)
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐策略精确编译调度差距</title><style>
body{{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.65 Georgia,"Times New Roman",serif;color:#111;background:#fff}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:right;padding:8px;border-bottom:1px solid #ccc}}td:first-child,th:first-child{{text-align:left}}.note{{border:1px solid #999;padding:16px;background:#f7f7f7}}code{{font-size:14px}}</style>
<h1>指标3：同一策略，当前编译调度离最优还差多少？</h1>
<p>对每一条固定策略p计算 g(p)=(T_C−T*)/T*。T_C是实际编译模拟的完成时间；T*保持同一数学图、设备放置、允许的预备加载对象、COPY库存、依赖和CPU/GPU/PCIe资源，只优化合法执行顺序。没有用“整个策略空间找到一个好结果”代替逐策略评价。</p>
<p class="note"><b>结果：</b>800条保留策略均通过独立策略忠实性审计。{len(exact)}条获得独立最优值证书，{len(unresolved)}条仍保留上下界而没有填T*。所有策略的gap最大值已通过全体上下界相等证明为<b>{100*maximum_interval[0]:.4f}%</b>；这不意味着其余未决策略的个人最优值也已证明。这个结果不支持“底层编译器已经没有多少优化空间”。</p>
<h2>最优值如何证明</h2><p>用独立数学图生成任务和唯一COPY库存；每个COPY的目标、字节数、初始来源与消费者都独立于编译器动作序列确定。demand等待至少一个完整消费者依赖分支就绪；spare只允许选定静态权重提前搬运。共享COPY使用OR就绪条件，不能被旧求解器的普通AND依赖替代或忽略。</p>
<p>下界来自前驱关系、各资源总工作量及“必经前驱/后继之间的同资源工作量”；上界必须有逐条验证过的整数时序。必要时使用含资源互斥与OR分支选择的MILP，保留求解状态、dual bound、整数可行解和完整模型节点。只有上下界相等才填T*；16种优先级仅提供可行上界，绝不直接充当最优。</p>
<table><thead><tr><th>族</th><th>策略数</th><th>已证个人T*</th><th>保留区间</th><th>已证子集max g</th><th>该族全体max g区间</th></tr></thead><tbody>{family_rows}</tbody></table>
<h2>最差反例</h2><p><code>{html.escape(worst['id'])}</code>：当前完成时间{worst['T_C_ns']}ns，独立最优{worst['certificate']['T_star_ns']}ns，差距{100*worst['certificate']['g']:.4f}%。请求0的三个宏任务均在CPU；请求1的Projection在CPU，Attention和FFN在GPU。当前调度先把请求0的CPU链全部做完，再启动请求1的Projection；更好的顺序提前准备请求1，让其GPU链与请求0的CPU链重叠。E/S放置和按需搬运规则没有改变。</p>
<p><b>离线与因果边界：</b>T*针对完整声明workload的离线时序，可利用已声明arrival并允许主动空闲。它不是未知未来请求下的在线最优。另一个只选择当前就绪任务的固定优先级可行方案已做到{best_ready['duration_ns']}ns，相对原调度仍有{100*result['ready_only_priority_witness']['observed_gap_to_this_feasible_witness']:.4f}%的差距；无需将这一更保守的反例称为最优。</p>
<h2>当前编译器重新运行</h2><p>800条主体来自2026-10-29保留版本，不能一概当作当前版本。下面重新调用当前公共compile/run，数学图、策略和价格保持相同，最优参考域逐字核对一致。</p><table><thead><tr><th>策略</th><th>旧T_C/ns</th><th>当前T_C/ns</th><th>T*/ns</th><th>当前g</th></tr></thead><tbody>{fresh_rows}</tbody></table>
<p>其中旧Dense某例14.467%的差距已由当前版本消除；最差pipeline反例仍然存在。另2条当前FFN实体分片补充实验单独保存，不混入800条统计：{ffntxt}。</p>
<h2>范围与未触发规则</h2><p>全部是H8/I16/q1的小图、固定解析成本和独立CPU/GPU内存。每条策略都核对任务、placement、依赖、COPY唯一库存/字节/价格、资源互斥、明确的预备对象、router可见边界与最终DRAM输出。保留全部物理分配也能容纳于1MiB容量，全部权重小于256KiB缓存，因此没有测试容量压力下的替换优化。</p>
<p>两请求pipeline只有Prefill，decode_first的跨阶段优先条件没有触发；MoE为单token top1，每次只有一个当前专家，没有专家间争用排序。这些限制明确保留。所有结果都不是实机耗时准确率；不证明数值MoE真实模型、UMA或设备波动性能。</p>
<h2>产物</h2><p><a href="PER_POLICY.csv">800条逐策略CSV</a> · <a href="RESULTS.json">完整结果与证书</a> · <a href="SUMMARY.json">总结</a> · <a href="WORST_CERTIFIED_COUNTEREXAMPLE.json">最差反例完整时序</a> · <a href="READY_ONLY_COUNTEREXAMPLE.json">当前就绪优先级可行方案</a> · <a href="RUN_COMMANDS.json">命令与依赖</a>。未改公共核心，无真机测量。7个独立oracle测试通过，首次实验脚本误写PCIe资源名称的失败结果保存在audit-attempt-v1，修正后800条审计全部通过。</p></html>'''
    (out/'REPORT_zh.html').write_text(content,encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'unresolved_policy_ids','optimal_persistent_inventory_cache_feasibility'}},ensure_ascii=False,indent=2),flush=True)

def fresh_checks(out=OUT):
    out=Path(out);rows=read(out/'RESULTS.json')['rows'];byid={r['id']:r for r in rows}
    wanted={'dense-transfer_constrained-003-spare','dense-transfer_constrained-006-spare',
        'pipeline-transfer_constrained-006-spare',max(rows,key=lambda r:r['certificate']['g_interval'][0])['id']}
    write(out/'FRESH_PROTOCOL.json',dict(policy_ids=sorted(wanted),purpose='Compare retained20261029 compiler timing to current core with identical workload/strategy/prices and same independent optimum certificate.',
        source_manifest={p.relative_to(ROOT).as_posix():sha(p) for directory in ('src/llm_simulator','mapping/tensor_mapping','modeling/llm_infer_model') for p in (ROOT/directory).rglob('*.py')},
        execution='real public compile+run, no native kernel timing'))
    fresh=[]
    for identity in sorted(wanted):
        original=byid[identity];name=original['scenario'];entry=next(x for x in base.SCENARIOS if x[0]==name)
        s=base.scenario(*entry,original['regime']);d=declared_dag(s)
        section=out/'fresh'/s['id'];folder=section/identity;write(section/'declared-dag.json',d)
        record={k:original[k] for k in ('id','scenario','regime','storage','choices')}
        before=time.perf_counter_ns()
        with base.build(s,d,original['choices'],original['storage'],folder) as compiled:
            compile_ns=time.perf_counter_ns()-before;write(folder/'compiled-ir.json',compiled.ir_document())
            before=time.perf_counter_ns();result=compiled.run();evaluate_ns=time.perf_counter_ns()-before
            write(folder/'events.json',[asdict(e) for e in result.events]);write(folder/'graph.json',result.raw[1].execution_document())
            record.update(status='pass' if result.success else 'failed',duration_ns=result.makespan_ns)
            write(folder/'result.json',record)
        new,nodes,upper=audit(record,folder)
        assert nodes==read(out/'cases'/identity/'nodes.json'),'fresh comparison changed independent optimization domain'
        c=deepcopy(original['certificate']);lower=c['lower_bound_ns'];bound=min(c['upper_bound_ns'],upper['duration_ns'])
        if upper['duration_ns']<c['upper_bound_ns']:c['feasible_optimal_or_upper_witness']=upper
        c.update(upper_bound_ns=bound,certified=lower==bound,T_star_ns=bound if lower==bound else None,
            g=(new['T_C_ns']-bound)/bound if lower==bound else None,
            g_interval=[(new['T_C_ns']-bound)/bound,(new['T_C_ns']-lower)/lower])
        assert new['T_C_ns']>=bound
        new.update(certificate=c,retained_T_C_ns=original['T_C_ns'],compile_wall_ns=compile_ns,evaluate_wall_ns=evaluate_ns,
            interpretation='Fresh current-core result, same policy, independent DAG and prices as retained source.')
        write(folder/'FRESH_RESULT.json',new);fresh.append(new)
        print('fresh',identity,original['T_C_ns'],'->',new['T_C_ns'],'gap',c['g_interval'],flush=True)
    write(out/'FRESH_RESULTS.json',dict(rows=fresh))

def ffn_checks(out=OUT):
    """Two fresh materialized FFN shard policies, kept outside the frozen800."""
    from experiments.coordination_metrics.physical_protocol import declarations
    from experiments.coordination_metrics.physical_run import build as physical_build,device_for
    from experiments.coordination_metrics.physical_ledger import replay as physical_replay
    from experiments.dac_validation_e1_numeric import check_numerical
    out=Path(out);cases=[r for r in declarations() if r['id'] in {'F01-development-A','F02-development-B'}]
    write(out/'FFN_PROTOCOL.json',dict(cases=cases,scope='Two fresh current-core FFN materialized partition policies, demand/spare; separate supplement, not inserted into frozen800.',
        method='Independently authored mathematical graph and physical allocation ledger, explicit fixed stage prices, same AND/OR/resource MILP; no native timing.',
        frozen_before_execution=True))
    rows=[]
    for declaration in cases:
        folder=out/'ffn'/declaration['id'];s=declaration['workload'];d=declared_dag(s)
        devices={oid:device_for(declaration,d,oid) for oid in d['operations']}
        prices=dict(state_update=36,projection=30000,attention=50000,ffn_norm=7000,ffn_gate=11000,
            ffn_up=13000,ffn_swiglu=17000,ffn_down=19000,ffn_sum=23000,ffn_residual=29000)
        producer={tid:oid for oid,op in d['operations'].items() for tid in [op['output'],*op['extra_outputs']]}
        nodes={oid:dict(id=oid,kind='compute',machine=devices[oid],deps=[],duration=prices[op['semantic_op']],release=0,bytes=0)
            for oid,op in d['operations'].items()};users=defaultdict(list)
        def copy(tid,dst,consumer=None):
            src='cpu' if d['tensors'][tid]['initial_locations'] else devices[producer[tid]]
            if src==dst:return None
            identity=f'copy:{tid}:{src}>{dst}'
            if identity not in nodes:nodes[identity]=dict(id=identity,kind='copy',machine='pcie',
                deps=[] if d['tensors'][tid]['initial_locations'] else [producer[tid]],
                duration=d['tensors'][tid]['storage_bytes'],release=0,bytes=d['tensors'][tid]['storage_bytes'],tensor=tid,src=src,dst=dst)
            if consumer is not None:users[identity].append(consumer)
            return identity
        for oid,op in d['operations'].items():
            for tid in op['inputs']:
                dep=copy(tid,devices[oid],oid) or producer.get(tid)
                if dep:nodes[oid]['deps'].append(dep)
        for tid in d['terminal_hidden']:copy(tid,'cpu')
        for identity,consumers in users.items():
            n=nodes[identity];static=d['tensors'][n['tensor']]['role']=='weight'
            if declaration['intent']['preparation']=='spare' and static:continue
            alternatives=[sorted({producer[t] for t in d['operations'][oid]['inputs'] if t in producer}) for oid in consumers]
            if len(alternatives)==1:n['deps'].extend(alternatives[0])
            else:n['dependency_alternatives']=alternatives
        for n in nodes.values():n['deps']=sorted(set(n['deps']))
        write(folder/'independent-workload.json',d);write(folder/'nodes.json',nodes)
        before=time.perf_counter_ns()
        with physical_build(declaration,d,folder) as compiled:
            compile_ns=time.perf_counter_ns()-before;write(folder/'compiled-ir.json',compiled.ir_document())
            before=time.perf_counter_ns();result=compiled.run();evaluate_ns=time.perf_counter_ns()-before
            assert result.success,result.summary()
            events=[asdict(e) for e in result.events];graph=result.raw[1].execution_document()
            write(folder/'events.json',events);write(folder/'graph.json',graph);write(folder/'production-summary.json',result.summary())
            graph_audit=check_graph(s,d,graph,events);physical=physical_replay(declaration,d,events,result.makespan_ns)
            numeric=check_numerical(s,d,graph,events);numeric.pop('outputs',None)
        trace=event_trace(events);tc=verify(nodes,trace);candidates=[dict(kind='actual_compiler',duration_ns=tc,events=trace)]
        for p in base.PRIORITIES:
            schedule=base.schedule(nodes,p);verify(nodes,schedule['events']);candidates.append(dict(kind='independent_priority',**schedule))
        upper=min(candidates,key=lambda p:p['duration_ns'])
        row=dict(id=declaration['id'],scenario='ffn_materialized_partition',storage=declaration['intent']['preparation'],
            T_C_ns=tc,graph_audit=graph_audit,numerical=numeric,physical_peak_audit={k:v for k,v in physical.items() if k!='samples'},
            lower_bound=strengthened_lower_bound(nodes),compile_wall_ns=compile_ns,evaluate_wall_ns=evaluate_ns,
            node_count=len(nodes),copy_nodes=sum(n['kind']=='copy' for n in nodes.values()),
            scope='Fresh current core; H8/I16/q1, CPU shard[0,4), GPU shard[4,16), GPU preprocess+merge, complete parent replacement; fixed stage prices.')
        row['certificate']=certify(row,nodes,upper,10)
        write(folder/'RESULT.json',row);rows.append(row)
        print('ffn',row['id'],row['certificate']['certified'],row['certificate']['g_interval'],flush=True)
    write(out/'FFN_RESULTS.json',dict(rows=rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('audit','solve','strengthen','deepen','closemax','fresh','ffn','report'),default='audit');parser.add_argument('--limit',type=float,default=3)
    args=parser.parse_args();campaign(mode=args.mode,limit=args.limit)
