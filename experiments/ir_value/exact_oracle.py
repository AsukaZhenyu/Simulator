"""Exact independent non-preemptive scheduler for the tiny Dense reference DAG.

Disjunctive MILP, not compiler action replay. Requires SciPy/HiGHS. Results are
exact only for the declared independent inventory, fixed nonpreemptive kernels,
shared PCIe resource, persistent copies and unconstrained-within1MiB capacity.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import hashlib
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix

def solve(nodes,time_limit=30):
    ids=list(nodes);lookup={k:i for i,k in enumerate(ids)};n=len(ids)
    machines=defaultdict(list)
    for k,row in nodes.items():machines[row['machine']].append(k)
    pairs=[(a,b) for group in machines.values() for i,a in enumerate(group) for b in group[i+1:]
           if nodes[a]['duration'] and nodes[b]['duration']]
    horizon=sum(row['duration'] for row in nodes.values())+max(row['release'] for row in nodes.values())+1
    rows=[];lower=[];upper=[]
    def add(items,lo=-np.inf,hi=np.inf):rows.append(items);lower.append(lo);upper.append(hi)
    for k,row in nodes.items():
        i=lookup[k]
        add({n:1,i:-1},lo=row['duration'])
        for parent in row['deps']:add({i:1,lookup[parent]:-1},lo=nodes[parent]['duration'])
    for j,(a,b) in enumerate(pairs):
        ai,bi=lookup[a],lookup[b];binary=n+1+j
        add({ai:1,bi:-1,binary:horizon},hi=horizon-nodes[a]['duration'])
        add({bi:1,ai:-1,binary:-horizon},hi=-nodes[b]['duration'])
    matrix=lil_matrix((len(rows),n+1+len(pairs)),dtype=float)
    for j,row in enumerate(rows):
        for i,v in row.items():matrix[j,i]=v
    c=np.zeros(n+1+len(pairs));c[n]=1
    lb=np.array([nodes[k]['release'] for k in ids]+[0]+[0]*len(pairs),dtype=float)
    ub=np.array([horizon]*(n+1)+[1]*len(pairs),dtype=float)
    integrality=np.array([0]*(n+1)+[1]*len(pairs))
    start=time.time()
    result=milp(c,integrality=integrality,bounds=Bounds(lb,ub),
        constraints=LinearConstraint(matrix.tocsr(),np.array(lower),np.array(upper)),
        options=dict(time_limit=time_limit,mip_rel_gap=0))
    doc=dict(status=int(result.status),message=result.message,elapsed_seconds=time.time()-start,
        variables=len(c),binary_disjunctions=len(pairs),constraints=len(rows),
        objective_ns=None if result.fun is None else float(result.fun),
        mip_gap=getattr(result,'mip_gap',None),dual_bound_ns=getattr(result,'mip_dual_bound',None))
    if result.status!=0:return doc
    starts={k:int(round(result.x[lookup[k]])) for k in ids};trace=[]
    for k,row in nodes.items():
        assert starts[k]>=row['release']
        assert all(starts[k]>=starts[parent]+nodes[parent]['duration'] for parent in row['deps'])
        trace.append(dict(id=k,machine=row['machine'],start_ns=starts[k],end_ns=starts[k]+row['duration']))
    for a,b in pairs:
        assert starts[a]+nodes[a]['duration']<=starts[b] or starts[b]+nodes[b]['duration']<=starts[a]
    makespan=max(r['end_ns'] for r in trace)
    assert abs(makespan-result.fun)<1e-5
    doc.update(duration_ns=makespan,events=trace,integer_tick_constraints_rechecked=True)
    return doc

def campaign(directory):
    directory=Path(directory);out=directory/'exact-dense';out.mkdir(exist_ok=True)
    if (out/'PROTOCOL.json').exists():raise ValueError('Exact oracle output already exists')
    protocol=dict(scope='All8CPU/GPU placements ×demand/spare under2declaredlinkprices; independent Dense DAG only. Exact nonpreemptive ordering over fixed independent copy inventory, persistent replicas and1MiBcapacity. No global completeness outside these contracts.',
        solver='SciPy milp / HiGHS',mip_relative_gap=0,time_limit_seconds=30,
        frozen_before_solves=True)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8',newline='\n')
    results=[]
    for path in sorted(directory.glob('dense-*/*/independent-reference.json')):
        source=json.loads(path.read_text(encoding='utf-8'));row=solve(source['nodes'])
        row['id']=path.parent.name;row['scenario_id']=path.parent.parent.name
        results.append(row)
        (out/f"{row['scenario_id']}-{row['id']}.json").write_text(json.dumps(row,indent=2),encoding='utf-8',newline='\n')
        print(row['scenario_id'],row['id'],row['status'],row.get('duration_ns'),flush=True)
    summary=[]
    for regime in sorted({r['scenario_id'] for r in results}):
        group=[r for r in results if r['scenario_id']==regime]
        solved=[r for r in group if r['status']==0]
        summary.append(dict(scenario_id=regime,conditions=len(group),certified_optima=len(solved),
            best_ns=min(r['duration_ns'] for r in solved) if solved else None))
    doc=dict(protocol=protocol,rows=results,summary=summary)
    (out/'REPORT.json').write_text(json.dumps(doc,indent=2),encoding='utf-8',newline='\n')
    return doc

def certify(directory):
    """Certify global best over32Dense points by lower/upper bounds.

    A precedence longest path plus per-resource total work is a valid lower
    bound for every non-preemptive ordering, including orders not in16ties.
    A feasible independently generated schedule supplies the upper bound.
    Competitive unresolved cases may invokeMILP. Bound-dominated cases need
    not each have an exact individual objective to certify the global best.
    """
    directory=Path(directory);out=directory/'exact-dense-certificates';out.mkdir(exist_ok=True)
    if (out/'PROTOCOL.json').exists():raise ValueError('certificate output alreadyexists')
    inputs=sorted(directory.glob('dense-*/*/independent-reference.json'))
    protocol=dict(method='Independent precedence/resource lower bounds; feasible16tie schedule upper bounds; disjunctiveMILP only if lowerbound is competitive.',
        scope='All8Dense placements ×demand/spare ×2analyticallinkprices; fixed independent copy inventory,persistentreplicas,CPU/GPU/sharedPCIe,1MiBcapacity.',
        inputs={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),milp_time_limit_seconds=10)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8',newline='\n')
    groups=defaultdict(list)
    for path in inputs:groups[path.parent.parent.name].append((path,json.loads(path.read_text(encoding='utf-8'))))
    summaries=[];allrows=[]
    for regime,items in groups.items():
        witness_path,witness=min(items,key=lambda item:item[1]['best']['duration_ns'])
        upper=witness['best']['duration_ns'];certificate_rows=[]
        for path,source in items:
            nodes=source['nodes'];memo={}
            def finish(k):
                if k not in memo:memo[k]=max([nodes[k]['release'],*[finish(parent) for parent in nodes[k]['deps']]])+nodes[k]['duration']
                return memo[k]
            work=defaultdict(int)
            for row in nodes.values():work[row['machine']]+=row['duration']
            bound=max(max(finish(k) for k in nodes),max(work.values()))
            row=dict(id=path.parent.name,scenario_id=regime,precedence_resource_lower_bound_ns=bound,
                best_feasible_reference_ns=source['best']['duration_ns'],status='bound_dominated')
            if bound<upper:
                result=solve(nodes,time_limit=10);row['milp']=result
                if result['status']==0:row.update(status='milp_optimal',certified_lower_bound_ns=result['duration_ns'])
                elif result.get('dual_bound_ns') is not None:row.update(status='milp_bounded',certified_lower_bound_ns=result['dual_bound_ns'])
                else:row.update(status='uncertified',certified_lower_bound_ns=bound)
            else:row['certified_lower_bound_ns']=bound
            certificate_rows.append(row)
        lower=min(row['certified_lower_bound_ns'] for row in certificate_rows)
        # Recheck the feasible witness schedule independently atinteger ticks.
        trace={r['id']:r for r in witness['best']['events']};nodes=witness['nodes']
        for identity,node in nodes.items():
            event=trace[identity]
            assert event['end_ns']-event['start_ns']==node['duration']
            assert event['start_ns']>=node['release']
            assert all(trace[parent]['end_ns']<=event['start_ns'] for parent in node['deps'])
        for a,ra in trace.items():
            for b,rb in trace.items():
                if a<b and ra['machine']==rb['machine']:
                    assert ra['end_ns']<=rb['start_ns'] or rb['end_ns']<=ra['start_ns']
        summary=dict(scenario_id=regime,conditions=len(items),lower_bound_ns=lower,upper_bound_ns=upper,
            global_optimum_certified=lower>=upper,witness=str(witness_path.relative_to(directory)))
        summaries.append(summary);allrows.extend(certificate_rows)
        print(summary,flush=True)
    doc=dict(protocol=protocol,summary=summaries,rows=allrows)
    (out/'REPORT.json').write_text(json.dumps(doc,indent=2),encoding='utf-8',newline='\n')
    return doc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--certify',action='store_true');args=p.parse_args()
    (certify if args.certify else campaign)(args.directory)
