"""Alias-aware reporting and deterministic semantic audit; no timing acquisition."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import statistics as st
from . import campaign as base
from .bridge import compile_expert_program
from .reference import verify_result
from llm_infer_model.performance.expert_policy_response import evaluate_expert_policy

def canonical(program):
    return sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def audit_semantics(folder):
    frozen=base.read(folder/'FROZEN_PREDICTIONS.json');records=[];causal=[]
    for case in frozen['cases']:
        cid=case['id'];pdir=folder/'programs'/cid;program=base.read(pdir/'program.json')
        actions=program['actions'];loads=sum(x['kind']=='LOAD' for x in actions)
        expected_loads=0 if case['policy']=='P0' or case['q']==1 and case['policy'] in ('P3','P4') else (
            16 if case['retention']=='release_after_use' else (2 if case['locality']=='hot' else (16 if case['k']==2 else 8)))
        if loads!=expected_loads:raise ValueError('independent declared load count mismatch '+cid)
        if sum(x['kind']=='EXPERT' for x in actions)!=16 or sum(x['kind']=='COMBINE' for x in actions)!=8:
            raise ValueError('independent expert/combine count mismatch')
        if case['retention']=='release_after_use':
            for i,action in enumerate(actions):
                if action['kind']=='EXPERT' and action['device']=='gpu' and actions[i+1]['kind']!='RELEASE':
                    raise ValueError('not released immediately afteruse')
        calibration=base.read(folder/'calibration'/f"q{case['q']}-k{case['k']}"/'calibration.json')
        kw=dict(config=program['config'],seed=program['seed'],steps=program['steps'],recipe=case['recipe'],retention=case['retention'],
                prices=calibration,resource_guards=program['resource_guards'])
        changed=deepcopy(kw)
        for step in changed['steps'][4:]:
            step['routes']=[[6,7] for _ in step['routes']]
        original=base.read(pdir/'compilation.json')['records'];alternative=compile_expert_program(**changed)['compilation_evidence']['records']
        left=[x for x in original if x['step']<4];right=[x for x in alternative if x['step']<4]
        if left!=right:raise ValueError('future route changes a knownprefix '+cid)
        records.append(dict(id=cid,expected_loads=expected_loads,observed_loads=loads,expert_actions=16,combine_actions=8))
        causal.append(dict(id=cid,prefix_steps=4,unchanged=True))
    # Explicit invalid actions are checked against independent model validation.
    case=next(x for x in frozen['cases'] if x['q']==1 and x['k']==2 and x['locality']=='hot' and x['policy']=='P2')
    program=base.read(folder/'programs'/case['id']/'program.json')
    calibration=base.read(folder/'calibration/q1-k2/calibration.json');badcases={}
    load=next(x for x in program['actions'] if x['kind']=='LOAD')
    firstexpert=next(x for x in program['actions'] if x['kind']=='EXPERT')
    bad=deepcopy(program);bad['actions'].insert(1,deepcopy(load));badcases['overwrite_live_slot']=bad
    bad=deepcopy(program);bad['actions']=[x for x in bad['actions'] if not(x['kind']=='LOAD' and x['expert']==firstexpert['expert'])];badcases['compute_without_load']=bad
    bad=deepcopy(program);index=next(i for i,x in enumerate(bad['actions']) if x['kind']=='EXPERT');bad['actions'].insert(index,deepcopy(firstexpert));badcases['duplicate_expert']=bad
    bad=deepcopy(program);bad['actions'].insert(0,dict(kind='COMBINE',step=0));badcases['premature_combine']=bad
    bad=deepcopy(program);bad['actions']=[x for x in bad['actions'] if not(x['kind']=='EXPERT' and x['step']==0 and x['expert']==0)];badcases['omitted_expert']=bad
    negatives=[]
    for name,bad in badcases.items():
        try:evaluate_expert_policy(bad,calibration)
        except ValueError as error:negatives.append(dict(name=name,rejected=True,message=str(error)))
        else:raise ValueError('invalidplanaccepted:'+name)
    native=deepcopy(base.read(folder/'validation'/case['id']/'process-0.json'))
    native['outputs'][0][0][0]+=program['steps'][0]['input'][0][0]
    result=verify_result(program,native,expected=base.read(folder/'programs'/case['id']/'reference.json'))
    if result['numerically_equal']:raise ValueError('extraresidualnotdetected')
    negatives.append(dict(name='incorrect_extra_residual',rejected=True,mechanism='independent elementwise numerical oracle'))
    return dict(ledger_checks=records,causal_prefix_checks=causal,negative_cases=negatives,
        statement='Positive native replay, independent declared-work counts, future-prefix invariance, and explicit invalid-plan/oracle checks are distinct evidence')

def analyze(folder):
    frozen=base.read(folder/'FROZEN_PREDICTIONS.json');raw=base.read(folder/'RESULTS.json');groups=defaultdict(list)
    for case in frozen['cases']:
        program=base.read(folder/'programs'/case['id']/'program.json')
        groups[(case['q'],case['k'],case['locality'],canonical(program))].append(case)
    rows=[];t={3:4.302652729911275,6:2.570581835636305,9:2.306004135204166}
    for (q,k,locality,chash),cases in groups.items():
        means=[]
        for case in cases:
            means.extend(st.mean(base.read(folder/'validation'/case['id']/f'process-{rep}.json')['samples_ns']) for rep in range(3))
        mean=st.mean(means);sd=st.stdev(means);pred=cases[0]['prediction']['duration_ns']
        if any(c['prediction']['duration_ns']!=pred for c in cases):raise ValueError('sameprogramdifferentprice')
        p=base.read(folder/'programs'/cases[0]['id']/'program.json');ledger=cases[0]['prediction']
        rows.append(dict(q=q,k=k,locality=locality,canonical_program_sha256=chash,policies=[c['policy'] for c in cases],
            independent_processes=len(means),process_means_ns=means,mean_ns=mean,process_std_ns=sd,
            mean_95ci_ns=[mean-t[len(means)]*sd/math.sqrt(len(means)),mean+t[len(means)]*sd/math.sqrt(len(means))],
            predicted_ns=pred,ape_percent=100*abs(pred-mean)/mean,
            loads=sum(a['kind']=='LOAD' for a in p['actions']),weight_h2d_bytes=ledger['weight_h2d_bytes'],
            total_h2d_bytes=ledger['total_h2d_bytes'],total_d2h_bytes=ledger['total_d2h_bytes'],
            peak_resident_experts=ledger['peak_resident_experts'],gpu_weight_pool_bytes=ledger['gpu_weight_logical_bytes']))
    selection=[]
    for q in (1,32):
        for k in (2,4):
            for locality in ('hot','capacity-sensitive'):
                part=[x for x in rows if (x['q'],x['k'],x['locality'])==(q,k,locality)]
                chosen=min(part,key=lambda x:(x['predicted_ns'],x['policies'][0]));best=min(part,key=lambda x:x['mean_ns'])
                lookup={policy:x for x in part for policy in x['policies']}
                selection.append(dict(q=q,k=k,locality=locality,chosen_policies=chosen['policies'],best_policies=best['policies'],
                    regret_percent=100*(chosen['mean_ns']/best['mean_ns']-1),
                    gpu_lru_vs_release_gain_percent=100*(1-lookup['P2']['mean_ns']/lookup['P1']['mean_ns']),
                    current_cost_lru_vs_cpu_gain_percent=100*(1-lookup['P4']['mean_ns']/lookup['P0']['mean_ns']),
                    current_cost_lru_vs_gpu_release_gain_percent=100*(1-lookup['P4']['mean_ns']/lookup['P1']['mean_ns']),
                    note='Identical native programs are one executable candidate. Point-mean selection is not a significance or universal optimality claim'))
    sem=audit_semantics(folder)
    output=dict(schema='moe-policy-final-analysis-1',created=base.utc(),campaign=str(folder.resolve()),
        source_results_sha256=base.digest(folder/'RESULTS.json'),frozen_predictions_sha256=base.digest(folder/'FROZEN_PREDICTIONS.json'),
        nominal_conditions=40,unique_executable_conditions=len(rows),new_independent_processes=120,
        alias_aware_mape_percent=st.mean(x['ape_percent'] for x in rows),alias_aware_max_ape_percent=max(x['ape_percent'] for x in rows),
        alias_aware_median_ape_percent=st.median(x['ape_percent'] for x in rows),label_mape_percent=raw['mape_percent'],
        label_max_ape_percent=raw['max_ape_percent'],max_abs_error=raw['max_abs_error'],max_relative_l2_error=raw['max_relative_l2_error'],
        rows=rows,selection=selection,semantic_audit=sem,
        limitations=['synthetic auditable routed F32 FFN boundary, not fullOLMoEpolicyreplay',
            'serialsynchronizedexecutor; noasynchronousoverlap orcausalprefetchperformanceclaim',
            'unknownCUDAprivateworkspace; boundedexpertweightpoolonly',
            'lifecyclecalibrationiscontextmatching, not adynamicdevicestatemodel',
            'wideprocess-levelconfidenceintervalsremain; zeropointselectionregretisnotproof ofsignificantwin',
            'current-costcollapses toCPU atq1 andGPU atq32; cannotclaimadistinctP4executionwhenitisequivalenttoP0orP2'])
    base.save(folder/'FINAL_ANALYSIS.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','selection','semantic_audit','limitations')},indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();analyze(a.folder)
