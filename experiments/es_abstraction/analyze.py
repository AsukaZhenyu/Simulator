"""Independent process-mean decision and space analysis of E/S studies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics as st
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)

def run(folder,output,rounds=0):
    folder=Path(folder);m=read(folder/'FROZEN_PREDICTIONS.json')
    labels=[x for x in m['labels'] if x['main']]
    main_ids={x['program_id'] for x in labels}
    programs={p['id']:p for p in m['programs'] if p['id'] in main_ids};n_boot=20000
    rng=np.random.default_rng(61030)
    rows=[];groups=[];all_verified=True;total_processes=0;maxabs=0
    for w in m['workloads']:
        wl=[x for x in labels if x['workload_id']==w['id']]
        wp=[p for p in programs.values() if p['workload_id']==w['id']]
        ids=[p['id'] for p in wp];index={p:i for i,p in enumerate(ids)}
        values=[];numerical=[]
        detected=[len(list((folder/'validation'/p['id']).glob('process-*.verification.json'))) for p in wp]
        if len(set(detected))!=1:raise ValueError('incomplete matched rounds:'+w['id'])
        nr=rounds or detected[0]
        if nr not in (5,10) or detected[0]<nr:raise ValueError('require 5/10 complete rounds:'+w['id'])
        for p in wp:
            ver=[read(folder/'validation'/p['id']/f'process-{r}.verification.json') for r in range(nr)]
            v=[x['process_mean_ns'] for x in ver];values.append(v);total_processes+=len(ver)
            correct=all(x['numerically_equal'] and x['compiled_action_sequence_equal'] and x['bounded_weight_slots'] for x in ver)
            all_verified &= correct;maxabs=max(maxabs,max(x['max_abs_error'] for x in ver))
            mean=st.mean(v)
            rows.append(dict(**p,mean_ns=mean,process_means_ns=v,numerically_equal=correct,
                ape_percent=100*abs(p['predicted_ns']-mean)/mean,
                max_abs_error=max(x['max_abs_error'] for x in ver),
                max_sample_cv=max(x['process_sample_cv'] for x in ver),
                native_wall_seconds=sum(x['native_wall_seconds'] for x in ver)))
        arr=np.asarray(values,dtype=float);means=arr.mean(axis=1)
        joint=min(wl,key=lambda x:(x['predicted_ns'],x['policy_id']))
        chosen=index[joint['program_id']]
        b_labels=[x for x in wl if 'B_STRONG' in x['baseline_families']]
        b_ids=sorted({index[x['program_id']] for x in b_labels})
        best=int(np.argmin(means));best_b=min(b_ids,key=lambda i:means[i])
        samples=rng.integers(0,nr,size=(n_boot,nr))
        draws=arr[:,samples].mean(axis=2)
        gain=100*(1-means[chosen]/means[best_b]);regret=100*(means[chosen]/means[best]-1)
        gain_draws=100*(1-draws[chosen]/draws[b_ids].min(axis=0))
        regret_draws=100*(draws[chosen]/draws.min(axis=0)-1)
        ci=np.quantile(gain_draws,[.025,.975]).tolist()
        # Conservative familywise interval; separately label it from unadjusted95%.
        family_alpha=.05/len(m['workloads'])
        ci_family=np.quantile(gain_draws,[family_alpha/2,1-family_alpha/2]).tolist()
        # Intersection-union superiority: beat every strong baseline. Paired
        # exact sign-flip p-values use the matched round as independent unit.
        signs=np.asarray([[1 if (s>>j)&1 else -1 for j in range(nr)] for s in range(2**nr)])
        ps=[]
        for bi in b_ids:
            d=arr[bi]-arr[chosen]
            ps.append(float(np.mean((signs@d/nr)>=d.mean()-1e-8)))
        raw_p=max(ps)
        extra=[p for p in wp if p['id'] not in {x['program_id'] for x in b_labels}]
        g=dict(workload_id=w['id'],geometry=w['geometry'],k=w['k'],pattern=w['pattern'],seed=w['seed'],
            group=w['group'],unique_programs=len(wp),nominal_labels=len(wl),additional_joint_programs=len(extra),
            joint_choice=joint['policy_id'],chosen_program=ids[chosen],chosen_aliases=programs[ids[chosen]]['policy_ids'],
            measured_best=programs[ids[best]]['policy_ids'],measured_best_program=ids[best],
            measured_strong_best=programs[ids[best_b]]['policy_ids'],strong_best_program=ids[best_b],
            choice_mean_ns=float(means[chosen]),finite_pool_best_ns=float(means[best]),strong_best_ns=float(means[best_b]),
            selection_regret_percent=regret,native_gain_vs_strong_percent=gain,
            gain_95ci_percent=ci,gain_familywise_95ci_percent=ci_family,
            regret_95ci_percent=np.quantile(regret_draws,[.025,.975]).tolist(),
            gain_ci_width_percent=ci[1]-ci[0],raw_superiority_p=raw_p,rounds=nr,
            all_candidate_checks_passed=all(x['numerically_equal'] for x in rows if x['workload_id']==w['id']),
            conditional_extra_rounds_required=nr==5 and ci[1]-ci[0]>10)
        families={}
        joint_pred=joint['predicted_ns']
        for family in ('B_E','B_S','B_FIXED_HYBRID','B_RESIDENT','B_STRONG'):
            subset=[x for x in wl if family in x['baseline_families']]
            bp=min(subset,key=lambda x:(x['predicted_ns'],x['policy_id']))
            families[family]=dict(predicted_choice=bp['policy_id'],predicted_space_gain_percent=100*(1-joint_pred/bp['predicted_ns']),
                native_choice_mean_ns=float(means[index[bp['program_id']]]),
                unique_programs=len({x['program_id'] for x in subset}))
        g['baseline_families']=families;groups.append(g)
    order=sorted(range(len(groups)),key=lambda i:groups[i]['raw_superiority_p'])
    prev=0
    for rank,i in enumerate(order):
        prev=max(prev,min(1,(len(groups)-rank)*groups[i]['raw_superiority_p']))
        groups[i]['holm_adjusted_p']=prev
        groups[i]['meaningful_positive_gain_target_met']=(groups[i]['native_gain_vs_strong_percent']>=5 and
            groups[i]['gain_familywise_95ci_percent'][0]>0 and groups[i]['all_candidate_checks_passed'])
    regrets=[x['selection_regret_percent'] for x in groups]
    result=dict(schema='es-abstraction-native-analysis-1',rounds_by_workload={x['workload_id']:x['rounds'] for x in groups},workloads=len(groups),
        nominal_labels=len(labels),unique_programs=len(rows),independent_processes=total_processes,
        all_numerical_and_physical_checks_passed=all_verified,max_abs_error=maxabs,
        mean_ape_percent=st.mean(x['ape_percent'] for x in rows),max_ape_percent=max(x['ape_percent'] for x in rows),
        median_regret_percent=float(np.median(regrets)),p90_regret_percent=float(np.quantile(regrets,.9)),
        worst_regret_percent=max(regrets),additional_joint_programs=sum(x['additional_joint_programs'] for x in groups),
        positive_gain_workloads=sum(x['meaningful_positive_gain_target_met'] for x in groups),
        extra_round_workloads=[x['workload_id'] for x in groups if x['conditional_extra_rounds_required']],
        compile_seconds=sum(x['compile_ns'] for x in labels)/1e9,
        evaluate_seconds=sum(x['evaluate_ns'] for x in labels)/1e9,
        native_process_wall_seconds=sum(x['native_wall_seconds'] for x in rows),
        inference='Matched-round process bootstrap,20000 draws; nominal Bonferroni-adjusted95% intervals, no finite-sample coverage guarantee with 5/10 rounds; Holm paired exact sign-flip p diagnostic only (five-round resolution insufficient for24 comparisons). Correctness failures block performance claims. Not an official DAC acceptance score.',
        rows=rows,selection=groups)
    save(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','selection')},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=0,help='0 auto-detect5/10 separately per workload');a=p.parse_args();run(a.campaign,a.out,a.rounds)
