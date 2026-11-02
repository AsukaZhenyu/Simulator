"""Profile-only estimation for v9. All response equations live in modeling.

Unstable baselines or unidentifiable recovery constants remain failures. This
module preserves every trial and never uses heldout observations to initialize,
select, fit or freeze a provider.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import least_squares, nnls

from . import h2d_bare_v9 as campaign
from llm_infer_model.performance.transfer_response import TransferResponse
from llm_infer_model.performance.transfer_recovery import AffineResponseImpact, TransferRecoveryScope, SizeAwareRecoveryResponse

read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
KINDS=('static_affine','zero_initial_recovery','prepared_initial_recovery')
MIB=1048576


def model_for(kind,theta,scope,memory,evidence):
    require(kind in KINDS,'unknown named model')
    base=TransferResponse('dram->vram',float(theta[0])*1000,float(theta[1])*1000/MIB,float(theta[1])*1000/MIB,
        None,131072,2097152,memory,campaign.BOUNDARY,tuple(evidence),False)
    if kind=='static_affine':return base
    if kind=='zero_initial_recovery':
        initial=(0.,0.);capacity=theta[2:4];taui,taur=theta[4:6]
    else:
        initial=theta[2:4];capacity=(theta[2]+theta[4],theta[3]+theta[5]);taui,taur=theta[6:8]
    return SizeAwareRecoveryResponse(TransferRecoveryScope(scope['context_sha256'],scope['preparation_sha256'],'dram->vram',memory,20000000),base,
        AffineResponseImpact(float(capacity[0])*1000,float(capacity[1])*1000/MIB,tuple(evidence)),
        AffineResponseImpact(float(initial[0])*1000,float(initial[1])*1000/MIB,tuple(evidence)),
        float(taui)*1000000,float(taur)*1000,tuple(evidence),False)


def curve(kind,theta,scope,memory,evidence,sizes,idles,*,continuous=True):
    model=model_for(kind,theta,scope,memory,evidence);rows=[]
    for size in sizes:
        for idle in idles:
            if kind=='static_affine':values=[model.predict_ns(size,require_qualified=False)]*16
            elif continuous:
                values=model.continuous_burst(size,idle*1000000,16,context_sha256=scope['context_sha256'],preparation_sha256=scope['preparation_sha256'])
            else:
                values=[r['predicted_ns'] for r in model.predict_burst(size,idle*1000000,16,context_sha256=scope['context_sha256'],preparation_sha256=scope['preparation_sha256'],require_qualified=False)['rows']]
            rows.append(values)
    return np.asarray(rows,dtype=float).reshape(len(sizes),len(idles),16)


def native_records(out,split):
    m=read(out/'manifest.json');executions=read(out/f'{split}-executions.json');require(len(executions)==6 and all(r['status']=='completed' for r in executions),'all six retained processes required')
    byid={x['job_id']:x for x in executions};records=[]
    for job in (j for j in m['jobs'] if j['split']==split):
        e=byid[job['id']];require(sha(e['native']['path'])==e['native']['sha256'],'retained native changed');n=read(e['native']['path'])
        campaign.validate_native(n,job['request'],m['programs'][split],e['process_id'])
        records.append(dict(job=job,native=e['native'],value=n))
    return m,records


def observations(records,memory,split):
    selected=[r for r in records if r['job']['memory_mode']==memory];require(len(selected)==3,'three fresh contexts required')
    matrices=[]
    for r in selected:
        groups=defaultdict(list)
        for b in r['value']['bursts']:groups[b['bytes'],b['requested_idle_ms']].append(b)
        matrix=[]
        for size in campaign.SIZES[split]:
            row=[]
            for idle in campaign.IDLES[split]:
                group=groups[size,idle];require(len(group)==12 and {b['trial'] for b in group}==set(range(12)),'complete repeated condition required')
                row.append([statistics.mean(b['copies'][i]['wall_ns'] for b in group) for i in range(16)])
            matrix.append(row)
        matrices.append(matrix)
    data=np.asarray(matrices,dtype=float)
    require(data.shape==(3,len(campaign.SIZES[split]),len(campaign.IDLES[split]),16) and np.all(np.isfinite(data)) and np.all(data>0),'positive complete observations required')
    return selected,data


def baseline_support(records):
    early_late=[];cross=[]
    for memory in ('registered','pageable'):
        bysize=defaultdict(list)
        for r in (x for x in records if x['job']['memory_mode']==memory):
            for size in campaign.SIZES['profile']:
                groups=[b for b in r['value']['bursts'] if b['bytes']==size and b['requested_idle_ms']==0]
                suffix=lambda group:statistics.mean(statistics.mean(c['wall_ns'] for c in b['copies'][1:]) for b in group)
                early=suffix([b for b in groups if b['trial']<6]);late=suffix([b for b in groups if b['trial']>=6]);average=suffix(groups)
                delta=abs(late/early-1);early_late.append(dict(job_id=r['job']['id'],memory_mode=memory,bytes=size,early_suffix_ns=early,late_suffix_ns=late,relative_difference=delta,passed=delta<=.10));bysize[size].append(average)
        for size,values in bysize.items():
            delta=(max(values)-min(values))/statistics.mean(values);cross.append(dict(memory_mode=memory,bytes=size,process_suffix_means_ns=values,relative_range=delta,passed=delta<=.10))
    return dict(early_late=early_late,fresh_processes=cross,passed=all(r['passed'] for r in early_late+cross),raw_samples_removed=0,threshold=.10)


def fit_hypothesis(kind,scope,memory,evidence,data):
    sizes,idles=campaign.SIZES['profile'],campaign.IDLES['profile'];target=data/1000
    if kind=='static_affine':
        design=np.asarray([[1,size/MIB] for size in sizes for _idle in idles for _ordinal in range(16)]*3)
        theta,residual=nnls(design,target.reshape(-1));singular=np.linalg.svd(design,compute_uv=False)
        return dict(theta=theta.tolist(),sum_squared_us=float(residual**2),rank=int(np.linalg.matrix_rank(design)),free_parameters=2,
            scaled_jacobian_condition=float(singular[0]/singular[-1]),equivalent_multistart_parameter_relative_spread=0.,tau_at_bound=False,
            converged=True,identifiable=True,starts=1,evaluations=1,source='all per-coordinate profile labels')
    tail=np.mean(data[:,:,:,8:],axis=(0,2,3))/1000
    design=np.asarray([[1,size/MIB] for size in sizes]);base,_=nnls(design,tail);base=np.maximum(base,.01)
    first=np.mean(data[:,:,3,0],axis=0)/1000
    cap,_=nnls(design,np.maximum(first-design@base,1.));cap=np.maximum(cap,.01)
    initfirst=np.mean(data[:,:,0,0],axis=0)/1000
    initial,_=nnls(design,np.maximum(initfirst-design@base,.1));initial=np.maximum(initial,.001)
    ceiling=max(1000.,float(np.max(data)/1000)*8)
    if kind=='zero_initial_recovery':
        low=[.001,0.,.001,0.,.001,.001];high=[ceiling]*4+[100.,10000.]
        starts=[[base[0],base[1],cap[0],cap[1],ti,tr] for ti in (.3,3.,15.) for tr in (50.,500.)]
    else:
        remaining=np.maximum(cap-initial,.01)
        low=[.001,0.,0.,0.,.001,0.,.001,.001];high=[ceiling]*6+[100.,10000.]
        starts=[[base[0],base[1],initial[0],initial[1],remaining[0],remaining[1],ti,tr] for ti in (.3,3.,15.) for tr in (50.,500.)]
    def residual(theta):
        prediction=curve(kind,theta,scope,memory,evidence,sizes,idles)/1000
        return (prediction[np.newaxis,...]-target).reshape(-1)
    fits=[least_squares(residual,np.clip(start,np.asarray(low)+1e-9,np.asarray(high)-1e-9),bounds=(low,high),loss='linear',
                        max_nfev=600,ftol=1e-8,xtol=1e-8,gtol=1e-7,x_scale='jac') for start in starts]
    best=min(fits,key=lambda r:float(r.fun@r.fun));cost=float(best.fun@best.fun)
    # Column normalization tests sensitivity geometry independently of chosen ns/us units.
    norms=np.linalg.norm(best.jac,axis=0);scaled=best.jac/np.maximum(norms,1e-300)
    singular=np.linalg.svd(scaled,compute_uv=False);rank=int(np.linalg.matrix_rank(scaled));condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    equivalent=[r.x for r in fits if float(r.fun@r.fun)<=cost*1.001+1e-8]
    spread=float(np.max(np.ptp(np.asarray(equivalent),axis=0)/np.maximum(np.abs(best.x),1.)))
    tau_bound=any((best.x[i]-low[i])/(high[i]-low[i])<.001 or (high[i]-best.x[i])/(high[i]-low[i])<.001 for i in (-2,-1))
    supported=bool(best.success and rank==len(best.x) and condition is not None and condition<=1e6 and spread<=.25 and not tau_bound)
    return dict(theta=best.x.tolist(),sum_squared_us=cost,rank=rank,free_parameters=len(best.x),scaled_jacobian_condition=condition,
        equivalent_multistart_parameter_relative_spread=spread,equivalent_multistart_spread_definition='max_j ptp(equivalent_theta_j)/max(abs(best_theta_j),1 in stated parameter units)',equivalent_solution_count=len(equivalent),tau_at_bound=tau_bound,
        converged=bool(best.success),identifiable=supported,starts=len(starts),evaluations=sum(r.nfev for r in fits),
        optimizer_message=str(best.message),parameter_units='fixed_us, byte_slope_us_per_MiB; final tau_idle_ms/tau_active_us',
        no_outlier_removal=True,fit_uses_full_LLM_target=False)


def fit_and_freeze(out):
    out=Path(out).resolve();require(not (out/'heldout-started.json').exists(),'heldout already acquired; no refit')
    require(not (out/'fit.json').exists() and not (out/'frozen-barrier.json').exists(),'fit is single frozen operation')
    m,records=native_records(out,'profile');scope=m['scope_contracts'];gate=baseline_support(records);write(out/'baseline-support.json',gate)
    models=[];predictions=[]
    for memory in ('registered','pageable'):
        selected,data=observations(records,memory,'profile');evidence=tuple(x['native']['path']+'#sha256='+x['native']['sha256'] for x in selected)
        memory_gate=all(x['passed'] for x in gate['early_late']+gate['fresh_processes'] if x['memory_mode']==memory)
        for kind in KINDS:
            fit=fit_hypothesis(kind,scope,memory,evidence,data)
            prediction=curve(kind,fit['theta'],scope,memory,evidence,campaign.SIZES['profile'],campaign.IDLES['profile'])
            fit.update(memory_mode=memory,kind=kind,evidence=evidence,profile_baseline_supported=memory_gate,
                profile_coordinate_mean_MAPE=float(np.mean(np.abs(prediction[np.newaxis,...]-data)/data)),
                parameters_supported=bool(memory_gate and fit['identifiable']),independently_qualified=False)
            models.append(fit)
            if fit['parameters_supported']:
                held=curve(kind,fit['theta'],scope,memory,evidence,campaign.SIZES['heldout'],campaign.IDLES['heldout'],continuous=False)
                for i,size in enumerate(campaign.SIZES['heldout']):
                    for j,idle in enumerate(campaign.IDLES['heldout']):
                        predictions.append(dict(memory_mode=memory,kind=kind,bytes=size,requested_idle_ms=idle,copy_predictions_ns=held[i,j].tolist(),
                            first_prediction_ns=float(held[i,j,0]),burst_prediction_ns=float(sum(held[i,j])),uses_observed_idle_overshoot=False,heldout_observation_used=False))
    result=dict(schema='dynamic-h2d-bare-fit-9',models=models,scope=scope,profile_native_receipts=[r['native'] for r in records],
        raw_samples_removed=0,whole_LLM_target_used=False,heldout_observations_read=False,baseline_support=receipt(out/'baseline-support.json'),
        supported_models=sum(x['parameters_supported'] for x in models),declared_models=6,hardware_accuracy_qualified=False)
    write(out/'fit.json',result);write(out/'heldout-predictions.json',dict(schema='dynamic-h2d-bare-predictions-9',rows=predictions,unsupported_models=[dict(memory_mode=x['memory_mode'],kind=x['kind']) for x in models if not x['parameters_supported']]))
    supported=gate['passed'] and all(any(x['parameters_supported'] for x in models if x['memory_mode']==memory) for memory in ('registered','pageable'))
    barrier=dict(schema='dynamic-h2d-bare-freeze-9',manifest_sha256=sha(out/'manifest.json'),profile_gate_passed=supported,
        parameters_frozen=True,heldout_predictions_frozen=True,heldout_data_used=False,fit_uses_whole_target_total=False,
        files={name:sha(out/name) for name in ('fit.json','baseline-support.json','heldout-predictions.json')})
    write(out/'frozen-barrier.json',barrier)
    return dict(fit=receipt(out/'fit.json'),barrier=receipt(out/'frozen-barrier.json'),profile_gate_passed=supported,supported_models=result['supported_models'],heldout_permitted=supported)


def evaluate(out):
    out=Path(out).resolve();m,records=native_records(out,'heldout');barrier=read(out/'frozen-barrier.json')
    require(barrier['profile_gate_passed'] and barrier['manifest_sha256']==sha(out/'manifest.json'),'unsupported/mismatched frozen profile')
    for name,value in barrier['files'].items():require(sha(out/name)==value,'frozen predictor changed')
    predictions=read(out/'heldout-predictions.json');rows=[]
    for p in predictions['rows']:
        for r in (x for x in records if x['job']['memory_mode']==p['memory_mode']):
            group=[b for b in r['value']['bursts'] if b['bytes']==p['bytes'] and b['requested_idle_ms']==p['requested_idle_ms']]
            first=[b['copies'][0]['wall_ns'] for b in group];burst=[b['burst_wall_ns'] for b in group]
            copy_sum=[sum(c['wall_ns'] for c in b['copies']) for b in group]
            meanfirst,meanburst=statistics.mean(first),statistics.mean(burst);cvfirst=statistics.pstdev(first)/meanfirst;cvburst=statistics.pstdev(burst)/meanburst
            ef,eb=abs(p['first_prediction_ns']/meanfirst-1),abs(p['burst_prediction_ns']/meanburst-1)
            rows.append(dict(job_id=r['job']['id'],memory_mode=p['memory_mode'],kind=p['kind'],bytes=p['bytes'],requested_idle_ms=p['requested_idle_ms'],
                first_predicted_ns=p['first_prediction_ns'],first_measured_mean_ns=meanfirst,first_APE=ef,first_measured_CV=cvfirst,
                burst_predicted_ns=p['burst_prediction_ns'],burst_measured_mean_ns=meanburst,burst_APE=eb,burst_measured_CV=cvburst,
                copy_response_sum_measured_mean_ns=statistics.mean(copy_sum),loop_gap_measured_mean_ns=meanburst-statistics.mean(copy_sum),
                forecast_scope='sum of completed COPY responses; actual acceptance includes outer-loop gaps, no observed residual added',
                mean_accuracy_passed=ef<=.20 and eb<=.20,raw_stability_passed=cvfirst<=.10 and cvburst<=.10,
                strict_passed=ef<=.20 and eb<=.20 and cvfirst<=.10 and cvburst<=.10))
    result=dict(schema='dynamic-h2d-bare-independent-evaluation-9',rows=rows,declared_all_provider_cells=72,
        supported_prediction_cells=len(rows),unsupported_provider_cells=72-len(rows),passed=sum(r['strict_passed'] for r in rows),
        mean_accuracy_passed=sum(r['mean_accuracy_passed'] for r in rows),no_refit=True,raw_trials_removed=0,
        hardware_scope='this single original GGML COPY/preparation/runtime and host-memory category, unseen sizes/idles/seeds',
        whole_LLM_accuracy_validated=False,baseline_or_heldout_gate_weakened=False)
    write(out/'independent-evaluation.json',result);return dict(output=receipt(out/'independent-evaluation.json'),strict_passed=result['passed'],supported=len(rows),declared=72)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fit','evaluate'));p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(fit_and_freeze(a.out) if a.command=='fit' else evaluate(a.out),ensure_ascii=False))

if __name__=='__main__':main()
