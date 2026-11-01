"""Select bounded named-parameter idle hypotheses using PROFILE labels only.

All observations and between-process spread survive. The primary fit uses
ordinary least squares with equal process/condition/ordinal weight; no unstable
process, outlier or nonmonotonic observation is removed. No DNN or target total
is involved. This command must finish before any heldout child starts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import least_squares

from llm_infer_model.performance.idle_recovery import (ConstantIdleResponse,CurrentIdleResponse,
    IdleRecoveryParameters,IdleRecoveryResponse,IdleRecoveryScope,ReadinessDebt)
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from profiling.dynamic_validation import idle_campaign as campaign
from profiling.dynamic_validation.baseline_audit import inspect_campaign as audit_baseline

ROOT=Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require


def model_for(kind,scope,theta,evidence):
    """All response calculation remains in modeling, including fit predictions."""
    if kind=='static':return ConstantIdleResponse(scope,float(theta[0])*1000,evidence),None
    if kind=='current_idle':return CurrentIdleResponse(scope,float(theta[0])*1000,float(theta[1])*1000,
        float(theta[2])*1000000,evidence),None
    require(kind=='stateful','unknown named hypothesis')
    return IdleRecoveryResponse(scope,IdleRecoveryParameters(float(theta[0])*1000,float(theta[1])*1000,
        float(theta[2])*1000000,float(theta[3])*1000,evidence)),ReadinessDebt(float(theta[4]))


def curve(kind,scope,theta,idles,evidence,*,continuous):
    model,initial=model_for(kind,scope,theta,evidence);result=[]
    for idle in idles:
        if continuous:
            values=model.continuous_burst(idle*1000000,16,**({'initial_state':initial} if initial is not None else {}))
        else:
            record=model.predict_burst(idle*1000000,16,**({'initial_state':initial} if initial is not None else {}))
            values=[r['predicted_ns'] for r in record['rows']]
        result.extend(values)
    return np.asarray(result,dtype=float)


def observations(records,memory):
    selected=[r for r in records if r['job']['memory_mode']==memory]
    require(len(selected)==3,'three retained profile processes per host-memory mode required')
    matrices=[]
    for r in selected:
        conditions={c['requested_idle_ms']:c for c in r['conditions']}
        require(set(conditions)==set(campaign.IDLES['profile']),'profile idle inventory changed')
        matrices.append([conditions[idle]['ordinal_mean_ns'] for idle in campaign.IDLES['profile']])
    a=np.asarray(matrices,dtype=float)
    require(a.shape==(3,4,16) and np.all(np.isfinite(a)) and np.all(a>0),'complete positive per-action observations required')
    return selected,a


def fit_hypothesis(kind,scope,data,evidence):
    idles=campaign.IDLES['profile'];target=data.reshape(3,-1)/1000 # microseconds for numerical conditioning
    base=float(np.mean(data[:,:,8:])/1000)
    extra=max(1.,float(np.mean(data[:,3,0])/1000-base))
    ceiling=max(1000.,float(np.max(data)/1000)*4)
    if kind=='static':
        theta=np.asarray([float(np.mean(data)/1000)])
        prediction=curve(kind,scope,theta,idles,evidence,continuous=True).reshape(1,-1)/1000
        return dict(theta=theta.tolist(),cost=float(np.sum((prediction-target)**2)),rank=1,
            converged=True,evaluations=0,starts=1,free_parameters=1)
    bounds=([1.,0.,.01],[ceiling,ceiling,1000.])
    starts=[[base,extra,tau] for tau in (.25,2.,20.)]
    if kind=='stateful':
        h0=min(.9,max(.05,(float(np.mean(data[:,0,0])/1000)-base)/max(extra,1)))
        bounds=([1.,0.,.01,1.,0.],[ceiling,ceiling,1000.,10000.,1.])
        starts=[[base,extra,taui,taur,h0] for taui in (.25,2.,20.) for taur in (25.,250.,1000.)]
    def residual(theta):
        prediction=curve(kind,scope,theta,idles,evidence,continuous=True).reshape(1,-1)/1000
        return (prediction-target).ravel()
    fits=[]
    for start in starts:
        result=least_squares(residual,start,bounds=bounds,method='trf',loss='linear',max_nfev=1200,
                             ftol=1e-9,xtol=1e-9,gtol=1e-8)
        fits.append(result)
    best=min(fits,key=lambda r:float(np.sum(r.fun**2)))
    rank=int(np.linalg.matrix_rank(best.jac))
    return dict(theta=best.x.tolist(),cost=float(np.sum(best.fun**2)),rank=rank,converged=bool(best.success),
        evaluations=int(sum(r.nfev for r in fits)),starts=len(starts),free_parameters=len(best.x),
        all_parameters_locally_identified=rank==len(best.x),optimizer_message=str(best.message))


def fit_and_freeze(out):
    out=Path(out).resolve();manifest=read(out/'manifest.json');campaign.check(manifest)
    require(not (out/'heldout-started.json').exists() and not any((out/'jobs'/j['id']).exists() for j in manifest['jobs'] if j['split']=='heldout'),
            'fit after heldout acquisition rejected')
    require(not (out/'frozen-barrier.json').exists(),'frozen response already exists')
    observations_path=out/'profile-observations.json'
    if not observations_path.exists():campaign.summarize(out,'profile')
    observation=read(observations_path);require(observation['completed']==observation['declared']==6,'all six profiles must remain')
    # A single steady-response parameter cannot silently pool chronological
    # baseline shifts. This is a declared support gate, not a measured device
    # constant, a filter, or a replacement for independent prediction testing.
    gate_path=out/'baseline-support.json'
    require(not gate_path.exists(),'baseline support already inspected; retain it and start a new cohort after repair')
    gate=audit_baseline(out,split='profile',role='calibration_support',
        tail_first_ordinal=1,split_trial_count=8,relative_limit=.10)
    write(gate_path,gate)
    if not gate['supports_one_fixed_baseline']:
        write(out/'fit-refusal.json',dict(schema='dynamic-idle-fit-refusal-1',
            reason='zero-idle baseline changes exceed the declared ten-percent support gate',
            baseline_support=receipt(gate_path),samples_removed=0,performance_parameters_fitted=False,
            predictions_frozen=False,heldout_observations_read=False,
            next_step='identify measurable state or preparation differences; preserve slow trials'))
        raise ValueError('fixed-baseline hypothesis unsupported; see baseline-support.json and fit-refusal.json')
    profile_program=read(manifest['programs']['profile']['program']['path'])
    context=dict(hardware=manifest['hardware'],runtime=manifest['runtime'],binary_sha256=manifest['binary']['sha256'],
        target=semantic_action(profile_program,manifest['programs']['profile']['target_action_index']),
        prime_semantics_digest=digest([semantic_action(profile_program,i) for i in range(len(profile_program['actions']))]),
        preparation=manifest['protocol']['initial_state'],boundary=manifest['protocol']['target'],
        warmup=dict(iterations=20,min_ms=500),thread_mask=4,process_mask=4294967295,
        fitting_coordinate_excludes=['program ID','strategy name','layer ID','seed','complete Dense target timing'])
    profiles={};predictions={kind:[] for kind in ('static','current_idle','stateful')};calibration=[]
    for memory in ('pageable','registered'):
        selected,data=observations(observation['records'],memory)
        for row in selected:require(sha(row['native']['path'])==row['native']['sha256'],'retained profile source changed')
        evidence=tuple(row['native']['path']+'#sha256='+row['native']['sha256'] for row in selected)
        scope=IdleRecoveryScope(digest(dict(context,memory_mode=memory)),'dram->vram',2097152,memory,20000000)
        profile={}
        for kind in predictions:
            fitted=fit_hypothesis(kind,scope,data,evidence);model,initial=model_for(kind,scope,fitted['theta'],evidence)
            held=curve(kind,scope,fitted['theta'],campaign.IDLES['heldout'],evidence,continuous=False).reshape(4,16)
            for idle,values in zip(campaign.IDLES['heldout'],held):
                for ordinal,value in enumerate(values):predictions[kind].append(dict(memory_mode=memory,
                    requested_idle_ms=idle,ordinal=ordinal,predicted_ns=int(value)))
            profile[kind]=dict(scope=asdict(scope),fit=fitted,evidence=list(evidence),
                parameters=asdict(model.parameters) if kind=='stateful' else
                    dict(response_ns=model.response_ns) if kind=='static' else
                    dict(steady_response_ns=model.steady_response_ns,maximum_extra_response_ns=model.maximum_extra_response_ns,
                         idle_fill_tau_ns=model.idle_fill_tau_ns),
                initial_proxy_after_prime=initial.value if initial is not None else None)
            pred_train=curve(kind,scope,fitted['theta'],campaign.IDLES['profile'],evidence,continuous=False).reshape(1,4,16)
            residual=pred_train-data
            calibration.append(dict(memory_mode=memory,provider=kind,process_mean_APE=[float(abs(np.mean(pred_train)-np.mean(v))/np.mean(v)) for v in data],
                maximum_ordinal_mean_ape=float(np.max(np.abs(residual)/data)),
                first_response_mean_ape=float(np.mean(np.abs(residual[:,:,0])/data[:,:,0])),
                RMS_residual_ns=float(np.sqrt(np.mean(residual**2))),
                process_first_response_ns=data[:,:,0].tolist(),
                raw_condition_CVs=[dict(job=r['job']['id'],values=[c['burst_cv'] for c in r['conditions']]) for r in selected]))
        profiles[memory]=profile
    files=[Path(__file__),ROOT/'modeling/llm_infer_model/performance/idle_recovery.py',
        ROOT/'profiling/dynamic_validation/baseline_audit.py']
    # Inline exact modeling/estimator source is covered by the parameter-file
    # hash, even if the mutable checkout changes later. No target tensor values.
    sources=[dict(**receipt(p),source_text=p.read_text(encoding='utf-8')) for p in files]
    modelset=dict(schema='dynamic-idle-model-set-1',fitted_at_utc=campaign.e2a.utc(),context=context,
        profiles=profiles,calibration=calibration,profile_observations=receipt(observations_path),sources=sources,
        response_unit='inclusive completed invocation ns',primary_provider='stateful',
        fitting_labels='all profile per-copy ordinal means; equal process-condition-ordinal least squares',
        unstable_processes_removed=False,actual_idle_used_as_primary_input=False,
        whole_Dense_target_used_for_fit=False,future_target_state_used=False,
        baseline_support=receipt(gate_path),
        interpretation='bounded empirical readiness proxy; not an identified OS queue or power governor',
        hardware_accuracy_qualified=False)
    prediction=dict(schema='dynamic-idle-frozen-predictions-1',providers=predictions,rows=predictions['stateful'],
        primary_provider='stateful',planned_idle_only=True,whole_Dense_target_used_for_fit=False,future_target_state_used=False)
    modelpath,predpath=out/'fitted-candidates.json',out/'fitted-predictions.json'
    write(modelpath,modelset);write(predpath,prediction)
    result=campaign.freeze_predictions(out,modelpath,predpath)
    result['provider_calibration']=[{k:v for k,v in r.items() if k in ('memory_mode','provider','process_mean_APE','first_response_mean_ape','RMS_residual_ns')} for r in calibration]
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    print(json.dumps(fit_and_freeze(args.out),ensure_ascii=False),flush=True)
