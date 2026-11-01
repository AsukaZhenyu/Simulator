"""Profile-only prices, optional transition identification, causal forecasts.

Whole-target timings are never fitting labels. The independent acquisition
cannot start until explicit frozen parameter/prediction artifacts exist.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

from llm_infer_model.performance.planned_response import PlannedCompletedResponse,STATES
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from llm_infer_model.performance.parameters import UnknownParameter
from profiling.dynamic_validation import external,external_fit_gate

# Predeclared logarithmic grid in inclusive-response protocol time, not a
# claimed physical OS/clock-state time constant. Zero denotes memoryless.
TAUS_NS=(50000,100000,200000,500000,1000000,2000000,5000000,10000000,20000000,50000000,100000000,200000000,500000000)
AGE_BINS_NS=(0,50000,100000,200000,500000,1000000,2000000,5000000,10000000,20000000,50000000,100000000,200000000,500000000)
ROOT=external.ROOT
read,write,sha,require=external.read,external.write,external.sha,external.require


def no_heldout(out,manifest):
    require(not (out/'heldout-started.json').exists() and not any((out/'jobs'/j['id']).exists() for j in manifest['jobs'] if j['split']=='heldout'),
        'fitting/prediction after independent measured target acquisition rejected')


def price_map(entries):
    result={}
    for row in entries:
        key=row['semantic_key'];require(digest(row['semantic'])==key,'normalized response coordinate changed')
        states={(r['cpu_offered'],r['gpu_offered']):None if r['mean_ns'] is None else Fraction(r['mean_ns']) for r in row['states']}
        if set(states)!=set(STATES) or any(value is None for value in states.values()):raise UnknownParameter('missing exact profile response state for '+key)
        result[key]=dict(kind=row['semantic']['kind'],states=states)
    return result


def simulator(context,prices,order,request,provider,evidence,tau=None):
    intervals=lambda name:tuple(tuple(int(x)*1000000 for x in pair) for pair in request[name]['intervals_ms'])
    return PlannedCompletedResponse(context,prices,order,intervals('cpu_actor'),intervals('gpu_actor'),
        provider=provider,evidence=evidence,tau_ns=tau)


def transition_observations(manifest,observations):
    """Aggregate every eligible action into declared edge-age strata.

Per-action calibration starts are observed training coordinates. They never
become the starting clock of an independent target forecast.
"""
    records=[];excluded=dict(no_planned_edge=0,outside_transition_window=0,logical_control=0,zero_clock_label=0)
    for record in observations['records']:
        job=record['job'];native=read(record['native']['path']);hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick']
        program=read(manifest['programs'][job['program_key']]['program']['path'])
        semantics={i:semantic_action(program,i) for i in range(len(program['actions']))};keys={i:digest(s) for i,s in semantics.items()}
        edges=external.boundaries(job['request']);groups={}
        for block in native['blocks']:
            for action in block['actions']:
                if not edges:excluded['no_planned_edge']+=1;continue
                key=keys[action['action_index']]
                if semantics[action['action_index']]['kind'] not in ('COMPUTE','COPY'):excluded['logical_control']+=1;continue
                start=(action['begin_tick']-epoch)*1000000000//hz
                edge=min(edges,key=lambda x:abs(x-start));age=start-edge
                if abs(age)>AGE_BINS_NS[-1]:excluded['outside_transition_window']+=1;continue
                if action['wall_ns']<=0:excluded['zero_clock_label']+=1;continue
                age_bin=min(len(AGE_BINS_NS)-2,bisect_right(AGE_BINS_NS,abs(age))-1)
                group=(key,edge,-1 if age<0 else 1,age_bin)
                g=groups.setdefault(group,dict(semantic_key=key,edge_ns=edge,side=group[2],age_bin=age_bin,count=0,start_sum_ns=0,response_sum_ns=0))
                g['count']+=1;g['start_sum_ns']+=start;g['response_sum_ns']+=action['wall_ns']
        samples=[]
        for g in groups.values():
            g['start_mean_ns']=round(Fraction(g['start_sum_ns'],g['count']))
            g['response_mean_ns']=str(Fraction(g['response_sum_ns'],g['count']));samples.append(g)
        records.append(dict(job=job,native=record['native'],samples=samples))
    return dict(schema='dynamic-external-transition-labels-1',records=records,excluded_from_tau_identification=excluded,
        excluded_rows_remain_in_raw=True,age_bins_ns=list(AGE_BINS_NS),maximum_distance_to_nearest_edge_ns=AGE_BINS_NS[-1],
        labels='mean individual completed-action response in profile process/semantic/edge/signed-age strata',
        actual_profile_start_used_as_training_coordinate=True,whole_target_used_as_label=False,
        independent_prediction_start_uses_actual_target_boundaries=False)


def identify_tau(losses,counts):
    """Conservative profile-identification gate; unsupported is a valid result."""
    jobs=[job for job,values in losses.items() if values]
    if not jobs:return dict(status='unsupported',reasons=['no eligible profile transition strata'])
    candidates=(0,*TAUS_NS);means={t:statistics.mean(losses[j][t] for j in jobs) for t in candidates}
    best=min(TAUS_NS,key=lambda t:means[t]);null=means[0]
    improvement=(null-means[best])/null if null>0 else 0.
    rounds={r:[j for j in jobs if counts[j]['round']==r] for r in range(3)}
    round_best={};checks=[]
    for r,validation in rounds.items():
        training=[j for j in jobs if counts[j]['round']!=r]
        if not training or not validation:continue
        selected=min(TAUS_NS,key=lambda t:statistics.mean(losses[j][t] for j in training))
        round_best[str(r)]=selected
        baseline=statistics.mean(losses[j][0] for j in validation);history=statistics.mean(losses[j][selected] for j in validation)
        checks.append(dict(held_profile_round=r,selected_tau_ns=selected,baseline_loss=baseline,history_loss=history))
    valnull=statistics.mean(c['baseline_loss'] for c in checks) if checks else 0.
    valhistory=statistics.mean(c['history_loss'] for c in checks) if checks else valnull
    valimprovement=(valnull-valhistory)/valnull if valnull>0 else 0.
    plateau=[t for t in TAUS_NS if means[t]<=means[best]*1.01]
    reasons=[]
    if sum(counts[j]['strata'] for j in jobs)<50 or len(jobs)<6 or len(checks)!=3:reasons.append('insufficient transition strata/process rounds')
    if best in (TAUS_NS[0],TAUS_NS[-1]):reasons.append('best transition time at predeclared grid boundary')
    if improvement<.05:reasons.append('profile transition loss improvement below5%')
    if valimprovement<.02:reasons.append('conditional held-profile-round transition improvement below2%')
    if not round_best or max(round_best.values())>4*min(round_best.values()):reasons.append('profile-round transition times unstable by more than4x')
    if plateau and max(plateau)>4*min(plateau):reasons.append('one-percent objective plateau wider than4x')
    return dict(status='identified' if not reasons else 'unsupported',tau_ns=best if not reasons else None,
        diagnostic_best_tau_ns=best,reasons=reasons,grid_ns=list(TAUS_NS),normalized_loss_by_tau={str(t):v for t,v in means.items()},
        pooled_profile_improvement=improvement,conditional_profile_round_improvement=valimprovement,
        conditional_profile_round_checks=checks,one_percent_objective_plateau_ns=plateau,
        profile_round_prices_shared=True,independent_accuracy_claimed=False,
        interpretation='conditional identification of one empirical response-regime transition parameter; not physical OS/power-state identification')


def fit(out):
    out=Path(out).resolve();manifest=read(out/'manifest.json');external.check(manifest);no_heldout(out,manifest)
    require(not (out/'frozen-barrier.json').exists(),'frozen external parameter set already exists')
    if not (out/'fit-support.json').exists():external_fit_gate.inspect(out)
    support=read(out/'fit-support.json');require(support['all_heldout_actions_supported'],'unknown exact response cells; refuse fallback')
    observations=read(out/'profile-observations.json');prices=price_map(support['entries'])
    context=dict(hardware=manifest['hardware'],runtime=manifest['runtime'],binary_sha256=manifest['binary']['sha256'],
        controller=manifest['protocol']['controller'],actor_components={name:external.component(name) for name in ('cpu','gpu')},
        preparation=manifest['protocol']['initial_target'],action_boundary='unchanged inclusive completed Executor.action',
        offered_state_scope='owned actor controls, not physical available FLOPs or measured utilization')
    context_sha=digest(context);evidence=tuple(r['native']['path']+'#sha256='+r['native']['sha256'] for r in observations['records'])
    labels=transition_observations(manifest,observations);write(out/'transition-labels.json',labels)
    losses={};counts={};unsupported=None
    for record in labels['records']:
        job=record['job'];samples=record['samples']
        if not samples:continue
        order=tuple(dict.fromkeys(s['semantic_key'] for s in samples));losses[job['id']]={};counts[job['id']]=dict(round=job['round'],strata=len(samples))
        for tau in (0,*TAUS_NS):
            try:model=simulator(context_sha,prices,order,job['request'],'memoryless' if tau==0 else 'stateful',evidence,tau or None)
            except UnknownParameter as error:unsupported=repr(error);break
            squared=[]
            for s in samples:
                response=model.evaluate(start_ns=s['start_mean_ns'],semantic_order=(s['semantic_key'],)).duration_ns
                observed=float(Fraction(s['response_mean_ns']));scale=float(sum(prices[s['semantic_key']]['states'].values())/4)
                require(scale>0,'zero response cannot identify history')
                squared.append(((response-observed)/scale)**2)
            losses[job['id']][tau]=statistics.mean(squared)
        if unsupported:break
        print(json.dumps(dict(phase='transition-profile',job=job['id'],strata=len(samples))),flush=True)
    identification=dict(status='unsupported',tau_ns=None,reasons=[unsupported]) if unsupported else identify_tau(losses,counts)
    source_paths=[Path(__file__),ROOT/'profiling/dynamic_validation/external_fit_gate.py',ROOT/'modeling/llm_infer_model/performance/planned_response.py',
        ROOT/'modeling/llm_infer_model/performance/service_response.py',ROOT/'modeling/llm_infer_model/performance/evaluator.py',
        ROOT/'modeling/llm_infer_model/performance/graph.py',ROOT/'modeling/llm_infer_model/performance/parameters.py',
        ROOT/'modeling/llm_infer_model/performance/reusable_action_costs.py']
    model=dict(schema='dynamic-external-model-set-1',manifest=external.receipt(out/'manifest.json'),context=context,context_sha256=context_sha,
        entries=support['entries'],transition_identification=identification,transition_labels=external.receipt(out/'transition-labels.json'),
        profile_transition_losses={job:{str(t):loss for t,loss in values.items()} for job,values in losses.items()},
        profile_transition_counts=counts,evidence=list(evidence),
        sources=[dict(**external.receipt(p),source_text=p.read_text(encoding='utf-8')) for p in source_paths],
        static='off/off exact semantic price',memoryless='four exact offered-state prices integrated by common event evaluator',
        stateful='four bounded empirical response-regime proxies with one shared identified tau; optional unsupported',
        whole_target_used_for_fit=False,future_target_state_used=False,
        fitting_labels='settled per-action process-cell means; individual transition per-action stratum means only',
        response_unit='inclusive completed invocation ns',hardware_accuracy_qualified=False)
    write(out/'candidate-model.json',model)
    return dict(model=str(out/'candidate-model.json'),stateful=identification)


def predict_and_freeze(out,model_path,primary):
    out=Path(out).resolve();manifest=read(out/'manifest.json');external.check(manifest);no_heldout(out,manifest)
    model=read(model_path);require(model['schema']=='dynamic-external-model-set-1','external model schema differs')
    require(model['manifest']['sha256']==sha(out/'manifest.json'),'calibrated plan changed')
    for source in model['sources']:require(sha(source['path'])==source['sha256'],'calibrated modeling/estimator source changed')
    providers=['static','memoryless'];identified=model['transition_identification']['status']=='identified'
    if identified:providers.append('stateful')
    require(primary in providers,'primary stateful requested without identified transition; unknown is not a coefficient')
    prices=price_map(model['entries']);rows={p:{} for p in providers};tau=model['transition_identification'].get('tau_ns')
    programs={}
    for workload in external.WORKLOADS:
        binding=manifest['programs'][workload+'-heldout'];require(sha(binding['program']['path'])==binding['program']['sha256'],'mathematical input changed')
        program=read(binding['program']['path']);programs[workload]=tuple(digest(semantic_action(program,i)) for i in range(len(program['actions'])))
    for job in (j for j in manifest['jobs'] if j['split']=='heldout'):
        for provider in providers:
            instance=simulator(model['context_sha256'],prices,programs[job['workload']],job['request'],provider,tuple(model['evidence']),tau if provider=='stateful' else None)
            rows[provider][job['id']]=instance.predict_replays()
        print(json.dumps(dict(phase='frozen-own-timeline-prediction',job=job['id'])),flush=True)
    predictions=dict(schema='dynamic-external-frozen-predictions-1',primary_provider=primary,providers=rows,
        unsupported_providers={} if identified else dict(stateful=model['transition_identification']),
        future_target_state_used=False,whole_target_time_used_for_fit=False,observed_target_boundaries_used=False,
        scoped_offered_actor_protocol=True,physical_FLOPs_or_utilization_claimed=False)
    path=out/'candidate-predictions.json';write(path,predictions)
    return external.freeze(out,model_path,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fit','predict-freeze'));p.add_argument('--out',required=True)
    p.add_argument('--model');p.add_argument('--primary',choices=('static','memoryless','stateful'),default='memoryless');a=p.parse_args()
    result=fit(a.out) if a.command=='fit' else predict_and_freeze(a.out,a.model or str(Path(a.out)/'candidate-model.json'),a.primary)
    print(json.dumps(result,ensure_ascii=False),flush=True)
