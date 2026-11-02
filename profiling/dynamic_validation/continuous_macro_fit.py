"""Continuous-lifecycle component calibration, with independent new schedules.

Each calibration process contributes its arithmetic mean completed Attention
invocation, then the three fresh-process means receive equal weight. The warm
invocation and every host gap are retained as separate diagnostics. No previous
validation result or whole LLM latency becomes a fitting label. Modeling owns
the unchanged four-state response integration and its own prediction clock.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from . import bare_macro_fit as shared
from . import continuous_macro_campaign as campaign

read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
STATES,PROVIDERS=shared.STATES,shared.PROVIDERS


def _interval(row,hz,label):
    begin,end=row['begin_tick'],row['end_tick']
    require(type(begin) is int and type(end) is int and begin<=end,label+' QPC interval malformed')
    value=Fraction((end-begin)*10**9,hz)
    if 'wall_ns' in row:require(abs(value-row['wall_ns'])<=1,label+' QPC/response differs')
    return value


def continuous_context(native):
    """Exact local component means, without target-wall/gap compensation."""
    hz=native['clock']['frequency_hz'];require(type(hz) is int and hz>0,'positive QPC frequency required')
    require(native['foreground_lifecycle']=='continuous_invocation'
            and native['reset_prime_between_timed_blocks'] is False,
            'continuous foreground lifecycle without inter-invocation reset/prime required')
    warm=native['under_state_warmup'];profile=native['continuous_profile']
    require(warm['action_index']==22 and warm['begin_tick']<warm['end_tick']<=profile['begin_tick'],
            'separate completed under-state warm invocation must precede continuous calibration')
    warm_response=_interval(warm,hz,'under-state warmup')
    elapsed=_interval(profile,hz,'continuous target')
    require(profile['min_duration_ns']==1000000000 and profile['max_calls']==512,
            'predeclared1s/512-call calibration bounds required')
    require(profile['accuracy_dependent_stop'] is False,'calibration stopping cannot depend on accuracy/variance')
    require(profile['minimum_duration_reached'] is True and elapsed>=1000000000,
            'safety-cap stop before1s remains unsupported; cannot fabricate a steady continuous price')
    require(profile['stop_reason']=='minimum_duration','continuous calibration must end by its declared minimum-duration rule')
    actions=profile['actions'];gaps=profile['gaps']
    require(isinstance(actions,list) and 1<=len(actions)<=512 and len(gaps)==len(actions)+1,
            'complete repeated-action and prefix/intercall/suffix gap inventory required')
    previous=profile['begin_tick'];labels=[];expected_gaps=[]
    for ordinal,action in enumerate(actions):
        require(action['ordinal']==ordinal and action['action_index']==22
                and previous<=action['begin_tick']<action['end_tick']<=profile['end_tick'],
                'continuous completed Attention action order/boundary differs')
        expected_gaps.append((previous,action['begin_tick']))
        labels.append(_interval(action,hz,'completed Attention invocation'));previous=action['end_tick']
    expected_gaps.append((previous,profile['end_tick']))
    require([(g['begin_tick'],g['end_tick']) for g in gaps]==expected_gaps,
            'all raw host gaps must form the exact complement of component intervals')
    gap_values=[_interval(g,hz,'outside-action gap') for g in gaps]
    require(sum(labels,Fraction(0))+sum(gap_values,Fraction(0))==elapsed,
            'component and gap QPC partition does not close')
    return dict(invocations=len(labels),invocation_response_mean_ns=sum(labels,Fraction(0))/len(labels),
        invocation_responses_ns=labels,
        within_context_invocation_CV=statistics.pstdev([float(x) for x in labels])/float(statistics.mean(labels)) if len(labels)>1 else None,
        within_context_repeated_response_variability_identified=len(labels)>1,
        single_long_postwarm_invocation=len(labels)==1,
        minimum_duration_or_sample_count_claims_low_variance=False,
        timed_window_ns=elapsed,outside_action_gap_ns=sum(gap_values,Fraction(0)),
        outside_action_gap_fraction=float(sum(gap_values,Fraction(0))/elapsed),gap_responses_ns=gap_values,
        under_state_warmup_response_ns=warm_response,warm_response_used_as_steady_price=False,
        outside_action_gap_added_to_component_price=False,whole_target_total_fitted=False)


def _calibrated_cells(out,manifest,verify):
    require(manifest['scope']['context']['foreground_lifecycle']=='continuous_invocation',
            'explicit continuous foreground lifecycle scope required')
    jobs=[j for j in manifest['jobs'] if j['split']=='profile'];require(len(jobs)==12,'all12 new continuous calibration processes required')
    grouped=defaultdict(list);identities=set();hashes=set();evidence=[]
    for job in jobs:
        require(job.get('imported') is not True,'prior single-call or heldout observations cannot be imported for continuous fitting')
        native,execution,path=verify(out,job,manifest)
        require(execution['status']=='completed','all continuous calibration contexts must complete their declared support checks')
        state=job.get('offered_state',job['request'].get('offered_state'))
        require(isinstance(state,(tuple,list)) and len(state)==2
                and all(type(x) is int and x in (0,1) for x in state),'explicit offered-state coordinate required')
        state=tuple(state);require(state==tuple(native['request']['offered_state']),'declared/native state differs')
        require(native['instrumentation']=='stamped' and native['callback_stage_prices_used'] is False,
                'only original stamped component responses; fine observer labels forbidden')
        identity=(execution['process_id'],execution.get('process_creation_time_100ns'))
        require(type(identity[1]) is int and identity[1]>0 and identity not in identities,
                'all12 fresh continuous processes require distinct verified PID+creation identities')
        require(execution['native']['sha256'] not in hashes,'native result reused under another context')
        identities.add(identity);hashes.add(execution['native']['sha256'])
        grouped[state].append(dict(job_id=job['id'],state=state,process_identity=identity,
            native=execution['native'],execution=receipt(path),**continuous_context(native)))
        evidence.append(execution['native']['path']+'#sha256='+execution['native']['sha256'])
    require(set(grouped)==set(STATES) and all(len(grouped[s])==3 for s in STATES),
            'three fresh continuous contexts in every exact offered state required')
    cells=[]
    for state in STATES:
        means=[r['invocation_response_mean_ns'] for r in grouped[state]];mean=sum(means,Fraction(0))/3
        cells.append(dict(state=state,mean_ns=mean,observations=grouped[state],fresh_contexts=3,
            weighting='equal context means; not pooled invocation weighting',
            raw_between_context_mean_CV=statistics.pstdev([float(x) for x in means])/float(mean),
            warm_or_gap_cost_added_to_component_mean=False))
    return cells,tuple(evidence)


def calibrated_cells(out,manifest):
    return _calibrated_cells(out,manifest,campaign.verified_job)


def fit_and_freeze(out):
    out=Path(out).resolve();manifest=campaign.check(out);shared.check_bound_files(manifest);shared.no_heldout(out,manifest)
    require(not (out/'component-profile.json').exists() and not (out/'frozen-barrier.json').exists(),
            'continuous calibration and forecasts are one frozen operation; no adaptive refitting')
    cells,evidence=calibrated_cells(out,manifest)
    shared.write_exact(out/'component-profile.json',dict(schema='continuous-Attention-component-profile-1',cells=cells,
        evidence=evidence,scope=manifest['scope'],manifest=receipt(out/'manifest.json'),
        calibration_target_is_component=True,foreground_lifecycle='continuous_invocation',
        whole_LLM_target_time_used_for_fit=False,previous_heldout_data_used_for_fit=False,
        fine_callback_prices_used=False,raw_invocations_or_gaps_removed=0,hardware_accuracy_qualified=False))
    held=[j for j in manifest['jobs'] if j['split']=='heldout'];require(len(held)==6,'six independent new schedule/process targets required')
    cache={};forecasts=[]
    for job in held:
        for provider in PROVIDERS:
            key=(shared.digest(job['request']),provider)
            if key not in cache:cache[key]=shared.predict(cells,manifest['scope'],job['request'],provider,evidence)
            forecasts.append(dict(job_id=job['id'],provider=provider,pulse_plan=job['pulse_plan'],prediction=cache[key],
                primary_planned_entry_ns=0,entry_controller_max_lateness_ns=10000000,
                actual_target_start_or_boundaries_used=False))
    shared.write_exact(out/'frozen-predictions.json',dict(schema='continuous-Attention-component-forecasts-1',rows=forecasts,
        declared_provider_process_cells=12,providers=PROVIDERS,primary_provider='memoryless_four_state_macro',
        planned_entry_ns=0,all_predictions_before_new_heldout_collection=True,previous_heldout_observations_used_for_fit=False,
        heldout_schedules_and_processes_new=True,heldout_mathematical_seed_previously_used=True,
        mathematical_seed_is_not_a_new_independence_axis=True,
        whole_LLM_target_time_used_for_fit=False,heldout10ms_entry_requirement_preserved=True))
    barrier=dict(schema='continuous-Attention-component-freeze-1',manifest_sha256=sha(out/'manifest.json'),
        profile_supported=True,parameters_frozen=True,predictions_frozen=True,
        calibration_target_is_component=True,whole_LLM_target_time_used_for_fit=False,
        files={name:sha(out/name) for name in ('component-profile.json','frozen-predictions.json')})
    write(out/'frozen-barrier.json',barrier)
    return dict(profile=receipt(out/'component-profile.json'),barrier=receipt(out/'frozen-barrier.json'),
                continuous_component_cells=4,calibration_contexts=12,heldout_forecasts=12,hardware_accuracy_validated=False)


def evaluate(out):
    out=Path(out).resolve();manifest=campaign.check(out);shared.check_bound_files(manifest)
    held,predictions=shared.verified_prediction_map(out,manifest);rows=[];processes=[]
    for job in held:
        try:
            native,execution,path=campaign.verified_job(out,job,manifest)
            total,blocks,details=shared.measured_intervals(native)
            processes.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],status='completed',
                native=execution['native'],execution=receipt(path),measured_total_ns=total,measured_block_ns=blocks,
                temporal_block_CV=statistics.pstdev(blocks)/statistics.mean(blocks),
                temporal_block_CV_is_stationary_noise=False,**details))
        except (ValueError,KeyError,FileNotFoundError) as error:
            processes.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],status='unsupported_or_not_completed',reason=repr(error)))
            rows.extend(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=p,
                status='unsupported_or_not_completed',accuracy_passed=False,reason=repr(error)) for p in PROVIDERS)
            continue
        for provider in PROVIDERS:
            prediction=predictions[job['id'],provider]['prediction'];pred=prediction['total_ns']
            errors=[abs(p/m-1) for p,m in zip(prediction['block_ns'],blocks)];p90=sorted(errors)[14];error=abs(pred/total-1)
            rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,status='evaluated',
                predicted_total_ns=pred,measured_total_ns=total,predicted_block_ns=prediction['block_ns'],measured_block_ns=blocks,
                mean_response_APE=error,signed_bias=(pred-total)/total,absolute_error_ns=abs(pred-total),
                block_APEs=errors,block_APE_P90=p90,accuracy_passed=error<=.10 and p90<=.20,
                raw_CV_is_prediction_gate=False,parameters_refitted=False,observed_boundary_reanchors=False))
    summaries=[]
    for provider in PROVIDERS:
        selected=[r for r in rows if r['provider']==provider];valid=[r for r in selected if r['status']=='evaluated']
        summaries.append(dict(provider=provider,declared=6,evaluated=len(valid),passed=sum(r['accuracy_passed'] for r in selected),
            all_six_passed=len(selected)==6 and all(r['accuracy_passed'] for r in selected),
            mean_process_APE=statistics.mean(r['mean_response_APE'] for r in valid) if valid else None,
            worst_process_APE=max((r['mean_response_APE'] for r in valid),default=None),
            worst_block_APE_P90=max((r['block_APE_P90'] for r in valid),default=None)))
    repeats=[]
    for plan in sorted({j['pulse_plan'] for j in held}):
        values=[r['measured_total_ns'] for r in processes if r['pulse_plan']==plan and r['status']=='completed']
        repeats.append(dict(pulse_plan=plan,declared_fresh_processes=3,completed_fresh_processes=len(values),total_samples_ns=values,
            raw_fresh_process_CV=statistics.pstdev(values)/statistics.mean(values) if len(values)==3 else None,
            raw_CV_is_prediction_gate=False))
    result=dict(schema='continuous-Attention-component-independent-evaluation-1',rows=rows,processes=processes,
        providers=summaries,repeat_variability=repeats,declared_provider_process_cells=12,
        primary_provider='memoryless_four_state_macro',primary_all_six_passed=next(r['all_six_passed'] for r in summaries if r['provider']=='memoryless_four_state_macro'),
        thresholds=dict(mean_response_APE_max=.10,block_APE_P90_max=.20),raw_invocations_or_slow_blocks_removed=0,
        no_refit=True,whole_LLM_dynamic_accuracy_validated=False,raw_CV_is_prediction_gate=False,
        actual_start_or_internal_boundaries_used_in_prediction=False,heldout10ms_entry_requirement_preserved=True,
        old_failures_preserved=True,heldout_schedules_and_processes_new=True,heldout_mathematical_seed_previously_used=True,
        entry_uncertainty='planned start0; owned target controller permits at most10ms lateness; observed start never reanchors forecast',
        limits='one complete fixed-shape Attention component with continuous calibration and owned GEMM actors; no identified OS/queue/recovery mechanism')
    shared.write_exact(out/'INDEPENDENT_EVALUATION.json',result)
    return dict(output=receipt(out/'INDEPENDENT_EVALUATION.json'),primary_all_six_passed=result['primary_all_six_passed'],providers=summaries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('fit-freeze','evaluate'));parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(fit_and_freeze(args.out) if args.command=='fit-freeze' else evaluate(args.out),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
