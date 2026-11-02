"""Revised constant-state calibration; unchanged dynamic heldout criterion.

The v2 campaign validates actual steady-state bracketing for calibration,
including a retained v1 observation rejected solely for late profile entry.
Original native/execution records stay immutable. Prediction equations and
heldout entry/QPC checks are reused from the frozen v1 pure helpers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from . import bare_macro_fit as shared
from . import bare_macro_v2_campaign as campaign

read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
STATES,PROVIDERS=shared.STATES,shared.PROVIDERS


def _calibrated_cells(out,manifest,verify):
    jobs=[j for j in manifest['jobs'] if j['split']=='profile']
    require(len(jobs)==12,'complete12-context calibration required; no selective deletion of prior OFF/ON observations')
    grouped=defaultdict(list);identities=set();hashes=set();evidence=[]
    for job in jobs:
        native,execution,path=verify(out,job,manifest)
        state=job.get('offered_state',job['request'].get('offered_state'))
        require(isinstance(state,(list,tuple)) and len(state)==2
                and all(type(x) is int and x in (0,1) for x in state),'explicit component calibration state required')
        state=tuple(state)
        require(state==tuple(native['request']['offered_state']) and native['instrumentation']=='stamped'
                and native['runtime_callbacks']['enabled'] is False and native['runtime_callbacks']['record_count']==0,
                'callback stage prices or relabeled states cannot enter macro calibration')
        identity=(execution['process_id'],execution.get('process_creation_time_100ns'))
        require(type(identity[1]) is int and identity[1]>0 and identity not in identities,
                'distinct verified process identities required for all12 calibration contexts')
        require(execution['native']['sha256'] not in hashes,'native record reused under another context label')
        identities.add(identity);hashes.add(execution['native']['sha256'])
        grouped[state].append(dict(job_id=job['id'],state=state,response_ns=shared.component_label(native),
            imported=job.get('imported',False),process_identity=identity,native=execution['native'],execution=receipt(path),
            original_execution_status=execution['status'],
            imported_native_binding=job.get('imported_native'),
            original_target_entry_controller_passed=native['target_entry_controller_passed'],
            original_start_lateness_ns=native['initial_start_lateness_ns'],
            actual_constant_state_verified_by='bare_macro_v2_campaign.verified_job',
            original_rejection_or_lateness_record_modified=False))
        evidence.append(execution['native']['path']+'#sha256='+execution['native']['sha256'])
    require(set(grouped)==set(STATES) and all(len(grouped[state])==3 for state in STATES),
            'three independent verified component observations in every state required')
    cells=[]
    for state in STATES:
        labels=[r['response_ns'] for r in grouped[state]];mean=sum(labels,Fraction(0))/3
        cells.append(dict(state=state,mean_ns=mean,observations=grouped[state],fresh_contexts=3,
            raw_profile_CV=statistics.pstdev([float(x) for x in labels])/float(mean),
            three_component_observations_establish_low_variance=False))
    return cells,tuple(evidence)


def calibrated_cells(out,manifest):
    return _calibrated_cells(out,manifest,campaign.verified_job)


def fit_and_freeze(out):
    out=Path(out).resolve();manifest=campaign.check(out);shared.check_bound_files(manifest);shared.no_heldout(out,manifest)
    require(not (out/'component-profile.json').exists() and not (out/'frozen-barrier.json').exists(),
            'profile and forecasts are one frozen operation; no adaptive refitting')
    cells,evidence=calibrated_cells(out,manifest)
    profile=dict(schema='bare-Attention-component-profile-2',cells=cells,evidence=evidence,scope=manifest['scope'],
        manifest=receipt(out/'manifest.json'),calibration_target_is_component=True,
        whole_LLM_target_time_used_for_fit=False,callback_stage_prices_used=False,raw_observations_removed=0,
        calibration_rule='actual complete steady actor/window bracketing and numerical/source/QPC/affinity checks; planned-entry10ms is not a steady-state label condition',
        prior_v1_rejection_retained=True,heldout_entry_criterion_ns=10000000,
        response_equation_reused_from_frozen_v1=True,hardware_accuracy_qualified=False)
    shared.write_exact(out/'component-profile.json',profile)
    held=[j for j in manifest['jobs'] if j['split']=='heldout'];require(len(held)==6,'six heldout processes required')
    cache={};rows=[]
    for job in held:
        for provider in PROVIDERS:
            key=(shared.digest(job['request']),provider)
            if key not in cache:
                cache[key]=shared.predict(cells,manifest['scope'],job['request'],provider,evidence)
            rows.append(dict(job_id=job['id'],provider=provider,pulse_plan=job['pulse_plan'],prediction=cache[key],
                primary_planned_entry_ns=0,entry_controller_max_lateness_ns=10000000,
                observed_entry_or_internal_boundaries_used=False))
    shared.write_exact(out/'frozen-predictions.json',dict(schema='bare-Attention-component-forecasts-2',rows=rows,
        declared_provider_process_cells=12,providers=PROVIDERS,primary_provider='memoryless_four_state_macro',
        planned_entry_ns=0,all_predictions_before_heldout_collection=True,actual_target_times_used=False,
        whole_LLM_target_time_used_for_fit=False,heldout10ms_entry_requirement_preserved=True,
        model='unchanged modeling.PlannedCompletedResponse and frozenv1 predict helper'))
    barrier=dict(schema='bare-Attention-component-freeze-2',manifest_sha256=sha(out/'manifest.json'),
        profile_supported=True,parameters_frozen=True,predictions_frozen=True,
        calibration_target_is_component=True,whole_LLM_target_time_used_for_fit=False,
        files={name:sha(out/name) for name in ('component-profile.json','frozen-predictions.json')})
    write(out/'frozen-barrier.json',barrier)
    return dict(profile=receipt(out/'component-profile.json'),barrier=receipt(out/'frozen-barrier.json'),
                component_cells=4,calibration_contexts=12,heldout_forecasts=12,hardware_accuracy_validated=False)


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
    result=dict(schema='bare-Attention-component-independent-evaluation-2',rows=rows,processes=processes,
        providers=summaries,repeat_variability=repeats,declared_provider_process_cells=12,
        primary_provider='memoryless_four_state_macro',primary_all_six_passed=next(r['all_six_passed'] for r in summaries if r['provider']=='memoryless_four_state_macro'),
        thresholds=dict(mean_response_APE_max=.10,block_APE_P90_max=.20),raw_trials_or_slow_blocks_removed=0,
        no_refit=True,whole_LLM_dynamic_accuracy_validated=False,raw_CV_is_prediction_gate=False,
        actual_start_or_internal_boundaries_used_in_prediction=False,heldout10ms_entry_requirement_preserved=True,
        prior_v1_profile_rejection_preserved=True,
        entry_uncertainty='planned start0; heldout owned controller permits at most10ms lateness; observed start never reanchors forecast',
        limits='one complete fixed-shape Attention component and owned GEMM actors; no OS/queue/recovery identification')
    shared.write_exact(out/'INDEPENDENT_EVALUATION.json',result)
    return dict(output=receipt(out/'INDEPENDENT_EVALUATION.json'),primary_all_six_passed=result['primary_all_six_passed'],providers=summaries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('fit-freeze','evaluate'));parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(fit_and_freeze(args.out) if args.command=='fit-freeze' else evaluate(args.out),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
