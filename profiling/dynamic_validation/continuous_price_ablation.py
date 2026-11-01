"""Paired price-estimator diagnostic from the same12 continuous cal processes.

The first under-state warm invocation supplies a prepared-single baseline;
the subsequent continuous invocations supply the primary estimator. Both use
the same modeling g and the same six new heldout schedules/processes. The
single estimator deliberately mismatches the target lifecycle and is never
promoted as a qualified continuous-execution cost. Existing primary providers,
their freeze and acceptance gates remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import shutil
import statistics

from . import bare_macro_fit as shared
from . import continuous_macro_campaign as campaign
from . import continuous_macro_fit as primary

read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
VARIANTS=('prepared_single','continuous')
PROVIDER='memoryless_four_state_macro'


def paired_cells(records):
    """Equal fresh-context weighting; no individual-call pooling or selection."""
    require(len(records)==12,'all12 same-process paired calibration records required')
    identities=[tuple(r['process_identity']) for r in records]
    require(len(set(identities))==12 and len({r['native']['sha256'] for r in records})==12,
            'paired records must retain12 distinct native process identities')
    grouped=defaultdict(list)
    for record in records:
        state=tuple(record['state']);require(state in primary.STATES,'unknown offered state')
        require(all(type(x) is int and x in (0,1) for x in state),'offered-state bits must be explicit integers')
        require(Fraction(record['prepared_single_ns'])>0 and Fraction(record['continuous_mean_ns'])>0,
                'positive measured paired responses required')
        grouped[state].append(record)
    require(set(grouped)==set(primary.STATES) and all(len(grouped[s])==3 for s in primary.STATES),
            'three paired fresh processes per state required; no missing-cell substitution')
    values={variant:[] for variant in VARIANTS}
    for state in primary.STATES:
        for variant,field in (('prepared_single','prepared_single_ns'),('continuous','continuous_mean_ns')):
            values[variant].append(dict(state=state,mean_ns=sum((Fraction(r[field]) for r in grouped[state]),Fraction(0))/3,
                observations=grouped[state],fresh_contexts=3,weighting='equal context means'))
    return values


def variant_scope(scope,variant):
    require(variant in VARIANTS,'unknown predeclared paired estimator')
    require(scope['context']['foreground_lifecycle']=='continuous_invocation',
            'source calibration/target continuous lifecycle must remain explicit')
    value=json.loads(json.dumps(scope))
    if variant=='prepared_single':
        value['context']['foreground_lifecycle']='prepared_single_invocation'
        value['context']['measurement_component_origin']='first under_state_warmup invocation, before timed continuous loop'
        value['context_sha256']=shared.digest(value['context'])
    return value


def freeze(out,sidecar):
    out,sidecar=Path(out).resolve(),Path(sidecar).resolve()
    manifest=campaign.check(out);shared.no_heldout(out,manifest)
    authorization=read(sidecar/'ROOT_AUTHORIZATION.json')
    require(authorization.get('forecast_freeze_authorized') is True
            and authorization.get('script_sha256')==sha(__file__)
            and authorization.get('manifest_sha256')==sha(out/'manifest.json')
            and authorization.get('primary_frozen_barrier_sha256')==sha(out/'frozen-barrier.json'),
            'root authorization must bind this exact additional sidecar and primary freeze')
    require(not (sidecar/'FREEZE_BARRIER.json').exists() and not (sidecar/'PROFILE_ABLATION.json').exists(),
            'paired ablation is one additional pre-heldout freeze; no adaptive repeat')
    require(not (sidecar/'EXECUTED_SOURCE.py').exists(),'new exclusive sidecar source archive required')
    shutil.copyfile(__file__,sidecar/'EXECUTED_SOURCE.py')
    held,existing=shared.verified_prediction_map(out,manifest)
    profile=read(out/'component-profile.json');require(profile['foreground_lifecycle']=='continuous_invocation'
        and profile['whole_LLM_target_time_used_for_fit'] is False,'supported target-blind primary profile required')
    records=[];evidence=[]
    for job in (j for j in manifest['jobs'] if j['split']=='profile'):
        native,execution,path=campaign.verified_job(out,job,manifest)
        context=primary.continuous_context(native)
        warm=native['under_state_warmup']['wall_ns']
        require(type(warm) is int and warm>0
                and abs(Fraction(warm)-context['under_state_warmup_response_ns'])<=1,
                'actual complete warm response/QPC binding missing')
        records.append(dict(job_id=job['id'],state=job['offered_state'],
            process_identity=(execution['process_id'],execution['process_creation_time_100ns']),
            native=execution['native'],execution=receipt(path),prepared_single_ns=warm,
            continuous_mean_ns=context['invocation_response_mean_ns'],continuous_invocations=context['invocations']))
        evidence.append(execution['native']['path']+'#sha256='+execution['native']['sha256'])
    cells=paired_cells(records)
    expected={tuple(c['state']):Fraction(c['mean_ns']) for c in profile['cells']}
    require({tuple(c['state']):c['mean_ns'] for c in cells['continuous']}==expected,
            'paired continuous estimator must equal frozen primary prices exactly')
    shared.write_exact(sidecar/'PROFILE_ABLATION.json',dict(schema='continuous-price-paired-ablation-1',variants=cells,
        primary_profile=receipt(out/'component-profile.json'),same_native_processes_for_both_estimators=True,
        invocation_order='prepared-single warm first, continuous invocations later; order is intentionally fixed',
        hardware_runtime_binary_and_actor_workloads_identical=True,background_between_processes_not_proven_identical=True,
        estimator_difference_is_measurement_lifecycle=True,complete_causal_physical_mechanism_identified=False,
        prepared_single_is_diagnostic_out_of_target_lifecycle_scope=True,prepared_single_cost_qualified=False,
        existing_primary_parameters_or_gates_changed=False,heldout_observations_used=False))
    forecasts=[];cache={}
    for job in held:
        for variant in VARIANTS:
            key=(shared.digest(job['request']),variant)
            if key not in cache:
                cache[key]=shared.predict(cells[variant],variant_scope(manifest['scope'],variant),job['request'],PROVIDER,tuple(evidence))
            result=cache[key]
            if variant=='continuous':
                original=existing[job['id'],PROVIDER]['prediction']
                require(result['total_ns']==original['total_ns'] and result['block_ns']==original['block_ns'],
                        'paired continuous forecast differs from frozen primary g forecast')
            forecasts.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],variant=variant,prediction=result,
                diagnostic_only=variant=='prepared_single',calibration_lifecycle_matches_target=variant=='continuous',
                target_observed_entry_or_boundaries_used=False))
    shared.write_exact(sidecar/'FROZEN_PREDICTIONS.json',dict(schema='continuous-price-ablation-predictions-1',rows=forecasts,
        same_g_for_both_estimators='unchanged modeling.PlannedCompletedResponse memoryless',same_six_new_target_cases=True,
        primary_two_providers_unchanged=True,prepared_single_is_not_a_qualified_cost_provider=True,
        current_heldout_observations_read=False,mathematical_seed_previously_used=True))
    inputs=dict(manifest=receipt(out/'manifest.json'),primary_barrier=receipt(out/'frozen-barrier.json'),
        primary_profile=receipt(out/'component-profile.json'),primary_predictions=receipt(out/'frozen-predictions.json'),
        script=receipt(__file__),archived_script=receipt(sidecar/'EXECUTED_SOURCE.py'),
        authorization=receipt(sidecar/'ROOT_AUTHORIZATION.json'))
    write(sidecar/'INPUTS.json',inputs)
    barrier=dict(schema='continuous-price-ablation-freeze-1',inputs=inputs,
        files={name:sha(sidecar/name) for name in ('PROFILE_ABLATION.json','FROZEN_PREDICTIONS.json','INPUTS.json')},
        all_additional_predictions_frozen_before_current_heldout=True,existing_primary_freeze_unchanged=True)
    write(sidecar/'FREEZE_BARRIER.json',barrier)
    return dict(barrier=receipt(sidecar/'FREEZE_BARRIER.json'),paired_estimators=2,frozen_target_forecasts=12,
                primary_forecasts_equal=True,prepared_single_qualified=False)


def evaluate(out,sidecar):
    out,sidecar=Path(out).resolve(),Path(sidecar).resolve();manifest=campaign.check(out)
    barrier=read(sidecar/'FREEZE_BARRIER.json')
    require(barrier['all_additional_predictions_frozen_before_current_heldout'] is True
            and barrier['existing_primary_freeze_unchanged'] is True,'complete sidecar pre-heldout freeze required')
    for item in barrier['inputs'].values():require(sha(item['path'])==item['sha256'],'paired source/primary/authorization binding changed')
    require(barrier['inputs']['manifest']['sha256']==sha(out/'manifest.json'),'paired manifest differs')
    for name,value in barrier['files'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts and sha(sidecar/path)==value,'frozen paired forecast changed')
    held,_=shared.verified_prediction_map(out,manifest)
    frozen=read(sidecar/'FROZEN_PREDICTIONS.json');predictions={(r['job_id'],r['variant']):r for r in frozen['rows']}
    require(len(frozen['rows'])==12 and set(predictions)=={(j['id'],v) for j in held for v in VARIANTS},
            'both predeclared estimator forecasts required for every target')
    rows=[];processes=[]
    for job in held:
        try:
            native,execution,path=campaign.verified_job(out,job,manifest)
            total,blocks,details=shared.measured_intervals(native)
            processes.append(dict(job_id=job['id'],status='completed',native=execution['native'],execution=receipt(path),**details))
        except (ValueError,KeyError,FileNotFoundError) as error:
            processes.append(dict(job_id=job['id'],status='unsupported_or_not_completed',reason=repr(error)))
            rows.extend(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],variant=v,
                status='unsupported_or_not_completed',thresholds_met=False,reason=repr(error)) for v in VARIANTS)
            continue
        for variant in VARIANTS:
            prediction=predictions[job['id'],variant]['prediction'];pred=prediction['total_ns']
            require(len(prediction['block_ns'])==16 and all(type(x) is int and x>=0 for x in prediction['block_ns'])
                    and sum(prediction['block_ns'])==pred,'paired predicted block clock differs')
            errors=[abs(p/m-1) for p,m in zip(prediction['block_ns'],blocks)];p90=sorted(errors)[14];error=abs(pred/total-1)
            rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],variant=variant,status='evaluated',
                predicted_total_ns=pred,measured_total_ns=total,mean_response_APE=error,
                signed_bias=(pred-total)/total,absolute_error_ns=abs(pred-total),block_APEs=errors,block_APE_P90=p90,
                thresholds_met=error<=.10 and p90<=.20,diagnostic_only=variant=='prepared_single',
                prepared_single_qualification_granted=False,actual_target_start_reanchored=False))
    summaries=[]
    for variant in VARIANTS:
        selected=[r for r in rows if r['variant']==variant];valid=[r for r in selected if r['status']=='evaluated']
        summaries.append(dict(variant=variant,declared=6,evaluated=len(valid),thresholds_met=sum(r['thresholds_met'] for r in selected),
            mean_process_APE=statistics.mean(r['mean_response_APE'] for r in valid) if valid else None,
            worst_process_APE=max((r['mean_response_APE'] for r in valid),default=None),
            worst_block_APE_P90=max((r['block_APE_P90'] for r in valid),default=None)))
    paired=[]
    for job in held:
        selected={r['variant']:r for r in rows if r['job_id']==job['id'] and r['status']=='evaluated'}
        if set(selected)==set(VARIANTS):
            paired.append(dict(job_id=job['id'],continuous_minus_single_mean_APE=
                selected['continuous']['mean_response_APE']-selected['prepared_single']['mean_response_APE'],
                continuous_minus_single_block_APE_P90=
                selected['continuous']['block_APE_P90']-selected['prepared_single']['block_APE_P90']))
    result=dict(schema='continuous-price-ablation-independent-score-1',rows=rows,processes=processes,summaries=summaries,
        paired_error_differences=paired,same_raw_targets_scored_for_both_estimators=True,
        existing_primary_protocol_and_acceptance_results_unchanged=True,prices_or_g_refitted=False,
        target_thresholds=dict(mean_response_APE_max=.10,block_APE_P90_max=.20),raw_CV_is_prediction_gate=False,
        prepared_single_remains_diagnostic_out_of_lifecycle_scope=True,prepared_single_qualification_granted=False,
        interpretation_limit='same-process first warm versus later continuous estimator; fixed order and temporal device changes cannot be separated into a unique physical cause',
        whole_LLM_accuracy_validated=False)
    shared.write_exact(sidecar/'PAIRED_EVALUATION.json',result)
    return dict(output=receipt(sidecar/'PAIRED_EVALUATION.json'),summaries=summaries,paired_target_cases=len(paired))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','evaluate'))
    parser.add_argument('--out',required=True);parser.add_argument('--sidecar',required=True);args=parser.parse_args()
    result=freeze(args.out,args.sidecar) if args.command=='freeze' else evaluate(args.out,args.sidecar)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
