"""Completed Attention-component prices and independent planned-state tests.

This separate protocol uses only stamped complete component observations. It
does not reuse unqualified callback-stage prices, fit a whole LLM target, or
adjust measured validation trajectories. The common modeling event engine owns
the response integration and recursive completion clock.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from llm_infer_model.performance.planned_response import PlannedCompletedResponse
from llm_infer_model.performance.reusable_action_costs import digest

from . import bare_macro_campaign as campaign

read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
STATES=((0,0),(1,0),(0,1),(1,1))
PROVIDERS=('static00','memoryless_four_state_macro')
HORIZON_NS=300000000000


def write_exact(path,value):
    from profiling.dac_validation.e2a import number
    write(path,number(value))


def check_bound_files(manifest):
    for item in manifest['sources']:
        for key in ('source','archived'):
            require(sha(item[key]['path'])==item[key]['sha256'],'bound source changed: '+item[key]['path'])
    for item in (manifest['binary'],*manifest['runtime']):
        require(sha(item['path'])==item['sha256'],'bound binary/runtime changed')
    for binding in manifest['programs'].values():
        for key in ('program','reference'):
            require(sha(binding[key]['path'])==binding[key]['sha256'],'bound mathematical input changed')


def no_heldout(out,manifest):
    require(not (out/'heldout-started.json').exists()
            and not any((out/'jobs'/j['id']).exists() for j in manifest['jobs'] if j['split']=='heldout'),
            'all component prices and forecasts must be frozen before heldout collection')


def component_label(native):
    """One completed component interval; never the complete target-wall label."""
    actions=[a for block in native['blocks'] for a in block['actions']]
    require(len(actions)==1 and actions[0]['action_index']==22 and native['continuous_replays']==1,
            'calibration requires exactly one declared complete Attention component')
    action=actions[0];hz=native['clock']['frequency_hz']
    require(type(hz) is int and hz>0 and type(action['begin_tick']) is int
            and type(action['end_tick']) is int and action['end_tick']>action['begin_tick'],
            'positive complete component QPC interval required')
    value=Fraction((action['end_tick']-action['begin_tick'])*10**9,hz)
    require(abs(value-action['wall_ns'])<=1,'component QPC/response label differs')
    return value


def calibrated_cells(out,manifest):
    jobs=[j for j in manifest['jobs'] if j['split']=='profile']
    require(len(jobs)==12,'all12 declared calibration contexts required, including all3 retained OFF controls')
    grouped=defaultdict(list);identities=set();hashes=set();evidence=[]
    for job in jobs:
        native,execution,path=campaign.verified_job(out,job,manifest)
        state=job.get('offered_state',job['request'].get('offered_state'))
        require(isinstance(state,(list,tuple)) and len(state)==2
                and all(type(x) is int and x in (0,1) for x in state),'explicit calibration offered state required')
        state=tuple(state)
        require(state==tuple(native['request']['offered_state']) and native['instrumentation']=='stamped'
                and native['runtime_callbacks']['enabled'] is False
                and native['runtime_callbacks']['record_count']==0,
                'only matched stamped complete-component labels are supported; callback-stage prices forbidden')
        identity=(execution['process_id'],execution.get('process_creation_time_100ns'))
        require(type(identity[1]) is int and identity[1]>0 and identity not in identities,
                'distinct verified PID+creation identities required for all12 calibration processes')
        require(execution['native']['sha256'] not in hashes,'same native record reused as another context')
        identities.add(identity);hashes.add(execution['native']['sha256'])
        label=component_label(native)
        row=dict(job_id=job['id'],state=state,response_ns=label,imported=job.get('imported',False),
                 process_identity=identity,native=execution['native'],execution=receipt(path))
        grouped[state].append(row);evidence.append(execution['native']['path']+'#sha256='+execution['native']['sha256'])
    require(set(grouped)==set(STATES) and all(len(grouped[s])==3 for s in STATES),
            'three declared fresh complete-component observations per state required; missing is not zero')
    cells=[]
    for state in STATES:
        labels=[row['response_ns'] for row in grouped[state]];mean=sum(labels,Fraction(0))/3
        cells.append(dict(state=state,mean_ns=mean,observations=grouped[state],
            fresh_contexts=3,raw_profile_CV=statistics.pstdev([float(x) for x in labels])/float(mean),
            three_component_observations_establish_low_variance=False))
    return cells,tuple(evidence)


def predict(cells,scope,request,provider,evidence):
    require(provider in PROVIDERS,'unknown predeclared component provider')
    require(len(cells)==4 and {tuple(c['state']) for c in cells}==set(STATES),
            'complete four-state component prices required')
    prices={tuple(c['state']):Fraction(c['mean_ns']) for c in cells}
    require(all(x>0 for x in prices.values()),'explicit positive component responses required')
    require(digest(scope['context'])==scope['context_sha256']
            and digest(scope['semantic_coordinate'])==scope['semantic_sha256'],
            'component/runtime semantic scope digest differs')
    for device in ('cpu','gpu'):
        actor={k:v for k,v in request[device+'_actor'].items() if k!='intervals_ms'}
        require(actor==scope['context']['actors'][device],
                'target actor workload/placement differs from exact calibrated scope')
    intervals=lambda device:tuple(tuple(value*1000000 for value in pair)
                                 for pair in request[device+'_actor']['intervals_ms'])
    key=scope['semantic_sha256']
    law=PlannedCompletedResponse(scope['context_sha256'],{key:dict(kind='RUNTIME_RESPONSE',states=prices)},(key,),
        intervals('cpu'),intervals('gpu'),provider='static' if provider=='static00' else 'memoryless',
        evidence=tuple(evidence),origin_ns=0,horizon_ns=HORIZON_NS)
    result=law.predict_replays(replays=request['replays'],block_replays=request['block_replays'],start_ns=0)
    return dict(result,component_scope='one complete original Executor.action Attention graph',
        API_or_gap_callback_prices_used=False,whole_LLM_target_time_used_in_fit=False,
        tau_or_transition_coefficient_estimated=False,OS_scheduler_simulated=False,
        internal_GPU_or_DMA_time_added=False,active_CPU_time_inferred=False)


def fit_and_freeze(out):
    out=Path(out).resolve();manifest=campaign.check(out);check_bound_files(manifest);no_heldout(out,manifest)
    require(not (out/'component-profile.json').exists() and not (out/'frozen-barrier.json').exists(),
            'component profile/forecast is one frozen operation; no adaptive refitting')
    cells,evidence=calibrated_cells(out,manifest);scope=manifest['scope']
    profile=dict(schema='bare-Attention-component-profile-1',cells=cells,evidence=evidence,scope=scope,
        manifest=receipt(out/'manifest.json'),calibration_target_is_component=True,
        whole_LLM_target_time_used_for_fit=False,callback_stage_prices_used=False,
        raw_observations_removed=0,imported_OFF_policy='all three pre-existing stamped OFF controls; no selective reuse',
        hardware_accuracy_qualified=False)
    write_exact(out/'component-profile.json',profile)
    held=[j for j in manifest['jobs'] if j['split']=='heldout'];require(len(held)==6,'six declared independent heldout processes required')
    forecasts=[];cache={}
    for job in held:
        for provider in PROVIDERS:
            cache_key=(digest(job['request']),provider)
            if cache_key not in cache:cache[cache_key]=predict(cells,scope,job['request'],provider,evidence)
            forecasts.append(dict(job_id=job['id'],provider=provider,pulse_plan=job['pulse_plan'],prediction=cache[cache_key],
                primary_planned_entry_ns=0,entry_controller_max_lateness_ns=10000000,
                observed_entry_or_internal_boundaries_used=False))
    write_exact(out/'frozen-predictions.json',dict(schema='bare-Attention-component-forecasts-1',rows=forecasts,
        declared_provider_process_cells=12,providers=PROVIDERS,primary_provider='memoryless_four_state_macro',
        planned_entry_ns=0,all_predictions_before_heldout_collection=True,
        actual_target_times_used=False,whole_LLM_target_time_used_for_fit=False,
        model='modeling.PlannedCompletedResponse integrating exact four-state component responses on own completion clock'))
    barrier=dict(schema='bare-Attention-component-freeze-1',manifest_sha256=sha(out/'manifest.json'),
        profile_supported=True,parameters_frozen=True,predictions_frozen=True,
        calibration_target_is_component=True,whole_LLM_target_time_used_for_fit=False,
        files={name:sha(out/name) for name in ('component-profile.json','frozen-predictions.json')})
    write(out/'frozen-barrier.json',barrier)
    return dict(profile=receipt(out/'component-profile.json'),barrier=receipt(out/'frozen-barrier.json'),
                component_cells=4,calibration_contexts=12,heldout_forecasts=12,hardware_accuracy_validated=False)


def verified_prediction_map(out,manifest):
    barrier=read(out/'frozen-barrier.json')
    require(barrier['manifest_sha256']==sha(out/'manifest.json') and barrier['profile_supported'] is True
            and barrier['parameters_frozen'] is True and barrier['predictions_frozen'] is True
            and barrier['calibration_target_is_component'] is True
            and barrier['whole_LLM_target_time_used_for_fit'] is False,'pre-acquisition exact prediction freeze required')
    for name,value in barrier['files'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts and sha(out/path)==value,'frozen predictor changed')
    forecast=read(out/'frozen-predictions.json');held=[j for j in manifest['jobs'] if j['split']=='heldout']
    predictions={(r['job_id'],r['provider']):r for r in forecast['rows']}
    require(len(held)==6 and len(forecast['rows'])==12
            and set(predictions)=={(j['id'],p) for j in held for p in PROVIDERS},'all12 exact provider/process forecasts required')
    for row in predictions.values():
        prediction=row['prediction'];blocks=prediction['block_ns']
        require(len(blocks)==16 and all(type(v) is int and v>=0 for v in blocks)
                and type(prediction['total_ns']) is int and sum(blocks)==prediction['total_ns'],
                'frozen prediction clock/total/block inventory differs')
    return held,predictions


def measured_intervals(native):
    hz=native['clock']['frequency_hz'];require(type(hz) is int and hz>0,'actual positive QPC frequency required')
    total=native['target_wall_ns'];blocks=[b['wall_ns'] for b in native['blocks']]
    require(type(total) is int and total>0 and len(blocks)==16 and all(type(v) is int and v>0 for v in blocks),
            'positive complete target and16 block times required')
    require(abs(Fraction((native['target_end_tick']-native['target_begin_tick'])*10**9,hz)-total)<=1,
            'complete measured target clock does not close')
    require(all(abs(Fraction((b['end_tick']-b['begin_tick'])*10**9,hz)-b['wall_ns'])<=1 for b in native['blocks']),
            'measured block clock does not close')
    lateness=Fraction((native['target_begin_tick']-native['planned_target_entry_tick'])*10**9,hz)
    require(0<=lateness<=10000000 and abs(lateness-native['initial_start_lateness_ns'])<=1,
            'actual target entry violated the predeclared10ms uncertainty envelope')
    actions=[a for b in native['blocks'] for a in b['actions']]
    require(len(actions)==256 and all(a['action_index']==22 for a in actions),'256 complete original Attention invocations required')
    shell=total-sum(a['wall_ns'] for a in actions)
    require(shell>=0,'target interval is smaller than its disjoint completed actions')
    return total,blocks,dict(initial_start_lateness_ns=native['initial_start_lateness_ns'],
        outside_action_host_shell_ns=shell,outside_action_host_shell_fraction=shell/total,
        outside_action_shell_used_for_prediction=False)


def evaluate(out):
    out=Path(out).resolve();manifest=campaign.check(out);check_bound_files(manifest)
    held,predictions=verified_prediction_map(out,manifest);rows=[];processes=[]
    for job in held:
        try:
            native,execution,path=campaign.verified_job(out,job,manifest)
            total,blocks,details=measured_intervals(native)
            processes.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],status='completed',
                native=execution['native'],execution=receipt(path),measured_total_ns=total,measured_block_ns=blocks,
                temporal_block_CV=statistics.pstdev(blocks)/statistics.mean(blocks),
                temporal_block_CV_is_stationary_noise=False,**details))
        except (ValueError,KeyError,FileNotFoundError) as error:
            processes.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],status='unsupported_or_not_completed',reason=repr(error)))
            for provider in PROVIDERS:
                rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,
                                 status='unsupported_or_not_completed',accuracy_passed=False,reason=repr(error)))
            continue
        for provider in PROVIDERS:
            prediction=predictions[job['id'],provider]['prediction'];pred=prediction['total_ns']
            block_errors=[abs(p/m-1) for p,m in zip(prediction['block_ns'],blocks)]
            p90=sorted(block_errors)[14]  # np.quantile(method='higher', p=.9),16 fixed blocks.
            error=abs(pred/total-1)
            rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,status='evaluated',
                predicted_total_ns=pred,measured_total_ns=total,predicted_block_ns=prediction['block_ns'],measured_block_ns=blocks,
                mean_response_APE=error,signed_bias=(pred-total)/total,absolute_error_ns=abs(pred-total),
                block_APEs=block_errors,block_APE_P90=p90,accuracy_passed=error<=.10 and p90<=.20,
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
        repeats.append(dict(pulse_plan=plan,declared_fresh_processes=3,completed_fresh_processes=len(values),
            total_samples_ns=values,raw_fresh_process_CV=statistics.pstdev(values)/statistics.mean(values) if len(values)==3 else None,
            raw_CV_is_prediction_gate=False))
    result=dict(schema='bare-Attention-component-independent-evaluation-1',rows=rows,processes=processes,
        providers=summaries,repeat_variability=repeats,declared_provider_process_cells=12,
        primary_provider='memoryless_four_state_macro',primary_all_six_passed=next(r['all_six_passed'] for r in summaries if r['provider']=='memoryless_four_state_macro'),
        thresholds=dict(mean_response_APE_max=.10,block_APE_P90_max=.20),raw_trials_or_slow_blocks_removed=0,
        no_refit=True,whole_LLM_dynamic_accuracy_validated=False,
        raw_CV_is_prediction_gate=False,actual_start_or_internal_boundaries_used_in_prediction=False,
        entry_uncertainty='planned start0, owned target controller permits at most10ms lateness; observed start never reanchors forecast',
        limits='one complete fixed-shape Attention component, owned GEMM actors and exact host-core placement; no physical OS/queue/recovery identification')
    write_exact(out/'INDEPENDENT_EVALUATION.json',result)
    return dict(output=receipt(out/'INDEPENDENT_EVALUATION.json'),primary_all_six_passed=result['primary_all_six_passed'],providers=summaries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('fit-freeze','evaluate'));parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(fit_and_freeze(args.out) if args.command=='fit-freeze' else evaluate(args.out),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
