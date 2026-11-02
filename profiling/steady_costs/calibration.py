"""Direct steady action means, common-Graph composition and fresh validation.

No residual model, interpolation, state predictor or end-to-end offset is fit.
Device-event/enqueue/wait diagnostics are retained separately, never added to
an inclusive complete-action price.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path
import statistics

from llm_infer_model.performance.steady_action_costs import (
    SteadyActionKey,SteadyActionCost,SteadyActionTable,compile_steady_actions,
)
from .campaign import fraction,mean,pack,read,receipt,stats,utc,verify_manifest,write_new
from .contracts import digest,require,semantic_program,sha


def load_job(root,manifest,job):
    path=Path(root)/'jobs'/job['id']/'execution.json';execution=read(path)
    require(execution.get('job')==job and execution.get('status')=='completed','job is not complete: '+job['id'])
    require(execution.get('returncode')==0 and execution.get('forced_termination') is False and
        execution.get('process_exited_verified') is True,'owned process did not exit normally')
    require(execution.get('numerical',{}).get('passed') is True,'independent numerical receipt missing')
    require(execution.get('resource_state_used_as_model_input') is False,'target state must not feed its prediction')
    require(execution['program']==manifest['programs'][job['program_key']],'program provenance differs')
    source=execution['native'];require(sha(source['path'])==source['sha256'],'native raw bytes changed')
    native=read(source['path'])
    require(native.get('process_id')==execution['process_id'] and native.get('request')==job['request'],'native identity/request differs')
    require(native.get('program_sha256')==read(execution['program']['program']['path'])['program_sha256'],'native program differs')
    require(len(native['trials'])==job['request']['trials'],'trial coverage differs')
    return execution,native


def action_key(manifest,binding,program,request,index=None):
    if index is None:
        action_signature=digest(dict(kind=program['kind'],device=program['device'],shape=program['shape']))
        backend=program['device'];boundary='compact GGML completed component action'
    else:
        action=program['actions'][index]
        action_signature=digest(dict(original_action_index=index,action=action))
        backend=action.get('device') or ('copy:'+action['src']+'->'+action['dst'] if action['kind']=='COPY' else 'host_dispatch')
        boundary='unchanged Executor.action in complete serial full-stamped context'
    hw=manifest.get('hardware',{});gpu=hw.get('gpu_nvml_identity',{})
    identity=dict(os=hw.get('os'),processor_identifier=hw.get('processor_identifier'),machine_architecture=hw.get('machine_architecture'),
        logical_processors=hw.get('logical_processors'),processor_topology=hw.get('logical_to_physical_processor_topology'),gpu_uuid=gpu.get('device_uuid'),
        gpu_name=gpu.get('device_name'),gpu_driver=gpu.get('driver_version'))
    scope=dict(boundary=boundary,binary_sha256=manifest['sources'][manifest['binary']],hardware_identity=identity,
               backend_environment=manifest.get('inherited_backend_environment',{}),dtype='F32',graphs='disabled',poll='0',main_thread_mask=4,
               repeated_fixed_input_working_set=True,batch_repeats=request['repeats'],measurement_context=request['mode'],
               warmup_min_ms=request['warmup_min_ms'],single_cold_call_or_streaming_cache_claimed=False,
               shape_and_strategy_exact=True,external_owned_background_actors=[])
    return SteadyActionKey(action_signature,backend,binding['signature'],digest(scope))


def freeze(root):
    root=Path(root).resolve();manifest=verify_manifest(root)
    require(not(root/'heldout-started.json').exists(),'cannot freeze after heldout acquisition starts')
    jobs=[j for j in manifest['jobs'] if j['split']=='profile']
    entries=[];case_keys={};processes=[];sources=[receipt(root/'manifest.json')]
    for case in manifest['cases']:
        selected=[j for j in jobs if j['case_id']==case['id']]
        require(len(selected)==3,'three fresh PROFILE processes required per exact case')
        loaded=[load_job(root,manifest,j) for j in selected]
        pids=tuple(e['process_id'] for e,n in loaded)
        require(len(set(pids))==3,'PROFILE process identities must be distinct')
        processes.extend(dict(job_id=j['id'],process_id=e['process_id'],seed=e['program']['seed']) for j,(e,n) in zip(selected,loaded))
        for j,(execution,native) in zip(selected,loaded):
            sources.extend((receipt(root/'jobs'/j['id']/'execution.json'),execution['native']))
        binding=manifest['programs'][selected[0]['program_key']];program=read(binding['program']['path'])
        count=len(program['actions']) if case['kind']=='layer' else 1
        keys=[]
        for index in range(count):
            key=action_key(manifest,binding,program,selected[0]['request'],index if case['kind']=='layer' else None)
            values=[]
            for execution,native in loaded:
                for trial in native['trials']:
                    if case['kind']=='layer':
                        require(len(trial['actions'])==count and trial['actions'][index]['action_index']==index,'PROFILE action coverage differs')
                        elapsed=trial['actions'][index]['wall_ns']
                    else:elapsed=trial['wall_ns']
                    values.append(Fraction(elapsed,trial['repeats']))
            value=mean(values);require(value is not None,'empty PROFILE costs must remain unknown')
            evidence=tuple(e['native']['path']+':sha256:'+e['native']['sha256'] for e,n in loaded)
            entries.append(SteadyActionCost(key,value,evidence,len(values),pids,zero_duration_approximation=value==0))
            keys.append(key.to_dict())
        case_keys[case['id']]=keys
    table=SteadyActionTable(tuple(entries))
    parameters=dict(schema='steady-cost-frozen-1',frozen_at_utc=utc(),manifest_sha256=sha(root/'manifest.json'),
        source_inventory_sha256=manifest['source_inventory_sha256'],table=table.to_dict(),case_keys=case_keys,
        source_receipts=sources,profile_processes=processes,heldout_measurements_read=False,whole_layer_offset_fitted=False,
        state_response_fitted=False,unknown_interpolation=False,device_diagnostics_added=False,
        cost_fit='arithmetic mean of all trial-total/repeats observations from three independent PROFILE processes',
        measurements_are_batch_averages=True,quality=manifest['protocol']['quality'],hardware_accuracy_qualified=False)
    parameters['artifact_sha256']=digest(parameters)
    write_new(root/'profile-frozen.json',parameters)
    barrier=dict(schema='steady-cost-frozen-barrier-1',frozen_at_utc=parameters['frozen_at_utc'],
        manifest_sha256=parameters['manifest_sha256'],frozen_file=str(root/'profile-frozen.json'),
        frozen_file_sha256=sha(root/'profile-frozen.json'),heldout_measurements_read_before_freeze=False)
    write_new(root/'frozen-barrier.json',barrier)
    return dict(frozen_file=barrier['frozen_file'],sha256=barrier['frozen_file_sha256'],entries=len(entries),profile_processes=len(processes))


def restore_table(document):
    return SteadyActionTable(tuple(SteadyActionCost(SteadyActionKey(**row['key']),
        None if row['mean_ns'] is None else fraction(row['mean_ns']),tuple(row['evidence']),row['sample_count'],tuple(row['process_ids']),
        None if row['device_interval_ns'] is None else fraction(row['device_interval_ns']),row['zero_duration_approximation'],
        row['device_interval_scope'],tuple(row['device_interval_evidence'])) for row in document['entries']))


def errors(predicted,observed):
    observed=list(map(Fraction,observed))
    require(observed and all(v>0 for v in observed),'nonpositive or empty observed trial means')
    predicted=Fraction(predicted);values=[abs(predicted-v)/v for v in observed]
    return dict(median_ape=float(statistics.median(values)),p90_ape=float(sorted(values)[math.ceil(.9*len(values))-1]),
        worst_ape=float(max(values)),mean_ape=float(mean(values)),count=len(values))


def evaluate(root):
    root=Path(root).resolve();manifest=verify_manifest(root)
    frozen=read(root/'profile-frozen.json');barrier=read(root/'frozen-barrier.json')
    require(frozen['artifact_sha256']==digest({k:v for k,v in frozen.items() if k!='artifact_sha256'}),'frozen semantic hash differs')
    require(sha(root/'profile-frozen.json')==barrier['frozen_file_sha256'] and sha(root/'manifest.json')==barrier['manifest_sha256'],'freeze bytes changed')
    require(read(root/'heldout-started.json')['barrier']==barrier,'heldout was not launched with this barrier')
    for source in frozen['source_receipts']:require(sha(source['path'])==source['sha256'],'frozen profile evidence changed')
    table=restore_table(frozen['table']);results=[];profiles={r['seed'] for r in frozen['profile_processes']};profile_pids={r['process_id'] for r in frozen['profile_processes']}
    quality=frozen['quality']
    for case in manifest['cases']:
        keys=tuple(SteadyActionKey(**k) for k in frozen['case_keys'][case['id']])
        model=compile_steady_actions(keys,table);prediction=model.evaluate().duration_ns
        own=[];all_observed=[]
        for job in [j for j in manifest['jobs'] if j['split']=='heldout' and j['case_id']==case['id']]:
            execution,native=load_job(root,manifest,job)
            binding=manifest['programs'][job['program_key']];program=read(binding['program']['path'])
            require(digest(semantic_program(program))==binding['signature'],'heldout shape/strategy signature differs')
            # Full-stamped PROFILE prices apply to bare full replay only under
            # the separately reported observer gate. All other boundaries,
            # including repetitions and prepared-input shape, must be exact.
            application_request=dict(job['request'],mode='full-stamped' if case['kind']=='layer' else 'action')
            application_keys=tuple(action_key(manifest,binding,program,application_request,index if case['kind']=='layer' else None)
                for index in range(len(program['actions']) if case['kind']=='layer' else 1))
            require(application_keys==keys,'heldout exact cost scope differs from frozen PROFILE keys')
            require(execution['program']['seed'] not in profiles,'heldout numerical seed overlaps PROFILE')
            require(execution['process_id'] not in profile_pids,'heldout process identity overlaps PROFILE')
            require(execution['started_at_utc']>=barrier['frozen_at_utc'],'heldout launch predates frozen prices')
            observed=[Fraction(t['wall_ns'],t['repeats']) for t in native['trials']]
            all_observed.extend(observed);average=mean(observed)
            err=float(abs(Fraction(prediction)-average)/average)
            summary=stats(observed)
            own.append(dict(job_id=job['id'],process_id=execution['process_id'],predicted_per_call_mean_ns=prediction,
                observed_trial_average_ns=pack(observed),observed_statistics=summary,aggregate_mean_ape=err,trial_average_errors=errors(prediction,observed),
                mean_gate=err<=quality['max_process_mean_ape'],trial_cv_gate=summary['cv'] is not None and summary['cv']<=quality['max_process_trial_cv']))
        require(len(own)==3,'heldout process coverage incomplete')
        require(len({p['process_id'] for p in own})==3,'heldout process identities must be distinct')
        summary=errors(prediction,all_observed)
        observer=[]
        if case['kind']=='layer':
            for repeat in range(3):
                ps=next(j for j in manifest['jobs'] if j['case_id']==case['id'] and j['split']=='profile' and j['repeat']==repeat)
                ob=next(j for j in manifest['jobs'] if j['case_id']==case['id'] and j['split']=='observer' and j['repeat']==repeat)
                pe,pn=load_job(root,manifest,ps);oe,on=load_job(root,manifest,ob)
                pm=mean(Fraction(t['wall_ns'],t['repeats']) for t in pn['trials']);om=mean(Fraction(t['wall_ns'],t['repeats']) for t in on['trials'])
                effect=float(pm/om-1)
                observer.append(dict(repeat=repeat,full_stamped_mean_ns=pack(pm),full_bare_mean_ns=pack(om),relative_difference=effect,
                    passed=abs(effect)<=quality['max_observer_relative_mean_difference']))
        passed=all(p['mean_gate'] for p in own) and summary['p90_ape']<=quality['max_trial_average_p90_ape']
        observer_gate=all(o['passed'] for o in observer) if observer else None
        results.append(dict(case_id=case['id'],kind=case['kind'],prediction_ns=prediction,processes=own,trial_average_errors=summary,
            observed_statistics=stats(all_observed),mean_and_trial_p90_gate=passed,observer_controls=observer,
            observer_gate=observer_gate,stability_gate=all(p['trial_cv_gate'] for p in own),
            hardware_accuracy_qualified=bool(passed and all(p['trial_cv_gate'] for p in own) and observer_gate is not False)))
    diagnostics=[]
    for job in [j for j in manifest['jobs'] if j['split']=='diagnostic']:
        execution,native=load_job(root,manifest,job)
        diagnostics.append(dict(job_id=job['id'],role=job['role'],request=job['request'],native=execution['native'],
            wall_per_call_ns=stats(Fraction(t['wall_ns'],t['repeats']) for t in native['trials']),
            intervals={field:stats(Fraction(t[field],t['repeats']) for t in native['trials'] if t.get(field) is not None)
                for field in ('enqueue_ns','sync_ns','stream_elapsed_ns')},diagnostic_intervals_added_to_price=False))
    result=dict(schema='steady-cost-validation-1',evaluated_at_utc=utc(),frozen_file_sha256=barrier['frozen_file_sha256'],
        cases=results,diagnostics=diagnostics,heldout_prices_fitted=False,target_state_used_as_prediction_input=False,
        external_resource_trajectory_estimated=False,all_trials_retained=True,individual_call_tail_accuracy_claimed=False,
        quality=quality,passed_cases=sum(c['hardware_accuracy_qualified'] for c in results),total_cases=len(results),
        all_cases_passed=all(c['hardware_accuracy_qualified'] for c in results),capacity_qualified=False)
    write_new(root/'evaluation.json',result)
    return dict(path=str(root/'evaluation.json'),passed_cases=result['passed_cases'],total_cases=result['total_cases'],
        cases=[dict(case_id=c['case_id'],mean_errors=[p['aggregate_mean_ape'] for p in c['processes']],p90=c['trial_average_errors']['p90_ape'],
                    passed=c['hardware_accuracy_qualified']) for c in results])
