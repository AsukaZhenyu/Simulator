"""Preserve historical host-pause evidence; heldout is diagnosis only.

This reads an existing physical campaign and launches no hardware. Only its
PROFILE observations enter host_supply_calibration. Historical heldout latency
is summarized and labelled diagnosis_only, never supplied as model parameters
or an external availability trace.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from profiling.host_supply_calibration import (
    OBSERVATION_SCHEMA, calibrate_blackout_hypothesis, digest, require,
)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,expected=None):
    require(expected is None or sha(path)==expected,"historical file hash differs: "+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def stats(values):
    return dict(n=len(values),median=None if not values else statistics.median(values),
        minimum=None if not values else min(values),maximum=None if not values else max(values))


def constant_prediction_oracle_bound(measured_ns,relative_error):
    """Diagnostic maximum interval overlap; deliberately returns no predictor.

    A measured value t permits predictions in [(1-e)t,(1+e)t]. Maximum overlap
    is an upper bound on the success fraction of ANY one constant predictor.
    This is an oracle using observed outcomes and must never become a fitted
    prediction or an independent validation result.
    """
    require(isinstance(measured_ns,list) and measured_ns and all(type(t) is int and t>0 for t in measured_ns),
            "positive measured durations required")
    require(type(relative_error) in (int,float) and 0<=relative_error<1,"relative oracle tolerance must be in [0,1)")
    endpoints=[]
    for duration in measured_ns:
        endpoints.extend(((duration*(1-relative_error),1),(duration*(1+relative_error),-1)))
    count=maximum=0
    # Inclusive interval boundaries: entries at a timestamp precede exits.
    for _,change in sorted(endpoints,key=lambda item:(item[0],-item[1])):
        count+=change;maximum=max(maximum,count)
    return dict(schema="constant-point-prediction-oracle-bound-1",observations=len(measured_ns),
        relative_error_tolerance=relative_error,maximum_covered_observations=maximum,
        maximum_coverage=maximum/len(measured_ns),ninety_percent_coverage_possible=maximum/len(measured_ns)>=.9,
        diagnostic_only=True,not_a_model_parameter=True,selected_prediction_returned=False,
        observations_used="recorded outcomes; this oracle is never a calibration feature")


def _competitors(manifest,job,program):
    actors=[]
    for device in ('cpu','gpu'):
        if not job[device+'_background']:continue
        index=manifest['protocol']['targets'][device+'_background']
        actors.append(dict(resource_id=device+':external-replay',
            workload_signature=dict(program_physical=program['physical'],action=program['actions'][index],
                program_graph_signature=digest({k:program[k] for k in ('graph','bindings','actions','boundary_contract')})),
            placement=dict(process_affinity_mask=manifest['protocol'][device+'_background_mask'])))
    return dict(external_actors=actors,target_placement=dict(process_affinity_mask=manifest['protocol']['main_process_mask']))


def _program_signature(program):
    return digest({k:program[k] for k in ('boundary_contract','graph','physical','actions','bindings')})


def _observations(loaded):
    job,raw,execution,program,manifest=loaded['job'],loaded['raw'],loaded['execution'],loaded['program'],loaded['manifest']
    evidence=[str(loaded['raw_path'])+':sha256:'+loaded['raw_sha256']]
    context=_competitors(manifest,job,program)
    observations=[]
    for invocation in raw['iterations']:
        actions=[dict(action_index=a['action_index'],begin_ns=a['begin_ns'],end_ns=a['end_ns']) for a in invocation['actions']]
        observations.append(dict(schema=OBSERVATION_SCHEMA,observation_id=job['id']+':'+str(invocation['iteration']),
            run_id=job['id'],process_id=raw['process_id'],seed=execution['program']['seed'],split=job['split'],
            signature=_program_signature(program),resource_id='cpu:host-affinity-mask:'+str(manifest['protocol']['main_process_mask']),
            competitor_context=context,clock_id=raw['clock']['clock_id'],clock_origin=raw['clock']['clock_origin'],
            begin_ns=invocation['begin_ns'],end_ns=invocation['end_ns'],actions=actions,
            numerical_verified=execution['numerical']['main']['passed'],evidence=evidence))
    return observations


def _background(loaded):
    source=loaded['execution']['raw_files'].get('cpu-worker')
    if source is None:return None
    raw=read(source['path'],source['sha256']);hz=raw['clock']['frequency_hz']
    intervals=[(i['begin_tick']*1_000_000_000//hz,i['end_tick']*1_000_000_000//hz)
        for block in raw['blocks'] for i in block['iterations']]
    return dict(source=source,intervals=intervals,ends=[end for _,end in intervals],process_id=raw['process_id'])


def _background_overlap(background,begin,end):
    if background is None:return None
    index=bisect_left(background['ends'],begin);covered=0;calls=0;endpoints=[]
    for left,right in background['intervals'][index:]:
        if left>=end:break
        lo,hi=max(begin,left),min(end,right)
        if lo<hi:covered+=hi-lo;calls+=1;endpoints.append(dict(begin_ns=left,end_ns=right))
    return dict(source=background['source'],process_id=background['process_id'],
        window=dict(begin_ns=begin,end_ns=end),overlapping_completed_host_calls=calls,
        host_call_intervals=endpoints,host_call_union_overlap_ns=covered,host_call_overlap_fraction=covered/(end-begin),
        overlap_is_CPU_utilization=False,available_CPU_service_inferred=False)


def diagnose_historical_campaign(campaign,*,minimum_excess_ns):
    campaign=Path(campaign).resolve();manifest=read(campaign/'manifest.json')
    index=read(campaign/'posttarget-job-index.json')
    require(index['parent_manifest_sha256']==sha(campaign/'manifest.json'),"historical index parent differs")
    jobs=[j for j in index['jobs'] if j['split'] in {'profile','heldout','heldout_bare'}]
    programs={};loaded=[];sources=[receipt(campaign/'manifest.json'),receipt(campaign/'posttarget-job-index.json')]
    for job in jobs:
        path=campaign/'jobs'/job['id']/'execution.json';execution=read(path)
        require(execution['job']==job and execution['status']=='completed',"historical job not complete")
        source=execution['raw_files']['main'];raw=read(source['path'],source['sha256'])
        require(raw['program_sha256']==execution['program']['program_sha256'],"native historical program differs")
        key=job['program_key']
        if key not in programs:
            binding=manifest['programs'][key]
            programs[key]=read(binding['path'],binding['sha256']);sources.append(receipt(binding['path']))
        item=dict(job=job,execution=execution,raw=raw,raw_path=source['path'],raw_sha256=source['sha256'],
            manifest=manifest,program=programs[key])
        loaded.append(item);sources.extend((receipt(path),dict(source)))
    profile_groups=defaultdict(list)
    for item in loaded:
        if item['job']['split']!='profile':continue
        observations=_observations(item)
        key=digest(dict(signature=observations[0]['signature'],resource=observations[0]['resource_id'],
            competitor_context=observations[0]['competitor_context']))
        profile_groups[key].extend(observations)
    hypotheses=[];profile_fast={}
    for key,observations in profile_groups.items():
        baseline={i:statistics.median([o['actions'][i]['end_ns']-o['actions'][i]['begin_ns'] for o in observations])
                  for i in range(len(observations[0]['actions']))}
        profile_fast[key]=baseline
        if not observations[0]['competitor_context']['external_actors']:continue
        hypotheses.append(dict(resource_context_sha256=key,
            calibration=calibrate_blackout_hypothesis(observations,observations[0]['signature'],
                minimum_excess_ns=minimum_excess_ns),
            old_profile_only=True,used_historical_heldout_for_fit=False))
    rows=[];summary=defaultdict(list)
    for item in loaded:
        job,raw=item['job'],item['raw'];observations=_observations(item)
        key=digest(dict(signature=observations[0]['signature'],resource=observations[0]['resource_id'],
            competitor_context=observations[0]['competitor_context']))
        baseline=profile_fast[key]
        background=None;events=[]
        correlations=defaultdict(set)
        for record in raw['cupti']['records']:
            if record['kind']=='external_correlation':correlations[record['external_id']].add(record['correlation_id'])
        records_by_correlation=defaultdict(list)
        for record in raw['cupti']['records']:
            if record['kind']!='external_correlation':records_by_correlation[record['correlation_id']].append(record)
        durations=[]
        for invocation in raw['iterations']:
            duration=invocation['end_ns']-invocation['begin_ns'];durations.append(duration)
            action=max(invocation['actions'],key=lambda a:a['end_ns']-a['begin_ns']-baseline[a['action_index']])
            excess=action['end_ns']-action['begin_ns']-baseline[action['action_index']]
            if excess<minimum_excess_ns:continue
            selected=[record for correlation in correlations[action['external_id']] for record in records_by_correlation[correlation]]
            apis=[record for record in selected if record['kind'] in {'runtime_api','driver_api'}]
            biggest=max(apis,key=lambda a:a['end_ns']-a['begin_ns'],default=None)
            big=None
            if biggest is not None:
                kernels=[record for record in records_by_correlation[biggest['correlation_id']] if record['kind']=='kernel']
                big=dict(record=biggest,correlated_kernels=kernels,
                    kernel_end_relative_to_API_entry_ns=[record['end_ns']-biggest['begin_ns'] for record in kernels],
                    API_return_after_last_correlated_kernel_ns=None if not kernels else biggest['end_ns']-max(k['end_ns'] for k in kernels))
            if background is None and job['cpu_background']:background=_background(item)
            left,right=(biggest['begin_ns'],biggest['end_ns']) if biggest else (action['begin_ns'],action['end_ns'])
            events.append(dict(iteration=invocation['iteration'],invocation_begin_ns=invocation['begin_ns'],
                invocation_duration_ns=duration,dominant_action=action,profile_action_fast_wall_ns=baseline[action['action_index']],
                excess_candidate_ns=excess,largest_exactly_attributed_API=big,
                external_CPU_worker_overlap=_background_overlap(background,left,right),
                target_outcome_used_for_diagnosis_only=True))
        starts=[e['dominant_action']['begin_ns'] for e in events]
        row=dict(job_id=job['id'],split=job['split'],condition_report_label=job['condition'],
            context_sha256=key,process_id=raw['process_id'],source=dict(path=item['raw_path'],sha256=item['raw_sha256']),
            parameter_use=job['split']=='profile',historical_heldout_diagnosis_only=job['split']!='profile',
            calls=len(durations),call_duration_statistics_ns=stats(durations),
            events=events,event_count=len(events),event_action_entry_intervals_ns=[b-a for a,b in zip(starts,starts[1:])],
            constant_prediction_oracle=constant_prediction_oracle_bound(durations,.20))
        rows.append(row);summary[(job['split'],job['condition'])].extend(durations)
    result=dict(schema='historical-host-blackout-diagnosis-1',campaign_root=str(campaign),
        sources=list({r['path']:r for r in sources}.values()),producer=receipt(__file__),
        calibration_module=receipt(Path(__file__).with_name('host_supply_calibration.py')),
        declared_minimum_excess_ns=minimum_excess_ns,profile_only_hypotheses=hypotheses,jobs=rows,
        cohort_oracle_diagnostics=[dict(split=split,condition_report_label=state,
            oracle=constant_prediction_oracle_bound(durations,.20),not_a_model_parameter=True)
            for (split,state),durations in summary.items()],
        historical_heldout_used_to_fit_parameters=False,new_hardware_experiment_performed=False,
        CPU_available_service_identified=False,OS_scheduler_simulated=False,pause_cause_proven=False,
        hardware_accuracy_qualified=False,
        limitations=['Historical heldout data is diagnosis only, not calibration or independent acceptance for a new model.',
            'Background host-call overlap is not CPU active time or availability.',
            'Kernel finishing early within a long API localizes delay but does not prove its OS/runtime cause.',
            'No phase or neutral CPU service is derived from target stalls.'])
    result['artifact_sha256']=digest(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--minimum-excess-ns',required=True,type=int)
    args=parser.parse_args();result=diagnose_historical_campaign(args.campaign,minimum_excess_ns=args.minimum_excess_ns)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(args.output.resolve()),sha256=sha(args.output),jobs=len(result['jobs']),
        profile_only_hypotheses=len(result['profile_only_hypotheses']),new_hardware_experiment_performed=False)))
