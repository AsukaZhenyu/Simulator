"""Local API/gap calibration and target-blind forecasts for Attention v8.

The only response labels are the individual disjoint runtime API/gap intervals
inside profile calls. Complete Attention times are used for observer gates and
independent scoring, never as fitting labels or correction multipliers. All
time integration is delegated to modeling, including the coarse macro baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import statistics

from llm_infer_model.performance.planned_response import PlannedCompletedResponse
from llm_infer_model.performance.reusable_action_costs import digest, semantic_action

from . import external_stage_v8 as campaign
from .runtime_stage_profile import aggregate_cells, profile_observations, to_model_objects
from .runtime_stage_records import coordinate_sha256

ROOT=campaign.ROOT
read,write,sha,receipt,require=campaign.read,campaign.write,campaign.sha,campaign.receipt,campaign.require
PROVIDERS=('static00','macro_four_state_sum_of_same_stage_prices','ordered_API_gap_four_state')
STATES=((0,0),(1,0),(0,1),(1,1))
HORIZON_NS=300000000000


def write_exact(path,value):
    write(path,campaign.base.e2a.number(value))


def check_sources(manifest):
    for item in manifest['sources']:
        for key in ('source','archived'):
            require(sha(item[key]['path'])==item[key]['sha256'],'bound source changed: '+item[key]['path'])
    for item in (manifest['binary'],*manifest['runtime']):
        require(sha(item['path'])==item['sha256'],'bound binary/runtime changed')
    for binding in manifest['programs'].values():
        for key in ('program','reference'):
            require(sha(binding[key]['path'])==binding[key]['sha256'],'bound input/reference changed')


def no_heldout(out,manifest):
    require(not (out/'heldout-started.json').exists()
            and not any((out/'jobs'/j['id']).exists() for j in manifest['jobs'] if j['split']=='heldout'),
            'profile estimation/forecast freeze after heldout acquisition is forbidden')


def semantic_template(manifest):
    profile=read(manifest['programs']['profile']['program']['path'])
    heldout=read(manifest['programs']['heldout']['program']['path'])
    require(profile['program_sha256']!=heldout['program_sha256'],'heldout mathematical seed must be independent')
    target=semantic_action(profile,22)
    require(target==semantic_action(heldout,22),'heldout Attention semantic shape/dtype/device/template differs')
    return dict(target=target,semantic_sha256=digest(target),
                profile_program_sha256=profile['program_sha256'],heldout_program_sha256=heldout['program_sha256'],
                heldout_actual_API_sequence_observed_before_execution=False,
                reuse_scope='same compiled runtime and declared shape/dtype/Attention template; independent values/seed')


def source_proof(manifest,suffix):
    matching=[x for x in manifest['sources']
              if str(x['source']['path']).replace('\\','/').endswith(suffix)]
    require(len(matching)==1,'unique archived implementation source required: '+suffix)
    item=matching[0]
    require(sha(item['source']['path'])==item['source']['sha256']
            and sha(item['archived']['path'])==item['archived']['sha256'],'source proof changed')
    return item


def stream_binding(native,manifest):
    """One observed target stream role; never infer default/PTDS semantics."""
    callbacks=native['runtime_callbacks']
    require(callbacks['enabled'] is True and callbacks['target_thread_id']==native['main_thread_id']
            and callbacks['foreign_actor_threads_filtered_before_recording'] is True
            and callbacks['record_serialization_after_target_and_actor_join'] is True,
            'target-thread/phase-filtered actual callbacks required')
    handles={m['stream_handle_opaque_process_local'] for m in callbacks['metadata']}
    require(len(handles)==1 and all(type(h) is int for h in handles),
            'ambiguous/multiple target streams require an explicit new stream-role contract')
    observer=source_proof(manifest,'/native/v6/runtime_callbacks.hpp')
    target=source_proof(manifest,'/native/v8/external_stage_probe.cpp')
    require(observer['source']['sha256']==native['callback_source_sha256']
            and target['source']['sha256']==native['source_sha256'],'actual source/native role identity differs')
    # Bind the observed role to source that filters the target thread and arms
    # only around the one completed Attention graph. No opaque handle is a
    # cross-process identity, and the role makes no PTDS/default-stream claim.
    observer_text=Path(observer['archived']['path']).read_text(encoding='utf-8')
    target_text=Path(target['archived']['path']).read_text(encoding='utf-8')
    require('GetCurrentThreadId()!=self->thread_' in observer_text
            and 'if(!self->phase_)return' in observer_text
            and 'callbacks.arm(uint64_t(index+1))' in target_text
            and 'e.action(actions.at(attention))' in target_text,
            'source-bound target-only stream role evidence missing')
    handle=next(iter(handles));role='sole_observed_target_Attention_runtime_stream'
    return {handle:role},dict(observed_process_local_handle=handle,role=role,
        unique_actual_target_stream=True,opaque_handle_is_cross_process_identity=False,
        PTDS_or_default_stream_semantics_inferred=False,observer_source=observer,target_source=target)


def scope_binding(native,execution,execution_path,manifest,template):
    runtime_identity=dict(binary_sha256=manifest['binary']['sha256'],
        runtime=[dict(name=Path(x['path']).name,sha256=x['sha256']) for x in manifest['runtime']])
    runtime_hash=digest(runtime_identity)
    context=dict(runtime_binding_sha256=runtime_hash,hardware=manifest['hardware'],
        native_source_sha256=manifest['source_sha256'],callback_source_sha256=manifest['callback_source_sha256'],
        actor_source_sha256=manifest['actor_source_sha256'],Attention_template=template['target'],
        preparation='all actors initialized; pinned init; full warmup and completed full prime; observer subscribed before future epoch',
        boundary='inclusive_host_response_partition',process_mask=4294967295,
        target_mask=4,CPU_actor_mask=4,GPU_actor_submission_mask=65536)
    topology=manifest['hardware'];selected=topology['selected']
    require(selected['2']['mask']==4 and selected['16']['mask']==65536,
            'verified physical core mapping does not match actual target/actor masks')
    return dict(native_sha256=execution['native']['sha256'],execution_sha256=sha(execution_path),
        runtime_binding_sha256=runtime_hash,context_sha256=digest(context),topology_sha256=digest(topology),
        process_id=execution['process_id'],process_creation_time_100ns=execution['process_creation_time_100ns'],
        target_physical_cores=(selected['2']['physical_core_record'],),
        CPU_actor_physical_cores=(selected['2']['physical_core_record'],),
        GPU_actor_physical_cores=(selected['16']['physical_core_record'],))


def verified_native(out,job,manifest):
    path=out/'jobs'/job['id']/'execution.json';execution=read(path)
    require(execution['status']=='completed' and execution['process_exited_verified'] is True
            and execution['returncode']==0,'all declared owned processes must complete; no retries/substitution')
    require(sha(execution['native']['path'])==execution['native']['sha256'],'retained native file changed')
    native=read(execution['native']['path'])
    campaign.verify(native,job['request'],manifest['programs'][job['split']],execution['process_id'])
    return native,execution,path


def profile_datasets(out,manifest):
    """Validate 24 processes, with 12 callback contexts and 12 matched controls."""
    template=semantic_template(manifest);datasets=[];native_evidence=[];pairs={}
    jobs=[j for j in manifest['jobs'] if j['split']=='profile']
    require(len(jobs)==24,'exact predeclared 24 profile process inventory required')
    callback_counts=Counter();identities=set();native_hashes=set()
    for job in jobs:
        native,execution,path=verified_native(out,job,manifest)
        identity=(execution['process_id'],execution.get('process_creation_time_100ns'))
        require(type(identity[1]) is int and identity[1]>0 and identity not in identities,
                'all 24 observer/control processes require distinct verified PID+creation identities')
        require(execution['native']['sha256'] not in native_hashes,'native record reused under another job label')
        identities.add(identity);native_hashes.add(execution['native']['sha256'])
        native_evidence.append(dict(job_id=job['id'],execution=receipt(path),native=execution['native']))
        pair=pairs.setdefault(job['pair_id'],{})
        require(job['request']['instrumentation'] not in pair,'duplicate observer pair role')
        pair[job['request']['instrumentation']]=native['target_wall_ns']
        if job['request']['instrumentation']!='runtime_callbacks':continue
        declared=tuple(job['offered_state']);require(declared in STATES,'unknown offered state')
        callback_counts[declared]+=1
        roles,proof=stream_binding(native,manifest)
        binding=scope_binding(native,execution,path,manifest,template)
        dataset=profile_observations(native,context_id=job['id'],dataset_role='profile',
            stream_roles=roles,expected_API_counts=campaign.API_COUNTS,source_binding=binding)
        require(dataset['declared_state']==declared,'job/native declared state differs')
        dataset['stream_role_evidence']=proof;datasets.append(dataset)
    require(callback_counts==Counter({s:3 for s in STATES}),'three callback processes per exact state required')
    require(len(pairs)==12,'exact 12 matched observer pairs required')
    pair_checks=[]
    for identity,values in sorted(pairs.items()):
        require(set(values)=={'stamped','runtime_callbacks'} and min(values.values())>0,'incomplete/invalid observer pair')
        delta=abs(values['runtime_callbacks']/values['stamped']-1)
        pair_checks.append(dict(pair_id=identity,relative_difference=delta,passed=delta<=.05))
    require(all(x['passed'] for x in pair_checks),'one or more exact-state observer pairs failed5%; no price transfer')
    return datasets,dict(template=template,native_evidence=native_evidence,observer_pairs=pair_checks,
                         observer_target_totals_used_only_for_neutrality=True)


def model_objects(cells,sequence,evidence):
    decoded=dict(cells,cells=[dict(c,mean_ns=None if c['mean_ns'] is None else Fraction(c['mean_ns']))
                              for c in cells['cells']])
    return to_model_objects(decoded,sequence,profile_evidence=tuple(evidence),
                            program_evidence=('complete source-bound ordered Attention runtime API+gap partition',))


def predict_provider(program,profile,request,provider):
    require(provider in PROVIDERS,'unknown predeclared provider')
    intervals=lambda device:tuple(tuple(x*1000000 for x in pair)
                                 for pair in request[device+'_actor']['intervals_ms'])
    shared=dict(cpu_intervals_ns=intervals('cpu'),gpu_intervals_ns=intervals('gpu'),
                origin_ns=0,horizon_ns=HORIZON_NS)
    counts=dict(replays=request['replays'],block_replays=request['block_replays'],start_ns=0)
    if provider!='macro_four_state_sum_of_same_stage_prices':
        return program.predict(profile,provider='static' if provider=='static00' else 'memoryless',
                               require_qualified=False,**shared,**counts)
    lookup={(c.coordinate_sha256,c.role,c.state):c.mean_ns for c in profile.cells}
    summed={state:sum((lookup[s.coordinate_sha256,s.role,state] for s in program.stages),Fraction(0))
            for state in STATES}
    key=digest(dict(kind='complete_Attention_sum_of_same_local_API_gap_cells',
                    ordered_coordinates=[(s.coordinate_sha256,s.role) for s in program.stages]))
    law=PlannedCompletedResponse(program.scope.context_sha256,
        {key:dict(kind='RUNTIME_RESPONSE',states=summed)},(key,),provider='memoryless',
        evidence=profile.evidence+program.evidence,**shared)
    result=law.predict_replays(**counts)
    return dict(result,macro_cell_source='sum of the same local calibration API/gap means with occurrence multiplicity',
                complete_target_duration_fitted=False,internal_GPU_or_DMA_time_added=False,
                active_CPU_time_inferred=False,measured_target_boundaries_used=False)


def fit_and_freeze(out):
    out=Path(out).resolve();manifest=read(out/'manifest.json');check_sources(manifest);no_heldout(out,manifest)
    require(not (out/'frozen-barrier.json').exists() and not (out/'local-runtime-profile.json').exists(),
            'profile/forecasts are one frozen operation; no adaptive refitting')
    for stage in ('00_observer_gate','ON_profiles'):
        gate=read(out/(stage+'-observer-gate.json'))
        require(gate['passed'] is True and not gate['missing_or_failed_jobs'],
                'both OFF and all ON observer gates must pass before local price calibration')
    datasets,support=profile_datasets(out,manifest)
    cells=aggregate_cells(datasets,minimum_contexts=3,minimum_samples_per_context=1)
    require(cells['state_protocol']=='constant_state_v8' and cells['all_cells_supported'],
            'one or more exact API/gap/state coordinates unsupported; no missing-state fallback')
    sequence=datasets[0]['stage_sequence']
    evidence=tuple(row['native']['path']+'#sha256='+row['native']['sha256'] for row in support['native_evidence']
                   if next(j for j in manifest['jobs'] if j['id']==row['job_id'])['request']['instrumentation']=='runtime_callbacks')
    program,profile=to_model_objects(cells,sequence,profile_evidence=evidence,
        program_evidence=('same source-bound complete Attention template; own recursive forecast clock',))
    local=dict(schema='attention-runtime-local-profile-8',cells=cells,stage_sequence=sequence,evidence=evidence,
        support=support,source_manifest=receipt(out/'manifest.json'),
        source_files=manifest['sources'],whole_target_time_used_in_fit=False,heldout_observations_used=False,
        tau_or_transition_coefficient_estimated=False,hardware_accuracy_qualified=False)
    write_exact(out/'local-runtime-profile.json',local)
    write_exact(out/'profile-stage-datasets.json',dict(schema='attention-runtime-profile-labels-8',datasets=datasets,
        whole_target_time_used_in_fit=False,heldout_observations_used=False,raw_stage_labels_removed=0))
    cache={};forecasts=[]
    held=[j for j in manifest['jobs'] if j['split']=='heldout'];require(len(held)==6,'six independent heldout processes required')
    for job in held:
        for device in ('cpu','gpu'):
            actor={k:v for k,v in job['request'][device+'_actor'].items() if k!='intervals_ms'}
            require(actor==cells['scope']['actor_workloads'][device],
                    'heldout actor workload/placement differs from exact calibrated scope')
        for provider in PROVIDERS:
            key=(digest(job['request']),provider)
            if key not in cache:cache[key]=predict_provider(program,profile,job['request'],provider)
            forecasts.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,
                prediction=cache[key],primary_start_ns=0,actual_target_entry_used=False,
                entry_controller_max_lateness_ns=10000000,entry_lateness_was_not_fitted_or_reanchored=True))
    write_exact(out/'frozen-predictions.json',dict(schema='attention-runtime-frozen-predictions-8',rows=forecasts,
        declared_provider_process_cells=18,providers=PROVIDERS,observed_target_boundaries_used=False,
        whole_target_time_used_for_fit=False,future_target_state_used=False,
        target_scope='256 repetitions of one fixed-shape complete Attention graph; not a whole LLM strategy',
        state_model='predeclared owned actor schedule; memoryless stage response, no inferred OS scheduling',
        heldout_mathematical_seed_independent=True))
    files=('local-runtime-profile.json','profile-stage-datasets.json','frozen-predictions.json',
           '00_observer_gate-observer-gate.json','ON_profiles-observer-gate.json')
    barrier=dict(schema='attention-runtime-freeze-8',manifest_sha256=sha(out/'manifest.json'),
        all_exact_state_cells_supported=True,parameters_and_all_predictions_frozen=True,
        whole_target_time_used_in_fit=False,heldout_data_used=False,
        files={name:sha(out/name) for name in files})
    write(out/'frozen-barrier.json',barrier)
    return dict(profile=receipt(out/'local-runtime-profile.json'),barrier=receipt(out/'frozen-barrier.json'),
                supported_state_cells=cells['supported_cells'],stage_occurrences=len(sequence),forecasts=18,
                hardware_accuracy_validated=False)


def evaluate(out):
    out=Path(out).resolve();manifest=read(out/'manifest.json');check_sources(manifest)
    barrier=read(out/'frozen-barrier.json')
    require(barrier['manifest_sha256']==sha(out/'manifest.json') and barrier['all_exact_state_cells_supported'] is True
            and barrier['parameters_and_all_predictions_frozen'] is True and barrier['whole_target_time_used_in_fit'] is False,
            'complete pre-acquisition predictor freeze required')
    for name,value in barrier['files'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts and sha(out/path)==value,'frozen predictor changed')
    forecast=read(out/'frozen-predictions.json');predictions={(r['job_id'],r['provider']):r for r in forecast['rows']}
    held=[j for j in manifest['jobs'] if j['split']=='heldout']
    require(len(held)==6 and len(forecast['rows'])==18
            and set(predictions)=={(j['id'],p) for j in held for p in PROVIDERS},
            'all predeclared provider/process forecasts required without duplicates')
    for frozen in predictions.values():
        predicted=frozen['prediction'];blocks=predicted['block_ns']
        require(len(blocks)==16 and all(type(x) is int and x>=0 for x in blocks)
                and type(predicted['total_ns']) is int and sum(blocks)==predicted['total_ns'],
                'frozen total/block prediction inventory differs')
    rows=[];processes=[]
    for job in held:
        try:
            native,execution,path=verified_native(out,job,manifest)
            measured_blocks=[b['wall_ns'] for b in native['blocks']]
            require(len(measured_blocks)==16 and all(type(x) is int and x>0 for x in measured_blocks),
                    '16 positive original complete-Attention blocks required')
            measured=native['target_wall_ns'];require(type(measured) is int and measured>0,'positive original complete target time required')
            hz=native['clock']['frequency_hz'];require(type(hz) is int and hz>0,'positive QPC frequency required')
            require(abs(Fraction((native['target_end_tick']-native['target_begin_tick'])*10**9,hz)-measured)<=1,
                    'complete target duration does not match its QPC interval')
            require(all(abs(Fraction((b['end_tick']-b['begin_tick'])*10**9,hz)-b['wall_ns'])<=1
                        for b in native['blocks']),'block duration does not match its QPC interval')
            lateness=Fraction((native['target_begin_tick']-native['planned_target_entry_tick'])*10**9,hz)
            require(0<=lateness<=10000000 and abs(lateness-native['initial_start_lateness_ns'])<=1,
                    'actual entry violated the predeclared10ms controller envelope')
            processes.append(dict(job_id=job['id'],status='completed',native=execution['native'],execution=receipt(path),
                target_wall_ns=measured,initial_start_lateness_ns=native['initial_start_lateness_ns'],
                outer_block_shell_ns=measured-sum(measured_blocks),
                temporal_block_CV=statistics.pstdev(measured_blocks)/statistics.mean(measured_blocks),
                temporal_block_CV_is_stationary_repeat_variance=False))
            for provider in PROVIDERS:
                frozen=predictions[job['id'],provider];predicted=frozen['prediction'];blocks=predicted['block_ns']
                total=predicted['total_ns'];errors=[abs(p/m-1) for p,m in zip(blocks,measured_blocks)]
                mean_error=abs(total/measured-1);p90=campaign.base.e2a.quantile(errors,.90)
                rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,status='evaluated',
                    predicted_total_ns=total,measured_total_ns=measured,mean_response_APE=mean_error,
                    signed_bias=(total-measured)/measured,absolute_error_ns=abs(total-measured),
                    predicted_block_ns=blocks,measured_block_ns=measured_blocks,block_APEs=errors,block_APE_P90=p90,
                    accuracy_passed=mean_error<=.10 and p90<=.20,raw_CV_is_accuracy_gate=False,
                    actual_internal_boundaries_used_in_forecast=False,parameters_refitted=False))
        except (ValueError,KeyError,FileNotFoundError) as error:
            processes.append(dict(job_id=job['id'],status='unsupported_or_not_completed',reason=repr(error)))
            for provider in PROVIDERS:
                rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],provider=provider,
                    status='unsupported_or_not_completed',accuracy_passed=False,reason=repr(error)))
    repeat_variability=[]
    for plan in campaign.PULSES:
        values=[p['target_wall_ns'] for p in processes if p['status']=='completed'
                and next(j for j in manifest['jobs'] if j['id']==p['job_id'])['pulse_plan']==plan]
        repeat_variability.append(dict(pulse_plan=plan,declared_fresh_processes=3,completed_fresh_processes=len(values),
            target_total_means_ns=values,raw_fresh_process_CV=statistics.pstdev(values)/statistics.mean(values) if len(values)==3 else None,
            raw_CV_is_accuracy_gate=False))
    result=dict(schema='attention-runtime-independent-evaluation-8',rows=rows,processes=processes,
        declared_provider_process_cells=18,evaluated=sum(r['status']=='evaluated' for r in rows),
        accuracy_passed=sum(r['accuracy_passed'] for r in rows),all_accuracy_passed=len(rows)==18 and all(r['accuracy_passed'] for r in rows),
        repeat_variability=repeat_variability,thresholds=dict(mean_response_APE_max=.10,block_APE_P90_max=.20),
        no_refit=True,raw_trials_or_slow_blocks_removed=0,raw_CV_is_prediction_gate=False,
        actual_target_start_or_internal_boundaries_used_in_forecast=False,
        unmodeled_outer_loop_shell_is_measured_for_diagnosis_only=True,
        uncertainty='planned entry0; owned entry controller allows at most10ms lateness; no actual-time reanchoring',
        whole_LLM_dynamic_accuracy_validated=False,
        scope='same one Attention graph/runtime/core placement, two unseen owned-actor pulse plans and independent values/seed')
    write_exact(out/'independent-evaluation.json',result)
    return dict(output=receipt(out/'independent-evaluation.json'),accuracy_passed=result['accuracy_passed'],
                evaluated=result['evaluated'],declared=18,all_accuracy_passed=result['all_accuracy_passed'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('fit-freeze','evaluate'));parser.add_argument('--out',required=True)
    args=parser.parse_args();result=fit_and_freeze(args.out) if args.command=='fit-freeze' else evaluate(args.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
