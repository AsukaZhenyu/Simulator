"""Read-only integrity, parking controls and raw-cycle diagnostics for event runs.

This auditor does not fit durations or infer availability from the target. Native
cycle counters and CUDA records explain observed runs only. Event flags plus
bound source establish the parking contract, not measured continuous quiescence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from profiling.available_service.calibration import digest, require
from profiling.available_service.physical_records import parse_physical_records, interval_length
from profiling.event_stage_recovery import event_effective_jobs


SCHEMA='event-stage-campaign-audit-1'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(path, expected=None):
    path=Path(path);raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    require(expected is None or actual==expected,'source hash differs: '+str(path))
    return json.loads(raw),dict(path=str(path.resolve()),sha256=actual)


def _stats(values):
    values=list(values)
    if not values:return dict(n=0,mean=None,median=None,p90=None,worst=None,cv=None)
    ordered=sorted(values);index=(len(values)-1)*.9;left=int(index)
    average=statistics.mean(values)
    return dict(n=len(values),mean=average,median=statistics.median(values),
        p90=ordered[left]+(ordered[min(left+1,len(values)-1)]-ordered[left])*(index-left),worst=max(values),
        cv=None if average<=0 or len(values)<2 else statistics.stdev(values)/average)


def _cycles(native, expected):
    if native.get('thread_cycles_enabled') is not expected:
        raise ValueError('native cycle switch differs from declared role')
    rows=[];unknown=0
    for iteration in native['iterations']:
        for action in iteration['actions']:
            value=action.get('thread_cycles')
            if not expected:
                require(value is None,'disabled actor unexpectedly exports per-action cycles');continue
            require(isinstance(value,dict) and value.get('enabled') is True,'enabled actor lacks cycle record')
            require(value.get('unit')=='raw_QueryThreadCycleTime_cycles' and value.get('cycles_converted_to_time') is False and
                value.get('usable_as_heldout_resource_input') is False and value.get('running_ready_blocked_separated') is False,
                'cycle semantics changed')
            sequence=[iteration['begin_ns'],value['before_query_begin_ns'],value['before_query_end_ns'],
                action['begin_ns'],action['end_ns'],value['after_query_begin_ns'],value['after_query_end_ns'],iteration['end_ns']]
            require(all(type(n) is int and n>=0 for n in sequence) and all(a<=b for a,b in zip(sequence,sequence[1:])),
                    'cycle query/action/invocation brackets are inconsistent')
            before,after,delta=value.get('before_cycles'),value.get('after_cycles'),value.get('delta_cycles')
            if before is not None and after is not None and after>=before:
                require(type(before) is int and type(after) is int and delta==after-before and value.get('monotonic') is True,
                        'cycle delta is not raw conserved counter difference')
            else:
                require(delta is None,'unknown/nonmonotonic cycle counter was repaired');unknown+=1
            rows.append(dict(iteration=iteration['iteration'],action_index=action['action_index'],delta_cycles=delta,
                elapsed_ns=action['end_ns']-action['begin_ns'],device=action.get('device'),kind=action['kind']))
    return dict(enabled=expected,rows=rows,unknown_records=unknown,
        raw_cycle_delta=_stats([r['delta_cycles'] for r in rows if r['delta_cycles'] is not None]),
        counters_are_active_nanoseconds=False,counters_are_exogenous_available_service=False)


def _parking(native, process, expected, names):
    if not expected:
        require('event_parking' not in native,'polling control unexpectedly uses events')
        return dict(expected_event_parking=False,periodic_polling_control=True)
    value=native.get('event_parking');ready=process.get('ready',{}).get('event_parking')
    require(isinstance(value,dict) and value==ready,'native/READY event metadata differs')
    require(value.get('mode')=='parent_owned_Windows_events' and value.get('parent_manual_reset_contract') is True and
            value.get('native_event_set_or_reset_performed') is False and value.get('post_READY_periodic_file_poll') is False and
            value.get('post_sampling_periodic_wakeup') is False and value.get('global_timer_settings_changed') is False,
            'event parking contract differs')
    owned=process.get('event_controller',{})
    require(owned.get('manual_reset') is True,'parent manual-reset receipt missing')
    for role in ('go','stop'):
        name=value.get(role+'_event')
        require(isinstance(name,str) and name.startswith('Local\\') and name not in names,'event names absent or reused')
        require(owned.get('names',{}).get(role)==name,'native/controller event identity differs')
        names.add(name)
    return dict(expected_event_parking=True,flags_and_controller_identity_verified=True,
        no_periodic_sampler_claim_source_bound=True,continuous_quiescence_measured=False)


def _numerical(value):
    require(isinstance(value,dict) and all(value.get(k) is True for k in ('passed','full_graph_equal','action_trace_equal',
        'full_port_inventory_equal','copy_routes_bytes_and_values_equal')),'independent numerical receipt incomplete')
    require(type(value.get('independently_checked_ports')) is int and value['independently_checked_ports']>0,
            'numerical port count missing')
    return value['independently_checked_ports']


def _job(root,parent,protocol,job,event_names):
    directory=root/'jobs'/job['id'];execution,execution_receipt=_checked(directory/'execution.json')
    require(execution.get('job')==job and execution.get('status')=='completed' and execution.get('cleanup_verified') is True,
            'job identity/status/cleanup differs')
    require(execution.get('source_inventory_sha256')==parent['source_inventory_sha256'],'execution source inventory differs')
    binding=parent['programs'][job['program_key']]
    require(execution.get('program')==binding,'execution program binding differs')
    program,program_receipt=_checked(binding['path'],binding['sha256'])
    _,oracle_receipt=_checked(binding['oracle'],binding['oracle_sha256'])
    require(program['program_sha256']==binding['program_sha256']==digest({k:v for k,v in program.items() if k!='program_sha256'}),
            'semantic program digest differs')
    processes={r['name']:r for r in execution['processes']}
    expected={'main'}|({'before-cpu','before-gpu','after-cpu','after-gpu'} if job['references'] else set())
    expected|={device+'-worker' for device in ('cpu','gpu') if job[device+'_background']}
    require(len(processes)==len(execution['processes']) and set(processes)==set(execution['raw_files'])==expected,
            'actor inventory differs from declared job')
    receipts=[]
    for path in directory.glob('*/process.json'):
        value,receipt=_checked(path);receipts.append(dict(receipt,name=value.get('name'),pid=value.get('pid'),
            returncode=value.get('returncode'),forced_termination=value.get('forced_termination'),
            process_exited_verified=value.get('process_exited_verified')))
        require(value.get('name') in processes and value.get('pid')==processes[value['name']]['pid'],
                'unlisted constructor process receipt retained')
        require(value.get('returncode')==0 and value.get('forced_termination') is False and value.get('process_exited_verified') is True,
                'process did not exit normally')
    require(len(receipts)==len(processes),'some owned process receipt files are missing')
    actors={};all_sources=[execution_receipt,program_receipt,oracle_receipt]
    main=None;parsed_main=None;cycles_main=None
    for name,process in processes.items():
        source=execution['raw_files'][name];native,receipt=_checked(source['path'],source['sha256']);all_sources.append(receipt)
        require(process.get('returncode')==0 and process.get('forced_termination') is False and process.get('process_exited_verified') is True,
                'execution process exit receipt differs')
        require(native.get('process_id')==process['pid']==process.get('ready',{}).get('process_id') and
                native.get('main_thread_id')==process['ready'].get('main_thread_id'),'native/READY PID or TID differs')
        require(native.get('program_sha256')==program['program_sha256'] and native.get('actions')==program['actions'],
                'actor program or action identity differs')
        expected_binary=parent['background_binary'] if name.endswith('-worker') else protocol[job['parking']+'_binary']
        require(Path(process['command'][0]).resolve()==Path(expected_binary).resolve() and
                Path(process['command'][1]).resolve()==Path(binding['path']).resolve() and
                Path(process['command'][-1]).resolve()==Path(source['path']).resolve(),'actor command provenance differs')
        ports=_numerical(execution['numerical'][name])
        record=dict(process_id=process['pid'],numerical_ports=ports,raw=receipt)
        if name.endswith('-worker'):
            require(native.get('format_version')=='state-response-native-v1','worker schema differs')
        else:
            expected_instrument=job['instrumentation'] if name=='main' else 'cupti'
            require(native.get('schema')=='physical-stage-native-1' and native.get('instrumentation')==expected_instrument,
                    'physical actor schema/instrumentation differs')
            parsed=parse_physical_records(native,program=program)
            require(expected_instrument!='cupti' or parsed['capture_qualified'],'CUPTI attribution/capture is unqualified')
            record.update(parking=_parking(native,process,job['parking']=='event',event_names),
                capture_enabled=expected_instrument=='cupti',capture_qualified=parsed['capture_qualified'],
                capture_issues=parsed['capture_issues'])
            expected_cycles=job['parking']=='event' and (job['cycles'] if name=='main' else protocol['event_reference_cycles_enabled'])
            if job['parking']=='event':
                cycle=_cycles(native,expected_cycles);record['cycle_summary']={k:v for k,v in cycle.items() if k!='rows'}
            else:
                require(not any('thread_cycles' in a for i in native['iterations'] for a in i['actions']),'polling actor unexpectedly records cycles')
                cycle=dict(enabled=False,rows=[],unknown_records=0)
            expected_samples=job['samples'] if name=='main' else parent['protocol']['reference_'+name.rsplit('-',1)[1]+'_samples']
            require(native['samples_completed']==native['samples_requested']==len(native['iterations'])==expected_samples,'sample count differs')
            completed=execution['completed'][name]
            require(completed['process_id']==process['pid'] and completed['sampling_begin_ns']==native['sampling_begin_ns'] and
                completed['sampling_end_ns']==native['sampling_end_ns'] and completed['completed_samples']==expected_samples,
                'sampling-complete receipt differs')
            if name=='main':main,parsed_main,cycles_main=native,parsed,cycle
        actors[name]=record
    require(main is not None,'missing main')
    require(not(job['split']=='heldout' and main.get('thread_cycles_enabled')),'heldout target is cycle-instrumented')
    wall=[i['end_ns']-i['begin_ns'] for i in main['iterations']]
    diag=None
    if job['split']=='diagnostic':
        lookup={(a['iteration'],a['action_index']):a for a in parsed_main['actions']};cycle_lookup={(a['iteration'],a['action_index']):a for a in cycles_main['rows']}
        calls=[]
        for invocation in main['iterations']:
            keys=[(invocation['iteration'],a['action_index']) for a in invocation['actions']]
            raw_cycles=[cycle_lookup[k]['delta_cycles'] for k in keys]
            api=sum(interval_length(lookup[k]['runtime_api_union']+lookup[k]['driver_api_union']) for k in keys)
            device=sum(lookup[k]['device_activity_union_ns'] for k in keys)
            kernels=sum(lookup[k]['kernel_union_ns'] for k in keys)
            copies=sum(lookup[k]['memcpy_union_ns'] for k in keys)
            slowest=max(invocation['actions'],key=lambda a:a['end_ns']-a['begin_ns'])
            calls.append(dict(iteration=invocation['iteration'],wall_ns=invocation['end_ns']-invocation['begin_ns'],
                raw_cycle_work=None if any(v is None for v in raw_cycles) else sum(raw_cycles),api_union_ns=api,
                device_activity_union_ns=device,GPU_kernel_envelope_ns=kernels,GPU_memcpy_envelope_ns=copies,
                slowest_action_index=slowest['action_index'],
                slowest_action_ns=slowest['end_ns']-slowest['begin_ns']))
        diag=dict(calls=calls,wall_ns=_stats([c['wall_ns'] for c in calls]),
            GPU_kernel_envelope_ns=_stats([c['GPU_kernel_envelope_ns'] for c in calls]),
            GPU_memcpy_envelope_ns=_stats([c['GPU_memcpy_envelope_ns'] for c in calls]),
            API_union_ns=_stats([c['api_union_ns'] for c in calls]),
            raw_cycle_work=_stats([c['raw_cycle_work'] for c in calls if c['raw_cycle_work'] is not None]),
            slowest_calls=sorted(calls,key=lambda c:c['wall_ns'],reverse=True)[:5],API_device_columns_additive=False,
            cycles_are_active_time=False,resource_trajectory_inferred=False,
            GPU_envelopes_are_useful_arithmetic_service=False,CPU_pure_compute_time_ns=None,
            cold_process_and_executor_initialization_in_target_timer=False)
    sidecar=execution.get('posttarget_cleanup_receipt',{})
    cleanup,cleanup_source=_checked(sidecar['path'],sidecar['sha256']);all_sources.append(cleanup_source)
    require(cleanup['index']['sha256']==_sha(root/'event-protocol.json') and cleanup['target_sampling_altered'] is False and
            cleanup['all_backgrounds_exited_before_after_construction'] is True,'post-target cleanup binding differs')
    return dict(job=job,status='included',execution=execution_receipt,source_receipts=all_sources,
        processes=receipts,actors=actors,main_process_id=main['process_id'],wall_ns=_stats(wall),
        diagnostic=diag,posttarget_after_reference_scope=execution.get('posttarget_after_reference_scope'),
        bound_numerical_receipts_verified=True,numerical_recomputed_here=False)


def _observer(rows,threshold):
    groups={}
    for row in rows:
        job=row['job']
        if job['split']!='observer' or row.get('status')!='included':continue
        repeat=int(job['program_key'].rsplit('-',1)[1]);key=(job['parking'],job['cycles'])
        groups.setdefault(repeat,{})[key]=row
    controls=[]
    for repeat,group in sorted(groups.items()):
        if not all(k in group for k in (('poll',False),('event',False),('event',True))):continue
        poll,plain,cycles=(group[k] for k in (('poll',False),('event',False),('event',True)))
        ratios={name:dict(mean=numerator['wall_ns']['mean']/denominator['wall_ns']['mean']-1,
            median=numerator['wall_ns']['median']/denominator['wall_ns']['median']-1)
            for name,numerator,denominator in (('event_vs_poll',plain,poll),('cycles_vs_event',cycles,plain))}
        controls.append(dict(repeat=repeat,poll=poll['wall_ns'],event=plain['wall_ns'],event_cycles=cycles['wall_ns'],relative_changes=ratios))
    return dict(matched_repeats=len(controls),controls=controls,threshold=threshold,
        cycle_observer_overhead_pass=bool(len(controls)==3 and all(abs(r['relative_changes']['cycles_vs_event']['mean'])<=threshold for r in controls)),
        parking_change_is_observer_overhead=False,point_predictions_or_parameters_fitted=False,
        stable_control_processes=bool(controls) and all(row['wall_ns']['cv'] is not None and row['wall_ns']['cv']<=.1 for row in rows if row['job']['split']=='observer' and row.get('status')=='included'))


def audit_campaign(root,allow_incomplete=False):
    root=Path(root).resolve();parent,parent_receipt=_checked(root/'manifest.json');protocol,protocol_receipt=_checked(root/'event-protocol.json')
    require(protocol.get('schema')=='event-stage-campaign-1' and protocol['parent_manifest_sha256']==parent_receipt['sha256'],
            'event protocol parent/schema differs')
    require(parent['source_inventory_sha256']==digest(parent['sources']),'source inventory digest differs')
    source_errors=[];checked=0
    for path,expected in dict(parent['sources'],**protocol['source_receipts']).items():
        checked+=1
        try:
            if _sha(path)!=expected:source_errors.append(dict(path=path,reason='hash differs'))
        except OSError as error:source_errors.append(dict(path=path,reason=repr(error)))
    effective=event_effective_jobs(root,protocol)
    recovery_path=root/'event-recovery-index.json'
    recovery=json.loads(recovery_path.read_text(encoding='utf-8')) if recovery_path.exists() else None
    retained=[]
    if recovery:
        for replacement in recovery['replacements']:
            errors=[]
            for source in replacement['retained_files']:
                if _sha(source['path'])!=source['sha256']:errors.append(source['path'])
            source_errors.extend(dict(path=p,reason='retained failed attempt changed') for p in errors)
            retained.append(dict(original_id=replacement['original_id'],failed_execution=replacement['failed_execution'],
                error="PermissionError(13, 'Permission denied')",all_failed_main_samples_excluded=True,
                retained_files_verified=not errors))
    rows=[];event_names=set();pending=[]
    for job in effective:
        path=root/'jobs'/job['id']/'execution.json'
        if not path.exists():pending.append(job['id']);continue
        try:rows.append(_job(root,parent,protocol,job,event_names))
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
            rows.append(dict(job=job,status='excluded',error=repr(error),execution_path=str(path)))
    require(allow_incomplete or not pending,'incomplete event campaign: '+','.join(pending))
    known={j['id'] for j in effective}|{r['original_id'] for r in retained}
    unindexed=[str(path.parent) for path in (root/'jobs').glob('*/execution.json') if path.parent.name not in known]
    counts=dict(planned=len(effective),with_execution=len(rows),included=sum(r['status']=='included' for r in rows),
        excluded=sum(r['status']=='excluded' for r in rows),pending=len(pending),unindexed=len(unindexed))
    return dict(schema=SCHEMA,campaign=str(root),auditor=dict(path=str(Path(__file__).resolve()),sha256=_sha(__file__)),
        manifest=parent_receipt,protocol=protocol_receipt,source_files_rehashed=checked,source_errors=source_errors,
        complete=not pending,passed=not source_errors and not pending and not unindexed and not counts['excluded'],
        counts=counts,jobs=rows,pending_job_ids=pending,unindexed_attempts=unindexed,retained_failed_attempts=retained,
        observer=_observer(rows,protocol['observer_error_max']),
        hardware_accuracy_qualified=False,continuous_available_resource_trajectory_measured=False,
        target_cycles_used_as_heldout_input=False,model_or_parameter_fit_performed=False,
        limitations=['Saved oracle/port receipts are hash-bound, not numerically recomputed here.',
            'Event parking flags and source identity do not independently measure exact idle-thread quiescence.',
            'Raw cycles include user/kernel counter work; they are not active CPU nanoseconds or neutral availability.',
            'CUPTI and cycle instrumentation can perturb target timing; matched controls remain separate from accuracy.',
            'API/device envelopes overlap and cannot be summed as independent physical demands.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    value=audit_campaign(a.campaign,a.allow_incomplete)
    with a.output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(a.output),counts=value['counts'],passed=value['passed'])))
