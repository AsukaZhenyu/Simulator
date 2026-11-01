"""Read-only stage attribution; observed spans never become fitted prices."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import statistics

from profiling.dynamic_validation import h2d_v3 as campaign
from profiling.h2d_history.analysis import correlation_map


def qpc_ns(tick,hz):return tick//hz*1000000000+(tick%hz)*1000000000//hz


def interval_union(intervals):
    result=0;first=last=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('reversed observed interval')
        if first is None:first,last=a,b
        elif a<=last:last=max(last,b)
        else:result+=last-first;first,last=a,b
    return result+(0 if first is None else last-first)


def physical_by_external(cupti):
    relevant=[r for r in cupti['records'] if r['kind']!='external_correlation' or r['external_kind']==cupti['external_correlation_kind']]
    mapping,methods,graph=correlation_map(relevant)
    grouped=defaultdict(list)
    for record in cupti['records']:
        if record['kind']=='external_correlation':continue
        ids=mapping.get(record['correlation_id'],set())
        if len(ids)==1:grouped[next(iter(ids))].append(record)
    return grouped,graph


def attribute(stage,records,hz):
    """Runtime union avoids nested driver/API double charging."""
    lower,upper=qpc_ns(stage['copy_API_enter_tick'],hz),qpc_ns(stage['copy_API_return_tick'],hz)
    runtime=[r for r in records if r['kind']=='runtime_api' and r.get('timestamp_valid')]
    memcpy=[r for r in records if r['kind']=='memcpy'];invalid=[r for r in records if not r.get('timestamp_valid')]
    within=lambda r:max(0,min(upper,r['end_ns'])-max(lower,r['begin_ns']))
    submit=[r for r in runtime if (r.get('name') or '').startswith('cudaMemcpy') and within(r)>0]
    sync=[r for r in runtime if 'Synchronize' in (r.get('name') or '') and within(r)>0]
    span=lambda rs:interval_union([(max(lower,r['begin_ns']),min(upper,r['end_ns'])) for r in rs if within(r)>0])
    covered=span(runtime);streams=sorted({(r['device_id'],r['context_id'],r['stream_id']) for r in memcpy})
    valid_device=bool(memcpy) and all(r.get('timestamp_valid') for r in memcpy) and len(streams)==1
    memcpy_span=max(r['end_ns'] for r in memcpy)-min(r['begin_ns'] for r in memcpy) if valid_device else None
    reasons=[]
    if len(submit)!=1:reasons.append('not_exactly_one_correlated_runtime_memcpy_API')
    if not sync:reasons.append('inner_synchronization_API_not_identified')
    if not valid_device:reasons.append('device_memcpy_stream_or_timestamps_unknown')
    if sum(r['bytes'] for r in memcpy)!=2097152:reasons.append('correlated_device_payload_not_two_MiB')
    if any(r['kind']=='kernel' for r in records):reasons.append('unexpected_kernel_in_copy_correlation')
    return dict(supported=not reasons,unsupported_reasons=reasons,runtime_API_names=[r.get('name') for r in runtime],
        copy_API_inclusive_ns=stage['copy_API_inclusive_ns'],runtime_memcpy_API_union_ns=span(submit) if submit else None,
        inner_synchronization_runtime_union_ns=span(sync) if sync else None,
        runtime_API_union_in_copy_wrapper_ns=covered,GGML_copy_outside_recorded_runtime_union_ns=upper-lower-covered,
        device_memcpy_span_ns=memcpy_span,device_memcpy_bytes=sum(r['bytes'] for r in memcpy),device_streams=[list(r) for r in streams],
        device_memcpy_activity_count=len(memcpy),invalid_timestamp_records=len(invalid),
        driver_API_intervals_not_added_to_nested_runtime=True,device_span_added_to_host_wait=False,
        physical_span_is_device_activity_envelope_not_available_bandwidth=True,raw_correlated_records=records)


def analyze(out):
    out=Path(out).resolve();read,require,receipt=campaign.read,campaign.require,campaign.receipt;m=read(out/'manifest.json')
    index=out/'executions.json';require(index.exists(),'all declared diagnostic jobs must have execution receipts before combined analysis')
    executions=read(index);require({e['job']['id'] for e in executions}=={j['id'] for j in m['jobs']},'diagnostic denominator differs')
    processes=[];rows=[]
    for execution in executions:
        job=execution['job'];record=dict(job_id=job['id'],memory_mode=job['memory_mode'],init_policy=job['init_policy'],round=job['round'],
            instrumentation=job['instrumentation'],status=execution['status'],phase_supported_rows=0)
        if execution['status']!='completed':record['error']=execution.get('error');processes.append(record);continue
        require(campaign.sha(execution['native']['path'])==execution['native']['sha256'],'native evidence changed')
        native=read(execution['native']['path']);hz=native['clock']['frequency_hz'];grouped={};ambiguity={}
        if job['instrumentation']=='stages':grouped,ambiguity=physical_by_external(native['cupti'])
        focus=[]
        for idle in (0,20):
            seq=[s for s in native['sequences'] if s['requested_idle_ms']==idle];trial=[]
            for s in seq:
                first=s['copies'][0]['wall_ns'];later=statistics.mean(c['wall_ns'] for c in s['copies'][1:]);trial.append(dict(trial=s['trial'],first_ns=first,later_ns=later,prime_ns=s['prime_ns'],
                    burst_wall_ns=s['burst_wall_ns'],actual_idle_ns=s['actual_idle_ns'],effective_prime_end_to_first_copy_ns=s['effective_prime_end_to_first_copy_ns']))
                for c in s['copies']:
                    if c['host_stages'] is None:continue
                    campaign.validate_stages(c['host_stages'],hz)
                    physical=attribute(c['host_stages'],grouped.get(c['external_id'],[]),hz)
                    record['phase_supported_rows']+=physical['supported']
                    rows.append(dict(job_id=job['id'],memory_mode=job['memory_mode'],init_policy=job['init_policy'],round=job['round'],
                        requested_idle_ms=idle,trial=s['trial'],ordinal=c['ordinal'],external_id=c['external_id'],begin_tick=c['begin_tick'],end_tick=c['end_tick'],
                        host_wall_ns=c['wall_ns'],host_stages=c['host_stages'],physical=physical))
            focus.append(dict(requested_idle_ms=idle,trial_count=len(trial),first_mean_ns=statistics.mean(r['first_ns'] for r in trial),later_mean_ns=statistics.mean(r['later_ns'] for r in trial),
                burst_mean_ns=statistics.mean(r['burst_wall_ns']/16 for r in trial),trials=trial))
        inventories={}
        for name in ('threads_before_initialization','threads_after_initialization','threads_after_warmup','threads_after_observer_setup'):
            snapshot=native[name];inventories[name]=dict(available=snapshot['available'],thread_count=len(snapshot['threads']),
                observed_masks=dict(Counter(str(t['affinity_mask']) for t in snapshot['threads'] if t['affinity_mask'] is not None)),
                unknown_affinity_count=sum(t['affinity_mask'] is None for t in snapshot['threads']),threads=snapshot['threads'])
        record.update(native=execution['native'],focus=focus,thread_inventories=inventories,correlation_ambiguities=ambiguity,
            cupti_integrity=native['cupti'].get('all_requested_records_retained') if job['instrumentation']=='stages' else None,
            phase_declared_rows=512 if job['instrumentation']=='stages' else 0)
        processes.append(record)
    pairs=[]
    for job in m['jobs']:
        if job['instrumentation']!='bare':continue
        base=next((r for r in processes if r['job_id']==job['id']),None)
        observer=next((r for r in processes if r['memory_mode']==job['memory_mode'] and r['init_policy']==job['init_policy'] and r['round']==job['round'] and r['instrumentation']=='stages'),None)
        pair=dict(memory_mode=job['memory_mode'],init_policy=job['init_policy'],round=job['round'],bare_job_id=job['id'],observer_job_id=observer['job_id'] if observer else None,
            qualified=False,difference_is_observer_and_process_context_discrepancy_not_pure_profiler_overhead=True)
        if base and observer and base['status']==observer['status']=='completed':
            errors=[]
            for b,o in zip(base['focus'],observer['focus']):
                errors.append(dict(requested_idle_ms=b['requested_idle_ms'],first_relative_discrepancy=abs(o['first_mean_ns']-b['first_mean_ns'])/b['first_mean_ns'],
                    later_relative_discrepancy=abs(o['later_mean_ns']-b['later_mean_ns'])/b['later_mean_ns'],
                    burst_relative_discrepancy=abs(o['burst_mean_ns']-b['burst_mean_ns'])/b['burst_mean_ns']))
            pair['conditions']=errors;pair['qualified']=all(r['burst_relative_discrepancy']<=m['protocol']['observer_process_discrepancy_limit'] for r in errors)
        pairs.append(pair)
    result=dict(schema='dynamic-h2d-stage-analysis-3',manifest=receipt(out/'manifest.json'),executions=receipt(index),processes=processes,stage_rows=rows,observer_pairs=pairs,
        generator=receipt(Path(__file__)),
        supporting_sources=[receipt(Path(correlation_map.__code__.co_filename))],
        declared_processes=len(m['jobs']),completed_processes=sum(r['status']=='completed' for r in processes),host_wall_partition_checked=True,
        prior_and_post_target_state_are_distinguished=True,all_raw_samples_retained=True,performance_parameters_fitted=False,new_model_prices_created=False,
        device_and_host_intervals_added=False,helper_affinity_inheritance_or_cause_inferred=False,
        new_diagnostic_process_is_equivalent_to_v2=False,
        measurement_boundary='per-copy bare Executor.action unchanged; new pre-burst snapshots, condition JSON and target-inclusive50ms telemetry change host/idle history',
        old_drift_absence_does_not_prove_wide_init_fix=True,telemetry_is_future_forecast_input=False)
    target=out/'stage-analysis.json';campaign.write(target,result)
    return dict(analysis=str(target),declared=result['declared_processes'],completed=result['completed_processes'],stage_rows=len(rows),
        stage_supported=sum(r['physical']['supported'] for r in rows),observer_pairs_qualified=sum(r['qualified'] for r in pairs),observer_pairs_declared=len(pairs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(analyze(a.out),ensure_ascii=False),flush=True)
