"""Diagnostic-only accounting for matched H2D and execution-history runs.

Inclusive host/API spans overlap device activity. This module never adds them,
never fits a timing coefficient, and never identifies an unobserved mechanism.
Uncorrelated device events remain unknown instead of being assigned by time.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,separators=(',',':'),allow_nan=False); stream.write('\n')


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition,message):
    if not condition: raise ValueError(message)


def describe(values):
    values = list(values)
    if not values:
        return dict(count=0, mean=None, min=None, max=None, cv=None)
    require(all(math.isfinite(v) for v in values), 'nonfinite diagnostic value')
    average = statistics.mean(values)
    return dict(count=len(values), mean=average, min=min(values), max=max(values),
                cv=statistics.stdev(values)/average if len(values)>1 and average>0 else None)


def union_ns(intervals, clip=None):
    """Length of a union, not sum of overlapping activity/API durations."""
    clean=[]
    for begin,end in intervals:
        require(end>=begin, 'interval reversed')
        if clip is not None:
            begin,end=max(begin,clip[0]),min(end,clip[1])
        if end>begin: clean.append((begin,end))
    clean.sort()
    if not clean: return 0
    begin,end=clean[0]; total=0
    for first,last in clean[1:]:
        if first>end: total+=end-begin; begin,end=first,last
        else: end=max(end,last)
    return total+end-begin


def trend(values):
    values=list(values); require(len(values)>=2, 'trend needs at least two values')
    midpoint=len(values)//2
    early,late=statistics.mean(values[:midpoint]),statistics.mean(values[midpoint:])
    return dict(early_mean=early,late_mean=late,
                late_over_early_minus_one=late/early-1 if early>0 else None)


def correlation_map(records):
    """Return explicit correlation graph; no timestamp-only MEMCPY assignment.

    A driver API nested in a unique innermost runtime API on the same thread can
    bridge two correlation IDs. Direct external mappings take precedence and a
    conflict is retained as ambiguity, never resolved by choosing one label.
    """
    direct=defaultdict(set); graph=defaultdict(set); bridges=[]; conflicts=[]
    for r in records:
        if r['kind']=='external_correlation':
            direct[r['correlation_id']].add(r['external_id'])
    runtime=[r for r in records if r['kind']=='runtime_api' and r.get('timestamp_valid')]
    driver=[r for r in records if r['kind']=='driver_api' and r.get('timestamp_valid')]
    indexed=defaultdict(list)
    for r in runtime: indexed[(r.get('process_id'),r.get('thread_id'))].append(r)
    timelines={}
    for thread,items in indexed.items():
        items.sort(key=lambda r:r['begin_ns']); prefix=[]; maximum=-1
        for r in items:
            maximum=max(maximum,r['end_ns']); prefix.append(maximum)
        timelines[thread]=(items,[r['begin_ns'] for r in items],prefix)
    for child in driver:
        items,begins,prefix=timelines.get((child.get('process_id'),child.get('thread_id')),([],[],[]))
        cursor=bisect.bisect_right(begins,child['begin_ns'])-1; parents=[]
        while cursor>=0 and prefix[cursor]>=child['end_ns']:
            parent=items[cursor]
            if child['end_ns']<=parent['end_ns']: parents.append(parent)
            cursor-=1
        if not parents: continue
        narrow=min(p['end_ns']-p['begin_ns'] for p in parents)
        candidates=[p for p in parents if p['end_ns']-p['begin_ns']==narrow]
        parent_ids={p['correlation_id'] for p in candidates}
        if len(parent_ids)!=1: continue
        parent=next(iter(parent_ids)); node=child['correlation_id']
        if parent==node: continue
        # No inference without an explicit external anchor somewhere on edge.
        anchors=direct[parent]|direct[node]
        if not anchors: continue
        if len(anchors)>1:
            conflicts.append(dict(runtime_correlation=parent,driver_correlation=node,
                                  external_ids=sorted(anchors)))
            continue
        graph[parent].add(node); graph[node].add(parent)
        bridges.append(dict(runtime_correlation=parent,driver_correlation=node,
                            method='unique_innermost_same_thread_nested_API'))
    mapped={}; methods={}
    for start in set(direct)|set(graph):
        seen={start}; queue=deque([start]); external=set()
        while queue:
            node=queue.popleft(); external.update(direct[node])
            for neighbor in graph[node]:
                if neighbor not in seen: seen.add(neighbor); queue.append(neighbor)
        mapped[start]=external
        methods[start]='direct_external' if direct[start] else 'nested_runtime_driver_bridge'
    return mapped,methods,dict(bridges=bridges,conflicts=conflicts,
                              ambiguous_correlation_ids=sorted(k for k,v in mapped.items() if len(v)>1),
                              device_assignment_by_time_only=False)


def physical_by_action(native, stamps):
    records=native.get('cupti',{}).get('records',[])
    mapped,methods,graph=correlation_map(records)
    known={r['external_id'] for r in stamps}; grouped=defaultdict(list)
    unmatched=defaultdict(int); association=defaultdict(int)
    for r in records:
        if r['kind']=='external_correlation': continue
        external=mapped.get(r.get('correlation_id'),set())
        if len(external)!=1:
            unmatched[r['kind']+(':ambiguous' if len(external)>1 else ':uncorrelated')]+=1
            continue
        target=next(iter(external))
        if target not in known:
            unmatched[r['kind']+':outside_timed_action_IDs']+=1; continue
        grouped[target].append(r)
        association[methods.get(r['correlation_id'],'unknown')]+=1
    graph.update(unmatched_records=dict(unmatched),associated_records_by_method=dict(association),
                 total_activity_records=len(records),timed_external_ids=len(known))
    return grouped,graph


def one_physical(stamp, semantic, records, capture_integrity):
    begin,end=stamp['begin_ns'],stamp['end_ns']; clip=(begin,end)
    copies=[r for r in records if r['kind']=='memcpy' and r.get('timestamp_valid')]
    kernels=[r for r in records if r['kind']=='kernel' and r.get('timestamp_valid')]
    by_api=defaultdict(list)
    for r in records:
        if r['kind'] in ('runtime_api','driver_api') and r.get('timestamp_valid'):
            name=r.get('name') or 'unknown_callback_'+str(r.get('callback_id'))
            by_api[r['kind']+':'+name].append((r['begin_ns'],r['end_ns']))
    api={k:union_ns(v,clip) for k,v in sorted(by_api.items())}
    bytes_seen=sum(r.get('bytes',0) for r in copies)
    direction_ok=all(r.get('copy_kind')==1 for r in copies) if semantic.get('src')=='dram' and semantic.get('dst')=='vram' else None
    contained=all(begin<=r['begin_ns']<=r['end_ns']<=end for r in copies)
    coverage=bool(capture_integrity and copies and semantic['kind']=='COPY'
                  and bytes_seen==semantic.get('bytes') and contained and direction_ok is not False)
    union=union_ns([(r['begin_ns'],r['end_ns']) for r in copies],clip) if copies else None
    # These are disjoint envelope regions, not identified OS/staging components.
    first=min((r['begin_ns'] for r in copies),default=None)
    last=max((r['end_ns'] for r in copies),default=None)
    return dict(memcpy_records=len(copies),memcpy_bytes=bytes_seen,
                expected_copy_bytes=semantic.get('bytes'),h2d_direction_matches=direction_ok,
                memcpy_events_inside_host_action=contained,
                exact_payload_activity_coverage=coverage,
                device_memcpy_union_ns=union,
                device_kernel_union_ns=union_ns([(r['begin_ns'],r['end_ns']) for r in kernels],clip) if kernels else None,
                pre_device_wall_ns=first-begin if coverage else None,
                post_device_wall_ns=end-last if coverage else None,
                host_wall_outside_memcpy_union_ns=end-begin-union if coverage else None,
                api_union_ns_by_name=api,
                device_memcpy_raw_spans=[dict(begin_ns=r['begin_ns'],end_ns=r['end_ns'],bytes=r['bytes'],
                                             correlation_id=r['correlation_id']) for r in copies],
                host_device_times_additive=False,
                unaccounted_host_interval_identified_as_staging_or_OS_wait=False)


def summarize_native(native):
    trials=native['trials']; require(len(trials)>=2,'insufficient chronological blocks')
    hz=native['clock']['frequency_hz']; semantics={r['action_index']:r for r in native['action_semantics']}
    stamps=[r for t in trials for r in t['action_intervals']]
    expected=len(trials)*native['request']['sampling']['repeats']*(len(native['actions']) if native['mode']=='full' else 1)
    require(len(stamps)==(0 if native['instrumentation']=='bare' else expected),'action stamp count mismatch')
    require(len({r['external_id'] for r in stamps})==len(stamps),'duplicate timed external ID')
    physical,mapping=physical_by_action(native,stamps)
    block_rows=[]
    for t in trials:
        require(t['end_tick']>t['begin_tick'],'block tick order differs')
        require(abs((t['end_tick']-t['begin_tick'])*1e9/hz-t['wall_ns'])<=1,'block QPC ledger differs')
        summed=sum(r['wall_ns'] for r in t['action_intervals'])
        require(summed<=t['wall_ns']+len(t['action_intervals']),'action spans exceed enclosing block')
        block_rows.append(dict(trial=t['trial'],wall_ns=t['wall_ns'],replays=t['repeats'],
                               mean_replay_ns=t['wall_ns']/t['repeats'],
                               accounted_action_ns=summed if stamps else None,
                               outside_stamped_action_ns=t['wall_ns']-summed if stamps else None,
                               cpu_accounting=t['cpu_accounting']))
    groups=defaultdict(list)
    for r in stamps: groups[r['action_index']].append(r)
    action_rows=[]
    for index,samples in sorted(groups.items()):
        semantic=semantics[index]; is_focus=semantic.get('src')=='dram' and semantic.get('dst')=='vram' and semantic.get('bytes')==2097152
        values=[r['wall_ns'] for r in samples]
        per_trial=[statistics.mean(r['wall_ns'] for r in samples if r['trial']==t['trial']) for t in trials]
        item=dict(action_index=index,semantic=semantic,count=len(samples),wall_ns=describe(values),
                  mean_per_trial_ns=per_trial,trend=trend(per_trial),focus_2MiB_H2D=is_focus)
        for key in ('copy_call_ns','gpu_sync_ns','cpu_sync_ns'):
            item[key]=describe(r[key] for r in samples if r[key] is not None)
        if native['instrumentation']=='cupti' and semantic['kind']=='COPY':
            details=[]
            for r in samples:
                detail=one_physical(r,semantic,physical.get(r['external_id'],[]),native.get('capture_integrity_pass',False))
                detail.update(trial=r['trial'],invocation=r['invocation'],external_id=r['external_id'],
                              host_wall_ns=r['wall_ns'],copy_call_ns=r['copy_call_ns'],
                              gpu_sync_ns=r['gpu_sync_ns'],cpu_sync_ns=r['cpu_sync_ns'])
                details.append(detail)
            item['physical']=dict(expected_invocations=len(samples),
                exact_payload_coverage_count=sum(d['exact_payload_activity_coverage'] for d in details),
                memcpy_invocations_matched=sum(d['memcpy_records']>0 for d in details),
                device_memcpy_union_ns=describe(d['device_memcpy_union_ns'] for d in details if d['device_memcpy_union_ns'] is not None),
                pre_device_wall_ns=describe(d['pre_device_wall_ns'] for d in details if d['pre_device_wall_ns'] is not None),
                post_device_wall_ns=describe(d['post_device_wall_ns'] for d in details if d['post_device_wall_ns'] is not None),
                host_wall_outside_memcpy_union_ns=describe(d['host_wall_outside_memcpy_union_ns'] for d in details if d['host_wall_outside_memcpy_union_ns'] is not None))
            names=sorted({name for d in details for name in d['api_union_ns_by_name']})
            item['physical']['api_union_ns_by_name']={name:describe(d['api_union_ns_by_name'].get(name,0) for d in details) for name in names}
            if is_focus: item['physical']['invocations']=details
        action_rows.append(item)
    return dict(mode=native['mode'],instrumentation=native['instrumentation'],memory_mode=native['memory_mode'],
                history_intervention=native.get('history_intervention','none'),
                preparation_contract=native['preparation_contract'],known_allocation_bytes=native['known_allocation_bytes'],
                arena_registration=native['arena_registration'],
                replay_time_ns=describe(r['mean_replay_ns'] for r in block_rows),
                replay_trend=trend(r['mean_replay_ns'] for r in block_rows),
                blocks=block_rows,actions=action_rows,
                capture_integrity_pass=native.get('capture_integrity_pass'),correlation_graph=mapping,
                accounting_total={key:describe(t['cpu_accounting'][key] for t in trials if t['cpu_accounting'][key] is not None)
                                  for key in ('thread_accounted_cpu_ns','process_accounted_cpu_ns','thread_cycles','process_page_fault_delta')},
                observed_state_used_for_future_prediction=False,
                physical_cause_identified=False)


def contrasts(rows):
    """Adjacent-condition comparisons, not paired within-process causal claims."""
    lookup={}
    for r in rows:
        match=re.search(r'diagnostic-r(\d+)-',r['job_id'])
        require(match is not None,'diagnostic round label absent')
        key=(r['program_key'],int(match.group(1)),r['memory_mode'],r['mode'],r['instrumentation'],r['history_intervention'])
        require(key not in lookup,'duplicate planned diagnostic condition')
        lookup[key]=r
    pairs=[]
    for key,a in sorted(lookup.items()):
        program,repeat,memory,mode,instrument,intervention=key
        plans=[]
        if intervention=='none':
            if mode=='target-hot': plans.append(('full_minus_hot',(program,repeat,memory,'full',instrument,'none')))
            if memory=='pageable': plans.append(('registered_minus_pageable',(program,repeat,'registered',mode,instrument,'none')))
            if instrument=='bare':
                plans += [(f'{i}_minus_bare',(program,repeat,memory,mode,i,'none')) for i in ('stamped','cupti')]
            if memory=='pageable' and mode=='full' and instrument=='cupti':
                plans.append(('source_read_minus_original',(program,repeat,memory,mode,instrument,'source-read')))
        for kind,other in plans:
            if other not in lookup: continue
            b=lookup[other]; per_action=[]
            ai={x['action_index']:x for x in a['actions']}
            for y in b['actions']:
                if y['action_index'] not in ai: continue
                x=ai[y['action_index']]
                if not (x['focus_2MiB_H2D'] or y['focus_2MiB_H2D']): continue
                changes={}
                for field in ('wall_ns','copy_call_ns','gpu_sync_ns','cpu_sync_ns'):
                    first,last=x[field]['mean'],y[field]['mean']
                    if first is not None and last is not None:
                        changes[field]=dict(first=first,last=last,delta=last-first,relative=last/first-1 if first>0 else None)
                if 'physical' in x and 'physical' in y:
                    for field in ('device_memcpy_union_ns','pre_device_wall_ns','post_device_wall_ns'):
                        first,last=x['physical'][field]['mean'],y['physical'][field]['mean']
                        if first is not None and last is not None: changes[field]=dict(first=first,last=last,delta=last-first,relative=last/first-1 if first>0 else None)
                per_action.append(dict(action_index=y['action_index'],changes=changes))
            first,last=a['replay_time_ns']['mean'],b['replay_time_ns']['mean']
            pairs.append(dict(comparison=kind,program_key=program,round=repeat,
                first_job=a['job_id'],last_job=b['job_id'],memory_mode=memory,mode=mode,instrumentation=instrument,
                replay_time_first_ns=first,replay_time_last_ns=last,replay_time_ratio_minus_one=last/first-1,
                replay_boundaries_are_equivalent=kind not in ('full_minus_hot','source_read_minus_original'),
                action_changes=per_action,same_process_comparison=False,
                unique_physical_cause_identified=False))
    return pairs


def summarize_diagnostics(out):
    out=Path(out).resolve(); m=read(out/'manifest.json'); jobs=[j for j in m['jobs'] if j['split']=='diagnostic']
    rows=[]; failures=[]
    for job in jobs:
        folder=out/'jobs'/job['id']; path=folder/'execution.json'
        if not path.exists(): failures.append(dict(job_id=job['id'],status='not_collected')); continue
        execution=read(path)
        if execution['status']!='completed': failures.append(dict(job_id=job['id'],status=execution['status'],error=execution.get('error'))); continue
        raw=folder/'result.json'; native=read(raw)
        require(native['request']==job['request'],'diagnostic request changed')
        row=summarize_native(native); row.update(job_id=job['id'],program_key=job['program_key'],native=receipt(raw))
        rows.append(row)
    result=dict(schema='h2d-history-diagnostic-analysis-1',analysis_source=receipt(Path(__file__)),manifest=receipt(out/'manifest.json'),
                declared=len(jobs),completed=len(rows),failures=failures,rows=rows,
                contrasts=contrasts(rows),
                interpretation=dict(inclusive_copy_is_enqueue_only=False,
                    host_and_device_durations_additive=False,
                    memory_capacity_measures_bandwidth=False,
                    endpoint_thread_CPU_separates_running_ready_waiting=False,
                    CPU_cycles_are_useful_FLOPs=False,
                    diagnostic_capture_qualifies_uninstrumented_timing=False,
                    unmatched_memcpy_means_zero_device_time=False,
                    unique_physical_cause_identified=False,
                    whole_target_time_used_for_parameter_fitting=False))
    write(out/'diagnostic-summary.json',result)
    return dict(path=str(out/'diagnostic-summary.json'),declared=len(jobs),completed=len(rows),
                contrasts=len(result['contrasts']),failures=len(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
    import json
    print(json.dumps(summarize_diagnostics(a.out)))


if __name__=='__main__': main()
