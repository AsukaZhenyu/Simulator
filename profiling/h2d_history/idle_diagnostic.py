"""Eight declared controlled idle/H2D diagnostics using the frozen native probe.

Sleep(5) changes more than PCIe state. This experiment can establish an idle
response difference but cannot uniquely attribute it to link power management.
Complete replay totals include the requested idle and are never accuracy data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics

from profiling.dac_validation import e2a
from profiling.h2d_history import analysis, campaign as history_campaign

ROOT = Path(__file__).resolve().parents[2]
read, write, receipt, sha, require = e2a.read, e2a.write, e2a.receipt, e2a.sha, e2a.require


def prepare(out, campaign, clock_evidence=None):
    out, campaign = Path(out).resolve(), Path(campaign).resolve()
    require(not out.exists(), 'new idle diagnostic output directory required')
    parent = read(campaign/'manifest.json'); history_campaign.check(parent)
    clock = Path(clock_evidence).resolve() if clock_evidence else campaign.parent/'clock-v1/summary.json'
    require(clock.is_file(), 'retained clock evidence required')
    clock_summary = read(clock)
    require(clock_summary.get('schema')=='h2d-history-clock-diagnostic-summary-1', 'clock evidence schema differs')
    key = 'l2-h512-p64-s2-profile'
    binding = parent['programs'][key]
    for field in ('program','reference'):
        require(sha(binding[field]['path'])==binding[field]['sha256'], 'retained mathematical input changed')
    program = read(binding['program']['path'])
    target = next(i for i,a in enumerate(program['actions']) if a['kind']=='COPY'
        and a['src']=='dram' and a['dst']=='vram' and a['target_id'].endswith(':W_up'))
    descriptor = next(t for t in program['graph']['workload']['tensors']
        if t['id']==program['actions'][target]['target_id'])
    require(descriptor['storage_bytes']==2097152, 'actual two-MiB W_up target required')
    source = next(s['source'] for s in parent['sources']
        if s['source']['path'].replace('\\','/').endswith('/h2d_history/native/history_probe.cpp'))
    out.mkdir(parents=True)
    binary = out/'executed_binary'/Path(parent['binary']['path']).name
    binary.parent.mkdir(); shutil.copy2(parent['binary']['path'],binary)
    sources = list(parent['sources'])
    for path in (Path(__file__).resolve(),Path(analysis.__file__).resolve()):
        destination = out/'executed_sources'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,destination)
        sources.append(dict(source=receipt(path),archived=receipt(destination)))
    jobs=[]
    for round_index in range(2):
        memories = ('pageable','registered') if round_index==0 else ('registered','pageable')
        for memory in memories:
            interventions = ('none','idle') if (round_index+int(memory=='registered'))%2==0 else ('idle','none')
            for intervention in interventions:
                sampling = e2a.request('action',trials=16,target=target)
                sampling['repeats']=8
                request = dict(schema='h2d-history-request-1',source_sha256=source['sha256'],
                    sampling=sampling,expected_process_mask=4294967295,mode='target-hot',
                    instrumentation='cupti',memory_mode=memory,target_action_index=target,
                    history_intervention=intervention)
                jobs.append(dict(id=f'diagnostic-r{round_index}-{key}-{memory}-target-hot-{intervention}',
                    split='diagnostic',program_key=key,request=request))
    require(len(jobs)==8,'exactly eight idle diagnostic processes required')
    manifest=dict(schema='h2d-history-campaign-1',study='controlled-idle-response-diagnostic',
        prepared_at_utc=e2a.utc(),parent_manifest=receipt(campaign/'manifest.json'),
        prior_clock_evidence=receipt(clock),binary=receipt(binary),original_binary=parent['binary'],
        sources=sources,programs={key:binding},runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],
        baseline_profile=parent.get('baseline_profile'),baseline_context=parent.get('baseline_context'),
        jobs=jobs,target_action=dict(action_index=target,action=program['actions'][target],
            bytes=descriptor['storage_bytes']),
        protocol=dict(counts=dict(diagnostic=8,profile=0,heldout=0),quality=parent['protocol']['quality'],
            retries=0,timeout_seconds=300,repeats=8,trials=16,warmup_min_ms=500,
            instrumentation='same CUPTI/action stamps in both intervention conditions',
            reset_and_prime='one reset plus one full prime before every block; unchanged native implementation',
            idle='existing native Sleep(5) immediately before every selected target; actual intervals retained',
            hypothesized_response='Five-ms requested idle may increase target H2D response or pre-device time relative to adjacent no-idle processes.',
            competing_explanations=['PCIe/device operating-state transitions','CPU readiness or scheduling changes',
                'driver/runtime preparation or admission changes','other unobserved run-state differences'],
            acceptance_scope='diagnostic target-action and correlated device/API contrasts; not complete strategy qualification',
            whole_block_time_used_for_conclusion=False,refit_existing_frozen_model=False,
            target_totals_used_for_fit=False,future_target_state_used=False,
            unique_link_power_cause_identifiable=False,resource_settings_modified=False))
    write(out/'manifest.json',manifest)
    return dict(out=str(out),declared_processes=len(jobs),target_action_index=target,
        prior_clock_evidence=manifest['prior_clock_evidence'])


def describe(values):
    values=list(values)
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        min=min(values) if values else None,max=max(values) if values else None)


def difference(first,last):
    return dict(no_idle=first,idle=last,delta=last-first if first is not None and last is not None else None,
        relative=last/first-1 if first is not None and last is not None and first>0 else None)


def summarize(out):
    out=Path(out).resolve(); manifest=read(out/'manifest.json'); history_campaign.check(manifest)
    require(sha(manifest['prior_clock_evidence']['path'])==manifest['prior_clock_evidence']['sha256'], 'clock evidence changed')
    # Generic accounting is reused unchanged, including payload/correlation guards.
    if not (out/'diagnostic-summary.json').exists(): analysis.summarize_diagnostics(out)
    generic=read(out/'diagnostic-summary.json')
    target=manifest['target_action']['action_index']; summaries=[]; lookup={}
    jobs={job['id']:job for job in manifest['jobs']}
    for row in generic['rows']:
        job=jobs[row['job_id']]; request=job['request']
        native=read(row['native']['path'])
        action=next(a for a in row['actions'] if a['action_index']==target)
        physical=action.get('physical',{})
        invocations=physical.get('invocations',[])
        exact=[r for r in invocations if r['exact_payload_activity_coverage']]
        require(len(native['trials'])==16 and request['sampling']['repeats']==8, 'idle history contract changed')
        intervention_rows=native['intervention_intervals']
        if request['history_intervention']=='idle': require(len(intervention_rows)==128,'idle interval count differs')
        else: require(not intervention_rows,'no-idle run contains interventions')
        metrics=dict(host_wall_ns=action['wall_ns']['mean'],copy_call_ns=action['copy_call_ns']['mean'],
            gpu_sync_ns=action['gpu_sync_ns']['mean'],cpu_sync_ns=action['cpu_sync_ns']['mean'])
        for field in ('device_memcpy_union_ns','pre_device_wall_ns','post_device_wall_ns'):
            metrics[field]=statistics.mean(r[field] for r in exact) if exact else None
        phases={}
        for label,condition in (('first_target_each_block',lambda r:r['invocation']==0),
                                 ('remaining_targets',lambda r:r['invocation']>0)):
            subset=[r for r in exact if condition(r)]
            phases[label]={field:describe(r[field] for r in subset) for field in
                ('host_wall_ns','device_memcpy_union_ns','pre_device_wall_ns','post_device_wall_ns')}
        round_index=int(job['id'].split('diagnostic-r',1)[1].split('-',1)[0])
        summary=dict(job_id=job['id'],round=round_index,memory_mode=request['memory_mode'],
            history_intervention=request['history_intervention'],target_action_index=target,
            metrics=metrics,first_and_remaining=phases,
            exact_payload_coverage_count=len(exact),expected_invocations=128,
            all_target_device_activity_qualified=(len(exact)==128 and native['capture_integrity_pass']),
            actual_intervention_wall_ns=describe(r['wall_ns'] for r in intervention_rows),
            native=row['native'],native_action_clock_source='absolute QPC shared with CUPTI',
            whole_block_total_includes_idle=True,whole_block_time_used_as_accuracy=False)
        summaries.append(summary)
        lookup[(round_index,request['memory_mode'],request['history_intervention'])]=summary
    pairs=[]
    for round_index in range(2):
        for memory in ('pageable','registered'):
            no_idle=lookup.get((round_index,memory,'none')); idle=lookup.get((round_index,memory,'idle'))
            if no_idle is None or idle is None:
                pairs.append(dict(round=round_index,memory_mode=memory,status='incomplete'))
                continue
            pairs.append(dict(round=round_index,memory_mode=memory,status='complete',
                no_idle_job=no_idle['job_id'],idle_job=idle['job_id'],
                target_action_changes={field:difference(no_idle['metrics'][field],idle['metrics'][field])
                    for field in no_idle['metrics']},
                complete_activity_coverage_both=no_idle['all_target_device_activity_qualified'] and idle['all_target_device_activity_qualified'],
                same_process_comparison=False,whole_block_time_compared=False,
                uniquely_identifies_PCIe_power_management=False))
    result=dict(schema='h2d-history-idle-diagnostic-summary-1',manifest=receipt(out/'manifest.json'),
        generic_analysis=receipt(out/'diagnostic-summary.json'),prior_clock_evidence=manifest['prior_clock_evidence'],
        declared=8,completed=len(summaries),failures=generic['failures'],rows=summaries,paired_contrasts=pairs,
        frozen_model_refitted=False,hardware_accuracy_qualified=False,unique_physical_cause_identified=False,
        interpretation=[
            'Only target action and explicitly correlated physical intervals are compared; whole block includes requested idle.',
            'Idle may change host scheduling, driver readiness and device operating state simultaneously.',
            'Adjacent independently initialized processes are not a within-process causal comparison.',
            'The same CUPTI instrumentation controls observer type but does not prove zero or identical overhead.',
            'Physical differences are descriptive unless payload coverage is complete in both processes.',
            'No clock telemetry is collected here and no unique PCIe/power attribution follows from an idle effect.'])
    write(out/'idle-summary.json',result)
    return dict(summary=str(out/'idle-summary.json'),declared=8,completed=len(summaries))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','summarize'))
    parser.add_argument('--out',required=True);parser.add_argument('--campaign');parser.add_argument('--clock-evidence')
    args=parser.parse_args()
    if args.command=='prepare':
        require(args.campaign is not None,'--campaign required for prepare')
        result=prepare(args.out,args.campaign,args.clock_evidence)
    else:result=summarize(args.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
