"""Independent v2 cohort using owned high-resolution timer actuation.

The v1 sources, data, frozen models and negative results remain unchanged.
Reusable mathematical inputs have distinct profile/heldout seeds. V2 acquires
new independent timing observations; it does not relabel old v1 measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from profiling.dynamic_validation import idle_campaign as base

CONTROLLER='owned-high-resolution-waitable-timer-v1'


def precision(actual_ns,requested_ms):
    requested=requested_ms*1000000;overshoot=[a-requested for a in actual_ns]
    base.require(all(a+100>=requested for a in actual_ns),'owned timer returned before requested interval')
    med=statistics.median(overshoot);p90=sorted(overshoot)[math.ceil(.9*(len(overshoot)-1))]
    return dict(samples=len(actual_ns),requested_ns=requested,actual_mean_ns=statistics.mean(actual_ns),
        minimum_overshoot_ns=min(overshoot),median_overshoot_ns=med,p90_overshoot_ns=p90,maximum_overshoot_ns=max(overshoot),
        zero_control=requested_ms==0,qualified=requested_ms==0 or med<=250000 and p90<=1000000,
        limits=dict(median_overshoot_ns=250000,p90_overshoot_ns=1000000),
        observations_removed=False)


def prepare(out,binary,*,smoke=False):
    result=base.prepare(out,binary,smoke=smoke);out=Path(out).resolve();m=base.read(out/'manifest.json')
    base.require(not (out/'profile-started.json').exists() and not (out/'heldout-started.json').exists()
        and not (out/'frozen-barrier.json').exists() and not (out/'jobs').exists(),'cannot replace a started or frozen preparation')
    preliminary=out/'base-preparation-manifest.json'
    base.require(not preliminary.exists(),'base preparation receipt already exists')
    # The base writer is exclusive. Preserve its original receipt instead of
    # attempting a second exclusive write or overwriting acquired evidence.
    (out/'manifest.json').rename(preliminary)
    m['base_preparation_manifest']=base.receipt(preliminary)
    folder=base.ROOT/'profiling/dynamic_validation/native';source=folder/'idle_probe_v2.cpp'
    for path in (Path(__file__),source,folder/'owned_timer.hpp',folder/'idle_v2/CMakeLists.txt'):
        target=out/'executed_sources'/path.resolve().relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);m['sources'].append(dict(source=base.receipt(path),archived=base.receipt(target)))
    m['schema']='dynamic-idle-campaign-v2'
    m['v1_evidence_untouched']=True;m['new_independent_timing_acquisition']=True
    for job in m['jobs']:
        job['request']['source_sha256']=base.sha(source);job['request']['idle_controller']=CONTROLLER
    m['protocol'].update(controller=dict(kind=CONTROLLER,global_timeBeginPeriod=False,busy_spin=False,Sleep_fallback=False,
        median_overshoot_limit_ns=250000,p90_overshoot_limit_ns=1000000,
        actual_elapsed_idle_retained=True,failed_precision_rows_retained=True),
        primary_forecast='whole-ahead planned requested idle with frozen response parameters',
        secondary_actual_past_forecast='not included in this v2 cohort; actual completed past idle is diagnostic only',
        timing_split='new profile6 and heldout6 timing processes; retained profile/heldout mathematical buffers have different original seeds')
    m['protocol']['initial_state']+='; v2 owned high-resolution waitable timer, no global timer resolution changes'
    base.write(out/'manifest.json',m)
    return dict(result,controller=CONTROLLER,v1_unchanged=True)


def audit(out,split):
    out=Path(out).resolve();m=base.read(out/'manifest.json');base.check(m)
    base.require(m['schema']=='dynamic-idle-campaign-v2','v2 manifest required');records=[]
    for execution in base.read(out/(split+'-executions.json')):
        job=execution['job'];record=dict(job_id=job['id'],memory_mode=job['memory_mode'],status=execution['status'],qualified=False)
        if execution['status']=='completed':
            base.require(base.sha(execution['native']['path'])==execution['native']['sha256'],'native timer evidence changed')
            native=base.read(execution['native']['path']);controller=native['idle_controller']
            base.require(controller['kind']==CONTROLLER and controller['busy_spin'] is False and controller['Sleep_fallback'] is False
                and controller['global_timeBeginPeriod_called'] is False and controller['timer_tolerable_delay_ms']==0,'owned timer proof differs')
            conditions=[]
            for idle in job['request']['idle_ms']:
                values=[r['actual_idle_ns'] for r in native['sequences'] if r['requested_idle_ms']==idle]
                conditions.append(dict(requested_idle_ms=idle,**precision(values,idle)))
            record.update(conditions=conditions,qualified=all(r['qualified'] for r in conditions),native=execution['native'])
        else:record['error']=execution.get('error')
        records.append(record)
    result=dict(schema='dynamic-idle-controller-precision-v2',split=split,records=records,
        declared=sum(j['split']==split for j in m['jobs']),completed=sum(r['status']=='completed' for r in records),
        qualified_processes=sum(r['qualified'] for r in records),primary_uses_actual_elapsed_idle=False,
        failed_or_slow_actuation_rows_removed=False,physical_recovery_mechanism_identified=False)
    base.write(out/(split+'-controller-precision.json'),result)
    return dict(precision=str(out/(split+'-controller-precision.json')),declared=result['declared'],completed=result['completed'],qualified=result['qualified_processes'])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','profile','heldout','audit-profile','audit-heldout','summarize-profile','summarize-heldout','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--binary');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.out,a.binary,smoke=a.smoke)
    elif a.command in ('profile','heldout'):
        result=base.collect(a.out,a.command);result['controller_precision']=audit(a.out,a.command)
    elif a.command.startswith('audit-'):result=audit(a.out,a.command.split('-',1)[1])
    elif a.command.startswith('summarize-'):result=base.summarize(a.out,a.command.split('-',1)[1])
    else:result=base.evaluate(a.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
