"""Inspect exact response support before selecting any dynamic predictor.

Only completed-action PROFILE observations estimate prices. Process totals,
heldout outcomes and action/layer/strategy identifiers cannot become keys.
This produces a calibration support report, not frozen target predictions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from profiling.dynamic_validation import external


STATES=((0,0),(1,0),(0,1),(1,1))


def combine(records):
    """Equal process-cell means; workload duration cannot overweight a process."""
    grouped=defaultdict(list);semantics={}
    for record in records:
        for cell in record['cells']:
            semantic_key=cell['semantic_key']
            if digest(cell['semantic'])!=semantic_key:raise ValueError('changed normalized semantic coordinate')
            semantics[semantic_key]=cell['semantic'];state=(cell['cpu_offered'],cell['gpu_offered'])
            if state not in STATES or cell['count']<1:raise ValueError('unknown offered state or empty action evidence')
            value=Fraction(cell['sum_ns'],cell['count'])
            if value<0:raise ValueError('negative completed-action response')
            grouped[semantic_key,state].append(dict(job_id=record['job']['id'],native=record['native'],
                count=cell['count'],mean_ns=str(value),action_indices=cell['action_indices']))
    rows=[]
    for semantic_key,semantic in semantics.items():
        states=[]
        for state in STATES:
            sources=grouped.get((semantic_key,state),[]);values=[Fraction(s['mean_ns']) for s in sources]
            price=sum(values,Fraction())/len(values) if values else None
            states.append(dict(cpu_offered=state[0],gpu_offered=state[1],mean_ns=str(price) if price is not None else None,
                process_cells=len(values),sample_count=sum(s['count'] for s in sources),
                minimum_process_cell_mean_ns=str(min(values)) if values else None,
                maximum_process_cell_mean_ns=str(max(values)) if values else None,
                sources=sources))
        rows.append(dict(semantic_key=semantic_key,semantic=semantic,states=states,
            all_four_states_known=all(s['mean_ns'] is not None for s in states)))
    return rows


def inspect(out):
    out=Path(out).resolve();read,sha,require=external.read,external.sha,external.require
    m=read(out/'manifest.json');external.check(m)
    require(not (out/'heldout-started.json').exists(),'calibration inspection after heldout acquisition rejected')
    require(not any((out/'jobs'/j['id']).exists() for j in m['jobs'] if j['split']=='heldout'),'heldout data must not exist before selecting model')
    executions=read(out/'profile-executions.json')
    require(len(executions)==24 and all(e['status']=='completed' for e in executions),'all24original profiles required, no silent failed-cell omission')
    observation=read(out/'profile-observations.json');require(observation['completed']==observation['declared']==24,'all24profile summaries required')
    for record in observation['records']:require(sha(record['native']['path'])==record['native']['sha256'],'profile raw changed')
    entries=combine(observation['records']);by_key={r['semantic_key']:r for r in entries}
    support=[]
    for workload in external.WORKLOADS:
        program=read(m['programs'][workload+'-heldout']['program']['path'])
        # Only mathematical metadata is read: no heldout native result or time.
        missing=[]
        for index in range(len(program['actions'])):
            key=digest(semantic_action(program,index));row=by_key.get(key)
            if row is None:missing.append(dict(action_index=index,semantic_key=key,states='all'))
            else:
                unknown=[(s['cpu_offered'],s['gpu_offered']) for s in row['states'] if s['mean_ns'] is None]
                if unknown:missing.append(dict(action_index=index,semantic_key=key,states=unknown))
        support.append(dict(workload=workload,declared_actions=len(program['actions']),missing=missing,supported=not missing))
    coverage=[];actor_delays=defaultdict(list)
    for record in observation['records']:
        job=record['job'];native=read(record['native']['path']);hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick']
        begin=(native['target_begin_tick']-epoch)*1e9/hz;end=(native['target_end_tick']-epoch)*1e9/hz
        intervals=[]
        for device in ('cpu','gpu'):
            for pulse in native['actors'][device]['intervals']:
                requested_start=pulse['planned_begin_ms']*1000000;requested_end=pulse['planned_end_ms']*1000000
                intervals.append(dict(device=device,planned_begin_ns=requested_start,planned_end_ns=requested_end,
                    target_requested_overlap_ns=max(0.,min(end,requested_end)-max(begin,requested_start)),
                    target_after_release_ns=max(0.,end-max(begin,requested_end))))
                actor_delays[device+'_start_ns'].append(pulse['start_lateness_ns'])
                actor_delays[device+'_stop_overrun_ns'].append(pulse['last_completion_overrun_ns'])
        coverage.append(dict(job_id=job['id'],target_begin_relative_ns=begin,target_end_relative_ns=end,
            actor_intervals=intervals,transition_action_count=record['transition_action_count'],
            all_planned_actor_pulses_overlap_target=all(i['target_requested_overlap_ns']>0 for i in intervals),
            complete_target_used_as_fit_label=False))
    result=dict(schema='dynamic-external-fit-support-1',profile_observations=external.receipt(out/'profile-observations.json'),
        entries=entries,heldout_semantic_support=support,profile_phase_coverage=coverage,
        actor_receipt_delays={name:dict(samples=len(values),minimum_ns=min(values),median_ns=statistics.median(values),
            maximum_ns=max(values),values_ns=values) for name,values in actor_delays.items()},
        cost_estimator='equal process-cell means of per-action PROFILE labels; sample counts and original sources retained',
        coordinate_excludes=['action ordinal','layer ID','strategy ID','input seed','complete target latency'],
        all_heldout_actions_supported=all(s['supported'] for s in support),
        missing_cells_use_fallback=False,models_selected=False,predictions_frozen=False,
        future_target_measurements_read=False,whole_target_used_for_fit=False,
        next_step='select only an interpretable response/history hypothesis supported by profile transitions; freeze every parameter and own-timeline prediction before heldout')
    external.write(out/'fit-support.json',result)
    return dict(report=str(out/'fit-support.json'),semantics=len(entries),all_heldout_actions_supported=result['all_heldout_actions_supported'],
        processes_with_all_actor_pulse_overlap=sum(r['all_planned_actor_pulses_overlap_target'] for r in coverage))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(inspect(a.out),ensure_ascii=False),flush=True)
