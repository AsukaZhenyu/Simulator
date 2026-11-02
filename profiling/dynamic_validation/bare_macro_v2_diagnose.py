"""Read-only QPC/lifecycle diagnosis of the frozen independent macro test.

Measured boundaries are used only to describe residuals. They never change a
forecast, reorder providers, fit parameters, or remove crossing actions from
the accuracy evaluation. State labels mean the declared offered-work plan.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from . import bare_macro_v2_campaign as campaign


def planned_state(request,at_ns):
    return tuple(int(any(a*1000000<=at_ns<b*1000000 for a,b in request[d+'_actor']['intervals_ms'])) for d in ('cpu','gpu'))


def diagnose(out):
    out=Path(out).resolve();manifest=campaign.check(out);barrier=campaign.read(out/'frozen-barrier.json')
    frozen_before={name:campaign.sha(out/name) for name in ('component-profile.json','frozen-predictions.json','frozen-barrier.json')}
    campaign.require(barrier['manifest_sha256']==campaign.sha(out/'manifest.json'),'mismatched freeze')
    for name,value in barrier['files'].items():campaign.require(campaign.sha(out/name)==value,'frozen predictor changed')
    profile=campaign.read(out/'component-profile.json');prices={tuple(c['state']):Fraction(c['mean_ns']) for c in profile['cells']}
    rows=[];all_groups=defaultdict(list);all_cross=[]
    for job in (j for j in manifest['jobs'] if j['split']=='heldout'):
        native,execution,path=campaign.verified_job(out,job,manifest);hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick']
        actions=[a for b in native['blocks'] for a in b['actions']];campaign.require(len(actions)==256,'all independent component actions required')
        edges=sorted({value*1000000 for d in ('cpu','gpu') for pair in job['request'][d+'_actor']['intervals_ms'] for value in pair})
        previous=native['target_begin_tick'];gaps=[];raw=[];groups=defaultdict(list);crossing=[];sum_ticks=0
        for index,a in enumerate(actions):
            begin,end=a['begin_tick'],a['end_tick'];campaign.require(previous<=begin<end<=native['target_end_tick'],'overlapping/outside action interval')
            gaps.append(dict(before_replay=index,begin_tick=previous,end_tick=begin,duration_ns=float(Fraction((begin-previous)*10**9,hz))))
            sum_ticks+=end-begin;start=Fraction((begin-epoch)*10**9,hz);stop=Fraction((end-epoch)*10**9,hz)
            crossed=[edge for edge in edges if start<edge<stop];state=planned_state(job['request'],start);duration=float(Fraction((end-begin)*10**9,hz))
            raw.append(dict(global_replay=index,begin_relative_plan_ns=float(start),end_relative_plan_ns=float(stop),response_ns=duration,
                planned_state_at_entry=state,contains_planned_transition=bool(crossed),crossed_edges_ns=crossed))
            if crossed:crossing.append(index);all_cross.append(dict(job_id=job['id'],global_replay=index))
            else:groups[state].append(duration);all_groups[state].append(duration)
            previous=end
        gaps.append(dict(before_replay='suffix',begin_tick=previous,end_tick=native['target_end_tick'],duration_ns=float(Fraction((native['target_end_tick']-previous)*10**9,hz))))
        gap_ticks=sum(g['end_tick']-g['begin_tick'] for g in gaps);total_ticks=native['target_end_tick']-native['target_begin_tick'];campaign.require(sum_ticks+gap_ticks==total_ticks,'complete QPC partition does not close')
        complete=float(Fraction(total_ticks*10**9,hz));sum_ns=float(Fraction(sum_ticks*10**9,hz));gap_ns=float(Fraction(gap_ticks*10**9,hz))
        comparisons=[]
        for state in campaign.STATES:
            values=groups[state];mean=statistics.mean(values) if values else None
            comparisons.append(dict(state=state,contained_actions=len(values),component_profile_mean_ns=float(prices[state]),
                heldout_contained_action_mean_ns=mean,heldout_to_calibration_ratio=mean/float(prices[state]) if mean is not None else None,
                raw_response_population_CV=statistics.pstdev(values)/mean if len(values)>1 else None,
                no_contained_action_is_unknown_not_zero=not values))
        rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],native=execution['native'],execution=campaign.receipt(path),
            total_wall_ns=complete,sum_completed_action_response_ns=sum_ns,all_outside_action_gaps_ns=gap_ns,outside_action_gap_fraction=gap_ticks/total_ticks,
            QPC_partition_closes=True,gap_segments=gaps,all_action_count=256,cross_transition_action_count=len(crossing),cross_transition_action_indices=crossing,
            same_state_comparisons=comparisons,raw_action_rows=raw,all_actions_retained_in_primary_accuracy=True,
            observed_boundaries_used_in_forecast=False))
    frozen_after={name:campaign.sha(out/name) for name in frozen_before};campaign.require(frozen_before==frozen_after,'diagnosis modified frozen predictor')
    result=dict(schema='bare-macro-v2-independent-lifecycle-diagnosis-1',purpose='read-only residual localization, no parameter update',rows=rows,
        declared_heldout_processes=6,all_complete_QPC_partitions_close=all(r['QPC_partition_closes'] for r in rows),
        aggregate_same_state_comparisons=[dict(state=state,contained_actions=len(all_groups[state]),component_profile_mean_ns=float(prices[state]),
            heldout_contained_action_mean_ns=statistics.mean(all_groups[state]) if all_groups[state] else None,
            heldout_to_calibration_ratio=statistics.mean(all_groups[state])/float(prices[state]) if all_groups[state] else None) for state in campaign.STATES],
        transition_crossing_actions=all_cross,transition_crossing_actions_removed_from_accuracy=False,raw_slow_actions_removed=False,
        actual_schedule_running_ready_blocked_identified=False,observed_plan_labels_are_available_service=False,
        prices_and_predictions_unchanged=frozen_before==frozen_after,frozen_files=frozen_after,
        observed_action_boundaries_used_for_prediction=False,parameters_refitted=False,
        limits='same-state labels use the declared offered-work plan; no physical utilization or causal OS mechanism is identified')
    campaign.write(out/'LIFECYCLE_DIAGNOSIS.json',result)
    return dict(output=campaign.receipt(out/'LIFECYCLE_DIAGNOSIS.json'),QPC_closed=result['all_complete_QPC_partitions_close'],
        gap_fractions=[dict(job_id=r['job_id'],gap_fraction=r['outside_action_gap_fraction']) for r in rows],
        same_state_comparisons=result['aggregate_same_state_comparisons'],predictor_unchanged=result['prices_and_predictions_unchanged'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(diagnose(a.out),ensure_ascii=False))

if __name__=='__main__':main()
