"""Read-only independent trajectory residuals; no price/forecast changes."""
from collections import defaultdict
from fractions import Fraction
import argparse
import json
from pathlib import Path
import statistics

from . import continuous_macro_campaign as campaign
from .bare_macro_v2_diagnose import planned_state


def diagnose(out):
    out=Path(out).resolve();manifest=campaign.check(out);names=('component-profile.json','frozen-predictions.json','frozen-barrier.json');before={n:campaign.sha(out/n) for n in names}
    barrier=campaign.read(out/'frozen-barrier.json');campaign.require(barrier['manifest_sha256']==campaign.sha(out/'manifest.json'),'freeze scope differs')
    for name,value in barrier['files'].items():campaign.require(campaign.sha(out/name)==value,'frozen predictor changed')
    profile=campaign.read(out/'component-profile.json');prices={tuple(c['state']):Fraction(c['mean_ns']) for c in profile['cells']}
    rows=[];groups=defaultdict(list);all_cross=[]
    for job in (j for j in manifest['jobs'] if j['split']=='heldout'):
        native,execution,path=campaign.verified_job(out,job,manifest);hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick'];ns=lambda a,b:Fraction((b-a)*10**9,hz)
        actions=[a for b in native['blocks'] for a in b['actions']];campaign.require(len(actions)==256,'all256 actions required');previous=native['target_begin_tick'];gaps=[];sum_ticks=0;raw=[];local=defaultdict(list);crossing=[]
        edges=sorted({v*1000000 for d in ('cpu','gpu') for pair in job['request'][d+'_actor']['intervals_ms'] for v in pair})
        for index,action in enumerate(actions):
            a,b=action['begin_tick'],action['end_tick'];campaign.require(previous<=a<b<=native['target_end_tick'],'action clock order differs')
            gaps.append(dict(before_action=index,begin_tick=previous,end_tick=a,duration_ns=float(ns(previous,a))));sum_ticks+=b-a
            start,stop=ns(epoch,a),ns(epoch,b);cross=[edge for edge in edges if start<edge<stop];state=planned_state(job['request'],start);duration=float(ns(a,b))
            raw.append(dict(global_replay=index,begin_relative_plan_ns=float(start),end_relative_plan_ns=float(stop),response_ns=duration,planned_state_at_entry=state,crossed_edges_ns=cross))
            if cross:crossing.append(index);all_cross.append(dict(job_id=job['id'],global_replay=index))
            else:local[state].append(duration);groups[state].append(duration)
            previous=b
        gaps.append(dict(before_action='suffix',begin_tick=previous,end_tick=native['target_end_tick'],duration_ns=float(ns(previous,native['target_end_tick']))))
        gap_ticks=sum(g['end_tick']-g['begin_tick'] for g in gaps);total_ticks=native['target_end_tick']-native['target_begin_tick'];campaign.require(sum_ticks+gap_ticks==total_ticks,'whole trajectory QPC partition does not close')
        comparisons=[]
        for state in campaign.STATES:
            values=local[state];mean=statistics.mean(values) if values else None
            comparisons.append(dict(state=state,contained_actions=len(values),continuous_calibration_mean_ns=float(prices[state]),heldout_contained_action_mean_ns=mean,
                heldout_to_calibration_ratio=mean/float(prices[state]) if mean is not None else None,missing_same_state_actions_are_unknown=not values))
        rows.append(dict(job_id=job['id'],pulse_plan=job['pulse_plan'],native=execution['native'],execution=campaign.receipt(path),
            total_wall_ns=float(ns(native['target_begin_tick'],native['target_end_tick'])),sum_completed_action_response_ns=float(Fraction(sum_ticks*10**9,hz)),all_outside_action_gaps_ns=float(Fraction(gap_ticks*10**9,hz)),
            outside_action_gap_fraction=gap_ticks/total_ticks,QPC_partition_closes=True,gap_segments=gaps,all_action_count=256,cross_transition_action_count=len(crossing),
            cross_transition_action_indices=crossing,same_state_comparisons=comparisons,raw_action_rows=raw,all_actions_retained_in_accuracy=True))
    after={n:campaign.sha(out/n) for n in names};campaign.require(before==after,'diagnosis modified frozen predictors')
    result=dict(schema='continuous-Attention-independent-diagnosis-1',rows=rows,declared_heldout_processes=6,all_complete_QPC_partitions_close=all(r['QPC_partition_closes'] for r in rows),
        aggregate_same_state_comparisons=[dict(state=state,contained_actions=len(groups[state]),continuous_calibration_mean_ns=float(prices[state]),
            heldout_contained_action_mean_ns=statistics.mean(groups[state]) if groups[state] else None,
            heldout_to_calibration_ratio=statistics.mean(groups[state])/float(prices[state]) if groups[state] else None) for state in campaign.STATES],
        transition_crossing_actions=all_cross,transition_crossing_actions_removed_from_accuracy=False,raw_slow_actions_removed=False,
        planned_state_is_measured_available_service=False,observed_boundaries_used_for_prediction=False,parameters_refitted=False,
        prices_and_predictions_unchanged=before==after,frozen_files=after,
        limits='descriptive localization under planned offered-work states; no OS scheduling, queue or physical recovery mechanism identified')
    campaign.write(out/'LIFECYCLE_DIAGNOSIS.json',result)
    return dict(output=campaign.receipt(out/'LIFECYCLE_DIAGNOSIS.json'),QPC_closed=result['all_complete_QPC_partitions_close'],
        gap_fractions=[dict(job_id=r['job_id'],gap_fraction=r['outside_action_gap_fraction']) for r in rows],
        same_state_comparisons=result['aggregate_same_state_comparisons'],predictor_unchanged=before==after)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps(diagnose(args.out),ensure_ascii=False))
