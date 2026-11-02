"""Describe frozen forecast/measured block alignment; never reanchor forecasts."""
from fractions import Fraction
import argparse
import json
from pathlib import Path

from . import continuous_macro_campaign as campaign


def diagnose(out):
    out=Path(out).resolve();m=campaign.read(out/'manifest.json');evaluation=campaign.read(out/'INDEPENDENT_EVALUATION.json')
    before={name:campaign.sha(out/name) for name in ('component-profile.json','frozen-predictions.json','frozen-barrier.json')}
    cases={r['job_id']:r for r in evaluation['rows'] if r['provider']=='memoryless_four_state_macro'};rows=[]
    for job in (j for j in m['jobs'] if j['split']=='heldout'):
        native,execution,_=campaign.verified_job(out,job,m);case=cases[job['id']];hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick'];pred_begin=0;blocks=[]
        edges=sorted({v*1000000 for d in ('cpu','gpu') for pair in job['request'][d+'_actor']['intervals_ms'] for v in pair})
        for index,(block,pred_duration,error) in enumerate(zip(native['blocks'],case['predicted_block_ns'],case['block_APEs'])):
            actual_begin=Fraction((block['begin_tick']-epoch)*10**9,hz);actual_end=Fraction((block['end_tick']-epoch)*10**9,hz);pred_end=pred_begin+pred_duration
            actual_edges=[v for v in edges if actual_begin<v<actual_end];pred_edges=[v for v in edges if pred_begin<v<pred_end]
            blocks.append(dict(block=index,absolute_percentage_error=error,predicted_begin_ms=pred_begin/1e6,predicted_end_ms=pred_end/1e6,
                measured_begin_ms=float(actual_begin)/1e6,measured_end_ms=float(actual_end)/1e6,start_phase_error_ms=float(pred_begin-actual_begin)/1e6,
                end_phase_error_ms=float(pred_end-actual_end)/1e6,predicted_duration_ms=pred_duration/1e6,measured_duration_ms=block['wall_ns']/1e6,
                predicted_plan_edges_ms=[v/1e6 for v in pred_edges],measured_plan_edges_ms=[v/1e6 for v in actual_edges],plan_edge_assignment_differs=actual_edges!=pred_edges))
            pred_begin=pred_end
        rows.append(dict(job_id=job['id'],accuracy_passed=case['accuracy_passed'],total_APE=case['mean_response_APE'],block_APE_P90=case['block_APE_P90'],
            initial_actual_lateness_ms=native['initial_start_lateness_ns']/1e6,blocks=blocks,worst_two=sorted(blocks,key=lambda b:b['absolute_percentage_error'],reverse=True)[:2],
            block_edges_describe_only=True,actual_time_used_to_reanchor_prediction=False))
    after={name:campaign.sha(out/name) for name in before};campaign.require(before==after,'diagnosis modified predictor')
    result=dict(schema='continuous-block-phase-diagnosis-1',rows=rows,predictor_unchanged=before==after,frozen_files=after,
        all_logical_blocks_retained_in_accuracy=True,forecasts_reanchored=False,parameters_refitted=False,
        limits='alignment diagnosis distinguishes misplaced slow intervals from total error; it does not identify the physical cause of timing drift')
    campaign.write(out/'BLOCK_PHASE_DIAGNOSIS.json',result)
    return dict(output=campaign.receipt(out/'BLOCK_PHASE_DIAGNOSIS.json'),failed_case_worst_blocks=[dict(job_id=r['job_id'],worst_two=r['worst_two']) for r in rows if not r['accuracy_passed']],predictor_unchanged=before==after)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(diagnose(a.out),ensure_ascii=False))
