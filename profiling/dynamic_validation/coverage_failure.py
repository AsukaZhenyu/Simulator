"""Seal a profile-only support failure and plot its observed timing evidence.

No missing cost is imputed. Cross-boundary inclusive responses are not joint
steady-state prices, GPU kernel times or evidence identifying a deadlock cause.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from fractions import Fraction

from profiling.dynamic_validation import external


def joint_intervals(request):
    rows=[]
    for cpu_index,(ca,cb) in enumerate(request['cpu_actor']['intervals_ms']):
        for gpu_index,(ga,gb) in enumerate(request['gpu_actor']['intervals_ms']):
            first,last=max(ca,ga)*1000000,min(cb,gb)*1000000
            if first<last:rows.append(dict(begin_ns=first,end_ns=last,cpu_interval_index=cpu_index,gpu_interval_index=gpu_index,
                planned_cpu_off_ns=cb*1000000,planned_gpu_off_ns=gb*1000000))
    return sorted(rows,key=lambda r:r['begin_ns'])


def seal(out):
    out=Path(out).resolve();read,receipt,require=external.read,external.receipt,external.require
    manifest=read(out/'manifest.json');support=read(out/'fit-support.json');executions=read(out/'profile-executions.json')
    require(not support['all_heldout_actions_supported'],'not a support-failed campaign')
    heldout=[j for j in manifest['jobs'] if j['split']=='heldout'];profile=[j for j in manifest['jobs'] if j['split']=='profile']
    require({j['id'] for j in profile}=={e['job']['id'] for e in executions},'original profile denominator differs')
    require(not (out/'heldout-started.json').exists() and not any((out/'jobs'/j['id']).exists() for j in heldout),'heldout acquisition exists; cannot claim profile-only closure')
    require(not any((out/name).exists() for name in ('candidate-model.json','candidate-predictions.json','model-frozen.json','predictions-frozen.json','frozen-barrier.json','evaluation.json')),
        'fitting/freezing/evaluation artifact exists; cannot claim support-only closure')
    missing=[];affected_keys={}
    for workload in support['heldout_semantic_support']:
        for gap in workload['missing']:
            for state in gap['states']:
                require(tuple(state)==(1,1),'this diagnostic requires the exact joint-state gap')
                entry=next(r for r in support['entries'] if r['semantic_key']==gap['semantic_key'])
                missing.append(dict(workload=workload['workload'],action_index=gap['action_index'],semantic_key=gap['semantic_key'],
                    cpu_offered=state[0],gpu_offered=state[1],kind=entry['semantic']['kind'],device=entry['semantic'].get('device'),
                    semantic_op=entry['semantic'].get('semantic_op'),shape=entry['semantic'].get('shape')))
                affected_keys.setdefault(workload['workload'],set()).add(gap['action_index'])
    calls=[];processes=[];natives=[]
    for execution in executions:
        job=execution['job'];joints=joint_intervals(job['request'])
        if job['workload'] not in affected_keys or not joints or execution['status']!='completed':continue
        require(external.sha(execution['native']['path'])==execution['native']['sha256'],'affected profile raw changed')
        native=read(execution['native']['path']);natives.append(receipt(execution['native']['path']))
        epoch,hz=native['target_expected_origin_tick'],native['clock']['frequency_hz']
        relative=lambda tick:(tick-epoch)*1000000000//hz
        actors={}
        for device in ('cpu','gpu'):
            actors[device]=[]
            for interval in native['actors'][device]['intervals']:
                actors[device].append(dict(planned_begin_ns=interval['planned_begin_ms']*1000000,planned_end_ns=interval['planned_end_ms']*1000000,
                    actual_first_tick=interval['actual_first_tick'],actual_last_completion_tick=interval['actual_last_completion_tick'],
                    actual_first_relative_ns=relative(interval['actual_first_tick']),actual_last_completion_relative_ns=relative(interval['actual_last_completion_tick']),
                    calls=interval['calls'],start_lateness_ns=interval['start_lateness_ns'],last_completion_overrun_ns=interval['last_completion_overrun_ns']))
        joint_copies=[]
        for block in native['blocks']:
            for action in block['actions']:
                first,last=relative(action['begin_tick']),relative(action['end_tick']);index=action['action_index']
                if native['actions'][index]['kind']=='COPY' and external.state_at(job['request'],first)==(1,1):
                    joint_copies.append(dict(action_index=index,global_replay=action['global_replay'],begin_relative_ns=first,end_relative_ns=last,
                        wall_ns=action['wall_ns'],route=native['actions'][index]['src']+'->'+native['actions'][index]['dst']))
                if index not in affected_keys[job['workload']]:continue
                overlaps=[r for r in joints if max(first,r['begin_ns'])<min(last,r['end_ns'])]
                if not overlaps:continue
                intersections=[]
                for interval in overlaps:
                    cpu=actors['cpu'][interval['cpu_interval_index']];gpu=actors['gpu'][interval['gpu_interval_index']]
                    intersections.append(dict(interval,overlap_ns=min(last,interval['end_ns'])-max(first,interval['begin_ns']),
                        starts_inside_joint=interval['begin_ns']<=first<interval['end_ns'],
                        spans_planned_cpu_off=first<interval['planned_cpu_off_ns']<last,
                        completion_minus_planned_cpu_off_ns=last-interval['planned_cpu_off_ns'],
                        completion_minus_actual_cpu_last_completion_ns=last-cpu['actual_last_completion_relative_ns'],
                        actual_gpu_first_call_relative_ns=gpu['actual_first_relative_ns'],
                        completion_minus_actual_gpu_first_call_ns=last-gpu['actual_first_relative_ns']))
                calls.append(dict(job_id=job['id'],round=job['round'],workload=job['workload'],scenario=job['scenario'],
                    action_index=index,global_replay=action['global_replay'],begin_tick=action['begin_tick'],end_tick=action['end_tick'],
                    clock_frequency_hz=hz,target_expected_origin_tick=epoch,begin_relative_ns=first,end_relative_ns=last,wall_ns=action['wall_ns'],
                    offered_state_at_begin=list(external.state_at(job['request'],first)),offered_state_before_end=list(external.state_at(job['request'],max(first,last-1))),
                    intersects_planned_joint=True,joint_intersections=intersections,
                    release_spanning=any(r['spans_planned_cpu_off'] for r in intersections),
                    marginal_start_overlap=all(not r['starts_inside_joint'] and not r['spans_planned_cpu_off'] for r in intersections),
                    timing_scope='inclusive completed unchanged Executor.action; not pure GPU kernel time',
                    admitted_as_joint_steady_cost=False,native=execution['native']))
        processes.append(dict(job_id=job['id'],round=job['round'],target_expected_origin_tick=epoch,clock_frequency_hz=hz,
            target_begin_relative_ns=relative(native['target_begin_tick']),target_end_relative_ns=relative(native['target_end_tick']),
            planned_joint_intervals=joints,actors=actors,joint_started_copy_count=len(joint_copies),
            largest_joint_started_copies=sorted(joint_copies,key=lambda r:r['wall_ns'],reverse=True)[:3],native=execution['native']))
    cells=[s for row in support['entries'] for s in row['states']]
    completed=sum(e['status']=='completed' for e in executions)
    workload_status=[]
    for workload in support['heldout_semantic_support']:
        workload_status.append(dict(workload=workload['workload'],all_exact_response_states_supported=workload['supported'],
            status='not_evaluated_after_campaign_gate' if workload['supported'] else 'unsupported_cost_coverage',
            declared_heldout_processes=sum(j['workload']==workload['workload'] for j in heldout),completed_heldout_processes=0))
    result=dict(schema='dynamic-external-coverage-failure-1',closed_at_utc=datetime.now(timezone.utc).isoformat(),status='coverage_failed',
        declared_profile_processes=len(profile),completed_profile_processes=completed,
        semantic_coordinates=len(support['entries']),fully_supported_semantics=sum(r['all_four_states_known'] for r in support['entries']),
        declared_response_cells=len(cells),supported_response_cells=sum(c['mean_ns'] is not None for c in cells),missing_response_cells=missing,
        declared_heldout_processes=len(heldout),completed_heldout_processes=0,model_fitted=False,predictions_frozen=False,accuracy_evaluated=False,
        affected_calls=sorted(calls,key=lambda r:(r['round'],r['begin_tick'])),processes=processes,
        overlapping_calls=len(calls),release_spanning_calls=sum(r['release_spanning'] for r in calls),
        marginal_start_overlap_calls=sum(r['marginal_start_overlap'] for r in calls),
        workload_status=workload_status,predictions_generated=0,
        predictor_status={p:'not_fitted_or_evaluated_after_campaign_support_gate' for p in ('static','memoryless','stateful')},
        closure_decision='stop original preregistered campaign at exact-support gate; no cost imputation, supplementary acquisition, fitting, freezing or heldout acquisition',
        outcome='24 independent predictions not generated; independent accuracy acceptance not started',
        scope='profile-only coverage failure in controlled owned-actor Dense responses; hybrid coverage complete but entire comparison not evaluated',
        interpretation=dict(cross_boundary_responses_are_steady_joint_prices=False,pure_GPU_kernel_time_claimed=False,
            deadlock_or_OS_cause_identified=False,actual_actor_receipts_used_as_forecast_input=False,
            unresolved_mechanism='completion timing is associated with planned actor release; these observations do not identify its physical or runtime cause'),
        evidence=dict(manifest=receipt(out/'manifest.json'),profile_executions=receipt(out/'profile-executions.json'),
            profile_observations=receipt(out/'profile-observations.json'),fit_support=receipt(out/'fit-support.json'),affected_native=natives,
            generator=receipt(Path(__file__))))
    target=out/'coverage-failure.json';external.write(target,result)
    external.write(out/'support-closure.json',dict(schema='dynamic-external-support-closure-1',status='coverage_failed',
        coverage_failure=receipt(target),decision=result['closure_decision'],profile_processes=dict(completed=completed,declared=len(profile)),
        response_cells=dict(supported=result['supported_response_cells'],declared=result['declared_response_cells']),
        heldout_processes=dict(completed=0,declared=len(heldout)),model_fitted=False,predictions_frozen=False,accuracy_evaluated=False,
        original_campaign_preserved=True,additional_hardware_acquisition=False,unknown_costs_imputed=False,
        accuracy_failures_count=None,reason='accuracy acceptance was not started; ungenerated predictions are not numeric failures'))
    return dict(record=str(target),sha256=external.sha(target),supported_cells=result['supported_response_cells'],declared_cells=result['declared_response_cells'],
        semantic_complete=result['fully_supported_semantics'],semantic_declared=result['semantic_coordinates'],
        overlaps=result['overlapping_calls'],release_spanning=result['release_spanning_calls'],marginal=result['marginal_start_overlap_calls'])


def plot(record_path,out):
    """Plot only the sealed failure record; never reopen measurement outcomes."""
    from profiling.dynamic_validation.report import figure_setup,save
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    record_path=Path(record_path).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    record=external.read(record_path);external.require(record['schema']=='dynamic-external-coverage-failure-1','coverage failure schema differs')
    plt,font=figure_setup();fig,axes=plt.subplots(len(record['processes']),1,figsize=(10.0,7.0),sharex=True,squeeze=False)
    for ax,process in zip(axes.flat,record['processes']):
        for interval in process['planned_joint_intervals']:ax.axvspan(max(-2.2,interval['begin_ns']/1e9),interval['end_ns']/1e9,color='#DFE5EC',alpha=.6)
        for device,y,color in (('cpu',3,'#B1764D'),('gpu',2,'#387CA0')):
            for interval in process['actors'][device]:
                a,b=interval['planned_begin_ns']/1e9,interval['planned_end_ns']/1e9
                ax.broken_barh([(a,b-a)],(y-.20,.40),facecolors=color)
                ax.plot(interval['actual_first_relative_ns']/1e9,y,marker='v',color='#111111',markersize=4)
                if device=='cpu':ax.axvline(b,color='#555555',linestyle=':',linewidth=.75)
        for call in (r for r in record['affected_calls'] if r['job_id']==process['job_id']):
            a,b=call['begin_relative_ns']/1e9,call['end_relative_ns']/1e9
            if call['release_spanning']:
                ax.broken_barh([(a,b-a)],(.80,.40),facecolors='#A94442')
                ax.plot(b,1,marker='o',color='#6E1D1D',markersize=3)
                ax.text((a+b)/2,1,f"{call['wall_ns']/1e9:.3f} s",ha='center',va='center',color='white',fontsize=8)
            else:
                ax.plot(b,1,marker='D',color='#6E569A',markersize=4)
                ax.annotate(f"{sum(r['overlap_ns'] for r in call['joint_intersections'])/1e6:.3f} ms joint overlap",(b,1),xytext=(-45,-27),textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',linewidth=.5,color='#6E569A'))
        ax.set_yticks([1,2,3],['Attention response','GPU offered (planned)','CPU offered (planned)'])
        ax.set_ylim(.35,3.55);ax.set_title('Profile round '+str(process['round'])+' — observed inclusive GPU Attention completion')
        ax.grid(axis='x',alpha=.18)
    axes[-1,0].set_xlim(-2.2,17.2);axes[-1,0].set_xticks([-2,0,2,4,6,8,10,12,14,16]);axes[-1,0].set_xlabel('Time from planned target epoch (s)')
    handles=[Patch(color='#B1764D',label='Planned CPU pulse'),Patch(color='#387CA0',label='Planned GPU pulse'),
        Patch(color='#A94442',label='Release-spanning response'),Line2D([0],[0],marker='D',color='#6E569A',linestyle='None',label='Marginal boundary overlap'),
        Line2D([0],[0],marker='v',color='#111111',linestyle='None',label='Actual actor first call')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.55,-.01),fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));name='external-coverage-failure-timeline';artifacts=save(fig,out,name);plt.close(fig)
    figures=[dict(figure=name+'.png',artifacts=artifacts)]
    selected=[r for r in record['affected_calls'] if r['release_spanning']]
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for index,call in enumerate(selected):
        interval=next(i for i in call['joint_intersections'] if i['spans_planned_cpu_off'])
        completion=interval['completion_minus_planned_cpu_off_ns']/1e6
        actual_stop=completion-interval['completion_minus_actual_cpu_last_completion_ns']/1e6
        ax.plot([actual_stop,completion],[index,index],color='#BFC8D0',linewidth=1)
        ax.plot(actual_stop,index,marker='x',color='#222222',label='Actual CPU last completion' if index==0 else None)
        ax.plot(completion,index,marker='o',color='#A94442',label='Attention completion' if index==0 else None)
    ax.axvline(0,color='#555555',linestyle=':',linewidth=.9)
    ax.set_yticks(range(len(selected)),['Round '+str(r['round'])+' / replay '+str(r['global_replay']) for r in selected]);ax.invert_yaxis()
    ax.set_xlabel('Completion offset from planned CPU OFF (ms)');ax.set_title('Six release-spanning responses; temporal association, not causal identification')
    ax.grid(axis='x',alpha=.18);ax.legend(loc='best');fig.tight_layout()
    name='external-coverage-release-offsets';artifacts=save(fig,out,name);plt.close(fig);figures.append(dict(figure=name+'.png',artifacts=artifacts))
    return dict(status='coverage_failed',coverage_failure=external.receipt(record_path),providers={},unsupported_providers={},figures=figures,
        scope=record['scope'],font=font)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('seal','plot'));p.add_argument('--out',required=True);p.add_argument('--record');a=p.parse_args()
    result=seal(a.out) if a.command=='seal' else plot(a.record,a.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)
