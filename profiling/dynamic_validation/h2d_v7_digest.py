"""Small completed-profile diagnosis; no fit, GPU launch or report rewriting."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import statistics

from . import h2d_v7 as campaign
from .h2d_v7_analysis import process_summary,host_partition,describe


def link_key(sample):
    values=sample.get('values',{})
    names=('pcie_current_gen','pcie_current_width','pcie_max_gen','pcie_max_width')
    return tuple(values[n] for n in names) if all(type(values.get(n)) is int and values[n]>0 for n in names) else None


def summarize(out):
    out=Path(out).resolve();manifest=campaign.read(out/'manifest.json');executions=campaign.read(out/'profile-executions.json')
    data=dict(schema='dynamic-h2d-primitive-profile-digest-7',manifest=campaign.receipt(out/'manifest.json'),
              executions=campaign.receipt(out/'profile-executions.json'),declared_processes=4,completed_processes=sum(x['status']=='completed' for x in executions),
              all_native_processes_exited=all(x.get('process_exited_verified',False) for x in executions),
              processes=[],performance_parameters_fitted=False,heldout_executed=False,
              host_partition_is_not_GPU_DMA_partition=True,CUDA_events_are_not_QPC_aligned=True,
              CUDA_event_and_sync_response_additive=False,original_runtime_or_old_results_modified=False)
    combined=defaultdict(list);gates=defaultdict(lambda:[0,0]);baseline=[0,0];total_copies=0
    for execution in executions:
        if execution['status']!='completed':
            data['processes'].append(dict(job_id=execution['job_id'],status=execution['status'],error=execution.get('error')));continue
        path=Path(execution['native']['path']);campaign.require(campaign.sha(path)==execution['native']['sha256'],'completed native changed')
        native=campaign.read(path);summary=process_summary(native);total_copies+=summary['validation']['copies'];groups=defaultdict(list)
        for burst in native['bursts']:
            for copy in burst['copies']:
                key=(burst['bytes'],burst['requested_idle_ms'],burst['mode'],'first' if copy['ordinal']==0 else 'suffix15','early' if burst['trial']<6 else 'late')
                part=host_partition(copy,native['clock']['frequency_hz'])
                row=dict(wall_ns=copy['wall_ns'],event_span_ns=copy.get('device_event_span_ns'),parts=part.get('response_parts_ns'),outside_ns=part.get('outside_segments_ns'))
                groups[key].append(row)
        stages=[]
        for (size,idle,mode,ordinal,half),rows in sorted(groups.items()):
            parts=rows[0]['parts']
            stages.append(dict(bytes=size,requested_idle_ms=idle,mode=mode,ordinals=ordinal,trial_half=half,count=len(rows),
                               host_wall_ns=describe(x['wall_ns'] for x in rows),
                               response_parts_mean_ns={k:statistics.mean(x['parts'][k] for x in rows) for k in parts} if parts else None,
                               outside_segments_mean_ns=statistics.mean(x['outside_ns'] for x in rows) if parts else None,
                               adjacent_device_event_ns=describe(x['event_span_ns'] for x in rows) if mode=='primitive_events' else None,
                               event_span_is_pure_DMA=False,event_span_added_to_host_parts=False))
        link_groups=defaultdict(Counter);query_ages=[]
        for b in native['bursts']:
            prior,post=link_key(b['prior_pcie_before_prime']),link_key(b['post_pcie_diagnostic_only'])
            link_groups[(b['bytes'],b['requested_idle_ms'],b['mode'])][str((prior,post))]+=1
            queries=b['prior_pcie_before_prime'].get('queries',[])
            ends=[q['qpc_end'] for q in queries if q.get('qpc_valid')]
            if ends:query_ages.append((b['copies'][0]['begin_tick']-max(ends))*1e9/native['clock']['frequency_hz'])
        for g in summary['groups']:
            if g['requested_idle_ms']==0:
                combined[(summary['memory_mode'],g['bytes'],g['mode'])].append(dict(job_id=execution['job_id'],mean_ns=g['suffix15_mean_ns']))
        for comparison in summary['observer_comparisons']:
            name=f"{comparison['observed']}_vs_{comparison['reference']}";gates[name][0]+=int(comparison['passed']);gates[name][1]+=1
        for b in summary['baseline_early_late']:baseline[0]+=int(b['passed']);baseline[1]+=1
        data['processes'].append(dict(job_id=execution['job_id'],status='completed',native=execution['native'],
                                       summary=summary,stage_groups=stages,
                                       boundary_link_distributions=[dict(bytes=k[0],requested_idle_ms=k[1],mode=k[2],before_after_counts=dict(v)) for k,v in sorted(link_groups.items())],
                                       prior_query_age_at_first_copy_ns=describe(query_ages),
                                       before_and_after_equal_proves_within_copy_link_constancy=False))
    cross=[]
    for (memory,size,mode),rows in sorted(combined.items()):
        values=[x['mean_ns'] for x in rows];ratio=(max(values)-min(values))/statistics.mean(values) if values else None
        cross.append(dict(memory_mode=memory,bytes=size,mode=mode,contexts=rows,
                          fresh_process_relative_range=ratio,passed=len(rows)==2 and ratio<=.10))
    data.update(total_completed_copies=total_copies,
                numeric_and_host_closure_processes=dict(passed=sum(p['status']=='completed' and p['summary']['validation']['passed'] for p in data['processes']),declared=4),
                observer_gates={k:dict(passed=v[0],declared=v[1],all_passed=v[0]==v[1]) for k,v in gates.items()},
                early_late_zero_idle_suffix_gate=dict(passed=baseline[0],declared=baseline[1]),
                cross_context_zero_idle_suffix_gate=dict(passed=sum(x['passed'] for x in cross),declared=len(cross),cells=cross),
                overall_profile_support_passed=bool(gates) and all(a==b for a,b in gates.values()) and baseline[0]==baseline[1] and bool(cross) and all(x['passed'] for x in cross),
                prices_exported=False,predictions_frozen=False,independent_accuracy_evaluated=False,
                interpretation='Host-stage observations remain local inclusive responses. Event observations need their own gate; passing stamped-vs-GGML does not qualify event spans. No cost exported or mean compensation applied.')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();result=summarize(a.out)
    target=Path(a.out)/'profile-diagnostic-digest.json';campaign.write(target,result)
    print(json.dumps(dict(output=campaign.receipt(target),completed=result['completed_processes'],copies=result['total_completed_copies'],
                          observer_gates=result['observer_gates'],early_late=result['early_late_zero_idle_suffix_gate'],
                          cross_context={k:v for k,v in result['cross_context_zero_idle_suffix_gate'].items() if k!='cells'},profile_support=result['overall_profile_support_passed'],fitted=False)))
