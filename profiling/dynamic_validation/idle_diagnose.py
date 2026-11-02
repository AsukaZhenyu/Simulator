"""Post-evaluation residual diagnosis, with the original forecasts unchanged.

Descriptive group means and variance decompositions are not new predictors.
They cannot identify a physical power/queue/operating-system mechanism.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from profiling.dynamic_validation import idle_campaign as campaign


def mean(values):return statistics.mean(values)


def decomposition(samples):
    """Equal sample hierarchy: process -> idle -> ordinal -> trial residual."""
    if not samples:return None
    groups=[defaultdict(list) for _ in range(3)]
    for job,idle,ordinal,value in samples:
        for depth,key in enumerate((job,(job,idle),(job,idle,ordinal))):groups[depth][key].append(value)
    means=[{key:mean(values) for key,values in group.items()} for group in groups]
    grand=mean(v for *_,v in samples);terms=[grand*grand,0.,0.,0.,0.]
    for job,idle,ordinal,value in samples:
        p,c,o=means[0][job],means[1][job,idle],means[2][job,idle,ordinal]
        for index,square in enumerate(((p-grand)**2,(c-p)**2,(o-c)**2,(value-o)**2),1):terms[index]+=square/len(samples)
    mse=mean(v*v for *_,v in samples)
    names=('global_signed_bias','between_process_mean','idle_condition_within_process','ordinal_shape_within_condition','remaining_trial_copy_variation')
    return dict(samples=len(samples),residual_definition='observed minus original frozen stateful prediction',
        signed_bias_ns=grand,MSE_ns_squared=mse,
        terms={name:dict(ns_squared=value,MSE_fraction=value/mse if mse else 0.) for name,value in zip(names,terms)},
        decomposition_sum_ns_squared=sum(terms),causal_variance_explained_claimed=False,
        remaining_variation_is_unmodelable_claimed=False)


def diagnose(out):
    out=Path(out).resolve();read,sha,require=campaign.read,campaign.sha,campaign.require
    manifest=read(out/'manifest.json');campaign.check(manifest);barrier=read(out/'frozen-barrier.json')
    require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen experiment changed')
    for name,h in barrier['files'].items():require(sha(out/name)==h,'original frozen artifact changed')
    predictions=read(out/'predictions-frozen.json')
    point={(r['memory_mode'],r['requested_idle_ms'],r['ordinal']):float(Fraction(str(r['predicted_ns']))) for r in predictions['rows']}
    original=read(out/'summary.json');cases=original['process_conditions']
    process_records=[];samples=defaultdict(list);burst_samples=defaultdict(list);raw_receipts=[]
    for split in ('profile','heldout'):
        executions=read(out/(split+'-executions.json'))
        require(len(executions)==6,'retain every original declared process')
        for execution in executions:
            job=execution['job'];record=dict(job_id=job['id'],split=split,memory_mode=job['memory_mode'],status=execution['status'])
            if execution['status']!='completed':record['error']=execution.get('error');process_records.append(record);continue
            require(sha(execution['native']['path'])==execution['native']['sha256'],'retained native observation changed')
            native=read(execution['native']['path']);raw_receipts.append(execution['native']);sequences=native['sequences']
            cells=[]
            for idle in job['request']['idle_ms']:
                rows=[r for r in sequences if r['requested_idle_ms']==idle]
                first=mean(r['copies'][0]['wall_ns'] for r in rows)
                later=mean(c['wall_ns'] for r in rows for c in r['copies'][1:])
                cell=dict(requested_idle_ms=idle,first_copy_mean_ns=first,later_copy_mean_ns=later,
                    observed_first_minus_later_ns=first-later,burst_mean_ns=mean(mean(c['wall_ns'] for c in r['copies']) for r in rows),
                    first_half_trial_later_ns=mean(c['wall_ns'] for r in rows if r['trial']<8 for c in r['copies'][1:]),
                    last_half_trial_later_ns=mean(c['wall_ns'] for r in rows if r['trial']>=8 for c in r['copies'][1:]),
                    requested_idle_ns=idle*1000000,actual_idle_mean_ns=mean(r['actual_idle_ns'] for r in rows))
                if split=='heldout':
                    pred=[point[job['memory_mode'],idle,i] for i in range(16)]
                    pfirst,plater=pred[0],mean(pred[1:]);pburst=mean(pred)
                    cell.update(frozen_first_ns=pfirst,frozen_later_mean_ns=plater,
                        first_signed_relative_bias=(pfirst-first)/first,later_signed_relative_bias=(plater-later)/later,
                        burst_signed_relative_bias=(pburst-cell['burst_mean_ns'])/cell['burst_mean_ns'])
                    for r in rows:
                        burst_samples[job['memory_mode']].append((job['id'],idle,0,mean(c['wall_ns'] for c in r['copies'])-pburst))
                        for i,c in enumerate(r['copies']):samples[job['memory_mode']].append((job['id'],idle,i,c['wall_ns']-pred[i]))
                cells.append(cell)
            by_trial=[]
            for trial in range(16):
                rs=[r for r in sequences if r['trial']==trial]
                by_trial.append(dict(trial=trial,first_copy_ns=mean(r['copies'][0]['wall_ns'] for r in rs),
                    later_copy_ns=mean(c['wall_ns'] for r in rs for c in r['copies'][1:]),
                    burst_ns=mean(c['wall_ns'] for r in rs for c in r['copies'])))
            first8=mean(t['later_copy_ns'] for t in by_trial[:8]);last8=mean(t['later_copy_ns'] for t in by_trial[8:])
            record.update(conditions=cells,trial_series=by_trial,first_copy_mean_ns=mean(c['first_copy_mean_ns'] for c in cells),
                later_copy_mean_ns=mean(c['later_copy_mean_ns'] for c in cells),first_half_later_mean_ns=first8,
                last_half_later_mean_ns=last8,late_minus_early_relative=(last8-first8)/first8,
                minimum_trial_later_ns=min(t['later_copy_ns'] for t in by_trial),maximum_trial_later_ns=max(t['later_copy_ns'] for t in by_trial))
            process_records.append(record)
    failures=[]
    for row in cases:
        if row['status']!='completed':continue
        mean_fail=row['burst_mean_ape']>.10;tail_fail=row['burst_ape_p90']>.20
        failures.append(dict(job_id=row['job']['id'],memory_mode=row['job']['memory_mode'],requested_idle_ms=row['requested_idle_ms'],
            burst_mean_fail=mean_fail,burst_tail_fail=tail_fail,passed=not mean_fail and not tail_fail,
            burst_mean_ape=row['burst_mean_ape'],burst_p90_ape=row['burst_ape_p90'],
            first_mean_fail=row['first_copy_mean_ape']>.10,raw_burst_cv=row['raw_burst_cv']))
    counts=dict(process_conditions=24,completed=len(failures),mean_failed=sum(r['burst_mean_fail'] for r in failures),
        tail_failed=sum(r['burst_tail_fail'] for r in failures),both_failed=sum(r['burst_mean_fail'] and r['burst_tail_fail'] for r in failures),
        qualified=sum(r['passed'] for r in failures),first_failed=sum(r['first_mean_fail'] for r in failures))
    result=dict(schema='dynamic-idle-post-evaluation-diagnosis-1',frozen_barrier=campaign.receipt(out/'frozen-barrier.json'),raw=raw_receipts,
        models_refitted=False,predictions_recomputed=False,processes_omitted=False,diagnosis_is_primary_accuracy_evidence=False,
        interpretation='post-evaluation residual summaries distinguish first response, later response, process offsets and trial drift; they do not identify their physical cause',
        original_case_results=dict(burst=original['burst_mean_qualified_cases'],first=original['first_response_accuracy_cases'],declared=8),
        process_condition_gate_counts=counts,process_condition_gates=failures,processes=process_records,
        residual_decompositions={memory:dict(all_copies=decomposition(rows),later_copies=decomposition([r for r in rows if r[2]>0]),
            burst_means=decomposition(burst_samples[memory])) for memory,rows in samples.items()})
    campaign.write(out/'diagnosis.json',result)
    return dict(diagnosis=str(out/'diagnosis.json'),gate_counts=counts,
        process_later_summary=[{k:r[k] for k in ('job_id','first_copy_mean_ns','later_copy_mean_ns','late_minus_early_relative')} for r in process_records if r['status']=='completed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(diagnose(a.out),ensure_ascii=False),flush=True)
