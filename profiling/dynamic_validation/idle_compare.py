"""Score every provider frozen before acquisition, preserving all repetitions.

This is scoring only: the observed idle, copy and completion times never enter
any new parameter estimation or prediction. The original stateful primary
provider and acceptance gates remain unchanged.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

from profiling.dynamic_validation import idle_campaign as campaign


def scores(predicted,bursts):
    """Mean latency, whole-burst tail and first response have different gates."""
    predicted_mean=statistics.mean(predicted)
    observed_bursts=[statistics.mean(row) for row in bursts]
    observed_mean=statistics.mean(observed_bursts)
    errors=sorted(abs(predicted_mean-v)/v for v in observed_bursts)
    # Identical to NumPy quantile(method='higher'), with no optional dependency.
    p90=errors[math.ceil(.9*(len(errors)-1))]
    first_mean=statistics.mean(row[0] for row in bursts)
    mean_ape=abs(predicted_mean-observed_mean)/observed_mean
    first_ape=abs(predicted[0]-first_mean)/first_mean
    return dict(observed_burst_mean_ns=observed_mean,predicted_burst_mean_ns=predicted_mean,
        burst_mean_ape=mean_ape,burst_ape_p90=p90,first_copy_mean_ape=first_ape,
        raw_burst_cv=statistics.stdev(observed_bursts)/observed_mean,
        mean_response_qualified=mean_ape<=.10 and p90<=.20,first_response_accuracy_passed=first_ape<=.10)


def compare(out):
    out=Path(out).resolve();read,sha,require=campaign.read,campaign.sha,campaign.require
    m=read(out/'manifest.json');campaign.check(m);barrier=read(out/'frozen-barrier.json')
    require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen idle manifest changed')
    for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen idle file changed')
    frozen=read(out/'predictions-frozen.json');executions=read(out/'heldout-executions.json')
    expected=[j for j in m['jobs'] if j['split']=='heldout']
    require(len(expected)==len(executions)==6 and {j['id'] for j in expected}=={e['job']['id'] for e in executions},'all six original independent processes required')
    truth={}
    for e in executions:
        if e['status']!='completed':continue
        require(sha(e['native']['path'])==e['native']['sha256'],'independent idle raw changed')
        n=read(e['native']['path']);campaign.verify(n,e['job']['request'],m['programs'][e['job']['program_key']],e['process_id'])
        truth[e['job']['id']]=n
    providers={}
    for name,predictions in frozen['providers'].items():
        points={(r['memory_mode'],r['requested_idle_ms'],r['ordinal']):float(Fraction(str(r['predicted_ns']))) for r in predictions}
        records=[]
        for e in executions:
            job=e['job']
            for idle in job['request']['idle_ms']:
                record=dict(job_id=job['id'],memory_mode=job['memory_mode'],requested_idle_ms=idle,
                    status=e['status'],mean_response_qualified=False,first_response_accuracy_passed=False)
                if e['status']=='completed':
                    predicted=[points[(job['memory_mode'],idle,i)] for i in range(16)]
                    bursts=[[c['wall_ns'] for c in r['copies']] for r in truth[job['id']]['sequences'] if r['requested_idle_ms']==idle]
                    record.update(scores(predicted,bursts))
                else:record['error']=e.get('error')
                records.append(record)
        cases=[]
        for memory in ('pageable','registered'):
            for idle in campaign.IDLES['heldout']:
                rs=[r for r in records if r['memory_mode']==memory and r['requested_idle_ms']==idle]
                cases.append(dict(memory_mode=memory,requested_idle_ms=idle,declared_processes=3,
                    completed_processes=sum(r['status']=='completed' for r in rs),
                    mean_response_qualified=len(rs)==3 and all(r['mean_response_qualified'] for r in rs),
                    first_response_accuracy_passed=len(rs)==3 and all(r['first_response_accuracy_passed'] for r in rs)))
        providers[name]=dict(cases=cases,process_conditions=records,declared_cases=8,
            burst_mean_qualified_cases=sum(c['mean_response_qualified'] for c in cases),
            first_response_accuracy_cases=sum(c['first_response_accuracy_passed'] for c in cases))
    result=dict(schema='dynamic-idle-frozen-provider-comparison-1',primary_provider=frozen['primary_provider'],providers=providers,
        frozen=campaign.receipt(out/'frozen-barrier.json'),planned_idle_only=True,new_parameters_fitted=False,
        failed_processes_remain_in_denominator=True,raw_cv_is_prediction_gate=False,
        general_device_fluctuation_model_claimed=False)
    campaign.write(out/'provider-comparison.json',result)
    return dict(comparison=str(out/'provider-comparison.json'),providers={k:{n:v[n] for n in ('burst_mean_qualified_cases','first_response_accuracy_cases','declared_cases')} for k,v in providers.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(compare(a.out),ensure_ascii=False),flush=True)
