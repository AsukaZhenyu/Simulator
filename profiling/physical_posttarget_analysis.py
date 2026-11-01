"""Post-target cleanup protocol adapter; frozen prediction code remains unchanged.

Only actors AFTER target completion differ. Original complete target runs are
retained; one failed attempt can be replaced solely for a retained readiness
failure. New AFTER probes run unloaded and cannot audit same-load drift. BEFORE
references, target executable, policy, frozen parameters and prediction function
are unchanged. Hardware accuracy qualification remains false.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import statistics

from profiling.physical_stage_analysis import (
    SCHEMA, _sha, _read, _receipt, _pack, _effective_jobs, _load_job,
    _signature, _references, _predict, _stats, _error_row, _components,
)
from profiling.available_service.calibration import require, utc, digest


def _posttarget_jobs(campaign, manifest, frozen):
    path=campaign/'posttarget-job-index.json'
    index=_read(path)
    require(index.get('schema')=='physical-posttarget-job-index-1','unsupported posttarget job index')
    require(index['parent_manifest_sha256']==_sha(campaign/'manifest.json'),'posttarget parent manifest differs')
    binding=index['frozen_file']
    saved=_read(binding['path'],binding['sha256'])
    require(digest(saved)==digest(frozen),'posttarget index binds a different frozen profile')
    utc(index['declared_at_utc'])
    require(utc(index['declared_at_utc'])>utc(frozen['frozen_at_utc']),'operational declaration precedes model freeze')
    require(index['posttarget_after_reference_scope']=='unloaded_drift_only','posttarget AFTER scope must be unloaded')
    receipts=[]
    for source in index['source_receipts']:
        require(_sha(source['path'])==source['sha256'],'posttarget source changed: '+source['path'])
        receipts.append(dict(source))
    require(any(Path(r['path']).resolve()==Path(__file__).resolve() for r in receipts),'posttarget analyzer source was not declared before collection')
    base,_,profile_replacements=_effective_jobs(campaign,manifest)
    require(profile_replacements==frozen.get('job_replacements',[]),'frozen profile replacement inventory changed')
    expected={job['id']:job for job in base}
    retained_failures=[]
    replaced=set()
    for replacement in index['replacements']:
        old_id=replacement['original_id']
        require(old_id in expected and old_id not in replaced,'duplicate or missing replaced heldout job')
        old=expected[old_id]
        require(old['split'] in {'heldout','heldout_bare'},'posttarget adapter may not replace training jobs')
        new=replacement['replacement_job']
        require(new['id'] not in expected,'retry identity is not fresh')
        require({k:v for k,v in old.items() if k!='id'}=={k:v for k,v in new.items() if k!='id'},'retry changes BEFORE or target protocol')
        failure=_read(replacement['failed_execution_path'],replacement['failed_execution_sha256'])
        require(failure.get('status')=='failed' and failure.get('job')==old and
                'readiness deadline' in failure.get('error',''),'retry requires a retained infrastructure readiness failure')
        require(utc(failure['ended_at_utc'])<=utc(index['declared_at_utc']),'retry was declared before original failure ended')
        expected.pop(old_id)
        expected[new['id']]=new
        replaced.add(old_id)
        retained_failures.append(dict(original_id=old_id,replacement_id=new['id'],
            failed_execution_path=replacement['failed_execution_path'],failed_execution_sha256=replacement['failed_execution_sha256'],
            whole_original_attempt_excluded=True,selection_reason='infrastructure readiness failure, no latency selection'))
    actual=index['jobs']
    require(len({job['id'] for job in actual})==len(actual),'duplicate operational job identity')
    require({job['id']:job for job in actual}==expected,'operational inventory altered original target treatments')
    old_complete=set(index['original_completed_job_ids'])
    new_cleanup=set(index['posttarget_cleanup_job_ids'])
    selected={j['id'] for j in actual if j['split'] in {'heldout','heldout_bare'}}
    require(not old_complete & new_cleanup and old_complete|new_cleanup==selected,'posttarget lifecycle inventory is not a disjoint cover')
    require(len(old_complete)==len(index['original_completed_job_ids']) and len(new_cleanup)==len(index['posttarget_cleanup_job_ids']), 'duplicate lifecycle IDs')
    for job_id in old_complete:
        row=_read(campaign/'jobs'/job_id/'execution.json')
        require(row.get('status')=='completed' and row.get('job')==expected[job_id],'retained original heldout job is incomplete')
        require(utc(row['ended_at_utc'])<=utc(index['declared_at_utc']),'original completed job ended after new protocol declaration')
    return actual,dict(index=_receipt(path),source_receipts=receipts,
        declared_at_utc=index['declared_at_utc'],frozen_file=binding,
        original_completed_job_ids=sorted(old_complete),posttarget_cleanup_job_ids=sorted(new_cleanup),
        retained_failed_attempts=retained_failures,after_references_used_for_prediction=False,
        posttarget_after_reference_scope='unloaded_drift_only',before_and_target_treatment_changed=False,
        fixed_parameters_or_predictor_changed=False,hardware_accuracy_qualified=False)


def evaluate_campaign_heldout(campaign_path, frozen, *, split='heldout'):
    """Use frozen candidates unchanged; retain every main invocation and high CV."""
    require(frozen.get('schema')==SCHEMA,'unsupported frozen diagnostic profile')
    campaign=Path(campaign_path).resolve()
    require(str(campaign)==frozen['campaign'],'campaign identity differs')
    for source in frozen['source_receipts']:
        require(_sha(source['path'])==source['sha256'],'frozen source changed: '+source['path'])
    manifest=_read(campaign/'manifest.json',frozen['manifest_sha256'])
    jobs,posttarget_protocol=_posttarget_jobs(campaign,manifest,frozen)
    require(split not in {'profile','observer','smoke'},'heldout split must be fresh')
    identities={(r['process_id'],r['seed']) for r in frozen['profile_identities']}
    profile_pids={r[0] for r in identities}
    profile_seeds={r[1] for r in identities}
    rows, metrics, coverage, physical = [],defaultdict(list),defaultdict(list),[]
    for job in jobs:
        if job['split']!=split:
            continue
        loaded=_load_job(campaign,manifest,job,physical=job['instrumentation']=='cupti')
        require(utc(loaded['execution']['started_at_utc'])>utc(frozen['frozen_at_utc']),'heldout began before freeze')
        require(loaded['native']['process_id'] not in profile_pids and loaded['execution']['program']['seed'] not in profile_seeds,
                'heldout PID or seed overlaps profile')
        require(_signature(loaded['program'])==frozen['signature_sha256'],'heldout policy/graph signature differs')
        changed_after=job['id'] in posttarget_protocol['posttarget_cleanup_job_ids']
        if changed_after:
            require(utc(loaded['execution']['started_at_utc'])>utc(posttarget_protocol['declared_at_utc']), 'posttarget job began before operational declaration')
            require(loaded['execution'].get('posttarget_cleanup_applied') is True and
                    loaded['execution'].get('posttarget_after_reference_scope')=='unloaded_drift_only',
                    'posttarget execution did not retain its changed AFTER protocol')
            cleanup_source=loaded['execution']['posttarget_cleanup_receipt']
            cleanup_audit=_read(cleanup_source['path'],cleanup_source['sha256'])
        else:
            cleanup_source=None
            cleanup_audit=None
        before=_references(loaded,before=True)
        predictions=_predict(frozen,job['condition'],before)
        # Main call durations enter comparison only, never the reference response.
        measured=[r['end_ns']-r['begin_ns'] for r in loaded['native']['iterations']]
        observed=_stats(measured)
        pid=loaded['native']['process_id']
        coverage[job['condition']].append(pid)
        after=_references(loaded,before=False)
        after['same_load_drift_valid']=not changed_after
        after['scope']='unloaded_drift_only' if changed_after else 'original_same_load_drift_audit'
        row=dict(job_id=job['id'],condition=job['condition'],process_id=pid,seed=loaded['execution']['program']['seed'],
            instrumentation=job['instrumentation'],observed_ns=measured,observed=observed,
            before=before,after_audit_only=after,predictions={},
            posttarget_background_cleanup_changed=changed_after,
            posttarget_cleanup_receipt=cleanup_source,posttarget_cleanup_audit=cleanup_audit,
            capture_qualified=False if loaded['parsed'] is None else loaded['parsed']['capture_qualified'],
            numerical_verified=True,reference_model_time_dependence='Batch-constant stale BEFORE response; no instantaneous trajectory inferred')
        for name,prediction in predictions.items():
            if not prediction['known']:
                row['predictions'][name]=prediction
                continue
            pred=prediction['predicted_ns']
            errors=[_error_row(pred,n) for n in measured]
            per_call=_stats([r['relative_error'] for r in errors])
            summary=dict(prediction,per_call_relative_error=per_call,
                signed_relative_bias=statistics.mean(r['signed_relative_error'] for r in errors),
                process_mean_truth_error=_error_row(pred,observed['mean']),
                process_median_truth_error=_error_row(pred,observed['median']))
            row['predictions'][name]=summary
            metrics[(name,job['condition'])].append(dict(process_id=pid,predicted_ns=pred,
                measured=measured,errors=errors,observed=observed,summary=summary))
        if loaded['parsed']:
            comp=_components(loaded)
            totals=defaultdict(Fraction)
            for key,value in comp['means'].items():
                totals[comp['template'][key]['category']]+=value
            physical.append(dict(job_id=job['id'],condition=job['condition'],mean_components_ns=totals,
                capture_issues=loaded['parsed']['capture_issues'],selected_action_issues=loaded['parsed']['selected_action_issues'],
                host_API_device_columns_additive=False,host_API_is_active_CPU_work=False))
        rows.append(row)
    require(rows,'no declared heldout observations')
    summaries=[]
    for (name,state),values in metrics.items():
        errors=[e for r in values for e in r['errors']]
        error_stats=_stats([e['relative_error'] for e in errors])
        process_count=len({r['process_id'] for r in values})
        gates=dict(minimum_processes=process_count>=3,
            minimum_calls=all(len(r['measured'])>=32 for r in values),
            median_error=error_stats['median']<=.20,p90_error=error_stats['p90']<=.20,
            empirical_stability=all(r['observed']['cv'] is not None and r['observed']['cv']<=.10 for r in values),
            observer_neutrality=frozen['observer']['neutrality_qualified'],
            numerical=True,capture=all(r['capture_qualified'] for r in rows if r['condition']==state),
            independent_instantaneous_resource_trajectory=False,
            physical_component_service_identified=False)
        summaries.append(dict(candidate=name,condition=state,processes=process_count,
            per_call_relative_error=error_stats,signed_relative_bias=statistics.mean(e['signed_relative_error'] for e in errors),
            process_mean_truth_relative_error=_stats([r['summary']['process_mean_truth_error']['relative_error'] for r in values]),
            process_median_truth_relative_error=_stats([r['summary']['process_median_truth_error']['relative_error'] for r in values]),
            qualification_gates=gates,hardware_accuracy_qualified=False))
    return _pack(dict(schema='physical-component-diagnostic-evaluation-1',split=split,
        frozen_profile_digest=digest(frozen),candidate_selection_after_heldout=False,
        coverage={state:dict(processes=len(set(pids)),minimum_processes_pass=len(set(pids))>=3) for state,pids in coverage.items()},
        summaries=summaries,runs=rows,physical_breakdown=physical,observer=frozen['observer'],
        posttarget_operational_protocol=posttarget_protocol,
        frozen_predictor_reused_without_refit=True, mixed_posttarget_protocols=len(set(r['posttarget_background_cleanup_changed'] for r in rows))>1,
        hardware_accuracy_qualified=False,scope='Diagnostic response comparison; stale reference probes, CUPTI observer overhead, no availability trajectory certification',
        crossing_metrics=dict(known=False,reason='No independently observed within-call resource transition trajectory; cohort labels are not transition times')))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--split',choices=('heldout','heldout_bare'),default='heldout')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=evaluate_campaign_heldout(args.campaign,_read(args.frozen),split=args.split)
    with args.output.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(path=str(args.output.resolve()),sha256=_sha(args.output))))


if __name__=='__main__':
    main()
