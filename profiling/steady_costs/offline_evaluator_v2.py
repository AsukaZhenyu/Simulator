"""Offline v2 correction: Windows numeric PID recycling is not process reuse.

The prepared collector, frozen price table and historical evaluator remain
unchanged. This entry point retains the v1 metric computation and thresholds,
replacing only numeric-PID uniqueness with verified launch-epoch identities.
It never launches native hardware, refits prices or drops measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from fractions import Fraction
import json
from pathlib import Path

from llm_infer_model.performance.steady_action_costs import SteadyActionKey,compile_steady_actions
from .calibration import action_key,errors,load_job,restore_table
from .campaign import mean,pack,read,receipt,stats,utc,verify_manifest,write_new
from .contracts import digest,require,semantic_program,sha


def _epoch(value):
    require(isinstance(value,str),'owned process launch/completion epoch missing')
    epoch=datetime.fromisoformat(value)
    require(epoch.tzinfo is not None,'owned process epoch must have timezone')
    return epoch


def process_identity(execution):
    """Receipt launch epoch, not an unmeasured Windows creation-time sample."""
    pid=execution.get('process_id');start=_epoch(execution.get('started_at_utc'));end=_epoch(execution.get('completed_at_utc'))
    require(type(pid) is int and pid>0 and end>=start,'owned process identity/interval differs')
    require(execution.get('status')=='completed' and execution.get('returncode')==0 and
        execution.get('forced_termination') is False and execution.get('process_exited_verified') is True,
        'owned process normal exit is not verified')
    return dict(pid=pid,launch_epoch_utc=start.astimezone(timezone.utc).isoformat())


def validate_fresh_process(execution,previous):
    identity=process_identity(execution);recycled=[]
    start=_epoch(execution['started_at_utc'])
    for earlier in previous:
        prior_identity=process_identity(earlier)
        if earlier['process_id']!=execution['process_id']:continue
        require(prior_identity!=identity,'same process launch identity was reused')
        require(_epoch(earlier['completed_at_utc'])<=start,'same-PID owned process intervals overlap')
        recycled.append(dict(identity=prior_identity,completed_at_utc=earlier['completed_at_utc'],
            verified_normal_exit_before_new_launch=True))
    return dict(identity=identity,numeric_pid_recycled=bool(recycled),prior_same_pid_executions=recycled,
        OS_creation_time_sample_claimed=False)


def correction_provenance(root):
    paths=[Path(__file__),Path(__file__).parent/'tests/test_offline_evaluator_v2.py']
    saved=read(root/'offline-evaluator-v2-sources.json')
    require(saved.get('schema')=='steady-cost-offline-evaluator-sources-2','offline correction source schema differs')
    verify_correction_sources(saved['corrected_sources'],paths)
    for field in ('original_manifest','frozen_prices'):
        source=saved[field];require(sha(source['path'])==source['sha256'],'offline correction input changed: '+field)
    for field,name in (('frozen_source_capsule','frozen-source-capsule-v1.json'),('original_rejected_attempt','evaluation-attempt-v1.json')):
        require(saved[field]==optional_receipt(root/name),'optional prior-attempt evidence changed: '+field)
    return dict(schema='steady-cost-offline-evaluation-correction-2',
        reason='numeric Windows PID recycling was incorrectly classified as process reuse',
        semantic_change='only process identity uses PID plus recorded launch epoch; recycled PID requires verified nonoverlap and normal exit',
        corrected_sources=[receipt(path) for path in paths],
        original_manifest=receipt(root/'manifest.json'),frozen_prices=receipt(root/'profile-frozen.json'),
        frozen_source_capsule=saved['frozen_source_capsule'],
        original_rejected_attempt=saved['original_rejected_attempt'],
        no_new_measurements=True,no_parameter_refit=True,no_threshold_change=True,
        OS_creation_time_sample_claimed=False,correction_source_binding=receipt(root/'offline-evaluator-v2-sources.json'))


def verify_correction_sources(sources,required_paths):
    require({str(Path(row['path']).resolve()) for row in sources}=={str(Path(path).resolve()) for path in required_paths},
        'offline correction source inventory differs')
    for source in sources:require(sha(source['path'])==source['sha256'],'offline correction source changed')


def optional_receipt(path):
    return dict(status='present',receipt=receipt(path)) if Path(path).is_file() else dict(
        status='not_applicable',reason='no prior v1 rejected attempt or frozen-source capsule exists at this optional path',path=str(Path(path).resolve()))


def bind_correction_sources(root):
    root=Path(root).resolve();verify_manifest(root)
    document=dict(schema='steady-cost-offline-evaluator-sources-2',bound_at_utc=utc(),
        corrected_sources=[receipt(path) for path in (Path(__file__),Path(__file__).parent/'tests/test_offline_evaluator_v2.py')],
        original_manifest=receipt(root/'manifest.json'),frozen_prices=receipt(root/'profile-frozen.json'),
        frozen_source_capsule=optional_receipt(root/'frozen-source-capsule-v1.json'),
        original_rejected_attempt=optional_receipt(root/'evaluation-attempt-v1.json'))
    write_new(root/'offline-evaluator-v2-sources.json',document)
    return dict(path=str(root/'offline-evaluator-v2-sources.json'),native_hardware_executed=False)


def evaluate_v2(root):
    root=Path(root).resolve();manifest=verify_manifest(root)
    frozen=read(root/'profile-frozen.json');barrier=read(root/'frozen-barrier.json')
    require(frozen['artifact_sha256']==digest({k:v for k,v in frozen.items() if k!='artifact_sha256'}),'frozen semantic hash differs')
    require(sha(root/'profile-frozen.json')==barrier['frozen_file_sha256'] and sha(root/'manifest.json')==barrier['manifest_sha256'],'freeze bytes changed')
    require(read(root/'heldout-started.json')['barrier']==barrier,'heldout was not launched with this barrier')
    for source in frozen['source_receipts']:require(sha(source['path'])==source['sha256'],'frozen profile evidence changed')
    table=restore_table(frozen['table']);results=[];profiles={r['seed'] for r in frozen['profile_processes']};profile_pids={r['process_id'] for r in frozen['profile_processes']}
    quality=frozen['quality']
    provenance=correction_provenance(root)
    prior_processes=[];process_checks=[]
    for saved in frozen['profile_processes']:
        job=next(j for j in manifest['jobs'] if j['id']==saved['job_id'])
        execution,_=load_job(root,manifest,job)
        require(execution['process_id']==saved['process_id'] and execution['program']['seed']==saved['seed'],
            'frozen PROFILE process metadata differs')
        process_identity(execution)
        prior_processes.append(execution)
    for case in manifest['cases']:
        keys=tuple(SteadyActionKey(**k) for k in frozen['case_keys'][case['id']])
        model=compile_steady_actions(keys,table);prediction=model.evaluate().duration_ns
        own=[];all_observed=[]
        for job in [j for j in manifest['jobs'] if j['split']=='heldout' and j['case_id']==case['id']]:
            execution,native=load_job(root,manifest,job)
            binding=manifest['programs'][job['program_key']];program=read(binding['program']['path'])
            require(digest(semantic_program(program))==binding['signature'],'heldout shape/strategy signature differs')
            # Full-stamped PROFILE prices apply to bare full replay only under
            # the separately reported observer gate. All other boundaries,
            # including repetitions and prepared-input shape, must be exact.
            application_request=dict(job['request'],mode='full-stamped' if case['kind']=='layer' else 'action')
            application_keys=tuple(action_key(manifest,binding,program,application_request,index if case['kind']=='layer' else None)
                for index in range(len(program['actions']) if case['kind']=='layer' else 1))
            require(application_keys==keys,'heldout exact cost scope differs from frozen PROFILE keys')
            require(execution['program']['seed'] not in profiles,'heldout numerical seed overlaps PROFILE')
            process_check=validate_fresh_process(execution,prior_processes)
            process_checks.append(dict(job_id=job['id'],**process_check));prior_processes.append(execution)
            require(execution['started_at_utc']>=barrier['frozen_at_utc'],'heldout launch predates frozen prices')
            observed=[Fraction(t['wall_ns'],t['repeats']) for t in native['trials']]
            all_observed.extend(observed);average=mean(observed)
            err=float(abs(Fraction(prediction)-average)/average)
            summary=stats(observed)
            own.append(dict(job_id=job['id'],process_id=execution['process_id'],process_identity=process_check['identity'],predicted_per_call_mean_ns=prediction,
                observed_trial_average_ns=pack(observed),observed_statistics=summary,aggregate_mean_ape=err,trial_average_errors=errors(prediction,observed),
                mean_gate=err<=quality['max_process_mean_ape'],trial_cv_gate=summary['cv'] is not None and summary['cv']<=quality['max_process_trial_cv']))
        require(len(own)==3,'heldout process coverage incomplete')
        require(len({(p['process_identity']['pid'],p['process_identity']['launch_epoch_utc']) for p in own})==3,'heldout process launch identities must be distinct')
        summary=errors(prediction,all_observed)
        observer=[]
        if case['kind']=='layer':
            for repeat in range(3):
                ps=next(j for j in manifest['jobs'] if j['case_id']==case['id'] and j['split']=='profile' and j['repeat']==repeat)
                ob=next(j for j in manifest['jobs'] if j['case_id']==case['id'] and j['split']=='observer' and j['repeat']==repeat)
                pe,pn=load_job(root,manifest,ps);oe,on=load_job(root,manifest,ob)
                pm=mean(Fraction(t['wall_ns'],t['repeats']) for t in pn['trials']);om=mean(Fraction(t['wall_ns'],t['repeats']) for t in on['trials'])
                effect=float(pm/om-1)
                observer.append(dict(repeat=repeat,full_stamped_mean_ns=pack(pm),full_bare_mean_ns=pack(om),relative_difference=effect,
                    passed=abs(effect)<=quality['max_observer_relative_mean_difference']))
        passed=all(p['mean_gate'] for p in own) and summary['p90_ape']<=quality['max_trial_average_p90_ape']
        observer_gate=all(o['passed'] for o in observer) if observer else None
        results.append(dict(case_id=case['id'],kind=case['kind'],prediction_ns=prediction,processes=own,trial_average_errors=summary,
            observed_statistics=stats(all_observed),mean_and_trial_p90_gate=passed,observer_controls=observer,
            observer_gate=observer_gate,stability_gate=all(p['trial_cv_gate'] for p in own),
            hardware_accuracy_qualified=bool(passed and all(p['trial_cv_gate'] for p in own) and observer_gate is not False)))
    diagnostics=[]
    for job in [j for j in manifest['jobs'] if j['split']=='diagnostic']:
        execution,native=load_job(root,manifest,job)
        diagnostics.append(dict(job_id=job['id'],role=job['role'],request=job['request'],native=execution['native'],
            wall_per_call_ns=stats(Fraction(t['wall_ns'],t['repeats']) for t in native['trials']),
            intervals={field:stats(Fraction(t[field],t['repeats']) for t in native['trials'] if t.get(field) is not None)
                for field in ('enqueue_ns','sync_ns','stream_elapsed_ns')},diagnostic_intervals_added_to_price=False))
    result=dict(schema='steady-cost-validation-2',evaluated_at_utc=utc(),frozen_file_sha256=barrier['frozen_file_sha256'],
        cases=results,diagnostics=diagnostics,heldout_prices_fitted=False,target_state_used_as_prediction_input=False,
        external_resource_trajectory_estimated=False,all_trials_retained=True,individual_call_tail_accuracy_claimed=False,
        quality=quality,passed_cases=sum(c['hardware_accuracy_qualified'] for c in results),total_cases=len(results),
        all_cases_passed=all(c['hardware_accuracy_qualified'] for c in results),capacity_qualified=False,
        evaluator_correction=provenance,process_identity_checks=process_checks)
    write_new(root/'evaluation-v2.json',result)
    # Recheck the independent binding before publishing the correction receipt.
    require(correction_provenance(root)==provenance,'offline evaluator changed during evaluation')
    write_new(root/'evaluation-correction-v2.json',dict(**provenance,evaluation=receipt(root/'evaluation-v2.json'),
        completed_at_utc=utc(),prices_modified=False,new_hardware_executed=False,raw_trials_excluded=False,
        process_identity_checks=process_checks))
    return dict(path=str(root/'evaluation-v2.json'),passed_cases=result['passed_cases'],total_cases=result['total_cases'],
        cases=[dict(case_id=c['case_id'],mean_errors=[p['aggregate_mean_ape'] for p in c['processes']],p90=c['trial_average_errors']['p90_ape'],
                    passed=c['hardware_accuracy_qualified']) for c in results])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bind-sources',action='store_true',help='write source binding only; do not evaluate')
    args=parser.parse_args(argv)
    print(json.dumps(bind_correction_sources(args.out) if args.bind_sources else evaluate_v2(args.out),indent=2,allow_nan=False))


if __name__=='__main__':main()
