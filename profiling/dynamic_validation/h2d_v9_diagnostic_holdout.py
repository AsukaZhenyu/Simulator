"""Independent diagnostic use of immutable, unsupported v9 candidate fits.

This separate experiment cannot authorize the failed original campaign, export
prices, refit parameters or turn independent mean accuracy into qualification.
It follows the original unused size/idle/seed grid to test what the failure
means. Every formal qualification remains false because baseline support failed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from . import h2d_bare_v9 as original
from . import h2d_bare_v9_fit as fitting

ROOT=original.ROOT
read,write,sha,receipt,require=original.read,original.write,original.sha,original.receipt,original.require


def protocol():
    return dict(schema='h2d-v9-independent-diagnostic-protocol-1',role='diagnostic only; original baseline unsupported',
        declared_fresh_processes=6,original_unused_heldout_requests_reused_exactly=True,
        sizes_bytes=list(original.SIZES['heldout']),idle_ms=list(original.IDLES['heldout']),
        fresh_processes_per_memory=3,trials=12,copies_per_burst=16,
        independent_mathematical_seed=True,predictors=['static_affine','zero_initial_recovery','prepared_initial_recovery'],
        use_identifiable_but_baseline_unsupported_fits=True,parameters_refitted=False,
        predictions_frozen_before_collection=True,original_failed_gate_unchanged=True,
        classification='independent diagnostic accuracy; no formal provider qualification',
        strict_qualification_always_passes=False,unsupported_profile_reason='zero-idle chronological and fresh-process baseline drift',
        prices_exported=False,additional_sampling=False,retries=0,slow_trials_removed=0,
        mean_accuracy_limits=dict(first_mean_APE=.20,actual_burst_wall_mean_APE=.20),
        stability_limits=dict(first_raw_population_CV=.10,actual_burst_wall_raw_population_CV=.10),
        acceptance_includes_outer_loop_gaps=True,forecast_adds_observed_gap_residual=False,
        whole_LLM_accuracy_claimed=False,device_variance_fully_modeled=False)


def prepare(out,parent):
    out,parent=Path(out).resolve(),Path(parent).resolve();require(not out.exists(),'new exclusive diagnostic directory required')
    m=read(parent/'manifest.json');fit=read(parent/'fit.json');barrier=read(parent/'frozen-barrier.json')
    require(not (parent/'heldout-started.json').exists() and not any((parent/'jobs'/j['id']).exists() for j in m['jobs'] if j['split']=='heldout'),'original heldout grid already used')
    require(barrier['manifest_sha256']==sha(parent/'manifest.json') and barrier['profile_gate_passed'] is False,'this diagnostic requires retained failed original support gate')
    for name,value in barrier['files'].items():require(sha(parent/name)==value,'original frozen fit changed')
    require(fit['heldout_observations_read'] is False and fit['whole_LLM_target_used'] is False,'target-blind fit required')
    require(len(fit['models'])==6 and all(x['identifiable'] and not x['parameters_supported'] and not x['profile_baseline_supported'] for x in fit['models']),
            'exact six locally identifiable but baseline-unsupported candidates required')
    jobs=copy.deepcopy([j for j in m['jobs'] if j['split']=='heldout']);require(len(jobs)==6,'original six unused requests required')
    predictions=[]
    for model in fit['models']:
        values=fitting.curve(model['kind'],model['theta'],fit['scope'],model['memory_mode'],tuple(model['evidence']),original.SIZES['heldout'],original.IDLES['heldout'],continuous=False)
        for i,size in enumerate(original.SIZES['heldout']):
            for j,idle in enumerate(original.IDLES['heldout']):
                predictions.append(dict(memory_mode=model['memory_mode'],kind=model['kind'],bytes=size,requested_idle_ms=idle,
                    copy_predictions_ns=values[i,j].tolist(),first_prediction_ns=float(values[i,j,0]),burst_prediction_ns=float(sum(values[i,j])),
                    profile_baseline_supported=False,price_export_permitted=False,heldout_observation_used=False))
    sources=[Path(__file__),Path(original.__file__),Path(fitting.__file__)]
    sources += [Path(x['source']['path']) for x in m['sources'] if 'modeling' in Path(x['source']['path']).parts]
    out.mkdir(parents=True);archived=[]
    for source in sources:
        target=out/'executed_sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        archived.append(dict(source=receipt(source),archived=receipt(target)))
    manifest=dict(schema='h2d-v9-independent-diagnostic-manifest-1',prepared_at_utc=original.base.e2a.utc(),
        original_manifest=receipt(parent/'manifest.json'),original_fit=receipt(parent/'fit.json'),original_failed_barrier=receipt(parent/'frozen-barrier.json'),
        original_baseline_support=receipt(parent/'baseline-support.json'),binary=m['binary'],build_receipt=m['build_receipt'],
        runtime=m['runtime'],runtime_dirs=m['runtime_dirs'],heldout_program=m['programs']['heldout'],jobs=jobs,sources=archived,
        protocol=protocol(),profile_gate_passed=False,qualified=False)
    write(out/'manifest.json',manifest);write(out/'predictions.json',dict(schema='h2d-v9-independent-diagnostic-predictions-1',rows=predictions,refit=False,qualification=False))
    frozen=dict(schema='h2d-v9-independent-diagnostic-freeze-1',manifest_sha256=sha(out/'manifest.json'),
        source_fit_sha256=manifest['original_fit']['sha256'],source_failed_barrier_sha256=manifest['original_failed_barrier']['sha256'],
        prediction_sha256=sha(out/'predictions.json'),no_heldout_observations=True,parameters_refitted=False,
        original_profile_gate_passed=False,formal_accuracy_qualification=False)
    write(out/'frozen-diagnostic.json',frozen)
    return dict(manifest=receipt(out/'manifest.json'),freeze=receipt(out/'frozen-diagnostic.json'),processes=6,prediction_cells=72,qualification=False,hardware_executed=False)


def check(out):
    m=read(out/'manifest.json');f=read(out/'frozen-diagnostic.json')
    require(f['manifest_sha256']==sha(out/'manifest.json') and f['prediction_sha256']==sha(out/'predictions.json'),'frozen diagnostic changed')
    require(f['original_profile_gate_passed'] is False and f['formal_accuracy_qualification'] is False and m['profile_gate_passed'] is False,'failed original support cannot become qualification')
    for name in ('original_manifest','original_fit','original_failed_barrier','original_baseline_support','binary','build_receipt'):
        r=m[name];require(sha(r['path'])==r['sha256'],'bound original resource changed: '+name)
    parent=Path(m['original_manifest']['path']).parent
    require(not (parent/'heldout-started.json').exists(),'original heldout acquisition now overlaps this independent diagnostic')
    for r in m['runtime']:require(sha(r['path'])==r['sha256'],'runtime changed')
    for s in m['sources']:
        for name in ('source','archived'):require(sha(s[name]['path'])==s[name]['sha256'],'bound calculation/diagnostic source changed')
    for name in ('program','reference'):
        r=m['heldout_program'][name];require(sha(r['path'])==r['sha256'],'independent input changed')
    return m


def collect(out):
    out=Path(out).resolve();m=check(out);auth=read(out/'authorization.json')
    require(auth.get('collection_authorized') is True and auth.get('manifest_sha256')==sha(out/'manifest.json')
        and auth.get('frozen_diagnostic_sha256')==sha(out/'frozen-diagnostic.json') and auth.get('diagnostic_only') is True,
        'root must authorize this exact independent diagnostic; original qualification remains failed')
    require(not (out/'started.json').exists(),'no retries/repeated independent acquisition');write(out/'started.json',dict(started_at_utc=original.base.e2a.utc(),diagnostic_only=True))
    executions=[]
    with original.base.run.measurement_lock():
        for job in m['jobs']:
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request'])
            command=[m['binary']['path'],m['heldout_program']['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],status='failed',command=command,diagnostic_only=True,original_profile_supported=False,qualified=False);process=None;handles=[]
            try:
                env=original.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=original.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=240);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                record['verification']=original.validate_native(read(folder/'result.json'),job['request'],m['heldout_program'],process.pid)
                record.update(status='completed',native=receipt(folder/'result.json'))
            except BaseException as error:
                record['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for h in handles:h.close()
                write(folder/'execution.json',record)
            executions.append(record);print(json.dumps(dict(job_id=job['id'],status=record['status'],error=record.get('error'))),flush=True)
            if record['status']!='completed':break
    write(out/'executions.json',executions)
    return dict(declared=6,acquired=len(executions),completed=sum(x['status']=='completed' for x in executions),not_run=6-len(executions),qualification=False)


def evaluate(out):
    out=Path(out).resolve();m=check(out);executions=read(out/'executions.json')
    require(len(executions)==6 and all(x['status']=='completed' and x['process_exited_verified'] for x in executions),'all six independent processes required')
    jobs={j['id']:j for j in m['jobs']};natives=[]
    for e in executions:
        require(sha(e['native']['path'])==e['native']['sha256'],'independent observation changed');native=read(e['native']['path']);job=jobs[e['job_id']]
        original.validate_native(native,job['request'],m['heldout_program'],e['process_id']);natives.append((job,native))
    rows=[]
    for p in read(out/'predictions.json')['rows']:
        for job,native in natives:
            if job['memory_mode']!=p['memory_mode']:continue
            bursts=[b for b in native['bursts'] if b['bytes']==p['bytes'] and b['requested_idle_ms']==p['requested_idle_ms']];require(len(bursts)==12,'complete condition required')
            first=[b['copies'][0]['wall_ns'] for b in bursts];wall=[b['burst_wall_ns'] for b in bursts];copy_sum=[sum(c['wall_ns'] for c in b['copies']) for b in bursts]
            mf,mw=statistics.mean(first),statistics.mean(wall);ef,ew=abs(p['first_prediction_ns']/mf-1),abs(p['burst_prediction_ns']/mw-1)
            cf,cw=statistics.pstdev(first)/mf,statistics.pstdev(wall)/mw
            rows.append(dict(job_id=job['id'],memory_mode=p['memory_mode'],kind=p['kind'],bytes=p['bytes'],requested_idle_ms=p['requested_idle_ms'],
                first_predicted_ns=p['first_prediction_ns'],first_measured_mean_ns=mf,first_APE=ef,first_measured_population_CV=cf,
                burst_predicted_ns=p['burst_prediction_ns'],burst_measured_mean_ns=mw,burst_APE=ew,burst_measured_population_CV=cw,
                measured_copy_sum_mean_ns=statistics.mean(copy_sum),measured_loop_gap_mean_ns=mw-statistics.mean(copy_sum),
                diagnostic_mean_accuracy_passed=ef<=.20 and ew<=.20,diagnostic_stability_passed=cf<=.10 and cw<=.10,
                diagnostic_mean_and_stability_passed=ef<=.20 and ew<=.20 and cf<=.10 and cw<=.10,
                strict_qualification_passed=False,strict_failure_reason='original profile baseline support failed',
                heldout_used_for_fit=False,prices_exported=False))
    require(len(rows)==72,'all three providers across all independent conditions/processes required')
    summaries=[]
    for memory in ('registered','pageable'):
        for kind in fitting.KINDS:
            subset=[r for r in rows if r['memory_mode']==memory and r['kind']==kind]
            summaries.append(dict(memory_mode=memory,kind=kind,declared=12,
                first_mean_APE=statistics.mean(r['first_APE'] for r in subset),burst_mean_APE=statistics.mean(r['burst_APE'] for r in subset),
                diagnostic_mean_passed=sum(r['diagnostic_mean_accuracy_passed'] for r in subset),
                diagnostic_mean_and_stability_passed=sum(r['diagnostic_mean_and_stability_passed'] for r in subset),
                strict_qualified=0))
    result=dict(schema='h2d-v9-independent-diagnostic-result-1',classification='independent diagnostic; not formal accuracy qualification',
        original_failed_profile_barrier=m['original_failed_barrier'],original_fit=m['original_fit'],predictions=receipt(out/'predictions.json'),
        declared_processes=6,completed_processes=6,declared_provider_cells=72,rows=rows,summaries=summaries,
        strict_qualified=0,original_profile_gate_unchanged=True,parameters_refitted=False,slow_trials_removed=0,
        diagnostic_mean_accuracy_passed=sum(r['diagnostic_mean_accuracy_passed'] for r in rows),
        diagnostic_mean_and_stability_passed=sum(r['diagnostic_mean_and_stability_passed'] for r in rows),
        hardware_accuracy_qualification=False,prices_exported=False,whole_LLM_accuracy_validated=False,
        next_model_revision_requires_new_profile_and_independent_holdout=True)
    write(out/'DIAGNOSTIC_EVALUATION.json',result)
    return dict(output=receipt(out/'DIAGNOSTIC_EVALUATION.json'),strict_qualified=0,declared=72,
        diagnostic_mean_passed=result['diagnostic_mean_accuracy_passed'],diagnostic_mean_and_stability_passed=result['diagnostic_mean_and_stability_passed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect','evaluate'));p.add_argument('--out');p.add_argument('--parent');a=p.parse_args()
    value=protocol() if a.command=='protocol' else prepare(a.out,a.parent) if a.command=='prepare' else collect(a.out) if a.command=='collect' else evaluate(a.out)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
