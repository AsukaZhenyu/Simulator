"""Bounded old-binary replication with byte-identical frozen requests.

No native source/DLL/request translation and no monkeypatching previous runners.
Preparation and software tests never authorize a model process.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from .dataset import read,write
from .run import child_environment,file_sha,receipt,sha,require
from .tap_campaign_v5 import available,verify_native
from .numerical_v6 import ENV_KEYS
from profiling.state_calibration import run as common

ROOT=Path(__file__).resolve().parents[2]
PLAN=(('private-00','private_v5'),('original-00','original_v4'),('private-01','private_v5'),('private-02','private_v5'),('original-01','original_v4'),('private-03','private_v5'))

def copy_exact_request(source,target,expected_sha):
    source=Path(source);target=Path(target);require(sha(source)==expected_sha,'frozen executed request changed')
    require(not target.exists(),'request destination exists; no overwrite');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    require(sha(target)==expected_sha,'byte-exact request copy differs');return receipt(target)

def prepare(out,old_smoke):
    out=Path(out).resolve();old_smoke=Path(old_smoke).resolve();require(not out.exists(),'new frozen-request cohort root required')
    old=read(old_smoke/'manifest.json');out.mkdir(parents=True);jobs=[];sources=[]
    for job,binding_key in PLAN:
        kind='hybrid-private-normal' if binding_key=='private_v5' else 'hybrid-original-normal'
        previous=read(old_smoke/'jobs'/kind/'execution.json');require(previous['status']=='completed','original executed request did not complete')
        executed=old_smoke/'jobs'/kind/'request.json';require(sha(executed)==previous['request']['sha256'],'old execution request binding differs')
        req_receipt=copy_exact_request(executed,out/'requests'/(job+'.json'),previous['request']['sha256']);req=read(executed);binding=old['bindings'][binding_key]
        require(req['mode']=='generate' and len(req['requests'])==2 and all(r['max_output_tokens']==8 for r in req['requests']),'old fixed two-request normal scope differs')
        jobs.append(dict(job_id=job,binding=binding_key,request=req_receipt,original_executed_request=previous['request'],original_execution=receipt(old_smoke/'jobs'/kind/'execution.json'),expected_request_schema=binding['request_schema'],expected_result_schema=binding['result_schema'],expected_source_sha256=binding['source_sha256']))
    for path in (Path(__file__),Path(__file__).parent/'test_frozen_request_runner.py',Path(__file__).parent/'tap_campaign_v5.py',Path(__file__).parent/'numerical_v6.py',Path(__file__).parent/'run.py',Path(__file__).parent/'dataset.py',ROOT/'profiling/state_calibration/run.py'):
        target=out/'executed_sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources.append(dict(source=receipt(path),archived=receipt(target)))
    m=dict(schema='real-moe-frozen-request-campaign-1',prepared_at_utc=datetime.now(timezone.utc).isoformat(),source_manifest=receipt(old_smoke/'manifest.json'),source_failure=receipt(old_smoke/'SMOKE_ANALYSIS.json'),model=old['model'],capacity_policy=old['capacity_policy'],native_guard_bytes=old['native_guard_bytes'],parent_preflight_bytes=old['parent_preflight_bytes'],resident_peak_limit_bytes=old['resident_peak_limit_bytes'],bindings=old['bindings'],fixed_order=[x[0] for x in PLAN],jobs=jobs,sources=sources,old_binary_DLL_and_request_bytes_unchanged=True,native_source_changed=False,native_sidecar_added=False,native_request_fields_added=False,native_CPU_or_GPU_build=False,max_fresh_processes=6,private_fresh_processes=4,original_control_fresh_processes=2,requests_per_process=2,max_output_tokens=8,warmup_replays=0,automatic_retries=False,whole_model_accuracy_evaluated=False,scope='replicate exact old two-request protocol without v6 host layout or four-request change; no performance truth or cause assumed',limits=['The six predeclared observations are finite diagnostics, not proof of randomness or all-state equivalence','Original v5b failed observation remains the main historical failure, never overwritten','Complete float vectors are unavailable with these frozen binaries; only existing full-row hashes/top logits/graph metadata are retained','hybrid remains unchanged buffer override/op_offload configuration; actual graph backend determines execution placement'])
    write(out/'manifest.json',m);return receipt(out/'manifest.json')

def freeze(out,tests):
    out=Path(out).resolve();t=read(tests);require(t['returncode']==0 and t.get('hardware_executed') is False,'passing software-only tests required')
    review=out/'EXISTING_HASH_REVIEW.json'
    write(out/'FREEZE_BARRIER.json',dict(schema='real-moe-frozen-request-freeze-1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),manifest=receipt(out/'manifest.json'),software_tests=receipt(tests),existing_read_only_review=receipt(review) if review.exists() else None,old_native_and_request_bytes_unchanged=True,collection_authorized=False,root_authorization_required=True))
    return receipt(out/'FREEZE_BARRIER.json')

def authorize(out):
    out=Path(out);barrier=read(out/'FREEZE_BARRIER.json');root=read(out/'ROOT_AUTHORIZATION_RECEIPT.json')
    require(root.get('collection_authorized') is True,'root collection authorization missing')
    require(root.get('manifest_sha256')==sha(out/'manifest.json') and root.get('freeze_barrier_sha256')==sha(out/'FREEZE_BARRIER.json'),'root authorization SHA differs')
    require(barrier['manifest']['sha256']==sha(out/'manifest.json'),'manifest changed after freeze')

def collect(out,job):
    out=Path(out).resolve();authorize(out);m=read(out/'manifest.json');require(job in m['fixed_order'],'job outside fixed order')
    for previous in m['fixed_order'][:m['fixed_order'].index(job)]:require(read(out/'jobs'/previous/'execution.json')['status']=='completed','previous job failed/missing; do not retry')
    spec=next(x for x in m['jobs'] if x['job_id']==job);binding=m['bindings'][spec['binding']];req=read(spec['request']['path']);folder=out/'jobs'/job;require(not folder.exists(),'no retry/overwrite')
    folder.mkdir(parents=True);request_copy=copy_exact_request(spec['request']['path'],folder/'request.json',spec['request']['sha256']);command=[binding['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-frozen-request-execution-1',job_id=job,status='failed',request=request_copy,original_request=spec['original_executed_request'],binary=binding['binary'],runtime=binding['runtime'],expected_schema=spec['expected_result_schema'],expected_source_sha256=spec['expected_source_sha256'],command=command,capacity_policy=m['capacity_policy'],external_observer_active=True,observer_interval_ms=200,observer_used_for_prediction=False,native_or_request_transformation_performed=False)
    proc=None;handles=[];samples=[];low=0
    with common.measurement_lock():
        try:
            require(os.environ.get('GGML_CUDA_DISABLE_GRAPHS') is None,'old default-enabled graph environment differs')
            resources=[spec['request'],spec['original_executed_request'],spec['original_execution'],m['source_manifest'],m['source_failure'],m['capacity_policy'],binding['binary'],*binding['runtime'],*sum(([x['source'],x['archived']] for x in m['sources']),[])]
            for resource in resources:require(sha(resource['path'])==resource['sha256'],'frozen input changed: '+resource['path'])
            require(file_sha(m['model']['path'])==m['model']['sha256'],'model content changed')
            env=child_environment(binding['runtime_dirs'],req)
            for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):env[key]='1'
            record['raw_child_environment']={k:env.get(k) for k in ENV_KEYS};record['available_before_launch_bytes']=available()
            require(record['available_before_launch_bytes']>=m['parent_preflight_bytes'],'unchanged parent capacity precondition not met')
            proc,a,b,affinity=common.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',req['expected_process_mask']);handles=[a,b];record.update(process_id=proc.pid,process_affinity=affinity)
            process=psutil.Process(proc.pid);start=time.monotonic();write(folder/'STARTED.json',dict(process_id=proc.pid,job_id=job,started_at_utc=datetime.now(timezone.utc).isoformat()));print(json.dumps(dict(started=job,process_id=proc.pid)),flush=True)
            with (folder/'observer.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                while proc.poll() is None:
                    sample=dict(observer_tick_ns=time.perf_counter_ns(),global_available_bytes=available())
                    try:sample['process_memory']=process.memory_info()._asdict()
                    except (psutil.NoSuchProcess,psutil.AccessDenied) as error:sample['process_memory_error']=type(error).__name__
                    samples.append(sample);stream.write(json.dumps(sample,separators=(',',':'))+'\n');stream.flush();low=low+1 if sample['global_available_bytes']<512*1024**2 else 0
                    if low>=2 or sample.get('process_memory',{}).get('peak_wset',0)>m['resident_peak_limit_bytes'] or time.monotonic()-start>300:
                        record['abort_reason']='unchanged resident/available watchdog or fixed five-minute timeout';proc.terminate();break
                    try:proc.wait(timeout=.2)
                    except subprocess.TimeoutExpired:pass
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            record.update(returncode=proc.returncode,process_exited_verified=True,forced_termination='abort_reason' in record)
            require(proc.returncode==0,'old native failed; preserve without retry');n=read(folder/'result.json');record['verification']=verify_native(n,req,proc.pid,spec,binding);record.update(status='completed',native=receipt(folder/'result.json'))
        except BaseException as error:
            record['error']=repr(error)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            if proc is not None:record.update(returncode=proc.returncode,process_exited_verified=proc.poll() is not None)
        finally:
            for handle in handles:handle.close()
            write(folder/'execution.json',record)
    write(folder/'OBSERVER_SUMMARY.json',dict(status=record['status'],execution=receipt(folder/'execution.json'),observed_global_min_available_bytes=min((x['global_available_bytes'] for x in samples),default=None),peak_wset_bytes=max((x.get('process_memory',{}).get('peak_wset',0) for x in samples),default=None),observer_used_as_prediction_input=False))
    print(json.dumps(dict(job_id=job,status=record['status'],error=record.get('error'))),flush=True);return record

def collect_all(out):
    out=Path(out).resolve();authorize(out);m=read(out/'manifest.json');write(out/'PIPELINE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),fixed_order=m['fixed_order'],automatic_retry=False))
    for job in m['fixed_order']:
        result=collect(out,job)
        if result['status']!='completed':write(out/'ACQUISITION_STOPPED.json',dict(failed_job=job,remaining_not_run=m['fixed_order'][m['fixed_order'].index(job)+1:],retry_performed=False));return dict(completed=False,failed_job=job)
    final=dict(schema='real-moe-frozen-request-acquisition-complete-1',completed_at_utc=datetime.now(timezone.utc).isoformat(),completed_processes=6,request_observations=12,all_native_processes_exited=True,native_or_request_changes=False,whole_model_accuracy_evaluated=False)
    write(out/'ACQUISITION_COMPLETE.json',final);return final

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','freeze','collect','collect-all'));p.add_argument('--out',required=True);p.add_argument('--old-smoke');p.add_argument('--tests');p.add_argument('--job');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.old_smoke)
    elif a.command=='freeze':r=freeze(a.out,a.tests)
    elif a.command=='collect':r=collect(a.out,a.job)
    else:r=collect_all(a.out)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
