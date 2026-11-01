"""Explicitly authorized six-process numerical cohort; unchanged safety policy."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import psutil
from .dataset import read,write
from .run import child_environment,file_sha,receipt,sha,require
from .numerical_v6 import ENV_KEYS,ORDER
from .tap_campaign_v5 import available
from .postcollection import expected_forwards
from .numerical_vectors import inspect
from profiling.state_calibration import run as common

def verify_native(n,request,pid,spec):
    require(n['schema']==spec['expected_result_schema']=='real-moe-numerical-native-6' and n['request']==request and n['process_id']==pid,'numerical native identity/schema differs')
    require(request['schema']==spec['expected_request_schema']=='real-moe-numerical-request-6','numerical request schema differs')
    require(n['source_sha256']==request['source_sha256']==spec['expected_source_sha256'],'new driver source binding differs')
    require(n['model_sha256_verified_by_parent']==request['model_sha256'] and n['observed_model_contract']==request['model_contract'],'model binding differs')
    require(n['model_weights_loaded'] is True and n['actual_placement'] is not None and n['graph_or_allocator_modified'] is False,'actual unchanged model execution evidence missing')
    require(n['raw_runtime_environment']==request['expected_runtime_environment'],'raw environment differs')
    require(n['scheduler_callback_enabled'] is False and n['trace_callback_enabled'] is False,'scheduler callback unexpectedly enabled')
    require(n['threadpool_policy']==request['threadpool_policy']=='explicit_persistent_owned' and n['process_affinity']['observed_mask']==request['expected_process_mask'],'threadpool/affinity differs')
    require(n['gpu_tap_hook_available'] is True and n['cpu_consumer_tap_enabled']==(request['mode']=='capture'),'v5b observer ABI/mode differs')
    require([r['id'] for r in n['results']]==[r['id'] for r in request['requests']],'observation inventory differs')
    artifact_checks=[]
    for index,(row,source) in enumerate(zip(n['results'],request['requests'])):
        require(row['origin_case_id']==source['origin_case_id'] and row['context_reset_index']==source['context_reset_index']==index,'origin/reset sequence differs')
        require(row['memory_clear_data_requested'] is True and row['memory_clear_call_count_for_this_request']==1,'normal reset request differs')
        require(row['prompt_token_ids']==source['prompt_token_ids'] and 1<=row['output_count']<=source['max_output_tokens'],'prompt/output inventory differs')
        require(row['output_count']==len(row['output_token_ids'])==len(row['sampled_greedy_token_ids'])==len(row['logit_observations']),'step inventory differs')
        if 'output_token_ids' in source:require(row['output_token_ids']==source['output_token_ids'],'frozen teacher-forced stream differs')
        forwards=expected_forwards(len(row['prompt_token_ids']),row['output_count'])
        require(row['forward_count']==len(forwards) and row['decode_forward_count']==row['output_count']-1,'normal forward/pruning boundary differs')
        require(len(row['actual_batch_inputs'])==len(forwards)==len(row['graph_observations']),'actual batch/graph coverage missing')
        for actual,f in zip(row['actual_batch_inputs'],forwards):
            tokens=(row['prompt_token_ids'][f['position']:f['position']+f['tokens']] if f['phase']=='prefill' else [row['output_token_ids'][f['position']-len(row['prompt_token_ids'])]])
            expected=[dict(token=token,position=f['position']+i,n_seq_id=1,seq_id=0,logits=bool(f['output_rows'] and i+1==len(tokens))) for i,token in enumerate(tokens)]
            require(actual['forward_index']==f['index'] and actual['rows']==expected,'actual executed token/position/sequence/output mask differs')
            import hashlib
            payload=b''.join(struct.pack('<5i',r['token'],r['position'],r['n_seq_id'],r['seq_id'],int(r['logits'])) for r in expected)
            require(actual['actual_input_rows_sha256']==hashlib.sha256(payload).hexdigest(),'actual batch input SHA differs')
        require(row['timing_usable_as_bare_reference'] is False and row['observer_extra_flush_ns']==0,'diagnostic timing/synchronization boundary differs')
        require(row['gpu_capture_error_code']==0,'GPU observer unsupported/error; preserve without retry')
        artifact=row['full_logits_artifact'];require(artifact['path']==source['logit_artifact_path'],'host artifact path differs')
        require(artifact['rows']==row['output_count'] and artifact['existing_host_copy_used'] is True and artifact['new_GPU_reads']==0 and artifact['export_in_target_interval'] is False,'host snapshot scope differs')
        artifact_checks.append(dict(observation_id=row['id'],inspection=inspect(artifact,row['logit_observations'])))
    return dict(passed=True,acquisition_and_host_artifact_contract_only=True,whole_model_accuracy_certified=False,numerical_equality_not_assumed=True,artifacts=artifact_checks)

def authorize(out):
    out=Path(out);barrier=read(out/'FREEZE_BARRIER.json');root=read(out/'ROOT_AUTHORIZATION_RECEIPT.json')
    require(root.get('collection_authorized') is True,'explicit root authorization required')
    require(root.get('manifest_sha256')==sha(out/'manifest.json') and root.get('freeze_barrier_sha256')==sha(out/'FREEZE_BARRIER.json'),'root review SHA differs')
    require(barrier['manifest']['sha256']==sha(out/'manifest.json'),'manifest changed after freeze')

def collect(out,job):
    out=Path(out).resolve();authorize(out);m=read(out/'manifest.json');order=m['fixed_order'];require(job in order,'job outside frozen order')
    for previous in order[:order.index(job)]:require(read(out/'jobs'/previous/'execution.json')['status']=='completed','earlier job failed/missing; no retry')
    spec=next(x for x in m['jobs'] if x['job_id']==job);request=read(spec['request']['path']);folder=out/'jobs'/job;require(not folder.exists(),'no overwrite/retry')
    folder.mkdir(parents=True);write(folder/'request.json',request);command=[m['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-numerical-execution-6',job_id=job,status='failed',request=receipt(folder/'request.json'),command=command,binary=m['binary'],runtime=m['runtime'],capacity_policy=m['capacity_policy'],observer_interval_ms=200,external_observer_active=True,observer_used_for_prediction=False,timings_are_observer_inclusive=True)
    proc=None;handles=[];samples=[];low=0
    with common.measurement_lock():
        try:
            resources=[spec['request'],m['driver_build'],m['source_smoke_manifest'],m['source_greedy_trajectory'],m['old_failure_preserved'],m['capacity_policy'],m['binary'],*m['runtime'],*sum(([r['source'],r['archived']] for r in m['sources']),[])]
            for resource in resources:require(sha(resource['path'])==resource['sha256'],'frozen diagnostic input changed: '+resource['path'])
            require(file_sha(m['model']['path'])==m['model']['sha256'],'model content changed')
            env=child_environment(m['runtime_dirs'],request)
            for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):env[key]='1'
            require({k:env.get(k) for k in ENV_KEYS}==m['raw_expected_environment'],'declared raw environment differs; do not silently alter workspace/graph policy')
            record['raw_child_environment']={k:env.get(k) for k in ENV_KEYS};record['available_before_launch_bytes']=available()
            require(record['available_before_launch_bytes']>=m['parent_preflight_bytes'],'unchanged parent capacity budget not met')
            proc,a,b,affinity=common.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',request['expected_process_mask']);handles=[a,b];record.update(process_id=proc.pid,process_affinity=affinity)
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
            require(proc.returncode==0,'native diagnostic failed; no retry');n=read(folder/'result.json');record['verification']=verify_native(n,request,proc.pid,spec);record.update(status='completed',native=receipt(folder/'result.json'))
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
    write(folder/'OBSERVER_SUMMARY.json',dict(status=record['status'],execution=receipt(folder/'execution.json'),observed_global_min_available_bytes=min((s['global_available_bytes'] for s in samples),default=None),peak_wset_bytes=max((s.get('process_memory',{}).get('peak_wset',0) for s in samples),default=None),observer_used_as_prediction_input=False))
    print(json.dumps(dict(job_id=job,status=record['status'],error=record.get('error'))),flush=True);return record

def collect_all(out):
    out=Path(out).resolve();authorize(out);m=read(out/'manifest.json')
    write(out/'PIPELINE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),fixed_order=m['fixed_order'],automatic_retry=False))
    for job in m['fixed_order']:
        result=collect(out,job)
        if result['status']!='completed':
            write(out/'ACQUISITION_STOPPED.json',dict(failed_job=job,remaining_not_run=m['fixed_order'][m['fixed_order'].index(job)+1:],retry_performed=False));return dict(completed=False,failed_job=job)
    final=dict(schema='real-moe-numerical-acquisition-complete-6',completed_at_utc=datetime.now(timezone.utc).isoformat(),completed_processes=6,request_observations=24,all_native_processes_exited=True,whole_model_accuracy_evaluated=False,numerical_comparison_pending=True)
    write(out/'ACQUISITION_COMPLETE.json',final);return final

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('collect','collect-all'));p.add_argument('--out',required=True);p.add_argument('--job');a=p.parse_args()
    print(json.dumps(collect_all(a.out) if a.command=='collect-all' else collect(a.out,a.job),ensure_ascii=False))
if __name__=='__main__':main()
