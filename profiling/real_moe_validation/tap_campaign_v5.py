"""Schema-bound v4 baseline / v5 GPU observer diagnostic, never a cost source.

prepare/freeze do not launch inference. collect requires a separate root receipt.
The frozen v5 driver requires its private CUDA ABI, so old-runtime baseline jobs
explicitly use the frozen v4 executable, schema and source instead.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from .dataset import read,write
from .run import child_environment,file_sha,receipt,sha,require
from profiling.state_calibration import run as common

ROOT=Path(__file__).resolve().parents[2]
ORDER=tuple(f'{placement}-{kind}' for placement in ('cpu','hybrid') for kind in ('original-normal','private-normal','replay','capture'))
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','ext')]

def available():
    s=Memory();s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('GlobalMemoryStatusEx failed')
    return s.available

def semantic_request(request):
    """Explicit cross-version normalization; never changes an acquired record."""
    return {k:v for k,v in request.items() if k not in ('schema','source_sha256')}

def prepare(out,v5_build,v4_smoke):
    out,v5_build,v4_smoke=map(lambda p:Path(p).resolve(),(out,v5_build,v4_smoke))
    require(not out.exists(),'new v5 diagnostic root required')
    old=read(v4_smoke/'manifest.json');build=read(v5_build/'BUILD_COMPLETE.json');prepared=read(v5_build/'PREPARED.json')
    require(build['schema']=='real-moe-gpu-consumer-build-complete-5','qualified private build schema required')
    old_req=read(v4_smoke/'requests/cpu-original-normal.json');require(old_req['schema']=='real-moe-tap-request-4','original request schema differs')
    require(len(old_req['requests'])==2 and all(r['max_output_tokens']==8 for r in old_req['requests']),'fixed two-case eight-token smoke required')
    out.mkdir(parents=True);(out/'original-bin').mkdir()
    original=dict(old['bindings']['original_cpu_dll']);require(sha(original['binary']['path'])==original['binary']['sha256'],'frozen v4 baseline binary changed');target=out/'original-bin'/Path(original['binary']['path']).name
    shutil.copy2(original['binary']['path'],target);original['binary']=receipt(target)
    original.update(request_schema='real-moe-tap-request-4',result_schema='real-moe-tap-native-4',source_sha256=old_req['source_sha256'])
    private=dict(binary=next(r for r in build['binaries'] if Path(r['path']).suffix=='.exe'),runtime=[r for r in build['binaries'] if Path(r['path']).suffix=='.dll'],runtime_dirs=[str(v5_build/'bin'),'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin'],request_schema='real-moe-tap-request-5',result_schema='real-moe-tap-native-5',source_sha256=prepared['native_source_sha256'])
    m=dict(schema='real-moe-gpu-tap-campaign-5',prepared_at_utc=datetime.now(timezone.utc).isoformat(),v4_source_manifest=receipt(v4_smoke/'manifest.json'),v5_build=receipt(v5_build/'BUILD_COMPLETE.json'),v5_review=receipt(v5_build/'GPU_OBSERVER_REVIEW.json'),model=old['model'],model_contract=old['model_contract'],capacity_policy=old['capacity_policy'],native_guard_bytes=old['native_guard_bytes'],parent_preflight_bytes=old['parent_preflight_bytes'],resident_peak_limit_bytes=old['resident_peak_limit_bytes'],bindings={'original_v4':original,'private_v5':private},fixed_order=list(ORDER),jobs=[],sources=[],whole_model_accuracy_evaluated=False,performance_parameters_fitted=False,timings_usable_as_bare_reference=False,observer_interval_ms=200,timeout_seconds=300,request_count_per_process=2,max_output_tokens=8,warmup_replays=0,limits=['hybrid label denotes unchanged tensor-buffer override/op_offload configuration, not CPU expert execution in every phase','actual graph node backend determines consumer placement; CPU weight residency is not execution placement','GPU-prefill time must not populate CPU expert cost cells','CUDAgraph expert capture/replay and unobserved fused consumers are unsupported','all timing is observer-inclusive diagnostic data; no end-to-end accuracy or cost fitting','old calibration short/long requests diagnose mechanism; not independent heldout accuracy'])
    for placement in ('cpu','hybrid'):
        old_spec=next(j for j in old['jobs'] if j['job_id']==f'{placement}-original-normal')
        require(sha(old_spec['request']['path'])==old_spec['request']['sha256'],'frozen v4 baseline request changed')
        original_req=read(old_spec['request']['path'])
        private_req=dict(original_req,schema=private['request_schema'],source_sha256=private['source_sha256'])
        require(semantic_request(original_req)==semantic_request(private_req),'cross-version semantic request differs')
        for label,binding,req in (('original-normal','original_v4',original_req),('private-normal','private_v5',private_req)):
            path=out/'requests'/f'{placement}-{label}.json';write(path,req)
            m['jobs'].append(dict(job_id=f'{placement}-{label}',binding=binding,request=receipt(path),expected_request_schema=m['bindings'][binding]['request_schema'],expected_result_schema=m['bindings'][binding]['result_schema'],expected_source_sha256=m['bindings'][binding]['source_sha256']))
    for job in ORDER:
        if job.endswith(('replay','capture')):m['jobs'].append(dict(job_id=job,binding='private_v5',request=None,request_derivation='actual private-normal output_token_ids only; no native/config/source change',normal_reference_job=job.rsplit('-',1)[0]+'-private-normal',expected_request_schema=private['request_schema'],expected_result_schema=private['result_schema'],expected_source_sha256=private['source_sha256']))
    for path in (Path(__file__),Path(__file__).parent/'tap_audit_v5.py',Path(__file__).parent/'tap_v5.py',Path(__file__).parent/'test_tap_campaign_v5.py',Path(__file__).parent/'test_tap_audit_v5.py',Path(__file__).parent/'run.py',Path(__file__).parent/'dataset.py',ROOT/'profiling/state_calibration/run.py'):
        require(path.is_file(),'frozen helper/test absent: '+str(path));target=out/'executed_sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        m['sources'].append(dict(source=receipt(path),archived=receipt(target)))
    m['jobs'].sort(key=lambda j:ORDER.index(j['job_id']));write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),fixed_jobs=8,normal_request_schemas=[4,5],hardware_started=False)

def freeze(out,tests_receipt):
    out=Path(out).resolve();m=read(out/'manifest.json');tests=read(tests_receipt)
    require(tests['returncode']==0 and tests.get('hardware_executed') is False,'passing software-only tests required')
    record=dict(schema='real-moe-gpu-tap-freeze-5',frozen_at_utc=datetime.now(timezone.utc).isoformat(),manifest=receipt(out/'manifest.json'),software_tests=receipt(tests_receipt),review=m['v5_review'],fixed_order=m['fixed_order'],dynamic_requests='only copy actual private-normal token IDs into predeclared paired requests',whole_model_accuracy_evaluated=False,collection_authorized=False,root_review_receipt_required=True)
    write(out/'FREEZE_BARRIER.json',record);return receipt(out/'FREEZE_BARRIER.json')

def authorization(out):
    out=Path(out);barrier=read(out/'FREEZE_BARRIER.json');root=read(out/'ROOT_AUTHORIZATION_RECEIPT.json')
    require(root.get('collection_authorized') is True,'explicit root collection authorization missing')
    require(root.get('manifest_sha256')==sha(out/'manifest.json') and root.get('freeze_barrier_sha256')==sha(out/'FREEZE_BARRIER.json'),'root review binding differs')
    require(barrier['manifest']['sha256']==sha(out/'manifest.json'),'manifest changed since freeze')
    return root

def make_pairs(out,placement):
    out=Path(out).resolve();m=read(out/'manifest.json');folder=out/'jobs'/f'{placement}-private-normal';execution=read(folder/'execution.json')
    require(execution['status']=='completed','actual private normal must complete before pair derivation')
    n=read(folder/'result.json');rows={r['id']:r for r in n['results']};plans=[]
    for mode in ('replay','capture'):
        req=dict(n['request'],mode=mode,requests=[dict(r,output_token_ids=rows[r['id']]['output_token_ids']) for r in n['request']['requests']])
        path=out/'requests'/f'{placement}-{mode}.json';write(path,req)
        spec=next(j for j in m['jobs'] if j['job_id']==f'{placement}-{mode}');plan=dict(spec,request=receipt(path),normal_reference=receipt(folder/'result.json'))
        write(out/'pair-plans'/f'{placement}-{mode}.json',plan);plans.append(plan)
    return plans

def verify_native(n,r,pid,spec,binding):
    require(r['schema']==spec['expected_request_schema']==binding['request_schema'],'per-job request schema differs')
    require(n['schema']==spec['expected_result_schema']==binding['result_schema'] and n['request']==r and n['process_id']==pid,'per-job native identity/schema differs')
    require(n['source_sha256']==r['source_sha256']==spec['expected_source_sha256']==binding['source_sha256'],'per-job source binding differs')
    require(n['model_sha256_verified_by_parent']==r['model_sha256'],'model binding differs')
    require(n['scheduler_callback_enabled'] is False and n['trace_callback_enabled'] is False,'scheduler callback enabled')
    require(n['cpu_consumer_tap_enabled']==(r['mode']=='capture'),'CPU tap mode differs')
    require(n['observed_model_contract']==r['model_contract'] and n['threadpool_policy']==r['threadpool_policy']=='explicit_persistent_owned','model/threadpool differs')
    require(n['process_affinity']['observed_mask']==r['expected_process_mask'],'affinity differs')
    require([x['id'] for x in n['results']]==[x['id'] for x in r['requests']],'request result inventory differs')
    private=binding['result_schema']=='real-moe-tap-native-5'
    if private:require(n['gpu_tap_hook_available'] is True,'private GPU ABI absent')
    for row,source in zip(n['results'],r['requests']):
        require(row['prompt_token_ids']==source['prompt_token_ids'],'prompt token stream differs')
        require(row['output_count']==len(row['sampled_greedy_token_ids'])==len(row['output_token_ids']),'output inventory differs')
        require(row['decode_forward_count']==row['output_count']-1,'extra/missing final forward')
        require(len(row['logit_observations'])==row['output_count'] and len(row['graph_observations'])==row['forward_count'],'full logit/graph inventory missing')
        require(row['timing_usable_as_bare_reference'] is False,'observer timing promoted to bare')
        if 'output_token_ids' in source:require(row['output_token_ids']==source['output_token_ids'],'teacher-forced output differs')
        if r['mode']!='capture':require(not row['routes'],'normal/replay unexpectedly captured routes')
        if private:
            require(row['gpu_capture_error_code']==0,'GPU observer explicitly unsupported/error; preserve without retry')
            require(row['gpu_observer_copy_count']==len(row['gpu_copy_inventory']),'GPU copy ledger count differs')
            require(row['observer_extra_flush_ns']==0 and row['observer_completion_boundary']=='existing llama_synchronize only','observer completion boundary differs')
    return dict(passed=True,acquisition_only=True,requests=len(n['results']),route_and_numerical_pair_audit_required=True,whole_model_accuracy_certified=False)

def collect(out,job_id):
    out=Path(out).resolve();authorization(out);m=read(out/'manifest.json')
    require(job_id in ORDER,'job outside frozen order')
    for previous in ORDER[:ORDER.index(job_id)]:require(read(out/'jobs'/previous/'execution.json')['status']=='completed','previous fixed-order job missing/failed; stop without retry')
    spec=next(j for j in m['jobs'] if j['job_id']==job_id)
    if spec['request'] is None:spec=read(out/'pair-plans'/f'{job_id}.json')
    binding=m['bindings'][spec['binding']];r=read(spec['request']['path']);folder=out/'jobs'/job_id;require(not folder.exists(),'no retry or overwrite')
    folder.mkdir(parents=True);write(folder/'request.json',r);command=[binding['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-gpu-tap-execution-5',job_id=job_id,status='failed',request=receipt(folder/'request.json'),command=command,binding=spec['binding'],expected_schema=spec['expected_result_schema'],expected_source_sha256=spec['expected_source_sha256'],runtime=binding['runtime'],capacity_policy=m['capacity_policy'],observer_interval_ms=200,external_observer_active=True,observer_used_for_prediction=False,performance_parameters_fitted=False)
    proc=None;handles=[];low=0;samples=[]
    with common.measurement_lock():
        try:
            require(os.environ.get('GGML_CUDA_DISABLE_GRAPHS') is None,'parent CUDA graph environment differs')
            for resource in [spec['request'],m['v4_source_manifest'],m['v5_build'],m['v5_review'],m['capacity_policy'],binding['binary'],*binding['runtime'],*sum(([s['source'],s['archived']] for s in m['sources']),[])]:require(sha(resource['path'])==resource['sha256'],'frozen request/source/runtime changed: '+resource['path'])
            require(file_sha(m['model']['path'])==m['model']['sha256'],'model content changed')
            record['available_before_launch_bytes']=available();require(record['available_before_launch_bytes']>=m['parent_preflight_bytes'],'original fixed parent capacity budget not met')
            env=child_environment(binding['runtime_dirs'],r)
            for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):env[key]='1'
            record['BLAS_environment']={k:env[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}
            proc,a,b,affinity=common.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',r['expected_process_mask']);handles=[a,b]
            record.update(process_id=proc.pid,process_affinity=affinity);process=psutil.Process(proc.pid);start=time.monotonic()
            write(folder/'STARTED.json',dict(process_id=proc.pid,job_id=job_id,request=record['request'],started_at_utc=datetime.now(timezone.utc).isoformat()));print(json.dumps(dict(started=job_id,process_id=proc.pid)),flush=True)
            with (folder/'observer.jsonl').open('x',encoding='utf-8',newline='\n') as f:
                while proc.poll() is None:
                    sample=dict(observer_tick_ns=time.perf_counter_ns(),global_available_bytes=available())
                    try:sample['process_memory']=process.memory_info()._asdict()
                    except (psutil.NoSuchProcess,psutil.AccessDenied) as error:sample['process_memory_error']=type(error).__name__
                    samples.append(sample);f.write(json.dumps(sample,separators=(',',':'))+'\n');f.flush();low=low+1 if sample['global_available_bytes']<512*1024**2 else 0
                    if sample.get('process_memory',{}).get('peak_wset',0)>m['resident_peak_limit_bytes'] or low>=2 or time.monotonic()-start>m['timeout_seconds']:
                        record['abort_reason']='unchanged resident/available watchdog or fixed five-minute timeout';proc.terminate();break
                    try:proc.wait(timeout=.2)
                    except subprocess.TimeoutExpired:pass
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            record.update(returncode=proc.returncode,process_exited_verified=True,forced_termination='abort_reason' in record)
            require(proc.returncode==0,'native failed; preserve without retry');n=read(folder/'result.json');record['verification']=verify_native(n,r,proc.pid,spec,binding);record.update(status='completed',native=receipt(folder/'result.json'))
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
    write(folder/'OBSERVER_SUMMARY.json',dict(status=record['status'],execution=receipt(folder/'execution.json'),observed_global_min_available_bytes=min((x['global_available_bytes'] for x in samples),default=None),peak_wset_bytes=max((x.get('process_memory',{}).get('peak_wset',0) for x in samples),default=None),observer_input_to_predictor=False))
    print(json.dumps(dict(job_id=job_id,status=record['status'],error=record.get('error'))),flush=True);return record

def collect_all(out):
    out=Path(out).resolve();authorization(out)
    write(out/'PIPELINE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),fixed_order=list(ORDER),automatic_retry=False))
    for job in ORDER:
        if job.endswith('-replay'):make_pairs(out,job.rsplit('-',1)[0])
        result=collect(out,job)
        if result['status']!='completed':
            write(out/'ACQUISITION_STOPPED.json',dict(stopped_at_utc=datetime.now(timezone.utc).isoformat(),failed_job=job,remaining_not_run=list(ORDER[ORDER.index(job)+1:]),failed_execution=receipt(out/'jobs'/job/'execution.json'),retry_performed=False));return dict(completed=False,failed_job=job)
    record=dict(schema='real-moe-gpu-tap-acquisition-complete-5',completed_at_utc=datetime.now(timezone.utc).isoformat(),completed_jobs=8,failed_jobs=0,normal_or_observer_requests=16,all_native_processes_exited=True,whole_model_accuracy_evaluated=False,strict_pair_audit_pending=True,executions=[receipt(out/'jobs'/job/'execution.json') for job in ORDER])
    write(out/'ACQUISITION_COMPLETE.json',record);return record

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','freeze','make-pairs','collect','collect-all'));p.add_argument('--out',required=True);p.add_argument('--v5-build');p.add_argument('--v4-smoke');p.add_argument('--tests-receipt');p.add_argument('--placement',choices=('cpu','hybrid'));p.add_argument('--job-id');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.v5_build,a.v4_smoke)
    elif a.command=='freeze':r=freeze(a.out,a.tests_receipt)
    elif a.command=='make-pairs':r=make_pairs(a.out,a.placement)
    elif a.command=='collect-all':r=collect_all(a.out)
    else:r=collect(a.out,a.job_id)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
