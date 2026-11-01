"""Bounded v4 routing-observer diagnostic; explicit serialized collection only.

Preparation never launches a child. The previous resident-capacity policy and
200 ms safety observer are retained. No cost is fitted from these observations.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from .dataset import read,write
from .run import child_environment,file_sha,receipt,sha,require
from profiling.state_calibration import run as common

ROOT=Path(__file__).resolve().parents[2]
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','ext')]
def available():
    s=Memory();s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('GlobalMemoryStatusEx failed')
    return s.available
def prepare(out,build,legacy_collection):
    out,build,legacy_collection=map(lambda p:Path(p).resolve(),(out,build,legacy_collection));require(not out.exists(),'new diagnostic root required')
    old=read(legacy_collection/'manifest.json');bm=read(build/'BUILD_COMPLETE.json');prepared=read(build/'PREPARED.json')
    suite=read(legacy_collection/'request-suite-v3.json');cal=[r for r in suite['requests'] if r['split']=='calibration']
    rows=[dict(next(r for r in cal if r['length_class']==length),max_output_tokens=8) for length in ('short','long')]
    require(len({r['id'] for r in rows})==2,'short/long diagnostic requests missing')
    out.mkdir(parents=True);(out/'legacy-bin').mkdir()
    exe=next(r for r in bm['binaries'] if Path(r['path']).suffix=='.exe');shutil.copy2(exe['path'],out/'legacy-bin'/Path(exe['path']).name)
    old_dirs=old['runtime_dirs'];cuda_bin='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin'
    bindings={'original_cpu_dll':dict(binary=receipt(out/'legacy-bin'/Path(exe['path']).name),runtime=old['runtime'],runtime_dirs=[*old_dirs,cuda_bin]),
              'private_consumer_dll':dict(binary=exe,runtime=[r for r in bm['binaries'] if Path(r['path']).suffix=='.dll'],runtime_dirs=[str(build/'bin'),cuda_bin])}
    policy=ROOT/'build/real-moe-validation-20261026/capacity-policy-v2-resident/RESIDENT_CAPACITY_POLICY-v2.json'
    p=read(policy);guard=p['budget']['native_guard_bytes'];parent=p['budget']['parent_preflight_budget_bytes'];limit=p['resident_peak_limit_bytes']
    m=dict(schema='real-moe-tap-diagnostic-campaign-4',prepared_at_utc=datetime.now(timezone.utc).isoformat(),build=receipt(build/'BUILD_COMPLETE.json'),
        source_manifest=receipt(legacy_collection/'manifest.json'),source_suite=receipt(legacy_collection/'request-suite-v3.json'),model=old['model'],model_contract=old['model_contract'],
        native_source_sha256=prepared['native_source_sha256'],capacity_policy=receipt(policy),native_guard_bytes=guard,parent_preflight_bytes=parent,resident_peak_limit_bytes=limit,
        bindings=bindings,request_ids=[r['id'] for r in rows],max_output_tokens=8,warmup_replays=0,observe_logits=True,observe_graph=True,
        performance_timing_scope='diagnostic observation-inclusive host interval; never bare latency truth',
        whole_model_accuracy_evaluated=False,performance_parameters_fitted=False,jobs=[],sources=[])
    for path in (Path(__file__),Path(__file__).parent/'tap_audit.py',Path(__file__).parent/'tap_v4.py'):
        target=out/'executed_sources'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);m['sources'].append(dict(source=receipt(path),archived=receipt(target)))
    for placement,label in (('cpu','cpu'),('cpu-experts-gpu-other','hybrid')):
        req=dict(schema='real-moe-tap-request-4',source_sha256=m['native_source_sha256'],mode='generate',model_contract=old['model_contract'],tokenization_add_special=False,
            threadpool_policy='explicit_persistent_owned',model_path=old['model']['path'],model_sha256=old['model']['sha256'],model_bytes=old['model']['bytes'],placement=placement,threads=16,
            context_size=2048,batch_size=256,expected_process_mask=4294967295,warmup_replays=0,minimum_free_host_bytes=guard,minimum_free_gpu_bytes=6*1024**3,
            cuda_graph_policy='runtime_default_enabled',tf32_override='0',observe_logits=True,observe_graph=True,requests=rows)
        for binding,job in (('original_cpu_dll',label+'-original-normal'),('private_consumer_dll',label+'-private-normal')):
            target=out/'requests'/(job+'.json');write(target,req);m['jobs'].append(dict(job_id=job,binding=binding,request=receipt(target)))
    write(out/'manifest.json',m);return dict(manifest=receipt(out/'manifest.json'),requests=len(rows),predeclared_normal_jobs=4,paired_replay_capture_jobs_after_actual_normal=4,no_hardware_started=True)
def make_pairs(out,placement):
    out=Path(out).resolve();m=read(out/'manifest.json');normal=out/'jobs'/(placement+'-private-normal');execution=read(normal/'execution.json')
    require(execution['status']=='completed','completed actual private normal output required')
    result=read(normal/'result.json');r=result['request'];rows={x['id']:x for x in result['results']};records=[]
    for mode in ('replay','capture'):
        new=dict(r,mode=mode,requests=[dict(x,output_token_ids=rows[x['id']]['output_token_ids']) for x in r['requests']])
        target=out/'requests'/(placement+'-'+mode+'.json');write(target,new)
        record=dict(job_id=placement+'-'+mode,binding='private_consumer_dll',request=receipt(target),normal_reference=receipt(normal/'result.json'))
        write(out/'pair-plans'/(placement+'-'+mode+'.json'),record);records.append(record)
    return records
def verify_native(n,r,pid):
    require(n['schema']=='real-moe-tap-native-4' and n['request']==r and n['process_id']==pid,'native identity/request differs')
    require(n['source_sha256']==r['source_sha256'] and n['model_sha256_verified_by_parent']==r['model_sha256'],'native source/model binding differs')
    require(n['scheduler_callback_enabled'] is False and n['trace_callback_enabled'] is False,'scheduler callback unexpectedly enabled')
    require(n['cpu_consumer_tap_enabled']==(r['mode']=='capture'),'consumer tap state differs')
    require(n['observed_model_contract']==r['model_contract'] and n['threadpool_policy']=='explicit_persistent_owned','model/threadpool contract differs')
    require(n['process_affinity']['observed_mask']==r['expected_process_mask'],'native affinity differs')
    require([x['id'] for x in n['results']]==[x['id'] for x in r['requests']],'native request inventory differs')
    for row,source in zip(n['results'],r['requests']):
        require(row['prompt_token_ids']==source['prompt_token_ids'],'actual prompt stream differs')
        require(row['output_count']==len(row['sampled_greedy_token_ids'])==len(row['output_token_ids']),'sampled/output inventory differs')
        require(row['decode_forward_count']==row['output_count']-1,'extra/missing last forward')
        require(len(row['logit_observations'])==row['output_count'] and len(row['graph_observations'])==row['forward_count'],'per-step logit/graph inventory missing')
        require(row['timing_usable_as_bare_reference'] is False,'diagnostic observer timing promoted to bare')
        if 'output_token_ids' in source:require(row['output_token_ids']==source['output_token_ids'],'fixed token stream differs')
        if r['mode']!='capture':require(not row['routes'],'normal/replay unexpectedly captured routes')
    return dict(passed=True,acquisition_only=True,simulator_accuracy_certified=False,requests=len(n['results']))
def collect(out,job_id,paired=False):
    out=Path(out).resolve();m=read(out/'manifest.json')
    spec=read(out/'pair-plans'/(job_id+'.json')) if paired else next(j for j in m['jobs'] if j['job_id']==job_id)
    r=read(spec['request']['path']);binding=m['bindings'][spec['binding']];folder=out/'jobs'/job_id;require(not folder.exists(),'no retry/overwrite')
    folder.mkdir(parents=True);write(folder/'request.json',r);command=[binding['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-tap-diagnostic-execution-4',job_id=job_id,status='failed',request=receipt(folder/'request.json'),command=command,binding=spec['binding'],
        capacity_policy=m['capacity_policy'],external_observer_active=True,observer_interval_ms=200,observer_used_for_prediction=False,performance_parameters_fitted=False)
    proc=None;handles=[];low=0;samples=[]
    with common.measurement_lock():
        try:
            for resource in [binding['binary'],*binding['runtime'],*sum(([s['source'],s['archived']] for s in m['sources']),[])]:
                require(sha(resource['path'])==resource['sha256'],'frozen source/runtime changed: '+resource['path'])
            require(file_sha(m['model']['path'])==m['model']['sha256'],'model content changed')
            record['available_before_launch_bytes']=available();require(record['available_before_launch_bytes']>=m['parent_preflight_bytes'],'fixed parent capacity budget not met')
            proc,a,b,affinity=common.launch(command,child_environment(binding['runtime_dirs'],r),folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b]
            record.update(process_id=proc.pid,process_affinity=affinity);process=psutil.Process(proc.pid);start=time.monotonic()
            write(folder/'STARTED.json',dict(process_id=proc.pid,job_id=job_id,request=record['request'],started_at_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(started=job_id,process_id=proc.pid)),flush=True)
            with (folder/'observer.jsonl').open('x',encoding='utf-8',newline='\n') as f:
                while proc.poll() is None:
                    sample=dict(observer_tick_ns=time.perf_counter_ns(),global_available_bytes=available())
                    try:sample['process_memory']=process.memory_info()._asdict()
                    except (psutil.NoSuchProcess,psutil.AccessDenied) as error:sample['process_memory_error']=type(error).__name__
                    samples.append(sample);f.write(json.dumps(sample,separators=(',',':'))+'\n');f.flush()
                    low=low+1 if sample['global_available_bytes']<512*1024**2 else 0
                    if sample.get('process_memory',{}).get('peak_wset',0)>m['resident_peak_limit_bytes'] or low>=2 or time.monotonic()-start>300:
                        record['abort_reason']='resident/available watchdog or fixed five-minute timeout';proc.terminate();break
                    try:proc.wait(timeout=.2)
                    except subprocess.TimeoutExpired:pass
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            record.update(returncode=proc.returncode,process_exited_verified=True,forced_termination='abort_reason' in record)
            require(proc.returncode==0,'native failed; evidence retained without retry');n=read(folder/'result.json')
            record['verification']=verify_native(n,r,proc.pid);record.update(status='completed',native=receipt(folder/'result.json'))
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
    summary=dict(status=record['status'],execution=receipt(folder/'execution.json'),observed_global_min_available_bytes=min((x['global_available_bytes'] for x in samples),default=None),
        peak_wset_bytes=max((x.get('process_memory',{}).get('peak_wset',0) for x in samples),default=None),observer_input_to_predictor=False)
    write(folder/'OBSERVER_SUMMARY.json',summary);print(json.dumps(dict(job_id=job_id,status=record['status'],error=record.get('error'))),flush=True);return record
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','collect','make-pairs'));p.add_argument('--out',required=True);p.add_argument('--build');p.add_argument('--legacy-collection');p.add_argument('--job-id');p.add_argument('--paired',action='store_true');p.add_argument('--placement',choices=('cpu','hybrid'));a=p.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.build,a.legacy_collection)
    elif a.command=='make-pairs':r=make_pairs(a.out,a.placement)
    else:r=collect(a.out,a.job_id,a.paired)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
