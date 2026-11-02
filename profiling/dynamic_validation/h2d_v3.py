"""Matched bare/CUPTI diagnosis; no fitting and no heldout performance claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import statistics

from profiling.dynamic_validation import idle_campaign as base
from profiling.state_calibration.telemetry import DeviceTelemetrySampler

read,write,sha,receipt,require=base.read,base.write,base.sha,base.receipt,base.require
ROOT=base.ROOT


def validate_stages(s,hz):
    require(hz>0,'positive QPC frequency required')
    fields=('begin_tick','copy_API_enter_tick','copy_API_return_tick','extra_GPU_sync_enter_tick','extra_GPU_sync_return_tick','CPU_sync_enter_tick','CPU_sync_return_tick','end_tick')
    ticks=[s[k] for k in fields];require(all(a<=b for a,b in zip(ticks,ticks[1:])),'host phase order differs')
    expected=(ticks[-1]-ticks[0])*1e9/hz
    require(abs(expected-s['host_wall_ns'])<=1,'host wall clock differs')
    summed=s['copy_API_inclusive_ns']+s['extra_GPU_sync_ns']+s['CPU_sync_ns']+s['QPC_wrapper_gaps_ns']
    require(abs(summed-s['host_wall_ns'])<=4,'host wall partition does not close')
    require(s['device_span_added_to_host_wall'] is False,'device/host overlap was added twice')


def prepare(out,binary,*,cupti_dll,rounds=1,telemetry_ms=50.,init_policy='both'):
    out,binary=Path(out).resolve(),Path(binary).resolve();require(not out.exists(),'new v3 diagnostic output required');require(rounds in (1,3),'explicit pilot1 or diagnostic3 rounds required')
    require(cupti_dll is not None,'explicit installed CUPTI DLL path required');cupti_dll=Path(cupti_dll).resolve();require(cupti_dll.is_file() and cupti_dll.name.lower().startswith('cupti64_') and cupti_dll.suffix.lower()=='.dll','explicit installed CUPTI DLL required')
    require(init_policy in ('pinned_init','wide_init','both'),'explicit initialization-affinity condition required')
    require(telemetry_ms>0,'positive telemetry cadence required')
    parent_path=ROOT/'build/dynamic-validation-20261026/idle-v2b/manifest.json';parent=read(parent_path)
    out.mkdir(parents=True);binding=parent['programs']['profile']
    for name in ('program','reference'):require(sha(binding[name]['path'])==binding[name]['sha256'],'retained mathematical input changed')
    paths=[Path(__file__),ROOT/'profiling/dynamic_validation/native/v3/h2d_probe.cpp',ROOT/'profiling/dynamic_validation/native/v3/common.hpp',
        ROOT/'profiling/dynamic_validation/native/v3/CMakeLists.txt',ROOT/'profiling/dynamic_validation/native/owned_timer.hpp',
        ROOT/'profiling/native/physical_stages/cupti_capture.hpp',ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
        ROOT/'profiling/state_calibration/telemetry.py',ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/dynamic_validation/idle_campaign.py',ROOT/'profiling/dac_validation/e2a.py']
    paths.append(ROOT/'profiling/dynamic_validation/native/v3/thread_inventory.hpp')
    paths+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'));sources=[]
    for path in paths:
        archive=out/'executed_sources'/path.resolve().relative_to(ROOT);archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archive)
        sources.append(dict(source=receipt(path),archived=receipt(archive)))
    archive=out/'executed_binary'/binary.name;archive.parent.mkdir();shutil.copy2(binary,archive)
    jobs=[]
    for r in range(rounds):
        blocks=(('registered','pinned_init',('bare','stages')),('registered','wide_init',('stages','bare')),
            ('pageable','wide_init',('bare','stages')),('pageable','pinned_init',('stages','bare')))
        for memory,policy,instruments in (blocks if r%2==0 else tuple(reversed(blocks))):
            if init_policy!='both' and policy!=init_policy:continue
            for instrument in instruments:
                sampling=base.e2a.request('action',trials=16,target=binding['target_action_index']);sampling['repeats']=16
                request=dict(schema='dynamic-h2d-stage-request-3',source_sha256=sha(paths[1]),sampling=sampling,expected_process_mask=4294967295,
                    memory_mode=memory,instrumentation=instrument,idle_ms=[0,1,5,20],orders=base.orders(r),focus_idle_ms=[0,20],init_policy=policy)
                jobs.append(dict(id=f'diagnostic-r{r}-{memory}-{policy}-{instrument}',round=r,memory_mode=memory,init_policy=policy,instrumentation=instrument,request=request))
    added_runtime=[receipt(p) for p in sorted(cupti_dll.parent.glob('*.dll'))]
    manifest=dict(schema='dynamic-h2d-stage-campaign-3',prepared_at_utc=base.e2a.utc(),parent_manifest=receipt(parent_path),binary=receipt(archive),sources=sources,
        runtime=[*parent['runtime'],*added_runtime],runtime_dirs=[*parent['runtime_dirs'],str(cupti_dll.parent)],hardware=parent['hardware'],program=binding,jobs=jobs,
        protocol=dict(rounds=rounds,declared_processes=len(jobs),timeout_seconds=180,retries=0,trials=16,conditions_per_trial=4,copies_per_burst=16,
            conditioning_history='retained0/1/5/20ms16x4 full-prime history; physical diagnostics focus0/20ms',telemetry_interval_ms=telemetry_ms,
            telemetry_role='diagnosis only; target-inclusive future utilization is not a forecast input',observer_process_discrepancy_limit=.05,
            init_policy=init_policy,initialization_extent='backend construction, pool attach and first full validation; target thread pinned before common warmup',
            observer_order='registered pinned bare/stages; registered wide stages/bare; pageable wide bare/stages; pageable pinned stages/bare',
            host_closure_tolerance_ns=4,device_times_added_to_host_times=False,performance_parameters_fitted=False))
    write(out/'manifest.json',manifest);return dict(out=str(out),declared=len(jobs),hardware_executed=False)


def verify(native,job,binding,pid):
    require(native['schema']=='dynamic-h2d-stage-native-3' and native['request']==job['request'],'native request/schema differs')
    require(native['process_id']==pid and native['source_sha256']==job['request']['source_sha256'],'native PID/source differs')
    require(native['program_sha256']==binding['program_sha256'] and native['target_bytes']==2097152,'retained H2D semantics differ')
    require(native['init_policy']==job['init_policy'] and native['initialization_affinity']['observed_mask']==(4 if job['init_policy']=='pinned_init' else 4294967295),'initialization affinity differs')
    require(native['main_thread_affinity']['observed_mask']==native['main_thread_affinity']['final_mask']==4 and native['main_thread_affinity']['restored'],'owned target thread mask differs')
    for value in (native['payload_before'],native['payload_after']):require(value['full_byte_equality'] and value['bytes']==2097152 and value['source_sha256']==value['target_sha256'],'copy-byte validation failed')
    require(native['payload_before']['source_sha256']==native['payload_after']['source_sha256'],'source changed')
    mode=job['memory_mode'];registration=native['arena_registration'];require(registration['during']['registered_host']==(mode=='registered') and registration['restored'],'actual host memory category/restore differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'full native numerical check failed')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=native['verification_native']['ports'];require(set(oracle)==set(ports),'numerical port inventory differs')
    for key,value in oracle.items():
        actual,expected=np.asarray(ports[key]),np.asarray(value);require(actual.shape==expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual,expected,rtol=5e-5,atol=5e-5),'independent port mismatch '+key)
    hz=native['clock']['frequency_hz'];copies=0;phases=0;last=0
    for sequence,row in enumerate(native['sequences']):
        require(row['trial']==sequence//4 and row['condition_slot']==sequence%4 and len(row['copies'])==16,'retained trial/condition count differs')
        require(last<=row['reset_begin_tick']<=row['reset_end_tick']<=row['prime_begin_tick']<=row['prime_end_tick']<=row['idle_begin_tick']<=row['idle_end_tick']<=row['burst_begin_tick'],'condition history clock differs')
        require(row['prior_state']['query_end_tick']<=row['reset_begin_tick'],'prior snapshot used a future target observation')
        require(row['before_burst_state']['query_end_tick']<=row['burst_begin_tick'] and row['after_burst_state']['query_begin_tick']>=row['burst_end_tick'],'burst snapshot boundary differs')
        prev=row['idle_end_tick']
        for ordinal,c in enumerate(row['copies']):
            require(c['ordinal']==ordinal and prev<=c['begin_tick']<=c['end_tick']<=row['burst_end_tick'],'copy interval order differs')
            require(abs((c['end_tick']-c['begin_tick'])*1e9/hz-c['wall_ns'])<=1,'copy QPC ledger differs')
            if c['host_stages'] is not None:validate_stages(c['host_stages'],hz);phases+=1
            prev=c['end_tick'];copies+=1
        last=row['burst_end_tick']
    require(len(native['sequences'])==64 and copies==1024,'whole conditioning sequence incomplete')
    if job['instrumentation']=='stages':require(phases==512 and native['cupti']['all_requested_records_retained'],'focused stage records or CUPTI integrity failed')
    else:require(phases==0 and native['cupti']['enabled'] is False,'bare observer unexpectedly enabled')
    return dict(passed=True,copies=copies,focused_stage_partitions=phases,numerical_ports=True,stage_times_are_new_performance_prices=False)


def collect(out,*,max_jobs=None):
    require(max_jobs is None or type(max_jobs) is int and max_jobs>0,'positive diagnostic process limit required')
    out=Path(out).resolve();m=read(out/'manifest.json');base.check(m);marker=out/'diagnostic-started.json'
    if not marker.exists():write(marker,dict(started_at_utc=base.e2a.utc(),model_fitting=False))
    acquired=0
    with base.run.measurement_lock():
        for job in m['jobs']:
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists():continue
            if max_jobs is not None and acquired>=max_jobs:break
            folder.mkdir(parents=True,exist_ok=False);request=folder/'request.json';write(request,job['request']);output=folder/'result.json'
            command=[m['binary']['path'],m['program']['program']['path'],str(request),str(output)]
            e=dict(job=job,status='failed',started_at_utc=base.e2a.utc(),command=command);process=None;handles=[];sampler=None
            try:
                sampler=DeviceTelemetrySampler(interval_ms=m['protocol']['telemetry_interval_ms']);sampler.start()
                env=base.run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
                process,stdout,stderr,affinity=base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[stdout,stderr]
                e.update(process_id=process.pid,process_affinity=affinity);code=process.wait(timeout=m['protocol']['timeout_seconds']);e.update(returncode=code,process_exited_verified=True)
                require(code==0,'owned native diagnostic failed; no retry or substitution')
                e['verification']=verify(read(output),job,m['program'],process.pid);e.update(status='completed',native=receipt(output))
            except BaseException as error:
                e['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:e.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                if sampler is not None:
                    try:sampler.stop();e['telemetry']=sampler.write_jsonl(folder/'telemetry.jsonl');e['telemetry_metadata']=sampler.metadata()
                    except BaseException as error:e['telemetry_error']=repr(error)
                e['completed_at_utc']=base.e2a.utc();write(folder/'execution.json',e)
            acquired+=1;print(json.dumps(dict(job=job['id'],status=e['status'],error=e.get('error'))),flush=True)
    entries=[read(out/'jobs'/j['id']/'execution.json') for j in m['jobs'] if (out/'jobs'/j['id']/'execution.json').exists()]
    target=out/('executions.json' if len(entries)==len(m['jobs']) else f'partial-executions-{len(entries)}.json')
    if not target.exists():write(target,entries)
    return dict(declared=len(m['jobs']),acquired=len(entries),completed=sum(e['status']=='completed' for e in entries),pending=len(m['jobs'])-len(entries))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','collect'));p.add_argument('--out',required=True);p.add_argument('--binary')
    p.add_argument('--rounds',type=int,choices=(1,3),default=1);p.add_argument('--telemetry-ms',type=float,default=50);p.add_argument('--max-jobs',type=int);p.add_argument('--cupti-dll');p.add_argument('--init-policy',choices=('pinned_init','wide_init','both'),default='both');a=p.parse_args()
    result=prepare(a.out,a.binary,cupti_dll=a.cupti_dll,rounds=a.rounds,telemetry_ms=a.telemetry_ms,init_policy=a.init_policy) if a.command=='prepare' else collect(a.out,max_jobs=a.max_jobs)
    print(json.dumps(result,ensure_ascii=False),flush=True)
