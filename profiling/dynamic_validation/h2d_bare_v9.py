"""Bare GGML size/history experiment, with profile-only fit and a freeze barrier.

v7's primitive observers failed neutrality. This new cohort times only the
original completed Executor.action and never imports those phase prices.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from . import idle_campaign as base
from llm_infer_model.performance.reusable_action_costs import digest, semantic_action

ROOT=base.ROOT
read,write,sha,receipt,require=base.read,base.write,base.sha,base.receipt,base.require
SIZES={'profile':(131072,524288,2097152),'heldout':(262144,1048576)}
IDLES={'profile':(0,1,5,20),'heldout':(2,10)}
BOUNDARY='host_response_including_copy_api_and_gpu_cpu_completion'


def orders(width,round_id):
    require(type(width) is int and width in (2,3,4) and type(round_id) is int and 0<=round_id<3,'declared balanced order required')
    rows=[[(t+round_id+s)%width for s in range(width)] for t in range(12)]
    return [list(reversed(r)) for r in rows] if round_id==1 else rows


def protocol():
    return dict(schema='dynamic-h2d-bare-protocol-9',profile_processes=6,heldout_processes=6,
        fresh_processes_per_memory_per_split=3,trials=12,copies_per_burst=16,
        memory_modes=['registered','pageable'],sizes_bytes=SIZES,idle_ms=IDLES,
        response_boundary=BOUNDARY,target='completed_original_Executor.action_COPY',
        CUPTI_initialized=False,CUDA_events_created=False,NVML_initialized=False,
        complete_original_prime_before_every_burst=True,prime_implies_zero_state=False,
        calibration='individual completed COPY responses; equal process/size/idle/ordinal weights',
        calibration_uses_full_LLM_target=False,raw_trials_removed=0,retries=0,
        providers=['static_affine','zero_initial_recovery','prepared_initial_recovery'],
        response_law='modeling.transfer_recovery; profiling only estimates named parameters',
        parameters=['baseline fixed/byte slope','prepared fixed/byte slope','remaining capacity fixed/byte slope','idle fill tau','active recovery tau'],
        tau_defaults=False,whole_target_residual_coefficient=False,
        profile_support=dict(early_late_zero_idle_suffix_relative_limit=.10,
                             fresh_process_zero_idle_suffix_range_relative_limit=.10),
        parameter_support=dict(full_local_jacobian_rank=True,scaled_jacobian_condition_max=1e6,
                               equivalent_multistart_parameter_relative_spread_max=.25,
                               tau_bound_fraction_min=.001),
        independent_holdout=dict(new_sizes=True,new_idles=True,new_mathematical_seed=True,
                                 predictions_frozen_before_collection=True),
        acceptance=dict(first_copy_mean_APE_max=.20,burst_mean_APE_max=.20,
                        raw_measured_population_CV_max=.10,raw_CV_separate_from_prediction_accuracy=True,
                        all_declared_denominators_retained=True),
        failed_profile_stops_qualification=True,unidentified_parameters_are_unknown=True,
        within_burst_physical_device_state_identified=False)


def plans(programs,source_sha):
    jobs=[]
    # Counterbalance memory acquisition order by fresh-process round.
    for split in SIZES:
        for round_id in range(3):
            for memory in (('registered','pageable') if round_id%2==0 else ('pageable','registered')):
                sampling=base.e2a.request('action',trials=12,target=programs[split]['target_action_index']);sampling['repeats']=16
                request=dict(schema='dynamic-h2d-bare-request-9',source_sha256=source_sha,split=split,sampling=sampling,
                    memory_mode=memory,expected_process_mask=4294967295,sizes_bytes=list(SIZES[split]),idle_ms=list(IDLES[split]),
                    size_orders=orders(len(SIZES[split]),round_id),condition_orders=orders(len(IDLES[split]),round_id),
                    performance_boundary='completed_original_Executor.action_COPY',telemetry_between_bursts=False)
                jobs.append(dict(id=f'{split}-{memory}-r{round_id}',split=split,memory_mode=memory,round=round_id,request=request))
    return jobs


def scope_contracts(manifest):
    program=read(manifest['programs']['profile']['program']['path'])
    preparation=dict(actions=[semantic_action(program,i) for i in range(len(program['actions']))],
        lifecycle='reset then completed full original graph before every burst',warmup_iterations=20,warmup_min_ms=500,
        additional_telemetry=False,events_created=False,registration='owned arena registered once before warmup, or observed pageable')
    # Semantic descriptors exclude model/program/tensor names and mathematical seed.
    other=read(manifest['programs']['heldout']['program']['path'])
    require(preparation['actions']==[semantic_action(other,i) for i in range(len(other['actions']))],'heldout semantic preparation differs')
    context=dict(runtime=manifest['runtime'],hardware=manifest['hardware'],binary=manifest['binary'],
        process_mask=4294967295,main_thread_mask=4,threads=1,boundary=BOUNDARY,
        preload_initialization='same retained Executor; graphs disabled; no telemetry/events lifecycle')
    return dict(context_sha256=digest(context),preparation_sha256=digest(preparation),context=context,preparation=preparation)


def prepare(out,binary,*,build_receipt):
    out,binary=Path(out).resolve(),Path(binary).resolve();require(not out.exists() and binary.is_file(),'new directory/existing compiled binary required')
    build_receipt=Path(build_receipt).resolve();build=read(build_receipt)
    require(build.get('binary',{}).get('sha256')==sha(binary),'explicit build receipt must bind compiled binary')
    require(build.get('source_sha256')==sha(Path(__file__).parent/'native/v9/h2d_bare.cpp'),'build receipt/source differs')
    parent_path=ROOT/'build/dynamic-validation-20261026/idle-v2b/manifest.json';parent=read(parent_path)
    programs={split:copy.deepcopy(parent['programs'][split]) for split in SIZES}
    require(programs['profile']['program_sha256']!=programs['heldout']['program_sha256'],'independent heldout seed required')
    for b in programs.values():
        for key in ('program','reference'):require(sha(b[key]['path'])==b[key]['sha256'],'retained input changed')
    here=Path(__file__).parent
    sources=[Path(__file__),here/'h2d_bare_v9_fit.py',here/'native/v9/h2d_bare.cpp',here/'native/v9/CMakeLists.txt',
        here/'native/v3/common.hpp',here/'native/owned_timer.hpp',here/'idle_campaign.py',
        ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
        ROOT/'profiling/native/physical_stages/cupti_capture.hpp',ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/dac_validation/e2a.py',
        ROOT/'modeling/llm_infer_model/performance/transfer_recovery.py',ROOT/'modeling/llm_infer_model/performance/idle_recovery.py',ROOT/'modeling/llm_infer_model/performance/transfer_response.py',
        ROOT/'modeling/llm_infer_model/performance/prepared_transfer.py',ROOT/'modeling/llm_infer_model/performance/parameters.py']
    sources+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    require(all(p.is_file() for p in sources),'complete software required before freeze')
    out.mkdir(parents=True);archived=[]
    for p in sources:
        dst=out/'executed_sources'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);archived.append(dict(source=receipt(p),archived=receipt(dst)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    m=dict(schema='dynamic-h2d-bare-campaign-9',prepared_at_utc=base.e2a.utc(),parent_manifest=receipt(parent_path),
        binary=receipt(target),build_receipt=receipt(build_receipt),sources=archived,runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],hardware=parent['hardware'],
        programs=programs,jobs=plans(programs,sha(here/'native/v9/h2d_bare.cpp')),protocol=protocol())
    m['scope_contracts']=scope_contracts(m);write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),profile_processes=6,heldout_processes=6,hardware_executed=False)


def validate_native(native,request,binding,pid):
    require(native['schema']=='dynamic-h2d-bare-native-9' and native['request']==request and native['process_id']==pid,'native request/process differs')
    require(native['source_sha256']==request['source_sha256'] and native['program_sha256']==binding['program_sha256'],'source/program differs')
    for key in ('CUPTI_initialized','CUDA_events_created','NVML_initialized','telemetry_between_bursts','host_API_response_is_CPU_active','performance_parameters_fitted'):
        require(native[key] is False,'boundary broadened '+key)
    field='registered_host' if request['memory_mode']=='registered' else 'unregistered_pageable'
    reg=native['arena_registration'];require(reg['mode']==request['memory_mode'] and reg['during'][field] and reg['restored'] and native['source_pointer_kind'][field],'actual memory category differs')
    require(native['source_buffer_host'] and not native['target_buffer_host'],'host/device buffers differ')
    require(native['process_affinity']['observed_mask']==native['process_affinity']['final_mask']==4294967295,'process affinity changed')
    require(native['main_thread_affinity']['observed_mask']==native['main_thread_affinity']['final_mask']==4,'target affinity changed')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'] and native['full_payload']['full_byte_equality'] and native['timed_payload_checked_before_final_reset'],'native numerical oracle failed')
    oracles=native['slice_copy_oracles'];require({o['bytes'] for o in oracles}==set(request['sizes_bytes']) and len(oracles)==len(request['sizes_bytes']),'range inventory differs')
    require(all(o['selected_range_initially_different'] and o['selected_bytes_equal_after_copy'] and o['outside_view_unchanged'] for o in oracles),'vacuous/failed range check')
    rows=native['bursts'];conditions=len(request['idle_ms']);width=len(request['sizes_bytes'])
    require(len(rows)==12*width*conditions,'burst inventory differs');hz=native['clock']['frequency_hz'];require(type(hz) is int and hz>0,'QPC unavailable')
    previous=0
    for index,b in enumerate(rows):
        trial,rem=divmod(index,width*conditions);slot,cs=divmod(rem,conditions);si=request['size_orders'][trial][slot];ci=request['condition_orders'][trial][cs]
        require((b['trial'],b['size_slot'],b['condition_slot'],b['size_index'],b['condition_index'],b['bytes'],b['requested_idle_ms'])==(trial,slot,cs,si,ci,request['sizes_bytes'][si],request['idle_ms'][ci]),'balanced acquisition differs')
        require(previous<=b['prime_begin_tick']<=b['prime_end_tick']<=b['idle_begin_tick']<=b['idle_end_tick']<=b['burst_begin_tick']<b['burst_end_tick'],'invalid prime/idle/burst clock');previous=b['burst_end_tick']
        for start,end,key in [('idle_begin_tick','idle_end_tick','actual_idle_ns'),('burst_begin_tick','burst_end_tick','burst_wall_ns')]:require(abs((b[end]-b[start])*1e9/hz-b[key])<=1,'QPC duration differs')
        require(len(b['copies'])==16,'copy inventory differs');last=b['burst_begin_tick']
        for ordinal,c in enumerate(b['copies']):
            require(c['ordinal']==ordinal and last<=c['begin_tick']<c['end_tick']<=b['burst_end_tick'],'copy order differs');last=c['end_tick']
            require(abs((c['end_tick']-c['begin_tick'])*1e9/hz-c['wall_ns'])<=1,'copy duration differs')
        require(abs((b['copies'][0]['begin_tick']-b['prime_end_tick'])*1e9/hz-b['effective_prime_to_first_copy_ns'])<=1,'preparation gap differs')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=native['verification_native']['ports'];require(set(oracle)==set(ports),'port inventory differs')
    for name in oracle:
        a,b=np.asarray(oracle[name]),np.asarray(ports[name]);require(a.shape==b.shape and np.all(np.isfinite(b)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent numerical mismatch '+name)
    return dict(passed=True,bursts=len(rows),copies=len(rows)*16,numerical_ports=True,range_oracles=True)


def authorize(out,split):
    require(split in SIZES,'unknown split');m=read(out/'manifest.json');a=read(out/f'{split}-authorization.json')
    require(a.get('collection_authorized') is True and a.get('manifest_sha256')==sha(out/'manifest.json') and a.get('split')==split,'root authorization for exact split/manifest required')
    if split=='heldout':
        barrier=read(out/'frozen-barrier.json');require(a.get('frozen_barrier_sha256')==sha(out/'frozen-barrier.json'),'authorization not bound to freeze')
        require(barrier['manifest_sha256']==sha(out/'manifest.json') and barrier['profile_gate_passed'] and barrier['heldout_predictions_frozen'],'supported profile/prediction freeze required')
        for name,value in barrier['files'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(out/p)==value,'frozen file changed')
    return m


def collect(out,split):
    out=Path(out).resolve();m=authorize(out,split)
    for s in m['sources']:
        for key in ('source','archived'):require(sha(s[key]['path'])==s[key]['sha256'],'bound source changed')
    for resource in [m['binary'],m['build_receipt'],*m['runtime']]:require(sha(resource['path'])==resource['sha256'],'bound runtime/binary/build receipt changed')
    for b in m['programs'].values():
        for key in ('program','reference'):require(sha(b[key]['path'])==b[key]['sha256'],'bound input changed')
    require(not (out/f'{split}-started.json').exists(),'no retries/repeated split');write(out/f'{split}-started.json',dict(started_at_utc=base.e2a.utc(),split=split))
    executions=[]
    with base.run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request']);binding=m['programs'][split]
            command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],split=split,status='failed',command=command,declared_whole_split_jobs=6,performance_parameters_fitted=False);process=None;handles=[]
            try:
                env=base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,stdout,stderr,affinity=base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[stdout,stderr];record.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=240);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                record['verification']=validate_native(read(folder/'result.json'),job['request'],binding,process.pid);record.update(status='completed',native=receipt(folder/'result.json'))
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
    write(out/f'{split}-executions.json',executions)
    return dict(declared=6,acquired=len(executions),completed=sum(r['status']=='completed' for r in executions),not_run=6-len(executions))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect'));p.add_argument('--out');p.add_argument('--binary');p.add_argument('--build-receipt');p.add_argument('--split',choices=tuple(SIZES));a=p.parse_args()
    result=protocol() if a.command=='protocol' else prepare(a.out,a.binary,build_receipt=a.build_receipt) if a.command=='prepare' else collect(a.out,a.split)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
