"""Write-once steady costs: collect PROFILE, freeze, then fresh heldout.

Only declared zero-owned-background-load scenarios are used here. Existing
applications are not stopped or assumed absent; no target-derived utilization
or latency supplies a prediction. All measured trials are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .contracts import component_reference,digest,representative_components,require,semantic_program,sha,validate_component,verify_float_output
from profiling import host

ROOT=Path(__file__).resolve().parents[2]


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def pack(x):
    if isinstance(x,Fraction):return dict(numerator=x.numerator,denominator=x.denominator)
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x
def fraction(x):return Fraction(x['numerator'],x['denominator']) if isinstance(x,dict) else Fraction(x)
def mean(values):
    values=list(map(Fraction,values))
    return None if not values else sum(values,Fraction())/len(values)
def stats(values):
    values=list(map(Fraction,values))
    if not values:return dict(n=0,mean=None,median=None,p90=None,cv=None)
    ordered=sorted(values);average=mean(values)
    return pack(dict(n=len(values),mean=average,median=statistics.median(values),p90=ordered[(9*len(values)+9)//10-1],
        cv=None if average<=0 or len(values)<2 else statistics.stdev(map(float,values))/float(average)))


def make_request(mode,*,target=None,split='profile',component=False,instrumentation='bare'):
    return dict(schema='steady-cost-request-1',mode=mode,target_action_index=target,warmup=20,
        warmup_min_ms=0 if mode=='sync-idle' else 500,trials=12 if split=='heldout' else 8,
        repeats=16 if component else 8,poll='0',instrumentation=instrumentation,main_thread_mask=4)


def prepare(out,binary,runtime_dirs,*,seed=202610220):
    from profiling.highlevel_accuracy.cases import build_case
    from profiling.highlevel_accuracy.bridge import export_plan,reference_ports
    from profiling.highlevel_accuracy.run import case_specs
    from profiling.state_response.collect import hardware_identity
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    binary=Path(binary).resolve();require(binary.is_file(),'explicit existing native binary required')
    build_path=binary.parent/'build-manifest.json';build=read(build_path)
    require(build.get('returncode')==0 and build['binary_sha256']==sha(binary),'native build receipt invalid')
    sources={str(binary):sha(binary),str(build_path):sha(build_path)}
    for name,expected in build['bound_files'].items():require(sha(name)==expected,'native source changed: '+name);sources[name]=expected
    programs={};cases=[];jobs=[]
    legacy=deepcopy(next(r['kwargs'] for r in case_specs('representative') if r['id']=='h128-q16-history0-attention-ffn-cut128'))
    legacy['threads']=1
    specifications=[('layer-h128-q16',legacy)]
    for tokens in (64,128):
        spec=deepcopy(legacy);spec.update(hidden=512,intermediate=1024,heads=8,kv_heads=2,head_dim=64,tokens=tokens,history=0,ffn_cut=512)
        specifications.append((f'layer-h512-q{tokens}',spec))
    for number,(case_id,spec) in enumerate(specifications):
        compiled=build_case(out/'inputs'/case_id,**spec).compile()
        for source in compiled._compiled.sources:sources[str(Path(source['path']).resolve())]=source['sha256']
        for split,offset in (('profile',0),('heldout',10000)):
            program=export_plan(compiled,seed=seed+number+offset)
            path=out/'programs'/f'{case_id}-{split}.json';oracle=out/'references'/f'{case_id}-{split}.json'
            write_new(path,program);write_new(oracle,reference_ports(program))
            programs[case_id+'-'+split]=dict(program=receipt(path),oracle=receipt(oracle),seed=program['seed'],
                signature=digest(semantic_program(program)),kind='layer')
        cases.append(dict(id=case_id,kind='layer'))
        for repeat in range(3):
            for split,mode in (('profile','full-stamped'),('observer','full'),('heldout','full')):
                key=case_id+('-heldout' if split=='heldout' else '-profile')
                jobs.append(dict(id=f'{split}-{case_id}-{repeat}',case_id=case_id,split=split,program_key=key,
                    repeat=repeat,request=make_request(mode,split=split),role='context_actions' if split=='profile' else 'whole_sequence'))
        if number==2:
            for repeat in range(3):jobs.append(dict(id=f'diagnostic-pv-{repeat}',case_id=case_id,split='diagnostic',
                program_key=case_id+'-profile',repeat=repeat,request=make_request('action',target=14),role='isolated_pv'))
    for number,case in enumerate(representative_components()):
        case_id=case['id'];cases.append(dict(id=case_id,kind='component'))
        for split,offset in (('profile',100),('heldout',10100)):
            program=dict(case['program'],seed=seed+number+offset)
            program['program_sha256']=digest(program)
            validate_component(program)
            path=out/'programs'/f'{case_id}-{split}.json';oracle=out/'references'/f'{case_id}-{split}.json'
            write_new(path,program);write_new(oracle,component_reference(program))
            programs[case_id+'-'+split]=dict(program=receipt(path),oracle=receipt(oracle),seed=program['seed'],
                signature=digest(semantic_program(program)),kind='component')
        for repeat in range(3):
            for split in ('profile','heldout'):
                jobs.append(dict(id=f'{split}-{case_id}-{repeat}',case_id=case_id,split=split,program_key=case_id+'-'+split,
                    repeat=repeat,request=make_request('action',split=split,component=True),role='complete_component'))
        if case_id in {'gpu-prefill4096-q64','gpu-pv4096-q16-kv2048'}:
            for repeat in range(3):
                for instrumentation in ('bare','events'):
                    jobs.append(dict(id=f'diagnostic-{case_id}-{repeat}-{instrumentation}',case_id=case_id,split='diagnostic',
                        program_key=case_id+'-profile',repeat=repeat,request=make_request('async-graph',component=True,instrumentation=instrumentation),
                        role='async_submission_and_completion'))
        if case_id=='gpu-decode4096':
            for repeat in range(3):jobs.append(dict(id=f'diagnostic-idle-sync-{repeat}',case_id=case_id,split='diagnostic',
                program_key=case_id+'-profile',repeat=repeat,request=make_request('sync-idle',component=True),role='idle_sync'))
    for binding in programs.values():
        for field in ('program','oracle'):sources[binding[field]['path']]=binding[field]['sha256']
    for folder in (Path(__file__).parent,ROOT/'profiling/highlevel_accuracy'):
        for path in folder.glob('*.py'):sources[str(path.resolve())]=sha(path)
    for name in ('profiling/state_calibration/run.py','profiling/state_calibration/telemetry.py',
                 'profiling/state_response/collect.py','profiling/state_response/diagnose_cpu.py',
                 'src/llm_simulator/steady_costs.py'):
        path=ROOT/name;sources[str(path.resolve())]=sha(path)
    for path in (Path(__file__).parent/'tests').glob('*.py'):sources[str(path.resolve())]=sha(path)
    for name in ('steady_action_costs.py','causal_inputs.py','graph.py','parameters.py','evaluator.py','__init__.py'):
        path=ROOT/'modeling/llm_infer_model/performance'/name;sources[str(path.resolve())]=sha(path)
    for folder in runtime_dirs:
        for path in Path(folder).glob('*.dll'):sources[str(path.resolve())]=sha(path)
    environment_flags={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','GGML_','CUBLAS_','OMP_','MKL_','OPENBLAS_','NVIDIA_'))}
    manifest=dict(schema='steady-cost-campaign-1',prepared_at_utc=utc(),binary=str(binary),runtime_dirs=[str(Path(p).resolve()) for p in runtime_dirs],
        hardware=hardware_identity(),cases=cases,programs=programs,jobs=jobs,sources=sources,source_inventory_sha256=digest(sources),
        inherited_backend_environment=environment_flags,
        protocol=dict(main_affinity_mask=4,process_timeout_seconds=180,independent_processes_per_split=3,
            dtype='F32',cuda_graphs='disabled',owned_background_actors=[],total_device_state_used_as_external_input=False,
            target_outcome_used_for_prediction=False,forecast_model_fitted=False,all_trials_retained=True,
            baseline='context action arithmetic means; complete component arithmetic mean',
            quality=dict(max_process_mean_ape=.10,max_trial_average_p90_ape=.20,max_observer_relative_mean_difference=.05,max_process_trial_cv=.10),
            measurements_are_trial_averages_not_individual_call_tail=True,allocation_capacity_qualified=False,
            observer_pairs='same profile program full-stamped versus full; no process startup in target wall'))
    write_new(out/'manifest.json',manifest)
    return dict(out=str(out),jobs=len(jobs),native_timed_work_started=False)


def verify_manifest(root):
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    require(manifest['schema']=='steady-cost-campaign-1' and digest(manifest['sources'])==manifest['source_inventory_sha256'],'manifest source inventory differs')
    for path,expected in manifest['sources'].items():require(sha(path)==expected,'bound source changed: '+path)
    return manifest


def verify_native(program,request,native,oracle):
    require(native.get('schema')=='steady-cost-native-1','native schema differs')
    require(native.get('program_sha256')==program['program_sha256'] and native.get('request')==request,'native program/request identity differs')
    require(type(native.get('process_id')) is int and native['process_id']>0,'native process identity missing')
    clock=native.get('clock',{});hz=clock.get('frequency_hz')
    require(clock.get('clock_id')==host.measurement_clock_id() and type(hz) is int and hz>0,'native monotonic clock unknown')
    def ticks(row):
        first,last=row.get('begin_tick'),row.get('end_tick')
        require(type(first) is int and type(last) is int and 0<=first<=last,'native monotonic interval differs')
        duration=row.get('wall_ns')
        require(type(duration) is int and duration>=0 and abs(Fraction((last-first)*1_000_000_000,hz)-duration)<=1,
            'native clock interval and wall duration differ')
        return first,last
    require(native.get('collected_cpu_cycles') is False and native.get('per_action_CPU_counter_queries') is False,
        'counter-free timing contract differs')
    require(native.get('observed_runtime_flags')==dict(GGML_CUDA_DISABLE_GRAPHS='1',NVIDIA_TF32_OVERRIDE='0',matmul_precision='GGML_PREC_F32'),
        'native graph/precision flags differ')
    affinity=native.get('main_thread_affinity',{})
    require(affinity.get('schema')=='steady-owned-main-thread-affinity-1' and
        affinity.get('requested_mask')==affinity.get('observed_mask')==affinity.get('final_mask')==request['main_thread_mask'] and
        affinity.get('set_succeeded') is True and affinity.get('applied_before_backend_allocation_and_warmup') is True and
        affinity.get('restored') is True,'owned main thread placement contract differs')
    warm=native.get('warmup',{})
    require(warm.get('completed_outside_all_trial_timers') is True and warm.get('actual_iterations',-1)>=request['warmup'] and
        warm.get('wall_ns',-1)>=request['warmup_min_ms']*1_000_000,'warmup boundary/count differs')
    _,previous_end=ticks(warm)
    require(native.get('native_repeat_consistency',{}).get('passed') is True,'fixed-input outputs changed')
    rows=native.get('trials');require(isinstance(rows,list) and len(rows)==request['trials'],'native trial coverage differs')
    for index,row in enumerate(rows):
        require(row.get('trial')==index and row.get('repeats')==request['repeats'],'native trial/repeat identity differs')
        require(type(row.get('wall_ns')) is int and row['wall_ns']>=0,'native wall duration unknown')
        first,last=ticks(row);require(first>=previous_end,'trial overlaps warmup or previous trial');previous_end=last
        if request['mode']=='full-stamped':
            actions=row.get('actions');require(isinstance(actions,list) and len(actions)==len(program['actions']),'stamped action coverage differs')
            require(all(a['action_index']==i and type(a.get('wall_ns')) is int and a['wall_ns']>=0 for i,a in enumerate(actions)),'stamped action duration/order differs')
            require(sum(a['wall_ns'] for a in actions)<=row['wall_ns'],'action sum exceeds whole trial')
    if program['format_version']=='steady-component-program-1':
        require(native.get('program_spec')==program,'native compact operation specification differs')
        result=native.get('component_output',{})
        require(result.get('kind')==program['kind'] and result.get('independent_oracle_applied_in_native') is False,
            'component numerical boundary differs')
        if oracle['kind']=='float_output':
            numerical=verify_float_output(oracle['values'],result.get('values'))
        else:
            require(result.get('full_byte_equality') is True and result.get('checked_bytes')==oracle['bytes'] and
                result.get('source_sha256')==result.get('sha256')==oracle['sha256'],'copy route bytes/hash differ')
            numerical=dict(passed=True,checked_bytes=oracle['bytes'],source_and_destination_sha256=oracle['sha256'])
    else:
        from profiling.highlevel_accuracy.bridge import verify_native as verify_layer
        numerical=dict(passed=True,**verify_layer(program,native['verification_native']))
    return numerical


def collect(root,split,*,barrier=None):
    from profiling.state_calibration import run
    root=Path(root).resolve();manifest=verify_manifest(root)
    frozen_barrier=None
    if split=='heldout':
        require(barrier is not None,'heldout requires frozen parameters')
        frozen_barrier=read(barrier)
        require(frozen_barrier['manifest_sha256']==sha(root/'manifest.json') and sha(frozen_barrier['frozen_file'])==frozen_barrier['frozen_file_sha256'],'frozen barrier differs')
    write_new(root/(split+'-started.json'),dict(started_at_utc=utc(),barrier=frozen_barrier,manifest_sha256=sha(root/'manifest.json')))
    results=[]
    with run.measurement_lock():
        for job in manifest['jobs']:
            if job['split']!=split:continue
            verify_manifest(root)
            if frozen_barrier is not None:
                require(sha(frozen_barrier['frozen_file'])==frozen_barrier['frozen_file_sha256'],'frozen parameters changed before heldout job')
            binding=manifest['programs'][job['program_key']]
            directory=root/'jobs'/job['id'];directory.mkdir(parents=True,exist_ok=False)
            request_path=directory/'request.json';output=directory/'result.json';write_new(request_path,job['request'])
            command=[manifest['binary'],binding['program']['path'],str(request_path),str(output)]
            env=run.child_environment(manifest['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
            inherited={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','GGML_','CUBLAS_','OMP_','MKL_','OPENBLAS_','NVIDIA_'))}
            require(inherited==manifest['inherited_backend_environment'],'backend environment changed since preparation')
            record=dict(schema='steady-cost-execution-1',job=job,started_at_utc=utc(),program=binding,
                request=receipt(request_path),command=command,status='failed',resource_state_used_as_model_input=False)
            record['observed_child_backend_environment']={k:v for k,v in env.items() if k.startswith(('CUDA_','GGML_','CUBLAS_','OMP_','MKL_','OPENBLAS_','NVIDIA_'))}
            process=None;handles=[]
            try:
                process,stdout,stderr,affinity=run.launch(command,env,directory/'stdout.txt',directory/'stderr.txt',manifest['protocol']['main_affinity_mask'])
                handles=[stdout,stderr];record.update(process_id=process.pid,affinity=affinity)
                code=process.wait(timeout=manifest['protocol']['process_timeout_seconds'])
                record.update(returncode=code,forced_termination=False,process_exited_verified=process.poll() is not None)
                require(code==0,'native process failed')
                native=read(output);require(native['process_id']==process.pid,'native process identity differs')
                program=read(binding['program']['path']);oracle=read(binding['oracle']['path'])
                record['numerical']=verify_native(program,job['request'],native,oracle)
                record['native']=receipt(output);record['status']='completed'
            except BaseException as error:
                record['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    record['forced_termination']=True
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                record['completed_at_utc']=utc();write_new(directory/'execution.json',record)
            require(record['status']=='completed','retained failed job: '+job['id']+' '+record.get('error',''))
            results.append(record);print(json.dumps(dict(job=job['id'],status=record['status'])),flush=True)
    write_new(root/(split+'-executions.json'),results)
    return dict(split=split,completed=len(results))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--out',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--runtime-dir',type=Path,action='append',required=True);p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('collect');p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=('profile','observer','diagnostic','heldout'),required=True);p.add_argument('--barrier',type=Path)
    p=sub.add_parser('freeze');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=prepare(args.out,args.binary,args.runtime_dir,seed=args.seed)
    elif args.command=='collect':result=collect(args.out,args.split,barrier=args.barrier)
    else:
        from .calibration import freeze,evaluate
        result=freeze(args.out) if args.command=='freeze' else evaluate(args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
