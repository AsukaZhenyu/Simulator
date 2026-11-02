"""State-aligned, source-bound native experiments. Historical evidence is read-only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT/'profiling',ROOT/'mapping',ROOT/'modeling'):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from state_calibration.telemetry import DeviceTelemetrySampler,NativeClock,align_intervals

def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
def cv(values):return statistics.pstdev(values)/statistics.mean(values) if len(values)>1 else 0.0

@contextmanager
def measurement_lock():
    path=ROOT/'profiling/.measurement.lock'
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(dict(pid=os.getpid(),started=utc(),protocol='state_calibration')))
    try:yield
    finally:path.unlink()

def sources(binary,runtime_dirs):
    paths={Path(binary).resolve(),Path(binary).parent/'build-manifest.json'}
    build=read(Path(binary).parent/'build-manifest.json')
    if build.get('returncode')!=0 or build.get('binary_sha256')!=sha(binary) or build.get('bound_inputs_unchanged') is not True:
        raise ValueError('native successful-build binding invalid')
    for name,digest in build['bound_files'].items():
        if sha(name)!=digest:raise ValueError('native build input changed: '+name)
        paths.add(Path(name))
    for folder in (ROOT/'profiling/state_calibration',ROOT/'profiling/native/state_calibration',ROOT/'profiling/highlevel_accuracy'):
        paths.update(folder.glob('*.py'));paths.update(folder.glob('*.cpp'))
    for folder in (ROOT/'modeling/llm_infer_model',ROOT/'mapping/tensor_mapping',ROOT/'src/llm_simulator'):
        paths.update(folder.rglob('*.py'))
    for folder in runtime_dirs:
        paths.update(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower()=='.dll')
        private=Path(folder)/'strict-f32-runtime.json'
        if private.is_file():
            paths.add(private)
            record=read(private)
            for field in ('original_source_sha256','inputs_sha256','output_sha256','evidence_sha256'):
                for name,digest in record[field].items():
                    if sha(name)!=digest:raise ValueError('private F32 runtime asset changed: '+name)
                    paths.add(Path(name))
    return {str(p.resolve()):sha(p) for p in sorted(paths)}

def prepare(out,binary,runtime_dirs,smoke=False,interval_ms=50.0):
    import math
    if isinstance(interval_ms,bool)or not isinstance(interval_ms,(int,float))or not math.isfinite(interval_ms)or interval_ms<=0:
        raise ValueError('sampling interval must be finite and positive')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    from highlevel_accuracy.bridge import export_plan,reference_ports
    from highlevel_accuracy.cases import build_case
    from highlevel_accuracy.run import case_specs
    specifications={row['id']:row['kwargs']for row in case_specs('representative')}
    cases=[('h128-q16','h128-q16-history0-attention-ffn-cut128'),('h256-q32','h256-q32-history0-attention-ffn-cut252')]
    if smoke:cases=cases[-1:]
    programs={}
    input_bindings={}
    for case_index,(name,old_name) in enumerate(cases):
        compiled=build_case(out/'inputs'/name,**specifications[old_name]).compile()
        input_bindings.update({str(Path(row['path']).resolve()):row['sha256']for row in compiled._compiled.sources})
        programs[name]={}
        for split in ('profile','heldout'):
            for repeat in range(1 if smoke else 3):
                target=out/'programs'/f'{name}-{split}-{repeat}.json'
                seed=27101+case_index if split=='profile'else 100000001+case_index*100+repeat
                program=export_plan(compiled,seed=seed);save(target,program);reference=reference_ports(program)
                ref=out/'references'/f'{name}-{split}-{repeat}.json';save(ref,reference)
                programs[name][f'{split}-{repeat}']=dict(path=str(target),sha256=sha(target),reference=str(ref),reference_sha256=sha(ref),seed=program['seed'])
    jobs=[]
    conditions=('baseline','cpu_replay_load','gpu_replay_load')
    for split in ('profile','heldout'):
        for repeat in range(1 if smoke else 3):
            # Matched cohorts, deterministic rotation, and all profiles precede any heldout.
            for name,_ in cases:
                for condition in conditions[repeat%len(conditions):]+conditions[:repeat%len(conditions)]:
                    for mode in (('isolated','mixed') if repeat%2==0 else ('mixed','isolated')):
                        ident=f'{split}-{repeat}-{name}-{condition}-{mode}'
                        jobs.append(dict(id=ident,split=split,case_id=name,seed=programs[name][f'{split}-{repeat}']['seed'],repeat=repeat,
                                         condition=condition,mode=mode,instrumentation='stamped',telemetry=True,
                                         program_key=f'{split}-{repeat}',fresh_native_process=True))
    # Controls quantify observation/instrumentation cost; excluded from every model fit.
    for repeat in range(1 if smoke else 3):
        for mode in ('isolated','mixed'):
            for stamped,telemetry in ((False,False),(False,True),(True,False),(True,True)):
                name=cases[-1][0]
                jobs.append(dict(id=f'diagnostic-{repeat}-{mode}-{int(stamped)}-{int(telemetry)}',split='diagnostic',case_id=name,
                    seed=programs[name][f'profile-{repeat}']['seed'],repeat=repeat,condition='baseline',mode=mode,
                    instrumentation='stamped' if stamped else 'bare',telemetry=telemetry,program_key=f'profile-{repeat}',fresh_native_process=True))
    boundaries=('legacy_double_sync','async_one_sync','async_events')
    for repeat in range(1 if smoke else 3):
        for mode in ('isolated','mixed'):
            for boundary in boundaries[repeat:]+boundaries[:repeat]:
                name=cases[-1][0]
                jobs.append(dict(id=f'boundary-{repeat}-{mode}-{boundary}',split='diagnostic',case_id=name,
                    seed=programs[name][f'profile-{repeat}']['seed'],repeat=repeat,condition='baseline',mode=mode,
                    instrumentation='stamped',telemetry=True,execution_boundary=boundary,
                    program_key=f'profile-{repeat}',fresh_native_process=True))
    manifest=dict(schema='state-calibration-campaign-0.1',created_utc=utc(),binary=str(Path(binary).resolve()),runtime_dirs=[str(Path(p).resolve())for p in runtime_dirs],
        sources={**sources(binary,runtime_dirs),**input_bindings},programs=programs,jobs=jobs,smoke=smoke,
        protocol=dict(blocks=3,fixed_iterations_per_block=256,minimum_block_ms=200,warmup=20,interval_ms=interval_ms,primary_execution_boundary='legacy_double_sync',
            process_timeout_seconds=180,background_timeout_seconds=240,within_process_cv_max=.15,between_process_cv_max=.20,error_max=.20,
            background='finite owned native replay; no user application/clock/power changes',
            condition_inference='explicit controlled execution regime; telemetry associations are descriptive; no utilization-to-throughput heuristic',
            price_scope='exact operation/context/regime; not a universal GPU multiplier',
            timing_units='all per-invocation samples; no block averaging or filtering',
            original_programs='Fresh public ModelPlan compile/exports and independent numerical references; no existing profile or historical measurement required'))
    save(out/'manifest.json',manifest)
    print(json.dumps(dict(prepared=str(out),jobs=len(jobs),hardware_executed=False)),flush=True)

def verify_sources(manifest):
    for path,digest in manifest['sources'].items():
        if sha(path)!=digest:raise ValueError('campaign source/runtime changed: '+path)

def choose_target(program,device='gpu'):
    bindings={b['operation_id']:b for b in program['bindings']}
    preferred='attention_pv' if device=='gpu' else 'ffn_down'
    for i,action in enumerate(program['actions']):
        if action['kind']=='COMPUTE' and action['device']==device and bindings[action['target_id']]['kind']==preferred:return i
    raise ValueError('no exact supported target '+device+'/'+preferred)

def validate_numeric(native,reference):
    import numpy as np
    maximum=0.0
    for field in ('initial_ports','ports'):
        if set(native[field])!=set(reference):raise ValueError('native numerical port identities differ')
        for key,want in reference.items():
            a=np.asarray(want,dtype=float);b=np.asarray(native[field][key],dtype=float)
            if a.shape!=b.shape or not np.all(np.isfinite(b)) or not np.allclose(a,b,atol=5e-5,rtol=5e-5):raise ValueError('independent numerical mismatch '+field+'/'+key)
            maximum=max(maximum,float(np.max(np.abs(a-b))))
    if not native['copy_values_equal'] or not native['native_repeat_consistency']['passed'] or not native['post_replay_consistency']['passed']:raise ValueError('native repeat/copy check failed')
    return dict(passed=True,maximum_absolute_error=maximum,atol=5e-5,rtol=5e-5,oracle='independent NumPy reference_ports outside all timed native work')

def _affinity(process,mask):
    if os.name!='nt':return dict(applied=False,reason='not requested on POSIX')
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t];kernel.SetProcessAffinityMask.restype=ctypes.c_int
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];kernel.GetProcessAffinityMask.restype=ctypes.c_int
    if not kernel.SetProcessAffinityMask(int(process._handle),mask):raise OSError('owned-child affinity set failed')
    observed=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(int(process._handle),__import__('ctypes').byref(observed),__import__('ctypes').byref(system)) or observed.value!=mask:raise OSError('owned-child affinity readback differs')
    return dict(applied=True,requested_mask=mask,observed_mask=observed.value)

def launch(command,env,stdout,stderr,mask):
    out=stdout.open('x',encoding='utf-8');err=stderr.open('x',encoding='utf-8')
    try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    except BaseException:out.close();err.close();raise
    try:affinity=_affinity(process,mask)
    except BaseException:process.terminate();process.wait(timeout=10);out.close();err.close();raise
    return process,out,err,affinity

def _wait_ready(process,path,timeout=30):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if process.poll() is not None:raise RuntimeError('owned background exited before readiness')
        if time.monotonic()>deadline:raise TimeoutError('owned background readiness timeout')
        time.sleep(.02)
    # Exclusive creation can become visible before the writer closes its JSON.
    while True:
        try:return read(path)
        except (OSError,json.JSONDecodeError):
            if process.poll() is not None:raise RuntimeError('background readiness file incomplete at process exit')
            if time.monotonic()>deadline:raise TimeoutError('background readiness JSON timeout')
            time.sleep(.02)

def child_environment(runtime_dirs,fixed_iterations=256):
    """Set precision flags before Windows loads native CUDA dependencies."""
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    env=dict(os.environ)
    env['PATH']=os.pathsep.join([*map(str,runtime_dirs),env.get('PATH','')])
    env['STATE_PROBE_FIXED_ITERATIONS']=str(fixed_iterations)
    env['STATE_PROBE_EXECUTION_BOUNDARY']='legacy_double_sync'
    env.update({key:value for key,value in RUNTIME_FLAGS.items()if key!='matmul_precision'})
    env.pop('STATE_PROBE_READY_FILE',None);env.pop('STATE_PROBE_STOP_FILE',None)
    return env

def verify_native_protocol(native,program,job,protocol,clock,process_id):
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    if native.get('format_version')!='state-calibration-native-v1' or native.get('program_sha256')!=program['program_sha256']:
        raise ValueError('native program identity mismatch')
    target=choose_target(program)
    if native.get('mode')!=job['mode'] or native.get('target_action_index')!=target or native.get('process_id')!=process_id:
        raise ValueError('native process/mode/target identity mismatch')
    if native.get('actions')!=program['actions'] or native.get('target',{}).get('action')!=program['actions'][target]:raise ValueError('native action trace mismatch')
    boundary=job.get('execution_boundary','legacy_double_sync')
    if boundary not in ('legacy_double_sync','async_one_sync','async_events')or native.get('execution_boundary')!=boundary:raise ValueError('native execution boundary mismatch')
    if boundary!='legacy_double_sync'and job.get('split')!='diagnostic':raise ValueError('alternate boundary must remain a separate diagnostic')
    if native.get('observed_runtime_flags')!=RUNTIME_FLAGS:raise ValueError('native precision/runtime flags differ')
    if native['clock']['clock_id']!=clock['clock_id'] or native['clock']['frequency_hz']!=clock['frequency_hz']:raise ValueError('native telemetry clock mismatch')
    if native.get('graceful_stop_requested') or native.get('iteration_policy')!='fixed_count':raise ValueError('primary run unexpectedly stopped or changed iteration policy')
    if len(native['blocks'])!=protocol['blocks'] or native.get('requested_iterations_per_block')!=protocol['fixed_iterations_per_block']:raise ValueError('native fixed sampling envelope mismatch')
    last=-1
    for bi,block in enumerate(native['blocks']):
        if block['block_index']!=bi or block['begin_tick']<last or block['end_tick']<=block['begin_tick']:raise ValueError('native block chronology mismatch')
        wall=(block['end_tick']-block['begin_tick'])*1e9/clock['frequency_hz']
        if abs(block['wall_elapsed_ns']-wall)>1:raise ValueError('native block wall duration mismatch')
        iterations=block['iterations']
        if block['iteration_count']!=protocol['fixed_iterations_per_block'] or len(iterations)!=block['iteration_count']:raise ValueError('native invocation coverage mismatch')
        if block['completed_work_ns']!=sum(item['elapsed_ns']for item in iterations):raise ValueError('native completed work duration mismatch')
        if block['minimum_elapsed_satisfied']!=(block['wall_elapsed_ns']>=protocol['minimum_block_ms']*1000000):raise ValueError('native minimum duration flag mismatch')
        if block['fixed_iteration_count_satisfied']is not True or native.get('full_fixed_count')is not True:raise ValueError('native fixed count flag mismatch')
        previous=block['begin_tick']
        for i,item in enumerate(iterations):
            if item['iteration']!=i or not previous<=item['begin_tick']<item['end_tick']<=block['end_tick']:raise ValueError('native invocation chronology mismatch')
            expected=(item['end_tick']-item['begin_tick'])*1e9/clock['frequency_hz']
            if abs(item['elapsed_ns']-expected)>1 or item['elapsed_ns']<=0:raise ValueError('native duration/clock mismatch')
            previous=item['end_tick']
        actions=block['action_intervals'];expected_indices=[target]if job['mode']=='isolated'else list(range(len(program['actions'])))
        if job['instrumentation']=='stamped':
            if len(actions)!=len(iterations)*len(expected_indices):raise ValueError('native stamped action coverage mismatch')
            for i,item in enumerate(iterations):
                local=actions[i*len(expected_indices):(i+1)*len(expected_indices)]
                end=item['begin_tick']
                for ai,action in zip(expected_indices,local):
                    if action['iteration']!=i or action['action_index']!=ai or not end<=action['begin_tick']<=action['end_tick']<=item['end_tick']:raise ValueError('native action identity/chronology mismatch')
                    expected=(action['end_tick']-action['begin_tick'])*1e9/clock['frequency_hz']
                    if abs(action['elapsed_ns']-expected)>1:raise ValueError('native action duration mismatch')
                    end=action['end_tick']
        elif actions:raise ValueError('bare run contains unexpected stamped actions')
        diagnostics=block.get('target_diagnostics')
        if boundary=='legacy_double_sync':
            if diagnostics!=[]:raise ValueError('legacy boundary unexpectedly has event/enqueue diagnostics')
        else:
            if not isinstance(diagnostics,list)or len(diagnostics)!=len(iterations):raise ValueError('target diagnostic coverage mismatch')
            for i,(item,diag)in enumerate(zip(iterations,diagnostics)):
                if diag['iteration']!=i or diag['action_index']!=target or diag['execution_boundary']!=boundary:raise ValueError('target diagnostic identity mismatch')
                ticks=[diag[k]for k in ('begin_tick','host_enqueue_begin_tick','host_enqueue_end_tick','host_sync_begin_tick','host_sync_end_tick','end_tick')]
                if ticks!=sorted(ticks)or not item['begin_tick']<=ticks[0]<=ticks[-1]<=item['end_tick']:raise ValueError('target diagnostic chronology mismatch')
                for field,a,b in (('host_elapsed_ns',ticks[0],ticks[5]),('host_enqueue_ns',ticks[1],ticks[2]),('host_sync_ns',ticks[3],ticks[4])):
                    if abs(diag[field]-(b-a)*1e9/clock['frequency_hz'])>1:raise ValueError('target diagnostic clock duration mismatch')
                if boundary=='async_one_sync':
                    if diag['stream_span_ns']is not None or diag['stream_span_ms']is not None:raise ValueError('non-event boundary invented CUDA elapsed time')
                else:
                    import math
                    ms=diag['stream_span_ms'];ns=diag['stream_span_ns']
                    if isinstance(ms,bool)or not isinstance(ms,(int,float))or not math.isfinite(ms)or ms<0 or not isinstance(ns,int)or abs(ns-ms*1e6)>1:raise ValueError('CUDA event span conversion invalid')
        last=block['end_tick']
    return True

def execute_job(out,manifest,job):
    directory=out/'jobs'/job['id'];directory.mkdir(parents=True,exist_ok=False)
    binding=manifest['programs'][job['case_id']][job['program_key']]
    if sha(binding['path'])!=binding['sha256'] or sha(binding['reference'])!=binding['reference_sha256']:raise ValueError('program/reference bytes changed')
    program=read(binding['path']);reference=read(binding['reference']);target=choose_target(program)
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    env=child_environment(manifest['runtime_dirs'],manifest['protocol']['fixed_iterations_per_block'])
    env['STATE_PROBE_EXECUTION_BOUNDARY']=job.get('execution_boundary','legacy_double_sync')
    native_file=directory/'native.json';p=manifest['protocol'];clock=NativeClock()
    command=[manifest['binary'],binding['path'],job['mode'],str(target),str(p['warmup']),str(p['blocks']),str(p['minimum_block_ms']),job['instrumentation'],str(native_file)]
    row=dict(job=job,started_utc=utc(),launch_tick=clock.now(),command=command,program_sha256=binding['sha256'],summary={},
             requested_runtime_flags=RUNTIME_FLAGS,flags_set_before_process_launch=True)
    sampler=None;background=None;main=None;handles=[]
    try:
        if job['condition']!='baseline':
            bg_device='gpu' if job['condition']=='gpu_replay_load' else 'cpu'
            bg_target=choose_target(program,bg_device);bg_env=dict(env);bg_env.pop('STATE_PROBE_FIXED_ITERATIONS',None)
            bg_ready=directory/'background-ready.json';bg_stop=directory/'background-stop.json'
            bg_env.update(STATE_PROBE_READY_FILE=str(bg_ready),STATE_PROBE_STOP_FILE=str(bg_stop))
            bg_command=[manifest['binary'],binding['path'],'isolated',str(bg_target),'20','1000','200','bare',str(directory/'background-native.json')]
            background,a,b,bg_affinity=launch(bg_command,bg_env,directory/'background.stdout',directory/'background.stderr',12 if bg_device=='gpu' else 3);handles.extend((a,b))
            row['background']=dict(command=bg_command,pid=background.pid,affinity=bg_affinity,ready=_wait_ready(background,bg_ready))
            # Parent-controlled settling time is recorded, not part of operator timing.
            time.sleep(.3)
        if job['telemetry']:sampler=DeviceTelemetrySampler(interval_ms=p['interval_ms']).start()
        main,a,b,affinity=launch(command,env,directory/'native.stdout',directory/'native.stderr',3);handles.extend((a,b));row['affinity']=affinity
        row['process_id']=main.pid
        row['returncode']=main.wait(timeout=p['process_timeout_seconds']);row['exit_tick']=clock.now()
        if sampler is not None:
            sampler.stop();sampler.write_jsonl(directory/'telemetry.jsonl');save(directory/'telemetry-metadata.json',sampler.metadata())
        if background is not None:
            save(directory/'background-stop.json',dict(requested_tick=clock.now()));row['background']['returncode']=background.wait(timeout=30)
        if row['returncode']!=0:raise RuntimeError('native returned nonzero; raw stdout/stderr retained')
        native=read(native_file);verify_native_protocol(native,program,job,p,clock.metadata(),main.pid)
        numeric=validate_numeric(native,reference);save(directory/'numerical.json',numeric)
        iterations=[x['elapsed_ns']for block in native['blocks']for x in block['iterations']]
        targets=[x['elapsed_ns']for block in native['blocks']for x in block['action_intervals']if x['action_index']==target]
        if job['mode']=='isolated' and not targets:targets=iterations
        condition_verified=True
        if background is not None:
            bg=read(directory/'background-native.json')
            if row['background']['returncode']!=0:raise ValueError('owned background failed')
            validate_numeric(bg,reference)
            intervals=[(x['begin_tick'],x['end_tick'])for block in bg['blocks']for x in block['iterations']]
            first=min(a for a,b in intervals);last=max(b for a,b in intervals)
            condition_verified=all(first<=block['begin_tick'] and block['end_tick']<=last for block in native['blocks'])
            row['background'].update(active_begin_tick=first,active_end_tick=last,condition_interval_covers_all_blocks=condition_verified,
                activity_scope='owned background invocation envelope; gaps between invocations are retained, not continuous GPU occupancy proof')
        alignments=[]
        if sampler is not None:
            alignments=align_intervals(native['blocks'],sampler.records,sampler.metadata()['clock'],interval_ms=p['interval_ms'],action_clock=native['clock'])
            save(directory/'alignment.json',alignments)
        ready=bool(alignments) and all(x['quality']['model_ready']for x in alignments)
        fixed=all(block['iteration_count']==p['fixed_iterations_per_block']for block in native['blocks']) and len(native['blocks'])==p['blocks']
        minimum=all(block['wall_elapsed_ns']>=p['minimum_block_ms']*1000000 for block in native['blocks'])
        state={key:statistics.median([x['observed_fields'][key]['median']for x in alignments if key in x['observed_fields']])
               for key in sorted({k for x in alignments for k in x['observed_fields']})}
        row['summary']=dict(process_id=native['process_id'],target_median_ns=statistics.median(targets)if targets else None,
            iteration_median_ns=statistics.median(iterations),target_cv=cv(targets)if targets else None,iteration_cv=cv(iterations),
            target_samples_ns=targets,iteration_samples_ns=iterations,native_numeric_passed=True,condition_verified=condition_verified,
            alignment_model_ready=ready and fixed and minimum,device_state=state,
            fixed_count_verified=fixed,minimum_block_elapsed_verified=minimum,native_sha256=sha(native_file),
            target_boundary='selected completed GGML action' if job['instrumentation']=='stamped'else'outer isolated invocation bracket; no per-action stamping')
        row['status']='measured'
    except Exception as error:
        row.update(status='failed',error_type=type(error).__name__,error=str(error));row['summary'].update(native_numeric_passed=False,condition_verified=False,alignment_model_ready=False)
    finally:
        if main is not None and main.poll() is None:main.terminate();main.wait(timeout=10)
        if sampler is not None and not sampler.metadata()['stopped']:
            sampler.stop();sampler.write_jsonl(directory/'telemetry.jsonl');save(directory/'telemetry-metadata.json',sampler.metadata())
        if background is not None and background.poll() is None:
            stop=directory/'background-stop.json'
            if not stop.exists():save(stop,dict(requested_tick=clock.now(),cleanup=True))
            try:background.wait(timeout=15)
            except subprocess.TimeoutExpired:background.terminate();background.wait(timeout=10);row['background_forced_termination']=True
        for handle in handles:handle.close()
    row.update(completed_utc=utc(),completed_tick=clock.now(),native_clock=clock.metadata())
    if sha(binding['path'])!=binding['sha256'] or sha(binding['reference'])!=binding['reference_sha256']:
        row.update(status='failed',error='program/reference bytes changed during native process');row['summary']['native_numeric_passed']=False
    row['raw_file_sha256']={path.name:sha(path)for path in directory.iterdir()if path.is_file()}
    row['artifacts_sha256']=dict(row['raw_file_sha256'])
    save(directory/'execution.json',row)
    return row

def collect(out,split):
    out=Path(out).resolve();manifest=read(out/'manifest.json');verify_sources(manifest)
    if split=='heldout':
        barrier=read(out/'profile-barrier.json')
        if sha(out/'frozen-model.json')!=barrier['model_sha256'] or sha(out/'profile-rows.json')!=barrier['profile_rows_sha256'] or sha(out/'manifest.json')!=barrier['manifest_sha256']:
            raise ValueError('frozen profile/model/manifest changed before heldout')
    rows=[]
    with measurement_lock():
        for job in (j for j in manifest['jobs']if j['split']==split):
            verify_sources(manifest);row=execute_job(out,manifest,job);rows.append(row)
            print(json.dumps(dict(id=job['id'],status=row['status'],target_ns=row['summary'].get('target_median_ns'),state=row['summary'].get('device_state'),error=row.get('error'))),flush=True)
        verify_sources(manifest)
    save(out/(split+'-rows.json'),rows)
    if split=='profile':
        from state_calibration.model import fit_model
        model=fit_model(rows,source_bindings=dict(manifest_sha256=sha(out/'manifest.json'),**manifest['sources']))
        save(out/'frozen-model.json',model);save(out/'profile-barrier.json',dict(frozen_utc=utc(),manifest_sha256=sha(out/'manifest.json'),model_sha256=sha(out/'frozen-model.json'),profile_rows_sha256=sha(out/'profile-rows.json'),all_sources_finished=True))
    elif split=='heldout':
        from state_calibration.model import evaluate
        save(out/'report.json',evaluate(read(out/'frozen-model.json'),rows))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--out',type=Path,required=True);prep.add_argument('--binary',type=Path,required=True);prep.add_argument('--runtime-dir',type=Path,action='append',required=True);prep.add_argument('--smoke',action='store_true');prep.add_argument('--interval-ms',type=float,default=50.0)
    run=sub.add_parser('collect');run.add_argument('--out',type=Path,required=True);run.add_argument('--split',choices=('diagnostic','profile','heldout'),required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.binary,args.runtime_dir,args.smoke,args.interval_ms)
    else:collect(args.out,args.split)
if __name__=='__main__':main()
