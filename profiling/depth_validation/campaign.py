"""Connected Dense depth validation; action calibration precedes holdout."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from profiling.dac_validation import e2a
from profiling.state_calibration import run
from llm_infer_model.performance.reusable_action_costs import digest, semantic_action
from llm_infer_model.performance.execution_history import HistoryScope, SerialExecutionHistory
from llm_infer_model.performance.history_conditioned_costs import (
    HistoryConditionedResponseTable, response_coordinate, SCHEMA, BUCKETING, INITIALIZATION)

ROOT=Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=e2a.read,e2a.write,e2a.sha,e2a.receipt,e2a.require
RECIPES=(('cpu','cpu','cpu'),('gpu','gpu','gpu'),('gpu','cpu','gpu'))
FIELDS=['h2d_source_reuse_bucket']


def _prepare_cohort(arguments):
    from profiling.depth_validation.programs import build_one, reseed_case
    out,case,cache=arguments;out=Path(out);result={}
    for strategy in range(3):
        key=f'{case["phase"]}-l{case["layers"]}-s{strategy}-profile'
        cached=Path(cache)/'inputs'/key/'program.json' if cache else None
        if cached and cached.is_file():
            try:
                read(cached)
            except json.JSONDecodeError:
                cached=None
        else:cached=None
        if cached is not None:
            binding=reseed_case(cached,out/'inputs'/key,seed=2026102621,asset_pool=out/'initial-assets')
        else:
            binding=build_one(out/'inputs'/key,layers=case['layers'],strategy=strategy,
                phase=case['phase'],seed=2026102621,asset_pool=out/'initial-assets')
        result[key]=dict(binding,**case,strategy=strategy,split='profile')
        key2=f'{case["phase"]}-l{case["layers"]}-s{strategy}-heldout'
        heldout=reseed_case(binding['program']['path'],out/'inputs'/key2,
            seed=2026102622,asset_pool=out/'initial-assets')
        result[key2]=dict(heldout,**case,strategy=strategy,split='heldout')
        print(json.dumps(dict(prepared=key,independently_reseeded=key2)),flush=True)
    return result


def prepare(out,binary,*,pilot=False,workers=1,cache=None):
    out,binary=Path(out).resolve(),Path(binary).resolve()
    if not out.exists():out.mkdir(parents=True)
    require(not (out/'manifest.json').exists(),'manifest already frozen')
    old=read(ROOT/'build/h2d-history-20261026/campaign-v1/manifest.json')
    matrix=[dict(phase='prefill',layers=l,tokens=64,history=0) for l in ((4,) if pilot else (4,8,16,32))]
    if not pilot:matrix += [dict(phase='decode',layers=l,tokens=1,history=128) for l in (16,32)]
    require(type(workers) is int and 1<=workers<=2,'one or two software preparation workers required')
    programs={};arguments=[(str(out),case,str(Path(cache).resolve()) if cache else None) for case in matrix]
    if workers==1:
        for item in arguments:programs.update(_prepare_cohort(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for cohort in pool.map(_prepare_cohort,arguments):programs.update(cohort)
    programs=dict(sorted(programs.items()))
    source_paths=[Path(__file__),ROOT/'profiling/depth_validation/programs.py',
        ROOT/'profiling/depth_validation/native/depth_probe.cpp',
        ROOT/'profiling/depth_validation/native/initial_assets.hpp',
        ROOT/'profiling/depth_validation/native/CMakeLists.txt',
        ROOT/'profiling/highlevel_accuracy/bridge.py',ROOT/'profiling/highlevel_accuracy/multilayer.py',
        ROOT/'modeling/llm_infer_model/tensor/spec.py',ROOT/'modeling/llm_infer_model/m2/causal.py',
        ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
        ROOT/'modeling/llm_infer_model/performance/execution_history.py',
        ROOT/'modeling/llm_infer_model/performance/history_conditioned_costs.py']
    source_paths+=list((ROOT/'build/depth-validation-20261026/generated').glob('*.inc'))
    sources=[]
    for source in source_paths:
        target=out/'sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources.append(dict(source=receipt(source),archived=receipt(target)))
    target=out/'binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    jobs=[]
    for split,repeats in (('profile',2),('heldout',4)):
        keys=[k for k,v in programs.items() if v['split']==split]
        for repeat in range(repeats):
            order=keys[repeat:]+keys[:repeat]
            if repeat%2:order.reverse()
            for key in order:
                request=dict(schema='h2d-history-request-1',source_sha256=sha(source_paths[2]),
                    sampling=e2a.request('full',trials=16),expected_process_mask=4294967295,
                    mode='full',instrumentation='stamped' if split=='profile' else 'bare',
                    memory_mode='pageable',target_action_index=None)
                jobs.append(dict(id=f'{split}-r{repeat}-{key}',split=split,program_key=key,request=request))
    context=dict(old['baseline_context'] if 'baseline_context' in old else old['context'],
        binary_sha256=sha(binary),measurement_boundary='same complete GGML actions with binary initial data outside timing',
        logical_initialization=INITIALIZATION,profile_split='per-action only')
    m=dict(schema='dense-depth-validation-1',prepared_at_utc=e2a.utc(),pilot=pilot,
        programs=programs,jobs=jobs,sources=sources,binary=receipt(target),runtime=old['runtime'],
        runtime_dirs=old['runtime_dirs'],context=context,
        protocol=dict(profile_processes=sum(j['split']=='profile' for j in jobs),
            heldout_processes=sum(j['split']=='heldout' for j in jobs),trials=16,replays=8,
            mean_ape_gate=.1,p90_ape_gate=.2,raw_CV_is_accuracy_gate=False,timeout_seconds=600,
            source_and_parameter_freeze_before_heldout=True,no_retries=True,
            numerical_atol=5e-5,numerical_rtol=5e-5,capacity_or_actual_eviction_qualified=False))
    m['protocol']['fresh_export_diagnostic_retention']='initial state only; all original snapshot calculations and causal transitions retained'
    m['protocol']['cached_software_templates_reused_without_timing']=cache is not None
    m['protocol']['membership_optimization']='exact tuple-identity cached membership and order-preserving visible-set lookup; no numerical or scheduling change'
    write(out/'manifest.json',m);return dict(out=str(out),programs=len(programs),**m['protocol'])


def check(m):
    for s in m['sources']:
        require(sha(s['source']['path'])==s['source']['sha256'],'bound implementation changed')
        require(sha(s['archived']['path'])==s['archived']['sha256'],'archived implementation changed')
    for s in [m['binary'],*m['runtime']]:require(sha(s['path'])==s['sha256'],'binary/runtime changed')


def verify_native(n,job,program,oracle,pid):
    require(n['process_id']==pid and n['request']==job['request'],'native identity differs')
    require(n['program_sha256']==program['program_sha256'],'program identity differs')
    require(n['source_sha256']==job['request']['source_sha256'],'source differs')
    require(n['process_affinity']['final_mask']==4294967295 and n['main_thread_affinity']['restored'],'affinity differs')
    require(n['native_repeat_consistency']['passed'] and n['verification_native']['copy_values_equal'],'native repeated/copy output failed')
    got=n['verification_native']['ports'];require(set(got)==set(oracle),'reference port inventory differs')
    for tid,expected in oracle.items():
        a,b=np.asarray(got[tid]),np.asarray(expected)
        require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,atol=5e-5,rtol=5e-5),'numeric mismatch '+tid)
    require(n['initial_binary_buffers_verified']==len(program['initial_assets']['tensors']),'binary initialization inventory differs')
    hz=n['clock']['frequency_hz'];require(len(n['trials'])==16,'trial count differs')
    for r in n['trials']:
        require(r['repeats']==8 and r['end_tick']>r['begin_tick'],'QPC interval differs')
        require(abs((r['end_tick']-r['begin_tick'])*1e9/hz-r['wall_ns'])<=1,'QPC duration differs')
    return dict(passed=True,ports=len(got),blocks=16)


def collect(out,split):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    require(split in ('profile','heldout'),'unknown split');barrier=None
    if split=='heldout':
        barrier=read(out/'frozen-barrier.json')
        for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen artifacts changed')
    marker=out/(split+'-started.json')
    if not marker.exists():write(marker,dict(at_utc=e2a.utc(),barrier=barrier))
    executions=[]
    with run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists():executions.append(read(folder/'execution.json'));continue
            folder.mkdir(parents=True,exist_ok=False);binding=m['programs'][job['program_key']]
            pp=binding['program'];rp=binding['reference']
            require(sha(pp['path'])==pp['sha256'] and sha(rp['path'])==rp['sha256'],'program/reference changed')
            request=folder/'request.json';result=folder/'result.json';write(request,job['request'])
            command=[m['binary']['path'],pp['path'],str(request),str(result)]
            execution=dict(job=job,status='failed',started_at_utc=e2a.utc(),barrier=barrier,command=command)
            process=None;handles=[]
            try:
                env=run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
                process,a,b,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295)
                handles=[a,b];execution.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=m['protocol']['timeout_seconds'])
                execution.update(returncode=code,process_exited_verified=True,forced_termination=False)
                require(code==0,'native child failed; retained without retry')
                execution['verification']=verify_native(read(result),job,read(pp['path']),read(rp['path']),process.pid)
                execution.update(status='completed',native=receipt(result))
            except BaseException as error:
                execution['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    execution['forced_termination']=True
                if process is not None:execution.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for h in handles:h.close()
                execution['completed_at_utc']=e2a.utc();write(folder/'execution.json',execution)
            executions.append(execution)
            print(json.dumps(dict(job=job['id'],status=execution['status'],error=execution.get('error'))),flush=True)
    path=out/(split+'-executions.json')
    if not path.exists():write(path,executions)
    return dict(split=split,declared=len(executions),completed=sum(e['status']=='completed' for e in executions))


def freeze(out):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    require(not(out/'heldout-started.json').exists(),'fit after heldout is forbidden')
    executions=read(out/'profile-executions.json')
    require(len(executions)==m['protocol']['profile_processes'] and all(e['status']=='completed' for e in executions),'missing profile evidence')
    scope=digest(m['context']);groups={};evidence=defaultdict(list);programs={};native_rows=[];static_coords={};seen_evidence=defaultdict(set)
    def add(coord,value,source,key):
        require(type(value) is int and value>=0,'nonnegative native action ns required')
        group=groups.setdefault(key,dict(coordinate=coord,sum_ns=0,count=0));group['sum_ns']+=value;group['count']+=1
        if source['native']['sha256'] not in seen_evidence[key]:
            evidence[key].append(source);seen_evidence[key].add(source['native']['sha256'])
    for e in executions:
        key=e['job']['program_key']
        if key not in programs:
            programs[key]=read(m['programs'][key]['program']['path'])
            coordinates=[dict(scope_sha256=scope,semantic=semantic_action(programs[key],i),condition={}) for i in range(len(programs[key]['actions']))]
            static_coords[key]=[(c,digest(c)) for c in coordinates]
        n=read(e['native']['path']);native_rows.append((e,key))
        for block in n['trials']:
            for r in block['action_intervals']:
                coord,h=static_coords[key][r['action_index']]
                add(coord,r['wall_ns'],dict(native=e['native'],split='profile'),h)
    def entries():
        return [dict(key=k,coordinate=g['coordinate'],mean_ns=str(Fraction(g['sum_ns'],g['count'])),
                     sample_count=g['count'],evidence=evidence[k]) for k,g in sorted(groups.items())]
    static={digest(g['coordinate']['semantic']):Fraction(g['sum_ns'],g['count']) for g in groups.values()}
    traces={}
    for key,p in programs.items():
        history=SerialExecutionHistory(p,HistoryScope(scope));durations=[static[digest(semantic_action(p,i))] for i in range(len(p['actions']))]
        for i,d in enumerate(durations):history.commit(i,d)
        trace={}
        for invocation in range(8):
            for i,d in enumerate(durations):
                c=response_coordinate(history.snapshot(i),FIELDS);trace[invocation,i]=(c,digest(c));history.commit(i,d)
        traces[key]=trace
    for e,key in native_rows:
        n=read(e['native']['path'])
        for block in n['trials']:
            for r in block['action_intervals']:
                coord,h=traces[key][r['invocation'],r['action_index']]
                if coord['condition']:add(coord,r['wall_ns'],dict(native=e['native'],split='profile'),h)
    profile=dict(schema=SCHEMA,scope_sha256=scope,scope=m['context'],condition_fields=FIELDS,bucketing=BUCKETING,
        initialization=INITIALIZATION,unmodeled_reset_accesses=True,entries=entries(),frozen_at_utc=e2a.utc(),
        target_time_used_for_fit=False,future_target_state_used=False,calibration_label='inclusive per-action intervals only')
    table=HistoryConditionedResponseTable.from_profile(profile);predictions=[]
    for key,b in m['programs'].items():
        if b['split']!='heldout':continue
        p=read(b['program']['path']);costs={}
        for name,enabled in (('context_static',False),('history_conditional',True)):
            prediction=table.predict_blocks(p,blocks=1,repeats=8,use_history=enabled)
            costs[name]=str(prediction['duration_ns'])
        predictions.append(dict(program_key=key,layers=b['layers'],phase=b['phase'],strategy=b['strategy'],costs=costs))
    write(out/'model-frozen.json',profile);write(out/'predictions-frozen.json',predictions)
    write(out/'frozen-barrier.json',dict(frozen_at_utc=e2a.utc(),files={n:sha(out/n) for n in ('model-frozen.json','predictions-frozen.json')},
        all_target_coordinates_known=True,target_outcomes_read=False))
    return dict(response_cells=len(profile['entries']),heldout_cases=len(predictions))


def evaluate(out):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    barrier=read(out/'frozen-barrier.json')
    for n,h in barrier['files'].items():require(sha(out/n)==h,'freeze changed')
    executions=read(out/'heldout-executions.json');require(len(executions)==m['protocol']['heldout_processes'],'heldout denominator differs')
    results=[]
    for pred in read(out/'predictions-frozen.json'):
        actual=[]
        for e in executions:
            if e['job']['program_key']!=pred['program_key']:continue
            row=dict(raw=e.get('native'),status=e['status'])
            if e['status']=='completed':
                n=read(e['native']['path']);values=[r['wall_ns']/8 for r in n['trials']]
                row.update(mean_ns=statistics.mean(values),trial_ns=values,cv=statistics.stdev(values)/statistics.mean(values),
                    late_over_early=statistics.mean(values[8:])/statistics.mean(values[:8])-1)
            actual.append(row)
        providers={}
        for name,value in pred['costs'].items():
            price=float(Fraction(value));good=[r for r in actual if r['status']=='completed']
            apes=[abs(price-r['mean_ns'])/r['mean_ns'] for r in good]
            p90=float(np.quantile([abs(price-v)/v for r in good for v in r['trial_ns']],.9,method='higher')) if good else None
            providers[name]=dict(predicted_ns=price,max_process_mean_ape=max(apes) if apes else None,p90_block_ape=p90,
                accuracy_passed=len(good)==4 and max(apes)<=.1 and p90<=.2,
                signed_mean_error=(price/statistics.mean(r['mean_ns'] for r in good)-1) if good else None)
        results.append(dict(**pred,processes=actual,providers=providers))
    result=dict(schema='dense-depth-validation-result-1',evaluated_at_utc=e2a.utc(),declared_cases=len(results),cases=results,
        providers={n:dict(accuracy_passes=sum(r['providers'][n]['accuracy_passed'] for r in results),cases=len(results)) for n in ('context_static','history_conditional')},
        declared_processes=len(executions),completed_processes=sum(e['status']=='completed' for e in executions),
        target_time_used_for_fit=False,autoregressive_generation=False,raw_cv_is_accuracy_gate=False)
    write(out/'summary.json',result);return {k:result[k] for k in ('declared_cases','providers','declared_processes','completed_processes')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','profile','freeze','heldout','evaluate'))
    parser.add_argument('--out',required=True);parser.add_argument('--binary');parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=1);parser.add_argument('--cache');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.binary,pilot=a.pilot,workers=a.workers,cache=a.cache)
    elif a.command=='freeze':r=freeze(a.out)
    elif a.command=='evaluate':r=evaluate(a.out)
    else:r=collect(a.out,a.command)
    print(json.dumps(r,ensure_ascii=False),flush=True)
