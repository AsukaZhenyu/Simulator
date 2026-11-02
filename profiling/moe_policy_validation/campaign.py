"""Frozen calibration and independent native expert-policy campaign.

Run from Simulator: python -m profiling.moe_policy_validation.campaign --out NEW
The shipped protocol fixes seeds 31001/41001, 40 conditions, 3 fresh processes
per condition and 10 bare complete-sequence samples per process.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT/'src', ROOT/'mapping', ROOT/'modeling', ROOT/'profiling'):
    sys.path.insert(0, str(folder))
from moe_policy_validation.bridge import compile_expert_program
from moe_policy_validation.reference import reference_outputs, verify_result
from llm_infer_model.performance.expert_policy_response import evaluate_expert_policy

POLICIES = (
    ('P0', 'cpu_experts_lru', 'lru'),
    ('P1', 'gpu_demand_lru', 'release_after_use'),
    ('P2', 'gpu_demand_lru', 'lru'),
    ('P3', 'current_cost_lru', 'release_after_use'),
    ('P4', 'current_cost_lru', 'lru'),
)
GUARDS = dict(host_reserve_bytes=512*1024**2, gpu_reserve_bytes=256*1024**2)

def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

@contextmanager
def measurement_lock():
    path=ROOT/'profiling/.measurement.lock'
    with path.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started=utc(),protocol='moe-policy-20261029'),f)
    try:yield
    finally:path.unlink()

def config(k):
    return dict(hidden=256, intermediate=512, experts=8, gpu_slots=k, threads=1, cuda_device=0)

def steps(q, locality, seed):
    pairs = [(0,1)]*8 if locality=='hot' else [(0,1),(2,3),(0,1),(2,3),(4,5),(6,7),(4,5),(6,7)]
    rng=np.random.default_rng(seed + q)
    return [dict(input=rng.uniform(-0.9,0.9,(q,256)).astype(np.float32).tolist(),
                 routes=[list(pair) for _ in range(q)], scores=[[0.6,0.4] for _ in range(q)]) for pair in pairs]

def calibration_program(q,k):
    s=steps(q,'hot',31001)[:1]
    actions=[dict(kind='EXPERT',step=0,expert=0,device='cpu'),dict(kind='LOAD',expert=1,slot=0),
             dict(kind='EXPERT',step=0,expert=1,device='gpu',slot=0),dict(kind='COMBINE',step=0),
             dict(kind='RELEASE',expert=1,slot=0)]
    return dict(schema='moe-expert-policy-native-1',config=config(k),seed=31001,
                resource_guards=GUARDS,steps=s,actions=actions)

def invoke(binary,env,program_path,request,out):
    request_path=out.with_suffix('.request.json');save(request_path,request)
    start=utc()
    proc=subprocess.run([str(binary),str(program_path),str(request_path),str(out)],env=env,
                        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    save(out.with_suffix('.process.json'),dict(start=start,end=utc(),returncode=proc.returncode,
        stdout=proc.stdout,stderr=proc.stderr,command=[str(binary),str(program_path),str(request_path),str(out)]))
    if proc.returncode:raise RuntimeError(f'native failed: {out}: {proc.stderr[-1500:]}')
    return read(out)

def source_snapshot(out,build):
    paths=list((ROOT/'profiling/moe_policy_validation').rglob('*.py'))+list((ROOT/'profiling/moe_policy_validation/native').glob('*'))
    paths += [ROOT/'mapping/tensor_mapping/composition_policy.py',ROOT/'mapping/tensor_mapping/composition_observation.py',
              ROOT/'mapping/tensor_mapping/causal.py', ROOT/'mapping/tensor_mapping/expert_scheduling.py',
              ROOT/'src/llm_simulator/expert_recipes.py',ROOT/'modeling/llm_infer_model/performance/expert_policy_response.py']
    result={str(p.resolve()):digest(p) for p in paths if p.is_file()}
    for p in paths:
        if p.is_file():
            target=out/'executed_sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for row in [build['binary']]+build['runtimes']:
        if digest(row['path'])!=row['sha256']:raise ValueError('build/runtime receipt changed: '+row['path'])
    return result

def summarize(out):
    frozen=read(out/'FROZEN_PREDICTIONS.json');protocol=read(out/'PROTOCOL.json');rows=[]
    for case in frozen['cases']:
        folder=out/'validation'/case['id'];vals=[];max_abs=0;max_l2=0
        for rep in range(3):
            r=read(folder/f'process-{rep}.json');v=read(folder/f'process-{rep}.verification.json')
            vals.append(st.mean(r['samples_ns']))
            max_abs=max(max_abs,v['max_abs_error']);max_l2=max(max_l2,max(x['relative_l2_error'] for x in v['steps']))
        mean=st.mean(vals);sd=st.stdev(vals);pred=case['prediction']['duration_ns']
        rows.append(dict(id=case['id'],q=case['q'],k=case['k'],locality=case['locality'],policy=case['policy'],
            predicted_ns=pred,mean_ns=mean,process_means_ns=vals,process_std_ns=sd,
            mean_95ci_ns=[mean-4.302652729911275*sd/math.sqrt(3),mean+4.302652729911275*sd/math.sqrt(3)],
            ape_percent=100*abs(pred-mean)/mean,max_abs_error=max_abs,max_relative_l2_error=max_l2,
            ledger={key:case['prediction'][key] for key in ('action_counts','weight_h2d_bytes','total_h2d_bytes','total_d2h_bytes','peak_occupied_expert_slots') if key in case['prediction']}))
    groups=[]
    for q in (1,32):
        for k in (2,4):
            for locality in ('hot','capacity-sensitive'):
                part=[x for x in rows if (x['q'],x['k'],x['locality'])==(q,k,locality)]
                chosen=min(part,key=lambda x:(x['predicted_ns'],x['policy']))
                fastest=min(part,key=lambda x:(x['mean_ns'],x['policy']))
                factor={x['policy']:x['mean_ns'] for x in part}
                groups.append(dict(q=q,k=k,locality=locality,predicted_choice=chosen['policy'],
                    measured_fastest=fastest['policy'],selection_regret_percent=100*(chosen['mean_ns']/fastest['mean_ns']-1),
                    cache_gpu_gain_percent=100*(1-factor['P2']/factor['P1']),
                    joint_vs_gpu_no_cache_gain_percent=100*(1-factor['P4']/factor['P1']),
                    joint_vs_cpu_gain_percent=100*(1-factor['P4']/factor['P0']),
                    interaction_ns=factor['P4']-factor['P3']-factor['P2']+factor['P1']))
    result=dict(schema='moe-policy-campaign-results-1',created=utc(),conditions=40,independent_processes=120,
        samples_per_process=10,calibration_seed=protocol['calibration_seed'],validation_seed=protocol['validation_seed'],
        mape_percent=st.mean(x['ape_percent'] for x in rows),max_ape_percent=max(x['ape_percent'] for x in rows),
        max_abs_error=max(x['max_abs_error'] for x in rows),max_relative_l2_error=max(x['max_relative_l2_error'] for x in rows),
        rows=rows,selection=groups,
        scope='Serial routed F32 expert FFN; synthetic auditable routes; native CPU/GPU arithmetic and bounded physical slots; no full model or overlap claim')
    save(out/'RESULTS.json',result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('rows','selection')},indent=2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path,default=ROOT/'build/moe-policy-native-20261029/BUILD_COMPLETE.json')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise ValueError('new output directory required')
    out.mkdir(parents=True);build=read(a.build);binary=Path(build['binary']['path'])
    sources=source_snapshot(out,build)
    env=dict(os.environ);env['PATH']=os.pathsep.join(build['runtime_dirs'])+os.pathsep+env['PATH']
    env['NVIDIA_TF32_OVERRIDE']='0';env['GGML_CUDA_DISABLE_GRAPHS']='1';env['OPENBLAS_NUM_THREADS']='1'
    save(out/'PROTOCOL.json',dict(created=utc(),policies=POLICIES,q=[1,32],gpu_slots=[2,4],
        locality=['hot','capacity-sensitive'],calibration_seed=31001,validation_seed=41001,
        calibration_processes_per_cell=2,profile_request=dict(mode='profile',warmup=3,samples=20,repeats=20),
        validation_request=dict(mode='bare',warmup=3,samples=10,repeats=1),fresh_processes_per_condition=3,
        guards=GUARDS,sources=sources,build=build,
        run_order='three rounds; rotate the 40-condition list by 13*round, preserving deterministic order within round'))
    with measurement_lock():
        calibrations={}
        for q in (1,32):
            for k in (2,4):
                program=calibration_program(q,k);folder=out/'calibration'/f'q{q}-k{k}'
                path=folder/'program.json';save(path,program);prices={};evidence={}
                keys=[f'EXPERT:cpu:q={q}','LOAD',f'EXPERT:gpu:q={q}',f'COMBINE:q={q}:topk=2','RELEASE']
                for idx,key in enumerate(keys):
                    means=[]
                    for rep in range(2):
                        result=invoke(binary,env,path,dict(mode='profile',warmup=3,samples=20,repeats=20,target_action_index=idx),folder/f'action{idx}-process{rep}.json')
                        means.append(st.mean(result['samples_ns']))
                    prices[key]=round(st.mean(means));evidence[key]=means
                c=dict(schema='moe-expert-policy-calibration-1',config=config(k),boundary='serial_host_expert_response_v1',
                    prices_ns=prices,measured_process_means_ns=evidence,created=utc(),seed=31001,
                    context='hot repeated action after actual prefix; sequence validation independent and not used to fit')
                save(folder/'calibration.json',c);calibrations[q,k]=c
                print('calibrated',q,k,prices,flush=True)
        cases=[]
        for q in (1,32):
            for k in (2,4):
                for locality in ('hot','capacity-sensitive'):
                    for policy,recipe,retention in POLICIES:
                        cid=f'q{q}-k{k}-{locality}-{policy}';folder=out/'programs'/cid
                        result=compile_expert_program(config=config(k),steps=steps(q,locality,41001),seed=41001,
                            recipe=recipe,retention=retention,prices=calibrations[q,k],resource_guards=GUARDS)
                        save(folder/'program.json',result['program']);save(folder/'compilation.json',result['compilation_evidence'])
                        prediction=evaluate_expert_policy(result['program'],calibrations[q,k])
                        save(folder/'prediction.json',prediction)
                        save(folder/'reference.json',reference_outputs(result['program']))
                        cases.append(dict(id=cid,q=q,k=k,locality=locality,policy=policy,recipe=recipe,retention=retention,
                            program_sha256=digest(folder/'program.json'),prediction=prediction))
        save(out/'FROZEN_PREDICTIONS.json',dict(created=utc(),cases=cases,source_hashes=sources,
            calibration_hashes={f'q{q}-k{k}':digest(out/'calibration'/f'q{q}-k{k}'/'calibration.json') for q,k in calibrations}))
        print('predictions frozen; starting 120 independent validation processes',flush=True)
        for rep in range(3):
            offset=13*rep;order=cases[offset:]+cases[:offset]
            for case in order:
                folder=out/'validation'/case['id'];pdir=out/'programs'/case['id'];program=read(pdir/'program.json')
                result=invoke(binary,env,pdir/'program.json',dict(mode='bare',warmup=3,samples=10,repeats=1),folder/f'process-{rep}.json')
                verification=verify_result(program,result,expected=read(pdir/'reference.json'))
                save(folder/f'process-{rep}.verification.json',verification)
                if not verification['numerically_equal']:raise RuntimeError('numerical failure retained: '+case['id'])
                print('validation',rep,case['id'],round(st.mean(result['samples_ns'])/1e6,4),'ms',flush=True)
        after={path:digest(path) for path in sources}
        save(out/'SOURCE_UNCHANGED.json',dict(unchanged=after==sources,before=sources,after=after))
        if after!=sources:raise RuntimeError('implementation changed during measurement')
        summarize(out)

if __name__=='__main__':main()
