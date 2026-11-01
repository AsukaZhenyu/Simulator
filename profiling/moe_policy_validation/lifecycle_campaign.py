"""Independent lifecycle-matched follow-up to the retained hot-action campaign.

Calibration stamps full synthetic sequences with separate weights/inputs. It
does not fit any v1 validation timing or add an empirical strategy correction.
Predictions are frozen before 120 NEW bare validation processes.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import json
import statistics as st
from pathlib import Path
import os
from . import campaign as base
from .bridge import compile_expert_program
from .reference import reference_outputs, verify_result
from llm_infer_model.performance.expert_policy_response import action_price_keys, evaluate_expert_policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path,default=base.ROOT/'build/moe-policy-native-20261029/BUILD_COMPLETE.json')
    a=p.parse_args()
    out=a.out.resolve()
    if out.exists():raise ValueError('new output directory required')
    out.mkdir(parents=True)
    build=base.read(a.build);binary=Path(build['binary']['path'])
    sources=base.source_snapshot(out,build)
    env=dict(os.environ);env['PATH']=os.pathsep.join(build['runtime_dirs'])+os.pathsep+env['PATH']
    env['NVIDIA_TF32_OVERRIDE']='0';env['GGML_CUDA_DISABLE_GRAPHS']='1'
    base.save(out/'PROTOCOL.json',dict(created=base.utc(),calibration_seed=31002,validation_seed=41002,
        rationale='v1 isolated hot-action price mismatch; acquire full-lifecycle parameters independently, no validation-time fitting',
        calibration='six sequences per config: CPU/GPU-release/GPU-LRU crossed with hot/capacity-sensitive; 2fresh processes each,20stamped complete samples',
        prices='mean of per-process category means, rounded to integer ns; all action contexts retained',
        validation='all40fixed conditions,3fresh processes each,10bare samples,3warmup,1repeat',
        run_order='three rounds rotated by13*round; predictions frozen beforevalidation',sources=sources,build=build,
        overhead='per-action timestamp acquisition in calibration only; no stamps inside bare sequence validation; no correction estimated from validation'))
    with base.measurement_lock():
        calibrations={}
        for q in (1,32):
            for k in (2,4):
                config=base.config(k);cells=defaultdict(list);calrows=[]
                bootstrap=dict(schema='moe-expert-policy-calibration-1',config=config,boundary='serial_host_expert_response_v1',
                    prices_ns={'LOAD':1,'RELEASE':1,f'EXPERT:cpu:q={q}':1,f'EXPERT:gpu:q={q}':1,f'COMBINE:q={q}:topk=2':1})
                for label,recipe,retention in (('CPU','cpu_experts_lru','lru'),('GPU-release','gpu_demand_lru','release_after_use'),('GPU-LRU','gpu_demand_lru','lru')):
                    for locality in ('hot','capacity-sensitive'):
                        folder=out/'calibration'/f'q{q}-k{k}'/f'{label}-{locality}'
                        compiled=compile_expert_program(config=config,steps=base.steps(q,locality,31002),seed=31002,
                            recipe=recipe,retention=retention,prices=bootstrap,resource_guards=base.GUARDS)
                        program=compiled['program'];keys=action_price_keys(program);path=folder/'program.json';base.save(path,program)
                        expected=reference_outputs(program)
                        for rep in range(2):
                            result=base.invoke(binary,env,path,dict(mode='stamped',warmup=3,samples=20,repeats=1),folder/f'process-{rep}.json')
                            v=verify_result(program,result,expected=expected);base.save(folder/f'process-{rep}.verification.json',v)
                            if not v['numerically_equal']:raise RuntimeError('calibrationnumericsfailed')
                            values=defaultdict(list)
                            for sample in result['action_samples_ns']:
                                if len(sample)!=len(keys):raise RuntimeError('stamp/actionlengthdiffers')
                                for key,value in zip(keys,sample):values[key].append(value)
                            means={key:st.mean(value) for key,value in values.items()}
                            for key,mean in means.items():cells[key].append(mean)
                            calrows.append(dict(label=label,locality=locality,rep=rep,action_process_means_ns=means,
                                sequence_mean_ns=st.mean(result['samples_ns']),stamped_action_sum_mean_ns=st.mean(sum(x) for x in result['action_samples_ns'])))
                calibration=dict(schema='moe-expert-policy-calibration-1',config=config,boundary='serial_host_expert_response_v1',
                    prices_ns={key:round(st.mean(value)) for key,value in cells.items()},
                    process_category_means_ns=dict(cells),sequences=calrows,seed=31002,created=base.utc(),
                    context='complete serial lifecycle; cold logical cache per repeat; actual changing experts and slots; stamped components')
                base.save(out/'calibration'/f'q{q}-k{k}'/'calibration.json',calibration);calibrations[q,k]=calibration
                print('lifecycle calibrated',q,k,calibration['prices_ns'],flush=True)
        cases=[]
        for q in (1,32):
            for k in (2,4):
                for locality in ('hot','capacity-sensitive'):
                    for policy,recipe,retention in base.POLICIES:
                        cid=f'q{q}-k{k}-{locality}-{policy}';folder=out/'programs'/cid
                        result=compile_expert_program(config=base.config(k),steps=base.steps(q,locality,41002),seed=41002,
                            recipe=recipe,retention=retention,prices=calibrations[q,k],resource_guards=base.GUARDS)
                        base.save(folder/'program.json',result['program']);base.save(folder/'compilation.json',result['compilation_evidence'])
                        pred=evaluate_expert_policy(result['program'],calibrations[q,k]);base.save(folder/'prediction.json',pred)
                        base.save(folder/'reference.json',reference_outputs(result['program']))
                        cases.append(dict(id=cid,q=q,k=k,locality=locality,policy=policy,recipe=recipe,retention=retention,
                            program_sha256=base.digest(folder/'program.json'),prediction=pred))
        base.save(out/'FROZEN_PREDICTIONS.json',dict(created=base.utc(),cases=cases,source_hashes=sources,
            calibration_hashes={f'q{q}-k{k}':base.digest(out/'calibration'/f'q{q}-k{k}'/'calibration.json') for q,k in calibrations}))
        print('lifecycle predictions frozen;120new bare processes start',flush=True)
        for rep in range(3):
            offset=13*rep
            for case in cases[offset:]+cases[:offset]:
                folder=out/'validation'/case['id'];pdir=out/'programs'/case['id'];program=base.read(pdir/'program.json')
                native=base.invoke(binary,env,pdir/'program.json',dict(mode='bare',warmup=3,samples=10,repeats=1),folder/f'process-{rep}.json')
                v=verify_result(program,native,expected=base.read(pdir/'reference.json'));base.save(folder/f'process-{rep}.verification.json',v)
                if not v['numerically_equal']:raise RuntimeError('validationnumericsfailed:'+case['id'])
                print('lifecycle validation',rep,case['id'],round(st.mean(native['samples_ns'])/1e6,4),'ms',flush=True)
        after={path:base.digest(path) for path in sources}
        base.save(out/'SOURCE_UNCHANGED.json',dict(unchanged=after==sources,before=sources,after=after))
        if after!=sources:raise RuntimeError('sources changed duringlifecyclecampaign')
        result=base.summarize(out)
        base.save(out/'LIFECYCLE_RECEIPT.json',dict(actual_calibration_seed=31002,actual_validation_seed=41002,
            scope=result['scope'],new_validation_processes=120))

if __name__=='__main__':main()
