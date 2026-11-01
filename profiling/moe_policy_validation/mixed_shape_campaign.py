"""Mixed CPU/GPU addendum using already-frozen lifecycle parameters only.

Eight causal routed calls: four q32 calls followed by four q1 calls. This is
not a full model P/D handoff: each call has its supplied host input/output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics as st
from . import campaign as base
from .bridge import compile_expert_program
from .reference import reference_outputs, verify_result
from llm_infer_model.performance.expert_policy_response import evaluate_expert_policy

def canonical(program):return sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def analyze(out):
    frozen=base.read(out/'FROZEN_PREDICTIONS.json');aliases=defaultdict(list)
    maxabs=maxl2=0
    for case in frozen['cases']:aliases[case['program_sha256']].append(case)
    rows=[];tcritical={3:4.302652729911275,6:2.570581835636305,9:2.306004135204166}
    for chash,cases in aliases.items():
        means=[]
        for case in cases:
            for rep in range(3):
                native=base.read(out/'validation'/case['id']/f'process-{rep}.json')
                verification=base.read(out/'validation'/case['id']/f'process-{rep}.verification.json')
                means.append(st.mean(native['samples_ns']));maxabs=max(maxabs,verification['max_abs_error'])
                maxl2=max(maxl2,max(s['relative_l2_error'] for s in verification['steps']))
        case=cases[0];pred=case['prediction']['duration_ns'];mean=st.mean(means);sd=st.stdev(means)
        p=base.read(out/'programs'/case['id']/'program.json')
        devices=[sorted({a['device'] for a in p['actions'] if a['kind']=='EXPERT' and a['step']==step}) for step in range(8)]
        rows.append(dict(k=case['k'],locality=case['locality'],policies=[c['policy'] for c in cases],program_sha256=chash,
            predicted_ns=pred,mean_ns=mean,process_means_ns=means,independent_processes=len(means),
            mean_95ci_ns=[mean-tcritical[len(means)]*sd/math.sqrt(len(means)),mean+tcritical[len(means)]*sd/math.sqrt(len(means))],
            ape_percent=100*abs(pred-mean)/mean,devices_by_step=devices,loads=case['prediction']['action_counts'].get('LOAD',0),
            weight_h2d_bytes=case['prediction']['weight_h2d_bytes'],total_h2d_bytes=case['prediction']['total_h2d_bytes'],
            total_d2h_bytes=case['prediction']['total_d2h_bytes'],peak_resident_experts=case['prediction']['peak_resident_experts']))
    selection=[]
    for k in (2,4):
        for locality in ('hot','capacity-sensitive'):
            group=[x for x in rows if x['k']==k and x['locality']==locality]
            chosen=min(group,key=lambda x:(x['predicted_ns'],x['policies'][0]));best=min(group,key=lambda x:x['mean_ns'])
            lookup={policy:row for row in group for policy in row['policies']}
            selection.append(dict(k=k,locality=locality,predicted_choice=chosen['policies'],measured_fastest=best['policies'],
                regret_percent=100*(chosen['mean_ns']/best['mean_ns']-1),
                mixed_lru_vs_cpu_gain_percent=100*(1-lookup['P4']['mean_ns']/lookup['P0']['mean_ns']),
                mixed_lru_vs_gpu_lru_gain_percent=100*(1-lookup['P4']['mean_ns']/lookup['P2']['mean_ns'])))
    result=dict(schema='moe-mixed-shape-addendum-1',created=base.utc(),nominal_conditions=20,distinct_programs=len(rows),
        fresh_validation_processes=60,samples_per_process=10,validation_seed=41003,
        parameters='Unmodified EXPERT/COMBINE cells from frozen lifecycle q1/q32 tables; LOAD/RELEASE anchored to the q32 table for every policy',
        scope='8serial externallysuppliedhostresponseexpertFFNcalls,first4q32then4q1;notfullmodelP/Dhandoff',
        mape_percent=st.mean(x['ape_percent'] for x in rows),max_ape_percent=max(x['ape_percent'] for x in rows),
        max_abs_error=maxabs,max_relative_l2_error=maxl2,rows=rows,selection=selection,
        unverified='No causalprefetch/asynchronousoverlap/fullmodelKVhandoff; deploymentforwardoutputsnotautoregressivelyfed')
    base.save(out/'FINAL_ANALYSIS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','selection')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-campaign',type=Path,default=base.ROOT/'build/moe-policy-lifecycle-20261029-v2')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise ValueError('new output directory required')
    out.mkdir(parents=True);source=a.source_campaign.resolve()
    build=base.read(source/'PROTOCOL.json')['build'];binary=Path(build['binary']['path']);sources=base.source_snapshot(out,build)
    env=dict(os.environ);env['PATH']=os.pathsep.join(build['runtime_dirs'])+os.pathsep+env['PATH']
    env['NVIDIA_TF32_OVERRIDE']='0';env['GGML_CUDA_DISABLE_GRAPHS']='1'
    calibrations={};source_prices={}
    for k in (2,4):
        small=source/'calibration'/f'q1-k{k}'/'calibration.json';large=source/'calibration'/f'q32-k{k}'/'calibration.json'
        c1,c32=base.read(small),base.read(large);prices=dict(c32['prices_ns'])
        prices.update({key:value for key,value in c1['prices_ns'].items() if key not in ('LOAD','RELEASE')})
        calibrations[k]=dict(schema=c32['schema'],config=base.config(k),boundary=c32['boundary'],prices_ns=prices)
        base.save(out/'calibration'/f'k{k}.json',dict(**calibrations[k],
            source_cells=[dict(path=str(x),sha256=base.digest(x)) for x in (small,large)],
            transfer_context='q32LOAD/RELEASEpricesheldconstantacrossall20candidates;nointerpolationormixedvalidationfit'))
        source_prices[str(small)]=base.digest(small);source_prices[str(large)]=base.digest(large)
    base.save(out/'PROTOCOL.json',dict(created=base.utc(),seed=41003,workload='first4q32then4q1,8serialexpertcalls',
        policies=base.POLICIES,slots=[2,4],locality=['hot','capacity-sensitive'],sources=sources,source_calibration_hashes=source_prices,
        request=dict(mode='bare',warmup=3,samples=10,repeats=1),fresh_processes_per_condition=3,
        run_order='3rounds,20conditionlistrotatedby7*round',no_refitting=True,
        boundary='externallysuppliedhostresponsepercall,notKVphasehandofforfullLLMgeneration'))
    cases=[]
    for k in (2,4):
        for locality in ('hot','capacity-sensitive'):
            steps=base.steps(32,locality,41003)[:4]+base.steps(1,locality,41003)[4:]
            for policy,recipe,retention in base.POLICIES:
                cid=f'k{k}-{locality}-{policy}';folder=out/'programs'/cid
                compiled=compile_expert_program(config=base.config(k),steps=steps,seed=41003,
                    recipe=recipe,retention=retention,prices=calibrations[k],resource_guards=base.GUARDS)
                program=compiled['program'];prediction=evaluate_expert_policy(program,calibrations[k])
                base.save(folder/'program.json',program);base.save(folder/'compilation.json',compiled['compilation_evidence'])
                base.save(folder/'prediction.json',prediction);base.save(folder/'reference.json',reference_outputs(program))
                if policy in ('P3','P4'):
                    devices=[{x['device'] for x in program['actions'] if x['kind']=='EXPERT' and x['step']==s} for s in range(8)]
                    if devices!=[{'gpu'}]*4+[{'cpu'}]*4:raise ValueError('currentcostdidnotproducedeclaredmixedtrace')
                cases.append(dict(id=cid,k=k,locality=locality,policy=policy,recipe=recipe,retention=retention,
                    program_sha256=canonical(program),prediction=prediction))
    base.save(out/'FROZEN_PREDICTIONS.json',dict(created=base.utc(),cases=cases,source_calibration_hashes=source_prices))
    with base.measurement_lock():
        print('mixedpredictionsfrozen;60newnativeprocessesstart',flush=True)
        for rep in range(3):
            offset=7*rep
            for case in cases[offset:]+cases[:offset]:
                folder=out/'validation'/case['id'];pdir=out/'programs'/case['id'];program=base.read(pdir/'program.json')
                result=base.invoke(binary,env,pdir/'program.json',dict(mode='bare',warmup=3,samples=10,repeats=1),folder/f'process-{rep}.json')
                verification=verify_result(program,result,expected=base.read(pdir/'reference.json'))
                base.save(folder/f'process-{rep}.verification.json',verification)
                if not verification['numerically_equal']:raise ValueError('mixednumericsfailed:'+case['id'])
                print('mixedvalidation',rep,case['id'],round(st.mean(result['samples_ns'])/1e6,4),'ms',flush=True)
        after={path:base.digest(path) for path in sources};pricesafter={path:base.digest(path) for path in source_prices}
        base.save(out/'SOURCE_UNCHANGED.json',dict(unchanged=after==sources,source_calibrations_unchanged=pricesafter==source_prices,
            before=sources,after=after,source_calibrations_before=source_prices,source_calibrations_after=pricesafter))
        if after!=sources or pricesafter!=source_prices:raise RuntimeError('sourceorpricechangedduringmixedvalidation')
    analyze(out)

if __name__=='__main__':main()
