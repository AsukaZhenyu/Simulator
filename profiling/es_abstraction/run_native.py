"""Blocked independent native acquisition of frozen E/S candidate programs."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import random
import statistics as st
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from profiling.moe_policy_validation import campaign as base
from profiling.moe_policy_validation.reference import reference_outputs, verify_result
from profiling.es_abstraction.calibration import native_env
from experiments.es_abstraction.moe import load_program

def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def compress_json(path):
    path=Path(path);dest=path.with_suffix(path.suffix+'.gz')
    original_sha=base.digest(path)
    with path.open('rb') as src,gzip.open(dest,'xb',compresslevel=6) as dst:
        while block:=src.read(1024**2):dst.write(block)
    if gzip.decompress(dest.read_bytes())!=path.read_bytes():raise RuntimeError('lossless compression failed')
    base.save(dest.with_suffix('.receipt.json'),dict(original_sha256=original_sha,compressed_sha256=base.digest(dest),
        uncompressed_bytes=path.stat().st_size,compressed_bytes=dest.stat().st_size))
    path.unlink()
    return dest

def prepare_refs(folder, manifest):
    for w in manifest['workloads']:
        dest=folder/'native-references'/f"{w['id']}.json.gz"
        if dest.exists():continue
        p=next(x for x in manifest['programs'] if x['workload_id']==w['id'])
        program=load_program(folder,p['id'])
        reference=reference_outputs(program)
        raw=dest.with_suffix('');base.save(raw,reference);compress_json(raw)
        print('reference',w['id'],flush=True)

def run(folder, *, rounds=5, start_round=0, workload_ids=None, pilot=False):
    folder=Path(folder).resolve()
    manifest=base.read(folder/'FROZEN_PREDICTIONS.json')
    build=base.read(base.ROOT/'build/moe-policy-native-20261029/BUILD_COMPLETE.json')
    binary=Path(build['binary']['path']);env=native_env(build)
    for row in [build['binary']]+build['runtimes']:
        if base.digest(row['path'])!=row['sha256']:raise ValueError('runtime changed')
    for path,digest in manifest['source_hashes'].items():
        if base.digest(path)!=digest:raise ValueError('frozen source changed:'+path)
    programs=[p for p in manifest['programs'] if not workload_ids or p['workload_id'] in workload_ids]
    if pilot:
        # Different supplied inputs/weights, used only for wall-time and correctness.
        p=next(x for x in programs if x['geometry']=='real' and any(z.startswith('E_GPU') for z in x['policy_ids']))
        program=load_program(folder,p['id']);program['seed']=33030
        pilotdir=folder/'pilot';path=pilotdir/'program.json';base.save(path,program)
        before=time.perf_counter()
        with base.measurement_lock():
            r=base.invoke(binary,env,path,dict(mode='bare',warmup=3,samples=10,repeats=1),pilotdir/'process.json')
        v=verify_result(program,r);base.save(pilotdir/'verification.json',v)
        result=dict(wall_seconds=time.perf_counter()-before,sequence_mean_ns=st.mean(r['samples_ns']),
                    numerically_equal=v['numerically_equal'],seed=33030,not_used_to_fit=True)
        base.save(pilotdir/'PILOT.json',result);compress_json(pilotdir/'process.json');print(result,flush=True)
        return
    prepare_refs(folder,manifest)
    receipt=folder/'NATIVE_PROTOCOL.json'
    if not receipt.exists():base.save(receipt,dict(created=base.utc(),manifest_sha256=base.digest(folder/'FROZEN_PREDICTIONS.json'),
        build=build,request=dict(mode='bare',warmup=3,samples=10,repeats=1),initial_rounds=5,max_rounds=10,
        adaptive_rule='after round5, if primary gain CI width >10pp add5 rounds for every unique program in affected workload',
        order='global blocked round, deterministic shuffle seed51030+round; distinct processes; exact aliases measured once',
        lossless_compression=True,measurement_sources={str(Path(__file__).resolve()):base.digest(__file__)}))
    with base.measurement_lock():
        for rep in range(start_round,rounds):
            order=list(programs);random.Random(51030+rep).shuffle(order)
            for index,p in enumerate(order):
                out=folder/'validation'/p['id']/f'process-{rep}.json'
                verified=out.with_suffix('.verification.json')
                if verified.exists() and (out.exists() or out.with_suffix('.json.gz').exists()):continue
                program=load_program(folder,p['id'])
                temp=folder/'native-current-program.json'
                if temp.exists():temp.unlink()
                base.save(temp,program)
                expected=read_gzip(folder/'native-references'/f"{p['workload_id']}.json.gz")
                before=time.perf_counter()
                r=base.read(out) if out.exists() else base.invoke(binary,env,temp,
                    dict(mode='bare',warmup=3,samples=10,repeats=1),out)
                v=verify_result(program,r,expected=expected)
                v.update(process_mean_ns=st.mean(r['samples_ns']),process_sample_cv=(st.stdev(r['samples_ns'])/st.mean(r['samples_ns'])),
                         native_wall_seconds=time.perf_counter()-before,program_sha256=p['program_sha256'],
                         round=rep,workload_id=p['workload_id'],gpu_weight_allocated_bytes=r['gpu_weight_allocated_bytes'],
                         workspace_bytes=r['workspace_bytes'],graph_port_allocated_bytes=r['graph_port_allocated_bytes'])
                base.save(verified,v);compress_json(out);temp.unlink()
                print('native',rep,index+1,'/',len(order),p['id'],p['workload_id'],round(v['process_mean_ns']/1e6,3),'ms',flush=True)
                if not v['numerically_equal']:raise RuntimeError('independent numerical failure retained:'+p['id'])
    after={p:base.digest(p) for p in manifest['source_hashes']}
    source=folder/f'NATIVE_SOURCE_UNCHANGED_{start_round}_{rounds}.json'
    if not source.exists():base.save(source,dict(unchanged=after==manifest['source_hashes'],after=after))
    if after!=manifest['source_hashes']:raise ValueError('source drift during acquisition')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--start-round',type=int,default=0)
    ap.add_argument('--workloads',type=Path);ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args();run(a.campaign,rounds=a.rounds,start_round=a.start_round,
                         workload_ids=base.read(a.workloads) if a.workloads else None,pilot=a.pilot)

if __name__=='__main__':main()
