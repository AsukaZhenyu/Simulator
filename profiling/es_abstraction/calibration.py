"""Acquire exact lifecycle component cells before E/S validation.

Uses the existing native executor. Measurement and parameters only; shared
modeling retains all duration calculations. No validation times are read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import statistics as st
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from profiling.moe_policy_validation import campaign as base
from profiling.moe_policy_validation.reference import reference_outputs, verify_result
from llm_infer_model.performance.expert_policy_response import action_price_keys

SEED = 31030

def config(geometry, k):
    h, i = (256, 512) if geometry == 'small' else (2048, 1024)
    return dict(hidden=h, intermediate=i, experts=8, gpu_slots=k, threads=1, cuda_device=0)

def make_program(geometry, k, light, mode, pattern):
    c = config(geometry, k)
    rng = np.random.default_rng(SEED + light)
    steps, actions, owners, age = [], [], {}, {}
    for t in range(4):
        heavy_pair, light_pair = ((0, 1), (2, 3)) if pattern == 'fixed' else (
            (2*(t % 4), 2*(t % 4)+1), (2*((t+1) % 4), 2*((t+1) % 4)+1))
        routes = [list(heavy_pair)] * 32 + [list(light_pair)] * light
        steps.append(dict(input=rng.uniform(-.9, .9, (32+light, c['hidden'])).astype(np.float32).tolist(),
                          routes=routes, scores=[[.6, .4] for _ in routes]))
        for e in sorted(set(x for row in routes for x in row)):
            if mode != 'cpu':
                if e not in owners:
                    if len(owners) == k:
                        victim = min(owners, key=lambda x: (age[x], x))
                        actions.append(dict(kind='RELEASE', expert=victim, slot=owners.pop(victim)))
                    slot = min(set(range(k))-set(owners.values()))
                    owners[e] = slot
                    actions.append(dict(kind='LOAD', expert=e, slot=slot))
                actions.append(dict(kind='EXPERT', step=t, expert=e, device='gpu', slot=owners[e]))
                age[e] = len(actions)
                if mode == 'release':
                    actions.append(dict(kind='RELEASE', expert=e, slot=owners.pop(e)))
            else:
                actions.append(dict(kind='EXPERT', step=t, expert=e, device='cpu'))
        actions.append(dict(kind='COMBINE', step=t))
    return dict(schema='moe-expert-policy-native-1', config=c, seed=SEED,
                resource_guards=base.GUARDS, steps=steps, actions=actions)

def native_env(build):
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(build['runtime_dirs']) + os.pathsep + env['PATH']
    env.update(NVIDIA_TF32_OVERRIDE='0', GGML_CUDA_DISABLE_GRAPHS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    return env

def acquire(out, geometry, lights):
    out = Path(out).resolve()
    build = base.read(base.ROOT/'build/moe-policy-native-20261029/BUILD_COMPLETE.json')
    for row in [build['binary']] + build['runtimes']:
        if base.digest(row['path']) != row['sha256']:
            raise ValueError('build/runtime hash mismatch')
    env = native_env(build)
    for k in (2, 4):
        dest = out/geometry/f'k{k}'
        final = dest/'calibration.json'
        if final.exists():
            print('existing calibration retained', final, flush=True)
            continue
        cells, evidence, refs = defaultdict(list), [], {}
        for light in lights:
            for pattern in ('fixed', 'rotate'):
                for mode in ('cpu', 'release', 'lru'):
                    p = make_program(geometry, k, light, mode, pattern)
                    folder = dest/f'q{light}-{pattern}-{mode}'
                    path = folder/'program.json'
                    if not path.exists(): base.save(path, p)
                    keys = action_price_keys(p)
                    # The arithmetic oracle is policy/slot-independent.
                    refkey = light, pattern
                    if refkey not in refs: refs[refkey] = reference_outputs(p)
                    for rep in range(2):
                        raw = folder/f'process-{rep}.json'
                        before = time.perf_counter()
                        r = base.read(raw) if raw.exists() else base.invoke(Path(build['binary']['path']), env, path,
                            dict(mode='stamped', warmup=3, samples=20, repeats=1), raw)
                        v = verify_result(p, r, expected=refs[refkey])
                        vp = raw.with_suffix('.verification.json')
                        if not vp.exists(): base.save(vp, v)
                        if not v['numerically_equal']: raise RuntimeError('calibration numerical failure')
                        values = defaultdict(list)
                        for sample in r['action_samples_ns']:
                            if len(sample) != len(keys): raise ValueError('stamp length differs')
                            for key, value in zip(keys, sample): values[key].append(value)
                        means = {key: st.mean(vals) for key, vals in values.items()}
                        for key, value in means.items(): cells[key].append(value)
                        evidence.append(dict(path=str(raw), sha256=base.digest(raw), component_means_ns=means,
                                             sequence_mean_ns=st.mean(r['samples_ns'])))
                        print('calibrated', geometry, k, light, pattern, mode, rep, round(time.perf_counter()-before, 2), 's', flush=True)
        calibration = dict(schema='moe-expert-policy-calibration-1', config=config(geometry,k),
            boundary='serial_host_expert_response_v1', seed=SEED, created=base.utc(),
            prices_ns={key: round(st.mean(vals)) for key, vals in cells.items()},
            process_category_means_ns=dict(cells), evidence=evidence,
            context='Independent four-call lifecycle, cold logical cache, fixed/rotating heavy32/lightq, CPU/GPU-release/GPU-LRU equally weighted per-process category means; no validation fitting')
        base.save(final, calibration)
        print('FROZEN', geometry, k, calibration['prices_ns'], flush=True)

def select_q(out):
    tables = {k: base.read(Path(out)/'small'/f'k{k}'/'calibration.json')['prices_ns'] for k in (2,4)}
    rows = []
    for q in (4,8,16):
        coords = {str(k): dict(C=p[f'EXPERT:cpu:q={q}'], G=p[f'EXPERT:gpu:q={q}'], L=p['LOAD']) for k,p in tables.items()}
        eligible = all(v['G'] < v['C'] < v['G']+v['L'] for v in coords.values())
        distance = sum(abs((v['C']-v['G'])/v['L']-.5) for v in coords.values())
        rows.append(dict(q=q, eligible=eligible, distance=distance, coordinates=coords))
    eligible = [x for x in rows if x['eligible']]
    q = min(eligible, key=lambda x: (x['distance'],x['q']))['q'] if eligible else 16
    record = dict(q_b=q, shared_crossover_found=bool(eligible), candidates=rows,
                  rule='Protocol fixed before acquisition; same q retained across K and real-geometry transfer', created=base.utc())
    path = Path(out)/'BOUNDARY_SELECTION.json'
    if path.exists():
        if base.read(path)['q_b'] != q: raise ValueError('frozen q changed')
    else: base.save(path, record)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--geometry',choices=('small','real'),required=True)
    a=ap.parse_args()
    lights=(1,4,8,16) if a.geometry=='small' else tuple(sorted({1,base.read(a.out/'BOUNDARY_SELECTION.json')['q_b']}))
    with base.measurement_lock(): acquire(a.out,a.geometry,lights)
    if a.geometry=='small': print(select_q(a.out),flush=True)

if __name__=='__main__': main()
