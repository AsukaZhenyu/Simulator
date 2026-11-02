"""Serial, source-bound H2D/history diagnosis and independent validation.

Diagnostics choose mechanisms, never provide complete-target fitting labels.
All jobs are declared before collection. Failed processes remain in denominators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from profiling.dac_validation import e2a
from profiling.state_calibration import run

ROOT = Path(__file__).resolve().parents[2]
read, write, receipt, sha, require = e2a.read, e2a.write, e2a.receipt, e2a.sha, e2a.require


def prepare(out, binary):
    out, binary = Path(out).resolve(), Path(binary).resolve()
    require(not out.exists(), 'new evidence directory required')
    old = read(ROOT/'build/dac-accuracy-repair-20261026/campaign-v1/manifest.json')
    native = ROOT/'profiling/h2d_history/native/history_probe.cpp'
    paths = [Path(__file__), native, native.parent/'CMakeLists.txt',
             ROOT/'profiling/native/steady_costs/probe.cpp',
             ROOT/'profiling/native/steady_costs/components.hpp',
             ROOT/'profiling/native/physical_stages/cupti_capture.hpp',
             ROOT/'profiling/dac_validation/e2a.py', ROOT/'profiling/state_calibration/run.py']
    paths += list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    require(all(p.is_file() for p in paths), 'missing bound source')
    out.mkdir(parents=True)
    sources = []
    for p in paths:
        target = out/'executed_sources'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
        sources.append(dict(source=receipt(p), archived=receipt(target)))
    target = out/'executed_binary'/binary.name
    target.parent.mkdir(); shutil.copy2(binary, target)
    jobs = []
    # Two previously problematic contexts; actual arena, buffer and exact work.
    for repeat in range(2):
        cases = ((1, 1), (2, 2)) if repeat == 0 else ((2, 2), (1, 1))
        for layers, strategy in cases:
            key = f'l{layers}-h512-p64-s{strategy}-profile'
            p = read(old['programs'][key]['program']['path'])
            target_index = next(i for i, a in enumerate(p['actions'])
                                if a['kind'] == 'COPY' and a['src'] == 'dram'
                                and a['target_id'].endswith(':W_up'))
            for memory in (('pageable', 'registered') if repeat == 0 else ('registered', 'pageable')):
                for mode in ('target-hot', 'full'):
                    for instrument in (('bare', 'stamped', 'cupti') if repeat == 0 else ('cupti', 'stamped', 'bare')):
                        sampling = e2a.request('full' if mode == 'full' else 'action', trials=16)
                        # Match replay and block history in hot/full and bare/observer.
                        sampling['repeats'] = 8
                        sampling['target_action_index'] = target_index if mode == 'target-hot' else None
                        req = dict(schema='h2d-history-request-1', source_sha256=sha(native),
                                   sampling=sampling, expected_process_mask=4294967295,
                                   mode=mode, instrumentation=instrument, memory_mode=memory,
                                   target_action_index=target_index if mode == 'target-hot' else None)
                        jobs.append(dict(id=f'diagnostic-r{repeat}-{key}-{memory}-{mode}-{instrument}',
                                         split='diagnostic', program_key=key, request=req))
            # Read-only source-touch intervention: target interval excludes the
            # intervention; whole-block times are diagnostic, never accuracy data.
            sampling = e2a.request('full', trials=16)
            req = dict(schema='h2d-history-request-1', source_sha256=sha(native), sampling=sampling,
                       expected_process_mask=4294967295, mode='full', instrumentation='cupti',
                       memory_mode='pageable', target_action_index=target_index,
                       history_intervention='source-read')
            jobs.append(dict(id=f'diagnostic-r{repeat}-{key}-source-read',split='diagnostic',program_key=key,request=req))
    # Calibration action measurements have different initial data from validation.
    for repeat in range(3):
        for layers in ((1, 2) if repeat % 2 == 0 else (2, 1)):
            for strategy in ((0, 1, 2) if repeat % 2 == 0 else (2, 1, 0)):
                key = f'l{layers}-h512-p64-s{strategy}-profile'
                req = dict(schema='h2d-history-request-1', source_sha256=sha(native),
                           sampling=e2a.request('full', trials=16), expected_process_mask=4294967295,
                           mode='full', instrumentation='stamped', memory_mode='pageable', target_action_index=None)
                jobs.append(dict(id=f'profile-r{repeat}-{key}', split='profile', program_key=key, request=req))
    for repeat in range(4):
        for layers in ((1, 2) if repeat % 2 == 0 else (2, 1)):
            for strategy in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))[repeat]:
                key = f'l{layers}-h512-p64-s{strategy}-heldout'
                for instrument in (('bare', 'stamped') if repeat % 2 == 0 else ('stamped', 'bare')):
                    req = dict(schema='h2d-history-request-1', source_sha256=sha(native),
                               sampling=e2a.request('full', trials=16), expected_process_mask=4294967295,
                               mode='full', instrumentation=instrument, memory_mode='pageable', target_action_index=None)
                    jobs.append(dict(id=f'heldout-r{repeat}-{key}-{instrument}', split='heldout', program_key=key, request=req))
    m = dict(schema='h2d-history-campaign-1', prepared_at_utc=e2a.utc(), binary=receipt(target),
             sources=sources, programs={k:v for k,v in old['programs'].items() if k.startswith('l')},
             runtime_dirs=old['runtime_dirs'], runtime=old['runtime'], jobs=jobs,
             baseline_profile=receipt(ROOT/'build/dac-accuracy-repair-20261026/campaign-v1/action-profile-frozen.json'),
             baseline_context=old['context'],
             protocol=dict(counts={s:sum(j['split']==s for j in jobs) for s in ('diagnostic','profile','heldout')},
                           quality=e2a.QUALITY, retries=0, timeout_seconds=300,
                           repeats=8, trials=16, warmup_min_ms=500,
                           observer_history='same replays, trials and block preparation; separate adjacent processes',
                           target_totals_used_for_fit=False, future_target_state_used=False,
                           boundary='inclusive completed GGML action; CPU staging not inferred from copy API time'))
    write(out/'manifest.json', m)
    return dict(out=str(out), **m['protocol'])


def check(m):
    for s in m['sources']:
        require(sha(s['source']['path'])==s['source']['sha256'], 'bound live source changed')
        require(sha(s['archived']['path'])==s['archived']['sha256'], 'archived source changed')
    for r in [m['binary'], *m['runtime']]:
        require(sha(r['path'])==r['sha256'], 'binary/runtime changed')


def collect(out, split):
    out=Path(out).resolve(); m=read(out/'manifest.json'); check(m)
    require(split in ('diagnostic','profile','heldout'), 'unknown split')
    barrier = None
    if split == 'heldout':
        barrier = read(out/'frozen-barrier.json')
        for name,h in barrier['files'].items(): require(sha(out/name)==h, 'frozen model/prediction changed')
    start = out/(split+'-started.json')
    if not start.exists(): write(start, dict(at=e2a.utc(), barrier=barrier))
    executions=[]
    with run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists(): executions.append(read(folder/'execution.json')); continue
            folder.mkdir(parents=True, exist_ok=False); binding=m['programs'][job['program_key']]
            req, result = folder/'request.json', folder/'result.json'; write(req,job['request'])
            for r in (binding['program'],binding['reference']): require(sha(r['path'])==r['sha256'], 'input changed')
            command=[m['binary']['path'],binding['program']['path'],str(req),str(result)]
            execution=dict(job=job, started_at_utc=e2a.utc(), status='failed', command=command,barrier=barrier)
            process=None; handles=[]
            try:
                env=run.child_environment(m['runtime_dirs']); env['OPENBLAS_NUM_THREADS']='1'
                process,stdout,stderr,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295)
                handles=[stdout,stderr]; execution.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=m['protocol']['timeout_seconds'])
                execution.update(returncode=code, process_exited_verified=True, forced_termination=False)
                require(code==0,'native child failed; retained without retry')
                native=read(result); require(native['request']==job['request'],'request mismatch')
                require(native['process_id']==process.pid,'PID mismatch')
                require(native['source_sha256']==job['request']['source_sha256'],'source mismatch')
                require(native['program_sha256']==read(binding['program']['path'])['program_sha256'],'program mismatch')
                require(native['native_repeat_consistency']['passed'],'fixed-input repeat mismatch')
                proof=native['verification_native']; oracle=read(binding['reference']['path'])
                require(proof['copy_values_equal'],'native full byte copy equality failed')
                import numpy as np
                require(set(proof['ports'])==set(oracle),'independent port inventory mismatch')
                for tid,values in oracle.items():
                    a,b=np.asarray(proof['ports'][tid]),np.asarray(values)
                    require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent numeric mismatch '+tid)
                require(len(native['trials'])==job['request']['sampling']['trials'],'trial count mismatch')
                execution.update(status='completed',native=receipt(result),full_oracle_passed=True)
            except BaseException as error:
                execution['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                    execution['forced_termination']=True
                if process is not None: execution.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for h in handles: h.close()
                execution['completed_at_utc']=e2a.utc(); write(folder/'execution.json',execution)
            executions.append(execution)
            print(json.dumps(dict(job=job['id'],status=execution['status'],error=execution.get('error'))),flush=True)
    path=out/(split+'-executions.json')
    if not path.exists():write(path,executions)
    return dict(split=split, declared=len(executions), completed=sum(e['status']=='completed' for e in executions))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('prepare','diagnostic','profile','heldout'))
    p.add_argument('--out',required=True); p.add_argument('--binary'); a=p.parse_args()
    result=prepare(a.out,a.binary) if a.command=='prepare' else collect(a.out,a.command)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
