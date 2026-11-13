"""Connected Dense depth validation; action calibration precedes holdout."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from profiling import host
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
# Ten timed blocks per process, and an estimate that drops the highest and the
# lowest of them.  The warmup is unchanged (twenty iterations and 500 ms before
# any block is timed, enforced by the probe); what changes is how the ten blocks
# become one number, at the process level and at the action level alike.
TRIALS=10


def trimmed(values):
    """Mean of the ten block values after the highest and the lowest are dropped."""
    ordered=sorted(values)
    require(len(ordered)>=3,'a trimmed mean needs at least three blocks')
    kept=ordered[1:-1]
    return Fraction(sum(Fraction(v) for v in kept),len(kept))


# The working-set law.  One exponent, shared by every coordinate, says how a
# coordinate's price grows with the bytes a program touches before its timed
# region; every stored price is then that coordinate's level at one reference
# working set, and a program's total is scaled by (ws/ws_ref)**theta.  It is
# fitted on the profile programs only, so it cannot see a target duration, and
# it cannot see the held-out split either.
MINIMUM_SHARED_COORDINATES=10


# Which statistic a coordinate's per-program price is.  The default is the mean,
# which is the estimator every frozen table so far uses.  `median` exists for one
# declared experiment: a small copy's latency is bimodal inside a single program
# (about 20 us with the device idle, several hundred behind queued work), so a
# per-program mean lands between the modes and describes neither.  The switch is
# read from the environment so the default path stays bit-identical.
PRICE_ESTIMATOR=os.environ.get('DEPTH_PRICE_ESTIMATOR','mean')


def mean_prices(prices):
    """{coordinate: {program: mean per-action ns}} from per-action samples."""
    means={}
    for h,per in prices.items():
        row={}
        for p,values in per.items():
            if not values:continue
            if PRICE_ESTIMATOR=='median':
                price=Fraction(statistics.median(values)).limit_denominator(10**9)
            else:
                price=sum((Fraction(v) for v in values),Fraction(0))/len(values)
            if price>0:row[p]=float(price)
        if row:means[h]=row
    return means


def reference_levels(means,x,exponent):
    """Coordinate prices moved to the reference working set by one exponent.

    A coordinate averaged over several programs keeps the geometric mean of its
    divided prices, which is what a least-squares fit with one free intercept
    per coordinate estimates; a coordinate measured in one program is divided
    by that program's own factor.
    """
    levels={};singletons=0
    for h,row in means.items():
        if len(row)>=2:
            levels[h]=math.exp(statistics.mean(math.log(v)-exponent*x[p] for p,v in row.items()))
        else:
            p=next(iter(row));levels[h]=row[p]/math.exp(exponent*x[p]);singletons+=1
    return levels,singletons


def working_set_law(prices,working_set,profile_programs,*,minimum_shared=MINIMUM_SHARED_COORDINATES):
    """One log-log slope for every coordinate, and levels at a reference size.

    For coordinate h and profile program p,

        log price(h, p) = log level(h) + theta * log(ws(p) / ws_ref),

    with ws_ref the median working set of the profile programs and one free
    intercept per coordinate, so the estimate of theta is the within-coordinate
    demeaned least squares of log price on log working set.

    `prices` maps coordinate digest to {program key: [per-action ns]} and
    `working_set` maps program key to the bytes the program touches before its
    timed region.  Refuses rather than invents: no exponent is reported when the
    profile programs share one working set (the slope is not identifiable),
    when too few coordinates are shared by two programs (the slope would be one
    program's accident), or when the estimate is not finite.
    """
    programs=sorted(set(profile_programs))
    require(len(programs)>=2,'a working-set law needs at least two profile programs')
    for p in programs:require(p in working_set and working_set[p]>0,'working set unknown for profile program '+p)
    reference=statistics.median([working_set[p] for p in programs])
    x={p:math.log(working_set[p]/reference) for p in sorted(working_set)}
    means=mean_prices(prices)
    numerator=denominator=0.0;shared=0
    for row in means.values():
        if len(row)<2:continue
        shared+=1
        mx=statistics.mean(x[p] for p in row);my=statistics.mean(math.log(v) for v in row.values())
        for p,v in row.items():
            numerator+=(x[p]-mx)*(math.log(v)-my);denominator+=(x[p]-mx)**2
    require(shared>=minimum_shared,'too few coordinates shared across profile programs: %d' % shared)
    require(denominator>0,'the profile programs share one working set; no exponent is identifiable')
    theta=numerator/denominator
    require(math.isfinite(theta),'the working-set exponent is not finite')
    levels,singletons=reference_levels(means,x,theta)
    return dict(exponent=theta,reference_bytes=reference,factors={p:math.exp(theta*x[p]) for p in x},
        levels=levels,shared=shared,singletons=singletons,programs=len(programs))


def levels_at_reference(prices,exponent,working_set,reference_bytes):
    """The same division for cells measured after the law was fitted."""
    x={p:math.log(working_set[p]/reference_bytes) for p in sorted(working_set)}
    means=mean_prices(prices)
    return reference_levels(means,x,exponent)


# The five generated Executor fragments this probe is compiled against, and the
# hashes the published campaign ran. They ship in the repository because the two
# properties that make them usable -- the binary initial-asset transport and the
# pre-existing allocation policy -- exist only in the historical highlevel probe,
# so regenerating from today's source would silently be a different instrument.
FRAGMENTS=Path(__file__).resolve().parent/'generated'
FRAGMENT_MANIFEST=FRAGMENTS/'source-manifest.json'


def executor_fragments(candidates=None):
    """Pick a directory holding exactly the recorded fragments.

    The campaign's own build directory and the shipped copy are interchangeable
    only when the bytes agree, and the recorded hashes settle that: a directory
    that differs is refused rather than measured under the same name.
    """
    recorded=read(FRAGMENT_MANIFEST)['generated_files']
    if candidates is None:
        candidates=((ROOT/'build/depth-validation-20261026/generated','campaign-build'),
                    (FRAGMENTS,'repository'))
    for directory,label in candidates:
        if not directory.is_dir():continue
        present={p.name:sha(p) for p in directory.glob('*.inc')}
        if present==recorded:return directory,label,recorded
    require(False,'no directory holds the recorded Executor fragments; '
        'run python -m profiling.depth_validation.generate_sources --check')


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
    generated,generated_source,generated_files=executor_fragments()
    source_paths=[Path(__file__),ROOT/'profiling/depth_validation/programs.py',
        ROOT/'profiling/depth_validation/native/depth_probe.cpp',
        ROOT/'profiling/depth_validation/native/initial_assets.hpp',
        ROOT/'profiling/depth_validation/native/CMakeLists.txt',
        ROOT/'profiling/depth_validation/generate_sources.py',
        ROOT/'profiling/highlevel_accuracy/bridge.py',ROOT/'profiling/highlevel_accuracy/multilayer.py',
        ROOT/'modeling/llm_infer_model/tensor/spec.py',ROOT/'modeling/llm_infer_model/m2/causal.py',
        ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
        ROOT/'modeling/llm_infer_model/performance/execution_history.py',
        ROOT/'modeling/llm_infer_model/performance/history_conditioned_costs.py']
    source_paths+=list(generated.glob('*.inc'))
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
                    sampling=e2a.request('full',trials=TRIALS),
                    expected_process_mask=host.process_affinity_mask(),
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
            heldout_processes=sum(j['split']=='heldout' for j in jobs),trials=TRIALS,replays=8,
            estimator='trimmed mean of the ten block times after dropping the highest and the lowest',
            mean_ape_gate=.1,p90_ape_gate=.2,raw_CV_is_accuracy_gate=False,timeout_seconds=600,
            source_and_parameter_freeze_before_heldout=True,no_retries=True,
            numerical_atol=5e-5,numerical_rtol=5e-5,capacity_or_actual_eviction_qualified=False))
    m['protocol']['fresh_export_diagnostic_retention']='initial state only; all original snapshot calculations and causal transitions retained'
    m['generated_fragments']=dict(directory=str(generated),resolved_from=generated_source,files=generated_files,
        instrument='historical highlevel probe plus the binary initial-asset transport; regenerated by generate_sources.py')
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
    require(n['process_affinity']['final_mask']==job['request']['expected_process_mask'] and n['main_thread_affinity']['restored'],'affinity differs')
    require(n['native_repeat_consistency']['passed'] and n['verification_native']['copy_values_equal'],'native repeated/copy output failed')
    got=n['verification_native']['ports'];require(set(got)==set(oracle),'reference port inventory differs')
    for tid,expected in oracle.items():
        a,b=np.asarray(got[tid]),np.asarray(expected)
        require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,atol=5e-5,rtol=5e-5),'numeric mismatch '+tid)
    require(n['initial_binary_buffers_verified']==len(program['initial_assets']['tensors']),'binary initialization inventory differs')
    hz=n['clock']['frequency_hz'];require(len(n['trials'])==TRIALS,'trial count differs')
    for r in n['trials']:
        require(r['repeats']==8 and r['end_tick']>r['begin_tick'],'QPC interval differs')
        require(abs((r['end_tick']-r['begin_tick'])*1e9/hz-r['wall_ns'])<=1,'QPC duration differs')
    return dict(passed=True,ports=len(got),blocks=TRIALS)


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
                process,a,b,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',
                                                 job['request']['expected_process_mask'])
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


def fit_consistency(per_program, minimum_samples=8):
    """Per-coordinate agreement across the profile programs that share it.

    One table serves every placement, so a coordinate that two programs both
    produce is averaged. That is only harmless if the same coordinate costs the
    same in both. This report says whether it does, before any target process
    runs: a large spread means the pooled price cannot be right for all of the
    programs it is applied to, and the heldout cases for those programs will
    miss however good the fit looks in aggregate.

    `per_program` maps coordinate digest to {program key: [sample ns]}. Means
    over fewer than `minimum_samples` samples are dropped rather than reported,
    because a single action occurrence is not a measurement of a coordinate.
    """
    rows=[]
    for h,per in per_program.items():
        means={p:sum(v)/len(v) for p,v in per.items() if len(v)>=minimum_samples}
        if len(means)<2:continue
        low,high=min(means.values()),max(means.values())
        rows.append(dict(key=h,programs=sorted(means),low_ns=low,high_ns=high,spread=high/low))
    rows.sort(key=lambda r:(-r['spread'],r['key']))
    spreads=sorted(r['spread'] for r in rows)
    return dict(schema='depth-fit-consistency-1',frozen_at_utc=e2a.utc(),
        coordinates=len(per_program),shared_coordinates=len(rows),
        max_spread=rows[0]['spread'] if rows else 1.0,
        median_spread=statistics.median(spreads) if spreads else 1.0,
        rows=rows,target_time_used=False)


def freeze(out):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    require(not(out/'heldout-started.json').exists(),'fit after heldout is forbidden')
    executions=read(out/'profile-executions.json')
    require(len(executions)==m['protocol']['profile_processes'] and all(e['status']=='completed' for e in executions),'missing profile evidence')
    scope=digest(m['context']);groups={};evidence=defaultdict(list);programs={};native_rows=[];static_coords={};seen_evidence=defaultdict(set);per_program=defaultdict(lambda:defaultdict(list))
    static_prices=defaultdict(lambda:defaultdict(list));conditioned_prices=defaultdict(lambda:defaultdict(list))
    def add(coord,value,source,key,program):
        require(isinstance(value,Fraction) or (type(value) is int and value>=0),'nonnegative native action ns required')
        group=groups.setdefault(key,dict(coordinate=coord,sum_ns=Fraction(0),count=0));group['sum_ns']+=value;group['count']+=1
        (conditioned_prices if coord['condition'] else static_prices)[key][program].append(value)
        if source['native']['sha256'] not in seen_evidence[key]:
            evidence[key].append(source);seen_evidence[key].add(source['native']['sha256'])
    by_block=defaultdict(dict);source_of={}
    for e in executions:
        key=e['job']['program_key']
        if key not in programs:
            programs[key]=read(m['programs'][key]['program']['path'])
            coordinates=[dict(scope_sha256=scope,semantic=semantic_action(programs[key],i),condition={}) for i in range(len(programs[key]['actions']))]
            static_coords[key]=[(c,digest(c)) for c in coordinates]
        n=read(e['native']['path']);native_rows.append((e,key))
        source_of[key]=dict(native=e['native'],split='profile')
        for number,block in enumerate(n['trials']):
            for r in block['action_intervals']:
                by_block[(key,r['invocation'],r['action_index'])][number]=r['wall_ns']
                coord,h=static_coords[key][r['action_index']]
                per_program[h][key].append(r['wall_ns'])
    for samples in by_block.values():require(len(samples)==TRIALS,'block count differs from the declared trials')
    def rational(value):return str(Fraction(value).limit_denominator(10**9))
    def entries():
        # Every stored price is the coordinate's level at the reference
        # working set, so one table no longer carries whichever program
        # happened to be pooled into it.
        return [dict(key=k,coordinate=g['coordinate'],
                     mean_ns=rational(levels.get(k,Fraction(g['sum_ns'],g['count']))),
                     sample_count=g['count'],evidence=evidence[k]) for k,g in sorted(groups.items())]
    # One observation per action per block: the highest and the lowest block are
    # dropped before the estimate, exactly as at the process level.  A static
    # coordinate is shared by the eight replays, so their trimmed values are
    # averaged; a history-conditioned coordinate belongs to one replay.
    static_by_replay=defaultdict(dict)
    for (key,invocation,index),samples in by_block.items():
        static_by_replay[(key,index)][invocation]=trimmed(list(samples.values()))
    for (key,index),per_replay in sorted(static_by_replay.items()):
        coord,h=static_coords[key][index]
        add(coord,Fraction(sum(per_replay.values()),len(per_replay)),source_of[key],h,key)
    working_set={k:int(b['initial_assets']['bytes']) for k,b in m['programs'].items()}
    law=working_set_law(static_prices,working_set,sorted(programs))
    levels=dict(law['levels'])
    # The logical history that constructs the conditioned cells is advanced
    # with the same levels the predictor commits, so a cell is looked up
    # under the conditions it was measured under.
    static={digest(g['coordinate']['semantic']):Fraction(levels.get(k,Fraction(g['sum_ns'],g['count']))).limit_denominator(10**9)
        for k,g in groups.items()}
    traces={}
    for key,p in programs.items():
        history=SerialExecutionHistory(p,HistoryScope(scope));durations=[static[digest(semantic_action(p,i))] for i in range(len(p['actions']))]
        for i,d in enumerate(durations):history.commit(i,d)
        trace={}
        for invocation in range(8):
            for i,d in enumerate(durations):
                c=response_coordinate(history.snapshot(i),FIELDS);trace[invocation,i]=(c,digest(c));history.commit(i,d)
        traces[key]=trace
    for (key,invocation,index),samples in by_block.items():
        coord,h=traces[key][invocation,index]
        if coord['condition']:add(coord,trimmed(list(samples.values())),source_of[key],h,key)
    conditioned,singleton_cells=levels_at_reference(conditioned_prices,law['exponent'],working_set,law['reference_bytes'])
    levels.update(conditioned)
    unleveled=sum(1 for k in groups if k not in levels)
    profile=dict(schema=SCHEMA,scope_sha256=scope,scope=m['context'],condition_fields=FIELDS,bucketing=BUCKETING,
        initialization=INITIALIZATION,unmodeled_reset_accesses=True,entries=entries(),frozen_at_utc=e2a.utc(),
        target_time_used_for_fit=False,future_target_state_used=False,calibration_label='inclusive per-action intervals only',
        working_set_law=dict(rule='one shared log-log working-set exponent with a level per coordinate; every stored price is that level at reference_bytes',
            exponent=law['exponent'],reference_bytes=law['reference_bytes'],
            shared_coordinates=law['shared'],singleton_coordinates=law['singletons']+singleton_cells,
            unleveled_coordinates=unleveled,profile_programs=law['programs'],target_time_used_for_fit=False))
    table=HistoryConditionedResponseTable.from_profile(profile);predictions=[]
    for key,b in m['programs'].items():
        if b['split']!='heldout':continue
        p=read(b['program']['path']);costs={}
        factor=Fraction(law['factors'][key]).limit_denominator(10**9)
        for name,enabled in (('context_static',False),('history_conditional',True)):
            prediction=table.predict_blocks(p,blocks=1,repeats=8,use_history=enabled)
            # The table prices one reference working set; this program's own
            # size enters as one factor for every action it runs.
            costs[name]=str(Fraction(prediction['duration_ns'])*factor)
        predictions.append(dict(program_key=key,layers=b['layers'],phase=b['phase'],strategy=b['strategy'],
            working_set_bytes=int(b['initial_assets']['bytes']),working_set_factor=str(factor),costs=costs))
    consistency=fit_consistency(per_program)
    write(out/'model-frozen.json',profile);write(out/'predictions-frozen.json',predictions)
    write(out/'fit-consistency.json',consistency)
    write(out/'frozen-barrier.json',dict(frozen_at_utc=e2a.utc(),files={n:sha(out/n) for n in ('model-frozen.json','predictions-frozen.json','fit-consistency.json')},
        all_target_coordinates_known=True,target_outcomes_read=False))
    return dict(response_cells=len(profile['entries']),heldout_cases=len(predictions),
        shared_coordinates=consistency['shared_coordinates'],max_coordinate_spread=consistency['max_spread'],
        working_set_exponent=law['exponent'],working_set_reference_bytes=law['reference_bytes'])


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
                half=len(values)//2
                row.update(mean_ns=float(trimmed(values)),trial_ns=values,cv=statistics.stdev(values)/statistics.mean(values),
                    late_over_early=statistics.mean(values[half:])/statistics.mean(values[:half])-1,
                    estimator='trimmed mean of ten blocks, highest and lowest dropped')
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
        working_set_law=read(out/'model-frozen.json').get('working_set_law'),
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
