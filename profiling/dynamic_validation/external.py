"""Controlled offered-work plans and independent continuous Dense validation.

Native actor threads execute real fixed GGML work. Their requested ON/OFF bits
are not percentages of available service. No target-inclusive future telemetry
is an input to prediction, and complete target time is not a fitting label.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess

from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from profiling.dynamic_validation import idle_campaign

ROOT=Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=idle_campaign.read,idle_campaign.write,idle_campaign.sha,idle_campaign.receipt,idle_campaign.require
e2a,run=idle_campaign.e2a,idle_campaign.run
WORKLOADS={'gpu_dense':(1,1),'hybrid_dense':(2,2)}
PLANS=('baseline','cpu_on_off','gpu_on_off','cpu_gpu_staggered')


def component(device):
    value=dict(format_version='steady-component-program-1',kind='matmul',device=device,threads=1,
        cuda_device=0,seed=2026102631 if device=='cpu' else 2026102632,
        shape=dict(k=256,m=256,n=256),allocation_budget_bytes=64<<20)
    value['program_sha256']=digest(value)
    return value


def plans(split,scenario,round_id):
    """Schedules are fixed before collection; pulse times never read a target."""
    require(split in ('profile','heldout') and scenario in PLANS,'unknown controlled plan')
    if scenario=='baseline':return [],[]
    if split=='profile':
        shift=250*round_id
        cpu=[[-2000,4000+shift],[8000+shift,14000+shift]]
        gpu=[[-2000,6000+shift],[10000+shift,16000+shift]]
    else:
        cpu=[[5000,11000]];gpu=[[8000,14000]]
        if scenario=='cpu_gpu_staggered':cpu=[[3000,9000]];gpu=[[6000,15000]]
    if scenario=='cpu_on_off':gpu=[]
    if scenario=='gpu_on_off':cpu=[]
    return cpu,gpu


def state_at(request,relative_ns):
    """Declared offered state at a model/observation clock, not measured load."""
    return tuple(int(any(a*1000000<=relative_ns<b*1000000 for a,b in request[name]['intervals_ms']))
                 for name in ('cpu_actor','gpu_actor'))


def boundaries(request):
    return sorted({x*1000000 for name in ('cpu_actor','gpu_actor') for pair in request[name]['intervals_ms'] for x in pair})


def prepare(out,binary,*,smoke=False):
    out,binary=Path(out).resolve(),Path(binary).resolve();require(not out.exists(),'new external evidence directory required')
    parent_path=ROOT/'build/h2d-history-20261026/campaign-v1/manifest.json';parent=read(parent_path)
    native=ROOT/'profiling/dynamic_validation/native/external_probe.cpp'
    programs={}
    for workload,(layers,strategy) in WORKLOADS.items():
        for split in ('profile','heldout'):
            key=f'{workload}-{split}';binding=parent['programs'][f'l{layers}-h512-p64-s{strategy}-{split}']
            require(all(sha(binding[name]['path'])==binding[name]['sha256'] for name in ('program','reference')),'retained mathematical input changed')
            programs[key]=dict(binding,workload=workload)
    out.mkdir(parents=True)
    source_paths=[Path(__file__),native,native.parent/'owned_timer.hpp',native.parent/'external/CMakeLists.txt',
        ROOT/'profiling/dynamic_validation/idle_campaign.py',ROOT/'profiling/dac_validation/e2a.py',
        ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/native/steady_costs/probe.cpp',
        ROOT/'profiling/native/steady_costs/components.hpp']
    source_paths+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    sources=[]
    for path in source_paths:
        target=out/'executed_sources'/path.resolve().relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);sources.append(dict(source=receipt(path),archived=receipt(target)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    jobs=[]
    for split in (('profile',) if smoke else ('profile','heldout')):
        for round_id in range(1 if smoke else 3):
            workloads=('gpu_dense',) if smoke else tuple(WORKLOADS) if round_id%2==0 else tuple(reversed(WORKLOADS))
            for workload in workloads:
                scenarios=PLANS[round_id:]+PLANS[:round_id]
                for scenario in scenarios:
                    cpu,gpu=plans(split,scenario,round_id)
                    sampling=e2a.request('full',trials=128);sampling['repeats']=16
                    request=dict(schema='dynamic-external-request-1',source_sha256=sha(native),
                        sampling=sampling,expected_process_mask=4294967295,
                        instrumentation='stamped' if split=='profile' else 'bare',
                        cpu_actor=dict(device='cpu',main_thread_mask=4,component=component('cpu'),intervals_ms=cpu),
                        gpu_actor=dict(device='gpu',main_thread_mask=65536,component=component('gpu'),intervals_ms=gpu))
                    jobs.append(dict(id=f'{split}-r{round_id}-{workload}-{scenario}',split=split,round=round_id,
                        workload=workload,scenario=scenario,program_key=f'{workload}-{split}',request=request))
    manifest=dict(schema='dynamic-external-campaign-1',prepared_at_utc=e2a.utc(),parent_manifest=receipt(parent_path),
        binary=receipt(target),sources=sources,runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],
        hardware=parent['baseline_context']['hardware'],programs=programs,jobs=jobs,smoke_only=smoke,
        protocol=dict(counts={s:sum(j['split']==s for j in jobs) for s in ('profile','heldout')},
            timeout_seconds=300,retries=0,replays=2048,report_block_replays=16,
            initial_target='one reset/full prime after all actors initialized; a shared future QPC epoch then continuous forwards',
            per_block_reset_or_prime=False,
            actor_scope='owned competing threads with independent backend instances; not universal external application service',
            actor_compute='fixed seeded F32 GGML GEMM256; CPU actor shares target CPU2, GPU submission CPU16 uses CUDA0',
            planned_actor_controls_are_measured_utilization=False,
            actual_actor_start_stop='reported independently; schedules never triggered by target timing',
            profile_prices='only per-action inclusive labels; transition observations retained, no target-total fitting',
            stable_cell_rule='action interval contains no planned edge and begins at least200ms after latest planned edge',
            expected_first_epoch_wait_ms=2200,
            controller='owned high-resolution waitable timers and owned events; no Sleep, busy spin or global timeBeginPeriod',
            primary_prediction_input='predeclared offered-work plan plus model-owned recursive history',
            future_target_state_used=False,
            quality=dict(max_process_mean_ape=.10,max_block_ape_p90=.20,raw_cv_is_prediction_gate=False)))
    write(out/'manifest.json',manifest)
    return dict(out=str(out),counts=manifest['protocol']['counts'],smoke_only=smoke)


def check(manifest):
    idle_campaign.check(manifest)


def verify(native,request,binding,pid):
    require(native['schema']=='dynamic-external-native-1' and native['request']==request,'native request differs')
    require(native['process_id']==pid and native['source_sha256']==request['source_sha256'],'native PID/source differs')
    program=read(binding['program']['path']);require(native['program_sha256']==program['program_sha256'] and native['actions']==program['actions'],'target program/action identity differs')
    require(native['main_thread_affinity']['requested_mask']==native['main_thread_affinity']['observed_mask']==
        native['main_thread_affinity']['final_mask']==4 and native['main_thread_affinity']['restored'],'target thread affinity differs')
    require(native['process_affinity']['requested_mask']==native['process_affinity']['observed_mask']==
        native['process_affinity']['final_mask']==4294967295 and native['process_affinity']['applied_before_backend_initialization'],'owned process mask differs')
    hz=native['clock']['frequency_hz'];require(hz>0,'invalid QPC frequency')
    epoch,begin,end=(native[k] for k in ('target_expected_origin_tick','target_begin_tick','target_end_tick'))
    require(epoch<=begin<end and abs((end-begin)*1e9/hz-native['target_wall_ns'])<=1,'target clock ledger differs')
    require(abs((begin-epoch)*1e9/hz-native['initial_start_lateness_ns'])<=1,'origin lateness ledger differs')
    blocks=native['blocks'];require(len(blocks)==128 and native['continuous_replays']==2048 and native['reset_prime_between_timed_blocks'] is False,'continuous target contract differs')
    require(native['instrumentation']==request['instrumentation'] and native['actor_timeline_triggered_by_target_latency'] is False
        and native['future_target_state_used'] is False,'timing/causality differs')
    require(native['controller']['kind']=='owned-high-resolution-waitable-timer-v1' and native['controller']['busy_spin'] is False
        and native['controller']['global_timeBeginPeriod_called'] is False,'precise owned controller contract differs')
    previous=begin;action_intervals=0;block_ns=0
    for block,row in enumerate(blocks):
        first,finish=row['begin_tick'],row['end_tick'];require(previous<=first<finish<=end and row['block']==block and row['replays']==16,'block clock/index differs')
        require(abs((finish-first)*1e9/hz-row['wall_ns'])<=1,'block QPC ledger differs');block_ns+=row['wall_ns']
        if request['instrumentation']=='bare':require(not row['actions'],'bare run contains action stamps')
        else:
            require(len(row['actions'])==len(program['actions'])*16,'profile action count differs')
            last=first
            for number,a in enumerate(row['actions']):
                repeat,index=divmod(number,len(program['actions']));a0,a1=a['begin_tick'],a['end_tick']
                require(a['global_replay']==block*16+repeat and a['action_index']==index and last<=a0<=a1<=finish,'profile action order/clock differs')
                require(abs((a1-a0)*1e9/hz-a['wall_ns'])<=1,'profile action QPC ledger differs');last=a1;action_intervals+=1
        previous=finish
    require(block_ns<=native['target_wall_ns']+128,'block sum exceeds target envelope')
    actor_receipts=[]
    for device,mask in (('cpu',4),('gpu',65536)):
        actor=native['actors'][device];specified=request[device+'_actor']
        require(actor['device']==device and actor['cancelled'] is False and actor['repeated_output_equal'], 'owned actor failed/cancelled')
        require(actor['thread_affinity']['requested_mask']==actor['thread_affinity']['observed_mask']==actor['thread_affinity']['final_mask']==mask and actor['thread_affinity']['restored'],'actor thread mask differs')
        require(actor['ready_tick']<epoch and actor['stopped_tick']>=end,'actor lifetime differs')
        require(actor['controller']['kind']=='owned-high-resolution-waitable-timer-v1' and actor['idle_actor_busy_polling'] is False,'actor wait controller differs')
        require(actor['component']['shape']==specified['component']['shape'] and actor['component']['device']==device and
            actor['component']['seed']==specified['component']['seed'] and actor['offered_work_is_available_service'] is False,'actor workload semantics differ')
        require(len(actor['intervals'])==len(specified['intervals_ms']),'actor interval inventory differs')
        for receipt_row,(a,b) in zip(actor['intervals'],specified['intervals_ms']):
            require(receipt_row['planned_begin_ms']==a and receipt_row['planned_end_ms']==b and receipt_row['calls']>0,'actor planned pulse was not executed')
            actual_first,actual_last=receipt_row['actual_first_tick'],receipt_row['actual_last_completion_tick']
            expected_first=epoch+a*hz//1000;expected_last=epoch+b*hz//1000
            require(expected_first<=actual_first<expected_last and actual_first<=actual_last<=actor['stopped_tick'],'actor activity escaped planned start/lifetime')
            require(abs((actual_first-expected_first)*1e9/hz-receipt_row['start_lateness_ns'])<=1,'actor start ledger differs')
        actor_receipts.append(dict(device=device,intervals=len(actor['intervals']),stopped_tick=actor['stopped_tick']))
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'target numerical replay/copy check failed')
    import numpy as np
    ports=native['verification_native']['ports'];oracle=read(binding['reference']['path']);require(set(ports)==set(oracle),'target numerical inventory differs')
    for key,value in oracle.items():
        a,b=np.asarray(ports[key]),np.asarray(value)
        require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent target port mismatch '+key)
    return dict(passed=True,blocks=128,continuous_replays=2048,profile_action_intervals=action_intervals,
                actors=actor_receipts,normal_target_reset_between_blocks=False,numerical_oracle=True,
                actual_capacity_or_available_rate_inferred=False)


def collect(out,split):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m);require(split in ('profile','heldout'),'unknown split')
    barrier=None
    if split=='heldout':
        barrier=read(out/'frozen-barrier.json');require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen plan changed')
        for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen predictions/parameters changed')
    marker=out/(split+'-started.json')
    if not marker.exists():write(marker,dict(started_at_utc=e2a.utc(),barrier=barrier))
    executions=[]
    with run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists():executions.append(read(folder/'execution.json'));continue
            folder.mkdir(parents=True,exist_ok=False);binding=m['programs'][job['program_key']]
            for r in (binding['program'],binding['reference']):require(sha(r['path'])==r['sha256'],'mathematical input changed')
            request,output=folder/'request.json',folder/'result.json';write(request,job['request'])
            command=[m['binary']['path'],binding['program']['path'],str(request),str(output)]
            e=dict(job=job,started_at_utc=e2a.utc(),command=command,status='failed',barrier=barrier);process=None;handles=[]
            try:
                env=run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
                process,stdout,stderr,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295)
                handles=[stdout,stderr];e.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=m['protocol']['timeout_seconds']);e.update(returncode=code,process_exited_verified=True,forced_termination=False)
                require(code==0,'owned native process failed; retained without retry')
                e['verification']=verify(read(output),job['request'],binding,process.pid);e.update(status='completed',native=receipt(output))
            except BaseException as error:
                e['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    e['forced_termination']=True
                if process is not None:e.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                e['completed_at_utc']=e2a.utc();write(folder/'execution.json',e)
            executions.append(e);print(json.dumps(dict(job=job['id'],status=e['status'],error=e.get('error'))),flush=True)
    if not (out/(split+'-executions.json')).exists():write(out/(split+'-executions.json'),executions)
    return dict(split=split,declared=len(executions),completed=sum(e['status']=='completed' for e in executions))


def summarize_profile(out):
    """Retain all transitions; only declared settled cells estimate constants."""
    out=Path(out);m=read(out/'manifest.json');check(m);records=[]
    for execution in read(out/'profile-executions.json'):
        if execution['status']!='completed':continue
        require(sha(execution['native']['path'])==execution['native']['sha256'],'profile raw changed')
        job=execution['job'];native=read(execution['native']['path']);program=read(m['programs'][job['program_key']]['program']['path'])
        semantics={i:semantic_action(program,i) for i in range(len(program['actions']))};keys={i:digest(s) for i,s in semantics.items()}
        groups={};transitions=0;hz=native['clock']['frequency_hz'];epoch=native['target_expected_origin_tick'];edges=boundaries(job['request'])
        for block in native['blocks']:
            for action in block['actions']:
                start=(action['begin_tick']-epoch)*1000000000//hz;end=(action['end_tick']-epoch)*1000000000//hz
                before,after=state_at(job['request'],start),state_at(job['request'],max(start,end-1))
                previous=max((x for x in edges if x<=start),default=None)
                stable=before==after and not any(start<x<end for x in edges) and (previous is None or start-previous>=200000000)
                if not stable:transitions+=1;continue
                index=action['action_index'];key=(keys[index],*before)
                g=groups.setdefault(key,dict(semantic=semantics[index],cpu_offered=before[0],gpu_offered=before[1],count=0,sum_ns=0,
                    sum_squared_ns=0,min_ns=None,max_ns=None,action_indices=set()))
                value=action['wall_ns'];g['count']+=1;g['sum_ns']+=value;g['sum_squared_ns']+=value*value
                g['min_ns']=value if g['min_ns'] is None else min(value,g['min_ns']);g['max_ns']=value if g['max_ns'] is None else max(value,g['max_ns']);g['action_indices'].add(index)
        values=[]
        for key,g in groups.items():
            g['semantic_key']=key[0];g['action_indices']=sorted(g['action_indices']);g['mean_ns']=str(g['sum_ns'])+'/'+str(g['count']);values.append(g)
        records.append(dict(job=job,native=execution['native'],cells=values,transition_action_count=transitions,
            actor_receipts=native['actors'],total_target_ns=native['target_wall_ns'],
            total_target_is_fit_label=False))
    result=dict(schema='dynamic-external-profile-observations-1',records=records,
        declared=sum(j['split']=='profile' for j in m['jobs']),completed=len(records),
        conditioning='requested actor state and predeclared200ms settling rule; not physical available service',
        dropped_for_variance=0,transition_samples_retained_in_native=True,whole_target_used_for_fit=False,
        future_target_state_used=False)
    write(out/'profile-observations.json',result)
    return dict(summary=str(out/'profile-observations.json'),declared=result['declared'],completed=len(records))


def freeze(out,model,predictions):
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    require(not (out/'heldout-started.json').exists() and not any((out/'jobs'/j['id']).exists() for j in m['jobs'] if j['split']=='heldout'),
            'freeze after independent target acquisition rejected')
    executions=read(out/'profile-executions.json');require(len(executions)==24 and all(e['status']=='completed' for e in executions),'all24profilesrequired')
    p=read(predictions);require(p['schema']=='dynamic-external-frozen-predictions-1','prediction schema differs')
    require(p['future_target_state_used'] is False and p['whole_target_time_used_for_fit'] is False,'invalid predictor provenance')
    expected={j['id'] for j in m['jobs'] if j['split']=='heldout'};require(p['primary_provider'] in p['providers'],'explicit primary provider required')
    from fractions import Fraction
    for name,rows in p['providers'].items():
        require(set(rows)==expected,'every planned target must have every declared prediction')
        for r in rows.values():
            require(len(r['block_ns'])==128 and all(Fraction(str(x))>=0 for x in r['block_ns']),'128known recursive block predictions required')
            require(Fraction(str(r['total_ns']))>=sum(Fraction(str(x)) for x in r['block_ns']),'predicted target envelope shorter than blocks')
    for name,source in (('model-frozen.json',model),('predictions-frozen.json',predictions)):
        require(not (out/name).exists(),'frozen output exists');shutil.copy2(source,out/name)
    barrier=dict(schema='dynamic-external-freeze-1',frozen_at_utc=e2a.utc(),manifest=receipt(out/'manifest.json'),
        files={name:sha(out/name) for name in ('model-frozen.json','predictions-frozen.json')},
        profile=receipt(out/'profile-executions.json'),all_heldout_coordinates_known=True,
        future_target_state_used=False,whole_target_time_used_for_fit=False)
    write(out/'frozen-barrier.json',barrier);return dict(barrier=receipt(out/'frozen-barrier.json'),providers=list(p['providers']))


def evaluate(out):
    """Same fixed work indices; predicted boundaries are never reset to truth."""
    from fractions import Fraction
    import math
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    barrier=read(out/'frozen-barrier.json');require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen plan changed')
    for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen model/predictions changed')
    predictions=read(out/'predictions-frozen.json');executions=read(out/'heldout-executions.json')
    expected={j['id'] for j in m['jobs'] if j['split']=='heldout'}
    require({e['job']['id'] for e in executions}==expected,'independent process denominator differs')
    providers={}
    for provider,rows in predictions['providers'].items():
        processes=[]
        for execution in executions:
            job=execution['job'];r=dict(job_id=job['id'],workload=job['workload'],scenario=job['scenario'],
                round=job['round'],status=execution['status'],passed=False)
            if execution['status']=='completed':
                require(sha(execution['native']['path'])==execution['native']['sha256'],'independent raw changed')
                native=read(execution['native']['path']);predicted=rows[job['id']]
                actual=[b['wall_ns'] for b in native['blocks']];estimated=[float(Fraction(str(x))) for x in predicted['block_ns']]
                block_ape=[abs(p-a)/a for p,a in zip(estimated,actual)]
                p90=sorted(block_ape)[math.ceil(.90*len(block_ape))-1]
                hz=native['clock']['frequency_hz'];origin=native['target_expected_origin_tick']
                # Epoch precedes target start. Its small actual dispatch lateness is
                # an observed error, never a post-hoc shift in predicted starts.
                observed_total=(native['target_end_tick']-origin)*1e9/hz
                predicted_total=float(Fraction(str(predicted['total_ns'])))
                ape=abs(predicted_total-observed_total)/observed_total
                cumulative=0.;boundary_errors=[]
                for i,(p,b) in enumerate(zip(estimated,native['blocks'])):
                    cumulative+=p;actual_end=(b['end_tick']-origin)*1e9/hz
                    boundary_errors.append(cumulative-actual_end)
                r.update(observed_planned_epoch_to_end_ns=observed_total,predicted_total_ns=predicted_total,
                    observed_target_wall_ns=native['target_wall_ns'],initial_actual_start_lateness_ns=native['initial_start_lateness_ns'],
                    process_mean_ape=ape,block_ape_p90_higher=p90,block_ape=block_ape,
                    fixed_work_boundary_error_ns=boundary_errors,block_error_denominator='same fixed replay indices, no actual-start re-anchoring',
                    passed=ape<=m['protocol']['quality']['max_process_mean_ape'] and p90<=m['protocol']['quality']['max_block_ape_p90'])
            else:r['error']=execution.get('error')
            processes.append(r)
        groups=[]
        for workload in WORKLOADS:
            for scenario in PLANS:
                rs=[r for r in processes if r['workload']==workload and r['scenario']==scenario]
                completed=[r for r in rs if r['status']=='completed'];actuals=[r['observed_planned_epoch_to_end_ns'] for r in completed]
                if actuals:
                    mean=sum(actuals)/len(actuals);raw_cv=(sum((x-mean)**2 for x in actuals)/len(actuals))**.5/mean
                else:raw_cv=None
                groups.append(dict(workload=workload,scenario=scenario,declared=len(rs),completed=len(completed),
                    passed=sum(r['passed'] for r in rs),raw_process_cv_population=raw_cv,
                    raw_cv_is_prediction_gate=False))
        providers[provider]=dict(processes=processes,groups=groups,declared=len(processes),passed=sum(r['passed'] for r in processes))
    result=dict(schema='dynamic-external-evaluation-1',primary_provider=predictions['primary_provider'],providers=providers,
        frozen_barrier=receipt(out/'frozen-barrier.json'),mean_gate=m['protocol']['quality']['max_process_mean_ape'],
        block_ape_p90_gate=m['protocol']['quality']['max_block_ape_p90'],raw_cv_is_prediction_gate=False,
        future_observations_used_for_prediction=False,failed_processes_remain_in_denominator=True,
        claim_scope='two continuous Dense programs under declared owned-actor workload pulses; not arbitrary device fluctuation')
    write(out/'evaluation.json',result)
    return dict(evaluation=str(out/'evaluation.json'),providers={p:dict(passed=v['passed'],declared=v['declared']) for p,v in providers.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','profile','heldout','summarize-profile','freeze','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--binary');p.add_argument('--smoke',action='store_true')
    p.add_argument('--model');p.add_argument('--predictions');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.out,a.binary,smoke=a.smoke)
    elif a.command=='summarize-profile':result=summarize_profile(a.out)
    elif a.command=='freeze':result=freeze(a.out,a.model,a.predictions)
    elif a.command=='evaluate':result=evaluate(a.out)
    else:result=collect(a.out,a.command)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
