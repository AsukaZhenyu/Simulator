"""Fresh continuous-invocation Attention prices and new planned-state tests.

Old heldout timings motivate the lifecycle correction but supply no price or
coefficient. Every price comes from a new constant-state run of the original
component. Independent schedules/processes are new; mathematical heldout values
are deliberately reused and are never described as a newly unseen seed.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

from . import bare_macro_v2_campaign as previous
from . import actor_source_v8
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action

ROOT=previous.ROOT
read,write,sha,receipt,require=previous.read,previous.write,previous.sha,previous.receipt,previous.require
STATES=previous.STATES
PULSES={
    'cpu_tail':dict(cpu=[[200,900],[1250,1900]],gpu=[[100,500],[1100,1500]]),
    'separated':dict(cpu=[[500,1100],[1500,1950]],gpu=[[100,750],[1250,1750]])}


def protocol():
    return dict(schema='continuous-Attention-protocol-10',target='one completed original Executor.action Attention component',
        foreground_lifecycle='continuous_invocation',profile_fresh_processes=12,profile_states=[list(x) for x in STATES],
        fresh_contexts_per_state=3,under_state_warmup_calls=1,heldout_extra_under_state_warmup_calls=0,warmup_in_price=False,
        continuous_profile_min_duration_ns=1000000000,continuous_profile_max_calls=512,
        maximum_calls_before_duration='safety-stop data retained but unsupported; no automatic retry',
        stop_depends_on_accuracy=False,per_context_price='arithmetic mean of completed invocation responses; excludes separately recorded gaps',
        state_price='equal mean of all three fresh context prices; no target/strategy residual multiplier',
        one_postwarm_call_suffices_if_duration_at_least_one_second=True,
        whole_calibration_interval_in_actual_and_planned_state_required=True,profile_arrival_lateness='recorded, not a steady-price deadline',
        ON_window_ms=[0,30000],ON_actor_stop_target_dependent=False,
        heldout_fresh_processes=6,heldout_replays=256,heldout_block_replays=16,heldout_pulses=PULSES,
        new_heldout_schedules=True,new_heldout_processes=True,mathematical_seed_previously_used=True,
        newly_unseen_mathematical_seed_claimed=False,previous_heldout_times_used_for_prices=False,
        actor_workload='retained CPU/GPU GEMM256; CPU2 foreground/CPU actor, CPU16 GPU submitter',
        measurement_observers=dict(CUPTI=False,NVML=False,additional_CUDA_events=False),
        providers=['static00','memoryless_four_state_macro'],tau='unidentified; no stateful provider',
        calculation='unchanged modeling.PlannedCompletedResponse/common evaluator; own completion clock',
        observed_target_boundaries_used_for_prediction=False,prices_frozen_before_heldout=True,
        initial_lifecycle='calibration: one under-state warm component then steady loop; heldout: existing full-prime then first timed component, no extra component warm',
        first_heldout_transient='unadjusted independent test of the steady-price approximation; no observed correction',
        acceptance=dict(heldout_entry_max_lateness_ns=10000000,process_mean_APE_max=.10,block_APE_P90_max=.20,
                        raw_CV_reported_separately=True,all_declared_denominators_retained=True),
        retries=0,slow_samples_removed=0,whole_LLM_dynamic_accuracy_claimed=False)


def plans(template,source_sha,actor_sha):
    jobs=[]
    def request(split,state=None,plan=None):
        r=copy.deepcopy(template)
        for key in ('callback_source_sha256','max_API_records','callback_scope','replays','block_replays'):r.pop(key,None)
        r.update(schema='dynamic-continuous-request-10',source_sha256=source_sha,actor_source_sha256=actor_sha,split=split,
            instrumentation='stamped',foreground_lifecycle='continuous_invocation',offered_state=list(state) if state is not None else None,
            profile_min_duration_ns=1000000000,profile_max_calls=512,under_state_warmup_calls=1 if split=='profile' else 0,
            actor_epoch_delay_ms=1000,target_entry_max_lateness_ns=10000000,target_entry_offset_ms=1000 if state and any(state) else 0)
        for device in ('cpu','gpu'):
            r[device+'_actor']['intervals_ms']=([[0,30000]] if state[device=='gpu'] else []) if split=='profile' else copy.deepcopy(PULSES[plan][device])
        sampling=previous.native_campaign.base.e2a.request('action',trials=1 if split=='profile' else 16,target=22);sampling['repeats']=1 if split=='profile' else 16;r['sampling']=sampling
        if split=='heldout':r.update(replays=256,block_replays=16)
        return r
    for round_id in range(3):
        states=STATES[round_id:]+STATES[:round_id]
        for state in states:
            jobs.append(dict(id=f'profile-{state[0]}{state[1]}-r{round_id}',split='profile',round=round_id,offered_state=list(state),imported=False,request=request('profile',state=state)))
    for round_id in range(3):
        order=tuple(PULSES) if round_id%2==0 else tuple(reversed(PULSES))
        for plan in order:jobs.append(dict(id=f'heldout-{plan}-r{round_id}',split='heldout',round=round_id,pulse_plan=plan,imported=False,request=request('heldout',plan=plan)))
    return jobs


def verify(native,request,binding,pid):
    require(native['schema']=='dynamic-continuous-native-10' and native['request']==request and native['process_id']==pid,'native identity/request differs')
    for key in ('source_sha256','actor_source_sha256'):require(native[key]==request[key],'source binding differs '+key)
    require(native['program_sha256']==binding['program_sha256'] and native['attention_action_index']==22,'mathematical target differs')
    program=read(binding['program']['path']);require(native['actions']==program['actions'] and native['target_action']==program['actions'][22],'executed action template differs')
    require(native['instrumentation']=='stamped' and native['foreground_lifecycle']=='continuous_invocation','exact lifecycle/boundary required')
    for flag in ('callback_stage_prices_used','profiling_CUPTI_initialized','profiling_NVML_initialized','profiling_extra_CUDA_events_created','internal_GPU_work_added','API_response_is_CPU_active','future_target_state_used','actor_timeline_triggered_by_target_latency','performance_parameters_fitted','complete_LLM_strategy_replayed','reset_prime_between_timed_blocks'):
        require(native[flag] is False,'scope broadened '+flag)
    require(native['complete_attention_graph_only'] is True and native['timed_result_checked_before_reset'] is True,'complete original component required')
    require(native['process_affinity']['observed_mask']==native['process_affinity']['final_mask']==4294967295,'process affinity differs')
    for key in ('initialization_affinity','target_thread_affinity'):require(native[key]['observed_mask']==native[key]['final_mask']==4,'foreground/init affinity differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'native numerical oracle failed')
    import numpy as np
    oracle,ports=read(binding['reference']['path']),native['verification_native']['ports'];require(set(oracle)==set(ports),'numerical port inventory differs')
    for name in oracle:
        a,b=np.asarray(oracle[name]),np.asarray(ports[name]);require(a.shape==b.shape and np.all(np.isfinite(b)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent numerical mismatch '+name)
    hz=native['clock']['frequency_hz'];require(native['clock']['clock_id']=='windows_qpc' and type(hz) is int and hz>0,'positive QPC required')
    ns=lambda a,b:Fraction((b-a)*10**9,hz)
    begin,end,epoch=native['target_begin_tick'],native['target_end_tick'],native['target_expected_origin_tick'];require(epoch<=begin<end,'target clock differs')
    profile=request['split']=='profile';entry=native['calibration_begin_tick'] if profile else begin
    require(native['entry_observed_tick']==entry and entry>=native['planned_target_entry_tick'],'actual entry differs')
    late=ns(native['planned_target_entry_tick'],entry);require(abs(late-native['initial_start_lateness_ns'])<=1 and native['target_entry_controller_passed']==(late<=10000000),'entry QPC/flag differs')
    require(abs(ns(begin,end)-native['target_wall_ns'])<=1,'complete target QPC differs')
    if not profile:require(native['target_entry_controller_passed'] is True and late<=10000000,'heldout entry late; original10ms gate retained')
    actions=[];last=begin
    require(len(native['blocks'])==(1 if profile else 16),'complete block inventory required')
    for index,block in enumerate(native['blocks']):
        require(block['block']==index and last<=block['begin_tick']<block['end_tick']<=end and len(block['actions'])==block['replays'],'block clock/count differs')
        require(abs(ns(block['begin_tick'],block['end_tick'])-block['wall_ns'])<=1,'block duration differs')
        current=block['begin_tick']
        for action in block['actions']:
            require(action['ordinal']==action['global_replay']==len(actions) and action['action_index']==22 and current<=action['begin_tick']<action['end_tick']<=block['end_tick'],'action clock/order differs')
            require(abs(ns(action['begin_tick'],action['end_tick'])-action['wall_ns'])<=1,'action duration differs');current=action['end_tick'];actions.append(action)
        last=block['end_tick']
    require(native['continuous_replays']==len(actions),'actual action count differs')
    whole_start=entry
    if profile:
        warm=native['under_state_warmup'];continuous=native['continuous_profile'];require(warm['action_index']==22 and warm['included_in_price'] is False and entry<=warm['begin_tick']<warm['end_tick']<=begin,'one under-state warm component required')
        require(abs(ns(warm['begin_tick'],warm['end_tick'])-warm['wall_ns'])<=1,'warm component duration differs')
        require(continuous['actions']==actions and continuous['begin_tick']==begin and continuous['end_tick']==end,'continuous labels differ')
        require(continuous['min_duration_ns']==1000000000 and continuous['max_calls']==512 and 1<=len(actions)<=512 and continuous['accuracy_dependent_stop'] is False,'predeclared stopping contract differs')
        require(continuous['minimum_duration_reached'] is True and ns(begin,end)>=1000000000 and continuous['stop_reason']=='minimum_duration','minimum duration not reached; safety stop retained but unsupported')
        previous_tick=begin;gap_ticks=0;gaps=continuous['gaps'];require(len(gaps)==len(actions)+1,'all prefix/intercall/suffix gaps required')
        for action,gap in zip(actions,gaps):
            require(gap['begin_tick']==previous_tick and gap['end_tick']==action['begin_tick'] and gap['begin_tick']<=gap['end_tick'],'gap partition differs')
            require(abs(ns(gap['begin_tick'],gap['end_tick'])-gap['wall_ns'])<=1,'gap duration differs');gap_ticks+=gap['end_tick']-gap['begin_tick'];previous_tick=action['end_tick']
        require(gaps[-1]['begin_tick']==previous_tick and gaps[-1]['end_tick']==end and previous_tick<=end,'suffix gap differs')
        require(abs(ns(gaps[-1]['begin_tick'],gaps[-1]['end_tick'])-gaps[-1]['wall_ns'])<=1,'suffix gap duration differs');gap_ticks+=end-previous_tick
        require(sum(a['end_tick']-a['begin_tick'] for a in actions)+gap_ticks==end-begin,'complete continuous QPC partition does not close')
        require(native['profile_window_contained'] is True and native['steady_window_begin_tick']==epoch and native['steady_window_end_tick']==epoch+30*hz and epoch<=entry<end<=epoch+30*hz,'warm+continuous calibration escaped constant window')
    else:require(request['under_state_warmup_calls']==0 and 'under_state_warmup' not in native and len(actions)==256 and all(b['replays']==16 for b in native['blocks']),'fixed256/16 independent workload/initial lifecycle differs')
    for device,mask in (('cpu',4),('gpu',65536)):
        actor=native['actors'][device];require(actor['thread_affinity']['observed_mask']==actor['thread_affinity']['final_mask']==mask and actor['ready_tick']<=epoch,'actor placement/readiness differs')
        require(actor['repeated_output_equal'] is True and actor['cancelled'] is False,'actor numerical/cancellation failure')
        requested=request[device+'_actor']['intervals_ms'];require(len(actor['intervals'])==len(requested),'actor interval inventory differs')
        for interval,pair in zip(actor['intervals'],requested):require([interval['planned_begin_ms'],interval['planned_end_ms']]==pair and interval['calls']>0,'actor planned/actual work absent')
        if profile:
            bit=request['offered_state'][device=='gpu'];require(requested==([[0,30000]] if bit else []),'constant offered state differs')
            if bit:
                interval=actor['intervals'][0];require(interval['actual_first_tick']<=whole_start<end<=interval['actual_last_completion_tick'],'actual ON actor did not bracket warm+timed calibration')
    return dict(passed=True,complete_Attention_replays=len(actions),mathematical_oracle=True,source_scope=True,foreground_lifecycle='continuous_invocation',
        constant_full_interval_verified=profile,initial_start_lateness_ns=native['initial_start_lateness_ns'],original_entry_deadline_passed=native['target_entry_controller_passed'],
        warm_component_not_in_price=profile,callback_records_used=0,hardware_accuracy_certified=False)


def verified_job(out,job,manifest):
    path=Path(out)/'jobs'/job['id']/'execution.json';execution=read(path);require(execution['status']=='completed' and execution['returncode']==0 and execution['process_exited_verified'],'declared component process failed/missing')
    require(execution.get('process_creation_time_100ns',0)>0 and sha(execution['native']['path'])==execution['native']['sha256'],'distinct native/process provenance missing')
    native=read(execution['native']['path']);verify(native,job['request'],manifest['programs'][job['split']],execution['process_id']);return native,execution,path


def prepare(out,binary,*,build_receipt,parent):
    out,binary,parent=Path(out).resolve(),Path(binary).resolve(),Path(parent).resolve();build_receipt=Path(build_receipt).resolve();require(not out.exists(),'new exclusive continuous campaign required')
    built=read(build_receipt);source=Path(__file__).parent/'native/v10/continuous_probe.cpp';actor=source.parent.parent/'v8/actors.inc';actor_source_v8.check()
    require(built['binary']['sha256']==sha(binary) and built['source_sha256']==sha(source) and built['actor_source_sha256']==sha(actor),'explicit native build receipt differs')
    p=read(parent/'manifest.json');programs=copy.deepcopy(p['programs']);profile=read(programs['profile']['program']['path']);held=read(programs['heldout']['program']['path'])
    semantic=semantic_action(profile,22);require(semantic==semantic_action(held,22) and profile['seed']!=held['seed'],'same exact geometry/different original mathematical values required')
    old_plans={json.dumps(j['request'][d+'_actor']['intervals_ms']) for j in p['jobs'] if j['split']=='heldout' for d in ('cpu','gpu')}
    require(all(json.dumps(pair) not in old_plans for plan in PULSES.values() for pair in plan.values()),'new independent schedules must differ from previously evaluated plans')
    template=next(j['request'] for j in p['jobs'] if j['split']=='profile');jobs=plans(template,sha(source),sha(actor))
    actors={d:{k:v for k,v in template[d+'_actor'].items() if k!='intervals_ms'} for d in ('cpu','gpu')}
    context=dict(binary_sha256=sha(binary),runtime=p['runtime'],topology=p['hardware'],source_sha256=sha(source),actor_source_sha256=sha(actor),actors=actors,
        foreground_lifecycle='continuous_invocation',under_state_warmup_calls=1,heldout_extra_under_state_warmup_calls=0,
        initial_lifecycle='calibration has one under-state warm component; heldout uses existing full prime only and independently tests the first-call approximation',
        continuous_profile_min_duration_ns=1000000000,continuous_profile_max_calls=512,
        process_mask=4294967295,target_thread_mask=4,threads=1,initialization_policy='pinned_init',
        boundary='completed original Attention invocation; no extra CUPTI/NVML/CUDA event observers',warmup_iterations=20,warmup_min_ms=500)
    scope=dict(context_sha256=digest(context),semantic_sha256=digest(semantic),context=context,semantic_coordinate=semantic)
    sources=[Path(__file__),Path(__file__).with_name('continuous_macro_fit.py'),source,source.parent/'CMakeLists.txt',actor,Path(actor_source_v8.__file__)]
    sources+=[Path(s['source']['path']) for s in p['sources'] if Path(s['source']['path']).name not in ('continuous_probe.cpp','CMakeLists.txt')]
    sources=list(dict.fromkeys(sources));require(all(s.is_file() for s in sources),'complete source required')
    out.mkdir(parents=True);archived=[]
    for s in sources:
        target=out/'executed_sources'/s.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,target);archived.append(dict(source=receipt(s),archived=receipt(target)))
    exe=out/'executed_binary'/binary.name;exe.parent.mkdir();shutil.copy2(binary,exe)
    m=dict(schema='continuous-Attention-campaign-10',prepared_at_utc=previous.native_campaign.base.e2a.utc(),binary=receipt(exe),build_receipt=receipt(build_receipt),
        parent_diagnostic_manifest=receipt(parent/'manifest.json'),prior_diagnostic_evaluation=receipt(parent/'INDEPENDENT_EVALUATION.json'),
        runtime=p['runtime'],runtime_dirs=p['runtime_dirs'],hardware=p['hardware'],programs=programs,jobs=jobs,sources=archived,scope=scope,protocol=protocol(),
        source_sha256=sha(source),actor_source_sha256=sha(actor),old_diagnostic_prices_imported=False)
    write(out/'manifest.json',m);return dict(manifest=receipt(out/'manifest.json'),fresh_profile=12,fresh_heldout=6,hardware_executed=False)


def check(out):
    m=read(out/'manifest.json')
    for name in ('binary','build_receipt','parent_diagnostic_manifest','prior_diagnostic_evaluation'):
        r=m[name];require(sha(r['path'])==r['sha256'],'bound receipt changed '+name)
    for r in m['runtime']:require(sha(r['path'])==r['sha256'],'runtime changed')
    for s in m['sources']:
        for name in ('source','archived'):require(sha(s[name]['path'])==s[name]['sha256'],'bound source changed')
    for b in m['programs'].values():
        for name in ('program','reference'):require(sha(b[name]['path'])==b[name]['sha256'],'mathematical input/reference changed')
    return m


def authorization(out,split):
    require(split in ('profile','heldout'),'unknown split');m=check(out);a=read(out/f'{split}-authorization.json')
    require(a.get('collection_authorized') is True and a.get('split')==split and a.get('manifest_sha256')==sha(out/'manifest.json'),'root authorization for exact split/manifest required')
    if split=='heldout':
        f=read(out/'frozen-barrier.json');require(a.get('frozen_barrier_sha256')==sha(out/'frozen-barrier.json') and f['manifest_sha256']==sha(out/'manifest.json'),'heldout authorization not bound to freeze')
        require(f['profile_supported'] and f['parameters_frozen'] and f['predictions_frozen'] and f['calibration_target_is_component'] and f['whole_LLM_target_time_used_for_fit'] is False,'complete independent forecast freeze required')
        for name,value in f['files'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(out/p)==value,'frozen forecast changed')
    return m


def collect(out,split):
    out=Path(out).resolve();m=authorization(out,split);marker=out/f'{split}-started.json';require(not marker.exists(),'no retry/repeated split');write(marker,dict(started_at_utc=previous.native_campaign.base.e2a.utc(),split=split))
    jobs=[j for j in m['jobs'] if j['split']==split];executions=[]
    with previous.native_campaign.base.run.measurement_lock():
        for job in jobs:
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request']);binding=m['programs'][split]
            command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],split=split,status='failed',command=command,request=receipt(folder/'request.json'),foreground_lifecycle='continuous_invocation');process=None;handles=[]
            try:
                env=previous.native_campaign.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=previous.native_campaign.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=process.pid,process_creation_time_100ns=previous.native_campaign.process_creation_100ns(process),process_affinity=affinity)
                code=process.wait(timeout=90 if split=='profile' else 120);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                record['raw_native']=receipt(folder/'result.json');record['verification']=verify(read(folder/'result.json'),job['request'],binding,process.pid);record.update(status='completed',native=record['raw_native'])
            except BaseException as error:
                record['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for h in handles:h.close()
                write(folder/'execution.json',record)
            executions.append(record);print(json.dumps(dict(job_id=job['id'],status=record['status'],error=record.get('error'))),flush=True)
            if record['status']!='completed':break
    write(out/f'{split}-executions.json',executions)
    return dict(declared=len(jobs),acquired=len(executions),completed=sum(e['status']=='completed' for e in executions),not_run=len(jobs)-len(executions),old_diagnostic_prices_used=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect'));p.add_argument('--out');p.add_argument('--parent');p.add_argument('--binary');p.add_argument('--build-receipt');p.add_argument('--split',choices=('profile','heldout'));a=p.parse_args()
    value=protocol() if a.command=='protocol' else prepare(a.out,a.binary,build_receipt=a.build_receipt,parent=a.parent) if a.command=='prepare' else collect(a.out,a.split)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
