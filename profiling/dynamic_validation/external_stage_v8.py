"""Staged, fixed-shape Attention protocol; no implicit inference or fitting.

OFF observer gate precedes expensive ON profiles. The exact owned actor
workload and verified CPU/core/runtime context are retained in every binding.
The model receives per-API/gap calibration labels, never a target multiplier.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from . import actor_source_v8
from . import idle_campaign as base
from .external_v6 import API_COUNTS, callback_ledger, physical_core_contract

ROOT=base.ROOT
read,write,sha,receipt,require=base.read,base.write,base.sha,base.receipt,base.require
STATES=((0,0),(1,0),(0,1),(1,1))
PULSES={
    'early_joint':dict(cpu=[[150,500],[1000,1400]],gpu=[[100,650],[1100,1600]]),
    'staggered':dict(cpu=[[300,750],[1250,1800]],gpu=[[100,450],[900,1500]])}


def protocol():
    return dict(schema='attention-runtime-stage-protocol-8',target='complete fixed-shape Attention graph through original Executor.action',
                target_action_index=22,whole_LLM_dynamic_accuracy_claimed=False,
                primary_target_CPU=2,CPU_actor_CPU=2,GPU_actor_submission_CPU=16,
                topology_verified_not_guessed=True,actor_workload='exact retained CPU/GPU GEMM256 component specifications',
                off_observer_gate_processes=6,ON_profile_processes=18,heldout_processes=6,
                independent_callback_contexts_per_state=3,minimum_samples_per_coordinate_per_context=1,
                rare_coordinate_one_sample_is_low_variance=False,
                rare_coordinate_gate_is_new_protocol_not_relaxation_of_previous_ten_sample_protocol=True,
                profile_complete_Attention_calls_per_process=1,profile_API_counts=API_COUNTS,
                profile_states=[list(x) for x in STATES],ON_planned_window_ms=[0,30000],ON_actor_stop_depends_on_target=False,
                total_ON_budget_seconds_minimum=540,estimated_profile_minutes_including_ready_warmup='approximately10–13, excluding unsupported/failed diagnostics',
                target_entry=dict(all_initialization_and_observer_ready_before_epoch=True,future_epoch_delay_ms=1000,
                                  OFF_entry_offset_ms=0,ON_entry_offset_ms=1000,max_lateness_ns=10000000),
                stage_order=['00_observer_gate','ON_profiles_after_passed_OFF_gate','profile_support_and_parameter_freeze','independent_heldout'],
                ON_state_observer_gate_also_required=True,all_actual_API_and_gap_coordinates_required=True,
                stages_must_finish_inside_constant_window=True,cross_state_stage_labels_are_not_steady_prices=True,
                heldout_replays=256,heldout_block_replays=16,heldout_pulses=PULSES,heldout_independent_mathematical_seed=True,
                heldout_initial_OFF_ms_at_least=100,heldout_instrumentation='stamped; callback-free target timing',
                acceptance=dict(observer_relative_target_difference_max=.05,all_three_matched_pairs_each_state_required=True,
                                model_mean_APE_max=.10,model_block_P90_APE_max=.20,
                                measured_raw_CV='reported independently; not substituted for prediction accuracy',
                                absolute_latency_and_signed_bias_reported=True,
                                all_failures_and_declared_denominators_retained=True),
                providers=['static00','macro_four_state_sum_of_same_stage_prices','ordered_API_gap_four_state'],
                macro_baseline_fitted_from_whole_target=False,macro_baseline_uses_same_stage_profile=True,
                simpler_macro_success_is_preserved=True,tau='unsupported unless separately identified transition data; no default',
                coefficient_source='profile individual API/gap responses only',
                prediction_clock='own recursive completion clock; no observed internal boundary reanchor',
                API_response_is_CPU_active=False,internal_GPU_cost_added=False,
                missing_state_is_zero=False,retries=0,automatic_extra_sampling=False)


def job_plans(template,source_sha,callback_sha,actor_sha):
    jobs=[]
    def request(split,mode,state,round_id,plan=None):
        r=copy.deepcopy(template)
        r.update(schema='dynamic-external-stage-request-8',source_sha256=source_sha,callback_source_sha256=callback_sha,
                 actor_source_sha256=actor_sha,split=split,instrumentation=mode,init_policy='pinned_init',attention_action_index=22,
                 actor_epoch_delay_ms=1000,target_entry_max_lateness_ns=10000000,offered_state=list(state) if state is not None else None)
        for device,mask in (('cpu',4),('gpu',65536)):
            r[device+'_actor']['main_thread_mask']=mask;r[device+'_actor']['init_policy']='pinned_init'
            r[device+'_actor']['intervals_ms']=([[0,30000]] if state[device=='gpu'] else []) if split=='profile' else copy.deepcopy(PULSES[plan][device])
        count,block=(1,1) if split=='profile' else (256,16)
        sampling=base.e2a.request('action',trials=count//block,target=22);sampling['repeats']=block
        r.update(sampling=sampling,replays=count,block_replays=block,max_API_records=1024 if split=='profile' else 131072,
                 target_entry_offset_ms=1000 if split=='profile' and any(state) else 0)
        return r
    for state in STATES:
        stage='00_observer_gate' if state==(0,0) else 'ON_profiles'
        for round_id in range(3):
            # Fresh processes, fixed counter-order; a subscribed but unarmed callback is not called bare.
            modes=('stamped','runtime_callbacks') if round_id%2==0 else ('runtime_callbacks','stamped')
            for mode in modes:
                label=f'{state[0]}{state[1]}'
                jobs.append(dict(id=f'profile-{label}-r{round_id}-{mode}',stage=stage,split='profile',offered_state=list(state),round=round_id,
                                 pair_id=f'profile-{label}-r{round_id}',request=request('profile',mode,state,round_id)))
    for plan in PULSES:
        for round_id in range(3):
            jobs.append(dict(id=f'heldout-{plan}-r{round_id}',stage='heldout',split='heldout',pulse_plan=plan,round=round_id,
                             request=request('heldout','stamped',None,round_id,plan)))
    return jobs


def prepare(out,binary,*,cupti_dll,build_receipt):
    out,binary,dll=Path(out).resolve(),Path(binary).resolve(),Path(cupti_dll).resolve()
    build_receipt=Path(build_receipt).resolve();built=read(build_receipt)
    require(not out.exists() and binary.is_file() and dll.is_file(),'new output and explicit existing private binary/CUPTI required')
    require(Path(built['binary']['path']).resolve()==binary and built['binary']['sha256']==sha(binary),
            'successful explicit build receipt must bind this exact native binary')
    derived=actor_source_v8.check()
    parent_path=ROOT/'build/dynamic-validation-20261026/external-v1/manifest.json';parent=read(parent_path)
    runtime_parent_path=ROOT/'build/dynamic-diagnosis-20261027/h2d-v3-round1/manifest.json';runtime_parent=read(runtime_parent_path)
    require(any(Path(x['path']).resolve()==dll and x['sha256']==sha(dll) for x in runtime_parent['runtime']),'matching retained CUPTI runtime required')
    programs={s:copy.deepcopy(parent['programs']['gpu_dense-'+s]) for s in ('profile','heldout')}
    program_values={}
    for binding in programs.values():
        for k in ('program','reference'):require(sha(binding[k]['path'])==binding[k]['sha256'],'retained input/reference changed')
        value=read(binding['program']['path'])
        binding['program_sha256']=value['program_sha256']
        binding['seed']=value['seed']
        program_values[binding['split']]=value
        action=value['actions'][22]
        require(action['kind']=='COMPUTE' and action['device']=='gpu' and action['target_id'].endswith(':attn'),'fixed GPU Attention template missing')
    require(programs['profile']['program_sha256']!=programs['heldout']['program_sha256']
            and programs['profile']['seed']!=programs['heldout']['seed'],'independent heldout mathematical seed required')
    require(program_values['profile']['physical']==program_values['heldout']['physical']
            and program_values['profile']['actions']==program_values['heldout']['actions'],
            'heldout must retain exact shapes/runtime/action template while changing values')
    from profiling.state_response.diagnose_cpu import topology
    hardware=physical_core_contract(topology())
    source=Path(__file__).parent/'native/v8/external_stage_probe.cpp';callback=source.parent.parent/'v6/runtime_callbacks.hpp';actors=source.parent/'actors.inc'
    for key,path in (('native_source',source),('callback_source',callback),('actor_source',actors)):
        require(built[key]['sha256']==sha(path),'binary build source differs: '+key)
    require(built['selftest'].get('passed') is True and built['hardware_executed'] is False,
            'successful no-backend build selftest required')
    template=next(x['request'] for x in parent['jobs'] if x['id']=='profile-r0-gpu_dense-cpu_gpu_staggered')
    jobs=job_plans(template,sha(source),sha(callback),sha(actors))
    paths=[Path(__file__),Path(actor_source_v8.__file__),Path(derived['original']),actors,source,source.parent/'CMakeLists.txt',callback,
           source.parent.parent/'v3/common.hpp',source.parent.parent/'v3/thread_inventory.hpp',source.parent.parent/'owned_timer.hpp',
           ROOT/'profiling/dynamic_validation/runtime_stage_records.py',ROOT/'profiling/dynamic_validation/runtime_stage_profile.py',
           ROOT/'profiling/dynamic_validation/external_stage_v8_fit.py',
           ROOT/'modeling/llm_infer_model/performance/runtime_response.py',
           ROOT/'modeling/llm_infer_model/performance/planned_response.py',
           ROOT/'modeling/llm_infer_model/performance/evaluator.py',
           ROOT/'modeling/llm_infer_model/performance/graph.py',
           ROOT/'modeling/llm_infer_model/performance/parameters.py',
           ROOT/'modeling/llm_infer_model/performance/service_response.py',
           ROOT/'profiling/dynamic_validation/idle_campaign.py',ROOT/'profiling/dynamic_validation/external_v6.py',
           ROOT/'profiling/native/physical_stages/cupti_capture.hpp',ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
           ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/dac_validation/e2a.py']
    paths+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    out.mkdir(parents=True);sources=[]
    for path in paths:
        archive=out/'executed_sources'/path.relative_to(ROOT);archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archive)
        sources.append(dict(source=receipt(path),archived=receipt(archive)))
    exe=out/'executed_binary'/binary.name;exe.parent.mkdir();shutil.copy2(binary,exe)
    m=dict(schema='attention-runtime-stage-campaign-8',prepared_at_utc=base.e2a.utc(),parent_manifest=receipt(parent_path),
           runtime_parent_manifest=receipt(runtime_parent_path),derived_actor=derived,binary=receipt(exe),build_receipt=receipt(build_receipt),runtime=runtime_parent['runtime'],
           runtime_dirs=runtime_parent['runtime_dirs'],sources=sources,programs=programs,jobs=jobs,hardware=hardware,protocol=protocol(),
           source_sha256=sha(source),callback_source_sha256=sha(callback),actor_source_sha256=sha(actors),
           source_context_policy='scope binds actual actor specs/native+callback/runtime/topology and semantic template; intervals are explicit controlled inputs',
           whole_LLM_dynamic_accuracy_claimed=False,parameters_fitted=False)
    write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),OFF_gate_processes=6,ON_processes=18,heldout_processes=6,hardware_started=False)


def verify(native,request,binding,pid):
    require(native['schema']=='dynamic-external-stage-native-8' and native['request']==request and native['process_id']==pid,'native identity/request differs')
    for key in ('source_sha256','callback_source_sha256','actor_source_sha256'):require(native[key]==request[key],'binary/source identity differs: '+key)
    require(native['program_sha256']==binding['program_sha256'] and native['attention_action_index']==22,'fixed mathematical template differs')
    require(native['complete_attention_graph_only'] and native['complete_LLM_strategy_replayed'] is False and native['timed_result_checked_before_reset'],'target scope changed')
    require(native['internal_GPU_work_added'] is False and native['API_response_is_CPU_active'] is False and native['activity_capture_instantiated'] is False,'response boundary relabeled')
    require(native['target_entry_controller_passed'] and native['profile_window_contained'],'entry late or full graph escaped steady window; unsupported, no cell filling')
    require(native['target_thread_affinity']['observed_mask']==native['target_thread_affinity']['final_mask']==4,'target CPU2 affinity differs')
    require(native['actors']['cpu']['thread_affinity']['observed_mask']==4 and native['actors']['gpu']['thread_affinity']['observed_mask']==65536,'actor host placement differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'native mathematical/copy oracle failed')
    import numpy as np
    oracle,ports=read(binding['reference']['path']),native['verification_native']['ports']
    require(set(ports)==set(oracle),'independent numerical port inventory differs')
    for key in oracle:
        a,b=np.asarray(ports[key]),np.asarray(oracle[key]);require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,atol=5e-5,rtol=5e-5),'independent port mismatch '+key)
    phases={};previous=native['target_begin_tick'];hz=native['clock']['frequency_hz']
    require(len(native['blocks'])==request['replays']//request['block_replays'],'full block count differs')
    for index,block in enumerate(native['blocks']):
        require(block['block']==index and block['replays']==request['block_replays'] and previous<=block['begin_tick']<=block['end_tick']<=native['target_end_tick'],'block clock differs')
        require(len(block['actions'])==request['block_replays'],'block action count differs')
        for j,row in enumerate(block['actions']):
            replay=index*request['block_replays']+j;pid_phase=replay+1
            require(row['action_index']==22 and row['global_replay']==replay and row['callback_phase_id']==pid_phase,'target phase inventory differs')
            require(previous<=row['begin_tick']<row['end_tick']<=block['end_tick'],'target action clock differs')
            require(abs((row['end_tick']-row['begin_tick'])*1e9/hz-row['wall_ns'])<=1,'target QPC duration differs')
            phases[pid_phase]=dict(begin_tick=row['begin_tick'],end_tick=row['end_tick']);previous=row['end_tick']
        previous=block['end_tick']
    ledger=callback_ledger(native['runtime_callbacks'],phases,hz)
    if request['split']=='profile':
        for device,bit in zip(('cpu','gpu'),request['offered_state']):
            intervals=native['actors'][device]['intervals']
            require(len(intervals)==bit,'actual actor receipt count differs')
            if bit:
                interval=intervals[0]
                require(interval['calls']>0 and interval['actual_first_tick']<=native['target_begin_tick'] and interval['actual_last_completion_tick']>=native['target_end_tick'],
                        'planned ON actor did not bracket complete target; no steady cell claim')
    return dict(passed=True,complete_Attention_replays=len(phases),callback_records=ledger['records'],
                callback_phases=ledger['phases'],mathematical_oracle=True,source_scope=True,
                performance_accuracy_certified=False)


def observer_gate(out,stage):
    """Every matched pair is retained; OFF cannot qualify ON states."""
    out=Path(out);m=read(out/'manifest.json');require(stage in ('00_observer_gate','ON_profiles'),'observer stage differs')
    groups={};missing=[]
    for job in (x for x in m['jobs'] if x['stage']==stage):
        path=out/'jobs'/job['id']/'execution.json'
        if not path.is_file():missing.append(job['id']);continue
        execution=read(path)
        if execution['status']!='completed':missing.append(job['id']);continue
        native=read(execution['native']['path']);require(sha(execution['native']['path'])==execution['native']['sha256'],'native receipt changed')
        groups.setdefault(job['pair_id'],{})[job['request']['instrumentation']]=native['target_wall_ns']
    pairs=[]
    for pair,values in sorted(groups.items()):
        complete=set(values)=={'stamped','runtime_callbacks'}
        delta=abs(values['runtime_callbacks']/values['stamped']-1) if complete else None
        pairs.append(dict(pair_id=pair,observed_target_ns=values,complete=complete,relative_difference=delta,passed=complete and delta<=.05))
    declared=3 if stage=='00_observer_gate' else 9
    result=dict(schema='attention-runtime-observer-gate-8',stage=stage,declared_pairs=declared,observed_pairs=len(pairs),missing_or_failed_jobs=missing,
                pairs=pairs,passed=not missing and len(pairs)==declared and all(p['passed'] for p in pairs),
                threshold=.05,OFF_gate_qualifies_ON=False,parameters_fitted=False,heldout_accuracy_evaluated=False)
    write(out/(stage+'-observer-gate.json'),result);return result


def authorization(out,stage):
    m=read(out/'manifest.json');require(stage in ('00_observer_gate','ON_profiles','heldout'),'unknown acquisition stage')
    auth=read(out/(stage+'-authorization.json'))
    require(auth.get('collection_authorized') is True and auth.get('stage')==stage and auth.get('manifest_sha256')==sha(out/'manifest.json'),'root must authorize exact manifest/stage')
    if stage in ('ON_profiles','heldout'):
        off=read(out/'00_observer_gate-observer-gate.json');require(off['passed'] is True,'OFF observer gate failed; no automatic ON collection')
    if stage=='heldout':
        on=read(out/'ON_profiles-observer-gate.json');require(on['passed'] is True,'ON observer qualification missing')
        frozen=read(out/'frozen-barrier.json');require(auth.get('frozen_barrier_sha256')==sha(out/'frozen-barrier.json'),'root authorization not bound to frozen prediction')
        require(frozen.get('manifest_sha256')==sha(out/'manifest.json') and frozen.get('all_exact_state_cells_supported') is True
                and frozen.get('parameters_and_all_predictions_frozen') is True and frozen.get('whole_target_time_used_in_fit') is False,'profile support and independent prediction freeze required')
        require(isinstance(frozen.get('files'),dict) and bool(frozen['files']),'explicit frozen profile/prediction files required')
        for name,digest in frozen['files'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(out/p)==digest,'frozen file absent/changed/outside campaign')
    return m


def process_creation_100ns(process):
    """Exact FILETIME on the borrowed owned handle; no PID-time approximation."""
    class FT(ctypes.Structure):_fields_=[('low',ctypes.c_uint32),('high',ctypes.c_uint32)]
    function=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
    function.argtypes=[ctypes.c_void_p,*([ctypes.POINTER(FT)]*4)];function.restype=ctypes.c_int
    values=[FT() for _ in range(4)]
    require(function(int(process._handle),*[ctypes.byref(x) for x in values]),'owned process creation FILETIME unavailable')
    return (values[0].high<<32)|values[0].low


def collect(out,stage):
    out=Path(out).resolve();m=authorization(out,stage);marker=out/(stage+'-started.json');require(not marker.exists(),'stage already started; no retries')
    for source in m['sources']:
        for key in ('source','archived'):require(sha(source[key]['path'])==source[key]['sha256'],'bound source changed')
    for binding in [m['binary'],m['build_receipt'],*m['runtime']]:require(sha(binding['path'])==binding['sha256'],'bound runtime/binary/build receipt changed')
    for b in m['programs'].values():
        for key in ('program','reference'):require(sha(b[key]['path'])==b[key]['sha256'],'bound program/reference changed')
    write(marker,dict(started_at_utc=base.e2a.utc(),stage=stage));records=[];jobs=[j for j in m['jobs'] if j['stage']==stage]
    with base.run.measurement_lock():
        for job in jobs:
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request'])
            binding=m['programs'][job['split']];command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],stage=stage,status='failed',command=command,request=receipt(folder/'request.json'),model_fitted=False)
            process,handles=None,[]
            try:
                env=base.run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
                process,a,b,affinity=base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b]
                record.update(process_id=process.pid,process_creation_time_100ns=process_creation_100ns(process),process_affinity=affinity)
                code=process.wait(timeout=90 if job['split']=='profile' else 120);record.update(returncode=code,process_exited_verified=True)
                require(code==0,'owned native failed; no retry or substitution')
                record['verification']=verify(read(folder/'result.json'),job['request'],binding,process.pid)
                record.update(status='completed',native=receipt(folder/'result.json'))
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
            records.append(record);print(json.dumps(dict(job=job['id'],status=record['status'],error=record.get('error'))),flush=True)
            if record['status']!='completed':break
    result=dict(schema='attention-runtime-acquisition-stage-8',stage=stage,declared=len(jobs),acquired=len(records),completed=sum(r['status']=='completed' for r in records),
                failed=sum(r['status']!='completed' for r in records),not_run=len(jobs)-len(records),all_native_processes_exited=all(r.get('process_exited_verified',False) for r in records),
                executions=[receipt(out/'jobs'/r['job_id']/'execution.json') for r in records],prediction_accuracy_evaluated=False)
    write(out/(stage+'-acquisition.json'),result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect','observer-gate'));p.add_argument('--out');p.add_argument('--binary');p.add_argument('--cupti-dll');p.add_argument('--build-receipt');p.add_argument('--stage',choices=('00_observer_gate','ON_profiles','heldout'));a=p.parse_args()
    if a.command=='protocol':r=protocol()
    elif a.command=='prepare':r=prepare(a.out,a.binary,cupti_dll=a.cupti_dll,build_receipt=a.build_receipt)
    elif a.command=='observer-gate':r=observer_gate(a.out,a.stage)
    else:r=collect(a.out,a.stage)
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
