"""Constant-state component calibration, with unchanged dynamic entry gates.

Version 1 incorrectly applied a future-epoch arrival deadline to a steady-state
price. This version validates the *actual entire component* inside the planned
constant window and actual actor execution. The original deadline failure and
false native flag remain recorded. Dynamic heldout still requires the original
10 ms deadline, without measured-start reanchoring or relaxed accuracy limits.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

from . import bare_macro_campaign as previous
from . import external_stage_v8 as native_campaign
from llm_infer_model.performance.reusable_action_costs import digest

ROOT=previous.ROOT
read,write,sha,receipt,require=previous.read,previous.write,previous.sha,previous.receipt,previous.require
STATES=previous.STATES
LATE_JOB='profile-10-r0-stamped'
OLD_ERROR="ValueError('entry late or full graph escaped steady window; unsupported, no cell filling')"


def protocol():
    value=previous.protocol()
    value.update(schema='bare-attention-macro-protocol-2',new_ON_calibration_processes=8,
        existing_OFF_calibration_processes_reused=3,existing_ON_calibration_processes_reused=1,
        old_deadline_failure_retained=True,calibration_criterion='actual entire completed component inside constant planned state and bracketed by actual ON actor execution',
        calibration_start_lateness='recorded; not a constant-state price support criterion',
        dynamic_heldout_start_deadline_ns=10000000,dynamic_entry_deadline_unchanged=True,
        performance_thresholds_unchanged=True,native_records_or_flags_rewritten=False,
        calibration_import_policy='all three original OFF and the first ON previously rejected solely for arrival deadline; no response-value selection')
    return value


def verify_constant_profile(native,request,binding,pid):
    """Full native/mathematical/state checks; never alter a source record."""
    require(request['split']=='profile' and request['instrumentation']=='stamped','constant stamped calibration only')
    require(native['schema']=='dynamic-external-stage-native-8' and native['request']==request and native['process_id']==pid,'native identity/request differs')
    for key in ('source_sha256','callback_source_sha256','actor_source_sha256'):require(native[key]==request[key],'binary/source differs '+key)
    require(native['program_sha256']==binding['program_sha256'] and native['attention_action_index']==22,'mathematical template differs')
    require(native['instrumentation']=='stamped' and native['runtime_callbacks']['enabled'] is False and native['runtime_callbacks']['record_count']==0,'fine callback labels forbidden')
    require(native['complete_attention_graph_only'] is True and native['complete_LLM_strategy_replayed'] is False and native['timed_result_checked_before_reset'] is True,'complete component scope missing')
    for flag in ('internal_GPU_work_added','API_response_is_CPU_active','activity_capture_instantiated','future_target_state_used','actor_timeline_triggered_by_target_latency','performance_parameters_fitted'):
        require(native[flag] is False,'scope broadened '+flag)
    require(native['process_affinity']['observed_mask']==native['process_affinity']['final_mask']==4294967295,'process affinity differs')
    for field in ('target_thread_affinity','initialization_affinity'):
        require(native[field]['observed_mask']==native[field]['final_mask']==4,'target/init affinity differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'native numerical/copy oracle failed')
    import numpy as np
    oracle,ports=read(binding['reference']['path']),native['verification_native']['ports'];require(set(oracle)==set(ports),'numerical port inventory differs')
    for name in oracle:
        a,b=np.asarray(oracle[name]),np.asarray(ports[name]);require(a.shape==b.shape and np.all(np.isfinite(b)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent numerical mismatch '+name)
    require(native['clock']['clock_id']=='windows_qpc','actual QPC clock required');hz=native['clock']['frequency_hz'];require(type(hz) is int and hz>0,'positive QPC frequency required')
    state=request['offered_state'];require(type(state) is list and len(state)==2 and all(type(x) is int and x in (0,1) for x in state),'explicit state required')
    begin,end,epoch=native['target_begin_tick'],native['target_end_tick'],native['target_expected_origin_tick']
    require(all(type(t) is int for t in (begin,end,epoch)) and epoch<=begin<end,'target/actor epoch differs')
    require(native['steady_window_begin_tick']==epoch and native['steady_window_end_tick']==epoch+30*hz,'planned constant state window differs')
    require(native['profile_window_contained'] is True and epoch<=begin<end<=epoch+30*hz,'component escaped constant planned window')
    planned=epoch+(1000 if any(state) else 0)*hz//1000
    require(native['planned_target_entry_tick']==planned and begin>=planned,'declared steady entry differs')
    late=Fraction((begin-planned)*10**9,hz)
    require(abs(late-native['initial_start_lateness_ns'])<=1,'recorded lateness differs from QPC')
    require(type(native['target_entry_controller_passed']) is bool and native['target_entry_controller_passed']==(late<=10000000),'native deadline flag was rewritten or inconsistent')
    require(abs(Fraction((end-begin)*10**9,hz)-native['target_wall_ns'])<=1,'target wall/QPC differs')
    require(request['replays']==request['block_replays']==native['continuous_replays']==1 and len(native['blocks'])==1,'one complete calibration component required')
    block=native['blocks'][0];require(block['block']==0 and block['replays']==1 and begin<=block['begin_tick']<block['end_tick']<=end and len(block['actions'])==1,'component block inventory/clock differs')
    require(abs(Fraction((block['end_tick']-block['begin_tick'])*10**9,hz)-block['wall_ns'])<=1,'block QPC differs')
    action=block['actions'][0]
    require(action['action_index']==22 and action['global_replay']==0 and action['callback_phase_id']==1 and block['begin_tick']<=action['begin_tick']<action['end_tick']<=block['end_tick'],'actual component action differs')
    require(abs(Fraction((action['end_tick']-action['begin_tick'])*10**9,hz)-action['wall_ns'])<=1,'component QPC differs')
    for device,bit,mask in zip(('cpu','gpu'),state,(4,65536)):
        actor=native['actors'][device]
        require(actor['thread_affinity']['observed_mask']==actor['thread_affinity']['final_mask']==mask and actor['ready_tick']<=epoch,'actor placement/readiness differs')
        require(actor['repeated_output_equal'] is True and actor['cancelled'] is False,'actor numerical/state execution failed')
        expected=[[0,30000]] if bit else []
        require(request[device+'_actor']['intervals_ms']==expected and len(actor['intervals'])==bit,'planned state interval differs')
        if bit:
            interval=actor['intervals'][0]
            require(interval['planned_begin_ms']==0 and interval['planned_end_ms']==30000 and interval['calls']>0,'constant ON actor receipt missing')
            require(interval['actual_first_tick']<=begin<end<=interval['actual_last_completion_tick'],'actual ON actor did not bracket entire component')
    ledger=native_campaign.callback_ledger(native['runtime_callbacks'],{1:dict(begin_tick=action['begin_tick'],end_tick=action['end_tick'])},hz)
    require(ledger['records']==0 and ledger['phases']==0,'fine callback observations present')
    return dict(passed=True,complete_Attention_replays=1,callback_records=0,callback_phases=0,mathematical_oracle=True,
        constant_actual_state_brackets_entire_component=True,profile_window_contained=True,
        initial_start_lateness_ns=native['initial_start_lateness_ns'],original_native_entry_deadline_passed=native['target_entry_controller_passed'],
        deadline_is_dynamic_heldout_gate_only=True,performance_accuracy_certified=False)


def verified_job(out,job,manifest):
    path=Path(job['imported_execution']['path']) if job['imported'] else Path(out)/'jobs'/job['id']/'execution.json'
    if job['imported']:require(sha(path)==job['imported_execution']['sha256'],'imported execution changed')
    execution=read(path);require(execution['returncode']==0 and execution['process_exited_verified'] is True,'native failed/incomplete')
    if 'native' not in execution:
        require(job['imported'] and job['id']==LATE_JOB and execution['status']=='failed' and execution.get('error')==OLD_ERROR,
                'missing native receipt is supported only for the retained original arrival-deadline failure')
        bound=job['imported_native'];require(Path(bound['path']).resolve()==Path(execution['command'][-1]).resolve()
            and sha(bound['path'])==bound['sha256'],'new manifest must bind the exact unmodified original command output')
        # An explicit view supplies missing provenance; original status/error and
        # immutable execution.json/native result bytes remain unchanged.
        execution=dict(execution,native=bound,original_execution=receipt(path),native_binding_origin='new v2 manifest; original record omitted receipt after verification exception')
    require(sha(execution['native']['path'])==execution['native']['sha256'],'native record changed')
    native=read(execution['native']['path']);binding=manifest['programs'][job['split']]
    if job['split']=='profile':
        proof=verify_constant_profile(native,job['request'],binding,execution['process_id'])
        if execution['status']!='completed':
            require(job['imported'] and job['id']==LATE_JOB and execution['status']=='failed' and execution.get('error')==OLD_ERROR
                    and proof['original_native_entry_deadline_passed'] is False,'only the explicit retained old deadline failure may be reclassified for steady-state calibration')
    else:
        require(execution['status']=='completed','heldout failed/incomplete')
        proof=native_campaign.verify(native,job['request'],binding,execution['process_id'])
        require(proof['callback_records']==0,'heldout callback prices forbidden')
    require(execution.get('process_creation_time_100ns',0)>0,'exact process identity required')
    return native,execution,path


def prepare(out,parent):
    out,parent=Path(out).resolve(),Path(parent).resolve();require(not out.exists(),'new exclusive v2 directory required')
    p=read(parent/'manifest.json');require(not (parent/'heldout-started.json').exists(),'independent heldout already used')
    jobs=copy.deepcopy(p['jobs']);late=next(j for j in jobs if j['id']==LATE_JOB)
    require(late['imported'] is False,'explicit first ON import differs');late.update(imported=True,imported_execution=receipt(parent/'jobs'/LATE_JOB/'execution.json'),imported_native=receipt(parent/'jobs'/LATE_JOB/'result.json'),
        import_reason='old macro-v1 arrival deadline failure; actual steady component state separately validated without modifying old status')
    require(sum(j['imported'] for j in jobs)==4,'exact three OFF plus one first ON reuse required')
    for job in (j for j in jobs if j['imported']):verified_job(parent,job,p)
    scope=copy.deepcopy(p['scope']);scope['context']['constant_calibration_support']='actual actor execution brackets full component within fixed30s window; native arrival lateness retained; heldout10msunchanged';scope['context_sha256']=digest(scope['context'])
    sources=[Path(__file__),Path(__file__).with_name('bare_macro_v2_fit.py')]+[Path(s['source']['path']) for s in p['sources']]
    sources=list(dict.fromkeys(sources));require(all(s.is_file() for s in sources),'complete v2 source required')
    out.mkdir(parents=True);archived=[]
    for source in sources:
        target=out/'executed_sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);archived.append(dict(source=receipt(source),archived=receipt(target)))
    m={key:copy.deepcopy(p[key]) for key in ('binary','build_receipt','runtime','runtime_dirs','hardware','programs','source_sha256','callback_source_sha256','actor_source_sha256','failed_fine_observer_gate')}
    m.update(schema='bare-attention-macro-campaign-2',prepared_at_utc=native_campaign.base.e2a.utc(),parent_macro_manifest=receipt(parent/'manifest.json'),
        retained_original_deadline_failure=receipt(parent/'jobs'/LATE_JOB/'execution.json'),parent_manifest=p['parent_manifest'],jobs=jobs,sources=archived,scope=scope,protocol=protocol(),fine_API_provider_qualified=False)
    write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),imported_OFF=3,imported_late_ON=1,new_ON=8,heldout=6,hardware_executed=False)


def check(out):
    m=read(out/'manifest.json')
    for name in ('parent_manifest','parent_macro_manifest','retained_original_deadline_failure','failed_fine_observer_gate','binary','build_receipt'):
        resource=m[name];require(sha(resource['path'])==resource['sha256'],'bound original resource changed '+name)
    require(read(m['retained_original_deadline_failure']['path'])['status']=='failed' and read(m['failed_fine_observer_gate']['path'])['passed'] is False and m['fine_API_provider_qualified'] is False,'original failures must remain unchanged')
    for resource in m['runtime']:require(sha(resource['path'])==resource['sha256'],'runtime changed')
    for source in m['sources']:
        for name in ('source','archived'):require(sha(source[name]['path'])==source[name]['sha256'],'frozen source changed')
    for binding in m['programs'].values():
        for name in ('program','reference'):require(sha(binding[name]['path'])==binding[name]['sha256'],'input/reference changed')
    for job in (j for j in m['jobs'] if j['imported']):verified_job(out,job,m)
    return m


def authorization(out,split):
    require(split in ('profile','heldout'),'unknown split');m=check(out);a=read(out/f'{split}-authorization.json')
    require(a.get('collection_authorized') is True and a.get('split')==split and a.get('manifest_sha256')==sha(out/'manifest.json'),'root authorization for exact v2 split/manifest required')
    if split=='heldout':
        f=read(out/'frozen-barrier.json');require(a.get('frozen_barrier_sha256')==sha(out/'frozen-barrier.json') and f['manifest_sha256']==sha(out/'manifest.json'),'heldout authorization not bound to freeze')
        require(f['profile_supported'] and f['parameters_frozen'] and f['predictions_frozen'] and f['calibration_target_is_component'] and f['whole_LLM_target_time_used_for_fit'] is False,'complete target-blind component forecast freeze required')
        for name,value in f['files'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(out/p)==value,'frozen forecast changed')
    return m


def collect(out,split):
    out=Path(out).resolve();m=authorization(out,split);marker=out/f'{split}-started.json';require(not marker.exists(),'no retries/repeated split');write(marker,dict(started_at_utc=native_campaign.base.e2a.utc(),split=split))
    jobs=[j for j in m['jobs'] if j['split']==split and not j['imported']];executions=[]
    with native_campaign.base.run.measurement_lock():
        for job in jobs:
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request']);binding=m['programs'][split]
            command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],split=split,status='failed',command=command,request=receipt(folder/'request.json'),component_macro=True,constant_profile_gate_separate_from_dynamic_deadline=True);process=None;handles=[]
            try:
                env=native_campaign.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=native_campaign.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=process.pid,process_creation_time_100ns=native_campaign.process_creation_100ns(process),process_affinity=affinity)
                code=process.wait(timeout=90 if split=='profile' else 120);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                native=read(folder/'result.json')
                record['verification']=verify_constant_profile(native,job['request'],binding,process.pid) if split=='profile' else native_campaign.verify(native,job['request'],binding,process.pid)
                require(record['verification']['callback_records']==0,'fine callback observations forbidden');record.update(status='completed',native=receipt(folder/'result.json'))
            except BaseException as error:
                record['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                write(folder/'execution.json',record)
            executions.append(record);print(json.dumps(dict(job_id=job['id'],status=record['status'],error=record.get('error'))),flush=True)
            if record['status']!='completed':break
    write(out/f'{split}-executions.json',executions)
    return dict(declared_new=len(jobs),acquired=len(executions),completed=sum(e['status']=='completed' for e in executions),not_run=len(jobs)-len(executions),imported_calibration=4 if split=='profile' else 0,fine_API_provider_qualified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect'));p.add_argument('--out');p.add_argument('--parent');p.add_argument('--split',choices=('profile','heldout'));a=p.parse_args()
    value=protocol() if a.command=='protocol' else prepare(a.out,a.parent) if a.command=='prepare' else collect(a.out,a.split)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
