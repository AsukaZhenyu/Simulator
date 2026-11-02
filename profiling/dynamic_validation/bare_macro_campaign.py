"""Original completed Attention response under planned CPU/GPU offered work.

The fine API observer failed its own neutrality gate. This separate provider
uses no callback labels: its unit is one complete, numerical-checked Attention
component. All three existing OFF stamped observations are retained. New ON
profiles and unseen-seed pulse executions use the exact same stamped boundary.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from . import external_stage_v8 as native_campaign
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action

ROOT=native_campaign.ROOT
read,write,sha,receipt,require=native_campaign.read,native_campaign.write,native_campaign.sha,native_campaign.receipt,native_campaign.require
STATES=((0,0),(1,0),(0,1),(1,1))


def protocol():
    return dict(schema='bare-attention-macro-protocol-1',target='one complete original Executor.action Attention component',
        whole_LLM_dynamic_accuracy_claimed=False,fine_API_provider_qualified=False,
        fine_observer_failure_retained=True,callback_timing_labels_used=False,
        instrumentation='same original stamped action boundary for calibration and heldout; callbacks disabled',
        existing_OFF_calibration_processes_reused=3,OFF_selection_or_averaging_of_callback_cases=False,
        new_ON_calibration_processes=9,three_distinct_contexts_per_state=True,
        single_observation_per_context=True,three_single_observations_claim_low_variance=False,
        independent_heldout_processes=6,heldout_replays=256,heldout_block_replays=16,
        heldout_new_mathematical_seed=True,heldout_pulses=native_campaign.PULSES,
        CPU_actor_and_target_same_physical_core=True,GPU_actor_submission_CPU=16,
        ON_window_ms=[0,30000],target_entry_ms=1000,actor_stop_target_dependent=False,
        actor_workload='same retained CPU/GPU GEMM256; offered workload bits are not utilization percentages',
        response_parameters='arithmetic mean of completed component invocation in each offered state',
        calibration_uses_whole_LLM_target=False,calibration_uses_heldout_target=False,
        providers=['static00','memoryless_four_state_macro'],tau='unknown; no stateful provider or guessed transition constant',
        calculation='modeling.PlannedCompletedResponse/common evaluator; own recursively predicted completion clock',
        actual_heldout_boundaries_used_to_reanchor_prediction=False,
        acceptance=dict(process_mean_APE_max=.10,within_process_block_APE_P90_max=.20,
                        raw_population_CV_reported_separately=True,all_declared_denominators_retained=True),
        unknown_cells_are_zero=False,retries=0,slow_observations_removed=0)


def plans(parent_manifest,parent_dir):
    jobs=[]
    off=[j for j in parent_manifest['jobs'] if j['split']=='profile' and j['offered_state']==[0,0] and j['request']['instrumentation']=='stamped']
    require(len(off)==3,'all three OFF stamped calibration cases required')
    for source in off:
        job=copy.deepcopy(source);job.update(imported=True,stage='profile',imported_execution=receipt(parent_dir/'jobs'/job['id']/'execution.json'));jobs.append(job)
    for round_id in range(3):
        states=STATES[1:];states=states[round_id:]+states[:round_id]
        for state in states:
            matches=[j for j in parent_manifest['jobs'] if j['split']=='profile' and tuple(j['offered_state'])==state and j['round']==round_id and j['request']['instrumentation']=='stamped']
            require(len(matches)==1,'exact original ON stamped request required');job=copy.deepcopy(matches[0]);job.update(imported=False,stage='profile');jobs.append(job)
    for source in (j for j in parent_manifest['jobs'] if j['split']=='heldout'):
        job=copy.deepcopy(source);require(job['request']['instrumentation']=='stamped','heldout must be callback-free stamped');job.update(imported=False);jobs.append(job)
    require(len(jobs)==18 and len({j['id'] for j in jobs})==18,'complete12profile+6heldout inventory required')
    return jobs


def verified_job(out,job,manifest):
    path=Path(job['imported_execution']['path']) if job['imported'] else Path(out)/'jobs'/job['id']/'execution.json'
    if job['imported']:require(sha(path)==job['imported_execution']['sha256'],'imported OFF execution changed')
    execution=read(path);require(execution['status']=='completed' and execution['process_exited_verified'] is True,'declared component process failed/missing')
    require(sha(execution['native']['path'])==execution['native']['sha256'],'native response changed')
    native=read(execution['native']['path']);binding=manifest['programs'][job['split']]
    require(job['request']['instrumentation']=='stamped' and native['instrumentation']=='stamped','callback labels cannot be used for macro provider')
    proof=native_campaign.verify(native,job['request'],binding,execution['process_id'])
    require(proof['callback_records']==0 and proof['callback_phases']==0,'macro provider must not import fine callback observations')
    require(execution.get('process_creation_time_100ns',0)>0,'exact process identity required')
    return native,execution,path


def prepare(out,parent):
    out,parent=Path(out).resolve(),Path(parent).resolve();require(not out.exists(),'new exclusive macro campaign required')
    p=read(parent/'manifest.json');gate=read(parent/'00_observer_gate-observer-gate.json')
    require(gate['passed'] is False,'retain original failed fine observer result')
    require(not (parent/'ON_profiles-started.json').exists() and not (parent/'heldout-started.json').exists(),'old fine-stage ON/heldout acquisition must remain unused')
    jobs=plans(p,parent)
    for job in (j for j in jobs if j['imported']):verified_job(parent,job,p)
    program=read(p['programs']['profile']['program']['path']);held=read(p['programs']['heldout']['program']['path'])
    semantic=semantic_action(program,22);require(semantic==semantic_action(held,22) and program['seed']!=held['seed'],'same component geometry/unseen mathematical seed required')
    template=jobs[0]['request'];actors={}
    for device in ('cpu','gpu'):
        actors[device]={k:v for k,v in template[device+'_actor'].items() if k!='intervals_ms'}
    context=dict(binary_sha256=p['binary']['sha256'],runtime=p['runtime'],topology=p['hardware'],source_sha256=p['source_sha256'],callback_source_sha256=p['callback_source_sha256'],actor_source_sha256=p['actor_source_sha256'],
        actors=actors,process_mask=4294967295,target_thread_mask=4,threads=1,initialization_policy='pinned_init',
        boundary='completed original Attention invocation; stamped outer boundary; fine callbacks disabled',
        clocks='Windows QPC and predeclared independent actor epoch',warmup_iterations=20,warmup_min_ms=500)
    scope=dict(context_sha256=digest(context),semantic_sha256=digest(semantic),context=context,semantic_coordinate=semantic)
    sources=[Path(__file__),Path(__file__).with_name('bare_macro_fit.py')]
    unused_fine={'external_stage_v8_fit.py','runtime_stage_profile.py','runtime_stage_records.py','runtime_response.py'}
    sources += [Path(x['source']['path']) for x in p['sources'] if Path(x['source']['path']).name not in unused_fine]
    sources += [ROOT/'modeling/llm_infer_model/performance/reusable_action_costs.py']
    # Deduplicate shared dependencies while preserving the exact frozen original sources.
    sources=list(dict.fromkeys(sources));require(all(x.is_file() for x in sources),'complete macro software required before freeze')
    out.mkdir(parents=True);archived=[]
    for source in sources:
        target=out/'executed_sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);archived.append(dict(source=receipt(source),archived=receipt(target)))
    m=dict(schema='bare-attention-macro-campaign-1',prepared_at_utc=native_campaign.base.e2a.utc(),parent_manifest=receipt(parent/'manifest.json'),
        failed_fine_observer_gate=receipt(parent/'00_observer_gate-observer-gate.json'),binary=p['binary'],build_receipt=p['build_receipt'],runtime=p['runtime'],runtime_dirs=p['runtime_dirs'],
        hardware=p['hardware'],programs=p['programs'],source_sha256=p['source_sha256'],callback_source_sha256=p['callback_source_sha256'],actor_source_sha256=p['actor_source_sha256'],
        jobs=jobs,sources=archived,scope=scope,protocol=protocol(),fine_API_provider_qualified=False)
    write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),imported_OFF=3,new_ON=9,heldout=6,hardware_executed=False)


def check(out):
    m=read(out/'manifest.json')
    for name in ('parent_manifest','failed_fine_observer_gate','binary','build_receipt'):
        resource=m[name];require(sha(resource['path'])==resource['sha256'],'original resource changed '+name)
    require(read(m['failed_fine_observer_gate']['path'])['passed'] is False and m['fine_API_provider_qualified'] is False,'fine observer failure must remain failed')
    for item in m['runtime']:require(sha(item['path'])==item['sha256'],'runtime changed')
    for source in m['sources']:
        for name in ('source','archived'):require(sha(source[name]['path'])==source[name]['sha256'],'bound software changed')
    for binding in m['programs'].values():
        for name in ('program','reference'):require(sha(binding[name]['path'])==binding[name]['sha256'],'input/reference changed')
    for job in (j for j in m['jobs'] if j['imported']):verified_job(out,job,m)
    return m


def authorization(out,split):
    require(split in ('profile','heldout'),'unknown acquisition split');m=check(out);a=read(out/f'{split}-authorization.json')
    require(a.get('collection_authorized') is True and a.get('split')==split and a.get('manifest_sha256')==sha(out/'manifest.json'),'root must authorize exact macro manifest/split')
    if split=='heldout':
        frozen=read(out/'frozen-barrier.json')
        require(a.get('frozen_barrier_sha256')==sha(out/'frozen-barrier.json') and frozen['manifest_sha256']==sha(out/'manifest.json'),'heldout authorization must bind frozen forecasts')
        require(frozen['profile_supported'] and frozen['parameters_frozen'] and frozen['predictions_frozen'] and frozen['calibration_target_is_component'] and frozen['whole_LLM_target_time_used_for_fit'] is False,'complete component cells and target-blind freeze required')
        for name,value in frozen['files'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(out/p)==value,'frozen provider changed')
    return m


def collect(out,split):
    out=Path(out).resolve();m=authorization(out,split);marker=out/f'{split}-started.json';require(not marker.exists(),'no retries/repeated split');write(marker,dict(started_at_utc=native_campaign.base.e2a.utc(),split=split))
    executions=[];newjobs=[j for j in m['jobs'] if j['split']==split and not j['imported']]
    with native_campaign.base.run.measurement_lock():
        for job in newjobs:
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request']);binding=m['programs'][split]
            command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],split=split,status='failed',command=command,request=receipt(folder/'request.json'),component_macro=True,fine_callback_labels_used=False);process=None;handles=[]
            try:
                env=native_campaign.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=native_campaign.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b]
                record.update(process_id=process.pid,process_creation_time_100ns=native_campaign.process_creation_100ns(process),process_affinity=affinity)
                code=process.wait(timeout=90 if split=='profile' else 120);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                record['verification']=native_campaign.verify(read(folder/'result.json'),job['request'],binding,process.pid)
                require(record['verification']['callback_records']==0,'callback data cannot enter macro calibration')
                record.update(status='completed',native=receipt(folder/'result.json'))
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
    return dict(declared_new=len(newjobs),acquired=len(executions),completed=sum(e['status']=='completed' for e in executions),not_run=len(newjobs)-len(executions),imported_OFF=3 if split=='profile' else 0,fine_provider_qualified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','collect'));p.add_argument('--out');p.add_argument('--parent');p.add_argument('--split',choices=('profile','heldout'));a=p.parse_args()
    result=protocol() if a.command=='protocol' else prepare(a.out,a.parent) if a.command=='prepare' else collect(a.out,a.split)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
