"""Independent real-MoE window calibration and bare-request validation.

Root orchestration owns hardware windows. Every native process takes the common
measurement lock and retains the original capacity policy and no-retry rule.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil

from .dataset import read,write
from .runtime_window_io import (child_environment,file_sha,receipt,sha,require,available,PLACEMENT_LABELS,LABELS,
    LEGACY_LABELS,LABEL_PLACEMENT,LABEL_FOR_PLACEMENT,PROBE_PLACEMENT,PLACEMENT_ENVIRONMENT,GRAPH_POLICIES)
from profiling.state_calibration import run as common
from profiling import host

ROOT=Path(__file__).resolve().parents[2]
# Two inputs of the published cohort are build artifacts of the machine that ran
# it: the numerical campaign's manifest (model, capacity policy, guards, suite)
# and the request template derived from that run's tokenizer pass. `prepare_portable`
# answers exactly these two paths from explicit arguments so a second machine can
# prepare the same cohort without owning the first machine's build tree.
LEGACY_DESCRIPTOR=ROOT/'build/real-moe-validation-20261028/mmid-fence-numerical-v2/manifest.json'
LEGACY_TEMPLATE=ROOT/'build/real-moe-validation-20261027/numerical-diagnostic-v6/requests/normal-SSLS.json'
# The build tree's own layout is a property of the host: `bin/` holds the probe
# and its libraries, and the probe's name carries `.exe` only on Windows.
PROBE_STEM='real-moe-runtime-window-probe'
TOOLKIT=host.newest_toolkit()
DEFAULT_CUDA_DIR=str(TOOLKIT/'bin') if TOOLKIT is not None else ''


def probe_receipt(binaries):
    """The probe among a build's binaries, and the libraries it loads beside it."""
    name=host.executable_name(PROBE_STEM).lower()
    binary=next(r for r in binaries if Path(r['path']).name.lower()==name)
    return binary,[r for r in binaries if Path(r['path']).match(host.shared_library_pattern())]

PLACEMENTS=tuple(placement for placement,_ in PLACEMENT_LABELS)
ADOPTED_JOBS=tuple(label+'-'+kind for label in LABELS for kind in ('calibration-profile','calibration-bare','inventory'))

def select(suite,split,index=0):
    require(isinstance(index,int) and not isinstance(index,bool) and index>=0,'nonnegative fixed stratum index required')
    rows=[]
    for domain in ('dialogue','code','math'):
        for length in ('short','long'):
            eligible=sorted((r for r in suite['requests'] if r['domain']==domain and r['length_class']==length and r['split']==split),key=lambda r:r['id'])
            require(len(eligible)>index,'missing frozen request stratum/index '+domain+'/'+length+'/'+split+'/'+str(index))
            rows.append(copy.deepcopy(eligible[index]))
    require(len({r['id'] for r in rows})==6,'six distinct requests required')
    return rows

def request(m,rows,placement,observe,*,warmups=1):
    r=copy.deepcopy(m['template'])
    # The observed binary is bound to its own placement vocabulary; the campaign
    # label travels separately so a placement that only differs in scheduler
    # policy keeps a distinct price key without changing the frozen executable.
    r.update(schema='real-moe-runtime-window-request-1',source_sha256=m['source_sha256'],mode='generate',
        placement=PROBE_PLACEMENT.get(placement,placement),analysis_label=LABEL_FOR_PLACEMENT[placement],
        # One declared runtime policy for the whole cohort: every placement is
        # measured under it, so a row never differs from its neighbour by
        # something other than the placement itself.
        cuda_graph_policy=m.get('graph_policy','runtime_default_enabled'),
        observe_logits=False,observe_graph=False,runtime_window_observer=observe,warmup_replays=warmups)
    r['requests']=[]
    for row in rows:
        row=copy.deepcopy(row);row['max_output_tokens']=8
        for field in ('output_token_ids','logit_artifact_path','context_reset_index'):row.pop(field,None)
        r['requests'].append(row)
    return r

def prepare(out,build,adopt_from=None,heldout_index=0,adopt_calibration_only=False,resident_peak_limit_bytes=None,resident_peak_reason=None,labels=None,graph_policy='runtime_default_enabled',runtime_dirs=None):
    out,build=Path(out).resolve(),Path(build).resolve();require(not out.exists(),'new real-MoE performance cohort required')
    require(graph_policy in GRAPH_POLICIES,'unknown declared runtime graph policy')
    old=read(LEGACY_DESCRIPTOR)
    # The resident guard exists to stop a process from thrashing the host during
    # measurement. A cohort that needs a larger peak must say so here, in the
    # manifest, instead of editing a frozen number afterwards.
    amendment=None
    resident_limit=old['resident_peak_limit_bytes']
    if resident_peak_limit_bytes is not None:
        require(isinstance(resident_peak_limit_bytes,int) and resident_peak_limit_bytes>resident_limit,'resident peak amendment may only raise the frozen budget')
        require(resident_peak_limit_bytes<=old['parent_preflight_bytes'],'resident peak amendment must stay within the parent preflight budget')
        require(isinstance(resident_peak_reason,str) and len(resident_peak_reason)>=20,'resident peak amendment needs a stated reason')
        amendment=dict(field='resident_peak_limit_bytes',old_value=resident_limit,new_value=resident_peak_limit_bytes,reason=resident_peak_reason,
            measurement_semantics_changed=False,still_below_parent_preflight_bytes=old['parent_preflight_bytes'])
        resident_limit=resident_peak_limit_bytes
    template=read(LEGACY_TEMPLATE)
    suite=read(old['source_suite']['path']);complete=read(build/'BUILD_COMPLETE.json');prepared=read(build/'PREPARED.json')
    require(complete['compiled'] and complete['extra_GPU_syncs']==0 and complete['CPU_barriers_added']==0,'source-bound no-sync build required')
    out.mkdir(parents=True)
    binary,runtime=probe_receipt(complete['binaries'])
    manifest=dict(schema='real-moe-runtime-window-campaign-1',prepared_at_utc=datetime.now(timezone.utc).isoformat(),build=receipt(build/'BUILD_COMPLETE.json'),
        binary=binary,runtime=runtime,
        runtime_dirs=[str(build/'bin'),*(runtime_dirs or (DEFAULT_CUDA_DIR,))],source_sha256=prepared['native_source_sha256'],
        model=old['model'],capacity_policy=old['capacity_policy'],native_guard_bytes=old['native_guard_bytes'],parent_preflight_bytes=old['parent_preflight_bytes'],resident_peak_limit_bytes=resident_limit,resident_peak_amendment=amendment,
        source_suite=old['source_suite'],template=template,calibration=select(suite,'calibration'),heldout=select(suite,'heldout',index=heldout_index),
        heldout_selection=dict(method='lexicographically sorted ID within each fixed domain/length stratum',index=heldout_index,
            previously_used_index_zero_request_ids=[r['id'] for r in select(suite,'heldout')],performance_values_used=False),
        primary_metric='complete request / completed Prefill / post-Prefill Decode with sampling mean latency error',
        target_APE=0.10,block_CV_is_acceptance_gate=False,calibration_prices='CPU actual-node and inclusive runtime bridge windows only',
        feature_policy='mechanism_v2',
        heldout_inventory='structure-only export excludes observed durations before prices/predictions freeze',
        model_load_reset_tokenization_in_target=False,whole_model_timing_used_to_fit=False,retries=0,
        sampling_window='one declared full-request warmup, then one normal greedy request; fixed model and persistent pool')
    if heldout_index>0:
        prior_rows=select(suite,'heldout');comparison=[*manifest['calibration'],*prior_rows]
        require(not({r['id'] for r in manifest['heldout']} & {r['id'] for r in comparison}),'new heldout request IDs overlap calibration/old diagnostic requests')
        require(not({tuple(r['prompt_token_ids']) for r in manifest['heldout']} & {tuple(r['prompt_token_ids']) for r in comparison}),'new heldout prompts duplicate calibration/old diagnostic requests')
        manifest['heldout_selection'].update(disjoint_from_calibration_and_index_zero_ids=True,
            disjoint_exact_prompt_token_sequences=True,old_diagnostic_results_not_used_for_selection=True)
    source_paths={Path(__file__),Path(__file__).with_name('runtime_window_io.py'),Path(__file__).with_name('dataset.py'),ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/state_calibration/telemetry.py',Path(__file__).with_name('runtime_windows.py'),Path(__file__).with_name('runtime_window_analysis.py'),Path(__file__).with_name('runtime_window_predict.py'),ROOT/'modeling/llm_infer_model/performance/moe_execution.py',ROOT/'modeling/llm_infer_model/performance/window_work.py'}
    if adopt_from is not None:
        import sys
        import numpy as np
        from . import runtime_window_predict
        for module in tuple(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if name is not None:
                path=Path(name).resolve()
                if path.suffix=='.py' and path.is_relative_to(ROOT):source_paths.add(path)
        manifest['analysis_environment']=dict(python=sys.version,numpy_version=np.__version__,
            coefficient_representation='nonnegative coefficients on a common 1e-9 ns grid; mechanism-scoped features only; no heldout-time fit',
            source_scope='all imported repository Python dependencies of the analysis plus native-build/campaign sources')
    sources=[]
    for path in sorted(source_paths):
        target=out/'executed_sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources.append(dict(source=receipt(path),archived=receipt(target)))
    manifest['sources']=sources
    # A cohort declares the placements it covers. Cohorts prepared before this
    # field existed keep the three original placements, so their frozen request
    # files and price keys stay reproducible byte for byte.
    labels=tuple(LEGACY_LABELS if labels is None else labels)
    require(bool(labels) and len(set(labels))==len(labels) and all(label in LABEL_PLACEMENT for label in labels),'distinct declared campaign labels required')
    manifest['labels']=list(labels)
    # The graph policy is a cohort-wide declaration, applied by the observer itself
    # and identical for every placement, so placement is the only thing that varies
    # between rows. Cohorts prepared before this field existed keep the default.
    manifest['graph_policy']=graph_policy
    manifest['graph_policy_scope']='applied by the native observer to its own process for every placement in this cohort; not exported by the parent shell'
    requests=out/'requests';requests.mkdir()
    for label in labels:
        placement=LABEL_PLACEMENT[label]
        for split,observe in (('calibration-profile',True),('calibration-bare',False),('inventory',True),('heldout-r0',False),('heldout-r1',False)):
            rows=manifest['calibration'] if split.startswith('calibration') else manifest['heldout']
            write(requests/(label+'-'+split+'.json'),request(manifest,rows,placement,observe))
        write(requests/(label+'-smoke.json'),request(manifest,[manifest['calibration'][0]],placement,True,warmups=0))
        write(requests/(label+'-smoke-bare.json'),request(manifest,[manifest['calibration'][0]],placement,False,warmups=0))
    manifest['requests']={path.stem:receipt(path) for path in sorted(requests.glob('*.json'))}
    if adopt_from is not None:
        previous=Path(adopt_from).resolve();old_manifest=read(previous/'manifest.json')
        for field in ('model','binary','runtime','source_sha256','capacity_policy','native_guard_bytes','parent_preflight_bytes'):
            require(manifest[field]==old_manifest[field],'analysis-only revision changed native/model/request policy: '+field)
        adopted=[]
        adopted_jobs=tuple(job for job in ADOPTED_JOBS if not(adopt_calibration_only and job.endswith('-inventory')))
        for job in adopted_jobs:
            source=previous/'jobs'/job;execution=read(source/'execution.json')
            require(execution['status']=='completed' and execution['returncode']==0,'only complete native evidence can be adopted: '+job)
            require(sha(execution['native']['path'])==execution['native']['sha256'],'adopted native evidence changed')
            require(sha(source/'request.json')==manifest['requests'][job]['sha256'],'adopted request differs from new frozen job')
            target=out/'jobs'/job;shutil.copytree(source,target)
            adopted.append(dict(job_id=job,original_execution=receipt(source/'execution.json'),
                original_native=execution['native'],original_request=receipt(source/'request.json'),
                adopted_execution=receipt(target/'execution.json'),adopted_native_copy=receipt(target/'result.json'),
                source_native_paths_preserved=True))
        manifest['analysis_only_revision']=dict(source_manifest=receipt(previous/'manifest.json'),
            source_freeze=receipt(previous/'FREEZE_BARRIER.json'),adopted_hardware_evidence=adopted,
            native_processes_repeated=False,revision_reasons=(['mechanism-scoped feature policy from actual operators and transfer calls','same calibration raw only; old diagnostic heldout is not reused for new validation','fixed disjoint heldout stratum index'] if adopt_calibration_only else ['source-forward attribution and node occurrence','uniform coefficient representation','linear exact static serial evaluation','dependency and prediction-freeze evidence']),
            calibration_only_adoption=adopt_calibration_only,heldout_native_jobs_must_be_new=True,
            old_heldout_used_for_fitting=False,old_heldout_retained_in_source_campaign=True)
    write(out/'manifest.json',manifest)
    write(out/'PLAN.json',dict(calibration_jobs=[p+'-'+s for p in labels for s in ('calibration-profile','calibration-bare')],inventory_jobs=[p+'-inventory' for p in labels],heldout_jobs=[p+'-heldout-r'+str(i) for i in (0,1) for p in labels],
        root_window_authorization_required=True,numerical_or_timing_outcomes_not_used_for_request_selection=True))
    return dict(manifest=receipt(out/'manifest.json'),requests=len(manifest['requests']),hardware_executed=False)

def freeze(out,tests):
    out=Path(out).resolve();t=read(tests);require(t['returncode']==0 and t['hardware_executed'] is False,'passing software-only tests required')
    write(out/'FREEZE_BARRIER.json',dict(manifest=receipt(out/'manifest.json'),tests=receipt(tests),hardware_executed=False,root_collection_authorization_required=True))
    return receipt(out/'FREEZE_BARRIER.json')

def verify_prediction_freeze(out):
    """A bare heldout launch is impossible before bound predictions exist."""
    out=Path(out);path=out/'PREDICTION_FREEZE.json'
    require(path.is_file(),'freeze predictions before launching any bare heldout')
    frozen=read(path)
    require(frozen['manifest']==receipt(out/'manifest.json'),'prediction freeze belongs to another campaign')
    for resource in (frozen['predictions'],frozen['prices'],frozen['price_freeze'],*frozen['inventories']):
        require(sha(resource['path'])==resource['sha256'],'prediction/prices/workload changed after freeze: '+resource['path'])
    return receipt(path)

def collect(out,job):
    out=Path(out).resolve();m=read(out/'manifest.json');barrier=read(out/'FREEZE_BARRIER.json');auth=read(out/'ROOT_AUTHORIZATION_RECEIPT.json')
    require(auth['collection_authorized'] is True and auth['manifest_sha256']==sha(out/'manifest.json') and auth['freeze_barrier_sha256']==sha(out/'FREEZE_BARRIER.json'),'exact root authorization missing')
    require(job in auth['jobs'],'job outside root-authorized window')
    reqpath=out/'requests'/(job+'.json')
    require(job in m['requests'] and sha(reqpath)==m['requests'][job]['sha256'],'frozen request changed')
    req=read(reqpath);folder=out/'jobs'/job;require(not folder.exists(),'no retry or result overwrite');folder.mkdir(parents=True)
    shutil.copyfile(reqpath,folder/'request.json');command=[m['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-runtime-window-execution-1',job_id=job,status='failed',request=receipt(folder/'request.json'),binary=m['binary'],runtime=m['runtime'],command=command,model=m['model'],capacity_policy=m['capacity_policy'],observer_interval_ms=200,
        collector_hash_buffer_bytes=1024*1024,collector_memory_before_verification=psutil.Process().memory_info()._asdict())
    process=None;handles=[];samples=[]
    with common.measurement_lock():
        try:
            for resource in [m['binary'],*m['runtime'],m['build'],m['source_suite'],m['capacity_policy'],*sum(([s['source'],s['archived']] for s in m['sources']),[])]:
                require(sha(resource['path'])==resource['sha256'],'frozen source/binary changed: '+resource['path'])
            require(file_sha(m['model']['path'])==m['model']['sha256'],'actual model content hash changed')
            if '-heldout-r' in job:record['prediction_freeze']=verify_prediction_freeze(out)
            record['collector_memory_after_complete_hashes']=psutil.Process().memory_info()._asdict()
            record['available_before_launch_bytes']=available();require(record['available_before_launch_bytes']>=m['parent_preflight_bytes'],'original host-memory preflight not met')
            env=child_environment(m['runtime_dirs'],req)
            for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):env[key]='1'
            env['NVIDIA_TF32_OVERRIDE']='0'
            policy=m.get('graph_policy','runtime_default_enabled')
            require(req['cuda_graph_policy']==policy,'request graph policy differs from the declared cohort policy')
            require(env.get('GGML_CUDA_DISABLE_GRAPHS') is None,'graph policy belongs to the declared native flag, not the parent shell')
            record['declared_graph_policy']=policy
            label=req.get('analysis_label') or LABEL_FOR_PLACEMENT.get(req['placement'],req['placement'])
            extras=PLACEMENT_ENVIRONMENT.get(label,{})
            require(not any(key in env for key in extras),'declared placement environment must not leak from the ambient shell')
            env.update(extras)
            record['declared_placement_environment']=dict(extras)
            record['declared_placement_label']=label
            record['started_at_utc']=datetime.now(timezone.utc).isoformat()
            process,a,b,affinity=common.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',req['expected_process_mask']);handles=[a,b]
            record.update(process_id=process.pid,process_affinity=affinity)
            observer=psutil.Process(process.pid);start=time.monotonic();low=0
            print(json.dumps(dict(started=job,process_id=process.pid)),flush=True)
            with (folder/'observer.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                while process.poll() is None:
                    s=dict(global_available_bytes=available(),observer_tick_ns=time.perf_counter_ns())
                    try:s['process_memory']=observer.memory_info()._asdict()
                    except(psutil.NoSuchProcess,psutil.AccessDenied):pass
                    samples.append(s);stream.write(json.dumps(s,separators=(',',':'))+'\n');stream.flush()
                    low=low+1 if s['global_available_bytes']<512*1024**2 else 0
                    if low>=2 or s.get('process_memory',{}).get('peak_wset',0)>m['resident_peak_limit_bytes'] or time.monotonic()-start>300:
                        record['abort_reason']='unchanged available/resident capacity watchdog or five-minute process limit';process.terminate();break
                    try:process.wait(timeout=.2)
                    except subprocess.TimeoutExpired:pass
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            record.update(returncode=process.returncode,process_exited_verified=True)
            require(process.returncode==0,'real native process failed; preserve failure')
            native=read(folder/'result.json')
            require(native['schema']=='real-moe-runtime-window-native-1' and native['source_sha256']==m['source_sha256'],'native schema/source mismatch')
            require(native['process_id']==process.pid and native['request']==req,'native process/request binding differs')
            require(native['model_contract_verified'] is True and len(native['results'])==len(req['requests']),'complete real-model request inventory missing')
            for row,r in zip(native['results'],req['requests']):
                require(row['prompt_token_ids']==r['prompt_token_ids'] and row['output_count']>=1,'actual inputs/outputs missing')
                require(row['forward_count']==len(row['forward_inventory']),'forward inventory differs')
                require(row['runtime_windows']['added_GPU_synchronizations']==0,'new GPU synchronization introduced')
                if req['runtime_window_observer']:require(row['runtime_windows']['events'],'profile observer emitted no windows')
                else:require(row['timing_usable_as_bare_reference'] is True and not row['runtime_windows']['events'],'heldout is not unrecorded normal execution')
            record.update(status='completed',native=receipt(folder/'result.json'),target_intervals='normal output masks; excludes load/reset/tokenization/warmup; includes sampling')
        except BaseException as error:
            record['error']=repr(error)
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
        finally:
            for h in handles:h.close()
            record['observed_min_available_bytes']=min((s['global_available_bytes'] for s in samples),default=None)
            write(folder/'execution.json',record)
    print(json.dumps(dict(job_id=job,status=record['status'],error=record.get('error'))),flush=True)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','freeze','collect'));p.add_argument('--out',required=True);p.add_argument('--build');p.add_argument('--tests');p.add_argument('--job');p.add_argument('--adopt-from');p.add_argument('--heldout-index',type=int,default=0);p.add_argument('--adopt-calibration-only',action='store_true');p.add_argument('--resident-peak-limit-bytes',type=int);p.add_argument('--resident-peak-reason');p.add_argument('--labels');p.add_argument('--graph-policy',default='runtime_default_enabled');a=p.parse_args()
    labels=tuple(x for x in a.labels.split(',') if x) if a.labels else None
    r=prepare(a.out,a.build,a.adopt_from,a.heldout_index,a.adopt_calibration_only,a.resident_peak_limit_bytes,a.resident_peak_reason,labels,a.graph_policy) if a.command=='prepare' else freeze(a.out,a.tests) if a.command=='freeze' else collect(a.out,a.job)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
