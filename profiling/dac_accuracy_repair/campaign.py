"""Frozen, independently validated transfer and complete-program repair study.

Old measurements are read only as mathematical programs and reference bytes;
old timing outcomes do not fit any new parameter. No failed job is retried.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np

from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable, coordinate, digest, semantic_action
from profiling.dac_validation import e2a
from profiling.dac_followup.task1 import TRAIN, HELDOUT, byte_reference
from profiling.state_calibration import run

ROOT = Path(__file__).resolve().parents[2]
read, write, sha, receipt, require = e2a.read, e2a.write, e2a.sha, e2a.receipt, e2a.require
API = 'ggml_tensor_copy_then_gpu_cpu_synchronize'
COMPLETION = 'host_response_including_copy_api_and_gpu_cpu_completion'
SPLITS = ('transfer-profile', 'profile', 'transfer-heldout', 'dense')


def sampling(full=False):
    return dict(e2a.request('full' if full else 'action'), repeats=8 if full else 256)


def request(mode, source_sha, **kwargs):
    return dict(schema='dac-accuracy-repair-request-1', mode=mode,
                source_sha256=source_sha, sampling=sampling(not mode.startswith('transfer')),
                order='ABBA', **kwargs)


def prepare(out, binary, process_mask):
    from profiling.state_response.collect import hardware_identity
    out, binary = Path(out).resolve(), Path(binary).resolve()
    require(not out.exists(), 'new output directory required; old evidence cannot be overwritten')
    require(process_mask > 0 and process_mask & 4, 'explicit process mask must allow the owned main thread')
    previous = read(ROOT/'build/dac-refinement-20261026/campaign-v1/manifest.json')
    native = ROOT/'profiling/dac_accuracy_repair/native/repair_probe.cpp'
    # Every implementation is frozen before the first calibration process.
    paths = [Path(__file__), native, native.parent/'CMakeLists.txt',
             ROOT/'profiling/dac_accuracy_repair/fit.py',
             ROOT/'modeling/llm_infer_model/performance/prepared_transfer.py',
             ROOT/'modeling/llm_infer_model/performance/transfer_response.py',
             ROOT/'modeling/llm_infer_model/performance/reusable_action_costs.py',
             ROOT/'profiling/dac_followup/transfer_fit.py',
             ROOT/'profiling/dac_followup/task1.py', ROOT/'profiling/dac_validation/e2a.py',
             ROOT/'profiling/state_calibration/run.py',
             ROOT/'profiling/state_response/collect.py',
             ROOT/'profiling/native/steady_costs/probe.cpp',
             ROOT/'profiling/native/steady_costs/components.hpp']
    paths += list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    require(all(p.exists() for p in paths), 'repair implementation is incomplete')
    out.mkdir(parents=True)
    sources = []
    for path in paths:
        destination = out/'executed_sources'/path.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources.append(dict(source=receipt(path), archived=receipt(destination)))
    target = out/'executed_binary'/binary.name
    target.parent.mkdir(); shutil.copy2(binary, target)
    runtime_dirs = previous['runtime_dirs']
    runtime = [receipt(p) for folder in runtime_dirs for p in Path(folder).glob('*.dll')]
    require(runtime, 'explicit runtime DLL evidence is required')
    context = dict(previous['context'], hardware=hardware_identity(), binary_sha256=sha(binary),
                   process_affinity_mask=process_mask,
                   response_context='independent hot action mean; Dense reset plus one complete prime; fixed input',
                   boundary='unchanged completed Executor.action')
    programs, jobs, seen = {}, [], set()
    groups = {}
    for key, binding in previous['programs'].items():
        if not key.startswith('l'): continue
        for field in ('program', 'reference'):
            require(sha(binding[field]['path']) == binding[field]['sha256'], 'retained mathematical input changed')
        programs[key] = binding
        if binding['split'] != 'profile': continue
        program = read(binding['program']['path']); indices = []
        for index in range(len(program['actions'])):
            h = digest(coordinate(program, index, context))
            if h not in seen: seen.add(h); indices.append(index)
        if indices: groups[key] = indices
    for key, binding in programs.items():
        if binding['split'] != 'heldout': continue
        program = read(binding['program']['path'])
        require(all(digest(coordinate(program, i, context)) in seen for i in range(len(program['actions']))),
                'heldout exact semantic action is absent from the independent calibration plan')
    for round_id in range(3):
        for key, indices in groups.items():
            jobs.append(dict(id=f'profile-r{round_id}-{key}', split='profile', program_key=key,
                             request=request('hot-actions', sha(native), indices=indices)))
    # Both host-memory modes are planned in advance. Registered is never silently
    # substituted for the unchanged pageable complete Dense program.
    for split, sizes in (('transfer-profile', TRAIN), ('transfer-heldout', HELDOUT)):
        for round_id in range(3):
            sizes_order = list(sizes[round_id:]+sizes[:round_id])
            if round_id % 2: sizes_order.reverse()
            for size in sizes_order:
                for memory in (('pageable', 'registered') if round_id % 2 == 0 else ('registered', 'pageable')):
                    key = f'd2h-{split}-{size}'
                    if key not in programs:
                        program = dict(format_version='steady-component-program-1', kind='transfer', device='gpu',
                                       threads=1, cuda_device=0, seed=2026102606 if split.endswith('profile') else 2026102607,
                                       shape=dict(bytes=size, src='vram', dst='dram'), allocation_budget_bytes=1 << 30)
                        program['program_sha256'] = digest(program)
                        pp, rp = out/'programs'/(key+'.json'), out/'references'/(key+'.json')
                        write(pp, program); write(rp, byte_reference(program['seed'], size))
                        programs[key] = dict(program=receipt(pp), reference=receipt(rp), bytes=size, split=split)
                    req = request('transfer-mode', sha(native), memory_mode=memory)
                    req['order'] = 'ABBA' if round_id % 2 == 0 else 'BAAB'
                    jobs.append(dict(id=f'{split}-{memory}-r{round_id}-{size}', split=split,
                                     program_key=key, memory_mode=memory, request=req))
    # Primary/observer are adjacent fresh processes with the same preparation,
    # rather than two acquisition phases with different reset/prime semantics.
    for round_id in range(4):
        for layers in ((1, 2) if round_id % 2 == 0 else (2, 1)):
            for strategy in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))[round_id]:
                key = f'l{layers}-h512-p64-s{strategy}-heldout'
                for mode in (('dense-primary', 'dense-observer') if round_id % 2 == 0 else ('dense-observer', 'dense-primary')):
                    req = request(mode, sha(native)); req['order'] = 'ABBA' if round_id % 2 == 0 else 'BAAB'
                    jobs.append(dict(id=f'{mode}-r{round_id}-{key}', split='dense', mode=mode,
                                     program_key=key, request=req))
    for job in jobs:job['request']['expected_process_mask']=process_mask
    manifest = dict(schema='dac-accuracy-repair-campaign-1', prepared_at_utc=e2a.utc(),
                    binary=receipt(target), original_binary=receipt(binary), sources=sources,
                    native_source_sha256=sha(native), runtime=runtime, runtime_dirs=runtime_dirs,
                    context=context, process_affinity_mask=process_mask, main_thread_mask=4,
                    programs=programs, jobs=jobs,
                    protocol=dict(total_processes=len(jobs), counts={s:sum(j['split']==s for j in jobs) for s in SPLITS},
                        quality=e2a.QUALITY, profile_groups=groups,
                        transfer_train_sizes=list(TRAIN), transfer_heldout_sizes=list(HELDOUT),
                        transfer_modes=['pageable', 'registered'], full_dense_memory_mode='pageable',
                        repeats={'transfer':256,'dense':8}, trials=8, warmup_min_ms=500,
                        per_child_timeout_seconds=300, retries=0,
                        selected_primary_provider='fresh independently measured semantic hot action table',
                        transfer_replacement='prespecified secondary; batch-mean-as-single-call assumption is explicit',
                        target_time_used_for_fit=False, original_results_modified=False,
                        passive_state_used_as_future_target_input=False,
                        registration_cost='outside timed steady target; retained transition evidence; not free system initialization'))
    write(out/'manifest.json', manifest)
    return dict(out=str(out), counts=manifest['protocol']['counts'], total=len(jobs))


def check_sources(manifest):
    for source in manifest['sources']:
        require(sha(source['source']['path'])==source['source']['sha256'], 'bound implementation changed')
        require(sha(source['archived']['path'])==source['archived']['sha256'], 'archived implementation changed')
    require(sha(manifest['binary']['path'])==manifest['binary']['sha256'], 'frozen executable changed')
    for item in manifest['runtime']:
        require(sha(item['path'])==item['sha256'], 'explicit runtime changed')


def verify(program, oracle, req, native, pid):
    require(native['schema']=='dac-accuracy-repair-native-1' and native['request']==req, 'native request identity differs')
    require(native['process_id']==pid and native['source_sha256']==req['source_sha256'] and native['caller_source_sha256_match'], 'native process/source differs')
    require(native['program_sha256']==program['program_sha256'], 'mathematical program identity differs')
    affinity=native['main_thread_affinity']
    require(affinity['requested_mask']==affinity['observed_mask']==affinity['final_mask']==4 and affinity['restored'], 'owned main-thread affinity differs')
    process_affinity=native['process_affinity']
    require(process_affinity['requested_mask']==process_affinity['observed_mask']==process_affinity['final_mask']==req['expected_process_mask']
            and process_affinity['enforced'] and process_affinity['applied_before_backend_initialization'],
            'explicit process affinity was not established before backend initialization')
    rows=list(native['trials']); extra=[]
    if req['mode']=='transfer-mode':
        require(native['measurement_api']==API and native['completion_semantic']==COMPLETION, 'copy API boundary differs')
        require(native['host_registration_restored'] and len(native['visits'])==1, 'host registration restore/visit differs')
        visit=native['visits'][0]; memory=req['memory_mode']
        require(visit['memory_mode']==memory and visit['host_pointer']['registered_host']==(memory=='registered'), 'actual host memory differs')
        proof=visit['full_byte_verification']
        require(proof['full_byte_equality'] and proof['checked_bytes']==oracle['bytes'] and proof['source_sha256']==proof['output_sha256']==oracle['sha256'], 'independent complete byte verification failed')
        require(len(rows)==req['sampling']['trials']*4, 'copy block count differs')
    else:
        proof=native['verification_native']
        require(proof['copy_values_equal'] and proof['actions']==program['actions'] and native['native_repeat_consistency']['passed'], 'Executor COPY/action/replay contract failed')
        require(set(proof['ports'])==set(oracle), 'numerical port inventory differs')
        for key, wanted in oracle.items():
            actual=np.asarray(proof['ports'][key]); wanted=np.asarray(wanted)
            require(actual.shape==wanted.shape and np.all(np.isfinite(actual)) and np.allclose(actual,wanted,atol=5e-5,rtol=5e-5), 'independent numerical oracle failed '+key)
        if req['mode']=='hot-actions':
            require([g['action_index'] for g in native['groups']]==req['indices'], 'calibration action inventory differs')
            extra=[g['trials'] for g in native['groups']]
        else:
            require(len(rows)==req['sampling']['trials']*(4 if req['mode']=='dense-observer' else 1), 'Dense block count differs')
            require(all(row['preparation_contract']==native['preparation_contract'] for row in rows), 'per-block Dense preparation differs')
    hz=native['clock']['frequency_hz']
    for group in [rows,*extra]:
        for row in group:
            require(row['end_tick']>row['begin_tick'] and row['repeats']==req['sampling']['repeats'], 'QPC order or repetitions differ')
            require(abs((row['end_tick']-row['begin_tick'])*1e9/hz-row['wall_ns'])<=1, 'QPC response ledger differs')
    return dict(passed=True, full_numerical_or_byte_oracle=True)


def collect(out, split):
    out=Path(out).resolve(); m=read(out/'manifest.json'); check_sources(m)
    require(split in SPLITS, 'unknown declared acquisition split')
    barrier=None
    if split in ('transfer-heldout','dense'):
        barrier=read(out/'frozen-barrier.json')
        for name,h in barrier['files'].items():require(sha(out/name)==h, 'frozen parameters changed')
    marker=out/(split+'-started.json')
    if not marker.exists():write(marker,dict(started_at_utc=e2a.utc(),barrier=barrier,manifest=receipt(out/'manifest.json')))
    jobs=[j for j in m['jobs'] if j['split']==split]; executions=[]
    with run.measurement_lock():
        for job in jobs:
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists():executions.append(read(folder/'execution.json'));continue
            folder.mkdir(parents=True,exist_ok=False); binding=m['programs'][job['program_key']]
            for name in ('program','reference'):require(sha(binding[name]['path'])==binding[name]['sha256'], 'prepared input changed')
            req=folder/'request.json'; output=folder/'result.json'; write(req,job['request'])
            command=[m['binary']['path'],binding['program']['path'],str(req),str(output)]
            execution=dict(job=job,started_at_utc=e2a.utc(),status='failed',command=command,barrier=barrier)
            process=None;handles=[]
            try:
                env=run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
                process,stdout,stderr,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',m['process_affinity_mask']);handles=[stdout,stderr]
                execution.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=m['protocol']['per_child_timeout_seconds'])
                execution.update(returncode=code,process_exited_verified=True,forced_termination=False)
                require(code==0, 'owned native child failed; no retry')
                execution['verification']=verify(read(binding['program']['path']),read(binding['reference']['path']),job['request'],read(output),process.pid)
                execution.update(status='completed',native=receipt(output))
            except BaseException as error:
                execution['error']=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    execution['forced_termination']=True
                if process is not None:execution.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                execution['completed_at_utc']=e2a.utc();write(folder/'execution.json',execution)
            executions.append(execution)
            print(dict(job=job['id'],status=execution['status'],error=execution.get('error')),flush=True)
    path=out/(split+'-executions.json')
    if not path.exists():write(path,executions)
    return dict(split=split,declared=len(jobs),completed=sum(e['status']=='completed' for e in executions),failed=sum(e['status']!='completed' for e in executions))


def transfer_rows(out, split):
    out=Path(out); m=read(out/'manifest.json'); rows=[]
    for execution in read(out/(split+'-executions.json')):
        require(execution['status']=='completed', 'missing transfer evidence; parameter remains unknown')
        native=read(execution['native']['path']);visit=native['visits'][0]
        bare=[r['wall_ns']/r['repeats'] for r in native['trials'] if not r['actual_stamped']]
        stamped=[r['wall_ns']/r['repeats'] for r in native['trials'] if r['actual_stamped']]
        rows.append(dict(split='calibration' if split=='transfer-profile' else 'heldout',
                         bytes=m['programs'][execution['job']['program_key']]['bytes'],
                         memory_mode=execution['job']['memory_mode'],measurement_api=native['measurement_api'],
                         completion_semantic=native['completion_semantic'],preparation_contract=native['preparation_contract'],
                         host_pointer=visit['host_pointer'],mean_ns=statistics.mean(bare),
                         trial_ns=bare,cv=statistics.stdev(bare)/statistics.mean(bare),
                         observer_effect=statistics.mean(stamped)/statistics.mean(bare)-1,
                         phases_ns={key:statistics.mean(r[key]/r['repeats'] for r in native['trials'] if r['actual_stamped']) for key in ('copy_call_ns','gpu_sync_ns','cpu_sync_ns')},
                         preblock_state=[r['preblock_state'] for r in native['trials']],raw=execution['native']))
    return rows


def transfer_contract(manifest, row):
    return dict(route='vram->dram',memory_mode=row['memory_mode'],measurement_api=API,completion_semantic=COMPLETION,
                preparation_contract=row['preparation_contract'],scope_sha256=digest(manifest['context']))


def freeze(out):
    from profiling.dac_accuracy_repair.fit import freeze_training_profile
    from llm_infer_model.performance.prepared_transfer import PreparedTransferModel
    out=Path(out).resolve();m=read(out/'manifest.json');check_sources(m)
    require(not any((out/(s+'-started.json')).exists() for s in ('transfer-heldout','dense')), 'fit after target acquisition rejected')
    provenance=dict(hardware=m['context']['hardware'],binary=m['binary'],runtime=m['runtime'],sources=[s['archived'] for s in m['sources']])
    training=transfer_rows(out,'transfer-profile');transfers={}
    for mode in ('pageable','registered'):
        rows=[r for r in training if r['memory_mode']==mode];contract=transfer_contract(m,rows[0])
        for row in rows:row['contract']=contract
        path=out/f'transfer-{mode}-frozen.json'
        freeze_training_profile(path,rows,contract=contract,provenance=provenance)
        transfers[mode]=read(path)
    coords={};samples={};evidence={}
    for execution in read(out/'profile-executions.json'):
        require(execution['status']=='completed','missing independent action price')
        program=read(m['programs'][execution['job']['program_key']]['program']['path']);native=read(execution['native']['path'])
        for group in native['groups']:
            index=group['action_index'];coord=coordinate(program,index,m['context']);key=digest(coord)
            coords[key]=coord;samples.setdefault(key,[]).extend(Fraction(str(r['wall_ns']))/r['repeats'] for r in group['trials'])
            evidence.setdefault(key,[]).append(dict(native=execution['native'],action_index=index,split='independent profile'))
    entries=[dict(key=key,coordinate=coords[key],mean_ns=str(sum(values,Fraction())/len(values)),sample_count=len(values),evidence=evidence[key]) for key,values in sorted(samples.items())]
    profile=dict(entries=entries,context=m['context'],frozen_at_utc=e2a.utc(),target_time_used_for_fit=False)
    write(out/'action-profile-frozen.json',profile);table=ReusableResponseTable(tuple(entries));predictions=[]
    model=PreparedTransferModel.from_profile(transfers['pageable'])
    for key,binding in m['programs'].items():
        if binding['split']!='heldout':continue
        program=read(binding['program']['path']);base=table.evaluate(program,m['context']);replacement=[];assumptions=[]
        for i,value in enumerate(base['action_ns']):
            semantic=semantic_action(program,i)
            if semantic['kind']=='COPY' and semantic['src']=='vram' and semantic['dst']=='dram':
                contract=dict(transfers['pageable']['contract'])
                contract['preparation_contract']=dict(contract['preparation_contract'],timed_replays_per_batch=1)
                prediction=model.prediction(semantic['tensor']['storage_bytes'],contract=contract,
                    allow_unqualified=True,allow_batch_mean_as_single_call=True)
                value=Fraction(str(prediction['duration_ns']));assumptions.extend(prediction.get('scope_assumptions',[]))
                assumptions.append('isolated transfer preparation and observed component host-memory category are assumed for in-program COPY; no prefix or Dense arena registration qualification')
            replacement.append(value)
        predictions.append(dict(program_key=key,layers=binding['layers'],strategy=binding['strategy'],
            hot_ns=str(base['duration_ns']),transfer_secondary_ns=str(sum(replacement,Fraction())),
            transfer_secondary_assumptions=sorted(set(assumptions)),target_time_used=False))
    write(out/'predictions.json',predictions)
    names=['action-profile-frozen.json','transfer-pageable-frozen.json','transfer-registered-frozen.json','predictions.json']
    write(out/'frozen-barrier.json',dict(frozen_at_utc=e2a.utc(),files={n:sha(out/n) for n in names},heldout_read=False))
    return dict(action_coordinates=len(entries),predictions=len(predictions),transfer_modes=list(transfers))


def percentile(values, q):return e2a.quantile(values,q)


def evaluate(out):
    from llm_infer_model.performance.prepared_transfer import PreparedTransferModel
    out=Path(out).resolve();m=read(out/'manifest.json');check_sources(m);barrier=read(out/'frozen-barrier.json')
    for file,h in barrier['files'].items():require(sha(out/file)==h,'frozen parameter or prediction changed')
    transfer=transfer_rows(out,'transfer-heldout');transfer_results={}
    for mode in ('pageable','registered'):
        frozen=read(out/f'transfer-{mode}-frozen.json');model=PreparedTransferModel.from_profile(frozen)
        rows=[r for r in transfer if r['memory_mode']==mode];sizes=[];processes=[]
        for row in rows:
            contract=transfer_contract(m,row)
            prediction=model.prediction(row['bytes'],contract=contract,allow_unqualified=True)['duration_ns']
            processes.append(dict(row,predicted_ns=prediction,ape=abs(prediction-row['mean_ns'])/row['mean_ns']))
        for size in HELDOUT:
            group=[r for r in processes if r['bytes']==size];observed=statistics.mean(r['mean_ns'] for r in group)
            sizes.append(dict(bytes=size,observed_ns=observed,predicted_ns=group[0]['predicted_ns'],
                ape=abs(group[0]['predicted_ns']-observed)/observed,
                maximum_process_ape=max(r['ape'] for r in group),
                between_process_cv=statistics.stdev(r['mean_ns'] for r in group)/observed,
                same_input_ten_percent_feasible=max(.9*r['mean_ns'] for r in group)<=min(1.1*r['mean_ns'] for r in group)))
        maximum=max(r['ape'] for r in sizes);cv_passes=sum(r['cv']<=.1 for r in processes);obs_passes=sum(abs(r['observer_effect'])<=.05 for r in processes)
        p90=percentile([abs(r['predicted_ns']-v)/v for r in processes for v in r['trial_ns']],.9)
        qualified=maximum<=.1 and cv_passes==24 and obs_passes==24
        transfer_results[mode]=dict(qualified_original_size_mean_gate=qualified,
            qualified_strict_process_mean_gate=qualified and all(r['ape']<=.1 for r in processes) and p90<=.2,
            maximum_size_mean_ape=maximum,maximum_process_ape=max(r['ape'] for r in processes),trial_p90_ape=p90,
            cv_passes=cv_passes,observer_passes=obs_passes,process_count=len(processes),sizes=sizes,processes=processes,
            source_profile=receipt(out/f'transfer-{mode}-frozen.json'),not_substituted_for_other_memory_mode=True)
    dense_executions=read(out/'dense-executions.json');program_results=[]
    for prediction in read(out/'predictions.json'):
        key=prediction['program_key'];primary=[];observers=[]
        for execution in dense_executions:
            if execution['job']['program_key']!=key:continue
            if execution['status']!='completed':continue
            native=read(execution['native']['path'])
            bare=[r['wall_ns']/r['repeats'] for r in native['trials'] if not r['actual_stamped']]
            row=dict(mean_ns=statistics.mean(bare),cv=statistics.stdev(bare)/statistics.mean(bare),trial_ns=bare,
                     preparation_contract=native['preparation_contract'],raw=execution['native'])
            if execution['job']['mode']=='dense-primary':primary.append(row)
            else:
                stamps=[r['wall_ns']/r['repeats'] for r in native['trials'] if r['actual_stamped']]
                row['observer_effect']=statistics.mean(stamps)/statistics.mean(bare)-1;observers.append(row)
        prepare_match=len(primary)==4 and len(observers)==4 and all(a['preparation_contract']==b['preparation_contract'] for a,b in zip(primary,observers))
        modes={}
        for field in ('hot_ns','transfer_secondary_ns'):
            price=float(Fraction(prediction[field]));apes=[abs(price-r['mean_ns'])/r['mean_ns'] for r in primary]
            p90=percentile([abs(price-v)/v for r in primary for v in r['trial_ns']],.9) if primary else None
            qualified=prepare_match and all(a<=.1 for a in apes) and all(r['cv']<=.1 for r in primary) and p90<=.2 and all(abs(r['observer_effect'])<=.05 for r in observers)
            modes[field]=dict(predicted_ns=price,maximum_process_ape=max(apes) if apes else None,
                mean_process_ape=statistics.mean(apes) if apes else None,trial_p90_ape=p90,fully_qualified=qualified,
                assumption=prediction['transfer_secondary_assumptions'] if field=='transfer_secondary_ns' else ['isolated-hot mean reused in prepared serial complete programs'])
        program_results.append(dict(program_key=key,layers=prediction['layers'],strategy=prediction['strategy'],
            primary=primary,observers=observers,preparation_match=prepare_match,modes=modes,
            measured_mean_ns=statistics.mean(r['mean_ns'] for r in primary) if primary else None))
    groups=[]
    for layers in (1,2):
        rows=[r for r in program_results if r['layers']==layers]
        if any(r['measured_mean_ns'] is None for r in rows):groups.append(dict(layers=layers,qualified=False));continue
        winner=min(rows,key=lambda r:r['measured_mean_ns']);modes={}
        for field in ('hot_ns','transfer_secondary_ns'):
            selected=min(rows,key=lambda r:r['modes'][field]['predicted_ns'])
            modes[field]=dict(selected=selected['strategy'],observed_mean_winner=winner['strategy'],
                regret=selected['measured_mean_ns']/winner['measured_mean_ns']-1,
                all_candidates_qualified=all(r['modes'][field]['fully_qualified'] for r in rows))
        groups.append(dict(layers=layers,modes=modes))
    executions=[e for split in SPLITS for e in read(out/(split+'-executions.json'))]
    summary=dict(schema='dac-accuracy-repair-result-1',evaluated_at_utc=e2a.utc(),D2H=transfer_results,
        programs=program_results,groups=groups,
        counts=dict(declared=len(m['jobs']),completed=sum(e['status']=='completed' for e in executions),failed=sum(e['status']!='completed' for e in executions),
            preparation_matched_programs=sum(r['preparation_match'] for r in program_results),
            hot_qualified_programs=sum(r['modes']['hot_ns']['fully_qualified'] for r in program_results),
            transfer_secondary_qualified_programs=sum(r['modes']['transfer_secondary_ns']['fully_qualified'] for r in program_results)),
        target_time_used_for_fit=False,old_results_modified=False,
        primary_provider=m['protocol']['selected_primary_provider'],context=m['context'],protocol=m['protocol'])
    write(out/'summary.json',summary)
    return dict(counts=summary['counts'],D2H={k:{f:v[f] for f in ('qualified_original_size_mean_gate','qualified_strict_process_mean_gate','maximum_size_mean_ape','cv_passes')} for k,v in transfer_results.items()},groups=groups)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare',*SPLITS,'freeze','evaluate'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--process-mask',type=lambda value:int(value,0))
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.out,args.binary,args.process_mask)
    elif args.command=='freeze':result=freeze(args.out)
    elif args.command=='evaluate':result=evaluate(args.out)
    else:result=collect(args.out,args.command)
    print(result,flush=True)


if __name__=='__main__':main()
