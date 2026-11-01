"""Finite-coordinate, independently validated completed-COPY lifecycle prices.

The v10 result remains frozen. First response is estimated independently of the
future window length. Its idle dependence is measured, not forced monotone.
All cost computation lives in modeling.performance.transfer_lifecycle.
"""
from __future__ import annotations
import argparse,copy,itertools,json,random,shutil,statistics,subprocess
from collections import defaultdict
from pathlib import Path

from . import h2d_lifecycle_v10 as prior
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from llm_infer_model.performance.transfer_lifecycle import CompletedTransferLifecycle

ROOT=prior.ROOT
read,write,sha,receipt,require=prior.read,prior.write,prior.sha,prior.receipt,prior.require
SIZES=(4096,65536,524288,1048576,2097152,4194304,8388608)
IDLES=(0,2,10)
WINDOWS=(1,16,64)
TRIALS=24

def protocol():
    return dict(schema='h2d-impulse-protocol-11',profile_processes=6,heldout_processes=6,memory_modes=['registered','pageable'],
        fresh_processes_per_mode_per_split=3,trials=TRIALS,sizes_bytes=list(SIZES),idle_ms=list(IDLES),windows=list(WINDOWS),
        target='original Executor.action COPY on separately owned eight-MiB arena',
        preparation='reset, complete original Dense graph, prescribed idle; no additional COPY primer',
        boundary=prior.BOUNDARY,model_equation='T(N,b,i)=b_m(b)+p_m(b,i)+(N-1)*b_m(b)',
        model_module='modeling.llm_infer_model.performance.transfer_lifecycle',
        parameter_estimation='sustained response from all nonfirst COPY labels; first difference from all first labels independent of future N',
        idle_interpolation=False,byte_interpolation='explicit continuous piecewise analytical curve; qualification only at independently measured bytes',
        qualified_window_lengths=[16,64],first_coordinate='memory/bytes/exact declared idle, pooled across future window lengths',
        first_samples_per_coordinate_per_split=216,window_samples_per_coordinate_per_split=72,
        primary_first_condition_mean_APE_max=.10,primary_window_condition_mean_APE_max=.10,raw_CV_is_prediction_gate=False,
        independent_processes_and_profile_heldout_inputs=True,previously_used_but_distinct_program_seeds=True,
        same_coordinates_across_profile_and_heldout=True,no_unseen_coordinate_claim=True,
        all_raw_data_retained=True,slow_trials_removed=0,no_retries=True,whole_LLM_target_fitting=False,
        future_N_is_first_response_input=False,prior_v10_data_and_flags_unchanged=True,
        scope='seven finite byte coordinates, declared 0/2/10ms idle and original full-graph-prime/runtime/host category; no arbitrary idle/preparation or physical bandwidth claim')

def conditions():
    return [dict(size_index=s,idle_index=i,window_index=n) for s,i,n in itertools.product(range(len(SIZES)),range(len(IDLES)),range(len(WINDOWS)))]

def plans(programs,source_sha):
    jobs=[];cs=conditions();ci={(c['size_index'],c['idle_index'],c['window_index']):i for i,c in enumerate(cs)}
    permutations=list(itertools.permutations(range(len(WINDOWS))))
    for split in ('profile','heldout'):
        for round_id in range(3):
            for memory in (('registered','pageable') if round_id%2==0 else ('pageable','registered')):
                rng=random.Random(2026102811+1000*round_id+(0 if split=='profile' else 10000));orders=[]
                for trial in range(TRIALS):
                    coordinates=list(itertools.product(range(len(SIZES)),range(len(IDLES))));rng.shuffle(coordinates)
                    permutation=permutations[(trial+round_id)%len(permutations)]
                    orders.append([ci[s,i,n] for s,i in coordinates for n in permutation])
                sampling=prior.v9.base.e2a.request('action',trials=TRIALS,target=programs[split]['target_action_index']);sampling['repeats']=64
                request=dict(schema='dynamic-h2d-impulse-request-11',source_sha256=source_sha,split=split,sampling=sampling,
                    memory_mode=memory,expected_process_mask=4294967295,sizes_bytes=list(SIZES),idle_ms=list(IDLES),windows=list(WINDOWS),
                    conditions=cs,condition_orders=orders,performance_boundary='completed_original_Executor.action_COPY',telemetry_between_bursts=False)
                jobs.append(dict(id=f'{split}-{memory}-r{round_id}',split=split,memory_mode=memory,round=round_id,request=request))
    return jobs

def prepare(out,binary,build_receipt):
    out,binary,build_receipt=Path(out).resolve(),Path(binary).resolve(),Path(build_receipt).resolve()
    require(not out.exists() and binary.is_file(),'new output and compiled binary required')
    source=Path(__file__).parent/'native/v11/h2d_impulse.cpp';build=read(build_receipt)
    require(build['binary']['sha256']==sha(binary) and build['source_sha256']==sha(source),'binary/source/build receipt differ')
    parent_path=ROOT/'build/dynamic-validation-20261026/idle-v2b/manifest.json';parent=read(parent_path);programs=copy.deepcopy(parent['programs'])
    require(programs['profile']['program_sha256']!=programs['heldout']['program_sha256'],'distinct mathematical inputs required')
    for p in programs.values():
        for key in ('program','reference'):require(sha(p[key]['path'])==p[key]['sha256'],'retained input/reference changed')
    sources=[Path(__file__),source,source.parent/'CMakeLists.txt',Path(prior.__file__),Path(prior.v9.__file__),
        ROOT/'modeling/llm_infer_model/performance/transfer_lifecycle.py',ROOT/'modeling/llm_infer_model/performance/parameters.py',
        ROOT/'profiling/dynamic_validation/native/v3/common.hpp',ROOT/'profiling/dynamic_validation/native/owned_timer.hpp',
        ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',ROOT/'profiling/native/highlevel_accuracy/highlevel_probe.cpp']
    sources+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    out.mkdir(parents=True);archived=[]
    for p in sources:
        target=out/'executed_sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);archived.append(dict(source=receipt(p),archived=receipt(target)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    context=dict(runtime=parent['runtime'],binary=receipt(target),hardware=parent['hardware'],boundary=prior.BOUNDARY,process_mask=4294967295,main_thread_mask=4,threads=1,arena_bytes=8388608)
    profile=read(programs['profile']['program']['path']);held=read(programs['heldout']['program']['path'])
    actions=[semantic_action(profile,i) for i in range(len(profile['actions']))];require(actions==[semantic_action(held,i) for i in range(len(held['actions']))],'preparation semantics differ')
    prep=dict(actions=actions,lifecycle='reset, full original graph, declared idle; no extra transfer primer',arena='separate owned eight-MiB source from repeated profile/heldout tensor values',original_graph_arena='unchanged pageable',observer=False)
    manifest=dict(schema='h2d-impulse-manifest-11',prepared_at_utc=prior.v9.base.e2a.utc(),parent_manifest=receipt(parent_path),programs=programs,binary=receipt(target),build_receipt=receipt(build_receipt),
        runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],hardware=parent['hardware'],sources=archived,scope=dict(context_sha256=digest(context),preparation_sha256=digest(prep),context=context,preparation=prep),
        jobs=plans(programs,sha(source)),protocol=protocol())
    write(out/'manifest.json',manifest);return dict(manifest=receipt(out/'manifest.json'),hardware_executed=False)

def validate(native,request,binding,pid):
    require(native['schema']=='dynamic-h2d-impulse-native-11' and native['request']==request and native['process_id']==pid,'native request/process differs')
    require(native['source_sha256']==request['source_sha256'] and native['program_sha256']==binding['program_sha256'],'binary/input differs')
    require(all(native[k] is False for k in ('CUPTI_initialized','CUDA_events_created','NVML_initialized','telemetry_between_bursts','performance_parameters_fitted')),'observer/fitting boundary differs')
    memory='registered_host' if request['memory_mode']=='registered' else 'unregistered_pageable'
    require(native['source_pointer_kind'][memory] and native['arena_registration']['during'][memory] and native['arena_registration']['restored'],'memory category differs')
    require(native['source_buffer_host'] and not native['target_buffer_host'],'host/device backing differs')
    require(native['process_affinity']['observed_mask']==native['process_affinity']['final_mask']==4294967295 and native['main_thread_affinity']['observed_mask']==native['main_thread_affinity']['final_mask']==4,'affinity differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'] and native['full_payload']['full_byte_equality'],'numerical payload differs')
    require(native['full_payload']['bytes']==max(SIZES) and native['full_payload']['arena_bytes']==8388608,'payload extent differs')
    oracles=native['slice_copy_oracles'];require({x['bytes'] for x in oracles}==set(SIZES) and all(x['selected_range_initially_different'] and x['selected_bytes_equal_after_copy'] and x['outside_view_unchanged'] for x in oracles),'range oracle failed')
    rows=native['bursts'];require(len(rows)==TRIALS*len(request['conditions']),'burst inventory differs');hz=native['clock']['frequency_hz'];previous=0
    for index,b in enumerate(rows):
        trial,slot=divmod(index,len(request['conditions']));ci=request['condition_orders'][trial][slot];c=request['conditions'][ci]
        require((b['trial'],b['slot'],b['condition'],b['bytes'],b['requested_idle_ms'],b['copies_count'])==(trial,slot,ci,SIZES[c['size_index']],IDLES[c['idle_index']],WINDOWS[c['window_index']]),'order differs')
        require(previous<=b['prime_begin_tick']<=b['prime_end_tick']<=b['idle_begin_tick']<=b['idle_end_tick']<=b['burst_begin_tick']<b['burst_end_tick'],'clock order differs');previous=b['burst_end_tick']
        require(len(b['copies'])==b['copies_count'],'copy inventory differs');last=b['burst_begin_tick']
        for i,v in enumerate(b['copies']):
            require(v['ordinal']==i and last<=v['begin_tick']<v['end_tick']<=b['burst_end_tick'],'COPY timing order differs');last=v['end_tick']
            require(abs((v['end_tick']-v['begin_tick'])*1e9/hz-v['wall_ns'])<=1,'COPY QPC duration differs')
        require(abs((b['burst_end_tick']-b['burst_begin_tick'])*1e9/hz-b['burst_wall_ns'])<=1,'window QPC duration differs')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=native['verification_native']['ports'];require(set(oracle)==set(ports),'port inventory differs')
    for key in oracle:require(np.allclose(np.asarray(oracle[key]),np.asarray(ports[key]),atol=5e-5,rtol=5e-5),'reference values differ')
    return dict(passed=True,bursts=len(rows),copies=sum(b['copies_count'] for b in rows),all_raw_samples_retained=True)

def collect(out,split):
    out=Path(out).resolve();m=prior.check(out);auth=read(out/'authorization.json')
    require(auth.get('collection_authorized') is True and auth.get('manifest_sha256')==sha(out/'manifest.json'),'root exact-manifest authorization required')
    if split=='heldout':
        b=read(out/'frozen-barrier.json');require(b['manifest_sha256']==sha(out/'manifest.json') and b['fit_sha256']==sha(out/'fit.json') and b['predictions_sha256']==sha(out/'heldout-predictions.json'),'freeze differs')
    require(not (out/f'{split}-started.json').exists(),'single acquisition, no retries');write(out/f'{split}-started.json',dict(at_utc=prior.v9.base.e2a.utc()))
    records=[]
    with prior.v9.base.run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request'])
            command=[m['binary']['path'],m['programs'][split]['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],status='failed',command=command);process=None;handles=[]
            try:
                env=prior.v9.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=prior.v9.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=480);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retry')
                record['verification']=validate(read(folder/'result.json'),job['request'],m['programs'][split],process.pid);record.update(status='completed',native=receipt(folder/'result.json'))
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
            records.append(record);print(json.dumps(dict(split=split,job_id=job['id'],status=record['status'],error=record.get('error'))),flush=True)
            if record['status']!='completed':break
    write(out/f'{split}-executions.json',records);return dict(split=split,declared=6,completed=sum(r['status']=='completed' for r in records),not_run=6-len(records))

def observations(out,split,memory):
    m=prior.check(out);executions=read(out/f'{split}-executions.json');require(len(executions)==6 and all(e['status']=='completed' for e in executions),'all independent fresh processes required')
    byid={r['job_id']:r for r in executions};records=[]
    for job in (j for j in m['jobs'] if j['split']==split and j['memory_mode']==memory):
        e=byid[job['id']];require(sha(e['native']['path'])==e['native']['sha256'],'native changed');native=read(e['native']['path']);validate(native,job['request'],m['programs'][split],e['process_id'])
        records.append(dict(job_id=job['id'],native=e['native'],bursts=native['bursts']))
    require(len(records)==3,'three fresh processes required');return m,records

def curve(knots,values,evidence,signed=False):
    return dict(byte_knots=list(knots),response_ns=list(values),evidence=list(evidence),signed=signed)

def fit_and_freeze(out):
    out=Path(out).resolve();require(not (out/'heldout-started.json').exists() and not (out/'fit.json').exists(),'single target-blind parameter fit required')
    models=[];predictions=[];diagnostics=[]
    for memory in ('registered','pageable'):
        m,records=observations(out,'profile',memory);evidence=[r['native']['path']+'#sha256='+r['native']['sha256'] for r in records];all_bursts=[b for r in records for b in r['bursts']]
        baseline=[];firsts={idle:[] for idle in IDLES}
        for size in SIZES:
            tails=[c['wall_ns'] for b in all_bursts if b['bytes']==size for c in b['copies'][1:]];require(tails,'sustained response labels required')
            baseline.append(statistics.mean(tails))
            for idle in IDLES:
                fs=[b['copies'][0]['wall_ns'] for b in all_bursts if b['bytes']==size and b['requested_idle_ms']==idle]
                require(len(fs)==216,'all first labels from three windows/fresh processes required');firsts[idle].append(statistics.mean(fs))
        profile=dict(schema='completed-transfer-lifecycle-1',context_sha256=m['scope']['context_sha256'],preparation_sha256=m['scope']['preparation_sha256'],route='dram->vram',memory_mode=memory,
            sustained=curve(SIZES,baseline,evidence),first_impulses=[dict(requested_idle_ns=i*1000000,response=curve(SIZES,[f-b for f,b in zip(firsts[i],baseline)],evidence,True)) for i in IDLES],
            maximum_copies=64,first_qualified=False,window_qualified=False,qualified_byte_coordinates=[],qualified_window_lengths=[],fit_uses_whole_LLM_target=False,future_target_observations_used=False,
            fitting_labels='completed individual COPY responses only; future N not a first-response input')
        model=CompletedTransferLifecycle.from_profile(profile);models.append(profile)
        for size,idle,n in itertools.product(SIZES,IDLES,WINDOWS):
            kw=dict(context_sha256=m['scope']['context_sha256'],preparation_sha256=m['scope']['preparation_sha256'],require_qualified=False)
            predictions.append(dict(memory_mode=memory,bytes=size,idle_ms=idle,copies=n,first_prediction_ns=model.first_response_ns(size,idle*1000000,**kw),window_prediction_ns=model.window_ns(size,idle*1000000,n,**kw)))
        diagnostics.append(dict(memory_mode=memory,sustained_ns=baseline,first_ns_by_idle={str(k):v for k,v in firsts.items()},first_labels_per_coordinate=216,
            response_parameters_empirical_not_unique_physical_cause=True))
    fit=dict(schema='h2d-impulse-fit-11',models=models,diagnostics=diagnostics,profile_only=True,heldout_read=False,whole_LLM_target_used=False,raw_samples_removed=0)
    write(out/'fit.json',fit);write(out/'heldout-predictions.json',dict(schema='h2d-impulse-predictions-11',rows=predictions))
    barrier=dict(schema='h2d-impulse-freeze-11',manifest_sha256=sha(out/'manifest.json'),fit_sha256=sha(out/'fit.json'),predictions_sha256=sha(out/'heldout-predictions.json'),heldout_observations_used=False)
    write(out/'frozen-barrier.json',barrier);return dict(frozen=receipt(out/'frozen-barrier.json'),models=2,prediction_cells=len(predictions),hardware_accuracy_qualified=False)

def evaluate(out):
    out=Path(out).resolve();m=prior.check(out);f=read(out/'frozen-barrier.json');require(f['manifest_sha256']==sha(out/'manifest.json') and f['fit_sha256']==sha(out/'fit.json') and f['predictions_sha256']==sha(out/'heldout-predictions.json'),'frozen parameters changed')
    predictions=read(out/'heldout-predictions.json')['rows'];first_rows=[];window_rows=[];process_rows=[]
    for memory in ('registered','pageable'):
        _,records=observations(out,'heldout',memory)
        all_bursts=[b for r in records for b in r['bursts']]
        for size,idle in itertools.product(SIZES,IDLES):
            fs=[b['copies'][0]['wall_ns'] for b in all_bursts if b['bytes']==size and b['requested_idle_ms']==idle];require(len(fs)==216,'complete first population required')
            p=next(p for p in predictions if (p['memory_mode'],p['bytes'],p['idle_ms'])==(memory,size,idle))
            mean=statistics.mean(fs);error=abs(p['first_prediction_ns']/mean-1)
            first_rows.append(dict(memory_mode=memory,bytes=size,idle_ms=idle,prediction_ns=p['first_prediction_ns'],measured_mean_ns=mean,APE=error,passed=error<=.10,raw_samples=216,raw_CV=statistics.pstdev(fs)/mean,future_N_is_input=False))
            for n in (16,64):
                bs=[b for b in all_bursts if (b['bytes'],b['requested_idle_ms'],b['copies_count'])==(size,idle,n)];require(len(bs)==72,'complete window population required')
                values=[b['burst_wall_ns'] for b in bs];mean=statistics.mean(values);p=next(p for p in predictions if (p['memory_mode'],p['bytes'],p['idle_ms'],p['copies'])==(memory,size,idle,n));error=abs(p['window_prediction_ns']/mean-1)
                window_rows.append(dict(memory_mode=memory,bytes=size,idle_ms=idle,copies=n,prediction_ns=p['window_prediction_ns'],measured_mean_ns=mean,APE=error,passed=error<=.10,raw_samples=72,raw_CV=statistics.pstdev(values)/mean,
                    measured_loop_gap_mean_ns=statistics.mean(b['burst_wall_ns']-sum(c['wall_ns'] for c in b['copies']) for b in bs)))
        for r in records:
            for size,idle in itertools.product(SIZES,IDLES):
                fs=[b['copies'][0]['wall_ns'] for b in r['bursts'] if b['bytes']==size and b['requested_idle_ms']==idle];p=next(p for p in predictions if (p['memory_mode'],p['bytes'],p['idle_ms'])==(memory,size,idle))
                process_rows.append(dict(job_id=r['job_id'],memory_mode=memory,bytes=size,idle_ms=idle,first_measured_mean_ns=statistics.mean(fs),first_APE=abs(p['first_prediction_ns']/statistics.mean(fs)-1),raw_samples=72))
    summaries=[];catalogs=[]
    for memory in ('registered','pageable'):
        fr=[r for r in first_rows if r['memory_mode']==memory];wr=[r for r in window_rows if r['memory_mode']==memory]
        fq,wq=all(r['passed'] for r in fr),all(r['passed'] for r in wr)
        summaries.append(dict(memory_mode=memory,first_MAPE=statistics.mean(r['APE'] for r in fr),first_worst_APE=max(r['APE'] for r in fr),first_passed=sum(r['passed'] for r in fr),first_conditions=len(fr),first_qualified=fq,
            window_MAPE=statistics.mean(r['APE'] for r in wr),window_worst_APE=max(r['APE'] for r in wr),window_passed=sum(r['passed'] for r in wr),window_conditions=len(wr),window_qualified=wq))
        profile=copy.deepcopy(next(p for p in read(out/'fit.json')['models'] if p['memory_mode']==memory));profile.update(first_qualified=fq,window_qualified=wq,qualified_byte_coordinates=list(SIZES),qualified_window_lengths=[16,64]);catalogs.append(profile)
    result=dict(schema='h2d-impulse-independent-evaluation-11',first_rows=first_rows,window_rows=window_rows,process_rows=process_rows,summaries=summaries,
        first_MAPE=statistics.mean(r['APE'] for r in first_rows),first_worst_APE=max(r['APE'] for r in first_rows),first_qualified=all(r['passed'] for r in first_rows),
        window_MAPE=statistics.mean(r['APE'] for r in window_rows),window_worst_APE=max(r['APE'] for r in window_rows),window_qualified=all(r['passed'] for r in window_rows),
        raw_CV_is_prediction_gate=False,raw_samples_removed=0,parameters_refitted=False,old_v10_results_unchanged=True,whole_LLM_accuracy_validated=False,scope=m['protocol']['scope'])
    write(out/'INDEPENDENT_EVALUATION.json',result);write(out/'validated-lifecycle-catalog.json',dict(schema='validated-completed-transfer-lifecycles-1',models=catalogs,qualification_evidence=receipt(out/'INDEPENDENT_EVALUATION.json')))
    return {k:result[k] for k in ('first_MAPE','first_worst_APE','first_qualified','window_MAPE','window_worst_APE','window_qualified')}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','profile','fit','heldout','evaluate'));p.add_argument('--out');p.add_argument('--binary');p.add_argument('--build-receipt');a=p.parse_args()
    value=protocol() if a.command=='protocol' else prepare(a.out,a.binary,a.build_receipt) if a.command=='prepare' else collect(a.out,'profile') if a.command=='profile' else fit_and_freeze(a.out) if a.command=='fit' else collect(a.out,'heldout') if a.command=='heldout' else evaluate(a.out)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
