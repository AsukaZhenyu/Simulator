"""Independent H2D lifecycle calibration with first and sustained windows.

Measurement/parameter estimation lives here. Response/state equations are reused
from modeling.transfer_recovery, without a held-out complete-runtime multiplier.
The old v9 result and its failed qualification flags remain unchanged.
"""
from __future__ import annotations
import argparse,copy,itertools,json,math,random,shutil,statistics,subprocess
from collections import defaultdict
from pathlib import Path

from . import h2d_bare_v9 as v9
from llm_infer_model.performance.reusable_action_costs import digest,semantic_action
from llm_infer_model.performance.transfer_response import TransferResponse
from llm_infer_model.performance.transfer_recovery import AffineResponseImpact,TransferRecoveryScope,SizeAwareRecoveryResponse

ROOT=v9.ROOT
read,write,sha,receipt,require=v9.read,v9.write,v9.sha,v9.receipt,v9.require
SIZES={'profile':(1024,131072,1048576,8388608),'heldout':(4096,65536,524288,2097152,4194304)}
IDLES={'profile':(0,1,5,20),'heldout':(2,10)}
WINDOWS={'profile':(1,16,128),'heldout':(8,64)}
KINDS=('static_affine','zero_initial_recovery','prepared_initial_recovery')
MIB=1048576
BOUNDARY=v9.BOUNDARY

def protocol():
    return dict(schema='h2d-lifecycle-protocol-10',profile_processes=6,heldout_processes=6,
        memory_modes=['registered','pageable'],fresh_processes_per_mode_per_split=3,trials=12,
        sizes_bytes=SIZES,idle_ms=IDLES,windows=WINDOWS,boundary=BOUNDARY,
        target='original Executor.action COPY on separately owned eight-MiB arena',
        preparation='complete original Dense graph, then prescribed idle before every COPY window',
        original_preparation_arena='unchanged pageable; explicit COPY arena registration only',
        calibration='first completed-COPY mean and summed completed-COPY response mean, per fresh process/condition',
        fitting_uses_outer_loop_gaps=False,acceptance_includes_outer_loop_gaps=True,
        providers=list(KINDS),model_equations='modeling.transfer_recovery.SizeAwareRecoveryResponse',
        selection='leave-one-profile-process-out mean relative error; prefer lower complexity within 1 percentage point',
        acceptance=dict(primary_first_COPY_condition_mean_APE_max=.10,primary_complete_window_condition_mean_APE_max=.10,
                        condition_mean='all 36 retained trials over three fresh processes at the same memory/bytes/idle/window coordinate',
                        first_and_window_qualification_reported_separately=True,raw_CV_is_prediction_gate=False,
                        per_process_condition_APE_reported_as_additional_diagnostic=True),
        independent_holdout=dict(new_bytes=True,new_idle=True,new_window_lengths=True,new_mathematical_seed=True),
        predictions_frozen_before_collection=True,all_samples_retained=True,slow_trials_removed=0,
        no_retries=True,no_LLM_target_fitting=True,no_physical_OS_clock_queue_cause_claim=True,
        old_v9_failures_preserved=True,scope='declared 1KiB–8MiB preparation/host-buffer/runtime contract, not arbitrary transfer paths')

def conditions(split):
    return [dict(size_index=s,idle_index=i,window_index=n) for s,i,n in itertools.product(range(len(SIZES[split])),range(len(IDLES[split])),range(len(WINDOWS[split])))]

def plans(programs,source_sha):
    jobs=[]
    for split in SIZES:
        cs=conditions(split)
        for round_id in range(3):
            for memory in (('registered','pageable') if round_id%2==0 else ('pageable','registered')):
                rng=random.Random(20261028+round_id*101+(0 if split=='profile' else 10000))
                orders=[]
                for _ in range(12):
                    order=list(range(len(cs)));rng.shuffle(order);orders.append(order)
                sampling=v9.base.e2a.request('action',trials=12,target=programs[split]['target_action_index']);sampling['repeats']=128
                request=dict(schema='dynamic-h2d-lifecycle-request-10',source_sha256=source_sha,split=split,sampling=sampling,
                    memory_mode=memory,expected_process_mask=4294967295,sizes_bytes=list(SIZES[split]),idle_ms=list(IDLES[split]),windows=list(WINDOWS[split]),
                    conditions=cs,condition_orders=orders,performance_boundary='completed_original_Executor.action_COPY',telemetry_between_bursts=False)
                jobs.append(dict(id=f'{split}-{memory}-r{round_id}',split=split,memory_mode=memory,round=round_id,request=request))
    return jobs

def prepare(out,binary,build_receipt):
    out,binary,build_receipt=Path(out).resolve(),Path(binary).resolve(),Path(build_receipt).resolve()
    require(not out.exists() and binary.is_file(),'exclusive new output and compiled binary required')
    build=read(build_receipt);source=Path(__file__).parent/'native/v10/h2d_lifecycle.cpp'
    require(build['binary']['sha256']==sha(binary) and build['source_sha256']==sha(source),'build receipt does not bind source/binary')
    parent_path=ROOT/'build/dynamic-validation-20261026/idle-v2b/manifest.json';parent=read(parent_path)
    programs=copy.deepcopy(parent['programs']);require(programs['profile']['program_sha256']!=programs['heldout']['program_sha256'],'independent mathematical input required')
    for p in programs.values():
        for key in ('program','reference'):require(sha(p[key]['path'])==p[key]['sha256'],'retained program/reference changed')
    sources=[Path(__file__),source,source.parent/'CMakeLists.txt',Path(v9.__file__),source.parent.parent/'v9/h2d_bare.cpp',
        source.parent.parent/'v3/common.hpp',ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',
        ROOT/'profiling/native/highlevel_accuracy/highlevel_probe.cpp',ROOT/'modeling/llm_infer_model/performance/transfer_recovery.py',
        ROOT/'modeling/llm_infer_model/performance/idle_recovery.py',ROOT/'modeling/llm_infer_model/performance/transfer_response.py',
        ROOT/'modeling/llm_infer_model/performance/prepared_transfer.py',ROOT/'modeling/llm_infer_model/performance/parameters.py']
    sources+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    require(all(p.is_file() for p in sources),'complete source inventory required')
    out.mkdir(parents=True);archived=[]
    for p in sources:
        target=out/'executed_sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        archived.append(dict(source=receipt(p),archived=receipt(target)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    context=dict(runtime=parent['runtime'],binary=receipt(target),hardware=parent['hardware'],boundary=BOUNDARY,
                 process_mask=4294967295,main_thread_mask=4,threads=1,arena_bytes=8388608)
    profile_program=read(programs['profile']['program']['path'])
    prep=dict(actions=[semantic_action(profile_program,i) for i in range(len(profile_program['actions']))],
              preceding_lifecycle='reset, complete original graph, explicit idle',arena='separate eight-MiB source from repeated independent input bytes; target owned on GPU',
              source_buffer_mode='explicit registered/pageable',original_graph_arena='unchanged pageable',graphs=False,observers=False)
    other=read(programs['heldout']['program']['path']);require(prep['actions']==[semantic_action(other,i) for i in range(len(other['actions']))],'preparation semantics differ')
    m=dict(schema='h2d-lifecycle-manifest-10',prepared_at_utc=v9.base.e2a.utc(),parent_manifest=receipt(parent_path),programs=programs,
           binary=receipt(target),build_receipt=receipt(build_receipt),runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],hardware=parent['hardware'],
           sources=archived,scope=dict(context_sha256=digest(context),preparation_sha256=digest(prep),context=context,preparation=prep),
           jobs=plans(programs,sha(source)),protocol=protocol())
    write(out/'manifest.json',m)
    return dict(manifest=receipt(out/'manifest.json'),hardware_executed=False,profile=6,heldout=6)

def check(out):
    m=read(out/'manifest.json')
    for r in [m['binary'],m['build_receipt'],*m['runtime']]:require(sha(r['path'])==r['sha256'],'runtime/build changed')
    for s in m['sources']:
        for key in ('source','archived'):require(sha(s[key]['path'])==s[key]['sha256'],'frozen source changed')
    for p in m['programs'].values():
        for key in ('program','reference'):require(sha(p[key]['path'])==p[key]['sha256'],'program/reference changed')
    return m

def validate(native,request,binding,pid):
    require(native['schema']=='dynamic-h2d-lifecycle-native-10' and native['request']==request and native['process_id']==pid,'request/process differs')
    require(native['source_sha256']==request['source_sha256'] and native['program_sha256']==binding['program_sha256'],'source/input differs')
    require(all(native[k] is False for k in ('CUPTI_initialized','CUDA_events_created','NVML_initialized','telemetry_between_bursts','performance_parameters_fitted')),'observer or fitting broadened')
    memory='registered_host' if request['memory_mode']=='registered' else 'unregistered_pageable'
    require(native['source_pointer_kind'][memory] and native['arena_registration']['during'][memory] and native['arena_registration']['restored'],'memory category differs')
    require(native['source_buffer_host'] and not native['target_buffer_host'],'host/device boundary differs')
    require(native['process_affinity']['observed_mask']==native['process_affinity']['final_mask']==4294967295 and native['main_thread_affinity']['observed_mask']==native['main_thread_affinity']['final_mask']==4,'affinity differs')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'] and native['full_payload']['full_byte_equality'],'numerical/full payload failed')
    require(native['full_payload']['bytes']==max(request['sizes_bytes']) and native['full_payload']['arena_bytes']==8388608,'payload range differs')
    oracles=native['slice_copy_oracles'];require({x['bytes'] for x in oracles}==set(request['sizes_bytes']) and all(x['selected_range_initially_different'] and x['selected_bytes_equal_after_copy'] and x['outside_view_unchanged'] for x in oracles),'range oracle failed')
    rows=native['bursts'];require(len(rows)==12*len(request['conditions']),'burst inventory differs');hz=native['clock']['frequency_hz'];previous=0
    for index,b in enumerate(rows):
        trial,slot=divmod(index,len(request['conditions']));ci=request['condition_orders'][trial][slot];c=request['conditions'][ci]
        require((b['trial'],b['slot'],b['condition'],b['bytes'],b['requested_idle_ms'],b['copies_count'])==(trial,slot,ci,request['sizes_bytes'][c['size_index']],request['idle_ms'][c['idle_index']],request['windows'][c['window_index']]),'declared order differs')
        require(previous<=b['prime_begin_tick']<=b['prime_end_tick']<=b['idle_begin_tick']<=b['idle_end_tick']<=b['burst_begin_tick']<b['burst_end_tick'],'invalid clocks');previous=b['burst_end_tick']
        require(len(b['copies'])==b['copies_count'],'copy count differs');last=b['burst_begin_tick']
        for i,v in enumerate(b['copies']):
            require(v['ordinal']==i and last<=v['begin_tick']<v['end_tick']<=b['burst_end_tick'],'copy clocks differ');last=v['end_tick']
            require(abs((v['end_tick']-v['begin_tick'])*1e9/hz-v['wall_ns'])<=1,'copy QPC duration differs')
        require(abs((b['burst_end_tick']-b['burst_begin_tick'])*1e9/hz-b['burst_wall_ns'])<=1,'burst QPC differs')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=native['verification_native']['ports'];require(set(oracle)==set(ports),'port inventory differs')
    for key in oracle:require(np.allclose(np.asarray(oracle[key]),np.asarray(ports[key]),atol=5e-5,rtol=5e-5),'reference values differ')
    return dict(passed=True,bursts=len(rows),copies=sum(x['copies_count'] for x in rows),all_raw_samples_retained=True)

def collect(out,split):
    out=Path(out).resolve();m=check(out);auth=read(out/'authorization.json')
    require(auth.get('collection_authorized') is True and auth.get('manifest_sha256')==sha(out/'manifest.json'),'root authorization required')
    if split=='heldout':
        f=read(out/'frozen-barrier.json');require(f['manifest_sha256']==sha(out/'manifest.json') and f['predictions_sha256']==sha(out/'heldout-predictions.json') and f['fit_sha256']==sha(out/'fit.json'),'freeze differs')
    require(not (out/f'{split}-started.json').exists(),'no repeated acquisition or retries');write(out/f'{split}-started.json',dict(at_utc=v9.base.e2a.utc()))
    records=[]
    with v9.base.run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False);write(folder/'request.json',job['request'])
            command=[m['binary']['path'],m['programs'][split]['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(job_id=job['id'],status='failed',command=command);process=None;handles=[]
            try:
                env=v9.base.run.child_environment(m['runtime_dirs']);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process,a,b,affinity=v9.base.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=300);record.update(returncode=code,process_exited_verified=True);require(code==0,'native failed; no retries')
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
    write(out/f'{split}-executions.json',records)
    return dict(split=split,declared=6,completed=sum(r['status']=='completed' for r in records),not_run=6-len(records))

def observations(out,split,memory):
    m=check(out);executions=read(out/f'{split}-executions.json');require(len(executions)==6 and all(r['status']=='completed' for r in executions),'all fresh processes required')
    byid={e['job_id']:e for e in executions};records=[]
    for job in (j for j in m['jobs'] if j['split']==split and j['memory_mode']==memory):
        e=byid[job['id']];require(sha(e['native']['path'])==e['native']['sha256'],'native changed');n=read(e['native']['path']);validate(n,job['request'],m['programs'][split],e['process_id'])
        cells=[]
        for i,c in enumerate(job['request']['conditions']):
            bs=[b for b in n['bursts'] if b['condition']==i];require(len(bs)==12,'condition trials missing')
            first=[b['copies'][0]['wall_ns'] for b in bs];summed=[sum(v['wall_ns'] for v in b['copies']) for b in bs];wall=[b['burst_wall_ns'] for b in bs]
            cells.append(dict(bytes=SIZES[split][c['size_index']],idle_ms=IDLES[split][c['idle_index']],copies=WINDOWS[split][c['window_index']],first_mean_ns=statistics.mean(first),copy_sum_mean_ns=statistics.mean(summed),window_mean_ns=statistics.mean(wall),
                first_CV=statistics.pstdev(first)/statistics.mean(first),window_CV=statistics.pstdev(wall)/statistics.mean(wall),raw_trials=12))
        records.append(dict(job_id=job['id'],native=e['native'],cells=cells))
    require(len(records)==3,'three process contexts required')
    return m,records

def model(kind,theta,scope,memory,evidence,qualified=False):
    require(kind in KINDS,'unknown model')
    base=TransferResponse('dram->vram',float(theta[0])*1000,float(theta[1])*1000/MIB,float(theta[1])*1000/MIB,None,min(SIZES['profile']),max(SIZES['profile']),memory,BOUNDARY,tuple(evidence),False)
    if kind=='static_affine':return base
    if kind=='zero_initial_recovery':initial=(0.,0.);capacity=theta[2:4];ti,tr=theta[4:6]
    else:initial=theta[2:4];capacity=(theta[2]+theta[4],theta[3]+theta[5]);ti,tr=theta[6:8]
    return SizeAwareRecoveryResponse(TransferRecoveryScope(scope['context_sha256'],scope['preparation_sha256'],'dram->vram',memory,20000000),base,
        AffineResponseImpact(float(capacity[0])*1000,float(capacity[1])*1000/MIB,tuple(evidence)),AffineResponseImpact(float(initial[0])*1000,float(initial[1])*1000/MIB,tuple(evidence)),float(ti)*1e6,float(tr)*1000,tuple(evidence),qualified)

def predict(kind,theta,scope,memory,evidence,cell,*,integer_clock=False):
    p=model(kind,theta,scope,memory,evidence)
    if kind=='static_affine':
        v=p.predict_ns(cell['bytes'],require_qualified=False)
        if integer_clock:v=math.ceil(v)
        return (v,v*cell['copies'])
    if integer_clock:
        r=p.predict_burst(cell['bytes'],cell['idle_ms']*1000000,cell['copies'],context_sha256=scope['context_sha256'],preparation_sha256=scope['preparation_sha256'],require_qualified=False)
        return (r['rows'][0]['predicted_ns'],r['duration_ns'])
    vs=p.continuous_burst(cell['bytes'],cell['idle_ms']*1000000,cell['copies'],context_sha256=scope['context_sha256'],preparation_sha256=scope['preparation_sha256'])
    return (float(vs[0]),float(sum(vs)))

def fit_model(kind,scope,memory,evidence,records):
    import numpy as np
    from scipy.optimize import least_squares,nnls
    cells=records[0]['cells'];target=np.asarray([[[c['first_mean_ns'],c['copy_sum_mean_ns']] for c in r['cells']] for r in records])/1000
    if kind=='static_affine':
        design=[];labels=[]
        for r in records:
            for c in r['cells']:
                design.extend([[1,c['bytes']/MIB],[1,c['bytes']/MIB]])
                labels.extend([c['first_mean_ns']/1000,c['copy_sum_mean_ns']/c['copies']/1000])
        theta,resid=nnls(np.asarray(design),np.asarray(labels));return dict(theta=theta.tolist(),converged=True,free_parameters=2,rank=2,identifiable=True,optimizer='nonnegative affine fit')
    tail=[];design=[];first0=[];first20=[]
    for size in SIZES['profile']:
        selected=[c for r in records for c in r['cells'] if c['bytes']==size and c['idle_ms']==0 and c['copies']==128]
        tail.append(statistics.mean((c['copy_sum_mean_ns']-c['first_mean_ns'])/127/1000 for c in selected));design.append([1,size/MIB])
        first0.append(statistics.mean(c['first_mean_ns']/1000 for r in records for c in r['cells'] if c['bytes']==size and c['idle_ms']==0))
        first20.append(statistics.mean(c['first_mean_ns']/1000 for r in records for c in r['cells'] if c['bytes']==size and c['idle_ms']==20))
    design=np.asarray(design);base=np.maximum(nnls(design,np.asarray(tail))[0],.001)
    initial=np.maximum(nnls(design,np.maximum(np.asarray(first0)-design@base,.01))[0],.001)
    cap=np.maximum(nnls(design,np.maximum(np.asarray(first20)-design@base,.01))[0],.001)
    ceiling=max(10000.,float(np.max(target))*4)
    if kind=='zero_initial_recovery':
        low=[.001,0.,.001,0.,.001,.001];high=[ceiling]*4+[100.,10000.]
        starts=[[base[0],base[1],cap[0],cap[1],ti,tr] for ti in (.5,5.) for tr in (50.,500.)]
    else:
        rem=np.maximum(cap-initial,.001);low=[.001,0.,0.,0.,.001,0.,.001,.001];high=[ceiling]*6+[100.,10000.]
        starts=[[base[0],base[1],initial[0],initial[1],rem[0],rem[1],ti,tr] for ti in (.5,5.) for tr in (50.,500.)]
    scale=np.maximum(np.mean(target,axis=0),.001)
    def residual(theta):
        pred=np.asarray([predict(kind,theta,scope,memory,evidence,c) for c in cells])/1000
        return ((pred[None,...]-target)/scale[None,...]).ravel()
    fits=[least_squares(residual,np.clip(s,np.asarray(low)+1e-8,np.asarray(high)-1e-8),bounds=(low,high),max_nfev=250,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7) for s in starts]
    best=min(fits,key=lambda f:float(f.fun@f.fun));norm=np.linalg.norm(best.jac,axis=0);scaled=best.jac/np.maximum(norm,1e-300);rank=int(np.linalg.matrix_rank(scaled))
    return dict(theta=best.x.tolist(),converged=bool(best.success),free_parameters=len(best.x),rank=rank,identifiable=bool(best.success and rank==len(best.x)),
                optimizer='nonnegative relative least squares of first and summed completed response means',score=float(best.fun@best.fun),message=str(best.message),evaluations=sum(f.nfev for f in fits))

def score(fit,kind,scope,memory,evidence,records):
    errors=[]
    for r in records:
        for c in r['cells']:
            p=predict(kind,fit['theta'],scope,memory,evidence,c)
            errors.extend([abs(p[0]/c['first_mean_ns']-1),abs(p[1]/c['copy_sum_mean_ns']-1)])
    return statistics.mean(errors)

def fit_and_freeze(out):
    out=Path(out).resolve();require(not (out/'heldout-started.json').exists() and not (out/'fit.json').exists(),'single target-blind fit required')
    fits=[];predictions=[]
    for memory in ('registered','pageable'):
        m,records=observations(out,'profile',memory);scope=m['scope'];evidence=tuple(r['native']['path']+'#sha256='+r['native']['sha256'] for r in records);rows=[]
        for kind in KINDS:
            validation=[]
            for k in range(3):
                fit=fit_model(kind,scope,memory,evidence,[r for i,r in enumerate(records) if i!=k]);validation.append(score(fit,kind,scope,memory,evidence,[records[k]]))
            fit=fit_model(kind,scope,memory,evidence,records);fit.update(kind=kind,memory_mode=memory,evidence=evidence,profile_MAPE=score(fit,kind,scope,memory,evidence,records),cross_validation_MAPEs=validation,cross_validation_MAPE=statistics.mean(validation))
            rows.append(fit)
        best=min(r['cross_validation_MAPE'] for r in rows if r['converged'])
        selected=min((r for r in rows if r['converged'] and r['cross_validation_MAPE']<=best+.01),key=lambda r:r['free_parameters'])
        for r in rows:r['selected']=r is selected
        fits.extend(rows)
        for c in conditions('heldout'):
            cell=dict(bytes=SIZES['heldout'][c['size_index']],idle_ms=IDLES['heldout'][c['idle_index']],copies=WINDOWS['heldout'][c['window_index']])
            for r in rows:
                first,total=predict(r['kind'],r['theta'],scope,memory,evidence,cell,integer_clock=True)
                predictions.append(dict(memory_mode=memory,kind=r['kind'],selected=r['selected'],**cell,first_prediction_ns=first,window_prediction_ns=total))
    fit=dict(schema='h2d-lifecycle-fit-10',models=fits,profile_only=True,heldout_read=False,complete_LLM_target_used=False,raw_trials_removed=0,
             parameter_interpretation='empirical completed-response state; no unique physical clock, queue or OS cause identified',
             identifiable_flag_is_local_fit_diagnostic=True)
    write(out/'fit.json',fit);write(out/'heldout-predictions.json',dict(schema='h2d-lifecycle-predictions-10',rows=predictions))
    barrier=dict(schema='h2d-lifecycle-freeze-10',manifest_sha256=sha(out/'manifest.json'),fit_sha256=sha(out/'fit.json'),predictions_sha256=sha(out/'heldout-predictions.json'),heldout_observations_used=False)
    write(out/'frozen-barrier.json',barrier)
    return dict(selected=[dict(memory_mode=r['memory_mode'],kind=r['kind'],cross_validation_MAPE=r['cross_validation_MAPE']) for r in fits if r['selected']],frozen=receipt(out/'frozen-barrier.json'))

def evaluate(out):
    out=Path(out).resolve();m=check(out);f=read(out/'frozen-barrier.json');require(f['manifest_sha256']==sha(out/'manifest.json') and f['fit_sha256']==sha(out/'fit.json') and f['predictions_sha256']==sha(out/'heldout-predictions.json'),'frozen estimates changed')
    preds=read(out/'heldout-predictions.json')['rows'];rows=[]
    for memory in ('registered','pageable'):
        _,records=observations(out,'heldout',memory)
        for r in records:
            for c in r['cells']:
                for p in (x for x in preds if x['memory_mode']==memory and (x['bytes'],x['idle_ms'],x['copies'])==(c['bytes'],c['idle_ms'],c['copies'])):
                    ef=abs(p['first_prediction_ns']/c['first_mean_ns']-1);et=abs(p['window_prediction_ns']/c['window_mean_ns']-1)
                    rows.append(dict(job_id=r['job_id'],memory_mode=memory,kind=p['kind'],selected=p['selected'],**c,first_predicted_ns=p['first_prediction_ns'],window_predicted_ns=p['window_prediction_ns'],first_APE=ef,window_APE=et,
                         window_mean_accuracy_passed=et<=.10,first_mean_accuracy_passed=ef<=.10,raw_CV_is_prediction_gate=False))
    summaries=[]
    for memory in ('registered','pageable'):
        for kind in KINDS:
            rs=[r for r in rows if r['memory_mode']==memory and r['kind']==kind]
            summaries.append(dict(memory_mode=memory,kind=kind,selected=rs[0]['selected'],cells=len(rs),first_MAPE=statistics.mean(r['first_APE'] for r in rs),first_worst_APE=max(r['first_APE'] for r in rs),window_MAPE=statistics.mean(r['window_APE'] for r in rs),window_worst_APE=max(r['window_APE'] for r in rs),window_passed=sum(r['window_mean_accuracy_passed'] for r in rs),first_passed=sum(r['first_mean_accuracy_passed'] for r in rs)))
    condition_rows=[]
    for memory in ('registered','pageable'):
        for kind in KINDS:
            groups=defaultdict(list)
            for r in rows:
                if r['memory_mode']==memory and r['kind']==kind:groups[r['bytes'],r['idle_ms'],r['copies']].append(r)
            for (size,idle,copies),rs in groups.items():
                require(len(rs)==3,'all independent fresh processes retained per condition')
                mf=statistics.mean(r['first_mean_ns'] for r in rs);mt=statistics.mean(r['window_mean_ns'] for r in rs)
                pf,pt=rs[0]['first_predicted_ns'],rs[0]['window_predicted_ns'];ef,et=abs(pf/mf-1),abs(pt/mt-1)
                condition_rows.append(dict(memory_mode=memory,kind=kind,selected=rs[0]['selected'],bytes=size,idle_ms=idle,copies=copies,
                    retained_fresh_processes=3,retained_trials=36,first_measured_mean_ns=mf,window_measured_mean_ns=mt,first_prediction_ns=pf,window_prediction_ns=pt,
                    first_APE=ef,window_APE=et,first_mean_accuracy_passed=ef<=.10,window_mean_accuracy_passed=et<=.10))
    selected=[r for r in rows if r['selected']];selected_conditions=[r for r in condition_rows if r['selected']]
    result=dict(schema='h2d-lifecycle-independent-evaluation-10',rows=rows,summaries=summaries,selected_cells=len(selected),selected_window_MAPE=statistics.mean(r['window_APE'] for r in selected),
                selected_window_worst_APE=max(r['window_APE'] for r in selected),selected_first_MAPE=statistics.mean(r['first_APE'] for r in selected),
                selected_all_window_conditions_passed=all(r['window_mean_accuracy_passed'] for r in selected),raw_CV_is_prediction_gate=False,
                no_refit=True,slow_trials_removed=0,old_v9_failed_flags_unchanged=True,whole_LLM_accuracy_validated=False,
                response_scope=m['protocol']['scope'],parameter_scope=m['scope'],predictions=receipt(out/'heldout-predictions.json'),condition_rows=condition_rows,
                primary_conditions=len(selected_conditions),primary_first_MAPE=statistics.mean(r['first_APE'] for r in selected_conditions),
                primary_first_worst_APE=max(r['first_APE'] for r in selected_conditions),primary_window_MAPE=statistics.mean(r['window_APE'] for r in selected_conditions),
                primary_window_worst_APE=max(r['window_APE'] for r in selected_conditions),
                first_COPY_qualified=all(r['first_mean_accuracy_passed'] for r in selected_conditions),window_qualified=all(r['window_mean_accuracy_passed'] for r in selected_conditions))
    write(out/'INDEPENDENT_EVALUATION.json',result)
    return {k:result[k] for k in ('primary_conditions','primary_first_MAPE','primary_first_worst_APE','primary_window_MAPE','primary_window_worst_APE','first_COPY_qualified','window_qualified')}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('protocol','prepare','profile','fit','heldout','evaluate'));p.add_argument('--out');p.add_argument('--binary');p.add_argument('--build-receipt');a=p.parse_args()
    v=protocol() if a.command=='protocol' else prepare(a.out,a.binary,a.build_receipt) if a.command=='prepare' else collect(a.out,'profile') if a.command=='profile' else fit_and_freeze(a.out) if a.command=='fit' else collect(a.out,'heldout') if a.command=='heldout' else evaluate(a.out)
    print(json.dumps(v,ensure_ascii=False))

if __name__=='__main__':main()
