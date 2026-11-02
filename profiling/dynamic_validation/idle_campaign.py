"""Balanced idle/recovery microbenchmarks; no heldout collection before freeze.

The actual native H2D work is retained from the existing Dense arena. Timing
labels are inclusive Executor.action responses, not physical PCIe bandwidth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from profiling.dac_validation import e2a
from profiling.state_calibration import run

ROOT = Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=e2a.read,e2a.write,e2a.sha,e2a.receipt,e2a.require
IDLES={'profile':(0,1,5,20),'heldout':(0,2,10,20)}


def orders(round_id, trials=16):
    """Each four-condition permutation has balanced positions in a process."""
    result=[]
    for trial in range(trials):
        row=[(trial+round_id+offset)%4 for offset in range(4)]
        if round_id%2:row.reverse()
        result.append(row)
    return result


def validate_protocol(request):
    sampling=request['sampling']
    require(sampling['trials']==16 and sampling['repeats']==16,'exact sixteen trial/burst contract required')
    idle=request['idle_ms'];sequence=request['orders']
    require(len(idle)==4 and len(set(idle))==4 and 0 in idle and
            all(type(v) is int and 0<=v<=1000 for v in idle),'four distinct nonnegative idle controls required')
    require(len(sequence)==16 and all(len(row)==4 and set(row)==set(range(4)) for row in sequence),
            'every trial requires a condition permutation')
    require(all(Counter(row[slot] for row in sequence)==Counter({i:4 for i in range(4)}) for slot in range(4)),
            'condition positions are not balanced within the process')


def prepare(out,binary,*,smoke=False):
    out,binary=Path(out).resolve(),Path(binary).resolve()
    require(not out.exists(),'new campaign directory required')
    parent_path=ROOT/'build/h2d-history-20261026/campaign-v1/manifest.json'
    parent=read(parent_path)
    source=ROOT/'profiling/dynamic_validation/native/idle_probe.cpp'
    programs={}
    for split in ('profile','heldout'):
        key=f'l2-h512-p64-s2-{split}'
        binding=parent['programs'][key]
        require(all(sha(binding[k]['path'])==binding[k]['sha256'] for k in ('program','reference')),'retained input changed')
        program=read(binding['program']['path'])
        target=next(i for i,a in enumerate(program['actions']) if a['kind']=='COPY' and a['src']=='dram'
                    and a['dst']=='vram' and a['target_id'].endswith(':W_up'))
        tensor=next(t for t in program['graph']['workload']['tensors'] if t['id']==program['actions'][target]['target_id'])
        require(tensor['storage_bytes']==2097152,'actual two-MiB target required')
        programs[split]=dict(**binding,target_action_index=target,target_action=program['actions'][target],
                             program_sha256=program['program_sha256'])
    out.mkdir(parents=True)
    paths=[Path(__file__),source,source.parent/'CMakeLists.txt',ROOT/'profiling/dac_validation/e2a.py',
           ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/native/steady_costs/probe.cpp',
           ROOT/'profiling/native/steady_costs/components.hpp']
    paths+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'))
    sources=[]
    for path in paths:
        target=out/'executed_sources'/path.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        sources.append(dict(source=receipt(path),archived=receipt(target)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    jobs=[]
    for split in (('profile',) if smoke else ('profile','heldout')):
        for round_id in range(1 if smoke else 3):
            memories=('pageable','registered') if round_id%2==0 else ('registered','pageable')
            if smoke:memories=('pageable',)
            for memory in memories:
                sampling=e2a.request('action',trials=16,target=programs[split]['target_action_index'])
                sampling['repeats']=16
                req=dict(schema='dynamic-idle-request-1',source_sha256=sha(source),sampling=sampling,
                    expected_process_mask=4294967295,memory_mode=memory,
                    idle_ms=list(IDLES[split]),orders=orders(round_id))
                validate_protocol(req)
                jobs.append(dict(id=f'{split}-{memory}-r{round_id}',split=split,round=round_id,
                                 memory_mode=memory,program_key=split,request=req))
    manifest=dict(schema='dynamic-idle-campaign-1',prepared_at_utc=e2a.utc(),parent_manifest=receipt(parent_path),
        binary=receipt(target),sources=sources,runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],
        hardware=parent['baseline_context']['hardware'],programs=programs,jobs=jobs,smoke_only=smoke,
        protocol=dict(counts={s:sum(j['split']==s for j in jobs) for s in ('profile','heldout')},
            retries=0,timeout_seconds=300,payload_bytes=2097152,trials=16,burst_copies=16,
            balanced_within_process=True,independent_heldout_seed=True,cupti_enabled=False,
            initial_state='one full untimed Dense prime before every idle/burst condition',
            primary_input='requested idle and frozen initialization contract; target future telemetry excluded',
            actual_idle='observed completed interval; retained separately as protocol adherence/conditional diagnosis',
            target='inclusive completed unchanged Executor.action response; no pure payload-rate identification',
            quality=dict(max_case_mean_ape=.10,max_block_ape_p90=.20,raw_cv_is_prediction_gate=False),
            physical_cause_identified=False))
    write(out/'manifest.json',manifest)
    return dict(out=str(out),counts=manifest['protocol']['counts'],smoke_only=smoke)


def check(manifest):
    for row in manifest['sources']:
        for field in ('source','archived'):
            require(sha(row[field]['path'])==row[field]['sha256'],'bound source changed')
    for row in [manifest['binary'],*manifest['runtime']]:require(sha(row['path'])==row['sha256'],'binary/runtime changed')


def verify(native,request,binding,pid):
    validate_protocol(request)
    require(native['schema']=='dynamic-idle-native-1' and native['request']==request,'native request changed')
    require(native['process_id']==pid and native['source_sha256']==request['source_sha256'] and native['caller_source_sha256_match'],
            'native PID/source changed')
    require(native['program_sha256']==binding['program_sha256'] and native['target_action']==binding['target_action']
            and native['target_action_index']==binding['target_action_index'] and native['target_bytes']==2097152,'native mathematical target differs')
    require(native['main_thread_affinity']['requested_mask']==native['main_thread_affinity']['observed_mask']==
            native['main_thread_affinity']['final_mask']==4 and native['main_thread_affinity']['restored'],'main affinity differs')
    require(native['process_affinity']['requested_mask']==native['process_affinity']['observed_mask']==
            native['process_affinity']['final_mask']==4294967295 and native['process_affinity']['applied_before_backend_initialization'],
            'process affinity differs')
    mode=request['memory_mode'];registration=native['arena_registration']
    require(registration['mode']==mode and registration['during']['registered_host']==(mode=='registered')
            and registration['restored'] and registration['after']['unregistered_pageable'],'actual memory/restore differs')
    require(native['source_pointer_during_measurement']['registered_host']==(mode=='registered'),'target source category differs')
    for p in (native['payload_before'],native['payload_after']):
        require(p['bytes']==2097152 and p['full_byte_equality'] and p['source_sha256']==p['target_sha256'],'full H2D byte check failed')
    require(native['payload_before']['source_sha256']==native['payload_after']['source_sha256'],'source payload mutated')
    require(native['native_repeat_consistency']['passed'] and native['verification_native']['copy_values_equal'],'native replay/copy oracle failed')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=native['verification_native']['ports']
    require(set(ports)==set(oracle),'numerical port inventory differs')
    for key,values in oracle.items():
        a,b=np.asarray(ports[key]),np.asarray(values)
        require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=5e-5,atol=5e-5),'independent port mismatch '+key)
    require(native['instrumentation']['cupti'] is False and native['instrumentation']['logging_inside_timer'] is False,
            'primary timing unexpectedly instrumented')
    hz=native['clock']['frequency_hz'];require(hz>0,'invalid QPC frequency')
    rows=native['sequences'];require(len(rows)==request['sampling']['trials']*4,'condition count differs')
    last=0;ledger=0
    for number,row in enumerate(rows):
        trial,slot=divmod(number,4);index=request['orders'][trial][slot]
        require((row['trial'],row['condition_slot'],row['condition_index'],row['requested_idle_ms'])==
                (trial,slot,index,request['idle_ms'][index]),'balanced condition order differs')
        ib,ie,bb,be=(row[k] for k in ('idle_begin_tick','idle_end_tick','burst_begin_tick','burst_end_tick'))
        require(last<=ib<=ie<=bb<be,'condition clocks overlap or move backwards')
        require(abs((ie-ib)*1e9/hz-row['actual_idle_ns'])<=1 and
                abs((be-bb)*1e9/hz-row['burst_wall_ns'])<=1,'QPC condition ledger differs')
        samples=row['copies'];require(len(samples)==16,'burst count differs')
        previous=ie;total=0
        for ordinal,item in enumerate(samples):
            first,finish=item['begin_tick'],item['end_tick'];require(item['ordinal']==ordinal and previous<=first<finish<=be,'copy clock/order differs')
            require(abs((finish-first)*1e9/hz-item['wall_ns'])<=1 and
                    abs((first-previous)*1e9/hz-item['gap_since_previous_completion_ns'])<=1,'copy QPC ledger differs')
            total+=item['wall_ns'];previous=finish;ledger+=1
        require(total<=row['burst_wall_ns']+16,'inclusive copy sum exceeds burst envelope')
        last=be
    return dict(passed=True,copy_intervals=ledger,numerical_oracle=True,full_two_MiB_bytes=True,
                requested_idle_is_not_actual_idle=True,cupti_primary=False)


def collect(out,split):
    out=Path(out).resolve();manifest=read(out/'manifest.json');check(manifest)
    require(split in IDLES,'unknown split')
    barrier=None
    if split=='heldout':
        barrier=read(out/'frozen-barrier.json')
        require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen manifest changed')
        for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen predictor changed')
    marker=out/(split+'-started.json')
    if not marker.exists():write(marker,dict(started_at_utc=e2a.utc(),barrier=barrier))
    executions=[]
    with run.measurement_lock():
        for job in (j for j in manifest['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id']
            if (folder/'execution.json').exists():executions.append(read(folder/'execution.json'));continue
            folder.mkdir(parents=True,exist_ok=False);binding=manifest['programs'][job['program_key']]
            for r in (binding['program'],binding['reference']):require(sha(r['path'])==r['sha256'],'mathematical input changed')
            request,output=folder/'request.json',folder/'result.json';write(request,job['request'])
            command=[manifest['binary']['path'],binding['program']['path'],str(request),str(output)]
            execution=dict(job=job,started_at_utc=e2a.utc(),command=command,status='failed',barrier=barrier)
            process=None;handles=[]
            try:
                env=run.child_environment(manifest['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1'
                process,stdout,stderr,affinity=run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295)
                handles=[stdout,stderr];execution.update(process_id=process.pid,process_affinity=affinity)
                code=process.wait(timeout=manifest['protocol']['timeout_seconds'])
                execution.update(returncode=code,process_exited_verified=True,forced_termination=False)
                require(code==0,'native child failed; no retry')
                execution['verification']=verify(read(output),job['request'],binding,process.pid)
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
            executions.append(execution);print(json.dumps(dict(job=job['id'],status=execution['status'],error=execution.get('error'))),flush=True)
    path=out/(split+'-executions.json')
    if not path.exists():write(path,executions)
    return dict(split=split,declared=len(executions),completed=sum(e['status']=='completed' for e in executions),
                failed=sum(e['status']!='completed' for e in executions))


def summarize(out,split):
    out=Path(out);m=read(out/'manifest.json');records=[]
    for e in read(out/(split+'-executions.json')):
        if e['status']!='completed':continue
        require(sha(e['native']['path'])==e['native']['sha256'],'raw idle evidence changed')
        n=read(e['native']['path']);conditions=[]
        for idle in e['job']['request']['idle_ms']:
            rows=[r for r in n['sequences'] if r['requested_idle_ms']==idle]
            values=[c['wall_ns'] for r in rows for c in r['copies']]
            burst=[sum(c['wall_ns'] for c in r['copies'])/16 for r in rows]
            conditions.append(dict(requested_idle_ms=idle,condition_count=len(rows),
                actual_idle_ns=[r['actual_idle_ns'] for r in rows],
                ordinal_mean_ns=[statistics.mean(r['copies'][i]['wall_ns'] for r in rows) for i in range(16)],
                first_copy_mean_ns=statistics.mean(r['copies'][0]['wall_ns'] for r in rows),
                later_copy_mean_ns=statistics.mean(c['wall_ns'] for r in rows for c in r['copies'][1:]),
                burst_mean_ns=statistics.mean(burst),burst_cv=statistics.stdev(burst)/statistics.mean(burst),
                all_copy_mean_ns=statistics.mean(values)))
        records.append(dict(job=e['job'],native=e['native'],conditions=conditions))
    summary=dict(schema='dynamic-idle-observations-1',split=split,records=records,
        declared=sum(j['split']==split for j in m['jobs']),completed=len(records),
        performance_fit=False,whole_Dense_target_used_as_parameter=False,physical_cause_identified=False)
    write(out/(split+'-observations.json'),summary)
    return dict(split=split,declared=summary['declared'],completed=len(records),summary=str(out/(split+'-observations.json')))


def freeze_predictions(out,model,predictions):
    """Bind an explicit model and all requested-idle response predictions."""
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    require(not (out/'heldout-started.json').exists() and not any((out/'jobs'/j['id']).exists() for j in m['jobs'] if j['split']=='heldout'),
            'cannot freeze after heldout acquisition')
    executions=read(out/'profile-executions.json')
    require(len(executions)==6 and all(e['status']=='completed' for e in executions),'six complete calibration processes required')
    p=read(predictions);require(p['schema']=='dynamic-idle-frozen-predictions-1','prediction schema differs')
    require(p.get('future_target_state_used') is False and p.get('whole_Dense_target_used_for_fit') is False,'invalid prediction provenance')
    wanted={(j['memory_mode'],idle,ordinal) for j in m['jobs'] if j['split']=='heldout'
            for idle in j['request']['idle_ms'] for ordinal in range(16)}
    rows=p['rows'];got={(r['memory_mode'],r['requested_idle_ms'],r['ordinal']) for r in rows}
    require(got==wanted and len(rows)==len(wanted),'all exact heldout idle/ordinal predictions required')
    from fractions import Fraction
    require(all(Fraction(str(r['predicted_ns']))>=0 for r in rows),'unknown/negative frozen response')
    for name,path in (('model-frozen.json',model),('predictions-frozen.json',predictions)):
        target=out/name;require(not target.exists(),'frozen output exists');shutil.copy2(path,target)
    barrier=dict(schema='dynamic-idle-freeze-1',frozen_at_utc=e2a.utc(),
        files={name:sha(out/name) for name in ('model-frozen.json','predictions-frozen.json')},
        manifest=receipt(out/'manifest.json'),calibration=receipt(out/'profile-executions.json'),
        sources=[receipt(Path(__file__))],all_heldout_coordinates_known=True,target_time_used_for_fit=False)
    write(out/'frozen-barrier.json',barrier)
    return dict(barrier=receipt(out/'frozen-barrier.json'),predictions=len(rows))


def evaluate(out):
    """Report independent planned-idle forecasts; raw variability stays separate."""
    from fractions import Fraction
    import numpy as np
    out=Path(out).resolve();m=read(out/'manifest.json');check(m)
    barrier=read(out/'frozen-barrier.json')
    require(sha(out/'manifest.json')==barrier['manifest']['sha256'],'frozen experiment manifest changed')
    for name,h in barrier['files'].items():require(sha(out/name)==h,'frozen model/predictions changed')
    frozen=read(out/'predictions-frozen.json')
    points={(r['memory_mode'],r['requested_idle_ms'],r['ordinal']):float(Fraction(str(r['predicted_ns']))) for r in frozen['rows']}
    expected=[j for j in m['jobs'] if j['split']=='heldout'];executions=read(out/'heldout-executions.json')
    require(len(expected)==len(executions)==6 and {e['job']['id'] for e in executions}=={j['id'] for j in expected},
            'all six original heldout processes must remain in denominator')
    records=[]
    for execution in executions:
        job=execution['job']
        if execution['status']!='completed':
            records.append(dict(job=job,status=execution['status'],qualified=False,error=execution.get('error')));continue
        require(sha(execution['native']['path'])==execution['native']['sha256'],'heldout raw changed')
        n=read(execution['native']['path']);verify(n,job['request'],m['programs'][job['program_key']],execution['process_id'])
        for idle in job['request']['idle_ms']:
            rows=[r for r in n['sequences'] if r['requested_idle_ms']==idle]
            predictions=[points[(job['memory_mode'],idle,i)] for i in range(16)]
            measured_bursts=[statistics.mean(c['wall_ns'] for c in r['copies']) for r in rows]
            predicted_burst=statistics.mean(predictions)
            mean_ape=abs(predicted_burst-statistics.mean(measured_bursts))/statistics.mean(measured_bursts)
            p90=float(np.quantile([abs(predicted_burst-v)/v for v in measured_bursts],.9,method='higher'))
            ordinal_means=[statistics.mean(r['copies'][i]['wall_ns'] for r in rows) for i in range(16)]
            first_ape=abs(predictions[0]-ordinal_means[0])/ordinal_means[0]
            records.append(dict(job=job,status='completed',requested_idle_ms=idle,predicted_burst_mean_ns=predicted_burst,
                observed_burst_mean_ns=statistics.mean(measured_bursts),burst_mean_ape=mean_ape,burst_ape_p90=p90,
                first_copy_mean_ape=first_ape,first_copy_mean_accuracy_passed=first_ape<=.1,
                predicted_ordinal_ns=predictions,observed_ordinal_mean_ns=ordinal_means,
                raw_burst_cv=statistics.stdev(measured_bursts)/statistics.mean(measured_bursts),
                actual_idle_ns=[r['actual_idle_ns'] for r in rows],
                qualified=mean_ape<=.1 and p90<=.2,native=execution['native']))
    cases=[]
    for memory in ('pageable','registered'):
        for idle in IDLES['heldout']:
            selected=[r for r in records if r['job']['memory_mode']==memory and r.get('requested_idle_ms')==idle]
            cases.append(dict(memory_mode=memory,requested_idle_ms=idle,declared_processes=3,
                completed_processes=len(selected),mean_response_qualified=len(selected)==3 and all(r['qualified'] for r in selected),
                first_response_accuracy_passed=len(selected)==3 and all(r['first_copy_mean_accuracy_passed'] for r in selected),
                maximum_process_burst_mean_ape=max((r['burst_mean_ape'] for r in selected),default=None),
                maximum_process_burst_p90_ape=max((r['burst_ape_p90'] for r in selected),default=None),
                maximum_first_copy_mean_ape=max((r['first_copy_mean_ape'] for r in selected),default=None)))
    result=dict(schema='dynamic-idle-independent-validation-1',evaluated_at_utc=e2a.utc(),
        cases=cases,process_conditions=records,declared_processes=6,
        completed_processes=sum(e['status']=='completed' for e in executions),
        failed_processes=sum(e['status']!='completed' for e in executions),
        burst_mean_qualified_cases=sum(r['mean_response_qualified'] for r in cases),
        first_response_accuracy_cases=sum(r['first_response_accuracy_passed'] for r in cases),
        primary_input='predeclared requested idle, not post-target observations',
        units='inclusive host-completed response ns',raw_cv_is_prediction_gate=False,
        physical_cause_identified=False,general_external_resource_accuracy_qualified=False,
        frozen=receipt(out/'frozen-barrier.json'))
    write(out/'summary.json',result)
    return dict(summary=str(out/'summary.json'),burst_mean_qualified_cases=result['burst_mean_qualified_cases'],
                first_response_accuracy_cases=result['first_response_accuracy_cases'],declared_cases=8)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','profile','heldout','summarize-profile','summarize-heldout','freeze','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--binary');p.add_argument('--smoke',action='store_true')
    p.add_argument('--model');p.add_argument('--predictions');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.out,a.binary,smoke=a.smoke)
    elif a.command=='freeze':result=freeze_predictions(a.out,a.model,a.predictions)
    elif a.command=='evaluate':result=evaluate(a.out)
    elif a.command.startswith('summarize-'):result=summarize(a.out,a.command.split('-',1)[1])
    else:result=collect(a.out,a.command)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
