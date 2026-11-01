"""Freeze event-parked PROFILE cycles and scoped completion-response fits.

No target heldout cycles/timing enter calibration or BEFORE resource inputs.
Cycles remain raw QueryThreadCycleTime work. The reference cycles/wall ratio
is an empirical workload response, never a conversion using an assumed GHz.
Completion residuals are constrained local fit candidates, not measured GPU
active time. All physical/causal qualification remains explicitly false.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

from profiling.host_supply_calibration import (
    OBSERVATION_SCHEMA, calibrate_blackout_hypothesis, digest, require,
)
from profiling.event_stage_recovery import event_effective_jobs

SCHEMA='event-stage-profile-fit-1'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path,expected=None):
    raw=Path(path).read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest()==expected,'source hash differs: '+str(path))
    return json.loads(raw)


def _receipt(path):return dict(path=str(Path(path).resolve()),sha256=_sha(path))


def _fraction(value):
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def _pack(value):
    if isinstance(value,Fraction):return dict(numerator=value.numerator,denominator=value.denominator)
    if isinstance(value,dict):return {k:_pack(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_pack(v) for v in value]
    return value


def _median(values):
    if not values:return None
    values=sorted(map(Fraction,values));n=len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


def _ceil(value):return None if value is None else (value.numerator+value.denominator-1)//value.denominator


def _stats(values):
    if not values:return dict(n=0,median=None,mean=None,minimum=None,maximum=None,cv=None)
    values=list(map(Fraction,values));mean=sum(values)/len(values)
    return dict(n=len(values),median=_median(values),mean=mean,minimum=min(values),maximum=max(values),
        cv=statistics.pstdev(map(float,values))/float(mean) if mean else 0.0)


def _signature(program):
    return digest({k:program[k] for k in ('boundary_contract','graph','physical','actions','bindings')})


def _cycles(action):
    c=action.get('thread_cycles')
    require(isinstance(c,dict) and c.get('enabled') is True and c.get('unit')=='raw_QueryThreadCycleTime_cycles',
        'declared per-action raw cycle receipt missing')
    require(c.get('cycles_converted_to_time') is False and c.get('usable_as_heldout_resource_input') is False,
        'raw-cycle scope or heldout-input boundary differs')
    fields=('before_query_begin_ns','before_query_end_ns','after_query_begin_ns','after_query_end_ns')
    require(all(type(c.get(k)) is int and c[k]>=0 for k in fields),'cycle query time brackets unknown')
    require(c[fields[0]]<=c[fields[1]]<=action['begin_ns']<=action['end_ns']<=c[fields[2]]<=c[fields[3]],
        'action/counter time brackets overlap or reverse')
    delta=c.get('delta_cycles');before=c.get('before_cycles');after=c.get('after_cycles')
    if delta is None:
        return None,c
    require(type(delta) is int and delta>=0 and type(before) is int and type(after) is int and
        after>=before and delta==after-before and c.get('monotonic') is True,'raw cycle delta differs')
    return delta,c


def _event(raw,process):
    e=raw.get('event_parking');ready=process.get('ready',{}).get('event_parking')
    require(isinstance(e,dict) and e==ready and e.get('schema')=='physical-event-parking-1' and
        e.get('mode')=='parent_owned_Windows_events' and e.get('post_READY_periodic_file_poll') is False and
        e.get('post_sampling_periodic_wakeup') is False,'actor event parking contract differs')
    controller=process.get('event_controller',{}).get('names',{})
    require(e.get('go_event')==controller.get('go') and e.get('stop_event')==controller.get('stop'),
        'parent/native event identities differ')


def _load_profile(root,parent,protocol,job):
    path=root/'jobs'/job['id']/'execution.json';execution=_read(path)
    require(execution.get('job')==job and execution.get('status')=='completed' and execution.get('cleanup_verified') is True,
        'PROFILE execution incomplete or job differs')
    binding=parent['programs'][job['program_key']]
    require(execution.get('program')==binding,'PROFILE program binding differs')
    program=_read(binding['path'],binding['sha256']);_read(binding['oracle'],binding['oracle_sha256'])
    require(program.get('program_sha256')==binding['program_sha256']==digest({k:v for k,v in program.items() if k!='program_sha256'}),
        'PROFILE semantic program digest differs')
    sources=[_receipt(path),_receipt(binding['path']),_receipt(binding['oracle'])]
    actors={p['name']:p for p in execution['processes']}
    require(len(actors)==len(execution['processes']) and set(actors)==set(execution['raw_files']), 'owned actor inventory differs')
    raw={}
    for name,process in actors.items():
        require(process.get('returncode')==0 and process.get('forced_termination') is False and
            process.get('process_exited_verified') is True,'owned PROFILE actor did not exit normally')
        source=execution['raw_files'][name];sources.append(dict(source))
        require(_sha(source['path'])==source['sha256'],'PROFILE actor raw bytes differ')
        numerical=execution['numerical'].get(name,{})
        require(all(numerical.get(k) is True for k in ('passed','full_graph_equal','action_trace_equal',
            'full_port_inventory_equal','copy_routes_bytes_and_values_equal')),'independent numerical receipt missing')
        if name not in {'main','before-cpu','before-gpu'}:continue
        native=_read(source['path'],source['sha256']);raw[name]=native
        require(native.get('schema')=='physical-stage-native-1' and native.get('process_id')==process['pid']==process['ready']['process_id'],
            'PROFILE native actor identity differs')
        require(native.get('main_thread_id')==process['ready']['main_thread_id'] and native['clock']==process['ready']['physical_clock'],
            'PROFILE thread/clock identity differs')
        require(native['clock']['clock_id']=='windows_qpc_floor_ns' and native['clock']['clock_origin']=='absolute_qpc_zero',
            'PROFILE common clock differs')
        require(native.get('program_sha256')==program['program_sha256'] and native.get('actions')==program['actions'],
            'PROFILE native program/action identity differs')
        require(Path(process['command'][0]).resolve()==Path(protocol['event_binary']).resolve() and
            Path(process['command'][1]).resolve()==Path(binding['path']).resolve(),'PROFILE binary/program command differs')
        _event(native,process)
        require(native.get('thread_cycles_enabled') is True,'PROFILE cycle measurements must be enabled')
        count=job['samples'] if name=='main' else parent['protocol']['reference_'+name.split('-')[-1]+'_samples']
        require(native.get('samples_requested')==native.get('samples_completed')==len(native['iterations'])==count,
            'PROFILE declared sample coverage differs')
        expected_target=None if name=='main' else parent['protocol']['targets']['reference_'+name.split('-')[-1]]
        require(native.get('target_action_index')==expected_target and native.get('mode')==('full' if name=='main' else 'target'),
            'PROFILE actor target differs')
        if name=='main':require(native.get('instrumentation')=='bare','PROFILE main must use bare timer')
    require(set(raw)=={'main','before-cpu','before-gpu'},'PROFILE BEFORE references missing')
    return dict(job=job,execution=execution,program=program,raw=raw,sources=sources)


def reference_profile_summary(raw):
    """PROFILE raw-cycle work / full iteration-wall empirical response.

    The full iteration denominator matches the independent BEFORE wall-only
    input path. The narrower query bracket stays visible as a diagnostic.
    """
    walls=[];brackets=[];iteration_walls=[];outside_query=[];work=[];rates=[];unknown=0
    for invocation in raw['iterations']:
        require(len(invocation['actions'])==1,'isolated reference must have exactly one action')
        action=invocation['actions'][0];delta,c=_cycles(action)
        walls.append(Fraction(action['end_ns']-action['begin_ns']))
        bracket=c['after_query_end_ns']-c['before_query_begin_ns'];brackets.append(Fraction(bracket))
        require(type(invocation.get('begin_ns')) is int and type(invocation.get('end_ns')) is int and
            invocation['begin_ns']<=c['before_query_begin_ns']<=c['after_query_end_ns']<=invocation['end_ns'],
            'reference counter brackets lie outside full iteration')
        iteration_wall=invocation['end_ns']-invocation['begin_ns']
        iteration_walls.append(Fraction(iteration_wall));outside_query.append(Fraction(iteration_wall-bracket))
        if delta is None or iteration_wall<=0:unknown+=1;continue
        work.append(Fraction(delta));rates.append(Fraction(delta*1_000_000_000,iteration_wall))
    return dict(work_cycles=_median(work),work_cycles_ceiling=_ceil(_median(work)),work_statistics=_stats(work),
        fast_action_wall_ns=_median(walls),outer_query_wall_ns=_median(brackets),
        outer_query_wall_statistics=_stats(brackets),full_iteration_wall_ns=_median(iteration_walls),
        full_iteration_wall_statistics=_stats(iteration_walls),iteration_outside_query_wall_statistics=_stats(outside_query),
        wall_statistic='exact median of all reference action observations; no pause/latency filtering',
        empirical_cycle_rate_per_s=_median(rates),cycle_rate_statistics=_stats(rates),
        unknown_cycle_samples=unknown,query_overhead_in_wall=True,
        cycle_rate_denominator='full_reference_iteration_QPC_wall_ns',
        scope='PROFILE raw reference work cycles / full reference QPC iteration wall; includes query/runtime/head/tail effects',
        physical_frequency_or_neutral_capacity_identified=False,cycles_converted_with_assumed_GHz=False)


def fit_nonnegative_affine(x,y,*,regularization):
    """Exact two-feature constrained least squares, with an explicit slope ridge."""
    require(len(x)==len(y) and len(x)>=3,'three profile feature/response points required')
    x=list(map(Fraction,x));y=list(map(Fraction,y));regularization=Fraction(regularization)
    require(regularization>=0 and all(v>=0 for v in x+y),'nonnegative feature/response/ridge required')
    if min(x)==max(x):return dict(identified=False,intercept=None,slope=None,reason='GPU reference feature does not vary')
    n=Fraction(len(x));sx=sum(x);sy=sum(y);sxx=sum(v*v for v in x)+regularization;sxy=sum(a*b for a,b in zip(x,y))
    candidates=[(sy/n,Fraction(0)),(Fraction(0),sxy/sxx if sxx else Fraction(0)),(Fraction(0),Fraction(0))]
    determinant=n*sxx-sx*sx
    if determinant>0:
        a=(sy*sxx-sx*sxy)/determinant;b=(n*sxy-sx*sy)/determinant
        if a>=0 and b>=0:candidates.append((a,b))
    objective=lambda pair:sum((pair[0]+pair[1]*a-b)**2 for a,b in zip(x,y))+regularization*pair[1]**2
    a,b=min(candidates,key=objective)
    return dict(identified=True,intercept=a,slope=b,objective=objective((a,b)),regularization=regularization,
        observations=len(x),model='nonnegative local intercept + slope * GPU reference relative slowdown',
        intercept_is_whole_layer_offset=False,physical_mechanism_identified=False)


def _cpu_context(parent,program):
    index=parent['protocol']['targets']['cpu_background']
    return dict(external_actors=[dict(resource_id='cpu:external-replay',
        workload_signature=dict(action=program['actions'][index],physical=program['physical']),
        placement=dict(process_affinity_mask=parent['protocol']['cpu_background_mask']))],
        target_placement=dict(process_affinity_mask=parent['protocol']['main_process_mask']))


def derive_before_phase_anchor(cpu_raw,hypothesis,*,cutoff_ns,target_process_id,evidence,minimum_excess_ns):
    """Use BEFORE wall intervals only. No raw cycle counters are accessed."""
    require(cpu_raw['process_id']!=target_process_id and cpu_raw['sampling_end_ns']<=cutoff_ns,
        'phase source is target or crosses target cutoff')
    require(cpu_raw['clock']['clock_id']==hypothesis['context']['clock_id'] and
        cpu_raw['clock']['clock_origin']==hypothesis['context']['clock_origin'],'phase clock differs')
    actions=[i['actions'][0] for i in cpu_raw['iterations']]
    walls=[a['end_ns']-a['begin_ns'] for a in actions]
    fast=_median(walls)
    long=[(index,a) for index,a in enumerate(actions) if a['end_ns']-a['begin_ns']-fast>=minimum_excess_ns]
    if not long or hypothesis['period_candidate_ns'] is None:
        return dict(known=False,reason='No BEFORE pause event or profile recurrence unknown',phase_anchor=None)
    index,action=long[-1];uncertainty=_ceil(fast)
    anchor=dict(schema='host-pause-before-anchor-1',sample_role='before_target',source_role='independent_resource_probe',
        derived_from_target_timing=False,uses_future_observations=False,signature_sha256=hypothesis['signature_sha256'],
        resource_id=hypothesis['context']['resource_id'],competitor_context=hypothesis['context']['competitor_context'],
        clock_id=cpu_raw['clock']['clock_id'],clock_origin=cpu_raw['clock']['clock_origin'],
        sensor_begin_ns=cpu_raw['sampling_begin_ns'],sensor_end_ns=cpu_raw['sampling_end_ns'],
        phase_start_window_ns=dict(minimum=action['begin_ns'],maximum=min(action['end_ns'],action['begin_ns']+uncertainty)),
        observed_process_ids=[cpu_raw['process_id']],source_observation_ids=[f'before-reference:{cpu_raw["process_id"]}:{index}'],
        evidence=list(evidence),derivation='last BEFORE long reference action entry + fast-wall active-work uncertainty',
        uncertainty_bound_is_physically_proven=False,raw_cycle_counters_used=False)
    return dict(known=True,phase_anchor=anchor,reference_fast_wall_ns=fast,last_long_reference_action=action['action_index'],
        pause_end_window_ns=dict(minimum=max(action['begin_ns'],action['end_ns']-uncertainty),maximum=action['end_ns']),
        pause_start_window_ns=anchor['phase_start_window_ns'],
        pause_endpoint_bounds_are_physically_proven=False,
        source_iteration=index,causal_cutoff_ns=cutoff_ns,scope='before-only diagnostic phase hypothesis')


def before_wall_inputs(frozen,cpu_raw,gpu_raw,*,cutoff_ns,target_process_id,cpu_evidence,gpu_evidence):
    """Heldout legal inputs: frozen PROFILE work and independent BEFORE walls.

    Deliberately never reads thread_cycles, target elapsed time or AFTER data.
    Changing any heldout raw-cycle counters cannot change this result.
    """
    for role,raw in (('cpu',cpu_raw),('gpu',gpu_raw)):
        require(raw['process_id']!=target_process_id and raw['sampling_end_ns']<=cutoff_ns,
            'BEFORE reference source is target or crosses cutoff')
        require(raw.get('schema')=='physical-stage-native-1' and raw.get('mode')=='target' and
            raw.get('target_action_index')==frozen['reference_targets'][role], 'BEFORE reference target identity differs')
        require(raw['clock']==frozen['clock'],'BEFORE reference clock differs')
        require(len(raw['iterations'])==raw.get('samples_requested')==raw.get('samples_completed') and
            all(len(i['actions'])==1 and i['actions'][0]['action_index']==frozen['reference_targets'][role]
                for i in raw['iterations']),'BEFORE reference sample/action inventory differs')
    # The full reference iteration wall includes its measurement-query overhead.
    # No field inside thread_cycles is read on this heldout-input path.
    cpu_walls=[i['end_ns']-i['begin_ns'] for i in cpu_raw['iterations']]
    gpu_walls=[a['end_ns']-a['begin_ns'] for i in gpu_raw['iterations'] for a in i['actions']]
    cpu_wall=_median(cpu_walls);gpu_wall=_median(gpu_walls)
    work=_fraction(frozen['baseline_cpu_reference_work_cycles'])
    require(cpu_wall is not None and cpu_wall>0 and gpu_wall is not None and gpu_wall>0,'BEFORE reference wall unknown')
    result=dict(cpu_cycle_rate_per_s=work*1_000_000_000/cpu_wall,
        gpu_reference_fast_action_wall_ns=gpu_wall,cpu_reference_fast_action_wall_ns=cpu_wall,
        measured_heldout_cycles_used=False,target_or_AFTER_outcomes_used=False,
        scope='frozen PROFILE reference work / independent BEFORE QPC iteration wall; query overhead retained; diagnostic response',
        evidence=dict(cpu=list(cpu_evidence),gpu=list(gpu_evidence)))
    hypothesis=frozen.get('CPU_blackout_hypothesis')
    result['before_phase']=None if hypothesis is None else derive_before_phase_anchor(cpu_raw,hypothesis,
        cutoff_ns=cutoff_ns,target_process_id=target_process_id,evidence=cpu_evidence,
        minimum_excess_ns=frozen['declared_protocol']['minimum_excess_ns'])
    return _pack(result)


def freeze_event_profiles(root,*,minimum_excess_ns=15_000_000,affine_regularization=Fraction(1,1000),additional_source_files=()):
    """Read PROFILE jobs only and return a JSON-safe immutable fit artifact."""
    root=Path(root).resolve();parent=_read(root/'manifest.json');protocol=_read(root/'event-protocol.json')
    require(protocol.get('schema')=='event-stage-campaign-1' and protocol['parent_manifest_sha256']==_sha(root/'manifest.json'),
        'event protocol parent differs')
    require(parent['source_inventory_sha256']==digest(parent['sources']),'parent source inventory differs')
    require(_sha(protocol['event_binary'])==parent['sources'][str(Path(protocol['event_binary']).resolve())], 'event binary changed')
    for path,expected in protocol['source_receipts'].items():require(_sha(path)==expected,'event protocol source changed: '+path)
    jobs=[j for j in event_effective_jobs(root,protocol) if j['split']=='profile']
    require(len(jobs)==12 and all(j['parking']=='event' and j['cycles'] is True and j['samples']==96 for j in jobs),
        'expected declared 12 x 96 event PROFILE jobs')
    loaded=[_load_profile(root,parent,protocol,job) for job in jobs]
    signatures={_signature(item['program']) for item in loaded};require(len(signatures)==1,'mixed profile program signatures')
    require(len({item['raw']['main']['process_id'] for item in loaded})==12,'profile processes not independent')
    action_count=len(loaded[0]['program']['actions']);per_job=[];all_cycles=defaultdict(list);all_gaps=defaultdict(list)
    head=[];tail=[];blackout_observations=[];sources=[_receipt(root/'manifest.json'),_receipt(root/'event-protocol.json')]
    for item in loaded:
        native=item['raw']['main'];sources.extend(item['sources']);cycles=defaultdict(list);gaps=defaultdict(list);walls=defaultdict(list)
        cpu=reference_profile_summary(item['raw']['before-cpu']);gpu=reference_profile_summary(item['raw']['before-gpu'])
        for invocation in native['iterations']:
            require(len(invocation['actions'])==action_count,'profile action count differs')
            obs_actions=[];previous=None
            for ordinal,action in enumerate(invocation['actions']):
                require(action['action_index']==ordinal,'profile original action order differs')
                delta,c=_cycles(action);walls[ordinal].append(Fraction(action['end_ns']-action['begin_ns']))
                if delta is not None:cycles[ordinal].append(Fraction(delta));all_cycles[ordinal].append(Fraction(delta))
                if previous is not None:
                    p=previous['thread_cycles'];before=c.get('before_cycles');after=p.get('after_cycles')
                    if before is not None and after is not None:
                        require(before>=after,'inter-action cycle counters moved backwards')
                        gap=Fraction(before-after);gaps[ordinal].append(gap);all_gaps[ordinal].append(gap)
                obs_actions.append(dict(action_index=ordinal,begin_ns=action['begin_ns'],end_ns=action['end_ns'],
                    active_cycles=delta,active_cycles_scope='main_thread_raw_cycles'))
                previous=action
            first=invocation['actions'][0];last=invocation['actions'][-1]
            head.append(dict(minimum_ns=first['thread_cycles']['before_query_begin_ns']-invocation['begin_ns'],
                maximum_ns=first['begin_ns']-invocation['begin_ns']))
            tail.append(dict(minimum_ns=invocation['end_ns']-last['thread_cycles']['after_query_end_ns'],
                maximum_ns=invocation['end_ns']-last['end_ns']))
            if item['job']['cpu_background']:
                blackout_observations.append(dict(schema=OBSERVATION_SCHEMA,observation_id=item['job']['id']+':'+str(invocation['iteration']),
                    run_id=item['job']['id'],process_id=native['process_id'],seed=item['execution']['program']['seed'],split='profile',
                    signature=next(iter(signatures)),resource_id='cpu:host-mask:'+str(parent['protocol']['main_process_mask']),
                    competitor_context=_cpu_context(parent,item['program']),clock_id=native['clock']['clock_id'],clock_origin=native['clock']['clock_origin'],
                    begin_ns=invocation['begin_ns'],end_ns=invocation['end_ns'],actions=obs_actions,numerical_verified=True,
                    evidence=[item['execution']['raw_files']['main']['path']+':sha256:'+item['execution']['raw_files']['main']['sha256']]))
        capacity=cpu['empirical_cycle_rate_per_s'];action_rows={}
        for index in range(action_count):
            work=_median(cycles[index]);wall=_median(walls[index]);semantic=item['program']['actions'][index]
            gpu_allowed=(semantic['kind']=='COMPUTE' and semantic.get('device')=='gpu') or (
                semantic['kind']=='COPY' and 'vram' in (semantic.get('src'),semantic.get('dst')))
            residual=None if work is None or capacity is None or capacity<=0 else wall-work*1_000_000_000/capacity
            projected=None if residual is None else max(Fraction(0),residual) if gpu_allowed else Fraction(0)
            action_rows[str(index)]=dict(cpu_cycles_median=work,cpu_cycles_ceiling=_ceil(work),raw_cycle_statistics=_stats(cycles[index]),
                fast_action_wall_ns=wall,wall_statistic='exact median of all profile action observations; no pause filtering',
                unprojected_completion_residual_ns=residual,completion_response_candidate_ns=projected,
                negative_residual=None if residual is None else residual<0,completion_allowed_by_semantics=gpu_allowed,
                residual_is_GPU_active_time=False,query_overhead_and_wait_policy_not_identified=True)
        per_job.append(dict(job_id=item['job']['id'],process_id=native['process_id'],seed=item['execution']['program']['seed'],
            before_cpu=cpu,before_gpu=gpu,actions=action_rows,gap_cycle_medians={str(i):_median(v) for i,v in gaps.items()},
            source_main=item['execution']['raw_files']['main']))
    gpu_references=sorted(j['before_gpu']['fast_action_wall_ns'] for j in per_job if j['before_gpu']['fast_action_wall_ns'] is not None)
    require(len(gpu_references)==12,'GPU reference profile coverage unknown')
    baseline_gpu=_median(gpu_references[:3]);require(baseline_gpu>0,'GPU baseline reference wall nonpositive')
    features=[max(Fraction(0),j['before_gpu']['fast_action_wall_ns']/baseline_gpu-1) for j in per_job]
    action_parameters={}
    for index in range(action_count):
        rows=[j['actions'][str(index)] for j in per_job];semantic=loaded[0]['program']['actions'][index]
        cycle_y=[r['cpu_cycles_median'] for r in rows];completion_y=[r['completion_response_candidate_ns'] for r in rows]
        work=_median(all_cycles[index]);allowed=rows[0]['completion_allowed_by_semantics']
        cycle_fit=fit_nonnegative_affine(features,cycle_y,regularization=affine_regularization) if all(v is not None for v in cycle_y) else None
        if semantic.get('device')=='cpu' or semantic['kind'] in {'ADVANCE','EVICT'}:
            cycle_fit=dict(identified=work is not None,intercept=work,slope=Fraction(0),scope='CPU-only constant-work hypothesis by declared semantics')
        completion_fit=(fit_nonnegative_affine(features,completion_y,regularization=affine_regularization)
            if allowed and all(v is not None for v in completion_y) else
            dict(identified=not allowed,intercept=Fraction(0) if not allowed else None,slope=Fraction(0) if not allowed else None,
                scope='zero completion response by declared CPU/control semantics' if not allowed else 'completion response unknown'))
        action_parameters[str(index)]=dict(action_index=index,kind=semantic['kind'],device=semantic.get('device'),
            target_id=semantic.get('target_id'),cpu_cycles_median=work,cpu_cycles_ceiling=_ceil(work),
            cpu_cycle_statistics=_stats(all_cycles[index]),cpu_work_affine=cycle_fit,completion_response_affine=completion_fit,
            fast_wall_ns=_median([r['fast_action_wall_ns'] for r in rows]),
            negative_profile_residuals=sum(r['negative_residual'] is True for r in rows),
            completion_is_measured_GPU_active_time=False)
    blackout=calibrate_blackout_hypothesis(blackout_observations,next(iter(signatures)),
        minimum_excess_ns=minimum_excess_ns,min_independent_runs=3)
    reference_work=_median([j['before_cpu']['work_cycles'] for j in per_job if j['before_cpu']['work_cycles'] is not None])
    require(reference_work is not None,'baseline CPU reference work unknown')
    for item,job in zip(loaded,per_job):
        job['before_phase']=derive_before_phase_anchor(item['raw']['before-cpu'],blackout,
            cutoff_ns=item['raw']['main']['sampling_begin_ns'],target_process_id=item['raw']['main']['process_id'],
            evidence=[item['execution']['raw_files']['before-cpu']['path']],minimum_excess_ns=minimum_excess_ns)
    model_dir=Path(__file__).resolve().parents[1]/'modeling/llm_infer_model/performance'
    amendments=[root/'event-recovery-index.json'] if (root/'event-recovery-index.json').exists() else []
    for path in [Path(__file__),Path(__file__).with_name('host_supply_calibration.py'),
                 Path(__file__).with_name('physical_posttarget_collection.py'),*amendments,*model_dir.glob('*.py'),
                 *Path(__file__).parent.glob('event_stage_*.py'),*map(Path,additional_source_files)]:sources.append(_receipt(path))
    result=dict(schema=SCHEMA,campaign_root=str(root),manifest_sha256=_sha(root/'manifest.json'),
        event_protocol_sha256=_sha(root/'event-protocol.json'),signature=next(iter(signatures)),
        clock=loaded[0]['raw']['main']['clock'],
        reference_targets=dict(cpu=parent['protocol']['targets']['reference_cpu'],gpu=parent['protocol']['targets']['reference_gpu']),
        source_receipts=list({s['path']:s for s in sources}.values()),profile_identities=[dict(job_id=j['job_id'],process_id=j['process_id'],seed=j['seed']) for j in per_job],
        declared_protocol=dict(minimum_excess_ns=minimum_excess_ns,affine_regularization=affine_regularization,
            baseline_gpu_reference_selection='median of three smallest PROFILE reference action medians; no state label',
            cycle_work_rounding='ceiling of exact median; exact rational retained',heldout_CPU_capacity='frozen PROFILE reference work / BEFORE reference QPC wall only'),
        baseline_gpu_reference_wall_ns=baseline_gpu,baseline_cpu_reference_work_cycles=reference_work,
        actions=action_parameters,gap_cycles={str(i):dict(median=_median(v),ceiling=_ceil(_median(v)),statistics=_stats(v)) for i,v in all_gaps.items()},
        head_wall_bounds=dict(minimum_statistics=_stats([h['minimum_ns'] for h in head]),maximum_statistics=_stats([h['maximum_ns'] for h in head])),
        tail_wall_bounds=dict(minimum_statistics=_stats([t['minimum_ns'] for t in tail]),maximum_statistics=_stats([t['maximum_ns'] for t in tail])),
        CPU_blackout_hypothesis=blackout,profile_jobs=per_job,heldout_data_read=False,heldout_cycles_used=False,
        target_wall_used_as_external_supply=False,whole_layer_offset_fitted=False,physical_stage_identification_qualified=False,
        hardware_accuracy_qualified=False,phase_fitted_from_heldout=False,
        limitations=['Raw cycles measure scoped active thread response, not useful instructions or elapsed time.',
            'Per-action completion residual is an underidentified constrained fit, never measured GPU active time.',
            'Counter-query overhead is present in profile and separately requires matched observer control.',
            'Periodic CPU pause and BEFORE phase remain diagnostic assumptions, not a simulated OS schedule.',
            'Profile-only CPU cycle work may depend on backend waiting/spinning; affinity/runtime context is fixed.'])
    result=_pack(result);result['artifact_sha256']=digest(result)
    return result
