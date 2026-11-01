"""Profile-only physical accounting and predeclared reference-response candidates.

This adapter never estimates hardware capacity from target latency. References
are separate processes, run BEFORE the main batch. They are stale workload
response measurements, not instantaneous available-service trajectories. AFTER
references audit drift only. CUPTI envelopes are not SM-active service, and API
wall time is not active CPU work. All predictions are explicitly diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

from profiling.available_service.calibration import digest, encoded, exact, require, utc
from profiling.available_service.physical_records import parse_physical_records, interval_union, interval_length
from profiling.physical_job_index import effective_jobs
from llm_infer_model.performance.reference_components import ReferenceComponent, evaluate_reference_components
from llm_infer_model.performance.parameters import UnknownParameter

SCHEMA = 'physical-component-diagnostic-profile-1'
CANDIDATES = ('idle_component_baseline', 'condition_component_baseline',
              'reference_activity_candidate', 'reference_completed_gpu_candidate',
              'observer_bare_idle_baseline')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, expected=None):
    path = Path(path)
    require(expected is None or _sha(path) == expected, 'source hash differs: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def _receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path))


def _effective_jobs(campaign, manifest):
    """Replace only known failed infrastructure attempts, never select by error."""
    jobs=effective_jobs(campaign,manifest)
    index_path=campaign/'job-replacements.json'
    receipts,notes=[],[]
    if index_path.exists():
        receipts.append(_receipt(index_path))
        index=_read(index_path)
        require(index.get('schema')=='physical-job-replacements-1','unsupported replacement index')
        for replacement in index['replacements']:
            source=dict(path=replacement['failed_execution_path'],sha256=replacement['failed_execution_sha256'])
            _read(source['path'],source['sha256'])
            receipts.append(source)
            # Preserve failed raw evidence as well as the failed execution receipt.
            for path in sorted(Path(source['path']).parent.glob('*/result.json')):
                receipts.append(_receipt(path))
            notes.append(dict(original_job_id=replacement['original_id'],effective_job_id=replacement['replacement_job']['id'],
                reason=replacement['selection_reason'],whole_original_attempt_excluded=True))
    bare_path=campaign/'bare-transfer-protocol.json'
    if bare_path.exists():
        receipts.append(_receipt(bare_path))
    return jobs,receipts,notes


def _pack(value):
    if isinstance(value, Fraction):
        return encoded(value)
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v) for v in value]
    return value


def _mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def _quantile(values, proportion):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * proportion
    left = int(index)
    return float(values[left]) + (float(values[min(left + 1, len(values)-1)]) - float(values[left])) * (index-left)


def _stats(values):
    if not values:
        return dict(n=0, mean=None, median=None, p90=None, worst=None, cv=None)
    values = list(map(float, values))
    mean = statistics.mean(values)
    return dict(n=len(values), mean=mean, median=statistics.median(values),
                p90=_quantile(values, .9), worst=max(values),
                cv=statistics.stdev(values)/mean if len(values)>1 and mean>0 else None)


def _signature(program):
    return digest({k: program[k] for k in ('boundary_contract','graph','physical','actions','bindings')})


def _bounded(intervals, begin, end):
    return [dict(begin_ns=max(begin, r['begin_ns']), end_ns=min(end, r['end_ns']))
            for r in intervals if max(begin, r['begin_ns']) < min(end, r['end_ns'])]


def action_components(action):
    """Disjoint descriptive wall partition; raw overlapping columns stay separate."""
    begin, end = action['begin_ns'], action['end_ns']
    kernel = _bounded(action['kernel_union'], begin, end)
    device = _bounded(action['device_activity_union'], begin, end)
    api = _bounded(action['runtime_api_union'] + action['driver_api_union'], begin, end)
    kernel_ns, device_ns = interval_length(kernel), interval_length(device)
    covered_ns = interval_length(device + api)
    parts = dict(kernel_activity=kernel_ns, copy_activity_outside_kernel=device_ns-kernel_ns,
                 host_api_outside_device=covered_ns-device_ns, other_host_wall=(end-begin)-covered_ns)
    require(all(n>=0 for n in parts.values()) and sum(parts.values()) == end-begin,
            'disjoint action accounting did not telescope')
    return parts


def _load_job(campaign, manifest, job, *, physical=True):
    path = campaign/'jobs'/job['id']/'execution.json'
    row = _read(path)
    require(row.get('schema') == 'physical-service-execution-1' and row.get('job') == job,
            'execution differs from declared job: ' + job['id'])
    require(row.get('status') == 'completed' and row.get('cleanup_verified') is True,
            'incomplete physical job: ' + job['id'])
    require(row.get('source_inventory_sha256') == manifest['source_inventory_sha256'],
            'physical job source inventory differs')
    binding = manifest['programs'][job['program_key']]
    require(row.get('program') == binding, 'job program binding differs')
    program = _read(binding['path'], binding['sha256'])
    require(program.get('program_sha256') == binding['program_sha256'], 'program semantic digest differs')
    raw, sources = {}, [_receipt(path), _receipt(binding['path']), _receipt(binding['oracle'])]
    for name, source in row['raw_files'].items():
        raw[name] = _read(source['path'], source['sha256'])
        sources.append(dict(source))
    native = raw['main']
    require(native['program_sha256'] == binding['program_sha256'], 'main native program differs')
    require(native['mode'] == 'full' and native['instrumentation'] == job['instrumentation'], 'main execution mode differs')
    require(len(native['iterations']) == job['samples'] == native['samples_completed'], 'main sample coverage differs')
    require(all(row['numerical'].get(name, {}).get('passed') is True for name in raw), 'numerical receipts missing')
    require(native['clock']['clock_id'] == 'windows_qpc_floor_ns' and native['clock']['clock_origin'] == 'absolute_qpc_zero',
            'main clock differs')
    parsed = parse_physical_records(native, program=program) if physical else None
    return dict(job=job, execution=row, program=program, native=native, raw=raw, parsed=parsed, sources=sources)


def _components(loaded):
    native, parsed = loaded['native'], loaded['parsed']
    template, vectors = {}, []
    lookup = {(a['iteration'], a['action_index']): a for a in parsed['actions']} if parsed else {}
    for iteration in native['iterations']:
        values, previous = {}, iteration['begin_ns']
        for action in iteration['actions']:
            prefix = 'action_' + str(action['action_index'])
            gap = action['begin_ns']-previous
            require(gap>=0 and action['begin_ns']<=action['end_ns']<=iteration['end_ns'], 'actions overlap or escape iteration')
            values[prefix+'__gap'] = gap
            template[prefix+'__gap'] = dict(reference='cpu_reference', category='inter_action_gap', action_index=action['action_index'])
            parts = action_components(lookup[(iteration['iteration'],action['action_index'])]) if parsed else dict(inclusive=action['end_ns']-action['begin_ns'])
            for part, value in parts.items():
                key = prefix+'__'+part
                reference = ('gpu_reference' if part=='kernel_activity' else
                             'static' if part=='copy_activity_outside_kernel' else 'cpu_reference')
                values[key] = value
                template[key] = dict(reference=reference,category=part,action_index=action['action_index'])
            previous = action['end_ns']
        values['tail'] = iteration['end_ns']-previous
        template['tail'] = dict(reference='cpu_reference',category='tail',action_index=None)
        require(sum(values.values()) == iteration['end_ns']-iteration['begin_ns'], 'whole accounting did not telescope')
        vectors.append(values)
    require(all(v.keys()==vectors[0].keys() for v in vectors), 'physical action layout changed')
    return dict(template=template, means={k:_mean([Fraction(v[k]) for v in vectors]) for k in vectors[0]},
                iterations=len(vectors), observed_ns=[r['end_ns']-r['begin_ns'] for r in native['iterations']])


def _reference(loaded, name, *, before):
    """Independent BEFORE response, with no target-duration-derived rates."""
    raw = loaded['raw'].get(name)
    if raw is None:
        return dict(known=False, issues=['missing reference: '+name], rates={})
    main, issues = loaded['native'], []
    if raw.get('clock') != main.get('clock'):
        issues.append('reference clock or QPC origin differs')
    if raw.get('program_sha256') != main.get('program_sha256'):
        issues.append('reference target program differs')
    expected = 39 if name.endswith('cpu') else 52
    if raw.get('mode') != 'target' or raw.get('target_action_index') != expected:
        issues.append('reference is not the declared exact target action')
    count = len(raw.get('iterations', []))
    span = raw['sampling_end_ns']-raw['sampling_begin_ns']
    if count != raw.get('samples_requested') or count != raw.get('samples_completed') or count==0 or span<=0:
        issues.append('reference batch coverage invalid')
    start, end = main['iterations'][0]['begin_ns'], main['iterations'][-1]['end_ns']
    if before and raw['sampling_end_ns']>start:
        issues.append('BEFORE reference ends after target begins')
    if not before and raw['sampling_begin_ns']<end:
        issues.append('AFTER reference begins before target ends')
    rates = {}
    if not issues:
        rates['completed_reference_response'] = Fraction(count*1_000_000_000,span)
    capture = None
    if name.endswith('gpu'):
        capture = parse_physical_records(raw,program=loaded['program'])
        activity = sum(a['kernel_union_ns'] for a in capture['actions'])
        if capture['capture_qualified'] and activity>0 and not issues:
            rates['kernel_union_reference_response'] = Fraction(count*1_000_000_000,activity)
        else:
            issues.append('reference GPU kernel attribution unknown')
    active = raw.get('batch_cpu',{}).get('main_thread_active_cpu_ns')
    cpu_share = Fraction(active,span) if type(active) is int and 0<=active<=span else None
    return dict(known=bool(rates), issues=issues, rates=rates, process_id=raw['process_id'], samples=count,
                begin_ns=raw['sampling_begin_ns'],end_ns=raw['sampling_end_ns'],
                staleness_ns=start-raw['sampling_end_ns'] if before else raw['sampling_begin_ns']-end,
                main_thread_active_share=cpu_share,
                active_share_issue=None if cpu_share is not None else 'counter absent or exceeds wall interval; not clipped',
                completed_reference_response_scope='Completed exact target GEMM reference invocations / batch wall; includes runtime and materialization',
                kernel_union_reference_response_scope='Completed exact target reference invocations / correlated kernel envelope union; not SM-active service',
                physical_available_capacity_claimed=False)


def _references(loaded, *, before):
    prefix = 'before-' if before else 'after-'
    cpu, gpu = _reference(loaded,prefix+'cpu',before=before), _reference(loaded,prefix+'gpu',before=before)
    return dict(cpu=cpu,gpu=gpu,
                activity_rates=dict(cpu_reference=cpu['rates'].get('completed_reference_response'),
                                    gpu_reference=gpu['rates'].get('kernel_union_reference_response')),
                completed_rates=dict(cpu_reference=cpu['rates'].get('completed_reference_response'),
                                     gpu_reference=gpu['rates'].get('completed_reference_response')))


def _combine(components):
    require(components and all(c['template']==components[0]['template'] for c in components), 'component signatures differ')
    return dict(template=components[0]['template'],means={k:_mean([c['means'][k] for c in components]) for k in components[0]['means']},
                processes=len(components), weighting='Mean per invocation within each process, then equal weight across processes')


def _inclusive_components(component, program):
    means, template = {}, {}
    for name, value in component['means'].items():
        meta = component['template'][name]
        index = meta['action_index']
        if index is None or meta['category']=='inter_action_gap':
            key, reference = name,'cpu_reference'
        else:
            action = program['actions'][index]
            key = 'action_'+str(index)+'__inclusive'
            reference = ('gpu_reference' if action['kind']=='COMPUTE' and action.get('device')=='gpu' else
                         'cpu_reference' if action['kind']=='COMPUTE' else 'static')
        means[key] = means.get(key,Fraction())+value
        template[key] = dict(reference=reference,category='inclusive_response',action_index=index)
    return dict(means=means,template=template)


def _source_code_receipts():
    root=Path(__file__).resolve().parents[1]
    return [_receipt(path) for path in (Path(__file__), root/'profiling/physical_job_index.py',root/'profiling/available_service/physical_records.py',
        root/'profiling/available_service/calibration.py', root/'modeling/llm_infer_model/performance/reference_components.py',
        root/'modeling/llm_infer_model/performance/evaluator.py',root/'modeling/llm_infer_model/performance/graph.py',
        root/'modeling/llm_infer_model/performance/parameters.py')]


def _observer_summary(campaign, manifest, jobs):
    groups, receipts, processes = defaultdict(list), [], []
    bare_components=[]
    for job in jobs:
        if job['split']!='observer':
            continue
        path=campaign/'jobs'/job['id']/'execution.json'
        if not path.exists():
            continue
        loaded=_load_job(campaign,manifest,job,physical=False)
        component=_components(loaded)
        key=job['instrumentation']+('_references' if job['references'] else '')
        groups[key].append(_mean(list(map(Fraction,component['observed_ns']))))
        processes.append(dict(job=job['id'],instrumentation=key,observed=_stats(component['observed_ns'])))
        receipts.extend(loaded['sources'])
        if key=='bare' and job['condition']=='idle':
            bare_components.append(component)
    means={k:_mean(v) for k,v in groups.items()}
    bare=means.get('bare')
    overheads={k:float(v/bare-1) for k,v in means.items() if bare and k!='bare'}
    qualified=(len(groups.get('bare',[]))>=3 and len(overheads)>=2 and all(abs(v)<=.05 for v in overheads.values()))
    return dict(processes=processes,mean_batch_call_ns=means,relative_observer_overheads=overheads,
                threshold=.05,neutrality_qualified=qualified,overhead_subtracted=False), receipts, (_combine(bare_components) if bare_components else None)


def freeze_campaign_profile(campaign_path, *, frozen_at_utc):
    """Read completed PROFILE/observer only. Save returned JSON before heldout."""
    utc(frozen_at_utc)
    campaign=Path(campaign_path).resolve()
    manifest=_read(campaign/'manifest.json')
    require(manifest.get('schema')=='physical-service-campaign-1','unsupported campaign')
    jobs,replacement_sources,replacement_notes=_effective_jobs(campaign,manifest)
    sources=[_receipt(campaign/'manifest.json')]+_source_code_receipts()+replacement_sources
    groups, references, identities, diagnostics = defaultdict(list),[],[],[]
    signature=None
    for job in jobs:
        if job['split']!='profile':
            continue
        loaded=_load_job(campaign,manifest,job)
        require(utc(loaded['execution']['ended_at_utc'])<=utc(frozen_at_utc),'profile ended after freeze')
        this_signature=_signature(loaded['program'])
        signature=this_signature if signature is None else signature
        require(signature==this_signature,'profile policy or graph signature differs')
        component=_components(loaded)
        # Capture issues remain visible; they do not become certified physical work.
        groups[job['condition']].append(component)
        before=_references(loaded,before=True)
        after=_references(loaded,before=False)
        if job['condition']=='idle':
            references.append(before)
        identities.append(dict(job_id=job['id'],process_id=loaded['native']['process_id'],seed=loaded['execution']['program']['seed']))
        diagnostics.append(dict(job_id=job['id'],condition=job['condition'],
            capture_qualified=loaded['parsed']['capture_qualified'],capture_issues=loaded['parsed']['capture_issues'],
            selected_action_issues=loaded['parsed']['selected_action_issues'],before=before,after_audit_only=after,
            wall=_stats(component['observed_ns'])))
        sources.extend(loaded['sources'])
        program=loaded['program']
    require(groups.get('idle') and len(groups['idle'])>=3,'three independent idle profile processes required')
    conditions={k:_combine(v) for k,v in groups.items()}
    reference_rates={variant:{kind:(_mean([r[variant][kind] for r in references])
                         if all(r[variant].get(kind) is not None for r in references) else None)
                    for kind in ('cpu_reference','gpu_reference')}
                    for variant in ('activity_rates','completed_rates')}
    observer,observer_sources,bare=_observer_summary(campaign,manifest,jobs)
    sources.extend(observer_sources)
    frozen=dict(schema=SCHEMA,frozen_at_utc=frozen_at_utc,campaign=str(campaign),
        manifest_sha256=_sha(campaign/'manifest.json'),signature_sha256=signature,
        source_inventory_sha256=manifest['source_inventory_sha256'],
        source_receipts=list({r['path']:r for r in sources}.values()),profile_identities=identities,
        candidates=list(CANDIDATES),conditions=conditions,idle_inclusive=_inclusive_components(conditions['idle'],program),
        profile_before_reference_rates=reference_rates,observer_bare_idle=bare,observer=observer,
        profile_diagnostics=diagnostics,job_replacements=replacement_notes,
        error_protocol=dict(minimum_processes_per_state=3,minimum_calls_per_process=32,
            per_call_median_relative_error_max=.20,per_call_p90_relative_error_max=.20,
            empirical_cv_max=.10,observer_overhead_absolute_max=.05),
        restrictions=dict(heldout_timings_fitted=False,whole_layer_offset_fitted=False,after_references_used_for_prediction=False,
            observer_overhead_subtracted=False,reference_is_neutral_available_capacity=False,
            host_API_wait_is_active_CPU_work=False,hardware_accuracy_qualified=False,
            traced_diagnostic_only=True,reference_staleness_preserved=True))
    return _pack(frozen)


def _predict(frozen, condition, current):
    results={}
    for candidate in frozen['candidates']:
        if candidate=='condition_component_baseline':
            component=frozen['conditions'].get(condition)
        elif candidate=='reference_completed_gpu_candidate':
            component=frozen['idle_inclusive']
        elif candidate=='observer_bare_idle_baseline':
            component=frozen.get('observer_bare_idle')
        else:
            component=frozen['conditions']['idle']
        if component is None:
            results[candidate]=dict(known=False,reason='profile coverage unknown')
            continue
        reference_conditioned=candidate in {'reference_activity_candidate','reference_completed_gpu_candidate'}
        kind='completed_rates' if candidate=='reference_completed_gpu_candidate' else 'activity_rates'
        tasks=tuple(ReferenceComponent(name,exact(value),component['template'][name]['reference'] if reference_conditioned else 'static')
                    for name,value in component['means'].items())
        try:
            prediction=evaluate_reference_components(tasks,
                {k:None if v is None else exact(v) for k,v in frozen['profile_before_reference_rates'][kind].items()},
                current[kind],evidence=('frozen profile components','separate BEFORE reference response'),
                scope='observed_diagnostic reference-conditioned serial component response')
            results[candidate]=dict(known=True,predicted_ns=prediction['execution'].duration_ns,
                component_count=len(tasks),hardware_accuracy_qualified=False)
        except UnknownParameter as error:
            results[candidate]=dict(known=False,reason=str(error),hardware_accuracy_qualified=False)
    return results


def _error_row(predicted, measured):
    return dict(relative_error=abs(predicted-measured)/measured,signed_relative_error=(predicted-measured)/measured)


def evaluate_campaign_heldout(campaign_path, frozen, *, split='heldout'):
    """Use frozen candidates unchanged; retain every main invocation and high CV."""
    require(frozen.get('schema')==SCHEMA,'unsupported frozen diagnostic profile')
    campaign=Path(campaign_path).resolve()
    require(str(campaign)==frozen['campaign'],'campaign identity differs')
    for source in frozen['source_receipts']:
        require(_sha(source['path'])==source['sha256'],'frozen source changed: '+source['path'])
    manifest=_read(campaign/'manifest.json',frozen['manifest_sha256'])
    jobs,replacement_sources,replacement_notes=_effective_jobs(campaign,manifest)
    require(replacement_notes==frozen.get('job_replacements',[]),'replacement index changed after freeze')
    require(split not in {'profile','observer','smoke'},'heldout split must be fresh')
    identities={(r['process_id'],r['seed']) for r in frozen['profile_identities']}
    profile_pids={r[0] for r in identities}
    profile_seeds={r[1] for r in identities}
    rows, metrics, coverage, physical = [],defaultdict(list),defaultdict(list),[]
    for job in jobs:
        if job['split']!=split:
            continue
        loaded=_load_job(campaign,manifest,job,physical=job['instrumentation']=='cupti')
        require(utc(loaded['execution']['started_at_utc'])>utc(frozen['frozen_at_utc']),'heldout began before freeze')
        require(loaded['native']['process_id'] not in profile_pids and loaded['execution']['program']['seed'] not in profile_seeds,
                'heldout PID or seed overlaps profile')
        require(_signature(loaded['program'])==frozen['signature_sha256'],'heldout policy/graph signature differs')
        before=_references(loaded,before=True)
        predictions=_predict(frozen,job['condition'],before)
        # Main call durations enter comparison only, never the reference response.
        measured=[r['end_ns']-r['begin_ns'] for r in loaded['native']['iterations']]
        observed=_stats(measured)
        pid=loaded['native']['process_id']
        coverage[job['condition']].append(pid)
        row=dict(job_id=job['id'],condition=job['condition'],process_id=pid,seed=loaded['execution']['program']['seed'],
            instrumentation=job['instrumentation'],observed_ns=measured,observed=observed,
            before=before,after_audit_only=_references(loaded,before=False),predictions={},
            capture_qualified=False if loaded['parsed'] is None else loaded['parsed']['capture_qualified'],
            numerical_verified=True,reference_model_time_dependence='Batch-constant stale BEFORE response; no instantaneous trajectory inferred')
        for name,prediction in predictions.items():
            if not prediction['known']:
                row['predictions'][name]=prediction
                continue
            pred=prediction['predicted_ns']
            errors=[_error_row(pred,n) for n in measured]
            per_call=_stats([r['relative_error'] for r in errors])
            summary=dict(prediction,per_call_relative_error=per_call,
                signed_relative_bias=statistics.mean(r['signed_relative_error'] for r in errors),
                process_mean_truth_error=_error_row(pred,observed['mean']),
                process_median_truth_error=_error_row(pred,observed['median']))
            row['predictions'][name]=summary
            metrics[(name,job['condition'])].append(dict(process_id=pid,predicted_ns=pred,
                measured=measured,errors=errors,observed=observed,summary=summary))
        if loaded['parsed']:
            comp=_components(loaded)
            totals=defaultdict(Fraction)
            for key,value in comp['means'].items():
                totals[comp['template'][key]['category']]+=value
            physical.append(dict(job_id=job['id'],condition=job['condition'],mean_components_ns=totals,
                capture_issues=loaded['parsed']['capture_issues'],selected_action_issues=loaded['parsed']['selected_action_issues'],
                host_API_device_columns_additive=False,host_API_is_active_CPU_work=False))
        rows.append(row)
    require(rows,'no declared heldout observations')
    summaries=[]
    for (name,state),values in metrics.items():
        errors=[e for r in values for e in r['errors']]
        error_stats=_stats([e['relative_error'] for e in errors])
        process_count=len({r['process_id'] for r in values})
        gates=dict(minimum_processes=process_count>=3,
            minimum_calls=all(len(r['measured'])>=32 for r in values),
            median_error=error_stats['median']<=.20,p90_error=error_stats['p90']<=.20,
            empirical_stability=all(r['observed']['cv'] is not None and r['observed']['cv']<=.10 for r in values),
            observer_neutrality=frozen['observer']['neutrality_qualified'],
            numerical=True,capture=all(r['capture_qualified'] for r in rows if r['condition']==state),
            independent_instantaneous_resource_trajectory=False,
            physical_component_service_identified=False)
        summaries.append(dict(candidate=name,condition=state,processes=process_count,
            per_call_relative_error=error_stats,signed_relative_bias=statistics.mean(e['signed_relative_error'] for e in errors),
            process_mean_truth_relative_error=_stats([r['summary']['process_mean_truth_error']['relative_error'] for r in values]),
            process_median_truth_relative_error=_stats([r['summary']['process_median_truth_error']['relative_error'] for r in values]),
            qualification_gates=gates,hardware_accuracy_qualified=False))
    return _pack(dict(schema='physical-component-diagnostic-evaluation-1',split=split,
        frozen_profile_digest=digest(frozen),candidate_selection_after_heldout=False,
        coverage={state:dict(processes=len(set(pids)),minimum_processes_pass=len(set(pids))>=3) for state,pids in coverage.items()},
        summaries=summaries,runs=rows,physical_breakdown=physical,observer=frozen['observer'],
        hardware_accuracy_qualified=False,scope='Diagnostic response comparison; stale reference probes, CUPTI observer overhead, no availability trajectory certification',
        crossing_metrics=dict(known=False,reason='No independently observed within-call resource transition trajectory; cohort labels are not transition times')))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','evaluate'))
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen',type=Path)
    parser.add_argument('--frozen-at-utc')
    parser.add_argument('--split',default='heldout')
    args=parser.parse_args(argv)
    if args.command=='freeze':
        result=freeze_campaign_profile(args.campaign,frozen_at_utc=args.frozen_at_utc)
    else:
        require(args.frozen is not None,'saved frozen profile required')
        result=evaluate_campaign_heldout(args.campaign,_read(args.frozen),split=args.split)
    with args.output.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(path=str(args.output.resolve()),sha256=_sha(args.output))))


if __name__=='__main__':
    main()
