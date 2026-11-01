"""Independent descriptive validation; execution and g belong to modeling."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import statistics

from .evidence import digest, require, state_for_interval, utc, validate_record
from .calibration import encoded, exact, median, validate_frozen


def make_barrier(frozen, *, created_at_utc, frozen_file_sha256, manifest_sha256, verify_sources=True):
    validate_frozen(frozen,verify_sources=verify_sources)
    require(utc(frozen['frozen_at_utc']) <= utc(created_at_utc), 'barrier precedes frozen parameters')
    require(all(r['manifest_sha256'] == manifest_sha256 for r in frozen['profile_sources']), 'barrier manifest differs from profile')
    require(all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef')
                for v in (frozen_file_sha256,manifest_sha256)), 'exact barrier file digests required')
    result=dict(schema='service-response-heldout-barrier-1',created_at_utc=created_at_utc,
        frozen_artifact_sha256=frozen['artifact_sha256'],frozen_file_sha256=frozen_file_sha256,
        manifest_sha256=manifest_sha256,protocol_sha256=frozen['protocol_sha256'],
        profile_record_sha256s=sorted(r['record_sha256'] for r in frozen['profile_sources']))
    result['barrier_sha256']=digest(result)
    return result


def quantile(values,p):
    ordered=sorted(values)
    if not ordered:return None
    x=Fraction(len(ordered)-1)*p; i=x.numerator//x.denominator
    return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(x-i)


def summary(rows,run_ids):
    processes=[];errors=[]
    for run_id,pid in run_ids:
        selected=[r for r in rows if r['run_id']==run_id]
        vals=[r['absolute_relative_error'] for r in selected if r['absolute_relative_error'] is not None]
        errors.extend(vals)
        walls=[r['observed_ns'] for r in selected]
        processes.append(dict(run_id=run_id,process_id=pid,sample_count=len(selected),predictable_count=len(vals),
            median_absolute_relative_error=None if not vals else statistics.median(vals),
            p90_absolute_relative_error=None if not vals else float(quantile(vals,Fraction(9,10))),
            worst_absolute_relative_error=max(vals,default=None),observed_median_ns=encoded(median(walls))))
    centers=[p['median_absolute_relative_error'] for p in processes if p['median_absolute_relative_error'] is not None]
    return dict(sample_count=len(rows),predictable_count=len(errors),complete_prediction_coverage=len(errors)==len(rows),
        process_balanced_median_absolute_relative_error=statistics.median(centers) if centers else None,
        worst_process_median_absolute_relative_error=max(centers,default=None),
        pooled_p90_absolute_relative_error=None if not errors else float(quantile(errors,Fraction(9,10))),
        worst_absolute_relative_error=max(errors,default=None),processes=processes)


def request_trajectory(record,protocol):
    """Use only source-bound exogenous requests and declared initial work.

    Stop at the first unknown/gap instead of inventing an idle state. Later
    observations outside that finite support remain unknown in validation.
    """
    from llm_infer_model.performance.service_response import ServiceRequestPoint,ServiceRequestTrajectory
    segments=record['segments'];require(segments and segments[0]['begin_ns']==0 and segments[0]['requests'] is not None,
                                        'initial known request segment required')
    names={s['name'] for s in protocol['competitors']};cursor=0;points=[];previous=None
    for segment in segments:
        if segment['begin_ns']!=cursor or segment['requests'] is None:break
        require(segment['host_memory_bytes']==segment['device_memory_bytes']==0,
                'this frozen protocol supports only explicitly fixed zero allocation holds')
        state=segment['requests'];require(set(state)==names,'request coordinate inventory differs')
        if previous is not None and state!=previous:points.append(ServiceRequestPoint(segment['begin_ns'],state))
        previous=state;cursor=segment['end_ns']
    require(cursor>0,'no supported request horizon')
    return ServiceRequestTrajectory(segments[0]['requests'],{k:exact(v) for k,v in protocol['initial_work'].items()},
        tuple(points),'predeclared_scenario',tuple(protocol['initial_history_evidence'])+('run:'+record['record_sha256'],),
        record['clock']['clock_id'],'planned_go_start_v1'),cursor


def classification(record,begin,end):
    segments=record['segments'];state=state_for_interval(segments,begin,end)
    category='request_constant' if state is not None else 'request_crossing'
    overlaps=[s for s in segments if s['begin_ns']<end and begin<s['end_ns']]
    covered=sum(max(0,min(end,s['end_ns'])-max(begin,s['begin_ns'])) for s in overlaps if s['requests'] is not None)
    if covered != end-begin:category='unknown_request_or_outside'
    entry=next((s for s in segments if s['begin_ns']<=begin<s['end_ns']),None)
    age=None if entry is None else begin-entry['begin_ns']
    prior=None
    if entry is not None:
        index=segments.index(entry)
        for preceding in reversed(segments[:index]):
            if preceding['requests'] is None:break
            if any(preceding['requests'].values()):prior=preceding['requests'];break
    recovering=state is not None and not any(state.values()) and prior is not None
    return dict(category=category,requested_state=state,entry_request_age_ns=age,
                recovery_after_requested_load=recovering,recovery_predecessor=prior if recovering else None)


def evaluate_frozen(frozen,records,*,barrier,verify_sources=True,numerical_comparison_max_step_ns=None):
    """Build one shared g per run, then delegate each serial replay to modeling.

    Observed invocation starts are explicit conditioning, never model fitting.
    Missing/unsupported parameters and horizon failures retain unknown rows.
    """
    from llm_infer_model.performance.parameters import ParameterBundle
    from llm_infer_model.performance.service_response import ServiceResponseModel,build_service_response
    validate_frozen(frozen,verify_sources=verify_sources)
    require(barrier.get('schema')=='service-response-heldout-barrier-1' and barrier.get('barrier_sha256')==digest({k:v for k,v in barrier.items() if k!='barrier_sha256'}), 'invalid heldout barrier')
    require(barrier['frozen_artifact_sha256']==frozen['artifact_sha256'] and barrier['protocol_sha256']==frozen['protocol_sha256'] and
            barrier['profile_record_sha256s']==sorted(r['record_sha256'] for r in frozen['profile_sources']), 'barrier model/profile inventory differs')
    records=tuple(records);require(records,'nonempty fresh heldout cohort required')
    profile_pids={r['process_id'] for r in frozen['profile_sources']};profile_seeds={r['seed'] for r in frozen['profile_sources']}
    require(len({r['process_id'] for r in records})==len(records) and len({r['run_id'] for r in records})==len(records), 'duplicate heldout identity')
    protocol=frozen['protocol'];run_ids=[];predictions=[];trajectories=[]
    if numerical_comparison_max_step_ns is not None:
        require(type(numerical_comparison_max_step_ns) is int and numerical_comparison_max_step_ns>0,
                'numerical comparison step must be an explicit positive integer')
    for record in records:
        validate_record(record,split='heldout',verify_sources=verify_sources)
        require(record['process_id'] not in profile_pids and record['seed'] not in profile_seeds,'profile identity reused as heldout')
        require(utc(record['started_at_utc'])>utc(barrier['created_at_utc']), 'heldout was not collected after frozen barrier')
        require(record['context_sha256']==frozen['context_sha256'] and record['contract_sha256']==frozen['contract_sha256'] and
                record['service_protocol_sha256']==frozen['protocol_sha256'] and record['manifest_sha256']==barrier['manifest_sha256'], 'heldout exact context/protocol differs')
        require(record['token_inventory']==frozen['token_inventory'], 'heldout action occurrence inventory differs')
        run_ids.append((record['run_id'],record['process_id']))
        result,comparison,problem,comparison_problem=None,None,None,None
        try:
            require(not frozen['unsupported_model_issues'],'unsupported frozen parameters: '+str(frozen['unsupported_model_issues']))
            trajectory,horizon=request_trajectory(record,protocol)
            result=build_service_response(ServiceResponseModel.from_dict(frozen['service_model']),
                ParameterBundle.from_dict(frozen['base_bundle']),trajectory,end_ns=horizon,
                max_step_ns=protocol['max_step_ns'],max_points=protocol['max_points'])
            trajectories.append(dict(run_id=record['run_id'],valid_until_ns=horizon,point_count=len(result.timeline.points),
                request_trajectory=trajectory.to_dict(),model_result_sha256=digest(result.to_dict()),
                primary_max_step_ns=protocol['max_step_ns'],
                primary_integrated_service_error_bound_work={k:encoded(v) for k,v in result.integrated_service_error_bound_work.items()},
                work_history_reset_per_foreground_invocation=False,accuracy_qualified=False))
            if numerical_comparison_max_step_ns is not None:
                try:
                    comparison=build_service_response(ServiceResponseModel.from_dict(frozen['service_model']),
                        ParameterBundle.from_dict(frozen['base_bundle']),trajectory,end_ns=horizon,
                        max_step_ns=numerical_comparison_max_step_ns,max_points=protocol['max_points'])
                    trajectories[-1]['numerical_comparison']=dict(max_step_ns=numerical_comparison_max_step_ns,
                        point_count=len(comparison.timeline.points),model_result_sha256=digest(comparison.to_dict()),
                        integrated_service_error_bound_work={k:encoded(v) for k,v in comparison.integrated_service_error_bound_work.items()})
                except ValueError as error:
                    comparison_problem=f'{type(error).__name__}: {error}'
                    trajectories[-1]['numerical_comparison']=dict(max_step_ns=numerical_comparison_max_step_ns,problem=comparison_problem)
        except ValueError as error:
            problem=f'{type(error).__name__}: {error}'
            trajectories.append(dict(run_id=record['run_id'],model_problem=problem))
        for item in record['invocations']:
            begin,end=item['begin_ns'],item['end_ns'];observed=end-begin
            predicted=None;issue=problem
            if result is not None:
                try:
                    prediction=result.evaluate_serial_response(tuple(frozen['action_parameters']),start_ns=begin)
                    predicted=prediction.duration_ns
                except ValueError as error:issue=f'{type(error).__name__}: {error}'
            error=None if predicted is None else float(abs(Fraction(predicted)-observed)/observed)
            compared=None;compared_issue=comparison_problem
            if comparison is not None:
                try: compared=comparison.evaluate_serial_response(tuple(frozen['action_parameters']),start_ns=begin).duration_ns
                except ValueError as comparison_error: compared_issue=f'{type(comparison_error).__name__}: {comparison_error}'
            predictions.append(dict(run_id=record['run_id'],process_id=record['process_id'],observation_id=item['observation_id'],
                begin_ns=begin,end_ns=end,observed_ns=observed,predicted_ns=predicted,
                absolute_relative_error=error,model_problem=issue,
                numerical_comparison_predicted_ns=compared,numerical_comparison_problem=compared_issue,
                numerical_absolute_prediction_delta_ns=None if compared is None or predicted is None else abs(compared-predicted),
                **classification(record,begin,end)))
    buckets={name:summary([r for r in predictions if r['category']==name],run_ids)
             for name in ('request_constant','request_crossing','unknown_request_or_outside')}
    names=[s['name'] for s in protocol['competitors']]
    state_keys=sorted({tuple(r['requested_state'][n] for n in names) for r in predictions if r['requested_state'] is not None})
    by_state=[dict(requested_state=dict(zip(names,key)),**summary([r for r in predictions if r['requested_state']==dict(zip(names,key))],run_ids)) for key in state_keys]
    recovery_rows=[r for r in predictions if r['recovery_after_requested_load']]
    age_ranges=[(0,50_000_000),(50_000_000,150_000_000),(150_000_000,500_000_000),(500_000_000,1_000_000_000),(1_000_000_000,None)]
    recoveries=[dict(entry_request_age_range_ns=[a,b],**summary([r for r in recovery_rows if r['entry_request_age_ns'] is not None and
        r['entry_request_age_ns']>=a and (b is None or r['entry_request_age_ns']<b)],run_ids)) for a,b in age_ranges]
    overall=summary(predictions,run_ids)
    deltas=[p['numerical_absolute_prediction_delta_ns'] for p in predictions if p['numerical_absolute_prediction_delta_ns'] is not None]
    result=dict(schema='service-response-evaluation-1',chosen_model=protocol['model_name'],primary_summary=overall,
        buckets=buckets,by_requested_state=by_state,recovery=summary(recovery_rows,run_ids),recovery_by_age=recoveries,
        independent_heldout_processes=len(records),minimum_independent_process_count_met=len(records)>=protocol['minimum_independent_runs'],
        frozen_artifact_sha256=frozen['artifact_sha256'],barrier=barrier,
        heldout_record_sha256s=[r['record_sha256'] for r in records],g_trajectories=trajectories,predictions=predictions,
        original_action_source_qualified=frozen['original_action_source_qualified'],original_action_failed_cells=frozen['original_action_failed_cells'],
        worker_proxy_source_qualified=frozen['worker_proxy_source_qualified'],negative_coefficients=frozen['negative_coefficients'],
        unsupported_model_issues=frozen['unsupported_model_issues'],controls_state_independence_qualified=False,
        actual_hardware_service_transfer_identified=False,accuracy_qualified=False,accepted=False,
        median_error_within_threshold=overall['process_balanced_median_absolute_relative_error'] is not None and
            overall['process_balanced_median_absolute_relative_error']<=protocol['relative_error_max'],
        heldout_used_for_fit=False,whole_layer_duration_used_for_fit=False,work_history_reset_per_invocation=False,
        total_telemetry_used_as_predictor=False,all_observations_retained=True,
        qualification_issues=['normalized replay-work proxies do not identify actual hardware queues/service',
            'baseline-fixed control overhead state independence unestablished','stamped-profile/bare-heldout instrumentation neutrality unestablished'],
        scope='Conditional on observed invocation starts; complete host-action wall responses, not isolated physical service',
        request_constant_is_physically_settled=False)
    result['numerical_step_sensitivity']=dict(primary_max_step_ns=protocol['max_step_ns'],comparison_max_step_ns=numerical_comparison_max_step_ns,
        paired_prediction_count=len(deltas),median_absolute_prediction_delta_ns=encoded(median(deltas)),
        p90_absolute_prediction_delta_ns=encoded(quantile(deltas,Fraction(9,10))),maximum_absolute_prediction_delta_ns=max(deltas,default=None),
        primary_model_or_parameters_selected_from_comparison=False,
        bound_scope='Service discretization only; not a measured calibration-error or whole-graph latency guarantee')
    result['evaluation_sha256']=digest(result)
    return result
