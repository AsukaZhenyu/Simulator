"""Profile-only isolated action/worker estimates for one predeclared g law.

No total-layer duration is a fitted parameter. Combined requests are never
coefficient calibration cells. Unstable estimates remain present and unqualified;
unknown values remain None. Signed inclusive-response effects are not clipped.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import statistics

from .evidence import (canonical, digest, integer, iter_tokens, require,
    state_for_interval, utc, validate_competitors, validate_record)

FROZEN_SCHEMA = "service-response-frozen-1"
PROTOCOL_KEYS = {"schema", "model_name", "competitors", "capacity", "activity_weight", "backlog_weight",
    "coefficient_domain", "profile_settle_ns", "minimum_independent_runs", "minimum_cell_samples_per_process",
    "within_process_cv_max", "between_process_cv_max", "relative_error_max", "max_step_ns", "max_points",
    "initial_work", "initial_history_evidence", "control_overhead_policy", "proxy_mapping", "combined_state_role",
    "model_selection", "heldout_refit"}


def exact(value):
    if type(value) in (int, Fraction): return Fraction(value)
    require(isinstance(value, dict) and set(value) == {"numerator", "denominator"}, "exact rational required")
    require(type(value["numerator"]) is int, "integer numerator required")
    return Fraction(value["numerator"], integer(value["denominator"], "denominator", 1))


def encoded(value):
    if value is None: return None
    value = exact(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def median(values):
    ordered = sorted(exact(v) for v in values)
    if not ordered: return None
    i = len(ordered)//2
    return ordered[i] if len(ordered)%2 else (ordered[i-1]+ordered[i])/2


def cv(values):
    values = [float(exact(v)) for v in values]
    if not values: return None
    mean = statistics.mean(values)
    return 0.0 if mean == 0 else statistics.pstdev(values)/abs(mean)


def validate_protocol(protocol):
    require(isinstance(protocol,dict) and set(protocol) == PROTOCOL_KEYS, "exact predeclared service-response protocol required")
    require(protocol["schema"] == "service-response-protocol-1" and protocol["model_name"] == "signed_inclusive_response_reservoir_v1", "unsupported predeclared model")
    require(protocol["coefficient_domain"] == "signed_inclusive_response", "primary coefficient domain must be explicit")
    specs = validate_competitors(protocol["competitors"])
    require(exact(protocol["capacity"]) > 0 and exact(protocol["activity_weight"]) >= 0 and exact(protocol["backlog_weight"]) >= 0,
            "invalid declared reservoir constants")
    require(exact(protocol["activity_weight"])+exact(protocol["backlog_weight"]) > 0, "positive declared pressure range required")
    for key, minimum in (("minimum_independent_runs",3),("minimum_cell_samples_per_process",1),
                         ("profile_settle_ns",0),("max_step_ns",1),("max_points",2)):
        integer(protocol[key],key,minimum)
    for key in ("within_process_cv_max","between_process_cv_max","relative_error_max"):
        require(type(protocol[key]) in (int,float) and math.isfinite(protocol[key]) and protocol[key] >= 0, "invalid gate " + key)
    require(set(protocol["initial_work"]) == {s["name"] for s in specs} and
            all(0 <= exact(v) <= exact(protocol["capacity"]) for v in protocol["initial_work"].values()), "explicit bounded initial work required")
    require(isinstance(protocol["initial_history_evidence"],list) and protocol["initial_history_evidence"] and
            all(isinstance(v,str) and v.strip() for v in protocol["initial_history_evidence"]), "explicit initial history evidence required")
    require(protocol["control_overhead_policy"] == "baseline_fixed_ns_unqualified" and
            protocol["proxy_mapping"] == "arrival=drain+capacity/fill_median; drain=capacity/off_tail_median" and
            protocol["combined_state_role"] == "validation_only" and
            protocol["model_selection"] == "predeclared_before_profile_and_heldout" and protocol["heldout_refit"] is False,
            "unsupported/undeclared calibration or model selection policy")
    return protocol


def estimate(by_process, pids, protocol, *, minimum_samples=None):
    minimum_samples = protocol["minimum_cell_samples_per_process"] if minimum_samples is None else minimum_samples
    rows, medians, issues = [], [], []
    for pid in pids:
        values = by_process.get(pid, [])
        center, variation = median(values), cv(values)
        if center is not None: medians.append(center)
        if len(values) < minimum_samples: issues.append(f"process:{pid}:too_few_samples")
        if variation is None or variation > protocol["within_process_cv_max"]: issues.append(f"process:{pid}:within_CV_failed_or_unknown")
        rows.append(dict(process_id=pid, sample_count=len(values), median_ns=encoded(center), within_process_cv=variation))
    if len(pids) < protocol["minimum_independent_runs"]: issues.append("too_few_independent_processes")
    between = cv(medians)
    if between is None or between > protocol["between_process_cv_max"]: issues.append("between_process_CV_failed_or_unknown")
    # Missing PID evidence is unknown, never a statistic silently pooled over a subset.
    value = median(medians) if len(medians) == len(pids) else None
    return dict(value_ns=encoded(value), processes=rows, between_process_cv=between,
                qualified=not issues, issues=issues, estimator="exact median of equally weighted per-process medians; population CV")


def worker_proxies(record, name):
    """Return all fill/stop samples, retaining censored edges as unknown.

    Fill is a completed isolated worker call duration. Stop is the last call
    authorized by the immediately preceding ON request. These do not identify
    a hardware queue or prove runtime-thread quiescence.
    """
    worker = record["workers"][name]
    calls = worker["authorized_calls"]
    fills = []
    for decision,left,right in calls:
        state = state_for_interval(record["calibration_segments"], decision, right)
        if state is not None and state[name] == 1 and sum(state.values()) == 1:
            fills.append(right-left)
    segments = record["segments"]
    previous = {key:0 for key in record["workers"]}
    on_begin, on_isolated, stops = None, False, []
    for i, segment in enumerate(segments):
        state = segment["requests"]
        if state is None:
            previous = None
            on_begin = None
            on_isolated = False
            continue
        begin = segment["begin_ns"]
        if previous is not None and previous[name] == 0 and state[name] == 1:
            on_begin = begin
            on_isolated = sum(state.values()) == 1
        elif state[name] == 1:
            on_isolated = on_isolated and sum(state.values()) == 1
        if previous is not None and previous[name] == 1 and state[name] == 0:
            next_on = next((s["begin_ns"] for s in segments[i+1:] if s["requests"] is not None and s["requests"][name]), None)
            authorized = [row for row in calls if on_begin is not None and on_begin <= row[0] < begin]
            issue, value, chosen = None, None, authorized[-1] if authorized else None
            if not worker["sampling_begin_ns"] <= begin <= worker["sampling_end_ns"]:
                issue = "worker_clock_does_not_cover_off_edge"
            elif chosen is None:
                issue = "no_old_ON_authorized_completion_proof"
            elif next_on is not None and chosen[2] >= next_on:
                issue = "completion_censored_by_next_ON"
            else:
                value = max(0, chosen[2]-begin)
            stops.append(dict(off_ns=begin,next_on_ns=next_on,tail_ns=value,last_authorized_call=chosen,issue=issue,
                              isolated_profile_fit_eligible=on_isolated))
            on_begin = None
            on_isolated = False
        previous = state
    return dict(fill_call_ns=fills, off_edges=stops,
                off_tail_ns=[edge["tail_ns"] for edge in stops if edge["tail_ns"] is not None and edge["isolated_profile_fit_eligible"]],
                physical_queue_identified=False, all_runtime_activity_ends_at_call_completion=False)


def freeze_profile(records, *, protocol, frozen_at_utc, verify_sources=True, allow_historical_diagnostic=False):
    """Build a small frozen artifact from profile actions, never total time."""
    from llm_infer_model.performance.parameters import Parameter, ParameterBundle
    from llm_infer_model.performance.service_response import (ServiceResponseParameter, BackgroundReservoir,
        ServiceInterference, ServiceResponseModel)
    validate_protocol(protocol)
    records = tuple(records)
    require(records, "nonempty profile cohort required")
    cutoff = utc(frozen_at_utc)
    first = records[0]
    pids = [integer(r["process_id"],"profile PID",1) for r in records]
    require(len(set(pids)) == len(pids) and len({r["run_id"] for r in records}) == len(records), "duplicate profile identity")
    names = [s["name"] for s in protocol["competitors"]]
    zero = tuple(0 for _ in names)
    required_states = [zero, *(tuple(int(i==j) for i in range(len(names))) for j in range(len(names)))]
    values = defaultdict(lambda:defaultdict(list))
    workers = {name:dict(fill=defaultdict(list),stop=defaultdict(list),runs=[]) for name in names}
    summaries = []
    for record in records:
        validate_record(record,split="profile",verify_sources=verify_sources)
        require(record["instrumentation"] == "stamped", "profile action timings required")
        require(record.get("service_protocol_sha256") == digest(protocol) or allow_historical_diagnostic,
                "service protocol was not bound before collection; historical outcomes are diagnostic only")
        require(record["context_sha256"] == first["context_sha256"] and record["contract_sha256"] == first["contract_sha256"] and
                record["token_inventory"] == first["token_inventory"] and record["competitors"] == protocol["competitors"], "profile exact context/contract differs")
        require(utc(record["ended_at_utc"]) <= cutoff, "profile collected after freeze")
        for invocation, token, begin, end in iter_tokens(record):
            state = state_for_interval(record["calibration_segments"],begin,end)
            if state is None: continue
            key = tuple(state[name] for name in names)
            if key in required_states: values[(token,key)][record["process_id"]].append(end-begin)
        for name in names:
            proxy = worker_proxies(record,name)
            workers[name]["fill"][record["process_id"]].extend(proxy["fill_call_ns"])
            workers[name]["stop"][record["process_id"]].extend(proxy["off_tail_ns"])
            workers[name]["runs"].append(dict(run_id=record["run_id"],process_id=record["process_id"],**proxy))
        summaries.append({key:record[key] for key in ("run_id","seed","process_id","started_at_utc","ended_at_utc","record_sha256",
            "manifest_sha256","original_normalized_sha256","source_files","source_verified")})
        if record.get('extraction_inputs') is not None:
            summaries[-1]['extraction_inputs']=record['extraction_inputs']
    cells, base_parameters, service_parameters, reservoirs, links, negatives, unsupported = [], [], [], [], [], [], []
    estimates = {}
    refs = tuple("profile:"+r["record_sha256"] for r in records)
    scope = "Inclusive completed action/control response in exact program/hardware/backend context " + first["context_sha256"]
    pressure = exact(protocol["activity_weight"])+exact(protocol["backlog_weight"])
    for item in first["token_inventory"]:
        token = item["token"]
        states = required_states if item["transformed"] else [zero]
        for state in states:
            stat = estimate(values[(token,state)],pids,protocol)
            estimates[(token,state)] = stat
            cells.append(dict(token=token,requested_state=dict(zip(names,state)),transformed=item["transformed"],**stat))
        baseline = exact(estimates[(token,zero)]["value_ns"]) if estimates[(token,zero)]["value_ns"] is not None else None
        if baseline is None:
            unsupported.append(token+":unknown_baseline_response")
        if item["transformed"] and baseline is not None and baseline <= 0:
            unsupported.append(token+":nonpositive_baseline")
            baseline = None
        base_parameters.append(Parameter(token,None if baseline is None else (Fraction(1_000_000_000)/baseline if item["transformed"] else baseline),
            "invocation/s" if item["transformed"] else "ns","profiled",refs,scope))
        if not item["transformed"]: continue
        lower = Fraction(1)
        for name,state in zip(names,required_states[1:]):
            measured = estimates[(token,state)]["value_ns"]
            coefficient = None if baseline is None or measured is None else (exact(measured)/baseline-1)/pressure
            pname = "interference:"+token+":"+name
            service_parameters.append(ServiceResponseParameter(pname,coefficient,"dimensionless","profiled",refs,scope))
            links.append(ServiceInterference(token,name,pname,protocol["coefficient_domain"]))
            if coefficient is None: unsupported.append(pname+":unknown_isolated_response")
            elif coefficient < 0:
                negatives.append(dict(target=token,competitor=name,coefficient=encoded(coefficient),
                    interpretation="empirical inclusive facilitation or unresolved context; not pure hardware contention"))
                lower += coefficient*pressure
        if lower <= 0: unsupported.append(token+":denominator_nonpositive_on_declared_pressure_box")
    proxy_stats = {}
    capacity = exact(protocol["capacity"])
    for name in names:
        group = workers[name]
        fill = estimate(group["fill"],pids,protocol)
        stop = estimate(group["stop"],pids,protocol,minimum_samples=1)
        f = None if fill["value_ns"] is None else exact(fill["value_ns"])
        t = None if stop["value_ns"] is None else exact(stop["value_ns"])
        if f is None or f <= 0 or t is None or t <= 0:
            arrival = drain = None
            unsupported.append(name+":unknown_or_nonpositive_worker_proxy")
        else:
            drain = capacity/t
            arrival = drain+capacity/f
        if f is not None and f > protocol['profile_settle_ns']:
            unsupported.append(name+':declared_settle_shorter_than_assumed_saturation_fill_time')
        if t is not None and t > protocol['profile_settle_ns']:
            unsupported.append(name+':declared_settle_shorter_than_assumed_drain_time')
        prefix="reservoir:"+name+":"
        fields=(('capacity',capacity,'normalized_work','declared'),('arrival',arrival,'normalized_work/ns','profiled'),
            ('drain',drain,'normalized_work/ns','profiled'),('activity_weight',exact(protocol['activity_weight']),'dimensionless','declared'),
            ('backlog_weight',exact(protocol['backlog_weight']),'dimensionless','declared'))
        for field,value,unit,origin in fields:
            service_parameters.append(ServiceResponseParameter(prefix+field,value,unit,origin,refs if origin=='profiled' else ('protocol:'+digest(protocol),),
                "Declared normalized replay-work proxy; not an identified hardware queue or available-service measurement"))
        reservoirs.append(BackgroundReservoir(name,*(prefix+field for field in ('capacity','arrival','drain','activity_weight','backlog_weight'))))
        proxy_stats[name]=dict(fill=fill,off_tail=stop,edge_evidence=group['runs'],
            saturated_after_declared_settle_by_proxy=(None if f is None else f <= protocol['profile_settle_ns']),
            drained_after_declared_settle_by_proxy=(None if t is None else t <= protocol['profile_settle_ns']),
            reservoir_mapping_is_declared=True,physical_queue_identified=False)
    assumptions=("capacity/weights and arrival/drain mapping are declared normalized-work structure, not measured literal queues",
        "isolated per-action signed response coefficients; combined requests are validation-only; no layer-duration parameter",
        "baseline-fixed host/control/gap/tail durations are an unqualified state-independence approximation",
        "one shared causal work history persists across foreground invocations; observed foreground timing is never a state input")
    model=ServiceResponseModel(first['context_sha256'],tuple(service_parameters),tuple(reservoirs),tuple(links),assumptions,False)
    original_cells=[c for c in cells if c['transformed']]
    result=dict(schema=FROZEN_SCHEMA,protocol=protocol,protocol_sha256=digest(protocol),frozen_at_utc=frozen_at_utc,
        context_sha256=first['context_sha256'],context=first['context'],contract_sha256=first['contract_sha256'],
        token_inventory=first['token_inventory'],action_parameters=[v['token'] for v in first['token_inventory']],
        base_bundle=ParameterBundle(tuple(base_parameters)).to_dict(),service_model=model.to_dict(),profile_sources=summaries,
        action_response_cells=cells,worker_proxy_evidence=proxy_stats,negative_coefficients=negatives,
        unsupported_model_issues=unsupported,model_parameters_available=not unsupported,
        original_action_source_qualified=bool(original_cells) and all(c['qualified'] for c in original_cells),
        original_action_failed_cells=sum(not c['qualified'] for c in original_cells),
        worker_proxy_source_qualified=all(s['fill']['qualified'] and s['off_tail']['qualified'] for s in proxy_stats.values()),
        controls_state_independence_qualified=False,physical_mechanism_identified=False,source_qualified=False,
        accuracy_qualified=False,whole_layer_duration_used_for_fit=False,combined_state_used_for_fit=False,
        historical_diagnostic_only=any(r.get('service_protocol_sha256') != digest(protocol) for r in records),
        heldout_used_for_fit=False,all_samples_retained=True)
    result['artifact_sha256']=digest(result)
    return result


def validate_frozen(frozen, *, verify_sources=True):
    require(frozen.get('schema') == FROZEN_SCHEMA and frozen.get('artifact_sha256') == digest({k:v for k,v in frozen.items() if k!='artifact_sha256'}),
            'frozen service artifact digest/schema differs')
    validate_protocol(frozen['protocol'])
    require(frozen['protocol_sha256'] == digest(frozen['protocol']) and not frozen['heldout_used_for_fit'] and
            not frozen['whole_layer_duration_used_for_fit'] and not frozen['combined_state_used_for_fit'], 'freeze protocol/fit scope differs')
    require(len({r['process_id'] for r in frozen['profile_sources']}) == len(frozen['profile_sources']), 'profile identities duplicated')
    if verify_sources:
        from .evidence import SourceAudit,extract_job
        audit=SourceAudit()
        rebuilt_records=[]
        for row in frozen['profile_sources']:
            require(row['source_verified'], 'unverified profile cannot establish source qualification')
            for name,receipt in row['source_files'].items(): audit.hash(name,receipt['sha256'])
            require(row.get('extraction_inputs') is not None,'frozen source recipe required for verified derivation')
            inputs=row['extraction_inputs']
            rebuilt=extract_job(inputs['manifest_path'],audit.json(inputs['execution_path']),
                                competitors=frozen['protocol']['competitors'],verify_sources=True)
            require(rebuilt['record_sha256']==row['record_sha256'],'frozen profile extraction does not reconstruct')
            rebuilt_records.append(rebuilt)
        reconstructed=freeze_profile(rebuilt_records,protocol=frozen['protocol'],frozen_at_utc=frozen['frozen_at_utc'],
            verify_sources=False,allow_historical_diagnostic=frozen.get('historical_diagnostic_only',False))
        require(canonical(reconstructed)==canonical(frozen),'frozen profile parameters/statistics do not reconstruct')
    return frozen
