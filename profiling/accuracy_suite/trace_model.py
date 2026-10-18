"""Profile-only host-observed stage/gap model with explicit fork/join timing.

Input traces bracket completed work on host threads, not device kernels. This
model keeps instrumentation costs rather than silently subtracting them. It
predicts the separate uninstrumented path for comparison; trace-vs-uninstrumented
differences must be disclosed. No strategy/shape extrapolation or fitting to
heldout totals is performed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import statistics

SCHEMA = "profile-inchain-fork-join-costs-0.1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _summary(values):
    mean = statistics.mean(values)
    return dict(count=len(values), median_ns=statistics.median(values), mean_ns=mean,
                cv=statistics.pstdev(values)/mean if mean else 0,
                min_ns=min(values), max_ns=max(values))


def _parts(trace):
    if trace.get("begin_ns") != 0 or type(trace.get("end_ns")) is not int or trace["end_ns"] <= 0:
        raise ValueError("relative positive trace bounds required")
    spans = trace.get("stages")
    if not isinstance(spans, list) or not spans:
        raise ValueError("actual stage intervals required")
    threads, keys = {"main": [], "dispatch": []}, set()
    for span in spans:
        name, thread = span.get("name"), span.get("thread")
        if thread not in threads or not isinstance(name, str) or not name or (thread,name) in keys:
            raise ValueError("unique stage/thread names required")
        keys.add((thread,name))
        begin, end = span.get("begin_ns"), span.get("end_ns")
        if any(type(v) is not int for v in (begin,end)) or not 0 <= begin <= end <= trace["end_ns"]:
            raise ValueError("stage outside completed trace")
        threads[thread].append(dict(name=name, begin_ns=begin, end_ns=end))
    for values in threads.values():
        values.sort(key=lambda s:(s["begin_ns"],s["end_ns"]))
        if any(a["end_ns"] > b["begin_ns"] for a,b in zip(values,values[1:])):
            raise ValueError("same-thread completed stages cannot overlap")
    if not threads["main"]:
        raise ValueError("main-thread program required")
    main, gpu = threads["main"], threads["dispatch"]
    main_index = {s["name"]:i for i,s in enumerate(main)}
    if gpu:
        if "dispatch" not in main_index or "join" not in main_index or main_index["dispatch"] >= main_index["join"]:
            raise ValueError("parallel stages require ordered dispatch and join")
        launch, join = main[main_index["dispatch"]], main[main_index["join"]]
        if gpu[0]["begin_ns"] < launch["begin_ns"] or gpu[-1]["end_ns"] > join["end_ns"]:
            raise ValueError("worker execution escapes dispatch/join boundary")
    elif "dispatch" in main_index or "join" in main_index:
        raise ValueError("fork/join without a worker branch")
    return threads


def _measure(trace):
    threads = _parts(trace)
    main, worker = threads["main"], threads["dispatch"]
    values = {}
    for thread, spans in threads.items():
        previous = 0
        for index, span in enumerate(spans):
            prefix = thread+":"+span["name"]
            if thread == "main" or index:
                values[prefix+":gap_before"] = span["begin_ns"] - previous
            if thread == "main" and span["name"] == "join":
                values[prefix+":completion_overhead"] = span["end_ns"] - max(span["begin_ns"], worker[-1]["end_ns"])
            else:
                values[prefix+":duration"] = span["end_ns"] - span["begin_ns"]
            previous = span["end_ns"]
    values["main:tail"] = trace["end_ns"] - main[-1]["end_ns"]
    if worker:
        dispatch = next(s for s in main if s["name"] == "dispatch")
        values["worker_start_after_dispatch_begin"] = worker[0]["begin_ns"] - dispatch["begin_ns"]
    return threads, values


def freeze(profile_records):
    """Freeze per-stage/gap medians from profile trace records only.

    Each record supplies id, split, status, case, raw_sha256 and chain_traces.
    A caller must validate original native values and provenance first. Complete
    chain totals are not coefficients or lookup prices, but profile joint trace
    intervals ARE calibration sources, explicitly disclosed in the output.
    """
    rows = list(profile_records)
    if not rows or any(row.get("split") != "profile" for row in rows):
        raise ValueError("profile-only trace records required")
    groups, cases, sources, seen = defaultdict(list), {}, [], set()
    for row in rows:
        rid = row.get("id")
        if row.get("status") != "measured" or not isinstance(rid,str) or not rid or rid in seen:
            raise ValueError("successful unique profile process records required")
        seen.add(rid)
        sha = row.get("raw_sha256")
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("raw profile SHA256 required")
        case = row.get("case")
        if not isinstance(case,dict) or not isinstance(case.get("id"),str):
            raise ValueError("explicit exact case geometry required")
        cid=case["id"]
        if cid in cases and cases[cid] != case:
            raise ValueError("conflicting geometry for case")
        cases[cid]=case
        traces = row.get("chain_traces")
        if not isinstance(traces,dict) or not traces:
            raise ValueError("profile chain trace inventory required")
        sources.append(dict(row_id=rid,case_id=cid,raw_sha256=sha))
        for strategy, observations in traces.items():
            if not isinstance(strategy,str) or not strategy or not isinstance(observations,list) or len(observations)<3:
                raise ValueError("at least three actual diagnostic invocations required")
            groups[(cid,strategy)].extend(observations)
    paths=[]
    for (cid,strategy), traces in sorted(groups.items()):
        collected=defaultdict(list)
        template=None
        for trace in traces:
            parts, values = _measure(trace)
            names={thread:[s["name"] for s in spans] for thread,spans in parts.items()}
            if template is not None and names != template:
                raise ValueError("within-path trace program changed")
            template=names
            for name,value in values.items():collected[name].append(value)
        paths.append(dict(case_id=cid,strategy=strategy,program=template,
            costs={name:_summary(values) for name,values in sorted(collected.items())},
            process_count=sum(row["case"]["id"]==cid and strategy in row["chain_traces"] for row in rows),
            diagnostic_invocations=len(traces)))
    return dict(schema=SCHEMA,case_geometry=cases,paths=paths,source_rows=sources,
        source_rows_sha256=_hash(sources), trained_on="profile_only",heldout_used_for_fitting=False,
        profile_joint_trace_intervals_used_for_calibration=True,
        profile_complete_chain_totals_used_as_prices=False,
        complete_chain_targets_used_for_offsets=False,intercept="none",
        timing_unit="host_completed_work_nanoseconds",instrumentation_overhead_subtracted=False,
        timeline_semantics="sequential main and worker resources; start-to-start launch edge; wait completion depends on both branches",
        physical_kernel_overlap_proved=False,physical_capacity_validated=False,
        scope="exact profiled strategy/geometry only; no unseen strategy or shape prediction")


def predict(model, case_id, strategy):
    """Reconstruct explicit host stage/gap timing; never add join wait twice."""
    if (model.get("schema")!=SCHEMA or model.get("trained_on")!="profile_only"
            or model.get("heldout_used_for_fitting") is not False
            or model.get("timing_unit")!="host_completed_work_nanoseconds"
            or model.get("instrumentation_overhead_subtracted") is not False):
        raise ValueError("unsupported in-chain trace cost model")
    selected=[x for x in model.get("paths",[]) if x["case_id"]==case_id and x["strategy"]==strategy]
    if len(selected)!=1:raise ValueError("exact strategy/geometry trace prices unavailable")
    path=selected[0]
    def cost(key):
        value=path["costs"][key]["median_ns"]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError("finite nonnegative profile stage/gap cost required")
        return value
    main_time,worker_end,events=0.0,None,[]
    for name in path["program"]["main"]:
        main_time+=cost("main:"+name+":gap_before")
        start=main_time
        if name=="join":
            if worker_end is None:raise ValueError("join without an executed worker")
            main_time=max(main_time,worker_end)+cost("main:join:completion_overhead")
        else:
            main_time+=cost("main:"+name+":duration")
        events.append(dict(thread="main",name=name,begin_ns=start,end_ns=main_time))
        if name=="dispatch":
            worker_end=start+cost("worker_start_after_dispatch_begin")
            for index,stage in enumerate(path["program"]["dispatch"]):
                if index:worker_end+=cost("dispatch:"+stage+":gap_before")
                begin=worker_end;worker_end+=cost("dispatch:"+stage+":duration")
                events.append(dict(thread="dispatch",name=stage,begin_ns=begin,end_ns=worker_end))
    main_time+=cost("main:tail")
    if not math.isfinite(main_time) or main_time<=0:raise ValueError("nonpositive or nonfinite predicted complete path")
    return dict(predicted_ns=main_time,modeled_host_events=events,
        physical_kernel_timeline=False,instrumentation_overhead_subtracted=False,
        profile_joint_trace_intervals_used_for_calibration=True,
        arbitrary_bundle_composition_validated=False)
