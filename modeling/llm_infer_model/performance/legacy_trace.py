"""Frozen host fork/join execution model, without trace extraction or fitting.

Host-completed interval prices are explicit inputs. This predicts host events;
no physical kernel overlap, capacity or instrumentation-subtraction claim is
introduced by relocating the existing deterministic calculation.
"""
from __future__ import annotations

import math

SCHEMA = "profile-inchain-fork-join-costs-0.1"


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
