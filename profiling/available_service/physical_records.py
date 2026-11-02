"""Exact CUPTI/external-correlation attribution for physical-stage-native-1.

This is descriptive metrology, not a duration predictor or available-resource
estimator. Device activity records describe execution envelopes; they do not
prove SM-active, preemption-free service. Host/API/device columns overlap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .calibration import canonical, digest, require

SCHEMA = "physical-stage-attribution-1"
KINDS = {"runtime_api", "driver_api", "kernel", "memcpy", "external_correlation"}


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def _span(row, name, *, zero=False):
    left, right = _integer(row.get("begin_ns"), name + " begin"), _integer(row.get("end_ns"), name + " end")
    require(left <= right if zero else 0 < left < right, name + " interval is invalid")
    return left, right


def interval_union(rows):
    """Merge valid explicit intervals without adding nested/overlapping work."""
    merged = []
    for left, right in sorted((r["begin_ns"], r["end_ns"]) for r in rows):
        require(type(left) is int and type(right) is int and 0 <= left < right, "invalid union interval")
        if not merged or merged[-1][1] < left:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [dict(begin_ns=left, end_ns=right) for left, right in merged]


def interval_length(rows):
    return sum(r["end_ns"] - r["begin_ns"] for r in interval_union(rows))


def _descriptive_gaps(host, device):
    begin, end = host
    if not device:
        return dict(before_first_device_ns=None, between_device_intervals_ns=None,
                    after_last_device_ns=None, host_without_device_activity_ns=end-begin,
                    device_union_within_host=False)
    if any(r["begin_ns"] < begin or r["end_ns"] > end for r in device):
        return dict(before_first_device_ns=None, between_device_intervals_ns=None,
                    after_last_device_ns=None, host_without_device_activity_ns=None,
                    device_union_within_host=False)
    return dict(before_first_device_ns=device[0]["begin_ns"]-begin,
                between_device_intervals_ns=sum(right["begin_ns"]-left["end_ns"] for left,right in zip(device,device[1:])),
                after_last_device_ns=end-device[-1]["end_ns"],
                host_without_device_activity_ns=end-begin-interval_length(device),
                device_union_within_host=True)


def parse_physical_records(native, *, program=None):
    """Keep every record and attribute only exact CUSTOM external correlations.

    Invalid times, orphan records, foreign identities and conflicting mappings
    remain visible. They are never attached to an action by timestamp proximity.
    ``capture_qualified`` covers selected-action capture integrity only, not
    numerical correctness, observer neutrality or physical-service calibration.
    """
    require(isinstance(native, dict) and native.get("schema") == "physical-stage-native-1", "unsupported physical native schema")
    clock = native.get("clock")
    require(isinstance(clock, dict) and clock.get("clock_id") == "windows_qpc_floor_ns" and
            clock.get("clock_origin") == "absolute_qpc_zero", "physical native QPC clock/origin differs")
    frequency = _integer(clock.get("frequency_hz"), "native QPC frequency", 1)
    pid, tid = _integer(native.get("process_id"), "native PID", 1), _integer(native.get("main_thread_id"), "native TID", 1)
    sampling = _span(dict(begin_ns=native.get("sampling_begin_ns"),end_ns=native.get("sampling_end_ns")), "sampling")
    inventory = native.get("actions")
    require(isinstance(inventory,list) and inventory, "native original action inventory required")
    if program is not None:
        require(isinstance(program,dict) and program.get("program_sha256") == native.get("program_sha256") and
                program.get("actions") == inventory, "program/native action identity differs")
        require(program["program_sha256"] == digest({k:v for k,v in program.items() if k != "program_sha256"}),
                "program semantic digest differs")
    require(native.get("mode") in {"full","target"} and native.get("instrumentation") in {"cupti","bare"}, "invalid native execution mode")
    rows = native.get("iterations")
    require(isinstance(rows,list) and rows and len(rows) == native.get("samples_completed") == native.get("samples_requested"),
            "native complete invocation inventory differs")
    actions, invocation_rows, by_external = [], [], {}
    previous_end = sampling[0]
    for ordinal, invocation in enumerate(rows):
        require(invocation.get("iteration") == ordinal, "invocation indices differ")
        left,right = _span(invocation,"invocation",zero=True)
        require(previous_end <= left <= right <= sampling[1], "invocations overlap or leave sampling envelope")
        previous_end=right
        stages=invocation.get("actions")
        expected_indices=list(range(len(inventory))) if native["mode"]=="full" else [native.get("target_action_index")]
        require(isinstance(stages,list) and [r.get("action_index") for r in stages]==expected_indices, "invocation original action indices differ")
        cursor=left;gaps=[]
        for stage in stages:
            begin,end=_span(stage,"action",zero=True)
            index=_integer(stage.get("action_index"),"original action index")
            require(index<len(inventory) and cursor <= begin <= end <= right,"action overlap or invalid original index")
            expected=inventory[index]
            require(stage.get("kind")==expected.get("kind") and stage.get("target_id")==expected.get("target_id") and
                    stage.get("device")==expected.get("device"), "action metadata differs from original inventory")
            external=_integer(stage.get("external_id"),"action external ID",1)
            require(external not in by_external,"duplicate action external ID")
            row=dict(iteration=ordinal,action_index=index,external_id=external,kind=stage["kind"],
                target_id=stage.get("target_id"),device=stage.get("device"),begin_ns=begin,end_ns=end,
                host_span_ns=end-begin,gap_before_ns=begin-cursor,record_indices=defaultdict(list),attribution_issues=[])
            by_external[external]=row;actions.append(row);gaps.append(begin-cursor);cursor=end
        invocation_rows.append(dict(iteration=ordinal,begin_ns=left,end_ns=right,duration_ns=right-left,
            gap_before_actions_ns=gaps,tail_ns=right-cursor,
            all_actions_and_disjoint_gaps_telescope=sum(gaps)+sum(s["end_ns"]-s["begin_ns"] for s in stages)+right-cursor==right-left))
    capture=native.get("cupti")
    require(isinstance(capture,dict) and capture.get("schema")=="physical-stage-cupti-1","unsupported CUPTI capture schema")
    enabled=native["instrumentation"]=="cupti"
    raw=capture.get("records")
    require(isinstance(raw,list),"CUPTI record array required")
    issues=[]
    if enabled:
        require((capture.get("clock_id"),capture.get("clock_origin"),capture.get("qpc_frequency_hz"))==
                (clock["clock_id"],clock["clock_origin"],frequency),"CUPTI/native clocks differ")
        capture_span=_span(dict(begin_ns=capture.get("capture_begin_ns"),end_ns=capture.get("capture_end_ns")),"CUPTI capture")
        require(capture_span[0] <= sampling[0] < sampling[1] <= capture_span[1],"sampling lies outside CUPTI capture")
        for key in ("cupti_dropped_records","local_record_overflow"):
            _integer(capture.get(key),key)
            if capture[key]:issues.append(key)
        require(isinstance(capture.get("callback_errors"),list),"callback error inventory required")
        if capture["callback_errors"]:issues.append("CUPTI_callback_errors")
        for key,expected in (("timestamp_callback_failed",False),("timestamp_callback_registered_before_activity_enable",True),
                             ("all_requested_records_retained",True)):
            require(type(capture.get(key)) is bool,key+" must be explicit boolean")
            if capture[key] != expected:issues.append(key)
        external_kind=_integer(capture.get("external_correlation_kind"),"external correlation kind")
    else:
        require(capture.get("enabled") is False and not raw,"bare mode cannot contain enabled CUPTI records")
        external_kind=None;issues.append("CUPTI_disabled")
    mappings=defaultdict(set);mapping_rows=defaultdict(list);unassigned=[];invalid_times=[];unknown_kinds=[]
    for index,record in enumerate(raw):
        require(isinstance(record,dict),"CUPTI record must be an object")
        if record.get("kind") not in KINDS:
            unknown_kinds.append(dict(record_index=index,record=record,reason="unsupported_activity_kind"));continue
        if record["kind"]!="external_correlation":continue
        correlation=record.get("correlation_id");external=record.get("external_id")
        if not(type(correlation) is int and correlation>0 and type(external) is int and external>0 and record.get("external_kind")==external_kind):
            unassigned.append(dict(record_index=index,record=record,reason="invalid_or_foreign_external_correlation"));continue
        mappings[correlation].add(external);mapping_rows[correlation].append(index)
    conflicts={str(key):sorted(value) for key,value in mappings.items() if len(value)!=1}
    if conflicts:issues.append("conflicting_external_correlation")
    selected_record_problems=[];nonzero_api_returns=[]
    for index,record in enumerate(raw):
        kind=record.get("kind")
        if kind not in KINDS or kind=="external_correlation":continue
        correlation=record.get("correlation_id")
        values=mappings.get(correlation,set()) if type(correlation) is int else set()
        owner=by_external.get(next(iter(values))) if len(values)==1 else None
        reason=None
        if not values:reason="no_exact_external_correlation"
        elif len(values)>1:reason="ambiguous_external_correlation"
        elif owner is None:reason="external_action_unknown"
        try:
            begin,end=_span(record,"CUPTI record")
            require(record.get("timestamp_valid") is True,"CUPTI timestamp validity false or unknown")
            valid=True
        except ValueError as error:
            valid=False
            invalid_times.append(dict(record_index=index,record=record,reason=str(error)))
        if owner is None:
            unassigned.append(dict(record_index=index,record=record,reason=reason))
            continue
        owner["record_indices"][kind].append(index)
        if not valid:
            owner["attribution_issues"].append("invalid_time_record:"+str(index));continue
        if not(owner["begin_ns"]<=begin<end<=owner["end_ns"]):
            owner["attribution_issues"].append("record_outside_host_action:"+str(index))
        if kind in {"runtime_api","driver_api"}:
            if record.get("process_id")!=pid or record.get("thread_id")!=tid:
                owner["attribution_issues"].append("foreign_API_process_or_thread:"+str(index))
            if record.get("return_value") not in {0,None}:
                # Poll/query APIs can legitimately report not-ready. Preserve
                # the observed status without equating it to lost attribution.
                nonzero_api_returns.append(dict(record_index=index,record=record))
    invalid_indices={r["record_index"] for r in invalid_times}
    for row in actions:
        row["record_indices"]={k:list(v) for k,v in row["record_indices"].items()}
        for kind in ("kernel","memcpy","runtime_api","driver_api"):
            records=[dict(record_index=i,**raw[i]) for i in row["record_indices"].get(kind,()) if i not in invalid_indices]
            row[kind+"_records"]=records
            union=interval_union(records)
            row[kind+"_union"]=union;row[kind+"_union_ns"]=interval_length(union)
        device_union=interval_union(row["kernel_union"]+row["memcpy_union"])
        row["device_activity_union"]=device_union;row["device_activity_union_ns"]=interval_length(device_union)
        row["runtime_and_driver_API_union_ns"]=interval_length(row["runtime_api_union"]+row["driver_api_union"])
        row.update(_descriptive_gaps((row["begin_ns"],row["end_ns"]),device_union))
        original=inventory[row["action_index"]]
        expect_gpu=(row["kind"]=="COMPUTE" and row["device"]=="gpu") or (
            row["kind"]=="COPY" and "vram" in (original.get("src"),original.get("dst")))
        row["GPU_activity_expected"]=expect_gpu
        if enabled and expect_gpu and not device_union:row["attribution_issues"].append("expected_GPU_activity_missing")
        row["selected_action_attribution_qualified"]=bool(enabled and not row["attribution_issues"])
        selected_record_problems.extend(dict(external_id=row["external_id"],reason=problem) for problem in row["attribution_issues"])
        row["host_API_device_columns_additive"]=False
        row["device_activity_is_SM_active_service"]=False
    if invalid_times:issues.append("invalid_time_records_retained")
    if unknown_kinds:issues.append("unknown_activity_kinds_retained")
    if selected_record_problems:issues.append("selected_action_attribution_issues")
    # Orphans are retained but never assigned by a host-time containment guess.
    # Unscoped setup/flag-query APIs outside action ranges can be legitimate.
    for orphan in unassigned:
        r=orphan["record"]
        if r.get("kind") in {"kernel","memcpy","runtime_api","driver_api"} and r.get("timestamp_valid") is True and any(
                a["begin_ns"]<r.get("end_ns",-1) and r.get("begin_ns",-1)<a["end_ns"] for a in actions):
            issues.append("unattributed_activity_overlaps_selected_action")
    result=dict(schema=SCHEMA,native_semantic_sha256=digest(native),program_sha256=native.get("program_sha256"),
        program_binding_checked=program is not None,process_id=pid,main_thread_id=tid,clock=clock,
        instrumentation=native["instrumentation"],mode=native["mode"],iterations=invocation_rows,actions=actions,
        raw_record_count=len(raw),record_kind_counts=dict(Counter(r.get("kind") for r in raw)),
        external_correlation_record_indices={str(k):v for k,v in mapping_rows.items()},
        conflicting_external_correlations=conflicts,orphan_records=unassigned,
        invalid_time_records=invalid_times,unknown_kind_records=unknown_kinds,
        nonzero_API_return_records=nonzero_api_returns,
        cupti_dropped_records=capture.get("cupti_dropped_records"),local_record_overflow=capture.get("local_record_overflow"),
        callback_errors=capture.get("callback_errors",[]),capture_issues=sorted(set(issues)),
        selected_action_issues=selected_record_problems,capture_qualified=bool(enabled and not issues),
        qualification_scope="Retained selected-action activity attribution and clock integrity only",
        timestamp_attribution_fallback_used=False,numerical_correctness_independently_verified=False,
        activity_instrumentation_neutrality_qualified=False,device_activity_is_SM_active_service=False,
        kernel_preemption_free_proven=False,host_API_device_columns_additive=False,
        available_resource_trajectory_measured=False,hardware_accuracy_qualified=False,
        batch_cpu=native.get("batch_cpu"))
    result["artifact_sha256"]=digest(result)
    return result


def load_physical_records(path, expected_sha256, *, program=None):
    path=Path(path).resolve();raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,"physical native source bytes differ")
    result=parse_physical_records(json.loads(raw),program=program)
    result["source_path"]=str(path);result["source_sha256"]=expected_sha256
    result["artifact_sha256"]=digest({k:v for k,v in result.items() if k!="artifact_sha256"})
    return result


def matmul_work(program, operation_id):
    """Known F32 GEMM work for a pure FFN matmul binding, excluding port copies."""
    require(isinstance(program,dict),"program object required")
    matches=[b for b in program.get("bindings",[]) if b.get("operation_id")==operation_id]
    require(len(matches)==1,"one exact operation binding required")
    binding=matches[0]
    require(binding.get("kind") in {"ffn_gate","ffn_up","ffn_down"},"only explicit pure matmul bindings have this work export")
    tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
    require(len(tensors)==len(program["graph"]["workload"]["tensors"]),"duplicate tensor identity")
    operands=binding.get("operands");outputs=binding.get("outputs")
    require(isinstance(operands,dict) and set(operands)=={"x0","x1"} and isinstance(outputs,dict) and set(outputs)=={"result"},
            "pure matmul operand/output roles differ")
    left,right,out=(tensors[operands["x0"]],tensors[operands["x1"]],tensors[outputs["result"]])
    for tensor in (left,right,out):
        require(tensor.get("dtype")=="GGML_TYPE_F32" and isinstance(tensor.get("ne"),list) and len(tensor["ne"])==4,
                "F32 four-dimensional tensor geometry required")
        for size in tensor["ne"]:_integer(size,"tensor dimension",1)
    k,m,a2,a3=left["ne"];rk,n,b2,b3=right["ne"]
    require(k==rk and b2%a2==0 and b3%a3==0 and out["ne"]==[m,n,b2,b3],"GGML matmul geometry/broadcast differs")
    return dict(operation_id=operation_id,semantic_kind=binding["kind"],left_tensor_id=left["id"],right_tensor_id=right["id"],
        output_tensor_id=out["id"],k=k,m=m,n=n,batch=b2*b3,gemm_flops=2*k*m*n*b2*b3,
        output_storage_bytes=4*m*n*b2*b3,flop_convention="multiply-add counts as two FLOPs",
        scope="Known GEMM arithmetic only; native materialization and other kernels are additional work",
        entire_action_physical_work_identified=False)
