"""Descriptive strategy ranking at an explicitly checked common Dense boundary.

Pure stdlib, no hardware, fitting, sample selection or frozen artifact mutation.
This does not replace verify_highlevel.py. It checks archived program/ledger
identities and recorded numerical verdicts, reconstructs common logical weight
contents across immutable FFN partitions, and requires identical external data,
locations, configuration and heldout seed/process cohorts before comparing paths.
All declared paths and all pairs are retained, including unstable/unknown ones.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import struct

SCHEMA="highlevel-common-boundary-rankings-v1"
DEFAULTS=dict(hidden=16,intermediate=32,heads=4,kv_heads=2,head_dim=4,tokens=1,history=3,
              attention_gate="sigmoid",qk_norm="rms")
VERDICTS=("full_graph_equal","action_trace_equal","full_port_inventory_equal","copy_routes_bytes_and_values_equal")


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0
def cv(values):return statistics.pstdev(values)/statistics.mean(values)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate JSON key: "+key);result[key]=value
        return result
    def invalid(value):raise ValueError("non-finite JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"),object_pairs_hook=pairs,parse_constant=invalid)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def check_raw(case,split,index,evidence):
    folder=(case/f"{split}-process-{index}").resolve()
    expected=dict(raw_native_file=str(folder/"native.json"),raw_native_sha256=file_sha(folder/"native.json"),
                  execution_file=str(folder/"execution.json"),execution_sha256=file_sha(folder/"execution.json"))
    require(canonical(evidence)==canonical(expected),"raw process identity/SHA differs")
    execution=read(folder/"execution.json")
    require(type(execution.get("returncode")) is int and execution["returncode"]==0 and execution.get("timed_out") is False and not execution.get("launch_failed"),"raw process failed or timed out")
    require(execution.get("pid_scope")=="one_fresh_native_process","raw process independence contract missing")


def sealed(value,key):
    unsigned=deepcopy(value);claimed=unsigned.pop(key,None)
    require(claimed==digest(unsigned),key+" identity changed")


def data_digest(values,dtype="GGML_TYPE_F32"):
    require(isinstance(values,list),"materialized numeric values must be an array")
    h=hashlib.sha256()
    for offset in range(0,len(values),4096):
        block=values[offset:offset+4096]
        if dtype=="GGML_TYPE_I32":
            require(all(type(v) is int and -(2**31)<=v<2**31 for v in block),"invalid I32 values")
            h.update(struct.pack("<"+"i"*len(block),*block))
        else:
            require(dtype=="GGML_TYPE_F32" and all(type(v) in (int,float) and math.isfinite(v) for v in block),"invalid F32 values")
            h.update(struct.pack("<"+"f"*len(block),*block))
    return h.hexdigest()


def layout(tensor):
    return {key:tensor[key] for key in ("dtype","ne","nb","storage_bytes")}


def external_boundary(program):
    """Normalize true start/finish ports; verify logical FFN parent contents.

    The static part excludes strategy-internal tensors and representations.
    The data part binds hidden/positions/history and reconstructed full weights.
    Parent matrices and contiguous partition leaves must have equal contents;
    equal seeds alone are insufficient evidence.
    """
    sealed(program,"program_sha256")
    for key,value in program["identity"].items():require(digest(program[key])==value,"program component changed: "+key)
    tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
    bindings=program["bindings"];projections=[b for b in bindings if b["kind"]=="projection"]
    require(len(projections)==1,"ranking supports exactly one request in one Dense invocation")
    projection=projections[0];shape=projection["shape"];config=program["physical"]["config"]
    require(all(b["shape"]==shape for b in bindings),"one path has mixed request shapes")
    require(shape["batch"]==1,"batch-one boundary required")
    goals=program["graph"]["scenario"]["goal"]["output_locations"]
    final=[b["outputs"]["result"] for b in bindings if b["kind"] in {"dense_ffn","ffn_residual"}]
    require(len(final)==1,"complete hidden output with FFN/residual required")
    history={};fresh={}
    for b in bindings:
        for row in b["history"]:
            if row["id"] in history:require(history[row["id"]]==row,"KV history metadata differs between consumers")
            history[row["id"]]=row
        for row in b["new_kv"]:
            if row["id"] in fresh:require(fresh[row["id"]]==row,"new KV metadata differs between producers")
            fresh[row["id"]]=row
    require(set(goals)=={*history,*fresh,final[0]},"external goals include unknown/omitted hidden or KV ports")
    initial=program["initial_values"]
    def input_port(tid):
        tensor=tensors[tid]
        locations=sorted(tensor.get("initial_locations",[]))
        require(locations,"external input has no declared initial copy")
        require(len(initial[tid])==math.prod(tensor["ne"]),"external input data shape mismatch")
        return dict(layout=layout(tensor),locations=locations),data_digest(initial[tid],tensor["dtype"])
    ports={};data={}
    for name in ("hidden","positions"):
        ports[name],data[name]=input_port(projection["operands"][name])
    old=[];new=[]
    for label,inventory,target in (("historical_kv",history,old),("new_kv",fresh,new)):
        for row in sorted(inventory.values(),key=lambda x:(x["first_position"],x["tokens"])):
            tid=row["id"];entry=dict(first_position=row["first_position"],tokens=row["tokens"],layout=layout(tensors[tid]),output_location=goals[tid])
            if label=="historical_kv":
                source,digest_value=input_port(tid);entry["initial_locations"]=source["locations"]
                data[f"history:{row['first_position']}:{row['tokens']}"]=digest_value
            target.append(entry)
    expected_inputs={projection["operands"]["hidden"],projection["operands"]["positions"],*history}
    require({tid for tid,t in tensors.items() if t["role"]!="weight" and t.get("initial_locations")}==expected_inputs,"unknown external input inventory")
    partitions=(program["graph"].get("boundary") or {}).get("ffn_execution",{}).get("weight_partitions",[])
    weights={};weight_data={}
    for group in program["physical"]["weight_groups"].values():
        for role,name in group.items():
            parent=f"w:{program['layer_id']}:{name}"
            if parent in tensors:
                tensor=tensors[parent];values=initial[parent];locations=sorted(tensor.get("initial_locations",[]));ne=tensor["ne"]
            else:
                leaves=sorted([p for p in partitions if p["source_tensor_id"]==parent],key=lambda p:p["start"])
                require(role in {"gate","up","down"} and leaves,"missing full parent or exact FFN partition")
                cursor=0;locations=None;ne=([config["intermediate"],config["hidden"],1,1] if role=="down" else [config["hidden"],config["intermediate"],1,1])
                values=[0.0]*math.prod(ne)
                for leaf in leaves:
                    require(leaf["start"]==cursor and leaf["end"]>cursor and leaf["stage"]==role,"incomplete/overlapping logical FFN partition")
                    cursor=leaf["end"];tid=leaf["tensor_id"];tensor=tensors[tid];source=initial[tid]
                    require(tensor["dtype"]=="GGML_TYPE_F32" and len(source)==math.prod(tensor["ne"]),"partition data/layout mismatch")
                    here=sorted(tensor.get("initial_locations",[]))
                    require(locations is None or locations==here,"partition leaves have different initial residency")
                    locations=here;width=leaf["end"]-leaf["start"]
                    expected_ne=[width,config["hidden"],1,1] if role=="down" else [config["hidden"],width,1,1]
                    require(tensor["ne"]==expected_ne,"partition matrix geometry mismatch")
                    if role=="down":
                        for row in range(config["hidden"]):values[row*ne[0]+leaf["start"]:row*ne[0]+leaf["end"]]=source[row*width:(row+1)*width]
                    else:values[leaf["start"]*ne[0]:leaf["end"]*ne[0]]=source
                require(cursor==config["intermediate"],"FFN leaves do not cover the parent matrix")
            require(locations and tensor["dtype"]=="GGML_TYPE_F32" and len(values)==math.prod(ne),"F32 initialized logical weights required")
            weights[role]=dict(dtype="GGML_TYPE_F32",ne=ne,storage_bytes=4*math.prod(ne),initial_locations=locations)
            weight_data[role]=data_digest(values)
    static=dict(config=config,shape=shape,position_start=projection["position_start"],
        inputs=ports,historical_kv=old,new_kv=new,hidden_output=dict(layout=layout(tensors[final[0]]),location=goals[final[0]]),
        logical_initial_weights=weights,required_runtime_flags=program["required_runtime_flags"],
        threads=program["physical"]["threads"],cuda_device=program["physical"].get("cuda_device",0),
        completion="all_declared_hidden_and_KV_outputs_delivered",allocation="preallocated_independent_backing; logical_eviction")
    return dict(static=static,static_sha256=digest(static),data_sha256=digest(dict(inputs=data,weights=weight_data)),
                seed=program["seed"],program_sha256=program["program_sha256"])


def interval(medians,seed,repetitions):
    rng=random.Random(seed)
    values=sorted(statistics.median(rng.choices(medians,k=len(medians))) for _ in range(repetitions))
    return [values[int(.025*(repetitions-1))],values[int(.975*(repetitions-1))]]


def path_evidence(root,item,report):
    require(isinstance(item.get("id"),str) and item["id"] and Path(item["id"]).name==item["id"] and item["id"] not in {".",".."},"unsafe case identity")
    case=root/item["id"];program=read(case/"program.json");boundary=external_boundary(program)
    require(program["program_sha256"]==item["profile_program_sha256"] and program["seed"]==item["profile_seed"],"prepared source program/seed identity differs")
    ledger=read(case/"frozen_profile.json");sealed(ledger,"ledger_sha256")
    require(ledger["program_sha256"]==program["program_sha256"],"ledger is bound to another source program")
    require(ledger["identities"]=={k:v for k,v in program["identity"].items() if k!="initial_values"},"ledger physical graph/operands differ")
    require(all(ledger.get("verification",{}).get(k) is True for k in VERDICTS),"recorded source numerical/copy verification incomplete")
    source_stats=ledger["independent_process_statistics"]
    source_indices=[p["process_index"] for p in source_stats]
    require(len(source_indices)>=3 and sorted(source_indices)==list(range(len(source_indices))),"source process coverage incomplete/duplicated")
    require(sorted(p.name for p in case.glob("profile-process-*"))==sorted(f"profile-process-{i}" for i in source_indices),"raw source process directories missing/duplicated/unplanned")
    source_evidence=ledger.get("evidence",{}).get("processes",[])
    require(len(source_evidence)==len(source_indices),"raw source evidence count differs")
    for i,evidence in enumerate(source_evidence):check_raw(case,"profile",i,evidence)
    local=sorted([r for r in report.get("results",[]) if r.get("case_id")==item["id"]],key=lambda r:r["process_index"])
    expected=report["independent_heldout_processes_per_case"]
    require(type(expected) is int and expected>=3 and [r["process_index"] for r in local]==list(range(expected)),"heldout process coverage incomplete/duplicated")
    require(sorted(p.name for p in case.glob("heldout-process-*"))==sorted(f"heldout-process-{i}" for i in range(expected)),"raw heldout process directories missing/duplicated/unplanned")
    require(sorted(p.name for p in case.glob("comparison-*.json"))==sorted(f"comparison-{i}.json" for i in range(expected)),"comparison files missing/duplicated/unplanned")
    require(not any(r.get("case_id")==item["id"] for r in report.get("failures",[])),"a declared path has failed/unknown process evidence")
    prediction=read(case/"modelplan_measured_prediction.json")
    require(prediction.get("actual_ModelPlan_policy_rerun") is True and prediction.get("action_trace_equal") is True and prediction.get("synthetic_costs_used") is False,"measured high-level policy prediction missing")
    predicted=prediction["makespan_ns"];require(positive(predicted),"invalid frozen prediction")
    samples=[];medians=[];cohorts=[]
    for row in local:
        i=row["process_index"];heldout=read(case/f"heldout-program-{i}.json");cohort=external_boundary(heldout)
        require(canonical(read(case/f"comparison-{i}.json"))==canonical(row),"reported comparison differs from its saved process result")
        check_raw(case,"heldout",i,row["raw_evidence"])
        require(cohort["static_sha256"]==boundary["static_sha256"],"heldout external boundary differs from source")
        require(heldout["seed"]!=program["seed"] and cohort["data_sha256"]!=boundary["data_sha256"],"heldout source data/seed leakage")
        require(row["program_sha256"]==heldout["program_sha256"] and row["ledger_sha256"]==ledger["ledger_sha256"],"heldout comparison identity differs")
        require(all(row.get("verification",{}).get(k) is True for k in VERDICTS),"recorded heldout numerical/copy verification incomplete")
        require(row["actual_modelplan_prediction"]["makespan_ns"]==predicted and row["baseline"]["makespan_ns"]==predicted,"heldout process changed its frozen prediction")
        values=row["observed_samples_ns"];require(isinstance(values,list) and len(values)>=3 and all(type(v) is int and v>0 for v in values),"invalid heldout process samples")
        require(statistics.median(values)==row["observed_whole_program_median_ns"],"heldout reported median differs from samples")
        samples.extend(values);medians.append(statistics.median(values));cohorts.append(dict(process_index=i,seed=heldout["seed"],data_sha256=cohort["data_sha256"],samples=len(values)))
    require(len({c["seed"] for c in cohorts})==len(cohorts),"heldout numerical cohorts are not independent seeds")
    groups=[g for g in report["case_groups"] if g["case_id"]==item["id"]]
    require(len(groups)==1,"missing/duplicate complete path group")
    group=groups[0];observed=statistics.median(samples)
    require(group.get("status")=="complete" and group["predicted_ns"]==predicted and group["pooled_heldout_median_ns"]==observed,"complete path aggregate differs from samples/prediction")
    quality=ledger.get("source_quality_passed") is True and all(cv(row["observed_samples_ns"])<=.15 for row in local) and cv(medians)<=.20 and abs(predicted-observed)/observed<=.20
    require(group.get("accepted") is quality,"saved complete path acceptance differs from fixed gates")
    return dict(case_id=item["id"],strategy_id=item["id"],known=True,predicted_ns=predicted,observed_ns=observed,
        heldout_process_medians_ns=medians,latency_quality_accepted=quality,
        source_process_indices=sorted(source_indices),profile_seed=program["seed"],profile_data_sha256=boundary["data_sha256"],
        heldout_cohorts=cohorts,boundary_sha256=boundary["static_sha256"],external_boundary=boundary["static"],
        attention_staged=any(b["kind"]=="attention_qk" for b in program["bindings"]),
        ffn_intermediate_cuts=sorted({b["op_params"][1] for b in program["bindings"] if b["kind"]=="ffn_gate" and b["op_params"][1]<program["physical"]["config"]["intermediate"]}))


def rank_highlevel(root,campaign,report,*,bootstrap_seed=1907,bootstrap_repetitions=2000):
    require(type(bootstrap_seed) is int and type(bootstrap_repetitions) is int and bootstrap_repetitions>=100,"fixed valid bootstrap protocol required")
    root=Path(root);paths=[];by_geometry=defaultdict(list)
    for item in campaign.get("cases",[]):
        geometry={key:item.get("kwargs",{}).get(key,default) for key,default in DEFAULTS.items()}
        try:path=path_evidence(root,item,report)
        except Exception as error:path=dict(case_id=item["id"],strategy_id=item["id"],known=False,reason=str(error),latency_quality_accepted=False)
        path["declared_geometry"]=geometry;paths.append(path);by_geometry[canonical(geometry)].append(path)
    rankings=[]
    for key,values in sorted(by_geometry.items()):
        geometry=json.loads(key);entry=dict(declared_geometry=geometry,declared_strategies=[p["strategy_id"] for p in values],known=False)
        reasons=[]
        if len(values)<2:reasons.append("fewer than two declared strategies at this geometry")
        if len({p["strategy_id"] for p in values})!=len(values):reasons.append("duplicate declared strategy identity")
        if any(not p["known"] for p in values):reasons.append("one or more declared paths are unknown/failed; no subset winner selected")
        if not reasons:
            if len({p["boundary_sha256"] for p in values})!=1:reasons.append("initial/final placement, logical weights, configuration or completion boundaries differ")
            if len({(p["profile_seed"],p["profile_data_sha256"]) for p in values})!=1:reasons.append("profile numerical input/weight cohorts differ")
            if len({canonical(p["source_process_indices"]) for p in values})!=1:reasons.append("source process coverage differs across strategies")
            if len({canonical(p["heldout_cohorts"]) for p in values})!=1:reasons.append("heldout seed/process/input/weight/sample-count cohorts differ")
        if reasons:
            entry.update(reasons=reasons,all_declared_pairs=[dict(left=a["strategy_id"],right=b["strategy_id"],comparison_known=False,reasons=reasons)
                         for a,b in itertools.combinations(values,2)])
            rankings.append(entry);continue
        intervals={p["strategy_id"]:interval(p["heldout_process_medians_ns"],bootstrap_seed,bootstrap_repetitions) for p in values}
        predicted=sorted(values,key=lambda p:(p["predicted_ns"],p["strategy_id"]));actual=sorted(values,key=lambda p:(p["observed_ns"],p["strategy_id"]))
        best=actual[0];selected=predicted[0];regret=selected["observed_ns"]-best["observed_ns"]
        separated=all(intervals[best["strategy_id"]][1]<intervals[p["strategy_id"]][0] for p in actual[1:])
        quality=all(p["latency_quality_accepted"] for p in values)
        pairs=[]
        for left,right in itertools.combinations(values,2):
            pl,pr=left["predicted_ns"],right["predicted_ns"];ol,orr=left["observed_ns"],right["observed_ns"]
            li,ri=intervals[left["strategy_id"]],intervals[right["strategy_id"]]
            pairs.append(dict(left=left["strategy_id"],right=right["strategy_id"],comparison_known=True,profile_prediction_ratio=max(pl,pr)/min(pl,pr),
                median_order_correct=None if pl==pr or ol==orr else (pl<pr)==(ol<orr),
                observed_intervals_overlap=not (li[1]<ri[0] or ri[1]<li[0]),
                latency_quality_accepted=left["latency_quality_accepted"] and right["latency_quality_accepted"]))
        entry.update(known=True,comparison_group=values[0]["boundary_sha256"],external_boundary=values[0]["external_boundary"],
            predicted_order=[p["strategy_id"] for p in predicted],observed_order=[p["strategy_id"] for p in actual],
            predicted_best=selected["strategy_id"],observed_best=best["strategy_id"],selected_best=selected["observed_ns"]==best["observed_ns"],
            selection_regret_ns=regret,selection_regret_relative=regret/best["observed_ns"],
            observed_winner_interval_separated=separated,ambiguous_overlapping_intervals=not separated,
            latency_quality_accepted=quality,selection_evidence_accepted=quality and separated,
            inference="inconclusive_overlapping_process_intervals" if not separated else "intervals_separated_but_latency_quality_failed" if not quality else "common_boundary_comparison_supported",
            heldout_process_median_intervals_ns=intervals,all_declared_pairs=pairs)
        rankings.append(entry)
    return dict(schema=SCHEMA,paths=paths,rankings=rankings,declared_paths=len(paths),known_paths=sum(p["known"] for p in paths),
        comparable_groups=sum(r["known"] for r in rankings),unknown_or_incomparable_groups=sum(not r["known"] for r in rankings),
        bootstrap_seed=bootstrap_seed,bootstrap_repetitions=bootstrap_repetitions,
        interval_unit="independent_process_medians; not isolated-request tail latency",
        pair_selection="all_declared_pairs; no heldout-selected proximity or winner filter",
        protocol_verifier_execution_claimed=False,numerical_status_scope="recorded full-port/copy verdicts; independently verify raw artifacts with verify_highlevel.py",
        physical_capacity_claimed=False,scope="one complete preallocated serial F32 Dense invocation; independent immutable FFN representations normalized to identical logical matrices")
