"""Fixed overlap protocol, validation and profile-only cost freezing."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

QUALITY=dict(max_within_process_cv=.15,max_between_process_cv=.20,max_heldout_relative_error=.20)
STRATEGIES=("serial/one_copy_compute","overlap/one_copy_compute","serial/two_copy_compute","overlap/two_copy_compute")
COMPONENTS=("copy/a","copy/b","compute/current_ffn")
CONTEXT_SHA="37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4"
MATCHED_SHA="2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
def cv(values):return statistics.pstdev(values)/statistics.mean(values) if statistics.mean(values) else 0.


def suite():
    cases=[]
    for h in (256,512,1024):
        for q in (1,32):
            for mib in (1,16,64):
                cases.append(dict(id=f"ffn-h{h}-i{2*h}-q{q}-copy{mib}MiB",hidden=h,intermediate=2*h,tokens=q,payload_bytes=mib*1024**2))
    return dict(schema="overlap-accuracy-suite-0.1",cases=cases,profile_seeds=[2903],heldout_seeds=[3911],repeats=3,
                protocol=dict(threads=1,cpu_id=0,device=0,warmup=10,samples=31,min_warmup_ms=100),quality=QUALITY)


def validate_suite(value):
    if set(value)!={"schema","cases","profile_seeds","heldout_seeds","repeats","protocol","quality"} or value["schema"]!="overlap-accuracy-suite-0.1" or value["quality"]!=QUALITY:
        raise ValueError("explicit fixed overlap suite and original quality gates required")
    seeds=value["profile_seeds"]+value["heldout_seeds"]
    if len(seeds)!=len(set(seeds)) or any(type(x)is not int or not 0<=x<=1000000 for x in seeds):raise ValueError("disjoint unique explicit seeds required")
    if type(value["repeats"])is not int or not 3<=value["repeats"]<=8 or not value["profile_seeds"] or not value["heldout_seeds"]:raise ValueError("at least three new processes per split")
    p=value["protocol"]
    if set(p)!={"threads","cpu_id","device","warmup","samples","min_warmup_ms"}:raise ValueError("exact protocol fields required")
    for key,lo,hi in (("threads",1,1),("cpu_id",0,63),("device",0,15),("warmup",0,10000),("samples",3,10000),("min_warmup_ms",0,60000)):
        if type(p[key])is not int or not lo<=p[key]<=hi:raise ValueError("invalid protocol "+key)
    ids=set()
    if not value["cases"]:raise ValueError("nonempty explicit cases required")
    for c in value["cases"]:
        if set(c)!={"id","hidden","intermediate","tokens","payload_bytes"} or not isinstance(c["id"],str) or not re.fullmatch("[A-Za-z0-9_-]+",c["id"]) or c["id"]in ids:raise ValueError("unique safe explicit case geometry required")
        ids.add(c["id"])
        for key,lo,hi in (("hidden",1,1024),("intermediate",2,4096),("tokens",1,128),("payload_bytes",1024**2,64*1024**2)):
            if type(c[key])is not int or not lo<=c[key]<=hi:raise ValueError("invalid bounded geometry "+key)
        if c["payload_bytes"]%4:raise ValueError("F32 aligned payload required")
    return copy.deepcopy(value)


def samples(values,n):
    if not isinstance(values,list)or len(values)!=n or any(type(v)is not int or v<=0 for v in values):raise ValueError("unfiltered positive integer samples required")


def validate_native(case,protocol,seed,payload,probe_sha):
    if payload.get("schema")!="overlap-accuracy-native-0.1":raise ValueError("native protocol differs")
    for key in ("hidden","intermediate","tokens","payload_bytes"):
        if payload.get(key)!=case[key]:raise ValueError("native geometry differs: "+key)
    for key,value in protocol.items():
        if payload.get(key)!=value:raise ValueError("native sampling/context differs: "+key)
    required=dict(mode="measure",seed=seed,prefetch_payload_checks_passed=True,prefetch_destinations_disjoint_from_each_other_and_all_current_ports=True,
        current_output_location="vram",prefetch_destination_location="vram",prefetch_source="preinitialized_page_locked_host_memory",
        completed_work_timer="steady_clock",matmul_precision="GGML_PREC_F32",nvidia_tf32_override="0",cuda_graphs="disabled",
        complete_cuda_workspace_bytes=None,physical_capacity_verified=False,kernel_overlap_proved=False,copy_events_prove_kernel_overlap=False,
        instrumentation_overhead_subtracted=False,all_checks_passed=True)
    if any(payload.get(k)!=v for k,v in required.items()):raise ValueError("native boundary/numeric/precision/evidence contract differs")
    if payload.get("source_binding")!={"probe_sha256":probe_sha,"context_sha256":CONTEXT_SHA,"matched_sha256":MATCHED_SHA}:raise ValueError("compiled source binding differs")
    times=payload.get("wall_ns",{})
    if set(times)!=set(COMPONENTS)|set(STRATEGIES):raise ValueError("native metric coverage differs")
    for v in times.values():samples(v,protocol["samples"])
    checks=payload.get("checks",{})
    expected={s+suffix for s in STRATEGIES for suffix in ("/output","/diagnostic_output")}|{"current_inputs_unchanged/"+n for n in ("gate_weight","up_weight","down_weight","input")}
    if set(checks)!=expected or any(x.get("passed")is not True or x.get("elements",0)<=0 for x in checks.values()):raise ValueError("numerical/current-input coverage failed")
    traces=payload.get("diagnostic_host_and_copy_events",{})
    if set(traces)!=set(STRATEGIES):raise ValueError("diagnostic path coverage differs")
    for strategy,entries in traces.items():
        if len(entries)!=protocol["samples"]:raise ValueError("diagnostic invocation coverage differs")
        two="two_copy"in strategy
        for t in entries:
            ordered=[t[k] for k in ("copy_a_submit_begin_ns","copy_a_submit_end_ns")]
            if two:ordered.extend(t[k]for k in ("copy_b_submit_begin_ns","copy_b_submit_end_ns"))
            ordered.extend(t[k]for k in ("compute_begin_ns","compute_end_ns","join_begin_ns","join_end_ns","total_ns"))
            if any(type(v)is not int or v<0 for v in ordered)or ordered!=sorted(ordered)or ordered[-1]<=0:raise ValueError("invalid actual host event ordering")
            if not two and (t["copy_b_submit_begin_ns"]!=-1 or t["copy_b_submit_end_ns"]!=-1 or t["copy_b_event_elapsed_ns"]!=-1):raise ValueError("single-copy path reports nonexistent second flow")
            if type(t["copy_a_event_elapsed_ns"])is not int or t["copy_a_event_elapsed_ns"]<0 or two and (type(t["copy_b_event_elapsed_ns"])is not int or t["copy_b_event_elapsed_ns"]<0):raise ValueError("invalid CUDA copy event observation")
    overhead=payload.get("instrumentation_overhead_ns",{})
    if set(overhead)!={"clock_pair","empty_cuda_event_pair_completed"}:raise ValueError("separate overhead observations missing")
    for v in overhead.values():samples(v,protocol["samples"])


def freeze(profile,spec):
    if not profile or any(r["split"]!="profile"for r in profile):raise ValueError("profile-only rows required")
    cases=[]
    for case in spec["cases"]:
        selected=[r for r in profile if r["case"]==case and r["status"]=="measured"]
        prices={}
        for metric in COMPONENTS:
            observations=[x for row in selected for x in row["metrics"][metric]]
            if observations:prices[metric]=dict(median_ns=statistics.median(observations),sources=[dict(id=r["id"],raw_sha256=r["raw_sha256"],split="profile",median_ns=statistics.median(r["metrics"][metric]),cv=cv(r["metrics"][metric]))for r in selected])
        cases.append(dict(case=case,prices=prices))
    return dict(schema="overlap-isolated-profile-model-0.1",trained_on="profile_only",heldout_used_for_fitting=False,
        joint_path_totals_used_as_prices=False,interpolation="none",cases=cases,source_rows=[dict(id=r["id"],raw_sha256=r["raw_sha256"],status=r["status"])for r in profile])
