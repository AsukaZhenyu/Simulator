"""Explicit process-affinity intervention with unchanged one-thread computation.

The same old binary and main-thread mask are kept. Process masks 4 and 20
permit different helper-thread resources; they are distinct profiling contexts.
No particular physical core, driver thread or operating-system wait is inferred.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from . import task1


def prepare(out, previous):
    out=Path(out).resolve();previous=Path(previous).resolve();out.mkdir(parents=True,exist_ok=False)
    old=task1.read(previous/"manifest.json");jobs=[];keys=set()
    combinations=[(src,dst,size) for size in (1024,1048576) for src,dst in task1.ROUTES]
    for repeat in range(4):
        order=combinations[repeat:]+combinations[:repeat]
        for src,dst,size in order:
            key=f"transfer-{src}-{dst}-{size}";keys.add(key)
            for mask in (4,20) if repeat%2==0 else (20,4):
                request=dict(task1.sample(),repeats=256)
                jobs.append(dict(id=f"affinity-{src}-{dst}-{size}-r{repeat}-mask{mask}",split="calibration",program_key=key,
                                 round=repeat,process_mask=mask,request=request))
    manifest=dict(old,schema="dac-followup-affinity-plan-1",prepared_at_utc=task1.utc(),jobs=jobs,
                  programs={key:old["programs"][key] for key in keys},sources=[*task1.current_sources(),task1.receipt(__file__)],
                  profile_context=task1.fresh_profile_context(repeats=256,process_masks=(4,20)))
    manifest.pop("pre_measurement_refinement",None)
    manifest["reused_program_manifest"]=task1.receipt(previous/"manifest.json")
    manifest["protocol"]=dict(kind="source-motivated independent process-affinity intervention",jobs=32,
        sizes=[1024,1048576],process_masks=[4,20],main_thread_mask=4,ggml_compute_threads=1,
        trials=8,repeats=256,warmup_iterations=20,minimum_active_warmup_ms=500,
        order="four fresh rounds; mask4/mask20 then reverse, balanced 2+2",
        physical_core_topology_assumed=False,old_price_coordinate_reused=False,
        claim_scope="overall effect of permitting another logical CPU for runtime/helper threads; not specific driver-thread causation",
        quality_thresholds=dict(max_process_trial_cv=.10),all_observations_retained=True)
    task1.write(out/"manifest.json",manifest)
    return dict(prepared=str(out),jobs=32,native_timing_started=False)


def collect(out):
    from profiling.state_calibration import run
    out=Path(out).resolve();manifest=task1.read(out/"manifest.json")
    assignment={str(out/"jobs"/j["id"]/"request.json"):j["process_mask"] for j in manifest["jobs"]}
    original=run.launch
    def explicit_affinity_launcher(command,env,stdout,stderr,mask):
        task1.require(mask==4 and command[2] in assignment,"unexpected launcher request outside frozen intervention")
        return original(command,env,stdout,stderr,assignment[command[2]])
    # This local adapter belongs to this separate collector process. Historical
    # Python sources, the old binary and the shared system settings are unchanged.
    run.launch=explicit_affinity_launcher
    try:result=task1.collect(out,"calibration")
    finally:run.launch=original
    for row in task1.read(out/"calibration-executions.json"):
        if row["status"]!="completed":continue
        native=task1.read(row["native"]["path"]);expected=row["job"]["process_mask"]
        task1.require(row["affinity"]["requested_mask"]==row["affinity"]["observed_mask"]==expected,
                      "parent process affinity readback differs")
        task1.require(native["main_thread_affinity"]["process_mask"]==expected,
                      "native process affinity readback differs")
        task1.require(native["main_thread_affinity"]["requested_mask"]==native["main_thread_affinity"]["final_mask"]==4,
                      "main affinity changed under intervention")
    return result


def evaluate(out):
    out=Path(out).resolve();manifest=task1.read(out/"manifest.json");rows=[]
    executions=task1.read(out/"calibration-executions.json")
    for row in executions:
        if row["status"]!="completed":continue
        native=task1.read(row["native"]["path"])
        times=[r["wall_ns"]/r["repeats"] for r in native["trials"]]
        shape=row["program"]
        rows.append(dict(route=shape["route"],bytes=shape["bytes"],round=row["job"]["round"],process_mask=row["job"]["process_mask"],
                         mean_ns=statistics.mean(times),trial_cv=statistics.stdev(times)/statistics.mean(times),native=row["native"]))
    conditions=[]
    for src,dst in task1.ROUTES:
        for size in (1024,1048576):
            route=src+"->"+dst;pairs=[];means={}
            for mask in (4,20):
                selected=[r for r in rows if r["route"]==route and r["bytes"]==size and r["process_mask"]==mask]
                values=[r["mean_ns"] for r in selected]
                means[mask]=dict(processes=len(values),mean_ns=statistics.mean(values) if values else None,
                                 between_process_cv=statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None,
                                 trial_cv_passes=sum(r["trial_cv"]<=.10 for r in selected))
            for repeat in range(4):
                selected={r["process_mask"]:r for r in rows if r["route"]==route and r["bytes"]==size and r["round"]==repeat}
                if set(selected)=={4,20}:pairs.append(dict(round=repeat,mask20_over_mask4_minus_one=selected[20]["mean_ns"]/selected[4]["mean_ns"]-1))
            conditions.append(dict(route=route,bytes=size,masks=means,pairs=pairs,
                                   aggregate_relative_effect=None if any(v["mean_ns"] is None for v in means.values()) else means[20]["mean_ns"]/means[4]["mean_ns"]-1))
    summary=dict(schema="dac-followup-affinity-result-1",manifest=task1.receipt(out/"manifest.json"),
                 native_expected=32,native_completed=len(rows),native_failed=sum(e["status"]!="completed" for e in executions),
                 conditions=conditions,process_rows=rows,main_thread_mask=4,compute_threads=1,
                 intervention_changes_profiling_context=True,specific_helper_thread_mechanism_identified=False,
                 old_results_requalified=False,all_observations_retained=True)
    task1.write(out/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","collect","evaluate"))
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--previous",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":result=prepare(args.out,args.previous)
    elif args.command=="collect":result=collect(args.out)
    else:result=evaluate(args.out)
    print(result)
