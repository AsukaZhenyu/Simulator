"""Bounded heldout Dense composition across connected layers on one CPU/GPU.

Uses an explicitly supplied frozen independent response table, never target
latencies as parameters. Native acquisition shares the repository measurement
lock and keeps every failed process. No observer qualification is inferred from
this smaller composition extension.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np

from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable
from profiling.dac_validation import e2a
from profiling.highlevel_accuracy.bridge import export_plan, reference_ports, validate_program
from profiling.highlevel_accuracy.multilayer import build_multilayer_case

ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    dict(id="l2-h256-p16",layers=2,hidden=256,tokens=16,history=0),
    dict(id="l4-h256-p64",layers=4,hidden=256,tokens=64,history=0),
    dict(id="l2-h512-p64",layers=2,hidden=512,tokens=64,history=0),
    dict(id="l4-h256-d1h128",layers=4,hidden=256,tokens=1,history=128),
)
RECIPES = (("cpu","cpu","cpu"),("gpu","gpu","gpu"),("gpu","cpu","gpu"))


def prepare(out, profile, native_manifest):
    out, profile, native_manifest = map(lambda p: Path(p).resolve(), (out,profile,native_manifest))
    out.mkdir(parents=True,exist_ok=False)
    frozen=e2a.read(profile);native=e2a.read(native_manifest)
    binary=Path(native["binary"])
    e2a.require(e2a.sha(binary)==frozen["context"]["binary_sha256"],"frozen profile native binary changed")
    for path, expected in frozen["context"]["runtime_dlls"].items():
        e2a.require(e2a.sha(path)==expected,"profile runtime asset changed")
    table=ReusableResponseTable(tuple(frozen["entries"]))
    programs={};predictions=[];matching=[]
    for case in MATRIX:
        cohort=[];oracle_cohort=[]
        for strategy, devices in enumerate(RECIPES):
            key=case["id"]+f"-s{strategy}"
            kwargs=dict(hidden=case["hidden"],intermediate=case["hidden"]*2,heads=case["hidden"]//64,
                kv_heads=2,head_dim=64,tokens=case["tokens"],history=case["history"],threads=1,devices=devices)
            plan=build_multilayer_case(out/"inputs"/key,layers=case["layers"],**kwargs)
            program=export_plan(plan.compile(),seed=2026102513)
            reference=reference_ports(program)
            pp,rp=out/"programs"/(key+".json"),out/"references"/(key+".json")
            e2a.write(pp,program);e2a.write(rp,reference)
            begin=time.perf_counter()
            prediction=table.evaluate(program,frozen["context"])
            wall=time.perf_counter()-begin
            programs[key]=dict(program=e2a.receipt(pp),reference=e2a.receipt(rp),seed=program["seed"])
            predictions.append(dict(workload=case["id"],strategy=strategy,program_key=key,
                predicted_ns=str(prediction["duration_ns"]),actions=len(program["actions"]),
                exact_coordinate_coverage=True,target_time_read=False,simulation_wall_seconds=wall,
                action_ns=[str(v) for v in prediction["action_ns"]]))
            cohort.append(program);oracle_cohort.append(reference)
        base=cohort[0]
        e2a.require(all(p["initial_values"]==base["initial_values"] for p in cohort),"recipe inputs/weights/KV differ")
        goals=base["graph"]["scenario"]["goal"]["output_locations"]
        e2a.require(all(p["graph"]["scenario"]["goal"]["output_locations"]==goals for p in cohort),"recipe goals differ")
        worst=0.
        for tid in goals:
            want=np.asarray(oracle_cohort[0].get(tid,base["initial_values"].get(tid)))
            for program, reference in zip(cohort[1:],oracle_cohort[1:]):
                got=np.asarray(reference.get(tid,program["initial_values"].get(tid)))
                e2a.require(np.allclose(got,want,atol=5e-5,rtol=5e-5),"recipe mathematical output differs")
                worst=max(worst,float(np.max(np.abs(got-want))))
        matching.append(dict(workload=case["id"],same_initial_tensors=True,same_parent_weights=True,
            same_layer_KV=True,same_final_goals=True,maximum_reference_difference=worst,
            actual_invocations=case["layers"],actual_connected_hidden_seams=case["layers"]-1,
            independent_numpy_ports=True))
    jobs=[]
    orders=((0,1,2),(1,2,0),(2,0,1),(2,1,0))
    for round_id, order in enumerate(orders):
        for offset in range(len(MATRIX)):
            case=MATRIX[(offset+round_id)%len(MATRIX)]
            for position,strategy in enumerate(order):
                jobs.append(dict(id=f"heldout-r{round_id}-{case['id']}-s{strategy}",split="heldout",
                    program_key=case["id"]+f"-s{strategy}",round=round_id,position=position,
                    request=e2a.request("full",trials=12,warmup_ms=500),role="new complete multilayer target; never used to fit prices"))
    shutil.copyfile(profile,out/"profile-frozen.json")
    e2a.write(out/"predictions.json",predictions)
    e2a.write(out/"matching.json",matching)
    source_paths={Path(p).resolve() for p in (Path(__file__),ROOT/"profiling/highlevel_accuracy/bridge.py",
        ROOT/"profiling/highlevel_accuracy/multilayer.py",ROOT/"profiling/dac_validation/e2a.py",
        ROOT/"modeling/llm_infer_model/performance/reusable_action_costs.py",profile,binary,native_manifest,
        binary.parent/"build-manifest.json")}
    source_paths.update(Path(p).resolve() for p in frozen["context"]["runtime_dlls"])
    for binding in programs.values():
        document=e2a.read(binding["program"]["path"])
        source_paths.update(Path(row["path"]).resolve() for row in document["source_bound_leaf_files"])
    sources=[e2a.receipt(p) for p in sorted(source_paths)]
    manifest=dict(schema="multilayer-heldout-composition-1",prepared_at_utc=e2a.utc(),matrix=MATRIX,
        recipes=[dict(id=f"S{i}",placements=dict(zip(("projection","attention","dense_ffn"),v))) for i,v in enumerate(RECIPES)],
        programs=programs,jobs=jobs,sources=sources,binary=str(binary),runtime_dirs=native["runtime_dirs"],
        context=frozen["context"],frozen_independent_profile=e2a.receipt(profile),native_asset_descriptor=e2a.receipt(native_manifest),
        protocol=dict(fresh_processes_per_program=4,trials=12,replays_per_trial=8,all_samples_retained=True,
            source_profile_frozen_before_new_programs=True,complete_target_times_used_for_calibration=False,
            precision="F32",cpu_threads=1,cuda_graphs="disabled",serial=True,observer_controls_run=False,
            physical_allocator_or_capacity_qualified=False,autoregressive_generation=False,
            decode_scope="one invocation per layer from supplied existing KV; no token-generation gate"))
    e2a.write(out/"manifest.json",manifest)
    e2a.write(out/"frozen-barrier.json",dict(frozen_at_utc=e2a.utc(),frozen_sha256=e2a.sha(out/"profile-frozen.json"),
        predictions_sha256=e2a.sha(out/"predictions.json"),heldout_read=False))
    return dict(programs=len(programs),processes=len(jobs),native_timing_started=False,out=str(out))


def evaluate(out):
    out=Path(out).resolve();manifest=e2a.read(out/"manifest.json");barrier=e2a.read(out/"frozen-barrier.json")
    predictions=e2a.read(out/"predictions.json");executions=e2a.read(out/"heldout-executions.json")
    e2a.require(e2a.sha(out/"profile-frozen.json")==barrier["frozen_sha256"] and
                e2a.sha(out/"predictions.json")==barrier["predictions_sha256"],"freeze barrier changed")
    e2a.require(len(executions)==48,"every declared process needs an outcome")
    for source in manifest["sources"]:
        e2a.require(e2a.sha(source["path"])==source["sha256"],"bound source/runtime changed during collection")
    rows=[];groups=[]
    for case in MATRIX:
        candidates=[]
        for strategy in range(3):
            key=case["id"]+f"-s{strategy}"
            pred=next(p for p in predictions if p["program_key"]==key)
            price=float(Fraction(pred["predicted_ns"]))
            results=[r for r in executions if r["job"]["program_key"]==key]
            for execution in results:
                e2a.require(execution["started_at_utc"]>=barrier["frozen_at_utc"],"measurement predates predictions")
                if execution["status"]!="completed":
                    rows.append(dict(workload=case["id"],strategy=strategy,job=execution["job"]["id"],status=execution["status"],error=execution.get("error")))
                    continue
                native=e2a.read(execution["native"]["path"])
                values=[r["wall_ns"]/r["repeats"] for r in native["trials"]]
                mean=statistics.mean(values);cv=statistics.stdev(values)/mean;ape=abs(price-mean)/mean
                rows.append(dict(workload=case["id"],strategy=strategy,job=execution["job"]["id"],round=execution["job"]["round"],
                    status="completed",real_ns=mean,predicted_ns=price,ape=ape,cv=cv,mean_gate=ape<=.10,cv_gate=cv<=.10,
                    numerical_pass=execution["numerical"]["passed"],max_absolute_error=execution["numerical"]["maximum_absolute_error"]))
            usable=[r for r in rows if r["workload"]==case["id"] and r["strategy"]==strategy and r["status"]=="completed"]
            candidates.append(dict(strategy=strategy,predicted_ns=price,measured_mean_ns=statistics.mean(r["real_ns"] for r in usable) if len(usable)==4 else None,
                completed_processes=len(usable),process_mean_and_cv_conditions_passed=len(usable)==4 and all(r["mean_gate"] and r["cv_gate"] for r in usable)))
        complete=all(c["measured_mean_ns"] is not None for c in candidates)
        selected=min(candidates,key=lambda c:(c["predicted_ns"],c["strategy"]))
        actual=min(candidates,key=lambda c:(c["measured_mean_ns"],c["strategy"])) if complete else None
        groups.append(dict(workload=case["id"],layers=case["layers"],hidden=case["hidden"],tokens=case["tokens"],history=case["history"],
            candidates=candidates,complete=complete,predicted_winner=selected["strategy"],real_winner=actual["strategy"] if actual else None,
            point_regret=selected["measured_mean_ns"]/actual["measured_mean_ns"]-1 if complete else None,
            full_original_quality_qualification="not_assessed_observer_controls_not_run",observer_controls_not_run=True))
    completed=[r for r in rows if r["status"]=="completed"]
    summary=dict(schema="multilayer-heldout-results-1",completed_at_utc=e2a.utc(),matrix=list(MATRIX),programs=12,
        declared_processes=48,completed_processes=len(completed),failed_processes=48-len(completed),
        numerical_processes_passed=sum(r["numerical_pass"] for r in completed),
        max_absolute_port_error=max((r["max_absolute_error"] for r in completed),default=None),
        median_process_mean_ape=statistics.median(r["ape"] for r in completed) if completed else None,
        maximum_process_mean_ape=max((r["ape"] for r in completed),default=None),
        process_mean_gate_passes=sum(r["mean_gate"] for r in completed),process_cv_gate_passes=sum(r["cv_gate"] for r in completed),
        matching=e2a.read(out/"matching.json"),groups=groups,processes=rows,
        groups_selected_observed_winner=sum(g["complete"] and g["predicted_winner"]==g["real_winner"] for g in groups),
        source_profile_sha256=e2a.sha(out/"profile-frozen.json"),predictions_sha256=e2a.sha(out/"predictions.json"),
        freeze_before_every_completed_target=True,full_original_E2_quality_requalified=False,
        bound_sources_and_runtimes_unchanged_after_collection=True,
        limitations=["single round supplied inputs; no autoregressive generation", "no new observer control matrix",
            "serial F32 Dense only; not concurrent TP", "preallocated native backing; logical eviction only",
            "no real MoE or UMA measurements", "private CUDA workspace unknown"])
    e2a.write(out/"summary.json",summary)
    lines=["# Connected multilayer Dense native validation", "", "This is a new-composition heldout extension, not a replacement for the original frozen experiment.","",
        "| Workload | Predicted winner | Measured winner | Point regret |", "|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['workload']} | S{g['predicted_winner']} | {('S'+str(g['real_winner'])) if g['complete'] else 'unknown'} | {g['point_regret']:.2%} |" if g["complete"] else f"| {g['workload']} | S{g['predicted_winner']} | unknown | unknown |")
    lines.extend(["",f"Completed numerical checks: {summary['numerical_processes_passed']}/48 processes. Process-mean APE median/max: {summary['median_process_mean_ape']:.2%}/{summary['maximum_process_mean_ape']:.2%}.",
        f"Old mean-error threshold passed {summary['process_mean_gate_passes']}/48; old CV threshold passed {summary['process_cv_gate_passes']}/48. The 4-process extension does not rerun the original observer-quality matrix.","",
        "All predictions use exact semantic coordinates in the pre-existing independently frozen component table. No complete multilayer target timing is used to fit a parameter.","",
        "The Decode case is a single connected pass through four layers using supplied old KV. It is not a measured multi-round token-generation loop. Native allocation is preallocated and EVICT remains logical. Real MoE/cache, UMA and asynchronous pipeline performance remain outside this measurement.",""])
    with (out/"REPORT.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(lines))
    return {k:summary[k] for k in ("completed_processes","numerical_processes_passed","median_process_mean_ape","maximum_process_mean_ape","groups_selected_observed_winner")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","collect","evaluate"));parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--profile",type=Path);parser.add_argument("--native-manifest",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":
        parser.error("prepare requires explicit --profile and --native-manifest") if not(args.profile and args.native_manifest) else None
        result=prepare(args.out,args.profile,args.native_manifest)
    elif args.command=="collect":result=e2a.collect(args.out,"heldout")
    else:result=evaluate(args.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
