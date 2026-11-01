"""One predeclared R256/R4096 sampling-window test, without parameter fitting.

Twelve adjacent independent-process calibration jobs precede a choice barrier.
Twenty-four new tiny heldouts always run, including if calibration rejects the
candidate. An R4096 batch mean is a different response contract, not an isolated
copy latency or a replacement for the retained R256 failures.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

from llm_infer_model.performance.transfer_response import inclusive_response_ns
from profiling.dac_accuracy_repair import campaign

read, write, sha, receipt, require = campaign.read, campaign.write, campaign.sha, campaign.receipt, campaign.require
ROOT = Path(__file__).resolve().parents[2]
MODES = ("pageable", "registered")
TINY = (64, 1024, 2048, 16384)
FROZEN = ("action-profile-frozen.json", "transfer-pageable-frozen.json", "transfer-registered-frozen.json", "predictions.json")

def preparation(request, repeats):
    result=copy.deepcopy(request);result["sampling"]["repeats"]=repeats
    return result

def prepare(out, source):
    out, source=Path(out).resolve(),Path(source).resolve()
    require(not out.exists(),"fresh window output directory required")
    parent=read(source/"manifest.json");campaign.check_sources(parent)
    barrier=read(source/"frozen-barrier.json")
    for name in FROZEN:require(sha(source/name)==barrier["files"][name],"original frozen model changed")
    out.mkdir(parents=True);manifest=copy.deepcopy(parent)
    archived=out/"executed_sources"/Path(__file__).resolve().relative_to(ROOT)
    archived.parent.mkdir(parents=True);shutil.copy2(__file__,archived)
    manifest["sources"].append(dict(source=receipt(__file__),archived=receipt(archived)))
    binary=out/"executed_binary"/Path(parent["binary"]["path"]).name
    binary.parent.mkdir();shutil.copy2(parent["binary"]["path"],binary);manifest["binary"]=receipt(binary)
    for name in (*FROZEN,"frozen-barrier.json"):shutil.copy2(source/name,out/name)
    template=next(j["request"] for j in parent["jobs"] if j["split"]=="transfer-profile")
    jobs=[]
    # Adjacent AB/BA short/long processes; neither process is a within-process control.
    for round_id in range(3):
        modes=MODES if round_id%2==0 else tuple(reversed(MODES))
        for memory in modes:
            pair_order=(256,4096) if (round_id+MODES.index(memory))%2==0 else (4096,256)
            for repeats in pair_order:
                request=preparation(template,repeats);request["memory_mode"]=memory
                request["order"]="ABBA" if round_id%2==0 else "BAAB"
                jobs.append(dict(id=f"window-cal-{memory}-r{round_id}-R{repeats}",split="transfer-profile",
                    program_key="d2h-transfer-profile-4096",memory_mode=memory,round=round_id,repeats=repeats,request=request))
    for round_id in range(3):
        sizes=TINY if round_id%2==0 else tuple(reversed(TINY))
        for size in sizes:
            for memory in (MODES if round_id%2==0 else tuple(reversed(MODES))):
                request=preparation(template,4096);request["memory_mode"]=memory
                request["order"]="ABBA" if round_id%2==0 else "BAAB"
                jobs.append(dict(id=f"window-heldout-{memory}-r{round_id}-{size}",split="transfer-heldout",
                    program_key=f"d2h-transfer-heldout-{size}",memory_mode=memory,round=round_id,repeats=4096,request=request))
    keys={j["program_key"] for j in jobs};manifest["programs"]={k:v for k,v in parent["programs"].items() if k in keys}
    require(set(manifest["programs"])==keys,"required original mathematical program/reference absent")
    manifest.update(schema="dac-accuracy-repair-window-plan-1",prepared_at_utc=campaign.e2a.utc(),jobs=jobs,
        source_campaign=receipt(source/"manifest.json"),source_summary=receipt(source/"summary.json"),
        source_frozen_barrier=receipt(source/"frozen-barrier.json"))
    manifest["protocol"]=dict(total_processes=36,counts={"transfer-profile":12,"transfer-heldout":24,"profile":0,"dense":0},
        per_child_timeout_seconds=300,retries=0,calibration_bytes=4096,candidate_repeats=4096,short_repeats=256,
        trials=8,warmup_min_ms=500,order="adjacent fresh-process R256/R4096 AB/BA; ABBA/BAAB bare/stamped within each process",
        tiny_heldouts=list(TINY),candidate_count=1,parameter_refit=False,
        calibration_acceptance=dict(long_cv_passes_required=6,long_observer_passes_required=6,
            maximum_trial_cv=.10,maximum_observer_effect=.05,maximum_per_mode_aggregate_mean_shift=.05),
        heldout_runs_even_if_calibration_rejects=True,original_R256_failures_retained=True,
        response_scope="means of completed fixed-input warm batches, not individual-copy latency distributions",
        formula_scope="unchanged frozen R256 formula projected onto a separately validated R4096 context; no context match silently asserted",
        combined_policy_scope="different sampling policy: tiny four sizes use newR4096, four large sizes retain originalR256")
    write(out/"manifest.json",manifest)
    return dict(out=str(out),declared=36,calibration=12,heldout=24)

def rows(out, split):
    out=Path(out);manifest=read(out/"manifest.json");result=[]
    for execution in read(out/(split+"-executions.json")):
        if execution["status"]!="completed":continue
        require(sha(execution["native"]["path"])==execution["native"]["sha256"],"native raw changed")
        native=read(execution["native"]["path"]);job=execution["job"]
        bare=[r["wall_ns"]/r["repeats"] for r in native["trials"] if not r["actual_stamped"]]
        stamps=[r["wall_ns"]/r["repeats"] for r in native["trials"] if r["actual_stamped"]]
        result.append(dict(job=job["id"],bytes=manifest["programs"][job["program_key"]]["bytes"],
            memory_mode=job["memory_mode"],round=job["round"],repeats=job["repeats"],mean_ns=statistics.mean(bare),
            cv=statistics.stdev(bare)/statistics.mean(bare),trial_ns=bare,
            observer_effect=statistics.mean(stamps)/statistics.mean(bare)-1,
            preparation_contract=native["preparation_contract"],raw=execution["native"]))
    return result

def calibration_decision(calibration):
    long=[r for r in calibration if r["repeats"]==4096];short=[r for r in calibration if r["repeats"]==256]
    groups=[];pairs=[]
    for mode in MODES:
        a=[r for r in short if r["memory_mode"]==mode];b=[r for r in long if r["memory_mode"]==mode]
        shift=statistics.mean(r["mean_ns"] for r in b)/statistics.mean(r["mean_ns"] for r in a)-1 if len(a)==len(b)==3 else None
        groups.append(dict(memory_mode=mode,short_completed=len(a),long_completed=len(b),aggregate_mean_shift=shift,
            mean_shift_pass=shift is not None and abs(shift)<=.05))
        for round_id in range(3):
            aa=[r for r in a if r["round"]==round_id];bb=[r for r in b if r["round"]==round_id]
            if len(aa)==len(bb)==1:pairs.append(dict(memory_mode=mode,round=round_id,
                short_mean_ns=aa[0]["mean_ns"],long_mean_ns=bb[0]["mean_ns"],individual_mean_shift=bb[0]["mean_ns"]/aa[0]["mean_ns"]-1,
                short_cv=aa[0]["cv"],long_cv=bb[0]["cv"],short_observer_effect=aa[0]["observer_effect"],long_observer_effect=bb[0]["observer_effect"],
                scope="adjacent different processes; temporal/process state remains a possible confound"))
    cvs=sum(r["cv"]<=.1 for r in long);obs=sum(abs(r["observer_effect"])<=.05 for r in long)
    accepted=len(calibration)==12 and len(long)==6 and cvs==6 and obs==6 and all(g["mean_shift_pass"] for g in groups)
    return dict(candidate_repeats=4096,accepted=accepted,completed_calibration_processes=len(calibration),expected_calibration_processes=12,
        long_cv_passes=cvs,long_observer_passes=obs,long_processes=len(long),modes=groups,individual_adjacent_pairs=pairs,
        qualification_allowed=accepted,no_larger_candidate_admitted=True,
        conclusion="candidate accepted for independent heldout validation" if accepted else "candidate insufficient on calibration; predeclared heldouts remain diagnostic and cannot qualify")

def freeze(out):
    out=Path(out);require(not(out/"transfer-heldout-started.json").exists(),"choice cannot be frozen after heldout starts")
    decision=calibration_decision(rows(out,"transfer-profile"));decision.update(frozen_at_utc=campaign.e2a.utc(),heldout_read=False,parameter_refit=False)
    write(out/"window-choice.json",decision)
    predictions=[]
    for mode in MODES:
        profile=read(out/f"transfer-{mode}-frozen.json")
        for size in TINY:
            predictions.append(dict(memory_mode=mode,bytes=size,predicted_ns=inclusive_response_ns(profile["parameters"],size),
                frozen_profile=receipt(out/f"transfer-{mode}-frozen.json"),parameter_refit=False,
                formula_from_repeats=256,measurement_repeats=4096,scope="explicit unchanged-formula projection across sampling windows"))
    write(out/"window-predictions.json",dict(frozen_at_utc=campaign.e2a.utc(),heldout_read=False,predictions=predictions))
    files=(*FROZEN,"window-choice.json","window-predictions.json")
    write(out/"window-choice-barrier.json",dict(frozen_at_utc=campaign.e2a.utc(),heldout_read=False,files={n:sha(out/n) for n in files}))
    return decision

def heldout(out):
    out=Path(out);barrier=read(out/"window-choice-barrier.json")
    for name,digest in barrier["files"].items():require(sha(out/name)==digest,"window choice, formula or original parameters changed")
    return campaign.collect(out,"transfer-heldout")

def qualify(processes, sizes, count, calibration_ok):
    known=len(processes)==count
    bysize=[]
    for size in sizes:
        group=[r for r in processes if r["bytes"]==size]
        if len(group)!=3:continue
        mean=statistics.mean(r["mean_ns"] for r in group);pred=group[0]["predicted_ns"]
        bysize.append(dict(bytes=size,observed_ns=mean,predicted_ns=pred,ape=abs(pred-mean)/mean,
            between_process_cv=statistics.stdev(r["mean_ns"] for r in group)/mean))
    maximum=max((r["ape"] for r in bysize),default=None);maximum_process=max((r["ape"] for r in processes),default=None)
    p90=campaign.percentile([abs(r["predicted_ns"]-v)/v for r in processes for v in r["trial_ns"]],.9) if processes else None
    cvs=sum(r["cv"]<=.1 for r in processes);obs=sum(abs(r["observer_effect"])<=.05 for r in processes)
    original=known and len(bysize)==len(sizes) and maximum<=.1 and cvs==count and obs==count
    strict=original and maximum_process<=.1 and p90<=.2
    return dict(expected_processes=count,completed_processes=len(processes),sizes=bysize,maximum_size_mean_ape=maximum,
        maximum_process_ape=maximum_process,trial_p90_ape=p90,cv_passes=cvs,observer_passes=obs,
        recorded_original_size_mean_predicates=original,recorded_strict_process_mean_predicates=strict,
        qualification_allowed_by_calibration=calibration_ok,qualified_original_size_mean_gate=calibration_ok and original,
        qualified_strict_process_mean_gate=calibration_ok and strict)

def csv_new(path, values):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator="\n");writer.writeheader();writer.writerows(values)

def evaluate(out):
    out=Path(out);m=read(out/"manifest.json");choice=read(out/"window-choice.json");barrier=read(out/"window-choice-barrier.json")
    for name,digest in barrier["files"].items():require(sha(out/name)==digest,"frozen window evidence changed")
    source=Path(m["source_summary"]["path"]);require(sha(source)==m["source_summary"]["sha256"],"original153-process summary changed")
    old=read(source);preds=read(out/"window-predictions.json")["predictions"]
    lookup={(p["memory_mode"],p["bytes"]):p["predicted_ns"] for p in preds};new=rows(out,"transfer-heldout")
    for row in new:
        row["predicted_ns"]=lookup[(row["memory_mode"],row["bytes"])];row["ape"]=abs(row["predicted_ns"]-row["mean_ns"])/row["mean_ns"]
    results={}
    for mode in MODES:
        tiny=[r for r in new if r["memory_mode"]==mode]
        large=[dict(r,repeats=256) for r in old["D2H"][mode]["processes"] if r["bytes"] not in TINY]
        combined=tiny+large
        results[mode]=dict(new_R4096_tiny=qualify(tiny,TINY,12,choice["accepted"]),
            combined_policy=qualify(combined,sorted(TINY+tuple(sorted({r["bytes"] for r in large}))),24,choice["accepted"]),
            original_R256_qualification_unchanged=old["D2H"][mode]["qualified_original_size_mean_gate"])
    executions=read(out/"transfer-profile-executions.json")+read(out/"transfer-heldout-executions.json")
    result=dict(schema="dac-accuracy-repair-window-result-1",evaluated_at_utc=campaign.e2a.utc(),
        declared=36,completed=sum(e["status"]=="completed" for e in executions),failed=sum(e["status"]!="completed" for e in executions),
        calibration=choice,D2H=results,parameter_refit=False,original_results_modified=False,
        passive_state_used_as_future_input=False,all_owned_children_exited=all(e.get("process_exited_verified") for e in executions),
        context_warning="newR4096 block-mean contract; combined policy reuses originalR256 large rows and newR4096 tiny rows, not a rerun or qualification of original153-process campaign",
        window_predictions=receipt(out/"window-predictions.json"),barrier=receipt(out/"window-choice-barrier.json"),source_summary=m["source_summary"])
    flat=[]
    for row in rows(out,"transfer-profile")+new:
        flat.append(dict(job=row["job"],bytes=row["bytes"],memory_mode=row["memory_mode"],round=row["round"],repeats=row["repeats"],
            mean_ns=row["mean_ns"],cv=row["cv"],observer_effect=row["observer_effect"],predicted_ns=row.get("predicted_ns"),ape=row.get("ape"),raw_path=row["raw"]["path"],raw_sha256=row["raw"]["sha256"]))
    csv_new(out/"process_metrics.csv",flat)
    block_rows=[];state_rows=[]
    for execution in executions:
        if execution["status"]!="completed":continue
        native=read(execution["native"]["path"])
        for row in native["trials"]:
            block_rows.append(dict(job=execution["job"]["id"],trial=row["trial"],block=row["block"],actual_stamped=row["actual_stamped"],
                repeats=row["repeats"],response_ns=row["wall_ns"]/row["repeats"],wall_ns=row["wall_ns"],
                copy_call_ns=row["copy_call_ns"],gpu_sync_ns=row["gpu_sync_ns"],cpu_sync_ns=row["cpu_sync_ns"],raw_sha256=execution["native"]["sha256"]))
            state_rows.append(dict(job=execution["job"]["id"],trial=row["trial"],block=row["block"],preblock_state_json=json.dumps(row["preblock_state"],sort_keys=True),raw_sha256=execution["native"]["sha256"]))
    if block_rows:csv_new(out/"block_metrics.csv",block_rows);csv_new(out/"preblock_states.csv",state_rows)
    write(out/"summary.json",result);return result

class Tee:
    def __init__(self,*targets):self.targets=targets
    def write(self,text):
        for target in self.targets:target.write(text);target.flush()
    def flush(self):
        for target in self.targets:target.flush()

def logged(out,name,function):
    with (Path(out)/(name+".log")).open("x",encoding="utf-8",newline="\n") as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)):
        value=function();print(json.dumps(value,allow_nan=False),flush=True);return value

def self_test():
    values=[dict(memory_mode=mode,round=i,repeats=r,mean_ns=20.,cv=.05,observer_effect=.01) for mode in MODES for i in range(3) for r in (256,4096)]
    require(calibration_decision(values)["accepted"],"valid calibration rejected")
    changed=copy.deepcopy(values);next(x for x in changed if x["repeats"]==4096)["cv"]=.11
    require(not calibration_decision(changed)["accepted"],"one CV failure accepted")
    changed=copy.deepcopy(values)
    for row in changed:
        if row["memory_mode"]=="registered" and row["repeats"]==4096:row["mean_ns"]=22.
    require(not calibration_decision(changed)["accepted"],"aggregate mean shift accepted")
    require(not calibration_decision(values[:-1])["accepted"],"missing process accepted")
    return dict(passed=True,calibration_acceptance_branches=4,native_acquisition=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("all","prepare","calibration","freeze","heldout","evaluate","self-test"));p.add_argument("--out",type=Path);p.add_argument("--source",type=Path,default=ROOT/"build/dac-accuracy-repair-20261026/campaign-v1");a=p.parse_args()
    if a.command=="self-test":print(json.dumps(self_test()));return
    require(a.out is not None,"explicit new output required")
    if a.command in ("prepare","all"):print(json.dumps(prepare(a.out,a.source)),flush=True)
    if a.command in ("calibration","all"):logged(a.out,"calibration",lambda:campaign.collect(a.out,"transfer-profile"))
    if a.command in ("freeze","all"):logged(a.out,"freeze",lambda:freeze(a.out))
    if a.command in ("heldout","all"):logged(a.out,"heldout",lambda:heldout(a.out))
    if a.command in ("evaluate","all"):logged(a.out,"evaluate",lambda:evaluate(a.out))
if __name__=="__main__":main()
