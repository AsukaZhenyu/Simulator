"""Calibrate independent action intervals, freeze, then test new complete targets.

This bounded campaign uses H512/P64 one/two-layer S0/S1/S2 only. The optional
preceding-service model is an empirical context model, not a claim that the
history axis identifies every physical cause. Complete target times never fit
prices. All raw failures, clocks, numerical ports and source bindings survive.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from llm_infer_model.performance.context_response import ContextResponseTable, context_coordinate
from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable, coordinate, digest
from profiling.dac_validation import e2a
from profiling.dac_followup.task1 import TRAIN, HELDOUT, byte_reference
from profiling.dac_followup.transfer_fit import fit_from_training, predict
from profiling.highlevel_accuracy.bridge import export_plan, reference_ports
from profiling.highlevel_accuracy.multilayer import build_multilayer_case

ROOT=Path(__file__).resolve().parents[2]
RECIPES=(("cpu","cpu","cpu"),("gpu","gpu","gpu"),("gpu","cpu","gpu"))
read,write,sha,receipt,require=e2a.read,e2a.write,e2a.sha,e2a.receipt,e2a.require


def sample(mode="full",repeats=8):
    return dict(e2a.request(mode),repeats=repeats)


def request(mode,sampling=None,**kwargs):
    return dict(schema="dac-refinement-request-1",mode=mode,sampling=sampling or sample(),**kwargs)


def prepare(out,binary):
    out,binary=Path(out).resolve(),Path(binary).resolve()
    if out.exists():
        require(not(out/"manifest.json").exists() and not list(out.glob("*-started.json")),
                "only an incomplete software preparation may be resumed")
    else:out.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/"build/dac-followup-20261025/multilayer-v1/campaign-v1/manifest.json")
    runtime_dirs=old["runtime_dirs"]
    context=dict(old["context"],binary_sha256=sha(binary),
        boundary="unchanged completed Executor.action; response history explicitly priced",
        response_context="warm fixed-input cyclic serial; no target timing fitted")
    programs={};profile_groups={};seen=set();jobs=[]
    for layers in (1,2):
        for split,seed in (("profile",2026102601),("heldout",2026102602)):
            cohort=[];refs=[]
            for strategy,devices in enumerate(RECIPES):
                key=f"l{layers}-h512-p64-s{strategy}-{split}"
                pp,rp=out/"programs"/(key+".json"),out/"references"/(key+".json")
                if pp.exists() and rp.exists():
                    p,oracle=read(pp),read(rp);require(p["seed"]==seed,"resumed software seed differs")
                else:
                    require(not pp.exists() and not rp.exists(),"partial JSON requires explicit inspection")
                    plan=build_multilayer_case(out/"inputs"/key,layers=layers,hidden=512,intermediate=1024,
                        heads=8,kv_heads=2,head_dim=64,tokens=64,history=0,threads=1,devices=devices)
                    p=export_plan(plan.compile(),seed=seed);oracle=reference_ports(p)
                    write(pp,p);write(rp,oracle)
                programs[key]=dict(program=receipt(pp),reference=receipt(rp),
                    layers=layers,strategy=strategy,split=split)
                cohort.append(p);refs.append(oracle)
                if split=="profile":
                    indices=[]
                    for i in range(len(p["actions"])):
                        c=context_coordinate(p,i,context);h=digest(c)
                        if h not in seen:seen.add(h);indices.append(i)
                    if indices:profile_groups[key]=indices
            require(all(p["initial_values"]==cohort[0]["initial_values"] for p in cohort),"candidate inputs/parent weights/KV differ")
            goals=cohort[0]["graph"]["scenario"]["goal"]["output_locations"]
            require(all(p["graph"]["scenario"]["goal"]["output_locations"]==goals for p in cohort),"candidate final goals differ")
            for tid in goals:
                baseline=np.asarray(refs[0].get(tid,cohort[0]["initial_values"].get(tid)))
                for p,reference in zip(cohort[1:],refs[1:]):
                    actual=np.asarray(reference.get(tid,p["initial_values"].get(tid)))
                    require(np.allclose(actual,baseline,atol=5e-5,rtol=5e-5),"candidate mathematical goals differ")
    for key,indices in profile_groups.items():
        for round_id in range(3):
            jobs.append(dict(id=f"profile-r{round_id}-{key}",split="profile",program_key=key,round=round_id,
                request=request("matched-actions",sample("full"),indices=indices)))
    # Check heldout semantic coverage before any acquisition. Identifiers/seeds
    # are absent from coordinates; shape/history unknowns are never approximated.
    coverage=[]
    for key,binding in programs.items():
        if binding["split"]!="heldout":continue
        p=read(binding["program"]["path"])
        missing=[i for i in range(len(p["actions"])) if digest(context_coordinate(p,i,context)) not in seen]
        require(not missing,"heldout preceding-service coordinates absent from independent plan")
        coverage.append(dict(program_key=key,actions=len(p["actions"]),all_coordinates_planned=True))
    for round_id in range(4):
        for layers in ((1,2) if round_id%2==0 else (2,1)):
            for strategy in ((0,1,2),(1,2,0),(2,0,1),(2,1,0))[round_id]:
                key=f"l{layers}-h512-p64-s{strategy}-heldout"
                jobs.append(dict(id=f"heldout-r{round_id}-{key}",split="heldout",program_key=key,round=round_id,
                    request=request("steady",sample("full"))))
                jobs.append(dict(id=f"observer-r{round_id}-{key}",split="observer",program_key=key,round=round_id,
                    request=request("observer",sample("full"),order="ABBA" if round_id%2==0 else "BAAB")))
    for split,sizes in (("transfer-profile",TRAIN),("transfer-heldout",HELDOUT)):
        for round_id in range(3):
            order=list(sizes[round_id:]+sizes[:round_id])
            if round_id%2:order.reverse()
            for size in order:
                key=f"d2h-{split}-{size}"
                if key not in programs:
                    p=dict(format_version="steady-component-program-1",kind="transfer",device="gpu",threads=1,cuda_device=0,
                        seed=2026102603 if split=="transfer-profile" else 2026102604,
                        shape=dict(bytes=size,src="vram",dst="dram"),allocation_budget_bytes=1<<30)
                    p["program_sha256"]=digest(p)
                    pp,rp=out/"programs"/(key+".json"),out/"references"/(key+".json")
                    write(pp,p);write(rp,byte_reference(p["seed"],size));programs[key]=dict(program=receipt(pp),reference=receipt(rp),bytes=size,split=split)
                jobs.append(dict(id=f"{split}-r{round_id}-{size}",split=split,program_key=key,round=round_id,
                    request=request("transfer-phases",sample("action",256),order="ABBA" if round_id%2==0 else "BAAB")))
    sourcepaths=[Path(__file__),ROOT/"modeling/llm_infer_model/performance/context_response.py",
        ROOT/"modeling/llm_infer_model/performance/reusable_action_costs.py",
        ROOT/"profiling/dac_followup/transfer_fit.py",ROOT/"profiling/highlevel_accuracy/bridge.py",
        ROOT/"profiling/highlevel_accuracy/multilayer.py",ROOT/"profiling/dac_validation/e2a.py",
        ROOT/"profiling/native/steady_costs/probe.cpp",ROOT/"profiling/native/steady_costs/components.hpp",
        ROOT/"profiling/dac_refinement/native/refinement_probe.cpp",ROOT/"profiling/dac_refinement/native/CMakeLists.txt",binary]
    sourcepaths.extend((ROOT/"profiling/native/steady_costs/build-release-v1").glob("*.inc"))
    runtime={str(p.resolve()):sha(p) for folder in runtime_dirs for p in Path(folder).glob("*.dll")}
    manifest=dict(schema="dac-context-refinement-plan-1",prepared_at_utc=e2a.utc(),context=context,
        binary=receipt(binary),runtime_dirs=runtime_dirs,runtime_sha256=runtime,sources=[receipt(p) for p in sourcepaths],
        programs=programs,jobs=jobs,coverage=coverage,
        protocol=dict(quality=e2a.QUALITY,independent_profile_processes_per_group=3,heldout_processes_per_program=4,
            observer_processes_per_program=4,profile_groups=profile_groups,
            total_processes=len(jobs),counts={s:sum(j["split"]==s for j in jobs) for s in ("profile","heldout","observer","transfer-profile","transfer-heldout")},
            timing="fixed-input cyclic serial completion; construction/reset/conditioning excluded",
            primary="preceding-service-conditioned independently timed action intervals",
            ablation="fresh hot-batch responses measured in the same calibration processes",
            exact_same_template_calibration="calibration includes independent one/two-layer templates where a new history coordinate requires it; not unseen-layer extrapolation",
            full_target_time_used_for_fit=False,no_residual_fitting=True,all_failures_retained=True,
            transfer_fit="training-only nonnegative affine/one hinge; old sizes/candidates/replay/trial count; bare blocks only",
            transfer_stamps="diagnosis only; GGML copy API may include internal staging and sync, not PCIe active time",
            original_results_unchanged=True))
    write(out/"manifest.json",manifest)
    return dict(prepared=str(out),**manifest["protocol"]["counts"],total=len(jobs),new_native_timing_started=False)


def verify(program,oracle,req,native,pid):
    require(native["process_id"]==pid and native["request"]==req and native["program_sha256"]==program["program_sha256"],"native identity/request changed")
    affinity=native["main_thread_affinity"]
    require(affinity["requested_mask"]==affinity["observed_mask"]==affinity["final_mask"]==4 and affinity["restored"],"owned affinity contract differs")
    hz=native["clock"]["frequency_hz"]
    if req["mode"]=="transfer-phases":
        v=native["component_output"]
        require(v["full_byte_equality"] and v["output_sha256"]==v["source_sha256"]==oracle["sha256"] and v["checked_bytes"]==oracle["bytes"],"transfer byte oracle failed")
        require(len(native["trials"])==req["sampling"]["trials"]*4,"transfer block count differs")
        groups=[native["trials"]];maximum=0.
    else:
        v=native["verification_native"]
        require(v["copy_values_equal"] and v["actions"]==program["actions"] and native["native_repeat_consistency"]["passed"],"native COPY/action/replay contract failed")
        require(set(v["ports"])==set(oracle),"port inventory differs")
        maximum=0.
        for key,wanted in oracle.items():
            actual=np.asarray(v["ports"][key]);wanted=np.asarray(wanted)
            require(actual.shape==wanted.shape and np.all(np.isfinite(actual)) and np.allclose(actual,wanted,atol=5e-5,rtol=5e-5),"native numerical mismatch "+key)
            maximum=max(maximum,float(np.max(np.abs(actual-wanted))))
        groups=[native["trials"]]
        for group in native.get("groups",[]):groups.extend((group["trials"],group["hot_trials"]))
    for rows in groups:
        for row in rows:
            require(row["repeats"]==req["sampling"]["repeats"],"replay count differs")
            if "segments" in row:
                ticks=sum(s["end_tick"]-s["begin_tick"] for s in row["segments"])
                require(all(s["end_tick"]>=s["begin_tick"] for s in row["segments"]),"negative target interval")
            else: ticks=row["end_tick"]-row["begin_tick"]
            require(ticks>=0 and abs(ticks*1e9/hz-row["wall_ns"])<=1,"QPC clock ledger differs")
    return dict(passed=True,maximum_absolute_error=maximum)


def collect(out,split):
    from profiling.state_calibration import run
    out=Path(out).resolve();m=read(out/"manifest.json")
    for source in m["sources"]:require(sha(source["path"])==source["sha256"],"bound refinement source changed "+source["path"])
    for path,h in m["runtime_sha256"].items():require(sha(path)==h,"bound runtime changed")
    barrier=None
    if split in {"heldout","observer"}:
        barrier=read(out/"frozen-barrier.json")
        for file in ("response-profile-frozen.json","predictions.json"):require(sha(out/file)==barrier[file],"frozen response or predictions changed")
    if split=="transfer-heldout":
        barrier=read(out/"transfer-barrier.json");require(sha(out/"transfer-frozen.json")==barrier["sha256"],"frozen transfer model changed")
    jobs=[j for j in m["jobs"] if j["split"]==split];rows=[]
    if not(out/(split+"-started.json")).exists():write(out/(split+"-started.json"),dict(started=e2a.utc(),manifest=receipt(out/"manifest.json"),barrier=barrier))
    with run.measurement_lock():
        for job in jobs:
            folder=out/"jobs"/job["id"]
            if(folder/"execution.json").exists():rows.append(read(folder/"execution.json"));continue
            folder.mkdir(parents=True,exist_ok=False);binding=m["programs"][job["program_key"]]
            for name in ("program","reference"):require(sha(binding[name]["path"])==binding[name]["sha256"],"prepared input changed")
            pp,oracle=read(binding["program"]["path"]),read(binding["reference"]["path"])
            req,output=folder/"request.json",folder/"result.json";write(req,job["request"])
            command=[m["binary"]["path"],binding["program"]["path"],str(req),str(output)]
            record=dict(job=job,started_at_utc=e2a.utc(),command=command,status="failed",barrier=barrier,program=binding)
            process=None;handles=[]
            try:
                env=run.child_environment(m["runtime_dirs"]);env["OPENBLAS_NUM_THREADS"]="1"
                process,stdout,stderr,affinity=run.launch(command,env,folder/"stdout.txt",folder/"stderr.txt",4);handles=[stdout,stderr]
                record.update(process_id=process.pid,affinity=affinity)
                code=process.wait(timeout=240);record.update(returncode=code,forced_termination=False,process_exited_verified=True)
                require(code==0,"native process failed")
                record["numerical"]=verify(pp,oracle,job["request"],read(output),process.pid)
                record.update(status="completed",native=receipt(output))
            except BaseException as error:
                record["error"]=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    record["forced_termination"]=True
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles:handle.close()
                record["completed_at_utc"]=e2a.utc();write(folder/"execution.json",record)
            rows.append(record);print(dict(job=job["id"],status=record["status"],error=record.get("error")),flush=True)
    target=out/(split+"-executions.json")
    if not target.exists():write(target,rows)
    return dict(split=split,declared=len(jobs),completed=sum(x["status"]=="completed" for x in rows),failed=sum(x["status"]!="completed" for x in rows))


def freeze(out):
    out=Path(out).resolve();m=read(out/"manifest.json")
    require(not(out/"heldout-started.json").exists(),"response freeze must precede complete targets")
    samples={"hot":{},"matched":{}};coords={"hot":{},"matched":{}};evidence={"hot":{},"matched":{}}
    for ex in read(out/"profile-executions.json"):
        require(ex["status"]=="completed","profile failed; parameter remains unknown")
        p=read(ex["program"]["program"]["path"]);n=read(ex["native"]["path"])
        for group in n["groups"]:
            i=group["action_index"]
            for mode,values in (("hot",group["hot_trials"]),("matched",group["trials"])):
                c=(coordinate if mode=="hot" else context_coordinate)(p,i,m["context"]);key=digest(c)
                coords[mode][key]=c;samples[mode].setdefault(key,[]).extend(Fraction(r["wall_ns"],r["repeats"]) for r in values)
                evidence[mode].setdefault(key,[]).append(dict(native=ex["native"],action_index=i,process_id=ex["process_id"],split="profile"))
    tables={}
    for mode in samples:
        entries=[dict(key=k,coordinate=coords[mode][k],mean_ns=str(sum(v,Fraction())/len(v)),sample_count=len(v),evidence=evidence[mode][k]) for k,v in sorted(samples[mode].items())]
        tables[mode]=dict(entries=entries,context=m["context"])
    frozen=dict(schema="dac-refinement-response-profile-1",frozen_at_utc=e2a.utc(),tables=tables,
        whole_target_times_read=False,prices="independent target-action intervals only; no full-target residual fit")
    write(out/"response-profile-frozen.json",frozen)
    predictions=[]
    for key,b in m["programs"].items():
        if b["split"]!="heldout":continue
        p=read(b["program"]["path"]);values={}
        for mode,cls in (("hot",ReusableResponseTable),("matched",ContextResponseTable)):
            result=cls(tuple(tables[mode]["entries"])).evaluate(p,m["context"])
            values[mode]=e2a.number(result)
        predictions.append(dict(program_key=key,layers=b["layers"],strategy=b["strategy"],predictions=values,target_time_read=False))
    write(out/"predictions.json",predictions)
    write(out/"frozen-barrier.json",dict(frozen_at_utc=e2a.utc(),**{f:sha(out/f) for f in ("response-profile-frozen.json","predictions.json")},heldout_read=False))
    return dict(entries={mode:len(table["entries"]) for mode,table in tables.items()},predictions=len(predictions))


def transfer_rows(out,split):
    rows=[]
    for ex in read(Path(out)/(split+"-executions.json")):
        require(ex["status"]=="completed","transfer process failed")
        n=read(ex["native"]["path"]);bare=[r for r in n["trials"] if not r["actual_stamped"]];stamped=[r for r in n["trials"] if r["actual_stamped"]]
        vals=[r["wall_ns"]/r["repeats"] for r in bare];stamps=[r["wall_ns"]/r["repeats"] for r in stamped]
        rows.append(dict(split="calibration" if split=="transfer-profile" else "heldout",bytes=ex["program"]["bytes"],
            mean_ns=statistics.mean(vals),cv=statistics.stdev(vals)/statistics.mean(vals),
            observer_effect=statistics.mean(stamps)/statistics.mean(vals)-1,
            host_pointer=n["host_pointer"],phases_ns={k:statistics.mean(r[k]/r["repeats"] for r in stamped) for k in ("copy_call_ns","gpu_sync_ns","cpu_sync_ns")},raw=ex["native"]))
    return rows


def transfer_freeze(out):
    out=Path(out).resolve();require(not(out/"transfer-heldout-started.json").exists(),"transfer freeze after heldout rejected")
    rows=transfer_rows(out,"transfer-profile");fit=fit_from_training(rows)
    write(out/"transfer-frozen.json",dict(schema="dac-refinement-transfer-profile-1",frozen_at_utc=e2a.utc(),fit=fit,calibration=rows,heldout_read=False))
    write(out/"transfer-barrier.json",dict(frozen_at_utc=e2a.utc(),sha256=sha(out/"transfer-frozen.json"),heldout_read=False))
    return fit["selected"]


def evaluate(out):
    out=Path(out).resolve();m=read(out/"manifest.json");ps=read(out/"predictions.json");bar=read(out/"frozen-barrier.json")
    require(all(sha(out/f)==bar[f] for f in ("response-profile-frozen.json","predictions.json")),"prediction barrier changed")
    held,observers=read(out/"heldout-executions.json"),read(out/"observer-executions.json");results=[]
    for pred in ps:
        key=pred["program_key"];times=[];obs=[]
        for ex in held:
            if ex["job"]["program_key"]!=key:continue
            if ex["status"]!="completed":times.append(dict(status="failed",job=ex["job"]["id"]));continue
            require(ex["started_at_utc"]>=bar["frozen_at_utc"],"target predates freeze")
            n=read(ex["native"]["path"]);vals=[r["wall_ns"]/r["repeats"] for r in n["trials"]]
            times.append(dict(status="completed",job=ex["job"]["id"],mean_ns=statistics.mean(vals),cv=statistics.stdev(vals)/statistics.mean(vals),trial_ns=vals,raw=ex["native"]))
        for ex in observers:
            if ex["job"]["program_key"]!=key:continue
            if ex["status"]!="completed":obs.append(dict(status="failed",passed=False));continue
            n=read(ex["native"]["path"]);vals={b:[r["wall_ns"]/r["repeats"] for r in n["trials"] if r["actual_stamped"]==b] for b in (True,False)}
            effect=statistics.mean(vals[True])/statistics.mean(vals[False])-1
            obs.append(dict(status="completed",effect=effect,passed=abs(effect)<=.05,raw=ex["native"]))
        modes={}
        for mode in ("hot","matched"):
            price=float(Fraction(pred["predictions"][mode]["duration_ns"]));completed=[t for t in times if t["status"]=="completed"]
            apes=[abs(price-t["mean_ns"])/t["mean_ns"] for t in completed];p90=e2a.quantile([abs(price-v)/v for t in completed for v in t["trial_ns"]],.9) if completed else None
            quality=len(completed)==4 and len(obs)==4 and all(a<=.10 for a in apes) and all(t["cv"]<=.10 for t in completed) and p90<=.20 and all(o["passed"] for o in obs)
            modes[mode]=dict(predicted_ns=price,mean_process_ape=statistics.mean(apes) if apes else None,maximum_process_ape=max(apes) if apes else None,process_mean_gate_passes=sum(a<=.1 for a in apes),trial_p90_ape=p90,fully_qualified=quality)
        results.append(dict(program_key=key,layers=pred["layers"],strategy=pred["strategy"],modes=modes,processes=times,observer=obs,
            measured_mean_ns=statistics.mean(t["mean_ns"] for t in times if t["status"]=="completed")))
    groups=[]
    for layers in (1,2):
        rows=[r for r in results if r["layers"]==layers];real=min(rows,key=lambda r:r["measured_mean_ns"])
        choices={}
        for mode in ("hot","matched"):
            chosen=min(rows,key=lambda r:r["modes"][mode]["predicted_ns"])
            choices[mode]=dict(selected=chosen["strategy"],measured_winner=real["strategy"],regret=chosen["measured_mean_ns"]/real["measured_mean_ns"]-1,all_candidates_qualified=all(r["modes"][mode]["fully_qualified"] for r in rows))
        groups.append(dict(layers=layers,modes=choices))
    fit=read(out/"transfer-frozen.json")["fit"]["selected"];tr=transfer_rows(out,"transfer-heldout");sizes=[]
    for size in HELDOUT:
        v=statistics.mean(r["mean_ns"] for r in tr if r["bytes"]==size);prediction=predict(fit,size)
        sizes.append(dict(bytes=size,predicted_ns=prediction,observed_ns=v,ape=abs(prediction-v)/v))
    transfer=dict(model=fit,heldout_sizes=sizes,processes=tr,maximum_size_mean_ape=max(r["ape"] for r in sizes),cv_passes=sum(r["cv"]<=.1 for r in tr),observer_passes=sum(abs(r["observer_effect"])<=.05 for r in tr))
    transfer["qualified"]=transfer["maximum_size_mean_ape"]<=.10 and transfer["cv_passes"]==24 and transfer["observer_passes"]==24
    summary=dict(schema="dac-context-refinement-result-1",evaluated_at_utc=e2a.utc(),programs=results,groups=groups,D2H=transfer,
        counts=dict(expected_processes=len(m["jobs"]),completed_processes=sum(x["status"]=="completed" for split in ("profile","heldout","observer","transfer-profile","transfer-heldout") for x in read(out/(split+"-executions.json"))),
            hot_qualified_programs=sum(r["modes"]["hot"]["fully_qualified"] for r in results),matched_qualified_programs=sum(r["modes"]["matched"]["fully_qualified"] for r in results)),
        whole_target_times_used_for_fit=False,old_outcomes_modified=False,limitations=m["protocol"])
    write(out/"summary.json",summary)
    return dict(counts=summary["counts"],groups=groups,D2H_qualified=transfer["qualified"],D2H_maximum_ape=transfer["maximum_size_mean_ape"])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","profile","freeze","heldout","observer","transfer-profile","transfer-freeze","transfer-heldout","evaluate"));p.add_argument("--out",required=True,type=Path);p.add_argument("--binary",type=Path)
    a=p.parse_args();result=prepare(a.out,a.binary) if a.command=="prepare" else freeze(a.out) if a.command=="freeze" else transfer_freeze(a.out) if a.command=="transfer-freeze" else evaluate(a.out) if a.command=="evaluate" else collect(a.out,a.command)
    print(result,flush=True)


if __name__=="__main__":main()
