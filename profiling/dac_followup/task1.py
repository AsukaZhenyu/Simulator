"""Portable independent transfer calibration and matched observer diagnostics.

Inputs are explicit native binaries/runtime folders and optional pre-exported
programs. No historical performance table is required. Old frozen campaigns
are never modified. A new paired-observer binary is diagnostic-only and cannot
retroactively qualify predictions made for the historical binary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from .transfer_fit import fit_from_training, predict

ROOT=Path(__file__).resolve().parents[2]
TRAIN=(4,256,4096,65536,1048576,4194304,16777216,67108864)
HELDOUT=(64,1024,2048,16384,262144,2097152,8388608,33554432)
ROUTES=(("dram","vram"),("vram","dram"))
HOST="GGML default CPU buffer; actual allocation path source-bound, page-lock status not queried; no pinned claim"

def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def current_sources():
    """New campaigns bind acquisition helpers as well as their cost mechanism."""
    paths=(Path(__file__),Path(__file__).with_name("transfer_fit.py"),
           ROOT/"modeling/llm_infer_model/performance/transfer_response.py",
           ROOT/"profiling/state_calibration/run.py",ROOT/"profiling/state_calibration/telemetry.py",
           ROOT/"profiling/state_response/collect.py",ROOT/"profiling/state_response/diagnose_cpu.py",
           ROOT/"profiling/highlevel_accuracy/bridge.py")
    return [receipt(path) for path in paths]

def fresh_profile_context(*, repeats=8, process_masks=(4,)):
    """Static identity and declared resources; not a load/frequency trajectory."""
    from profiling.state_response.collect import hardware_identity
    return dict(schema="dac-followup-explicit-profile-context-1",collected_at_utc=utc(),
                hardware=hardware_identity(),process_affinity_masks=list(process_masks),
                main_thread_affinity_mask=4,ggml_compute_threads=1,trials=8,repeats=repeats,
                minimum_active_warmup_ms=500,poll="0",precision="GGML_PREC_F32",
                cuda_graphs="disabled",host_memory_contract=HOST,
                static_identity_is_not_dynamic_state_trajectory=True,
                different_process_masks_are_distinct_cost_contexts=True)
def require(test,message):
    if not test:raise ValueError(message)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:json.dump(value,stream,allow_nan=False,separators=(",",":"));stream.write("\n")

def byte_reference(seed,size):
    """Independent bounded-memory implementation of procedural source bytes."""
    digest=hashlib.sha256()
    for begin in range(0,size,1<<18):
        z=(np.arange(begin,min(size,begin+(1<<18)),dtype=np.uint64)+1)*np.uint64(0x9e3779b9)
        z=(z+np.uint64(seed)+np.uint64(0x85ebca6b)) & np.uint64(0xffffffff)
        z^=z>>np.uint64(16);z=(z*np.uint64(0x7feb352d))&np.uint64(0xffffffff)
        z^=z>>np.uint64(15);z=(z*np.uint64(0x846ca68b))&np.uint64(0xffffffff)
        z^=z>>np.uint64(16)
        digest.update((z&np.uint64(255)).astype(np.uint8).tobytes())
    return dict(kind="byte_sha256",sha256=digest.hexdigest(),bytes=size,oracle="independent chunked mix32 and hashlib; complete payload")

def sample(mode="action"):
    return dict(schema="steady-cost-request-1",mode=mode,target_action_index=None,warmup=20,
                warmup_min_ms=500,trials=8,repeats=8,poll="0",instrumentation="bare",main_thread_mask=4)

def prepare(out,binary,observer_binary,runtime_dirs,program_manifest=None):
    from llm_infer_model.performance.reusable_action_costs import digest
    from profiling.steady_costs.contracts import validate_component
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    programs={};jobs=[]
    for split,sizes in (("calibration",TRAIN),("transfer-heldout",HELDOUT)):
        for src,dst in ROUTES:
            # Balanced deterministic rotation rather than all increasing sizes.
            for ordinal,size in enumerate(sizes):
                key=f"transfer-{src}-{dst}-{size}"
                p=dict(format_version="steady-component-program-1",kind="transfer",device="gpu",threads=1,cuda_device=0,
                       seed=2026102511 if split=="calibration" else 2026102512,
                       shape=dict(bytes=size,src=src,dst=dst),allocation_budget_bytes=1<<30)
                p["program_sha256"]=digest(p);validate_component(p)
                pp=out/"programs"/(key+".json");rp=out/"references"/(key+".json")
                write(pp,p);write(rp,byte_reference(p["seed"],size))
                programs[key]=dict(program=receipt(pp),reference=receipt(rp),bytes=size,route=src+"->"+dst)
                for repeat in range(3):jobs.append(dict(id=f"{split}-{key}-r{repeat}",split=split,program_key=key,round=repeat,request=sample()))
    # Round-first alternating directions and rotated size orders avoid a monotone
    # payload/time confound. Inputs and all choices are frozen before sampling.
    transfers=[]
    for split,sizes in (("calibration",TRAIN),("transfer-heldout",HELDOUT)):
        for repeat in range(3):
            order=list(sizes[repeat:]+sizes[:repeat])
            if repeat%2:order.reverse()
            for size in order:
                for src,dst in ROUTES if repeat%2==0 else tuple(reversed(ROUTES)):
                    transfers.append(next(j for j in jobs if j["id"]==f"{split}-transfer-{src}-{dst}-{size}-r{repeat}"))
    jobs=transfers
    old=read(program_manifest) if program_manifest else None
    for h in (256,512):
        for phase,q,history in (("p16",16,0),("p64",64,0),("d1h128",1,128)):
            wid=f"h{h}-{phase}"
            for strategy in range(4):
                key=f"{wid}-s{strategy}-heldout"
                if old:
                    binding=old["programs"][key]
                    for field in ("program","reference"):require(sha(binding[field]["path"])==binding[field]["sha256"],"input artifact changed")
                    programs[key]=dict(binding)
                else:
                    from profiling.highlevel_accuracy.cases import build_case
                    from profiling.highlevel_accuracy.bridge import export_plan,reference_ports
                    kwargs=dict(hidden=h,intermediate=2*h,heads=h//64,kv_heads=2,head_dim=64,tokens=q,history=history,
                                threads=1,devices=(("cpu","cpu","cpu"),("gpu","gpu","gpu"),("gpu","cpu","gpu"),("gpu","gpu","gpu"))[strategy])
                    if strategy==3:kwargs.update(ffn_cut=h,ffn_devices=("cpu","gpu"),norm_device="gpu",merge_device="gpu")
                    p=export_plan(build_case(out/"inputs"/key,**kwargs).compile(),seed=2026102513+h+q+history)
                    pp=out/"programs"/(key+".json");rp=out/"references"/(key+".json")
                    write(pp,p);write(rp,reference_ports(p));programs[key]=dict(program=receipt(pp),reference=receipt(rp),workload=wid,strategy=strategy)
                for repeat in range(4):
                    order="ABBA" if repeat%2==0 else "BAAB"
                    req=dict(schema="dac-followup-paired-request-1",sampling=sample("full"),order=order,sham_bare_bare=False)
                    jobs.append(dict(id=f"observer-{wid}-s{strategy}-r{repeat}",split="observer",program_key=key,round=repeat,request=req))
    sham_keys=("h256-p64-s2-heldout","h512-p16-s2-heldout","h512-p64-s2-heldout","h512-d1h128-s3-heldout")
    for key in sham_keys:
        for repeat in range(4):
            req=dict(schema="dac-followup-paired-request-1",sampling=sample("full"),order="ABBA" if repeat%2==0 else "BAAB",sham_bare_bare=True)
            jobs.append(dict(id=f"sham-{key}-r{repeat}",split="observer",program_key=key,round=repeat,request=req))
    transfer_jobs=[j for j in jobs if j["split"]!="observer"]
    observers=[j for j in jobs if j["split"]=="observer"]
    # Deterministic round/candidate rotation; sham controls are interspersed.
    observers.sort(key=lambda j:(j["round"],hashlib.sha256(j["id"].encode()).hexdigest()))
    sources=current_sources()
    binaries={"transfer":receipt(binary),"observer":receipt(observer_binary)}
    runtime={str(p.resolve()):sha(p) for folder in runtime_dirs for p in Path(folder).glob("*.dll")}
    require(bool(runtime),"explicit runtime DLL directory empty")
    manifest=dict(schema="dac-followup-task1-plan-1",prepared_at_utc=utc(),programs=programs,jobs=transfer_jobs+observers,
                  binaries=binaries,runtime_dirs=[str(Path(folder).resolve()) for folder in runtime_dirs],runtime_sha256=runtime,sources=sources,
                  profile_context=fresh_profile_context(),
                  reused_program_manifest=receipt(program_manifest) if program_manifest else None,
                  historical_cost_tables_read=False,
                  protocol=dict(calibration_sizes=list(TRAIN),heldout_sizes=list(HELDOUT),training_processes=48,
                    transfer_heldout_processes=48,observer_processes=96,sham_processes=16,
                    maximum_independent_size_mean_ape=.10,maximum_process_trial_cv=.10,
                    maximum_observer_relative_difference=.05,all_observations_retained=True,
                    transfer_context=HOST,transfer_boundary="original compact Component.action GGML tensor_copy + GPU/CPU synchronize; allocation excluded",
                    transfer_model_candidates="nonnegative relative affine, one continuous hinge at16KiB/256KiB/1MiB; training-only LOO selection",
                    observer_context="new diagnostic binary; within-process ABBA/BAAB; no historical prediction qualification",
                    no_background_injection=True,old_frozen_results_unchanged=True))
    write(out/"manifest.json",manifest)
    print(json.dumps(dict(prepared=str(out),jobs=len(jobs),calibration=48,transfer_heldout=48,observer=112,native_timing_started=False)),flush=True)


def verify(native,program,reference,request,pid,observer):
    require(native["process_id"]==pid and native["program_sha256"]==program["program_sha256"] and native["request"]==request,"owned native identity differs")
    affinity=native["main_thread_affinity"]
    require(affinity["requested_mask"]==affinity["observed_mask"]==affinity["final_mask"]==4 and affinity["restored"],"affinity receipt differs")
    if observer:
        require(native["native_repeat_consistency"]["passed"],"fixed-input repeated outputs changed")
        final=native["verification_native"]
        require(final["copy_values_equal"] and final["actions"]==program["actions"],"native COPY/action trace differs")
        require(set(final["ports"])==set(reference),"port coverage differs")
        maximum=0.
        for key,expected in reference.items():
            actual=np.asarray(final["ports"][key],dtype=float);expected=np.asarray(expected,dtype=float)
            require(actual.shape==expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual,expected,atol=5e-5,rtol=5e-5),"paired numerical mismatch "+key)
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        require(len(native["trials"])==request["sampling"]["trials"]*4,"paired block coverage differs")
        sampling=request["sampling"]
    else:
        output=native["component_output"]
        require(output["full_byte_equality"] and output["checked_bytes"]==reference["bytes"] and output["sha256"]==output["source_sha256"]==reference["sha256"],"transfer full payload mismatch")
        require(len(native["trials"])==request["trials"],"transfer trial coverage differs")
        maximum=0.;sampling=request
    hz=native["clock"]["frequency_hz"]
    for row in native["trials"]:
        require(row["end_tick"]>=row["begin_tick"] and abs((row["end_tick"]-row["begin_tick"])*1e9/hz-row["wall_ns"])<=1,"QPC interval differs")
        require(row["repeats"]==sampling["repeats"],"native replay count differs")
    return dict(passed=True,maximum_absolute_error=maximum)


def collect(out,split):
    from profiling.state_calibration import run
    out=Path(out).resolve();manifest=read(out/"manifest.json")
    require(split in {"calibration","transfer-heldout","observer"},"unknown collection split")
    if split=="transfer-heldout":
        frozen=read(out/"frozen-barrier.json")
        require(sha(out/"transfer-model-frozen.json")==frozen["frozen_sha256"],"transfer model changed after freeze")
    for item in manifest["sources"]:require(sha(item["path"])==item["sha256"],"experiment source changed")
    for path,expected in manifest["runtime_sha256"].items():require(sha(path)==expected,"runtime changed")
    observer=split=="observer";binary=manifest["binaries"]["observer" if observer else "transfer"]
    require(sha(binary["path"])==binary["sha256"],"native binary changed")
    started=out/(split+"-started.json")
    if not started.exists():write(started,dict(started_at_utc=utc(),manifest=receipt(out/"manifest.json")))
    rows=[];programs={};references={}
    with run.measurement_lock():
        for job in (j for j in manifest["jobs"] if j["split"]==split):
            folder=out/"jobs"/job["id"]
            if (folder/"execution.json").exists():rows.append(read(folder/"execution.json"));continue
            folder.mkdir(parents=True,exist_ok=False);binding=manifest["programs"][job["program_key"]]
            for field in ("program","reference"):require(sha(binding[field]["path"])==binding[field]["sha256"],"input binding changed")
            if job["program_key"] not in programs:
                programs[job["program_key"]]=read(binding["program"]["path"]);references[job["program_key"]]=read(binding["reference"]["path"])
            pp=programs[job["program_key"]];ref=references[job["program_key"]]
            req=folder/"request.json";output=folder/"result.json";write(req,job["request"])
            command=[binary["path"],binding["program"]["path"],str(req),str(output)]
            record=dict(job=job,started_at_utc=utc(),command=command,status="failed",program=binding)
            process=None;handles=[]
            try:
                env=run.child_environment(manifest["runtime_dirs"]);env["OPENBLAS_NUM_THREADS"]="1"
                process,stdout,stderr,affinity=run.launch(command,env,folder/"stdout.txt",folder/"stderr.txt",4);handles=[stdout,stderr]
                record.update(process_id=process.pid,affinity=affinity)
                code=process.wait(timeout=180)
                record.update(returncode=code,forced_termination=False,process_exited_verified=True)
                require(code==0,"native process failed")
                native=read(output);record["numerical"]=verify(native,pp,ref,job["request"],process.pid,observer)
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
                record["completed_at_utc"]=utc();write(folder/"execution.json",record)
            rows.append(record);print(json.dumps(dict(split=split,job=job["id"],status=record["status"],error=record.get("error"))),flush=True)
    write(out/(split+"-executions.json"),rows)
    return dict(split=split,completed=sum(r["status"]=="completed" for r in rows),expected=len(rows),failed=sum(r["status"]!="completed" for r in rows))


def transfer_rows(out,split):
    rows=[]
    for e in read(out/(split+"-executions.json")):
        require(e["status"]=="completed","failed transfer process; model remains unqualified")
        result=read(e["native"]["path"]);times=[r["wall_ns"]/r["repeats"] for r in result["trials"]]
        binding=e["program"]
        rows.append(dict(split=split,route=binding["route"],bytes=binding["bytes"],mean_ns=statistics.mean(times),
                         cv=statistics.stdev(times)/statistics.mean(times),evidence=e["native"],process_id=e["process_id"]))
    return rows


def freeze(out):
    out=Path(out).resolve();require(not(out/"transfer-heldout-started.json").exists(),"cannot fit after independent heldout starts")
    rows=transfer_rows(out,"calibration");models={}
    for src,dst in ROUTES:
        route=src+"->"+dst
        selected=[r for r in rows if r["route"]==route]
        models[route]=fit_from_training(selected)
        means={size:statistics.mean(r["mean_ns"] for r in selected if r["bytes"]==size) for size in TRAIN}
        slope,intercept=np.polyfit(list(means),list(means.values()),1)
        models[route]["ordinary_absolute_ols_baseline"]=dict(fixed_ns=float(intercept),small_ns_per_byte=float(slope),
                    large_ns_per_byte=float(slope),breakpoint_bytes=None,
                    physically_admissible=bool(intercept>=0 and slope>=0),
                    fitted_on="new calibration sizes only; same global affine family as former decomposition")
    value=dict(schema="dac-followup-transfer-model-1",frozen_at_utc=utc(),models=models,training_observations=rows,
               host_memory_contract=HOST,calibration_only=True,independent_qualification=False,
               formula="fixed + slope_small*min(bytes,k) + slope_large*max(bytes-k,0); affine is equal slopes",
               payload_bandwidth_scaling_physical_validity_not_established=True)
    write(out/"transfer-model-frozen.json",value)
    write(out/"frozen-barrier.json",dict(frozen_at_utc=utc(),frozen_sha256=sha(out/"transfer-model-frozen.json"),heldout_read=False))
    return {route:row["selected"] for route,row in models.items()}


def evaluate(out):
    from llm_infer_model.performance.transfer_response import TransferResponse
    out=Path(out).resolve();manifest=read(out/"manifest.json");frozen=read(out/"transfer-model-frozen.json")
    barrier=read(out/"frozen-barrier.json");require(sha(out/"transfer-model-frozen.json")==barrier["frozen_sha256"],"frozen transfer model changed")
    held=transfer_rows(out,"transfer-heldout");transfer={}
    for src,dst in ROUTES:
        route=src+"->"+dst;model=frozen["models"][route]["selected"]
        rows=[r for r in held if r["route"]==route]
        grouped={size:statistics.mean(r["mean_ns"] for r in rows if r["bytes"]==size) for size in HELDOUT}
        checked=[dict(bytes=size,observed_ns=mean,predicted_ns=predict(model,size),ape=abs(predict(model,size)-mean)/mean) for size,mean in grouped.items()]
        maximum=max(r["ape"] for r in checked);stable=all(r["cv"]<=.10 for r in rows)
        baselines={"ordinary_absolute_ols":frozen["models"][route]["ordinary_absolute_ols_baseline"],
                   "nonnegative_relative_affine":next(m for m in frozen["models"][route]["candidates"] if m["breakpoint_bytes"] is None)}
        baseline_metrics={name:dict(model=candidate,maximum_heldout_size_mean_ape=max(abs(predict(candidate,size)-mean)/mean for size,mean in grouped.items()),
                          median_heldout_size_mean_ape=statistics.median(abs(predict(candidate,size)-mean)/mean for size,mean in grouped.items())) for name,candidate in baselines.items()}
        response=TransferResponse(route=route,**{k:model[k] for k in ("fixed_ns","small_ns_per_byte","large_ns_per_byte","breakpoint_bytes")},
                  minimum_bytes=min(TRAIN+HELDOUT),maximum_bytes=max(TRAIN+HELDOUT),host_memory_contract=HOST,
                  boundary=manifest["protocol"]["transfer_boundary"],evidence=tuple(r["evidence"]["sha256"] for r in rows),qualified=maximum<=.10 and stable)
        transfer[route]=dict(model=model,heldout_size_means=checked,maximum_heldout_size_mean_ape=maximum,
                             process_cv_passes=sum(r["cv"]<=.10 for r in rows),processes=len(rows),qualified=response.qualified,
                             baseline_metrics=baseline_metrics,heldout_process_scatter=rows,
                             calibration_process_scatter=[r for r in frozen["training_observations"] if r["route"]==route],
                             bandwidth_counterfactual_qualified=False,
                             interpretation="Observed byte-response interpolation is tested; fitted byte terms include staging/backend service, not identified PCIe-only bandwidth.")
    observer=[]
    for e in read(out/"observer-executions.json"):
        if e["status"]!="completed":observer.append(dict(job=e["job"]["id"],passed=False,status=e["status"]));continue
        native=read(e["native"]["path"]);contrast=[];bare=[];stamped=[]
        for trial in range(8):
            rows=[r for r in native["trials"] if r["trial"]==trial]
            a=statistics.mean(r["wall_ns"]/r["repeats"] for r in rows if r["assigned_label"]=="A")
            b=statistics.mean(r["wall_ns"]/r["repeats"] for r in rows if r["assigned_label"]=="B")
            contrast.append(b/a-1);bare.append(a);stamped.append(b)
        effect=statistics.mean(stamped)/statistics.mean(bare)-1
        observer.append(dict(job=e["job"]["id"],program_key=e["job"]["program_key"],round=e["job"]["round"],
                             sham=native["sham_bare_bare"],order=native["order"],relative_difference=effect,
                             contrasts=contrast,bare_mean_ns=statistics.mean(bare),labeled_b_mean_ns=statistics.mean(stamped),
                             bare_trial_cv=statistics.stdev(bare)/statistics.mean(bare),
                             passed=abs(effect)<=.05,status="completed",evidence=e["native"]))
    real=[r for r in observer if not r.get("sham",False)];sham=[r for r in observer if r.get("sham",False)]
    groups=[]
    for key in sorted({r.get("program_key") for r in real if r.get("program_key")}):
        rows=[r for r in real if r.get("program_key")==key]
        groups.append(dict(program=key,processes=len(rows),passed=len(rows)==4 and all(r["passed"] for r in rows),
                           maximum_absolute_effect=max(abs(r["relative_difference"]) for r in rows),
                           average_effect=statistics.mean(r["relative_difference"] for r in rows)))
    result=dict(schema="dac-followup-task1-result-1",evaluated_at_utc=utc(),transfer=transfer,
                observer=dict(processes=len(real),passed_processes=sum(r["passed"] for r in real),programs=groups,
                 qualified_programs=sum(g["passed"] for g in groups),sham_processes=len(sham),sham_passes=sum(r["passed"] for r in sham),
                 rows=observer,old_results_requalified=False,complete_strategy_accuracy_new_binary_validated=False,
                 interpretation="Matched within-process contrasts diagnose observer overhead versus residual temporal variation; they neither prove OS causation nor repair historical 0/6 qualification."),
                all_native_processes=sum(len(read(out/(split+"-executions.json"))) for split in ("calibration","transfer-heldout","observer")),
                all_failures_retained=True,old_frozen_results_unchanged=True)
    write(out/"summary.json",result)
    return dict(transfer={k:dict(qualified=v["qualified"],maximum_ape=v["maximum_heldout_size_mean_ape"]) for k,v in transfer.items()},
                observer_processes_passed=sum(r["passed"] for r in real),observer_programs_passed=sum(g["passed"] for g in groups),sham_passes=sum(r["passed"] for r in sham))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","calibration","freeze","transfer-heldout","observer","evaluate"));parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--binary",type=Path);parser.add_argument("--observer-binary",type=Path);parser.add_argument("--runtime-dir",action="append",type=Path)
    parser.add_argument("--reuse-program-manifest",type=Path,help="Optional compiler/reference artifacts only; no old performance table read")
    args=parser.parse_args()
    if args.command=="prepare":
        if not(args.binary and args.observer_binary and args.runtime_dir):parser.error("prepare requires explicit binary, observer-binary and runtime-dir")
        result=prepare(args.out,args.binary,args.observer_binary,args.runtime_dir,args.reuse_program_manifest)
    elif args.command=="freeze":result=freeze(args.out)
    elif args.command=="evaluate":result=evaluate(args.out)
    else:result=collect(args.out,args.command)
    print(json.dumps(result,allow_nan=False),flush=True)

if __name__=="__main__":main()
