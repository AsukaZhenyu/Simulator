"""Matched Dense strategies, independent isolated action profiles, frozen holdout.

Run prepare -> profile -> freeze -> heldout -> observer -> evaluate. No complete
target latency, stamped target action cost or target utilization enters prices.
Every failed process is retained; collection continues to preserve denominators.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable, coordinate, digest, semantic_action
from profiling.steady_costs.contracts import component_reference, validate_component

ROOT = Path(__file__).resolve().parents[2]
QUALITY = dict(max_process_mean_ape=.10, max_trial_average_p90_ape=.20,
    max_process_trial_cv=.10, max_observer_relative_mean_difference=.05)
SEQUENCES = ((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1),
             (2,3,1,0),(3,0,2,1),(0,1,3,2),(1,2,0,3))


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def require(ok, text):
    if not ok: raise ValueError(text)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":")); stream.write("\n")
def number(value):
    if isinstance(value, Fraction): return str(value)
    if isinstance(value, dict): return {k:number(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [number(v) for v in value]
    return value
def request(mode, *, trials=8, warmup_ms=500, target=None):
    return dict(schema="steady-cost-request-1", mode=mode, target_action_index=target,
        warmup=20, warmup_min_ms=warmup_ms, trials=trials, repeats=8,
        poll="0", instrumentation="bare", main_thread_mask=4)
def quantile(values, p): return float(np.quantile(values,p,method="higher"))


def matched(programs, references):
    """Check common parent data and partition reconstructions independently."""
    base = programs[0]; tensors = {t["id"]:t for t in base["graph"]["workload"]["tensors"]}
    common = set.intersection(*(set(p["initial_values"]) for p in programs))
    for tid in common:
        require(all(p["initial_values"][tid] == base["initial_values"][tid] for p in programs), "candidate input differs: "+tid)
    reconstructed = []
    split = programs[3]
    parts = split["graph"]["boundary"]["ffn_execution"]["weight_partitions"]
    for parent in sorted({p["source_tensor_id"] for p in parts}):
        relevant = sorted((p for p in parts if p["source_tensor_id"]==parent),key=lambda p:p["start"])
        width,height = tensors[parent]["ne"][:2]
        restored = np.empty((height,width),dtype=np.float32)
        for part in relevant:
            leaf = np.asarray(split["initial_values"][part["tensor_id"]],dtype=np.float32)
            if part["stage"] == "down": restored[:,part["start"]:part["end"]] = leaf.reshape(height,part["end"]-part["start"])
            else: restored[part["start"]:part["end"],:] = leaf.reshape(part["end"]-part["start"],width)
        expected = np.asarray(base["initial_values"][parent],dtype=np.float32)
        require(np.array_equal(restored.ravel(),expected),"partition does not reconstruct parent")
        reconstructed.append(dict(parent=parent,bytes=int(restored.nbytes),sha256=hashlib.sha256(restored.tobytes()).hexdigest()))
    goals = base["graph"]["scenario"]["goal"]["output_locations"]
    require(all(p["graph"]["scenario"]["goal"]["output_locations"]==goals for p in programs),"candidate final goals differ")
    worst=0.
    for tid in goals:
        expected=np.asarray(references[0].get(tid,programs[0]["initial_values"].get(tid)))
        require(expected.ndim>0,"goal has neither produced reference nor retained initial bytes")
        for candidate,ref in zip(programs[1:],references[1:]):
            actual=np.asarray(ref.get(tid,candidate["initial_values"].get(tid)))
            require(np.allclose(expected,actual,atol=5e-5,rtol=5e-5),"candidate numerical semantics differ")
            worst=max(worst,float(np.max(np.abs(expected-actual))))
    return dict(passed=True,common_initial_tensors=len(common),parent_reconstruction=reconstructed,
        final_goals=len(goals),maximum_reference_difference=worst,atol=5e-5,rtol=5e-5)


def prepare(out):
    from profiling.highlevel_accuracy.cases import build_case
    from profiling.highlevel_accuracy.bridge import export_plan,reference_ports
    from profiling.state_response.collect import hardware_identity
    old=read(ROOT/"profiling/steady_costs/results/steady-v1-f32/manifest.json")
    out=Path(out).resolve()
    require(not(out/"manifest.json").exists(),"already prepared campaign cannot be replaced")
    out.mkdir(parents=True,exist_ok=True)
    binary=Path(old["binary"]);build=read(binary.parent/"build-manifest.json")
    require(build["returncode"]==0 and sha(binary)==build["binary_sha256"],"native binary identity differs")
    # Verify the existing successful build without changing or rebuilding it.
    for path,expected in build["bound_files"].items(): require(sha(path)==expected,"native build input changed: "+path)
    bindings={};workloads=[];jobs=[];matches=[]
    for h in (256,512):
        for phase,q,history in (("p16",16,0),("p64",64,0),("d1h128",1,128)):
            wid=f"h{h}-{phase}";workloads.append(wid)
            for split,offset in (("profile",0),("heldout",10000)):
                programs=[];refs=[]
                seed=202610250+len(workloads)+offset
                for strategy in range(4):
                    cid=wid+f"-s{strategy}";key=cid+"-"+split
                    kwargs=dict(hidden=h,intermediate=h*2,heads=h//64,kv_heads=2,head_dim=64,
                        tokens=q,history=history,threads=1,devices=(("cpu","cpu","cpu"),("gpu","gpu","gpu"),("gpu","cpu","gpu"),("gpu","gpu","gpu"))[strategy])
                    if strategy==3: kwargs.update(ffn_cut=h,ffn_devices=("cpu","gpu"),norm_device="gpu",merge_device="gpu")
                    pp=out/"programs"/(key+".json");rp=out/"references"/(key+".json")
                    if pp.exists() and rp.exists():
                        from profiling.highlevel_accuracy.bridge import validate_program
                        program=read(pp);validate_program(program);reference=read(rp)
                        require(program["seed"]==seed,"resumed preparation seed differs")
                    else:
                        directory=out/"inputs"/cid
                        plan=build_case(directory,**kwargs,reuse=directory.exists() and any(directory.iterdir()))
                        program=export_plan(plan.compile(),seed=seed);reference=reference_ports(program)
                        require(not pp.exists() and not rp.exists(),"partially written artifacts require explicit inspection")
                        write(pp,program);write(rp,reference)
                    bindings[key]=dict(program=receipt(pp),reference=receipt(rp),seed=seed,workload=wid,strategy=strategy)
                    programs.append(program);refs.append(reference)
                    print(json.dumps(dict(prepared=key,actions=len(program["actions"]),elements=sum(math.prod(t["ne"]) for t in program["graph"]["workload"]["tensors"]))),flush=True)
                matches.append(dict(workload=wid,split=split,**matched(programs,refs)))
            for repeat in range(3):
                for strategy in range(4):
                    jobs.append(dict(id=f"profile-{wid}-s{strategy}-r{repeat}",split="profile",program_key=f"{wid}-s{strategy}-profile",round=repeat,
                        request=request("all-actions",warmup_ms=0),role="independent isolated hot prepared-input actions"))
            for repeat,order in enumerate(SEQUENCES):
                for position,strategy in enumerate(order):
                    jobs.append(dict(id=f"heldout-{wid}-r{repeat}-p{position}-s{strategy}",split="heldout",program_key=f"{wid}-s{strategy}-heldout",round=repeat,position=position,
                        request=request("full",trials=12),role="complete matched target strategy"))
            for strategy in range(4):
                for pair in range(4):
                    for position,mode in enumerate(("full-stamped","full") if pair<2 else ("full","full-stamped")):
                        jobs.append(dict(id=f"observer-{wid}-s{strategy}-pair{pair}-p{position}",split="observer",program_key=f"{wid}-s{strategy}-heldout",pair=pair,position=position,
                            request=request(mode),role="post-freeze observer diagnostic; never calibration"))
    for src,dst in (("dram","vram"),("vram","dram")):
        for size in (4096,16384,65536,262144,1048576,4194304):
            key=f"transfer-{src}-{dst}-{size}"
            p=dict(format_version="steady-component-program-1",kind="transfer",device="gpu",threads=1,cuda_device=0,
                seed=202610259,shape=dict(bytes=size,src=src,dst=dst),allocation_budget_bytes=1<<30)
            p["program_sha256"]=digest(p);validate_component(p)
            pp=out/"programs"/(key+".json");rp=out/"references"/(key+".json")
            write(pp,p);write(rp,component_reference(p));bindings[key]=dict(program=receipt(pp),reference=receipt(rp),seed=p["seed"])
            for repeat in range(3):jobs.insert(0,dict(id=f"profile-{key}-r{repeat}",split="profile",program_key=key,round=repeat,request=request("action"),role="independent transfer mechanism probe"))
    hardware=hardware_identity()
    context=dict(binary_sha256=sha(binary),runtime_dlls={str(p):sha(p) for p in Path(old["runtime_dirs"][0]).glob("*.dll")},hardware=hardware,
        dtype="F32",cuda_graphs="disabled",precision="GGML_PREC_F32",threads=1,poll="0",main_thread_mask=4,
        boundary="unchanged synchronous Executor.action; hot prepared inputs; isolated batch-8",allocation_timed=False)
    sources=[receipt(Path(__file__)),receipt(ROOT/"modeling/llm_infer_model/performance/reusable_action_costs.py"),receipt(binary.parent/"build-manifest.json"),receipt(binary)]
    manifest=dict(schema="dac-real-selection-manifest-1",prepared_at_utc=utc(),workloads=workloads,programs=bindings,jobs=jobs,
        binary=str(binary),runtime_dirs=old["runtime_dirs"],context=context,sources=sources,
        protocol=dict(quality=QUALITY,sequences=SEQUENCES,rounds=8,bootstrap_seed=20261024,bootstrap_resamples=10000,
            simultaneous_quantile=1-.05/6,all_samples_retained=True,allocation_capacity_qualified=False,
            timing="warm fixed-input serial replay averages; no startup/allocation/reset/verification",target_complete_timing_used_for_calibration=False,
            profile_scope="all-actions isolated hot batches with untimed full initialization; no mixed full-stamped prices",
            extra_transition_correction="not identified; zero additional term, assessed on heldout",split_is_concurrent=False,
            cuda_private_workspace="unknown; no zero substitution"))
    write(out/"matching.json",matches);write(out/"manifest.json",manifest)
    return dict(prepared=True,workloads=len(workloads),programs=len(bindings),jobs=len(jobs),matching_passed=len(matches))


def verify_output(program, oracle, req, native):
    require(native["program_sha256"]==program["program_sha256"] and native["request"]==req,"native program/request mismatch")
    require(native["observed_runtime_flags"]==dict(GGML_CUDA_DISABLE_GRAPHS="1",NVIDIA_TF32_OVERRIDE="0",matmul_precision="GGML_PREC_F32"),"runtime flags differ")
    require(native["collected_cpu_cycles"] is False and native["per_action_CPU_counter_queries"] is False,"unexpected observer instrumentation")
    affinity=native["main_thread_affinity"]
    require(affinity["requested_mask"]==affinity["observed_mask"]==affinity["final_mask"]==4 and affinity["restored"],"main affinity differs")
    warm=native["warmup"]
    require(warm["actual_iterations"]>=req["warmup"] and warm["wall_ns"]>=req["warmup_min_ms"]*1_000_000,"warmup incomplete")
    hz=native["clock"]["frequency_hz"]
    def interval(row):
        require(row["end_tick"]>=row["begin_tick"] and abs((row["end_tick"]-row["begin_tick"])*1e9/hz-row["wall_ns"])<=1,"clock duration mismatch")
    if program["format_version"]=="steady-component-program-1":
        c=native["component_output"]
        require(c["full_byte_equality"] and c["sha256"]==c["source_sha256"]==oracle["sha256"] and c["checked_bytes"]==oracle["bytes"],"component bytes differ")
        groups=[dict(trials=native["trials"])];worst=0.
    else:
        require(native["native_repeat_consistency"]["passed"],"fixed-input outputs changed")
        v=native["verification_native"]
        require(v["final_execution_ports_captured_outside_timing"] and v["copy_values_equal"],"copy/final check missing")
        require(v["actions"]==program["actions"],"native action trace differs")
        tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
        copies=[dict(index=i,tensor_id=a["target_id"],src=a["src"],dst=a["dst"],storage_bytes=tensors[a["target_id"]]["storage_bytes"])
            for i,a in enumerate(program["actions"]) if a["kind"]=="COPY"]
        require(v["copies"]==copies,"native copy route/bytes inventory differs")
        require(set(v["ports"])==set(oracle),"numerical port inventory differs")
        worst=0.
        for tid,expected in oracle.items():
            actual=np.asarray(v["ports"][tid],dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
            require(actual.shape==expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual,expected,atol=5e-5,rtol=5e-5),"numerical mismatch: "+tid)
            worst=max(worst,float(np.max(np.abs(actual-expected))))
        groups=native["groups"] if req["mode"]=="all-actions" else [dict(trials=native["trials"])]
        if req["mode"]=="all-actions":
            require(len(groups)==len(program["actions"]),"isolated action coverage differs")
            for index,group in enumerate(groups):require(group["action_index"]==index and group["action"]==program["actions"][index],"isolated action identity differs")
    for group in groups:
        require(len(group["trials"])==req["trials"],"trial coverage differs")
        for i,row in enumerate(group["trials"]):
            require(row["trial"]==i and row["repeats"]==req["repeats"],"trial identity differs");interval(row)
            if req["mode"]=="full-stamped":require(len(row["actions"])==len(program["actions"]),"stamped inventory differs")
    return dict(passed=True,maximum_absolute_error=worst,atol=5e-5,rtol=5e-5)


def refine_prethermal(out, prepared):
    """Freeze a pre-measurement warmup refinement in a new campaign version.

    The old prepared programs remain immutable; only the probe launch inventory
    changes. No native trial may exist in the preparation campaign.
    """
    prepared=Path(prepared).resolve();old=read(prepared/"manifest.json")
    require(not(prepared/"profile-started.json").exists() and not(prepared/"heldout-started.json").exists(),"prethermal refinement requires no old measurements")
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    representatives={}
    for key,binding in old["programs"].items():
        if not key.endswith("-profile"):continue
        program=read(binding["program"]["path"])
        for index in range(len(program["actions"])):
            semantic=semantic_action(program,index)
            representatives.setdefault(digest(semantic),dict(program_key=key,index=index,semantic=semantic))
    jobs=[j for j in old["jobs"] if j["split"]!="profile" or j["role"]=="independent transfer mechanism probe"]
    programs=dict(old["programs"])
    for src,dst in (("dram","vram"),("vram","dram")):
        for size in (4,64,256,1024):
            key=f"transfer-{src}-{dst}-{size}"
            program=dict(format_version="steady-component-program-1",kind="transfer",device="gpu",threads=1,cuda_device=0,
                seed=202610259,shape=dict(bytes=size,src=src,dst=dst),allocation_budget_bytes=1<<30)
            program["program_sha256"]=digest(program);validate_component(program)
            pp=out/"programs"/(key+".json");rp=out/"references"/(key+".json")
            write(pp,program);write(rp,component_reference(program));programs[key]=dict(program=receipt(pp),reference=receipt(rp),seed=program["seed"])
            for repeat in range(3):jobs.insert(0,dict(id=f"profile-{key}-r{repeat}",split="profile",program_key=key,round=repeat,request=request("action"),role="independent transfer mechanism probe"))
    # Every unique primitive gets its own fresh process and 500ms active warmup.
    for ordinal,(key,row) in enumerate(sorted(representatives.items())):
        for repeat in range(3):
            jobs.insert(0,dict(id=f"profile-primitive-{ordinal:03}-r{repeat}",split="profile",program_key=row["program_key"],round=repeat,
                semantic_key=key,request=request("action",target=row["index"]),role="independent isolated exact semantic action; 500ms active warmup"))
    manifest=dict(old,jobs=jobs,programs=programs,prepared_at_utc=utc(),schema="dac-real-selection-manifest-1")
    manifest["context"]=dict(old["context"],minimum_active_primitive_warmup_ms=500)
    manifest["sources"]=[receipt(Path(__file__)),receipt(ROOT/"modeling/llm_infer_model/performance/reusable_action_costs.py"),*old["sources"][2:]]
    manifest["pre_measurement_refinement"]=dict(prepared=receipt(prepared/"manifest.json"),reason="Align minimum active warmup of independently priced primitive and target; avoid implicit thermal equivalence from 20 short iterations",
        no_native_data_collected_before_refinement=True,primitive_coordinates=len(representatives),fresh_processes_per_coordinate=3,
        old_preparation_only_no_timing_claim=True)
    manifest["protocol"]=dict(old["protocol"],profile_scope="isolated exact action mode, independent profile seed, 500ms+20 iterations per primitive, 8 trial batches of 8 replays",
        minimum_active_primitive_warmup_ms=500,residual_hot_vs_mixed_cache_equivalence="hypothesis assessed by heldout; not asserted")
    write(out/"matching.json",read(prepared/"matching.json"));write(out/"manifest.json",manifest)
    return dict(out=str(out),primitive_coordinates=len(representatives),jobs=len(jobs),profile_jobs=sum(j["split"]=="profile" for j in jobs),native_timing_started=False)


def collect(out, split):
    from profiling.state_calibration import run
    out=Path(out).resolve();manifest=read(out/"manifest.json")
    for source in manifest["sources"]:require(sha(source["path"])==source["sha256"],"bound executable source changed")
    barrier=None
    if split!="profile":
        barrier=read(out/"frozen-barrier.json");require(sha(out/"profile-frozen.json")==barrier["frozen_sha256"] and sha(out/"predictions.json")==barrier["predictions_sha256"],"prediction freeze changed")
    started=out/(split+"-started.json")
    if not started.exists():write(started,dict(started_at_utc=utc(),manifest=receipt(out/"manifest.json"),barrier=barrier))
    rows=[];program_cache={};reference_cache={}
    with run.measurement_lock():
        for job in (j for j in manifest["jobs"] if j["split"]==split):
            directory=out/"jobs"/job["id"]
            if (directory/"execution.json").exists():
                existing=read(directory/"execution.json");require(existing["job"]==job,"resume job differs");rows.append(existing);continue
            directory.mkdir(parents=True,exist_ok=False)
            binding=manifest["programs"][job["program_key"]]
            for field in ("program","reference"):require(sha(binding[field]["path"])==binding[field]["sha256"],"prepared artifact changed")
            if job["program_key"] not in program_cache:
                program_cache[job["program_key"]]=read(binding["program"]["path"]);reference_cache[job["program_key"]]=read(binding["reference"]["path"])
            program=program_cache[job["program_key"]];oracle=reference_cache[job["program_key"]]
            rp=directory/"request.json";output=directory/"result.json";write(rp,job["request"])
            command=[manifest["binary"],binding["program"]["path"],str(rp),str(output)]
            env=run.child_environment(manifest["runtime_dirs"]);env["OPENBLAS_NUM_THREADS"]="1"
            record=dict(schema="dac-e2a-process-1",job=job,started_at_utc=utc(),program=binding,status="failed",command=command,
                target_outcome_used_as_model_input=False,barrier=barrier)
            process=None;handles=[]
            try:
                process,stdout,stderr,affinity=run.launch(command,env,directory/"stdout.txt",directory/"stderr.txt",4);handles=[stdout,stderr]
                record.update(process_id=process.pid,affinity=affinity)
                code=process.wait(timeout=180);record.update(returncode=code,forced_termination=False,process_exited_verified=True)
                require(code==0,"native process failed")
                native=read(output);require(native["process_id"]==process.pid,"owned process identity differs")
                record["numerical"]=verify_output(program,oracle,job["request"],native)
                record.update(native=receipt(output),status="completed")
            except BaseException as error:
                record["error"]=repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    record["forced_termination"]=True
                if process is not None:record.update(returncode=process.returncode,process_exited_verified=process.poll() is not None)
            finally:
                for h in handles:h.close()
                record["completed_at_utc"]=utc();write(directory/"execution.json",record)
            rows.append(record);print(json.dumps(dict(job=job["id"],status=record["status"],error=record.get("error"))),flush=True)
    target=out/(split+"-executions.json")
    if not target.exists():write(target,rows)
    return dict(split=split,processes=len(rows),completed=sum(r["status"]=="completed" for r in rows),failed=sum(r["status"]!="completed" for r in rows))


def freeze(out):
    out=Path(out).resolve();manifest=read(out/"manifest.json")
    require(not(out/"heldout-started.json").exists(),"cannot freeze after target measurement")
    samples={};transfer={};evidence={};coordinates={}
    for execution in read(out/"profile-executions.json"):
        require(execution["status"]=="completed","independent profile failed; leave parameters unknown")
        native=read(execution["native"]["path"]);program=read(execution["program"]["program"]["path"])
        if program["format_version"]=="steady-component-program-1":
            shape=program["shape"];route=shape["src"]+"->"+shape["dst"]
            values=[r["wall_ns"]/r["repeats"] for r in native["trials"]]
            transfer.setdefault(route,[]).append(dict(bytes=shape["bytes"],mean_ns=statistics.mean(values),evidence=execution["native"]))
        else:
            groups=(native["groups"] if execution["job"]["request"]["mode"]=="all-actions" else
                [dict(action_index=execution["job"]["request"]["target_action_index"],trials=native["trials"])])
            for group in groups:
                coord=coordinate(program,group["action_index"],manifest["context"]);key=digest(coord)
                coordinates[key]=coord;samples.setdefault(key,[]).extend(Fraction(r["wall_ns"],r["repeats"]) for r in group["trials"])
                evidence.setdefault(key,[]).append(dict(native=execution["native"],action_index=group["action_index"],process_id=execution["process_id"],launch=execution["started_at_utc"]))
    entries=[dict(key=key,coordinate=coordinates[key],mean_ns=str(sum(values,Fraction())/len(values)),sample_count=len(values),evidence=evidence[key]) for key,values in sorted(samples.items())]
    transfer_fit={}
    for route,rows in transfer.items():
        sizes=sorted({r["bytes"] for r in rows});means={n:statistics.mean(r["mean_ns"] for r in rows if r["bytes"]==n) for n in sizes}
        x=np.asarray(sizes,dtype=float);y=np.asarray([means[n] for n in sizes]);slope,intercept=np.polyfit(x,y,1)
        predicted=intercept+slope*x;residual=np.abs(predicted-y)/y
        transfer_fit[route]=dict(fixed_ns=float(intercept),payload_ns_per_byte=float(slope),qualified=bool(intercept>=0 and slope>0 and np.max(residual)<=.10),
            maximum_relative_size_mean_residual=float(np.max(residual)),observations=rows,fit="ordinary least squares on ten size arithmetic process means; heldout extrapolation not claimed",
            fixed_term_not_added_to_inclusive_prices=True,contract="compact native copy exact backend path, known contiguous bytes; source residency initialized outside timing")
    frozen=dict(schema="dac-reusable-action-profile-1",frozen_at_utc=utc(),manifest=receipt(out/"manifest.json"),context=manifest["context"],entries=entries,
        transfer_decomposition=transfer_fit,fit="arithmetic means of isolated action batches only",complete_target_times_read=False,
        calibration_harness_full_initialization_and_warmup_are_untimed=True,
        calibration_harness_warmup_envelope_recorded_but_not_used_as_parameter=True,
        whole_program_identity_in_coordinate=False,extra_transition_parameters="unknown; omitted explicit zero additional term",cuda_private_workspace=None)
    write(out/"profile-frozen.json",frozen)
    table=ReusableResponseTable(tuple(entries));predictions=[]
    for wid in manifest["workloads"]:
        for strategy in range(4):
            key=f"{wid}-s{strategy}-heldout";program=read(manifest["programs"][key]["program"]["path"])
            begin=time.perf_counter();prediction=table.evaluate(program,manifest["context"]);wall=time.perf_counter()-begin
            breakdown={};copy_bytes={};action_counts={}
            for index,(action,elapsed) in enumerate(zip(program["actions"],prediction["action_ns"])):
                semantic=semantic_action(program,index)
                category=("compute:"+action["device"]+":"+semantic["semantic_op"] if action["kind"]=="COMPUTE" else
                    "copy:"+action["src"]+"->"+action["dst"] if action["kind"]=="COPY" else action["kind"].lower())
                breakdown[category]=breakdown.get(category,Fraction())+elapsed
                action_counts[category]=action_counts.get(category,0)+1
                if action["kind"]=="COPY":copy_bytes[category]=copy_bytes.get(category,0)+semantic["tensor"]["storage_bytes"]
            predictions.append(dict(workload=wid,strategy=strategy,program_key=key,prediction=number(prediction),simulation_wall_seconds=wall,
                prediction_breakdown_ns=number(breakdown),action_counts=action_counts,copy_bytes=copy_bytes,
                keys=[digest(coordinate(program,i,manifest["context"])) for i in range(len(program["actions"]))]))
    write(out/"predictions.json",predictions)
    write(out/"frozen-barrier.json",dict(frozen_at_utc=utc(),frozen_sha256=sha(out/"profile-frozen.json"),predictions_sha256=sha(out/"predictions.json"),heldout_read=False))
    return dict(entries=len(entries),predictions=len(predictions),transfer_fit={k:v["qualified"] for k,v in transfer_fit.items()})


def evaluate(out):
    out=Path(out).resolve();manifest=read(out/"manifest.json");predictions=read(out/"predictions.json");barrier=read(out/"frozen-barrier.json")
    require(sha(out/"profile-frozen.json")==barrier["frozen_sha256"] and sha(out/"predictions.json")==barrier["predictions_sha256"],"frozen prediction changed")
    held=read(out/"heldout-executions.json");observer=read(out/"observer-executions.json")
    rows=[];groups=[];rng=np.random.default_rng(20261024)
    for wid in manifest["workloads"]:
        observed=np.full((8,4),np.nan);qualified=[]
        for strategy in range(4):
            selected=[e for e in held if e["job"]["program_key"]==f"{wid}-s{strategy}-heldout"]
            prediction=float(Fraction(next(p["prediction"]["duration_ns"] for p in predictions if p["workload"]==wid and p["strategy"]==strategy)))
            processes=[];alltrials=[];observer_pairs=[]
            for e in selected:
                if e["status"]!="completed":continue
                require(e["started_at_utc"]>=barrier["frozen_at_utc"],"heldout predates freeze")
                native=read(e["native"]["path"]);values=[r["wall_ns"]/r["repeats"] for r in native["trials"]]
                avg=statistics.mean(values);cv=statistics.stdev(values)/avg;ape=abs(prediction-avg)/avg;alltrials.extend(values)
                observed[e["job"]["round"],strategy]=avg
                processes.append(dict(job_id=e["job"]["id"],mean_ns=avg,cv=cv,ape=ape,mean_gate=ape<=.10,cv_gate=cv<=.10))
            for pair in range(4):
                es=[e for e in observer if e["job"]["program_key"]==f"{wid}-s{strategy}-heldout" and e["job"]["pair"]==pair]
                if len(es)!=2 or any(e["status"]!="completed" for e in es):observer_pairs.append(dict(pair=pair,passed=False,status="incomplete"));continue
                means={e["job"]["request"]["mode"]:statistics.mean(r["wall_ns"]/r["repeats"] for r in read(e["native"]["path"])["trials"]) for e in es}
                effect=means["full-stamped"]/means["full"]-1
                observer_pairs.append(dict(pair=pair,relative_difference=effect,passed=abs(effect)<=.05))
            p90=None if not alltrials else quantile([abs(prediction-v)/v for v in alltrials],.9)
            quality=bool(len(processes)==8 and all(p["mean_gate"] and p["cv_gate"] for p in processes) and p90 is not None and p90<=.2 and all(p["passed"] for p in observer_pairs))
            qualified.append(quality)
            rows.append(dict(workload=wid,strategy=strategy,prediction_ns=prediction,completed_processes=len(processes),processes=processes,
                mean_real_ns=None if not processes else statistics.mean(p["mean_ns"] for p in processes),trial_p90_ape=p90,
                between_process_cv=None if len(processes)<2 else statistics.stdev(p["mean_ns"] for p in processes)/statistics.mean(p["mean_ns"] for p in processes),
                observer_pairs=observer_pairs,fully_qualified=quality))
        pred=np.asarray([r["prediction_ns"] for r in rows[-4:]]);chosen=int(np.argmin(pred))
        group=dict(workload=wid,predicted_choice=chosen,all_candidates_qualified=all(qualified),all_processes_complete=bool(np.all(np.isfinite(observed))))
        if group["all_processes_complete"]:
            means=observed.mean(axis=0);best=int(np.argmin(means));regret=float(means[chosen]/means[best]-1)
            boot=np.asarray([observed[index].mean(axis=0) for index in rng.integers(0,8,size=(10000,8))]);br=boot[:,chosen]/boot.min(axis=1)-1
            fixed_regret=(means/means.min()-1).tolist();restricted=float(means[:2].min()/means.min()-1)
            fullbest=boot.min(axis=1);bg=boot[:,:2].min(axis=1)/fullbest-1
            group.update(real_choice=best,real_mean_ns=means.tolist(),regret=regret,regret_simultaneous_upper=quantile(br,1-.05/6),
                regret_interval=[quantile(br,.025),quantile(br,.975)],fixed_strategy_regret=fixed_regret,
                cpu_gpu_only_gap=restricted,cpu_gpu_only_gap_familywise_lower=quantile(bg,.05/6),
                point_regret_gate=regret<=.05,upper_regret_gate=quantile(br,1-.05/6)<=.10,
                limitation_witness=restricted>=.10 and quantile(bg,.05/6)>0)
        groups.append(group)
    summary=dict(schema="dac-real-selection-result-1",evaluated_at_utc=utc(),manifest=receipt(out/"manifest.json"),profile=receipt(out/"profile-frozen.json"),predictions=receipt(out/"predictions.json"),
        workloads=groups,programs=rows,counts=dict(declared_groups=6,programs=24,heldout_expected=192,observer_expected=192,
            heldout_completed=sum(e["status"]=="completed" for e in held),observer_completed=sum(e["status"]=="completed" for e in observer),
            qualified_programs=sum(r["fully_qualified"] for r in rows),qualified_groups=sum(g["all_candidates_qualified"] for g in groups),
            complete_groups=sum(g["all_processes_complete"] for g in groups),regret_point_passes=sum(g.get("point_regret_gate",False) for g in groups),
            regret_upper_passes=sum(g.get("upper_regret_gate",False) for g in groups),limitation_witnesses=sum(g.get("limitation_witness",False) for g in groups)),
        quality=QUALITY,claims=dict(independent_component_profiles=True,target_times_used_to_fit=False,unseen_shape_extrapolation=False,
            real_prefetch_or_eviction_validated=False,concurrent_tensor_parallelism=False,autogressive_generation=False,
            numerical_scope="complete declared Dense block, every materialized port; shared parent weights"),
        limitations=manifest["protocol"])
    write(out/"summary.json",summary)
    process_rows=[];trial_rows=[]
    for row in rows:
        for process in row["processes"]:
            process_rows.append(dict(workload=row["workload"],strategy=row["strategy"],prediction_ns=row["prediction_ns"],
                program_fully_qualified=row["fully_qualified"],**process))
    for execution in held:
        if execution["status"]!="completed":continue
        binding=manifest["programs"][execution["job"]["program_key"]]
        for trial in read(execution["native"]["path"])["trials"]:
            trial_rows.append(dict(workload=binding["workload"],strategy=binding["strategy"],round=execution["job"]["round"],
                process_id=execution["process_id"],job_id=execution["job"]["id"],trial=trial["trial"],
                wall_ns=trial["wall_ns"],replays=trial["repeats"],average_ns=trial["wall_ns"]/trial["repeats"],
                native_sha256=execution["native"]["sha256"]))
    with (out/"process_metrics.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(process_rows[0]) if process_rows else ["workload","strategy","status"],lineterminator="\n")
        writer.writeheader();writer.writerows(process_rows)
    with (out/"trial_metrics.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trial_rows[0]) if trial_rows else ["workload","strategy","status"],lineterminator="\n")
        writer.writeheader();writer.writerows(trial_rows)
    with (out/"selection_metrics.csv").open("x",encoding="utf-8",newline="") as stream:
        fields=["workload","predicted_choice","real_choice","all_candidates_qualified","all_processes_complete","regret","regret_simultaneous_upper","cpu_gpu_only_gap","cpu_gpu_only_gap_familywise_lower","point_regret_gate","upper_regret_gate","limitation_witness"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore",lineterminator="\n");writer.writeheader();writer.writerows(groups)
    artifacts=[receipt(p) for p in out.rglob("*") if p.is_file() and p.name!="artifact-manifest.json"]
    write(out/"artifact-manifest.json",dict(schema="dac-e2a-artifacts-1",files=artifacts,files_sha256=digest(artifacts)))
    return summary["counts"]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","refine-prethermal","profile","freeze","heldout","observer","evaluate"));parser.add_argument("--out",type=Path,required=True);parser.add_argument("--prepared",type=Path)
    args=parser.parse_args();result=(prepare(args.out) if args.command=="prepare" else refine_prethermal(args.out,args.prepared) if args.command=="refine-prethermal" else
        freeze(args.out) if args.command=="freeze" else evaluate(args.out) if args.command=="evaluate" else collect(args.out,args.command))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
if __name__=="__main__":main()
