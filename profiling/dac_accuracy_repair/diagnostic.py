"""Six fixed paired D2H diagnostics; no parameter fitting or gate replacement.

Run prepare, collect, then summarize. Existing output and failed jobs are never
overwritten or retried. Registration is a mechanistic control, not permission to
substitute pinned memory for the historical pageable target.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from profiling.dac_followup.task1 import byte_reference
from profiling.state_calibration import run

ROOT = Path(__file__).resolve().parents[2]
API = "ggml_tensor_copy_then_gpu_cpu_synchronize"

def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def require(ok, message):
    if not ok: raise ValueError(message)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False); stream.write("\n")
def cv(values): return statistics.pstdev(values) / statistics.mean(values)
def csv_new(path, rows):
    if not rows: return
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

def prepare(out, binary):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = Path(binary).resolve()
    runtime_dirs = [str(ROOT/"profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin"),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/lib64"]
    native = ROOT/"profiling/dac_accuracy_repair/native/repair_probe.cpp"
    source_paths = [Path(__file__), native, native.parent/"CMakeLists.txt",
        ROOT/"profiling/native/steady_costs/probe.cpp", ROOT/"profiling/native/steady_costs/components.hpp",
        ROOT/"profiling/dac_followup/task1.py", ROOT/"profiling/state_calibration/run.py"]
    source_paths += list((ROOT/"profiling/native/steady_costs/build-release-v1").glob("*.inc"))
    sources = []
    for source in source_paths:
        target = out/"executed_sources"/source.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        sources.append(dict(source=receipt(source), archived=receipt(target)))
    archived_binary = out/"executed_binary"/binary.name
    archived_binary.parent.mkdir(); shutil.copy2(binary, archived_binary)
    runtime = {str(path.resolve()): sha(path) for folder in runtime_dirs for path in Path(folder).glob("*.dll")}
    require(runtime, "no explicit runtime DLLs found")
    programs = {}
    for size in (262144, 33554432):
        program = dict(format_version="steady-component-program-1", kind="transfer", device="gpu", threads=1,
            cuda_device=0, seed=2026102605, shape=dict(bytes=size, src="vram", dst="dram"), allocation_budget_bytes=1<<30)
        program["program_sha256"] = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = out/"programs"/f"d2h-{size}.json"; oracle = out/"references"/f"d2h-{size}.json"
        write(path, program); write(oracle, byte_reference(program["seed"], size))
        programs[str(size)] = dict(program=receipt(path), reference=receipt(oracle))
    sampling = dict(schema="steady-cost-request-1", mode="action", target_action_index=None, warmup=20,
        warmup_min_ms=500, trials=8, repeats=256, poll="0", instrumentation="bare", main_thread_mask=4)
    jobs = []
    for round_id in range(3):
        for size in ((262144, 33554432) if round_id % 2 == 0 else (33554432, 262144)):
            job_id = f"paired-r{round_id}-{size}"
            request = dict(schema="dac-accuracy-repair-request-1", mode="transfer-diagnostic", sampling=sampling,
                source_sha256=sha(native), order="ABBA" if round_id%2==0 else "BAAB",
                memory_order="ABBA" if round_id%2==0 else "BAAB")
            request_path = out/"requests"/(job_id+".json"); write(request_path, request)
            jobs.append(dict(id=job_id, bytes=size, round=round_id, request=receipt(request_path)))
    manifest = dict(schema="dac-accuracy-repair-paired-diagnostic-1", prepared_at_utc=utc(),
        expected_native_processes=6, per_child_timeout_seconds=600, retries=0, jobs=jobs, programs=programs,
        binary=receipt(archived_binary), original_binary=receipt(binary), source_sha256=sha(native), sources=sources,
        runtime_dirs=runtime_dirs, runtime_sha256=runtime, runtime_archive_scope="SHA256/path manifest; installed runtime DLLs not duplicated",
        protocol=dict(sizes=[262144,33554432], fresh_processes_per_size=3, memory_visits=4,
            each_visit_observer_blocks="ABBA or BAAB", sampling=sampling, same_owned_destination=True,
            changed_registered_control_does_not_replace_pageable_target=True, parameter_fit=False,
            hypothesis="registered control may reduce paired response and between-process spread; failure rejects that proposed repair",
            diagnostic_thresholds=dict(max_trial_cv=.10,max_observer_relative_difference=.05,
                same_input_ten_percent_feasible_max_min_ratio=1.1/.9),
            causal_scope="balanced within-process registration intervention; no identification of a unique internal staging or clock cause"))
    write(out/"manifest.json", manifest)
    return dict(prepared_jobs=6, out=str(out))

def verify(native, request, binding, pid):
    require(native["schema"]=="dac-accuracy-repair-native-1" and native["request"]==request, "native/request identity differs")
    require(native["process_id"]==pid and native["source_sha256"]==request["source_sha256"] and native["caller_source_sha256_match"], "source/process differs")
    program=read(binding["program"]["path"]); oracle=read(binding["reference"]["path"])
    require(native["program_sha256"]==program["program_sha256"], "program identity differs")
    require(native["measurement_api"]==API and native["host_registration_restored"] and native["same_owned_destination_buffer_all_modes"], "transfer mechanism/restore differs")
    require(native["main_thread_affinity"]["requested_mask"]==4 and native["main_thread_affinity"]["restored"], "owned thread affinity differs")
    require(len(native["visits"])==4 and len(native["trials"])==128, "diagnostic visit/block count differs")
    hz=native["clock"]["frequency_hz"]; previous=-1
    for visit, group in enumerate(native["visits"]):
        expected="pageable" if request["memory_order"][visit]=="A" else "registered"
        require(group["visit"]==visit and group["memory_mode"]==expected, "memory order differs")
        checked=group["full_byte_verification"]
        require(checked["full_byte_equality"] and checked["checked_bytes"]==oracle["bytes"] and checked["source_sha256"]==checked["output_sha256"]==oracle["sha256"], "complete independent byte oracle failed")
        for row in group["trials"]:
            require(row["begin_tick"]>=previous and row["end_tick"]>row["begin_tick"], "QPC chronology differs")
            require(abs((row["end_tick"]-row["begin_tick"])*1e9/hz-row["wall_ns"])<=1, "QPC duration differs")
            require(row["repeats"]==256 and row["memory_mode"]==expected, "repeat/memory identity differs")
            require(row["host_pointer"]["registered_host" if expected=="registered" else "unregistered_pageable"], "actual host class differs")
            require(row["preblock_state"]["sample_tick"]<=row["begin_tick"] and row["preblock_state"]["derived_from_target_duration"] is False, "state is not independent preblock observation")
            require(row["actual_stamped"]==(request["order"][row["block"]]=="B"), "observer order differs")
            previous=row["end_tick"]
    return dict(passed=True, full_payload_bytes=oracle["bytes"], visits=4, QPC_rows=128,
        actual_registration_and_restore=True, full_byte_oracle=oracle)

def collect(out):
    out=Path(out).resolve(); manifest=read(out/"manifest.json")
    require(manifest["expected_native_processes"]==6, "fixed process count differs")
    for item in manifest["sources"]:
        require(sha(item["source"]["path"])==item["source"]["sha256"], "executed source changed")
        require(sha(item["archived"]["path"])==item["archived"]["sha256"], "archived source changed")
    for path,digest in manifest["runtime_sha256"].items(): require(sha(path)==digest,"runtime DLL changed")
    require(sha(manifest["binary"]["path"])==manifest["binary"]["sha256"], "frozen diagnostic binary changed")
    records=[]
    with run.measurement_lock():
        for job in manifest["jobs"]:
            folder=out/"jobs"/job["id"]
            if (folder/"execution.json").exists(): records.append(read(folder/"execution.json")); continue
            folder.mkdir(parents=True,exist_ok=False)
            binding=manifest["programs"][str(job["bytes"])]; req=read(job["request"]["path"])
            for item in [job["request"],binding["program"],binding["reference"]]: require(sha(item["path"])==item["sha256"],"frozen request/program/oracle changed")
            command=[manifest["binary"]["path"],binding["program"]["path"],job["request"]["path"],str(folder/"result.json")]
            record=dict(job=job,started_at_utc=utc(),command=command,status="failed"); process=None; handles=[]
            try:
                env=run.child_environment(manifest["runtime_dirs"]);env["OPENBLAS_NUM_THREADS"]="1"
                process,stdout,stderr,affinity=run.launch(command,env,folder/"stdout.txt",folder/"stderr.txt",4)
                handles=[stdout,stderr];record.update(process_id=process.pid,affinity=affinity)
                code=process.wait(timeout=600);record.update(returncode=code,forced_termination=False,process_exited_verified=True)
                require(code==0,"native diagnostic failed")
                record["verification"]=verify(read(folder/"result.json"),req,binding,process.pid)
                record.update(status="completed",raw=receipt(folder/"result.json"))
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
            records.append(record);print(json.dumps(dict(job=job["id"],status=record["status"],completed=len(records),of=6,error=record.get("error"))),flush=True)
    write(out/"executions.json",records)
    return dict(completed=sum(r["status"]=="completed" for r in records),failed=sum(r["status"]!="completed" for r in records),expected=6)

def summarize(out):
    out=Path(out).resolve();manifest=read(out/"manifest.json");records=read(out/"executions.json")
    processes=[]; blocks=[]; states=[]
    for record in records:
        if record["status"]!="completed": continue
        native=read(record["raw"]["path"])
        for row in native["trials"]:
            flat=dict(job=record["job"]["id"],bytes=record["job"]["bytes"],visit=row["visit"],trial=row["trial"],block=row["block"],memory_mode=row["memory_mode"],actual_stamped=row["actual_stamped"],response_ns=row["wall_ns"]/row["repeats"],copy_api_ns=None if row["copy_call_ns"] is None else row["copy_call_ns"]/row["repeats"],gpu_sync_ns=None if row["gpu_sync_ns"] is None else row["gpu_sync_ns"]/row["repeats"],cpu_sync_ns=None if row["cpu_sync_ns"] is None else row["cpu_sync_ns"]/row["repeats"],raw_sha256=record["raw"]["sha256"])
            blocks.append(flat);states.append(dict(job=flat["job"],visit=row["visit"],trial=row["trial"],block=row["block"],memory_mode=row["memory_mode"],preblock_state_json=json.dumps(row["preblock_state"],sort_keys=True),raw_sha256=flat["raw_sha256"]))
        for mode in ("pageable","registered"):
            local=[r for r in native["trials"] if r["memory_mode"]==mode]
            bare=[r["wall_ns"]/r["repeats"] for r in local if not r["actual_stamped"]]
            stamped=[r["wall_ns"]/r["repeats"] for r in local if r["actual_stamped"]]
            # Each visit/trial contributes the average of its two bare blocks.
            trials=[statistics.mean(r["wall_ns"]/r["repeats"] for r in local if r["visit"]==v and r["trial"]==t and not r["actual_stamped"]) for v in sorted({r["visit"] for r in local}) for t in range(8)]
            processes.append(dict(job=record["job"]["id"],bytes=record["job"]["bytes"],memory_mode=mode,mean_ns=statistics.mean(bare),trial_cv=cv(trials),observer_relative_difference=statistics.mean(stamped)/statistics.mean(bare)-1,raw_sha256=record["raw"]["sha256"]))
    sizes=[]
    for size in (262144,33554432):
        group=dict(bytes=size,modes={})
        for mode in ("pageable","registered"):
            local=[r for r in processes if r["bytes"]==size and r["memory_mode"]==mode];means=[r["mean_ns"] for r in local]
            group["modes"][mode]=dict(completed_processes=len(local),means_ns=means,
                between_process_sample_cv=statistics.stdev(means)/statistics.mean(means) if len(means)>1 else None,
                maximum_minimum_ratio=max(means)/min(means) if means else None,
                ten_percent_same_input_feasible=max(means)/min(means)<=1.1/.9 if len(means)==3 else None,
                trial_cv_passes=sum(r["trial_cv"]<=.1 for r in local),observer_passes=sum(abs(r["observer_relative_difference"])<=.05 for r in local))
        pairs=[]
        for job in manifest["jobs"]:
            if job["bytes"]!=size:continue
            local={r["memory_mode"]:r for r in processes if r["job"]==job["id"]}
            if len(local)==2:pairs.append(dict(job=job["id"],registered_over_pageable=local["registered"]["mean_ns"]/local["pageable"]["mean_ns"]))
        group["within_process_pairs"]=pairs;sizes.append(group)
    result=dict(schema="dac-accuracy-repair-paired-diagnostic-result-1",summarized_at_utc=utc(),expected_processes=6,
        completed=sum(r["status"]=="completed" for r in records),failed=sum(r["status"]!="completed" for r in records),sizes=sizes,
        independent_parameter_fit=False,historical_gates_recomputed=False,
        scope="paired memory-registration diagnostic only; does not qualify or replace historical pageable target",
        all_owned_children_exited=all(r.get("process_exited_verified") for r in records),
        manifest=receipt(out/"manifest.json"),raws=[r["raw"] for r in records if "raw" in r])
    csv_new(out/"paired_process_metrics.csv",processes);csv_new(out/"paired_block_metrics.csv",blocks);csv_new(out/"preblock_states.csv",states)
    write(out/"summary.json",result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","collect","summarize"));parser.add_argument("--out",type=Path,required=True);parser.add_argument("--binary",type=Path)
    args=parser.parse_args();result=prepare(args.out,args.binary) if args.command=="prepare" else collect(args.out) if args.command=="collect" else summarize(args.out)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=="__main__": main()
