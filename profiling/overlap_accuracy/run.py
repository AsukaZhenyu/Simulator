"""Source-bound exclusive collection, profile freeze, heldout and offline replay."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCK_PATH=ROOT/"profiling/.measurement.lock"
if str(HERE)not in sys.path:sys.path.insert(0,str(HERE))
import contract as c
import bridge
import importlib.util
_runtime_spec=importlib.util.spec_from_file_location("overlap_strict_runtime_manifest",ROOT/"profiling/chain_calibration/runtime_manifest.py")
runtime_manifest=importlib.util.module_from_spec(_runtime_spec);_runtime_spec.loader.exec_module(runtime_manifest)


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def inventory(paths):return{str(Path(p).resolve()):c.sha(p)for p in sorted(set(map(str,paths)))}


def command(binary,case,protocol,seed):
    options=dict(family="ffn",backend="gpu",hidden=case["hidden"],intermediate=case["intermediate"],tokens=case["tokens"],
        history=0,heads=0,kv_heads=0,head_dim=0,**protocol,seed=seed,dtype="f32",cuda_graphs="disabled",payload_bytes=case["payload_bytes"],mode="measure")
    return[str(binary),*(value for key,item in options.items()for value in("--"+key.replace("_","-"),str(item)))]


def jobs(suite):
    plan=[]
    for split in("profile","heldout"):
        for seed in suite[split+"_seeds"]:
            for repeat in range(suite["repeats"]):
                cases=list(suite["cases"]);random.Random(seed+repeat).shuffle(cases)
                for case in cases:plan.append(dict(id=f"{split}-{case['id']}-s{seed}-r{repeat}",case=case,split=split,seed=seed,repeat=repeat))
    return plan


def execute(job,args,suite,out,environment,probe_sha):
    row=copy.deepcopy(job);row["command"]=command(args.binary,row["case"],suite["protocol"],row["seed"])
    raw=dict(command=row["command"],started_utc=now(),frozen_model_sha256_at_start=row.get("frozen_model_sha256_at_start"),
             private_runtime_manifest_sha256_at_start=row.get("private_runtime_manifest_sha256_at_start",{}),
             input_environment={key:environment.get(key)for key in sorted(environment)if key.startswith(("GGML_","CUDA_","NVIDIA_","CUBLAS_"))})
    start=time.perf_counter_ns()
    try:
        process=subprocess.Popen(row["command"],env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE);raw["child_pid"]=process.pid
        try:stdout,stderr=process.communicate(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            process.kill();stdout,stderr=process.communicate();raw["timed_out"]=True
        raw.update(returncode=process.returncode,stdout=stdout.decode(errors="replace"),stderr=stderr.decode(errors="replace"))
        if process.returncode or raw.get("timed_out"):raise ValueError("native nonzero exit/timeout")
        payload=json.loads(raw["stdout"]);c.validate_native(row["case"],suite["protocol"],row["seed"],payload,probe_sha)
        row.update(status="measured",metrics=payload["wall_ns"],known_workspace_bytes=payload["known_gallocr_workspace_bytes"],child_pid=process.pid,
                   diagnostic_host_and_copy_events=payload["diagnostic_host_and_copy_events"],instrumentation_overhead_ns=payload["instrumentation_overhead_ns"])
    except(OSError,ValueError,KeyError,TypeError)as exc:
        row.update(status="failed",error=str(exc));raw["validation_error"]=str(exc)
    raw.update(completed_utc=now(),process_wall_ns=time.perf_counter_ns()-start)
    path=out/"raw"/(row["id"]+".json");c.dump(path,raw);row.update(raw_path=str(path),raw_sha256=c.sha(path))
    print(row["id"]+": "+row["status"],flush=True)
    return row


def frozen_bundle(profile,suite):
    costs=c.freeze(profile,suite);programs=[]
    for case_model in costs["cases"]:
        case=case_model["case"]
        sources=[r for r in profile if r["case"]==case and r["status"]=="measured"]
        workspace=max((r["known_workspace_bytes"]for r in sources),default=0)
        for strategy in c.STRATEGIES:
            try:
                if set(case_model["prices"])!=set(c.COMPONENTS):raise ValueError("missing independent profile components")
                program=bridge.bind(case,strategy,case_model["prices"],workspace)
            except(ValueError,KeyError,TypeError)as exc:program=dict(case_id=case["id"],strategy=strategy,status="unknown",error=str(exc))
            programs.append(program)
    return dict(schema="overlap-frozen-cost-and-compiled-programs-0.1",cost_model=costs,programs=programs,
                heldout_used_for_fitting=False,all_programs_frozen_before_heldout=True)


def evaluate(rows,bundle,suite):
    results=[]
    price_cases={m["case"]["id"]:m for m in bundle["cost_model"]["cases"]}
    for program in bundle["programs"]:
        cid,strategy=program["case_id"],program["strategy"]
        selected=[r for r in rows if r["case"]["id"]==cid]
        target=[r for r in selected if r["split"]=="heldout"and r["status"]=="measured"]
        metric_sources=["copy/a","compute/current_ffn"]+(["copy/b"]if"two_copy"in strategy else[])
        prices=price_cases[cid]["prices"]
        sources=[s for metric in metric_sources for s in prices.get(metric,{}).get("sources",[])]
        source_between={metric:c.cv([s["median_ns"]for s in prices[metric]["sources"]])for metric in metric_sources if metric in prices}
        values=[v for row in target for v in row["metrics"][strategy]]
        observed=statistics.median(values)if values else None;prediction=program.get("predicted_ns")
        error=abs(prediction-observed)/observed if prediction is not None and observed else None
        target_cv=[c.cv(row["metrics"][strategy])for row in target]
        between=c.cv([statistics.median(row["metrics"][strategy])for row in target])if len(target)>=2 else None
        expected={split:len(suite[split+"_seeds"])*suite["repeats"]for split in("profile","heldout")}
        gates=dict(all_declared_processes_measured=all(r["status"]=="measured"for r in selected)and all(sum(r["split"]==split for r in selected)==n for split,n in expected.items()),
            minimum_processes=len(target)>=3 and len({s["id"]for s in sources})>=3,
            common_fixed_compiler_runtime=program.get("common_fixed_tensor_compiler_runtime_exercised")is True,
            frozen_before_heldout=bool(target)and all(r.get("frozen_model_sha256_at_start")for r in target),
            within_process_cv=bool(sources)and bool(target_cv)and all(x["cv"]<=.15 for x in sources)and all(v<=.15 for v in target_cv),
            between_process_cv=len(source_between)==len(metric_sources)and all(v<=.20 for v in source_between.values())and between is not None and between<=.20,
            heldout_relative_error=error is not None and error<=.20)
        results.append(dict(case_id=cid,strategy=strategy,predicted_ns=prediction,observed_ns=observed,absolute_relative_error=error,
            independent_links_ideal_ns=program.get("independent_links_ideal_ns"),
            independent_links_ideal_relative_error=abs(program["independent_links_ideal_ns"]-observed)/observed if program.get("independent_links_ideal_ns")is not None and observed else None,
            source_evidence=sources,source_between_process_cv=source_between,heldout_processes=len(target),
            heldout_within_process_cv=target_cv,heldout_between_process_cv=between,gates=gates,accepted=all(gates.values()),
            unknown_reason=program.get("error")))
    return dict(schema="overlap-accuracy-report-0.1",quality=c.QUALITY,results=results,
        failed_rows=[r["id"]for r in rows if r["status"]!="measured"],all_paths_accepted=bool(results)and all(r["accepted"]for r in results),
        kernel_overlap_proved=False,copy_events_prove_kernel_overlap=False,physical_capacity_verified=False,
        interpretation="tests fixed serial-sum and shared-link overlap-max laws against new completed-work heldout processes; no arbitrary contention law",
        highlevel_ModelPlan_coverage=False,
        common_fixed_tensor_compiler_runtime_exercised=bool(bundle["programs"])and all(p.get("common_fixed_tensor_compiler_runtime_exercised")is True for p in bundle["programs"]),
        instrumentation_overhead_subtracted=False)


def source_files():
    return[*HERE.glob("*.py"),*HERE.glob("*.json"),*ROOT.joinpath("profiling/native/overlap_accuracy").glob("*.py"),
        ROOT/"profiling/native/overlap_accuracy/compute.cpp",ROOT/"profiling/native/overlap_accuracy/CMakeLists.txt",
        ROOT/"profiling/native/execution_transforms_dual/compute_context.cpp",ROOT/"profiling/native/matched_mechanisms/compute.cpp",
        ROOT/"profiling/chain_calibration/runtime_manifest.py",
        *ROOT.joinpath("modeling/llm_infer_model/tensor").glob("*.py"),
        ROOT/"mapping/tensor_mapping/strategy_compiler.py",ROOT/"mapping/tensor_mapping/strategy_ir.py"]


def run(args):
    suite=c.validate_suite(load(args.suite));args.binary=args.binary.resolve();out=args.output.resolve()
    if out.exists():raise ValueError("new output directory required")
    if args.timeout_s<1 or not args.binary.is_file()or any(not p.is_dir()for p in args.runtime_dir):raise ValueError("explicit binary/runtime paths and timeout required")
    plan=jobs(suite)
    if args.dry_run:print(json.dumps(dict(schema="overlap-dry-run-0.1",jobs=plan),indent=2));return 0
    build=args.binary.parent/"build-record.json";record=load(build)
    probe=ROOT/"profiling/native/overlap_accuracy/compute.cpp"
    if record.get("returncode")!=0 or record.get("binary_sha256")!=c.sha(args.binary)or record["sources"]["probe_sha256"]!=c.sha(probe):raise ValueError("native build/source binding differs")
    expected_sources=dict(probe_sha256=c.sha(probe),context_sha256=c.sha(ROOT/"profiling/native/execution_transforms_dual/compute_context.cpp"),
        matched_sha256=c.sha(ROOT/"profiling/native/matched_mechanisms/compute.cpp"),
        generated_include_sha256=c.sha(args.binary.parent/"context_generated.inc"),cmake_sha256=c.sha(probe.parent/"CMakeLists.txt"),build_script_sha256=c.sha(probe.parent/"build.py"))
    if record["sources"]!=expected_sources or expected_sources["context_sha256"]!=c.CONTEXT_SHA or expected_sources["matched_sha256"]!=c.MATCHED_SHA:
        raise ValueError("generated/helper/build sources differ from native build")
    if any(c.sha(path)!=value for group in("linked_libraries","ggml_headers")for path,value in record[group].items()):raise ValueError("linked libraries/headers changed after build")
    assets=[args.suite,args.binary,build,args.binary.parent/"context_generated.inc",*source_files(),*record["linked_libraries"],*record["ggml_headers"]]
    for directory in args.runtime_dir:assets.extend(p for p in directory.iterdir()if p.is_file()and p.suffix.lower()==".dll")
    private_runtimes,private_assets=runtime_manifest.discover(args.runtime_dir)
    assets.extend(private_assets)
    before=inventory(assets)
    environment=os.environ.copy();environment["PATH"]=os.pathsep.join([*(str(p.resolve())for p in args.runtime_dir),environment.get("PATH","")]);environment["NVIDIA_TF32_OVERRIDE"]="0";environment["GGML_CUDA_DISABLE_GRAPHS"]="1"
    lock=LOCK_PATH;fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,"w",encoding="utf-8")as stream:stream.write(json.dumps(dict(pid=os.getpid(),output=str(out),started_utc=now())))
    manifest=None
    try:
        out.mkdir(parents=True);(out/"raw").mkdir();(out/"source").mkdir()
        snapshots={}
        for n,path in enumerate(source_files()):
            dest=out/"source"/(f"{n:03d}-"+path.name);dest.write_bytes(path.read_bytes());snapshots[str(path.resolve())]=dict(path=str(dest.relative_to(out)),sha256=c.sha(dest))
        c.dump(out/"suite.json",suite);c.dump(out/"plan.json",plan)
        manifest=dict(schema="overlap-accuracy-manifest-0.1",started_utc=now(),inventory_before=before,source_snapshots=snapshots,
            bound_binary=str(args.binary),private_runtime_builds=private_runtimes,
            plan_sha256_before_collection=c.sha(out/"plan.json"),sample_filtering="none",requested_runtime_flags=dict(NVIDIA_TF32_OVERRIDE="0",GGML_CUDA_DISABLE_GRAPHS="1"))
        c.dump(out/"manifest.json",manifest);rows=[]
        for split in("profile","heldout"):
            for job in(x for x in plan if x["split"]==split):
                job=dict(job,private_runtime_manifest_sha256_at_start=runtime_manifest.hash_bindings(private_runtimes))
                if inventory(assets)!=before:raise ValueError("source/binary/runtime changed before process")
                if c.sha(out/"plan.json")!=manifest["plan_sha256_before_collection"]:raise ValueError("predeclared process plan changed")
                if split=="heldout":
                    if c.sha(out/"frozen-model.json")!=manifest["model_frozen_before_heldout_sha256"]:raise ValueError("frozen costs/programs changed")
                    job=dict(job,frozen_model_sha256_at_start=manifest["model_frozen_before_heldout_sha256"])
                rows.append(execute(job,args,suite,out,environment,c.sha(probe)));c.dump(out/"measurements.json",rows)
                if inventory(assets)!=before:raise ValueError("source/binary/runtime changed during process")
            if split=="profile":
                bundle=frozen_bundle(rows,suite);c.dump(out/"frozen-model.json",bundle)
                manifest.update(model_frozen_before_heldout_sha256=c.sha(out/"frozen-model.json"),model_frozen_utc=now());c.dump(out/"manifest.json",manifest)
        if c.sha(out/"frozen-model.json")!=manifest["model_frozen_before_heldout_sha256"]:raise ValueError("model changed during final heldout process")
        report=evaluate(rows,bundle,suite);c.dump(out/"report.json",report)
        manifest.update(completed_utc=now(),inventory_after=inventory(assets),assets_unchanged=inventory(assets)==before,report_sha256=c.sha(out/"report.json"));c.dump(out/"manifest.json",manifest)
        print(json.dumps(dict(output=str(out),processes=len(rows),paths=len(report["results"]),accepted=sum(r["accepted"]for r in report["results"]))))
        return 0 if not report["failed_rows"] else 2
    except BaseException as exc:
        if manifest is not None:manifest.update(aborted_utc=now(),error=str(exc));c.dump(out/"manifest.json",manifest)
        raise
    finally:lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in("suite","binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True);parser.add_argument("--timeout-s",type=int,required=True);parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    try:return run(args)
    except(OSError,ValueError,KeyError,TypeError)as exc:print("overlap accuracy: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
