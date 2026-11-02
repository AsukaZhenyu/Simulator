"""Staged hardware campaign for actual compiled high-level Dense strategies.

Prepare never executes hardware. profile-all freezes ALL cases before any
heldout-all execution, which runs in separate processes with disjoint data seeds.
Both hardware phases honor profiling/.measurement.lock. Existing outputs are
never overwritten. There is deliberately no default all-in-one hardware action.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
for folder in (ROOT/"modeling",ROOT/"mapping",ROOT/"profiling"):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))

from highlevel_accuracy.bridge import RUNTIME_FLAGS, BridgeError, export_plan, sha, validate_program, verify_native
from highlevel_accuracy.cases import build_case
from highlevel_accuracy.profiles import GATES, binary_sha, compare_heldout, cv, freeze_profile, run_plan_with_profile, utc_now, validate_ledger


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def hardware_lock():
    path=ROOT/"profiling/.measurement.lock"
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:raise BridgeError(f"another hardware run holds {path}; inspect its PID before clearing a stale lock") from exc
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:stream.write(json.dumps(dict(pid=os.getpid(),campaign="highlevel_accuracy",started_at=utc_now())))
        yield
    finally:path.unlink(missing_ok=True)


def case_specs(suite):
    if suite=="smoke":
        base=dict(hidden=16,intermediate=32,heads=4,kv_heads=2,head_dim=4,tokens=4,history=0)
        return [dict(id="cpu-macros",kwargs={**base,"devices":["cpu"]*3}),
                dict(id="cuda-macros",kwargs={**base,"devices":["gpu"]*3}),
                dict(id="mixed-split",kwargs={**base,"attention_stages":True,"ffn_cut":16})]
    cases=[]
    for shape in (dict(hidden=64,intermediate=128,heads=4,kv_heads=2,head_dim=16,tokens=1,history=128),
                  dict(hidden=128,intermediate=256,heads=4,kv_heads=2,head_dim=32,tokens=16,history=0),
                  dict(hidden=256,intermediate=512,heads=8,kv_heads=2,head_dim=32,tokens=32,history=0)):
        prefix=f"h{shape['hidden']}-q{shape['tokens']}-history{shape['history']}"
        cases.append(dict(id=prefix+"-hybrid-macros",kwargs=shape))
        if shape["tokens"]==1:
            for device in ("cpu","gpu"):cases.append(dict(id=prefix+"-"+device+"-macros",kwargs={**shape,"devices":[device]*3}))
        for offset in (-4,0,4):
            cut=shape["intermediate"]//2+offset
            cases.append(dict(id=prefix+f"-attention-ffn-cut{cut}",kwargs={**shape,"attention_stages":True,"ffn_cut":cut}))
    return cases


def prepare(directory,suite):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise BridgeError("prepare output must be a new or empty directory")
    rows=[]
    for index,item in enumerate(case_specs(suite)):
        case=directory/item["id"];plan=build_case(case/"inputs",**item["kwargs"]);compiled=plan.compile()
        geometry={key:item["kwargs"].get(key,default) for key,default in dict(hidden=16,intermediate=32,heads=4,kv_heads=2,head_dim=4,tokens=1,history=3,attention_gate="sigmoid",qk_norm="rms").items()}
        # Competing placements/partition ratios receive identical numerical
        # inputs at the same geometry. Heldout seeds remain profile-disjoint.
        seed=1000+int(sha(geometry)[:6],16)
        program=export_plan(compiled,seed=seed)
        write_new(case/"program.json",program)
        rows.append({**item,"profile_seed":seed,"profile_program_sha256":program["program_sha256"]})
    campaign=dict(format_version="highlevel-accuracy-campaign-v1",suite=suite,cases=rows,
        created_at=utc_now(),coverage="actual_ModelPlan_compile_run_graph_and_actions",hardware_executed=False,
        profile_heldout_protocol="all_component_ledgers_frozen_before_any_heldout; fresh_processes; disjoint_numerical_seeds")
    campaign["campaign_sha256"]=sha(campaign);write_new(directory/"campaign.json",campaign)
    return dict(status="prepared",cases=len(rows),directory=str(directory),hardware_executed=False)


def load_case(directory,item):
    case=directory/item["id"];program=read(case/"program.json");validate_program(program)
    if program["program_sha256"]!=item["profile_program_sha256"]:raise BridgeError("prepared program changed")
    plan=build_case(case/"inputs",reuse=True,**item["kwargs"]).compile()
    if export_plan(plan,seed=item["profile_seed"])["program_sha256"]!=program["program_sha256"]:raise BridgeError("high-level source/compiler graph changed since prepare")
    return case,plan,program


def private_runtime_builds(runtime_dirs):
    """Bind an optional private backend build to all recorded current bytes."""
    private_builds={}
    for folder in runtime_dirs:
        record_file=Path(folder)/"strict-f32-runtime.json"
        if not record_file.exists():continue
        record=read(record_file)
        if record.get("format_version")!="ggml-private-strict-f32-runtime-v1" or record.get("patch_hunks")!=1:
            raise BridgeError("invalid private F32 runtime build record")
        for field in ("original_source_sha256","inputs_sha256","output_sha256","evidence_sha256"):
            if not record.get(field) or any(binary_sha(path)!=digest for path,digest in record[field].items()):
                raise BridgeError("private F32 runtime build input/output changed: "+field)
        if str((Path(folder)/"ggml-cuda.dll").resolve()) not in record["output_sha256"]:
            raise BridgeError("private F32 build record does not bind supplied CUDA DLL")
        private_builds[str(record_file.resolve())]=dict(sha256=binary_sha(record_file),record=record)
    return private_builds


def software(executable,runtime_dirs):
    from tensor_mapping.unified_strategy import runtime_fingerprint
    files={Path(executable).resolve()}
    for folder in runtime_dirs:
        files.update(p for p in Path(folder).iterdir() if p.is_file() and (p.suffix.lower() in {".dll",".so",".dylib"} or ".so." in p.name))
    native_source=ROOT/"profiling/native/highlevel_accuracy"
    source_files=list(Path(__file__).parent.glob("*.py"))+list(native_source.glob("*.cpp"))+list(native_source.glob("*.py"))+[native_source/"CMakeLists.txt"]
    build_file=Path(executable).parent/"build-manifest.json";build=read(build_file)
    if build["binary_sha256"]!=binary_sha(executable) or any(binary_sha(path)!=digest for path,digest in build["source_sha256"].items()):raise BridgeError("native binary/build source linkage is stale; rebuild before profiling")
    return dict(binary_sha256=binary_sha(executable),runtime_sha256={str(p):binary_sha(p) for p in sorted(files)},
        source_sha256={str(p):binary_sha(p) for p in sorted(source_files)},runtime_search_dirs=[str(p) for p in runtime_dirs],
        successful_build_record=build,successful_build_record_sha256=binary_sha(build_file),
        simulator_runtime_sha256=runtime_fingerprint(),private_runtime_builds=private_runtime_builds(runtime_dirs),
        runtime_hash_scope="executable_and_all_shared_libraries_in_supplied_search_dirs; resolved_loader_paths_not_observed")


def native_run(executable,program_file,mode,warmup,samples,directory,runtime_dirs):
    directory.mkdir(parents=True,exist_ok=False)
    output=directory/"native.json"
    env=os.environ.copy();env["PATH"]=os.pathsep.join([str(p) for p in runtime_dirs]+[env.get("PATH","")])
    env.update({k:v for k,v in RUNTIME_FLAGS.items() if k!="matmul_precision"})
    if os.name!="nt":env["LD_LIBRARY_PATH"]=os.pathsep.join([str(p) for p in runtime_dirs]+[env.get("LD_LIBRARY_PATH","")])
    command=[str(executable),str(program_file),mode,str(warmup),str(samples),str(output)]
    started=utc_now()
    try:
        completed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
        record=dict(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,timed_out=False)
    except subprocess.TimeoutExpired as error:
        def decoded(value):return value.decode("utf-8",errors="replace") if isinstance(value,bytes) else value or ""
        record=dict(returncode=None,stdout=decoded(error.stdout),stderr=decoded(error.stderr),timed_out=True)
    except OSError as error:
        record=dict(returncode=None,stdout="",stderr=str(error),timed_out=False,launch_failed=True,error_type=type(error).__name__)
    record.update(command=command,started_at=started,completed_at=utc_now(),pid_scope="one_fresh_native_process",requested_runtime_flags=RUNTIME_FLAGS)
    write_new(directory/"execution.json",record)
    if record["timed_out"] or record["returncode"]!=0:raise BridgeError("native execution failed; raw evidence preserved: "+str(directory/"execution.json")+" "+record["stderr"][-3000:])
    native=read(output)
    evidence=dict(raw_native_file=str(output),raw_native_sha256=binary_sha(output),execution_file=str(directory/"execution.json"),execution_sha256=binary_sha(directory/"execution.json"))
    return native,evidence,started


def profile_all(directory,campaign,executable,runtime_dirs,processes,warmup,samples):
    software_record=software(executable,runtime_dirs);ledgers=[];failures=[];case_records=[]
    if (directory/"all_profiles_frozen.json").exists():raise BridgeError("all profiles already frozen")
    with hardware_lock():
        for item in campaign["cases"]:
            try:
                case,plan,program=load_case(directory,item);merged=None;evidence=[];process_rows=[];process_failures=[]
                for process in range(processes):
                    try:
                        if software(executable,runtime_dirs)!=software_record:raise BridgeError("software changed during profile collection")
                        native,raw,_=native_run(executable,case/"program.json","components",warmup,samples,case/f"profile-process-{process}",runtime_dirs)
                        verify_native(program,native);evidence.append(raw)
                        process_rows.append(dict(process_index=process,components=deepcopy(native["components"]),raw_evidence=raw))
                        if merged is None:merged=deepcopy(native)
                        else:
                            if [r["action"] for r in merged["components"]]!=[r["action"] for r in native["components"]]:raise BridgeError("profile process action inventory changed")
                            for left,right in zip(merged["components"],native["components"]):left["samples_ns"].extend(right["samples_ns"])
                            merged["samples"]+=native["samples"]
                    except Exception as error:
                        process_failures.append(dict(process_index=process,status="failed",error_type=type(error).__name__,error=str(error)))
                if process_failures:raise BridgeError("one or more source processes failed: "+json.dumps(process_failures))
                merged["independent_processes"]=process_rows
                ledger=freeze_profile(program,merged,executable_sha256=software_record["binary_sha256"],evidence=dict(processes=evidence,software=software_record,independent_process_count=processes))
                prediction=run_plan_with_profile(plan,program,ledger)
                write_new(case/"frozen_profile.json",ledger);write_new(case/"modelplan_measured_prediction.json",prediction)
                ledgers.append(dict(case_id=item["id"],ledger_sha256=ledger["ledger_sha256"]))
                case_records.append(dict(case_id=item["id"],status="frozen",source_quality_passed=ledger["source_quality_passed"],ledger_sha256=ledger["ledger_sha256"]))
            except Exception as error:
                failure=dict(case_id=item["id"],status="failed",error_type=type(error).__name__,error=str(error),coverage="unknown_failed_not_passing")
                failures.append(failure);case_records.append(failure)
        try:
            if software(executable,runtime_dirs)!=software_record:failures.append(dict(status="failed",error="software changed before profile phase completed"))
        except Exception as error:failures.append(dict(status="failed",error="profile final software inventory: "+str(error)))
        report=dict(format_version="highlevel-profile-phase-status-v1",campaign_sha256=campaign["campaign_sha256"],
            cases=case_records,failures=failures,all_cases_frozen=not failures,
            source_quality_passed=not failures and all(row["source_quality_passed"] for row in case_records),fixed_quality_gates=GATES)
        write_new(directory/"profile_phase_status.json",report)
        if not failures:
            barrier=dict(format_version="highlevel-all-profiles-frozen-v1",campaign_sha256=campaign["campaign_sha256"],
                frozen_at=utc_now(),ledgers=ledgers,software=software_record,profile_processes_per_case=processes,samples_per_process=samples,
                source_quality_passed=report["source_quality_passed"],fixed_quality_gates=GATES)
            barrier["barrier_sha256"]=sha(barrier);write_new(directory/"all_profiles_frozen.json",barrier)
    return dict(status="profile_failures_preserved" if failures else "all_profiles_frozen",cases=len(ledgers),failures=len(failures),
                source_quality_passed=report["source_quality_passed"],report=str(directory/"profile_phase_status.json"))


def heldout_all(directory,campaign,executable,runtime_dirs,processes,warmup,samples,tolerance):
    barrier=read(directory/"all_profiles_frozen.json");unsigned=deepcopy(barrier);digest=unsigned.pop("barrier_sha256",None)
    barrier_file_sha=binary_sha(directory/"all_profiles_frozen.json")
    if digest!=sha(unsigned) or barrier["campaign_sha256"]!=campaign["campaign_sha256"]:raise BridgeError("all-profile freeze barrier changed")
    current_software=software(executable,runtime_dirs)
    if current_software!=barrier["software"]:raise BridgeError("native/software source changed after profile freeze")
    # Validate EVERY ledger before the first heldout process starts.
    if len(campaign["cases"])!=len(barrier["ledgers"]) or [item["id"] for item in campaign["cases"]]!=[row["case_id"] for row in barrier["ledgers"]]:raise BridgeError("incomplete or reordered global profile barrier")
    for item,row in zip(campaign["cases"],barrier["ledgers"]):
        case,_,program=load_case(directory,item);ledger=read(case/"frozen_profile.json");validate_ledger(program,ledger)
        if row!={"case_id":item["id"],"ledger_sha256":ledger["ledger_sha256"]}:raise BridgeError("frozen ledger inventory changed")
    results=[];failures=[];groups=[]
    with hardware_lock():
        for index,item in enumerate(campaign["cases"]):
            local=[]
            try:
                case,plan,_=load_case(directory,item);ledger=read(case/"frozen_profile.json");ledger_file_sha=binary_sha(case/"frozen_profile.json")
                for process in range(processes):
                    try:
                        if software(executable,runtime_dirs)!=current_software:raise BridgeError("software changed during heldout collection")
                        if binary_sha(directory/"all_profiles_frozen.json")!=barrier_file_sha or binary_sha(case/"frozen_profile.json")!=ledger_file_sha:raise BridgeError("profile freeze evidence changed during heldout")
                        program=export_plan(plan,seed=100_000_000+item["profile_seed"]*10+process)
                        program_file=case/f"heldout-program-{process}.json";write_new(program_file,program)
                        native,raw,started=native_run(executable,program_file,"replay",warmup,samples,case/f"heldout-process-{process}",runtime_dirs)
                        result=compare_heldout(program,ledger,native,heldout_started_at=started,executable_sha256=current_software["binary_sha256"],max_relative_error=tolerance)
                        result.update(case_id=item["id"],process_index=process,raw_evidence=raw,
                            actual_modelplan_prediction=run_plan_with_profile(plan,program,ledger))
                        write_new(case/f"comparison-{process}.json",result);results.append(result);local.append(result)
                    except Exception as error:
                        failure=dict(case_id=item["id"],process_index=process,status="failed",error_type=type(error).__name__,error=str(error),coverage="unknown_failed_not_passing")
                        failures.append(failure);write_new(case/f"failure-{process}.json",failure)
                if len(local)==processes:
                    medians=[row["observed_whole_program_median_ns"] for row in local]
                    pooled=statistics.median([sample for row in local for sample in row["observed_samples_ns"]])
                    prediction=local[0]["baseline"]["makespan_ns"];error=abs(prediction-pooled)/pooled
                    within=all(row["heldout_within_process_cv_passed"] for row in local)
                    between=cv(medians)<=GATES["between_process_median_cv_max"]
                    accepted=ledger["source_quality_passed"] and within and between and error<=GATES["pooled_median_absolute_relative_error_max"]
                    groups.append(dict(case_id=item["id"],status="complete",independent_processes=len(local),source_quality_passed=ledger["source_quality_passed"],
                        heldout_within_process_cv_passed=within,heldout_process_medians_ns=medians,heldout_between_process_median_cv=cv(medians),
                        heldout_between_process_median_cv_passed=between,pooled_heldout_median_ns=pooled,predicted_ns=prediction,
                        pooled_median_absolute_relative_error=error,fixed_quality_gates=GATES,accepted=accepted))
                else:groups.append(dict(case_id=item["id"],status="incomplete",accepted=False,coverage="unknown_failed_not_passing"))
            except Exception as error:
                failures.append(dict(case_id=item["id"],status="failed",error_type=type(error).__name__,error=str(error)))
                groups.append(dict(case_id=item["id"],status="failed",accepted=False,coverage="unknown_failed_not_passing"))
        try:
            if software(executable,runtime_dirs)!=current_software:failures.append(dict(status="failed",error="software changed before heldout phase completed"))
        except Exception as error:failures.append(dict(status="failed",error="heldout final software inventory: "+str(error)))
        report=dict(format_version="highlevel-accuracy-campaign-report-v1",campaign_sha256=campaign["campaign_sha256"],
            barrier_sha256=barrier["barrier_sha256"],all_profiles_frozen_before_any_heldout=True,
            independent_heldout_processes_per_case=processes,results=results,case_groups=groups,failures=failures,
            fixed_quality_gates=GATES,all_cases_accepted=not failures and all(row["accepted"] for row in groups),
            max_absolute_relative_error=max((row["absolute_relative_error"] for row in results),default=None),
            within_explicit_tolerance=None if tolerance is None else not failures and all(row["within_explicit_tolerance"] for row in results))
        write_new(directory/"report.json",report)
    return dict(status="heldout_complete",cases=len(campaign["cases"]),process_comparisons=len(results),
        report=str(directory/"report.json"),max_absolute_relative_error=report["max_absolute_relative_error"],within_explicit_tolerance=report["within_explicit_tolerance"],
        failures=len(failures),all_cases_accepted=report["all_cases_accepted"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","profile-all","heldout-all"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--suite",choices=("smoke","representative"),default="representative")
    parser.add_argument("--executable",type=Path)
    parser.add_argument("--runtime-dir",type=Path,action="append",default=[])
    parser.add_argument("--processes",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--samples",type=int,default=31)
    parser.add_argument("--max-relative-error",type=float)
    args=parser.parse_args(argv)
    try:
        directory=args.out.resolve()
        if args.phase=="prepare":result=prepare(directory,args.suite)
        else:
            if not args.executable or not args.executable.is_file():raise BridgeError("hardware phase requires an existing --executable")
            if args.processes<3 or args.samples<3 or args.warmup<0:raise BridgeError("hardware protocol requires >=3 independent processes, >=3 samples and nonnegative warmup")
            campaign=read(directory/"campaign.json");unsigned=deepcopy(campaign);digest=unsigned.pop("campaign_sha256",None)
            if digest!=sha(unsigned):raise BridgeError("campaign definition changed")
            executable=args.executable.resolve();runtime_dirs=[p.resolve() for p in args.runtime_dir]
            result=(profile_all(directory,campaign,executable,runtime_dirs,args.processes,args.warmup,args.samples) if args.phase=="profile-all" else
                    heldout_all(directory,campaign,executable,runtime_dirs,args.processes,args.warmup,args.samples,args.max_relative_error))
        print(json.dumps(result,allow_nan=False))
        return 3 if result.get("failures",0) or result.get("all_cases_accepted") is False or result.get("source_quality_passed") is False else 0
    except (BridgeError,OSError,ValueError,subprocess.SubprocessError) as error:
        if args.phase!="prepare" and args.out.is_dir():
            failure_file=args.out/(args.phase+"_phase_failure.json")
            if not failure_file.exists():write_new(failure_file,dict(status="failed",coverage="unknown_failed_not_passing",error_type=type(error).__name__,error=str(error),recorded_at=utc_now()))
        print("highlevel_accuracy: "+str(error),file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
