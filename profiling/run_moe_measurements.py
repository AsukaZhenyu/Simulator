"""Measure the declared resident F32 MoE FFN variant on explicit CPU runtime.

Inputs are an external token routing trace and random reproducible tensors, not
original model weights. Every case yields isolated stages and a joint graph.
Timing stability and prediction accuracy are separate from numeric correctness.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from moe_contract import (CONTRACT, MEASUREMENT_SCHEMA, STABILITY_CV_LIMIT,
    build_plan, contract_for, contract_hash, physical_signature, sampling_mode, statistics_ns, validate_native)


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_command(binary, case, *, warmup, samples, seed):
    c=case["config"]
    routes=";".join(",".join(str(expert) for expert in route) for route in case["routes"])
    args=[str(binary)]
    for key,value in (("hidden",c["hidden"]),("intermediate",c["intermediate"]),("experts",c["experts"]),
                      ("top-k",c["top_k"]),("tokens",c["tokens"]),("threads",case["threads"]),
                      ("warmup",warmup),("samples",samples),("seed",seed),("routes",routes),("backend",case["backend"]),("sampling-mode",sampling_mode(case))):
        args.extend(["--"+key,str(value)])
    return args


def validate_result(native, case, *, warmup, samples, seed):
    validate_native(native,case)
    for key,value in (("warmup",warmup),("samples",samples),("seed",seed)):
        if native[key]!=value:
            raise ValueError(f"native {key} disagrees with runner arguments")
    return native


def summarize_native(native):
    summary={stage:statistics_ns(value["samples_ns"]) for stage,value in native["stages"].items()}
    flags=["high_cv:"+stage for stage,stats in summary.items() if stats["cv"]>STABILITY_CV_LIMIT]
    return summary,flags


def run_suite(args):
    suite_bytes=args.suite.read_bytes()
    cases=build_plan(json.loads(suite_bytes))
    if args.warmup<0 or args.samples<3 or args.samples>10000 or args.seed<0 or args.seed>1000000:
        raise ValueError("explicit warmup>=0, samples in 3..10000 and seed in 0..1000000 required")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be absent or empty; evidence is never overwritten")
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/"plan.json",{"schema_version":"moe_measurement_plan_v1","contract":contract_for(cases[0]),"cases":cases,
        "commands":[native_command(args.native_bin,case,warmup=args.warmup,samples=args.samples,seed=args.seed) for case in cases],
        "hardware_execution":not args.dry_run})
    if args.dry_run:
        return {"dry_run":True,"case_count":len(cases),"output":str(args.output)}
    if not args.native_bin.is_file():
        raise ValueError("native MoE binary does not exist")
    runtime_files={}
    for directory in args.runtime_dir:
        if not directory.is_dir():
            raise ValueError("explicit runtime directory does not exist")
        for name in ("ggml-base.dll","ggml-cpu.dll","libggml-base.so","libggml-cpu.so","libggml-base.dylib","libggml-cpu.dylib"):
            path=directory/name
            if path.is_file():
                runtime_files[str(path.resolve())]=sha256(path)
    if not runtime_files:
        raise ValueError("no declared GGML base/CPU runtime libraries found in runtime directories")
    executable_hash=sha256(args.native_bin)
    env=os.environ.copy()
    paths=os.pathsep.join(str(path.resolve()) for path in args.runtime_dir)
    env["PATH"]=paths+os.pathsep+env.get("PATH","")
    if sys.platform!="win32":
        key="DYLD_LIBRARY_PATH" if sys.platform=="darwin" else "LD_LIBRARY_PATH"
        env[key]=paths+os.pathsep+env.get(key,"")
    raw=args.output/"raw"
    raw.mkdir()
    records=[]
    for index,case in enumerate(cases):
        command=native_command(args.native_bin,case,warmup=args.warmup,samples=args.samples,seed=args.seed)
        process=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="strict",env=env,
            timeout=args.timeout_seconds,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        prefix=f"{index:03d}"
        (raw/(prefix+".stderr.txt")).write_text(process.stderr,encoding="utf-8",newline="\n")
        if process.returncode:
            (raw/(prefix+".stdout.txt")).write_text(process.stdout,encoding="utf-8",newline="\n")
            raise ValueError(f"native case {case['id']} failed with exit {process.returncode}; see raw stderr")
        native=json.loads(process.stdout)
        validate_result(native,case,warmup=args.warmup,samples=args.samples,seed=args.seed)
        dump(raw/(prefix+".json"),native)
        summary,flags=summarize_native(native)
        signature=physical_signature(case,executable_hash)
        records.append({"case_id":case["id"],"case":case,"status":"measured_validation_only" if flags else "measured",
            "validation_only":bool(flags),"quality_flags":flags,"physical_signature":signature,
            "physical_contract_sha256":contract_hash(signature),"native":native,"summary":summary,
            "raw_output":"raw/"+prefix+".json","raw_sha256":sha256(raw/(prefix+".json"))})
        print(json.dumps({"case_id":case["id"],"status":records[-1]["status"],"active_experts":len(native["stages"])-3,
            "joint_median_ns":summary["joint"]["median_ns"],"quality_flags":flags}),flush=True)
    if sha256(args.native_bin)!=executable_hash or hashlib.sha256(args.suite.read_bytes()).hexdigest()!=hashlib.sha256(suite_bytes).hexdigest():
        raise ValueError("binary or suite changed during measurement")
    if any(sha256(path)!=digest for path,digest in runtime_files.items()):
        raise ValueError("GGML runtime library changed during measurement")
    report={"schema_version":MEASUREMENT_SCHEMA,"contract":contract_for(cases[0]),"records":records}
    dump(args.output/"measurements.json",report)
    dump(args.output/"manifest.json",{"schema_version":"moe_measurement_manifest_v1",
        "created_at":datetime.now(timezone.utc).isoformat(),"suite":str(args.suite.resolve()),
        "suite_sha256":hashlib.sha256(suite_bytes).hexdigest(),"native_binary":str(args.native_bin.resolve()),
        "native_binary_sha256":executable_hash,"runtime_libraries_sha256":runtime_files,
        "platform":platform.platform(),"machine":platform.machine(),"warmup":args.warmup,"samples":args.samples,"seed":args.seed,
        "stability_cv_limit":STABILITY_CV_LIMIT,"record_count":len(records),"CPU_frozen_assets_touched_by_runner":False,
        "scope":"resident_moe_ffn_subgraph","routing_mode":"prescribed_trace_replay",
        "original_model_weights":False,"original_model_kernel_reproduced":False,
        "attention_or_KV_measured":False,"GPU_measured":False,
        "numeric_validation":"independent scalar reference plus same-materialized-port digests",
        "timing_validation":"joint measurements included; held-out prediction verification is a separate step"})
    return {"case_count":len(records),"validation_only_count":sum(record["validation_only"] for record in records),"output":str(args.output)}


def load_measurements(directory, *, verify_live_sources=False):
    """Load portable evidence and reject changed raw ports, records, or summaries.

    Live source checks are optional because an evidence directory can be copied
    to another host. Raw JSON and binary/runtime hashes remain bound in artifacts.
    """
    directory=Path(directory).resolve()
    report=json.loads((directory/"measurements.json").read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if report.get("schema_version")!=MEASUREMENT_SCHEMA or report.get("contract") not in (CONTRACT,"ggml_cpu_resident_moe_ffn_f32_v2"):
        raise ValueError("unsupported MoE measurement report")
    records=report.get("records")
    if not isinstance(records,list) or not records or manifest.get("record_count")!=len(records):
        raise ValueError("measurement record coverage mismatch")
    if manifest.get("schema_version")!="moe_measurement_manifest_v1":
        raise ValueError("measurement manifest schema mismatch")
    seen=set()
    for record in records:
        if not isinstance(record,dict) or record.get("case_id") in seen:
            raise ValueError("duplicate or invalid measurement record")
        seen.add(record["case_id"])
        raw_path=(directory/record["raw_output"]).resolve()
        if not raw_path.is_relative_to(directory) or sha256(raw_path)!=record.get("raw_sha256"):
            raise ValueError("raw measurement source changed or path escapes evidence directory")
        native=json.loads(raw_path.read_text(encoding="utf-8"))
        if contract_hash(native)!=contract_hash(record.get("native")):
            raise ValueError("embedded native measurement changed")
        case=record["case"]
        if report["contract"]!=contract_for(case):
            raise ValueError("report mixes incompatible timing contracts")
        if record["case_id"]!=case["id"]:
            raise ValueError("record id does not match declared case")
        validate_result(native,case,warmup=manifest["warmup"],samples=manifest["samples"],seed=manifest["seed"])
        signature=physical_signature(case,manifest["native_binary_sha256"])
        if (contract_hash(signature)!=contract_hash(record.get("physical_signature")) or
            contract_hash(record["physical_signature"])!=record.get("physical_contract_sha256")):
            raise ValueError("physical contract source changed")
        summary,flags=summarize_native(native)
        if contract_hash(summary)!=contract_hash(record.get("summary")) or flags!=record.get("quality_flags") or record.get("validation_only") is not bool(flags):
            raise ValueError("measurement quality or statistics changed")
        if record.get("status")!=("measured_validation_only" if flags else "measured"):
            raise ValueError("measurement status does not match stability evidence")
    if verify_live_sources:
        for path,digest in {manifest["suite"]:manifest["suite_sha256"],manifest["native_binary"]:manifest["native_binary_sha256"],
                           **manifest["runtime_libraries_sha256"]}.items():
            if sha256(path)!=digest:
                raise ValueError("live measurement source changed")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--native-bin",type=Path,required=True)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--warmup",type=int,required=True)
    parser.add_argument("--samples",type=int,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=120)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    if args.timeout_seconds<1:
        parser.error("timeout-seconds must be positive")
    try:
        print(json.dumps(run_suite(args)),flush=True)
        return 0
    except (ValueError,OSError,json.JSONDecodeError,subprocess.SubprocessError) as error:
        print(str(error),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
