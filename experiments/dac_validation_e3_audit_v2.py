"""Read-only independent denominator, physical-ledger and capacity audit for E3.

This module imports no E3 runner, compiler, tensor engine or performance model.
It reuses an independently authored standard-library event auditor, then checks
all rejected capacity admissions from retained data rather than trusting status.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.hardware_system_audit import audit_trace


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def fraction(value):return Fraction(value["numerator"],value["denominator"]) if isinstance(value,dict) else Fraction(str(value))


def expected_points():
    keys=set()
    for architecture,axes in (("separate",("cpu","gpu","link","capacity")),("shared",("cpu","gpu","capacity"))):
        keys.add(architecture+"-base")
        for axis in axes:
            for level in (0.5,0.8,1.2,1.5):keys.add(f"{architecture}-{axis}-{level:g}")
        other="link" if architecture=="separate" else "capacity"
        for a,b in itertools.product((0.8,1.2),repeat=2):keys.add(f"{architecture}-gpu-{a:g}-{other}-{b:g}")
    require(len(keys)==38,"independent point definition failed")
    return keys


def hardware(document):
    raw=document["architecture"]
    return dict(raw,memory_mode=raw.get("memory_mode","separate"))


def capacity_rejection(evidence,row):
    hw=hardware(evidence["scenario"])
    shared=hw["memory_mode"]=="shared"
    pool=lambda loc:"shared" if shared else loc
    budgets={"shared":hw["shared_capacity_bytes"]} if shared else {
        "dram":hw["dram_capacity_bytes"],"vram":hw["vram_capacity_bytes"]}
    workload=evidence["workload"]
    tensors={t["id"]:t for t in workload["tensors"]}
    if not evidence["events"] and not evidence["tensor_catalogue"]:
        current=Counter()
        for t in tensors.values():
            locations=set(t.get("initial_locations",[]))
            if shared:
                if locations & {"dram","vram"}:current["shared"]+=t["storage_bytes"]
            else:
                for loc in locations & {"dram","vram"}:current[loc]+=t["storage_bytes"]
        require(any(current[p]>n for p,n in budgets.items()),"initial capacity failure without exceeded initial inventory")
        return dict(initial=True,current=dict(current),budgets=budgets)
    allocations={}
    for t in evidence["tensor_catalogue"]:
        sizes=dict(t["allocation_bytes"])
        for loc,identity in t["allocation_refs"]:
            if identity is not None:
                key=(pool(loc),identity)
                value=(t["id"],sizes[loc])
                require(key not in allocations or allocations[key]==value,"capacity replay allocation alias conflict")
                allocations[key]=value
    current=Counter()
    for (p,_),(_,amount) in allocations.items():current[p]+=amount
    # Independent pools use logical READY/RESERVED copies and deliberately have
    # no shared-allocation identities. Count each present separate copy once.
    if not shared:
        require(not allocations, "separate pools unexpectedly expose shared allocation identities")
        for t in evidence["tensor_catalogue"]:
            sizes=dict(t["allocation_bytes"])
            for loc,status in t["copies"]:
                if loc in budgets and status != "ABSENT":
                    current[loc]+=sizes[loc]
    action=row["rejected_action"]
    require(action is not None,"capacity status lacks rejected admission")
    if action["kind"]=="COPY":
        destination=pool(action["dst"])
        need=tensors[action["target_id"]]["storage_bytes"]
    elif action["kind"]=="COMPUTE":
        loc="dram" if action["device"]=="cpu" else "vram"
        destination=pool(loc)
        op=next(op for op in workload["operations"] if op["id"]==action["target_id"])
        need=sum(tensors[tid]["storage_bytes"] for tid in [op["output"],*op.get("extra_outputs",[])])
        price=evidence["scenario"]["costs"]["compute"][action["target_id"]][action["device"]]
        need+=price["workspace_bytes"]
    else:
        raise ValueError("capacity rejection is neither fresh COPY nor COMPUTE admission")
    require(current[destination]+need>budgets[destination],"rejected capacity admission independently fits")
    return dict(initial=False,pool=destination,current=current[destination],needed=need,budget=budgets[destination])


def price_override_checks(evidence,row,program,models):
    """Check declared changes independently, including fixed COPY residuals."""
    ledger=evidence["response_ledger"]
    require(len(ledger)==len(program["actions"]),"action ledger coverage differs")
    tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
    point=row["point"]
    copy_wait=False
    for index,(price,action) in enumerate(zip(ledger,program["actions"])):
        require(price["index"]==index and price["native_kind"]==action["kind"],"ledger action binding differs")
        original=fraction(price["original_ns"])
        expected=original
        if point["architecture"]=="shared" and action["kind"]=="ADVANCE" and copy_wait:
            expected=Fraction()
            require(price["modeled_kind"]=="OMITTED_COPY_WAIT","shared copy wait retained")
            copy_wait=False
        elif action["kind"]=="COMPUTE":
            expected=original*fraction(point[action["device"]])
            copy_wait=False
        elif action["kind"]=="COPY":
            if point["architecture"]=="shared":
                expected=Fraction(point["handoff_ns"])
                require(price["modeled_kind"]=="SHARE","shared recipe still copies")
                copy_wait=True
            elif point["link"]!=1:
                route=action["src"]+"->"+action["dst"]
                model=models.get(route,models.get(route.replace("->","_to_")))
                require(model is not None and model.get("qualified"),"unqualified payload decomposition used")
                slope=model.get("payload_ns_per_byte",model.get("slope_ns_per_byte"))
                require(slope is not None,"missing payload slope")
                payload=fraction(slope)*tensors[action["target_id"]]["storage_bytes"]
                fixed=original-payload
                require(fixed>=0 and payload>=0,"negative physical COPY term")
                expected=fixed+payload/fraction(point["link"])
                copy_wait=False
            else:copy_wait=False
        elif action["kind"]!="EVICT":copy_wait=False
        require(fraction(price["scenario_ns"])==expected,"undeclared action/fixed/payload price change")
    return len(ledger)


def audit(directory):
    directory=Path(directory).resolve()
    manifest=read(directory/"execution-manifest.json")
    summary=read(directory/"summary.json")
    require(sha(directory/"inventory.json")==manifest["inventory_sha256"],"prepared inventory changed")
    for relative,digest in manifest["source_sha256"].items():
        require(sha(ROOT/relative)==digest,"executed source changed: "+relative)
    for receipt in [manifest["frozen_profile"],*manifest["anchor_programs"].values()]:
        require(sha(receipt["path"])==receipt["sha256"],"anchor bytes changed")
    receipt=manifest["frozen_profile"].get("program_manifest")
    if receipt:require(sha(receipt["path"])==receipt["sha256"],"frozen E2A manifest changed")
    receipt=manifest["frozen_profile"]["predictions"]
    require(sha(receipt["path"])==receipt["sha256"],"frozen E2A predictions changed")
    predictions={(r["workload"],r["strategy"]):r for r in read(receipt["path"])}
    cross=read(directory/"anchor-cross-check.json")
    require(cross==manifest["frozen_profile"]["cross_prediction_check"] and cross["passed"]
            and cross["checked"]==cross["required"]==8,"anchor cross-check receipt differs")
    frozen=read(manifest["frozen_profile"]["path"])
    transfer_receipt=manifest.get("transfer_models")
    if transfer_receipt and transfer_receipt.get("path"):
        require(sha(transfer_receipt["path"])==transfer_receipt["sha256"],"transfer model bytes changed")
        transfer_models=read(transfer_receipt["path"])
        if "models" in transfer_models:transfer_models=transfer_models["models"]
    else:transfer_models=frozen.get("transfer_decomposition",{})
    programs={key:read(receipt["path"]) for key,receipt in manifest["anchor_programs"].items()}
    rows=read(directory/"main-records.json")
    expected={(w,p,c) for w in ("h512-p64-h0","h512-d1-h128") for p in expected_points() for c in ("S0","S1","S2","S3")}
    actual=[(r["workload"],r["point"]["id"],r["candidate"]) for r in rows]
    require(len(actual)==len(set(actual))==304 and set(actual)==expected,"304 main denominator differs")
    counts=Counter()
    receipts=[]
    for family in ("main","robustness"):
        path=directory/f"{family}-records.json"
        if not path.exists():continue
        records=read(path)
        if family=="robustness":
            require(len(records)==1152,"robustness denominator differs")
            keys=[(r["workload"],r["point"]["id"],r["candidate"]) for r in records]
            require(len(keys)==len(set(keys)),"duplicate robustness case")
            for w in ("h512-p64-h0","h512-d1-h128"):
                for architecture in ("separate","shared"):
                    directions=("baseline","cpu-faster","gpu-faster","more-capacity")+(("more-link-bandwidth",) if architecture=="separate" else ())
                    for corner,name,candidate in itertools.product(range(16),directions,("S0","S1","S2","S3")):
                        require((w,f"robust-{architecture}-c{corner:02}-{name}",candidate) in keys,"missing nominated direction/corner/candidate")
        for row in records:
            counts[family+"_cases"]+=1
            artifact=directory/row["artifact_directory"]/"evidence.json"
            if row["status"]=="unsupported_contract":
                require(row["completed_makespan_ns"] is None and not row["success"],"unsupported execution ranked as complete")
                counts["unsupported_contract"]+=1
                continue
            require(sha(artifact)==row["evidence_sha256"],"case artifact changed")
            evidence=read(artifact)
            counts["independently_checked_action_price_terms"]+=price_override_checks(
                evidence,row,programs[row["workload"]+"/"+row["candidate"]],transfer_models)
            if row["point"]["id"]=="separate-base":
                work={"h512-p64-h0":"h512-p64","h512-d1-h128":"h512-d1h128"}[row["workload"]]
                predicted=predictions[work,int(row["candidate"][1:])]["prediction"]
                require([fraction(r["original_ns"]) for r in evidence["response_ledger"]]
                    ==[fraction(value) for value in predicted["action_ns"]],"baseline action responses differ from frozen heldout prediction")
                require(sum((fraction(r["scenario_ns"]) for r in evidence["response_ledger"]),Fraction())
                    ==fraction(predicted["duration_ns"]),"baseline exact sum differs from frozen heldout prediction")
                counts["baseline_heldout_prediction_exact_crosschecks"]+=1
            if row["success"]:
                semantic=dict(row,completed_makespan_ns=row["semantic_engine_makespan_ns"])
                result=audit_trace(evidence["events"],evidence["operation_catalogue"],evidence["tensor_catalogue"],semantic,hardware(evidence["scenario"]))
                counts["successful_physical_replays"]+=1
                counts["events"]+=result["events"]
                require(abs(float(sum((fraction(r["scenario_ns"]) for r in evidence["response_ledger"]),Fraction()))-row["completed_makespan_ns"])<1e-6,"response accounting mismatch")
                if row["point"]["architecture"]=="shared":
                    require(row["h2d_bytes"]==row["d2h_bytes"]==0,"idealized shared view moves payload")
                require(row["peak_total_physical_bytes"]>=0,"negative physical peak")
            else:
                require(row["completed_makespan_ns"] is None,"failed partial duration ranked")
                require(row["status"]=="capacity_infeasible","unresolved noncapacity failure")
                receipt=capacity_rejection(evidence,row)
                counts["independently_confirmed_capacity_rejections"]+=1
                receipts.append(dict(case=row["artifact_directory"],**receipt))
    require(counts["main_cases"]==304,"main cases lost")
    require(counts["baseline_heldout_prediction_exact_crosschecks"]==8,"8 anchor cross-checks incomplete")
    return dict(schema="dac-validation-e3-independent-audit-2",passed=True,counts=dict(counts),
                capacity_receipts=receipts,source_files_checked=len(manifest["source_sha256"]),
                accuracy_qualification="not checked by this physical/scenario audit",summary_sha256=sha(directory/"summary.json"),
                auditor_sha256=sha(Path(__file__)),
                audit_revision="v2 corrects separate-pool accounting: present copies have no shared allocation refs; execution outputs unchanged")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    receipt=audit(args.directory)
    path=args.directory/"independent-audit.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="capacity_receipts"}))


if __name__=="__main__":main()
