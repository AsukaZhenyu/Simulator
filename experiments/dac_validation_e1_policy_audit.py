"""Independent post-run audit of declared device/storage policy witnesses.

This intentionally reads frozen declarations/low-level contracts and emitted
events without importing production code or asking a compiler for expectations.
"""
import hashlib
import json
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,label):
    if not x:raise AssertionError(label)


def audit(folder):
    scenario=read(folder/"declaration.json");expected=read(folder/"handwritten-low-level-dag.json")
    events=read(folder/"events.json");computes=[e for e in events if e["kind"]=="COMPUTE"]
    p=scenario["pattern"];checks=[]
    for event in computes:
        identity=event["target_id"];op=expected["operations"][identity];kind=op["semantic_op"]
        c=expected["contexts"][identity]
        if kind=="state_update":device="cpu"
        elif kind.startswith("ffn_"):
            device="cpu" if ".host." in identity else "gpu"
        elif kind in {"attention_qk","attention_softmax"}:device="cpu"
        elif kind in {"attention_pv","attention_output"}:device="gpu"
        elif kind=="expert_ffn":device="gpu"  # These primary scenarios have CPU 120000ns / GPU60000ns; all declared boundary copies are <=3072ns.
        elif scenario["handoff"]:device="gpu" if c["step"]==0 else "cpu"
        elif kind=="attention":device="cpu"
        elif p=="A01" and kind=="projection":device="cpu"
        else:device="gpu"
        need(event["device"]==device,f"{identity}: expected {device}, got {event['device']}")
        checks.append(dict(operation=identity,kind=kind,expected_device=device))
    copies=[e for e in events if e["kind"]=="COPY"]
    weight_copies={}
    if scenario["cache"] and p!="A12":
        gpu_read_weights={ti for event in computes if event["device"]=="gpu" for ti in expected["operations"][event["target_id"]]["inputs"] if ti.startswith("w:")}
        for ti in sorted(gpu_read_weights):
            matching=[e for e in copies if e["target_id"]==ti and e["src"]=="dram" and e["dst"]=="vram"]
            need(len(matching)==1,f"cache residency reused weight {ti} must load exactly once")
            weight_copies[ti]=len(matching)
    handoffs=[]
    if scenario["handoff"]:
        for r in range(scenario["requests"]):
            gate=f"g:request{r}-request{r}-decode0:trace_next:request{r}"
            gate_event=next(e for e in computes if e["target_id"]==gate)
            for l in range(scenario["layers"]):
                for position in range(scenario["q"]):
                    ti=f"kv:layer{l}:request{r}:{position}"
                    matched=[e for e in copies if e["target_id"]==ti and e["src"]=="vram" and e["dst"]=="dram"]
                    need(len(matched)==1,f"one charged prefill KV handoff required: {ti}")
                    need(matched[0]["end_ns"]<=gate_event["start_ns"],"decode gate precedes KV handoff")
                    handoffs.append(ti)
    if p=="A12":
        need(not copies,"shared parent-view pattern must not perform payload COPY")
        shares=[e for e in events if e["kind"]=="SHARE"]
        need(shares,"shared views not exercised")
        need(all(e["bytes_transferred"]==0 and e["start_ns"]==e["end_ns"] for e in shares),"shared view is not zero-payload visibility")
    return dict(id=scenario["id"],status="pass",device_checks=len(checks),weight_load_once_checks=len(weight_copies),
        handoff_tensor_checks=len(handoffs),devices=checks,weight_copy_counts=weight_copies,handoff_tensors=handoffs,
        source_hashes={name:sha(folder/name) for name in ("declaration.json","handwritten-low-level-dag.json","events.json")})


def main():
    root=Path(sys.argv[1]);rows=[]
    for folder in sorted((root/"positive").iterdir()):
        if not folder.is_dir():continue
        try:row=audit(folder)
        except Exception as error:row=dict(id=folder.name,status="fail",error=repr(error))
        rows.append(row)
    result=dict(schema="e1-independent-policy-witness-audit-v1",passed=sum(r["status"]=="pass" for r in rows),
        denominator=48,script_sha256=sha(__file__),main_manifest_sha256=sha(root/"manifest.json"),cases=rows,
        limitations="Additional policy witness checks; dynamic expert choice expectation is scoped to this primary suite's fixed synthetic prices, not all hardware regimes.")
    output=root.parent/(root.name+"-policy-witness-audit.json")
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"}));print([(r["id"],r.get("error")) for r in rows if r["status"]!="pass"])
    return 0 if result["passed"]==48 else 1


if __name__=="__main__":raise SystemExit(main())
