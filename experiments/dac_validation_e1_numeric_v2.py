"""Revalidate frozen E1 actions against the exact declared physical contract.

V1 used epsilon=1e-5 in both experimental numerical paths, whereas the real
builder declares float32(1e-6). V1 structural/physical events remain valid. This
new immutable numerical analysis binds the same inputs/graphs/events, validates
the complete physical contract first and corrects both numerical paths without
changing tolerances, any production code or any recorded simulation trace.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"mapping",ROOT/"modeling"):sys.path.insert(0,str(p))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def need(ok,message):
    if not ok:raise AssertionError(message)
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def config_receipts(s,folder):
    receipts=[]
    for l in range(s["layers"]):
        path=folder/f"inputs/layer{l}.layer.json";doc=read(path);physical=doc.get("physical")
        if s["kind"]=="moe":
            need(physical is None and doc["format_version"]=="m2-layer-0.6","MoE must remain an explicit opaque synthetic boundary")
            receipts.append(dict(source=str(path),source_sha256=sha(path),physical=None,
                status="synthetic_contract_only",real_numerical_moe=False))
            continue
        expected_config=dict(hidden=s["hidden"],intermediate=s["intermediate"],heads=2,kv_heads=1,
            head_dim=s["head_dim"],rope_dims=s["head_dim"],rope_mode="normal",rope_base=10000.0,
            norm_epsilon=float(np.float32(1e-6)),qk_norm="none",attention_gate="none",
            weight_dtypes={k:"f32" for k in ("input_norm","q","k","v","output","post_attention_norm","gate","up","down")})
        expected=dict(contract="ggml_hybrid_dense_materialized_macros_v1",scope="complete_declared_decoder_block",
            input_boundary="hidden_activation",port_layout="materialized_stage_ports",kv_allocation="logical_blocks",
            activation_dtype="f32",kv_dtype="f32",config=expected_config,threads=1,
            weight_groups=dict(projection={k:"W_"+k for k in ("input_norm","q","k","v")},
                attention=dict(output="W_output"),dense_ffn={k:"W_"+k for k in ("post_attention_norm","gate","up","down")}),
            excluded_operations=[],cuda_device=0,cuda_graphs="disabled")
        need(doc["format_version"]=="m2-layer-0.5","Dense physical boundary version differs")
        need(physical==expected,"complete Dense physical config differs from fixed numerical contract: "+str(path))
        receipts.append(dict(source=str(path),source_sha256=sha(path),status="pass",actual=physical,expected=expected))
    return receipts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("new immutable numerical output directory required")
    out.mkdir(parents=True,exist_ok=True)
    original=ROOT/"experiments/dac_validation_e1_numeric.py"
    code=original.read_text(encoding="utf-8")
    need(code.count("1e-5")==3,"v1 numerical source epsilon sites changed; explicit review required")
    corrected=code.replace("1e-5",repr(float(np.float32(1e-6))))
    corrected_path=out/"code/numeric_contract_v2.py";corrected_path.parent.mkdir()
    corrected_path.write_text(corrected,encoding="utf-8",newline="\n")
    numeric_spec=importlib.util.spec_from_file_location("dac_e1_numerical_contract_v2",corrected_path)
    numeric=importlib.util.module_from_spec(numeric_spec);sys.modules[numeric_spec.name]=numeric;numeric_spec.loader.exec_module(numeric)
    cases=read(source/"manifest.json")["cases"]
    manifest=dict(schema="e1-numerical-v2-manifest",created_utc=datetime.now(timezone.utc).isoformat(),
        source_campaign=str(source),source_manifest_sha256=sha(source/"manifest.json"),cases=cases,
        original_numerical_source_sha256=sha(original),corrected_numerical_source_sha256=sha(corrected_path),
        runner_sha256=sha(__file__),atol=5e-5,rtol=5e-5,contract=numeric.NUMERICAL_CONTRACT,
        independent_scope="Logical request/layer reference with scalar dot/RMS/RoPE/attention/FFN versus compiled physical action interpreter with vectorized implementations. Both share only seeded initial data generation, ID helpers, NumPy arithmetic and port F32 casting conventions; no norm/RoPE/operator evaluation helper is shared.",
        supersession="V1 numerical epsilon mismatch is replaced by v2; retain v1 structural/physical/source receipts unchanged. No numerical-tolerance change and no production model change.")
    write(out/"manifest.json",manifest)
    rows=[]
    for s in cases:
        folder=source/"positive"/s["id"];target=out/s["id"]
        try:
            receipts=config_receipts(s,folder);write(target/"physical-config-receipts.json",receipts)
            expected=read(folder/"handwritten-low-level-dag.json");graph=read(folder/"execution-graph.json");events=read(folder/"events.json")
            evidence=numeric.check_numerical(s,expected,graph,events)
            write(target/"outputs.json",evidence.pop("outputs"))
            row=dict(id=s["id"],status="pass",numerical=evidence,
                source_hashes={name:sha(folder/name) for name in ("declaration.json","handwritten-low-level-dag.json","execution-graph.json","events.json")},
                physical_config_receipts=len(receipts),physical_contract="complete_declared_dense" if s["kind"]=="dense" else "explicit_synthetic_opaque_moe")
        except Exception as error:row=dict(id=s["id"],status="fail",error=repr(error),traceback=traceback.format_exc())
        write(target/"result.json",row);rows.append(row)
    summary=dict(schema="e1-numerical-v2-summary",manifest_sha256=sha(out/"manifest.json"),cases=rows,
        total=48,passed=sum(r["status"]=="pass" for r in rows),
        dense_passed=sum(r["status"]=="pass" and r["physical_contract"]=="complete_declared_dense" for r in rows),
        synthetic_moe_passed=sum(r["status"]=="pass" and r["physical_contract"]=="explicit_synthetic_opaque_moe" for r in rows),
        max_absolute_error=max((r.get("numerical",{}).get("max_absolute_error",0) for r in rows),default=0),
        checked_logical_ports=sum(r.get("numerical",{}).get("checked_logical_ports",0) for r in rows),
        checked_elements=sum(r.get("numerical",{}).get("checked_elements",0) for r in rows),
        independently_replayed_copies=sum(r.get("numerical",{}).get("physically_replayed_copy_values",0) for r in rows),
        complete=all(r["status"]=="pass" for r in rows),source_campaign=str(source),
        limitations=["CPU-only numerical action interpretation, not GGML/CUDA execution or performance accuracy.",
            "8 MoE cases use a declared synthetic opaque arithmetic contract; not production MoE numerical implementation.",
            "The production simulator does not execute tensor values; this experiment checks preservation by external numerical interpretation of its declared actions."])
    write(out/"summary.json",summary)
    write(out/"artifact-hashes.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2));print([(r["id"],r.get("error")) for r in rows if r["status"]!="pass"])
    return 0 if summary["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
