"""Run the frozen, finite 2026-10-24 E1 suite; no native timing is collected.

Public Python recipes are compiled and executed unchanged.  Hand-authored DAGs,
physical replay, logical numerical reference, and lowering audit are independent
experiment code. Synthetic MoE arithmetic is explicitly not a production kernel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "modeling", ROOT / "mapping"):
    sys.path.insert(0, str(folder))

from llm_simulator import (AnalyticalCosts, Cache, Device, Hardware, Model, Prefetch,
    PhaseHandoff, PhasePlacement, Request, RequestFlow, RequestPipeline, Route,
    Scheduling, Simulator)
from experiments import strategy_fidelity as ref
from experiments.dac_validation_e1_oracle import declared_dag, check_graph, replay_physical
from experiments.dac_validation_e1_numeric import check_numerical, NUMERICAL_CONTRACT
from experiments.dac_validation_e1_lowering import run_lowering


PATTERNS = [
    ("A01", "Whole-macro CPU/GPU placement"),
    ("A02", "Attention and FFN refinement"),
    ("A03", "Materialized FFN partition and merge"),
    ("A04", "Residency, demand loading and prefetch"),
    ("A05", "Two-request execution pipeline"),
    ("A06", "Causal MoE expert scheduling"),
    ("A07", "Refinement with caching"),
    ("A08", "Refinement with prefetch"),
    ("A09", "Request pipeline with cross-layer prefetch"),
    ("A10", "P/D handoff with retained KV"),
    ("A11", "MoE cache and migration"),
    ("A12", "Shared physical storage with parent views/partitions"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(pattern, hidden, q, *, future=1, causal_mode=None):
    layers = 1 if pattern in {"A01", "A02", "A03", "A12"} else 2
    decode = 2 if pattern in {"A04", "A06", "A07", "A10", "A11"} else 0
    requests = 2 if pattern in {"A05", "A09", "A12"} else 1
    moe = pattern in {"A06", "A11"}
    return dict(id=f"{pattern}-h{hidden}-q{q}", pattern=pattern, hidden=hidden,
        intermediate=2*hidden, heads=2, kv_heads=1, head_dim=hidden//2,
        layers=layers, q=q, decode_steps=decode, requests=requests,
        memory_mode="shared" if pattern=="A12" else "separate",
        kind="moe" if moe else "dense", future=future, causal_mode=causal_mode,
        attention_refined=pattern in {"A02", "A07", "A08"},
        ffn_refined=pattern in {"A02", "A03", "A07", "A08", "A12"},
        prefetch=pattern in {"A04", "A08", "A09", "A11"} or causal_mode=="prefetch",
        cache=pattern in {"A04", "A06", "A07", "A08", "A09", "A10", "A11", "A12"},
        handoff=pattern in {"A10", "A11"},
        seed=20261024, atol=5e-5, rtol=5e-5,
        cache_budget_bytes=512*hidden*hidden,
        physical_capacity_bytes=2048*hidden*hidden)


def routes_for(s):
    if s["kind"] != "moe":
        return ()
    return tuple(Route(f"request{r}", f"layer{l}", p,
        ((p % 2) if p < s["q"] else s["future"],))
        for r in range(s["requests"]) for l in range(s["layers"])
        for p in range(s["q"]+s["decode_steps"]))


class WitnessCosts(AnalyticalCosts):
    """Explicit large compute windows expose overlap, without hardware fitting."""
    def __init__(self,regime="compute"):
        super().__init__()
        object.__setattr__(self,"regime",regime)

    def _cost_document(self, *args, **kwargs):
        doc, details = super()._cost_document(*args, **kwargs)
        for row in [*doc["entries"], *details]:
            row["duration_ns"] = {"projection":30000, "attention":50000,
                "dense_ffn":60000, "router":9000, "expert_ffn":60000,
                "combine":12000}[row["kind"]]
            if row["device"]=="cpu" and row["kind"] in {"dense_ffn", "expert_ffn"}:
                row["duration_ns"] *= 2
            if self.regime=="compute_constrained":row["duration_ns"]*=10
        doc["source"] += "; E1 exact hypothetical witness prices, never measured"
        return doc, details


def transfer_rows(s):
    h, i, d, q = s["hidden"], s["intermediate"], s["head_dim"], s["q"]
    sizes={4,4*h,8*d,4*q,4*h*h,4*h*d,4*h*i,4*h*(i//4),4*h*(3*i//4)}
    for n in {1,q}:
        for width in {h,i,i//4,3*i//4,d}:
            sizes.add(4*n*width)
        for past in range(q+3):
            sizes.add(4*2*n*(past+n))
    return [dict(route=route, storage_bytes=size, duration_ns=size*(1000 if s.get("regime")=="transfer_constrained" else 1))
        for route in ("dram_to_vram","vram_to_dram") for size in sorted(sizes)]


def attention_spec(s):
    return dict(format_version="attention-stages-0.1", variant="qk_softmax_pv",
        provenance="synthetic_execution", geometry=dict(hidden=s["hidden"],heads=2,
            kv_heads=1,head_dim=s["head_dim"],attention_gate="none"),
        frontend=dict(contract="existing_materialized_post_rope_qkv"),
        entries=[dict(kind=kind,device=device,phase=phase,batch=1,tokens=n,
            visible_history=past,duration_ns=duration,workspace_bytes=4)
            for phase,n,past in [("prefill",s["q"],0),
                *[("decode",1,s["q"]+j) for j in range(s["decode_steps"])]]
            for device in ("cpu","gpu")
            for kind,duration in (("attention_qk",11000),("attention_softmax",13000),
                ("attention_pv",17000),("attention_output",19000))],
        transfer_entries=transfer_rows(s))


def ffn_spec(s, *, shared=False, cut=None):
    i=s["intermediate"];cut=i//4 if cut is None else cut
    return dict(format_version="ffn-execution-0.1",target_kinds=["dense_ffn"],
        intermediate_size=i,shards=[dict(id="host",start=0,end=cut,device="cpu"),
            dict(id="accelerator",start=cut,end=i,device="gpu")],merge_input_count=2,
        preprocess_device="gpu",merge_device="gpu",provenance="synthetic_execution",
        weight_storage_contract="shared_parent_slice_views" if shared else "materialized_partition_replaces_parent",
        entries=[dict(kind=kind,device=device,phase=phase,batch=1,tokens=n,
            visible_history=past,hidden_size=s["hidden"],intermediate_size=width,
            duration_ns=duration,workspace_bytes=4)
            for phase,n,past in [("prefill",s["q"],0),
                *[("decode",1,s["q"]+j) for j in range(s["decode_steps"])]]
            for device in ("cpu","gpu") for width in sorted({cut,i-cut,i})
            for kind,duration in (("ffn_norm",7000),("ffn_gate",11000),
                ("ffn_up",13000),("ffn_swiglu",17000),("ffn_down",19000),
                ("ffn_sum",23000),("ffn_residual",29000))],
        transfer_entries=transfer_rows(s))


def build(s, directory):
    h=s["hidden"];cap=s["physical_capacity_bytes"]
    common=dict(cpu=Device(1e9,1e9),gpu=Device(2e9,2e9),
        transfer_bandwidth_bytes_per_second=1e6 if s.get("regime")=="transfer_constrained" else 1e9,ssd_read_bandwidth_bytes_per_second=1e9,
        ssd_write_bandwidth_bytes_per_second=1e9)
    hw=(Hardware.shared_memory(capacity_bytes=cap,**common) if s["memory_mode"]=="shared"
        else Hardware(dram_capacity_bytes=cap,vram_capacity_bytes=s.get("vram_capacity_bytes",cap),**common))
    params=dict(layers=s["layers"],hidden_size=h,intermediate_size=2*h,heads=2,kv_heads=1)
    model=Model.moe(experts=3,top_k=1,**params) if s["kind"]=="moe" else Model.dense(**params)
    flow=RequestFlow(requests=tuple(Request(f"request{r}",s["q"],s["decode_steps"],r*2000)
        for r in range(s["requests"])),routes=routes_for(s),
        route_source="E1 private fixed model-owned route trace" if s["kind"]=="moe" else None)
    simulator=Simulator(model=model,hardware=hw,workload=flow,cost_model=WitnessCosts(s.get("regime","compute")))
    def configure(plan):
        if s["requests"]>1:
            plan.pipeline(RequestPipeline(ready_rule="decode_first"))
        if s["handoff"]:
            kinds=("projection","attention","router","combine") if s["kind"]=="moe" else ("projection","attention","dense_ffn")
            PhasePlacement(prefill={k:"gpu" for k in kinds},decode={k:"cpu" for k in kinds}).apply(plan)
            plan.handoff(PhaseHandoff(destination="dram"))
        else:
            if not s["attention_refined"]:
                plan.place("attention",s.get("attention_device","cpu"))
            if s["pattern"]=="A01":
                plan.place("projection","cpu")
            elif s.get("projection_device"):
                plan.place("projection",s["projection_device"])
        if s["attention_refined"] or s["ffn_refined"]:
            for r in range(s["requests"]):
                for step in range(s["decode_steps"]+1):
                    label="prefill" if step==0 else f"decode{step-1}"
                    for l in range(s["layers"]):
                        invocation=f"request{r}-{label}-layer{l}"
                        if s["attention_refined"]:
                            plan.transform(invocation,attention_spec(s))
                        if s["ffn_refined"]:
                            cut=(s["intermediate"]*3//4 if r else s["intermediate"]//4)
                            plan.transform(invocation,ffn_spec(s,shared=s["pattern"]=="A12",cut=cut))
            if s["attention_refined"]:
                for kind in ("attention_qk","attention_softmax"):
                    plan.place(kind,"cpu")
    schedule=Scheduling(default_device="gpu",release="last_use",
        cache=Cache(rule="lru",budget_bytes=s["cache_budget_bytes"]) if s["cache"] else Cache(),
        prefetch=Prefetch(next_layers=1,max_count=3,include_dense_weights=True) if s["prefetch"] else Prefetch(),
        expert_rule="current_cost" if s["kind"]=="moe" else "fixed",
        expert_order="history_frequency_desc" if s["kind"]=="moe" else "graph_order")
    return simulator.compile(directory=Path(directory)/"inputs",configure=configure,scheduling=schedule)


def snapshot():
    files=[]
    for source in (ROOT/"src/llm_simulator",ROOT/"mapping/tensor_mapping",ROOT/"modeling/llm_infer_model"):
        files.extend(source.rglob("*.py"))
    files.extend((ROOT/"experiments").glob("dac_validation_e1*.py"))
    files.extend([ROOT/"experiments/strategy_fidelity.py",ROOT/"experiments/ir_acceptance/negative.py",
        ROOT/"experiments/shared_storage_audit.py",ROOT/"experiments/serving_acceptance/oracles.py"])
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}


def _write(path, value):
    ref.write_json(path,value)


def execute_case(s, folder):
    _write(folder/"declaration.json",s)
    expected=declared_dag(s)
    _write(folder/"handwritten-low-level-dag.json",expected)
    with build(s,folder) as compiled:
        _write(folder/"compiled-ir.json",compiled.ir_document())
        result=compiled.run();result.save(folder/"execution")
        ref.require(result.success,f"positive recipe failed: {result.summary()}")
        graph=result.raw[1].execution_document()
        _write(folder/"execution-graph.json",graph)
        events=[asdict(e) for e in result.events]
        _write(folder/"events.json",events)
        observed=check_graph(s,expected,graph,events)
        physical=replay_physical(s,expected,graph,events,result.makespan_ns)
        numeric=check_numerical(s,expected,graph,events)
        policy_audit=result.raw[2].audit_document()
        _write(folder/"policy-audit.json",policy_audit)
        prefetched=[r for r in policy_audit.get("decisions",[]) if r.get("reason")=="cross_layer_public_history_prefetch"]
        if s["prefetch"]:
            ref.require(prefetched,"declared prefetch pattern has no charged prefetch action")
        if s["kind"]=="moe":
            from experiments.serving_acceptance.oracles import expert_visibility
            observed["causal_expert_checks"]=expert_visibility(result)
        observed["charged_prefetch_actions"]=len(prefetched)
        if s["requests"]>1:
            computes=[e for e in events if e["kind"]=="COMPUTE"]
            overlaps=[dict(a=a["target_id"],b=b["target_id"],overlap_ns=min(a["end_ns"],b["end_ns"])-max(a["start_ns"],b["start_ns"]))
                for a in computes for b in computes if a["device"]=="cpu" and b["device"]=="gpu"
                and (":request0:" in a["target_id"] or a["target_id"]=="g:request0:request_arrival")
                and ":request1:" in b["target_id"] and min(a["end_ns"],b["end_ns"])>max(a["start_ns"],b["start_ns"])]
            if s["pattern"] in {"A05","A09"}:ref.require(overlaps,"pipeline pattern has no cross-request overlap")
            observed["cross_request_overlap_witnesses"]=overlaps
        _write(folder/"numerical-outputs.json",numeric.pop("outputs"))
        return dict(status="pass",graph=observed,physical=physical,numerical=numeric,
            summary=result.summary(),backend_patches=0,manually_edited_lowered_artifacts=0)


def negatives(folder):
    from experiments.ir_acceptance import negative
    from llm_simulator import Decision,ModelPlan,InvalidInput
    from tensor_mapping.causal import compile_command
    from tensor_mapping.unified_strategy import create_session
    from llm_infer_model.tensor.runtime import dispatch_command
    checks=[("N01-bad-partition",negative.n01),("N04-capacity",negative.n06),
        ("N05-live-reader",negative.n05),("N06-unrevealed-expert",negative.n08)]
    def missing_merge(p):
        s=spec("A03",8,1)
        with build(s,p) as c:
            setup=c.run();ref.require(setup.success,"missing-merge negative setup failed")
            session=create_session(c.plan._compiled);prefix=[]
            for record in setup.raw[0].decisions:
                before=session.observe()
                found=[o for o in before.operations if o.kind=="ffn_sum"]
                if found:break
                accepted=dispatch_command(session,compile_command(record.decision,before))
                ref.require(accepted.ok,"negative setup prefix rejected");prefix.append(asdict(record.decision))
            else:raise AssertionError("merge never revealed")
            op=found[0]
            step=dispatch_command(session,compile_command(Decision("COMPUTE",op.id,device="gpu"),before))
            _write(p/"before.json",asdict(before));_write(p/"rejected-step.json",asdict(step))
            _write(p/"setup-prefix.json",prefix)
            ref.require(not step.ok and step.error_code=="INPUT_NOT_READY","missing merge predecessor accepted")
            return dict(error_code=step.error_code,message=step.error_message)
    def wrong_version(p):
        s=spec("A10",8,1)
        # Construct valid input files, close the unused program, then inject an
        # invalid *source* state version before a fresh public compilation.
        c=build(s,p);c.close()
        inv=p/"inputs/request0-decode0-layer0.json"
        doc=json.loads(inv.read_text(encoding="utf-8"))
        doc["state_in"]["requests"][0]["version"]+=1
        _write(inv,doc)
        try:
            ModelPlan(id="bad-version",model_kind="global_m2",model_file="model.global.json",
                environment_file="environment.json",base_dir=p/"inputs",
                scheduling=Scheduling(default_device="gpu")).compile()
        except InvalidInput as error:
            ref.require("KV seam mismatch" in str(error),"wrong state-version rejection reason")
            return dict(exception=type(error).__name__,message=str(error))
        raise AssertionError("wrong tensor state version compiled")
    checks.extend([("N02-missing-merge",missing_merge),("N03-wrong-version",wrong_version)])
    rows=[]
    for identity,fn in sorted(checks):
        p=folder/identity;p.mkdir(parents=True)
        try:row=dict(id=identity,status="pass",evidence=fn(p))
        except Exception as error:row=dict(id=identity,status="fail",error=repr(error),traceback=traceback.format_exc())
        _write(p/"result.json",row);rows.append(row)
    return rows


def causality(folder):
    from experiments.serving_acceptance.oracles import canonical_prefix,expert_visibility
    rows=[]
    for mode in ("placement","prefetch"):
        prefixes=[];runs=[]
        for future in (1,2):
            s=spec("A06",8,3,future=future,causal_mode=mode)
            p=folder/mode/f"future-{future}"
            _write(p/"declaration.json",s)
            with build(s,p) as c:
                result=c.run();result.save(p/"execution");ref.require(result.success,"causal alternative failed")
                prefixes.append(canonical_prefix(result,before_route_position=3))
                runs.append(dict(future=future,expert_checks=expert_visibility(result),success=result.success))
        passed=bool(prefixes[0]) and prefixes[0]==prefixes[1]
        _write(folder/mode/"prefixes.json",prefixes)
        row=dict(id=mode,status="pass" if passed else "fail",prefix_decisions=len(prefixes[0]),alternatives=runs,
            oracle="same nonempty public observation/decision prefix before different private future route is revealed")
        _write(folder/mode/"result.json",row);rows.append(row)
    return rows


def run(output, *, only=None):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("output must be new or empty")
    output.mkdir(parents=True,exist_ok=True)
    cases=[spec(identity,h,q) for identity,_ in PATTERNS for h in (8,16) for q in (1,3)]
    manifest=dict(schema="dac-validation-e1-manifest-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        protocol="docs/DAC_VALIDATION_EXPERIMENT_DESIGN_20261024.md",patterns=PATTERNS,cases=cases,
        denominators=dict(positive=48,negative=6,causal_pairs=2,lowering_groups=12,lowering_candidates=48),
        numeric=NUMERICAL_CONTRACT,seed=20261024,atol=5e-5,rtol=5e-5,
        synthetic_times=True,native_timing=False,partial_debug_filter=only,
        external_comparison="source/setup receipts; executed external comparisons require runnable pinned tools")
    _write(output/"manifest.json",manifest);before=snapshot();_write(output/"sources-before.json",before)
    rows=[]
    for s in cases:
        if only and s["id"] not in only:continue
        p=output/"positive"/s["id"];p.mkdir(parents=True)
        start=time.monotonic()
        try:row=dict(id=s["id"],**execute_case(s,p))
        except Exception as error:row=dict(id=s["id"],status="fail",error=repr(error),traceback=traceback.format_exc())
        row["wall_seconds"]=time.monotonic()-start;_write(p/"result.json",row);rows.append(row)
        print(s["id"],row["status"],row.get("error",""),flush=True)
    neg=[];causal=[];lower={};external={}
    if not only:
        neg=negatives(output/"negative");print("negative",sum(r["status"]=="pass" for r in neg),"/6",flush=True)
        try:causal=causality(output/"causal")
        except Exception as error:causal=[dict(id="causal-suite",status="fail",error=repr(error),traceback=traceback.format_exc())]
        lower=run_lowering(output/"lowering")
        from experiments.dac_validation_e1_external import run_external
        external=run_external(output/"external")
    after=snapshot();_write(output/"sources-after.json",after)
    summary=dict(schema="dac-validation-e1-summary-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(output/"manifest.json"),source_unchanged=before==after,
        source_changes=[k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)],
        python=sys.version,platform=platform.platform(),positive=rows,negative=neg,causal_pairs=causal,lowering=lower,external=external,
        counts=dict(positive=sum(r["status"]=="pass" for r in rows),positive_total=48,
            numeric=sum(r.get("numerical",{}).get("status")=="pass" for r in rows),numeric_total=48,
            dense_numeric=sum(r.get("numerical",{}).get("status")=="pass" and not r["id"].startswith(("A06","A11")) for r in rows),
            synthetic_moe_numeric=sum(r.get("numerical",{}).get("status")=="pass" and r["id"].startswith(("A06","A11")) for r in rows),
            negative=sum(r["status"]=="pass" for r in neg),negative_total=6,
            causal=sum(r["status"]=="pass" for r in causal),causal_total=2),
        limitations=["Finite predeclared geometries/patterns, not universal expression coverage or blind holdout.",
            "NumPy action interpretation checks declared graph semantics; does not measure CPU/GPU kernel execution.",
            "MoE physical=None: explicitly synthetic opaque arithmetic, not production MoE/GGML numerical validation.",
            "All compute/copy costs hypothetical; no hardware accuracy claim from E1.",
            "External setup failure does not imply missing expression capability."])
    summary["complete"]=not only and before==after and summary["counts"]["positive"]==48 and summary["counts"]["numeric"]==48 and summary["counts"]["negative"]==6 and summary["counts"]["causal"]==2 and lower.get("complete",False)
    _write(output/"summary.json",summary)
    _write(output/"artifact-hashes.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--only",nargs="*");args=p.parse_args();r=run(args.out,only=args.only)
    print(json.dumps(r["counts"]),flush=True)
    return 0 if r["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
