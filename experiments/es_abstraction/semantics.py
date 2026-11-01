"""Predeclared Dense E/S semantics matrix; analytical, never native timing.

Independent weight algebra is specified before compilation. The public frontend,
shared compiler and runtime are the observed side. No production edits or costs
fitted to these outcomes are made by this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "mapping", ROOT / "modeling"):
    sys.path.insert(0, str(path))

from llm_simulator import (AnalyticalCosts, Cache, DataPreparation, Device,
                          Hardware, Model, Scheduling, Simulator, Workload)
from experiments import strategy_fidelity as ref

CONTRACTS = ("materialized_partition_replaces_parent", "shared_parent_slice_views")


def write(path, value):
    ref.write_json(path, value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    for width, contract, preparation, requests, budget_index in itertools.product(
            (4, 8, 12), CONTRACTS, ("demand", "spare"), (1, 2), range(3)):
        shard_bytes, parent_bytes = 3 * 8 * width * 4, 3 * 8 * 16 * 4
        budgets = (3 * shard_bytes // 4, (shard_bytes + parent_bytes) // 2,
                   math.ceil(1.25 * parent_bytes))
        yield dict(id=f"g{width}-{contract.split('_')[0]}-{preparation}-r{requests}-b{budget_index}",
                   gpu_width=width, cpu_width=16-width, contract=contract,
                   preparation=preparation, requests=requests, budget_index=budget_index,
                   weight_budget_bytes=budgets[budget_index],
                   shard_group_bytes=shard_bytes, parent_group_bytes=parent_bytes)


def expectation(case):
    """Handwritten per-matrix algebra and a serial feasible witness.

    The tensor LRU can stream individual matrices. A budget below the SUM of
    gate/up/down is not inherently infeasible; each matrix is separately used.
    """
    w = case["gpu_width"]
    parent = case["contract"] == CONTRACTS[1]
    weights = {}
    for stage in ("gate", "up", "down"):
        if parent:
            weights[f"w:layer0:W_{stage}"] = 512
        else:
            weights[f"w:layer0:W_{stage}:ffn:host"] = 32*(16-w)
            weights[f"w:layer0:W_{stage}:ffn:accelerator"] = 32*w
    gpu_ids = [f"w:layer0:W_{stage}" + ("" if parent else ":ffn:accelerator")
               for stage in ("gate", "up", "down")]
    # All non-FFN kernels and FFN norm/merge/residual run on CPU. The GPU may
    # retain norm + gate/up/SwiGLU + partial for every request simultaneously.
    gpu_nonweight = case["requests"] * (64 + 12*w)
    gpu_workspace = 4
    minimum_matrix = 512 if parent else 32*w
    return dict(ffn_weights=weights, gpu_weight_ids=gpu_ids,
                parent_weight_bytes=1536, total_ffn_weight_bytes=1536,
                minimum_required_gpu_matrix_bytes=minimum_matrix,
                independently_feasible=minimum_matrix <= case["weight_budget_bytes"],
                feasibility_scope="existence under declared indivisible tensor cache quota; not simultaneous residency of all three matrices",
                gpu_nonweight_inventory_upper_bound=gpu_nonweight,
                gpu_workspace_reserve=gpu_workspace,
                vram_capacity_bytes=case["weight_budget_bytes"]+gpu_nonweight+gpu_workspace,
                dram_capacity_bytes=1<<20,
                simplified_slice_feasible=32*w <= case["weight_budget_bytes"],
                simplified_matrix_copy_bytes=32*w,
                exact_matrix_copy_bytes=minimum_matrix,
                witness="Process requests serially; copy and use one GPU matrix at a time, retaining derived activations; evict matrix before loading the next if needed. CPU performs its own shard and merges both outputs.")


def build(case, exp, folder):
    w = case["gpu_width"]
    spec = ref.ffn_spec()
    spec.update(weight_storage_contract=case["contract"], preprocess_device="cpu", merge_device="cpu")
    spec["shards"][0]["end"] = 16-w
    spec["shards"][1]["start"] = 16-w
    durations = (("ffn_norm",7),("ffn_gate",11),("ffn_up",13),("ffn_swiglu",17),
                 ("ffn_down",19),("ffn_sum",23),("ffn_residual",29))
    spec["entries"] = [dict(kind=k,device=d,phase="prefill",batch=case["requests"],
        tokens=1,visible_history=0,hidden_size=8,intermediate_size=width,
        duration_ns=t,workspace_bytes=4) for k,t in durations for d in ("cpu","gpu")
        for width in sorted({w,16-w,16})]
    sizes = {4,8,16,32,48,64,128,256,384,512,4*w,4*(16-w),32*w,32*(16-w)}
    spec["transfer_entries"] = [dict(route=route,storage_bytes=n,duration_ns=n)
        for route in ("dram_to_vram","vram_to_dram") for n in sorted(sizes)]
    hardware=Hardware(cpu=Device(1e9,1e9),gpu=Device(2e9,2e9),
        dram_capacity_bytes=exp["dram_capacity_bytes"],vram_capacity_bytes=exp["vram_capacity_bytes"],
        transfer_bandwidth_bytes_per_second=1e9,
        ssd_read_bandwidth_bytes_per_second=1e9,ssd_write_bandwidth_bytes_per_second=1e9)
    simulator=Simulator(model=Model.dense(layers=1,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
        hardware=hardware,workload=Workload(1,batch_size=case["requests"]),cost_model=AnalyticalCosts())
    def configure(plan):
        plan.transform("layer0-prefill",spec)
        if case["preparation"] == "spare":
            plan.prepare(DataPreparation(tensor_ids=tuple(exp["gpu_weight_ids"]),trigger="spare"))
    return simulator.compile(directory=folder/"inputs",configure=configure,max_decisions=5000,
        scheduling=Scheduling(default_device="cpu",release="last_use",
            cache=Cache(rule="lru",unit="tensor",budget_bytes=case["weight_budget_bytes"])))


def full_inventory(case, exp):
    """All byte sizes follow the fixture algebra, not the compiler's manifest."""
    sizes={"w:layer0:W_input_norm":32,"w:layer0:W_q":256,
           "w:layer0:W_k":128,"w:layer0:W_v":128,
           "w:layer0:W_output":256,"w:layer0:W_post_attention_norm":32,
           **exp["ffn_weights"]}
    initial=set(sizes)
    for r in range(case["requests"]):
        prefix=f"g:layer0-prefill:t:layer0:request{r}:"
        for suffix,n in (("in.hidden",32),("in.positions",4),("proj.q",32),
                         ("proj.k",16),("proj.v",16),("attn",32),("ffn",32),
                         ("ffn.ffn.norm",32),("ffn.ffn.sum",32)):
            sizes[prefix+suffix]=n
        initial.update((prefix+"in.hidden",prefix+"in.positions"))
        sizes[f"kv:layer0:request{r}:0"]=32
        for shard,w in (("host",case["cpu_width"]),("accelerator",case["gpu_width"])):
            for stage in ("gate","up","swiglu","partial"):
                sizes[prefix+f"ffn.ffn.{shard}.{stage}"]=32 if stage=="partial" else 4*w
    return sizes,initial


def inventory_checks(case, exp, result):
    ops,tensors,computes=ref.observed(result)
    expected_sizes,initial=full_inventory(case,exp)
    ref.equal({tid:t.storage_bytes for tid,t in tensors.items()},expected_sizes,
              "independent complete physical inventory")
    raw_tensors=result.raw[1].execution_document()["workload"]["tensors"]
    for t in raw_tensors:
        ref.equal(t.get("initial_locations",[]),["dram"] if t["id"] in initial else [],
                  "independent initial residency")
    ref.equal(len(ops),13*case["requests"],"independent operation count")
    executed=[e.target_id for e in result.events if e.kind=="COMPUTE"]
    ref.equal(len(executed),len(set(executed)),"no repeated compute introduced")
    if result.success:ref.equal(set(executed),set(ops),"exact required computation coverage")
    actual_weights={tid:t.storage_bytes for tid,t in tensors.items()
                    if tid.startswith("w:layer0:W_") and any(tid.startswith("w:layer0:W_"+s) for s in ("gate","up","down"))}
    ref.equal(actual_weights,exp["ffn_weights"],"independent FFN physical inventory")
    for r in range(case["requests"]):
        prefix=f"g:layer0-prefill:t:layer0:request{r}:"
        def tid(suffix): return prefix+suffix
        wiring={"projection":dict(inputs=["w:layer0:W_input_norm","w:layer0:W_q","w:layer0:W_k","w:layer0:W_v",tid("in.hidden"),tid("in.positions")],outputs=[tid("proj.q"),tid("proj.k"),tid("proj.v")]),
            "attention":dict(inputs=["w:layer0:W_output",tid("proj.q"),tid("proj.k"),tid("proj.v"),tid("in.hidden")],outputs=[tid("attn"),f"kv:layer0:request{r}:0"]),
            "ffn_norm":dict(inputs=["w:layer0:W_post_attention_norm",tid("attn")],outputs=[tid("ffn.ffn.norm")]),
            "ffn_sum":dict(inputs=[tid("ffn.ffn.host.partial"),tid("ffn.ffn.accelerator.partial")],outputs=[tid("ffn.ffn.sum")]),
            "ffn_residual":dict(inputs=[tid("ffn.ffn.sum"),tid("attn")],outputs=[tid("ffn")])}
        for shard,start,end,device in (("host",0,case["cpu_width"],"cpu"),("accelerator",case["cpu_width"],16,"gpu")):
            width=end-start
            for stage in ("gate","up","down"):
                weight=f"w:layer0:W_{stage}"+(f":ffn:{shard}" if case["contract"]==CONTRACTS[0] else "")
                source=tid("ffn.ffn.norm") if stage!="down" else tid(f"ffn.ffn.{shard}.swiglu")
                output=tid(f"ffn.ffn.{shard}."+("partial" if stage=="down" else stage))
                wiring[f"{shard}.{stage}"]=dict(kind="ffn_"+stage,inputs=[weight,source],outputs=[output])
            wiring[f"{shard}.swiglu"]=dict(kind="ffn_swiglu",inputs=[tid(f"ffn.ffn.{shard}.gate"),tid(f"ffn.ffn.{shard}.up")],outputs=[tid(f"ffn.ffn.{shard}.swiglu")])
            for stage in ("gate","up","swiglu","partial"):
                output=tid(f"ffn.ffn.{shard}.{stage}")
                ref.equal(tensors[output].storage_bytes,32 if stage=="partial" else 4*width,"derived activation bytes")
                op=next(o for o in ops.values() if output in o.outputs)
                if op.id in computes:ref.equal(computes[op.id].device,device,"shard placement")
        ref.check_expected_wiring(ops,wiring)
    copy_rows=[]
    for e in result.events:
        if e.kind=="COPY":
            ref.equal(e.bytes_transferred,tensors[e.target_id].storage_bytes,"exact full allocation COPY")
            if e.target_id in exp["gpu_weight_ids"] and e.dst=="vram":
                ref.equal(e.bytes_transferred,exp["exact_matrix_copy_bytes"],"parent-vs-shard COPY payload")
                copy_rows.append(dict(tensor=e.target_id,bytes=e.bytes_transferred,start=e.start_ns,end=e.end_ns))
    return dict(physical_ffn_allocations=len(actual_weights),ffn_weight_bytes=sum(actual_weights.values()),
                complete_inventory_count=len(expected_sizes),initial_dram_bytes=sum(expected_sizes[t] for t in initial),
                checked_operations=len(ops),executed_operations=len(executed),
                weight_h2d_count=len(copy_rows),weight_h2d_bytes=sum(x["bytes"] for x in copy_rows),weight_copies=copy_rows)


def physical_checks(case, exp, result):
    """Replay physical availability/reservations against independently checked sizes."""
    graph=result.raw[1].execution_document()
    tensors={t["id"]:t for t in graph["workload"]["tensors"]}
    ops={o["id"]:o for o in graph["workload"]["operations"]}
    resident={(tid,loc):t["storage_bytes"] for tid,t in tensors.items() for loc in t.get("initial_locations",())}
    pending=[]; active=[];checks=0;peaks={"dram":0,"vram":0};peak_weight=0
    evictions=Counter()
    def finish(t):
        nonlocal pending,active
        remain=[]
        for end,changes in pending:
            if end<=t:resident.update(changes)
            else:remain.append((end,changes))
        pending=remain;active=[a for a in active if a[0]>t]
    def memory():
        nonlocal peak_weight
        allocations=dict(resident)
        for _,changes in pending:allocations.update(changes)
        for loc,capacity in (("dram",exp["dram_capacity_bytes"]),("vram",exp["vram_capacity_bytes"])):
            amount=sum(n for (tid,d),n in allocations.items() if d==loc)+sum(a[3] for a in active if a[1]==loc)
            peaks[loc]=max(peaks[loc],amount)
            ref.require(amount<=capacity,"independent whole-pool capacity/reservation violation")
        weight=sum(n for (tid,d),n in allocations.items() if d=="vram" and tensors[tid]["role"]=="weight")
        peak_weight=max(peak_weight,weight)
        ref.require(weight<=case["weight_budget_bytes"],"independent FFN cache quota violation")
    for e in result.events:
        finish(e.t_ns)
        if e.kind=="COPY":
            ref.require((e.target_id,e.src) in resident,"COPY source unavailable")
            pending.append((e.end_ns,{(e.target_id,e.dst):tensors[e.target_id]["storage_bytes"]}))
            active.append((e.end_ns,e.src,{e.target_id},0))
        elif e.kind=="COMPUTE":
            op=ops[e.target_id];loc="dram" if e.device=="cpu" else "vram"
            for tid in op["inputs"]:ref.require((tid,loc) in resident,"COMPUTE operand unavailable");checks+=1
            outputs=[op["output"],*op.get("extra_outputs",[])]
            pending.append((e.end_ns,{(tid,loc):tensors[tid]["storage_bytes"] for tid in outputs}))
            workspace=4 if op["semantic_op"].startswith("ffn_") else 0
            ref.equal(e.workspace_bytes,workspace,"declared independent workspace size")
            active.append((e.end_ns,loc,set(op["inputs"]),workspace))
        elif e.kind=="EVICT":
            ref.require((e.target_id,e.location) in resident,"EVICT missing allocation")
            ref.require(not any(loc==e.location and e.target_id in readers for _,loc,readers,_ in active),"EVICT live reader")
            resident.pop((e.target_id,e.location))
            evictions[tensors[e.target_id]["role"]]+=1
        elif e.kind!="ADVANCE":raise AssertionError("unexpected physical action "+e.kind)
        memory()
    ref.equal(peaks["dram"],result.summary()["peak_dram_bytes"],"independent DRAM peak ledger")
    ref.equal(peaks["vram"],result.summary()["peak_vram_bytes"],"independent VRAM peak ledger")
    return dict(checked_operand_availability=checks,peak_physical_bytes=peaks,peak_weight_bytes=peak_weight,
                evictions_by_role=dict(evictions),
                reservation_and_live_reader_violations=0,scope="separate DRAM/VRAM allocation inventory; FFN workspace4B, other synthetic operators0B; no CUDA scratch claim")


def numeric_checks(case, exp, result):
    """FFN-only scalar mathematical reference vs compiled-action NumPy replay."""
    import numpy as np
    rng=np.random.default_rng(20261030)
    gate=rng.uniform(-.5,.5,(16,8)).astype(np.float32)
    up=rng.uniform(-.5,.5,(16,8)).astype(np.float32)
    down=rng.uniform(-.5,.5,(8,16)).astype(np.float32)
    gamma=rng.uniform(.8,1.2,8).astype(np.float32)
    inputs=[rng.uniform(-.5,.5,(1,8)).astype(np.float32) for _ in range(case["requests"])]
    def scalar_mm(x,w):return np.asarray([[math.fsum(float(a)*float(b) for a,b in zip(row,col)) for col in w] for row in x],dtype=np.float64)
    refs=[]
    for x in inputs:
        norm=np.asarray([[float(v)/math.sqrt(math.fsum(float(z)**2 for z in row)/8+1e-5)*float(gamma[c]) for c,v in enumerate(row)] for row in x])
        ga=scalar_mm(norm,gate);uu=scalar_mm(norm,up)
        sw=np.asarray([[float(v)/(1+math.exp(-float(v)))*float(uu[t,c]) for c,v in enumerate(row)] for t,row in enumerate(ga)])
        refs.append((scalar_mm(sw,down)+x).astype(np.float32))
    graph=result.raw[1].execution_document();ops={o["id"]:o for o in graph["workload"]["operations"]}
    buffers={};pending=[];outputs={};numeric_copies=0;numeric_ops=0
    for stage,array in (("gate",gate),("up",up),("down",down)):
        base=f"w:layer0:W_{stage}"
        if case["contract"]==CONTRACTS[1]:buffers[base,"dram"]=array
        else:
            for shard,a,b in (("host",0,case["cpu_width"]),("accelerator",case["cpu_width"],16)):
                buffers[base+":ffn:"+shard,"dram"]=(array[:,a:b] if stage=="down" else array[a:b]).copy()
    buffers["w:layer0:W_post_attention_norm","dram"]=gamma
    def finish(t):
        nonlocal pending
        later=[]
        for end,values in pending:
            if end<=t:buffers.update(values);outputs.update({k[0]:v for k,v in values.items()})
            else:later.append((end,values))
        pending=later
    for e in result.events:
        finish(e.t_ns)
        if e.kind=="EVICT":buffers.pop((e.target_id,e.location),None)
        elif e.kind=="COPY" and (e.target_id,e.src) in buffers:
            pending.append((e.end_ns,{(e.target_id,e.dst):buffers[e.target_id,e.src].copy()}));numeric_copies+=1
        elif e.kind=="COMPUTE":
            op=ops[e.target_id];kind=op["semantic_op"];loc="dram" if e.device=="cpu" else "vram"
            outs=[op["output"],*op.get("extra_outputs",[])]
            if kind=="attention":
                ri=int(outs[0].split(":request")[1].split(":")[0])
                pending.append((e.end_ns,{(outs[0],loc):inputs[ri]}))
            elif kind.startswith("ffn_"):
                ins=[np.asarray(buffers[t,loc],dtype=np.float64) for t in op["inputs"]]
                if kind=="ffn_norm":z=ins[1]/np.sqrt(np.mean(ins[1]*ins[1],axis=-1,keepdims=True)+1e-5)*ins[0]
                elif kind in {"ffn_gate","ffn_up","ffn_down"}:
                    w,x=ins;a,b=op["op_params"]
                    if case["contract"]==CONTRACTS[1]:w=w[:,a:b] if kind=="ffn_down" else w[a:b]
                    z=x@w.T
                elif kind=="ffn_swiglu":z=ins[0]/(1+np.exp(-ins[0]))*ins[1]
                elif kind in {"ffn_sum","ffn_residual"}:z=sum(ins)
                else:raise AssertionError(kind)
                pending.append((e.end_ns,{(outs[0],loc):np.asarray(z,dtype=np.float32)}));numeric_ops+=1
    finish(float("inf"))
    errors=[]
    for r,expected in enumerate(refs):
        actual=outputs[f"g:layer0-prefill:t:layer0:request{r}:ffn"]
        ref.require(np.allclose(actual,expected,atol=5e-5,rtol=5e-5),"independent FFN numeric result mismatch")
        errors.append(float(np.max(np.abs(actual-expected))))
    return dict(max_absolute_error=max(errors),checked_requests=len(refs),numeric_compute_actions=numeric_ops,
                numeric_copies=numeric_copies,atol=5e-5,rtol=5e-5,
                scope="seeded FFN attention-output input boundary, scalar unsplit FFN versus NumPy compiled FFN actions; other Dense operators checked structurally only; no native timing")


def run(out, limit=None):
    if out.exists() and any(out.iterdir()):raise ValueError("output must be new or empty")
    out.mkdir(parents=True,exist_ok=True)
    (out/"source").mkdir()
    (out/"source"/"semantics.executed.py").write_bytes(Path(__file__).read_bytes())
    plan=ROOT.parent/"paper-write/第八教学周/experiment-reports/es-abstraction-plan-20261030/EXPERIMENT_PROTOCOL.json"
    matrix=list(cases())
    if limit is not None:matrix=matrix[:limit]
    protocol=dict(schema="es-semantics-72-v2",utc=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(plan),
        source_sha256=digest(__file__),cases=matrix,analytical=True,native_measurements=False,
        non_ffn_placement="cpu",norm_merge_residual="cpu",copy_ns_per_byte=1,
        cache_unit="tensor",quota_note="FFN-only GPU weight sub-budget because all other weight-consuming work is CPU; whole GPU pool adds independent maximum nonweight inventory and workspace.",
        feasibility_note="budget levels use total three-matrix sizes but feasibility uses required indivisible matrices; streaming and eviction are allowed. No total-resident requirement invented.",
        execution_before_observation_clarification="Sent to the coordinator before the first pilot: total gate/up/down size is not a mandatory simultaneous residency bound; use per-matrix quota and an independent live allocation ledger. Original 72 combinations and budget values unchanged.",
        pilot_record="pilot-v1: two oracle-harness KeyErrors for optional initial_locations; simulator itself completed both. pilot-v2: 24 cases,20 feasible,4 correct rejections. Retained separately; no production-core change.",
        ablation="artificial independent slice-as-allocation model; not attributed to any named simulator",
        numerical_scope="FFN input/output logical numerical preservation, not full Dense/native numerical accuracy",
        novelty_boundary="Physical allocation/view correctness is a necessary semantic obligation, not automatically a novel abstraction; existing approaches such as WriteScope may represent related mechanisms.",
        causal_scope="Static Dense workloads; no hidden expert routes, future route predictor or causal-prefix test is present in this matrix.")
    write(out/"PROTOCOL.json",protocol)
    before=ref.source_snapshot();write(out/"source-manifest-before.json",before)
    results=[];start=time.perf_counter()
    for case in matrix:
        folder=out/case["id"];folder.mkdir();exp=expectation(case);write(folder/"expected.json",exp)
        row=dict(case=case,expected=exp,status="error",failure_class=None)
        phase="compile"
        try:
            with build(case,exp,folder) as compiled:
                write(folder/"compiled-ir.json",compiled.ir_document())
                phase="runtime"
                result=compiled.run();result.save(folder/"execution")
                write(folder/"graph.json",result.raw[1].execution_document())
                write(folder/"events.json",[asdict(e) for e in result.events])
                phase="independent_oracle"
                row.update(actual_success=result.success,execution_summary=result.summary(),
                    inventory=inventory_checks(case,exp,result),physical=physical_checks(case,exp,result))
                if result.success:
                    ref.require(exp["independently_feasible"],"false feasible admitted execution")
                    row["numerical"]=numeric_checks(case,exp,result)
                    row.update(status="valid",duration_ns=result.makespan_ns)
                else:
                    row.update(status="rejected",failure_class="physically_infeasible" if not exp["independently_feasible"] else "compile_or_runtime_bug")
        except Exception as error:
            row.update(error_type=type(error).__name__,error=str(error),phase=phase,traceback=traceback.format_exc())
            row["failure_class"]="harness_error" if phase=="independent_oracle" and not isinstance(error,AssertionError) else "compile_or_runtime_bug"
        results.append(row);write(folder/"result.json",row);write(out/"progress.json",results)
        print(f"{len(results)}/{len(matrix)} {case['id']} {row['status']} {row.get('error','')}",flush=True)
    after=ref.source_snapshot();modules=ref.imported_sources();ref.verify_source_integrity(before,after,modules)
    ref.equal(digest(__file__),protocol["source_sha256"],"executed experiment bytes unchanged")
    write(out/"source-manifest-after.json",after);write(out/"imported-sources.json",modules)
    summary=dict(schema="es-semantics-result-1",protocol_sha256=digest(out/"PROTOCOL.json"),
        status_counts=dict(Counter(x["status"] for x in results)),
        failure_classes=dict(Counter(x["failure_class"] for x in results if x["failure_class"])),
        nominal_cases=len(results),elapsed_seconds=time.perf_counter()-start,
        simplified_false_feasible=sum(x["expected"]["simplified_slice_feasible"] and not x["expected"]["independently_feasible"] for x in results),
        independent_feasible=sum(x["expected"]["independently_feasible"] for x in results),
        independent_infeasible=sum(not x["expected"]["independently_feasible"] for x in results),
        false_feasible=sum(x.get("actual_success",False) and not x["expected"]["independently_feasible"] for x in results),
        false_infeasible=sum(x.get("actual_success") is False and x["expected"]["independently_feasible"] for x in results),
        max_ffn_absolute_error=max((x.get("numerical",{}).get("max_absolute_error",0) for x in results),default=0),
        numerical_requests=sum(x.get("numerical",{}).get("checked_requests",0) for x in results),
        source_integrity_verified=True,
        evidence_scope=protocol,results=results)
    write(out/"REPORT.json",summary)
    report(out,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"results","evidence_scope"}},indent=2),flush=True)
    return summary


def report(out, summary):
    columns=("case","gpu_width","contract","preparation","requests","budget_bytes",
             "oracle_feasible","status","weight_copy_bytes","weight_copy_count",
             "peak_vram_bytes","peak_weight_bytes","duration_ns","numerical_error")
    rows=[]
    for x in summary["results"]:
        c=x["case"];p=x.get("physical",{});i=x.get("inventory",{})
        rows.append(dict(zip(columns,(c["id"],c["gpu_width"],c["contract"],c["preparation"],
            c["requests"],c["weight_budget_bytes"],x["expected"]["independently_feasible"],x["status"],
            i.get("weight_h2d_bytes"),i.get("weight_h2d_count"),p.get("peak_physical_bytes",{}).get("vram"),
            p.get("peak_weight_bytes"),x.get("duration_ns"),x.get("numerical",{}).get("max_absolute_error")))))
    with (out/"CASES.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    headers="".join(f"<th>{html.escape(c)}</th>" for c in columns)
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(r[c]))}</td>" for c in columns)+"</tr>" for r in rows)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>E/S 语义矩阵</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 20px;line-height:1.6;color:#17212b;background:white}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #cbd5e1;padding:5px}}th{{background:#edf2f7}}.note{{border-left:4px solid #567; padding:12px;background:#f5f7fa}}code{{background:#eef;padding:2px}}</style>
<h1>实验 A：执行分片与存储物理身份的语义验收</h1>
<p>共 {summary['nominal_cases']} 个预设组合，独立可行 {summary['independent_feasible']} 个，不可行 {summary['independent_infeasible']} 个。运行状态：{html.escape(str(summary['status_counts']))}。误判可行 {summary['false_feasible']}，误判不可行 {summary['false_infeasible']}。</p>
<p>独立 FFN 数值对照覆盖 {summary['numerical_requests']} 个请求输出，最大绝对误差 {summary['max_ffn_absolute_error']:.3g}；输入为固定种子 F32 张量。</p>
<h2>验证什么</h2><p>H=8、I=16、F32；GPU 分片宽度 4/8/12；实体分片/完整父张量视图；按需/空闲预载；1/2 请求；3 档权重预算。实际调用公开 Python 接口→现有 IR/Compiler→事件执行器。独立参考先写出权重与激活字节数、算子接线和初始位置，再按事件检查输入就绪、复制字节、输出预留、workspace、活跃读者和两存储池峰值。</p>
<h2>关键见证</h2><p>宽度4时，一片矩阵是128B，但父矩阵是512B。288B权重预算可容纳实体矩阵分片，却放不下完整父矩阵。把逻辑视图错当成独立片段，会将 {summary['simplified_false_feasible']} 个不可行组合误判为可行。该消融是人工简化模型，不是对任何具名模拟器的实现指控。</p>
<p>三个矩阵总量不是同时驻留的必要条件：tensor-LRU可以逐矩阵加载、执行和驱逐。预算档位沿用原计划；判定依据在首次试运行之前已澄清为最小必需矩阵及真实活跃峰值，不能用“三权重之和超过预算”制造失败。</p>
<h2>证据边界</h2><div class="note">本实验是解析语义证据，不是真机性能或CUDA物理分配验收。图内耗时使用明确合成价格，不能引用为加速比。FFN数值解释器比较未拆分的标量数学参考与编译动作；Attention等其余算子仅检查结构/资源语义。静态Dense矩阵未验证MoE因果路由。必要的物理身份约束也不自动构成新颖性：与WriteScope等相关抽象的区别需由另行源码对比给出。</div>
<p>首个试运行有两条独立检查器 KeyError（可省略 initial_locations 字段），当时模拟器执行已成功；已保留原始失败并修复实验检查器，公共核心没有修改。第二次24条试运行保留在相邻目录。正式运行已冻结脚本与核心源码hash。</p>
<p>完整证据：<a href="REPORT.json">REPORT.json</a>、<a href="CASES.csv">CASES.csv</a>、<a href="PROTOCOL.json">PROTOCOL.json</a>、<a href="source/semantics.executed.py">执行脚本</a>；每个case目录含预先期望、IR、计算图、事件、执行器报告及独立验证结果。</p>
<h2>全部结果</h2><table><tr>{headers}</tr>{table}</table></html>'''
    (out/"REPORT_zh.html").write_text(text,encoding="utf-8",newline="\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);parser.add_argument("--limit",type=int)
    args=parser.parse_args();run(args.out.resolve(),args.limit)
