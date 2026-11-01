"""Finite paper expression evidence; no production edits or hardware timing claims.

Four existing reference mechanisms and three newly declared compositions share
the public ModelPlan -> IR -> compiler -> runtime. All denominators, oracle
obligations and source bytes are persisted before the first case executes.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "mapping", ROOT / "modeling"):
    sys.path.insert(0, str(folder))

from experiments import strategy_fidelity as ref
from experiments.shared_storage_audit import audit_documents
from experiments.ir_acceptance import basic, combinations
from experiments.serving_acceptance import cases as serving, oracles
from llm_simulator import (
    AnalyticalCosts, Cache, DataPreparation, EnvironmentEvent, Hardware, Model, PhaseHandoff,
    PhasePlacement, Prefetch, Request, RequestFlow, RequestPipeline,
    Scheduling, Simulator, Workload,
)


CASES = [
    dict(id="R01", name="Unequal CPU/GPU FFN partition and merge", group="reference_rerun",
         oracle="Exact gate/up/down operands; 128/384-byte leaves; 1536-byte conserved weights; 32-byte CPU partial returned before SUM; dependencies."),
    dict(id="R02", name="Two-request CPU Attention/GPU FFN pipeline with prefetch", group="reference_rerun",
         oracle="One runtime; exact request/layer/round scopes; positive cross-request CPU/GPU overlap; charged next-layer prefetch; no capacity overflow."),
    dict(id="R03", name="Causal expert placement, shared cache and prefetch", group="reference_rerun",
         oracle="Revealed expert rows only; 1536-byte E0 identity reused across requests and rounds without reload; charged next-layer prefetch; exact weights."),
    dict(id="R04", name="All-layer PD KV handoff with cache and prefetch", group="reference_rerun",
         oracle="Every prefill layer/token KV is 32 bytes, transferred once; Decode gate consumes final hidden and all KV; no early Decode; growing identities."),
    dict(id="X01", name="Attention stages plus FFN partition, startup and CPU recovery", group="new_composition",
         oracle="All Attention/FFN operand contracts together; three 384-byte startup weights; 32-byte partial return; no CPU compute before 10000ns; unchanged 1536-byte FFN inventory."),
    dict(id="X02", name="Asynchronous request pipeline, prefetch and GPU recovery", group="new_composition",
         oracle="Exact request scopes/arrivals/retirement; CPU Attention/GPU FFN overlap and async progress; real charged prefetch; no GPU compute before 10000ns; one capacity ledger."),
    dict(id="X03", name="PD MoE with history order, prefetch and hidden routes under CPU recovery", group="new_composition",
         oracle="Two complete alternatives; exact all-layer PD handoff and route visibility; real prefetch; CPU recovery at 10000ns; identical nonempty public prefixes before first differing route."),
    dict(id="X04", name="Request-local unequal FFN slices sharing parent weights in one physical pool", group="new_composition",
         oracle="Request0 cut4/12 and request1 cut12/4; both read the same three 512-byte parent weights; total FFN1536/allweights2368 bytes; zero-payload SHARE, no fake partial weight COPY; independently replayed physical capacity."),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources():
    files = set()
    for directory in ref.SOURCE_ROOTS.values():
        files.update(directory.rglob("*.py"))
    files.add(Path(__file__))
    files.add(Path(ref.__file__))
    files.add(ROOT / "experiments/shared_storage_audit.py")
    for directory in (ROOT / "experiments/ir_acceptance", ROOT / "experiments/serving_acceptance"):
        files.update(directory.glob("*.py"))
        files.update(directory.glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def source_metrics(source):
    """Count actual supplied Python, including helper/literal declarations."""
    parsed = ast.parse(textwrap.dedent(source))
    return dict(nonblank_noncomment_lines=sum(bool(line.strip()) and not line.lstrip().startswith("#")
                for line in source.splitlines()), ast_statements=sum(isinstance(n, ast.stmt) for n in ast.walk(parsed)))


def write_authoring(directory, functions):
    parts = [textwrap.dedent(inspect.getsource(f)) for f in dict.fromkeys(functions)]
    source = "\n\n".join(parts)
    path = directory / "authored-recipe-and-helpers.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    return dict(**source_metrics(source), source_file=str(path), sha256=sha(path),
        scope="Conservative full recipe/helper source, including workload declarations, exact synthetic-price literals and embedded checks; not a pure user-interface LOC count.",
        backend_files_edited=0, manual_lowered_graph_edits=0,
        low_level_baseline_lines=None, human_minutes=None,
        note="No matched hand-written low-level baseline or user study; no compression or productivity claim.")


def check_flat_record(document):
    """Compiler-independent event invariants; expected geometry is supplied separately.

    Full physical-tensor COPY semantics apply to these finite cases. Exact
    operation wiring is checked by separately handwritten geometry references.
    """
    ops = {o["id"]: o for o in document["operations"]}
    tensors = {t["id"]: t for t in document["tensors"]}
    computes = {}
    for event in document["events"]:
        if event["kind"] == "COMPUTE":
            target = event["target_id"]
            ref.require(target not in computes, "duplicate compute charge")
            ref.require(event["end_ns"] >= event["start_ns"], "negative compute interval")
            computes[target] = event
        if event["kind"] == "COPY":
            ref.equal(event["bytes_transferred"], tensors[event["target_id"]]["storage_bytes"], "COPY physical payload")
            ref.require(event["end_ns"] > event["start_ns"], "uncharged COPY")
    producer = {}
    for op in ops.values():
        for tensor in op["outputs"]:
            ref.require(tensor not in producer, "duplicate tensor producer")
            producer[tensor] = op["id"]
    edges = 0
    for op in ops.values():
        ref.require(op["id"] in computes, "incomplete operation")
        for tensor in op["inputs"]:
            parent = producer.get(tensor)
            if parent is not None:
                ref.require(computes[op["id"]]["start_ns"] >= computes[parent]["end_ns"], "consumer before producer")
                edges += 1
    for observation in document["memory"]:
        for memory in observation:
            if memory["capacity_bytes"] is not None:
                ref.require(memory["used_bytes"] <= memory["capacity_bytes"], "physical capacity exceeded")
    for event in document["events"]:
        if event["kind"] != "EVICT":
            continue
        for target, compute in computes.items():
            if compute["start_ns"] <= event["start_ns"] < compute["end_ns"]:
                domain = "dram" if compute["device"] == "cpu" else "vram"
                ref.require(event["location"] != domain or event["target_id"] not in ops[target]["inputs"], "live input evicted")
    blackout = document.get("initial_blackout")
    if blackout:
        affected = [e for e in computes.values() if e["device"] == blackout["device"]]
        ref.require(affected, "blackout device was never exercised")
        ref.require(all(e["start_ns"] >= blackout["until_ns"] for e in affected), "compute during initial blackout")
    return dict(computes=len(computes), dependency_edges=edges,
                copies=sum(e["kind"] == "COPY" for e in document["events"]),
                memory_observations=len(document["memory"]))


def save_flat(result, directory, blackout=None):
    ops, tensors, _ = ref.observed(result)
    document = dict(operations=[asdict(o) for o in ops.values()], tensors=[asdict(t) for t in tensors.values()],
        events=[asdict(e) for e in result.events],
        memory=[[asdict(m) for m in row.observation.memory] for row in result.raw[0].decisions],
        initial_blackout=blackout)
    ref.write_json(directory / "flat-observed.json", document)
    return check_flat_record(document)


def x01(directory):
    expected = dict(attention_operands=ref.attention_wiring(), ffn_operands=ref.ffn_wiring(),
        leaves={"host":128,"accelerator":384}, total_ffn_bytes=1536, startup_total_bytes=1152,
        partial_bytes=32, cpu_available_after_ns=10000, cache_budget_bytes=4096)
    ref.write_json(directory / "expected.json", expected)
    def configure(plan):
        plan.transform("layer0-prefill", ref.attention_spec())
        plan.transform("layer0-prefill", ref.ffn_spec())
        plan.place("attention_qk", "cpu")
        plan.place("attention_softmax", "cpu")
        plan.prepare(DataPreparation(tensor_ids=tuple(f"w:layer0:W_{s}:ffn:accelerator" for s in ("gate","up","down")), trigger="startup"))
    with ref.dense().compile(directory=directory / "inputs", configure=configure,
        scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=4096)),
        environment_events=(EnvironmentEvent(t_ns=0,cpu_available=False),EnvironmentEvent(t_ns=10000,cpu_available=True))) as compiled:
        ref.write_json(directory / "compiled-ir.json", compiled.ir_document())
        result = ref.completed(compiled,directory)
        ops,tensors,events = ref.observed(result)
        operands=ref.check_expected_wiring(ops,expected["attention_operands"])+ref.check_expected_wiring(ops,expected["ffn_operands"])
        for stage in ("gate","up","down"):
            ref.require(f"w:layer0:W_{stage}" not in tensors,"materialized shards retained parent")
            for shard,size in expected["leaves"].items():
                ref.equal(tensors[f"w:layer0:W_{stage}:ffn:{shard}"].storage_bytes,size,"FFN leaf geometry")
            copied=ref.check_copy(result,f"w:layer0:W_{stage}:ffn:accelerator","dram","vram",384)
            ref.require(copied.end_ns <= min(e.start_ns for e in events.values()),"startup did not precede compute")
        moved=ref.check_copy(result,ref.tensor_id("ffn.ffn.host.partial"),"dram","vram",32)
        ref.require(events[ref.one_kind(ops,"ffn_sum").id].start_ns>=moved.end_ns,"merge before partial return")
        return dict(exact_operand_contracts=operands,peak_weight_bytes=combinations._check_budget(result,4096),
            flat_oracle=save_flat(result,directory,dict(device="cpu",until_ns=10000)),summary=result.summary())


def blackout_events(blackout):
    if blackout is None:
        return ()
    if blackout["device"]=="cpu":
        return (EnvironmentEvent(t_ns=0,cpu_available=False),
                EnvironmentEvent(t_ns=blackout["until_ns"],cpu_available=True))
    return (EnvironmentEvent(t_ns=0,gpu_service_rate_numerator=0,gpu_service_rate_denominator=1),
            EnvironmentEvent(t_ns=blackout["until_ns"],gpu_service_rate_numerator=1,gpu_service_rate_denominator=1))


def build_serving(directory, *, kind="dense", handoff=False, blackout=None, future_expert=0):
    prefill=3 if kind=="moe" else 1
    requests,scopes,routes=serving._declarations(kind=kind,prefill=prefill,decode=(1,2),future_expert=future_expert)
    expected=dict(requests=requests,scope_map=scopes,route_rows=[asdict(r) for r in routes],
        weight_inventory=serving._weight_inventory(kind,2),cache_budget_bytes=32768,
        prefill_tokens=prefill,kv_bytes_per_token=32,blackout=blackout,
        execution="CPU Attention/GPU FFN" if not handoff else "GPU Prefill/CPU Decode with causal expert placement",
        prefetch="one next layer; max_count3",expert_order="history_frequency_desc")
    ref.write_json(directory / "expected.json",expected)
    model=(Model.dense(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1) if kind=="dense" else
           Model.moe(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1,experts=3,top_k=1))
    flow=RequestFlow(requests=tuple(Request(request_id=r,prefill_tokens=d["prefill_tokens"],decode_steps=d["decode_steps"],arrival_ns=d["arrival_ns"]) for r,d in requests.items()),
        routes=routes,route_source="paper-expression-declared-private-routes" if routes else None)
    simulator=Simulator(model=model,hardware=ref.hardware(),workload=flow,cost_model=serving.DeclaredSyntheticCosts())
    def configure(plan):
        plan.pipeline(RequestPipeline(ready_rule="decode_first"))
        if handoff:
            kinds=("projection","attention","router","combine")
            PhasePlacement(prefill={k:"gpu" for k in kinds},decode={k:"cpu" for k in kinds}).apply(plan)
            plan.handoff(PhaseHandoff(destination="dram"))
        else:
            plan.place("projection","cpu")
            plan.place("attention","cpu")
    environment=blackout_events(blackout)
    compiled=simulator.compile(directory=directory / "inputs",configure=configure,environment_events=environment,
        scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=32768),
            prefetch=Prefetch(next_layers=1,max_count=3,include_dense_weights=True),
            expert_rule="fixed" if kind=="dense" else "current_cost",expert_order="history_frequency_desc"))
    ref.write_json(directory / "compiled-ir.json",compiled.ir_document())
    return compiled,expected


def serving_checks(result,expected,directory):
    evidence=oracles.complete_and_capacity(result,cache_budget=32768,weight_inventory=expected["weight_inventory"])
    evidence["lifecycle"]=oracles.lifecycle(result,requests=expected["requests"],scope_map=expected["scope_map"],
        kinds=dict(dense=("projection","attention","dense_ffn"),moe=("projection","attention","router","combine")))
    evidence["prefetch"]=oracles.prefetch_witnesses(result,expected["scope_map"])
    evidence["flat_oracle"]=save_flat(result,directory,expected["blackout"])
    return evidence


def r02(directory):
    with build_serving(directory)[0] as compiled:
        expected=json.loads((directory / "expected.json").read_text(encoding="utf-8"))
        result=ref.completed(compiled,directory)
        evidence=serving_checks(result,expected,directory)
        evidence["overlap"]=oracles.execution_overlap(result,expected["scope_map"])
        evidence["async_progress"]=oracles.asynchronous_progress(result,expected["scope_map"])
        return evidence


def x02(directory):
    compiled,expected=build_serving(directory,blackout=dict(device="gpu",until_ns=10000))
    with compiled:
        result=ref.completed(compiled,directory)
        evidence=serving_checks(result,expected,directory)
        evidence["overlap"]=oracles.execution_overlap(result,expected["scope_map"])
        evidence["async_progress"]=oracles.asynchronous_progress(result,expected["scope_map"])
        return evidence


def x03(directory):
    prefixes=[];evidence=[]
    ref.write_json(directory / "expected.json",dict(private_future_experts=[1,2],first_different_position=3,
        required="identical nonempty canonical observation/decision prefixes until first differing route; both entire workloads finish"))
    for expert in (1,2):
        child=directory / f"future-{expert}"
        compiled,expected=build_serving(child,kind="moe",handoff=True,blackout=dict(device="cpu",until_ns=10000),future_expert=expert)
        with compiled:
            result=ref.completed(compiled,child)
            row=serving_checks(result,expected,child)
            row["handoff"]=oracles.pd_handoff(result,requests=expected["requests"],scope_map=expected["scope_map"])
            row["expert_visibility"]=oracles.expert_visibility(result)
            prefixes.append(oracles.canonical_prefix(result,before_route_position=3))
            evidence.append(row)
    ref.equal(prefixes[0],prefixes[1],"private future routes changed earlier public behavior")
    ref.write_json(directory / "causal-prefix.json",prefixes[0])
    return dict(alternatives=evidence,identical_prefix_decisions=len(prefixes[0]),
        prefix_sha256=sha(directory / "causal-prefix.json"))


def x04(directory):
    expected=dict(request_cuts={"request0":4,"request1":12},parent_matrix_bytes=512,
        parent_ids=[f"w:layer0:W_{s}" for s in ("gate","up","down")],
        ffn_weights_bytes=1536,all_weights_bytes=2368,physical_capacity_bytes=16384,
        memory_mode="shared",storage_contract="shared_parent_slice_views",
        copy_payload_bytes=0,partial_bytes=32)
    ref.write_json(directory / "expected.json",expected)
    discrete=ref.hardware()
    hardware=Hardware.shared_memory(cpu=discrete.cpu,gpu=discrete.gpu,capacity_bytes=16384)
    simulator=Simulator(model=Model.dense(layers=1,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
        hardware=hardware,workload=Workload(1,batch_size=2),cost_model=AnalyticalCosts())
    def configure(plan):
        for request,cut in expected["request_cuts"].items():
            spec=ref.ffn_spec()
            spec["weight_storage_contract"]="shared_parent_slice_views"
            for row in spec["entries"]:
                row["batch"]=2
            spec["shards"][0]["end"]=cut
            spec["shards"][1]["start"]=cut
            with plan.module("slice-"+request,request_id=request):
                plan.transform("layer0-prefill",spec)
    with simulator.compile(directory=directory / "inputs",configure=configure,
            scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=8192))) as compiled:
        ref.write_json(directory / "compiled-ir.json",compiled.ir_document())
        result=ref.completed(compiled,directory)
        ops,tensors,events=ref.observed(result)
        graph=result.raw[1].execution_document()
        raw_ops={o["id"]:o for o in graph["workload"]["operations"]}
        for parent in expected["parent_ids"]:
            ref.equal(tensors[parent].storage_bytes,512,"full physical parent geometry")
        ref.equal(sum(t.storage_bytes for t in tensors.values() if t.role=="weight"),2368,"one physical weight inventory")
        ref.require(not any(t.role=="weight" and ":ffn:" in t.id for t in tensors.values()),"slice created duplicate weights")
        operands=0
        for request,cut in expected["request_cuts"].items():
            wiring=json.loads(json.dumps(ref.ffn_wiring()).replace(":request0:",":"+request+":"))
            for contract in wiring.values():
                contract["inputs"]=[tid.split(":ffn:")[0] if tid.startswith("w:") else tid for tid in contract["inputs"]]
            operands+=ref.check_expected_wiring(ops,wiring)
            for shard,start,end,device in (("host",0,cut,"cpu"),("accelerator",cut,16,"gpu")):
                for stage in ("gate","up","swiglu","down"):
                    output=f"g:layer0-prefill:t:layer0:{request}:ffn.ffn.{shard}."+("partial" if stage=="down" else stage)
                    producer=next(o for o in ops.values() if output in o.outputs)
                    ref.equal(raw_ops[producer.id]["op_params"],[start,end],"request-specific slice range")
                    ref.equal(events[producer.id].device,device,"request-local slice execution device")
                    ref.equal(tensors[output].storage_bytes,32 if stage=="down" else 4*(end-start),"slice activation geometry")
        shares=[e for e in result.events if e.kind=="SHARE"]
        ref.require(shares,"one-pool strategy never exercised sharing")
        ref.require(not any(e.kind=="COPY" for e in result.events),"shared views incorrectly copied payload")
        ref.require(all(e.bytes_transferred==0 and e.start_ns==e.end_ns for e in shares),"SHARE charged physical transfer")
        summary=dict(result.summary(),completed_makespan_ns=result.makespan_ns)
        audit=audit_documents(graph,[asdict(e) for e in result.events],
                              [asdict(s) for s in result.evaluation.states],summary)
        ref.write_json(directory / "physical-replay-audit.json",audit)
        return dict(exact_operand_contracts=operands,shared_events=len(shares),
            parent_weights_bytes=1536,complete_weights_bytes=2368,independent_physical_replay=audit,
            flat_oracle=save_flat(result,directory),summary=result.summary())


RUNNERS={"R01":ref.ffn_partition,"R02":r02,"R03":serving.sc03,"R04":serving.sc02,
         "X01":x01,"X02":x02,"X03":x03,"X04":x04}


def authoring_functions(case_id):
    if case_id=="R01":
        return [ref.ffn_partition,ref.split_plan,ref.ffn_spec,ref.transfers,ref.dense,ref.hardware]
    if case_id in ("R03","R04"):
        return [RUNNERS[case_id],serving._run,serving._build,serving._declarations,
                serving._weight_inventory,serving.DeclaredSyntheticCosts,ref.hardware]
    if case_id=="X01":
        return [x01,ref.attention_spec,ref.ffn_spec,ref.transfers,ref.dense,ref.hardware]
    if case_id=="X04":
        return [x04,ref.ffn_spec,ref.transfers,ref.hardware]
    return [RUNNERS[case_id],build_serving,blackout_events,serving._declarations,serving._weight_inventory,
            serving.DeclaredSyntheticCosts,ref.hardware]


def run(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty; evidence cannot be overwritten")
    output.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema="paper-expression-study-1",declared_cases=CASES,
        denominators=dict(reference_rerun=4,new_composition=4,total=8),
        definition_of_new="Newly authored combinations of existing mechanisms; no backend adaptation, not blind-heldout or novel scheduling algorithms.",
        hardware_accuracy_claimed=False,external_tool_comparison="not yet performed")
    ref.write_json(output / "manifest.json",manifest)
    before=freeze_sources()
    ref.write_json(output / "source-before.json",before)
    for case in CASES:
        folder=output / case["id"]
        folder.mkdir()
        ref.write_json(folder / "preregistered-obligations.json",case)
        ref.write_json(folder / "authoring.json",write_authoring(folder,authoring_functions(case["id"])))
    rows=[]
    for case in CASES:
        folder=output / case["id"]
        started=time.monotonic()
        row=dict(case)
        try:
            row.update(status="pass",evidence=RUNNERS[case["id"]](folder))
        except Exception as error:
            row.update(status="fail",error=repr(error),traceback=traceback.format_exc())
        row["wall_seconds"]=time.monotonic()-started
        row["authoring"]=json.loads((folder / "authoring.json").read_text(encoding="utf-8"))
        ref.write_json(folder / "result.json",row)
        rows.append(row)
        print(case["id"],row["status"],f"{row['wall_seconds']:.2f}s",row.get("error",""),flush=True)
    after=freeze_sources()
    modules=ref.imported_sources()
    ref.write_json(output / "source-after.json",after)
    ref.write_json(output / "imported-modules.json",modules)
    integrity_error=None
    try:
        ref.verify_source_integrity(before,after,modules)
    except Exception as error:
        integrity_error=repr(error)
    summary=dict(schema="paper-expression-results-1",created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(output / "manifest.json"),source_unchanged=before==after,
        source_integrity_error=integrity_error,source_file_count=len(before),imported_module_count=len(modules),
        counts={group:dict(passed=sum(r["status"]=="pass" and r["group"]==group for r in rows),
                          total=sum(r["group"]==group for r in rows)) for group in ("reference_rerun","new_composition")},
        total=dict(passed=sum(r["status"]=="pass" for r in rows),total=len(rows)),
        cases=rows,raw_directory=str(output),
        limitations=["Finite small synthetic semantic references, not all-strategy completeness or numerical tensor correctness.",
            "No hardware timing claims; all prices are declared synthetic.",
            "New compositions were specified by experiment authors; not blind heldout generalization.",
            "Existing reference recipes and independent geometry/serving checkers are reused and source-bound.",
            "No external-tool benchmark, matched low-level authoring baseline or human productivity study.",
            "Full helper-source counts include workload and oracle code; report complexity rather than claimed usability speedup.",
            "Blackout witnesses test dependency/resource composition, not calibrated real OS interference."],
        complete=all(r["status"]=="pass" for r in rows) and integrity_error is None)
    ref.write_json(output / "summary.json",summary)
    hashes={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    ref.write_json(output / "artifact-hashes.json",hashes)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.out)
    return 0 if result["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
