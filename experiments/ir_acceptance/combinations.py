"""Frozen C01--C12 combination probes; no production changes or timing fitting.

Expected semantics are written before compilation/execution.  The small geometry
oracle is reused transparently from experiments.strategy_fidelity; prices are
declared synthetic inputs and are not hardware accuracy evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import traceback

from experiments import strategy_fidelity as ref
from llm_simulator import (AnalyticalCosts, Cache, DataPreparation, Model, Prefetch,
                           Scheduling, Select, Simulator, Transport, Workload,
                           InvalidInput, Unsupported, ModelPlan)
from llm_simulator.simulation import CompiledSimulation
from tensor_mapping.composition_ir import compile_composition_document
from tensor_mapping.storage_rules import ScopedTransportRule, TensorSelect
from tensor_mapping.unified_strategy import (compile_unified_strategy_file,
    run_unified_strategy, write_unified_result)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SHA256 = "cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882"


def _write(path, doc):
    ref.write_json(path, doc)


def _check_budget(result, budget):
    peak = 0
    for record in result.raw[0].decisions:
        amount = sum(t.storage_bytes for t in record.observation.tensors
                     if t.role == "weight" and t.status_at("vram") != "ABSENT")
        peak = max(peak, amount)
        ref.require(amount <= budget, "single shared weight cache budget exceeded")
        for memory in record.observation.memory:
            if memory.capacity_bytes is not None:
                ref.require(memory.used_bytes <= memory.capacity_bytes,
                            "physical capacity exceeded")
    return peak


def _write_compile(compiled, directory):
    _write(directory / "compiled-ir.json", compiled.ir_document())


def _raw_dependencies(run,session):
    observation=session.observe()
    ops={o.id:o for o in observation.operations};tensors={t.id:t for t in observation.tensors}
    events=run.evaluation.events
    computes={e.target_id:e for e in events if e.kind=="COMPUTE"}
    producers={t:op.id for op in ops.values() for t in op.outputs}
    checked=0
    for op in ops.values():
        if op.id not in computes:continue
        for tid in op.inputs:
            if tid in producers and producers[tid] in computes:
                ref.require(computes[op.id].start_ns>=computes[producers[tid]].end_ns,"consumer began before producer")
                checked+=1
    for e in events:
        if e.kind=="COPY":ref.equal(e.bytes_transferred,tensors[e.target_id].storage_bytes,"charged COPY payload")
        if e.kind=="EVICT":
            for oid,c in computes.items():
                if c.start_ns<=e.start_ns<c.end_ns:
                    domain="dram" if c.device=="cpu" else "vram"
                    ref.require(e.location!=domain or e.target_id not in ops[oid].inputs,"in-flight input evicted")
    return checked


def _split_case(directory, *, attention, startup):
    expected = dict(hidden=8, intermediate=16, cuts=[0, 4, 16],
        leaf_bytes=dict(host=128, accelerator=384), total_leaf_bytes=1536,
        shard_intermediate_bytes=dict(host=16,accelerator=48),shard_partial_bytes=32,
        shard_devices=dict(host="cpu",accelerator="gpu"),
        unsharded_devices=dict(ffn_norm="gpu",ffn_sum="gpu",ffn_residual="gpu"),
        attention_devices=dict(projection="gpu",attention_qk="cpu",attention_softmax="cpu",attention_pv="gpu",attention_output="gpu") if attention else None,
        cpu_partial_bytes=32, merge_arity=2, cache_budget_bytes=4096,
        ffn_operands=ref.ffn_wiring(), attention_operands=ref.attention_wiring() if attention else None,
        startup_weights=[f"w:layer0:W_{stage}:ffn:accelerator" for stage in ("gate", "up", "down")] if startup else [],
        startup_bytes=1152 if startup else 0,
        source="handwritten geometry/algebra in experiments/strategy_fidelity.py")
    _write(directory / "expected.json", expected)
    with ref.split_plan(directory, attention=attention, startup=startup) as compiled:
        _write_compile(compiled, directory)
        result = ref.completed(compiled, directory)
        operations, tensors, events = ref.observed(result)
        n = ref.check_expected_wiring(operations, expected["ffn_operands"])
        if attention:
            n += ref.check_expected_wiring(operations, expected["attention_operands"])
            for suffix, size in (("proj.q",32),("proj.k",16),("proj.v",16),
                                 ("attn.scores",8),("attn.probabilities",8),("attn.context",32)):
                ref.equal(ref.one_suffix(tensors,suffix).storage_bytes,size,suffix)
            ref.check_copy(result,ref.tensor_id("proj.q"),"vram","dram",32)
            ref.check_copy(result,ref.tensor_id("proj.k"),"vram","dram",16)
            moved=ref.check_copy(result,ref.tensor_id("attn.probabilities"),"dram","vram",8)
            pv=ref.one_kind(operations,"attention_pv")
            ref.require(events[pv.id].start_ns >= moved.end_ns,"PV starts before probabilities arrive")
        total=0
        for stage in ("gate","up","down"):
            ref.require(f"w:layer0:W_{stage}" not in tensors,"parent matrix retained alongside materialized shards")
            for shard,size in expected["leaf_bytes"].items():
                ref.equal(tensors[f"w:layer0:W_{stage}:ffn:{shard}"].storage_bytes,size,"partitioned weight bytes")
                total+=size
        ref.equal(total,1536,"partition conserves matrix bytes")
        checked_devices={}
        for shard,width_bytes in expected["shard_intermediate_bytes"].items():
            for stage in ("gate","up","swiglu"):
                suffix=f"ffn.{shard}.{stage}"
                tensor=ref.one_suffix(tensors,suffix)
                ref.equal(tensor.storage_bytes,width_bytes,f"{shard} {stage} activation bytes")
                producer=next(op for op in operations.values() if tensor.id in op.outputs)
                ref.equal(events[producer.id].device,expected["shard_devices"][shard],f"{shard} {stage} execution device")
                checked_devices[producer.id]=events[producer.id].device
            partial_tensor=ref.one_suffix(tensors,f"ffn.{shard}.partial")
            ref.equal(partial_tensor.storage_bytes,32,f"{shard} partial output bytes")
            down=next(op for op in operations.values() if partial_tensor.id in op.outputs)
            ref.equal(events[down.id].device,expected["shard_devices"][shard],f"{shard} down execution device")
            checked_devices[down.id]=events[down.id].device
        for kind,device in expected["unsharded_devices"].items():
            op=ref.one_kind(operations,kind)
            ref.equal(events[op.id].device,device,f"{kind} execution device")
            checked_devices[op.id]=device
        if attention:
            for kind,device in expected["attention_devices"].items():
                op=ref.one_kind(operations,kind)
                ref.equal(events[op.id].device,device,f"{kind} execution device")
                checked_devices[op.id]=device
        merge=ref.one_kind(operations,"ffn_sum")
        ref.equal(len(merge.inputs),2,"merge arity")
        partial=ref.check_copy(result,ref.tensor_id("ffn.ffn.host.partial"),"dram","vram",32)
        ref.require(events[merge.id].start_ns >= partial.end_ns,"merge waits for CPU partial")
        if startup:
            first=min(e.start_ns for e in result.events if e.kind=="COMPUTE")
            for tid in expected["startup_weights"]:
                copied=ref.check_copy(result,tid,"dram","vram",384)
                ref.require(copied.end_ns<=first,"startup shard weights not prepared before compute")
            ref.equal(len(result.raw[2].audit_document()["explicit_data_preparations"]),3,"startup audit count")
        return dict(operand_contracts=n,checked_compute_devices=checked_devices,dependency_edges=ref.check_dependency_lifetimes(result),
                    peak_weight_bytes=_check_budget(result,4096),summary=result.summary())


def c01(directory): return _split_case(directory, attention=True, startup=False)
def c02(directory): return _split_case(directory, attention=False, startup=True)
def c11(directory): return _split_case(directory, attention=True, startup=True)


def _early_prefetch(result):
    ops, tensors, computes=ref.observed(result)
    audits=[row for row in result.raw[2].audit_document()["decisions"]
            if row["reason"]=="cross_layer_public_history_prefetch"]
    ref.require(audits,"prefetch enabled but no charged speculative action")
    copies=[e for e in result.events if e.kind=="COPY" and e.target_id.startswith("w:layer1:")
            and e.src=="dram" and e.dst=="vram"]
    layer0=[computes[op.id] for op in ops.values() if op.layer_id=="layer0"]
    last=max(e.end_ns for e in layer0)
    # Match the audit decision to a concrete COPY. Merely finding a demand copy
    # before a later round's layer0 work would not prove cross-layer prefetch.
    matched=[]
    for row in audits:
        d=row.get("evidence",{})
        for e in copies:
            if e.start_ns==row["t_ns"] and e.target_id==d.get("tensor"):
                active=[x for x in layer0 if x.start_ns<=e.start_ns<x.end_ns]
                if active:
                    ref.require(e.bytes_transferred==tensors[e.target_id].storage_bytes,"prefetch payload uncharged")
                    matched.append(dict(target=e.target_id,start_ns=e.start_ns,end_ns=e.end_ns,
                                        bytes=e.bytes_transferred,overlapped_layer0=[x.target_id for x in active]))
    ref.require(matched,"no audited next-layer COPY overlaps active current-layer computation")
    return matched


def c03(directory):
    _write(directory/"expected.json",dict(layers=2,layer0_ffn="cpu",layer1_ffn="gpu",budget=4096,
        prefetch="charged layer1 weight COPY overlapping active layer0 compute",source="generated analytical F32 H8 I16"))
    def configure(plan): plan.place("dense_ffn","cpu",select=Select(layer_id="layer0"))
    with ref.dense(layers=2,decode=1).compile(directory=directory/"inputs",configure=configure,
            scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=4096),
                prefetch=Prefetch(next_layers=1,max_count=2,include_dense_weights=True))) as compiled:
        _write_compile(compiled,directory); result=ref.completed(compiled,directory)
        ops,_,computes=ref.observed(result)
        for op in ops.values():
            if op.kind=="dense_ffn": ref.equal(computes[op.id].device,"cpu" if op.layer_id=="layer0" else "gpu","layer placement")
        return dict(prefetch=_early_prefetch(result),peak_weight_bytes=_check_budget(result,4096),
                    dependencies=ref.check_dependency_lifetimes(result),summary=result.summary())


def c04(directory):
    return ref.phase_kv_persistence(directory)


def c05(directory):
    source=ROOT/"mapping/examples/composition/block.strategy.json"
    _write(directory/"expected.json",dict(source=str(source.relative_to(ROOT)),
        layer_weights=dict(L0=64,L1=64),cache_budget_bytes=96,
        engines={"w:L0":"ce","w:L1":"sm"},required_eviction="w:L0",
        include_inflight_reservations=True,source_note="existing synthetic block fixture; explicit services"))
    document=json.loads(source.read_text(encoding="utf-8")); document["cache"]["budget_bytes"]=96
    _write(directory/"strategy-input.json",document)
    rules=(ScopedTransportRule("weight","dram","vram","sm",TensorSelect(layer_id="L1")),)
    compiled=compile_composition_document(document,base_dir=source.parent,scoped_transport=rules)
    raw=run_unified_strategy(compiled); run,session,policy,_=raw
    write_unified_result(directory/"execution",compiled,raw)
    ref.equal(run.evaluation.status,"valid","block execution")
    ref.equal(run.termination_reason,"goal_reached","block inference goal reached")
    obs=[r.observation for r in run.decisions]+[session.observe()]
    peak=0
    for o in obs:
        amount=sum(t.storage_bytes for t in o.tensors if t.role=="weight" and t.status_at("vram")!="ABSENT")
        peak=max(peak,amount); ref.require(amount<=96,"scopes duplicated shared budget")
    # Payload events are serialized by write_unified_result; use the same
    # session public event stream exposed by the runtime adapters.
    events=run.evaluation.events
    chosen=[e for e in events if e.kind=="COPY" and e.target_id in ("w:L0","w:L1") and e.dst=="vram"]
    engine_entries=[e for e in session.service_document()["transfer_engines"] if e["route"]=="dram_to_vram" and e["tensor_id"] in ("w:L0","w:L1")]
    ref.equal({(e["tensor_id"],e["engine"]) for e in engine_entries},{("w:L0","ce"),("w:L1","sm")},"scoped engines")
    for row in engine_entries:
        event=events[row["event_index"]]
        ref.equal(event.target_id,row["tensor_id"],"engine ledger links actual COPY")
        ref.equal(event.kind,"COPY","engine ledger event kind")
        ref.equal(event.resource,"pcie_"+row["engine"],"engine ledger physical COPY resource")
    for e in chosen: ref.equal(e.bytes_transferred,64,"weight copy bytes")
    evicted=[r for r in run.decisions if r.action.kind=="EVICT" and r.action.target_id=="w:L0"]
    ref.require(evicted,"undersized cache did not safely evict layer0 weight")
    for row in evicted:
        ref.require(not any(row.action.target_id in op.inputs and any(x.target_id==op.id for x in row.observation.running)
                            for op in row.observation.operations),"in-flight input eviction")
    return dict(peak_weight_bytes=peak,copy_count=len(chosen),evictions=len(evicted),
                dependency_edges=_raw_dependencies(run,session),audit=policy.audit_document())


def c06(directory):
    expert_devices={"layer0-prefill":"gpu","layer0-decode0":"cpu","layer0-decode1":"gpu"}
    positions={"layer0-prefill":0,"layer0-decode0":1,"layer0-decode1":2}
    _write(directory/"expected.json",dict(routes=[0,"private_future_1_or_2",0],invocation_expert_devices=expert_devices,
        expert_cache_unit="expert_group",cache_budget_bytes=4096,
        causal_pair="unrevealed future expert1 versus expert2; identical public prefixes",
        cached_weight_ids=["w:layer0:E0"],cached_expert_payload_bytes=1536,
        cache_reuse="packed expert0 payload (3 HxI F32 matrices) copied to GPU exactly once, retained until third invocation reuses the same identity without reloading",
        source="generated trace-driven MoE, H8 I16 experts3 topk1"))
    def configure(plan):
        for invocation,device in expert_devices.items():
            plan.place("expert_ffn",device,select=Select(invocation_id=invocation))
    prefixes=[]; summaries=[]
    for future in (1,2):
        sub=directory/f"future-{future}"
        simulator=Simulator(model=Model.moe(layers=1,hidden_size=8,intermediate_size=16,
            heads=2,kv_heads=1,experts=3,top_k=1),hardware=ref.hardware(),
            workload=Workload(1,decode_steps=2,routes=(ref.Route("request0","layer0",0,(0,)),
                ref.Route("request0","layer0",1,(future,)),ref.Route("request0","layer0",2,(0,))),
                route_source="C06-explicit-causal-pair-with-expert0-reuse"),cost_model=AnalyticalCosts())
        with simulator.compile(directory=sub/"inputs",configure=configure,
            scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",unit="expert_group",budget_bytes=4096))) as compiled:
            _write_compile(compiled,sub); result=ref.completed(compiled,sub)
            ops,tensors,events=ref.observed(result)
            experts=[op for op in ops.values() if op.kind=="expert_ffn"]
            ref.equal(len(experts),3,"three revealed expert tasks")
            for op in experts:
                ref.equal(events[op.id].device,expert_devices[op.invocation_id],"heterogeneous expert placement")
                for tid in op.inputs:
                    if tensors[tid].role=="weight" and op.phase=="prefill":
                        found=[e for e in result.events if e.kind=="COPY" and e.target_id==tid and e.dst=="vram"]
                        ref.require(found,"GPU expert weight never loaded")
                        ref.equal(found[0].bytes_transferred,tensors[tid].storage_bytes,"expert payload")
            first=next(op for op in experts if op.invocation_id=="layer0-prefill")
            repeated=next(op for op in experts if op.invocation_id=="layer0-decode1")
            ref.equal(first.expert_id,0,"first expert identity")
            ref.equal(repeated.expert_id,0,"repeated expert identity")
            first_weights=tuple(tid for tid in first.inputs if tensors[tid].role=="weight")
            reused_weights=tuple(tid for tid in repeated.inputs if tensors[tid].role=="weight")
            ref.equal(first_weights,("w:layer0:E0",),"handwritten physical expert payload identity")
            ref.equal(reused_weights,first_weights,"expert cache retains physical identities across invocations")
            ref.require(reused_weights,"empty expert cache reuse oracle")
            for tid in reused_weights:
                ref.equal(tensors[tid].storage_bytes,1536,"packed expert payload is 3*8*16*4 bytes")
                copies=[e for e in result.events if e.kind=="COPY" and e.target_id==tid and e.dst=="vram"]
                ref.equal(len(copies),1,"retained GPU expert weight loaded exactly once")
                ref.require(copies[0].end_ns<=events[first.id].start_ns,"initial cached weight arrived too late")
                ref.require(not any(e.kind=="EVICT" and e.target_id==tid and e.location=="vram"
                    and events[first.id].start_ns<=e.start_ns<=events[repeated.id].end_ns for e in result.events),
                    "GPU expert cache discarded weight before repeated use")
            repeated_decision=next(record for record in result.raw[0].decisions if record.action.kind=="COMPUTE" and record.action.target_id==repeated.id)
            repeated_tensors={t.id:t for t in repeated_decision.observation.tensors}
            for tid in reused_weights:ref.equal(repeated_tensors[tid].status_at("vram"),"READY","cached expert ready at repeated GPU COMPUTE")
            prefix=[]
            for record in result.raw[0].decisions:
                # Validate all public expert tasks, not only the compared prefix.
                for op in record.observation.operations:
                    if op.kind=="expert_ffn":
                        ref.require(any(route.absolute_position==positions[op.invocation_id]
                            and op.expert_id in route.expert_ids for route in record.observation.route_history),
                            "private expert task exposed before its invocation route")
                if record.action.kind=="COMPUTE":
                    op=next(o for o in record.observation.operations if o.id==record.action.target_id)
                    if op.kind=="expert_ffn":
                        ref.require(any(r.absolute_position==positions[op.invocation_id] and op.expert_id in r.expert_ids
                            for r in record.observation.route_history),"expert compute before reveal")
                if not any(route.absolute_position==1 for route in record.observation.route_history):
                    prefix.append(dict(observation=asdict(record.observation),action=asdict(record.action)))
            prefixes.append(prefix)
            summaries.append(dict(reused_gpu_expert_weight_ids=reused_weights,weight_reload_count=0,
                peak_weight_bytes=_check_budget(result,4096),dependencies=ref.check_dependency_lifetimes(result),summary=result.summary()))
    ref.equal(prefixes[0],prefixes[1],"private future changes earlier decisions")
    return dict(runs=summaries,identical_prefix_decisions=len(prefixes[0]))


def _local_attempt(directory, family):
    expectation=dict(batch_size=2,invocation="layer0-prefill",local_requests=["request0","request1"],
        required="only request0 attention refined" if family=="attention" else "request0 split4/12 and request1 split12/4 in SAME invocation",
        forbidden_substitute="whole-invocation refinement or separate invocation requests",
        source="public module request scope + common region execution pipeline",
        storage_contract="shared_parent_slice_views" if family=="ffn" else "unchanged weights and KV")
    _write(directory/"expected.json",expectation)
    attempts=[]
    for i in range(1 if family=="attention" else 2):
        spec=ref.attention_spec() if family=="attention" else ref.ffn_spec()
        for row in spec["entries"]:row["batch"]=2
        if family=="ffn":spec["weight_storage_contract"]="shared_parent_slice_views"
        if i==1:
            spec["shards"][0]["end"]=12;spec["shards"][1]["start"]=12
        attempts.append(dict(request_id=f"request{i}",spec=spec))
    _write(directory/"attempted-declarations.json",attempts)
    def configure(plan):
        if family=="attention":
            for kind in ("projection","attention","dense_ffn","attention_qk",
                         "attention_softmax","attention_pv","attention_output"):
                plan.place(kind,"gpu",select=Select(phase="prefill"))
            for kind in ("projection","attention","dense_ffn"):
                plan.place(kind,"cpu",select=Select(phase="decode"))
        for i,attempt in enumerate(attempts):
            spec=attempt["spec"]
            with plan.module(f"selected-request{i}",request_id=f"request{i}"):
                plan.transform("layer0-prefill",spec)
    plan,generated=_explicit_batch2_plan(directory,layers=1,decode=1 if family=="attention" else 0,
        scheduling=Scheduling(default_device="gpu"))
    configure(plan)
    compiled=CompiledSimulation(plan.compile(),directory/"inputs",dict(generated.get("metadata",{}),cost_model=generated["cost_model"]),None)
    with compiled:
        _write_compile(compiled,directory);result=ref.completed(compiled,directory)
        return (_check_local_attention(result) if family=="attention" else _check_local_ffns(result))


def _check_local_attention(result):
    """Handwritten local IO/KV oracle; no expected values from the compiler."""
    ops,tensors,events=ref.observed(result)
    wiring=ref.attention_wiring()
    ref.check_expected_wiring(ops,wiring)
    expected_stages={"projection","attention_qk","attention_softmax","attention_pv","attention_output","dense_ffn"}
    selected=[o for o in ops.values() if o.invocation_id=="layer0-prefill" and o.request_id=="request0"]
    ref.equal({o.kind for o in selected},expected_stages,"only selected prefill request refined")
    ref.equal(len(selected),6,"selected operation count")
    unchanged=[o for o in ops.values() if o.invocation_id=="layer0-prefill" and o.request_id=="request1"]
    ref.equal({o.kind for o in unchanged},{"projection","attention","dense_ffn"},"unselected request stays macro graph")
    ref.equal(len(unchanged),3,"unselected operation count")
    def u(suffix):return "g:layer0-prefill:t:layer0:request1:"+suffix
    unselected_wiring={
        "projection":dict(inputs=["w:layer0:W_input_norm","w:layer0:W_q","w:layer0:W_k","w:layer0:W_v",u("in.hidden"),u("in.positions")],
                          outputs=[u("proj.q"),u("proj.k"),u("proj.v")]),
        "attention":dict(inputs=["w:layer0:W_output",u("proj.q"),u("proj.k"),u("proj.v"),u("in.hidden")],
                         outputs=[u("attn"),"kv:layer0:request1:0"]),
        "dense_ffn":dict(inputs=["w:layer0:W_post_attention_norm","w:layer0:W_gate","w:layer0:W_up","w:layer0:W_down",u("attn")],outputs=[u("ffn")])}
    ref.check_expected_wiring(ops,unselected_wiring)
    ref.require(not any(":request1:attn.scores" in tid or ":request1:attn.probabilities" in tid
                        or ":request1:attn.context" in tid for tid in tensors),"unselected request received derived Attention allocations")
    for suffix,size in (("proj.q",32),("proj.k",16),("proj.v",16),("attn.scores",8),
                       ("attn.probabilities",8),("attn.context",32),("attn",32),("ffn",32)):
        ref.equal(tensors[ref.tensor_id(suffix)].storage_bytes,size,"selected IO bytes "+suffix)
    for request in ("request0","request1"):
        for position in (0,1):
            tid=f"kv:layer0:{request}:{position}"
            ref.equal(tensors[tid].storage_bytes,32,"one immutable logical KV block")
            producers=[o for o in ops.values() if tid in o.outputs]
            ref.equal(len(producers),1,"KV block generated exactly once")
        ref.check_copy(result,f"kv:layer0:{request}:0","vram","dram",32)
    for op in ops.values():
        if op.kind=="state_update":continue
        ref.equal(events[op.id].device,"gpu" if op.phase=="prefill" else "cpu","phase placement survives local refinement")
    for request in ("request0","request1"):
        decode=[o for o in ops.values() if o.invocation_id=="layer0-decode0" and o.request_id==request and o.kind!="state_update"]
        ref.equal({o.kind for o in decode},{"projection","attention","dense_ffn"},"decode remains unrefined")
        ref.equal(len(decode),3,"decode kernel count")
        attn=next(o for o in decode if o.kind=="attention")
        ref.require(f"kv:layer0:{request}:0" in attn.inputs,"decode consumes persistent selected/unselected history")
    state=result.raw[1].persistent_state_document()
    ref.equal(set(state["kv_ids"]),{f"kv:layer0:{r}:{p}" for r in ("request0","request1") for p in (0,1)},"global KV inventory")
    return dict(selected_requests=["request0"],unchanged_requests=["request1"],
        exact_attention_operand_contracts=len(wiring),unchanged_macro_operand_contracts=len(unselected_wiring),
        kv_blocks=4,kv_bytes=128,
        dependency_edges=ref.check_dependency_lifetimes(result),summary=result.summary())


def _check_local_ffns(result):
    """Different slice views read one physical parent inventory, never shards."""
    ops,tensors,events=ref.observed(result)
    raw_ops={o["id"]:o for o in result.raw[1].execution_document()["workload"]["operations"]}
    parents=["w:layer0:W_"+s for s in ("gate","up","down")]
    for tid in parents:ref.equal(tensors[tid].storage_bytes,512,"full parent matrix payload")
    ref.equal(sum(tensors[tid].storage_bytes for tid in parents),1536,"one shared FFN parent inventory")
    ref.require(not any(t.role=="weight" and ":ffn:" in t.id for t in tensors.values()),"slice views must not allocate duplicate weights")
    ref.equal(sum(t.storage_bytes for t in tensors.values() if t.role=="weight"),2368,"complete Dense weight inventory remains single")
    checked=0;copy_ids=[];slices={}
    for request,cut in (("request0",4),("request1",12)):
        prefix=f"g:layer0-prefill:t:layer0:{request}:"
        def t(suffix):return prefix+suffix
        scoped=[o for o in ops.values() if o.invocation_id=="layer0-prefill" and o.request_id==request]
        ref.require(not any(o.kind=="dense_ffn" for o in scoped),"selected macro replaced")
        ref.equal(len([o for o in scoped if o.kind.startswith("ffn_")]),11,"one complete FFN replacement per request")
        expected={
            "ffn_norm":dict(inputs=["w:layer0:W_post_attention_norm",t("attn")],outputs=[t("ffn.ffn.norm")]),
            "ffn_sum":dict(inputs=[t("ffn.ffn.host.partial"),t("ffn.ffn.accelerator.partial")],outputs=[t("ffn.ffn.sum")]),
            "ffn_residual":dict(inputs=[t("ffn.ffn.sum"),t("attn")],outputs=[t("ffn")])}
        for shard,start,end,device in (("host",0,cut,"cpu"),("accelerator",cut,16,"gpu")):
            width=end-start;slices[request+":"+shard]=[start,end]
            q="ffn.ffn."+shard+"."
            for stage in ("gate","up"):
                expected[q+stage]=dict(kind="ffn_"+stage,inputs=["w:layer0:W_"+stage,t("ffn.ffn.norm")],outputs=[t(q+stage)])
            expected[q+"swiglu"]=dict(kind="ffn_swiglu",inputs=[t(q+"gate"),t(q+"up")],outputs=[t(q+"swiglu")])
            expected[q+"down"]=dict(kind="ffn_down",inputs=["w:layer0:W_down",t(q+"swiglu")],outputs=[t(q+"partial")])
            for stage in ("gate","up","swiglu","down"):
                output=t(q+("partial" if stage=="down" else stage))
                producer=next(o for o in scoped if output in o.outputs)
                ref.equal(raw_ops[producer.id]["op_params"],[start,end],"explicit intermediate slice")
                ref.equal(events[producer.id].device,device,"per-region shard binding")
                ref.equal(tensors[output].storage_bytes,32 if stage=="down" else width*4,"per-region activation shape")
                ref.require(events[producer.id].end_ns>events[producer.id].start_ns,"derived stage has charged duration")
        checked+=ref.check_expected_wiring(ops,expected)
        merge=next(o for o in scoped if o.kind=="ffn_sum")
        for kind in ("ffn_norm","ffn_sum","ffn_residual"):
            op=next(o for o in scoped if o.kind==kind)
            ref.equal(events[op.id].device,"gpu","common FFN stage binding")
            ref.require(events[op.id].end_ns>events[op.id].start_ns,"common FFN stage has charged duration")
        ref.equal(raw_ops[merge.id]["op_params"],[2],"merge explicitly receives both partials")
        copied=ref.check_copy(result,t("ffn.ffn.host.partial"),"dram","vram",32)
        ref.require(events[merge.id].start_ns>=copied.end_ns,"merge waits for host partial transfer")
        copy_ids.append(copied.target_id)
    parent_copies=[]
    for tid in parents:
        copied=ref.check_copy(result,tid,"dram","vram",512)
        parent_copies.append(dict(tensor_id=tid,bytes=copied.bytes_transferred))
    return dict(region_slices=slices,parent_matrix_bytes=1536,complete_weight_bytes=2368,
        operand_contracts=checked,full_parent_gpu_copies=parent_copies,host_partial_copies=copy_ids,
        dependency_edges=ref.check_dependency_lifetimes(result),peak_weight_bytes=_check_budget(result,8192),summary=result.summary())


def c08(directory): return _local_attempt(directory,"attention")
def c09(directory): return _local_attempt(directory,"ffn")


def _services(directory):
    path=directory/"services.json"
    _write(path,dict(format_version="tensor-services-0.1",pcie=dict(bandwidth_bytes_per_second=1_000_000_000,
        ce_bytes_per_second=1_000_000_000,sm_bytes_per_second=500_000_000,
        arbitration="max_min_fair",timing_scope="payload_only_no_launch_latency"),slots=[]))
    return path


def _explicit_batch2_plan(directory, *, layers, decode, scheduling):
    """Fresh explicit B2 inputs with independent per-request kernel accounting."""
    sim=Simulator(model=Model.dense(layers=layers,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
        hardware=ref.hardware(),workload=Workload(1,batch_size=2,decode_steps=decode),cost_model=AnalyticalCosts())
    generated=sim.cost_model.build_inputs(directory/"inputs",sim.model,sim.hardware,sim.workload)
    root=directory/"inputs"
    _write(directory/"benchmark-input-provenance.json",dict(
        generated_geometry="AnalyticalCosts batch2 F32 H8 I16 with independent request kernel accounting",
        batch2_costs="explicit analytical per-request prices under invocation batch=2; no measured/fitted batch scaling claim",
        production_modified=False))
    plan=ModelPlan(id="acceptance-explicit-batch2",model_kind=generated["model_kind"],
        model_file=generated["model_file"],environment_file=generated["environment_file"],
        base_dir=root,scheduling=scheduling,max_decisions=100000,visibility_mode="data_dependencies")
    return plan,generated


def c10(directory):
    _write(directory/"expected.json",dict(layers=2,phase_devices=dict(prefill="gpu",decode="cpu"),
        engines=dict(layer0="ce",layer1="sm"),prepared_layers=["layer0","layer1"],
        preparation_rule_owners={"0":"layer0","1":"layer1"},both_preparation_rules_must_issue_copy=True,
        preparation_trigger="startup",budget_bytes=8192,kv_blocks=4,kv_bytes=32,
        source="generated analytical dense with explicit payload-only shared-link contract"))
    services=_services(directory)
    def configure(plan):
        for i,engine in ((0,"ce"),(1,"sm")):
            with plan.module(f"layer{i}",layer_id=f"layer{i}"):
                plan.transfer(engine=engine)
                plan.prepare(destination="vram",trigger="startup")
        for kind in ("projection","attention","dense_ffn"):
            plan.place(kind,"cpu",select=Select(phase="decode"))
    with ref.dense(layers=2,decode=1).compile(directory=directory/"inputs",services_file=services,configure=configure,
        scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=8192),
                              transport=Transport(engine="ce"))) as compiled:
        _write_compile(compiled,directory); result=ref.completed(compiled,directory)
        ops,tensors,computes=ref.observed(result)
        for op in ops.values():
            if op.kind in ("projection","attention","dense_ffn"):
                ref.equal(computes[op.id].device,"cpu" if op.phase=="decode" else "gpu","phase devices")
        owners={row[0]:row[1] for row in result.raw[0].decisions[0].observation.weight_catalogue}
        copies=[e for e in result.events if e.kind=="COPY" and e.src=="dram" and e.dst=="vram" and e.target_id in owners]
        ref.require(copies,"no weight preparation")
        ref.equal({owners[e.target_id] for e in copies},{"layer0","layer1"},"both layers prepared")
        first=min(e.start_ns for e in result.events if e.kind=="COMPUTE")
        services=result.raw[1].service_document()
        engines={row["event_index"]:row["engine"] for row in services["transfer_engines"]}
        for e in copies:
            engine=engines[result.events.index(e)]
            ref.equal(engine,"ce" if owners[e.target_id]=="layer0" else "sm","owner-scoped engine")
            ref.equal(e.kind,"COPY","service engine links real COPY")
            ref.equal(e.resource,"pcie_"+engine,"service engine physical COPY resource")
            ref.equal(e.bytes_transferred,tensors[e.target_id].storage_bytes,"prepared weight bytes")
            ref.require(e.end_ns<=first,"startup weight preparation ends after computation begins")
        preparation_audit=result.raw[2].audit_document()["explicit_data_preparations"]
        ref.require(preparation_audit,"no explicit preparation audit records")
        rule_owners={0:"layer0",1:"layer1"};checked_preparations=[];seen_rules=set()
        for row in preparation_audit:
            ref.require(row["rule_index"] in rule_owners,"unexpected preparation rule index")
            ref.equal(owners[row["tensor_id"]],rule_owners[row["rule_index"]],"preparation rule uses its own static layer owner")
            ref.equal(row["trigger"],"startup","preparation trigger")
            ref.equal(row["destination"],"vram","preparation destination")
            ref.equal(row["action"]["kind"],"COPY","preparation issued a real copy rather than just enabled")
            ref.equal(row["action"]["target_id"],row["tensor_id"],"preparation audit/action identity")
            matches=[e for e in result.events if e.kind=="COPY" and e.target_id==row["tensor_id"]
                and e.start_ns==row["t_ns"] and e.dst=="vram"]
            ref.equal(len(matches),1,"preparation audit matches one concrete COPY")
            seen_rules.add(row["rule_index"])
            checked_preparations.append(dict(rule_index=row["rule_index"],tensor_id=row["tensor_id"],
                static_layer_owner=owners[row["tensor_id"]],event_index=result.events.index(matches[0])))
        ref.equal(seen_rules,{0,1},"both scoped preparation rules produced actual COPY actions")
        for layer in range(2):
            tid=f"kv:layer{layer}:request0:0"
            found=[e for e in result.events if e.kind=="COPY" and e.target_id==tid and e.src=="vram" and e.dst=="dram"]
            ref.equal(len(found),1,"persistent KV migrated once")
            ref.equal(found[0].bytes_transferred,32,"KV migration bytes")
        return dict(weight_copies=len(copies),checked_preparation_rules=checked_preparations,
                    peak_weight_bytes=_check_budget(result,8192),
                    dependencies=ref.check_dependency_lifetimes(result),summary=result.summary())


def c12(directory):
    source=ROOT/"mapping/examples/reconfiguration/global.strategy.json"
    _write(directory/"expected.json",dict(source=str(source.relative_to(ROOT)),
        request_id="global-external-app",notice_ns=3000,deadline_ns=5500,target_external_bytes=1046000,
        required=["actual cross-layer speculative COPY","safe weight EVICT while reclaim pending",
                  "DECIDE cost 2 ns for reclaim","ACK cost 3 ns only after release","accepted request"],
        scope="existing synthetic global Dense fixture, exact prices, dedicated serial control resource"))
    compiled=compile_unified_strategy_file(source)
    raw=run_unified_strategy(compiled);run,session,policy,_=raw
    write_unified_result(directory/"execution",compiled,raw)
    ref.equal(run.evaluation.status,"valid","reclaim inference completion")
    ref.equal(run.termination_reason,"goal_reached","reclaim inference goal reached")
    audit=policy.audit_document()
    prefetch=[r for r in audit["decisions"] if r["reason"]=="cross_layer_public_history_prefetch"]
    ref.require(prefetch,"prefetch did not produce decisions")
    events=run.evaluation.events
    matched=[]
    for row in prefetch:
        d=row.get("evidence",{})
        found=[e for e in events if e.kind=="COPY" and e.target_id==d.get("tensor") and e.start_ns==row["t_ns"]]
        ref.require(found,"prefetch has no real COPY")
        ref.require(found[0].bytes_transferred>0,"prefetch COPY free/empty")
        matched.append(asdict(found[0]))
    releases=[r for r in run.decisions if r.observation.pending_reclaim and r.action.kind=="EVICT"]
    ref.require(releases,"no actual reclaim release")
    for row in releases:
        ref.require(row.action.target_id in row.observation.safe_weight_release_candidates,"unsafe reclaim target")
    records=run.runtime_result.reclaim_records
    ref.equal([(r.id,r.status) for r in records],[("global-external-app","accepted")],"reclaim accepted")
    controls=run.runtime_result.control_events
    decision=[e for e in controls if e.kind=="DECIDE" and e.request_id=="global-external-app"]
    ack=[e for e in controls if e.kind=="ACK_RECLAIM"]
    ref.equal(len(decision),1,"reclaim decision count");ref.equal(decision[0].cost_ns,2,"reclaim decision charged")
    ref.equal(len(ack),1,"ACK count");ref.equal(ack[0].cost_ns,3,"ACK charged")
    ref.require(ack[0].start_ns>=max(r.observation.t_ns for r in releases),"ACK before safe release")
    ref.require(records[0].responded_ns<=5500,"reclaim deadline missed")
    for row in run.decisions:
        ref.require(sum(t.storage_bytes for t in row.observation.tensors if t.role=="weight" and t.status_at("vram")!="ABSENT")<=200000,"shared cache budget exceeded")
        for m in row.observation.memory:
            if m.capacity_bytes is not None:ref.require(m.used_bytes<=m.capacity_bytes,"memory capacity exceeded")
    # Same published history/reclaim notice, different private future recovery.
    # This comparison is independent of the controller implementation.
    document=json.loads(source.read_text(encoding="utf-8"))
    for field in ("services_file","environment_file","history_file","routing_scores_file"):
        if document[field] is not None:document[field]=str((source.parent/document[field]).resolve())
    document["model"]["file"]=str((source.parent/document["model"]["file"]).resolve())
    for row in document["execution"]:
        if "invocation_files" in row["spec"]:
            row["spec"]["invocation_files"]=[str((source.parent/p).resolve()) for p in row["spec"]["invocation_files"]]
    environment=json.loads(Path(document["environment_file"]).read_text(encoding="utf-8"))
    for event in environment["events"]:
        if "vram_external_bytes" in event:event["t_ns"]=8000
    alternative=directory/"alternative-recovery";alternative.mkdir(parents=True,exist_ok=True)
    environment_file=(alternative/"environment.json").resolve();_write(environment_file,environment)
    document["environment_file"]=str(environment_file)
    strategy=(alternative/"strategy.json").resolve();_write(strategy,document)
    other_compiled=compile_unified_strategy_file(strategy);other_raw=run_unified_strategy(other_compiled)
    write_unified_result(alternative/"execution",other_compiled,other_raw)
    ref.equal(other_raw[0].evaluation.status,"valid","alternative restoration completes")
    ref.equal(other_raw[0].termination_reason,"goal_reached","alternative restoration inference goal reached")
    prefix=lambda r:[dict(observation=asdict(x.observation),decision=asdict(x.decision)) for x in r.decisions if x.observation.t_ns<6500]
    first_prefix,second_prefix=prefix(run),prefix(other_raw[0])
    ref.require(first_prefix,"empty memory-response causal prefix")
    ref.equal(first_prefix,second_prefix,"private future restoration changed earlier memory decisions")
    return dict(charged_prefetch_copies=matched,reclaim_evictions=len(releases),
                reclaim_records=[asdict(r) for r in records],control_events=[asdict(e) for e in controls],
                identical_private_recovery_prefix_decisions=len(first_prefix),
                dependency_edges=_raw_dependencies(run,session))


def c07(directory):
    # One common declaration, graph compiler and runtime: neither a legacy
    # paper adapter nor independent per-subbatch sessions can satisfy C07.
    from experiments.ir_acceptance.basic import _pipeline_public_attempt
    evidence=_pipeline_public_attempt(directory,layers=2,prefetch=True)
    ref.require(evidence["unified_frontend_supported"],"pipeline bypassed the common frontend")
    ref.require(evidence["single_runtime"],"pipeline duplicated the resource ledger")
    return evidence


CASES={f"C{i:02d}":globals()[f"c{i:02d}"] for i in range(1,13)}


def run_group(output: Path) -> list[dict]:
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=Path(__file__).with_name("manifest.json")
    ref.equal(hashlib.sha256(manifest.read_bytes()).hexdigest(),MANIFEST_SHA256,"frozen acceptance manifest")
    rows=[]
    for identifier,case in CASES.items():
        directory=output/identifier;directory.mkdir(parents=True,exist_ok=True)
        try:
            evidence=case(directory)
            status=evidence.pop("_status","pass");expression=evidence.pop("_expression","complete")
            fidelity=evidence.pop("_fidelity","pass");reason=evidence.pop("reason","all frozen combination obligations checked")
        except Exception as error:
            evidence=dict(exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            status="fail";expression="unverified";fidelity="fail";reason=str(error)
        _write(directory/"observed.json",evidence)
        row=dict(id=identifier,status=status,expression_status=expression,fidelity_status=fidelity,
                 reason=reason,evidence=evidence,
                 artifacts=[str(p.resolve()) for p in sorted(directory.rglob("*")) if p.is_file()])
        rows.append(row);_write(output/"results.json",rows)
    return rows


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    rows=run_group(args.out)
    print(json.dumps([dict(id=r["id"],status=r["status"],reason=r["reason"]) for r in rows],indent=2))
