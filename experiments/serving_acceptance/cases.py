"""Adapters for the preregistered finite serving extension.

The common public builder/IR/compiler/runtime is the only executor. Explicit
synthetic cost overrides expose scheduling witnesses without hardware fitting.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from experiments import strategy_fidelity as ref
from experiments.serving_acceptance import oracles as check
from llm_simulator import (AnalyticalCosts,Cache,Decision,InvalidInput,Model,ModelPlan,
    PhasePlacement,Prefetch,Route,Scheduling,Select,Simulator,Unsupported,Workload)
from llm_simulator.simulation import CompiledSimulation
from tensor_mapping.causal import compile_command
from tensor_mapping.unified_strategy import create_session
from llm_infer_model.tensor.runtime import dispatch_command


class DeclaredSyntheticCosts(AnalyticalCosts):
    """Exact hypothetical rows, not measured or fitted kernel timings."""
    def _cost_document(self,*args,**kwargs):
        document,details=super()._cost_document(*args,**kwargs)
        for row in [*document["entries"],*details]:
            if row["kind"]=="attention" and row["device"]=="cpu":row["duration_ns"]=50_000
            if row["kind"] in {"dense_ffn","expert_ffn"}:
                row["duration_ns"]=50_000 if row["device"]=="gpu" else 100_000
        document["source"]+="; explicit serving witness prices CPU attention 50000ns, FFN CPU100000/GPU50000ns"
        return document,details


def _weight_inventory(kind,layers):
    """Handwritten F32 H8/I16/GQA-2:1 geometry, three experts for MoE."""
    per_layer=({"W_input_norm":32,"W_q":256,"W_k":128,"W_v":128,"W_output":256,
                "W_post_attention_norm":32,"W_gate":512,"W_up":512,"W_down":512} if kind=="dense" else
               {"W_proj":768,"W_router":96,"E0":1536,"E1":1536,"E2":1536})
    return {f"w:layer{layer}:{name}":size for layer in range(layers) for name,size in per_layer.items()}


def _declarations(*,kind="dense",arrival=2_000,prefill=1,decode=(1,2),future_expert=0,future_prefill=None):
    requests={f"request{i}":dict(arrival_ns=0 if i==0 else arrival,
        prefill_tokens=future_prefill if i==1 and future_prefill is not None else prefill,
        decode_steps=steps,layers=2) for i,steps in enumerate(decode)}
    scopes={}
    for request,declaration in requests.items():
        for step in range(declaration["decode_steps"]+1):
            phase="prefill" if step==0 else "decode"
            label="prefill" if step==0 else f"decode{step-1}"
            for layer in range(2):
                scope=f"{request}-{label}-layer{layer}"
                scopes[scope]=dict(request_id=request,step_index=step,phase=phase,layer_id=f"layer{layer}",kind=kind)
    routes=[]
    if kind=="moe":
        for request,declaration in requests.items():
            for layer in range(2):
                request_prefill=declaration["prefill_tokens"]
                for position in range(request_prefill+declaration["decode_steps"]):
                    # A three-token prefill creates expert0 one row / expert1
                    # two rows for a meaningful token-count order oracle.
                    expert=(0,1,1)[position] if request_prefill==3 and position<3 else 0
                    if position>=request_prefill:expert=future_expert
                    routes.append(Route(request,f"layer{layer}",position,(expert,)))
    return requests,scopes,tuple(routes)


def _build(directory,*,kind="dense",arrival=2_000,prefill=1,decode=(1,2),handoff=False,
           prefetch=False,pipeline="decode_first",future_expert=0,expert_order="token_count_desc",future_prefill=None):
    from llm_simulator import Request,RequestFlow,RequestPipeline,PhaseHandoff
    requests,scopes,routes=_declarations(kind=kind,arrival=arrival,prefill=prefill,decode=decode,
        future_expert=future_expert,future_prefill=future_prefill)
    expected=dict(requests=requests,scope_map=scopes,route_rows=[asdict(r) for r in routes],
        weight_inventory=_weight_inventory(kind,2),
        explicit_prices=dict(cpu_attention_ns=50_000,cpu_ffn_ns=100_000,gpu_ffn_ns=50_000),
        batch_accounting="independent B1 request kernels; no fused dynamic batch",
        cache_budget_bytes=32768,handoff_destination="dram" if handoff else None,
        request_pipeline=pipeline,expert_order=expert_order,
        prefill_expert_order=[1,0] if kind=="moe" and prefill==3 and expert_order=="token_count_desc" else None,
        hardware_accuracy_claimed=False)
    ref.write_json(directory/"expected.json",expected)
    model=(Model.dense(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1) if kind=="dense" else
           Model.moe(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1,experts=3,top_k=1))
    flow=RequestFlow(requests=tuple(Request(request_id=r,prefill_tokens=d["prefill_tokens"],
        decode_steps=d["decode_steps"],arrival_ns=d["arrival_ns"]) for r,d in requests.items()),
        routes=routes,route_source="serving-acceptance-explicit-routes" if routes else None)
    simulator=Simulator(model=model,hardware=ref.hardware(),workload=flow,cost_model=DeclaredSyntheticCosts())
    def configure(plan):
        plan.pipeline(RequestPipeline(ready_rule=pipeline))
        if handoff:
            kinds=("projection","attention","dense_ffn") if kind=="dense" else ("projection","attention","router","combine")
            PhasePlacement(prefill={k:"gpu" for k in kinds},decode={k:"cpu" for k in kinds}).apply(plan)
            plan.handoff(PhaseHandoff(destination="dram"))
        else:
            plan.place("projection","cpu")
            plan.place("attention","cpu")
    scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=32768),
        prefetch=Prefetch(next_layers=1,max_count=3,include_dense_weights=True) if prefetch else Prefetch(),
        expert_rule="fixed" if kind=="dense" else "current_cost",expert_order=expert_order)
    compiled=simulator.compile(directory=directory/"inputs",configure=configure,scheduling=scheduling)
    ref.write_json(directory/"compiled-ir.json",compiled.ir_document())
    return compiled,expected


def _run(directory,**options):
    compiled,expected=_build(directory,**options)
    with compiled:
        result=ref.completed(compiled,directory)
        evidence=check.complete_and_capacity(result,cache_budget=expected["cache_budget_bytes"],weight_inventory=expected["weight_inventory"])
        kinds=dict(dense=("projection","attention","dense_ffn"),moe=("projection","attention","router","combine"))
        evidence["lifecycle"]=check.lifecycle(result,requests=expected["requests"],scope_map=expected["scope_map"],kinds=kinds)
        return result,expected,evidence


def s01(directory):
    _,_,evidence=_run(directory)
    return evidence


def s02(directory):
    result,expected,evidence=_run(directory)
    evidence["asynchronous_progress"]=check.asynchronous_progress(result,expected["scope_map"])
    return evidence


def s03(directory):
    result,expected,evidence=_run(directory)
    evidence["overlap"]=check.execution_overlap(result,expected["scope_map"])
    return evidence


def _batch_inputs(directory):
    simulator=Simulator(model=Model.dense(layers=1,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
        hardware=ref.hardware(),workload=Workload(1,batch_size=2),cost_model=AnalyticalCosts())
    generated=simulator.cost_model.build_inputs(directory/"inputs",simulator.model,simulator.hardware,simulator.workload)
    return generated


def s04(directory):
    generated=_batch_inputs(directory);root=directory/"inputs"
    costs=json.loads((root/"layer0-prefill.costs.json").read_text(encoding="utf-8"))
    prices={(r["kind"],r["device"]):r["duration_ns"] for r in costs["entries"]}
    ref.write_json(directory/"expected.json",dict(batch=2,kernel_batch=1,accounting="independent_per_request_kernel",
        requests=["request0","request1"],compute_instances=6,weight_inventory=_weight_inventory("dense",1),
        prices=[dict(kind=k,device=d,duration_ns=v) for (k,d),v in prices.items()]))
    for row in costs["entries"]:
        signature=row["physical_signature"]
        ref.equal(row["batch"],2,"exact invocation batch axis")
        ref.equal(signature["request_accounting"],"independent_per_request_kernel","explicit kernel accounting")
        ref.equal(signature["invocation_batch"],2,"signature invocation batch")
        ref.equal(signature["kernel_batch"],1,"signature kernel batch")
    plan=ModelPlan(id="serving-exact-batch-axis",model_kind=generated["model_kind"],model_file=generated["model_file"],
        environment_file=generated["environment_file"],base_dir=root,visibility_mode="data_dependencies",
        scheduling=Scheduling(default_device="gpu",cache=Cache(rule="lru",budget_bytes=32768)))
    with CompiledSimulation(plan.compile(),root,generated["metadata"],None) as compiled:
        ref.write_json(directory/"compiled-ir.json",compiled.ir_document())
        result=ref.completed(compiled,directory);ops,_,events=check.observed(result)
        ref.equal(len(ops),6,"independent B1 kernels rather than one fused batch kernel")
        for op in ops.values():
            event=events[op.id]
            ref.equal(event.end_ns-event.start_ns,prices[(op.kind,event.device)],"each request exact cost charged once")
        return check.complete_and_capacity(result,cache_budget=32768,weight_inventory=_weight_inventory("dense",1))


def s05(directory):
    result,expected,evidence=_run(directory,handoff=True,prefill=2)
    evidence["handoff"]=check.pd_handoff(result,requests=expected["requests"],scope_map=expected["scope_map"])
    return evidence


def s06(directory):
    result,expected,evidence=_run(directory,kind="moe",prefill=3)
    evidence["expert_compute_checks"]=check.expert_visibility(result)
    ops,_,events=check.observed(result);orders={}
    for scope,contract in expected["scope_map"].items():
        if contract["phase"]!="prefill":continue
        experts=sorted((o for o in ops.values() if o.invocation_id==scope and o.kind=="expert_ffn"),key=lambda o:events[o.id].start_ns)
        ref.equal([o.expert_id for o in experts],[1,0],"token_count_desc selects two-row expert before one-row expert")
        orders[scope]=[o.expert_id for o in experts]
    evidence["expert_orders"]=orders
    return evidence


def sc01(directory):
    result,expected,evidence=_run(directory)
    evidence["asynchronous_progress"]=check.asynchronous_progress(result,expected["scope_map"])
    evidence["overlap"]=check.execution_overlap(result,expected["scope_map"])
    return evidence


def sc02(directory):
    result,expected,evidence=_run(directory,handoff=True,prefetch=True,prefill=2)
    evidence["handoff"]=check.pd_handoff(result,requests=expected["requests"],scope_map=expected["scope_map"])
    evidence["prefetch"]=check.prefetch_witnesses(result,expected["scope_map"])
    return evidence


def sc03(directory):
    result,expected,evidence=_run(directory,kind="moe",prefetch=True,prefill=3,expert_order="history_frequency_desc")
    evidence["expert_compute_checks"]=check.expert_visibility(result)
    evidence["prefetch"]=check.prefetch_witnesses(result,expected["scope_map"])
    ops,tensors,events=check.observed(result);reused=[]
    for layer in range(2):
        tid=f"w:layer{layer}:E0"
        consumers=[o for o in ops.values() if o.kind=="expert_ffn" and tid in o.inputs and events[o.id].device=="gpu"]
        ref.require(len(consumers)>=2,"no repeated GPU expert physical identity")
        copies=[e for e in result.events if e.kind=="COPY" and e.target_id==tid and e.dst=="vram"]
        ref.equal(len(copies),1,"retained expert copied once across requests and rounds")
        ref.equal(copies[0].bytes_transferred,1536,"packed expert three F32 8x16 matrices")
        reused.append(dict(tensor_id=tid,gpu_consumers=len(consumers),copies=len(copies)))
    evidence["expert_cache_reuse"]=reused
    return evidence


def sc04(directory):
    result,expected,evidence=_run(directory,kind="moe",handoff=True,prefetch=True,prefill=2)
    evidence["handoff"]=check.pd_handoff(result,requests=expected["requests"],scope_map=expected["scope_map"])
    evidence["expert_compute_checks"]=check.expert_visibility(result)
    return evidence


def sn01(directory):
    compiled,expected=_build(directory,arrival=100_000)
    with compiled:
        session=create_session(compiled.plan._compiled);before=session.observe()
        ref.require(not any(o.request_id=="request1" for o in before.operations),"future request initially visible")
        target="g:request1-prefill-layer0:op:layer0:request1:proj"
        decision=Decision("COMPUTE",target,device="cpu")
        ref.write_json(directory/"expected-action.json",dict(decision=asdict(decision),
            required="has not been revealed",request_arrival_ns=100_000,current_time_ns=0))
        ref.write_json(directory/"before.json",asdict(before))
        try:compile_command(decision,before)
        except Unsupported as error:
            ref.require("has not been revealed" in str(error),"wrong arrival visibility rejection")
            return dict(exception=type(error).__name__,message=str(error),arrival_ns=100_000,decision=asdict(decision))
        raise AssertionError("unarrived request execution accepted")


def sn02(directory):
    target="g:request0-request0-decode0:trace_next:request0"
    hidden="g:request0-prefill-layer1:t:layer1:request0:ffn"
    kv_ids=[f"kv:layer{layer}:request0:{position}" for layer in range(2) for position in range(2)]
    withheld=kv_ids[-1]
    ref.write_json(directory/"expected-action.json",dict(
        decision=asdict(Decision("COMPUTE",target,device="cpu")),
        gate_inputs=[hidden,*kv_ids],withheld_kv=withheld,
        precondition="all six request0 Prefill macros DONE; final hidden and all other KV READY in dram; CPU idle",
        required="INPUT_NOT_READY and the precise withheld KV identity in the error"))
    compiled,expected=_build(directory,handoff=True,prefill=2,arrival=1_000_000)
    with compiled:
        setup=compiled.run();ref.require(setup.success,"PD negative setup full run failed")
        setup.save(directory/"setup-execution")
        session=create_session(compiled.plan._compiled);prefix=[]
        # This is an explicit negative-case state constructor using ordinary
        # public commands on the same compiled runtime, not another simulator
        # or a serving scheduling policy. Run only the permitted Prefill chain.
        for _ in range(300):
            observation=session.observe()
            if any(o.id==target for o in observation.operations):break
            if observation.running:
                decision=Decision("ADVANCE")
            else:
                tensors={t.id:t for t in observation.tensors}
                ready=[o for o in observation.operations if o.status=="NOT_STARTED"
                       and all(any(status=="READY" for _,status in tensors[tid].copies) for tid in o.inputs)]
                ref.require(ready,"negative PD setup lacks a ready Prefill/input operation")
                op=ready[0]
                ref.require(op.phase!="decode","negative setup advanced beyond Prefill")
                # Prefill macros are GPU-bound. The model's request-arrival
                # producer has its explicit CPU source-materialization price.
                device="cpu" if op.kind=="state_update" else "gpu"
                destination="dram" if device=="cpu" else "vram"
                missing=next((tid for tid in op.inputs if tensors[tid].status_at(destination)!="READY"),None)
                decision=(Decision("COPY",missing,src="dram" if destination=="vram" else "vram",dst=destination) if missing else
                          Decision("COMPUTE",op.id,device=device))
            outcome=dispatch_command(session,compile_command(decision,observation))
            ref.require(outcome.ok,"negative PD setup command failed")
            prefix.append(asdict(decision))
        else:raise AssertionError("negative PD setup did not reach first-Decode gate")
        observation=session.observe()
        prefills=[o for o in observation.operations if o.request_id=="request0" and o.phase=="prefill" and o.kind!="state_update"]
        ref.equal(len(prefills),6,"six completed Prefill macros")
        ref.require(all(o.status=="DONE" for o in prefills),"Prefill unfinished in handoff-specific negative state")
        gate=next(o for o in observation.operations if o.id==target)
        ref.equal(gate.inputs,tuple([hidden,*kv_ids]),"gate has exactly the independently required full handoff inputs")
        for tid in (hidden,*kv_ids[:-1]):
            decision=Decision("COPY",tid,src="vram",dst="dram")
            outcome=dispatch_command(session,compile_command(decision,session.observe()))
            ref.require(outcome.ok,"negative setup handoff COPY failed")
            prefix.append(asdict(decision))
            decision=Decision("ADVANCE")
            outcome=dispatch_command(session,compile_command(decision,session.observe()))
            ref.require(outcome.ok,"negative setup COPY completion failed")
            prefix.append(asdict(decision))
        before=session.observe();tensors={t.id:t for t in before.tensors}
        ref.require(all(tensors[tid].status_at("dram")=="READY" for tid in (hidden,*kv_ids[:-1])),"other handoff operands not ready")
        ref.require(tensors[withheld].status_at("dram")!="READY","negative setup did not withhold the selected KV")
        ref.require(not before.running,"CPU/resource contention would confound the negative")
        decision=Decision("COMPUTE",target,device="cpu")
        outcome=dispatch_command(session,compile_command(decision,before))
        ref.write_json(directory/"before.json",asdict(before));ref.write_json(directory/"setup-prefix.json",prefix)
        ref.write_json(directory/"step.json",asdict(outcome))
        ref.require(not outcome.ok,"gate accepted incomplete all-layer KV handoff")
        ref.equal(outcome.error_code,"INPUT_NOT_READY","specific handoff operand-readiness diagnostic")
        ref.require(withheld in outcome.error_message,"rejection does not identify the sole missing KV operand")
        return dict(error_code=outcome.error_code,message=outcome.error_message,withheld_kv=withheld,
            prefill_complete=True,all_other_gate_operands_ready=True,setup_summary=setup.summary())


def sn03(directory):
    generated=_batch_inputs(directory);root=directory/"inputs";path=root/"layer0-prefill.costs.json"
    document=json.loads(path.read_text(encoding="utf-8"))
    for row in document["entries"]:row["batch"]=1
    ref.write_json(path,document)
    ref.write_json(directory/"expected.json",dict(invocation_batch=2,provided_cost_batch=1,required="strict missing/mismatched exact batch rejection"))
    try:
        plan=ModelPlan(id="serving-missing-batch-price",model_kind=generated["model_kind"],model_file=generated["model_file"],
            environment_file=generated["environment_file"],base_dir=root,scheduling=Scheduling(default_device="gpu"))
        plan.compile()
    except (InvalidInput,Unsupported) as error:
        ref.require("batch" in str(error) or "exact" in str(error) or "成本表" in str(error),"wrong batch-price rejection reason")
        return dict(exception=type(error).__name__,message=str(error))
    raise AssertionError("batch1 prices silently satisfy a batch2 invocation")


def sn04(directory):
    compiled,expected=_build(directory,kind="moe",future_expert=2)
    with compiled:
        before=create_session(compiled.plan._compiled).observe()
        ref.equal(before.route_history,(),"no initial public routes")
        ref.require(not any(o.kind=="expert_ffn" for o in before.operations),"expert task visible before router")
        decision=Decision("COMPUTE","g:request0-decode0-layer0:op:layer0:__moe_batch__:expert.2",device="gpu")
        ref.write_json(directory/"expected-action.json",dict(decision=asdict(decision),
            required="has not been revealed",public_route_history=[]))
        ref.write_json(directory/"before.json",asdict(before))
        try:compile_command(decision,before)
        except Unsupported as error:
            ref.require("has not been revealed" in str(error),"wrong future-route rejection")
            return dict(exception=type(error).__name__,message=str(error),decision=asdict(decision))
        raise AssertionError("private future expert execution accepted")


def sq01(directory):
    prefixes=[];runs=[]
    for arrival,length in ((400_000,2),(500_000,3)):
        folder=directory/str(arrival);folder.mkdir(parents=True)
        result,_,evidence=_run(folder,arrival=arrival,decode=(1,length),future_prefill=length,prefetch=True)
        prefixes.append(check.canonical_prefix(result,before_ns=400_000));runs.append(evidence)
    ref.equal(prefixes[0],prefixes[1],"different private future arrivals changed earlier public observation/decisions")
    ref.write_json(directory/"compared-prefix.json",prefixes[0])
    return dict(identical_prefix_decisions=len(prefixes[0]),runs=runs)


def sq02(directory):
    prefixes=[];runs=[]
    for future in (1,2):
        folder=directory/str(future);folder.mkdir(parents=True)
        result,_,evidence=_run(folder,kind="moe",handoff=True,prefill=2,prefetch=True,future_expert=future)
        prefixes.append(check.canonical_prefix(result,before_route_position=2));runs.append(evidence)
    ref.equal(prefixes[0],prefixes[1],"different private future routes changed earlier public observation/decisions")
    ref.write_json(directory/"compared-prefix.json",prefixes[0])
    return dict(identical_prefix_decisions=len(prefixes[0]),runs=runs)


CASES={**{f"S{i:02d}":globals()[f"s{i:02d}"] for i in range(1,7)},
       **{f"SC{i:02d}":globals()[f"sc{i:02d}"] for i in range(1,5)},
       **{f"SN{i:02d}":globals()[f"sn{i:02d}"] for i in range(1,5)},
       **{f"SQ{i:02d}":globals()[f"sq{i:02d}"] for i in range(1,3)}}
