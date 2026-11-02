"""Exact profile costs through the common fixed tensor IR/compiler/runtime.

This is timing-only: no public high-level ModelPlan coverage or physical memory
capacity claim. The single PCIe resource explicitly queues the second flow.
"""
from __future__ import annotations
from dataclasses import asdict
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for folder in (ROOT/"modeling",ROOT/"mapping"):
    if str(folder)not in sys.path:sys.path.insert(0,str(folder))
from llm_infer_model.tensor.spec import build_scenario
from tensor_mapping.strategy_ir import parse_strategy_ir
from tensor_mapping.strategy_compiler import compile_strategy,run_strategy


def bind(case,strategy,prices,known_workspace_bytes):
    if strategy not in ("serial/one_copy_compute","overlap/one_copy_compute","serial/two_copy_compute","overlap/two_copy_compute"):
        raise ValueError("undeclared overlap strategy")
    rounded={name:math.ceil(value["median_ns"])for name,value in prices.items()}
    if any(value<1 for value in rounded.values()):raise ValueError("positive exact profile costs required")
    if any(source.get("split")!="profile"for value in prices.values()for source in value["sources"]):raise ValueError("heldout cost sources forbidden")
    tensors=[]
    def tensor(name,shape,role,location=None):
        ne=[*shape,*([1]*(4-len(shape)))];nb=[4]
        for n in ne[:-1]:nb.append(nb[-1]*n)
        tensors.append(dict(id=name,name=name,role=role,dtype="GGML_TYPE_F32",ne=ne,nb=nb,
                            storage_bytes=4*math.prod(ne),initial_locations=[]if location is None else[location]))
    h,i,q=(case[k]for k in ("hidden","intermediate","tokens"))
    tensor("current_input",(h,q),"input","vram")
    tensor("current_gate",(h,i),"weight","vram");tensor("current_up",(h,i),"weight","vram");tensor("current_down",(i,h),"weight","vram")
    tensor("current_output",(h,q),"output")
    two="two_copy"in strategy
    payloads=["next_payload_a"]+(["next_payload_b"]if two else[])
    for name in payloads:tensor(name,(case["payload_bytes"]//4,),"input","dram")
    operation=dict(id="current_ffn",semantic_op="expert_ffn",ggml_op="MEASURED_GGML_F32_COMPACT_SWIGLU_FFN",
        unary_op=None,op_params=[],inputs=["current_input","current_gate","current_up","current_down"],output="current_output",extra_outputs=[])
    workload=dict(schema_version="0.2",id=case["id"]+"-"+strategy.replace("/","-"),origin=dict(kind="synthetic_layer"),
                  tensors=tensors,operations=[operation],outputs=["current_output",*payloads])
    routes=("dram_to_vram","vram_to_dram","ssd_to_dram","dram_to_ssd")
    transfers={route:{}for route in routes}
    transfers["dram_to_vram"]={name:dict(duration_ns=rounded["copy/"+("a"if name.endswith("a")else"b")])for name in payloads}
    scenario=dict(schema_version="0.2",id=workload["id"],architecture=dict(dram_capacity_bytes=1<<60,vram_capacity_bytes=1<<60,
        dram_runtime_reserved_bytes=0,vram_runtime_reserved_bytes=0,dram_alignment_bytes=1,vram_alignment_bytes=1,
        cpu_compute_slots=1,gpu_compute_slots=1,pcie_copy_slots=1,ssd_io_slots=1),
        costs=dict(source="frozen_exact_isolated_overlap_profile_NOT_capacity",compute={"current_ffn":{"gpu":dict(duration_ns=rounded["compute/current_ffn"],workspace_bytes=known_workspace_bytes)}},
            transfer=transfers),
        mapspace=dict(allowed_devices={"current_ffn":["gpu"]},allowed_transfers={route:payloads if route=="dram_to_vram"else[]for route in routes},
                      allow_eviction=False,allow_copy_compute_overlap=True,allow_cpu_gpu_overlap=True),
        goal=dict(output_locations={name:"vram"for name in ["current_output",*payloads]}),mapper=dict(algorithm="uniform_cost"))
    def transfer(name):return dict(kind="transfer",tensor_id=name,src="dram",dst="vram")
    def wait(name):return dict(kind="wait",operations=[],copies=[dict(tensor_id=name,location="vram")])
    compute=dict(kind="compute",operation_id="current_ffn")
    if strategy.startswith("serial/"):
        program=[]
        for name in payloads:program.extend((transfer(name),wait(name)))
        program.append(compute)
    else:
        program=[transfer(payloads[0]),compute]
        if two:program.extend((wait(payloads[0]),transfer(payloads[1])))
    program.append(dict(kind="drain"))
    document=dict(format_version="strategy-ir-0.1",id=workload["id"],scenario_file="scenario.json",information_mode="offline_fixed",
                  regions=[],placements={"current_ffn":"gpu"},program=program,device_states={})
    state=build_scenario(workload,scenario)
    compiled=compile_strategy(parse_strategy_ir(document),state)
    run=run_strategy(compiled)
    if run.evaluation.status!="valid":raise ValueError("common fixed runtime rejected path: "+run.evaluation.reason)
    copies=rounded["copy/a"]+(rounded["copy/b"]if two else 0)
    expected=copies+rounded["compute/current_ffn"]if strategy.startswith("serial/")else max(copies,rounded["compute/current_ffn"])
    if run.evaluation.makespan_ns!=expected:raise ValueError("common runtime differs from declared sum/max hypothesis")
    return dict(case_id=case["id"],strategy=strategy,predicted_ns=run.evaluation.makespan_ns,
        independent_links_ideal_ns=max(rounded["copy/a"],rounded["copy/b"]if two else 0,rounded["compute/current_ffn"]),
        workload=workload,scenario=scenario,strategy_ir=document,
        common_fixed_tensor_compiler_runtime_exercised=True,highlevel_ModelPlan_coverage=False,
        known_workspace_lower_bound_only=known_workspace_bytes,complete_cuda_workspace_bytes=None,physical_capacity_verified=False,
        queue_semantics="one shared PCIe service resource; second copy service explicitly queued while compute may run",
        native_submission_difference="two-flow native enqueues both streams before compute; IR queues second link service without modeling host submission overhead",
        events=[asdict(event)for event in run.evaluation.events])
