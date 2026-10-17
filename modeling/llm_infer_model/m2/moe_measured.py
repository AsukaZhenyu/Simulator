"""Bind measured resident MoE FFN macros to the existing tensor simulator.

This is an explicit physical subgraph, separate from the synthetic .6 layer.
Attention, KV and offloading are outside its measured compute boundary. The
external route prescribes work; random probe weights are not original weights.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from ..tensor.engine import Action, ACTION_ADVANCE, ACTION_COMPUTE, evaluate_mapping
from ..tensor.spec import InvalidInput, Unsupported, build_scenario
from .moe_ffn_contract import (STABILITY_CV_LIMIT, contract_hash, groups,
    physical_signature, sampling_mode, stage_names, stage_ports, statistics_ns, validate_case, validate_native)


@dataclass(frozen=True)
class MeasuredMoEFFN:
    case: dict[str, Any]
    record: dict[str, Any]
    scenario: Any
    workload_document: dict[str, Any]
    scenario_document: dict[str, Any]
    bindings: dict[str, Any]
    validation_only: bool


def _physical_tensor(name, port, *, initial):
    a, b, c, d = port["ne"]
    return dict(id=name, name=name,
        role="weight" if port["category"] == "weight" else "input" if initial else
             "output" if name == "hidden_output" else "intermediate",
        dtype="GGML_TYPE_I32" if port["dtype"] == "i32" else "GGML_TYPE_F32",
        ne=[a,b,c,d], nb=[4,4*a,4*a*b,4*a*b*c], storage_bytes=port["bytes"],
        initial_locations=["dram"] if initial else [])


def compile_measured_ffn(record: dict, hardware: dict, *, allow_unstable=False) -> MeasuredMoEFFN:
    """Use exact measured stage prices; refuse missing, unstable or unknown data.

    The caller supplies the complete DRAM/VRAM budget explicitly. CPU-only
    execution has no implicit GPU fallback. Joint samples never become prices.
    """
    if type(allow_unstable) is not bool:
        raise InvalidInput("allow_unstable must be an explicit boolean")
    if not isinstance(record, dict) or "case" not in record or "native" not in record:
        raise InvalidInput("measured MoE record must contain case and native evidence")
    if record.get("status") not in ("measured", "measured_validation_only"):
        raise InvalidInput("measured MoE requires a completed measurement status")
    try:
        case = validate_case(record["case"])
        native = validate_native(record["native"], case)
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidInput(f"invalid measured MoE evidence: {error}") from error
    signature = record.get("physical_signature")
    if not isinstance(signature, dict):
        raise InvalidInput("physical_signature is required")
    try:
        expected_signature = physical_signature(case, signature.get("native_binary_sha256"))
        expected_hash = contract_hash(expected_signature)
        supplied_hash = contract_hash(signature)
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidInput(f"invalid measured MoE signature: {error}") from error
    if supplied_hash != expected_hash or record.get("physical_contract_sha256") != expected_hash:
        raise InvalidInput("measured MoE contract does not match the exact case/signature")
    if signature.get("native_binary_sha256") is None:
        raise InvalidInput("measured MoE requires the native binary SHA256")
    architecture_keys = {"dram_capacity_bytes","vram_capacity_bytes","dram_runtime_reserved_bytes",
        "vram_runtime_reserved_bytes","dram_alignment_bytes","vram_alignment_bytes"}
    if not isinstance(hardware, dict) or set(hardware) != architecture_keys:
        raise InvalidInput(f"hardware must explicitly contain {sorted(architecture_keys)}")

    grouped = groups(case)
    contextual = sampling_mode(case) == "interleaved_joint_context"
    control_edges = []
    stages = stage_names(case)[:-1]
    stats = {stage: statistics_ns(native["stages"][stage]["samples_ns"]) for stage in stages}
    unstable = [stage for stage in stages if stats[stage]["cv"] > STABILITY_CV_LIMIT]
    if unstable and not allow_unstable:
        raise Unsupported(f"unstable MoE stage costs (CV>{STABILITY_CV_LIMIT}): {unstable}")
    ports = stage_ports(case, "joint")
    tensors = [_physical_tensor(name, port, initial=port["category"] in ("input","weight"))
               for name, port in ports.items()]
    operations = [dict(id="router",semantic_op="router",ggml_op="M2_MEASURED_ROUTER",unary_op=None,
        op_params=[],inputs=["hidden_input","router_weight","route_ids"],
        output="selected_scores",extra_outputs=["discovered_topk"])]
    previous_expert = None
    for expert in grouped:
        inputs = ["hidden_input",f"gate.{expert}",f"up.{expert}",f"down.{expert}",
                  f"rows.{expert}","selected_scores"]
        if contextual and previous_expert is not None:
            inputs.append(f"expert_output.{previous_expert}")
            control_edges.append(dict(before=f"expert.{previous_expert}",after=f"expert.{expert}",
                                      readiness_tensor=f"expert_output.{previous_expert}"))
        operations.append(dict(id=f"expert.{expert}",semantic_op="expert_ffn",
            ggml_op="M2_MEASURED_EXPERT_FFN",unary_op=None,op_params=[],
            inputs=inputs,output=f"expert_output.{expert}"))
        previous_expert = expert
    operations.append(dict(id="combine",semantic_op="combine",ggml_op="M2_MEASURED_COMBINE",
        unary_op=None,op_params=[],inputs=["hidden_input","selected_scores",
            *(name for expert in grouped for name in (f"expert_output.{expert}",f"rows.{expert}",f"slots.{expert}"))],
        output="hidden_output"))
    identifier = "measured-moe-ffn:" + case["id"]
    fingerprint = record["physical_contract_sha256"]
    workload = dict(schema_version="0.2",id=identifier,
        origin=dict(kind="measured_macro",ggml_version=None,sample=native["contract"]+";sha256="+fingerprint),
        tensors=tensors,operations=operations,outputs=["hidden_output"])
    compute = {stage:{"cpu":dict(duration_ns=math.ceil(stats[stage]["median_ns"]),
        workspace_bytes=native["stages"][stage]["workspace_bytes"])} for stage in stages}
    scenario_doc = dict(schema_version="0.2",id=identifier,
        architecture={**copy.deepcopy(hardware),"cpu_compute_slots":1,"gpu_compute_slots":1,
                      "pcie_copy_slots":1,"ssd_io_slots":1},
        goal=dict(output_locations={"hidden_output":"dram"}),
        costs=dict(source="measured_macro:"+fingerprint+";validation_only="+str(bool(unstable)).lower(),
                   compute=compute,transfer={}),
        mapspace=dict(allowed_devices={stage:["cpu"] for stage in stages},allowed_transfers={},
            allow_eviction=False,allow_copy_compute_overlap=False,allow_cpu_gpu_overlap=False,
            allow_recomputation=False),
        mapper=dict(algorithm="uniform_cost",max_expanded_states=10000,wall_time_limit_s=1))
    scenario = build_scenario(workload, scenario_doc)
    bindings = dict(contract=native["contract"],physical_signature=copy.deepcopy(signature),
        physical_contract_sha256=fingerprint,scope="resident_moe_ffn_subgraph",
        provenance="measured_macro",stage_statistics=stats,unstable_cost_stages=unstable,
        validation_only=bool(unstable),duration_rounding="ceil_of_positive_sample_median_ns",
        expert_accounting="one_task_per_active_expert_group",groups={str(e):len(rows) for e,rows in grouped.items()},
        capacity_semantics="declared_logical_ports_plus_measured_stage_internal_workspace",
        physical_capacity_verified=False,whole_decoder_layer_measured=False,
        original_model_kernel_reproduced=False,random_weights_not_model_weights=True,
        sampling_mode=sampling_mode(case),context_sensitive_prices=contextual,
        cost_execution_order="fixed_serial_slice_order" if contextual else "isolated_warm_stage_prices",
        control_only_edges=control_edges,control_edges_are_numeric_port_reads=False,
        routing_mode="prescribed_trace_replay",router_discovery_used_for_grouping=False,
        performance_calibrated=False,online_policy_claimed=False)
    return MeasuredMoEFFN(copy.deepcopy(case),copy.deepcopy(record),scenario,workload,scenario_doc,bindings,bool(unstable))


def plan_measured_ffn(compiled: MeasuredMoEFFN):
    return tuple(action for stage in stage_names(compiled.case)[:-1]
                 for action in (Action(ACTION_COMPUTE,stage,device="cpu"),Action(ACTION_ADVANCE)))


def replay_measured_ffn(compiled: MeasuredMoEFFN):
    return evaluate_mapping(compiled.scenario,plan_measured_ffn(compiled))
