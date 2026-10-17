"""Synthetic full-causal MoE layer expansion using the existing tensor kernel.

Expert tasks group assigned tokens across requests once. This defines a
scheduling model, not a numerical MoE implementation or a measured GPU kernel.
"""
from __future__ import annotations

from typing import Mapping

from ..tensor.engine import ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD, Action
from ..tensor.spec import (ALGORITHM_UNIFORM_COST, DTYPE_I32, ROLE_INPUT, ROLE_INTERMEDIATE, ROLE_OUTPUT,
                          ROLE_WEIGHT, SCHEMA_VERSION_M1, InvalidInput, Unsupported, build_scenario)
from .costs import MacroShape, expand_costs_document, expand_transfer_costs
from .spec import block_id, tensor_id, op_id, weight_id, validate_invocation, KvBlockRef

MOE_FORMAT_VERSION = "m2-layer-0.6"
_BATCH = "__moe_batch__"


def compile_moe_invocation(layer, invocation, cost_table, *, state_in=None, where="<memory>"):
    from .expand import CompiledInvocation, CompiledTask, _tensor, _physical_tensor
    history = invocation.state_in if state_in is None else state_in
    validate_invocation(layer, invocation, history)
    if cost_table.format_version != MOE_FORMAT_VERSION:
        raise InvalidInput("MoE requires an explicit m2-layer-0.6 cost table")
    if cost_table.validation_only and not invocation.allow_validation_only_costs:
        raise Unsupported("validation_only costs require explicit invocation opt-in")
    # Direct cost-table construction must not bypass the synthetic boundary.
    if any(c.provenance != "synthetic_layer" or c.physical_signature is not None
           or c.contract_hash is not None for c in cost_table.entries.values()):
        raise Unsupported("m2-layer-0.6 MoE currently supports synthetic macro costs only")
    kinds = {t.kind: t for t in layer.tasks}
    tensors, operations, tasks = [], [], []
    goals, appended, new_blocks, hidden_out_ids = {}, {}, {}, {}
    cache = {w.id: w for w in history.weight_copies}
    for weight in layer.weights:
        locations = cache[weight.id].locations
        name = weight_id(layer.id, weight.id)
        tensors.append(_tensor(name, ROLE_WEIGHT, weight.storage_bytes, locations))
        goals[name] = "ssd" if "ssd" in locations else locations[0]
    refs = {r.request_id: r for r in invocation.requests}
    hiddens, attention_outputs, route_outputs = {}, {}, {}
    group_items = {}
    for entry in invocation.route_trace:
        for expert in entry.expert_ids:
            group_items.setdefault(str(expert), []).append((entry.request_id, entry.absolute_position))
    group_items = {key: tuple(sorted(items)) for key,items in sorted(group_items.items(),key=lambda x:int(x[0]))}
    if layer.ffn_params.get("shared_expert_weight") is not None:
        group_items["shared"] = tuple((r.request_id,p) for r in invocation.requests
            for p in range(r.position_start,r.position_start+r.tokens))
    # Validate aggregate work with the existing family component (no dropping).
    from ..components.ffn import MoEFFN
    family = MoEFFN(experts=layer.ffn_params["experts"], top_k=layer.ffn_params["top_k"],
                   expert_weight_bytes=layer.ffn_params["expert_weight_bytes"],
                   shared_expert_bytes=layer.ffn_params.get("shared_expert_bytes",0))
    family.plan(tokens=sum(r.tokens for r in invocation.requests),
                expert_tokens={int(key):len(items) for key,items in group_items.items() if key!="shared"})

    def add_task(request_id, task_id, kind, shape, inputs, output, *, extras=(), expert=None, work_items=(), deps=()):
        operation_id = op_id(layer.id, request_id, task_id)
        tasks.append(CompiledTask(request_id=request_id, task_id=task_id, kind=kind,
            operation_id=operation_id, device=invocation.devices[0], shape=shape, output_tensor=output,
            weight_id=None, depends_on=tuple(deps), expert_id=expert, work_items=tuple(work_items)))
        operation = dict(id=operation_id, semantic_op="expert_ffn" if kind=="shared_expert_ffn" else kind,
            ggml_op="M2_"+kind.upper(), unary_op=None, op_params=[], inputs=list(inputs), output=output)
        if extras: operation["extra_outputs"]=list(extras)
        operations.append(operation)
        return operation_id

    for request in invocation.requests:
        state = history.for_request(request.request_id)
        appended[request.request_id] = tuple(range(request.position_start,request.position_start+request.tokens))
        for block in state.blocks:
            tensors.append(_tensor(block.id,ROLE_INPUT,block.storage_bytes,block.locations))
            goals[block.id]=state.state_location
        name=tensor_id(layer.id,request.request_id,"in")
        hiddens[request.request_id]=name
        tensors.append(_tensor(name,ROLE_INPUT,request.tokens*layer.hidden_bytes_per_token,(request.input_location,)))
    for stage in ("projection","attention","router"):
        template=kinds[stage]
        for request in invocation.requests:
            req=request.request_id; state=history.for_request(req)
            shape=MacroShape(invocation.phase,invocation.batch,request.tokens,state.next_position)
            output=tensor_id(layer.id,req,template.id)
            weight_inputs=[] if template.weight is None else [weight_id(layer.id,template.weight)]
            if stage=="projection":
                inputs=[*weight_inputs,hiddens[req]]; extras=()
            elif stage=="attention":
                inputs=[*weight_inputs,tensor_id(layer.id,req,kinds["projection"].id),hiddens[req],
                        *(b.id for b in state.blocks)]
                blocks=[]
                for position in appended[req]:
                    name=block_id(layer.id,req,position)
                    tensors.append(_tensor(name,ROLE_OUTPUT,layer.kv_bytes_per_token))
                    goals[name]=state.state_location
                    blocks.append(KvBlockRef(name,position,1,layer.kv_bytes_per_token,(state.state_location,)))
                new_blocks[req]=tuple(blocks); extras=tuple(b.id for b in blocks)
                attention_outputs[req]=output
            else:
                inputs=[*weight_inputs,attention_outputs[req]]; extras=()
                route_outputs[req]=output
            add_task(req,template.id,stage,shape,inputs,output,extras=extras,deps=template.depends_on)
            if stage=="router":
                tensors.append(_physical_tensor(output,ROLE_INTERMEDIATE,DTYPE_I32,
                    (layer.ffn_params["top_k"],request.tokens,1,1)))
            else:
                tensors.append(_tensor(output,ROLE_INTERMEDIATE,request.tokens*layer.hidden_bytes_per_token))
    group_outputs, transfer_dependencies = {}, {}
    for group,items in group_items.items():
        req_ids=tuple(dict.fromkeys(req for req,pos in items))
        shared=group=="shared"
        weight_ref=(layer.ffn_params["shared_expert_weight"] if shared
                    else layer.ffn_params["expert_weights"][int(group)])
        wname=weight_id(layer.id,weight_ref)
        output=tensor_id(layer.id,_BATCH,"expert."+group)
        group_outputs[group]=output
        inputs=[wname,*(attention_outputs[req] for req in req_ids)]
        if not shared: inputs.extend(route_outputs[req] for req in req_ids)
        shape=MacroShape(invocation.phase,invocation.batch,len(items),0)
        add_task(_BATCH,"expert."+group,"shared_expert_ffn" if shared else "expert_ffn",shape,inputs,output,
                 expert=None if shared else int(group),work_items=items)
        tensors.append(_tensor(output,ROLE_INTERMEDIATE,len(items)*layer.hidden_bytes_per_token))
        if not shared and invocation.information_mode=="causal_replay":
            transfer_dependencies[wname]=[op_id(layer.id,req,kinds["router"].id) for req in req_ids]
    combine=kinds["combine"]
    for request in invocation.requests:
        req=request.request_id; state=history.for_request(req)
        shape=MacroShape(invocation.phase,invocation.batch,request.tokens,state.next_position)
        groups=[key for key,items in group_items.items() if any(r==req for r,pos in items)]
        output=tensor_id(layer.id,req,combine.id)
        inputs=[attention_outputs[req],route_outputs[req],*(group_outputs[key] for key in groups)]
        if combine.weight is not None: inputs.insert(0,weight_id(layer.id,combine.weight))
        add_task(req,combine.id,"combine",shape,inputs,output,deps=combine.depends_on)
        tensors.append(_tensor(output,ROLE_OUTPUT,request.tokens*layer.hidden_bytes_per_token))
        goals[output]=request.output_location; hidden_out_ids[req]=output
    requirements=[(task.operation_id,task.kind,dev,task.shape) for task in tasks for dev in invocation.devices]
    scratch=[]
    for op,kind,dev,shape in requirements:
        cost=cost_table.lookup(kind,dev,shape)
        expected="per_expert_group" if kind in ("expert_ffn","shared_expert_ffn") else "per_request"
        if cost.accounting != expected:
            raise InvalidInput(f"{kind} requires accounting={expected!r}")
        scratch.append(cost.workspace_bytes)
    maximum=max(scratch)
    if layer.workspace_bytes and layer.workspace_bytes!=maximum:
        raise InvalidInput("MoE workspace_bytes declares exact serial scratch maximum, not summed scratch")
    sizes={t["id"]:t["storage_bytes"] for t in tensors}
    read_weights={tid for op in operations for tid in op["inputs"] if tid.startswith("w:")}
    # An inactive expert can retain an old cached copy, but is not selectively
    # prefetched by this invocation. Its immutable source remains a goal.
    transferable=[t["id"] for t in tensors if t["role"]!=ROLE_WEIGHT or t["id"] in read_weights]
    allowed_transfers={route:list(transferable) for route in
        ("dram_to_vram","vram_to_dram","ssd_to_dram","dram_to_ssd")}
    transfer_sources,transfers=expand_transfer_costs(cost_table,sizes,allowed_transfers)
    source,costs=expand_costs_document(cost_table,requirements,transfer_provenances=transfer_sources)
    costs["transfer"]=transfers
    identity=f"m2:{layer.id}:{invocation.id}"
    workload=dict(schema_version=SCHEMA_VERSION_M1,id=identity,
        origin=dict(kind="synthetic_layer",ggml_version=None,sample=invocation.id+"; abstract MoE macros"),
        tensors=tensors,operations=operations,outputs=list(goals))
    hardware=invocation.hardware
    scenario_doc=dict(schema_version=SCHEMA_VERSION_M1,id=identity,
        architecture=dict(dram_capacity_bytes=hardware.dram_capacity_bytes,vram_capacity_bytes=hardware.vram_capacity_bytes,
            dram_runtime_reserved_bytes=hardware.dram_runtime_reserved_bytes,vram_runtime_reserved_bytes=hardware.vram_runtime_reserved_bytes,
            dram_alignment_bytes=hardware.dram_alignment_bytes,vram_alignment_bytes=hardware.vram_alignment_bytes,
            cpu_compute_slots=1,gpu_compute_slots=1,pcie_copy_slots=1,ssd_io_slots=1),
        goal=dict(output_locations=goals),costs=costs,
        mapspace=dict(allowed_devices={t.operation_id:list(invocation.devices) for t in tasks},
            allowed_transfers=allowed_transfers,transfer_dependencies=transfer_dependencies,allow_eviction=True,
            allow_copy_compute_overlap=True,allow_cpu_gpu_overlap=True,allow_recomputation=False),
        mapper=dict(invocation.mapper or dict(algorithm=ALGORITHM_UNIFORM_COST,max_expanded_states=10000,wall_time_limit_s=1)))
    scenario=build_scenario(workload,scenario_doc,where=where)
    boundary=dict(scope="abstract_moe_layer",numeric_execution_verified=False,physical_macro_contract=False,
        original_model_kernel_reproduced=False,expert_accounting="per_expert_group",
        expert_history_axis="explicit_zero_not_applicable_to_synthetic_expert_cost",
        replay_information_mode=invocation.information_mode,route_source=invocation.route_source,
        graph_information_mode="offline_oracle",online_policy_verified=False,
        routed_weight_gate="all_relevant_request_routers_done" if invocation.information_mode=="causal_replay" else "none",
        port_semantics="whole_request_inputs_and_grouped_expert_outputs; token rows recorded in expert_work_items",
        inactive_expert_prefetch_allowed=False,capacity_semantics="declared_tensor_bytes_plus_running_scratch",
        expert_work_items={group:[dict(request_id=req,absolute_position=pos) for req,pos in items]
                           for group,items in group_items.items()})
    return CompiledInvocation(layer,invocation,history,cost_table,scenario,source,workload,scenario_doc,
        tuple(tasks),appended,new_blocks,hidden_out_ids,boundary,expert_work_items=group_items)


def plan_moe(compiled, *, placements: Mapping[str,str] | None=None):
    """Serial placement plan with charged SSD staging and declared route gates."""
    selected={t.operation_id:t.device for t in compiled.tasks} if placements is None else dict(placements)
    if set(selected)!=set(compiled.compute_ids):
        raise InvalidInput("placements must name exactly every compiled operation_id")
    for operation,device in selected.items():
        if device not in compiled.scenario.mapspace.devices_for(operation):
            raise InvalidInput(f"device {device!r} is not allowed for {operation!r}")
    locations={t.id:set(t.initial_locations) for t in compiled.scenario.workload.tensors}
    actions=[]
    def transfer(kind,tensor,*,src=None,dst=None):
        actions.extend((Action(kind,tensor,src=src,dst=dst),Action(ACTION_ADVANCE)))
    def materialize(tensor,destination):
        if destination in locations[tensor]: return
        if destination=="ssd":
            materialize(tensor,"dram")
            transfer(ACTION_WRITE_SSD,tensor);locations[tensor].add("ssd");return
        if destination=="vram":
            materialize(tensor,"dram")
            transfer(ACTION_COPY,tensor,src="dram",dst="vram");locations[tensor].add("vram");return
        if "vram" in locations[tensor]:
            transfer(ACTION_COPY,tensor,src="vram",dst="dram")
        elif "ssd" in locations[tensor]:
            transfer(ACTION_READ_SSD,tensor)
        else:
            raise InvalidInput(f"serial MoE plan has no source for {tensor!r}")
        locations[tensor].add("dram")
    for task in compiled.tasks:
        device=selected[task.operation_id]; domain="dram" if device=="cpu" else "vram"
        op=compiled.scenario.workload.operation_by_id[task.operation_id]
        for tensor in op.inputs: materialize(tensor,domain)
        actions.extend((Action(ACTION_COMPUTE,task.operation_id,device=device),Action(ACTION_ADVANCE)))
        for tensor in op.outputs: locations[tensor].add(domain)
    for tensor,destination in compiled.scenario_document["goal"]["output_locations"].items():
        materialize(tensor,destination)
    return tuple(actions)
