"""Pure-attention request subbatches with materialized decoder-block ports.

The complete .5 frontend already produces post-normalization/post-RoPE Q/K,
V, and optional gate. This variant isolates causal GQA attention from the
output projection/residual, using independent exact synthetic stage prices.
Source invocations remain individually validated B=1 kernels. Group metadata
does not turn their operators/prices into fused request-batch kernels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from ..tensor.spec import InvalidInput, Unsupported, ROLE_INTERMEDIATE, build_scenario, ROUTE_ENDS
from .costs import M2CostTable, MacroShape, TransferEntry, expand_transfer_costs
from .expand import CompiledInvocation, CompiledTask, _physical_tensor
from .hybrid_physical import HybridDensePhysical
from .dense_contract import validate_config
from .spec import StateIn, tensor_id, op_id

FORMAT_VERSION = "dense-attention-execution-0.1"
_KINDS = frozenset({"attention_core", "attention_output"})


@dataclass(frozen=True)
class AttentionStageCost:
    kind: str
    device: str
    phase: str
    batch: int
    tokens: int
    visible_history: int
    duration_ns: int
    workspace_bytes: int

    def key(self):
        return (self.kind,self.device,self.phase,self.batch,self.tokens,self.visible_history)


@dataclass(frozen=True)
class AttentionTransfer:
    route: str
    storage_bytes: int
    duration_ns: int


@dataclass(frozen=True)
class RequestSubbatch:
    id: str
    request_ids: tuple[str, ...]
    attention_device: str


@dataclass(frozen=True)
class DenseAttentionExecutionSpec:
    format_version: str
    variant: str
    provenance: str
    invocation_files: tuple[str, ...]
    frontend_config: dict[str, Any]
    batch_contract: str
    subbatches: tuple[RequestSubbatch, ...]
    entries: tuple[AttentionStageCost, ...]
    transfer_entries: tuple[AttentionTransfer, ...]

    @property
    def expert_variant(self):
        # Generic execution export shares the existing optional field, without
        # implying that this Dense variant contains expert computations.
        return None


def _object(raw, keys, label):
    if not isinstance(raw,dict) or any(not isinstance(k,str) for k in raw) or set(raw)!=keys:
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")
    return raw


def _int(raw,label,minimum=0):
    if type(raw) is not int or raw<minimum:raise InvalidInput(f"{label} must be integer >= {minimum}")
    return raw


def _choice(raw,values,label):
    if not isinstance(raw,str) or raw not in values:raise Unsupported(f"{label} must be one of {sorted(values)}")
    return raw


def parse_attention_execution(document: Any) -> DenseAttentionExecutionSpec:
    required={"format_version","variant","provenance","invocation_files","frontend_config",
              "batch_contract","subbatches","entries","transfer_entries"}
    raw=_object(document,required,"dense_attention_execution")
    for name,values in (("format_version",{FORMAT_VERSION}),("variant",{"pure_attention"}),
                        ("provenance",{"synthetic_execution"}),
                        ("batch_contract",{"independent_requests_not_fused_batch_kernel"})):
        _choice(raw[name],values,name)
    files=raw["invocation_files"]
    if not isinstance(files,list) or not files or any(not isinstance(path,str) or not path.strip() for path in files):
        raise InvalidInput("invocation_files must be a nonempty array of explicit paths")
    if len(set(files))!=len(files):raise InvalidInput("invocation_files contains duplicates")
    config=validate_config(raw["frontend_config"])
    if not isinstance(raw["subbatches"],list) or not raw["subbatches"]:
        raise InvalidInput("subbatches must be a nonempty array")
    groups,seen_ids,seen_requests=[],set(),set()
    for index,item in enumerate(raw["subbatches"]):
        _object(item,{"id","request_ids","attention_device"},f"subbatches[{index}]")
        if not isinstance(item["id"],str) or not item["id"] or item["id"] in seen_ids:
            raise InvalidInput("subbatch IDs must be distinct nonempty strings")
        requests=item["request_ids"]
        if not isinstance(requests,list) or not requests or any(not isinstance(r,str) or not r for r in requests):
            raise InvalidInput("subbatch request_ids must be explicit nonempty strings")
        if len(set(requests))!=len(requests) or seen_requests.intersection(requests):
            raise InvalidInput("subbatches must not duplicate a request")
        device=_choice(item["attention_device"],{"cpu","gpu"},"attention_device")
        seen_ids.add(item["id"]);seen_requests.update(requests)
        groups.append(RequestSubbatch(item["id"],tuple(requests),device))
    cost_keys={"kind","device","phase","batch","tokens","visible_history","duration_ns","workspace_bytes"}
    if not isinstance(raw["entries"],list) or not isinstance(raw["transfer_entries"],list):
        raise InvalidInput("entries and transfer_entries must be explicit arrays")
    entries,seen=[],set()
    for index,item in enumerate(raw["entries"]):
        _object(item,cost_keys,f"entries[{index}]")
        _choice(item["kind"],_KINDS,"kind");_choice(item["device"],{"cpu","gpu"},"device")
        _choice(item["phase"],{"prefill","decode"},"phase")
        for name in ("batch","tokens","duration_ns"):_int(item[name],name,1)
        for name in ("visible_history","workspace_bytes"):_int(item[name],name)
        if item["batch"]!=1:raise Unsupported("attention stage prices must explicitly account for one request")
        entry=AttentionStageCost(**item)
        if entry.key() in seen:raise InvalidInput("duplicate attention stage cost key")
        seen.add(entry.key());entries.append(entry)
    transfers,seen=[],set()
    for index,item in enumerate(raw["transfer_entries"]):
        _object(item,{"route","storage_bytes","duration_ns"},f"transfer_entries[{index}]")
        _choice(item["route"],{"dram_to_vram","vram_to_dram"},"route")
        _int(item["storage_bytes"],"storage_bytes",1);_int(item["duration_ns"],"duration_ns",1)
        key=(item["route"],item["storage_bytes"])
        if key in seen:raise InvalidInput("duplicate attention transfer cost key")
        seen.add(key);transfers.append(AttentionTransfer(**item))
    return DenseAttentionExecutionSpec(FORMAT_VERSION,"pure_attention","synthetic_execution",tuple(files),config,
        "independent_requests_not_fused_batch_kernel",tuple(groups),tuple(entries),tuple(transfers))


def attention_execution_document(spec: DenseAttentionExecutionSpec):
    if not isinstance(spec,DenseAttentionExecutionSpec):raise InvalidInput("expected DenseAttentionExecutionSpec")
    if not isinstance(spec.invocation_files,tuple) or not isinstance(spec.subbatches,tuple) or \
            any(not isinstance(group,RequestSubbatch) or not isinstance(group.request_ids,tuple) for group in spec.subbatches):
        raise InvalidInput("typed attention spec has invalid source/group tuples")
    if not isinstance(spec.entries,tuple) or any(not isinstance(e,AttentionStageCost) for e in spec.entries) or \
            not isinstance(spec.transfer_entries,tuple) or any(not isinstance(e,AttentionTransfer) for e in spec.transfer_entries):
        raise InvalidInput("typed attention spec has invalid price tuples")
    return dict(format_version=spec.format_version,variant=spec.variant,provenance=spec.provenance,
        invocation_files=list(spec.invocation_files),frontend_config=deepcopy(spec.frontend_config),
        batch_contract=spec.batch_contract,subbatches=[dict(id=g.id,request_ids=list(g.request_ids),
            attention_device=g.attention_device) for g in spec.subbatches],
        entries=[asdict(e) for e in spec.entries],transfer_entries=[asdict(e) for e in spec.transfer_entries])


def compile_attention_subbatches(compilations: Sequence[CompiledInvocation], execution_spec: DenseAttentionExecutionSpec):
    """Combine validated B=1 frontend invocations and lower pure attention.

    Historical/new KV identities and output destinations stay request-specific.
    The root driver resolves and hashes invocation_files; this function neither
    opens filenames nor presents source paths or private traces to a policy.
    """
    spec=parse_attention_execution(attention_execution_document(execution_spec))
    sources=tuple(compilations)
    if not sources or any(not isinstance(c,CompiledInvocation) for c in sources):
        raise InvalidInput("attention subbatches require nonempty validated CompiledInvocation sources")
    if len(sources)!=len(spec.invocation_files):raise InvalidInput("source count differs from explicit invocation_files")
    first=sources[0]
    for source in sources:
        if source.layer.format_version!="m2-layer-0.5" or not isinstance(source.layer.physical,HybridDensePhysical) \
                or source.layer.ffn!="dense" or source.layer.attention!="full_causal":
            raise Unsupported("pure attention requires complete materialized hybrid Dense m2-layer-0.5")
        if source.invocation.batch!=1:raise Unsupported("each source must retain its validated B=1 contract")
        if source.layer!=first.layer or source.layer.physical.config!=spec.frontend_config:
            raise InvalidInput("source layer/config differs from exact attention frontend_config")
        if source.invocation.hardware!=first.invocation.hardware or source.scenario.architecture!=first.scenario.architecture:
            raise InvalidInput("subbatches must use the same explicit hardware")
        if source.invocation.phase!=first.invocation.phase or source.invocation.devices!=first.invocation.devices:
            raise Unsupported("first pure-attention bundle requires a single phase and identical allowed devices")
        if (source.physical_boundary or {}).get("attention_execution"):
            raise InvalidInput("attention variant was already materialized")
        # The shared frontend is the only authority for post-RoPE ports. A
        # caller cannot replace their byte/shape inventory while retaining a
        # LayerDescriptor that promises a different materialized boundary.
        from .hybrid_physical import validate_hybrid_physical_layer
        validate_hybrid_physical_layer(source.layer)
    requests=tuple(request for source in sources for request in source.invocation.requests)
    ids=[request.request_id for request in requests]
    if len(set(ids))!=len(ids):raise InvalidInput("source request IDs must be globally distinct")
    binding={request:group for group in spec.subbatches for request in group.request_ids}
    if set(binding)!=set(ids):raise InvalidInput("subbatches must cover exactly every source request")
    if any(group.attention_device=="cpu" and first.invocation.phase!="decode" for group in spec.subbatches):
        raise Unsupported("NEO-style CPU attention subbatch requires decode; CPU prefill is outside this slice")
    tensors,operations,tasks,compute,allowed,transfers,goals={},[],[],{},{},{},{}
    entries,transfer_prices={},{}
    input_history=[];appended={};new_blocks={};hidden_out={};source_boundaries=[]
    for source in sources:
        for tensor in source.workload_document["tensors"]:
            previous=tensors.get(tensor["id"])
            if previous is not None and previous!=tensor:raise InvalidInput("shared tensor inventory differs across sources")
            tensors[tensor["id"]]=deepcopy(tensor)
        operations.extend(deepcopy(source.workload_document["operations"]))
        tasks.extend(source.tasks);compute.update(deepcopy(source.scenario_document["costs"]["compute"]))
        allowed.update(deepcopy(source.scenario_document["mapspace"]["allowed_devices"]))
        goals.update(source.scenario_document["goal"]["output_locations"])
        input_history.extend(source.state_in.requests)
        appended.update(source.appended_positions);new_blocks.update(source.new_blocks);hidden_out.update(source.hidden_out_ids)
        source_boundaries.append(deepcopy(source.physical_boundary))
        for key,value in source.cost_table.entries.items():
            if key[0]=="attention":continue
            if key in entries and entries[key]!=value:raise InvalidInput("source compute costs disagree for identical keys")
            entries[key]=value
        for key,value in source.cost_table.transfers.items():
            if key in transfer_prices and transfer_prices[key]!=value:raise InvalidInput("source exact transfer prices disagree")
            transfer_prices[key]=value
        for route,names in source.scenario_document["mapspace"]["allowed_transfers"].items():
            transfers.setdefault(route,[])
            transfers[route]=list(dict.fromkeys(transfers[route]+names))
    prices={entry.key():entry for entry in spec.entries};config=spec.frontend_config
    metadata,scopes,regions={},{},[]
    all_tasks=[];all_ops=[];operations_by_id={op["id"]:op for op in operations}
    for task in tasks:
        if task.kind!="attention":
            all_tasks.append(task);all_ops.append(operations_by_id[task.operation_id]);continue
        old=operations_by_id[task.operation_id];request=next(r for r in requests if r.request_id==task.request_id)
        group=binding[task.request_id]
        if group.attention_device not in first.invocation.devices:raise Unsupported("subbatch attention device is not allowed by sources")
        projection=next(t for t in tasks if t.kind=="projection" and t.request_id==task.request_id)
        ports={port:tensor_id(first.layer.id,task.request_id,f"{projection.task_id}.{port}")
               for port in first.layer.physical.projection_ports(request.tokens)}
        if any(name not in tensors for name in ports.values()):raise Unsupported("frontend lacks declared materialized QKV ports")
        for port,shape in first.layer.physical.projection_ports(request.tokens).items():
            tensor=tensors[ports[port]]
            if tuple(tensor["ne"])!=shape or tensor["dtype"]!="GGML_TYPE_F32":
                raise InvalidInput("frontend QKV/gate ports do not match declared GQA geometry")
        history=next(state for state in input_history if state.request_id==task.request_id)
        core_id=op_id(first.layer.id,task.request_id,task.task_id+".core")
        out_id=op_id(first.layer.id,task.request_id,task.task_id+".output")
        context=tensor_id(first.layer.id,task.request_id,task.task_id+".context")
        if context in tensors or core_id in operations_by_id or out_id in operations_by_id:
            raise InvalidInput("derived pure-attention IDs collide with frontend graph")
        tensors[context]=_physical_tensor(context,ROLE_INTERMEDIATE,"GGML_TYPE_F32",
                                          (config["heads"]*config["head_dim"],request.tokens,1,1))
        core_inputs=[ports["q"],ports["k"],ports["v"],*(block.id for block in history.blocks)]
        output_inputs=[context]
        if "gate" in ports:output_inputs.append(ports["gate"])
        output_inputs += [name for name in old["inputs"] if tensors[name]["role"]=="weight"]
        output_inputs.append(tensor_id(first.layer.id,task.request_id,"in.hidden"))
        items=tuple((task.request_id,position) for position in appended[task.request_id])
        for stage,oid,inputs,output,devices,extras in (
                ("attention_core",core_id,core_inputs,context,(group.attention_device,),old.get("extra_outputs",[])),
                ("attention_output",out_id,output_inputs,task.output_tensor,first.invocation.devices,[])):
            stage_task=CompiledTask(task.request_id,task.task_id+(".core" if stage=="attention_core" else ".output"),stage,oid,devices[0],task.shape,
                                    output,None,(),None,items)
            all_tasks.append(stage_task)
            operation=dict(id=oid,semantic_op=stage,ggml_op="M2_"+stage.upper(),unary_op=None,
                           op_params=[],inputs=inputs,output=output)
            if extras:operation["extra_outputs"]=list(extras)
            all_ops.append(operation);allowed[oid]=list(devices)
            compute[oid]={}
            for device in devices:
                key=(stage,device,task.shape.phase,1,task.shape.tokens,task.shape.visible_history)
                cost=prices.get(key)
                if cost is None:raise Unsupported(f"missing exact independent attention stage cost {key!r}")
                compute[oid][device]=dict(duration_ns=cost.duration_ns,workspace_bytes=cost.workspace_bytes)
            metadata[oid]=dict(region_id=group.id,execution_stage=stage,execution_variant="pure_attention",
                               request_id=task.request_id,attention_device_binding=group.attention_device)
        scopes[context]=[dict(request_id=req,absolute_position=pos) for req,pos in items]
        del compute[task.operation_id];del allowed[task.operation_id]
        regions.append(dict(subbatch_id=group.id,request_id=task.request_id,
                            attention_device=group.attention_device,projection_operation_id=projection.operation_id,
                            core_operation_id=core_id,output_operation_id=out_id,context_tensor=context,
                            qkv_ports=ports,historical_kv_ids=[block.id for block in history.blocks],
                            new_kv_ids=[block.id for block in new_blocks[task.request_id]]))
        for route in ("dram_to_vram","vram_to_dram"):
            if context not in transfers[route]:transfers[route].append(context)
    for transfer in spec.transfer_entries:
        transfer_prices[(transfer.route,transfer.storage_bytes)]=TransferEntry(transfer.duration_ns,"synthetic_transfer")
    table=M2CostTable(source=first.cost_table.source+";independent_attention_stages",entries=entries,
        validation_only=any(source.cost_table.validation_only for source in sources),
        transfers=transfer_prices,format_version="m2-layer-0.5")
    _,transfer_document=expand_transfer_costs(table,{name:t["storage_bytes"] for name,t in tensors.items()},transfers)
    workload=deepcopy(first.workload_document)
    workload.update(id=first.workload_document["id"]+":attention-subbatches",tensors=list(tensors.values()),
                    operations=all_ops,outputs=list(goals))
    workload["origin"]=dict(kind="synthetic_layer",ggml_version=None,
        sample="materialized pure attention; independent B1 source kernels; synthetic split stage costs")
    scenario_doc=deepcopy(first.scenario_document)
    scenario_doc.update(id=workload["id"],goal=dict(output_locations=goals))
    scenario_doc["costs"]=dict(source=";".join(sorted({source.costs_source for source in sources}))+";dense-attention:synthetic_execution",
                               compute=compute,transfer=transfer_document)
    scenario_doc["mapspace"].update(allowed_devices=allowed,allowed_transfers=transfers,transfer_dependencies={})
    scenario=build_scenario(workload,scenario_doc,where="attention_subbatches")
    state_in=StateIn(requests=tuple(input_history),layer=first.state_in.layer)
    invocation=replace(first.invocation,id=first.invocation.id+":attention-subbatches",requests=requests,state_in=state_in)
    boundary=dict(scope="pure_attention_request_subbatches",numeric_execution_verified=False,
                  original_model_kernel_reproduced=False,physical_macro_contract=False,
                  source_physical_boundaries=source_boundaries,operation_stages=metadata,tensor_row_scopes=scopes)
    boundary["attention_execution"]=dict(format_version=FORMAT_VERSION,
        execution_spec=attention_execution_document(spec),subbatches=[asdict(group) for group in spec.subbatches],
        regions=regions,operation_stages=metadata,tensor_row_scopes=scopes,
        batch_accounting="one_exact_B1_price_per_request_not_a_fused_batch_price",
        projection_semantics="existing_materialized_RMSNorm_QKV_QKnorm_normal_RoPE_and_optional_gate",
        core_semantics="causal_QK_softmax_PV_GQA_only_and_logical_KV_append_no_output_projection_or_residual",
        output_semantics="optional_sigmoid_gate_then_output_projection_plus_hidden_residual",
        kv_bytes_per_token=8*config["kv_heads"]*config["head_dim"],
        gqa_head_mapping="query_head // (heads // kv_heads)",
        limitations=["single_phase_bundle", "CPU_attention_decode_only", "no_fused_batch_kernel",
                     "no_request_admission_queues", "no_multilayer_pipeline", "no_physical_calibration",
                     "next_invocation_recompiles_individual_B1_sources_from_request_state"])
    return CompiledInvocation(first.layer,invocation,state_in,table,scenario,scenario_doc["costs"]["source"],
        workload,scenario_doc,tuple(all_tasks),appended,new_blocks,hidden_out,boundary)


def transform_dense_attention(compiled: CompiledInvocation, execution_spec: DenseAttentionExecutionSpec):
    return compile_attention_subbatches((compiled,),execution_spec)
