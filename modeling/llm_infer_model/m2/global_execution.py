"""Finite, causally revealed multi-layer/multi-round M2 execution.

Every invocation is validated by its original M2 compiler before its ports are
connected. A single tensor runtime then owns weights, KV versions, resources,
environment and clock. Trace-supplied token gates do not compute sampling or
embedding, and declared prices are never extrapolated across token/history axes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from ..tensor.engine import (Action, ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY,
    ACTION_EVICT, ACTION_READ_SSD, ACTION_WRITE_SSD)
from ..tensor.spec import (InvalidInput, Unsupported, build_scenario, OP_DONE,
    ROLE_WEIGHT, ROLE_INTERMEDIATE, COPY_READY, ROUTE_ENDS)
from ..tensor.runtime import create_runtime
from .causal import CausalM2Session, CausalObservation, PublicExpert, RevealedRoute
from .expand import CompiledInvocation, CompiledTask, load_and_compile
from .costs import MacroShape
from .spec import weight_id

FORMAT_VERSION = "global-execution-0.1"
VISIBILITY_MODES = ("scope_barriers", "data_dependencies")


@dataclass(frozen=True)
class InvocationScope:
    id: str
    round_id: str
    layer_id: str
    phase: str
    operation_ids: tuple[str, ...]
    static_operation_ids: tuple[str, ...]
    router_operation_ids: tuple[str, ...]
    visible_tensor_ids: tuple[str, ...]
    hidden_outputs: tuple[str, ...]
    incoming_hidden: tuple[str, ...]
    dependency_operations: tuple[str, ...]
    routes: tuple[tuple[str, Any], ...]
    execution_variant: str = "whole_request"


@dataclass(frozen=True)
class CompiledGlobalExecution:
    document: dict
    aggregate: CompiledInvocation
    invocation_scopes: tuple[InvocationScope, ...]
    sources: tuple[dict, ...]
    gate_operations: tuple[tuple[str, tuple[str, ...]], ...]
    visibility_mode: str = "scope_barriers"

    @property
    def scenario(self):
        return self.aggregate.scenario


@dataclass(frozen=True)
class GlobalObservation(CausalObservation):
    release_candidates: tuple[tuple[str, str], ...] = ()
    completed_invocations: tuple[str, ...] = ()


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise InvalidInput(f"{label} must contain exactly {sorted(keys)}")
    return value


def _identity(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in (":", "\n", "\r")):
        raise InvalidInput(f"{label} must be a nonempty identity without colon or newline")
    return value


def _read(path, sources):
    path=Path(path).resolve(); raw=path.read_bytes()
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise InvalidInput(f"duplicate JSON key {key!r} in {path}")
            result[key]=value
        return result
    try:
        document=json.loads(raw.decode("utf-8"),object_pairs_hook=pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(InvalidInput(f"nonfinite JSON {value}")))
    except (ValueError, UnicodeError) as exc:
        raise InvalidInput(f"invalid JSON in {path}: {exc}") from exc
    row=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
    old=sources.get(str(path))
    if old is not None and old!=row:raise InvalidInput(f"source changed while compiling: {path}")
    sources[str(path)]=row
    return document


def verify_global_sources(compiled):
    for row in compiled.sources:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()!=row["sha256"]:
            raise InvalidInput(f"global source changed after compile: {row['path']}")


def compile_global_execution_file(path: str | Path, *, execution_overrides=(),
                                  visibility_mode="scope_barriers") -> CompiledGlobalExecution:
    """Compile finite model ports with optional scope-bound mapping choices.

    Each override is ``(invocation_scope_id, parsed_execution_spec)``. It
    replaces that scope's legacy ``execution_file``; the caller owns hashing
    the override source. The already validated model invocation remains the
    sole source of layer/shape/KV data, including for pure attention. Legacy
    spec ``invocation_files`` are not opened as alternate model inputs.

    ``scope_barriers`` preserves the legacy visibility contract. Explicit
    ``data_dependencies`` exposes same-round static modules together, leaving
    real input producers to constrain their execution. Round token gates and
    MoE router revelation are retained in both modes.
    """
    if visibility_mode not in VISIBILITY_MODES:
        raise Unsupported(f"global visibility_mode must be one of {VISIBILITY_MODES}")
    if not isinstance(execution_overrides, tuple):
        raise InvalidInput("execution_overrides must be an explicit tuple of scope/spec pairs")
    overrides={}
    for binding in execution_overrides:
        if not isinstance(binding, tuple) or len(binding)!=2:
            raise InvalidInput("execution override must be a (scope_id, parsed_execution_spec) pair")
        sid=_identity(binding[0],"execution override scope id")
        if sid in overrides:raise InvalidInput("duplicate execution override scope")
        overrides[sid]=binding[1]
    source_path=Path(path).resolve(); sources={}
    document=_read(source_path,sources)
    _exact(document,{"format_version","id","rounds"},"global_execution")
    if document["format_version"]!=FORMAT_VERSION:raise Unsupported("unsupported global execution version")
    _identity(document["id"],"global_execution.id")
    if not isinstance(document["rounds"],list) or not document["rounds"]:
        raise InvalidInput("global rounds must be a nonempty explicit list")
    rounds=[]; scope_ids=set(); round_ids=set(); all_compiled=[]
    for index,raw in enumerate(document["rounds"]):
        _exact(raw,{"id","input_mode","gate_cost","invocations"},f"rounds[{index}]")
        rid=_identity(raw["id"],"round.id")
        if rid in round_ids:raise InvalidInput("duplicate round id")
        round_ids.add(rid)
        expected="initial_supplied_hidden" if index==0 else "trace_supplied_next_token"
        if raw["input_mode"]!=expected:raise Unsupported(f"round {index} requires input_mode={expected!r}")
        if index==0:
            if raw["gate_cost"] is not None:raise InvalidInput("initial round gate_cost must be null")
        else:
            cost=_exact(raw["gate_cost"],{"device","duration_ns","workspace_bytes"},"gate_cost")
            if cost["device"] not in ("cpu","gpu"):raise Unsupported("gate device must be cpu/gpu")
            for field,minimum in (("duration_ns",1),("workspace_bytes",0)):
                if type(cost[field]) is not int or cost[field]<minimum:raise InvalidInput(f"gate_cost.{field} must be integer >= {minimum}")
        if not isinstance(raw["invocations"],list) or not raw["invocations"]:raise InvalidInput("round invocations must be nonempty")
        leaves=[]
        for entry in raw["invocations"]:
            _exact(entry,{"id","invocation_file","execution_file"},"global invocation")
            sid=_identity(entry["id"],"invocation scope id")
            if sid in scope_ids:raise InvalidInput("global invocation scope IDs must be unique")
            scope_ids.add(sid)
            if not isinstance(entry["invocation_file"],str) or not entry["invocation_file"]:raise InvalidInput("invocation_file must be explicit path")
            ipath=(source_path.parent/entry["invocation_file"]).resolve()
            idoc=_read(ipath,sources)
            from .spec import parse_invocation_document,parse_layer_document
            from .costs import parse_cost_table
            from .expand import compile_invocation
            invocation=parse_invocation_document(idoc,where=str(ipath))
            if invocation.layer_file is None:raise InvalidInput("global invocation requires layer_file")
            layer_path=ipath.parent/invocation.layer_file
            layer=parse_layer_document(_read(layer_path,sources),where=str(layer_path))
            cost_doc=invocation.raw_cost_table if invocation.raw_cost_table is not None else _read(ipath.parent/invocation.costs_file,sources)
            compiled=compile_invocation(layer,invocation,parse_cost_table(cost_doc,where=str(ipath)+".costs"),where=str(ipath))
            if entry["execution_file"] is not None:
                if not isinstance(entry["execution_file"],str) or not entry["execution_file"]:raise InvalidInput("execution_file must be null or explicit path")
            execution=overrides.get(sid)
            if sid not in overrides and entry["execution_file"] is not None:
                edoc=_read(source_path.parent/entry["execution_file"],sources)
                from .execution_pipeline import parse_transform
                execution=parse_transform(edoc)
            if execution is not None:
                from .execution_pipeline import apply_execution_transform
                compiled=apply_execution_transform(compiled,execution)
            elif sid in overrides:
                raise InvalidInput("execution override cannot be null; omit a scope to retain its model binding")
            if compiled.layer.ffn=="moe" and compiled.invocation.information_mode!="causal_replay":
                raise Unsupported("global MoE refuses offline_oracle inputs")
            leaves.append((sid,compiled));all_compiled.append(compiled)
        rounds.append((raw,leaves))
    if set(overrides)-scope_ids:raise InvalidInput("execution override references an absent invocation scope")
    return _combine(document,rounds,sources,visibility_mode=visibility_mode)


def _combine(document,rounds,sources,*,visibility_mode="scope_barriers"):
    first=rounds[0][1][0][1]
    request_ids=tuple(request.request_id for request in first.invocation.requests)
    layer_order=tuple(compiled.layer.id for _,compiled in rounds[0][1])
    if len(set(layer_order))!=len(layer_order):raise InvalidInput("each round must invoke each layer once")
    hardware=first.scenario_document["architecture"]
    tensors={}; operations=[]; tasks=[]; compute={}; transfer={route:{} for route in ROUTE_ENDS}
    allowed={route:set() for route in ROUTE_ENDS}; goals={}; kv_ids=set(); weights=set()
    scopes=[]; gate_rows=[]; stage_metadata={}; row_scopes={}; subweight_catalog={}; weight_owners={}
    latest_blocks={}; layer_signatures={}; parent_versions={}; previous_round_hidden=None
    previous_scope_ops=(); hidden_uses={}; tensor_owners={}; expert_catalog={}; final_hidden_goals={}
    global_weight_children={}; global_weight_partitions={}; pipeline_bindings=[]

    def add_tensor(entry):
        tid=entry["id"]
        if tid in tensors:
            a,b=deepcopy(tensors[tid]),deepcopy(entry)
            # A future declared KV location is a requested starting condition
            # of a standalone call; it is not a new copy in this global session.
            if tid in kv_ids or entry["role"]==ROLE_WEIGHT:
                b.pop("initial_locations",None);a.pop("initial_locations",None)
            if a!=b:raise InvalidInput(f"persistent tensor definition changed: {tid}")
            return
        tensors[tid]=entry

    def add_price(route,tid,price):
        previous=transfer[route].get(tid)
        if previous is not None and previous!=price:raise InvalidInput(f"conflicting exact transfer costs for {tid}/{route}")
        transfer[route][tid]=deepcopy(price);allowed[route].add(tid)

    for round_index,(round_doc,leaves) in enumerate(rounds):
        if tuple(compiled.layer.id for _,compiled in leaves)!=layer_order:raise InvalidInput("layer order must remain identical across rounds")
        previous_layer_hidden=None
        round_gates=()
        for layer_index,(sid,leaf) in enumerate(leaves):
            if leaf.scenario_document["architecture"]!=hardware:raise InvalidInput("all invocations must share one hardware contract")
            if tuple(r.request_id for r in leaf.invocation.requests)!=request_ids:raise InvalidInput("global request order/set must be identical")
            if round_index and any(r.tokens!=1 for r in leaf.invocation.requests):raise Unsupported("trace_supplied_next_token supports decode q=1 only")
            if round_index and leaf.invocation.phase!="decode":raise Unsupported("later rounds must be decode")
            if layer_index and any(a.tokens!=b.tokens for a,b in zip(leaf.invocation.requests,leaves[0][1].invocation.requests)):
                raise InvalidInput("all layers of a round must have matching token counts")
            if layer_index and leaf.invocation.phase!=leaves[0][1].invocation.phase:
                raise InvalidInput("all layers of a round must have matching phase")
            if layer_index and any(a.position_start!=b.position_start for a,b in zip(leaf.invocation.requests,leaves[0][1].invocation.requests)):
                raise InvalidInput("all layers of a round must have matching absolute token positions")
            layer_key=leaf.layer.id
            old_signature=layer_signatures.get(layer_key)
            if old_signature is not None and old_signature!=leaf.layer:raise InvalidInput(f"layer descriptor changed across rounds: {layer_key}")
            layer_signatures[layer_key]=leaf.layer
            for copy in leaf.state_in.weight_copies:
                key=weight_id(layer_key,copy.id)
                previous=parent_versions.setdefault(key,copy.version)
                if previous!=copy.version:raise Unsupported("weight mutation across global rounds requires an explicit update mechanism")
            for request in leaf.invocation.requests:
                state=leaf.state_in.for_request(request.request_id)
                key=(layer_key,request.request_id)
                if key in latest_blocks:
                    old_end,expected,expected_version=latest_blocks[key]
                    actual=tuple((block.id,block.first_position,block.tokens,block.storage_bytes) for block in state.blocks)
                    if state.next_position!=old_end or actual!=expected or state.version!=expected_version:
                        raise InvalidInput(f"KV seam mismatch for {key}: exact contiguous prior block identity/shape required")
                blocks=(*state.blocks,*leaf.new_blocks[request.request_id])
                latest_blocks[key]=(request.position_start+request.tokens,
                    tuple((block.id,block.first_position,block.tokens,block.storage_bytes) for block in blocks),state.version+1)
                kv_ids.update(block.id for block in blocks)
            remap={}
            incoming={};scope_hidden={};hidden_input_ids=set()
            for tensor in leaf.scenario.workload.tensors:
                if tensor.role==ROLE_WEIGHT or tensor.id in kv_ids:remap[tensor.id]=tensor.id
                else:remap[tensor.id]=f"g:{sid}:{tensor.id}"
            for request in leaf.invocation.requests:
                candidates=[tid for tid in remap if tid in (f"t:{layer_key}:{request.request_id}:in",f"t:{layer_key}:{request.request_id}:in.hidden")]
                if len(candidates)!=1:raise Unsupported("global frontends require a unique explicit request hidden input port")
                tid=candidates[0];hidden_input_ids.add(tid)
                if previous_layer_hidden is not None:
                    source=previous_layer_hidden[request.request_id]
                    expected=leaf.scenario.workload.tensor_by_id[tid].storage_bytes
                    source_entry=tensors[source]
                    input_entry=leaf.scenario.workload.tensor_by_id[tid]
                    if source_entry["storage_bytes"]!=expected or source_entry["dtype"]!=input_entry.dtype or tuple(source_entry["ne"])!=input_entry.ne or tuple(source_entry["nb"])!=input_entry.nb:
                        raise InvalidInput("layer hidden port physical shapes/layouts differ")
                    remap[tid]=source
                incoming[request.request_id]=remap[tid]
                scope_hidden[request.request_id]=remap[leaf.hidden_out_ids[request.request_id]]
            for original in leaf.workload_document["tensors"]:
                entry=deepcopy(original); old=entry["id"];entry["id"]=remap[old];entry["name"]=entry["id"]
                if old in hidden_input_ids and previous_layer_hidden is not None:continue
                if old in tensors and old in kv_ids:continue
                if round_index and layer_index==0 and entry["id"] in incoming.values():
                    entry.pop("initial_locations",None);entry["role"]=ROLE_INTERMEDIATE
                add_tensor(entry)
                if entry["role"]==ROLE_WEIGHT:
                    weights.add(entry["id"])
                    previous_owner=weight_owners.setdefault(entry["id"],layer_key)
                    if previous_owner!=layer_key:raise InvalidInput("one immutable weight identity cannot belong to two layers")
                elif entry["id"] not in kv_ids:tensor_owners[entry["id"]]=sid
            if round_index and layer_index==0:
                gates=[]
                for request in request_ids:
                    oid=f"g:{round_doc['id']}:trace_next:{request}"
                    source=previous_round_hidden[request];output=incoming[request]
                    operations.append(dict(id=oid,semantic_op="state_update",ggml_op="TRACE_SUPPLIED_NEXT_TOKEN",
                                           unary_op=None,op_params=[],inputs=[source],output=output))
                    gate=round_doc["gate_cost"]
                    compute[oid]={gate["device"]:dict(duration_ns=gate["duration_ns"],workspace_bytes=gate["workspace_bytes"])}
                    tasks.append(CompiledTask(request,"trace_next_token","state_update",oid,gate["device"],
                                 MacroShape("decode",leaf.invocation.batch,1,0),output,None,()))
                    stage_metadata[oid]=dict(layer_id=layer_key,invocation_id=sid,execution_stage="trace_supplied_next_token")
                    gate_rows.append((oid,previous_scope_ops));gates.append(oid)
                    hidden_uses.setdefault(source,set()).add(sid)
                round_gates=tuple(gates)
            opmap={op.id:f"g:{sid}:{op.id}" for op in leaf.scenario.workload.operations}
            producer={tid:op.id for op in leaf.scenario.workload.operations for tid in op.outputs}
            for old in leaf.workload_document["operations"]:
                entry=deepcopy(old);entry["id"]=opmap[old["id"]]
                entry["inputs"]=[remap[tid] for tid in old["inputs"]];entry["output"]=remap[old["output"]]
                if "extra_outputs" in entry:entry["extra_outputs"]=[remap[tid] for tid in old["extra_outputs"]]
                operations.append(entry)
                compute[entry["id"]]=deepcopy(leaf.scenario_document["costs"]["compute"][old["id"]])
            for task in leaf.tasks:
                tasks.append(replace(task,operation_id=opmap[task.operation_id],output_tensor=remap[task.output_tensor],
                    weight_id=remap[task.weight_id] if task.weight_id else None,
                    depends_on=tuple(opmap[producer[tid]] for tid in leaf.scenario.workload.operation_by_id[task.operation_id].inputs if tid in producer)))
                stage_metadata[opmap[task.operation_id]]=dict(layer_id=layer_key,invocation_id=sid)
            boundary=leaf.physical_boundary or {}
            row=boundary.get("row_execution",{})
            attention=boundary.get("attention_execution",{})
            from .execution_pipeline import execution_metadata, weight_children, STATIC_TASK_KINDS
            local_stages, local_rows=execution_metadata(boundary)
            for tid,scope in local_rows.items():
                if tid in remap:row_scopes[remap[tid]]=deepcopy(scope)
            for oid,metadata in local_stages.items():
                if oid not in opmap:continue
                staged=deepcopy(metadata)
                if "region_id" in staged:staged["region_id"]=f"{sid}:{staged['region_id']}"
                stage_metadata[opmap[oid]].update(staged)
            for parent,children in weight_children(boundary).items():
                old=global_weight_children.setdefault(parent,list(children))
                if old!=list(children):raise InvalidInput("persistent weight partition changed across rounds")
            for part in boundary.get("ffn_execution",{}).get("weight_partitions",()):
                tid=part["tensor_id"]
                old=global_weight_partitions.setdefault(tid,deepcopy(part))
                if old!=part:raise InvalidInput("persistent weight partition geometry changed across rounds")
            if boundary.get("execution_pipeline"):
                pipeline_bindings.append(dict(invocation_id=sid,pipeline=deepcopy(boundary["execution_pipeline"])))
            for part in row.get("subweights",[]):
                previous=subweight_catalog.get(part["parent_weight_id"])
                if previous is not None and previous!=part:raise InvalidInput("subweight manifest changed across rounds")
                subweight_catalog[part["parent_weight_id"]]=deepcopy(part)
            for route,costs in leaf.scenario_document["costs"]["transfer"].items():
                for tid,price in costs.items():add_price(route,remap[tid],price)
            # Catalogue prefetch is public even for inactive experts. Exact
            # costs must exist, rather than being inferred from an active peer.
            for tensor in leaf.scenario.workload.tensors:
                if tensor.role!=ROLE_WEIGHT:continue
                for route in ROUTE_ENDS:
                    price=leaf.cost_table.transfers.get((route,tensor.storage_bytes))
                    if price is not None:add_price(route,tensor.id,dict(duration_ns=price.duration_ns))
            for tid,location in leaf.scenario_document["goal"]["output_locations"].items():
                if tid in kv_ids or leaf.scenario.workload.tensor_by_id[tid].role==ROLE_WEIGHT:
                    previous=goals.get(tid)
                    if previous is not None and previous!=location:raise InvalidInput("persistent goal location changed")
                    goals[tid]=location
            # A module boundary need not wait for every request in the previous
            # layer. In the opt-in mode, actual hidden-port producer edges gate
            # downstream work. Cross-round token gates and private MoE router
            # revelation remain unchanged; this does not enable oracle work.
            if visibility_mode=="data_dependencies":
                dependencies=round_gates
            else:
                dependencies=round_gates if round_index and layer_index==0 else previous_scope_ops if layer_index else ()
            static=tuple(opmap[t.operation_id] for t in leaf.tasks if leaf.layer.ffn!="moe" or t.kind in STATIC_TASK_KINDS)
            routers=tuple(opmap[t.operation_id] for t in leaf.tasks if t.kind=="router")
            routes=[]
            for entry in leaf.invocation.route_trace:
                router=next(t for t in leaf.tasks if t.kind=="router" and t.request_id==entry.request_id)
                routes.append((opmap[router.operation_id],entry))
            scope=InvocationScope(sid,round_doc["id"],layer_key,leaf.invocation.phase,tuple(opmap.values()),
                static,routers,tuple(set(remap.values())),tuple(scope_hidden.values()),tuple(incoming.values()),
                tuple(dependencies),tuple(routes),
                "execution_pipeline" if boundary.get("execution_pipeline") else
                attention.get("execution_spec",{}).get("variant",row.get("expert_variant","whole_request")))
            scopes.append(scope)
            if leaf.layer.ffn=="moe":
                for index,name in enumerate(leaf.layer.ffn_params["expert_weights"]):
                    expert_catalog[(layer_key,index)]=dict(layer_id=layer_key,expert_id=index,parent_weight_id=weight_id(layer_key,name))
                shared=leaf.layer.ffn_params.get("shared_expert_weight")
                if shared is not None:expert_catalog[(layer_key,"shared")]=dict(layer_id=layer_key,expert_id="shared",parent_weight_id=weight_id(layer_key,shared))
            if previous_layer_hidden is not None:
                for tid in previous_layer_hidden.values():hidden_uses.setdefault(tid,set()).add(sid)
            previous_layer_hidden=scope_hidden;previous_scope_ops=scope.operation_ids
            final_hidden_goals={scope_hidden[request.request_id]:request.output_location for request in leaf.invocation.requests}
        previous_round_hidden=previous_layer_hidden
    goals.update(final_hidden_goals)
    # All per-invocation hidden outputs are ports; intermediate layer ports can
    # be released once the next declared scope closes, never via hidden routes.
    hidden_ports={tid for scope in scopes for tid in scope.hidden_outputs}
    wd=dict(schema_version=first.workload_document["schema_version"],id=document["id"],
            origin=dict(kind="synthetic_layer",ggml_version=None,sample="finite connected M2 invocations; explicit trace token gates"),
            tensors=list(tensors.values()),operations=operations,outputs=list(goals))
    sd=deepcopy(first.scenario_document);sd["id"]=document["id"]
    sd["goal"]={"output_locations":goals}
    sd["costs"]={"source":"global execution exact leaf prices; trace gate declared control cost",
                 "compute":compute,"transfer":{route:costs for route,costs in transfer.items() if costs}}
    sd["mapspace"].update(allowed_devices={oid:list(prices) for oid,prices in compute.items()},
        allowed_transfers={route:sorted(ids) for route,ids in allowed.items() if ids},transfer_dependencies={})
    scenario=build_scenario(wd,sd,where="global_execution")
    global_boundary=dict(row_execution=dict(tensor_row_scopes=row_scopes,operation_stages=stage_metadata,
                            subweights=list(subweight_catalog.values())),
        ffn_execution=dict(weight_children=global_weight_children,weight_partitions=list(global_weight_partitions.values())),
        execution_pipelines=pipeline_bindings,
        global_execution=dict(format_version=FORMAT_VERSION,invocation_scope_ids=[s.id for s in scopes],
            execution_bindings=[dict(invocation_id=s.id,variant=s.execution_variant) for s in scopes],
            kv_ids=sorted(kv_ids),weight_ids=sorted(weights),weight_owners=weight_owners,hidden_ports=sorted(hidden_ports),
            tensor_owners=tensor_owners,hidden_port_successors={tid:sorted(ids) for tid,ids in hidden_uses.items()},
            experts=list(expert_catalog.values()),declared_parent_versions=parent_versions,
            numeric_sampling=False,numeric_embedding=False,global_performance_calibrated=False,
            visibility_mode=visibility_mode,
            future_standalone_initial_placements="validated_leaf_inputs_only_not_reapplied_to_persistent_runtime",
            continuation="one_runtime_no_clock_inventory_or_KV_reset"))
    aggregate=replace(first,scenario=scenario,workload_document=wd,scenario_document=sd,tasks=tuple(tasks),
                      physical_boundary=global_boundary,costs_source=sd["costs"]["source"])
    result=CompiledGlobalExecution(deepcopy(document),aggregate,tuple(scopes),tuple(sources.values()),tuple(gate_rows),visibility_mode)
    verify_global_sources(result)
    return result


class _GlobalVisibility:
    def __init__(self,compiled):
        self.compiled=compiled
        self.ops=compiled.scenario.workload.operation_by_id
        self.weights=tuple(t.id for t in compiled.scenario.workload.tensors if t.role==ROLE_WEIGHT)
    def opened(self,scope,state):return all(state.op(oid)==OP_DONE for oid in scope.dependency_operations)
    def closed(self,scope,state):return all(state.op(oid)==OP_DONE for oid in scope.operation_ids)
    def visible_operation_ids(self,state):
        result=[]
        for oid,dependencies in self.compiled.gate_operations:
            if all(state.op(d)==OP_DONE for d in dependencies):result.append(oid)
        for scope in self.compiled.invocation_scopes:
            if self.opened(scope,state):
                result.extend(scope.operation_ids if all(state.op(o)==OP_DONE for o in scope.router_operation_ids) else scope.static_operation_ids)
        return tuple(result)
    def visible_tensor_ids(self,state):
        result=set(self.weights)
        for oid in self.visible_operation_ids(state):
            operation=self.ops[oid];result.update(operation.inputs);result.update(operation.outputs)
        return tuple(sorted(result))
    def task_graph_complete(self,state):return len(self.visible_operation_ids(state))==len(self.ops)
    def revealed_routes(self,state):
        return tuple(RevealedRoute(row.request_id,row.layer_id,row.absolute_position,row.expert_ids,row.scores)
                     for scope in self.compiled.invocation_scopes for oid,row in scope.routes if state.op(oid)==OP_DONE)
    def allow_action(self,action,state):
        if action.kind==ACTION_ADVANCE:return True
        if action.kind==ACTION_COMPUTE:return action.target_id in self.visible_operation_ids(state)
        if action.kind in (ACTION_COPY,ACTION_READ_SSD,ACTION_WRITE_SSD,ACTION_EVICT):return action.target_id in self.visible_tensor_ids(state)
        return False


class GlobalM2Session(CausalM2Session):
    def __init__(self,compiled:CompiledGlobalExecution,*,environment_events=(),services=None,routing_scores=(),
                 reclaim_requests=(),control_costs=None,visibility_mode=None):
        if not isinstance(compiled,CompiledGlobalExecution):raise InvalidInput("GlobalM2Session requires CompiledGlobalExecution")
        if compiled.visibility_mode not in VISIBILITY_MODES:
            raise Unsupported("compiled global visibility_mode is unsupported")
        if visibility_mode is not None and visibility_mode!=compiled.visibility_mode:
            raise InvalidInput("session visibility_mode must match its compiled global model")
        verify_global_sources(compiled)
        self.global_compiled=compiled
        from .routing_scores import validate_routing_score_rows
        routes=tuple(row for scope in compiled.invocation_scopes for _,row in scope.routes)
        counts={}
        for row in compiled.aggregate.physical_boundary["global_execution"]["experts"]:
            if type(row["expert_id"]) is int:counts[row["layer_id"]]=max(counts.get(row["layer_id"],0),row["expert_id"]+1)
        self._score_rows=validate_routing_score_rows(routes,counts,routing_scores)
        self._compiled=compiled.aggregate;self._execution_spec=None
        self._scenario_document=deepcopy(compiled.aggregate.scenario_document)
        self._visibility=_GlobalVisibility(compiled)
        self._runtime=create_runtime(compiled.scenario,services=services,environment_events=environment_events,
                                     visibility=self._visibility,reclaim_requests=reclaim_requests,control_costs=control_costs)
        from .execution_pipeline import materialized_weight_members
        available={tensor.id for tensor in compiled.scenario.workload.tensors if tensor.role==ROLE_WEIGHT}
        catalog=[]
        # The catalogue is static model architecture, not route disclosure.
        for row in compiled.aggregate.physical_boundary["global_execution"]["experts"]:
            parent=row["parent_weight_id"]
            children=materialized_weight_members(compiled.aggregate.physical_boundary,parent,available)
            catalog.append(PublicExpert(row["expert_id"],parent if parent in available else None,row["layer_id"],children))
        self._experts=tuple(catalog)
    def observe(self):
        base=super().observe()
        metadata=self._compiled.physical_boundary["row_execution"]["operation_stages"]
        operations=tuple(replace(op,layer_id=metadata[op.id]["layer_id"],invocation_id=metadata[op.id]["invocation_id"])
                         for op in base.operations)
        state=self._runtime.state
        closed={scope.id for scope in self.global_compiled.invocation_scopes if self._visibility.closed(scope,state)}
        boundary=self._compiled.physical_boundary["global_execution"]
        protected=set(boundary["kv_ids"])|set(boundary["weight_ids"])|set(self._runtime.scenario.goal.output_locations)
        candidates=[]
        for tensor in base.tensors:
            if tensor.id in protected:continue
            owner=boundary["tensor_owners"].get(tensor.id)
            if owner not in closed:continue
            if any(scope not in closed for scope in boundary["hidden_port_successors"].get(tensor.id,())):continue
            candidates.extend((tensor.id,location) for location in ("dram","vram")
                              if tensor.status_at(location)==COPY_READY)
        observation=GlobalObservation(**{**base.__dict__,"operations":operations},release_candidates=tuple(sorted(candidates)),
                                      completed_invocations=tuple(scope.id for scope in self.global_compiled.invocation_scopes if scope.id in closed))
        from .routing_scores import reveal_routing_scores
        return reveal_routing_scores(observation,self._score_rows)
    def state_out(self):
        raise Unsupported("global execution exports persistent_state_document, not a single-layer state_out")
    def execution_document(self):
        result=super().execution_document()
        result.update(global_execution=deepcopy(self.global_compiled.document),
                      global_sources=deepcopy(list(self.global_compiled.sources)),
                      visibility_mode=self.global_compiled.visibility_mode,
                      scope="finite_connected_M2_invocations_single_runtime")
        return result
    def persistent_state_document(self):
        return dict(format_version="global-persistent-state-0.1",t_ns=self._runtime.state.t,
            tensor_inventory=[dict(tensor_id=t.id,storage_bytes=t.storage_bytes,role=t.role,
                locations=[loc for loc in ("dram","vram","ssd") if self._runtime.state.copy_at(t.id,loc)==COPY_READY])
                for t in self._runtime.scenario.workload.tensors],
            kv_ids=self._compiled.physical_boundary["global_execution"]["kv_ids"],
            completed_invocations=list(self.observe().completed_invocations),
            weight_versions=self._compiled.physical_boundary["row_execution"]["subweights"],
            weight_children=self._compiled.physical_boundary.get("ffn_execution",{}).get("weight_children",{}),
            weight_partitions=self._compiled.physical_boundary.get("ffn_execution",{}).get("weight_partitions",[]),
            declared_parent_versions=self._compiled.physical_boundary["global_execution"]["declared_parent_versions"],
            runtime_state=asdict(self._runtime.state),
            compute_progress=[asdict(row) for row in self._runtime.compute_progress_document()],
            services=self.service_document() if hasattr(self._runtime,"service_document") else None,
            resumable_checkpoint=False,scope="runtime_inventory_not_a_checkpoint")
