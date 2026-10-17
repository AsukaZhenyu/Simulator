"""Materialized selected-row execution variants for synthetic MoE layers.

This is a graph/storage contract, not a numerical kernel or calibrated timing
model. New stages have independent exact prices; no macro price is divided.
Version 0.1 keeps whole expert allocations. Version 0.2 explicitly replaces
them with immutable materialized F32 gate/up/down partitions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..tensor.spec import InvalidInput, Unsupported, ROLE_INTERMEDIATE, build_scenario, ROUTE_ENDS
from .costs import MacroShape, TransferEntry
from .expand import CompiledInvocation, CompiledTask, _tensor
from .spec import op_id, tensor_id

FORMAT_VERSION = "moe-execution-0.1"
SUBWEIGHT_FORMAT_VERSION = "moe-execution-0.2"
NEW_KINDS = frozenset({"gather_rows", "scatter_rows", "combine_rows", "expert_gate",
                       "expert_up", "expert_swiglu", "expert_down"})
_COST_KEYS = {"kind", "device", "phase", "batch", "token_count", "hidden_bytes_per_token",
              "intermediate_bytes_per_token", "duration_ns", "workspace_bytes"}


@dataclass(frozen=True)
class ExecutionCost:
    kind: str
    device: str
    phase: str
    batch: int
    token_count: int
    hidden_bytes_per_token: int
    intermediate_bytes_per_token: int
    duration_ns: int
    workspace_bytes: int

    def key(self):
        return (self.kind, self.device, self.phase, self.batch, self.token_count,
                self.hidden_bytes_per_token, self.intermediate_bytes_per_token)


@dataclass(frozen=True)
class ExecutionTransfer:
    route: str
    storage_bytes: int
    duration_ns: int


@dataclass(frozen=True)
class MoEExecutionSpec:
    format_version: str
    variant: str
    expert_variant: str
    provenance: str
    entries: tuple[ExecutionCost, ...]
    transfer_entries: tuple[ExecutionTransfer, ...]
    index_storage_contract: str
    opaque_cost_source: str | None = None
    intermediate_bytes_per_token: int = 0
    weight_storage_contract: str | None = None
    subweights: tuple = ()

    def to_document(self):
        return execution_spec_to_document(self)


def _object(value, keys, label, *, required=None):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise InvalidInput(f"{label} must be an object with string keys")
    extra = set(value) - keys
    missing = (keys if required is None else required) - set(value)
    if extra or missing:
        raise InvalidInput(f"{label}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def parse_execution_spec(document: Any) -> MoEExecutionSpec:
    keys = {"format_version", "variant", "expert_variant", "provenance", "entries", "transfer_entries",
            "opaque_cost_source", "intermediate_bytes_per_token", "index_storage_contract"}
    version = document.get("format_version") if isinstance(document, dict) else None
    if version == SUBWEIGHT_FORMAT_VERSION:
        keys |= {"weight_storage_contract", "subweights"}
    required = keys - {"opaque_cost_source", "intermediate_bytes_per_token"}
    obj = _object(document, keys, "moe_execution", required=required)
    for field, accepted in (("format_version", {FORMAT_VERSION, SUBWEIGHT_FORMAT_VERSION}), ("variant", {"selected_rows"}),
                            ("expert_variant", {"opaque", "swiglu_staged"}),
                            ("provenance", {"synthetic_execution"}),
                            ("index_storage_contract", {"stage_workspace_includes_row_index_temporaries"})):
        if not isinstance(obj[field], str) or obj[field] not in accepted:
            raise Unsupported(f"moe_execution.{field} must be one of {sorted(accepted)}")
    opaque = obj["expert_variant"] == "opaque"
    partitions = ()
    if version == SUBWEIGHT_FORMAT_VERSION:
        if opaque:
            raise Unsupported("moe-execution-0.2 subweights require swiglu_staged")
        if obj["weight_storage_contract"] != "materialized_partition_replaces_parent":
            raise Unsupported("subweight aliases and duplicate parent allocations are not supported")
        from .subweights import parse_subweights
        partitions = parse_subweights(obj["subweights"])
    if opaque:
        if obj.get("opaque_cost_source") != "existing_synthetic_per_expert_group":
            raise InvalidInput("opaque execution requires explicit existing_synthetic_per_expert_group acknowledgement")
        if "intermediate_bytes_per_token" in obj:
            raise InvalidInput("opaque execution does not accept intermediate_bytes_per_token")
        intermediate = 0
    else:
        if "opaque_cost_source" in obj:
            raise InvalidInput("staged execution does not accept opaque_cost_source")
        intermediate = _integer(obj.get("intermediate_bytes_per_token"), "intermediate_bytes_per_token", 4)
        if intermediate % 4:
            raise Unsupported("synthetic staged intermediates must be contiguous F32 bytes")
    if not isinstance(obj["entries"], list) or not isinstance(obj["transfer_entries"], list):
        raise InvalidInput("entries and transfer_entries must be explicit lists")
    entries, seen = [], set()
    for index, raw in enumerate(obj["entries"]):
        item = _object(raw, _COST_KEYS, f"entries[{index}]")
        allowed = {"gather_rows", "scatter_rows", "combine_rows"} if opaque else NEW_KINDS
        for field, values in (("kind", allowed), ("device", {"cpu", "gpu"}), ("phase", {"prefill", "decode"})):
            if not isinstance(item[field], str) or item[field] not in values:
                raise Unsupported(f"entries[{index}].{field} must be one of {sorted(values)}")
        for field in ("batch", "token_count", "hidden_bytes_per_token", "duration_ns"):
            _integer(item[field], f"entries[{index}].{field}", 1)
        _integer(item["workspace_bytes"], f"entries[{index}].workspace_bytes")
        _integer(item["intermediate_bytes_per_token"], f"entries[{index}].intermediate_bytes_per_token")
        if item["hidden_bytes_per_token"] % 4 or item["intermediate_bytes_per_token"] != intermediate:
            raise InvalidInput("cost row widths must match the explicit contiguous F32 execution shape")
        entry = ExecutionCost(**item)
        if entry.key() in seen:
            raise InvalidInput(f"duplicate execution cost key {entry.key()!r}")
        seen.add(entry.key()); entries.append(entry)
    transfers, seen_transfers = [], set()
    for index, raw in enumerate(obj["transfer_entries"]):
        item = _object(raw, {"route", "storage_bytes", "duration_ns"}, f"transfer_entries[{index}]")
        if not isinstance(item["route"], str) or item["route"] not in ROUTE_ENDS:
            raise Unsupported(f"transfer_entries[{index}].route is unsupported")
        _integer(item["storage_bytes"], f"transfer_entries[{index}].storage_bytes", 1)
        _integer(item["duration_ns"], f"transfer_entries[{index}].duration_ns", 1)
        key = (item["route"], item["storage_bytes"])
        if key in seen_transfers:
            raise InvalidInput(f"duplicate execution transfer key {key!r}")
        seen_transfers.add(key); transfers.append(ExecutionTransfer(**item))
    return MoEExecutionSpec(version, "selected_rows", obj["expert_variant"], "synthetic_execution",
                            tuple(entries), tuple(transfers), obj["index_storage_contract"],
                            obj.get("opaque_cost_source"), intermediate,
                            obj.get("weight_storage_contract"), partitions)


def execution_spec_to_document(spec: MoEExecutionSpec) -> dict[str, Any]:
    if not isinstance(spec, MoEExecutionSpec):
        raise InvalidInput("execution_spec must be a parsed MoEExecutionSpec")
    if not isinstance(spec.entries, tuple) or any(not isinstance(e, ExecutionCost) for e in spec.entries):
        raise InvalidInput("execution_spec.entries must be a tuple of ExecutionCost values")
    if not isinstance(spec.transfer_entries, tuple) or any(not isinstance(e, ExecutionTransfer) for e in spec.transfer_entries):
        raise InvalidInput("execution_spec.transfer_entries must be a tuple of ExecutionTransfer values")
    if spec.expert_variant == "opaque" and (type(spec.intermediate_bytes_per_token) is not int
                                            or spec.intermediate_bytes_per_token != 0):
        raise InvalidInput("opaque typed execution spec must have intermediate_bytes_per_token=0")
    if spec.expert_variant == "swiglu_staged" and spec.opaque_cost_source is not None:
        raise InvalidInput("staged typed execution spec must not specify opaque_cost_source")
    doc = asdict(spec)
    doc["entries"] = [asdict(entry) for entry in spec.entries]
    doc["transfer_entries"] = [asdict(entry) for entry in spec.transfer_entries]
    if spec.format_version == SUBWEIGHT_FORMAT_VERSION:
        from .subweights import subweights_to_document
        doc["subweights"] = subweights_to_document(spec.subweights)
    else:
        if spec.weight_storage_contract is not None or spec.subweights:
            raise InvalidInput("legacy execution must not silently discard subweight declarations")
        del doc["weight_storage_contract"]
        del doc["subweights"]
    if spec.expert_variant == "opaque":
        del doc["intermediate_bytes_per_token"]
    else:
        del doc["opaque_cost_source"]
    return doc


def transform_moe_execution(compiled: CompiledInvocation, execution_spec: MoEExecutionSpec) -> CompiledInvocation:
    """Expand the model-owned graph; dynamic shapes are revealed only at routing.

    Gather consumes whole request hidden values at its chosen device. Experts
    consume compact grouped values. Scatter produces one compact contribution
    per expert/request. Combine consumes these contributions plus residual and
    router results. No slices are aliases; every new tensor is charged storage.
    """
    spec = parse_execution_spec(execution_spec_to_document(execution_spec))
    if compiled.layer.format_version != "m2-layer-0.6" or compiled.layer.ffn != "moe":
        raise Unsupported("selected rows currently requires synthetic MoE m2-layer-0.6")
    if compiled.physical_boundary and "row_execution" in compiled.physical_boundary:
        raise InvalidInput("execution variant is already materialized")
    if any(price.provenance != "synthetic_layer" for price in compiled.cost_table.entries.values()):
        raise Unsupported("selected-row execution refuses measured macro reinterpretation")
    hidden = compiled.layer.hidden_bytes_per_token
    workload, scenario_doc = deepcopy(compiled.workload_document), deepcopy(compiled.scenario_document)
    old_experts = [t for t in compiled.tasks if t.kind in ("expert_ffn", "shared_expert_ffn")]
    old_combine = [t for t in compiled.tasks if t.kind == "combine"]
    removed_ops = {t.operation_id for t in old_experts + old_combine}
    removed_tensors = {t.output_tensor for t in old_experts + old_combine}
    tensors = [t for t in workload["tensors"] if t["id"] not in removed_tensors]
    stage_weights = {}
    if spec.format_version == SUBWEIGHT_FORMAT_VERSION:
        from .subweights import materialize_subweights
        children, stage_weights = materialize_subweights(compiled, spec.subweights,
                                                         spec.intermediate_bytes_per_token)
        parent_ids = set(stage_weights)
        tensors = [item for item in tensors if item["id"] not in parent_ids] + children
        workload["outputs"] = [tid for tid in workload["outputs"] if tid not in parent_ids]
        goals = scenario_doc["goal"]["output_locations"]
        for parent, child_ids in stage_weights.items():
            destination = goals.pop(parent)
            for child in child_ids.values():
                goals[child] = destination
                workload["outputs"].append(child)
    operations = [op for op in workload["operations"] if op["id"] not in removed_ops]
    tasks = [t for t in compiled.tasks if t.operation_id not in removed_ops]
    old_ops = compiled.scenario.workload.operation_by_id
    attention = {t.request_id: t.output_tensor for t in compiled.tasks if t.kind in ("attention", "attention_output")}
    routers = {t.request_id: t.output_tensor for t in compiled.tasks if t.kind == "router"}
    rows, stages, regions, contributions = {}, {}, [], {}
    compute = {key: value for key, value in scenario_doc["costs"]["compute"].items() if key not in removed_ops}
    price_by_key = {p.key(): p for p in spec.entries}

    def add_task(template, kind, label, inputs, output, work_items, nbytes, region, *, opaque_price=None):
        oid = op_id(compiled.layer.id, template.request_id, label)
        shape = MacroShape(compiled.invocation.phase, compiled.invocation.batch, len(work_items), 0)
        weight_ids = {item["id"] for item in tensors if item["role"] == "weight"}
        weight = next((i for i in inputs if i in weight_ids), None)
        task = CompiledTask(template.request_id, label, kind, oid, template.device, shape, output,
                            weight, (), template.expert_id, tuple(work_items))
        tasks.append(task)
        operations.append(dict(id=oid, semantic_op=kind, ggml_op="M2_" + kind.upper(), unary_op=None,
                               op_params=[], inputs=list(inputs), output=output))
        tensors.append(_tensor(output, ROLE_INTERMEDIATE, nbytes))
        rows[output] = [dict(request_id=req, absolute_position=pos) for req, pos in work_items]
        stages[oid] = dict(region_id=region, execution_stage=kind, execution_variant=spec.expert_variant,
                           expert_id="shared" if template.kind == "shared_expert_ffn" else template.expert_id)
        device_costs = {}
        for device in compiled.invocation.devices:
            if opaque_price is not None:
                p = compiled.cost_table.lookup(template.kind, device, template.shape)
                if p.accounting != "per_expert_group":
                    raise InvalidInput("opaque expert price must be per_expert_group")
            else:
                key = (kind, device, shape.phase, shape.batch, shape.tokens, hidden, spec.intermediate_bytes_per_token)
                p = price_by_key.get(key)
                if p is None:
                    raise Unsupported(f"missing exact execution cost {key!r}; no interpolation or macro splitting")
            device_costs[device] = dict(duration_ns=p.duration_ns, workspace_bytes=p.workspace_bytes)
        compute[oid] = device_costs
        return oid

    # Every expert group is a deterministic ordered set of selected rows.
    for template in old_experts:
        items = tuple(sorted(template.work_items))
        if not items or len(set(items)) != len(items):
            raise InvalidInput("expert rows must be nonempty and unique")
        expert = "shared" if template.kind == "shared_expert_ffn" else template.expert_id
        group = str(expert); region = f"expert:{group}"
        req_ids = tuple(dict.fromkeys(req for req, _ in items))
        prefix = "selected." + group
        compact = tensor_id(compiled.layer.id, template.request_id, prefix + ".input")
        gather = add_task(template, "gather_rows", prefix + ".gather",
                          [*(attention[req] for req in req_ids), *(routers[req] for req in req_ids)],
                          compact, items, len(items) * hidden, region)
        weight = next(i for i in old_ops[template.operation_id].inputs
                      if compiled.scenario.workload.tensor_by_id[i].role == "weight")
        output = tensor_id(compiled.layer.id, template.request_id, prefix + ".output")
        expert_ids = []
        if spec.expert_variant == "opaque":
            expert_ids.append(add_task(template, "expert_ffn", prefix + ".expert", [weight, compact],
                                       output, items, len(items) * hidden, region, opaque_price=True))
        else:
            gate = tensor_id(compiled.layer.id, template.request_id, prefix + ".gate")
            up = tensor_id(compiled.layer.id, template.request_id, prefix + ".up")
            activated = tensor_id(compiled.layer.id, template.request_id, prefix + ".swiglu")
            middle = len(items) * spec.intermediate_bytes_per_token
            stage_ids = stage_weights.get(weight, {stage: weight for stage in ("gate", "up", "down")})
            expert_ids.append(add_task(template, "expert_gate", prefix + ".gate", [stage_ids["gate"], compact], gate, items, middle, region))
            expert_ids.append(add_task(template, "expert_up", prefix + ".up", [stage_ids["up"], compact], up, items, middle, region))
            expert_ids.append(add_task(template, "expert_swiglu", prefix + ".swiglu", [gate, up], activated, items, middle, region))
            expert_ids.append(add_task(template, "expert_down", prefix + ".down", [stage_ids["down"], activated], output, items, len(items)*hidden, region))
        scatter_ids = []
        for req in req_ids:
            selected = tuple(item for item in items if item[0] == req)
            contribution = tensor_id(compiled.layer.id, req, "selected." + group + ".contribution")
            label = prefix + ".scatter." + req
            scatter_ids.append(add_task(template, "scatter_rows", label, [output], contribution,
                                        selected, len(selected) * hidden, region))
            contributions.setdefault(req, []).append(contribution)
        regions.append(dict(expert_id=expert, gather_operation_id=gather,
                            expert_operation_ids=expert_ids, scatter_operation_ids=scatter_ids,
                            grouped_input_tensor=compact, grouped_output_tensor=output,
                            work_items=[dict(request_id=req, absolute_position=pos) for req, pos in items]))
    for template in old_combine:
        req = template.request_id
        request = next(r for r in compiled.invocation.requests if r.request_id == req)
        items = tuple((req, pos) for pos in range(request.position_start, request.position_start+request.tokens))
        inputs = [attention[req], routers[req], *contributions[req]]
        inputs = [i for i in old_ops[template.operation_id].inputs
                  if compiled.scenario.workload.tensor_by_id[i].role == "weight"] + inputs
        add_task(template, "combine_rows", template.task_id, inputs, template.output_tensor, items,
                 request.tokens * hidden, f"request:{req}")
        tensors[-1]["role"] = "output"
    workload["operations"], workload["tensors"] = operations, tensors
    workload["origin"]["sample"] += "; materialized selected rows " + spec.expert_variant
    sizes = {t["id"]:t["storage_bytes"] for t in tensors}
    allowed = scenario_doc["mapspace"]["allowed_transfers"]
    for route in allowed:
        retained = [tid for tid in allowed[route] if tid not in removed_tensors and tid not in stage_weights]
        new_ids = [*rows, *(child for children in stage_weights.values() for child in children.values())]
        allowed[route] = retained + [tid for tid in new_ids if tid not in retained]
    transfers = dict(compiled.cost_table.transfers)
    for p in spec.transfer_entries:
        transfers[(p.route,p.storage_bytes)] = TransferEntry(p.duration_ns,"synthetic_transfer")
    table = replace(compiled.cost_table, transfers=transfers)
    from .costs import expand_transfer_costs
    _, scenario_doc["costs"]["transfer"] = expand_transfer_costs(table, sizes, allowed)
    scenario_doc["costs"]["compute"] = compute
    scenario_doc["costs"]["source"] += ";moe-execution:synthetic_execution"
    scenario_doc["mapspace"]["allowed_devices"] = {t.operation_id:list(compiled.invocation.devices) for t in tasks}
    scenario_doc["mapspace"]["transfer_dependencies"] = {}
    scenario = build_scenario(workload, scenario_doc, where="selected_row_moe")
    boundary = deepcopy(compiled.physical_boundary or {})
    boundary.update(port_semantics="materialized_compact_selected_rows", numeric_execution_verified=False,
                    physical_macro_contract=False, original_model_kernel_reproduced=False,
                    inactive_expert_prefetch_allowed=True)
    boundary["row_execution"] = dict(format_version=spec.format_version, expert_variant=spec.expert_variant,
        hidden_bytes_per_token=hidden, intermediate_bytes_per_token=spec.intermediate_bytes_per_token,
        regions=regions, tensor_row_scopes=rows, operation_stages=stages,
        stage_weight_semantics="whole_expert_weight_required_by_gate_up_down; no aliases or subweight_slices",
        gather_semantics="whole_request_hidden_inputs_to_compact_group_rows",
        scatter_semantics="compact_per_request_contribution_before_weighted_combine",
        route_score_semantics="trace_metadata_only; no numerical weighting_or_default_scores_executed",
        index_storage_contract=spec.index_storage_contract,
        index_validation="row_scope_is_host_control_metadata; physical_index_workspace_unverified_until_profile",
        cost_semantics="independent_exact_synthetic_stage_prices; not_divided_macro_costs")
    if stage_weights:
        from .subweights import subweights_to_document
        boundary["row_execution"].update(
            weight_storage_contract=spec.weight_storage_contract,
            subweights=subweights_to_document(spec.subweights),
            stage_weight_semantics="distinct_immutable_gate_up_down_F32_allocations; parent_removed; no_aliases",
            partition_materialization="initial_representation_only; no_free_runtime_parent_conversion",
            evidence_scope="layout_declaration_not_weight_values",
            numeric_weights_loaded=False, measured_costs_supported=False)
    return replace(compiled, scenario=scenario, scenario_document=scenario_doc, workload_document=workload,
                   tasks=tuple(tasks), cost_table=table, costs_source=scenario_doc["costs"]["source"],
                   physical_boundary=boundary)
