"""Composable materialized causal-GQA stages using the existing tensor runtime.

This module describes exact graph, storage and independent synthetic costs. It
does not execute a numerical kernel or turn an old macro price into stage costs.
Dense .5 supplies real post-RoPE ports. Synthetic MoE .6 requires an explicit
F32 Q/K/V/O packed frontend with sufficient weight storage; its old opaque
projection alone cannot establish those ports.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from ..tensor.spec import (DTYPE_F32, DTYPE_I32, ROLE_INPUT, ROLE_INTERMEDIATE,
                           InvalidInput, Unsupported, build_scenario)
from .attention_execution import AttentionStageCost, AttentionTransfer
from .costs import TransferEntry, expand_transfer_costs
from .expand import CompiledInvocation, CompiledTask, _physical_tensor
from .hybrid_physical import HybridDensePhysical, validate_hybrid_physical_layer
from .spec import op_id, tensor_id, weight_id

FORMAT_VERSION = "attention-stages-0.1"
KINDS = frozenset({"attention_projection", "attention_qk", "attention_softmax",
                   "attention_pv", "attention_output"})
DENSE_FRONTEND = "existing_materialized_post_rope_qkv"
MOE_FRONTEND = "synthetic_f32_qkv_rope_output_pack_v1"


@dataclass(frozen=True)
class AttentionStagesSpec:
    format_version: str
    variant: str
    provenance: str
    geometry: dict[str, Any]
    frontend: dict[str, Any]
    entries: tuple[AttentionStageCost, ...]
    transfer_entries: tuple[AttentionTransfer, ...]

    def to_document(self):
        return attention_stages_document(self)

    @property
    def expert_variant(self):
        return None


def _object(value, keys, label):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value) or set(value) != keys:
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def parse_attention_stages(document: Any) -> AttentionStagesSpec:
    raw = _object(document, {"format_version", "variant", "provenance", "geometry", "frontend",
                             "entries", "transfer_entries"}, "attention_stages")
    for field, expected in (("format_version", FORMAT_VERSION), ("variant", "qk_softmax_pv"),
                            ("provenance", "synthetic_execution")):
        if raw[field] != expected:
            raise Unsupported(f"attention_stages.{field} must be {expected!r}")
    geometry = _object(raw["geometry"], {"hidden", "heads", "kv_heads", "head_dim", "attention_gate"}, "geometry")
    for field in ("hidden", "heads", "kv_heads", "head_dim"):
        _integer(geometry[field], "geometry." + field, 1)
    if geometry["heads"] % geometry["kv_heads"]:
        raise InvalidInput("heads must be divisible by kv_heads")
    if geometry["attention_gate"] not in ("none", "sigmoid"):
        raise Unsupported("attention_gate must be none or sigmoid")
    frontend = raw["frontend"]
    contract = frontend.get("contract") if isinstance(frontend, dict) else None
    if contract == DENSE_FRONTEND:
        _object(frontend, {"contract"}, "frontend")
    elif contract == MOE_FRONTEND:
        _object(frontend, {"contract", "weight_id", "rope_dims", "rope_base"}, "frontend")
        if not isinstance(frontend["weight_id"], str) or not frontend["weight_id"]:
            raise InvalidInput("frontend.weight_id must be a nonempty string")
        dims = _integer(frontend["rope_dims"], "rope_dims", 2)
        if dims % 2 or dims > geometry["head_dim"]:
            raise InvalidInput("rope_dims must be even and <= head_dim")
        base = frontend["rope_base"]
        if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 1:
            raise InvalidInput("rope_base must be finite and > 1")
        if geometry["attention_gate"] != "none":
            raise Unsupported("the explicit MoE frontend has no gate projection")
    else:
        raise Unsupported("attention stages require an explicit materialized frontend contract")
    entries, seen = [], set()
    if not isinstance(raw["entries"], list) or not isinstance(raw["transfer_entries"], list):
        raise InvalidInput("entries and transfer_entries must be explicit arrays")
    fields = {"kind", "device", "phase", "batch", "tokens", "visible_history", "duration_ns", "workspace_bytes"}
    for row in raw["entries"]:
        _object(row, fields, "stage cost")
        if not isinstance(row["kind"], str) or row["kind"] not in KINDS or \
                row["device"] not in ("cpu", "gpu") or row["phase"] not in ("prefill", "decode"):
            raise Unsupported("unsupported attention stage kind/device/phase")
        for field in ("batch", "tokens", "duration_ns"):
            _integer(row[field], field, 1)
        for field in ("visible_history", "workspace_bytes"):
            _integer(row[field], field)
        if row["phase"] == "decode" and row["tokens"] != 1:
            raise InvalidInput("decode stage must have one new token")
        entry = AttentionStageCost(**row)
        if entry.key() in seen:
            raise InvalidInput("duplicate exact attention stage cost")
        entries.append(entry); seen.add(entry.key())
    transfers, seen = [], set()
    for row in raw["transfer_entries"]:
        _object(row, {"route", "storage_bytes", "duration_ns"}, "transfer cost")
        if row["route"] not in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd"):
            raise Unsupported("unknown attention transfer route")
        _integer(row["storage_bytes"], "storage_bytes", 1)
        _integer(row["duration_ns"], "duration_ns", 1)
        key = (row["route"], row["storage_bytes"])
        if key in seen:
            raise InvalidInput("duplicate attention transfer cost")
        transfers.append(AttentionTransfer(**row)); seen.add(key)
    return AttentionStagesSpec(FORMAT_VERSION, "qk_softmax_pv", "synthetic_execution",
                               deepcopy(geometry), deepcopy(frontend), tuple(entries), tuple(transfers))


def attention_stages_document(spec: AttentionStagesSpec):
    if not isinstance(spec, AttentionStagesSpec) or not isinstance(spec.entries, tuple) or \
            any(not isinstance(e, AttentionStageCost) for e in spec.entries) or \
            not isinstance(spec.transfer_entries, tuple) or any(not isinstance(e, AttentionTransfer) for e in spec.transfer_entries):
        raise InvalidInput("expected typed AttentionStagesSpec and typed price tuples")
    return dict(format_version=spec.format_version, variant=spec.variant, provenance=spec.provenance,
                geometry=deepcopy(spec.geometry), frontend=deepcopy(spec.frontend),
                entries=[asdict(e) for e in spec.entries], transfer_entries=[asdict(e) for e in spec.transfer_entries])


def _frontend(compiled, spec):
    g, frontend, layer = spec.geometry, spec.frontend, compiled.layer
    if layer.attention != "full_causal" or layer.hidden_size != g["hidden"] or layer.hidden_bytes_per_token != 4 * g["hidden"]:
        raise InvalidInput("attention frontend hidden/causal geometry differs from layer")
    if layer.kv_bytes_per_token != 8 * g["kv_heads"] * g["head_dim"]:
        raise InvalidInput("KV bytes must equal F32 K plus V for the exact GQA geometry")
    if frontend["contract"] == DENSE_FRONTEND:
        if layer.format_version != "m2-layer-0.5" or not isinstance(layer.physical, HybridDensePhysical):
            raise Unsupported("existing QKV frontend requires complete hybrid Dense .5")
        validate_hybrid_physical_layer(layer)
        if any(layer.physical.config[name] != value for name, value in g.items()):
            raise InvalidInput("attention geometry differs from the materialized Dense frontend")
        return None
    if layer.format_version != "m2-layer-0.6" or layer.ffn != "moe":
        raise Unsupported("packed QKV/O frontend currently requires synthetic MoE .6")
    if any(price.provenance != "synthetic_layer" for price in compiled.cost_table.entries.values()):
        raise Unsupported("packed frontend cannot reinterpret measured macro costs")
    projection = layer.tasks_of_kind("projection")[0]
    if projection.weight != frontend["weight_id"]:
        raise InvalidInput("frontend weight_id must name the existing projection pack")
    pack = layer.weight_by_id(frontend["weight_id"])
    # q, k, v and output matrices, contiguous F32; no hidden norm/gate weights.
    required = 4 * g["hidden"] * g["head_dim"] * (2 * g["heads"] + 2 * g["kv_heads"])
    if pack.storage_bytes != required:
        raise InvalidInput(f"QKV/O pack requires exactly {required} bytes; opaque projection has {pack.storage_bytes}")
    return weight_id(layer.id, pack.id)


def transform_attention_stages(compiled: CompiledInvocation, execution_spec: AttentionStagesSpec) -> CompiledInvocation:
    """Lower one supported attention macro without altering other refinements.

    All stage costs apply to this spec's exact frontend/geometry and retain the
    source accounting batch. Requests remain separate kernels, not fused batch.
    PV writes appended KV blocks once, from the declared new K and V ports.
    """
    if not isinstance(compiled, CompiledInvocation):
        raise InvalidInput("attention stages require a validated CompiledInvocation")
    spec = parse_attention_stages(attention_stages_document(execution_spec))
    boundary = deepcopy(compiled.physical_boundary or {})
    if boundary.get("attention_stages"):
        raise InvalidInput("attention execution was already refined")
    pack = _frontend(compiled, spec)
    g = spec.geometry
    workload, document = deepcopy(compiled.workload_document), deepcopy(compiled.scenario_document)
    tensors = {t["id"]: t for t in workload["tensors"]}
    old_ops = {op["id"]: op for op in workload["operations"]}
    prices = {price.key(): price for price in spec.entries}
    compute = document["costs"]["compute"]
    allowed = document["mapspace"]["allowed_devices"]
    requests = {r.request_id: r for r in compiled.invocation.requests}
    projections = {t.request_id: t for t in compiled.tasks if t.kind == "projection"}
    attentions = {t.request_id: t for t in compiled.tasks if t.kind in ("attention", "attention_core")}
    if set(attentions) != set(requests) or set(projections) != set(requests):
        raise Unsupported("requires one projection and one attention macro/core per request")
    core_only = any(task.kind == "attention_core" for task in attentions.values())
    if core_only and (pack is not None or not boundary.get("attention_execution")):
        raise Unsupported("existing pure attention cores require their materialized Dense frontend boundary")
    tasks, operations, regions, metadata, scopes, new_ids = [], [], [], {}, {}, []

    def allocation(name, shape, *, role=ROLE_INTERMEDIATE, dtype=DTYPE_F32, locations=()):
        if name in tensors:
            raise InvalidInput(f"derived attention tensor collides: {name}")
        tensors[name] = _physical_tensor(name, role, dtype, shape, locations)
        new_ids.append(name)

    def stage(template, kind, label, inputs, output, *, extras=(), operation_id=None):
        oid = operation_id or op_id(compiled.layer.id, template.request_id, label)
        if oid in metadata or oid in old_ops and oid != template.operation_id:
            raise InvalidInput("derived attention operation IDs collide")
        device_prices = {}
        for device in compiled.invocation.devices:
            shape = template.shape
            key = (kind, device, shape.phase, shape.batch, shape.tokens, shape.visible_history)
            price = prices.get(key)
            if price is None:
                raise Unsupported(f"missing exact independent attention stage cost {key!r}")
            device_prices[device] = dict(duration_ns=price.duration_ns, workspace_bytes=price.workspace_bytes)
        compute[oid] = device_prices
        allowed[oid] = list(compiled.invocation.devices)
        items = tuple((template.request_id, pos) for pos in compiled.appended_positions[template.request_id])
        tasks.append(CompiledTask(template.request_id, label, kind, oid, template.device, template.shape,
                                  output, pack if kind in ("attention_projection", "attention_output") else None,
                                  (), None, items))
        op = dict(id=oid, semantic_op=kind, ggml_op="M2_" + kind.upper(), unary_op=None,
                  op_params=[], inputs=list(inputs), output=output)
        if extras:
            op["extra_outputs"] = list(extras)
        operations.append(op)
        metadata[oid] = dict(region_id="attention:" + template.request_id, execution_stage=kind,
                             execution_variant="qk_softmax_pv", request_id=template.request_id)
        scopes[output] = [dict(request_id=req, absolute_position=pos) for req, pos in items]
        return oid

    for task in compiled.tasks:
        if task.kind == "projection" and pack is not None:
            request = requests[task.request_id]
            prefix = task.task_id + ".staged"
            ports = {p: tensor_id(compiled.layer.id, task.request_id, prefix + "." + p) for p in ("q", "k", "v")}
            for port, heads in (("q", g["heads"]), ("k", g["kv_heads"]), ("v", g["kv_heads"])):
                allocation(ports[port], (g["head_dim"], request.tokens, heads, 1))
            position = tensor_id(compiled.layer.id, task.request_id, prefix + ".positions")
            allocation(position, (request.tokens, 1, 1, 1), role=ROLE_INPUT, dtype=DTYPE_I32,
                       locations=(request.input_location,))
            compute.pop(task.operation_id); allowed.pop(task.operation_id)
            tensors.pop(task.output_tensor)
            stage(task, "attention_projection", task.task_id, old_ops[task.operation_id]["inputs"] + [position],
                  ports["q"], extras=(ports["k"], ports["v"]), operation_id=task.operation_id)
            continue
        if task.kind not in ("attention", "attention_core"):
            tasks.append(task); operations.append(deepcopy(old_ops[task.operation_id])); continue
        request = requests[task.request_id]
        projection = projections[task.request_id]
        history = compiled.state_in.for_request(task.request_id)
        historical = [block.id for block in history.blocks]
        if pack is None:
            ports = {p: tensor_id(compiled.layer.id, task.request_id, f"{projection.task_id}.{p}")
                     for p in compiled.layer.physical.projection_ports(request.tokens)}
            expected = compiled.layer.physical.projection_ports(request.tokens)
            for name, shape in expected.items():
                tensor = tensors.get(ports[name])
                if tensor is None or tuple(tensor["ne"]) != shape or tensor["dtype"] != DTYPE_F32:
                    raise InvalidInput("materialized QKV/gate port shape differs from physical frontend")
            residual = tensor_id(compiled.layer.id, task.request_id, "in.hidden")
            output_weights = [tid for tid in old_ops[task.operation_id]["inputs"] if tensors[tid]["role"] == "weight"]
        else:
            ports = {p: tensor_id(compiled.layer.id, task.request_id, f"{projection.task_id}.staged.{p}") for p in ("q", "k", "v")}
            residual = tensor_id(compiled.layer.id, task.request_id, "in")
            output_weights = [pack]
        shape = (task.shape.visible_history + request.tokens, request.tokens, g["heads"], 1)
        scores = tensor_id(compiled.layer.id, task.request_id, task.task_id + ".scores")
        probabilities = tensor_id(compiled.layer.id, task.request_id, task.task_id + ".probabilities")
        context = (task.output_tensor if task.kind == "attention_core" else
                   tensor_id(compiled.layer.id, task.request_id, task.task_id + ".context"))
        allocation(scores, shape); allocation(probabilities, shape)
        if task.kind != "attention_core":
            allocation(context, (g["heads"] * g["head_dim"], request.tokens, 1, 1))
        compute.pop(task.operation_id); allowed.pop(task.operation_id)
        qk = stage(task, "attention_qk", task.task_id + ".qk", [ports["q"], ports["k"], *historical], scores)
        softmax = stage(task, "attention_softmax", task.task_id + ".softmax", [scores], probabilities)
        pv = stage(task, "attention_pv", task.task_id + ".pv", [probabilities, ports["k"], ports["v"], *historical],
                   context, extras=old_ops[task.operation_id].get("extra_outputs", ()))
        if task.kind == "attention_core":
            outputs = [t for t in compiled.tasks if t.request_id == task.request_id and t.kind == "attention_output"]
            if len(outputs) != 1:
                raise InvalidInput("existing pure attention must retain one output projection/residual stage")
            output = outputs[0].operation_id
        else:
            output = stage(task, "attention_output", task.task_id + ".output",
                           [context, *([ports["gate"]] if "gate" in ports else []), *output_weights, residual], task.output_tensor)
        regions.append(dict(request_id=task.request_id, qkv_ports=ports, qk_operation_id=qk,
                            softmax_operation_id=softmax, pv_operation_id=pv, output_operation_id=output,
                            historical_kv_ids=historical, new_kv_ids=[b.id for b in compiled.new_blocks[task.request_id]],
                            historical_kv_blocks=[asdict(b) for b in history.blocks],
                            absolute_positions=list(compiled.appended_positions[task.request_id]),
                            scores_tensor=scores, probabilities_tensor=probabilities, context_tensor=context))
    workload["tensors"], workload["operations"] = list(tensors.values()), operations
    workload["origin"] = dict(kind="synthetic_layer", ggml_version=None,
                               sample="explicit independent attention stages; not calibrated numerical kernels")
    transfer_prices = dict(compiled.cost_table.transfers)
    for price in spec.transfer_entries:
        key = (price.route, price.storage_bytes)
        existing = transfer_prices.get(key)
        if existing is not None and existing.duration_ns != price.duration_ns:
            raise InvalidInput(f"attention transfer cost conflicts with existing exact price {key!r}")
        transfer_prices[key] = TransferEntry(price.duration_ns, "synthetic_transfer")
    table = replace(compiled.cost_table, transfers=transfer_prices)
    routes = document["mapspace"]["allowed_transfers"]
    for route, ids in routes.items():
        additions = new_ids if route in table.allowed_transfer_routes else []
        routes[route] = list(dict.fromkeys([tid for tid in ids if tid in tensors] + additions))
    _, document["costs"]["transfer"] = expand_transfer_costs(table, {tid: t["storage_bytes"] for tid, t in tensors.items()}, routes)
    document["costs"]["source"] += ";attention-stages:synthetic_execution"
    scenario = build_scenario(workload, document, where="attention_stages")
    boundary.update(numeric_execution_verified=False, physical_macro_contract=False,
                    original_model_kernel_reproduced=False)
    boundary["attention_stages"] = dict(format_version=FORMAT_VERSION, execution_spec=attention_stages_document(spec),
        regions=regions, operation_stages=metadata, tensor_row_scopes=scopes,
        cost_semantics="independent_exact_stage_prices_scoped_to_declared_geometry_frontend; no_macro_division",
        qk_semantics="Q_times_K_transpose_scaled_by_inverse_sqrt_head_dim; query_head//(heads//kv_heads)",
        softmax_semantics="stable_softmax_with_absolute_causal_mask_key_position<=query_position",
        pv_semantics="probabilities_times_V; append_new_post_RoPE_K_and_V_once_to_logical_F32_blocks",
        output_semantics="optional_sigmoid_gate_then_O_projection_plus_hidden_residual_once",
        kv_block_layout="position_major; each_position_contiguous_K[kv_heads,head_dim]_followed_by_V[kv_heads,head_dim]",
        frontend_semantics=("existing_validated_RMSNorm_QKV_QKnorm_RoPE_and_optional_gate" if pack is None else
                            "F32_QKV_linear_projection_then_normal_adjacent_pair_RoPE_at_absolute_positions; no_norm_no_gate"),
        pack_storage_semantics="whole_immutable_Q_K_V_O_pack_read_at_each_selected_device; no_subweight_aliases" if pack else None,
        batch_accounting="independent_per_request_kernel_with_exact_source_batch_axis; not_fused_batch_kernel",
        limitations=["synthetic_stage_costs_only", "no_fused_attention_kernel", "no_numeric_weight_loading", "no_head_sharding"])
    if pack is not None:
        offset, layout = 0, []
        for role, shape in (("q", (g["hidden"], g["heads"] * g["head_dim"])),
                            ("k", (g["hidden"], g["kv_heads"] * g["head_dim"])),
                            ("v", (g["hidden"], g["kv_heads"] * g["head_dim"])),
                            ("output", (g["heads"] * g["head_dim"], g["hidden"]))):
            size = 4 * math.prod(shape)
            layout.append(dict(role=role, dtype=DTYPE_F32, ne=[*shape, 1, 1], offset_bytes=offset, storage_bytes=size))
            offset += size
        boundary["attention_stages"]["frontend_pack_layout"] = dict(tensor_id=pack, matrices=layout,
                                                                    storage_bytes=offset, aliasing=False)
    return replace(compiled, workload_document=workload, scenario_document=document, scenario=scenario,
                   tasks=tuple(tasks), physical_boundary=boundary, cost_table=table,
                   costs_source=document["costs"]["source"])
