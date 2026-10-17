"""Materialized F32 SwiGLU stages and intermediate-axis tensor partitions.

This is a semantic graph/storage transformation, not a calibrated kernel. A
partition is an initial immutable weight representation, never a free runtime
slice. Every new stage needs its own exact price and workspace declaration.
Dense preprocessing/residual are performed once; MoE gather/scatter/combine
remain outside this common FFN region.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..tensor.spec import DTYPE_F32, InvalidInput, Unsupported, ROLE_INTERMEDIATE, ROLE_WEIGHT, ROUTE_ENDS, build_scenario
from .costs import TransferEntry, expand_transfer_costs
from .expand import CompiledInvocation, CompiledTask, _physical_tensor
from .spec import op_id, tensor_id, weight_id

FORMAT_VERSION = "ffn-execution-0.1"
KINDS = frozenset({"ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down", "ffn_sum", "ffn_residual"})
_TARGETS = frozenset({"dense_ffn", "expert_ffn", "shared_expert_ffn"})


@dataclass(frozen=True)
class FFNShard:
    id: str
    start: int
    end: int
    device: str


@dataclass(frozen=True)
class FFNStageCost:
    kind: str
    device: str
    phase: str
    batch: int
    tokens: int
    visible_history: int
    hidden_size: int
    intermediate_size: int
    duration_ns: int
    workspace_bytes: int

    def key(self):
        return (self.kind, self.device, self.phase, self.batch, self.tokens,
                self.visible_history, self.hidden_size, self.intermediate_size)


@dataclass(frozen=True)
class FFNTransferCost:
    route: str
    storage_bytes: int
    duration_ns: int


@dataclass(frozen=True)
class FFNExecutionSpec:
    format_version: str
    target_kinds: tuple[str, ...]
    intermediate_size: int
    shards: tuple[FFNShard, ...]
    merge_input_count: int
    preprocess_device: str
    merge_device: str
    provenance: str
    weight_storage_contract: str
    entries: tuple[FFNStageCost, ...]
    transfer_entries: tuple[FFNTransferCost, ...]

    @property
    def variant(self):
        return "ffn_partitioned"

    @property
    def expert_variant(self):
        return "swiglu_partitioned"


def _object(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")
    return value


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def _choice(value, values, label):
    if not isinstance(value, str) or value not in values:
        raise Unsupported(f"{label} must be one of {sorted(values)}")
    return value


def parse_ffn_execution(document: Any) -> FFNExecutionSpec:
    raw = _object(document, FFNExecutionSpec.__dataclass_fields__, "ffn_execution")
    _choice(raw["format_version"], {FORMAT_VERSION}, "format_version")
    _choice(raw["provenance"], {"synthetic_execution"}, "provenance")
    _choice(raw["weight_storage_contract"], {"materialized_partition_replaces_parent"}, "weight_storage_contract")
    for field in ("preprocess_device", "merge_device"):
        _choice(raw[field], {"cpu", "gpu"}, field)
    middle = _int(raw["intermediate_size"], "intermediate_size", 1)
    targets = raw["target_kinds"]
    if not isinstance(targets, list) or not targets or any(not isinstance(target, str) for target in targets) or len(set(targets)) != len(targets):
        raise InvalidInput("target_kinds must be a nonempty distinct list")
    for target in targets:
        _choice(target, _TARGETS, "target_kinds")
    if not isinstance(raw["shards"], list) or not raw["shards"]:
        raise InvalidInput("shards must be a nonempty explicit list")
    shards, ids, cursor = [], set(), 0
    for item in raw["shards"]:
        _object(item, FFNShard.__dataclass_fields__, "shard")
        name = item["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in ids:
            raise InvalidInput("shard IDs must be distinct nonempty alphanumeric/underscore/hyphen strings")
        _int(item["start"], "shard.start"); _int(item["end"], "shard.end", 1)
        if item["start"] != cursor or item["end"] <= cursor:
            raise InvalidInput("ordered shards must partition the intermediate axis without gaps or overlap")
        _choice(item["device"], {"cpu", "gpu"}, "shard.device")
        cursor = item["end"]; ids.add(name); shards.append(FFNShard(**item))
    if cursor != middle:
        raise InvalidInput("shards must cover the complete declared intermediate_size")
    merge_count = _int(raw["merge_input_count"], "merge_input_count", 1)
    if merge_count != len(shards):
        raise InvalidInput("merge_input_count must equal len(shards); changing merge arity requires a new exact merge cost contract")
    if not isinstance(raw["entries"], list) or not isinstance(raw["transfer_entries"], list):
        raise InvalidInput("entries and transfer_entries must be explicit arrays")
    entries, seen = [], set()
    for item in raw["entries"]:
        _object(item, FFNStageCost.__dataclass_fields__, "ffn stage cost")
        _choice(item["kind"], KINDS, "stage.kind")
        _choice(item["device"], {"cpu", "gpu"}, "stage.device")
        _choice(item["phase"], {"prefill", "decode"}, "stage.phase")
        for field in ("batch", "tokens", "hidden_size", "intermediate_size", "duration_ns"):
            _int(item[field], field, 1)
        for field in ("visible_history", "workspace_bytes"):
            _int(item[field], field)
        entry = FFNStageCost(**item)
        if entry.key() in seen:
            raise InvalidInput("duplicate exact FFN stage cost key")
        seen.add(entry.key()); entries.append(entry)
    transfers, seen = [], set()
    for item in raw["transfer_entries"]:
        _object(item, FFNTransferCost.__dataclass_fields__, "ffn transfer cost")
        _choice(item["route"], ROUTE_ENDS, "transfer.route")
        for field in ("storage_bytes", "duration_ns"):
            _int(item[field], field, 1)
        key = (item["route"], item["storage_bytes"])
        if key in seen:
            raise InvalidInput("duplicate exact FFN transfer cost")
        seen.add(key); transfers.append(FFNTransferCost(**item))
    return FFNExecutionSpec(FORMAT_VERSION, tuple(targets), middle, tuple(shards), merge_count,
                            raw["preprocess_device"], raw["merge_device"], raw["provenance"],
                            raw["weight_storage_contract"], tuple(entries), tuple(transfers))


def ffn_execution_document(spec: FFNExecutionSpec) -> dict[str, Any]:
    if not isinstance(spec, FFNExecutionSpec):
        raise InvalidInput("expected FFNExecutionSpec")
    for field, expected in (("shards", FFNShard), ("entries", FFNStageCost), ("transfer_entries", FFNTransferCost)):
        value = getattr(spec, field)
        if not isinstance(value, tuple) or any(not isinstance(item, expected) for item in value):
            raise InvalidInput(f"typed FFN spec.{field} has invalid values")
    if not isinstance(spec.target_kinds, tuple):
        raise InvalidInput("typed FFN target_kinds must be a tuple")
    result = asdict(spec)
    result["target_kinds"] = list(spec.target_kinds)
    for field in ("shards", "entries", "transfer_entries"):
        result[field] = [asdict(item) for item in getattr(spec, field)]
    return result


def transform_ffn_execution(compiled: CompiledInvocation, execution_spec: FFNExecutionSpec) -> CompiledInvocation:
    """Lower complete Dense or already selected-row/staged MoE FFN regions.

    Dense requires the existing complete F32 decoder contract. MoE requires
    materialized gate/up/down subweights (0.2), including inactive experts.
    Original output identities and all Attention/KV ports are preserved.
    """
    spec = parse_ffn_execution(ffn_execution_document(execution_spec))
    boundary = deepcopy(compiled.physical_boundary or {})
    if "ffn_execution" in boundary:
        raise InvalidInput("FFN execution is already materialized")
    if compiled.workload_document["origin"]["kind"] != "synthetic_layer":
        raise Unsupported("FFN refinement requires synthetic_layer; measured macro prices cannot be split")
    devices = {spec.preprocess_device, spec.merge_device, *(shard.device for shard in spec.shards)}
    if not devices.issubset(compiled.invocation.devices):
        raise Unsupported("FFN execution binds a device outside the invocation")
    hidden = compiled.layer.hidden_size
    if compiled.layer.hidden_bytes_per_token != 4 * hidden:
        raise Unsupported("FFN F32 execution requires hidden_bytes_per_token == 4 * hidden_size")
    dense = compiled.layer.ffn == "dense"
    row = boundary.get("row_execution", {})
    tensors = {item["id"]: deepcopy(item) for item in compiled.workload_document["tensors"]}
    old_operations = {item["id"]: deepcopy(item) for item in compiled.workload_document["operations"]}
    task_by_id = {task.operation_id: task for task in compiled.tasks}
    regions = []
    matrix_sources = {}  # source matrix -> (stage, logical packed parent or None)
    if dense:
        if "dense_ffn" not in spec.target_kinds or set(spec.target_kinds) != {"dense_ffn"}:
            raise InvalidInput("Dense refinement target_kinds must be exactly dense_ffn")
        physical = compiled.layer.physical
        if physical is None or not hasattr(physical, "weight_groups") or not hasattr(physical, "config"):
            raise Unsupported("Dense refinement requires complete materialized decoder weights")
        config = physical.config
        if config["intermediate"] != spec.intermediate_size:
            raise InvalidInput("FFN intermediate_size differs from Dense physical config")
        group = physical.weight_groups["dense_ffn"]
        for stage in ("gate", "up", "down"):
            matrix_sources[weight_id(compiled.layer.id, group[stage])] = (stage, None)
        norm_weight = weight_id(compiled.layer.id, group["post_attention_norm"])
        norm_tensor = tensors[norm_weight]
        if norm_tensor["dtype"] != DTYPE_F32 or tuple(norm_tensor["ne"]) != (hidden, 1, 1, 1):
            raise Unsupported("Dense FFN RMSNorm requires its exact F32 gamma vector")
        for task in compiled.tasks:
            if task.kind != "dense_ffn":
                continue
            op = old_operations[task.operation_id]
            values = [tid for tid in op["inputs"] if tensors[tid]["role"] != ROLE_WEIGHT]
            if len(values) != 1 or set(op["inputs"]) != {values[0], norm_weight, *matrix_sources} or op.get("extra_outputs"):
                raise InvalidInput("Dense FFN inputs do not match norm/gate/up/down + one residual activation")
            regions.append(dict(template=task, old_ids=[task.operation_id], input_tensor=values[0],
                output_tensor=task.output_tensor, weights={stage: weight_id(compiled.layer.id, group[stage]) for stage in ("gate", "up", "down")},
                norm_weight=norm_weight, expert_id=None, parent_region=f"dense:{task.request_id}:{task.task_id}"))
    else:
        if "dense_ffn" in spec.target_kinds:
            raise InvalidInput("MoE refinement cannot target dense_ffn")
        if row.get("format_version") != "moe-execution-0.2" or row.get("expert_variant") != "swiglu_staged":
            raise Unsupported("MoE FFN refinement requires selected_rows swiglu_staged materialized subweights 0.2")
        if row["intermediate_bytes_per_token"] != 4 * spec.intermediate_size:
            raise InvalidInput("FFN intermediate_size differs from staged MoE width")
        partitions = {item["parent_weight_id"]: item for item in row["subweights"]}
        # All experts, including inactive experts, retain a complete immutable
        # leaf inventory so future demand/prefetch never refers to removed parents.
        for parent, part in partitions.items():
            for matrix in part["matrices"]:
                matrix_sources[matrix["tensor_id"]] = (matrix["stage"], parent)
        for source_region in row["regions"]:
            expert = source_region["expert_id"]
            target = "shared_expert_ffn" if expert == "shared" else "expert_ffn"
            if target not in spec.target_kinds:
                continue
            ids = source_region["expert_operation_ids"]
            original_tasks = [task_by_id[oid] for oid in ids]
            if [task.kind for task in original_tasks] != ["expert_gate", "expert_up", "expert_swiglu", "expert_down"]:
                raise InvalidInput("staged MoE region is not gate/up/swiglu/down")
            weights = {stage: original_tasks[index].weight_id for stage, index in (("gate", 0), ("up", 1), ("down", 3))}
            if any(weight not in matrix_sources for weight in weights.values()):
                raise InvalidInput("staged MoE region has no materialized matrix source")
            gate, up, activation, down = (old_operations[oid] for oid in ids)
            expected_inputs = ([weights["gate"], source_region["grouped_input_tensor"]],
                               [weights["up"], source_region["grouped_input_tensor"]],
                               [gate["output"], up["output"]], [weights["down"], activation["output"]])
            if any(op["inputs"] != expected or op.get("extra_outputs")
                   for op, expected in zip((gate, up, activation, down), expected_inputs)) or down["output"] != source_region["grouped_output_tensor"]:
                raise InvalidInput("staged MoE FFN input/dependency contract differs from declared SwiGLU")
            regions.append(dict(template=original_tasks[-1], old_ids=list(ids),
                input_tensor=source_region["grouped_input_tensor"], output_tensor=source_region["grouped_output_tensor"],
                weights=weights, norm_weight=None, expert_id=expert, parent_region=f"expert:{expert}", source_region=source_region))
        if set(spec.target_kinds) != {"expert_ffn", "shared_expert_ffn"}:
            # A shared expert may be absent, but partial transformation would
            # leave unsplit operations referring to removed matrix parents.
            actual = {"shared_expert_ffn" if region["expert_id"] == "shared" else "expert_ffn" for region in row["regions"]}
            if not actual.issubset(spec.target_kinds):
                raise Unsupported("this materialization must target every active routed/shared expert")
    if not regions:
        raise Unsupported("FFN refinement has no matching region")
    for region in regions:
        count = region["template"].shape.tokens
        for port in (region["input_tensor"], region["output_tensor"]):
            if tensors[port]["dtype"] != DTYPE_F32 or tensors[port]["storage_bytes"] != 4 * hidden * count:
                raise InvalidInput("FFN input/output row counts must match exact F32 hidden geometry")
    weights, weight_children, logical_children, weight_partitions = {}, {}, {}, []
    for source_id, (stage, logical_parent) in matrix_sources.items():
        source = tensors[source_id]
        expected_shape = (spec.intermediate_size, hidden, 1, 1) if stage == "down" else (hidden, spec.intermediate_size, 1, 1)
        expected_bytes = 4 * hidden * spec.intermediate_size
        expected_strides = (4, 4 * expected_shape[0], expected_bytes, expected_bytes)
        if source["dtype"] != DTYPE_F32 or tuple(source["ne"]) != expected_shape or source["storage_bytes"] != expected_bytes or tuple(source["nb"]) != expected_strides:
            raise Unsupported(f"FFN source {source_id!r} must be the exact complete contiguous F32 {stage} matrix")
        weight_children[source_id] = []
        for shard in spec.shards:
            width = shard.end - shard.start
            shape = (width, hidden, 1, 1) if stage == "down" else (hidden, width, 1, 1)
            leaf = f"{source_id}:ffn:{shard.id}"
            if leaf in tensors or leaf in weights:
                raise InvalidInput("derived FFN weight identity collides with existing tensor")
            weights[leaf] = _physical_tensor(leaf, ROLE_WEIGHT, DTYPE_F32, shape, tuple(source.get("initial_locations", ())))
            weight_children[source_id].append(leaf)
            if logical_parent:
                logical_children.setdefault(logical_parent, []).append(leaf)
            weight_partitions.append(dict(tensor_id=leaf, source_tensor_id=source_id,
                logical_parent_weight_id=logical_parent, stage=stage, shard_id=shard.id,
                start=shard.start, end=shard.end, input_size=shape[0], output_size=shape[1],
                layout="independent_contiguous_f32", materialization="initial_representation_only"))
    removed_ops = {oid for region in regions for oid in region["old_ids"]}
    kept_outputs = {region["output_tensor"] for region in regions}
    removed_tensors = {tid for oid in removed_ops for tid in [old_operations[oid]["output"], *old_operations[oid].get("extra_outputs", [])]} - kept_outputs
    if any(set(op["inputs"]) & removed_tensors for oid, op in old_operations.items() if oid not in removed_ops):
        raise Unsupported("cannot replace FFN internal values that have external consumers")
    for tid in [*removed_tensors, *matrix_sources]:
        del tensors[tid]
    tensors.update(weights)
    compute = {oid: deepcopy(value) for oid, value in compiled.scenario_document["costs"]["compute"].items() if oid not in removed_ops}
    allowed_devices = {oid: list(value) for oid, value in compiled.scenario_document["mapspace"]["allowed_devices"].items() if oid not in removed_ops}
    costs = {entry.key(): entry for entry in spec.entries}
    operations, new_tasks, stages, scopes, result_regions = [], [], {}, {}, []
    used_ids = set(old_operations) - removed_ops

    def add(region, stage, label, inputs, output, size, device, width, shard=None):
        template = region["template"]
        oid = op_id(compiled.layer.id, template.request_id, label)
        if oid in used_ids:
            raise InvalidInput("derived FFN operation identity collides with existing graph")
        used_ids.add(oid)
        if output not in tensors:
            tensors[output] = _physical_tensor(output, ROLE_INTERMEDIATE, DTYPE_F32, (size, template.shape.tokens, 1, 1))
        elif output != region["output_tensor"]:
            raise InvalidInput("derived FFN activation identity collides with existing graph")
        key = (stage, device, template.shape.phase, template.shape.batch, template.shape.tokens,
               template.shape.visible_history, hidden, width)
        price = costs.get(key)
        if price is None:
            raise Unsupported(f"missing exact FFN stage cost {key!r}; no interpolation or macro splitting")
        compute[oid] = {device: dict(duration_ns=price.duration_ns, workspace_bytes=price.workspace_bytes)}
        allowed_devices[oid] = [device]
        task = CompiledTask(template.request_id, label, stage, oid, device, template.shape, output,
                            next((tid for tid in inputs if tensors[tid]["role"] == ROLE_WEIGHT), None),
                            (), template.expert_id, template.work_items)
        new_tasks.append(task)
        params = ([spec.merge_input_count] if stage == "ffn_sum" else
                  [] if shard is None else [shard.start, shard.end])
        operations.append(dict(id=oid, semantic_op=stage, ggml_op="M2_" + stage.upper(),
                               unary_op=None, op_params=params,
                               inputs=list(inputs), output=output))
        region_id = region["parent_region"] + ":ffn:" + (shard.id if shard else stage)
        stages[oid] = dict(region_id=region_id, parent_expert_region=region["parent_region"] if not dense else None,
                           execution_stage=stage, execution_variant="ffn_partitioned", expert_id=region["expert_id"],
                           shard_id=None if shard is None else shard.id, input_count=len(inputs))
        if template.work_items:
            scopes[output] = [dict(request_id=req, absolute_position=pos) for req, pos in template.work_items]
        return oid

    for region in regions:
        template = region["template"]; prefix = template.task_id + ".ffn"
        new_start = len(new_tasks)
        source = region["input_tensor"]
        if region["norm_weight"]:
            normalized = tensor_id(compiled.layer.id, template.request_id, prefix + ".norm")
            add(region, "ffn_norm", prefix + ".norm", [region["norm_weight"], source], normalized,
                hidden, spec.preprocess_device, spec.intermediate_size)
            source = normalized
        partials = []
        for shard in spec.shards:
            width = shard.end - shard.start
            values = {}
            for stage in ("gate", "up"):
                values[stage] = tensor_id(compiled.layer.id, template.request_id, prefix + "." + shard.id + "." + stage)
                leaf = f"{region['weights'][stage]}:ffn:{shard.id}"
                add(region, "ffn_" + stage, prefix + "." + shard.id + "." + stage, [leaf, source],
                    values[stage], width, shard.device, width, shard)
            activated = tensor_id(compiled.layer.id, template.request_id, prefix + "." + shard.id + ".swiglu")
            add(region, "ffn_swiglu", prefix + "." + shard.id + ".swiglu", [values["gate"], values["up"]],
                activated, width, shard.device, width, shard)
            partial = tensor_id(compiled.layer.id, template.request_id, prefix + "." + shard.id + ".partial")
            add(region, "ffn_down", prefix + "." + shard.id + ".down", [f"{region['weights']['down']}:ffn:{shard.id}", activated],
                partial, hidden, shard.device, width, shard)
            partials.append(partial)
        summed = tensor_id(compiled.layer.id, template.request_id, prefix + ".sum") if dense else region["output_tensor"]
        add(region, "ffn_sum", prefix + ".sum", partials, summed, hidden, spec.merge_device, spec.intermediate_size)
        if dense:
            add(region, "ffn_residual", prefix + ".residual", [summed, region["input_tensor"]], region["output_tensor"],
                hidden, spec.merge_device, spec.intermediate_size)
        new_ids = [task.operation_id for task in new_tasks[new_start:]]
        if "source_region" in region:
            region["source_region"]["expert_operation_ids"] = new_ids
        result_regions.append(dict(request_id=template.request_id, expert_id=region["expert_id"],
            input_tensor=region["input_tensor"], output_tensor=region["output_tensor"],
            operation_ids=new_ids, partial_output_tensors=partials,
            merge_input_count=spec.merge_input_count,
            residual_add_count=1 if dense else 0, norm_count=1 if dense else 0))
    # Insert each replacement at its source region's first operation, preserving
    # a topological serial replay order while permitting runtime overlap.
    by_first = {region["old_ids"][0]: result for region, result in zip(regions, result_regions)}
    new_by_id = {task.operation_id: task for task in new_tasks}
    task_order = []
    for task in compiled.tasks:
        if task.operation_id in by_first:
            task_order.extend(new_by_id[oid] for oid in by_first[task.operation_id]["operation_ids"])
        elif task.operation_id not in removed_ops:
            task_order.append(task)
    operation_map = {oid: value for oid, value in old_operations.items() if oid not in removed_ops}
    operation_map.update({item["id"]: item for item in operations})
    workload = deepcopy(compiled.workload_document)
    workload["tensors"] = list(tensors.values())
    workload["operations"] = [operation_map[task.operation_id] for task in task_order]
    workload["origin"]["sample"] += "; materialized F32 intermediate-axis FFN partitions"
    scenario_doc = deepcopy(compiled.scenario_document)
    for parent, children in weight_children.items():
        if parent in workload["outputs"]:
            workload["outputs"] = [tid for tid in workload["outputs"] if tid != parent] + children
        destination = scenario_doc["goal"]["output_locations"].pop(parent, None)
        if destination:
            scenario_doc["goal"]["output_locations"].update({child: destination for child in children})
    scenario_doc["costs"]["compute"] = compute
    scenario_doc["costs"]["source"] += ";ffn-execution:synthetic_execution"
    scenario_doc["mapspace"]["allowed_devices"] = allowed_devices
    transfers = dict(compiled.cost_table.transfers)
    for entry in spec.transfer_entries:
        key = (entry.route, entry.storage_bytes)
        price = TransferEntry(entry.duration_ns, "synthetic_transfer")
        if key in transfers and transfers[key] != price:
            raise InvalidInput(f"FFN transfer cost conflicts with existing exact price {key!r}")
        transfers[key] = price
    table = replace(compiled.cost_table, transfers=transfers)
    new_tensor_ids = set(tensors) - set(compiled.scenario.workload.tensor_by_id)
    for route, allowed in scenario_doc["mapspace"]["allowed_transfers"].items():
        additions = sorted(new_tensor_ids) if route in table.allowed_transfer_routes else []
        scenario_doc["mapspace"]["allowed_transfers"][route] = [tid for tid in allowed if tid in tensors] + additions
    # Old route-completion gates are tensor-specific. Carry retained gates and
    # replace matrix gates with identical gates for each new allocation.
    dependencies = scenario_doc["mapspace"].get("transfer_dependencies", {})
    for parent, children in weight_children.items():
        if parent in dependencies:
            gate = dependencies.pop(parent)
            dependencies.update({child: deepcopy(gate) for child in children})
    scenario_doc["mapspace"]["transfer_dependencies"] = {key: value for key, value in dependencies.items() if key in tensors}
    _, scenario_doc["costs"]["transfer"] = expand_transfer_costs(table, {tid: item["storage_bytes"] for tid, item in tensors.items()},
                                                                   scenario_doc["mapspace"]["allowed_transfers"])
    if row:
        row["operation_stages"] = {oid: value for oid, value in row.get("operation_stages", {}).items() if oid not in removed_ops}
        row["operation_stages"].update(stages)
        row["tensor_row_scopes"] = {tid: value for tid, value in row.get("tensor_row_scopes", {}).items() if tid not in removed_tensors}
        row["tensor_row_scopes"].update(scopes)
    boundary["ffn_execution"] = dict(format_version=FORMAT_VERSION, execution_spec=ffn_execution_document(spec),
        operation_stages=stages, tensor_row_scopes=scopes, regions=result_regions,
        weight_children={**weight_children, **logical_children}, weight_partitions=weight_partitions,
        weight_storage_contract=spec.weight_storage_contract,
        merge_input_count=spec.merge_input_count,
        merge_cost_contract=dict(scope="outer_ffn_execution_spec", input_count=spec.merge_input_count,
                                 changed_input_count_requires_new_exact_cost_declaration=True),
        numerical_semantics="SwiGLU gate/up output columns paired with down input rows; SUM partial outputs; dense RMSNorm and residual once; no bias in source contract",
        numeric_execution_verified=False, numeric_weights_loaded=False, measured_costs_supported=False,
        cost_semantics="independent exact declared synthetic stage prices scoped to outer FFN spec including merge_input_count; no macro splitting",
        partition_materialization="initial_representation_only; no free runtime slicing or copies")
    boundary.update(physical_macro_contract=False, original_model_kernel_reproduced=False)
    scenario = build_scenario(workload, scenario_doc, where="ffn_execution")
    return replace(compiled, workload_document=workload, scenario_document=scenario_doc, scenario=scenario,
                   tasks=tuple(task_order), cost_table=table, costs_source=scenario_doc["costs"]["source"], physical_boundary=boundary)
