"""Exact physical boundary for a declared CPU Dense decoder block.

This includes both residuals, RMS normalization and normal RoPE. It is a fixed,
explicit implementation contract, not a claim to reproduce any model's fused
kernel. Its materialized ports and logical KV allocation are deliberately bound
into signatures independently of the previous partial decoder format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout
from .dense_contract import (
    CONTRACT, signature, stage_weight_roles, validate_config, weight_shapes,
)
from .physical import DTYPES

FULL_FORMAT_VERSION = "m2-layer-0.3"
STAGES = ("projection", "attention", "dense_ffn")


@dataclass(frozen=True)
class FullDensePhysical:
    config: dict[str, Any]
    threads: int
    weight_groups: dict[str, dict[str, str]]
    contract: str = CONTRACT
    scope: str = "complete_declared_decoder_block"
    input_boundary: str = "hidden_activation"
    port_layout: str = "materialized_stage_ports"
    kv_allocation: str = "logical_blocks"
    activation_dtype: str = "f32"
    kv_dtype: str = "f32"
    excluded_operations: tuple[str, ...] = ()

    def input_ports(self, tokens: int) -> dict[str, tuple[int, ...]]:
        return {"hidden": (self.config["hidden"], tokens, 1, 1),
                "positions": (tokens, 1, 1, 1)}

    def projection_ports(self, tokens: int) -> dict[str, tuple[int, ...]]:
        c = self.config
        ports = {"q": (c["head_dim"], tokens, c["heads"], 1),
                 "k": (c["head_dim"], tokens, c["kv_heads"], 1),
                 "v": (c["head_dim"], tokens, c["kv_heads"], 1)}
        if c["attention_gate"] == "sigmoid":
            ports["gate"] = (c["heads"] * c["head_dim"], tokens, 1, 1)
        return ports


def parse_full_physical(raw: Any) -> FullDensePhysical:
    if not isinstance(raw, dict):
        raise InvalidInput("physical must be an explicit object for m2-layer-0.3")
    keys = {"contract", "scope", "input_boundary", "port_layout", "kv_allocation",
            "activation_dtype", "kv_dtype", "config", "threads", "weight_groups",
            "excluded_operations"}
    if set(raw) != keys:
        raise InvalidInput(f"full physical fields mismatch: missing={sorted(keys-set(raw))}, unknown={sorted(set(raw)-keys)}")
    fixed = {"contract": CONTRACT, "scope": "complete_declared_decoder_block",
             "input_boundary": "hidden_activation", "port_layout": "materialized_stage_ports",
             "kv_allocation": "logical_blocks", "activation_dtype": "f32", "kv_dtype": "f32"}
    for key, expected in fixed.items():
        if raw[key] != expected:
            raise Unsupported(f"physical.{key}={raw[key]!r}; declared full block requires {expected!r}")
    if raw["excluded_operations"] != []:
        raise InvalidInput("full physical.excluded_operations must be [] for the complete declared block")
    threads = raw["threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise InvalidInput("physical.threads must be a positive integer")
    config = validate_config(raw["config"])
    groups = parse_weight_groups(raw["weight_groups"], config)
    return FullDensePhysical(config=config, threads=threads, weight_groups=groups)


def parse_weight_groups(raw_groups: Any, config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """One explicit materialized weight inventory for CPU and CUDA contracts."""
    if not isinstance(raw_groups, dict) or set(raw_groups) != set(STAGES):
        raise InvalidInput("full physical.weight_groups requires projection, attention and dense_ffn")
    seen: set[str] = set()
    groups: dict[str, dict[str, str]] = {}
    for stage in STAGES:
        roles = stage_weight_roles(config, stage)
        group = raw_groups[stage]
        if not isinstance(group, dict) or set(group) != set(roles):
            raise InvalidInput(f"physical.weight_groups.{stage} requires {list(roles)}")
        groups[stage] = {}
        for role in roles:
            name = group[role]
            if not isinstance(name, str) or not name or name in seen:
                raise InvalidInput("full physical weights must name distinct nonempty ids; aliases are unsupported")
            seen.add(name)
            groups[stage][role] = name
    return groups


def validate_full_physical_layer(layer: Any) -> None:
    physical = layer.physical
    if not isinstance(physical, FullDensePhysical) or layer.format_version != FULL_FORMAT_VERSION:
        raise InvalidInput("full physical boundary requires m2-layer-0.3 FullDensePhysical")
    # Direct dataclass construction has the same requirements as document parsing.
    checked = parse_full_physical({
        "contract": physical.contract, "scope": physical.scope,
        "input_boundary": physical.input_boundary, "port_layout": physical.port_layout,
        "kv_allocation": physical.kv_allocation, "activation_dtype": physical.activation_dtype,
        "kv_dtype": physical.kv_dtype, "config": physical.config,
        "threads": physical.threads, "weight_groups": physical.weight_groups,
        "excluded_operations": list(physical.excluded_operations),
    })
    validate_full_geometry_layer(layer, checked)


def validate_full_geometry_layer(layer: Any, physical: Any) -> None:
    """Validate shared Dense DAG, geometry and true GGML weight bytes."""
    config = physical.config
    if layer.attention != "full_causal" or layer.ffn != "dense":
        raise Unsupported("full physical boundary currently supports full_causal + dense only")
    if layer.hidden_size != config["hidden"] or layer.hidden_bytes_per_token != config["hidden"] * 4:
        raise InvalidInput("full physical hidden geometry and bytes must match the F32 hidden input")
    if layer.kv_bytes_per_token != 8 * config["kv_heads"] * config["head_dim"]:
        raise InvalidInput("full physical KV bytes must match logical F32 K plus V")
    if len(layer.tasks) != 3 or {task.kind for task in layer.tasks} != set(STAGES):
        raise Unsupported("full physical block requires exactly projection -> attention -> dense_ffn")
    projection, attention, ffn = (layer.tasks_of_kind(kind)[0] for kind in STAGES)
    if (projection.depends_on or attention.depends_on != (projection.id,)
            or ffn.depends_on != (attention.id,)):
        raise InvalidInput("full physical task DAG must be projection -> attention -> dense_ffn")
    if any(task.weight is not None for task in layer.tasks):
        raise InvalidInput("full physical tasks use explicit weight_groups, not a single task.weight")
    shapes = weight_shapes(config)
    expected_ids = {name for group in physical.weight_groups.values() for name in group.values()}
    if len(layer.weights) != len(expected_ids) or {weight.id for weight in layer.weights} != expected_ids:
        raise InvalidInput("full physical weight inventory must equal every declared norm and matrix")
    for stage in STAGES:
        for role, name in physical.weight_groups[stage].items():
            weight = layer.weight_by_id(name)
            if weight.shape != tuple(shapes[role]) or weight.dtype != config["weight_dtypes"][role]:
                raise InvalidInput(f"full physical weight {name}: shape/dtype does not match role {stage}.{role}")
            ne = (*weight.shape, *(1 for _ in range(4-len(weight.shape))))
            _, size = ggml_contiguous_layout(DTYPES[weight.dtype], ne)
            if weight.storage_bytes != size:
                raise InvalidInput(f"full physical weight {name}: declared {weight.storage_bytes} bytes, actual layout requires {size}")


def signature_for(layer: Any, kind: str, shape: Any) -> dict[str, Any]:
    physical = layer.physical
    if not isinstance(physical, FullDensePhysical):
        raise InvalidInput("full signature requires FullDensePhysical")
    return signature(physical.config, kind, shape.phase, shape.tokens,
                     shape.visible_history, physical.threads, batch=shape.batch)
