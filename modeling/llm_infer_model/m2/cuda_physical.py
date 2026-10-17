"""A single CUDA device with the complete materialized Dense boundary.

This has no CPU thread field. The model allocates resident weights, inputs,
outputs and persistent KV in VRAM. Scratch from an actual profile can remain
unknown; such a profile must not be executed for capacity conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tensor.spec import InvalidInput, Unsupported
from .dense_cuda_contract import CONTRACT, integer, signature, validate_config
from .full_physical import (
    FullDensePhysical, parse_weight_groups, validate_full_geometry_layer,
)

CUDA_FORMAT_VERSION = "m2-layer-0.4"


@dataclass(frozen=True)
class CudaDensePhysical:
    config: dict[str, Any]
    cuda_device: int
    cuda_graphs: str
    weight_groups: dict[str, dict[str, str]]
    contract: str = CONTRACT
    scope: str = "complete_declared_decoder_block"
    input_boundary: str = "hidden_activation"
    port_layout: str = "materialized_stage_ports"
    kv_allocation: str = "logical_blocks"
    activation_dtype: str = "f32"
    kv_dtype: str = "f32"
    excluded_operations: tuple[str, ...] = ()

    # Both contracts have the same mathematics and materialized port geometry.
    input_ports = FullDensePhysical.input_ports
    projection_ports = FullDensePhysical.projection_ports


def parse_cuda_physical(raw: Any) -> CudaDensePhysical:
    if not isinstance(raw, dict):
        raise InvalidInput("physical must be an explicit object for m2-layer-0.4")
    keys = {"contract", "scope", "input_boundary", "port_layout", "kv_allocation",
            "activation_dtype", "kv_dtype", "config", "cuda_device", "cuda_graphs",
            "weight_groups", "excluded_operations"}
    if set(raw) != keys:
        raise InvalidInput(f"CUDA physical fields mismatch: missing={sorted(keys-set(raw))}, unknown={sorted(set(raw)-keys)}")
    fixed = {"contract": CONTRACT, "scope": "complete_declared_decoder_block",
             "input_boundary": "hidden_activation", "port_layout": "materialized_stage_ports",
             "kv_allocation": "logical_blocks", "activation_dtype": "f32", "kv_dtype": "f32",
             "cuda_graphs": "disabled"}
    for key, expected in fixed.items():
        if raw[key] != expected:
            raise Unsupported(f"physical.{key}={raw[key]!r}; CUDA full block requires {expected!r}")
    if raw["excluded_operations"] != []:
        raise InvalidInput("CUDA physical.excluded_operations must be []")
    device = integer(raw["cuda_device"], "physical.cuda_device", 0)
    config = validate_config(raw["config"])
    groups = parse_weight_groups(raw["weight_groups"], config)
    return CudaDensePhysical(config=config, cuda_device=device,
                             cuda_graphs=raw["cuda_graphs"], weight_groups=groups)


def validate_cuda_physical_layer(layer: Any) -> None:
    physical = layer.physical
    if not isinstance(physical, CudaDensePhysical) or layer.format_version != CUDA_FORMAT_VERSION:
        raise InvalidInput("CUDA physical boundary requires m2-layer-0.4 CudaDensePhysical")
    checked = parse_cuda_physical({
        "contract": physical.contract, "scope": physical.scope,
        "input_boundary": physical.input_boundary, "port_layout": physical.port_layout,
        "kv_allocation": physical.kv_allocation, "activation_dtype": physical.activation_dtype,
        "kv_dtype": physical.kv_dtype, "config": physical.config,
        "cuda_device": physical.cuda_device, "cuda_graphs": physical.cuda_graphs,
        "weight_groups": physical.weight_groups,
        "excluded_operations": list(physical.excluded_operations),
    })
    validate_full_geometry_layer(layer, checked)
    if any(weight.initial_location != "vram" for weight in layer.weights):
        raise Unsupported("CUDA full block requires every weight initially in vram; transfers are not implemented")


def signature_for(layer: Any, kind: str, shape: Any) -> dict[str, Any]:
    physical = layer.physical
    if not isinstance(physical, CudaDensePhysical):
        raise InvalidInput("CUDA signature requires CudaDensePhysical")
    return signature(physical.config, kind, shape.phase, shape.tokens,
                     shape.visible_history, physical.cuda_device, batch=shape.batch,
                     cuda_graphs=physical.cuda_graphs)
