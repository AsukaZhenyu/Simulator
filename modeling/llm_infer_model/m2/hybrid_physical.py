"""Complete Dense macros with explicit CPU/CUDA placement and materialized copies.

The hybrid envelope selects two existing execution contracts; it does not invent
an intermediate backend or erase either backend's precise timing signature.
Tensor movement and capacity remain responsibilities of the M1 engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tensor.spec import DEVICE_CPU, DEVICE_GPU, LOC_DRAM, LOC_VRAM, InvalidInput, Unsupported
from .dense_contract import integer, signature as cpu_signature, validate_config
from .dense_cuda_contract import signature as cuda_signature
from .full_physical import FullDensePhysical, parse_weight_groups, validate_full_geometry_layer

HYBRID_FORMAT_VERSION = "m2-layer-0.5"
CONTRACT = "ggml_hybrid_dense_materialized_macros_v1"


@dataclass(frozen=True)
class HybridDensePhysical:
    config: dict[str, Any]
    threads: int
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

    input_ports = FullDensePhysical.input_ports
    projection_ports = FullDensePhysical.projection_ports


def parse_hybrid_physical(raw: Any) -> HybridDensePhysical:
    if not isinstance(raw, dict):
        raise InvalidInput("physical must be an explicit object for m2-layer-0.5")
    keys = {"contract", "scope", "input_boundary", "port_layout", "kv_allocation",
            "activation_dtype", "kv_dtype", "config", "threads", "cuda_device",
            "cuda_graphs", "weight_groups", "excluded_operations"}
    if set(raw) != keys:
        raise InvalidInput(f"hybrid physical fields mismatch: missing={sorted(keys-set(raw))}, unknown={sorted(set(raw)-keys)}")
    fixed = {"contract": CONTRACT, "scope": "complete_declared_decoder_block",
             "input_boundary": "hidden_activation", "port_layout": "materialized_stage_ports",
             "kv_allocation": "logical_blocks", "activation_dtype": "f32", "kv_dtype": "f32",
             "cuda_graphs": "disabled"}
    for key, expected in fixed.items():
        if raw[key] != expected:
            raise Unsupported(f"physical.{key}={raw[key]!r}; hybrid macros require {expected!r}")
    if raw["excluded_operations"] != []:
        raise InvalidInput("hybrid physical.excluded_operations must be []")
    config = validate_config(raw["config"])
    return HybridDensePhysical(config=config,
        threads=integer(raw["threads"], "physical.threads", 1),
        cuda_device=integer(raw["cuda_device"], "physical.cuda_device", 0),
        cuda_graphs=raw["cuda_graphs"],
        weight_groups=parse_weight_groups(raw["weight_groups"], config))


def validate_hybrid_physical_layer(layer: Any) -> None:
    physical = layer.physical
    if not isinstance(physical, HybridDensePhysical) or layer.format_version != HYBRID_FORMAT_VERSION:
        raise InvalidInput("hybrid physical boundary requires m2-layer-0.5 HybridDensePhysical")
    checked = parse_hybrid_physical({
        "contract": physical.contract, "scope": physical.scope,
        "input_boundary": physical.input_boundary, "port_layout": physical.port_layout,
        "kv_allocation": physical.kv_allocation, "activation_dtype": physical.activation_dtype,
        "kv_dtype": physical.kv_dtype, "config": physical.config,
        "threads": physical.threads, "cuda_device": physical.cuda_device,
        "cuda_graphs": physical.cuda_graphs, "weight_groups": physical.weight_groups,
        "excluded_operations": list(physical.excluded_operations),
    })
    validate_full_geometry_layer(layer, checked)
    if any(weight.initial_location not in (LOC_DRAM, LOC_VRAM) for weight in layer.weights):
        raise Unsupported("hybrid weights require explicit dram/vram locations; SSD is not in this slice")


def signature_for(layer: Any, kind: str, shape: Any, device: str) -> dict[str, Any]:
    physical = layer.physical
    if not isinstance(physical, HybridDensePhysical):
        raise InvalidInput("hybrid signature requires HybridDensePhysical")
    if device == DEVICE_CPU:
        return cpu_signature(physical.config, kind, shape.phase, shape.tokens,
                             shape.visible_history, physical.threads, batch=shape.batch)
    if device == DEVICE_GPU:
        return cuda_signature(physical.config, kind, shape.phase, shape.tokens,
                              shape.visible_history, physical.cuda_device, batch=shape.batch,
                              cuda_graphs=physical.cuda_graphs)
    raise Unsupported(f"hybrid macro device {device!r} is not cpu/gpu")
