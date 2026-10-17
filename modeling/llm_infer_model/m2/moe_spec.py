"""Explicit synthetic MoE declarations and replay-trace validation.

This boundary describes scheduling semantics, not measured MoE kernels. Expert
IDs are supplied for each token; causal replay may consume them only after the
corresponding router finishes. The trace is not a predictor or a router cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..tensor.spec import InvalidInput, Unsupported

MOE_FORMAT_VERSION = "m2-layer-0.6"
STATE_CACHE_FORMAT_VERSION = "m2-state-0.2"
INFORMATION_MODES = ("causal_replay", "offline_oracle")


@dataclass(frozen=True)
class RouteTraceEntry:
    request_id: str
    layer_id: str
    absolute_position: int
    expert_ids: tuple[int, ...]
    scores: tuple[float, ...] | None = None


def parse_route_trace(raw: Any, *, where: str) -> tuple[RouteTraceEntry, ...]:
    from .spec import (_require_list, _require_mapping, _require_str,
                       _require_int, _reject_unknown_keys)
    entries = _require_list(raw, where)
    parsed = []
    for i, raw_entry in enumerate(entries):
        label = f"{where}[{i}]"
        entry = _require_mapping(raw_entry, label)
        _reject_unknown_keys(entry, {"request_id", "layer_id", "absolute_position", "expert_ids", "scores"}, label)
        expert_ids = tuple(_require_int(v, f"{label}.expert_ids[{j}]")
                           for j, v in enumerate(_require_list(entry.get("expert_ids"), f"{label}.expert_ids")))
        scores = None
        if "scores" in entry:
            values = _require_list(entry["scores"], f"{label}.scores")
            if len(values) != len(expert_ids):
                raise InvalidInput(f"{label}.scores must match expert_ids length")
            checked = []
            for j, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidInput(f"{label}.scores[{j}] must be a finite number")
                try:
                    valid = math.isfinite(value)
                except (OverflowError, ValueError):
                    valid = False
                if not valid:
                    raise InvalidInput(f"{label}.scores[{j}] must be a finite number")
                checked.append(float(value))
            scores = tuple(checked)
        parsed.append(RouteTraceEntry(
            request_id=_require_str(entry.get("request_id"), f"{label}.request_id"),
            layer_id=_require_str(entry.get("layer_id"), f"{label}.layer_id"),
            absolute_position=_require_int(entry.get("absolute_position"), f"{label}.absolute_position"),
            expert_ids=expert_ids, scores=scores,
        ))
    return tuple(parsed)


def validate_moe_layer(layer: Any) -> None:
    from .spec import (ATTENTION_FULL_CAUSAL, FFN_MOE, PROVENANCE_SYNTHETIC,
                       _require_int, _require_str)
    if layer.format_version != MOE_FORMAT_VERSION or layer.physical is not None:
        raise InvalidInput("MoE requires the explicit synthetic m2-layer-0.6 boundary")
    if layer.provenance != PROVENANCE_SYNTHETIC:
        raise Unsupported("m2-layer-0.6 only supports synthetic_layer provenance; no measured MoE kernel is declared")
    if layer.attention != ATTENTION_FULL_CAUSAL or layer.ffn != FFN_MOE:
        raise Unsupported("m2-layer-0.6 supports full_causal + moe only")
    for label, size in [("hidden_bytes_per_token", layer.hidden_bytes_per_token),
                        ("kv_bytes_per_token", layer.kv_bytes_per_token),
                        *((f"weight {w.id!r}.storage_bytes", w.storage_bytes) for w in layer.weights)]:
        _require_int(size, label, minimum=4)
        if size % 4:
            raise InvalidInput(f"{label} must be a multiple of 4 for the synthetic F32 byte-vector boundary")
    if layer.hidden_bytes_per_token != layer.hidden_size * 4:
        raise InvalidInput("hidden_bytes_per_token must equal hidden_size * 4 for the synthetic F32 byte-vector boundary")
    params = layer.ffn_params or {}
    experts = _require_int(params.get("experts"), "ffn_params.experts", minimum=1)
    top_k = _require_int(params.get("top_k"), "ffn_params.top_k", minimum=1)
    if top_k > experts:
        raise InvalidInput("ffn_params.top_k cannot exceed experts")
    ids = params.get("expert_weights")
    if not isinstance(ids, (list, tuple)) or len(ids) != experts:
        raise InvalidInput("ffn_params.expert_weights must explicitly list one weight ID per expert")
    if len(set(ids)) != len(ids):
        raise InvalidInput("ffn_params.expert_weights must not repeat IDs")
    expected_bytes = _require_int(params.get("expert_weight_bytes"), "ffn_params.expert_weight_bytes", minimum=4)
    if expected_bytes % 4:
        raise InvalidInput("ffn_params.expert_weight_bytes must be a multiple of 4")
    for i, weight in enumerate(ids):
        _require_str(weight, f"ffn_params.expert_weights[{i}]")
        ref = layer.weight_by_id(weight)
        if ref.storage_bytes != expected_bytes:
            raise InvalidInput(f"expert {i} weight {weight!r}: storage_bytes must equal expert_weight_bytes")
    shared = params.get("shared_expert_weight")
    has_bytes = "shared_expert_bytes" in params
    if (shared is not None) != has_bytes:
        raise InvalidInput("shared_expert_weight and shared_expert_bytes must be declared together")
    if shared is not None:
        _require_str(shared, "ffn_params.shared_expert_weight")
        if shared in ids:
            raise InvalidInput("shared expert must have an independent weight ID")
        expected_shared = _require_int(params["shared_expert_bytes"], "ffn_params.shared_expert_bytes", minimum=4)
        if layer.weight_by_id(shared).storage_bytes != expected_shared:
            raise InvalidInput("shared expert storage_bytes must equal shared_expert_bytes")
    kinds = ("projection", "attention", "router", "expert_ffn", "combine")
    if len(layer.tasks) != len(kinds) or tuple(t.kind for t in layer.tasks) != kinds:
        raise InvalidInput("MoE tasks must be exactly projection -> attention -> router -> expert_ffn -> combine")
    used_weights = set(ids) | ({shared} if shared is not None else set())
    previous = None
    for task in layer.tasks:
        expected = () if previous is None else (previous,)
        if task.depends_on != expected:
            raise InvalidInput(f"MoE task {task.id!r} must follow the declared five-stage chain")
        if task.kind not in ("projection", "router") and task.weight is not None:
            raise InvalidInput(f"MoE {task.kind} template must not bind a single weight; expert_weights supplies expert bindings")
        if task.weight is not None:
            layer.weight_by_id(task.weight)
            if task.weight in used_weights:
                raise InvalidInput("projection/router weights must be independent from expert weights and each other")
            used_weights.add(task.weight)
        previous = task.id
    if used_weights != {w.id for w in layer.weights}:
        raise InvalidInput("MoE weight inventory contains undeclared or unused weights")


def validate_moe_invocation(layer: Any, invocation: Any, history: Any) -> None:
    from .spec import (_require_one_of, _require_str, _require_int,
                       _state_for, _validate_full_causal_history,
                       PHASE_DECODE, _DEVICES)
    validate_moe_layer(layer)
    if invocation.allow_validation_only_costs:
        raise InvalidInput("synthetic MoE cannot opt into measured validation-only costs")
    source = _require_str(invocation.route_source, "invocation.route_source")
    if not source.strip():
        raise InvalidInput("invocation.route_source must not be blank")
    _require_one_of(invocation.information_mode, "invocation.information_mode", INFORMATION_MODES)
    if not invocation.devices or len(set(invocation.devices)) != len(invocation.devices):
        raise InvalidInput("MoE devices must be a nonempty unique subset of cpu/gpu")
    for device in invocation.devices:
        _require_one_of(device, "invocation.devices", _DEVICES)
    if history.layer is not None and history.layer != layer.id:
        raise InvalidInput(f"MoE weight cache snapshot belongs to layer {history.layer!r}, not {layer.id!r}")
    if any(r.request_id == "__moe_batch__" for r in invocation.requests):
        raise InvalidInput("request_id __moe_batch__ is reserved for internal expert groups")
    known = {r.request_id for r in invocation.requests}
    if len(known) != len(invocation.requests):
        raise InvalidInput("MoE request IDs must be unique")
    if len({s.request_id for s in history.requests}) != len(history.requests):
        raise InvalidInput("MoE state request IDs must be unique")
    if {s.request_id for s in history.requests} != known:
        raise InvalidInput("MoE state_in must exactly cover invocation requests")
    seen_blocks = set()
    expected_trace = set()
    for request in invocation.requests:
        state = _state_for(history, request, invocation)
        if request.position_start != state.next_position:
            raise InvalidInput(f"request {request.request_id!r}: position_start differs from next_position")
        if invocation.phase == PHASE_DECODE and request.tokens != 1:
            raise InvalidInput(f"request {request.request_id!r}: decode requires q_i=1")
        _require_one_of(request.input_location, "request.input_location", ("dram", "vram", "ssd"))
        _require_one_of(request.output_location, "request.output_location", ("dram", "vram", "ssd"))
        _require_one_of(state.state_location, "request.state_location", ("dram", "vram", "ssd"))
        for block in state.blocks:
            if block.id in seen_blocks:
                raise InvalidInput("MoE history block IDs must be unique")
            seen_blocks.add(block.id)
            if not block.locations or len(set(block.locations)) != len(block.locations):
                raise InvalidInput("MoE history blocks require nonempty unique locations")
            for location in block.locations:
                _require_one_of(location, "history.blocks.locations", ("dram", "vram", "ssd"))
        _validate_full_causal_history(layer, request, state)
        expected_trace.update((request.request_id, layer.id, p)
                              for p in range(request.position_start, request.position_start + request.tokens))
    params = layer.ffn_params
    seen_trace = set()
    for route in invocation.route_trace:
        _require_int(route.absolute_position, "MoE route_trace.absolute_position", minimum=0)
        key = (route.request_id, route.layer_id, route.absolute_position)
        if key in seen_trace:
            raise InvalidInput(f"MoE route_trace duplicates token {key!r}")
        seen_trace.add(key)
        if key not in expected_trace:
            raise InvalidInput(f"MoE route_trace contains a token outside this invocation: {key!r}")
        if len(route.expert_ids) != params["top_k"] or len(set(route.expert_ids)) != len(route.expert_ids):
            raise InvalidInput("MoE route_trace must list exactly top_k distinct expert IDs")
        for expert in route.expert_ids:
            if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < params["experts"]:
                raise InvalidInput("MoE route_trace expert ID is outside [0, experts)")
        if route.scores is not None:
            if len(route.scores) != len(route.expert_ids):
                raise InvalidInput("MoE route_trace scores must match expert IDs")
            for score in route.scores:
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise InvalidInput("MoE route_trace scores must be finite")
                try:
                    valid = math.isfinite(score)
                except (OverflowError, ValueError):
                    valid = False
                if not valid:
                    raise InvalidInput("MoE route_trace scores must be finite")
    if seen_trace != expected_trace:
        raise InvalidInput("MoE route_trace must exactly cover every invocation token")
    inventory = {w.id: w for w in layer.weights}
    copies = history.weight_copies
    if len({w.id for w in copies}) != len(copies) or {w.id for w in copies} != set(inventory):
        raise InvalidInput("MoE state_in.weight_copies must exactly cover layer.weights; no implicit weight reset")
    for copy in copies:
        if isinstance(copy.version, bool) or not isinstance(copy.version, int) or copy.version < 1:
            raise InvalidInput("weight_copies.version must be a positive integer")
        if copy.storage_bytes != inventory[copy.id].storage_bytes:
            raise InvalidInput(f"weight_copies {copy.id!r}: storage_bytes differs from layer inventory")
        if not copy.locations or len(set(copy.locations)) != len(copy.locations):
            raise InvalidInput("weight_copies.locations must be nonempty and unique")
        for location in copy.locations:
            _require_one_of(location, "weight_copies.locations", ("dram", "vram", "ssd"))
