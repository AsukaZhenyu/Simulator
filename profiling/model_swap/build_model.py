# -*- coding: utf-8 -*-
"""Build an M2 model skeleton (geometry + one layer + a blank cost template).

The generator exists so that "swap in another model" is a mechanical step.  It
reads only model *structure*:

* ``--gguf PATH`` reads the GGUF header and tensor directory through
  ``GGUFReader``.  Weights are never loaded and no inference is run.  The
  ``gguf`` package is an optional dependency (``pip install -e ".[gguf]"``).
* ``--metadata PATH`` reads the same normalized fields from a hand-written
  JSON file, so a machine without a GGUF can exercise the identical code path.

Nothing here invents a timing.  ``duration_ns``, ``workspace_bytes`` and their
physical signatures stay null in ``<name>.costs.template.json``; a null field is
deliberately not loadable by ``llm_infer_model.m2.costs.parse_cost_table`` until
a real measurement fills it in.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

GEOMETRY_SCHEMA = "m2-model-geometry-0.1"
METADATA_SCHEMA = "m2-model-metadata-0.1"

ATTENTION_KINDS = ("full_causal", "sliding_window", "recurrent_linear", "unknown")
FFN_KINDS = ("dense", "moe")

_ATTENTION_ALIASES = {
    "full": "full_causal",
    "full_attention": "full_causal",
    "attention": "full_causal",
    "linear": "recurrent_linear",
    "ssm": "recurrent_linear",
    "linear_attention": "recurrent_linear",
    "swa": "sliding_window",
    "sliding": "sliding_window",
}

# Tensor role -> M2 weight id for the dense full-attention skeleton.  Unknown
# roles are still kept (with a sanitized id) so a new model never loses bytes
# silently; the generated layer then makes the reviewer see what is unclassified.
_DENSE_ROLE_IDS = {
    "attn_norm": "W_input_norm",
    "attn_q": "W_q",
    "attn_k": "W_k",
    "attn_v": "W_v",
    "attn_q_norm": "W_q_norm",
    "attn_k_norm": "W_k_norm",
    "attn_output": "W_output",
    "attn_gate": "W_gate_proj",
    "q_gate_packed": "W_q_gate_packed",
    "post_attention_norm": "W_post_attention_norm",
    "ffn_gate": "W_gate",
    "ffn_up": "W_up",
    "ffn_down": "W_down",
}

_LAYER_NAME = re.compile(r"^blk\.(\d+)\.(.+)$")


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------


def _int(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def _optional_int(value: Any, label: str, minimum: int = 1) -> int | None:
    return None if value is None else _int(value, label, minimum)


def _float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _plain(value: Any) -> Any:
    """Turn numpy scalars/arrays into JSON-native values without importing numpy."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# normalized metadata
# ---------------------------------------------------------------------------


def _pick(raw: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
    return None


def _normalize_attention_kind(value: Any, label: str) -> str:
    kind = _text(value, label)
    kind = _ATTENTION_ALIASES.get(kind, kind)
    if kind not in ATTENTION_KINDS:
        raise ValueError(f"{label} must be one of {ATTENTION_KINDS}, got {kind!r}")
    return kind


def _empty_layer(index: int, attention: str, ffn: str) -> dict[str, Any]:
    return {
        "index": index,
        "attention": attention,
        "ffn": ffn,
        "tensor_count": 0,
        "weight_bytes": None,
        "tensors": [],
    }


def _normalize_tensor(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    storage = _int(raw.get("storage_bytes"), f"{label}.storage_bytes", minimum=1)
    shape_raw = raw.get("shape")
    shape = None
    if shape_raw is not None:
        if not isinstance(shape_raw, list) or not shape_raw:
            raise ValueError(f"{label}.shape must be a nonempty list")
        shape = [_int(v, f"{label}.shape[{i}]", minimum=1) for i, v in enumerate(shape_raw)]
    dtype = raw.get("dtype")
    if dtype is not None:
        dtype = _text(dtype, f"{label}.dtype")
    name = raw.get("name")
    if name is not None:
        name = _text(name, f"{label}.name")
    return {"name": name, "shape": shape, "dtype": dtype, "storage_bytes": storage}


def _derive_layer_kinds(
    block_count: int,
    interval: int | None,
    explicit: Any,
    ffn_default: str,
) -> list[dict[str, Any]]:
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != block_count:
            raise ValueError(
                f"layer_attention/layer_types must list exactly {block_count} entries"
            )
        attention = [
            _normalize_attention_kind(value, f"layer_attention[{i}]")
            for i, value in enumerate(explicit)
        ]
    elif interval is not None:
        interval = _int(interval, "full_attention_interval")
        # Convention shared by Qwen3.5 and the published examples: the
        # interval-th layer (1-based) is full attention; the rest is linear.
        attention = [
            "full_causal" if (index + 1) % interval == 0 else "recurrent_linear"
            for index in range(block_count)
        ]
    else:
        attention = ["full_causal"] * block_count
    return [_empty_layer(i, kind, ffn_default) for i, kind in enumerate(attention)]


def normalize_metadata(raw: Any, *, source: dict[str, Any]) -> dict[str, Any]:
    """Validate a hand-written metadata object and return the normalized form."""
    if not isinstance(raw, dict):
        raise ValueError("metadata JSON must be an object")
    if raw.get("schema_version") not in (None, METADATA_SCHEMA):
        raise ValueError(
            f"metadata.schema_version must be {METADATA_SCHEMA!r} when present"
        )

    architecture = _text(_pick(raw, "architecture"), "architecture")
    block_count = _int(_pick(raw, "block_count", "n_layer"), "block_count")
    hidden = _int(_pick(raw, "hidden_size", "embedding_length", "n_embd"), "hidden_size")
    heads = _int(_pick(raw, "heads", "head_count", "attention.head_count"), "heads")
    kv_heads = _optional_int(
        _pick(raw, "kv_heads", "head_count_kv", "attention.head_count_kv"),
        "kv_heads",
    ) or heads
    key_length = _optional_int(
        _pick(raw, "key_length", "attention.key_length"), "key_length"
    )
    value_length = _optional_int(
        _pick(raw, "value_length", "attention.value_length"), "value_length"
    )
    declared_head_dim = _optional_int(_pick(raw, "head_dim"), "head_dim")
    if key_length is not None and value_length is not None and key_length != value_length:
        raise ValueError(
            "key_length and value_length differ; the M2 layer descriptor has one head_dim"
        )
    if declared_head_dim is not None and key_length is not None and declared_head_dim != key_length:
        raise ValueError(
            f"head_dim ({declared_head_dim}) disagrees with key_length ({key_length})"
        )
    if declared_head_dim is not None and value_length is not None and declared_head_dim != value_length:
        raise ValueError(
            f"head_dim ({declared_head_dim}) disagrees with value_length ({value_length})"
        )
    head_dim = declared_head_dim or key_length or value_length
    head_dim_source = "metadata"
    if head_dim is None:
        if hidden % heads:
            raise ValueError(
                "head_dim is absent and hidden_size is not divisible by heads"
            )
        head_dim = hidden // heads
        head_dim_source = "hidden_size/heads"
    if heads % kv_heads:
        raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_heads})")

    experts = _optional_int(_pick(raw, "experts", "expert_count"), "experts")
    top_k = _optional_int(
        _pick(raw, "top_k", "expert_used_count"), "top_k"
    )
    if experts is None and top_k is not None:
        raise ValueError("top_k is declared without experts")
    if experts is not None:
        if top_k is None:
            raise ValueError("experts is declared but top_k is missing")
        if top_k > experts:
            raise ValueError(f"top_k ({top_k}) must not exceed experts ({experts})")

    ffn_default = "moe" if experts is not None else "dense"
    interval = _optional_int(
        _pick(raw, "full_attention_interval", "attention_interval"),
        "full_attention_interval",
    )
    explicit_attention = _pick(raw, "layer_attention", "layer_types")
    layers = _derive_layer_kinds(block_count, interval, explicit_attention, ffn_default)

    raw_layers = raw.get("layers")
    if raw_layers is not None:
        if not isinstance(raw_layers, list) or len(raw_layers) != block_count:
            raise ValueError(f"layers must contain exactly {block_count} entries")
        seen: set[int] = set()
        for position, entry in enumerate(raw_layers):
            label = f"layers[{position}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{label} must be an object")
            index = _int(entry.get("index"), f"{label}.index", minimum=0)
            if index in seen or index >= block_count:
                raise ValueError(f"{label}.index {index} is duplicate or out of range")
            seen.add(index)
            attention = entry.get("attention")
            if attention is None:
                attention = layers[index]["attention"]
            else:
                attention = _normalize_attention_kind(attention, f"{label}.attention")
            ffn = entry.get("ffn", layers[index]["ffn"])
            if ffn not in FFN_KINDS:
                raise ValueError(f"{label}.ffn must be one of {FFN_KINDS}, got {ffn!r}")
            tensors_raw = entry.get("tensors")
            tensors: list[dict[str, Any]] = []
            if tensors_raw is not None:
                if not isinstance(tensors_raw, list):
                    raise ValueError(f"{label}.tensors must be a list")
                tensors = [
                    _normalize_tensor(item, f"{label}.tensors[{j}]")
                    for j, item in enumerate(tensors_raw)
                ]
            weight_bytes = entry.get("weight_bytes")
            if weight_bytes is None and tensors:
                weight_bytes = sum(t["storage_bytes"] for t in tensors)
            weight_bytes = _optional_int(weight_bytes, f"{label}.weight_bytes", minimum=1)
            layers[index] = {
                "index": index,
                "attention": attention,
                "ffn": ffn,
                "tensor_count": int(entry.get("tensor_count", len(tensors))),
                "weight_bytes": weight_bytes,
                "tensors": tensors,
            }
        if len(seen) != block_count:
            raise ValueError("layers must cover every block index exactly once")

    return {
        "schema_version": METADATA_SCHEMA,
        "source": dict(source),
        "architecture": architecture,
        "model_name": _pick(raw, "model_name", "name", "general.name"),
        "size_label": _pick(raw, "size_label", "general.size_label"),
        "block_count": block_count,
        "hidden_size": hidden,
        "intermediate_size": _optional_int(
            _pick(raw, "intermediate_size", "feed_forward_length", "n_ff"),
            "intermediate_size",
        ),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "head_dim_source": raw.get("head_dim_source", head_dim_source),
        "rope_dims": _optional_int(
            _pick(raw, "rope_dims", "rope_dimension_count"), "rope_dims"
        ),
        "rope_base": _float(_pick(raw, "rope_base", "rope_freq_base"), "rope_base"),
        "norm_epsilon": _float(
            _pick(raw, "norm_epsilon", "layer_norm_rms_epsilon"), "norm_epsilon"
        ),
        "full_attention_interval": interval,
        "experts": experts,
        "top_k": top_k,
        "expert_intermediate_size": _optional_int(
            _pick(raw, "expert_intermediate_size", "expert_feed_forward_length"),
            "expert_intermediate_size",
        ),
        "shared_expert_intermediate_size": _optional_int(
            _pick(
                raw,
                "shared_expert_intermediate_size",
                "expert_shared_feed_forward_length",
            ),
            "shared_expert_intermediate_size",
        ),
        "vocab_size": _optional_int(_pick(raw, "vocab_size"), "vocab_size"),
        "global_tensor_count": _optional_int(
            _pick(raw, "global_tensor_count"), "global_tensor_count", minimum=0
        ),
        "global_weight_bytes": _optional_int(
            _pick(raw, "global_weight_bytes"), "global_weight_bytes", minimum=0
        ),
        # Manual byte fallbacks for a MoE layer when no tensor directory exists.
        "expert_weight_bytes": _optional_int(
            _pick(raw, "expert_weight_bytes"), "expert_weight_bytes"
        ),
        "shared_expert_bytes": _optional_int(
            _pick(raw, "shared_expert_bytes"), "shared_expert_bytes"
        ),
        "router_weight_bytes": _optional_int(
            _pick(raw, "router_weight_bytes"), "router_weight_bytes"
        ),
        "attention_weight_bytes": _optional_int(
            _pick(raw, "attention_weight_bytes"), "attention_weight_bytes"
        ),
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# GGUF entry
# ---------------------------------------------------------------------------


def _layer_attention(names: list[str]) -> str:
    if any(".ssm_" in name or name.endswith(".ssm_a") for name in names):
        return "recurrent_linear"
    if any(
        marker in name
        for name in names
        for marker in (".attn_q.", ".attn_k.", ".attn_v.", ".attn_qkv.", ".attn_output.")
    ):
        return "full_causal"
    return "unknown"


def _layer_ffn(names: list[str]) -> str:
    if any("ffn_gate_exps" in name or "ffn_gate_inp" in name for name in names):
        return "moe"
    return "dense"


def _tensor_width(layers: list[dict[str, Any]], stem: str, rank: int) -> int | None:
    """Feed-forward width read from an expert tensor's own shape.

    ``ffn_gate_exps.weight`` is [hidden, width, experts] and
    ``ffn_gate_shexp.weight`` is [hidden, width].  The first layer is enough:
    a GGUF whose experts disagree on width would not load at all.
    """
    suffix = f"{stem}.weight"
    for layer in layers:
        for tensor in layer["tensors"]:
            shape = tensor["shape"]
            if tensor["name"].endswith(suffix) and len(shape) == rank:
                return int(shape[1])
    return None


def metadata_from_gguf(path: str | Path) -> dict[str, Any]:
    """Read a GGUF header/tensor directory only; never load tensor data."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "GGUF input requires the optional 'gguf' package: pip install -e \".[gguf]\""
        ) from exc

    reader = GGUFReader(str(source_path), "r")

    def field(key: str) -> Any:
        entry = reader.fields.get(key)
        return None if entry is None else _plain(entry.contents())

    architecture = field("general.architecture")
    if not architecture:
        raise ValueError("GGUF has no general.architecture")

    def value(suffix: str) -> Any:
        return field(f"{architecture}.{suffix}")

    key_length = value("attention.key_length")
    value_length = value("attention.value_length")
    if key_length is not None and value_length is not None and int(key_length) != int(value_length):
        raise ValueError(
            "GGUF key_length and value_length differ; the M2 layer descriptor "
            "has a single head_dim"
        )

    raw_layers: dict[int, list[dict[str, Any]]] = {}
    global_count = 0
    global_bytes = 0
    vocab_size = None
    for tensor in reader.tensors:
        name = str(tensor.name)
        record = {
            "name": name,
            "shape": [int(v) for v in tensor.shape],
            "dtype": str(tensor.tensor_type.name).lower(),
            "storage_bytes": int(tensor.n_bytes),
        }
        if name == "token_embd.weight" and record["shape"]:
            vocab_size = int(record["shape"][-1])
        match = _LAYER_NAME.match(name)
        if match:
            raw_layers.setdefault(int(match.group(1)), []).append(record)
        else:
            global_count += 1
            global_bytes += record["storage_bytes"]

    block_count = int(value("block_count") or 0)
    if block_count <= 0:
        raise ValueError("GGUF has no positive block_count")
    found = sorted(raw_layers)
    if found != list(range(block_count)):
        raise ValueError(
            f"GGUF block tensors cover {found[:5]}... but block_count={block_count}"
        )

    layers = []
    for index in range(block_count):
        tensors = raw_layers[index]
        names = [t["name"] for t in tensors]
        layers.append(
            {
                "index": index,
                "attention": _layer_attention(names),
                "ffn": _layer_ffn(names),
                "tensor_count": len(tensors),
                "weight_bytes": sum(t["storage_bytes"] for t in tensors),
                "tensors": tensors,
            }
        )

    expert_width = _optional_int(value("expert_feed_forward_length"),
                                 "expert_intermediate_size")
    if expert_width is None:
        # granitemoe and qwen2moe publish neither expert_feed_forward_length nor
        # expert_shared_feed_forward_length: for a pure-MoE block
        # feed_forward_length is the expert width, not a dense FFN.  The tensor
        # directory still carries the exact width, so read it there instead of
        # reporting "unknown" for a number the file states plainly.
        expert_width = _tensor_width(layers, "ffn_gate_exps", 3)
    shared_width = _optional_int(value("expert_shared_feed_forward_length"),
                                 "shared_expert_intermediate_size")
    if shared_width is None:
        shared_width = _tensor_width(layers, "ffn_gate_shexp", 2)

    normalized = {
        "schema_version": METADATA_SCHEMA,
        "source": {
            "entry": "gguf",
            "file": str(source_path),
            "size_bytes": source_path.stat().st_size,
        },
        "architecture": str(architecture),
        "model_name": field("general.name"),
        "size_label": field("general.size_label"),
        "block_count": block_count,
        "hidden_size": value("embedding_length"),
        "intermediate_size": value("feed_forward_length"),
        "heads": value("attention.head_count"),
        "kv_heads": value("attention.head_count_kv"),
        "head_dim": key_length or value_length,
        "head_dim_source": "gguf_attention_key_length"
        if key_length is not None
        else "gguf_attention_value_length",
        "rope_dims": value("rope.dimension_count"),
        "rope_base": value("rope.freq_base"),
        "norm_epsilon": value("attention.layer_norm_rms_epsilon"),
        "full_attention_interval": value("full_attention_interval"),
        "experts": value("expert_count"),
        "top_k": value("expert_used_count"),
        "expert_intermediate_size": expert_width,
        "shared_expert_intermediate_size": shared_width,
        "vocab_size": vocab_size,
        "global_tensor_count": global_count,
        "global_weight_bytes": global_bytes,
        "layers": layers,
    }
    return normalize_metadata(normalized, source=normalized["source"])


def load_metadata_json(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    return normalize_metadata(
        raw,
        source={"entry": "metadata", "file": str(source_path)},
    )


# ---------------------------------------------------------------------------
# derived documents
# ---------------------------------------------------------------------------


def _kv_bytes_per_token(meta: dict[str, Any]) -> int:
    # K and V, F32.  This is the M2 dense contract, not a GGUF observation.
    return meta["kv_heads"] * meta["head_dim"] * 2 * 4


def _hidden_bytes_per_token(meta: dict[str, Any]) -> int:
    # F32 hidden activation, per the M2 contract.
    return meta["hidden_size"] * 4


def representative_layer(meta: dict[str, Any], requested: int | None = None) -> int:
    if requested is not None:
        index = _int(requested, "layer", minimum=0)
        if index >= meta["block_count"]:
            raise ValueError(f"layer {index} is out of range (block_count={meta['block_count']})")
        if meta["layers"][index]["attention"] != "full_causal":
            raise ValueError(
                f"layer {index} is {meta['layers'][index]['attention']!r}; the M2 "
                "executor currently implements full_causal attention only"
            )
        return index
    for index, layer in enumerate(meta["layers"]):
        if layer["attention"] == "full_causal":
            return index
    raise ValueError(
        "no full_causal layer found; M2 currently executes full_causal only. "
        "The geometry file is still written; run layer generation with --structure-only"
    )


def build_geometry(meta: dict[str, Any], *, layer_index: int | None) -> dict[str, Any]:
    layers = meta["layers"]
    counts: dict[str, int] = {}
    for layer in layers:
        counts[layer["attention"]] = counts.get(layer["attention"], 0) + 1
    ffn_kinds = {layer["ffn"] for layer in layers}
    ffn_kind = ffn_kinds.pop() if len(ffn_kinds) == 1 else "mixed"
    representative = layer_index
    if representative is None:
        for index, layer in enumerate(layers):
            if layer["attention"] == "full_causal":
                representative = index
                break
    target = layers[representative] if representative is not None else None

    notes = [
        "hidden_bytes_per_token and kv_bytes_per_token assume the M2 F32 activation "
        "and F32 KV contract; they are derived, not read from GGUF.",
        "Weight bytes and tensor shapes/dtypes are exact GGUF tensor-directory values.",
        "No duration_ns or workspace_bytes is derived here; see the cost template.",
    ]
    if counts.get("recurrent_linear"):
        notes.append(
            "recurrent_linear layers are recorded for structure only; the M2 "
            "executor currently implements full_causal attention."
        )
    if ffn_kind == "moe":
        notes.append(
            "m2-layer-0.6 is the synthetic MoE boundary; it declares semantics, "
            "not a measured MoE kernel."
        )
    packed_layers = [
        layer["index"] for layer in layers if _packed_q_gate(layer, meta)
    ]
    if packed_layers:
        notes.append(
            "Q and gate share an interleaved attn_q tensor in layers "
            f"{packed_layers}; the M2 dense contract declares separate Q and "
            "gate_proj, so split it before a physical layer."
        )

    geometry: dict[str, Any] = {
        "schema_version": GEOMETRY_SCHEMA,
        "id": meta.get("id"),
        "source": meta["source"],
        "architecture": meta["architecture"],
        "model_name": meta["model_name"],
        "size_label": meta["size_label"],
        "block_count": meta["block_count"],
        "hidden_size": meta["hidden_size"],
        "intermediate_size": meta["intermediate_size"],
        "vocab_size": meta["vocab_size"],
        "attention": {
            "heads": meta["heads"],
            "kv_heads": meta["kv_heads"],
            "head_dim": meta["head_dim"],
            "head_dim_source": meta["head_dim_source"],
            "heads_per_kv": meta["heads"] // meta["kv_heads"],
            "rope_dims": meta["rope_dims"],
            "rope_base": meta["rope_base"],
            "norm_epsilon": meta["norm_epsilon"],
            "layer_kind_counts": counts,
        },
        "ffn": {
            "kind": ffn_kind,
            "experts": meta["experts"],
            "top_k": meta["top_k"],
            "expert_intermediate_size": meta["expert_intermediate_size"],
            "shared_expert_intermediate_size": meta[
                "shared_expert_intermediate_size"
            ],
        },
        "weight_layout": (
            {
                "q_gate_interleaved_layers": packed_layers,
                "note": (
                    "attn_q output width is 2*heads*head_dim and no separate "
                    "attn_gate tensor exists: Q and gate are interleaved in one "
                    "tensor. The M2 dense contract declares separate Q and "
                    "gate_proj, so split it before binding a physical layer."
                ),
            }
            if packed_layers
            else {"q_gate_interleaved_layers": []}
        ),
        "global_tensor_count": meta["global_tensor_count"],
        "global_weight_bytes": meta["global_weight_bytes"],
        "layer_weight_bytes_total": (
            sum(layer["weight_bytes"] for layer in layers if layer["weight_bytes"] is not None)
            if any(layer["weight_bytes"] is not None for layer in layers)
            else None
        ),
        "m2_contract": {
            "layer_format": None if target is None else (
                "m2-layer-0.6" if target["ffn"] == "moe" else "m2-layer-0.1"
            ),
            "representative_layer": representative,
            "representative_attention": None if target is None else target["attention"],
            "representative_ffn": None if target is None else target["ffn"],
            "activation_dtype": "f32",
            "kv_dtype": "f32",
            "hidden_bytes_per_token": _hidden_bytes_per_token(meta),
            "kv_bytes_per_token": _kv_bytes_per_token(meta),
            "notes": notes,
        },
        "layers": [
            {
                key: layer[key]
                for key in ("index", "attention", "ffn", "tensor_count", "weight_bytes")
            }
            for layer in layers
        ],
        "measurements_required": [
            "Fill every entry of <name>.costs.template.json with a measured "
            "duration_ns and workspace_bytes; shapes are never interpolated.",
            "Record the physical signature and contract_hash for each measured "
            "macro task.",
            "Confirm activation dtype, KV dtype, thread count and workspace scope "
            "on the target machine; none of these is GGUF metadata.",
        ],
    }
    return geometry


def _tensor_role(name: str) -> str:
    match = _LAYER_NAME.match(name)
    tail = match.group(2) if match else name
    for suffix in (".weight", ".bias", ""):
        if suffix and tail.endswith(suffix):
            tail = tail[: -len(suffix)]
            break
    return tail


def _packed_q_gate(layer: dict[str, Any], meta: dict[str, Any]) -> bool:
    """True when Q and gate share one interleaved tensor (Qwen3.5 layout)."""
    tensors = layer.get("tensors") or []
    q = next((t for t in tensors if _tensor_role(t["name"] or "") == "attn_q"), None)
    has_gate = any(_tensor_role(t["name"] or "") == "attn_gate" for t in tensors)
    if q is None or not q.get("shape") or has_gate:
        return False
    return int(q["shape"][-1]) == 2 * meta["heads"] * meta["head_dim"]


def _weight_id(role: str, used: set[str]) -> str:
    base = _DENSE_ROLE_IDS.get(role, "W_" + re.sub(r"[^0-9A-Za-z_]+", "_", role))
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _moe_bytes(meta: dict[str, Any], layer: dict[str, Any]) -> dict[str, int]:
    tensors = layer.get("tensors") or []
    experts = meta["experts"]
    if tensors:
        stacked = [t for t in tensors if "_exps" in (t["name"] or "")]
        if not stacked:
            raise ValueError("MoE layer has no *_exps tensors")
        for tensor in stacked:
            if tensor["shape"] and int(tensor["shape"][-1]) != experts:
                raise ValueError(
                    f"{tensor['name']} last dimension {tensor['shape'][-1]} != experts {experts}"
                )
            if tensor["storage_bytes"] % experts:
                raise ValueError(
                    f"{tensor['name']} bytes {tensor['storage_bytes']} are not divisible "
                    f"by experts {experts}"
                )
        expert_bytes = sum(t["storage_bytes"] for t in stacked) // experts
        router = [
            t for t in tensors
            if (t["name"] or "").endswith(".ffn_gate_inp.weight")
        ]
        if len(router) != 1:
            raise ValueError("MoE layer must have exactly one ffn_gate_inp router tensor")
        shared = [t for t in tensors if "_shexp" in (t["name"] or "")]
        excluded = set(id(t) for t in stacked + router + shared)
        attention = [t for t in tensors if id(t) not in excluded]
        result = {
            "expert_weight_bytes": expert_bytes,
            "router_weight_bytes": router[0]["storage_bytes"],
            "shared_expert_bytes": sum(t["storage_bytes"] for t in shared),
            "attention_weight_bytes": sum(t["storage_bytes"] for t in attention),
        }
        for declared, computed in (
            ("expert_weight_bytes", "expert_weight_bytes"),
            ("router_weight_bytes", "router_weight_bytes"),
            ("shared_expert_bytes", "shared_expert_bytes"),
            ("attention_weight_bytes", "attention_weight_bytes"),
        ):
            if meta.get(declared) is not None and meta[declared] != result[computed]:
                raise ValueError(
                    f"declared {declared}={meta[declared]} disagrees with GGUF-derived "
                    f"{result[computed]}"
                )
        return result
    missing = [
        key
        for key in (
            "expert_weight_bytes",
            "router_weight_bytes",
            "attention_weight_bytes",
        )
        if meta.get(key) is None
    ]
    if missing:
        raise ValueError(
            "MoE layer generation needs a tensor directory or explicit "
            f"metadata fields: {', '.join(missing)}"
        )
    return {
        "expert_weight_bytes": meta["expert_weight_bytes"],
        "router_weight_bytes": meta["router_weight_bytes"],
        "shared_expert_bytes": meta.get("shared_expert_bytes") or 0,
        "attention_weight_bytes": meta["attention_weight_bytes"],
    }


def _require_multiple_of_four(value: int, label: str) -> int:
    if value % 4:
        raise ValueError(f"{label}={value} is not a multiple of 4 (M2 F32 boundary)")
    return value


def _dense_layer(meta: dict[str, Any], layer: dict[str, Any], layer_id: str) -> dict[str, Any]:
    used: set[str] = set()
    packed = _packed_q_gate(layer, meta)
    weights = []
    for tensor in layer.get("tensors") or []:
        role = _tensor_role(tensor["name"] or "")
        if packed and role == "attn_q":
            role = "q_gate_packed"
        weights.append(
            {
                "id": _weight_id(role, used),
                "storage_bytes": tensor["storage_bytes"],
                "initial_location": "dram",
            }
        )
    note = (
        "Structure-only skeleton generated from "
        f"{meta['source'].get('file', meta['source'].get('entry'))}. "
        "Weights are GGUF-derived when available; duration_ns and workspace_bytes "
        "are deliberately absent and must be measured."
    )
    if packed:
        note += (
            " attn_q holds interleaved Q and gate in one tensor; the M2 dense "
            "contract declares separate Q and gate_proj, so split it before "
            "binding a physical layer."
        )
    return {
        "format_version": "m2-layer-0.1",
        "id": layer_id,
        "provenance": "synthetic_layer",
        "hidden_size": meta["hidden_size"],
        "hidden_bytes_per_token": _hidden_bytes_per_token(meta),
        "kv_bytes_per_token": _kv_bytes_per_token(meta),
        "attention": "full_causal",
        "ffn": "dense",
        "weights": weights,
        "tasks": [
            {"id": "projection", "kind": "projection", "depends_on": []},
            {"id": "attention", "kind": "attention", "depends_on": ["projection"]},
            {"id": "dense_ffn", "kind": "dense_ffn", "depends_on": ["attention"]},
        ],
        "comment": note,
    }


def _moe_layer(meta: dict[str, Any], layer: dict[str, Any], layer_id: str) -> dict[str, Any]:
    if meta["experts"] is None:
        raise ValueError("MoE layer requires experts and top_k")
    sizes = _moe_bytes(meta, layer)
    expert_bytes = _require_multiple_of_four(
        sizes["expert_weight_bytes"], "expert_weight_bytes"
    )
    router_bytes = _require_multiple_of_four(
        sizes["router_weight_bytes"], "router_weight_bytes"
    )
    attention_bytes = _require_multiple_of_four(
        sizes["attention_weight_bytes"], "attention_weight_bytes"
    )
    shared_bytes = _require_multiple_of_four(
        sizes["shared_expert_bytes"], "shared_expert_bytes"
    )
    experts = meta["experts"]
    expert_ids = [f"E{index}" for index in range(experts)]
    weights = [
        {"id": "W_proj", "storage_bytes": attention_bytes, "initial_location": "dram"},
        {"id": "W_router", "storage_bytes": router_bytes, "initial_location": "dram"},
    ] + [
        {"id": expert_id, "storage_bytes": expert_bytes, "initial_location": "dram"}
        for expert_id in expert_ids
    ]
    ffn_params = {
        "experts": experts,
        "top_k": meta["top_k"],
        "expert_weight_bytes": expert_bytes,
        "expert_weights": expert_ids,
    }
    if shared_bytes:
        weights.append(
            {"id": "W_shared", "storage_bytes": shared_bytes, "initial_location": "dram"}
        )
        ffn_params["shared_expert_bytes"] = shared_bytes
        ffn_params["shared_expert_weight"] = "W_shared"
    return {
        "format_version": "m2-layer-0.6",
        "id": layer_id,
        "provenance": "synthetic_layer",
        "hidden_size": meta["hidden_size"],
        "hidden_bytes_per_token": _hidden_bytes_per_token(meta),
        "kv_bytes_per_token": _kv_bytes_per_token(meta),
        "attention": "full_causal",
        "ffn": "moe",
        "ffn_params": ffn_params,
        "weights": weights,
        "tasks": [
            {"id": "proj", "kind": "projection", "depends_on": [], "weight": "W_proj"},
            {"id": "attn", "kind": "attention", "depends_on": ["proj"]},
            {"id": "router", "kind": "router", "depends_on": ["attn"], "weight": "W_router"},
            {"id": "expert", "kind": "expert_ffn", "depends_on": ["router"]},
            {"id": "combine", "kind": "combine", "depends_on": ["expert"]},
        ],
        "comment": (
            "Structure-only m2-layer-0.6 skeleton generated from "
            f"{meta['source'].get('file', meta['source'].get('entry'))}. "
            "W_proj is the sum of the attention-side tensors of this layer (the "
            "synthetic template binds one projection weight); expert bytes are exact "
            "GGUF payload bytes divided by expert count. Costs remain unmeasured."
        ),
    }


def build_layer(meta: dict[str, Any], *, layer_index: int) -> dict[str, Any]:
    layer = meta["layers"][layer_index]
    if layer["attention"] != "full_causal":
        raise ValueError(
            f"layer {layer_index} is {layer['attention']!r}; the M2 executor "
            "currently implements full_causal attention only"
        )
    layer_id = f"{meta.get('id', meta['architecture'])}-layer{layer_index}"
    if layer["ffn"] == "moe":
        return _moe_layer(meta, layer, layer_id)
    return _dense_layer(meta, layer, layer_id)


def build_costs_template(meta: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    if layer["ffn"] == "moe":
        kinds = ["projection", "attention", "router", "expert_ffn"]
        if "shared_expert_weight" in (layer.get("ffn_params") or {}):
            kinds.append("shared_expert_ffn")
        kinds.append("combine")
        version = "m2-layer-0.6"
        provenance = "synthetic_layer"
        note = (
            "m2-layer-0.6 accepts synthetic MoE costs only; a measured_macro MoE "
            "table is not a supported boundary. Numbers filled here are declared "
            "estimates, not measurements. Expert-group token counts come from the "
            "route trace, not from model geometry."
        )
    else:
        kinds = ("projection", "attention", "dense_ffn")
        version = "m2-layer-0.1"
        provenance = "measured_macro"
        note = (
            "Every duration_ns and workspace_bytes must come from a measurement "
            "with a matching physical signature; no interpolation."
        )
    entries = []
    for kind in kinds:
        for phase, tokens, history in (("prefill", 3, 0), ("decode", 1, 3)):
            entry: dict[str, Any] = {
                "kind": kind,
                "device": "cpu",
                "phase": phase,
                "batch": 1,
                "tokens": tokens,
                "visible_history": history,
                "duration_ns": None,
                "workspace_bytes": None,
                "provenance": provenance,
            }
            if version == "m2-layer-0.6":
                entry["visible_history"] = 0
                entry["accounting"] = (
                    "per_expert_group"
                    if kind in ("expert_ffn", "shared_expert_ffn")
                    else "per_request"
                )
            else:
                entry["accounting"] = "per_request"
                entry["workspace_scope"] = "macro_internal_scratch"
                entry["workspace_source"] = None
                entry["physical_signature"] = None
                entry["contract_hash"] = None
            entries.append(entry)
    document = {
        "format_version": version,
        "source": f"{meta['id']}-unmeasured",
        "provenance": provenance,
        "validation_only": True,
        "entries": entries,
        "comment": (
            "Blank template. duration_ns=null is intentional and is rejected by "
            "parse_cost_table until measured. " + note
        ),
    }
    if version == "m2-layer-0.6":
        document["transfer_entries"] = []
    return document


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or re.search(r"[\\/:*?\"<>|]", name):
        raise ValueError(f"--name is not a usable file stem: {name!r}")
    return name


def _write_json(path: Path, document: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_model(
    meta: dict[str, Any],
    *,
    name: str,
    out_dir: str | Path,
    layer_index: int | None = None,
    structure_only: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    name = _safe_name(name)
    meta = dict(meta, id=name)
    output = Path(out_dir)
    target: int | None = None
    if not structure_only:
        target = representative_layer(meta, layer_index)
    geometry = build_geometry(
        meta, layer_index=target if not structure_only else layer_index
    )
    documents: dict[str, dict[str, Any]] = {
        f"{name}.geometry.json": geometry,
    }
    if not structure_only:
        assert target is not None
        layer = build_layer(meta, layer_index=target)
        documents[f"{name}.layer.json"] = layer
        documents[f"{name}.costs.template.json"] = build_costs_template(meta, layer)
    paths: dict[str, Path] = {}
    for filename, document in documents.items():
        path = output / filename
        _write_json(path, document, force=force)
        paths[filename] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gguf", type=Path, help="GGUF file; header/tensor directory only")
    source.add_argument("--metadata", type=Path, help="hand-written normalized metadata JSON")
    parser.add_argument("--name", required=True, help="model id / output file stem")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="layer index for the skeleton (default: first full_causal layer)",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="write geometry only; skip the layer and cost-template files",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args(argv)
    try:
        if args.gguf is not None:
            meta = metadata_from_gguf(args.gguf)
        else:
            meta = load_metadata_json(args.metadata)
        paths = write_model(
            meta,
            name=args.name,
            out_dir=args.out,
            layer_index=args.layer,
            structure_only=args.structure_only,
            force=args.force,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"build_model: {exc}", file=sys.stderr)
        return 2
    for filename, path in paths.items():
        print(f"{filename}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
