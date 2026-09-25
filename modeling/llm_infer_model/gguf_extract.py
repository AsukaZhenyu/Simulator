from __future__ import annotations

import copy
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .model import config_from_dict


LAYER_PATTERN = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_gguf_reader(gguf_python_path: str | Path | None) -> Any:
    """Import GGUFReader from the installed ``gguf`` distribution.

    ``gguf_python_path`` remains as an escape hatch for pointing at a gguf-py
    checkout from a llama.cpp source tree instead of the installed package.
    """
    if gguf_python_path is not None:
        path = Path(gguf_python_path)
        if not path.is_dir():
            raise RuntimeError(f"--gguf-python-path is not a directory: {path}")
        sys.path.insert(0, str(path.resolve()))
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "GGUF extraction requires the 'gguf' package, which is an optional "
            'dependency. Install it with:  pip install -e ".[gguf]"'
        ) from exc
    return GGUFReader


def _layer_kind(tensor_names: list[str]) -> str:
    if any(
        ".ssm_" in name or ".attn_gate." in name or ".attn_qkv." in name
        for name in tensor_names
    ):
        return "ssm"
    if any(
        marker in name
        for name in tensor_names
        for marker in (".attn_q.", ".attn_k.", ".attn_v.", ".attn_output.")
    ):
        return "attention"
    return "unknown"


def extract_gguf_manifest(
    model_path: str | Path,
    *,
    gguf_python_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    GGUFReader = _load_gguf_reader(gguf_python_path)
    reader = GGUFReader(str(source), "r")

    def metadata_value(key: str) -> Any:
        field = reader.fields.get(key)
        return _plain(field.contents()) if field is not None else None

    architecture = metadata_value("general.architecture")
    block_count_key = f"{architecture}.block_count" if architecture else None
    block_count = metadata_value(block_count_key) if block_count_key else None
    selected_metadata: dict[str, Any] = {}
    for key in (
        "general.architecture",
        "general.name",
        "general.size_label",
        block_count_key,
        f"{architecture}.context_length" if architecture else None,
        f"{architecture}.embedding_length" if architecture else None,
        f"{architecture}.feed_forward_length" if architecture else None,
        f"{architecture}.attention.head_count" if architecture else None,
        f"{architecture}.attention.head_count_kv" if architecture else None,
        f"{architecture}.attention.key_length" if architecture else None,
        f"{architecture}.attention.value_length" if architecture else None,
        f"{architecture}.ssm.conv_kernel" if architecture else None,
        f"{architecture}.ssm.state_size" if architecture else None,
        f"{architecture}.ssm.group_count" if architecture else None,
        f"{architecture}.ssm.time_step_rank" if architecture else None,
        f"{architecture}.ssm.inner_size" if architecture else None,
        f"{architecture}.full_attention_interval" if architecture else None,
    ):
        if key and key in reader.fields:
            selected_metadata[key] = metadata_value(key)

    layer_tensors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    global_tensors: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        shape = [int(dimension) for dimension in tensor.shape]
        row = {
            "name": tensor.name,
            "shape": shape,
            "type": tensor.tensor_type.name,
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
            "estimated_matvec_flops": (
                2 * int(tensor.n_elements) if len(shape) >= 2 else 0
            ),
        }
        match = LAYER_PATTERN.search(tensor.name)
        if match:
            layer_index = int(match.group(1))
            row["layer"] = layer_index
            layer_tensors[layer_index].append(row)
        else:
            row["layer"] = None
            global_tensors.append(row)
        tensor_rows.append(row)

    layer_indices = sorted(layer_tensors)
    if not layer_indices:
        raise ValueError("no blk.<index> tensors found in GGUF")
    expected_indices = list(range(layer_indices[-1] + 1))
    if layer_indices != expected_indices:
        raise ValueError(
            f"non-contiguous layer indices: found {layer_indices}, expected {expected_indices}"
        )
    if block_count is not None and int(block_count) != len(layer_indices):
        raise ValueError(
            f"metadata block_count={block_count} but found {len(layer_indices)} layers"
        )

    layers: list[dict[str, Any]] = []
    for layer_index in layer_indices:
        tensors = layer_tensors[layer_index]
        tensor_names = [str(tensor["name"]) for tensor in tensors]
        layers.append(
            {
                "index": layer_index,
                "kind": _layer_kind(tensor_names),
                "tensor_count": len(tensors),
                "weight_bytes": sum(int(tensor["n_bytes"]) for tensor in tensors),
                "n_elements": sum(int(tensor["n_elements"]) for tensor in tensors),
                "estimated_matvec_flops": sum(
                    int(tensor["estimated_matvec_flops"]) for tensor in tensors
                ),
                "tensor_names": tensor_names,
            }
        )

    layer_bytes = sum(int(layer["weight_bytes"]) for layer in layers)
    global_bytes = sum(int(tensor["n_bytes"]) for tensor in global_tensors)
    tensor_data_bytes = layer_bytes + global_bytes
    return {
        "schema_version": 1,
        "source_file": source.name,
        "source_path": source.as_posix(),
        "source_size_bytes": source.stat().st_size,
        "architecture": architecture,
        "metadata": selected_metadata,
        "tensor_count": len(tensor_rows),
        "layer_count": len(layers),
        "layer_tensor_bytes": layer_bytes,
        "global_tensor_bytes": global_bytes,
        "tensor_data_bytes": tensor_data_bytes,
        "container_overhead_bytes": source.stat().st_size - tensor_data_bytes,
        "estimated_layer_matvec_flops_per_token": sum(
            int(layer["estimated_matvec_flops"]) for layer in layers
        ),
        "estimated_global_matvec_flops": sum(
            int(tensor["estimated_matvec_flops"]) for tensor in global_tensors
        ),
        "layers": layers,
        "global_tensors": global_tensors,
        "tensors": tensor_rows,
        "estimation_notes": [
            "Tensor bytes are exact GGUF payload sizes and exclude container padding/header bytes.",
            "Matvec FLOPs use 2 * n_elements for rank>=2 tensors; elementwise, KV, attention, SSM scan, and launch costs are excluded.",
            "Non-blk tensors are classified as global/resident memory, not streamed layer weights.",
        ],
    }


def manifest_layer_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer["index"]),
            "kind": str(layer["kind"]),
            "tensor_count": int(layer["tensor_count"]),
            "weight_bytes": int(layer["weight_bytes"]),
            "weight_MiB": round(int(layer["weight_bytes"]) / 1024**2, 6),
            "estimated_matvec_flops": int(layer["estimated_matvec_flops"]),
        }
        for layer in manifest["layers"]
    ]


def config_from_gguf_manifest(
    manifest: dict[str, Any],
    template_raw: dict[str, Any],
    *,
    reference_manifest: dict[str, Any] | None = None,
    reference_compute_seconds: float | None = None,
    keep_template_cpu: bool = False,
) -> dict[str, Any]:
    if (reference_manifest is None) != (reference_compute_seconds is None):
        raise ValueError(
            "reference_manifest and reference_compute_seconds must be provided together"
        )
    if reference_compute_seconds is not None and reference_compute_seconds <= 0:
        raise ValueError("reference_compute_seconds must be positive")

    template = config_from_dict(template_raw)
    hardware = copy.deepcopy(template_raw["hardware"])
    policy = copy.deepcopy(template_raw["policy"])

    if reference_manifest is not None and reference_compute_seconds is not None:
        reference_flops = float(
            reference_manifest["estimated_layer_matvec_flops_per_token"]
        )
        hardware["gpu_effective_flops"] = reference_flops / reference_compute_seconds
    if not keep_template_cpu:
        hardware["cpu_effective_flops"] = None
    hardware["global_bytes"] = int(manifest["global_tensor_bytes"])
    policy["window_size"] = min(int(policy["window_size"]), int(manifest["layer_count"]))
    policy["static_gpu_layers"] = min(
        int(policy.get("static_gpu_layers", 0)), int(manifest["layer_count"])
    )

    architecture = str(manifest.get("architecture") or "")
    metadata = manifest.get("metadata", {})
    state_keys = {
        "attention_kv_head_count": f"{architecture}.attention.head_count_kv",
        "attention_key_length": f"{architecture}.attention.key_length",
        "attention_value_length": f"{architecture}.attention.value_length",
        "ssm_conv_kernel": f"{architecture}.ssm.conv_kernel",
        "ssm_state_size": f"{architecture}.ssm.state_size",
        "ssm_group_count": f"{architecture}.ssm.group_count",
        "ssm_inner_size": f"{architecture}.ssm.inner_size",
    }
    state = None
    if architecture and all(key in metadata for key in state_keys.values()):
        state = {
            "context_tokens": 0,
            "sequence_count": 1,
            **{
                field: int(metadata[key])
                for field, key in state_keys.items()
            },
            "kv_element_bytes": 2,
            "recurrent_element_bytes": 4,
        }
        # Replace the template's opaque KV reserve with explicit per-layer
        # attention KV and recurrent-state capacity.
        hardware["kv_bytes"] = 0
        policy.setdefault("state_placement", "roundtrip")
        policy.setdefault("token_state_roundtrip_s", 0.0)

    layers = []
    for layer in manifest["layers"]:
        estimated_flops = max(1, int(layer["estimated_matvec_flops"]))
        layers.append(
            {
                "name": f"blk.{int(layer['index'])}",
                "kind": str(layer["kind"]),
                "weight_bytes": int(layer["weight_bytes"]),
                "flops": estimated_flops,
            }
        )

    calibration_note = "template GPU throughput"
    if reference_manifest is not None and reference_compute_seconds is not None:
        calibration_note = (
            f"{reference_manifest['source_file']} total layer matvec proxy calibrated to "
            f"{reference_compute_seconds * 1000:.6g} ms/token"
        )
    source_stem = Path(str(manifest["source_file"])).stem
    result = {
        "name": f"{source_stem}-gguf-structure-v0.3",
        "notes": (
            "Layer/global bytes and SSM/attention classification come from the GGUF tensor "
            f"directory. Compute uses 2*n_elements matvec proxies with {calibration_note}; "
            "attention KV and recurrent-state geometry come from GGUF metadata with F16 KV "
            "and FP32 recurrent-state storage; H2D bandwidth, workspace, and VRAM capacity "
            "remain template assumptions."
        ),
        "layers": layers,
        "hardware": hardware,
        "policy": policy,
        "provenance": {
            "source_file": manifest["source_file"],
            "manifest_schema_version": manifest["schema_version"],
            "compute_calibration": calibration_note,
            "template_name": template.name,
        },
    }
    if state is not None:
        result["state"] = state
    return result
