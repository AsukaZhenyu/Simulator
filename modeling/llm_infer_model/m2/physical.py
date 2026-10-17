"""Physical boundary for the CPU Attention -> SwiGLU FFN partial block.

Inputs are materialized, post-position-transform Q/K, V and optional gate.
This is deliberately not a complete decoder layer or the original Qwen kernel.
The scheduling kernel accounts independent resident allocations and running
macro scratch; it does not infer aliases, padded caches or allocator reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..tensor.spec import InvalidInput, Unsupported

CONTRACT = "ggml_cpu_partial_dense_macros_v1"
ATTENTION_VARIANT = "ggml_cpu_explicit_causal_gqa_f32_kv_v1"
FFN_UNIFORM_VARIANT = "ggml_cpu_swiglu_split_f32_activation_uniform_weight_dtype_v1"
FFN_MIXED_VARIANT = "ggml_cpu_swiglu_split_f32_activation_mixed_weight_dtype_v1"
EXCLUDED = ("qkv_projection", "normalization", "RoPE", "residual_add")
DTYPES = {"f32": "GGML_TYPE_F32", "f16": "GGML_TYPE_F16",
          "q4_k": "GGML_TYPE_Q4_K", "q6_k": "GGML_TYPE_Q6_K"}
SIGNATURE_KEYS = (
    "contract", "scope", "kind", "device", "backend", "implementation_variant",
    "phase", "batch", "tokens", "visible_history", "geometry", "weight_dtypes",
    "activation_dtype", "kv_dtype", "output_dtype", "attention_gate",
    "threads_requested", "threads_applied", "timing_scope", "setup_compute_repetitions",
    "weight_policy", "operations", "excluded_operations", "history_semantics",
)


@dataclass(frozen=True)
class DensePhysical:
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    attention_gate: str
    threads: int
    ffn_weight_policy: str
    weight_groups: dict[str, dict[str, str]]
    scope: str = "partial_decoder_block"
    kv_allocation: str = "logical_blocks"
    port_layout: str = "materialized_contiguous"
    excluded_operations: tuple[str, ...] = EXCLUDED

    def input_ports(self, tokens: int) -> dict[str, tuple[int, ...]]:
        ports = {"q": (self.head_dim, tokens, self.heads, 1),
                 "k": (self.head_dim, tokens, self.kv_heads, 1),
                 "v": (self.head_dim, tokens, self.kv_heads, 1)}
        if self.attention_gate == "sigmoid":
            ports["gate"] = (self.head_dim * self.heads, tokens, 1, 1)
        return ports


def _int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput(f"physical.{key} must be a positive integer")
    return value


def parse_physical(raw: Any) -> DensePhysical:
    if not isinstance(raw, dict):
        raise InvalidInput("physical must be an explicit object for m2-layer-0.2")
    keys = {"contract", "scope", "input_boundary", "port_layout", "intermediate",
            "heads", "kv_heads", "head_dim", "attention_gate", "threads",
            "ffn_weight_policy", "weight_groups", "activation_dtype", "kv_dtype",
            "kv_allocation", "excluded_operations"}
    if set(raw) != keys:
        raise InvalidInput(f"physical fields mismatch: missing={sorted(keys-set(raw))}, unknown={sorted(set(raw)-keys)}")
    fixed = {"contract": CONTRACT, "scope": "partial_decoder_block",
             "input_boundary": "post_positional_qkv", "port_layout": "materialized_contiguous",
             "activation_dtype": "f32", "kv_dtype": "f32", "kv_allocation": "logical_blocks"}
    for key, expected in fixed.items():
        if raw[key] != expected:
            raise Unsupported(f"physical.{key}={raw[key]!r}; implemented boundary requires {expected!r}")
    if raw["excluded_operations"] != list(EXCLUDED):
        raise InvalidInput(f"physical.excluded_operations must declare exactly {list(EXCLUDED)}")
    if raw["attention_gate"] not in ("none", "sigmoid"):
        raise InvalidInput("physical.attention_gate must be none or sigmoid")
    policy = raw["ffn_weight_policy"]
    if policy not in ("uniform_declared_weight_dtype", "explicit_per_matrix_dtypes"):
        raise InvalidInput("physical.ffn_weight_policy must explicitly select uniform or per-matrix weights")
    values = {key: _int(raw, key) for key in ("intermediate", "heads", "kv_heads", "head_dim", "threads")}
    if values["heads"] % values["kv_heads"]:
        raise InvalidInput("physical.heads must be divisible by kv_heads")
    groups = raw["weight_groups"]
    roles = {"attention": {"output"}, "dense_ffn": {"gate", "up", "down"}}
    if not isinstance(groups, dict) or set(groups) != set(roles):
        raise InvalidInput("physical.weight_groups must contain attention and dense_ffn")
    seen: set[str] = set()
    for kind, required in roles.items():
        group = groups[kind]
        if not isinstance(group, dict) or set(group) != required:
            raise InvalidInput(f"physical.weight_groups.{kind} requires {sorted(required)}")
        for name in group.values():
            if not isinstance(name, str) or not name or name in seen:
                raise InvalidInput("physical weights must name distinct nonempty matrix ids; aliases are unsupported")
            seen.add(name)
    return DensePhysical(**values, attention_gate=raw["attention_gate"],
                         ffn_weight_policy=policy,
                         weight_groups={kind: dict(group) for kind, group in groups.items()})


def validate_physical_layer(layer: Any) -> None:
    """Cross-check matrix shapes, storage and the declared partial graph."""
    from ..tensor.spec import ggml_contiguous_layout
    physical = layer.physical
    if not isinstance(physical, DensePhysical) or layer.format_version != "m2-layer-0.2":
        raise InvalidInput("physical boundary requires a m2-layer-0.2 DensePhysical descriptor")
    # Construction via dataclasses must obey the same boundary as JSON parsing.
    parse_physical({
        "contract": CONTRACT, "scope": physical.scope, "input_boundary": "post_positional_qkv",
        "port_layout": physical.port_layout, "intermediate": physical.intermediate,
        "heads": physical.heads, "kv_heads": physical.kv_heads, "head_dim": physical.head_dim,
        "attention_gate": physical.attention_gate, "threads": physical.threads,
        "ffn_weight_policy": physical.ffn_weight_policy, "weight_groups": physical.weight_groups,
        "activation_dtype": "f32", "kv_dtype": "f32", "kv_allocation": physical.kv_allocation,
        "excluded_operations": list(physical.excluded_operations),
    })
    if layer.attention != "full_causal" or layer.ffn != "dense":
        raise Unsupported("physical boundary currently supports full_causal + dense only")
    if layer.hidden_bytes_per_token != layer.hidden_size * 4:
        raise InvalidInput("physical hidden_bytes_per_token must equal hidden_size * sizeof(f32)")
    if layer.kv_bytes_per_token != 2 * physical.kv_heads * physical.head_dim * 4:
        raise InvalidInput("physical KV bytes must match F32 K plus V, not the original F16 cache")
    kinds = [task.kind for task in layer.tasks]
    if sorted(kinds) != ["attention", "dense_ffn"]:
        raise Unsupported("physical partial block requires exactly attention -> dense_ffn; QKV projection is an external boundary")
    attention = layer.tasks_of_kind("attention")[0]
    ffn = layer.tasks_of_kind("dense_ffn")[0]
    if attention.depends_on or ffn.depends_on != (attention.id,):
        raise InvalidInput("physical task DAG must be attention -> dense_ffn")
    if any(task.weight is not None for task in layer.tasks):
        raise InvalidInput("physical tasks use explicit weight_groups, not a single task.weight")
    shapes = {"attention": {"output": (physical.heads * physical.head_dim, layer.hidden_size)},
              "dense_ffn": {"gate": (layer.hidden_size, physical.intermediate),
                            "up": (layer.hidden_size, physical.intermediate),
                            "down": (physical.intermediate, layer.hidden_size)}}
    expected_ids = {name for group in physical.weight_groups.values() for name in group.values()}
    if {weight.id for weight in layer.weights} != expected_ids:
        raise InvalidInput("physical weight inventory must equal the four declared matrices")
    for kind, group in physical.weight_groups.items():
        for role, name in group.items():
            weight = layer.weight_by_id(name)
            if weight.shape != shapes[kind][role] or weight.dtype not in DTYPES:
                raise InvalidInput(f"physical matrix {name}: shape/dtype does not match role {kind}.{role}")
            _, size = ggml_contiguous_layout(DTYPES[weight.dtype], (*weight.shape, 1, 1))
            if size != weight.storage_bytes:
                raise InvalidInput(f"physical matrix {name}: declared {weight.storage_bytes} bytes, actual layout requires {size}")
    if physical.ffn_weight_policy == "uniform_declared_weight_dtype":
        dtypes = {layer.weight_by_id(name).dtype for name in physical.weight_groups["dense_ffn"].values()}
        if len(dtypes) != 1:
            raise InvalidInput("uniform FFN weight policy cannot describe mixed matrix dtypes")


def select_measurement_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Stable semantic/physical subset of profiling.macro_contract.macro_signature.

    Executable/hash/source evidence stays in the exported measurement metadata.
    It is never used to relax a geometry, implementation or timing mismatch.
    """
    required = set(SIGNATURE_KEYS)
    if signature.get("kind") == "attention":
        required.add("query_key_semantics")
    missing = sorted(required - set(signature))
    if missing:
        raise InvalidInput(f"measurement signature missing physical boundary fields {missing}")
    return {key: signature[key] for key in sorted(required)}


def signature_for(layer: Any, kind: str, shape: Any) -> dict[str, Any]:
    physical = layer.physical
    group = physical.weight_groups[kind]
    dtypes = {role: layer.weight_by_id(name).dtype for role, name in group.items()}
    attention = kind == "attention"
    variant = ATTENTION_VARIANT if attention else (
        FFN_MIXED_VARIANT if physical.ffn_weight_policy == "explicit_per_matrix_dtypes" else FFN_UNIFORM_VARIANT)
    gate = physical.attention_gate if attention else "none"
    operations = (["append_new_KV", "QK_transpose", "scale", "causal_mask", "softmax", "PV"] if attention
                  else ["gate_projection", "up_projection", "SiLU_gate", "gate_times_up", "down_projection"])
    if gate == "sigmoid":
        operations += ["sigmoid_gate", "attention_times_gate"]
    if attention:
        operations += ["output_projection"]
    operations += ["final_output_copy"]
    signature = {
        "contract": CONTRACT, "scope": "partial_decoder_block_macro", "kind": kind,
        "device": "cpu", "backend": "CPU", "implementation_variant": variant,
        "phase": shape.phase, "batch": shape.batch, "tokens": shape.tokens,
        "visible_history": shape.visible_history,
        "geometry": {"hidden": layer.hidden_size, "intermediate": 0 if attention else physical.intermediate,
                     "heads": physical.heads if attention else 0, "kv_heads": physical.kv_heads if attention else 0,
                     "head_dim": physical.head_dim if attention else 0},
        "weight_dtypes": dtypes, "activation_dtype": "f32", "kv_dtype": "f32" if attention else "not_applicable",
        "output_dtype": "f32", "attention_gate": gate, "threads_requested": physical.threads,
        "threads_applied": physical.threads, "timing_scope": "single_completed_cpu_graph_wall_ns",
        "setup_compute_repetitions": 1,
        "weight_policy": "uniform_declared_weight_dtype" if attention else physical.ffn_weight_policy,
        "operations": operations, "excluded_operations": ["normalization", "RoPE", "residual_add"],
        "history_semantics": "existing_KV_plus_q_new_KV; query_i_reads_history_plus_i_plus_one" if attention
            else "not_used_by_dense_ffn; separate measurement record retains requested history",
    }
    if attention:
        signature["query_key_semantics"] = "Q_and_K_are_supplied_after_positional_transformation"
    return select_measurement_signature(signature)
