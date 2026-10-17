"""Explicit contracts for the first CPU, batch-one GGML macro probes.

These profiles measure partial decoder-block subgraphs, not a complete LLM
layer. They do not export M2 costs or infer missing model geometry.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from profile_contract import ContractError


SCHEMA = "macro_measurement_suite_v1"
CONTRACT = "ggml_cpu_partial_dense_macros_v1"
ATTENTION_VARIANT = "ggml_cpu_explicit_causal_gqa_f32_kv_v1"
FFN_UNIFORM_VARIANT = "ggml_cpu_swiglu_split_f32_activation_uniform_weight_dtype_v1"
FFN_PER_MATRIX_VARIANT = "ggml_cpu_swiglu_split_f32_activation_mixed_weight_dtype_v1"
_DTYPES = {"f32", "f16", "q4_k", "q6_k"}
_CASE_KEYS = {"id", "kind", "phase", "batch", "tokens", "visible_history", "hidden", "intermediate",
    "heads", "kv_heads", "head_dim", "weight_dtype", "weight_dtypes", "attention_gate", "backend", "threads",
    "comment", "source", "metadata"}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ContractError(f"{name} must be one of {sorted(choices)}")
    return value


def validate_macro_case(case: dict[str, Any]) -> dict[str, Any]:
    """Validate every applicable physical axis; batch > 1 remains unsupported.

    No absent geometry is inferred from hidden, heads, or a model name. The
    query-head output dimension may differ from hidden. Prefill can describe
    a chunk with existing history; decode is one new token per sequence.
    """
    if not isinstance(case, dict):
        raise ContractError("macro case must be an object")
    unknown = set(case) - _CASE_KEYS
    if unknown:
        raise ContractError(f"unsupported macro case fields: {sorted(unknown)}; execution parameters are not silently ignored")
    for key in ("comment", "source"):
        if key in case and not isinstance(case[key], (str, dict)):
            raise ContractError(f"{key} metadata must be a string or object")
    if "metadata" in case and not isinstance(case["metadata"], dict):
        raise ContractError("metadata must be an object; it is descriptive and does not change execution")
    result = copy.deepcopy(case)
    if not isinstance(result.get("id"), str) or not result["id"].strip():
        raise ContractError("id must be a nonempty string")
    result["id"] = result["id"].strip()
    kind = _choice(result.get("kind"), "kind", {"attention", "dense_ffn"})
    phase = _choice(result.get("phase"), "phase", {"prefill", "decode"})
    result["batch"] = _integer(result.get("batch"), "batch")
    if result["batch"] != 1:
        raise ContractError("this macro probe supports batch=1 only")
    for key, minimum in (("tokens", 1), ("visible_history", 0), ("hidden", 1)):
        result[key] = _integer(result.get(key), key, minimum)
    if phase == "decode" and result["tokens"] != 1:
        raise ContractError("decode requires tokens=1 for the batch-one probe")
    uniform = "weight_dtype" in result
    per_matrix = "weight_dtypes" in result
    if uniform == per_matrix:
        raise ContractError("provide exactly one of weight_dtype or weight_dtypes")
    if uniform:
        dtype = _choice(result["weight_dtype"], "weight_dtype", _DTYPES)
    elif kind != "dense_ffn":
        raise ContractError("per-matrix weight_dtypes are currently supported only for dense_ffn")
    else:
        dtypes = result["weight_dtypes"]
        if not isinstance(dtypes, dict) or set(dtypes) != {"gate", "up", "down"}:
            raise ContractError("FFN weight_dtypes must specify exactly gate, up and down")
        for key in ("gate", "up", "down"):
            _choice(dtypes[key], f"weight_dtypes.{key}", _DTYPES)
    if "backend" in result and result["backend"] != "CPU":
        raise ContractError("macro probe supports backend=CPU only; no fallback is allowed")
    result["backend"] = "CPU"
    if "threads" in result:
        result["threads"] = _integer(result["threads"], "threads")
    if kind == "dense_ffn":
        result["intermediate"] = _integer(result.get("intermediate"), "intermediate")
        for key in ("heads", "kv_heads", "head_dim"):
            if key in result and (isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] != 0):
                raise ContractError(f"{key} is not applicable to dense_ffn and must be absent or zero")
            result[key] = 0
        if "attention_gate" in result and result["attention_gate"] != "none":
            raise ContractError("attention_gate is not applicable to dense_ffn")
        result["attention_gate"] = "none"
        dtypes = physical_weight_dtypes(result)
        for weight, reduction in (("gate", result["hidden"]), ("up", result["hidden"]), ("down", result["intermediate"])):
            if dtypes[weight] in {"q4_k", "q6_k"} and reduction % 256:
                raise ContractError(f"quantized FFN {weight} weight row length must be divisible by 256")
    else:
        for key in ("heads", "kv_heads", "head_dim"):
            result[key] = _integer(result.get(key), key)
        if result["heads"] % result["kv_heads"]:
            raise ContractError("heads must be divisible by kv_heads for GQA")
        result["attention_gate"] = _choice(result.get("attention_gate"), "attention_gate", {"none", "sigmoid"})
        if "intermediate" in result and (isinstance(result["intermediate"], bool) or not isinstance(result["intermediate"], int) or result["intermediate"] != 0):
            raise ContractError("intermediate is not applicable to attention and must be absent or zero")
        result["intermediate"] = 0
        if dtype in {"q4_k", "q6_k"} and (result["heads"] * result["head_dim"]) % 256:
            raise ContractError("quantized attention output-projection row length heads*head_dim must be divisible by 256")
    return result


def physical_weight_dtypes(case: dict[str, Any]) -> dict[str, str]:
    """Return actual matrix dtypes rather than a model-file quantization label."""
    if case["kind"] == "attention":
        return {"output": case["weight_dtype"]}
    if "weight_dtypes" in case:
        return dict(case["weight_dtypes"])
    return {key: case["weight_dtype"] for key in ("gate", "up", "down")}


def implementation_variant(case: dict[str, Any]) -> str:
    if case["kind"] == "attention":
        return ATTENTION_VARIANT
    return FFN_PER_MATRIX_VARIANT if "weight_dtypes" in case else FFN_UNIFORM_VARIANT


def build_macro_plan(suite: dict[str, Any], threads: int) -> list[dict[str, Any]]:
    if not isinstance(suite, dict) or suite.get("schema_version") != SCHEMA:
        raise ContractError(f"suite schema_version must be {SCHEMA}")
    _integer(threads, "threads")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("suite cases must be a nonempty list")
    plan = []
    for original in cases:
        case = validate_macro_case(original)
        if "threads" in case and case["threads"] != threads:
            raise ContractError("suite case threads disagree with the requested runner threads")
        case["threads"] = threads
        plan.append(case)
    ids = [case["id"] for case in plan]
    if len(set(ids)) != len(ids):
        raise ContractError("macro case ids must be unique")
    return plan


def macro_signature(case: dict[str, Any], *, native_binary_sha256: str | None = None) -> dict[str, Any]:
    """Bind the semantic lookup axes to one explicitly scoped physical probe."""
    case = validate_macro_case(case)
    signature = {
        "contract": CONTRACT, "scope": "partial_decoder_block_macro",
        "kind": case["kind"], "device": "cpu", "backend": "CPU",
        "implementation_variant": implementation_variant(case),
        "phase": case["phase"], "batch": case["batch"], "tokens": case["tokens"],
        "visible_history": case["visible_history"],
        "geometry": {key: case[key] for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim")},
        "weight_dtype": case.get("weight_dtype", "mixed"), "weight_dtypes": physical_weight_dtypes(case),
        "activation_dtype": "f32", "kv_dtype": "f32" if case["kind"] == "attention" else "not_applicable",
        "output_dtype": "f32", "attention_gate": case["attention_gate"],
        "threads_requested": case.get("threads"), "threads_applied": case.get("threads"),
        "timing_scope": "single_completed_cpu_graph_wall_ns",
        "setup_compute_repetitions": 1,
        "weight_policy": "explicit_per_matrix_dtypes" if "weight_dtypes" in case else "uniform_declared_weight_dtype",
        "excluded_operations": ["normalization", "RoPE", "residual_add"],
    }
    if case["kind"] == "dense_ffn":
        signature["operations"] = ["gate_projection", "up_projection", "SiLU_gate", "gate_times_up", "down_projection"]
        signature["history_semantics"] = "not_used_by_dense_ffn; separate measurement record retains requested history"
    else:
        signature["operations"] = ["append_new_KV", "QK_transpose", "scale", "causal_mask", "softmax", "PV"]
        if case["attention_gate"] == "sigmoid":
            signature["operations"] += ["sigmoid_gate", "attention_times_gate"]
        signature["operations"] += ["output_projection"]
        signature["query_key_semantics"] = "Q_and_K_are_supplied_after_positional_transformation"
        signature["history_semantics"] = "existing_KV_plus_q_new_KV; query_i_reads_history_plus_i_plus_one"
    signature["operations"] += ["final_output_copy"]
    signature["physical_contract_sha256"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    signature["native_binary_sha256"] = native_binary_sha256
    return signature
