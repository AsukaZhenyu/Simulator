"""CUDA execution envelope for the shared complete Dense mathematics.

CPU and CUDA declare separate contracts, devices and timing protocols. Sharing
geometry does not permit sharing cost entries. Unknown CUDA backend scratch
remains unknown; a correct output is not a capacity-validation claim.
"""
from __future__ import annotations

from typing import Any

from ..tensor.spec import InvalidInput, Unsupported
from .dense_contract import (
    CONFIG_KEYS, DTYPES, EXCLUDED, STAGES, SUITE_SCHEMA,
    float32_parameter, integer, operations, ports,
    resident_accounting as _resident_accounting,
    stage_weight_roles, validate_case as validate_math_case, validate_config,
    weight_shapes,
)

CONTRACT = "ggml_cuda_dense_decoder_block_v1"
NATIVE_SCHEMA = "ggml_dense_layer_cuda_probe/1"
# The cache element type is a protocol of the measurement, not a geometry: an f32 cache
# and an f16 cache hold different bytes, move different bytes and are different prices.
# `f32` is the default so every CUDA table measured before the flag existed keeps its
# identity; `f16` is what the CPU tables declare (`dense_contract.KV_DTYPE`), and a GPU
# price that has to sit beside a CPU one in the same model must store the cache the same way.
KV_DTYPES = {"f32": 4, "f16": 2}
DEFAULT_KV_DTYPE = "f32"
VARIANTS = {stage: f"ggml_cuda_dense_materialized_{stage}_v1" for stage in STAGES}


def variant(kind: str, kv_dtype: str = DEFAULT_KV_DTYPE) -> str:
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA dense kv_dtype must be one of {sorted(KV_DTYPES)}")
    return f"ggml_cuda_dense_materialized_{kind}_" + ("v2" if kv_dtype == "f16" else "v1")


def resident_accounting(config: dict[str, Any], kind: str, tokens: int, history: int,
                        kv_dtype: str = DEFAULT_KV_DTYPE) -> dict[str, Any]:
    """Per-cache element size follows the declared cache type (f32: 4, f16: 2 bytes)."""
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA dense kv_dtype must be one of {sorted(KV_DTYPES)}")
    return _resident_accounting(config, kind, tokens, history, kv_element_bytes=KV_DTYPES[kv_dtype])


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, cuda_device: int, batch: int = 1,
              cuda_graphs: str = "disabled",
              kv_dtype: str = DEFAULT_KV_DTYPE) -> dict[str, Any]:
    case = validate_math_case(dict(id="signature", kind=kind, phase=phase, batch=batch,
        tokens=tokens, visible_history=visible_history, config=config))
    integer(cuda_device, "cuda_device", 0)
    if cuda_graphs != "disabled":
        raise Unsupported("this CUDA contract requires cuda_graphs='disabled'; no fallback")
    return {"contract": CONTRACT, "scope": "complete_declared_decoder_block", "kind": kind,
            "device": "gpu", "backend": f"CUDA{cuda_device}",
            "cuda_device": cuda_device, "cuda_graphs": cuda_graphs,
            "backend_execution": "single_cuda_backend_no_cpu_fallback",
            "matmul_precision": "ggml_cuda_backend_default",
            "implementation_variant": variant(kind, kv_dtype),
            "config": case["config"], "phase": phase, "batch": batch, "tokens": tokens,
            "visible_history": visible_history,
            "activation_dtype": "f32", "kv_dtype": kv_dtype, "position_dtype": "i32",
            "numeric_parameter_semantics": "GGML_float32_parameters",
            "timing_scope": "single_completed_cuda_graph_wall_ns", "setup_compute_repetitions": 1,
            "stage_boundary_policy": "timed_materialized_copies_in_isolated_and_joint_graphs",
            "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
            "operations": operations(case["config"], kind), "excluded_operations": list(EXCLUDED),
            "original_model_implementation_reproduced": False}


def validate_case(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidInput("CUDA dense case must be an explicit object")
    if "threads" in raw:
        raise Unsupported("CUDA dense case has no CPU threads parameter")
    math_case = validate_math_case({key: value for key, value in raw.items()
                                    if key not in ("cuda_device", "cuda_graphs", "kv_dtype")})
    if "cuda_device" in raw:
        math_case["cuda_device"] = integer(raw["cuda_device"], "case.cuda_device", 0)
    if "cuda_graphs" in raw:
        if raw["cuda_graphs"] != "disabled":
            raise Unsupported("CUDA dense case requires cuda_graphs='disabled'")
        math_case["cuda_graphs"] = raw["cuda_graphs"]
    kv_dtype = raw.get("kv_dtype", DEFAULT_KV_DTYPE)
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA dense kv_dtype must be one of {sorted(KV_DTYPES)}")
    math_case["kv_dtype"] = kv_dtype
    return math_case


def build_plan(suite: Any, cuda_device: int, cuda_graphs: str = "disabled",
               kv_dtype: str | None = None) -> list[dict[str, Any]]:
    integer(cuda_device, "cuda_device", 0)
    if cuda_graphs != "disabled":
        raise Unsupported("CUDA dense plan requires cuda_graphs='disabled'")
    if kv_dtype is not None and kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA dense kv_dtype must be one of {sorted(KV_DTYPES)}")
    if not isinstance(suite, dict) or suite.get("schema_version") != SUITE_SCHEMA:
        raise InvalidInput(f"dense suite schema_version must be {SUITE_SCHEMA}")
    if set(suite) - {"schema_version", "cases", "comment"}:
        raise InvalidInput("CUDA dense suite contains unsupported fields")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InvalidInput("CUDA dense suite must have cases")
    result = []
    for raw in cases:
        case = validate_case(raw)
        if "cuda_device" in case and case["cuda_device"] != cuda_device:
            raise InvalidInput("case CUDA device differs from requested measurement protocol")
        if "cuda_graphs" in case and case["cuda_graphs"] != cuda_graphs:
            raise InvalidInput("case CUDA graph mode differs from requested measurement protocol")
        # The requested protocol stamps cases that do not name a cache type; a case that
        # names a different one is a conflict, not an override.
        if raw.get("kv_dtype") is not None and kv_dtype is not None and raw["kv_dtype"] != kv_dtype:
            raise InvalidInput("case cache dtype differs from requested measurement protocol")
        result.append({**case, "cuda_device": cuda_device, "cuda_graphs": cuda_graphs,
                       "kv_dtype": kv_dtype or case["kv_dtype"]})
    if len({case["id"] for case in result}) != len(result):
        raise InvalidInput("CUDA dense case ids must be unique")
    return result


def native_arguments(case: dict[str, Any], warmup: int, samples: int, seed: int) -> list[str]:
    case = validate_case(case)
    integer(warmup, "warmup", 0); integer(samples, "samples", 3); integer(seed, "seed", 0)
    if samples > 10000 or seed > 0xffffffff:
        raise InvalidInput("samples or seed exceed native protocol range")
    if "cuda_device" not in case or "cuda_graphs" not in case:
        raise InvalidInput("CUDA native invocation requires explicit cuda_device and cuda_graphs")
    config = case["config"]
    args = ["--stage", case["kind"]]
    for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim"):
        args += ["--"+key.replace("_", "-"), str(config[key])]
    args += ["--q", str(case["tokens"]), "--history", str(case["visible_history"]),
             "--cuda-device", str(case["cuda_device"]), "--cuda-graphs", case["cuda_graphs"],
             "--kv-dtype", case["kv_dtype"],
             "--warmup", str(warmup), "--samples", str(samples), "--seed", str(seed)]
    for key, flag in (("attention_gate", "attention-gate"), ("qk_norm", "qk-norm"),
                      ("rope_mode", "rope-mode"), ("rope_dims", "rope-dims"),
                      ("rope_base", "rope-base"), ("norm_epsilon", "norm-eps")):
        args += ["--"+flag, str(config[key])]
    if config.get("attention_bias"):
        args += ["--attention-bias", "1"]
    flags = {"q": "q-dtype", "k": "k-dtype", "v": "v-dtype", "gate_proj": "gate-proj-dtype",
             "output": "o-dtype", "gate": "ffn-gate-dtype", "up": "up-dtype", "down": "down-dtype",
             "q_bias": "q-bias-dtype", "k_bias": "k-bias-dtype", "v_bias": "v-bias-dtype"}
    for role, flag in flags.items():
        if role in config["weight_dtypes"]:
            args += ["--"+flag, config["weight_dtypes"][role]]
    return args
