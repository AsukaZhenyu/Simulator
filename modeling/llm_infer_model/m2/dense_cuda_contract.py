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
    resident_accounting, stage_weight_roles, validate_case as validate_math_case, validate_config,
    weight_shapes,
)

CONTRACT = "ggml_cuda_dense_decoder_block_v1"
NATIVE_SCHEMA = "ggml_dense_layer_cuda_probe/1"
VARIANTS = {stage: f"ggml_cuda_dense_materialized_{stage}_v1" for stage in STAGES}


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, cuda_device: int, batch: int = 1,
              cuda_graphs: str = "disabled") -> dict[str, Any]:
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
            "implementation_variant": VARIANTS[kind],
            "config": case["config"], "phase": phase, "batch": batch, "tokens": tokens,
            "visible_history": visible_history,
            "activation_dtype": "f32", "kv_dtype": "f32", "position_dtype": "i32",
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
                                    if key not in ("cuda_device", "cuda_graphs")})
    if "cuda_device" in raw:
        math_case["cuda_device"] = integer(raw["cuda_device"], "case.cuda_device", 0)
    if "cuda_graphs" in raw:
        if raw["cuda_graphs"] != "disabled":
            raise Unsupported("CUDA dense case requires cuda_graphs='disabled'")
        math_case["cuda_graphs"] = raw["cuda_graphs"]
    return math_case


def build_plan(suite: Any, cuda_device: int, cuda_graphs: str = "disabled") -> list[dict[str, Any]]:
    integer(cuda_device, "cuda_device", 0)
    if cuda_graphs != "disabled":
        raise Unsupported("CUDA dense plan requires cuda_graphs='disabled'")
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
        result.append({**case, "cuda_device": cuda_device, "cuda_graphs": cuda_graphs})
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
             "--warmup", str(warmup), "--samples", str(samples), "--seed", str(seed)]
    for key, flag in (("attention_gate", "attention-gate"), ("qk_norm", "qk-norm"),
                      ("rope_mode", "rope-mode"), ("rope_dims", "rope-dims"),
                      ("rope_base", "rope-base"), ("norm_epsilon", "norm-eps")):
        args += ["--"+flag, str(config[key])]
    flags = {"q": "q-dtype", "k": "k-dtype", "v": "v-dtype", "gate_proj": "gate-proj-dtype",
             "output": "o-dtype", "gate": "ffn-gate-dtype", "up": "up-dtype", "down": "down-dtype"}
    for role, flag in flags.items():
        if role in config["weight_dtypes"]:
            args += ["--"+flag, config["weight_dtypes"][role]]
    return args
