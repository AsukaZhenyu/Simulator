"""One explicit CPU decoder-block contract shared by modeling and profiling.

Separate Q/K/V/(gate) matrices, RMSNorm, normal adjacent-pair RoPE, causal GQA,
O projection, SwiGLU and both residuals. All stage ports are materialized even
in the joint graph. This specifies a complete variant, not a model runtime.
"""
from __future__ import annotations

import math
import struct
from typing import Any, Mapping

from ..tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout

CONTRACT = "ggml_cpu_dense_decoder_block_v1"
SUITE_SCHEMA = "dense_layer_measurement_suite_v1"
NATIVE_SCHEMA = "ggml_dense_layer_probe/1"
STAGES = ("projection", "attention", "dense_ffn", "block")
DTYPES = {"f32": "GGML_TYPE_F32", "f16": "GGML_TYPE_F16",
          "q4_k": "GGML_TYPE_Q4_K", "q6_k": "GGML_TYPE_Q6_K", "i32": "GGML_TYPE_I32"}
VARIANTS = {stage: f"ggml_cpu_dense_materialized_{stage}_v1" for stage in STAGES}
CONFIG_KEYS = {"hidden", "intermediate", "heads", "kv_heads", "head_dim",
               "attention_gate", "qk_norm", "rope_mode", "rope_dims", "rope_base",
               "norm_epsilon", "weight_dtypes"}
EXCLUDED = ("allocation", "quantization", "io", "external_concurrency")


def integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def float32_parameter(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidInput(f"{label} must be a finite numeric parameter")
    try:
        applied = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise InvalidInput(f"{label} does not fit GGML float32") from error
    if not math.isfinite(applied) or applied <= minimum:
        raise InvalidInput(f"{label} must remain > {minimum} after GGML float32 conversion")
    return applied


def weight_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    h, i, d = config["hidden"], config["intermediate"], config["head_dim"]
    q, kv = config["heads"] * d, config["kv_heads"] * d
    result = {"input_norm": (h,), "q": (h, q), "k": (h, kv), "v": (h, kv),
              "output": (q, h), "post_attention_norm": (h,),
              "gate": (h, i), "up": (h, i), "down": (i, h)}
    if config["attention_gate"] == "sigmoid":
        result["gate_proj"] = (h, q)
    if config["qk_norm"] == "rms":
        result.update(q_norm=(d,), k_norm=(d,))
    return result


def stage_weight_roles(config: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    if kind == "projection":
        roles = ["input_norm", "q", "k", "v"]
        if config["attention_gate"] == "sigmoid":
            roles.append("gate_proj")
        if config["qk_norm"] == "rms":
            roles += ["q_norm", "k_norm"]
        return tuple(roles)
    if kind == "attention":
        return ("output",)
    if kind == "dense_ffn":
        return ("post_attention_norm", "gate", "up", "down")
    if kind == "block":
        return tuple(role for stage in STAGES[:3] for role in stage_weight_roles(config, stage))
    raise Unsupported(f"unsupported dense stage {kind!r}")


def validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise InvalidInput(f"dense config must declare exactly {sorted(CONFIG_KEYS)}")
    config = {key: integer(raw[key], f"config.{key}") for key in
              ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "rope_dims")}
    if config["heads"] % config["kv_heads"]:
        raise InvalidInput("config.heads must be divisible by kv_heads")
    if config["rope_dims"] % 2 or config["rope_dims"] > config["head_dim"]:
        raise InvalidInput("rope_dims must be positive, even and <= head_dim")
    for key, allowed in (("attention_gate", ("none", "sigmoid")), ("qk_norm", ("none", "rms"))):
        if raw[key] not in allowed:
            raise InvalidInput(f"config.{key} must be one of {allowed}")
        config[key] = raw[key]
    if raw["rope_mode"] != "normal":
        raise Unsupported("this dense contract supports normal adjacent-pair RoPE only; no fallback")
    config["rope_mode"] = "normal"
    config["rope_base"] = float32_parameter(raw["rope_base"], "config.rope_base", 1.0)
    config["norm_epsilon"] = float32_parameter(raw["norm_epsilon"], "config.norm_epsilon")
    shapes = weight_shapes(config)
    dtypes = raw["weight_dtypes"]
    if not isinstance(dtypes, dict) or set(dtypes) != set(shapes):
        raise InvalidInput(f"weight_dtypes must declare exactly {sorted(shapes)}")
    config["weight_dtypes"] = {}
    for role, shape in shapes.items():
        dtype = dtypes[role]
        if dtype not in ("f32", "f16", "q4_k", "q6_k"):
            raise InvalidInput(f"unsupported weight dtype for {role}: {dtype!r}")
        if role.endswith("norm") and dtype != "f32":
            raise Unsupported("RMSNorm gamma vectors in this contract must be f32")
        ggml_contiguous_layout(DTYPES[dtype], (*shape, *([1] * (4-len(shape)))))
        config["weight_dtypes"][role] = dtype
    return config


def validate_case(raw: Any) -> dict[str, Any]:
    required = {"id", "kind", "phase", "batch", "tokens", "visible_history", "config"}
    optional = {"threads", "comment", "source", "metadata"}
    if not isinstance(raw, dict) or not required.issubset(raw) or set(raw) - required - optional:
        raise InvalidInput("dense case has missing or unsupported fields")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise InvalidInput("dense case id must be nonempty")
    if raw["kind"] not in STAGES or raw["phase"] not in ("prefill", "decode"):
        raise InvalidInput("dense case must select an explicit stage and phase")
    if integer(raw["batch"], "case.batch") != 1:
        raise Unsupported("dense layer measurement supports batch=1 only")
    tokens = integer(raw["tokens"], "case.tokens")
    history = integer(raw["visible_history"], "case.visible_history", 0)
    if raw["phase"] == "decode" and tokens != 1:
        raise InvalidInput("decode requires one new token")
    if history + tokens > 0x7fffffff:
        raise Unsupported("absolute positions must fit the GGML I32 RoPE/mask interface")
    result = {**raw, "config": validate_config(raw["config"])}
    if "threads" in raw:
        result["threads"] = integer(raw["threads"], "case.threads")
    for key in ("comment", "source"):
        if key in raw and not isinstance(raw[key], (str, dict)):
            raise InvalidInput(f"case.{key} must be descriptive string/object")
    if "metadata" in raw and not isinstance(raw["metadata"], dict):
        raise InvalidInput("case.metadata must be descriptive object")
    return result


def build_plan(suite: Any, threads: int) -> list[dict[str, Any]]:
    integer(threads, "threads")
    if not isinstance(suite, dict) or suite.get("schema_version") != SUITE_SCHEMA:
        raise InvalidInput(f"dense suite schema_version must be {SUITE_SCHEMA}")
    if set(suite) - {"schema_version", "cases", "comment"}:
        raise InvalidInput("dense suite contains unsupported fields")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InvalidInput("dense suite must have cases")
    result = []
    for raw in cases:
        case = validate_case(raw)
        if "threads" in case and case["threads"] != threads:
            raise InvalidInput("case threads differ from requested measurement protocol")
        result.append({**case, "threads": threads})
    if len({case["id"] for case in result}) != len(result):
        raise InvalidInput("dense case ids must be unique")
    return result


def operations(config: Mapping[str, Any], kind: str) -> list[str]:
    if kind == "projection":
        result = ["input_RMSNorm", "input_norm_scale", "Q_projection", "K_projection", "V_projection"]
        if config["attention_gate"] == "sigmoid":
            result += ["gate_projection"]
        if config["qk_norm"] == "rms":
            result += ["Q_per_head_RMSNorm", "Q_norm_scale", "K_per_head_RMSNorm", "K_norm_scale"]
        result += ["RoPE_Q", "RoPE_K", "materialize_Q", "materialize_K", "materialize_V"]
        if config["attention_gate"] == "sigmoid":
            result += ["materialize_gate"]
        return result
    if kind == "attention":
        result = ["append_new_KV", "QK_transpose", "scale", "causal_mask", "softmax", "PV"]
        if config["attention_gate"] == "sigmoid":
            result += ["sigmoid_gate", "attention_times_gate"]
        return result + ["output_projection", "attention_residual_add", "materialize_attention_hidden"]
    if kind == "dense_ffn":
        return ["post_attention_RMSNorm", "post_attention_norm_scale", "gate_projection", "up_projection",
                "SiLU_gate", "gate_times_up", "down_projection", "ffn_residual_add", "materialize_hidden"]
    if kind == "block":
        return [op for stage in STAGES[:3] for op in operations(config, stage)]
    raise Unsupported(f"unsupported dense stage {kind!r}")


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, threads: int, batch: int = 1) -> dict[str, Any]:
    case = validate_case(dict(id="signature", kind=kind, phase=phase, batch=batch,
        tokens=tokens, visible_history=visible_history, config=config, threads=threads))
    config = case["config"]
    return {"contract": CONTRACT, "scope": "complete_declared_decoder_block", "kind": kind,
            "device": "cpu", "backend": "CPU", "implementation_variant": VARIANTS[kind],
            "config": config, "phase": phase, "batch": batch, "tokens": tokens,
            "visible_history": visible_history, "threads_requested": threads, "threads_applied": threads,
             "thread_management": "persistent_ggml_threadpool",
            "activation_dtype": "f32", "kv_dtype": "f32", "position_dtype": "i32",
            "numeric_parameter_semantics": "GGML_float32_parameters",
            "timing_scope": "single_completed_cpu_graph_wall_ns", "setup_compute_repetitions": 1,
            "stage_boundary_policy": "timed_materialized_copies_in_isolated_and_joint_graphs",
            "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
            "operations": operations(config, kind), "excluded_operations": list(EXCLUDED),
            "original_model_implementation_reproduced": False}


def ports(config: Mapping[str, Any], kind: str, tokens: int) -> tuple[dict, dict]:
    h, d, nh, nk = config["hidden"], config["head_dim"], config["heads"], config["kv_heads"]
    hidden = {"dtype": "f32", "ne": (h, tokens, 1, 1)}
    positions = {"dtype": "i32", "ne": (tokens, 1, 1, 1)}
    projection = {"q": {"dtype": "f32", "ne": (d, tokens, nh, 1)},
                  "k": {"dtype": "f32", "ne": (d, tokens, nk, 1)},
                  "v": {"dtype": "f32", "ne": (d, tokens, nk, 1)}}
    if config["attention_gate"] == "sigmoid":
        projection["gate"] = {"dtype": "f32", "ne": (d*nh, tokens, 1, 1)}
    if kind == "projection":
        return {"hidden": hidden, "positions": positions}, projection
    if kind == "attention":
        return {"hidden": hidden, **projection}, {"attention_hidden": hidden}
    if kind == "dense_ffn":
        return {"attention_hidden": hidden}, {"hidden_out": hidden}
    if kind == "block":
        return {"hidden": hidden, "positions": positions}, {**projection, "attention_hidden": hidden, "hidden_out": hidden}
    raise Unsupported(f"unsupported dense stage {kind!r}")


def resident_accounting(config: dict[str, Any], kind: str, tokens: int, history: int) -> dict[str, Any]:
    config = validate_config(config)
    integer(tokens, "tokens"); integer(history, "history", 0)
    shapes = weight_shapes(config)
    weight_bytes = {}
    for role in stage_weight_roles(config, kind):
        shape = shapes[role]
        _, size = ggml_contiguous_layout(DTYPES[config["weight_dtypes"][role]], (*shape, *([1]*(4-len(shape)))))
        weight_bytes[role] = size
    inputs, outputs = ports(config, kind, tokens)
    def sizes(values):
        return {name: math.prod(port["ne"]) * 4 for name, port in values.items()}
    input_sizes, output_sizes = sizes(inputs), sizes(outputs)
    kv = 8 * config["kv_heads"] * config["head_dim"] if kind in ("attention", "block") else 0
    result = {"weight_roles_bytes": weight_bytes, "input_ports_bytes": input_sizes,
              "output_ports_bytes": output_sizes, "weight_bytes": sum(weight_bytes.values()),
              "input_bytes": sum(input_sizes.values()), "output_bytes": sum(output_sizes.values()),
              "kv_history_bytes": history*kv, "kv_new_bytes": tokens*kv, "kv_cache_bytes": (history+tokens)*kv}
    result["resident_logical_bytes"] = result["weight_bytes"]+result["input_bytes"]+result["output_bytes"]+result["kv_cache_bytes"]
    return result


def native_arguments(case: dict[str, Any], warmup: int, samples: int, seed: int) -> list[str]:
    case = validate_case(case)
    integer(warmup, "warmup", 0); integer(samples, "samples", 3); integer(seed, "seed", 0)
    if samples > 10000 or seed > 0xffffffff:
        raise InvalidInput("samples or seed exceed native protocol range")
    if "threads" not in case:
        raise InvalidInput("native invocation requires explicit threads")
    config = case["config"]
    args = ["--stage", case["kind"]]
    for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim"):
        args += ["--"+key.replace("_", "-"), str(config[key])]
    args += ["--q", str(case["tokens"]), "--history", str(case["visible_history"]),
             "--threads", str(case["threads"]), "--warmup", str(warmup), "--samples", str(samples), "--seed", str(seed)]
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
