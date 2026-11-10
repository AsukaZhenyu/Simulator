"""CUDA execution envelope for the declared sparse decoder block.

The sparse block's *mathematics* is declared once, in `moe_contract`: geometry, routing
variant, weight roles, operation list, ports and accounting.  This module adds only what is
different about running that same graph on an accelerator: a CUDA device, no CPU fallback,
a disabled CUDA graph, a cache element type, and a timing scope that names a completed CUDA
graph instead of a CPU thread pool.

Sharing the mathematics does not permit sharing prices: a CPU sparse price and a CUDA one
answer different keys.
"""
from __future__ import annotations

from typing import Any

from ..tensor.spec import InvalidInput, Unsupported
from . import moe_contract
from .moe_contract import EXCLUDED, SUITE_SCHEMA

CONTRACT = "ggml_cuda_moe_decoder_block_v1"
NATIVE_SCHEMA = "ggml_dense_layer_cuda_probe/1"
# Same rule as the dense CUDA envelope: the cache element type is a property of the
# measurement, so it is part of the price's identity, and f32 stays the default.
KV_DTYPES = {"f32": 4, "f16": 2}
DEFAULT_KV_DTYPE = "f32"
STAGES = moe_contract.STAGES


def variant(kind: str, kv_dtype: str = DEFAULT_KV_DTYPE) -> str:
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA sparse kv_dtype must be one of {sorted(KV_DTYPES)}")
    return f"ggml_cuda_moe_materialized_{kind}_" + ("v2" if kv_dtype == "f16" else "v1")


def resident_accounting(config: dict[str, Any], kind: str, tokens: int, history: int,
                        kv_dtype: str = DEFAULT_KV_DTYPE) -> dict[str, Any]:
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA sparse kv_dtype must be one of {sorted(KV_DTYPES)}")
    return moe_contract.resident_accounting(config, kind, tokens, history,
                                            kv_element_bytes=KV_DTYPES[kv_dtype])


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, cuda_device: int, batch: int = 1,
              cuda_graphs: str = "disabled",
              kv_dtype: str = DEFAULT_KV_DTYPE) -> dict[str, Any]:
    if cuda_graphs != "disabled":
        raise Unsupported("this CUDA contract requires cuda_graphs='disabled'; no fallback")
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA sparse kv_dtype must be one of {sorted(KV_DTYPES)}")
    # The CPU signature is the authority on everything that is not about the device; the
    # process fields it stamps are what this envelope has to replace.
    cpu = moe_contract.signature(config, kind, phase, tokens, visible_history, threads=1,
                                 batch=batch)
    result = {key: value for key, value in cpu.items()
              if key not in ("threads_requested", "threads_applied", "thread_management",
                             "timing_scope", "device", "backend", "contract",
                             "implementation_variant", "kv_dtype")}
    result.update({"contract": CONTRACT, "device": "gpu", "backend": f"CUDA{cuda_device}",
                   "cuda_device": cuda_device, "cuda_graphs": cuda_graphs,
                   "backend_execution": "single_cuda_backend_no_cpu_fallback",
                   "matmul_precision": "ggml_cuda_backend_default",
                   "implementation_variant": variant(kind, kv_dtype),
                   "kv_dtype": kv_dtype,
                   "timing_scope": "single_completed_cuda_graph_wall_ns"})
    return result


def validate_case(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidInput("CUDA sparse case must be an explicit object")
    if "threads" in raw:
        raise Unsupported("CUDA sparse case has no CPU threads parameter")
    math_case = moe_contract.validate_case(
        {key: value for key, value in raw.items() if key not in ("cuda_device", "cuda_graphs",
                                                                 "kv_dtype")})
    if "cuda_device" in raw:
        math_case["cuda_device"] = moe_contract.integer(raw["cuda_device"],
                                                        "case.cuda_device", 0)
    if "cuda_graphs" in raw:
        if raw["cuda_graphs"] != "disabled":
            raise Unsupported("CUDA sparse case requires cuda_graphs='disabled'")
        math_case["cuda_graphs"] = raw["cuda_graphs"]
    kv_dtype = raw.get("kv_dtype", DEFAULT_KV_DTYPE)
    if kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA sparse kv_dtype must be one of {sorted(KV_DTYPES)}")
    math_case["kv_dtype"] = kv_dtype
    return math_case


def build_plan(suite: Any, cuda_device: int, cuda_graphs: str = "disabled",
               kv_dtype: str | None = None) -> list[dict[str, Any]]:
    moe_contract.integer(cuda_device, "cuda_device", 0)
    if cuda_graphs != "disabled":
        raise Unsupported("CUDA sparse plan requires cuda_graphs='disabled'")
    if kv_dtype is not None and kv_dtype not in KV_DTYPES:
        raise Unsupported(f"CUDA sparse kv_dtype must be one of {sorted(KV_DTYPES)}")
    if not isinstance(suite, dict) or suite.get("schema_version") != SUITE_SCHEMA:
        raise InvalidInput(f"sparse suite schema_version must be {SUITE_SCHEMA}")
    if set(suite) - {"schema_version", "cases", "comment"}:
        raise InvalidInput("CUDA sparse suite contains unsupported fields")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InvalidInput("CUDA sparse suite must have cases")
    result = []
    for raw in cases:
        case = validate_case(raw)
        if "cuda_device" in case and case["cuda_device"] != cuda_device:
            raise InvalidInput("case CUDA device differs from requested measurement protocol")
        if "cuda_graphs" in case and case["cuda_graphs"] != cuda_graphs:
            raise InvalidInput("case CUDA graph mode differs from requested measurement protocol")
        if raw.get("kv_dtype") is not None and kv_dtype is not None and raw["kv_dtype"] != kv_dtype:
            raise InvalidInput("case cache dtype differs from requested measurement protocol")
        result.append({**case, "cuda_device": cuda_device, "cuda_graphs": cuda_graphs,
                       "kv_dtype": kv_dtype or case["kv_dtype"]})
    if len({case["id"] for case in result}) != len(result):
        raise InvalidInput("CUDA sparse case ids must be unique")
    return result


def native_arguments(case: dict[str, Any], warmup: int, samples: int, seed: int) -> list[str]:
    case = validate_case(case)
    moe_contract.integer(warmup, "warmup", 0)
    moe_contract.integer(samples, "samples", 3)
    moe_contract.integer(seed, "seed", 0)
    if samples > 10000 or seed > 0xffffffff:
        raise InvalidInput("samples or seed exceed native protocol range")
    if "cuda_device" not in case or "cuda_graphs" not in case:
        raise InvalidInput("CUDA native invocation requires explicit cuda_device and cuda_graphs")
    config = case["config"]
    # The CPU command is the same graph on a different device, so it is the base; the
    # device, the graph mode and the cache type are what this envelope changes.
    math_case = {key: value for key, value in case.items()
                 if key not in ("cuda_device", "cuda_graphs", "kv_dtype")}
    args = moe_contract.native_arguments({**math_case, "threads": 1}, warmup, samples, seed)
    for flag, value in (("--threads", None), ("--kv-dtype", None)):
        if flag in args:
            at = args.index(flag)
            del args[at:at + 2]
    args += ["--cuda-device", str(case["cuda_device"]), "--cuda-graphs", case["cuda_graphs"],
             "--kv-dtype", case["kv_dtype"]]
    for role, flag in (("q_bias", "q-bias-dtype"), ("k_bias", "k-bias-dtype"),
                       ("v_bias", "v-bias-dtype")):
        if role in config["weight_dtypes"]:
            args += ["--" + flag, config["weight_dtypes"][role]]
    return args


def operations(config: dict[str, Any], kind: str) -> list[str]:
    return moe_contract.operations(config, kind)


def ports(config: dict[str, Any], kind: str, tokens: int) -> tuple[dict, dict]:
    return moe_contract.ports(config, kind, tokens)


def validate_config(raw: Any) -> dict[str, Any]:
    return moe_contract.validate_config(raw)


__all__ = ["CONTRACT", "NATIVE_SCHEMA", "EXCLUDED", "STAGES", "SUITE_SCHEMA", "build_plan",
           "native_arguments", "operations", "ports", "resident_accounting", "signature",
           "validate_case", "validate_config", "variant"]
