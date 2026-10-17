"""Validation and statistics for measured GGML task costs.

These records are profiling data, not M1Costs. In particular, an unknown
workspace stays ``None`` and must not be silently exported as zero bytes.
``dtype`` in a compute case denotes the weight tensor type; ``k`` is its
innermost GGML dimension: weights [k, m], activations [k, n].

All validators return a copy and preserve extra provenance fields. Input
errors raise ContractError (a ValueError). No backend is imported or run.
"""

from __future__ import annotations

import copy
import math
import statistics
from typing import Any


class ContractError(ValueError):
    """A profiling input or successful measurement violates the contract."""


_UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_DTYPES = {"f32", "f16", "q4_k", "q6_k"}
_UNKNOWN_SOURCES = {"unknown", "unmeasured", "not_measured", "none", "n/a", "unspecified"}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return copy.deepcopy(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value.strip()


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _backend(value: Any) -> str:
    name = _text(value, "backend").upper()
    if name not in {"CPU", "CUDA0"}:
        raise ContractError("backend must be CPU or CUDA0; other compute devices are not supported by this probe")
    return name


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Validate a compute or transfer case; normalize names and preserve metadata.

    Compute requires id/kind/m/k/n/dtype; transfer requires
    id/kind/direction/host_memory/size_bytes. Backend is optional until the
    runner assigns it. Threads, when supplied, must be a positive integer.
    """
    result = _object(case, "case")
    result["id"] = _text(result.get("id"), "id")
    kind = _text(result.get("kind"), "kind").lower()
    if kind not in {"compute", "transfer"}:
        raise ContractError("kind must be compute or transfer")
    result["kind"] = kind
    if "backend" in result:
        result["backend"] = _backend(result["backend"])
    if "threads" in result:
        result["threads"] = _integer(result["threads"], "threads")

    if kind == "compute":
        op = result.get("op", "MUL_MAT")
        if not isinstance(op, str) or op.upper() != "MUL_MAT":
            raise ContractError("this compute probe supports MUL_MAT only; other operations need their own probe")
        result["op"] = "MUL_MAT"
        for axis in ("m", "k", "n"):
            result[axis] = _integer(result.get(axis), axis)
        dtype = _text(result.get("dtype"), "dtype").lower()
        if dtype not in _DTYPES:
            raise ContractError(f"unsupported dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
        if dtype in {"q4_k", "q6_k"} and result["k"] % 256:
            raise ContractError("q4_k/q6_k require k (the weight row length) to be divisible by 256")
        result["dtype"] = dtype
    else:
        for field, choices in (
            ("direction", {"h2d", "d2h"}),
            ("host_memory", {"pinned", "pageable"}),
        ):
            value = _text(result.get(field), field).lower()
            if value not in choices:
                raise ContractError(f"{field} must be one of {sorted(choices)}")
            result[field] = value
        result["size_bytes"] = _integer(result.get("size_bytes"), "size_bytes")
    return result


def _samples_ns(samples: Any, unit: str) -> list[float]:
    if not isinstance(unit, str) or unit not in _UNIT_NS:
        raise ContractError(f"unit must be one of {list(_UNIT_NS)}")
    if not isinstance(samples, list) or not samples:
        raise ContractError("samples must be a nonempty list")
    values = []
    for index, value in enumerate(samples):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"samples[{index}] must be a positive finite number")
        try:
            ns = float(value) * _UNIT_NS[unit]
        except OverflowError as exc:
            raise ContractError(f"samples[{index}] overflows nanoseconds") from exc
        if not math.isfinite(ns) or ns <= 0:
            raise ContractError(f"samples[{index}] must be positive and finite in nanoseconds")
        values.append(ns)
    return values


def _percentile(ordered: list[float], fraction: float) -> float:
    # Linear interpolation with position (N-1)*p; a singleton returns itself.
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_samples(samples: list[int | float], unit: str = "ns") -> dict[str, int | float]:
    """Summarize positive samples in ns without dropping invalid observations.

    p10/p90 use linear interpolation. stdev_ns is the sample standard
    deviation (N-1 denominator), set to zero for one observation; cv is
    stdev/mean. These describe observed variability, not prediction error or
    a confidence interval. This function does not divide batched timings by
    operations per sample; the caller must retain that timing distinction.
    """
    values = _samples_ns(samples, unit)
    ordered = sorted(values)
    try:
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        result = {
            "count": len(values),
            "median_ns": _percentile(ordered, 0.5),
            "mean_ns": mean,
            "min_ns": ordered[0],
            "max_ns": ordered[-1],
            "p10_ns": _percentile(ordered, 0.1),
            "p90_ns": _percentile(ordered, 0.9),
            "stdev_ns": stdev,
            "cv": stdev / mean,
        }
    except (OverflowError, ValueError) as exc:
        raise ContractError("sample statistics cannot be represented as finite numbers") from exc
    if any(not math.isfinite(value) for value in result.values()):
        raise ContractError("sample statistics must be finite")
    return result


def validate_measurement(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one successful native measurement and attach its summary.

    ``wall_ns`` is mandatory. ``expected_samples`` and ``repetitions``, when
    present, refer to the number of timed samples, not inner operations or
    warmups, and must exactly match the observations. ``workspace_bytes`` may
    be absent/None. Any numeric value requires an explicit non-unknown
    ``workspace_source``; a benchmark must not claim zero for an unmeasured
    CUDA workspace. This function does not create M1Costs.
    """
    result = _object(payload, "measurement")
    status = result.get("status")
    if status is not None and not isinstance(status, str):
        raise ContractError("status must be a string when present")
    if status is not None and status.lower() in {"failed", "error", "unsupported"}:
        raise ContractError("failed or unsupported measurements cannot supply successful cost samples")
    if "case" in result:
        result["case"] = validate_case(result["case"])
    if "case_id" in result:
        result["case_id"] = _text(result["case_id"], "case_id")
        if "case" in result and result["case_id"] != result["case"]["id"]:
            raise ContractError("case_id does not match case.id")
    if "backend" in result:
        result["backend"] = _backend(result["backend"])
    if "threads" in result:
        result["threads"] = _integer(result["threads"], "threads")
    if "case" in result:
        for field in ("backend", "threads"):
            if field in result and field in result["case"] and result[field] != result["case"][field]:
                raise ContractError(f"measurement {field} does not match case.{field}")

    values = _samples_ns(result.get("wall_ns"), "ns")
    for field in ("expected_samples", "repetitions"):
        if field in result:
            expected = _integer(result[field], field)
            if expected != len(values):
                raise ContractError(f"{field}={expected} but received {len(values)} wall_ns samples")
    workspace = result.get("workspace_bytes")
    if workspace is None:
        result["workspace_bytes"] = None
    else:
        result["workspace_bytes"] = _integer(workspace, "workspace_bytes", minimum=0)
        source = _text(result.get("workspace_source"), "workspace_source for numeric workspace")
        source_name = source.lower()
        if source_name in _UNKNOWN_SOURCES or source_name.startswith(("unknown", "unmeasured", "not_measured", "not_observed")):
            raise ContractError("unknown workspace must be null, not a numeric byte count")
        result["workspace_source"] = source
    result["summary"] = summarize_samples(result["wall_ns"])
    return result
