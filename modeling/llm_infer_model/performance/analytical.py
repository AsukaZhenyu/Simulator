"""Legacy analytical entry points owned by modeling, with unchanged rounding."""
from __future__ import annotations

import math

from .parameters import InvalidModel, integer


def _positive_rate(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value > 0)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise InvalidModel(f"{name} must be an explicit positive finite number")


def _duration(work, rate, label):
    try:
        duration = 1e9 * work / rate
    except (OverflowError, ZeroDivisionError) as error:
        raise InvalidModel(f"{label} cannot be represented as finite nanoseconds") from error
    if not math.isfinite(duration):
        raise InvalidModel(f"{label} cannot be represented as finite nanoseconds")
    return duration


def analytical_compute_duration_ns(*, flops, traffic_bytes, flops_per_second,
                                   memory_bytes_per_second, launch_overhead_ns):
    """Uncalibrated roofline using caller-declared effective hardware values."""
    integer(flops, "flops")
    integer(traffic_bytes, "traffic_bytes")
    integer(launch_overhead_ns, "launch_overhead_ns")
    _positive_rate(flops_per_second, "flops_per_second")
    _positive_rate(memory_bytes_per_second, "memory_bytes_per_second")
    elapsed = max(_duration(flops, flops_per_second, "compute time"),
                  _duration(traffic_bytes, memory_bytes_per_second, "memory time"))
    return max(1, math.ceil(elapsed) + launch_overhead_ns)


def analytical_transfer_duration_ns(*, storage_bytes, bytes_per_second, latency_ns):
    integer(storage_bytes, "storage_bytes", 1)
    integer(latency_ns, "latency_ns")
    _positive_rate(bytes_per_second, "bytes_per_second")
    return max(1, math.ceil(_duration(storage_bytes, bytes_per_second, "transfer time")) + latency_ns)
