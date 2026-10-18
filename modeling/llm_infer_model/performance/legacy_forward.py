"""Interpretable forward equations consumed by the legacy calibration workflow.

These effective streamed-lane and two-tier storage assumptions remain those of
the historical model. Profiling estimates their parameters and reports residuals;
it does not implement another copy of their performance equations. Seconds and
the original floating-point evaluation order are retained for compatibility.
"""
from __future__ import annotations

import math

from .analytical import _positive_rate
from .parameters import InvalidModel, integer


def _seconds(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0)
    except (ValueError, OverflowError):
        valid = False
    if not valid:
        raise InvalidModel(f"{name} must be finite nonnegative seconds")


def streamed_pipeline_seconds(*, weight_bytes, layer_count, bytes_per_second,
                              layer_latency_s, scheduler_overhead_s):
    """Return (streamed-lane service, total) with overhead charged once."""
    integer(weight_bytes, "weight_bytes")
    integer(layer_count, "layer_count")
    _positive_rate(bytes_per_second, "bytes_per_second")
    _seconds(layer_latency_s, "layer_latency_s")
    _seconds(scheduler_overhead_s, "scheduler_overhead_s")
    service = float(weight_bytes) / bytes_per_second + float(layer_count) * layer_latency_s
    return service, service + scheduler_overhead_s


def payload_transfer_seconds(*, storage_bytes, bytes_per_second):
    """Payload-only transfer service, without silently inferred startup cost."""
    integer(storage_bytes, "storage_bytes")
    _positive_rate(bytes_per_second, "bytes_per_second")
    return storage_bytes / bytes_per_second


def kv_context_seconds(*, one_way_bytes, cache_bytes, cached_bytes_per_second,
                       uncached_bytes_per_second, baseline_seconds, pcie_seconds):
    """Return (KV backing-file service, context total) for two storage tiers.

    Both writing and rereading the KV backing file are charged. The PCIe term
    is a separately modeled payload transfer; baseline and PCIe are added once.
    Allocated bytes alone do not change compute throughput.
    """
    integer(one_way_bytes, "one_way_bytes")
    integer(cache_bytes, "cache_bytes")
    _positive_rate(cached_bytes_per_second, "cached_bytes_per_second")
    _positive_rate(uncached_bytes_per_second, "uncached_bytes_per_second")
    _seconds(baseline_seconds, "baseline_seconds")
    _seconds(pcie_seconds, "pcie_seconds")
    cached_bytes = min(one_way_bytes, cache_bytes)
    uncached_bytes = max(0, one_way_bytes - cache_bytes)
    storage = 2.0 * (cached_bytes / cached_bytes_per_second
                     + uncached_bytes / uncached_bytes_per_second)
    return storage, baseline_seconds + float(pcie_seconds) + storage
