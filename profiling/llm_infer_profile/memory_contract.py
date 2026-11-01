"""Explicit allocation and timing boundaries for future memory measurements.

This helper validates metadata only. It does not run a native probe, turn a
separate-buffer measurement into a shared-buffer measurement, or infer a zero
handoff cost from the absence of payload movement.
"""
from __future__ import annotations

from copy import deepcopy


def validate_memory_context(document):
    """Return a checked copy; preserve unknown measured time as ``None``.

    Required keys: memory_mode, allocation_kind, timing_boundary, timing_source,
    duration_ns, bytes_transferred. Measured values require an evidence string.
    Shared visibility may have zero bytes while measured synchronization still
    takes nonzero time. An actual copy always has an independent destination.
    """
    if not isinstance(document, dict):
        raise ValueError("memory context must be an object")
    result = deepcopy(document)
    required = {"memory_mode", "allocation_kind", "timing_boundary", "timing_source",
                "duration_ns", "bytes_transferred"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"memory context is missing required fields: {sorted(missing)}")
    mode = result.get("memory_mode")
    if mode not in ("separate", "shared"):
        raise ValueError("memory_mode must be separate or shared")
    allocation = result.get("allocation_kind")
    if allocation not in ("separate_buffers", "shared_buffer"):
        raise ValueError("allocation_kind must be separate_buffers or shared_buffer")
    if mode == "separate" and allocation == "shared_buffer":
        raise ValueError("shared_buffer requires shared memory mode")
    boundary = result.get("timing_boundary")
    if boundary not in ("actual_copy", "shared_visibility", "synchronize", "compute"):
        raise ValueError("unknown timing boundary")
    if boundary == "shared_visibility" and allocation != "shared_buffer":
        raise ValueError("shared_visibility requires the same shared buffer")
    if boundary == "actual_copy" and allocation != "separate_buffers":
        raise ValueError("actual_copy requires an independent destination buffer")
    source = result.get("timing_source")
    if source not in ("measured", "declared", "unknown"):
        raise ValueError("timing_source must be measured, declared or unknown")
    duration = result.get("duration_ns")
    if source == "unknown":
        if duration is not None:
            raise ValueError("unknown timing must retain duration_ns=None")
    elif type(duration) is not int or duration < 0:
        raise ValueError("known duration_ns must be a nonnegative integer")
    if source == "measured" and not (
            isinstance(result.get("evidence"), str) and result["evidence"].strip()):
        raise ValueError("measured memory timing needs explicit evidence")
    payload = result.get("bytes_transferred")
    if type(payload) is not int or payload < 0:
        raise ValueError("bytes_transferred must be a nonnegative integer")
    if boundary == "shared_visibility" and payload != 0:
        raise ValueError("same-buffer visibility moves zero payload bytes")
    if boundary == "actual_copy" and payload == 0:
        raise ValueError("actual copy requires positive payload bytes")
    return result


__all__ = ["validate_memory_context"]
