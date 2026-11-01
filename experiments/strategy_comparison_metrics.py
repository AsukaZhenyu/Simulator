"""Read-only timeline summaries for completed strategy simulations.

These quantities describe unions/intersections of recorded action spans. They
are not CPU/GPU hardware utilization, useful work, causal waiting time or a
critical-path decomposition. An action's elapsed span can include a service
slowdown or blackout. No profiling parameter or modeling state is changed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass


_TRANSFER_KINDS = frozenset({"COPY", "COPY_H2D", "READ_SSD", "WRITE_SSD"})
_ADMINISTRATIVE_KINDS = frozenset({"ADVANCE", "EVICT", "SHARE"})


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer nanosecond value")
    return value


def _record(event, index):
    if isinstance(event, Mapping):
        return event
    if is_dataclass(event) and not isinstance(event, type):
        return {field.name: getattr(event, field.name) for field in fields(event)}
    raise ValueError(f"events[{index}] must be an event dataclass or mapping")


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _length(intervals):
    return sum(end - start for start, end in intervals)


def _intersection_length(left, right):
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def timeline_metrics(events, makespan_ns):
    """Return integer nanosecond event-span metrics over ``[0, makespan_ns)``.

    ``events`` accepts engine Event dataclasses or corresponding dictionary
    records. Call this for a successful completed run; unfinished COMPUTE or
    transfer records are errors rather than silently omitted work. Explicit
    zero-duration work is validated and contributes zero coverage.

    ``copy_event_span_ns`` includes all transfer actions: COPY, legacy COPY_H2D,
    READ_SSD and WRITE_SSD. Overlapping transfers count once. Compute spans are
    classified by explicit cpu/gpu device, or by the recorded cpu_compute /
    gpu_compute resource when the device is absent (legacy engine contract).
    A provided resource must agree with a provided compute device.

    ADVANCE's actual engine shape is start=None, end=current time; EVICT is a
    zero-duration point. Fully untimed administrative records are also ignored.
    Their supplied times are still validated. Unknown kinds or positive-duration
    administrative intervals fail so that new engine semantics cannot disappear
    from the denominator silently. Input order is immaterial and is not mutated.
    """
    makespan_ns = _nonnegative_integer(makespan_ns, "makespan_ns")
    spans = {"cpu": [], "gpu": [], "copy": []}
    try:
        event_iterator = iter(events)
    except TypeError as error:
        raise ValueError("events must be an iterable of event records") from error
    for index, event in enumerate(event_iterator):
        row = _record(event, index)
        prefix = f"events[{index}]"
        kind = row.get("kind")
        if not isinstance(kind, str) or kind not in _TRANSFER_KINDS | _ADMINISTRATIVE_KINDS | {"COMPUTE"}:
            raise ValueError(f"{prefix}.kind is unsupported: {kind!r}")
        for name in ("t_ns", "start_ns", "end_ns"):
            value = row.get(name)
            if value is not None:
                _nonnegative_integer(value, f"{prefix}.{name}")
                if value > makespan_ns:
                    raise ValueError(f"{prefix}.{name} exceeds makespan_ns")
        start, end = row.get("start_ns"), row.get("end_ns")
        if kind in _ADMINISTRATIVE_KINDS:
            if start is None and end is None:
                continue
            if kind == "ADVANCE" and start is None:
                continue
            if start is None or end is None or start != end:
                raise ValueError(f"{prefix} administrative event must be untimed or a point")
            continue
        if start is None or end is None:
            raise ValueError(f"{prefix} {kind} requires complete start_ns and end_ns")
        if end < start:
            raise ValueError(f"{prefix}.end_ns precedes start_ns")
        if kind == "COMPUTE":
            device, resource = row.get("device"), row.get("resource")
            if resource is not None and not isinstance(resource, str):
                raise ValueError(f"{prefix}.resource must be a string or None")
            if device is None:
                device = {"cpu_compute": "cpu", "gpu_compute": "gpu"}.get(resource)
            if device not in ("cpu", "gpu"):
                raise ValueError(f"{prefix} COMPUTE requires a known cpu/gpu device or compute resource")
            if resource is not None and resource != f"{device}_compute":
                raise ValueError(f"{prefix} compute device and resource disagree")
            spans[device].append((start, end))
        else:
            spans["copy"].append((start, end))
    cpu, gpu, copy = (_union(spans[name]) for name in ("cpu", "gpu", "copy"))
    compute = _union(cpu + gpu)
    any_engine = _union(compute + copy)
    any_engine_ns = _length(any_engine)
    return {
        "cpu_event_span_ns": _length(cpu),
        "gpu_event_span_ns": _length(gpu),
        "copy_event_span_ns": _length(copy),
        "compute_event_span_ns": _length(compute),
        "cpu_gpu_event_overlap_ns": _intersection_length(cpu, gpu),
        "copy_compute_event_overlap_ns": _intersection_length(copy, compute),
        "any_engine_event_span_ns": any_engine_ns,
        "all_engine_event_uncovered_ns": makespan_ns - any_engine_ns,
    }


__all__ = ["timeline_metrics"]
