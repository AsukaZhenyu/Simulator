"""Optional read-only counters for handles of caller-owned Windows children.

This module never opens or enumerates processes. A supplied handle must belong
to a child owned by the caller; the caller retains and closes it. CPU time
includes polling/spinning and is not a measurement of useful instructions or
instantaneous physical CPU contention. No third-party package is required.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from profiling.state_calibration.telemetry import NativeClock

SCHEMA = "owned-process-resource-counters-0.1"
PROCESS_FIELDS = ("creation_time_100ns", "kernel_time_100ns", "user_time_100ns",
                  "working_set_bytes", "private_commit_bytes")
MEMORY_FIELDS = ("total_physical_bytes", "available_physical_bytes")
CPU_SEMANTICS = ("Delta of owned-process kernel plus user CPU time divided by "
                 "monotonic wall time, in core equivalents; may exceed one; "
                 "not divided by machine or affinity CPU count. Polling/spinning "
                 "counts as CPU activity; not useful-work or per-kernel utilization.")


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def integer(self) -> int:
        return (self.high << 32) | self.low


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32),
                *[(name, ctypes.c_uint64) for name in
                  ("total_phys", "avail_phys", "total_pagefile", "avail_pagefile",
                   "total_virtual", "avail_virtual", "avail_extended_virtual")]]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("page_fault_count", ctypes.c_uint32),
                *[(name, ctypes.c_size_t) for name in
                  ("peak_working_set", "working_set", "quota_peak_paged_pool",
                   "quota_paged_pool", "quota_peak_nonpaged_pool", "quota_nonpaged_pool",
                   "pagefile_usage", "peak_pagefile_usage", "private_usage")]]


def _record(values: dict[str, Any], unknown: dict[str, str]) -> dict[str, Any]:
    return {"status": "partial" if values and unknown else "ok" if values else "unavailable",
            "values": values, "unknown": unknown}


def _handle_value(handle: Any) -> int | None:
    if handle is None:
        return None
    if isinstance(handle, ctypes.c_void_p):
        handle = handle.value
    if isinstance(handle, bool):
        raise ValueError("handle must be a positive caller-owned child handle")
    try:
        value = handle.__index__()
    except (AttributeError, TypeError):
        raise ValueError("handle must be a positive caller-owned child handle") from None
    if value <= 0:
        raise ValueError("handle must be a positive caller-owned child handle")
    return value


def _bind(kernel: Any, name: str, arguments: list[Any]) -> Any:
    function = getattr(kernel, name)
    function.argtypes, function.restype = arguments, ctypes.c_int
    return function


def _error(operation: str) -> OSError:
    return OSError(f"{operation} failed (Windows error {getattr(ctypes, 'get_last_error', lambda: 0)()})")


def read_snapshot(handle: Any = None, *, clock: Any = None,
                  windows: bool | None = None, kernel: Any = None) -> dict[str, Any]:
    """Read host DRAM and, optionally, a borrowed caller-owned process handle.

    ``clock``, ``windows`` and ``kernel`` permit deterministic fixture tests.
    Missing APIs, exited children and denied queries remain explicit unknowns.
    Working set and private committed bytes are distinct OS quantities; neither
    is a GPU allocation count or proof of physical capacity available to a task.
    """
    handle_value = _handle_value(handle)
    clock = clock or NativeClock()
    windows = os.name == "nt" if windows is None else windows
    begin = clock.now()
    memory_values: dict[str, Any] = {}
    memory_unknown: dict[str, str] = {}
    process_values: dict[str, Any] = {}
    process_unknown: dict[str, str] = {}
    host_begin = host_end = process_begin = process_end = cpu_begin = cpu_end = None
    if not windows:
        memory_unknown = {key: "Windows counter provider is unavailable on this platform" for key in MEMORY_FIELDS}
        process_unknown = {key: "Windows counter provider is unavailable on this platform" for key in PROCESS_FIELDS}
    else:
        try:
            kernel = kernel if kernel is not None else ctypes.WinDLL("kernel32", use_last_error=True)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            memory_unknown = {key: reason for key in MEMORY_FIELDS}
            process_unknown = {key: reason for key in PROCESS_FIELDS}
        else:
            host_begin = clock.now()
            try:
                function = _bind(kernel, "GlobalMemoryStatusEx", [ctypes.POINTER(_MemoryStatus)])
                status = _MemoryStatus()
                status.length = ctypes.sizeof(status)
                if not function(ctypes.byref(status)):
                    raise _error("GlobalMemoryStatusEx")
                if not 0 <= status.avail_phys <= status.total_phys or status.total_phys <= 0:
                    raise ValueError("invalid host physical memory counters")
                memory_values.update(total_physical_bytes=status.total_phys, available_physical_bytes=status.avail_phys)
            except Exception as error:
                memory_unknown = {key: f"{type(error).__name__}: {error}" for key in MEMORY_FIELDS}
            host_end = clock.now()
            if handle_value is not None:
                process_begin = clock.now()
                cpu_begin = clock.now()
                try:
                    function = _bind(kernel, "GetProcessTimes", [ctypes.c_void_p, *([ctypes.POINTER(_FileTime)] * 4)])
                    created, exited, system, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
                    if not function(handle_value, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                        raise _error("GetProcessTimes")
                    process_values.update(creation_time_100ns=created.integer(), kernel_time_100ns=system.integer(), user_time_100ns=user.integer())
                except Exception as error:
                    process_unknown.update({key: f"{type(error).__name__}: {error}" for key in PROCESS_FIELDS[:3]})
                cpu_end = clock.now()
                try:
                    function = _bind(kernel, "K32GetProcessMemoryInfo", [ctypes.c_void_p, ctypes.POINTER(_ProcessMemory), ctypes.c_uint32])
                    memory = _ProcessMemory()
                    memory.cb = ctypes.sizeof(memory)
                    if not function(handle_value, ctypes.byref(memory), memory.cb):
                        raise _error("K32GetProcessMemoryInfo")
                    process_values.update(working_set_bytes=memory.working_set, private_commit_bytes=memory.private_usage)
                except Exception as error:
                    process_unknown.update({key: f"{type(error).__name__}: {error}" for key in PROCESS_FIELDS[3:]})
                process_end = clock.now()
    if handle_value is None:
        process_unknown = {key: "No caller-owned child handle supplied; no process queried" for key in PROCESS_FIELDS}
    end = clock.now()
    if end < begin:
        raise ValueError("monotonic counter query clock moved backwards")
    return {"schema": SCHEMA, "clock": clock.metadata(), "query_begin_tick": begin, "query_end_tick": end,
            "query_duration_ticks": end - begin,
            "host_memory": {**_record(memory_values, memory_unknown), "provider": "GlobalMemoryStatusEx",
                            "query_begin_tick": host_begin, "query_end_tick": host_end,
                            "scope": "host_physical_memory; not task-specific available capacity"},
            "process": {**_record(process_values, process_unknown), "provider": "GetProcessTimes+K32GetProcessMemoryInfo",
                        "handle_supplied": handle_value is not None, "handle_identity": handle_value,
                        "query_begin_tick": process_begin, "query_end_tick": process_end,
                        "cpu_counter_begin_tick": cpu_begin, "cpu_counter_end_tick": cpu_end,
                        "scope": "borrowed caller-owned child handle only; no process opened or enumerated"}}


def delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compute owned CPU activity between two counter observations, exactly.

    The interval endpoints are CPU-counter query completion ticks when
    available, otherwise the snapshot query completion ticks. The OS CPU
    counters are not atomically read with QPC; both original query spans remain
    available in the snapshots. Nonpositive intervals/resets remain unknown.
    """
    if left.get("schema") != SCHEMA or right.get("schema") != SCHEMA:
        raise ValueError("counter snapshot schema differs")
    lclock, rclock = left.get("clock", {}), right.get("clock", {})
    if (lclock.get("clock_id"), lclock.get("frequency_hz")) != (rclock.get("clock_id"), rclock.get("frequency_hz")):
        raise ValueError("counter clocks differ")
    frequency = rclock.get("frequency_hz")
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
        raise ValueError("counter clock frequency must be a positive integer")
    lp, rp = left["process"], right["process"]
    if lp.get("handle_identity") != rp.get("handle_identity"):
        raise ValueError("owned process handle changed between observations")
    lv, rv = lp.get("values", {}), rp.get("values", {})
    if ("creation_time_100ns" in lv and "creation_time_100ns" in rv
            and lv["creation_time_100ns"] != rv["creation_time_100ns"]):
        raise ValueError("owned process creation identity changed")
    begin = lp.get("cpu_counter_end_tick") if lp.get("cpu_counter_end_tick") is not None else left["query_end_tick"]
    end = rp.get("cpu_counter_end_tick") if rp.get("cpu_counter_end_tick") is not None else right["query_end_tick"]
    result = {"interval_begin_tick": begin, "interval_end_tick": end, "interval_clock": dict(rclock),
              "normalization": CPU_SEMANTICS, "exact_core_equivalents": None}
    fields = ("kernel_cpu_time_100ns", "user_cpu_time_100ns", "cpu_time_100ns", "core_equivalents")
    reason = None
    if lp.get("handle_identity") is None:
        reason = "No caller-owned child handle supplied"
    elif any(key not in lv or key not in rv for key in ("kernel_time_100ns", "user_time_100ns")):
        reason = "One or both owned-process CPU counter reads are unknown"
    elif any(isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 0
             for values in (lv, rv) for key in ("kernel_time_100ns", "user_time_100ns")):
        reason = "Owned-process CPU counter value is invalid"
    elif isinstance(begin, bool) or isinstance(end, bool) or not isinstance(begin, int) or not isinstance(end, int) or end <= begin:
        reason = "Monotonic observation interval is nonpositive or invalid"
    if reason:
        return {**result, **_record({}, {key: reason for key in fields})}
    system = rv["kernel_time_100ns"] - lv["kernel_time_100ns"]
    user = rv["user_time_100ns"] - lv["user_time_100ns"]
    if (isinstance(system, bool) or isinstance(user, bool) or not isinstance(system, int) or not isinstance(user, int)
            or system < 0 or user < 0):
        return {**result, **_record({}, {key: "Owned-process CPU counter reset or invalid counter delta" for key in fields})}
    ratio = Fraction((system + user) * frequency, 10_000_000 * (end - begin))
    return {**result, **_record({"kernel_cpu_time_100ns": system, "user_cpu_time_100ns": user,
                               "cpu_time_100ns": system + user, "core_equivalents": float(ratio)}, {}),
            "exact_core_equivalents": {"numerator": ratio.numerator, "denominator": ratio.denominator}}


class ProcessCounterSampler:
    """Optional bounded-cadence sampler of caller-owned children and host DRAM.

    No process handles are opened/closed. ``set_handles`` may update a label's
    owned child between phases; its previous counter is discarded on change.
    Deadlines, actual query spans and missed periods are all retained. The OS
    wait has no guaranteed resolution; samples must be checked before use.
    """

    def __init__(self, handles: Mapping[str, Any] | None = None, *, interval_ms: float = 50,
                 clock: Any = None, reader: Callable[..., dict[str, Any]] = read_snapshot) -> None:
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, (int, float)) or not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("interval_ms must be finite and positive")
        self.clock, self.reader = clock or NativeClock(), reader
        self.interval_ms = float(interval_ms)
        self.period_ticks = max(1, round(self.clock.frequency_hz * self.interval_ms / 1000))
        self.records: list[dict[str, Any]] = []
        self._handles: dict[str, int] = {}
        self._previous: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = self._stopped = False
        self._thread_error: str | None = None
        self.set_handles(handles or {})

    def set_handles(self, handles: Mapping[str, Any]) -> None:
        normalized: dict[str, int] = {}
        for name, handle in handles.items():
            if not isinstance(name, str) or not name or name == "host":
                raise ValueError("owned child labels must be nonempty strings other than host")
            value = _handle_value(handle)
            if value is None:
                raise ValueError("owned child handle may not be None")
            normalized[name] = value
        with self._lock:
            for name in set(self._previous):
                if self._handles.get(name) != normalized.get(name):
                    self._previous.pop(name)
            self._handles = normalized

    def _collect(self, deadline: int) -> dict[str, Any]:
        begin = self.clock.now()
        snapshots, activities = {}, {}
        with self._lock:
            handles = {"host": None, **self._handles}
            for name, handle in handles.items():
                try:
                    snapshot = self.reader(handle, clock=self.clock)
                    snapshots[name] = snapshot
                    if name != "host" and name in self._previous:
                        activities[name] = delta(self._previous[name], snapshot)
                    elif name != "host":
                        activities[name] = {**_record({}, {"core_equivalents": "No previous owned-process counter read"}),
                                            "exact_core_equivalents": None, "normalization": CPU_SEMANTICS}
                    self._previous[name] = snapshot
                except Exception as error:
                    snapshots[name] = {"status": "error", "values": {}, "unknown": {"provider": f"{type(error).__name__}: {error}"}}
                    self._previous.pop(name, None)
        end = self.clock.now()
        if end < begin:
            raise ValueError("sampler monotonic clock moved backwards")
        return {"sample_index": len(self.records), "deadline_tick": deadline, "query_begin_tick": begin,
                "query_end_tick": end, "query_duration_ticks": end - begin,
                "lateness_ticks": max(0, begin - deadline), "missed_periods": max(0, (end - deadline) // self.period_ticks),
                "snapshots": snapshots, "activities": activities}

    def _run(self) -> None:
        try:
            deadline = self.clock.now()
            while not self._stop.is_set():
                row = self._collect(deadline)
                self.records.append(row)
                deadline += (row["missed_periods"] + 1) * self.period_ticks
                self._stop.wait(max(0, (deadline - self.clock.now()) / self.clock.frequency_hz))
        except Exception as error:
            self._thread_error = f"{type(error).__name__}: {error}"

    def start(self) -> ProcessCounterSampler:
        if self._started:
            raise RuntimeError("owned-process sampler may only start once")
        self._started = True
        self._thread = threading.Thread(target=self._run, name="owned-process-counter-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self._started:
            raise RuntimeError("owned-process sampler was not started")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("owned-process sampler did not stop within timeout")
        self._stopped = True

    def metadata(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "clock": self.clock.metadata(), "interval_ms": self.interval_ms,
                "period_ticks": self.period_ticks, "sample_count": len(self.records),
                "started": self._started, "stopped": self._stopped, "thread_error": self._thread_error,
                "wait_policy": "threading.Event.wait; no resolution guarantee or global timer changes",
                "cpu_semantics": CPU_SEMANTICS, "process_enumeration": False,
                "handles_opened_or_closed": False, "physical_contention_claimed": False}

    def write_jsonl(self, path: str | Path) -> None:
        if self._started and not self._stopped:
            raise RuntimeError("stop the owned-process sampler before writing")
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            for row in self.records:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
