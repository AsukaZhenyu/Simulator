"""Read-only, bounded device observations on the native probe's monotonic clock.

NVML utilization is a driver-defined rolling observation, not instantaneous
contention. Samples are never interpolated into an action shorter than cadence.
This module has no third-party dependencies and never changes device settings.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable

SCHEMA = "device-state-telemetry-0.1"
ALIGNMENT_SCHEMA = "device-state-alignment-0.1"


class _NVMLUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NVMLMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _EventWaiter:
    """Explicit portable fallback, with no claimed high-resolution guarantee."""

    def __init__(self, stop_event: threading.Event, reason: str) -> None:
        self.stop_event = stop_event
        self.reason = reason

    def wait(self, seconds: float) -> bool:
        return self.stop_event.wait(seconds)

    def interrupt(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {"requested": "private_high_resolution_waitable_timer" if os.name == "nt" else "portable_interruptible_wait",
                "actual": "threading.Event.wait", "high_resolution_supported": False,
                "fallback_reason": self.reason, "global_timer_settings_modified": False,
                "busy_spin": False, "cadence_guaranteed": False}


class _WindowsHighResolutionWaiter:
    """Process-owned one-shot timer; no global timer-resolution changes."""

    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
    TIMER_MODIFY_STATE_AND_SYNCHRONIZE = 0x00100002
    INFINITE = 0xFFFFFFFF

    def __init__(self, kernel: Any = None) -> None:
        self.kernel = kernel if kernel is not None else ctypes.WinDLL("kernel32", use_last_error=True)
        self._timer = self._stop_handle = None
        self._closed = False
        self._close_errors: list[str] = []
        try:
            self._create_timer = self._bind("CreateWaitableTimerExW", [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32], ctypes.c_void_p)
            self._create_event = self._bind("CreateEventW", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p], ctypes.c_void_p)
            self._set_timer = self._bind("SetWaitableTimer", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_int)
            self._wait_multiple = self._bind("WaitForMultipleObjects", [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_uint32], ctypes.c_uint32)
            self._set_event = self._bind("SetEvent", [ctypes.c_void_p], ctypes.c_int)
            self._close_handle = self._bind("CloseHandle", [ctypes.c_void_p], ctypes.c_int)
            self._timer = self._create_timer(None, None, self.CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, self.TIMER_MODIFY_STATE_AND_SYNCHRONIZE)
            if not self._timer:
                raise self._failure("CreateWaitableTimerExW(HIGH_RESOLUTION)")
            self._stop_handle = self._create_event(None, True, False, None)
            if not self._stop_handle:
                raise self._failure("CreateEventW")
        except Exception:
            self.close()
            raise

    def _bind(self, name: str, arguments: list[Any], result: Any) -> Any:
        function = getattr(self.kernel, name)
        function.argtypes = arguments
        function.restype = result
        return function

    @staticmethod
    def _failure(operation: str) -> OSError:
        last_error = getattr(ctypes, "get_last_error", lambda: 0)()
        return OSError(f"{operation} failed (Windows error {last_error})")

    def wait(self, seconds: float) -> bool:
        if self._closed:
            raise RuntimeError("private high-resolution waiter closed")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wait delay must be finite and nonnegative")
        if seconds == 0:
            return False
        due_100ns = max(1, math.ceil(seconds * 10_000_000))
        if due_100ns > (1 << 63) - 1:
            raise ValueError("wait delay exceeds signed Windows relative timer range")
        due = ctypes.c_longlong(-due_100ns)  # Negative means relative, not UTC.
        if not self._set_timer(self._timer, ctypes.byref(due), 0, None, None, False):
            raise self._failure("SetWaitableTimer")
        handles = (ctypes.c_void_p * 2)(self._stop_handle, self._timer)
        status = self._wait_multiple(2, handles, False, self.INFINITE)
        if status not in (0, 1):
            raise self._failure(f"WaitForMultipleObjects returned {status}")
        return status == 0

    def interrupt(self) -> None:
        if not self._closed and not self._set_event(self._stop_handle):
            raise self._failure("SetEvent")

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self, "_close_handle", None)
        if close is not None:
            for handle in (self._stop_handle, self._timer):
                if handle:
                    if not close(handle):
                        self._close_errors.append(str(self._failure("CloseHandle")))
        self._closed = True

    def metadata(self) -> dict[str, Any]:
        return {"requested": "private_high_resolution_waitable_timer",
                "actual": "CreateWaitableTimerExW(HIGH_RESOLUTION)+WaitForMultipleObjects(stop,timer)",
                "high_resolution_supported": True, "fallback_reason": None,
                "global_timer_settings_modified": False, "busy_spin": False,
                "cadence_guaranteed": False, "closed": self._closed,
                "close_errors": self._close_errors,
                "due_time_semantics": "one-shot negative relative 100ns delay derived from remaining monotonic deadline; no APC, resume or periodic/global timer change"}


def _make_waiter(stop_event: threading.Event, *, windows: bool | None = None, kernel: Any = None) -> Any:
    windows = os.name == "nt" if windows is None else windows
    if windows:
        try:
            return _WindowsHighResolutionWaiter(kernel)
        except Exception as error:
            return _EventWaiter(stop_event, f"Private Windows high-resolution timer unavailable: {type(error).__name__}: {error}")
    return _EventWaiter(stop_event, "Windows high-resolution timer not applicable on this platform; portable OS wait has no guaranteed cadence")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class NativeClock:
    """The same process-independent clock used by the C++ state probe."""

    def __init__(self) -> None:
        if os.name == "nt":
            self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self._counter = self._kernel.QueryPerformanceCounter
            self._counter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            self._counter.restype = ctypes.c_int
            frequency = ctypes.c_longlong()
            query = self._kernel.QueryPerformanceFrequency
            query.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            query.restype = ctypes.c_int
            if not query(ctypes.byref(frequency)) or frequency.value <= 0:
                raise OSError("QueryPerformanceFrequency failed")
            self.frequency_hz = frequency.value
            self.clock_id = "windows_qpc"
        else:
            # CPython monotonic_ns uses CLOCK_MONOTONIC on supported POSIX hosts.
            implementation = time.get_clock_info("monotonic").implementation
            if "CLOCK_MONOTONIC" not in implementation:
                raise OSError(f"CLOCK_MONOTONIC unavailable: {implementation}")
            self.frequency_hz = 1_000_000_000
            self.clock_id = "posix_clock_monotonic"

    def now(self) -> int:
        if self.clock_id == "windows_qpc":
            value = ctypes.c_longlong()
            if not self._counter(ctypes.byref(value)):
                raise OSError("QueryPerformanceCounter failed")
            return value.value
        return time.monotonic_ns()

    def metadata(self) -> dict[str, Any]:
        return {"clock_id": self.clock_id, "frequency_hz": self.frequency_hz,
                "cross_process": True, "wall_clock_used_for_alignment": False}


def _result(values: dict[str, Any], unknown: dict[str, str], **extra: Any) -> dict[str, Any]:
    return {"status": "partial" if values and unknown else "ok" if values else "unavailable",
            "values": values, "unknown": unknown, **extra}


class NVMLProvider:
    """Minimal direct NVML binding. Every unsupported field remains unknown."""

    FIELDS = ("pstate", "sm_clock_mhz", "memory_clock_mhz", "graphics_clock_mhz",
              "temperature_c", "power_w", "gpu_util_percent", "memory_util_percent",
              "memory_used_bytes", "memory_total_bytes", "memory_free_bytes")

    def __init__(self, device_index: int = 0, library: Any = None) -> None:
        self.device_index = _integer(device_index, "device_index")
        self._lib = None
        self._handle = ctypes.c_void_p()
        self._initialized = False
        self._closed = False
        self._error = None
        self._functions: dict[str, Any] = {}
        self._info: dict[str, Any] = {
            "provider": "nvml-ctypes", "device_index": device_index, "device_uuid": None,
            "query_mode": "read_only", "subprocess_per_sample": False,
            "utilization_semantics": "Driver-defined rolling sample window; NVML documents product-dependent windows approximately 1 second to 1/6 second. Window boundaries are not exposed by these queries; values do not establish instantaneous activity, contention, kernel concurrency or causality.",
            "gpu_util_semantics": "Percent of rolling interval with one or more kernels executing; includes the measured workload.",
            "memory_util_semantics": "Percent of rolling interval during which device memory was read or written, not bandwidth saturation or memory capacity fraction.",
            "power_semantics": "NVML power usage is averaged over 1 second on Ampere (except GA100) and newer GPUs; architecture-dependent instantaneous semantics on older devices. This query does not expose averaging boundaries.",
            "clock_semantics": "Observed clock query; not a per-action effective throughput measurement.",
            "memory_semantics": "Device-wide NVML occupancy, not process allocator bytes or private workspace.",
        }
        try:
            self._lib = library if library is not None else self._load()
            init = self._function(("nvmlInit_v2", "nvmlInit"), [])
            self._check(init(), "nvmlInit")
            self._initialized = True
            handle = self._function(("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"),
                                    [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
            self._check(handle(device_index, ctypes.byref(self._handle)), "nvmlDeviceGetHandleByIndex")
            self._info["device_uuid"] = self._string("nvmlDeviceGetUUID", self._handle)
            self._info["device_name"] = self._string("nvmlDeviceGetName", self._handle)
            self._info["nvml_version"] = self._string("nvmlSystemGetNVMLVersion")
            self._info["driver_version"] = self._string("nvmlSystemGetDriverVersion")
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            # A failed device lookup must release a successful initialization.
            if self._initialized:
                self.close()

    def _load(self) -> Any:
        candidates = ["nvml.dll", str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvml.dll")] if os.name == "nt" else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        failures = []
        for candidate in candidates:
            try:
                library = ctypes.CDLL(candidate)
                self._info["library_candidate"] = candidate
                return library
            except OSError as error:
                failures.append(f"{candidate}: {error}")
        raise OSError("; ".join(failures))

    def _function(self, names: tuple[str, ...], argtypes: list[Any]) -> Any:
        for name in names:
            if name in self._functions:
                return self._functions[name]
            try:
                function = getattr(self._lib, name)
            except AttributeError:
                continue
            function.argtypes = argtypes
            function.restype = ctypes.c_int
            self._functions[name] = function
            return function
        raise OSError(f"NVML symbol unavailable: {' or '.join(names)}")

    def _check(self, status: int, operation: str) -> None:
        if status != 0:
            raise OSError(f"{operation} failed: NVML status {status}")

    def _string(self, symbol: str, handle: Any = None) -> str | None:
        try:
            buffer = ctypes.create_string_buffer(128)
            arguments = [ctypes.c_char_p, ctypes.c_uint]
            call = [buffer, len(buffer)]
            if handle is not None:
                arguments.insert(0, ctypes.c_void_p)
                call.insert(0, handle)
            function = self._function((symbol,), arguments)
            self._check(function(*call), symbol)
            return buffer.value.decode("utf-8", errors="replace")
        except Exception as error:
            self._info.setdefault("identity_unknown", {})[symbol] = f"{type(error).__name__}: {error}"
            return None

    def metadata(self) -> dict[str, Any]:
        return {**self._info, "available": self._error is None, "closed": self._closed,
                "error": self._error, "symbols": sorted(self._functions)}

    def read(self) -> dict[str, Any]:
        if self._error or self._closed:
            reason = self._error or "NVML provider closed"
            return _result({}, {field: reason for field in self.FIELDS})
        values: dict[str, Any] = {}
        unknown: dict[str, str] = {}

        def query(field: str, symbol: str, parameters: tuple[int, ...] = (), scale: float = 1) -> None:
            try:
                value = ctypes.c_uint()
                function = self._function((symbol,), [ctypes.c_void_p] + [ctypes.c_uint] * len(parameters) + [ctypes.POINTER(ctypes.c_uint)])
                self._check(function(self._handle, *parameters, ctypes.byref(value)), symbol)
                values[field] = value.value if scale == 1 else value.value * scale
            except Exception as error:
                unknown[field] = f"{type(error).__name__}: {error}"

        query("pstate", "nvmlDeviceGetPerformanceState")
        query("sm_clock_mhz", "nvmlDeviceGetClockInfo", (1,))
        query("memory_clock_mhz", "nvmlDeviceGetClockInfo", (2,))
        query("graphics_clock_mhz", "nvmlDeviceGetClockInfo", (0,))
        query("temperature_c", "nvmlDeviceGetTemperature", (0,))
        query("power_w", "nvmlDeviceGetPowerUsage", scale=0.001)

        for symbol, structure, fields in (
            ("nvmlDeviceGetUtilizationRates", _NVMLUtilization, {"gpu_util_percent": "gpu", "memory_util_percent": "memory"}),
            ("nvmlDeviceGetMemoryInfo", _NVMLMemory, {"memory_used_bytes": "used", "memory_total_bytes": "total", "memory_free_bytes": "free"}),
        ):
            try:
                value = structure()
                function = self._function((symbol,), [ctypes.c_void_p, ctypes.POINTER(structure)])
                self._check(function(self._handle, ctypes.byref(value)), symbol)
                values.update({field: getattr(value, attribute) for field, attribute in fields.items()})
            except Exception as error:
                unknown.update({field: f"{type(error).__name__}: {error}" for field in fields})
        return _result(values, unknown)

    def close(self) -> None:
        if self._initialized and not self._closed:
            try:
                function = self._function(("nvmlShutdown",), [])
                self._check(function(), "nvmlShutdown")
            except Exception as error:
                self._info["shutdown_error"] = f"{type(error).__name__}: {error}"
        self._closed = True


class CPUBusyProvider:
    """Machine-wide CPU busy fraction over consecutive OS counter reads."""

    def __init__(self, clock: Any = None, counter_reader: Any = None) -> None:
        self.clock = clock or NativeClock()
        self._previous: tuple[int, int, int] | None = None
        self._reader = counter_reader or self._system_counters

    def _system_counters(self) -> tuple[int, int]:
        if os.name == "nt":
            class FileTime(ctypes.Structure):
                _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            function = kernel.GetSystemTimes
            function.argtypes = [ctypes.POINTER(FileTime)] * 3
            function.restype = ctypes.c_int
            idle, system, user = FileTime(), FileTime(), FileTime()
            if not function(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
                raise OSError("GetSystemTimes failed")
            to_int = lambda item: (item.high << 32) | item.low
            # Windows kernel time includes idle; never add idle twice.
            return to_int(system) + to_int(user), to_int(idle)
        line = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
        if not line or line[0] != "cpu" or len(line) < 9:
            raise OSError("aggregate /proc/stat CPU counters unavailable")
        counters = [int(item) for item in line[1:9]]
        # guest/guest_nice are already included in user/nice and are not added.
        return sum(counters), counters[3] + counters[4]

    def metadata(self) -> dict[str, Any]:
        count = os.cpu_count()
        return {"provider": "GetSystemTimes" if os.name == "nt" else "/proc/stat",
                "scope": "primary_processor_group" if os.name == "nt" and count is not None and count > 64 else "machine_wide",
                "semantics": "Busy CPU time / total CPU time between consecutive successful OS reads; Windows kernel includes idle and scopes >64-CPU hosts to primary processor group; Linux iowait treated as idle; not per-process contention or frequency.",
                "logical_cpu_count": count, "interval_clock": self.clock.metadata()}

    def read(self) -> dict[str, Any]:
        try:
            begin = self.clock.now()
            total, idle = self._reader()
            end = self.clock.now()
            if total < 0 or idle < 0 or idle > total:
                raise ValueError("invalid OS CPU counters")
            previous = self._previous
            self._previous = (total, idle, end)
            if previous is None:
                return _result({}, {"busy_percent": "No previous counter read; interval undefined"},
                               counter_read_begin_tick=begin, counter_read_end_tick=end, interval_begin_tick=None, interval_end_tick=end)
            delta_total, delta_idle = total - previous[0], idle - previous[1]
            if delta_total <= 0 or delta_idle < 0 or delta_idle > delta_total:
                return _result({}, {"busy_percent": "Counter reset, zero interval or inconsistent delta"},
                               counter_read_begin_tick=begin, counter_read_end_tick=end, interval_begin_tick=previous[2], interval_end_tick=end)
            return _result({"busy_percent": 100.0 * (delta_total - delta_idle) / delta_total}, {},
                           counter_read_begin_tick=begin, counter_read_end_tick=end,
                           interval_begin_tick=previous[2], interval_end_tick=end)
        except Exception as error:
            return _result({}, {"busy_percent": f"{type(error).__name__}: {error}"})

    def close(self) -> None:
        pass


class DeviceTelemetrySampler:
    """Absolute-deadline sampling thread; raw records are retained in memory."""

    def __init__(self, device_index: int = 0, interval_ms: float = 10.0,
                 clock: Any = None, providers: dict[str, Any] | None = None,
                 wait_policy: Any = None) -> None:
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, (int, float)) or not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("interval_ms must be finite and positive")
        self.clock = clock or NativeClock()
        self.interval_ms = float(interval_ms)
        self.period_ticks = max(1, round(self.interval_ms * self.clock.frequency_hz / 1000))
        self.providers = providers if providers is not None else {"gpu": NVMLProvider(device_index), "cpu": CPUBusyProvider(self.clock)}
        self.records: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._waiter = wait_policy
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._thread_error: str | None = None
        self._start_tick: int | None = None
        self._end_tick: int | None = None

    def _collect(self, deadline: int) -> dict[str, Any]:
        begin = self.clock.now()
        result: dict[str, Any] = {"sample_index": len(self.records), "deadline_tick": deadline,
                                  "query_begin_tick": begin}
        for name, provider in self.providers.items():
            provider_begin = self.clock.now()
            try:
                observation = provider.read()
                if not isinstance(observation, dict) or not isinstance(observation.get("values"), dict) or not isinstance(observation.get("unknown"), dict):
                    raise ValueError("provider must return values and unknown dictionaries")
            except Exception as error:
                observation = {"status": "error", "values": {}, "unknown": {"provider": f"{type(error).__name__}: {error}"}}
            result[name] = {**observation, "provider_query_begin_tick": provider_begin,
                            "provider_query_end_tick": self.clock.now()}
        end = self.clock.now()
        if end < begin:
            raise ValueError("monotonic clock moved backwards")
        result.update(query_end_tick=end, lateness_ticks=max(0, begin - deadline),
                      missed_periods=max(0, (end - deadline) // self.period_ticks),
                      query_duration_ticks=end - begin)
        return result

    def _run(self) -> None:
        try:
            deadline = self._start_tick
            while not self._stop.is_set():
                record = self._collect(deadline)
                self.records.append(record)
                deadline += (record["missed_periods"] + 1) * self.period_ticks
                delay = max(0.0, (deadline - self.clock.now()) / self.clock.frequency_hz)
                self._waiter.wait(delay)
        except Exception as error:
            self._thread_error = f"{type(error).__name__}: {error}"

    def start(self) -> "DeviceTelemetrySampler":
        if self._started:
            raise RuntimeError("sampler can only start once")
        self._started = True
        if self._waiter is None:
            self._waiter = _make_waiter(self._stop)
        self._start_tick = self.clock.now()
        self._thread = threading.Thread(target=self._run, name="device-state-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
        if not self._started:
            raise RuntimeError("sampler has not started")
        if not self._stopped:
            self._stop.set()
            self._waiter.interrupt()
            self._thread.join(timeout_seconds)
            if self._thread.is_alive():
                raise TimeoutError("telemetry provider did not return; sampler still running")
            self._end_tick = self.clock.now()
            for provider in self.providers.values():
                provider.close()
            self._waiter.close()
            self._stopped = True
        return self.records

    def __enter__(self) -> "DeviceTelemetrySampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def metadata(self) -> dict[str, Any]:
        source = Path(__file__)
        return {"schema": SCHEMA, "clock": self.clock.metadata(), "interval_ms": self.interval_ms,
                "period_ticks": self.period_ticks, "start_tick": self._start_tick, "end_tick": self._end_tick,
                "started": self._started, "stopped": self._stopped, "thread_error": self._thread_error,
                "sample_count": len(self.records), "missed_periods": sum(row["missed_periods"] for row in self.records),
                "providers": {name: provider.metadata() for name, provider in self.providers.items()},
                "wait_policy": self._waiter.metadata() if self._waiter is not None else {"actual": "not_started", "high_resolution_supported": None, "global_timer_settings_modified": False, "busy_spin": False},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "python_version": platform.python_version(),
                "process_id": os.getpid(), "captured_utc": datetime.now(timezone.utc).isoformat(),
                "state_inference": "Observations only; no interpolation, throughput scaling or causal attribution.",
                "sampler_overhead_subtracted": False}

    def write_jsonl(self, path: str | Path) -> dict[str, Any]:
        if not self._stopped:
            raise RuntimeError("stop sampler before freezing raw records")
        path = Path(path)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in self.records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_count": len(self.records)}


def _clock(clock: dict[str, Any]) -> tuple[str, int]:
    if clock.get("clock_id") not in ("windows_qpc", "posix_clock_monotonic", "synthetic_test_clock"):
        raise ValueError("unsupported shared clock_id")
    return clock["clock_id"], _integer(clock.get("frequency_hz"), "frequency_hz", 1)


def align_intervals(actions: Iterable[dict[str, Any]], records: list[dict[str, Any]],
                    clock_metadata: dict[str, Any], interval_ms: float = 10.0,
                    action_clock: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return bounded observations, not inferred microsecond action states.

    A sample's entire query interval must lie inside an action to enter its
    numerical summaries. Boundary-overlapping samples are retained separately.
    CPU busy averages enter summaries only when their complete OS interval also
    lies inside the action. NVML rolling utilization remains rolling even then.
    """
    clock_id, frequency = _clock(clock_metadata)
    if action_clock is not None and _clock(action_clock) != (clock_id, frequency):
        raise ValueError("action and telemetry clocks differ")
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, (int, float)) or not math.isfinite(interval_ms) or interval_ms <= 0:
        raise ValueError("interval_ms must be finite and positive")
    period = max(1, round(interval_ms * frequency / 1000))
    previous_end = None
    for row in records:
        begin = _integer(row.get("query_begin_tick"), "sample query_begin_tick")
        end = _integer(row.get("query_end_tick"), "sample query_end_tick")
        if end < begin or (previous_end is not None and begin < previous_end):
            raise ValueError("sample query intervals must be ordered and non-overlapping")
        previous_end = end
    aligned = []
    for index, action in enumerate(actions):
        begin = _integer(action.get("begin_tick"), "action begin_tick")
        end = _integer(action.get("end_tick"), "action end_tick")
        if end <= begin:
            raise ValueError("action end_tick must exceed begin_tick")
        contained, boundary, rejected = [], [], []
        field_values: dict[str, list[float]] = {}
        field_unknown: dict[str, int] = {}
        for sample_index, row in enumerate(records):
            sample_begin, sample_end = row["query_begin_tick"], row["query_end_tick"]
            # Inclusive point observations are admitted only at strict interior
            # instants; ordinary bounded intervals use half-open overlap.
            overlaps = (sample_begin < end and sample_end > begin) or (sample_begin == sample_end and begin < sample_begin < end)
            if not overlaps:
                continue
            if sample_begin < begin or sample_end > end:
                boundary.append(sample_index)
                continue
            contained.append(sample_index)
            for name, observation in row.items():
                if not isinstance(observation, dict) or "values" not in observation:
                    continue
                if name == "cpu":
                    cpu_begin = observation.get("interval_begin_tick")
                    cpu_end = observation.get("interval_end_tick")
                    if cpu_begin is None or cpu_end is None or cpu_begin < begin or cpu_end > end:
                        rejected.append({"sample_index": sample_index, "provider": "cpu", "reason": "CPU rolling interval not contained in action"})
                        continue
                for field, value in observation["values"].items():
                    key = f"{name}.{field}"
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"non-finite or non-numeric telemetry field {key}")
                    field_values.setdefault(key, []).append(value)
                for field in observation.get("unknown", {}):
                    key = f"{name}.{field}"
                    field_unknown[key] = field_unknown.get(key, 0) + 1
        inside = [records[item] for item in contained]
        gap_ticks = [right["query_begin_tick"] - left["query_begin_tick"] for left, right in zip(inside, inside[1:])]
        span_ticks = inside[-1]["query_end_tick"] - inside[0]["query_begin_tick"] if inside else 0
        duration = end - begin
        reasons = []
        if duration < 3 * period:
            reasons.append("action_shorter_than_three_sampling_periods")
        if len(contained) < 3:
            reasons.append("fewer_than_three_contained_observations")
        if inside and (inside[0]["query_begin_tick"] - begin > 2.5 * period or end - inside[-1]["query_end_tick"] > 2.5 * period):
            reasons.append("unobserved_action_edge_exceeds_2_5_periods")
        if gap_ticks and max(gap_ticks) > 2.5 * period:
            reasons.append("sampling_gap_exceeds_2_5_periods")
        if inside and max(row["query_end_tick"] - row["query_begin_tick"] for row in inside) > period:
            reasons.append("observer_query_exceeds_sampling_period")
        if not field_values:
            reasons.append("no_observed_fields")
        aligned.append({"schema": ALIGNMENT_SCHEMA, "action_index": action.get("action_index", index),
                        "begin_tick": begin, "end_tick": end, "duration_ms": duration * 1000 / frequency,
                        "clock": {"clock_id": clock_id, "frequency_hz": frequency},
                        "contained_sample_indices": contained, "boundary_ambiguous_sample_indices": boundary,
                        "excluded_rolling_intervals": rejected, "observation_span_fraction": span_ticks / duration,
                        "query_fraction": sum(row["query_end_tick"] - row["query_begin_tick"] for row in inside) / duration,
                        "max_sample_gap_ms": max(gap_ticks) * 1000 / frequency if gap_ticks else None,
                        "observed_fields": {key: {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)} for key, values in sorted(field_values.items())},
                        "unknown_counts": field_unknown,
                        "quality": {"model_ready": not reasons, "reasons": reasons,
                                    "model_ready_scope": "Sustained-block descriptive state association only; not causal, instantaneous, per-kernel or effective throughput evidence."},
                        "interpolation_performed": False})
    return aligned
