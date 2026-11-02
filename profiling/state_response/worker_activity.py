"""Opt-in observations of caller-owned actors; never an inference predictor.

Collection is opt-in. This module opens, closes, stops or changes no process.
Borrowed handles must be supplied by the owner that launched the child.
Process CPU includes every thread, runtime spin,
and background cleanup; completed target calls do not prove process quiescence.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from profiling.resource_calibration import counters
from profiling.state_calibration.telemetry import NativeClock

SCHEMA = "owned-worker-activity-snapshot-1"
ENVIRONMENT_PREFIXES = ("OMP_", "KMP_", "VCOMP_")
KNOWN_ENVIRONMENT_KEYS = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_NESTED", "OMP_THREAD_LIMIT",
    "OMP_MAX_ACTIVE_LEVELS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY",
    "KMP_BLOCKTIME", "KMP_AFFINITY", "KMP_LIBRARY", "KMP_SETTINGS")
API_REFERENCES = (
    "https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-enumprocessmodulesex",
    "https://learn.microsoft.com/en-us/windows/win32/api/verrsrc/ns-verrsrc-vs_fixedfileinfo",
)


@dataclass(frozen=True)
class OwnedActor:
    actor_id: str
    process_id: int
    process_handle: Any  # Borrowed from the caller's existing Popen/owned child.

    def __post_init__(self):
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise ValueError("owned actor label required")
        if type(self.process_id) is not int or self.process_id <= 0:
            raise ValueError("positive owned process PID required")
        counters._handle_value(self.process_handle)

    @classmethod
    def from_popen(cls, actor_id, process):
        """Borrow an existing child; this does not open or duplicate a handle."""
        return cls(actor_id, process.pid, getattr(process, "_handle", None))


def capture_openmp_environment(environment: Mapping[str, str] | None, *, source: str) -> dict:
    """Record a supplied launch environment verbatim, without runtime defaults.

    An absent mapping is unknown, not the collector's current environment.
    This does not read a remote process environment or prove how its runtime
    interpreted variables. Pass the exact mapping supplied when launching it.
    """
    if not isinstance(source, str) or not source:
        raise ValueError("environment provenance source required")
    if environment is None:
        return dict(schema="owned-worker-openmp-environment-1", status="unknown", source=source,
            raw_values=None, known_keys=None, unknown="launch environment was not supplied",
            runtime_defaults_inferred=False, child_runtime_interpretation_verified=False)
    if not isinstance(environment, Mapping) or any(not isinstance(k, str) or not isinstance(v, str)
                                                   for k, v in environment.items()):
        raise ValueError("environment must map exact string names to exact string values")
    selected = {k: v for k, v in environment.items() if k.upper().startswith(ENVIRONMENT_PREFIXES)}
    by_upper = {}
    for name, value in selected.items():
        by_upper.setdefault(name.upper(), []).append(dict(name=name, value=value))
    known = {name: dict(status="set" if name in by_upper else "unset", occurrences=by_upper.get(name, []))
             for name in KNOWN_ENVIRONMENT_KEYS}
    return dict(schema="owned-worker-openmp-environment-1", status="recorded", source=source,
        raw_values=selected, known_keys=known, matching_prefixes=list(ENVIRONMENT_PREFIXES),
        runtime_defaults_inferred=False, child_runtime_interpretation_verified=False,
        scope="supplied launch mapping only; arbitrary additional matching keys are retained, including empty strings")


def _kernel():
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _bind(library, name, arguments, result):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, result
    return function


def _failure(operation):
    return OSError(f"{operation} failed (Windows error {getattr(ctypes, 'get_last_error', lambda: 0)()})")


def _identity(actor, *, windows, kernel=None, identity_reader=None):
    if actor.process_handle is None:
        return dict(status="unknown", observed_pid=None, reason="no borrowed owned-process handle")
    if not windows:
        return dict(status="unknown", observed_pid=None, reason="Windows process handle identity provider unavailable")
    try:
        if identity_reader is not None:
            observed = identity_reader(actor.process_handle)
        else:
            function = _bind(kernel or _kernel(), "GetProcessId", [ctypes.c_void_p], ctypes.c_uint32)
            observed = function(counters._handle_value(actor.process_handle))
        if type(observed) is not int or observed <= 0:
            raise _failure("GetProcessId")
    except Exception as error:
        return dict(status="unknown", observed_pid=None, reason=f"{type(error).__name__}: {error}")
    if observed != actor.process_id:
        raise ValueError("borrowed process handle PID differs from the declared owned actor")
    return dict(status="verified", observed_pid=observed,
                provider="GetProcessId" if identity_reader is None else "injected identity provider",
                scope="handle-to-PID binding verified; ownership itself is the caller's launch precondition")


def actor_cpu_snapshot(actor: OwnedActor, *, clock=None, windows=None, kernel=None,
                       reader=counters.read_snapshot, identity_reader=None):
    """Read cumulative CPU counters with the actual GetProcessTimes query span."""
    clock = clock or NativeClock()
    windows = os.name == "nt" if windows is None else windows
    begin = clock.now()
    identity = _identity(actor, windows=windows, kernel=kernel, identity_reader=identity_reader)
    raw = reader(actor.process_handle, clock=clock, windows=windows, kernel=kernel)
    end = clock.now()
    if end < begin:
        raise ValueError("activity observation clock moved backwards")
    return dict(schema=SCHEMA, actor_id=actor.actor_id, process_id=actor.process_id,
        process_identity=identity, clock=clock.metadata(), query_begin_tick=begin, query_end_tick=end,
        cpu_counter_snapshot=raw,
        scope="all owned-process threads; CPU includes runtime polling/spin and work after a target invocation completes",
        process_quiescence_inferred=False)


def make_counter_sampler(actors, *, interval_ms=50, clock=None, reader=counters.read_snapshot):
    """Configure, but do not start, the existing owned-process counter sampler.

    Retain these OwnedActor receipts and verify handle/PID identity with snapshots
    before interpreting sampler labels. The sampler opens/closes no handles.
    """
    actors = tuple(actors)
    if len({actor.actor_id for actor in actors}) != len(actors):
        raise ValueError("duplicate owned actor label")
    return counters.ProcessCounterSampler({actor.actor_id: actor.process_handle for actor in actors},
        interval_ms=interval_ms, clock=clock, reader=reader)


def cpu_counter_window(before, after, *, boundary_tick=None):
    """Describe one CPU increment window; never automatically declare silence.

    If its entire outer query envelope is after a worker-call completion or an
    external request boundary, a positive CPU increment proves some later owned
    process activity. A zero increment does not prove silence: OS accounting has
    finite resolution, and work can occur between sparse observations.
    """
    if before.get("schema") != SCHEMA or after.get("schema") != SCHEMA:
        raise ValueError("worker activity snapshot schema differs")
    if (before["actor_id"], before["process_id"]) != (after["actor_id"], after["process_id"]):
        raise ValueError("owned actor changed")
    left, right = before["cpu_counter_snapshot"], after["cpu_counter_snapshot"]
    activity = counters.delta(left, right)
    lp, rp = left["process"], right["process"]
    lb, le = lp.get("cpu_counter_begin_tick"), lp.get("cpu_counter_end_tick")
    rb, re = rp.get("cpu_counter_begin_tick"), rp.get("cpu_counter_end_tick")
    spans_known = all(type(value) is int for value in (lb, le, rb, re)) and lb <= le <= rb <= re
    identity_ok = all(row["process_identity"]["status"] == "verified" for row in (before, after))
    cpu = activity.get("values", {}).get("cpu_time_100ns")
    if boundary_tick is not None and (type(boundary_tick) is not int or boundary_tick < 0):
        raise ValueError("boundary must be a nonnegative tick on the same clock")
    after_boundary = None if boundary_tick is None or not spans_known else lb >= boundary_tick
    evidence = "unknown_counter_or_identity"
    if identity_ok and spans_known and cpu is not None:
        if cpu == 0:
            evidence = "no_counter_increase_observed; process_silence_not_proven"
        elif after_boundary is True:
            evidence = "positive_owned_process_CPU_increment_in_query_envelope_wholly_after_boundary"
        elif boundary_tick is not None:
            evidence = "positive_CPU_increment_but_query_envelope_not_wholly_after_boundary"
        else:
            evidence = "positive_owned_process_CPU_increment_in_query_envelope"
    return dict(schema="owned-worker-cpu-window-1", actor_id=before["actor_id"], process_id=before["process_id"],
        activity=activity, handle_pid_identity_verified=identity_ok,
        before_cpu_query_span=None if not spans_known else dict(begin_tick=lb, end_tick=le),
        after_cpu_query_span=None if not spans_known else dict(begin_tick=rb, end_tick=re),
        possible_counter_increment_outer_envelope=None if not spans_known else dict(begin_tick=lb, end_tick=re),
        possible_elapsed_ticks_min=None if not spans_known else rb - le,
        possible_elapsed_ticks_max=None if not spans_known else re - lb,
        boundary_tick=boundary_tick, entire_outer_envelope_after_boundary=after_boundary, evidence=evidence,
        process_quiescence_inferred=False, exact_CPU_event_timestamps_known=False,
        scope="CPU delta reuses existing GetProcessTimes primitive; query spans bound alignment uncertainty, not individual CPU events")


class _FixedFileInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("signature", "structure_version", "file_ms", "file_ls",
        "product_ms", "product_ls", "flags_mask", "flags", "os", "type", "subtype", "date_ms", "date_ls")]


def _file_version(path):
    library = ctypes.WinDLL("version", use_last_error=True)
    size_fn = _bind(library, "GetFileVersionInfoSizeW", [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_uint32)
    unused = ctypes.c_uint32()
    size = size_fn(str(path), ctypes.byref(unused))
    if not 0 < size <= 1 << 20:
        raise OSError("file version resource unavailable or exceeds explicit 1 MiB bound")
    buffer = ctypes.create_string_buffer(size)
    read = _bind(library, "GetFileVersionInfoW", [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p], ctypes.c_int)
    if not read(str(path), 0, size, buffer):
        raise _failure("GetFileVersionInfoW")
    query = _bind(library, "VerQueryValueW", [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int)
    pointer, length = ctypes.c_void_p(), ctypes.c_uint32()
    if not query(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)) or length.value < ctypes.sizeof(_FixedFileInfo):
        raise _failure("VerQueryValueW")
    fixed = ctypes.cast(pointer, ctypes.POINTER(_FixedFileInfo)).contents
    if fixed.signature != 0xFEEF04BD:
        raise ValueError("file version resource signature differs")
    return dict(file_version=[fixed.file_ms >> 16, fixed.file_ms & 65535, fixed.file_ls >> 16, fixed.file_ls & 65535],
                product_version=[fixed.product_ms >> 16, fixed.product_ms & 65535, fixed.product_ls >> 16, fixed.product_ls & 65535],
                source="on-disk VS_FIXEDFILEINFO; not a mapped-image memory read")


def read_file_version(path, *, windows=None, reader=None):
    """Optional disk version receipt for a caller's deduplicated file phase."""
    windows = os.name == "nt" if windows is None else windows
    if not windows and reader is None:
        return dict(status="unknown", path=str(path), value=None, reason="Windows file-version provider unavailable")
    try:
        return dict(status="observed", path=str(path), value=(reader or _file_version)(Path(path)),
                    loaded_image_memory_bytes_verified=False)
    except Exception as error:
        return dict(status="unknown", path=str(path), value=None, reason=f"{type(error).__name__}: {error}")


def _enumerate_modules(actor, *, clock, kernel, max_modules):
    enum = _bind(kernel, "K32EnumProcessModulesEx", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32], ctypes.c_int)
    filename = _bind(kernel, "K32GetModuleFileNameExW", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32], ctypes.c_uint32)
    modules = (ctypes.c_void_p * max_modules)()
    needed = ctypes.c_uint32()
    begin = clock.now()
    if not enum(counters._handle_value(actor.process_handle), modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
        raise _failure("K32EnumProcessModulesEx")
    end = clock.now()
    if needed.value > ctypes.sizeof(modules) or needed.value % ctypes.sizeof(ctypes.c_void_p):
        raise OSError("module enumeration exceeds fixed capacity or has invalid returned size")
    rows = []
    for module in modules[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buffer = ctypes.create_unicode_buffer(32768)
        left = clock.now()
        length = filename(counters._handle_value(actor.process_handle), module, buffer, len(buffer))
        right = clock.now()
        rows.append(dict(base_address=module, path=buffer.value if 0 < length < len(buffer) - 1 else None,
            path_query_begin_tick=left, path_query_end_tick=right,
            path_unknown=None if 0 < length < len(buffer) - 1 else "module filename unavailable or possibly truncated"))
    return dict(modules=rows, query_begin_tick=begin, query_end_tick=end)


def observe_loaded_module_paths(actor: OwnedActor, *, clock=None, windows=None, kernel=None,
                                identity_reader=None, module_reader=None, max_modules=4096):
    """Cheap path-only observation for a separate transaction-local disk hash.

    No DLL file bytes or version resources are read here. The caller can dedupe
    observed paths across actors for a subsequent bounded byte-verification
    phase; that disk receipt must stay distinct from this handle observation.
    """
    if type(max_modules) is not int or not 1 <= max_modules <= 16384:
        raise ValueError("bounded positive module capacity required")
    clock = clock or NativeClock()
    windows = os.name == "nt" if windows is None else windows
    native_provider = module_reader is None
    begin = clock.now()
    identity = _identity(actor, windows=windows, kernel=kernel, identity_reader=identity_reader)
    result = dict(schema="owned-worker-loaded-module-paths-1", actor_id=actor.actor_id, process_id=actor.process_id,
        clock=clock.metadata(), query_begin_tick=begin, process_identity=identity, modules=[], status="unknown",
        observed_loaded_path_proof=False, on_disk_bytes_hashed=False, file_versions_read=False,
        loaded_image_memory_bytes_verified=False, atomic_module_snapshot_claimed=False,
        observed_absence_proves_not_loaded=False,
        enumeration_provider="K32EnumProcessModulesEx+K32GetModuleFileNameExW" if native_provider else "injected fixture/provider")
    if identity["status"] != "verified":
        result.update(unknown=identity["reason"], query_end_tick=clock.now())
        return result
    try:
        if module_reader is None:
            kernel = kernel or _kernel()
            module_reader = lambda: _enumerate_modules(actor, clock=clock, kernel=kernel, max_modules=max_modules)
        before, after = module_reader(), module_reader()
        if any(len(item["modules"]) > max_modules for item in (before, after)):
            raise ValueError("module provider exceeded fixed capacity")
        signature = lambda item: [(row.get("base_address"), row.get("path")) for row in item["modules"]]
        stable = signature(before) == signature(after)
        known_paths = any(row.get("path") is not None for row in before["modules"])
        result.update(status="observed" if stable and all(row.get("path") is not None for row in before["modules"]) else "partial",
            modules=before["modules"], enumeration_before=before, enumeration_after=after,
            same_module_inventory_before_after=stable, module_paths_observed_by_provider=known_paths,
            observed_loaded_path_proof=native_provider and identity.get("provider") == "GetProcessId" and known_paths,
            scope="actual owned-handle path observation only; module races and datafile-only mappings remain limitations")
    except Exception as error:
        result["unknown"] = f"{type(error).__name__}: {error}"
    result["query_end_tick"] = clock.now()
    return result


def snapshot_loaded_modules(actor: OwnedActor, *, clock=None, windows=None, kernel=None,
                            identity_reader=None, module_reader=None, version_reader=None,
                            max_modules=4096, max_total_hash_bytes=4 << 30):
    """Observe actual owned-handle module paths; never infer them from PE/PATH.

    Disk SHA/version identify the observed path's current file, not the loaded
    memory image. Before/after enumeration equality does not prove an atomic or
    complete lifetime snapshot. Failure, races, cross-bitness limitations and
    budget limits remain explicit unknowns. No remote process is opened.
    """
    if type(max_modules) is not int or not 1 <= max_modules <= 16384:
        raise ValueError("bounded positive module capacity required")
    if type(max_total_hash_bytes) is not int or max_total_hash_bytes < 0:
        raise ValueError("explicit nonnegative file hashing byte budget required")
    native_provider = module_reader is None
    clock = clock or NativeClock()
    windows = os.name == "nt" if windows is None else windows
    begin = clock.now()
    identity = _identity(actor, windows=windows, kernel=kernel, identity_reader=identity_reader)
    result = dict(schema="owned-worker-loaded-modules-1", actor_id=actor.actor_id, process_id=actor.process_id,
        clock=clock.metadata(), query_begin_tick=begin, process_identity=identity, status="unknown", modules=[],
        observed_loaded_path_proof=False, loaded_image_memory_bytes_verified=False,
        atomic_module_snapshot_claimed=False, observed_absence_proves_not_loaded=False,
        max_modules=max_modules, max_total_hash_bytes=max_total_hash_bytes, bytes_hashed=0, api_references=list(API_REFERENCES))
    result["enumeration_provider"] = "K32EnumProcessModulesEx+K32GetModuleFileNameExW" if native_provider else "injected module provider; no native module API verified here"
    if identity["status"] != "verified":
        result.update(unknown=identity["reason"], query_end_tick=clock.now())
        return result
    try:
        if module_reader is None:
            kernel = kernel or _kernel()
            module_reader = lambda: _enumerate_modules(actor, clock=clock, kernel=kernel, max_modules=max_modules)
        before = module_reader()
        if len(before["modules"]) > max_modules:
            raise ValueError("module provider exceeded fixed capacity")
        rows, remaining = [], max_total_hash_bytes
        for item in before["modules"]:
            row = dict(item, on_disk_sha256=None, on_disk_version=None, disk_unknown={})
            path = item.get("path")
            if path is None:
                row["disk_unknown"]["path"] = item.get("path_unknown", "module path unknown")
            else:
                left = clock.now()
                try:
                    target = Path(path)
                    stat_before = target.stat()
                    if not target.is_absolute() or not target.is_file():
                        raise ValueError("module API did not return an absolute regular file path")
                    if stat_before.st_size > remaining:
                        raise OSError("explicit total on-disk hashing budget exhausted")
                    value, count = hashlib.sha256(), 0
                    with target.open("rb") as stream:
                        for block in iter(lambda: stream.read(1 << 20), b""):
                            if len(block) > remaining:
                                raise OSError("file grew beyond remaining byte budget")
                            count += len(block)
                            remaining -= len(block)
                            value.update(block)
                    row.update(on_disk_sha256=value.hexdigest(), file_size_bytes=count)
                    try:
                        row["on_disk_version"] = (version_reader or _file_version)(target)
                    except Exception as error:
                        row["disk_unknown"]["version"] = f"{type(error).__name__}: {error}"
                    stat_after = target.stat()
                    row["file_metadata_unchanged_during_read"] = (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
                    if not row["file_metadata_unchanged_during_read"]:
                        row["disk_unknown"]["consistency"] = "file metadata changed during hash/version query; loaded-byte equivalence unknown"
                except Exception as error:
                    row["disk_unknown"]["hash"] = f"{type(error).__name__}: {error}"
                row.update(disk_query_begin_tick=left, disk_query_end_tick=clock.now())
            rows.append(row)
        after = module_reader()
        signature = lambda observation: [(item.get("base_address"), item.get("path")) for item in observation["modules"]]
        stable = signature(before) == signature(after)
        result.update(status="observed" if stable and all(not row["disk_unknown"] for row in rows) else "partial",
            modules=rows, enumeration_before=before, enumeration_after=after,
            same_module_inventory_before_after=stable, bytes_hashed=max_total_hash_bytes - remaining,
            module_paths_observed_by_provider=any(row.get("path") is not None for row in rows),
            observed_loaded_path_proof=native_provider and identity.get("provider") == "GetProcessId" and any(row.get("path") is not None for row in rows),
            module_scope="only modules reported by the owned process handle API; datafile-only mappings may be absent; unload/load races remain possible",
            disk_identity_scope="current on-disk path bytes/version only; not mapped-image bytes or PE-import resolution guess")
    except Exception as error:
        result.update(unknown=f"{type(error).__name__}: {error}")
    result["query_end_tick"] = clock.now()
    return result
