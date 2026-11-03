"""Cache-bypassing, exact-size SSD read/write measurements.

One timed sample is one read or one write of the requested byte count. The
tool never interpolates between sizes, never exports a bandwidth scalar, and
retains every raw duration. Windows opens files with
``FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`` through ``CreateFileW``
and transfers into a ``VirtualAlloc`` buffer; POSIX uses ``O_DIRECT`` when
available and otherwise invalidates the page cache with
``posix_fadvise(POSIX_FADV_DONTNEED)`` before every sample. Darwin has neither,
so it uses per-handle ``fcntl(F_NOCACHE)`` and ``F_FULLFSYNC`` for the flush —
the same mechanism an on-device iOS app must use. The mechanism that was
actually achieved is recorded in every record.

A write sample stops its timer when the write syscall returns. ``fsync`` /
``FlushFileBuffers`` / ``F_FULLFSYNC`` is timed into a separate ``flush_ns``
field: it is reported alongside the write duration and is never added to it
silently.

Only alignment-compatible sizes (multiples of 4096 bytes) are measured when
cache bypass is requested. Nothing here proves controller persistence, queue
depth, or write amplification; see README.md.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    from ..profile_contract import summarize_samples
except ImportError:  # direct script / flat sys.path execution
    from profile_contract import summarize_samples

SCHEMA_VERSION = "ssd_storage_measurements_v1"
ALIGNMENT_BYTES = 4096
CACHE_BYPASS_MODES = ("required", "best-effort", "off")
CACHE_BYPASS_MECHANISMS = ("unbuffered", "fadvise_dontneed", "f_nocache", "buffered")
VERIFIED_MECHANISMS = frozenset(("unbuffered", "fadvise_dontneed", "f_nocache"))
ROUTES = {"READ_FILE": ("ssd", "dram"), "WRITE_FILE": ("dram", "ssd")}
# Darwin has neither O_DIRECT nor a working FADV_DONTNEED. ``fcntl(F_NOCACHE)``
# is the mechanism the platform actually offers, and it is the same one an
# on-device iOS app must use, so the label means the same thing on both.
F_NOCACHE = 48
F_FULLFSYNC = 51


class SsdMeasureError(ValueError):
    """A record or protocol violates the measurement contract."""


class CacheBypassError(RuntimeError):
    """The requested cache bypass could not be established."""


class AlignedAllocationError(RuntimeError):
    """The aligned transfer buffer could not be allocated; never fall back."""


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SsdMeasureError(f"{name} must be an integer >= {minimum}")
    return value


def _round_up(value: int, alignment: int = ALIGNMENT_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated byte integers") from exc
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(size < 1 or size > 1 << 30 for size in sizes)):
        raise argparse.ArgumentTypeError("sizes must be unique integers in 1..1073741824")
    return sizes


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Backends. Every backend exposes read(size, offset), write(size, offset),
# flush(), prepare_sample(), fill(size) and close(). Offsets and transfer
# lengths are 4096-byte aligned whenever cache bypass is requested.
# ---------------------------------------------------------------------------

class _BufferedFdBackend:
    """Plain os-level I/O; used for --cache-bypass off and as last resort."""

    mechanism = "buffered"
    label = "buffered-file"

    def __init__(self, path: Path, direction: str, size: int) -> None:
        if os.name == "nt":
            flags = os.O_RDONLY if direction == "READ_FILE" else os.O_WRONLY
            flags |= getattr(os, "O_BINARY", 0)
        else:
            flags = os.O_RDONLY if direction == "READ_FILE" else os.O_WRONLY
        self.fd = os.open(path, flags)
        self._payload = b"\xa5" * size if direction == "WRITE_FILE" else None

    def prepare_sample(self) -> None:
        return None

    def fill(self, size: int) -> None:
        return None

    def read(self, size: int, offset: int) -> None:
        os.lseek(self.fd, offset, os.SEEK_SET)
        data = os.read(self.fd, size)
        if len(data) != size:
            raise OSError(f"short buffered read: {len(data)} of {size} bytes")

    def write(self, size: int, offset: int) -> None:
        assert self._payload is not None
        os.lseek(self.fd, offset, os.SEEK_SET)
        written = os.write(self.fd, self._payload)
        if written != size:
            raise OSError(f"short buffered write: {written} of {size} bytes")

    def flush(self) -> None:
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


class _PosixBackend:
    """O_DIRECT, posix_fadvise_dontneed or Darwin F_NOCACHE, on an aligned mmap buffer."""

    def __init__(self, path: Path, direction: str, size: int, flags: int,
                 mechanism: str, label: str, *, nocache: bool = False) -> None:
        self.fd = os.open(path, flags)
        self.mechanism = mechanism
        self.label = label
        self._darwin = nocache
        try:
            if nocache:
                import fcntl
                # Per-handle: every read and write through this handle skips the
                # unified buffer cache, which is exactly the scope measured here.
                fcntl.fcntl(self.fd, F_NOCACHE, 1)
        except (OSError, ImportError):
            os.close(self.fd)
            raise
        try:
            self._buffer = mmap.mmap(-1, _round_up(size))
        except (OSError, ValueError) as exc:
            os.close(self.fd)
            raise AlignedAllocationError(
                "aligned mmap buffer allocation failed; refusing a buffered fallback"
            ) from exc
        self._view = memoryview(self._buffer)

    def prepare_sample(self) -> None:
        if self.mechanism == "fadvise_dontneed":
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def fill(self, size: int) -> None:
        chunk = b"\xa5" * min(size, 1 << 20)
        done = 0
        while done < size:
            amount = min(len(chunk), size - done)
            self._view[done:done + amount] = chunk[:amount]
            done += amount

    def read(self, size: int, offset: int) -> None:
        os.lseek(self.fd, offset, os.SEEK_SET)
        view = self._view[:size]
        try:
            count = os.readv(self.fd, [view])
        finally:
            view.release()
        if count != size:
            raise OSError(f"short direct read: {count} of {size} bytes")

    def write(self, size: int, offset: int) -> None:
        os.lseek(self.fd, offset, os.SEEK_SET)
        view = self._view[:size]
        try:
            count = os.writev(self.fd, [view])
        finally:
            view.release()
        if count != size:
            raise OSError(f"short direct write: {count} of {size} bytes")

    def flush(self) -> None:
        if self._darwin:
            import fcntl
            # fsync on Darwin returns once the data reaches the drive's cache,
            # not the media. F_FULLFSYNC is the barrier that means "durable".
            fcntl.fcntl(self.fd, F_FULLFSYNC)
        else:
            os.fsync(self.fd)

    def close(self) -> None:
        self._view.release()
        self._buffer.close()
        os.close(self.fd)


class _WindowsUnbufferedBackend:
    """CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH).

    The handle is wrapped with msvcrt.open_osfhandle so lifetime and close
    semantics match the POSIX backends. ctypes ReadFile/WriteFile are used
    because Python's own os.read buffer is not guaranteed 4096-byte aligned.
    """

    mechanism = "unbuffered"
    label = "windows-unbuffered"

    def __init__(self, path: Path, direction: str, size: int) -> None:
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                              ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
        kernel32.SetFilePointerEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                       ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        kernel32.WriteFile.restype = wintypes.BOOL
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t,
                                          wintypes.DWORD, wintypes.DWORD]
        kernel32.VirtualAlloc.restype = wintypes.LPVOID
        kernel32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
        kernel32.VirtualFree.restype = wintypes.BOOL

        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._msvcrt = msvcrt
        self._DWORD = wintypes.DWORD
        access = 0x80000000 if direction == "READ_FILE" else 0x40000000
        share = 0x00000001 | 0x00000002
        attributes = 0x00000080 | 0x20000000 | 0x80000000  # NORMAL | NO_BUFFERING | WRITE_THROUGH
        handle = kernel32.CreateFileW(str(path), access, share, None, 3, attributes, None)
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, 0, invalid):
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = handle
        open_flags = (os.O_RDONLY if direction == "READ_FILE" else os.O_WRONLY) | os.O_BINARY
        try:
            self.fd = msvcrt.open_osfhandle(handle, open_flags)
        except OSError:
            kernel32.CloseHandle(handle)
            raise
        if self.fd < 0:
            kernel32.CloseHandle(handle)
            raise OSError("msvcrt.open_osfhandle failed")
        self._os_handle = msvcrt.get_osfhandle(self.fd)
        buffer_bytes = _round_up(size)
        address = kernel32.VirtualAlloc(None, buffer_bytes, 0x3000, 0x04)
        if not address:
            error = ctypes.get_last_error()
            os.close(self.fd)
            raise AlignedAllocationError(
                f"VirtualAlloc({buffer_bytes}) failed with Win32 error {error}; "
                "refusing a buffered fallback"
            )
        self._address = address
        self._size = size

    def prepare_sample(self) -> None:
        return None

    def fill(self, size: int) -> None:
        self._ctypes.memset(self._address, 0xA5, size)

    def _seek(self, offset: int) -> None:
        if not self._kernel32.SetFilePointerEx(self._os_handle, offset, None, 0):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def _transfer(self, size: int, writing: bool) -> None:
        done = 0
        while done < size:
            chunk = min(size - done, 1 << 28)
            count = self._DWORD(0)
            target = self._ctypes.c_void_p(self._address + done)
            if writing:
                ok = self._kernel32.WriteFile(self._os_handle, target, chunk,
                                              self._ctypes.byref(count), None)
            else:
                ok = self._kernel32.ReadFile(self._os_handle, target, chunk,
                                             self._ctypes.byref(count), None)
            if not ok:
                raise self._ctypes.WinError(self._ctypes.get_last_error())
            if count.value == 0:
                raise OSError("zero-byte Windows unbuffered transfer")
            done += count.value

    def read(self, size: int, offset: int) -> None:
        self._seek(offset)
        self._transfer(size, writing=False)

    def write(self, size: int, offset: int) -> None:
        self._seek(offset)
        self._transfer(size, writing=True)

    def flush(self) -> None:
        if not self._kernel32.FlushFileBuffers(self._os_handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def close(self) -> None:
        try:
            self._kernel32.VirtualFree(self._address, 0, 0x8000)
        finally:
            os.close(self.fd)


def _buffered_pair(path: Path, size: int) -> tuple[Any, Any, str, str]:
    reader = _BufferedFdBackend(path, "READ_FILE", size)
    try:
        writer = _BufferedFdBackend(path, "WRITE_FILE", size)
    except BaseException:
        reader.close()
        raise
    return reader, writer, reader.mechanism, reader.label


def _posix_pair(path: Path, size: int, *, direct: bool,
                nocache: bool = False) -> tuple[Any, Any, str, str]:
    if direct:
        mechanism, label = "unbuffered", "posix-o_direct"
        read_flags = os.O_RDONLY | os.O_DIRECT
        write_flags = os.O_WRONLY | os.O_DIRECT
    elif nocache:
        mechanism, label = "f_nocache", "darwin-f_nocache"
        read_flags, write_flags = os.O_RDONLY, os.O_WRONLY
    else:
        mechanism, label = "fadvise_dontneed", "posix-fadvise_dontneed"
        read_flags, write_flags = os.O_RDONLY, os.O_WRONLY
    reader = _PosixBackend(path, "READ_FILE", size, read_flags, mechanism, label,
                           nocache=nocache)
    try:
        writer = _PosixBackend(path, "WRITE_FILE", size, write_flags, mechanism, label,
                               nocache=nocache)
    except BaseException:
        reader.close()
        raise
    return reader, writer, mechanism, label


def _windows_unbuffered_pair(path: Path, size: int) -> tuple[Any, Any, str, str]:
    reader = _WindowsUnbufferedBackend(path, "READ_FILE", size)
    try:
        writer = _WindowsUnbufferedBackend(path, "WRITE_FILE", size)
    except BaseException:
        reader.close()
        raise
    return reader, writer, reader.mechanism, reader.label


def _darwin_nocache_available() -> bool:
    """``fcntl(F_NOCACHE)`` exists only on Darwin and needs readv/writev here."""
    if sys.platform != "darwin":
        return False
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return False
    return hasattr(os, "readv") and hasattr(os, "writev")


def open_backends(path: Path, size: int, cache_mode: str) -> tuple[Any, Any, str, str]:
    """Open read/write backends and report the mechanism actually achieved."""
    if cache_mode not in CACHE_BYPASS_MODES:
        raise SsdMeasureError(f"unknown cache bypass mode {cache_mode!r}")
    if cache_mode == "off":
        return _buffered_pair(path, size)
    if _darwin_nocache_available():
        # Darwin has no O_DIRECT, and FADV_DONTNEED is not honoured there, so
        # this is the only mechanism on the platform that actually skips the
        # unified buffer cache. Reporting "fadvise_dontneed" here would claim a
        # bypass that never happened.
        try:
            return _posix_pair(path, size, direct=False, nocache=True)
        except AlignedAllocationError:
            raise
        except OSError as exc:
            if cache_mode == "required":
                raise CacheBypassError(f"F_NOCACHE unavailable: {exc}") from exc
            return _buffered_pair(path, size)
    if os.name == "nt":
        try:
            return _windows_unbuffered_pair(path, size)
        except AlignedAllocationError:
            raise
        except OSError as exc:
            if cache_mode == "required":
                raise CacheBypassError(f"unbuffered Windows I/O unavailable: {exc}") from exc
            return _buffered_pair(path, size)
    if hasattr(os, "O_DIRECT") and hasattr(os, "readv"):
        try:
            return _posix_pair(path, size, direct=True)
        except AlignedAllocationError:
            raise
        except OSError:
            pass
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") \
            and hasattr(os, "readv") and hasattr(os, "writev"):
        return _posix_pair(path, size, direct=False)
    if cache_mode == "required":
        raise CacheBypassError("neither O_DIRECT nor posix_fadvise(POSIX_FADV_DONTNEED) is available")
    return _buffered_pair(path, size)


# ---------------------------------------------------------------------------
# Records and validation.
# ---------------------------------------------------------------------------

def build_record(direction: str, size: int, samples: list[int], *, cache_bypass: str,
                 warmup: int, random_offset: bool, backend: str,
                 flush_samples: list[int] | None = None,
                 sample_offsets: list[int] | None = None) -> dict:
    """Build and validate one exact-size direction record."""
    if direction not in ROUTES:
        raise SsdMeasureError("unsupported direction")
    source, destination = ROUTES[direction]
    statistics = summarize_samples(samples)
    record = {
        "id": f"{direction.lower()}_ssd_{size}",
        "status": "measured",
        "direction": direction,
        "route": {"source": source, "destination": destination},
        "size_bytes": size,
        "backend": backend,
        "cache_bypass": cache_bypass,
        "physical_ssd_bandwidth_verified": cache_bypass in VERIFIED_MECHANISMS,
        "alignment_bytes": ALIGNMENT_BYTES,
        "timing_scope": "single_aligned_io_wall",
        "warmup": warmup,
        "sample_count": len(samples),
        "samples_ns": list(samples),
        "statistics": statistics,
        "duration_ns": math.ceil(statistics["median_ns"]),
        "quality_flags": ["cv_above_0.1"] if statistics["cv"] > 0.1 else [],
        "random_offset": bool(random_offset),
        "sample_offsets_bytes": list(sample_offsets) if sample_offsets is not None else None,
        "flush_samples_ns": list(flush_samples) if flush_samples is not None else None,
        "flush_ns": (math.ceil(summarize_samples(flush_samples)["median_ns"])
                     if flush_samples is not None else None),
        "includes": (["reused open handle", "logical seek to a random aligned offset",
                      "one aligned read", "read syscall return"] if direction == "READ_FILE" else
                     ["reused open handle", "logical seek to a random aligned offset",
                      "one aligned write", "write syscall return"]),
        "excludes": (["file creation", "prefill", "buffer allocation",
                      "offset selection", "cache-bypass probe"] if direction == "READ_FILE" else
                     ["file creation", "prefill", "buffer allocation", "offset selection",
                      "cache-bypass probe",
                      "flush/fsync; reported separately as flush_ns and never added"]),
    }
    validate_record(record)
    return record


def validate_record(record: dict) -> None:
    """Enforce exactness and the declared cache-bypass mechanism.

    ``physical_ssd_bandwidth_verified`` is true exactly when the achieved
    mechanism bypasses the OS page cache (``unbuffered``, ``fadvise_dontneed``
    or ``f_nocache``). This is a declaration of mechanism, not an independent
    measurement of the drive controller.
    """
    if not isinstance(record, dict):
        raise SsdMeasureError("record must be an object")
    if record.get("status") != "measured":
        raise SsdMeasureError("record is not measured")
    size = _integer(record.get("size_bytes"), "size_bytes")
    direction = record.get("direction")
    if direction not in ROUTES:
        raise SsdMeasureError("unsupported direction")
    source, destination = ROUTES[direction]
    if record.get("route") != {"source": source, "destination": destination}:
        raise SsdMeasureError("route does not match direction")
    samples = record.get("samples_ns")
    if not isinstance(samples, list) or len(samples) < 3:
        raise SsdMeasureError("at least three samples are required")
    if record.get("sample_count") != len(samples):
        raise SsdMeasureError("sample_count disagrees with raw samples")
    _integer(record.get("warmup"), "warmup", minimum=0)
    for sample in samples:
        _integer(sample, "sample_ns")
    statistics = summarize_samples(samples)
    if record.get("statistics") != statistics:
        raise SsdMeasureError("statistics are inconsistent with raw samples")
    if record.get("duration_ns") != math.ceil(statistics["median_ns"]):
        raise SsdMeasureError("duration must be ceil(median_ns)")
    expected_flags = ["cv_above_0.1"] if statistics["cv"] > 0.1 else []
    if record.get("quality_flags") != expected_flags:
        raise SsdMeasureError("quality flags disagree with raw samples")
    mechanism = record.get("cache_bypass")
    if mechanism not in CACHE_BYPASS_MECHANISMS:
        raise SsdMeasureError("cache_bypass must name the achieved mechanism")
    verified = record.get("physical_ssd_bandwidth_verified")
    if not isinstance(verified, bool) or verified != (mechanism in VERIFIED_MECHANISMS):
        raise SsdMeasureError(
            "physical_ssd_bandwidth_verified must be true iff cache_bypass is "
            "unbuffered, fadvise_dontneed or f_nocache"
        )
    flush_samples = record.get("flush_samples_ns")
    if direction == "READ_FILE":
        if flush_samples is not None or record.get("flush_ns") is not None:
            raise SsdMeasureError("read records must not report a flush")
    else:
        if not isinstance(flush_samples, list) or len(flush_samples) != len(samples):
            raise SsdMeasureError("write records must report one flush per sample")
        for sample in flush_samples:
            _integer(sample, "flush_ns")
        if record.get("flush_ns") != math.ceil(summarize_samples(flush_samples)["median_ns"]):
            raise SsdMeasureError("flush_ns must be ceil(median(flush_samples_ns))")
    if not size:
        raise SsdMeasureError("empty I/O")


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------

def _file_bytes(size: int) -> int:
    # A wider offset window makes cross-sample sequential prefetch less useful.
    # The extra region is capped so very large sizes do not double disk use.
    extra = min(max(size, 4 * ALIGNMENT_BYTES), 64 << 20)
    return _round_up(size) + extra


def _prepare_backing_file(path: Path, file_bytes: int) -> None:
    chunk = b"\xa5" * min(file_bytes, 1 << 20)
    with path.open("wb") as handle:
        remaining = file_bytes
        while remaining > 0:
            amount = min(len(chunk), remaining)
            handle.write(chunk[:amount])
            remaining -= amount
        handle.flush()
        os.fsync(handle.fileno())


def _choose_offset(rng: random.Random, size: int, file_bytes: int,
                   random_offset: bool) -> int:
    if not random_offset:
        return 0
    blocks = (file_bytes - size) // ALIGNMENT_BYTES
    return rng.randrange(blocks + 1) * ALIGNMENT_BYTES


def measure_size(directory: Path, size: int, *, samples: int, warmup: int,
                 cache_mode: str, random_offset: bool,
                 rng: random.Random) -> list[dict]:
    """Measure both directions for one exact size, alternating each round."""
    if cache_mode != "off" and size % ALIGNMENT_BYTES:
        raise SsdMeasureError(
            f"size {size} is not a multiple of {ALIGNMENT_BYTES}; cache-bypassing "
            "I/O requires aligned transfer lengths"
        )
    path = directory / f"ssd_measure_{os.getpid()}_{size}.bin"
    if path.exists():
        path.unlink()
    file_bytes = _file_bytes(size)
    read_samples: list[int] = []
    write_samples: list[int] = []
    flush_samples: list[int] = []
    read_offsets: list[int] = []
    write_offsets: list[int] = []
    try:
        _prepare_backing_file(path, file_bytes)
        reader, writer, mechanism, backend = open_backends(path, size, cache_mode)
        try:
            writer.fill(size)
            for index in range(warmup + samples):
                timed = index >= warmup
                read_offset = _choose_offset(rng, size, file_bytes, random_offset)
                write_offset = _choose_offset(rng, size, file_bytes, random_offset)
                reader.prepare_sample()
                read_started = time.perf_counter_ns()
                reader.read(size, read_offset)
                read_stopped = time.perf_counter_ns()
                if read_stopped <= read_started:
                    raise SsdMeasureError("clock did not report a positive read interval")
                writer.prepare_sample()
                write_started = time.perf_counter_ns()
                writer.write(size, write_offset)
                write_stopped = time.perf_counter_ns()
                if write_stopped <= write_started:
                    raise SsdMeasureError("clock did not report a positive write interval")
                # The write timer stopped above. Flush is an additional,
                # separately reported cost; it is never added to duration_ns.
                flush_started = time.perf_counter_ns()
                writer.flush()
                flush_stopped = time.perf_counter_ns()
                if flush_stopped <= flush_started:
                    raise SsdMeasureError("clock did not report a positive flush interval")
                if timed:
                    read_samples.append(read_stopped - read_started)
                    write_samples.append(write_stopped - write_started)
                    flush_samples.append(flush_stopped - flush_started)
                    read_offsets.append(read_offset)
                    write_offsets.append(write_offset)
        finally:
            reader.close()
            writer.close()
        return [
            build_record("READ_FILE", size, read_samples, cache_bypass=mechanism,
                         warmup=warmup, random_offset=random_offset, backend=backend,
                         sample_offsets=read_offsets),
            build_record("WRITE_FILE", size, write_samples, cache_bypass=mechanism,
                         warmup=warmup, random_offset=random_offset, backend=backend,
                         flush_samples=flush_samples, sample_offsets=write_offsets),
        ]
    finally:
        path.unlink(missing_ok=True)


def _least_squares_fit(points: list[tuple[int, int]]) -> dict:
    """Descriptive fit of duration_ns = latency_ns + ceil(size / bytes_per_ns)."""
    base = {
        "model": "duration_ns = latency_ns + ceil(size_bytes / bytes_per_ns)",
        "is_descriptive": True,
        "n_points": len(points),
        "latency_ns": None,
        "bytes_per_ns": None,
        "slope_ns_per_byte": None,
        "r_squared": None,
        "relative_residuals": {str(size): None for size, _ in points},
    }
    if len(points) < 2:
        base["note"] = "fewer than two sizes; a linear fit is undefined"
        return base
    count = len(points)
    sum_x = sum(size for size, _ in points)
    sum_y = sum(duration for _, duration in points)
    sum_xx = sum(size * size for size, _ in points)
    sum_xy = sum(size * duration for size, duration in points)
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0:
        base["note"] = "all sizes are identical; a linear fit is undefined"
        return base
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        base["note"] = ("non-positive or non-finite slope; no bytes_per_ns and no "
                        "residuals are reported")
        return base
    bytes_per_ns = 1.0 / slope
    if not math.isfinite(bytes_per_ns) or bytes_per_ns <= 0:
        base["note"] = "non-finite bytes_per_ns; no residuals are reported"
        return base
    residuals: dict[str, float | None] = {}
    for size, duration in points:
        predicted = intercept + math.ceil(size / bytes_per_ns)
        residuals[str(size)] = _finite_or_none((duration - predicted) / duration)
    mean_y = sum_y / count
    total = sum((duration - mean_y) ** 2 for _, duration in points)
    residual_sum = sum((duration - (intercept + slope * size)) ** 2
                       for size, duration in points)
    r_squared = None if total == 0 else _finite_or_none(1.0 - residual_sum / total)
    base.update(latency_ns=_finite_or_none(intercept),
                bytes_per_ns=_finite_or_none(bytes_per_ns),
                slope_ns_per_byte=_finite_or_none(slope),
                r_squared=r_squared,
                relative_residuals=residuals)
    base["note"] = ("least-squares fit over the reported medians; residuals are "
                    "(measured - fitted) / measured")
    return base


def summarize_measurements(records: list[dict]) -> dict:
    directions: dict[str, dict] = {}
    for direction in ROUTES:
        measured = sorted(
            (record for record in records
             if record.get("status") == "measured" and record.get("direction") == direction),
            key=lambda record: record["size_bytes"],
        )
        sizes = []
        for record in measured:
            statistics = record["statistics"]
            sizes.append({
                "size_bytes": record["size_bytes"],
                "sample_count": record["sample_count"],
                "cache_bypass": record["cache_bypass"],
                "duration_ns": record["duration_ns"],
                "median_ns": statistics["median_ns"],
                "p10_ns": statistics["p10_ns"],
                "p90_ns": statistics["p90_ns"],
                "cv": statistics["cv"],
                "statistics": statistics,
            })
        source, destination = ROUTES[direction]
        directions[direction] = {
            "route": {"source": source, "destination": destination},
            "sizes": sizes,
            "fit": _least_squares_fit(
                [(record["size_bytes"], record["duration_ns"]) for record in measured]),
        }
    mechanisms = sorted({record["cache_bypass"] for record in records
                         if record.get("status") == "measured"})
    return {
        "fit_is_descriptive_not_used_for_lookup": True,
        "cache_bypass_mechanisms_observed": mechanisms,
        "directions": directions,
        "measurement_counts": {
            "measured": sum(record.get("status") == "measured" for record in records),
            "failed": sum(record.get("status") != "measured" for record in records),
        },
    }


def platform_block(label: str) -> dict:
    return {
        "label": label,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _windows_storage_device() -> Any:
    script = ("Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size "
              "| ConvertTo-Json -Compress")
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    value = json.loads(completed.stdout)
    if isinstance(value, dict):
        value = [value]
    return value if value else None


def _linux_storage_device() -> Any:
    devices = []
    for model_path in sorted(Path("/sys/block").glob("*/device/model")):
        name = model_path.parent.parent.name
        model = model_path.read_text(encoding="utf-8", errors="replace").strip() or None
        size_path = Path("/sys/block") / name / "size"
        sectors = None
        try:
            sectors = int(size_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
        devices.append({"device": name, "model": model, "size_sectors": sectors,
                        "size_bytes": sectors * 512 if sectors is not None else None})
    return devices if devices else None


def _macos_storage_device() -> Any:
    import plistlib

    completed = subprocess.run(["diskutil", "info", "-plist", "/"],
                               capture_output=True, timeout=20, check=False)
    if completed.returncode != 0 or not completed.stdout:
        return None
    value = plistlib.loads(completed.stdout)
    keys = ("DeviceIdentifier", "MediaName", "SolidState", "BusProtocol", "TotalSize")
    result = {key: value.get(key) for key in keys}
    return result if any(item is not None for item in result.values()) else None


def storage_device_block(probe: bool = True) -> Any:
    """Cheaply obtainable device identity; None on any failure (never invented)."""
    if not probe:
        return None
    try:
        if os.name == "nt":
            return _windows_storage_device()
        if sys.platform == "darwin":
            return _macos_storage_device()
        return _linux_storage_device()
    except Exception:
        return None


def measure(sizes: list[int], directory: Path, *, samples: int = 15, warmup: int = 3,
            cache_bypass: str = "off", random_offset: bool = True, seed: int | None = None,
            platform_label: str = "unspecified", device_probe: bool = True,
            command_line: list[str] | None = None) -> dict:
    """Run the protocol and return the complete JSON-serializable document."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise SsdMeasureError("--directory must be an existing directory")
    if not sizes or len(set(sizes)) != len(sizes):
        raise SsdMeasureError("sizes must be a nonempty list of unique integers")
    for size in sizes:
        _integer(size, "size_bytes")
    _integer(samples, "samples", minimum=3)
    _integer(warmup, "warmup", minimum=0)
    if cache_bypass not in CACHE_BYPASS_MODES:
        raise SsdMeasureError(f"unknown cache bypass mode {cache_bypass!r}")
    if cache_bypass != "off":
        unaligned = [size for size in sizes if size % ALIGNMENT_BYTES]
        if unaligned:
            raise SsdMeasureError(
                f"sizes {unaligned} are not multiples of {ALIGNMENT_BYTES}; "
                "cache-bypassing I/O requires aligned transfer lengths"
            )
    started_at = datetime.now(timezone.utc).isoformat()
    rng = random.Random(seed)
    records: list[dict] = []
    for size in sizes:
        try:
            records.extend(measure_size(directory, size, samples=samples, warmup=warmup,
                                        cache_mode=cache_bypass,
                                        random_offset=random_offset, rng=rng))
        except AlignedAllocationError:
            raise
        except (OSError, SsdMeasureError) as exc:
            for direction in ROUTES:
                source, destination = ROUTES[direction]
                records.append({
                    "id": f"{direction.lower()}_ssd_{size}",
                    "status": "failed",
                    "direction": direction,
                    "route": {"source": source, "destination": destination},
                    "size_bytes": size,
                    "reason": str(exc),
                })
    finished_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "command_line": command_line or [],
        "platform": platform_block(platform_label),
        "storage_device": storage_device_block(device_probe),
        "protocol": {
            "sizes_bytes": list(sizes),
            "samples": samples,
            "warmup": warmup,
            "random_offset": bool(random_offset),
            "cache_bypass_requested": cache_bypass,
            "alignment_bytes": ALIGNMENT_BYTES,
            "seed": seed,
            "directory": str(directory),
            "one_read_and_one_write_per_round": True,
            "interpolation": False,
            "bandwidth_scalar_exported": False,
        },
        "all_measured": all(record.get("status") == "measured" for record in records),
        "measurements": records,
        "summary": summarize_measurements(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-bytes", required=True, type=_sizes,
                        help="comma-separated exact sizes in bytes")
    parser.add_argument("--directory", required=True, type=Path,
                        help="existing directory for the temporary backing file")
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON file to write")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--random-offset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-bypass", choices=CACHE_BYPASS_MODES, default="required")
    parser.add_argument("--platform", default=platform.node() or "unknown")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device-probe", choices=("auto", "off"), default="auto")
    args = parser.parse_args(argv)
    if args.samples < 3 or args.samples > 10000:
        parser.error("--samples must be in 3..10000")
    if args.warmup < 0 or args.warmup > 10000:
        parser.error("--warmup must be in 0..10000")
    if not args.directory.is_dir():
        parser.error("--directory must exist")
    if args.output.exists():
        parser.error("--output must not exist; measurements are new evidence")
    if args.cache_bypass != "off":
        unaligned = [size for size in args.sizes_bytes if size % ALIGNMENT_BYTES]
        if unaligned:
            parser.error(f"cache-bypassing measurement requires sizes divisible by "
                         f"{ALIGNMENT_BYTES}; got {unaligned}")
    command_line = [sys.executable, str(Path(__file__).resolve()), *list(
        argv if argv is not None else sys.argv[1:])]
    try:
        document = measure(args.sizes_bytes, args.directory, samples=args.samples,
                           warmup=args.warmup, cache_bypass=args.cache_bypass,
                           random_offset=args.random_offset, seed=args.seed,
                           platform_label=args.platform,
                           device_probe=args.device_probe == "auto",
                           command_line=command_line)
    except (AlignedAllocationError, CacheBypassError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    dump(args.output, document)
    measured = document["summary"]["measurement_counts"]["measured"]
    failed = document["summary"]["measurement_counts"]["failed"]
    print(json.dumps({"output": str(args.output), "measured": measured,
                      "failed_or_unsupported": failed,
                      "cache_bypass": document["summary"]["cache_bypass_mechanisms_observed"]},
                     ensure_ascii=False))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
