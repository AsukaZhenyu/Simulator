"""Exact-size CUDA copy and OS-buffered file-I/O measurements; no bandwidth guesses.

This runner does not call filesystem I/O physical SSD bandwidth. The file cache
is deliberately reused and not flushed. File writes include flush + os.fsync,
but that does not prove controller/drive persistence. Every sample is retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

try:
    from .profile_contract import summarize_samples
except ImportError:
    from profile_contract import summarize_samples

SCHEMA_VERSION = "moe_transfer_measurements_v1"
ROUTES = {"H2D": ("dram", "vram"), "D2H": ("vram", "dram"),
          "READ_FILE": ("ssd", "dram"), "WRITE_FILE": ("dram", "ssd")}

class TransferError(ValueError):
    pass

def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TransferError(f"{name} must be an integer >= {minimum}")
    return value

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

def _base(direction: str, size: int, backend: str, host_memory: str | None) -> dict:
    source, destination = ROUTES[direction]
    return {"id": f"{direction.lower()}_{size}_{host_memory or 'os_buffered'}",
            "status": "measured", "route": {"source": source, "destination": destination},
            "size_bytes": size, "backend": backend, "host_memory": host_memory,
            "physical_ssd_bandwidth_verified": False,
            "workspace_bytes": None, "workspace_source": None,
            "source_kind": "measured_transfer"}

def _finish(record: dict, samples: list[int]) -> dict:
    record["samples_ns"] = samples
    record["statistics"] = summarize_samples(samples)
    record["duration_ns"] = math.ceil(record["statistics"]["median_ns"])
    record["quality_flags"] = (["cv_above_0.1"] if record["statistics"]["cv"] > 0.1 else [])
    validate_record(record)
    return record

def validate_record(record: dict) -> None:
    """Validate measurable evidence, including exact byte count and declared I/O scope."""
    if record.get("status") != "measured":
        raise TransferError("record is not measured")
    size = _integer(record.get("size_bytes"), "size_bytes")
    direction = record.get("direction")
    if direction not in ROUTES:
        raise TransferError("unsupported direction")
    source, destination = ROUTES[direction]
    if record.get("route") != {"source": source, "destination": destination}:
        raise TransferError("route does not match direction")
    samples = record.get("samples_ns")
    if not isinstance(samples, list) or len(samples) < 3:
        raise TransferError("at least three samples are required")
    if record.get("sample_count") != len(samples):
        raise TransferError("sample_count disagrees with raw samples")
    _integer(record.get("warmup"), "warmup", minimum=0)
    for sample in samples:
        _integer(sample, "sample_ns")
    statistics = summarize_samples(samples)
    if record.get("statistics") != statistics:
        raise TransferError("statistics are inconsistent with raw samples")
    if record.get("duration_ns") != math.ceil(statistics["median_ns"]):
        raise TransferError("duration must be ceil(median_ns)")
    expected_flags = ["cv_above_0.1"] if statistics["cv"] > 0.1 else []
    if record.get("quality_flags") != expected_flags:
        raise TransferError("quality flags disagree with raw samples")
    if record.get("copy_validation", {}).get("passed") is not True:
        raise TransferError("copy validation did not pass")
    if record.get("physical_ssd_bandwidth_verified") is not False:
        raise TransferError("physical SSD bandwidth was not measured by this tool")
    if direction in ("H2D", "D2H"):
        if record.get("backend") != "cuda" or record.get("host_memory") not in ("pinned", "pageable"):
            raise TransferError("CUDA copy must declare pinned/pageable host memory")
        if record.get("timing_scope") != "single_transfer_wall":
            raise TransferError("CUDA wall timing scope mismatch")
    else:
        if record.get("backend") != "os-buffered-file" or record.get("host_memory") is not None:
            raise TransferError("file I/O backend mismatch")
        if record.get("timing_scope") != "os_buffered_file_io_wall":
            raise TransferError("file I/O timing scope mismatch")
        if record.get("cache_policy") != "reused_file_os_cache_not_flushed":
            raise TransferError("file cache policy must be declared")
        if not record.get("file_io", {}).get("os_buffered"):
            raise TransferError("file I/O buffering policy missing")
        if direction == "WRITE_FILE" and record["file_io"].get("fsync_in_timing") is not True:
            raise TransferError("file write must include flush + fsync")
    if not size:
        raise TransferError("empty transfer")

def transfer_cost_entry(record: dict, *, allow_unstable: bool = False,
                        allow_os_buffered_file: bool = False) -> dict:
    """Return an exact-size observation; the caller must bind timing and memory policy.

    This is not an M2 synthetic cost table and does not manufacture a physical
    MoE contract or workspace budget. OS-buffered storage needs explicit opt-in.
    """
    validate_record(record)
    if record["quality_flags"] and not allow_unstable:
        raise TransferError("unstable measurement requires explicit validation-only opt-in")
    if record["backend"] == "os-buffered-file" and not allow_os_buffered_file:
        raise TransferError("OS-buffered file I/O needs explicit acceptance; it is not physical SSD bandwidth")
    return {"source": record["route"]["source"], "destination": record["route"]["destination"],
            "size_bytes": record["size_bytes"], "duration_ns": record["duration_ns"],
            "source_kind": "measured_transfer", "validation_only": bool(record["quality_flags"]),
            "host_memory": record["host_memory"], "timing_scope": record["timing_scope"],
            "cache_policy": record["cache_policy"], "physical_ssd_bandwidth_verified": False,
            "provenance": record.get("provenance"),
            "observation_id": record["id"], "backend": record["backend"],
            "device_index": record.get("device_index"), "device_name": record.get("device_name"),
            "warmup": record["warmup"], "sample_count": record["sample_count"],
            "statistics": record["statistics"], "quality_flags": record["quality_flags"],
            "copy_validation": record["copy_validation"], "file_io": record.get("file_io"),
            "workspace_bytes": None, "workspace_source": None,
            "capacity_feasibility_verified": False, "calibrated_moe_model": False}

def lookup_transfer(records: list[dict], source: str, destination: str, size_bytes: int,
                    *, host_memory: str | None, allow_unstable: bool = False,
                    allow_os_buffered_file: bool = False) -> dict:
    _integer(size_bytes, "size_bytes")
    matches = [r for r in records if r.get("status") == "measured"
               and r.get("route") == {"source": source, "destination": destination}
               and r.get("size_bytes") == size_bytes and r.get("host_memory") == host_memory]
    if len(matches) != 1:
        raise TransferError(f"expected one exact-size/policy record; found {len(matches)}")
    return transfer_cost_entry(matches[0], allow_unstable=allow_unstable,
                               allow_os_buffered_file=allow_os_buffered_file)

def cuda_record(probe: Path, direction: str, size: int, host_memory: str,
                warmup: int, samples: int, device: int, runtime_dirs: list[Path],
                timeout_seconds: float) -> tuple[dict, dict]:
    record = _base(direction, size, "cuda", host_memory)
    record["direction"] = direction
    command = [str(probe.resolve()), "--bytes", str(size), "--direction", direction,
               "--host-memory", host_memory, "--warmup", str(warmup),
               "--samples", str(samples), "--device", str(device)]
    raw = {"command": command, "stdout": None, "stderr": None, "returncode": None}
    if not probe.is_file():
        record.update(status="unsupported", reason="native probe does not exist")
        return record, raw
    env = os.environ.copy()
    path_key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[path_key] = os.pathsep.join([str(p.resolve()) for p in runtime_dirs] + [env.get(path_key, "")])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, env=env)
        raw.update(stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw.update(error=str(exc))
        if isinstance(exc, subprocess.TimeoutExpired):
            raw.update(stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
                       stderr=exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
        record.update(status="failed", reason=str(exc))
        return record, raw
    if completed.returncode != 0:
        record.update(status="failed", reason="native probe exited unsuccessfully")
        return record, raw
    try:
        native = json.loads(completed.stdout)
        raw["native"] = native
        if (native.get("status") != "measured" or native.get("direction") != direction
                or native.get("size_bytes") != size or native.get("host_memory") != host_memory
                or native.get("warmup") != warmup or native.get("samples") != samples
                or native.get("device_index") != device or native.get("validation") != "all_bytes_exact"
                or native.get("timing_scope") != "single_transfer_wall"
                or native.get("ops_per_sample") != 1 or len(native.get("wall_ns", [])) != samples):
            raise TransferError("native output is not bound to requested copy/protocol")
        record.update(timing_scope=native["timing_scope"], cache_policy=native["cache_policy"],
                      copy_validation={"passed": True, "scope": "final_copy_all_bytes"},
                      warmup=warmup, sample_count=samples, device_name=native.get("device"),
                      device_index=device, cuda_runtime_version=native.get("cuda_runtime_version"),
                      cuda_driver_version=native.get("cuda_driver_version"),
                      synchronization=native.get("synchronization"), includes=native.get("includes"),
                      excludes=native.get("excludes"), device_samples_ns=native.get("device_ns"),
                      device_interval_note=native.get("device_interval_note"))
        _finish(record, native["wall_ns"])
    except (ValueError, TypeError, KeyError) as exc:
        record.update(status="failed", reason=str(exc))
    return record, raw

def _write_all(handle: Any, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = handle.write(view[offset:])
        if not written:
            raise TransferError("short file write")
        offset += written

def _read_all(handle: Any, data: bytearray) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = handle.readinto(view[offset:])
        if not count:
            raise TransferError("short file read")
        offset += count

def file_record(path: Path, direction: str, size: int, warmup: int, samples: int) -> tuple[dict, dict]:
    """Only touches a caller-created temporary file; repeated buffered reads are warm."""
    if direction not in ("READ_FILE", "WRITE_FILE"):
        raise TransferError("not a file I/O direction")
    _integer(size, "size_bytes")
    _integer(warmup, "warmup", minimum=0)
    _integer(samples, "samples", minimum=3)
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    destination = bytearray(size)
    with path.open("xb", buffering=0) as handle:
        _write_all(handle, payload)
        handle.flush()
        os.fsync(handle.fileno())
    timings: list[int] = []
    validation = True
    try:
        with path.open("r+b", buffering=0) as handle:
            for index in range(warmup + samples):
                started = time.perf_counter_ns()
                handle.seek(0)
                if direction == "READ_FILE":
                    _read_all(handle, destination)
                else:
                    _write_all(handle, payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                stopped = time.perf_counter_ns()
                if index >= warmup:
                    if stopped <= started:
                        raise TransferError("clock did not report a positive file-I/O interval")
                    timings.append(stopped - started)
                # Validation is explicitly outside timing.
                if direction == "READ_FILE":
                    validation = validation and destination == payload
            if direction == "WRITE_FILE":
                handle.seek(0)
                _read_all(handle, destination)
                validation = destination == payload and path.stat().st_size == size
        if not validation:
            raise TransferError("file copy validation failed")
        record = _base(direction, size, "os-buffered-file", None)
        record.update(direction=direction, timing_scope="os_buffered_file_io_wall",
                      cache_policy="reused_file_os_cache_not_flushed",
                      copy_validation={"passed": True, "scope": "each_read_all_bytes" if direction == "READ_FILE" else "final_write_all_bytes"},
                      warmup=warmup, sample_count=samples,
                      file_io={"os_buffered": True, "python_buffering": 0,
                               "fsync_in_timing": direction == "WRITE_FILE",
                               "file_path": str(path.resolve()), "temporary_file_removed_after_run": True,
                               "storage_device_identity": None, "physical_storage_type": "unknown",
                               "persistence_verified": False},
                      includes=["seek", "OS-buffered file read completion" if direction == "READ_FILE" else "OS-buffered file write", "flush+fsync" if direction == "WRITE_FILE" else "copy into preallocated bytearray"],
                      excludes=["file creation", "allocation", "validation", "open/close", "cache flushing"],
                      synchronization="synchronous read return" if direction == "READ_FILE" else "flush + os.fsync")
        _finish(record, timings)
        raw = {"command": None, "stdout": None, "stderr": None, "returncode": 0,
               "python_probe": {"samples_ns": timings, "copy_validation": record["copy_validation"],
                                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                                "file_io": record["file_io"]}}
        return record, raw
    finally:
        path.unlink(missing_ok=True)

def load_transfer_profile(directory: Path) -> dict:
    """Load a captured profile and bind each observation to immutable raw evidence.

    Source/runtime hashes describe the capture; local dependency paths need not
    exist when reading an archived profile. They are not a signature from a
    trusted publisher. No native programs or hardware are invoked by this read.
    """
    directory = directory.resolve()
    document = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise TransferError("unsupported transfer profile schema")
    records = document.get("records")
    if not isinstance(records, list):
        raise TransferError("records must be a list")
    manifest_hash = sha256_file(manifest_path)
    identifiers = set()
    for record in records:
        if not isinstance(record, dict) or record.get("id") in identifiers:
            raise TransferError("invalid or duplicate record ID")
        identifiers.add(record.get("id"))
        provenance = record.get("provenance", {})
        relative = provenance.get("raw_output")
        if not isinstance(relative, str):
            raise TransferError("raw evidence path is missing")
        raw_path = (directory / relative).resolve()
        if not raw_path.is_relative_to(directory):
            raise TransferError("raw evidence escapes profile directory")
        if provenance.get("manifest_sha256") != manifest_hash or provenance.get("raw_sha256") != sha256_file(raw_path):
            raise TransferError("raw/manifest evidence hash mismatch")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if record.get("status") != "measured":
            continue
        validate_record(record)
        protocol = manifest.get("protocol", {})
        if record["size_bytes"] not in protocol.get("sizes_bytes", []) or record["backend"] not in protocol.get("backends", []):
            raise TransferError("record was not requested by manifest")
        if record["warmup"] != protocol.get("warmup") or record["sample_count"] != protocol.get("samples"):
            raise TransferError("record protocol disagrees with manifest")
        if record["backend"] == "cuda":
            native = raw.get("native")
            if not isinstance(native, dict) or raw.get("returncode") != 0:
                raise TransferError("CUDA raw result is missing or failed")
            if json.loads(raw.get("stdout", "")) != native:
                raise TransferError("CUDA native document differs from stdout")
            if (native.get("direction") != record["direction"] or native.get("size_bytes") != record["size_bytes"]
                    or native.get("host_memory") != record["host_memory"] or native.get("warmup") != record["warmup"]
                    or native.get("samples") != record["sample_count"] or native.get("wall_ns") != record["samples_ns"]
                    or native.get("validation") != "all_bytes_exact" or native.get("device_index") != protocol.get("device")
                    or native.get("timing_scope") != record["timing_scope"] or native.get("cache_policy") != record["cache_policy"]
                    or record["host_memory"] != protocol.get("host_memory")):
                raise TransferError("CUDA observation is not bound to raw copy")
        else:
            probe = raw.get("python_probe", {})
            if (probe.get("samples_ns") != record["samples_ns"] or probe.get("copy_validation") != record["copy_validation"]
                    or probe.get("file_io") != record["file_io"] or raw.get("returncode") != 0):
                raise TransferError("file I/O observation is not bound to raw evidence")
    if document.get("all_succeeded") != all(record.get("status") == "measured" for record in records):
        raise TransferError("profile completion flag disagrees with records")
    if document.get("calibrated_moe_model") is not False:
        raise TransferError("this transfer tool does not calibrate a full MoE model")
    return document


def _sizes(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated byte integers") from exc
    if not values or len(set(values)) != len(values) or any(v < 1 or v > 1 << 30 for v in values):
        raise argparse.ArgumentTypeError("sizes must be unique integers in 1..1073741824")
    return values

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", required=True, type=_sizes)
    parser.add_argument("--backends", required=True, help="cuda,os-buffered-file (explicit subset)")
    parser.add_argument("--host-memory", choices=("pinned", "pageable"))
    parser.add_argument("--native-probe", type=Path)
    parser.add_argument("--runtime-dir", action="append", type=Path, default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--io-directory", type=Path)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    backends = args.backends.split(",")
    if not backends or len(set(backends)) != len(backends) or any(x not in ("cuda", "os-buffered-file") for x in backends):
        parser.error("--backends must explicitly list cuda and/or os-buffered-file")
    if args.warmup < 0 or args.samples < 3 or args.samples > 10000 or args.warmup > 10000 or args.device < 0:
        parser.error("invalid warmup/samples/device")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be finite and positive")
    if "cuda" in backends and (args.host_memory is None or args.native_probe is None):
        parser.error("CUDA requires --host-memory and --native-probe")
    if "os-buffered-file" in backends and (args.io_directory is None or not args.io_directory.is_dir()):
        parser.error("file I/O requires an existing --io-directory")
    if any(not p.is_dir() for p in args.runtime_dir):
        parser.error("each --runtime-dir must exist")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).parent / "profile_contract.py",
                    Path(__file__).parent / "native" / "transfer_probe.cpp",
                    Path(__file__).parent / "native" / "moe_transfer_probe.cpp",
                    Path(__file__).parent / "native" / "moe-transfer" / "CMakeLists.txt"]
    manifest = {"schema_version": SCHEMA_VERSION, "machine": {"platform": platform.platform(),
                    "python_version": sys.version, "processor": platform.processor() or None},
                "protocol": {"sizes_bytes": args.sizes, "backends": backends, "warmup": args.warmup,
                    "samples": args.samples, "device": args.device, "host_memory": args.host_memory,
                    "io_directory": str(args.io_directory.resolve()) if args.io_directory else None,
                    "interpolation": False, "physical_ssd_bandwidth_verified": False},
                "source_hashes": {str(p.resolve()): sha256_file(p) for p in source_paths if p.is_file()},
                "native_probe": {"path": str(args.native_probe.resolve()), "sha256": sha256_file(args.native_probe)} if args.native_probe and args.native_probe.is_file() else None,
                "provided_runtime_hashes": {str(p.resolve()): sha256_file(p) for d in args.runtime_dir for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in (".dll", ".so", ".dylib")},
                "actual_loaded_runtime_paths_verified": False}
    dump(args.output / "manifest.json", manifest)
    records = []
    for backend in backends:
        for size in args.sizes:
            for direction in (("H2D", "D2H") if backend == "cuda" else ("READ_FILE", "WRITE_FILE")):
                if backend == "cuda":
                    record, raw = cuda_record(args.native_probe, direction, size, args.host_memory,
                                               args.warmup, args.samples, args.device,
                                               args.runtime_dir, args.timeout_seconds)
                else:
                    with tempfile.TemporaryDirectory(prefix="moe-transfer-", dir=args.io_directory) as temporary:
                        try:
                            record, raw = file_record(Path(temporary) / "payload.bin", direction, size,
                                                       args.warmup, args.samples)
                        except (OSError, TransferError) as exc:
                            record = _base(direction, size, backend, None)
                            record.update(direction=direction, status="failed", reason=str(exc))
                            raw = {"error": str(exc), "stdout": None, "stderr": None, "returncode": None}
                raw_path = args.output / "raw" / f"{len(records):03d}.json"
                dump(raw_path, raw)
                record["provenance"] = {"raw_output": raw_path.relative_to(args.output).as_posix(),
                                        "raw_sha256": sha256_file(raw_path),
                                        "manifest_sha256": sha256_file(args.output / "manifest.json")}
                records.append(record)
                dump(args.output / "measurements.json", {"schema_version": SCHEMA_VERSION, "records": records,
                         "all_succeeded": all(r["status"] == "measured" for r in records),
                         "calibrated_moe_model": False})
    print(json.dumps({"output": str(args.output), "measured": sum(r["status"] == "measured" for r in records),
                      "failed_or_unsupported": sum(r["status"] != "measured" for r in records)}, ensure_ascii=False))
    return 0 if all(r["status"] == "measured" for r in records) else 2

if __name__ == "__main__":
    raise SystemExit(main())
