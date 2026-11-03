"""Import on-device (macOS / iOS) observations into the canonical document.

The Windows/Linux apparatus in ``ssd_measure.py`` lives in the same process as
the statistics. A phone or a Mac app cannot be: it runs on another machine, in
another language, and it must not be trusted to compute ``statistics``,
``duration_ns`` or ``quality_flags`` -- those are derived values, and a device
that reports them has already made a claim this repository cannot check.

So the device reports **observations only**: raw nanosecond samples, the
offsets it used, and the mechanism it achieved. This module turns that into the
same ``ssd_storage_measurements_v1`` document the Python apparatus emits, by
running the same ``build_record``/``validate_record`` code over the raw samples.
One measurement, one validator, whichever machine produced it.

What the device document must say (and why):

* ``platform_api`` -- ``F_NOCACHE`` or ``F_GLOBAL_NOCACHE``. It is the whole
  basis for ``cache_bypass: "f_nocache"``; a document that does not name a
  bypass API cannot claim one, exactly as on Darwin the fallback to
  ``fadvise_dontneed`` is forbidden.
* one raw sample list per direction, at least three long, plus one
  ``F_FULLFSYNC`` sample per write. A write without a flush sample is refused
  rather than filled in: the two are measured separately on purpose.
* the offsets it used, so that "random aligned placement inside the declared
  window" is checkable rather than asserted.

Nothing here invents a number. A missing field is an error, not a zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # package import
    from .ssd_measure import (
        ALIGNMENT_BYTES, SCHEMA_VERSION, build_record, summarize_measurements,
    )
except ImportError:  # direct script / flat sys.path execution
    from ssd_measure import (
        ALIGNMENT_BYTES, SCHEMA_VERSION, build_record, summarize_measurements,
    )

DEVICE_SCHEMA_VERSION = "ssd_device_observations_v1"
BYPASS_APIS = ("F_NOCACHE", "F_GLOBAL_NOCACHE")
DEVICE_TOP_KEYS = {
    "schema_version", "started_at", "finished_at", "platform_label", "platform_api",
    "system", "release", "machine", "device", "thermal_state", "battery_level",
    "directory", "protocol", "observations", "command_line",
}
OBSERVATION_KEYS = {
    "size_bytes", "backing_file_bytes", "read_samples_ns", "write_samples_ns",
    "write_flush_samples_ns", "read_offsets_bytes", "write_offsets_bytes",
}
PROTOCOL_KEYS = {
    "sizes_bytes", "samples", "warmup", "alignment_bytes", "random_offset",
}


class DeviceDocumentError(ValueError):
    """The device document cannot be turned into a checkable measurement."""


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeviceDocumentError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise DeviceDocumentError(f"{label} must be at least {minimum}")
    return value


def _integer_list(value: Any, label: str, *, minimum_count: int) -> list[int]:
    if not isinstance(value, list):
        raise DeviceDocumentError(f"{label} must be a list")
    if len(value) < minimum_count:
        raise DeviceDocumentError(f"{label} must hold at least {minimum_count} samples")
    return [_integer(item, f"{label}[{index}]", minimum=1)
            for index, item in enumerate(value)]


def _reject_unknown_keys(mapping: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise DeviceDocumentError(
            f"{label} has keys this importer does not understand: {unknown}; "
            "a field nobody reads is a field nobody checks"
        )


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise DeviceDocumentError(f"{label} must be an object")
    return value


def _check_offsets(offsets: Any, size: int, backing_bytes: int, samples: int,
                   label: str) -> list[int] | None:
    """Offsets must be aligned, inside the window, and one per sample."""
    if offsets is None:
        return None
    if not isinstance(offsets, list) or len(offsets) != samples:
        raise DeviceDocumentError(f"{label} must hold one offset per sample")
    checked = []
    for index, offset in enumerate(offsets):
        offset = _integer(offset, f"{label}[{index}]", minimum=0)
        if offset % ALIGNMENT_BYTES:
            raise DeviceDocumentError(
                f"{label}[{index}]={offset} is not a multiple of {ALIGNMENT_BYTES}; "
                "cache-bypassing I/O is aligned or it is not this measurement"
            )
        if offset + size > backing_bytes:
            raise DeviceDocumentError(
                f"{label}[{index}]={offset} plus size {size} leaves the declared "
                f"backing file of {backing_bytes} bytes"
            )
        checked.append(offset)
    return checked


def import_document(raw: dict) -> dict:
    """Turn a device observation document into the canonical measurement document."""
    raw = _require_mapping(raw, "document")
    _reject_unknown_keys(raw, DEVICE_TOP_KEYS, "document")
    if raw.get("schema_version") != DEVICE_SCHEMA_VERSION:
        raise DeviceDocumentError(
            f"schema_version must be {DEVICE_SCHEMA_VERSION!r}; "
            f"got {raw.get('schema_version')!r}"
        )
    api = raw.get("platform_api")
    if api not in BYPASS_APIS:
        raise DeviceDocumentError(
            f"platform_api must name the bypass actually used ({' / '.join(BYPASS_APIS)}); "
            f"got {api!r}. Without it the run cannot claim cache_bypass='f_nocache'."
        )
    protocol = _require_mapping(raw.get("protocol"), "protocol")
    _reject_unknown_keys(protocol, PROTOCOL_KEYS, "protocol")
    alignment = _integer(protocol.get("alignment_bytes", ALIGNMENT_BYTES),
                         "protocol.alignment_bytes", minimum=1)
    if alignment != ALIGNMENT_BYTES:
        raise DeviceDocumentError(
            f"protocol.alignment_bytes={alignment}; this protocol is {ALIGNMENT_BYTES}"
        )
    samples_expected = _integer(protocol.get("samples"), "protocol.samples", minimum=3)
    warmup = _integer(protocol.get("warmup"), "protocol.warmup", minimum=0)
    sizes = protocol.get("sizes_bytes")
    if not isinstance(sizes, list) or not sizes:
        raise DeviceDocumentError("protocol.sizes_bytes must be a nonempty list")
    sizes = [_integer(size, "protocol.sizes_bytes[]", minimum=1) for size in sizes]
    unaligned = [size for size in sizes if size % ALIGNMENT_BYTES]
    if unaligned:
        raise DeviceDocumentError(
            f"sizes {unaligned} are not multiples of {ALIGNMENT_BYTES}"
        )
    if len(set(sizes)) != len(sizes):
        raise DeviceDocumentError("protocol.sizes_bytes repeats a size")

    observations = raw.get("observations")
    if not isinstance(observations, list) or not observations:
        raise DeviceDocumentError("observations must be a nonempty list")
    seen: set[int] = set()
    records: list[dict] = []
    for index, observation in enumerate(observations):
        observation = _require_mapping(observation, f"observations[{index}]")
        _reject_unknown_keys(observation, OBSERVATION_KEYS, f"observations[{index}]")
        size = _integer(observation.get("size_bytes"), f"observations[{index}].size_bytes")
        if size not in sizes:
            raise DeviceDocumentError(
                f"observations[{index}] measures {size} bytes, which protocol.sizes_bytes "
                "does not declare; the protocol and the data must agree"
            )
        if size in seen:
            raise DeviceDocumentError(f"observations repeats size {size}")
        seen.add(size)
        backing = _integer(observation.get("backing_file_bytes"),
                           f"observations[{index}].backing_file_bytes", minimum=size)
        reads = _integer_list(observation.get("read_samples_ns"),
                              f"observations[{index}].read_samples_ns",
                              minimum_count=3)
        writes = _integer_list(observation.get("write_samples_ns"),
                               f"observations[{index}].write_samples_ns",
                               minimum_count=3)
        flush_samples = observation.get("write_flush_samples_ns")
        if not isinstance(flush_samples, list) or len(flush_samples) != len(writes):
            count = len(flush_samples) if isinstance(flush_samples, list) else "no"
            raise DeviceDocumentError(
                f"observations[{index}] has {len(writes)} write samples but {count} "
                "flush samples; one F_FULLFSYNC per write is the protocol"
            )
        flushes = _integer_list(flush_samples,
                                f"observations[{index}].write_flush_samples_ns",
                                minimum_count=len(writes))
        read_offsets = _check_offsets(observation.get("read_offsets_bytes"), size, backing,
                                      len(reads), f"observations[{index}].read_offsets_bytes")
        write_offsets = _check_offsets(observation.get("write_offsets_bytes"), size, backing,
                                       len(writes),
                                       f"observations[{index}].write_offsets_bytes")
        for direction, samples, offsets in (("READ_FILE", reads, read_offsets),
                                            ("WRITE_FILE", writes, write_offsets)):
            flush = flushes if direction == "WRITE_FILE" else None
            records.append(build_record(
                direction, size, samples,
                cache_bypass="f_nocache", warmup=warmup,
                random_offset=bool(protocol.get("random_offset", True)),
                backend="darwin_device",
                flush_samples=flush,
                sample_offsets=offsets,
            ))
    missing = sorted(set(sizes) - seen)
    if missing:
        raise DeviceDocumentError(
            f"protocol.sizes_bytes declares {missing} but observations never measured them"
        )

    label = raw.get("platform_label")
    if not isinstance(label, str) or not label:
        raise DeviceDocumentError("platform_label must name the device")
    platform = {
        "label": label,
        "system": raw.get("system"),
        "release": raw.get("release"),
        "machine": raw.get("machine"),
        "platform_api": api,
    }
    if raw.get("thermal_state") is not None and not isinstance(raw["thermal_state"], str):
        raise DeviceDocumentError("thermal_state must be a string")
    if raw.get("battery_level") is not None:
        battery = raw["battery_level"]
        if isinstance(battery, bool) or not isinstance(battery, (int, float)) \
                or not 0.0 <= float(battery) <= 1.0:
            raise DeviceDocumentError("battery_level must be a number in [0, 1]")
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "command_line": raw.get("command_line") or [],
        "platform": platform,
        "storage_device": raw.get("device"),
        "thermal_state": raw.get("thermal_state"),
        "battery_level": raw.get("battery_level"),
        "protocol": {
            "sizes_bytes": sizes,
            "samples": samples_expected,
            "warmup": warmup,
            "random_offset": bool(protocol.get("random_offset", True)),
            "cache_bypass_requested": "required",
            "alignment_bytes": ALIGNMENT_BYTES,
            "seed": None,
            "directory": raw.get("directory"),
            "one_read_and_one_write_per_round": True,
            "interpolation": False,
            "bandwidth_scalar_exported": False,
            "imported_from": DEVICE_SCHEMA_VERSION,
        },
        "all_measured": True,
        "measurements": records,
        "summary": summarize_measurements(records),
    }


def load_device_document(path: Path) -> dict:
    return import_document(json.loads(Path(path).read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path,
                        help="device observation JSON produced on the Mac or iPhone")
    parser.add_argument("--output", required=True, type=Path,
                        help="new canonical measurement JSON")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite")
    document = load_device_document(args.observations)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"imported {len(document['measurements'])} records -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
