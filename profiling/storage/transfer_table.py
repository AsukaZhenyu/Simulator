# -*- coding: utf-8 -*-
"""Export measured SSD records as ``m2-layer-0.7`` transfer entries.

The measurement tool clocks one aligned I/O per exact byte count. M2 prices a
copy by exact byte count and refuses to interpolate, so this exporter maps
every measured record to exactly one ``(route, storage_bytes)`` key. A size
that was not measured stays missing, and the kernel will refuse it at compile
time instead of borrowing a neighbour's number.

Two things the measurement record keeps apart have to be joined or named here:

* A write's ``duration_ns`` stops when the write call returns; the flush that
  follows is reported separately. ``flush="exclude"`` (the default) exports the
  write exactly as measured and says so in ``timing_scope``.
  ``flush="include"`` adds the two medians into one number -- the sum of two
  medians, not the median of their sums -- and names the interval accordingly.
  Neither choice hides a number: durability cost is either absent from the
  duration and named, or present and named.
* ``cache_bypass`` is the mechanism that was actually achieved. A policy that
  does not bypass the page cache measures the page cache *and* the device, so
  an exported table containing such an entry is forced to
  ``validation_only: true``; M2 refuses it otherwise.

The result is a complete cost table, not a fragment: M2 requires compute
``entries`` next to ``transfer_entries``, so a base table is always supplied
and its compute entries are carried through untouched.

A consumer SSD does not have *one* write rate. It absorbs writes until its
cache and the free-block pool behind it are exhausted, then settles at the
media's own rate -- measured here at about 2.5 GB/s burst and 0.46 GB/s
sustained, a 5.4x difference that no single number can stand in for. So the
exporter can take a second campaign (``--sustained-measurements``) whose writes
were taken after the drive had already been filled, and writes its write
durations into ``sustained_duration_ns`` next to the burst ones. The threshold
that tells the kernel when to stop charging the burst price is supplied by the
operator as ``--sustained-after-bytes``: it is a property of the campaign, not
of a single record, and inventing it here would be exactly the kind of silent
default the rest of this package refuses.

Only writes get a second price. Reads measured under garbage-collection pressure
*do* slow down, but that is a different effect with a different trigger and is
not folded into the write regime.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

try:
    from . import ssd_measure
except ImportError:  # direct script execution
    import ssd_measure

M2_STORAGE_VERSION = "m2-layer-0.7"
# .5 is dense hybrid too, so promoting it to .7 adds the tier without changing
# any execution contract. .6 is MoE: its macros are expert groups, not dense
# layers, and must not be relabelled as a dense hybrid.
PROMOTABLE_BASE_VERSIONS = frozenset({"m2-layer-0.5", M2_STORAGE_VERSION})
FLUSH_POLICIES = ("exclude", "include")
ROUTE_BY_DIRECTION = {"READ_FILE": "ssd_to_dram", "WRITE_FILE": "dram_to_ssd"}
MEASURED_PROVENANCE = "measured_transfer"
# Mirrors m2.costs.CACHE_BYPASS_POLICIES. Duplicated on purpose: the
# measurement package must not import the model to state what it measured.
CACHE_BYPASS_POLICIES = frozenset({"unbuffered", "fadvise_dontneed", "f_nocache"})
READ_SCOPE = "single_aligned_io_wall"
WRITE_SCOPE_EXCLUDING_FLUSH = "single_aligned_write_wall_excluding_flush"
WRITE_SCOPE_WITH_FLUSH = "single_aligned_write_plus_flush_medians"


class TransferTableError(ValueError):
    """A measurement document or base table cannot be exported honestly."""


def _require_table(entries: list[dict], sizes: list[int] | None) -> None:
    if not sizes:
        return
    present = {(entry["route"], entry["storage_bytes"]) for entry in entries}
    missing = [(route, size) for size in sizes for route in ROUTE_BY_DIRECTION.values()
               if (route, size) not in present]
    if missing:
        rendered = ", ".join(f"{route}/{size}" for route, size in missing)
        raise TransferTableError(
            f"测量里没有这些精确尺寸：{rendered}。M2 不插值，"
            "要么把这些字节数加进 --sizes-bytes 重测，要么不要在这些尺寸上搬权重"
        )


def transfer_entries(document: Any, *, flush: str = "exclude") -> list[dict]:
    """One M2 transfer entry per measured record, keyed by exact bytes."""
    if flush not in FLUSH_POLICIES:
        raise TransferTableError(f"flush must be one of {FLUSH_POLICIES}, got {flush!r}")
    if not isinstance(document, dict):
        raise TransferTableError("measurement document must be an object")
    if document.get("schema_version") != ssd_measure.SCHEMA_VERSION:
        raise TransferTableError(
            f"measurement document must declare schema_version={ssd_measure.SCHEMA_VERSION!r}; "
            f"got {document.get('schema_version')!r}"
        )
    records = document.get("measurements")
    if not isinstance(records, list) or not records:
        raise TransferTableError("measurement document has no measurements")
    failed = [record.get("id") for record in records if record.get("status") != "measured"]
    if failed:
        raise TransferTableError(
            "measurement document contains records that were not measured: "
            + ", ".join(str(item) for item in failed)
            + "; export would look complete while carrying a hole"
        )

    entries: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        try:
            ssd_measure.validate_record(record)
        except ssd_measure.SsdMeasureError as exc:
            raise TransferTableError(f"{record.get('id')}: {exc}") from exc
        if record.get("timing_scope") != READ_SCOPE:
            raise TransferTableError(
                f"{record.get('id')}: timing_scope={record.get('timing_scope')!r} is not "
                f"{READ_SCOPE!r}; the exporter would have to invent what the clock covered"
            )
        direction = record["direction"]
        route = ROUTE_BY_DIRECTION[direction]
        size = record["size_bytes"]
        key = (route, size)
        if key in seen:
            raise TransferTableError(f"duplicate measured record for {route}/{size}")
        seen.add(key)
        duration = record["duration_ns"]
        if direction == "WRITE_FILE":
            if flush == "include":
                duration += record["flush_ns"]
                scope = WRITE_SCOPE_WITH_FLUSH
            else:
                scope = WRITE_SCOPE_EXCLUDING_FLUSH
        else:
            scope = READ_SCOPE
        entries.append({
            "route": route,
            "storage_bytes": size,
            "duration_ns": duration,
            "provenance": MEASURED_PROVENANCE,
            "timing_scope": scope,
            "cache_policy": record["cache_bypass"],
        })
    entries.sort(key=lambda entry: (entry["route"], entry["storage_bytes"]))
    return entries


def warm_cache_entries(entries: list[dict]) -> list[dict]:
    """Entries whose policy does not bypass the page cache."""
    return [entry for entry in entries if entry["cache_policy"] not in CACHE_BYPASS_POLICIES]


def measurement_label(document: dict) -> str:
    platform_block = document.get("platform") or {}
    label = platform_block.get("label") if isinstance(platform_block, dict) else None
    return str(label or "unlabelled")


def sustained_writes(document: Any, *, flush: str = "exclude") -> dict[int, int]:
    """``storage_bytes -> sustained duration`` for the writes in a campaign.

    Only the write half is used: the read records in a sustained campaign were
    taken under garbage-collection pressure and describe a different effect, so
    silently taking them would replace a good read price with a slow one and
    attribute it to the write regime.
    """
    entries = transfer_entries(document, flush=flush)
    return {entry["storage_bytes"]: entry["duration_ns"]
            for entry in entries if entry["route"] == "dram_to_ssd"}


def build_cost_table(base: Any, document: dict, *, flush: str = "exclude",
                     replace: bool = False, require_sizes: list[int] | None = None,
                     sustained_document: dict | None = None,
                     sustained_after_bytes: int | None = None,
                     ) -> dict:
    """Merge measured transfer entries into ``base`` and return a complete .7 table."""
    if not isinstance(base, dict):
        raise TransferTableError("base cost table must be an object")
    version = base.get("format_version")
    if version not in PROMOTABLE_BASE_VERSIONS:
        raise TransferTableError(
            f"base cost table must be {sorted(PROMOTABLE_BASE_VERSIONS)}; got {version!r}. "
            "It needs the dense-hybrid compute entries that .7 executes; .6 expert-group "
            "macros cannot be promoted to a dense hybrid"
        )
    compute = base.get("entries")
    if not isinstance(compute, list) or not compute:
        raise TransferTableError("base cost table must carry its compute entries")
    existing = base.get("transfer_entries", [])
    if not isinstance(existing, list):
        raise TransferTableError("base cost table transfer_entries must be a list")

    measured = transfer_entries(document, flush=flush)
    _require_table(measured, require_sizes)
    regime: dict[str, int] | None = None
    sustained: dict[int, int] = {}
    if sustained_document is not None:
        if (sustained_after_bytes is None or isinstance(sustained_after_bytes, bool)
                or not isinstance(sustained_after_bytes, int) or sustained_after_bytes < 1):
            raise TransferTableError(
                "--sustained-after-bytes must be a positive integer: the kernel needs "
                "the cumulative write volume at which the burst price stops, and no "
                "single measurement record carries it"
            )
        sustained = sustained_writes(sustained_document, flush=flush)
        write_sizes = {entry["storage_bytes"] for entry in measured
                       if entry["route"] == "dram_to_ssd"}
        missing = sorted(write_sizes - set(sustained))
        if missing:
            raise TransferTableError(
                f"the sustained campaign has no write measurement for {missing}; "
                "those sizes would keep the burst price forever with nothing saying why"
            )
        extra = sorted(set(sustained) - write_sizes)
        if extra:
            raise TransferTableError(
                f"the sustained campaign measured writes at {extra}, which the burst "
                "campaign did not; one price would have no second to sit next to"
            )
        slowed = False
        for entry in measured:
            if entry["route"] != "dram_to_ssd":
                continue
            duration = sustained[entry["storage_bytes"]]
            if duration < entry["duration_ns"]:
                raise TransferTableError(
                    f"dram_to_ssd/{entry['storage_bytes']}: sustained {duration} ns is "
                    f"faster than the burst {entry['duration_ns']} ns it is supposed to "
                    "follow; the two campaigns were probably swapped"
                )
            if duration > entry["duration_ns"]:
                slowed = True
            entry["sustained_duration_ns"] = duration
        if not slowed:
            # Equal prices everywhere mean no cliff was captured: either the
            # drive had already recovered when the second campaign ran, or the
            # same document was passed twice. Writing a threshold anyway would
            # declare a second tier that no sample supports. A single size that
            # did not slow down is fine -- small writes can stay in cache.
            raise TransferTableError(
                "the sustained campaign is not slower than the burst campaign at any "
                "measured size, so there is no second tier to declare; check that it "
                "ran while the drive was still filled (and is not the same file twice)"
            )
        regime = {"after_cumulative_bytes": sustained_after_bytes}
    elif sustained_after_bytes is not None:
        raise TransferTableError(
            "--sustained-after-bytes without --sustained-measurements would declare a "
            "threshold that changes no price"
        )
    if sustained_document is None and base.get("storage_write_regime") is not None:
        raise TransferTableError(
            "the base table already declares a write regime; exporting a burst-only "
            "campaign over it would leave the threshold in the table with no new "
            "sustained price behind it. Pass --sustained-measurements too, or start "
            "from a base table without a regime"
        )

    new_keys = {(entry["route"], entry["storage_bytes"]) for entry in measured}
    clashes = [entry for entry in existing
               if (entry.get("route"), entry.get("storage_bytes")) in new_keys]
    if clashes and not replace:
        rendered = ", ".join(f"{entry.get('route')}/{entry.get('storage_bytes')}"
                             for entry in clashes)
        raise TransferTableError(
            f"base table already prices {rendered}; pass replace=True to price them "
            "with the measurement instead of keeping two numbers for one move"
        )
    kept = [entry for entry in existing
            if (entry.get("route"), entry.get("storage_bytes")) not in new_keys]

    table = copy.deepcopy(base)
    table["format_version"] = M2_STORAGE_VERSION
    table["transfer_entries"] = sorted(
        kept + measured,
        key=lambda entry: (entry.get("route", ""), entry.get("storage_bytes", 0)),
    )
    if regime is not None:
        table["storage_write_regime"] = regime
    warm = warm_cache_entries(measured)
    # M2 requires the declaration to be explicit for *any* measured transfer, not
    # only for warm ones: silence must not read as "these numbers are sound".
    table["validation_only"] = bool(base.get("validation_only", False)) or bool(warm)
    source = base.get("source")
    table["source"] = (f"{source}+ssd-measurement:{measurement_label(document)}"
                       if isinstance(source, str) and source else
                       f"ssd-measurement:{measurement_label(document)}")
    table["comment"] = measurement_comment(document, flush=flush, warm=bool(warm),
                                           sustained_document=sustained_document,
                                           sustained_after_bytes=sustained_after_bytes)
    return table


def measurement_comment(document: dict, *, flush: str, warm: bool,
                        sustained_document: dict | None = None,
                        sustained_after_bytes: int | None = None) -> str:
    protocol = document.get("protocol") or {}
    records = [record for record in document.get("measurements", [])
               if record.get("status") == "measured"]
    return json.dumps({
        "exported_from": "ssd_measure",
        "measurement_started_at": document.get("started_at"),
        "platform": measurement_label(document),
        "storage_device": document.get("storage_device"),
        "sizes_bytes": protocol.get("sizes_bytes"),
        "samples": protocol.get("samples"),
        "warmup": protocol.get("warmup"),
        "cache_bypass_mechanisms": sorted({record["cache_bypass"] for record in records}),
        "flush_policy": flush,
        "read_timing_scope": READ_SCOPE,
        "write_timing_scope": WRITE_SCOPE_WITH_FLUSH if flush == "include"
        else WRITE_SCOPE_EXCLUDING_FLUSH,
        "validation_only_because_of_page_cache": warm,
        "interpolated": False,
        "queue_depth_modelled": False,
        "write_amplification_modelled": False,
        "write_regime_exported": sustained_document is not None,
        "sustained_measurement_started_at": (
            None if sustained_document is None else sustained_document.get("started_at")),
        "sustained_platform": (None if sustained_document is None
                               else measurement_label(sustained_document)),
        "sustained_after_cumulative_bytes": sustained_after_bytes,
        "sustained_reads_not_used": (
            "read prices come from the burst campaign only; reads slow under "
            "garbage collection by a different mechanism"),
    }, ensure_ascii=False, sort_keys=True)


def missing_sizes(entries: list[dict], sizes: list[int]) -> list[str]:
    """Which ``route/size`` pairs among ``sizes`` the entries do not price."""
    present = {(entry["route"], entry["storage_bytes"]) for entry in entries}
    return [f"{route}/{size}" for size in sizes for route in ROUTE_BY_DIRECTION.values()
            if (route, size) not in present]


def _sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated byte integers") from exc
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be unique positive integers")
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path,
                        help="m2-layer-0.5 or 0.7 cost table that holds the compute entries")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flush", choices=FLUSH_POLICIES, default="exclude")
    parser.add_argument("--replace", action="store_true",
                        help="price a move the base table already prices")
    parser.add_argument("--require-sizes", type=_sizes, default=None,
                        help="refuse to export unless both routes cover these exact sizes")
    parser.add_argument("--sustained-measurements", type=Path, default=None,
                        help="second campaign whose writes were taken after the drive "
                             "was filled; its write durations become sustained_duration_ns")
    parser.add_argument("--sustained-after-bytes", type=int, default=None,
                        help="cumulative bytes written when the sustained campaign "
                             "started; required with --sustained-measurements")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not exist; a measured table is new evidence")
    try:
        document = json.loads(args.measurements.read_text(encoding="utf-8"))
        base = json.loads(args.base.read_text(encoding="utf-8"))
        sustained = (None if args.sustained_measurements is None
                     else json.loads(args.sustained_measurements.read_text(encoding="utf-8")))
        table = build_cost_table(base, document, flush=args.flush, replace=args.replace,
                                 require_sizes=args.require_sizes,
                                 sustained_document=sustained,
                                 sustained_after_bytes=args.sustained_after_bytes)
    except (OSError, json.JSONDecodeError, TransferTableError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    ssd_measure.dump(args.output, table)
    print(json.dumps({
        "output": str(args.output),
        "transfer_entries": len(table["transfer_entries"]),
        "measured_entries": len(transfer_entries(document, flush=args.flush)),
        "sustained_write_entries": (0 if args.sustained_measurements is None
                                    else len(sustained_writes(sustained, flush=args.flush))),
        "storage_write_regime": table.get("storage_write_regime"),
        "validation_only": table.get("validation_only", False),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
