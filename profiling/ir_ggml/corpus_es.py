#!/usr/bin/env python3
"""The E/S strategy corpus: 800 declared policies, packed and unpacked.

What this corpus is
-------------------
Every case is one *declared policy*: a fixed DAG of semantic operators with each
operator's device already assigned, one copy of every weight, one cache budget, and a
release rule.  The space is the power set over per-operator placement bits, so what
varies between cases is *where work runs and what is kept*, never how much work there
is -- `nodes.json` carries the same durations at the same shapes in all 800.

That is the "E/S space" the paper searches: **E**xecution placement x **S**torage
release.  The corpus is the test set for the search, not a set of measurements; nothing
here is timed, and no number in it came from a device.

Why it is a 83 KiB pack and not 800 directories
-----------------------------------------------
The 800 `nodes.json` files total 7.2 MiB and gzip to 83 KiB: they are 90% shared
structure, differing only in machine assignments.  The repository keeps the pack, the
unpacker verifies every case by name and content hash, and a clone gets the same corpus
without 1 600 tiny files in the history.

Provenance
----------
Packed from ``build/coordination-metrics-approved-20261030/exact/cases`` -- the output
of ``exact_fairness.stage1.py``, which depends on the ``experiments/ir_value`` and
``experiments/dac_validation_e1_oracle`` trees that live with the paper, not in this
repository.  ``--pack`` regenerates the pack from such a directory; the archive names
and hashes travel inside it, so a pack built elsewhere can be compared field by field.

Usage::

    python profiling/ir_ggml/corpus_es.py --list            # names + count
    python profiling/ir_ggml/corpus_es.py --unpack DIR      # write DIR/<case>/nodes.json
    python profiling/ir_ggml/corpus_es.py --pack SRC --out PACK   # rebuild the pack
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE / "corpus_es.json.gz"
SCHEMA = "ir-ggml-es-corpus/v1"


def _canonical(nodes: dict) -> str:
    """The one spelling this module writes.  Hashes are taken over *this*, not the
    archived file's bytes, so a pack can be verified after a round trip."""
    return json.dumps(nodes, indent=2)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_dir(source: Path) -> dict:
    """Read ``source/<case>/nodes.json`` into the pack form.  Order is by case name.

    The node table stays a plain object: Python's ``json`` preserves the file's key
    order, and the pack is written *without* ``sort_keys``.  The order is part of the
    case -- ``instantiate.py`` numbers nodes in the order the file lists them, so a pack
    that re-sorted the keys would emit the same graph with different indices under the
    same case name, silently changing every saved measurement.
    """
    cases = {}
    for case in sorted(p for p in source.iterdir() if (p / "nodes.json").exists()):
        nodes = json.loads((case / "nodes.json").read_text(encoding="utf-8"))
        cases[case.name] = {"sha256": _sha(_canonical(nodes)), "nodes": nodes}
    if not cases:
        raise SystemExit(f"no <case>/nodes.json below {source}")
    return {"schema": SCHEMA, "count": len(cases), "cases": cases}


def load(pack: Path = PACK) -> dict:
    with gzip.open(pack, "rt", encoding="utf-8") as handle:
        blob = json.load(handle)
    if blob.get("schema") != SCHEMA:
        raise SystemExit(f"{pack}: schema {blob.get('schema')!r} is not {SCHEMA!r}")
    if blob["count"] != len(blob["cases"]):
        raise SystemExit(f"{pack}: count {blob['count']} != {len(blob['cases'])} cases")
    return blob


def nodes_of(entry: dict) -> dict:
    """The case's node table, in its archived order."""
    return entry["nodes"]


def unpack(pack: Path, out: Path) -> int:
    blob = load(pack)
    for name, entry in blob["cases"].items():
        nodes = nodes_of(entry)
        if _sha(_canonical(nodes)) != entry["sha256"]:
            raise SystemExit(f"{name}: content does not match its recorded hash")
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "nodes.json").write_text(_canonical(nodes) + "\n", encoding="utf-8",
                                           newline="\n")
    print(f"{blob['count']} cases -> {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, metavar="SRC",
                        help="rebuild the pack from a directory of <case>/nodes.json")
    parser.add_argument("--out", type=Path, default=PACK, help="pack path (with --pack)")
    parser.add_argument("--unpack", type=Path, metavar="DIR",
                        help="write every case's nodes.json under DIR")
    parser.add_argument("--corpus", type=Path, default=PACK, help="the pack to read")
    parser.add_argument("--list", action="store_true", help="print case names and the count")
    parser.add_argument("--verify", action="store_true",
                        help="recompute every case's hash and refuse if it differs")
    args = parser.parse_args(argv)

    if args.pack:
        blob = pack_dir(args.pack)
        # No `sort_keys`: it recurses, and it would reorder every case's node table.
        text = json.dumps(blob, separators=(",", ":"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as raw:
            # mtime=0 keeps the pack byte-identical across rebuilds, so committing it
            # twice shows no diff.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                               compresslevel=9) as handle:
                handle.write(text.encode("utf-8"))
        print(f"packed {blob['count']} cases -> {args.out} ({args.out.stat().st_size} bytes)")
        return 0

    blob = load(args.corpus)
    if args.verify:
        bad = [name for name, entry in blob["cases"].items()
               if _sha(_canonical(nodes_of(entry))) != entry["sha256"]]
        if bad:
            raise SystemExit(f"{len(bad)} case(s) do not match their recorded hash: {bad[:5]}")
        print(f"{blob['count']} cases verified")
        return 0
    if args.list:
        for name in blob["cases"]:
            print(name)
        print(f"{blob['count']} cases", file=sys.stderr)
        return 0
    if args.unpack:
        return unpack(args.corpus, args.unpack)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
