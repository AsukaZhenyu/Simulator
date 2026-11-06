#!/usr/bin/env python3
"""Write ``prices/PROVENANCE.json``: what each frozen export is and where it came from.

The tables themselves travel with the repository because they are the *input* the
search priced against -- without them a clone can instantiate the corpus but cannot say
whether the prices still describe this machine.  What must travel with them is the fact
that they were measured on one specific laptop, not on the reader's.

Usage::

    python profiling/ir_ggml/prices_provenance.py           # rewrite PROVENANCE.json
    python profiling/ir_ggml/prices_provenance.py --check   # refuse a stale record
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICES = HERE / "prices"
RECORD = PRICES / "PROVENANCE.json"

# Where each export was produced, relative to the paper's archive root.  Kept here
# rather than inferred, because the archive is not part of the repository.
ORIGIN = {
    "qwen3-0.6b": "profiling/results/20261104-multimodel-dense-v2/qwen3-0.6b-pinned-export",
    "qwen3-1.7b": "profiling/results/20261104-multimodel-dense-v2/qwen3-1.7b-pinned-export",
    "qwen3-4b": "profiling/results/20261104-multimodel-dense-v2/qwen3-4b-pinned-export",
    "granite-3.0-3b-a800m": "profiling/results/20261104-multimodel-moe-v1/granite-3.0-3b-a800m-pinned-export",
    "qwen1.5-moe-a2.7b": "profiling/results/20261104-multimodel-moe-v1/qwen1.5-moe-a2.7b-pinned-export",
    "olmoe-1b-7b": "profiling/results/20261104-multimodel-moe-olmo/olmoe-1b-7b-pinned-r2-export",
}

NOTE = (
    "Frozen per-action prices, measured on the campaign machine (RTX 4070 Laptop, two "
    "pinned CPU threads, cpu_mask 2,4). They are inputs to the IR->ggml comparison, not "
    "outputs of this repository: a fresh clone on other hardware should read A1 as the "
    "device-to-device ratio against these, never as 1.0."
)


def describe(model: str) -> dict:
    path = PRICES / model / "costs.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = blob.get("entries", [])
    devices = sorted({entry["device"] for entry in entries if "device" in entry})
    return {"path": f"{model}/costs.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "export_source": blob.get("source"),
            "entries": len(entries),
            "devices": devices,
            "origin": ORIGIN.get(model)}


def build() -> dict:
    return {"schema": "ir-ggml-price-provenance/v1", "note": NOTE,
            "tables": {model: describe(model) for model in sorted(ORIGIN)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="compare against the recorded hashes and exit non-zero on drift")
    args = parser.parse_args(argv)

    current = build()
    if args.check:
        if not RECORD.exists():
            print(f"{RECORD} is missing; run prices_provenance.py", file=sys.stderr)
            return 1
        recorded = json.loads(RECORD.read_text(encoding="utf-8"))["tables"]
        drift = [model for model, entry in current["tables"].items()
                 if recorded.get(model, {}).get("sha256") != entry["sha256"]]
        if drift:
            print(f"price table(s) changed: {drift}", file=sys.stderr)
            return 1
        print(f"{len(current['tables'])} price tables match their record")
        return 0

    RECORD.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
