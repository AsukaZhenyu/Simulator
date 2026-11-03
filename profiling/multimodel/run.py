# -*- coding: utf-8 -*-
"""The three steps every device runs before it is allowed to measure anything.

    survey     what this machine is, from observation only
    fetch      materialise the declared model bytes and prove their SHA-256
    structure  re-read each local file and prove its geometry matches the
               manifest, so a substituted or re-quantised GGUF cannot pass

Those are the gates.  Measurement and prediction are not reimplemented here:
they are the existing suites (see README.md), because a second copy of a
measurement protocol is a second chance to get the protocol wrong.

    python -m profiling.multimodel.run status
    python -m profiling.multimodel.run survey --out campaign/device.json
    python -m profiling.multimodel.run fetch
    python -m profiling.multimodel.run structure --out campaign/structure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from profiling.multimodel import devices, fetch_models

HERE = Path(__file__).resolve().parent


def _flatten(node, prefix=""):
    """Yield (dotted path, value) for every leaf, so a comparison names the field."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1], node


def _equal(declared, observed) -> bool:
    if isinstance(declared, float) or isinstance(observed, float):
        if declared is None or observed is None:
            return declared is observed
        return math.isclose(float(declared), float(observed), rel_tol=1e-5, abs_tol=1e-12)
    return declared == observed


def compare(declared: dict, observed: dict) -> dict:
    """Compare only the fields the manifest asserts; null means "not asserted".

    A null is not a permission slip: every field with a value must match exactly,
    and a field the manifest does not mention is reported as unasserted rather
    than silently ignored.
    """
    mismatches, checked, unasserted = [], [], []
    seen = set()
    for path, value in _flatten(declared):
        seen.add(path)
        current = observed
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = KeyError
                break
            current = current[part]
        if current is KeyError:
            mismatches.append({"field": path, "declared": value, "observed": None,
                               "reason": "absent from the generated geometry"})
            continue
        if value is None:
            unasserted.append(path)
            continue
        checked.append(path)
        if not _equal(value, current):
            mismatches.append({"field": path, "declared": value, "observed": current,
                               "reason": "differs"})
    extra = [path for path, _ in _flatten(observed) if path not in seen]
    return {"ok": not mismatches, "checked": sorted(checked),
            "unasserted_null": sorted(unasserted), "mismatches": mismatches,
            "observed_only": sorted(extra)}


def structure(model: dict, gguf: Path) -> dict:
    from profiling.model_swap.build_model import build_geometry, metadata_from_gguf

    meta = metadata_from_gguf(gguf)
    geometry = build_geometry(meta, layer_index=None)
    result = compare(model["geometry"], geometry)
    return {"id": model["id"], "gguf": str(gguf), "sha256_manifest": model["sha256"],
            "geometry": geometry, "comparison": result}


def _write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def command_status(args) -> int:
    document = fetch_models.read_manifest(args.manifest)
    print(f"root: {args.root}")
    print(f"{'id':<24} {'role':<5} {'GB':>6}  bytes")
    for model in document["models"]:
        path = fetch_models.target(args.root, model)
        reason = fetch_models.verify(path, model)
        state = "ok" if reason is None else reason
        print(f"{model['id']:<24} {model['role']:<5} {model['bytes'] / 1e9:6.2f}  {state}")
    return 0


def command_structure(args) -> int:
    document = fetch_models.read_manifest(args.manifest)
    problems = 0
    for model in document["models"]:
        if args.only and model["id"] not in args.only:
            continue
        path = fetch_models.target(args.root, model)
        reason = fetch_models.verify(path, model)
        if reason is not None:
            print(f"{model['id']:<24} skipped: {reason} (run fetch first)")
            problems += 1
            continue
        record = structure(model, path)
        comparison = record["comparison"]
        _write_new(args.out / f"{model['id']}.structure.json", record)
        if comparison["ok"]:
            print(f"{model['id']:<24} ok ({len(comparison['checked'])} fields, "
                  f"{len(comparison['unasserted_null'])} not asserted)")
        else:
            problems += 1
            print(f"{model['id']:<24} MISMATCH")
            for row in comparison["mismatches"]:
                print(f"  {row['field']}: declared {row['declared']!r} observed {row['observed']!r}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=fetch_models.default_root())
    parser.add_argument("--manifest", type=Path, default=fetch_models.MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)

    survey = sub.add_parser("survey", help="describe this machine")
    survey.add_argument("--out", type=Path)

    fetch = sub.add_parser("fetch", help="download and hash-verify the model set")
    fetch.add_argument("--only", action="append", default=[])

    form = sub.add_parser("structure", help="verify each local file's geometry")
    form.add_argument("--out", type=Path, required=True)
    form.add_argument("--only", action="append", default=[])

    sub.add_parser("status", help="which models are present and verified")

    args = parser.parse_args(argv)
    if args.command == "survey":
        argv_ = ["--out", str(args.out)] if args.out else []
        return devices.main(argv_)
    if args.command == "fetch":
        argv_ = ["--root", str(args.root), "--manifest", str(args.manifest)]
        for only in args.only:
            argv_ += ["--only", only]
        return fetch_models.main(argv_)
    if args.command == "structure":
        return command_structure(args)
    return command_status(args)


if __name__ == "__main__":
    sys.exit(main())
