"""Command line for the assembly layer.

Four commands, one per question, in the order they are allowed to be asked::

    python -m composing predict  --output DIR --table id=DIR ... --models models.json --models-dir DIR
    python -m composing compare  --output DIR
    python -m composing decompose --e2e DIR [--json PATH]
    python -m composing check    --result DIR [--json PATH]

``measure`` is not here: running llama-bench is measurement, and it lives in
``profiling/run_dense_endtoend.py measure``.  A prediction must exist before
that command will run, which is the pre-registration rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compare import compare
from .decompose import decompose, show
from .predict import models_document, predict, write_prediction
from .result_check import inspect_result, write_manifest
from .spec import dump_json, sha256_file
from .terms import Term, spec_of


def _terms(items) -> tuple[Term, ...]:
    terms = []
    for item in items or []:
        target, _, value = item.partition("=")
        name, _, phase = target.partition(":")
        if not name or not phase or not value:
            raise ValueError(f"--term needs name:phase=value, got {item!r}")
        spec = spec_of(name)
        terms.append(Term(name=name, phase=phase, value=float(value), enabled=True,
                          note=f"measured by: {spec.knob}"))
    return tuple(terms)


def _tables(items) -> dict[str, Path]:
    tables = {}
    for item in items:
        model_id, _, path = item.partition("=")
        if not model_id or not path:
            raise ValueError(f"--table needs model_id=directory, got {item!r}")
        tables[model_id] = Path(path)
    return tables


def _require_prediction(output: Path) -> dict:
    path = output / "predicted.json"
    if not path.is_file():
        raise ValueError("predicted.json must exist before any wall clock is measured")
    return {"path": str(path), "sha256": sha256_file(path),
            "mtime_ns": path.stat().st_mtime_ns,
            "content": json.loads(path.read_text(encoding="utf-8"))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m composing", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prediction = sub.add_parser("predict", help="price the declared shapes before measuring")
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--table", action="append", required=True,
                            help="model_id=export directory, repeatable")
    prediction.add_argument("--models", type=Path, required=True)
    prediction.add_argument("--models-dir", type=Path, required=True)
    prediction.add_argument("--term", action="append", default=None,
                            help="name:phase=value, repeatable; a value is a measurement, "
                                 "so every term given is enabled and recorded")

    comparing = sub.add_parser("compare", help="measured wall clock against the prediction")
    comparing.add_argument("--output", type=Path, required=True)

    decomposition = sub.add_parser("decompose", help="intercept and slope of the decode error")
    decomposition.add_argument("--e2e", type=Path, required=True,
                               help="directory holding predicted.json (and comparison.json)")
    decomposition.add_argument("--comparison", type=Path, action="append", default=None,
                               help="another comparison.json whose models replace the base one's, "
                                    "repeatable")
    decomposition.add_argument("--shape", action="append", default=None,
                               help="decode shape to use, repeatable (default: tg128, tg512, tg1024)")
    decomposition.add_argument("--json", type=Path, default=None)

    checking = sub.add_parser("check", help="accept a result that came back from another machine")
    checking.add_argument("--result", action="append", required=True, type=Path,
                          help="result directory holding measured.json, repeatable")
    checking.add_argument("--prediction", type=Path, default=None,
                          help="the prediction the rows must belong to, if not in the directory")
    checking.add_argument("--expect-llama-bench-sha256", default=None)
    checking.add_argument("--json", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "predict":
        document = predict(_tables(args.table), models_document(args.models), args.models_dir,
                           _terms(args.term))
        path = write_prediction(args.output, document)
        declared = document["composition_terms"]
        print(f"predicted {len(document['models'])} models -> {path}"
              + (f"  ({len(declared)} declared term(s))" if declared else "  (no declared terms)"))
        return 0
    if args.command == "compare":
        recorded = _require_prediction(args.output)
        measured = json.loads((args.output / "measured.json").read_text(encoding="utf-8"))
        if measured["prediction"]["sha256"] != recorded["sha256"]:
            raise ValueError("the prediction on disk is not the one the measurement was run against")
        document = compare(recorded["content"], measured)
        dump_json(args.output / "comparison.json", document)
        for model_id, rows in document["models"].items():
            for config, row in rows.items():
                print(f"{model_id:12s} {config:7s} predicted {row['predicted_ns']/1e6:9.3f} ms "
                      f"observed {row['observed_ns']/1e6:9.3f} ms  "
                      f"{100*row['signed_relative_error']:+6.2f}%")
        return 0
    if args.command == "decompose":
        prediction_doc = json.loads((args.e2e / "predicted.json").read_text(encoding="utf-8"))
        comparison = {"models": {}}
        for path in [args.e2e / "comparison.json", *(args.comparison or [])]:
            document = json.loads(path.read_text(encoding="utf-8"))
            comparison["models"].update(document.get("models") or {})
        report = decompose(prediction_doc, comparison, args.shape)
        show(report)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")
        return 0
    manifests = [inspect_result(directory, prediction=args.prediction,
                                expect_llama_bench_sha256=args.expect_llama_bench_sha256)
                 for directory in args.result]
    for manifest in manifests:
        status = "OK" if manifest["ok"] else "REJECTED"
        print(f"{status:8s} {manifest['result_directory']}")
        for problem in manifest["problems"]:
            print(f"  - {problem}")
    if args.json:
        value = manifests[0] if len(manifests) == 1 else manifests
        write_manifest(args.json, value)
    return 0 if all(manifest["ok"] for manifest in manifests) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"composing: {error}", file=sys.stderr)
        raise SystemExit(1)
