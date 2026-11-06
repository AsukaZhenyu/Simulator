"""Command line for the assembly layer.

Six commands, one per question, in the order they are allowed to be asked::

    python -m composing predict  --output DIR --table id=DIR ... --models models.json --models-dir DIR
    python -m composing compare  --output DIR
    python -m composing decompose --e2e DIR [--json PATH]
    python -m composing check    --result DIR [--json PATH]
    python -m composing pairs    --a DIR --b DIR [--json PATH]
    python -m composing band     --run PROBE.json [--run ...] [--json PATH]
    python -m composing table-state --table DIR --sensor PROBE.json [--sensor ...] [--json PATH]

``table-state`` is the one that decides whether a prediction may be compared with
a wall clock at all: a table is priced in one machine state, and this machine has
moved between states by ~9-14% with the same binaries.

``measure`` is not here: running llama-bench is measurement, and it lives in
``profiling/run_dense_endtoend.py measure``.  A prediction must exist before
that command will run, which is the pre-registration rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from . import variability
from .compare import compare
from .decompose import decompose, show
from .pairs import pairs, show as show_pairs
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


_SHAPE_KEY = re.compile(r"^(?P<phase>[a-z_]+)/B(?P<batch>\d+)/q(?P<q>\d+)/hist(?P<history>\d+)$")


def _table_blocks(directory: Path) -> dict[str, int]:
    """The joint ``block`` record of every shape: the state the table was priced in."""
    report = json.loads((directory / "export_report.json").read_text(encoding="utf-8"))
    blocks = {entry["shape"]: int(entry["sample_summary"]["median_ns"])
              for entry in report["entries"] if entry["kind"] == "block"}
    if not blocks:
        raise ValueError(f"{directory}: the report keeps no joint block record, so the "
                         "state it was priced in cannot be checked")
    return blocks


def _sensor_blocks(paths, reference: dict[str, int]) -> dict[str, int]:
    """Today's block medians, matched to the table's shapes by (q, history).

    A sensor file may hold several repeats of the same shape; the state is one
    number per shape, so the median of the repeats is used and no repeat can
    pick the answer by being last in the file.
    """
    by_shape = {}
    for shape in reference:
        match = _SHAPE_KEY.match(shape)
        if not match:
            raise ValueError(f"{shape!r} is not a shape key this reader understands")
        by_shape[(int(match["q"]), int(match["history"]))] = shape
    found: dict[str, list[int]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for record in variability.block_records(document, where=str(path)):
            shape = by_shape.get((record["q"], record["history"]))
            if shape is not None:
                found.setdefault(shape, []).append(int(record["median_ns"]))
    sensor = {}
    for shape, values in found.items():
        ordered = sorted(values)
        middle = len(ordered) // 2
        sensor[shape] = (ordered[middle] if len(ordered) % 2
                         else (ordered[middle - 1] + ordered[middle]) // 2)
    return sensor


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

    pairing = sub.add_parser("pairs", help="two wall clocks of one prediction, side by side")
    pairing.add_argument("--a", type=Path, required=True,
                         help="directory holding a measured.json, or the file itself")
    pairing.add_argument("--b", type=Path, required=True, help="the same, for the second round")
    pairing.add_argument("--json", type=Path, default=None)

    banding = sub.add_parser("band", help="where the clock sat during a probe run")
    banding.add_argument("--run", action="append", required=True, type=Path,
                         help="a probe result, archived ({'document': ...}) or printed by the "
                              "probe itself; repeatable")
    banding.add_argument("--json", type=Path, default=None)

    state = sub.add_parser("table-state",
                           help="was this table priced in the state the clock is in now")
    state.add_argument("--table", type=Path, required=True,
                       help="an export directory holding export_report.json")
    state.add_argument("--sensor", action="append", required=True, type=Path,
                       help="a joint block run of the same probe, taken now; repeatable")
    state.add_argument("--tolerance", type=float, default=variability.DEFAULT_TABLE_TOLERANCE,
                       help=f"how far the block may move and still be one state "
                            f"(default {variability.DEFAULT_TABLE_TOLERANCE})")
    state.add_argument("--json", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "pairs":
        def measured_at(path: Path) -> dict:
            source = path / "measured.json" if path.is_dir() else path
            return json.loads(source.read_text(encoding="utf-8"))
        report = pairs(measured_at(args.a), measured_at(args.b),
                       a_name=str(args.a), b_name=str(args.b))
        show_pairs(report)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            dump_json(args.json, report)
        return 0
    if args.command == "band":
        report = {"schema_version": variability.SCHEMA_VERSION,
                  "band": variability.DEFAULT_BAND, "runs": []}
        for path in args.run:
            document = json.loads(path.read_text(encoding="utf-8"))
            described = variability.describe(variability.wall_ns_of(document, where=str(path)),
                                             where=str(path))
            described["path"] = str(path)
            report["runs"].append(described)
            summary, modes = described["summary"], described["modes"]
            head = (f"{path.name:44s} n={summary['n']:5d} "
                    f"med={summary['median_ns']/1e3:8.1f}us "
                    f"spread={100*summary['spread']:5.1f}%  ")
            if modes["single_mode"] is True:
                print(head + "one mode")
            elif modes["single_mode"] is None:
                print(head + modes["reason"])
            else:
                print(head + f"fast={modes['fast_ns']/1e3:7.1f} slow={modes['slow_ns']/1e3:7.1f} "
                             f"x{modes['ratio']:.3f} slow={100*modes['slow_fraction']:4.1f}%")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            dump_json(args.json, report)
        return 0
    if args.command == "table-state":
        reference = _table_blocks(args.table)
        sensor = _sensor_blocks(args.sensor, reference)
        document = variability.table_state(reference, sensor, tolerance=args.tolerance)
        print(f"{args.table.name}/block  {len(document['rows'])} shape(s) compared, "
              f"median ratio {document['median_ratio']:.4f}, "
              f"widest {document['widest_shape']} x{document['widest_ratio']:.4f} "
              f"(tolerance {document['tolerance']:.2%})")
        for row in document["rows"]:
            print(f"  {row['shape']:28s} table {row['table_ns']/1e6:9.3f} ms  "
                  f"now {row['sensor_ns']/1e6:9.3f} ms  x{row['ratio']:.4f}"
                  + ("" if row["within_band"] else "   outside the band"))
        missing = document["uncompared"]["table_only"]
        if missing:
            print(f"  not compared: {', '.join(missing)}")
        if document["verdict"] == "same_band":
            print("verdict: one state -- the table and this clock are one comparison")
        else:
            print("verdict: different state -- the table was priced in another band, so an "
                  "error read across it is not a statement about the cost model.  The ratio "
                  "is not a correction: the probe moves less than the engine does.")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            dump_json(args.json, document)
        return 0
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
            # The header travels with the rows.  The decode grouping decides
            # which depths paid a process's first-decode cost, so a file that
            # lost it would report a slope with no protocol.
            for key in ("decode_grouping", "warmup"):
                declared, seen = document.get(key), comparison.get(key)
                if declared is None:
                    continue
                if seen is not None and seen != declared:
                    raise ValueError(f"{path}: declares {key} {declared!r} but another "
                                     f"comparison in this merge declares {seen!r}; the "
                                     "rows answer different questions")
                comparison[key] = declared
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
