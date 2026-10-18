"""Optional evidence calibration CLI; no benchmark or hardware launches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .observations import SCHEMA, load_observations
from .parameters import estimate_parameters


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("estimate", help="profile observations -> model-owned physical parameters")
    estimate.add_argument("--observations", required=True)
    estimate.add_argument("--context", required=True, help="JSON file containing one exact applicability context")
    estimate.add_argument("--frozen-at-utc", default=None)
    estimate.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="heldout observations vs modeling prediction")
    evaluate.add_argument("--parameters", required=True)
    evaluate.add_argument("--graph", required=True)
    evaluate.add_argument("--heldout", required=True)
    evaluate.add_argument("--timeline")
    evaluate.add_argument("--task-id")
    evaluate.add_argument("--reference-timing-scope", choices=("whole_action_completed", "whole_iteration_completed"))
    evaluate.add_argument("--output", required=True)
    legacy = commands.add_parser("import-legacy", help="historical resource samples as measured references only")
    legacy.add_argument("--campaign", required=True)
    legacy.add_argument("--split", choices=("profile", "heldout"), default="profile")
    legacy.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "estimate":
        rows = load_observations(args.observations)
        frozen = args.frozen_at_utc or datetime.now(timezone.utc).isoformat()
        result = estimate_parameters(rows, context=_read(args.context), frozen_at_utc=frozen)
        _write(args.output, result)
    elif args.command == "evaluate":
        from llm_infer_model.performance import Graph, ResourceTimeline
        from .evaluator import evaluate_calibration
        graph = Graph.from_dict(_read(args.graph))
        timeline = None if args.timeline is None else ResourceTimeline.from_dict(_read(args.timeline))
        result = evaluate_calibration(graph, _read(args.parameters), load_observations(args.heldout),
            timeline=timeline, task_id=args.task_id, source_base=Path(args.parameters).resolve().parent,
            reference_timing_scope=args.reference_timing_scope)
        _write(args.output, result)
    else:
        from .legacy import import_legacy_resource_campaign
        result = import_legacy_resource_campaign(args.campaign, split=args.split)
        directory = Path(args.output_directory)
        directory.mkdir(parents=True, exist_ok=False)
        _write(directory / "observations.json", dict(schema=SCHEMA,
            observations=[row.to_dict() for row in result.observations]))
        _write(directory / "receipt.json", result.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
