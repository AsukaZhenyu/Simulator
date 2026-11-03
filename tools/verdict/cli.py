"""Command line front end for the P1/P2 verdict tool.

Usage (from the repository root):

    modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json
    modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json --json
    modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json --tol 0.05

The JSON mode is stable machine-readable output; the default mode is a short
human-readable report. Input validation errors go to stderr and exit with 2.
No hardware, GPU or benchmark process is touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verdict import (OUTPUT_SCHEMA, ValidationError, analyze_crossover, analyze_ranking, load,
                     validate_tolerance)


def _tolerance(text):
    try:
        return validate_tolerance(float(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _percent(value):
    return "n/a" if value is None else "%.2f%%" % (100.0 * value)


def _interval_text(interval):
    return "%s -> %s" % (interval["from_label"], interval["to_label"])


def _endpoint_text(view):
    if view is None:
        return "missing"
    if view["faster"] is None:
        return "tied (gap %s)" % _percent(view["gap"])
    return "%s faster (gap %s)" % (view["faster"], _percent(view["gap"]))


def _p1_lines(result):
    summary = result["summary"]
    lines = [
        "P1 crossover  (tolerance %s, %d condition(s), %d interval(s))"
        % (_percent(result["tolerance"]), summary["conditions"], summary["intervals"]),
    ]
    if result["events"]:
        lines.append("  predicted crossovers:")
        for event in result["events"]:
            lines.append("    [%d] %s  %s vs %s" % (
                event["interval"]["from_index"], _interval_text(event["interval"]),
                event["config_pair"][0], event["config_pair"][1]))
            lines.append("        predicted: %s -> %s" % (
                _endpoint_text(event["predicted"]["from"]), _endpoint_text(event["predicted"]["to"])))
            lines.append("        measured:  %s -> %s  [%s / %s]" % (
                _endpoint_text(event["measured"]["from"]),
                _endpoint_text(event["measured"]["to"]), event["relation"],
                event["reason_code"]))
            lines.append("        verdict:   %s - %s" % (event["verdict"], event["reason"]))
    else:
        lines.append("  predicted crossovers: none")
    if result["ambiguous_crossovers"]:
        lines.append("  ambiguous predicted crossovers (tie at one endpoint): %d"
                     % len(result["ambiguous_crossovers"]))
    if result["skipped_pairs"]:
        lines.append("  skipped predicted pairs (missing value or configuration): %d"
                     % len(result["skipped_pairs"]))
    lines.append("  measured crossovers: %d" % len(result["measured_crossovers"]))
    for flip in result["measured_crossovers"]:
        lines.append("    %s  %s vs %s: %s -> %s" % (
            _interval_text(flip["interval"]), flip["config_pair"][0], flip["config_pair"][1],
            flip["from_faster"], flip["to_faster"]))
    lines.append("  summary: same-interval confirmed %d, adjacent-interval confirmed %d "
                 "(sensitivity only)" % (summary["confirmed_same_interval"],
                                          summary["confirmed_adjacent_interval"]))
    lines.append("           not confirmed (flip outside the adjacent intervals) %d, "
                 "inconclusive %d (coverage_unresolved %d)"
                 % (summary["not_confirmed"], summary["inconclusive"],
                    summary["coverage_unresolved"]))
    lines.append("  verdict: %s [%s]" % (result["verdict"], result["reason_code"]))
    lines.append("  %s" % result["reason"])
    return lines


def _p2_lines(result):
    pairs = result["pairs"]
    top1 = result["top1"]
    regret = result["regret"]
    lines = [
        "P2 ranking fidelity  (tolerance %s, %d condition(s))"
        % (_percent(result["tolerance"]), result["conditions_total"]),
        "  conditions: %d evaluated, %d incomplete, %d single-config, %d empty" % (
            len(result["conditions_evaluated"]), len(result["conditions_incomplete"]),
            len(result["conditions_single_config"]), len(result["conditions_empty"])),
        "  pairs: %d total, %d comparable (%d concordant, %d discordant), "
        "%d predicted tie, %d measured tie, %d both tie, %d missing" % (
            pairs["total"], pairs["comparable"], pairs["concordant"], pairs["discordant"],
            pairs["predicted_tie"], pairs["measured_tie"], pairs["both_tie"], pairs["missing"]),
        "  pairwise agreement: %s" % _percent(pairs["agreement"]),
        "  Kendall tau-b (tolerance ties): %s; (exact ties): %s" % (
            "n/a" if pairs["kendall_tau_b"] is None else "%.4f" % pairs["kendall_tau_b"],
            "n/a" if pairs["kendall_tau_b_exact"] is None else "%.4f" % pairs["kendall_tau_b_exact"]),
    ]
    if top1["evaluated"]:
        lines.append("  top-1: %d/%d (%s), predicted tie %d, measured tie %d" % (
            top1["hits"], top1["evaluated"], _percent(top1["hit_rate"]),
            top1["predicted_tie"], top1["measured_tie"]))
    else:
        lines.append("  top-1: not evaluable")
    if regret["evaluated"]:
        lines.append("  regret: mean %s, median %s, max %s over %d condition(s)" % (
            _percent(regret["mean"]), _percent(regret["median"]),
            _percent(regret["max"]), regret["evaluated"]))
    else:
        lines.append("  regret: not evaluable")
    for note in result["notes"]:
        lines.append("  note: " + note)
    return lines


def _render(document, kind, tolerance):
    payload = {
        "schema": OUTPUT_SCHEMA,
        "title": document.get("title"),
        "tolerance": tolerance,
        "tie_rule": "tie when |a-b| / ((|a|+|b|)/2) < tolerance",
        "tau_b_variants": {
            "kendall_tau_b": "ties induced by the relative-gap tolerance",
            "kendall_tau_b_exact": "ties only when the two values are exactly equal",
        },
    }
    text = []
    if kind in ("p1", "both"):
        payload["p1"] = analyze_crossover(document, tolerance)
        text += _p1_lines(payload["p1"]) + [""]
    if kind in ("p2", "both"):
        payload["p2"] = analyze_ranking(document, tolerance)
        text += _p2_lines(payload["p2"]) + [""]
    return payload, "\n".join(text).rstrip()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="path to the verdict input JSON")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--tol", type=_tolerance, default=0.02,
                        help="relative tie tolerance, default 0.02 (2%%)")
    parser.add_argument("--kind", choices=("p1", "p2", "both"), default="both",
                        help="which analysis to run (default: both)")
    args = parser.parse_args(argv)
    try:
        document = load(args.input)
    except ValidationError as error:
        for item in error.errors:
            print("%s: %s" % (item["path"], item["message"]), file=sys.stderr)
        return 2
    payload, report = _render(document, args.kind, args.tol)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        if document.get("title"):
            print(document["title"])
            print()
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
