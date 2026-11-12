"""Per-window prediction residual of one inventory run, grouped by scope and node.

    python -m profiling.real_moe_validation.diag_window_residuals <cohort> <placement> <request_id> [phase]

The inventory run is timed, so its own windows can be compared with the frozen
prices window by window. That comparison is a diagnostic of where a condition's
error lives; it is not a fit and it is not used by any published number.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from profiling.real_moe_validation import runtime_window_analysis as analysis
from profiling.real_moe_validation.runtime_window_io import LABEL_PLACEMENT, LEGACY_LABELS
from llm_infer_model.performance.moe_execution import RuntimeWindowPrice


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    parser.add_argument("placement")
    parser.add_argument("request_id")
    parser.add_argument("phase", nargs="?", default="prefill")
    args = parser.parse_args(argv)
    out = args.cohort.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = tuple(manifest.get("labels", LEGACY_LABELS))
    policy = manifest.get("feature_policy", "mechanism_v1")
    table = json.loads((out / "WINDOW_PRICES.json").read_text(encoding="utf-8"))
    prices = {p.key: p for p in (RuntimeWindowPrice.from_dict(x) for x in table["prices"])}
    label = next(name for name in labels if LABEL_PLACEMENT[name] == args.placement)
    execution = json.loads((out / "jobs" / (label + "-inventory") / "execution.json").read_text(encoding="utf-8"))
    native = json.loads(Path(execution["native"]["path"]).read_text(encoding="utf-8"))
    row = next(r for r in native["results"] if r["id"] == args.request_id)
    ledger = analysis.extract_windows(row, placement=LABEL_PLACEMENT[label],
                                      frequency_hz=native["clock"]["frequency_hz"], feature_policy=policy)
    buckets = defaultdict(lambda: [0.0, 0.0, 0])
    for window in ledger["windows"]:
        data = window["inventory"]
        if data["phase"] != args.phase:
            continue
        price = prices[data["key"]]
        action = analysis.action_from_inventory(data)
        predicted = float(price.estimate(action, allow_extrapolation=True).mean_ns)
        measured = float(window["observed_ns"])
        name = data["metadata"]["CPU_node"]["name"] if data["metadata"]["CPU_node"] else data["scope"]
        bucket = buckets[(data["scope"], name, data["query_tokens"])]
        bucket[0] += measured
        bucket[1] += predicted
        bucket[2] += 1
    total_m = sum(v[0] for v in buckets.values()) / 1e6
    total_p = sum(v[1] for v in buckets.values()) / 1e6
    print("%s %s %s phase %s measured %.3f ms predicted %.3f ms (%+.2f%%)" % (
        args.placement, args.request_id, "", args.phase, total_m, total_p, 100 * (total_p / total_m - 1)))
    ranked = sorted(buckets.items(), key=lambda item: -abs(item[1][1] - item[1][0]))
    for (scope, name, query), (measured, predicted, count) in ranked[:15]:
        print("  %-14s %-22s q=%4d n=%4d measured %8.3f predicted %8.3f  diff %+8.3f ms (%+6.2f%%)" % (
            scope, name, query, count, measured / 1e6, predicted / 1e6,
            (predicted - measured) / 1e6, 100 * (predicted / measured - 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
