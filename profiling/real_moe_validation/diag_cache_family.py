"""How much of each condition's prefill error is the unclassified KV-cache write family?

    python -m profiling.real_moe_validation.diag_cache_family <cohort>

For every held-out inventory of a cohort, sums the measured and the frozen-price
windows of the prefill phase, split into the `cache_k*`/`cache_v*` CPU nodes and
everything else. Read-only; the numbers are a diagnostic, not a published score.
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
    args = parser.parse_args(argv)
    out = args.cohort.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = tuple(manifest.get("labels", LEGACY_LABELS))
    policy = manifest.get("feature_policy", "mechanism_v1")
    table = json.loads((out / "WINDOW_PRICES.json").read_text(encoding="utf-8"))
    prices = {p.key: p for p in (RuntimeWindowPrice.from_dict(x) for x in table["prices"])}
    for label in labels:
        execution = json.loads((out / "jobs" / (label + "-inventory") / "execution.json").read_text(encoding="utf-8"))
        native = json.loads(Path(execution["native"]["path"]).read_text(encoding="utf-8"))
        for row in native["results"]:
            ledger = analysis.extract_windows(row, placement=LABEL_PLACEMENT[label],
                                              frequency_hz=native["clock"]["frequency_hz"], feature_policy=policy)
            sums = defaultdict(lambda: [0.0, 0.0])
            for window in ledger["windows"]:
                data = window["inventory"]
                if data["phase"] != "prefill":
                    continue
                name = (data["metadata"]["CPU_node"] or {}).get("name") or data["scope"]
                group = "kv_write" if name.startswith(("cache_k", "cache_v")) else "rest"
                predicted = float(prices[data["key"]].estimate(analysis.action_from_inventory(data),
                                                               allow_extrapolation=True).mean_ns)
                sums[group][0] += float(window["observed_ns"])
                sums[group][1] += predicted
            measured = sum(v[0] for v in sums.values())
            predicted = sum(v[1] for v in sums.values())
            kv = sums["kv_write"]
            print("%-22s %-26s prefill %+6.2f%% | kv_write %6.2f ms meas %6.2f ms pred %+7.2f%% | rest %+6.2f%%" % (
                LABEL_PLACEMENT[label], row["id"], 100 * (predicted / measured - 1),
                kv[0] / 1e6, kv[1] / 1e6, 100 * (kv[1] / kv[0] - 1) if kv[0] else 0,
                100 * ((predicted - kv[1]) / (measured - kv[0]) - 1) if measured > kv[0] else 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
