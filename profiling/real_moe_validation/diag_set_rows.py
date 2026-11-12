"""Raw measured cost of the KV-cache write windows in one calibration job.

    python -m profiling.real_moe_validation.diag_set_rows <cohort> <placement>

SET_ROWS (GGML op 42) writes one row per query token into the cache. The price
of that window is currently a constant because the classifier leaves the op
unclassified. This prints the calibration samples it would be a function of:
query rows, padded KV depth, and the measured window.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from profiling.real_moe_validation import runtime_window_analysis as analysis
from profiling.real_moe_validation.runtime_window_io import LABEL_PLACEMENT, LEGACY_LABELS


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    parser.add_argument("placement")
    args = parser.parse_args(argv)
    out = args.cohort.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = tuple(manifest.get("labels", LEGACY_LABELS))
    policy = manifest.get("feature_policy", "mechanism_v1")
    label = next(name for name in labels if LABEL_PLACEMENT[name] == args.placement)
    execution = json.loads((out / "jobs" / (label + "-calibration-profile") / "execution.json").read_text(encoding="utf-8"))
    native = json.loads(Path(execution["native"]["path"]).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in native["results"]:
        ledger = analysis.extract_windows(row, placement=LABEL_PLACEMENT[label],
                                          frequency_hz=native["clock"]["frequency_hz"], feature_policy=policy)
        for window in ledger["windows"]:
            data = window["inventory"]
            node = data["metadata"]["CPU_node"]
            if data["phase"] != "prefill" or data["scope"] != "CPU_NODE" or not node or node["op"] != 42:
                continue
            grouped[(data["history"], data["kv_padded"])].append(
                (data["query_tokens"], float(window["observed_ns"]) / 1e6, row["id"]))
    for grid, samples in sorted(grouped.items()):
        print("history %s kv %s  n=%d" % (grid[0], grid[1], len(samples)))
        for query, ms, request in sorted(samples)[:10]:
            print("   q=%4d %8.4f ms  %s" % (query, ms, request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
