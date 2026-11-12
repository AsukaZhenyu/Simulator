"""Print the measured GPU bridge cost of each decode forward of one held-out request.

    python -m profiling.real_moe_validation.diag_decode_shape <cohort> <placement> <request_id>

Reads the timing-free inventory run of a cohort (which does retain window
durations) and sums the RUNTIME_BRIDGE windows of each decode forward. Print
only; nothing is fitted or written.
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
    parser.add_argument("request_id")
    args = parser.parse_args(argv)
    out = args.cohort.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = tuple(manifest.get("labels", LEGACY_LABELS))
    policy = manifest.get("feature_policy", "mechanism_v1")
    label = next(name for name in labels if LABEL_PLACEMENT[name] == args.placement)
    execution = json.loads((out / "jobs" / (label + "-inventory") / "execution.json").read_text(encoding="utf-8"))
    native = json.loads(Path(execution["native"]["path"]).read_text(encoding="utf-8"))
    row = next(r for r in native["results"] if r["id"] == args.request_id)
    ledger = analysis.extract_windows(row, placement=LABEL_PLACEMENT[label],
                                      frequency_hz=native["clock"]["frequency_hz"], feature_policy=policy)
    by_forward = defaultdict(lambda: [0, None, set()])
    for window in ledger["windows"]:
        data = window["inventory"]
        if data["phase"] != "decode" or data["scope"] != "RUNTIME_BRIDGE":
            continue
        entry = by_forward[data["forward_index"]]
        entry[0] += float(window["observed_ns"]) / 1e6
        entry[1] = data["kv_padded"]
        entry[2].add(data["history"])
    print("placement", args.placement, "request", args.request_id)
    for forward in sorted(by_forward):
        total, kv, histories = by_forward[forward]
        print("forward %3d  kv %5d  bridge %8.3f ms  %s" % (forward, kv, total, ",".join(sorted(histories))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
