"""Summarize separate host stage diagnostics without changing accepted totals."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {"count": len(values), "min_ns": values[0], "median_ns": statistics.median(values),
            "p95_invocation_ns": values[min(len(values)-1, int(.95*(len(values)-1)))], "max_ns": values[-1]}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "measured":
            record = json.loads(Path(row["raw_path"]).read_text(encoding="utf-8"))
            payload = json.loads(record["stdout"])
            for strategy, traces in payload["chain_traces"].items():
                groups[(row["case"]["id"], row["split"], strategy)].append((row, payload, traces))
    result = []
    for (cid, split, strategy), entries in sorted(groups.items()):
        stages = defaultdict(list)
        totals, uninstrumented, skew, waits, overlap, noop = [], [], [], [], [], []
        for row, payload, traces in entries:
            uninstrumented.extend(payload["wall_ns"][strategy])
            noop.extend(payload["timestamp_overhead_ns"]["noop_trace/" + strategy])
            for trace in traces:
                totals.append(trace["end_ns"])
                for stage in trace["stages"]:
                    stages[(stage["thread"], stage["name"])].append(stage["end_ns"]-stage["begin_ns"])
                if trace["join_wait_ns"] is not None:
                    skew.append(trace["branch_start_skew_ns"])
                    waits.append(trace["join_wait_ns"])
                    overlap.append(trace["completed_work_bracket_overlap_ns"])
        result.append({"case_id": cid, "split": split, "strategy": strategy,
                       "process_count": len(entries), "uninstrumented_total": summary(uninstrumented),
                       "instrumented_total": summary(totals), "noop_trace_outer_timer": summary(noop),
                       "instrumented_to_uninstrumented_median_ratio": statistics.median(totals)/statistics.median(uninstrumented),
                       "branch_start_skew": summary(skew), "join_wait": summary(waits),
                       "completed_work_bracket_overlap": summary(overlap),
                       "stages": [{"thread": thread, "name": name, **summary(values)}
                                  for (thread, name), values in sorted(stages.items())]})
    return {"schema": "chain-diagnostic-summary-0.1", "timestamp_overhead_subtracted": False,
            "timing_unit": "integer nanoseconds per separate diagnostic invocation",
            "p95_unit": "pooled diagnostic invocation, not process or prediction error",
            "kernel_overlap_proved": False, "context_switch_attribution_proved": False,
            "results": result}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.experiment / "measurements.json").read_text(encoding="utf-8"))
    output = args.experiment / "diagnostic-summary.json"
    if output.exists():
        raise ValueError("diagnostic output already exists")
    output.write_text(json.dumps(summarize(rows), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
