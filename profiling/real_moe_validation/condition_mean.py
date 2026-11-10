"""Average the independent evaluation into one row per condition.

`runtime_window_predict evaluate` scores every bare run it is handed: two fresh
processes per placement, six held-out requests each. The paper reports a mean
over *conditions* -- one placement and one request -- not over launches, so this
pass averages each condition's repeats first and only then across conditions. A
condition counts only when every declared repeat produced an endpoint match; a
condition whose token or logit endpoint failed is retained with its reason and
excluded from the mean rather than silently averaged in.

    python -m profiling.real_moe_validation.condition_mean --out <cohort> \\
        --evaluation <cohort>/INDEPENDENT_EVALUATION.json

Writes `CONDITION_MEAN_EVALUATION.json` next to the cohort's manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .dataset import read, write
from .runtime_window_io import receipt

METRICS = ("total", "prefill", "decode_with_sampling")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)

    evaluation_path = (args.evaluation or args.out / "INDEPENDENT_EVALUATION.json").resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows = evaluation["rows"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["placement"], row["request_id"])].append(row)

    conditions = []
    for (placement, request_id), repeats in sorted(grouped.items()):
        scored = [r for r in repeats if r["status"] == "evaluated"]
        record = dict(placement=placement, request_id=request_id, repeats=len(repeats),
                      endpoint_passed=len(scored), status="evaluated" if len(scored) == len(repeats) else "incomplete")
        if len(scored) != len(repeats):
            record["reasons"] = sorted({r.get("reason") or r.get("status") for r in repeats if r["status"] != "evaluated"})
            conditions.append(record)
            continue
        mean_measured = {m: sum(r["measured_ns"][m] for r in scored) / len(scored) for m in METRICS}
        predicted = {m: float(scored[0]["predicted_ns"][m]) if not isinstance(scored[0]["predicted_ns"][m], dict)
                     else scored[0]["predicted_ns"][m]["numerator"] / scored[0]["predicted_ns"][m]["denominator"]
                     for m in METRICS}
        # The error of a condition is the prediction against that condition's mean
        # measured time -- not the mean of the per-launch errors. The two differ
        # whenever the repeats differ from each other, and the published table used
        # the former.
        ape = {m: abs(predicted[m] / mean_measured[m] - 1) for m in METRICS}
        record.update(mean_measured_ns=mean_measured, predicted_ns=predicted, APE=ape,
                      passed=all(v <= evaluation.get("target_APE", 0.1) for v in ape.values()),
                      extrapolated_windows=sum(r.get("extrapolated_windows", 0) for r in scored))
        conditions.append(record)

    complete = [c for c in conditions if c["status"] == "evaluated"]
    result = dict(schema="real-moe-condition-mean-evaluation-1",
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  evaluation=receipt(evaluation_path),
                  parameters_refitted=evaluation.get("parameters_refitted"),
                  declared_conditions=len(conditions), complete_endpoint_conditions=len(complete),
                  passed_conditions=sum(c["passed"] for c in complete),
                  mean_APE={m: sum(c["APE"][m] for c in complete) / len(complete) if complete else None for m in METRICS},
                  worst_APE={m: max((c["APE"][m] for c in complete), default=None) for m in METRICS},
                  conditions=conditions,
                  unit="one placement and one held-out request; the prediction is scored against the repeats' mean measured time",
                  incomplete_conditions_excluded_not_averaged=True)
    write(args.out / "CONDITION_MEAN_EVALUATION.json", result)
    print(json.dumps(dict(declared=len(conditions), complete=len(complete),
                          mean_total=result["mean_APE"]["total"], worst_total=result["worst_APE"]["total"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
