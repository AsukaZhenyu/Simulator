"""G4/G5 read-out for one model-column cohort: does the split price the depth entry higher?

    python -m profiling.real_moe_validation.prereg_check <cohort/model/formal>

G4 groups the bridge prices of one physical bridge by their key with the history
class removed, and compares the three classes at a coordinate that lies inside
all three observed domains. The coefficients are read from the frozen table; no
duration and no window is re-fitted here.

Two readings are printed. Per bridge, the three classes' means at a shared
coordinate. And the aggregate the v2 rule is judged on: each bridge contributes
its continuation sample count times the class mean, so that a model whose bridge
splits into many sub-resolution windows is read as one ordering rather than as
seventeen noisy ones.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path


def number(value):
    return Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else Fraction(value)


def mean_ms(price, query, kv, payload):
    return float(number(price["intercept_ns"])
                 + number(price["per_query_token_ns"]) * query
                 + number(price["per_attention_pair_ns"]) * query * kv
                 + number(price["per_payload_byte_ns"]) * payload) / 1e6


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args(argv)
    cohort = args.cohort.resolve()

    table = json.loads((cohort / "WINDOW_PRICES.json").read_text(encoding="utf-8"))
    weights = {row["key"]: row["sample_count"] for row in table["per_price_fit_diagnostics"]}
    groups = {}
    for price in table["prices"]:
        if price["scope"] != "RUNTIME_BRIDGE":
            continue
        key = json.loads(price["key"])
        history = key.pop("history", None)
        if history is None or not history.startswith("decode:"):
            continue
        groups.setdefault(json.dumps(key, sort_keys=True), {})[history] = price

    compared, violations = 0, []
    aggregate = {history: Fraction(0) for history in ("decode:first", "decode:settling", "decode:continuation")}
    weighted_groups, unweighted_groups = 0, 0
    for _, by_history in sorted(groups.items(), key=lambda item: str(item[1].get("decode:first", {}).get("observed_kv_padded"))):
        if len(by_history) < 2:
            continue
        low = max(min(p["observed_kv_padded"]) for p in by_history.values())
        high = min(max(p["observed_kv_padded"]) for p in by_history.values())
        payload = max(min(p.get("observed_payload_bytes") or [0]) for p in by_history.values())
        if low > high:
            print("no shared depth:", {k: (min(p["observed_kv_padded"]), max(p["observed_kv_padded"]))
                                       for k, p in by_history.items()})
            continue
        means = {history: mean_ms(price, 1, low, payload) for history, price in by_history.items()}
        compared += 1
        if "decode:continuation" in means:
            weight = weights.get(by_history["decode:continuation"]["key"])
            if weight:
                weighted_groups += 1
                for history, mean in means.items():
                    aggregate[history] += Fraction(mean) * weight
            else:
                unweighted_groups += 1
            for history in ("decode:first", "decode:settling"):
                if history in means and means[history] <= means["decode:continuation"]:
                    violations.append(dict(depth=low, history=history,
                                           means={k: round(v, 4) for k, v in means.items()}))
        print("bridge at depth", low, "payload", payload,
              {k: round(v, 4) for k, v in sorted(means.items())})

    evaluation = json.loads((cohort / "INDEPENDENT_EVALUATION.json").read_text(encoding="utf-8"))
    scored = [row for row in evaluation["rows"] if row["status"] == "evaluated"]
    unpriced = [row for row in evaluation["rows"] if row["status"] == "unsupported"]
    totals = {history: round(float(value), 4) for history, value in aggregate.items()}
    holds = (aggregate["decode:first"] >= aggregate["decode:settling"] > aggregate["decode:continuation"])
    print(json.dumps(dict(
        cohort=str(cohort),
        bridges_compared=compared,
        classes=sorted({history for by_history in groups.values() for history in by_history}),
        g4_violations=violations,
        g4_holds_per_bridge=not violations,
        g4_aggregate_ms=totals,
        g4_aggregate_weighted_groups=weighted_groups,
        g4_aggregate_unweighted_groups=unweighted_groups,
        g4_aggregate_holds=holds,
        extrapolated_windows=sum(row.get("extrapolated_windows", 0) for row in scored),
        unpriced_rows=len(unpriced),
        unpriced_reasons=sorted({str(row.get("reason")) for row in unpriced}),
        other_statuses=sorted({row["status"] for row in evaluation["rows"]} - {"evaluated", "unsupported"}),
    ), indent=1))
    return 0 if holds and not unpriced else 1


if __name__ == "__main__":
    raise SystemExit(main())
