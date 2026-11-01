"""Post-hoc cross-context calibration analysis of the 133-program campaign.

It reuses the 865 already-measured processes in
``build/es-abstraction-20261030/candidates/ANALYSIS_FINAL.json``. No new
measurement, no refit of the physical profiles, no change to the frozen
predictions. The only question is: if a new context is calibrated with a few of
its own programs, how much of the 21.95% mean APE survives on the held-out
programs?

Variants, all leave-one-workload-out:

  global          one scalar for every context
  per-cell        one scalar per (geometry, k) cell, fitted on other workloads
  cross-context   fit on all ``small`` cells, apply to ``real`` (and the reverse)

A scalar calibrated inside a cell multiplies every prediction in that cell, so
it cannot change the ranking among programs of one workload; the selection
regret is therefore reported as unchanged by construction and recomputed only
to confirm that.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "build", "es-abstraction-20261030", "candidates",
                      "ANALYSIS_FINAL.json")
OUT = os.path.join(ROOT, "build", "context-calibration-20261030")
SEED = 20261030
DRAWS = 20000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ape(pred, meas):
    return abs(pred - meas) / meas


def load():
    data = json.load(open(SOURCE, encoding="utf-8"))
    rows = data["rows"]
    for row in rows:
        row["cell"] = (row["geometry"], row["k"])
    return data, rows


def factors(rows):
    return [r["mean_ns"] / r["predicted_ns"] for r in rows]


def apply(rows, factor):
    return [ape(r["predicted_ns"] * factor, r["mean_ns"]) for r in rows]


def workload_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["workload_id"]].append(row)
    return groups


def loco(rows):
    """Leave-one-workload-out calibration inside each (geometry, k) cell."""
    groups = workload_groups(rows)
    errors = {}
    factors_used = {}
    for workload, wrows in groups.items():
        cell = wrows[0]["cell"]
        others = [r for r in rows
                  if r["cell"] == cell and r["workload_id"] != workload]
        factor = st.median(factors(others))
        factors_used[workload] = factor
        errors[workload] = apply(wrows, factor)
    return errors, factors_used


def loco_global(rows):
    groups = workload_groups(rows)
    errors = {}
    for workload, wrows in groups.items():
        others = [r for r in rows if r["workload_id"] != workload]
        factor = st.median(factors(others))
        errors[workload] = apply(wrows, factor)
    return errors


def cross_context(rows):
    """Fit one scalar on one geometry family and apply it to the other."""
    out = {}
    for fit_geom, test_geom in (("small", "real"), ("real", "small")):
        fit_rows = [r for r in rows if r["geometry"] == fit_geom]
        test_rows = [r for r in rows if r["geometry"] == test_geom]
        factor = st.median(factors(fit_rows))
        errors = apply(test_rows, factor)
        baseline = [ape(r["predicted_ns"], r["mean_ns"]) for r in test_rows]
        out[f"{fit_geom}->{test_geom}"] = {
            "factor": factor,
            "rows": len(test_rows),
            "baseline_mean_ape": sum(baseline) / len(baseline),
            "baseline_max_ape": max(baseline),
            "mean_ape": sum(errors) / len(errors),
            "max_ape": max(errors),
        }
    return out


def summarise(errors_by_workload):
    flat = [e for errs in errors_by_workload.values() for e in errs]
    per_workload = {w: sum(errs) / len(errs) for w, errs in errors_by_workload.items()}
    return {
        "rows": len(flat),
        "mean_ape": sum(flat) / len(flat),
        "max_ape": max(flat),
        "median_workload_ape": st.median(per_workload.values()),
        "per_workload": per_workload,
    }


def bootstrap(baseline_by_workload, calibrated_by_workload):
    """Workload-level paired bootstrap on the mean APE difference."""
    workloads = sorted(baseline_by_workload)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(DRAWS):
        sample = [rng.choice(workloads) for _ in workloads]
        b = [e for w in sample for e in baseline_by_workload[w]]
        c = [e for w in sample for e in calibrated_by_workload[w]]
        diffs.append((sum(b) / len(b)) - (sum(c) / len(c)))
    diffs.sort()
    return {
        "draws": DRAWS,
        "seed": SEED,
        "mean_difference": sum(diffs) / len(diffs),
        "ci95": [diffs[int(0.025 * len(diffs))],
                 diffs[int(0.975 * len(diffs)) - 1]],
        "fraction_positive": sum(1 for d in diffs if d > 0) / len(diffs),
    }


def selection(rows):
    """Chosen program per workload (min prediction) and regret vs measured best.
    A within-cell scalar cannot change either, so this only confirms it."""
    groups = workload_groups(rows)
    out = []
    for workload, wrows in groups.items():
        best = min(wrows, key=lambda r: r["mean_ns"])
        chosen = min(wrows, key=lambda r: r["predicted_ns"])
        out.append({
            "workload_id": workload,
            "chosen": chosen["program_id"],
            "measured_best": best["program_id"],
            "correct": chosen["program_id"] == best["program_id"],
            "regret": chosen["mean_ns"] / best["mean_ns"] - 1,
        })
    regrets = [x["regret"] for x in out]
    return {
        "workloads": len(out),
        "correct": sum(1 for x in out if x["correct"]),
        "median_regret": st.median(regrets),
        "max_regret": max(regrets),
        "scalar_invariant": True,
        "detail": out,
    }


def main():
    data, rows = load()
    baseline_by_workload = {w: [ape(r["predicted_ns"], r["mean_ns"]) for r in rs]
                            for w, rs in workload_groups(rows).items()}
    baseline = summarise(baseline_by_workload)

    loco_errors, factors_used = loco(rows)
    loco_summary = summarise(loco_errors)
    glob_errors = loco_global(rows)
    glob_summary = summarise(glob_errors)

    per_cell = {}
    for cell in sorted({r["cell"] for r in rows}):
        cell_rows = [r for r in rows if r["cell"] == cell]
        cell_workloads = sorted({r["workload_id"] for r in cell_rows})
        base = [e for w in cell_workloads for e in baseline_by_workload[w]]
        cal = [e for w in cell_workloads for e in loco_errors[w]]
        per_cell["%s-k%d" % cell] = {
            "rows": len(cell_rows),
            "workloads": len(cell_workloads),
            "baseline_mean_ape": sum(base) / len(base),
            "calibrated_mean_ape": sum(cal) / len(cal),
            "median_ratio": st.median(factors(cell_rows)),
        }

    result = {
        "schema": "context-calibration-posthoc-1",
        "source": {"path": SOURCE, "sha256": sha256(SOURCE)},
        "campaign": {
            "workloads": data["workloads"],
            "unique_programs": data["unique_programs"],
            "independent_processes": data["independent_processes"],
        },
        "no_new_measurement": True,
        "no_profile_refit": True,
        "baseline": baseline,
        "per_cell": per_cell,
        "leave_one_workload_out": {
            "per_cell_scalar": loco_summary,
            "global_scalar": glob_summary,
            "bootstrap_per_cell_minus_baseline": bootstrap(
                baseline_by_workload, loco_errors),
            "factors": factors_used,
        },
        "cross_context_transfer": cross_context(rows),
        "selection": selection(rows),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "RESULTS.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(path)
    print("baseline mean %.2f%% max %.2f%%" % (
        100 * baseline["mean_ape"], 100 * baseline["max_ape"]))
    print("per-cell LOCO mean %.2f%% max %.2f%%" % (
        100 * loco_summary["mean_ape"], 100 * loco_summary["max_ape"]))
    print("global LOCO mean %.2f%% max %.2f%%" % (
        100 * glob_summary["mean_ape"], 100 * glob_summary["max_ape"]))
    bs = result["leave_one_workload_out"]["bootstrap_per_cell_minus_baseline"]
    print("bootstrap diff mean %.2f pp, CI [%.2f, %.2f]" % (
        100 * bs["mean_difference"], 100 * bs["ci95"][0], 100 * bs["ci95"][1]))
    for key, val in per_cell.items():
        print("  %-10s rows=%3d workloads=%d base %.2f%% -> %.2f%% (median ratio %.3f)" % (
            key, val["rows"], val["workloads"], 100 * val["baseline_mean_ape"],
            100 * val["calibrated_mean_ape"], val["median_ratio"]))
    for key, val in result["cross_context_transfer"].items():
        print("  transfer %-12s factor %.3f %.2f%% -> %.2f%% (max %.2f%%)" % (
            key, val["factor"], 100 * val["baseline_mean_ape"],
            100 * val["mean_ape"], 100 * val["max_ape"]))
    sel = result["selection"]
    print("selection: %d/%d correct, median regret %.2f%%, max %.2f%%" % (
        sel["correct"], sel["workloads"], 100 * sel["median_regret"],
        100 * sel["max_regret"]))


if __name__ == "__main__":
    main()
