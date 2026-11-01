"""Decision-level counterfactual for load-state (volatility) pricing.

It reuses two already-frozen independent evaluations. No parameter is refitted,
no new measurement is taken, and no measured start/internal boundary enters a
prediction. The question is only how the frozen predictions behave when they are
used to make a decision:

  M1 time        : already recorded per-process APE / total time
  M2 feasibility : will a plan finish inside a deadline D?
  M3 admission   : is there any plan inside D at all (best-of-set)?
  M4 ranking     : which of six measured plans is fastest?

Two campaigns are analysed separately and never pooled:

  continuous   build/dynamic-diagnosis-20261028/external-continuous-macro-v10
               (continuous calibration; plans cpu_tail / separated)
  bare_macro   build/dynamic-diagnosis-20261028/external-bare-macro-v2
               (prepared single-call calibration; plans early_joint / staggered)

Providers are the frozen ones: ``static00`` (one constant per-call price,
i.e. idle-only / no load-state dependence) and
``memoryless_four_state_macro`` (the response model that is integrated through
its own predicted completion times).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "volatility-decision-20261030")

SOURCES = {
    "continuous": os.path.join(
        ROOT, "build", "dynamic-diagnosis-20261028",
        "external-continuous-macro-v10", "INDEPENDENT_EVALUATION.json"),
    "bare_macro": os.path.join(
        ROOT, "build", "dynamic-diagnosis-20261028",
        "external-bare-macro-v2", "INDEPENDENT_EVALUATION.json"),
}

STATIC = "static00"
STATEFUL = "memoryless_four_state_macro"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cells(path):
    data = json.load(open(path, encoding="utf-8"))
    cells = {}
    for row in data["rows"]:
        key = (row["job_id"], row["provider"])
        cells[key] = {
            "job": row["job_id"],
            "plan": row["pulse_plan"],
            "provider": row["provider"],
            "predicted_ns": row["predicted_total_ns"],
            "measured_ns": row["measured_total_ns"],
            "ape": row["mean_response_APE"],
            "block_p90": row["block_APE_P90"],
            "block_predicted": row["predicted_block_ns"],
            "block_measured": row["measured_block_ns"],
        }
    jobs = sorted({k[0] for k in cells})
    plans = {j: cells[(j, STATIC)]["plan"] for j in jobs}
    return data, cells, jobs, plans


def deadline_sweep(cells, jobs, providers, lo, hi, step):
    """Return, per provider, the deadline intervals in which the verdict for a
    plan differs from measurement. Verdict = predicted_total <= D."""
    out = {}
    for provider in providers:
        wrong = []
        for job in jobs:
            c = cells[(job, provider)]
            pred, meas = c["predicted_ns"], c["measured_ns"]
            # verdicts differ exactly between the two boundaries
            start, end = min(pred, meas), max(pred, meas)
            if start == end:
                continue
            # direction of the error over that interval
            direction = "false_accept" if pred < meas else "false_reject"
            wrong.append({
                "job": job,
                "plan": c["plan"],
                "predicted_ns": pred,
                "measured_ns": meas,
                "wrong_interval_ns": [start, end],
                "wrong_interval_ms": round((end - start) / 1e6, 3),
                "wrong_interval_fraction_of_measured": (end - start) / meas,
                "direction": direction,
            })
        # how much of the swept grid each provider gets wrong
        total = 0
        bad = 0
        for d in range(int(lo), int(hi) + 1, int(step)):
            for job in jobs:
                c = cells[(job, provider)]
                total += 1
                if (c["predicted_ns"] <= d) != (c["measured_ns"] <= d):
                    bad += 1
        out[provider] = {
            "wrong_cells": wrong,
            "wrong_cells_all6": len(wrong) == len(jobs),
            "grid_points": total,
            "grid_mismatches": bad,
            "grid_mismatch_fraction": bad / total if total else 0.0,
        }
    return out


def admission(cells, jobs, providers):
    """Best-of-set decision: the provider picks the plan with the smallest
    predicted total; ties are declared as ties. The realised cost is the
    measured total of the picked plan."""
    measured = {job: cells[(job, STATIC)]["measured_ns"] for job in jobs}
    best_measured = min(measured.values())
    best_jobs = [j for j in jobs if measured[j] == best_measured]
    out = {}
    for provider in providers:
        predictions = {job: cells[(job, provider)]["predicted_ns"] for job in jobs}
        best_pred = min(predictions.values())
        picked = [j for j in jobs if predictions[j] == best_pred]
        tied = len(picked) == len(jobs)
        # if the provider cannot separate anything, report expected regret over
        # the uniform choice among the tied set
        expected = sum(measured[j] for j in picked) / len(picked)
        out[provider] = {
            "distinct_predicted_counts": len({predictions[j] for j in jobs}),
            "all_six_tied": tied,
            "picked_jobs_if_uniform_tie_break": picked,
            "picked_plans": sorted({cells[(j, provider)]["plan"] for j in picked}),
            "measured_best_jobs": best_jobs,
            "measured_best_plan": sorted({cells[(j, STATIC)]["plan"] for j in best_jobs}),
            "picked_measured_mean_ns": expected,
            "measured_best_ns": best_measured,
            "expected_regret_ns": expected - best_measured,
            "expected_regret_fraction_of_best": (expected - best_measured) / best_measured,
            "chosen_plan_family_matches_measured_best": (
                sorted({cells[(j, provider)]["plan"] for j in picked})
                == sorted({cells[(j, STATIC)]["plan"] for j in best_jobs})),
        }
    return out


def ranking(cells, jobs, providers):
    """Pairwise plan ordering over the six measured processes."""
    measured = {job: cells[(job, STATIC)]["measured_ns"] for job in jobs}
    out = {}
    for provider in providers:
        predicted = {job: cells[(job, provider)]["predicted_ns"] for job in jobs}
        concordant = discordant = tied_pred = tied_meas = 0
        pairs = []
        for i, a in enumerate(jobs):
            for b in jobs[i + 1:]:
                pd = predicted[a] - predicted[b]
                md = measured[a] - measured[b]
                if pd == 0:
                    tied_pred += 1
                    verdict = "tie_prediction"
                elif md == 0:
                    tied_meas += 1
                    verdict = "tie_measurement"
                elif (pd > 0) == (md > 0):
                    concordant += 1
                    verdict = "concordant"
                else:
                    discordant += 1
                    verdict = "discordant"
                pairs.append({"a": a, "b": b, "verdict": verdict,
                              "pred_diff_ns": pd, "meas_diff_ns": md})
        out[provider] = {
            "pairs": len(pairs),
            "concordant": concordant,
            "discordant": discordant,
            "tied_prediction": tied_pred,
            "tied_measurement": tied_meas,
            "kendall_tau_b": (
                (concordant - discordant) /
                (((concordant + discordant + tied_pred) *
                  (concordant + discordant + tied_meas)) ** 0.5)
                if concordant + discordant else None),
            "pair_detail": pairs,
        }
    return out


def family_admission(cells, jobs, providers, plans):
    """Same decision but at plan-family granularity: the three processes of one
    plan are repeats, not three different choices. The realised cost of a family
    is the mean measured total over its repeats."""
    families = sorted(set(plans.values()))
    family_measured = {}
    for family in families:
        vals = [cells[(j, STATIC)]["measured_ns"] for j in jobs
                if plans[j] == family]
        family_measured[family] = sum(vals) / len(vals)
    best_family = min(family_measured, key=family_measured.get)
    out = {}
    for provider in providers:
        family_pred = {}
        for family in families:
            vals = [cells[(j, provider)]["predicted_ns"] for j in jobs
                    if plans[j] == family]
            family_pred[family] = sum(vals) / len(vals)
        decisive = len(set(family_pred.values())) > 1
        if decisive:
            picked = [min(family_pred, key=family_pred.get)]
        else:
            # every family carries the same predicted cost: the provider has no
            # basis to choose, so charge the uniform choice over the tie
            picked = list(families)
        realised = sum(family_measured[f] for f in picked) / len(picked)
        regret = realised - family_measured[best_family]
        out[provider] = {
            "family_predicted_ns": family_pred,
            "family_measured_ns": family_measured,
            "picked_family": picked[0] if decisive else None,
            "tied_families": None if decisive else picked,
            "measured_best_family": best_family,
            "correct_family": (picked[0] == best_family) if decisive else None,
            "decisive": decisive,
            "expected_realised_ns": realised,
            "regret_ns": regret,
            "regret_fraction_of_best": regret / family_measured[best_family],
        }
    out["measured"] = {
        "family_measured_ns": family_measured,
        "measured_best_family": best_family,
    }
    return out


def boundaries(cells, jobs, providers, plans):
    """Deadline at which the workload stops fitting: min predicted total for a
    provider, versus the fastest measured process and the best measured family.
    The gap is the window in which that provider's accept/reject verdict is
    wrong for the best available plan."""
    measured_process = min(cells[(j, STATIC)]["measured_ns"] for j in jobs)
    families = sorted(set(plans.values()))
    family_mean = {}
    for family in families:
        vals = [cells[(j, STATIC)]["measured_ns"] for j in jobs
                if plans[j] == family]
        family_mean[family] = sum(vals) / len(vals)
    best_family_mean = min(family_mean.values())
    out = {"measured_best_process_ns": measured_process,
           "measured_best_family_mean_ns": best_family_mean}
    for provider in providers:
        pred_best = min(cells[(j, provider)]["predicted_ns"] for j in jobs)
        out[provider] = {
            "predicted_best_ns": pred_best,
            "wrong_window_vs_best_process_ns": abs(pred_best - measured_process),
            "wrong_window_vs_best_process_fraction": (
                abs(pred_best - measured_process) / measured_process),
            "direction": ("false_accept" if pred_best < measured_process
                          else "false_reject"),
        }
    return out


def block_level(cells, jobs, providers):
    """Could a scheduler that must place a delay have known which blocks are
    slow? Static predicts identical blocks; report the spread it misses."""
    out = {}
    for provider in providers:
        spreads = []
        for job in jobs:
            c = cells[(job, provider)]
            measured_blocks = c["block_measured"]
            pred_blocks = c["block_predicted"]
            spreads.append({
                "job": job,
                "plan": c["plan"],
                "measured_min_ns": min(measured_blocks),
                "measured_max_ns": max(measured_blocks),
                "measured_max_over_min": max(measured_blocks) / min(measured_blocks),
                "predicted_max_over_min": max(pred_blocks) / min(pred_blocks),
                "block_ape_p90": c["block_p90"],
            })
        out[provider] = spreads
    return out


def analyse_campaign(name, path):
    data, cells, jobs, plans = load_cells(path)
    providers = [STATIC, STATEFUL]
    measured_totals = {j: cells[(j, STATIC)]["measured_ns"] for j in jobs}
    lo = min(measured_totals.values()) * 0.6
    hi = max(measured_totals.values()) * 1.15
    result = {
        "campaign": name,
        "source": {"path": path, "sha256": sha256(path)},
        "schema": data.get("schema"),
        "plans": plans,
        "processes": len(jobs),
        "repeats_per_plan": defaultdict(int),
        "measured_totals_ns": measured_totals,
        "time": {},
        "feasibility": deadline_sweep(cells, jobs, providers, lo, hi, 1_000_000),
        "admission": admission(cells, jobs, providers),
        "family_admission": family_admission(cells, jobs, providers, plans),
        "boundaries": boundaries(cells, jobs, providers, plans),
        "ranking": ranking(cells, jobs, providers),
        "block_level": block_level(cells, jobs, providers),
    }
    for plan in sorted(set(plans.values())):
        result["repeats_per_plan"][plan] = sum(1 for p in plans.values() if p == plan)
    for provider in providers:
        apes = [cells[(j, provider)]["ape"] for j in jobs]
        p90 = [cells[(j, provider)]["block_p90"] for j in jobs]
        result["time"][provider] = {
            "mean_ape": sum(apes) / len(apes),
            "max_ape": max(apes),
            "mean_block_p90": sum(p90) / len(p90),
            "worst_block_p90": max(p90),
        }
    return result


def main():
    results = {
        "schema": "volatility-decision-counterfactual-1",
        "generated_by": os.path.abspath(__file__),
        "no_refit": True,
        "no_new_measurement": True,
        "measured_data_reused": True,
        "campaigns": {},
    }
    for name, path in SOURCES.items():
        if not os.path.exists(path):
            results["campaigns"][name] = {"missing": path}
            continue
        results["campaigns"][name] = analyse_campaign(name, path)
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=dict)
        f.write("\n")
    print(out_path)
    for name, c in results["campaigns"].items():
        if "missing" in c:
            print(name, "MISSING", c["missing"])
            continue
        print("==", name)
        for provider in (STATIC, STATEFUL):
            t = c["time"][provider]
            f = c["feasibility"][provider]
            a = c["admission"][provider]
            r = c["ranking"][provider]
            print("  %-32s meanAPE=%.2f%% max=%.2f%% blockP90=%.1f%%" % (
                provider, t["mean_ape"] * 100, t["max_ape"] * 100,
                t["worst_block_p90"] * 100))
            print("      feasibility: %d/%d cells wrong, grid mismatch %.1f%%" % (
                len(f["wrong_cells"]), c["processes"],
                f["grid_mismatch_fraction"] * 100))
            print("      admission  : tied=%s picked=%s best=%s regret=%.1fms (%.2f%%)" % (
                a["all_six_tied"], a["picked_plans"], a["measured_best_plan"],
                a["expected_regret_ns"] / 1e6,
                a["expected_regret_fraction_of_best"] * 100))
            print("      ranking    : concordant=%d discordant=%d tied=%d/%d tau=%.3f" % (
                r["concordant"], r["discordant"], r["tied_prediction"],
                r["tied_measurement"],
                r["kendall_tau_b"] if r["kendall_tau_b"] is not None else float("nan")))
        fam = c["family_admission"]
        b = c["boundaries"]
        print("  family: measured best=%s %s" % (
            fam["measured"]["measured_best_family"],
            {k: round(v / 1e6, 1) for k, v in fam["measured"]["family_measured_ns"].items()}))
        for provider in (STATIC, STATEFUL):
            print("    %-32s picked=%s correct=%s regret=%.1fms (%.2f%%)" % (
                provider, fam[provider]["picked_family"] or "TIE:" + ",".join(fam[provider]["tied_families"]),
                fam[provider]["correct_family"], fam[provider]["regret_ns"] / 1e6,
                fam[provider]["regret_fraction_of_best"] * 100))
            print("      boundary: pred best %.1f ms vs measured best process %.1f ms (%.1f%%)" % (
                b[provider]["predicted_best_ns"] / 1e6,
                b["measured_best_process_ns"] / 1e6,
                b[provider]["wrong_window_vs_best_process_fraction"] * 100))


if __name__ == "__main__":
    main()
