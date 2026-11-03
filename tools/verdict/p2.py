"""P2: ranking fidelity between predicted and measured latencies.

For every condition, all configuration pairs with both a predicted and a
measured latency contribute to the pairwise agreement and to Kendall's tau-b.
Pairs whose two sides are within ``tolerance`` are ties and are counted
separately, never silently dropped. Pairwise agreement is
``concordant / (concordant + discordant)``, i.e. both-tie, one-side-tie and
missing pairs are excluded from its denominator and reported in ``pairs``.

Top-1 and regret use whole conditions only. A condition is *evaluated* when it
has at least two configurations and every configuration carries both latencies;
otherwise it is classified as ``single_config``, ``incomplete`` or ``empty`` and
listed under those names. Top-1 counts a hit when the predicted fastest
configuration is also the measured fastest, with both "fastest" labels unique
outside the tolerance (a within-tolerance top tie is reported, not guessed).
Regret is ``(measured[selected] - measured_best) / measured_best`` for the
uniquely predicted-fastest configuration, so it is always non-negative.
"""
from __future__ import annotations

from itertools import combinations
import math
import statistics

from verdict.schema import order, validate_tolerance

_PAIR_KEYS = (
    "total",
    "missing", "missing_predicted", "missing_measured", "missing_both",
    "concordant", "discordant", "predicted_tie", "measured_tie", "both_tie",
    "exact_concordant", "exact_discordant",
    "exact_predicted_tie", "exact_measured_tie", "exact_both_tie",
)
_PRESENT = ("concordant", "discordant", "predicted_tie", "measured_tie", "both_tie")


def analyze(document, tolerance=0.02):
    """Return the P2 ranking-fidelity report for a validated input document."""
    tolerance = validate_tolerance(tolerance)
    rows = [_condition(condition, tolerance) for condition in document["conditions"]]
    pairs = _sum_pairs(rows)
    top1 = _summarize_top1(rows)
    regret = _summarize_regret(rows)
    single = [row["label"] for row in rows if row["kind"] == "single_config"]
    empty = [row["label"] for row in rows if row["kind"] == "empty"]
    incomplete = [{"label": row["label"], "missing_values": row["missing_values"]}
                  for row in rows if row["kind"] == "incomplete"]
    evaluated = [row["label"] for row in rows if row["kind"] == "evaluated"]
    notes = []
    if single:
        notes.append("%d condition(s) have a single configuration and cannot be ranked: %s"
                     % (len(single), ", ".join(single)))
    if empty:
        notes.append("%d condition(s) have no configurations: %s" % (len(empty), ", ".join(empty)))
    if incomplete:
        notes.append("%d condition(s) have missing values; complete pairs still contribute to the "
                     "pairwise metrics, but the condition is excluded from top-1/regret: %s"
                     % (len(incomplete), ", ".join(item["label"] for item in incomplete)))
    if pairs["missing"]:
        notes.append("%d configuration pair(s) were excluded from pairwise metrics because a value "
                     "is missing (%d with a missing predicted value, %d with a missing measured "
                     "value, %d with both)"
                     % (pairs["missing"], pairs["missing_predicted"],
                        pairs["missing_measured"], pairs["missing_both"]))
    return {
        "tolerance": tolerance,
        "conditions_total": len(rows),
        "conditions_evaluated": evaluated,
        "conditions_incomplete": incomplete,
        "conditions_single_config": single,
        "conditions_empty": empty,
        "pairs": pairs,
        "top1": top1,
        "regret": regret,
        "conditions": rows,
        "notes": notes,
    }


def _condition(condition, tolerance):
    entries = [
        {
            "name": config["name"],
            "predicted": config.get("predicted_latency_ns"),
            "measured": config.get("measured_latency_ns"),
        }
        for config in condition["configs"]
    ]
    complete = [entry for entry in entries
                if entry["predicted"] is not None and entry["measured"] is not None]
    missing = [
        {
            "config": entry["name"],
            "missing": [field for field, side in (
                ("predicted_latency_ns", "predicted"), ("measured_latency_ns", "measured"))
                if entry[side] is None],
        }
        for entry in entries
        if entry["predicted"] is None or entry["measured"] is None
    ]
    if not entries:
        kind = "empty"
    elif len(entries) == 1:
        kind = "single_config"
    elif len(complete) == len(entries):
        kind = "evaluated"
    else:
        kind = "incomplete"
    pairs = _pairs(entries, tolerance)
    top1 = _top1(entries, tolerance) if kind == "evaluated" else None
    regret = _regret(entries, tolerance) if kind == "evaluated" else None
    return {
        "label": condition["label"],
        "kind": kind,
        "configs_total": len(entries),
        "configs_complete": len(complete),
        "missing_values": missing,
        "pairs": pairs,
        "top1": top1,
        "regret": regret,
    }


def _pairs(entries, tolerance):
    counts = dict.fromkeys(_PAIR_KEYS, 0)
    for first, second in combinations(entries, 2):
        counts["total"] += 1
        missing_predicted = first["predicted"] is None or second["predicted"] is None
        missing_measured = first["measured"] is None or second["measured"] is None
        if missing_predicted or missing_measured:
            counts["missing"] += 1
            if missing_predicted and missing_measured:
                counts["missing_both"] += 1
            elif missing_predicted:
                counts["missing_predicted"] += 1
            else:
                counts["missing_measured"] += 1
            continue
        predicted_sign = order(first["predicted"], second["predicted"], tolerance)
        measured_sign = order(first["measured"], second["measured"], tolerance)
        if predicted_sign == 0 and measured_sign == 0:
            counts["both_tie"] += 1
        elif predicted_sign == 0:
            counts["predicted_tie"] += 1
        elif measured_sign == 0:
            counts["measured_tie"] += 1
        elif predicted_sign == measured_sign:
            counts["concordant"] += 1
        else:
            counts["discordant"] += 1
        exact_predicted = (0 if first["predicted"] == second["predicted"]
                           else 1 if first["predicted"] > second["predicted"] else -1)
        exact_measured = (0 if first["measured"] == second["measured"]
                          else 1 if first["measured"] > second["measured"] else -1)
        if exact_predicted == 0 and exact_measured == 0:
            counts["exact_both_tie"] += 1
        elif exact_predicted == 0:
            counts["exact_predicted_tie"] += 1
        elif exact_measured == 0:
            counts["exact_measured_tie"] += 1
        elif exact_predicted == exact_measured:
            counts["exact_concordant"] += 1
        else:
            counts["exact_discordant"] += 1
    return _finish_pairs(counts)


def _finish_pairs(counts):
    counts = dict(counts)
    counts["comparable"] = counts["concordant"] + counts["discordant"]
    present = sum(counts[key] for key in _PRESENT)
    counts["present_pairs"] = present
    counts["agreement"] = (counts["concordant"] / counts["comparable"]
                           if counts["comparable"] else None)
    counts["kendall_tau_b"] = _tau_b(
        counts["concordant"], counts["discordant"],
        counts["predicted_tie"] + counts["both_tie"],
        counts["measured_tie"] + counts["both_tie"], present)
    counts["kendall_tau_b_exact"] = _tau_b(
        counts["exact_concordant"], counts["exact_discordant"],
        counts["exact_predicted_tie"] + counts["exact_both_tie"],
        counts["exact_measured_tie"] + counts["exact_both_tie"], present)
    return counts


def _tau_b(concordant, discordant, predicted_ties, measured_ties, present):
    denominator = math.sqrt((present - predicted_ties) * (present - measured_ties))
    return None if denominator == 0.0 else (concordant - discordant) / denominator


def _sum_pairs(rows):
    totals = {key: sum(row["pairs"][key] for row in rows) for key in _PAIR_KEYS}
    return _finish_pairs(totals)


def _unique_best(entries, side, tolerance):
    best = min(entries, key=lambda entry: entry[side])
    tied = any(entry is not best and order(best[side], entry[side], tolerance) == 0
               for entry in entries)
    return (None if tied else best["name"]), tied


def _top1(entries, tolerance):
    predicted_best, predicted_tie = _unique_best(entries, "predicted", tolerance)
    measured_best, measured_tie = _unique_best(entries, "measured", tolerance)
    if predicted_tie:
        outcome = "predicted_tie"
    elif measured_tie:
        outcome = "measured_tie"
    elif predicted_best == measured_best:
        outcome = "hit"
    else:
        outcome = "miss"
    return {
        "predicted_best": predicted_best,
        "measured_best": measured_best,
        "outcome": outcome,
    }


def _regret(entries, tolerance):
    selected, tied = _unique_best(entries, "predicted", tolerance)
    if tied:
        return None
    measured_best = min(entry["measured"] for entry in entries)
    selected_value = next(entry["measured"] for entry in entries if entry["name"] == selected)
    best_name = next(entry["name"] for entry in entries if entry["measured"] == measured_best)
    loss = max(0.0, (selected_value - measured_best) / measured_best)
    return {"selected": selected, "measured_best": best_name, "loss": loss}


def _summarize_top1(rows):
    details = []
    counts = {"hit": 0, "miss": 0, "predicted_tie": 0, "measured_tie": 0}
    for row in rows:
        if row["top1"] is None:
            continue
        details.append(dict(row["top1"], condition=row["label"]))
        counts[row["top1"]["outcome"]] += 1
    evaluated = counts["hit"] + counts["miss"]
    return {
        "evaluated": evaluated,
        "hits": counts["hit"],
        "misses": counts["miss"],
        "hit_rate": counts["hit"] / evaluated if evaluated else None,
        "predicted_tie": counts["predicted_tie"],
        "measured_tie": counts["measured_tie"],
        "excluded": {
            "single_config_conditions": [row["label"] for row in rows if row["kind"] == "single_config"],
            "incomplete_conditions": [row["label"] for row in rows if row["kind"] == "incomplete"],
            "empty_conditions": [row["label"] for row in rows if row["kind"] == "empty"],
        },
        "details": details,
    }


def _summarize_regret(rows):
    details = []
    losses = []
    predicted_ties = []
    for row in rows:
        if row["kind"] != "evaluated":
            continue
        if row["regret"] is None:
            predicted_ties.append(row["label"])
            continue
        details.append(dict(row["regret"], condition=row["label"]))
        losses.append(row["regret"]["loss"])
    return {
        "evaluated": len(losses),
        "mean": statistics.mean(losses) if losses else None,
        "median": statistics.median(losses) if losses else None,
        "max": max(losses) if losses else None,
        "predicted_tie_conditions": predicted_ties,
        "details": details,
    }
