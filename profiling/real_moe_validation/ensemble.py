"""Read cohorts and combine them into the number the device protocol defines.

The unit of a campaign is one cohort; the unit of a paper cell is an ensemble of
cohorts taken in one declared device state. This module is the reading half of
that: it takes no measurement, refits nothing, and never touches a cohort it is
asked to summarise.

The three things it does are the three things the protocol fixes in advance:
the per-request first-forward duration that the state occupancy is counted
from, the band boundary that separates the two modes, and the ensemble statistic
(mean of the cohort means with its between-run standard deviation, beside the
pooled per-condition phase APEs).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


class Unreadable(RuntimeError):
    """A cohort is missing the file that would have been summarised."""


def cohort_root(path: Path) -> Path:
    return Path(path)


def total_ape(value: Any) -> float:
    """The total APE a cohort states, whether as one number or one per phase.

    A cohort's evaluation states its mean APE per phase (total, prefill,
    decode_with_sampling); an older cohort states a single number.  The
    ensemble's headline number is the total, so both shapes are read here
    rather than at each call site.
    """
    if isinstance(value, dict):
        value = value.get("total")
    if value is None:
        raise Unreadable("a cohort states no total APE")
    return float(value)


def evaluation(path: Path) -> dict[str, Any]:
    target = Path(path) / "model" / "formal" / "CONDITION_MEAN_EVALUATION.json"
    if not target.is_file():
        raise Unreadable(f"{path} has no {target.name}; the cohort did not reach evaluation")
    return json.loads(target.read_text(encoding="utf-8"))


def first_forward_ms(path: Path) -> list[dict[str, Any]]:
    """Every bare held-out request's prefill first forward, in milliseconds.

    Ticks are 100 ns.  A request whose inventory has no prefill forward is left
    out and counted, not recorded as zero: zero would read as a very fast
    forward and move the occupancy in the direction of the state being absent.
    """
    path = Path(path)
    out: list[dict[str, Any]] = []
    missing = 0
    for results in sorted((path / "model" / "formal" / "jobs").glob("*-heldout-r*/result.json")):
        payload = json.loads(results.read_text(encoding="utf-8"))
        job = results.parent.name
        for result in payload.get("results", []):
            inventory = result.get("forward_inventory") or []
            prefill = [item for item in inventory if item.get("phase") == "prefill"]
            if not prefill:
                missing += 1
                continue
            first = prefill[0]
            out.append({"job": job, "request_id": result.get("id"),
                        "ms": (first["end_tick"] - first["begin_tick"]) * 1e-4})
    if missing:
        out.append({"job": None, "request_id": None, "ms": None, "missing": missing})
    return out


def boundary(durations_ms: list[float]) -> tuple[float, str]:
    """The band boundary: midpoint of the largest ratio gap in the sorted values.

    Fixed once per machine from that machine's first cohort and never moved
    afterwards (device protocol section 3.4).  The rule is written down rather
    than tuned: the largest ratio gap is where the two modes are furthest apart,
    and its midpoint is the value furthest from both under a ratio reading.
    """
    ordered = sorted(value for value in durations_ms if value is not None)
    if len(ordered) < 3:
        raise Unreadable("at least three durations are needed to declare a boundary")
    best = None
    for left, right in zip(ordered, ordered[1:]):
        if left <= 0:
            continue
        ratio = right / left
        if best is None or ratio > best[0]:
            best = (ratio, left, right)
    ratio, left, right = best
    return (left + right) / 2.0, "largest ratio gap %.2f between %.2f and %.2f ms" % (
        ratio, left, right)


def occupancy(durations_ms: list[float], boundary_ms: float) -> dict[str, Any]:
    values = [value for value in durations_ms if value is not None]
    slow = [value for value in values if value > boundary_ms]
    return {"boundary_ms": boundary_ms, "requests": len(values), "slow": len(slow),
            "fraction": (len(slow) / len(values)) if values else None}


def aggregate(entries: list[dict[str, Any]], attempts: int,
              declared_boundary_ms: float | None = None) -> dict[str, Any]:
    """The ensemble of the cohorts that were counted; excluded ones are only counted.

    The mean of the cohort means is the paper number, with the between-run
    standard deviation beside it: a single cohort is one of the N and is never
    reported alone.  The pooled per-condition phase APEs are reported because
    the pre-registration's rule 2' is stated on that pooled distribution and a
    mean of condition means can hide a heavy tail.
    """
    counted = [entry for entry in entries if entry.get("status") == "counted"]
    if not counted:
        raise Unreadable("no cohort was counted; there is nothing to combine")
    means = [total_ape(entry["mean_APE"]) for entry in counted]
    ensemble: dict[str, Any] = {
        "cohorts": len(counted), "attempts": attempts,
        "excluded": len(entries) - len(counted),
        "mean_of_means": statistics.fmean(means),
        "sd_of_means": statistics.stdev(means) if len(means) > 1 else 0.0,
        "min_of_means": min(means), "max_of_means": max(means),
        "pass_counts": [entry["passed"] for entry in counted],
        "mean_worst_APE": statistics.fmean(total_ape(entry["worst_APE"]) for entry in counted),
    }
    phase_values = [value for entry in counted for condition in entry.get("phase_APEs", [])
                    for value in condition["APE"].values() if value is not None]
    if phase_values:
        ordered = sorted(phase_values)
        index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        ensemble["phase_APE"] = {"n": len(ordered), "mean": statistics.fmean(ordered),
                                 "median": statistics.median(ordered),
                                 "p90": ordered[index], "max": ordered[-1]}
    durations = [row["ms"] for entry in counted for row in entry.get("first_forward_ms", [])]
    if durations:
        frozen, why = ((declared_boundary_ms, "declared in the profile")
                       if declared_boundary_ms is not None else boundary(durations))
        ensemble["state"] = occupancy(durations, frozen)
        ensemble["state"]["source"] = why
    return ensemble
