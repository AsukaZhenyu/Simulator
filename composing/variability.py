"""How far one wall clock moved, and whether it moved between two places.

A price is a point estimate, but the clock that produced it is a distribution,
and this machine's distribution is not always a single one.  Measured on
2026-11-05 with the probe that built the dense tables, the same binary and the
same command gave a single mode at 1092 us on 2026-11-04 and two modes today:
about 1085 us and about 1270 us, with the two interleaved every few
milliseconds and the mix changing from second to second.  Frequency was
identical in both (4.99 vs 5.00 cycles/ns), so nothing was clocked down; a
thread sharing the SMT sibling that a probe thread runs on reproduced the slow
mode exactly, and a thread on another CCD reproduced nothing.  The cause is
therefore a *state*, not a drift, and a number measured in one state cannot be
compared with a number measured in the other without saying so.

This module only describes.  It does not decide which mode is right, does not
discard the samples that disagree, and does not calibrate anything.  What it
produces is the band a comparison has to be read inside:

* a **summary** of any list of wall times -- what a row's repetitions span;
* a **two-mode split** of a long run -- where the two centres are, how much of
  the run sits in each, and whether calling it two modes is even justified;
* a **session** record -- which rows of one process were wider than the band;
* a **table state** -- whether the table being predicted from was priced in the
  state the wall clock is in now.

The distinction matters for the paper: an error inside the session band is not
evidence about the cost model, and an error outside it is not explained by the
machine.  Keeping them as two columns is the whole point; a single "within x%"
number would silently merge the two.
"""

from __future__ import annotations

import math

SCHEMA_VERSION = "composing_variability_v1"

# A row's own repetitions are only a handful, so the band they describe is the
# observed spread, not a confidence interval.  15% is the state flip measured
# directly on this machine (MoE report 6.1: 49.89/49.39/50.36/49.41 vs 42.33 ms)
# and is the same value ``decompose`` refuses to attribute further.
DEFAULT_BAND = 0.15

# Below this many samples a two-mode split is fitting noise: five llama-bench
# repetitions can say how far apart they are, never that they came from two
# processes.  The probe's runs carry thousands, which is where modes are read.
MODE_MIN_SAMPLES = 200
# Two centres closer than this are one mode with a wide skirt.
MODE_SEPARATION = 0.02
# A mode holding less than this share is a tail, not a state.  The 2026-11-04
# archive has 5.2% of its samples above 1100 us and k-means will call that a
# second centre at 1136 us; a process that spends a twentieth of a run
# somewhere is a hiccup, not a state a measurement was taken in.
MODE_FLOOR = 0.10
# A state is narrow.  Without this, k-means splits a single population's tail
# into a "second centre" -- the 2026-11-04 archive (one mode at 1092 us with a
# 29.7% tail) came out as 1091.9 / 1136.0, a 4% split that means nothing.
MODE_WIDTH = 0.05


def _positive(samples, *, where: str) -> list[float]:
    values = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{where}: wall clock samples must be positive numbers, "
                             f"got {value!r}")
        values.append(float(value))
    if not values:
        raise ValueError(f"{where}: no samples")
    return values


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _cv(values: list[float]) -> float | None:
    mean = sum(values) / len(values)
    if not mean:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean


def summarise(samples_ns, *, where: str = "samples") -> dict:
    """What one row's wall times span.  Never a decision, only a description."""
    values = _positive(samples_ns, where=where)
    ordered = sorted(values)
    return {"n": len(ordered),
            "min_ns": ordered[0],
            "p10_ns": _percentile(ordered, 0.10),
            "median_ns": _percentile(ordered, 0.50),
            "p90_ns": _percentile(ordered, 0.90),
            "max_ns": ordered[-1],
            "cv": _cv(values),
            # How far the slowest sample sits from the fastest.  On a row with
            # five repetitions this is the only band the row itself can claim.
            "spread": ordered[-1] / ordered[0] - 1}


def split_modes(samples_ns, *, min_samples: int = MODE_MIN_SAMPLES,
                separation: float = MODE_SEPARATION,
                floor: float = MODE_FLOOR, width: float = MODE_WIDTH) -> dict:
    """Split a long run into two centres, or say why it must not be split.

    The split is two-means in log time, so a 16% ratio is 0.15 in the space the
    centres live in and the two are found symmetrically.  ``single_mode`` is
    True whenever calling it two modes would be an over-read: too few samples,
    centres closer than ``separation``, or one side holding less than ``floor``.
    """
    values = _positive(samples_ns, where="mode split")
    n = len(values)
    if n < min_samples:
        return {"n": n, "single_mode": None, "reason": f"needs >= {min_samples} samples to "
                                                       f"call a mode, has {n}"}
    logs = sorted(math.log(value) for value in values)
    low, high = logs[n // 4], logs[(3 * n) // 4]
    for _ in range(200):
        left = [x for x in logs if abs(x - low) <= abs(x - high)]
        right = [x for x in logs if abs(x - low) > abs(x - high)]
        if not left or not right:
            break
        moved_low = sum(left) / len(left)
        moved_high = sum(right) / len(right)
        if abs(moved_low - low) < 1e-15 and abs(moved_high - high) < 1e-15:
            low, high = moved_low, moved_high
            break
        low, high = moved_low, moved_high
    left = [value for value in values if math.log(value) <= (low + high) / 2]
    right = [value for value in values if math.log(value) > (low + high) / 2]
    if not left or not right:
        return {"n": n, "single_mode": True, "reason": "the run does not separate"}
    fast_ns, slow_ns = math.exp(low), math.exp(high)
    ratio = slow_ns / fast_ns
    slow_fraction = len(right) / n
    reasons = []
    if ratio < 1 + separation:
        reasons.append(f"centres are {100 * (ratio - 1):.2f}% apart")
    if min(slow_fraction, 1 - slow_fraction) < floor:
        reasons.append(f"the smaller mode holds {100 * min(slow_fraction, 1 - slow_fraction):.1f}%")
    fast_cv, slow_cv = _cv(left), _cv(right)
    if fast_cv is None or slow_cv is None or max(fast_cv, slow_cv) > width:
        widest = max(value for value in (fast_cv, slow_cv) if value is not None)
        reasons.append(f"the wide side has cv {100 * widest:.1f}%, a tail not a centre")
    return {"n": n,
            "fast_ns": fast_ns, "slow_ns": slow_ns,
            "band_ns": [fast_ns, slow_ns],
            "ratio": ratio,
            "slow_fraction": slow_fraction,
            "fast_cv": fast_cv, "slow_cv": slow_cv,
            # Having no reason to call it one distribution is exactly the case
            # where the run has two centres.
            "single_mode": bool(reasons),
            "reason": "; ".join(reasons) if reasons else "two centres"}


def describe(samples_ns, *, where: str = "samples") -> dict:
    """Both descriptions of one run: the span of it, and its modes if it has any."""
    return {"schema_version": SCHEMA_VERSION,
            "summary": summarise(samples_ns, where=where),
            "modes": split_modes(samples_ns)}


def session_summary(observations: dict, *, band: float = DEFAULT_BAND) -> dict:
    """Which rows of one process were wider than the state band.

    ``observations`` maps a label such as ``qwen3-0.6b/tg512`` to a
    ``summarise`` result.  The question answered is narrow on purpose: not
    whether the numbers are right, but whether they were taken in one state.
    """
    if not observations:
        return {"band": band, "rows": 0, "widest_spread": None, "outside_band": [],
                "median_cv": None}
    spreads = {label: row["spread"] for label, row in observations.items()}
    cvs = [row["cv"] for row in observations.values() if row.get("cv") is not None]
    ordered_cv = sorted(cvs)
    return {"band": band,
            "rows": len(observations),
            "widest_spread": max(spreads.values()),
            "widest_row": max(spreads, key=spreads.get),
            "outside_band": sorted(label for label, spread in spreads.items()
                                   if spread > band),
            "median_cv": _percentile(ordered_cv, 0.5) if ordered_cv else None}


def wall_ns_of(document: dict, *, where: str = "document") -> list[int]:
    """The samples of a probe result, whether archived or straight from the probe.

    ``profiling`` archives a run as ``{"document": {...}}`` with the round and
    the command beside it; the probe itself prints the document.  Both carry
    ``wall_ns``, and a reader here must not care which one it was handed.
    """
    inner = document.get("document") if isinstance(document.get("document"), dict) else document
    wall = inner.get("wall_ns")
    if not isinstance(wall, list) or not wall:
        raise ValueError(f"{where}: no wall_ns samples")
    return wall


# A table is priced by the probe in one machine state.  Months of wall clock are
# taken afterwards, in whatever state the machine is in then.  The two are one
# comparison only if the state has not moved, and on 2026-11-05 it had: with the
# same probe binary, the same bench binary and the same echoed flags, five of the
# six tables' models ran 8.7-10.6% faster in the later session, while the sixth
# (priced after the step) did not move at all.
#
# 5% sits between the two numbers that bound the decision.  The probe's own
# reading moved less than the engine's (0.9224 against 0.9023), and the
# composition ratio between two sessions drifted about 2.4 points by itself; the
# step is ~9-14%.  So 5% does not fire on the drift and does fire on the step.
DEFAULT_TABLE_TOLERANCE = 0.05


def block_records(document: dict, *, where: str = "document") -> list[dict]:
    """Every joint ``block`` record a probe result file holds, whichever shape it has.

    Three layouts are in the archive and all three are the same claim: the probe's
    own printout (the record at the top level), ``profiling``'s archive of one run
    (``{"document": ...}``) and a campaign's raw file (``{"record": ...}`` per
    labelled run, sometimes gathered into a map).  A reader that wants the block
    price must not care which one it was handed; one that wants a stage price
    gets a refusal instead of a number from a different stage.
    """
    if not isinstance(document, dict):
        raise ValueError(f"{where}: not a probe document")
    for key in ("record", "document"):
        inner = document.get(key)
        if isinstance(inner, dict):
            return block_records(inner, where=where)
    if "wall_ns" in document:
        stage = document.get("stage")
        if stage != "block":
            raise ValueError(f"{where}: this is a {stage!r} record, not the joint block")
        sample = summarise(document["wall_ns"], where=where)
        return [{"stage": stage, "q": int(document["q"]), "history": int(document["history"]),
                 "batch": int(document.get("batch", 1)),
                 "median_ns": sample["median_ns"], "cv": sample["cv"], "n": sample["n"]}]
    nested = [value for value in document.values() if isinstance(value, dict)]
    if nested:
        # A campaign raw file holds every stage and both phases it ran.  The
        # block records are the ones this reader wants; the rest are other
        # questions, not errors.
        found = []
        for value in nested:
            try:
                found.extend(block_records(value, where=where))
            except ValueError:
                continue
        if not found:
            raise ValueError(f"{where}: no block record in this document")
        return found
    raise ValueError(f"{where}: no block record in this document")


def _positive_ns(values: dict, *, where: str) -> dict:
    for shape, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{where}/{shape}: not a wall time: {value!r}")
    return values


def table_state(reference: dict, sensor: dict, *,
                tolerance: float = DEFAULT_TABLE_TOLERANCE) -> dict:
    """Was this table priced in the state the clock is in now?

    ``reference`` maps a shape, such as ``prefill/B1/q256/hist0``, to the table's
    own joint ``block`` median -- the one record the exporter keeps as its
    composition reference and never sums.  ``sensor`` maps the same shapes to
    the same record measured now, with the same probe.  The ratio is the state
    of the machine, not a property of the model.

    The ratio is deliberately **not** a correction.  Rescaling prices by it
    would be fitting a state with one number, and the probe and the engine do
    not move by the same amount: measured on 2026-11-05, the probe's block fell
    to 0.9224 of its old value while the wall clock fell to 0.9023.  What this
    answers is whether a comparison is admissible at all.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    _positive_ns(reference, where="reference")
    _positive_ns(sensor, where="sensor")
    common = sorted(set(reference) & set(sensor))
    if not common:
        raise ValueError("no shape was measured both ways: the table prices "
                         f"{sorted(reference)}, the sensor {sorted(sensor)}")
    rows = []
    for shape in common:
        ratio = sensor[shape] / reference[shape]
        rows.append({"shape": shape, "table_ns": reference[shape], "sensor_ns": sensor[shape],
                     "ratio": ratio, "within_band": abs(ratio - 1.0) <= tolerance})
    ratios = sorted(row["ratio"] for row in rows)
    widest = max(rows, key=lambda row: abs(row["ratio"] - 1.0))
    outside = [row["shape"] for row in rows if not row["within_band"]]
    return {"schema_version": SCHEMA_VERSION,
            "tolerance": tolerance,
            "rows": rows,
            "median_ratio": _percentile(ratios, 0.5),
            "widest_shape": widest["shape"],
            "widest_ratio": widest["ratio"],
            "uncompared": {"table_only": sorted(set(reference) - set(sensor)),
                           "sensor_only": sorted(set(sensor) - set(reference))},
            "verdict": "different_band" if outside else "same_band",
            "outside_band": outside}

