"""Split an end-to-end decode error into a history-independent and a per-KV-token term.

The unit of the claim stays the action: a strategy compiles to (op, placement,
shape, contract) -> one exact price, and the sum of those prices is compared with
llama.cpp's wall clock.  Decode latency is very nearly affine in the KV history,
so the error can be attributed to two named terms:

    measured(h) = intercept + slope * h

* intercept -- the work every token does regardless of history: boundary actions,
  projection and feed-forward of every block, and the fixed part of attention;
* slope -- what one more visible KV token costs, model-wide.

Neither term is allowed to be "fixed" here.  The only thing this module may do is
refuse to attribute an error that a per-process state factor already explains, so
a slope ratio is reported both raw and after the intercept's factor.
"""

from __future__ import annotations

import math

DECODE_ORDER = ("tg128", "tg512", "tg1024")
HISTORY = {"tg128": 128, "tg512": 512, "tg1024": 1024}
# How far one process sits from another on this machine, measured directly as
# the fast/slow flip inside a single llama-bench row (MoE report §6.1:
# 49.89/49.39/50.36/49.41 vs 42.33 ms).  A row that would have to sit further
# out than this is not explained by process state, so the fit is flagged.
STATE_BAND = 0.15
# A row whose own repetitions spread wider than this is not a point estimate.
MAX_CV = 0.15


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Unweighted least squares; every row came from the same process."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("need at least two distinct history lengths")
    slope = sxy / sxx
    return my - slope * mx, slope


def _stage_ns(entry: dict, shape: str, stage: str) -> int:
    value = entry["shapes"][shape][stage]
    return value["median_ns"] if isinstance(value, dict) else value


def _declared_terms_ms(entry: dict, shapes) -> dict:
    """What the declared terms contributed to each shape, if any were enabled."""
    contributions = {}
    for name in shapes:
        breakdown = entry["shapes"][name].get("terms_ns") or []
        if breakdown:
            contributions[name] = sum(item["contribution_ns"] for item in breakdown) / 1e6
    return contributions


def decompose(prediction: dict, comparison: dict,
              shapes: list[str] | None = None, state_band: float = STATE_BAND,
              max_cv: float = MAX_CV) -> dict:
    """One record per measured model; unmeasured models are named, not guessed."""
    wanted = list(shapes or DECODE_ORDER)
    report = {"schema_version": "dense_endtoend_decomposition_v1",
              "unit": "decode_ms_vs_visible_history",
              "state_band": state_band, "max_cv": max_cv, "models": {}, "not_measured": []}
    for model_id, entry in prediction["models"].items():
        observed = (comparison.get("models") or {}).get(model_id)
        if not observed:
            report["not_measured"].append(model_id)
            continue
        usable = [name for name in wanted if name in entry["shapes"] and name in observed]
        if len(usable) < 2:
            report["not_measured"].append(model_id)
            continue
        # A row that moves more than max_cv inside its own repetitions is not a
        # point estimate of anything: whatever the model says, one number cannot
        # stand for a spread that wide.
        unsteady = {name: observed[name]["observed_cv"] for name in usable
                    if (observed[name].get("observed_cv") or 0) > max_cv}
        usable = [name for name in usable if name not in unsteady]
        if len(usable) < 2:
            report["not_measured"].append(model_id)
            report["models"][model_id] = {"unsteady_shapes": unsteady}
            continue
        hs = {name: HISTORY[name] for name in usable}
        predicted = {name: entry["shapes"][name]["total_ns"] / 1e6 for name in usable}
        measured = {name: observed[name]["observed_ns"] / 1e6 for name in usable}
        # A deeper history that measured faster is not a slope: one row ran in
        # another state, and the effect is per process, not per stage.  Which row
        # is not decidable from a single number, so every candidate is scored by
        # how far the dropped row would have to sit from the line its neighbours
        # give it; the least such deviation wins and the rest stay on record.
        kept, dropped, candidates, ambiguous = usable, None, [], False
        if any(measured[b] < measured[a] for a, b in zip(usable, usable[1:])):
            scored = []
            for name in usable:
                rest = [other for other in usable if other != name]
                if len(rest) < 2:
                    continue
                xs = [hs[other] for other in rest]
                ip, sp = _fit(xs, [predicted[other] for other in rest])
                im, sm = _fit(xs, [measured[other] for other in rest])
                implied = (im + sm * hs[name]) / measured[name]
                scored.append({"dropped": name, "implied_state_factor": implied,
                               "deviation": abs(math.log(implied)),
                               "within_measured_state_band": abs(math.log(implied)) <= state_band,
                               # The line that would be reported if this row were
                               # the bad one.  Dropping the middle of three points
                               # always fits the other two exactly, so the sole
                               # criterion above can prefer a reading whose slope
                               # is nothing like the neighbours' -- the slopes have
                               # to be shown next to the deviation, not instead of
                               # it.
                               "slope_us": sm * 1e3, "intercept_ms": im})
            scored.sort(key=lambda item: item["deviation"])
            candidates = scored
            best = scored[0]
            dropped = best["dropped"]
            kept = [name for name in usable if name != dropped]
            ambiguous = best["deviation"] > state_band
        admissible = [item for item in candidates if item["within_measured_state_band"]]
        xs = [hs[name] for name in kept]
        intercept_p, slope_p = _fit(xs, [predicted[name] for name in kept])
        intercept_m, slope_m = _fit(xs, [measured[name] for name in kept])
        layers = entry["layers"]
        attention = [_stage_ns(entry, name, "attention") * layers / 1e6 for name in kept]
        _, slope_attention = _fit(xs, attention)
        state = intercept_m / intercept_p
        deepest = kept[-1]
        deepest_error = measured[deepest] - predicted[deepest]
        # A share of an error that is zero is not a number; say so instead of
        # printing a fabricated 0%.
        share = ((slope_m - slope_p) * hs[deepest] / deepest_error
                 if deepest_error else None)
        # Only attention reads the history.  The projection and the feed-forward
        # are priced once per depth although their work does not change, so the
        # table's own slope carries that difference; the attention-only slope is
        # the part that is supposed to vary with depth at all.
        attention_share = ((slope_m - slope_attention) * hs[deepest] / deepest_error
                           if deepest_error else None)
        report["models"][model_id] = {
            "shapes_used": kept, "shape_excluded_unphysical": dropped,
            "unsteady_shapes": unsteady,
            "exclusion_candidates": candidates, "ambiguous": ambiguous,
            # Every reading the data cannot rule out, with the slope each one
            # would report.  More than one means the slope of this run is a
            # choice, not a measurement.
            "admissible_readings": [{"dropped": item["dropped"], "slope_us": item["slope_us"]}
                                    for item in admissible],
            "admissible_slope_range_us": ((max(item["slope_us"] for item in admissible)
                                           - min(item["slope_us"] for item in admissible))
                                          if len(admissible) > 1 else None),
            "layers": layers,
            # The declared terms' share of each predicted point, when any were
            # enabled.  Empty means the prediction priced the table alone.
            "declared_terms_ms": _declared_terms_ms(entry, kept),
            "intercept_predicted_ms": intercept_p, "intercept_measured_ms": intercept_m,
            "intercept_ratio": state,
            "slope_predicted_us": slope_p * 1e3, "slope_measured_us": slope_m * 1e3,
            "slope_ratio": slope_m / slope_p,
            "slope_ratio_after_state_factor": slope_m / (slope_p * state),
            "attention_only_table_slope_us": slope_attention * 1e3,
            "history_invariant_table_slope_us": (slope_p - slope_attention) * 1e3,
            # The like-for-like ratio: the only part of the table's slope whose
            # work really changes with depth, against what the wall clock moved.
            "slope_ratio_attention_only": slope_m / slope_attention,
            "slope_ratio_attention_only_after_state_factor": slope_m / (slope_attention * state),
            "slope_share_of_deepest_error": share,
            "attention_only_share_of_deepest_error": attention_share,
            "fit_residual_predicted": max(abs(y - (intercept_p + slope_p * x))
                                          for x, y in zip(xs, [predicted[n] for n in kept]))
                                       / predicted[deepest],
            "fit_residual_measured": max(abs(y - (intercept_m + slope_m * x))
                                         for x, y in zip(xs, [measured[n] for n in kept]))
                                      / measured[deepest],
        }
    return report


def show(report: dict) -> None:
    for model_id, row in report["models"].items():
        if "shapes_used" not in row:
            # Every measured row moved more inside its own repetitions than a
            # point estimate may; there is no line to fit and none is invented.
            unsteady = row.get("unsteady_shapes") or {}
            print(f"\n{model_id}: no slope -- every measured shape is unsteady "
                  f"within itself ("
                  + ", ".join(f"{name} cv {cv:.2f}" for name, cv in unsteady.items()) + ")")
            continue
        used = ",".join(row["shapes_used"])
        dropped = row["shape_excluded_unphysical"]
        unsteady = row.get("unsteady_shapes") or {}
        print(f"\n{model_id}  [{used}]" + (f"  dropped(not a slope): {dropped}" if dropped else ""))
        if unsteady:
            print("  excluded(unsteady within itself): "
                  + ", ".join(f"{name} cv {cv:.2f}" for name, cv in unsteady.items()))
        print(f"  intercept  predicted {row['intercept_predicted_ms']:.3f} ms  "
              f"measured {row['intercept_measured_ms']:.3f} ms  ratio {row['intercept_ratio']:.3f}")
        print(f"  slope      predicted {row['slope_predicted_us']:.3f} us/KVtok  "
              f"(attention-only {row['attention_only_table_slope_us']:.3f}, "
              f"not reading history {row['history_invariant_table_slope_us']:+.3f})")
        print(f"             measured {row['slope_measured_us']:.3f} us/KVtok  "
              f"ratio {row['slope_ratio']:.2f}x  like-for-like "
              f"{row['slope_ratio_attention_only']:.2f}x  (attention-only)")
        print(f"             after the intercept's state factor "
              f"{row['slope_ratio_after_state_factor']:.2f}x, like-for-like "
              f"{row['slope_ratio_attention_only_after_state_factor']:.2f}x")
        share_text = (f"{row['slope_share_of_deepest_error']:.0%}"
                      if row["slope_share_of_deepest_error"] is not None else "n/a (no error)")
        attention_share_text = (f"{row['attention_only_share_of_deepest_error']:.0%}"
                                if row["attention_only_share_of_deepest_error"] is not None
                                else "n/a (no error)")
        print(f"  slope share of the deepest error {share_text} "
              f"(attention-only {attention_share_text}); "
              f"fit residual predicted {row['fit_residual_predicted']:.1%} "
              f"measured {row['fit_residual_measured']:.1%}"
              + ("   AMBIGUOUS: no exclusion stays inside the state band"
                 if row["ambiguous"] else ""))
        for candidate in row["exclusion_candidates"]:
            mark = "  <-- chosen" if candidate["dropped"] == dropped else ""
            inside = "inside" if candidate["within_measured_state_band"] else "outside"
            print(f"    if {candidate['dropped']} were the bad row, the line through the others "
                  f"would put it at {candidate['implied_state_factor']:.2f}x what it measured "
                  f"({inside} the {report['state_band']:.0%} state band), "
                  f"slope {candidate['slope_us']:.3f} us/KVtok{mark}")
        if len(row["admissible_readings"]) > 1:
            # Saying which row is bad is a choice here, and the slope is whatever
            # the choice implies; quoting the chosen one alone would report a
            # number the data never singled out.
            print(f"  NOT DETERMINED: {len(row['admissible_readings'])} readings stay inside the "
                  f"state band and their slopes differ by {row['admissible_slope_range_us']:.3f} "
                  f"us/KVtok ("
                  + ", ".join(f"drop {item['dropped']} -> {item['slope_us']:.3f}"
                              for item in row["admissible_readings"]) + ")")
    for model_id in report["not_measured"]:
        print(f"\n{model_id}: prediction only (no measurement)")
