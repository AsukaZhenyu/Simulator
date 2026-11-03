"""P1: is a predicted ranking crossover visible in the measurement?

Conditions are analysed in the order they appear in the input. For every pair of
configurations present in two adjacent conditions, the predicted ordering is
scanned for a flip; each predicted flip is then checked against the measured
ordering of the same two configurations.

The decision rule, with ``tolerance`` deciding ties on both sides:

* ``CONFIRMED`` - the measured ordering flips in the predicted interval
  (``reason_code = "same_interval"``), or in one of its two adjacent intervals
  (``reason_code = "adjacent_interval"``, a sensitivity-only result). The two
  counts are reported separately; only ``same_interval`` is the main claim.
* ``NOT_CONFIRMED`` - the measured ordering flips only *outside* the predicted
  and adjacent intervals (``reason_code = "other_interval"``).
* ``INCONCLUSIVE`` - a value needed for the decision is missing
  (``missing_values``), the measured ordering is tied within tolerance
  (``tied_measurement``), no measured flip occurs anywhere in the sampled range
  (``coverage_unresolved``: the sweep may not cover the crossover, so it is
  neither confirmation nor contradiction), or no predicted crossover exists at
  all (``no_predicted_crossover``).

The overall verdict is conservative: any ``NOT_CONFIRMED`` event makes the whole
result ``NOT_CONFIRMED``; otherwise any ``INCONCLUSIVE`` event makes it
``INCONCLUSIVE``; it is ``CONFIRMED`` only if every predicted crossover is.
Nothing is discarded silently: missing configuration/value pairs are reported
in ``skipped_pairs``/``measured_skipped_pairs``, and crossovers that reach a tie
at one endpoint are reported in ``ambiguous_crossovers``.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from verdict.schema import order, relative_gap, validate_tolerance

CONFIRMED = "CONFIRMED"
NOT_CONFIRMED = "NOT_CONFIRMED"
INCONCLUSIVE = "INCONCLUSIVE"


def analyze(document, tolerance=0.02):
    """Return the P1 crossover verdict for a validated input document."""
    tolerance = validate_tolerance(tolerance)
    conditions = document["conditions"]
    maps = [{config["name"]: config for config in condition["configs"]} for condition in conditions]
    predicted, ambiguous, skipped, considered = _scan(
        conditions, maps, "predicted_latency_ns", tolerance)
    measured, _measured_ambiguous, measured_skipped, _measured_considered = _scan(
        conditions, maps, "measured_latency_ns", tolerance)
    measured_indices = defaultdict(list)
    for flip in measured:
        measured_indices[tuple(sorted(flip["config_pair"]))].append(flip["interval"]["from_index"])
    events = [_judge(event, conditions, maps, measured_indices, tolerance) for event in predicted]
    verdict, reason, reason_code = _overall(events, ambiguous)
    confirmed_same = sum(1 for event in events if event["reason_code"] == "same_interval")
    confirmed_adjacent = sum(1 for event in events if event["reason_code"] == "adjacent_interval")
    return {
        "tolerance": tolerance,
        "conditions": [
            {"index": index, "label": condition["label"],
             "value": condition.get("value"), "unit": condition.get("unit")}
            for index, condition in enumerate(conditions)
        ],
        "events": events,
        "predicted_crossovers": [_compact(event) for event in predicted],
        "measured_crossovers": [_compact(flip) for flip in measured],
        "ambiguous_crossovers": ambiguous,
        "skipped_pairs": skipped,
        "measured_skipped_pairs": measured_skipped,
        "summary": {
            "conditions": len(conditions),
            "intervals": max(0, len(conditions) - 1),
            "pairs_considered": considered,
            "skipped_pairs": len(skipped),
            "predicted_crossovers": len(predicted),
            "ambiguous_crossovers": len(ambiguous),
            "measured_crossovers": len(measured),
            "confirmed_same_interval": confirmed_same,
            "confirmed_adjacent_interval": confirmed_adjacent,
            "not_confirmed": sum(1 for event in events if event["verdict"] == NOT_CONFIRMED),
            "inconclusive": sum(1 for event in events if event["verdict"] == INCONCLUSIVE),
            "coverage_unresolved": sum(1 for event in events
                                       if event["reason_code"] == "coverage_unresolved"),
        },
        "verdict": verdict,
        "reason_code": reason_code,
        "reason": reason,
    }


def _interval(conditions, index):
    return {
        "from_index": index,
        "from_label": conditions[index]["label"],
        "to_index": index + 1,
        "to_label": conditions[index + 1]["label"],
    }


def _interval_text(interval):
    return "%r -> %r" % (interval["from_label"], interval["to_label"])


def _view(first, second, first_value, second_value, tolerance):
    sign = order(first_value, second_value, tolerance)
    faster = first if sign < 0 else second if sign > 0 else None
    return {
        "faster": faster,
        "gap": relative_gap(first_value, second_value),
        "values": {first: first_value, second: second_value},
    }


def _missing_value(maps, index, name, key):
    if name not in maps[index]:
        return "conditions[%d].configs[%s] (configuration not declared)" % (index, name)
    if maps[index][name].get(key) is None:
        return "conditions[%d].configs[%s].%s" % (index, name, key)
    return None


def _scan(conditions, maps, key, tolerance):
    """Find ordering flips of ``key`` between every adjacent condition pair."""
    flips, ambiguous, skipped = [], [], []
    considered = 0
    for index in range(len(conditions) - 1):
        interval = _interval(conditions, index)
        names = list(maps[index])
        names += [name for name in maps[index + 1] if name not in maps[index]]
        for first, second in combinations(names, 2):
            considered += 1
            missing = []
            values = {0: {}, 1: {}}
            for offset in (0, 1):
                for name in (first, second):
                    problem = _missing_value(maps, index + offset, name, key)
                    if problem is not None:
                        missing.append(problem)
                    else:
                        values[offset][name] = maps[index + offset][name][key]
            if missing:
                skipped.append({
                    "interval": interval,
                    "config_pair": [first, second],
                    "missing": missing,
                })
                continue
            left = order(values[0][first], values[0][second], tolerance)
            right = order(values[1][first], values[1][second], tolerance)
            if left == 0 and right == 0:
                continue
            from_view = _view(first, second, values[0][first], values[0][second], tolerance)
            to_view = _view(first, second, values[1][first], values[1][second], tolerance)
            if left == 0 or right == 0:
                ambiguous.append({
                    "interval": interval,
                    "config_pair": [first, second],
                    "from": from_view,
                    "to": to_view,
                    "reason": "one endpoint is tied within tolerance, so this is not a strict flip",
                })
                continue
            if left == right:
                continue
            flips.append({
                "interval": interval,
                "config_pair": [first, second],
                "from": from_view,
                "to": to_view,
            })
    return flips, ambiguous, skipped, considered


def _measured_state(maps, index, pair, key, tolerance):
    """State of the measured ordering for one interval: missing, tied or ordered."""
    missing = []
    values = {}
    for offset in (0, 1):
        for name in pair:
            problem = _missing_value(maps, index + offset, name, key)
            if problem is not None:
                missing.append(problem)
            else:
                values[(offset, name)] = maps[index + offset][name][key]
    if missing:
        return {"missing": missing, "tied": False}
    left = order(values[(0, pair[0])], values[(0, pair[1])], tolerance)
    right = order(values[(1, pair[0])], values[(1, pair[1])], tolerance)
    return {"missing": [], "tied": left == 0 or right == 0}


def _measured_views(maps, index, pair, tolerance):
    views = []
    for offset in (0, 1):
        if any(_missing_value(maps, index + offset, name, "measured_latency_ns") is not None
               for name in pair):
            views.append(None)
        else:
            first_value = maps[index + offset][pair[0]]["measured_latency_ns"]
            second_value = maps[index + offset][pair[1]]["measured_latency_ns"]
            views.append(_view(pair[0], pair[1], first_value, second_value, tolerance))
    return views


def _judge(event, conditions, maps, measured_indices, tolerance):
    """Attach the measured check and a verdict to one predicted crossover."""
    interval = event["interval"]
    index = interval["from_index"]
    pair = event["config_pair"]
    indices = measured_indices.get(tuple(sorted(pair)), [])
    measured_from, measured_to = _measured_views(maps, index, pair, tolerance)
    flip_interval = None

    if index in indices:
        relation = "same_interval"
        reason_code = "same_interval"
        verdict = CONFIRMED
        flip_interval = interval
        reason = "same_interval: measured ordering also flips in the predicted interval %s" % _interval_text(interval)
    else:
        adjacent = [candidate for candidate in (index - 1, index + 1) if candidate in indices]
        if adjacent:
            relation = "adjacent_interval"
            reason_code = "adjacent_interval"
            verdict = CONFIRMED
            flip_interval = _interval(conditions, adjacent[0])
            reason = ("adjacent_interval: measured ordering flips in adjacent interval %s, one "
                      "condition away from the predicted interval %s (sensitivity only)"
                      % (_interval_text(flip_interval), _interval_text(interval)))
        elif indices:
            relation = "other_interval"
            reason_code = "other_interval"
            verdict = NOT_CONFIRMED
            distant = ", ".join(_interval_text(_interval(conditions, candidate)) for candidate in indices)
            reason = ("other_interval: measured ordering flips only in non-adjacent interval(s) %s; "
                      "not confirmed in %s" % (distant, _interval_text(interval)))
        else:
            checked = [candidate for candidate in (index - 1, index, index + 1)
                       if 0 <= candidate < len(conditions) - 1]
            missing, tied = [], []
            for candidate in checked:
                state = _measured_state(maps, candidate, pair, "measured_latency_ns", tolerance)
                missing.extend(state["missing"])
                if state["tied"]:
                    tied.append(_interval(conditions, candidate))
            if missing:
                relation = "missing"
                reason_code = "missing_values"
                verdict = INCONCLUSIVE
                reason = ("missing_values: measured value missing, so the measured ordering cannot "
                          "be checked: " + "; ".join(dict.fromkeys(missing)))
            elif tied:
                relation = "tie"
                reason_code = "tied_measurement"
                verdict = INCONCLUSIVE
                reason = ("tied_measurement: measured ordering is tied within tolerance in %s, so a "
                          "flip cannot be distinguished"
                          % ", ".join(_interval_text(item) for item in tied))
            else:
                relation = "no_flip"
                reason_code = "coverage_unresolved"
                verdict = INCONCLUSIVE
                reason = ("coverage_unresolved: measured ordering does not flip anywhere in the "
                          "sampled range (predicted interval or either adjacent interval); the sweep "
                          "may not cover the crossover, so this is neither confirmation nor "
                          "contradiction")
    return {
        "interval": interval,
        "config_pair": pair,
        "predicted": {"from": event["from"], "to": event["to"]},
        "measured": {"from": measured_from, "to": measured_to},
        "relation": relation,
        "reason_code": reason_code,
        "measured_flip_interval": flip_interval,
        "verdict": verdict,
        "reason": reason,
    }


def _compact(event):
    return {
        "interval": event["interval"],
        "config_pair": event["config_pair"],
        "from_faster": event["from"]["faster"],
        "from_gap": event["from"]["gap"],
        "to_faster": event["to"]["faster"],
        "to_gap": event["to"]["gap"],
    }


def _overall(events, ambiguous):
    if not events:
        note = "no predicted ranking crossover was found between adjacent conditions"
        if ambiguous:
            note += ("; %d configuration pair(s) reached a within-tolerance tie at one endpoint, "
                     "which is not a strict crossover" % len(ambiguous))
        return INCONCLUSIVE, note + "; there is nothing for the measurement to confirm", "no_predicted_crossover"
    counts = {
        CONFIRMED: sum(1 for event in events if event["verdict"] == CONFIRMED),
        NOT_CONFIRMED: sum(1 for event in events if event["verdict"] == NOT_CONFIRMED),
        INCONCLUSIVE: sum(1 for event in events if event["verdict"] == INCONCLUSIVE),
    }
    same = sum(1 for event in events if event["reason_code"] == "same_interval")
    adjacent = sum(1 for event in events if event["reason_code"] == "adjacent_interval")
    coverage = sum(1 for event in events if event["reason_code"] == "coverage_unresolved")
    if counts[NOT_CONFIRMED]:
        return (NOT_CONFIRMED,
                "%d of %d predicted crossover(s) are contradicted: the measured ordering flips "
                "only outside the predicted and adjacent intervals"
                % (counts[NOT_CONFIRMED], len(events)),
                "not_confirmed")
    if counts[INCONCLUSIVE]:
        reason_code = ("coverage_unresolved" if coverage == counts[INCONCLUSIVE] else "inconclusive")
        return (INCONCLUSIVE,
                "%d of %d predicted crossover(s) lack usable measured evidence "
                "(coverage_unresolved: %d; missing/tied: %d)"
                % (counts[INCONCLUSIVE], len(events), coverage,
                   counts[INCONCLUSIVE] - coverage),
                reason_code)
    if same and adjacent:
        reason_code = "mixed_confirmed"
    elif same:
        reason_code = "same_interval"
    else:
        reason_code = "adjacent_interval"
    return (CONFIRMED,
            "all %d predicted crossover(s) are confirmed by the measured ordering "
            "(%d same interval, %d adjacent interval)" % (len(events), same, adjacent),
            reason_code)
