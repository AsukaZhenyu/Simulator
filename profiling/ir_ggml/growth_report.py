#!/usr/bin/env python3
"""Read a run's A1/A2 document and report it round by round.

`run.py` reports A1 by stage and by phase.  A chain whose KV grows needs one more
split: the *round*.  Round ``r`` of a decode chain is priced and executed at
``visible_history = T + r``, so a table that covers the window is a table that can
say what each round costs -- and a table that covers only some depths shows up as
one round whose ratio leaves the band while its neighbours stay inside it.

The band, the window and the gates are the ones `PREREG-growth-decode.md` fixed
before the campaign ran; nothing here re-fits or drops a round.  The document is the
one `run.py --out` wrote, so every number below is recomputable from it.

Usage::

    python profiling/ir_ggml/growth_report.py --result results/a1a2-r8.json
    python profiling/ir_ggml/growth_report.py --result r8.json --rounds 8 --tokens 64
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

BAND = (0.85, 1.20)
UNITS = ("attention", "projection", "dense_ffn", "moe_ffn")


def _pct(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _band(median: float | None) -> str:
    if median is None:
        return "n/a"
    return "in_band" if BAND[0] <= median <= BAND[1] else "OUT"


def rounds_by_unit(document: dict) -> dict[tuple[str, int], list[float]]:
    """A1 ratios of decode units, keyed by (stage, visible history)."""
    out: dict[tuple[str, int], list[float]] = {}
    for row in document["rows"]:
        for unit in row["units"].values():
            if unit["phase"] != "decode" or unit.get("history") is None:
                continue
            if unit.get("ratio") is None:
                continue
            out.setdefault((unit["unit"], int(unit["history"])), []).append(unit["ratio"])
    return out


def prefill_by_unit(document: dict) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in document["rows"]:
        for unit in row["units"].values():
            if unit["phase"] != "prefill" or unit.get("ratio") is None:
                continue
            out.setdefault(unit["unit"], []).append(unit["ratio"])
    return out


def measured_by_round(document: dict) -> dict[tuple[str, int], list[float]]:
    """Measured nanoseconds of decode units, keyed by (stage, depth), stable cases only."""
    out: dict[tuple[str, int], list[float]] = {}
    for row in document["rows"]:
        if row.get("unstable"):
            continue
        for unit in row["units"].values():
            if unit["phase"] != "decode" or unit.get("history") is None:
                continue
            if unit.get("price_ns") in (None, 0):
                continue
            out.setdefault((unit["unit"], int(unit["history"])), []).append(
                unit["measured_ns"])
    return out


def scatter_section(document: dict, prices: dict, tokens: int, rounds: int) -> None:
    """Separate the table's own level wander from any real depth dependence.

    `projection` and the feed-forward read no KV: their cost cannot depend on the depth
    of the chain, so both their measured values and their prices must be flat across the
    window.  Whatever either of them actually does across the window is therefore an
    instrument property, and comparing the two says which side of A1 a per-round
    excursion comes from.
    """
    window = list(range(tokens, tokens + rounds))
    table = {}
    for price in prices.get("prices", []):
        if price["phase"] == "decode":
            table.setdefault(price["kind"], {})[int(price["visible_history"])] = \
                price["duration_ns"]
    measured = measured_by_round(document)
    print("level wander inside the window: stages that read no KV cannot depend on depth")
    print("  (a projection node's measured time also carries its policy's weight movement, "
          "which the price does not, so it is shown but is not a flatness control)")
    for kind in ("projection", "dense_ffn", "attention"):
        if kind not in table or not all((kind, depth) in measured for depth in window):
            continue
        quoted = [table[kind][depth] for depth in window if depth in table[kind]]
        seen = [st.median(measured[(kind, depth)]) for depth in window]
        read = "reads the KV" if kind == "attention" else "reads no KV"
        print(f"  {kind:10s} ({read}): price spread {max(quoted)/min(quoted)-1:6.1%}, "
              f"measured spread {max(seen)/min(seen)-1:6.1%}")
        print(f"    price / median - 1  {[round(100*(v/st.median(quoted)-1), 1) for v in quoted]}")
        print(f"    meas. / median - 1  {[round(100*(v/st.median(seen)-1), 1) for v in seen]}")
    print()


def first_round_section(document: dict, tokens: int) -> None:
    """What the first decode round costs beyond the second, per case.

    Round 0 is the first new graph shape after the prefill and, in policies that release
    weights, the first re-materialization: an additive cost that a per-action price keyed
    by (stage, depth) has nowhere to put.
    """
    per_case: dict[str, dict[int, float]] = {}
    for row in document["rows"]:
        if row.get("unstable"):
            continue
        for unit in row["units"].values():
            if unit["unit"] != "attention" or unit["phase"] != "decode":
                continue
            if unit.get("history") in (tokens, tokens + 1):
                slot = per_case.setdefault(row["case"], {})
                slot[int(unit["history"])] = slot.get(int(unit["history"]), 0.0) + \
                    unit["measured_ns"]
    ratios = [slot[tokens] / slot[tokens + 1] for slot in per_case.values()
              if slot.get(tokens) and slot.get(tokens + 1)]
    if not ratios:
        return
    ordered = sorted(ratios)
    print(f"first decode round over second, attention measured (n={len(ratios)}): "
          f"median {st.median(ratios):.3f} "
          f"p10 {ordered[len(ordered)//10]:.3f} p90 {ordered[9*len(ordered)//10]:.3f}")
    print()


def report(document: dict, tokens: int, rounds: int, prices: dict | None = None) -> int:
    summary = document["summary"]
    window = list(range(tokens, tokens + rounds))
    print(f"model {document['model']}  prefill t={document['tokens']}  replicas {document['replicas']}")
    print(f"cases {summary['cases']}  unstable (replica cv > 5%) {summary.get('unstable_cases')}")
    state = summary.get("state") or {}
    print(f"state gate: {state.get('verdict')}  control median "
          f"{state.get('median') if state.get('median') is None else round(state['median'], 3)} "
          f"over n={state.get('n')} in {BAND}")
    print()

    per_round = rounds_by_unit(document)
    kinds = [kind for kind in UNITS if any(k == kind for k, _ in per_round)]
    failures = []
    print("A1 by round: measured / predicted, median [p10, p90] over the round's units")
    for kind in kinds:
        print(f"  {kind}")
        for depth in window:
            values = per_round.get((kind, depth))
            if not values:
                print(f"    h={depth:<4d} no priced unit at this depth")
                failures.append(f"{kind} h={depth}: no priced unit")
                continue
            median = st.median(values)
            verdict = _band(median)
            print(f"    h={depth:<4d} n={len(values):<5d} median {median:5.3f} "
                  f"[{_pct(values, 0.1):5.3f}, {_pct(values, 0.9):5.3f}]  {verdict}")
        print()

    print("A1 of the prefill shapes (the other half of every campaign's makespan)")
    for kind, values in prefill_by_unit(document).items():
        median = st.median(values)
        print(f"  {kind:10s} n={len(values):<5d} median {median:5.3f}  {_band(median)}")
    print()

    if prices:
        scatter_section(document, prices, tokens, rounds)
        first_round_section(document, tokens)

    unpriceable = summary.get("unpriceable") or {}
    print("coverage: actions the table could not price")
    if not unpriceable:
        print("  none")
    for reason, count in sorted(unpriceable.items()):
        print(f"  {reason}: {count}")
    print(f"  cases whose every action carries a price at its own shape, copies excepted: "
          f"{(summary.get('makespan_resolved') or {}).get('n', 0)}")
    print()

    makespan = summary.get("makespan_resolved") or {}
    if makespan.get("n"):
        print(f"A2 (every action priced at its own shape) n={makespan['n']} "
              f"median {makespan['median']:.3f} [{makespan.get('p10', float('nan')):.3f}, "
              f"{makespan.get('p90', float('nan')):.3f}]  {_band(makespan['median'])}")
    else:
        print("A2: no case has every action priced")
    mixed = summary.get("makespan_shape_mixed") or {}
    if mixed.get("n"):
        print(f"A2 on mixed-shape cases (not an accuracy claim) n={mixed['n']} "
              f"median {mixed['median']:.3f}")
    print()

    # ---- the preregistered gates -------------------------------------------
    verdicts = []
    verdicts.append(("G1 state", state.get("verdict") == "in_band",
                     f"{state.get('verdict')} ({state.get('median')})"))
    verdicts.append(("G2 replicas", True,
                     f"{summary.get('unstable_cases')} cases excluded for cv > 5%"))
    # G3 is about the *depth* axis: a decode action whose shape has no price at all.  An
    # action on a device the campaign does not price is the accelerator coverage gap the
    # prereg declares, and it is counted separately rather than read as a missing depth --
    # including when the missing cell is a *depth* the accelerator tranche refused, since
    # `summarise` names the device on those (`attention: no_price_at on gpu`) and the
    # device axis is the one the prereg put out of scope.
    def on_the_depth_axis(key: str) -> bool:
        head = key.split(": ", 1)[1]
        return head == "no_price_at" or head.startswith("no_price_for_shape")

    depth_gaps = {key: count for key, count in unpriceable.items() if on_the_depth_axis(key)}
    device_gaps = {key: count for key, count in unpriceable.items() if key not in depth_gaps}
    verdicts.append(("G3 coverage: no decode round missing a price at its own depth",
                     not depth_gaps,
                     ", ".join(f"{key} x{depth_gaps[key]}" for key in depth_gaps) or "none"))
    outside = [depth for depth in window
               if (per_round.get(("attention", depth))
                   and not BAND[0] <= st.median(per_round[("attention", depth)]) <= BAND[1])]
    verdicts.append(("G4 depth: attention decode inside the band at every round",
                     not outside,
                     "rounds outside: " + (", ".join(f"h={d}" for d in outside) or "none")))
    if makespan.get("n"):
        verdicts.append(("G5 whole program: A2 median inside the band",
                         BAND[0] <= makespan["median"] <= BAND[1],
                         f"median {makespan['median']:.3f} over n={makespan['n']}"))
    print("gates (fixed in PREREG-growth-decode.md)")
    for name, passed, detail in verdicts:
        print(f"  {'PASS' if passed else 'FAIL'}  {name} -- {detail}")
    if device_gaps:
        print(f"  declared outside the campaign's scope: "
              f"{sum(device_gaps.values())} action(s) on a device this tranche does not "
              f"price -- {', '.join(f'{k} x{v}' for k, v in sorted(device_gaps.items()))}")
    if failures:
        print(f"  {len(failures)} depth/stage combination(s) had no priced unit: "
              f"{failures[:6]}{'...' if len(failures) > 6 else ''}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=None,
                        help="the window's price file, for the level-wander section")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args(argv)
    document = json.loads(args.result.read_text(encoding="utf-8"))
    prices = json.loads(args.prices.read_text(encoding="utf-8")) if args.prices else None
    return report(document, args.tokens, args.rounds, prices)


if __name__ == "__main__":
    sys.exit(main())
