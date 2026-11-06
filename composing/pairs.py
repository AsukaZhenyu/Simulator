"""Two wall clocks of one prediction, side by side.

A pair answers one question and refuses the rest: given the *same* prediction,
did two rounds land on the same number, and if not, was the protocol or the
machine state the thing that moved?

The refusal is the load-bearing part.  Two measured files that were taken
against different predictions are not two observations of one claim -- the
table changed between them, so the difference would mix a model change with a
machine change and no arithmetic can pull them apart afterwards.  Same for a
row whose KV element type differs between the rounds: different byte count,
different question.

Nothing here is a correction.  Neither round is promoted, averaged, or used to
adjust the other: each row keeps its own point estimate and its own spread, and
the pair adds the ratio between them plus the fact of whether either point sits
inside the other round's span.  A difference smaller than that span is not a
finding about the machine, and a difference larger than the state band is not
one about the table.
"""

from __future__ import annotations

from .spec import now_utc
from .variability import DEFAULT_BAND, summarise

SCHEMA_VERSION = "dense_endtoend_pairs_v1"
PROTOCOL_KEYS = ("decode_grouping", "warmup", "threads", "cpu_mask", "llama_bench_sha256")


def _prediction_sha(document: dict, side: str) -> str:
    sha = (document.get("prediction") or {}).get("sha256")
    if not sha:
        raise ValueError(f"{side}: the header declares no prediction sha256; a wall clock "
                         "that does not say what it was compared against cannot be paired")
    return sha


def _kv_types(document: dict) -> set[str]:
    return {(row.get("llama_bench") or {}).get("type_k")
            for rows in (document.get("models") or {}).values()
            for row in rows.values()} - {None}


def pairs(a: dict, b: dict, *, a_name: str = "a", b_name: str = "b") -> dict:
    """Pair the rows of two measured documents; refuse anything that is not a pair."""
    sha_a, sha_b = _prediction_sha(a, a_name), _prediction_sha(b, b_name)
    if sha_a != sha_b:
        raise ValueError(f"{a_name} and {b_name} were measured against different predictions "
                         f"({sha_a[:12]} vs {sha_b[:12]}); the difference between them is a "
                         "table change and a machine change at once")
    for side, document in ((a_name, a), (b_name, b)):
        types = _kv_types(document)
        if len(types) > 1:
            raise ValueError(f"{side}: rows used several K cache types {sorted(types)}; "
                             "they are not one protocol")
    kv_a, kv_b = _kv_types(a), _kv_types(b)
    if kv_a and kv_b and kv_a != kv_b:
        raise ValueError(f"the two rounds read different K cache types "
                         f"({sorted(kv_a)} vs {sorted(kv_b)}); the byte counts differ")

    rows: dict[str, dict] = {}
    unpaired: list[str] = []
    for model_id, a_rows in (a.get("models") or {}).items():
        b_rows = (b.get("models") or {}).get(model_id) or {}
        for config, a_row in a_rows.items():
            label = f"{model_id}/{config}"
            b_row = b_rows.get(config)
            if b_row is None:
                unpaired.append(label)
                continue
            observation_a = summarise(a_row["samples_ns"], where=f"{a_name}:{label}")
            observation_b = summarise(b_row["samples_ns"], where=f"{b_name}:{label}")
            rows[label] = {
                "observed_a_ns": a_row["avg_ns"], "observed_b_ns": b_row["avg_ns"],
                "delta_ns": b_row["avg_ns"] - a_row["avg_ns"],
                "ratio": b_row["avg_ns"] / a_row["avg_ns"],
                "observation_a": observation_a, "observation_b": observation_b,
                # Facts, not a verdict: is the point estimate of one round inside
                # the span the other round's own repetitions covered?  When both
                # are, the two rounds are not separable at this row's resolution.
                "a_inside_b_span": observation_b["min_ns"] <= a_row["avg_ns"]
                                   <= observation_b["max_ns"],
                "b_inside_a_span": observation_a["min_ns"] <= b_row["avg_ns"]
                                   <= observation_a["max_ns"],
                # One definition of "the rounds did not separate this row": the
                # move between the two point estimates is no larger than the
                # wider of the two rounds' own ranges.  The two flags above are
                # the per-direction facts; this is the summary's rule, and it is
                # stated here rather than left to the reader to invent.
                "not_separable": abs(b_row["avg_ns"] - a_row["avg_ns"])
                                  <= max(observation_a["max_ns"] - observation_a["min_ns"],
                                         observation_b["max_ns"] - observation_b["min_ns"]),
            }
    for model_id, b_rows in (b.get("models") or {}).items():
        a_rows = (a.get("models") or {}).get(model_id) or {}
        unpaired += [f"{model_id}/{config}" for config in b_rows if config not in a_rows]

    protocol = {side: {key: document.get(key) for key in PROTOCOL_KEYS}
                for side, document in ((a_name, a), (b_name, b))}
    outside = sorted((label for label, row in rows.items()
                      if abs(row["ratio"] - 1) > DEFAULT_BAND),
                     key=lambda label: -abs(rows[label]["ratio"] - 1))
    widest = max(rows, key=lambda label: abs(rows[label]["ratio"] - 1)) if rows else None
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": now_utc(),
            "claimed_metric": "same_prediction_two_wall_clocks",
            "prediction_sha256": sha_a,
            "a": a_name, "b": b_name,
            "protocol": protocol,
            "same_grouping": protocol[a_name]["decode_grouping"]
                             == protocol[b_name]["decode_grouping"],
            "models": rows, "unpaired": sorted(unpaired),
            "summary": {"band": DEFAULT_BAND, "rows": len(rows),
                        "outside_band": outside,
                        "widest_row": widest,
                        "widest_ratio": rows[widest]["ratio"] if widest else None,
                        # Rows this pair cannot separate.  Counted, not
                        # subtracted: a pair that cannot separate a row says
                        # nothing about it either way.
                        "not_separable": sorted(label for label, row in rows.items()
                                                if row["not_separable"])}}


def show(report: dict) -> None:
    a_name, b_name = report["a"], report["b"]
    grouping_a = report["protocol"][a_name]["decode_grouping"]
    grouping_b = report["protocol"][b_name]["decode_grouping"]
    print(f"\n{len(report['models'])} paired rows against one prediction "
          f"({report['prediction_sha256'][:12]})")
    print(f"  {a_name}: {grouping_a}\n  {b_name}: {grouping_b}"
          + ("" if report["same_grouping"] else "   (different protocols)"))
    for label, row in report["models"].items():
        mark = "  <-- beyond the band" if label in report["summary"]["outside_band"] else ""
        span = ", inside both spans" if (row["a_inside_b_span"] and row["b_inside_a_span"]) \
            else ""
        print(f"  {label:34s} {row['observed_a_ns'] / 1e6:9.3f} -> "
              f"{row['observed_b_ns'] / 1e6:9.3f} ms  x{row['ratio']:.4f}"
              f"  cv {row['observation_a']['cv']:.3f} / {row['observation_b']['cv']:.3f}"
              f"{span}{mark}")
    for label in report["unpaired"]:
        print(f"  {label:34s} measured in one round only -- not paired")
    summary = report["summary"]
    if summary["rows"]:
        print(f"  widest x{summary['widest_ratio']:.4f} ({summary['widest_row']}); "
              f"{len(summary['outside_band'])} of {summary['rows']} beyond the "
              f"{summary['band']:.0%} band; {len(summary['not_separable'])} not separable")
