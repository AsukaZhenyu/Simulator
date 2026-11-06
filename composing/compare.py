"""Measured wall clock against the prediction, never the other way round.

``compare`` reads ``predicted.json`` and ``measured.json`` and writes
``comparison.json``.  It does not edit the prediction, does not re-price
anything, and does not call the measurement: the three documents stay three
separate claims, and the order between them is the pre-registration rule.

Two refusals live here because a wall clock can silently answer another
question:

* a row measured under a different KV element type is a different byte count,
  so it cannot be compared with a table that prices an f16 cache;
* a measured set that used several K cache types is not one protocol at all.

Every row also carries the spread of its own repetitions, and the document
carries a session record naming the rows wider than the state band.  That is
not a correction: a prediction is still compared with the observed point.  It
is the second column a reader needs, because an error smaller than the band is
a statement about this machine and not about the cost model.

The document also carries the decode grouping it was measured under.  That is
not bookkeeping either: the slope of a decode run depends on who pays the
process's first-decode cost, and the two groupings decide that differently.
"""

from __future__ import annotations

from .spec import now_utc
from .variability import session_summary, summarise

SCHEMA_VERSION = "dense_endtoend_comparison_v3"


def compare(prediction: dict, measured: dict) -> dict:
    taken = measured.get("measured_at_utc", {})
    report = {"schema_version": SCHEMA_VERSION, "created_at_utc": now_utc(),
              "claimed_metric": "complete_token_step_vs_llama_bench_wall_clock",
              "calibration_claimed": False,
              # When each model was measured: rows from different processes are
              # not one observation of one machine state.
              "measured_at_utc": dict(taken),
              # The prediction's declared terms, echoed so a comparison file is
              # readable without its prediction.  A prediction that predates
              # terms carries none, which is zero declared terms.
              "composition_terms": list(prediction.get("composition_terms") or []),
              # How the decode rows were grouped, and which sample the protocol
              # threw away.  A per-shape run opens one process per depth, so
              # every depth pays its own first-decode cost; a grouped run opens
              # one process for all depths and discards the first sample of the
              # first shape.  The two do not give the same slope on the same
              # three depths (2026-11-05 B2, olmoe: 6.525 vs 8.481 us/KVtok), so
              # a slope read from this file is unreadable without this line.
              "decode_grouping": measured.get("decode_grouping"),
              "warmup": measured.get("warmup"),
              "models": {}}
    for model_id, rows in measured["models"].items():
        shapes = prediction["models"][model_id]["shapes"]
        # A table that prices an f16 cache cannot be compared with a run that
        # read another element type: the two numbers answer different questions.
        declared_kv = prediction["models"][model_id].get("kv_dtype")
        observed_kv = {(row.get("llama_bench") or {}).get("type_k") for row in rows.values()}
        observed_kv.discard(None)
        if len(observed_kv) > 1:
            raise ValueError(f"{model_id}: measured rows used several K cache types "
                             f"{sorted(observed_kv)}; they are not one protocol")
        if declared_kv and observed_kv and observed_kv != {declared_kv}:
            raise ValueError(f"{model_id}: the table prices a {declared_kv} KV cache but "
                             f"llama-bench ran {sorted(observed_kv)}; the comparison would "
                             f"be between two different byte counts")
        per_config = {}
        for config, row in rows.items():
            predicted = shapes[config]["total_ns"]
            boundary = shapes[config]["embedding"]["median_ns"] + shapes[config]["head"]["median_ns"]
            observed = row["avg_ns"]
            observation = summarise(row["samples_ns"], where=f"{model_id}/{config}")
            per_config[config] = {"predicted_ns": predicted, "observed_ns": observed,
                                  "predicted_boundary_ns": boundary,
                                  "boundary_fraction": boundary / predicted if predicted else None,
                                  "signed_relative_error": observed / predicted - 1,
                                  "observed_cv": observation["cv"],
                                  "repetitions": observation["n"],
                                  # A fact, not a verdict: is the prediction
                                  # inside the span this row's own repetitions
                                  # covered?  A row where it is not is a row the
                                  # band cannot explain.
                                  "predicted_inside_observation":
                                      observation["min_ns"] <= predicted <= observation["max_ns"],
                                  # What the clock did inside this row.  An error
                                  # inside this band is not evidence about the
                                  # model, so it is recorded rather than averaged
                                  # away.
                                  "observation": observation}
        report["models"][model_id] = per_config
    report["session"] = session_summary(
        {f"{model_id}/{config}": row["observation"]
         for model_id, rows in report["models"].items()
         for config, row in rows.items()})
    return report
