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
"""

from __future__ import annotations

from .spec import now_utc

SCHEMA_VERSION = "dense_endtoend_comparison_v2"


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
            samples = row["samples_ns"]
            mean = sum(samples) / len(samples)
            variance = sum((value - mean) ** 2 for value in samples) / len(samples)
            per_config[config] = {"predicted_ns": predicted, "observed_ns": observed,
                                  "predicted_boundary_ns": boundary,
                                  "boundary_fraction": boundary / predicted if predicted else None,
                                  "signed_relative_error": observed / predicted - 1,
                                  "observed_cv": (variance ** 0.5) / mean if mean else None,
                                  "repetitions": len(samples)}
        report["models"][model_id] = per_config
    return report
