"""Price a complete token step from an exported table, before anything is timed.

A token step is ``embedding lookup -> block x layers -> final norm + LM head``.
The block stack is priced per shape; the boundary is priced from its own
measured entries.  The head is priced at one token even for a prefill, because
llama.cpp computes logits for the last position of a batch and only that row
reaches the vocabulary-wide matmul.

Everything the prediction does not cover is named in ``excluded_terms``; it is
never silently absorbed into the block sum.  Declared terms (see
:mod:`composing.terms`) are the exception with a name and a knob: each one is
recorded in ``composition_terms`` and, when enabled, added by the one summation
rule in :mod:`composing.compose`.

This module writes ``predicted.json`` and nothing else.  ``measure`` refuses to
run without one, which is the pre-registration rule: a prediction that could be
edited after the wall clock would not be a prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compose import compose_with_breakdown
from .spec import (FFN_STAGES, STAGE_KINDS, SHAPES, declared_policies, dump_json,
                   entry_signature, now_utc, parse_exported_table, sha256_file,
                   shape_key, table_prices, table_regime)
from .terms import document_terms

SCHEMA_VERSION = "dense_endtoend_prediction_v2"


def boundary_cost(report: dict, kind: str, phase: str, tokens: int, vocab: int,
                  prices: dict | None = None) -> dict:
    """One boundary stage at one exact token count; history is not an input.

    A boundary stage is measured at history 0 because it has no cache.  The
    vocabulary it multiplies is checked against the model the prediction names:
    a table measured for a different vocabulary is a different price key.
    """
    wanted = shape_key(phase, tokens, 0)
    entries = [entry for entry in report["entries"]
               if entry["kind"] == kind and entry["shape"] == wanted]
    if len(entries) != 1:
        raise ValueError(f"{wanted}: expected exactly one {kind} entry, found {len(entries)}")
    price = None if prices is None else prices.get((kind, phase, int(tokens), 0))
    if prices is not None and price is None:
        raise ValueError(f"{wanted}: the table exports no {kind} price for it")
    # The export report keeps the physical signature inside the measurement
    # signature it re-derived from the raw probe output.
    signature = entry_signature(entries[0])
    config = signature.get("config") or {}
    if config.get("vocab") != vocab:
        raise ValueError(f"{wanted}: {kind} was measured at vocab "
                         f"{config.get('vocab')}, not the declared {vocab}")
    return {"median_ns": (int(entries[0]["sample_summary"]["median_ns"]) if price is None
                          else price),
            "cv": entries[0]["sample_summary"]["cv"],
            "vocab": vocab,
            "token_embd_dtype": signature["config"]["token_embd_dtype"],
            "lm_head_dtype": signature["config"]["lm_head_dtype"]}


def block_cost(report: dict, phase: str, tokens: int, history: int,
               stages: tuple[str, ...] = STAGE_KINDS,
               prices: dict | None = None) -> dict:
    """Sum the three declared stages for one shape; refuse anything incomplete.

    The report also carries the joint ``block`` record for the same shape; it is
    the composition reference, not a schedulable entry, and must not be summed.

    ``prices`` is the table's own price map.  Without it the raw per-case medians
    are summed, which is what every campaign before the pooled-median policy
    exported anyway.
    """
    wanted = shape_key(phase, tokens, history)
    found = {entry["kind"]: entry for entry in report["entries"]
             if entry["shape"] == wanted and entry["kind"] in stages}
    missing = set(stages) - set(found)
    if missing:
        raise ValueError(f"{wanted} lacks {sorted(missing)}; no interpolation is allowed")
    if report.get("validation_only") is not False:
        raise ValueError("a validation_only table cannot back a main prediction")

    def price(kind):
        if prices is None:
            return int(found[kind]["sample_summary"]["median_ns"])
        key = (kind, phase, int(tokens), int(history))
        if key not in prices:
            raise ValueError(f"{wanted}: the table exports no {kind} price for it")
        return prices[key]

    values = {kind: price(kind) for kind in sorted(found)}
    values["block_sum_ns"] = sum(values.values())
    values["stage_cv"] = {kind: found[kind]["sample_summary"]["cv"] for kind in sorted(found)}
    return values


def predict(tables: dict[str, Path], models: dict, models_dir: Path,
            terms: tuple = ()) -> dict:
    """Price a complete token step, boundary included, from the exported tables."""
    result = {"schema_version": SCHEMA_VERSION, "created_at_utc": now_utc(),
              "scope": "complete token step: token embedding, declared block stack, "
                       "final norm and LM head",
              "composition_rule": "block_sum_ns * layers + embedding + head "
                                  "+ enabled declared terms",
              # Every declared term, applied or not.  An empty list is a claim:
              # no quantity outside the table was priced.
              "composition_terms": document_terms(terms),
              "excluded_terms": ["sampling", "graph_construction", "detokenization",
                                 "tokenizer"],
              "declared_boundary_conflicts": [
                  "the declared stages materialize their outputs; llama.cpp keeps the "
                  "same values in registers or in place",
                  "the head is priced at one token: llama.cpp evaluates the vocabulary "
                  "matmul for the last position of a batch only"],
              "models": {}}
    for model_id, export_dir in tables.items():
        meta = models[model_id]
        report = json.loads((export_dir / "export_report.json").read_text(encoding="utf-8"))
        # Priced from the table, not from the evidence: a table may have been
        # re-priced under a declared policy, and then the raw medians the report
        # keeps are no longer the prices this model exports.  An export directory
        # without a table is priced from the report and says so.
        costs_path = export_dir / "costs.json"
        table_document = (json.loads(costs_path.read_text(encoding="utf-8"))
                          if costs_path.is_file() else None)
        table = (None if table_document is None
                 else parse_exported_table(table_document, where=str(costs_path)))
        prices = None if table is None else table_prices(table)
        layers = int(meta["geometry"]["block_count"])
        vocab = meta["geometry"].get("vocab_size")
        if not vocab:
            raise ValueError(f"{model_id}: the manifest declares no vocab_size, so the "
                             "boundary cannot be priced against this model")
        # The report names all four declared stages; the first three are the ones
        # a token step sums, the fourth is the joint reference they are checked
        # against and must never be added into the price.
        declared = tuple(report.get("decoder_stages") or (*STAGE_KINDS, "block"))
        # The manifest declares the feed-forward as its own object (kind, experts,
        # widths); the stage name has to come from that object, not from the label.
        feed_forward = (meta.get("geometry") or {}).get("ffn")
        if isinstance(feed_forward, dict):
            feed_forward = feed_forward.get("kind")
        expected_ffn = FFN_STAGES.get(feed_forward)
        if expected_ffn is None:
            raise ValueError(f"{model_id}: the manifest declares no feed-forward kind, so the "
                             "table cannot be matched to it")
        if len(declared) != 4 or declared[3] != "block" or declared[2] != expected_ffn:
            raise ValueError(f"{model_id}: the table prices {declared} but this model's declared "
                             f"feed-forward is {expected_ffn!r}")
        stages = declared[:3]
        # The wall clock is only comparable to this table if llama.cpp read the
        # same KV element type: the declared block prices an f16 cache, and a
        # quantized run would answer a different question.  Carried here so
        # ``compare`` can check it against the bench row instead of assuming it.
        kv_dtypes = {entry_signature(item).get("kv_dtype")
                     for item in report["entries"] if item["kind"] == "attention"}
        kv_dtypes.discard(None)
        if len(kv_dtypes) > 1:
            raise ValueError(f"{model_id}: the table prices attention under several KV dtypes "
                             f"{sorted(kv_dtypes)}; its decode slope is not one quantity")
        policies = [] if table is None else declared_policies(table)
        entry = {"layers": layers, "table": str(export_dir.resolve()),
                 "contract": report.get("contract"), "decoder_stages": list(declared),
                 "residency_regime": table_regime(report),
                 "history_invariant_price_policy": (policies[0] if len(policies) == 1 else None),
                 "price_source": ("costs.json" if table_document is not None
                                  else "export_report_medians"),
                 "kv_dtype": kv_dtypes.pop() if kv_dtypes else None,
                 "table_sha256": sha256_file(export_dir / "export_report.json"),
                 "costs_sha256": (sha256_file(costs_path) if table_document is not None else None),
                 "gguf": str((models_dir / meta["filename"]).resolve()),
                 "vocab": vocab, "shapes": {}}
        # The head runs once per generated position or batch position, always at
        # one visible token: the measured q=1 price is the one that applies.
        head = boundary_cost(report, "head", "decode", 1, vocab, prices)
        entry["head"] = head
        for name, (phase, tokens, history) in SHAPES.items():
            cost = block_cost(report, phase, tokens, history, stages, prices)
            embed = boundary_cost(report, "embedding", phase, tokens, vocab, prices)
            total_ns, breakdown = compose_with_breakdown(
                block_sum_ns=cost["block_sum_ns"], layers=layers,
                boundary_ns=embed["median_ns"] + head["median_ns"],
                phase=phase, tokens=tokens, terms=terms)
            shape = {**cost, "layers": layers, "embedding": embed, "head": head,
                     "total_ns": total_ns}
            if breakdown:
                # Only present when something was declared: a zero-term shape is
                # byte-for-byte the shape every archived comparison was made
                # against.
                shape["terms_ns"] = breakdown
            entry["shapes"][name] = shape
        # Decode time cannot fall as the history grows: attention is the only
        # stage whose price depends on history and it can only grow.  A fall means
        # the history-independent stages were priced at different points of the
        # per-process measurement state, and that noise then enters the predicted
        # slope.  Recorded, never corrected: the prediction is what it is.
        decode = sorted((SHAPES[name][2], name) for name in entry["shapes"]
                        if SHAPES[name][0] == "decode")
        for (_, short), (_, longer) in zip(decode, decode[1:]):
            shallower = entry["shapes"][short]["total_ns"]
            deeper = entry["shapes"][longer]["total_ns"]
            if deeper < shallower:
                result["declared_boundary_conflicts"].append(
                    f"{model_id}: predicted decode time falls from {short} ({shallower} ns) to "
                    f"{longer} ({deeper} ns) although only attention grows with history")
        result["models"][model_id] = entry
    return result


def models_document(path: Path) -> dict:
    """``models.json`` as ``id -> manifest``."""
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document["models"] if isinstance(document, dict) and "models" in document else document
    return {model["id"]: model for model in models}


def write_prediction(output: Path, document: dict) -> Path:
    path = output / "predicted.json"
    dump_json(path, document)
    return path
