"""Diagnostic: price the request-start decode windows as their own class.

This is not the campaign. It re-runs the calibration/reforecast/evaluate
arithmetic in memory on a cohort that was already measured, so that the effect
of one analysis change can be read on the same runs. The control arm must
reproduce the cohort's own published condition means before the treatment arm
is read; otherwise the replay is not the pipeline.

    python -m profiling.real_moe_validation.diag_history_split --out <cohort> \\
        --history-policy first-continuation|first-settling-continuation
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re
import sys


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patched_analysis(analysis_path):
    """The shipped analysis with the previous decode-history rule put back.

    The shipped rule names a decode forward's position among the forwards of its
    padded KV depth. The control arm is the rule that preceded it: ``first`` is
    the request's first decode forward and every later one is a continuation.
    """
    source = Path(analysis_path).read_text(encoding="utf-8")
    old = "history=decode_history(f,depth_rank,first_decode,closes_request=(kind==26))"
    if old not in source:
        raise SystemExit("shipped history rule moved; update this diagnostic before reading it")
    new = "history='decode:'+('first' if f<=first_decode else 'continuation')"
    source = source.replace(old, new)
    import tempfile
    target = Path(tempfile.mkdtemp()) / "_diag_rwa_legacy.py"
    target.write_text(source, encoding="utf-8")
    return load_module(target, "_diag_rwa_legacy")


def number(value):
    return Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else Fraction(value)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-policy", choices=("first-continuation", "first-settling-continuation"),
                        default="first-continuation")
    args = parser.parse_args(argv)

    from profiling.real_moe_validation import runtime_window_analysis as analysis
    from profiling.real_moe_validation.runtime_window_io import LABEL_PLACEMENT, LEGACY_LABELS

    out = args.out.resolve()
    if args.history_policy == "first-continuation":
        analysis = patched_analysis(analysis.__file__)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = tuple(manifest.get("labels", LEGACY_LABELS))
    policy = manifest.get("feature_policy", "mechanism_v1")

    def native_of(job):
        execution = json.loads((out / "jobs" / job / "execution.json").read_text(encoding="utf-8"))
        assert execution["status"] == "completed", job
        return json.loads(Path(execution["native"]["path"]).read_text(encoding="utf-8"))

    ledgers, evidence = [], []
    for label in labels:
        native = native_of(label + "-calibration-profile")
        evidence.append(str(out / "jobs" / (label + "-calibration-profile") / "result.json"))
        for row in native["results"]:
            ledgers.append(analysis.extract_windows(
                row, placement=LABEL_PLACEMENT[label],
                frequency_hz=native["clock"]["frequency_hz"], feature_policy=policy))
    prices = analysis.fit_prices(ledgers, evidence, policy)
    keys = defaultdict(int)
    for ledger in ledgers:
        for window in ledger["windows"]:
            keys[(window["inventory"]["phase"], window["inventory"]["history"])] += 1

    forecasts, inventories, truth = {}, {}, {}
    for label in labels:
        native = native_of(label + "-inventory")
        for row in native["results"]:
            ledger = analysis.extract_windows(row, placement=LABEL_PLACEMENT[label],
                                              frequency_hz=native["clock"]["frequency_hz"],
                                              feature_policy=policy)
            inventory = dict(placement=LABEL_PLACEMENT[label], request_id=row["id"],
                             prompt_token_ids=row["prompt_token_ids"],
                             generated_token_ids=row["output_token_ids"],
                             final_full_logits_sha256=row["final_full_logits_sha256"],
                             windows=ledger["workload_inventory"])
            key = (inventory["placement"], inventory["request_id"])
            inventories[key] = inventory
            forecasts[key] = analysis.predict_inventory(inventory["windows"], prices,
                                                        allow_extrapolation=True)
        for repeat in (0, 1):
            bare = native_of(label + "-heldout-r" + str(repeat))
            for row in bare["results"]:
                truth[(LABEL_PLACEMENT[label], row["id"], repeat)] = row

    grouped = defaultdict(list)
    for (placement, request_id, repeat), row in truth.items():
        inventory = inventories[(placement, request_id)]
        same_work = (row["prompt_token_ids"] == inventory["prompt_token_ids"]
                     and row["output_token_ids"] == inventory["generated_token_ids"])
        same_final = row["final_full_logits_sha256"] == inventory["final_full_logits_sha256"]
        grouped[(placement, request_id)].append(
            row if same_work and same_final else "mismatch")

    conditions = []
    for (placement, request_id), repeats in sorted(grouped.items()):
        scored = [row for row in repeats if row != "mismatch"]
        record = dict(placement=placement, request_id=request_id, repeats=len(repeats),
                      status="evaluated" if len(scored) == len(repeats) else "incomplete")
        if len(scored) != len(repeats):
            conditions.append(record)
            continue
        measured = {
            "total": sum(row["whole_request_wall_ns"] for row in scored) / len(scored),
            "prefill": sum(row["prefill_completed_ns"] for row in scored) / len(scored),
            "decode_with_sampling": sum(row["whole_request_wall_ns"] - row["prefill_completed_ns"]
                                        for row in scored) / len(scored),
        }
        prediction = forecasts[(placement, request_id)]
        predicted = {"total": number(prediction["total_ns"]),
                     "prefill": number(prediction["phase_ns"]["prefill"]),
                     "decode_with_sampling": number(prediction["phase_ns"]["decode"])}
        ape = {name: float(abs(float(predicted[name]) / measured[name] - 1)) for name in measured}
        record.update(measured_ms={k: v / 1e6 for k, v in measured.items()},
                      predicted_ms={k: float(v) / 1e6 for k, v in predicted.items()},
                      APE=ape, passed=all(v <= 0.10 for v in ape.values()))
        conditions.append(record)

    complete = [c for c in conditions if c["status"] == "evaluated"]
    summary = dict(history_policy=args.history_policy, cohort=str(out),
                   calibration_windows_by_phase_and_history=dict(
                       (str(k), v) for k, v in sorted(keys.items())),
                   price_count=len(prices),
                   declared=len(conditions), complete=len(complete),
                   passed=sum(c["passed"] for c in complete),
                   mean_APE={m: sum(c["APE"][m] for c in complete) / len(complete) for m in
                             ("total", "prefill", "decode_with_sampling")},
                   worst_APE={m: max(c["APE"][m] for c in complete) for m in
                              ("total", "prefill", "decode_with_sampling")},
                   conditions=conditions)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=1))
    for c in complete:
        print(c["placement"], c["request_id"], "total %.4f prefill %.4f decode %.4f %s" % (
            c["APE"]["total"], c["APE"]["prefill"], c["APE"]["decode_with_sampling"],
            "PASS" if c["passed"] else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
