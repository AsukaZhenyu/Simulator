"""Common evidence gates for independent performance-validation campaigns.

Pure stdlib: no fitting, source mutation, hardware execution or sample removal.
Each row is one independently launched process for one declared strategy. The
same native process may yield several strategy rows with the same process ID.
Intervals resample independent process medians, not every timing as if it came
from an independent process. Batch means and isolated invocations stay distinct.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import random
import re
import statistics

SCHEMA = "independent-performance-evidence-0.1"
QUALITY = dict(min_processes_per_split=3, min_samples_per_process=3,
               max_within_process_cv=.15, max_between_process_cv=.20,
               max_heldout_relative_error=.20)


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timezone-aware collection/freeze timestamps required")
    return result


def _stats(values):
    mean = statistics.mean(values)
    return dict(n=len(values), median_ns=statistics.median(values), mean_ns=mean,
                cv=statistics.pstdev(values) / mean, min_ns=min(values), max_ns=max(values))


def _interval(values, seed, repetitions):
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [estimates[int(.025 * (repetitions - 1))], estimates[int(.975 * (repetitions - 1))]]


def evaluate(rows, predictions, *, model_frozen_utc, model_sha256,
             quality=None, bootstrap_seed=1907, bootstrap_repetitions=2000,
             historically_inspected=False):
    """Evaluate explicit complete-path predictions and never fit from targets.

    Rows require id, case_id, strategy_id, comparison_group, family, split,
    process_id, seed, status, samples_ns, numerical_valid, source_sha256,
    physical_signature_sha256, timing_unit and started_utc. Failed rows may omit
    samples but remain failures. comparison_group must describe an identical
    input/output placement, warm/cold state and completion boundary.

    Predictions map (case_id, strategy_id) to positive ns values. Physical
    signatures are checked per path; different strategies may have different
    internal physical graphs while sharing their declared external boundary.
    Source/binary/model provenance must additionally be verified by each runner.
    """
    quality = dict(QUALITY if quality is None else quality)
    if set(quality) != set(QUALITY):
        raise ValueError("all quality thresholds must be explicit")
    if any(type(quality[k]) is not int or quality[k] < 3 for k in ("min_processes_per_split", "min_samples_per_process")):
        raise ValueError("at least three processes and samples are required")
    if any(not _positive(quality[k]) for k in ("max_within_process_cv", "max_between_process_cv", "max_heldout_relative_error")):
        raise ValueError("finite positive quality thresholds required")
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap draws required")
    if not _sha(model_sha256):
        raise ValueError("frozen model SHA256 required")
    frozen = _time(model_frozen_utc)
    groups, seen = defaultdict(list), set()
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("unique nonempty row IDs required")
        seen.add(rid)
        for key in ("case_id", "strategy_id", "comparison_group", "family", "process_id", "timing_unit"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("explicit row field required: " + key)
        if row.get("split") not in ("profile", "heldout") or type(row.get("seed")) is not int:
            raise ValueError("explicit split and seed required")
        if row.get("status") not in ("measured", "failed"):
            raise ValueError("unknown measurement status")
        if not _sha(row.get("source_sha256")) or not _sha(row.get("physical_signature_sha256")):
            raise ValueError("raw-record and physical-signature SHA256 required")
        _time(row["started_utc"])
        if row["status"] == "measured":
            samples = row.get("samples_ns")
            if (not isinstance(samples, list) or len(samples) < quality["min_samples_per_process"]
                    or any(type(x) is not int or x < 1 for x in samples)):
                raise ValueError("unfiltered positive integer ns sample list required")
            if type(row.get("numerical_valid")) is not bool:
                raise ValueError("explicit numerical validity required")
        groups[(row["case_id"], row["strategy_id"])].append(row)
    if not groups:
        raise ValueError("empty validation campaign")
    if set(predictions) - set(groups):
        raise ValueError("prediction for an undeclared path")
    results = []
    for key, selected in sorted(groups.items()):
        def common(field):
            values = {row[field] for row in selected}
            if len(values) != 1:
                raise ValueError("path has mixed " + field)
            return next(iter(values))
        signature, unit, comparison, family = (common(field) for field in (
            "physical_signature_sha256", "timing_unit", "comparison_group", "family"))
        splits = {split: [r for r in selected if r["split"] == split] for split in ("profile", "heldout")}
        measured = {split: [r for r in values if r["status"] == "measured"] for split, values in splits.items()}
        pids = {split: [r["process_id"] for r in values] for split, values in splits.items()}
        seeds = {split: {r["seed"] for r in values} for split, values in splits.items()}
        process = [{"row_id": r["id"], "split": r["split"], "process_id": r["process_id"],
                    **_stats(r["samples_ns"])} for r in selected if r["status"] == "measured"]
        cvs = {}
        for split in measured:
            medians = [x["median_ns"] for x in process if x["split"] == split]
            cvs[split] = statistics.pstdev(medians) / statistics.mean(medians) if len(medians) >= 2 else None
        heldout_medians = [x["median_ns"] for x in process if x["split"] == "heldout"]
        pooled = [x for r in measured["heldout"] for x in r["samples_ns"]]
        observed = statistics.median(pooled) if pooled else None
        prediction = predictions.get(key)
        if prediction is not None and not _positive(prediction):
            raise ValueError("finite positive prediction required")
        error = abs(prediction - observed) / observed if prediction is not None and observed is not None else None
        gates = dict(
            all_declared_processes_measured=all(r["status"] == "measured" for r in selected),
            numerical_validity=all(r.get("numerical_valid") is True for r in selected),
            independent_process_coverage=all(len(ids) >= quality["min_processes_per_split"] and len(ids) == len(set(ids)) for ids in pids.values()) and not (set(pids["profile"]) & set(pids["heldout"])),
            disjoint_profile_heldout_seeds=not (seeds["profile"] & seeds["heldout"]),
            frozen_before_heldout=bool(splits["heldout"]) and all(_time(r["started_utc"]) >= frozen for r in splits["heldout"]),
            within_process_cv=bool(process) and all(x["cv"] <= quality["max_within_process_cv"] for x in process),
            between_process_cv=all(cv is not None and cv <= quality["max_between_process_cv"] for cv in cvs.values()),
            heldout_relative_error=error is not None and error <= quality["max_heldout_relative_error"],
        )
        accepted = all(gates.values())
        results.append(dict(case_id=key[0], strategy_id=key[1], family=family,
            comparison_group=comparison, timing_unit=unit, physical_signature_sha256=signature,
            predicted_ns=prediction, observed_ns=observed, absolute_relative_error=error,
            signed_relative_error=(prediction - observed) / observed if error is not None else None,
            heldout_process_median_interval_ns=_interval(heldout_medians, bootstrap_seed, bootstrap_repetitions) if heldout_medians else None,
            process_statistics=process, between_process_cv=cvs,
            gates=gates, original_quality_accepted=accepted,
            fresh_validation_accepted=accepted and not historically_inspected,
            failed_rows=[r["id"] for r in selected if r["status"] != "measured"],
            unknown_prediction=prediction is None))
    comparable = defaultdict(list)
    for row in results:
        comparable[(row["case_id"], row["comparison_group"], row["timing_unit"])].append(row)
    rankings = []
    for (cid, boundary, unit), values in sorted(comparable.items()):
        known = [r for r in values if r["predicted_ns"] is not None and r["observed_ns"] is not None]
        if len(values) < 2:
            continue
        coverage = len(known) == len(values)
        if not coverage:
            rankings.append(dict(case_id=cid, comparison_group=boundary, timing_unit=unit,
                                 known=False, reason="missing prediction or observed path"))
            continue
        predicted = sorted(known, key=lambda r: (r["predicted_ns"], r["strategy_id"]))
        actual = sorted(known, key=lambda r: (r["observed_ns"], r["strategy_id"]))
        regret = predicted[0]["observed_ns"] - actual[0]["observed_ns"]
        best_interval = actual[0]["heldout_process_median_interval_ns"]
        decisive = all(best_interval[1] < x["heldout_process_median_interval_ns"][0] for x in actual[1:])
        rankings.append(dict(case_id=cid, comparison_group=boundary, timing_unit=unit, known=True,
            predicted_order=[x["strategy_id"] for x in predicted], observed_order=[x["strategy_id"] for x in actual],
            selected_best=predicted[0]["strategy_id"] == actual[0]["strategy_id"],
            observed_winner_interval_separated=decisive, selection_regret_ns=regret,
            selection_regret_relative=regret / actual[0]["observed_ns"],
            latency_quality_accepted=all(x["original_quality_accepted"] for x in values),
            fresh_validation_accepted=all(x["fresh_validation_accepted"] for x in values)))
    errors = [x["absolute_relative_error"] for x in results if x["absolute_relative_error"] is not None]
    return dict(schema=SCHEMA, quality=quality, model_sha256=model_sha256,
        model_frozen_utc=model_frozen_utc, historically_inspected=historically_inspected,
        interval_unit="independent_process_medians; not isolated-request tail latency",
        bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions,
        paths=len(results), unknown_predictions=sum(x["unknown_prediction"] for x in results),
        median_absolute_relative_error=statistics.median(errors) if errors else None,
        maximum_absolute_relative_error=max(errors) if errors else None,
        original_quality_accepted=sum(x["original_quality_accepted"] for x in results),
        fresh_validation_accepted=sum(x["fresh_validation_accepted"] for x in results),
        results=results, rankings=rankings, physical_capacity_validated=False,
        provenance_scope="row/source/model identity binding; original raw evidence must be independently verified by campaign runner")
