"""Profile-only exact-context prices for declared, externally verified regimes.

No utilization-to-slowdown rule, interpolation, outlier removal or heldout fit is
performed. A named induced condition is not an automatic telemetry classifier.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

SCHEMA = "named-regime-cost-model-0.1"
EVALUATION_SCHEMA = "named-regime-cost-evaluation-0.1"
CONDITIONS = ("baseline", "cpu_replay_load", "gpu_replay_load")
METRICS = ("target", "iteration")
GATES = {"within_process_cv_max": 0.15, "between_process_cv_max": 0.20,
         "pooled_median_relative_error_max": 0.20,
         "minimum_process_records_per_split": 3,
         "equal_samples_per_process_required": True}


class UnknownCost(LookupError):
    """No supported measured price for the exact requested context/regime."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fraction(value: Fraction | None) -> dict[str, int] | None:
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def _median(values: list[int | Fraction]) -> Fraction | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return Fraction(ordered[middle]) if len(ordered) % 2 else (Fraction(ordered[middle - 1]) + Fraction(ordered[middle])) / 2


def _round_ns(value: Fraction | None) -> int | None:
    # Explicit half-up rule, preserving the unrounded rational alongside it.
    return None if value is None else (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _cv(values: list[int | float | Fraction]) -> float | None:
    if not values or statistics.mean(values) <= 0:
        return None
    return float(statistics.pstdev([float(item) for item in values]) / statistics.mean([float(item) for item in values]))


def _context(job: dict[str, Any]) -> dict[str, Any]:
    case_id, mode, instrumentation, telemetry = (job.get(name) for name in ("case_id", "mode", "instrumentation", "telemetry"))
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a nonempty string")
    if mode not in ("isolated", "mixed") or instrumentation not in ("stamped", "bare") or not isinstance(telemetry, bool):
        raise ValueError("invalid exact execution context")
    return {"case_id": case_id, "mode": mode, "instrumentation": instrumentation, "telemetry": telemetry}


def _key(context: dict[str, Any], condition: str, metric: str) -> str:
    return _canonical([context, condition, metric]).decode("utf-8")


def _rows(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = list(rows)
    # JSON roundtrip freezes caller-owned containers and rejects NaN/Infinity.
    rows = json.loads(_canonical(rows))
    identifiers = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("job"), dict) or not isinstance(row.get("summary"), dict):
            raise ValueError("each row requires job and summary dictionaries")
        job = row["job"]
        if job.get("split") != split:
            raise ValueError(f"{split}-only operation rejected split {job.get('split')!r}")
        if job.get("execution_boundary", "legacy_double_sync") != "legacy_double_sync":
            raise ValueError("nonlegacy execution boundaries are diagnostic-only and cannot enter profile/heldout model rows")
        identifier = job.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("job.id must be a nonempty string")
        if not isinstance(job.get("seed"), int) or isinstance(job["seed"], bool):
            raise ValueError("job.seed must be an integer")
        if not isinstance(job.get("repeat"), int) or isinstance(job["repeat"], bool) or job["repeat"] < 0:
            raise ValueError("job.repeat must be a nonnegative integer")
        identifiers.append(identifier)
        _context(job)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate job.id records are not independent processes")
    return rows


def _process(row: dict[str, Any], metric: str) -> dict[str, Any]:
    job, summary = row["job"], row["summary"]
    failures, quality = [], []
    process_id = summary.get("process_id", job.get("process_id"))
    if not _positive_int(process_id):
        failures.append("missing_or_invalid_process_id")
    if summary.get("native_numeric_passed") is not True:
        failures.append("native_numerical_verification_not_passed")
    if summary.get("condition_verified") is not True:
        failures.append("named_condition_not_verified")
    if job.get("condition") not in CONDITIONS:
        failures.append("unknown_named_condition")
    if job["telemetry"] and summary.get("alignment_model_ready") is not True:
        quality.append("telemetry_alignment_not_ready")
    samples = summary.get(f"{metric}_samples_ns")
    if not isinstance(samples, list) or not samples or any(not _positive_int(item) for item in samples):
        failures.append("missing_or_invalid_positive_integer_samples")
        samples = []
    median, cv = _median(samples), _cv(samples)
    declared_median = summary.get(f"{metric}_median_ns")
    if median is not None:
        if isinstance(declared_median, bool) or not isinstance(declared_median, (int, float)) or not math.isfinite(declared_median) or declared_median <= 0:
            failures.append("missing_or_invalid_declared_median")
        elif abs(Fraction(str(declared_median)) - median) > Fraction(1, 2):
            failures.append("declared_median_disagrees_with_samples")
    declared_cv = summary.get(f"{metric}_cv")
    if cv is not None:
        if isinstance(declared_cv, bool) or not isinstance(declared_cv, (float, int)) or not math.isfinite(declared_cv) or declared_cv < 0:
            failures.append("missing_or_invalid_declared_cv")
        elif abs(declared_cv - cv) > 1e-9:
            failures.append("declared_cv_disagrees_with_samples")
    if cv is None or cv > GATES["within_process_cv_max"]:
        quality.append("within_process_cv_gate_failed")
    return {"job_id": job["id"], "process_id": process_id, "seed": job["seed"], "repeat": job["repeat"],
            "condition": job.get("condition"), "samples_ns": samples,
            "sample_count": len(samples), "median_ns_exact": _fraction(median),
            "median_ns": _round_ns(median), "cv": cv,
            "valid": not failures, "validation_failures": failures, "quality_failures": quality,
            "row_sha256": _sha(row)}


def _aggregate(processes: list[dict[str, Any]]) -> dict[str, Any]:
    # No invalid process is removed to rescue a cell's validity or gates.
    failures = []
    if len(processes) < GATES["minimum_process_records_per_split"]:
        failures.append("fewer_than_three_process_records")
    if any(not item["valid"] for item in processes):
        failures.append("one_or_more_process_records_invalid")
    counts = [item["sample_count"] for item in processes]
    equal_count = bool(counts) and min(counts) > 0 and len(set(counts)) == 1
    if not equal_count:
        failures.append("unequal_or_missing_per_process_sample_counts")
    medians = [Fraction(item["median_ns_exact"]["numerator"], item["median_ns_exact"]["denominator"])
               for item in processes if item["median_ns_exact"] is not None]
    between_cv = _cv(medians) if len(medians) == len(processes) else None
    if between_cv is None or between_cv > GATES["between_process_cv_max"]:
        failures.append("between_process_cv_gate_failed")
    if any("within_process_cv_gate_failed" in item["quality_failures"] for item in processes):
        failures.append("one_or_more_within_process_cv_gates_failed")
    if any("telemetry_alignment_not_ready" in item["quality_failures"] for item in processes):
        failures.append("one_or_more_telemetry_alignments_not_ready")
    supported = bool(processes) and all(item["valid"] for item in processes)
    pooled = _median([sample for item in processes for sample in item["samples_ns"]]) if supported and equal_count else None
    median_of_medians = _median(medians) if supported and len(medians) == len(processes) else None
    price = pooled if pooled is not None else median_of_medians
    return {"process_count": len(processes), "processes": processes, "equal_sample_counts": equal_count,
            "samples_per_process": counts[0] if equal_count else None,
            "between_process_cv": between_cv, "supported": supported,
            "pooled_median_ns_exact": _fraction(pooled), "pooled_median_ns": _round_ns(pooled),
            "price_ns_exact": _fraction(price), "price_ns": _round_ns(price),
            "price_estimator": "equal_count_process_pooled_median" if pooled is not None else "diagnostic_median_of_process_medians" if price is not None else None,
            "gate_failures": failures, "accepted": supported and not failures}


def fit_model(profile_rows: Iterable[dict[str, Any]], source_bindings: dict[str, str],
              frozen_utc: str | None = None) -> dict[str, Any]:
    """Freeze prices from profile rows only; retain rejected cells and sources."""
    rows = _rows(profile_rows, "profile")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("nonempty source SHA-256 bindings required")
    for name, digest in source_bindings.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source bindings require nonempty names and lowercase SHA-256 hex")
    if frozen_utc is None:
        frozen_utc = datetime.now(timezone.utc).isoformat()
    try:
        freeze_time = datetime.fromisoformat(frozen_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("frozen_utc must be an ISO timestamp with timezone") from error
    if freeze_time.tzinfo is None:
        raise ValueError("frozen_utc must have timezone")
    families = {}
    for family in ("constant", "conditional"):
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            context = _context(row["job"])
            condition = "*" if family == "constant" else row["job"].get("condition")
            for metric in METRICS:
                key = _key(context, condition, metric)
                cell = grouped.setdefault(key, {"context": context, "condition": condition, "metric": metric, "processes": []})
                cell["processes"].append(_process(row, metric))
        cells = []
        for key in sorted(grouped):
            cell = grouped[key]
            processes = cell.pop("processes")
            cells.append({**cell, **_aggregate(processes)})
        families[family] = {"condition_policy": "pool_all_profile_conditions" if family == "constant" else "exact_named_condition_only", "cells": cells}
    result = {"schema": SCHEMA, "frozen_utc": frozen_utc, "gates": dict(GATES),
              "source_bindings": dict(sorted(source_bindings.items())),
              "model_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_rows_sha256": _sha(rows), "profile_job_ids": [row["job"]["id"] for row in rows],
              "profile_seeds": sorted(set(row["job"]["seed"] for row in rows)),
              "profile_row_count": len(rows), "families": families,
              "scope": "Exact measured operator, geometry, precision, execution context and externally verified named condition. Not a universal device-speed multiplier or automatic state classifier.",
              "fresh_process_evidence": "Unique runner job IDs and recorded process IDs; PID reuse across separate launches is permitted. Fresh launches and non-overlapping executions must be verified by the source-bound runner.",
              "cv_definition": "population_standard_deviation_divided_by_mean",
              "integer_price_rounding": "nearest_integer_nanosecond_half_up; unrounded rational retained",
              "heldout_used_for_fit": False, "outlier_removal": False,
              "interpolation": False, "unknown_state_fallback": False}
    result["model_sha256"] = _sha(result)
    return result


def _verify(model: dict[str, Any]) -> None:
    if model.get("schema") != SCHEMA or model.get("gates") != GATES:
        raise ValueError("unsupported model schema or modified acceptance gates")
    copy = {key: value for key, value in model.items() if key != "model_sha256"}
    if model.get("model_sha256") != _sha(copy):
        raise ValueError("frozen model content hash mismatch")


def _cell(model: dict[str, Any], context: dict[str, Any], condition: str, metric: str, family: str) -> dict[str, Any] | None:
    target = "*" if family == "constant" else condition
    return next((cell for cell in model["families"][family]["cells"] if cell["context"] == context and cell["condition"] == target and cell["metric"] == metric), None)


def lookup(model: dict[str, Any], case_id: str, mode: str, instrumentation: str,
           telemetry: bool, condition: str, metric: str = "target", family: str = "conditional") -> dict[str, Any]:
    """Return an exact-context measured price, including failed stability gates."""
    _verify(model)
    context = _context({"case_id": case_id, "mode": mode, "instrumentation": instrumentation, "telemetry": telemetry})
    if family not in ("constant", "conditional") or metric not in METRICS:
        raise ValueError("unknown model family or metric")
    if condition not in CONDITIONS:
        raise UnknownCost(f"unknown named condition {condition!r}; no fallback")
    observed_condition = _cell(model, context, condition, metric, "conditional")
    if observed_condition is None or not observed_condition["supported"]:
        raise UnknownCost(f"unobserved or invalid exact-context named condition {condition!r}; no fallback")
    cell = _cell(model, context, condition, metric, family)
    if cell is None or not cell["supported"] or cell["price_ns"] is None:
        raise UnknownCost(f"unsupported exact context/regime: {context}, {condition}, {metric}, {family}")
    reference = _cell(model, context, "baseline", metric, "conditional")
    rate = None
    if reference is not None and reference["supported"] and reference["price_ns_exact"] is not None:
        base = Fraction(**reference["price_ns_exact"])
        current = Fraction(**cell["price_ns_exact"])
        rate = _fraction(base / current)
    return {"family": family, "context": context, "condition": condition, "metric": metric,
            "price_ns": cell["price_ns"], "price_ns_exact": cell["price_ns_exact"],
            "profile_accepted": cell["accepted"], "profile_gate_failures": cell["gate_failures"],
            "reference_price_ns": reference["price_ns"] if reference is not None and reference["supported"] else None,
            "reference_service_rate": rate,
            "reference_service_rate_unknown_reason": None if rate is not None else "No supported exact-context baseline reference",
            "rate_scope": "Reference-work nanoseconds per elapsed nanosecond for this exact measured operator and execution context only; no universal CPU/GPU multiplier.",
            "model_sha256": model["model_sha256"]}


def evaluate(model: dict[str, Any], heldout_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a frozen model without changing prices, gates or source rows."""
    _verify(model)
    rows = _rows(heldout_rows, "heldout")
    if set(model["profile_job_ids"]) & {row["job"]["id"] for row in rows}:
        raise ValueError("profile and heldout job IDs overlap")
    if set(model["profile_seeds"]) & {row["job"]["seed"] for row in rows}:
        raise ValueError("profile and heldout numerical seed sets overlap")
    grouped = {}
    for row in rows:
        context, condition = _context(row["job"]), row["job"].get("condition")
        for metric in METRICS:
            key = _key(context, condition, metric)
            cell = grouped.setdefault(key, {"context": context, "condition": condition, "metric": metric, "processes": []})
            cell["processes"].append(_process(row, metric))
    # Missing heldout cells must not disappear from reported coverage.
    for cell in model["families"]["conditional"]["cells"]:
        key = _key(cell["context"], cell["condition"], cell["metric"])
        grouped.setdefault(key, {"context": cell["context"], "condition": cell["condition"], "metric": cell["metric"], "processes": []})
    families = {}
    for family in ("constant", "conditional"):
        results = []
        for key in sorted(grouped):
            group = grouped[key]
            aggregate = _aggregate(group["processes"])
            context, condition, metric = group["context"], group["condition"], group["metric"]
            source = _cell(model, context, condition, metric, family)
            observed_condition = _cell(model, context, condition, metric, "conditional")
            failures = [f"heldout:{reason}" for reason in aggregate["gate_failures"]]
            unknown = None
            if condition not in CONDITIONS:
                unknown = "Unknown named heldout condition; no fallback"
            elif observed_condition is None or not observed_condition["supported"]:
                unknown = "Named heldout condition was not supported in this exact profile context; no fallback"
            elif source is None or not source["supported"] or source["price_ns_exact"] is None:
                unknown = "No supported exact-context profile price"
            if unknown is not None:
                failures.append("profile_price_unknown")
                prediction, error = None, None
            else:
                failures.extend(f"profile:{reason}" for reason in source["gate_failures"])
                prediction = Fraction(**source["price_ns_exact"])
                observed = aggregate["pooled_median_ns_exact"]
                error = float(abs(prediction - Fraction(**observed)) / Fraction(**observed)) if observed is not None else None
            if source is not None and source["samples_per_process"] != aggregate["samples_per_process"]:
                failures.append("profile_heldout_sample_counts_differ")
            if error is None or error > GATES["pooled_median_relative_error_max"]:
                failures.append("pooled_median_error_gate_failed_or_unknown")
            per_process_errors = []
            for item in aggregate["processes"]:
                median = item["median_ns_exact"]
                per_process_errors.append({"job_id": item["job_id"], "relative_error": float(abs(prediction - Fraction(**median)) / Fraction(**median)) if prediction is not None and median is not None and item["valid"] else None})
            results.append({"context": context, "condition": condition, "metric": metric,
                            "prediction_ns": _round_ns(prediction), "prediction_ns_exact": _fraction(prediction),
                            "profile_process_count": source["process_count"] if source is not None else 0,
                            "heldout": aggregate, "relative_error": error,
                            "per_process_errors": per_process_errors, "unknown_reason": unknown,
                            "gate_failures": failures, "accepted": not failures})
        errors = [item["relative_error"] for item in results if item["relative_error"] is not None]
        families[family] = {"cells": results, "cell_count": len(results),
                            "accepted_count": sum(item["accepted"] for item in results),
                            "known_error_count": len(errors), "unknown_error_count": len(results) - len(errors),
                            "median_relative_error_known_only": statistics.median(errors) if errors else None,
                            "worst_relative_error_known_only": max(errors) if errors else None,
                            "all_cells_accepted": bool(results) and all(item["accepted"] for item in results),
                            "aggregate_scope": "Known-error summaries are descriptive only; unknown/failed cells remain in the denominator and acceptance records."}
    result = {"schema": EVALUATION_SCHEMA, "model_sha256": model["model_sha256"],
              "model_frozen_utc": model["frozen_utc"], "heldout_rows_sha256": _sha(rows),
              "heldout_job_ids": [row["job"]["id"] for row in rows],
              "heldout_seeds": sorted(set(row["job"]["seed"] for row in rows)),
              "gates": dict(GATES), "families": families, "heldout_used_for_fit": False,
              "all_requested_scopes_accepted": bool(rows) and all(item["all_cells_accepted"] for item in families.values()),
              "causal_device_slowdown_proven": False, "universal_device_multiplier": False}
    result["evaluation_sha256"] = _sha(result)
    return result
