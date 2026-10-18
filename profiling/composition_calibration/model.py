"""Profile-only effective composition corrections for retained batch timings.

This is an empirical performance model, not an identification of physical
context-switch latency. It never reads files, runs hardware, or fits heldout
rows. Complete *profile* strategy totals are calibration targets. Callers must
bind the exact case geometry and measurement protocol before selecting prices.
The boundary feature is a fixed critical-path stage-boundary proxy, not a count
of all graph edges or physical device switches.

Public durations are microseconds. Retained row metrics are normalized integer
nanoseconds per completed invocation, as recorded by batched_calibration/run.py.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import itertools
import json
import math
import re
import statistics


from llm_infer_model.performance.legacy_composition import (
    SCHEMA, FEATURE_RECIPE_VERSION, FEATURE_NAMES, FEATURE_UNITS,
    _STAGES, _CASE_FIELDS, _BOUNDARIES, _case, primary_strategies,
    recipe, _terms, component_metrics, _number, _evaluate, _features,
)

RANK_TOLERANCE = 1e-12


def _canonical_hash(value):
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ValueError("training rows must be finite JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def _samples(values, label):
    if (not isinstance(values, list) or len(values) < 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values)):
        raise ValueError("at least three positive integer ns samples required: " + label)
    return values


def _profile_rows(rows, *, targets):
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("a nonempty iterable of profile process rows is required")
    try:
        rows = list(rows)
    except TypeError as exc:
        raise ValueError("profile process rows are required") from exc
    if not rows:
        raise ValueError("profile absence: at least one profile process is required")
    # Reject the entire input before reading any metric from a nonprofile row.
    if any(not isinstance(row, Mapping) or row.get("split") != "profile" for row in rows):
        raise ValueError("profile-only input required; heldout/unknown splits are forbidden")
    seen, cases, repetitions = set(), {}, set()
    for row in rows:
        if row.get("status") != "measured":
            raise ValueError("failed or unmeasured profile process: " + str(row.get("id")))
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("unique explicit profile row IDs are required")
        seen.add(rid)
        case = _case(row.get("case"))
        if case["id"] in cases and cases[case["id"]] != case:
            raise ValueError("case ID is bound to conflicting geometry: " + case["id"])
        cases[case["id"]] = dict(case)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("raw_sha256", ""))):
            raise ValueError("profile raw-record SHA256 is required: " + rid)
        batch = row.get("batch_repetitions")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("explicit positive batch_repetitions required: " + rid)
        repetitions.add(batch)
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("profile component metrics are required: " + rid)
        required = set(component_metrics(case))
        if targets:
            required.update(primary_strategies(case))
        for name in sorted(required):
            if name not in metrics:
                raise ValueError("missing profile metric: " + rid + ": " + name)
            _samples(metrics[name], rid + ": " + name)
    if len(repetitions) != 1:
        raise ValueError("mixed batch protocols require separate calibration models")
    return rows


def pooled_component_prices(profile_rows):
    """Pool independent profile samples by case, without reading target samples.

    This helper validates source status/provenance and independent components;
    unlike fit, it does not require or inspect complete-strategy target metrics.
    The caller is responsible for exact protocol and shape lookup.
    """
    rows = _profile_rows(profile_rows, targets=False)
    result = {}
    for cid in sorted({r["case"]["id"] for r in rows}):
        selected = [r for r in rows if r["case"]["id"] == cid]
        result[cid] = {
            name: _number(statistics.median([x for r in selected for x in r["metrics"][name]])
                          / 1000.0, name, positive=True)
            for name in component_metrics(selected[0]["case"])
        }
    return result


def _dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _least_squares(columns, target):
    """Scaled, twice-orthogonalized QR; return None for a dependent subset."""
    scales = [math.hypot(*column) for column in columns]
    if any(not math.isfinite(s) or s == 0 for s in scales):
        return None
    normalized = [[v / s for v in col] for col, s in zip(columns, scales)]
    q, upper = [], [[0.0] * len(columns) for _ in columns]
    for j, column in enumerate(normalized):
        work = list(column)
        for _ in range(2):
            for i, previous in enumerate(q):
                projection = _dot(previous, work)
                upper[i][j] += projection
                work = [v - projection * qi for v, qi in zip(work, previous)]
        diagonal = math.hypot(*work)
        if diagonal <= RANK_TOLERANCE:
            return None
        upper[j][j] = diagonal
        q.append([v / diagonal for v in work])
    rhs = [_dot(column, target) for column in q]
    scaled_solution = [0.0] * len(columns)
    for i in reversed(range(len(columns))):
        scaled_solution[i] = (rhs[i] - math.fsum(upper[i][j] * scaled_solution[j]
                                                for j in range(i + 1, len(columns)))) / upper[i][i]
    solution = [v / s for v, s in zip(scaled_solution, scales)]
    diagonals = [upper[i][i] for i in range(len(columns))]
    return solution, max(diagonals) / min(diagonals)


def _nnls(design, target):
    """Enumerate all 2**3 faces of the nonnegative cone, with no dependencies."""
    columns = [list(column) for column in zip(*design)]
    best, best_loss, best_subset, best_ratio = [0.0] * 3, _dot(target, target), (), None
    rank, rejected = 0, []
    for size in range(1, 4):
        for subset in itertools.combinations(range(3), size):
            solved = _least_squares([columns[i] for i in subset], target)
            if solved is None:
                rejected.append([FEATURE_NAMES[i] for i in subset])
                continue
            rank = max(rank, size)
            values, ratio = solved
            tolerance = 1e-10 * max(1.0, *(abs(v) for v in values))
            if any(not math.isfinite(v) or v < -tolerance for v in values):
                continue
            candidate = [0.0] * 3
            for index, value in zip(subset, values):
                candidate[index] = max(0.0, value)
            residuals = [_dot(row, candidate) - value for row, value in zip(design, target)]
            loss = _dot(residuals, residuals)
            tied = math.isclose(loss, best_loss, rel_tol=1e-12, abs_tol=1e-24)
            if (loss < best_loss and not tied) or (tied and (len(subset), subset) < (len(best_subset), best_subset)):
                best, best_loss, best_subset, best_ratio = candidate, loss, subset, ratio
    return best, {
        "solver": "enumerated_nonnegative_active_sets_scaled_reorthogonalized_qr",
        "objective": "sum(((correction_us - signed_residual_us) / profile_total_us)**2)",
        "weight_rule": "row multiplier = 1 / profile_total_us",
        "squared_relative_error_objective": best_loss,
        "weighted_design_rank": rank,
        "feature_count": 3,
        "rank_deficient": rank < 3,
        "rank_tolerance_after_column_scaling": RANK_TOLERANCE,
        "zero_feature_columns": [FEATURE_NAMES[i] for i, column in enumerate(columns) if not any(column)],
        "rank_rejected_active_sets": rejected,
        "active_features": [FEATURE_NAMES[i] for i in best_subset if best[i] > 0],
        "selected_scaled_qr_diagonal_ratio": best_ratio,
        "conditioning_note": "QR diagonal ratio is a diagnostic, not a singular-value condition number; near-dependent active sets are rejected",
        "tie_rule": "equal objective within 1e-12 relative / 1e-24 absolute: fewer active features, then feature order",
    }


def fit(profile_rows):
    """Fit three global nonnegative coefficients; reject every nonprofile row.

    Every primary target uses the same process's component metric medians.
    Negative target-minus-recipe residuals remain signed observations in the
    least-squares objective; they are neither discarded nor clamped to zero.
    """
    rows = sorted(_profile_rows(profile_rows, targets=True), key=lambda row: row["id"])
    observations, sources = [], []
    for row in rows:
        case = row["case"]
        row_sha = _canonical_hash(row)
        sources.append({"row_id": row["id"], "row_sha256": row_sha,
                        "raw_sha256": row["raw_sha256"], "raw_path": row.get("raw_path"),
                        "case_id": case["id"], "seed": row.get("seed"), "repeat": row.get("repeat")})
        prices = {name: statistics.median(row["metrics"][name]) / 1000.0
                  for name in component_metrics(case)}
        for strategy in primary_strategies(case):
            observed = _number(statistics.median(row["metrics"][strategy]) / 1000.0,
                               strategy, positive=True)
            baseline = _evaluate(recipe(case, strategy), prices)
            observations.append({
                "row_id": row["id"], "row_sha256": row_sha, "raw_sha256": row["raw_sha256"],
                "case_id": case["id"], "strategy": strategy, "target_metric": strategy,
                "profile_total_us": observed, "row_component_baseline_us": baseline,
                "signed_residual_us": observed - baseline,
                "row_multiplier": 1.0 / observed, "features": _features(case, strategy, prices),
            })
    design = [[observation["features"][name] * observation["row_multiplier"] for name in FEATURE_NAMES]
              for observation in observations]
    target = [observation["signed_residual_us"] * observation["row_multiplier"] for observation in observations]
    coefficients, diagnostics = _nnls(design, target)
    for observation in observations:
        correction = _dot([observation["features"][name] for name in FEATURE_NAMES], coefficients)
        observation["fitted_correction_us"] = correction
        observation["remaining_signed_residual_us"] = observation["signed_residual_us"] - correction
    diagnostics.update(observation_count=len(observations), process_count=len(rows),
                       negative_residual_count=sum(o["signed_residual_us"] < 0 for o in observations),
                       sample_filtering="none", negative_residual_policy="retained_signed_in_objective")
    return {
        "schema": SCHEMA, "feature_recipe_version": FEATURE_RECIPE_VERSION,
        "feature_names": list(FEATURE_NAMES), "feature_units": dict(FEATURE_UNITS),
        "coefficients": dict(zip(FEATURE_NAMES, coefficients)),
        "coefficient_units": {FEATURE_NAMES[0]: "us/boundary", FEATURE_NAMES[1]: "dimensionless",
                              FEATURE_NAMES[2]: "us/preload_event"},
        "feature_recipe": {
            "version": FEATURE_RECIPE_VERSION,
            "composition_boundaries": {family: dict(counts) for family, counts in _BOUNDARIES.items()},
            "composition_boundaries_semantics": "fixed critical-path stage-boundary proxy: GPU five ordered stages -> 4; hybrid attention six stages -> 5; hybrid FFN four-stage critical branch plus D2H plus sum -> 5; preload three ordered copies plus resident path -> 8; not total graph edges or physical switches",
            "hybrid_ffn_critical_branch_us": "max(sum(cpu/shard_gate,up,activation,down), sum(gpu/shard_gate,up,activation,down)); zero for other paths",
            "preload_event": "1 only for hybrid/preload FFN, otherwise 0",
            "intercept": "none", "case_or_path_specific_offsets": False,
            "prediction": "original component sum/max recipe + dot(features, nonnegative shared coefficients)",
            "recipes": {family: {strategy: recipe(next(r["case"] for r in rows if r["case"]["family"] == family), strategy)
                                  for strategy in _BOUNDARIES[family]}
                        for family in sorted({r["case"]["family"] for r in rows})},
        },
        "interpretation": "effective empirical composition correction; not physically identified context-switch latency",
        "evaluation_contract": "freeze implementation, feature recipe and fitted model before retrospective heldout evaluation; do not tune from heldout results",
        "trained_on": "profile_only", "profile_complete_strategy_totals_used_for_calibration": True,
        "heldout_used_for_calibration": False, "accepted_capacity_costs": False,
        "gpu_capacity_proved": False, "timing_unit": "microseconds_per_completed_invocation_within_batch",
        "batch_repetitions": rows[0]["batch_repetitions"],
        "target_statistics": "median of each metric within each profile process; components evaluated before subtraction",
        "component_pooling": "median of all retained profile component samples for the exact case, divided by 1000",
        "profile_cases": {r["case"]["id"]: dict(r["case"]) for r in rows},
        "profile_component_prices_us": pooled_component_prices(rows),
        "training": {"rows": sources, "rows_sha256": _canonical_hash(sources),
                     "target_metrics": sorted({o["target_metric"] for o in observations}),
                     "observations": observations},
        "fit_diagnostics": diagnostics,
    }


def predict(case, strategy, component_prices, model):
    """Compatibility; frozen composition equations are owned by modeling."""
    from llm_infer_model.performance.legacy_composition import predict as calculate
    return calculate(case, strategy, component_prices, model)
