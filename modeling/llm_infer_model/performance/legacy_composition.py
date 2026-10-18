"""Frozen effective composition model calculation and applicability contract.

This is the retained empirical three-feature correction, not a claim that its
coefficients identify physical context-switch latency. Profiling estimates its
coefficients and component prices; modeling owns recipes, features, validation
and execution-time calculation. No raw samples, medians, or fitting occur here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


SCHEMA = "effective-composition-overhead-model-0.1"
FEATURE_RECIPE_VERSION = "serial-boundary-critical-branch-preload-0.1"
FEATURE_NAMES = (
    "composition_boundaries",
    "hybrid_ffn_critical_branch_us",
    "preload_event",
)
FEATURE_UNITS = {
    "composition_boundaries": "count",
    "hybrid_ffn_critical_branch_us": "microseconds",
    "preload_event": "count",
}
_STAGES = ("gate", "up", "activation", "down")
_CASE_FIELDS = {
    "id", "family", "backend", "dtype", "hidden", "intermediate",
    "cpu_intermediate", "tokens", "history", "heads", "kv_heads", "head_dim",
}
_BOUNDARIES = {
    "ffn": {"gpu/resident": 4, "hybrid/resident": 5, "hybrid/preload": 8},
    "attention": {"gpu/resident": 4, "hybrid/resident": 5},
}


def _case(case):
    if not isinstance(case, Mapping) or set(case) != _CASE_FIELDS:
        raise ValueError("explicit retained case geometry fields are required")
    if (not isinstance(case["id"], str) or not case["id"]
            or case["family"] not in _BOUNDARIES
            or case["backend"] != "hybrid" or case["dtype"] != "f32"):
        raise ValueError("only explicit retained F32 FFN/attention cases are supported")
    for key in _CASE_FIELDS - {"id", "family", "backend", "dtype"}:
        if isinstance(case[key], bool) or not isinstance(case[key], int) or case[key] < 0:
            raise ValueError("nonnegative integer case geometry required: " + key)
    if not case["hidden"] or not case["tokens"]:
        raise ValueError("hidden and tokens must be positive")
    if case["family"] == "ffn":
        if (not 0 < case["cpu_intermediate"] < case["intermediate"]
                or any(case[k] for k in ("history", "heads", "kv_heads", "head_dim"))):
            raise ValueError("FFN requires nonempty CPU/GPU partitions and no attention geometry")
    elif (case["intermediate"] or case["cpu_intermediate"] or not case["heads"]
          or not case["kv_heads"] or not case["head_dim"]
          or case["heads"] * case["head_dim"] != case["hidden"]
          or case["heads"] % case["kv_heads"]):
        raise ValueError("invalid explicit attention geometry")
    return case


def primary_strategies(case):
    """Return only the fixed primary paths, without dispatch-model alternatives."""
    return tuple(_BOUNDARIES[_case(case)["family"]])


def recipe(case, strategy):
    """Return the original sum/max recipe, using three independent H2D probes."""
    family = _case(case)["family"]
    if strategy not in _BOUNDARIES[family]:
        raise ValueError("unsupported primary strategy: " + str(strategy))
    if family == "ffn":
        if strategy == "gpu/resident":
            return {"sum": [*["gpu/full_" + x for x in _STAGES], "gpu/output_d2h"]}
        mixed = {"sum": [
            {"max": [{"sum": ["cpu/shard_" + x for x in _STAGES]},
                     {"sum": ["gpu/shard_" + x for x in _STAGES]}]},
            "hybrid/output_d2h", "cpu/sum",
        ]}
        if strategy == "hybrid/preload":
            return {"sum": [
                {"sum": ["hybrid/preload_" + x + "_weight_h2d"
                         for x in ("down", "gate", "up")]}, mixed,
            ]}
        return mixed
    if strategy == "gpu/resident":
        return {"sum": ["gpu/qk", "gpu/softmax", "gpu/pv", "gpu/output", "gpu/output_d2h"]}
    return {"sum": ["cpu/qk", "cpu/softmax", "hybrid/probabilities_h2d",
                    "gpu/pv", "gpu/output", "gpu/output_d2h"]}


def _terms(expression):
    if isinstance(expression, str):
        return {expression}
    return set().union(*(_terms(child) for children in expression.values() for child in children))


def component_metrics(case):
    return tuple(sorted(set().union(*(_terms(recipe(case, s)) for s in primary_strategies(case)))))


def _number(value, label, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite numeric value required: " + label) from exc
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(result) or result < 0 or (positive and result == 0)):
        raise ValueError("finite nonnegative value required: " + label)
    return result


def _evaluate(expression, prices):
    if isinstance(expression, str):
        if expression not in prices:
            raise ValueError("missing component price: " + expression)
        return _number(prices[expression], expression, positive=True)
    operation, children = next(iter(expression.items()))
    values = [_evaluate(child, prices) for child in children]
    value = math.fsum(values) if operation == "sum" else max(values)
    return _number(value, "component recipe", positive=True)


def _features(case, strategy, prices):
    family = case["family"]
    critical = 0.0
    if family == "ffn" and strategy.startswith("hybrid/"):
        critical = max(_evaluate({"sum": [device + "/shard_" + x for x in _STAGES]}, prices)
                       for device in ("cpu", "gpu"))
    return dict(zip(FEATURE_NAMES, (_BOUNDARIES[family][strategy], critical,
                                  int(strategy == "hybrid/preload"))))


def predict(case, strategy, component_prices, model):
    """Account for baseline plus effective correction, using explicit us prices.

    No observed complete-strategy duration is consulted. Missing components or
    unsupported paths raise ValueError, allowing a caller to retain an unknown
    prediction instead of substituting an unrelated case or protocol.
    """
    expression = recipe(case, strategy)
    if not isinstance(component_prices, Mapping):
        raise ValueError("explicit component prices in microseconds are required")
    if (not isinstance(model, Mapping) or model.get("schema") != SCHEMA
            or model.get("feature_recipe_version") != FEATURE_RECIPE_VERSION
            or model.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("unsupported effective composition model or feature recipe")
    values = model.get("coefficients")
    if not isinstance(values, Mapping) or set(values) != set(FEATURE_NAMES):
        raise ValueError("exactly three shared feature coefficients are required")
    coefficients = [_number(values[name], name) for name in FEATURE_NAMES]
    baseline = _evaluate(expression, component_prices)
    features = _features(case, strategy, component_prices)
    contributions = {name: features[name] * coefficient for name, coefficient in zip(FEATURE_NAMES, coefficients)}
    correction = _number(math.fsum(contributions.values()), "effective correction")
    total = _number(baseline + correction, "prediction", positive=True)
    return {"baseline_us": baseline, "correction_us": correction, "total_us": total,
            "baseline_ns": baseline * 1000.0, "correction_ns": correction * 1000.0,
            "predicted_ns": total * 1000.0,
            "features": features, "feature_units": dict(FEATURE_UNITS),
            "correction_contributions_us": contributions,
            "recipe": expression, "feature_recipe_version": FEATURE_RECIPE_VERSION}
