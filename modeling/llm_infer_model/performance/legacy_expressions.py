"""Declared sum/max execution expressions; no measurements or estimators.

This preserves the historical execution calibration arithmetic. Prices are
explicit model inputs; source selection, medians and quality gates are owned
by profiling, not by this evaluator.
"""
from __future__ import annotations


def terms(expression):
    if isinstance(expression, str) and expression:
        return {expression}
    if not isinstance(expression, dict) or len(expression) != 1:
        raise ValueError("expression requires a metric or a single sum/max node")
    operation, children = next(iter(expression.items()))
    if operation not in ("sum", "max") or not isinstance(children, list) or not children:
        raise ValueError("sum/max requires explicit nonempty children")
    return set().union(*(terms(child) for child in children))


def predict(expression, prices):
    if isinstance(expression, str):
        return prices[expression]
    terms(expression)
    operation, children = next(iter(expression.items()))
    values = [predict(child, prices) for child in children]
    return sum(values) if operation == "sum" else max(values)


def expression_text(expression):
    if isinstance(expression, str):
        return expression
    operation, children = next(iter(expression.items()))
    formatted = [expression_text(child) for child in children]
    return "(" + " + ".join(formatted) + ")" if operation == "sum" else "max(" + ", ".join(formatted) + ")"
