"""The single summation rule: table prices plus declared terms make one step.

A token step is::

    embedding lookup  ->  declared block x layers  ->  final norm + LM head

so the price of the step is the block sum times the layer count plus the two
boundary actions.  That expression is the whole rule, and it lives here so that
a declared term cannot be added anywhere else.

The zero-term case is bit-for-bit the historical expression.  It is not a
special case for compatibility: with nothing declared there is nothing to round,
so the integer arithmetic must not acquire a float round-trip.  Every archived
comparison depends on that, and ``composing/tests/test_compose.py`` pins it.
"""

from __future__ import annotations

from .terms import UNIT_FRACTION_OF_BLOCK_STACK, UNIT_PER_LAYER, UNIT_PER_TOKEN, apply_to


def compose_with_breakdown(*, block_sum_ns: int, layers: int, boundary_ns: int,
                           phase: str, tokens: int, terms=()) -> tuple[int, list[dict]]:
    """``(total_ns, applied terms)`` for one token step of ``phase``.

    ``block_sum_ns`` is one declared block at this shape (the sum of its stages,
    never the joint ``block`` record); ``boundary_ns`` is the embedding plus the
    head.  Terms act only where their unit says they act.
    """
    if layers < 1:
        raise ValueError(f"a step has at least one declared block, got {layers}")
    if tokens < 1:
        raise ValueError(f"a shape has at least one token, got {tokens}")
    base = block_sum_ns * layers + boundary_ns
    applied = apply_to(phase, terms)
    if not applied:
        # Nothing declared: integer arithmetic, no rounding, identical to the
        # historical sum on every archived shape.
        return base, []
    extra = 0.0
    breakdown = []
    stack = block_sum_ns * layers
    for term in applied:
        if term.unit == UNIT_PER_TOKEN:
            contribution = term.value * tokens
        elif term.unit == UNIT_PER_LAYER:
            contribution = term.value * layers
        elif term.unit == UNIT_FRACTION_OF_BLOCK_STACK:
            # A share of the block stack that does not serialize.  It can only
            # cancel per-layer work: the boundary actions run once and are not
            # what overlaps with anything.
            contribution = -term.value * stack
        else:  # pragma: no cover - terms validate their unit at construction
            raise ValueError(f"{term.name}: unknown unit {term.unit!r}")
        extra += contribution
        breakdown.append({"name": term.name, "phase": term.phase, "unit": term.unit,
                          "value": term.value, "contribution_ns": contribution})
    return int(round(base + extra)), breakdown


def compose_total(**kwargs) -> int:
    """The composed price alone; see :func:`compose_with_breakdown`."""
    return compose_with_breakdown(**kwargs)[0]
