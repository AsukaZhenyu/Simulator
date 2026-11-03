"""P1/P2 verdict tool for offline predicted-versus-measured comparisons."""
from __future__ import annotations

from verdict.p1 import CONFIRMED, INCONCLUSIVE, NOT_CONFIRMED, analyze as analyze_crossover
from verdict.p2 import analyze as analyze_ranking
from verdict.schema import (SCHEMA, ValidationError, load, order, relative_gap,
                            validate, validate_tolerance)

OUTPUT_SCHEMA = "verdict-output-1"

__all__ = [
    "CONFIRMED",
    "INCONCLUSIVE",
    "NOT_CONFIRMED",
    "OUTPUT_SCHEMA",
    "SCHEMA",
    "ValidationError",
    "analyze_crossover",
    "analyze_ranking",
    "load",
    "order",
    "relative_gap",
    "validate",
    "validate_tolerance",
]
