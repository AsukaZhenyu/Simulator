"""Optional preceding-service-conditioned responses for serial cyclic replay.

The cost remains an independently measured inclusive action response. A context
is a previous physical service's semantics, never a strategy/program identity,
layer index, numerical seed, or a complete target's observed time. This is an
empirical response model, not a causal identification of cache or runtime state.
Untested contexts are rejected rather than replaced with an isolated price.
"""
from dataclasses import dataclass
from fractions import Fraction

from .reusable_action_costs import ReusableResponseTable, digest, semantic_action


def context_coordinate(program, index, context):
    """Describe a physical predecessor in a warm fixed-input cyclic sequence."""
    action = program["actions"][index]
    previous = None
    if action["kind"] in {"COPY", "COMPUTE"}:
        count = len(program["actions"])
        for offset in range(1, count + 1):
            j = (index - offset) % count
            if program["actions"][j]["kind"] in {"COPY", "COMPUTE"}:
                previous = semantic_action(program, j)
                break
    return dict(schema="preceding-service-response-coordinate-1",
                semantic=semantic_action(program, index),
                previous_service=previous, context=context)


@dataclass(frozen=True)
class ContextResponseTable:
    entries: tuple[dict, ...]

    def __post_init__(self):
        # Reuse strict duplicate, evidence and nonnegative-price validation.
        ReusableResponseTable(self.entries)
        for entry in self.entries:
            if entry["coordinate"].get("schema") != "preceding-service-response-coordinate-1":
                raise ValueError("context-conditioned response schema differs")

    def evaluate(self, program, context):
        prices = {entry["key"]: entry for entry in self.entries}
        values = []
        for i in range(len(program["actions"])):
            coord = context_coordinate(program, i, context)
            entry = prices.get(digest(coord))
            if entry is None or entry["coordinate"] != coord:
                raise KeyError("unmeasured preceding-service response: " + digest(coord))
            values.append(Fraction(entry["mean_ns"]))
        return dict(duration_ns=sum(values, Fraction()), action_ns=tuple(values),
                    composition="serial sum of preceding-service-conditioned responses",
                    transition_cost_added_separately=False, target_outcome_used=False,
                    causal_mechanism_identified=False,
                    scope="measured warm cyclic fixed-input serial contexts only")
