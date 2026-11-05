"""The quantities a cost table does not price, declared before they are added.

A term is a *measured derivative* with units, not a correction factor.  Three
rules decide what may become one, and all three are enforced here rather than
written in a document:

1. **Undeclared is not added.**  The composition is exactly what
   :func:`composing.compose_total` is given.  Nothing is inferred from an
   end-to-end error and nothing is silently absorbed into a stage.
2. **Every term names the knob that measures it.**  A quantity with no
   executable knob is not a term yet -- it is a hypothesis.  :data:`KNOWN_TERMS`
   is the closed set; :class:`Term` refuses a name outside it.
3. **A term is off until it is measured.**  ``enabled`` defaults to ``False``,
   so declaring a quantity changes no number.  The prediction records both the
   declared and the applied set, so a later reader can tell "not priced" from
   "priced at zero".

The unit is part of the term's identity (it comes from the registry, not from
the caller): ``ns_per_token`` and ``ns_per_layer`` are different quantities, and
a value moved from one to the other silently changes every shape it touches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# How a term scales with one token step.  The unit decides the multiplication in
# :func:`composing.compose_total`; it is never chosen by the caller.  Only units
# a registered term actually uses are listed: an unused unit would be a branch
# no measurement could ever exercise.
UNIT_PER_TOKEN = "ns_per_token"            # per batch token of this step
UNIT_PER_LAYER = "ns_per_layer"            # per declared decoder block
UNIT_FRACTION_OF_BLOCK_STACK = "fraction_of_block_stack"  # subtracted share
UNITS = (UNIT_PER_TOKEN, UNIT_PER_LAYER, UNIT_FRACTION_OF_BLOCK_STACK)


@dataclass(frozen=True)
class TermSpec:
    """A quantity that may be declared, and the measurement that would give it.

    ``knob`` has to name a command or a comparison that exists.  A knob that
    cannot be run is the same as no knob: the quantity would then be fitted from
    the very error it is supposed to explain.
    """

    name: str
    unit: str
    phases: tuple[str, ...]
    knob: str
    quantity: str


# The four candidate gaps named in the accuracy plan (PLAN §0.2).  This is the
# whole set: adding a name here is a claim that a knob can measure it.
KNOWN_TERMS: dict[str, TermSpec] = {
    "per_layer_marginal": TermSpec(
        name="per_layer_marginal",
        unit=UNIT_PER_LAYER,
        phases=("prefill", "decode"),
        knob="dense_layer_probe --repeat-layers N: the slope of the declared-block "
             "loop against N, isolated cost against in-situ cost",
        quantity="what one more declared block costs inside a running stack, "
                 "after the isolated probe's per-call fixed cost is amortized away",
    ),
    "per_token_fixed": TermSpec(
        name="per_token_fixed",
        unit=UNIT_PER_TOKEN,
        phases=("prefill",),
        knob="prompt-token sweep at fixed depth (-p 128/256/512): the measured "
             "per-token slope minus what the table's per-token stages already price",
        quantity="work every prompt token pays regardless of its position, "
                 "which a per-token price key cannot express",
    ),
    "kv_append_per_token": TermSpec(
        name="kv_append_per_token",
        unit=UNIT_PER_TOKEN,
        phases=("decode",),
        knob="KV element type (-ctk/-ctv f16 against f32): the decode slope "
             "against bytes appended per token",
        quantity="what writing one token's K/V costs, a write the declared "
                 "attention stage reads but never performs",
    ),
    "stage_overlap": TermSpec(
        name="stage_overlap",
        unit=UNIT_FRACTION_OF_BLOCK_STACK,
        phases=("prefill",),
        knob="-ub micro-batch and -t thread count: the share of block time that "
             "disappears when the serialized stages overlap inside a batch",
        quantity="the fraction of the summed block stack that does not serialize, "
                 "because adding prices in a table is not adding work on a device",
    ),
}


@dataclass(frozen=True)
class Term:
    """One declared quantity at one phase, with the value a knob gave it.

    ``value`` is a measurement, never a residual: if no knob has been run, the
    value is unknown and the term stays disabled rather than defaulting to zero.
    """

    name: str
    phase: str
    value: float
    enabled: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        spec = spec_of(self.name)
        if self.phase not in spec.phases:
            raise ValueError(f"{self.name} is a {spec.phases} quantity, not {self.phase!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError(f"{self.name}.value must be a number, got {self.value!r}")
        if not math.isfinite(self.value):
            raise ValueError(f"{self.name}.value must be finite, got {self.value!r}")
        if self.value < 0:
            raise ValueError(f"{self.name}.value is a cost and cannot be negative: "
                             f"{self.value!r}")
        if spec.unit == UNIT_FRACTION_OF_BLOCK_STACK and self.value > 1:
            raise ValueError(f"{self.name}.value is a fraction of the block stack, "
                             f"so it cannot exceed 1: {self.value!r}")

    @property
    def spec(self) -> TermSpec:
        return KNOWN_TERMS[self.name]

    @property
    def unit(self) -> str:
        return self.spec.unit

    def to_document(self) -> dict:
        return {"name": self.name, "phase": self.phase, "value": self.value,
                "unit": self.unit, "enabled": self.enabled, "note": self.note,
                "knob": self.spec.knob, "quantity": self.spec.quantity}


def spec_of(name: str) -> TermSpec:
    """The registry entry for ``name``; an unknown name is a refusal, not a guess."""
    if not isinstance(name, str) or name not in KNOWN_TERMS:
        raise ValueError(f"no declared quantity named {name!r}; declared quantities are "
                         f"{sorted(KNOWN_TERMS)}. A quantity with no knob is a hypothesis, "
                         f"not a term -- add it to KNOWN_TERMS together with its knob")
    return KNOWN_TERMS[name]


def apply_to(phase: str, terms) -> tuple[Term, ...]:
    """The enabled terms that act on ``phase``, in declaration order."""
    return tuple(term for term in terms if term.enabled and term.phase == phase)


def document_terms(terms) -> list[dict]:
    """Every declared term, applied or not, for the prediction's own record."""
    return [term.to_document() for term in terms]


def terms_from_document(items) -> tuple[Term, ...]:
    """Rebuild declared terms from a prediction document.

    A prediction written before terms existed carries none; that is zero
    declared terms, not a missing field.
    """
    if not items:
        return ()
    terms = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"composition_terms must be objects, got {item!r}")
        terms.append(Term(name=item["name"], phase=item["phase"],
                          value=item["value"], enabled=bool(item.get("enabled", False)),
                          note=item.get("note", "")))
    return tuple(terms)
