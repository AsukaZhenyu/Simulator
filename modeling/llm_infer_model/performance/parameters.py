"""Model-owned physical parameter contract; no measurements or estimators."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

SCHEMA_VERSION = "performance-parameters-1"
UNITS = frozenset({"flop/s", "byte/s", "ns", "byte"})


class InvalidModel(ValueError):
    """A physical model or execution specification is malformed."""


class UnknownParameter(InvalidModel):
    """A required physical parameter has not been established."""


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidModel(f"{name} must be an integer >= {minimum}")
    return value


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InvalidModel(f"{name} must be a nonempty string")
    return value


def rational(value, name):
    if type(value) is int:
        result = Fraction(value)
    elif type(value) is Fraction:
        result = value
    else:
        raise InvalidModel(f"{name} must be an integer or Fraction, never a float")
    if result < 0:
        raise InvalidModel(f"{name} must be nonnegative")
    return result


def encode_fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def decode_fraction(value, name):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise InvalidModel(f"{name} requires numerator and denominator")
    return Fraction(integer(value["numerator"], name),
                    integer(value["denominator"], name, 1))


def exact_fields(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names):
        raise InvalidModel(f"{name} requires exactly {sorted(names)}")
    return value


@dataclass(frozen=True)
class Parameter:
    """One scoped rate, latency, or capacity with explicit provenance.

    ``None`` means unknown, not zero. Scope identifies the hardware, backend,
    precision, shape family and timing boundary to which the value applies.
    Profiling supplies values and evidence; modeling consumes this contract.
    """

    name: str
    value: int | Fraction | None
    unit: str
    origin: str
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        nonempty(self.name, "parameter name")
        if not isinstance(self.unit, str) or self.unit not in UNITS:
            raise InvalidModel(f"unsupported physical unit: {self.unit}")
        if not isinstance(self.origin, str) or self.origin not in {"declared", "profiled"}:
            raise InvalidModel("origin must be declared or profiled")
        if type(self.evidence) is not tuple or any(
                not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise InvalidModel("evidence must be a tuple of nonempty references")
        if self.origin == "profiled" and not self.evidence:
            raise InvalidModel("profiled parameters require evidence")
        nonempty(self.scope, "parameter scope")
        if self.value is not None:
            value = rational(self.value, self.name)
            if self.unit == "byte" and value.denominator != 1:
                raise InvalidModel("capacity bytes must be integral")
            object.__setattr__(self, "value", value)

    def to_dict(self):
        return {"name": self.name,
                "value": None if self.value is None else encode_fraction(self.value),
                "unit": self.unit, "origin": self.origin,
                "evidence": list(self.evidence), "scope": self.scope}

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"name", "value", "unit", "origin", "evidence", "scope"},
                           "parameter")
        if not isinstance(obj["evidence"], list):
            raise InvalidModel("parameter evidence must be a JSON list")
        return cls(obj["name"], None if obj["value"] is None else
                   decode_fraction(obj["value"], "parameter value"), obj["unit"],
                   obj["origin"], tuple(obj["evidence"]), obj["scope"])


@dataclass(frozen=True)
class ParameterBundle:
    parameters: tuple[Parameter, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise InvalidModel(f"requires {SCHEMA_VERSION}")
        if type(self.parameters) is not tuple or any(
                not isinstance(item, Parameter) for item in self.parameters):
            raise InvalidModel("parameters must be a tuple of Parameter")
        names = [item.name for item in self.parameters]
        if len(set(names)) != len(names):
            raise InvalidModel("duplicate parameter name")

    def get(self, name):
        for item in self.parameters:
            if item.name == name:
                return item
        raise UnknownParameter(f"missing physical parameter: {name}")

    def require(self, name, unit):
        item = self.get(name)
        if item.unit != unit:
            raise InvalidModel(f"{name} has unit {item.unit}; expected {unit}")
        if item.value is None:
            raise UnknownParameter(f"unknown physical parameter: {name}")
        return item.value

    def to_dict(self):
        return {"schema_version": self.schema_version,
                "parameters": [item.to_dict() for item in self.parameters]}

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "parameters"}, "parameter bundle")
        if not isinstance(obj["parameters"], list):
            raise InvalidModel("parameters must be a JSON list")
        return cls(tuple(Parameter.from_dict(item) for item in obj["parameters"]),
                   obj["schema_version"])
