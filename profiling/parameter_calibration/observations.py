"""Strict raw parameter evidence, with explicit units and timing boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA = "physical-parameter-observations-1"
KINDS = {
    "compute_throughput": ("flop/s", {"device_compute_active"}),
    "transfer_bandwidth": ("byte/s", {"payload_transfer_active"}),
    "host_fixed_overhead": ("ns", {"host_exclusive_completed"}),
    "capacity": ("byte", {"capacity_snapshot"}),
    "reference_duration": ("ns", {"whole_action_completed", "whole_iteration_completed",
                                        "overlapping_device_stream_interval"}),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value: Any, label: str = "value") -> Fraction:
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        n, d = value["numerator"], value["denominator"]
        if type(n) is int and type(d) is int and d > 0:
            return Fraction(n, d)
    raise ValueError(f"{label} requires an integer or exact rational; floats and booleans are not evidence")


def rational(value: Fraction | None) -> dict[str, int] | None:
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamps require ISO-8601 strings with a UTC offset")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("timestamps must explicitly use UTC")
    return timestamp


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ParameterObservation:
    observation_id: str
    parameter_name: str
    kind: str
    value: Fraction | None
    unit: str
    timing_scope: str
    context_json: str
    run_id: str
    process_id: int
    seed: int
    split: str
    started_at_utc: str
    ended_at_utc: str
    source_path: str
    source_sha256: str
    evidence: tuple[str, ...]
    source_verified: bool = False
    numerical_passed: bool | None = None

    def __post_init__(self):
        for name in ("observation_id", "parameter_name", "run_id", "source_path"):
            _text(getattr(self, name), name)
        if self.kind not in KINDS:
            raise ValueError("unknown parameter kind")
        unit, scopes = KINDS[self.kind]
        if self.unit != unit or self.timing_scope not in scopes:
            raise ValueError(f"{self.kind} requires unit {unit!r} and timing_scope in {sorted(scopes)}; total/overlapping time cannot become hardware throughput")
        if self.value is not None:
            value = exact(self.value)
            if value < 0 or (self.kind in {"compute_throughput", "transfer_bandwidth", "reference_duration"} and value == 0):
                raise ValueError("parameter value is outside the physical range")
            object.__setattr__(self, "value", value)
        if type(self.process_id) is not int or self.process_id <= 0 or type(self.seed) is not int:
            raise ValueError("positive integer process_id and integer seed are required")
        if self.split not in {"profile", "heldout", "diagnostic"}:
            raise ValueError("split must be profile, heldout, or diagnostic")
        if utc(self.ended_at_utc) < utc(self.started_at_utc):
            raise ValueError("observation ends before it starts")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.evidence, tuple) or not self.evidence or any(not isinstance(x, str) or not x.strip() for x in self.evidence):
            raise ValueError("nonempty evidence references are required")
        if type(self.source_verified) is not bool:
            raise ValueError("source_verified must be boolean")
        if self.numerical_passed is not None and type(self.numerical_passed) is not bool:
            raise ValueError("numerical_passed must be an explicit boolean or unknown")
        try:
            context = json.loads(self.context_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("context_json must encode an exact context dictionary") from exc
        if not isinstance(context, dict) or not context:
            raise ValueError("a nonempty exact context dictionary is required")
        object.__setattr__(self, "context_json", canonical(context).decode())

    @property
    def context(self) -> dict[str, Any]:
        return json.loads(self.context_json)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name not in {"context_json", "value", "evidence"}}
        result.update(value=rational(self.value), context=self.context, evidence=list(self.evidence))
        return result

    @classmethod
    def from_dict(cls, row: Mapping[str, Any], *, source_verified: bool = False):
        if not isinstance(row, Mapping):
            raise ValueError("each observation must be an object")
        required = set(cls.__dataclass_fields__) - {"context_json", "source_verified", "numerical_passed"} | {"context"}
        if set(row) - {"source_verified", "numerical_passed"} != required:
            raise ValueError(f"observation keys must be exactly {sorted(required)}")
        values = dict(row)
        values.pop("source_verified", None)  # A JSON assertion cannot verify its own source bytes.
        values["context_json"] = canonical(values.pop("context")).decode()
        if values["value"] is not None:
            values["value"] = exact(values["value"])
        if not isinstance(values["evidence"], (list, tuple)):
            raise ValueError("evidence must be an array")
        values["evidence"] = tuple(values["evidence"])
        return cls(**values, source_verified=source_verified)


def derive_observation(*, work: int, duration_ns: int, **metadata) -> ParameterObservation:
    """Convert known work and a compatible exclusive interval to a physical value.

    No subtractive estimate is made: overlapping host/device intervals cannot be
    separated by subtraction, and total PV time cannot measure FLOP throughput.
    The raw work and duration must remain available in the referenced source.
    """
    if type(work) is not int or work < 0 or type(duration_ns) is not int or duration_ns <= 0:
        raise ValueError("work and duration_ns require nonnegative/positive integers")
    kind = metadata.get("kind")
    if kind not in {"compute_throughput", "transfer_bandwidth"}:
        raise ValueError("work/duration derivation supports compute or transfer only")
    return ParameterObservation(value=Fraction(work * 1_000_000_000, duration_ns), **metadata)


def load_observations(path: str | Path) -> tuple[ParameterObservation, ...]:
    """Load JSON, hash every referenced source file, and reject changed evidence."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "observations"} or document["schema"] != SCHEMA:
        raise ValueError("unsupported observations document")
    if not isinstance(document["observations"], list):
        raise ValueError("observations must be an array")
    result = []
    for row in document["observations"]:
        observation = ParameterObservation.from_dict(row)
        source = Path(observation.source_path)
        source = source if source.is_absolute() else path.parent / source
        try:
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot read bound observation source: {source}") from exc
        if actual != observation.source_sha256:
            raise ValueError(f"observation source hash mismatch: {source}")
        result.append(ParameterObservation.from_dict(row, source_verified=True))
    identifiers = [item.observation_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate observation_id")
    return tuple(result)
