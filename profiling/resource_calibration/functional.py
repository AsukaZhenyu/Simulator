"""One bounded, diagnostic resource-response functional, using only stdlib.

For each *exact* operation and execution protocol a frozen training fit learns
``duration(z) = theta . [1, cpu, gpu, dram/cap, vram/cap, cpu*gpu]``.  The
interaction feature is optional and is chosen before fitting.  Different
operations need different coefficients, but there is one fitting and prediction
path, never a model selected by a condition label.  No utilization-to-speed rule
is assumed.  Inputs and least squares use exact rational arithmetic.

The predictor integrates reference service at ``reference_ns / duration(z)``
over an exogenous piecewise-constant resource trace.  This is a phenomenological
completed-operation response, not a decomposition of host and kernel progress.
Its domain is the observed *marginal feature box*, not a joint convex support or
an accuracy certificate.  Positive predictions are required, never clipped.

Two input meanings are deliberately distinct: ``external`` means observed
external compute occupancy and memory use; ``offered_load_controlled`` means
requested owned compute loads and owned external allocations.  The latter is a
diagnostic control response, with other applications an unobserved residual.
Target-inclusive rolling utilization is endogenous and is not accepted here.
Capacity is separate from speed: only external state plus a complete declared
operation footprint can verify physical capacity.  Unknown footprint stays
unknown; a zero-byte declaration must be explicit.

The caller supplies the workload constants (shape, dtype, hardware/backend,
execution context and timing boundary in particular) in ``Operation.constants``.
They are hashed and matched exactly; this module never generalizes over them.
Only training rows enter a fit.  Heldout rows must be later than the freeze and
use disjoint runs and seeds, in the same declared monotonic clock domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


class UnknownSupport(ValueError):
    """The frozen response has no justified price for this request."""


class UnknownCapacity(ValueError):
    """A physical capacity assertion needs missing occupancy/footprint facts."""


class Infeasible(ValueError):
    """Even the known memory lower bound exceeds physical capacity."""


_SEMANTICS = ("external", "offered_load_controlled")
_BASE_FEATURES = ("intercept", "cpu_busy", "gpu_busy", "dram_fraction", "vram_fraction")


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _rational(value: Any, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction, str)):
        raise ValueError(f"{name} must be an integer, Fraction, or rational string; floats are not implicit")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a finite rational") from exc


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], kind: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{kind} has missing or unexpected fields")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResourceState:
    cpu_busy: Fraction
    gpu_busy: Fraction
    dram_used_bytes: int
    vram_used_bytes: int

    def __post_init__(self) -> None:
        for name in ("cpu_busy", "gpu_busy"):
            value = _rational(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        _integer(self.dram_used_bytes, "dram_used_bytes")
        _integer(self.vram_used_bytes, "vram_used_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {"cpu_busy": str(self.cpu_busy), "gpu_busy": str(self.gpu_busy),
                "dram_used_bytes": self.dram_used_bytes, "vram_used_bytes": self.vram_used_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceState:
        _exact_keys(value, ("cpu_busy", "gpu_busy", "dram_used_bytes", "vram_used_bytes"), "ResourceState")
        return cls(**value)


@dataclass(frozen=True)
class TraceSegment:
    start_ns: int
    end_ns: int
    state: ResourceState

    def __post_init__(self) -> None:
        _integer(self.start_ns, "start_ns")
        _integer(self.end_ns, "end_ns")
        if self.end_ns <= self.start_ns:
            raise ValueError("trace segment must have a positive interval")
        if not isinstance(self.state, ResourceState):
            raise ValueError("state must be ResourceState")

    def to_dict(self) -> dict[str, Any]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns, "state": self.state.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TraceSegment:
        _exact_keys(value, ("start_ns", "end_ns", "state"), "TraceSegment")
        return cls(value["start_ns"], value["end_ns"], ResourceState.from_dict(value["state"]))


@dataclass(frozen=True)
class ResourceTrace:
    segments: tuple[TraceSegment, ...]
    state_semantics: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError("resource trace must be nonempty")
        if self.state_semantics not in _SEMANTICS:
            raise ValueError("unsupported state_semantics; target-inclusive totals are not exogenous inputs")
        previous = None
        for segment in self.segments:
            if not isinstance(segment, TraceSegment):
                raise ValueError("segments must contain TraceSegment objects")
            if previous is not None and segment.start_ns != previous:
                raise ValueError("trace must be ordered, contiguous and non-overlapping")
            previous = segment.end_ns

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [item.to_dict() for item in self.segments], "state_semantics": self.state_semantics}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceTrace:
        _exact_keys(value, ("segments", "state_semantics"), "ResourceTrace")
        return cls(tuple(TraceSegment.from_dict(item) for item in value["segments"]), value["state_semantics"])


@dataclass(frozen=True)
class Hardware:
    identity: str
    dram_capacity_bytes: int
    vram_capacity_bytes: int

    def __post_init__(self) -> None:
        _text(self.identity, "hardware identity")
        _integer(self.dram_capacity_bytes, "dram_capacity_bytes", positive=True)
        _integer(self.vram_capacity_bytes, "vram_capacity_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "dram_capacity_bytes": self.dram_capacity_bytes,
                "vram_capacity_bytes": self.vram_capacity_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Hardware:
        _exact_keys(value, ("identity", "dram_capacity_bytes", "vram_capacity_bytes"), "Hardware")
        return cls(**value)


@dataclass(frozen=True)
class Operation:
    key: str
    resource: str
    reference_ns: int
    constants: tuple[tuple[str, str], ...] = ()
    dram_bytes: int | None = None
    vram_bytes: int | None = None
    memory_footprint_complete: bool = False

    def __post_init__(self) -> None:
        _text(self.key, "operation key")
        if self.resource not in ("cpu", "gpu", "copy"):
            raise ValueError("resource must be cpu, gpu or copy")
        _integer(self.reference_ns, "reference_ns", positive=True)
        try:
            items = tuple(self.constants)
            if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
                raise ValueError("constants must be key/value pairs")
            constants = tuple((item[0], item[1]) for item in items)
        except (TypeError, IndexError) as exc:
            raise ValueError("constants must be key/value pairs") from exc
        for key, value in constants:
            _text(key, "constant name")
            _text(value, "constant value")
        if len({key for key, _ in constants}) != len(constants):
            raise ValueError("duplicate operation constant")
        object.__setattr__(self, "constants", tuple(sorted(constants)))
        for name in ("dram_bytes", "vram_bytes"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name)
        if not isinstance(self.memory_footprint_complete, bool):
            raise ValueError("memory_footprint_complete must be boolean")
        if self.memory_footprint_complete and (self.dram_bytes is None or self.vram_bytes is None):
            raise ValueError("a complete memory footprint cannot contain unknown bytes")

    @property
    def signature(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "resource": self.resource, "reference_ns": self.reference_ns,
                "constants": [list(item) for item in self.constants], "dram_bytes": self.dram_bytes,
                "vram_bytes": self.vram_bytes, "memory_footprint_complete": self.memory_footprint_complete}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Operation:
        _exact_keys(value, ("key", "resource", "reference_ns", "constants", "dram_bytes", "vram_bytes", "memory_footprint_complete"), "Operation")
        return cls(**value)


@dataclass(frozen=True)
class CalibrationRow:
    row_id: str
    split: str
    run_id: str
    seed: int
    started_ns: int
    elapsed_ns: Fraction
    measured_until_ns: int
    operation: Operation
    state: ResourceState
    source_sha256: str
    protocol_id: str
    clock_id: str
    state_semantics: str
    numerical_passed: bool

    def __post_init__(self) -> None:
        for name in ("row_id", "run_id", "protocol_id", "clock_id"):
            _text(getattr(self, name), name)
        if self.split not in ("train", "heldout"):
            raise ValueError("split must be train or heldout")
        _integer(self.seed, "seed")
        _integer(self.started_ns, "started_ns")
        elapsed = _rational(self.elapsed_ns, "elapsed_ns")
        if elapsed <= 0:
            raise ValueError("elapsed_ns must be positive")
        object.__setattr__(self, "elapsed_ns", elapsed)
        _integer(self.measured_until_ns, "measured_until_ns")
        if self.measured_until_ns < self.started_ns + self.elapsed_ns:
            raise ValueError("measurement window must cover the reported invocation/aggregate duration")
        if not isinstance(self.operation, Operation) or not isinstance(self.state, ResourceState):
            raise ValueError("operation and state must be typed immutable values")
        _sha(self.source_sha256, "source_sha256")
        if self.state_semantics not in _SEMANTICS:
            raise ValueError("unsupported state_semantics; target-inclusive totals are not exogenous inputs")
        if not isinstance(self.numerical_passed, bool):
            raise ValueError("numerical_passed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "split": self.split, "run_id": self.run_id, "seed": self.seed,
                "started_ns": self.started_ns, "elapsed_ns": str(self.elapsed_ns), "measured_until_ns": self.measured_until_ns,
                "operation": self.operation.to_dict(), "state": self.state.to_dict(),
                "source_sha256": self.source_sha256, "protocol_id": self.protocol_id,
                "clock_id": self.clock_id, "state_semantics": self.state_semantics,
                "numerical_passed": self.numerical_passed}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationRow:
        _exact_keys(value, cls.__dataclass_fields__, "CalibrationRow")
        options = dict(value)
        options["operation"] = Operation.from_dict(options["operation"])
        options["state"] = ResourceState.from_dict(options["state"])
        return cls(**options)


@dataclass(frozen=True)
class FitConfig:
    regularization: Fraction = Fraction(1, 1000)
    include_cpu_gpu_interaction: bool = True
    min_training_runs: int = 3

    def __post_init__(self) -> None:
        value = _rational(self.regularization, "regularization")
        if value < 0:
            raise ValueError("regularization must be nonnegative")
        object.__setattr__(self, "regularization", value)
        if not isinstance(self.include_cpu_gpu_interaction, bool):
            raise ValueError("include_cpu_gpu_interaction must be boolean")
        _integer(self.min_training_runs, "min_training_runs", positive=True)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _BASE_FEATURES + (("cpu_gpu_interaction",) if self.include_cpu_gpu_interaction else ())

    def to_dict(self) -> dict[str, Any]:
        return {"regularization": str(self.regularization), "include_cpu_gpu_interaction": self.include_cpu_gpu_interaction,
                "min_training_runs": self.min_training_runs}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FitConfig:
        _exact_keys(value, cls.__dataclass_fields__, "FitConfig")
        return cls(**value)


def _features(state: ResourceState, hardware: Hardware, config: FitConfig) -> tuple[Fraction, ...]:
    result = (Fraction(1), state.cpu_busy, state.gpu_busy,
              Fraction(state.dram_used_bytes, hardware.dram_capacity_bytes),
              (Fraction(state.vram_used_bytes, hardware.vram_capacity_bytes)
               if hardware.vram_capacity_bytes else Fraction(0)))
    if not hardware.vram_capacity_bytes and state.vram_used_bytes:
        raise Infeasible("nonzero VRAM occupancy on hardware with zero VRAM capacity")
    return result + ((state.cpu_busy * state.gpu_busy,) if config.include_cpu_gpu_interaction else ())


def _solve(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """Gauss-Jordan solve; rank deficiency is explicit, not a hidden fallback."""
    size = len(vector)
    work = [list(matrix[index]) + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = next((index for index in range(column, size) if work[index][column]), None)
        if pivot is None:
            raise ValueError("calibration design is rank deficient; choose regularization before fitting")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        work[column] = [entry / divisor for entry in work[column]]
        for index in range(size):
            if index != column:
                factor = work[index][column]
                work[index] = [left - factor * right for left, right in zip(work[index], work[column])]
    return tuple(work[index][-1] for index in range(size))


@dataclass(frozen=True)
class ResponseSurface:
    operation: Operation
    coefficients_ns: tuple[Fraction, ...]
    feature_bounds: tuple[tuple[Fraction, Fraction], ...]
    training_rows: int
    training_runs: int
    training_relative_errors: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, Operation):
            raise ValueError("surface operation must be Operation")
        coefficients = tuple(_rational(item, "coefficient") for item in self.coefficients_ns)
        bounds = tuple(tuple(_rational(item, "feature bound") for item in pair) for pair in self.feature_bounds)
        errors = tuple(_rational(item, "training error") for item in self.training_relative_errors)
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in bounds):
            raise ValueError("feature bounds must be ordered low/high pairs")
        if len(bounds) != len(coefficients):
            raise ValueError("coefficient and feature-bound dimensions differ")
        _integer(self.training_rows, "training_rows", positive=True)
        _integer(self.training_runs, "training_runs", positive=True)
        if self.training_runs > self.training_rows or len(errors) != self.training_rows or any(item < 0 for item in errors):
            raise ValueError("training counts/errors are inconsistent")
        object.__setattr__(self, "coefficients_ns", coefficients)
        object.__setattr__(self, "feature_bounds", bounds)
        object.__setattr__(self, "training_relative_errors", errors)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.to_dict(), "coefficients_ns": [str(item) for item in self.coefficients_ns],
                "feature_bounds": [[str(low), str(high)] for low, high in self.feature_bounds],
                "training_rows": self.training_rows, "training_runs": self.training_runs,
                "training_relative_errors": [str(item) for item in self.training_relative_errors]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResponseSurface:
        _exact_keys(value, cls.__dataclass_fields__, "ResponseSurface")
        options = dict(value)
        options["operation"] = Operation.from_dict(options["operation"])
        return cls(**options)


@dataclass(frozen=True)
class ResourceFunctional:
    hardware: Hardware
    config: FitConfig
    surfaces: tuple[ResponseSurface, ...]
    frozen_at_ns: int
    source_sha256: str
    protocol_id: str
    clock_id: str
    state_semantics: str
    training_row_ids: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    training_data_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.hardware, Hardware) or not isinstance(self.config, FitConfig):
            raise ValueError("model hardware/config must be typed immutable values")
        surfaces = tuple(self.surfaces)
        if not surfaces or any(not isinstance(item, ResponseSurface) for item in surfaces):
            raise ValueError("model must contain typed response surfaces")
        if len({item.operation.signature for item in surfaces}) != len(surfaces):
            raise ValueError("duplicate operation response")
        for surface in surfaces:
            if len(surface.coefficients_ns) != len(self.config.feature_names):
                raise ValueError("surface feature dimension differs from frozen feature configuration")
            if surface.feature_bounds[0] != (Fraction(1), Fraction(1)):
                raise ValueError("intercept must have constant support [1, 1]")
            if any(low < 0 or high > 1 for low, high in surface.feature_bounds[1:]):
                raise ValueError("normalized feature bounds must lie in [0, 1]")
            if surface.training_runs < self.config.min_training_runs:
                raise ValueError("surface has insufficient training runs")
        _integer(self.frozen_at_ns, "frozen_at_ns")
        _sha(self.source_sha256, "source_sha256")
        _sha(self.training_data_sha256, "training_data_sha256")
        _text(self.protocol_id, "protocol_id")
        _text(self.clock_id, "clock_id")
        if self.state_semantics not in _SEMANTICS:
            raise ValueError("unsupported frozen state semantics")
        for name in ("training_row_ids", "training_run_ids", "training_seeds"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            for item in values:
                _integer(item, "training seed") if name == "training_seeds" else _text(item, name)
            object.__setattr__(self, name, values)
        if sum(surface.training_rows for surface in surfaces) != len(self.training_row_ids):
            raise ValueError("training row count is inconsistent with surfaces")
        object.__setattr__(self, "surfaces", surfaces)

    @property
    def compute_semantics(self) -> str:
        return "external_observed" if self.state_semantics == "external" else "offered_load"

    @property
    def memory_semantics(self) -> str:
        return "external_observed" if self.state_semantics == "external" else "owned_external_allocations"

    @property
    def model_sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "model_kind": "diagnostic_affine_resource_response",
                "hardware": self.hardware.to_dict(), "config": self.config.to_dict(),
                "feature_names": list(self.config.feature_names), "support": "observed_marginal_feature_box_not_joint_support",
                "surfaces": [item.to_dict() for item in self.surfaces], "frozen_at_ns": self.frozen_at_ns,
                "source_sha256": self.source_sha256, "protocol_id": self.protocol_id, "clock_id": self.clock_id,
                "state_semantics": self.state_semantics, "compute_semantics": self.compute_semantics,
                "memory_semantics": self.memory_semantics, "training_row_ids": list(self.training_row_ids),
                "training_run_ids": list(self.training_run_ids), "training_seeds": list(self.training_seeds),
                "training_data_sha256": self.training_data_sha256,
                "heldout_used_for_fit": False, "hardware_accuracy_accepted": False}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceFunctional:
        required = set(cls.__dataclass_fields__) | {"schema_version", "model_kind", "feature_names", "support",
                    "compute_semantics", "memory_semantics", "heldout_used_for_fit", "hardware_accuracy_accepted"}
        _exact_keys(value, required, "ResourceFunctional")
        if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
            raise ValueError("unsupported model schema version")
        if value["model_kind"] != "diagnostic_affine_resource_response" or value["support"] != "observed_marginal_feature_box_not_joint_support":
            raise ValueError("model kind/support metadata are inconsistent")
        if value["heldout_used_for_fit"] is not False or value["hardware_accuracy_accepted"] is not False:
            raise ValueError("this diagnostic schema cannot assert hardware accuracy or heldout fitting")
        options = {key: value[key] for key in cls.__dataclass_fields__}
        options["hardware"] = Hardware.from_dict(options["hardware"])
        options["config"] = FitConfig.from_dict(options["config"])
        options["surfaces"] = tuple(ResponseSurface.from_dict(item) for item in options["surfaces"])
        model = cls(**options)
        if list(model.config.feature_names) != value["feature_names"] or model.compute_semantics != value["compute_semantics"] or model.memory_semantics != value["memory_semantics"]:
            raise ValueError("feature/state semantics metadata are inconsistent")
        return model


def fit_model(rows: Iterable[CalibrationRow], hardware: Hardware, frozen_at_ns: int,
              config: FitConfig = FitConfig()) -> ResourceFunctional:
    """Freeze one diagnostic affine response from training rows only.

    The unpenalized intercept and a predeclared ridge penalty on all other
    coefficients minimize sum squared duration error.  Rows are not filtered,
    reweighted, or selected using heldout results.  This function makes no
    hardware-accuracy acceptance claim and does not diagnose causality.
    """
    _integer(frozen_at_ns, "frozen_at_ns")
    if not isinstance(hardware, Hardware) or not isinstance(config, FitConfig):
        raise ValueError("hardware and config must be typed immutable values")
    entries = tuple(rows)
    if not entries or any(not isinstance(row, CalibrationRow) for row in entries):
        raise ValueError("training rows must be nonempty CalibrationRow values")
    if any(row.split != "train" for row in entries):
        raise ValueError("fit is training-only; heldout rows cannot enter calibration")
    if len({row.row_id for row in entries}) != len(entries):
        raise ValueError("duplicate training row_id")
    entries = tuple(sorted(entries, key=lambda row: row.row_id))
    first = entries[0]
    binding = (first.source_sha256, first.protocol_id, first.clock_id, first.state_semantics)
    for row in entries:
        if not row.numerical_passed:
            raise ValueError("numerical failure cannot be silently dropped from calibration")
        if (row.source_sha256, row.protocol_id, row.clock_id, row.state_semantics) != binding:
            raise ValueError("training source, protocol, clock and state semantics must match")
        if row.measured_until_ns > frozen_at_ns:
            raise ValueError("model freeze precedes a training measurement end")
        _check_known_capacity(row.operation, row.state, hardware)
    groups: dict[str, list[CalibrationRow]] = {}
    for row in entries:
        groups.setdefault(row.operation.signature, []).append(row)
    surfaces = []
    for signature in sorted(groups):
        group = groups[signature]
        run_count = len({row.run_id for row in group})
        if run_count < config.min_training_runs:
            raise ValueError("insufficient independent training runs for an operation")
        design = [_features(row.state, hardware, config) for row in group]
        size = len(config.feature_names)
        normal = [[sum((point[left] * point[right] for point in design), Fraction(0))
                   + (config.regularization if left == right and left != 0 else 0)
                   for right in range(size)] for left in range(size)]
        rhs = [sum((point[column] * row.elapsed_ns for point, row in zip(design, group)), Fraction(0))
               for column in range(size)]
        coefficients = _solve(normal, rhs)
        predicted = [sum((weight * value for weight, value in zip(coefficients, point)), Fraction(0)) for point in design]
        if any(value <= 0 for value in predicted):
            raise ValueError("fitted duration is nonpositive on training support; no clipping is allowed")
        bounds = tuple((min(point[index] for point in design), max(point[index] for point in design)) for index in range(size))
        errors = tuple(abs(value - row.elapsed_ns) / row.elapsed_ns for value, row in zip(predicted, group))
        surfaces.append(ResponseSurface(group[0].operation, coefficients, bounds, len(group), run_count, errors))
    return ResourceFunctional(hardware, config, tuple(surfaces), frozen_at_ns, *binding,
                              tuple(sorted(row.row_id for row in entries)), tuple(sorted({row.run_id for row in entries})),
                              tuple(sorted({row.seed for row in entries})),
                              _digest([row.to_dict() for row in sorted(entries, key=lambda item: item.row_id)]))


def _surface(model: ResourceFunctional, operation: Operation) -> ResponseSurface:
    if not isinstance(model, ResourceFunctional) or not isinstance(operation, Operation):
        raise ValueError("model and operation must be typed immutable values")
    for surface in model.surfaces:
        if operation == surface.operation:
            return surface
    raise UnknownSupport("exact operation constants/resource/reference/footprint are not calibrated")


def predicted_duration_ns(model: ResourceFunctional, operation: Operation, state: ResourceState) -> Fraction:
    """Constant-state response; an exact rational, not a rounded price."""
    if not isinstance(state, ResourceState):
        raise ValueError("state must be ResourceState")
    surface = _surface(model, operation)
    point = _features(state, model.hardware, model.config)
    for name, value, (low, high) in zip(model.config.feature_names, point, surface.feature_bounds):
        if not low <= value <= high:
            raise UnknownSupport(f"{name} lies outside the frozen observed marginal feature range")
    duration = sum((coefficient * value for coefficient, value in zip(surface.coefficients_ns, point)), Fraction(0))
    if duration <= 0:
        raise UnknownSupport("fitted duration is nonpositive at this state; no clipping is allowed")
    return duration


def _check_known_capacity(operation: Operation, state: ResourceState, hardware: Hardware) -> None:
    for resource, external, own, capacity in (
        ("DRAM", state.dram_used_bytes, operation.dram_bytes, hardware.dram_capacity_bytes),
        ("VRAM", state.vram_used_bytes, operation.vram_bytes, hardware.vram_capacity_bytes),
    ):
        if external > capacity or (own is not None and external + own > capacity):
            raise Infeasible(f"known {resource} occupancy plus operation reservation exceeds capacity")


@dataclass(frozen=True)
class ServiceSlice:
    start_ns: Fraction
    end_ns: Fraction
    state: ResourceState
    duration_at_state_ns: Fraction
    service_rate: Fraction
    reference_service_ns: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {"start_ns": str(self.start_ns), "end_ns": str(self.end_ns), "state": self.state.to_dict(),
                "duration_at_state_ns": str(self.duration_at_state_ns), "service_rate": str(self.service_rate),
                "reference_service_ns": str(self.reference_service_ns)}


@dataclass(frozen=True)
class Prediction:
    operation: Operation
    start_ns: int
    exact_finish_ns: Fraction
    finish_ns: int
    slices: tuple[ServiceSlice, ...]
    physical_capacity_verified: bool
    state_semantics: str
    model_sha256: str

    @property
    def elapsed_ns(self) -> int:
        return self.finish_ns - self.start_ns

    @property
    def exact_elapsed_ns(self) -> Fraction:
        return self.exact_finish_ns - self.start_ns

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.to_dict(), "start_ns": self.start_ns,
                "exact_finish_ns": str(self.exact_finish_ns), "finish_ns": self.finish_ns,
                "exact_elapsed_ns": str(self.exact_elapsed_ns), "elapsed_ns": self.elapsed_ns,
                "slices": [item.to_dict() for item in self.slices],
                "physical_capacity_verified": self.physical_capacity_verified,
                "state_semantics": self.state_semantics, "model_sha256": self.model_sha256,
                "hardware_accuracy_accepted": False}


def predict(model: ResourceFunctional, operation: Operation, trace: ResourceTrace, *, start_ns: int | None = None,
            require_physical_capacity: bool = False) -> Prediction:
    """Integrate one operation across resource changes; no policy sees the trace.

    A supplied trace is an exogenous evaluation input, not future information
    that a causal scheduling policy is permitted to inspect.  A sequence uses
    ``predict_sequence``; resource concurrency and policy decisions remain the
    execution simulator's responsibility.
    """
    _surface(model, operation)
    if not isinstance(trace, ResourceTrace):
        raise ValueError("trace must be ResourceTrace")
    if not isinstance(require_physical_capacity, bool):
        raise ValueError("require_physical_capacity must be boolean")
    if trace.state_semantics != model.state_semantics:
        raise UnknownSupport("trace state semantics differ from frozen calibration semantics")
    start = trace.segments[0].start_ns if start_ns is None else _integer(start_ns, "start_ns")
    capacity_verified = trace.state_semantics == "external" and operation.memory_footprint_complete
    if require_physical_capacity and not capacity_verified:
        raise UnknownCapacity("external occupancy and a complete declared operation footprint are required")
    remaining = Fraction(operation.reference_ns)
    cursor = Fraction(start)
    slices = []
    for segment in trace.segments:
        if segment.end_ns <= cursor:
            continue
        if segment.start_ns > cursor:
            raise UnknownSupport("resource trace does not cover the operation start")
        _check_known_capacity(operation, segment.state, model.hardware)
        duration = predicted_duration_ns(model, operation, segment.state)
        rate = Fraction(operation.reference_ns) / duration
        available = Fraction(segment.end_ns) - cursor
        needed = remaining / rate
        elapsed = min(available, needed)
        service = elapsed * rate
        slices.append(ServiceSlice(cursor, cursor + elapsed, segment.state, duration, rate, service))
        cursor += elapsed
        remaining -= service
        if remaining == 0:
            finish = -(-cursor.numerator // cursor.denominator)
            return Prediction(operation, start, cursor, finish, tuple(slices), capacity_verified,
                              model.state_semantics, model.model_sha256)
    raise UnknownSupport("resource trace ends before operation completion; no last-state extrapolation")


def predict_sequence(model: ResourceFunctional, operations: Iterable[Operation], trace: ResourceTrace, *,
                     start_ns: int | None = None, require_physical_capacity: bool = False) -> tuple[Prediction, ...]:
    """A serial reference path using the same primitive for CPU/GPU/copy.

    Each operation starts at the preceding integer nanosecond completion.  No
    implicit fusion, overlap, transition penalty or persistent allocation is
    fabricated; such choices belong in workload constants/actions and traces.
    """
    result = []
    cursor = start_ns
    for operation in operations:
        item = predict(model, operation, trace, start_ns=cursor, require_physical_capacity=require_physical_capacity)
        result.append(item)
        cursor = item.finish_ns
    return tuple(result)


def evaluate_heldout(model: ResourceFunctional, rows: Iterable[CalibrationRow]) -> dict[str, Any]:
    """Read-only heldout evaluation; unsupported points stay explicit unknowns.

    Constant-state rows are evaluated as such.  Dynamic traces should be passed
    to ``predict`` and compared with independently measured trace replays; this
    function does not turn an in-interval average into an exogenous trace.
    """
    entries = tuple(rows)
    if not entries:
        raise ValueError("heldout rows must be nonempty")
    seen = set()
    for row in entries:
        if not isinstance(row, CalibrationRow) or row.split != "heldout":
            raise ValueError("evaluation requires heldout rows only")
        if row.row_id in seen or row.row_id in model.training_row_ids:
            raise ValueError("duplicate or leaked heldout row_id")
        seen.add(row.row_id)
        if row.run_id in model.training_run_ids or row.seed in model.training_seeds:
            raise ValueError("heldout runs and seeds must be disjoint from training")
        if (row.source_sha256, row.protocol_id, row.clock_id, row.state_semantics) != (
                model.source_sha256, model.protocol_id, model.clock_id, model.state_semantics):
            raise ValueError("heldout source, protocol, clock and state semantics must match frozen calibration")
        if row.started_ns < model.frozen_at_ns:
            raise ValueError("heldout measurement started before model freeze")
        if not row.numerical_passed:
            raise ValueError("heldout numerical failure cannot be silently excluded")
    results = []
    for row in entries:
        try:
            _check_known_capacity(row.operation, row.state, model.hardware)
            duration = predicted_duration_ns(model, row.operation, row.state)
            results.append({"row_id": row.row_id, "status": "predicted", "observed_ns": str(row.elapsed_ns),
                            "predicted_ns": str(duration), "relative_error": str(abs(duration - row.elapsed_ns) / row.elapsed_ns)})
        except (UnknownSupport, Infeasible) as exc:
            results.append({"row_id": row.row_id, "status": "unknown" if isinstance(exc, UnknownSupport) else "infeasible",
                            "observed_ns": str(row.elapsed_ns), "reason": str(exc)})
    errors = sorted(Fraction(row["relative_error"]) for row in results if row["status"] == "predicted")
    count = len(errors)
    median = None if not count else (errors[count // 2] if count % 2 else (errors[count // 2 - 1] + errors[count // 2]) / 2)
    return {"model_sha256": model.model_sha256, "heldout_used_for_fit": False, "hardware_accuracy_accepted": False,
            "row_count": len(entries), "predicted_count": count, "unknown_or_infeasible_count": len(entries) - count,
            "median_relative_error": None if median is None else str(median),
            "worst_relative_error": None if not errors else str(errors[-1]), "rows": results}
