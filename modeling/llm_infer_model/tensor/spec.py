"""M0 spec: the frozen contract objects, input loading, and validation.

The M0 contract lives in ``mapping/DESIGN.md`` §4 and ``mapping/ACCEPTANCE.md``.
This module turns the JSON documents into frozen dataclasses and refuses
everything the contract does not cover. Three failure kinds are kept strictly
apart, because the contract gives them different statuses rather than one
generic "error" (``DESIGN.md`` §8):

``invalid_input``
    Structurally broken or semantically inconsistent: unknown references,
    cycles, duplicate ids, several producers, a missing cost, an illegal shape.
``unsupported``
    Well-formed JSON asking for something M0 deliberately does not model:
    a non-F32 dtype, a view, a non-contiguous layout, in-place shared storage,
    more than one resource slot, a DRAM capacity, recomputation. These must be
    refused explicitly instead of approximated (``DESIGN.md`` §1) -- never
    silently defaulted away.
``initial_capacity_exceeded``
    The initial layout does not fit. This is *not* an input error: both entry
    points report it as ``infeasible`` carrying this reason, because the
    scenario itself is legitimate and only the starting point is impossible.

Two naming notes, both recorded because the source documents disagree:

* ``DESIGN.md``:84 names the provenance field ``origin`` while
  ``ACCEPTANCE.md``:113 writes ``graph_origin``. ``README.md``:11 says this
  directory is authoritative and ``DESIGN.md`` is read first, so ``origin`` is
  what is implemented.
* ``DESIGN.md``:167 requires two SHA-256 fingerprints over canonically
  serialised JSON but never names the fields. They are exposed here as
  ``workload_fingerprint`` and ``scenario_fingerprint``; the canonicalisation
  is a single pure function so the test suite can pin it down, as the spec asks.

All times are integers in nanoseconds and all byte counts are integers. This is
deliberately unlike ``modeling/``, which works in float seconds; the ``_ns``
suffix exists only in M0 and the two conventions are not to be merged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "SpecError",
    "InvalidInput",
    "Unsupported",
    "InitialCapacityExceeded",
    "Tensor",
    "Operation",
    "Origin",
    "Workload",
    "Architecture",
    "ComputeCost",
    "H2DCost",
    "Costs",
    "Mapspace",
    "MapperConfig",
    "Scenario",
    "aligned_size",
    "load_workload",
    "load_and_validate",
    "workload_fingerprint",
    "scenario_fingerprint",
    "check_initial_capacity",
    # 契约字段的取值域。它们是 `Tensor.dtype/location/role`、`Operation.semantic`、
    # `Origin.kind` 等公开数据类的合法取值，调用方构造或核对文档时必须用到，
    # 所以和上面那些类一样属于公开面。此前它们不在 __all__ 里，只是「按名可导入」
    # 的既成事实；星号转出（兼容层正是这么做的）会静默漏掉它们，故显式列出。
    "ALGORITHM_UNIFORM_COST",
    "COPY_ABSENT",
    "COPY_DEVICE",
    "COPY_READY",
    "COPY_RESERVED_COPY",
    "COPY_RESERVED_OUTPUT",
    "DTYPE_F32",
    "LOC_DRAM",
    "LOC_VRAM",
    "OP_DONE",
    "OP_NOT_STARTED",
    "OP_RUNNING",
    "ORIGIN_GGML",
    "ORIGIN_SYNTHETIC",
    "RESOURCE_COMPUTE",
    "RESOURCE_COPY",
    "ROLE_INPUT",
    "ROLE_INTERMEDIATE",
    "ROLE_OUTPUT",
    "ROLE_WEIGHT",
    "SEMANTIC_ADD",
    "SEMANTIC_MUL_MAT",
    "SEMANTIC_RELU",
]

# All M0 JSON documents -- workload, scenario and mapping alike -- carry this
# same version string (DESIGN.md §4).
SCHEMA_VERSION = "0.1"

# ---------------------------------------------------------------------------
# Enumerations. Kept as plain strings rather than enum.Enum so that the values
# round-trip through JSON artefacts byte-for-byte (DESIGN.md §8).
# ---------------------------------------------------------------------------

DTYPE_F32 = "GGML_TYPE_F32"
_DTYPE_SIZE_BYTES = {DTYPE_F32: 4}

ROLE_INPUT = "input"
ROLE_WEIGHT = "weight"
ROLE_INTERMEDIATE = "intermediate"
ROLE_OUTPUT = "output"
_ROLES = frozenset({ROLE_INPUT, ROLE_WEIGHT, ROLE_INTERMEDIATE, ROLE_OUTPUT})

SEMANTIC_MUL_MAT = "MUL_MAT"
SEMANTIC_RELU = "RELU"
SEMANTIC_ADD = "ADD"
_SEMANTIC_OPS = frozenset({SEMANTIC_MUL_MAT, SEMANTIC_RELU, SEMANTIC_ADD})

LOC_DRAM = "dram"
LOC_VRAM = "vram"
_LOCATIONS = frozenset({LOC_DRAM, LOC_VRAM})

# Operation status (DESIGN.md §5).
OP_NOT_STARTED = "NOT_STARTED"
OP_RUNNING = "RUNNING"
OP_DONE = "DONE"

# GPU-copy status (DESIGN.md §5). RESERVED_COPY is a copy in flight,
# RESERVED_OUTPUT is the destination of a running compute.
COPY_ABSENT = "ABSENT"
COPY_RESERVED_COPY = "RESERVED_COPY"
COPY_RESERVED_OUTPUT = "RESERVED_OUTPUT"
COPY_READY = "READY"

# Provenance. "ggml" is reserved for a graph that actually came out of a real
# exporter run; hand-written fixtures must say "synthetic_fixture" so that a
# passing core test is never mistaken for GGML front-end acceptance
# (DESIGN.md §9, ACCEPTANCE.md §7).
ORIGIN_GGML = "ggml"
ORIGIN_SYNTHETIC = "synthetic_fixture"
_ORIGIN_KINDS = frozenset({ORIGIN_GGML, ORIGIN_SYNTHETIC})

# Resource names used in states, events and error reports.
RESOURCE_COMPUTE = "gpu_compute"
RESOURCE_COPY = "h2d_copy"

ALGORITHM_UNIFORM_COST = "uniform_cost"

COPY_DEVICE = "gpu"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SpecError(Exception):
    """Base class for everything this module refuses."""

    code = "invalid_input"


class InvalidInput(SpecError):
    """Structurally broken or semantically inconsistent input."""

    code = "invalid_input"


class Unsupported(SpecError):
    """Well-formed input using semantics M0 deliberately does not model."""

    code = "unsupported"


class InitialCapacityExceeded(SpecError):
    """The initial layout alone exceeds the VRAM budget.

    Carried as a *status* (``infeasible`` + this reason), not an input error:
    the scenario is well-formed and the same scenario may be feasible once the
    capacity is raised (``ACCEPTANCE.md`` §4).
    """

    code = "initial_capacity_exceeded"


# ---------------------------------------------------------------------------
# Contract objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    """Immutable description of one tensor in the DAG.

    ``ne`` and ``nb`` follow GGML, **not** numpy: ``ne[0]`` is the innermost
    dimension, so a row-major ``(rows, cols)`` matrix is ``ne=[cols, rows, 1, 1]``.
    They are never reinterpreted (``DESIGN.md`` §3).
    """

    id: str
    name: str
    role: str
    dtype: str
    ne: tuple[int, int, int, int]
    nb: tuple[int, int, int, int]
    storage_bytes: int
    initial_locations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Operation:
    """One node of the compute DAG."""

    id: str
    semantic_op: str
    ggml_op: str
    unary_op: str | None
    op_params: tuple[int, ...]
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Origin:
    """Provenance of a workload.

    ``kind`` is the semantic part (was this graph actually exported, or
    hand-written?). The version and sample fields are provenance metadata and
    are excluded from the fingerprint.
    """

    kind: str
    ggml_version: str | None = None
    sample: str | None = None


@dataclass(frozen=True)
class Workload:
    """A frozen GGML DAG plus the derived indices the engine needs."""

    id: str
    origin: Origin
    tensors: tuple[Tensor, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[str, ...]

    # Derived, read-only lookup tables. Built once in __post_init__ so that the
    # engine never re-scans the tensor list per action.
    tensor_by_id: dict[str, Tensor] = field(default_factory=dict, repr=False)
    operation_by_id: dict[str, Operation] = field(default_factory=dict, repr=False)
    producer_of: dict[str, str] = field(default_factory=dict, repr=False)
    consumers_of: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    leaf_tensors: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        _validate_workload(self)


@dataclass(frozen=True)
class Architecture:
    """The single-GPU resource model of M0.

    The invariants are checked here rather than only in the JSON parser, so that
    a programmatically built ``Architecture`` -- which is how a capacity sweep
    varies a scenario -- is held to the same rules as a parsed one.
    """

    vram_capacity_bytes: int
    runtime_reserved_bytes: int = 0
    allocation_alignment_bytes: int = 1
    gpu_compute_slots: int = 1
    h2d_copy_slots: int = 1
    # Must be null: M0 treats DRAM as unbounded. A finite DRAM budget would need
    # a second eviction decision the search has no action for (DESIGN.md §4.2).
    dram_capacity_bytes: None = None

    def __post_init__(self) -> None:
        if self.dram_capacity_bytes is not None:
            raise Unsupported(
                "architecture.dram_capacity_bytes must be null: M0 treats DRAM as "
                "unbounded, and a finite DRAM budget needs an eviction decision the "
                "M0 action space does not have"
            )
        if self.vram_capacity_bytes < 1:
            raise InvalidInput(
                f"architecture.vram_capacity_bytes must be >= 1, got "
                f"{self.vram_capacity_bytes}"
            )
        if self.runtime_reserved_bytes < 0:
            raise InvalidInput(
                f"architecture.runtime_reserved_bytes must be >= 0, got "
                f"{self.runtime_reserved_bytes}"
            )
        if self.runtime_reserved_bytes > self.vram_capacity_bytes:
            raise InvalidInput(
                f"architecture.runtime_reserved_bytes ({self.runtime_reserved_bytes}) "
                f"exceeds vram_capacity_bytes ({self.vram_capacity_bytes})"
            )
        if self.allocation_alignment_bytes < 1:
            raise InvalidInput(
                f"architecture.allocation_alignment_bytes must be >= 1, got "
                f"{self.allocation_alignment_bytes}"
            )
        for name, slots in (
            ("gpu_compute_slots", self.gpu_compute_slots),
            ("h2d_copy_slots", self.h2d_copy_slots),
        ):
            if slots != 1:
                raise Unsupported(
                    f"architecture.{name} is {slots}; M0 models exactly one resource "
                    "of each kind"
                )


@dataclass(frozen=True)
class ComputeCost:
    """Duration of one compute node, plus the transient workspace it needs."""

    duration_ns: int
    workspace_bytes: int = 0


@dataclass(frozen=True)
class H2DCost:
    """Duration of one host-to-device copy."""

    duration_ns: int


@dataclass(frozen=True)
class Costs:
    """The cost model. M0 accepts only externally supplied fixed costs."""

    source: str
    compute: dict[str, ComputeCost]
    h2d: dict[str, H2DCost]


@dataclass(frozen=True)
class Mapspace:
    """What the mapper is allowed to do.

    Three switches are load-bearing for `unsupported` handling: recomputation is
    not modelled, and the two slot counts are fixed at one each.
    """

    compute_device: str
    copy_tensor_ids: tuple[str, ...]
    allow_copy_compute_overlap: bool
    allow_eviction: bool
    allow_recomputation: bool = False


@dataclass(frozen=True)
class MapperConfig:
    """Search budget. Zero means unlimited for both fields."""

    algorithm: str
    max_expanded_states: int = 0
    wall_time_limit_s: float = 0.0


@dataclass(frozen=True)
class Scenario:
    """A workload bound to an architecture, a cost model, and a mapspace."""

    schema_version: str
    id: str
    workload: Workload
    architecture: Architecture
    costs: Costs
    mapspace: Mapspace
    mapper: MapperConfig

    def size_alloc(self, tensor_id: str) -> int:
        """VRAM actually occupied by ``tensor_id``, after alignment."""
        tensor = self.workload.tensor_by_id[tensor_id]
        return aligned_size(
            tensor.storage_bytes, self.architecture.allocation_alignment_bytes
        )

    def workspace_alloc(self, operation_id: str) -> int:
        """VRAM reserved for the running compute's transient workspace."""
        cost = self.costs.compute[operation_id]
        return aligned_size(
            cost.workspace_bytes, self.architecture.allocation_alignment_bytes
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def aligned_size(storage_bytes: int, alignment: int) -> int:
    """Round ``storage_bytes`` up to a multiple of ``alignment``.

    Allocation granularity and transferred bytes are different quantities: a
    16-byte vector with ``alignment=32`` *occupies* 32 bytes but still *moves*
    16, so this helper is used for the memory ledger only and never for the
    byte counters in the statistics.
    """
    if alignment <= 1:
        return storage_bytes
    return -(-storage_bytes // alignment) * alignment


def _expect(condition: bool, message: str, exc: type[SpecError] = InvalidInput) -> None:
    if not condition:
        raise exc(message)


def _require_int(value: Any, what: str, minimum: int = 0) -> int:
    """Accept a real integer only. ``bool`` is rejected: it is an ``int`` in
    Python and ``true`` is never a legal duration or byte count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{what} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidInput(f"{what} must be >= {minimum}, got {value}")
    return value


def _require_bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidInput(f"{what} must be a boolean, got {value!r}")
    return value


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise InvalidInput(f"{what} must be a string, got {value!r}")
    return value


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidInput(f"{what} must be a JSON object, got {value!r}")
    return value


def _require_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidInput(f"{what} must be a JSON array, got {value!r}")
    return value


def _reject_unknown_keys(block: dict[str, Any], allowed: set[str], what: str) -> None:
    """Unknown keys are refused rather than ignored.

    Silently dropping an unrecognised key is how a typo turns into a silently
    disabled constraint, which is exactly the failure mode ``DESIGN.md`` §1
    warns about.
    """
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise InvalidInput(f"{what}: unknown key(s) {', '.join(unknown)}")


def _check_schema_version(raw: Any, where: str) -> str:
    """Every M0 document declares ``schema_version`` and ``"0.1"`` is the only one.

    A missing version is an error rather than a default: an unversioned file is
    of unknown vintage, and guessing would defeat the point of versioning it.
    """
    version = _require_str(raw, f"{where}.schema_version")
    if version != SCHEMA_VERSION:
        raise Unsupported(
            f"{where}.schema_version is {version!r}; this build implements "
            f"{SCHEMA_VERSION!r}"
        )
    return version


def _require_f32_nbytes(ne: tuple[int, ...], nb: tuple[int, ...]) -> int:
    """Validate a contiguous F32 layout and return the byte size.

    GGML stores the strides ``nb`` explicitly, so contiguity is checkable
    rather than assumed: ``nb[0]`` is the type size and each outer stride is the
    previous stride times the previous extent. A non-contiguous tensor is
    refused as ``unsupported`` -- M0 has no way to model it and must not
    pretend the tensor is dense.
    """
    for i in range(4):
        _expect(ne[i] >= 1, f"ne[{i}] must be >= 1, got {ne[i]}")

    expected = _DTYPE_SIZE_BYTES[DTYPE_F32]
    for i in range(4):
        if nb[i] != expected:
            raise Unsupported(
                f"non-contiguous or non-F32 layout: nb[{i}]={nb[i]} but a "
                f"contiguous {DTYPE_F32} tensor needs nb[{i}]={expected}"
            )
        expected = expected * ne[i]
    return nb[3] * ne[3]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_origin(raw: Any, where: str) -> Origin:
    block = _require_mapping(raw, f"{where}.origin")
    _reject_unknown_keys(block, {"kind", "ggml_version", "sample"}, f"{where}.origin")
    kind = _require_str(block.get("kind"), f"{where}.origin.kind")
    _expect(
        kind in _ORIGIN_KINDS,
        f"{where}.origin.kind must be one of {sorted(_ORIGIN_KINDS)}, got {kind!r}",
    )
    ggml_version = block.get("ggml_version")
    if ggml_version is not None:
        ggml_version = _require_str(ggml_version, f"{where}.origin.ggml_version")
    sample = block.get("sample")
    if sample is not None:
        sample = _require_str(sample, f"{where}.origin.sample")
    return Origin(kind=kind, ggml_version=ggml_version, sample=sample)


def _parse_tensor(raw: Any, where: str) -> Tensor:
    block = _require_mapping(raw, where)
    _reject_unknown_keys(
        block,
        {
            "id",
            "name",
            "role",
            "dtype",
            "ne",
            "nb",
            "storage_bytes",
            "initial_locations",
            "view_src",
        },
        where,
    )

    # ggml views alias another tensor's storage. M0 has one allocation per
    # storage tensor and no way to express the aliasing, so a view is refused
    # outright rather than being flattened into a second, independent tensor --
    # which would double-count the bytes and hide the dependency the view
    # creates (DESIGN.md §3).
    view_src = block.get("view_src")
    if view_src is not None:
        raise Unsupported(
            f"{where}.view_src is {view_src!r}; M0 models neither views nor any other "
            "aliasing of storage"
        )

    tensor_id = _require_str(block.get("id"), f"{where}.id")
    _expect(bool(tensor_id), f"{where}.id must not be empty")
    name = _require_str(block.get("name", tensor_id), f"{where}.name")

    role = _require_str(block.get("role"), f"{where}.role")
    _expect(
        role in _ROLES,
        f"{where}.role must be one of {sorted(_ROLES)}, got {role!r}",
    )

    dtype = _require_str(block.get("dtype"), f"{where}.dtype")
    if dtype != DTYPE_F32:
        raise Unsupported(
            f"{where}.dtype is {dtype!r}; M0 models only {DTYPE_F32} because its "
            "costs are supplied per operation, not derived from the dtype"
        )

    ne_raw = _require_list(block.get("ne"), f"{where}.ne")
    nb_raw = _require_list(block.get("nb"), f"{where}.nb")
    _expect(len(ne_raw) == 4, f"{where}.ne must have exactly 4 entries, got {len(ne_raw)}")
    _expect(len(nb_raw) == 4, f"{where}.nb must have exactly 4 entries, got {len(nb_raw)}")
    ne = tuple(_require_int(v, f"{where}.ne[{i}]", minimum=1) for i, v in enumerate(ne_raw))
    nb = tuple(_require_int(v, f"{where}.nb[{i}]", minimum=1) for i, v in enumerate(nb_raw))
    assert len(ne) == 4 and len(nb) == 4  # narrowed for the type checker

    computed_bytes = _require_f32_nbytes(ne, nb)
    storage_bytes = _require_int(block.get("storage_bytes"), f"{where}.storage_bytes", minimum=1)
    _expect(
        storage_bytes == computed_bytes,
        f"{where}.storage_bytes is {storage_bytes} but the ne/nb layout implies "
        f"{computed_bytes} bytes",
    )

    locs_raw = _require_list(block.get("initial_locations", []), f"{where}.initial_locations")
    locations: list[str] = []
    for i, loc in enumerate(locs_raw):
        loc = _require_str(loc, f"{where}.initial_locations[{i}]")
        _expect(
            loc in _LOCATIONS,
            f"{where}.initial_locations[{i}] must be one of {sorted(_LOCATIONS)}, got {loc!r}",
        )
        _expect(loc not in locations, f"{where}.initial_locations lists {loc!r} twice")
        locations.append(loc)

    return Tensor(
        id=tensor_id,
        name=name,
        role=role,
        dtype=dtype,
        ne=ne,  # type: ignore[arg-type]
        nb=nb,  # type: ignore[arg-type]
        storage_bytes=storage_bytes,
        initial_locations=tuple(locations),
    )


def _parse_operation(raw: Any, where: str) -> Operation:
    block = _require_mapping(raw, where)
    _reject_unknown_keys(
        block,
        {"id", "semantic_op", "ggml_op", "unary_op", "op_params", "inputs", "output"},
        where,
    )

    op_id = _require_str(block.get("id"), f"{where}.id")
    _expect(bool(op_id), f"{where}.id must not be empty")

    semantic_op = _require_str(block.get("semantic_op"), f"{where}.semantic_op")
    if semantic_op not in _SEMANTIC_OPS:
        raise Unsupported(
            f"{where}.semantic_op is {semantic_op!r}; M0 models only "
            f"{sorted(_SEMANTIC_OPS)}"
        )

    # The raw GGML enum name is carried through for traceability. RELU is not a
    # GGML_OP of its own -- ggml_relu() emits GGML_OP_UNARY plus a unary-op tag
    # (DESIGN.md §3) -- so callers that build the graph with ggml_relu() must
    # normalise before they get here, and unary_op records which tag it was.
    ggml_op = _require_str(block.get("ggml_op"), f"{where}.ggml_op")
    unary_op = block.get("unary_op")
    if unary_op is not None:
        unary_op = _require_str(unary_op, f"{where}.unary_op")

    params_raw = _require_list(block.get("op_params", []), f"{where}.op_params")
    op_params = tuple(
        _require_int(v, f"{where}.op_params[{i}]") for i, v in enumerate(params_raw)
    )

    inputs_raw = _require_list(block.get("inputs"), f"{where}.inputs")
    inputs: list[str] = []
    for i, value in enumerate(inputs_raw):
        value = _require_str(value, f"{where}.inputs[{i}]")
        _expect(bool(value), f"{where}.inputs[{i}] must not be empty")
        inputs.append(value)

    output = _require_str(block.get("output"), f"{where}.output")
    _expect(bool(output), f"{where}.output must not be empty")

    return Operation(
        id=op_id,
        semantic_op=semantic_op,
        ggml_op=ggml_op,
        unary_op=unary_op,
        op_params=op_params,
        inputs=tuple(inputs),
        output=output,
    )


def load_workload(path: str | Path) -> Workload:
    """Load and validate a workload JSON document."""
    return _parse_workload(_read_json(Path(path)), Path(path).name)


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidInput(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInput(f"{path} is not valid JSON: {exc}") from exc


def _parse_workload(raw: Any, where: str) -> Workload:
    block = _require_mapping(raw, where)
    # "comment" is allowed at the top level of each M0 document -- here, in a
    # scenario (spec.py), and in a mapping (engine.py) -- and on a mapping's
    # action entries; nowhere deeper. These files are hand-maintained contract
    # examples, and an explanation that has nowhere to live ends up in a commit
    # message nobody reads. It carries no semantics.
    _reject_unknown_keys(
        block, {"schema_version", "id", "origin", "tensors", "operations", "outputs", "comment"}, where
    )

    _check_schema_version(block.get("schema_version"), where)

    workload_id = _require_str(block.get("id"), f"{where}.id")
    origin = _parse_origin(block.get("origin"), where)

    tensors_raw = _require_list(block.get("tensors"), f"{where}.tensors")
    tensors = tuple(
        _parse_tensor(entry, f"{where}.tensors[{i}]") for i, entry in enumerate(tensors_raw)
    )

    operations_raw = _require_list(block.get("operations"), f"{where}.operations")
    operations = tuple(
        _parse_operation(entry, f"{where}.operations[{i}]")
        for i, entry in enumerate(operations_raw)
    )

    outputs_raw = _require_list(block.get("outputs"), f"{where}.outputs")
    outputs = tuple(
        _require_str(entry, f"{where}.outputs[{i}]") for i, entry in enumerate(outputs_raw)
    )

    # The tensor/operation indices are derived inside Workload.__post_init__,
    # which also runs the cross-reference and shape checks.
    return Workload(
        id=workload_id,
        origin=origin,
        tensors=tensors,
        operations=operations,
        outputs=outputs,
    )


def _validate_workload(workload: Workload) -> None:
    """Cross-reference, shape and reachability checks.

    Runs on construction, so a ``Workload`` value that exists is always one the
    engine may assume well-formed.
    """
    if not workload.tensors:
        raise InvalidInput(f"workload {workload.id!r}: tensors must not be empty")
    if not workload.operations:
        raise InvalidInput(f"workload {workload.id!r}: operations must not be empty")
    if not workload.outputs:
        raise InvalidInput(f"workload {workload.id!r}: outputs must not be empty")

    tensor_by_id: dict[str, Tensor] = {}
    for tensor in workload.tensors:
        if tensor.id in tensor_by_id:
            raise InvalidInput(f"duplicate tensor id {tensor.id!r}")
        tensor_by_id[tensor.id] = tensor

    operation_by_id: dict[str, Operation] = {}
    for operation in workload.operations:
        if operation.id in operation_by_id:
            raise InvalidInput(f"duplicate operation id {operation.id!r}")
        operation_by_id[operation.id] = operation

    # --- per-operation shape and reference checks -------------------------
    producer_of: dict[str, str] = {}
    for operation in workload.operations:
        # Prefer the operation *name* in messages; the id is only a key.
        label = f"operation {operation.id!r}"
        if not operation.inputs:
            raise InvalidInput(f"{label}: inputs must not be empty")

        seen_inputs: set[str] = set()
        for tensor_id in operation.inputs:
            if tensor_id not in tensor_by_id:
                raise InvalidInput(f"{label}: unknown input tensor {tensor_id!r}")
            # A repeated operand is legal -- add(x, x) is a real graph -- but it
            # is one allocation and one read lock, never two.
            seen_inputs.add(tensor_id)

        if operation.output not in tensor_by_id:
            raise InvalidInput(f"{label}: unknown output tensor {operation.output!r}")
        if operation.output in operation.inputs:
            raise Unsupported(
                f"{label}: reads and writes {operation.output!r} in place; M0 has "
                "no in-place reuse and no way to model the aliasing"
            )
        if operation.output in producer_of:
            raise InvalidInput(
                f"tensor {operation.output!r} has several producers "
                f"({producer_of[operation.output]!r} and {operation.id!r}); M0 needs "
                "a single producer per tensor so that liveness is a DAG property"
            )
        producer_of[operation.output] = operation.id

        _validate_operation_shape(operation, tensor_by_id, label)

    # --- every tensor is either an operation output or a leaf --------------
    leaf_tensors: frozenset[str] = frozenset(tensor_by_id) - set(producer_of)

    # outputs[] is simply "tensors that must end up resident in VRAM". Normally
    # these are operation results, but a leaf named here is legal too -- it is
    # already resident, so the goal is satisfied with respect to it from t=0.
    outputs_set = set(workload.outputs)
    if len(outputs_set) != len(workload.outputs):
        raise InvalidInput(f"outputs: {workload.id!r} lists the same tensor twice")
    for tensor_id in workload.outputs:
        if tensor_id not in tensor_by_id:
            raise InvalidInput(f"outputs: unknown tensor {tensor_id!r}")

    for tensor in workload.tensors:
        is_leaf = tensor.id in leaf_tensors
        if is_leaf:
            # A leaf must have somewhere to come from, or no action can ever
            # make it resident.
            if not tensor.initial_locations:
                raise InvalidInput(
                    f"leaf tensor {tensor.id!r} has no initial copy: M0 can only "
                    "load a leaf that already exists in DRAM"
                )
        else:
            # A produced tensor starts nowhere; it is materialised in VRAM.
            if tensor.initial_locations:
                raise InvalidInput(
                    f"tensor {tensor.id!r} is produced by "
                    f"{producer_of[tensor.id]!r} but also declares an initial copy "
                    f"in {list(tensor.initial_locations)}"
                )
        if tensor.role == ROLE_WEIGHT and not is_leaf:
            raise InvalidInput(f"weight {tensor.id!r} must be a leaf, not a computed result")

    # --- reachability: no orphans, no cycles ------------------------------
    consumers_of: dict[str, tuple[str, ...]] = {tid: () for tid in tensor_by_id}
    for operation in workload.operations:
        for tensor_id in operation.inputs:
            consumers_of[tensor_id] = consumers_of[tensor_id] + (operation.id,)

    required = _required_operations(workload, operation_by_id, producer_of, outputs_set)
    orphans = sorted(set(operation_by_id) - required)
    if orphans:
        raise InvalidInput(
            f"operation(s) {', '.join(orphans)} do not contribute to any requested "
            "output; M0 validates that every operation is required"
        )

    object.__setattr__(workload, "tensor_by_id", tensor_by_id)
    object.__setattr__(workload, "operation_by_id", operation_by_id)
    object.__setattr__(workload, "producer_of", producer_of)
    object.__setattr__(workload, "consumers_of", consumers_of)
    object.__setattr__(workload, "leaf_tensors", leaf_tensors)


def _required_operations(
    workload: Workload,
    operation_by_id: dict[str, Operation],
    producer_of: dict[str, str],
    outputs_set: set[str],
) -> frozenset[str]:
    """Ancestor closure of the requested outputs; a cycle shows up as no progress."""
    # Cycle detection is unconditional, and must come first. A cycle whose
    # members all happen to lie inside the output's ancestor closure is reached
    # by the traversal below and leaves no orphan behind, so an orphan-based
    # check alone would let it through -- and the engine would then report the
    # graph *infeasible* when the honest answer is that it is not a DAG at all.
    if _has_cycle(workload):
        raise InvalidInput(
            "the operation graph contains a cycle; M0 requires a DAG because "
            "every tensor has exactly one producer and liveness is a graph property"
        )

    required: set[str] = set()
    # A requested output that is a leaf has no producer; it is resident from the
    # start and contributes no required operations.
    frontier = [producer_of[t] for t in outputs_set if t in producer_of]
    while frontier:
        op_id = frontier.pop()
        if op_id in required:
            continue
        required.add(op_id)
        for tensor_id in operation_by_id[op_id].inputs:
            upstream = producer_of.get(tensor_id)
            if upstream is not None:
                frontier.append(upstream)

    return frozenset(required)


def _has_cycle(workload: Workload) -> bool:
    """Depth-first cycle detection over the tensor -> producing-operation edges."""
    producer_of: dict[str, str] = {}
    for operation in workload.operations:
        producer_of[operation.output] = operation.id
    op_by_id = {op.id: op for op in workload.operations}

    WHITE, GREY, BLACK = 0, 1, 2
    colour = {op_id: WHITE for op_id in op_by_id}

    for root in op_by_id:
        if colour[root] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        colour[root] = GREY
        while stack:
            op_id, index = stack.pop()
            inputs = op_by_id[op_id].inputs
            if index < len(inputs):
                stack.append((op_id, index + 1))
                upstream = producer_of.get(inputs[index])
                if upstream is None:
                    continue
                if colour[upstream] == GREY:
                    return True
                if colour[upstream] == WHITE:
                    colour[upstream] = GREY
                    stack.append((upstream, 0))
            else:
                colour[op_id] = BLACK
    return False


def _validate_operation_shape(
    operation: Operation, tensor_by_id: dict[str, Tensor], label: str
) -> None:
    """Check the operand shapes against the semantic op.

    MUL_MAT is written the ggml way, ``[weight, activation]``, and M0 requires
    a genuinely rectangular weight. A square weight would make a transposed or
    swapped operand pair produce identical numbers, so the rectangular case is
    the one that actually tests the convention (``DESIGN.md`` §3).
    """
    output = tensor_by_id[operation.output]
    inputs = [tensor_by_id[tid] for tid in operation.inputs]

    def vector_like(tensor: Tensor) -> bool:
        return tensor.ne[1] == 1 and tensor.ne[2] == 1 and tensor.ne[3] == 1

    if operation.semantic_op == SEMANTIC_MUL_MAT:
        if len(inputs) != 2:
            raise InvalidInput(
                f"{label}: {SEMANTIC_MUL_MAT} takes exactly 2 inputs, got {len(inputs)}"
            )
        weight, activation = inputs
        if not vector_like(activation):
            raise Unsupported(
                f"{label}: activation {activation.id!r} has ne={list(activation.ne)}; "
                "M0 models matrix-vector products only"
            )
        if not vector_like(output):
            raise Unsupported(
                f"{label}: output {output.id!r} has ne={list(output.ne)}; M0 models "
                "matrix-vector products only"
            )
        # ggml: dst = W * x, with ne = [in_features, out_features, 1, 1].
        if weight.ne[0] != activation.ne[0]:
            raise InvalidInput(
                f"{label}: weight {weight.id!r} ne[0]={weight.ne[0]} does not match "
                f"activation {activation.id!r} ne[0]={activation.ne[0]}"
            )
        if weight.ne[1] != output.ne[0]:
            raise InvalidInput(
                f"{label}: weight {weight.id!r} ne[1]={weight.ne[1]} does not match "
                f"output {output.id!r} ne[0]={output.ne[0]}"
            )

    elif operation.semantic_op == SEMANTIC_RELU:
        if len(inputs) != 1:
            raise InvalidInput(
                f"{label}: {SEMANTIC_RELU} takes exactly 1 input, got {len(inputs)}"
            )
        if inputs[0].ne != output.ne:
            raise InvalidInput(
                f"{label}: {SEMANTIC_RELU} must preserve the shape; input "
                f"{inputs[0].id!r} has ne={list(inputs[0].ne)} but output "
                f"{output.id!r} has ne={list(output.ne)}"
            )

    elif operation.semantic_op == SEMANTIC_ADD:
        if len(inputs) != 2:
            raise InvalidInput(
                f"{label}: {SEMANTIC_ADD} takes exactly 2 inputs, got {len(inputs)}"
            )
        if inputs[0].ne != inputs[1].ne or inputs[0].ne != output.ne:
            raise InvalidInput(
                f"{label}: {SEMANTIC_ADD} needs elementwise-identical shapes, got "
                f"{list(inputs[0].ne)}, {list(inputs[1].ne)} -> {list(output.ne)}"
            )


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def load_and_validate(scenario_path: str | Path) -> Scenario:
    """Load a scenario, resolve its workload, and validate both.

    ``scenario.workload_file`` is resolved **relative to the scenario file**,
    never to the shell's working directory, so a scenario is location-independent
    once checked out (``DESIGN.md`` §4.1).
    """
    path = Path(scenario_path)
    raw = _require_mapping(_read_json(path), path.name)

    allowed = {
        "schema_version",
        "id",
        "workload_file",
        "architecture",
        "costs",
        "mapspace",
        "mapper",
        "comment",
    }
    _reject_unknown_keys(raw, allowed, path.name)

    schema_version = _check_schema_version(raw.get("schema_version"), path.name)

    scenario_id = _require_str(raw.get("id"), f"{path.name}.id")

    workload_file = _require_str(raw.get("workload_file"), f"{path.name}.workload_file")
    workload_path = path.parent / workload_file
    workload = load_workload(workload_path)

    architecture = _parse_architecture(raw.get("architecture"), path.name)
    costs = _parse_costs(raw.get("costs"), path.name, workload)
    mapspace = _parse_mapspace(raw.get("mapspace"), path.name, workload)
    mapper_config = _parse_mapper(raw.get("mapper"), path.name)

    return Scenario(
        schema_version=schema_version,
        id=scenario_id,
        workload=workload,
        architecture=architecture,
        costs=costs,
        mapspace=mapspace,
        mapper=mapper_config,
    )


def _parse_architecture(raw: Any, where: str) -> Architecture:
    block = _require_mapping(raw, f"{where}.architecture")
    _reject_unknown_keys(
        block,
        {
            "dram_capacity_bytes",
            "vram_capacity_bytes",
            "runtime_reserved_bytes",
            "allocation_alignment_bytes",
            "gpu_compute_slots",
            "h2d_copy_slots",
        },
        f"{where}.architecture",
    )

    # The invariants themselves live on Architecture.__post_init__, so that a
    # directly constructed value cannot slip past them; here we only coerce and
    # hand over. dram_capacity_bytes is passed through as-is so the dataclass can
    # be the one to reject a non-null value.
    return Architecture(
        vram_capacity_bytes=_require_int(
            block.get("vram_capacity_bytes"),
            f"{where}.architecture.vram_capacity_bytes",
            minimum=1,
        ),
        runtime_reserved_bytes=_require_int(
            block.get("runtime_reserved_bytes", 0),
            f"{where}.architecture.runtime_reserved_bytes",
        ),
        allocation_alignment_bytes=_require_int(
            block.get("allocation_alignment_bytes", 1),
            f"{where}.architecture.allocation_alignment_bytes",
            minimum=1,
        ),
        gpu_compute_slots=_require_int(
            block.get("gpu_compute_slots", 1),
            f"{where}.architecture.gpu_compute_slots",
            minimum=1,
        ),
        h2d_copy_slots=_require_int(
            block.get("h2d_copy_slots", 1),
            f"{where}.architecture.h2d_copy_slots",
            minimum=1,
        ),
        dram_capacity_bytes=block.get("dram_capacity_bytes", None),
    )


def _parse_costs(raw: Any, where: str, workload: Workload) -> Costs:
    block = _require_mapping(raw, f"{where}.costs")
    _reject_unknown_keys(block, {"source", "compute", "h2d"}, f"{where}.costs")

    source = _require_str(block.get("source"), f"{where}.costs.source")

    compute_raw = _require_mapping(block.get("compute"), f"{where}.costs.compute")
    compute: dict[str, ComputeCost] = {}
    for op_id, entry in compute_raw.items():
        label = f"{where}.costs.compute[{op_id!r}]"
        if op_id not in workload.operation_by_id:
            raise InvalidInput(f"{label}: no such operation in the workload")
        _expect(not isinstance(entry, list), f"{label} must be a JSON object")
        body = _require_mapping(entry, label)
        _reject_unknown_keys(body, {"duration_ns", "workspace_bytes"}, label)
        compute[op_id] = ComputeCost(
            duration_ns=_require_int(body.get("duration_ns"), f"{label}.duration_ns", minimum=1),
            workspace_bytes=_require_int(
                body.get("workspace_bytes", 0), f"{label}.workspace_bytes"
            ),
        )

    missing = sorted(set(workload.operation_by_id) - set(compute))
    if missing:
        raise InvalidInput(
            f"{where}.costs.compute is missing an entry for operation(s) "
            f"{', '.join(missing)}; M0 never guesses a cost"
        )

    h2d_raw = _require_mapping(block.get("h2d"), f"{where}.costs.h2d")
    h2d: dict[str, H2DCost] = {}
    for tensor_id, entry in h2d_raw.items():
        label = f"{where}.costs.h2d[{tensor_id!r}]"
        if tensor_id not in workload.tensor_by_id:
            raise InvalidInput(f"{label}: no such tensor in the workload")
        body = _require_mapping(entry, label)
        _reject_unknown_keys(body, {"duration_ns"}, label)
        h2d[tensor_id] = H2DCost(
            duration_ns=_require_int(body.get("duration_ns"), f"{label}.duration_ns", minimum=1)
        )

    return Costs(source=source, compute=compute, h2d=h2d)


def _parse_mapspace(raw: Any, where: str, workload: Workload) -> Mapspace:
    block = _require_mapping(raw, f"{where}.mapspace")
    _reject_unknown_keys(
        block,
        {
            "compute_device",
            "copy_tensor_ids",
            "allow_copy_compute_overlap",
            "allow_eviction",
            "allow_recomputation",
        },
        f"{where}.mapspace",
    )

    device = _require_str(block.get("compute_device", COPY_DEVICE), f"{where}.mapspace.compute_device")
    if device != COPY_DEVICE:
        raise Unsupported(
            f"{where}.mapspace.compute_device is {device!r}; M0 runs on {COPY_DEVICE!r} only"
        )

    ids_raw = _require_list(block.get("copy_tensor_ids"), f"{where}.mapspace.copy_tensor_ids")
    copy_ids: list[str] = []
    for i, value in enumerate(ids_raw):
        value = _require_str(value, f"{where}.mapspace.copy_tensor_ids[{i}]")
        if value not in workload.tensor_by_id:
            raise InvalidInput(
                f"{where}.mapspace.copy_tensor_ids[{i}]: unknown tensor {value!r}"
            )
        if value in copy_ids:
            raise InvalidInput(f"{where}.mapspace.copy_tensor_ids lists {value!r} twice")
        copy_ids.append(value)

    # A copy is only meaningful for a tensor that exists in DRAM and is not yet
    # in VRAM. Requiring this up front means the engine can treat "not in the
    # mapspace" as a static property rather than re-deriving it per action.
    #
    # There is no separate "must be a leaf" check here: requiring a DRAM copy
    # already implies it, because _validate_workload forbids a produced tensor
    # from declaring any initial location. Adding the check back would be dead
    # code that documents a rule it never enforces.
    for tensor_id in copy_ids:
        tensor = workload.tensor_by_id[tensor_id]
        if LOC_DRAM not in tensor.initial_locations:
            raise InvalidInput(
                f"{where}.mapspace.copy_tensor_ids includes {tensor_id!r}, but that "
                "tensor has no initial DRAM copy to load from; a copy source must be "
                "a leaf resident in DRAM, since a computed result is materialised in "
                "VRAM and M0 has no recomputation"
            )
        if LOC_VRAM in tensor.initial_locations:
            raise InvalidInput(
                f"{where}.mapspace.copy_tensor_ids includes {tensor_id!r}, but that "
                "tensor is already resident in VRAM initially; M0 never reloads a "
                "tensor it has not evicted"
            )

    recomputation = _require_bool(
        block.get("allow_recomputation", False), f"{where}.mapspace.allow_recomputation"
    )
    if recomputation:
        raise Unsupported(
            f"{where}.mapspace.allow_recomputation is true; M0 has no recomputation "
            "and must not silently ignore the request"
        )

    return Mapspace(
        compute_device=device,
        copy_tensor_ids=tuple(copy_ids),
        allow_copy_compute_overlap=_require_bool(
            block.get("allow_copy_compute_overlap"),
            f"{where}.mapspace.allow_copy_compute_overlap",
        ),
        allow_eviction=_require_bool(
            block.get("allow_eviction"), f"{where}.mapspace.allow_eviction"
        ),
        allow_recomputation=False,
    )


def _parse_mapper(raw: Any, where: str) -> MapperConfig:
    block = _require_mapping(raw, f"{where}.mapper")
    _reject_unknown_keys(
        block, {"algorithm", "max_expanded_states", "wall_time_limit_s"}, f"{where}.mapper"
    )

    algorithm = _require_str(block.get("algorithm"), f"{where}.mapper.algorithm")
    if algorithm != ALGORITHM_UNIFORM_COST:
        raise Unsupported(
            f"{where}.mapper.algorithm is {algorithm!r}; M0 implements "
            f"{ALGORITHM_UNIFORM_COST!r} only"
        )

    max_states = _require_int(
        block.get("max_expanded_states", 0), f"{where}.mapper.max_expanded_states"
    )

    limit = block.get("wall_time_limit_s", 0)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise InvalidInput(f"{where}.mapper.wall_time_limit_s must be a number, got {limit!r}")
    limit = float(limit)
    if limit < 0 or limit != limit:  # NaN fails self-equality
        raise InvalidInput(
            f"{where}.mapper.wall_time_limit_s must be finite and >= 0, got {limit!r}"
        )

    return MapperConfig(
        algorithm=algorithm, max_expanded_states=max_states, wall_time_limit_s=limit
    )


# ---------------------------------------------------------------------------
# Initial capacity
# ---------------------------------------------------------------------------


def check_initial_capacity(scenario: Scenario) -> None:
    """Raise :class:`InitialCapacityExceeded` if the initial layout cannot fit.

    Called before the search starts and again before a mapping replay, so both
    entry points agree on the verdict (``DESIGN.md`` §8).
    """
    architecture = scenario.architecture
    used = architecture.runtime_reserved_bytes
    resident: list[str] = []
    for tensor in scenario.workload.tensors:
        if LOC_VRAM in tensor.initial_locations:
            used += scenario.size_alloc(tensor.id)
            resident.append(tensor.id)
    if used > architecture.vram_capacity_bytes:
        raise InitialCapacityExceeded(
            f"initial layout needs {used} bytes (runtime reserve "
            f"{architecture.runtime_reserved_bytes} + tensors "
            f"{', '.join(sorted(resident))}) but vram_capacity_bytes is "
            f"{architecture.vram_capacity_bytes}"
        )


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _canonical_workload(workload: Workload) -> dict[str, Any]:
    """Semantic content of a workload.

    ``origin`` contributes only its ``kind``. The GGML version and sample name
    are provenance: they say where the graph came from, not what it does, and
    including them would invalidate every stored mapping the moment the same
    graph was re-exported from a different build (``DESIGN.md`` §4.3).

    Tensors and operations are sorted by id, so reordering the JSON file --
    which changes nothing semantically -- does not change the fingerprint.
    """
    return {
        "id": workload.id,
        "origin": {"kind": workload.origin.kind},
        "tensors": [
            {
                "id": t.id,
                "name": t.name,
                "role": t.role,
                "dtype": t.dtype,
                "ne": list(t.ne),
                "nb": list(t.nb),
                "storage_bytes": t.storage_bytes,
                "initial_locations": sorted(t.initial_locations),
            }
            for t in sorted(workload.tensors, key=lambda t: t.id)
        ],
        "operations": [
            {
                "id": op.id,
                "semantic_op": op.semantic_op,
                "ggml_op": op.ggml_op,
                "unary_op": op.unary_op,
                "op_params": list(op.op_params),
                # Operand order is semantic for MUL_MAT (it is [weight, activation]),
                # so inputs keep their declared order while everything else sorts.
                "inputs": list(op.inputs),
                "output": op.output,
            }
            for op in sorted(workload.operations, key=lambda op: op.id)
        ],
        "outputs": list(workload.outputs),
    }


def _canonical_scenario(scenario: Scenario) -> dict[str, Any]:
    """Semantic content of a scenario.

    ``workload_file`` is excluded -- it is an external path, exactly what the
    spec says to drop -- and the workload is represented by its fingerprint
    instead, so a scenario fingerprint still pins the exact graph without
    depending on where the file happens to live.
    """
    architecture = scenario.architecture
    mapspace = scenario.mapspace
    mapper = scenario.mapper
    return {
        "schema_version": scenario.schema_version,
        "id": scenario.id,
        "workload_fingerprint": workload_fingerprint(scenario.workload),
        "architecture": {
            "dram_capacity_bytes": None,
            "vram_capacity_bytes": architecture.vram_capacity_bytes,
            "runtime_reserved_bytes": architecture.runtime_reserved_bytes,
            "allocation_alignment_bytes": architecture.allocation_alignment_bytes,
            "gpu_compute_slots": architecture.gpu_compute_slots,
            "h2d_copy_slots": architecture.h2d_copy_slots,
        },
        "costs": {
            "source": scenario.costs.source,
            "compute": [
                {
                    "operation_id": op_id,
                    "duration_ns": cost.duration_ns,
                    "workspace_bytes": cost.workspace_bytes,
                }
                for op_id, cost in sorted(scenario.costs.compute.items())
            ],
            "h2d": [
                {"tensor_id": tensor_id, "duration_ns": cost.duration_ns}
                for tensor_id, cost in sorted(scenario.costs.h2d.items())
            ],
        },
        "mapspace": {
            "compute_device": mapspace.compute_device,
            "copy_tensor_ids": sorted(mapspace.copy_tensor_ids),
            "allow_copy_compute_overlap": mapspace.allow_copy_compute_overlap,
            "allow_eviction": mapspace.allow_eviction,
            "allow_recomputation": mapspace.allow_recomputation,
        },
        "mapper": {
            "algorithm": mapper.algorithm,
            "max_expanded_states": mapper.max_expanded_states,
            "wall_time_limit_s": mapper.wall_time_limit_s,
        },
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    """SHA-256 over canonically serialised JSON.

    Sorted keys, no insignificant whitespace, UTF-8 -- so the digest depends on
    the content and nothing else (``DESIGN.md`` §4.3).
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def workload_fingerprint(workload: Workload) -> str:
    """Stable digest of a workload's semantics."""
    return _fingerprint(_canonical_workload(workload))


def scenario_fingerprint(scenario: Scenario) -> str:
    """Stable digest of a scenario's semantics, including its workload's."""
    return _fingerprint(_canonical_scenario(scenario))
