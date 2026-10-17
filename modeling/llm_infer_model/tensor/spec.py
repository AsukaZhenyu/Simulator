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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_M1",
    "SUPPORTED_SCHEMA_VERSIONS",
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
    # --- 0.2 ---
    "M1Costs",
    "M1Goal",
    "M1Mapspace",
    "TransferCost",
    "aligned_size",
    "load_workload",
    "load_and_validate",
    "build_scenario",
    "reachable_locations",
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
    "DTYPE_I32",
    "DTYPE_F16",
    "DTYPE_Q4_K",
    "DTYPE_Q6_K",
    "ggml_contiguous_layout",
    "LOC_DRAM",
    "LOC_VRAM",
    "OP_DONE",
    "OP_NOT_STARTED",
    "OP_RUNNING",
    "ORIGIN_GGML",
    "ORIGIN_SYNTHETIC",
    "ORIGIN_SYNTHETIC_LAYER",
    "ORIGIN_MEASURED_MACRO",
    "RESOURCE_COMPUTE",
    "RESOURCE_COPY",
    "ROLE_INPUT",
    "ROLE_INTERMEDIATE",
    "ROLE_OUTPUT",
    "ROLE_WEIGHT",
    "SEMANTIC_ADD",
    "SEMANTIC_MUL_MAT",
    "SEMANTIC_RELU",
    # M2 的宏任务语义（M2_LAYER_DESIGN.md §3）。它们不是 GGML 算子，只在一个
    # 声明了 synthetic_layer / measured_macro 来源的负载上被接受。
    "SEMANTIC_PROJECTION",
    "SEMANTIC_ATTENTION",
    "SEMANTIC_STATE_UPDATE",
    "SEMANTIC_ROUTER",
    "SEMANTIC_EXPERT_FFN",
    "SEMANTIC_COMBINE",
    "SEMANTIC_DENSE_FFN",
    # 0.2 的取值域：位置多了一个 SSD，设备、路线、资源各是一组新名字。
    # 与上面那些常量同理，调用方按名比较它们，所以属于公开面。
    "DEVICE_CPU",
    "DEVICE_GPU",
    "DEVICE_OF_DOMAIN",
    "DOMAIN_OF_DEVICE",
    "DOMAINS",
    "LOC_SSD",
    "M1_RESOURCES",
    "RESOURCE_CPU_COMPUTE",
    "RESOURCE_GPU_COMPUTE",
    "RESOURCE_OF_DEVICE",
    "RESOURCE_OF_ROUTE",
    "RESOURCE_PCIE_COPY",
    "RESOURCE_SSD_IO",
    "ROUTE_DRAM_TO_SSD",
    "ROUTE_DRAM_TO_VRAM",
    "ROUTE_ENDS",
    "ROUTE_SSD_TO_DRAM",
    "ROUTE_VRAM_TO_DRAM",
    "ROUTES",
]

# All M0 JSON documents -- workload, scenario and mapping alike -- carry this
# same version string (DESIGN.md §4).
SCHEMA_VERSION = "0.1"

# M1 uses 0.2 (``M1_DESIGN.md`` §4). The two are not one schema with optional
# fields: 0.1 fixes one device, one copy route and an unbounded DRAM source,
# while 0.2 makes all three explicit. ``SUPPORTED_SCHEMA_VERSIONS`` is what the
# parsers accept; everything downstream branches on the version string carried
# on the ``Scenario``, so a 0.1 document can never accidentally acquire 0.2
# semantics by having a field default to something new.
SCHEMA_VERSION_M1 = "0.2"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_M1)

# ---------------------------------------------------------------------------
# Enumerations. Kept as plain strings rather than enum.Enum so that the values
# round-trip through JSON artefacts byte-for-byte (DESIGN.md §8).
# ---------------------------------------------------------------------------

DTYPE_F32 = "GGML_TYPE_F32"
DTYPE_I32 = "GGML_TYPE_I32"
DTYPE_F16 = "GGML_TYPE_F16"
DTYPE_Q4_K = "GGML_TYPE_Q4_K"
DTYPE_Q6_K = "GGML_TYPE_Q6_K"
_DTYPE_SIZE_BYTES = {DTYPE_F32: 4}
# GGML ne[0] is the reduction/row dimension. Quantized strides use the size
# of one 256-element block, not a rounded bytes-per-element estimate. These
# formats describe M2 macro storage only; ordinary M0/M1 ops remain F32.
_GGML_BLOCK_STORAGE = {
    DTYPE_F32: (1, 4),
    DTYPE_I32: (1, 4),
    DTYPE_F16: (1, 2),
    DTYPE_Q4_K: (256, 144),
    DTYPE_Q6_K: (256, 210),
}

ROLE_INPUT = "input"
ROLE_WEIGHT = "weight"
ROLE_INTERMEDIATE = "intermediate"
ROLE_OUTPUT = "output"
_ROLES = frozenset({ROLE_INPUT, ROLE_WEIGHT, ROLE_INTERMEDIATE, ROLE_OUTPUT})

SEMANTIC_MUL_MAT = "MUL_MAT"
SEMANTIC_RELU = "RELU"
SEMANTIC_ADD = "ADD"

# M2's schedulable macro-tasks (M2_LAYER_DESIGN.md §3). They are deliberately
# NOT GGML ops: they name task groups a layer descriptor declares, and their
# shapes are checked by the M2 layer validator rather than by the matrix-vector
# rules below. ``dense_ffn`` is the acceptance's own third task name
# (M2_LAYER_ACCEPTANCE.md §2) and sits beside the design's six.
SEMANTIC_PROJECTION = "projection"
SEMANTIC_ATTENTION = "attention"
SEMANTIC_STATE_UPDATE = "state_update"
SEMANTIC_ROUTER = "router"
SEMANTIC_EXPERT_FFN = "expert_ffn"
SEMANTIC_COMBINE = "combine"
SEMANTIC_DENSE_FFN = "dense_ffn"
# Materialized row execution is a synthetic graph/storage contract only. These
# labels do not certify numerical GGML or measured physical kernel semantics.
_SYNTHETIC_ROW_OPS = frozenset({"gather_rows", "scatter_rows", "combine_rows", "expert_gate",
                                  "expert_up", "expert_swiglu", "expert_down", "attention_core", "attention_output",
                                  "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down", "ffn_sum", "ffn_residual",
                                  "attention_qk", "attention_softmax", "attention_pv", "attention_projection"})
_M2_MACRO_OPS = frozenset(
    {
        SEMANTIC_PROJECTION,
        SEMANTIC_ATTENTION,
        SEMANTIC_STATE_UPDATE,
        SEMANTIC_ROUTER,
        SEMANTIC_EXPERT_FFN,
        SEMANTIC_COMBINE,
        SEMANTIC_DENSE_FFN,
    }
) | _SYNTHETIC_ROW_OPS

_SEMANTIC_OPS = frozenset({SEMANTIC_MUL_MAT, SEMANTIC_RELU, SEMANTIC_ADD}) | _M2_MACRO_OPS

LOC_DRAM = "dram"
LOC_VRAM = "vram"
# M1.3 makes the SSD a real third location: tensors can start there and, unlike
# M1.2, the engine can also write immutable snapshots back to it. It is still
# not an allocation domain -- there is no SSD capacity, no alignment and no SSD
# eviction action; what has to be tracked is only *which* SSD copies exist and
# whether a write has finished (M1_DESIGN.md §3).
LOC_SSD = "ssd"
_LOCATIONS = frozenset({LOC_DRAM, LOC_VRAM})
_LOCATIONS_M1 = frozenset({LOC_DRAM, LOC_VRAM, LOC_SSD})

# Storage domains = locations a copy can live on and that are *ledgered*. SSD is
# deliberately absent: it holds unbounded static scene content plus runtime
# snapshots, and the design gives it no capacity constraint.
DOMAINS = (LOC_DRAM, LOC_VRAM)
# Where a 0.2 goal may ask for an output. An SSD target is delivered only when
# its write has completed (M1_DESIGN.md §4).
GOAL_LOCATIONS_M1 = (LOC_DRAM, LOC_VRAM, LOC_SSD)

# Compute devices.
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
_DEVICES = frozenset({DEVICE_CPU, DEVICE_GPU})
# Which storage domain a device reads its operands from and writes its result
# to. This is the whole of the "CPU computes in DRAM, GPU computes in VRAM"
# rule (M1_DESIGN.md §5).
DOMAIN_OF_DEVICE = {DEVICE_CPU: LOC_DRAM, DEVICE_GPU: LOC_VRAM}
DEVICE_OF_DOMAIN = {LOC_DRAM: DEVICE_CPU, LOC_VRAM: DEVICE_GPU}

# Transfer routes. Each is a fixed (source, destination) pair; the direction is
# part of the route's identity, never a per-action choice.
ROUTE_DRAM_TO_VRAM = "dram_to_vram"
ROUTE_VRAM_TO_DRAM = "vram_to_dram"
ROUTE_SSD_TO_DRAM = "ssd_to_dram"
ROUTE_DRAM_TO_SSD = "dram_to_ssd"
ROUTES = (
    ROUTE_DRAM_TO_VRAM,
    ROUTE_VRAM_TO_DRAM,
    ROUTE_SSD_TO_DRAM,
    ROUTE_DRAM_TO_SSD,
)
ROUTE_ENDS = {
    ROUTE_DRAM_TO_VRAM: (LOC_DRAM, LOC_VRAM),
    ROUTE_VRAM_TO_DRAM: (LOC_VRAM, LOC_DRAM),
    ROUTE_SSD_TO_DRAM: (LOC_SSD, LOC_DRAM),
    ROUTE_DRAM_TO_SSD: (LOC_DRAM, LOC_SSD),
}

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

# M2's abstract layer graphs (M2_LAYER_DESIGN.md §3:36). An abstract macro-task
# graph must be labelled so it is never read as a real GGML export, and the
# label is load-bearing rather than decorative: the macro semantics above are
# accepted *only* on a workload declaring one of these two kinds. Without that
# gate, adding them to ``_SEMANTIC_OPS`` would quietly widen what a 0.1 or 0.2
# document may say, not merely what it may hash to.
ORIGIN_SYNTHETIC_LAYER = "synthetic_layer"
ORIGIN_MEASURED_MACRO = "measured_macro"
_M2_ORIGIN_KINDS = frozenset({ORIGIN_SYNTHETIC_LAYER, ORIGIN_MEASURED_MACRO})

_ORIGIN_KINDS = frozenset({ORIGIN_GGML, ORIGIN_SYNTHETIC}) | _M2_ORIGIN_KINDS

# Resource names used in states, events and error reports.
#
# The two 0.1 names are kept exactly as they were: they appear in stored events
# and states artefacts and in every M0 test, so renaming them would rewrite
# history rather than describe it. In 0.2 the GPU compute resource *is* this
# same name, and the new ones sit beside it.
RESOURCE_COMPUTE = "gpu_compute"
RESOURCE_COPY = "h2d_copy"

RESOURCE_CPU_COMPUTE = "cpu_compute"
RESOURCE_GPU_COMPUTE = RESOURCE_COMPUTE
RESOURCE_PCIE_COPY = "pcie_copy"
# One slot shared by SSD reads and SSD writes: the two directions are mutually
# exclusive, while the SSD and the PCIe engine can run together
# (M1_DESIGN.md §3).
RESOURCE_SSD_IO = "ssd_io"

# M1_DESIGN.md §3: exactly one slot of each. Declared here so the parser, the
# engine and the report all read the same list.
M1_RESOURCES = (
    RESOURCE_CPU_COMPUTE,
    RESOURCE_GPU_COMPUTE,
    RESOURCE_PCIE_COPY,
    RESOURCE_SSD_IO,
)
RESOURCE_OF_DEVICE = {
    DEVICE_CPU: RESOURCE_CPU_COMPUTE,
    DEVICE_GPU: RESOURCE_GPU_COMPUTE,
}
_RESOURCE_SLOT_FIELDS = {
    RESOURCE_CPU_COMPUTE: "cpu_compute_slots",
    RESOURCE_GPU_COMPUTE: "gpu_compute_slots",
    RESOURCE_PCIE_COPY: "pcie_copy_slots",
    RESOURCE_SSD_IO: "ssd_io_slots",
}

# Which resource a route's copy occupies. H2D and D2H share the one PCIe
# engine; both SSD directions share the one SSD slot (M1_DESIGN.md §3, §6).
RESOURCE_OF_ROUTE = {
    ROUTE_DRAM_TO_VRAM: RESOURCE_PCIE_COPY,
    ROUTE_VRAM_TO_DRAM: RESOURCE_PCIE_COPY,
    ROUTE_SSD_TO_DRAM: RESOURCE_SSD_IO,
    ROUTE_DRAM_TO_SSD: RESOURCE_SSD_IO,
}

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
    """One node of the compute DAG.

    ``output`` is the tensor the operation is named after; ``extra_outputs``
    holds any further tensors it materialises. For every M0/M1 graph the tuple
    is empty and ``outputs`` is exactly ``(output,)``. M2 needs more than one:
    its ``attention`` macro-task produces the new hidden state *and* one
    immutable KV block per new token, in a single schedulable step. Splitting
    that into two operations would put two rows in the trace where the
    acceptance fixes one task on one CPU slot (M2_LAYER_ACCEPTANCE.md §2), and
    an append task carrying 0 ns is refused by the cost parser anyway.
    """

    id: str
    semantic_op: str
    ggml_op: str
    unary_op: str | None
    op_params: tuple[int, ...]
    inputs: tuple[str, ...]
    output: str
    extra_outputs: tuple[str, ...] = ()

    @property
    def outputs(self) -> tuple[str, ...]:
        """Every tensor this operation produces, the named one first."""
        return (self.output,) + self.extra_outputs


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
    # 0.1 spells DRAM as "null, i.e. unbounded" and the 0.1 parser still refuses
    # any other value there (``_parse_architecture``). It is not refused *here*
    # any more, because 0.2 makes it the real DRAM budget; moving the check to
    # the parser is what keeps a 0.1 document from silently acquiring a 0.2
    # field while leaving a 0.2 document able to set it.
    dram_capacity_bytes: int | None = None

    # --- 0.2 only. Defaults describe the 0.1 machine exactly: no DRAM reserve,
    # both domains aligned and reserved as the 0.1 single pair says, one slot of
    # the two new resources. ``None`` therefore means "0.1 wrote it this way and
    # there is one number for both domains", never "unset".
    dram_runtime_reserved_bytes: int = 0
    vram_runtime_reserved_bytes: int | None = None
    dram_alignment_bytes: int | None = None
    vram_alignment_bytes: int | None = None
    cpu_compute_slots: int = 1
    pcie_copy_slots: int = 1
    ssd_io_slots: int = 1

    # -- per-domain accessors. Everything downstream goes through these rather
    # than reading a field, so the 0.1/0.2 spelling difference lives in one
    # place.

    def capacity_of(self, domain: str) -> int | None:
        """Bytes available in ``domain``; ``None`` means unbounded.

        ``None`` covers two different situations on purpose: 0.1's DRAM, which
        has no budget at all, and the SSD, which the design gives no capacity
        either. Neither is ever charged, so neither needs a number
        (``M1_DESIGN.md`` §3, §5).
        """
        if domain == LOC_DRAM:
            return self.dram_capacity_bytes
        if domain == LOC_VRAM:
            return self.vram_capacity_bytes
        return None

    def reserved_of(self, domain: str) -> int:
        """Static runtime reservation charged to ``domain`` from t=0."""
        if domain == LOC_DRAM:
            return self.dram_runtime_reserved_bytes
        if domain == LOC_VRAM:
            if self.vram_runtime_reserved_bytes is None:
                return self.runtime_reserved_bytes
            return self.vram_runtime_reserved_bytes
        return 0

    def alignment_of(self, domain: str) -> int:
        """Allocation granularity in ``domain``."""
        if domain == LOC_DRAM:
            if self.dram_alignment_bytes is None:
                return self.allocation_alignment_bytes
            return self.dram_alignment_bytes
        if domain == LOC_VRAM:
            if self.vram_alignment_bytes is None:
                return self.allocation_alignment_bytes
            return self.vram_alignment_bytes
        # The SSD is not an allocation domain: no capacity, no alignment, no
        # eviction. A caller asking for one is asking a question M1 does not
        # answer, and one is the answer that changes nothing.
        return 1

    def __post_init__(self) -> None:
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
            ("cpu_compute_slots", self.cpu_compute_slots),
            ("pcie_copy_slots", self.pcie_copy_slots),
            ("ssd_io_slots", self.ssd_io_slots),
        ):
            if slots != 1:
                raise Unsupported(
                    f"architecture.{name} is {slots}; M0 models exactly one resource "
                    "of each kind"
                )
        for name, capacity, reserved in (
            ("dram", self.dram_capacity_bytes, self.dram_runtime_reserved_bytes),
            ("vram", self.vram_capacity_bytes, self.reserved_of(LOC_VRAM)),
        ):
            if capacity is None:
                continue
            if capacity < 1:
                raise InvalidInput(
                    f"architecture.{name}_capacity_bytes must be >= 1, got {capacity}"
                )
            if reserved < 0:
                raise InvalidInput(
                    f"architecture.{name}_runtime_reserved_bytes must be >= 0, got "
                    f"{reserved}"
                )
            # The static reservation is charged from t=0, so a reservation that
            # does not fit is not a search failure but an impossible starting
            # point -- reported as `initial_capacity_exceeded`, never as a
            # capacity the search could free up (M1_ACCEPTANCE.md §8).
            if reserved > capacity:
                raise InvalidInput(
                    f"architecture.{name}_runtime_reserved_bytes ({reserved}) exceeds "
                    f"{name}_capacity_bytes ({capacity})"
                )
        for name in ("dram_alignment_bytes", "vram_alignment_bytes"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise InvalidInput(f"architecture.{name} must be >= 1, got {value}")


@dataclass(frozen=True)
class ComputeCost:
    """Duration of one compute node, plus the transient workspace it needs."""

    duration_ns: int
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        # Python callers must obey the same contract as parsed cost tables:
        # invalid durations otherwise let ADVANCE move simulated time backwards.
        _require_int(self.duration_ns, "ComputeCost.duration_ns", minimum=1)
        _require_int(self.workspace_bytes, "ComputeCost.workspace_bytes")


@dataclass(frozen=True)
class H2DCost:
    """Duration of one host-to-device copy."""

    duration_ns: int

    def __post_init__(self) -> None:
        _require_int(self.duration_ns, "H2DCost.duration_ns", minimum=1)


@dataclass(frozen=True)
class Costs:
    """The cost model. M0 accepts only externally supplied fixed costs."""

    source: str
    compute: dict[str, ComputeCost]
    h2d: dict[str, H2DCost]


@dataclass(frozen=True)
class TransferCost:
    """Duration of one transfer along one route.

    Same shape as :class:`H2DCost` and deliberately a different name: in 0.2 the
    direction is part of what is being costed, and a reader who sees
    ``H2DCost`` in an M1 context would reasonably assume host-to-device.
    """

    duration_ns: int

    def __post_init__(self) -> None:
        _require_int(self.duration_ns, "TransferCost.duration_ns", minimum=1)


@dataclass(frozen=True)
class M1Costs:
    """The 0.2 cost model: per-device compute, per-route transfer.

    Kept apart from :class:`Costs` rather than folded into it. A 0.1 ``Costs``
    is a single duration per operation and a single duration per tensor; 0.2 is
    indexed twice over. One dataclass with both shapes would have to let one set
    of fields be empty, and every reader would then have to know which. Two
    types make the version test unavoidable.
    """

    source: str
    compute: dict[str, dict[str, ComputeCost]]
    transfer: dict[str, dict[str, TransferCost]]

    def compute_cost(self, operation_id: str, device: str) -> ComputeCost | None:
        return self.compute.get(operation_id, {}).get(device)

    def transfer_cost(self, route: str, tensor_id: str) -> TransferCost | None:
        return self.transfer.get(route, {}).get(tensor_id)


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
class M1Mapspace:
    """0.2's mapspace: which device may run which op, which tensor may take
    which route.

    The 0.1 single ``compute_device`` and flat ``copy_tensor_ids`` are gone
    because they *are* the answer to a question 0.2 asks: "gpu for every
    operation" and "dram_to_vram for these tensors". Expressing 0.1 as this
    shape would have been possible; refusing to parse 0.2 into the old shape is
    what keeps a 0.1 document from silently gaining a CPU option.
    """

    allowed_devices: dict[str, tuple[str, ...]]
    allowed_transfers: dict[str, tuple[str, ...]]
    allow_eviction: bool
    allow_copy_compute_overlap: bool
    allow_cpu_gpu_overlap: bool
    allow_recomputation: bool = False
    # Optional M2 causal-replay gate: selective expert transfers wait until
    # the router that reveals the selection has completed. Resident copies
    # remain usable; this gates transfer actions, never local computation.
    transfer_dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dependencies = _require_mapping(
            self.transfer_dependencies, "M1Mapspace.transfer_dependencies"
        )
        for tensor_id, operation_ids in dependencies.items():
            _require_str(tensor_id, "M1Mapspace.transfer_dependencies tensor id")
            if not isinstance(operation_ids, tuple):
                raise InvalidInput(
                    f"M1Mapspace.transfer_dependencies[{tensor_id!r}] must be a tuple"
                )
            seen: set[str] = set()
            for operation_id in operation_ids:
                _require_str(operation_id, "transfer dependency operation id")
                if operation_id in seen:
                    raise InvalidInput(
                        f"transfer_dependencies[{tensor_id!r}] lists "
                        f"{operation_id!r} twice"
                    )
                seen.add(operation_id)

    def devices_for(self, operation_id: str) -> tuple[str, ...]:
        return self.allowed_devices.get(operation_id, ())

    def tensors_for(self, route: str) -> tuple[str, ...]:
        return self.allowed_transfers.get(route, ())

    def allows(self, route: str, tensor_id: str) -> bool:
        return tensor_id in self.allowed_transfers.get(route, ())


@dataclass(frozen=True)
class M1Goal:
    """Where each requested output has to end up.

    One location per output, mandatory and exhaustive: "the result may be in
    DRAM or VRAM" is not a goal, it is the absence of one. A missing or extra
    key is an input error rather than a default, because the default would
    silently change what the search is optimising.
    """

    output_locations: dict[str, str]

    def location_of(self, tensor_id: str) -> str | None:
        return self.output_locations.get(tensor_id)


@dataclass(frozen=True)
class MapperConfig:
    """Search budget. Zero means unlimited for both fields."""

    algorithm: str
    max_expanded_states: int = 0
    wall_time_limit_s: float = 0.0


@dataclass(frozen=True)
class Scenario:
    """A workload bound to an architecture, a cost model, and a mapspace.

    ``costs`` and ``mapspace`` are typed loosely because 0.1 and 0.2 genuinely
    have different shapes; ``schema_version`` is what tells them apart, and
    every accessor below branches on it exactly once.
    """

    schema_version: str
    id: str
    workload: Workload
    architecture: Architecture
    costs: Costs | M1Costs
    mapspace: Mapspace | M1Mapspace
    mapper: MapperConfig
    # 0.2 only. ``None`` for a 0.1 scenario, whose goal is "every requested
    # output resident in VRAM" (DESIGN.md §5).
    goal: M1Goal | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mapspace, M1Mapspace) and self.mapspace.transfer_dependencies:
            if not self.is_m1 or self.workload.origin.kind not in _M2_ORIGIN_KINDS:
                raise Unsupported(
                    "transfer_dependencies requires an M1 scenario with M2 "
                    "synthetic_layer or measured_macro workload provenance"
                )
            for tensor_id, operation_ids in self.mapspace.transfer_dependencies.items():
                if tensor_id not in self.workload.tensor_by_id:
                    raise InvalidInput(
                        f"transfer_dependencies: unknown tensor {tensor_id!r}"
                    )
                for operation_id in operation_ids:
                    if operation_id not in self.workload.operation_by_id:
                        raise InvalidInput(
                            f"transfer_dependencies[{tensor_id!r}]: "
                            f"unknown operation {operation_id!r}"
                        )

    @property
    def is_m1(self) -> bool:
        return self.schema_version == SCHEMA_VERSION_M1

    def size_alloc(self, tensor_id: str, domain: str = LOC_VRAM) -> int:
        """Bytes ``tensor_id`` occupies in ``domain``, after that domain's alignment."""
        tensor = self.workload.tensor_by_id[tensor_id]
        return aligned_size(
            tensor.storage_bytes, self.architecture.alignment_of(domain)
        )

    def workspace_alloc(self, operation_id: str, device: str = DEVICE_GPU) -> int:
        """Workspace reserved for a running compute, in that device's domain."""
        assert isinstance(self.costs, M1Costs) or device == DEVICE_GPU
        cost = (
            self.costs.compute_cost(operation_id, device)
            if isinstance(self.costs, M1Costs)
            else self.costs.compute[operation_id]
        )
        assert cost is not None
        domain = DOMAIN_OF_DEVICE[device] if self.is_m1 else LOC_VRAM
        return aligned_size(cost.workspace_bytes, self.architecture.alignment_of(domain))

    def goal_location(self, tensor_id: str) -> str:
        """Where ``tensor_id`` has to end up for the goal to hold.

        0.1 has one answer for every output -- VRAM -- and it is spelled here
        rather than at each call site so that "0.1 means VRAM" appears once.
        """
        if self.goal is None:
            return LOC_VRAM
        location = self.goal.location_of(tensor_id)
        assert location is not None  # checked exhaustively by the parser
        return location


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
    """Every document declares ``schema_version``; the parser accepts 0.1 and 0.2.

    A missing version is an error rather than a default: an unversioned file is
    of unknown vintage, and guessing would defeat the point of versioning it.

    Accepting a second version is not the same as reinterpreting the first: the
    returned string is carried on the ``Scenario`` and every later decision
    branches on it, so a 0.1 document keeps 0.1 semantics exactly.
    """
    version = _require_str(raw, f"{where}.schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise Unsupported(
            f"{where}.schema_version is {version!r}; this build implements "
            f"{', '.join(repr(v) for v in SUPPORTED_SCHEMA_VERSIONS)}"
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


def ggml_contiguous_layout(
    dtype: str, ne: tuple[int, ...] | list[int]
) -> tuple[tuple[int, int, int, int], int]:
    """Return exact GGML contiguous strides and bytes for four positive extents.

    F16 and K-quantized layouts are storage declarations for M2 macro operands,
    not a claim that the ordinary tensor operations support these datatypes.
    Quantized rows must contain whole blocks: no implicit row padding is used.
    """
    if not isinstance(dtype, str) or dtype not in _GGML_BLOCK_STORAGE:
        raise Unsupported(f"unsupported contiguous GGML storage dtype {dtype!r}")
    if not isinstance(ne, (tuple, list)) or len(ne) != 4:
        raise InvalidInput("GGML ne must have exactly four positive integer extents")
    for i, extent in enumerate(ne):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
            raise InvalidInput(f"ne[{i}] must be a positive integer, got {extent!r}")
    block_elements, block_bytes = _GGML_BLOCK_STORAGE[dtype]
    if ne[0] % block_elements:
        raise Unsupported(
            f"{dtype} contiguous rows require ne[0] divisible by {block_elements}; "
            f"got {ne[0]} (row padding is not modeled)"
        )
    row_bytes = ne[0] // block_elements * block_bytes
    nb = (block_bytes, row_bytes, row_bytes * ne[1], row_bytes * ne[1] * ne[2])
    return nb, nb[3] * ne[3]


def _require_ggml_nbytes(dtype: str, ne: tuple[int, ...], nb: tuple[int, ...]) -> int:
    expected, nbytes = ggml_contiguous_layout(dtype, ne)
    for i, stride in enumerate(nb):
        if stride != expected[i]:
            raise Unsupported(
                f"non-contiguous {dtype} layout: nb[{i}]={stride} but contiguous "
                f"GGML storage needs nb[{i}]={expected[i]}"
            )
    return nbytes


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


def _parse_tensor(
    raw: Any, where: str, locations_allowed: frozenset[str], *, m2_origin: bool = False
) -> Tensor:
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
    if dtype != DTYPE_F32 and not m2_origin:
        raise Unsupported(
            f"{where}.dtype is {dtype!r}; M0/M1 non-M2 workloads model only "
            f"{DTYPE_F32}; additional GGML storage types require M2 macro origin"
        )
    if dtype not in _GGML_BLOCK_STORAGE:
        raise Unsupported(f"{where}.dtype is {dtype!r}; unsupported M2 GGML storage dtype")

    ne_raw = _require_list(block.get("ne"), f"{where}.ne")
    nb_raw = _require_list(block.get("nb"), f"{where}.nb")
    _expect(len(ne_raw) == 4, f"{where}.ne must have exactly 4 entries, got {len(ne_raw)}")
    _expect(len(nb_raw) == 4, f"{where}.nb must have exactly 4 entries, got {len(nb_raw)}")
    ne = tuple(_require_int(v, f"{where}.ne[{i}]", minimum=1) for i, v in enumerate(ne_raw))
    nb = tuple(_require_int(v, f"{where}.nb[{i}]", minimum=1) for i, v in enumerate(nb_raw))
    assert len(ne) == 4 and len(nb) == 4  # narrowed for the type checker

    computed_bytes = (
        _require_f32_nbytes(ne, nb)
        if dtype == DTYPE_F32
        else _require_ggml_nbytes(dtype, ne, nb)
    )
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
            loc in locations_allowed,
            f"{where}.initial_locations[{i}] must be one of "
            f"{sorted(locations_allowed)}, got {loc!r}",
        )
        _expect(loc not in locations, f"{where}.initial_locations lists {loc!r} twice")
        if loc == LOC_SSD:
            # M1_DESIGN.md §4: an SSD source is legal for a *leaf* -- a weight or
            # an input. A computed value cannot start there: it would mean
            # skipping the compute that produces it, which is the same rule that
            # already forbids any initial copy on a produced tensor
            # (``_validate_workload``).
            _expect(
                role in (ROLE_WEIGHT, ROLE_INPUT),
                f"{where}: only a leaf weight or input may have an SSD source; "
                f"{tensor_id!r} has role {role!r}",
            )
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
        {
            "id",
            "semantic_op",
            "ggml_op",
            "unary_op",
            "op_params",
            "inputs",
            "output",
            "extra_outputs",
        },
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

    extras_raw = _require_list(block.get("extra_outputs", []), f"{where}.extra_outputs")
    extra_outputs: list[str] = []
    for i, value in enumerate(extras_raw):
        value = _require_str(value, f"{where}.extra_outputs[{i}]")
        _expect(bool(value), f"{where}.extra_outputs[{i}] must not be empty")
        extra_outputs.append(value)

    return Operation(
        id=op_id,
        semantic_op=semantic_op,
        ggml_op=ggml_op,
        unary_op=unary_op,
        op_params=op_params,
        inputs=tuple(inputs),
        output=output,
        extra_outputs=tuple(extra_outputs),
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

    version = _check_schema_version(block.get("schema_version"), where)

    workload_id = _require_str(block.get("id"), f"{where}.id")
    origin = _parse_origin(block.get("origin"), where)

    # The initial locations a tensor may declare depend on the document's own
    # version, not on the scenario that references it: a 0.1 workload has no
    # `ssd` even when a 0.2 scenario points at it.
    allowed_locations = _LOCATIONS_M1 if version == SCHEMA_VERSION_M1 else _LOCATIONS
    tensors_raw = _require_list(block.get("tensors"), f"{where}.tensors")
    tensors = tuple(
        _parse_tensor(
            entry, f"{where}.tensors[{i}]", allowed_locations,
            m2_origin=origin.kind in _M2_ORIGIN_KINDS,
        )
        for i, entry in enumerate(tensors_raw)
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

    # --- M2 macro-task gate ------------------------------------------------
    # The macro semantics live in ``_SEMANTIC_OPS``, but a 0.1 or 0.2 document
    # must not be able to reach them. Accepting one there would widen what those
    # versions *mean*, which is a different and worse thing than changing what
    # they hash to. Both the semantics and the second-output field are therefore
    # gated on the origin the workload declares.
    m2_origin = workload.origin.kind in _M2_ORIGIN_KINDS
    if not m2_origin:
        for tensor in workload.tensors:
            if tensor.dtype != DTYPE_F32:
                raise Unsupported(
                    f"tensor {tensor.id!r}: non-M2 workloads require F32; "
                    f"got {tensor.dtype!r}"
                )
    for operation in workload.operations:
        if operation.semantic_op in _SYNTHETIC_ROW_OPS and workload.origin.kind != ORIGIN_SYNTHETIC_LAYER:
            raise Unsupported(f"operation {operation.id!r}: row execution requires synthetic_layer origin")
        if operation.semantic_op in _M2_MACRO_OPS and not m2_origin:
            raise Unsupported(
                f"operation {operation.id!r}: semantic_op "
                f"{operation.semantic_op!r} is an M2 macro-task; it requires a "
                f"workload whose origin.kind is one of {sorted(_M2_ORIGIN_KINDS)}, "
                f"not {workload.origin.kind!r}"
            )
        if operation.extra_outputs and operation.semantic_op not in _M2_MACRO_OPS:
            raise Unsupported(
                f"operation {operation.id!r}: extra_outputs is only meaningful on "
                f"an M2 macro-task, not on {operation.semantic_op!r}"
            )

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

        # Checked before the producer map so a repeat is reported as what it is,
        # rather than as the operation being its own second producer.
        if len(set(operation.outputs)) != len(operation.outputs):
            raise InvalidInput(
                f"{label}: lists the same output tensor twice "
                f"({list(operation.outputs)})"
            )

        for produced in operation.outputs:
            if produced not in tensor_by_id:
                raise InvalidInput(f"{label}: unknown output tensor {produced!r}")
            if produced in operation.inputs:
                raise Unsupported(
                    f"{label}: reads and writes {produced!r} in place; M0 has "
                    "no in-place reuse and no way to model the aliasing"
                )
            if produced in producer_of:
                raise InvalidInput(
                    f"tensor {produced!r} has several producers "
                    f"({producer_of[produced]!r} and {operation.id!r}); M0 needs "
                    "a single producer per tensor so that liveness is a DAG property"
                )
            producer_of[produced] = operation.id

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
        # Every produced tensor, not just the named one: a cycle closed through
        # an extra output would otherwise go undetected, and the engine would
        # then call a non-DAG infeasible instead of rejecting it here.
        for produced in operation.outputs:
            producer_of[produced] = operation.id
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

    if operation.semantic_op not in _M2_MACRO_OPS:
        operands = inputs + [tensor_by_id[tid] for tid in operation.outputs]
        non_f32 = [tensor for tensor in operands if tensor.dtype != DTYPE_F32]
        if non_f32:
            raise Unsupported(
                f"{label}: ordinary {operation.semantic_op} operations require F32; "
                f"tensor {non_f32[0].id!r} has dtype {non_f32[0].dtype!r}. "
                "F16 and quantized storage are supported only as M2 macro operands"
            )

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

    elif operation.semantic_op in _M2_MACRO_OPS:
        # An M2 macro-task's operands are the layer descriptor's business. A
        # projection, an attention over an arbitrary visible history and a
        # router over top-k experts share no shape rule, and inventing one here
        # would be the kernel guessing at a contract it does not own --
        # ``m2.spec.validate_invocation`` is where those shapes are checked.
        return

    else:
        # Reachable only when a semantic op was added to ``_SEMANTIC_OPS`` and
        # no branch was added for it here. Without this the function would fall
        # off the end and *accept* the operation, which is the failure mode this
        # project keeps meeting: a missing rule looks exactly like a passing one.
        raise Unsupported(
            f"{label}: no shape rule for semantic_op {operation.semantic_op!r}; "
            "every entry in _SEMANTIC_OPS needs a branch here"
        )


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


# The keys a scenario document may carry. ``workload_file`` is a property of the
# file-shaped document only -- an in-memory pair names no file (build_scenario).
_SCENARIO_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "workload_file",
        "architecture",
        "costs",
        "mapspace",
        "mapper",
        "goal",
        "comment",
    }
)


def load_and_validate(scenario_path: str | Path) -> Scenario:
    """Load a scenario, resolve its workload, and validate both.

    ``scenario.workload_file`` is resolved **relative to the scenario file**,
    never to the shell's working directory, so a scenario is location-independent
    once checked out (``DESIGN.md`` §4.1).
    """
    path = Path(scenario_path)
    raw = _require_mapping(_read_json(path), path.name)
    _reject_unknown_keys(raw, set(_SCENARIO_KEYS), path.name)

    schema_version = _check_schema_version(raw.get("schema_version"), path.name)
    scenario_id = _require_str(raw.get("id"), f"{path.name}.id")

    workload_file = _require_str(raw.get("workload_file"), f"{path.name}.workload_file")
    workload = load_workload(path.parent / workload_file)

    return _scenario_from(
        workload, raw, path.name, version=schema_version, scenario_id=scenario_id
    )


def build_scenario(
    workload_document: Mapping[str, Any],
    scenario_document: Mapping[str, Any],
    *,
    where: str = "<memory>",
) -> Scenario:
    """Validate an in-memory workload/scenario pair exactly as a file pair would be.

    :func:`load_and_validate` is the file-shaped entry point; this is the same
    validation with no path anywhere. It exists because the M2 layer compiler
    builds its scenario in memory, and it must still go through
    ``_parse_costs_m1`` and ``_parse_mapspace_m1``: those enforce the
    bidirectional allowed-set/cost-table agreement and the reachability check,
    and constructing the dataclasses directly would skip both **silently** --
    the under-counting failure the cost bridge is built to refuse.

    The two documents are what the corresponding files would contain, except
    that the scenario names no ``workload_file``; the workload is passed here
    instead. Everything else, including ``comment``, is as on disk.
    """
    raw = _require_mapping(scenario_document, where)
    if "workload_file" in raw:
        raise InvalidInput(
            f"{where}.workload_file: an in-memory scenario names no file; pass the "
            "workload document to build_scenario() instead"
        )
    _reject_unknown_keys(raw, set(_SCENARIO_KEYS) - {"workload_file"}, where)

    schema_version = _check_schema_version(raw.get("schema_version"), where)
    scenario_id = _require_str(raw.get("id"), f"{where}.id")

    workload = _parse_workload(workload_document, f"{where}:workload")

    return _scenario_from(
        workload, raw, where, version=schema_version, scenario_id=scenario_id
    )


def _scenario_from(
    workload: Workload,
    raw: dict[str, Any],
    where: str,
    *,
    version: str,
    scenario_id: str,
) -> Scenario:
    """The tail shared by both entry points: parse the rest and assemble.

    Kept in one place so the file and in-memory routes cannot drift in which
    parsers run or in what order -- the order is load-bearing for 0.2, where the
    mapspace must exist before the costs are checked against it.
    """
    architecture = _parse_architecture(raw.get("architecture"), where, version)
    mapspace: Mapspace | M1Mapspace
    costs: Costs | M1Costs
    goal: M1Goal | None
    if version == SCHEMA_VERSION_M1:
        # The mapspace is parsed first: the cost tables are checked against the
        # allowed sets, so the sets have to exist before the costs are read.
        mapspace_m1 = _parse_mapspace_m1(raw.get("mapspace"), where, workload)
        mapspace = mapspace_m1
        costs = _parse_costs_m1(raw.get("costs"), where, workload, mapspace_m1)
        goal = _parse_goal(raw.get("goal"), where, workload)
    else:
        if raw.get("goal") is not None:
            raise Unsupported(
                f"{where}.goal is a 0.2 field; a 0.1 scenario's outputs are "
                "always requested in VRAM and there is nothing to choose"
            )
        costs = _parse_costs(raw.get("costs"), where, workload)
        mapspace = _parse_mapspace(raw.get("mapspace"), where, workload)
        goal = None
    mapper_config = _parse_mapper(raw.get("mapper"), where)

    return Scenario(
        schema_version=version,
        id=scenario_id,
        workload=workload,
        architecture=architecture,
        costs=costs,
        mapspace=mapspace,
        mapper=mapper_config,
        goal=goal,
    )


def _parse_architecture(raw: Any, where: str, version: str) -> Architecture:
    if version == SCHEMA_VERSION_M1:
        return _parse_architecture_m1(raw, where)

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

    # The 0.1 rule that DRAM must be null lives here rather than on
    # ``Architecture.__post_init__`` because 0.2 makes the same field the real
    # DRAM budget. Keeping it on the parser is what makes it a statement about
    # *this document version* instead of about the type.
    if block.get("dram_capacity_bytes", None) is not None:
        raise Unsupported(
            "architecture.dram_capacity_bytes must be null: M0 treats DRAM as "
            "unbounded, and a finite DRAM budget needs an eviction decision the "
            "M0 action space does not have"
        )

    # The remaining invariants live on Architecture.__post_init__, so that a
    # directly constructed value cannot slip past them; here we only coerce and
    # hand over.
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
        dram_capacity_bytes=None,
    )


def _parse_architecture_m1(raw: Any, where: str) -> Architecture:
    """0.2's architecture block (``M1_DESIGN.md`` §4).

    Every capacity, reservation and alignment is stated per domain and every
    slot count is present. There is no "default to the other domain's value":
    a missing field here is a document that has not said what machine it means.
    """
    block = _require_mapping(raw, f"{where}.architecture")
    _reject_unknown_keys(
        block,
        {
            "dram_capacity_bytes",
            "vram_capacity_bytes",
            "dram_runtime_reserved_bytes",
            "vram_runtime_reserved_bytes",
            "dram_alignment_bytes",
            "vram_alignment_bytes",
            "cpu_compute_slots",
            "gpu_compute_slots",
            "pcie_copy_slots",
            "ssd_io_slots",
        },
        f"{where}.architecture",
    )
    label = f"{where}.architecture"
    return Architecture(
        dram_capacity_bytes=_require_int(
            block.get("dram_capacity_bytes"), f"{label}.dram_capacity_bytes", minimum=1
        ),
        vram_capacity_bytes=_require_int(
            block.get("vram_capacity_bytes"), f"{label}.vram_capacity_bytes", minimum=1
        ),
        dram_runtime_reserved_bytes=_require_int(
            block.get("dram_runtime_reserved_bytes", 0),
            f"{label}.dram_runtime_reserved_bytes",
        ),
        vram_runtime_reserved_bytes=_require_int(
            block.get("vram_runtime_reserved_bytes", 0),
            f"{label}.vram_runtime_reserved_bytes",
        ),
        dram_alignment_bytes=_require_int(
            block.get("dram_alignment_bytes", 1), f"{label}.dram_alignment_bytes", minimum=1
        ),
        vram_alignment_bytes=_require_int(
            block.get("vram_alignment_bytes", 1), f"{label}.vram_alignment_bytes", minimum=1
        ),
        cpu_compute_slots=_require_int(
            block.get("cpu_compute_slots", 1), f"{label}.cpu_compute_slots", minimum=1
        ),
        gpu_compute_slots=_require_int(
            block.get("gpu_compute_slots", 1), f"{label}.gpu_compute_slots", minimum=1
        ),
        pcie_copy_slots=_require_int(
            block.get("pcie_copy_slots", 1), f"{label}.pcie_copy_slots", minimum=1
        ),
        ssd_io_slots=_require_int(
            block.get("ssd_io_slots", 1), f"{label}.ssd_io_slots", minimum=1
        ),
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


def _parse_goal(raw: Any, where: str, workload: Workload) -> M1Goal:
    """``goal``: one final location per requested output, exactly (0.2).

    The keys must match ``outputs`` exactly in both directions. A missing key
    would have to default to something, and the default would decide the answer
    to the question the field exists to ask; an extra key names a tensor nobody
    requested.
    """
    block = _require_mapping(raw, f"{where}.goal")
    _reject_unknown_keys(block, {"output_locations"}, f"{where}.goal")
    locations_raw = _require_mapping(
        block.get("output_locations"), f"{where}.goal.output_locations"
    )
    label = f"{where}.goal.output_locations"
    locations: dict[str, str] = {}
    for tensor_id, location in locations_raw.items():
        if tensor_id not in workload.outputs:
            raise InvalidInput(
                f"{label}[{tensor_id!r}]: {tensor_id!r} is not one of the workload's "
                f"outputs ({', '.join(workload.outputs)})"
            )
        location = _require_str(location, f"{label}[{tensor_id!r}]")
        if location not in GOAL_LOCATIONS_M1:
            raise InvalidInput(
                f"{label}[{tensor_id!r}] must be one of {list(GOAL_LOCATIONS_M1)}, "
                f"got {location!r}"
            )
        locations[tensor_id] = location
    missing = sorted(set(workload.outputs) - set(locations))
    if missing:
        raise InvalidInput(
            f"{label} is missing {', '.join(missing)}; every requested output needs a "
            "final location, and a default would silently change what is optimised"
        )
    return M1Goal(output_locations=locations)


def _parse_costs_m1(raw: Any, where: str, workload: Workload, mapspace: M1Mapspace) -> M1Costs:
    """0.2 costs: per (operation, device) and per (route, tensor).

    M1_DESIGN.md §4: the allowed sets and the cost tables must agree. Requiring
    agreement in both directions is what makes "the cost of a route this scenario
    allows" impossible to be missing, and a cost for something the scenario does
    not allow impossible to look meaningful.
    """
    block = _require_mapping(raw, f"{where}.costs")
    _reject_unknown_keys(block, {"source", "compute", "transfer"}, f"{where}.costs")
    source = _require_str(block.get("source"), f"{where}.costs.source")

    compute_raw = _require_mapping(block.get("compute"), f"{where}.costs.compute")
    compute: dict[str, dict[str, ComputeCost]] = {}
    for op_id, entry in compute_raw.items():
        label = f"{where}.costs.compute[{op_id!r}]"
        if op_id not in workload.operation_by_id:
            raise InvalidInput(f"{label}: no such operation in the workload")
        allowed = mapspace.devices_for(op_id)
        if not allowed:
            raise InvalidInput(
                f"{label}: operation {op_id!r} has no allowed device, so costing it "
                "is meaningless"
            )
        body = _require_mapping(entry, label)
        unknown = sorted(set(body) - set(allowed))
        if unknown:
            raise InvalidInput(
                f"{label}: {', '.join(unknown)} is not an allowed device for "
                f"{op_id!r} (allowed: {', '.join(allowed)})"
            )
        missing = sorted(set(allowed) - set(body))
        if missing:
            raise InvalidInput(
                f"{label} is missing a duration for {', '.join(missing)}; M1 never "
                "guesses a compute cost"
            )
        _reject_unknown_keys(body, set(_DEVICES), label)
        compute[op_id] = {
            device: _parse_device_cost(body, device, label) for device in allowed
        }

    missing_ops = sorted(
        operation.id
        for operation in workload.operations
        if mapspace.devices_for(operation.id) and operation.id not in compute
    )
    if missing_ops:
        raise InvalidInput(
            f"{where}.costs.compute is missing a duration for "
            f"{', '.join(missing_ops)}; M1 never guesses a compute cost"
        )

    transfer_raw = _require_mapping(block.get("transfer"), f"{where}.costs.transfer")
    transfer: dict[str, dict[str, TransferCost]] = {}
    for route, entry in transfer_raw.items():
        label = f"{where}.costs.transfer[{route!r}]"
        if route not in ROUTES:
            raise Unsupported(
                f"{label}: {route!r} is not a route M1 models; the only ones are "
                f"{list(ROUTES)} (SSD traffic goes through DRAM: there is no direct "
                "SSD-to-VRAM path and no SSD-to-SSD copy)"
            )
        allowed_tensors = mapspace.tensors_for(route)
        body = _require_mapping(entry, label)
        if not allowed_tensors:
            # An empty table is the shape the design gives an unused route, and
            # it costs nothing because nothing may travel it. Anything in it
            # would be a cost for a move the mapspace forbids.
            if body:
                raise InvalidInput(
                    f"{label}: the mapspace allows no tensor on route {route!r}, so "
                    f"costing {', '.join(sorted(body))} is meaningless"
                )
            transfer[route] = {}
            continue
        missing = sorted(set(allowed_tensors) - set(body))
        if missing:
            raise InvalidInput(
                f"{label} is missing a duration for {', '.join(missing)}; M1 never "
                "guesses a transfer cost"
            )
        unknown = sorted(set(body) - set(allowed_tensors))
        if unknown:
            raise InvalidInput(
                f"{label}: {', '.join(unknown)} is not allowed on route {route!r} "
                f"(allowed: {', '.join(allowed_tensors)})"
            )
        transfer[route] = {
            tensor_id: _parse_transfer_cost(body[tensor_id], f"{label}[{tensor_id!r}]")
            for tensor_id in allowed_tensors
        }

    missing_routes = sorted(
        route for route in ROUTES if mapspace.tensors_for(route) and route not in transfer
    )
    if missing_routes:
        raise InvalidInput(
            f"{where}.costs.transfer is missing a table for "
            f"{', '.join(missing_routes)}; M1 never guesses a transfer cost"
        )

    return M1Costs(source=source, compute=compute, transfer=transfer)


def _parse_transfer_cost(raw: Any, label: str) -> TransferCost:
    """One route+tensor duration.

    The object form, exactly as a compute entry and exactly as the 0.1 ``h2d``
    table this replaces: ``{"duration_ns": N}``. A bare integer is refused rather
    than accepted as a shorthand, so the field has one spelling on disk and a
    reader never has to work out which of the two they are looking at.
    """
    _expect(not isinstance(raw, list), f"{label} must be a JSON object")
    item = _require_mapping(raw, label)
    _reject_unknown_keys(item, {"duration_ns"}, label)
    return TransferCost(
        duration_ns=_parse_duration_ns(item.get("duration_ns"), f"{label}.duration_ns")
    )


def _parse_device_cost(body: dict[str, Any], device: str, label: str) -> ComputeCost:
    entry_label = f"{label}[{device!r}]"
    _expect(not isinstance(body[device], list), f"{entry_label} must be a JSON object")
    item = _require_mapping(body[device], entry_label)
    _reject_unknown_keys(item, {"duration_ns", "workspace_bytes"}, entry_label)
    return ComputeCost(
        duration_ns=_parse_duration_ns(item.get("duration_ns"), f"{entry_label}.duration_ns"),
        workspace_bytes=_require_int(
            item.get("workspace_bytes", 0), f"{entry_label}.workspace_bytes"
        ),
    )


def _parse_duration_ns(raw: Any, what: str) -> int:
    """A duration must be a positive, finite integer number of nanoseconds.

    ``_require_int`` already refuses bools, floats, strings and anything below
    the minimum; this wrapper exists so the message names the field the way the
    contract does and so a float like ``2.5`` is refused rather than truncated
    into a plan that is 500 ns short.
    """
    if isinstance(raw, float):
        raise InvalidInput(f"{what} must be an integer number of ns, got {raw!r}")
    return _require_int(raw, what, minimum=1)


def _parse_mapspace_m1(raw: Any, where: str, workload: Workload) -> M1Mapspace:
    block = _require_mapping(raw, f"{where}.mapspace")
    _reject_unknown_keys(
        block,
        {
            "allowed_devices",
            "allowed_transfers",
            "transfer_dependencies",
            "allow_copy_compute_overlap",
            "allow_cpu_gpu_overlap",
            "allow_eviction",
            "allow_recomputation",
        },
        f"{where}.mapspace",
    )

    devices_raw = _require_mapping(
        block.get("allowed_devices"), f"{where}.mapspace.allowed_devices"
    )
    label = f"{where}.mapspace.allowed_devices"
    allowed_devices: dict[str, tuple[str, ...]] = {}
    for op_id, entry in devices_raw.items():
        item_label = f"{label}[{op_id!r}]"
        if op_id not in workload.operation_by_id:
            raise InvalidInput(f"{item_label}: no such operation in the workload")
        entries = _require_list(entry, item_label)
        if not entries:
            raise InvalidInput(
                f"{item_label} is empty; every operation must be runnable somewhere"
            )
        seen: list[str] = []
        for i, device in enumerate(entries):
            device = _require_str(device, f"{item_label}[{i}]")
            if device not in _DEVICES:
                raise Unsupported(
                    f"{item_label}[{i}] is {device!r}; M1 models the devices "
                    f"{list(_DEVICES)}"
                )
            if device in seen:
                raise InvalidInput(f"{item_label} lists {device!r} twice")
            seen.append(device)
        # A fixed order, not the document's: the search's candidate order is part
        # of the deterministic contract, and it must not depend on how the JSON
        # happened to be written.
        allowed_devices[op_id] = tuple(sorted(seen))

    missing = sorted(set(workload.operation_by_id) - set(allowed_devices))
    if missing:
        raise InvalidInput(
            f"{label} is missing {', '.join(missing)}; every operation needs a "
            "non-empty device set"
        )

    transfers_raw = _require_mapping(
        block.get("allowed_transfers"), f"{where}.mapspace.allowed_transfers"
    )
    tlabel = f"{where}.mapspace.allowed_transfers"
    allowed_transfers: dict[str, tuple[str, ...]] = {}
    for route, entry in transfers_raw.items():
        item_label = f"{tlabel}[{route!r}]"
        if route not in ROUTES:
            raise Unsupported(
                f"{item_label}: {route!r} is not a route M1 models; the only ones are "
                f"{list(ROUTES)} (SSD traffic goes through DRAM: there is no direct "
                "SSD-to-VRAM path, and no SSD-to-SSD copy)"
            )
        entries = _require_list(entry, item_label)
        seen_tensors: list[str] = []
        for i, tensor_id in enumerate(entries):
            tensor_id = _require_str(tensor_id, f"{item_label}[{i}]")
            if tensor_id not in workload.tensor_by_id:
                raise InvalidInput(f"{item_label}[{i}]: unknown tensor {tensor_id!r}")
            if tensor_id in seen_tensors:
                raise InvalidInput(f"{item_label} lists {tensor_id!r} twice")
            seen_tensors.append(tensor_id)
        allowed_transfers[route] = tuple(sorted(seen_tensors))

    # A route the scenario lists must be reachable for the tensors on it. This
    # is a check on the *document*, not on the search: without it a scenario can
    # claim a tensor may take a route whose source end the tensor can never be
    # at, and the only symptom is a search that quietly never uses the option.
    #
    # A produced tensor is exempt from the source-end requirement on purpose:
    # it is materialised wherever its compute ran, and M1_DESIGN.md §4 says a
    # computed intermediate may be listed.
    for tensor in workload.tensors:
        if tensor.id in workload.producer_of:
            continue
        reachable = reachable_locations(
            tensor.initial_locations, allowed_transfers, tensor.id
        )
        for route, tensors in allowed_transfers.items():
            if tensor.id not in tensors:
                continue
            source, _ = ROUTE_ENDS[route]
            if source not in reachable:
                raise InvalidInput(
                    f"{tlabel}[{route!r}] includes {tensor.id!r}, but no allowed route "
                    f"can get it to {source} (initial_locations: "
                    f"{list(tensor.initial_locations)})"
                )

    recomputation = _require_bool(
        block.get("allow_recomputation", False),
        f"{where}.mapspace.allow_recomputation",
    )
    if recomputation:
        raise Unsupported(
            f"{where}.mapspace.allow_recomputation is true; M1 has no recomputation "
            "and must not silently ignore the request"
        )

    dependencies_raw = _require_mapping(
        block.get("transfer_dependencies", {}),
        f"{where}.mapspace.transfer_dependencies",
    )
    transfer_dependencies: dict[str, tuple[str, ...]] = {}
    for tensor_id, entry in dependencies_raw.items():
        item_label = f"{where}.mapspace.transfer_dependencies[{tensor_id!r}]"
        operation_ids = _require_list(entry, item_label)
        transfer_dependencies[tensor_id] = tuple(
            _require_str(operation_id, f"{item_label}[{i}]")
            for i, operation_id in enumerate(operation_ids)
        )

    return M1Mapspace(
        allowed_devices=allowed_devices,
        allowed_transfers=allowed_transfers,
        transfer_dependencies=transfer_dependencies,
        allow_eviction=_require_bool(
            block.get("allow_eviction"), f"{where}.mapspace.allow_eviction"
        ),
        allow_copy_compute_overlap=_require_bool(
            block.get("allow_copy_compute_overlap"),
            f"{where}.mapspace.allow_copy_compute_overlap",
        ),
        allow_cpu_gpu_overlap=_require_bool(
            block.get("allow_cpu_gpu_overlap", True),
            f"{where}.mapspace.allow_cpu_gpu_overlap",
        ),
        allow_recomputation=False,
    )


def reachable_locations(
    starts: tuple[str, ...],
    allowed_transfers: Mapping[str, tuple[str, ...]],
    tensor_id: str,
) -> frozenset[str]:
    """Every location ``tensor_id`` can be *brought to* using allowed routes.

    A plain closure over the route graph restricted to this one tensor: start at
    the places it already is and repeatedly apply any allowed route whose source
    end has been reached. It says nothing about capacity, timing or device
    choice -- this is a property of the document, and the search is what decides
    whether a route is actually usable.
    """
    reached = set(starts)
    changed = True
    while changed:
        changed = False
        for route, tensors in allowed_transfers.items():
            if tensor_id not in tensors:
                continue
            source, destination = ROUTE_ENDS[route]
            if source in reached and destination not in reached:
                reached.add(destination)
                changed = True
    return frozenset(reached)


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

    Both domains are checked in 0.2. A tensor that starts in DRAM *and* VRAM is
    charged to both, because that is two allocations and the model says so
    (``M1_ACCEPTANCE.md`` §8). SSD is not a domain: it has no capacity.
    """
    architecture = scenario.architecture
    for domain in DOMAINS:
        capacity = architecture.capacity_of(domain)
        if capacity is None:
            continue  # 0.1's DRAM: unbounded, and nothing is charged to it
        used = architecture.reserved_of(domain)
        resident: list[str] = []
        for tensor in scenario.workload.tensors:
            if domain in tensor.initial_locations:
                used += scenario.size_alloc(tensor.id, domain)
                resident.append(tensor.id)
        if used > capacity:
            # The domain is named by the trailing field: for 0.1 that reproduces
            # the original sentence word for word, and for 0.2 it says which of
            # the two budgets was exceeded (``M1_ACCEPTANCE.md`` §8).
            raise InitialCapacityExceeded(
                f"initial layout needs {used} bytes (runtime reserve "
                f"{architecture.reserved_of(domain)} + tensors "
                f"{', '.join(sorted(resident))}) but {domain}_capacity_bytes is "
                f"{capacity}"
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
    operations: list[dict[str, Any]] = []
    for op in sorted(workload.operations, key=lambda op: op.id):
        entry: dict[str, Any] = {
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
        # Present only when non-empty. This digest identifies *stored mappings*,
        # so an unconditional key would re-hash every 0.1 and 0.2 workload --
        # invalidating artefacts on disk -- over a field none of them uses.
        if op.extra_outputs:
            entry["extra_outputs"] = list(op.extra_outputs)
        operations.append(entry)

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
        "operations": operations,
        "outputs": list(workload.outputs),
    }


def _canonical_scenario(scenario: Scenario) -> dict[str, Any]:
    """Semantic content of a scenario.

    Two algorithms, chosen by ``schema_version``, and the choice is deliberate
    rather than incidental: these digests identify *stored mappings*, and a 0.1
    mapping written before M1 must keep hashing to the same value afterwards
    (``M1_DESIGN.md`` §8). Extending the 0.1 payload with new keys would have
    silently invalidated every historical mapping, so 0.2 gets its own shape.

    ``workload_file`` is excluded in both -- it is an external path, exactly what
    the spec says to drop -- and the workload is represented by its fingerprint
    instead, so a scenario fingerprint still pins the exact graph without
    depending on where the file happens to live.
    """
    if scenario.is_m1:
        return _canonical_scenario_m1(scenario)
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


def _canonical_scenario_m1(scenario: Scenario) -> dict[str, Any]:
    """Semantic content of a 0.2 scenario.

    Everything that changes what the search may do is in here: the device sets,
    the routes, both domains' capacities, reservations and alignments, the cost
    tables, the goal locations and the two switches. ``M1_DESIGN.md`` §8 asks
    for exactly this list, because a mapping made for one machine must not be
    replayable against another.
    """
    architecture = scenario.architecture
    mapspace = scenario.mapspace
    costs = scenario.costs
    mapper = scenario.mapper
    assert isinstance(mapspace, M1Mapspace)
    assert isinstance(costs, M1Costs)
    assert scenario.goal is not None
    return {
        "schema_version": scenario.schema_version,
        "id": scenario.id,
        "workload_fingerprint": workload_fingerprint(scenario.workload),
        "architecture": {
            "dram_capacity_bytes": architecture.dram_capacity_bytes,
            "vram_capacity_bytes": architecture.vram_capacity_bytes,
            "dram_runtime_reserved_bytes": architecture.dram_runtime_reserved_bytes,
            "vram_runtime_reserved_bytes": architecture.reserved_of(LOC_VRAM),
            "dram_alignment_bytes": architecture.alignment_of(LOC_DRAM),
            "vram_alignment_bytes": architecture.alignment_of(LOC_VRAM),
            "cpu_compute_slots": architecture.cpu_compute_slots,
            "gpu_compute_slots": architecture.gpu_compute_slots,
            "pcie_copy_slots": architecture.pcie_copy_slots,
            "ssd_io_slots": architecture.ssd_io_slots,
        },
        "goal": {
            "output_locations": [
                {"tensor_id": t, "location": scenario.goal.output_locations[t]}
                for t in sorted(scenario.goal.output_locations)
            ]
        },
        "costs": {
            "source": costs.source,
            "compute": [
                {
                    "operation_id": op_id,
                    "device": device,
                    "duration_ns": cost.duration_ns,
                    "workspace_bytes": cost.workspace_bytes,
                }
                for op_id in sorted(costs.compute)
                for device, cost in sorted(costs.compute[op_id].items())
            ],
            "transfer": [
                {
                    "route": route,
                    "tensor_id": tensor_id,
                    "duration_ns": cost.duration_ns,
                }
                for route in sorted(costs.transfer)
                for tensor_id, cost in sorted(costs.transfer[route].items())
            ],
        },
        "mapspace": {
            "allowed_devices": [
                {"operation_id": op_id, "devices": sorted(devices)}
                for op_id, devices in sorted(mapspace.allowed_devices.items())
            ],
            "allowed_transfers": [
                {"route": route, "tensor_ids": sorted(tensor_ids)}
                for route, tensor_ids in sorted(mapspace.allowed_transfers.items())
            ],
            "allow_copy_compute_overlap": mapspace.allow_copy_compute_overlap,
            "allow_cpu_gpu_overlap": mapspace.allow_cpu_gpu_overlap,
            **(
                {"transfer_dependencies": [
                    {"tensor_id": tensor_id, "operation_ids": sorted(operation_ids)}
                    for tensor_id, operation_ids
                    in sorted(mapspace.transfer_dependencies.items())
                ]}
                if mapspace.transfer_dependencies else {}
            ),
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
