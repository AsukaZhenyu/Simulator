"""M0 engine: the state, the four actions, and fixed-mapping evaluation.

This module holds the *only* implementation of the transition rules. The mapper
imports :func:`transition` rather than repeating any capacity or dependency
check, so a mapping the search produces can never be one the evaluator would
reject -- and a bug in the rules shows up once, not twice
(``mapping/README.md`` §「engine 是唯一的语义实现」).

State design
------------
``State`` carries the minimum that determines the future:

* the status of every operation and every GPU copy,
* the id and **remaining** time of each in-flight task.

``used_vram``, the read locks and the allocation set are all *derived* from
those fields by :func:`used_vram_bytes`, never stored. Two independent ledgers
would be two things to keep in sync, and the contract asks for one
(``DESIGN.md`` §5).

The absolute simulation time ``t`` is carried on the state but is deliberately
**excluded from :meth:`State.key`**. With fixed costs and no external events,
``t`` is exactly the cost of the path taken to reach the state, so two states
differing only in ``t`` have identical futures and the cheaper one dominates.
That is what makes uniform-cost search correct here (``DESIGN.md`` §7).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Sequence

from .spec import (
    COPY_ABSENT,
    COPY_READY,
    COPY_RESERVED_COPY,
    COPY_RESERVED_OUTPUT,
    DEVICE_GPU,
    DOMAINS,
    DOMAIN_OF_DEVICE,
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    M1Costs,
    M1Mapspace,
    OP_DONE,
    OP_NOT_STARTED,
    OP_RUNNING,
    RESOURCE_COMPUTE,
    RESOURCE_COPY,
    RESOURCE_CPU_COMPUTE,
    RESOURCE_GPU_COMPUTE,
    RESOURCE_OF_DEVICE,
    RESOURCE_OF_ROUTE,
    RESOURCE_PCIE_COPY,
    RESOURCE_SSD_IO,
    ROUTE_ENDS,
    ROUTES,
    ROUTE_DRAM_TO_SSD,
    ROUTE_SSD_TO_DRAM,
    SCHEMA_VERSION,
    SCHEMA_VERSION_M1,
    InitialCapacityExceeded,
    InvalidInput,
    Scenario,
    Unsupported,
    check_initial_capacity,
    reachable_locations,
    scenario_fingerprint,
    workload_fingerprint,
)

# The JSON validation vocabulary lives in spec.py. It is imported privately
# because it is not part of the documented API, but engine.py is inside the same
# package boundary -- re-deriving a second set of checks here is exactly how the
# two documents' validation rules would drift apart.
from .spec import (  # noqa: E402  (grouped with the import above for clarity)
    _check_schema_version,
    _reject_unknown_keys,
    _require_list,
    _require_mapping,
    _require_str,
)

__all__ = [
    "ACTION_ADVANCE",
    "ACTION_COMPUTE",
    "ACTION_COPY",
    "ACTION_COPY_H2D",
    "ACTION_EVICT",
    "ACTION_READ_SSD",
    "ACTION_WRITE_SSD",
    "Action",
    "MappingError",
    "MappingIdentityError",
    "MappingDocument",
    "RunningTask",
    "State",
    "StateSnapshot",
    "Event",
    "EvaluationResult",
    "initial_state",
    "is_goal",
    "legal_actions",
    "load_mapping",
    "transition",
    "used_bytes",
    "used_vram_bytes",
    "evaluate_mapping",
    # `MappingError.code`、`EvaluationResult.status` / `.reason` 的取值域。
    # 这三个字段是公开的，比较它们却要拿到这些常量，所以它们同属公开面。
    # 此前不在 __all__ 里（只是按名可导入），星号转出的兼容层会漏掉，故显式列出。
    "CODE_INPUT_NOT_READY",
    "CODE_RESOURCE_BUSY",
    "CODE_CAPACITY_EXCEEDED",
    "CODE_TENSOR_IN_USE",
    "CODE_LIVE_VALUE_LOSS",
    "CODE_ILLEGAL_ACTION",
    "REEVALUATION_CODES",
    "STATUS_VALID",
    "STATUS_INVALID_MAPPING",
    "STATUS_INCOMPLETE_MAPPING",
    "STATUS_INFEASIBLE",
    "REASON_INITIAL_CAPACITY_EXCEEDED",
    "REASON_GOAL_REACHED",
    "REASON_ACTIONS_REMAIN_AFTER_GOAL",
    "REASON_UNFINISHED",
    "WallClock",
]

ACTION_COPY_H2D = "COPY_H2D"
ACTION_COMPUTE = "COMPUTE"
ACTION_EVICT = "EVICT"
ACTION_ADVANCE = "ADVANCE"

# 0.2 spells a transfer as COPY with an explicit direction, and an SSD read or
# write as its own kind. Keeping the two SSD kinds separate rather than as COPies
# is what lets a scenario say "this tensor may be saved to SSD but never copied
# between domains" and have the action space contain no way to express the other
# one.
#
# COPY_H2D stays exactly what it was. It is one action, always dram -> vram,
# always on the one copy resource -- which is what every stored M0 mapping and
# every M0 event says. 0.2 could have been written as `COPY(x, dram, vram)`
# spelled with the old name; it is not, because the old name also carries the
# old *resource* name and the old single-slot assumption with it.
ACTION_COPY = "COPY"
ACTION_READ_SSD = "READ_SSD"
ACTION_WRITE_SSD = "WRITE_SSD"

# The five rejection codes of DESIGN.md §8, plus one for actions that are not
# even representable in the mapspace. Keeping the extra case separate means the
# five model-level codes stay meaningful rather than becoming a catch-all.
CODE_INPUT_NOT_READY = "INPUT_NOT_READY"
CODE_RESOURCE_BUSY = "RESOURCE_BUSY"
CODE_CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
CODE_TENSOR_IN_USE = "TENSOR_IN_USE"
CODE_LIVE_VALUE_LOSS = "LIVE_VALUE_LOSS"
CODE_ILLEGAL_ACTION = "ILLEGAL_ACTION"

REEVALUATION_CODES = frozenset(
    {
        CODE_INPUT_NOT_READY,
        CODE_RESOURCE_BUSY,
        CODE_CAPACITY_EXCEEDED,
        CODE_TENSOR_IN_USE,
        CODE_LIVE_VALUE_LOSS,
    }
)


class MappingError(Exception):
    """A mapping action that the model rejects.

    Carries enough context to point at the exact step: the action index, the
    simulated time, and the resource or tensor involved (``DESIGN.md`` §8).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        action_index: int | None = None,
        t_ns: int | None = None,
        tensor_id: str | None = None,
        operation_id: str | None = None,
        resource: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.action_index = action_index
        self.t_ns = t_ns
        self.tensor_id = tensor_id
        self.operation_id = operation_id
        self.resource = resource


# ---------------------------------------------------------------------------
# Actions and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One step of a mapping.

    ``target_id`` is a tensor id for COPY_H2D/COPY/READ_SSD/WRITE_SSD/EVICT, an
    operation id for COMPUTE, and ``None`` for ADVANCE.

    The four extra fields are 0.2 only and are ``None`` for a 0.1 action, where
    each of them has exactly one possible value. They are not defaulted to that
    value, because a default is how a 0.2 action that forgot to say which device
    it meant would silently become a GPU action.
    """

    kind: str
    target_id: str | None = None
    device: str | None = None
    src: str | None = None
    dst: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        if self.target_id is None:
            return self.kind
        if self.kind == ACTION_COMPUTE and self.device is not None:
            return f"{self.kind}({self.target_id}, {self.device})"
        if self.kind == ACTION_COPY:
            return f"{self.kind}({self.target_id}, {self.src}->{self.dst})"
        if self.kind == ACTION_EVICT and self.location is not None:
            return f"{self.kind}({self.target_id}, {self.location})"
        return f"{self.kind}({self.target_id})"

    @property
    def route(self) -> str | None:
        """The route this action moves along, or ``None`` if it moves nothing.

        READING a transfer action's direction off one name is what stops the two
        ends from being compared separately and disagreeing. COPY is defined as
        the DRAM<->VRAM pair only; the SSD directions have their own kinds, so
        ``COPY(x, dram, ssd)`` has no route and is refused as an action.
        """
        if self.kind == ACTION_READ_SSD:
            return ROUTE_SSD_TO_DRAM
        if self.kind == ACTION_WRITE_SSD:
            return ROUTE_DRAM_TO_SSD
        if self.kind != ACTION_COPY:
            return None
        for route, (source, destination) in ROUTE_ENDS.items():
            if route in (ROUTE_SSD_TO_DRAM, ROUTE_DRAM_TO_SSD):
                continue
            if source == self.src and destination == self.dst:
                return route
        return None


@dataclass(frozen=True, order=True)
class RunningTask:
    """An in-flight copy or compute, identified by what it acts on.

    Only ``remaining_ns`` varies over the task's life; the total duration is
    looked up in the cost model when it is needed, so it is not duplicated here.

    ``device`` and ``route`` are the 0.2 fields: a compute needs to know which
    device it runs on so its output lands in the right domain, and a transfer
    needs its route so its *destination* is unambiguous. Both are ``None`` for a
    0.1 task, whose single device and single route are fixed by the version.
    """

    resource: str
    target_id: str
    remaining_ns: int
    device: str | None = None
    route: str | None = None


@dataclass(frozen=True)
class State:
    """An immutable model state.

    ``op_status`` and ``copy_status`` are sorted ``(id, status)`` tuples so that
    two equal states are equal by construction, regardless of the order in which
    they were built.

    ``dram_status`` is the 0.2 addition. It is a *separate* tuple rather than a
    ``(tensor, location, status)`` tuple replacing ``copy_status`` because every
    stored M0 state artefact, every M0 test and the whole viewer read
    ``copy_status`` as "the VRAM copy", and 0.1 has no DRAM ledger at all -- its
    DRAM is unbounded and static, so nothing is ever charged to it and nothing
    is ever released from it. A 0.1 state therefore leaves ``dram_status`` empty,
    and its :meth:`key` is the tuple it always was.

    ``ssd_status`` is the third ledger. Unlike the other two it is *not* a
    capacity ledger -- the SSD has no budget (``M1_DESIGN.md`` §3) -- it records
    only which SSD copies exist and whether a write has landed: ``READY`` for a
    copy the scene started with or a finished write, ``RESERVED_COPY`` for a
    write still in flight, ``ABSENT`` for everything else. A 0.1 scenario has no
    SSD location, so the tuple is empty and its key is unchanged; the 0.2 parser
    forbids an SSD initial location in a 0.1 workload, so no 0.1 state can hide
    one here.
    """

    t: int
    op_status: tuple[tuple[str, str], ...]
    copy_status: tuple[tuple[str, str], ...]
    running: tuple[RunningTask, ...]
    dram_status: tuple[tuple[str, str], ...] = ()
    ssd_status: tuple[tuple[str, str], ...] = ()

    @cached_property
    def _ops(self) -> dict[str, str]:
        return dict(self.op_status)

    @cached_property
    def _copies(self) -> dict[str, str]:
        return dict(self.copy_status)

    @cached_property
    def _dram(self) -> dict[str, str]:
        return dict(self.dram_status)

    @cached_property
    def _ssd(self) -> dict[str, str]:
        return dict(self.ssd_status)

    def op(self, operation_id: str) -> str:
        """Status of ``operation_id``, or ``NOT_STARTED`` if unknown to the graph."""
        return self._ops.get(operation_id, OP_NOT_STARTED)

    def copy(self, tensor_id: str) -> str:
        """GPU-copy status of ``tensor_id``, or ``ABSENT`` if unknown."""
        return self._copies.get(tensor_id, COPY_ABSENT)

    def copy_at(self, tensor_id: str, location: str) -> str:
        """Copy status of ``tensor_id`` in ``location``.

        The one place that knows the three locations are stored in different
        fields. ``ssd`` has a status here because a 0.2 scenario may write to it
        at runtime (``M1_DESIGN.md`` §5) -- what it never has is a *budget*.
        """
        if location == LOC_DRAM:
            return self._dram.get(tensor_id, COPY_ABSENT)
        if location == LOC_SSD:
            return self._ssd.get(tensor_id, COPY_ABSENT)
        return self.copy(tensor_id)

    def statuses(self, location: str) -> tuple[tuple[str, str], ...]:
        """The whole ``location`` ledger, in the same sorted form as every field."""
        if location == LOC_DRAM:
            return self.dram_status
        if location == LOC_SSD:
            return self.ssd_status
        return self.copy_status

    def task_on(self, resource: str) -> RunningTask | None:
        """The task currently occupying ``resource``, if any."""
        for task in self.running:
            if task.resource == resource:
                return task
        return None

    def key(self) -> tuple[Any, ...]:
        """Canonical identity used for state de-duplication.

        Excludes ``t`` on purpose: accumulated time is the search cost, not part
        of what the state *is*. Includes each task's remaining time, because a
        copy with 2 ms left is genuinely a different state from one with 1 ms
        left even when every status agrees.

        ``M1_DESIGN.md`` §7 requires the DRAM residency to be part of this, and
        it is: a state with W cached in DRAM is genuinely different from one
        that has to re-read it from SSD. The same holds for the *runtime* SSD
        copies -- "no copy", "a write is in flight" and "the snapshot is on disk"
        are three different futures, and two branches that differ only in one of
        them must not be merged (``M1_ACCEPTANCE.md`` §11). Copies the scenario
        started with are a constant across every state of that scenario, so
        carrying them here costs a suffix and hides nothing.

        Every ledger is sorted by id and the task tuple is sorted, so the key
        does not depend on insertion order.
        """
        return (
            self.op_status,
            self.copy_status,
            self.running,
            self.dram_status,
            self.ssd_status,
        )


@dataclass(frozen=True)
class StateSnapshot:
    """A checkable summary of a state, for artefacts and assertions."""

    t: int
    used_vram_bytes: int
    op_status: tuple[tuple[str, str], ...]
    copy_status: tuple[tuple[str, str], ...]
    used_dram_bytes: int = 0
    dram_status: tuple[tuple[str, str], ...] = ()
    # No ``used_ssd_bytes`` beside the two above: the SSD has no capacity, so
    # there is no usage number to report. What a reader needs is the same thing
    # the search needs -- which copies exist and whether a write has landed.
    ssd_status: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Event:
    """One applied action, as recorded in the event log.

    Transfer/release amounts are kept in separate fields on purpose. A copy
    moves ``storage_bytes``; an eviction releases the *aligned* allocation. Those
    differ whenever ``allocation_alignment_bytes`` is greater than one, and
    conflating them is the mistake ``ACCEPTANCE.md`` §5 calls out.

    ``device``/``src``/``dst``/``location`` are the 0.2 additions and are ``None``
    for a 0.1 event. They are recorded rather than derived because the event log
    is what a reader -- human or GUI -- uses to see *where* work happened, and
    deriving it from the scenario would make a mis-tagged event look correct.
    """

    action_index: int
    kind: str
    target_id: str | None
    resource: str | None
    t_ns: int
    start_ns: int | None = None
    end_ns: int | None = None
    bytes_transferred: int = 0
    workspace_bytes: int = 0
    bytes_released: int = 0
    device: str | None = None
    src: str | None = None
    dst: str | None = None
    location: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of replaying a fixed mapping."""

    status: str
    reason: str
    makespan_ns: int | None
    peak_vram_bytes: int
    h2d_bytes: int
    action_count: int
    actions: tuple[Action, ...]
    events: tuple[Event, ...]
    states: tuple[StateSnapshot, ...]
    error_code: str | None = None
    error_action_index: int | None = None
    error_t_ns: int | None = None
    # 0.2 additions. ``h2d_bytes`` keeps its exact 0.1 meaning (bytes moved on
    # dram_to_vram), so a 0.1 artefact reads the same as it always did; the other
    # directions are counted separately rather than folded into it, because
    # "total transferred" is a sum the reader can do and a single field that
    # silently changed meaning is not (M1_DESIGN.md §8).
    d2h_bytes: int = 0
    ssd_read_bytes: int = 0
    ssd_write_bytes: int = 0
    peak_dram_bytes: int = 0


# ---------------------------------------------------------------------------
# Derived quantities -- the single memory ledger
# ---------------------------------------------------------------------------


def _task_device(scenario: Scenario, task: RunningTask) -> str:
    """Which compute device a task occupies.

    ``None`` means "a 0.1 task", and 0.1 has exactly one device. Spelling the
    fallback here rather than defaulting the field keeps the three call sites
    from each inventing their own answer.
    """
    return task.device or DEVICE_GPU


def task_domain(scenario: Scenario, task: RunningTask) -> str:
    """The storage domain a running task occupies and, for a compute, writes to."""
    if task.resource in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE):
        return DOMAIN_OF_DEVICE[_task_device(scenario, task)]
    assert task.route is not None or not scenario.is_m1
    if task.route is None:
        return LOC_VRAM  # 0.1's COPY_H2D
    return ROUTE_ENDS[task.route][1]


def used_bytes(scenario: Scenario, state: State, domain: str) -> int:
    """Bytes in use in ``domain``.

    Allocations are (a) the domain's static reservation, (b) every tensor whose
    copy there is not ``ABSENT`` -- a reservation counts, because the space is
    committed the moment the copy or the compute starts -- and (c) the workspace
    of every running task that occupies the domain.

    ``None`` capacity means unbounded, and this returns a number anyway: a 0.1
    DRAM ledger is never consulted, and inventing ``None`` arithmetic for it
    would push the branch into every caller.

    Only the two storage domains have a ledger. The SSD is not one: a copy there
    is charged to nothing, and asking this function for its "usage" would be
    asking for a number the design does not define (``M1_DESIGN.md`` §5).
    """
    assert domain in DOMAINS, f"{domain!r} is not a storage domain"
    architecture = scenario.architecture
    used = architecture.reserved_of(domain)
    for tensor_id, status in state.statuses(domain):
        if status != COPY_ABSENT:
            used += scenario.size_alloc(tensor_id, domain)
    for task in state.running:
        if task.resource not in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE):
            continue
        if task_domain(scenario, task) != domain:
            continue
        used += scenario.workspace_alloc(task.target_id, _task_device(scenario, task))
    return used


def used_vram_bytes(scenario: Scenario, state: State) -> int:
    """VRAM in use. The 0.1 name, kept because artefacts and the viewer use it."""
    return used_bytes(scenario, state, LOC_VRAM)


def _read_locks(scenario: Scenario, state: State) -> dict[tuple[str, str], int]:
    """``(tensor, domain)`` pairs currently locked by a running task, counted.

    A lock exists exactly while a task that reads the copy is in flight; it is
    derived from ``state.running`` rather than stored, so it cannot drift away
    from the task list. The count matters for ``add(x, x)``: one allocation, but
    the same operand named twice.

    Locks are per *copy*, not per tensor. That is what lets a CPU compute
    reading the DRAM copy coexist with an eviction of the untouched VRAM copy
    (``M1_DESIGN.md`` §6).

    A ``WRITE_SSD`` locks the DRAM copy it reads, exactly as any other transfer
    locks its source: the design is explicit that the source stays resident and
    locked until the write completes (``M1_DESIGN.md`` §5). An SSD *source*
    locks nothing, because an SSD copy is not an allocation and no action can
    take it away while a read is in flight.
    """
    locks: dict[tuple[str, str], int] = {}
    for task in state.running:
        if task.resource in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE):
            operation = scenario.workload.operation_by_id[task.target_id]
            domain = task_domain(scenario, task)
        else:
            if task.route is None:
                source = LOC_DRAM  # 0.1's COPY_H2D reads DRAM
            else:
                source = ROUTE_ENDS[task.route][0]
            if source == LOC_SSD:
                continue  # nothing can invalidate an SSD copy, so nothing to lock
            operation = None
            domain = source
        if operation is not None:
            for tensor_id in set(operation.inputs):
                key = (tensor_id, domain)
                locks[key] = locks.get(key, 0) + 1
        else:
            key = (task.target_id, domain)
            locks[key] = locks.get(key, 0) + 1
    return locks


def _has_recoverable_dram_source(scenario: Scenario, tensor_id: str) -> bool:
    """Whether an evicted tensor could be brought back.

    For a leaf with a DRAM copy that is also in the mapspace, yes. For an
    intermediate result, no: M0 has no recomputation, so evicting it destroys it
    forever (``DESIGN.md`` §6).
    """
    if tensor_id not in scenario.mapspace.copy_tensor_ids:
        return False
    tensor = scenario.workload.tensor_by_id[tensor_id]
    return LOC_DRAM in tensor.initial_locations


# ---------------------------------------------------------------------------
# Initial state and goal
# ---------------------------------------------------------------------------


def initial_state(scenario: Scenario) -> State:
    """The state at ``t = 0``.

    Raises :class:`InitialCapacityExceeded` when the starting layout alone does
    not fit -- reported as ``infeasible`` by callers, not as an input error.
    """
    check_initial_capacity(scenario)

    op_status = tuple(
        (op.id, OP_NOT_STARTED) for op in sorted(scenario.workload.operations, key=lambda o: o.id)
    )
    copy_status = tuple(
        (t.id, COPY_READY if LOC_VRAM in t.initial_locations else COPY_ABSENT)
        for t in sorted(scenario.workload.tensors, key=lambda t: t.id)
    )
    # Silent for 0.1: DRAM there is unbounded and static, so it has no ledger.
    # Recording READY copies anyway would put a constant into every state and
    # every state key, which is harmless for correctness and misleading for a
    # reader comparing a 0.1 state artefact with a 0.2 one.
    dram_status: tuple[tuple[str, str], ...] = ()
    ssd_status: tuple[tuple[str, str], ...] = ()
    if scenario.is_m1:
        dram_status = tuple(
            (t.id, COPY_READY if LOC_DRAM in t.initial_locations else COPY_ABSENT)
            for t in sorted(scenario.workload.tensors, key=lambda t: t.id)
        )
        # The SSD ledger starts as the scene's static content, and only a write
        # ever changes it. Seeding it here rather than special-casing "the SSD is
        # static" at each read is what makes one rule -- "a copy is READY where
        # it is READY" -- cover initial leaves and runtime snapshots alike.
        ssd_status = tuple(
            (t.id, COPY_READY if LOC_SSD in t.initial_locations else COPY_ABSENT)
            for t in sorted(scenario.workload.tensors, key=lambda t: t.id)
        )
    return State(
        t=0,
        op_status=op_status,
        copy_status=copy_status,
        running=(),
        dram_status=dram_status,
        ssd_status=ssd_status,
    )


def is_goal(scenario: Scenario, state: State) -> bool:
    """All work done, nothing in flight, every requested output *delivered*.

    "Delivered" is the 0.2 word: the output has to be READY in the domain the
    scenario named, not merely produced somewhere. A finished compute whose
    result is still on the other device is not a goal, which is exactly the
    mistake ``M1_ACCEPTANCE.md`` §3 calls out ("4 ms 时全部算子 DONE 但 y 仍在
    VRAM，不能判 goal").

    An SSD target is delivered by the same test and needs no extra branch: an
    SSD copy is READY only once its write has completed, so the in-flight write
    is a ``RESERVED_COPY`` and the goal does not hold yet
    (``M1_ACCEPTANCE.md`` §9).
    """
    if state.running:
        return False
    for operation in scenario.workload.operations:
        if state.op(operation.id) != OP_DONE:
            return False
    for tensor_id in scenario.workload.outputs:
        if state.copy_at(tensor_id, scenario.goal_location(tensor_id)) != COPY_READY:
            return False
    return True


# ---------------------------------------------------------------------------
# Action legality -- one implementation, shared by search and evaluation
# ---------------------------------------------------------------------------


def _check(scenario: Scenario, state: State, action: Action) -> None:
    """Raise :class:`MappingError` if ``action`` is not legal in ``state``.

    This is the single source of truth for the transition rules.
    :func:`legal_actions` is implemented on top of it by trial, so the two can
    never disagree about what is allowed.
    """
    if action.kind == ACTION_ADVANCE:
        _check_advance(scenario, state, action)
    elif scenario.is_m1:
        _check_m1(scenario, state, action)
    elif action.kind == ACTION_COPY_H2D:
        _check_copy(scenario, state, action)
    elif action.kind == ACTION_COMPUTE:
        _check_compute(scenario, state, action)
    elif action.kind == ACTION_EVICT:
        _check_evict(scenario, state, action)
    else:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown action kind {action.kind!r}",
            t_ns=state.t,
        )


def _check_advance(scenario: Scenario, state: State, action: Action) -> None:
    if not state.running:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            "ADVANCE with nothing in flight would move no time forward",
            t_ns=state.t,
        )


def _check_copy(scenario: Scenario, state: State, action: Action) -> None:
    tensor_id = action.target_id
    assert tensor_id is not None
    architecture = scenario.architecture

    if tensor_id not in scenario.workload.tensor_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown tensor {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if tensor_id not in scenario.mapspace.copy_tensor_ids:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} is not in the mapspace's copy_tensor_ids",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if state.copy(tensor_id) != COPY_ABSENT:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} already has a GPU copy in state {state.copy(tensor_id)}; "
            "M0 never issues a second copy of a resident tensor",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if state.task_on(RESOURCE_COPY) is not None:
        raise MappingError(
            CODE_RESOURCE_BUSY,
            f"the {RESOURCE_COPY} resource is busy",
            t_ns=state.t,
            tensor_id=tensor_id,
            resource=RESOURCE_COPY,
        )
    # A prefetch that nothing is waiting for is not a legal action: the state
    # would be indistinguishable from one that never issued it, and the search
    # would explore a pointless branch for every such tensor.
    waiting = [
        op_id
        for op_id in scenario.workload.consumers_of[tensor_id]
        if state.op(op_id) == OP_NOT_STARTED
    ]
    if not waiting:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"no consumer of {tensor_id!r} still needs it; every consumer is "
            "running or done",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if not scenario.mapspace.allow_copy_compute_overlap:
        if state.task_on(RESOURCE_COMPUTE) is not None:
            raise MappingError(
                CODE_RESOURCE_BUSY,
                "copy/compute overlap is disabled and a compute is in flight",
                t_ns=state.t,
                tensor_id=tensor_id,
                resource=RESOURCE_COMPUTE,
            )
    needed = scenario.size_alloc(tensor_id)
    used = used_vram_bytes(scenario, state)
    if used + needed > architecture.vram_capacity_bytes:
        raise MappingError(
            CODE_CAPACITY_EXCEEDED,
            f"loading {tensor_id!r} needs {needed} bytes on top of {used} in use, "
            f"over the {architecture.vram_capacity_bytes}-byte budget",
            t_ns=state.t,
            tensor_id=tensor_id,
        )


def _check_compute(scenario: Scenario, state: State, action: Action) -> None:
    operation_id = action.target_id
    assert operation_id is not None
    architecture = scenario.architecture

    if operation_id not in scenario.workload.operation_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown operation {operation_id!r}",
            t_ns=state.t,
            operation_id=operation_id,
        )
    status = state.op(operation_id)
    if status != OP_NOT_STARTED:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"operation {operation_id!r} is already {status}",
            t_ns=state.t,
            operation_id=operation_id,
        )
    if state.task_on(RESOURCE_COMPUTE) is not None:
        raise MappingError(
            CODE_RESOURCE_BUSY,
            f"the {RESOURCE_COMPUTE} resource is busy",
            t_ns=state.t,
            operation_id=operation_id,
            resource=RESOURCE_COMPUTE,
        )
    if not scenario.mapspace.allow_copy_compute_overlap:
        if state.task_on(RESOURCE_COPY) is not None:
            raise MappingError(
                CODE_RESOURCE_BUSY,
                "copy/compute overlap is disabled and a copy is in flight",
                t_ns=state.t,
                operation_id=operation_id,
                resource=RESOURCE_COPY,
            )

    operation = scenario.workload.operation_by_id[operation_id]
    missing = sorted({tid for tid in operation.inputs if state.copy(tid) != COPY_READY})
    if missing:
        raise MappingError(
            CODE_INPUT_NOT_READY,
            f"operation {operation_id!r} needs {', '.join(missing)} resident in VRAM "
            "before it can start; M0 reserves nothing ahead of time",
            t_ns=state.t,
            operation_id=operation_id,
        )

    if state.copy(operation.output) != COPY_ABSENT:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"output {operation.output!r} of {operation_id!r} is already "
            f"{state.copy(operation.output)}; M0 has no in-place reuse and never "
            "writes a destination twice",
            t_ns=state.t,
            operation_id=operation_id,
            tensor_id=operation.output,
        )

    # Capacity is checked on the state as it is now. The engine does not
    # speculate that a currently-live tensor might be freed before this compute
    # finishes -- that would be reserving against a future the mapping has not
    # committed to.
    needed = scenario.size_alloc(operation.output) + scenario.workspace_alloc(operation_id)
    used = used_vram_bytes(scenario, state)
    if used + needed > architecture.vram_capacity_bytes:
        raise MappingError(
            CODE_CAPACITY_EXCEEDED,
            f"running {operation_id!r} needs {needed} bytes for {operation.output!r} "
            f"and its workspace on top of {used} in use, over the "
            f"{architecture.vram_capacity_bytes}-byte budget",
            t_ns=state.t,
            operation_id=operation_id,
            tensor_id=operation.output,
        )


def _check_evict(scenario: Scenario, state: State, action: Action) -> None:
    tensor_id = action.target_id
    assert tensor_id is not None

    if tensor_id not in scenario.workload.tensor_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown tensor {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if not scenario.mapspace.allow_eviction:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"eviction is disabled by the mapspace, cannot evict {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    status = state.copy(tensor_id)
    if status != COPY_READY:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} is {status}; only a READY GPU copy can be evicted "
            "(a reservation is still in flight and owns its bytes)",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    # A running reader comes first: it is a transient conflict, and reporting it
    # as data loss would point at the wrong problem. 0.1 has one domain, so the
    # lock it could hold is the VRAM one.
    if _read_locks(scenario, state).get((tensor_id, LOC_VRAM)):
        raise MappingError(
            CODE_TENSOR_IN_USE,
            f"{tensor_id!r} is being read by an in-flight compute",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    if tensor_id in scenario.workload.outputs:
        raise MappingError(
            CODE_LIVE_VALUE_LOSS,
            f"{tensor_id!r} is a requested output; evicting it would have to be "
            "undone before the goal is reachable",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    if _has_recoverable_dram_source(scenario, tensor_id):
        return

    # No DRAM source and no recomputation: every consumer must already have run,
    # or the value is gone for good.
    unfinished = sorted(
        op_id
        for op_id in scenario.workload.consumers_of[tensor_id]
        if state.op(op_id) != OP_DONE
    )
    if unfinished:
        raise MappingError(
            CODE_LIVE_VALUE_LOSS,
            f"{tensor_id!r} has no DRAM copy to reload from and no recomputation is "
            f"allowed, but {', '.join(unfinished)} still need it",
            t_ns=state.t,
            tensor_id=tensor_id,
        )


# ---------------------------------------------------------------------------
# M1 (0.2) action legality
#
# A second rule set rather than a parameterised version of the first, and the
# reason is not that the rules are unrelated -- they are close cousins -- but
# that the 0.1 rules are a *frozen contract*: their codes and their wording are
# pinned by 202 tests, by stored mappings and by artefacts on disk, and 0.1's
# semantics are explicitly kept (M1_DESIGN.md §8). Rewriting them into a
# generalised form and then asserting the generalisation reproduces them is a
# much larger claim than it looks, and the failure mode is a silently different
# M0 answer. The parts that must not drift -- the state, the transition, the
# clock, the ledger, the evaluator -- are shared, and both rule sets are
# exercised by the same ``legal_actions``/``transition`` pair, so M1's search and
# M1's evaluator still cannot disagree.
# ---------------------------------------------------------------------------


def _m1_mapspace(scenario: Scenario) -> M1Mapspace:
    assert isinstance(scenario.mapspace, M1Mapspace)
    return scenario.mapspace


def _check_m1(scenario: Scenario, state: State, action: Action) -> None:
    mapspace = _m1_mapspace(scenario)
    if action.kind == ACTION_COMPUTE:
        _m1_check_compute(scenario, state, action, mapspace)
    elif action.kind in (ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD):
        # One checker for all three: they differ only in which route they name,
        # and every precondition in M1_DESIGN.md §6 is a statement about the
        # route's two ends rather than about the verb.
        _m1_check_transfer(scenario, state, action, mapspace)
    elif action.kind == ACTION_EVICT:
        _m1_check_evict(scenario, state, action, mapspace)
    else:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown 0.2 action kind {action.kind!r}",
            t_ns=state.t,
        )


def _m1_check_resources_free(
    scenario: Scenario,
    state: State,
    action: Action,
    mapspace: M1Mapspace,
    *,
    resource: str,
    device: str | None,
) -> None:
    """The two overlap switches, checked from whichever side is starting.

    ``allow_copy_compute_overlap=false`` makes *any* compute and *any*
    PCIe/SSD transfer mutually exclusive, and it says nothing about the SSD
    overlapping the PCIe engine -- those are two different resources and they
    still overlap, in both directions (``M1_DESIGN.md`` §6).
    ``allow_cpu_gpu_overlap=false`` only forbids CPU and GPU computing at the
    same time; copies are unaffected.
    """
    occupant = state.task_on(resource)
    if occupant is not None:
        raise MappingError(
            CODE_RESOURCE_BUSY,
            f"the {resource} resource is busy with {occupant.target_id!r}",
            t_ns=state.t,
            operation_id=action.target_id if action.kind == ACTION_COMPUTE else None,
            tensor_id=action.target_id if action.kind != ACTION_COMPUTE else None,
            resource=resource,
        )

    compute_resources = (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE)
    is_compute = resource in compute_resources
    if not mapspace.allow_copy_compute_overlap:
        blocking = (RESOURCE_PCIE_COPY, RESOURCE_SSD_IO) if is_compute else compute_resources
        for other in blocking:
            if state.task_on(other) is not None:
                raise MappingError(
                    CODE_RESOURCE_BUSY,
                    f"copy/compute overlap is disabled and the {other} resource is "
                    "in use",
                    t_ns=state.t,
                    resource=other,
                )
    if is_compute and not mapspace.allow_cpu_gpu_overlap:
        other = (
            RESOURCE_GPU_COMPUTE if resource == RESOURCE_CPU_COMPUTE
            else RESOURCE_CPU_COMPUTE
        )
        if state.task_on(other) is not None:
            raise MappingError(
                CODE_RESOURCE_BUSY,
                "CPU/GPU compute overlap is disabled and the other device is busy",
                t_ns=state.t,
                resource=other,
            )


def _m1_check_compute(
    scenario: Scenario, state: State, action: Action, mapspace: M1Mapspace
) -> None:
    operation_id = action.target_id
    assert operation_id is not None
    if operation_id not in scenario.workload.operation_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown operation {operation_id!r}",
            t_ns=state.t,
            operation_id=operation_id,
        )
    device = action.device
    if device is None:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"0.2 COMPUTE({operation_id!r}) does not say which device to run on; "
            "there is no default device",
            t_ns=state.t,
            operation_id=operation_id,
        )
    allowed = mapspace.devices_for(operation_id)
    if device not in allowed:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{device!r} is not an allowed device for {operation_id!r} "
            f"(allowed: {', '.join(allowed) or 'none'})",
            t_ns=state.t,
            operation_id=operation_id,
        )

    status = state.op(operation_id)
    if status != OP_NOT_STARTED:
        # Covers the "two devices try the same node" case without a separate
        # rule: whichever starts second finds the operation no longer
        # NOT_STARTED, whatever device it named (M1_ACCEPTANCE.md §8).
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"operation {operation_id!r} is already {status}",
            t_ns=state.t,
            operation_id=operation_id,
        )

    domain = DOMAIN_OF_DEVICE[device]
    _m1_check_resources_free(
        scenario, state, action, mapspace,
        resource=RESOURCE_OF_DEVICE[device],
        device=device,
    )

    operation = scenario.workload.operation_by_id[operation_id]
    missing = sorted({t for t in operation.inputs if state.copy_at(t, domain) != COPY_READY})
    if missing:
        raise MappingError(
            CODE_INPUT_NOT_READY,
            f"operation {operation_id!r} on {device} needs {', '.join(missing)} "
            f"resident in {domain}; M1 never moves an input to satisfy a compute",
            t_ns=state.t,
            operation_id=operation_id,
        )

    if any(state.copy_at(operation.output, d) != COPY_ABSENT for d in DOMAINS):
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"output {operation.output!r} of {operation_id!r} already has a copy; "
            "M1 has no in-place reuse and never writes a destination twice",
            t_ns=state.t,
            operation_id=operation_id,
            tensor_id=operation.output,
        )

    needed = scenario.size_alloc(operation.output, domain) + scenario.workspace_alloc(
        operation_id, device
    )
    _m1_check_capacity(
        scenario, state, domain, needed,
        what=f"running {operation_id!r} on {device} (output {operation.output!r} "
             "plus its workspace)",
        operation_id=operation_id,
        tensor_id=operation.output,
    )


def _m1_check_capacity(
    scenario: Scenario,
    state: State,
    domain: str,
    needed: int,
    *,
    what: str,
    tensor_id: str | None = None,
    operation_id: str | None = None,
) -> None:
    """Charge ``needed`` to ``domain`` or refuse, naming the domain and the numbers.

    The check is made against the state as it is. M1 does not speculate that a
    live tensor might be freed before this task finishes: that would be
    reserving against a future the mapping has not committed to
    (``M1_DESIGN.md`` §5, and ``M1_ACCEPTANCE.md`` §5's "c1 运行中尝试 READ W2
    必须 CAPACITY_EXCEEDED，不能隐式排队至 t=2").
    """
    capacity = scenario.architecture.capacity_of(domain)
    if capacity is None:
        return
    used = used_bytes(scenario, state, domain)
    if used + needed > capacity:
        raise MappingError(
            CODE_CAPACITY_EXCEEDED,
            f"{what} needs {needed} bytes on top of {used} in use in {domain}, over "
            f"the {capacity}-byte budget",
            t_ns=state.t,
            tensor_id=tensor_id,
            operation_id=operation_id,
        )


def _m1_check_transfer(
    scenario: Scenario, state: State, action: Action, mapspace: M1Mapspace
) -> None:
    tensor_id = action.target_id
    assert tensor_id is not None
    if tensor_id not in scenario.workload.tensor_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown tensor {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    route = action.route
    if route is None:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{action.kind} does not name a route M1 models (src={action.src!r}, "
            f"dst={action.dst!r})",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if not mapspace.allows(route, tensor_id):
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} is not allowed on route {route!r} "
            f"(allowed: {', '.join(mapspace.tensors_for(route)) or 'none'})",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    source, destination = ROUTE_ENDS[route]
    resource = RESOURCE_OF_ROUTE[route]

    if state.copy_at(tensor_id, source) != COPY_READY:
        # One rule for all four routes. An SSD source used to need its own
        # branch, back when the SSD was read-only scene content with no copy
        # status at all; now that a write creates a real copy with a real
        # status, "the source must be READY" says everything: a leaf with a
        # static SSD copy is READY from t=0, a snapshot is READY only after its
        # write landed, and a tensor with no SSD copy at all is ABSENT
        # (``M1_ACCEPTANCE.md`` §10).
        raise MappingError(
            CODE_INPUT_NOT_READY,
            f"{tensor_id!r} is {state.copy_at(tensor_id, source)} in {source}; only a "
            f"READY {source} copy can be the source of a {route} transfer "
            "(a reservation is still in flight and cannot be read)",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    target_status = state.copy_at(tensor_id, destination)
    if target_status != COPY_ABSENT:
        # Refused rather than ignored: M1 never issues a second copy of a
        # resident tensor, never overwrites an SSD snapshot with a second write
        # of the same name, and never re-charges a reservation it already made
        # (M1_ACCEPTANCE.md §10).
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} is already {target_status} in {destination}; M1 never "
            "copies onto a READY or RESERVED destination",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    _m1_check_resources_free(
        scenario, state, action, mapspace, resource=resource, device=None
    )

    if not _m1_transfer_has_a_purpose(scenario, state, tensor_id, destination):
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"nothing needs {tensor_id!r} in {destination}: every consumer has "
            "started and it is not an output still to be delivered there",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    # Charged to the *destination*. For the two SSD directions the destination
    # is the SSD, which has no capacity, so the check passes without a number --
    # and, deliberately, a WRITE_SSD charges nothing to DRAM either: it reads a
    # copy that is already resident and adds no second buffer
    # (``M1_DESIGN.md`` §5).
    _m1_check_capacity(
        scenario, state, destination, scenario.size_alloc(tensor_id, destination),
        what=f"copying {tensor_id!r} into {destination}",
        tensor_id=tensor_id,
    )


def _m1_transfer_has_a_purpose(
    scenario: Scenario, state: State, tensor_id: str, destination: str
) -> bool:
    """Whether a new copy in ``destination`` could still be needed.

    Two ways to be needed, and both are required by ``M1_DESIGN.md`` §6: a
    consumer that has not started, or a requested output that can still be
    carried from ``destination`` to the location its goal names.

    The second test is reachability rather than equality, and that is load
    bearing: an output whose goal is the SSD cannot be written there from VRAM,
    so the D2H that gets it into DRAM is not a detour, it is the first leg
    (``M1_ACCEPTANCE.md`` §9's 4 ms GPU plan). Demanding ``destination == goal``
    would make that plan unrepresentable and the case infeasible.

    The equality case is a special case of reachability -- a location is always
    reachable from itself -- so the delivery itself is still covered.
    """
    for op_id in scenario.workload.consumers_of[tensor_id]:
        if state.op(op_id) == OP_NOT_STARTED:
            return True
    if tensor_id in scenario.workload.outputs:
        goal = scenario.goal_location(tensor_id)
        return goal in reachable_locations(
            (destination,), _m1_mapspace(scenario).allowed_transfers, tensor_id
        )
    return False


def _m1_check_evict(
    scenario: Scenario, state: State, action: Action, mapspace: M1Mapspace
) -> None:
    tensor_id = action.target_id
    assert tensor_id is not None
    if tensor_id not in scenario.workload.tensor_by_id:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"unknown tensor {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    if not mapspace.allow_eviction:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"eviction is disabled by the mapspace, cannot evict {tensor_id!r}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )
    location = action.location
    if location not in DOMAINS:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"0.2 EVICT({tensor_id!r}) names location {location!r}; it must be one of "
            f"{list(DOMAINS)} and there is no default",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    status = state.copy_at(tensor_id, location)
    if status != COPY_READY:
        raise MappingError(
            CODE_ILLEGAL_ACTION,
            f"{tensor_id!r} is {status} in {location}; only a READY copy can be "
            "evicted (a reservation is in flight and owns its bytes)",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    if _read_locks(scenario, state).get((tensor_id, location)):
        raise MappingError(
            CODE_TENSOR_IN_USE,
            f"{tensor_id!r} in {location} is being read by an in-flight task",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    if (
        tensor_id in scenario.workload.outputs
        and scenario.goal_location(tensor_id) == location
    ):
        raise MappingError(
            CODE_LIVE_VALUE_LOSS,
            f"{tensor_id!r} in {location} is where the goal wants it; evicting it "
            "would have to be undone before the goal is reachable",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    _m1_check_evict_reachability(scenario, state, tensor_id, location)


def _m1_check_evict_reachability(
    scenario: Scenario, state: State, tensor_id: str, location: str
) -> None:
    """After dropping this copy, is every value still obtainable where it is needed?

    ``M1_DESIGN.md`` §6 spells the check out: build the reachability relation
    over the three locations from the *remaining* READY copies plus any valid
    SSD source, then require that every not-yet-started consumer has at least one
    of its allowed devices reachable, and that every not-yet-delivered output can
    still reach the domain its goal names.

    A *finished* SSD copy counts as a backup whether it was initial scene data
    or written at runtime, because both can be read back -- a runtime snapshot
    being the whole point of taking one. An SSD copy still being written does
    not count, for the same reason a ``RESERVED_COPY`` in DRAM does not: it is a
    promise, not a value (``M1_DESIGN.md`` §6).

    Two things it deliberately does not claim. It ignores capacity and
    busy-ness, so passing it is not a proof that the rest of the plan can be
    executed -- it says the *value* is recoverable, not that the machine can
    afford the trip. And it does not require keeping an option for every device a
    future consumer might use: while the device is still unchosen, keeping one
    reachable device is enough.
    """
    mapspace = _m1_mapspace(scenario)

    remaining = [
        loc
        for loc in DOMAINS
        if loc != location and state.copy_at(tensor_id, loc) == COPY_READY
    ]
    # A RESERVED copy -- in DRAM, in VRAM or on the SSD -- is *not* a backup: it
    # is bytes committed to a transfer that has not landed, and counting it as
    # present would let the only READY value be deleted on the strength of a
    # copy that does not exist yet (``M1_ACCEPTANCE.md`` §8).
    if state.copy_at(tensor_id, LOC_SSD) == COPY_READY:
        remaining.append(LOC_SSD)

    reachable = reachable_locations(tuple(remaining), mapspace.allowed_transfers, tensor_id)

    unreachable: list[str] = []
    for op_id in scenario.workload.consumers_of[tensor_id]:
        if state.op(op_id) == OP_DONE:
            continue
        devices = mapspace.devices_for(op_id)
        if devices and any(DOMAIN_OF_DEVICE[d] in reachable for d in devices):
            continue
        unreachable.append(op_id)
    if unreachable:
        raise MappingError(
            CODE_LIVE_VALUE_LOSS,
            f"evicting {tensor_id!r} from {location} leaves no way to get it to "
            f"{', '.join(sorted(unreachable))}; the remaining copies are "
            f"{', '.join(sorted(remaining)) or 'none'}",
            t_ns=state.t,
            tensor_id=tensor_id,
        )

    if tensor_id in scenario.workload.outputs:
        goal = scenario.goal_location(tensor_id)
        if state.copy_at(tensor_id, goal) != COPY_READY and goal not in reachable:
            raise MappingError(
                CODE_LIVE_VALUE_LOSS,
                f"evicting {tensor_id!r} from {location} leaves no way to deliver the "
                f"requested output to {goal}",
                t_ns=state.t,
                tensor_id=tensor_id,
            )


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def transition(scenario: Scenario, state: State, action: Action) -> State:
    """Apply ``action``, returning the successor state.

    COPY_H2D, COPY, READ_SSD, COMPUTE and EVICT are instantaneous commitments:
    they start a task or drop a copy but never move the clock. Only ADVANCE moves
    time, by the smallest remaining duration, completing everything that ends at
    that instant as one batch (``DESIGN.md`` §6).
    """
    _check(scenario, state, action)

    if action.kind == ACTION_ADVANCE:
        return _advance(scenario, state)
    if scenario.is_m1:
        if action.kind == ACTION_COMPUTE:
            return _m1_start_compute(scenario, state, action)
        if action.kind == ACTION_EVICT:
            return _m1_evict(scenario, state, action)
        return _m1_start_transfer(scenario, state, action)
    if action.kind == ACTION_COPY_H2D:
        return _start_copy(scenario, state, action.target_id)  # type: ignore[arg-type]
    if action.kind == ACTION_COMPUTE:
        return _start_compute(scenario, state, action.target_id)  # type: ignore[arg-type]
    return _evict(scenario, state, action.target_id)  # type: ignore[arg-type]


def _start_copy(scenario: Scenario, state: State, tensor_id: str) -> State:
    duration = scenario.costs.h2d[tensor_id].duration_ns
    copies = dict(state.copy_status)
    copies[tensor_id] = COPY_RESERVED_COPY
    running = _with_task(
        state.running,
        RunningTask(resource=RESOURCE_COPY, target_id=tensor_id, remaining_ns=duration),
    )
    return _rebuild(state, copies=copies, running=running)


def _start_compute(scenario: Scenario, state: State, operation_id: str) -> State:
    operation = scenario.workload.operation_by_id[operation_id]
    duration = scenario.costs.compute[operation_id].duration_ns

    ops = dict(state.op_status)
    ops[operation_id] = OP_RUNNING
    copies = dict(state.copy_status)
    copies[operation.output] = COPY_RESERVED_OUTPUT
    running = _with_task(
        state.running,
        RunningTask(resource=RESOURCE_COMPUTE, target_id=operation_id, remaining_ns=duration),
    )
    return _rebuild(state, ops=ops, copies=copies, running=running)


def _evict(scenario: Scenario, state: State, tensor_id: str) -> State:
    copies = dict(state.copy_status)
    copies[tensor_id] = COPY_ABSENT
    return _rebuild(state, copies=copies)


def _m1_start_transfer(scenario: Scenario, state: State, action: Action) -> State:
    """Reserve the destination copy and occupy the route's resource."""
    assert action.target_id is not None
    route = action.route
    assert route is not None  # _m1_check_transfer rejected a route-less action
    costs = scenario.costs
    assert isinstance(costs, M1Costs)
    duration = costs.transfer_cost(route, action.target_id)
    assert duration is not None  # the mapspace/costs agreement was checked at parse

    source, destination = ROUTE_ENDS[route]
    copies = dict(state.copy_status)
    dram = dict(state.dram_status)
    ssd = dict(state.ssd_status)
    # The source is not touched. A transfer *adds* a copy; it never consumes the
    # one it reads, which is what lets a weight serve several consumers from the
    # same resident original (M1_DESIGN.md §6). A WRITE_SSD is the same rule: the
    # DRAM copy stays READY afterwards and only an explicit EVICT frees it.
    _set_copy(copies, dram, ssd, action.target_id, destination, COPY_RESERVED_COPY)

    running = _with_task(
        state.running,
        RunningTask(
            resource=RESOURCE_OF_ROUTE[route],
            target_id=action.target_id,
            remaining_ns=duration.duration_ns,
            route=route,
        ),
    )
    return _rebuild(state, copies=copies, dram=dram, ssd=ssd, running=running)


def _m1_start_compute(scenario: Scenario, state: State, action: Action) -> State:
    assert action.target_id is not None and action.device is not None
    operation = scenario.workload.operation_by_id[action.target_id]
    costs = scenario.costs
    assert isinstance(costs, M1Costs)
    duration = costs.compute_cost(action.target_id, action.device)
    assert duration is not None

    ops = dict(state.op_status)
    ops[action.target_id] = OP_RUNNING
    copies = dict(state.copy_status)
    dram = dict(state.dram_status)
    ssd = dict(state.ssd_status)
    # The output is reserved *in its device's domain*, so the reservation and the
    # completion cannot disagree about where the bytes will be.
    _set_copy(
        copies, dram, ssd, operation.output, DOMAIN_OF_DEVICE[action.device],
        COPY_RESERVED_OUTPUT,
    )

    running = _with_task(
        state.running,
        RunningTask(
            resource=RESOURCE_OF_DEVICE[action.device],
            target_id=action.target_id,
            remaining_ns=duration.duration_ns,
            device=action.device,
        ),
    )
    return _rebuild(state, ops=ops, copies=copies, dram=dram, ssd=ssd, running=running)


def _m1_evict(scenario: Scenario, state: State, action: Action) -> State:
    assert action.target_id is not None and action.location is not None
    copies = dict(state.copy_status)
    dram = dict(state.dram_status)
    ssd = dict(state.ssd_status)
    _set_copy(copies, dram, ssd, action.target_id, action.location, COPY_ABSENT)
    return _rebuild(state, copies=copies, dram=dram, ssd=ssd)


def _set_copy(
    copies: dict[str, str],
    dram: dict[str, str],
    ssd: dict[str, str],
    tensor_id: str,
    location: str,
    status: str,
) -> None:
    """Write one copy's status into the ledger that owns ``location``.

    The mirror of :meth:`State.copy_at`, and the only place that writes. Three
    separate dicts exist because 0.1 has no DRAM ledger and because the SSD is
    not an allocation domain; funnelling every write through here keeps the read
    and write sides from picking different fields.
    """
    if location == LOC_DRAM:
        dram[tensor_id] = status
    elif location == LOC_SSD:
        ssd[tensor_id] = status
    else:
        copies[tensor_id] = status


def _advance(scenario: Scenario, state: State) -> State:
    """Move time forward to the earliest completion and finish that whole batch.

    Everything ending at the same instant completes together. Completing them
    one at a time would create intermediate states that no action list can
    actually produce, and would let a state key distinguish orderings that are
    simultaneous and therefore indistinguishable.
    """
    delta = min(task.remaining_ns for task in state.running)

    ops = dict(state.op_status)
    copies = dict(state.copy_status)
    dram = dict(state.dram_status)
    ssd = dict(state.ssd_status)
    still_running: list[RunningTask] = []
    for task in state.running:
        remaining = task.remaining_ns - delta
        if remaining > 0:
            # ``device``/``route`` are carried, not dropped: a 0.2 task that
            # forgets which device it is on would complete into the wrong domain.
            still_running.append(
                RunningTask(
                    resource=task.resource,
                    target_id=task.target_id,
                    remaining_ns=remaining,
                    device=task.device,
                    route=task.route,
                )
            )
            continue
        _complete(scenario, task, ops, copies, dram, ssd)

    return State(
        t=state.t + delta,
        op_status=tuple(sorted(ops.items())),
        copy_status=tuple(sorted(copies.items())),
        running=tuple(sorted(still_running)),
        dram_status=tuple(sorted(dram.items())),
        ssd_status=tuple(sorted(ssd.items())),
    )


def _complete(
    scenario: Scenario,
    task: RunningTask,
    ops: dict[str, str],
    copies: dict[str, str],
    dram: dict[str, str],
    ssd: dict[str, str],
) -> None:
    """Finish one task in place: a compute its output, a transfer its destination.

    One function for both rule sets. The version difference is entirely in what
    the task carries -- a 0.1 transfer has no route, so it lands in VRAM, which
    is the one place 0.1 ever copies to.
    """
    if task.resource in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE):
        operation = scenario.workload.operation_by_id[task.target_id]
        _set_copy(
            copies,
            dram,
            ssd,
            operation.output,
            DOMAIN_OF_DEVICE[_task_device(scenario, task)],
            COPY_READY,
        )
        ops[task.target_id] = OP_DONE
        return
    if task.route is None:
        copies[task.target_id] = COPY_READY
        return
    # For a WRITE_SSD this is the moment the snapshot becomes readable: until
    # now the SSD copy was a RESERVED_COPY and no READ_SSD could use it, and the
    # DRAM source was locked because a task was reading it. Both end here.
    _set_copy(copies, dram, ssd, task.target_id, ROUTE_ENDS[task.route][1], COPY_READY)


def _with_task(running: Sequence[RunningTask], task: RunningTask) -> tuple[RunningTask, ...]:
    return tuple(sorted((*running, task)))


def _rebuild(
    state: State,
    *,
    ops: dict[str, str] | None = None,
    copies: dict[str, str] | None = None,
    running: Sequence[RunningTask] | None = None,
    dram: dict[str, str] | None = None,
    ssd: dict[str, str] | None = None,
) -> State:
    return State(
        t=state.t,
        op_status=tuple(sorted(ops.items())) if ops is not None else state.op_status,
        copy_status=tuple(sorted(copies.items())) if copies is not None else state.copy_status,
        running=tuple(sorted(running)) if running is not None else state.running,
        dram_status=tuple(sorted(dram.items())) if dram is not None else state.dram_status,
        ssd_status=tuple(sorted(ssd.items())) if ssd is not None else state.ssd_status,
    )


# ---------------------------------------------------------------------------
# Legal action enumeration
# ---------------------------------------------------------------------------


def legal_actions(scenario: Scenario, state: State) -> tuple[Action, ...]:
    """Every legal action from ``state``, in a fixed order.

    The order is part of the deterministic contract: it is what makes two runs
    of the search pick the same one of several equally optimal mappings. It is
    *not* a priority -- the search considers all of these -- so callers must not
    read it as "best first".
    """
    candidates = _m1_candidates(scenario) if scenario.is_m1 else _m0_candidates(scenario)
    legal: list[Action] = []
    for action in candidates:
        try:
            _check(scenario, state, action)
        except MappingError:
            continue
        legal.append(action)
    return tuple(legal)


def _m0_candidates(scenario: Scenario) -> list[Action]:
    candidates: list[Action] = []
    for tensor_id in sorted(scenario.mapspace.copy_tensor_ids):
        candidates.append(Action(ACTION_COPY_H2D, tensor_id))
    for operation_id in sorted(scenario.workload.operation_by_id):
        candidates.append(Action(ACTION_COMPUTE, operation_id))
    for tensor_id in sorted(scenario.workload.tensor_by_id):
        candidates.append(Action(ACTION_EVICT, tensor_id))
    candidates.append(Action(ACTION_ADVANCE))
    return candidates


def _m1_candidates(scenario: Scenario) -> list[Action]:
    """Every 0.2 action worth trying, in a fixed order.

    Grouped by *choice*, not by kind: all the ways to move a tensor come first
    (route by route, then tensor), then all the ways to run an operation (device
    by device), then all the ways to free space. Every axis is a sorted or
    declared-order tuple and none of them depends on the state, so two runs over
    the same scenario enumerate in the same order.

    Enumerating every (tensor, domain) pair rather than only the occupied ones is
    deliberate: the illegal transitions are what the search would otherwise have
    to rediscover as dead ends, and ``_check`` has to reject them anyway.
    """
    mapspace = _m1_mapspace(scenario)
    candidates: list[Action] = []
    for route in ROUTES:
        source, destination = ROUTE_ENDS[route]
        for tensor_id in sorted(mapspace.tensors_for(route)):
            if route == ROUTE_SSD_TO_DRAM:
                candidates.append(Action(ACTION_READ_SSD, tensor_id))
            elif route == ROUTE_DRAM_TO_SSD:
                candidates.append(Action(ACTION_WRITE_SSD, tensor_id))
            else:
                candidates.append(
                    Action(ACTION_COPY, tensor_id, src=source, dst=destination)
                )
    for operation_id in sorted(scenario.workload.operation_by_id):
        for device in mapspace.devices_for(operation_id):
            candidates.append(Action(ACTION_COMPUTE, operation_id, device=device))
    for tensor_id in sorted(scenario.workload.tensor_by_id):
        for domain in DOMAINS:
            candidates.append(Action(ACTION_EVICT, tensor_id, location=domain))
    candidates.append(Action(ACTION_ADVANCE))
    return candidates


# ---------------------------------------------------------------------------
# Fixed-mapping evaluation
# ---------------------------------------------------------------------------

STATUS_VALID = "valid"
STATUS_INVALID_MAPPING = "invalid_mapping"
STATUS_INCOMPLETE_MAPPING = "incomplete_mapping"
STATUS_INFEASIBLE = "infeasible"

REASON_INITIAL_CAPACITY_EXCEEDED = "initial_capacity_exceeded"
REASON_GOAL_REACHED = "goal_reached"
REASON_ACTIONS_REMAIN_AFTER_GOAL = "actions_remain_after_goal"
REASON_UNFINISHED = "mapping_ended_before_the_goal"


def evaluate_mapping(
    scenario: Scenario, actions: Iterable[Action], start: State | None = None
) -> EvaluationResult:
    """Replay a fixed mapping and report exactly what happens.

    The evaluator never repairs or corrects the mapping: it does not insert a
    prefetch, drop an eviction it thinks is unwise, or advance the clock on its
    own. A mapping is a claim about a concrete action order, and the job here is
    to say whether the claim holds (``DESIGN.md`` §6).
    """
    try:
        state = start if start is not None else initial_state(scenario)
    except InitialCapacityExceeded as exc:
        return EvaluationResult(
            status=STATUS_INFEASIBLE,
            reason=REASON_INITIAL_CAPACITY_EXCEEDED,
            makespan_ns=None,
            peak_vram_bytes=0,
            h2d_bytes=0,
            action_count=0,
            actions=(),
            events=(),
            states=(),
            error_code=REASON_INITIAL_CAPACITY_EXCEEDED,
        )

    peak = used_vram_bytes(scenario, state)
    peak_dram = used_bytes(scenario, state, LOC_DRAM)
    h2d_bytes = 0
    d2h_bytes = 0
    ssd_read_bytes = 0
    ssd_write_bytes = 0
    events: list[Event] = []
    snapshots = [_snapshot(scenario, state, 0)]

    action_list = tuple(actions)
    for index, action in enumerate(action_list):
        if is_goal(scenario, state):
            # The goal was already reachable; anything after it is dead code and
            # is reported rather than silently ignored, because a stored mapping
            # that overshoots is a mapping whose stated makespan is wrong.
            return _failure(
                scenario,
                state,
                index,
                STATUS_INVALID_MAPPING,
                REASON_ACTIONS_REMAIN_AFTER_GOAL,
                CODE_ILLEGAL_ACTION,
                f"the goal was reached at action index {index - 1}, but "
                f"{len(action_list) - index} more action(s) follow",
                peak,
                h2d_bytes,
                action_list,
                events,
                snapshots,
                peak_dram=peak_dram,
                d2h_bytes=d2h_bytes,
                ssd_read_bytes=ssd_read_bytes,
                ssd_write_bytes=ssd_write_bytes,
            )

        try:
            state = transition(scenario, state, action)
        except MappingError as error:
            return _failure(
                scenario,
                state,
                index,
                STATUS_INVALID_MAPPING,
                error.code,
                error.code,
                str(error),
                peak,
                h2d_bytes,
                action_list,
                events,
                snapshots,
                peak_dram=peak_dram,
                d2h_bytes=d2h_bytes,
                ssd_read_bytes=ssd_read_bytes,
                ssd_write_bytes=ssd_write_bytes,
            )

        event = _event_for(scenario, state, index, action)
        events.append(event)
        h2d_bytes, d2h_bytes, ssd_read_bytes, ssd_write_bytes = _accumulate_transfer_bytes(
            event, h2d_bytes, d2h_bytes, ssd_read_bytes, ssd_write_bytes
        )
        peak = max(peak, used_vram_bytes(scenario, state))
        peak_dram = max(peak_dram, used_bytes(scenario, state, LOC_DRAM))
        snapshots.append(_snapshot(scenario, state, index + 1))

    if not is_goal(scenario, state):
        return _failure(
            scenario,
            state,
            len(action_list),
            STATUS_INCOMPLETE_MAPPING,
            REASON_UNFINISHED,
            None,
            "the mapping ran out of actions before every requested output was "
            "resident and every operation done",
            peak,
            h2d_bytes,
            action_list,
            events,
            snapshots,
            peak_dram=peak_dram,
            d2h_bytes=d2h_bytes,
            ssd_read_bytes=ssd_read_bytes,
            ssd_write_bytes=ssd_write_bytes,
        )

    return EvaluationResult(
        status=STATUS_VALID,
        reason=REASON_GOAL_REACHED,
        makespan_ns=state.t,
        peak_vram_bytes=peak,
        h2d_bytes=h2d_bytes,
        action_count=len(action_list),
        actions=action_list,
        events=tuple(events),
        states=tuple(snapshots),
        d2h_bytes=d2h_bytes,
        ssd_read_bytes=ssd_read_bytes,
        ssd_write_bytes=ssd_write_bytes,
        peak_dram_bytes=peak_dram,
    )


def _accumulate_transfer_bytes(
    event: Event, h2d: int, d2h: int, ssd_read: int, ssd_write: int
) -> tuple[int, int, int, int]:
    """Add one event's bytes to the counter for the direction it moved in.

    Direction comes from the event's recorded ``src``/``dst``, not from its kind,
    so a mis-directed transfer cannot be counted as an H2D or as an SSD write.
    The 0.1 ``COPY_H2D`` event carries no endpoints and keeps its own branch,
    which is how its byte count stays exactly what it always was.
    """
    if event.kind == ACTION_COPY_H2D:
        return h2d + event.bytes_transferred, d2h, ssd_read, ssd_write
    if event.src == LOC_DRAM and event.dst == LOC_VRAM:
        return h2d + event.bytes_transferred, d2h, ssd_read, ssd_write
    if event.src == LOC_VRAM and event.dst == LOC_DRAM:
        return h2d, d2h + event.bytes_transferred, ssd_read, ssd_write
    if event.src == LOC_SSD:
        return h2d, d2h, ssd_read + event.bytes_transferred, ssd_write
    if event.dst == LOC_SSD:
        return h2d, d2h, ssd_read, ssd_write + event.bytes_transferred
    return h2d, d2h, ssd_read, ssd_write


def _event_for(scenario: Scenario, state: State, index: int, action: Action) -> Event:
    """Describe an applied action.

    ``start_ns`` and ``end_ns`` are both recorded. A copy begins at the current
    time and finishes ``duration_ns`` later; writing those as two fields removes
    any chance of reading a start time as a completion time, which is the
    confusion ``ACCEPTANCE.md`` §6 warns about.

    The 0.2 branches record *where* the work happened -- device, route ends,
    evicted location -- for the same reason: the event log is the thing a reader
    checks, so deriving the location from the scenario would let a mis-tagged
    event look right.
    """
    if action.kind == ACTION_ADVANCE:
        return Event(
            action_index=index,
            kind=ACTION_ADVANCE,
            target_id=None,
            resource=None,
            t_ns=state.t,
            start_ns=None,
            end_ns=state.t,
        )
    if scenario.is_m1:
        return _m1_event_for(scenario, state, index, action)
    if action.kind == ACTION_COPY_H2D:
        assert action.target_id is not None
        tensor = scenario.workload.tensor_by_id[action.target_id]
        duration = scenario.costs.h2d[action.target_id].duration_ns
        start = state.t
        return Event(
            action_index=index,
            kind=ACTION_COPY_H2D,
            target_id=action.target_id,
            resource=RESOURCE_COPY,
            t_ns=state.t,
            start_ns=start,
            end_ns=start + duration,
            # Bytes *moved* are the tensor's own size, never the aligned
            # allocation: those differ as soon as alignment exceeds one.
            bytes_transferred=tensor.storage_bytes,
        )
    if action.kind == ACTION_COMPUTE:
        assert action.target_id is not None
        operation = scenario.workload.operation_by_id[action.target_id]
        duration = scenario.costs.compute[action.target_id].duration_ns
        start = state.t
        return Event(
            action_index=index,
            kind=ACTION_COMPUTE,
            target_id=action.target_id,
            resource=RESOURCE_COMPUTE,
            t_ns=state.t,
            start_ns=start,
            end_ns=start + duration,
            workspace_bytes=scenario.workspace_alloc(action.target_id),
        )
    assert action.target_id is not None
    return Event(
        action_index=index,
        kind=ACTION_EVICT,
        target_id=action.target_id,
        resource=None,
        t_ns=state.t,
        start_ns=state.t,
        end_ns=state.t,
        bytes_released=scenario.size_alloc(action.target_id),
    )


def _m1_event_for(scenario: Scenario, state: State, index: int, action: Action) -> Event:
    """The 0.2 branch of :func:`_event_for`."""
    assert action.target_id is not None
    costs = scenario.costs
    assert isinstance(costs, M1Costs)

    if action.kind == ACTION_EVICT:
        assert action.location is not None
        return Event(
            action_index=index,
            kind=ACTION_EVICT,
            target_id=action.target_id,
            resource=None,
            t_ns=state.t,
            start_ns=state.t,
            end_ns=state.t,
            # Released bytes are the *aligned* allocation, the same number the
            # ledger charged, so the event and the ledger cannot disagree.
            bytes_released=scenario.size_alloc(action.target_id, action.location),
            location=action.location,
        )

    if action.kind == ACTION_COMPUTE:
        assert action.device is not None
        duration = costs.compute_cost(action.target_id, action.device)
        assert duration is not None
        start = state.t
        return Event(
            action_index=index,
            kind=ACTION_COMPUTE,
            target_id=action.target_id,
            resource=RESOURCE_OF_DEVICE[action.device],
            t_ns=state.t,
            start_ns=start,
            end_ns=start + duration.duration_ns,
            workspace_bytes=scenario.workspace_alloc(action.target_id, action.device),
            device=action.device,
        )

    tensor = scenario.workload.tensor_by_id[action.target_id]
    route = action.route
    assert route is not None
    duration = costs.transfer_cost(route, action.target_id)
    assert duration is not None
    source, destination = ROUTE_ENDS[route]
    start = state.t
    return Event(
        action_index=index,
        kind=action.kind,
        target_id=action.target_id,
        resource=RESOURCE_OF_ROUTE[route],
        t_ns=state.t,
        start_ns=start,
        end_ns=start + duration.duration_ns,
        bytes_transferred=tensor.storage_bytes,
        src=source,
        dst=destination,
    )


def _snapshot(scenario: Scenario, state: State, action_index: int) -> StateSnapshot:
    return StateSnapshot(
        t=state.t,
        used_vram_bytes=used_vram_bytes(scenario, state),
        op_status=state.op_status,
        copy_status=state.copy_status,
        used_dram_bytes=used_bytes(scenario, state, LOC_DRAM),
        dram_status=state.dram_status,
        ssd_status=state.ssd_status,
    )


def _failure(
    scenario: Scenario,
    state: State,
    index: int,
    status: str,
    reason: str,
    error_code: str | None,
    message: str,
    peak: int,
    h2d_bytes: int,
    actions: tuple[Action, ...],
    events: list[Event],
    snapshots: list[StateSnapshot],
    *,
    peak_dram: int = 0,
    d2h_bytes: int = 0,
    ssd_read_bytes: int = 0,
    ssd_write_bytes: int = 0,
) -> EvaluationResult:
    return EvaluationResult(
        status=status,
        reason=f"{reason}: {message}" if reason != error_code else f"{error_code}: {message}",
        makespan_ns=None,
        peak_vram_bytes=peak,
        h2d_bytes=h2d_bytes,
        action_count=len(events),
        actions=actions,
        events=tuple(events),
        states=tuple(snapshots),
        error_code=error_code,
        error_action_index=index,
        error_t_ns=state.t,
        d2h_bytes=d2h_bytes,
        ssd_read_bytes=ssd_read_bytes,
        ssd_write_bytes=ssd_write_bytes,
        peak_dram_bytes=peak_dram,
    )


# ---------------------------------------------------------------------------
# Timing helper for budgeted search
# ---------------------------------------------------------------------------


class WallClock:
    """Monotonic high-resolution wall-clock budget.

    Kept here rather than in the mapper so tests can substitute a deterministic
    clock (``M1_DESIGN.md``:225 asks for exactly that, so no budget test depends
    on how fast the machine happens to be).

    ``perf_counter`` and **not** ``monotonic``: on Windows ``monotonic`` is
    ``GetTickCount64``, whose resolution is one 64 Hz tick (measured 15.6 / 16.0
    ms on this machine, with 2000 consecutive calls returning the same value).
    Below one tick every ``elapsed_s()`` reads 0.0, so a budget of a few
    milliseconds never fired at all -- a 1 ms limit ran ~16 ms of search. It is
    still monotonic -- ``QueryPerformanceCounter`` is not moved by system clock
    adjustments -- so the property this class needs is kept.
    """

    def __init__(self, limit_s: float) -> None:
        self.limit_s = limit_s
        self._start = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    def expired(self) -> bool:
        return self.limit_s > 0 and self.elapsed_s() >= self.limit_s


# ---------------------------------------------------------------------------
# Mapping documents
# ---------------------------------------------------------------------------


class MappingIdentityError(Exception):
    """A mapping does not belong to the scenario it was replayed against.

    Separate from :class:`MappingError` because the mapping may be perfectly
    well-formed and legal -- it is simply a mapping of a *different* workload or
    configuration, and replaying it would produce confident nonsense
    (``DESIGN.md`` §6).
    """

    code = "identity_mismatch"


@dataclass(frozen=True)
class MappingDocument:
    """A parsed mapping file.

    ``fingerprint_verified`` is False for a hand-written mapping with no
    fingerprints. It is never set optimistically: an unverified mapping must not
    be able to present itself as an identity-checked one (``DESIGN.md`` §4.3).
    """

    schema_version: str
    scenario_id: str
    actions: tuple[Action, ...]
    workload_fingerprint: str | None
    scenario_fingerprint: str | None
    fingerprint_verified: bool
    path: str | None = None


_ACTION_KINDS = (ACTION_COPY_H2D, ACTION_COMPUTE, ACTION_EVICT, ACTION_ADVANCE)
_ACTION_KINDS_M1 = (
    ACTION_COPY,
    ACTION_READ_SSD,
    ACTION_WRITE_SSD,
    ACTION_COMPUTE,
    ACTION_EVICT,
    ACTION_ADVANCE,
)
_M1_ACTION_KEYS = {
    "kind",
    "tensor_id",
    "operation_id",
    "device",
    "src",
    "dst",
    "location",
    "comment",
}
# ``COPY`` is the domain-to-domain pair only. The SSD directions are actions of
# their own, which is what keeps "may be read from SSD" and "may be copied to
# VRAM" from being the same permission.
_COPY_ENDS_M1 = tuple(
    ROUTE_ENDS[route]
    for route in ROUTES
    if route not in (ROUTE_SSD_TO_DRAM, ROUTE_DRAM_TO_SSD)
)
_DEVICES_ORDER = tuple(sorted(DOMAIN_OF_DEVICE))  # sorted for the message's sake


def load_mapping(path: str | Path, scenario: Scenario) -> MappingDocument:
    """Load a mapping document and check it belongs to ``scenario``.

    Checking *shape* is all this function does. Whether an action is legal in a
    given state is decided by :func:`transition`, alone -- the loader must not
    grow a second opinion about the semantics, or the two would eventually
    disagree about which mappings are valid.
    """
    mapping_path = Path(path)
    raw = _require_mapping(_read_mapping_json(mapping_path), mapping_path.name)
    _reject_unknown_keys(
        raw,
        {
            "schema_version",
            "scenario_id",
            "actions",
            "workload_fingerprint",
            "scenario_fingerprint",
            "comment",
        },
        mapping_path.name,
    )

    schema_version = _check_schema_version(raw.get("schema_version"), mapping_path.name)

    scenario_id = _require_str(raw.get("scenario_id"), f"{mapping_path.name}.scenario_id")
    if scenario_id != scenario.id:
        raise MappingIdentityError(
            f"{mapping_path.name} names scenario {scenario_id!r} but was replayed "
            f"against {scenario.id!r}"
        )

    actual_workload = workload_fingerprint(scenario.workload)
    actual_scenario = scenario_fingerprint(scenario)
    declared_workload = _optional_str(raw, "workload_fingerprint", mapping_path.name)
    declared_scenario = _optional_str(raw, "scenario_fingerprint", mapping_path.name)

    for label, declared, actual in (
        ("workload_fingerprint", declared_workload, actual_workload),
        ("scenario_fingerprint", declared_scenario, actual_scenario),
    ):
        if declared is not None and declared != actual:
            raise MappingIdentityError(
                f"{mapping_path.name}.{label} is {declared} but this scenario's is "
                f"{actual}; the mapping was made for different semantics and "
                "replaying it would produce numbers that look right and are not"
            )

    actions_raw = _require_list(raw.get("actions"), f"{mapping_path.name}.actions")
    actions = tuple(
        _parse_action(entry, f"{mapping_path.name}.actions[{i}]", schema_version)
        for i, entry in enumerate(actions_raw)
    )

    return MappingDocument(
        schema_version=schema_version,
        scenario_id=scenario_id,
        actions=actions,
        workload_fingerprint=declared_workload,
        scenario_fingerprint=declared_scenario,
        # Verified only when both digests were present and matched. A partial
        # claim is not a verification.
        fingerprint_verified=declared_workload is not None and declared_scenario is not None,
        path=str(mapping_path),
    )


def _read_mapping_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidInput(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInput(f"{path} is not valid JSON: {exc}") from exc


def _optional_str(block: dict[str, Any], key: str, where: str) -> str | None:
    """Read an optional string field, rejecting an explicit ``null``-like blank."""
    if key not in block or block[key] is None:
        return None
    value = _require_str(block[key], f"{where}.{key}")
    if not value:
        raise InvalidInput(f"{where}.{key} must not be empty")
    return value


def _parse_action(raw: Any, where: str, schema_version: str = SCHEMA_VERSION) -> Action:
    """Turn one action object into an :class:`Action`.

    Only the presence of the right id field is checked here. Whether
    ``tensor_id`` names a real tensor, or whether the action is legal where it
    appears, are semantic questions for the evaluator.

    ``schema_version`` selects the spelling. It defaults to 0.1 so that every
    existing caller -- and the 0.1 mapping files on disk -- keeps parsing exactly
    as before; a 0.2 document must say ``COPY`` with its two endpoints, because
    ``COPY_H2D`` in 0.2 would be a mapping that forgot DRAM and VRAM are now two
    budgets rather than one.
    """
    block = _require_mapping(raw, where)
    m1 = schema_version == SCHEMA_VERSION_M1
    _reject_unknown_keys(
        block,
        _M1_ACTION_KEYS if m1 else {"kind", "tensor_id", "operation_id", "comment"},
        where,
    )

    kinds = _ACTION_KINDS_M1 if m1 else _ACTION_KINDS
    kind = _require_str(block.get("kind"), f"{where}.kind")
    if kind not in kinds:
        raise InvalidInput(f"{where}.kind must be one of {list(kinds)}, got {kind!r}")

    tensor_id = _optional_field(block, "tensor_id", where)
    operation_id = _optional_field(block, "operation_id", where)

    if m1:
        return _parse_action_m1(block, kind, tensor_id, operation_id, where)

    if kind in (ACTION_COPY_H2D, ACTION_EVICT):
        if tensor_id is None:
            raise InvalidInput(f"{where}: {kind} requires a tensor_id")
        if operation_id is not None:
            raise InvalidInput(f"{where}: {kind} does not take an operation_id")
        return Action(kind, tensor_id)

    if kind == ACTION_COMPUTE:
        if operation_id is None:
            raise InvalidInput(f"{where}: {ACTION_COMPUTE} requires an operation_id")
        if tensor_id is not None:
            raise InvalidInput(f"{where}: {ACTION_COMPUTE} does not take a tensor_id")
        return Action(kind, operation_id)

    if tensor_id is not None or operation_id is not None:
        raise InvalidInput(f"{where}: {ACTION_ADVANCE} takes neither id")
    return Action(ACTION_ADVANCE, None)


def _parse_action_m1(
    block: dict[str, Any],
    kind: str,
    tensor_id: str | None,
    operation_id: str | None,
    where: str,
) -> Action:
    """The 0.2 spelling.

    ``device``, ``src``/``dst`` and ``location`` are required wherever they are
    meaningful. None of them has a default: a document that omits the device is
    a document whose author has not decided, and guessing "gpu" for them is how a
    CPU plan would silently be scored as a GPU one.
    """
    device = _optional_field(block, "device", where)
    src = _optional_field(block, "src", where)
    dst = _optional_field(block, "dst", where)
    location = _optional_field(block, "location", where)

    if kind == ACTION_ADVANCE:
        if any(v is not None for v in (tensor_id, operation_id, device, src, dst, location)):
            raise InvalidInput(f"{where}: {ACTION_ADVANCE} takes no other field")
        return Action(ACTION_ADVANCE, None)

    if kind == ACTION_COMPUTE:
        if operation_id is None:
            raise InvalidInput(f"{where}: {ACTION_COMPUTE} requires an operation_id")
        if tensor_id is not None:
            raise InvalidInput(f"{where}: {ACTION_COMPUTE} does not take a tensor_id")
        if device is None:
            raise InvalidInput(
                f"{where}: 0.2 {ACTION_COMPUTE} requires a device; there is no default"
            )
        if device not in _DEVICES_ORDER:
            raise InvalidInput(
                f"{where}.device must be one of {list(_DEVICES_ORDER)}, got {device!r}"
            )
        if src is not None or dst is not None or location is not None:
            raise InvalidInput(f"{where}: {ACTION_COMPUTE} takes only operation_id and device")
        return Action(kind, operation_id, device=device)

    if tensor_id is None:
        raise InvalidInput(f"{where}: {kind} requires a tensor_id")
    if operation_id is not None:
        raise InvalidInput(f"{where}: {kind} does not take an operation_id")
    if device is not None:
        raise InvalidInput(f"{where}: {kind} does not take a device")

    if kind == ACTION_COPY:
        if src is None or dst is None:
            raise InvalidInput(
                f"{where}: 0.2 {ACTION_COPY} requires both src and dst; a route is a "
                "direction, not a default"
            )
        if location is not None:
            raise InvalidInput(f"{where}: {ACTION_COPY} does not take a location")
        if LOC_SSD in (src, dst):
            # Well-formed JSON asking for a move M1 does not model, rather than a
            # malformed action: the SSD is only reachable *through* DRAM, and the
            # two directions have names of their own (M1_DESIGN.md §5).
            raise Unsupported(
                f"{where}: {src!r}->{dst!r} involves the SSD, and M1 has no such copy; "
                f"SSD traffic goes through DRAM as {ACTION_READ_SSD} (ssd->dram) or "
                f"{ACTION_WRITE_SSD} (dram->ssd)"
            )
        if (src, dst) not in _COPY_ENDS_M1:
            raise InvalidInput(
                f"{where}: {src!r}->{dst!r} is not a route M1 models; the routes are "
                f"{[f'{s}->{d}' for s, d in _COPY_ENDS_M1]}"
            )
        return Action(kind, tensor_id, src=src, dst=dst)

    if kind in (ACTION_READ_SSD, ACTION_WRITE_SSD):
        if src is not None or dst is not None:
            raise InvalidInput(
                f"{where}: {kind} names only a tensor; its direction is fixed"
            )
        if location is not None:
            raise InvalidInput(f"{where}: {kind} does not take a location")
        return Action(kind, tensor_id)

    # EVICT
    if src is not None or dst is not None:
        raise InvalidInput(f"{where}: {ACTION_EVICT} does not take src/dst")
    if location is None:
        raise InvalidInput(
            f"{where}: 0.2 {ACTION_EVICT} requires a location; both domains have a "
            "budget, so which one is being freed is part of the action"
        )
    if location == LOC_SSD:
        # Same distinction as the COPY case: the document is well formed and asks
        # for something the design excludes, so it is `unsupported` rather than
        # `invalid_input` (M1_ACCEPTANCE.md §10).
        raise Unsupported(
            f"{where}: M1 does not model eviction from the SSD; only the two "
            "capacity-bounded domains can be evicted, and SSD content is dropped "
            "by simply not being written"
        )
    if location not in DOMAINS:
        raise InvalidInput(
            f"{where}.location must be one of {list(DOMAINS)}, got {location!r}"
        )
    return Action(kind, tensor_id, location=location)


def _optional_field(block: dict[str, Any], key: str, where: str) -> str | None:
    if key not in block or block[key] is None:
        return None
    value = _require_str(block[key], f"{where}.{key}")
    if not value:
        # An empty string is a present-but-meaningless id. Treating it as absent
        # would silently turn a typo into ADVANCE-shaped semantics.
        raise InvalidInput(f"{where}.{key} must not be empty")
    return value
