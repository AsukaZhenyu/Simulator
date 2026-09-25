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
    LOC_DRAM,
    LOC_VRAM,
    OP_DONE,
    OP_NOT_STARTED,
    OP_RUNNING,
    RESOURCE_COMPUTE,
    RESOURCE_COPY,
    InitialCapacityExceeded,
    InvalidInput,
    Scenario,
    check_initial_capacity,
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
    "ACTION_COPY_H2D",
    "ACTION_EVICT",
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
    "used_vram_bytes",
    "evaluate_mapping",
]

ACTION_COPY_H2D = "COPY_H2D"
ACTION_COMPUTE = "COMPUTE"
ACTION_EVICT = "EVICT"
ACTION_ADVANCE = "ADVANCE"

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

    ``target_id`` is a tensor id for COPY_H2D/EVICT, an operation id for
    COMPUTE, and ``None`` for ADVANCE.
    """

    kind: str
    target_id: str | None = None

    def __str__(self) -> str:
        return self.kind if self.target_id is None else f"{self.kind}({self.target_id})"


@dataclass(frozen=True, order=True)
class RunningTask:
    """An in-flight copy or compute, identified by what it acts on.

    Only ``remaining_ns`` varies over the task's life; the total duration is
    looked up in the cost model when it is needed, so it is not duplicated here.
    """

    resource: str
    target_id: str
    remaining_ns: int


@dataclass(frozen=True)
class State:
    """An immutable model state.

    ``op_status`` and ``copy_status`` are sorted ``(id, status)`` tuples so that
    two equal states are equal by construction, regardless of the order in which
    they were built.
    """

    t: int
    op_status: tuple[tuple[str, str], ...]
    copy_status: tuple[tuple[str, str], ...]
    running: tuple[RunningTask, ...]

    @cached_property
    def _ops(self) -> dict[str, str]:
        return dict(self.op_status)

    @cached_property
    def _copies(self) -> dict[str, str]:
        return dict(self.copy_status)

    def op(self, operation_id: str) -> str:
        """Status of ``operation_id``, or ``NOT_STARTED`` if unknown to the graph."""
        return self._ops.get(operation_id, OP_NOT_STARTED)

    def copy(self, tensor_id: str) -> str:
        """GPU-copy status of ``tensor_id``, or ``ABSENT`` if unknown."""
        return self._copies.get(tensor_id, COPY_ABSENT)

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
        """
        return (self.op_status, self.copy_status, self.running)


@dataclass(frozen=True)
class StateSnapshot:
    """A checkable summary of a state, for artefacts and assertions."""

    t: int
    used_vram_bytes: int
    op_status: tuple[tuple[str, str], ...]
    copy_status: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Event:
    """One applied action, as recorded in the event log.

    Transfer/release amounts are kept in separate fields on purpose. A copy
    moves ``storage_bytes``; an eviction releases the *aligned* allocation. Those
    differ whenever ``allocation_alignment_bytes`` is greater than one, and
    conflating them is the mistake ``ACCEPTANCE.md`` §5 calls out.
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


# ---------------------------------------------------------------------------
# Derived quantities -- the single memory ledger
# ---------------------------------------------------------------------------


def used_vram_bytes(scenario: Scenario, state: State) -> int:
    """VRAM in use: the runtime reserve plus every allocation the state implies.

    Allocations are (a) any tensor whose GPU copy is not ``ABSENT`` -- a
    reservation counts, because the space is committed the moment the copy or
    the compute starts -- and (b) the workspace of every running compute.
    """
    used = scenario.architecture.runtime_reserved_bytes
    for tensor_id, status in state.copy_status:
        if status != COPY_ABSENT:
            used += scenario.size_alloc(tensor_id)
    for task in state.running:
        if task.resource == RESOURCE_COMPUTE:
            used += scenario.workspace_alloc(task.target_id)
    return used


def _read_locks(scenario: Scenario, state: State) -> dict[str, int]:
    """Tensors currently locked by a running compute, with reference counts.

    A lock exists exactly while a compute that reads the tensor is in flight;
    it is derived from ``state.running`` rather than stored, so it cannot drift
    away from the task list. The count matters for ``add(x, x)``: one
    allocation, but the same operand named twice.
    """
    locks: dict[str, int] = {}
    for task in state.running:
        if task.resource != RESOURCE_COMPUTE:
            continue
        operation = scenario.workload.operation_by_id[task.target_id]
        for tensor_id in set(operation.inputs):
            locks[tensor_id] = locks.get(tensor_id, 0) + 1
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
    return State(t=0, op_status=op_status, copy_status=copy_status, running=())


def is_goal(scenario: Scenario, state: State) -> bool:
    """All work done, nothing in flight, every requested output resident."""
    if state.running:
        return False
    for operation in scenario.workload.operations:
        if state.op(operation.id) != OP_DONE:
            return False
    for tensor_id in scenario.workload.outputs:
        if state.copy(tensor_id) != COPY_READY:
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
    # as data loss would point at the wrong problem.
    if _read_locks(scenario, state).get(tensor_id):
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
# Transition
# ---------------------------------------------------------------------------


def transition(scenario: Scenario, state: State, action: Action) -> State:
    """Apply ``action``, returning the successor state.

    COPY_H2D, COMPUTE and EVICT are instantaneous commitments: they start a task
    or drop a copy but never move the clock. Only ADVANCE moves time, by the
    smallest remaining duration, completing everything that ends at that instant
    as one batch (``DESIGN.md`` §6).
    """
    _check(scenario, state, action)

    if action.kind == ACTION_ADVANCE:
        return _advance(scenario, state)
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
    still_running: list[RunningTask] = []
    for task in state.running:
        remaining = task.remaining_ns - delta
        if remaining > 0:
            still_running.append(
                RunningTask(
                    resource=task.resource, target_id=task.target_id, remaining_ns=remaining
                )
            )
            continue
        if task.resource == RESOURCE_COPY:
            copies[task.target_id] = COPY_READY
        else:
            copies[scenario.workload.operation_by_id[task.target_id].output] = COPY_READY
            ops[task.target_id] = OP_DONE

    return State(
        t=state.t + delta,
        op_status=tuple(sorted(ops.items())),
        copy_status=tuple(sorted(copies.items())),
        running=tuple(sorted(still_running)),
    )


def _with_task(running: Sequence[RunningTask], task: RunningTask) -> tuple[RunningTask, ...]:
    return tuple(sorted((*running, task)))


def _rebuild(
    state: State,
    *,
    ops: dict[str, str] | None = None,
    copies: dict[str, str] | None = None,
    running: Sequence[RunningTask] | None = None,
) -> State:
    return State(
        t=state.t,
        op_status=tuple(sorted(ops.items())) if ops is not None else state.op_status,
        copy_status=tuple(sorted(copies.items())) if copies is not None else state.copy_status,
        running=tuple(sorted(running)) if running is not None else state.running,
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
    candidates: list[Action] = []
    for tensor_id in sorted(scenario.mapspace.copy_tensor_ids):
        candidates.append(Action(ACTION_COPY_H2D, tensor_id))
    for operation_id in sorted(scenario.workload.operation_by_id):
        candidates.append(Action(ACTION_COMPUTE, operation_id))
    for tensor_id in sorted(scenario.workload.tensor_by_id):
        candidates.append(Action(ACTION_EVICT, tensor_id))
    candidates.append(Action(ACTION_ADVANCE))

    legal: list[Action] = []
    for action in candidates:
        try:
            _check(scenario, state, action)
        except MappingError:
            continue
        legal.append(action)
    return tuple(legal)


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
    h2d_bytes = 0
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
            )

        event = _event_for(scenario, state, index, action)
        events.append(event)
        h2d_bytes += event.bytes_transferred
        peak = max(peak, used_vram_bytes(scenario, state))
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
    )


def _event_for(scenario: Scenario, state: State, index: int, action: Action) -> Event:
    """Describe an applied action.

    ``start_ns`` and ``end_ns`` are both recorded. A copy begins at the current
    time and finishes ``duration_ns`` later; writing those as two fields removes
    any chance of reading a start time as a completion time, which is the
    confusion ``ACCEPTANCE.md`` §6 warns about.
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


def _snapshot(scenario: Scenario, state: State, action_index: int) -> StateSnapshot:
    return StateSnapshot(
        t=state.t,
        used_vram_bytes=used_vram_bytes(scenario, state),
        op_status=state.op_status,
        copy_status=state.copy_status,
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
    )


# ---------------------------------------------------------------------------
# Timing helper for budgeted search
# ---------------------------------------------------------------------------


class WallClock:
    """Monotonic wall-clock budget.

    Kept here rather than in the mapper so tests can substitute a deterministic
    clock. ``time.monotonic`` because the search budget must not be affected by
    system clock adjustments.
    """

    def __init__(self, limit_s: float) -> None:
        self.limit_s = limit_s
        self._start = time.monotonic()

    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

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
        _parse_action(entry, f"{mapping_path.name}.actions[{i}]")
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


def _parse_action(raw: Any, where: str) -> Action:
    """Turn one action object into an :class:`Action`.

    Only the presence of the right id field is checked here. Whether
    ``tensor_id`` names a real tensor, or whether the action is legal where it
    appears, are semantic questions for the evaluator.
    """
    block = _require_mapping(raw, where)
    _reject_unknown_keys(block, {"kind", "tensor_id", "operation_id", "comment"}, where)

    kind = _require_str(block.get("kind"), f"{where}.kind")
    if kind not in _ACTION_KINDS:
        raise InvalidInput(f"{where}.kind must be one of {list(_ACTION_KINDS)}, got {kind!r}")

    tensor_id = _optional_field(block, "tensor_id", where)
    operation_id = _optional_field(block, "operation_id", where)

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


def _optional_field(block: dict[str, Any], key: str, where: str) -> str | None:
    if key not in block or block[key] is None:
        return None
    value = _require_str(block[key], f"{where}.{key}")
    if not value:
        # An empty string is a present-but-meaningless id. Treating it as absent
        # would silently turn a typo into ADVANCE-shaped semantics.
        raise InvalidInput(f"{where}.{key} must not be empty")
    return value
