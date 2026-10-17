"""Causal execution sessions with explicit, private environment timelines.

External memory is an absolute, non-evictable allocation in addition to the
scenario's runtime reserve. CPU availability is a service blackout: an in-flight
CPU task pauses without losing its input locks, output reservation or workspace.
It is not a CPU-utilization-to-latency formula or an operating-system model.
Costs and workspace sizes remain fixed throughout a session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Protocol

from .engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_EVICT, Action, CODE_CAPACITY_EXCEEDED,
    CODE_ILLEGAL_ACTION, CODE_RESOURCE_BUSY, EvaluationResult, Event, MappingError,
    REASON_ACTIONS_REMAIN_AFTER_GOAL, REASON_GOAL_REACHED, REASON_UNFINISHED,
    STATUS_INCOMPLETE_MAPPING, STATUS_INVALID_MAPPING, STATUS_VALID, State,
    StateSnapshot, _accumulate_transfer_bytes, _event_for, _snapshot,
    advance_elapsed, initial_state, is_goal, transition, used_bytes,
)
from .spec import (
    COPY_ABSENT, COPY_READY, DOMAINS, LOC_DRAM, LOC_SSD, LOC_VRAM, OP_DONE,
    RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE, ROLE_WEIGHT, Scenario, InvalidInput,
)

CODE_EXTERNAL_MEMORY_CONFLICT = "external_memory_conflict"
CODE_ENVIRONMENT_DEADLOCK = "environment_deadlock"
CODE_INFORMATION_NOT_KNOWN = "information_not_known"
CODE_RECLAIM_ADMISSION = "reclaim_admission_blocked"
CODE_CONTROL_BUSY = "control_busy"


@dataclass(frozen=True)
class EnvironmentReclaimRequest:
    id: str
    t_notice_ns: int
    target_external_bytes: int
    deadline_ns: int
    domain: str = LOC_VRAM

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidInput("reclaim id must be a nonempty string")
        for field in ("t_notice_ns", "target_external_bytes", "deadline_ns"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise InvalidInput(f"reclaim.{field} must be a non-negative integer")
        if self.domain != LOC_VRAM:
            raise InvalidInput("reclaim supports only vram")
        if self.deadline_ns < self.t_notice_ns:
            raise InvalidInput("reclaim deadline precedes notice")


@dataclass(frozen=True)
class ControlCosts:
    ack_ns: int
    decline_ns: int
    decision_ns: int
    source: str
    timing_scope: str = "dedicated_serial_control_resource"

    def __post_init__(self):
        for field in ("ack_ns", "decline_ns", "decision_ns"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise InvalidInput(f"control_costs.{field} must be an explicit non-negative integer")
        if not isinstance(self.source, str) or not self.source.strip():
            raise InvalidInput("control_costs.source must be explicit")
        if self.timing_scope != "dedicated_serial_control_resource":
            raise InvalidInput("only dedicated_serial_control_resource is modeled")


@dataclass(frozen=True)
class ReclaimObservation:
    id: str
    domain: str
    target_external_bytes: int
    deadline_ns: int
    status: str
    t_notice_ns: int


@dataclass(frozen=True)
class ReclaimRecord:
    id: str
    domain: str
    target_external_bytes: int
    t_notice_ns: int
    deadline_ns: int
    status: str
    responded_ns: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ControlRunningObservation:
    kind: str
    request_id: str | None
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class ControlEvent:
    index: int
    kind: str
    request_id: str | None
    start_ns: int
    end_ns: int | None
    cost_ns: int
    status: str
    reason: str | None = None
    source_index: int | None = None


class Visibility(Protocol):
    """Privileged adapter interface; policy callbacks receive only observations."""

    def visible_operation_ids(self, state: State) -> tuple[str, ...]: ...
    def visible_tensor_ids(self, state: State) -> tuple[str, ...]: ...
    def allow_action(self, action: Action, state: State) -> bool: ...


@dataclass(frozen=True)
class EnvironmentEvent:
    t_ns: int
    dram_external_bytes: int | None = None
    vram_external_bytes: int | None = None
    cpu_available: bool | None = None

    def __post_init__(self) -> None:
        for name in ("t_ns", "dram_external_bytes", "vram_external_bytes"):
            value = getattr(self, name)
            if value is None and name != "t_ns":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidInput(f"environment.{name} must be a non-negative integer")
        if self.cpu_available is not None and not isinstance(self.cpu_available, bool):
            raise InvalidInput("environment.cpu_available must be a boolean")
        if all(getattr(self, key) is None for key in
               ("dram_external_bytes", "vram_external_bytes", "cpu_available")):
            raise InvalidInput("environment event must change at least one field")


@dataclass(frozen=True)
class MemoryUsage:
    domain: str
    capacity_bytes: int | None
    runtime_reserved_bytes: int
    external_bytes: int
    tensor_bytes: int
    workspace_bytes: int
    used_bytes: int

    @property
    def available_bytes(self) -> int | None:
        return None if self.capacity_bytes is None else self.capacity_bytes - self.used_bytes


@dataclass(frozen=True)
class RunningObservation:
    resource: str
    target_id: str
    device: str | None
    route: str | None


@dataclass(frozen=True)
class Observation:
    t_ns: int
    op_status: tuple[tuple[str, str], ...]
    copy_status: tuple[tuple[str, str], ...]
    dram_status: tuple[tuple[str, str], ...]
    ssd_status: tuple[tuple[str, str], ...]
    running: tuple[RunningObservation, ...]
    memory: tuple[MemoryUsage, ...]
    cpu_available: bool
    finished: bool
    error_code: str | None
    pending_reclaim: ReclaimObservation | None = None
    safe_weight_release_candidates: tuple[str, ...] = ()
    reclaim_records: tuple[ReclaimRecord, ...] = ()
    control_running: ControlRunningObservation | None = None
    control_costs: ControlCosts | None = None


@dataclass(frozen=True)
class EnvironmentRecord:
    t_ns: int
    dram_external_bytes: int
    vram_external_bytes: int
    cpu_available: bool
    memory: tuple[MemoryUsage, ...]
    accepted: bool


@dataclass(frozen=True)
class ServiceInterval:
    action_index: int
    resource: str
    target_id: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class SessionStep:
    ok: bool
    t_ns: int
    action: Action | None = None
    completed_operation_ids: tuple[str, ...] = ()
    completed_copies: tuple[tuple[str, str], ...] = ()
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class SessionResult:
    evaluation: EvaluationResult
    final_state: State
    environment_events: tuple[EnvironmentRecord, ...]
    service_intervals: tuple[ServiceInterval, ...]
    nominal_ends: tuple[int | None, ...]
    action_sources: tuple[int | None, ...]
    reclaim_records: tuple[ReclaimRecord, ...] = ()
    control_events: tuple[ControlEvent, ...] = ()


class SimulationSession:
    """Trusted modeling driver; mapping policies must receive ``observe()`` only.

    ``state`` and ``scenario`` are privileged read-only adapter properties. The
    future environment queue is deliberately absent from every observation.
    All same-time completions occur before that time's atomic environment group.
    An overcommitted external allocation fails explicitly without automatic
    eviction. Rejected allocations are recorded but never added to the ledger.
    """

    def __init__(self, scenario: Scenario, *, environment_events: Iterable[EnvironmentEvent] = (),
                 visibility: Visibility | None = None,
                 reclaim_requests: Iterable[EnvironmentReclaimRequest] = (),
                 control_costs: ControlCosts | None = None) -> None:
        if not isinstance(scenario, Scenario):
            raise InvalidInput("SimulationSession requires a Scenario")
        # Direct dataclass construction must obey the same complete contract
        # as a loaded JSON scenario, including exact costs and boolean checks.
        from .program import validate_device_state
        validate_device_state(scenario, scenario)
        queue = tuple(environment_events)
        if any(not isinstance(event, EnvironmentEvent) for event in queue):
            raise InvalidInput("environment_events must contain EnvironmentEvent objects")
        if any(a.t_ns > b.t_ns for a, b in zip(queue, queue[1:])):
            raise InvalidInput("environment events must be ordered by t_ns")
        group_fields: set[str] = set()
        group_time = None
        for event in queue:
            if event.t_ns != group_time:
                group_time, group_fields = event.t_ns, set()
            for field in ("dram_external_bytes", "vram_external_bytes", "cpu_available"):
                if getattr(event, field) is not None:
                    if field in group_fields:
                        raise InvalidInput(f"same-time environment group repeats {field}")
                    group_fields.add(field)
        requests = tuple(reclaim_requests)
        if any(not isinstance(request, EnvironmentReclaimRequest) for request in requests):
            raise InvalidInput("reclaim_requests must contain EnvironmentReclaimRequest objects")
        if len({request.id for request in requests}) != len(requests):
            raise InvalidInput("reclaim request ids must be unique")
        if any(a.deadline_ns >= b.t_notice_ns for a, b in zip(requests, requests[1:])):
            raise InvalidInput("reclaim windows must be strictly ordered and non-overlapping")
        if control_costs is not None and not isinstance(control_costs, ControlCosts):
            raise InvalidInput("control_costs requires ControlCosts")
        if requests and control_costs is None:
            raise InvalidInput("reclaim requires explicit control_costs, including a source")
        self._reclaim_queue = requests
        self._reclaim_queue_index = 0
        self._reclaim_records: list[ReclaimRecord] = []
        self._pending_reclaim: EnvironmentReclaimRequest | None = None
        self._control_costs = control_costs
        self._control_running: ControlRunningObservation | None = None
        self._control_command = None
        self._control_events: list[ControlEvent] = []
        self._scenario = scenario
        self._state = initial_state(scenario)
        self._queue = queue
        self._queue_index = 0
        self._visibility = visibility
        self._dram_external = self._vram_external = 0
        self._cpu_available = True
        self._error_code: str | None = None
        self._error_message: str | None = None
        self._error_t: int | None = None
        self._error_action_index: int | None = None
        self._actions: list[Action] = []
        self._events: list[Event] = []
        self._nominal_ends: list[int | None] = []
        self._sources: list[int | None] = []
        self._environment_records: list[EnvironmentRecord] = []
        self._service: list[ServiceInterval] = []
        self._launches: dict[tuple[str, str], int] = {}
        self._amounts = (0, 0, 0, 0)
        self._peak_dram = self._peak_vram = 0
        self._apply_environment()
        self._apply_reclaim_notices()
        self._update_peaks()
        self._snapshots: list[StateSnapshot] = [self._snapshot(0)]

    @property
    def state(self) -> State:
        return self._state

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def error_code(self) -> str | None:
        return self._error_code

    @property
    def finished(self) -> bool:
        return (not self._error_code and is_goal(self._scenario, self._state)
                and self._pending_reclaim is None and self._control_running is None)

    def _memory(self, state: State | None = None, *, dram: int | None = None,
                vram: int | None = None) -> tuple[MemoryUsage, ...]:
        state = self._state if state is None else state
        externals = {LOC_DRAM: self._dram_external if dram is None else dram,
                     LOC_VRAM: self._vram_external if vram is None else vram}
        result = []
        for domain in DOMAINS:
            reserve = self._scenario.architecture.reserved_of(domain)
            tensors = sum(self._scenario.size_alloc(tid, domain)
                          for tid, status in state.statuses(domain) if status != COPY_ABSENT)
            baseline = used_bytes(self._scenario, state, domain)
            result.append(MemoryUsage(domain, self._scenario.architecture.capacity_of(domain),
                                      reserve, externals[domain], tensors,
                                      baseline - reserve - tensors, baseline + externals[domain]))
        return tuple(result)

    def _capacity_message(self, memory: tuple[MemoryUsage, ...]) -> str | None:
        for usage in memory:
            if usage.capacity_bytes is not None and usage.used_bytes > usage.capacity_bytes:
                return (f"{usage.domain} needs {usage.used_bytes} bytes: runtime reserve "
                        f"{usage.runtime_reserved_bytes}, external {usage.external_bytes}, "
                        f"tensors {usage.tensor_bytes}, workspace {usage.workspace_bytes}; "
                        f"capacity is {usage.capacity_bytes}")
        return None

    def _fail(self, code: str, message: str, *, action_index: int | None = None) -> SessionStep:
        if self._error_code is None:
            self._error_code, self._error_message = code, message
            self._error_t = self._state.t
            # Environment failure at t=0 has no triggering action. Later
            # failures identify the attempted action, including the ADVANCE
            # whose elapsed time exposed an external allocation conflict.
            self._error_action_index = action_index
        return SessionStep(False, self._state.t, error_code=self._error_code,
                           error_message=self._error_message)

    def _apply_environment(self, *, action_index: int | None = None) -> None:
        if self._queue_index >= len(self._queue) or self._queue[self._queue_index].t_ns != self._state.t:
            return
        dram, vram, cpu = self._dram_external, self._vram_external, self._cpu_available
        while self._queue_index < len(self._queue) and self._queue[self._queue_index].t_ns == self._state.t:
            event = self._queue[self._queue_index]
            if event.dram_external_bytes is not None:
                dram = event.dram_external_bytes
            if event.vram_external_bytes is not None:
                vram = event.vram_external_bytes
            if event.cpu_available is not None:
                cpu = event.cpu_available
            self._queue_index += 1
        memory = self._memory(dram=dram, vram=vram)
        conflict = self._capacity_message(memory)
        self._environment_records.append(EnvironmentRecord(self._state.t, dram, vram, cpu,
                                                           memory, conflict is None))
        if conflict:
            self._fail(CODE_EXTERNAL_MEMORY_CONFLICT,
                       "under this environment trace, external allocation was rejected: " + conflict,
                       action_index=action_index)
        else:
            self._dram_external, self._vram_external, self._cpu_available = dram, vram, cpu

    def _apply_reclaim_notices(self) -> None:
        if self._error_code:
            return
        if (self._reclaim_queue_index >= len(self._reclaim_queue) or
                self._reclaim_queue[self._reclaim_queue_index].t_notice_ns != self._state.t):
            return
        request = self._reclaim_queue[self._reclaim_queue_index]
        self._reclaim_queue_index += 1
        invalid = request.target_external_bytes <= self._vram_external
        self._reclaim_records.append(ReclaimRecord(request.id, request.domain,
            request.target_external_bytes, request.t_notice_ns, request.deadline_ns,
            "invalid_target" if invalid else "pending", self._state.t if invalid else None,
            "target must exceed actual external allocation at notice" if invalid else None))
        if not invalid:
            self._pending_reclaim = request

    def _resolve_reclaim(self, status: str, reason: str | None = None) -> None:
        request = self._pending_reclaim
        if request is None:
            return
        self._reclaim_records[-1] = replace(self._reclaim_records[-1], status=status,
                                           responded_ns=self._state.t, reason=reason)
        self._pending_reclaim = None

    def _seal_deadline(self) -> None:
        if self._pending_reclaim and self._state.t >= self._pending_reclaim.deadline_ns:
            self._resolve_reclaim("deadline_rejected", "no completed response within deadline")

    def _extra_boundary_times(self) -> list[int]:
        times = []
        if self._reclaim_queue_index < len(self._reclaim_queue):
            times.append(self._reclaim_queue[self._reclaim_queue_index].t_notice_ns)
        if self._pending_reclaim and self._pending_reclaim.deadline_ns > self._state.t:
            times.append(self._pending_reclaim.deadline_ns)
        if self._control_running:
            times.append(self._control_running.end_ns)
        return times

    def _safe_weight_release(self, tensor_id: str) -> bool:
        tensor = self._scenario.workload.tensor_by_id.get(tensor_id)
        if not self._scenario.is_m1 or tensor is None or tensor.role != ROLE_WEIGHT:
            return False
        # Weights are immutable leaves. The identical tensor ID is therefore
        # the same version; mutable KV/outputs are never admitted by this rule.
        if (self._state.copy_at(tensor_id, LOC_VRAM) != COPY_READY or
                self._state.copy_at(tensor_id, LOC_DRAM) != COPY_READY):
            return False
        try:
            transition(self._scenario, self._state, Action(ACTION_EVICT, tensor_id, location=LOC_VRAM))
        except MappingError:
            return False
        return True

    def _admission_message(self, action: Action, candidate: State) -> str | None:
        if self._control_running and self._control_running.kind != "WAIT_UNTIL":
            return "serial controller is busy; advance until its operation completes"
        if not self._pending_reclaim:
            return None
        if action.kind == ACTION_EVICT and action.location == LOC_VRAM:
            if not self._safe_weight_release(action.target_id):
                return "pending reclaim permits only safe clean DRAM-backed weight eviction"
        before = dict(self._state.statuses(LOC_VRAM))
        if any(status != COPY_ABSENT and before.get(tid, COPY_ABSENT) == COPY_ABSENT
               for tid, status in candidate.statuses(LOC_VRAM)):
            return "pending reclaim blocks new VRAM tensor/output reservations"
        # Even a preallocated output cannot hide a newly reserved workspace.
        prior_workspace = next(m.workspace_bytes for m in self._memory() if m.domain == LOC_VRAM)
        new_workspace = next(m.workspace_bytes for m in self._memory(candidate) if m.domain == LOC_VRAM)
        if new_workspace > prior_workspace:
            return "pending reclaim blocks new GPU workspace reservations"
        return None

    def _finish_control(self, *, interrupted: bool = False) -> None:
        running = self._control_running
        if running is None or (not interrupted and running.end_ns != self._state.t):
            return
        command = self._control_command
        status, reason = "completed", command.reason
        if command.kind in ("ACK_RECLAIM", "DECLINE_RECLAIM"):
            request = self._pending_reclaim
            if request is None or request.id != command.request_id:
                status, reason = "rejected", "request is no longer pending"
            elif self._state.t > request.deadline_ns:
                status, reason = "rejected", "response completed after deadline"
                self._resolve_reclaim("deadline_rejected", reason)
            elif command.kind == "DECLINE_RECLAIM":
                self._resolve_reclaim("declined", command.reason)
            else:
                conflict = self._capacity_message(self._memory(vram=request.target_external_bytes))
                if conflict:
                    status, reason = "rejected", conflict
                    self._resolve_reclaim("capacity_rejected", conflict)
                else:
                    self._vram_external = request.target_external_bytes
                    self._resolve_reclaim("accepted")
        if interrupted and command.kind == "WAIT_UNTIL":
            status, reason = "interrupted", "earlier runtime boundary made observation available"
        self._control_events[-1] = replace(self._control_events[-1], end_ns=self._state.t,
            cost_ns=self._state.t - running.start_ns, status=status, reason=reason)
        self._control_running = None
        self._control_command = None

    def _after_time_boundary(self, *, action_index: int) -> None:
        # Existing tensor completions have already run. A completed ACK is
        # charged and committed before same-time actual environment updates.
        self._finish_control()
        self._apply_environment(action_index=action_index)
        self._apply_reclaim_notices()
        if self._control_running and self._control_running.kind == "WAIT_UNTIL":
            self._finish_control(interrupted=True)
        # Equality deliberately preserves the final response window. Sealing
        # occurs on the next ADVANCE before clock progression.
        if self._pending_reclaim and self._state.t > self._pending_reclaim.deadline_ns:
            self._seal_deadline()

    def submit_control(self, command, source_index: int | None = None) -> SessionStep:
        from .runtime import ControlCommand
        if not isinstance(command, ControlCommand):
            raise InvalidInput("submit_control requires ControlCommand")
        self._validate_source(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        if self._control_costs is None:
            raise InvalidInput("control commands require explicit ControlCosts")
        if self._control_running:
            return SessionStep(False, self._state.t, error_code=CODE_CONTROL_BUSY,
                               error_message="serial controller is busy")
        record = next((record for record in self._reclaim_records if record.id == command.request_id), None)
        reason = None
        if command.request_id is not None and record is None:
            reason = "request is not publicly known"
        elif command.kind in ("ACK_RECLAIM", "DECLINE_RECLAIM") and (
                self._pending_reclaim is None or self._pending_reclaim.id != command.request_id):
            reason = "request is not pending"
        elif command.kind == "WAIT_UNTIL" and command.until_ns <= self._state.t:
            reason = "WAIT_UNTIL must be strictly after current time"
        elif self.finished:
            reason = "control follows completed goal"
        if reason:
            self._control_events.append(ControlEvent(len(self._control_events), command.kind,
                command.request_id, self._state.t, self._state.t, 0, "rejected", reason, source_index))
            return SessionStep(False, self._state.t, error_code=CODE_ILLEGAL_ACTION, error_message=reason)
        cost = ({"ACK_RECLAIM": self._control_costs.ack_ns,
                 "DECLINE_RECLAIM": self._control_costs.decline_ns,
                 "DECIDE": self._control_costs.decision_ns}.get(command.kind)
                if command.kind != "WAIT_UNTIL" else command.until_ns - self._state.t)
        self._control_running = ControlRunningObservation(command.kind, command.request_id,
                                                          self._state.t, self._state.t + cost)
        self._control_command = command
        self._control_events.append(ControlEvent(len(self._control_events), command.kind,
            command.request_id, self._state.t, None, cost, "running", command.reason, source_index))
        if cost == 0:
            self._finish_control()
        self._update_peaks()
        return SessionStep(True, self._state.t)

    def _update_peaks(self) -> None:
        for usage in self._memory():
            if usage.domain == LOC_DRAM:
                self._peak_dram = max(self._peak_dram, usage.used_bytes)
            else:
                self._peak_vram = max(self._peak_vram, usage.used_bytes)

    def _snapshot(self, index: int) -> StateSnapshot:
        raw = _snapshot(self._scenario, self._state, index)
        return replace(raw, used_dram_bytes=raw.used_dram_bytes + self._dram_external,
                       used_vram_bytes=raw.used_vram_bytes + self._vram_external)

    def observe(self) -> Observation:
        operations = set(self._visibility.visible_operation_ids(self._state)) if self._visibility else set(
            self._scenario.workload.operation_by_id)
        tensors = set(self._visibility.visible_tensor_ids(self._state)) if self._visibility else set(
            self._scenario.workload.tensor_by_id)
        ledger = lambda rows: tuple((tid, status) for tid, status in rows if tid in tensors)
        return Observation(
            self._state.t, tuple((oid, status) for oid, status in self._state.op_status if oid in operations),
            ledger(self._state.copy_status), ledger(self._state.dram_status), ledger(self._state.ssd_status),
            tuple(RunningObservation(task.resource, task.target_id, task.device, task.route)
                  for task in self._state.running
                  if task.target_id in (operations if task.resource in
                      (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE) else tensors)),
            self._memory(), self._cpu_available, self.finished, self._error_code,
            None if self._pending_reclaim is None else ReclaimObservation(
                self._pending_reclaim.id, self._pending_reclaim.domain,
                self._pending_reclaim.target_external_bytes, self._pending_reclaim.deadline_ns,
                "pending", self._pending_reclaim.t_notice_ns),
            tuple(sorted(tid for tid in tensors if self._safe_weight_release(tid)))
            if self._control_costs is not None else (),
            tuple(self._reclaim_records), self._control_running, self._control_costs,
        )

    def submit(self, action: Action, source_index: int | None = None) -> SessionStep:
        if not isinstance(action, Action):
            raise InvalidInput("session.submit requires an Action")
        self._validate_source(source_index)
        if action.kind == ACTION_ADVANCE:
            return self.advance(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        action_index = len(self._actions)
        self._actions.append(action)
        self._sources.append(source_index)
        if self.finished:
            return self._fail(REASON_ACTIONS_REMAIN_AFTER_GOAL, "actions follow the completed goal",
                              action_index=action_index)
        if self._visibility and not self._visibility.allow_action(action, self._state):
            return self._fail(CODE_INFORMATION_NOT_KNOWN, "action references information not yet revealed",
                              action_index=action_index)
        if action.kind == ACTION_COMPUTE and action.device == "cpu" and not self._cpu_available:
            return self._fail(CODE_RESOURCE_BUSY, "CPU service is currently unavailable",
                              action_index=action_index)
        try:
            candidate = transition(self._scenario, self._state, action)
        except MappingError as error:
            return self._fail(error.code, str(error), action_index=action_index)
        admission = self._admission_message(action, candidate)
        if admission:
            return self._fail(CODE_RECLAIM_ADMISSION if self._pending_reclaim else CODE_CONTROL_BUSY,
                              admission, action_index=action_index)
        conflict = self._capacity_message(self._memory(candidate))
        if conflict:
            return self._fail(CODE_CAPACITY_EXCEEDED, conflict, action_index=action_index)
        self._state = candidate
        event = _event_for(self._scenario, self._state, len(self._events), action)
        self._nominal_ends.append(event.end_ns)
        if event.resource is not None and action.kind != "EVICT":
            self._launches[(event.resource, event.target_id)] = len(self._events)
            event = replace(event, end_ns=None)
        self._events.append(event)
        self._amounts = _accumulate_transfer_bytes(event, *self._amounts)
        self._update_peaks()
        self._snapshots.append(self._snapshot(len(self._events)))
        return SessionStep(True, self._state.t, action=action)

    def advance(self, source_index: int | None = None) -> SessionStep:
        self._validate_source(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        was_finished = self.finished
        self._seal_deadline()
        if not was_finished and self.finished:
            return SessionStep(True, self._state.t)
        action = Action(ACTION_ADVANCE)
        action_index = len(self._actions)
        self._actions.append(action)
        self._sources.append(source_index)
        if self.finished:
            return self._fail(REASON_ACTIONS_REMAIN_AFTER_GOAL, "advance follows the completed goal",
                              action_index=action_index)
        paused = () if self._cpu_available else (RESOURCE_CPU_COMPUTE,)
        active = [task for task in self._state.running if task.resource not in paused]
        times = [self._state.t + task.remaining_ns for task in active]
        times.extend(self._extra_boundary_times())
        if self._queue_index < len(self._queue):
            times.append(self._queue[self._queue_index].t_ns)
        if not times:
            if self._state.running:
                return self._fail(CODE_ENVIRONMENT_DEADLOCK,
                                  "all running tasks are paused and no future environment event can resume them",
                                  action_index=action_index)
            return self._fail(CODE_ILLEGAL_ACTION, "ADVANCE has no running task or future environment event",
                              action_index=action_index)
        destination = min(times)
        before = self._state
        for task in active:
            self._service.append(ServiceInterval(self._launches[(task.resource, task.target_id)],
                                                 task.resource, task.target_id, before.t, destination))
        self._state = advance_elapsed(self._scenario, before, destination - before.t, paused)
        after_keys = {(task.resource, task.target_id) for task in self._state.running}
        for task in before.running:
            key = (task.resource, task.target_id)
            if key not in after_keys:
                index = self._launches.pop(key)
                self._events[index] = replace(self._events[index], end_ns=destination)
        completed_ops = tuple(oid for oid, status in self._state.op_status
                              if status == OP_DONE and before.op(oid) != OP_DONE)
        completed_copies = tuple((tid, location) for location in (LOC_DRAM, LOC_VRAM, LOC_SSD)
                                 for tid, status in self._state.statuses(location)
                                 if status == COPY_READY and before.copy_at(tid, location) != COPY_READY)
        event = _event_for(self._scenario, self._state, len(self._events), action)
        self._events.append(event)
        self._nominal_ends.append(event.end_ns)
        self._after_time_boundary(action_index=action_index)
        self._update_peaks()
        self._snapshots.append(self._snapshot(len(self._events)))
        return SessionStep(not self._error_code, self._state.t, action, completed_ops,
                           completed_copies, self._error_code, self._error_message)

    def result(self) -> SessionResult:
        if self._error_code:
            status, reason = STATUS_INVALID_MAPPING, f"{self._error_code}: {self._error_message}"
        elif self.finished:
            status, reason = STATUS_VALID, REASON_GOAL_REACHED
        else:
            status, reason = STATUS_INCOMPLETE_MAPPING, REASON_UNFINISHED
        evaluation = EvaluationResult(
            status=status, reason=reason, makespan_ns=self._state.t if self.finished else None,
            peak_vram_bytes=self._peak_vram, h2d_bytes=self._amounts[0], action_count=len(self._events),
            actions=tuple(self._actions), events=tuple(self._events), states=tuple(self._snapshots),
            error_code=self._error_code, error_action_index=self._error_action_index,
            error_t_ns=self._error_t, d2h_bytes=self._amounts[1], ssd_read_bytes=self._amounts[2],
            ssd_write_bytes=self._amounts[3], peak_dram_bytes=self._peak_dram,
        )
        return SessionResult(evaluation, self._state, tuple(self._environment_records),
                             tuple(self._service), tuple(self._nominal_ends), tuple(self._sources),
                             tuple(self._reclaim_records), tuple(self._control_events))

    @staticmethod
    def _validate_source(source_index: int | None) -> None:
        if source_index is not None and (isinstance(source_index, bool)
                or not isinstance(source_index, int) or source_index < 0):
            raise InvalidInput("source_index must be a non-negative integer or None")
