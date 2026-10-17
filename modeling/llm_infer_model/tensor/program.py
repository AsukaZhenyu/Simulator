"""Offline execution programs over the existing tensor transition rules.

This module adds explicit waits and idle-boundary environment changes to a
fixed action program. It does not choose actions, repair plans, react online,
or change the duration of a task already in flight. Every physical action is
applied by :func:`engine.transition` to the same State.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterable

from .engine import (
    ACTION_ADVANCE, Action, EvaluationResult, InitialCapacityExceeded,
    MappingError, State, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION,
    CODE_INPUT_NOT_READY, CODE_RESOURCE_BUSY, REASON_ACTIONS_REMAIN_AFTER_GOAL,
    REASON_GOAL_REACHED, REASON_INITIAL_CAPACITY_EXCEEDED, REASON_UNFINISHED,
    STATUS_INCOMPLETE_MAPPING, STATUS_INFEASIBLE, STATUS_INVALID_MAPPING,
    STATUS_VALID, _accumulate_transfer_bytes, _event_for, _failure, _snapshot,
    initial_state, is_goal, task_domain, transition, used_bytes, used_vram_bytes,
)
from .spec import (
    COPY_READY, COPY_RESERVED_COPY, COPY_RESERVED_OUTPUT, DOMAINS, LOC_DRAM,
    LOC_SSD, LOC_VRAM, OP_DONE, OP_RUNNING, RESOURCE_CPU_COMPUTE,
    RESOURCE_GPU_COMPUTE, Architecture, ComputeCost, Costs, H2DCost, M1Costs,
    M1Goal, Mapspace, M1Mapspace, TransferCost, InvalidInput, Scenario, Unsupported,
    _canonical_workload, _require_int, _require_str,
    build_scenario, scenario_fingerprint,
)

__all__ = [
    "ActionStep", "WaitStep", "DeviceStateStep", "EnvironmentEvent",
    "ProgramResult", "validate_device_state", "execute_program",
]


@dataclass(frozen=True)
class ActionStep:
    action: Action
    source_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise InvalidInput("ActionStep.action must be an Action")
        _require_int(self.source_index, "ActionStep.source_index")


@dataclass(frozen=True)
class WaitStep:
    operations: tuple[str, ...] = ()
    copies: tuple[tuple[str, str], ...] = ()
    drain: bool = False
    source_index: int = 0

    def __post_init__(self) -> None:
        _require_int(self.source_index, "WaitStep.source_index")
        if not isinstance(self.drain, bool):
            raise InvalidInput("WaitStep.drain must be a boolean")
        if not isinstance(self.operations, tuple) or not isinstance(self.copies, tuple):
            raise InvalidInput("WaitStep operations and copies must be tuples")
        for operation in self.operations:
            _require_str(operation, "WaitStep operation")
            if not operation:
                raise InvalidInput("WaitStep operation id must not be empty")
        if len(set(self.operations)) != len(self.operations):
            raise InvalidInput("WaitStep operations must be unique")
        for copy in self.copies:
            if not isinstance(copy, tuple) or len(copy) != 2:
                raise InvalidInput("WaitStep copy must be a (tensor_id, location) tuple")
            _require_str(copy[0], "WaitStep tensor_id")
            if not copy[0]:
                raise InvalidInput("WaitStep tensor id must not be empty")
            if copy[1] not in (LOC_DRAM, LOC_VRAM, LOC_SSD):
                raise InvalidInput(f"WaitStep has unknown location {copy[1]!r}")
        if len(set(self.copies)) != len(self.copies):
            raise InvalidInput("WaitStep copies must be unique")
        if not (self.operations or self.copies or self.drain):
            raise InvalidInput("WaitStep must name a target or request drain")


@dataclass(frozen=True)
class DeviceStateStep:
    name: str
    scenario: Scenario
    source_index: int = 0

    def __post_init__(self) -> None:
        _require_str(self.name, "DeviceStateStep.name")
        if not self.name:
            raise InvalidInput("DeviceStateStep.name must not be empty")
        _require_int(self.source_index, "DeviceStateStep.source_index")
        if not isinstance(self.scenario, Scenario):
            raise InvalidInput("DeviceStateStep.scenario must be a Scenario")


@dataclass(frozen=True)
class EnvironmentEvent:
    source_index: int
    name: str
    t_ns: int
    previous_profile: str
    profile: str
    used_dram_bytes: int
    used_vram_bytes: int


@dataclass(frozen=True)
class ProgramResult:
    evaluation: EvaluationResult
    final_state: State | None
    environment_events: tuple[EnvironmentEvent, ...]
    action_sources: tuple[int, ...]
    action_profiles: tuple[str, ...]
    error_source_index: int | None = None


ProgramStep = ActionStep | WaitStep | DeviceStateStep


def _lists(value):
    """Restore document arrays without coercing scalar values or costs."""
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    if isinstance(value, list):
        return [_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _lists(item) for key, item in value.items()}
    return value


def _validate_scenario(scenario: Scenario) -> None:
    """Re-use the JSON contract for directly constructed profile objects.

    No initial-capacity check belongs here: a later profile is checked against
    the actual runtime inventory, not the original workload initial layout.
    """
    if not isinstance(scenario, Scenario):
        raise InvalidInput("device state profile must be a Scenario")
    if not isinstance(scenario.architecture, Architecture):
        raise InvalidInput("device state architecture must be an Architecture")
    if scenario.is_m1:
        if not isinstance(scenario.costs, M1Costs) or not isinstance(scenario.mapspace, M1Mapspace):
            raise InvalidInput("0.2 device state requires M1Costs and M1Mapspace")
        if not isinstance(scenario.goal, M1Goal):
            raise InvalidInput("0.2 device state requires M1Goal")
    elif not isinstance(scenario.costs, Costs) or not isinstance(scenario.mapspace, Mapspace):
        raise InvalidInput("0.1 device state requires Costs and Mapspace")
    elif scenario.goal is not None:
        raise InvalidInput("0.1 device state has no explicit goal")
    tables = (("compute", ComputeCost), ("transfer", TransferCost)) if scenario.is_m1 else (
        ("compute", ComputeCost), ("h2d", H2DCost),
    )
    for name, cost_type in tables:
        table = getattr(scenario.costs, name)
        if not isinstance(table, dict):
            raise InvalidInput(f"device state costs.{name} must be a dict")
        for key, entry in table.items():
            _require_str(key, f"device state costs.{name} key")
            if scenario.is_m1:
                if not isinstance(entry, dict):
                    raise InvalidInput(f"device state costs.{name}[{key!r}] must be a dict")
                entries = entry.items()
            else:
                entries = ((key, entry),)
            for cost_key, cost in entries:
                _require_str(cost_key, f"device state costs.{name} entry key")
                if not isinstance(cost, cost_type):
                    raise InvalidInput(f"device state costs.{name} requires {cost_type.__name__} entries")
    # Canonical serialization uses effective alignments/reservations. Check
    # every raw field too, so an unused bool/float alias cannot be hidden.
    optional = {
        "dram_capacity_bytes", "dram_alignment_bytes", "vram_alignment_bytes",
        "vram_runtime_reserved_bytes",
    }
    positive = {
        "dram_capacity_bytes", "vram_capacity_bytes", "allocation_alignment_bytes",
        "dram_alignment_bytes", "vram_alignment_bytes", "cpu_compute_slots",
        "gpu_compute_slots", "h2d_copy_slots", "pcie_copy_slots", "ssd_io_slots",
    }
    for field in fields(Architecture):
        value = getattr(scenario.architecture, field.name)
        if value is None and field.name in optional:
            continue
        _require_int(value, f"device state architecture.{field.name}",
                     minimum=1 if field.name in positive else 0)
    workload = _canonical_workload(scenario.workload)
    workload["schema_version"] = scenario.schema_version
    workload["origin"] = asdict(scenario.workload.origin)
    arch = scenario.architecture
    if scenario.is_m1:
        architecture = {
            "dram_capacity_bytes": arch.dram_capacity_bytes,
            "vram_capacity_bytes": arch.vram_capacity_bytes,
            "dram_runtime_reserved_bytes": arch.reserved_of(LOC_DRAM),
            "vram_runtime_reserved_bytes": arch.reserved_of(LOC_VRAM),
            "dram_alignment_bytes": arch.alignment_of(LOC_DRAM),
            "vram_alignment_bytes": arch.alignment_of(LOC_VRAM),
            **{name: getattr(arch, name) for name in
               ("cpu_compute_slots", "gpu_compute_slots", "pcie_copy_slots", "ssd_io_slots")},
        }
    else:
        architecture = {name: getattr(arch, name) for name in (
            "dram_capacity_bytes", "vram_capacity_bytes", "runtime_reserved_bytes",
            "allocation_alignment_bytes", "gpu_compute_slots", "h2d_copy_slots",
        )}
    document = {
        "schema_version": scenario.schema_version, "id": scenario.id,
        "architecture": architecture, "costs": _lists(asdict(scenario.costs)),
        "mapspace": _lists(asdict(scenario.mapspace)), "mapper": asdict(scenario.mapper),
    }
    if scenario.goal is not None:
        document["goal"] = asdict(scenario.goal)
    build_scenario(workload, document, where="device_state")


def validate_device_state(base: Scenario, candidate: Scenario) -> None:
    """Validate a complete idle-boundary profile without inspecting a State.

    Only capacities, runtime reservations, and the complete cost table/source
    may change. Graph, tensor layouts, mapspace, goal, format, id, mapper, and
    resource slot counts remain identical. Raises InvalidInput/Unsupported.
    """
    _validate_scenario(base)
    _validate_scenario(candidate)
    for name in ("schema_version", "id", "workload", "mapspace", "goal", "mapper"):
        if getattr(base, name) != getattr(candidate, name):
            raise Unsupported(f"device_state cannot change scenario.{name}")
    mutable = {
        "dram_capacity_bytes", "vram_capacity_bytes", "runtime_reserved_bytes",
        "dram_runtime_reserved_bytes", "vram_runtime_reserved_bytes",
    }
    for field in fields(Architecture):
        if field.name not in mutable and (
            getattr(base.architecture, field.name)
            != getattr(candidate.architecture, field.name)
        ):
            raise Unsupported(f"device_state cannot change architecture.{field.name}")


def _copy_ready(scenario: Scenario, state: State, tensor: str, location: str) -> bool:
    if not scenario.is_m1 and location == LOC_DRAM:
        return location in scenario.workload.tensor_by_id[tensor].initial_locations
    return state.copy_at(tensor, location) == COPY_READY


def _check_wait(scenario: Scenario, state: State, step: WaitStep) -> None:
    for operation in step.operations:
        if operation not in scenario.workload.operation_by_id:
            raise MappingError(CODE_ILLEGAL_ACTION, f"wait names unknown operation {operation!r}")
        if state.op(operation) == OP_DONE:
            continue
        if state.op(operation) != OP_RUNNING or not any(
            task.target_id == operation
            and task.resource in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE)
            for task in state.running
        ):
            raise MappingError(CODE_INPUT_NOT_READY,
                               f"wait operation {operation!r} has not been started")
    for tensor, location in step.copies:
        if tensor not in scenario.workload.tensor_by_id:
            raise MappingError(CODE_ILLEGAL_ACTION, f"wait names unknown tensor {tensor!r}")
        if location == LOC_SSD and not scenario.is_m1:
            raise MappingError(CODE_ILLEGAL_ACTION, "0.1 has no SSD copies")
        if _copy_ready(scenario, state, tensor, location):
            continue
        if state.copy_at(tensor, location) not in (COPY_RESERVED_COPY, COPY_RESERVED_OUTPUT):
            raise MappingError(CODE_INPUT_NOT_READY,
                               f"wait copy ({tensor!r}, {location!r}) has no running producer")
        producing = False
        for task in state.running:
            if task_domain(scenario, task) != location:
                continue
            if task.resource in (RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE):
                producing |= tensor in scenario.workload.operation_by_id[task.target_id].outputs
            else:
                producing |= task.target_id == tensor
        if not producing:
            raise MappingError(CODE_INPUT_NOT_READY,
                               f"wait copy ({tensor!r}, {location!r}) has no running producer")


def _wait_done(scenario: Scenario, state: State, step: WaitStep) -> bool:
    return (
        (not step.drain or not state.running)
        and all(state.op(operation) == OP_DONE for operation in step.operations)
        and all(_copy_ready(scenario, state, tensor, location)
                for tensor, location in step.copies)
    )


def execute_program(
    scenario: Scenario, steps: Iterable[ProgramStep], start: State | None = None,
) -> ProgramResult:
    """Replay explicit actions/waits and piecewise constant device profiles.

    Malformed steps/profile changes raise InvalidInput/Unsupported. Illegal
    execution (including an unsatisfiable wait, busy switch, or capacity shrink)
    returns a failed evaluation with the original source index. Action-only
    programs retain evaluate_mapping's exact success/failure result, including
    its full supplied action tuple. Expanded programs retain the emitted action
    prefix (including a rejected action, if any); no future waits are guessed.

    Environment events carry post-switch usage. ``evaluation.states`` remains
    the initial snapshot plus one snapshot per applied action; peaks also include
    successful environment boundaries. All steps after the goal are rejected.
    """
    step_list = tuple(steps)
    for step in step_list:
        if not isinstance(step, (ActionStep, WaitStep, DeviceStateStep)):
            raise InvalidInput(f"unknown execution-program step {type(step).__name__}")
        if isinstance(step, DeviceStateStep):
            validate_device_state(scenario, step.scenario)
    static = all(isinstance(step, ActionStep) for step in step_list)
    profile = scenario_fingerprint(scenario)
    actions = [step.action for step in step_list] if static else []
    sources = [step.source_index for step in step_list] if static else []
    profiles = [profile] * len(actions)
    environments: list[EnvironmentEvent] = []
    try:
        state = start if start is not None else initial_state(scenario)
    except InitialCapacityExceeded:
        evaluation = EvaluationResult(
            status=STATUS_INFEASIBLE, reason=REASON_INITIAL_CAPACITY_EXCEEDED,
            makespan_ns=None, peak_vram_bytes=0, h2d_bytes=0, action_count=0,
            actions=(), events=(), states=(), error_code=REASON_INITIAL_CAPACITY_EXCEEDED,
        )
        return ProgramResult(evaluation, None, (), (), ())
    current = scenario
    peak = used_vram_bytes(current, state)
    peak_dram = used_bytes(current, state, LOC_DRAM)
    amounts = (0, 0, 0, 0)
    events = []
    snapshots = [_snapshot(current, state, 0)]

    def result(evaluation: EvaluationResult, source: int | None = None) -> ProgramResult:
        return ProgramResult(evaluation, state, tuple(environments), tuple(sources),
                             tuple(profiles), source)

    def failure(code: str | None, reason: str, message: str, source: int | None,
                status: str = STATUS_INVALID_MAPPING) -> ProgramResult:
        evaluation = _failure(
            current, state, len(events), status, reason, code, message,
            peak, amounts[0], tuple(actions), events, snapshots,
            peak_dram=peak_dram, d2h_bytes=amounts[1],
            ssd_read_bytes=amounts[2], ssd_write_bytes=amounts[3],
        )
        return result(evaluation, source)

    def apply(action: Action, source: int) -> ProgramResult | None:
        nonlocal state, amounts, peak, peak_dram
        index = len(events)
        if not static:
            actions.append(action)
            sources.append(source)
            profiles.append(profile)
        try:
            state = transition(current, state, action)
        except MappingError as error:
            return failure(error.code, error.code, str(error), source)
        event = _event_for(current, state, index, action)
        events.append(event)
        amounts = _accumulate_transfer_bytes(event, *amounts)
        peak = max(peak, used_vram_bytes(current, state))
        peak_dram = max(peak_dram, used_bytes(current, state, LOC_DRAM))
        snapshots.append(_snapshot(current, state, index + 1))
        return None

    for step_index, step in enumerate(step_list):
        if is_goal(current, state):
            if static:
                message = (f"the goal was reached at action index {len(events) - 1}, but "
                           f"{len(actions) - len(events)} more action(s) follow")
            else:
                message = (f"the goal was reached, but {len(step_list) - step_index} "
                           "more execution-program step(s) follow")
            return failure(CODE_ILLEGAL_ACTION, REASON_ACTIONS_REMAIN_AFTER_GOAL,
                           message, step.source_index)
        if isinstance(step, ActionStep):
            failed = apply(step.action, step.source_index)
            if failed is not None:
                return failed
        elif isinstance(step, WaitStep):
            try:
                _check_wait(current, state, step)
                while not _wait_done(current, state, step):
                    if not state.running:
                        raise MappingError(CODE_INPUT_NOT_READY, "wait cannot be satisfied")
                    failed = apply(Action(ACTION_ADVANCE), step.source_index)
                    if failed is not None:
                        return failed
            except MappingError as error:
                return failure(error.code, error.code, str(error), step.source_index)
        else:
            if state.running:
                return failure(CODE_RESOURCE_BUSY, CODE_RESOURCE_BUSY,
                               "device_state requires an idle boundary", step.source_index)
            for domain in DOMAINS:
                used = used_bytes(step.scenario, state, domain)
                capacity = step.scenario.architecture.capacity_of(domain)
                if capacity is not None and used > capacity:
                    return failure(CODE_CAPACITY_EXCEEDED, CODE_CAPACITY_EXCEEDED,
                                   f"device_state {step.name!r} needs {used} bytes in "
                                   f"{domain}, capacity is {capacity}", step.source_index)
            previous = profile
            current = step.scenario
            profile = scenario_fingerprint(current)
            peak = max(peak, used_vram_bytes(current, state))
            peak_dram = max(peak_dram, used_bytes(current, state, LOC_DRAM))
            environments.append(EnvironmentEvent(
                step.source_index, step.name, state.t, previous, profile,
                used_bytes(current, state, LOC_DRAM), used_vram_bytes(current, state),
            ))
    if not is_goal(current, state):
        return failure(None, REASON_UNFINISHED,
                       "the mapping ran out of actions before every requested output was "
                       "resident and every operation done", None, STATUS_INCOMPLETE_MAPPING)
    return result(EvaluationResult(
        status=STATUS_VALID, reason=REASON_GOAL_REACHED, makespan_ns=state.t,
        peak_vram_bytes=peak, h2d_bytes=amounts[0], action_count=len(events),
        actions=tuple(actions), events=tuple(events), states=tuple(snapshots),
        d2h_bytes=amounts[1], ssd_read_bytes=amounts[2],
        ssd_write_bytes=amounts[3], peak_dram_bytes=peak_dram,
    ))
