"""Exact-grid completed-action responses evaluated by the common event engine.

The reference work is one inclusive completed invocation, not GPU FLOPs or
memory traffic. Its progress follows d(work)/dt = 1 / T(state(t)). This is an
explicit memoryless response approximation, which needs independent dynamic
validation; it does not identify submission, device and wakeup components.

All intervals are serial on one composite resource. CPU/GPU replay controls are
offered binary workloads, not measured utilization or available service ratios.
Memory holds condition response lookup and do not establish physical capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

from .evaluator import Evaluation, evaluate
from .graph import Graph, ResourcePoint, ResourceTimeline, Task, WorkDemand
from .parameters import (INSTANTANEOUS_RATE, InstantaneousRate, InvalidModel,
    Parameter, ParameterBundle, UnknownParameter, decode_fraction,
    encode_fraction, exact_fields, integer, nonempty, rational)


@dataclass(frozen=True)
class ResponseState:
    cpu_replay: int
    gpu_replay: int
    host_memory_bytes: int
    device_memory_bytes: int

    def __post_init__(self):
        for name in ("cpu_replay", "gpu_replay"):
            integer(getattr(self, name), name)
            if getattr(self, name) not in (0, 1):
                raise InvalidModel("replay controls must be integer zero or one, not utilization")
        integer(self.host_memory_bytes, "host memory hold bytes")
        integer(self.device_memory_bytes, "device memory hold bytes")

    def to_dict(self):
        return dict(cpu_replay=self.cpu_replay, gpu_replay=self.gpu_replay,
            host_memory_bytes=self.host_memory_bytes, device_memory_bytes=self.device_memory_bytes)

    @classmethod
    def from_dict(cls, value):
        return cls(**exact_fields(value, {"cpu_replay", "gpu_replay", "host_memory_bytes",
                                        "device_memory_bytes"}, "response state"))


@dataclass(frozen=True)
class ResponseCell:
    action_id: str
    state: ResponseState
    duration_ns: int | Fraction | None
    evidence: tuple[str, ...]
    qualified: bool

    def __post_init__(self):
        nonempty(self.action_id, "response action id")
        if not isinstance(self.state, ResponseState):
            raise InvalidModel("response cell requires an exact ResponseState")
        if self.duration_ns is not None:
            object.__setattr__(self, "duration_ns", rational(self.duration_ns, "completed duration ns"))
        if type(self.evidence) is not tuple or not self.evidence or any(
                not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise InvalidModel("response cells require explicit evidence references")
        if type(self.qualified) is not bool:
            raise InvalidModel("response qualification must be a boolean")

    def to_dict(self):
        return dict(action_id=self.action_id, state=self.state.to_dict(),
            duration_ns=None if self.duration_ns is None else encode_fraction(self.duration_ns),
            evidence=list(self.evidence), qualified=self.qualified)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"action_id", "state", "duration_ns", "evidence", "qualified"}, "response cell")
        if not isinstance(obj["evidence"], list):
            raise InvalidModel("response cell evidence must be a JSON list")
        duration = None if obj["duration_ns"] is None else decode_fraction(obj["duration_ns"], "duration ns")
        return cls(obj["action_id"], ResponseState.from_dict(obj["state"]), duration,
                   tuple(obj["evidence"]), obj["qualified"])


@dataclass(frozen=True)
class ResponseTable:
    context_sha256: str
    action_ids: tuple[str, ...]
    cells: tuple[ResponseCell, ...]
    execution_mode: str = "serial_host_completed"
    _by_key: Mapping[tuple[str, ResponseState], ResponseCell] = field(init=False,
        repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.context_sha256, str) or len(self.context_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.context_sha256)):
            raise InvalidModel("exact program/hardware/backend context SHA256 required")
        if self.execution_mode != "serial_host_completed":
            raise InvalidModel("inclusive completed responses cannot be decomposed or overlapped")
        if type(self.action_ids) is not tuple or not self.action_ids:
            raise InvalidModel("ordered response action ids must be a nonempty tuple")
        for name in self.action_ids:
            nonempty(name, "response action id")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise InvalidModel("response action occurrences need unique ids")
        if type(self.cells) is not tuple or any(not isinstance(c, ResponseCell) for c in self.cells):
            raise InvalidModel("response cells must be a tuple")
        keys = [(c.action_id, c.state) for c in self.cells]
        if len(keys) != len(set(keys)):
            raise InvalidModel("duplicate exact-grid response cell")
        if any(c.action_id not in self.action_ids for c in self.cells):
            raise InvalidModel("response cell has an undeclared action")
        object.__setattr__(self, "_by_key", MappingProxyType(dict(zip(keys, self.cells))))

    def lookup(self, action_id, state):
        nonempty(action_id, "response action id")
        if not isinstance(state, ResponseState):
            raise InvalidModel("response lookup requires an exact ResponseState")
        cell = self._by_key.get((action_id, state))
        if cell is None:
            raise UnknownParameter(f"unprofiled exact response cell: {action_id}, {state}")
        if cell.duration_ns is None:
            raise UnknownParameter(f"unknown completed response: {action_id}, {state}")
        return cell

    def to_dict(self):
        return dict(schema_version="conditioned-completed-responses-1", context_sha256=self.context_sha256,
            execution_mode=self.execution_mode, action_ids=list(self.action_ids),
            cells=[cell.to_dict() for cell in self.cells])

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "context_sha256", "execution_mode", "action_ids", "cells"}, "response table")
        if obj["schema_version"] != "conditioned-completed-responses-1":
            raise InvalidModel("unsupported conditioned response schema")
        if not isinstance(obj["action_ids"], list) or not isinstance(obj["cells"], list):
            raise InvalidModel("response actions and cells must be JSON lists")
        return cls(obj["context_sha256"], tuple(obj["action_ids"]),
            tuple(ResponseCell.from_dict(c) for c in obj["cells"]), obj["execution_mode"])


@dataclass(frozen=True)
class ResponsePoint:
    """An external state boundary and explicit composite service availability.

    ``available=False`` means a declared service blackout, never an unknown
    state or missing price. Unknown response cells remain rejected by lookup.
    """

    at_ns: int
    state: ResponseState
    available: bool = True

    def __post_init__(self):
        integer(self.at_ns, "response boundary ns")
        if not isinstance(self.state, ResponseState) or type(self.available) is not bool:
            raise InvalidModel("response point requires exact state and explicit service availability")

    def to_dict(self):
        return dict(at_ns=self.at_ns, state=self.state.to_dict(), available=self.available)


@dataclass(frozen=True)
class ResponseTrajectory:
    initial_state: ResponseState
    points: tuple[ResponsePoint, ...] = ()
    initial_available: bool = True

    def __post_init__(self):
        if not isinstance(self.initial_state, ResponseState) or type(self.initial_available) is not bool:
            raise InvalidModel("trajectory requires exact initial state and service availability")
        if type(self.points) is not tuple or any(not isinstance(p, ResponsePoint) for p in self.points):
            raise InvalidModel("response boundaries must be a tuple")
        if any(a.at_ns >= b.at_ns for a, b in zip(self.points, self.points[1:])):
            raise InvalidModel("response boundaries must be strictly increasing")

    def to_dict(self):
        return dict(initial_state=self.initial_state.to_dict(), initial_available=self.initial_available,
                    points=[p.to_dict() for p in self.points])

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"initial_state", "initial_available", "points"}, "response trajectory")
        if not isinstance(obj["points"], list):
            raise InvalidModel("response points must be a JSON list")
        points = []
        for item in obj["points"]:
            item = exact_fields(item, {"at_ns", "state", "available"}, "response point")
            points.append(ResponsePoint(item["at_ns"], ResponseState.from_dict(item["state"]), item["available"]))
        return cls(ResponseState.from_dict(obj["initial_state"]), tuple(points), obj["initial_available"])


@dataclass(frozen=True)
class RequestResponseDelays:
    """Scoped actuator delays supplied from independent worker observations.

    These are delays between an external request edge and effective offered
    work, not utilization, GPU throughput, or a fit to foreground completion.
    ``None`` is unknown; an observed zero delay must be explicitly supplied.
    """

    cpu_start_ns: int | None
    cpu_stop_ns: int | None
    gpu_start_ns: int | None
    gpu_stop_ns: int | None
    evidence: tuple[str, ...]
    qualified: bool

    def __post_init__(self):
        for name in ("cpu_start_ns", "cpu_stop_ns", "gpu_start_ns", "gpu_stop_ns"):
            value = getattr(self, name)
            if value is not None:
                integer(value, name)
        if type(self.evidence) is not tuple or not self.evidence or any(
                not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise InvalidModel("request-response delays require explicit worker evidence")
        if type(self.qualified) is not bool:
            raise InvalidModel("delay source qualification must be a boolean")

    def to_dict(self):
        return dict(schema_version="request-response-delays-1",
            cpu_start_ns=self.cpu_start_ns, cpu_stop_ns=self.cpu_stop_ns,
            gpu_start_ns=self.gpu_start_ns, gpu_stop_ns=self.gpu_stop_ns,
            evidence=list(self.evidence), qualified=self.qualified)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "cpu_start_ns", "cpu_stop_ns",
            "gpu_start_ns", "gpu_stop_ns", "evidence", "qualified"}, "request-response delays")
        if obj["schema_version"] != "request-response-delays-1":
            raise InvalidModel("unsupported request-response delay schema")
        if not isinstance(obj["evidence"], list):
            raise InvalidModel("delay evidence must be a JSON list")
        return cls(obj["cpu_start_ns"], obj["cpu_stop_ns"], obj["gpu_start_ns"],
                   obj["gpu_stop_ns"], tuple(obj["evidence"]), obj["qualified"])


def apply_request_response_delays(trajectory: ResponseTrajectory, delays: RequestResponseDelays,
                                 *, initial_effective_state: ResponseState) -> ResponseTrajectory:
    """Transform exogenous requests; this does not calculate execution time.

    Origin zero must precede the first requested activation; the caller supplies
    explicit effective CPU/GPU controls 0/0 at that origin. Repeated identical
    requests preserve a pending edge. An opposite request cancels it, including
    at the exact due timestamp (requests precede due activations). CPU and GPU
    evolve independently. Memory holds and explicit availability change at the
    original request timestamps, without an invented memory or service delay.

    Every needed direction requires a known delay, even if a later edge might
    cancel it; unknown timing cannot establish whether it would activate first.
    """
    if not isinstance(trajectory, ResponseTrajectory) or not isinstance(delays, RequestResponseDelays):
        raise InvalidModel("request transformation requires trajectory and scoped delays")
    if not isinstance(initial_effective_state, ResponseState):
        raise InvalidModel("explicit initial effective state is required")
    if (initial_effective_state.cpu_replay, initial_effective_state.gpu_replay) != (0, 0):
        raise InvalidModel("this actuator protocol requires explicit initially idle CPU/GPU controls 0/0")
    if (initial_effective_state.host_memory_bytes, initial_effective_state.device_memory_bytes) != (
            trajectory.initial_state.host_memory_bytes, trajectory.initial_state.device_memory_bytes):
        raise InvalidModel("initial effective memory must match the external memory holds")
    # A point at zero overrides the initial request just as in evaluate_response.
    requests = {0: ResponsePoint(0, trajectory.initial_state, trajectory.initial_available)}
    requests.update((point.at_ns, point) for point in trajectory.points)
    requests = tuple(requests.values())
    requested = {"cpu": None, "gpu": None}
    effective = {"cpu": 0, "gpu": 0}
    pending = {"cpu": None, "gpu": None}
    index, initial, points, previous = 0, None, [], None
    memory = (initial_effective_state.host_memory_bytes, initial_effective_state.device_memory_bytes)
    available = trajectory.initial_available
    while index < len(requests) or any(item is not None for item in pending.values()):
        times = [item[0] for item in pending.values() if item is not None]
        if index < len(requests):
            times.append(requests[index].at_ns)
        now = min(times)
        if index < len(requests) and requests[index].at_ns == now:
            point = requests[index]
            index += 1
            memory = (point.state.host_memory_bytes, point.state.device_memory_bytes)
            available = point.available
            for device in ("cpu", "gpu"):
                target = getattr(point.state, device + "_replay")
                if target == requested[device]:
                    continue
                requested[device] = target
                pending[device] = None
                if target != effective[device]:
                    name = device + ("_start_ns" if target else "_stop_ns")
                    delay = getattr(delays, name)
                    if delay is None:
                        raise UnknownParameter("unknown request-response delay: " + name)
                    pending[device] = (now + delay, target)
        for device in ("cpu", "gpu"):
            if pending[device] is not None and pending[device][0] == now:
                effective[device] = pending[device][1]
                pending[device] = None
        point = ResponsePoint(now, ResponseState(effective["cpu"], effective["gpu"], *memory), available)
        if initial is None:
            initial = point
        elif (point.state, point.available) != previous:
            points.append(point)
        previous = (point.state, point.available)
    return ResponseTrajectory(initial.state, tuple(points), initial.available)


@dataclass(frozen=True)
class ConditionedResponseEvaluation:
    context_sha256: str
    start_ns: int
    execution: Evaluation
    response_support_qualified: bool
    zero_duration_approximation_used: bool
    request_delays: RequestResponseDelays | None = None
    initial_effective_state: ResponseState | None = None
    effective_trajectory: ResponseTrajectory | None = None

    @property
    def duration_ns(self):
        return self.execution.duration_ns

    @property
    def end_ns(self):
        return self.start_ns + self.duration_ns

    def to_dict(self):
        result = dict(schema_version="conditioned-completed-response-prediction-1",
            context_sha256=self.context_sha256, start_ns=self.start_ns, end_ns=self.end_ns,
            duration_ns=self.duration_ns, execution=self.execution.to_dict(),
            absolute_tasks=[dict(task_id=t.task_id, start_ns=self.start_ns + t.start_ns,
                end_ns=self.start_ns + t.end_ns) for t in self.execution.tasks],
            response_support_qualified=self.response_support_qualified,
            zero_duration_approximation_used=self.zero_duration_approximation_used,
            zero_duration_scope="source-resolution/statistical zero-cost approximation; not physical instantaneous work",
            timing_scope="inclusive host-completed serial action responses under the exact source context",
            progress_rule="one reference invocation; d(progress)/dt = 1 / completed_duration(state(t))",
            within_action_memoryless_response_assumed=True,
            cpu_gpu_controls_are_measured_utilization=False,
            host_device_overlap_decomposed=False, physical_gpu_throughput_claimed=False,
            physical_memory_capacity_claimed=False,
            memory_holds_scope="exact conditioning coordinates; capacity and unknown workspace are not inferred",
            prediction_provider="llm_infer_model.performance.evaluate",
            parameter_estimation_or_measurement_performed_here=False,
            independently_validated_accuracy_claimed=False)
        if self.request_delays is not None:
            result.update(request_response_lag=dict(parameters=self.request_delays.to_dict(),
                initial_effective_state=self.initial_effective_state.to_dict(),
                effective_trajectory=self.effective_trajectory.to_dict(),
                pending_policy="opposite request cancels pending edge; repeated same request does not reset it",
                same_timestamp_order="new request before pending activation",
                scope="deterministic worker request-response lag; not an operating-system forecast or hardware utilization",
                worker_evidence_estimated_here=False),
                request_response_support_qualified=self.request_delays.qualified,
                all_model_parameter_sources_qualified=self.response_support_qualified and self.request_delays.qualified)
        return result


def evaluate_response(table: ResponseTable, trajectory: ResponseTrajectory, *, start_ns: int = 0,
                      request_delays: RequestResponseDelays | None = None,
                      initial_effective_state: ResponseState | None = None):
    """Compile responses to invocation work and use the existing event evaluator.

    ``start_ns`` and boundary timestamps share an external integer-ns origin.
    Boundaries at/before entry select its initial state. Remaining boundaries
    are shifted to invocation-relative times; no heldout action timestamps are
    consulted. All future supplied cells must be known, a conservative support
    check even when a late boundary might follow completion.

    A supplied unavailable interval pauses work until an explicit recovery.
    It does not establish the performance of an unobserved/unknown interval.
    Optional actuator delays transform the full request timeline from its own
    origin before selecting the invocation entry state. They do not reset at
    each invocation, fit main work, or add a second execution-time predictor.
    """
    if not isinstance(table, ResponseTable) or not isinstance(trajectory, ResponseTrajectory):
        raise InvalidModel("response evaluation requires ResponseTable and ResponseTrajectory")
    integer(start_ns, "invocation entry ns")
    if request_delays is not None:
        trajectory = apply_request_response_delays(trajectory, request_delays,
                                                   initial_effective_state=initial_effective_state)
    elif initial_effective_state is not None:
        raise InvalidModel("initial effective state requires an explicit request-delay model")
    state, available = trajectory.initial_state, trajectory.initial_available
    future = []
    for point in trajectory.points:
        if point.at_ns <= start_ns:
            state, available = point.state, point.available
        else:
            future.append(point)
    points = [ResponsePoint(0, state, available)] + [
        ResponsePoint(p.at_ns - start_ns, p.state, p.available) for p in future]
    cells = [table.lookup(action, point.state) for point in points for action in table.action_ids]
    def rate(cell, enabled):
        if not enabled:
            return Fraction(0)
        return INSTANTANEOUS_RATE if not cell.duration_ns else Fraction(1_000_000_000) / cell.duration_ns
    parameters = tuple(Parameter(f"response_rate:{i}", rate(table.lookup(action, state), available),
        "invocation/s", "profiled", table.lookup(action, state).evidence,
        "derived inverse inclusive completed-action response; not physical device service")
        for i, action in enumerate(table.action_ids))
    tasks = tuple(Task(action, "completed_action_response", (WorkDemand(f"response_rate:{i}", 1, "invocation"),),
        depends_on=() if i == 0 else (table.action_ids[i - 1],)) for i, action in enumerate(table.action_ids))
    timeline = ResourceTimeline(tuple(ResourcePoint(point.at_ns, rates={f"response_rate:{i}":
        rate(table.lookup(action, point.state), point.available) for i, action in enumerate(table.action_ids)}) for point in points))
    result = evaluate(Graph(tasks), ParameterBundle(parameters), timeline)
    return ConditionedResponseEvaluation(table.context_sha256, start_ns, result,
        all(cell.qualified for cell in cells), any(cell.duration_ns == 0 for cell in cells),
        request_delays, initial_effective_state, trajectory if request_delays is not None else None)
