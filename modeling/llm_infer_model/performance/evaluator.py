"""Deterministic event evaluation of physical work and resource availability.

This is a performance model, not an operating-system or device scheduler. The
input graph specifies execution order among ready tasks and resource sharing.
Piecewise rates are available physical service, not utilization percentages.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .graph import Graph, ResourceTimeline
from .parameters import InvalidModel, ParameterBundle


class CapacityExceeded(InvalidModel):
    """Resident allocations and running workspace exceed an explicit capacity."""


class BlockedExecution(InvalidModel):
    """Unfinished work has no service/capacity recovery in the supplied trace."""


@dataclass(frozen=True)
class TaskExecution:
    task_id: str
    resource: str
    ready_ns: int
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self):
        return self.end_ns - self.start_ns

    @property
    def waiting_ns(self):
        return self.start_ns - self.ready_ns

    def to_dict(self):
        return dict(task_id=self.task_id, resource=self.resource, ready_ns=self.ready_ns,
                    start_ns=self.start_ns, end_ns=self.end_ns, duration_ns=self.duration_ns,
                    waiting_ns=self.waiting_ns)


@dataclass(frozen=True)
class WaitInterval:
    task_id: str
    start_ns: int
    end_ns: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    duration_ns: int
    tasks: tuple[TaskExecution, ...]
    waits: tuple[WaitInterval, ...]
    # Events contain only model-computed state transitions, never raw samples.
    events: tuple[tuple[int, str, str], ...]

    def to_dict(self):
        return {"duration_ns": self.duration_ns, "tasks": [t.to_dict() for t in self.tasks],
                "waits": [dict(task_id=w.task_id, start_ns=w.start_ns, end_ns=w.end_ns,
                               reasons=list(w.reasons)) for w in self.waits],
                "events": [dict(at_ns=at, kind=kind, subject=subject)
                           for at, kind, subject in self.events]}


def evaluate(graph: Graph, bundle: ParameterBundle,
             timeline: ResourceTimeline | None = None) -> Evaluation:
    """Evaluate a DAG with exact work progress and integer-ns completions.

    The graph's tuple order is the explicit tie-break between ready tasks.
    Each named resource is exclusive; independent resources overlap. A task may
    additionally lock shared resources. Rates change remaining work only: past
    work is retained, zero pauses, and fixed latency tasks are never scaled.
    Memory reservations are held throughout execution; live capacity violations
    fail explicitly, rather than silently evicting or preempting allocations.
    """
    if not isinstance(graph, Graph) or not isinstance(bundle, ParameterBundle):
        raise InvalidModel("evaluate requires Graph and ParameterBundle")
    timeline = ResourceTimeline() if timeline is None else timeline
    if not isinstance(timeline, ResourceTimeline):
        raise InvalidModel("timeline must be ResourceTimeline")
    rates, fixed = {}, {}
    for task in graph.tasks:
        if task.fixed_parameter is not None:
            fixed[task.id] = bundle.require(task.fixed_parameter, "ns")
        for demand in task.demands:
            rates[demand.parameter] = bundle.require(demand.parameter, demand.unit + "/s")
    domains = {d.name: d for d in graph.memory_domains}
    capacities = {d.name: int(bundle.require(d.capacity_parameter, "byte"))
                  for d in graph.memory_domains}
    external = {d.name: d.external_bytes for d in graph.memory_domains}
    for point in timeline.points:
        for name in point.rates:
            if bundle.get(name).unit not in {"flop/s", "byte/s"}:
                raise InvalidModel(f"timeline can only change physical rates: {name}")
        if not set(point.capacities) <= domains.keys() or not set(point.external_memory) <= domains.keys():
            raise InvalidModel("timeline references an undeclared memory domain")

    pending = list(graph.tasks)
    active, completed, locks, records, events, waits = {}, {}, set(), {}, [], []
    time, point_index = 0, 0

    def usage():
        used = {name: d.resident_bytes + external[name] for name, d in domains.items()}
        for item in active.values():
            for reservation in item["task"].memory:
                used[reservation.domain] += reservation.bytes
        return used

    def check_capacity():
        for name, used in usage().items():
            if used > capacities[name]:
                raise CapacityExceeded(f"at {time} ns: {name} uses {used} bytes; capacity {capacities[name]}")

    def blockers(task):
        blocked = []
        if locks.intersection((task.resource,) + task.extra_resources):
            blocked.append("resource_busy")
        used = usage()
        if any(used[m.domain] + m.bytes > capacities[m.domain] for m in task.memory):
            blocked.append("memory_capacity")
        return tuple(blocked)

    def finish():
        for task_id, item in tuple(active.items()):
            if all(value <= 0 for value in item["remaining"]):
                task = item["task"]
                completed[task_id] = time
                locks.difference_update((task.resource,) + task.extra_resources)
                records[task_id] = TaskExecution(task_id, task.resource, item["ready"], item["start"], time)
                events.append((time, "finish", task_id))
                del active[task_id]

    while pending or active or (time == 0 and not events):
        finish()
        # Completion releases workspace before a state change at that instant.
        if point_index < len(timeline.points) and timeline.points[point_index].at_ns == time:
            point = timeline.points[point_index]
            rates.update(point.rates)
            capacities.update(point.capacities)
            external.update(point.external_memory)
            events.append((time, "resource_state", str(point_index)))
            point_index += 1
        check_capacity()
        while True:
            launched = False
            for task in tuple(pending):
                if not set(task.depends_on) <= completed.keys() or blockers(task):
                    continue
                active[task.id] = dict(task=task, start=time,
                    ready=max((completed[x] for x in task.depends_on), default=0),
                    remaining=([fixed[task.id]] if task.fixed_parameter is not None else
                               [Fraction(d.amount) for d in task.demands]))
                locks.update((task.resource,) + task.extra_resources)
                pending.remove(task)
                events.append((time, "start", task.id))
                launched = True
            finish()
            if not launched:
                break
        if not pending and not active:
            break
        next_times = []
        if point_index < len(timeline.points):
            next_times.append(timeline.points[point_index].at_ns)
        for item in active.values():
            task, remaining = item["task"], item["remaining"]
            if task.fixed_parameter is not None:
                duration = remaining[0]
            else:
                durations = [work * 1_000_000_000 / rates[d.parameter]
                             if rates[d.parameter] else None
                             for d, work in zip(task.demands, remaining) if work > 0]
                if any(value is None for value in durations):
                    continue
                duration = max(durations, default=Fraction(0))
            next_times.append(time + max(1, math.ceil(duration)))
        if not next_times:
            raise BlockedExecution(f"at {time} ns: no future completion or resource recovery")
        next_time = min(next_times)
        elapsed = next_time - time
        for task in pending:
            if set(task.depends_on) <= completed.keys():
                reasons = blockers(task)
                if reasons:
                    waits.append(WaitInterval(task.id, time, next_time, reasons))
        for item in active.values():
            task = item["task"]
            if task.fixed_parameter is not None:
                item["remaining"][0] = max(Fraction(0), item["remaining"][0] - elapsed)
            else:
                for i, demand in enumerate(task.demands):
                    item["remaining"][i] = max(Fraction(0), item["remaining"][i]
                        - Fraction(elapsed) * rates[demand.parameter] / 1_000_000_000)
        time = next_time
    return Evaluation(time, tuple(records[t.id] for t in graph.tasks), tuple(waits), tuple(events))
