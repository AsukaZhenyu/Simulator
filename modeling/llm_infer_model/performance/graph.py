"""Physical work, dependency and resource-state specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

from .parameters import (InstantaneousRate, InvalidModel, decode_fraction, encode_fraction,
                         exact_fields, integer, nonempty, rational)


@dataclass(frozen=True)
class WorkDemand:
    """Conserved work, including explicitly scoped active time of one thread.

    ``thread_cpu_ns`` progresses only when its named service rate is available.
    Such work may include declared runtime work; it is not automatically useful
    FLOPs, GPU service, or an inclusive completed-action duration. A supplied
    per-thread service trace must have an independent source/declared policy;
    this contract does not derive it from heldout scheduling or utilization.
    ``cpu_cycle`` is raw scoped CPU counter work, serviced by an empirical
    ``cpu_cycle/s`` trajectory without converting cycles to active nanoseconds.
    Its amount can include user/kernel runtime work; no useful-FLOP equivalence
    or operating-system scheduling policy is inferred.
    """
    parameter: str
    amount: int
    unit: str

    def __post_init__(self):
        nonempty(self.parameter, "work parameter")
        integer(self.amount, "work amount")
        if not isinstance(self.unit, str) or self.unit not in {"flop", "byte", "invocation", "thread_cpu_ns", "cpu_cycle"}:
            raise InvalidModel("work unit must be flop, byte, invocation, thread_cpu_ns or cpu_cycle")

    def to_dict(self):
        return {"parameter": self.parameter, "amount": self.amount, "unit": self.unit}


@dataclass(frozen=True)
class MemoryReservation:
    domain: str
    bytes: int

    def __post_init__(self):
        nonempty(self.domain, "memory domain")
        integer(self.bytes, "reserved memory bytes")


@dataclass(frozen=True)
class MemoryDomain:
    name: str
    capacity_parameter: str
    resident_bytes: int
    external_bytes: int

    def __post_init__(self):
        nonempty(self.name, "memory domain name")
        nonempty(self.capacity_parameter, "capacity parameter")
        integer(self.resident_bytes, "resident memory bytes")
        integer(self.external_bytes, "external memory bytes")


@dataclass(frozen=True)
class Task:
    """One resource-exclusive execution stage.

    Multiple demands declare a roofline-style overlap assumption: every demand
    progresses concurrently and the slowest determines completion. To express
    sequential layout/compute/copy stages, use separate dependent tasks. Fixed
    latency is also a separate task, not an addition to already measured waits.
    ``extra_resources`` explicitly reserves shared engines or bandwidth domains;
    no implicit contention or proportional resource sharing is inferred.
    Naming a resource for a thread only creates its exclusive execution lane;
    explicit ``thread_cpu_ns/s`` inputs describe its CPU supply separately.
    They do not automatically conserve shared CPU capacity between lanes or
    implement preemption, runnable queues or an operating-system scheduler.

    ``waits_for`` declares a completion gate with no separate work or fixed
    duration. Launch dependencies still control when it may start. The gate
    holds its explicitly named resources and reservations until every awaited
    task completes. A CPU resource can express polling occupancy; a separate
    virtual wait resource can express a blocked caller that releases the CPU.
    The model never infers either choice from utilization or resource names.
    """

    id: str
    resource: str
    demands: tuple[WorkDemand, ...] = ()
    depends_on: tuple[str, ...] = ()
    fixed_parameter: str | None = None
    extra_resources: tuple[str, ...] = ()
    memory: tuple[MemoryReservation, ...] = ()
    waits_for: tuple[str, ...] = ()

    def __post_init__(self):
        nonempty(self.id, "task id")
        nonempty(self.resource, "task resource")
        for name, value, item_type in (("demands", self.demands, WorkDemand),
                                      ("memory", self.memory, MemoryReservation)):
            if type(value) is not tuple or any(not isinstance(x, item_type) for x in value):
                raise InvalidModel(f"task {name} must be a tuple of {item_type.__name__}")
        for name, value in (("depends_on", self.depends_on),
                            ("extra_resources", self.extra_resources),
                            ("waits_for", self.waits_for)):
            if type(value) is not tuple:
                raise InvalidModel(f"task {name} must be a tuple")
            for item in value:
                nonempty(item, name)
            if len(set(value)) != len(value):
                raise InvalidModel(f"duplicate {name}")
        if self.waits_for and (self.fixed_parameter is not None or self.demands):
            raise InvalidModel("completion gate cannot also have demands or fixed latency")
        if not self.waits_for and (self.fixed_parameter is None) == (not self.demands):
            raise InvalidModel("task requires either physical demands or one fixed parameter")
        if self.fixed_parameter is not None:
            nonempty(self.fixed_parameter, "fixed parameter")
        if self.resource in self.extra_resources:
            raise InvalidModel("primary resource repeated in extra_resources")
        if len({d.parameter for d in self.demands}) != len(self.demands):
            raise InvalidModel("duplicate work-demand parameter")
        if any(d.unit == "invocation" for d in self.demands) and len(self.demands) != 1:
            raise InvalidModel("an inclusive invocation response cannot be combined with physical demands")
        if len({m.domain for m in self.memory}) != len(self.memory):
            raise InvalidModel("duplicate task memory domain")

    def to_dict(self):
        result = {"id": self.id, "resource": self.resource,
                "demands": [d.to_dict() for d in self.demands],
                "depends_on": list(self.depends_on), "fixed_parameter": self.fixed_parameter,
                "extra_resources": list(self.extra_resources),
                "memory": [{"domain": m.domain, "bytes": m.bytes} for m in self.memory]}
        if self.waits_for:
            result["waits_for"] = list(self.waits_for)
        return result


@dataclass(frozen=True)
class Graph:
    tasks: tuple[Task, ...]
    memory_domains: tuple[MemoryDomain, ...] = ()

    def __post_init__(self):
        if type(self.tasks) is not tuple or any(not isinstance(t, Task) for t in self.tasks):
            raise InvalidModel("tasks must be a tuple of Task")
        if type(self.memory_domains) is not tuple or any(
                not isinstance(d, MemoryDomain) for d in self.memory_domains):
            raise InvalidModel("memory_domains must be a tuple of MemoryDomain")
        task_ids = {t.id for t in self.tasks}
        domains = {d.name for d in self.memory_domains}
        if len(task_ids) != len(self.tasks) or len(domains) != len(self.memory_domains):
            raise InvalidModel("duplicate task or memory domain")
        for task in self.tasks:
            if not set(task.depends_on + task.waits_for) <= task_ids:
                raise InvalidModel(f"unknown dependency for {task.id}")
            if any(m.domain not in domains for m in task.memory):
                raise InvalidModel(f"unknown memory domain for {task.id}")
        visited = set()
        while len(visited) != len(self.tasks):
            ready = {t.id for t in self.tasks if t.id not in visited
                     and set(t.depends_on + t.waits_for) <= visited}
            if not ready:
                raise InvalidModel("task graph contains a dependency cycle")
            visited.update(ready)

    def to_dict(self):
        return {"tasks": [t.to_dict() for t in self.tasks], "memory_domains": [
            {"name": d.name, "capacity_parameter": d.capacity_parameter,
             "resident_bytes": d.resident_bytes, "external_bytes": d.external_bytes}
            for d in self.memory_domains]}

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"tasks", "memory_domains"}, "work graph")
        if not isinstance(obj["tasks"], list) or not isinstance(obj["memory_domains"], list):
            raise InvalidModel("graph tasks and memory_domains must be lists")
        tasks = []
        for item in obj["tasks"]:
            fields = {"id", "resource", "demands", "depends_on",
                "fixed_parameter", "extra_resources", "memory"}
            if isinstance(item, dict) and "waits_for" in item:
                fields.add("waits_for")
            item = exact_fields(item, fields, "task")
            for name in ("demands", "depends_on", "extra_resources", "memory") + (("waits_for",) if "waits_for" in item else ()):
                if not isinstance(item[name], list):
                    raise InvalidModel(f"task {name} must be a list")
            demands = tuple(WorkDemand(**exact_fields(d, {"parameter", "amount", "unit"},
                                                       "work demand")) for d in item["demands"])
            memory = tuple(MemoryReservation(**exact_fields(m, {"domain", "bytes"},
                                                             "reservation")) for m in item["memory"])
            tasks.append(Task(item["id"], item["resource"], demands, tuple(item["depends_on"]),
                              item["fixed_parameter"], tuple(item["extra_resources"]), memory,
                              tuple(item.get("waits_for", ()))))
        domains = tuple(MemoryDomain(**exact_fields(d, {"name", "capacity_parameter",
            "resident_bytes", "external_bytes"}, "memory domain")) for d in obj["memory_domains"])
        return cls(tuple(tasks), domains)


@dataclass(frozen=True)
class ResourcePoint:
    at_ns: int
    rates: Mapping[str, int | Fraction | InstantaneousRate] = field(default_factory=dict)
    capacities: Mapping[str, int] = field(default_factory=dict)
    external_memory: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        integer(self.at_ns, "resource point time")
        for attr in ("rates", "capacities", "external_memory"):
            mapping = getattr(self, attr)
            if not isinstance(mapping, Mapping):
                raise InvalidModel(f"{attr} must be a mapping")
            copied = {}
            for name, value in mapping.items():
                nonempty(name, attr)
                copied[name] = (value if isinstance(value, InstantaneousRate) else rational(value, name)) if attr == "rates" else integer(value, name)
            object.__setattr__(self, attr, MappingProxyType(copied))

    def to_dict(self):
        return {"at_ns": self.at_ns,
                "rates": {k: encode_fraction(v) for k, v in self.rates.items()},
                "capacities": dict(self.capacities), "external_memory": dict(self.external_memory)}


@dataclass(frozen=True)
class ResourceTimeline:
    points: tuple[ResourcePoint, ...] = ()

    def __post_init__(self):
        if type(self.points) is not tuple or any(not isinstance(p, ResourcePoint) for p in self.points):
            raise InvalidModel("points must be a tuple of ResourcePoint")
        if any(a.at_ns >= b.at_ns for a, b in zip(self.points, self.points[1:])):
            raise InvalidModel("resource point times must be strictly increasing")

    def to_dict(self):
        return {"points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"points"}, "resource timeline")
        if not isinstance(obj["points"], list):
            raise InvalidModel("timeline points must be a list")
        points = []
        for p in obj["points"]:
            p = exact_fields(p, {"at_ns", "rates", "capacities", "external_memory"}, "resource point")
            if not isinstance(p["rates"], dict):
                raise InvalidModel("rates must be a mapping")
            points.append(ResourcePoint(p["at_ns"], {k: decode_fraction(v, k)
                for k, v in p["rates"].items()}, p["capacities"], p["external_memory"]))
        return cls(tuple(points))
