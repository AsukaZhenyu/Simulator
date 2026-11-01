"""Physical progress under externally supplied LLM-available resources.

The external trajectory already includes operating-system/background effects.
This module neither infers supply from utilization nor applies an additional
background-pressure multiplier. Sequential boundaries, own resource contention,
completion gates and memory reservations remain work of the common evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .evaluator import BlockedExecution, Evaluation, evaluate
from .graph import Graph, ResourcePoint, ResourceTimeline
from .parameters import (
    InstantaneousRate, InvalidModel, Parameter, ParameterBundle, UnknownParameter,
    decode_fraction, encode_fraction, exact_fields, integer, nonempty, rational,
)

PHYSICAL_RATE_UNITS = frozenset({"flop/s", "byte/s", "thread_cpu_ns/s", "cpu_cycle/s"})
SOURCE_KINDS = frozenset({"declared_scenario", "independent_measurement", "observed_diagnostic"})


def _evidence(origin, evidence, scope):
    if origin not in {"declared", "profiled"}:
        raise InvalidModel("binding origin must be declared or profiled")
    if type(evidence) is not tuple or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise InvalidModel("evidence must be a tuple of nonempty references")
    if origin == "profiled" and not evidence:
        raise InvalidModel("profiled efficiency requires evidence")
    nonempty(scope, "efficiency scope")


@dataclass(frozen=True)
class ServiceBinding:
    """Scoped action efficiency against an absolute physical supply.

    ``rate[target](t) = efficiency * available[supply](t)`` is the initial,
    explicit service law. Different shape/backend scopes require separately
    evidenced bindings. ``None`` is unknown; zero is an explicitly known
    absence of useful progress. Inclusive invocation wall times are not
    physical efficiencies and are not accepted by this interface.
    """

    target_parameter: str
    supply_parameter: str
    efficiency: int | Fraction | None
    origin: str
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        nonempty(self.target_parameter, "binding target")
        nonempty(self.supply_parameter, "binding supply")
        _evidence(self.origin, self.evidence, self.scope)
        if self.efficiency is not None:
            value = rational(self.efficiency, "physical efficiency")
            if value > 1:
                raise InvalidModel("physical efficiency must be in [0,1]; do not clip invalid estimates")
            object.__setattr__(self, "efficiency", value)

    def to_dict(self):
        return dict(target_parameter=self.target_parameter, supply_parameter=self.supply_parameter,
                    efficiency=None if self.efficiency is None else encode_fraction(self.efficiency),
                    origin=self.origin, evidence=list(self.evidence), scope=self.scope)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, cls.__dataclass_fields__, "service binding")
        if not isinstance(obj["evidence"], list):
            raise InvalidModel("binding evidence must be a JSON list")
        return cls(obj["target_parameter"], obj["supply_parameter"],
                   None if obj["efficiency"] is None else decode_fraction(obj["efficiency"], "efficiency"),
                   obj["origin"], tuple(obj["evidence"]), obj["scope"])


@dataclass(frozen=True)
class AvailableResources:
    """One finite trajectory of absolute resources available to this LLM.

    Rates use physical units, not total utilization or background requests.
    Byte parameters are already-available memory budgets: external/background
    bytes must not be subtracted again. ``ResourcePoint.capacities`` names the
    corresponding Graph memory domains; rate keys name supply parameters.
    Sparse updates are piecewise constant on the explicitly bounded horizon.
    Declaring an independently measured source is provenance, not an automatic
    accuracy qualification or proof of measurement independence.
    """

    supply: ParameterBundle
    timeline: ResourceTimeline
    valid_until_ns: int
    clock_id: str
    clock_origin: str
    source_kind: str
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        if not isinstance(self.supply, ParameterBundle) or not isinstance(self.timeline, ResourceTimeline):
            raise InvalidModel("available resources require ParameterBundle and ResourceTimeline")
        integer(self.valid_until_ns, "available-resource horizon", 1)
        nonempty(self.clock_id, "available-resource clock")
        nonempty(self.clock_origin, "available-resource origin")
        if self.source_kind not in SOURCE_KINDS:
            raise InvalidModel("unsupported available-resource source kind")
        _evidence("profiled" if self.source_kind != "declared_scenario" else "declared", self.evidence, self.scope)
        for parameter in self.supply.parameters:
            if parameter.unit not in PHYSICAL_RATE_UNITS | {"byte"} or isinstance(parameter.value, InstantaneousRate):
                raise InvalidModel("external supply requires physical rates or byte capacity, not inclusive responses")
        for point in self.timeline.points:
            if point.at_ns > self.valid_until_ns:
                raise InvalidModel("resource point exceeds the declared availability horizon")
            if point.external_memory:
                raise InvalidModel("available budgets already exclude external memory; do not subtract it twice")
            for name, value in point.rates.items():
                parameter = self.supply.get(name)
                if parameter.unit not in PHYSICAL_RATE_UNITS or isinstance(value, InstantaneousRate):
                    raise InvalidModel("external rate changes require physical supply parameters")
                if parameter.unit == "thread_cpu_ns/s" and value > 1_000_000_000:
                    raise InvalidModel("one thread cannot receive more than 1e9 active CPU ns/s")

    def _check_window(self, start_ns, end_ns):
        integer(start_ns, "availability window start")
        integer(end_ns, "availability window end")
        if not start_ns <= end_ns <= self.valid_until_ns:
            raise UnknownParameter("requested window exceeds the available-resource horizon")

    def value_at(self, supply_parameter: str, at_ns: int):
        """Known physical rate at an absolute external-clock timestamp."""
        self._check_window(at_ns, at_ns)
        parameter = self.supply.get(supply_parameter)
        if parameter.unit not in PHYSICAL_RATE_UNITS:
            raise InvalidModel("value_at requires a physical rate, not a memory budget")
        value = parameter.value
        for point in self.timeline.points:
            if point.at_ns > at_ns:
                break
            value = point.rates.get(supply_parameter, value)
        if value is None:
            raise UnknownParameter("unknown external available service: " + supply_parameter)
        return value

    def service_integral(self, supply_parameter: str, start_ns: int, end_ns: int) -> Fraction:
        """Exact supplied work, not an estimate from target completion times."""
        self._check_window(start_ns, end_ns)
        rate = self.value_at(supply_parameter, start_ns)
        total, cursor = Fraction(0), start_ns
        for point in self.timeline.points:
            if not start_ns < point.at_ns < end_ns:
                continue
            total += rate * Fraction(point.at_ns - cursor, 1_000_000_000)
            rate = point.rates.get(supply_parameter, rate)
            cursor = point.at_ns
        return total + rate * Fraction(end_ns - cursor, 1_000_000_000)

    def evaluate(self, graph: Graph, fixed_parameters: ParameterBundle,
                 bindings: tuple[ServiceBinding, ...], **kwargs):
        return evaluate_available_resources(graph, fixed_parameters, self, bindings, **kwargs)

    def to_dict(self):
        return dict(schema_version="available-resources-1", supply=self.supply.to_dict(),
                    timeline=self.timeline.to_dict(), valid_until_ns=self.valid_until_ns,
                    clock_id=self.clock_id, clock_origin=self.clock_origin, source_kind=self.source_kind,
                    evidence=list(self.evidence), scope=self.scope)

    @classmethod
    def from_dict(cls, value):
        names = set(cls.__dataclass_fields__) | {"schema_version"}
        obj = exact_fields(value, names, "available resources")
        if obj["schema_version"] != "available-resources-1" or not isinstance(obj["evidence"], list):
            raise InvalidModel("unsupported available-resource artifact")
        return cls(ParameterBundle.from_dict(obj["supply"]), ResourceTimeline.from_dict(obj["timeline"]),
                   obj["valid_until_ns"], obj["clock_id"], obj["clock_origin"], obj["source_kind"],
                   tuple(obj["evidence"]), obj["scope"])


@dataclass(frozen=True)
class AvailableResourceEvaluation:
    """Result plus the ordinary common-engine inputs used to calculate it."""

    graph: Graph
    parameters: ParameterBundle
    timeline: ResourceTimeline
    evaluation: Evaluation
    available: AvailableResources
    bindings: tuple[ServiceBinding, ...]
    fixed_parameters: ParameterBundle
    start_ns: int
    end_ns: int
    diagnostic_replay: bool

    @property
    def duration_ns(self):
        return self.evaluation.duration_ns

    @property
    def tasks(self):
        return self.evaluation.tasks

    @property
    def waits(self):
        return self.evaluation.waits

    @property
    def events(self):
        return self.evaluation.events

    @property
    def accuracy_qualified(self):
        return False

    def to_dict(self):
        return dict(schema_version="available-resource-evaluation-1", graph=self.graph.to_dict(),
                    parameters=self.parameters.to_dict(), timeline=self.timeline.to_dict(),
                    evaluation=self.evaluation.to_dict(), available=self.available.to_dict(),
                    bindings=[b.to_dict() for b in self.bindings], fixed_parameters=self.fixed_parameters.to_dict(),
                    start_ns=self.start_ns, end_ns=self.end_ns, diagnostic_replay=self.diagnostic_replay,
                    task_time_scope="execution-relative ns; absolute timestamp is start_ns + task time",
                    service_law="physical effective rate = explicit efficiency * external available service",
                    external_background_applied_again=False, accuracy_qualified=False)


def _validate_shared_supply(graph, binding_by_target):
    users = {}
    # A valid DAG may be stored in reverse topological order and have much
    # deeper chains than Python's recursion limit. Compute transitive ancestry
    # iteratively. Integer bit sets also avoid a separate Python set of every
    # ancestor for every stage in long serialized graphs.
    indices = {task.id: index for index, task in enumerate(graph.tasks)}
    ancestry = {task.id: 0 for task in graph.tasks}
    children = {task.id: [] for task in graph.tasks}
    remaining = {}
    ready = []
    for task in graph.tasks:
        direct = set(task.depends_on + task.waits_for)
        remaining[task.id] = len(direct)
        if not direct:
            ready.append(task.id)
        for parent in direct:
            children[parent].append(task.id)
    cursor = 0
    while cursor < len(ready):
        parent = ready[cursor]
        cursor += 1
        inherited = ancestry[parent] | (1 << indices[parent])
        for child in children[parent]:
            ancestry[child] |= inherited
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    for task in graph.tasks:
        sources = []
        for demand in task.demands:
            if demand.unit == "invocation":
                raise InvalidModel("inclusive invocation work cannot be used as external physical service")
            try:
                binding = binding_by_target[demand.parameter]
            except KeyError:
                raise UnknownParameter("no external service binding for " + demand.parameter) from None
            sources.append(binding.supply_parameter)
        if len(sources) != len(set(sources)):
            raise InvalidModel("one task cannot duplicate a shared supply; combine its physical work first")
        for source in sources:
            users.setdefault(source, []).append(task)
    for source, tasks in users.items():
        for index, left in enumerate(tasks):
            for right in tasks[index + 1:]:
                if ancestry[right.id] & (1 << indices[left.id]) or ancestry[left.id] & (1 << indices[right.id]):
                    continue
                if not set((left.resource,) + left.extra_resources).intersection((right.resource,) + right.extra_resources):
                    raise InvalidModel("shared supply " + source + " requires a common physical resource lock; "
                                       "declare partitioned supplies for independent lanes")


def evaluate_available_resources(graph: Graph, fixed_parameters: ParameterBundle,
                                 available: AvailableResources, bindings: tuple[ServiceBinding, ...], *,
                                 start_ns: int = 0, end_ns: int | None = None,
                                 allow_observed_diagnostic: bool = False) -> AvailableResourceEvaluation:
    """Evaluate physical work on one finite, externally supplied time axis.

    Fixed ``ns`` overhead and zero-work completion gates retain common-engine
    semantics. Multiple demands in a Task remain an explicit roofline maximum;
    sequential mechanisms require distinct Tasks. Shared supply aliases require
    common locks, rather than each lane silently receiving the entire budget.
    This does not infer proportional sharing or operating-system scheduling.
    """
    if not isinstance(graph, Graph) or not isinstance(fixed_parameters, ParameterBundle) or not isinstance(available, AvailableResources):
        raise InvalidModel("requires Graph, fixed ParameterBundle and AvailableResources")
    if type(bindings) is not tuple or any(not isinstance(b, ServiceBinding) for b in bindings):
        raise InvalidModel("bindings must be a tuple of ServiceBinding")
    if type(allow_observed_diagnostic) is not bool:
        raise InvalidModel("diagnostic replay option must be boolean")
    diagnostic = available.source_kind == "observed_diagnostic"
    if diagnostic and not allow_observed_diagnostic:
        raise InvalidModel("observed diagnostic service requires explicit replay permission")
    end_ns = available.valid_until_ns if end_ns is None else end_ns
    available._check_window(start_ns, end_ns)
    by_target = {binding.target_parameter: binding for binding in bindings}
    if len(by_target) != len(bindings):
        raise InvalidModel("duplicate service-binding target")
    _validate_shared_supply(graph, by_target)
    if any(p.unit != "ns" for p in fixed_parameters.parameters):
        raise InvalidModel("fixed_parameters contains only explicitly scoped fixed ns overhead")
    parameters = list(fixed_parameters.parameters)
    domains = {domain.name: domain for domain in graph.memory_domains}
    if len({domain.capacity_parameter for domain in domains.values()}) != len(domains):
        raise InvalidModel("separate memory domains require separate available budgets; use one domain for shared memory")
    if any(domain.external_bytes for domain in domains.values()):
        raise InvalidModel("available memory budgets already exclude external allocations")
    for point in available.timeline.points:
        if not set(point.capacities) <= domains.keys():
            raise InvalidModel("external capacity update references an undeclared graph memory domain")
    for domain in domains.values():
        original = available.supply.get(domain.capacity_parameter)
        if original.unit != "byte":
            raise InvalidModel("memory budget requires byte units")
        value = original.value
        for point in available.timeline.points:
            if point.at_ns > start_ns:
                break
            value = point.capacities.get(domain.name, value)
        if value is None:
            raise UnknownParameter("unknown available memory budget: " + domain.name)
        if not any(p.name == original.name for p in parameters):
            parameters.append(Parameter(original.name, value, "byte", original.origin,
                                        original.evidence, original.scope))
    for binding in bindings:
        supply = available.supply.get(binding.supply_parameter)
        if supply.unit not in PHYSICAL_RATE_UNITS:
            raise InvalidModel("service binding requires a physical rate supply")
        if binding.efficiency is None:
            raise UnknownParameter("unknown action efficiency: " + binding.target_parameter)
        value = available.value_at(binding.supply_parameter, start_ns) * binding.efficiency
        evidence = binding.evidence + available.evidence + supply.evidence
        parameters.append(Parameter(binding.target_parameter, value, supply.unit, binding.origin,
                                    evidence, binding.scope + "; external available-resource trajectory"))
    bundle = ParameterBundle(tuple(parameters))
    points = [ResourcePoint(0)]
    for point in available.timeline.points:
        if not start_ns < point.at_ns <= end_ns:
            continue
        rates = {binding.target_parameter: point.rates[binding.supply_parameter] * binding.efficiency
                 for binding in bindings if binding.supply_parameter in point.rates}
        if rates or point.capacities:
            points.append(ResourcePoint(point.at_ns - start_ns, rates, point.capacities))
    timeline = ResourceTimeline(tuple(points))
    try:
        result = evaluate(graph, bundle, timeline)
    except BlockedExecution as error:
        raise UnknownParameter("execution cannot complete within the known available-resource horizon: " + str(error)) from error
    if start_ns + result.duration_ns > end_ns:
        raise UnknownParameter("execution finish exceeds available-resource horizon; extrapolation rejected")
    return AvailableResourceEvaluation(graph, bundle, timeline, result, available, bindings,
                                       fixed_parameters, start_ns, end_ns, diagnostic)
