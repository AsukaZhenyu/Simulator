"""One explicit stateful service-response law for named execution resources.

This module converts external workload requests into a ResourceTimeline. It
does not schedule foreground tasks: the existing event evaluator does that.
CPU, GPU, copy engines and inclusive action responses use the same equation.

For competitor c, a normalized finite reservoir follows
    dq_c/dt = arrival_c * request_c - drain_c, clipped to [0, capacity_c].
Its pressure is activity_weight * active_fraction + backlog_weight * q/capacity.
Active fraction is one while q>0; at an empty reservoir it is
min(arrival*request/drain, 1). Thus continuously executed work can interfere
without building a backlog. Reaching capacity discards excess offered work;
this is a saturating proxy, not a lossless admission queue.

For each named target j the single law is
    g_j = base_j / (1 + sum(coefficient_jc * pressure_c)).
Every coefficient, weight, capacity and work rate is explicit and evidenced.
This reservoir is NOT identified as a literal CUDA/OS queue. In particular,
worker call duration and OFF-to-completion measurements alone do not prove
queue occupancy, arrival rate or physical service. Unknown parameters stay
unknown. There is no fitting, utilization-to-service conversion, device-type
branch, automatic contention arbitration or future interrupt inference here.
The measured replay actor is closed-loop (one outstanding call); this reservoir
does not assert an actor FIFO or solve the actor/foreground feedback system.

Nonnegative coefficients describe slowdown. An explicitly signed empirical
domain is allowed only for inclusive invocation responses; a negative value
describes facilitation or unresolved context, not physical negative contention.
Its denominator is proven positive over the full declared pressure box.
Inclusive responses must never be added to their internal physical stages.

Reservoir evolution is exact rational arithmetic. The continuous response is
approximated by explicitly bounded left-endpoint steps; each interval records a
conservative service-rate error bound and the accumulated work-supply bound.
These are discretization bounds, not empirical model-error or latency bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Mapping

from .evaluator import evaluate
from .graph import Graph, ResourcePoint, ResourceTimeline, Task, WorkDemand
from .parameters import (InstantaneousRate, InvalidModel, Parameter, ParameterBundle,
                         UnknownParameter, encode_fraction,
                         exact_fields, integer, nonempty, rational)

MODEL_SCHEMA = "stateful-service-response-model-1"
REQUEST_SCHEMA = "stateful-service-response-requests-1"
RESULT_SCHEMA = "stateful-service-response-result-1"
POINT_LIMIT = 200_000
RATE_UNITS = {"flop/s", "byte/s", "thread_cpu_ns/s", "invocation/s"}


def _references(value, name):
    if type(value) is not tuple or not value:
        raise InvalidModel(name + " requires a nonempty tuple")
    for item in value:
        nonempty(item, name)


def _signed(value, name):
    if type(value) not in {int, Fraction}:
        raise InvalidModel(name + " requires an integer or Fraction, never a float")
    return Fraction(value)


def _fraction_map(value, name):
    if not isinstance(value, Mapping):
        raise InvalidModel(name + " must be a mapping")
    result = {}
    for key, amount in value.items():
        nonempty(key, name + " key")
        result[key] = rational(amount, name + ":" + key)
    return MappingProxyType(result)


def _requests(value):
    if not isinstance(value, Mapping):
        raise InvalidModel("requests must be a mapping")
    copied = {}
    for key, request in value.items():
        nonempty(key, "competitor request name")
        if type(request) is not int or request not in (0, 1):
            raise InvalidModel("requests require exact integer 0/1, not utilization")
        copied[key] = request
    return MappingProxyType(copied)


def _decode(value, name):
    # Service-response signed coefficients use signed rational numerators;
    # existing physical-parameter decoding deliberately permits only >=0.
    obj = exact_fields(value, {"numerator", "denominator"}, name)
    if type(obj["numerator"]) is not int:
        raise InvalidModel(name + " numerator must be an integer")
    return Fraction(obj["numerator"], integer(obj["denominator"], name + " denominator", 1))


@dataclass(frozen=True)
class ServiceResponseParameter:
    name: str
    value: int | Fraction | None
    unit: str
    origin: str
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        nonempty(self.name, "service parameter name")
        if not isinstance(self.unit, str) or self.unit not in {"normalized_work", "normalized_work/ns", "dimensionless"}:
            raise InvalidModel("unsupported service-response parameter unit")
        if not isinstance(self.origin, str) or self.origin not in {"declared", "profiled"}:
            raise InvalidModel("service parameter origin must be declared or profiled")
        _references(self.evidence, "service parameter evidence")
        nonempty(self.scope, "service parameter scope")
        if self.value is not None:
            object.__setattr__(self, "value", _signed(self.value, self.name) if self.unit == "dimensionless"
                               else rational(self.value, self.name))

    def to_dict(self):
        return dict(name=self.name, value=None if self.value is None else encode_fraction(self.value),
                    unit=self.unit, origin=self.origin, evidence=list(self.evidence), scope=self.scope)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"name", "value", "unit", "origin", "evidence", "scope"}, "service parameter")
        if type(obj["evidence"]) is not list:
            raise InvalidModel("service parameter evidence must be a list")
        return cls(obj["name"], None if obj["value"] is None else _decode(obj["value"], "service value"),
                   obj["unit"], obj["origin"], tuple(obj["evidence"]), obj["scope"])


@dataclass(frozen=True)
class BackgroundReservoir:
    name: str
    capacity_parameter: str
    arrival_rate_parameter: str
    drain_rate_parameter: str
    activity_weight_parameter: str
    backlog_weight_parameter: str

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            nonempty(getattr(self, key), "reservoir " + key)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class ServiceInterference:
    target_parameter: str
    competitor: str
    coefficient_parameter: str
    coefficient_domain: str = "nonnegative_contention"

    def __post_init__(self):
        for key in ("target_parameter", "competitor", "coefficient_parameter"):
            nonempty(getattr(self, key), "interference " + key)
        if not isinstance(self.coefficient_domain, str) or self.coefficient_domain not in {"nonnegative_contention", "signed_inclusive_response"}:
            raise InvalidModel("unknown interference coefficient domain")

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class ServiceResponseModel:
    context_sha256: str
    parameters: tuple[ServiceResponseParameter, ...]
    competitors: tuple[BackgroundReservoir, ...]
    interference: tuple[ServiceInterference, ...]
    assumptions: tuple[str, ...]
    source_qualified: bool = False

    def __post_init__(self):
        if (not isinstance(self.context_sha256, str) or len(self.context_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.context_sha256)):
            raise InvalidModel("exact context SHA256 required")
        for key, item_type in (("parameters", ServiceResponseParameter), ("competitors", BackgroundReservoir),
                               ("interference", ServiceInterference)):
            values = getattr(self, key)
            if type(values) is not tuple or not values or any(not isinstance(item, item_type) for item in values):
                raise InvalidModel(key + " requires a nonempty typed tuple")
        _references(self.assumptions, "model structure assumptions")
        if type(self.source_qualified) is not bool:
            raise InvalidModel("source_qualified must be a bool")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise InvalidModel("duplicate service parameter")
        names = {c.name for c in self.competitors}
        if len(names) != len(self.competitors):
            raise InvalidModel("duplicate competitor")
        pairs = {(i.target_parameter, i.competitor) for i in self.interference}
        targets = {i.target_parameter for i in self.interference}
        if len(pairs) != len(self.interference) or pairs != {(t, c) for t in targets for c in names}:
            raise InvalidModel("require one explicit coefficient for every target/competitor pair")
        for competitor in self.competitors:
            for field_name, unit, positive in (
                    ("capacity_parameter", "normalized_work", True),
                    ("arrival_rate_parameter", "normalized_work/ns", False),
                    ("drain_rate_parameter", "normalized_work/ns", True),
                    ("activity_weight_parameter", "dimensionless", False),
                    ("backlog_weight_parameter", "dimensionless", False)):
                parameter = self.parameter(getattr(competitor, field_name), unit)
                if parameter.value is not None and (parameter.value <= 0 if positive else parameter.value < 0):
                    raise InvalidModel("reservoir capacity/drain must be positive; arrival and weights nonnegative")
        for link in self.interference:
            parameter = self.parameter(link.coefficient_parameter, "dimensionless")
            if parameter.value is not None and parameter.value < 0 and link.coefficient_domain != "signed_inclusive_response":
                raise InvalidModel("negative coefficient requires explicit signed inclusive-response domain")

    def parameter(self, name, unit):
        for parameter in self.parameters:
            if parameter.name == name:
                if parameter.unit != unit:
                    raise InvalidModel(name + " has wrong service-response unit; expected " + unit)
                return parameter
        raise UnknownParameter("missing service-response parameter: " + name)

    def require(self, name, unit):
        parameter = self.parameter(name, unit)
        if parameter.value is None:
            raise UnknownParameter("unknown service-response parameter: " + name)
        return parameter.value

    def to_dict(self):
        return dict(schema_version=MODEL_SCHEMA, context_sha256=self.context_sha256,
                    parameters=[p.to_dict() for p in self.parameters],
                    competitors=[c.to_dict() for c in self.competitors],
                    interference=[i.to_dict() for i in self.interference],
                    assumptions=list(self.assumptions), source_qualified=self.source_qualified)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "context_sha256", "parameters", "competitors",
                                  "interference", "assumptions", "source_qualified"}, "service response model")
        if obj["schema_version"] != MODEL_SCHEMA:
            raise InvalidModel("unsupported service response model schema")
        for key in ("parameters", "competitors", "interference", "assumptions"):
            if type(obj[key]) is not list:
                raise InvalidModel(key + " must be a list")
        return cls(obj["context_sha256"], tuple(ServiceResponseParameter.from_dict(p) for p in obj["parameters"]),
            tuple(BackgroundReservoir(**exact_fields(c, BackgroundReservoir.__dataclass_fields__, "reservoir"))
                  for c in obj["competitors"]),
            tuple(ServiceInterference(**exact_fields(i, ServiceInterference.__dataclass_fields__, "interference"))
                  for i in obj["interference"]), tuple(obj["assumptions"]), obj["source_qualified"])


@dataclass(frozen=True)
class ServiceRequestPoint:
    at_ns: int
    requests: Mapping[str, int]

    def __post_init__(self):
        integer(self.at_ns, "request time")
        object.__setattr__(self, "requests", _requests(self.requests))

    def to_dict(self):
        return dict(at_ns=self.at_ns, requests=dict(self.requests))


@dataclass(frozen=True)
class ServiceRequestTrajectory:
    initial_requests: Mapping[str, int]
    initial_work: Mapping[str, int | Fraction]
    points: tuple[ServiceRequestPoint, ...]
    input_origin: str
    evidence: tuple[str, ...]
    clock_id: str
    origin_id: str

    def __post_init__(self):
        object.__setattr__(self, "initial_requests", _requests(self.initial_requests))
        object.__setattr__(self, "initial_work", _fraction_map(self.initial_work, "initial reservoir work"))
        if type(self.points) is not tuple or any(not isinstance(p, ServiceRequestPoint) for p in self.points):
            raise InvalidModel("request points must be a typed tuple")
        if any(a.at_ns >= b.at_ns for a, b in zip(self.points, self.points[1:])):
            raise InvalidModel("request times must be strictly increasing")
        if not isinstance(self.input_origin, str) or self.input_origin not in {"external_scenario", "predeclared_scenario", "observed_diagnostic"}:
            raise InvalidModel("explicit service-request input origin required")
        _references(self.evidence, "request/initial-state evidence")
        nonempty(self.clock_id, "service-request clock id")
        nonempty(self.origin_id, "service-request origin id")

    def to_dict(self):
        return dict(schema_version=REQUEST_SCHEMA, initial_requests=dict(self.initial_requests),
                    initial_work={k: encode_fraction(v) for k, v in self.initial_work.items()},
                    points=[p.to_dict() for p in self.points], input_origin=self.input_origin,
                    evidence=list(self.evidence), clock_id=self.clock_id, origin_id=self.origin_id)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "initial_requests", "initial_work", "points",
            "input_origin", "evidence", "clock_id", "origin_id"}, "service request trajectory")
        if obj["schema_version"] != REQUEST_SCHEMA:
            raise InvalidModel("unsupported service-request schema")
        if type(obj["initial_work"]) is not dict or type(obj["points"]) is not list or type(obj["evidence"]) is not list:
            raise InvalidModel("invalid request trajectory containers")
        return cls(obj["initial_requests"], {k: _decode(v, k) for k, v in obj["initial_work"].items()},
            tuple(ServiceRequestPoint(**exact_fields(p, {"at_ns", "requests"}, "request point")) for p in obj["points"]),
            obj["input_origin"], tuple(obj["evidence"]), obj["clock_id"], obj["origin_id"])


@dataclass(frozen=True)
class ServiceResponseSample:
    at_ns: int
    interval_end_ns: int
    requests: Mapping[str, int]
    backlog: Mapping[str, Fraction]
    pressure: Mapping[str, Fraction]
    rates: Mapping[str, Fraction]
    rate_error_bounds: Mapping[str, Fraction]

    def __post_init__(self):
        integer(self.at_ns, "service sample time")
        integer(self.interval_end_ns, "service interval end")
        if self.interval_end_ns < self.at_ns:
            raise InvalidModel("service sample interval moves backwards")
        object.__setattr__(self, "requests", _requests(self.requests))
        for name in ("backlog", "pressure", "rates", "rate_error_bounds"):
            object.__setattr__(self, name, _fraction_map(getattr(self, name), name))

    def to_dict(self):
        return dict(at_ns=self.at_ns, interval_end_ns=self.interval_end_ns, requests=dict(self.requests),
                    **{name: {k: encode_fraction(v) for k, v in getattr(self, name).items()}
                       for name in ("backlog", "pressure", "rates", "rate_error_bounds")})

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, cls.__dataclass_fields__, "service response sample")
        for name in ("backlog", "pressure", "rates", "rate_error_bounds"):
            if type(obj[name]) is not dict:
                raise InvalidModel(name + " must be a dictionary")
        return cls(obj["at_ns"], obj["interval_end_ns"], obj["requests"],
                   *({k: _decode(v, k) for k, v in obj[name].items()}
                     for name in ("backlog", "pressure", "rates", "rate_error_bounds")))


@dataclass(frozen=True)
class ServiceResponseResult:
    model: ServiceResponseModel
    base_bundle: ParameterBundle
    request_trajectory: ServiceRequestTrajectory
    baseline_timeline: ResourceTimeline
    timeline: ResourceTimeline
    samples: tuple[ServiceResponseSample, ...]
    valid_until_ns: int
    max_step_ns: int
    max_points: int
    integrated_service_error_bound_work: Mapping[str, Fraction]
    diagnostic_replay: bool
    accuracy_qualified: bool = field(default=False, init=False)
    _times: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _minimum_base_rates: Mapping[str, Fraction | InstantaneousRate | None] = field(init=False, repr=False, compare=False)
    _serial_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "integrated_service_error_bound_work",
                           _fraction_map(self.integrated_service_error_bound_work, "integrated service error bound"))
        object.__setattr__(self, "_times", tuple(point.at_ns for point in self.timeline.points))
        minima = {p.name: p.value for p in self.base_bundle.parameters if p.unit in RATE_UNITS}
        for point in self.baseline_timeline.points:
            if point.at_ns > self.valid_until_ns:
                break
            for name, value in point.rates.items():
                old = minima[name]
                if old is None:
                    continue
                if isinstance(old, InstantaneousRate):
                    minima[name] = value
                elif not isinstance(value, InstantaneousRate):
                    minima[name] = min(old, value)
        object.__setattr__(self, "_minimum_base_rates", MappingProxyType(minima))

    def window(self, start_ns: int, end_ns: int) -> ResourceTimeline:
        """Slice precomputed g without replaying/resetting reservoir history."""
        integer(start_ns, "execution entry time"); integer(end_ns, "execution window end")
        if not 0 <= start_ns <= end_ns <= self.valid_until_ns:
            raise UnknownParameter("execution window exceeds service trajectory horizon")
        left, right = bisect_right(self._times, start_ns), bisect_right(self._times, end_ns)
        rates, capacities, external = {}, {}, {}
        for point in self.baseline_timeline.points:
            if point.at_ns > start_ns:
                break
            rates.update(point.rates); capacities.update(point.capacities); external.update(point.external_memory)
        # Every g sample contains all transformed target rates; unrelated state
        # is reconstructed from the sparse explicit baseline, not every g step.
        rates.update(self.timeline.points[left - 1].rates)
        future = tuple(ResourcePoint(p.at_ns - start_ns, p.rates, p.capacities, p.external_memory)
                       for p in self.timeline.points[left:right])
        return ResourceTimeline((ResourcePoint(0, rates, capacities, external), *future))

    def evaluate(self, graph: Graph, *, start_ns: int = 0, end_ns: int | None = None):
        """Delegate execution to the common engine, conditional on explicit entry.

        The timeline is valid only on [0,valid_until_ns]. This wrapper rejects
        extrapolation instead of inventing a permanent final state or blackout.
        Direct users of ``timeline`` must enforce the same horizon themselves.
        """
        end_ns = self.valid_until_ns if end_ns is None else end_ns
        relative = self.window(start_ns, end_ns)
        result = evaluate(graph, self.base_bundle, relative)
        if start_ns + result.duration_ns > end_ns:
            raise UnknownParameter("execution finish exceeds service trajectory horizon; no rate extrapolation accepted")
        return result

    def evaluate_serial_response(self, action_parameters: tuple[str, ...], *, start_ns: int = 0):
        """Build serial complete-response tasks, including explicit gaps/tail.

        Named parameters must be invocation/s or fixed ns. Physical work cannot
        be inferred from complete wall times and requires an explicit Graph.
        """
        if type(action_parameters) is not tuple or not action_parameters:
            raise InvalidModel("ordered response parameters require a nonempty tuple")
        for name in action_parameters:
            nonempty(name, "ordered response parameter")
        cached = self._serial_cache.get(action_parameters)
        if cached is not None:
            graph, bound_ns = cached
            return self.evaluate(graph, start_ns=start_ns,
                                 end_ns=min(self.valid_until_ns, start_ns + bound_ns))
        tasks = []
        bound_ns = 0
        for index, name in enumerate(action_parameters):
            parameter = self.base_bundle.get(name)
            dependencies = (tasks[-1].id,) if tasks else ()
            if parameter.unit == "invocation/s":
                item = Task("response:" + str(index), "serial_completed_response",
                            (WorkDemand(name, 1, "invocation"),), dependencies)
                rate = self._minimum_base_rates[name]
                if rate is None:
                    raise UnknownParameter("unknown baseline response: " + name)
                if isinstance(rate, InstantaneousRate):
                    duration = 0
                elif rate == 0:
                    raise UnknownParameter("zero baseline response has no finite support window")
                else:
                    denominator = 1
                    for link in self.model.interference:
                        if link.target_parameter == name:
                            k = self.model.require(link.coefficient_parameter, "dimensionless")
                            competitor = next(c for c in self.model.competitors if c.name == link.competitor)
                            pressure_max = (self.model.require(competitor.activity_weight_parameter, "dimensionless")
                                            + self.model.require(competitor.backlog_weight_parameter, "dimensionless"))
                            denominator += max(Fraction(0), k) * pressure_max
                    duration = math.ceil(Fraction(1_000_000_000) * denominator / rate)
            elif parameter.unit == "ns":
                item = Task("response:" + str(index), "serial_completed_response",
                            depends_on=dependencies, fixed_parameter=name)
                duration = math.ceil(self.base_bundle.require(name, "ns"))
            else:
                raise InvalidModel("serial complete responses require invocation/s or ns, not physical service")
            tasks.append(item)
            bound_ns += duration
        graph = Graph(tuple(tasks))
        self._serial_cache[action_parameters] = (graph, bound_ns)
        return self.evaluate(graph, start_ns=start_ns,
                             end_ns=min(self.valid_until_ns, start_ns + bound_ns))

    def to_dict(self):
        return dict(schema_version=RESULT_SCHEMA, model=self.model.to_dict(), base_bundle=self.base_bundle.to_dict(),
            request_trajectory=self.request_trajectory.to_dict(), baseline_timeline=self.baseline_timeline.to_dict(),
            timeline=self.timeline.to_dict(),
            samples=[s.to_dict() for s in self.samples], valid_until_ns=self.valid_until_ns,
            max_step_ns=self.max_step_ns, max_points=self.max_points,
            integrated_service_error_bound_work={k: encode_fraction(v) for k, v in self.integrated_service_error_bound_work.items()},
            diagnostic_replay=self.diagnostic_replay, accuracy_qualified=False,
            approximation="left-endpoint piecewise-constant service; exact clipped reservoir evolution; integer-ns event ceiling <=1ns",
            bound_scope="service discretization only, not calibrated model error or a whole-graph latency guarantee")

    @classmethod
    def from_dict(cls, value):
        fields = {"schema_version", "model", "base_bundle", "request_trajectory", "baseline_timeline", "timeline", "samples",
            "valid_until_ns", "max_step_ns", "max_points", "integrated_service_error_bound_work",
            "diagnostic_replay", "accuracy_qualified", "approximation", "bound_scope"}
        obj = exact_fields(value, fields, "service response result")
        if obj["schema_version"] != RESULT_SCHEMA or obj["accuracy_qualified"] is not False:
            raise InvalidModel("unsupported or accuracy-qualified service-response artifact")
        # Reconstruct the model law and compare every derived sample/timeline;
        # deserialization cannot silently accept a modified prediction artifact.
        result = build_service_response(ServiceResponseModel.from_dict(obj["model"]),
            ParameterBundle.from_dict(obj["base_bundle"]), ServiceRequestTrajectory.from_dict(obj["request_trajectory"]),
            end_ns=obj["valid_until_ns"], max_step_ns=obj["max_step_ns"], max_points=obj["max_points"],
            baseline_timeline=ResourceTimeline.from_dict(obj["baseline_timeline"]),
            allow_observed_diagnostic=obj["diagnostic_replay"])
        if result.to_dict() != obj:
            raise InvalidModel("serialized service response differs from model reconstruction")
        return result


def build_service_response(model: ServiceResponseModel, base_bundle: ParameterBundle,
                           trajectory: ServiceRequestTrajectory, *, end_ns: int,
                           max_step_ns: int, max_points: int,
                           baseline_timeline: ResourceTimeline | None = None,
                           allow_observed_diagnostic: bool = False) -> ServiceResponseResult:
    """Construct one shared g trajectory; execution remains in ``evaluate``.

    Target baseline prices must exclude the interference represented here.
    ``baseline_timeline`` uses the same explicit ns clock/origin as requests;
    unbound rates and memory changes are preserved inside the finite horizon.
    The full explicit coefficient matrix is checked, including zero cross terms.
    """
    if not isinstance(model, ServiceResponseModel) or not isinstance(base_bundle, ParameterBundle) or not isinstance(trajectory, ServiceRequestTrajectory):
        raise InvalidModel("service response requires model, base ParameterBundle and request trajectory")
    integer(end_ns, "service horizon", 1); integer(max_step_ns, "maximum service step", 1)
    integer(max_points, "maximum service points", 2)
    if max_points > POINT_LIMIT:
        raise InvalidModel("explicit service point budget exceeded")
    if type(allow_observed_diagnostic) is not bool:
        raise InvalidModel("allow_observed_diagnostic must be a bool")
    observed = trajectory.input_origin == "observed_diagnostic"
    if observed and not allow_observed_diagnostic:
        raise InvalidModel("observed diagnostic requests/initial work require explicit replay, not forecast input")
    baseline = ResourceTimeline() if baseline_timeline is None else baseline_timeline
    if not isinstance(baseline, ResourceTimeline):
        raise InvalidModel("baseline_timeline must be ResourceTimeline")
    competitors = {c.name: c for c in model.competitors}
    names = set(competitors)
    if set(trajectory.initial_requests) != names or set(trajectory.initial_work) != names or any(
            not set(point.requests) <= names for point in trajectory.points):
        raise InvalidModel("explicit complete initial requests/work and known competitor updates required")
    settings = {}
    for name, competitor in competitors.items():
        values = tuple(model.require(getattr(competitor, key), unit) for key, unit in (
            ("capacity_parameter", "normalized_work"), ("arrival_rate_parameter", "normalized_work/ns"),
            ("drain_rate_parameter", "normalized_work/ns"), ("activity_weight_parameter", "dimensionless"),
            ("backlog_weight_parameter", "dimensionless")))
        if trajectory.initial_work[name] > values[0]:
            raise InvalidModel("initial reservoir work exceeds declared capacity")
        settings[name] = values
    links = {}
    for link in model.interference:
        parameter = base_bundle.get(link.target_parameter)
        if parameter.unit not in RATE_UNITS:
            raise InvalidModel("interference targets must name explicit service-rate parameters")
        if link.coefficient_domain == "signed_inclusive_response" and parameter.unit != "invocation/s":
            raise InvalidModel("signed empirical response coefficients require invocation/s; physical contention stays nonnegative")
        value = model.require(link.coefficient_parameter, "dimensionless")
        links.setdefault(link.target_parameter, {})[link.competitor] = value
    base_rates = {}
    for target, coefficients in links.items():
        parameter = base_bundle.get(target)
        value = base_bundle.require(target, parameter.unit)
        if isinstance(value, InstantaneousRate):
            raise InvalidModel("instantaneous response cannot identify an interference coefficient")
        base_rates[target] = value
        lower = 1 + sum(k * (settings[c][3] + settings[c][4]) for c, k in coefficients.items() if k < 0)
        if lower <= 0:
            raise InvalidModel("signed response denominator is not positive over the full allowed pressure domain")
    for point in baseline.points:
        for name, value in point.rates.items():
            source = base_bundle.get(name)
            if source.unit not in RATE_UNITS:
                raise InvalidModel("baseline may only update service rates")
            Parameter(name, value, source.unit, source.origin, source.evidence, source.scope)
            if name in links and isinstance(value, InstantaneousRate):
                raise InvalidModel("instantaneous baseline cannot be an interference target")

    requests = dict(trajectory.initial_requests)
    backlog = dict(trajectory.initial_work)
    request_points = {p.at_ns: p for p in trajectory.points if p.at_ns <= end_ns}
    baseline_points = {p.at_ns: p for p in baseline.points if p.at_ns <= end_ns}
    external_times = sorted(set(request_points) | set(baseline_points) | {end_ns})
    samples, points = [], []
    error_work = {target: Fraction(0) for target in links}

    def pressure(name, q):
        capacity, arrival, drain, active_weight, backlog_weight = settings[name]
        fraction = Fraction(1) if q > 0 else min(Fraction(1), arrival * requests[name] / drain)
        return active_weight * fraction + backlog_weight * q / capacity

    now = 0
    while True:
        if len(samples) >= max_points:
            raise InvalidModel("explicit service point budget exceeded at reservoir/request boundaries")
        if now in request_points:
            requests.update(request_points[now].requests)
        source = baseline_points.get(now)
        updates = dict(source.rates) if source else {}
        base_rates.update({k: v for k, v in updates.items() if k in links})
        pressures = {name: pressure(name, backlog[name]) for name in names}
        rates = {target: base_rates[target] / (1 + sum(k * pressures[c] for c, k in coefficients.items()))
                 for target, coefficients in links.items()}
        if now == end_ns:
            next_time, next_backlog = now, backlog
        else:
            candidates = [end_ns]
            next_external = bisect_right(external_times, now)
            if next_external < len(external_times):
                candidates.append(external_times[next_external])
            for name, q in backlog.items():
                capacity, arrival, drain, _, _ = settings[name]
                net = arrival * requests[name] - drain
                if net > 0 and q < capacity:
                    candidates.append(now + max(1, math.ceil((capacity - q) / net)))
                elif net < 0 and q > 0:
                    candidates.append(now + max(1, math.ceil(q / -net)))
                changing = (net > 0 and q < capacity) or (net < 0 and q > 0)
                if changing and settings[name][4] > 0 and any(coefficients[name] != 0 for coefficients in links.values()):
                    candidates.append(now + max_step_ns)
            next_time = min(candidates)
            dt = next_time - now
            next_backlog = {name: min(settings[name][0], max(Fraction(0), q +
                            (settings[name][1] * requests[name] - settings[name][2]) * dt))
                            for name, q in backlog.items()}
        next_pressure = {name: pressure(name, next_backlog[name]) for name in names}
        errors = {}
        for target, coefficients in links.items():
            lo = {c: min(pressures[c], next_pressure[c]) for c in names}
            hi = {c: max(pressures[c], next_pressure[c]) for c in names}
            denominator_lo = 1 + sum(k * (lo[c] if k >= 0 else hi[c]) for c, k in coefficients.items())
            denominator_hi = 1 + sum(k * (hi[c] if k >= 0 else lo[c]) for c, k in coefficients.items())
            errors[target] = max(abs(rates[target] - base_rates[target] / denominator_lo),
                                 abs(rates[target] - base_rates[target] / denominator_hi))
            error_work[target] += errors[target] * (next_time - now) / 1_000_000_000
        updates.update(rates)
        points.append(ResourcePoint(now, updates, source.capacities if source else {},
                                    source.external_memory if source else {}))
        samples.append(ServiceResponseSample(now, next_time, requests, backlog, pressures, rates, errors))
        if now == end_ns:
            break
        now, backlog = next_time, next_backlog
    return ServiceResponseResult(model, base_bundle, trajectory, baseline, ResourceTimeline(tuple(points)), tuple(samples),
                                 end_ns, max_step_ns, max_points, error_work, observed)
