"""Finite periodic external-service scenarios, not an operating-system model.

Period, blackout width and phase are explicit inputs measured by profiling or
declared by a scenario. They are never inferred from target completion, replay
ON/OFF labels or utilization. An unknown phase remains unknown. Observed target
replays need explicit diagnostic permission and cannot certify prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evaluator import evaluate
from .graph import Graph, ResourcePoint, ResourceTimeline
from .parameters import InstantaneousRate, InvalidModel, Parameter, ParameterBundle, UnknownParameter, integer, nonempty


SOURCE_KINDS = frozenset({"declared_scenario", "independent_measurement", "observed_diagnostic"})


@dataclass(frozen=True)
class PeriodicServiceCalendar:
    parameter: str
    period_ns: int | None
    blackout_ns: int | None
    phase_ns: int | None
    valid_until_ns: int
    source_kind: str
    evidence: tuple[str, ...]
    scope: str
    clock_id: str
    clock_origin: str

    def __post_init__(self):
        nonempty(self.parameter, "calendar service parameter")
        nonempty(self.scope, "calendar scope")
        nonempty(self.clock_id, "calendar clock")
        nonempty(self.clock_origin, "calendar clock origin")
        integer(self.valid_until_ns, "calendar horizon", 1)
        if self.source_kind not in SOURCE_KINDS:
            raise InvalidModel("calendar source must be explicitly declared/measured/diagnostic")
        if type(self.evidence) is not tuple or not self.evidence:
            raise InvalidModel("calendar evidence required")
        for item in self.evidence:
            nonempty(item, "calendar evidence")
        for key in ("period_ns", "blackout_ns", "phase_ns"):
            value = getattr(self, key)
            if value is not None:
                integer(value, key, 1 if key == "period_ns" else 0)
        if self.period_ns is not None:
            if self.blackout_ns is not None and self.blackout_ns > self.period_ns:
                raise InvalidModel("blackout width cannot exceed period")
            if self.phase_ns is not None and self.phase_ns >= self.period_ns:
                raise InvalidModel("phase must be in [0, period)")

    def blackout_at(self, at_ns):
        integer(at_ns, "calendar timestamp")
        for key in ("period_ns", "blackout_ns", "phase_ns"):
            if getattr(self, key) is None:
                raise UnknownParameter("unknown calendar " + key)
        if at_ns > self.valid_until_ns:
            raise UnknownParameter("calendar timestamp exceeds the finite horizon")
        return (at_ns - self.phase_ns) % self.period_ns < self.blackout_ns


@dataclass(frozen=True)
class CompiledServiceCalendars:
    timeline: ResourceTimeline
    valid_until_ns: int
    calendars: tuple[PeriodicServiceCalendar, ...]
    diagnostic_replay: bool

    def to_dict(self):
        from dataclasses import asdict
        return dict(schema_version="periodic-external-service-calendars-1",
                    timeline=self.timeline.to_dict(), valid_until_ns=self.valid_until_ns,
                    calendars=[asdict(c) for c in self.calendars], diagnostic_replay=self.diagnostic_replay,
                    clock_id=self.calendars[0].clock_id, clock_origin=self.calendars[0].clock_origin,
                    external_service_scenario=True, operating_system_simulated=False,
                    phase_inferred_here=False, hardware_accuracy_qualified=False)

    @property
    def hardware_accuracy_qualified(self):
        return False

    def evaluate(self, graph: Graph, parameters: ParameterBundle):
        result = evaluate(graph, parameters, self.timeline)
        if result.duration_ns > self.valid_until_ns:
            raise UnknownParameter("execution exceeds explicitly supplied service-calendar horizon")
        return result


def compile_service_calendars(parameters: ParameterBundle, calendars: tuple[PeriodicServiceCalendar, ...],
                              *, baseline_timeline=None, allow_observed_diagnostic=False):
    """Mask explicit rates during periodic half-open blackouts and restore them.

    Blackouts are [phase + k*period, phase + k*period + width), for integer k;
    a previous-period blackout can therefore cover time zero. Baseline rate
    updates inside a blackout update the stored base, not its masked zero. The
    common evaluator still handles every dependent task and work integration.
    Calendar and baseline times must already be in the same evaluation-relative
    ns domain. The common clock/origin names are checked across calendars; this
    function does not convert/align raw timestamps or certify reference probes
    as neutral resource capacity. A measured label is provenance, not accuracy.
    """
    if not isinstance(parameters, ParameterBundle):
        raise InvalidModel("service calendar requires ParameterBundle")
    if type(calendars) is not tuple or not calendars or any(
            not isinstance(c, PeriodicServiceCalendar) for c in calendars):
        raise InvalidModel("explicit nonempty calendar tuple required")
    if len({c.parameter for c in calendars}) != len(calendars):
        raise InvalidModel("duplicate service calendars would double-mask one parameter")
    if len({(c.clock_id,c.clock_origin) for c in calendars}) != 1:
        raise InvalidModel("service calendars must use one explicit clock and origin")
    if type(allow_observed_diagnostic) is not bool:
        raise InvalidModel("diagnostic replay permission must be boolean")
    observed = any(c.source_kind == "observed_diagnostic" for c in calendars)
    if observed and not allow_observed_diagnostic:
        raise InvalidModel("observed target calendar needs explicit diagnostic replay permission")
    baseline = ResourceTimeline() if baseline_timeline is None else baseline_timeline
    if not isinstance(baseline, ResourceTimeline):
        raise InvalidModel("baseline calendar requires ResourceTimeline")
    horizon = min(c.valid_until_ns for c in calendars)
    current, by_name, points, times = {}, {}, {}, {0}
    for calendar in calendars:
        calendar.blackout_at(0)  # validates unknown period/width/phase without guessing.
        parameter = parameters.get(calendar.parameter)
        if parameter.unit not in {"thread_cpu_ns/s", "flop/s", "byte/s", "invocation/s"} or isinstance(parameter.value, InstantaneousRate):
            raise InvalidModel("calendar requires a finite explicit service rate")
        current[calendar.parameter] = parameters.require(calendar.parameter, parameter.unit)
        by_name[calendar.parameter] = calendar
        if 0 < calendar.blackout_ns < calendar.period_ns:
            for boundary in (calendar.phase_ns, calendar.phase_ns + calendar.blackout_ns):
                first = boundary % calendar.period_ns
                count = (horizon - first) // calendar.period_ns + 1 if first <= horizon else 0
                if count > 100_000 or len(times) + count > 200_000:
                    raise InvalidModel("service-calendar event limit exceeded")
                times.update(range(first, horizon + 1, calendar.period_ns))
    baseline_by_time = {}
    for point in baseline.points:
        if point.at_ns > horizon:
            raise InvalidModel("baseline resource boundary exceeds calendar horizon")
        for name, value in point.rates.items():
            parameter = parameters.get(name)
            if parameter.unit not in {"thread_cpu_ns/s", "flop/s", "byte/s", "invocation/s"} or isinstance(value, InstantaneousRate):
                raise InvalidModel("calendar baseline rate must be finite service")
            Parameter(name, value, parameter.unit, parameter.origin, parameter.evidence, parameter.scope)
        baseline_by_time[point.at_ns] = point
        times.add(point.at_ns)
    for timestamp in sorted(times):
        base = baseline_by_time.get(timestamp)
        updates = dict(base.rates) if base is not None else {}
        for name in by_name:
            if name in updates:
                current[name] = updates[name]
            updates[name] = 0 if by_name[name].blackout_at(timestamp) else current[name]
        point = ResourcePoint(timestamp, updates,
                              {} if base is None else base.capacities,
                              {} if base is None else base.external_memory)
        points[timestamp] = point
    return CompiledServiceCalendars(ResourceTimeline(tuple(points.values())), horizon, calendars, observed)
