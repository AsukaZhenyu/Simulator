"""Explicit same-core ISR/DPC service interruptions, with no OS inference.

An OS context-switch interval can say that a thread owns a logical CPU while
interrupt or deferred-procedure work actually uses that CPU. This module masks
an explicitly bound thread's CPU service during the *union* of such intervals.
The common evaluator still calculates work progress and completion times.

This is a scenario/replay mechanism, not an interrupt predictor, OS scheduler,
or calibrated latency model. Bindings are static and must hold for the entire
timeline. Base CPU service must exclude the interruptions supplied here;
otherwise the same loss would be counted twice. Rates describe one thread,
not an aggregate CPU budget. No service sharing or migration is inferred.

Observed diagnostic schedules, including heldout ETW schedules, are rejected
by default. Explicit diagnostic replay can help explain an observation, but
does not constitute a causal prediction or independent accuracy validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import ResourcePoint, ResourceTimeline
from .parameters import InvalidModel, Parameter, ParameterBundle, integer, nonempty


@dataclass(frozen=True)
class CpuInterruptionTimebase:
    """Explicit ns alignment and logical-CPU identity for a single scenario.

    All interval/timeline timestamps are integer nanoseconds relative to the
    same ``origin_id``. ``frequency_hz`` records the source clock's frequency;
    this module does not convert raw ticks or align clocks. The namespace and
    explicit inventory identify logical processors (including processor group
    conventions); a CPU number from another host/session is not interchangeable.
    """

    clock_id: str
    origin_id: str
    frequency_hz: int
    logical_cpu_namespace: str
    logical_cpu_ids: tuple[int, ...]

    def __post_init__(self):
        for name in ("clock_id", "origin_id", "logical_cpu_namespace"):
            nonempty(getattr(self, name), name)
        integer(self.frequency_hz, "source clock frequency", 1)
        if type(self.logical_cpu_ids) is not tuple or not self.logical_cpu_ids:
            raise InvalidModel("logical_cpu_ids requires a nonempty tuple of known CPU ids")
        for cpu_id in self.logical_cpu_ids:
            integer(cpu_id, "logical CPU id")
        if len(set(self.logical_cpu_ids)) != len(self.logical_cpu_ids):
            raise InvalidModel("duplicate logical CPU id")

    def to_dict(self):
        return dict(clock_id=self.clock_id, origin_id=self.origin_id,
                    frequency_hz=self.frequency_hz,
                    logical_cpu_namespace=self.logical_cpu_namespace,
                    logical_cpu_ids=list(self.logical_cpu_ids))


@dataclass(frozen=True)
class ThreadCpuBinding:
    """One named per-thread service parameter statically bound to one CPU.

    This declaration does not prove affinity, reserve a core, prevent migration,
    or arbitrate multiple threads. Those facts must be supplied independently.
    """

    parameter: str
    logical_cpu_id: int

    def __post_init__(self):
        nonempty(self.parameter, "bound CPU service parameter")
        integer(self.logical_cpu_id, "bound logical CPU id")

    def to_dict(self):
        return dict(parameter=self.parameter, logical_cpu_id=self.logical_cpu_id)


@dataclass(frozen=True)
class CpuInterruption:
    """Half-open [start_ns,end_ns) ISR/DPC interval and source provenance.

    ``external_scenario`` and ``predeclared_scenario`` designate independently
    supplied scenario inputs, not an inferred future interrupt schedule.
    ``observed_diagnostic`` always requires explicit replay permission.
    Evidence identifies the source, including scenario declarations.
    """

    logical_cpu_id: int
    start_ns: int
    end_ns: int
    kind: str
    origin: str
    evidence: tuple[str, ...]

    def __post_init__(self):
        integer(self.logical_cpu_id, "interrupted logical CPU id")
        integer(self.start_ns, "interruption start time")
        integer(self.end_ns, "interruption end time")
        if self.end_ns <= self.start_ns:
            raise InvalidModel("interruption end must be after start")
        if not isinstance(self.kind, str) or self.kind not in {"dpc", "isr"}:
            raise InvalidModel("interruption kind must be dpc or isr")
        if not isinstance(self.origin, str) or self.origin not in {
                "predeclared_scenario", "external_scenario", "observed_diagnostic"}:
            raise InvalidModel("interruption origin must be explicit scenario or observed_diagnostic")
        if type(self.evidence) is not tuple or not self.evidence:
            raise InvalidModel("interruption evidence requires a nonempty tuple")
        for reference in self.evidence:
            nonempty(reference, "interruption evidence reference")

    def to_dict(self):
        return dict(logical_cpu_id=self.logical_cpu_id, start_ns=self.start_ns,
                    end_ns=self.end_ns, kind=self.kind, origin=self.origin,
                    evidence=list(self.evidence))


@dataclass(frozen=True)
class CpuInterruptionTrace:
    timebase: CpuInterruptionTimebase
    intervals: tuple[CpuInterruption, ...]

    def __post_init__(self):
        if not isinstance(self.timebase, CpuInterruptionTimebase):
            raise InvalidModel("interruption trace requires CpuInterruptionTimebase")
        if type(self.intervals) is not tuple or any(
                not isinstance(item, CpuInterruption) for item in self.intervals):
            raise InvalidModel("interruptions must be a tuple of CpuInterruption")
        if any(item.logical_cpu_id not in self.timebase.logical_cpu_ids for item in self.intervals):
            raise InvalidModel("interruption references an unknown logical CPU id")

    def to_dict(self):
        return dict(timebase=self.timebase.to_dict(),
                    intervals=[item.to_dict() for item in self.intervals])


@dataclass(frozen=True)
class CpuInterruptionUnion:
    """Merged interval with indices into the retained, unmodified source trace."""

    logical_cpu_id: int
    start_ns: int
    end_ns: int
    input_indices: tuple[int, ...]

    def to_dict(self):
        return dict(logical_cpu_id=self.logical_cpu_id, start_ns=self.start_ns,
                    end_ns=self.end_ns, input_indices=list(self.input_indices))


@dataclass(frozen=True)
class CpuInterruptionComposition:
    timeline: ResourceTimeline
    bindings: tuple[ThreadCpuBinding, ...]
    trace: CpuInterruptionTrace
    unions: tuple[CpuInterruptionUnion, ...]
    diagnostic_replay: bool
    # This transformation establishes semantics, never empirical accuracy.
    accuracy_qualified: bool = field(default=False, init=False)

    def to_dict(self):
        return dict(timeline=self.timeline.to_dict(),
                    bindings=[item.to_dict() for item in self.bindings],
                    trace=self.trace.to_dict(),
                    unions=[item.to_dict() for item in self.unions],
                    diagnostic_replay=self.diagnostic_replay,
                    accuracy_qualified=self.accuracy_qualified,
                    scope="explicit same-core CPU-service interruption; no schedule prediction or accuracy qualification")


def _union_intervals(trace):
    result = []
    for cpu_id in sorted(trace.timebase.logical_cpu_ids):
        selected = sorted((item.start_ns, item.end_ns, index)
                          for index, item in enumerate(trace.intervals)
                          if item.logical_cpu_id == cpu_id)
        for start, end, index in selected:
            if result and result[-1].logical_cpu_id == cpu_id and start <= result[-1].end_ns:
                previous = result.pop()
                result.append(CpuInterruptionUnion(cpu_id, previous.start_ns,
                    max(previous.end_ns, end), previous.input_indices + (index,)))
            else:
                result.append(CpuInterruptionUnion(cpu_id, start, end, (index,)))
    return tuple(result)


def compose_cpu_interruptions(
        bundle: ParameterBundle,
        bindings: tuple[ThreadCpuBinding, ...],
        trace: CpuInterruptionTrace,
        *, baseline_timebase: CpuInterruptionTimebase,
        baseline_timeline: ResourceTimeline | None = None,
        allow_observed_diagnostic: bool = False) -> CpuInterruptionComposition:
    """Mask same-core thread CPU supply, then restore its latest base rate.

    Baseline rates must exclude these interruptions. Baseline changes occurring
    inside an interruption update the stored base rate, while effective service
    stays zero; the most recent base rate is restored on union exit. Unbound
    rates, capacities and external-memory changes are preserved. Overlapping,
    nested or adjacent ISR/DPC intervals on one CPU are charged only once.

    Diagnostic replay is explicit and remains accuracy-unqualified. In
    particular, a heldout observation's interrupt schedule must not become a
    prediction input. This function neither estimates unknown parameters nor
    schedules competing threads or predicts interrupt arrival/duration.
    """
    if not isinstance(bundle, ParameterBundle):
        raise InvalidModel("CPU interruption composition requires ParameterBundle")
    if not isinstance(trace, CpuInterruptionTrace):
        raise InvalidModel("CPU interruption composition requires CpuInterruptionTrace")
    if not isinstance(baseline_timebase, CpuInterruptionTimebase) or baseline_timebase != trace.timebase:
        raise InvalidModel("baseline and interruption clock, origin and CPU namespace must match exactly")
    if type(allow_observed_diagnostic) is not bool:
        raise InvalidModel("allow_observed_diagnostic must be a bool")
    observed = any(item.origin == "observed_diagnostic" for item in trace.intervals)
    if observed and not allow_observed_diagnostic:
        raise InvalidModel("observed diagnostic interruption schedule requires explicit diagnostic replay; not a prediction input")
    if type(bindings) is not tuple or not bindings or any(
            not isinstance(item, ThreadCpuBinding) for item in bindings):
        raise InvalidModel("bindings require a nonempty tuple of ThreadCpuBinding")
    if len({item.parameter for item in bindings}) != len(bindings):
        raise InvalidModel("duplicate bound CPU service parameter")
    if any(item.logical_cpu_id not in trace.timebase.logical_cpu_ids for item in bindings):
        raise InvalidModel("binding references an unknown logical CPU id")
    timeline = ResourceTimeline() if baseline_timeline is None else baseline_timeline
    if not isinstance(timeline, ResourceTimeline):
        raise InvalidModel("baseline_timeline must be ResourceTimeline")

    base_rates = {item.parameter: bundle.require(item.parameter, "thread_cpu_ns/s")
                  for item in bindings}
    # Enforce the existing physical rate contract even for updates that would
    # temporarily be hidden by a zero-service mask.
    for point in timeline.points:
        for name, value in point.rates.items():
            source = bundle.get(name)
            if source.unit not in {"flop/s", "byte/s", "invocation/s", "thread_cpu_ns/s"}:
                raise InvalidModel(f"timeline can only change service rates: {name}")
            Parameter(name, value, source.unit, source.origin, source.evidence, source.scope)

    unions = _union_intervals(trace)
    bound_cpus = {item.logical_cpu_id for item in bindings}
    endpoints = {}
    for interval in unions:
        if interval.logical_cpu_id in bound_cpus:
            endpoints.setdefault(interval.start_ns, []).append((interval.logical_cpu_id, 1))
            endpoints.setdefault(interval.end_ns, []).append((interval.logical_cpu_id, -1))
    baseline_points = {point.at_ns: point for point in timeline.points}
    active = {cpu_id: 0 for cpu_id in bound_cpus}
    effective = dict(base_rates)
    points = []
    for at_ns in sorted(set(baseline_points) | set(endpoints)):
        baseline = baseline_points.get(at_ns)
        rates = dict(baseline.rates) if baseline is not None else {}
        for name in base_rates.keys() & rates.keys():
            base_rates[name] = rates[name]
        # Half-open intervals and baseline changes at the same timestamp are
        # applied atomically, with no artificial recovery between adjacent work.
        for cpu_id, delta in endpoints.get(at_ns, ()):
            active[cpu_id] += delta
        for binding in bindings:
            name = binding.parameter
            value = 0 if active[binding.logical_cpu_id] else base_rates[name]
            if name in rates or value != effective[name]:
                rates[name] = value
            effective[name] = value
        if baseline is not None or rates:
            points.append(ResourcePoint(at_ns, rates,
                baseline.capacities if baseline is not None else {},
                baseline.external_memory if baseline is not None else {}))
    return CpuInterruptionComposition(ResourceTimeline(tuple(points)), bindings,
                                      trace, unions, observed)
