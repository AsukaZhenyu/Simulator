"""Coarse critical-response hypotheses with explicit host CPU-cycle work.

This is not a reconstructed CUDA/API submission DAG. Profiling supplies scoped
cycle work and nonnegative completion-response latencies; this module performs
no residual fitting. A serial-after-host or overlapping completion assumption
is explicitly declared. The common Graph evaluator alone calculates duration.
Completion latency is not GPU FLOPs, kernel-active service or an OS schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .evaluator import evaluate
from .graph import Graph, ResourcePoint, ResourceTimeline, Task, WorkDemand
from .parameters import InvalidModel, Parameter, ParameterBundle, UnknownParameter, integer, nonempty, rational
from .service_calendar import PeriodicServiceCalendar


HOST_CYCLE_SERVICE = "host_cycle_service"
ZERO_EVENT_PARAMETER = "__cycle_response_structural_zero_ns"
COMPLETION_PHASES = frozenset({"after_host", "overlap_host"})


def _evidence(value):
    if type(value) is not tuple or not value:
        raise InvalidModel("explicit response-model evidence required")
    for item in value:
        nonempty(item,"response evidence")


def _latency(value,label):
    return None if value is None else rational(value,label)


@dataclass(frozen=True)
class CycleResponseStage:
    action_index: int
    kind: str
    cpu_cycles: int | None
    completion_latency_ns: int | Fraction | None
    gap_cycles: int | None

    def __post_init__(self):
        integer(self.action_index,"response action index")
        nonempty(self.kind,"response action kind")
        for key in ("cpu_cycles","gap_cycles"):
            if getattr(self,key) is not None:
                integer(getattr(self,key),key)
        object.__setattr__(self,"completion_latency_ns",_latency(self.completion_latency_ns,"scoped completion latency"))


@dataclass(frozen=True)
class CycleResponseProgram:
    stages: tuple[CycleResponseStage, ...]
    head_ns: int | Fraction | None
    tail_ns: int | Fraction | None
    evidence: tuple[str, ...]
    scope: str
    origin: str = "declared"

    def __post_init__(self):
        if type(self.stages) is not tuple or not self.stages or any(not isinstance(s,CycleResponseStage) for s in self.stages):
            raise InvalidModel("explicit nonempty CycleResponseStage tuple required")
        if tuple(s.action_index for s in self.stages)!=tuple(range(len(self.stages))):
            raise InvalidModel("response stages must retain contiguous original action indices")
        object.__setattr__(self,"head_ns",_latency(self.head_ns,"response head"))
        object.__setattr__(self,"tail_ns",_latency(self.tail_ns,"response tail"))
        _evidence(self.evidence)
        nonempty(self.scope,"response-model scope")
        if self.origin not in {"declared","profiled"}:
            raise InvalidModel("response parameter origin must be explicit")


def _aligned_timeline(timeline,start_ns):
    """Select caller-supplied absolute history and rebase, without inference."""
    if not isinstance(timeline,ResourceTimeline):
        raise InvalidModel("response evaluation requires ResourceTimeline")
    integer(start_ns,"response start alignment")
    if start_ns==0:
        return timeline
    rates,capacities,external={},{},{}
    future=[]
    for point in timeline.points:
        if point.at_ns<=start_ns:
            rates.update(point.rates);capacities.update(point.capacities);external.update(point.external_memory)
        else:
            future.append(ResourcePoint(point.at_ns-start_ns,point.rates,point.capacities,point.external_memory))
    entry=[] if not(rates or capacities or external) else [ResourcePoint(0,rates,capacities,external)]
    return ResourceTimeline(tuple(entry+future))


@dataclass(frozen=True)
class CompiledCycleResponse:
    graph: Graph
    internal_parameters: ParameterBundle
    program: CycleResponseProgram
    completion_phase: str

    @property
    def hardware_accuracy_qualified(self):
        return False

    def evaluate(self, parameters: ParameterBundle, timeline: ResourceTimeline | None = None, *, start_ns=0):
        if not isinstance(parameters,ParameterBundle):
            raise InvalidModel("explicit CPU-cycle service ParameterBundle required")
        # Even an all-zero sampled cycle count cannot silently establish an
        # unknown service parameter or accept active-time units instead.
        parameters.require(HOST_CYCLE_SERVICE,"cpu_cycle/s")
        reserved={p.name for p in self.internal_parameters.parameters}
        if reserved.intersection(p.name for p in parameters.parameters):
            raise InvalidModel("caller collides with response internal parameters")
        merged=ParameterBundle(parameters.parameters+self.internal_parameters.parameters)
        aligned=_aligned_timeline(ResourceTimeline() if timeline is None else timeline,start_ns)
        return evaluate(self.graph,merged,aligned)

    def to_dict(self):
        from .parameters import encode_fraction
        return dict(schema_version="host-cycle-critical-response-1",graph=self.graph.to_dict(),
            internal_parameters=self.internal_parameters.to_dict(),completion_phase=self.completion_phase,
            scope=self.program.scope,evidence=list(self.program.evidence),
            head_ns=encode_fraction(self.program.head_ns),tail_ns=encode_fraction(self.program.tail_ns),
            stages=[dict(action_index=s.action_index,kind=s.kind,cpu_cycles=s.cpu_cycles,
                completion_latency_ns=encode_fraction(s.completion_latency_ns),gap_cycles=s.gap_cycles) for s in self.program.stages],
            submission_acceptance_reconstructed=False,completion_is_GPU_active_service=False,
            completion_latency_is_scoped_response_hypothesis=True,layer_offset_fitted_here=False,
            cpu_cycles_converted_to_active_time=False,hardware_accuracy_qualified=False)


def compile_cycle_response(program: CycleResponseProgram, *, completion_phase="after_host"):
    """Lower one predeclared coarse timing assumption; missing costs stay unknown.

    after_host: gap cycle work -> host cycle work -> completion response.
    overlap_host: gap cycle work -> host work and completion start together;
    the next action waits for BOTH. A CPU-service blackout affects only cycle
    work; an already released completion response continues in wall time.
    """
    if not isinstance(program,CycleResponseProgram):
        raise InvalidModel("CycleResponseProgram required")
    if completion_phase not in COMPLETION_PHASES:
        raise InvalidModel("explicit after_host or overlap_host response assumption required")
    for key in ("head_ns","tail_ns"):
        if getattr(program,key) is None:
            raise UnknownParameter("unknown response "+key)
    for stage in program.stages:
        for key in ("cpu_cycles","completion_latency_ns","gap_cycles"):
            if getattr(stage,key) is None:
                raise UnknownParameter(f"unknown action {stage.action_index} {key}; not replaced by zero")
    parameters=[]

    def latency(name,value):
        parameters.append(Parameter(name,value,"ns",program.origin,program.evidence,program.scope))
        return name

    zero=Parameter(ZERO_EVENT_PARAMETER,0,"ns","declared",("explicit structural boundary, not a measured unknown",),"response graph bookkeeping")
    parameters.append(zero)
    head="cycle_response:head"
    tasks=[Task(head,"response_head",fixed_parameter=latency("cycle_response:head_ns",program.head_ns))]
    previous=(head,)
    for stage in program.stages:
        stem=f"action:{stage.action_index}"
        gap,release,host,response,joined=(stem+":"+s for s in ("gap_cycles","release","host_cycles","completion_response","join"))
        tasks.append(Task(gap,"caller_host_cycles",(WorkDemand(HOST_CYCLE_SERVICE,stage.gap_cycles,"cpu_cycle"),),previous))
        tasks.append(Task(release,"boundary:"+stem,depends_on=(gap,),fixed_parameter=ZERO_EVENT_PARAMETER))
        tasks.append(Task(host,"caller_host_cycles",(WorkDemand(HOST_CYCLE_SERVICE,stage.cpu_cycles,"cpu_cycle"),),(release,)))
        tasks.append(Task(response,"completion_response:"+stem,
            depends_on=(host,) if completion_phase=="after_host" else (release,),
            fixed_parameter=latency(stem+":completion_response_ns",stage.completion_latency_ns)))
        tasks.append(Task(joined,"completion_gate:"+stem,depends_on=(release,),waits_for=(host,response)))
        previous=(joined,)
    tasks.append(Task("cycle_response:tail","response_tail",depends_on=previous,
        fixed_parameter=latency("cycle_response:tail_ns",program.tail_ns)))
    return CompiledCycleResponse(Graph(tuple(tasks)),ParameterBundle(tuple(parameters)),program,completion_phase)


build_cycle_response_graph=compile_cycle_response


@dataclass(frozen=True)
class CompiledCycleCalendar:
    timeline: ResourceTimeline
    valid_until_ns: int
    calendar: PeriodicServiceCalendar
    diagnostic_replay: bool

    def evaluate(self, compiled: CompiledCycleResponse, parameters: ParameterBundle, *, start_ns=0):
        integer(start_ns,"calendar evaluation start")
        if start_ns>self.valid_until_ns:
            raise UnknownParameter("cycle response begins beyond external calendar horizon")
        result=compiled.evaluate(parameters,self.timeline,start_ns=start_ns)
        if start_ns+result.duration_ns>self.valid_until_ns:
            raise UnknownParameter("cycle response exceeds external calendar horizon")
        return result

    def to_dict(self):
        from dataclasses import asdict
        return dict(schema_version="external-cycle-service-calendar-1",timeline=self.timeline.to_dict(),
            valid_until_ns=self.valid_until_ns,calendar=asdict(self.calendar),diagnostic_replay=self.diagnostic_replay,
            cycles_converted_to_active_ns=False,phase_inferred_here=False,
            independent_instantaneous_service_claimed=False,hardware_accuracy_qualified=False)


def compile_cycle_calendar(parameters: ParameterBundle, calendar: PeriodicServiceCalendar, *, allow_observed_diagnostic=False):
    """Apply the SAME explicit periodic-calendar rule to raw cycle/s service.

    This adapter adds no unit conversion and leaves the earlier source-bound
    general calendar implementation untouched. Phase/clock/origin are caller
    aligned. Unknown phase/width/period/rate never becomes a fabricated default.
    """
    if not isinstance(parameters,ParameterBundle) or not isinstance(calendar,PeriodicServiceCalendar):
        raise InvalidModel("ParameterBundle and PeriodicServiceCalendar required")
    if type(allow_observed_diagnostic) is not bool:
        raise InvalidModel("diagnostic replay permission must be boolean")
    diagnostic=calendar.source_kind=="observed_diagnostic"
    if diagnostic and not allow_observed_diagnostic:
        raise InvalidModel("observed response calendar requires explicit diagnostic permission")
    calendar.blackout_at(0)
    rate=parameters.require(calendar.parameter,"cpu_cycle/s")
    times={0}
    if 0<calendar.blackout_ns<calendar.period_ns:
        for endpoint in (calendar.phase_ns,calendar.phase_ns+calendar.blackout_ns):
            first=endpoint%calendar.period_ns
            count=(calendar.valid_until_ns-first)//calendar.period_ns+1 if first<=calendar.valid_until_ns else 0
            if count>100_000 or len(times)+count>200_000:
                raise InvalidModel("cycle-calendar event limit exceeded")
            times.update(range(first,calendar.valid_until_ns+1,calendar.period_ns))
    points=tuple(ResourcePoint(t,{calendar.parameter:0 if calendar.blackout_at(t) else rate}) for t in sorted(times))
    return CompiledCycleCalendar(ResourceTimeline(points),calendar.valid_until_ns,calendar,diagnostic)
