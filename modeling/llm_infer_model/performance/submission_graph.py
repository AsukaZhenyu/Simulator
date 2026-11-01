"""Explicit host submission, device-stream and completion dependencies.

This lowering replaces an inclusive host-action price; it never adds another
launch fee to that price. Enqueue acceptance is a named boundary distinct from
API return. Host post-acceptance work can overlap device execution. No CPU work,
enqueue position, stream dependency, wait mode or parameter is inferred here.
The existing evaluator alone integrates work against external service rates.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evaluator import evaluate
from .graph import Graph, ResourceTimeline, Task, WorkDemand
from .parameters import InvalidModel, Parameter, ParameterBundle, nonempty


ACCEPTANCE_BOUNDARIES = frozenset({"after_prefix", "api_entry", "api_return"})
ZERO_EVENT_PARAMETER = "__submission_graph_structural_zero_ns"


def _texts(value, label, *, empty=False):
    if type(value) is not tuple or (not empty and not value):
        raise InvalidModel(label + " requires a tuple" + ("" if empty else " with evidence"))
    for item in value:
        nonempty(item, label)
    if len(set(value)) != len(value):
        raise InvalidModel("duplicate " + label)


def _work(value, label, *, host=False):
    if type(value) is not tuple or not value or any(not isinstance(w, WorkDemand) for w in value):
        raise InvalidModel(label + " requires explicit nonempty WorkDemand tuple; use explicit zero work if known")
    if len({w.parameter for w in value}) != len(value):
        raise InvalidModel("duplicate work parameter in " + label)
    if host and any(w.unit == "invocation" for w in value):
        raise InvalidModel("inclusive host invocation response cannot also charge device work/wait")


@dataclass(frozen=True)
class DeviceCommand:
    id: str
    stream: str
    resource: str
    demands: tuple[WorkDemand, ...]
    depends_on_commands: tuple[str, ...] = ()
    extra_resources: tuple[str, ...] = ()

    def __post_init__(self):
        for label in ("id", "stream", "resource"):
            nonempty(getattr(self, label), "device " + label)
        _work(self.demands, "device command")
        _texts(self.depends_on_commands, "device command dependencies", empty=True)
        _texts(self.extra_resources, "extra device resource", empty=True)
        if self.resource in self.extra_resources:
            raise InvalidModel("device primary resource repeated as extra")


@dataclass(frozen=True)
class HostSubmission:
    id: str
    command: DeviceCommand
    prefix_demands: tuple[WorkDemand, ...]
    suffix_demands: tuple[WorkDemand, ...]
    acceptance_boundary: str

    def __post_init__(self):
        nonempty(self.id, "host submission id")
        if not isinstance(self.command, DeviceCommand):
            raise InvalidModel("host submission requires one DeviceCommand")
        _work(self.prefix_demands, "host prefix", host=True)
        _work(self.suffix_demands, "host suffix", host=True)
        if self.acceptance_boundary not in ACCEPTANCE_BOUNDARIES:
            raise InvalidModel("explicit after_prefix, api_entry or api_return acceptance required")


@dataclass(frozen=True)
class CompletionWait:
    id: str
    awaited_commands: tuple[str, ...]
    enter_demands: tuple[WorkDemand, ...]
    continuation_demands: tuple[WorkDemand, ...]
    mode: str = "blocked"

    def __post_init__(self):
        nonempty(self.id, "completion wait id")
        _texts(self.awaited_commands, "awaited command")
        _work(self.enter_demands, "wait entry", host=True)
        _work(self.continuation_demands, "wait continuation", host=True)
        if self.mode not in {"blocked", "spin_occupancy"}:
            raise InvalidModel("wait mode must be blocked or explicit spin_occupancy")


@dataclass(frozen=True)
class SubmissionProgram:
    host_resource: str
    steps: tuple[HostSubmission | CompletionWait, ...]
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        nonempty(self.host_resource, "host resource")
        nonempty(self.scope, "submission program scope")
        _texts(self.evidence, "submission evidence")
        if type(self.steps) is not tuple or not self.steps or any(
                not isinstance(step, (HostSubmission, CompletionWait)) for step in self.steps):
            raise InvalidModel("nonempty tuple of host submissions/completion waits required")
        if len({step.id for step in self.steps}) != len(self.steps):
            raise InvalidModel("duplicate host step")
        commands = {}
        for step in self.steps:
            if isinstance(step, HostSubmission):
                command = step.command
                if command.id in commands:
                    raise InvalidModel("device command submitted more than once")
                if command.resource == self.host_resource or self.host_resource in command.extra_resources:
                    raise InvalidModel("device command cannot lock its submitting host resource")
                if not set(command.depends_on_commands) <= commands.keys():
                    raise InvalidModel("device dependencies require earlier submitted commands")
                commands[command.id] = command
            elif not set(step.awaited_commands) <= commands.keys():
                raise InvalidModel("completion wait references an unsubmitted command")


@dataclass(frozen=True)
class SubmissionBoundary:
    submission_id: str
    command_id: str
    entry_task: str
    acceptance_task: str
    return_task: str
    device_task: str
    acceptance_assumption: str


@dataclass(frozen=True)
class CompiledSubmissionGraph:
    graph: Graph
    structural_parameters: ParameterBundle
    boundaries: tuple[SubmissionBoundary, ...]
    scope: str
    evidence: tuple[str, ...]

    def to_dict(self):
        from dataclasses import asdict
        return dict(schema_version="host-device-submission-graph-1", graph=self.graph.to_dict(),
                    structural_parameters=self.structural_parameters.to_dict(),
                    boundaries=[asdict(b) for b in self.boundaries], scope=self.scope,
                    evidence=list(self.evidence), hardware_accuracy_qualified=False,
                    inclusive_action_price_added=False, enqueue_boundary_inferred=False,
                    device_stream_dependencies_explicit=True)

    def evaluate(self, parameters: ParameterBundle, timeline: ResourceTimeline | None = None):
        if not isinstance(parameters, ParameterBundle):
            raise InvalidModel("explicit measured/declared parameters required")
        reserved = {p.name for p in self.structural_parameters.parameters}
        if reserved.intersection(p.name for p in parameters.parameters):
            raise InvalidModel("caller parameter collides with structural event parameter")
        merged = ParameterBundle(parameters.parameters + self.structural_parameters.parameters)
        return evaluate(self.graph, merged, timeline)

    @property
    def hardware_accuracy_qualified(self):
        return False


def build_submission_graph(program: SubmissionProgram, *, acceptance_override=None) -> CompiledSubmissionGraph:
    """Lower explicit microprogram dependencies into the common Graph.

    An override of api_entry/api_return is a declared enqueue-boundary scenario,
    not a proven total-latency lower/upper bound for arbitrary sharing graphs.
    Entry is the model's call-entry boundary; dispatch work needed to reach it
    must already be represented by the caller. With after_prefix, prefix work
    finishes before acceptance and suffix work can overlap device execution.
    """
    if not isinstance(program, SubmissionProgram):
        raise InvalidModel("SubmissionProgram required")
    if acceptance_override is not None and acceptance_override not in {"api_entry", "api_return"}:
        raise InvalidModel("acceptance override must explicitly select api_entry or api_return")
    tasks, boundaries, previous, streams, commands = [], [], (), {}, {}

    def event(name, depends):
        tasks.append(Task(name, "boundary:" + name, depends_on=depends, fixed_parameter=ZERO_EVENT_PARAMETER))
        return name

    for step in program.steps:
        stem = "host:" + step.id
        if isinstance(step, HostSubmission):
            command = step.command
            entry = event(stem + ":entry", previous)
            prefix = stem + ":prefix"
            suffix = stem + ":suffix"
            returned = stem + ":return"
            tasks.append(Task(prefix, program.host_resource, step.prefix_demands, (entry,)))
            tasks.append(Task(suffix, program.host_resource, step.suffix_demands, (prefix,)))
            event(returned, (suffix,))
            assumption = acceptance_override or step.acceptance_boundary
            source = {"api_entry": entry, "after_prefix": prefix, "api_return": returned}[assumption]
            accepted = event(stem + ":accepted", (source,))
            device_id = "device:" + command.id
            dependencies = [accepted]
            dependencies.extend(commands[c] for c in command.depends_on_commands)
            if command.stream in streams:
                dependencies.append(streams[command.stream])
            dependencies = tuple(dict.fromkeys(dependencies))
            tasks.append(Task(device_id, command.resource, command.demands, dependencies,
                              extra_resources=command.extra_resources))
            streams[command.stream] = device_id
            commands[command.id] = device_id
            boundaries.append(SubmissionBoundary(step.id, command.id, entry, accepted,
                                                 returned, device_id, assumption))
            previous = (returned,)
        else:
            entered = stem + ":wait_entry"
            gate = stem + ":completion_gate"
            continued = stem + ":continuation"
            tasks.append(Task(entered, program.host_resource, step.enter_demands, previous))
            wait_resource = (program.host_resource if step.mode == "spin_occupancy"
                             else "blocked_wait:" + step.id)
            tasks.append(Task(gate, wait_resource, depends_on=(entered,),
                              waits_for=tuple(commands[c] for c in step.awaited_commands)))
            tasks.append(Task(continued, program.host_resource, step.continuation_demands, (gate,)))
            previous = (continued,)
    zero = Parameter(ZERO_EVENT_PARAMETER, 0, "ns", "declared",
                     ("structural logical boundary; explicitly zero work, not a measured unknown",),
                     "submission graph bookkeeping only")
    return CompiledSubmissionGraph(Graph(tuple(tasks)), ParameterBundle((zero,)), tuple(boundaries),
                                   program.scope, program.evidence)


def evaluate_submission_graph(program: SubmissionProgram, parameters: ParameterBundle,
                              timeline: ResourceTimeline | None = None, *, acceptance_override=None):
    return build_submission_graph(program, acceptance_override=acceptance_override).evaluate(parameters, timeline)
