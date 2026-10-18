"""Explicit serial host-completed action timing, without profiling statistics.

A completed action price includes everything within its measured boundary:
for example graph submission, device work and synchronization. It must not be
added again to prices for those internal phases. Gaps and the tail are separate
nonoverlapping intervals supplied by the caller, including explicit zero values.
This model neither infers a dispatch overhead nor subtracts instrumentation.

This is a bounded adapter for a serial native action protocol. It does not
change the tensor scheduler, establish hardware capacity, or claim a physical
GPU kernel timeline. Profiling binds the exact action inventory and observation
scope, estimates its parameters, and validates independent uninstrumented runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .evaluator import Evaluation, evaluate
from .graph import Graph, Task
from .parameters import (InvalidModel, ParameterBundle, encode_fraction,
                         exact_fields, integer, nonempty)

SCHEMA_VERSION = "serial-host-action-timing-1"
TIMING_SCOPE = "serial_host_completed_actions_and_explicit_nonoverlapping_gaps"
ACTION_KINDS = frozenset({"COMPUTE", "COPY", "ADVANCE", "EVICT"})


@dataclass(frozen=True)
class SerialActionTiming:
    """One original action and its explicitly priced preceding host interval.

    ``duration_parameter`` is an inclusive completed-action duration in ns, not
    an exclusive compute-service or transfer-bandwidth parameter. A zero-cost
    control action or gap still requires a named known parameter; missing or
    unknown observations cannot silently become zero.
    """

    action_index: int
    kind: str
    duration_parameter: str
    gap_before_parameter: str

    def __post_init__(self):
        integer(self.action_index, "action index")
        if not isinstance(self.kind, str) or self.kind not in ACTION_KINDS:
            raise InvalidModel("serial timing supports COMPUTE, COPY, ADVANCE and EVICT")
        nonempty(self.duration_parameter, "completed-action duration parameter")
        nonempty(self.gap_before_parameter, "gap-before parameter")
        if self.duration_parameter == self.gap_before_parameter:
            raise InvalidModel("a completed action and its preceding gap cannot charge the same parameter")

    def to_dict(self):
        return dict(action_index=self.action_index, kind=self.kind,
                    duration_parameter=self.duration_parameter,
                    gap_before_parameter=self.gap_before_parameter)


@dataclass(frozen=True)
class SerialTimingProgram:
    actions: tuple[SerialActionTiming, ...]
    tail_parameter: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise InvalidModel(f"requires {SCHEMA_VERSION}")
        if type(self.actions) is not tuple or not self.actions or any(
                not isinstance(a, SerialActionTiming) for a in self.actions):
            raise InvalidModel("serial actions must be a nonempty tuple of SerialActionTiming")
        if tuple(a.action_index for a in self.actions) != tuple(range(len(self.actions))):
            raise InvalidModel("original serial action indices must be unique, ordered and contiguous from zero")
        nonempty(self.tail_parameter, "tail parameter")
        durations = {a.duration_parameter for a in self.actions}
        gaps = {a.gap_before_parameter for a in self.actions} | {self.tail_parameter}
        if durations.intersection(gaps):
            raise InvalidModel("inclusive completed-action prices cannot also price gaps or the tail")

    def validate_action_inventory(self, kinds: tuple[str, ...]):
        """Check all original actions, including nominally free controls.

        The caller must additionally bind tensor identities, geometry, locations,
        execution flags and parameter provenance. Matching kinds alone does not
        establish that two native programs have the same physical contract.
        """
        if type(kinds) is not tuple or kinds != tuple(a.kind for a in self.actions):
            raise InvalidModel("serial timing inventory differs from the original action sequence")

    def to_dict(self):
        return dict(schema_version=self.schema_version,
                    actions=[a.to_dict() for a in self.actions],
                    tail_parameter=self.tail_parameter)

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"schema_version", "actions", "tail_parameter"},
                           "serial timing program")
        if not isinstance(obj["actions"], list):
            raise InvalidModel("serial actions must be a JSON list")
        actions = tuple(SerialActionTiming(**exact_fields(a, {"action_index", "kind",
            "duration_parameter", "gap_before_parameter"}, "serial action timing"))
                        for a in obj["actions"])
        return cls(actions, obj["tail_parameter"], obj["schema_version"])


def build_serial_action_graph(program: SerialTimingProgram) -> Graph:
    """Compile nonoverlapping host intervals to the existing modeling engine."""
    if not isinstance(program, SerialTimingProgram):
        raise InvalidModel("serial timing requires SerialTimingProgram")
    tasks = []
    previous = ()
    for action in program.actions:
        for suffix, parameter in (("gap_before", action.gap_before_parameter),
                                  ("completed", action.duration_parameter)):
            task_id = f"action:{action.action_index}:{suffix}"
            tasks.append(Task(task_id, "serial_host_completed", depends_on=previous,
                              fixed_parameter=parameter))
            previous = (task_id,)
    tasks.append(Task("program:tail", "serial_host_completed", depends_on=previous,
                      fixed_parameter=program.tail_parameter))
    return Graph(tuple(tasks))


@dataclass(frozen=True)
class SerialActionExecution:
    action_index: int
    kind: str
    gap_before_ns: int
    begin_ns: int
    end_ns: int

    @property
    def duration_ns(self):
        return self.end_ns - self.begin_ns

    def to_dict(self):
        return dict(action_index=self.action_index, kind=self.kind,
                    gap_before_ns=self.gap_before_ns, begin_ns=self.begin_ns,
                    end_ns=self.end_ns, duration_ns=self.duration_ns)


@dataclass(frozen=True)
class SerialTimingEvaluation:
    execution: Evaluation
    actions: tuple[SerialActionExecution, ...]
    tail_ns: int
    unrounded_interval_sum_ns: Fraction

    @property
    def duration_ns(self):
        return self.execution.duration_ns

    def to_dict(self):
        contributions = {kind: 0 for kind in sorted(ACTION_KINDS)}
        contributions.update(gap_before=0, tail=self.tail_ns)
        for action in self.actions:
            contributions[action.kind] += action.duration_ns
            contributions["gap_before"] += action.gap_before_ns
        return dict(schema_version="serial-host-action-prediction-1",
            duration_ns=self.duration_ns, actions=[a.to_dict() for a in self.actions],
            tail_ns=self.tail_ns, interval_contributions_ns=contributions,
            unrounded_interval_sum_ns=encode_fraction(self.unrounded_interval_sum_ns),
            integer_stage_rounding_ns=encode_fraction(
                Fraction(self.duration_ns) - self.unrounded_interval_sum_ns),
            timing_scope=TIMING_SCOPE, prediction_provider=
                "llm_infer_model.performance.serial_actions.evaluate_serial_actions",
            execution=self.execution.to_dict(),
            physical_kernel_timeline_claimed=False, physical_capacity_claimed=False,
            instrumentation_overhead_subtracted=False,
            independently_validated_accuracy_claimed=False,
            profile_statistics_calculated_here=False)


def evaluate_serial_actions(program: SerialTimingProgram,
                            bundle: ParameterBundle) -> SerialTimingEvaluation:
    """Calculate serial timing using explicit known ns prices, with no fitting.

    Each stage is rounded by the common engine to integer nanoseconds. The
    result retains the exact fractional input sum and resulting rounding delta.
    Reusing a parameter for repeated equivalent actions is explicit and allowed;
    an inclusive action parameter may never also be charged as a host gap.
    """
    graph = build_serial_action_graph(program)
    if not isinstance(bundle, ParameterBundle):
        raise InvalidModel("serial timing requires ParameterBundle")
    exact = sum((bundle.require(t.fixed_parameter, "ns") for t in graph.tasks), Fraction())
    result = evaluate(graph, bundle)
    spans = {t.task_id: t for t in result.tasks}
    actions = []
    for action in program.actions:
        completed = spans[f"action:{action.action_index}:completed"]
        gap = spans[f"action:{action.action_index}:gap_before"]
        actions.append(SerialActionExecution(action.action_index, action.kind,
            gap.duration_ns, completed.start_ns, completed.end_ns))
    return SerialTimingEvaluation(result, tuple(actions),
                                  spans["program:tail"].duration_ns, exact)
