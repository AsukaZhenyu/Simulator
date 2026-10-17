"""Global episode uniform-cost search over model-neutral tensor schedules.

The tensor engine owns time, resources, dependencies and capacity. The schedule
helper owns the declared operation order, preparation window and copy pool.
An episode adapter owns only workload binding and concrete result decoding:
the search never dispatches on a model family or a task kind.

Elapsed time is a path cost, while engine State.key() records remaining work.
Dominance is therefore valid only for the static costs of each bound scenario
and an adapter identity covering all actual boundary state that can affect its
future. Calls are serial; work may overlap within each call.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Any, Hashable, Mapping, Protocol, Sequence

from llm_infer_model.tensor.engine import (
    ACTION_COMPUTE, STATUS_VALID, Action, EvaluationResult, MappingError,
    State, StateSnapshot, evaluate_mapping, initial_state, is_goal,
)
from llm_infer_model.tensor.spec import (
    InitialCapacityExceeded, InvalidInput, Scenario, Unsupported,
    scenario_fingerprint,
)

from .schedule import (
    ScheduleContext, ScheduleFacts, SchedulePlan, SchedulePolicy,
    legal_schedule_actions, make_schedule_context, plan_schedule,
    replay_schedule_plan, step_schedule,
)

SEARCH_SPACE = "global_episode_fixed_operation_start_order_windowed_prefetch_concurrent_resources"


@dataclass(frozen=True)
class ScheduleInvocationResult:
    """One replayed call and opaque adapter-owned persistent boundary records."""
    compiled: Any
    placements: tuple[tuple[str, str], ...]
    plan: SchedulePlan | None
    result: Any
    error: MappingError | InitialCapacityExceeded | None = None
    boundary_state_in: Any = None
    boundary_state_out: Any = None

    @property
    def status(self) -> str:
        if isinstance(self.error, InitialCapacityExceeded):
            return "infeasible"
        if self.error is not None:
            return "invalid_mapping"
        return self.result.status

    @property
    def reason(self) -> str:
        return str(self.error) if self.error is not None else self.result.evaluation.reason


@dataclass(frozen=True)
class ScheduleSequenceCandidate:
    index: int
    policy: SchedulePolicy
    invocations: tuple[ScheduleInvocationResult, ...]
    declared_invocation_count: int
    decisions: tuple[int, ...] = ()

    @property
    def id(self) -> str:
        return f"candidate-{self.index:04d}"

    @property
    def is_valid(self) -> bool:
        return len(self.invocations) == self.declared_invocation_count and all(
            item.status == STATUS_VALID for item in self.invocations)

    @property
    def status(self) -> str:
        return STATUS_VALID if self.is_valid else self.invocations[-1].status

    @property
    def reason(self) -> str:
        return "all_invocations_replayed" if self.is_valid else self.invocations[-1].reason

    @property
    def total_makespan_ns(self) -> int | None:
        return sum(item.result.makespan_ns for item in self.invocations) if self.is_valid else None

    @property
    def makespan_ns(self) -> int | None:
        return self.total_makespan_ns

    @property
    def cache_hits(self) -> int:
        return sum(item.plan.cache_hits for item in self.invocations if item.plan is not None)

    @property
    def cache_misses(self) -> int:
        return sum(item.plan.cache_misses for item in self.invocations if item.plan is not None)

    @property
    def eviction_count(self) -> int:
        return sum(item.plan.eviction_count for item in self.invocations if item.plan is not None)


class ScheduleEpisodeAdapter(Protocol):
    """Binding contract, independent of any graph/model family.

    ``initial`` preserves the declared template. ``bind`` must use the previous
    verified call's actual persistent state, never the next template's sample
    state. ``identity`` must include the scenario fingerprint and *all* boundary
    metadata that can change a later call. Adapters may refuse unsupported
    bindings; only actual initial-capacity failures prune individual branches.
    ``finish_call`` replays the exact actions and exposes its persistent output.
    ``validate_seed_call`` checks immutable workload, actual input boundary and
    the declared scenario including resource switches before any seed is used.
    """
    def validate_templates(self, templates: Sequence[Any]) -> tuple[Any, ...]: ...
    def initial(self, template: Any) -> Any: ...
    def bind(self, template: Any, previous: ScheduleInvocationResult) -> Any: ...
    def make_context(self, compiled: Any, policy: SchedulePolicy,
                     placements: Mapping[str, str] | None) -> ScheduleContext: ...
    def identity(self, compiled: Any) -> Hashable: ...
    def finish_call(self, context: ScheduleContext, actions: Sequence[Action], state: State,
                    *, require_complete: bool = True) -> ScheduleInvocationResult: ...
    def validate_seed_call(self, compiled: Any, seed: ScheduleInvocationResult) -> None: ...


@dataclass(frozen=True)
class ScheduleSearchResult:
    templates: tuple[Any, ...]
    policy: SchedulePolicy
    candidates: tuple[ScheduleSequenceCandidate, ...]
    best_index: int | None
    max_expanded_states: int
    max_generated_states: int
    expanded_states: int
    generated_states: int
    popped_states: int
    duplicate_states: int
    stale_pops: int
    pruned_states: int
    rejected_actions: int
    rejected_boundaries: int
    boundary_refusals: tuple[Mapping[str, Any], ...]
    search_complete: bool
    termination_reason: str
    fixed_placements: bool
    incumbent_seeded: bool
    incumbent_generated: bool
    goal_states_generated: int
    initial_refusal: Mapping[str, Any] | None = None

    @property
    def best(self) -> ScheduleSequenceCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]

    @property
    def ranking(self) -> tuple[ScheduleSequenceCandidate, ...]:
        return tuple(sorted((item for item in self.candidates if item.is_valid),
                            key=lambda item: (item.total_makespan_ns, item.index)))

    @property
    def complete(self) -> bool:
        return self.search_complete

    @property
    def enumeration_complete(self) -> bool:
        return self.termination_reason == "exhausted_declared_space"

    @property
    def optimality_proven(self) -> bool:
        return self.best is not None and self.termination_reason == "episode_goal_popped"

    @property
    def status(self) -> str:
        if self.optimality_proven:
            return "optimal_within_declared_space"
        if self.search_complete and self.best is None:
            return "infeasible_within_declared_space"
        return "feasible" if self.best is not None else "unknown"


@dataclass(frozen=True)
class _Trace:
    action: Action
    previous: _Trace | None


@dataclass(frozen=True)
class _Node:
    call_index: int
    compiled: Any
    context: ScheduleContext
    state: State
    start_ns: int
    past: tuple[ScheduleInvocationResult, ...]
    trace: _Trace | None
    identity: Hashable

    @property
    def cost(self) -> int:
        return self.start_ns + self.state.t

    @property
    def key(self) -> tuple[Any, ...]:
        return self.call_index, self.identity, self.state.key()


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput(f"{label} must be a positive integer")
    return value


def _templates(adapter: ScheduleEpisodeAdapter, templates: Sequence[Any]) -> tuple[Any, ...]:
    result = tuple(adapter.validate_templates(templates))
    if not result:
        raise InvalidInput("schedule episode requires a nonempty sequence of templates")
    return result


def _placements(templates: tuple[Any, ...], placements: Sequence[Mapping[str, str]] | None):
    if placements is None:
        return None
    try:
        result = tuple(dict(item) for item in placements)
    except (TypeError, ValueError) as error:
        raise InvalidInput("placements must supply one operation-to-device map per invocation") from error
    if len(result) != len(templates):
        raise InvalidInput("placements must supply exactly one map per invocation")
    return result


def _identity(adapter: ScheduleEpisodeAdapter, compiled: Any) -> Hashable:
    value = adapter.identity(compiled)
    try:
        hash(value)
    except TypeError as error:
        raise InvalidInput("schedule adapter identity must be hashable") from error
    return value


def _actions(trace: _Trace | None) -> tuple[Action, ...]:
    result = []
    while trace is not None:
        result.append(trace.action)
        trace = trace.previous
    return tuple(reversed(result))


def verify_replayed_state(evaluation: EvaluationResult, state: State) -> None:
    """A boundary must reproduce time and all three actual tensor ledgers."""
    if not evaluation.states:
        raise InvalidInput("schedule replay has no final state snapshot")
    snapshot = evaluation.states[-1]
    fields = ("t", "op_status", "copy_status", "dram_status", "ssd_status")
    if any(getattr(snapshot, field) != getattr(state, field) for field in fields):
        raise InvalidInput("schedule replay final state differs from the searched tensor state")
    if evaluation.status == STATUS_VALID and state.running:
        raise InvalidInput("completed schedule boundary cannot contain in-flight work")


def _call_result(adapter: ScheduleEpisodeAdapter, node: _Node) -> ScheduleInvocationResult:
    actions = _actions(node.trace)
    item = adapter.finish_call(node.context, actions, node.state, require_complete=True)
    if item.status != STATUS_VALID or item.result.makespan_ns != node.state.t:
        raise InvalidInput("generated schedule goal did not reproduce its tensor-engine replay")
    verify_replayed_state(item.result.evaluation, node.state)
    if item.result.evaluation.actions != actions:
        raise InvalidInput("schedule replay must retain the complete searched action sequence")
    expected = set(node.context.facts.operation_order)
    placements = tuple((action.target_id, action.device) for action in actions
                       if action.kind == ACTION_COMPUTE)
    if len(placements) != len(expected) or {operation for operation, _ in placements} != expected:
        raise InvalidInput("completed schedule must start every declared operation exactly once")
    if item.plan is None or item.plan.actions != actions or item.placements != placements:
        raise InvalidInput("schedule adapter replay must retain the exact searched actions and placements")
    return item


def _candidate(adapter, node, policy, count, index):
    return ScheduleSequenceCandidate(index, policy,
        node.past + (_call_result(adapter, node),), count)


def _validated_seed(adapter, templates, policy, placements, seed):
    """Rebuild every seed call and its actual boundary; never trust a price."""
    if not isinstance(seed, ScheduleSequenceCandidate) or not seed.is_valid or len(seed.invocations) != len(templates):
        raise InvalidInput("schedule incumbent must be a completed ScheduleSequenceCandidate for this episode")
    past = []
    start_ns = 0
    last_node = None
    for index, (template, item) in enumerate(zip(templates, seed.invocations)):
        if item.plan is None:
            raise InvalidInput("schedule incumbent lacks a concrete call plan")
        compiled = adapter.initial(template) if index == 0 else adapter.bind(template, past[-1])
        adapter.validate_seed_call(compiled, item)
        selected = None if placements is None else placements[index]
        context = adapter.make_context(compiled, policy, selected)
        state = initial_state(context.scenario)
        trace = None
        try:
            for action in item.plan.actions:
                state = step_schedule(context, state, action)
                trace = _Trace(action, trace)
        except MappingError as error:
            raise InvalidInput(f"schedule incumbent is outside the declared legal space: {error}") from error
        if not is_goal(context.scenario, state):
            raise InvalidInput("schedule incumbent call does not reach its required goal")
        last_node = _Node(index, compiled, context, state, start_ns, tuple(past), trace,
                          _identity(adapter, compiled))
        replayed = _call_result(adapter, last_node)
        if replayed.result.makespan_ns != item.result.makespan_ns:
            raise InvalidInput("schedule incumbent claimed call latency differs from verified replay")
        past.append(replayed)
        start_ns += replayed.result.makespan_ns
    assert last_node is not None
    candidate = _candidate(adapter, last_node, policy, len(templates), 0)
    if candidate.total_makespan_ns != seed.total_makespan_ns:
        raise InvalidInput("schedule incumbent claimed episode latency differs from verified replay")
    return last_node, candidate


def search_schedule_episode(
    templates: Sequence[Any], *, adapter: ScheduleEpisodeAdapter, policy: SchedulePolicy,
    placements: Sequence[Mapping[str, str]] | None = None,
    max_expanded_states: int, max_generated_states: int,
    incumbent: ScheduleSequenceCandidate | None = None,
) -> ScheduleSearchResult:
    """Search the entire serial-call episode under explicit state budgets.

    Generated counts accepted nondominated queue insertions, including initial
    and boundary states and a queued validated seed. Expanded counts non-stale
    pops including goals. A generated goal is only feasible; a final goal pop
    proves optimality within this declared static-cost space. A later hardware
    initial-capacity failure prunes that branch, never the whole episode.
    """
    templates = _templates(adapter, templates)
    expanded_limit = _positive(max_expanded_states, "max_expanded_states")
    generated_limit = _positive(max_generated_states, "max_generated_states")
    policy.validate()
    selected = _placements(templates, placements)
    if selected is not None:
        for template, placement in zip(templates, selected):
            try:
                adapter.make_context(adapter.initial(template), policy, placement)
            except InitialCapacityExceeded:
                # A template's sample boundary is not the actual future call.
                # Capacity is classified at the first real or bound boundary.
                continue
    candidates = []
    best_index = None
    expanded = generated = popped = duplicates = stale = pruned = rejected_actions = rejected_boundaries = 0
    refusals = []
    seeded = generated_goal = False
    goal_count = 0
    initial_refusal = None
    reason = "exhausted_declared_space"
    complete = True
    heap = []
    sequence = itertools.count()
    distances = {}

    def result():
        return ScheduleSearchResult(templates, policy, tuple(candidates), best_index,
            expanded_limit, generated_limit, expanded, generated, popped, duplicates,
            stale, pruned, rejected_actions, rejected_boundaries, tuple(refusals),
            complete, reason, selected is not None, seeded, generated_goal, goal_count,
            initial_refusal)

    try:
        first = adapter.initial(templates[0])
        context = adapter.make_context(first, policy, None if selected is None else selected[0])
        initial = initial_state(context.scenario)
    except InitialCapacityExceeded as error:
        initial_refusal = {"call_index": 0, "reason": str(error), "code": "initial_capacity_exceeded"}
        reason = "initial_capacity_exceeded"
        return result()
    root = _Node(0, first, context, initial, 0, (), None, _identity(adapter, first))
    distances[root.key] = root.cost
    heapq.heappush(heap, (root.cost, next(sequence), root))
    generated = 1
    if incumbent is not None:
        seeded_node, seeded_candidate = _validated_seed(adapter, templates, policy, selected, incumbent)
        candidates.append(seeded_candidate)
        best_index = 0
        seeded = True
        # A verified feasible path survives even when the queue budget is one.
        if generated < generated_limit:
            distances[seeded_node.key] = seeded_node.cost
            heapq.heappush(heap, (seeded_node.cost, next(sequence), seeded_node))
            generated += 1

    def enqueue(node):
        nonlocal generated, duplicates, pruned, best_index, generated_goal, goal_count, complete, reason
        best_cost = None if best_index is None else candidates[best_index].total_makespan_ns
        if best_cost is not None and node.cost > best_cost:
            pruned += 1
            return True
        previous = distances.get(node.key)
        if previous is not None and previous <= node.cost:
            duplicates += 1
            return True
        if generated >= generated_limit:
            complete = False
            reason = "max_generated_states"
            return False
        distances[node.key] = node.cost
        heapq.heappush(heap, (node.cost, next(sequence), node))
        generated += 1
        if node.call_index == len(templates) - 1 and is_goal(node.context.scenario, node.state):
            candidate = _candidate(adapter, node, policy, len(templates), len(candidates))
            candidates.append(candidate)
            goal_count += 1
            generated_goal = True
            if best_index is None or candidate.total_makespan_ns < candidates[best_index].total_makespan_ns:
                best_index = candidate.index
        return True

    halted = False
    while heap:
        if expanded >= expanded_limit:
            complete = False
            reason = "max_expanded_states"
            break
        _, _, node = heapq.heappop(heap)
        popped += 1
        if distances.get(node.key) != node.cost:
            stale += 1
            continue
        expanded += 1
        if is_goal(node.context.scenario, node.state):
            if node.call_index == len(templates) - 1:
                if best_index is None or node.cost < candidates[best_index].total_makespan_ns:
                    candidate = _candidate(adapter, node, policy, len(templates), len(candidates))
                    candidates.append(candidate)
                    best_index = candidate.index
                reason = "episode_goal_popped"
                complete = True
                break
            item = _call_result(adapter, node)
            next_index = node.call_index + 1
            try:
                following = adapter.bind(templates[next_index], item)
                next_context = adapter.make_context(following, policy,
                    None if selected is None else selected[next_index])
                state = initial_state(next_context.scenario)
            except InitialCapacityExceeded as error:
                rejected_boundaries += 1
                if len(refusals) < 32:
                    refusals.append({"call_index": next_index, "completed_prefix_time_ns": node.cost,
                                     "code": "initial_capacity_exceeded", "reason": str(error)})
                continue
            following_node = _Node(next_index, following, next_context, state, node.cost,
                node.past + (item,), None, _identity(adapter, following))
            if not enqueue(following_node):
                halted = True
                break
            continue
        for action in legal_schedule_actions(node.context, node.state):
            try:
                successor = step_schedule(node.context, node.state, action)
            except MappingError:
                rejected_actions += 1
                continue
            child = _Node(node.call_index, node.compiled, node.context, successor, node.start_ns,
                          node.past, _Trace(action, node.trace), node.identity)
            if not enqueue(child):
                halted = True
                break
        if halted:
            break
    return result()


def run_schedule_sequence(
    templates: Sequence[Any], *, adapter: ScheduleEpisodeAdapter, policy: SchedulePolicy,
    placements: Sequence[Mapping[str, str]] | None = None, max_actions: int, index: int = 0,
) -> ScheduleSequenceCandidate:
    """Replay the shared greedy policy, binding each actual completed boundary."""
    templates = _templates(adapter, templates)
    _positive(max_actions, "max_actions")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise InvalidInput("candidate index must be a nonnegative integer")
    policy.validate()
    selected = _placements(templates, placements)
    declared_contexts = []
    for i, template in enumerate(templates):
        try:
            declared_contexts.append(adapter.make_context(adapter.initial(template), policy,
                None if selected is None else selected[i]))
        except InitialCapacityExceeded:
            declared_contexts.append(None)
    invocations = []
    history = None
    for i, template in enumerate(templates):
        compiled = template
        try:
            compiled = adapter.initial(template) if i == 0 else adapter.bind(template, invocations[-1])
            context = adapter.make_context(compiled, policy, None if selected is None else selected[i])
            future = tuple((tensor, "dram" if device == "cpu" else "vram")
                for future_context in declared_contexts[i + 1:]
                if future_context is not None
                for operation in future_context.facts.operation_order
                for device in future_context.devices_for(operation)
                for tensor in future_context.scenario.workload.operation_by_id[operation].inputs
                if tensor in context.facts.managed_reusable_tensors)
            plan = plan_schedule(context, history=history, max_actions=max_actions, future_accesses=future)
            state = initial_state(context.scenario)
            for action in plan.actions:
                state = step_schedule(context, state, action)
            item = adapter.finish_call(context, plan.actions, state, require_complete=False)
            verify_replayed_state(item.result.evaluation, state)
            # Keep the actual carried history, not a fresh per-call observer.
            item = ScheduleInvocationResult(item.compiled, item.placements, plan, item.result,
                item.error, item.boundary_state_in, item.boundary_state_out)
        except (MappingError, InitialCapacityExceeded) as error:
            placement = () if selected is None else tuple(selected[i].items())
            item = ScheduleInvocationResult(getattr(compiled, "compiled", compiled), placement,
                                             None, None, error)
        invocations.append(item)
        if item.status != STATUS_VALID:
            break
        history = item.plan.history
    return ScheduleSequenceCandidate(index, policy, tuple(invocations), len(templates))


@dataclass(frozen=True)
class TensorInvocationResult:
    """Concrete tensor replay, without a model-specific state decoder."""
    scenario: Scenario
    evaluation: EvaluationResult
    state_out: StateSnapshot | None

    @property
    def status(self) -> str:
        return self.evaluation.status

    @property
    def makespan_ns(self) -> int | None:
        return self.evaluation.makespan_ns


@dataclass(frozen=True)
class TensorScenarioAdapter:
    """One arbitrary tensor graph; multi-call boundaries need an explicit binder."""
    facts: ScheduleFacts | None = None

    def validate_templates(self, templates):
        result = tuple(templates)
        if not result or any(not isinstance(item, Scenario) for item in result):
            raise InvalidInput("tensor schedule requires Scenario templates")
        if len(result) != 1:
            raise Unsupported("multiple tensor calls require an explicit persistent-state episode adapter")
        return result

    def initial(self, template):
        return template

    def bind(self, template, previous):
        raise Unsupported("tensor scenario adapter has no implicit cross-call state binding")

    def make_context(self, compiled, policy, placements):
        facts = self.facts
        if facts is None:
            facts = ScheduleFacts(tuple(op.id for op in compiled.workload.operations), frozenset(), ())
        return make_schedule_context(compiled, facts, policy, placements, payload=compiled)

    def identity(self, compiled):
        return scenario_fingerprint(compiled)

    def validate_seed_call(self, compiled, seed):
        if not isinstance(seed.compiled, Scenario) or scenario_fingerprint(seed.compiled) != scenario_fingerprint(compiled):
            raise InvalidInput("schedule incumbent uses a different tensor scenario")

    def finish_call(self, context, actions, state, *, require_complete=True):
        plan = replay_schedule_plan(context, actions, require_complete=require_complete)
        replay = evaluate_mapping(context.scenario, plan.actions)
        verify_replayed_state(replay, state)
        snapshot = replay.states[-1] if replay.status == STATUS_VALID else None
        result = TensorInvocationResult(context.scenario, replay, snapshot)
        placements = tuple((action.target_id, action.device) for action in plan.actions
                           if action.kind == ACTION_COMPUTE)
        return ScheduleInvocationResult(context.scenario, placements, plan, result,
            boundary_state_in=initial_state(context.scenario), boundary_state_out=snapshot)


def search_schedule(
    scenario: Scenario, *, policy: SchedulePolicy, facts: ScheduleFacts | None = None,
    placements: Mapping[str, str] | None = None, max_expanded_states: int,
    max_generated_states: int, incumbent: ScheduleSequenceCandidate | None = None,
) -> ScheduleSearchResult:
    """Convenience entry point using exactly the same episode search engine."""
    return search_schedule_episode((scenario,), adapter=TensorScenarioAdapter(facts), policy=policy,
        placements=None if placements is None else (placements,), max_expanded_states=max_expanded_states,
        max_generated_states=max_generated_states, incumbent=incumbent)


__all__ = ["SEARCH_SPACE", "ScheduleEpisodeAdapter", "ScheduleInvocationResult",
    "ScheduleSequenceCandidate", "ScheduleSearchResult", "TensorScenarioAdapter",
    "TensorInvocationResult", "search_schedule", "search_schedule_episode", "run_schedule_sequence",
    "verify_replayed_state"]
