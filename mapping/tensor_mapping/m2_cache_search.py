"""Budgeted offline sequence search over serial demand-driven MoE cache plans.

The declared space fixes task order and JIT transfer order, while preserving
the authoritative READY weight-source goals selected for each invocation.
It branches only when an expert-cache quota or the complete engine capacity
requires deleting a non-source expert copy. It does not enumerate anticipatory
boundary eviction, prefetch, overlap, inactive expert loading or recomputation.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation, compile_invocation
from llm_infer_model.m2.runner import LayerInvocationResult, run_invocation
from llm_infer_model.m2.spec import state_in_from_state_out
from llm_infer_model.tensor.engine import MappingError, STATUS_VALID
from llm_infer_model.tensor.spec import InitialCapacityExceeded, InvalidInput, Unsupported

from .m2_cache import CacheChoiceRequired, CacheHistory, CachePlan, CachePolicy, expert_accesses, plan_cache

SEARCH_SPACE = "serial_jit_macro_placements_and_demand_pressure_expert_copy_evictions"


@dataclass(frozen=True)
class CacheInvocationResult:
    compiled: CompiledInvocation
    placements: tuple[tuple[str, str], ...]
    plan: CachePlan | None
    result: LayerInvocationResult | None
    error: MappingError | InitialCapacityExceeded | None = None

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
class CacheSequenceCandidate:
    index: int
    policy: CachePolicy
    invocations: tuple[CacheInvocationResult, ...]
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
        if not self.is_valid:
            return None
        return sum(item.result.makespan_ns for item in self.invocations)

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


@dataclass(frozen=True)
class CacheSearchResult:
    templates: tuple[CompiledInvocation, ...]
    candidates: tuple[CacheSequenceCandidate, ...]
    best_index: int | None
    max_candidates: int
    max_expanded_states: int
    expanded_states: int
    declared_placement_count: int
    placements_visited: int
    enumeration_complete: bool
    termination_reason: str
    fixed_placements: bool

    @property
    def best(self) -> CacheSequenceCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]

    @property
    def ranking(self) -> tuple[CacheSequenceCandidate, ...]:
        return tuple(sorted((c for c in self.candidates if c.is_valid),
                            key=lambda c: (c.total_makespan_ns, c.index)))

    @property
    def status(self) -> str:
        if self.enumeration_complete:
            return "optimal_within_declared_space" if self.best is not None else "infeasible_within_declared_space"
        return "feasible" if self.best is not None else "unknown"


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput(f"{label} must be a positive integer")
    return value


def _templates(templates: Sequence[CompiledInvocation]) -> tuple[CompiledInvocation, ...]:
    result = tuple(templates)
    if not result or any(not isinstance(item, CompiledInvocation) for item in result):
        raise InvalidInput("cache sequence requires a nonempty sequence of CompiledInvocation templates")
    first = result[0]
    if first.layer.format_version != "m2-layer-0.6" or first.layer.ffn != "moe":
        raise Unsupported("cache sequence requires synthetic m2-layer-0.6 MoE")
    if any(item.layer != first.layer for item in result):
        raise InvalidInput("cache sequence must preserve the exact same immutable layer definition")
    return result


def _placement_choices(templates: tuple[CompiledInvocation, ...]) -> tuple[tuple[str, ...], ...]:
    choices = []
    for item in templates:
        for task in item.tasks:
            allowed = item.scenario.mapspace.devices_for(task.operation_id)
            if not allowed or any(device not in {"cpu", "gpu"} for device in allowed):
                raise InvalidInput("cache search needs declared CPU/GPU choices for every macro")
            choices.append(tuple(device for device in ("cpu", "gpu") if device in allowed))
    return tuple(choices)


def _placements(templates: tuple[CompiledInvocation, ...], placements):
    if placements is None:
        return tuple({t.operation_id: t.device for t in item.tasks} for item in templates)
    result = tuple(dict(p) for p in placements)
    if len(result) != len(templates):
        raise InvalidInput("placements must supply exactly one map per invocation")
    for template, placement in zip(templates, result):
        if set(placement) != set(template.compute_ids):
            raise InvalidInput("placement map must name exactly every macro in its invocation")
        for operation, device in placement.items():
            if device not in template.scenario.mapspace.devices_for(operation):
                raise InvalidInput(f"device {device!r} is not allowed for {operation!r}")
    return result


def run_cache_sequence(
    templates: Sequence[CompiledInvocation], *, placements: Sequence[Mapping[str, str]] | None = None,
    policy: CachePolicy | None = None, decisions: Sequence[int] = (), index: int = 0,
) -> CacheSequenceCandidate:
    """Replay a policy while binding every later call to the preceding state_out.

    The later template's declared history is not substituted into the episode:
    it supplies invocation/cost/schema metadata; actual cache and KV residency
    are recompiled from the previous successful result. Schema/cost refusals
    propagate. Mapping/initial-capacity failures remain explicit outcomes.
    """
    templates = _templates(templates)
    selected = _placements(templates, placements)
    policy = CachePolicy() if policy is None else policy
    policy.validate()
    path = tuple(decisions)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in path):
        raise InvalidInput("cache decisions must be nonnegative integer choice indices")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise InvalidInput("candidate index must be a nonnegative integer")
    history = CacheHistory(templates[0].layer.id)
    invocations = []
    cursor = 0
    state_in = templates[0].state_in
    accesses = tuple(expert_accesses(item, placement) for item, placement in zip(templates, selected))
    for i, (template, placement) in enumerate(zip(templates, selected)):
        compiled = compile_invocation(template.layer, template.invocation, template.cost_table,
            state_in=state_in, where=f"cache_sequence[{i}]")
        future = tuple(item for sequence in accesses[i + 1:] for item in sequence)
        try:
            plan = plan_cache(compiled, placements=placement, policy=policy, history=history,
                              decisions=path[cursor:], future_accesses=future)
        except CacheChoiceRequired as choice:
            raise CacheChoiceRequired(choice.choices, decision_index=cursor + choice.decision_index,
                                      actions=choice.actions, invocation_index=i) from choice
        except (MappingError, InitialCapacityExceeded) as error:
            invocations.append(CacheInvocationResult(compiled, tuple(placement.items()), None, None, error))
            return CacheSequenceCandidate(index, policy, tuple(invocations), len(templates), path)
        cursor += len(plan.decisions)
        result = run_invocation(compiled, actions=plan.actions)
        invocations.append(CacheInvocationResult(compiled, tuple(placement.items()), plan, result))
        if result.status != STATUS_VALID:
            return CacheSequenceCandidate(index, policy, tuple(invocations), len(templates), path)
        history = plan.history
        state_in = state_in_from_state_out(result.state_out)
    if cursor != len(path):
        raise InvalidInput("completed cache decision path contains unused trailing choices")
    return CacheSequenceCandidate(index, policy, tuple(invocations), len(templates), path)


def search_cache_sequence(
    templates: Sequence[CompiledInvocation], *, placements: Sequence[Mapping[str, str]] | None = None,
    policy: CachePolicy | None = None, max_candidates: int = 256, max_expanded_states: int = 4096,
) -> CacheSearchResult:
    """Explore bounded offline eviction paths, with fixed or enumerated devices.

    A candidate is a whole episode (or its first explicit mapping refusal).
    ``max_expanded_states`` counts decision-prefix replay attempts; this prevents
    an eviction tree from consuming unbounded work before producing one leaf.
    A completely visited space can establish optimality only among these fixed
    task-order, pressure-triggered plans. Incomplete searches return feasible or
    unknown and never infer global infeasibility from unvisited branches.
    """
    templates = _templates(templates)
    candidate_budget = _positive(max_candidates, "max_candidates")
    state_budget = _positive(max_expanded_states, "max_expanded_states")
    policy = CachePolicy(kind="branch") if policy is None else policy
    policy.validate()
    if policy.kind != "branch":
        raise InvalidInput("cache search requires branch policy; use run_cache_sequence for baselines")
    fixed = placements is not None
    if fixed:
        selected_fixed = _placements(templates, placements)
        placement_iter = iter((selected_fixed,))
        declared_placement_count = 1
    else:
        choices = _placement_choices(templates)
        declared_placement_count = math.prod(len(choice) for choice in choices)
        def assignments():
            for devices in itertools.product(*choices):
                cursor = 0
                episode = []
                for item in templates:
                    count = len(item.tasks)
                    episode.append(dict(zip(item.compute_ids, devices[cursor:cursor + count])))
                    cursor += count
                yield tuple(episode)
        placement_iter = assignments()
    candidates = []
    expanded = 0
    visited = 0
    complete = True
    reason = "exhausted_declared_space"
    def children(prefix, count):
        # Keep one lazy sibling iterator per depth. A large expert vocabulary
        # must not materialize thousands of copies of every decision prefix.
        return (prefix + (n,) for n in range(count))
    for selected in placement_iter:
        visited += 1
        frontier = [iter(((),))]
        while frontier:
            path = None
            while frontier:
                try:
                    path = next(frontier[-1])
                    break
                except StopIteration:
                    frontier.pop()
            if path is None:
                break
            if len(candidates) >= candidate_budget or expanded >= state_budget:
                complete = False
                reason = "max_candidates" if len(candidates) >= candidate_budget else "max_expanded_states"
                break
            expanded += 1
            try:
                candidate = run_cache_sequence(templates, placements=selected, policy=policy,
                                               decisions=path, index=len(candidates))
            except CacheChoiceRequired as choice:
                if choice.decision_index != len(path) or not choice.choices:
                    raise InvalidInput("cache branch did not extend its exact decision prefix")
                frontier.append(children(path, len(choice.choices)))
            else:
                candidates.append(candidate)
        if not complete:
            break
        # A budget hit exactly on the final leaf is complete only if no further
        # placements remain. Peek one item rather than falsely assuming either.
        if len(candidates) >= candidate_budget or expanded >= state_budget:
            marker = next(placement_iter, None)
            if marker is not None:
                complete = False
                reason = "max_candidates" if len(candidates) >= candidate_budget else "max_expanded_states"
            break
    best = min((c for c in candidates if c.is_valid),
               key=lambda c: (c.total_makespan_ns, c.index), default=None)
    return CacheSearchResult(templates, tuple(candidates), None if best is None else best.index,
        candidate_budget, state_budget, expanded, declared_placement_count, visited, complete, reason, fixed)
