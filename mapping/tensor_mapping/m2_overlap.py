"""Historical MoE API, delegated to the generic tensor schedule core.

Expert-only copy quotas remain an explicit adapter scope for compatibility.
New Dense/MoE scheduling uses schedule.py and M2ScheduleAdapter directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation
from llm_infer_model.tensor.engine import Action, State
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from .m2_cache import CacheHistory, CachePlan, CacheStamp
from .m2_schedule import M2ScheduleAdapter, compile_schedule_invocation
from . import schedule as generic

_DOMAINS = ("dram", "vram")

@dataclass(frozen=True)
class SchedulePolicy:
    prefetch_depth: int
    overlap: bool
    eviction_kind: str = "lru"
    expert_cache_capacity_bytes: Mapping[str, int | None] = field(
        default_factory=lambda: {"dram": None, "vram": None})

    def validate(self) -> None:
        if isinstance(self.prefetch_depth, bool) or not isinstance(self.prefetch_depth, int) or self.prefetch_depth < 0:
            raise InvalidInput("prefetch_depth must be a nonnegative integer")
        if not isinstance(self.overlap, bool):
            raise InvalidInput("overlap must be an explicit boolean")
        if self.eviction_kind not in {"lru", "fifo", "offline_farthest"}:
            raise InvalidInput("eviction_kind must be lru, fifo or offline_farthest")
        if not isinstance(self.expert_cache_capacity_bytes, Mapping) or set(
                self.expert_cache_capacity_bytes) != set(_DOMAINS):
            raise InvalidInput("expert_cache_capacity_bytes must explicitly name dram and vram")
        for domain, value in self.expert_cache_capacity_bytes.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise InvalidInput(f"{domain} expert cache quota must be nonnegative integer or null")


@dataclass(frozen=True)
class ScheduleContext:
    compiled: CompiledInvocation
    policy: SchedulePolicy
    placements: tuple[tuple[str, str], ...] | None
    experts: frozenset[str]
    sources: tuple[tuple[str, str], ...]
    _core_context: generic.ScheduleContext | None = field(default=None, repr=False, compare=False)

    def devices_for(self, operation_id: str) -> tuple[str, ...]:
        if self.placements is not None:
            return (dict(self.placements)[operation_id],)
        allowed = self.compiled.scenario.mapspace.devices_for(operation_id)
        return tuple(device for device in ("cpu", "gpu") if device in allowed)


@dataclass(frozen=True)
class OverlapPlan(CachePlan):
    termination_reason: str
    complete: bool



def to_generic_policy(policy: SchedulePolicy) -> generic.SchedulePolicy:
    policy.validate()
    return generic.SchedulePolicy(policy.prefetch_depth, policy.overlap, policy.eviction_kind,
                                  policy.expert_cache_capacity_bytes)


def to_generic_history(history: CacheHistory | None):
    if history is None:
        return None
    return generic.ScheduleHistory(history.layer_id, history.clock,
        tuple(generic.ScheduleStamp(e.tensor_id, e.location, e.last_use, e.loaded_at) for e in history.entries))


def to_legacy_plan(plan: generic.SchedulePlan) -> OverlapPlan:
    return OverlapPlan(plan.actions, CacheHistory(plan.history.binding_id, plan.history.clock,
        tuple(CacheStamp(e.tensor_id, e.location, e.last_use, e.loaded_at) for e in plan.history.entries)),
        plan.decisions, plan.eviction_count, plan.cache_hits, plan.cache_misses,
        plan.cache_hit_by_domain, plan.cache_miss_by_domain, plan.termination_reason, plan.complete)


def _generic_context(context: ScheduleContext):
    if context._core_context is not None:
        return context._core_context
    return M2ScheduleAdapter('experts').make_context(
        context.compiled, to_generic_policy(context.policy),
        None if context.placements is None else dict(context.placements))


def make_schedule_context(compiled: CompiledInvocation, policy: SchedulePolicy,
                          placements: Mapping[str, str] | None = None) -> ScheduleContext:
    context = M2ScheduleAdapter('experts').make_context(compiled, to_generic_policy(policy), placements)
    return ScheduleContext(compiled, policy, context.placements,
        context.facts.managed_reusable_tensors, context.facts.authoritative_sources, context)


def next_macro_index(context: ScheduleContext, state: State) -> int:
    return generic.next_operation_index(_generic_context(context), state)


def preparation_window(context: ScheduleContext, state: State):
    start = next_macro_index(context, state)
    return context.compiled.tasks[start:start + context.policy.prefetch_depth + 1]


def expert_pool_bytes(context: ScheduleContext, state: State, domain: str) -> int:
    return generic.managed_pool_bytes(_generic_context(context), state, domain)


def legal_schedule_actions(context: ScheduleContext, state: State) -> tuple[Action, ...]:
    return generic.legal_schedule_actions(_generic_context(context), state)


def step_schedule(context: ScheduleContext, state: State, action: Action) -> State:
    return generic.step_schedule(_generic_context(context), state, action)


def replay_schedule_plan(context: ScheduleContext, actions: Sequence[Action],
                         history: CacheHistory | None = None, *, require_complete: bool = False) -> OverlapPlan:
    return to_legacy_plan(generic.replay_schedule_plan(_generic_context(context), actions,
        to_generic_history(history), require_complete=require_complete))


def plan_overlap(compiled: CompiledInvocation, *, policy: SchedulePolicy,
                 placements: Mapping[str, str] | None = None, history: CacheHistory | None = None,
                 max_actions: int, future_accesses: Sequence[tuple[str, str]] = ()) -> OverlapPlan:
    selected = {task.operation_id: task.device for task in compiled.tasks} if placements is None else dict(placements)
    context = _generic_context(make_schedule_context(compiled, policy, selected))
    return to_legacy_plan(generic.plan_schedule(context, history=to_generic_history(history),
        max_actions=max_actions, future_accesses=future_accesses))
