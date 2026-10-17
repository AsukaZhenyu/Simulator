"""Historical MoE search API, forwarded to the shared tensor episode UCS.

This facade preserves expert-only quotas and historical result records. The
search implementation and all ordering, overlap and budget semantics live in
the model-neutral schedule_search module; model binding lives in m2_schedule.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation
from llm_infer_model.tensor.spec import InvalidInput

from .m2_cache import CachePolicy
from .m2_cache_search import CacheInvocationResult, CacheSequenceCandidate
from .m2_overlap import SchedulePolicy, to_generic_history, to_generic_policy, to_legacy_plan
from .m2_schedule import M2ScheduleAdapter
from .schedule import SchedulePlan
from .schedule_search import (
    ScheduleInvocationResult, ScheduleSearchResult, ScheduleSequenceCandidate,
    search_schedule_episode as _search_schedule_episode,
)

SEARCH_SPACE = "global_episode_fixed_macro_start_order_windowed_prefetch_concurrent_resources"


def _generic_seed(seed, adapter, policy):
    if not isinstance(seed, CacheSequenceCandidate):
        raise InvalidInput("schedule incumbent must be a completed CacheSequenceCandidate for this episode")
    invocations = []
    for item in seed.invocations:
        plan = item.plan
        if plan is not None:
            plan = SchedulePlan(plan.actions, to_generic_history(plan.history), plan.decisions,
                plan.eviction_count, plan.cache_hits, plan.cache_misses, plan.cache_hit_by_domain,
                plan.cache_miss_by_domain, getattr(plan, "termination_reason", "goal_reached"),
                item.result is not None and item.result.status == "valid")
        boundary_in = adapter.boundary_from_compiled(item.compiled)
        boundary_out = None if item.result is None or item.result.state_out is None else (
            adapter.boundary_from_result(item.compiled, item.result))
        invocations.append(ScheduleInvocationResult(item.compiled, item.placements, plan,
            item.result, item.error, boundary_in, boundary_out))
    return ScheduleSequenceCandidate(seed.index, policy, tuple(invocations),
        seed.declared_invocation_count, seed.decisions)


def _legacy_candidate(candidate, policy):
    calls = tuple(CacheInvocationResult(item.compiled, item.placements,
        None if item.plan is None else to_legacy_plan(item.plan), item.result, item.error)
        for item in candidate.invocations)
    cache_policy = CachePolicy(kind="branch",
        expert_cache_capacity_bytes=dict(policy.expert_cache_capacity_bytes))
    return CacheSequenceCandidate(candidate.index, cache_policy, calls,
                                  candidate.declared_invocation_count, candidate.decisions)


def search_schedule_episode(
    templates: Sequence[CompiledInvocation], *, policy: SchedulePolicy,
    placements: Sequence[Mapping[str, str]] | None = None,
    max_expanded_states: int, max_generated_states: int,
    incumbent: CacheSequenceCandidate | None = None,
) -> ScheduleSearchResult:
    adapter = M2ScheduleAdapter(reusable_scope="experts")
    generic_policy = to_generic_policy(policy)
    generic_seed = None if incumbent is None else _generic_seed(incumbent, adapter, generic_policy)
    result = _search_schedule_episode(templates, adapter=adapter, policy=generic_policy,
        placements=placements, max_expanded_states=max_expanded_states,
        max_generated_states=max_generated_states, incumbent=generic_seed)
    return replace(result, policy=policy,
        candidates=tuple(_legacy_candidate(candidate, policy) for candidate in result.candidates))


search_overlap_episode = search_schedule_episode

__all__ = ["SEARCH_SPACE", "ScheduleSearchResult", "search_schedule_episode", "search_overlap_episode"]
