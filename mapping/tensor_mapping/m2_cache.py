"""Serial MoE expert-cache policies validated by the shared tensor engine.

Only non-source DRAM/VRAM copies of expert weights are cache entries. Cache
quotas are an explicit policy restriction on aligned resident bytes, in addition
to the engine's complete capacity ledger. KV, activations and immutable weight
sources are never deleted. This module introduces no measured performance cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation
from llm_infer_model.m2.spec import weight_id
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    ACTION_WRITE_SSD, CODE_CAPACITY_EXCEEDED, Action, MappingError,
    initial_state, transition,
)
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported

POLICIES = ("keep_all", "drop_all", "lru", "fifo", "offline_farthest", "branch")
_DOMAINS = ("dram", "vram")


@dataclass(frozen=True)
class CachePolicy:
    kind: str = "lru"
    expert_cache_capacity_bytes: Mapping[str, int | None] = field(
        default_factory=lambda: {"dram": None, "vram": None})

    def validate(self) -> None:
        if self.kind not in POLICIES:
            raise InvalidInput(f"cache policy must be one of {POLICIES}")
        if not isinstance(self.expert_cache_capacity_bytes, Mapping) or set(
                self.expert_cache_capacity_bytes) != set(_DOMAINS):
            raise InvalidInput("expert_cache_capacity_bytes must explicitly name dram and vram")
        for domain, value in self.expert_cache_capacity_bytes.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise InvalidInput(f"{domain} expert cache quota must be a nonnegative integer or null")


@dataclass(frozen=True)
class CacheStamp:
    tensor_id: str
    location: str
    last_use: int
    loaded_at: int


@dataclass(frozen=True)
class CacheHistory:
    """Policy recency metadata; tensor residency still comes only from state_out.

    A preexisting copy absent from this history is deterministically seeded at
    time zero. We do not invent its past LRU/FIFO order.
    """
    layer_id: str | None = None
    clock: int = 0
    entries: tuple[CacheStamp, ...] = ()


@dataclass(frozen=True)
class CachePlan:
    actions: tuple[Action, ...]
    history: CacheHistory
    decisions: tuple[int, ...]
    eviction_count: int
    cache_hits: int
    cache_misses: int
    cache_hit_by_domain: Mapping[str, int]
    cache_miss_by_domain: Mapping[str, int]


class CacheChoiceRequired(Exception):
    """A finite legal eviction branch encountered by the offline search."""
    def __init__(self, choices: Sequence[tuple[str, str]], *, decision_index: int,
                 actions: Sequence[Action] = (), invocation_index: int | None = None):
        super().__init__("an expert-cache eviction choice is required")
        self.choices = tuple(choices)
        self.decision_index = decision_index
        self.actions = tuple(actions)
        self.invocation_index = invocation_index


def expert_weight_ids(compiled: CompiledInvocation) -> frozenset[str]:
    ids = list(compiled.layer.ffn_params["expert_weights"])
    shared = compiled.layer.ffn_params.get("shared_expert_weight")
    if shared is not None:
        ids.append(shared)
    return frozenset(weight_id(compiled.layer.id, name) for name in ids)


def expert_accesses(compiled: CompiledInvocation, placements: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    weights = expert_weight_ids(compiled)
    return tuple((tensor, "dram" if placements[task.operation_id] == "cpu" else "vram")
        for task in compiled.tasks if task.kind in {"expert_ffn", "shared_expert_ffn"}
        for tensor in compiled.scenario.workload.operation_by_id[task.operation_id].inputs if tensor in weights)


def plan_cache(
    compiled: CompiledInvocation, *, placements: Mapping[str, str] | None = None,
    policy: CachePolicy | None = None, history: CacheHistory | None = None,
    decisions: Sequence[int] = (), future_accesses: Sequence[tuple[str, str]] = (),
) -> CachePlan:
    """Plan fixed-order JIT execution with demand-driven expert-copy eviction.

    ``branch`` consumes an integer path selecting among legal eviction choices;
    an unfinished path raises CacheChoiceRequired. Extra decisions are returned
    unconsumed by virtue of CachePlan.decisions containing only choices used.
    Sequence search validates that a completed path has no trailing decisions.
    Every committed action is first applied through engine.transition.
    """
    if compiled.layer.format_version != "m2-layer-0.6" or compiled.layer.ffn != "moe":
        raise Unsupported("expert cache planning requires synthetic m2-layer-0.6 MoE")
    selected = ({t.operation_id: t.device for t in compiled.tasks} if placements is None else dict(placements))
    if set(selected) != set(compiled.compute_ids):
        raise InvalidInput("placements must name exactly every compiled operation_id")
    for op, device in selected.items():
        if device not in compiled.scenario.mapspace.devices_for(op):
            raise InvalidInput(f"device {device!r} is not allowed for {op!r}")
    policy = CachePolicy() if policy is None else policy
    policy.validate()
    path = tuple(decisions)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in path):
        raise InvalidInput("cache decisions must be nonnegative integer choice indices")
    scenario = compiled.scenario
    state = initial_state(scenario)  # No invented eviction before the initial capacity check.
    experts = expert_weight_ids(compiled)
    # The authoritative snapshot may have migrated the original declared
    # location. Preserve the current persistent source selected by expansion;
    # never resurrect layer.initial_location or reset immutable versions.
    sources = {weight_id(compiled.layer.id, ref.id): scenario.goal_location(
        weight_id(compiled.layer.id, ref.id)) for ref in compiled.layer.weights}
    for tensor, source in sources.items():
        if state.copy_at(tensor, source) != COPY_READY:
            raise Unsupported(f"cache planning requires current persistent source {tensor!r} READY in {source}")
    history = CacheHistory(compiled.layer.id) if history is None else history
    if history.layer_id not in {None, compiled.layer.id}:
        raise InvalidInput("cache policy history belongs to another layer")
    if isinstance(history.clock, bool) or not isinstance(history.clock, int) or history.clock < 0:
        raise InvalidInput("cache history clock must be a nonnegative integer")
    stamps: dict[tuple[str, str], tuple[int, int]] = {}
    for entry in history.entries:
        key = (entry.tensor_id, entry.location)
        if key in stamps or entry.tensor_id not in experts or entry.location not in _DOMAINS:
            raise InvalidInput("cache history entries must name unique expert copies in dram/vram")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > history.clock
               for value in (entry.last_use, entry.loaded_at)):
            raise InvalidInput("cache recency values must be integer clocks within history.clock")
        if entry.location != sources[entry.tensor_id] and state.copy_at(*key) == COPY_READY:
            stamps[key] = (entry.last_use, entry.loaded_at)
    for tensor in sorted(experts):
        for domain in _DOMAINS:
            if domain != sources[tensor] and state.copy_at(tensor, domain) == COPY_READY:
                stamps.setdefault((tensor, domain), (0, 0))
    clock = history.clock
    actions: list[Action] = []
    consumed: list[int] = []
    hits, misses = {domain: 0 for domain in _DOMAINS}, {domain: 0 for domain in _DOMAINS}
    eviction_count = 0
    accesses = expert_accesses(compiled, selected) + tuple(future_accesses)
    access_cursor = 0
    for tensor, domain in future_accesses:
        if tensor not in experts or domain not in _DOMAINS:
            raise InvalidInput("future_accesses must name this layer's experts and explicit dram/vram domains")

    def cached(domain: str) -> tuple[str, ...]:
        return tuple(t for t in sorted(experts) if sources[t] != domain and state.copy_at(t, domain) == COPY_READY)

    def cache_bytes(domain: str) -> int:
        return sum(scenario.size_alloc(t, domain) for t in cached(domain))

    def commit(action: Action) -> None:
        nonlocal state, clock, eviction_count
        state = transition(scenario, state, action)
        actions.append(action)
        if action.kind == ACTION_EVICT:
            stamps.pop((action.target_id, action.location), None)
            eviction_count += 1
        if action.kind in {ACTION_COPY, ACTION_WRITE_SSD} and action.target_id in experts:
            source = "dram" if action.kind == ACTION_WRITE_SSD else action.src
            key = (action.target_id, source)
            if key in stamps:
                clock += 1
                stamps[key] = (clock, stamps[key][1])
        if action.kind in {ACTION_COPY, ACTION_READ_SSD} and action.target_id in experts:
            domain = "dram" if action.kind == ACTION_READ_SSD else action.dst
            if domain != sources[action.target_id]:
                clock += 1
                stamps[(action.target_id, domain)] = (clock, clock)

    def eviction_options(domain: str, protected: set[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        legal = []
        for tensor in cached(domain):
            key = (tensor, domain)
            if key in protected:
                continue
            try:
                transition(scenario, state, Action(ACTION_EVICT, tensor, location=domain))
            except MappingError:
                continue
            legal.append(key)
        if policy.kind in {"lru", "branch", "drop_all"}:
            legal.sort(key=lambda key: (stamps.get(key, (0, 0))[0], key))
        elif policy.kind == "fifo":
            legal.sort(key=lambda key: (stamps.get(key, (0, 0))[1], key))
        elif policy.kind == "offline_farthest":
            future = accesses[access_cursor:]
            def next_use(key):
                # A future GPU demand can read DRAM staging. This conservative
                # projection does not assume the VRAM copy will miss, and is a
                # lookahead heuristic rather than a cache-optimality theorem.
                return next((i for i, item in enumerate(future) if item == key or (
                    key[1] == "dram" and item == (key[0], "vram"))), len(future) + 1)
            legal.sort(key=lambda key: (-next_use(key), key))
        return tuple(legal)

    def evict_one(domain: str, protected: set[tuple[str, str]], error: MappingError) -> None:
        choices = eviction_options(domain, protected)
        if policy.kind == "keep_all" or not choices:
            error.actions = tuple(actions)
            if error.action_index is None:
                error.action_index = len(actions)
            raise error
        index = 0
        # A forced release adds no search alternative. Apply it directly rather
        # than consuming a branch budget merely to choose the only legal copy.
        if policy.kind == "branch" and len(choices) > 1:
            if len(consumed) == len(path):
                raise CacheChoiceRequired(choices, decision_index=len(consumed), actions=actions)
            index = path[len(consumed)]
            if index >= len(choices):
                raise InvalidInput("cache decision choice index is outside the encountered legal choices")
            consumed.append(index)
        commit(Action(ACTION_EVICT, choices[index][0], location=domain))

    def trim(domain: str, protected: set[tuple[str, str]], additional: int = 0) -> None:
        quota = policy.expert_cache_capacity_bytes[domain]
        while quota is not None and cache_bytes(domain) + additional > quota:
            evict_one(domain, protected, MappingError(CODE_CAPACITY_EXCEEDED,
                f"expert cache in {domain} needs {cache_bytes(domain) + additional} aligned bytes, quota={quota}",
                t_ns=state.t))

    def start(action: Action, domain: str, protected: set[tuple[str, str]]) -> None:
        additional = 0
        if action.kind in {ACTION_COPY, ACTION_READ_SSD} and action.target_id in experts and sources[action.target_id] != domain:
            additional = scenario.size_alloc(action.target_id, domain)
        if domain in _DOMAINS:
            trim(domain, protected, additional)
        while True:
            try:
                commit(action)
                return
            except MappingError as error:
                if error.code != CODE_CAPACITY_EXCEEDED:
                    error.actions = tuple(actions)
                    if error.action_index is None:
                        error.action_index = len(actions)
                    raise
                evict_one(domain, protected, error)

    def transfer(action: Action, domain: str, protected: set[tuple[str, str]]) -> None:
        if action.kind == ACTION_COPY:
            protected = protected | {(action.target_id, action.src)}
        start(action, domain, protected)
        commit(Action(ACTION_ADVANCE))

    def materialize(tensor: str, destination: str, protected: set[tuple[str, str]]) -> None:
        if state.copy_at(tensor, destination) == COPY_READY:
            return
        if destination == "ssd":
            materialize(tensor, "dram", protected)
            transfer(Action(ACTION_WRITE_SSD, tensor), "ssd", protected)
        elif destination == "vram":
            materialize(tensor, "dram", protected)
            transfer(Action(ACTION_COPY, tensor, src="dram", dst="vram"), "vram", protected)
        elif state.copy_at(tensor, "vram") == COPY_READY:
            transfer(Action(ACTION_COPY, tensor, src="vram", dst="dram"), "dram", protected)
        elif state.copy_at(tensor, "ssd") == COPY_READY:
            transfer(Action(ACTION_READ_SSD, tensor), "dram", protected)
        else:
            raise InvalidInput(f"serial cache plan has no READY source for {tensor!r}")

    # Quota restrictions also apply to authoritative inherited copies.
    for domain in _DOMAINS:
        trim(domain, set())
    dropped = False
    for task in compiled.tasks:
        device = selected[task.operation_id]
        domain = "dram" if device == "cpu" else "vram"
        operation = scenario.workload.operation_by_id[task.operation_id]
        protected = {(tensor, domain) for tensor in operation.inputs}
        if task.kind == "combine" and policy.kind == "drop_all" and not dropped:
            for cache_domain in _DOMAINS:
                for tensor in cached(cache_domain):
                    commit(Action(ACTION_EVICT, tensor, location=cache_domain))
            dropped = True
        requested = [tensor for tensor in operation.inputs if tensor in experts]
        for tensor in requested:
            (hits if state.copy_at(tensor, domain) == COPY_READY else misses)[domain] += 1
        for tensor in operation.inputs:
            materialize(tensor, domain, protected)
        start(Action(ACTION_COMPUTE, task.operation_id, device=device), domain, protected)
        for tensor in requested:
            clock += 1
            if domain != sources[tensor]:
                previous = stamps.get((tensor, domain), (clock, clock))
                stamps[(tensor, domain)] = (clock, previous[1])
            access_cursor += 1
        commit(Action(ACTION_ADVANCE))
    for tensor, destination in compiled.scenario_document["goal"]["output_locations"].items():
        materialize(tensor, destination, set())
    final_history = CacheHistory(compiled.layer.id, clock, tuple(
        CacheStamp(tensor, domain, *stamp) for (tensor, domain), stamp in sorted(stamps.items())))
    return CachePlan(tuple(actions), final_history, tuple(consumed), eviction_count,
                     sum(hits.values()), sum(misses.values()), hits, misses)
