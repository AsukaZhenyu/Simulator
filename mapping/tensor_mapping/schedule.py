"""Model-neutral event-driven scheduling over a heterogeneous tensor graph.

Workload adapters explicitly declare operation order, reusable tensors, source
copies and protected values. This core does not classify tensor names or model
families. The tensor engine is the only implementation of execution legality,
physical memory, resource conflicts and transfer/compute dependency barriers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    ACTION_WRITE_SSD, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION,
    Action, MappingError, State, initial_state, is_goal, transition,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT, COPY_READY, OP_NOT_STARTED, InvalidInput, Scenario, Unsupported,
)

_DOMAINS = ("dram", "vram")


@dataclass(frozen=True)
class SchedulePolicy:
    prefetch_depth: int
    overlap: bool
    eviction_kind: str = "lru"
    managed_capacity_bytes: Mapping[str, int | None] = field(
        default_factory=lambda: {"dram": None, "vram": None})

    def validate(self) -> None:
        if isinstance(self.prefetch_depth, bool) or not isinstance(self.prefetch_depth, int) or self.prefetch_depth < 0:
            raise InvalidInput("prefetch_depth must be a nonnegative integer")
        if not isinstance(self.overlap, bool):
            raise InvalidInput("overlap must be an explicit boolean")
        if self.eviction_kind not in {"lru", "fifo", "offline_farthest"}:
            raise InvalidInput("eviction_kind must be lru, fifo or offline_farthest")
        if not isinstance(self.managed_capacity_bytes, Mapping) or set(self.managed_capacity_bytes) != set(_DOMAINS):
            raise InvalidInput("managed_capacity_bytes must explicitly name dram and vram")
        for domain, value in self.managed_capacity_bytes.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise InvalidInput(f"{domain} managed quota must be nonnegative integer or null")


@dataclass(frozen=True)
class ScheduleFacts:
    operation_order: tuple[str, ...]
    managed_reusable_tensors: frozenset[str]
    authoritative_sources: tuple[tuple[str, str], ...]
    protected_tensors: frozenset[str] = frozenset()
    binding_id: str | None = None


@dataclass(frozen=True)
class ScheduleStamp:
    tensor_id: str
    location: str
    last_use: int
    loaded_at: int


@dataclass(frozen=True)
class ScheduleHistory:
    binding_id: str | None = None
    clock: int = 0
    entries: tuple[ScheduleStamp, ...] = ()


@dataclass(frozen=True)
class SchedulePlan:
    actions: tuple[Action, ...]
    history: ScheduleHistory
    decisions: tuple[int, ...]
    eviction_count: int
    cache_hits: int
    cache_misses: int
    cache_hit_by_domain: Mapping[str, int]
    cache_miss_by_domain: Mapping[str, int]
    termination_reason: str
    complete: bool


@dataclass(frozen=True)
class ScheduleContext:
    scenario: Scenario
    facts: ScheduleFacts
    policy: SchedulePolicy
    placements: tuple[tuple[str, str], ...] | None
    payload: object = None

    def devices_for(self, operation_id: str) -> tuple[str, ...]:
        if self.placements is not None:
            return (dict(self.placements)[operation_id],)
        allowed = self.scenario.mapspace.devices_for(operation_id)
        return tuple(device for device in ("cpu", "gpu") if device in allowed)


def make_schedule_context(scenario: Scenario, facts: ScheduleFacts, policy: SchedulePolicy,
                          placements: Mapping[str, str] | None = None, *, payload: object = None) -> ScheduleContext:
    policy.validate()
    if not scenario.is_m1:
        raise Unsupported("heterogeneous scheduling requires tensor scenario 0.2")
    operations = scenario.workload.operation_by_id
    tensors = scenario.workload.tensor_by_id
    order = tuple(facts.operation_order)
    if len(set(order)) != len(order) or set(order) != set(operations):
        raise InvalidInput("operation_order must name every operation exactly once")
    managed = frozenset(facts.managed_reusable_tensors)
    protected = frozenset(facts.protected_tensors)
    if not managed <= set(tensors) or not protected <= set(tensors) or managed & protected:
        raise InvalidInput("managed/protected tensor sets must be known and disjoint")
    produced = {tensor for op in operations.values() for tensor in op.outputs}
    if managed & produced:
        raise InvalidInput("managed reusable tensors must be immutable graph leaves")
    sources = tuple(facts.authoritative_sources)
    if len({tensor for tensor, _ in sources}) != len(sources) or {tensor for tensor, _ in sources} != managed:
        raise InvalidInput("authoritative_sources must name each managed reusable tensor exactly once")
    for tensor, domain in sources:
        if domain not in {"dram", "vram", "ssd"} or domain not in tensors[tensor].initial_locations:
            raise InvalidInput(f"authoritative source for {tensor!r} must be an initial READY copy")
    if facts.binding_id is not None and (not isinstance(facts.binding_id, str) or not facts.binding_id):
        raise InvalidInput("binding_id must be a nonempty string or null")
    selected = None if placements is None else dict(placements)
    if selected is not None:
        if set(selected) != set(order):
            raise InvalidInput("placements must name exactly every operation")
        for op, device in selected.items():
            if device not in {"cpu", "gpu"} or device not in scenario.mapspace.devices_for(op):
                raise InvalidInput(f"device {device!r} is not allowed for {op!r}")
    for op in order:
        allowed = scenario.mapspace.devices_for(op)
        if not allowed or any(device not in {"cpu", "gpu"} for device in allowed):
            raise InvalidInput("each operation requires declared CPU/GPU device choices")
    facts = ScheduleFacts(order, managed, sources, protected, facts.binding_id)
    return ScheduleContext(scenario, facts, policy, None if selected is None else tuple(selected.items()), payload)


def next_operation_index(context: ScheduleContext, state: State) -> int:
    return next((i for i, task in enumerate(context.facts.operation_order)
                 if state.op(task) == OP_NOT_STARTED), len(context.facts.operation_order))


def preparation_window(context: ScheduleContext, state: State):
    start = next_operation_index(context, state)
    return context.facts.operation_order[start:start + context.policy.prefetch_depth + 1]


def managed_pool_bytes(context: ScheduleContext, state: State, domain: str) -> int:
    if domain not in _DOMAINS:
        raise InvalidInput("managed reusable tensor pool is only defined in dram/vram")
    source = dict(context.facts.authoritative_sources)
    return sum(context.scenario.size_alloc(tensor, domain) for tensor in context.facts.managed_reusable_tensors
        if source[tensor] != domain and state.copy_at(tensor, domain) != COPY_ABSENT)


def _over_quota(context: ScheduleContext, state: State) -> tuple[str, ...]:
    return tuple(domain for domain in _DOMAINS if context.policy.managed_capacity_bytes[domain] is not None
        and managed_pool_bytes(context, state, domain) > context.policy.managed_capacity_bytes[domain])


def _destination(action: Action) -> str | None:
    if action.kind == ACTION_COMPUTE:
        return "dram" if action.device == "cpu" else "vram"
    if action.kind == ACTION_COPY:
        return action.dst
    if action.kind == ACTION_READ_SSD:
        return "dram"
    if action.kind == ACTION_WRITE_SSD:
        return "ssd"
    return None


def _transfer_demands(state: State, tensor: str, destination: str) -> tuple[Action, ...]:
    # A reservation already owns destination bytes. A second transfer cannot
    # replace it or treat the not-yet-arrived value as READY.
    if state.copy_at(tensor, destination) != COPY_ABSENT:
        return ()
    if destination == "vram":
        if state.copy_at(tensor, "dram") == COPY_READY:
            return (Action(ACTION_COPY, tensor, src="dram", dst="vram"),)
        if state.copy_at(tensor, "dram") == COPY_ABSENT and state.copy_at(tensor, "ssd") == COPY_READY:
            return (Action(ACTION_READ_SSD, tensor),)
        return ()
    if destination == "dram":
        result = []
        if state.copy_at(tensor, "vram") == COPY_READY:
            result.append(Action(ACTION_COPY, tensor, src="vram", dst="dram"))
        if state.copy_at(tensor, "ssd") == COPY_READY:
            result.append(Action(ACTION_READ_SSD, tensor))
        return tuple(result)
    if destination == "ssd":
        if state.copy_at(tensor, "dram") == COPY_READY:
            return (Action(ACTION_WRITE_SSD, tensor),)
        return _transfer_demands(state, tensor, "dram")
    raise InvalidInput("schedule transfer destination must be dram/vram/ssd")


def _demands(context: ScheduleContext, state: State) -> tuple[tuple[Action, tuple[int, ...]], ...]:
    demands: dict[Action, tuple[int, ...]] = {}
    def add(action, rank):
        if action not in demands or rank < demands[action]:
            demands[action] = rank
    window = preparation_window(context, state)
    if window:
        for device_index, device in enumerate(context.devices_for(window[0])):
            add(Action(ACTION_COMPUTE, window[0], device=device), (0, 0, device_index, 0))
    for distance, task in enumerate(window):
        op = context.scenario.workload.operation_by_id[task]
        for device_index, device in enumerate(context.devices_for(task)):
            domain = "dram" if device == "cpu" else "vram"
            for port, tensor in enumerate(op.inputs):
                for route_index, action in enumerate(_transfer_demands(state, tensor, domain)):
                    add(action, (1 if distance == 0 else 2, distance, port, device_index, route_index))
    for port, (tensor, domain) in enumerate(context.scenario.goal.output_locations.items()):
        for route_index, action in enumerate(_transfer_demands(state, tensor, domain)):
            add(action, (2, len(context.facts.operation_order) + 1, port, route_index))
    return tuple(sorted(demands.items(), key=lambda item: (item[1], str(item[0]))))


def _legal_details(context: ScheduleContext, state: State):
    if is_goal(context.scenario, state):
        return {}, {}
    scenario = context.scenario
    ranks = {}
    pressure = {}
    over = _over_quota(context, state)
    for domain in over:
        pressure[domain] = (0, 0)
    if not over:
        for action, rank in _demands(context, state):
            if state.running and not context.policy.overlap:
                continue
            try:
                successor = transition(scenario, state, action)
            except MappingError as error:
                if error.code == CODE_CAPACITY_EXCEEDED:
                    domain = _destination(action)
                    if domain in _DOMAINS:
                        pressure[domain] = min(pressure.get(domain, rank), rank)
                continue
            violations = _over_quota(context, successor)
            if violations:
                for domain in violations:
                    pressure[domain] = min(pressure.get(domain, rank), rank)
            else:
                ranks[action] = rank
    sources = dict(context.facts.authoritative_sources)
    for domain, rank in pressure.items():
        for tensor in sorted(context.facts.managed_reusable_tensors):
            if sources[tensor] == domain or state.copy_at(tensor, domain) != COPY_READY:
                continue
            action = Action(ACTION_EVICT, tensor, location=domain)
            try:
                transition(scenario, state, action)
            except MappingError:
                continue
            ranks[action] = (3, *rank)
    if state.running:
        ranks[Action(ACTION_ADVANCE)] = (4,)
    return ranks, pressure


def legal_schedule_actions(context: ScheduleContext, state: State) -> tuple[Action, ...]:
    """All legal actions in the declared window/order/pressure-eviction space.

    Ready and RESERVED managed reusable tensor copies both consume the cache quota. Copy and
    compute reservations, read locks, router gates and hardware slots are
    checked by transition, including for prefetched inputs. No inactive managed reusable tensor
    or tensor outside the current preparation window is speculatively loaded.
    """
    ranks, _ = _legal_details(context, state)
    return tuple(sorted(ranks, key=lambda action: (ranks[action], str(action))))


def step_schedule(context: ScheduleContext, state: State, action: Action) -> State:
    if action not in legal_schedule_actions(context, state):
        raise MappingError(CODE_ILLEGAL_ACTION, "action is outside the declared legal schedule space",
                           t_ns=state.t, tensor_id=action.target_id)
    return transition(context.scenario, state, action)


class _Observer:
    def __init__(self, context: ScheduleContext, state: State, history: ScheduleHistory | None):
        self.context = context
        self.sources = dict(context.facts.authoritative_sources)
        history = ScheduleHistory(context.facts.binding_id) if history is None else history
        if history.binding_id not in {None, context.facts.binding_id}:
            raise InvalidInput("schedule cache history belongs to another workload binding")
        if isinstance(history.clock, bool) or not isinstance(history.clock, int) or history.clock < 0:
            raise InvalidInput("cache history clock must be a nonnegative integer")
        self.clock = history.clock
        self.stamps = {}
        seen = set()
        for entry in history.entries:
            key = (entry.tensor_id, entry.location)
            if key in seen or entry.tensor_id not in context.facts.managed_reusable_tensors or entry.location not in _DOMAINS:
                raise InvalidInput("cache history requires unique managed reusable tensor copies in dram/vram")
            seen.add(key)
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > history.clock
                   for n in (entry.last_use, entry.loaded_at)):
                raise InvalidInput("cache history recency must be integer clocks within history.clock")
            if entry.location != self.sources[entry.tensor_id] and state.copy_at(*key) == COPY_READY:
                self.stamps[key] = (entry.last_use, entry.loaded_at)
        for tensor in sorted(context.facts.managed_reusable_tensors):
            for domain in _DOMAINS:
                if self.sources[tensor] != domain and state.copy_at(tensor, domain) == COPY_READY:
                    self.stamps.setdefault((tensor, domain), (0, 0))
        self.first_demand = {}
        self.hits = {domain: 0 for domain in _DOMAINS}
        self.misses = {domain: 0 for domain in _DOMAINS}
        self.evictions = 0

    def observe_window(self, state):
        for task in preparation_window(self.context, state):
            op = self.context.scenario.workload.operation_by_id[task]
            for device in self.context.devices_for(task):
                domain = "dram" if device == "cpu" else "vram"
                for tensor in op.inputs:
                    if tensor in self.context.facts.managed_reusable_tensors:
                        self.first_demand.setdefault((task, tensor, domain),
                                                    state.copy_at(tensor, domain) == COPY_READY)

    def touch(self, key):
        if key in self.stamps:
            self.clock += 1
            self.stamps[key] = (self.clock, self.stamps[key][1])

    def committed(self, action, state_before):
        if action.kind == ACTION_EVICT:
            self.stamps.pop((action.target_id, action.location), None)
            self.evictions += 1
        elif action.kind in {ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD} and action.target_id in self.context.facts.managed_reusable_tensors:
            if action.kind in {ACTION_COPY, ACTION_WRITE_SSD}:
                source = "dram" if action.kind == ACTION_WRITE_SSD else action.src
                self.touch((action.target_id, source))
            destination = _destination(action)
            if destination in _DOMAINS and self.sources[action.target_id] != destination:
                self.clock += 1
                self.stamps[(action.target_id, destination)] = (self.clock, self.clock)
        elif action.kind == ACTION_COMPUTE:
            domain = "dram" if action.device == "cpu" else "vram"
            op = self.context.scenario.workload.operation_by_id[action.target_id]
            for tensor in op.inputs:
                if tensor in self.context.facts.managed_reusable_tensors:
                    ready = self.first_demand[(action.target_id, tensor, domain)]
                    (self.hits if ready else self.misses)[domain] += 1
                    self.touch((tensor, domain))

    def plan(self, actions, state, termination_reason):
        entries = tuple(ScheduleStamp(tensor, domain, *stamp) for (tensor, domain), stamp in sorted(self.stamps.items())
            if state.copy_at(tensor, domain) == COPY_READY)
        return SchedulePlan(tuple(actions), ScheduleHistory(self.context.facts.binding_id, self.clock, entries), (),
            self.evictions, sum(self.hits.values()), sum(self.misses.values()), dict(self.hits), dict(self.misses),
            termination_reason, is_goal(self.context.scenario, state))


def replay_schedule_plan(context: ScheduleContext, actions: Sequence[Action],
                         history: ScheduleHistory | None = None, *, require_complete: bool = False) -> SchedulePlan:
    """Rebuild policy metadata from concrete restricted-space actions.

    Hits/misses mean managed reusable tensor-input residency on first entry into the preparation
    window, for the domain eventually selected by COMPUTE. They are not counts
    of SSD reads or PCIe copies; those are the engine's event/byte metrics.
    """
    if not isinstance(require_complete, bool):
        raise InvalidInput("require_complete must be boolean")
    state = initial_state(context.scenario)
    observer = _Observer(context, state, history)
    applied = []
    for action in actions:
        observer.observe_window(state)
        following = step_schedule(context, state, action)
        observer.committed(action, state)
        applied.append(action)
        state = following
    complete = is_goal(context.scenario, state)
    if require_complete and not complete:
        raise InvalidInput("schedule actions do not complete this invocation")
    return observer.plan(applied, state, "goal_reached" if complete else "unfinished")


def plan_schedule(
    context: ScheduleContext, *, history: ScheduleHistory | None = None,
    max_actions: int, future_accesses: Sequence[tuple[str, str]] = (),
) -> SchedulePlan:
    """Greedy event-driven compute/copy/prefetch baseline, with an action budget.

    Compute starts precede demand transfers, then ready prefetch transfers.
    Cache eviction is necessary-demand driven; prefetch-only eviction waits for
    an in-flight event instead of cycling future managed reusable tensors through a full cache.
    Repeated engine states are detected, and truncation remains unfinished.
    """
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise InvalidInput("max_actions must be a positive integer")
    policy = context.policy
    state = initial_state(context.scenario)
    observer = _Observer(context, state, history)
    for tensor, domain in future_accesses:
        if tensor not in context.facts.managed_reusable_tensors or domain not in _DOMAINS:
            raise InvalidInput("future accesses must name declared managed reusable tensors and explicit dram/vram domains")
    actions = []
    visited = set()
    reason = "no_legal_schedule_action"
    while not is_goal(context.scenario, state):
        if len(actions) >= max_actions:
            reason = "max_actions"
            break
        if state.key() in visited:
            reason = "repeated_schedule_state"
            break
        visited.add(state.key())
        observer.observe_window(state)
        ranks, pressure = _legal_details(context, state)
        if not ranks:
            break
        non_evictions = [action for action in ranks if action.kind != ACTION_EVICT and action.kind != ACTION_ADVANCE]
        action = min(non_evictions, key=lambda item: (ranks[item], str(item)), default=None)
        if action is None:
            next_index = next_operation_index(context, state)
            current = context.facts.operation_order[next_index] if next_index < len(context.facts.operation_order) else None
            protected = set()
            if current is not None:
                op = context.scenario.workload.operation_by_id[current]
                for device in context.devices_for(current):
                    domain = "dram" if device == "cpu" else "vram"
                    protected.update((tensor, domain) for tensor in op.inputs)
            victims = [item for item in ranks if item.kind == ACTION_EVICT
                       and (item.target_id, item.location) not in protected]
            if state.running:
                victims = [item for item in victims if pressure[item.location][0] < 2]
            def victim_rank(item):
                key = (item.target_id, item.location)
                recency = observer.stamps.get(key, (0, 0))
                if policy.eviction_kind == "offline_farthest":
                    future = []
                    for task in context.facts.operation_order[next_index:]:
                        op = context.scenario.workload.operation_by_id[task]
                        for device in context.devices_for(task):
                            domain = "dram" if device == "cpu" else "vram"
                            future.extend((tensor, domain) for tensor in op.inputs if tensor in context.facts.managed_reusable_tensors)
                    future.extend(future_accesses)
                    distance = next((i for i, demand in enumerate(future) if demand == key or (
                        key[1] == "dram" and demand == (key[0], "vram"))), len(future) + 1)
                    return (ranks[item], -distance, key)
                return (ranks[item], recency[0 if policy.eviction_kind == "lru" else 1], key)
            action = min(victims, key=victim_rank, default=None)
            if action is None and Action(ACTION_ADVANCE) in ranks:
                action = Action(ACTION_ADVANCE)
        if action is None:
            break
        following = step_schedule(context, state, action)
        observer.committed(action, state)
        actions.append(action)
        state = following
    if is_goal(context.scenario, state):
        reason = "goal_reached"
    return observer.plan(actions, state, reason)
