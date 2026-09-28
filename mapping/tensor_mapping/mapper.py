"""M0 mapper: uniform-cost (Dijkstra) search over the mapping action space.

Edge costs are exactly the simulated time each action consumes: COPY_H2D,
COMPUTE and EVICT are instantaneous commitments and cost 0, while ADVANCE costs
the delta it moves the clock. The priority queue is ordered by
``(cumulative_simulated_time, increasing_sequence)`` -- the sequence number
makes ties deterministic, so two runs on the same input pick the same one of
several equally optimal mappings (``DESIGN.md`` §7).

Why Dijkstra is correct here, and where it stops being correct
--------------------------------------------------------------
The state key deliberately excludes the absolute time ``t``: with fixed costs
and no external arrivals, ``t`` is simply the cost of the path that reached the
state, so two states agreeing on everything else have identical futures and the
cheaper one dominates. This is a property of *this* model, not a general one. If
M0 ever gains time-varying costs or external events, the key has to change with
it -- ``ACCEPTANCE.md`` §6 says so explicitly.

Termination
-----------
There is no zero-cost cycle to worry about. COPY_H2D and EVICT are both free,
but they cannot alternate: a copy leaves the tensor RESERVED_COPY, which EVICT
refuses, and turning a reservation into READY requires an ADVANCE with a
strictly positive duration. So every cycle costs time and ``best_cost`` is a
well-founded ordering.

Reported metrics come from the evaluator
----------------------------------------
When the search proves a goal, it does not compute the makespan, peak memory or
transferred bytes itself. It hands the reconstructed action list to
:func:`~tensor_mapping.engine.evaluate_mapping` and reports that result. The
consistency required by ``ACCEPTANCE.md`` §6 -- that a stored mapping replayed
through the model reproduces the search's numbers -- then holds by
construction, rather than by two code paths happening to agree.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Protocol

from llm_infer_model.tensor.engine import (
    Action,
    EvaluationResult,
    State,
    WallClock,
    evaluate_mapping,
    initial_state,
    is_goal,
    legal_actions,
    transition,
)
from llm_infer_model.tensor.spec import InitialCapacityExceeded, Scenario, scenario_fingerprint, workload_fingerprint

__all__ = [
    "STATUS_FEASIBLE",
    "STATUS_INFEASIBLE",
    "STATUS_OPTIMAL",
    "STATUS_UNKNOWN",
    "SearchRecorder",
    "SearchResult",
    "search",
]

STATUS_OPTIMAL = "optimal"
STATUS_FEASIBLE = "feasible"
STATUS_INFEASIBLE = "infeasible"
STATUS_UNKNOWN = "unknown"

TERMINATION_GOAL_POPPED = "goal_popped"
TERMINATION_EXHAUSTED = "search_space_exhausted"
TERMINATION_STATE_BUDGET = "state_budget_exhausted"
TERMINATION_TIME_BUDGET = "time_budget_exhausted"
TERMINATION_INITIAL_CAPACITY = "initial_capacity_exceeded"

StateKey = tuple[Any, ...]


class Clock(Protocol):
    """Minimal clock interface, so tests can drive the budget deterministically."""

    def elapsed_s(self) -> float: ...

    def expired(self) -> bool: ...


class SearchRecorder(Protocol):
    """Observes a search, and takes no part in it.

    ``M1_ACCEPTANCE.md`` §11: there must be no second Dijkstra. A viewer that
    wants to draw the search tree used to re-implement this module's loop outside
    the package and reconcile the two afterwards; the tree it drew was therefore
    only as trustworthy as that copy. These hooks are the alternative -- the one
    search announces what it does, and a display layer listens.

    **Passive** is a promise with teeth, and it is kept by construction rather
    than by discipline: every hook returns ``None``, nothing the recorder holds is
    read back, and no branch in :func:`search` consults one. So the same budget
    with and without a recorder must yield the same status, the same termination
    reason, the same expansion count and the same action tuple -- which is
    exactly the equality ``M1_ACCEPTANCE.md`` §11 asks for, and it is tested.

    A recorder that raises aborts the search. That is a caller's bug, not a
    supported way to prune: a hook is called *after* the decision it reports, so
    there is nothing left to influence.

    ``on_cut`` and ``on_edge`` are separate rather than one call with an
    ``incumbent`` field, so that the dominance test stays in this module. A
    recorder that had to recompute ``successor_cost >= previous`` would be
    re-implementing a rule with a silent failure mode: get it backwards and the
    tree shows an edge where the search pruned one, which looks like a bug in
    the search.
    """

    def on_start(self, state: State, cost: int) -> None:
        """The initial state, before anything is expanded."""

    def on_expand(self, state: State, cost: int, order: int) -> None:
        """A state was popped and is being expanded. ``order`` counts from 1."""

    def on_edge(
        self, parent: State, action: Action, successor: State, cost: int
    ) -> None:
        """A successor was accepted and pushed, possibly re-parenting an old one."""

    def on_cut(
        self, parent: State, action: Action, successor: State, cost: int, incumbent: int
    ) -> None:
        """A successor was dominated: a path no worse than ``cost`` already exists.

        ``incumbent`` is that path's cost. The successor is not pushed and not
        expanded, so it may never be reached any other way -- which is why a
        display layer has to be told about it at all.
        """

    def on_pending_goal(self, state: State, cost: int) -> None:
        """A goal is now the cheapest one seen but not yet popped.

        Called only when that record improves, so the last call is the goal a
        budget cut-off will report as ``feasible``. A recorder does not have to
        re-derive ``_budget_cut``'s tie-breaking, which is the point.
        """

    def on_goal_popped(self, state: State, cost: int) -> None:
        """A goal came out of the queue: the search is about to return ``optimal``."""


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one search, with the budget accounting that produced it."""

    status: str
    termination_reason: str
    # True only when the status is a proven claim: an optimal makespan, or a
    # proven infeasibility. A feasible result cut short by a budget is not a
    # proof, and neither is `unknown` (DESIGN.md §7).
    optimality_proven: bool
    makespan_ns: int | None
    peak_vram_bytes: int | None
    h2d_bytes: int | None
    actions: tuple[Action, ...]
    evaluation: EvaluationResult | None
    expanded_states: int
    visited_states: int
    wall_time_s: float
    scenario_fingerprint: str
    workload_fingerprint: str
    max_expanded_states: int
    wall_time_limit_s: float


def search(
    scenario: Scenario,
    *,
    max_expanded_states: int | None = None,
    wall_time_limit_s: float | None = None,
    clock: Clock | None = None,
    recorder: SearchRecorder | None = None,
) -> SearchResult:
    """Find a minimum-makespan mapping, or report why it could not.

    ``max_expanded_states`` and ``wall_time_limit_s`` default to the scenario's
    mapper block, where 0 means unlimited. The budget is checked *before*
    expanding a state, so a goal already popped is always confirmed; running out
    of budget is never reported as infeasible, because an unfinished search has
    proven nothing.

    ``recorder`` is optional and passive -- see :class:`SearchRecorder`. With one
    attached the search does exactly what it would have done without it; the
    hooks are called after each decision and their return values are unused.
    """
    state_limit = (
        scenario.mapper.max_expanded_states
        if max_expanded_states is None
        else max_expanded_states
    )
    time_limit = (
        scenario.mapper.wall_time_limit_s
        if wall_time_limit_s is None
        else wall_time_limit_s
    )
    if clock is None:
        clock = WallClock(time_limit)

    # ``perf_counter``, same clock as ``WallClock``: with ``monotonic`` (a 64 Hz
    # tick on Windows) a small search reported ``wall_time_s`` as exactly 0.0 --
    # the elapsed time was under one tick and the clock never moved. Reporting a
    # duration of zero for a search that did run is a wrong number, not a
    # rounded one.
    started = time.perf_counter()
    expanded = 0
    best_cost: dict[StateKey, int] = {}

    def assemble(
        *,
        status: str,
        termination_reason: str,
        optimality_proven: bool,
        actions: tuple[Action, ...] = (),
        evaluation: EvaluationResult | None = None,
    ) -> SearchResult:
        return SearchResult(
            status=status,
            termination_reason=termination_reason,
            optimality_proven=optimality_proven,
            makespan_ns=evaluation.makespan_ns if evaluation else None,
            peak_vram_bytes=evaluation.peak_vram_bytes if evaluation else None,
            h2d_bytes=evaluation.h2d_bytes if evaluation else None,
            actions=actions,
            evaluation=evaluation,
            expanded_states=expanded,
            visited_states=len(best_cost),
            wall_time_s=time.perf_counter() - started,
            scenario_fingerprint=scenario_fingerprint(scenario),
            workload_fingerprint=workload_fingerprint(scenario.workload),
            max_expanded_states=state_limit,
            wall_time_limit_s=time_limit,
        )

    try:
        start = initial_state(scenario)
    except InitialCapacityExceeded:
        # The scenario is well-formed and only its starting point is impossible;
        # this is a verdict about the scenario, not an input error.
        return assemble(
            status=STATUS_INFEASIBLE,
            termination_reason=TERMINATION_INITIAL_CAPACITY,
            optimality_proven=True,
        )

    start_key = start.key()
    best_cost[start_key] = 0
    parents: dict[StateKey, tuple[StateKey, Action]] = {}
    if recorder is not None:
        recorder.on_start(start, 0)

    # Entries are (cumulative time, sequence, state). The sequence number is
    # unique, so the heap never has to compare two State objects -- which it
    # could not do anyway, since State is not ordered.
    queue: list[tuple[int, int, State]] = [(0, 0, start)]
    sequence = 1
    # Cheapest goal seen but not yet popped. Only consulted if a budget runs out:
    # it is what lets the search answer `feasible` instead of `unknown`.
    pending_goal: tuple[int, StateKey] | None = None

    while queue:
        cost, _, state = heapq.heappop(queue)
        key = state.key()
        if cost > best_cost[key]:
            continue  # stale entry, superseded by a cheaper path

        if is_goal(scenario, state):
            # The first goal out of a uniform-cost queue has minimum cost, so
            # this is the only way `optimal` is ever concluded.
            actions = _reconstruct(parents, key, start_key)
            if recorder is not None:
                recorder.on_goal_popped(state, cost)
            return assemble(
                status=STATUS_OPTIMAL,
                termination_reason=TERMINATION_GOAL_POPPED,
                optimality_proven=True,
                actions=actions,
                evaluation=_replay(scenario, actions, cost),
            )

        # Checked before expanding, never after, so a goal that has already been
        # popped is confirmed rather than discarded.
        if state_limit and expanded >= state_limit:
            return _budget_cut(scenario, TERMINATION_STATE_BUDGET, pending_goal, start_key,
                               parents, assemble)
        if clock.expired():
            return _budget_cut(scenario, TERMINATION_TIME_BUDGET, pending_goal, start_key,
                               parents, assemble)

        expanded += 1
        if recorder is not None:
            recorder.on_expand(state, cost, expanded)
        for action in legal_actions(scenario, state):
            successor = transition(scenario, state, action)
            successor_key = successor.key()
            # Only ADVANCE costs anything; the other three leave t untouched.
            successor_cost = cost + (successor.t - state.t)
            previous = best_cost.get(successor_key)
            if previous is not None and successor_cost >= previous:
                # Strictly-less-only updates. Equal cost keeps the first path
                # found, which is what stops zero-cost actions from being
                # explored in every possible order (DESIGN.md §7).
                if recorder is not None:
                    recorder.on_cut(state, action, successor, successor_cost, previous)
                continue
            best_cost[successor_key] = successor_cost
            parents[successor_key] = (key, action)
            heapq.heappush(queue, (successor_cost, sequence, successor))
            sequence += 1
            if recorder is not None:
                recorder.on_edge(state, action, successor, successor_cost)
            if is_goal(scenario, successor):
                if pending_goal is None or successor_cost < pending_goal[0]:
                    pending_goal = (successor_cost, successor_key)
                    if recorder is not None:
                        recorder.on_pending_goal(successor, successor_cost)

    # Every reachable state was expanded and none was a goal. This exhaustiveness
    # is the only thing that ever justifies concluding infeasible.
    return assemble(
        status=STATUS_INFEASIBLE,
        termination_reason=TERMINATION_EXHAUSTED,
        optimality_proven=True,
    )


def _replay(scenario: Scenario, actions: tuple[Action, ...], claimed_ns: int) -> EvaluationResult:
    """Replay a found mapping through the evaluator and cross-check the makespan."""
    evaluation = evaluate_mapping(scenario, actions)
    if evaluation.status != "valid" or evaluation.makespan_ns != claimed_ns:
        # The search and the evaluator share one transition function, so this
        # cannot happen without a bug in one of them. Fail loudly rather than
        # publish a makespan the model disagrees with.
        raise RuntimeError(
            f"internal inconsistency: search reached a goal at {claimed_ns} ns but "
            f"replaying the mapping gave status={evaluation.status!r}, "
            f"makespan={evaluation.makespan_ns!r} ({evaluation.reason})"
        )
    return evaluation


def _budget_cut(
    scenario: Scenario,
    reason: str,
    pending_goal: tuple[int, StateKey] | None,
    start_key: StateKey,
    parents: dict[StateKey, tuple[StateKey, Action]],
    assemble: Any,
) -> SearchResult:
    """Report a budget cut-off, as ``feasible`` if a goal is already in hand.

    A goal discovered but not yet popped is a real, replayable mapping, so it is
    reported as ``feasible`` -- never as ``optimal``, because the search has not
    proven that nothing cheaper exists.
    """
    if pending_goal is None:
        return assemble(
            status=STATUS_UNKNOWN,
            termination_reason=reason,
            optimality_proven=False,
        )

    _, goal_key = pending_goal
    actions = _reconstruct(parents, goal_key, start_key)
    return assemble(
        status=STATUS_FEASIBLE,
        termination_reason=reason,
        optimality_proven=False,
        actions=actions,
        evaluation=evaluate_mapping(scenario, actions),
    )


def _reconstruct(
    parents: dict[StateKey, tuple[StateKey, Action]],
    goal_key: StateKey,
    start_key: StateKey,
) -> tuple[Action, ...]:
    """Walk the parent chain back to the start and reverse it."""
    actions: list[Action] = []
    key = goal_key
    while key != start_key:
        parent_key, action = parents[key]
        actions.append(action)
        key = parent_key
    actions.reverse()
    return tuple(actions)
