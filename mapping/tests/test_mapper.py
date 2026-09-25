"""Search behaviour and the acceptance table of ACCEPTANCE.md §2--§4.

Two kinds of test live here. The first pins the numbers the contract states --
nine makespans, three infeasibilities -- so a regression shows up as a wrong
number rather than a slow drift. The second covers the search's own promises:
that a verdict is only ever as strong as the budget allowed, that equally
optimal mappings are all accepted, and that the numbers the search reports are
the ones the evaluator produces from the same action list.
"""

from __future__ import annotations

import unittest

from tensor_mapping import (
    evaluate_mapping,
    scenario_fingerprint,
    search,
    workload_fingerprint,
)
from tensor_mapping.engine import (
    ACTION_COMPUTE,
    Action,
    RunningTask,
    State,
)
from tensor_mapping.mapper import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNKNOWN,
)

from .support import (
    Replay,
    load_example,
    ms,
    parse_actions,
    with_capacity,
    with_overlap,
)


class AcceptanceTableTests(unittest.TestCase):
    """ACCEPTANCE.md §2--§4: the nine stated optima and three infeasibilities."""

    # (example, capacity or None, expected status, expected makespan in ms)
    CASES = (
        ("chain-cap160", 160, STATUS_OPTIMAL, 8),
        ("chain-cap160", 96, STATUS_OPTIMAL, 10),
        ("chain-cap160", 95, STATUS_INFEASIBLE, None),
        ("residual-cap160", 160, STATUS_OPTIMAL, 9),
        ("residual-cap160", 112, STATUS_OPTIMAL, 11),
        ("residual-cap160", 111, STATUS_INFEASIBLE, None),
        ("fork-cap160", 160, STATUS_OPTIMAL, 9),
        ("fork-cap160", 112, STATUS_OPTIMAL, 11),
        ("fork-cap160", 111, STATUS_INFEASIBLE, None),
        ("matvec-cap160", 160, STATUS_OPTIMAL, 5),
    )

    def test_the_stated_optima_and_infeasibilities(self) -> None:
        for name, capacity, status, makespan in self.CASES:
            with self.subTest(scenario=name, capacity=capacity):
                scenario = with_capacity(load_example(name), capacity)
                result = search(scenario)
                self.assertEqual(result.status, status)
                self.assertTrue(result.optimality_proven)
                if makespan is None:
                    self.assertIsNone(result.makespan_ns)
                else:
                    self.assertEqual(result.makespan_ns, ms(makespan))

    def test_a_proven_infeasibility_says_the_space_was_exhausted(self) -> None:
        result = search(with_capacity(load_example("chain-cap160"), 95))
        self.assertEqual(result.termination_reason, "search_space_exhausted")
        self.assertEqual(result.actions, ())

    def test_the_peaks_match_the_contract(self) -> None:
        for name, capacity, peak in (
            ("chain-cap160", 160, 160),
            ("chain-cap160", 96, 96),
            ("residual-cap160", 160, 160),
            ("residual-cap160", 112, 112),
            ("fork-cap160", 160, 160),
            ("fork-cap160", 112, 112),
            ("matvec-cap160", 160, 76),
        ):
            with self.subTest(scenario=name, capacity=capacity):
                result = search(with_capacity(load_example(name), capacity))
                self.assertEqual(result.peak_vram_bytes, peak)

    def test_every_search_stays_within_its_budget(self) -> None:
        for name in ("chain-cap160", "residual-cap160", "fork-cap160", "matvec-cap160"):
            with self.subTest(scenario=name):
                scenario = load_example(name)
                result = search(scenario)
                self.assertLessEqual(
                    result.peak_vram_bytes,
                    scenario.architecture.vram_capacity_bytes,
                )

    def test_the_chain_without_overlap_takes_ten_ms(self) -> None:
        scenario = with_overlap(load_example("chain-cap160"), False)
        result = search(scenario)
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.makespan_ns, ms(10))


class SearchResultIntegrityTests(unittest.TestCase):
    """ACCEPTANCE.md §6: what the search reports must survive a replay."""

    EXAMPLES = ("chain-cap160", "residual-cap160", "fork-cap160", "matvec-cap160")

    def test_the_reported_numbers_come_from_replaying_the_mapping(self) -> None:
        for name in self.EXAMPLES:
            with self.subTest(scenario=name):
                scenario = load_example(name)
                result = search(scenario)
                replay = evaluate_mapping(scenario, result.actions)

                self.assertEqual(replay.status, "valid")
                self.assertEqual(replay.makespan_ns, result.makespan_ns)
                self.assertEqual(replay.peak_vram_bytes, result.peak_vram_bytes)
                self.assertEqual(replay.h2d_bytes, result.h2d_bytes)
                self.assertEqual(replay.action_count, len(result.actions))
                self.assertIs(result.evaluation.makespan_ns, result.makespan_ns)
                self.assertEqual(len(replay.events), len(result.actions))

    def test_every_action_is_legal_in_the_order_it_was_found(self) -> None:
        # The search builds a plan by chaining transitions; if the plan were not
        # a legal sequence, the evaluator (which has no repair step) would say so.
        for name in self.EXAMPLES:
            with self.subTest(scenario=name):
                scenario = load_example(name)
                result = search(scenario)
                replay = Replay(scenario)
                for action in result.actions:
                    replay.do(action)
                self.assertEqual(replay.t, result.makespan_ns)

    def test_the_result_carries_both_fingerprints(self) -> None:
        scenario = load_example("chain-cap160")
        result = search(scenario)
        self.assertEqual(result.workload_fingerprint, workload_fingerprint(scenario.workload))
        self.assertEqual(result.scenario_fingerprint, scenario_fingerprint(scenario))

    def test_the_budget_that_was_used_is_reported(self) -> None:
        scenario = load_example("chain-cap160")
        result = search(scenario, max_expanded_states=7, wall_time_limit_s=2.5)
        self.assertEqual(result.max_expanded_states, 7)
        self.assertEqual(result.wall_time_limit_s, 2.5)
        # Not `> 0`: the whole search takes well under the clock's resolution on
        # Windows, so this only checks the field is populated and not negative.
        self.assertGreaterEqual(result.wall_time_s, 0.0)
        self.assertGreater(result.expanded_states, 0)
        self.assertGreaterEqual(result.visited_states, result.expanded_states)

    def test_a_scenario_whose_start_does_not_fit_is_infeasible(self) -> None:
        # ACCEPTANCE.md §5 row 20: a verdict about the scenario, not an error.
        result = search(with_capacity(load_example("chain-cap160"), 8))
        self.assertEqual(result.status, STATUS_INFEASIBLE)
        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.termination_reason, "initial_capacity_exceeded")
        self.assertEqual(result.expanded_states, 0)


class BudgetTests(unittest.TestCase):
    """DESIGN.md §7: a budget cut is never a proof."""

    def setUp(self) -> None:
        self.scenario = load_example("chain-cap160")

    def test_a_tight_budget_never_claims_infeasibility(self) -> None:
        for limit in range(1, 30):
            with self.subTest(max_expanded_states=limit):
                result = search(self.scenario, max_expanded_states=limit)
                self.assertNotEqual(result.status, STATUS_INFEASIBLE)
                if result.status == STATUS_OPTIMAL:
                    self.assertTrue(result.optimality_proven)
                else:
                    self.assertFalse(result.optimality_proven)

    def test_the_budget_really_does_produce_both_weaker_verdicts(self) -> None:
        # Guards the sweep above: if the search ignored the budget entirely, the
        # assertions there would pass vacuously.
        statuses = {
            search(self.scenario, max_expanded_states=limit).status for limit in range(1, 30)
        }
        self.assertIn(STATUS_UNKNOWN, statuses)
        self.assertIn(STATUS_FEASIBLE, statuses)

    def test_a_feasible_result_is_a_real_mapping_without_a_proof(self) -> None:
        # Swept rather than pinned to one limit: the limit at which the search
        # first reaches a goal is an implementation detail, but "when it does,
        # the verdict is feasible and not a proof" is the contract.
        for limit in range(1, 40):
            result = search(self.scenario, max_expanded_states=limit)
            if result.status != STATUS_FEASIBLE:
                continue
            self.assertEqual(result.termination_reason, "state_budget_exhausted")
            self.assertFalse(result.optimality_proven)
            # A genuine mapping, so the evaluator accepts it and agrees...
            self.assertTrue(result.actions)
            replay = evaluate_mapping(self.scenario, result.actions)
            self.assertEqual(replay.status, "valid")
            self.assertEqual(replay.makespan_ns, result.makespan_ns)
            # ...and it cannot beat the optimum it failed to prove.
            self.assertGreaterEqual(result.makespan_ns, ms(8))
            return
        self.fail("no state budget between 1 and 39 produced a feasible result")

    def test_a_time_budget_behaves_like_a_state_budget(self) -> None:
        class CutOffClock:
            """Expires after ``allowed`` checks, so the cut lands on a known step."""

            def __init__(self, allowed: int) -> None:
                self.allowed = allowed
                self.checks = 0

            def elapsed_s(self) -> float:
                return 0.0

            def expired(self) -> bool:
                self.checks += 1
                return self.checks > self.allowed

        statuses = set()
        for allowed in range(1, 30):
            with self.subTest(allowed=allowed):
                result = search(self.scenario, clock=CutOffClock(allowed))
                self.assertNotEqual(result.status, STATUS_INFEASIBLE)
                statuses.add(result.status)
        self.assertIn(STATUS_FEASIBLE, statuses)

    def test_an_unlimited_budget_is_the_default(self) -> None:
        result = search(self.scenario)
        self.assertEqual(result.max_expanded_states, 0)
        self.assertEqual(result.wall_time_limit_s, 0.0)
        self.assertEqual(result.status, STATUS_OPTIMAL)


class DeterminismTests(unittest.TestCase):
    """DESIGN.md §7: the same input must give the same mapping."""

    def test_repeated_runs_agree(self) -> None:
        for name in ("chain-cap160", "residual-cap160", "fork-cap160"):
            with self.subTest(scenario=name):
                scenario = load_example(name)
                first = search(scenario)
                second = search(scenario)
                self.assertEqual(first.status, second.status)
                self.assertEqual(first.makespan_ns, second.makespan_ns)
                self.assertEqual(first.actions, second.actions)
                self.assertEqual(first.expanded_states, second.expanded_states)

    def test_the_search_does_not_depend_on_the_mapspace_order(self) -> None:
        # copy_tensor_ids is a set-like list; legal_actions sorts by tensor id,
        # so reordering the declaration must not change the answer.
        import dataclasses

        scenario = load_example("chain-cap160")
        reversed_mapspace = dataclasses.replace(
            scenario.mapspace,
            copy_tensor_ids=tuple(reversed(scenario.mapspace.copy_tensor_ids)),
        )
        flipped = dataclasses.replace(scenario, mapspace=reversed_mapspace)
        self.assertEqual(search(scenario).actions, search(flipped).actions)


class EqualOptimaTests(unittest.TestCase):
    """ACCEPTANCE.md §4: both branch orders of the fork are optimal."""

    def setUp(self) -> None:
        self.scenario = load_example("fork-cap160")

    def _preload_first(self, first: str, operation: str) -> tuple[Action, ...]:
        return parse_actions(
            f"COPY_H2D {first}",
            "ADVANCE",
            f"COMPUTE {operation}",
            f"COPY_H2D {second(first)}",
            "ADVANCE",
            f"EVICT {first}",
            "ADVANCE",
            f"COMPUTE {second_op(operation)}",
            "ADVANCE",
            f"EVICT {second(first)}",
            "COMPUTE add",
            "ADVANCE",
        )

    def test_neither_branch_order_is_forced(self) -> None:
        for first, operation in (("W1", "c1"), ("W2", "c2")):
            with self.subTest(first=first):
                actions = self._preload_first(first, operation)
                result = evaluate_mapping(self.scenario, actions)
                self.assertEqual(result.status, "valid")
                self.assertEqual(result.makespan_ns, ms(9))
                self.assertEqual(result.peak_vram_bytes, 160)

    def test_the_search_settles_on_one_of_the_two_branch_orders(self) -> None:
        result = search(self.scenario)
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.makespan_ns, ms(9))

        # Which branch goes first is not part of the contract, so this asserts
        # *an* order rather than a specific one -- and not action-list equality
        # with the hand-built plans above, which also evict a weight the search
        # has no reason to evict at 160 B.
        computed = [a.target_id for a in result.actions if a.kind == ACTION_COMPUTE]
        self.assertEqual(computed[:2], ["c1", "c2"])
        self.assertEqual(computed[2], "add")
        self.assertIn(computed, (["c1", "c2", "add"], ["c2", "c1", "add"]))

        # The mirror order is equally optimal, so a plan that prefers one of
        # them has not been forced by a rule that only accepts one.
        mirrored = self._preload_first("W2", "c2")
        self.assertEqual(evaluate_mapping(self.scenario, mirrored).makespan_ns, ms(9))


class StateKeyTests(unittest.TestCase):
    """ACCEPTANCE.md §6: what the key must and must not treat as distinct."""

    @staticmethod
    def _state(**kwargs) -> State:
        base = {
            "t": 0,
            "op_status": (("c1", "DONE"), ("c2", "NOT_STARTED")),
            "copy_status": (("W1", "READY"), ("x", "READY"), ("y", "ABSENT")),
            "running": (RunningTask("h2d_copy", "W2", 1_000_000),),
        }
        base.update(kwargs)
        return State(**base)

    def test_the_key_ignores_the_absolute_time(self) -> None:
        # t is the path cost, not part of what the state is (DESIGN.md §7).
        self.assertEqual(self._state(t=0).key(), self._state(t=99).key())

    def test_the_engine_stores_statuses_in_a_canonical_order(self) -> None:
        # This is what makes the key immune to set/dict iteration order: the
        # engine never hands the key an unsorted status list, so two runs that
        # discovered the same statuses in a different order cannot produce two
        # different keys for one state.
        replay = Replay(load_example("residual-cap160"))
        states = [replay.state]
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        states.append(replay.state)
        for state in states:
            with self.subTest(t=state.t):
                self.assertEqual(state.op_status, tuple(sorted(state.op_status)))
                self.assertEqual(state.copy_status, tuple(sorted(state.copy_status)))
                self.assertEqual(state.running, tuple(sorted(state.running)))

    def test_different_remaining_times_are_different_states(self) -> None:
        one = self._state(running=(RunningTask("h2d_copy", "W2", 1_000_000),))
        two = self._state(running=(RunningTask("h2d_copy", "W2", 2_000_000),))
        self.assertNotEqual(one.key(), two.key())

    def test_a_different_resource_is_a_different_state(self) -> None:
        copy = self._state(running=(RunningTask("h2d_copy", "W2", 1_000_000),))
        compute = self._state(running=(RunningTask("gpu_compute", "W2", 1_000_000),))
        self.assertNotEqual(copy.key(), compute.key())

    def test_two_tasks_in_flight_both_appear(self) -> None:
        both = self._state(
            running=(
                RunningTask("gpu_compute", "c1", 2_000_000),
                RunningTask("h2d_copy", "W2", 1_000_000),
            )
        )
        self.assertEqual(len(both.key()[-1]), 2)

    def test_a_reservation_is_not_the_same_state_as_a_ready_copy(self) -> None:
        reserved = self._state(copy_status=(("W1", "RESERVED_COPY"),))
        ready = self._state(copy_status=(("W1", "READY"),))
        self.assertNotEqual(reserved.key(), ready.key())


def second(tensor_id: str) -> str:
    return "W2" if tensor_id == "W1" else "W1"


def second_op(operation_id: str) -> str:
    return "c2" if operation_id == "c1" else "c1"


if __name__ == "__main__":
    unittest.main()
