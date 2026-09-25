"""Engine semantics: the 22 cases of ACCEPTANCE.md §5, plus the §6 properties.

Where a case is of the form "do these actions, then attempt this illegal one",
the test drives the transition function directly through :class:`Replay`. That
is deliberate: ``evaluate_mapping`` would report the *first* thing wrong with a
whole action list, which is the wrong tool for asserting that one specific
action is refused for one specific reason.
"""

from __future__ import annotations

import unittest

from tensor_mapping import (
    Action,
    evaluate_mapping,
    initial_state,
    is_goal,
    load_mapping,
    used_vram_bytes,
)
from tensor_mapping.engine import (
    ACTION_COPY_H2D,
    ACTION_EVICT,
    CODE_CAPACITY_EXCEEDED,
    CODE_ILLEGAL_ACTION,
    CODE_INPUT_NOT_READY,
    CODE_LIVE_VALUE_LOSS,
    CODE_RESOURCE_BUSY,
    CODE_TENSOR_IN_USE,
)

from .support import (
    EXAMPLES,
    Replay,
    example_document,
    load_example,
    ms,
    parse_actions,
    scenario_document,
    with_alignment,
    with_capacity,
    with_compute,
    with_eviction,
    with_h2d,
    with_overlap,
)


class CopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = load_example("chain-cap160")

    def test_compute_before_the_copy_finishes_is_input_not_ready(self) -> None:
        # ACCEPTANCE.md §5 row 1.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1")
        self.assertEqual(replay.refuse("COMPUTE c1").code, CODE_INPUT_NOT_READY)

    def test_a_copy_reserves_its_bytes_immediately(self) -> None:
        replay = Replay(self.scenario)
        self.assertEqual(replay.used_bytes(), 16)  # x is resident from the start
        replay.do("COPY_H2D W1")
        self.assertEqual(replay.used_bytes(), 16 + 64)
        self.assertEqual(replay.t, 0, "starting a copy must not move the clock")

    def test_duplicate_copy_of_a_ready_tensor_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 2.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE")
        before = replay.used_bytes()
        self.assertEqual(replay.refuse("COPY_H2D W1").code, CODE_ILLEGAL_ACTION)
        self.assertEqual(replay.used_bytes(), before, "a refused copy must not allocate")

    def test_duplicate_copy_of_an_in_flight_tensor_is_refused(self) -> None:
        # The reservation already owns its bytes; a second copy would double-count.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1")
        before = replay.used_bytes()
        self.assertEqual(replay.refuse("COPY_H2D W1").code, CODE_ILLEGAL_ACTION)
        self.assertEqual(replay.used_bytes(), before)

    def test_second_task_on_the_copy_resource_is_busy(self) -> None:
        # ACCEPTANCE.md §5 row 3.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1")
        self.assertEqual(replay.refuse("COPY_H2D W2").code, CODE_RESOURCE_BUSY)

    def test_starting_on_the_other_resource_keeps_the_clock_and_overlaps(self) -> None:
        # ACCEPTANCE.md §5 row 4.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE")
        replay.do("COMPUTE c1")
        clock_before = replay.t
        replay.do("COPY_H2D W2")
        self.assertEqual(replay.t, clock_before, "a start action never moves the clock")
        self.assertIsNotNone(replay.state.task_on("h2d_copy"))
        self.assertIsNotNone(replay.state.task_on("gpu_compute"))
        self.assertEqual(replay.used_bytes(), 160)

    def test_prefetching_for_nobody_is_refused(self) -> None:
        # "没有未启动消费者时禁止多余复制" (DESIGN.md §6): once c2 is the only
        # consumer left and it is done, reloading W2 would be pure waste, and the
        # state would be indistinguishable from one that skipped it.
        replay = Replay(self.scenario)
        replay.do(*_optimal_chain_actions())
        self.assertTrue(is_goal(self.scenario, replay.state))
        self.assertEqual(replay.refuse("COPY_H2D W2").code, CODE_ILLEGAL_ACTION)

    def test_a_tensor_outside_the_mapspace_cannot_be_copied(self) -> None:
        replay = Replay(self.scenario)
        self.assertEqual(replay.refuse("COPY_H2D x").code, CODE_ILLEGAL_ACTION)


class ComputeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = load_example("chain-cap160")

    def test_second_compute_on_the_same_resource_is_busy(self) -> None:
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        self.assertEqual(replay.refuse("COMPUTE c2").code, CODE_RESOURCE_BUSY)

    def test_recomputing_a_finished_operation_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 11: an operation runs exactly once.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.state.op("c1"), "DONE")
        self.assertEqual(replay.refuse("COMPUTE c1").code, CODE_ILLEGAL_ACTION)

    def test_an_in_flight_compute_holds_its_output_reservation(self) -> None:
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        # h is RESERVED_OUTPUT while c1 runs: the bytes are committed before the
        # value exists, which is why the reservation cannot be dropped either.
        self.assertEqual(replay.state.copy("h"), "RESERVED_OUTPUT")
        self.assertEqual(replay.state.copy("y"), "ABSENT")

    def test_releasing_a_running_target_reservation_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 5: the reservation cannot be dropped to make room.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        before = replay.used_bytes()
        self.assertEqual(replay.refuse("EVICT h").code, CODE_ILLEGAL_ACTION)
        self.assertEqual(replay.used_bytes(), before, "no capacity may be clawed back")

    def test_a_compute_cannot_free_an_input_it_is_still_reading(self) -> None:
        # ACCEPTANCE.md §5 row 5, and residual §3 line 57: while c1 runs, x is
        # locked.
        residual = load_example("residual-cap160")
        replay = Replay(residual)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        self.assertEqual(replay.refuse("EVICT x").code, CODE_TENSOR_IN_USE)
        self.assertEqual(replay.refuse("EVICT W1").code, CODE_TENSOR_IN_USE)

    def test_inputs_are_not_evicted_automatically(self) -> None:
        # DESIGN.md §6: completion releases the read lock and the workspace, and
        # nothing else. W1 is still resident after c1 finishes.
        replay = Replay(self.scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.state.copy("W1"), "READY")
        self.assertEqual(replay.state.copy("x"), "READY")
        self.assertEqual(replay.t, ms(5))


class EvictTests(unittest.TestCase):
    def test_evicting_a_tensor_with_no_copy_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 7: no no-op evictions, so no zero-cost cycles.
        replay = Replay(load_example("chain-cap160"))
        self.assertEqual(replay.refuse("EVICT W1").code, CODE_ILLEGAL_ACTION)
        self.assertEqual(replay.refuse("EVICT h").code, CODE_ILLEGAL_ACTION)

    def test_eviction_disabled_by_the_mapspace(self) -> None:
        replay = Replay(with_eviction(load_example("chain-cap160"), False))
        replay.do("COPY_H2D W1", "ADVANCE")
        self.assertEqual(replay.refuse("EVICT W1").code, CODE_ILLEGAL_ACTION)

    def test_evicting_a_live_intermediate_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 6: h has no DRAM copy and r still needs it.
        replay = Replay(load_example("residual-cap160"))
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.state.copy("h"), "READY")
        self.assertEqual(replay.refuse("EVICT h").code, CODE_LIVE_VALUE_LOSS)

    def test_evicting_an_input_that_a_later_consumer_needs_is_refused(self) -> None:
        # ACCEPTANCE.md §3: x must survive until the final add.
        replay = Replay(load_example("residual-cap160"))
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.refuse("EVICT x").code, CODE_LIVE_VALUE_LOSS)

    def test_a_shared_input_cannot_be_released_after_the_first_branch(self) -> None:
        # ACCEPTANCE.md §4: b still needs x, and add still needs a.
        replay = Replay(load_example("fork-cap160"))
        replay.do(
            "COPY_H2D W1",
            "ADVANCE",
            "COMPUTE c1",
            "ADVANCE",
        )
        self.assertEqual(replay.state.copy("a"), "READY")
        self.assertEqual(replay.refuse("EVICT x").code, CODE_LIVE_VALUE_LOSS)
        self.assertEqual(replay.refuse("EVICT a").code, CODE_LIVE_VALUE_LOSS)

    def test_evicting_a_requested_output_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 12.
        scenario = load_example("chain-cap160")
        replay = Replay(scenario)
        replay.do(*_optimal_chain_actions())
        self.assertTrue(is_goal(scenario, replay.state))
        self.assertEqual(replay.refuse("EVICT y").code, CODE_LIVE_VALUE_LOSS)

    def test_the_last_consumer_frees_the_sole_copy(self) -> None:
        # ACCEPTANCE.md §5 row 13.
        replay = Replay(load_example("chain-cap160"))
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.used_bytes(), 16 + 64 + 16)  # x, W1, h
        replay.do("EVICT x")
        self.assertEqual(replay.used_bytes(), 64 + 16)
        replay.do("EVICT W1")
        self.assertEqual(replay.used_bytes(), 16)
        self.assertEqual(replay.t, ms(5), "eviction is instantaneous")

    def test_a_weight_can_be_reloaded_after_eviction(self) -> None:
        # ACCEPTANCE.md §5 row 14: W1 has a DRAM source and a consumer (c1) that
        # has not started, so evicting it is allowed and reloading is normal.
        replay = Replay(load_example("chain-cap160"))
        replay.do("COPY_H2D W1", "ADVANCE")
        replay.do("EVICT W1")
        self.assertEqual(replay.used_bytes(), 16)
        replay.do("COPY_H2D W1")
        self.assertEqual(replay.used_bytes(), 16 + 64)
        replay.do("ADVANCE")
        self.assertEqual(replay.t, ms(6), "the reload costs its full duration again")
        replay.do("COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.state.op("c1"), "DONE")

    def test_a_reload_is_counted_again_in_the_transferred_bytes(self) -> None:
        scenario = load_example("chain-cap160")
        actions = parse_actions(
            "COPY_H2D W1",
            "ADVANCE",
            "EVICT W1",
            "COPY_H2D W1",
            "ADVANCE",
            "COMPUTE c1",
            "ADVANCE",
            "COPY_H2D W2",
            "ADVANCE",
            "EVICT W1",
            "EVICT x",
            "COMPUTE c2",
            "ADVANCE",
        )
        result = evaluate_mapping(scenario, actions)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.h2d_bytes, 3 * 64, "every copy moves bytes again")


class AdvanceTests(unittest.TestCase):
    def test_advance_with_nothing_running_is_refused(self) -> None:
        # ACCEPTANCE.md §5 row 8: no unbounded idling.
        replay = Replay(load_example("chain-cap160"))
        self.assertEqual(replay.refuse("ADVANCE").code, CODE_ILLEGAL_ACTION)

    def test_advance_moves_by_the_earliest_completion(self) -> None:
        replay = Replay(load_example("chain-cap160"))
        replay.do("COPY_H2D W1", "ADVANCE")
        self.assertEqual(replay.t, ms(3))
        replay.do("COMPUTE c1", "COPY_H2D W2")
        replay.do("ADVANCE")
        self.assertEqual(replay.t, ms(5), "c1 finishes two ms after W2 starts")
        # W2 still has a millisecond to go and keeps its reservation.
        self.assertEqual(replay.state.copy("W2"), "RESERVED_COPY")
        self.assertEqual(replay.state.copy("h"), "READY")

    def test_simultaneous_completions_finish_in_one_advance(self) -> None:
        # ACCEPTANCE.md §5 row 9. Making W2's copy 2 ms puts it on the same
        # instant as c1.
        scenario = with_h2d(load_example("chain-cap160"), "W2", 2)
        replay = Replay(scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1", "COPY_H2D W2")
        self.assertEqual(replay.t, ms(3))
        replay.do("ADVANCE")
        self.assertEqual(replay.t, ms(5))
        self.assertEqual(replay.state.copy("h"), "READY")
        self.assertEqual(replay.state.copy("W2"), "READY")
        self.assertEqual(replay.state.op("c1"), "DONE")
        self.assertIsNone(replay.state.task_on("h2d_copy"))
        self.assertIsNone(replay.state.task_on("gpu_compute"))
        # Nothing is released by completion: W1 and x are still resident, so the
        # ledger holds x + W1 + h + W2.
        self.assertEqual(replay.used_bytes(), 16 + 64 + 16 + 64)


class CapacityTests(unittest.TestCase):
    def test_capacity_is_checked_against_the_state_as_it_is(self) -> None:
        # ACCEPTANCE.md §5 row 15 and §2 line 41: at 96 B the W2 prefetch does
        # not fit beside W1, x and h, and the engine must say so rather than
        # helpfully deferring it.
        scenario = with_capacity(load_example("chain-cap160"), 96)
        replay = Replay(scenario)
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        self.assertEqual(replay.used_bytes(), 96)
        error = replay.refuse("COPY_H2D W2")
        self.assertEqual(error.code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(replay.t, ms(3), "a refused action must not advance the clock")

    def test_an_infeasible_capacity_is_rejected_at_the_first_compute(self) -> None:
        # ACCEPTANCE.md §2: c1 alone needs x(16) + W1(64) + h(16) = 96 B.
        for capacity in (95, 96):
            with self.subTest(capacity=capacity):
                scenario = with_capacity(load_example("chain-cap160"), capacity)
                replay = Replay(scenario)
                replay.do("COPY_H2D W1", "ADVANCE")
                self.assertEqual(replay.used_bytes(), 80)
                if capacity == 96:
                    # Exactly 96 fits for c1, but then W2 cannot prefetch.
                    self.assertIsNotNone(replay.do("COMPUTE c1"))
                else:
                    self.assertEqual(
                        replay.refuse("COMPUTE c1").code, CODE_CAPACITY_EXCEEDED
                    )

    def test_workspace_must_fit_alongside_the_output(self) -> None:
        # ACCEPTANCE.md §5 row 15: with a 16 B workspace, c1 needs
        # x(16) + W1(64) + h(16) + ws(16) = 112 B.
        for capacity in (95, 96):
            with self.subTest(capacity=capacity):
                scenario = with_compute(
                    with_capacity(load_example("chain-cap160"), capacity),
                    "c1",
                    duration_ms=2,
                    workspace_bytes=16,
                )
                replay = Replay(scenario)
                replay.do("COPY_H2D W1", "ADVANCE")
                self.assertEqual(replay.refuse("COMPUTE c1").code, CODE_CAPACITY_EXCEEDED)

    def test_workspace_is_released_when_the_compute_finishes(self) -> None:
        # 112 B is exactly x(16) + W1(64) + h(16) + ws(16).
        def build(capacity: int):
            return with_compute(
                with_capacity(load_example("chain-cap160"), capacity),
                "c1",
                duration_ms=2,
                workspace_bytes=16,
            )

        replay = Replay(build(111))
        replay.do("COPY_H2D W1", "ADVANCE")
        self.assertEqual(replay.used_bytes(), 80)
        self.assertEqual(replay.refuse("COMPUTE c1").code, CODE_CAPACITY_EXCEEDED)

        replay = Replay(build(112))
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        self.assertEqual(replay.used_bytes(), 112)
        replay.do("ADVANCE")
        self.assertEqual(replay.t, ms(5))
        # The workspace goes with the task; the output reservation stays.
        self.assertEqual(replay.used_bytes(), 16 + 64 + 16)

    def test_runtime_reserve_counts_towards_the_budget(self) -> None:
        from .support import with_runtime_reserve

        scenario = with_runtime_reserve(load_example("chain-cap160"), 96)
        replay = Replay(scenario)
        self.assertEqual(replay.used_bytes(), 96 + 16)
        self.assertEqual(replay.refuse("COPY_H2D W1").code, CODE_CAPACITY_EXCEEDED)


class AlignmentTests(unittest.TestCase):
    def test_allocation_is_padded_but_transferred_bytes_are_not(self) -> None:
        # ACCEPTANCE.md §5 row 16. matvec's W is 48 B; at 32 B alignment it
        # occupies 64 but still moves 48.
        scenario = with_alignment(load_example("matvec-cap160"), 32)
        self.assertEqual(scenario.size_alloc("W"), 64)
        self.assertEqual(scenario.workload.tensor_by_id["W"].storage_bytes, 48)

        replay = Replay(scenario)
        self.assertEqual(replay.used_bytes(), 32, "the 12 B input rounds up to 32")
        replay.do("COPY_H2D W", "ADVANCE", "COMPUTE c1", "ADVANCE")
        self.assertEqual(replay.used_bytes(), 32 + 64 + 32)
        self.assertEqual(replay.state.op("c1"), "DONE")

    def test_transferred_bytes_ignore_padding(self) -> None:
        scenario = with_alignment(load_example("matvec-cap160"), 32)
        result = evaluate_mapping(
            scenario, parse_actions("COPY_H2D W", "ADVANCE", "COMPUTE c1", "ADVANCE")
        )
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.h2d_bytes, 48, "not the 64 B aligned allocation")
        self.assertEqual(result.peak_vram_bytes, 32 + 64 + 32)

    def test_eviction_releases_the_aligned_allocation(self) -> None:
        # x is 16 B but occupies 32 at this alignment; W1 is 64 B either way, so
        # freeing W1 after c1 drops the ledger by exactly its aligned size.
        scenario = with_alignment(load_example("chain-cap160"), 32)
        replay = Replay(scenario)
        self.assertEqual(replay.used_bytes(), 32, "the 16 B input rounds up to 32")
        replay.do("COPY_H2D W1", "ADVANCE", "COMPUTE c1")
        self.assertEqual(replay.used_bytes(), 32 + 64 + 32)
        replay.do("ADVANCE", "EVICT W1")
        self.assertEqual(replay.used_bytes(), 32 + 32)


class RepeatedOperandTests(unittest.TestCase):
    """ACCEPTANCE.md §5 row 10: ``add(b, b)`` is one allocation and one lock."""

    _TIMELINE = (
        "COPY_H2D W1",
        "ADVANCE",
        "COMPUTE c1",
        "COPY_H2D W2",
        "ADVANCE",
        "EVICT W1",
        "COMPUTE r",
        "ADVANCE",
        "COMPUTE c2",
        "ADVANCE",
        "EVICT W2",
        "COMPUTE add",
        "ADVANCE",
    )

    @classmethod
    def _scenario(cls):
        workload = example_document("workload", "residual")
        next(op for op in workload["operations"] if op["id"] == "add")["inputs"] = ["b", "b"]
        return workload

    def test_a_repeated_operand_is_allocated_once(self) -> None:
        with scenario_document(self._scenario(), "residual-cap160") as scenario:
            replay = Replay(scenario)
            replay.do(*self._TIMELINE)
            # Resident while add runs: x, h, a, b, y. A ledger that walked the
            # operand list instead of the tensor set would say 96.
            self.assertEqual(replay.state.op("add"), "DONE")
            self.assertEqual(replay.used_bytes(), 16 * 5)

    def test_a_repeated_operand_does_not_leak_a_read_lock(self) -> None:
        with scenario_document(self._scenario(), "residual-cap160") as scenario:
            replay = Replay(scenario)
            replay.do(*self._TIMELINE)
            # b has no DRAM source, but add is done, so the value is dead and
            # the copy may go. A leaked second read lock would report
            # TENSOR_IN_USE here instead.
            replay.do("EVICT b")
            self.assertEqual(replay.used_bytes(), 16 * 4)


class MappingDocumentTests(unittest.TestCase):
    def test_the_worked_example_replays_to_the_optimum(self) -> None:
        scenario = load_example("chain-cap160")
        document = load_mapping(EXAMPLES / "chain.mapping.json", scenario)
        self.assertFalse(document.fingerprint_verified)
        self.assertEqual(document.scenario_id, "chain-cap160")

        result = evaluate_mapping(scenario, document.actions)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.makespan_ns, ms(8))
        self.assertEqual(result.peak_vram_bytes, 160)
        self.assertEqual(result.h2d_bytes, 128)

    def test_the_cap160_mapping_is_rejected_by_a_cap96_scenario(self) -> None:
        # ACCEPTANCE.md §2 line 41. The scenario id is relabelled so the replay
        # reaches the capacity rule rather than tripping the identity check
        # first -- identity is tested separately below.
        scenario = with_capacity(load_example("chain-cap160"), 96)
        document = load_mapping(EXAMPLES / "chain.mapping.json", load_example("chain-cap160"))

        result = evaluate_mapping(scenario, document.actions)
        self.assertEqual(result.status, "invalid_mapping")
        self.assertEqual(result.error_code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(result.error_action_index, 3, "the W2 copy, not the ADVANCE")
        self.assertEqual(result.error_t_ns, ms(3), "at t=3, not silently deferred to 5")

    def test_a_mapping_naming_another_scenario_is_rejected(self) -> None:
        from tensor_mapping import MappingIdentityError

        scenario = with_capacity(load_example("chain-cap160"), 96)
        with self.assertRaises(MappingIdentityError) as caught:
            load_mapping(EXAMPLES / "chain.mapping.json", scenario)
        self.assertIn("chain-cap96", str(caught.exception))

    def test_a_stale_fingerprint_is_rejected(self) -> None:
        # ACCEPTANCE.md §6 line 112.
        import json
        import tempfile
        from pathlib import Path

        from tensor_mapping import MappingIdentityError

        scenario = load_example("chain-cap160")
        payload = json.loads((EXAMPLES / "chain.mapping.json").read_text(encoding="utf-8"))
        payload["workload_fingerprint"] = "0" * 64
        payload["scenario_fingerprint"] = "1" * 64
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            path = Path(raw_dir) / "m.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(MappingIdentityError) as caught:
                load_mapping(path, scenario)
        self.assertIn("workload_fingerprint", str(caught.exception))

    def test_a_correct_fingerprint_is_verified(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from tensor_mapping import scenario_fingerprint, workload_fingerprint

        scenario = load_example("chain-cap160")
        payload = json.loads((EXAMPLES / "chain.mapping.json").read_text(encoding="utf-8"))
        payload["workload_fingerprint"] = workload_fingerprint(scenario.workload)
        payload["scenario_fingerprint"] = scenario_fingerprint(scenario)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            path = Path(raw_dir) / "m.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            document = load_mapping(path, scenario)
        self.assertTrue(document.fingerprint_verified)


class EvaluationOutcomeTests(unittest.TestCase):
    """ACCEPTANCE.md §5 rows 21 and 22, and the no-repair rule."""

    def setUp(self) -> None:
        self.scenario = load_example("chain-cap160")
        self.optimal = _optimal_chain_actions()

    def test_a_short_mapping_is_incomplete_not_repaired(self) -> None:
        actions = self.optimal[:-1]
        result = evaluate_mapping(self.scenario, actions)
        self.assertEqual(result.status, "incomplete_mapping")
        self.assertIsNone(result.makespan_ns)
        self.assertEqual(result.action_count, len(actions), "nothing was appended")

    def test_actions_after_the_goal_are_refused(self) -> None:
        result = evaluate_mapping(
            self.scenario, (*self.optimal, Action(ACTION_EVICT, "x"))
        )
        self.assertEqual(result.status, "invalid_mapping")
        self.assertEqual(result.error_action_index, len(self.optimal))

    def test_a_missing_advance_is_not_inserted(self) -> None:
        # The evaluator never advances the clock on the mapping's behalf, so an
        # action that needed an ADVANCE fails instead of quietly succeeding.
        result = evaluate_mapping(
            self.scenario, parse_actions("COPY_H2D W1", "COMPUTE c1")
        )
        self.assertEqual(result.status, "invalid_mapping")
        self.assertEqual(result.error_code, CODE_INPUT_NOT_READY)

    def test_an_empty_mapping_is_incomplete(self) -> None:
        result = evaluate_mapping(self.scenario, ())
        self.assertEqual(result.status, "incomplete_mapping")

    def test_the_event_log_records_both_ends_of_a_copy(self) -> None:
        result = evaluate_mapping(self.scenario, self.optimal)
        first = result.events[0]
        self.assertEqual(first.kind, ACTION_COPY_H2D)
        self.assertEqual((first.start_ns, first.end_ns), (0, ms(3)))
        self.assertEqual(first.bytes_transferred, 64)

    def test_state_snapshots_track_the_ledger(self) -> None:
        result = evaluate_mapping(self.scenario, self.optimal)
        self.assertEqual(result.states[0].t, 0)
        self.assertEqual(result.states[0].used_vram_bytes, 16)
        self.assertEqual(result.states[-1].t, ms(8))
        # x and W1 were evicted along the way, but nothing evicts W2, so the
        # final ledger is h + y + W2. Completion is not a release.
        self.assertEqual(result.states[-1].used_vram_bytes, 16 + 16 + 64)


class InitialCapacityTests(unittest.TestCase):
    def test_an_impossible_start_is_infeasible_before_any_replay(self) -> None:
        scenario = with_capacity(load_example("chain-cap160"), 8)
        result = evaluate_mapping(scenario, _optimal_chain_actions())
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.reason, "initial_capacity_exceeded")
        self.assertEqual(result.action_count, 0)


class SearchPropertyTests(unittest.TestCase):
    """ACCEPTANCE.md §5 line 102: properties that hold without a known optimum."""

    def test_more_capacity_never_worsens_the_optimum(self) -> None:
        from tensor_mapping import search

        for name in ("chain-cap160", "residual-cap160", "fork-cap160"):
            with self.subTest(graph=name):
                base = load_example(name)
                previous = None
                for capacity in range(16, 260, 8):
                    result = search(with_capacity(base, capacity))
                    if result.makespan_ns is None:
                        self.assertEqual(result.status, "infeasible")
                        continue
                    if previous is not None:
                        self.assertLessEqual(
                            result.makespan_ns,
                            previous,
                            f"{name} got worse when capacity rose to {capacity}",
                        )
                    previous = result.makespan_ns

    def test_serial_execution_is_never_better_than_overlap(self) -> None:
        from tensor_mapping import search

        for name in ("chain-cap160", "residual-cap160", "fork-cap160"):
            with self.subTest(graph=name):
                base = load_example(name)
                for capacity in (160, 96):
                    overlapping = search(with_capacity(base, capacity))
                    serial = search(with_overlap(with_capacity(base, capacity), False))
                    if overlapping.makespan_ns is None or serial.makespan_ns is None:
                        continue
                    self.assertGreaterEqual(serial.makespan_ns, overlapping.makespan_ns)

    def test_chain_without_overlap_takes_ten_ms(self) -> None:
        # ACCEPTANCE.md §5 line 102.
        from tensor_mapping import search

        scenario = with_overlap(load_example("chain-cap160"), False)
        self.assertEqual(search(scenario).makespan_ns, ms(10))


class GoalTests(unittest.TestCase):
    def test_the_initial_state_is_never_a_goal(self) -> None:
        # Every workload has at least one operation, so work is always needed.
        for name in ("chain-cap160", "residual-cap160", "fork-cap160", "matvec-cap160"):
            with self.subTest(scenario=name):
                scenario = load_example(name)
                self.assertFalse(is_goal(scenario, initial_state(scenario)))

    def test_a_running_task_prevents_the_goal(self) -> None:
        scenario = load_example("chain-cap160")
        replay = Replay(scenario)
        replay.do(*_optimal_chain_actions()[:-1])
        self.assertIsNotNone(replay.state.task_on("gpu_compute"))
        self.assertFalse(is_goal(scenario, replay.state))

    def test_an_evicted_output_prevents_the_goal(self) -> None:
        # Reachable only by hand: the evaluator would refuse the eviction. It
        # matters because `is_goal` must not be satisfied by an operation being
        # DONE when its result is no longer resident.
        scenario = load_example("chain-cap160")
        replay = Replay(scenario)
        replay.do(*_optimal_chain_actions())
        self.assertTrue(is_goal(scenario, replay.state))
        self.assertEqual(replay.state.copy("y"), "READY")


def _optimal_chain_actions() -> tuple[Action, ...]:
    """The worked 8 ms mapping from DESIGN.md §4.3."""
    return parse_actions(
        "COPY_H2D W1",
        "ADVANCE",
        "COMPUTE c1",
        "COPY_H2D W2",
        "ADVANCE",
        "EVICT W1",
        "EVICT x",
        "ADVANCE",
        "COMPUTE c2",
        "ADVANCE",
    )


if __name__ == "__main__":
    unittest.main()
