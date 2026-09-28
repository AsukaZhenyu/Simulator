"""M1 (``schema_version = "0.2"``) acceptance tests.

``M1_ACCEPTANCE.md`` §2--§9 is a table of makespans, byte counts and refusals
(§8 and §9 are the SSD spill and SSD-output cases the M1.3 revision added);
§10 is the semantic table and §11 the search and compatibility rules.
This file is that table, in order, one method per row, each naming its section.
Where a row is a capacity or a device restriction the scenario is mutated in
memory by the helpers in ``support`` rather than cloned into a near-identical
JSON file (``M1_ACCEPTANCE.md`` §10 asks for exactly that).

Every cost here is synthetic (``costs.source = "synthetic_fixed"``; the four
real GGML exports under ``examples/ggml/`` are the only exceptions in the
repository), so a makespan in this file is a statement about the *scheduler*,
never about any real hardware.

Three pieces of the contract carry most of the weight:

* the read lock is per ``(tensor, domain)`` -- a CPU compute reading DRAM does
  not stop the VRAM copy of the same tensor from being released;
* an eviction is judged by *reachability*: after the copy is gone, every
  unfinished consumer must still have one allowed device whose domain can be
  reached, and every undelivered output must still be able to reach the domain
  its goal names. One reachable device is enough; capacity, busy-ness and
  reservations are not part of that judgement;
* the goal is "every operation DONE, nothing in flight, and every requested
  output READY in the domain the goal names" -- and for an SSD goal, "READY"
  means the write has completed.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from tensor_mapping import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_M1,
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNKNOWN,
    InitialCapacityExceeded,
    InvalidInput,
    MappingIdentityError,
    Unsupported,
    evaluate_mapping,
    initial_state,
    is_goal,
    load_and_validate,
    load_mapping,
    scenario_fingerprint,
    search,
    workload_fingerprint,
)
from tensor_mapping.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY,
    ACTION_EVICT,
    ACTION_READ_SSD,
    ACTION_WRITE_SSD,
    CODE_CAPACITY_EXCEEDED,
    CODE_ILLEGAL_ACTION,
    CODE_INPUT_NOT_READY,
    CODE_LIVE_VALUE_LOSS,
    CODE_RESOURCE_BUSY,
    CODE_TENSOR_IN_USE,
    REASON_ACTIONS_REMAIN_AFTER_GOAL,
    REASON_GOAL_REACHED,
    REASON_INITIAL_CAPACITY_EXCEEDED,
    REASON_UNFINISHED,
    STATUS_INCOMPLETE_MAPPING,
    STATUS_INVALID_MAPPING,
    STATUS_VALID,
    used_bytes,
)
from tensor_mapping.spec import (
    COPY_ABSENT,
    COPY_READY,
    COPY_RESERVED_COPY,
    DOMAIN_OF_DEVICE,
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    OP_DONE,
    OP_NOT_STARTED,
    RESOURCE_CPU_COMPUTE,
    RESOURCE_GPU_COMPUTE,
    RESOURCE_PCIE_COPY,
    RESOURCE_SSD_IO,
    ROUTES,
    ROUTE_DRAM_TO_SSD,
    ROUTE_DRAM_TO_VRAM,
    ROUTE_SSD_TO_DRAM,
    ROUTE_VRAM_TO_DRAM,
)
from tests.support import (
    EXAMPLES,
    M1Replay,
    allow_only,
    drop_route,
    m1_scenario,
    ms,
    parse_action_m1,
    set_capacity,
    set_compute,
    set_transfer,
    transfer_only,
)


def drop_ssd_route(workload: dict, scenario: dict) -> None:
    """Remove the only route that can reach a tensor living on the SSD."""
    drop_route(scenario, ROUTE_SSD_TO_DRAM)


class M1Case(unittest.TestCase):
    """Search helpers shared by the table tests."""

    def solve(self, workload: str, scenario: str, mutate=None, **budget):
        with m1_scenario(workload, scenario, mutate) as loaded:
            return search(loaded, **budget)

    def makespan(self, workload: str, scenario: str, mutate=None) -> int:
        result = self.solve(workload, scenario, mutate)
        self.assertEqual(result.status, STATUS_OPTIMAL, result.termination_reason)
        self.assertIsNotNone(result.makespan_ns)
        return result.makespan_ns

    def assertMakespan(self, expected_ms: float, workload: str, scenario: str, mutate=None):
        self.assertEqual(self.makespan(workload, scenario, mutate), ms(expected_ms))

    def assertInfeasible(self, workload: str, scenario: str, mutate=None, reason=None):
        result = self.solve(workload, scenario, mutate)
        self.assertEqual(result.status, STATUS_INFEASIBLE)
        self.assertEqual(result.actions, ())
        if reason is not None:
            self.assertEqual(result.termination_reason, reason)
        return result


# ---------------------------------------------------------------------------
# §2  One operation, two devices
# ---------------------------------------------------------------------------


class ComputeDeviceChoice(M1Case):
    """M1_ACCEPTANCE.md §2: ``y = W * x`` costed on the CPU and on the GPU.

    The CPU needs no transfer at all -- both operands already live in DRAM,
    which is the domain the CPU computes in -- while the GPU path always pays
    for the weight, the input and the result.
    """

    def test_cpu_wins_when_the_cpu_is_cheaper(self):
        # c1 is 6 ms on the CPU against 2 ms on the GPU, but the GPU path also
        # moves 80 bytes in at 3 + 1 ms and 16 bytes out at 1 ms.
        result = self.solve("m1-matvec", "m1-a-cpu6")
        self.assertEqual(result.makespan_ns, ms(6))
        self.assertEqual(result.peak_vram_bytes, 0)
        self.assertEqual(result.h2d_bytes, 0)
        self.assertEqual([action.kind for action in result.actions],
                         [ACTION_COMPUTE, ACTION_ADVANCE])

    def test_the_gpu_wins_once_the_cpu_is_slower_than_the_round_trip(self):
        result = self.solve("m1-matvec", "m1-a-cpu6",
                            lambda w, s: set_compute(s, "c1", "cpu", 9))
        self.assertEqual(result.makespan_ns, ms(7))
        self.assertEqual(result.h2d_bytes, 80)
        self.assertEqual(result.peak_vram_bytes, 96)
        self.assertEqual(result.evaluation.d2h_bytes, 16)

    def test_a_cpu_only_scenario_never_enters_vram(self):
        result = self.solve("m1-matvec", "m1-a-cpu6",
                            lambda w, s: allow_only(s, "c1", "cpu"))
        self.assertEqual(result.makespan_ns, ms(6))
        self.assertEqual(result.peak_vram_bytes, 0)
        # The whole plan is one compute: nothing is copied at all.
        self.assertEqual([action.kind for action in result.actions],
                         [ACTION_COMPUTE, ACTION_ADVANCE])

    def test_a_gpu_only_scenario_pays_both_directions(self):
        result = self.solve("m1-matvec", "m1-a-cpu6",
                            lambda w, s: allow_only(s, "c1", "gpu"))
        self.assertEqual(result.makespan_ns, ms(7))
        self.assertEqual(result.h2d_bytes, 80)
        self.assertEqual(result.peak_vram_bytes, 96)


# ---------------------------------------------------------------------------
# §3  Delivering a chained result across devices
# ---------------------------------------------------------------------------


class HybridDelivery(M1Case):
    """M1_ACCEPTANCE.md §3: ``h = relu(x)``, ``y = h + h``.

    All four pairings are exercised because the delivery path -- and therefore
    the makespan -- differs in each: CPU to CPU stays in DRAM, CPU to GPU lets
    the GPU read ``h`` straight out of DRAM, GPU to GPU never describes ``h`` at
    all, and GPU to CPU has to save the intermediate before the CPU can read it.
    """

    def test_cpu_to_cpu(self):
        self.assertMakespan(9, "m1-chain", "m1-b-hybrid",
                            lambda w, s: (allow_only(s, "c1", "cpu"),
                                          allow_only(s, "c2", "cpu")))

    def test_gpu_to_gpu_keeps_the_delivery_in_vram(self):
        result = self.solve("m1-chain", "m1-b-hybrid",
                            lambda w, s: (allow_only(s, "c1", "gpu"),
                                          allow_only(s, "c2", "gpu")))
        self.assertEqual(result.makespan_ns, ms(10))
        self.assertEqual(result.h2d_bytes, 16)
        # x comes in, and y goes out because the goal names DRAM -- h is the one
        # value that never crosses, which is what makes this pairing cheap.
        self.assertEqual([(a.kind, a.target_id) for a in result.actions
                          if a.kind == ACTION_COPY],
                         [(ACTION_COPY, "x"), (ACTION_COPY, "y")])

    def test_cpu_to_gpu_is_the_cheapest_pairing(self):
        # h is produced in DRAM by the CPU and read there by the GPU: the save
        # of h and the second half of the round trip both disappear.
        result = self.solve("m1-chain", "m1-b-hybrid",
                            lambda w, s: (allow_only(s, "c1", "cpu"),
                                          allow_only(s, "c2", "gpu")))
        self.assertEqual(result.makespan_ns, ms(5))
        self.assertEqual([(a.kind, a.target_id, a.dst) for a in result.actions
                          if a.kind == ACTION_COPY],
                         [(ACTION_COPY, "h", LOC_VRAM), (ACTION_COPY, "y", LOC_DRAM)])

    def test_gpu_to_cpu_must_save_the_intermediate(self):
        result = self.solve("m1-chain", "m1-b-hybrid",
                            lambda w, s: (allow_only(s, "c1", "gpu"),
                                          allow_only(s, "c2", "cpu")))
        self.assertEqual(result.makespan_ns, ms(17))
        self.assertEqual(result.h2d_bytes, 16)
        self.assertEqual(result.evaluation.d2h_bytes, 16)

    def test_a_hand_plan_for_the_gpu_to_cpu_delivery(self):
        """The 17 ms above, timed action by action.

        §3 asks for a hand plan per pairing so the search is not the only witness
        of its own number. The delivery here is the expensive one: the GPU
        produces h in VRAM and the CPU can only read it once it is in DRAM.
        """
        def pin(workload, scenario):
            allow_only(scenario, "c1", "gpu")
            allow_only(scenario, "c2", "cpu")

        with m1_scenario("m1-chain", "m1-b-hybrid", pin) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COMPUTE c1 gpu", "ADVANCE")
            self.assertEqual(replay.t, ms(8))
            replay.do("COPY h vram dram")
            # The destination is what is reserved while the copy runs; the source
            # stays READY, so an eviction of VRAM h would still be possible here.
            self.assertEqual(dict(replay.state.dram_status)["h"], COPY_RESERVED_COPY)
            self.assertEqual(dict(replay.state.copy_status)["h"], COPY_READY)
            replay.do("ADVANCE")
            self.assertEqual(replay.t, ms(9))
            replay.do("COMPUTE c2 cpu", "ADVANCE")
            self.assertEqual(replay.t, ms(17))
            result = evaluate_mapping(loaded, replay.applied)
            self.assertEqual(result.status, STATUS_VALID)
            self.assertEqual(result.reason, REASON_GOAL_REACHED)
            self.assertEqual(result.makespan_ns, ms(17))
            self.assertEqual(result.h2d_bytes, 16)
            self.assertEqual(result.d2h_bytes, 16)
            self.assertEqual(result.peak_dram_bytes, 48)


# ---------------------------------------------------------------------------
# §4  Overlap switches
# ---------------------------------------------------------------------------


class OverlapSwitches(M1Case):
    """M1_ACCEPTANCE.md §4: one switch per axis of concurrency.

    A fork is the smallest graph with genuinely independent CPU and GPU work, so
    it is where ``allow_cpu_gpu_overlap`` becomes a number.
    """

    def test_independent_devices_run_together(self):
        with m1_scenario("m1-fork", "m1-c-fork") as loaded:
            replay = M1Replay(loaded)
            replay.do("COMPUTE ca cpu", "COMPUTE cb gpu")
            self.assertEqual(sorted(task.resource for task in replay.state.running),
                             [RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE])
            self.assertEqual(replay.t, 0)
        self.assertMakespan(6, "m1-fork", "m1-c-fork")

    def test_disabling_cpu_gpu_overlap_serialises_the_fork(self):
        self.assertMakespan(8, "m1-fork", "m1-c-fork",
                            lambda w, s: s["mapspace"].__setitem__("allow_cpu_gpu_overlap", False))

    def test_the_overlap_switch_reports_itself(self):
        with m1_scenario("m1-fork", "m1-c-fork",
                         lambda w, s: s["mapspace"].__setitem__("allow_cpu_gpu_overlap", False)) as loaded:
            replay = M1Replay(loaded)
            replay.do("COMPUTE ca cpu")
            error = replay.refuse("COMPUTE cb gpu")
            self.assertEqual(error.code, CODE_RESOURCE_BUSY)
            self.assertIn("CPU/GPU compute overlap is disabled", str(error))

    def test_disabling_copy_compute_overlap_serialises_the_bus(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6",
                         lambda w, s: s["mapspace"].__setitem__("allow_copy_compute_overlap", False)) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram")
            error = replay.refuse("COMPUTE c1 gpu")
            self.assertEqual(error.code, CODE_RESOURCE_BUSY)
            self.assertIn("copy/compute overlap is disabled", str(error))

    def test_two_outputs_are_delivered_in_opposite_directions(self):
        # u starts in DRAM and is produced into VRAM; v is produced in VRAM and
        # has to be delivered to DRAM. One goal, two directions.
        result = self.solve("m1-two-outputs", "m1-c-two-outputs")
        self.assertEqual(result.makespan_ns, ms(6))
        self.assertEqual(result.evaluation.d2h_bytes, 16)
        self.assertGreater(result.h2d_bytes, 0)


# ---------------------------------------------------------------------------
# §5  DRAM capacity
# ---------------------------------------------------------------------------


class DramCapacity(M1Case):
    """M1_ACCEPTANCE.md §5: two weights that cannot both be resident.

    ``m1-two-weights-dram`` starts with W1 (64 B) in DRAM and W2 (64 B) on the
    SSD, plus a 16-byte input. Holding a weight, the input and the intermediate
    needs 96 bytes, so a budget below that forces W1 out and W2 back in.
    """

    def test_a_generous_budget_never_re_reads(self):
        result = self.solve("m1-two-weights-dram", "m1-d-dram96",
                            lambda w, s: set_capacity(s, dram=160))
        self.assertEqual(result.makespan_ns, ms(5))
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)

    def test_a_tight_budget_forces_an_eviction_and_a_re_read(self):
        result = self.solve("m1-two-weights-dram", "m1-d-dram96",
                            lambda w, s: set_capacity(s, dram=96))
        self.assertEqual(result.makespan_ns, ms(7))
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)
        kinds = [(a.kind, a.target_id, a.location) for a in result.actions]
        self.assertIn((ACTION_EVICT, "W1", LOC_DRAM), kinds)
        self.assertIn((ACTION_READ_SSD, "W2", None), kinds)

    def test_one_byte_short_is_infeasible(self):
        self.assertInfeasible("m1-two-weights-dram", "m1-d-dram96",
                              lambda w, s: set_capacity(s, dram=95))

    def test_losing_the_ssd_route_is_infeasible(self):
        self.assertInfeasible("m1-two-weights-dram", "m1-d-dram96", drop_ssd_route)

    def test_the_re_read_stops_overlapping_when_the_bus_is_exclusive(self):
        self.assertMakespan(
            7, "m1-two-weights-dram", "m1-d-dram96",
            lambda w, s: (set_capacity(s, dram=160),
                          s["mapspace"].__setitem__("allow_copy_compute_overlap", False)),
        )

    def test_the_dram_ledger_charges_alignment_and_reserve(self):
        def configure(workload, scenario):
            scenario["architecture"]["dram_alignment_bytes"] = 32
            scenario["architecture"]["dram_runtime_reserved_bytes"] = 32
            set_capacity(scenario, dram=192)

        with m1_scenario("m1-two-weights-dram", "m1-d-dram96", configure) as loaded:
            state = initial_state(loaded)
            # reserve 32 + W1 64 + x rounded up from 16 to 32
            self.assertEqual(used_bytes(loaded, state, LOC_DRAM), 128)

    def test_alignment_is_what_pushes_a_budget_over_the_edge(self):
        """The same 96-byte budget, with and without a 32-byte alignment.

        Unaligned, 96 bytes holds x, then h beside W1, then W2 beside h: the 7 ms
        plan fits. Rounding the 16-byte tensors up to 32 makes each of those
        steps 16 bytes wider than the budget allows.
        """
        def aligned(workload, scenario):
            scenario["architecture"]["dram_alignment_bytes"] = 32

        self.assertMakespan(7, "m1-two-weights-dram", "m1-d-dram96",
                            lambda w, s: set_capacity(s, dram=96))
        self.assertInfeasible("m1-two-weights-dram", "m1-d-dram96",
                              lambda w, s: (aligned(w, s), set_capacity(s, dram=96)))
        # At 95 the initial layout alone is over, and that is reported up front
        # rather than as a search that found nothing.
        self.assertInfeasible("m1-two-weights-dram", "m1-d-dram96",
                              lambda w, s: (aligned(w, s), set_capacity(s, dram=95)),
                              reason=REASON_INITIAL_CAPACITY_EXCEEDED)


# ---------------------------------------------------------------------------
# §6  A two-stage weight pipeline
# ---------------------------------------------------------------------------


class TwoStageWeightPipeline(M1Case):
    """M1_ACCEPTANCE.md §6: both weights live on the SSD.

    Nothing is resident at the start, so every byte of weight traffic is
    scheduled: read to DRAM, copy to VRAM, compute. The DRAM budget decides
    whether the second read can overlap a copy or has to wait for one.
    """

    def test_both_reads_fit(self):
        self.assertMakespan(13, "m1-two-weights-ssd", "m1-e-dram128",
                            lambda w, s: set_capacity(s, dram=128))

    def test_one_weight_at_a_time(self):
        self.assertMakespan(16, "m1-two-weights-ssd", "m1-e-dram128",
                            lambda w, s: set_capacity(s, dram=64))

    def test_below_one_weight_is_infeasible(self):
        self.assertInfeasible("m1-two-weights-ssd", "m1-e-dram128",
                              lambda w, s: set_capacity(s, dram=63))

    def test_a_tight_vram_budget_costs_one_millisecond(self):
        self.assertMakespan(14, "m1-two-weights-ssd", "m1-e-dram128",
                            lambda w, s: set_capacity(s, dram=128, vram=96))

    def test_the_ssd_read_has_a_slot_of_its_own(self):
        # A read and a copy are different resources, so the mapping can have both
        # in flight; two reads, and two copies, are each one resource (the second
        # read is covered in Refusals, where two SSD tensors exist).
        with m1_scenario("m1-matvec-ssd", "m1-f-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_READ_SSD} W", f"{ACTION_COPY} x dram vram")
            self.assertEqual(sorted(task.resource for task in replay.state.running),
                             [RESOURCE_PCIE_COPY, RESOURCE_SSD_IO])
            self.assertEqual(replay.t, 0)


# ---------------------------------------------------------------------------
# §7  Capacity changes the placement of the whole plan
# ---------------------------------------------------------------------------


class CapacityDrivenPlacement(M1Case):
    """M1_ACCEPTANCE.md §7: one graph, two schedules.

    At 96 bytes of DRAM the weight is read once and kept while the input stays
    put; at 80 the input has to leave first and come back afterwards, so the
    plan mirrors itself.
    """

    def test_the_roomy_budget_reads_once(self):
        result = self.solve("m1-matvec-ssd", "m1-f-dram96")
        self.assertEqual(result.makespan_ns, ms(6))
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)
        self.assertEqual([(a.kind, a.target_id) for a in result.actions
                          if a.kind != ACTION_ADVANCE],
                         [(ACTION_READ_SSD, "W"), (ACTION_COMPUTE, "c1")])

    def test_the_tight_budget_mirrors_the_plan(self):
        result = self.solve("m1-matvec-ssd", "m1-f-dram96",
                            lambda w, s: set_capacity(s, dram=80))
        self.assertEqual(result.makespan_ns, ms(10))
        moves = [(a.kind, a.target_id, a.src, a.dst, a.location) for a in result.actions]
        self.assertIn((ACTION_COPY, "x", LOC_DRAM, LOC_VRAM, None), moves)
        self.assertIn((ACTION_EVICT, "x", None, None, LOC_DRAM), moves)
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)

    def test_a_vram_budget_below_the_working_set_is_infeasible(self):
        self.assertInfeasible("m1-matvec-ssd", "m1-f-dram96",
                              lambda w, s: set_capacity(s, dram=80, vram=95))


# ---------------------------------------------------------------------------
# §8  Spilling a long-lived intermediate to the SSD and reading it back
# ---------------------------------------------------------------------------


def drop_write_route(workload: dict, scenario: dict) -> None:
    """Remove the only route that can put a snapshot on the SSD."""
    drop_route(scenario, ROUTE_DRAM_TO_SSD)


class SpillAndRecover(M1Case):
    """M1_ACCEPTANCE.md §8: h is live across an operation that fills DRAM alone.

    ``m1-spill`` has h produced by c1 and consumed by c3, with c2 needing W (64),
    z (16) and its own output (16) at the same moment. At 112 bytes h can simply
    stay resident; at 96 the only way through is to write h to the SSD, drop the
    DRAM copy, and read it back afterwards. The relu intermediate is a stand-in
    for a long-lived activation, not a claim about real KV state.
    """

    def test_a_roomy_budget_needs_no_write(self):
        result = self.solve("m1-spill", "m1-g-spill-dram96",
                            lambda w, s: set_capacity(s, dram=112))
        self.assertEqual(result.makespan_ns, ms(6))
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)
        self.assertEqual(result.evaluation.ssd_write_bytes, 0)

    def test_the_tight_budget_spills_and_reads_back(self):
        result = self.solve("m1-spill", "m1-g-spill-dram96")
        self.assertEqual(result.makespan_ns, ms(10))
        self.assertEqual(result.evaluation.ssd_read_bytes, 80)   # W 64 + h 16
        self.assertEqual(result.evaluation.ssd_write_bytes, 16)  # h

    def test_one_byte_short_is_infeasible(self):
        # c2 needs W 64 + z 16 + its own 16-byte output, so 95 can never hold the
        # middle of this graph together.
        self.assertInfeasible("m1-spill", "m1-g-spill-dram96",
                              lambda w, s: set_capacity(s, dram=95))

    def test_disabling_the_write_route_is_infeasible(self):
        self.assertInfeasible("m1-spill", "m1-g-spill-dram96", drop_write_route)

    def test_disabling_the_read_back_route_is_infeasible(self):
        # A backup that cannot be read is not a recovery: W may still be read, so
        # the route survives, but h cannot come back and the case dies with it.
        self.assertInfeasible(
            "m1-spill", "m1-g-spill-dram96",
            lambda w, s: transfer_only(s, ROUTE_SSD_TO_DRAM, "W"),
        )

    def test_the_hand_plan_replays_to_the_stated_ledger(self):
        """The 10 ms plan of M1_ACCEPTANCE.md §8, action by action."""
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE)          # 0-1
            self.assertEqual(replay.t, ms(1))
            replay.do(f"{ACTION_EVICT} x {LOC_DRAM}")                      # h must fit
            replay.do(f"{ACTION_WRITE_SSD} h", ACTION_ADVANCE)             # 1-3
            self.assertEqual(replay.t, ms(3))
            replay.do(f"{ACTION_EVICT} h {LOC_DRAM}")                      # only after the write
            replay.do(f"{ACTION_READ_SSD} W", ACTION_ADVANCE)              # 3-6
            replay.do(f"{ACTION_COMPUTE} c2 cpu", ACTION_ADVANCE)          # 6-8
            self.assertEqual(replay.used_bytes(LOC_DRAM), 96)
            replay.do(f"{ACTION_EVICT} W {LOC_DRAM}", f"{ACTION_EVICT} z {LOC_DRAM}")
            replay.do(f"{ACTION_READ_SSD} h", ACTION_ADVANCE)              # 8-9
            replay.do(f"{ACTION_COMPUTE} c3 cpu", ACTION_ADVANCE)          # 9-10
            self.assertEqual(replay.t, ms(10))
            self.assertTrue(is_goal(loaded, replay.state))

    def test_evicting_the_source_while_the_write_runs_is_refused(self):
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE, f"{ACTION_EVICT} x {LOC_DRAM}")
            replay.do(f"{ACTION_WRITE_SSD} h")
            error = replay.refuse(f"{ACTION_EVICT} h {LOC_DRAM}")
            self.assertEqual(error.code, CODE_TENSOR_IN_USE)

    def test_an_unfinished_write_is_not_a_backup(self):
        # While the write is in flight the SSD copy is RESERVED_COPY, so (a) it
        # cannot be read and (b) the DRAM copy is still the only read value.
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE, f"{ACTION_EVICT} x {LOC_DRAM}")
            replay.do(f"{ACTION_WRITE_SSD} h")
            error = replay.refuse(f"{ACTION_READ_SSD} h")
            self.assertEqual(error.code, CODE_INPUT_NOT_READY)
            self.assertIn("RESERVED_COPY", str(error))

    def test_a_finished_write_reads_back(self):
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE, f"{ACTION_EVICT} x {LOC_DRAM}")
            replay.do(f"{ACTION_WRITE_SSD} h", ACTION_ADVANCE)
            self.assertEqual(replay.state.copy_at("h", LOC_SSD), COPY_READY)
            replay.do(f"{ACTION_EVICT} h {LOC_DRAM}", f"{ACTION_READ_SSD} h")
            self.assertEqual(replay.state.copy_at("h", LOC_DRAM), COPY_RESERVED_COPY)

    def test_the_write_and_a_read_share_one_ssd_slot(self):
        """M1_DESIGN.md §4: one ``ssd_io`` slot, so the two directions exclude.

        Both orders are checked with their other preconditions satisfied: h has
        been computed and released from DRAM's second buffer, and W's SSD copy is
        the initial one, so neither refusal can come from a missing source.
        """
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            reading = M1Replay(loaded)
            reading.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                       f"{ACTION_EVICT} x {LOC_DRAM}", f"{ACTION_READ_SSD} W")
            writing = reading.refuse(f"{ACTION_WRITE_SSD} h")
            self.assertEqual(writing.code, CODE_RESOURCE_BUSY)
            self.assertIn(RESOURCE_SSD_IO, str(writing))

        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            writing = M1Replay(loaded)
            writing.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                       f"{ACTION_EVICT} x {LOC_DRAM}", f"{ACTION_WRITE_SSD} h")
            reading = writing.refuse(f"{ACTION_READ_SSD} W")
            self.assertEqual(reading.code, CODE_RESOURCE_BUSY)
            self.assertIn(RESOURCE_SSD_IO, str(reading))


# ---------------------------------------------------------------------------
# §9  An SSD output is delivered only once the write finishes
# ---------------------------------------------------------------------------


class SsdOutputDelivery(M1Case):
    """M1_ACCEPTANCE.md §9: the goal is a location, and one of them is the SSD.

    ``y = relu(x)`` with the goal on the SSD. Finishing the compute is not the
    goal, and neither is finishing the copy: the run ends only when the write has
    completed. The CPU row pays 1 ms of math plus 2 ms of writing; the GPU row
    also has to come back through DRAM first, because there is no SSD-to-VRAM
    path to write along.
    """

    def test_the_cpu_row(self):
        result = self.solve("m1-relu", "m1-h-ssd-cpu")
        self.assertEqual(result.makespan_ns, ms(3))
        self.assertEqual(result.evaluation.ssd_write_bytes, 16)
        self.assertEqual(result.evaluation.ssd_read_bytes, 0)
        self.assertEqual(result.evaluation.peak_dram_bytes, 32)  # x + y
        self.assertEqual(result.evaluation.peak_vram_bytes, 0)

    def test_the_gpu_row(self):
        result = self.solve("m1-relu-vram", "m1-h-ssd-gpu")
        self.assertEqual(result.makespan_ns, ms(4))
        self.assertEqual(result.evaluation.ssd_write_bytes, 16)
        self.assertEqual(result.evaluation.d2h_bytes, 16)
        self.assertEqual(result.evaluation.peak_dram_bytes, 16)  # the D2H target
        self.assertEqual(result.evaluation.peak_vram_bytes, 32)  # x + y

    def test_the_gpu_row_needs_a_real_dram_buffer(self):
        # 15 bytes cannot hold the 16-byte D2H target; the write may not borrow
        # the buffer it is about to read from.
        self.assertInfeasible("m1-relu-vram", "m1-h-ssd-gpu",
                              lambda w, s: set_capacity(s, dram=15))

    def test_without_the_write_route_the_case_is_infeasible(self):
        self.assertInfeasible("m1-relu", "m1-h-ssd-cpu", drop_write_route)

    def test_a_running_write_is_not_delivery(self):
        with m1_scenario("m1-relu", "m1-h-ssd-cpu") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE)
            self.assertFalse(is_goal(loaded, replay.state))   # computed, not delivered
            replay.do(f"{ACTION_WRITE_SSD} y")
            self.assertFalse(is_goal(loaded, replay.state))   # RESERVED_COPY is not READY
            replay.do(ACTION_ADVANCE)
            self.assertTrue(is_goal(loaded, replay.state))

    def test_the_write_locks_its_dram_source(self):
        with m1_scenario("m1-relu", "m1-h-ssd-cpu") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE, f"{ACTION_WRITE_SSD} y")
            error = replay.refuse(f"{ACTION_EVICT} y {LOC_DRAM}")
            self.assertEqual(error.code, CODE_TENSOR_IN_USE)

    def test_the_delivered_copy_stays_in_memory_too(self):
        # M1_ACCEPTANCE.md §9: the DRAM source is not released for free once the
        # write is done. Nothing requires an extra action to reach the goal, and
        # nothing performs one either.
        result = self.solve("m1-relu", "m1-h-ssd-cpu")
        self.assertEqual([a.kind for a in result.actions if a.kind != ACTION_ADVANCE],
                         [ACTION_COMPUTE, ACTION_WRITE_SSD])

    def test_writing_the_output_twice_is_refused(self):
        with m1_scenario("m1-relu", "m1-h-ssd-cpu") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                      f"{ACTION_WRITE_SSD} y", ACTION_ADVANCE)
            error = replay.refuse(f"{ACTION_WRITE_SSD} y")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("never copies onto", str(error))


# ---------------------------------------------------------------------------
# §10  What is refused, and with which code
# ---------------------------------------------------------------------------


class Refusals(M1Case):
    """M1_ACCEPTANCE.md §10: the rows that must not be silently accepted."""

    def test_an_unknown_tensor_is_illegal(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            error = M1Replay(loaded).refuse(f"{ACTION_COPY} nope dram vram")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)

    def test_an_unknown_operation_is_illegal(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            error = M1Replay(loaded).refuse(f"{ACTION_COMPUTE} nope gpu")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)

    def test_a_copy_onto_a_reserved_destination_is_illegal(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram")
            error = replay.refuse("COPY x dram vram")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("never copies onto", str(error))

    def test_a_tensor_the_route_does_not_allow_is_illegal(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            error = M1Replay(loaded).refuse("COPY W vram dram")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("is not allowed on route", str(error))

    def test_reading_a_tensor_with_no_ssd_source_is_illegal(self):
        # Nothing in this scenario is on the SSD, so the route allows nothing.
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            error = M1Replay(loaded).refuse(f"{ACTION_READ_SSD} W")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn(ROUTE_SSD_TO_DRAM, str(error))

    def test_a_copy_whose_source_is_not_ready_is_illegal(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            error = M1Replay(loaded).refuse("COPY y vram dram")
            self.assertEqual(error.code, CODE_INPUT_NOT_READY)
            self.assertIn("ABSENT in vram", str(error))

    def test_a_compute_whose_input_is_elsewhere_is_refused(self):
        # The mapping decides when an input moves; a compute never does.
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            error = M1Replay(loaded).refuse("COMPUTE c1 gpu")
            self.assertEqual(error.code, CODE_INPUT_NOT_READY)
            self.assertIn("never moves an input", str(error))

    def test_a_busy_resource_refuses_the_second_task(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_READ_SSD} W1")
            error = replay.refuse(f"{ACTION_READ_SSD} W2")
            self.assertEqual(error.code, CODE_RESOURCE_BUSY)
            self.assertIn(RESOURCE_SSD_IO, str(error))

    def test_a_busy_device_refuses_a_second_compute(self):
        with m1_scenario("m1-fork", "m1-c-fork") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} ca cpu")
            error = replay.refuse(f"{ACTION_COMPUTE} ca cpu")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("already RUNNING", str(error))

    def test_capacity_refuses_the_task_that_would_not_fit(self):
        def configure(workload, scenario):
            scenario["architecture"]["vram_alignment_bytes"] = 32
            scenario["architecture"]["vram_runtime_reserved_bytes"] = 32
            set_capacity(scenario, vram=64)

        with m1_scenario("m1-matvec", "m1-a-cpu6", configure) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE")
            self.assertEqual(replay.used_bytes(), 64)
            error = replay.refuse("COPY W dram vram")
            self.assertEqual(error.code, CODE_CAPACITY_EXCEEDED)
            self.assertIn("over the 64-byte budget", str(error))

    def test_eviction_can_be_switched_off(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6",
                         lambda w, s: s["mapspace"].__setitem__("allow_eviction", False)) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COPY W dram vram", "ADVANCE",
                      "COMPUTE c1 gpu", "ADVANCE")
            error = replay.refuse("EVICT x vram")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("eviction", str(error))

    def test_evicting_a_tensor_an_in_flight_task_reads_is_illegal(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram")
            error = replay.refuse(f"{ACTION_EVICT} x dram")
            self.assertEqual(error.code, CODE_TENSOR_IN_USE)
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COMPUTE c1 cpu")
            error = replay.refuse(f"{ACTION_EVICT} x dram")
            self.assertEqual(error.code, CODE_TENSOR_IN_USE)
            self.assertIn("in-flight task", str(error))

    def test_evicting_a_reservation_is_illegal(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram")
            self.assertEqual(dict(replay.state.copy_status)["x"], COPY_RESERVED_COPY)
            error = replay.refuse(f"{ACTION_EVICT} x vram")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn(COPY_RESERVED_COPY, str(error))

    def test_evicting_the_goal_output_is_a_live_value_loss(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COPY W dram vram", "ADVANCE",
                      "COMPUTE c1 gpu", "ADVANCE")
            error = replay.refuse(f"{ACTION_EVICT} y vram")
            self.assertEqual(error.code, CODE_LIVE_VALUE_LOSS)
            self.assertIn("deliver the requested output to dram", str(error))

    # -- the M1.3 rows: SSD destinations and SSD copies ---------------------

    def test_an_ssd_read_target_reservation_cannot_be_released(self):
        # M1_ACCEPTANCE.md §10: an SSD read's DRAM target is a reservation, and a
        # reservation is in-flight space, not a copy to be reclaimed.
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                      f"{ACTION_EVICT} x {LOC_DRAM}", f"{ACTION_READ_SSD} W")
            self.assertEqual(replay.state.copy_at("W", LOC_DRAM), COPY_RESERVED_COPY)
            error = replay.refuse(f"{ACTION_EVICT} W {LOC_DRAM}")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn(COPY_RESERVED_COPY, str(error))

    def test_a_second_write_while_one_is_in_flight_is_refused(self):
        # M1_ACCEPTANCE.md §10: "SSD目标已经READY/RESERVED时WRITE -> 拒绝，不覆盖
        # 同名快照、不重复计费". The destination check runs before the resource
        # check, so the refusal names the reservation rather than the busy slot.
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                      f"{ACTION_EVICT} x {LOC_DRAM}", f"{ACTION_WRITE_SSD} h")
            self.assertEqual(replay.state.copy_at("h", LOC_SSD), COPY_RESERVED_COPY)
            error = replay.refuse(f"{ACTION_WRITE_SSD} h")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("never copies onto", str(error))
            self.assertEqual(replay.state.copy_at("h", LOC_SSD), COPY_RESERVED_COPY)

    def test_releasing_the_dram_copy_needs_a_readable_snapshot(self):
        """M1_ACCEPTANCE.md §10: a written snapshot is a backup only if it can be read.

        Same state both times -- h is computed, written to the SSD and still has an
        unfinished CPU consumer -- with the read-back route present and absent.
        The SSD copy is READY either way; only its reachability differs, which is
        what decides whether dropping the DRAM copy loses a live value.
        """
        def plan(replay):
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE,
                      f"{ACTION_EVICT} x {LOC_DRAM}", f"{ACTION_WRITE_SSD} h", ACTION_ADVANCE)

        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            plan(replay)
            self.assertEqual(replay.state.copy_at("h", LOC_SSD), COPY_READY)
            replay.do(f"{ACTION_EVICT} h {LOC_DRAM}")
            self.assertEqual(replay.state.copy_at("h", LOC_DRAM), COPY_ABSENT)

        with m1_scenario("m1-spill", "m1-g-spill-dram96",
                         lambda w, s: transfer_only(s, ROUTE_SSD_TO_DRAM, "W")) as loaded:
            replay = M1Replay(loaded)
            plan(replay)
            error = replay.refuse(f"{ACTION_EVICT} h {LOC_DRAM}")
            self.assertEqual(error.code, CODE_LIVE_VALUE_LOSS)
            self.assertIn("c3", str(error))

    # -- through the loader -------------------------------------------------

    def assertDocumentRefused(self, mutate, error_type, needle: str, workload="m1-matvec",
                              scenario="m1-a-cpu6"):
        with self.assertRaises(error_type) as caught:
            with m1_scenario(workload, scenario, mutate):
                pass
        if needle:
            self.assertIn(needle, str(caught.exception))

    def assertDocumentAccepted(self, mutate, workload="m1-matvec", scenario="m1-a-cpu6"):
        """The counterpart of :meth:`assertDocumentRefused`: this must load.

        A refusal row is only meaningful if the same document without the one
        broken field is accepted; otherwise the test would pass on a document
        that is broken for an unrelated reason.
        """
        with m1_scenario(workload, scenario, mutate) as loaded:
            return loaded

    def test_a_missing_compute_cost_is_invalid_input(self):
        def drop_device(workload, scenario):
            del scenario["costs"]["compute"]["c1"]["gpu"]

        self.assertDocumentRefused(drop_device, InvalidInput, "never guesses a compute cost")

    def test_a_whole_missing_compute_entry_is_invalid_input(self):
        def drop_entry(workload, scenario):
            del scenario["costs"]["compute"]["c1"]

        self.assertDocumentRefused(drop_entry, InvalidInput, "never guesses a compute cost")

    def test_a_missing_transfer_table_is_invalid_input(self):
        def drop_table(workload, scenario):
            del scenario["costs"]["transfer"][ROUTE_DRAM_TO_VRAM]

        self.assertDocumentRefused(drop_table, InvalidInput, "never guesses a transfer cost")

    def test_a_missing_transfer_cost_is_invalid_input(self):
        def drop_tensor(workload, scenario):
            del scenario["costs"]["transfer"][ROUTE_DRAM_TO_VRAM]["x"]

        self.assertDocumentRefused(drop_tensor, InvalidInput, "never guesses a transfer cost")

    def test_a_cost_for_a_disallowed_device_is_invalid_input(self):
        def add_device(workload, scenario):
            allow_only(scenario, "c1", "cpu")
            scenario["costs"]["compute"]["c1"]["gpu"] = {"duration_ns": ms(2)}

        self.assertDocumentRefused(add_device, InvalidInput, "not an allowed device")

    def test_an_unknown_device_is_unsupported(self):
        def add_device(workload, scenario):
            scenario["mapspace"]["allowed_devices"]["c1"] = ["cpu", "tpu"]

        self.assertDocumentRefused(add_device, Unsupported, "tpu")

    def test_an_ssd_route_for_a_dram_only_tensor_is_invalid_input(self):
        def add_route(workload, scenario):
            scenario["mapspace"]["allowed_transfers"][ROUTE_SSD_TO_DRAM] = ["W"]
            scenario["costs"]["transfer"][ROUTE_SSD_TO_DRAM] = {"W": {"duration_ns": ms(4)}}

        self.assertDocumentRefused(add_route, InvalidInput, "no allowed route can get it to ssd")

    def test_only_a_leaf_may_have_an_ssd_source(self):
        # M1_DESIGN.md §4: a weight or an input may start on the SSD. A computed
        # value may not -- that would be skipping the compute that produces it.
        def move_the_output(workload, scenario):
            for tensor in workload["tensors"]:
                if tensor["id"] == "y":
                    tensor["initial_locations"] = [LOC_SSD]

        self.assertDocumentRefused(move_the_output, InvalidInput,
                                   "only a leaf weight or input may have an SSD source")

    def test_a_leaf_input_may_start_on_the_ssd(self):
        def move_the_input(workload, scenario):
            for tensor in workload["tensors"]:
                if tensor["id"] == "x":
                    tensor["initial_locations"] = [LOC_SSD]
            scenario["mapspace"]["allowed_transfers"][ROUTE_SSD_TO_DRAM] = ["x"]
            scenario["costs"]["transfer"][ROUTE_SSD_TO_DRAM] = {"x": {"duration_ns": ms(1)}}

        loaded = self.assertDocumentAccepted(move_the_input)
        tensor = next(t for t in loaded.workload.tensors if t.id == "x")
        self.assertEqual(tensor.initial_locations, (LOC_SSD,))

    def test_an_unknown_route_name_is_unsupported(self):
        def add_route(workload, scenario):
            scenario["mapspace"]["allowed_transfers"]["ssd_to_vram"] = ["W"]

        self.assertDocumentRefused(add_route, Unsupported, "no direct SSD-to-VRAM path")

    def test_an_incomplete_goal_is_invalid_input(self):
        def drop_output(workload, scenario):
            del scenario["goal"]["output_locations"]["y"]

        self.assertDocumentRefused(drop_output, InvalidInput, "is missing y")

    def test_a_goal_naming_something_that_is_not_an_output_is_invalid_input(self):
        def rename(workload, scenario):
            scenario["goal"]["output_locations"]["W"] = LOC_DRAM

        self.assertDocumentRefused(rename, InvalidInput, "is not one of the workload's outputs")

    def test_a_goal_in_a_domain_m1_does_not_have_is_invalid_input(self):
        # M1's three goal domains are dram, vram and ssd; a fourth name is a
        # document error, not a silently-ignored target.
        def bogus(workload, scenario):
            scenario["goal"]["output_locations"]["y"] = "pinned_host"

        self.assertDocumentRefused(bogus, InvalidInput, "must be one of")

    def test_an_ssd_goal_is_a_domain_m1_does_have(self):
        # The counterpart: the same field with the third legal domain loads.
        def to_ssd(workload, scenario):
            scenario["goal"]["output_locations"]["y"] = LOC_SSD
            scenario["mapspace"]["allowed_transfers"][ROUTE_DRAM_TO_SSD] = ["y"]
            scenario["costs"]["transfer"][ROUTE_DRAM_TO_SSD] = {"y": {"duration_ns": ms(2)}}

        loaded = self.assertDocumentAccepted(to_ssd)
        self.assertEqual(loaded.goal_location("y"), LOC_SSD)

    def test_a_goal_that_names_no_location_at_all_is_invalid_input(self):
        def drop_block(workload, scenario):
            del scenario["goal"]["output_locations"]

        self.assertDocumentRefused(drop_block, InvalidInput, "must be a JSON object")

    def test_the_architecture_still_takes_exactly_one_slot_per_resource(self):
        def widen(workload, scenario):
            scenario["architecture"]["pcie_copy_slots"] = 2

        self.assertDocumentRefused(widen, Unsupported, "exactly one resource of each kind")

    def test_a_0_1_scenario_cannot_carry_the_new_architecture_fields(self):
        def downgrade(workload, scenario):
            scenario["schema_version"] = SCHEMA_VERSION

        self.assertDocumentRefused(downgrade, InvalidInput, "dram_alignment_bytes")


class RefusalPairs(M1Case):
    """M1_ACCEPTANCE.md §10's refusal rows, the halves 0.1 could not reach.

    The table lists five pairs per shared resource and two directions per
    "input is elsewhere". 0.1 could only ever try the GPU half of each: it had
    one compute device and one host domain, so "D2H while an H2D is in flight"
    was not expressible and neither was "the CPU cannot read VRAM". Adding the
    CPU makes those reachable, which is exactly why §10 lists them.
    """

    def test_an_input_only_in_vram_is_not_ready_for_the_cpu(self):
        # The mirror of test_a_compute_whose_input_is_elsewhere_is_refused above:
        # that one is a GPU compute reaching for DRAM, this one is a CPU compute
        # reaching for VRAM. Same code, other direction.
        def only_in_vram(workload, scenario):
            for tensor in workload["tensors"]:
                if tensor["id"] == "x":
                    tensor["initial_locations"] = [LOC_VRAM]
            # The routes have to follow the move, and the loader says so: leaving
            # x on dram_to_vram would promise a copy into DRAM for a tensor that
            # has no way to get there.
            transfers = scenario["costs"]["transfer"]
            scenario["mapspace"]["allowed_transfers"][ROUTE_DRAM_TO_VRAM] = ["W"]
            transfers[ROUTE_DRAM_TO_VRAM] = {"W": transfers[ROUTE_DRAM_TO_VRAM]["W"]}
            scenario["mapspace"]["allowed_transfers"][ROUTE_VRAM_TO_DRAM].append("x")
            transfers[ROUTE_VRAM_TO_DRAM]["x"] = {"duration_ns": ms(1)}

        with m1_scenario("m1-matvec", "m1-a-cpu6", only_in_vram) as loaded:
            replay = M1Replay(loaded)
            error = replay.refuse(f"{ACTION_COMPUTE} c1 cpu")
            self.assertEqual(error.code, CODE_INPUT_NOT_READY)
            self.assertIn("never moves an input", str(error))
            # Readiness, not permission: one copy later the same action is legal.
            replay.do(f"{ACTION_COPY} x vram dram", ACTION_ADVANCE)
            replay.do(f"{ACTION_COMPUTE} c1 cpu")

    def test_a_write_whose_source_is_absent_is_illegal(self):
        # A write reads its source in DRAM, so "not ready" covers the WRITE half
        # of §10's "COPY/READ/WRITE source missing or not ready" row too. The
        # rejection must not leave the SSD target reserved behind it.
        with m1_scenario("m1-relu", "m1-h-ssd-cpu") as loaded:
            replay = M1Replay(loaded)
            error = replay.refuse(f"{ACTION_WRITE_SSD} y")
            self.assertEqual(error.code, CODE_INPUT_NOT_READY)
            self.assertEqual(replay.used_bytes(LOC_DRAM), 16)

    def test_the_bus_excludes_an_h2d_and_a_d2h(self):
        # §10 lists "H2D+D2H" among the pairs that must collide. The two moves
        # here are on different tensors, so nothing but the shared copy resource
        # can be what refuses the second one.
        with m1_scenario("m1-two-outputs", "m1-c-two-outputs") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COPY} u dram vram")
            error = replay.refuse(f"{ACTION_COPY} v vram dram")
            self.assertEqual(error.code, CODE_RESOURCE_BUSY)
            self.assertIn(RESOURCE_PCIE_COPY, str(error))

    def test_two_writes_collide_on_the_ssd_slot(self):
        # The other missing pair. Two writes of the *same* tensor are refused by
        # the destination check before the resource check ever runs (see
        # test_writing_the_output_twice_is_refused), so a second writable tensor
        # is what it takes to put two writes on ssd_io at once.
        def two_writable(workload, scenario):
            scenario["mapspace"]["allowed_transfers"][ROUTE_DRAM_TO_SSD] = ["x", "z"]
            scenario["costs"]["transfer"][ROUTE_DRAM_TO_SSD] = {
                "x": {"duration_ns": ms(2)},
                "z": {"duration_ns": ms(2)},
            }

        with m1_scenario("m1-spill", "m1-g-spill-dram96", two_writable) as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_WRITE_SSD} x")
            error = replay.refuse(f"{ACTION_WRITE_SSD} z")
            self.assertEqual(error.code, CODE_RESOURCE_BUSY)
            self.assertIn(RESOURCE_SSD_IO, str(error))

    def test_the_same_operation_cannot_start_on_the_other_device(self):
        # §10: "the second attempt is refused **even though the device differs**".
        # The existing case (test_a_busy_device_refuses_a_second_compute) retries
        # on the *same* device, where a check keyed on the device would refuse it
        # for the wrong reason and still look right. Here both device slots are
        # free -- c1 has never run, nothing is in flight -- so "already RUNNING"
        # can only come from the operation, not from the device it ran on.
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            replay = M1Replay(loaded)
            for tensor in ("W", "x"):
                replay.do(f"{ACTION_COPY} {tensor} dram vram", ACTION_ADVANCE)
            replay.do(f"{ACTION_COMPUTE} c1 gpu")
            error = replay.refuse(f"{ACTION_COMPUTE} c1 cpu")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)
            self.assertIn("already RUNNING", str(error))

    def test_an_in_flight_write_is_not_a_backup(self):
        """§10: the target being in flight does not license releasing the source.

        This row cannot be isolated on this graph, so the test pins the refusal
        and not the reason for it: h is c3's only way to its input, so an EVICT
        of the DRAM copy is refused with no write running as well (reachability:
        "leaves no way to get it to c3"). The write locks its source, and that
        check comes first, so what is asserted here is CODE_TENSOR_IN_USE.

        The mistake the row guards against -- treating a write that has not
        landed as a backup -- cannot happen on any graph: no code path ever asks
        an in-flight target for a backup, so there is nothing to pin.
        """
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu", ACTION_ADVANCE)
            replay.do(f"{ACTION_WRITE_SSD} h")
            error = replay.refuse(f"{ACTION_EVICT} h {LOC_DRAM}")
            self.assertEqual(error.code, CODE_TENSOR_IN_USE)


class LedgerDetail(M1Case):
    """M1_ACCEPTANCE.md §10: the rows about the two ledgers, not about a plan.

    Each one is a clause 0.1 satisfied trivially because it had one domain and
    one alignment, and which only becomes checkable once there are two.
    """

    def peak_bytes(self, scenario, actions, domain: str) -> int:
        """The high-water mark of one domain over a plan, from the ledger.

        ``SearchResult`` carries ``peak_vram_bytes`` only -- the 0.2 DRAM peak
        is written into ``stats.json`` -- so the peak is measured by replaying
        rather than read off the result. That also makes this the ledger's own
        number, not a second count kept somewhere else.
        """
        replay = M1Replay(scenario)
        peak = replay.used_bytes(domain)
        for action in actions:
            replay.do(action)
            peak = max(peak, replay.used_bytes(domain))
        return peak

    def workspace_run(self, device: str, workspace_bytes: int):
        """One op run with a given workspace, as (during, after) domain ledgers.

        The workspace is measured as a **difference against the same run with a
        zero workspace**, not against the ledger before the op starts: the op
        also allocates its output when it starts, so the raw delta is
        ``output + workspace`` and would be asserting two things at once.
        """
        def mutate(workload, scenario):
            set_compute(scenario, "c1", device, 1, workspace_bytes=workspace_bytes)

        with m1_scenario("m1-matvec", "m1-a-cpu6", mutate) as loaded:
            replay = M1Replay(loaded)
            if device != "cpu":
                for tensor in ("W", "x"):
                    replay.do(f"{ACTION_COPY} {tensor} dram vram", ACTION_ADVANCE)
            replay.do(f"{ACTION_COMPUTE} c1 {device}")
            during = (replay.used_bytes(LOC_DRAM), replay.used_bytes(LOC_VRAM))
            replay.do(ACTION_ADVANCE)
            after = (replay.used_bytes(LOC_DRAM), replay.used_bytes(LOC_VRAM))
        return during, after

    def test_a_cpu_workspace_is_charged_to_dram(self):
        # §10: "CPU/GPU workspace = 16 B -- reserved in DRAM/VRAM respectively,
        # released on completion, not double-charged". Every M1 fixture sets
        # workspace to 0, so this is the first case that exercises the DRAM arm.
        idle, idle_after = self.workspace_run("cpu", 0)
        during, after = self.workspace_run("cpu", 16)
        self.assertEqual(during[0] - idle[0], 16)   # charged to DRAM
        self.assertEqual(during[1] - idle[1], 0)    # and only to DRAM
        self.assertEqual(after, idle_after)         # released on completion

    def test_a_gpu_workspace_is_charged_to_vram(self):
        # The other arm, so "in DRAM/VRAM respectively" is pinned in both
        # directions rather than only where 0.1 already looked.
        idle, idle_after = self.workspace_run("gpu", 0)
        during, after = self.workspace_run("gpu", 16)
        self.assertEqual(during[1] - idle[1], 16)   # charged to VRAM
        self.assertEqual(during[0] - idle[0], 0)    # and only to VRAM
        self.assertEqual(after, idle_after)         # released on completion

    def test_a_runtime_reserve_is_inside_the_dram_peak(self):
        # §10: "reserve is charged from the start, and the peak includes it".
        # "From the start" is covered elsewhere (the t=0 ledger); this is the
        # peak half, and it is stated as a difference so it does not depend on
        # how many bytes x and y happen to be.
        def reserve(workload, scenario):
            scenario["architecture"]["dram_runtime_reserved_bytes"] = 16

        with m1_scenario("m1-relu", "m1-h-ssd-cpu") as plain:
            plain_result = search(plain)
            self.assertEqual(plain_result.status, STATUS_OPTIMAL)
            plain_peak = self.peak_bytes(plain, plain_result.actions, LOC_DRAM)

        with m1_scenario("m1-relu", "m1-h-ssd-cpu", reserve) as loaded:
            result = search(loaded)
            self.assertEqual(result.status, STATUS_OPTIMAL)
            self.assertEqual(
                self.peak_bytes(loaded, result.actions, LOC_DRAM), plain_peak + 16
            )

    def test_alignment_is_applied_per_domain(self):
        # §10: "a 16 B vector under a 32 B alignment occupies 32; two domains
        # with **different** alignments each apply their own". 0.1 has the first
        # two clauses (test_engine's AlignmentTests, one domain); a scenario that
        # only sets one of the two fields cannot pin the third.
        def skewed(workload, scenario):
            scenario["architecture"]["dram_alignment_bytes"] = 32
            scenario["architecture"]["vram_alignment_bytes"] = 1

        with m1_scenario("m1-two-outputs", "m1-c-two-outputs", skewed) as loaded:
            state = initial_state(loaded)
            # u (16 B) is in DRAM and rounds up to 32; v (16 B) is in VRAM and
            # does not round at all.
            self.assertEqual(used_bytes(loaded, state, LOC_DRAM), 32)
            self.assertEqual(used_bytes(loaded, state, LOC_VRAM), 16)

    def test_a_leaf_in_two_domains_is_charged_in_each(self):
        # §10: "two initial copies -- one charge in each domain".
        with m1_scenario("m1-fork", "m1-c-fork") as loaded:
            state = initial_state(loaded)
            self.assertEqual(used_bytes(loaded, state, LOC_DRAM), 16)
            self.assertEqual(used_bytes(loaded, state, LOC_VRAM), 16)

    def test_an_impossible_initial_layout_names_the_domain_and_the_numbers(self):
        # The other half of that row: the report has to say *which* budget was
        # exceeded and by how much, because "the scenario does not fit" with two
        # budgets and two numbers is not a diagnosis.
        def squeeze(workload, scenario):
            scenario["architecture"]["vram_capacity_bytes"] = 1

        with m1_scenario("m1-fork", "m1-c-fork", squeeze) as loaded:
            with self.assertRaises(InitialCapacityExceeded) as caught:
                initial_state(loaded)

        message = str(caught.exception)
        self.assertIn("vram_capacity_bytes", message)
        self.assertIn("needs 16 bytes", message)
        self.assertIn("is 1", message)
        # And the search reports the same thing as a verdict rather than as a
        # search that happened to find nothing.
        self.assertInfeasible(
            "m1-fork", "m1-c-fork", squeeze, reason=REASON_INITIAL_CAPACITY_EXCEEDED
        )

    def test_a_negative_or_non_finite_duration_is_refused(self):
        # §10: "0 / negative / non-integer / non-finite durations". The zero and
        # the non-integer are in test_spec; the negative and the two non-finite
        # spellings have no case anywhere, and JSON has no infinity literal, so
        # the non-finite ones can only arrive through an API or a patched file.
        #
        # Each spelling names its own reason: a negative is a *range* fault and
        # a NaN is a *type* fault, and reporting the second as the first would
        # still contain "duration_ns" and still pass a looser assertion.
        cases = (
            (-1, "must be >= 1, got -1"),
            (float("nan"), "must be an integer number"),
            (float("inf"), "must be an integer number"),
        )
        for bad, reason in cases:
            with self.subTest(duration_ns=bad):
                def break_duration(workload, scenario, value=bad):
                    scenario["costs"]["compute"]["c1"]["cpu"]["duration_ns"] = value

                with self.assertRaises(InvalidInput) as caught:
                    with m1_scenario("m1-matvec", "m1-a-cpu6", break_duration):
                        pass
                self.assertIn("duration_ns", str(caught.exception))
                self.assertIn(reason, str(caught.exception))


# ---------------------------------------------------------------------------
# §11  Invariants -- capacity and monotonicity
# ---------------------------------------------------------------------------


class Invariants(M1Case):
    """M1_ACCEPTANCE.md §9: what must hold for every scenario, not just the table."""

    def test_a_budget_cut_is_never_reported_as_infeasible(self):
        result = self.solve("m1-two-weights-ssd", "m1-e-dram128", max_expanded_states=3)
        self.assertIn(result.status, (STATUS_FEASIBLE, STATUS_UNKNOWN))
        self.assertFalse(result.optimality_proven)

    def test_an_exhausted_search_space_is_infeasible_and_says_so(self):
        result = self.assertInfeasible("m1-two-weights-ssd", "m1-e-dram128",
                                       lambda w, s: set_capacity(s, dram=63))
        self.assertEqual(result.termination_reason, "search_space_exhausted")

    def test_a_dead_end_mapping_is_incomplete_not_infeasible(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COMPUTE c1 cpu", "ADVANCE")
            result = evaluate_mapping(loaded, replay.applied)
            self.assertEqual(result.status, STATUS_INCOMPLETE_MAPPING)
            self.assertIn(REASON_UNFINISHED, result.reason)
            self.assertIsNone(result.makespan_ns)

    def test_an_overshooting_mapping_is_invalid(self):
        with m1_scenario("m1-fork", "m1-c-fork") as loaded:
            replay = M1Replay(loaded)
            replay.do("COMPUTE ca cpu", "ADVANCE", "COMPUTE cb gpu", "ADVANCE",
                      "COPY b vram dram", "ADVANCE", "COMPUTE add cpu", "ADVANCE")
            self.assertTrue(is_goal(loaded, replay.state))
            self.assertEqual(evaluate_mapping(loaded, replay.applied).reason, REASON_GOAL_REACHED)
            # One more legal action after the goal: `a` is dead the moment `add`
            # is done, so releasing it is allowed -- and makes the mapping no
            # longer a minimal plan. Saying so is not the same as being illegal.
            replay.do(f"{ACTION_EVICT} a dram")
            result = evaluate_mapping(loaded, replay.applied)
            self.assertEqual(result.status, STATUS_INVALID_MAPPING)
            self.assertIn(REASON_ACTIONS_REMAIN_AFTER_GOAL, result.reason)

    def test_a_goal_needs_the_output_in_its_named_domain(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COPY W dram vram", "ADVANCE",
                      "COMPUTE c1 gpu", "ADVANCE")
            self.assertEqual(dict(replay.state.op_status)["c1"], OP_DONE)
            self.assertEqual(replay.state.running, ())
            self.assertEqual(dict(replay.state.copy_status)["y"], COPY_READY)
            # Computed, nothing in flight, and still not a goal: the result is
            # in VRAM and the goal asked for DRAM.
            self.assertFalse(is_goal(loaded, replay.state))
            replay.do("COPY y vram dram", "ADVANCE")
            self.assertTrue(is_goal(loaded, replay.state))

    def test_higher_capacity_never_makes_a_mapping_worse(self):
        previous = None
        for budget in (63, 80, 96, 128, 160):
            result = self.solve("m1-matvec-ssd", "m1-f-dram96",
                                lambda w, s, b=budget: set_capacity(s, dram=b))
            if result.status == STATUS_INFEASIBLE:
                continue
            if previous is not None:
                self.assertLessEqual(result.makespan_ns, previous)
            previous = result.makespan_ns
        self.assertIsNotNone(previous)

    def test_allowing_one_more_device_never_makes_a_mapping_worse(self):
        cpu_only = self.makespan("m1-matvec", "m1-a-cpu6",
                                 lambda w, s: allow_only(s, "c1", "cpu"))
        both = self.makespan("m1-matvec", "m1-a-cpu6")
        self.assertLessEqual(both, cpu_only)

    def test_an_infeasible_scenario_reports_its_own_makespan_as_none(self):
        result = self.assertInfeasible("m1-two-weights-ssd", "m1-e-dram128",
                                       lambda w, s: set_capacity(s, dram=63))
        self.assertIsNone(result.makespan_ns)

    def test_the_state_key_ignores_the_order_things_were_completed_in(self):
        # Two orders, the same residency: one key, or the search would expand
        # the same state twice.
        with m1_scenario("m1-fork", "m1-c-fork") as loaded:
            first = M1Replay(loaded)
            first.do("COMPUTE ca cpu", "ADVANCE", "COMPUTE cb gpu", "ADVANCE")
            second = M1Replay(loaded)
            second.do("COMPUTE cb gpu", "ADVANCE", "COMPUTE ca cpu", "ADVANCE")
            self.assertEqual(first.t, second.t)
            self.assertEqual(first.state.key(), second.state.key())

    def test_the_state_key_separates_the_two_domains(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            evicted = M1Replay(loaded)
            evicted.do("COPY x dram vram", "ADVANCE", f"{ACTION_EVICT} x dram")
            kept = M1Replay(loaded)
            kept.do("COPY x dram vram", "ADVANCE")
            self.assertNotEqual(evicted.state.key(), kept.state.key())

    def test_the_state_key_separates_the_direction_of_a_running_task(self):
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            gpu = M1Replay(loaded)
            gpu.do("COPY x dram vram", "ADVANCE", "COPY W dram vram", "ADVANCE",
                   f"{ACTION_COMPUTE} c1 gpu")
            cpu = M1Replay(loaded)
            cpu.do(f"{ACTION_COMPUTE} c1 cpu")
            self.assertEqual(gpu.state.running[0].device, "gpu")
            self.assertEqual(cpu.state.running[0].device, "cpu")
            self.assertNotEqual(gpu.state.key(), cpu.state.key())

    def test_the_state_key_ignores_a_reservation_that_is_only_in_flight(self):
        # A reservation owns bytes but is not residency: two states that differ
        # only in which copy is reserved are the same state.
        with m1_scenario("m1-matvec", "m1-a-cpu6") as loaded:
            x_first = M1Replay(loaded)
            x_first.do("COPY x dram vram")
            w_first = M1Replay(loaded)
            w_first.do("COPY W dram vram")
            self.assertNotEqual(x_first.state.key(), w_first.state.key())
            self.assertEqual(dict(x_first.state.copy_status)["x"], COPY_RESERVED_COPY)


# ---------------------------------------------------------------------------
# §11  Fingerprints
# ---------------------------------------------------------------------------


class Fingerprints(M1Case):
    """What ``scenario_fingerprint`` must cover, so identity means identity."""

    def fingerprint(self, workload: str, scenario: str, mutate=None) -> str:
        with m1_scenario(workload, scenario, mutate) as loaded:
            return scenario_fingerprint(loaded)

    def assertFingerprintChanges(self, label: str, mutate):
        reference = self.fingerprint("m1-matvec", "m1-a-cpu6")
        self.assertNotEqual(self.fingerprint("m1-matvec", "m1-a-cpu6", mutate), reference,
                            f"{label} is not part of the fingerprint")

    def test_the_device_set_is_covered(self):
        self.assertFingerprintChanges("the device set",
                                      lambda w, s: allow_only(s, "c1", "gpu"))

    def test_the_route_set_is_covered(self):
        self.assertFingerprintChanges("the route set",
                                      lambda w, s: transfer_only(s, ROUTE_DRAM_TO_VRAM, "W"))

    def test_an_ssd_route_is_covered(self):
        reference = self.fingerprint("m1-shared-weight", "m1-recovery-shared")
        self.assertNotEqual(
            self.fingerprint("m1-shared-weight", "m1-recovery-shared", drop_ssd_route),
            reference,
        )

    def test_both_capacities_are_covered(self):
        self.assertFingerprintChanges("dram_capacity_bytes", lambda w, s: set_capacity(s, dram=128))
        self.assertFingerprintChanges("vram_capacity_bytes", lambda w, s: set_capacity(s, vram=128))

    def test_the_ssd_read_cost_is_covered(self):
        reference = self.fingerprint("m1-matvec-ssd", "m1-f-dram96")
        self.assertNotEqual(
            self.fingerprint("m1-matvec-ssd", "m1-f-dram96",
                             lambda w, s: set_transfer(s, ROUTE_SSD_TO_DRAM, "W", 9)),
            reference,
        )

    def test_the_goal_domain_is_covered(self):
        self.assertFingerprintChanges(
            "the goal domain",
            lambda w, s: s["goal"]["output_locations"].__setitem__("y", LOC_VRAM),
        )

    def test_the_initial_placement_is_covered(self):
        self.assertFingerprintChanges(
            "initial_locations",
            lambda w, s: w["tensors"][0]["initial_locations"].append(LOC_SSD),
        )

    def test_every_duration_is_covered(self):
        self.assertFingerprintChanges("a transfer duration",
                                      lambda w, s: set_transfer(s, ROUTE_DRAM_TO_VRAM, "W", 9))
        self.assertFingerprintChanges("a compute duration",
                                      lambda w, s: set_compute(s, "c1", "cpu", 9))
        self.assertFingerprintChanges("a d2h duration",
                                      lambda w, s: set_transfer(s, ROUTE_VRAM_TO_DRAM, "y", 9))

    def test_the_alignment_the_reserve_and_the_switches_are_covered(self):
        self.assertFingerprintChanges(
            "vram_alignment_bytes",
            lambda w, s: s["architecture"].__setitem__("vram_alignment_bytes", 32),
        )
        self.assertFingerprintChanges(
            "dram_runtime_reserved_bytes",
            lambda w, s: s["architecture"].__setitem__("dram_runtime_reserved_bytes", 8),
        )
        for switch in ("allow_copy_compute_overlap", "allow_eviction", "allow_cpu_gpu_overlap"):
            with self.subTest(switch=switch):
                self.assertFingerprintChanges(
                    switch, lambda w, s, k=switch: s["mapspace"].__setitem__(k, False)
                )

    def test_the_cost_source_is_covered(self):
        self.assertFingerprintChanges(
            "costs.source", lambda w, s: s["costs"].__setitem__("source", "derived")
        )

    def test_the_domain_and_route_vocabulary_is_fixed(self):
        self.assertEqual(DOMAIN_OF_DEVICE["cpu"], LOC_DRAM)
        self.assertEqual(DOMAIN_OF_DEVICE["gpu"], LOC_VRAM)
        self.assertEqual(ROUTES, (ROUTE_DRAM_TO_VRAM, ROUTE_VRAM_TO_DRAM,
                                  ROUTE_SSD_TO_DRAM, ROUTE_DRAM_TO_SSD))


# ---------------------------------------------------------------------------
# §10  Recovering a released weight from the SSD
# ---------------------------------------------------------------------------


class WeightRecovery(M1Case):
    """The shared-weight case: release the DRAM cache, then read it back.

    ``m1-recovery-shared`` has one SSD-resident weight with two consumers and no
    PCIe route at all, so the only way to run both multiplies is to read the
    weight twice -- unless it never has to leave DRAM.
    """

    def test_one_read_serves_both_consumers_when_it_fits(self):
        result = self.solve("m1-shared-weight", "m1-recovery-shared")
        self.assertEqual(result.makespan_ns, ms(9))
        self.assertEqual(result.evaluation.ssd_read_bytes, 64)

    def test_the_release_and_re_read_plan_costs_thirteen(self):
        """The hand plan from M1_ACCEPTANCE.md §8, timed and counted per action.

        Nothing here is subtle except that the plan is *legal*: the DRAM copy of
        W is released while the second consumer still needs it, and the SSD
        source is what makes that legal. The search does not choose this plan --
        9 ms beats 13 -- but a legal plan that is worse must still evaluate, and
        its bytes must still add up.
        """
        with m1_scenario("m1-shared-weight", "m1-recovery-shared") as loaded:
            replay = M1Replay(loaded)
            replay.do(
                f"{ACTION_READ_SSD} W", "ADVANCE",   #  0 -> 4  the first read
                "COMPUTE c1 cpu", "ADVANCE",          #  4 -> 6
                f"{ACTION_EVICT} W dram",             #  release the cache
                f"{ACTION_READ_SSD} W", "ADVANCE",   #  6 -> 10 the re-read
                "COMPUTE c2 cpu", "ADVANCE",          # 10 -> 12
                "COMPUTE add cpu", "ADVANCE",         # 12 -> 13
            )
            result = evaluate_mapping(loaded, replay.applied)
            self.assertEqual(result.status, STATUS_VALID)
            self.assertEqual(result.makespan_ns, ms(13))
            self.assertEqual(result.ssd_read_bytes, 128)
            self.assertEqual(result.peak_dram_bytes, 144)
            # No PCIe route exists in this scenario, so no copy may appear.
            self.assertEqual(
                [event.kind for event in result.events if event.kind == ACTION_COPY], []
            )

    def test_the_release_needs_the_ssd_source_to_be_legal(self):
        with m1_scenario("m1-shared-weight", "m1-recovery-shared") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_READ_SSD} W", "ADVANCE", "COMPUTE c1 cpu", "ADVANCE")
            self.assertEqual(dict(replay.state.op_status)["c2"], OP_NOT_STARTED)
            replay.do(f"{ACTION_EVICT} W dram")
            self.assertEqual(dict(replay.state.dram_status)["W"], COPY_ABSENT)
            # ... and the second consumer is still served, by reading W back
            # into the same domain it was released from.
            replay.do(f"{ACTION_READ_SSD} W", "ADVANCE")
            self.assertEqual(dict(replay.state.dram_status)["W"], COPY_READY)

    def test_without_the_ssd_route_the_whole_plan_is_infeasible(self):
        result = self.assertInfeasible("m1-shared-weight", "m1-recovery-shared", drop_ssd_route)
        self.assertEqual(result.termination_reason, "search_space_exhausted")


# ---------------------------------------------------------------------------
# §10  Reachability, and the read locks that are deliberately not part of it
# ---------------------------------------------------------------------------


class Reachability(M1Case):
    """Why an eviction is allowed, and why "allowed" is judged on routes."""

    def test_a_reservation_in_another_domain_is_not_a_backup(self):
        """A hand-built state: the VRAM copy of h exists, but only as a reservation.

        Through legal moves this state is not reachable with a *live* c2 -- the
        source would be locked, or the reservation would already be READY -- so
        it is constructed directly. It is discriminating: counting the
        reservation as a backup would make VRAM look reachable and the eviction
        wrongly legal.
        """
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            state = initial_state(loaded)

            def replace(mapping, tensor, value):
                return tuple((name, value if name == tensor else other)
                             for name, other in mapping)

            state = dataclasses.replace(
                state,
                t=ms(3),
                op_status=replace(state.op_status, "c1", OP_DONE),
                copy_status=replace(state.copy_status, "h", COPY_RESERVED_COPY),
                dram_status=replace(replace(state.dram_status, "h", COPY_READY),
                                    "x", COPY_ABSENT),
            )
            replay = M1Replay(loaded, state)
            error = replay.refuse(f"{ACTION_EVICT} h dram")
            self.assertEqual(error.code, CODE_LIVE_VALUE_LOSS)
            self.assertIn("remaining copies", str(error))

    def test_one_reachable_device_is_enough(self):
        """h sits in DRAM; c2 can still run on the CPU, so releasing VRAM is fine."""
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COMPUTE c1 gpu", "ADVANCE",
                      "COPY h vram dram", "ADVANCE")
            self.assertEqual(dict(replay.state.dram_status)["h"], COPY_READY)
            replay.do(f"{ACTION_EVICT} h vram")
            self.assertEqual(dict(replay.state.copy_status)["h"], COPY_ABSENT)

    def test_the_same_release_is_refused_when_no_route_can_restore_it(self):
        """The same release, with the only device that could still be served cut off.

        c2 is pinned to the GPU and the route that could carry h back to VRAM is
        removed, so the DRAM copy is no longer enough. This is the pair that
        shows the rule is about reachability and not about "is there a copy".
        """
        def cut(workload, scenario):
            transfer_only(scenario, ROUTE_DRAM_TO_VRAM, "x")
            allow_only(scenario, "c2", "gpu")

        with m1_scenario("m1-chain", "m1-b-hybrid", cut) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", "COMPUTE c1 gpu", "ADVANCE",
                      "COPY h vram dram", "ADVANCE")
            error = replay.refuse(f"{ACTION_EVICT} h vram")
            self.assertEqual(error.code, CODE_LIVE_VALUE_LOSS)
            self.assertIn("c2", str(error))
            self.assertIn(f"remaining copies are {LOC_DRAM}", str(error))

    def test_a_gpu_compute_locks_the_vram_copy_and_not_the_dram_one(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", "ADVANCE", f"{ACTION_COMPUTE} c1 gpu")
            self.assertEqual(replay.refuse(f"{ACTION_EVICT} x vram").code, CODE_TENSOR_IN_USE)
            replay.do(f"{ACTION_EVICT} x dram")
            self.assertEqual(dict(replay.state.dram_status)["x"], COPY_ABSENT)

    def test_a_cpu_compute_locks_the_dram_copy_instead(self):
        with m1_scenario("m1-chain", "m1-b-hybrid") as loaded:
            replay = M1Replay(loaded)
            replay.do(f"{ACTION_COMPUTE} c1 cpu")
            self.assertEqual(replay.refuse(f"{ACTION_EVICT} x dram").code, CODE_TENSOR_IN_USE)
            # There is no VRAM copy to release at all, which is a different error.
            self.assertEqual(replay.refuse(f"{ACTION_EVICT} x vram").code, CODE_ILLEGAL_ACTION)

    def test_a_saved_intermediate_is_restored_for_the_next_gpu_consumer(self):
        """M1_ACCEPTANCE.md §10's second recovery case, played to the end.

        ``test_one_reachable_device_is_enough`` stops at the eviction; this runs
        the whole cycle -- save h to DRAM, release VRAM, copy h back for c2 -- and
        checks that nothing was recomputed along the way. The saved copy is what
        makes the release legal, and the route is what makes it useful.
        """
        with m1_scenario("m1-chain", "m1-b-hybrid",
                         lambda w, s: (allow_only(s, "c1", "gpu"),
                                       allow_only(s, "c2", "gpu"))) as loaded:
            replay = M1Replay(loaded)
            replay.do("COPY x dram vram", ACTION_ADVANCE, f"{ACTION_COMPUTE} c1 gpu",
                      ACTION_ADVANCE)
            replay.do("COPY h vram dram")               # save before the squeeze
            replay.do(ACTION_ADVANCE, f"{ACTION_EVICT} h vram")
            self.assertEqual(replay.state.copy_at("h", LOC_DRAM), COPY_READY)
            replay.do("COPY h dram vram", ACTION_ADVANCE)  # restore for c2
            replay.do(f"{ACTION_COMPUTE} c2 gpu", ACTION_ADVANCE)
            replay.do("COPY y vram dram", ACTION_ADVANCE)
            self.assertTrue(is_goal(loaded, replay.state))
            # One compute each: the intermediate came back from DRAM, it was not
            # recomputed (the mapspace forbids recomputation in the first place).
            self.assertEqual(sum(1 for a in replay.applied if a.kind == ACTION_COMPUTE), 2)
            self.assertEqual(sum(1 for a in replay.applied
                                 if a.kind == ACTION_READ_SSD), 0)


# ---------------------------------------------------------------------------
# §11  Compatibility with 0.1
# ---------------------------------------------------------------------------


class Compatibility(M1Case):
    """M1_ACCEPTANCE.md §10: 0.1 keeps working, unchanged."""

    def test_both_schema_versions_are_what_they_were(self):
        self.assertEqual(SCHEMA_VERSION, "0.1")
        self.assertEqual(SCHEMA_VERSION_M1, "0.2")

    def test_a_0_1_scenario_still_loads_and_searches(self):
        scenario = load_and_validate(EXAMPLES / "chain-cap160.json")
        self.assertEqual(scenario.schema_version, SCHEMA_VERSION)
        self.assertEqual(scenario.id, "chain-cap160")
        result = search(scenario)
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.makespan_ns, ms(8))

    def test_a_0_1_mapping_still_replays_against_its_0_1_scenario(self):
        scenario = load_and_validate(EXAMPLES / "chain-cap160.json")
        document = load_mapping(EXAMPLES / "chain.mapping.json", scenario)
        # The checked-in 0.1 mapping carries no fingerprints, so it is replayed
        # rather than verified -- and that is a documented outcome, not a pass.
        self.assertFalse(document.fingerprint_verified)
        result = evaluate_mapping(scenario, document.actions)
        self.assertEqual(result.status, STATUS_VALID)

    def test_a_0_1_workload_may_still_be_used_by_a_0_2_scenario(self):
        """``M1_DESIGN.md`` §2: M1 scenarios may reference an old 0.1 workload.

        ``chain.workload.json`` is the 0.1 fixture: its input starts in VRAM, the
        0.1 way. The scenario has to describe it as it is -- a route for a tensor
        the workload cannot move to that route's source end is refused -- so the
        mapspace and the costs below are written for chain's own tensors.
        """
        workload_path = EXAMPLES / "chain.workload.json"
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        self.assertEqual(workload["schema_version"], SCHEMA_VERSION)

        scenario = json.loads((EXAMPLES / "m1-b-hybrid.json").read_text(encoding="utf-8"))
        scenario["workload_file"] = "chain.workload.json"
        scenario["mapspace"] = {
            "allowed_devices": {"c1": ["cpu", "gpu"], "c2": ["cpu", "gpu"]},
            "allowed_transfers": {
                ROUTE_DRAM_TO_VRAM: ["W1", "W2", "h"],
                ROUTE_VRAM_TO_DRAM: ["h", "y"],
            },
            "allow_eviction": True,
            "allow_copy_compute_overlap": True,
            "allow_cpu_gpu_overlap": True,
            "allow_recomputation": False,
        }
        compute = {"cpu": {"duration_ns": ms(1), "workspace_bytes": 0},
                   "gpu": {"duration_ns": ms(1), "workspace_bytes": 0}}
        scenario["costs"] = {
            "source": "synthetic_fixed",
            "compute": {"c1": dict(compute), "c2": dict(compute)},
            "transfer": {
                ROUTE_DRAM_TO_VRAM: {t: {"duration_ns": ms(2)} for t in ("W1", "W2", "h")},
                ROUTE_VRAM_TO_DRAM: {t: {"duration_ns": ms(1)} for t in ("h", "y")},
            },
        }
        scenario["goal"] = {"output_locations": {"y": LOC_DRAM}}

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            directory = Path(raw)
            (directory / "chain.workload.json").write_text(
                workload_path.read_text(encoding="utf-8"), encoding="utf-8")
            target = directory / "chain-0-2.json"
            target.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
            loaded = load_and_validate(target)
            result = search(loaded)

        self.assertEqual(loaded.schema_version, SCHEMA_VERSION_M1)
        self.assertEqual(loaded.workload.id, "chain")
        self.assertEqual(result.status, STATUS_OPTIMAL)
        # The optimum splits the chain across the two devices: W1 goes to VRAM
        # for c1, then h comes back so the CPU can run c2 next to W2 in DRAM.
        # Only one weight crosses, which is the point of a hybrid plan.
        self.assertEqual(result.makespan_ns, ms(5))
        self.assertEqual(result.h2d_bytes, 64)

    def test_the_mapping_document_spells_the_new_actions(self):
        """A saved 0.2 mapping round-trips, and the two illegal spellings do not.

        ``WRITE_SSD`` is the one action M1.3 added, so it is the one that had no
        0.1 spelling to be confused with; the two refusals are the vocabulary
        boundary -- an SSD-to-VRAM copy and an SSD eviction are the two things
        M1 deliberately does not have (``M1_DESIGN.md`` §4).
        """
        scenario = load_and_validate(EXAMPLES / "m1-g-spill-dram96.json")
        plan = [
            {"kind": ACTION_COMPUTE, "operation_id": "c1", "device": "cpu"},
            {"kind": ACTION_ADVANCE},
            {"kind": ACTION_EVICT, "tensor_id": "x", "location": LOC_DRAM},
            {"kind": ACTION_WRITE_SSD, "tensor_id": "h"},
            {"kind": ACTION_ADVANCE},
            {"kind": ACTION_EVICT, "tensor_id": "h", "location": LOC_DRAM},
            {"kind": ACTION_READ_SSD, "tensor_id": "W"},
            {"kind": ACTION_ADVANCE},
            {"kind": ACTION_COMPUTE, "operation_id": "c2", "device": "cpu"},
            {"kind": ACTION_ADVANCE},
            {"kind": ACTION_EVICT, "tensor_id": "W", "location": LOC_DRAM},
            {"kind": ACTION_EVICT, "tensor_id": "z", "location": LOC_DRAM},
            {"kind": ACTION_READ_SSD, "tensor_id": "h"},
            {"kind": ACTION_ADVANCE},
            {"kind": ACTION_COMPUTE, "operation_id": "c3", "device": "cpu"},
            {"kind": ACTION_ADVANCE},
        ]
        header = {
            "schema_version": SCHEMA_VERSION_M1,
            "scenario_id": scenario.id,
            "comment": "hand-written, for the loader test",
            "scenario_fingerprint": scenario_fingerprint(scenario),
            "workload_fingerprint": workload_fingerprint(scenario.workload),
        }

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            directory = Path(raw)

            def load(actions):
                target = directory / "m.json"
                target.write_text(json.dumps({**header, "actions": actions}, indent=2),
                                  encoding="utf-8", newline="\n")
                return load_mapping(target, scenario)

            document = load(plan)
            result = evaluate_mapping(scenario, document.actions)
            self.assertEqual(result.status, STATUS_VALID)
            self.assertEqual(result.makespan_ns, ms(10))

            for bad, needle in (
                ({"kind": ACTION_COPY, "tensor_id": "h", "src": LOC_SSD, "dst": LOC_DRAM},
                 "'ssd'->'dram' involves the SSD"),
                ({"kind": ACTION_COPY, "tensor_id": "h", "src": LOC_VRAM, "dst": LOC_SSD},
                 "'vram'->'ssd' involves the SSD"),
                ({"kind": ACTION_EVICT, "tensor_id": "h", "location": LOC_SSD},
                 "does not model eviction from the SSD"),
            ):
                with self.assertRaises(Unsupported) as caught:
                    load([bad])
                self.assertIn(needle, str(caught.exception))

    def test_a_0_1_scenario_id_is_not_a_0_2_scenario(self):
        """Identity is checked before anything is replayed."""
        scenario = load_and_validate(EXAMPLES / "m1-a-cpu6.json")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            target = Path(raw) / "m.json"
            target.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION_M1,
                "scenario_id": "chain-cap160",
                "scenario_fingerprint": None,
                "workload_fingerprint": None,
                "actions": [],
            }), encoding="utf-8")
            with self.assertRaises(MappingIdentityError):
                load_mapping(target, scenario)


# ---------------------------------------------------------------------------
# §12  The 0.2 report
# ---------------------------------------------------------------------------


class Report(M1Case):
    """``M1_ACCEPTANCE.md`` §12.3: the artefacts carry the 0.2 numbers, and 0.1's
    artefacts carry nothing new.

    Deliberately short. ``M1_DESIGN.md`` §12 asks for coverage of placement,
    retention, capacity and optimality, not for one case per emitted key -- what
    is tested here is the handful of ways the *writer* can be wrong in a way the
    engine is not: dropping a field the loader requires, inventing a zero where
    the honest answer is "absent", and letting an extension leak into a 0.1
    document.
    """

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from tensor_mapping import cli

        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def read(self, directory: Path, name: str) -> dict:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    def test_a_search_artefact_replays_to_its_own_numbers(self) -> None:
        """The saved mapping must be loadable by the tool that wrote it.

        This is the round trip ``DESIGN.md`` §4.3 exists for, and the 0.2 writer
        is the only place it can break without the engine noticing: an action
        written without its ``device`` or its ``src``/``dst`` is well-formed JSON
        that ``load_mapping`` refuses, so the failure surfaces as "the search
        wrote a file the model command cannot read" rather than as a wrong
        number. The four documents must also agree afterwards -- same makespan,
        same bytes -- which is what makes the replay a replay.
        """
        from tensor_mapping import cli

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            searched = Path(raw) / "search"
            replayed = Path(raw) / "model"
            scenario = str(EXAMPLES / "m1-g-spill-dram96.json")

            code, _, err = self.run_cli(
                "search", "--scenario", scenario, "--out", str(searched)
            )
            self.assertEqual(code, cli.EXIT_OK, err)
            code, _, err = self.run_cli(
                "model",
                "--scenario",
                scenario,
                "--mapping",
                str(searched / cli.MAPPING_FILE),
                "--out",
                str(replayed),
            )
            self.assertEqual(code, cli.EXIT_OK, err)

            # `mode` names the command that ran, not the plan, so the two logs
            # are compared on their payload rather than byte for byte. `model`
            # writes no mapping -- it was given one -- so only these two exist on
            # both sides.
            for name, key in ((cli.EVENTS_FILE, "events"), (cli.STATES_FILE, "states")):
                with self.subTest(name=name):
                    self.assertEqual(
                        self.read(searched, name)[key],
                        self.read(replayed, name)[key],
                        f"{name} must record the same run whichever command made it",
                    )

            first = self.read(searched, cli.STATS_FILE)
            second = self.read(replayed, cli.STATS_FILE)
            # `mode`, `status` and `reason` name the command that ran rather than
            # the plan, `search` is the budget accounting only a search has, and
            # only `model` records the path of the mapping it was handed. What is
            # left is the run itself: the plan's making, its bytes and its peaks.
            for key in ("mode", "status", "reason", "mapping_file", "search"):
                first.pop(key, None)
                second.pop(key, None)
            self.assertEqual(first, second)
            self.assertEqual(first["makespan_ns"], ms(10))
            self.assertEqual(first["ssd_read_bytes"], 80)
            self.assertEqual(first["ssd_write_bytes"], 16)

    def test_the_writer_round_trips_every_0_2_action_kind(self) -> None:
        """``action_to_dict`` writes what ``load_mapping`` reads, kind by kind.

        The four action kinds that carry 0.2 fields are all here, because the
        failure mode is per-kind: it would be entirely possible to remember
        ``device`` for COMPUTE and forget ``location`` for EVICT, and the three
        kinds that carry nothing would still round-trip.
        """
        from tensor_mapping.artifacts import action_to_dict

        scenario = load_and_validate(EXAMPLES / "m1-g-spill-dram96.json")
        actions = tuple(
            parse_action_m1(spec)
            for spec in (
                "COMPUTE c1 cpu",
                "ADVANCE",
                "COPY x dram vram",
                "WRITE_SSD h",
                "READ_SSD W",
                "EVICT h dram",
            )
        )
        header = {
            "schema_version": SCHEMA_VERSION_M1,
            "scenario_id": scenario.id,
            "scenario_fingerprint": scenario_fingerprint(scenario),
            "workload_fingerprint": workload_fingerprint(scenario.workload),
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            target = Path(raw) / "m.json"
            target.write_text(
                json.dumps(
                    {**header, "actions": [action_to_dict(a, m1=True) for a in actions]},
                    indent=2,
                ),
                encoding="utf-8",
                newline="\n",
            )
            document = load_mapping(target, scenario)

        self.assertEqual(document.actions, actions)
        # And the 0.1 spelling of the same list is *not* loadable as 0.2, which is
        # what makes the round trip above a test rather than a tautology.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            target = Path(raw) / "m.json"
            target.write_text(
                json.dumps(
                    {**header, "actions": [action_to_dict(a) for a in actions]},
                    indent=2,
                ),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaises(InvalidInput):
                load_mapping(target, scenario)

    def test_a_0_2_stats_document_reports_the_four_directions_and_the_goal(self) -> None:
        """The H row: an SSD output is delivered, and the report says where to."""
        from tensor_mapping.artifacts import limits_block, m1_metrics, stats_document

        scenario = load_and_validate(EXAMPLES / "m1-h-ssd-cpu.json")
        result = search(scenario)
        document = stats_document(
            mode="search",
            scenario=scenario,
            status=result.status,
            reason=result.termination_reason,
            makespan_ns=result.makespan_ns,
            peak_vram_bytes=result.peak_vram_bytes,
            h2d_bytes=result.h2d_bytes,
            action_count=result.evaluation.action_count,
            declared_action_count=len(result.actions),
            limits=limits_block(
                scenario, max_expanded_states=0, wall_time_limit_s=0.0
            ),
            m1=m1_metrics(scenario, result.evaluation),
        )

        self.assertEqual(document["schema_version"], SCHEMA_VERSION_M1)
        self.assertEqual(document["makespan_ns"], ms(3))
        self.assertEqual(document["ssd_write_bytes"], 16)
        self.assertEqual(document["ssd_read_bytes"], 0)
        self.assertEqual(document["h2d_bytes"], 0)
        self.assertEqual(document["d2h_bytes"], 0)
        self.assertEqual(document["peak_dram_bytes"], 32)
        self.assertEqual(document["peak_vram_bytes"], 0)
        self.assertEqual(document["target_locations"], {"y": LOC_SSD})
        # Service time is a fold over the events, so the two must agree: 1 ms of
        # CPU compute then 2 ms of the DRAM-to-SSD write that delivers `y`, which
        # is exactly the 3 ms makespan -- nothing overlaps because nothing can.
        self.assertEqual(
            document["service_ns"],
            {
                RESOURCE_CPU_COMPUTE: ms(1),
                RESOURCE_GPU_COMPUTE: 0,
                ROUTE_DRAM_TO_VRAM: 0,
                ROUTE_VRAM_TO_DRAM: 0,
                ROUTE_SSD_TO_DRAM: 0,
                ROUTE_DRAM_TO_SSD: ms(2),
            },
        )
        for key in ("cpu_compute_slots", "ssd_io_slots", "dram_alignment_bytes"):
            self.assertIn(key, document["limits"])

    def test_an_infeasible_search_reports_absent_rather_than_zero(self) -> None:
        """No plan was replayed, so there is no peak and no service time.

        ``0`` here would be a claim about a run, and the one thing this document
        must not do is answer a question it never asked. ``target_locations`` is
        the exception and stays, because it is the question rather than an
        answer to it -- and a reader looking at an infeasible result needs to see
        which output could not be delivered where.
        """
        from tensor_mapping.artifacts import m1_metrics

        # The acceptance table's own infeasible row: one byte less than the spill
        # case needs, so no plan leaves the initial state at all.
        with m1_scenario(
            "m1-spill",
            "m1-g-spill-dram96",
            lambda workload, scenario: set_capacity(scenario, dram=95),
        ) as scenario:
            result = search(scenario)
            self.assertEqual(result.status, STATUS_INFEASIBLE)
            self.assertIsNone(result.evaluation)

            metrics = m1_metrics(scenario, result.evaluation)
            self.assertIsNone(metrics["peak_dram_bytes"])
            self.assertIsNone(metrics["d2h_bytes"])
            self.assertIsNone(metrics["ssd_read_bytes"])
            self.assertIsNone(metrics["ssd_write_bytes"])
            self.assertIsNone(metrics["service_ns"])
            self.assertEqual(metrics["target_locations"], {"y": LOC_DRAM})

    def test_a_0_2_stats_document_without_its_metrics_is_refused(self) -> None:
        """A silently ``null`` peak is a valid document that is wrong.

        The writer is the only place that can tell "this run measured nothing"
        from "the caller forgot", and it is the place that has the scenario in
        hand, so it refuses rather than emitting the empty field.
        """
        from tensor_mapping.artifacts import stats_document

        scenario = load_and_validate(EXAMPLES / "m1-h-ssd-cpu.json")
        with self.assertRaises(ValueError) as caught:
            stats_document(
                mode="search",
                scenario=scenario,
                status=STATUS_INFEASIBLE,
                reason="initial_capacity_exceeded",
                makespan_ns=None,
                peak_vram_bytes=None,
                h2d_bytes=None,
                action_count=None,
                declared_action_count=0,
                limits={},
            )
        self.assertIn("m1_metrics", str(caught.exception))

    def test_a_0_1_document_gains_no_0_2_key(self) -> None:
        """The compatibility claim, stated as the absence it actually is.

        Checked as a whole-document key set rather than field by field: "0.1
        artefacts stay byte-identical" is a claim about every key, and the way it
        would break is a new field appearing in one of the four documents.
        """
        from tensor_mapping import cli

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw:
            out = Path(raw) / "out"
            code, _, err = self.run_cli(
                "search",
                "--scenario",
                str(EXAMPLES / "chain-cap160.json"),
                "--out",
                str(out),
            )
            self.assertEqual(code, cli.EXIT_OK, err)

            new_keys = {
                "peak_dram_bytes",
                "d2h_bytes",
                "ssd_read_bytes",
                "ssd_write_bytes",
                "target_locations",
                "service_ns",
            }
            for name in (cli.STATS_FILE, cli.EVENTS_FILE, cli.STATES_FILE, cli.MAPPING_FILE):
                document = self.read(out, name)
                with self.subTest(name=name):
                    self.assertEqual(document["schema_version"], SCHEMA_VERSION)
                    self.assertEqual(new_keys & set(document), set())
            event = self.read(out, cli.EVENTS_FILE)["events"][0]
            self.assertEqual(
                set(event),
                {
                    "action_index",
                    "kind",
                    "target_id",
                    "resource",
                    "t_ns",
                    "start_ns",
                    "end_ns",
                    "bytes_transferred",
                    "workspace_bytes",
                    "bytes_released",
                },
            )
            self.assertEqual(
                set(self.read(out, cli.STATES_FILE)["states"][0]),
                {
                    "action_index",
                    "t_ns",
                    "used_vram_bytes",
                    "op_status",
                    "copy_status",
                },
            )


if __name__ == "__main__":
    unittest.main()
