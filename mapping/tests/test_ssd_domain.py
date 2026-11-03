"""A declared SSD budget turns the disk into a ledgered storage domain.

The historical machine gives the SSD no capacity: it is unbounded scene content
plus runtime snapshots, and ``M1_DESIGN.md`` §3 says so. This suite is the other
machine -- an edge device whose application storage budget is finite -- and the
rule it pins is that **the budget alone** is the switch. Declaring one gives the
SSD a capacity, an alignment, a reserve, an eviction and a peak; not declaring
one must leave every existing artefact byte-identical, which is why the first
class is about what *does not* change.
"""
from __future__ import annotations

import dataclasses
import unittest

from llm_infer_model.tensor.spec import Architecture
from tensor_mapping import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    InitialCapacityExceeded,
    Scenario,
    evaluate_mapping,
    initial_state,
    search,
)
from tensor_mapping.artifacts import m1_metrics
from tensor_mapping.engine import (
    ACTION_EVICT,
    ACTION_READ_SSD,
    ACTION_WRITE_SSD,
    CODE_CAPACITY_EXCEEDED,
    CODE_ILLEGAL_ACTION,
    STATUS_INCOMPLETE_MAPPING,
    used_bytes,
)
from tensor_mapping.spec import (
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    InvalidInput,
    scenario_fingerprint,
)
from tests.support import M1Replay, m1_scenario


def with_ssd(scenario: Scenario, capacity: int, *, alignment: int | None = None,
             reserved: int = 0) -> Scenario:
    return dataclasses.replace(
        scenario,
        architecture=dataclasses.replace(
            scenario.architecture,
            ssd_capacity_bytes=capacity,
            ssd_alignment_bytes=alignment,
            ssd_runtime_reserved_bytes=reserved,
        ),
    )


class UndeclaredSsdIsUnchanged(unittest.TestCase):
    """No budget, no domain -- and no artefact churn."""

    def test_the_default_machine_has_two_storage_domains(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            self.assertFalse(scenario.architecture.is_ssd_domain)
            self.assertEqual(scenario.architecture.storage_domains, (LOC_DRAM, LOC_VRAM))
            self.assertIsNone(scenario.architecture.capacity_of(LOC_SSD))
            self.assertEqual(scenario.architecture.alignment_of(LOC_SSD), 1)

    def test_evicting_from_an_undeclared_ssd_is_refused(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            error = M1Replay(scenario).refuse(f"{ACTION_EVICT} W1 ssd")
            self.assertEqual(error.code, CODE_ILLEGAL_ACTION)

    def test_spelling_the_default_explicitly_does_not_change_the_fingerprint(self):
        """``"ssd_capacity_bytes": null`` names the same machine as omitting it."""

        def spell_defaults(workload: dict, scenario: dict) -> None:
            scenario["architecture"].update(
                ssd_capacity_bytes=None, ssd_runtime_reserved_bytes=0,
                ssd_alignment_bytes=None)

        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as plain:
            baseline = scenario_fingerprint(plain)
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128", spell_defaults) as spelled:
            self.assertEqual(scenario_fingerprint(spelled), baseline)


class CapacityIsEnforced(unittest.TestCase):
    """The budget is charged to every copy on the disk, initial or written."""

    def test_the_initial_layout_is_charged(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            budgeted = with_ssd(scenario, 128)
            self.assertEqual(used_bytes(budgeted, initial_state(budgeted), LOC_SSD), 128)

    def test_one_byte_short_is_an_impossible_starting_point(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            with self.assertRaises(InitialCapacityExceeded):
                initial_state(with_ssd(scenario, 127))

    def test_a_write_that_does_not_fit_is_refused(self):
        """W 64 still on the disk plus the spilled h 16 does not fit in 64."""
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as scenario:
            replay = M1Replay(with_ssd(scenario, 64))
            replay.do("COMPUTE c1 cpu", "ADVANCE", "EVICT x dram", f"{ACTION_READ_SSD} W", "ADVANCE")
            error = replay.refuse(f"{ACTION_WRITE_SSD} h")
            self.assertEqual(error.code, CODE_CAPACITY_EXCEEDED)

    def test_evicting_the_weight_makes_room_for_the_spill(self):
        """The budget binds the *plan*, not the machine: 64 is enough if W goes."""
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as scenario:
            result = search(with_ssd(scenario, 64))
            self.assertEqual(result.status, STATUS_OPTIMAL)
            self.assertEqual(result.evaluation.peak_ssd_bytes, 64)
            self.assertEqual(result.evaluation.ssd_write_bytes, 16)

    def test_a_write_that_fits_is_found(self):
        with m1_scenario("m1-spill", "m1-g-spill-dram96") as scenario:
            result = search(with_ssd(scenario, 80))
            self.assertEqual(result.status, STATUS_OPTIMAL)
            self.assertEqual(result.evaluation.peak_ssd_bytes, 80)

    def test_eviction_frees_the_budget(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            replay = M1Replay(with_ssd(scenario, 128))
            self.assertEqual(replay.used_bytes(LOC_SSD), 128)
            replay.do(f"{ACTION_READ_SSD} W1", "ADVANCE")
            replay.do(f"{ACTION_EVICT} W1 ssd")
            self.assertEqual(replay.used_bytes(LOC_SSD), 64)

    def test_the_alignment_is_applied_to_disk_copies(self):
        # 64-byte tensors on a 4096-byte disk: each copy occupies one sector.
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            replay = M1Replay(with_ssd(scenario, 8192, alignment=4096))
            self.assertEqual(replay.used_bytes(LOC_SSD), 8192)


class PeakAndArtefacts(unittest.TestCase):
    """A budgeted run reports where the disk peaked; an unbudgeted one does not."""

    def test_peak_ssd_is_reported_for_a_budgeted_run(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            budgeted = with_ssd(scenario, 128)
            result = search(budgeted)
            self.assertEqual(result.status, STATUS_OPTIMAL)
            self.assertEqual(result.evaluation.peak_ssd_bytes, 128)
            self.assertEqual(m1_metrics(budgeted, result.evaluation)["peak_ssd_bytes"], 128)

    def test_peak_ssd_is_absent_without_a_budget(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            result = search(scenario)
            self.assertEqual(result.evaluation.peak_ssd_bytes, 0)
            self.assertNotIn("peak_ssd_bytes", m1_metrics(scenario, result.evaluation))

    def test_an_unfinished_replay_still_reports_the_peak(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            result = evaluate_mapping(with_ssd(scenario, 128), ())
            self.assertEqual(result.status, STATUS_INCOMPLETE_MAPPING)
            self.assertEqual(result.peak_ssd_bytes, 128)


class Parsing(unittest.TestCase):
    """Budget, reserve and alignment are one declaration, not three."""

    def test_an_alignment_without_a_budget_is_refused(self):
        # Programmatic construction and the JSON parser are held to one rule.
        with self.assertRaises(InvalidInput):
            Architecture(vram_capacity_bytes=16, ssd_alignment_bytes=4096)
        with self.assertRaises(InvalidInput):
            Architecture(vram_capacity_bytes=16, ssd_runtime_reserved_bytes=8)

    def test_a_reserve_above_the_budget_is_refused(self):
        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128") as scenario:
            with self.assertRaises(InvalidInput):
                with_ssd(scenario, 64, reserved=65)

    def test_the_json_parser_accepts_the_declaration(self):
        def declare(workload: dict, scenario: dict) -> None:
            scenario["architecture"].update(
                ssd_capacity_bytes=256, ssd_runtime_reserved_bytes=16,
                ssd_alignment_bytes=4096)

        with m1_scenario("m1-two-weights-ssd", "m1-e-dram128", declare) as scenario:
            self.assertTrue(scenario.architecture.is_ssd_domain)
            self.assertEqual(scenario.architecture.capacity_of(LOC_SSD), 256)
            self.assertEqual(scenario.architecture.reserved_of(LOC_SSD), 16)
            self.assertEqual(scenario.architecture.alignment_of(LOC_SSD), 4096)
            self.assertEqual(scenario.architecture.storage_domains,
                             (LOC_DRAM, LOC_VRAM, LOC_SSD))


if __name__ == "__main__":
    unittest.main()
