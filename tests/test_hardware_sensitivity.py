"""Scientific protocol checks, independent of timing-provider internals."""
from pathlib import Path
import tempfile
import unittest

from experiments.hardware_sensitivity import (
    BASELINE, GROUPS, Point, best_candidate, candidate_catalogue, derive_tables,
    experiment_points, run_candidate,
)


class HardwareSensitivityTests(unittest.TestCase):
    def test_equal_budget_distinct_candidates_and_no_nested_joint_claim(self):
        for members in GROUPS.values():
            self.assertEqual(len(members), 4)
            self.assertEqual(len(set(members)), 4)
            self.assertIn(BASELINE, members)
        self.assertFalse(set(GROUPS["execution_only"]) <= set(GROUPS["joint"]))
        self.assertEqual(len(candidate_catalogue(True)), 16)

    def test_each_sweep_changes_only_its_declared_axis(self):
        points, grids = experiment_points()
        self.assertEqual(points[0], Point("base"))
        self.assertEqual(len(points), len({point.id for point in points}))
        field = {"cpu_compute": "cpu_factor", "gpu_compute": "gpu_factor",
                 "transfer_bandwidth": "link_factor", "vram_capacity": "vram_bytes"}
        for axis, grid in grids.items():
            self.assertEqual([value for value, _ in grid], sorted(value for value, _ in grid))
            for _, point in grid:
                for name in field.values():
                    if name != field[axis]:
                        self.assertEqual(getattr(point, name), getattr(Point("base"), name))

    def test_invalid_partial_time_cannot_win_and_ties_are_stable(self):
        rows = [dict(candidate_id="e0-s0", success=False, makespan_ns=1),
                dict(candidate_id="e2-s0", success=True, makespan_ns=100),
                dict(candidate_id="e1-s0", success=True, makespan_ns=100)]
        self.assertEqual(best_candidate(rows, GROUPS["execution_only"])["candidate_id"], "e1-s0")
        self.assertIsNone(best_candidate(rows[:1], GROUPS["execution_only"]))

    def test_frozen_and_reselected_policies_are_different_quantities(self):
        records = []
        ids = candidate_catalogue()
        for point in ("base", "faster"):
            for candidate in ids:
                # At centre baseline is fastest. After hardware change, e1-s0
                # becomes faster. Frozen means baseline, not the new winner.
                duration = 100 if point == "base" and candidate == BASELINE else 200
                if point == "faster" and candidate == "e1-s0":
                    duration = 50
                records.append(dict(model_kind="dense", point_id=point,
                    candidate_id=candidate, success=True, makespan_ns=duration,
                    makespan_ms=duration / 1e6))
        selections, _, _ = derive_tables(records, {"cpu_compute": [(1, Point("base")), (2, Point("faster"))]})
        target = next(row for row in selections if row["point_id"] == "faster" and row["group"] == "execution_only")
        self.assertEqual(target["selected_candidate"], "e1-s0")
        self.assertEqual(target["frozen_at_centre_candidate"], BASELINE)
        self.assertEqual(target["frozen_to_reselected_latency_ratio"], 4)

    def test_elasticity_keeps_infeasible_intervals_null(self):
        records = []
        for point, scale in (("base", 1), ("twice", 0.5), ("impossible", None)):
            for candidate in candidate_catalogue():
                records.append(dict(model_kind="dense", point_id=point,
                    candidate_id=candidate, success=scale is not None,
                    makespan_ns=None if scale is None else int(100 * scale),
                    makespan_ms=None if scale is None else 0.1 * scale))
        _, elasticities, _ = derive_tables(records, {"gpu_compute": [(1, Point("base")), (2, Point("twice")), (4, Point("impossible"))]})
        self.assertTrue(all(row["log_latency_elasticity"] == -1 for row in elasticities if row["parameter_low"] == 1))
        self.assertTrue(all(row["log_latency_elasticity"] is None for row in elasticities if row["parameter_low"] == 2))

    def test_public_runtime_success_and_impossible_vram_stay_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            # CPU placements do not silently consume the impossible GPU.
            cpu = run_candidate("dense", Point("cpu", vram_bytes=1), "e1-s0", Path(folder))
            gpu = run_candidate("dense", Point("gpu", vram_bytes=1), "e0-s0", Path(folder))
        self.assertTrue(cpu["success"], cpu)
        self.assertEqual(cpu["h2d_bytes"], 0)
        self.assertFalse(cpu["calibrated"])
        self.assertFalse(cpu["hardware_measurement"])
        self.assertFalse(gpu["success"], gpu)
        self.assertNotEqual(gpu["status"], "valid")

    def test_declared_digests_and_events_are_portable_across_fresh_projects(self):
        with tempfile.TemporaryDirectory() as folder:
            first = run_candidate("dense", Point("base"), "e1-s0", Path(folder))
            second = run_candidate("dense", Point("base"), "e1-s0", Path(folder))
        for name in ("strategy_sha256", "declared_inputs_sha256", "event_sha256"):
            self.assertEqual(first[name], second[name])
        self.assertIn("compiled_ir_audit_sha256", first)


if __name__ == "__main__":
    unittest.main()
