import unittest
from copy import deepcopy

from profiling.state_response.schedule import make_jobs, timed_segments, validate_schedule


class StateResponseScheduleTests(unittest.TestCase):
    def test_declared_asymmetric_profile_holds_do_not_change_heldout_schedule(self):
        baseline = make_jobs(seed=20261031)
        changed = make_jobs(seed=20261031, profile_state_holds_ms=[600, 1500, 3000, 3000])
        mapping = {(0, 0): 600, (0, 1): 1500, (1, 0): 3000, (1, 1): 3000}
        for original, new in zip(baseline, changed):
            if new["split"] == "heldout":
                self.assertEqual(original, new)
            else:
                self.assertEqual(new["schedule"]["duration_ns"], 8100 * 1000000)
                for phase in new["schedule"]["phases"]:
                    vector = tuple(phase["state"][key] for key in ("cpu_replay", "gpu_replay"))
                    self.assertEqual(phase["end_ns"] - phase["begin_ns"], mapping[vector] * 1000000)
        for bad in ([1, 200, 200, 200], [200, 200, 200], [True, 200, 200, 200]):
            with self.assertRaises(ValueError):
                make_jobs(profile_state_holds_ms=bad)

    def test_balanced_independent_cell_support_and_different_temporal_holdout(self):
        jobs = make_jobs()
        self.assertEqual(len(jobs), 24)
        supports = {}
        for split in ("profile", "heldout"):
            by_cell = {}
            for job in jobs:
                if job["split"] == split:
                    for cell in {tuple(p["state"].values()) for p in job["schedule"]["phases"]}:
                        by_cell.setdefault(cell, set()).add(job["id"])
            self.assertEqual(len(by_cell), 16)
            self.assertTrue(all(len(runs) == 3 for runs in by_cell.values()))
            supports[split] = set(by_cell)
        self.assertEqual(supports["profile"], supports["heldout"])
        self.assertTrue({j["numeric_seed"] for j in jobs if j["split"] == "profile"}.isdisjoint(
            {j["numeric_seed"] for j in jobs if j["split"] == "heldout"}))
        self.assertEqual(jobs, make_jobs())
        self.assertEqual({p["end_ns"] - p["begin_ns"] for p in jobs[0]["schedule"]["phases"]}, {1200000000})
        self.assertNotEqual({p["end_ns"] - p["begin_ns"] for p in jobs[-1]["schedule"]["phases"]}, {1200000000})

    def test_exact_external_clock_binding_and_reject_midrun_memory_change(self):
        schedule = make_jobs()[0]["schedule"]
        segments = timed_segments(schedule, start_tick=123, frequency_hz=10000000)
        self.assertEqual(segments[0]["begin_tick"], 123)
        self.assertEqual(segments[0]["end_tick"], segments[1]["begin_tick"])
        bad = deepcopy(schedule)
        bad["phases"][1]["state"]["host_memory_bytes"] += 1
        with self.assertRaises(ValueError):
            validate_schedule(bad)
        for bad_value in (True, 1.5, -1):
            bad = deepcopy(schedule)
            bad["phases"][0]["state"]["cpu_replay"] = bad_value
            with self.assertRaises(ValueError):
                validate_schedule(bad)


if __name__ == "__main__":
    unittest.main()
