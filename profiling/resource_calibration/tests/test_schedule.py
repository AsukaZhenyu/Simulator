"""Resource trajectory controls and interval annotation, no hardware."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from resource_calibration import schedule


class ScheduleTests(unittest.TestCase):
    def test_full_factorial_independent_shuffles_and_reproducibility(self):
        rows = schedule.make_schedules()
        self.assertEqual(rows, schedule.make_schedules())
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["split"] for r in rows}, {"train", "heldout"})
        controls = []
        for row in rows:
            self.assertEqual(len(row["phases"]), 16)
            values = {tuple(p["resources"][key] for key in schedule.RESOURCE_KEYS) for p in row["phases"]}
            self.assertEqual(len(values), 16)
            controls.append(values)
        self.assertTrue(all(value == controls[0] for value in controls))
        self.assertNotEqual(rows[0]["phases"], rows[3]["phases"])
        self.assertTrue({r["seed"] for r in rows[:3]}.isdisjoint(r["seed"] for r in rows[3:]))

    def test_smoke_keeps_all_sixteen_combinations_in_each_split(self):
        rows = schedule.make_schedules(smoke=True, phase_duration_ms=300)
        self.assertEqual([r["split"] for r in rows], ["train", "heldout"])
        self.assertTrue(all(len(row["phases"]) == 16 for row in rows))

    def test_controls_are_requests_never_utilization_fractions(self):
        resources = schedule.make_schedules()[0]["phases"][0]["resources"]
        for key in ("cpu_replay", "gpu_replay"):
            for value in (0, 1, .5, "on", None):
                broken = {**resources, key: value}
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    schedule.validate_resources(broken)

    def test_invalid_bounds_unknown_fields_and_no_implicit_coercion(self):
        for kwargs in ({"phase_duration_ms": 199}, {"phase_duration_ms": True}, {"phase_duration_ms": 300.0},
                       {"host_memory_bytes": -1}, {"device_memory_bytes": 2 << 30}, {"repeats": 0},
                       {"seed": -1}, {"smoke": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                schedule.make_schedules(**kwargs)
        original = schedule.make_schedules()[0]
        for mutate in (lambda x: x.update(extra=True), lambda x: x.update(id="../escape"),
                       lambda x: x["phases"][0].update(index=True), lambda x: x["phases"][0].update(extra=True),
                       lambda x: x.update(phases=[]), lambda x: x["phases"][0]["resources"].update(host_memory_bytes=None)):
            value = deepcopy(original)
            mutate(value)
            with self.assertRaises(ValueError):
                schedule.validate_schedule(value)

    def test_validation_copies_mutable_inputs(self):
        original = schedule.make_schedules()[0]
        validated = schedule.validate_schedule(original)
        validated["phases"][0]["resources"]["host_memory_bytes"] = -1
        self.assertGreaterEqual(original["phases"][0]["resources"]["host_memory_bytes"], 0)

    @staticmethod
    def events():
        controls = {"cpu_replay": False, "gpu_replay": True, "host_memory_bytes": 100, "device_memory_bytes": 200}
        return [{"phase_index": 0, "steady_begin_tick": 10, "steady_end_tick": 20, "requested_resources": controls},
                {"phase_index": 1, "steady_begin_tick": 30, "steady_end_tick": 40, "requested_resources": {**controls, "cpu_replay": True}}]

    def test_exact_containment_and_transition_rows_are_retained_unknown(self):
        for begin, end, index in ((10, 20, 0), (11, 19, 0), (30, 40, 1)):
            row = schedule.phase_for_interval(self.events(), begin, end)
            self.assertEqual(row["phase_index"], index)
            self.assertEqual(row["kind"], "steady")
        for begin, end in ((0, 9), (9, 11), (19, 31), (20, 30), (39, 41), (41, 42)):
            row = schedule.phase_for_interval(self.events(), begin, end)
            self.assertEqual(row["kind"], "transition_or_outside")
            self.assertIsNone(row["offered_resources"])

    def test_invalid_or_overlapping_time_intervals_rejected(self):
        for begin, end in ((1, 1), (-1, 2), (1, True)):
            with self.assertRaises(ValueError):
                schedule.phase_for_interval(self.events(), begin, end)
        events = self.events()
        events[1]["steady_begin_tick"] = 19
        with self.assertRaises(ValueError):
            schedule.phase_for_interval(events, 21, 22)


if __name__ == "__main__":
    unittest.main()
