import json
import tempfile
import unittest
from pathlib import Path

from profiling.real_moe_validation import ensemble


def write_cohort(root, results, mean_APE=0.03, worst_APE=0.07, phases=None):
    formal = Path(root) / "model" / "formal"
    jobs = formal / "jobs" / "hybrid-heldout-r0"
    jobs.mkdir(parents=True)
    (jobs / "result.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    conditions = []
    for placement, request, values in phases or []:
        conditions.append({"placement": placement, "request_id": request,
                           "APE": dict(total=values[0], prefill=values[1],
                                       decode_with_sampling=values[2]),
                           "passed": max(values) <= 0.10})
    (formal / "CONDITION_MEAN_EVALUATION.json").write_text(json.dumps({
        "mean_APE": mean_APE, "worst_APE": worst_APE, "passed_conditions": 8,
        "complete_endpoint_conditions": 12, "conditions": conditions}), encoding="utf-8")
    return Path(root)


def result(request, ticks):
    return {"id": request,
            "forward_inventory": [{"phase": "prefill", "begin_tick": 0, "end_tick": ticks},
                                  {"phase": "decode", "begin_tick": ticks, "end_tick": ticks * 2}]}


class FirstForwardTest(unittest.TestCase):
    def test_ticks_are_hundreds_of_nanoseconds_and_only_prefill_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = write_cohort(folder, [result("a", 220_000), result("b", 430_000)])
            rows = ensemble.first_forward_ms(root)
            self.assertEqual([row["ms"] for row in rows], [22.0, 43.0])
            self.assertTrue(all(row["job"] == "hybrid-heldout-r0" for row in rows))

    def test_a_request_without_a_prefill_forward_is_counted_not_zeroed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = write_cohort(folder, [result("a", 220_000), {"id": "b", "forward_inventory": []}])
            rows = ensemble.first_forward_ms(root)
            values = [row["ms"] for row in rows if row["ms"] is not None]
            self.assertEqual(values, [22.0])
            self.assertEqual(rows[-1]["missing"], 1)


class BoundaryTest(unittest.TestCase):
    def test_the_boundary_sits_in_the_largest_ratio_gap(self):
        value, why = ensemble.boundary([20.9, 22.5, 21.3, 41.3, 33.9, 21.4])
        self.assertAlmostEqual(value, 28.2, places=6)          # gap 22.5 -> 33.9 is the widest
        self.assertIn("largest ratio gap", why)

    def test_two_durations_are_not_enough_to_declare_a_boundary(self):
        with self.assertRaises(ensemble.Unreadable):
            ensemble.boundary([20.0, 40.0])

    def test_occupancy_counts_strictly_above_the_boundary(self):
        where = ensemble.occupancy([20.0, 31.0, 31.1, 40.0], 31.0)
        self.assertEqual((where["slow"], where["requests"]), (2, 4))
        self.assertAlmostEqual(where["fraction"], 0.5)


class AggregateTest(unittest.TestCase):
    def entry(self, root, mean, status="counted", passes=8):
        return {"root": str(root), "status": status, "mean_APE": mean, "worst_APE": mean * 2,
                "passed": passes, "conditions": 12,
                "phase_APEs": [{"placement": "cpu", "request_id": "a",
                                "APE": {"total": mean, "prefill": mean * 2,
                                        "decode_with_sampling": mean / 2}}],
                "first_forward_ms": [{"ms": 21.0}, {"ms": 41.0}]}

    def test_the_ensemble_is_the_mean_of_the_cohort_means_with_its_spread(self):
        entries = [self.entry("a", 0.02), self.entry("b", 0.06), self.entry("c", 0.90, "excluded")]
        result = ensemble.aggregate(entries, attempts=3)
        self.assertEqual(result["cohorts"], 2)
        self.assertEqual(result["excluded"], 1)
        self.assertAlmostEqual(result["mean_of_means"], 0.04)
        self.assertAlmostEqual(result["sd_of_means"], 0.028284, places=6)
        self.assertEqual(result["pass_counts"], [8, 8])

    def test_the_pooled_phase_apes_carry_a_tail_the_mean_would_hide(self):
        entries = [self.entry("a", 0.02), self.entry("b", 0.20)]
        result = ensemble.aggregate(entries, attempts=2)
        pooled = result["phase_APE"]
        self.assertEqual(pooled["n"], 6)
        self.assertAlmostEqual(pooled["max"], 0.40, places=6)

    def test_a_declared_boundary_is_used_and_beyond_it_nothing_is_guessed(self):
        entries = [self.entry("a", 0.02)]
        with_declared = ensemble.aggregate(entries, attempts=1, declared_boundary_ms=31.0)
        self.assertEqual(with_declared["state"]["boundary_ms"], 31.0)
        self.assertEqual(with_declared["state"]["slow"], 1)
        self.assertIn("declared in the profile", with_declared["state"]["source"])

    def test_no_counted_cohort_is_an_error_not_an_empty_ensemble(self):
        with self.assertRaises(ensemble.Unreadable):
            ensemble.aggregate([self.entry("a", 0.02, "excluded")], attempts=1)


class EvaluationTest(unittest.TestCase):
    def test_a_cohort_that_never_evaluated_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ensemble.Unreadable):
                ensemble.evaluation(Path(folder))


if __name__ == "__main__":
    unittest.main()
