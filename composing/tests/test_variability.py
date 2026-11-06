"""The band is described, never corrected: no sample is dropped or repaired."""
import unittest

from composing import schema, variability
from composing.compare import compare


def ramp(centre, jitter, n):
    """A deterministic spread: no random, so a failure is reproducible."""
    return [centre + (i % (2 * jitter + 1)) - jitter for i in range(n)]


def two_modes(fast=1085, slow=1270, n_fast=5000, n_slow=3000, jitter=5):
    return ramp(fast, jitter, n_fast) + ramp(slow, jitter, n_slow)


class SummaryTests(unittest.TestCase):
    def test_the_span_of_a_row_is_reported_without_changing_it(self):
        summary = variability.summarise([1000, 900, 1100, 1000, 1000])
        self.assertEqual(summary["n"], 5)
        self.assertEqual((summary["min_ns"], summary["max_ns"]), (900, 1100))
        self.assertEqual(summary["median_ns"], 1000)
        self.assertAlmostEqual(summary["spread"], 1100 / 900 - 1)

    def test_one_sample_is_a_band_of_zero(self):
        summary = variability.summarise([1000])
        self.assertEqual(summary["spread"], 0.0)
        self.assertEqual(summary["cv"], 0.0)

    def test_a_clock_that_is_not_positive_is_refused(self):
        for bad in ([], [0], [-1], [True], ["1000"]):
            with self.assertRaises(ValueError):
                variability.summarise(bad)


class ModeSplitTests(unittest.TestCase):
    def test_two_centres_are_found_where_they_are(self):
        modes = variability.split_modes(two_modes())
        self.assertFalse(modes["single_mode"])
        self.assertAlmostEqual(modes["fast_ns"], 1085, delta=5)
        self.assertAlmostEqual(modes["slow_ns"], 1270, delta=5)
        self.assertAlmostEqual(modes["ratio"], 1270 / 1085, delta=0.01)
        self.assertAlmostEqual(modes["slow_fraction"], 3000 / 8000, delta=0.02)

    def test_one_population_is_not_split_into_two(self):
        modes = variability.split_modes(ramp(1092, 10, 4000))
        self.assertTrue(modes["single_mode"])
        self.assertIn("apart", modes["reason"])

    def test_a_tail_is_not_a_second_state(self):
        modes = variability.split_modes(ramp(1092, 5, 4000) + ramp(1400, 5, 20))
        self.assertTrue(modes["single_mode"])
        self.assertIn("holds", modes["reason"])

    def test_the_11_04_shape_is_one_mode_not_two(self):
        # The archived 2026-11-04 run: 94.8% of 8000 samples between 1085 and
        # 1100, the rest a tail out to 1400.  k-means alone calls that two
        # centres (1092 / 1136, 4% apart); it is one population.
        samples = ramp(1092, 6, 7600) + [1200 + 50 * (i % 5) for i in range(400)]
        modes = variability.split_modes(samples)
        self.assertTrue(modes["single_mode"])
        self.assertIn("holds", modes["reason"])

    def test_a_handful_of_repetitions_cannot_claim_a_mode(self):
        modes = variability.split_modes([1000, 1010, 990, 1005, 995])
        self.assertIsNone(modes["single_mode"])
        self.assertIn("needs", modes["reason"])

    def test_the_samples_that_disagree_are_kept(self):
        samples = two_modes()
        described = variability.describe(samples)
        self.assertEqual(described["summary"]["n"], len(samples))
        self.assertFalse(described["modes"]["single_mode"])


class ProbeDocumentTests(unittest.TestCase):
    def test_an_archived_run_and_a_bare_one_read_the_same(self):
        wall = [1000, 1010, 1020]
        self.assertEqual(variability.wall_ns_of({"wall_ns": wall}), wall)
        self.assertEqual(variability.wall_ns_of({"document": {"wall_ns": wall}}), wall)

    def test_a_document_without_samples_is_refused(self):
        with self.assertRaises(ValueError):
            variability.wall_ns_of({"document": {}})


class SessionTests(unittest.TestCase):
    def test_rows_wider_than_the_band_are_named(self):
        observations = {"a/tg128": variability.summarise([1000, 1001, 1002]),
                        "b/tg512": variability.summarise([1000, 1200, 1005])}
        session = variability.session_summary(observations)
        self.assertEqual(session["rows"], 2)
        self.assertEqual(session["outside_band"], ["b/tg512"])
        self.assertEqual(session["widest_row"], "b/tg512")

    def test_no_rows_says_nothing_rather_than_zero(self):
        session = variability.session_summary({})
        self.assertIsNone(session["widest_spread"])
        self.assertEqual(session["rows"], 0)


class TableStateTests(unittest.TestCase):
    """A table is priced in one state; the clock it is compared with is in another."""

    def test_a_table_repriced_in_the_same_state_is_one_comparison(self):
        table = {"prefill/B1/q256/hist0": 163_840_000, "decode/B1/q1/hist1024": 1_100_000}
        today = {"prefill/B1/q256/hist0": 162_000_000, "decode/B1/q1/hist1024": 1_090_000}
        state = variability.table_state(table, today)
        self.assertEqual(state["verdict"], "same_band")
        self.assertEqual(state["outside_band"], [])
        self.assertAlmostEqual(state["median_ratio"], (162_000_000 / 163_840_000
                                                       + 1_090_000 / 1_100_000) / 2)

    def test_the_11_05_step_is_caught_and_named(self):
        # 2026-11-05: the probe's joint block for qwen3-0.6b prefill q256 fell from
        # 163.84 ms (the frozen table) to 151.14 ms with the same binary.
        state = variability.table_state({"prefill/B1/q256/hist0": 163_840_000},
                                        {"prefill/B1/q256/hist0": 151_140_000})
        self.assertEqual(state["verdict"], "different_band")
        self.assertEqual(state["outside_band"], ["prefill/B1/q256/hist0"])
        self.assertAlmostEqual(state["widest_ratio"], 151_140_000 / 163_840_000)

    def test_a_shape_measured_only_one_way_is_reported_not_ignored(self):
        state = variability.table_state({"prefill/B1/q256/hist0": 1000,
                                         "decode/B1/q1/hist1024": 500},
                                        {"prefill/B1/q256/hist0": 1000})
        self.assertEqual(state["verdict"], "same_band")
        self.assertEqual(state["uncompared"]["table_only"], ["decode/B1/q1/hist1024"])
        self.assertEqual(state["uncompared"]["sensor_only"], [])

    def test_nothing_in_common_is_refused_rather_than_called_the_same(self):
        with self.assertRaises(ValueError):
            variability.table_state({"prefill/B1/q256/hist0": 1000},
                                    {"decode/B1/q1/hist1024": 500})

    def test_a_sensor_that_is_not_a_wall_time_is_refused(self):
        for bad in ({"prefill/B1/q256/hist0": 0}, {"prefill/B1/q256/hist0": "1000"},
                    {"prefill/B1/q256/hist0": True}):
            with self.assertRaises(ValueError):
                variability.table_state({"prefill/B1/q256/hist0": 1000}, bad)

    def test_a_zero_tolerance_would_call_every_table_different(self):
        with self.assertRaises(ValueError):
            variability.table_state({"a": 1000}, {"a": 1000}, tolerance=0)


class BlockRecordTests(unittest.TestCase):
    """A joint block record, whichever of the three archive layouts it is in."""

    def test_a_campaign_entry_wraps_the_record_one_level_deeper(self):
        entry = {"label": "prefill/block", "command": [], "seconds": 8.6,
                 "record": {"stage": "block", "q": 256, "history": 0,
                            "wall_ns": [1000, 1010, 1020]}}
        found = variability.block_records(entry)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]["q"], found[0]["history"]), (256, 0))
        self.assertEqual(found[0]["median_ns"], 1010)

    def test_another_stage_is_not_read_as_the_block(self):
        with self.assertRaises(ValueError):
            variability.block_records({"stage": "attention", "q": 256, "history": 0,
                                       "wall_ns": [1000]})

    def test_a_map_of_runs_yields_only_the_block_records(self):
        document = {"0/prefill/block": {"record": {"stage": "block", "q": 256, "history": 0,
                                                   "wall_ns": [1000, 1020]}},
                    "0/prefill/attention": {"record": {"stage": "attention", "q": 256,
                                                       "history": 0, "wall_ns": [300, 310]}}}
        found = variability.block_records(document)
        self.assertEqual([row["q"] for row in found], [256])
        self.assertEqual(found[0]["median_ns"], 1010)



def prediction(total_ns=1000):
    shape = {"total_ns": total_ns, "block_sum_ns": 900,
             "embedding": {"median_ns": 50}, "head": {"median_ns": 50}}
    return {"schema_version": schema.PREDICTION,
            "models": {"m": {"kv_dtype": "f16", "layers": 4, "shapes": {"tg128": shape}}}}


def measurement(samples):
    return {"llama_bench_sha256": "x", "threads": 2, "cpu_mask": [2, 4],
            "bench_protocol": {}, "decode_grouping": "one_process_per_shape",
            "warmup": {"discarded": None},
            "prediction": {"path": "predicted.json", "sha256": "y"},
            "models": {"m": {"tg128": {"avg_ns": round(sum(samples) / len(samples)),
                                       "samples_ns": samples,
                                       "llama_bench": {"type_k": "f16"}}}}}


class ComparisonTests(unittest.TestCase):
    def test_a_comparison_carries_the_band_it_was_read_inside(self):
        document = compare(prediction(), measurement([990, 1000, 1010]))
        row = document["models"]["m"]["tg128"]
        self.assertEqual(row["observation"]["n"], 3)
        self.assertEqual(row["observed_cv"], row["observation"]["cv"])
        self.assertEqual(document["session"]["rows"], 1)
        schema.validate_comparison(document)

    def test_the_point_estimate_still_decides_the_error(self):
        document = compare(prediction(), measurement([1100, 1000, 1000]))
        row = document["models"]["m"]["tg128"]
        self.assertEqual(row["observed_ns"], 1033)
        self.assertAlmostEqual(row["signed_relative_error"], 0.033, places=6)

    def test_a_wide_row_is_named_in_the_session(self):
        document = compare(prediction(), measurement([1000, 1300, 1000]))
        self.assertEqual(document["session"]["outside_band"], ["m/tg128"])

    def test_inside_the_band_is_reported_beside_the_error_not_instead_of_it(self):
        inside = compare(prediction(), measurement([950, 1000, 1050]))
        row = inside["models"]["m"]["tg128"]
        self.assertTrue(row["predicted_inside_observation"])
        outside = compare(prediction(), measurement([1200, 1300, 1250]))
        self.assertFalse(outside["models"]["m"]["tg128"]["predicted_inside_observation"])
        # The error is still the point estimate's, whichever way the flag went.
        self.assertGreater(outside["models"]["m"]["tg128"]["signed_relative_error"], 0.2)

    def test_a_comparison_carries_the_grouping_its_slope_came_from(self):
        document = compare(prediction(), measurement([990, 1000, 1010]))
        self.assertEqual(document["decode_grouping"], "one_process_per_shape")
        self.assertEqual(document["warmup"], {"discarded": None})

    def test_a_comparison_that_cannot_say_its_grouping_is_refused(self):
        # A missing grouping is not "the grouping did not matter": it is a slope
        # whose protocol is unknown, and B2 measured a 1.3x difference between
        # the two protocols on the same three depths.
        document = compare(prediction(), measurement([990, 1000, 1010]))
        document["decode_grouping"] = None
        with self.assertRaises(ValueError) as caught:
            schema.validate_comparison(document)
        self.assertIn("grouping", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
