"""A pair is two wall clocks of one prediction, and nothing else is a pair."""
import unittest

from composing.pairs import pairs


def measured(rows, *, prediction="sha-1", grouping="one_process_per_model_decode", type_k="f16"):
    return {"prediction": {"path": "predicted.json", "sha256": prediction},
            "decode_grouping": grouping, "warmup": {"discarded": None},
            "threads": 2, "cpu_mask": [2, 4], "llama_bench_sha256": "bench",
            "models": {model: {config: {"avg_ns": round(sum(samples) / len(samples)),
                                        "samples_ns": samples,
                                        "llama_bench": {"type_k": type_k}}
                               for config, samples in configs.items()}
                       for model, configs in rows.items()}}


def steady(ns, jitter=10):
    return [ns - jitter, ns, ns + jitter]


class PairsTests(unittest.TestCase):
    def test_two_rounds_of_one_prediction_give_a_ratio_per_row(self):
        report = pairs(measured({"m": {"tg128": steady(1000)}}),
                       measured({"m": {"tg128": steady(1200)}}))
        row = report["models"]["m/tg128"]
        self.assertEqual(row["observed_a_ns"], 1000)
        self.assertEqual(row["observed_b_ns"], 1200)
        self.assertAlmostEqual(row["ratio"], 1.2)
        self.assertTrue(report["same_grouping"])

    def test_two_predictions_are_not_a_pair(self):
        # The difference would be a table change and a machine change at once,
        # and no arithmetic separates them after the fact.
        with self.assertRaises(ValueError) as caught:
            pairs(measured({"m": {"tg128": steady(1000)}}, prediction="sha-1"),
                  measured({"m": {"tg128": steady(1000)}}, prediction="sha-2"))
        self.assertIn("different predictions", str(caught.exception))

    def test_a_wall_clock_that_names_no_prediction_cannot_be_paired(self):
        document = measured({"m": {"tg128": steady(1000)}})
        document["prediction"] = {}
        with self.assertRaises(ValueError) as caught:
            pairs(document, measured({"m": {"tg128": steady(1000)}}))
        self.assertIn("no prediction sha256", str(caught.exception))

    def test_a_row_measured_once_is_named_not_compared(self):
        report = pairs(measured({"m": {"tg128": steady(1000)}}),
                       measured({"m": {"tg128": steady(1000), "tg512": steady(4000)}}))
        self.assertEqual(report["unpaired"], ["m/tg512"])
        self.assertNotIn("m/tg512", report["models"])

    def test_a_ratio_beyond_the_band_is_listed(self):
        report = pairs(measured({"m": {"tg128": steady(1000), "tg512": steady(1000)}}),
                       measured({"m": {"tg128": steady(1010), "tg512": steady(1400)}}))
        self.assertEqual(report["summary"]["outside_band"], ["m/tg512"])
        self.assertEqual(report["summary"]["widest_row"], "m/tg512")

    def test_rounds_neither_span_separates_are_counted(self):
        report = pairs(measured({"m": {"tg128": [990, 1000, 1010]}}),
                       measured({"m": {"tg128": [1005, 1010, 1015]}}))
        self.assertEqual(report["summary"]["not_separable"], ["m/tg128"])

    def test_rounds_a_span_separates_are_not_counted(self):
        report = pairs(measured({"m": {"tg128": [980, 1000, 1020]}}),
                       measured({"m": {"tg128": [1380, 1400, 1420]}}))
        self.assertEqual(report["summary"]["not_separable"], [])
        self.assertFalse(report["models"]["m/tg128"]["a_inside_b_span"])

    def test_a_kv_cache_read_in_two_types_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            pairs(measured({"m": {"tg128": steady(1000)}}, type_k="f16"),
                  measured({"m": {"tg128": steady(1000)}}, type_k="q8_0"))
        self.assertIn("byte counts differ", str(caught.exception))
        with self.assertRaises(ValueError):
            pairs(measured({"m": {"tg128": steady(1000)}}, type_k="f16"),
                  measured({"m": {"tg128": steady(1000)}}, type_k="f16")
                  | {"models": {"m": {"tg512": {"avg_ns": 4000, "samples_ns": steady(4000),
                                                "llama_bench": {"type_k": "q8_0"}}}}})


if __name__ == "__main__":
    unittest.main()
