"""The error decomposition must not average away a row it cannot stand behind."""
import unittest

from composing.decompose import decompose

HISTORY = {"tg128": 128, "tg512": 512, "tg1024": 1024}


def prediction(shapes_ns, layers=2):
    shapes = {name: {"total_ns": total, "attention": total // 10}
              for name, total in shapes_ns.items()}
    return {"models": {"m": {"layers": layers, "shapes": shapes}}}


def comparison(observed, cv=None, grouping="one_process_per_model_decode"):
    cv = cv or {}
    return {"decode_grouping": grouping,
            "models": {"m": {name: {"observed_ns": value, "observed_cv": cv.get(name, 0.01)}
                             for name, value in observed.items()}}}


def prediction_with(intercept_ns, slope_ns_per_token, attention_slope_ns_per_token):
    """A table whose total slope is partly the one stage that reads history."""
    shapes = {name: {"total_ns": int(intercept_ns + slope_ns_per_token * history),
                     "attention": int(attention_slope_ns_per_token * history)}
              for name, history in HISTORY.items()}
    return {"models": {"m": {"layers": 1, "shapes": shapes}}}


def line(intercept_ns, slope_ns_per_token, shapes=("tg128", "tg512", "tg1024")):
    return {name: int(intercept_ns + slope_ns_per_token * HISTORY[name]) for name in shapes}


class DecomposeTests(unittest.TestCase):
    def test_an_affine_pair_gives_back_the_terms_it_was_built_from(self):
        table = line(1_000_000, 10_000)
        report = decompose(prediction(table), comparison(table))["models"]["m"]
        self.assertAlmostEqual(report["intercept_ratio"], 1.0)
        self.assertAlmostEqual(report["slope_ratio"], 1.0)
        self.assertAlmostEqual(report["slope_ratio_after_state_factor"], 1.0)
        self.assertAlmostEqual(report["slope_predicted_us"], 10.0)
        self.assertFalse(report["ambiguous"])
        self.assertIsNone(report["shape_excluded_unphysical"])

    def test_the_like_for_like_ratio_compares_only_the_stage_that_reads_history(self):
        # Total 10 us/KVtok of which attention is 4: the other 6 is stages that
        # do not read history at all, so the honest question is what the wall
        # clock did against the 4.
        table = prediction_with(1_000_000, 10_000, 4_000)
        report = decompose(table, comparison(line(1_000_000, 10_000)))["models"]["m"]
        self.assertAlmostEqual(report["slope_ratio"], 1.0)
        self.assertAlmostEqual(report["slope_ratio_attention_only"], 2.5)
        self.assertAlmostEqual(report["attention_only_table_slope_us"], 4.0)
        self.assertAlmostEqual(report["history_invariant_table_slope_us"], 6.0)

    def test_a_table_that_already_agrees_with_itself_reports_one_ratio_twice(self):
        report = decompose(prediction_with(1_000_000, 4_000, 4_000),
                           comparison(line(1_000_000, 4_000)))["models"]["m"]
        self.assertAlmostEqual(report["history_invariant_table_slope_us"], 0.0)
        self.assertAlmostEqual(report["slope_ratio_attention_only"], report["slope_ratio"])

    def test_a_slope_that_is_under_priced_is_reported_with_the_state_factor_removed(self):
        table = line(1_000_000, 10_000)
        measured = line(1_200_000, 15_000)          # 20% state on every token, 1.5x slope
        report = decompose(prediction(table), comparison(measured))["models"]["m"]
        self.assertAlmostEqual(report["intercept_ratio"], 1.2)
        self.assertAlmostEqual(report["slope_ratio"], 1.5)
        self.assertAlmostEqual(report["slope_ratio_after_state_factor"], 1.25)

    def test_a_row_that_moves_inside_itself_is_not_a_point(self):
        table = line(1_000_000, 10_000)
        measured = line(1_000_000, 10_000)
        measured["tg512"] *= 3
        report = decompose(prediction(table),
                           comparison(measured, cv={"tg512": 0.30}))["models"]["m"]
        self.assertEqual(report["shapes_used"], ["tg128", "tg1024"])
        self.assertIn("tg512", report["unsteady_shapes"])

    def test_a_deeper_shape_measuring_faster_names_the_row_it_drops_and_its_alternatives(self):
        table = line(1_000_000, 50_000)
        measured = line(1_000_000, 50_000)
        measured["tg512"] *= 2                      # 53.2 ms, above the deeper 52.2 ms
        report = decompose(prediction(table), comparison(measured))["models"]["m"]
        self.assertEqual(report["shape_excluded_unphysical"], "tg512")
        self.assertEqual(report["shapes_used"], ["tg128", "tg1024"])
        # The row would have to be half its measured value: not a state factor.
        self.assertTrue(report["ambiguous"])
        self.assertEqual([item["dropped"] for item in report["exclusion_candidates"]],
                         ["tg512", "tg1024", "tg128"])
        self.assertFalse(report["exclusion_candidates"][0]["within_measured_state_band"])
        self.assertAlmostEqual(report["exclusion_candidates"][0]["implied_state_factor"], 0.5,
                               places=2)

    def test_two_readings_inside_the_band_report_both_slopes_not_the_smallest_deviation(self):
        # Measured OLMoE, one process, 2026-11-05: 51.659 / 48.269 / 52.898 ms.
        # Dropping the middle row fits the outer two exactly and leaves slope
        # 1.38 us; dropping the first leaves 9.04 us.  Both stay inside the state
        # band, so the run measured no slope -- it measured two, and the report
        # has to carry both instead of the one with the smaller deviation.
        shapes = {"tg128": 51_659_000, "tg512": 48_269_000, "tg1024": 52_898_000}
        report = decompose(prediction(line(1_000_000, 10_000)), comparison(shapes))["models"]["m"]
        self.assertEqual(report["shape_excluded_unphysical"], "tg512")
        self.assertEqual([item["dropped"] for item in report["admissible_readings"]],
                         ["tg512", "tg128"])
        self.assertAlmostEqual(report["admissible_readings"][0]["slope_us"], 1.383, places=3)
        self.assertAlmostEqual(report["admissible_readings"][1]["slope_us"], 9.041, places=3)
        self.assertAlmostEqual(report["admissible_slope_range_us"], 7.658, places=3)

    def test_a_single_explanation_leaves_no_slope_range_to_report(self):
        shapes = {"tg128": 10_000_000, "tg512": 10_700_000, "tg1024": 10_600_000}
        report = decompose(prediction(shapes), comparison(shapes))["models"]["m"]
        self.assertGreater(len(report["admissible_readings"]), 1)
        self.assertFalse(report["ambiguous"])
        no_inversion = line(1_000_000, 10_000)
        report = decompose(prediction(no_inversion), comparison(no_inversion))["models"]["m"]
        self.assertEqual(report["admissible_readings"], [])
        self.assertIsNone(report["admissible_slope_range_us"])

    def test_an_inversion_inside_the_state_band_is_not_flagged_ambiguous(self):
        shapes = {"tg128": 10_000_000, "tg512": 10_700_000, "tg1024": 10_600_000}
        report = decompose(prediction(shapes), comparison(shapes))["models"]["m"]
        self.assertEqual(report["shape_excluded_unphysical"], "tg512")
        self.assertFalse(report["ambiguous"])
        self.assertTrue(report["exclusion_candidates"][0]["within_measured_state_band"])

    def test_a_model_with_no_measurement_is_named_not_fitted(self):
        table = line(1_000_000, 10_000)
        document = prediction(table)
        document["models"]["other"] = document["models"]["m"]
        report = decompose(document, comparison(table))
        self.assertIn("other", report["not_measured"])
        self.assertNotIn("other", report["models"])

    def test_every_shape_moving_inside_itself_yields_no_slope_and_says_why(self):
        from composing.decompose import show
        import io
        from contextlib import redirect_stdout
        table = line(1_000_000, 10_000)
        measured = line(1_000_000, 10_000)
        report = decompose(prediction(table),
                           comparison(measured, cv={name: 0.3 for name in HISTORY}))
        self.assertEqual(report["not_measured"], ["m"])
        self.assertNotIn("shapes_used", report["models"]["m"])
        # Printing it must not raise: a run that measured nothing usable is still
        # a result, and the report has to say that instead of inventing a line.
        out = io.StringIO()
        with redirect_stdout(out):
            show(report)
        self.assertIn("no slope", out.getvalue())
        for name in HISTORY:
            self.assertIn(name, out.getvalue())

    def test_the_state_band_is_the_one_variability_reports(self):
        # Two constants with the same value drift apart on the first edit that
        # touches one of them; the decomposition must be judging against the
        # same band a comparison file prints beside each row.
        from composing import decompose as decompose_module
        from composing import variability
        self.assertIs(decompose_module.STATE_BAND, variability.DEFAULT_BAND)

    def test_the_grouping_its_slope_came_from_is_carried_and_shown(self):
        from composing.decompose import show
        import io
        from contextlib import redirect_stdout
        table = line(1_000_000, 10_000)
        report = decompose(prediction(table),
                           comparison(table, grouping="one_process_per_shape"))
        self.assertEqual(report["decode_grouping"], "one_process_per_shape")
        out = io.StringIO()
        with redirect_stdout(out):
            show(report)
        # One process per depth charges every depth its own first decode and
        # reports a lower slope; the print has to say so, not just carry a string.
        self.assertIn("biased low", out.getvalue())


class DecomposeCommandTests(unittest.TestCase):
    def test_merging_two_groupings_into_one_fit_is_refused(self):
        # --comparison exists to put one model's rows beside another's; letting
        # two protocols in would fit one slope across rows that answered
        # different questions, and the file would not even say which.
        import json
        import pathlib
        import tempfile
        from composing.__main__ import main
        table = line(1_000_000, 10_000)
        with tempfile.TemporaryDirectory() as raw:
            base = pathlib.Path(raw)
            (base / "predicted.json").write_text(json.dumps(prediction(table)), encoding="utf-8")
            (base / "comparison.json").write_text(
                json.dumps(comparison(table, grouping="one_process_per_shape")), encoding="utf-8")
            other = base / "other.json"
            other.write_text(json.dumps(comparison(table, grouping="one_process_per_model_decode")),
                             encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                main(["decompose", "--e2e", str(base), "--comparison", str(other)])
        self.assertIn("decode_grouping", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
