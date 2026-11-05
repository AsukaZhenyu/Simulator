"""Margin, flip and regret, checked against arithmetic rather than a campaign.

Every case is built by hand so the expected answer can be computed on paper: a
strategy is a sum of prices, a margin is a ratio of two of those sums, and a
flip is a comparison between them.  A test that needed the archived campaign
would be testing the archive, not the accounting.
"""
from __future__ import annotations

from fractions import Fraction
import json
import random
import unittest

from profiling.decision_margin import margin as M


def instance(prices, measured, labels=("a", "b")):
    programs = tuple(M.Program("w", i, (label,)) for i, label in enumerate(labels))
    return M.Instance("w", programs, {i: Fraction(str(v)) for i, v in enumerate(measured)})


class MarginIsTheRatioOfTheTopTwo(unittest.TestCase):
    def test_margin_uses_the_two_cheapest_not_the_cheapest_and_the_worst(self):
        case = instance({"a": 100, "b": 110, "c": 400}, measured=[1, 1, 1], labels=("a", "b", "c"))
        best, runner_up, span, _ = case.margins({"a": 100, "b": 110, "c": 400})
        self.assertEqual((best, runner_up), (0, 1))
        self.assertAlmostEqual(span, 0.10)

    def test_a_single_candidate_has_no_runner_up_and_no_margin(self):
        case = instance({"a": 100}, measured=[1], labels=("a",))
        best, runner_up, span, _ = case.margins({"a": 100})
        self.assertEqual((best, runner_up, span), (0, None, 0.0))

    def test_ties_are_broken_by_strategy_index_not_by_float_noise(self):
        case = instance({"a": 100, "b": 100}, measured=[1, 1])
        self.assertEqual(case.margins({"a": 100, "b": 100})[0], 0)


class TheRadiusIsTheExactThreshold(unittest.TestCase):
    """T1(1+eps) < T2(1-eps)  <=>  eps < (T2-T1)/(T1+T2)."""

    def threshold(self):
        return (110 - 100) / (110 + 100)

    def flip(self, epsilon):
        case = instance({"a": 100, "b": 110}, measured=[1, 1])
        prices = {"a": 100 * (1 + epsilon), "b": 110 * (1 - epsilon)}
        return case.margins(prices)[0] != 0

    def test_just_below_the_threshold_keeps_the_order(self):
        self.assertFalse(self.flip(self.threshold() * 0.999))

    def test_just_above_the_threshold_reverses_it(self):
        self.assertTrue(self.flip(self.threshold() * 1.001))

    def test_the_reported_radius_matches_the_arithmetic(self):
        case = instance({"a": 100, "b": 110}, measured=[1, 1])
        _, _, span, _ = case.margins({"a": 100, "b": 110})
        radius = span / (2.0 + span)
        self.assertAlmostEqual(radius, self.threshold())


class CommonModeCannotFlip(unittest.TestCase):
    def test_one_factor_on_every_price_preserves_every_order(self):
        prices = {"a": 100, "b": 103, "c": 250}
        case = instance(prices, measured=[1, 1, 1], labels=("a", "b", "c"))
        self.assertTrue(M.common_mode_cannot_flip(case, prices, 0.5))

    def test_an_unequal_factor_does_flip(self):
        case = instance({"a": 100, "b": 103}, measured=[1, 1])
        self.assertNotEqual(case.margins({"a": 100 * 1.05, "b": 103 * 0.95})[0], 0)


class RegretIsMeasuredNotPredicted(unittest.TestCase):
    def test_regret_compares_the_chosen_strategy_with_the_measured_best(self):
        case = instance({"a": 100, "b": 100.1}, measured=[1.0, 1.25])
        self.assertAlmostEqual(case.regret(1), 0.25)
        self.assertAlmostEqual(case.regret(0), 0.0)


class FlipStudyCountsAndPricesTheCost(unittest.TestCase):
    def test_a_near_tie_flips_often_but_costs_almost_nothing(self):
        prices = {"a": 100.0, "b": 100.5}
        case = instance({"a": 100, "b": 100.5}, measured=[1.00, 1.001])
        rng = random.Random(3)
        draws = [M.scaled_prices(prices, 0.05, rng) for _ in range(400)]
        result = M.flip_study(case, draws, 400, rng)
        self.assertGreater(result.flips, 0)
        self.assertLess(max(result.regrets), 0.01)

    def test_a_wide_margin_never_flips_at_a_small_error(self):
        prices = {"a": 100.0, "b": 200.0}
        case = instance({"a": 100, "b": 200}, measured=[1.0, 2.0])
        rng = random.Random(5)
        draws = [M.scaled_prices(prices, 0.05, rng) for _ in range(400)]
        self.assertEqual(M.flip_study(case, draws, 400, rng).flips, 0)

    def test_every_draw_is_accounted_for_in_the_choice_counts(self):
        prices = {"a": 100.0, "b": 101.0}
        case = instance({"a": 100, "b": 101}, measured=[1.0, 1.0])
        rng = random.Random(7)
        draws = [M.scaled_prices(prices, 0.05, rng) for _ in range(300)]
        result = M.flip_study(case, draws, 300, rng)
        self.assertEqual(sum(result.choices.values()), 300)


class BootstrapStaysInsideTheMeasuredRepeats(unittest.TestCase):
    def test_a_resampled_price_is_a_repeat_mean_of_that_price_only(self):
        prices = {"a": 100.0, "b": 200.0}
        samples = {"a": [Fraction(90), Fraction(100), Fraction(110)],
                   "b": [Fraction(200), Fraction(200), Fraction(200)]}
        rng = random.Random(11)
        for _ in range(50):
            draw = M.bootstrap_prices(prices, samples, rng)
            self.assertGreaterEqual(draw["a"], 90.0)
            self.assertLessEqual(draw["a"], 110.0)
            self.assertEqual(draw["b"], 200.0)

    def test_a_price_without_enough_repeats_is_left_alone(self):
        prices = {"a": 100.0}
        samples = {"a": [Fraction(100)]}
        draw = M.bootstrap_prices(prices, samples, random.Random(0))
        self.assertEqual(draw["a"], 100.0)


class EvidenceIsReadFromItsOwnShape(unittest.TestCase):
    def test_grouped_native_output_selects_the_requested_action(self):
        native = {"groups": [{"action_index": 0, "trials": [{"wall_ns": 800, "repeats": 8}]},
                             {"action_index": 1, "trials": [{"wall_ns": 1600, "repeats": 8}]}]}
        self.assertEqual(M.trial_values(native, 1), [Fraction(200)])

    def test_single_action_output_falls_back_to_the_whole_trial_list(self):
        native = {"trials": [{"wall_ns": 400, "repeats": 4}]}
        self.assertEqual(M.trial_values(native, 3), [Fraction(100)])

    def test_a_trial_without_a_wall_clock_is_skipped(self):
        native = {"trials": [{"wall_ns": None, "repeats": 8}, {"wall_ns": 80, "repeats": 8}]}
        self.assertEqual(M.trial_values(native, 0), [Fraction(10)])


class ArchivedPathsStillResolve(unittest.TestCase):
    def test_a_receipt_under_the_old_build_root_is_redirected(self):
        old = r"C:\work\Simulator\build\dac-validation-20261025\e2a-v2\jobs\a\result.json"
        got = M.relocated(old, r"C:\work\paper-write\simulator-archive\build")
        self.assertEqual(str(got).replace("\\", "/"),
                         "C:/work/paper-write/simulator-archive/build/dac-validation-20261025/e2a-v2/jobs/a/result.json")

    def test_a_receipt_outside_the_build_root_is_not_rewritten(self):
        other = r"C:\work\elsewhere\result.json"
        self.assertEqual(M.relocated(other, r"C:\work\archive"), M.relocated(other, None))


if __name__ == "__main__":
    unittest.main()
