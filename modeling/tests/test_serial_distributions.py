"""Independent hand calculations for serial distribution propagation; no hardware."""
from fractions import Fraction
import json
import unittest
from unittest import mock

from llm_infer_model.performance import InvalidModel, Parameter, ParameterBundle, UnknownParameter
from llm_infer_model.performance import serial_distributions as distributions
from llm_infer_model.performance.serial_actions import (
    SerialActionTiming, SerialTimingProgram, evaluate_serial_actions,
)


def point(value):
    return distributions.DiscreteDuration((value,), (1,))


def program():
    return SerialTimingProgram((SerialActionTiming(0, "COMPUTE", "compute", "gap0"),
                                SerialActionTiming(1, "COPY", "copy", "gap1")), "tail")


def process(identity="p0", **updates):
    values = dict(compute=point(10), gap0=point(1), copy=point(4), gap1=point(2), tail=point(3))
    values.update(updates)
    return distributions.ProcessDurationDistribution(identity, values)


def evaluate(processes, *, timing_program=None, trials=1001, seed=17):
    return distributions.evaluate_serial_distributions(program() if timing_program is None else timing_program,
        processes, trials=trials, seed=seed)


class DiscreteDurationContractTests(unittest.TestCase):
    def test_exact_frequency_weighted_expectation(self):
        value = distributions.DiscreteDuration((0, 2, 9), (1, 3, 2))
        self.assertEqual(value.total_count, 6)
        self.assertEqual(value.expectation_ns, 4)

    def test_unknown_is_not_observed_zero(self):
        self.assertEqual(point(0).expectation_ns, 0)
        with self.assertRaises(UnknownParameter):
            distributions.DiscreteDuration((None,), (1,))
        with self.assertRaises(UnknownParameter):
            process(compute=None)

    def test_invalid_support_frequency_and_shapes_rejected(self):
        bad = (((), ()), ((1,), ()), ((1, 1), (1, 2)), ((2, 1), (1, 1)),
               ((1,), (0,)), ((1,), (-1,)), ((1,), (True,)), ((1,), (1.5,)),
               ((True,), (1,)), ((-1,), (1,)), ((0.5,), (1,)))
        for values, counts in bad:
            with self.subTest(values=values, counts=counts), self.assertRaises(InvalidModel):
                distributions.DiscreteDuration(values, counts)

    def test_json_roundtrip_is_explicit_and_does_not_accept_raw_trace_fields(self):
        values = (process(), process("p1", compute=distributions.DiscreteDuration((5, 15), (2, 1))))
        document = distributions.distributions_to_dict(values)
        restored = distributions.distributions_from_dict(json.loads(json.dumps(document)))
        self.assertEqual([p.to_dict() for p in values], [p.to_dict() for p in restored])
        document["processes"][0]["whole_iteration_totals_ns"] = [20]
        with self.assertRaises(InvalidModel):
            distributions.distributions_from_dict(document)

    def test_process_mapping_is_frozen_and_ids_must_be_unique(self):
        original = dict(process().durations)
        source = distributions.ProcessDurationDistribution("a", original)
        original["compute"] = point(1000)
        self.assertEqual(source.durations["compute"].expectation_ns, 10)
        with self.assertRaises(TypeError):
            source.durations["compute"] = point(1000)
        with self.assertRaises(InvalidModel):
            evaluate((source, source))


class SerialDistributionCalculationTests(unittest.TestCase):
    def test_point_distributions_reduce_to_deterministic_graph(self):
        result = evaluate((process(),), trials=101)
        self.assertEqual(result.expectation_ns, 20)
        self.assertEqual(set(result.simulated_totals_ns), {20})
        self.assertEqual(result.deterministic_reference.duration_ns, 20)
        self.assertEqual(dict(result.quantiles_ns), {Fraction(1, 10): 20, Fraction(1, 2): 20, Fraction(9, 10): 20})
        self.assertEqual(result.minimum_possible_ns, 20)
        self.assertEqual(result.maximum_possible_ns, 20)

    def test_exact_mean_equal_process_weight_not_pooled_sample_count(self):
        # Process A observed 0 ten times, process B observed 10 once. Equal
        # process weights imply E[compute]=5, not pooled-sample value 10/11.
        a = process("a", compute=distributions.DiscreteDuration((0,), (10,)))
        b = process("b", compute=distributions.DiscreteDuration((10,), (1,)))
        result = evaluate((a, b), trials=101)
        self.assertEqual(result.expectation_ns, 15)
        self.assertEqual(next(x.expectation_ns for x in result.expected_intervals if x.parameter == "compute"), 5)
        self.assertEqual(sum(x.expectation_ns for x in result.expected_intervals), result.expectation_ns)

    def test_frequency_counts_control_draws_instead_of_uniform_support_weight(self):
        p = process(compute=distributions.DiscreteDuration((0, 10), (999, 1)))
        result = evaluate((p,), trials=10001)
        self.assertEqual(result.expectation_ns, Fraction(1001, 100))
        self.assertEqual(dict(result.quantiles_ns)[Fraction(9, 10)], 10)

    def test_median_is_not_sum_of_marginal_medians(self):
        # Each variable is 0 with probability .6 and 10 with probability .4.
        # Marginal medians are 0. The independent sum has probabilities
        # P(0)=.36, P(10)=.48, P(20)=.16, hence median 10 and exact mean 8.
        timing = SerialTimingProgram((SerialActionTiming(0, "COMPUTE", "a", "zero"),
                                      SerialActionTiming(1, "COMPUTE", "b", "zero")), "zero")
        marginal = distributions.DiscreteDuration((0, 10), (3, 2))
        p = distributions.ProcessDurationDistribution("p", dict(a=marginal, b=marginal, zero=point(0)))
        result = evaluate((p,), timing_program=timing, trials=10001)
        self.assertEqual(result.expectation_ns, 8)
        self.assertEqual(dict(result.quantiles_ns)[Fraction(1, 2)], 10)
        self.assertEqual(set(result.simulated_totals_ns), {0, 10, 20})

    def test_process_common_regime_is_retained_instead_of_per_interval_process_mixing(self):
        timing = SerialTimingProgram((SerialActionTiming(0, "COMPUTE", "a", "zero"),
                                      SerialActionTiming(1, "COMPUTE", "b", "zero")), "zero")
        fast = distributions.ProcessDurationDistribution("fast", dict(a=point(0), b=point(0), zero=point(0)))
        slow = distributions.ProcessDurationDistribution("slow", dict(a=point(10), b=point(10), zero=point(0)))
        result = evaluate((fast, slow), timing_program=timing, trials=1001)
        self.assertEqual(result.expectation_ns, 10)
        self.assertEqual(set(result.simulated_totals_ns), {0, 20})
        self.assertNotIn(10, result.simulated_totals_ns)
        self.assertEqual(sum(count for _, count in result.process_draw_counts), result.trials)

    def test_same_parameter_is_redrawn_for_each_interval_without_replaying_one_trace(self):
        timing = SerialTimingProgram((SerialActionTiming(0, "COMPUTE", "a", "zero"),
                                      SerialActionTiming(1, "COMPUTE", "a", "zero")), "zero")
        p = distributions.ProcessDurationDistribution("p", dict(
            a=distributions.DiscreteDuration((0, 10), (1, 1)), zero=point(0)))
        result = evaluate((p,), timing_program=timing, trials=1001)
        self.assertEqual(result.expectation_ns, 10)
        self.assertIn(10, result.simulated_totals_ns)

    def test_fractional_mean_uses_exact_arithmetic_and_reference_rounding_is_separate(self):
        p = process(compute=distributions.DiscreteDuration((0, 1), (1, 1)))
        result = evaluate((p,), trials=101)
        self.assertEqual(result.expectation_ns, Fraction(21, 2))
        self.assertEqual(result.deterministic_reference.unrounded_interval_sum_ns, Fraction(21, 2))
        self.assertEqual(result.deterministic_reference.duration_ns, 11)
        self.assertEqual(result.to_dict()["deterministic_mean_parameter_reference"]["integer_stage_rounding_ns"],
                         dict(numerator=1, denominator=2))

    def test_reference_uses_existing_model_once_not_per_monte_carlo_trial(self):
        with mock.patch.object(distributions, "evaluate_serial_actions", wraps=evaluate_serial_actions) as spy:
            result = evaluate((process(),), trials=101)
        spy.assert_called_once()
        self.assertEqual(result.deterministic_reference.duration_ns, 20)

    def test_drawn_integer_totals_match_graph_for_each_possible_small_assignment(self):
        timing = SerialTimingProgram((SerialActionTiming(0, "COPY", "a", "zero"),
                                      SerialActionTiming(1, "ADVANCE", "b", "zero")), "zero")
        for a in (0, 2):
            for b in (0, 3):
                with self.subTest(a=a, b=b):
                    p = distributions.ProcessDurationDistribution("p", dict(a=point(a), b=point(b), zero=point(0)))
                    result = evaluate((p,), timing_program=timing, trials=1)
                    costs = ParameterBundle(tuple(Parameter(name, value, "ns", "declared", (), "test")
                        for name, value in (("a", a), ("b", b), ("zero", 0))))
                    self.assertEqual(result.simulated_totals_ns[0], evaluate_serial_actions(timing, costs).duration_ns)

    def test_fixed_seed_trial_count_reproduces_samples_and_digest(self):
        p = process(compute=distributions.DiscreteDuration((2, 8, 10), (1, 3, 2)))
        a, b = evaluate((p,), trials=101, seed=77), evaluate((p,), trials=101, seed=77)
        self.assertEqual(a.simulated_totals_ns, b.simulated_totals_ns)
        self.assertEqual(a.to_dict()["monte_carlo"]["simulated_sequence_sha256"],
                         b.to_dict()["monte_carlo"]["simulated_sequence_sha256"])

    def test_rank_error_explains_only_monte_carlo_not_hardware_accuracy(self):
        small = evaluate((process(),), trials=101)
        large = evaluate((process(),), trials=401)
        self.assertGreater(small.cdf_rank_error_bound, large.cdf_rank_error_bound)
        for _, lower, upper in small.quantile_value_intervals_ns:
            self.assertEqual((lower, upper), (20, 20))
        doc = small.to_dict()
        self.assertFalse(doc["independently_validated_accuracy_claimed"])
        self.assertFalse(doc["monte_carlo"]["source_estimation_or_hardware_uncertainty_included"])
        self.assertTrue(doc["conditional_cross_interval_independence_assumed"])
        self.assertFalse(doc["whole_trace_resampling"])
        self.assertFalse(doc["same_invocation_cross_interval_correlation_modeled"])
        self.assertFalse(doc["whole_iteration_totals_used_as_parameters"])

    def test_single_trial_bounds_include_unsampled_finite_support_extremes(self):
        p = process(compute=distributions.DiscreteDuration((0, 1000), (999999, 1)))
        result = evaluate((p,), trials=1)
        for _, lower, upper in result.quantile_value_intervals_ns:
            self.assertEqual((lower, upper), (10, 1010))


class SerialDistributionRejectionTests(unittest.TestCase):
    def test_missing_interval_rejected_even_for_nominally_zero_gap_or_tail(self):
        for missing in ("compute", "gap0", "gap1", "tail"):
            values = dict(process().durations)
            del values[missing]
            with self.subTest(missing=missing), self.assertRaises(UnknownParameter):
                evaluate((distributions.ProcessDurationDistribution("p", values),))

    def test_complete_iteration_total_cannot_be_an_extra_input_price(self):
        with self.assertRaises(InvalidModel):
            evaluate((process(whole_iteration_total=point(20)),))

    def test_bad_process_collection_trial_count_and_seed_rejected(self):
        for processes in ((), [], [process()]):
            with self.subTest(processes=processes), self.assertRaises(InvalidModel):
                evaluate(processes)
        for trials in (0, -1, True, 1.5):
            with self.subTest(trials=trials), self.assertRaises(InvalidModel):
                evaluate((process(),), trials=trials)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(InvalidModel):
                evaluate((process(),), seed=seed)
        with self.assertRaises(InvalidModel):
            evaluate((process(),), timing_program={})

    def test_invalid_cdf_probability_and_arbitrary_process_weights_rejected(self):
        for value in (0, 0.01, Fraction(0), Fraction(1), Fraction(-1, 3)):
            with self.subTest(value=value), self.assertRaises(InvalidModel):
                distributions.evaluate_serial_distributions(program(), (process(),),
                    trials=101, seed=17, cdf_failure_probability=value)
        document = distributions.distributions_to_dict((process(),))
        document["process_weight"] = "sample_count_weighted"
        with self.assertRaises(InvalidModel):
            distributions.distributions_from_dict(document)


if __name__ == "__main__":
    unittest.main()
