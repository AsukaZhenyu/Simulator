# -*- coding: utf-8 -*-
"""The frozen table must say where one price serves two different programs.

`freeze` writes one price per semantic coordinate, pooled over every profile
program that produces that coordinate. When two placements realize the same
coordinate at different costs, the pooled price is wrong for at least one of
them, and the heldout cases for the affected placements miss by the size of the
spread. These tests pin the report that makes that visible before the heldout
window opens.
"""
from __future__ import annotations

import unittest

from profiling.depth_validation.campaign import fit_consistency


def samples(n, value):
    return [value] * n


class FitConsistency(unittest.TestCase):
    def test_a_coordinate_seen_in_one_program_is_not_reported(self):
        report = fit_consistency({"a": {"p0": samples(8, 100)}})
        self.assertEqual(report["shared_coordinates"], 0)
        self.assertEqual(report["rows"], [])
        self.assertEqual(report["max_spread"], 1.0)
        self.assertEqual(report["median_spread"], 1.0)

    def test_agreement_across_programs_reports_a_spread_of_one(self):
        report = fit_consistency({"a": {"p0": samples(8, 100), "p1": samples(8, 100)}})
        self.assertEqual(report["shared_coordinates"], 1)
        self.assertAlmostEqual(report["rows"][0]["spread"], 1.0)

    def test_the_worst_disagreement_comes_first(self):
        report = fit_consistency({
            "mild": {"p0": samples(8, 100), "p1": samples(8, 120)},
            "severe": {"p0": samples(8, 22.5), "p1": samples(8, 301.6)},
        })
        self.assertEqual([row["key"] for row in report["rows"]], ["severe", "mild"])
        self.assertAlmostEqual(report["max_spread"], 301.6 / 22.5)
        self.assertEqual(report["shared_coordinates"], 2)

    def test_a_program_with_too_few_samples_does_not_count_as_a_second_program(self):
        # One action occurrence is not a measurement of the coordinate; counting
        # it would report a spread between two numbers of different quality.
        report = fit_consistency({"a": {"p0": samples(8, 100), "p1": samples(3, 400)}})
        self.assertEqual(report["shared_coordinates"], 0)

    def test_the_report_never_claims_a_target_was_used(self):
        report = fit_consistency({"a": {"p0": samples(8, 1), "p1": samples(8, 2)}})
        self.assertIs(report["target_time_used"], False)

    def test_an_empty_profile_does_not_divide_by_zero(self):
        report = fit_consistency({})
        self.assertEqual(report["coordinates"], 0)
        self.assertEqual(report["max_spread"], 1.0)


if __name__ == "__main__":
    unittest.main()
