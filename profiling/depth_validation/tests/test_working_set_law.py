# -*- coding: utf-8 -*-
"""The working-set law must recover its own exponent and refuse to invent one.

`freeze` stores each coordinate's price at a reference working set and scales a
program's total by ``(ws / ws_ref) ** theta``.  These tests generate prices from
a known law and check that the fit returns it, that a cell measured in a single
program is divided out by that program's factor, and that the degenerate
profiles are refused rather than fitted with an arbitrary slope.
"""
from __future__ import annotations

import math
import unittest
from fractions import Fraction

from profiling.depth_validation.campaign import levels_at_reference, working_set_law

REFERENCE = 100_000_000
THETA = 0.1216
WORKING_SET = {"p0": 50_000_000, "p1": 100_000_000, "p2": 200_000_000}


def prices_from(exponent, coordinates, programs=WORKING_SET, reference=REFERENCE):
    """Per-action samples that follow price = level * (ws / reference) ** theta."""
    prices = {}
    for h, level in coordinates.items():
        row = {}
        for p, ws in programs.items():
            row[p] = [Fraction(level * math.exp(exponent * math.log(ws / reference)))
                      .limit_denominator(10 ** 6)]
        prices[h] = row
    return prices


class WorkingSetLaw(unittest.TestCase):
    def test_the_fitted_exponent_recovers_the_generating_one(self):
        coordinates = {"c%02d" % i: 1000.0 + 37 * i for i in range(12)}
        report = working_set_law(prices_from(THETA, coordinates), WORKING_SET,
                                 sorted(WORKING_SET))
        self.assertAlmostEqual(report["exponent"], THETA, places=6)
        self.assertEqual(report["reference_bytes"], REFERENCE)
        self.assertEqual(report["shared"], 12)
        self.assertEqual(report["singletons"], 0)
        self.assertEqual(report["programs"], 3)
        for h, level in coordinates.items():
            self.assertAlmostEqual(report["levels"][h] / level, 1.0, places=6)

    def test_a_program_factor_is_the_law_at_its_own_size(self):
        coordinates = {"c%02d" % i: 1000.0 + 37 * i for i in range(12)}
        report = working_set_law(prices_from(THETA, coordinates), WORKING_SET,
                                 sorted(WORKING_SET))
        for p, ws in WORKING_SET.items():
            self.assertAlmostEqual(report["factors"][p], (ws / REFERENCE) ** THETA,
                                   places=6)

    def test_a_cell_seen_in_one_program_is_divided_by_that_program(self):
        levels, singletons = levels_at_reference(
            {"single": {"p0": [Fraction(5000)]}}, THETA, {"p0": 50_000_000}, REFERENCE)
        self.assertEqual(singletons, 1)
        self.assertAlmostEqual(levels["single"],
                               5000 / math.exp(THETA * math.log(0.5)), places=6)

    def test_one_working_set_across_the_profile_is_refused(self):
        same = {"p0": 100_000_000, "p1": 100_000_000}
        coordinates = {"c%02d" % i: 1000.0 + i for i in range(12)}
        with self.assertRaises(ValueError):
            working_set_law(prices_from(THETA, coordinates, programs=same), same,
                            sorted(same))

    def test_too_few_shared_coordinates_is_refused(self):
        coordinates = {"c%d" % i: 1000.0 + i for i in range(3)}
        with self.assertRaises(ValueError):
            working_set_law(prices_from(THETA, coordinates), WORKING_SET,
                            sorted(WORKING_SET))

    def test_a_single_profile_program_is_refused(self):
        with self.assertRaises(ValueError):
            working_set_law({}, {"p0": 1}, ["p0"])


if __name__ == "__main__":
    unittest.main()
