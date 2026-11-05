"""The summation rule: with nothing declared it must not change a single ns."""
import unittest

from composing.compose import compose_total, compose_with_breakdown
from composing.terms import Term, spec_of


def term(name="per_layer_marginal", phase="prefill", value=1000.0, enabled=True):
    return Term(name=name, phase=phase, value=value, enabled=enabled)


class ZeroTermTests(unittest.TestCase):
    def test_the_historical_expression_is_returned_as_an_integer(self):
        total = compose_total(block_sum_ns=1000, layers=28, boundary_ns=30,
                              phase="decode", tokens=1)
        self.assertEqual(total, 1000 * 28 + 30)
        self.assertIsInstance(total, int)

    def test_a_declared_but_disabled_term_changes_nothing(self):
        disabled = (term(enabled=False), term(phase="prefill", value=9.0, enabled=False))
        total = compose_total(block_sum_ns=1000, layers=28, boundary_ns=30,
                              phase="decode", tokens=1, terms=disabled)
        self.assertEqual(total, 1000 * 28 + 30)

    def test_an_empty_breakdown_says_nothing_was_added(self):
        total, breakdown = compose_with_breakdown(block_sum_ns=10, layers=2, boundary_ns=3,
                                                  phase="decode", tokens=1)
        self.assertEqual((total, breakdown), (23, []))


class UnitTests(unittest.TestCase):
    BASE = dict(block_sum_ns=100, layers=4, boundary_ns=7, phase="prefill", tokens=256)

    def test_per_layer(self):
        total = compose_total(**self.BASE, terms=(term(value=1000.0),))
        self.assertEqual(total, 100 * 4 + 7 + 1000 * 4)

    def test_per_token(self):
        terms = (term(name="per_token_fixed", phase="prefill", value=2.0),)
        self.assertEqual(compose_total(**self.BASE, terms=terms), 100 * 4 + 7 + 2 * 256)

    def test_a_fraction_only_cancels_the_block_stack(self):
        terms = (term(name="stage_overlap", phase="prefill", value=0.25),)
        # 25% of 100*4 = 100 is not serialized; the boundary still runs once.
        self.assertEqual(compose_total(**self.BASE, terms=terms), 400 + 7 - 100)

    def test_a_term_acts_only_on_its_own_phase(self):
        prefill_only = (term(name="per_token_fixed", phase="prefill", value=2.0),)
        self.assertEqual(compose_total(**self.BASE, terms=prefill_only), 407 + 512)
        decode = dict(self.BASE, phase="decode", tokens=1)
        self.assertEqual(compose_total(**decode, terms=prefill_only), 407)

    def test_the_breakdown_names_what_it_added(self):
        total, breakdown = compose_with_breakdown(
            **self.BASE, terms=(term(value=1000.0),))
        self.assertEqual(len(breakdown), 1)
        self.assertEqual(breakdown[0]["name"], "per_layer_marginal")
        self.assertEqual(breakdown[0]["contribution_ns"], 4000.0)
        self.assertEqual(total, 407 + 4000)


class RefusalTests(unittest.TestCase):
    def test_a_quantity_with_no_knob_is_not_a_term(self):
        with self.assertRaisesRegex(ValueError, "no declared quantity"):
            Term(name="whatever_is_missing", phase="decode", value=1.0)

    def test_a_term_refuses_a_phase_it_does_not_apply_to(self):
        with self.assertRaisesRegex(ValueError, "not 'prefill'"):
            Term(name="kv_append_per_token", phase="prefill", value=1.0)

    def test_a_cost_cannot_be_negative(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Term(name="per_layer_marginal", phase="decode", value=-1.0)

    def test_a_share_cannot_exceed_one(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed 1"):
            Term(name="stage_overlap", phase="prefill", value=1.5)

    def test_a_non_finite_value_is_refused(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            Term(name="per_layer_marginal", phase="decode", value=float("nan"))

    def test_every_registered_term_names_a_knob(self):
        for name in ("per_layer_marginal", "per_token_fixed", "kv_append_per_token",
                     "stage_overlap"):
            self.assertTrue(spec_of(name).knob.strip(), name)


if __name__ == "__main__":
    unittest.main()
