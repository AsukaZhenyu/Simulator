"""Preparation mismatches must not certify a different timing boundary."""
from dataclasses import replace
import unittest
from llm_infer_model.performance.preparation_contract import (
    PreparationContract, compare_preparation, require_matched_preparation)
from profiling.dac_refinement.preparation_adapter import (
    IMPLEMENTATION_SOURCES, refinement_preparation)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.contract = PreparationContract(True, 0, 8, True, 20, 500)

    def test_same_contract_allows_check_but_does_not_certify_bias(self):
        result = require_matched_preparation(self.contract, self.contract)
        self.assertEqual(result["status"], "match")
        self.assertTrue(result["allows_unbiased_observer_check"])
        self.assertIn("not an observer-bias certificate", result["scope"])

    def test_one_extra_prime_rejects(self):
        with self.assertRaises(ValueError):
            require_matched_preparation(self.contract,
                replace(self.contract, untimed_complete_prime_passes=1))

    def test_replay_count_mismatch_rejects(self):
        with self.assertRaises(ValueError):
            require_matched_preparation(self.contract,
                replace(self.contract, timed_replays_per_batch=16))

    def test_unknown_is_not_equal_or_zero(self):
        unknown = PreparationContract()
        self.assertEqual(compare_preparation(unknown, unknown)["status"], "unknown")
        with self.assertRaises(ValueError):require_matched_preparation(self.contract, unknown)

    def test_native_source_binding_and_boundary_mapping(self):
        sample = dict(schema="steady-cost-request-1", mode="full", repeats=8,
                      warmup=20, warmup_min_ms=500)
        request = dict(schema="dac-refinement-request-1", mode="steady", sampling=sample)
        first = refinement_preparation(request, source_hashes=IMPLEMENTATION_SOURCES)
        second = refinement_preparation(dict(request, mode="observer"), source_hashes=IMPLEMENTATION_SOURCES)
        result = compare_preparation(first, second)
        self.assertEqual(set(result["differences"]), {"untimed_complete_prime_passes"})
        self.assertFalse(result["allows_unbiased_observer_check"])
        self.assertEqual(refinement_preparation(request, source_hashes={}), PreparationContract())

    def test_missing_replays_remains_unknown_and_invalid_count_rejects(self):
        request = dict(schema="dac-refinement-request-1", mode="steady",
            sampling=dict(schema="steady-cost-request-1", mode="full", warmup=20, warmup_min_ms=500))
        contract = refinement_preparation(request, source_hashes=IMPLEMENTATION_SOURCES)
        self.assertIsNone(contract.timed_replays_per_batch)
        self.assertFalse(compare_preparation(self.contract, contract)["allows_unbiased_observer_check"])
        with self.assertRaises(ValueError):replace(self.contract, timed_replays_per_batch=0)


if __name__=="__main__":unittest.main()
