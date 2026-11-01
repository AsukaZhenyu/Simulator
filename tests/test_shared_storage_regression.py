"""Regression auditing must reject missing results and altered event traces."""
from copy import deepcopy
import unittest

from experiments.shared_storage_regression import compare_records


class RegressionComparisonTests(unittest.TestCase):
    def rows(self):
        return [dict(scenario_id="s", candidate_id="e0-s0", success=True,
            completed_makespan_ns=5, peak_dram_bytes=9, event_sha256="x",
            declared_inputs_sha256="old", compiled_ir_audit_sha256="old", harness_wall_seconds=1)]

    def test_optional_metadata_change_does_not_change_execution(self):
        rows = self.rows()
        new = deepcopy(rows)
        new[0].update(declared_inputs_sha256="new", compiled_ir_audit_sha256="new", harness_wall_seconds=3)
        self.assertTrue(compare_records(rows, new, expected=1)["passed"])

    def test_changed_time_peak_or_event_is_a_regression(self):
        for key, value in (("completed_makespan_ns", 6), ("peak_dram_bytes", 10), ("event_sha256", "y")):
            new = self.rows()
            new[0][key] = value
            self.assertFalse(compare_records(self.rows(), new, expected=1)["passed"])

    def test_partial_matrix_cannot_pass(self):
        self.assertFalse(compare_records(self.rows(), [], expected=1)["passed"])
        self.assertFalse(compare_records(self.rows(), self.rows(), expected=352)["passed"])

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            compare_records(self.rows() * 2, self.rows(), expected=1)


if __name__ == "__main__":
    unittest.main()
