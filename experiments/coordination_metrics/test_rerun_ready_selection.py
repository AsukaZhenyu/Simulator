"""Re-run certificates never weaken a frozen bound or hide a gap."""
import unittest

from experiments.coordination_metrics.rerun_ready_selection import refresh_certificate, coverage


def row():
    return dict(id='x', audit_status='pass', T_C_ns=130, certificate=dict(
        lower_bound_ns=110, upper_bound_ns=125, g=None, T_star_ns=None,
        feasible_optimal_or_upper_witness=dict(kind='old', duration_ns=125, events=[]),
        g_interval=[(130 - 125) / 125, (130 - 110) / 110]))


class RerunCertificateTests(unittest.TestCase):
    def test_slower_trace_keeps_the_frozen_upper_bound_and_records_the_bigger_gap(self):
        record = row()
        certificate = refresh_certificate(record, 140, [{'id': 'x'}])
        self.assertEqual(certificate['upper_bound_ns'], 125)
        self.assertFalse(certificate['certified'])
        self.assertEqual(certificate['feasible_optimal_or_upper_witness']['kind'], 'old')
        self.assertEqual(certificate['g_interval'], [(140 - 125) / 125, (140 - 110) / 110])
        self.assertFalse(certificate['rerun_tightened_upper_bound'])

    def test_faster_trace_becomes_the_new_feasible_witness_and_can_certify(self):
        trace = [dict(id='x', machine='cpu', start_ns=0, end_ns=110)]
        certificate = refresh_certificate(row(), 110, trace)
        self.assertTrue(certificate['certified'])
        self.assertEqual(certificate['T_star_ns'], 110)
        self.assertEqual(certificate['g'], 0.0)
        self.assertEqual(certificate['feasible_optimal_or_upper_witness']['kind'], 'rerun_compiler_trace')
        self.assertEqual(certificate['g_interval'], [0.0, 0.0])

    def test_a_trace_below_the_valid_lower_bound_is_a_contract_error_not_a_negative_gap(self):
        with self.assertRaises(AssertionError):
            refresh_certificate(row(), 100, [{'id': 'x'}])

    def test_coverage_counts_improvement_against_the_frozen_value(self):
        rows = [dict(id='a', audit_status='pass', T_C_ns=10, frozen_T_C_ns=12,
                     certificate=dict(g_interval=[0.0, 0.0])),
                dict(id='b', audit_status='pass', T_C_ns=12, frozen_T_C_ns=12,
                     certificate=dict(g_interval=[0.0, 0.0]))]
        counts = coverage(rows)
        self.assertEqual((counts['improved'], counts['unchanged'], counts['regressed']), (1, 1, 0))
        self.assertEqual(counts['rerun_policies'], 2)


if __name__ == '__main__':
    unittest.main()
