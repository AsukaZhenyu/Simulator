import unittest
from profiling.host_blackout_diagnosis import constant_prediction_oracle_bound


class ConstantOracleDiagnosticTests(unittest.TestCase):
    def test_separated_fast_and_pause_modes_cannot_be_constant(self):
        result=constant_prediction_oracle_bound([6_000_000]*13+[38_000_000]*3,.2)
        self.assertEqual(result['maximum_covered_observations'],13)
        self.assertFalse(result['ninety_percent_coverage_possible'])
        self.assertFalse(result['selected_prediction_returned'])
        self.assertTrue(result['not_a_model_parameter'])

    def test_compatible_observations_can_share_one_prediction(self):
        result=constant_prediction_oracle_bound([100,110,120],.2)
        self.assertEqual(result['maximum_covered_observations'],3)
        self.assertTrue(result['ninety_percent_coverage_possible'])

    def test_inclusive_boundary(self):
        result=constant_prediction_oracle_bound([100,150],.2)
        self.assertEqual(result['maximum_covered_observations'],2)

    def test_invalid_oracle_inputs(self):
        for durations,tolerance in (([],.2),([0],.2),([100],1),([True],.2)):
            with self.assertRaises(ValueError):constant_prediction_oracle_bound(durations,tolerance)


if __name__=='__main__':unittest.main()
