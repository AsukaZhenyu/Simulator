"""Software evidence that workload exports do not contain timing labels."""
from copy import deepcopy
from fractions import Fraction
import unittest

from profiling.real_moe_validation.test_runtime_windows import fixture
from profiling.real_moe_validation.runtime_window_analysis import extract_windows,fit_prices,predict_inventory


class TimingFreeWindowWorkloadTests(unittest.TestCase):
    def test_observed_time_changes_do_not_change_workload_or_prediction(self):
        first=fixture();second=deepcopy(first)
        for event in second['runtime_windows']['events']:event['tick']*=7
        second['end_tick']*=7;second['prefill_end_tick']*=7
        a=extract_windows(first,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        b=extract_windows(second,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        self.assertEqual(b['total_ns'],a['total_ns']*7)
        self.assertEqual(a['workload_inventory'],b['workload_inventory'])
        prices=fit_prices([a],['software-only calibration fixture'])
        self.assertEqual(predict_inventory(a['workload_inventory'],prices)['total_ns'],
            predict_inventory(b['workload_inventory'],prices)['total_ns'])
        self.assertNotIn('observed_ns',str(b['workload_inventory']))


if __name__=='__main__':unittest.main()
