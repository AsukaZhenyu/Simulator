"""Post-run audit guards against incomplete, incomparable or contradictory data."""
from copy import deepcopy
import unittest

from experiments.strategy_comparison_campaign import verify_row_metrics, verify_equivalence_controls


class CampaignAuditTests(unittest.TestCase):
    def good(self):
        return dict(success=True,completed_makespan_ns=100,
            any_engine_event_span_ns=95,all_engine_event_uncovered_ns=5,
            cpu_event_span_ns=50,gpu_event_span_ns=70,cpu_gpu_event_overlap_ns=30,
            compute_event_span_ns=90,copy_event_span_ns=10,copy_compute_event_overlap_ns=5,
            copy_bytes_by_direction_and_role={'dram->vram':{'weight':10,'input':2},
                                              'vram->dram':{'output':4}},
            h2d_bytes=12,d2h_bytes=4)

    def test_valid_union_and_physical_traffic_accounting(self):
        verify_row_metrics(self.good())

    def test_failed_partial_time_is_not_completed_latency(self):
        verify_row_metrics(dict(success=False,completed_makespan_ns=None,makespan_ns=1))
        with self.assertRaises(ValueError):
            verify_row_metrics(dict(success=False,completed_makespan_ns=1))

    def test_overlap_and_payload_corruption_fail(self):
        for key,value in [('cpu_gpu_event_overlap_ns',51),('copy_compute_event_overlap_ns',11),
                          ('all_engine_event_uncovered_ns',6),('h2d_bytes',11),('d2h_bytes',5)]:
            row=self.good();row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_row_metrics(row)

    def test_equivalence_control_uses_observed_events_not_only_success(self):
        row=dict(scenario_id='control',candidate_id='e0-s0',success=True,status='valid',
                 completed_makespan_ns=100,event_sha256='same')
        ref=deepcopy(row);ref['scenario_id']='reference'
        declaration={'equivalence_controls':{'control':'reference'}}
        self.assertEqual(len(verify_equivalence_controls([row,ref],declaration)),1)
        ref['event_sha256']='different'
        with self.assertRaises(ValueError):
            verify_equivalence_controls([row,ref],declaration)


if __name__=='__main__':
    unittest.main()
