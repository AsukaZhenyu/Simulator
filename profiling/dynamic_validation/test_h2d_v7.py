"""Pure protocol/host-ledger checks; no CUDA, native launch or fitting."""
import unittest

from . import h2d_v7 as campaign
from .h2d_v7_analysis import host_partition


class PrimitiveProtocolTests(unittest.TestCase):
    def test_sizes_idle_and_future_seed_are_explicit(self):
        p=campaign.protocol()
        self.assertFalse(set(p['split_sizes_bytes']['profile']) & set(p['split_sizes_bytes']['heldout']))
        self.assertFalse(set(p['split_idle_ms']['profile']) & set(p['split_idle_ms']['heldout']))
        self.assertTrue(p['independent_heldout_program_seed'])
        self.assertFalse(p['fit_uses_target_total'])

    def test_orders_are_balanced_both_directions(self):
        for n in (2,3,6):
            for reverse in (False,True):
                rows=campaign.balanced_orders(n,reverse)
                self.assertEqual(len(rows),12)
                for row in rows:self.assertEqual(sorted(row),list(range(n)))
                for slot in range(n):
                    for value in range(n):self.assertEqual(sum(row[slot]==value for row in rows),12//n)

    def sample(self):
        names=['cudaMemcpyAsync_API_response','inner_PTDS_sync_API_response','extra_GGML_GPU_sync_response','extra_GGML_CPU_sync_response']
        return dict(begin_tick=10,end_tick=110,wall_ns=100,inner_response_parts_available=True,
                    host_response_segments=[dict(name=name,begin_tick=a,end_tick=b,wall_ns=b-a) for name,a,b in zip(names,(15,40,80,95),(35,75,90,105))],
                    outside_segments_ticks=25,device_event_span_ns=None)

    def test_response_partition_closes_without_device_addition(self):
        r=host_partition(self.sample(),10**9)
        self.assertEqual(r['host_wall_ns'],100)
        self.assertEqual(sum(r['response_parts_ns'].values())+r['outside_segments_ns'],100)
        self.assertFalse(r['host_API_is_CPU_active'])
        self.assertFalse(r['GPU_event_QPC_position_known'])

    def test_overlapping_or_missing_host_segments_reject(self):
        a=self.sample();a['host_response_segments'][1]['begin_tick']=30
        with self.assertRaises(ValueError):host_partition(a,10**9)
        a=self.sample();a['outside_segments_ticks']=0
        with self.assertRaises(ValueError):host_partition(a,10**9)

    def test_bare_control_cannot_fabricate_inner_parts(self):
        a=dict(begin_tick=10,end_tick=110,wall_ns=100,inner_response_parts_available=False)
        self.assertFalse(host_partition(a,10**9)['parts_supported'])
        a['device_event_span_ns']=50
        with self.assertRaises(ValueError):host_partition(a,10**9)

    def test_predeclared_accuracy_and_stability_both_required(self):
        a=campaign.protocol()['acceptance']
        self.assertEqual(a['observer_relative_difference_limit'],.05)
        self.assertEqual(a['profile_early_late_suffix_relative_difference_limit'],.10)
        self.assertEqual(a['heldout_mean_relative_error_limit'],.20)
        self.assertEqual(a['heldout_measured_population_cv_limit'],.10)


if __name__=='__main__':unittest.main()
