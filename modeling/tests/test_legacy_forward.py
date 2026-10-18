"""Physical forward equations used by, but not implemented in, profiling."""
import unittest

from llm_infer_model.performance import InvalidModel
from llm_infer_model.performance.legacy_forward import (
    kv_context_seconds, payload_transfer_seconds, streamed_pipeline_seconds,
)


class LegacyForwardTests(unittest.TestCase):
    def test_streamed_lane_adds_payload_per_layer_latency_and_one_overhead(self):
        service, total = streamed_pipeline_seconds(weight_bytes=500, layer_count=3,
            bytes_per_second=1000, layer_latency_s=.25, scheduler_overhead_s=.1)
        self.assertEqual(service, 1.25)
        self.assertEqual(total, 1.35)

    def test_streamed_zero_work_keeps_explicit_fixed_overhead(self):
        self.assertEqual(streamed_pipeline_seconds(weight_bytes=0, layer_count=0,
            bytes_per_second=1000, layer_latency_s=0, scheduler_overhead_s=.2), (0, .2))

    def test_payload_only_transfer_uses_actual_bytes_and_allows_empty_work(self):
        self.assertEqual(payload_transfer_seconds(storage_bytes=100, bytes_per_second=400), .25)
        self.assertEqual(payload_transfer_seconds(storage_bytes=0, bytes_per_second=400), 0)

    def test_kv_roundtrip_accounts_for_both_cache_tiers(self):
        storage, total = kv_context_seconds(one_way_bytes=300, cache_bytes=100,
            cached_bytes_per_second=1000, uncached_bytes_per_second=100,
            baseline_seconds=.5, pcie_seconds=.25)
        self.assertEqual(storage, 4.2)
        self.assertEqual(total, 4.95)

    def test_kv_cache_boundary_and_zero_history(self):
        kwargs = dict(cache_bytes=100, cached_bytes_per_second=1000,
            uncached_bytes_per_second=100, baseline_seconds=.5, pcie_seconds=.25)
        self.assertEqual(kv_context_seconds(one_way_bytes=100, **kwargs), (.2, .95))
        self.assertEqual(kv_context_seconds(one_way_bytes=0, **kwargs), (0, .75))

    def test_unknown_rates_and_negative_overheads_are_rejected(self):
        for rate in (None, 0, float("inf"), True):
            with self.subTest(rate=rate), self.assertRaises(InvalidModel):
                payload_transfer_seconds(storage_bytes=1, bytes_per_second=rate)
        with self.assertRaises(InvalidModel):
            streamed_pipeline_seconds(weight_bytes=1, layer_count=1,
                bytes_per_second=1, layer_latency_s=0, scheduler_overhead_s=-1)


if __name__ == "__main__":
    unittest.main()
