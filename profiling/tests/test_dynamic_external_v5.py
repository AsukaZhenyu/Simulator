import unittest

from profiling.dynamic_validation.external_v5 import validate_attention_stages


class AttentionStageTests(unittest.TestCase):
    def stage(self):
        return dict(begin_tick=10, graph_compute_enter_tick=11, graph_compute_return_tick=50,
                    extra_sync_enter_tick=51, extra_sync_return_tick=52, end_tick=53,
                    graph_compute_inclusive_ns=3900, extra_sync_ns=100,
                    wrapper_gaps_ns=300, host_wall_ns=4300,
                    original_graph_compute_internal_sync_retained=True,
                    device_span_added_to_host_wait=False)

    def test_original_sync_partition_closes(self):
        validate_attention_stages(self.stage(), 10_000_000)

    def test_device_span_not_added_again(self):
        stage = self.stage()
        stage["device_span_added_to_host_wait"] = True
        with self.assertRaises(ValueError):
            validate_attention_stages(stage, 10_000_000)

    def test_reversed_or_incomplete_host_stage_rejected(self):
        for key, value in (("extra_sync_enter_tick", 49), ("wrapper_gaps_ns", 0)):
            stage = self.stage()
            stage[key] = value
            with self.assertRaises(ValueError):
                validate_attention_stages(stage, 10_000_000)


if __name__ == "__main__":
    unittest.main()
