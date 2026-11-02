import unittest

from profiling.dynamic_validation.external_v5_analysis import physical, counter_difference


class AttentionAttributionTests(unittest.TestCase):
    def stage(self):
        return dict(begin_tick=10, end_tick=110, graph_compute_enter_tick=11, graph_compute_return_tick=105)

    def test_kernel_gaps_are_not_added_to_sync_wait(self):
        records = [dict(kind="kernel", timestamp_valid=True, begin_ns=2000, end_ns=3000, name="QK"),
                   dict(kind="kernel", timestamp_valid=True, begin_ns=6000, end_ns=7000, name="PV"),
                   dict(kind="runtime_api", timestamp_valid=True, begin_ns=3000, end_ns=10000, name="cudaStreamSynchronize")]
        result = physical(self.stage(), records, 10_000_000)
        self.assertTrue(result["supported"])
        self.assertEqual(result["kernel_activity_union_ns"], 2000)
        self.assertEqual(result["device_kernel_envelope_ns"], 5000)
        self.assertEqual(result["gaps_inside_device_envelope_ns"], 3000)
        self.assertEqual(result["runtime_synchronize_union_in_graph_compute_ns"], 7000)
        self.assertFalse(result["device_activity_added_to_host_wait"])

    def test_no_kernel_correlation_is_unsupported(self):
        self.assertFalse(physical(self.stage(), [], 10_000_000)["supported"])

    def test_cpu_counter_unknown_is_not_zero_or_OS_inference(self):
        result = counter_difference(dict(thread_user_100ns=None, thread_kernel_100ns=2, thread_cycles=5),
                                    dict(thread_user_100ns=8, thread_kernel_100ns=7, thread_cycles=9))
        self.assertIsNone(result["user_ns"])
        self.assertEqual(result["kernel_ns"], 500)
        self.assertEqual(result["cycles"], 4)
        self.assertFalse(result["cycles_converted_to_time"])


if __name__ == "__main__":
    unittest.main()
