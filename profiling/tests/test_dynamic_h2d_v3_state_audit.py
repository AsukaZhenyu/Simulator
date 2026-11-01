import unittest

from profiling.dynamic_validation.h2d_v3_state_audit import activity_partition, preceding


class StateAuditTests(unittest.TestCase):
    def test_only_completed_prior_query_is_selected(self):
        samples = [{"query": "past"}, {"query": "future"}]
        self.assertIsNone(preceding(samples, [10, 30], 9))
        self.assertEqual(preceding(samples, [10, 30], 20), {"query": "past"})
        self.assertEqual(preceding(samples, [10, 30], 30), {"query": "future"})

    def test_wall_partition_is_not_runtime_wait_plus_device_span(self):
        row = {"begin_tick": 10, "end_tick": 20, "host_wall_ns": 1000,
               "physical": {"supported": True, "raw_correlated_records": [
                   {"kind": "memcpy", "timestamp_valid": True,
                    "begin_ns": 1200, "end_ns": 1800}]}}
        result = activity_partition(row, 10_000_000)
        self.assertEqual(result, {"host_entry_to_device_begin_ns": 200,
                                  "device_activity_span_ns": 600,
                                  "device_end_to_host_completion_ns": 200})
        self.assertEqual(sum(result.values()), row["host_wall_ns"])

    def test_outside_device_interval_rejected(self):
        row = {"begin_tick": 10, "end_tick": 20, "host_wall_ns": 1000,
               "physical": {"supported": True, "raw_correlated_records": [
                   {"kind": "memcpy", "timestamp_valid": True,
                    "begin_ns": 900, "end_ns": 1800}]}}
        with self.assertRaises(ValueError):
            activity_partition(row, 10_000_000)


if __name__ == "__main__":
    unittest.main()
