import unittest

from profiling.physical_cuda_waits import diagnose_cuda_waits


def record(kind, correlation, begin, end, name=None, **extra):
    return dict(kind=kind, correlation_id=correlation, begin_ns=begin, end_ns=end,
                timestamp_valid=True, name=name, **extra)


def action(kernels, runtime, **extra):
    return dict(iteration=0, action_index=0, external_id=1, begin_ns=1, end_ns=1000,
                selected_action_attribution_qualified=True, kernel_records=kernels, runtime_api_records=runtime,
                driver_api_records=[], memcpy_records=[], **extra)


def parsed(actions, **extra):
    return dict(schema="physical-stage-attribution-1", actions=actions,
                capture_qualified=True, timestamp_attribution_fallback_used=False, **extra)


class CudaWaitDiagnosticTests(unittest.TestCase):
    def test_exact_launch_pair_reports_lower_bound_and_descriptive_tail(self):
        row = action([record("kernel", 11, 40, 80)], [
            record("runtime_api", 11, 10, 20, "cudaLaunchKernel_v7000", return_value=0),
            record("runtime_api", 12, 25, 100, "cudaStreamSynchronize", return_value=0)])
        result = diagnose_cuda_waits(parsed([row]))
        out = result["actions"][0]
        pair = out["launch_kernel_pairs"][0]
        self.assertEqual(pair["launch_api_span_ns"], 10)
        self.assertEqual(pair["queue_after_launch_lower_bound_ns"], 20)
        self.assertEqual(out["synchronization_completion"]["post_last_kernel_tail_ns"], 20)
        self.assertEqual(out["sync_API_time_overlapping_attributed_device_activity_ns"], 40)
        self.assertEqual(out["sync_API_time_without_attributed_device_activity_ns"], 35)
        self.assertFalse(out["synchronization_completion"]["pure_scheduler_latency_identified"])

    def test_kernel_inside_launch_preserves_negative_gap_and_overlap(self):
        row = action([record("kernel", 1, 20, 60)], [record("runtime_api", 1, 10, 40, "cudaLaunchKernel", return_value=0)])
        pair = diagnose_cuda_waits(parsed([row]))["actions"][0]["launch_kernel_pairs"][0]
        self.assertEqual(pair["signed_kernel_begin_minus_launch_end_ns"], -20)
        self.assertEqual(pair["queue_after_launch_lower_bound_ns"], 0)
        self.assertEqual(pair["launch_api_and_kernel_overlap_ns"], 20)
        self.assertTrue(pair["kernel_begins_before_launch_api_return"])
        self.assertTrue(pair["causal_order_consistent"])

    def test_no_time_based_or_cross_action_attribution(self):
        first = action([], [record("runtime_api", 1, 10, 20, "cudaLaunchKernel", return_value=0)])
        second = action([record("kernel", 1, 21, 30), record("kernel", 2, 22, 31)], [])
        result = diagnose_cuda_waits(parsed([first, second]))
        self.assertEqual(result["counts"]["exact_launch_kernel_pairs"], 0)
        self.assertEqual(result["counts"]["unpaired_kernel_records"], 2)
        self.assertEqual(result["counts"]["unpaired_launch_api_records"], 1)
        self.assertFalse(result["timestamp_attribution_fallback_used"])

    def test_ambiguous_launch_id_is_not_picked_by_closest_time(self):
        row = action([record("kernel", 1, 50, 60)], [
            record("runtime_api", 1, 10, 20, "cudaLaunchKernel", return_value=0),
            record("runtime_api", 1, 30, 40, "cudaLaunchKernelExC", return_value=0)])
        result = diagnose_cuda_waits(parsed([row]))
        self.assertEqual(result["counts"]["exact_launch_kernel_pairs"], 0)
        self.assertEqual(result["counts"]["unpaired_launch_api_records"], 2)
        self.assertIn("ambiguous", result["actions"][0]["unpaired_kernel_records"][0]["reason"])

    def test_overlapping_runtime_device_driver_are_not_added(self):
        row = action([record("kernel", 1, 20, 60), record("kernel", 2, 40, 80)], [
            record("runtime_api", 1, 10, 30, "cudaLaunchKernel", return_value=0),
            record("runtime_api", 2, 20, 40, "cudaLaunchKernel", return_value=0),
            record("runtime_api", 3, 35, 100, "cudaDeviceSynchronize", return_value=0)])
        row["driver_api_records"] = [record("driver_api", 1, 12, 29, "cuLaunchKernel")]
        out = diagnose_cuda_waits(parsed([row]))["actions"][0]
        self.assertEqual(out["launch_api_union_ns"], 30)
        self.assertEqual(out["runtime_api_union_ns"], 90)
        self.assertEqual(out["kernel_activity_union_ns"], 60)
        self.assertEqual(out["sync_API_time_overlapping_attributed_device_activity_ns"], 45)
        self.assertFalse(out["timing_columns_additive"])

    def test_final_sync_before_last_kernel_is_not_called_completion(self):
        row = action([record("kernel", 1, 40, 80)], [
            record("runtime_api", 1, 10, 20, "cudaLaunchKernel", return_value=0),
            record("runtime_api", 2, 30, 60, "cudaEventSynchronize", return_value=0)])
        tail = diagnose_cuda_waits(parsed([row]))["actions"][0]["synchronization_completion"]
        self.assertEqual(tail["last_kernel_end_to_final_sync_return_signed_ns"], -20)
        self.assertTrue(tail["final_sync_returns_before_last_kernel"])
        self.assertIsNone(tail["post_last_kernel_tail_ns"])
        self.assertFalse(tail["kernel_to_sync_dependency_proven"])

    def test_orphans_invalid_times_and_nonnamed_apis_remain_visible(self):
        orphan = dict(record=record("kernel", 99, 10, 20), reason="no_exact_external_correlation")
        invalid = dict(record=record("runtime_api", 100, 20, 10), reason="invalid time")
        row = action([], [record("runtime_api", 1, 10, 20)])
        result = diagnose_cuda_waits(parsed([row], orphan_records=[orphan], invalid_time_records=[invalid]))
        self.assertEqual(result["orphan_records"], [orphan])
        self.assertEqual(result["invalid_time_records"], [invalid])
        self.assertEqual(len(result["actions"][0]["other_runtime_api_records"]), 1)
        self.assertFalse(result["available_resource_rate_inferred"])

    def test_unqualified_capture_never_promotes_pair(self):
        row = action([record("kernel", 1, 40, 80)], [record("runtime_api", 1, 10, 20, "cudaLaunchKernel", return_value=0)])
        source = parsed([row])
        source["capture_qualified"] = False
        result = diagnose_cuda_waits(source)
        self.assertFalse(result["actions"][0]["launch_kernel_pairs"][0]["pair_metrology_qualified"])
        self.assertFalse(result["hardware_accuracy_qualified"])

    def test_timestamp_fallback_input_rejected(self):
        source = parsed([])
        source["timestamp_attribution_fallback_used"] = True
        with self.assertRaisesRegex(ValueError, "timestamp attribution"):
            diagnose_cuda_waits(source)


if __name__ == "__main__":
    unittest.main()
