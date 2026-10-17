from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from profiling.reconfiguration import run as r


def allocator_fixture():
    # Synthetic unit-test fixture, never a measurement artifact or cost entry.
    case = {"id": "unit-test", "bytes": 4096, "device": 0, "warmup": 1, "samples": 3, "seed": 17}
    payload = dict(case)
    payload.pop("id")
    payload.update(schema="reconfiguration-allocator-native-0.1", device_name="unit-test-placeholder",
                   cuda_driver_version=12000, cuda_runtime_version=12000,
                   host_memory="cudaMallocHost_pinned",
                   device_allocator="direct_cudaMalloc_cudaFree_not_GGML_or_async_pool",
                   timer="steady_clock_completed_api_wall", cuda_graphs="not_used",
                   validation="all_bytes_exact_each_iteration", workspace_bytes=None,
                   logical_evict_cost_ns=None, mapper_control_cost_ns=None,
                   allocator_release_guaranteed_by_observed_free_delta=False,
                   wall_ns={key: [100, 110, 120] for key in r.ALLOC_METRICS},
                   memory_checkpoints=[dict(before_free_bytes=10000, allocated_free_bytes=5000,
                                           after_free_bytes=9000, total_bytes=16000) for _ in range(3)])
    return case, payload


class ContractTests(unittest.TestCase):
    def test_declared_suite_valid(self):
        self.assertEqual(r.validate_suite(r.matched.document(r.ROOT / "suite.json"))["format_version"], r.VERSION)

    def test_no_silent_missing_field(self):
        suite = r.matched.document(r.ROOT / "suite.json")
        del suite["run_context"]
        with self.assertRaises(ValueError):
            r.validate_suite(suite)

    def test_duplicate_case_id_rejected(self):
        suite = r.matched.document(r.ROOT / "suite.json")
        suite["allocator_cases"][1]["id"] = suite["allocator_cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            r.validate_suite(suite)

    def test_cpu_fallback_is_not_a_resident_reload_pair(self):
        suite = r.matched.document(r.ROOT / "suite.json")
        suite["ffn_suite"]["cases"][0]["backend"] = "cpu"
        with self.assertRaisesRegex(ValueError, "hybrid"):
            r.validate_suite(suite)

    def test_allocator_signature_binding(self):
        case, payload = allocator_fixture()
        payload["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "binding"):
            r.validate_allocator(case, payload)

    def test_boolean_device_is_not_a_native_index(self):
        case, payload = allocator_fixture()
        payload["device"] = False
        with self.assertRaises(ValueError):
            r.validate_allocator(case, payload)

    def test_boolean_provenance_is_not_numeric_zero(self):
        case, payload = allocator_fixture()
        payload["allocator_release_guaranteed_by_observed_free_delta"] = 0
        with self.assertRaises(ValueError):
            r.validate_allocator(case, payload)

    def test_unknown_workspace_cannot_be_filled_zero(self):
        case, payload = allocator_fixture()
        payload["workspace_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "semantics"):
            r.validate_allocator(case, payload)

    def test_logical_evict_cannot_be_claimed_measured(self):
        case, payload = allocator_fixture()
        payload["logical_evict_cost_ns"] = 100
        with self.assertRaisesRegex(ValueError, "semantics"):
            r.validate_allocator(case, payload)

    def test_checkpoints_are_observations_not_guaranteed_returns(self):
        case, payload = allocator_fixture()
        payload["memory_checkpoints"][0]["after_free_bytes"] = 4000
        self.assertIs(r.validate_allocator(case, payload), payload)

    def test_impossible_memory_checkpoints_rejected(self):
        case, payload = allocator_fixture()
        payload["memory_checkpoints"][0]["before_free_bytes"] = 16001
        with self.assertRaisesRegex(ValueError, "impossible"):
            r.validate_allocator(case, payload)

    def test_all_native_samples_required(self):
        case, payload = allocator_fixture()
        payload["wall_ns"]["cuda_malloc"].pop()
        with self.assertRaises(ValueError):
            r.validate_allocator(case, payload)

    def test_boolean_is_not_integer_timing(self):
        case, payload = allocator_fixture()
        payload["wall_ns"]["cuda_malloc"][0] = True
        with self.assertRaises(ValueError):
            r.validate_allocator(case, payload)

    def test_unknown_mapper_control_must_remain_unknown(self):
        case, payload = allocator_fixture()
        payload["mapper_control_cost_ns"] = 100
        with self.assertRaises(ValueError):
            r.validate_allocator(case, payload)

    def test_degenerate_original_input_rejected(self):
        payload = {"source_values": {name: [i / 2048 for i in range(16)] for name in
                                    ("input", "gate_weight", "up_weight", "down_weight")}}
        self.assertFalse(any(c["passed"] for c in r.nondegenerate(payload).values()))

    def test_fine_f32_fixture_has_actual_variation(self):
        payload = {"source_values": {name: [(i + .1234567) / 97.031 for i in range(16)] for name in
                                    ("input", "gate_weight", "up_weight", "down_weight")}}
        self.assertTrue(all(c["passed"] for c in r.nondegenerate(payload).values()))

    def test_device_mismatch_rejected(self):
        suite = r.matched.document(r.ROOT / "suite.json")
        suite["allocator_cases"][0]["device"] = 1
        with self.assertRaisesRegex(ValueError, "device"):
            r.validate_suite(suite)

    def test_quality_thresholds_not_implicitly_relaxed(self):
        suite = r.matched.document(r.ROOT / "suite.json")
        suite["ffn_suite"]["quality"]["max_within_process_cv"] = 0
        with self.assertRaises(ValueError):
            r.validate_suite(suite)


if __name__ == "__main__":
    unittest.main()
