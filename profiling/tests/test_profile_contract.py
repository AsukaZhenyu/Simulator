"""Contract tests; no accelerator, backend build, or benchmark required."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_contract import ContractError, summarize_samples, validate_case, validate_measurement


def compute_case(**overrides):
    case = {"id": "ffn-up", "kind": "compute", "m": 9216, "k": 2560, "n": 1, "dtype": "q4_k"}
    return {**case, **overrides}


class CaseContractTests(unittest.TestCase):
    def test_ggml_quantization_checks_reduction_width_not_tokens_or_output_width(self):
        valid = validate_case(compute_case(m=33, n=1, dtype="Q6_K", backend="cuda0"))
        self.assertEqual(valid["dtype"], "q6_k")
        self.assertEqual(valid["backend"], "CUDA0")
        with self.assertRaises(ContractError):
            validate_case(compute_case(m=256, k=33, n=256))

    def test_floating_weights_do_not_require_quantization_blocks(self):
        for dtype in ("f32", "f16"):
            self.assertEqual(validate_case(compute_case(k=33, dtype=dtype))["k"], 33)

    def test_unsupported_cuda_indices_are_rejected_before_planning(self):
        for backend in ("CUDA1", "cuda12", "CUDA00"):
            with self.subTest(backend=backend), self.assertRaises(ContractError):
                validate_case(compute_case(backend=backend))

    def test_unsupported_precision_is_not_silently_substituted(self):
        for dtype in ("bf16", "q4_k_m", "q4_0", "int4", "", None):
            with self.subTest(dtype=dtype), self.assertRaises(ContractError):
                validate_case(compute_case(dtype=dtype))

    def test_shape_and_thread_counts_are_positive_integers(self):
        for field in ("m", "k", "n", "threads"):
            for value in (0, -1, True, 1.5, "256", None):
                with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                    validate_case(compute_case(**{field: value}))

    def test_transfer_needs_direction_memory_type_and_byte_count(self):
        case = {"id": "weights-h2d", "kind": "transfer", "direction": "H2D", "host_memory": "PINNED", "size_bytes": 4096}
        original = copy.deepcopy(case)
        self.assertEqual(validate_case(case)["direction"], "h2d")
        self.assertEqual(case, original)
        for field in ("id", "direction", "host_memory", "size_bytes"):
            incomplete = {key: value for key, value in case.items() if key != field}
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_case(incomplete)
        for field, bad in (("direction", "d2d"), ("host_memory", "auto"), ("size_bytes", False)):
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_case({**case, field: bad})

    def test_validators_do_not_mutate_nested_provenance(self):
        case = compute_case(source={"tensor_names": ["blk.3.ffn_up.weight"]})
        normalized = validate_case(case)
        normalized["source"]["tensor_names"].append("another")
        self.assertEqual(len(case["source"]["tensor_names"]), 1)


class StatisticsTests(unittest.TestCase):
    def test_units_produce_the_same_known_distribution(self):
        reference = summarize_samples([1000, 2000, 3000, 4000, 5000])
        for samples, unit in (([1, 2, 3, 4, 5], "us"), ([0.001, 0.002, 0.003, 0.004, 0.005], "ms"), ([0.000001, 0.000002, 0.000003, 0.000004, 0.000005], "s")):
            with self.subTest(unit=unit):
                actual = summarize_samples(samples, unit=unit)
                for key, expected in reference.items():
                    self.assertAlmostEqual(actual[key], expected)
        self.assertEqual(reference["median_ns"], 3000)
        self.assertEqual(reference["p10_ns"], 1400)
        self.assertEqual(reference["p90_ns"], 4600)
        self.assertAlmostEqual(reference["stdev_ns"], math.sqrt(2_500_000))

    def test_no_dropping_bad_or_missing_observations(self):
        for samples in ([], None, [1, None], [1, True], [1, "2"], [1, 0], [1, -2], [1, math.nan], [1, math.inf], [1, -math.inf]):
            with self.subTest(samples=samples), self.assertRaises(ContractError):
                summarize_samples(samples)

    def test_unknown_units_and_conversion_overflow_are_rejected(self):
        for unit in ("milliseconds", "NS", "", None):
            with self.subTest(unit=unit), self.assertRaises(ContractError):
                summarize_samples([1], unit=unit)
        with self.assertRaises(ContractError):
            summarize_samples([1e308], unit="s")

    def test_single_sample_has_no_observed_spread(self):
        result = summarize_samples([12.5])
        self.assertEqual(result["count"], 1)
        for key in ("median_ns", "mean_ns", "min_ns", "max_ns", "p10_ns", "p90_ns"):
            self.assertEqual(result[key], 12.5)
        self.assertEqual(result["stdev_ns"], 0)
        self.assertEqual(result["cv"], 0)

    def test_samples_remain_in_acquisition_order(self):
        samples = [9, 1, 5]
        result = summarize_samples(samples)
        self.assertEqual(samples, [9, 1, 5])
        self.assertEqual(result["mean_ns"], 5)
        self.assertEqual(result["stdev_ns"], 4)
        self.assertEqual(result["cv"], 0.8)


class MeasurementContractTests(unittest.TestCase):
    def test_cuda_unknown_workspace_remains_unknown(self):
        payload = {"case_id": "ffn-up", "backend": "CUDA0", "wall_ns": [100, 120], "expected_samples": 2, "workspace_bytes": None}
        normalized = validate_measurement(payload)
        self.assertIsNone(normalized["workspace_bytes"])
        self.assertEqual(normalized["summary"]["median_ns"], 110)
        self.assertNotIn("summary", payload)
        self.assertIsNone(validate_measurement({"wall_ns": [1]})["workspace_bytes"])

    def test_missing_samples_and_partial_runs_cannot_be_valid_costs(self):
        for payload in ({}, {"wall_ns": []}, {"wall_ns": [1], "repetitions": 2}, {"wall_ns": [1, 2], "expected_samples": 1}, {"wall_ns": [1], "status": "failed"}):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                validate_measurement(payload)

    def test_zero_workspace_requires_evidence_and_unknown_cannot_mean_zero(self):
        for source in (None, "unknown", "not_measured", "", "unspecified"):
            payload = {"wall_ns": [1], "workspace_bytes": 0}
            if source is not None:
                payload["workspace_source"] = source
            with self.subTest(source=source), self.assertRaises(ContractError):
                validate_measurement(payload)
        valid = validate_measurement({"wall_ns": [1], "workspace_bytes": 0, "workspace_source": "backend_allocation_audit"})
        self.assertEqual(valid["workspace_bytes"], 0)
        for value in (-1, False, 1.5):
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_measurement({"wall_ns": [1], "workspace_bytes": value})

    def test_positive_workspace_cannot_bypass_the_evidence_requirement(self):
        for source in (None, "", "   ", "unknown", "unknown_backend_private_allocations",
                       "unmeasured_allocator", "not_measured_cuda", "not_observed_allocations"):
            payload = {"wall_ns": [1], "workspace_bytes": 4096}
            if source is not None:
                payload["workspace_source"] = source
            with self.subTest(source=source), self.assertRaises(ContractError):
                validate_measurement(payload)
        measured = validate_measurement({"wall_ns": [1], "workspace_bytes": 4096,
                                         "workspace_source": "ggml_graph_plan"})
        self.assertEqual(measured["workspace_bytes"], 4096)
        self.assertEqual(measured["workspace_source"], "ggml_graph_plan")

    def test_binding_wrong_case_or_backend_is_rejected(self):
        with self.assertRaises(ContractError):
            validate_measurement({"case": compute_case(), "case_id": "another", "wall_ns": [1]})
        for backend in ("gpu", "CUDA1"):
            with self.subTest(backend=backend), self.assertRaises(ContractError):
                validate_measurement({"backend": backend, "wall_ns": [1]})
        for field, planned, measured in (("backend", "CPU", "CUDA0"), ("threads", 8, 16)):
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_measurement({"case": compute_case(**{field: planned}), field: measured, "wall_ns": [1]})


if __name__ == "__main__":
    unittest.main()
