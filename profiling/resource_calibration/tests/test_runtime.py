"""Actual common-session resource response bridge; no native/hardware launch."""

from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

from profiling.resource_calibration.functional import ResourceTrace, TraceSegment, UnknownSupport
from profiling.resource_calibration.runtime import OperationRuntime, build_runtime
from profiling.resource_calibration.tests.test_functional import model_for, operation, state


def trace(*segments, semantics="external"):
    return ResourceTrace(tuple(TraceSegment(*item) for item in segments), semantics)


class CommonRuntimeTests(unittest.TestCase):
    def test_cpu_mid_operation_change_matches_exact_functional(self):
        op = operation("cpu")
        runtime = build_runtime(model_for(op), op, trace((0, 50, state()), (50, 1000, state(1))))
        self.assertTrue(runtime.start().ok)
        self.assertEqual(runtime.observe().t_ns, 0)
        self.assertEqual(runtime.advance().t_ns, 50)
        observation = runtime.observe()
        self.assertEqual((observation.cpu_service_rate_numerator, observation.cpu_service_rate_denominator), (1, 2))
        remaining, = runtime.result().compute_progress
        self.assertEqual(Fraction(remaining.remaining_reference_ns_numerator,
                                  remaining.remaining_reference_ns_denominator), 50)
        result = runtime.run()
        self.assertEqual(result.evaluation.makespan_ns, 150)
        self.assertEqual([event.target_id for event in result.evaluation.events if event.target_id is not None], [op.key])
        self.assertEqual([interval.resource for interval in result.service_intervals], ["cpu_compute", "cpu_compute"])

    def test_gpu_mid_operation_change_uses_gpu_coefficients_and_common_runtime(self):
        op = operation("gpu")
        fitted = model_for(op, coefficients=(100, 0, 100, 0, 0, 0))
        runtime = build_runtime(fitted, op, trace((0, 50, state()), (50, 1000, state(gpu=1))))
        self.assertTrue(runtime.start().ok)
        self.assertEqual(runtime.advance().t_ns, 50)
        observation = runtime.observe()
        self.assertEqual((observation.gpu_service_rate_numerator, observation.gpu_service_rate_denominator), (1, 2))
        self.assertEqual((observation.cpu_service_rate_numerator, observation.cpu_service_rate_denominator), (1, 1))
        result = runtime.run()
        self.assertEqual(result.evaluation.makespan_ns, 150)
        self.assertEqual([row.resource for row in result.service_intervals], ["gpu_compute", "gpu_compute"])

    def test_slow_fast_slow_integrates_exact_remaining_work(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 40, state(1)), (40, 80, state()), (80, 1000, state(1))))
        result = runtime.run()
        self.assertEqual(result.evaluation.makespan_ns, 160)
        self.assertEqual(sum(Fraction(item.reference_service_ns_numerator,
                                      item.reference_service_ns_denominator) for item in result.service_intervals), 100)

    def test_fractional_completion_rounds_at_common_runtime_completion(self):
        op = operation()
        fitted = model_for(op, coefficients=(100, 3, 0, 0, 0, 0))
        runtime = build_runtime(fitted, op, trace((0, 50, state(1)), (50, 1000, state())))
        self.assertEqual(runtime.run().evaluation.makespan_ns, 102)
        self.assertEqual(runtime.provenance()["functional_exact_elapsed_ns"], "10450/103")

    def test_nonzero_trace_origin_is_relative_to_runtime_zero(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((100, 150, state()), (150, 1000, state(1))))
        self.assertEqual(runtime.observe().t_ns, 0)
        self.assertEqual(runtime.run().evaluation.makespan_ns, 150)
        self.assertEqual(runtime.provenance()["origin_ns"], 100)

    def test_explicit_start_inside_trace_segment(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 100, state()), (100, 1000, state(1))), start_ns=50)
        self.assertEqual(runtime.run().evaluation.makespan_ns, 150)
        self.assertEqual(runtime.provenance()["origin_ns"], 50)

    def test_observation_contains_current_rates_only(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 50, state()), (50, 1000, state(1))))
        observation = runtime.observe()
        for name in ("environment_events", "segments", "trace", "future_rates", "functional_elapsed_ns"):
            self.assertFalse(hasattr(observation, name))
        self.assertEqual((observation.cpu_service_rate_numerator, observation.cpu_service_rate_denominator), (1, 1))
        evidence = runtime.provenance()
        self.assertNotIn("environment_events", evidence)
        self.assertNotIn("segments", evidence)
        self.assertEqual(len(evidence["trace_sha256"]), 64)

    def test_controlled_load_retains_diagnostic_semantics(self):
        op = operation("gpu", dram_bytes=None, vram_bytes=None, memory_footprint_complete=False)
        fitted = model_for(op, semantics="offered_load_controlled", coefficients=(100, 0, 100, 0, 0, 0))
        runtime = build_runtime(fitted, op, trace((0, 50, state()), (50, 1000, state(gpu=1)),
                                                semantics="offered_load_controlled"))
        self.assertEqual(runtime.run().evaluation.makespan_ns, 150)
        evidence = runtime.provenance()
        self.assertEqual(evidence["compute_semantics"], "offered_load")
        self.assertEqual(evidence["memory_semantics"], "owned_external_allocations")
        self.assertTrue(evidence["diagnostic_only"])
        self.assertFalse(evidence["physical_capacity_claimed"])
        self.assertIsNone(evidence["source_operation_footprint"]["private_workspace_bytes"])
        self.assertIsNone(evidence["source_operation_footprint"]["vram_bytes"])
        self.assertEqual(evidence["synthetic_fixture_memory"]["workspace_bytes"], 0)

    def test_even_external_complete_footprint_remains_timing_carrier_only(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 1000, state())))
        evidence = runtime.provenance()
        self.assertFalse(evidence["physical_capacity_claimed"])
        self.assertFalse(evidence["source_memory_states_used_in_carrier_ledger"])
        self.assertFalse(evidence["numerical_operation_executed"])
        self.assertFalse(evidence["native_hardware_executed"])
        self.assertFalse(evidence["heldout_used_for_fit"])
        self.assertFalse(evidence["hardware_accuracy_accepted"])

    def test_provenance_returns_a_copy(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 1000, state())))
        runtime.provenance()["operation"]["key"] = "other"
        self.assertEqual(runtime.provenance()["operation"]["key"], op.key)


class RuntimeGuardTests(unittest.TestCase):
    def test_copy_cannot_be_promoted_to_compute_rate(self):
        op = operation("copy")
        with self.assertRaisesRegex(ValueError, "link-specific"):
            build_runtime(model_for(op), op, trace((0, 1000, state())))

    def test_unknown_tail_is_rejected_before_session_construction(self):
        op = operation()
        with patch("llm_infer_model.tensor.environment.SimulationSession") as constructor:
            with self.assertRaisesRegex(UnknownSupport, "ends before"):
                build_runtime(model_for(op), op, trace((0, 10, state())))
            constructor.assert_not_called()

    def test_out_of_support_state_rejected_before_session_construction(self):
        op = operation()
        with patch("llm_infer_model.tensor.environment.SimulationSession") as constructor:
            with self.assertRaises(UnknownSupport):
                build_runtime(model_for(op), op, trace((0, 1000, state(dram=600))))
            constructor.assert_not_called()

    def test_exact_operation_and_semantics_cannot_be_substituted(self):
        op = operation()
        fitted = model_for(op)
        for replacement in (replace(op, resource="gpu"), replace(op, key="other"),
                            replace(op, constants=op.constants + (("tokens", "different"),))):
            with self.subTest(replacement=replacement), self.assertRaises(UnknownSupport):
                build_runtime(fitted, replacement, trace((0, 1000, state())))
        with self.assertRaisesRegex(UnknownSupport, "semantics"):
            build_runtime(fitted, op, trace((0, 1000, state()), semantics="offered_load_controlled"))

    def test_no_arbitrary_runtime_actions_or_pipeline_injection(self):
        with self.assertRaisesRegex(ValueError, "arbitrary runtime injection"):
            OperationRuntime(None, None, None, "{}")
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 1000, state())))
        for name in ("submit", "submit_control", "scenario", "add_operation", "add_environment_event", "set_policy"):
            self.assertFalse(hasattr(runtime, name))
        with self.assertRaisesRegex(ValueError, "before advancing"):
            runtime.advance()
        runtime.start()
        with self.assertRaisesRegex(ValueError, "already started"):
            runtime.start()

    def test_future_state_after_completion_does_not_change_price_or_memory(self):
        op = operation()
        runtime = build_runtime(model_for(op), op, trace((0, 100, state()), (100, 1000, state(dram=1000))))
        self.assertEqual(runtime.run().evaluation.makespan_ns, 100)
        self.assertEqual(len(runtime.result().environment_events), 1)


if __name__ == "__main__":
    unittest.main()
