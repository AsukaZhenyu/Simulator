"""Preparation, size and causal-clock contracts, using declared test coefficients.

These are mathematical/model-contract checks, not hardware accuracy evidence.
"""
from dataclasses import replace
import math
from unittest import TestCase

from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter
from llm_infer_model.performance.prepared_transfer import COMPLETION_SEMANTIC
from llm_infer_model.performance.transfer_recovery import (
    AffineResponseImpact, SizeAwareRecoveryResponse, TransferRecoveryScope,
)
from llm_infer_model.performance.transfer_response import TransferResponse


EVIDENCE = ("explicit unit-test coefficients; not hardware calibration",)
CONTEXT = "a" * 64
PREPARATION = "b" * 64


class TransferRecoveryTests(TestCase):
    def model(self, *, qualified=True):
        return SizeAwareRecoveryResponse(
            TransferRecoveryScope(CONTEXT, PREPARATION, "dram->vram", "registered", 10000),
            TransferResponse("dram->vram", 100., 2., 2., None, 10, 100,
                             "registered", COMPLETION_SEMANTIC, EVIDENCE),
            AffineResponseImpact(40., .5, EVIDENCE),
            AffineResponseImpact(20., .1, EVIDENCE),
            1000., 200., EVIDENCE, qualified=qualified,
        )

    def predict(self, model=None, *, size=10, idle=0, copies=2, **kwargs):
        return (model or self.model()).predict_burst(size, idle, copies,
            context_sha256=CONTEXT, preparation_sha256=PREPARATION, **kwargs)

    def test_known_preparation_does_not_automatically_mean_zero_debt(self):
        result = self.predict(copies=1)
        self.assertEqual(result["initial_prepared_impact_ns"], 21.)
        self.assertEqual(result["rows"][0]["predicted_ns"], 141)
        self.assertEqual(result["baseline_completed_response_ns"], 120.)
        self.assertGreater(result["rows"][0]["state_before"], 0)

    def test_idle_fills_remaining_capacity_instead_of_double_adding_penalties(self):
        result = self.predict(idle=1000, copies=1)
        expected = math.ceil(120. + 45. - (45. - 21.) * math.exp(-1.))
        self.assertEqual(result["duration_ns"], expected)
        self.assertLessEqual(result["duration_ns"], 120 + 45)
        self.assertFalse(result["switching_and_idle_impacts_added_independently"])

    def test_recovery_uses_its_own_predicted_completion_clock(self):
        result = self.predict(copies=3)
        first, second, third = result["rows"]
        self.assertEqual(second["begin_predicted_ns"], first["predicted_ns"])
        self.assertEqual(third["begin_predicted_ns"], first["predicted_ns"] + second["predicted_ns"])
        expected_debt = first["state_before"] * math.exp(-first["predicted_ns"] / 200.)
        self.assertAlmostEqual(first["state_after"], expected_debt)
        self.assertEqual(second["state_before"], first["state_after"])
        self.assertGreater(first["predicted_ns"], third["predicted_ns"])

    def test_unmeasured_size_idle_and_preparation_are_rejected(self):
        for size in (9, 101):
            with self.assertRaises(UnknownParameter):
                self.predict(size=size)
        with self.assertRaises(UnknownParameter):
            self.predict(idle=10001)
        for context, preparation in (("c" * 64, PREPARATION), (CONTEXT, "c" * 64)):
            with self.assertRaises(UnknownParameter):
                self.model().predict_burst(10, 0, 1, context_sha256=context,
                                          preparation_sha256=preparation)

    def test_size_interpolation_changes_baseline_and_prepared_impact(self):
        result = self.predict(size=50, copies=1)
        self.assertEqual(result["baseline_completed_response_ns"], 200.)
        self.assertEqual(result["initial_prepared_impact_ns"], 25.)
        self.assertEqual(result["maximum_response_impact_ns"], 65.)
        self.assertEqual(result["duration_ns"], 225)

    def test_explicit_zero_preparation_is_supported_without_an_implicit_reset(self):
        model = replace(self.model(), prepared_impact=AffineResponseImpact(0., 0., EVIDENCE))
        zero = self.predict(model, copies=2)
        self.assertEqual(zero["duration_ns"], 240)
        with_idle = self.predict(model, idle=1000, copies=1)
        self.assertGreater(with_idle["duration_ns"], 120)

    def test_prepared_impact_must_not_exceed_capacity_anywhere_in_size_envelope(self):
        # The minimum size is valid while the maximum is invalid: checking only
        # one representative payload would accept a negative remaining capacity.
        with self.assertRaises(InvalidModel):
            replace(self.model(), prepared_impact=AffineResponseImpact(0., 1., EVIDENCE))
        with self.assertRaises(InvalidModel):
            replace(self.model(), maximum_impact=AffineResponseImpact(0., 0., EVIDENCE))

    def test_qualification_is_explicit_and_not_inherited_from_a_baseline_component(self):
        model = self.model(qualified=False)
        with self.assertRaises(UnknownParameter):
            self.predict(model)
        result = self.predict(model, require_qualified=False)
        self.assertFalse(result["profile_independently_qualified"])
        # Conversely, a separately unqualified baseline component may be a
        # parameter of an independently qualified composed law.
        self.assertFalse(self.model().baseline.qualified)
        self.assertTrue(self.predict()["profile_independently_qualified"])

    def test_device_event_boundary_and_cross_route_memory_reuse_are_rejected(self):
        for patch in (dict(boundary="CUDA event device interval"),
                      dict(route="vram->dram"), dict(host_memory_contract="pageable")):
            with self.assertRaises(InvalidModel):
                replace(self.model(), baseline=replace(self.model().baseline, **patch))

    def test_numerical_unknowns_and_overflow_are_not_silently_clipped(self):
        for value in (None, True, float("nan"), float("inf"), -1):
            with self.assertRaises(InvalidModel):
                AffineResponseImpact(value, 0., EVIDENCE)
        with self.assertRaises(InvalidModel):
            replace(self.model(), maximum_impact=AffineResponseImpact(1e308, 1e308, EVIDENCE))
        with self.assertRaises(InvalidModel):
            replace(self.model(), baseline=replace(self.model().baseline, fixed_ns=0.,
                                                  small_ns_per_byte=0., large_ns_per_byte=0.))

    def test_smooth_estimation_path_reuses_same_initialization_and_law(self):
        model = self.model(qualified=False)
        values = model.continuous_burst(10, 1000, 3, context_sha256=CONTEXT,
                                       preparation_sha256=PREPARATION)
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 120. + 45. - 24. * math.exp(-1.))
        result = self.predict(model, idle=1000, copies=3, require_qualified=False)
        self.assertEqual(result["rows"][0]["predicted_ns"], math.ceil(values[0]))
        self.assertGreater(values[0], values[-1])

    def test_no_observed_target_or_device_event_is_consumed_as_a_prediction_input(self):
        result = self.predict()
        self.assertFalse(result["target_outcome_used"])
        self.assertFalse(result["future_observed_state_used"])
        self.assertFalse(result["event_or_device_work_added"])
        self.assertFalse(result["physical_CPU_or_DMA_time_identified"])
        with self.assertRaises(TypeError):
            self.predict(observed_target_ns=999999)

    def test_invalid_or_unbounded_work_requests_are_rejected_in_both_paths(self):
        for copies in (0, True, 100001):
            with self.assertRaises(InvalidModel):
                self.predict(copies=copies)
            with self.assertRaises(InvalidModel):
                self.model().continuous_burst(10, 0, copies, context_sha256=CONTEXT,
                                              preparation_sha256=PREPARATION)
        for field in ("idle_fill_tau_ns", "active_recovery_tau_ns"):
            with self.assertRaises(InvalidModel):
                replace(self.model(), **{field: 0.})


class TransferRecoveryPublicTests(TestCase):
    def test_public_modeling_and_library_exports_reference_same_classes(self):
        from llm_infer_model import performance as modeling
        from llm_simulator import performance as library
        for name in ("AffineResponseImpact", "TransferRecoveryScope", "SizeAwareRecoveryResponse"):
            self.assertIs(getattr(modeling, name), getattr(library, name))
