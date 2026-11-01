"""Hand-calculated measured-state selection and causal-scope checks."""
from dataclasses import replace
import unittest

from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter
from llm_infer_model.performance.pcie_response import (PCIeState, PCIeResponseCell,
    PCIeResponseTable, PCIeStatePoint, PCIeStatePlan)
from llm_infer_model.performance.transfer_response import TransferResponse

CONTEXT = "a" * 64
G3, G4 = PCIeState(3, 8), PCIeState(4, 8)


def response(slope, *, qualified=True):
    return TransferResponse("dram->vram", 5, slope, slope, None, 1, 100,
                            "explicit registered arena", "inclusive COPY then existing sync",
                            ("independent scoped fixture",), qualified=qualified)


def table(*, qualified=True):
    # Measured slopes deliberately do NOT follow a theoretical 2x ratio.
    return PCIeResponseTable(CONTEXT, (PCIeResponseCell(G3, response(14, qualified=qualified)),
                                      PCIeResponseCell(G4, response(10, qualified=qualified))))


class PCIeResponseTests(unittest.TestCase):
    def test_uses_measured_state_without_generation_ratio(self):
        self.assertEqual(table().prediction(10, G3, context_sha256=CONTEXT)["duration_ns"], 145)
        result = table().prediction(10, G4, context_sha256=CONTEXT)
        self.assertEqual(result["duration_ns"], 105)
        self.assertEqual(result["components"]["fixed_ns"], 5)
        self.assertFalse(result["physical_DMA_service_identified"])
        self.assertFalse(result["device_activity_added_to_completed_response"])

    def test_unknown_state_context_and_size_do_not_fallback(self):
        for state, context in ((None, CONTEXT), (PCIeState(4, 4), CONTEXT), (G4, "b" * 64)):
            with self.subTest(state=state), self.assertRaises(UnknownParameter):
                table().prediction(10, state, context_sha256=context)
        with self.assertRaises(ValueError):
            table().prediction(101, G4, context_sha256=CONTEXT)

    def test_unqualified_fit_remains_explicit_candidate(self):
        with self.assertRaises(ValueError):
            table(qualified=False).prediction(10, G4, context_sha256=CONTEXT)
        result = table(qualified=False).prediction(10, G4, context_sha256=CONTEXT, require_qualified=False)
        self.assertFalse(result["qualified"])

    def test_constant_state_and_horizon(self):
        plan = PCIeStatePlan((PCIeStatePoint(0, G4), PCIeStatePoint(200, G3)), 500,
                             "declared_scenario", ("explicit input",))
        self.assertEqual(table().predict_at(10, plan, start_ns=10, context_sha256=CONTEXT)["end_ns"], 115)
        with self.assertRaises(UnknownParameter):
            table().predict_at(10, plan, start_ns=400, context_sha256=CONTEXT)

    def test_crossing_change_is_not_invented_physical_service(self):
        plan = PCIeStatePlan((PCIeStatePoint(0, G4), PCIeStatePoint(50, G3)), 500,
                             "declared_scenario", ("explicit input",))
        with self.assertRaisesRegex(UnknownParameter, "crosses"):
            table().predict_at(10, plan, start_ns=0, context_sha256=CONTEXT)

    def test_realized_future_requires_diagnostic_role(self):
        plan = PCIeStatePlan((PCIeStatePoint(0, G4),), 500, "diagnostic_replay", ("actual trace",))
        with self.assertRaisesRegex(InvalidModel, "diagnostic"):
            table().predict_at(10, plan, start_ns=0, context_sha256=CONTEXT)
        result = table().predict_at(10, plan, start_ns=0, context_sha256=CONTEXT, allow_diagnostic_replay=True)
        self.assertTrue(result["diagnostic_replay"])

    def test_only_known_prefix_can_supply_online_state(self):
        args = dict(observed_at_ns=1, known_at_ns=2, start_ns=5, maximum_age_ns=10,
                    context_sha256=CONTEXT, assume_constant_for_invocation=True)
        result = table().predict_from_observation(10, G4, **args)
        self.assertEqual(result["state_age_ns"], 4)
        self.assertFalse(result["future_target_observation_used"])
        with self.assertRaises(InvalidModel):
            table().predict_from_observation(10, G4, **dict(args, known_at_ns=6))
        with self.assertRaises(UnknownParameter):
            table().predict_from_observation(10, G4, **dict(args, maximum_age_ns=2))
        with self.assertRaises(UnknownParameter):
            table().predict_from_observation(10, G4, **dict(args, assume_constant_for_invocation=False))

    def test_duplicate_or_mixed_scopes_are_not_merged(self):
        with self.assertRaises(InvalidModel):
            PCIeResponseTable(CONTEXT, (PCIeResponseCell(G4, response(10)), PCIeResponseCell(G4, response(11))))
        with self.assertRaises(InvalidModel):
            PCIeResponseTable(CONTEXT, (PCIeResponseCell(G4, response(10)),
                PCIeResponseCell(G3, replace(response(14), host_memory_contract="pageable"))))
        with self.assertRaises(InvalidModel):
            PCIeState(True, 8)


if __name__ == "__main__":
    unittest.main()
