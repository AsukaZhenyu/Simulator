"""Fixed commands use the same services ledger as causal execution."""
from fractions import Fraction
import unittest

from llm_infer_model.tensor.engine import Action, STATUS_VALID, CODE_TENSOR_IN_USE
from llm_infer_model.tensor.environment import SimulationSession
from llm_infer_model.tensor.runtime import ExecutionCommand, dispatch_command
from llm_infer_model.tensor.services import ServiceSession, parse_services
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from test_services import scene, document, frac


class ExecutionRuntimeTests(unittest.TestCase):
    def test_envelope_rejects_inapplicable_engine_and_non_actions(self):
        for action in (Action("ADVANCE"), Action("COMPUTE", "op0", device="cpu"),
                       Action("READ_SSD", "x0")):
            with self.assertRaises(InvalidInput):
                ExecutionCommand(action, "ce")
        with self.assertRaises(InvalidInput):
            ExecutionCommand({"kind": "COPY"})
        for engine in ("auto", "cuda", True, 0):
            with self.assertRaises(Unsupported):
                ExecutionCommand(Action("COPY", "x0", src="dram", dst="vram"), engine)

    def test_explicit_engine_cannot_fall_back_to_legacy_copy(self):
        runtime = SimulationSession(scene())
        with self.assertRaisesRegex(Unsupported, "services-enabled"):
            dispatch_command(runtime, ExecutionCommand(Action("COPY", "x0", src="dram", dst="vram"), "ce"))
        self.assertEqual(runtime.result().evaluation.actions, ())

    def test_missing_engine_is_a_loud_modeled_failure(self):
        runtime = ServiceSession(scene(), parse_services(document()))
        result = dispatch_command(runtime, ExecutionCommand(Action("COPY", "x0", src="dram", dst="vram")))
        self.assertFalse(result.ok)
        self.assertIn("explicit ce/sm", result.error_message)
        self.assertEqual(runtime.service_document()["transfer_engines"], [])

    def test_fixed_program_hand_timeline_and_payload_budget(self):
        runtime = ServiceSession(scene(), parse_services(document()))
        commands = (
            ExecutionCommand(Action("COPY", "x0", src="dram", dst="vram"), "ce"),
            ExecutionCommand(Action("COPY", "x1", src="dram", dst="vram"), "sm"),
            ExecutionCommand(Action("ADVANCE")), ExecutionCommand(Action("ADVANCE")),
            ExecutionCommand(Action("COMPUTE", "op0", device="cpu")),
            ExecutionCommand(Action("COMPUTE", "op1", device="gpu")),
            ExecutionCommand(Action("ADVANCE")),
            ExecutionCommand(Action("COPY", "y0", src="dram", dst="vram"), "ce"),
            ExecutionCommand(Action("ADVANCE")),
        )
        for index, command in enumerate(commands):
            result = dispatch_command(runtime, command, source_index=index)
            self.assertTrue(result.ok, result.error_message)
        self.assertEqual(runtime.result().evaluation.status, STATUS_VALID)
        self.assertEqual(runtime.result().evaluation.makespan_ns, 154)
        self.assertEqual(runtime.result().evaluation.h2d_bytes, 144)
        self.assertEqual(runtime.result().action_sources, tuple(range(len(commands))))
        intervals = runtime.service_document()["link_service_intervals"]
        self.assertEqual([(row["start_ns"], row["end_ns"]) for row in intervals],
                         [(0, 32), (32, 80), (90, 154)])
        self.assertEqual(sum(frac(flow["served_bytes"]) for row in intervals for flow in row["flows"]), 144)
        self.assertTrue(all(sum(frac(flow["served_bytes"]) for flow in row["flows"])
            <= frac(row["link_budget_bytes"]) for row in intervals))

    def test_fixed_slot_last_reader_lock_is_not_bypassed_by_envelope(self):
        slot = dict(id="promotion", domain="vram", capacity_bytes=64, members=["x0", "x1"])
        runtime = ServiceSession(scene(), parse_services(document(slots=[slot])))
        for command in (
            ExecutionCommand(Action("COPY", "x0", src="dram", dst="vram"), "ce"),
            ExecutionCommand(Action("ADVANCE")),
            ExecutionCommand(Action("COMPUTE", "op0", device="gpu")),
        ):
            self.assertTrue(dispatch_command(runtime, command).ok)
        result = dispatch_command(runtime, ExecutionCommand(Action("EVICT", "x0", location="vram")))
        self.assertEqual(result.error_code, CODE_TENSOR_IN_USE)


if __name__ == "__main__":
    unittest.main()
