"""Shared-pool public API, profiling provenance, and artifact integration."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from llm_simulator import AnalyticalCosts, Hardware, Model, Request, RequestFlow, Simulator, Workload
from llm_simulator.demo import make_demo
from llm_infer_profile.memory_contract import validate_memory_context


class MemoryContractTests(unittest.TestCase):
    def context(self, **changes):
        value = dict(memory_mode="shared", allocation_kind="shared_buffer",
            timing_boundary="shared_visibility", timing_source="unknown",
            duration_ns=None, bytes_transferred=0)
        value.update(changes)
        return value

    def test_unknown_shared_handoff_stays_unknown(self):
        value = self.context()
        self.assertIsNone(validate_memory_context(value)["duration_ns"])
        with self.assertRaises(ValueError):
            validate_memory_context(self.context(duration_ns=0))

    def test_measured_sync_is_distinct_from_payload_copy(self):
        value = validate_memory_context(self.context(timing_boundary="synchronize",
            timing_source="measured", duration_ns=72, evidence="raw/example.json"))
        self.assertEqual(value["duration_ns"], 72)
        self.assertEqual(value["bytes_transferred"], 0)
        with self.assertRaises(ValueError):
            validate_memory_context(self.context(timing_source="measured", duration_ns=0))

    def test_actual_copy_and_shared_visibility_cannot_be_confused(self):
        with self.assertRaises(ValueError):
            validate_memory_context(self.context(timing_boundary="actual_copy", bytes_transferred=16))
        with self.assertRaises(ValueError):
            validate_memory_context(self.context(memory_mode="separate"))
        value = self.context(allocation_kind="separate_buffers", timing_boundary="actual_copy",
            bytes_transferred=16, timing_source="declared", duration_ns=5)
        self.assertEqual(validate_memory_context(value)["duration_ns"], 5)


class SharedMemoryFacadeTests(unittest.TestCase):
    def shared(self, kind="dense"):
        demo, scheduling = make_demo(kind)
        hardware = Hardware.shared_memory(cpu=demo.hardware.cpu, gpu=demo.hardware.gpu,
            capacity_bytes=64 * 1024 * 1024,
            ssd_read_bandwidth_bytes_per_second=1e9,
            ssd_write_bandwidth_bytes_per_second=500e6)
        return Simulator(model=demo.model, hardware=hardware, workload=demo.workload,
                         cost_model=demo.cost_model), scheduling

    def test_shared_hardware_validation_and_legacy_default(self):
        demo, _ = make_demo()
        self.assertEqual(demo.hardware.memory_mode, "separate")
        shared, _ = self.shared()
        self.assertIsNone(shared.hardware.transfer_bandwidth_bytes_per_second)
        with self.assertRaises(ValueError):
            replace(demo.hardware, shared_capacity_bytes=1024)
        with self.assertRaises(ValueError):
            replace(shared.hardware, shared_runtime_reserved_bytes=100 * 1024 * 1024)

    def test_shared_dense_moe_inputs_results_and_physical_artifacts(self):
        for kind in ("dense", "moe"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                simulator, scheduling = self.shared(kind)
                with simulator.compile(scheduling=scheduling, directory=Path(folder)/"inputs") as compiled:
                    result = compiled.run()
                    self.assertTrue(result.success, result.summary())
                    summary = result.summary()
                    self.assertGreater(summary["peak_shared_bytes"], 0)
                    self.assertEqual(summary["physical_memory_peaks_bytes"], {"shared": summary["peak_shared_bytes"]})
                    self.assertEqual((summary["h2d_bytes"], summary["d2h_bytes"]), (0, 0))
                    self.assertGreater(summary["shared_visibility_count"], 0)
                    self.assertTrue(all(event.bytes_transferred == 0 for event in result.events if event.kind == "SHARE"))
                    metadata = compiled.metadata["memory_contract"]
                    self.assertFalse(metadata["same_buffer_handoff"]["measured"])
                    self.assertFalse(metadata["automatic_shared_bandwidth_contention"])
                    output = result.save(Path(folder)/"result")
                    states = json.loads((output/"states.json").read_text(encoding="utf-8"))
                    self.assertTrue(any("used_shared_bytes" in state for state in states["states"]))
                    self.assertTrue(any(state.get("storage_allocations") for state in states["states"]))

    def test_request_flow_shared_pool(self):
        simulator, scheduling = self.shared()
        flow = RequestFlow(requests=(Request("a", 2, 1), Request("b", 1, 1, arrival_ns=5)))
        simulator = Simulator(model=simulator.model, hardware=simulator.hardware,
            workload=flow, cost_model=simulator.cost_model)
        with simulator.compile(scheduling=scheduling) as compiled:
            result = compiled.run()
            self.assertTrue(result.success, result.summary())
            self.assertEqual(result.summary()["memory_mode"], "shared")
            self.assertEqual(result.summary()["h2d_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
