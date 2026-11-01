"""Shared physical occupancy, reclaim and fixed-slot integration acceptance."""
from dataclasses import replace
import unittest

from test_shared_storage_core import documents, scene, share, compute
from test_services import document as services_document
from llm_infer_model.tensor.engine import Action, used_bytes
from llm_infer_model.tensor.environment import (
    SimulationSession, EnvironmentEvent, EnvironmentReclaimRequest, ControlCosts,
)
from llm_infer_model.tensor.runtime import ControlCommand
from llm_infer_model.tensor.services import ServiceSession, parse_services
from llm_infer_model.tensor.spec import build_scenario, InvalidInput
from llm_infer_model.tensor.program import execute_program, ActionStep, WaitStep


class SharedEnvironmentTests(unittest.TestCase):
    def test_external_occupancy_is_one_pool_and_recovers(self):
        runtime = SimulationSession(scene(), environment_events=(
            EnvironmentEvent(0, shared_external_bytes=17),
            EnvironmentEvent(3, shared_external_bytes=1)))
        self.assertTrue(runtime.submit(share()).ok)
        self.assertTrue(runtime.submit(compute()).ok)
        self.assertEqual(runtime.observe().memory[0].used_bytes, 81)
        self.assertEqual(runtime.advance().t_ns, 3)
        self.assertEqual(runtime.observe().memory[0].used_bytes, 65)
        self.assertTrue(runtime.advance().ok)
        result = runtime.result()
        self.assertTrue(runtime.finished)
        self.assertEqual(result.evaluation.peak_shared_bytes, 81)
        self.assertEqual(result.evaluation.states[-1].used_shared_bytes, 65)
        self.assertEqual(result.environment_events[0].shared_external_bytes, 17)
        self.assertTrue(all(m.physical_pool == "shared" for m in runtime.observe().memory))

    def test_ambiguous_external_inputs_rejected(self):
        for event in (EnvironmentEvent(0, dram_external_bytes=3),
                      EnvironmentEvent(0, vram_external_bytes=3)):
            with self.assertRaisesRegex(InvalidInput, "shared_external_bytes"):
                SimulationSession(scene(), environment_events=(event,))
        with self.assertRaises(InvalidInput):
            SimulationSession(scene(mode="separate"), environment_events=(EnvironmentEvent(0, shared_external_bytes=3),))

    def test_external_overcommit_is_atomic(self):
        runtime = SimulationSession(scene(capacity=64), environment_events=(EnvironmentEvent(0, shared_external_bytes=33),))
        self.assertEqual(runtime.error_code, "external_memory_conflict")
        self.assertEqual(runtime.observe().memory[0].external_bytes, 0)
        self.assertFalse(runtime.result().environment_events[0].accepted)

    def test_both_compute_workspaces_and_outputs_share_capacity(self):
        runtime = SimulationSession(scene(sizes=(16, 16), workspace=8),
            environment_events=(EnvironmentEvent(0, shared_external_bytes=9),))
        self.assertTrue(runtime.submit(share("x1")).ok)
        self.assertTrue(runtime.submit(compute(0, "cpu")).ok)
        self.assertTrue(runtime.submit(compute(1, "gpu")).ok)
        memory = runtime.observe().memory[0]
        self.assertEqual((memory.tensor_bytes, memory.workspace_bytes, memory.external_bytes, memory.used_bytes),
                         (64, 16, 9, 89))
        self.assertTrue(runtime.advance().ok)
        self.assertEqual(runtime.observe().memory[0].workspace_bytes, 0)

    def reclaim_scene(self, initial):
        workload, config = documents(capacity=64, initial=[initial])
        workload["tensors"][0]["role"] = "weight"
        return build_scenario(workload, config)

    def reclaim_runtime(self, initial):
        return SimulationSession(self.reclaim_scene(initial), reclaim_requests=(
            EnvironmentReclaimRequest("r", 0, 48, 20, domain="shared"),),
            control_costs=ControlCosts(0, 0, 0, "declared-test"))

    def test_alias_drop_cannot_pretend_to_release_physical_memory(self):
        runtime = self.reclaim_runtime(["dram", "vram"])
        self.assertEqual(runtime.observe().safe_weight_release_candidates, ())
        result = runtime.submit(Action("EVICT", "x0", location="vram"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "reclaim_admission_blocked")
        self.assertEqual(used_bytes(runtime.scenario, runtime.state, "shared"), 32)

    def test_ack_rechecks_physical_capacity_even_without_eviction(self):
        runtime = self.reclaim_runtime(["dram", "vram"])
        self.assertTrue(runtime.submit_control(ControlCommand("ACK_RECLAIM", request_id="r")).ok)
        self.assertEqual(runtime.result().reclaim_records[0].status, "capacity_rejected")
        self.assertEqual(runtime.observe().memory[0].external_bytes, 0)

    def test_last_view_with_ssd_backup_releases_and_ack_succeeds(self):
        runtime = self.reclaim_runtime(["dram", "ssd"])
        self.assertEqual(runtime.observe().safe_weight_release_candidates, ("x0",))
        self.assertTrue(runtime.submit(Action("EVICT", "x0", location="dram")).ok)
        self.assertEqual(runtime.result().evaluation.events[0].bytes_released, 32)
        self.assertTrue(runtime.submit_control(ControlCommand("ACK_RECLAIM", request_id="r")).ok)
        self.assertEqual(runtime.result().reclaim_records[0].status, "accepted")
        self.assertEqual(runtime.observe().memory[0].used_bytes, 48)

    def test_reclaim_domain_must_match_architecture(self):
        with self.assertRaises(InvalidInput):
            SimulationSession(scene(), reclaim_requests=(EnvironmentReclaimRequest("r", 0, 1, 2),),
                              control_costs=ControlCosts(0, 0, 0, "test"))

    def test_ssd_backed_aliases_can_be_released_in_two_steps(self):
        runtime = self.reclaim_runtime(["dram", "vram", "ssd"])
        for location, released in (("vram", 0), ("dram", 32)):
            self.assertTrue(runtime.submit(Action("EVICT", "x0", location=location)).ok)
            self.assertEqual(runtime.result().evaluation.events[-1].bytes_released, released)
        self.assertTrue(runtime.submit_control(ControlCommand("ACK_RECLAIM", request_id="r")).ok)
        self.assertEqual(runtime.result().reclaim_records[0].status, "accepted")

    def test_program_preserves_release_evidence(self):
        value = scene()
        result = execute_program(value, (ActionStep(share(), 0),
            ActionStep(Action("EVICT", "x0", location="dram"), 1),
            ActionStep(compute(), 2), WaitStep(drain=True, source_index=3)))
        self.assertEqual(result.evaluation.status, "valid")
        self.assertEqual(result.evaluation.events[1].bytes_released, 0)
        self.assertEqual(result.evaluation.peak_shared_bytes, 64)


class SharedSlotTests(unittest.TestCase):
    def runtime(self):
        slot = dict(id="shared-slot", domain="shared", capacity_bytes=32, members=["x0", "x1"])
        value = scene(sizes=(32, 32), initial=[["dram", "ssd"], ["ssd"]], capacity=256)
        return ServiceSession(value, parse_services(services_document(slots=[slot])))

    def test_slot_reserved_once_and_alias_keeps_owner(self):
        runtime = self.runtime()
        self.assertEqual(runtime.observe().memory[0].used_bytes, 32)
        self.assertTrue(runtime.submit(share()).ok)
        self.assertTrue(runtime.submit(Action("EVICT", "x0", location="dram")).ok)
        self.assertEqual(runtime.public_services(("x0", "x1")).slots[0].owner, "x0")
        self.assertEqual(runtime.service_document()["slots"][0]["owner"], "x0")
        self.assertEqual(runtime.public_services(("x0", "x1")).flows, ())
        failure = runtime.submit(Action("READ_SSD", "x1"))
        self.assertEqual(failure.error_code, "SLOT_OCCUPIED")

    def test_all_views_released_before_slot_reuse(self):
        runtime = self.runtime()
        self.assertTrue(runtime.submit(share()).ok)
        for location in ("dram", "vram"):
            self.assertTrue(runtime.submit(Action("EVICT", "x0", location=location)).ok)
        self.assertTrue(runtime.submit(Action("READ_SSD", "x1")).ok)
        self.assertEqual(runtime.public_services(("x1",)).slots[0].owner, "x1")

    def test_real_copy_cannot_create_two_buffers_in_one_slot(self):
        runtime = self.runtime()
        result = runtime.submit_transfer(Action("COPY", "x0", src="dram", dst="vram"), engine="ce")
        self.assertEqual(result.error_code, "SLOT_OCCUPIED")

    def test_cpu_reader_locks_shared_slot_gpu_view(self):
        runtime = self.runtime()
        self.assertTrue(runtime.submit(share()).ok)
        self.assertTrue(runtime.submit(compute(device="cpu")).ok)
        result = runtime.submit(Action("EVICT", "x0", location="vram"))
        self.assertEqual(result.error_code, "TENSOR_IN_USE")

    def test_one_tensor_cannot_bind_two_pool_slots(self):
        slots = [dict(id="a", domain="dram", capacity_bytes=32, members=["x0"]),
                 dict(id="b", domain="vram", capacity_bytes=32, members=["x0"])]
        with self.assertRaises(InvalidInput):
            ServiceSession(scene(), parse_services(services_document(slots=slots)))


if __name__ == "__main__":
    unittest.main()
