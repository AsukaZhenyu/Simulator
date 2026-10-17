"""Independent boundary checks for the services ledger and link arithmetic.

These use hand-calculated payload timelines, not physical measurements.
"""
from fractions import Fraction
import unittest

from llm_infer_model.tensor.engine import (
    Action, CODE_CAPACITY_EXCEEDED, CODE_TENSOR_IN_USE,
)
from llm_infer_model.tensor.environment import EnvironmentEvent
from llm_infer_model.tensor.services import CODE_SLOT_OCCUPIED, ServiceSession, parse_services
from llm_infer_model.tensor.spec import COPY_READY, OP_DONE

from test_services import copy, document, frac, runtime, scene


class ServicesIndependentReviewTests(unittest.TestCase):
    def test_both_engines_below_link_cap_do_not_invent_extra_service(self):
        contract = document()
        contract["pcie"].update(ce_bytes_per_second=250_000_000,
                                sm_bytes_per_second=250_000_000)
        session = ServiceSession(scene((8, 4)), parse_services(contract))
        self.assertTrue(session.submit_transfer(copy(0), engine="ce").ok)
        self.assertTrue(session.submit_transfer(copy(1), engine="sm").ok)
        self.assertEqual(session.advance().t_ns, 16)
        self.assertEqual(session.advance().t_ns, 32)
        records = session.service_document()["link_service_intervals"]
        self.assertEqual(sum(frac(flow["served_bytes"]) for interval in records
                             for flow in interval["flows"]), 12)
        self.assertEqual(sum(frac(interval["unused_budget_bytes"]) for interval in records), 20)
        self.assertEqual([flow["resource"] for flow in records[-1]["flows"]], ["pcie_ce"])
        self.assertEqual(frac(records[-1]["flows"][0]["rate_bytes_per_ns"]), Fraction(1, 4))

    def test_fractional_residue_survives_engine_reuse_and_rearbitration(self):
        contract = document()
        contract["pcie"].update(bandwidth_bytes_per_second=3_000_000_000,
                                ce_bytes_per_second=3_000_000_000,
                                sm_bytes_per_second=3_000_000_000)
        session = ServiceSession(scene((8, 4, 4)), parse_services(contract))
        session.submit_transfer(copy(0), engine="ce")
        session.submit_transfer(copy(1), engine="sm")
        self.assertEqual(session.advance().t_ns, 3)
        remaining = session.service_document()["running_byte_progress"]
        self.assertEqual(frac(remaining[0]["remaining_bytes"]), Fraction(7, 2))
        self.assertTrue(session.submit_transfer(copy(2), engine="sm").ok)
        self.assertEqual(session.advance().t_ns, 6)
        self.assertFalse(session.state.running)
        service = session.service_document()["link_service_intervals"]
        served = {f"x{i}": sum(frac(flow["served_bytes"]) for interval in service
                                for flow in interval["flows"] if flow["tensor_id"] == f"x{i}")
                  for i in range(3)}
        self.assertEqual(served, {"x0": 8, "x1": 4, "x2": 4})
        self.assertEqual(sum(frac(interval["unused_budget_bytes"]) for interval in service), 2)
        self.assertEqual(session.result().evaluation.h2d_bytes, 16)
        self.assertEqual([event.end_ns for event in session.result().evaluation.events
                          if event.kind == "COPY"], [6, 3, 6])

    def test_same_time_cpu_and_two_copy_completions_precede_environment_and_unlock_sm(self):
        session = ServiceSession(
            scene((16, 16, 16, 16), initial=[["dram"], ["dram"],
                                           ["dram", "vram"], ["dram", "vram"]],
                  gpu_duration=32), parse_services(document()),
            environment_events=[EnvironmentEvent(32, cpu_available=False, vram_external_bytes=64),
                                EnvironmentEvent(48, cpu_available=True)])
        session.submit(Action("COMPUTE", "op2", device="cpu"))
        session.submit_transfer(copy(0), engine="ce")
        session.submit_transfer(copy(1), engine="sm")
        boundary = session.advance()
        self.assertTrue(boundary.ok)
        self.assertEqual(boundary.t_ns, 32)
        self.assertEqual(boundary.completed_operation_ids, ("op2",))
        self.assertEqual(set(boundary.completed_copies),
                         {("y2", "dram"), ("x0", "vram"), ("x1", "vram")})
        self.assertEqual(session.state.op("op2"), OP_DONE)
        self.assertFalse(session.observe().cpu_available)
        self.assertFalse(session.state.running)
        self.assertTrue(session.submit(Action("COMPUTE", "op3", device="gpu")).ok)
        self.assertEqual(session.advance().t_ns, 48)
        self.assertEqual(session.advance().t_ns, 64)
        self.assertEqual(session.state.op("op3"), OP_DONE)
        self.assertEqual([event.end_ns for event in session.result().evaluation.events[:3]], [32, 32, 32])

    def test_ssd_read_requires_explicit_release_of_dram_slot_owner(self):
        slot = dict(id="host-buffer", domain="dram", capacity_bytes=64, members=["x0", "x1"])
        rejected = runtime(slots=[slot], initial=[["ssd"], ["dram", "ssd"]])
        before = rejected.state
        failed = rejected.submit(Action("READ_SSD", "x0"))
        self.assertEqual(failed.error_code, CODE_SLOT_OCCUPIED)
        self.assertEqual(rejected.state, before)
        self.assertEqual(rejected.result().evaluation.ssd_read_bytes, 0)
        session = runtime(slots=[slot], initial=[["ssd"], ["dram", "ssd"]])
        self.assertTrue(session.submit(Action("EVICT", "x1", location="dram")).ok)
        self.assertTrue(session.submit(Action("READ_SSD", "x0")).ok)
        self.assertEqual(session.advance().t_ns, 12345)
        self.assertEqual(session.state.copy_at("x0", "dram"), COPY_READY)
        self.assertEqual(session.service_document()["slots"][0]["owner"], "x0")
        memory = next(item for item in session.observe().memory if item.domain == "dram")
        self.assertEqual((memory.used_bytes, memory.tensor_bytes, memory.runtime_reserved_bytes), (64, 0, 64))
        self.assertEqual(session.result().evaluation.ssd_read_bytes, 64)

    def test_slot_source_lock_lasts_until_d2h_completion_with_reloadable_source(self):
        slot = dict(id="device-buffer", domain="vram", capacity_bytes=64, members=["x0", "x1"])
        rejected = runtime(slots=[slot], initial=[["vram", "ssd"], ["dram"]])
        rejected.submit_transfer(copy(0, "vram", "dram"), engine="ce")
        self.assertEqual(rejected.submit(Action("EVICT", "x0", location="vram")).error_code,
                         CODE_TENSOR_IN_USE)
        session = runtime(slots=[slot], initial=[["vram", "ssd"], ["dram"]])
        session.submit_transfer(copy(0, "vram", "dram"), engine="ce")
        self.assertEqual(session.advance().t_ns, 64)
        self.assertTrue(session.submit(Action("EVICT", "x0", location="vram")).ok)
        self.assertTrue(session.submit_transfer(copy(1), engine="sm").ok)
        self.assertEqual(session.advance().t_ns, 80)
        self.assertEqual(session.service_document()["slots"][0]["owner"], "x1")
        self.assertEqual(session.result().evaluation.d2h_bytes, 64)
        self.assertEqual(session.result().evaluation.h2d_bytes, 16)
        self.assertEqual(next(item.used_bytes for item in session.observe().memory if item.domain == "dram"), 80)
        self.assertEqual(next(item.used_bytes for item in session.observe().memory if item.domain == "vram"), 64)

    def test_empty_and_small_slot_members_keep_full_reserve_plus_compute_workspace(self):
        slot = dict(id="device-buffer", domain="vram", capacity_bytes=64, members=["x0", "x1"])
        rejected = runtime(slots=[slot], capacity=83)
        rejected.submit_transfer(copy(1), engine="ce")
        rejected.advance()
        before = rejected.state
        failed = rejected.submit(Action("COMPUTE", "op1", device="gpu"))
        self.assertEqual(failed.error_code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(rejected.state, before)
        session = runtime(slots=[slot], capacity=84)
        session.submit_transfer(copy(1), engine="ce")
        session.advance()
        self.assertTrue(session.submit(Action("COMPUTE", "op1", device="gpu")).ok)
        self.assertEqual(session.result().evaluation.peak_vram_bytes, 84)
        memory = next(item for item in session.observe().memory if item.domain == "vram")
        self.assertEqual((memory.runtime_reserved_bytes, memory.tensor_bytes, memory.workspace_bytes), (64, 16, 4))


if __name__ == "__main__":
    unittest.main()
