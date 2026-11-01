"""Mutation tests for the independent shared-storage trace auditor."""
from copy import deepcopy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from shared_storage_audit import AuditError, audit_documents


def fixture():
    graph = dict(workload=dict(tensors=[dict(id="x", storage_bytes=4, initial_locations=["dram"]),
                                        dict(id="y", storage_bytes=4)],
        operations=[dict(id="op", inputs=["x"], output="y")]),
        scenario=dict(architecture=dict(memory_mode="shared", shared_capacity_bytes=32),
            costs=dict(compute=dict(op=dict(gpu=dict(duration_ns=10, workspace_bytes=4))), transfer={}),
            mapspace=dict(allowed_devices=dict(op=["gpu"]), allowed_transfers=dict(
                dram_to_vram=["x", "y"], vram_to_dram=["x", "y"])),
            goal=dict(output_locations=dict(y="dram"))))
    events = [dict(action_index=0, kind="SHARE", target_id="x", t_ns=0, start_ns=0, end_ns=0,
                   src="dram", dst="vram", resource=None, bytes_transferred=0),
              dict(action_index=1, kind="EVICT", target_id="x", t_ns=0, location="dram", bytes_released=0),
              dict(action_index=2, kind="COMPUTE", target_id="op", t_ns=0, start_ns=0, end_ns=10,
                   device="gpu", resource="gpu_compute", workspace_bytes=4),
              dict(action_index=3, kind="ADVANCE", target_id=None, t_ns=10, resource=None),
              dict(action_index=4, kind="SHARE", target_id="y", t_ns=10, start_ns=10, end_ns=10,
                   src="vram", dst="dram", resource=None, bytes_transferred=0)]

    def snap(t, used, xdram="READY", xvram="ABSENT", ydram="ABSENT", yvram="ABSENT", op="NOT_STARTED"):
        statuses = dict(dram=dict(x=xdram, y=ydram), vram=dict(x=xvram, y=yvram))
        refs = [(tid, loc, "a0" if tid == "x" else "a1")
                for loc, table in statuses.items() for tid, state in table.items() if state != "ABSENT"]
        return dict(t=t, used_shared_bytes=used, used_dram_bytes=used, used_vram_bytes=used,
                    dram_status=list(statuses["dram"].items()), copy_status=list(statuses["vram"].items()),
                    ssd_status=[("x", "ABSENT"), ("y", "ABSENT")], op_status=[("op", op)],
                    storage_allocations=refs, next_allocation_id=2 if yvram != "ABSENT" else 1)
    states = [snap(0, 4), snap(0, 4, xvram="READY"), snap(0, 4, xdram="ABSENT", xvram="READY"),
              snap(0, 12, xdram="ABSENT", xvram="READY", yvram="RESERVED_OUTPUT", op="RUNNING"),
              snap(10, 8, xdram="ABSENT", xvram="READY", yvram="READY", op="DONE"),
              snap(10, 8, xdram="ABSENT", xvram="READY", ydram="READY", yvram="READY", op="DONE")]
    summary = dict(success=True, completed_makespan_ns=10, peak_shared_bytes=12,
                   peak_dram_bytes=12, peak_vram_bytes=12, h2d_bytes=0, d2h_bytes=0)
    return graph, events, states, summary


class SharedStorageIndependentAuditTests(unittest.TestCase):
    def test_hand_calculated_alias_release_and_workspace_pass(self):
        result = audit_documents(*fixture())
        self.assertTrue(result["pass_"])
        self.assertEqual((result["share_events"], result["zero_release_events"],
                          result["independent_peak_shared_bytes"]), (2, 1, 12))

    def test_wrong_peak_is_rejected(self):
        args = fixture()
        args[3]["peak_shared_bytes"] = 8
        with self.assertRaisesRegex(AuditError, "peak_shared_bytes"):
            audit_documents(*args)

    def test_alias_drop_cannot_claim_physical_release(self):
        args = fixture()
        args[1][1]["bytes_released"] = 4
        with self.assertRaisesRegex(AuditError, "physical release"):
            audit_documents(*args)

    def test_missing_alias_reference_is_rejected(self):
        args = fixture()
        args[2][1]["storage_allocations"].pop()
        with self.assertRaisesRegex(AuditError, "reference coverage"):
            audit_documents(*args)

    def test_wrong_alias_class_is_rejected(self):
        args = fixture()
        args[2][1]["storage_allocations"][-1] = ("x", "vram", "a1")
        with self.assertRaisesRegex(AuditError, "allocation aliases"):
            audit_documents(*args)

    def test_share_payload_is_not_a_copy(self):
        args = fixture()
        args[1][0]["bytes_transferred"] = 4
        with self.assertRaisesRegex(AuditError, "zero-time zero-payload"):
            audit_documents(*args)

    def test_compute_cannot_read_absent_logical_view(self):
        args = fixture()
        args[1][0] = dict(action_index=0, kind="COMPUTE", target_id="op", device="gpu",
                          t_ns=0, start_ns=0, end_ns=10, resource="gpu_compute", workspace_bytes=4)
        with self.assertRaisesRegex(AuditError, "input/producer is not ready"):
            audit_documents(*args)

    def test_share_cannot_read_output_before_production(self):
        args = fixture()
        args[1][0]["target_id"] = "y"
        with self.assertRaisesRegex(AuditError, "producer/source not ready"):
            audit_documents(*args)

    def test_undelivered_goal_cannot_be_success(self):
        graph, events, states, summary = fixture()
        with self.assertRaisesRegex(AuditError, "undelivered goal"):
            audit_documents(graph, events[:-1], states[:-1], summary)

    def test_true_copy_cannot_have_zero_duration(self):
        args = fixture()
        args[1][0]["kind"] = "COPY"
        with self.assertRaisesRegex(AuditError, "positive duration"):
            audit_documents(*args)

    def test_physical_capacity_is_checked_independently(self):
        args = fixture()
        args[0]["scenario"]["architecture"]["shared_capacity_bytes"] = 11
        with self.assertRaisesRegex(AuditError, "capacity exceeded"):
            audit_documents(*args)

    def test_nonzero_workspace_cannot_disappear_from_snapshot(self):
        args = fixture()
        args[2][3]["used_shared_bytes"] = 8
        with self.assertRaisesRegex(AuditError, "physical usage"):
            audit_documents(*args)


if __name__ == "__main__":
    unittest.main()
