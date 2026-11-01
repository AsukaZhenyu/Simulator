"""Independent capacity-auditor regression: separate copies have no alias ID."""
import unittest

from experiments.dac_validation_e3_audit_v2 import capacity_rejection


def fixture():
    evidence = {
        "scenario": {
            "architecture": {"memory_mode": "separate", "dram_capacity_bytes": 100, "vram_capacity_bytes": 100},
            "costs": {"compute": {"op": {"cpu": {"workspace_bytes": 0}}}},
        },
        "workload": {
            "tensors": [{"id": "in", "storage_bytes": 80}, {"id": "out", "storage_bytes": 30}],
            "operations": [{"id": "op", "output": "out"}],
        },
        "events": [],
        "tensor_catalogue": [
            {"id": "in", "allocation_bytes": [["dram", 80], ["vram", 80]],
             "allocation_refs": [["dram", None], ["vram", None]],
             "copies": [["dram", "READY"], ["vram", "ABSENT"]]},
            {"id": "out", "allocation_bytes": [["dram", 30], ["vram", 30]],
             "allocation_refs": [["dram", None], ["vram", None]],
             "copies": [["dram", "ABSENT"], ["vram", "ABSENT"]]},
        ],
    }
    row = {"rejected_action": {"kind": "COMPUTE", "target_id": "op", "device": "cpu"}}
    return evidence, row


class SeparatePoolCapacityAuditTests(unittest.TestCase):
    def test_counts_present_copy_without_shared_identity(self):
        evidence, row = fixture()
        receipt = capacity_rejection(evidence, row)
        self.assertEqual((receipt["current"], receipt["needed"], receipt["budget"]), (80, 30, 100))

    def test_absent_copy_does_not_consume_storage(self):
        evidence, row = fixture()
        evidence["tensor_catalogue"][0]["copies"][0][1] = "ABSENT"
        with self.assertRaisesRegex(ValueError, "independently fits"):
            capacity_rejection(evidence, row)

    def test_workspace_is_in_admission_budget(self):
        evidence, row = fixture()
        evidence["scenario"]["architecture"]["dram_capacity_bytes"] = 120
        evidence["scenario"]["costs"]["compute"]["op"]["cpu"]["workspace_bytes"] = 20
        receipt = capacity_rejection(evidence, row)
        self.assertEqual(receipt["needed"], 50)


if __name__ == "__main__":
    unittest.main()
