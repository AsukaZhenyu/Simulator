"""Post-run traffic classification must follow actual device consumers."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.hardware_system_unused_copy_audit import audit_case


class UnusedWeightCopyAuditTests(unittest.TestCase):
    def test_only_real_weight_copies_without_any_accelerator_consumer_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            documents = {
                "summary.json": dict(system_id="s", variant_id="gpu", workload_id="w",
                    candidate_id="e2-s1", success=True, completed_makespan_ns=100),
                "tensor-catalogue.json": [dict(id="gpu_weight", role="weight"),
                    dict(id="cpu_weight", role="weight"), dict(id="input", role="activation")],
                "operation-catalogue.json": [dict(id="cpu_op", inputs=["cpu_weight"]),
                    dict(id="gpu_op", inputs=["gpu_weight"])],
                "events.json": [
                    dict(kind="COPY", target_id="gpu_weight", dst="vram", bytes_transferred=11),
                    dict(kind="COPY", target_id="cpu_weight", dst="vram", bytes_transferred=7),
                    dict(kind="COPY", target_id="cpu_weight", dst="vram", bytes_transferred=7),
                    dict(kind="COPY", target_id="input", dst="vram", bytes_transferred=3),
                    dict(kind="COPY", target_id="cpu_weight", dst="dram", bytes_transferred=7),
                    dict(kind="SHARE", target_id="cpu_weight", dst="vram", bytes_transferred=0),
                    dict(kind="COMPUTE", target_id="cpu_op", device="cpu"),
                    dict(kind="COMPUTE", target_id="gpu_op", device="gpu"),
                ],
            }
            for name, value in documents.items():
                (folder / name).write_text(json.dumps(value), encoding="utf-8")
            row = audit_case(folder)
            self.assertEqual(row["total_weight_copy_to_accelerator_bytes"], 25)
            self.assertEqual(row["weight_copy_bytes_with_no_accelerator_consumer"], 14)
            self.assertEqual(row["weight_copy_count_with_no_accelerator_consumer"], 2)
            self.assertEqual(row["unused_weight_copy_bytes_by_tensor"], {"cpu_weight": 14})

    def test_missing_evidence_is_unknown_instead_of_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "summary.json").write_text(json.dumps(dict(system_id="s", variant_id="gpu",
                workload_id="w", candidate_id="e0-s0", success=False)), encoding="utf-8")
            row = audit_case(folder)
            self.assertFalse(row["evaluated"])
            self.assertNotIn("weight_copy_bytes_with_no_accelerator_consumer", row)


if __name__ == "__main__":
    unittest.main()
