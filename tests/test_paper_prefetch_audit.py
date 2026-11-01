import unittest
from experiments.paper_prefetch_audit import unused_weight_copies


class ConsumerAuditTests(unittest.TestCase):
    def setUp(self):
        self.tensors = [{"id": "w", "role": "weight"}, {"id": "x", "role": "input"}]
        self.operations = [{"id": "op", "inputs": ["w", "x"]}]
        self.copy = {"kind": "COPY", "target_id": "w", "dst": "vram", "bytes_transferred": 64}

    def result(self, events):
        return unused_weight_copies(events, self.operations, self.tensors)

    def test_cpu_consumer_does_not_consume_gpu_copy(self):
        result = self.result([self.copy, {"kind": "COMPUTE", "target_id": "op", "device": "cpu"}])
        self.assertEqual(result["bytes_without_gpu_consumer"], 64)

    def test_gpu_consumer_excludes_weight_conservatively(self):
        result = self.result([self.copy, {"kind": "COMPUTE", "target_id": "op", "device": "gpu"}, self.copy])
        self.assertEqual(result["physical_weight_copy_to_gpu_bytes"], 128)
        self.assertEqual(result["bytes_without_gpu_consumer"], 0)

    def test_share_and_activation_are_not_weight_traffic(self):
        result = self.result([dict(self.copy, kind="SHARE", bytes_transferred=0),
                              dict(self.copy, target_id="x"), dict(self.copy, dst="dram")])
        self.assertEqual(result["physical_weight_copy_to_gpu_bytes"], 0)

    def test_missing_compute_metadata_is_rejected(self):
        with self.assertRaises(KeyError):
            self.result([{"kind": "COMPUTE", "target_id": "unknown", "device": "gpu"}])


if __name__ == "__main__":
    unittest.main()
