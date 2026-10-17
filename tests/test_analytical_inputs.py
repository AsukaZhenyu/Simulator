"""Independent edge cases for honest analytical inputs and route validation."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Route, Unsupported, Workload
from llm_infer_model.m2.expand import load_and_compile


class AnalyticalInputTests(unittest.TestCase):
    def setUp(self):
        self.device = Device(1e9, 10e9)
        self.hardware = Hardware(self.device, self.device, 1 << 24, 1 << 24,
                                 2e9, 1e9, 500e6)

    def test_transfer_rejects_wrong_typed_inputs_with_clear_errors(self):
        costs = AnalyticalCosts()
        with self.assertRaisesRegex(ValueError, "declared Hardware"):
            costs.transfer_duration_ns(None, route="dram_to_vram", storage_bytes=4)
        with self.assertRaisesRegex(ValueError, "must be a string"):
            costs.transfer_duration_ns(self.hardware, route=[], storage_bytes=4)
        with self.assertRaisesRegex(ValueError, "unsupported.*route"):
            costs.transfer_duration_ns(self.hardware, route="gpu_to_ssd", storage_bytes=4)
        with self.assertRaises(ValueError):
            costs.transfer_duration_ns(self.hardware, route="dram_to_vram", storage_bytes=True)

    def test_optional_rates_remain_unknown_and_invalid_rates_are_rejected(self):
        hardware = replace(self.hardware, ssd_read_bandwidth_bytes_per_second=None)
        self.assertIsNone(hardware.ssd_read_bandwidth_bytes_per_second)
        with self.assertRaisesRegex(Unsupported, "ssd_read"):
            AnalyticalCosts().transfer_duration_ns(hardware, route="ssd_to_dram", storage_bytes=4)
        for field in ("transfer_bandwidth_bytes_per_second", "ssd_read_bandwidth_bytes_per_second",
                      "ssd_write_bandwidth_bytes_per_second"):
            for invalid in (False, -1, 0, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    replace(self.hardware, **{field: invalid})

    def test_no_implicit_device_or_zero_capacity(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            replace(self.hardware, cpu=None, gpu=None)
        with self.assertRaisesRegex(ValueError, "must be Device"):
            replace(self.hardware, cpu="CPU")
        for field in ("dram_capacity_bytes", "vram_capacity_bytes"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.hardware, **{field: 0})
        for invalid in (True, -1, 0.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.hardware, transfer_latency_ns=invalid)

    def test_routes_cannot_hide_mutable_expert_or_score_lists(self):
        for route in (Route("request0", "layer0", 0, [0, 1]),
                      Route("request0", "layer0", 0, (0, 1), [0.5, 0.5])):
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, "immutable tuples"):
                Workload(1, routes=(route,), route_source="explicit")

    def test_incomplete_large_route_domain_is_rejected_without_materializing_it(self):
        model = Model.moe(layers=1, experts=2, top_k=1)
        work = Workload(0x7fffffff, routes=(Route("request0", "layer0", 0, (0,)),),
                        route_source="intentionally incomplete")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "missing=2147483646"):
                AnalyticalCosts().build_inputs(folder, model, self.hardware, work)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_duplicate_and_out_of_range_routes_are_explicit_errors(self):
        model = Model.moe(layers=1, experts=2, top_k=1)
        route = Route("request0", "layer0", 0, (0,))
        cases = [((route, route), "duplicate"),
                 ((replace(route, layer_id="layer00"),), "outside"),
                 ((replace(route, request_id="request1"),), "outside"),
                 ((replace(route, absolute_position=1),), "outside"),
                 ((replace(route, expert_ids=(2,)),), "outside the declared model")]
        for routes, message in cases:
            with self.subTest(routes=routes), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, message):
                    AnalyticalCosts().build_inputs(folder, model, self.hardware,
                        Workload(1, routes=routes, route_source="invalid test routes"))

    def test_history_increases_attention_work_without_repricing_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = AnalyticalCosts().build_inputs(folder, Model.dense(layers=1),
                self.hardware, Workload(2, decode_steps=2))
            rows = inputs["metadata"]["compute_estimates"]
            def cost(scope, kind):
                return next(row for row in rows if row["invocation_id"] == scope
                            and row["kind"] == kind and row["device"] == "cpu")
            # Two queries x two keys are computed before causal masking:
            # QK/PV 256 + softmax 80 + O projection 1024 + residual 32.
            self.assertEqual(cost("layer0-prefill", "attention")["flops"], 1392)
            early = cost("layer0-decode0", "attention")
            later = cost("layer0-decode1", "attention")
            # One additional key for one query: QK+PV = 4H FLOPs,
            # plus five softmax operations per head. H=16, heads=4.
            self.assertEqual(later["flops"] - early["flops"], 84)
            self.assertGreater(later["duration_ns"], early["duration_ns"])
            self.assertEqual(cost("layer0-decode0", "projection")["duration_ns"],
                             cost("layer0-decode1", "projection")["duration_ns"])

    def test_generated_inputs_bind_exact_costs_and_use_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = AnalyticalCosts(workspace_bytes=0).build_inputs(folder,
                Model.dense(layers=1), self.hardware, Workload(1))
            for file in Path(folder).glob("*.json"):
                raw = file.read_bytes()
                self.assertNotIn(b"\r", raw)
                json.loads(raw)
            metadata = inputs["metadata"]
            self.assertFalse(metadata["calibrated"])
            self.assertFalse(metadata["hardware_measurements_used"])
            self.assertIn("not a measurement", metadata["workspace_semantics"])
            path = Path(folder) / "layer0-prefill.costs.json"
            costs = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(all(row["provenance"] == "synthetic_layer" for row in costs["entries"]))
            # Saved inputs use exact tables, not the internal payload enumerator.
            load_and_compile(Path(folder) / "layer0-prefill.json")
            costs["transfer_entries"] = costs["transfer_entries"][1:]
            path.write_text(json.dumps(costs), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(Unsupported, "精确条目"):
                load_and_compile(Path(folder) / "layer0-prefill.json")

    def test_unrepresentable_time_is_reported_instead_of_returning_infinity(self):
        with self.assertRaisesRegex(ValueError, "finite nanoseconds"):
            AnalyticalCosts().compute_duration_ns(Device(1e-300, 1e9), flops=10**30,
                                                  traffic_bytes=0)


if __name__ == "__main__":
    unittest.main()
