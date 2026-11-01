"""Price substitution and declaration controls, independent of vendor claims."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Route, Simulator, Workload
from experiments.hardware_system_protocol import (
    CACHE_BUDGET_BYTES, GEOMETRY, NPUProxyCosts, candidate_ids, candidate_spec,
    make_workload, protocol_document, proxy_duration_ns, workload_specs,
)


class HardwareSystemProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hardware = Hardware(
            cpu=Device(1e9, 2e9, 71), gpu=Device(3e9, 4e9, 113),
            dram_capacity_bytes=2**26, vram_capacity_bytes=2**26,
            transfer_bandwidth_bytes_per_second=1e9,
            ssd_read_bandwidth_bytes_per_second=1e9,
            ssd_write_bandwidth_bytes_per_second=1e9,
            transfer_latency_ns=41)

    def test_exact_scaling_includes_complete_price_and_large_integers(self):
        self.assertEqual(proxy_duration_ns(101, 0.5), 51)
        self.assertEqual(proxy_duration_ns(101, 1), 101)
        self.assertEqual(proxy_duration_ns(101, 2), 202)
        self.assertEqual(proxy_duration_ns(2**60 + 1, 0.5), 2**59 + 1)
        for alpha in (True, 0, -1, float("nan"), float("inf"), "1"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                NPUProxyCosts(alpha=alpha)
        for duration in (True, 0, -1, 1.5):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                proxy_duration_ns(duration, 1)

    def test_generated_prices_change_only_accelerator_duration(self):
        model, workload = Model.dense(layers=1), Workload(2, decode_steps=1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = AnalyticalCosts(workspace_bytes=128).build_inputs(
                root / "reference", model, self.hardware, workload)
            proxy = NPUProxyCosts(workspace_bytes=128, alpha=0.5).build_inputs(
                root / "proxy", model, self.hardware, workload)
            for path in (root / "reference").glob("*.costs.json"):
                baseline = json.loads(path.read_text(encoding="utf-8"))
                changed = json.loads((root / "proxy" / path.name).read_text(encoding="utf-8"))
                self.assertEqual(baseline["transfer_entries"], changed["transfer_entries"])
                for expected, actual in zip(baseline["entries"], changed["entries"]):
                    expected = dict(expected)
                    if expected["device"] == "gpu":
                        expected["duration_ns"] = (expected["duration_ns"] + 1) // 2
                    self.assertEqual(expected, actual)
            metadata = proxy["metadata"]
            self.assertEqual(metadata["npu_proxy"]["execution_slot"], "gpu")
            self.assertFalse(metadata["npu_proxy"]["simultaneous_gpu_and_npu"])
            self.assertFalse(metadata["hardware_measurements_used"])
            self.assertIn("F32", metadata["npu_proxy"]["precision"])
            self.assertIn("not NPU ABI", metadata["npu_proxy"]["physical_signatures"])
            self.assertEqual(metadata["workspace_bytes_per_macro"], 128)
            self.assertEqual(metadata["npu_proxy"]["reference_gpu_formula"],
                             reference["metadata"]["compute_formula"])
            rows = metadata["compute_estimates"]
            for row in rows:
                if row["device"] == "gpu":
                    self.assertEqual(row["duration_ns"],
                                     (row["reference_gpu_duration_ns"] + 1) // 2)
            self.assertNotIn(b"\r", (root / "proxy" / "analytical-metadata.json").read_bytes())

    def test_alpha_one_is_event_identical_and_cpu_only_ignores_alpha(self):
        model, workload = Model.dense(layers=1), Workload(2, decode_steps=1)
        for candidate, alphas in (("e0-s0", (1,)), ("e1-s0", (0.5, 1, 2))):
            events = None
            for cost_model in (AnalyticalCosts(), *(NPUProxyCosts(alpha=a) for a in alphas)):
                simulator = Simulator(model=model, hardware=self.hardware,
                                      workload=workload, cost_model=cost_model)
                with simulator.compile(scheduling=candidate_spec("dense", candidate)) as compiled:
                    result = compiled.run()
                    self.assertTrue(result.success)
                    current = [asdict(event) for event in result.events]
                    if events is None:
                        events = current
                    else:
                        self.assertEqual(events, current)

    def test_alpha_one_dense_and_moe_cost_files_are_byte_identical(self):
        cases = (
            (Model.dense(layers=1), Workload(2, decode_steps=1)),
            (Model.moe(layers=1, experts=2, top_k=1), Workload(2, decode_steps=1,
                routes=tuple(Route("request0", "layer0", p, (p % 2,)) for p in range(3)),
                route_source="declared_synthetic_test_not_measured")),
        )
        for model, workload in cases:
            with self.subTest(kind=model.kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                AnalyticalCosts().build_inputs(root / "reference", model, self.hardware, workload)
                NPUProxyCosts(alpha=1).build_inputs(root / "proxy", model, self.hardware, workload)
                for path in (root / "reference").glob("*.costs.json"):
                    self.assertEqual(path.read_bytes(), (root / "proxy" / path.name).read_bytes())

    def test_proxy_requires_cpu_and_reference_gpu(self):
        for hardware in (replace(self.hardware, cpu=None), replace(self.hardware, gpu=None)):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "CPU and reference GPU"):
                    NPUProxyCosts().build_inputs(folder, Model.dense(layers=1), hardware, Workload(1))
                self.assertFalse(any(Path(folder).iterdir()))

    def test_protocol_geometry_routes_and_candidate_fairness(self):
        self.assertEqual(GEOMETRY, dict(layers=2, hidden_size=6144,
            intermediate_size=16384, heads=24, kv_heads=6))
        self.assertIsNone(CACHE_BUDGET_BYTES)
        specs = workload_specs()
        self.assertEqual(len(specs), 4)
        self.assertEqual({(s.prefill_tokens, s.decode_steps) for s in specs}, {(32, 0), (8, 2)})
        for spec in specs:
            model, work = make_workload(spec)
            self.assertEqual(model.hidden_size, 6144)
            if spec.model_kind == "moe":
                self.assertEqual(len(work.routes), 20)
                self.assertIn("not_measured", work.route_source)
                for route in work.routes:
                    pair = ((0, 1) if spec.route_pattern == "hot" else
                            ((route.absolute_position + int(route.layer_id[5:])) % 4,
                             (route.absolute_position + int(route.layer_id[5:]) + 1) % 4))
                    self.assertEqual(route.expert_ids, pair)
            else:
                self.assertEqual(work.routes, ())
            self.assertEqual(len(candidate_ids(spec.model_kind)), 8)
            for execution in range(4):
                left = asdict(candidate_spec(spec.model_kind, f"e{execution}-s0"))
                right = asdict(candidate_spec(spec.model_kind, f"e{execution}-s1"))
                self.assertIsNone(left["cache"]["budget_bytes"])
                self.assertNotEqual(left.pop("prefetch"), right.pop("prefetch"))
                self.assertEqual(left, right)
        document = protocol_document()
        self.assertEqual(document, protocol_document())
        self.assertEqual(len(document["workloads"]), 4)
        self.assertEqual(json.loads(json.dumps(document))["cache_budget_bytes"], None)

    def test_unknown_candidate_and_invalid_quota_are_rejected(self):
        for kind, candidate in (("dense", "e4-s0"), ("moe", "e1-s2"), ("unknown", "e0-s0")):
            with self.assertRaises(ValueError):
                candidate_spec(kind, candidate)
        for quota in (False, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                candidate_spec("dense", "e0-s0", cache_budget_bytes=quota)


if __name__ == "__main__":
    unittest.main()
