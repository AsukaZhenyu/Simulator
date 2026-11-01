"""Independent declaration checks for the fixed matrix, without simulation."""
from dataclasses import FrozenInstanceError, asdict
import json
import unittest

from experiments.strategy_comparison_scenarios import (
    ScenarioSpec, candidate_ids, candidate_spec, catalogue_document, make_inputs, scenarios,
)


class StrategyComparisonScenarioTests(unittest.TestCase):
    def test_fixed_complete_matrix(self):
        declared = scenarios()
        self.assertIsInstance(declared, tuple)
        self.assertEqual(len(declared), 16)
        self.assertEqual(len({spec.id for spec in declared}), 16)
        dense = [spec for spec in declared if spec.model_kind == "dense"]
        moe = [spec for spec in declared if spec.model_kind == "moe"]
        self.assertEqual(len(dense), 4)
        self.assertEqual(len(moe), 12)
        self.assertEqual({(s.prefill_tokens, s.decode_steps) for s in dense},
                         {(8, 0), (128, 0), (8, 8), (128, 8)})
        self.assertEqual({(s.prefill_tokens, s.decode_steps, s.route_pattern) for s in moe},
                         {(p, d, pattern) for p, d in ((8, 0), (128, 0), (8, 8), (128, 8))
                          for pattern in ("hot", "rotating", "phase_shift")})
        self.assertEqual(candidate_ids("dense"), tuple(f"e{e}-s{s}" for e in range(4) for s in range(4)))
        self.assertEqual(candidate_ids("moe"), tuple(f"e{e}-s{s}" for e in range(6) for s in range(4)))
        self.assertEqual(sum(len(candidate_ids(spec.model_kind)) for spec in declared), 352)
        with self.assertRaises(FrozenInstanceError):
            declared[0].decode_steps = 10

    def test_synthetic_route_coverage_and_exact_pairs(self):
        for spec in scenarios():
            with self.subTest(spec=spec.id):
                model, _, workload = make_inputs(spec)
                self.assertEqual(workload.batch_size, 1)
                if spec.model_kind == "dense":
                    self.assertEqual(workload.routes, ())
                    self.assertIsNone(workload.route_source)
                    continue
                self.assertEqual((model.experts, model.top_k), (4, 2))
                self.assertIn("synthetic", workload.route_source)
                self.assertIn("not_measured", workload.route_source)
                rows = {(r.layer_id, r.absolute_position): r for r in workload.routes}
                expected = {(f"layer{layer}", position) for layer in range(2)
                            for position in range(spec.prefill_tokens + spec.decode_steps)}
                self.assertEqual(set(rows), expected)
                self.assertEqual(len(rows), len(workload.routes))
                for (layer_id, position), route in rows.items():
                    self.assertEqual(route.request_id, "request0")
                    self.assertEqual(len(route.expert_ids), 2)
                    self.assertEqual(len(set(route.expert_ids)), 2)
                    self.assertTrue(all(type(e) is int and 0 <= e < 4 for e in route.expert_ids))
                    layer = int(layer_id[5:])
                    if spec.route_pattern == "hot":
                        pair = (0, 1)
                    elif spec.route_pattern == "rotating":
                        pair = ((position + layer) % 4, (position + layer + 1) % 4)
                    else:
                        pair = (0, 1) if position < spec.prefill_tokens else (2, 3)
                    self.assertEqual(route.expert_ids, pair)

    def test_phase_shift_boundary_is_absolute_prefill_end(self):
        for spec in scenarios():
            if spec.route_pattern != "phase_shift" or not spec.decode_steps:
                continue
            _, _, workload = make_inputs(spec)
            rows = {(r.layer_id, r.absolute_position): r.expert_ids for r in workload.routes}
            for layer in ("layer0", "layer1"):
                self.assertEqual(rows[layer, spec.prefill_tokens - 1], (0, 1))
                self.assertEqual(rows[layer, spec.prefill_tokens], (2, 3))

    def test_exact_hardware_and_geometry_are_fixed_across_scenarios(self):
        actual = [make_inputs(spec) for spec in scenarios()]
        reference = asdict(actual[0][1])
        for spec, (model, hardware, workload) in zip(scenarios(), actual):
            self.assertEqual(asdict(hardware), reference)
            self.assertEqual((model.layers, model.hidden_size, model.intermediate_size,
                              model.heads, model.kv_heads), (2, 64, 192, 4, 2))
            self.assertEqual((workload.prefill_tokens, workload.decode_steps),
                             (spec.prefill_tokens, spec.decode_steps))
        self.assertEqual(reference, dict(
            cpu=dict(compute_flops_per_second=100e9, memory_bandwidth_bytes_per_second=20e9,
                     launch_overhead_ns=500),
            gpu=dict(compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9,
                     launch_overhead_ns=2000),
            dram_capacity_bytes=128 * 1024**2, vram_capacity_bytes=512 * 1024,
            transfer_bandwidth_bytes_per_second=8e9, ssd_read_bandwidth_bytes_per_second=1e9,
            ssd_write_bandwidth_bytes_per_second=500e6, transfer_latency_ns=1000,
            memory_mode="separate", shared_capacity_bytes=None,
            shared_runtime_reserved_bytes=0, shared_alignment_bytes=1))

    def test_storage_factorial_changes_one_dimension_at_a_time(self):
        for kind in ("dense", "moe"):
            for e in range(4 if kind == "dense" else 6):
                settings = [asdict(candidate_spec(kind, f"e{e}-s{s}")) for s in range(4)]
                self.assertTrue(all(s["cache"]["budget_bytes"] == 192 * 1024 for s in settings))
                self.assertEqual([s["cache"]["rule"] for s in settings],
                                 ["lru", "lru", "history_frequency", "history_frequency"])
                self.assertEqual(settings[0]["prefetch"], settings[2]["prefetch"])
                self.assertEqual(settings[1]["prefetch"], settings[3]["prefetch"])
                self.assertEqual(settings[1]["prefetch"], dict(
                    next_layers=1, max_count=2, ranking="history_frequency", include_dense_weights=True))
                for left, right, changing in ((0, 1, "prefetch"), (2, 3, "prefetch"),
                                               (0, 2, "cache"), (1, 3, "cache")):
                    a, b = settings[left].copy(), settings[right].copy()
                    self.assertNotEqual(a.pop(changing), b.pop(changing))
                    self.assertEqual(a, b)

    def test_execution_declarations_use_public_causal_fields(self):
        for kind in ("dense", "moe"):
            expected_ffn = "dense_ffn" if kind == "dense" else "expert_ffn"
            for storage in range(4):
                self.assertEqual(candidate_spec(kind, f"e0-s{storage}").default_device, "gpu")
                self.assertEqual(candidate_spec(kind, f"e1-s{storage}").default_device, "cpu")
                self.assertEqual(candidate_spec(kind, f"e2-s{storage}").placements,
                                 {expected_ffn: "cpu"})
                self.assertEqual(candidate_spec(kind, f"e3-s{storage}").placements,
                                 {"attention": "cpu"})
        for execution, rule in ((4, "resident_first"), (5, "current_cost")):
            for storage in range(4):
                spec = candidate_spec("moe", f"e{execution}-s{storage}")
                self.assertEqual(spec.default_device, "gpu")
                self.assertEqual(spec.placements, {})
                self.assertEqual(spec.expert_rule, rule)
                self.assertEqual(spec.expert_order, "graph_order")
                self.assertEqual(spec.sections()["policy"]["experts"], dict(
                    device_rule=rule, order_rule="graph_order", selection_unit="expert_region"))

    def test_catalogue_is_complete_serializable_and_reproducible(self):
        doc = catalogue_document()
        self.assertEqual(doc, catalogue_document())
        self.assertEqual(json.loads(json.dumps(doc))["declared_run_count"], 352)
        self.assertEqual((doc["scenario_count"], doc["candidate_count_by_model"]),
                         (16, {"dense": 16, "moe": 24}))
        self.assertEqual(len(doc["scenarios"]), 16)
        self.assertFalse(doc["calibrated"])
        self.assertFalse(doc["hardware_measurement"])
        self.assertEqual(doc["dtype"], "F32")
        self.assertEqual(doc["workspace_bytes"], 0)
        for row, spec in zip(doc["scenarios"], scenarios()):
            self.assertEqual(row["spec"], asdict(spec))
            model, hardware, workload = make_inputs(spec)
            self.assertEqual(row["model"], asdict(model))
            self.assertEqual(row["hardware"], asdict(hardware))
            self.assertEqual(row["workload"], asdict(workload))

    def test_no_decode_phase_shift_is_an_explicit_equivalence_control(self):
        doc = catalogue_document()
        declared = {spec.id: spec for spec in scenarios()}
        self.assertEqual(doc["distinct_workload_count"], 14)
        self.assertEqual(doc["equivalence_controls"], {
            "moe-short_prefill-phase_shift": "moe-short_prefill-hot",
            "moe-long_prefill-phase_shift": "moe-long_prefill-hot",
        })
        for control, reference in doc["equivalence_controls"].items():
            cm, ch, cw = make_inputs(declared[control])
            rm, rh, rw = make_inputs(declared[reference])
            self.assertEqual((cm, ch), (rm, rh))
            self.assertEqual((cw.prefill_tokens, cw.decode_steps), (rw.prefill_tokens, 0))
            self.assertEqual(cw.routes, rw.routes)

    def test_unknown_or_inconsistent_declarations_are_rejected(self):
        for kind, candidate in (("dense", "e4-s0"), ("moe", "e6-s0"),
                                 ("dense", "e0-s4"), ("wrong", "e0-s0")):
            with self.assertRaises(ValueError):
                candidate_spec(kind, candidate)
        for arguments in (
            ("bad", "dense", 8, 0, "hot"), ("bad", "moe", 8, 0, None),
            ("bad", "moe", 8, 0, "unknown"), ("bad", "dense", 0, 0, None),
            ("bad", "dense", 8, -1, None), ("../bad", "dense", 8, 0, None),
            ("bad", "dense", True, 0, None),
        ):
            with self.assertRaises(ValueError):
                ScenarioSpec(*arguments)


if __name__ == "__main__":
    unittest.main()
