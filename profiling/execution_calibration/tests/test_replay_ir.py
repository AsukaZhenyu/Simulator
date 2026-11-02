"""Measured binding invariants, checked without GPU or retained local assets."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis as a
import replay_ir as bridge
import run_dual as protocol


def fixture(family):
    case = dict(id="example-" + family, family=family, backend="hybrid", dtype="f32",
                hidden=8, intermediate=12 if family == "ffn" else 0,
                cpu_intermediate=4 if family == "ffn" else 0, tokens=2,
                history=3 if family == "attention" else 0, heads=2 if family == "attention" else 0,
                kv_heads=1 if family == "attention" else 0, head_dim=4 if family == "attention" else 0)
    boundary = dict(input_location="all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu",
        output_location="dram", merge_device="cpu", weights="generated_f32_not_model_weights",
        ffn_contract="compact_selected_rows_gate_up_swiglu_down_no_norm_or_residual",
        ffn_partition="gate_up_output_columns_and_down_input_rows_materialized_before_timing",
        attention_contract="post_RoPE_QKV_to_causal_GQA_QK_softmax_PV_sigmoid_gate_O_residual",
        kv_materialization="QK_private_K_scratch_then_PV_KV_append_outputs; old_cache_inputs_immutable")
    if family == "ffn":
        prices = dict(zip(("cpu/shard_gate", "cpu/shard_up", "cpu/shard_activation", "cpu/shard_down",
                           "gpu/shard_gate", "gpu/shard_up", "gpu/shard_activation", "gpu/shard_down",
                           "gpu/full_gate", "gpu/full_up", "gpu/full_activation", "gpu/full_down",
                           "hybrid/output_d2h", "cpu/sum", "gpu/output_d2h", "hybrid/preload_weights"),
                          (11, 17, 19, 23, 7, 13, 5, 31, 2, 3, 5, 7, 9, 3, 11, 29)))
        keys = [device + "/" + stage for device in ("cpu/shard", "gpu/shard", "gpu/full")
                for stage in ("gate", "up", "activation", "down")] + ["cpu/sum"]
    else:
        prices = dict(zip(("cpu/qk", "cpu/softmax", "gpu/qk", "gpu/softmax", "gpu/pv", "gpu/output",
                           "hybrid/probabilities_h2d", "gpu/output_d2h"), (2, 3, 17, 19, 7, 11, 5, 13)))
        keys = ["cpu/qk", "cpu/softmax", "gpu/qk", "gpu/softmax", "gpu/pv", "gpu/attention_output"]
    workspaces = {key: dict(known_gallocr_plus_cplan_bytes=16,
        workspace_bytes=None if key.startswith("gpu/") else 16,
        source="unknown_private_cuda_scratch" if key.startswith("gpu/") else "gallocr_plus_explicit_cplan") for key in keys}
    rows = []
    for repeat in (0, 1):
        rows.append(dict(id="profile-" + case["id"] + "-" + str(repeat), split="profile", repeat=repeat, seed=7,
            raw_sha256=str(repeat) * 64, raw_path="not-required-for-offline-unit-fixture",
            case=case, status="measured", boundary=boundary, workspaces=workspaces,
            metrics={metric: [value, value, value] for metric, value in prices.items()},
            ports={name: dict(bytes=count, input_fixture_fnv1a64="a" * 16)
                   for name, count in protocol.expected_port_bytes(case).items()}))
    model = a.freeze_model(rows, {case["id"]: protocol.recipes(case)})["cases"][0]
    signature = bridge.physical_signature(case, rows, dict(threads=1, cuda_graphs="disabled"))
    return case, rows, model, signature


class FixedMeasuredIRTests(unittest.TestCase):
    def test_ffn_two_sequential_branches_join_then_copy_then_cpu_sum(self):
        case, _, model, signature = fixture("ffn")
        _, run, binding, workload, _ = bridge.bind_resident(case, "hybrid/resident", model, signature)
        # CPU 70 ns, GPU 56 ns; copy is after BOTH complete; CPU sum is last.
        self.assertEqual(run.evaluation.makespan_ns, max(70, 56) + 9 + 3)
        events = run.evaluation.events
        starts = {event.target_id: event.start_ns for event in events if event.kind == "COMPUTE"}
        self.assertEqual(starts["cpu/shard/gate"], 0)
        self.assertEqual(starts["gpu/shard/gate"], 0)
        self.assertEqual(starts["cpu/shard/up"], 11)
        copy = next(event for event in events if event.kind == "COPY")
        self.assertEqual((copy.start_ns, copy.end_ns), (70, 79))
        self.assertEqual(starts["cpu/sum"], 79)
        self.assertEqual(len(workload["operations"]), 9)
        self.assertFalse(binding["physical_capacity_validated"])
        self.assertFalse(binding["high_level_ModelPlan_transform_lowering_exercised"])

    def test_gpu_ffn_uses_full_independent_prices_and_delivers_to_dram(self):
        case, _, model, signature = fixture("ffn")
        _, run, binding, _, scene = bridge.bind_resident(case, "gpu/resident", model, signature)
        self.assertEqual(run.evaluation.makespan_ns, 2 + 3 + 5 + 7 + 11)
        self.assertEqual(scene["goal"]["output_locations"], {"gpu/full/output": "dram"})
        self.assertEqual({row["metric"] for row in binding["component_bindings"]},
                         {"gpu/full_gate", "gpu/full_up", "gpu/full_activation", "gpu/full_down", "gpu/output_d2h"})

    def test_attention_hybrid_has_real_probability_transfer_and_kv_append_outputs(self):
        case, _, model, signature = fixture("attention")
        _, run, _, workload, _ = bridge.bind_resident(case, "hybrid/resident", model, signature)
        self.assertEqual(run.evaluation.makespan_ns, 2 + 3 + 5 + 7 + 11 + 13)
        copies = [event for event in run.evaluation.events if event.kind == "COPY"]
        self.assertEqual([(event.src, event.dst) for event in copies], [("dram", "vram"), ("vram", "dram")])
        pv = next(op for op in workload["operations"] if op["id"] == "gpu/pv")
        self.assertEqual(pv["extra_outputs"], ["gpu/key_appended", "gpu/value_appended"])
        self.assertIn("gpu/key_cache", pv["inputs"])

    def test_unknown_workspace_never_becomes_known_and_lower_bound_explicit(self):
        case, _, model, signature = fixture("attention")
        _, _, binding, _, scene = bridge.bind_resident(case, "gpu/resident", model, signature)
        self.assertTrue(all(row["measured_workspace_bytes"] is None for row in binding["workspace_assumptions"]))
        self.assertTrue(all(row["runtime_workspace_bytes"] == 16 for row in binding["workspace_assumptions"]))
        self.assertEqual(scene["architecture"]["vram_capacity_bytes"], 1 << 60)
        self.assertTrue(binding["capacity_fields_are_nonphysical"])
        self.assertIsNone(binding["measured_physical_signature"]["workspaces"]["gpu/qk"]["workspace_bytes"])

    def test_missing_component_is_rejected_without_interpolation(self):
        case, _, model, signature = fixture("attention")
        del model["prices"]["gpu/pv"]
        with self.assertRaisesRegex(ValueError, "profile-only"):
            bridge.bind_resident(case, "gpu/resident", model, signature)

    def test_heldout_cost_or_other_source_cannot_be_bound(self):
        for mutation in ("split", "source"):
            case, _, model, signature = fixture("attention")
            source = model["prices"]["gpu/pv"]["sources"][0]
            source["split" if mutation == "split" else "raw_sha256"] = "heldout" if mutation == "split" else "f" * 64
            with self.assertRaisesRegex(ValueError, "profile-only|physical profile"):
                bridge.bind_resident(case, "gpu/resident", model, signature)

    def test_wrong_kernel_contract_or_same_bytes_wrong_case_is_rejected(self):
        case, rows, model, signature = fixture("attention")
        rows[0]["boundary"] = deepcopy(rows[0]["boundary"])
        rows[0]["boundary"]["attention_contract"] = "FlashAttention"
        with self.assertRaisesRegex(ValueError, "boundary mismatch"):
            bridge.physical_signature(case, rows, {})
        changed = deepcopy(case)
        changed["history"] += 1
        with self.assertRaisesRegex(ValueError, "signature case mismatch"):
            bridge.bind_resident(changed, "gpu/resident", model, signature)

    def test_zero_instead_of_unknown_workspace_rejected(self):
        case, rows, _, _ = fixture("attention")
        rows[0]["workspaces"]["gpu/qk"]["workspace_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "remain unknown"):
            bridge.physical_signature(case, rows, {})

    def test_port_mismatch_and_aggregate_preload_rejected(self):
        case, rows, model, signature = fixture("ffn")
        rows[0]["ports"]["cpu/shard/input"]["bytes"] += 4
        with self.assertRaisesRegex(ValueError, "port byte"):
            bridge.physical_signature(case, rows, {})
        with self.assertRaisesRegex(ValueError, "only resident"):
            bridge.bind_resident(case, "hybrid/preload", model, signature)

    def test_measured_total_cannot_replace_recipe_component(self):
        case, _, model, signature = fixture("ffn")
        model["strategies"]["gpu/resident"] = "gpu/resident"
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            bridge.bind_resident(case, "gpu/resident", model, signature)

    def test_batch_preload_replays_three_independent_copies_in_native_order(self):
        case, rows, _, signature = fixture("ffn")
        signature["protocol"]["measurement_contract"] = dict(kind="completed_batch_mean", batch_repetitions=16)
        for row in rows:
            row["metrics"].update({bridge.WEIGHT_H2D[stage]: [value] * 3
                                   for stage, value in (("gate", 2), ("up", 3), ("down", 5))})
        declared = protocol.recipes(case)
        declared["hybrid/preload"] = bridge.declared_recipe(case, "hybrid/preload", signature)
        frozen = a.freeze_model(rows, {case["id"]: declared})["cases"][0]
        _, run, binding, workload, _ = bridge.bind_resident(case, "hybrid/preload", frozen, signature)
        self.assertEqual(run.evaluation.makespan_ns, 2 + 3 + 5 + max(70, 56) + 9 + 3)
        copies = [event for event in run.evaluation.events if event.kind == "COPY"]
        self.assertEqual([event.target_id for event in copies[:3]],
                         ["gpu/shard/" + stage + "_weight" for stage in ("gate", "up", "down")])
        self.assertEqual([(event.start_ns, event.end_ns) for event in copies[:3]], [(0, 2), (2, 5), (5, 10)])
        self.assertFalse(binding["preload_buffer_lifecycle_equivalence_validated"])
        self.assertFalse(binding["physical_capacity_validated"])
        weights = {tensor["id"]: tensor for tensor in workload["tensors"] if tensor["role"] == "weight"}
        self.assertEqual(weights["gpu/shard/gate_weight"]["initial_locations"], ["dram"])
        self.assertNotIn("hybrid/preload_weights", {row["metric"] for row in binding["component_bindings"]})


if __name__ == "__main__":
    unittest.main()
