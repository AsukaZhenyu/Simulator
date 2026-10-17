"""Independent shape, storage and operation coverage for the shared block contract."""
from __future__ import annotations

import copy
import math
import unittest

from llm_infer_model.m2.dense_contract import (
    CONTRACT, build_plan, native_arguments, operations, ports, resident_accounting,
    signature, stage_weight_roles, validate_case, validate_config, weight_shapes,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def config(*, hidden=16, intermediate=32, gate="none", qk="none"):
    result = dict(hidden=hidden, intermediate=intermediate, heads=4, kv_heads=2,
        head_dim=4, attention_gate=gate, qk_norm=qk, rope_mode="normal", rope_dims=4,
        rope_base=10000, norm_epsilon=1e-5)
    result["weight_dtypes"] = {role: "f32" for role in weight_shapes(result)}
    return result


def case(**changes):
    return dict(id="small", kind="block", phase="prefill", batch=1, tokens=3,
                visible_history=0, config=config(), **changes)


class DenseContract(unittest.TestCase):
    def test_matrix_rows_and_norm_vectors_follow_declared_geometry(self):
        actual = weight_shapes(config(hidden=13))
        self.assertEqual(actual["q"], (13, 16))
        self.assertEqual(actual["k"], (13, 8))
        self.assertEqual(actual["output"], (16, 13))
        self.assertEqual(actual["input_norm"], (13,))
        self.assertEqual(actual["down"], (32, 13))

    def test_optional_norm_and_gate_weights_are_mandatory_when_enabled(self):
        raw = config(gate="sigmoid", qk="rms")
        self.assertTrue({"gate_proj", "q_norm", "k_norm"}.issubset(raw["weight_dtypes"]))
        raw["weight_dtypes"].pop("gate_proj")
        with self.assertRaisesRegex(InvalidInput, "weight_dtypes"):
            validate_config(raw)

    def test_norm_gamma_cannot_silently_use_quantized_matrix_dtype(self):
        raw = config()
        raw["weight_dtypes"]["input_norm"] = "f16"
        with self.assertRaisesRegex(Unsupported, "gamma"):
            validate_config(raw)

    def test_quantized_matrix_storage_uses_whole_rows(self):
        raw = config(hidden=256, intermediate=512)
        raw["head_dim"] = 64
        raw["rope_dims"] = 64
        for role in ("q", "k", "v", "output", "gate", "up"):
            raw["weight_dtypes"][role] = "q4_k"
        raw["weight_dtypes"]["down"] = "q6_k"
        checked = validate_config(raw)
        actual = resident_accounting(checked, "dense_ffn", 3, 0)
        self.assertEqual(actual["weight_roles_bytes"]["gate"], 512*144)
        self.assertEqual(actual["weight_roles_bytes"]["down"], 256*2*210)
        raw["hidden"] = 255
        with self.assertRaisesRegex(Unsupported, "divisible"):
            validate_config(raw)

    def test_normal_rope_dimensions_and_gqa_axes_are_checked(self):
        for key, value in (("rope_dims", 3), ("rope_dims", 6), ("heads", 3), ("head_dim", True)):
            raw = config(); raw[key] = value
            with self.assertRaises(InvalidInput):
                validate_config(raw)
        raw = config(); raw["rope_mode"] = "neox"
        with self.assertRaisesRegex(Unsupported, "no fallback"):
            validate_config(raw)

    def test_float_parameters_record_actual_ggml_float32_values(self):
        checked = validate_config(config())
        self.assertEqual(checked["norm_epsilon"], 9.999999747378752e-06)
        self.assertEqual(validate_config(checked), checked)
        for value in (True, float("nan"), float("inf"), 0, -1, 1e-60):
            raw = config(); raw["norm_epsilon"] = value
            with self.assertRaises(InvalidInput):
                validate_config(raw)

    def test_every_compute_operation_belongs_to_one_isolated_stage(self):
        checked = config(gate="sigmoid", qk="rms")
        projection, attention, ffn = (operations(checked, stage) for stage in ("projection", "attention", "dense_ffn"))
        self.assertIn("input_RMSNorm", projection)
        self.assertIn("RoPE_Q", projection)
        self.assertIn("Q_per_head_RMSNorm", projection)
        self.assertIn("attention_residual_add", attention)
        self.assertIn("ffn_residual_add", ffn)
        self.assertEqual(operations(checked, "block"), projection+attention+ffn)

    def test_absolute_rope_positions_are_real_i32_input_storage(self):
        inputs, outputs = ports(config(), "projection", 3)
        self.assertEqual(inputs["positions"], {"dtype": "i32", "ne": (3, 1, 1, 1)})
        self.assertEqual(outputs["q"]["ne"], (4, 3, 4, 1))
        self.assertEqual(outputs["k"]["ne"], (4, 3, 2, 1))

    def test_full_logical_resident_bytes_have_an_independent_manual_oracle(self):
        actual = resident_accounting(config(), "block", 3, 0)
        # 2 gamma vectors:128, Q/O:2048, K/V:1024, FFN:6144.
        self.assertEqual(actual["weight_bytes"], 9344)
        self.assertEqual(actual["input_bytes"], 192+12)
        self.assertEqual(actual["output_bytes"], 192+96+96+192+192)
        self.assertEqual(actual["kv_cache_bytes"], 192)
        self.assertEqual(actual["resident_logical_bytes"], 10508)

    def test_new_kv_input_ports_and_appended_cache_capacity_are_separate(self):
        actual = resident_accounting(config(), "attention", 3, 8)
        self.assertEqual(actual["input_ports_bytes"]["k"]+actual["input_ports_bytes"]["v"], 192)
        self.assertEqual(actual["kv_new_bytes"], 192)
        self.assertEqual(actual["kv_history_bytes"], 512)

    def test_no_implicit_phase_batch_or_history_equivalence(self):
        for key, value in (("batch", 2), ("visible_history", -1), ("kind", "matmul"), ("phase", "pd")):
            raw = case(); raw[key] = value
            with self.assertRaises((InvalidInput, Unsupported)):
                validate_case(raw)
        raw = case(); raw["phase"] = "decode"
        with self.assertRaisesRegex(InvalidInput, "one new token"):
            validate_case(raw)

    def test_unknown_execution_parameters_are_rejected(self):
        raw = config(); raw["flash_attention"] = True
        with self.assertRaises(InvalidInput):
            validate_config(raw)
        raw = case(); raw["kv_dtype"] = "f16"
        with self.assertRaises(InvalidInput):
            validate_case(raw)

    def test_signature_binds_parameters_and_keeps_full_coverage_qualified(self):
        one = signature(config(), "projection", "prefill", 3, 0, 2)
        changed = config(); changed["rope_base"] = 500000
        self.assertNotEqual(one, signature(changed, "projection", "prefill", 3, 0, 2))
        self.assertEqual(one["contract"], CONTRACT)
        self.assertFalse(one["original_model_implementation_reproduced"])
        self.assertEqual(one["scope"], "complete_declared_decoder_block")

    def test_native_command_contains_all_declared_matrix_dtypes_and_parameters(self):
        raw = case(threads=2)
        checked = config(gate="sigmoid", qk="rms"); raw["config"] = checked
        command = native_arguments(raw, 2, 5, 17)
        options = dict(zip(command[::2], command[1::2]))
        self.assertEqual(options["--stage"], "block")
        self.assertEqual(options["--gate-proj-dtype"], "f32")
        self.assertEqual(options["--rope-mode"], "normal")
        self.assertEqual(float(options["--norm-eps"]), validate_config(checked)["norm_epsilon"])

    def test_plan_does_not_replace_declared_threads_or_duplicate_case_ids(self):
        suite = dict(schema_version="dense_layer_measurement_suite_v1", cases=[case(threads=2)])
        with self.assertRaisesRegex(InvalidInput, "threads"):
            build_plan(suite, 4)
        suite["cases"].append(copy.deepcopy(suite["cases"][0]))
        with self.assertRaisesRegex(InvalidInput, "unique"):
            build_plan(suite, 2)


if __name__ == "__main__":
    unittest.main()
