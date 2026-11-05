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
    base = dict(id="small", kind="block", phase="prefill", batch=1, tokens=3,
                visible_history=0, config=config())
    base.update(changes)
    return base


def word_config(*, hidden=16, intermediate=32, vocab=7, embd="f32", head="f32"):
    """A configuration that also declares the two vocabulary-wide matrices."""
    raw = config(hidden=hidden, intermediate=intermediate)
    raw.update(vocab=vocab, token_embd_dtype=embd, lm_head_dtype=head)
    raw["weight_dtypes"].update(final_norm="f32", lm_head=head, token_embd=embd)
    return raw


class DenseBoundary(unittest.TestCase):
    """The two stages a complete token step runs outside the decoder block."""

    def test_a_vocabulary_declares_both_matrices_or_neither(self):
        self.assertNotIn("token_embd", weight_shapes(config()))
        raw = word_config(hidden=13, vocab=5)
        self.assertEqual(weight_shapes(raw)["token_embd"], (13, 5))
        self.assertEqual(weight_shapes(raw)["lm_head"], (13, 5))
        self.assertEqual(weight_shapes(raw)["final_norm"], (13,))
        raw.pop("token_embd_dtype")
        with self.assertRaisesRegex(InvalidInput, "both"):
            validate_config(raw)
        raw = word_config(); raw["weight_dtypes"].pop("token_embd")
        with self.assertRaisesRegex(InvalidInput, "weight_dtypes"):
            validate_config(raw)

    def test_the_embedding_and_the_head_may_be_stored_in_different_types(self):
        # A Q4_K_M checkpoint stores token_embd as q4_k and the LM head as q6_k.
        raw = word_config(hidden=256, vocab=512, embd="q4_k", head="q6_k")
        checked = validate_config(raw)
        embed = resident_accounting(checked, "embedding", 4, 0)
        head = resident_accounting(checked, "head", 4, 0)
        self.assertEqual(embed["weight_roles_bytes"]["token_embd"], 512*144)
        self.assertEqual(head["weight_roles_bytes"]["lm_head"], 512*210)
        self.assertEqual(head["weight_roles_bytes"]["final_norm"], 256*4)
        raw["lm_head_dtype"] = "q4_k"
        with self.assertRaisesRegex(InvalidInput, "lm_head_dtype"):
            validate_config(raw)

    def test_boundary_cases_are_history_free_and_block_cases_are_vocabulary_free(self):
        raw = case(kind="head", config=word_config())
        self.assertEqual(validate_case(raw)["config"]["vocab"], 7)
        raw = case(kind="embedding", config=word_config(), visible_history=3)
        with self.assertRaisesRegex(InvalidInput, "visible_history=0"):
            validate_case(raw)
        with self.assertRaisesRegex(InvalidInput, "must not declare"):
            validate_case(case(config=word_config()))
        with self.assertRaisesRegex(InvalidInput, "must declare config.vocab"):
            validate_case(case(kind="head"))

    def test_boundary_ports_and_operations_are_the_ends_of_a_token_step(self):
        checked = word_config(hidden=13, vocab=5)
        inputs, outputs = ports(checked, "embedding", 3)
        self.assertEqual(inputs, {"token_ids": {"dtype": "i32", "ne": (3, 1, 1, 1)}})
        self.assertEqual(outputs["hidden_out"]["ne"], (13, 3, 1, 1))
        inputs, outputs = ports(checked, "head", 3)
        self.assertEqual(inputs["hidden"]["ne"], (13, 3, 1, 1))
        self.assertEqual(outputs["logits"], {"dtype": "f32", "ne": (5, 3, 1, 1)})
        self.assertEqual(operations(checked, "embedding"),
                         ["token_embedding_lookup", "materialize_hidden"])
        self.assertEqual(operations(checked, "head"),
                         ["final_RMSNorm", "final_norm_scale", "lm_head_projection",
                          "materialize_logits"])

    def test_boundary_resident_bytes_have_an_independent_manual_oracle(self):
        checked = validate_config(word_config(hidden=256, intermediate=512, vocab=512))
        actual = resident_accounting(checked, "head", 3, 0)
        self.assertEqual(actual["weight_bytes"], 1024 + 256*512*4)
        self.assertEqual(actual["input_bytes"], 256*3*4)
        self.assertEqual(actual["output_bytes"], 512*3*4)
        self.assertEqual(actual["kv_cache_bytes"], 0)
        self.assertEqual(actual["resident_logical_bytes"],
                         525312 + 3072 + 6144)

    def test_boundary_scopes_do_not_claim_a_decoder_block(self):
        checked = word_config(hidden=13, vocab=5)
        block = signature(config(), "projection", "prefill", 3, 0, 2)
        for kind, scope in (("embedding", "resident_repeated_declared_token_embedding"),
                            ("head", "resident_repeated_declared_final_norm_and_lm_head")):
            actual = signature(checked, kind, "prefill", 3, 0, 2)
            self.assertEqual(actual["scope"], scope)
            self.assertNotEqual(actual["scope"], block["scope"])
            self.assertEqual(actual["config"]["vocab"], 5)

    def test_native_command_declares_the_vocabulary_and_both_matrix_types(self):
        raw = case(kind="head", config=word_config(hidden=256, vocab=512, embd="q4_k",
                                                   head="q6_k"), threads=2)
        options = dict(zip(*[iter(native_arguments(raw, 2, 5, 17))]*2))
        self.assertEqual(options["--vocab"], "512")
        self.assertEqual(options["--token-embd-dtype"], "q4_k")
        self.assertEqual(options["--lm-head-dtype"], "q6_k")
        self.assertNotIn("--vocab", native_arguments(case(threads=2), 2, 5, 17))

    def test_a_declared_residency_regime_changes_the_command_and_the_signature(self):
        resident_args = native_arguments(case(threads=2), 2, 5, 17)
        self.assertNotIn("--evict-bytes", resident_args)
        cold_args = native_arguments(case(threads=2), 2, 5, 17, evict_bytes=1 << 28)
        self.assertEqual(cold_args[cold_args.index("--evict-bytes") + 1], str(1 << 28))
        # Everything except the declared flag must stay identical, or a regime
        # change would silently move some other part of the protocol.
        self.assertEqual([a for a in cold_args if a != "--evict-bytes" and a != str(1 << 28)],
                         resident_args)
        warm = signature(config(), "block", "prefill", 3, 0, 2)
        cold = signature(config(), "block", "prefill", 3, 0, 2, residency="streaming")
        self.assertNotIn("residency_regime", warm)
        self.assertEqual(cold["residency_regime"], "streamed_cold_cache_before_each_sample")
        self.assertEqual({k: v for k, v in cold.items() if k != "residency_regime"}, warm)
        with self.assertRaises(InvalidInput):
            native_arguments(case(threads=2), 2, 5, 17, evict_bytes=-1)


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
        # f16 K and V: 2 bytes per element, two caches, 3 new tokens.
        self.assertEqual(actual["kv_cache_bytes"], 96)
        self.assertEqual(actual["resident_logical_bytes"], 10412)

    def test_new_kv_input_ports_and_appended_cache_capacity_are_separate(self):
        actual = resident_accounting(config(), "attention", 3, 8)
        self.assertEqual(actual["input_ports_bytes"]["k"]+actual["input_ports_bytes"]["v"], 192)
        self.assertEqual(actual["kv_new_bytes"], 96)
        self.assertEqual(actual["kv_history_bytes"], 256)

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
