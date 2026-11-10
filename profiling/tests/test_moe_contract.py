"""The sparse block contract and the documents the probe consumes.

No model file, no probe and no hardware: what has to be true here is that a
sparse layer document which disagrees with itself -- wrong bytes, wrong role,
wrong stage order -- can never be measured, exported or priced.
"""
from __future__ import annotations

import copy
import unittest

from profiling.multimodel import suite
from llm_infer_model.m2 import dense_contract, moe_contract, moe_cuda_contract
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout
import contracts


def config(**overrides) -> dict:
    base = {"hidden": 8, "intermediate": 16, "heads": 2, "kv_heads": 1, "head_dim": 4,
            "attention_gate": "none", "qk_norm": "none", "rope_mode": "normal",
            "rope_dims": 4, "rope_base": 10000.0, "norm_epsilon": 1e-5,
            "experts": 4, "top_k": 2, "shared_intermediate": 0, "normalize_topk": True,
            "attention_bias": False}
    base.update(overrides)
    base["weight_dtypes"] = {role: (dtype_for(role))
                             for role in moe_contract.weight_shapes(base)}
    return base


def dtype_for(role: str) -> str:
    # Small F32 matrices keep the arithmetic exact and the block alignment
    # irrelevant; the quantized types are exercised by their own cases below.
    return "f32"


def weights(cfg: dict) -> list[dict]:
    shapes = moe_contract.weight_shapes(cfg)
    result = []
    for role, shape in shapes.items():
        _, size = ggml_contiguous_layout(moe_contract.DTYPES[cfg["weight_dtypes"][role]],
                                         (*shape, *([1] * (4 - len(shape)))))
        result.append({"id": "W_" + role, "name": "W_" + role,
                       "dtype": cfg["weight_dtypes"][role], "shape": list(shape),
                       "storage_bytes": size, "initial_location": "dram"})
    return result


def layer_document(**overrides) -> dict:
    cfg = config(**overrides)
    description = {"layer_index": 1, "config": cfg, "weights": weights(cfg), "defaults": [],
                   "contract": moe_contract, "ffn": "moe"}
    return suite.layer_document({"architecture": "granitemoe"}, description,
                                model_id="toy", threads=2)


# F32/F16 keep the tiny fixture legal: q4_k rows need ne0 % 256 == 0.
BOUNDARY = {"vocab": 6, "token_embd_dtype": "f32", "lm_head_dtype": "f16"}


class TheLayerDocumentIsVerifiedAgainstItself(unittest.TestCase):
    def test_a_consistent_sparse_layer_round_trips(self):
        document = layer_document()
        self.assertEqual(document["format_version"], "moe-layer-0.1")
        self.assertEqual(document["ffn"], "moe")
        self.assertEqual([task["kind"] for task in document["tasks"]],
                         ["projection", "attention", "moe_ffn"])
        layer = moe_contract.load_layer_document(document)
        self.assertEqual(layer.id, "toy")
        self.assertEqual(layer.physical.threads, 2)
        self.assertEqual(layer.physical.config["experts"], 4)
        self.assertTrue(layer.physical.config["normalize_topk"])
        roles = {weight["id"] for weight in layer.weights}
        self.assertIn("W_router", roles)
        self.assertIn("W_down_exps", roles)
        self.assertNotIn("W_gate_shexp", roles)

    def test_a_declared_placement_reaches_the_sparse_signature(self):
        # A sparse block is as sensitive to which logical CPUs ran it as a dense
        # one, and the declaration must reach the signature the same way -- and
        # only when it is declared, so every table measured before the dimension
        # existed keeps its identity.
        free = moe_contract.signature(config(), "moe_ffn", "decode", 1, 0, 2)
        pinned = moe_contract.signature(config(), "moe_ffn", "decode", 1, 0, 2,
                                        cpu_ids=[2, 4])
        self.assertNotIn("cpu_mask", free)
        self.assertEqual(pinned["cpu_mask"], [2, 4])
        self.assertEqual(pinned["thread_placement"], "explicit_strict_cpu_mask")
        self.assertEqual({key: value for key, value in pinned.items()
                          if key not in ("cpu_mask", "thread_placement")}, free)
        document = layer_document()
        document["physical"]["cpu_ids"] = [2, 4]
        layer = moe_contract.load_layer_document(document)
        self.assertEqual(layer.physical.cpu_ids, (2, 4))
        document["physical"]["cpu_ids"] = [2]
        with self.assertRaisesRegex(InvalidInput, "every worker"):
            moe_contract.load_layer_document(document)

    def test_the_shared_exporter_builds_a_sparse_layer_signature(self):
        # One exporter serves both contracts, so the sparse module has to answer
        # every signature question the dense one does -- including the process
        # scope -- or no measured sparse table can be exported at all.
        import export_dense_layer as exporter
        from llm_infer_model.m2.costs import MacroShape
        shape = MacroShape("decode", 1, 1, 0)
        layer = moe_contract.load_layer_document(layer_document())
        self.assertEqual(exporter.model_signature(layer, "moe_ffn", shape),
                         moe_contract.signature(layer.physical.config, "moe_ffn",
                                                "decode", 1, 0, 2))
        document = layer_document()
        document["physical"]["cpu_ids"] = [2, 4]
        pinned = moe_contract.load_layer_document(document)
        self.assertEqual(exporter.model_signature(pinned, "moe_ffn", shape)["cpu_mask"], [2, 4])
        self.assertNotIn("cpu_mask", exporter.model_signature(layer, "moe_ffn", shape))

    def test_arithmetic_that_disagrees_with_the_config_is_refused(self):
        for field, value in (("hidden_size", 9), ("hidden_bytes_per_token", 31),
                             ("kv_bytes_per_token", 15)):
            document = layer_document()
            document[field] = value
            with self.assertRaisesRegex(InvalidInput, field):
                moe_contract.load_layer_document(document)

    def test_a_weight_whose_bytes_or_dtype_do_not_match_is_refused(self):
        document = layer_document()
        for index, weight in enumerate(document["weights"]):
            if weight["id"] == "W_gate_exps":
                weight["storage_bytes"] += 1
                break
        with self.assertRaisesRegex(InvalidInput, "storage_bytes"):
            moe_contract.load_layer_document(document)

        document = layer_document()
        for weight in document["weights"]:
            if weight["id"] == "W_router":
                weight["dtype"] = "f16"
        with self.assertRaisesRegex(InvalidInput, "dtype disagrees"):
            moe_contract.load_layer_document(document)

    def test_a_missing_role_is_a_missing_weight_not_a_zero_price(self):
        document = layer_document()
        document["weights"] = [weight for weight in document["weights"]
                               if weight["id"] != "W_down_exps"]
        with self.assertRaisesRegex(InvalidInput, "not exactly the sparse block"):
            moe_contract.load_layer_document(document)

    def test_the_stage_order_is_the_declared_chain(self):
        document = layer_document()
        document["tasks"][2]["depends_on"] = ["projection"]
        with self.assertRaisesRegex(InvalidInput, "depends_on"):
            moe_contract.load_layer_document(document)
        document = layer_document()
        document["tasks"] = document["tasks"][:2]
        with self.assertRaisesRegex(InvalidInput, "tasks"):
            moe_contract.load_layer_document(document)

    def test_a_shared_expert_adds_its_four_matrices_to_the_same_stage(self):
        document = layer_document(shared_intermediate=6)
        layer = moe_contract.load_layer_document(document)
        self.assertEqual(set(layer.physical.weight_groups["moe_ffn"]),
                         {"post_attention_norm", "router", "gate_exps", "up_exps",
                          "down_exps", "gate_shexp", "up_shexp", "down_shexp",
                          "gate_inp_shexp"})
        roles = {weight["id"] for weight in layer.weights}
        self.assertIn("W_gate_inp_shexp", roles)

    def test_q8_0_is_a_declared_storage_type_only_where_its_blocks_fit(self):
        cfg = config(intermediate=32)
        cfg["weight_dtypes"]["down_exps"] = "q8_0"
        self.assertEqual(moe_contract.validate_config(cfg)["weight_dtypes"]["down_exps"], "q8_0")
        cfg = config(intermediate=48)  # 48 % 32 != 0
        cfg["weight_dtypes"]["down_exps"] = "q8_0"
        with self.assertRaises(Unsupported):
            moe_contract.validate_config(cfg)


class TheSuiteSaysWhichBlockItMeasures(unittest.TestCase):
    def description(self, **overrides) -> dict:
        cfg = config(**overrides)
        return {"config": cfg, "weights": weights(cfg), "boundary": BOUNDARY,
                "defaults": [], "contract": moe_contract, "ffn": "moe", "layer_index": 1}

    def test_the_third_stage_is_the_sparse_feed_forward(self):
        description = self.description()
        document = suite.suite_document(description, {"batch": 1,
                                         "prefill": [{"tokens": 4}],
                                         "decode": [{"tokens": 1, "visible_history": 0},
                                                    {"tokens": 1, "visible_history": 64}]},
                                        model_id="toy")
        self.assertEqual(document["schema_version"], moe_contract.SUITE_SCHEMA)
        kinds = {case["kind"] for case in document["cases"]}
        self.assertEqual(kinds, {"projection", "attention", "moe_ffn", "block",
                                 "embedding", "head"})
        self.assertFalse(any(case["kind"] == "dense_ffn" for case in document["cases"]))
        moe_contract.build_plan(document, 2)

    def test_the_exporter_receives_the_sparse_contract(self):
        document = suite.suite_document(self.description(), {"batch": 1,
                                        "prefill": [{"tokens": 4}], "decode": []},
                                        model_id="toy")
        self.assertIs(contracts.for_suite(document), moe_contract)
        self.assertIs(contracts.for_suite({"schema_version": dense_contract.SUITE_SCHEMA}),
                      dense_contract)
        with self.assertRaisesRegex(ValueError, "unknown measurement suite schema"):
            contracts.for_suite({"schema_version": "something-else"})


class TheProbeIsInvokedForTheDeclaredGraph(unittest.TestCase):
    def test_the_native_arguments_name_the_experts_and_the_routing_variant(self):
        case = moe_contract.build_plan({"schema_version": moe_contract.SUITE_SCHEMA, "cases": [
            {"id": "c", "kind": "moe_ffn", "phase": "decode", "batch": 1, "tokens": 1,
             "visible_history": 0, "config": config()}]}, 2)[0]
        arguments = moe_contract.native_arguments(case, 1, 3, 7)
        for flag, value in (("--experts", "4"), ("--top-k", "2"),
                            ("--shared-intermediate", "0"), ("--normalize-topk", "1"),
                            ("--stage", "moe_ffn")):
            index = arguments.index(flag)
            self.assertEqual(arguments[index + 1], value)

    def test_a_declared_residency_regime_reaches_the_expert_command(self):
        case = moe_contract.build_plan({"schema_version": moe_contract.SUITE_SCHEMA, "cases": [
            {"id": "c", "kind": "moe_ffn", "phase": "decode", "batch": 1, "tokens": 1,
             "visible_history": 0, "config": config()}]}, 2)[0]
        resident = moe_contract.native_arguments(case, 1, 3, 7)
        self.assertNotIn("--evict-bytes", resident)
        cold = moe_contract.native_arguments(case, 1, 3, 7, evict_bytes=1 << 28)
        self.assertEqual(cold[cold.index("--evict-bytes") + 1], str(1 << 28))
        self.assertEqual([a for a in cold if a not in ("--evict-bytes", str(1 << 28))], resident)

    def test_the_operation_list_covers_the_selected_experts_and_the_variant(self):
        operations = moe_contract.operations(config(), "moe_ffn")
        self.assertIn("top_k_weight_normalize", operations)
        self.assertEqual(operations.count("expert_accumulate"), 1)  # top_k - 1
        self.assertEqual(moe_contract.operations(config(normalize_topk=False),
                                                 "moe_ffn").count("top_k_weight_normalize"), 0)
        self.assertNotIn("shared_sigmoid", moe_contract.operations(config(), "moe_ffn"))
        self.assertIn("shared_sigmoid", moe_contract.operations(
            config(shared_intermediate=6), "moe_ffn"))

    def test_the_block_is_exactly_the_three_declared_stages(self):
        operations = moe_contract.operations(config(), "block")
        self.assertEqual(operations, [op for stage in moe_contract.STAGES[:3]
                                      for op in moe_contract.operations(config(), stage)])

    def test_resident_bytes_include_every_selected_expert_tensor(self):
        accounting = moe_contract.resident_accounting(config(), "moe_ffn", 1, 0)
        cfg = config()
        shapes = moe_contract.weight_shapes(cfg)
        self.assertEqual(accounting["weight_bytes"], sum(
            ggml_contiguous_layout("GGML_TYPE_F32",
                                   (*shapes[role], *([1] * (4 - len(shapes[role])))))[1]
            for role in moe_contract.stage_weight_roles(cfg, "moe_ffn")))

    def test_qkv_biases_are_part_of_the_declared_projection(self):
        cfg = config(attention_bias=True)
        self.assertEqual({role: shape for role, shape in moe_contract.weight_shapes(cfg).items()
                          if role in moe_contract.BIAS_ROLES},
                         {"q_bias": (8,), "k_bias": (4,), "v_bias": (4,)})
        self.assertEqual(moe_contract.stage_weight_roles(cfg, "projection")[-3:],
                         moe_contract.BIAS_ROLES)
        operations = moe_contract.operations(cfg, "projection")
        self.assertEqual(operations[operations.index("V_projection") + 1:][:3],
                         ["Q_bias_add", "K_bias_add", "V_bias_add"])
        # A stage that does not project reads no bias.
        self.assertNotIn("W_q_bias", moe_contract.stage_weight_roles(cfg, "moe_ffn"))
        case = moe_contract.build_plan({"schema_version": moe_contract.SUITE_SCHEMA, "cases": [
            {"id": "c", "kind": "projection", "phase": "decode", "batch": 1, "tokens": 1,
             "visible_history": 0, "config": cfg}]}, 2)[0]
        arguments = moe_contract.native_arguments(case, 1, 3, 7)
        self.assertEqual(arguments[arguments.index("--attention-bias") + 1], "1")

    def test_a_bias_that_is_declared_but_missing_is_an_error(self):
        with self.assertRaisesRegex(InvalidInput, "not exactly the sparse block"):
            document = layer_document(attention_bias=True)
            document["weights"] = [weight for weight in document["weights"]
                                   if weight["id"] != "W_k_bias"]
            moe_contract.load_layer_document(document)

    def test_the_routing_variant_is_declared_per_architecture(self):
        self.assertTrue(moe_contract.routing_for("granitemoe")["normalize_topk"])
        self.assertFalse(moe_contract.routing_for("qwen2moe")["normalize_topk"])
        # OLMoE's config.json states norm_topk_prob=false and the pinned
        # llama.cpp builds its expert mix with norm_w=false: the selected
        # probabilities are used as softmax emits them.
        self.assertFalse(moe_contract.routing_for("olmoe")["normalize_topk"])
        with self.assertRaisesRegex(Unsupported, "routing variant"):
            moe_contract.routing_for("mixtral")


class TheContractDispatchCannotSilentlyPickDense(unittest.TestCase):
    def test_a_layer_is_dispatched_by_its_own_contract_field(self):
        layer = moe_contract.load_layer_document(layer_document())
        self.assertIs(contracts.for_layer(layer), moe_contract)
        dense = type("Dense", (), {"physical": type("P", (), {"contract": dense_contract.CONTRACT})()})
        self.assertIs(contracts.for_layer(dense), dense_contract)
        with self.assertRaisesRegex(ValueError, "unknown block contract"):
            contracts.for_contract("something-else")


class TheCUDAEnvelopeChangesOnlyTheDevice(unittest.TestCase):
    """The accelerator envelope must not restate the block, only where it ran."""

    def plan(self, **case):
        raw = {"id": "c", "kind": "moe_ffn", "phase": "decode", "batch": 1, "tokens": 1,
               "visible_history": 0, "config": config()}
        raw.update(case)
        return moe_cuda_contract.build_plan(
            {"schema_version": moe_contract.SUITE_SCHEMA, "cases": [raw]}, 0,
            kv_dtype="f16")

    def test_the_variant_names_the_stage_and_the_cache_type(self):
        self.assertEqual(moe_cuda_contract.variant("moe_ffn", "f16"),
                         "ggml_cuda_moe_materialized_moe_ffn_v2")
        self.assertEqual(moe_cuda_contract.variant("projection", "f32"),
                         "ggml_cuda_moe_materialized_projection_v1")
        with self.assertRaises(Unsupported):
            moe_cuda_contract.variant("moe_ffn", "bf16")

    def test_the_signature_moves_the_process_fields_and_nothing_else(self):
        cfg = config()
        cpu = moe_contract.signature(cfg, "moe_ffn", "decode", 1, 0, threads=1)
        gpu = moe_cuda_contract.signature(cfg, "moe_ffn", "decode", 1, 0, cuda_device=0,
                                          kv_dtype="f16")
        self.assertEqual(gpu["contract"], moe_cuda_contract.CONTRACT)
        self.assertEqual((gpu["device"], gpu["backend"]), ("gpu", "CUDA0"))
        self.assertEqual(gpu["timing_scope"], "single_completed_cuda_graph_wall_ns")
        self.assertEqual(gpu["implementation_variant"],
                         "ggml_cuda_moe_materialized_moe_ffn_v2")
        for field in ("threads_requested", "threads_applied", "thread_management"):
            self.assertNotIn(field, gpu)
        for field in ("config", "operations", "scope", "tokens", "visible_history",
                      "activation_dtype", "position_dtype", "excluded_operations"):
            self.assertEqual(gpu[field], cpu[field], field)

    def test_the_cache_type_is_part_of_the_price_identity(self):
        cfg = config()
        f32 = moe_cuda_contract.signature(cfg, "moe_ffn", "decode", 1, 0, cuda_device=0,
                                          kv_dtype="f32")
        f16 = moe_cuda_contract.signature(cfg, "moe_ffn", "decode", 1, 0, cuda_device=0,
                                          kv_dtype="f16")
        self.assertNotEqual(f32, f16)

    def test_an_enabled_graph_is_refused_rather_than_timed(self):
        with self.assertRaises(Unsupported):
            moe_cuda_contract.signature(config(), "moe_ffn", "decode", 1, 0, cuda_device=0,
                                        cuda_graphs="enabled")
        with self.assertRaises(Unsupported):
            moe_cuda_contract.build_plan(
                {"schema_version": moe_contract.SUITE_SCHEMA, "cases": []}, 0,
                cuda_graphs="enabled")

    def test_a_cuda_case_carries_no_cpu_thread_parameter(self):
        with self.assertRaises(Unsupported):
            moe_cuda_contract.build_plan({"schema_version": moe_contract.SUITE_SCHEMA, "cases": [
                {"id": "c", "kind": "moe_ffn", "phase": "decode", "batch": 1, "tokens": 1,
                 "visible_history": 0, "config": config(), "threads": 2}]}, 0)

    def test_the_build_plan_declares_the_device_and_refuses_a_disagreeing_case(self):
        case = self.plan()[0]
        self.assertEqual((case["cuda_device"], case["cuda_graphs"], case["kv_dtype"]),
                         (0, "disabled", "f16"))
        with self.assertRaisesRegex(InvalidInput, "CUDA device"):
            moe_cuda_contract.build_plan({"schema_version": moe_contract.SUITE_SCHEMA, "cases": [
                {"id": "c", "kind": "moe_ffn", "phase": "decode", "batch": 1, "tokens": 1,
                 "visible_history": 0, "config": config(), "cuda_device": 1}]}, 0)
        with self.assertRaisesRegex(InvalidInput, "schema_version"):
            moe_cuda_contract.build_plan({"schema_version": dense_contract.SUITE_SCHEMA,
                                          "cases": []}, 0)

    def test_the_native_command_is_the_cpu_graph_on_the_accelerator(self):
        case = self.plan(kind="moe_ffn")[0]
        arguments = moe_cuda_contract.native_arguments(case, 1, 3, 7)
        self.assertNotIn("--threads", arguments)
        for flag, value in (("--cuda-device", "0"), ("--cuda-graphs", "disabled"),
                            ("--kv-dtype", "f16"), ("--experts", "4"), ("--top-k", "2")):
            self.assertEqual(arguments[arguments.index(flag) + 1], value, flag)
        math_case = {key: value for key, value in case.items()
                     if key not in ("cuda_device", "cuda_graphs", "kv_dtype")}
        cpu = moe_contract.native_arguments({**math_case, "threads": 1}, 1, 3, 7)

        def without(arguments, flags):
            kept, index = [], 0
            while index < len(arguments):
                if arguments[index] in flags:
                    index += 2
                    continue
                kept.append(arguments[index])
                index += 1
            return kept

        self.assertEqual(
            sorted(without(arguments, {"--cuda-device", "--cuda-graphs", "--kv-dtype"})),
            sorted(without(cpu, {"--threads", "--kv-dtype"})))

    def test_a_biased_sparse_model_passes_its_bias_types(self):
        case = self.plan(kind="projection", config=config(attention_bias=True))[0]
        arguments = moe_cuda_contract.native_arguments(case, 1, 3, 7)
        for flag in ("--q-bias-dtype", "--k-bias-dtype", "--v-bias-dtype"):
            self.assertEqual(arguments[arguments.index(flag) + 1], "f32", flag)

    def test_the_cache_element_type_reaches_the_accounting(self):
        cfg = config(heads=2, kv_heads=1, head_dim=4)
        f16 = moe_cuda_contract.resident_accounting(cfg, "attention", 1, 32, kv_dtype="f16")
        f32 = moe_cuda_contract.resident_accounting(cfg, "attention", 1, 32, kv_dtype="f32")
        per_token = 2 * cfg["kv_heads"] * cfg["head_dim"]
        self.assertEqual(f32["kv_cache_bytes"] - f16["kv_cache_bytes"],
                         33 * per_token * 2)
        sparse = moe_cuda_contract.resident_accounting(cfg, "moe_ffn", 1, 32, kv_dtype="f16")
        self.assertEqual(sparse["kv_cache_bytes"], 0)

    def test_the_sparse_stage_declares_one_materialized_input_and_output(self):
        inputs, outputs = moe_cuda_contract.ports(config(), "moe_ffn", 4)
        self.assertEqual(list(inputs), ["attention_hidden"])
        self.assertEqual(list(outputs), ["hidden_out"])
        self.assertNotIn("key_cache", outputs)


if __name__ == "__main__":
    unittest.main()
