"""Full Dense physical boundaries, residual lifetimes and strict cost composition."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
import unittest

from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.dense_contract import (
    CONTRACT, resident_accounting, stage_weight_roles, validate_config, weight_shapes,
)
from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.full_physical import signature_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import (
    parse_invocation_document, parse_layer_document, state_in_from_state_out,
)
from llm_infer_model.tensor.engine import (
    Action, ACTION_ADVANCE, ACTION_COMPUTE, ACTION_EVICT, evaluate_mapping,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"f32": "GGML_TYPE_F32", "f16": "GGML_TYPE_F16",
          "q4_k": "GGML_TYPE_Q4_K", "q6_k": "GGML_TYPE_Q6_K"}


def full_layer_document(*, gate="none", qk_norm="none", large=False):
    config = dict(hidden=256 if large else 16, intermediate=256 if large else 32,
                  heads=4, kv_heads=2, head_dim=64 if large else 4,
                  attention_gate=gate, qk_norm=qk_norm, rope_mode="normal",
                  rope_dims=64 if large else 4, rope_base=10000.0,
                  norm_epsilon=1e-6)
    shapes = weight_shapes(config)
    config["weight_dtypes"] = {role: ("q6_k" if large and role == "down" else
                                    "q4_k" if large and len(shape) == 2 else "f32")
                               for role, shape in shapes.items()}
    config = validate_config(config)
    groups = {stage: {role: "W_"+role for role in stage_weight_roles(config, stage)}
              for stage in ("projection", "attention", "dense_ffn")}
    weights = []
    for stage in groups.values():
        for role, name in stage.items():
            shape = shapes[role]
            dtype = config["weight_dtypes"][role]
            _, size = ggml_contiguous_layout(DTYPES[dtype], (*shape, *([1]*(4-len(shape)))))
            weights.append(dict(id=name, name=name, dtype=dtype, shape=list(shape),
                                storage_bytes=size, initial_location="dram"))
    return dict(format_version="m2-layer-0.3", id="full-small", provenance="synthetic_layer",
                hidden_size=config["hidden"], hidden_bytes_per_token=4*config["hidden"],
                kv_bytes_per_token=8*config["head_dim"]*config["kv_heads"],
                attention="full_causal", ffn="dense", weights=weights,
                tasks=[dict(id="proj", kind="projection", depends_on=[]),
                       dict(id="attn", kind="attention", depends_on=["proj"]),
                       dict(id="ffn", kind="dense_ffn", depends_on=["attn"])],
                physical=dict(contract=CONTRACT, scope="complete_declared_decoder_block",
                              input_boundary="hidden_activation", port_layout="materialized_stage_ports",
                              kv_allocation="logical_blocks", activation_dtype="f32", kv_dtype="f32",
                              config=config, threads=2, weight_groups=groups, excluded_operations=[]))


def full_invocation(layer, *, phase="prefill", state=None, capacity=1 << 28):
    raw = json.loads((ROOT / "mapping/examples/m2/fcd-prefill-3.json").read_text(encoding="utf-8"))
    raw.update(format_version="m2-layer-0.3", layer=layer.id, phase=phase)
    raw["hardware"]["dram_capacity_bytes"] = capacity
    if phase == "decode":
        raw["requests"][0].update(tokens=1, position_start=3)
    invocation = parse_invocation_document(raw)
    return dataclasses.replace(invocation, state_in=state) if state is not None else invocation


def full_cost_table(layer, invocation, *, scratch=(100, 200, 300), durations=(1000, 2000, 3000)):
    request = invocation.requests[0]
    shape = MacroShape(invocation.phase, invocation.batch, request.tokens, request.position_start)
    entries = []
    for i, task in enumerate(layer.tasks):
        envelope = signature_for(layer, task.kind, shape)
        entries.append(dict(kind=task.kind, device="cpu", phase=shape.phase, batch=shape.batch,
                            tokens=shape.tokens, visible_history=shape.visible_history,
                            duration_ns=durations[i], workspace_bytes=scratch[i],
                            provenance="synthetic_layer", accounting="per_request",
                            physical_signature=envelope, contract_hash=canonical_contract_hash(envelope)))
    return parse_cost_table(dict(format_version="m2-layer-0.3", source="full-boundary-test", entries=entries))


def retained_history_cases():
    """Three existing authoring formats, all with actual prefill-derived state."""
    if __package__:
        from .test_m2_physical import layer_document, invocation_document, table_for
    else:
        from test_m2_physical import layer_document, invocation_document, table_for
    full = parse_layer_document(full_layer_document())
    full_inv = full_invocation(full)
    partial = parse_layer_document(layer_document())
    partial_inv = invocation_document(partial)
    first_calls = [
        ("synthetic", load_and_compile(ROOT / "mapping/examples/m2/fcd-prefill-3.json")),
        ("partial", compile_invocation(partial, partial_inv, table_for(partial, partial_inv))),
        ("full", compile_invocation(full, full_inv, full_cost_table(full, full_inv))),
    ]
    for name, first in first_calls:
        result = run_invocation(first)
        if result.status != "valid":
            raise AssertionError(f"{name} prefill did not succeed: {result.evaluation.reason}")
        history = state_in_from_state_out(result.state_out)
        invocation = dataclasses.replace(first.invocation, id=first.invocation.id+"-decode", phase="decode",
            requests=tuple(dataclasses.replace(request, tokens=1, position_start=3)
                           for request in first.invocation.requests), state_in=history)
        table = (full_cost_table(first.layer, invocation) if name == "full" else
                 table_for(first.layer, invocation) if name == "partial" else first.cost_table)
        yield name, compile_invocation(first.layer, invocation, table)


class FullDenseBoundary(unittest.TestCase):
    def compile(self, raw=None, *, capacity=1 << 28):
        layer = parse_layer_document(raw or full_layer_document())
        invocation = full_invocation(layer, capacity=capacity)
        return compile_invocation(layer, invocation, full_cost_table(layer, invocation))

    def test_projection_outputs_have_true_qkv_dimensions_and_one_producer(self):
        compiled = self.compile()
        tensors = compiled.scenario.workload.tensor_by_id
        projection, attention, ffn = compiled.scenario.workload.operations
        self.assertEqual(projection.output, "t:full-small:r0:proj.q")
        self.assertEqual(set(projection.extra_outputs), {"t:full-small:r0:proj.k", "t:full-small:r0:proj.v"})
        self.assertEqual(tensors[projection.output].ne, (4, 3, 4, 1))
        self.assertEqual(tensors["t:full-small:r0:proj.k"].ne, (4, 3, 2, 1))
        self.assertTrue(set(projection.outputs).issubset(attention.inputs))
        self.assertIn(attention.output, ffn.inputs)
        self.assertEqual(len(compiled.tasks), 3)
        self.assertNotIn("t:full-small:r0:in.q", tensors)

    def test_original_hidden_is_retained_for_attention_residual(self):
        compiled = self.compile()
        projection, attention, ffn = compiled.scenario.workload.operations
        hidden = "t:full-small:r0:in.hidden"
        self.assertIn(hidden, projection.inputs)
        self.assertIn(hidden, attention.inputs)
        self.assertNotIn(hidden, ffn.inputs)
        self.assertEqual(set(ffn.inputs), {attention.output, "w:full-small:W_post_attention_norm",
                                          "w:full-small:W_gate", "w:full-small:W_up", "w:full-small:W_down"})

    def test_norm_vectors_and_rope_positions_are_physical_allocations(self):
        compiled = self.compile(full_layer_document(qk_norm="rms"))
        tensors = compiled.scenario.workload.tensor_by_id
        self.assertEqual(tensors["w:full-small:W_input_norm"].ne, (16, 1, 1, 1))
        self.assertEqual(tensors["w:full-small:W_q_norm"].ne, (4, 1, 1, 1))
        positions = tensors["t:full-small:r0:in.positions"]
        self.assertEqual(positions.dtype, "GGML_TYPE_I32")
        self.assertEqual(positions.storage_bytes, 12)
        self.assertIn(positions.id, compiled.scenario.workload.operations[0].inputs)
        self.assertEqual(compiled.physical_boundary["absolute_positions"], {"r0": [0, 1, 2]})

    def test_gate_has_its_own_projection_matrix_and_materialized_output(self):
        compiled = self.compile(full_layer_document(gate="sigmoid"))
        projection, attention, _ = compiled.scenario.workload.operations
        self.assertIn("w:full-small:W_gate_proj", projection.inputs)
        self.assertIn("t:full-small:r0:proj.gate", projection.extra_outputs)
        self.assertIn("t:full-small:r0:proj.gate", attention.inputs)
        self.assertNotIn("w:full-small:W_gate_proj", attention.inputs)

    def test_isolated_cost_sum_is_explicit_and_is_not_joint_block_cost(self):
        compiled = self.compile(full_layer_document(gate="sigmoid", qk_norm="rms"))
        result = run_invocation(compiled)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.makespan_ns, 6000)
        boundary = compiled.physical_boundary
        self.assertTrue(boundary["complete_declared_decoder_block"])
        self.assertFalse(boundary["original_model_kernel_reproduced"])
        self.assertIn("three_isolated_macro_costs", boundary["timing_composition"])
        self.assertEqual(boundary["excluded_operations"], [])

    def test_prefill_decode_reads_old_kv_and_uses_absolute_rope_positions(self):
        first = self.compile()
        result = run_invocation(first)
        self.assertEqual(result.status, "valid")
        state = state_in_from_state_out(result.state_out)
        invocation = full_invocation(first.layer, phase="decode", state=state)
        second = compile_invocation(first.layer, invocation, full_cost_table(first.layer, invocation))
        old_ids = {block.id for block in state.requests[0].blocks}
        projection, attention, _ = second.scenario.workload.operations
        self.assertTrue(old_ids.issubset(attention.inputs))
        self.assertTrue(old_ids.isdisjoint(projection.inputs))
        self.assertEqual(second.physical_boundary["absolute_positions"], {"r0": [3]})
        next_result = run_invocation(second)
        self.assertEqual(next_result.status, "valid")
        self.assertEqual(next_result.state_out.requests[0].next_position, 4)
        self.assertEqual(next_result.state_out.requests[0].version, 3)
        self.assertEqual(next_result.state_out.total_kv_bytes("r0"), 256)

    def test_live_ports_kv_and_scratch_have_exact_capacity_boundary(self):
        raw = full_layer_document(gate="sigmoid", qk_norm="rms")
        logical = resident_accounting(raw["physical"]["config"], "block", 3, 0)["resident_logical_bytes"]
        exact = logical + 300
        self.assertEqual(run_invocation(self.compile(raw, capacity=exact)).status, "valid")
        self.assertNotEqual(run_invocation(self.compile(raw, capacity=exact-1)).status, "valid")

    def test_new_kv_is_a_separate_allocation_from_projection_kv(self):
        compiled = self.compile()
        tensors = compiled.scenario.workload.tensor_by_id
        projection_kv = sum(tensors[f"t:full-small:r0:proj.{port}"].storage_bytes for port in ("k", "v"))
        persistent = sum(tensors[name].storage_bytes for name in compiled.block_ids)
        self.assertEqual(projection_kv, 192)
        self.assertEqual(persistent, 192)
        self.assertEqual(len(compiled.scenario.workload.operations[1].extra_outputs), 3)

    def test_quantized_matrix_storage_and_f32_norms_are_preserved(self):
        compiled = self.compile(full_layer_document(large=True))
        tensors = compiled.scenario.workload.tensor_by_id
        self.assertEqual(tensors["w:full-small:W_q"].dtype, "GGML_TYPE_Q4_K")
        self.assertEqual(tensors["w:full-small:W_down"].dtype, "GGML_TYPE_Q6_K")
        self.assertEqual(tensors["w:full-small:W_input_norm"].dtype, "GGML_TYPE_F32")
        self.assertEqual(tensors["w:full-small:W_q"].storage_bytes, 36864)

    def test_query_width_may_differ_from_hidden_width(self):
        raw = full_layer_document()
        config = raw["physical"]["config"]
        config["heads"] = 6
        config["weight_dtypes"] = {role: "f32" for role in weight_shapes(config)}
        shapes = weight_shapes(config)
        for weight in raw["weights"]:
            role = weight["id"][2:]
            weight["shape"] = list(shapes[role])
            weight["storage_bytes"] = 4 * math.prod(shapes[role])
        compiled = self.compile(raw)
        self.assertEqual(compiled.scenario.workload.tensor_by_id["t:full-small:r0:proj.q"].ne, (4, 3, 6, 1))
        self.assertEqual(compiled.scenario.workload.tensor_by_id["w:full-small:W_output"].ne, (24, 16, 1, 1))

    def test_physical_config_changes_cannot_reuse_an_old_cost_table(self):
        layer = parse_layer_document(full_layer_document())
        invocation = full_invocation(layer)
        table = full_cost_table(layer, invocation)
        for key, value in (("rope_base", 20000.0), ("norm_epsilon", 1e-5)):
            raw = full_layer_document()
            raw["physical"]["config"][key] = value
            changed = parse_layer_document(raw)
            with self.assertRaisesRegex(Unsupported, "不匹配"):
                compile_invocation(changed, invocation, table)
        changed = dataclasses.replace(layer, physical=dataclasses.replace(layer.physical, threads=4))
        with self.assertRaises(Unsupported):
            compile_invocation(changed, invocation, table)

    def test_history_is_an_exact_cost_lookup_dimension(self):
        first = self.compile()
        state = state_in_from_state_out(run_invocation(first).state_out)
        decode = full_invocation(first.layer, phase="decode", state=state)
        with self.assertRaises(Unsupported):
            compile_invocation(first.layer, decode, first.cost_table)

    def test_full_json_validation_cost_opt_in_preserves_the_quality_gate(self):
        layer = parse_layer_document(full_layer_document())
        regular = full_invocation(layer)
        table = dataclasses.replace(full_cost_table(layer, regular), validation_only=True)
        with self.assertRaisesRegex(Unsupported, "validation_only"):
            compile_invocation(layer, regular, table)
        raw = json.loads((ROOT / "mapping/examples/m2/fcd-prefill-3.json").read_text(encoding="utf-8"))
        raw.update(format_version="m2-layer-0.3", layer=layer.id,
                   allow_validation_only_costs=True)
        raw["hardware"]["dram_capacity_bytes"] = 1 << 28
        explicit = parse_invocation_document(raw)
        self.assertTrue(explicit.allow_validation_only_costs)
        self.assertEqual(run_invocation(compile_invocation(layer, explicit, table)).status, "valid")
        raw["allow_validation_only_costs"] = "true"
        with self.assertRaises(InvalidInput):
            parse_invocation_document(raw)

    def test_missing_norm_or_matrix_bytes_are_rejected(self):
        raw = full_layer_document()
        raw["weights"] = [w for w in raw["weights"] if w["id"] != "W_input_norm"]
        with self.assertRaisesRegex(InvalidInput, "inventory"):
            parse_layer_document(raw)
        raw = full_layer_document()
        raw["weights"][0]["storage_bytes"] += 4
        with self.assertRaisesRegex(InvalidInput, "actual layout"):
            parse_layer_document(raw)

    def test_non_f32_norm_and_undeclared_optional_weights_are_rejected(self):
        raw = full_layer_document()
        raw["physical"]["config"]["weight_dtypes"]["input_norm"] = "f16"
        with self.assertRaises(Unsupported):
            parse_layer_document(raw)
        raw = full_layer_document()
        raw["physical"]["weight_groups"]["projection"]["gate_proj"] = "W_extra_gate"
        with self.assertRaises(InvalidInput):
            parse_layer_document(raw)

    def test_aliases_padding_and_other_rope_variants_are_unsupported(self):
        for key, value in (("port_layout", "aliased_views"), ("kv_allocation", "padded_preallocation"),
                           ("kv_dtype", "f16")):
            raw = full_layer_document(); raw["physical"][key] = value
            with self.assertRaises(Unsupported):
                parse_layer_document(raw)
        raw = full_layer_document(); raw["physical"]["config"]["rope_mode"] = "mrope"
        with self.assertRaises(Unsupported):
            parse_layer_document(raw)

    def test_residual_chain_cannot_be_rewired_and_ports_cannot_collide(self):
        raw = full_layer_document()
        raw["tasks"][1]["depends_on"] = []
        with self.assertRaisesRegex(InvalidInput, "task DAG"):
            parse_layer_document(raw)
        raw = full_layer_document()
        raw["tasks"][1]["id"] = "proj.k"
        raw["tasks"][2]["depends_on"] = ["proj.k"]
        with self.assertRaisesRegex(InvalidInput, "collide"):
            self.compile(raw)

    def test_full_format_cannot_reinterpret_an_existing_partial_or_synthetic_invocation(self):
        layer = parse_layer_document(full_layer_document())
        invocation = full_invocation(layer)
        for version in ("m2-layer-0.1", "m2-layer-0.2"):
            with self.assertRaisesRegex(InvalidInput, "format_version mismatch"):
                compile_invocation(layer, dataclasses.replace(invocation, format_version=version),
                                   full_cost_table(layer, invocation))
        raw = full_layer_document(); raw["format_version"] = "m2-layer-0.2"
        with self.assertRaises((InvalidInput, Unsupported)):
            parse_layer_document(raw)

    def test_dataclass_construction_cannot_bypass_contract_validation(self):
        compiled = self.compile()
        for key, value in (("scope", "partial_decoder_block"), ("contract", "invented"),
                           ("threads", True), ("excluded_operations", ("RoPE",))):
            changed = dataclasses.replace(compiled.layer,
                physical=dataclasses.replace(compiled.layer.physical, **{key: value}))
            with self.assertRaises((InvalidInput, Unsupported)):
                compile_invocation(changed, compiled.invocation, compiled.cost_table)


    def test_all_formats_preserve_old_kv_as_leaf_delivery_goals(self):
        for name, compiled in retained_history_cases():
            with self.subTest(format=name):
                history = compiled.state_in.requests[0].blocks
                goals = compiled.scenario.goal.output_locations
                producers = {output for op in compiled.scenario.workload.operations for output in op.outputs}
                for block in history:
                    self.assertEqual(goals[block.id], "dram")
                    tensor = compiled.scenario.workload.tensor_by_id[block.id]
                    self.assertEqual(tensor.role, "input")
                    self.assertEqual(tensor.initial_locations, ("dram",))
                    self.assertNotIn(block.id, producers)
                old = history[0].id
                actions = []
                for task in compiled.tasks:
                    actions += [Action(ACTION_COMPUTE, task.operation_id, device="cpu"), Action(ACTION_ADVANCE)]
                    if task.kind == "attention":
                        actions += [Action(ACTION_EVICT, old, location="dram")]
                rejected = evaluate_mapping(compiled.scenario, tuple(actions))
                self.assertEqual(rejected.status, "invalid_mapping")
                self.assertIn("LIVE_VALUE_LOSS", rejected.reason)
                normal = run_invocation(compiled)
                self.assertEqual(normal.status, "valid")
                self.assertEqual(len(normal.state_out.requests[0].blocks), 4)

    def test_retention_goals_do_not_duplicate_kv_storage_or_change_capacity(self):
        for name, compiled in retained_history_cases():
            with self.subTest(format=name):
                result = run_invocation(compiled)
                expected = ({"synthetic": 256, "partial": 7880}.get(name) if name != "full" else
                    resident_accounting(compiled.layer.physical.config, "block", 1, 3)["resident_logical_bytes"]+300)
                self.assertEqual(result.evaluation.peak_dram_bytes, expected)
                old_names = {block.id for block in compiled.state_in.requests[0].blocks}
                self.assertEqual(sum(t.storage_bytes for t in compiled.scenario.workload.tensors if t.id in old_names),
                                 3*compiled.layer.kv_bytes_per_token)
                for capacity, status in ((expected, "valid"), (expected-1, "invalid_mapping")):
                    invocation = dataclasses.replace(compiled.invocation,
                        hardware=dataclasses.replace(compiled.invocation.hardware, dram_capacity_bytes=capacity))
                    bounded = compile_invocation(compiled.layer, invocation, compiled.cost_table)
                    self.assertEqual(run_invocation(bounded).status, status)

    def test_workspace_limit_bounds_the_running_macro_scratch(self):
        raw = full_layer_document(); raw["workspace_limit_bytes"] = 300
        self.assertEqual(run_invocation(self.compile(raw)).status, "valid")
        raw["workspace_limit_bytes"] = 299
        with self.assertRaisesRegex(InvalidInput, "workspace_limit_bytes"):
            self.compile(raw)


if __name__ == "__main__":
    unittest.main()
