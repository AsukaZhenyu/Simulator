"""Physical boundary and capacity regressions, independent of measured latency."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.physical import DTYPES, signature_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import parse_layer_document, parse_invocation_document, state_in_from_state_out
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "mapping" / "examples" / "m2"


def layer_document(*, small=True):
    raw = json.loads((EXAMPLES / "qwen35-geometry.cpu-core.layer.json").read_text(encoding="utf-8"))
    raw["provenance"] = "synthetic_layer"
    if small:
        raw.update(id="physical-small", hidden_size=16, hidden_bytes_per_token=64, kv_bytes_per_token=64)
        raw["physical"].update(intermediate=32, heads=4, kv_heads=2, head_dim=4,
                               attention_gate="none", ffn_weight_policy="uniform_declared_weight_dtype")
        shapes = [(16, 16), (16, 32), (16, 32), (32, 16)]
        for weight, shape in zip(raw["weights"], shapes):
            weight.update(dtype="f32", shape=list(shape), storage_bytes=shape[0]*shape[1]*4)
    return raw


def invocation_document(layer, *, phase="prefill", state=None, capacity=1 << 28):
    raw = json.loads((EXAMPLES / "fcd-prefill-3.json").read_text(encoding="utf-8"))
    raw.update(format_version="m2-layer-0.2", layer=layer.id, phase=phase)
    raw["hardware"]["dram_capacity_bytes"] = capacity
    if phase == "decode":
        raw["requests"][0].update(position_start=3, tokens=1)
    invocation = parse_invocation_document(raw)
    if state is not None:
        import dataclasses
        invocation = dataclasses.replace(invocation, state_in=state)
    return invocation


def table_for(layer, invocation, *, scratch=(100, 200)):
    request = invocation.requests[0]
    shape = MacroShape(invocation.phase, invocation.batch, request.tokens, request.position_start)
    entries = []
    for i, task in enumerate(layer.tasks):
        signature = signature_for(layer, task.kind, shape)
        entries.append(dict(kind=task.kind, device="cpu", phase=shape.phase, batch=shape.batch,
            tokens=shape.tokens, visible_history=shape.visible_history, duration_ns=(i+1)*1000,
            provenance="synthetic_layer", workspace_bytes=scratch[i], accounting="per_request",
            physical_signature=signature, contract_hash=canonical_contract_hash(signature)))
    return parse_cost_table(dict(source="physical-boundary-test", entries=entries))


class PhysicalBoundary(unittest.TestCase):
    def compile(self, *, document=None, capacity=1 << 28):
        layer = parse_layer_document(document or layer_document())
        invocation = invocation_document(layer, capacity=capacity)
        return compile_invocation(layer, invocation, table_for(layer, invocation))

    def test_materialized_ports_and_grouped_weight_dependencies(self):
        compiled = self.compile()
        tensors = compiled.scenario.workload.tensor_by_id
        self.assertEqual(tensors["t:physical-small:r0:in.q"].ne, (4, 3, 4, 1))
        self.assertEqual(tensors["t:physical-small:r0:in.k"].ne, (4, 3, 2, 1))
        attention, ffn = compiled.scenario.workload.operations
        self.assertEqual(set(attention.inputs), {"w:physical-small:W_o",
            "t:physical-small:r0:in.q", "t:physical-small:r0:in.k", "t:physical-small:r0:in.v"})
        self.assertEqual(set(ffn.inputs), {"w:physical-small:W_gate", "w:physical-small:W_up",
                                         "w:physical-small:W_down", attention.output})
        self.assertEqual(len(attention.extra_outputs), 3)
        self.assertEqual(len(compiled.tasks), 2)

    def test_quantized_weight_storage_is_preserved_at_real_geometry(self):
        compiled = self.compile(document=layer_document(small=False))
        tensors = compiled.scenario.workload.tensor_by_id
        self.assertEqual(tensors["w:qwen35-geometry-cpu-core:W_o"].dtype, "GGML_TYPE_Q4_K")
        self.assertEqual(tensors["w:qwen35-geometry-cpu-core:W_down"].dtype, "GGML_TYPE_Q6_K")
        self.assertEqual(sum(t.storage_bytes for t in tensors.values() if t.role == "weight"), 51793920)
        self.assertEqual(tensors["t:qwen35-geometry-cpu-core:r0:in.gate"].ne, (4096, 3, 1, 1))

    def test_prefill_decode_preserves_history_identity_and_reads_every_block(self):
        compiled = self.compile()
        first = run_invocation(compiled)
        self.assertEqual(first.status, "valid")
        self.assertEqual(first.makespan_ns, 3000)
        state = state_in_from_state_out(first.state_out)
        invocation = invocation_document(compiled.layer, phase="decode", state=state)
        second = compile_invocation(compiled.layer, invocation, table_for(compiled.layer, invocation))
        old = {block.id for block in state.requests[0].blocks}
        self.assertTrue(old.issubset(second.scenario.workload.operations[0].inputs))
        result = run_invocation(second)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.state_out.requests[0].next_position, 4)
        self.assertEqual(len(result.state_out.requests[0].blocks), 4)

    def test_new_kv_inputs_and_cache_slots_are_counted_separately(self):
        compiled = self.compile()
        tensors = compiled.scenario.workload.tensor_by_id
        input_bytes = sum(tensors[f"t:physical-small:r0:in.{p}"].storage_bytes for p in ("k", "v"))
        cache_bytes = sum(tensors[name].storage_bytes for name in compiled.block_ids)
        self.assertEqual(input_bytes, 192)
        self.assertEqual(cache_bytes, 192)

    def test_capacity_has_independent_resident_and_running_scratch(self):
        # No evictions in plan_for: 7168 weights + 384 QKV + 192 new cache
        # + two 192-byte outputs + current FFN scratch 200 = 8328.
        self.assertEqual(run_invocation(self.compile(capacity=8328)).status, "valid")
        self.assertNotEqual(run_invocation(self.compile(capacity=8327)).status, "valid")

    def test_workspace_limit_bounds_scratch_not_resident_memory(self):
        raw = layer_document()
        raw["workspace_limit_bytes"] = 200
        self.assertEqual(run_invocation(self.compile(document=raw)).status, "valid")
        raw["workspace_limit_bytes"] = 199
        with self.assertRaisesRegex(InvalidInput, "workspace_limit_bytes"):
            self.compile(document=raw)

    def test_dtype_geometry_and_variant_cannot_reuse_the_same_cost(self):
        layer = parse_layer_document(layer_document())
        invocation = invocation_document(layer)
        table = table_for(layer, invocation)
        import dataclasses
        for key, value in (("threads", 4), ("attention_gate", "sigmoid")):
            changed = dataclasses.replace(layer, physical=dataclasses.replace(layer.physical, **{key: value}))
            with self.assertRaises(Unsupported):
                compile_invocation(changed, invocation, table)
        raw = layer_document()
        raw["weights"][0].update(dtype="f16", storage_bytes=512)
        with self.assertRaises(Unsupported):
            compile_invocation(parse_layer_document(raw), invocation, table)
        raw = layer_document()
        raw["physical"]["intermediate"] = 64
        for weight in raw["weights"][1:]:
            weight["shape"] = [16, 64] if weight["id"] != "W_down" else [64, 16]
            weight["storage_bytes"] = 4096
        changed = parse_layer_document(raw)
        with self.assertRaises(Unsupported):
            compile_invocation(changed, invocation, table)

    def test_missing_bound_signature_is_not_an_implicit_synthetic_fallback(self):
        layer = parse_layer_document(layer_document())
        invocation = invocation_document(layer)
        table = table_for(layer, invocation)
        import dataclasses
        table = dataclasses.replace(table, entries={key: dataclasses.replace(value, physical_signature=None, contract_hash=None)
                                                    for key, value in table.entries.items()})
        with self.assertRaises(Unsupported):
            compile_invocation(layer, invocation, table)

    def test_physical_batch_two_is_explicitly_unsupported(self):
        layer = parse_layer_document(layer_document())
        raw = json.loads((EXAMPLES / "fcd-prefill-3.json").read_text(encoding="utf-8"))
        raw.update(layer=layer.id, format_version="m2-layer-0.2")
        second = copy.deepcopy(raw["requests"][0]); second["request_id"] = "r1"
        state = copy.deepcopy(raw["state_in"]["requests"][0]); state["request_id"] = "r1"
        raw["requests"].append(second); raw["state_in"]["requests"].append(state)
        invocation = parse_invocation_document(raw)
        with self.assertRaisesRegex(Unsupported, "batch=1"):
            compile_invocation(layer, invocation, table_for(layer, invocation))

    def test_real_matrix_bytes_are_checked_instead_of_trusted(self):
        raw = layer_document(small=False)
        raw["weights"][0]["storage_bytes"] += 4
        with self.assertRaisesRegex(InvalidInput, "actual layout"):
            parse_layer_document(raw)

    def test_unsupported_alias_padded_cache_and_full_layer_claim_are_refused(self):
        for key, value in (("port_layout", "interleaved_q_gate_view"), ("kv_allocation", "padded_preallocation"),
                           ("scope", "full_decoder_block"), ("kv_dtype", "f16")):
            raw = layer_document(); raw["physical"][key] = value
            with self.assertRaises(Unsupported):
                parse_layer_document(raw)

    def test_exclusions_cannot_be_removed_to_claim_full_layer_coverage(self):
        raw = layer_document(); raw["physical"]["excluded_operations"] = []
        with self.assertRaisesRegex(InvalidInput, "excluded_operations"):
            parse_layer_document(raw)

    def test_physical_fields_need_new_explicit_format(self):
        raw = layer_document(); raw["format_version"] = "m2-layer-0.1"
        with self.assertRaisesRegex(InvalidInput, "require m2-layer-0.2"):
            parse_layer_document(raw)

    def test_legacy_hidden_invocation_cannot_be_reinterpreted_as_external_qkv(self):
        import dataclasses
        layer = parse_layer_document(layer_document())
        invocation = invocation_document(layer)
        with self.assertRaisesRegex(InvalidInput, "format_version mismatch"):
            compile_invocation(layer, dataclasses.replace(invocation, format_version="m2-layer-0.1"),
                               table_for(layer, invocation))

    def test_dataclass_construction_cannot_bypass_physical_boundary_validation(self):
        import dataclasses
        layer = parse_layer_document(layer_document())
        invocation = invocation_document(layer)
        table = table_for(layer, invocation)
        for key, value in (("scope", "full_decoder_block"), ("kv_allocation", "padded_preallocation"),
                           ("port_layout", "interleaved_q_gate_view"), ("threads", True),
                           ("excluded_operations", ()), ("heads", 3)):
            invalid = dataclasses.replace(layer, physical=dataclasses.replace(layer.physical, **{key: value}))
            with self.assertRaises((InvalidInput, Unsupported)):
                compile_invocation(invalid, invocation, table)

    def test_validation_only_costs_need_explicit_execution_opt_in(self):
        import dataclasses
        layer = parse_layer_document(layer_document())
        invocation = invocation_document(layer)
        table = dataclasses.replace(table_for(layer, invocation), validation_only=True)
        with self.assertRaisesRegex(Unsupported, "validation_only=true"):
            compile_invocation(layer, invocation, table)
        explicit = dataclasses.replace(invocation, allow_validation_only_costs=True)
        self.assertEqual(run_invocation(compile_invocation(layer, explicit, table)).status, "valid")

    def test_legacy_scratch_uses_serial_maximum_instead_of_sum(self):
        raw = json.loads((EXAMPLES / "fcd.layer.json").read_text(encoding="utf-8"))
        raw["workspace_bytes"] = 300
        layer = parse_layer_document(raw)
        invocation = parse_invocation_document(json.loads((EXAMPLES / "fcd-prefill-3.json").read_text(encoding="utf-8")))
        costs = json.loads((EXAMPLES / "fcd.costs.json").read_text(encoding="utf-8"))
        for entry in costs["entries"]:
            entry["workspace_bytes"] = {"projection": 100, "attention": 200, "dense_ffn": 300}[entry["kind"]]
        table = parse_cost_table(costs)
        self.assertEqual(run_invocation(compile_invocation(layer, invocation, table)).status, "valid")
        raw["workspace_bytes"] = 600
        with self.assertRaisesRegex(InvalidInput, "任务最大值"):
            compile_invocation(parse_layer_document(raw), invocation, table)


if __name__ == "__main__":
    unittest.main()
