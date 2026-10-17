"""Attention stage storage/causality/composition and tiny semantic oracles.

Numerical tests check the declared mathematical decomposition, not timing-model
accuracy or a numerical kernel in the event runtime (which tracks identities).
"""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import unittest

from llm_infer_model.m2.attention_stages import (
    AttentionStagesSpec, KINDS, DENSE_FRONTEND, MOE_FRONTEND,
    parse_attention_stages, attention_stages_document, transform_attention_stages,
)
from llm_infer_model.m2.attention_execution import parse_attention_execution, transform_dense_attention
from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.costs import TransferEntry
from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.m2.ffn_execution import (KINDS as FFN_KINDS, parse_ffn_execution, transform_ffn_execution)
from llm_infer_model.m2.row_execution import parse_execution_spec, transform_moe_execution, NEW_KINDS
from llm_infer_model.tensor.engine import evaluate_mapping, STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

EXAMPLES = Path(__file__).resolve().parents[2] / "mapping/examples/m2"


def dense_source(phase="decode"):
    return load_and_compile(EXAMPLES / ("hybrid-interface-synthetic-decode-1.json" if phase == "decode"
                                        else "hybrid-interface-synthetic-prefill-3.json"))


def moe_source():
    source = load_and_compile(EXAMPLES / "moe-interface-synthetic-prefill-2.json")
    # A new, explicitly sized synthetic workload. The original fixture's 32B
    # opaque projection is intentionally insufficient for these physical ports.
    weights = tuple(replace(w, storage_bytes=192) if w.id == "W_proj" else w for w in source.layer.weights)
    layer = replace(source.layer, weights=weights)
    state = replace(source.state_in, weight_copies=tuple(
        replace(w, storage_bytes=192) if w.id == "W_proj" else w for w in source.state_in.weight_copies))
    invocation = replace(source.invocation, state_in=state)
    transfers = dict(source.cost_table.transfers)
    for route in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd"):
        transfers[(route, 192)] = TransferEntry(13, "synthetic_transfer")
    return compile_invocation(layer, invocation, replace(source.cost_table, transfers=transfers))


def document(source):
    if source.layer.ffn == "dense":
        config = source.layer.physical.config
        geometry = {name: config[name] for name in ("hidden", "heads", "kv_heads", "head_dim", "attention_gate")}
        frontend = dict(contract=DENSE_FRONTEND)
    else:
        geometry = dict(hidden=4, heads=2, kv_heads=1, head_dim=2, attention_gate="none")
        frontend = dict(contract=MOE_FRONTEND, weight_id="W_proj", rope_dims=2, rope_base=10000.0)
    shapes = {task.shape for task in source.tasks if task.kind in ("attention", "attention_core")}
    entries = [dict(kind=kind, device=device, phase=shape.phase, batch=shape.batch,
                    tokens=shape.tokens, visible_history=shape.visible_history,
                    duration_ns=100 + 10 * index + (1 if device == "gpu" else 0), workspace_bytes=4)
               for shape in sorted(shapes) for index, kind in enumerate(sorted(KINDS))
               for device in source.invocation.devices]
    # Exact declarations for this tiny suite; these are synthetic durations.
    sizes = {4, 8, 16, 32, 48, 64, 96, 128, 144, 192, 256, 512, 1024}
    transfers = [dict(route=route, storage_bytes=size, duration_ns=size + 1)
                 for route in source.cost_table.allowed_transfer_routes for size in sorted(sizes)
                 if (route, size) not in source.cost_table.transfers]
    return dict(format_version="attention-stages-0.1", variant="qk_softmax_pv", provenance="synthetic_execution",
                geometry=geometry, frontend=frontend, entries=entries, transfer_entries=transfers)


def refined(source):
    return transform_attention_stages(source, parse_attention_stages(document(source)))


def row_spec():
    return parse_execution_spec(dict(format_version="moe-execution-0.1", variant="selected_rows",
        expert_variant="swiglu_staged", provenance="synthetic_execution",
        index_storage_contract="stage_workspace_includes_row_index_temporaries",
        intermediate_bytes_per_token=32, transfer_entries=[],
        entries=[dict(kind=kind, device=device, phase="prefill", batch=1, token_count=count,
                      hidden_bytes_per_token=16, intermediate_bytes_per_token=32,
                      duration_ns=50, workspace_bytes=4)
                 for kind in sorted(NEW_KINDS) for device in ("cpu", "gpu") for count in (1, 2)]))


def dense_ffn_spec(source):
    shape = next(task.shape for task in source.tasks if task.kind == "dense_ffn")
    return parse_ffn_execution(dict(format_version="ffn-execution-0.1", target_kinds=["dense_ffn"],
        intermediate_size=32, merge_input_count=2, shards=[dict(id="host", start=0, end=8, device="cpu"),
                                     dict(id="accelerator", start=8, end=32, device="gpu")],
        preprocess_device="gpu", merge_device="gpu", provenance="synthetic_execution",
        weight_storage_contract="materialized_partition_replaces_parent",
        entries=[dict(kind=kind, device=device, phase=shape.phase, batch=shape.batch, tokens=shape.tokens,
                      visible_history=shape.visible_history, hidden_size=source.layer.hidden_size,
                      intermediate_size=width, duration_ns=50, workspace_bytes=4)
                 for kind in FFN_KINDS for device in source.invocation.devices for width in (8, 24, 32)],
        transfer_entries=[dict(route=route, storage_bytes=size, duration_ns=size + 1)
                          for route in source.cost_table.allowed_transfer_routes for size in (32, 96, 512, 1536)
                          if (route, size) not in source.cost_table.transfers]))


def run(source):
    placements = {task.operation_id: ("gpu" if task.kind in ("attention_qk", "attention_output") else "cpu")
                  for task in source.tasks}
    return evaluate_mapping(source.scenario, plan_moe(source, placements=placements))


class AttentionStagesTests(unittest.TestCase):
    def test_strict_roundtrip_and_no_implicit_workspace(self):
        raw = document(dense_source())
        self.assertEqual(attention_stages_document(parse_attention_stages(raw)), raw)
        for key in ("duration_ns", "workspace_bytes"):
            bad = deepcopy(raw); bad["entries"][0][key] = None
            with self.assertRaises(InvalidInput): parse_attention_stages(bad)
        bad = deepcopy(raw); bad["provenance"] = "measured_macro"
        with self.assertRaises(Unsupported): parse_attention_stages(bad)
        bad = deepcopy(raw); bad["geometry"]["heads"] = 3
        with self.assertRaises(InvalidInput): parse_attention_stages(bad)

    def test_dense_stages_use_real_qkv_and_only_pv_appends_kv(self):
        original = dense_source(); snapshot = deepcopy(original.workload_document)
        result = refined(original)
        self.assertEqual(original.workload_document, snapshot)
        self.assertEqual(result.invocation.id, original.invocation.id)
        self.assertEqual(result.hidden_out_ids, original.hidden_out_ids)
        region = result.physical_boundary["attention_stages"]["regions"][0]
        ops, tensors = result.scenario.workload.operation_by_id, result.scenario.workload.tensor_by_id
        qk, softmax, pv, output = [ops[region[key + "_operation_id"]] for key in ("qk", "softmax", "pv", "output")]
        self.assertEqual(qk.inputs, (region["qkv_ports"]["q"], region["qkv_ports"]["k"], *region["historical_kv_ids"]))
        self.assertEqual(softmax.inputs, (qk.output,))
        self.assertIn(region["qkv_ports"]["v"], pv.inputs)
        self.assertEqual(pv.extra_outputs, tuple(region["new_kv_ids"]))
        self.assertEqual(qk.extra_outputs, ())
        self.assertEqual(output.extra_outputs, ())
        self.assertIn("w:hybrid-interface-synthetic:W_output", output.inputs)
        self.assertNotIn("w:hybrid-interface-synthetic:W_output", pv.inputs)
        self.assertIn("t:hybrid-interface-synthetic:r0:in.hidden", output.inputs)
        self.assertEqual(tensors[qk.output].storage_bytes, 4 * 1 * 4 * 4)
        self.assertEqual(result.block_ids, original.block_ids)
        self.assertEqual(run(result).status, STATUS_VALID)

    def test_prefill_score_matrix_counts_future_masked_entries_physically(self):
        result = refined(dense_source("prefill"))
        region = result.physical_boundary["attention_stages"]["regions"][0]
        tensor = result.scenario.workload.tensor_by_id[region["scores_tensor"]]
        self.assertEqual(tensor.ne, (3, 3, 4, 1))
        self.assertEqual(tensor.storage_bytes, 144)
        self.assertEqual(region["absolute_positions"], [0, 1, 2])
        self.assertEqual(run(result).status, STATUS_VALID)

    def test_moe_frontend_requires_sufficient_packed_weights(self):
        original = load_and_compile(EXAMPLES / "moe-interface-synthetic-prefill-2.json")
        with self.assertRaisesRegex(InvalidInput, "requires exactly 192 bytes"):
            refined(original)
        result = refined(moe_source())
        region = result.physical_boundary["attention_stages"]["regions"][0]
        projection = next(task for task in result.tasks if task.kind == "attention_projection")
        op = result.scenario.workload.operation_by_id[projection.operation_id]
        self.assertEqual(set(op.outputs), set(region["qkv_ports"].values()))
        self.assertIn("w:moe-small:W_proj", op.inputs)
        self.assertTrue(any(result.scenario.workload.tensor_by_id[tid].dtype == "GGML_TYPE_I32" for tid in op.inputs))
        self.assertNotIn("t:moe-small:r0:proj", result.scenario.workload.tensor_by_id)
        self.assertEqual(run(result).status, STATUS_VALID)

    def test_missing_exact_stage_or_transfer_is_rejected(self):
        source = dense_source("prefill"); raw = document(source)
        bad = deepcopy(raw); bad["entries"] = [p for p in bad["entries"] if p["kind"] != "attention_softmax"]
        with self.assertRaisesRegex(Unsupported, "missing exact independent"):
            transform_attention_stages(source, parse_attention_stages(bad))
        bad = deepcopy(raw); bad["transfer_entries"] = []
        with self.assertRaises(Unsupported): transform_attention_stages(source, parse_attention_stages(bad))

    def test_transfer_conflict_is_not_a_silent_global_price_replacement(self):
        source = dense_source(); raw = document(source)
        (route, size), entry = next(iter(source.cost_table.transfers.items()))
        raw["transfer_entries"].append(dict(route=route, storage_bytes=size, duration_ns=entry.duration_ns + 1))
        with self.assertRaisesRegex(InvalidInput, "conflicts with existing exact price"):
            transform_attention_stages(source, parse_attention_stages(raw))

    def test_changing_stage_price_changes_only_that_stage(self):
        source = dense_source(); raw = document(source)
        first = transform_attention_stages(source, parse_attention_stages(raw))
        for price in raw["entries"]:
            if price["kind"] == "attention_softmax" and price["device"] == "cpu": price["duration_ns"] += 12345
        second = transform_attention_stages(source, parse_attention_stages(raw))
        self.assertEqual(run(second).makespan_ns - run(first).makespan_ns, 12345)
        for old in source.tasks:
            if old.kind != "attention":
                self.assertEqual(first.scenario_document["costs"]["compute"][old.operation_id],
                                 source.scenario_document["costs"]["compute"][old.operation_id])

    def test_row_and_attention_transforms_commute_on_independent_regions(self):
        source = moe_source(); spec = parse_attention_stages(document(source))
        a = transform_attention_stages(transform_moe_execution(source, row_spec()), spec)
        b = transform_moe_execution(transform_attention_stages(source, spec), row_spec())
        for name in ("tensors", "operations"):
            self.assertEqual(sorted(a.workload_document[name], key=lambda item: item["id"]),
                             sorted(b.workload_document[name], key=lambda item: item["id"]))
        for name in ("compute", "transfer"):
            self.assertEqual(a.scenario_document["costs"][name], b.scenario_document["costs"][name])
        self.assertEqual(run(a).status, STATUS_VALID)
        self.assertEqual(run(a).makespan_ns, run(b).makespan_ns)

    def test_moe_stages_visible_before_router_but_experts_are_hidden(self):
        compiled = refined(moe_source())
        obs = CausalM2Session(compiled).observe()
        self.assertTrue({"attention_projection", "attention_qk", "attention_softmax", "attention_pv", "attention_output"}
                        .issubset({op.kind for op in obs.operations}))
        self.assertFalse(any(op.kind == "expert_ffn" for op in obs.operations))
        self.assertEqual(obs.route_history, ())

    def test_dense_ffn_and_attention_preserve_each_others_semantics_in_both_orders(self):
        source = dense_source()
        attention_spec = parse_attention_stages(document(source))
        ffn_spec = dense_ffn_spec(source)
        first_ffn = transform_ffn_execution(source, ffn_spec)
        first_attention = transform_attention_stages(source, attention_spec)
        a = transform_attention_stages(first_ffn, attention_spec)
        b = transform_ffn_execution(first_attention, ffn_spec)
        self.assertEqual(a.physical_boundary["ffn_execution"], first_ffn.physical_boundary["ffn_execution"])
        self.assertEqual(b.physical_boundary["attention_stages"], first_attention.physical_boundary["attention_stages"])
        for name in ("tensors", "operations"):
            self.assertEqual(sorted(a.workload_document[name], key=lambda item: item["id"]),
                             sorted(b.workload_document[name], key=lambda item: item["id"]))
        self.assertEqual(a.block_ids, source.block_ids)
        self.assertEqual(a.hidden_out_ids, source.hidden_out_ids)
        evaluation = evaluate_mapping(a.scenario, plan_moe(a))
        self.assertEqual(evaluation.status, STATUS_VALID)

    def test_existing_pure_attention_keeps_output_once_and_its_exact_cost(self):
        source = dense_source()
        raw = dict(format_version="dense-attention-execution-0.1", variant="pure_attention",
            provenance="synthetic_execution", invocation_files=["not-opened.json"],
            frontend_config=deepcopy(source.layer.physical.config), batch_contract="independent_requests_not_fused_batch_kernel",
            subbatches=[dict(id="cpu", request_ids=["r0"], attention_device="cpu")], transfer_entries=[],
            entries=[dict(kind=kind, device=device, phase="decode", batch=1, tokens=1, visible_history=3,
                          duration_ns=771, workspace_bytes=4)
                     for kind in ("attention_core", "attention_output") for device in ("cpu", "gpu")])
        core = transform_dense_attention(source, parse_attention_execution(raw))
        result = refined(core)
        self.assertEqual(sum(t.kind == "attention_output" for t in result.tasks), 1)
        output = next(t for t in result.tasks if t.kind == "attention_output")
        self.assertEqual(result.scenario.costs.compute_cost(output.operation_id, "cpu").duration_ns, 771)
        self.assertEqual(run(result).status, STATUS_VALID)

    def test_repeated_refinement_and_wrong_frontend_are_rejected(self):
        source = dense_source(); spec = parse_attention_stages(document(source))
        with self.assertRaisesRegex(InvalidInput, "already refined"):
            transform_attention_stages(transform_attention_stages(source, spec), spec)
        raw = document(source); raw["geometry"]["head_dim"] = 2
        with self.assertRaises(InvalidInput): transform_attention_stages(source, parse_attention_stages(raw))


class AttentionSemanticOracleTests(unittest.TestCase):
    def test_actual_lowered_ports_execute_masked_gqa_gate_projection_and_residual(self):
        compiled = refined(dense_source())
        region = compiled.physical_boundary["attention_stages"]["regions"][0]
        geometry = document(dense_source())["geometry"]
        heads, kvheads, dim = geometry["heads"], geometry["kv_heads"], geometry["head_dim"]
        tokens, history, hidden = 1, 3, geometry["hidden"]
        q = [[[math.sin(h * 7 + t * 3 + d + .23) for d in range(dim)] for t in range(tokens)] for h in range(heads)]
        k = [[[math.cos(h * 5 + t * 2 + d + .19) for d in range(dim)] for t in range(history + tokens)] for h in range(kvheads)]
        v = [[[math.sin(h * 11 + t * 4 + d + .37) for d in range(dim)] for t in range(history + tokens)] for h in range(kvheads)]
        gates = [[math.cos(i + .17) for i in range(heads * dim)] for _ in range(tokens)]
        residual = [[math.sin(i + .31) for i in range(hidden)] for _ in range(tokens)]
        matrix = [[math.sin(i * 3 + j + .41) for j in range(hidden)] for i in range(heads * dim)]
        ports = region["qkv_ports"]
        values = {ports["q"]: q, ports["k"]: [rows[history:] for rows in k],
                  ports["v"]: [rows[history:] for rows in v], ports["gate"]: gates,
                  "t:hybrid-interface-synthetic:r0:in.hidden": residual,
                  "w:hybrid-interface-synthetic:W_output": matrix}
        for block in region["historical_kv_blocks"]:
            start, count = block["first_position"], block["tokens"]
            values[block["id"]] = dict(k=[rows[start:start + count] for rows in k],
                                        v=[rows[start:start + count] for rows in v])
        block_ids = set(region["historical_kv_ids"])
        for task in compiled.tasks:
            if task.kind not in ("attention_qk", "attention_softmax", "attention_pv", "attention_output"):
                continue
            op = compiled.scenario.workload.operation_by_id[task.operation_id]
            if task.kind == "attention_qk":
                queries, newkeys = values[op.inputs[0]], values[op.inputs[1]]
                keys = [[row for tid in op.inputs if tid in block_ids for row in values[tid]["k"][h]] + newkeys[h]
                        for h in range(kvheads)]
                values[op.output] = [[[sum(a*b for a,b in zip(query, key)) / math.sqrt(dim)
                                       for key in keys[h // (heads // kvheads)]] for query in rows]
                                     for h, rows in enumerate(queries)]
            elif task.kind == "attention_softmax":
                logits = values[op.inputs[0]]; probabilities = []
                for rows in logits:
                    output_rows = []
                    for t, row in enumerate(rows):
                        visible = region["absolute_positions"][t] + 1
                        maximum = max(row[:visible])
                        numerators = [math.exp(x-maximum) if p < visible else 0 for p,x in enumerate(row)]
                        output_rows.append([x/sum(numerators) for x in numerators])
                    probabilities.append(output_rows)
                values[op.output] = probabilities
            elif task.kind == "attention_pv":
                probabilities, newkeys, newvalues = [values[tid] for tid in op.inputs[:3]]
                allvalues = [[row for tid in op.inputs if tid in block_ids for row in values[tid]["v"][h]] + newvalues[h]
                             for h in range(kvheads)]
                context = [[sum(probabilities[h][t][p] * allvalues[h // (heads // kvheads)][p][d]
                                for p in range(history + tokens)) for h in range(heads) for d in range(dim)]
                           for t in range(tokens)]
                values[op.output] = context
                for t, tid in enumerate(op.extra_outputs):
                    values[tid] = dict(k=[rows[t] for rows in newkeys], v=[rows[t] for rows in newvalues])
            else:
                context, gate, weights, skip = [values[tid] for tid in op.inputs]
                values[op.output] = [[sum(context[t][i] / (1 + math.exp(-gate[t][i])) * weights[i][j]
                                         for i in range(heads * dim)) + skip[t][j] for j in range(hidden)]
                                     for t in range(tokens)]
        # Independent monolithic formulation directly from original Q/K/V.
        reference_context = []
        for h in range(heads):
            kvh = h // (heads // kvheads)
            weights = [math.exp(sum(a*b for a,b in zip(q[h][0], key)) / math.sqrt(dim)) for key in k[kvh]]
            reference_context.extend(sum(w*value[d] for w,value in zip(weights, v[kvh])) / sum(weights)
                                     for d in range(dim))
        expected = [sum(x/(1+math.exp(-gates[0][i])) * matrix[i][j] for i,x in enumerate(reference_context))
                    + residual[0][j] for j in range(hidden)]
        result_op = compiled.scenario.workload.operation_by_id[region["output_operation_id"]]
        for actual, reference in zip(values[result_op.output][0], expected):
            self.assertAlmostEqual(actual, reference, places=12)
        for tid in region["new_kv_ids"]:
            self.assertEqual(values[tid]["k"], [rows[-1] for rows in k])
            self.assertEqual(values[tid]["v"], [rows[-1] for rows in v])

    def test_staged_causal_gqa_equals_independent_reference_with_history(self):
        # Query heads 0/1 share KV head 0; 2/3 share KV head 1.
        heads, kv_heads, dim, history, tokens = 4, 2, 2, 2, 3
        q = [[[math.sin(1 + h * 11 + t * 3 + d) for d in range(dim)] for t in range(tokens)] for h in range(heads)]
        k = [[[math.cos(2 + h * 13 + t * 5 + d) for d in range(dim)] for t in range(history + tokens)] for h in range(kv_heads)]
        v = [[[math.sin(3 + h * 7 + t * 2 + d) for d in range(dim)] for t in range(history + tokens)] for h in range(kv_heads)]
        scores = [[[sum(q[h][t][d] * k[h // 2][p][d] for d in range(dim)) / math.sqrt(dim)
                    for p in range(history + tokens)] for t in range(tokens)] for h in range(heads)]
        probabilities = []
        for h in range(heads):
            rows = []
            for t in range(tokens):
                valid = history + t + 1; maximum = max(scores[h][t][:valid])
                values = [math.exp(s - maximum) if p < valid else 0.0 for p, s in enumerate(scores[h][t])]
                total = sum(values); rows.append([x / total for x in values])
            probabilities.append(rows)
        staged = [[[sum(probabilities[h][t][p] * v[h // 2][p][d] for p in range(history + tokens))
                    for d in range(dim)] for t in range(tokens)] for h in range(heads)]
        # Deliberately computes only visible keys, without constructing the
        # materialized score matrix or reusing the staged probabilities.
        for h in range(heads):
            for t in range(tokens):
                weights = [math.exp(sum(a * b for a, b in zip(q[h][t], key)) / math.sqrt(dim))
                           for key in k[h // 2][:history + t + 1]]
                for d in range(dim):
                    reference = sum(w * value[d] for w, value in zip(weights, v[h // 2])) / sum(weights)
                    self.assertAlmostEqual(staged[h][t][d], reference, places=13)
                self.assertTrue(all(value == 0 for value in probabilities[h][t][history + t + 1:]))

    def test_qkv_pack_layout_and_rope_preserve_head_pair_norms(self):
        source = moe_source(); geometry = document(source)["geometry"]
        hidden, q_width, kv_width = geometry["hidden"], 4, 2
        widths = [q_width, kv_width, kv_width, hidden]
        matrix_sizes = [hidden * q_width, hidden * kv_width, hidden * kv_width, q_width * hidden]
        self.assertEqual(4 * sum(matrix_sizes), source.layer.weight_by_id("W_proj").storage_bytes)
        x = [0.17, -0.41, 0.93, -0.27]
        packed = [math.sin(i + 0.3) for i in range(sum(matrix_sizes))]
        offset, projected = 0, []
        for width in widths[:3]:
            matrix = packed[offset:offset + hidden * width]; offset += hidden * width
            projected.append([sum(x[i] * matrix[j * hidden + i] for i in range(hidden)) for j in range(width)])
        for position in (0, 3, 11):
            for values in projected[:2]:
                for index in range(0, len(values), 2):
                    a, b = values[index:index + 2]
                    rotated = (a * math.cos(position) - b * math.sin(position),
                               a * math.sin(position) + b * math.cos(position))
                    self.assertAlmostEqual(a * a + b * b, sum(z * z for z in rotated), places=13)


if __name__ == "__main__":
    unittest.main()
