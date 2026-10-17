"""Common Dense/MoE SwiGLU lowering: exact costs, values and allocations."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import struct
import unittest

from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.ffn_execution import (
    KINDS, FFNShard, ffn_execution_document, parse_ffn_execution, transform_ffn_execution,
)
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.m2.row_execution import parse_execution_spec, transform_moe_execution
from llm_infer_model.tensor.engine import evaluate_mapping, STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, build_scenario

ROOT = Path(__file__).resolve().parents[2]


def dense_source():
    return load_and_compile(ROOT / "mapping/examples/m2/hybrid-interface-synthetic-decode-1.json")


def moe_source():
    fixture = Path(__file__).parent / "fixtures/subweights"
    source = load_and_compile(fixture / "invocation.json")
    return transform_moe_execution(source, parse_execution_spec(json.loads((fixture / "execution.json").read_text(encoding="utf-8"))))


def document(source, *, single=False):
    dense = source.layer.ffn == "dense"
    middle = source.layer.physical.config["intermediate"] if dense else source.physical_boundary["row_execution"]["intermediate_bytes_per_token"] // 4
    split = middle // 4
    shards = [dict(id="all", start=0, end=middle, device="gpu")] if single else [
        dict(id="host", start=0, end=split, device="cpu"), dict(id="accelerator", start=split, end=middle, device="gpu")]
    result = dict(format_version="ffn-execution-0.1", target_kinds=["dense_ffn"] if dense else ["expert_ffn", "shared_expert_ffn"],
        intermediate_size=middle, shards=shards, merge_input_count=len(shards), preprocess_device="gpu", merge_device="gpu",
        provenance="synthetic_execution", weight_storage_contract="materialized_partition_replaces_parent",
        entries=[], transfer_entries=[])
    widths = {middle, *(s["end"]-s["start"] for s in shards)}
    shapes = {(task.shape.phase, task.shape.batch, task.shape.tokens, task.shape.visible_history)
              for task in source.tasks if task.kind in ("dense_ffn", "expert_down")}
    for kind in sorted(KINDS):
        for device in ("cpu", "gpu"):
            for phase, batch, tokens, history in shapes:
                for width in sorted(widths):
                    result["entries"].append(dict(kind=kind, device=device, phase=phase, batch=batch, tokens=tokens,
                        visible_history=history, hidden_size=source.layer.hidden_size, intermediate_size=width,
                        duration_ns=37 + len(result["entries"]), workspace_bytes=4))
    sizes = {4 * source.layer.hidden_size * width for width in widths}
    for _, _, tokens, _ in shapes:
        sizes.update({4 * tokens * source.layer.hidden_size, *(4 * tokens * width for width in widths)})
    for route in source.scenario_document["mapspace"]["allowed_transfers"]:
        for size in sorted(sizes):
            if (route, size) not in source.cost_table.transfers:
                result["transfer_entries"].append(dict(route=route, storage_bytes=size, duration_ns=size*10))
    return result


def replay(compiled):
    return evaluate_mapping(compiled.scenario, plan_moe(compiled))


def mm(rows, matrix, cast=lambda value: value):
    # Conventional [input][output] matrices, separate from GGML storage order.
    return [[cast(sum(row[k] * matrix[k][j] for k in range(len(matrix))))
             for j in range(len(matrix[0]))] for row in rows]


def numerical_region(compiled, region, rng, *, round_f32=False):
    """Interpret actual lowered dependencies and leaf layouts, not a kernel benchmark."""
    spec = compiled.physical_boundary["ffn_execution"]["execution_spec"]
    h, middle = compiled.layer.hidden_size, spec["intermediate_size"]
    cast=(lambda value: struct.unpack("<f",struct.pack("<f",value))[0]) if round_f32 else (lambda value: value)
    ids = region["operation_ids"]
    tasks = {task.operation_id: task for task in compiled.tasks}
    tokens = tasks[ids[0]].shape.tokens
    rows = [[cast(rng.uniform(-0.7, 0.9)) for _ in range(h)] for _ in range(tokens)]
    parents = {stage: [[cast(rng.uniform(-0.4, 0.6)) for _ in range(h if stage == "down" else middle)]
                       for _ in range(middle if stage == "down" else h)] for stage in ("gate", "up", "down")}
    gamma = [cast(rng.uniform(0.7, 1.3)) for _ in range(h)]
    values = {region["input_tensor"]: rows}
    epsilon = compiled.layer.physical.config["norm_epsilon"] if compiled.layer.ffn == "dense" else None
    normalized = [[cast(v * gamma[k] / math.sqrt(sum(x*x for x in row)/h + epsilon))
                   for k, v in enumerate(row)] for row in rows] if epsilon else rows
    gates, ups = mm(normalized, parents["gate"],cast), mm(normalized, parents["up"],cast)
    activated = [[cast(g/(1+math.exp(-g))*u) for g,u in zip(gate,up)] for gate,up in zip(gates,ups)]
    reference = mm(activated, parents["down"],cast)
    if epsilon:
        reference = [[cast(y+x) for y,x in zip(out,row)] for out,row in zip(reference,rows)]
    partitions = {part["tensor_id"]: part for part in compiled.physical_boundary["ffn_execution"]["weight_partitions"]}
    for oid in ids:
        op = compiled.scenario.workload.operation_by_id[oid]
        if op.semantic_op in ("ffn_gate", "ffn_up", "ffn_down"):
            part = partitions[op.inputs[0]]; lo, hi = part["start"], part["end"]
            matrix = parents[part["stage"]]
            sliced = matrix[lo:hi] if part["stage"] == "down" else [row[lo:hi] for row in matrix]
            values[op.output] = mm(values[op.inputs[1]], sliced,cast)
        elif op.semantic_op == "ffn_norm":
            values[op.output] = normalized
        elif op.semantic_op == "ffn_swiglu":
            values[op.output] = [[cast(g/(1+math.exp(-g))*u) for g,u in zip(gate,up)]
                                 for gate,up in zip(values[op.inputs[0]],values[op.inputs[1]])]
        elif op.semantic_op in ("ffn_sum", "ffn_residual"):
            values[op.output] = [[cast(sum(values[tid][n][j] for tid in op.inputs)) for j in range(h)] for n in range(tokens)]
        else:
            raise AssertionError(op.semantic_op)
    return reference, values[region["output_tensor"]]


class FFNExecutionTests(unittest.TestCase):
    def test_strict_roundtrip_single_shard_and_partition_coverage(self):
        for source in (dense_source(), moe_source()):
            for single in (False, True):
                raw = document(source, single=single)
                self.assertEqual(ffn_execution_document(parse_ffn_execution(raw)), raw)
        for mutate in (lambda d: d["shards"][1].update(start=0),
                       lambda d: d["shards"][1].update(start=99),
                       lambda d: d["shards"][1].update(end=d["intermediate_size"]-1)):
            raw=document(dense_source()); mutate(raw)
            with self.assertRaises(InvalidInput): parse_ffn_execution(raw)

    def test_changed_shard_count_cannot_silently_reuse_merge_price_or_workspace(self):
        source=dense_source(); spec=parse_ffn_execution(document(source))
        alternatives=((FFNShard("all",0,32,"gpu"),),
                      (FFNShard("first",0,8,"cpu"),FFNShard("second",8,16,"gpu"),FFNShard("third",16,32,"gpu")))
        for shards in alternatives:
            with self.subTest(count=len(shards)), self.assertRaisesRegex(InvalidInput,"merge_input_count must equal len"):
                transform_ffn_execution(source,replace(spec,shards=shards))
        for value in (None,True,0,-1):
            raw=document(source);raw["merge_input_count"]=value
            with self.subTest(value=value),self.assertRaises(InvalidInput): parse_ffn_execution(raw)

    def test_explicit_single_shard_merge_contract_is_recorded_in_graph_and_boundary(self):
        source=dense_source(); raw=document(source,single=True)
        self.assertEqual(raw["merge_input_count"],1)
        compiled=transform_ffn_execution(source,parse_ffn_execution(raw))
        boundary=compiled.physical_boundary["ffn_execution"]
        self.assertEqual(boundary["merge_input_count"],1)
        self.assertEqual(boundary["merge_cost_contract"]["input_count"],1)
        merge=next(task for task in compiled.tasks if task.kind=="ffn_sum")
        operation=compiled.scenario.workload.operation_by_id[merge.operation_id]
        self.assertEqual(len(operation.inputs),1)
        self.assertEqual(operation.op_params,(1,))
        self.assertEqual(boundary["operation_stages"][merge.operation_id]["input_count"],1)
        self.assertEqual(replay(compiled).status,STATUS_VALID)

    def test_dense_graph_stages_paired_weights_partial_sum_residual_once(self):
        base = dense_source(); snapshot=deepcopy(base.workload_document)
        compiled = transform_ffn_execution(base, parse_ffn_execution(document(base)))
        kinds = [task.kind for task in compiled.tasks]
        self.assertEqual(kinds.count("ffn_norm"), 1)
        self.assertEqual(kinds.count("ffn_residual"), 1)
        self.assertEqual(kinds.count("ffn_sum"), 1)
        self.assertEqual(kinds.count("ffn_down"), 2)
        self.assertEqual(base.workload_document, snapshot)
        self.assertEqual(compiled.hidden_out_ids, base.hidden_out_ids)
        self.assertEqual(compiled.block_ids, base.block_ids)
        boundary = compiled.physical_boundary["ffn_execution"]
        for part in boundary["weight_partitions"]:
            tensor = compiled.scenario.workload.tensor_by_id[part["tensor_id"]]
            self.assertEqual(tensor.storage_bytes, 4*part["input_size"]*part["output_size"])
            self.assertNotIn(part["source_tensor_id"], compiled.scenario.workload.tensor_by_id)
        for task in compiled.tasks:
            if task.kind.startswith("ffn_"):
                self.assertEqual(compiled.scenario.mapspace.devices_for(task.operation_id), (task.device,))
        self.assertEqual(sum(t.storage_bytes for t in compiled.scenario.workload.tensors if t.role=="weight"),
                         sum(t.storage_bytes for t in base.scenario.workload.tensors if t.role=="weight"))
        self.assertEqual(replay(compiled).status, STATUS_VALID)

    def test_moe_uses_same_transform_and_preserves_gather_scatter_routing_and_inactive_weights(self):
        base=moe_source(); compiled=transform_ffn_execution(base,parse_ffn_execution(document(base)))
        self.assertFalse(any(task.kind in ("expert_gate","expert_up","expert_swiglu","expert_down") for task in compiled.tasks))
        for kind in ("gather_rows", "scatter_rows", "combine_rows", "router", "attention"):
            self.assertEqual([task for task in compiled.tasks if task.kind==kind], [task for task in base.tasks if task.kind==kind])
        self.assertFalse(any(task.kind in ("ffn_norm", "ffn_residual") for task in compiled.tasks))
        inactive = compiled.physical_boundary["ffn_execution"]["weight_children"]["w:moe-small:E3"]
        self.assertEqual(len(inactive),6)
        self.assertTrue(all(compiled.scenario.workload.tensor_by_id[tid].initial_locations==("ssd",) for tid in inactive))
        self.assertEqual(compiled.block_ids,base.block_ids)
        for region in compiled.physical_boundary["row_execution"]["regions"]:
            self.assertTrue(all(oid in compiled.compute_ids for oid in region["expert_operation_ids"]))
        self.assertEqual(replay(compiled).status,STATUS_VALID)

    def test_numerical_equivalence_dense_and_compact_moe_multiple_rows_nonuniform_shards(self):
        rng=random.Random(20261017)
        for source in (dense_source(),moe_source()):
            for single in (False,True):
                compiled=transform_ffn_execution(source,parse_ffn_execution(document(source,single=single)))
                for region in compiled.physical_boundary["ffn_execution"]["regions"]:
                    expected,actual=numerical_region(compiled,region,rng)
                    for e,a in zip(expected,actual):
                        for x,y in zip(e,a): self.assertAlmostEqual(x,y,places=11)

    def test_f32_materialized_rounding_has_bounded_partition_reduction_error(self):
        rng=random.Random(361712)
        maximum=0.0
        for source in (dense_source(),moe_source()):
            compiled=transform_ffn_execution(source,parse_ffn_execution(document(source)))
            for region in compiled.physical_boundary["ffn_execution"]["regions"]:
                for _ in range(5):
                    expected,actual=numerical_region(compiled,region,rng,round_f32=True)
                    maximum=max(maximum,max(abs(x-y) for e,a in zip(expected,actual) for x,y in zip(e,a)))
        self.assertGreater(maximum,0.0)  # Catch accidentally exact toy data.
        self.assertLess(maximum,2e-6)

    def test_costs_are_independent_and_unknown_workspace_missing_shape_rejected(self):
        base=dense_source(); raw=document(base)
        before=transform_ffn_execution(base,parse_ffn_execution(raw))
        for entry in raw["entries"]:
            if entry["kind"]=="ffn_down": entry["duration_ns"]+=1234
        after=transform_ffn_execution(base,parse_ffn_execution(raw))
        self.assertEqual(replay(after).makespan_ns-replay(before).makespan_ns,2468)
        raw=document(base); raw["entries"]=[]
        with self.assertRaisesRegex(Unsupported,"missing exact FFN stage cost"):
            transform_ffn_execution(base,parse_ffn_execution(raw))
        for value in (None, True, -1):
            raw=document(base); raw["entries"][0]["workspace_bytes"]=value
            with self.assertRaises(InvalidInput): parse_ffn_execution(raw)
        typed=parse_ffn_execution(document(base))
        typed=replace(typed,entries=(replace(typed.entries[0],workspace_bytes=None),*typed.entries[1:]))
        with self.assertRaises(InvalidInput): transform_ffn_execution(base,typed)

    def test_capacity_charges_every_shard_partial_and_workspace_at_exact_boundary(self):
        base=dense_source(); compiled=transform_ffn_execution(base,parse_ffn_execution(document(base)))
        result=replay(compiled)
        self.assertEqual(result.status,STATUS_VALID)
        self.assertGreater(result.h2d_bytes,0); self.assertGreater(result.d2h_bytes,0)
        for delta,valid in ((0,True),(-1,False)):
            scenario=deepcopy(compiled.scenario_document)
            scenario["architecture"]["vram_capacity_bytes"]=result.peak_vram_bytes+delta
            changed=replace(compiled,scenario=build_scenario(compiled.workload_document,scenario))
            run=replay(changed)
            self.assertEqual(run.status==STATUS_VALID,valid)
            if not valid: self.assertIn("CAPACITY",run.reason.upper())

    def test_attention_kv_graph_prices_and_metadata_preserved_and_double_transform_rejected(self):
        base=dense_source(); compiled=transform_ffn_execution(base,parse_ffn_execution(document(base)))
        for task in base.tasks:
            if task.kind=="dense_ffn": continue
            self.assertIn(task,compiled.tasks)
            self.assertEqual(base.scenario.workload.operation_by_id[task.operation_id],compiled.scenario.workload.operation_by_id[task.operation_id])
            self.assertEqual(base.scenario_document["costs"]["compute"][task.operation_id],compiled.scenario_document["costs"]["compute"][task.operation_id])
        with self.assertRaisesRegex(InvalidInput,"already materialized"):
            transform_ffn_execution(compiled,parse_ffn_execution(document(base)))

    def test_new_shape_transfer_must_have_exact_price(self):
        base=dense_source(); raw=document(base)
        needed=4*base.layer.hidden_size*(raw["shards"][0]["end"]-raw["shards"][0]["start"])
        table=replace(base.cost_table,transfers={key:value for key,value in base.cost_table.transfers.items() if key!=("dram_to_vram",needed)})
        raw["transfer_entries"]=[entry for entry in raw["transfer_entries"] if (entry["route"],entry["storage_bytes"])!=("dram_to_vram",needed)]
        with self.assertRaises(Unsupported): transform_ffn_execution(replace(base,cost_table=table),parse_ffn_execution(raw))

    def test_compose_attention_then_ffn_preserves_both_boundaries_and_runs(self):
        from llm_infer_model.m2.attention_execution import parse_attention_execution, transform_dense_attention
        base=dense_source(); task=next(task for task in base.tasks if task.kind=="attention")
        raw=dict(format_version="dense-attention-execution-0.1",variant="pure_attention",
            provenance="synthetic_execution",invocation_files=["source.json"],
            frontend_config=deepcopy(base.layer.physical.config),
            batch_contract="independent_requests_not_fused_batch_kernel",
            subbatches=[dict(id="cpu-attention",request_ids=[task.request_id],attention_device="cpu")],
            entries=[dict(kind=kind,device=device,phase=task.shape.phase,batch=1,
                          tokens=task.shape.tokens,visible_history=task.shape.visible_history,
                          duration_ns=101,workspace_bytes=4)
                     for kind in ("attention_core","attention_output") for device in ("cpu","gpu")],
            transfer_entries=[])
        attention=transform_dense_attention(base,parse_attention_execution(raw))
        compiled=transform_ffn_execution(attention,parse_ffn_execution(document(attention)))
        self.assertEqual(compiled.physical_boundary["attention_execution"],attention.physical_boundary["attention_execution"])
        self.assertIn("ffn_execution",compiled.physical_boundary)
        self.assertEqual(replay(compiled).status,STATUS_VALID)
        self.assertEqual(compiled.block_ids,base.block_ids)

    def test_shared_expert_uses_same_partition_and_keeps_shared_identity(self):
        from llm_infer_model.m2.spec import WeightRef, WeightStateRef
        from llm_infer_model.m2.subweights import partition_digest
        fixture=Path(__file__).parent/"fixtures/subweights"
        base=load_and_compile(fixture/"invocation.json")
        layer=replace(base.layer,ffn_params=dict(base.layer.ffn_params,shared_expert_weight="S",shared_expert_bytes=384),
                      weights=(*base.layer.weights,WeightRef("S","S",384,"ssd")))
        history=replace(base.state_in,weight_copies=(*base.state_in.weight_copies,WeightStateRef("S",1,384,("ssd",))))
        costs=dict(base.cost_table.entries)
        for (kind,device,shape),price in base.cost_table.entries.items():
            if kind=="expert_ffn" and shape.tokens==2: costs[("shared_expert_ffn",device,shape)]=price
        base=compile_invocation(layer,replace(base.invocation,state_in=history),replace(base.cost_table,entries=costs))
        rows=json.loads((fixture/"execution.json").read_text(encoding="utf-8"))
        manifest=deepcopy(rows["subweights"][0]); manifest["parent_weight_id"]="w:moe-small:S"
        for matrix in manifest["matrices"]:
            matrix["tensor_id"]=f"w:moe-small:S:v1:{matrix['stage']}"
        manifest["source_evidence"]["sha256"]=partition_digest(manifest["parent_weight_id"],1,manifest["matrices"])
        rows["subweights"].append(manifest)
        selected=transform_moe_execution(base,parse_execution_spec(rows))
        compiled=transform_ffn_execution(selected,parse_ffn_execution(document(selected)))
        shared=next(region for region in compiled.physical_boundary["ffn_execution"]["regions"] if region["expert_id"]=="shared")
        stages=compiled.physical_boundary["ffn_execution"]["operation_stages"]
        self.assertTrue(all(stages[oid]["expert_id"]=="shared" for oid in shared["operation_ids"]))
        self.assertEqual(len(compiled.physical_boundary["ffn_execution"]["weight_children"]["w:moe-small:S"]),6)
        self.assertEqual(replay(compiled).status,STATUS_VALID)
        expected,actual=numerical_region(compiled,shared,random.Random(817))
        for e,a in zip(expected,actual):
            for x,y in zip(e,a): self.assertAlmostEqual(x,y,places=11)

    def test_f32_source_geometry_and_swiglu_dependencies_are_checked(self):
        base=dense_source(); raw=document(base)
        for change in ({"dtype":"GGML_TYPE_F16"},{"ne":[32,16,1,1]}):
            workload=deepcopy(base.workload_document)
            next(item for item in workload["tensors"] if item["id"].endswith(":W_gate")).update(change)
            with self.assertRaises(Unsupported):
                transform_ffn_execution(replace(base,workload_document=workload),parse_ffn_execution(raw))
        base=moe_source(); workload=deepcopy(base.workload_document)
        gate=next(op for op in workload["operations"] if op["semantic_op"]=="expert_gate")
        gate["inputs"]=list(reversed(gate["inputs"]))
        with self.assertRaisesRegex(InvalidInput,"input/dependency contract"):
            transform_ffn_execution(replace(base,workload_document=workload),parse_ffn_execution(document(base)))


if __name__ == "__main__":
    unittest.main()
