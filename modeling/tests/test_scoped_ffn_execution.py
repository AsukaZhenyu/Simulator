"""Request-local Dense FFN lowering with one shared physical weight inventory."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import random
import tempfile
import unittest

from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Workload
from llm_infer_model.m2.expand import load_and_compile
from llm_infer_model.m2.ffn_execution import KINDS, parse_ffn_execution, transform_ffn_execution
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.tensor.engine import STATUS_VALID, evaluate_mapping
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def source():
    # Explicit synthetic parameters, independent request kernels under B=2.
    hardware = Hardware(Device(1e9, 1e9), Device(2e9, 2e9), 1 << 20, 1 << 20,
                        1e9, 1e9, 1e9)
    with tempfile.TemporaryDirectory() as temporary:
        AnalyticalCosts().build_inputs(temporary,
            Model.dense(layers=1, hidden_size=8, intermediate_size=16, heads=2, kv_heads=1),
            hardware, Workload(1, batch_size=2))
        return load_and_compile(Path(temporary) / "layer0-prefill.json")


def specification(compiled, cut=4, *, contract="shared_parent_slice_views"):
    shape = next(task.shape for task in compiled.tasks if task.kind == "dense_ffn")
    raw = dict(format_version="ffn-execution-0.1", target_kinds=["dense_ffn"],
        intermediate_size=16, shards=[dict(id="host", start=0, end=cut, device="cpu"),
                                     dict(id="accelerator", start=cut, end=16, device="gpu")],
        merge_input_count=2, preprocess_device="gpu", merge_device="gpu",
        provenance="synthetic_execution", weight_storage_contract=contract,
        entries=[dict(kind=kind, device=device, phase=shape.phase, batch=shape.batch,
                      tokens=shape.tokens, visible_history=shape.visible_history, hidden_size=8,
                      intermediate_size=width, duration_ns=101, workspace_bytes=4)
                 for kind in sorted(KINDS) for device in ("cpu", "gpu") for width in (4, 12, 16)],
        transfer_entries=[dict(route=route, storage_bytes=size, duration_ns=size)
            for route in compiled.cost_table.allowed_transfer_routes for size in (4, 16, 32, 48, 128, 384)
            if (route, size) not in compiled.cost_table.transfers])
    return parse_ffn_execution(raw)


def replay(compiled):
    return evaluate_mapping(compiled.scenario, plan_moe(compiled))


class ScopedFFNExecutionTests(unittest.TestCase):
    def test_disjoint_requests_keep_one_parent_inventory_and_full_parent_copies(self):
        base = source()
        original = deepcopy(base.workload_document)
        first = transform_ffn_execution(base, specification(base, 4), request_ids=("request0",))
        untouched = [task for task in first.tasks if task.request_id == "request1"]
        self.assertEqual(untouched, [task for task in base.tasks if task.request_id == "request1"])
        self.assertEqual(first.scenario.workload.operation_by_id[untouched[-1].operation_id],
                         base.scenario.workload.operation_by_id[untouched[-1].operation_id])
        both = transform_ffn_execution(first, specification(first, 12), request_ids=("request1",))
        self.assertEqual(base.workload_document, original)
        boundary = both.physical_boundary["ffn_execution"]
        self.assertEqual(len(boundary["regions"]), 2)
        self.assertEqual(len(boundary["execution_specs"]), 2)
        self.assertEqual(boundary["weight_children"], {})
        self.assertEqual(boundary["weight_partitions"], [])
        self.assertEqual(len(boundary["weight_views"]), 12)
        for index, request in enumerate(("request0", "request1")):
            views = [view for view in boundary["weight_views"] if view["execution_spec_index"] == index]
            self.assertEqual(len(views), 6)
            self.assertTrue(all(view["request_ids"] == [request] and not view["allocation"] and
                                view["storage_bytes"] == 0 for view in views))
        parents = [both.scenario.workload.tensor_by_id[f"w:layer0:W_{stage}"]
                   for stage in ("gate", "up", "down")]
        self.assertEqual(sum(tensor.storage_bytes for tensor in parents), 1536)
        self.assertTrue(all(tensor.storage_bytes == 512 for tensor in parents))
        self.assertFalse(any(":ffn:" in tensor.id and tensor.role == "weight"
                             for tensor in both.scenario.workload.tensors))
        self.assertEqual(sum(t.storage_bytes for t in both.scenario.workload.tensors if t.role == "weight"),
                         sum(t.storage_bytes for t in base.scenario.workload.tensors if t.role == "weight"))
        for request, cut in (("request0", 4), ("request1", 12)):
            selected = [task for task in both.tasks if task.request_id == request and task.kind.startswith("ffn_")]
            self.assertEqual(len(selected), 11)
            for task in selected:
                op = both.scenario.workload.operation_by_id[task.operation_id]
                if task.kind in ("ffn_gate", "ffn_up", "ffn_down"):
                    stage = task.kind.removeprefix("ffn_")
                    self.assertEqual(op.inputs[0], f"w:layer0:W_{stage}")
                    expected = (0, cut) if ".host." in task.task_id else (cut, 16)
                    self.assertEqual(op.op_params, expected)
                    self.assertEqual(task.device, "cpu" if ".host." in task.task_id else "gpu")
                    width = expected[1] - expected[0]
                    self.assertEqual(both.scenario.workload.tensor_by_id[op.output].storage_bytes,
                                     32 if stage == "down" else 4 * width)
            for kind in ("ffn_norm", "ffn_sum", "ffn_residual"):
                self.assertEqual(sum(task.kind == kind for task in selected), 1)
        self.assertEqual(len(boundary["operation_stages"]), 22)
        run = replay(both)
        self.assertEqual(run.status, STATUS_VALID)
        parent_copies = [event for event in run.events if event.kind == "COPY" and
                         event.target_id in {tensor.id for tensor in parents} and event.dst == "vram"]
        self.assertEqual(len(parent_copies), 3)
        self.assertTrue(all(event.bytes_transferred == 512 for event in parent_copies))

    def test_selection_validation_and_overlap_are_explicit(self):
        base = source()
        spec = specification(base)
        for selection in ((), ("missing",), ("request0", "request0"), ["request0"]):
            with self.subTest(selection=selection), self.assertRaises(InvalidInput):
                transform_ffn_execution(base, spec, request_ids=selection)
        with self.assertRaisesRegex(Unsupported, "shared_parent_slice_views"):
            transform_ffn_execution(base, specification(base, contract="materialized_partition_replaces_parent"),
                                    request_ids=("request0",))
        first = transform_ffn_execution(base, spec, request_ids=("request0",))
        with self.assertRaisesRegex(InvalidInput, "already materialized"):
            transform_ffn_execution(first, spec, request_ids=("request0",))
        with self.assertRaisesRegex(InvalidInput, "already materialized"):
            transform_ffn_execution(first, spec)

    def test_legacy_complete_materialization_still_replaces_all_parents(self):
        base = source()
        transformed = transform_ffn_execution(base,
            specification(base, contract="materialized_partition_replaces_parent"))
        self.assertEqual(sum(task.kind == "dense_ffn" for task in transformed.tasks), 0)
        for stage in ("gate", "up", "down"):
            self.assertNotIn(f"w:layer0:W_{stage}", transformed.scenario.workload.tensor_by_id)
        self.assertEqual(replay(transformed).status, STATUS_VALID)

    def test_shared_parent_slice_algebra_matches_complete_swiglu_and_one_residual(self):
        base = source()
        first = transform_ffn_execution(base, specification(base, 4), request_ids=("request0",))
        compiled = transform_ffn_execution(first, specification(first, 12), request_ids=("request1",))
        rng = random.Random(2209)
        hidden, middle = 8, 16
        matrix = {stage: [[rng.uniform(-0.5, 0.6) for _ in range(hidden if stage == "down" else middle)]
                          for _ in range(middle if stage == "down" else hidden)]
                  for stage in ("gate", "up", "down")}
        gamma = [rng.uniform(0.7, 1.2) for _ in range(hidden)]
        def mm(vector, weights):
            return [sum(vector[k] * weights[k][j] for k in range(len(vector)))
                    for j in range(len(weights[0]))]
        for region in compiled.physical_boundary["ffn_execution"]["regions"]:
            x = [rng.uniform(-0.9, 1.1) for _ in range(hidden)]
            scale = math.sqrt(sum(v * v for v in x) / hidden + base.layer.physical.config["norm_epsilon"])
            normalized = [v * g / scale for v, g in zip(x, gamma)]
            gate, up = mm(normalized, matrix["gate"]), mm(normalized, matrix["up"])
            expected = [a + b for a, b in zip(mm([g / (1 + math.exp(-g)) * u for g, u in zip(gate, up)],
                                                 matrix["down"]), x)]
            values = {region["input_tensor"]: x}
            for oid in region["operation_ids"]:
                op = compiled.scenario.workload.operation_by_id[oid]
                kind = op.semantic_op
                if kind == "ffn_norm":
                    values[op.output] = normalized
                elif kind in ("ffn_gate", "ffn_up", "ffn_down"):
                    stage = kind.removeprefix("ffn_")
                    lo, hi = op.op_params
                    sliced = matrix[stage][lo:hi] if stage == "down" else [row[lo:hi] for row in matrix[stage]]
                    values[op.output] = mm(values[op.inputs[1]], sliced)
                elif kind == "ffn_swiglu":
                    values[op.output] = [g / (1 + math.exp(-g)) * u
                                         for g, u in zip(values[op.inputs[0]], values[op.inputs[1]])]
                elif kind in ("ffn_sum", "ffn_residual"):
                    values[op.output] = [sum(values[tid][j] for tid in op.inputs) for j in range(hidden)]
                else:
                    self.fail(kind)
            for actual, reference in zip(values[region["output_tensor"]], expected):
                self.assertAlmostEqual(actual, reference, places=12)


if __name__ == "__main__":
    unittest.main()
