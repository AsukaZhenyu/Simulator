"""Independent Dense request kernels share the ordinary global runtime.

These synthetic checks establish graph/accounting/composition behavior. They
do not assert fused-batch throughput or extend the measured B=1 contract.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_simulator import (
    AnalyticalCosts, Cache, Device, Hardware, Model, Prefetch, Scheduling,
    Simulator, Unsupported, Workload,
)
from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.dense_contract import validate_case
from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.hybrid_physical import signature_for
from llm_infer_model.m2.spec import parse_layer_document


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DenseSubbatchTests(unittest.TestCase):
    def setUp(self):
        device = Device(1e8, 1e10)
        self.hardware = Hardware(device, device, 1 << 24, 1 << 24, 1e9)
        self.model = Model.dense(layers=2)

    def inputs(self, directory, *, batch=2, gpu=True, decode=0):
        hardware = self.hardware if gpu else replace(self.hardware, gpu=None)
        return AnalyticalCosts().build_inputs(directory, self.model, hardware,
            Workload(2, batch_size=batch, decode_steps=decode))

    def test_prices_are_per_request_without_dividing_or_repeating_a_fused_batch(self):
        for gpu in (False, True):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as folder:
                one, two = Path(folder) / "one", Path(folder) / "two"
                self.inputs(one, batch=1, gpu=gpu)
                inputs = self.inputs(two, gpu=gpu)
                b1, b2 = read(one / "layer0-prefill.costs.json"), read(two / "layer0-prefill.costs.json")
                self.assertEqual(len(b1["entries"]), len(b2["entries"]))
                for first, second in zip(b1["entries"], b2["entries"]):
                    self.assertEqual(first["duration_ns"], second["duration_ns"])
                    self.assertEqual(second["batch"], 2)
                    self.assertEqual(second["accounting"], "per_request")
                    signature = second["physical_signature"]
                    self.assertEqual(signature["batch"], 1)
                    self.assertEqual(signature["kernel_batch"], 1)
                    self.assertEqual(signature["invocation_batch"], 2)
                self.assertIn("not_fused_batch_kernel", inputs["metadata"]["batch_accounting"])
                compiled = load_and_compile(two / "layer0-prefill.json")
                self.assertEqual(len(compiled.tasks), 6)
                self.assertEqual({task.shape.batch for task in compiled.tasks}, {2})
                self.assertEqual(compiled.physical_boundary["kernel_batch"], 1)

    def test_weights_shared_once_and_request_kv_blocks_are_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inputs(folder)
            compiled = load_and_compile(Path(folder) / "layer0-prefill.json")
            weights = [tensor for tensor in compiled.scenario.workload.tensors if tensor.role == "weight"]
            self.assertEqual(len(weights), len(compiled.layer.weights))
            self.assertEqual(sum(tensor.storage_bytes for tensor in weights),
                             sum(weight.storage_bytes for weight in compiled.layer.weights))
            blocks = compiled.new_blocks
            first, second = ({block.id for block in blocks[name]} for name in ("request0", "request1"))
            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertFalse(first & second)
            self.assertEqual(sum(block.storage_bytes for rows in blocks.values() for block in rows),
                             4 * self.model.kv_bytes_per_token)

    def test_multirequest_requires_explicit_synthetic_accounting_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inputs(folder)
            compiled = load_and_compile(Path(folder) / "layer0-prefill.json")
            raw = read(Path(folder) / "layer0.layer.json")
            raw["physical"].pop("request_accounting")
            layer = parse_layer_document(raw)
            with self.assertRaisesRegex(Unsupported, "requires batch=1"):
                compile_invocation(layer, compiled.invocation, compiled.cost_table)

    def test_measured_envelopes_and_prices_do_not_gain_batch_support(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inputs(folder)
            raw = read(Path(folder) / "layer0.layer.json")
            raw["provenance"] = "measured_macro"
            with self.assertRaisesRegex(Unsupported, "synthetic execution contract"):
                parse_layer_document(raw)
            table = read(Path(folder) / "layer0-prefill.costs.json")
            for entry in table["entries"]:
                entry.update(provenance="measured_macro", workspace_scope="macro_internal_scratch",
                             workspace_source="intentionally invalid B2 measured test")
            with self.assertRaisesRegex(Unsupported, "batch=1"):
                parse_cost_table(table)

    def test_native_measurement_case_still_requires_single_request(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inputs(folder)
            config = read(Path(folder) / "layer0.layer.json")["physical"]["config"]
            with self.assertRaisesRegex(Unsupported, "batch=1"):
                validate_case(dict(id="not-measured", kind="attention", phase="prefill", batch=2,
                                   tokens=2, visible_history=0, config=config, threads=1))

    def test_independent_mode_does_not_relabel_single_request_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            self.inputs(folder, batch=1)
            source = load_and_compile(Path(folder) / "layer0-prefill.json")
            raw = read(Path(folder) / "layer0.layer.json")
            raw["physical"]["request_accounting"] = "independent_per_request_kernel"
            layer = parse_layer_document(raw)
            table = read(Path(folder) / "layer0-prefill.costs.json")
            table["validation_only"] = False
            for entry in table["entries"]:
                shape = MacroShape(entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"])
                signature = signature_for(layer, entry["kind"], shape, entry["device"])
                entry.update(provenance="measured_macro", workspace_scope="macro_internal_scratch",
                    workspace_source="intentionally mismatched measured contract", physical_signature=signature,
                    contract_hash=canonical_contract_hash(signature))
            with self.assertRaisesRegex(Unsupported, "requires synthetic_layer per-request prices"):
                compile_invocation(layer, source.invocation, parse_cost_table(table))

    def test_pipeline_and_next_layer_prefetch_overlap_in_one_runtime(self):
        simulator = Simulator(model=self.model, hardware=self.hardware,
            workload=Workload(2, batch_size=2), cost_model=AnalyticalCosts(workspace_bytes=256))
        scheduling = Scheduling(placements={"projection": "cpu", "attention": "cpu", "dense_ffn": "gpu"},
            cache=Cache("lru", budget_bytes=1 << 20),
            prefetch=Prefetch(next_layers=1, max_count=9, include_dense_weights=True))
        with simulator.compile(scheduling=scheduling) as compiled:
            result = compiled.run()
            self.assertTrue(result.success, result.summary())
            compute = [event for event in result.events if event.kind == "COMPUTE"]
            attention = next(e for e in compute if e.target_id.endswith("layer0:request1:attn"))
            ffn = next(e for e in compute if e.target_id.endswith("layer0:request0:ffn"))
            self.assertLess(max(attention.start_ns, ffn.start_ns), min(attention.end_ns, ffn.end_ns))
            self.assertEqual((attention.device, ffn.device), ("cpu", "gpu"))
            self.assertEqual((attention.workspace_bytes, ffn.workspace_bytes), (256, 256))
            projection = next(e for e in compute if e.target_id.endswith("layer0:request0:proj"))
            prefetched = [e for e in result.events if e.kind == "COPY" and e.target_id.startswith("w:layer1:")]
            self.assertTrue(any(max(e.start_ns, projection.start_ns) < min(e.end_ns, projection.end_ns)
                                for e in prefetched))
            # Each device has one real compute slot, so two requests cannot
            # double-charge scratch on that device, even while CPU/GPU overlap.
            for device in ("cpu", "gpu"):
                events = sorted((e for e in compute if e.device == device), key=lambda e: e.start_ns)
                self.assertTrue(all(first.end_ns <= second.start_ns for first, second in zip(events, events[1:])))
            weights = [t for t in compiled.plan._compiled.model.scenario.workload.tensors if t.role == "weight"]
            self.assertEqual(len(weights), 2 * 9)

    def test_two_requests_keep_independent_state_across_layers_and_rounds(self):
        simulator = Simulator(model=self.model, hardware=self.hardware,
            workload=Workload(2, batch_size=2, decode_steps=2), cost_model=AnalyticalCosts())
        with simulator.compile(scheduling=Scheduling(default_device="cpu")) as compiled:
            result = compiled.run()
            self.assertTrue(result.success, result.summary())
            aggregate = compiled.plan._compiled.model.aggregate
            kv = [t for t in aggregate.scenario.workload.tensors if t.id.startswith("kv:")]
            self.assertEqual(len(kv), 2 * 2 * 4)
            self.assertEqual(len({t.id for t in kv}), len(kv))
            for layer in ("layer0", "layer1"):
                for request in ("request0", "request1"):
                    self.assertEqual({t.id for t in kv if t.id.startswith(f"kv:{layer}:{request}:")},
                                     {f"kv:{layer}:{request}:{position}" for position in range(4)})


if __name__ == "__main__":
    unittest.main()
