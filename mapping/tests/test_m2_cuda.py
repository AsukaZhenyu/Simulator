"""CUDA Dense contract and capacity/state semantics with explicit synthetic costs.

These tests validate the simulator's logic, not actual CUDA scratch measurement.
No hardware probe is executed here. An observed unknown scratch stays rejected.
"""
from __future__ import annotations

import copy
import dataclasses
import unittest

from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.cuda_physical import CUDA_FORMAT_VERSION, CudaDensePhysical, signature_for
from llm_infer_model.m2.dense_cuda_contract import (
    CONTRACT, build_plan, native_arguments, resident_accounting, signature, validate_case,
)
from llm_infer_model.m2.dense_contract import signature as cpu_signature
from llm_infer_model.m2.expand import compile_invocation, load_and_compile, plan_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import (
    parse_layer_document, state_in_from_state_out, validate_invocation,
)
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_EVICT, Action, evaluate_mapping,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

if __package__:
    from .test_m2_full import ROOT, full_layer_document, full_invocation
else:
    from test_m2_full import ROOT, full_layer_document, full_invocation


def cuda_layer_document(**kwargs):
    raw = full_layer_document(**kwargs)
    raw["format_version"] = CUDA_FORMAT_VERSION
    raw["id"] = "cuda-full-small"
    physical = raw["physical"]
    physical.pop("threads")
    physical.update(contract=CONTRACT, cuda_device=0, cuda_graphs="disabled")
    for weight in raw["weights"]:
        weight["initial_location"] = "vram"
    return raw


def cuda_invocation(layer, *, capacity=1 << 28, phase="prefill", state=None):
    invocation = full_invocation(layer, phase=phase)
    requests = tuple(dataclasses.replace(request, input_location="vram", output_location="vram")
                     for request in invocation.requests)
    history = (state if state is not None else dataclasses.replace(invocation.state_in,
        requests=tuple(dataclasses.replace(request, state_location="vram")
                       for request in invocation.state_in.requests)))
    return dataclasses.replace(invocation, format_version=CUDA_FORMAT_VERSION, devices=("gpu",),
        requests=requests, state_in=history,
        hardware=dataclasses.replace(invocation.hardware, dram_capacity_bytes=1,
                                     vram_capacity_bytes=capacity))


def cuda_cost_table(layer, invocation, *, scratch=(100, 200, 300), measured=False):
    request = invocation.requests[0]
    shape = MacroShape(invocation.phase, invocation.batch, request.tokens, request.position_start)
    entries = []
    for i, task in enumerate(layer.tasks):
        envelope = signature_for(layer, task.kind, shape)
        entry = dict(kind=task.kind, device="gpu", phase=shape.phase, batch=shape.batch,
                     tokens=shape.tokens, visible_history=shape.visible_history,
                     duration_ns=(i+1)*1000, workspace_bytes=scratch[i],
                     provenance="measured_macro" if measured else "synthetic_layer",
                     accounting="per_request", physical_signature=envelope,
                     contract_hash=canonical_contract_hash(envelope))
        if measured:
            entry.update(workspace_scope="macro_internal_scratch",
                         workspace_source="unit-test-observation-not-a-real-profile")
        entries.append(entry)
    return parse_cost_table(dict(format_version=CUDA_FORMAT_VERSION,
        source="CUDA-synthetic-contract-test", entries=entries, validation_only=False))


class CudaDenseBoundary(unittest.TestCase):
    def compile(self, *, raw=None, capacity=1 << 28):
        layer = parse_layer_document(raw or cuda_layer_document())
        invocation = cuda_invocation(layer, capacity=capacity)
        return compile_invocation(layer, invocation, cuda_cost_table(layer, invocation))

    def test_synthetic_file_fixtures_preserve_vram_state_without_performance_claim(self):
        folder = ROOT / "mapping/examples/m2"
        first = load_and_compile(folder / "cuda-interface-synthetic-prefill-3.json")
        prefill = run_invocation(first)
        second = load_and_compile(folder / "cuda-interface-synthetic-decode-1.json")
        decode = run_invocation(second)
        self.assertEqual(prefill.status, "valid")
        self.assertEqual(decode.status, "valid")
        self.assertEqual(prefill.makespan_ns, 6000)
        self.assertEqual(decode.makespan_ns, 6000)
        self.assertEqual(state_in_from_state_out(prefill.state_out), second.state_in)
        self.assertEqual({cost.provenance for cost in first.cost_table.entries.values()}, {"synthetic_layer"})
        self.assertIn("synthetic", first.cost_table.source)
        self.assertEqual(decode.state_out.requests[0].next_position, 4)
        self.assertEqual(decode.state_out.requests[0].state_location, "vram")

    def test_signature_is_cuda_specific_and_does_not_invent_cpu_threads(self):
        config = cuda_layer_document()["physical"]["config"]
        envelope = signature(config, "attention", "decode", 1, 3, 0)
        self.assertEqual(envelope["backend"], "CUDA0")
        self.assertEqual(envelope["device"], "gpu")
        self.assertEqual(envelope["cuda_graphs"], "disabled")
        self.assertEqual(envelope["backend_execution"], "single_cuda_backend_no_cpu_fallback")
        self.assertEqual(envelope["matmul_precision"], "ggml_cuda_backend_default")
        self.assertEqual(envelope["timing_scope"], "single_completed_cuda_graph_wall_ns")
        self.assertFalse(any("thread" in key for key in envelope))
        original = cpu_signature(config, "attention", "decode", 1, 3, 2)
        self.assertEqual(original["contract"], "ggml_cpu_dense_decoder_block_v1")
        self.assertEqual(original["threads_requested"], 2)
        self.assertNotIn("cuda_device", original)
        self.assertEqual(original["operations"], envelope["operations"])
        self.assertNotEqual(canonical_contract_hash(original), canonical_contract_hash(envelope))

    def test_vram_compute_slot_and_declared_ports_are_used(self):
        compiled = self.compile(raw=cuda_layer_document(gate="sigmoid", qk_norm="rms"))
        self.assertIsInstance(compiled.layer.physical, CudaDensePhysical)
        self.assertFalse(hasattr(compiled.layer.physical, "threads"))
        self.assertEqual({task.device for task in compiled.tasks}, {"gpu"})
        self.assertEqual({action.device for action in plan_for(compiled) if action.kind == ACTION_COMPUTE}, {"gpu"})
        result = run_invocation(compiled)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.makespan_ns, 6000)
        self.assertEqual(result.evaluation.peak_dram_bytes, 0)
        self.assertGreater(result.evaluation.peak_vram_bytes, 0)
        for tensor in compiled.scenario.workload.tensors:
            if tensor.initial_locations:
                self.assertEqual(tensor.initial_locations, ("vram",))
        self.assertEqual(set(compiled.scenario.goal.output_locations.values()), {"vram"})
        self.assertEqual(compiled.physical_boundary["backend"], "CUDA0")
        self.assertTrue(compiled.physical_boundary["complete_declared_decoder_block"])
        self.assertFalse(compiled.physical_boundary["original_model_kernel_reproduced"])

    def test_vram_capacity_boundary_includes_live_kv_ports_and_known_scratch(self):
        raw = cuda_layer_document(gate="sigmoid", qk_norm="rms")
        logical = resident_accounting(raw["physical"]["config"], "block", 3, 0)["resident_logical_bytes"]
        exact = logical+300
        self.assertEqual(run_invocation(self.compile(raw=raw, capacity=exact)).status, "valid")
        self.assertEqual(run_invocation(self.compile(raw=raw, capacity=exact-1)).status, "invalid_mapping")

    def test_prefill_decode_preserves_vram_kv_and_absolute_positions(self):
        first = self.compile()
        prefill = run_invocation(first)
        history = state_in_from_state_out(prefill.state_out)
        decode = cuda_invocation(first.layer, phase="decode", state=history)
        second = compile_invocation(first.layer, decode, cuda_cost_table(first.layer, decode))
        old = {block.id for block in history.requests[0].blocks}
        self.assertTrue(old.issubset(second.scenario.workload.operations[1].inputs))
        for block in history.requests[0].blocks:
            self.assertEqual(block.locations, ("vram",))
            self.assertEqual(second.scenario.goal.output_locations[block.id], "vram")
        self.assertEqual(second.physical_boundary["absolute_positions"], {"r0": [3]})
        result = run_invocation(second)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.state_out.requests[0].next_position, 4)
        self.assertEqual(result.state_out.total_kv_bytes("r0"), 256)
        self.assertEqual(result.state_out.requests[0].state_location, "vram")
        actions = []
        for task in second.tasks:
            actions.extend((Action(ACTION_COMPUTE, task.operation_id, device="gpu"), Action(ACTION_ADVANCE)))
            if task.kind == "attention":
                actions.append(Action(ACTION_EVICT, next(iter(old)), location="vram"))
        self.assertIn("LIVE_VALUE_LOSS", evaluate_mapping(second.scenario, tuple(actions)).reason)

    def test_unknown_observed_cuda_workspace_is_preserved_and_rejects_execution(self):
        layer = parse_layer_document(cuda_layer_document())
        invocation = cuda_invocation(layer)
        table = cuda_cost_table(layer, invocation, scratch=(None, None, None), measured=True)
        self.assertTrue(all(cost.workspace_bytes is None for cost in table.entries.values()))
        with self.assertRaisesRegex(Unsupported, "workspace_bytes.*未知"):
            compile_invocation(layer, invocation, table)

    def test_cpu_costs_cannot_be_relabelled_as_cuda_costs(self):
        layer = parse_layer_document(cuda_layer_document())
        invocation = cuda_invocation(layer)
        table = cuda_cost_table(layer, invocation)
        entries = {}
        for key, cost in table.entries.items():
            envelope = cpu_signature(layer.physical.config, key[0], key[2].phase,
                key[2].tokens, key[2].visible_history, 2)
            entries[key] = dataclasses.replace(cost, physical_signature=envelope,
                                               contract_hash=canonical_contract_hash(envelope))
        with self.assertRaisesRegex(Unsupported, "不匹配"):
            compile_invocation(layer, invocation, dataclasses.replace(table, entries=entries))

    def test_cuda_index_is_a_cost_contract_dimension(self):
        compiled = self.compile()
        changed = dataclasses.replace(compiled.layer,
            physical=dataclasses.replace(compiled.layer.physical, cuda_device=1))
        with self.assertRaisesRegex(Unsupported, "不匹配"):
            compile_invocation(changed, compiled.invocation, compiled.cost_table)

    def test_cpu_and_hybrid_device_sets_are_rejected(self):
        compiled = self.compile()
        for devices in (("cpu",), ("cpu", "gpu"), ("gpu", "cpu"), ()):
            with self.subTest(devices=devices), self.assertRaisesRegex(Unsupported, "devices"):
                validate_invocation(compiled.layer, dataclasses.replace(compiled.invocation, devices=devices))

    def test_implicit_input_output_state_and_weight_transfers_are_rejected(self):
        compiled = self.compile()
        for field in ("input_location", "output_location"):
            changed = dataclasses.replace(compiled.invocation,
                requests=(dataclasses.replace(compiled.invocation.requests[0], **{field: "dram"}),))
            with self.subTest(field=field), self.assertRaises(Unsupported):
                validate_invocation(compiled.layer, changed)
        state = dataclasses.replace(compiled.state_in,
            requests=(dataclasses.replace(compiled.state_in.requests[0], state_location="dram"),))
        with self.assertRaises(Unsupported):
            validate_invocation(compiled.layer, compiled.invocation, state)
        raw = cuda_layer_document(); raw["weights"][0]["initial_location"] = "dram"
        with self.assertRaisesRegex(Unsupported, "weight"):
            parse_layer_document(raw)

    def test_history_block_domain_cannot_bypass_vram_requirement(self):
        first = self.compile()
        history = state_in_from_state_out(run_invocation(first).state_out)
        request = history.requests[0]
        bad = dataclasses.replace(history, requests=(dataclasses.replace(request,
            blocks=(dataclasses.replace(request.blocks[0], locations=("dram",)), *request.blocks[1:])),))
        decode = cuda_invocation(first.layer, phase="decode", state=bad)
        with self.assertRaisesRegex(Unsupported, "history KV"):
            validate_invocation(first.layer, decode)

    def test_no_graph_mode_device_or_thread_defaults_are_inferred(self):
        for missing in ("cuda_device", "cuda_graphs"):
            raw = cuda_layer_document(); raw["physical"].pop(missing)
            with self.subTest(missing=missing), self.assertRaises(InvalidInput):
                parse_layer_document(raw)
        for key, value in (("cuda_device", True), ("cuda_device", -1), ("cuda_graphs", "enabled")):
            raw = cuda_layer_document(); raw["physical"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((InvalidInput, Unsupported)):
                parse_layer_document(raw)
        raw = cuda_layer_document(); raw["physical"]["threads"] = 2
        with self.assertRaises(InvalidInput):
            parse_layer_document(raw)

    def test_direct_dataclasses_are_checked_and_formats_are_not_reinterpreted(self):
        compiled = self.compile()
        for key, value in (("cuda_device", True), ("cuda_graphs", "enabled"), ("contract", "invented")):
            changed = dataclasses.replace(compiled.layer,
                physical=dataclasses.replace(compiled.layer.physical, **{key: value}))
            with self.subTest(key=key), self.assertRaises((InvalidInput, Unsupported)):
                compile_invocation(changed, compiled.invocation, compiled.cost_table)
        for version in ("m2-layer-0.1", "m2-layer-0.2", "m2-layer-0.3"):
            with self.subTest(version=version), self.assertRaisesRegex(InvalidInput, "format_version mismatch"):
                compile_invocation(compiled.layer, dataclasses.replace(compiled.invocation, format_version=version),
                                   compiled.cost_table)

    def test_validation_only_quality_gate_is_preserved(self):
        compiled = self.compile()
        table = dataclasses.replace(compiled.cost_table, validation_only=True)
        with self.assertRaisesRegex(Unsupported, "validation_only"):
            compile_invocation(compiled.layer, compiled.invocation, table)
        opted = dataclasses.replace(compiled.invocation, allow_validation_only_costs=True)
        self.assertEqual(run_invocation(compile_invocation(compiled.layer, opted, table)).status, "valid")


class CudaMeasurementCaseContract(unittest.TestCase):
    def case(self):
        return dict(id="cuda-small", kind="attention", phase="decode", batch=1,
                    tokens=1, visible_history=3, config=cuda_layer_document()["physical"]["config"])

    def test_explicit_device_binding_and_native_arguments_have_no_cpu_threads(self):
        suite = dict(schema_version="dense_layer_measurement_suite_v1", cases=[self.case()])
        case = build_plan(suite, 1)[0]
        self.assertEqual(case["cuda_device"], 1)
        self.assertEqual(case["cuda_graphs"], "disabled")
        args = native_arguments(case, 5, 16, 17)
        self.assertNotIn("--threads", args)
        self.assertEqual(args[args.index("--cuda-device")+1], "1")
        self.assertEqual(args[args.index("--cuda-graphs")+1], "disabled")
        self.assertEqual(args[args.index("--history")+1], "3")
        self.assertNotIn("threads", case)

    def test_case_device_mismatch_cpu_threads_and_missing_native_binding_are_rejected(self):
        case = self.case()
        with self.assertRaises(InvalidInput):
            native_arguments(case, 5, 16, 17)
        for key, value in (("threads", 1), ("cuda_graphs", "enabled"), ("cuda_device", True)):
            with self.subTest(key=key), self.assertRaises((InvalidInput, Unsupported)):
                validate_case({**case, key: value})
        with self.assertRaisesRegex(InvalidInput, "device differs"):
            build_plan(dict(schema_version="dense_layer_measurement_suite_v1",
                            cases=[{**case, "cuda_device": 1}]), 0)
        with self.assertRaisesRegex(InvalidInput, "unique"):
            build_plan(dict(schema_version="dense_layer_measurement_suite_v1", cases=[case, copy.deepcopy(case)]), 0)


if __name__ == "__main__":
    unittest.main()
