"""Hybrid Dense placement, copies, capacities and persistent history."""
from __future__ import annotations

import copy
import dataclasses
import math
import unittest

from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.expand import compile_invocation, plan_for
from llm_infer_model.m2.hybrid_physical import CONTRACT, HYBRID_FORMAT_VERSION, signature_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import parse_layer_document, parse_invocation_document, state_in_from_state_out
from llm_infer_model.tensor.engine import ACTION_COPY, ACTION_COMPUTE
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from .test_m2_full import full_layer_document


def hybrid_layer_document(*, gate="none", qk_norm="none"):
    raw = full_layer_document(gate=gate, qk_norm=qk_norm)
    raw.update(format_version=HYBRID_FORMAT_VERSION, id="hybrid-small")
    raw["physical"].update(contract=CONTRACT, cuda_device=0, cuda_graphs="disabled")
    return raw


def hybrid_invocation(layer, *, phase="prefill", state=None, capacity=1 << 20, devices=("cpu", "gpu")):
    raw = dict(format_version=HYBRID_FORMAT_VERSION, id="hybrid-"+phase, layer=layer.id,
               layer_file="hybrid-interface-synthetic.layer.json",
               costs_file="hybrid-interface-synthetic.costs.json", phase=phase, devices=list(devices),
               hardware=dict(dram_capacity_bytes=capacity, vram_capacity_bytes=capacity,
                             dram_alignment_bytes=1, vram_alignment_bytes=1,
                             dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0),
               requests=[dict(request_id="r0", position_start=0 if phase=="prefill" else 3,
                              tokens=3 if phase=="prefill" else 1, input_location="dram", output_location="dram")],
               state_in=dict(requests=[dict(request_id="r0", next_position=0, version=1,
                                           state_location="dram", blocks=[])]))
    invocation = parse_invocation_document(raw)
    return dataclasses.replace(invocation, state_in=state) if state is not None else invocation


def hybrid_cost_document(layer, invocation):
    request = invocation.requests[0]
    history = invocation.state_in.for_request(request.request_id)
    shape = MacroShape(invocation.phase, invocation.batch, request.tokens, history.next_position)
    durations = {"cpu": (1000, 9000, 1000), "gpu": (10000, 1000, 10000)}
    entries = []
    for i, task in enumerate(layer.tasks):
        for dev in invocation.devices:
            sig = signature_for(layer, task.kind, shape, dev)
            entries.append(dict(kind=task.kind, device=dev, phase=shape.phase, batch=1, tokens=shape.tokens,
                visible_history=shape.visible_history, duration_ns=durations[dev][i],
                workspace_bytes=(64, 128, 64)[i] if dev=="cpu" else (128, 256, 128)[i],
                provenance="synthetic_layer", accounting="per_request",
                physical_signature=sig, contract_hash=canonical_contract_hash(sig)))
    sizes = {w.storage_bytes for w in layer.weights}
    sizes.update(b.storage_bytes for b in history.blocks)
    sizes.update({layer.kv_bytes_per_token, layer.hidden_bytes_per_token * request.tokens})
    for port, ne in layer.physical.input_ports(request.tokens).items():
        sizes.add(4 * math.prod(ne))
    for ne in layer.physical.projection_ports(request.tokens).values():
        sizes.add(4 * math.prod(ne))
    # Fixed toy costs are declared per exact size. They do not model bandwidth.
    transfers = [dict(route=route, storage_bytes=n, duration_ns=10, provenance="synthetic_transfer")
                 for route in ("dram_to_vram", "vram_to_dram") for n in sorted(sizes)]
    return dict(format_version=HYBRID_FORMAT_VERSION, source="hybrid-interface-synthetic",
                entries=entries, transfer_entries=transfers)


def hybrid_compiled(*, devices=("cpu", "gpu"), state=None, phase="prefill", raw=None):
    layer = parse_layer_document(raw or hybrid_layer_document())
    invocation = hybrid_invocation(layer, devices=devices, state=state, phase=phase)
    return compile_invocation(layer, invocation, parse_cost_table(hybrid_cost_document(layer, invocation)))


def placements_for(compiled, devices=("cpu", "gpu", "cpu")):
    return {task.operation_id: dev for task, dev in zip(compiled.tasks, devices)}


class HybridReplayTests(unittest.TestCase):
    def test_both_devices_have_distinct_original_contracts(self):
        compiled = hybrid_compiled()
        for task in compiled.tasks:
            self.assertEqual(compiled.scenario.mapspace.devices_for(task.operation_id), ("cpu", "gpu"))
            envelopes = compiled.physical_boundary["physical_signatures"][task.operation_id]
            self.assertNotEqual(envelopes["cpu"]["contract"], envelopes["gpu"]["contract"])
        self.assertIn("synthetic_transfer", compiled.costs_source)

    def test_mixed_plan_includes_copy_and_delivers_kv_and_hidden(self):
        compiled = hybrid_compiled()
        selected = placements_for(compiled)
        result = run_invocation(compiled, placements=selected)
        self.assertEqual(result.status, "valid", result.evaluation.reason)
        self.assertEqual([a.device for a in result.evaluation.actions if a.kind==ACTION_COMPUTE],
                         ["cpu", "gpu", "cpu"])
        self.assertTrue(any(a.kind==ACTION_COPY and a.dst=="vram" for a in result.evaluation.actions))
        self.assertTrue(any(a.kind==ACTION_COPY and a.dst=="dram" for a in result.evaluation.actions))
        self.assertEqual(result.state_out.requests[0].next_position, 3)
        self.assertTrue(all("dram" in b.locations for b in result.state_out.requests[0].blocks))
        self.assertEqual(result.state_out.hidden[0].locations, ("dram",))

    def test_removing_input_copy_rejects_mapping(self):
        compiled = hybrid_compiled()
        actions = list(plan_for(compiled, placements=placements_for(compiled)))
        i = next(i for i,a in enumerate(actions) if a.kind==ACTION_COPY)
        del actions[i:i+2]
        result = run_invocation(compiled, actions=actions)
        self.assertEqual(result.status, "invalid_mapping")
        self.assertIsNone(result.state_out)

    def test_capacity_includes_copy_destinations_and_running_scratch(self):
        compiled = hybrid_compiled()
        selected = placements_for(compiled)
        base = run_invocation(compiled, placements=selected)
        peak = base.evaluation.peak_vram_bytes
        self.assertGreater(peak, 256)
        exact_inv = dataclasses.replace(compiled.invocation, hardware=dataclasses.replace(
            compiled.invocation.hardware, vram_capacity_bytes=peak))
        exact = compile_invocation(compiled.layer, exact_inv, compiled.cost_table)
        self.assertEqual(run_invocation(exact, placements=selected).status, "valid")
        short_inv = dataclasses.replace(exact_inv, hardware=dataclasses.replace(exact_inv.hardware, vram_capacity_bytes=peak-1))
        short = compile_invocation(compiled.layer, short_inv, compiled.cost_table)
        self.assertEqual(run_invocation(short, placements=selected).status, "invalid_mapping")

    def test_cpu_prefill_history_is_copied_for_gpu_attention_decode(self):
        first = hybrid_compiled()
        prefill = run_invocation(first, placements=placements_for(first, ("cpu", "cpu", "cpu")))
        state = state_in_from_state_out(prefill.state_out)
        second = hybrid_compiled(state=state, phase="decode")
        result = run_invocation(second, placements=placements_for(second))
        self.assertEqual(result.status, "valid", result.evaluation.reason)
        copied = {a.target_id for a in result.evaluation.actions if a.kind==ACTION_COPY and a.dst=="vram"}
        self.assertTrue({b.id for b in state.requests[0].blocks} <= copied)
        self.assertEqual(result.state_out.requests[0].next_position, 4)
        self.assertEqual(result.state_out.requests[0].version, 3)
        self.assertEqual(len(result.state_out.requests[0].blocks), 4)

    def test_unknown_gpu_workspace_not_excluded_or_zeroed(self):
        layer = parse_layer_document(hybrid_layer_document())
        inv = hybrid_invocation(layer)
        raw = hybrid_cost_document(layer, inv)
        gpu = next(e for e in raw["entries"] if e["device"]=="gpu")
        gpu.update(workspace_bytes=None, provenance="measured_macro",
                   workspace_scope="macro_internal_scratch", workspace_source="unknown_backend_scratch")
        raw["validation_only"] = False
        with self.assertRaisesRegex(Unsupported, "workspace_bytes"):
            compile_invocation(layer, inv, parse_cost_table(raw))

    def test_missing_exact_transfer_is_an_input_failure(self):
        layer = parse_layer_document(hybrid_layer_document())
        inv = hybrid_invocation(layer)
        raw = hybrid_cost_document(layer, inv)
        raw["transfer_entries"] = raw["transfer_entries"][1:]
        with self.assertRaises(Unsupported):
            compile_invocation(layer, inv, parse_cost_table(raw))

    def test_placement_must_be_complete_and_allowed(self):
        compiled = hybrid_compiled()
        with self.assertRaises(InvalidInput):
            plan_for(compiled, placements={})
        selected = placements_for(compiled)
        selected[compiled.compute_ids[0]] = "npu"
        with self.assertRaises(InvalidInput):
            plan_for(compiled, placements=selected)
        with self.assertRaises(InvalidInput):
            run_invocation(compiled, actions=(), placements=placements_for(compiled))

    def test_optional_gate_and_qk_norm_keep_same_copy_semantics(self):
        compiled = hybrid_compiled(raw=hybrid_layer_document(gate="sigmoid", qk_norm="rms"))
        result = run_invocation(compiled, placements=placements_for(compiled))
        self.assertEqual(result.status, "valid", result.evaluation.reason)
        self.assertIn("t:hybrid-small:r0:proj.gate", compiled.scenario.workload.tensor_by_id)

    def test_gpu_to_cpu_weight_copy_is_explicit(self):
        raw = hybrid_layer_document()
        for w in raw["weights"]:
            w["initial_location"] = "vram"
        compiled = hybrid_compiled(raw=raw)
        result = run_invocation(compiled)
        self.assertEqual(result.status, "valid", result.evaluation.reason)
        copied = {a.target_id for a in result.evaluation.actions if a.kind==ACTION_COPY and a.dst=="dram"}
        self.assertTrue({t.id for t in compiled.scenario.workload.tensors if t.role=="weight"} <= copied)

    def test_backend_parameter_changes_do_not_reuse_other_signatures(self):
        compiled = hybrid_compiled()
        changed = dataclasses.replace(compiled.layer, physical=dataclasses.replace(compiled.layer.physical, threads=4))
        with self.assertRaisesRegex(Unsupported, "physical_signature"):
            compile_invocation(changed, compiled.invocation, compiled.cost_table)
        changed = dataclasses.replace(compiled.layer, physical=dataclasses.replace(compiled.layer.physical, cuda_device=1))
        with self.assertRaisesRegex(Unsupported, "physical_signature"):
            compile_invocation(changed, compiled.invocation, compiled.cost_table)

    def test_cross_version_invocation_cannot_change_input_boundary(self):
        compiled = hybrid_compiled()
        for version in ("m2-layer-0.1", "m2-layer-0.2", "m2-layer-0.3", "m2-layer-0.4"):
            with self.subTest(version=version), self.assertRaisesRegex(InvalidInput, "format_version mismatch"):
                compile_invocation(compiled.layer, dataclasses.replace(compiled.invocation,
                    format_version=version), compiled.cost_table)

    def test_hybrid_document_cannot_infer_weight_placement(self):
        raw = hybrid_layer_document()
        del raw["weights"][0]["initial_location"]
        with self.assertRaisesRegex(InvalidInput, "explicit"):
            parse_layer_document(raw)

    def test_ssd_state_remains_unsupported(self):
        compiled = hybrid_compiled()
        first = run_invocation(compiled)
        state = state_in_from_state_out(first.state_out)
        request = state.requests[0]
        changed = dataclasses.replace(request, blocks=(dataclasses.replace(request.blocks[0],
            locations=("ssd",)), *request.blocks[1:]))
        bad = dataclasses.replace(state, requests=(changed,))
        with self.assertRaisesRegex(Unsupported, "SSD"):
            hybrid_compiled(phase="decode", state=bad)

    def test_cpu_and_gpu_only_subsets_have_no_implicit_execution_device(self):
        for device in ("cpu", "gpu"):
            with self.subTest(device=device):
                compiled = hybrid_compiled(devices=(device,))
                result = run_invocation(compiled)
                self.assertEqual(result.status, "valid", result.evaluation.reason)
                self.assertEqual({a.device for a in result.evaluation.actions if a.kind==ACTION_COMPUTE}, {device})

    def test_reserved_and_aligned_vram_budget_are_applied_to_copies(self):
        compiled = hybrid_compiled()
        selected = placements_for(compiled)
        inv = dataclasses.replace(compiled.invocation, hardware=dataclasses.replace(compiled.invocation.hardware,
            vram_alignment_bytes=128, vram_runtime_reserved_bytes=512))
        aligned = compile_invocation(compiled.layer, inv, compiled.cost_table)
        result = run_invocation(aligned, placements=selected)
        base = run_invocation(compiled, placements=selected)
        self.assertEqual(result.status, "valid")
        self.assertGreaterEqual(result.evaluation.peak_vram_bytes, base.evaluation.peak_vram_bytes+512)


if __name__ == "__main__":
    unittest.main()
