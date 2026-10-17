"""F32 subweight ranges, transfer accounting, and independent SwiGLU checks."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import random
import struct
import unittest

from llm_infer_model.m2.expand import load_and_compile, compile_invocation
from llm_infer_model.m2.costs import TransferEntry
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.m2.row_execution import (
    NEW_KINDS, parse_execution_spec, execution_spec_to_document, transform_moe_execution,
)
from llm_infer_model.m2.subweights import partition_digest
from llm_infer_model.tensor.engine import evaluate_mapping, STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

EXAMPLES = Path(__file__).resolve().parents[2] / "mapping/examples/m2"


def source():
    base = load_and_compile(EXAMPLES / "moe-interface-synthetic-prefill-2.json")
    # Physical F32 dimensions H=4, I=8: 3 * 4 * 8 * 4 = 384 B/expert.
    params = dict(base.layer.ffn_params, expert_weight_bytes=384)
    weights = tuple(replace(item, storage_bytes=384) if item.id.startswith("E") else item
                    for item in base.layer.weights)
    history = replace(base.state_in, weight_copies=tuple(
        replace(item, storage_bytes=384) if item.id.startswith("E") else item
        for item in base.state_in.weight_copies))
    prices = dict(base.cost_table.transfers)
    for route in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd"):
        prices[(route, 384)] = TransferEntry(3840, "synthetic_transfer")
    return compile_invocation(replace(base.layer, weights=weights, ffn_params=params),
                              replace(base.invocation, state_in=history),
                              replace(base.cost_table, transfers=prices))


def manifest(parent, version=1):
    matrices = [dict(stage=stage, tensor_id=f"{parent}:v{version}:{stage}", dtype="F32",
                    physical_shape=[8, 4] if stage == "down" else [4, 8],
                    byte_offset=index * 128, byte_length=128, version=version)
                for index, stage in enumerate(("gate", "up", "down"))]
    return dict(parent_weight_id=parent, parent_version=version, matrices=matrices,
                source_evidence=dict(kind="synthetic_layout_manifest", id="test-F32-H4-I8",
                sha256=partition_digest(parent, version, matrices)))


def document():
    value = dict(format_version="moe-execution-0.2", variant="selected_rows",
                 expert_variant="swiglu_staged", provenance="synthetic_execution",
                 intermediate_bytes_per_token=32, entries=[], transfer_entries=[],
                 index_storage_contract="stage_workspace_includes_row_index_temporaries",
                 weight_storage_contract="materialized_partition_replaces_parent",
                 subweights=[manifest(f"w:moe-small:E{i}") for i in range(4)])
    for kind in sorted(NEW_KINDS):
        for device in ("cpu", "gpu"):
            for count in (1, 2):
                value["entries"].append(dict(kind=kind, device=device, phase="prefill", batch=1,
                    token_count=count, hidden_bytes_per_token=16, intermediate_bytes_per_token=32,
                    duration_ns=100*count, workspace_bytes=4))
    for route in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd"):
        value["transfer_entries"].append(dict(route=route, storage_bytes=128, duration_ns=1280))
    return value


def refreshed(raw):
    for part in raw["subweights"]:
        part["source_evidence"]["sha256"] = partition_digest(
            part["parent_weight_id"], part["parent_version"], part["matrices"])
    return raw


class SubweightTests(unittest.TestCase):
    def test_saved_fixture_compiles_and_declares_same_storage_as_generated_case(self):
        import json
        fixture=Path(__file__).parent/"fixtures/subweights"
        raw=json.loads((fixture/"execution.json").read_text(encoding="utf-8"))
        compiled=transform_moe_execution(load_and_compile(fixture/"invocation.json"),parse_execution_spec(raw))
        self.assertEqual(raw,document())
        self.assertEqual({t.id:t.storage_bytes for t in compiled.scenario.workload.tensors},
                         {t.id:t.storage_bytes for t in transform_moe_execution(source(),parse_execution_spec(document())).scenario.workload.tensors})

    def test_roundtrip_exact_stage_inputs_complete_inventory_and_no_parent_charge(self):
        raw = document()
        self.assertEqual(execution_spec_to_document(parse_execution_spec(raw)), raw)
        compiled = transform_moe_execution(source(), parse_execution_spec(raw))
        tensors = compiled.scenario.workload.tensor_by_id
        for index in range(4):
            parent = f"w:moe-small:E{index}"
            self.assertNotIn(parent, tensors)
            self.assertNotIn(parent, compiled.scenario_document["goal"]["output_locations"])
            for stage in ("gate", "up", "down"):
                child = tensors[f"{parent}:v1:{stage}"]
                self.assertEqual(child.storage_bytes, 128)
                self.assertEqual(child.initial_locations, ("ssd",))
                self.assertEqual(child.ne, (8,4,1,1) if stage == "down" else (4,8,1,1))
        self.assertEqual(sum(t.storage_bytes for t in tensors.values() if t.role == "weight"), 4*384+48)
        for task in compiled.tasks:
            if task.kind in ("expert_gate", "expert_up", "expert_down"):
                stage = task.kind.removeprefix("expert_")
                expected = f"w:moe-small:E{task.expert_id}:v1:{stage}"
                self.assertEqual(task.weight_id, expected)
                self.assertEqual(compiled.scenario.workload.operation_by_id[task.operation_id].inputs[0], expected)
        self.assertEqual(len(compiled.physical_boundary["row_execution"]["subweights"]), 4)
        self.assertFalse(compiled.physical_boundary["row_execution"]["numeric_weights_loaded"])

    def test_cpu_gpu_transfer_bytes_and_exact_capacity_boundary(self):
        compiled = transform_moe_execution(source(), parse_execution_spec(document()))
        placement = {task.operation_id: "gpu" if task.kind in ("expert_gate", "expert_up") else "cpu"
                     for task in compiled.tasks}
        run = evaluate_mapping(compiled.scenario, plan_moe(compiled, placements=placement))
        self.assertEqual(run.status, STATUS_VALID, run.reason)
        # Active expert row counts 1,2,1. Gate/up GPU, SwiGLU/down CPU.
        # 6 matrices * 128 + 4 compact rows * 16; two intermediate vectors / row.
        self.assertEqual(run.h2d_bytes, 6*128+4*16)
        self.assertEqual(run.d2h_bytes, 2*4*32)
        self.assertEqual(run.ssd_read_bytes, 3*384)
        self.assertEqual(run.ssd_write_bytes, 0)
        self.assertEqual(run.peak_vram_bytes, 6*128+4*16+2*4*32+4)
        base = source()
        for capacity, valid in ((1092, True), (1091, False)):
            changed = compile_invocation(base.layer, replace(base.invocation,
                hardware=replace(base.invocation.hardware, vram_capacity_bytes=capacity)), base.cost_table)
            changed = transform_moe_execution(changed, parse_execution_spec(document()))
            result = evaluate_mapping(changed.scenario, plan_moe(changed, placements=placement))
            self.assertEqual(result.status == STATUS_VALID, valid)
            if not valid:
                self.assertIn("CAPACITY", result.reason.upper())

    def test_initial_parent_copies_replaced_and_alignment_charged_per_child(self):
        base = source()
        history = replace(base.state_in, weight_copies=tuple(
            replace(item, locations=("dram", "vram")) if item.id.startswith("E") else item
            for item in base.state_in.weight_copies))
        hardware = replace(base.invocation.hardware, dram_capacity_bytes=10000, vram_capacity_bytes=10000,
                           dram_alignment_bytes=256, vram_alignment_bytes=256)
        changed = compile_invocation(base.layer, replace(base.invocation, state_in=history, hardware=hardware), base.cost_table)
        compiled = transform_moe_execution(changed, parse_execution_spec(document()))
        for tensor in compiled.scenario.workload.tensors:
            if ":v1:" in tensor.id:
                self.assertEqual(tensor.initial_locations, ("dram", "vram"))
                self.assertEqual(compiled.scenario.size_alloc(tensor.id, "dram"), 256)
                self.assertEqual(compiled.scenario.size_alloc(tensor.id, "vram"), 256)
        # The children incur independent allocation alignment, not proportional
        # pieces of the old parent-rounded 512 B charge.
        self.assertEqual(sum(compiled.scenario.size_alloc(t.id, "vram") for t in compiled.scenario.workload.tensors
                             if ":v1:" in t.id), 12*256)

    def test_bad_ranges_identity_evidence_and_dtype_are_refused(self):
        mutations = [
            lambda p:p["matrices"][0].update(byte_offset=2),
            lambda p:p["matrices"][1].update(byte_offset=0),
            lambda p:p["matrices"][2].update(byte_offset=384),
            lambda p:p["matrices"][0].update(byte_length=124),
            lambda p:p["matrices"][0].update(version=2),
            lambda p:p["matrices"][0].update(tensor_id="w:fake"),
            lambda p:p["matrices"][0].update(dtype="Q4_K"),
            lambda p:p["source_evidence"].update(kind="recorded_contiguous_f32_manifest"),
        ]
        for mutation in mutations:
            raw=document(); mutation(raw["subweights"][0]); refreshed(raw)
            with self.subTest(raw=raw["subweights"][0]), self.assertRaises((InvalidInput, Unsupported)):
                parse_execution_spec(raw)
        raw=document(); raw["subweights"][0]["source_evidence"]["sha256"]="0"*64
        with self.assertRaisesRegex(InvalidInput,"sha256"):
            parse_execution_spec(raw)
        raw=document(); raw["weight_storage_contract"]="alias_parent"
        with self.assertRaises(Unsupported):parse_execution_spec(raw)

    def test_wrong_layer_geometry_and_out_of_bounds_parent_are_refused(self):
        raw=document()
        for part in raw["subweights"]:
            for item in part["matrices"]:item["physical_shape"]=[4,8]
        refreshed(raw)
        with self.assertRaisesRegex(InvalidInput,"shape does not match"):
            transform_moe_execution(source(),parse_execution_spec(raw))
        # Legacy 64 B expert cannot silently become three real 128 B matrices.
        old=load_and_compile(EXAMPLES/"moe-interface-synthetic-prefill-2.json")
        with self.assertRaisesRegex(InvalidInput,"cover parent"):
            transform_moe_execution(old,parse_execution_spec(document()))
        raw=document();raw["subweights"].pop()
        with self.assertRaisesRegex(InvalidInput,"inactive"):
            transform_moe_execution(source(),parse_execution_spec(raw))

    def test_version_change_requires_new_child_identity_and_matching_state(self):
        raw=document(); raw["subweights"][0]=manifest("w:moe-small:E0",2)
        with self.assertRaisesRegex(InvalidInput,"parent_version"):
            transform_moe_execution(source(),parse_execution_spec(raw))
        base=source(); history=replace(base.state_in,weight_copies=tuple(
            replace(item,version=2) if item.id=="E0" else item for item in base.state_in.weight_copies))
        changed=compile_invocation(base.layer,replace(base.invocation,state_in=history),base.cost_table)
        new=transform_moe_execution(changed,parse_execution_spec(raw))
        self.assertIn("w:moe-small:E0:v2:gate",new.scenario.workload.tensor_by_id)
        self.assertNotIn("w:moe-small:E0:v1:gate",new.scenario.workload.tensor_by_id)

    def test_typed_specs_cannot_bypass_partition_validation_and_legacy_is_opt_in(self):
        spec=parse_execution_spec(document())
        with self.assertRaises(InvalidInput):
            transform_moe_execution(source(),replace(spec,subweights=(None,)))
        bad=replace(spec.subweights[0],parent_version=2)
        with self.assertRaises(InvalidInput):
            transform_moe_execution(source(),replace(spec,subweights=(bad,*spec.subweights[1:])))
        raw=document();raw["format_version"]="moe-execution-0.1"
        with self.assertRaisesRegex(InvalidInput,"unknown keys"):
            parse_execution_spec(raw)

    def test_shared_expert_requires_its_own_complete_partition(self):
        from llm_infer_model.m2.spec import WeightRef,WeightStateRef
        base=source()
        layer=replace(base.layer,ffn_params=dict(base.layer.ffn_params,shared_expert_weight="S",shared_expert_bytes=384),
                      weights=(*base.layer.weights,WeightRef("S","S",384,"ssd")))
        history=replace(base.state_in,weight_copies=(*base.state_in.weight_copies,WeightStateRef("S",1,384,("ssd",))))
        entries=dict(base.cost_table.entries)
        for (kind,device,shape),price in base.cost_table.entries.items():
            if kind=="expert_ffn" and shape.tokens==2:
                entries[("shared_expert_ffn",device,shape)]=price
        source_shared=compile_invocation(layer,replace(base.invocation,state_in=history),
                                        replace(base.cost_table,entries=entries))
        with self.assertRaisesRegex(InvalidInput,"shared"):
            transform_moe_execution(source_shared,parse_execution_spec(document()))
        raw=document();raw["subweights"].append(manifest("w:moe-small:S"))
        compiled=transform_moe_execution(source_shared,parse_execution_spec(raw))
        region=next(item for item in compiled.physical_boundary["row_execution"]["regions"] if item["expert_id"]=="shared")
        self.assertEqual(len(region["work_items"]),2)
        for oid in region["expert_operation_ids"]:
            task=next(task for task in compiled.tasks if task.operation_id==oid)
            if task.kind!="expert_swiglu":self.assertTrue(task.weight_id.startswith("w:moe-small:S:v1:"))

    def test_independent_packed_reference_and_split_swiglu_nontrivial_f32_error(self):
        # No backend or real model claim: random non-binary-exact values test
        # contiguous offsets, GGML dimension order, and split tensor identity.
        generator=random.Random(71023)
        f32=lambda v:struct.unpack("<f",struct.pack("<f",v))[0]
        values=[f32(generator.uniform(-0.87,0.93)) for _ in range(96)]
        packed=struct.pack("<96f",*values)
        maximum=0.0
        for _ in range(7):
            x=[f32(generator.uniform(-1.31,1.27)) for _ in range(4)]
            # Fused reference accesses all matrices by packed element offset.
            gate=[sum(x[i]*values[j*4+i] for i in range(4)) for j in range(8)]
            up=[sum(x[i]*values[32+j*4+i] for i in range(4)) for j in range(8)]
            reference=[sum((gate[j]/(1+math.exp(-gate[j])))*up[j]*values[64+o*8+j]
                           for j in range(8)) for o in range(4)]
            decoded={matrix["stage"]:struct.unpack("<32f",packed[matrix["byte_offset"]:
                       matrix["byte_offset"]+matrix["byte_length"]])
                     for matrix in document()["subweights"][0]["matrices"]}
            def matmul(vector,matrix,outputs):
                return [f32(sum(vector[i]*matrix[o*len(vector)+i] for i in range(len(vector))))
                        for o in range(outputs)]
            g=matmul(x,decoded["gate"],8); u=matmul(x,decoded["up"],8)
            activation=[f32(a/(1+math.exp(-a))*b) for a,b in zip(g,u)]
            result=matmul(activation,decoded["down"],4)
            error=max(abs(a-b) for a,b in zip(reference,result));maximum=max(maximum,error)
            self.assertLess(error,1e-5)
        self.assertGreater(maximum,0.0)


if __name__ == "__main__":
    unittest.main()
