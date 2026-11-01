"""The frozen composition gaps use independent small-graph semantic oracles.

These tests run the benchmark adapters, not native profiling or hardware timing.
Each adapter persists its handwritten expectations before compiling the graph.
"""
from pathlib import Path
import json
import tempfile
import unittest

from experiments import strategy_fidelity as ref
from experiments.ir_acceptance.basic import _pipeline
from experiments.ir_acceptance.combinations import c07, c08, c09, _early_prefetch, _check_budget
from llm_simulator import AnalyticalCosts, Cache, Model, ModelPlan, Prefetch, Scheduling, Simulator, Workload
from llm_simulator.simulation import CompiledSimulation


def _combined_regions(directory):
    """Additional regression, outside the frozen acceptance denominators.

    All three formerly missing composition mechanisms coexist in one plan.
    Expected geometry, slices and dependencies are specified before lowering;
    long CPU prices are explicit hypothetical inputs, never fitted timings.
    """
    expected = dict(layers=2, batch=2, one_runtime=True, selected_invocation="layer0-prefill",
        region_nodes=[dict(request_ids=["request0"], family="attention-stages-0.1"),
                      dict(request_ids=["request0"], family="ffn-execution-0.1"),
                      dict(request_ids=["request1"], family="ffn-execution-0.1")],
        cuts={"request0":4,"request1":12}, parent_ffn_bytes_per_layer=1536,
        complete_weight_bytes=4736, cache_budget_bytes=8192, operations=35,
        required=["positive cross-request CPU Attention / GPU FFN-stage overlap",
                  "actual charged layer1 prefetch overlaps active layer0 compute",
                  "shared full-parent matrices, two charged 32-byte CPU partial returns"],
        cpu_attention_duration_ns=50_000, costs="explicit synthetic batch2 per-request stage contracts")
    ref.write_json(directory/"expected.json", expected)
    simulator=Simulator(model=Model.dense(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
        hardware=ref.hardware(),workload=Workload(1,batch_size=2),cost_model=AnalyticalCosts())
    root=directory/"inputs"
    generated=simulator.cost_model.build_inputs(root,simulator.model,simulator.hardware,simulator.workload)
    for path in root.glob("*.costs.json"):
        document=json.loads(path.read_text(encoding="utf-8"))
        for entry in document["entries"]:
            if entry["kind"]=="attention" and entry["device"]=="cpu":entry["duration_ns"]=50_000
        ref.write_json(path,document)
    attention=ref.attention_spec()
    for entry in attention["entries"]:
        entry["batch"]=2
        if entry["kind"]=="attention_qk" and entry["device"]=="cpu":entry["duration_ns"]=50_000
    plan=ModelPlan(id="three-region-pipeline-prefetch",model_kind=generated["model_kind"],
        model_file=generated["model_file"],environment_file=generated["environment_file"],base_dir=root,
        max_decisions=100000,visibility_mode="data_dependencies",
        scheduling=Scheduling(default_device="gpu",release="last_use",cache=Cache(rule="lru",budget_bytes=8192),
            prefetch=Prefetch(next_layers=1,max_count=2,include_dense_weights=True)))
    plan.place("attention","cpu")
    plan.place("attention_qk","cpu")
    plan.place("attention_softmax","cpu")
    with plan.module("selected-attention",request_id="request0"):
        plan.transform("layer0-prefill",attention)
    for request,cut in expected["cuts"].items():
        ffn=ref.ffn_spec();ffn["weight_storage_contract"]="shared_parent_slice_views"
        ffn["shards"][0]["end"]=cut;ffn["shards"][1]["start"]=cut
        for entry in ffn["entries"]:entry["batch"]=2
        with plan.module("ffn-"+request,request_id=request):plan.transform("layer0-prefill",ffn)
    compiled=CompiledSimulation(plan.compile(),root,dict(generated["metadata"],cost_model=generated["cost_model"]),None)
    with compiled:
        ir=compiled.ir_document();ref.write_json(directory/"compiled-ir.json",ir)
        execution=ir["backend"]["execution"]
        ref.equal(len(execution),1,"one invocation execution declaration")
        ref.equal(execution[0]["scope_id"],"layer0-prefill","selected invocation identity")
        regions=execution[0]["spec"]["transforms"]
        ref.equal([dict(request_ids=r["request_ids"],family=r["transform"]["format_version"]) for r in regions],
                  expected["region_nodes"],"three region nodes in one common IR")
        result=ref.completed(compiled,directory)
        ops,tensors,events=ref.observed(result)
        ref.equal(len(ops),35,"complete two-layer combined graph")
        ref.require(all(o.status=="DONE" for o in ops.values()),"every intended operation finishes")
        ref.check_expected_wiring(ops,ref.attention_wiring())
        ref.equal(sum(t.storage_bytes for t in tensors.values() if t.role=="weight"),4736,"one complete physical weight inventory")
        ref.require(not any(t.role=="weight" and ":ffn:" in t.id for t in tensors.values()),"views cannot allocate duplicate weights")
        raw_ops={o["id"]:o for o in result.raw[1].execution_document()["workload"]["operations"]}
        for request,cut in expected["cuts"].items():
            def tid(suffix):return f"g:layer0-prefill:t:layer0:{request}:"+suffix
            scoped=[o for o in ops.values() if o.invocation_id=="layer0-prefill" and o.request_id==request]
            for shard,start,end,device in (("host",0,cut,"cpu"),("accelerator",cut,16,"gpu")):
                prefix="ffn.ffn."+shard+"."
                for stage in ("gate","up","swiglu","down"):
                    output=tid(prefix+("partial" if stage=="down" else stage))
                    op=next(o for o in scoped if output in o.outputs)
                    ref.equal(raw_ops[op.id]["op_params"],[start,end],"request-specific slice")
                    ref.equal(events[op.id].device,device,"request-specific shard device")
                    ref.equal(tensors[output].storage_bytes,32 if stage=="down" else 4*(end-start),"slice activation bytes")
                    if stage in ("gate","up","down"):
                        ref.require("w:layer0:W_"+stage in op.inputs,"shard reads the shared full physical parent")
            merge=next(o for o in scoped if o.kind=="ffn_sum")
            ref.equal(merge.inputs,(tid("ffn.ffn.host.partial"),tid("ffn.ffn.accelerator.partial")),"both slice partials reach merge")
            copied=ref.check_copy(result,tid("ffn.ffn.host.partial"),"dram","vram",32)
            ref.require(events[merge.id].start_ns>=copied.end_ns,"merge waits for charged host return")
        for layer in (0,1):
            ref.equal(sum(tensors[f"w:layer{layer}:W_{s}"].storage_bytes for s in ("gate","up","down")),1536,"per-layer shared FFN parents")
        for stage in ("gate","up","down"):ref.check_copy(result,"w:layer0:W_"+stage,"dram","vram",512)
        overlaps=[]
        for cpu in ops.values():
            if cpu.kind not in ("attention","attention_qk","attention_softmax") or events[cpu.id].device!="cpu":continue
            for gpu in ops.values():
                if not gpu.kind.startswith("ffn_") or events[gpu.id].device!="gpu" or gpu.request_id==cpu.request_id:continue
                a,b=events[cpu.id],events[gpu.id]
                duration=min(a.end_ns,b.end_ns)-max(a.start_ns,b.start_ns)
                if duration>0:overlaps.append(dict(cpu=cpu.id,gpu=gpu.id,overlap_ns=duration))
        ref.require(overlaps,"combined regions failed to produce cross-request execution pipeline")
        evidence=dict(region_nodes=3,operations=len(ops),overlap_witnesses=overlaps,
            prefetch=_early_prefetch(result),dependency_edges=ref.check_dependency_lifetimes(result),
            peak_weight_bytes=_check_budget(result,8192),summary=result.summary(),
            frozen_acceptance_denominator_changed=False)
        ref.write_json(directory/"observed.json",evidence)
        return evidence


class IRRegionAcceptanceTests(unittest.TestCase):
    def run_case(self, callback):
        with tempfile.TemporaryDirectory() as temporary:
            return callback(Path(temporary))

    def test_subbatch_pipeline_uses_common_frontend(self):
        evidence = self.run_case(_pipeline)
        self.assertEqual(evidence["entry_path"], "python_model_plan")
        self.assertTrue(evidence["unified_frontend_supported"])
        self.assertTrue(evidence["single_runtime"])
        self.assertGreater(len(evidence["overlap_witnesses"]), 0)
        self.assertEqual(evidence["completed_operations"], 6)

    def test_pipeline_and_prefetch_share_one_runtime(self):
        evidence = self.run_case(c07)
        self.assertTrue(evidence["single_runtime"])
        self.assertEqual(evidence["completed_operations"], 12)
        self.assertGreater(len(evidence["overlap_witnesses"]), 0)
        self.assertGreater(len(evidence["prefetch"]), 0)
        self.assertLessEqual(evidence["peak_weight_bytes"], 8192)

    def test_request_local_attention_preserves_other_request_and_kv(self):
        evidence = self.run_case(c08)
        self.assertEqual(evidence["selected_requests"], ["request0"])
        self.assertEqual(evidence["unchanged_requests"], ["request1"])
        self.assertEqual(evidence["exact_attention_operand_contracts"], 5)
        self.assertEqual(evidence["kv_blocks"], 4)
        self.assertEqual(evidence["kv_bytes"], 128)

    def test_disjoint_ffn_slices_share_physical_parents(self):
        evidence = self.run_case(c09)
        self.assertEqual(evidence["region_slices"], {
            "request0:host": [0, 4], "request0:accelerator": [4, 16],
            "request1:host": [0, 12], "request1:accelerator": [12, 16]})
        self.assertEqual(evidence["parent_matrix_bytes"], 1536)
        self.assertEqual(evidence["complete_weight_bytes"], 2368)
        self.assertEqual(len(evidence["full_parent_gpu_copies"]), 3)
        self.assertEqual(len(evidence["host_partial_copies"]), 2)
        self.assertEqual(evidence["operand_contracts"], 22)

    def test_three_new_compositions_share_one_plan(self):
        evidence=self.run_case(_combined_regions)
        self.assertEqual(evidence["region_nodes"],3)
        self.assertEqual(evidence["operations"],35)
        self.assertGreater(len(evidence["overlap_witnesses"]),0)
        self.assertGreater(len(evidence["prefetch"]),0)
        self.assertLessEqual(evidence["peak_weight_bytes"],8192)


if __name__ == "__main__":
    unittest.main()
