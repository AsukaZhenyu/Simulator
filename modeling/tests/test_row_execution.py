"""Selected-row materialization, independent costs, and causal visibility."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.expand import load_and_compile, compile_invocation
from llm_infer_model.m2.moe_spec import RouteTraceEntry
from llm_infer_model.m2.spec import WeightRef, WeightStateRef
from llm_infer_model.m2.row_execution import (
    parse_execution_spec, execution_spec_to_document, transform_moe_execution, NEW_KINDS,
)
from llm_infer_model.tensor.engine import ACTION_ADVANCE, Action, evaluate_mapping, STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, build_scenario

_EXAMPLES = Path(__file__).resolve().parents[2] / "mapping/examples/m2"


def source():
    return load_and_compile(_EXAMPLES / "moe-interface-synthetic-prefill-2.json")


def document(staged=False):
    kinds = sorted(NEW_KINDS if staged else {"gather_rows", "scatter_rows", "combine_rows"})
    value = dict(format_version="moe-execution-0.1", variant="selected_rows",
                 expert_variant="swiglu_staged" if staged else "opaque",
                 provenance="synthetic_execution", entries=[], transfer_entries=[],
                 index_storage_contract="stage_workspace_includes_row_index_temporaries")
    if staged:
        value["intermediate_bytes_per_token"] = 32
    else:
        value["opaque_cost_source"] = "existing_synthetic_per_expert_group"
    for kind in kinds:
        for device in ("cpu", "gpu"):
            for count in (1, 2):
                value["entries"].append(dict(kind=kind,device=device,phase="prefill",batch=1,
                    token_count=count,hidden_bytes_per_token=16,
                    intermediate_bytes_per_token=32 if staged else 0,
                    duration_ns={"gather_rows":100,"scatter_rows":200,"combine_rows":300}.get(kind,500)*count,
                    workspace_bytes=4))
    return value


def replay_serial(compiled, *, select=None):
    # Exercise the model graph through the existing primitive kernel; not a
    # replacement policy or an oracle exposed to mapping.
    from llm_infer_model.m2.moe import plan_moe
    placements = {t.operation_id:(select(t) if select else "cpu") for t in compiled.tasks}
    return evaluate_mapping(compiled.scenario, plan_moe(compiled, placements=placements))


class RowExecutionTests(unittest.TestCase):
    def test_default_session_and_input_compilation_remain_unchanged(self):
        original = source(); snapshot = deepcopy(original.workload_document)
        evaluation = replay_serial(original)
        self.assertEqual(evaluation.makespan_ns,34_000_000)
        transform_moe_execution(original,parse_execution_spec(document()))
        self.assertEqual(original.workload_document,snapshot)
        self.assertEqual(CausalM2Session(original).execution_document()["execution_variant"],"whole_request")

    def test_opaque_gather_compact_rows_and_no_route_port_on_expert(self):
        compiled = transform_moe_execution(source(),parse_execution_spec(document()))
        tensors=compiled.scenario.workload.tensor_by_id
        regions=compiled.physical_boundary["row_execution"]["regions"]
        self.assertEqual([r["expert_id"] for r in regions],[0,1,2])
        for region in regions:
            expected=len(region["work_items"])*16
            self.assertEqual(tensors[region["grouped_input_tensor"]].storage_bytes,expected)
            self.assertEqual(tensors[region["grouped_output_tensor"]].storage_bytes,expected)
            expert=compiled.scenario.workload.operation_by_id[region["expert_operation_ids"][0]]
            self.assertEqual(expert.inputs,(f"w:moe-small:E{region['expert_id']}",region["grouped_input_tensor"]))
        self.assertEqual(compiled.hidden_out_ids,source().hidden_out_ids)
        self.assertEqual(compiled.block_ids,source().block_ids)
        run=replay_serial(compiled)
        self.assertEqual(run.status,STATUS_VALID)
        # Original 1ms combine is replaced by 600ns, with 400ns gathers and
        # 800ns scatters. Whole expert costs are retained only by explicit opt-in.
        self.assertEqual(run.makespan_ns,33_001_800)

    def test_staged_swiglu_materializes_storage_and_whole_weight_requirements(self):
        compiled=transform_moe_execution(source(),parse_execution_spec(document(True)))
        region=compiled.physical_boundary["row_execution"]["regions"][1]
        tasks={t.operation_id:t for t in compiled.tasks}
        stage_tasks=[tasks[oid] for oid in region["expert_operation_ids"]]
        self.assertEqual([t.kind for t in stage_tasks],["expert_gate","expert_up","expert_swiglu","expert_down"])
        for task in stage_tasks:
            operation=compiled.scenario.workload.operation_by_id[task.operation_id]
            self.assertEqual(compiled.scenario.workload.tensor_by_id[task.output_tensor].storage_bytes,
                             32 if task.kind=="expert_down" else 64)
            if task.kind!="expert_swiglu":
                self.assertIn("w:moe-small:E1",operation.inputs)
            self.assertEqual(compiled.scenario.costs.compute_cost(task.operation_id,"cpu").duration_ns,1000)
        run=replay_serial(compiled,select=lambda task:"gpu" if task.kind in ("expert_gate","expert_up") else "cpu")
        self.assertEqual(run.status,STATUS_VALID)
        self.assertGreater(run.d2h_bytes,0)
        self.assertIn("no aliases",compiled.physical_boundary["row_execution"]["stage_weight_semantics"])

    def test_independent_stage_prices_are_not_divided_original_macro_prices(self):
        original=source(); base=parse_execution_spec(document(True))
        modified=document(True)
        for entry in modified["entries"]:
            if entry["kind"]=="expert_down":entry["duration_ns"]+=12345
        before=replay_serial(transform_moe_execution(original,base))
        after=replay_serial(transform_moe_execution(original,parse_execution_spec(modified)))
        self.assertEqual(after.makespan_ns-before.makespan_ns,3*12345)

    def test_explicit_stage_workspace_and_alignment_in_public_prices(self):
        original=source()
        hardware=replace(original.invocation.hardware,dram_alignment_bytes=64,vram_alignment_bytes=128)
        compiled=compile_invocation(original.layer,replace(original.invocation,hardware=hardware),original.cost_table)
        session=CausalM2Session(compiled,execution_spec=parse_execution_spec(document()))
        obs=session.observe()
        first=obs.operations[0]
        self.assertEqual([(p.device,p.allocated_workspace_bytes) for p in first.prices],[("cpu",64),("gpu",128)])
        weight=next(t for t in obs.tensors if t.id=="w:moe-small:E0")
        self.assertEqual(weight.allocation_bytes,(("dram",64),("vram",128)))
        self.assertEqual(weight.initial_locations,("ssd",))
        self.assertEqual(weight.goal_location,"ssd")

    def test_prefix_noninterference_hides_all_rows_stages_and_shapes(self):
        from llm_infer_model.m2.moe import plan_moe
        original=source()
        changed=replace(original.invocation,route_trace=(RouteTraceEntry("r0","moe-small",0,(2,3)),
                                                       RouteTraceEntry("r0","moe-small",1,(3,0))))
        alternative=compile_invocation(original.layer,changed,original.cost_table)
        specs=parse_execution_spec(document(True))
        sessions=(CausalM2Session(original,execution_spec=specs),CausalM2Session(alternative,execution_spec=specs))
        plan=plan_moe(original)
        for action in plan[:6]:
            self.assertEqual(sessions[0].observe(),sessions[1].observe())
            self.assertFalse(sessions[0].observe().task_graph_complete)
            for session in sessions:
                step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok,step.error_message)
        self.assertNotEqual(sessions[0].observe(),sessions[1].observe())
        self.assertTrue(all(session.observe().task_graph_complete for session in sessions))
        obs=sessions[0].observe()
        gate=next(op for op in obs.operations if op.kind=="expert_gate" and op.expert_id==1)
        self.assertEqual(gate.work_items,(("r0",0),("r0",1)))
        tensor=next(t for t in obs.tensors if t.id==gate.outputs[0])
        self.assertEqual(tensor.row_scope,gate.work_items)
        self.assertEqual(gate.region_id,"expert:1")

    def test_unknown_cost_missing_shape_and_nonpositive_or_unknown_workspace_refused(self):
        value=document(); value["entries"]=[e for e in value["entries"] if not
                    (e["kind"]=="gather_rows" and e["token_count"]==2 and e["device"]=="cpu")]
        with self.assertRaisesRegex(Unsupported,"missing exact execution cost"):
            transform_moe_execution(source(),parse_execution_spec(value))
        for field,value in (("workspace_bytes",None),("workspace_bytes",True),("duration_ns",0)):
            obj=document();obj["entries"][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(InvalidInput):parse_execution_spec(obj)

    def test_strict_spec_roundtrip_duplicates_unknown_keys_and_explicit_ack(self):
        for staged in (False,True):
            obj=document(staged)
            self.assertEqual(execution_spec_to_document(parse_execution_spec(obj)),obj)
        obj=document();obj["entries"].append(dict(obj["entries"][0]))
        with self.assertRaises(InvalidInput):parse_execution_spec(obj)
        obj=document();obj["arbitrary"]=1
        with self.assertRaises(InvalidInput):parse_execution_spec(obj)
        obj=document();del obj["opaque_cost_source"]
        with self.assertRaisesRegex(InvalidInput,"acknowledgement"):parse_execution_spec(obj)
        obj=document(True);obj["intermediate_bytes_per_token"]=6
        with self.assertRaises(Unsupported):parse_execution_spec(obj)
        obj=document();obj["transfer_entries"]=[dict(route="dram_to_vram",storage_bytes=16,duration_ns=3)]*2
        with self.assertRaises(InvalidInput):parse_execution_spec(obj)

    def test_synthetic_semantics_cannot_be_called_measured(self):
        compiled=transform_moe_execution(source(),parse_execution_spec(document()))
        workload=deepcopy(compiled.workload_document);workload["origin"]["kind"]="measured_macro"
        with self.assertRaisesRegex(Unsupported,"row execution requires synthetic_layer"):
            build_scenario(workload,compiled.scenario_document)

    def test_debug_export_detached_and_contains_actual_causal_catalogue(self):
        session=CausalM2Session(source(),execution_spec=parse_execution_spec(document(True)))
        doc=session.execution_document();self.assertEqual(doc["expert_variant"],"swiglu_staged")
        self.assertIn("w:moe-small:E3",doc["scenario"]["mapspace"]["allowed_transfers"]["ssd_to_dram"])
        doc["workload"]["tensors"].clear()
        self.assertTrue(session.execution_document()["workload"]["tensors"])

    def test_expert_cpu_receives_compact_rows_with_charged_transfer_events(self):
        compiled=transform_moe_execution(source(),parse_execution_spec(document()))
        run=replay_serial(compiled,select=lambda task:"cpu" if task.kind in ("expert_ffn","scatter_rows") else "gpu")
        self.assertEqual(run.status,STATUS_VALID)
        row_input_ids={r["grouped_input_tensor"] for r in compiled.physical_boundary["row_execution"]["regions"]}
        moved=[event for event in run.events if event.target_id in row_input_ids and event.kind=="COPY"]
        self.assertEqual(sorted(event.bytes_transferred for event in moved),[16,16,32])
        self.assertTrue(all(event.src=="vram" and event.dst=="dram" for event in moved))

    def test_workspace_unknown_from_direct_typed_spec_cannot_bypass_parser(self):
        spec=parse_execution_spec(document())
        price=replace(spec.entries[0],workspace_bytes=None)
        spec=replace(spec,entries=(price,*spec.entries[1:]))
        with self.assertRaises(InvalidInput):transform_moe_execution(source(),spec)

    def test_cross_request_group_order_and_per_request_scatter_no_drop_or_duplicate(self):
        original=source()
        requests=(original.invocation.requests[0],replace(original.invocation.requests[0],request_id="r1"))
        history=replace(original.state_in,requests=(original.state_in.requests[0],
                        replace(original.state_in.requests[0],request_id="r1")))
        trace=tuple(RouteTraceEntry(req,"moe-small",position,(0,1) if position==0 else (2,3))
                    for req in ("r0","r1") for position in (0,1))
        entries={(kind,device,replace(shape,batch=2)):price
                 for (kind,device,shape),price in original.cost_table.entries.items()}
        # Explicit synthetic batch-2 table for this graph, never a claim of
        # measured batch-1 equivalence or a backend batch kernel.
        invocation=replace(original.invocation,requests=requests,state_in=history,route_trace=trace)
        batched=compile_invocation(original.layer,invocation,replace(original.cost_table,entries=entries))
        raw=document(True)
        for entry in raw["entries"]:entry["batch"]=2
        compiled=transform_moe_execution(batched,parse_execution_spec(raw))
        regions=compiled.physical_boundary["row_execution"]["regions"]
        scopes=compiled.physical_boundary["row_execution"]["tensor_row_scopes"]
        for region in regions:
            self.assertEqual([row["request_id"] for row in region["work_items"]],["r0","r1"])
            scatter_rows=[]
            for oid in region["scatter_operation_ids"]:
                operation=compiled.scenario.workload.operation_by_id[oid]
                scatter_rows.extend(scopes[operation.output])
                self.assertEqual(compiled.scenario.workload.tensor_by_id[operation.output].storage_bytes,16)
            self.assertEqual(scatter_rows,region["work_items"])
        self.assertEqual(sum(len(region["work_items"]) for region in regions),8)
        self.assertEqual(replay_serial(compiled).status,STATUS_VALID)
        from llm_infer_model.m2.moe import plan_moe
        session=CausalM2Session(batched,execution_spec=parse_execution_spec(raw))
        for index,action in enumerate(plan_moe(batched)[:12]):
            self.assertFalse(session.observe().task_graph_complete)
            step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
            self.assertTrue(step.ok,step.error_message)
            if index==9:
                self.assertEqual({entry.request_id for entry in session.observe().route_history},{"r0"})
                self.assertFalse(session.observe().task_graph_complete)
        self.assertTrue(session.observe().task_graph_complete)

    def test_shared_expert_has_full_request_rows_and_independent_whole_weight(self):
        original=source()
        params=dict(original.layer.ffn_params,shared_expert_weight="S",shared_expert_bytes=64)
        layer=replace(original.layer,ffn_params=params,weights=original.layer.weights+
                      (WeightRef("S","S",64,"ssd"),))
        history=replace(original.state_in,weight_copies=original.state_in.weight_copies+
                        (WeightStateRef("S",1,64,("ssd",)),))
        entries=dict(original.cost_table.entries)
        for (kind,device,shape),price in original.cost_table.entries.items():
            if kind=="expert_ffn":entries[("shared_expert_ffn",device,shape)]=price
        invocation=replace(original.invocation,state_in=history)
        compiled=compile_invocation(layer,invocation,replace(original.cost_table,entries=entries))
        transformed=transform_moe_execution(compiled,parse_execution_spec(document(True)))
        region=transformed.physical_boundary["row_execution"]["regions"][-1]
        self.assertEqual(region["expert_id"],"shared")
        self.assertEqual(region["work_items"],[dict(request_id="r0",absolute_position=0),dict(request_id="r0",absolute_position=1)])
        for oid in region["expert_operation_ids"]:
            op=transformed.scenario.workload.operation_by_id[oid]
            if op.semantic_op!="expert_swiglu":self.assertIn("w:moe-small:S",op.inputs)
        self.assertEqual(replay_serial(transformed).status,STATUS_VALID)


if __name__=="__main__":unittest.main()
