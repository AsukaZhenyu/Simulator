"""NEO-style pure attention ports, subbatch replay and independent tiny maths."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import unittest

from llm_infer_model.m2.attention_execution import (
    parse_attention_execution,attention_execution_document,compile_attention_subbatches,
)
from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.expand import load_and_compile,compile_invocation
from llm_infer_model.tensor.engine import (
    Action,ACTION_ADVANCE,ACTION_COMPUTE,ACTION_COPY,STATUS_VALID,
)
from llm_infer_model.tensor.spec import COPY_READY,OP_NOT_STARTED,InvalidInput,Unsupported
from llm_infer_model.tensor.environment import EnvironmentEvent

EXAMPLES=Path(__file__).resolve().parents[2]/"mapping/examples/m2"


def sources():
    base=load_and_compile(EXAMPLES/"hybrid-interface-synthetic-decode-1.json")
    result=[]
    for req,location in (("r0","dram"),("r1","vram")):
        request=replace(base.invocation.requests[0],request_id=req)
        history=base.state_in.requests[0]
        blocks=tuple(replace(block,id=block.id.replace(":r0:",f":{req}:"),locations=(location,))
                     for block in history.blocks)
        history=replace(history,request_id=req,blocks=blocks,state_location=location)
        state_in=replace(base.state_in,requests=(history,))
        invocation=replace(base.invocation,id="decode-"+req,requests=(request,),state_in=state_in)
        result.append(compile_invocation(base.layer,invocation,base.cost_table))
    return tuple(result)


def document():
    first=sources()[0]
    return dict(format_version="dense-attention-execution-0.1",variant="pure_attention",
        provenance="synthetic_execution",invocation_files=["r0.json","r1.json"],
        frontend_config=deepcopy(first.layer.physical.config),
        batch_contract="independent_requests_not_fused_batch_kernel",
        subbatches=[dict(id="cpu-decode",request_ids=["r0"],attention_device="cpu"),
                    dict(id="gpu-decode",request_ids=["r1"],attention_device="gpu")],
        entries=[dict(kind=kind,device=device,phase="decode",batch=1,tokens=1,visible_history=3,
                      duration_ns=(10_000 if device=="cpu" else 2000) if kind=="attention_core" else 300,
                      workspace_bytes=64 if kind=="attention_core" else 16)
                 for kind in ("attention_core","attention_output") for device in ("cpu","gpu")],
        transfer_entries=[])


def replay(session):
    """Explicit test driver, not the production policy or hidden future oracle."""
    for _ in range(2000):
        obs=session.observe()
        if obs.complete:return session.result()
        tensors={tensor.id:tensor for tensor in obs.tensors}
        busy={running.resource for running in obs.running}
        selected=None
        for op in obs.operations:
            if op.status!=OP_NOT_STARTED:continue
            if any(not any(status==COPY_READY for _,status in tensors[tid].copies) for tid in op.inputs):continue
            # The model's declared core binding is exact; all other stages are
            # explicitly placed GPU in this test schedule.
            device=op.prices[0].device if op.kind=="attention_core" else "gpu"
            location="dram" if device=="cpu" else "vram"
            missing=[tid for tid in op.inputs if tensors[tid].status_at(location)!=COPY_READY]
            if missing:
                if "pcie_copy" in busy:continue
                for tid in missing:
                    src="vram" if location=="dram" else "dram"
                    if tensors[tid].status_at(src)==COPY_READY:
                        selected=Action(ACTION_COPY,tid,src=src,dst=location);break
                if selected:break
            elif ("cpu_compute" if device=="cpu" else "gpu_compute") not in busy:
                selected=Action(ACTION_COMPUTE,op.id,device=device);break
        if selected is None:
            for tid,destination in obs.deliveries:
                tensor=tensors[tid]
                if tensor.status_at(destination)==COPY_READY:continue
                if "pcie_copy" in busy:break
                src="vram" if destination=="dram" else "dram"
                if tensor.status_at(src)==COPY_READY:
                    selected=Action(ACTION_COPY,tid,src=src,dst=destination);break
        step=session.advance() if selected is None else session.submit(selected)
        if not step.ok:raise AssertionError(step.error_message)
    raise AssertionError("test schedule exceeded explicit action bound")


class AttentionExecutionTests(unittest.TestCase):
    def test_pure_core_excludes_output_weight_gate_and_hidden_residual(self):
        compiled=compile_attention_subbatches(sources(),parse_attention_execution(document()))
        by_id=compiled.scenario.workload.operation_by_id
        for region in compiled.physical_boundary["attention_execution"]["regions"]:
            core=by_id[region["core_operation_id"]]
            output=by_id[region["output_operation_id"]]
            self.assertEqual(core.inputs,tuple(region["qkv_ports"][port] for port in ("q","k","v"))+
                             tuple(region["historical_kv_ids"]))
            self.assertEqual(core.extra_outputs,tuple(region["new_kv_ids"]))
            self.assertNotIn("w:hybrid-interface-synthetic:W_output",core.inputs)
            self.assertIn("w:hybrid-interface-synthetic:W_output",output.inputs)
            self.assertIn(region["qkv_ports"]["gate"],output.inputs)
            self.assertIn(f"t:hybrid-interface-synthetic:{region['request_id']}:in.hidden",output.inputs)
            self.assertEqual(compiled.scenario.workload.tensor_by_id[core.output].storage_bytes,64)
            self.assertEqual(len(region["historical_kv_ids"]),3)
        self.assertEqual(compiled.invocation.batch,2)
        self.assertTrue(all(task.shape.batch==1 for task in compiled.tasks))

    def test_two_subbatches_cpu_gpu_interleave_and_keep_each_request_kv(self):
        compiled=compile_attention_subbatches(sources(),parse_attention_execution(document()))
        session=CausalM2Session(compiled)
        self.assertTrue(session.observe().task_graph_complete)
        self.assertEqual(session.observe().experts,())
        self.assertEqual(session.observe().route_history,())
        run=replay(session)
        self.assertEqual(run.evaluation.status,STATUS_VALID)
        events=[event for event in run.evaluation.events if event.kind==ACTION_COMPUTE]
        kinds={task.operation_id:task.kind for task in compiled.tasks}
        cores=[event for event in events if kinds[event.target_id]=="attention_core"]
        self.assertEqual({event.device for event in cores},{"cpu","gpu"})
        cpu=next(event for event in cores if event.device=="cpu")
        self.assertTrue(any(event.device=="gpu" and max(cpu.start_ns,event.start_ns)<min(cpu.end_ns,event.end_ns)
                            for event in events))
        output=session.state_out()
        self.assertEqual({state.request_id for state in output.requests},{"r0","r1"})
        for state in output.requests:
            self.assertEqual(state.next_position,4)
            self.assertEqual(len(state.blocks),4)
            self.assertTrue(all(state.state_location in block.locations for block in state.blocks))

    def test_unchanged_projection_ffn_exact_costs_and_independent_split_price(self):
        originals=sources();before=deepcopy(originals[0].scenario_document)
        compiled=compile_attention_subbatches(originals,parse_attention_execution(document()))
        for original in originals:
            for task in original.tasks:
                if task.kind in ("projection","dense_ffn"):
                    for device in ("cpu","gpu"):
                        self.assertEqual(compiled.scenario.costs.compute_cost(task.operation_id,device),
                                         original.scenario.costs.compute_cost(task.operation_id,device))
                else:
                    self.assertNotIn(task.operation_id,compiled.scenario_document["costs"]["compute"])
        self.assertEqual(originals[0].scenario_document,before)
        modified=document()
        for price in modified["entries"]:
            if price["kind"]=="attention_output":price["duration_ns"]+=1234
        changed=compile_attention_subbatches(originals,parse_attention_execution(modified))
        for task in changed.tasks:
            if task.kind=="attention_output":
                self.assertEqual(changed.scenario.costs.compute_cost(task.operation_id,"gpu").duration_ns,1534)

    def test_roundtrip_strict_missing_prices_geometry_and_request_coverage(self):
        raw=document();self.assertEqual(attention_execution_document(parse_attention_execution(raw)),raw)
        bad=deepcopy(raw);bad["subbatches"][1]["request_ids"]=["r0"]
        with self.assertRaises(InvalidInput):parse_attention_execution(bad)
        bad=deepcopy(raw);bad["subbatches"][1]["request_ids"]=["unknown"]
        with self.assertRaises(InvalidInput):compile_attention_subbatches(sources(),parse_attention_execution(bad))
        bad=deepcopy(raw);bad["entries"]=bad["entries"][1:]
        with self.assertRaisesRegex(Unsupported,"missing exact independent"):
            compile_attention_subbatches(sources(),parse_attention_execution(bad))
        bad=deepcopy(raw);bad["entries"][0]["workspace_bytes"]=None
        with self.assertRaises(InvalidInput):parse_attention_execution(bad)
        bad=deepcopy(raw);bad["frontend_config"]["rope_base"]=9000.0
        with self.assertRaises(InvalidInput):compile_attention_subbatches(sources(),parse_attention_execution(bad))
        bad=deepcopy(raw);bad["batch_contract"]="fused_batch"
        with self.assertRaises(Unsupported):parse_attention_execution(bad)
        bad=deepcopy(raw);bad["extra"]=1
        with self.assertRaises(InvalidInput):parse_attention_execution(bad)

    def test_opaque_or_partial_frontend_refused(self):
        opaque=load_and_compile(EXAMPLES/"fcd.prefill.json") if (EXAMPLES/"fcd.prefill.json").exists() else sources()[0]
        opaque=replace(opaque,layer=replace(opaque.layer,format_version="m2-layer-0.1",physical=None))
        raw=document();raw["invocation_files"]=["opaque.json"];raw["subbatches"]=raw["subbatches"][:1]
        with self.assertRaisesRegex(Unsupported,"complete materialized"):
            compile_attention_subbatches((opaque,),parse_attention_execution(raw))

    def test_bundle_rejects_mixed_phase_and_cpu_prefill(self):
        decode=sources()[0]
        prefill=load_and_compile(EXAMPLES/"hybrid-interface-synthetic-prefill-3.json")
        prefill=replace(prefill,invocation=replace(prefill.invocation,requests=(replace(prefill.invocation.requests[0],request_id="r1"),)))
        with self.assertRaisesRegex(Unsupported,"single phase"):
            compile_attention_subbatches((decode,prefill),parse_attention_execution(document()))
        raw=document();raw["invocation_files"]=["prefill.json"];raw["subbatches"]=raw["subbatches"][:1]
        with self.assertRaisesRegex(Unsupported,"requires decode"):
            compile_attention_subbatches((load_and_compile(EXAMPLES/"hybrid-interface-synthetic-prefill-3.json"),),
                                        parse_attention_execution(raw))

    def test_new_kv_dependency_capacity_and_export_contract(self):
        compiled=compile_attention_subbatches(sources(),parse_attention_execution(document()))
        session=CausalM2Session(compiled)
        region=compiled.physical_boundary["attention_execution"]["regions"][0]
        self.assertFalse(session.submit(Action(ACTION_COMPUTE,region["core_operation_id"],device="cpu")).ok)
        session=CausalM2Session(compiled)
        graph=session.execution_document()
        self.assertEqual(graph["execution_variant"],"pure_attention")
        self.assertIsNone(graph["expert_variant"])
        self.assertEqual(graph["boundary"]["attention_execution"]["kv_bytes_per_token"],64)
        self.assertEqual(graph["boundary"]["attention_execution"]["gqa_head_mapping"],"query_head // (heads // kv_heads)")
        first=next(op for op in session.observe().operations if op.kind=="attention_core")
        self.assertEqual(first.region_id,"cpu-decode")
        self.assertEqual(first.prices[0].device,"cpu")
        context=next(t for t in session.observe().tensors if t.id==first.outputs[0])
        self.assertEqual(context.row_scope,(("r0",3),))

    def test_core_reserves_context_new_kv_and_workspace_and_rejects_one_byte_short(self):
        compiled=compile_attention_subbatches(sources(),parse_attention_execution(document()))
        baseline=replay(CausalM2Session(compiled))
        cpu=next(event for event in baseline.evaluation.events
                 if event.kind==ACTION_COMPUTE and event.device=="cpu")
        index=cpu.action_index
        before=baseline.evaluation.states[index].used_dram_bytes
        after=baseline.evaluation.states[index+1].used_dram_bytes
        self.assertEqual(after-before,64+64+64)  # context + appended KV + scratch
        available_limit=before+191
        external=compiled.scenario.architecture.dram_capacity_bytes-available_limit
        session=CausalM2Session(compiled,environment_events=(EnvironmentEvent(0,dram_external_bytes=external),))
        for action in baseline.evaluation.actions[:index]:
            step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
            self.assertTrue(step.ok,step.error_message)
        refused=session.submit(baseline.evaluation.actions[index])
        self.assertFalse(refused.ok)
        self.assertIn("capacity",refused.error_message)

    def test_tiny_independent_scalar_attention_matches_split_matrix_ports(self):
        # Independent arithmetic check for the declared split. Inputs are
        # explicitly post-normal-RoPE Q/K; neither a GGML/CUDA kernel nor its
        # launch/time profile is executed by this test.
        def rotate(pair,position):
            c,s=math.cos(position),math.sin(position)
            return (pair[0]*c-pair[1]*s,pair[0]*s+pair[1]*c)
        q=[rotate((0.31,-0.27),2),rotate((-0.41,0.17),2)]
        k=[[rotate((0.11+0.09*p,-0.23+0.03*p),p)] for p in range(3)]
        v=[[(0.19+0.07*p,-0.16-0.04*p)] for p in range(3)]
        gate=[0.23,-0.47,0.13,0.61];residual=[0.37,-0.29]
        weight=[[0.3,-0.2],[-0.4,0.5],[0.7,0.1],[-0.6,0.8]]
        # Split port path: materialize QK scores, normalized row, then PV and
        # context, before performing gate/O/residual as a distinct stage.
        context=[]
        for query in q:
            scores=[sum(a*b for a,b in zip(query,key[0]))/math.sqrt(2) for key in k]
            e=[math.exp(score-max(scores)) for score in scores];prob=[x/sum(e) for x in e]
            context.extend(sum(prob[p]*v[p][0][dim] for p in range(3)) for dim in range(2))
        split=[residual[out]+sum(context[i]/(1+math.exp(-gate[i]))*weight[i][out]
                                  for i in range(4)) for out in range(2)]
        # Unsplit scalar expression sums directly over heads, keys and dims.
        direct=[]
        for out in range(2):
            total=residual[out]
            for head in range(2):
                logits=[(q[head][0]*key[0][0]+q[head][1]*key[0][1])/math.sqrt(2) for key in k]
                denominator=sum(math.exp(logit-max(logits)) for logit in logits)
                for p,logit in enumerate(logits):
                    probability=math.exp(logit-max(logits))/denominator
                    for dim in range(2):
                        index=head*2+dim
                        total+=probability*v[p][0][dim]*weight[index][out]/(1+math.exp(-gate[index]))
            direct.append(total)
        for actual,expected in zip(split,direct):self.assertAlmostEqual(actual,expected,places=13)
        self.assertEqual(len(context),4)  # two query heads share one KV head


if __name__=="__main__":unittest.main()
