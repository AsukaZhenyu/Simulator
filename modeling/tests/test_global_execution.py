"""Continuous layers/rounds, private routes, exact KV seams and source binding."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from llm_infer_model.m2.global_execution import (compile_global_execution_file,GlobalM2Session,
                                               verify_global_sources)
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.tensor.engine import ACTION_ADVANCE,Action,ACTION_COMPUTE,STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput,Unsupported,COPY_READY

FIXTURES=Path(__file__).parent/"fixtures/global_execution"
SCRATCH=Path(__file__).parent/"_tmp_global_execution"


def execute(session,aggregate,*,until=None):
    for index,action in enumerate(plan_moe(aggregate)):
        step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
        if not step.ok:raise AssertionError(f"action {index}: {step.error_message}")
        if until is not None and until(session):break


class GlobalExecutionTests(unittest.TestCase):
    def copied(self):
        SCRATCH.mkdir(exist_ok=True)
        holder=tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(holder.cleanup)
        destination=Path(holder.name)
        for path in FIXTURES.glob("*.json"):shutil.copyfile(path,destination/path.name)
        return destination

    def mutate(self,path,callback):
        doc=json.loads(path.read_text(encoding="utf-8"));callback(doc)
        path.write_text(json.dumps(doc)+"\n",encoding="utf-8",newline="\n")

    def test_dense_two_layers_P3_D1_runs_one_clock_and_exact_KV_versions(self):
        compiled=compile_global_execution_file(FIXTURES/"dense.global.json")
        session=GlobalM2Session(compiled)
        before=session.observe()
        self.assertEqual({op.invocation_id for op in before.operations},{"denseL0-prefill"})
        self.assertEqual(len(before.operations),3)
        self.assertFalse(before.task_graph_complete)
        execute(session,compiled.aggregate)
        result=session.result().evaluation
        self.assertEqual(result.status,STATUS_VALID)
        self.assertEqual(result.makespan_ns,44_007) # Exact leaf costs + 7 ns trace gate.
        self.assertEqual(session.observe().t_ns,result.makespan_ns)
        self.assertEqual(len(session.observe().completed_invocations),4)
        inventory=session.persistent_state_document()
        self.assertEqual(len(inventory["kv_ids"]),8)
        for layer in ("denseL0","denseL1"):
            for position in range(4):
                key=f"kv:{layer}:r0:{position}"
                self.assertEqual(session._runtime.state.copy_at(key,"dram"),COPY_READY)
            # The old standalone decode fixture claims both copies; the global
            # runtime does not invent its unexecuted VRAM copies.
            self.assertNotEqual(session._runtime.state.copy_at(f"kv:{layer}:r0:0","vram"),COPY_READY)
        weights=[t for t in compiled.scenario.workload.tensors if t.role=="weight"]
        self.assertEqual(len(weights),len({t.id for t in weights}))
        self.assertEqual({op.layer_id for op in session.observe().operations},{"denseL0","denseL1"})
        self.assertFalse(inventory["resumable_checkpoint"])

    def test_layer_hidden_ports_and_token_gate_are_real_dependency_edges(self):
        compiled=compile_global_execution_file(FIXTURES/"dense.global.json")
        scopes=compiled.invocation_scopes
        self.assertEqual(scopes[1].incoming_hidden,scopes[0].hidden_outputs)
        self.assertEqual(scopes[3].incoming_hidden,scopes[2].hidden_outputs)
        gate=compiled.scenario.workload.operation_by_id[compiled.gate_operations[0][0]]
        self.assertEqual(gate.inputs,scopes[1].hidden_outputs)
        self.assertEqual(gate.outputs,scopes[2].incoming_hidden)
        session=GlobalM2Session(compiled)
        execute(session,compiled.aggregate,until=lambda s:len(s.observe().completed_invocations)==2)
        obs=session.observe()
        self.assertIn(gate.id,{op.id for op in obs.operations})
        self.assertFalse(any(op.invocation_id=="denseL0-decode" and op.kind!="state_update" for op in obs.operations))
        finalhidden=scopes[1].hidden_outputs[0]
        self.assertNotIn(finalhidden,{tid for tid,_ in obs.release_candidates})
        self.assertNotIn(scopes[0].hidden_outputs[0],{t.id for t in obs.tensors if t.goal_location})
        self.assertIn(scopes[0].hidden_outputs[0],{tid for tid,_ in obs.release_candidates})

    def test_moe_future_routing_has_prefix_noninterference_and_true_revelations(self):
        copied=self.copied()
        self.mutate(copied/"moeL1.prefill.json",lambda doc:doc["route_trace"][0].update(expert_ids=[2,3]))
        original=compile_global_execution_file(FIXTURES/"moe.global.json")
        changed=compile_global_execution_file(copied/"moe.global.json")
        a,b=GlobalM2Session(original),GlobalM2Session(changed)
        self.assertEqual(a.observe(),b.observe())
        stop=original.invocation_scopes[1].router_operation_ids[0]
        for action in plan_moe(original.aggregate):
            self.assertEqual(a.observe(),b.observe())
            for session in (a,b):
                step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok,step.error_message)
            if a._runtime.state.op(stop)=="DONE":break
        self.assertNotEqual(a.observe().route_history,b.observe().route_history)
        self.assertTrue(all(row.layer_id in ("moeL0","moeL1") for row in a.observe().route_history))
        self.assertEqual(len(a.observe().experts),8)

    def test_moe_round_cache_is_retained_and_KV_never_release_candidate(self):
        compiled=compile_global_execution_file(FIXTURES/"moe.global.json")
        session=GlobalM2Session(compiled)
        execute(session,compiled.aggregate)
        result=session.result().evaluation
        self.assertEqual(result.status,STATUS_VALID)
        self.assertEqual(result.makespan_ns,98_000_007)
        self.assertEqual(result.ssd_read_bytes,2*3*64) # Decode E0/E1 reuse prior DRAM copies.
        ids={tid for tid,_ in session.observe().release_candidates}
        self.assertFalse(any(tid.startswith("kv:") or tid.startswith("w:") for tid in ids))
        self.assertEqual(len(session.persistent_state_document()["kv_ids"]),6)

    def test_future_compute_submission_is_rejected_before_scope_opens(self):
        compiled=compile_global_execution_file(FIXTURES/"dense.global.json")
        session=GlobalM2Session(compiled)
        oid=compiled.invocation_scopes[1].operation_ids[0]
        self.assertFalse(session.submit(Action(ACTION_COMPUTE,oid,device="cpu")).ok)

    def test_exact_history_prices_and_KV_state_versions_are_not_guessed(self):
        for change in (lambda doc:doc["state_in"]["requests"][0].update(version=99),
                       lambda doc:doc["state_in"]["requests"][0]["blocks"].pop()):
            copied=self.copied();self.mutate(copied/"denseL0.decode.json",change)
            with self.assertRaises((InvalidInput,Unsupported)):
                compile_global_execution_file(copied/"dense.global.json")
        copied=self.copied()
        self.mutate(copied/"dense.costs.json",lambda doc:doc.update(entries=[e for e in doc["entries"] if not
                    (e["phase"]=="decode" and e["visible_history"]==3)]))
        with self.assertRaises((InvalidInput,Unsupported)):
            compile_global_execution_file(copied/"dense.global.json")

    def test_manifest_unknowns_missing_gate_cost_and_changed_sources_fail(self):
        for mutation in (lambda d:d.update(unknown=True),lambda d:d["rounds"][1].update(gate_cost=None),
                         lambda d:d["rounds"][1]["gate_cost"].update(duration_ns=0)):
            copied=self.copied();self.mutate(copied/"dense.global.json",mutation)
            with self.assertRaises((InvalidInput,Unsupported)):
                compile_global_execution_file(copied/"dense.global.json")
        copied=self.copied();compiled=compile_global_execution_file(copied/"dense.global.json")
        self.mutate(copied/"denseL0.layer.json",lambda d:d.update(comment="changed source"))
        with self.assertRaisesRegex(InvalidInput,"changed"):
            verify_global_sources(compiled)
        with self.assertRaisesRegex(InvalidInput,"changed"):
            GlobalM2Session(compiled)

    def test_weight_versions_and_shared_hardware_must_remain_stable(self):
        for mutation in (lambda d:d["hardware"].update(dram_capacity_bytes=8192),
                         lambda d:d["state_in"]["weight_copies"][2].update(version=2)):
            copied=self.copied();self.mutate(copied/"moeL0.decode.json",mutation)
            with self.assertRaises((InvalidInput,Unsupported)):
                compile_global_execution_file(copied/"moe.global.json")

    def test_partial_inventory_preserves_inflight_reserved_state_and_opaque_versions(self):
        from llm_infer_model.tensor.engine import ACTION_READ_SSD
        compiled=compile_global_execution_file(FIXTURES/"moe.global.json")
        session=GlobalM2Session(compiled)
        self.assertTrue(session.submit(Action(ACTION_READ_SSD,"w:moeL0:E0")).ok)
        inventory=session.persistent_state_document()
        self.assertEqual(inventory["t_ns"],0)
        self.assertTrue(inventory["runtime_state"]["running"])
        self.assertEqual(dict(inventory["runtime_state"]["dram_status"])["w:moeL0:E0"],"RESERVED_COPY")
        self.assertEqual(inventory["declared_parent_versions"]["w:moeL0:E0"],1)
        self.assertFalse(inventory["resumable_checkpoint"])

    def test_materialized_fragments_remain_distinct_across_layers_and_rounds(self):
        compiled=compile_global_execution_file(FIXTURES/"moe-fragments.global.json")
        session=GlobalM2Session(compiled)
        initial=session.observe()
        self.assertEqual(len(initial.experts),8)
        self.assertTrue(all(expert.weight_tensor_id is None and len(expert.weight_tensor_ids)==3
                            for expert in initial.experts))
        execute(session,compiled.aggregate)
        self.assertEqual(session.result().evaluation.status,STATUS_VALID)
        self.assertEqual(session.result().evaluation.ssd_read_bytes,2*3*384)
        tensors=compiled.scenario.workload.tensor_by_id
        self.assertNotIn("w:fragmentL0:E0",tensors)
        self.assertIn("w:fragmentL0:E0:v1:gate",tensors)
        self.assertEqual(len(compiled.aggregate.physical_boundary["row_execution"]["subweights"]),8)

    def delivery(self,location):
        copied=self.copied()
        self.mutate(copied/"moeL1.decode.json",lambda doc:doc["requests"][0].update(output_location=location))
        compiled=compile_global_execution_file(copied/"moe.global.json")
        terminal=compiled.invocation_scopes[-1].hidden_outputs[0]
        self.assertEqual(compiled.scenario.goal.output_locations[terminal],location)
        session=GlobalM2Session(compiled);execute(session,compiled.aggregate)
        result=session.result().evaluation
        self.assertEqual(result.status,STATUS_VALID,result.reason)
        self.assertEqual(session._runtime.state.copy_at(terminal,location),COPY_READY)
        transfers=[event for event in result.events if event.target_id==terminal and event.dst==location]
        self.assertEqual(len(transfers),1)
        self.assertEqual(transfers[0].bytes_transferred,16)
        return result

    def test_final_hidden_is_delivered_to_declared_vram(self):
        result=self.delivery("vram")
        self.assertEqual(result.h2d_bytes,16)
        self.assertEqual(result.ssd_write_bytes,0)

    def test_final_hidden_is_written_to_declared_ssd(self):
        result=self.delivery("ssd")
        self.assertEqual(result.ssd_write_bytes,16)
        self.assertEqual(result.h2d_bytes,0)


if __name__=="__main__":unittest.main()
