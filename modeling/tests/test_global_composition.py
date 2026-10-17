"""Scope-bound execution choices and causal score sidecars across real M2 ports.

Prices are explicit tiny synthetic test prices. These checks validate graph,
data and information contracts, never physical kernel speed or calibration.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from llm_infer_model.m2.attention_execution import parse_attention_execution
from llm_infer_model.m2.expand import load_and_compile
from llm_infer_model.m2.global_execution import compile_global_execution_file, GlobalM2Session
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.m2.routing_scores import parse_routing_scores
from llm_infer_model.tensor.engine import Action, ACTION_ADVANCE, ACTION_COMPUTE, STATUS_VALID
from llm_infer_model.tensor.spec import COPY_READY, OP_DONE, InvalidInput, Unsupported

FIXTURES=Path(__file__).parent / "fixtures/global_execution"


def attention_choice(name, *, device):
    source=load_and_compile(FIXTURES / f"{name}.json")
    shape=next(task.shape for task in source.tasks if task.kind=="attention")
    return parse_attention_execution(dict(
        format_version="dense-attention-execution-0.1",variant="pure_attention",
        provenance="synthetic_execution",invocation_files=[f"{name}.json"],
        frontend_config=deepcopy(source.layer.physical.config),
        batch_contract="independent_requests_not_fused_batch_kernel",
        subbatches=[dict(id="attention-subset",request_ids=["r0"],attention_device=device)],
        entries=[dict(kind=kind,device=target,phase=shape.phase,batch=1,tokens=shape.tokens,
                      visible_history=shape.visible_history,duration_ns=123 if target=="gpu" else 456,
                      workspace_bytes=64)
                 for kind in ("attention_core","attention_output") for target in ("cpu","gpu")],
        transfer_entries=[]))


def dense_bindings():
    return tuple((f"{layer}-{phase}",attention_choice(f"{layer}.{phase}",device="gpu" if phase=="prefill" else "cpu"))
                 for phase in ("prefill","decode") for layer in ("denseL0","denseL1"))


def global_scores(compiled):
    entries=[]
    for scope in compiled.invocation_scopes:
        for _,row in scope.routes:
            # Preserve actual top-k, while declaring every unselected score.
            scores={expert:(.4 if expert in row.expert_ids else .1) for expert in range(4)}
            entries.append(dict(request_id=row.request_id,layer_id=row.layer_id,
                absolute_position=row.absolute_position,
                router_scores=[dict(expert_id=expert,score=value) for expert,value in scores.items()]))
    return dict(format_version="routing-scores-0.1",provenance="synthetic_router_probabilities",entries=entries)


def run_plan(session, compiled, *, placements=None, until=None):
    for action in plan_moe(compiled.aggregate,placements=placements):
        step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
        if not step.ok:raise AssertionError(step.error_message)
        if until is not None and until(session):break


class GlobalCompositionTests(unittest.TestCase):
    def test_attention_choices_cross_layers_and_rounds_with_exact_KV_ports(self):
        compiled=compile_global_execution_file(FIXTURES / "dense.global.json",execution_overrides=dense_bindings())
        session=GlobalM2Session(compiled)
        self.assertEqual({operation.invocation_id for operation in session.observe().operations},{"denseL0-prefill"})
        placements={task.operation_id:(task.device if task.kind in ("attention_core","state_update") else "gpu")
                    for task in compiled.aggregate.tasks}
        run_plan(session,compiled,placements=placements)
        self.assertEqual(session.result().evaluation.status,STATUS_VALID)
        self.assertEqual(len(session.observe().completed_invocations),4)
        self.assertEqual({scope.execution_variant for scope in compiled.invocation_scopes},{"pure_attention"})
        kinds={task.operation_id:task.kind for task in compiled.aggregate.tasks}
        core_events=[event for event in session.result().evaluation.events
                     if event.kind==ACTION_COMPUTE and kinds[event.target_id]=="attention_core"]
        self.assertEqual([event.device for event in core_events],["gpu","gpu","cpu","cpu"])
        operations=session.observe().operations
        core_ops=[operation for operation in operations if operation.kind=="attention_core"]
        self.assertEqual(len({operation.region_id for operation in core_ops}),4)
        self.assertTrue(all(operation.execution_stage=="attention_core" and operation.execution_variant=="pure_attention"
                            for operation in core_ops))
        tensors={tensor.id:tensor for tensor in session.observe().tensors}
        self.assertEqual([tensors[operation.outputs[0]].row_scope for operation in core_ops],
                         [(('r0',0),('r0',1),('r0',2)),(('r0',0),('r0',1),('r0',2)),(('r0',3),),(('r0',3),)])
        for layer in ("denseL0","denseL1"):
            for position in range(4):
                self.assertEqual(session._runtime.state.copy_at(f"kv:{layer}:r0:{position}","dram"),COPY_READY)
        self.assertFalse(any(tensor.id.endswith(":in.hidden") and tensor.id.startswith("g:denseL1")
                             for tensor in compiled.scenario.workload.tensors))
        self.assertFalse(session.persistent_state_document()["resumable_checkpoint"])

    def test_override_scope_and_type_are_strict_and_no_cpu_prefill_is_invented(self):
        bindings=dense_bindings()
        for invalid in ((bindings[0],bindings[0]),(("absent",bindings[0][1]),),((bindings[0][0],None),),
                        ((bindings[0][0],{}),)):
            with self.assertRaises((InvalidInput,Unsupported)):
                compile_global_execution_file(FIXTURES / "dense.global.json",execution_overrides=invalid)
        with self.assertRaisesRegex(Unsupported,"requires decode"):
            compile_global_execution_file(FIXTURES / "dense.global.json",execution_overrides=(
                ("denseL0-prefill",attention_choice("denseL0.prefill",device="cpu")),))

    def test_full_scores_reveal_only_after_each_scoped_router_finishes(self):
        compiled=compile_global_execution_file(FIXTURES / "moe.global.json")
        scores=parse_routing_scores(global_scores(compiled))
        session=GlobalM2Session(compiled,routing_scores=scores)
        self.assertEqual(session.observe().route_history,())
        first_router=compiled.invocation_scopes[0].router_operation_ids[0]
        run_plan(session,compiled,until=lambda runtime:runtime._runtime.state.op(first_router)==OP_DONE)
        history=session.observe().route_history
        self.assertEqual({row.layer_id for row in history},{"moeL0"})
        self.assertEqual({row.absolute_position for row in history},{0,1})
        self.assertTrue(all(row.router_scores_scope=="all_experts" and len(row.router_scores)==4 for row in history))
        complete=GlobalM2Session(compiled,routing_scores=scores)
        run_plan(complete,compiled)
        self.assertEqual(complete.result().evaluation.status,STATUS_VALID)
        self.assertEqual(len(complete.observe().route_history),6)
        self.assertEqual({row.layer_id for row in complete.observe().route_history},{"moeL0","moeL1"})

    def test_future_score_change_has_identical_observations_and_action_prefix(self):
        compiled=compile_global_execution_file(FIXTURES / "moe.global.json")
        original=global_scores(compiled);changed=deepcopy(original)
        affected=next(row for row in changed["entries"] if row["layer_id"]=="moeL1" and row["absolute_position"]==0)
        affected["router_scores"][0]["score"]+=.02
        affected["router_scores"][1]["score"]-=.02
        a=GlobalM2Session(compiled,routing_scores=parse_routing_scores(original))
        b=GlobalM2Session(compiled,routing_scores=parse_routing_scores(changed))
        router=compiled.invocation_scopes[1].router_operation_ids[0]
        for action in plan_moe(compiled.aggregate):
            self.assertEqual(a.observe(),b.observe())
            for session in (a,b):
                step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok,step.error_message)
            if a._runtime.state.op(router)==OP_DONE:break
        self.assertNotEqual(a.observe().route_history,b.observe().route_history)

    def test_missing_contradictory_nonfinite_and_partial_full_scores_fail(self):
        compiled=compile_global_execution_file(FIXTURES / "moe.global.json")
        document=global_scores(compiled)
        partial=deepcopy(document);partial["entries"].pop()
        with self.assertRaisesRegex(InvalidInput,"cover exactly"):
            GlobalM2Session(compiled,routing_scores=parse_routing_scores(partial))
        contradictory=deepcopy(document)
        first=contradictory["entries"][0]["router_scores"]
        first[0]["score"],first[3]["score"]=first[3]["score"],first[0]["score"]
        with self.assertRaisesRegex(InvalidInput,"contradict"):
            GlobalM2Session(compiled,routing_scores=parse_routing_scores(contradictory))
        typed=parse_routing_scores(document)
        typed=(replace(typed[0],scores=((0,float("nan")),(1,.4),(2,.1),(3,.1))),*typed[1:])
        with self.assertRaises(InvalidInput):GlobalM2Session(compiled,routing_scores=typed)
        plain=GlobalM2Session(compiled);run_plan(plain,compiled)
        self.assertTrue(all(not row.router_scores and row.router_scores_scope=="not_reported"
                            for row in plain.observe().route_history))

    def test_fragment_scopes_share_causal_scores_without_reintroducing_parent_weights(self):
        compiled=compile_global_execution_file(FIXTURES / "moe-fragments.global.json")
        session=GlobalM2Session(compiled,routing_scores=parse_routing_scores(global_scores(compiled)))
        self.assertTrue(all(expert.weight_tensor_id is None and len(expert.weight_tensor_ids)==3
                            for expert in session.observe().experts))
        run_plan(session,compiled)
        self.assertEqual(session.result().evaluation.status,STATUS_VALID)
        self.assertEqual(len(session.observe().route_history),6)
        self.assertTrue(all(len(row.router_scores)==4 and row.router_scores_scope=="all_experts"
                            for row in session.observe().route_history))
        self.assertEqual({scope.execution_variant for scope in compiled.invocation_scopes},{"swiglu_staged"})


if __name__=="__main__":unittest.main()
