"""Independent review of source binding, immutable context and replay causality."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.tensor.engine import ACTION_ADVANCE, STATUS_VALID
from llm_infer_model.tensor.spec import InvalidInput, SpecError
from tensor_mapping.unified_strategy import (
    compile_unified_strategy_file,create_session,run_unified_strategy,checkpoint_document,
)
from .test_composition_policy import inline_policy

ROOT=Path(__file__).resolve().parents[2]
FIXTURES=ROOT / "modeling/tests/fixtures/global_execution"
SCRATCH=ROOT / "mapping/outputs/composition-review-tmp"


def write(path,document):
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")


def ir_document(model_file, *, scores=None, mrs=False):
    policy=inline_policy(cache="mrs" if mrs else "lru",unit="expert_group")
    policy["placements"].update(projection="gpu",attention="gpu",router="gpu",expert_ffn="cpu",
                                combine="gpu",attention_core="execution_binding",attention_output="gpu")
    return dict(format_version="unified-strategy-0.2",id="independent-composition-review",
        model=dict(kind="global_m2",file=str(model_file)),execution=[],
        policy={key:policy[key] for key in ("placements","request_placements","experts","release")},
        cache=policy["cache"],prefetch=policy["prefetch"],transport=policy["transport"],
        services_file=None,environment_file="environment.json",history_file=None,
        routing_scores_file=scores,max_decisions=1000)


def scores_for_folder(folder):
    document=dict(format_version="routing-scores-0.1",provenance="synthetic_router_probabilities",entries=[])
    for layer in ("moeL0","moeL1"):
        for phase in ("prefill","decode"):
            invocation=json.loads((folder / f"{layer}.{phase}.json").read_text(encoding="utf-8"))
            for trace in invocation["route_trace"]:
                document["entries"].append(dict(request_id=trace["request_id"],layer_id=trace["layer_id"],
                    absolute_position=trace["absolute_position"],router_scores=[dict(expert_id=expert,
                        score=.4 if expert in trace["expert_ids"] else .1) for expert in range(4)]))
    return document


class CompositionIRReviewTests(unittest.TestCase):
    def folder(self):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        holder=tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(holder.cleanup)
        folder=Path(holder.name)
        for path in FIXTURES.glob("*.json"):shutil.copyfile(path,folder/path.name)
        write(folder/"environment.json",{"format_version":"environment-trace-0.1","events":[]})
        return folder

    def strategy(self,folder, *, model="moe.global.json", mrs=False):
        if mrs:write(folder/"scores.json",scores_for_folder(folder))
        write(folder/"strategy.json",ir_document(folder/model,scores="scores.json" if mrs else None,mrs=mrs))
        return compile_unified_strategy_file(folder/"strategy.json")

    def test_catalogue_does_not_depend_on_future_routes_and_is_immutable(self):
        a_folder,b_folder=self.folder(),self.folder()
        changed=json.loads((b_folder/"moeL1.prefill.json").read_text(encoding="utf-8"))
        changed["route_trace"][0]["expert_ids"]=[2,3]
        write(b_folder/"moeL1.prefill.json",changed)
        a_model,b_model=self.strategy(a_folder),self.strategy(b_folder)
        a,b=create_session(a_model),create_session(b_model)
        self.assertEqual(a.observe(),b.observe())
        observation=a.observe()
        self.assertEqual(observation.layer_order,("moeL0","moeL1"))
        self.assertEqual(observation.active_invocations,(("moeL0-prefill","moeL0"),))
        self.assertEqual(len([entry for entry in observation.weight_catalogue if entry[3] is not None]),8)
        with self.assertRaises(FrozenInstanceError):observation.layer_order=("bad",)
        target=a_model.model.invocation_scopes[1].router_operation_ids[0]
        for action in plan_moe(a_model.model.aggregate):
            self.assertEqual(a.observe(),b.observe())
            for session in (a,b):
                step=session.advance() if action.kind==ACTION_ADVANCE else session.submit(action)
                self.assertTrue(step.ok,step.error_message)
            if a._runtime.state.op(target)=="DONE":break
        self.assertNotEqual(a.observe().route_history,b.observe().route_history)
        self.assertEqual(a.observe().weight_catalogue,b.observe().weight_catalogue)

    def test_inline_override_is_bound_to_IR_and_unused_legacy_sidecar_is_not_read(self):
        folder=self.folder()
        model=json.loads((folder/"moe-fragments.global.json").read_text(encoding="utf-8"))
        model["rounds"][0]["invocations"][0]["execution_file"]="unused-missing-sidecar.json"
        write(folder/"moe-fragments.global.json",model)
        inline=json.loads((folder/"fragmentL0.execution.json").read_text(encoding="utf-8"))
        document=ir_document(folder/"moe-fragments.global.json")
        document["execution"]=[dict(scope_id="fragmentL0-prefill",spec=inline)]
        write(folder/"strategy.json",document)
        compiled=compile_unified_strategy_file(folder/"strategy.json")
        sources={Path(entry["path"]).name:entry["sha256"] for entry in compiled.sources}
        self.assertNotIn("unused-missing-sidecar.json",sources)
        self.assertEqual(sources["strategy.json"],hashlib.sha256((folder/"strategy.json").read_bytes()).hexdigest())
        document["execution"][0]["spec"]["entries"][0]["duration_ns"]+=1
        write(folder/"strategy.json",document)
        with self.assertRaisesRegex(InvalidInput,"changed"):
            run_unified_strategy(compiled,max_decisions=1)

    def test_mrs_scores_and_checkpoint_replay_match_uninterrupted_commands(self):
        folder=self.folder();compiled=self.strategy(folder,mrs=True)
        start=run_unified_strategy(compiled,max_decisions=1)
        self.assertEqual(start[1].observe().route_history,())
        self.assertTrue(start[1].observe().running)
        checkpoint=checkpoint_document(compiled,start)
        full=run_unified_strategy(compiled)
        restored=run_unified_strategy(compiled,checkpoint=checkpoint)
        self.assertEqual(full[0].evaluation.status,STATUS_VALID,full[0].compilation_error)
        self.assertEqual(restored[0].evaluation.status,STATUS_VALID,restored[0].compilation_error)
        self.assertEqual(full[0].evaluation.actions,restored[0].evaluation.actions)
        self.assertEqual(full[1].observe(),restored[1].observe())
        self.assertEqual(full[2].history_document(),restored[2].history_document())
        self.assertEqual(len(restored[1].observe().route_history),6)
        self.assertTrue(all(row.router_scores_scope=="all_experts" for row in restored[1].observe().route_history))
        changed=deepcopy(checkpoint);changed["commands"][0]["decision"]["dst"]="dram"
        with self.assertRaisesRegex(InvalidInput,"differs"):
            run_unified_strategy(compiled,checkpoint=changed)

    def test_malformed_global_model_is_reported_as_contract_error(self):
        for mutation in (lambda document:document.update(rounds=None),
                         lambda document:document["rounds"][0].pop("invocations"),
                         lambda document:document["rounds"][0]["invocations"][0].pop("id")):
            folder=self.folder()
            model=json.loads((folder/"moe.global.json").read_text(encoding="utf-8"));mutation(model)
            write(folder/"moe.global.json",model)
            with self.subTest(model=model),self.assertRaises(SpecError):self.strategy(folder)

    def test_catalogue_layer_ownership_uses_declarations_for_overlapping_layer_names(self):
        folder=self.folder()
        def rename(value):
            if isinstance(value,dict):return {key:rename(item) for key,item in value.items()}
            if isinstance(value,list):return [rename(item) for item in value]
            if isinstance(value,str):return value.replace("moeL0","L").replace("moeL1","L:x")
            return value
        # Existing M2 allows colon in a layer ID. Model fields and KV IDs are
        # renamed, while file references retain their real fixture filenames.
        for layer in ("moeL0","moeL1"):
            path=folder/f"{layer}.layer.json"
            write(path,rename(json.loads(path.read_text(encoding="utf-8"))))
            for phase in ("prefill","decode"):
                path=folder/f"{layer}.{phase}.json"
                original=json.loads(path.read_text(encoding="utf-8"));changed=rename(original)
                changed["layer_file"]=original["layer_file"]
                write(path,changed)
        compiled=self.strategy(folder)
        catalogue=create_session(compiled).observe().weight_catalogue
        self.assertEqual({entry[1] for entry in catalogue},{"L","L:x"})
        self.assertEqual(next(entry[1] for entry in catalogue if entry[0]=="w:L:x:W_proj"),"L:x")

    def test_single_pure_attention_inline_binding_and_replay_are_faithful(self):
        folder=self.folder()
        model=folder/"denseL0.decode.json"
        config=json.loads((folder/"denseL0.layer.json").read_text(encoding="utf-8"))["physical"]["config"]
        execution=dict(format_version="dense-attention-execution-0.1",variant="pure_attention",
            provenance="synthetic_execution",invocation_files=[str(model)],frontend_config=config,
            batch_contract="independent_requests_not_fused_batch_kernel",
            subbatches=[dict(id="explicit-cpu-core",request_ids=["r0"],attention_device="cpu")],
            entries=[dict(kind=kind,device=device,phase="decode",batch=1,tokens=1,visible_history=3,
                          duration_ns=111 if device=="cpu" else 222,workspace_bytes=64)
                     for kind in ("attention_core","attention_output") for device in ("cpu","gpu")],
            transfer_entries=[])
        document=ir_document(model)
        document["model"]["kind"]="m2"
        document["execution"]=[dict(scope_id="single",spec=execution)]
        document["policy"]["placements"]["dense_ffn"]="gpu"
        write(folder/"strategy.json",document)
        compiled=compile_unified_strategy_file(folder/"strategy.json")
        session=create_session(compiled)
        core=next(operation for operation in session.observe().operations if operation.kind=="attention_core")
        self.assertEqual(tuple(price.device for price in core.prices),("cpu",))
        self.assertEqual(core.execution_variant,"pure_attention")
        partial=run_unified_strategy(compiled,max_decisions=3)
        checkpoint=checkpoint_document(compiled,partial)
        full=run_unified_strategy(compiled)
        restored=run_unified_strategy(compiled,checkpoint=checkpoint)
        self.assertEqual(full[0].evaluation.status,STATUS_VALID,full[0].compilation_error)
        self.assertEqual(full[0].evaluation,restored[0].evaluation)
        self.assertEqual(full[1].persistent_state_document(),restored[1].persistent_state_document())
        compute=next(row for row in full[0].decisions if row.action.kind=="COMPUTE" and row.action.target_id==core.id)
        self.assertEqual(compute.action.device,"cpu")
        invalid=deepcopy(document)
        invalid["execution"][0]["spec"]["invocation_files"]=[str(folder/"denseL1.decode.json")]
        write(folder/"wrong-source.json",invalid)
        with self.assertRaisesRegex(InvalidInput,"exact bound model scope"):
            compile_unified_strategy_file(folder/"wrong-source.json")



if __name__=="__main__":unittest.main()
