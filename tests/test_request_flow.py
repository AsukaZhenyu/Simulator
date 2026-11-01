"""Public portable request-flow use, source binding and explicit limitations."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_simulator import (
    AnalyticalCosts, Cache, InvalidInput, ModelPlan, PhaseHandoff, Request,
    RequestFlow, RequestPipeline, Scheduling, Simulator, Unsupported,
)
from examples.library.request_pipeline import make_simulator, configure


class RequestFlowLibraryTests(unittest.TestCase):
    def test_dense_and_moe_compile_through_public_ir_and_finish(self):
        for kind in ("dense","moe"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                simulator=make_simulator(kind)
                schedule=Scheduling(default_device="gpu",release="last_use",
                    expert_rule="current_cost" if kind=="moe" else "fixed",
                    expert_order="token_count_desc" if kind=="moe" else "graph_order",
                    cache=Cache(rule="lru",budget_bytes=65536))
                with simulator.compile(scheduling=schedule,configure=lambda p:configure(p,kind),directory=folder) as compiled:
                    ir=compiled.ir_document()
                    self.assertEqual(ir["request_pipeline"]["ready_rule"],"decode_first")
                    self.assertEqual(ir["phase_handoffs"][0]["destination"],"dram")
                    self.assertEqual(ir["contracts"]["resource_ledger"],"single_modeling_runtime")
                    result=compiled.run()
                    self.assertTrue(result.success,result.summary())
                    self.assertFalse(result.summary()["hardware_measurement"])
                    self.assertIn("independent",compiled.metadata["batch_accounting"])

    def test_request_flow_inspection_does_not_open_missing_prices(self):
        simulator=make_simulator()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            inputs=simulator.cost_model.build_inputs(root,simulator.model,simulator.hardware,simulator.workload)
            for path in root.glob("*.json"):
                doc=json.loads(path.read_text(encoding="utf-8"))
                if isinstance(doc,dict) and "costs_file" in doc:
                    doc["costs_file"]="no-measurement.json"
                    path.write_text(json.dumps(doc),encoding="utf-8",newline="\n")
            plan=ModelPlan(id="unpriced-flow",model_file=inputs["model_file"],model_kind="global_m2",
                environment_file=inputs["environment_file"],base_dir=root,scheduling=Scheduling(default_device="gpu"))
            configure(plan,"dense")
            report=plan.inspect()
            self.assertEqual(report.declarations.status,"valid")
            self.assertEqual(report.cost_coverage.status,"missing")
            self.assertFalse(report.executable)
            with self.assertRaises(FileNotFoundError):plan.compile()

    def test_handoff_conflicting_attention_placement_rejects(self):
        simulator=make_simulator()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(InvalidInput,"handoff destination"):
                simulator.compile(scheduling=Scheduling(default_device="gpu"),
                    configure=lambda p:p.handoff(PhaseHandoff("dram")),directory=folder)

    def test_checkpoint_replay_uses_same_request_flow_and_policy(self):
        simulator=make_simulator()
        with tempfile.TemporaryDirectory() as folder:
            with simulator.compile(scheduling=Scheduling(default_device="gpu",release="last_use"),
                configure=lambda p:configure(p,"dense"),directory=folder) as compiled:
                partial=compiled.plan.run(max_decisions=12)
                checkpoint=compiled.plan.checkpoint(partial)
                resumed=compiled.plan.run(checkpoint=checkpoint)
                full=compiled.plan.run()
                self.assertEqual(resumed[0].termination_reason,"goal_reached")
                self.assertEqual(resumed[0].evaluation.events,full[0].evaluation.events)
                self.assertEqual(resumed[0].evaluation.makespan_ns,full[0].evaluation.makespan_ns)
                source=Path(folder)/"model.global.json"
                source.write_bytes(source.read_bytes()+b"\n")
                with self.assertRaises(InvalidInput):compiled.plan.run(checkpoint=checkpoint)

    def test_no_profile_inference_or_fused_batch_substitution(self):
        with self.assertRaises(Unsupported):RequestPipeline(batch_accounting="fused")
        with self.assertRaises(InvalidInput):RequestFlow(requests=(Request("a",1),Request("a",2)))
        simulator=make_simulator()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(Unsupported):
                Simulator(model=simulator.model,hardware=replace(simulator.hardware,gpu=None),
                    workload=simulator.workload,cost_model=AnalyticalCosts()).compile(
                        scheduling=Scheduling(default_device="cpu"),directory=folder)


if __name__=="__main__":unittest.main()
