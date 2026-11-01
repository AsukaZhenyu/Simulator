"""Independent request-region target, preservation and conflict checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments import strategy_fidelity as ref
from llm_simulator import InvalidInput, ModelPlan, Scheduling, Simulator, Unsupported, Workload
from dataclasses import replace
from unittest.mock import patch
from llm_infer_model.m2.execution_pipeline import (
    ExecutionRegionSpec, apply_execution_transform, parse_transform, transform_document,
)
from llm_infer_model.m2.expand import load_and_compile


class ExecutionRegionTests(unittest.TestCase):
    def test_region_roundtrip_and_nonoverlapping_family_composition(self):
        def region(request):
            return dict(format_version="execution-region-0.1", request_ids=[request], transform=ref.attention_spec())
        document = dict(format_version="execution-pipeline-0.1", transforms=[region("r0"), region("r1")])
        self.assertEqual(transform_document(parse_transform(document)), document)
        for second in (region("r0"), ref.attention_spec()):
            with self.subTest(second=second.get("format_version")), self.assertRaisesRegex(InvalidInput, "overlapping"):
                parse_transform(dict(document, transforms=[region("r0"), second]))

    def test_region_malformed_unknown_and_nested_targets_fail(self):
        document = dict(format_version="execution-region-0.1", request_ids=["r0"], transform=ref.attention_spec())
        for selected in ([], ["r0", "r0"], [None], [True], "r0", ["bad:identity"]):
            with self.subTest(selected=selected), self.assertRaises(InvalidInput):
                parse_transform(dict(document, request_ids=selected))
        with self.assertRaises(Unsupported):
            parse_transform(dict(document, transform=document))
        with self.assertRaises(InvalidInput):
            transform_document(ExecutionRegionSpec(["r0"], parse_transform(ref.attention_spec())))

    def test_public_request_module_records_region_in_same_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            simulator = ref.dense()
            inputs = simulator.cost_model.build_inputs(root / "inputs", simulator.model, simulator.hardware, simulator.workload)
            plan = ModelPlan(id="scoped", model_kind=inputs["model_kind"], model_file=inputs["model_file"],
                environment_file=inputs["environment_file"], scheduling=Scheduling(default_device="gpu"))
            with plan.module("one_request", request_id="request0"):
                plan.transform("layer0-prefill", ref.attention_spec())
            choice = plan.declaration_document()["backend"]["execution"][0]["spec"]["transforms"][0]
            self.assertEqual(choice["format_version"], "execution-region-0.1")
            self.assertEqual(choice["request_ids"], ["request0"])
            compiled = plan.compile()
            raw = compiled.run()
            self.assertEqual(raw[0].termination_reason, "goal_reached")
            self.assertEqual(len(compiled.ir_document()["backend"]["execution"]), 1)

    def test_request_module_does_not_silently_narrow_contradictory_explicit_region(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            simulator = ref.dense()
            inputs = simulator.cost_model.build_inputs(root / "inputs", simulator.model, simulator.hardware, simulator.workload)
            plan = ModelPlan(id="scoped", model_kind=inputs["model_kind"], model_file=inputs["model_file"],
                environment_file=inputs["environment_file"], scheduling=Scheduling(default_device="gpu"))
            with plan.module("request0", request_id="request0"):
                with self.assertRaisesRegex(InvalidInput, "contradicts"):
                    plan.transform("layer0-prefill", dict(format_version="execution-region-0.1", request_ids=["different"],
                                                          transform=ref.attention_spec()))

    def test_unknown_region_request_cannot_compile_as_a_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            simulator = ref.dense()
            inputs = simulator.cost_model.build_inputs(root / "inputs", simulator.model, simulator.hardware, simulator.workload)
            leaf = load_and_compile(root / "inputs/layer0-prefill.json")
            with self.assertRaisesRegex(InvalidInput, "absent request"):
                apply_execution_transform(leaf, ExecutionRegionSpec(("absent",), parse_transform(ref.attention_spec())))

    def test_region_guard_detects_unselected_task_cost_device_or_storage_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = ref.dense()
            simulator = Simulator(model=original.model, hardware=original.hardware,
                                  workload=Workload(1, batch_size=2), cost_model=original.cost_model)
            simulator.cost_model.build_inputs(root, simulator.model, simulator.hardware, simulator.workload)
            leaf = load_and_compile(root / "layer0-prefill.json")
            raw = ref.attention_spec()
            for entry in raw["entries"]:
                entry["batch"] = 2
            region = ExecutionRegionSpec(("request0",), parse_transform(raw))
            oid = next(task.operation_id for task in leaf.tasks if task.request_id == "request1")
            tid = "w:layer0:W_q"
            for corruption in ("task", "cost", "allowed_devices", "tensor"):
                with self.subTest(corruption=corruption):
                    workload, scenario = deepcopy(leaf.workload_document), deepcopy(leaf.scenario_document)
                    tasks = leaf.tasks
                    if corruption == "task":
                        tasks = tuple(replace(task, task_id="changed") if task.operation_id == oid else task for task in tasks)
                    elif corruption == "cost":
                        scenario["costs"]["compute"][oid]["gpu"]["duration_ns"] += 1
                    elif corruption == "allowed_devices":
                        scenario["mapspace"]["allowed_devices"][oid] = ["cpu"]
                    else:
                        next(tensor for tensor in workload["tensors"] if tensor["id"] == tid)["storage_bytes"] += 4
                    broken = replace(leaf, tasks=tasks, workload_document=workload, scenario_document=scenario)
                    with patch("llm_infer_model.m2.attention_stages.transform_attention_stages", return_value=broken):
                        with self.assertRaisesRegex(InvalidInput, "unselected"):
                            apply_execution_transform(leaf, region)

    def test_checkpoint_binds_region_target_and_resumes_same_timeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = ref.dense()
            simulator = Simulator(model=original.model, hardware=original.hardware,
                                  workload=Workload(1, batch_size=2), cost_model=original.cost_model)
            inputs = simulator.cost_model.build_inputs(root, simulator.model, simulator.hardware, simulator.workload)
            raw = ref.attention_spec()
            for entry in raw["entries"]:
                entry["batch"] = 2
            def make_plan(request):
                plan = ModelPlan(id="checkpoint-regions", model_kind=inputs["model_kind"],
                    model_file=inputs["model_file"], environment_file=inputs["environment_file"],
                    scheduling=Scheduling(default_device="gpu", release="last_use"))
                with plan.module("selected", request_id=request):
                    plan.transform("layer0-prefill", raw)
                return plan.compile()
            compiled = make_plan("request0")
            checkpoint = compiled.checkpoint(compiled.run(max_decisions=12))
            resumed, complete = compiled.run(checkpoint=checkpoint), compiled.run()
            self.assertEqual(resumed[0].termination_reason, "goal_reached")
            self.assertEqual(resumed[0].evaluation.events, complete[0].evaluation.events)
            with self.assertRaisesRegex(InvalidInput, "sources or runtime changed"):
                make_plan("request1").run(checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
