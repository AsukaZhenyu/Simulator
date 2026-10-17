"""Python strategy composition through the real common compiler and runtime.

Synthetic prices verify lowering, causality and accounting, not hardware speed.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.python_frontend import ModelPlan, Scheduling, Select
from tensor_mapping.unified_strategy import _runtime_signature, create_session


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "mapping/examples/python/execution_transforms_demo.py"
loader = importlib.util.spec_from_file_location("execution_transform_example", EXAMPLE)
example = importlib.util.module_from_spec(loader)
loader.loader.exec_module(example)
FIXTURES = example.FIXTURES


def bare_dense(scheduling):
    return ModelPlan(id="execution-test", model_kind="global_m2",
        model_file=FIXTURES / "dense.global.json", environment_file=FIXTURES / "environment.json",
        visibility_mode="data_dependencies", scheduling=scheduling)


class ExecutionTransformFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = example.build_dense_plan().compile(source_files=example.source_files("dense"))
        cls.dense_result = cls.dense.run()
        cls.moe = example.build_moe_plan().compile(source_files=example.source_files("moe"))
        cls.moe_result = cls.moe.run()

    def assert_complete(self, result):
        self.assertEqual(result[0].termination_reason, "goal_reached", result[0].compilation_error)
        self.assertEqual(result[0].evaluation.status, "valid", result[0].evaluation.reason)

    def test_dense_lowering_has_real_parallel_shards_and_charged_sum_return(self):
        self.assert_complete(self.dense_result)
        run, session, policy, _ = self.dense_result
        operations = {op.id: op for op in session.observe().operations}
        computation = [event for event in run.evaluation.events if event.kind == "COMPUTE"]
        kinds = {operations[event.target_id].kind for event in computation}
        self.assertTrue({"attention_qk", "attention_softmax", "attention_pv", "attention_output",
                         "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down",
                         "ffn_sum", "ffn_residual"} <= kinds)
        self.assertFalse({"attention", "dense_ffn"} & kinds)
        shards = [event for event in computation if operations[event.target_id].kind in
                  {"ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down"}]
        self.assertEqual({event.device for event in shards}, {"cpu", "gpu"})
        self.assertTrue(any(a.device != b.device and a.start_ns < b.end_ns and b.start_ns < a.end_ns
                            for a in shards for b in shards))
        returns = [event for event in run.evaluation.events if event.kind == "COPY" and
                   event.src == "dram" and event.dst == "vram" and "partial" in event.target_id]
        self.assertTrue(returns)
        self.assertTrue(all(event.bytes_transferred > 0 and event.end_ns > event.start_ns for event in returns))
        records = policy.audit_document()["explicit_data_preparations"]
        self.assertEqual(len(records), 3)
        self.assertTrue(all(row["trigger"] == "startup" for row in records))
        first_compute = min(event.start_ns for event in computation)
        preload_ids = {row["tensor_id"] for row in records}
        preloads = [event for event in run.evaluation.events if event.kind == "COPY" and event.target_id in preload_ids]
        self.assertTrue(all(event.end_ns <= first_compute for event in preloads))

    def test_moe_shares_pipeline_but_keeps_router_gather_and_scatter(self):
        self.assert_complete(self.moe_result)
        run, session, policy, _ = self.moe_result
        kinds = {op.kind for op in session.observe().operations}
        self.assertTrue({"router", "gather_rows", "scatter_rows", "combine_rows", "ffn_gate",
                         "ffn_down", "ffn_sum", "attention_qk", "attention_pv"} <= kinds)
        initial = create_session(self.moe._compiled).observe()
        self.assertEqual(initial.route_history, ())
        self.assertFalse(any(op.kind.startswith("ffn_") for op in initial.operations))
        self.assertEqual(len(initial.experts), 4)
        self.assertTrue(all(len(expert.weight_tensor_ids) == 6 for expert in initial.experts))
        self.assertIn(run.decisions[0].action.kind, {"COPY", "READ_SSD"})
        self.assertEqual(run.decisions[0].observation.route_history, ())
        final = session.persistent_state_document()
        fragments = final["weight_partitions"]
        self.assertTrue(fragments)
        actual_weights = {tid for tid, _, _, _ in session.observe().weight_catalogue}
        self.assertTrue(all(row["tensor_id"] in actual_weights for row in fragments))
        inventoried = {row["tensor_id"] for row in final["tensor_inventory"] if row["role"] == "weight"}
        self.assertEqual(inventoried, actual_weights)
        records = policy.audit_document()["explicit_data_preparations"]
        self.assertTrue(any(row["trigger"] == "startup" for row in records))

    def test_checkpoint_during_startup_and_shard_execution_replays_same_state(self):
        for compiled, full in ((self.dense, self.dense_result), (self.moe, self.moe_result)):
            for limit in (1, 22):
                with self.subTest(model=compiled._compiled.document["id"], prefix=limit):
                    partial = compiled.run(max_decisions=limit)
                    resumed = compiled.run(checkpoint=compiled.checkpoint(partial))
                    self.assert_complete(resumed)
                    self.assertEqual(full[0].evaluation.events, resumed[0].evaluation.events)
                    self.assertEqual(_runtime_signature(full[1]), _runtime_signature(resumed[1]))
                    self.assertEqual(full[2].audit_document(), resumed[2].audit_document())

    def test_full_scoped_bindings_need_no_global_fallback(self):
        plan = bare_dense(Scheduling())
        for kind in ("projection", "attention", "dense_ffn"):
            plan.place(kind, "gpu", select=Select(invocation_id="dense-decode"))
        compiled = plan.compile()
        self.assertEqual(set(compiled.ir_document()["backend"]["policy"]["placements"].values()), {"scoped_binding"})
        self.assert_complete(compiled.run())
        incomplete = bare_dense(Scheduling())
        incomplete.place("projection", "gpu")
        with self.assertRaisesRegex(Unsupported, "preference|placement"):
            incomplete.compile()

    def test_explicit_default_only_completes_device_choices_not_missing_prices(self):
        compiled = bare_dense(Scheduling(default_device="gpu")).compile()
        self.assertEqual(set(compiled.ir_document()["backend"]["policy"]["placements"].values()), {"gpu"})
        plan = example.build_dense_plan()
        plan.place("ffn_gate", "cpu", select=Select(invocation_id="dense-decode"))
        with self.assertRaisesRegex(Unsupported, "exact|allowed"):
            plan.compile()

    def test_adaptive_whole_expert_queue_cannot_override_fixed_partition_devices(self):
        plan = example.build_moe_plan()
        plan._document["policy"]["experts"]["device_rule"] = "hybrimoe_queues"
        with self.assertRaisesRegex(Unsupported, "whole-expert|per-shard"):
            plan.compile()

    def test_resident_rule_respects_fixed_shard_devices(self):
        plan = example.build_moe_plan()
        plan._document["policy"]["experts"]["device_rule"] = "resident_first"
        compiled = plan.compile()
        result = compiled.run()
        self.assert_complete(result)
        operations = {op.id: op for op in result[1].observe().operations}
        for event in result[0].evaluation.events:
            if event.kind == "COMPUTE" and operations[event.target_id].kind.startswith("ffn_"):
                prices = operations[event.target_id].prices
                self.assertEqual(len(prices), 1)
                self.assertEqual(event.device, prices[0].device)

    def test_transform_append_retains_inherited_model_pass_and_binds_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            attention = directory / "attention.json"
            attention.write_bytes((FIXTURES / "dense.attention-stages.json").read_bytes())
            model = json.loads((FIXTURES / "dense.global.json").read_text(encoding="utf-8"))
            entry = model["rounds"][0]["invocations"][0]
            entry["invocation_file"] = str((ROOT / "mapping/examples/m2/hybrid-interface-synthetic-decode-1.json").resolve())
            entry["execution_file"] = str(attention)
            model_file = directory / "global.json"
            model_file.write_text(json.dumps(model), encoding="utf-8", newline="\n")
            plan = ModelPlan(id="inherited-passes", model_kind="global_m2", model_file=model_file,
                environment_file=FIXTURES / "environment.json", scheduling=Scheduling(default_device="gpu"))
            ffn = example._ffn(FIXTURES / "dense.ffn-costs.json", cpu_end=8, intermediate=32)
            plan.transform("dense-decode", ffn)
            compiled = plan.compile()
            passes = compiled.ir_document()["backend"]["execution"][0]["spec"]["transforms"]
            self.assertEqual([row["format_version"] for row in passes], ["attention-stages-0.1", "ffn-execution-0.1"])
            self.assertIn(str(attention), {row["path"] for row in compiled._compiled.sources})
            self.assert_complete(compiled.run())
            attention.write_text(attention.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "changed"):
                plan.compile()

    def test_duplicate_transform_fails_without_replacing_previous_pipeline(self):
        plan = example.build_dense_plan()
        previous = deepcopy(plan._document["execution"])
        with self.assertRaisesRegex(InvalidInput, "duplicate"):
            plan.transform_file("dense-decode", FIXTURES / "dense.attention-stages.json")
        self.assertEqual(previous, plan._document["execution"])


if __name__ == "__main__":
    unittest.main()
