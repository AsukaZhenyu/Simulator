"""One persistent run combines Attention/FFN/P-D/preparation mechanisms.

The physical Dense .5 model comes from existing global fixtures. All new
stage durations are independent synthetic declarations, not measured prices.
"""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from llm_infer_model.tensor.spec import COPY_READY


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "mapping/examples/python"


def _example_module():
    # The example's own CLI helper is intentionally reused, not reimplemented.
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location("combined_execution_regression", EXAMPLES / "execution_combined_demo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class ExecutionPhaseCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = _example_module()
        cls.compiled = cls.example.build_plan().compile(source_files=(cls.example.__file__,))
        cls.result_run, cls.session, cls.policy, _ = cls.compiled.run()
        if cls.result_run.termination_reason != "goal_reached" or cls.result_run.evaluation.status != "valid":
            raise AssertionError((cls.result_run.termination_reason, cls.result_run.compilation_error, cls.result_run.evaluation.reason))

    def test_all_four_scopes_contain_both_transforms_and_real_dependency_tasks(self):
        bindings = self.compiled.ir_document()["backend"]["execution"]
        self.assertEqual({row["scope_id"] for row in bindings},
            {f"denseL{layer}-{phase}" for layer in (0, 1) for phase in ("prefill", "decode")})
        for binding in bindings:
            pipeline = binding["spec"]
            self.assertEqual(pipeline["format_version"], "execution-pipeline-0.1")
            self.assertEqual([row["format_version"] for row in pipeline["transforms"]],
                             ["attention-stages-0.1", "ffn-execution-0.1"])
            self.assertEqual(pipeline["transforms"][1]["merge_input_count"], 2)
        graph = self.session.execution_document()
        self.assertEqual(sum(task["kind"] == "attention_qk" for task in graph["tasks"]), 4)
        self.assertEqual(sum(task["kind"] == "ffn_sum" for task in graph["tasks"]), 4)
        self.assertFalse(any(task["kind"] in {"attention", "dense_ffn"} for task in graph["tasks"]))
        sums = [operation for operation in graph["workload"]["operations"] if operation["semantic_op"] == "ffn_sum"]
        self.assertEqual(len(sums), 4)
        self.assertTrue(all(len(operation["inputs"]) == 2 and operation["op_params"] == [2] for operation in sums))

    def test_phase_placement_changes_attention_but_keeps_explicit_ffn_shards(self):
        attention = {"projection", "attention_qk", "attention_softmax", "attention_pv", "attention_output"}
        shard_kinds = {"ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down"}
        phase_counts = {"prefill": 0, "decode": 0}
        shards = set()
        for record in self.result_run.decisions:
            if record.action.kind != "COMPUTE":
                continue
            operation = next(op for op in record.observation.operations if op.id == record.action.target_id)
            if operation.kind in attention:
                phase_counts[operation.phase] += 1
                self.assertEqual(record.action.device, "gpu" if operation.phase == "prefill" else "cpu")
            elif operation.kind in shard_kinds:
                suffix = operation.region_id.rsplit(":", 1)[-1]
                self.assertIn(suffix, {"host", "accelerator"})
                self.assertEqual(record.action.device, "cpu" if suffix == "host" else "gpu")
                shards.add((operation.phase, suffix))
        self.assertEqual(phase_counts, {"prefill": 10, "decode": 10})
        self.assertEqual(shards, {(phase, shard) for phase in ("prefill", "decode") for shard in ("host", "accelerator")})

    def test_startup_and_spare_both_charge_and_same_weight_leaves_serve_both_phases(self):
        audit = self.policy.audit_document()["explicit_data_preparations"]
        self.assertEqual(len(audit), 6)
        self.assertEqual(sum(row["trigger"] == "startup" for row in audit), 3)
        self.assertEqual(sum(row["trigger"] == "spare" for row in audit), 3)
        events = self.result_run.evaluation.events
        first_compute = next(event for event in events if event.kind == "COMPUTE")
        for declaration in audit:
            target = declaration["tensor_id"]
            copies = [event for event in events if event.kind == "COPY" and event.target_id == target]
            self.assertEqual(len(copies), 1, target)
            self.assertEqual(copies[0].bytes_transferred, 1536)
            self.assertGreater(copies[0].end_ns, copies[0].start_ns)
            if declaration["trigger"] == "startup":
                self.assertLessEqual(copies[0].end_ns, first_compute.start_ns)
            phases = set()
            for record in self.result_run.decisions:
                if record.action.kind != "COMPUTE":
                    continue
                operation = next(op for op in record.observation.operations if op.id == record.action.target_id)
                if target in operation.inputs:
                    phases.add(operation.phase)
                    tensor = next(tensor for tensor in record.observation.tensors if tensor.id == target)
                    self.assertEqual(tensor.status_at("vram"), COPY_READY)
            self.assertEqual(phases, {"prefill", "decode"}, target)

    def test_kv_handoff_and_round_gate_preserve_one_clock_and_eight_identities(self):
        final = self.session.persistent_state_document()
        self.assertEqual(final["t_ns"], self.result_run.evaluation.makespan_ns)
        expected = {f"kv:denseL{layer}:r0:{position}" for layer in (0, 1) for position in range(4)}
        self.assertEqual(set(final["kv_ids"]), expected)
        self.assertEqual(len(final["kv_ids"]), 8)
        self.assertEqual(len(final["completed_invocations"]), 4)
        events = self.result_run.evaluation.events
        gate = next(event for event in events if event.kind == "COMPUTE" and ":trace_next:" in event.target_id)
        phase_events = {"prefill": [], "decode": []}
        for record in self.result_run.decisions:
            if record.action.kind != "COMPUTE":
                continue
            operation = next(op for op in record.observation.operations if op.id == record.action.target_id)
            if operation.phase in phase_events and operation.kind != "state_update":
                phase_events[operation.phase].append(next(event for event in events if event.kind == "COMPUTE" and
                                                         event.target_id == operation.id))
        self.assertGreaterEqual(gate.start_ns, max(event.end_ns for event in phase_events["prefill"]))
        self.assertLessEqual(gate.end_ns, min(event.start_ns for event in phase_events["decode"]))
        for layer in (0, 1):
            decode_qk = next(event for event in events if event.kind == "COMPUTE" and
                             f"g:denseL{layer}-decode:" in event.target_id and event.target_id.endswith("attn.qk"))
            for position in range(3):
                target = f"kv:denseL{layer}:r0:{position}"
                copy = next(event for event in events if event.kind == "COPY" and event.target_id == target and
                            event.src == "vram" and event.dst == "dram")
                self.assertGreater(copy.bytes_transferred, 0)
                self.assertLessEqual(copy.end_ns, decode_qk.start_ns)


if __name__ == "__main__":
    unittest.main()
