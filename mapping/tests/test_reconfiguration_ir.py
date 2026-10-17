"""Resource response is explicit, charged, causal and replayable."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.causal import ControlDecision, Decision, compile_command
from tensor_mapping.reconfiguration import parse_environment_trace, parse_reconfiguration
from tensor_mapping.unified_strategy import (checkpoint_document, compile_unified_strategy_file,
    create_session, run_unified_strategy, write_unified_result, _runtime_signature)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/reconfiguration"


class ReconfigurationIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = {name: compile_unified_strategy_file(EXAMPLES / (name + ".strategy.json"))
                        for name in ("m2", "global", "block")}
        cls.full = {name: run_unified_strategy(model) for name, model in cls.compiled.items()}
        cls.moe_compiled = compile_unified_strategy_file(EXAMPLES / "moe.strategy.json")
        cls.moe_full = run_unified_strategy(cls.moe_compiled)

    def setUp(self):
        (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def variant(self, name="m2", *, strategy=None, environment=None):
        source = EXAMPLES / (name + ".strategy.json")
        document = json.loads(source.read_text(encoding="utf-8"))
        for field in ("services_file", "environment_file", "history_file", "routing_scores_file"):
            if document[field] is not None:
                document[field] = str((source.parent / document[field]).resolve())
        document["model"]["file"] = str((source.parent / document["model"]["file"]).resolve())
        for entry in document["execution"]:
            if "invocation_files" in entry["spec"]:
                entry["spec"]["invocation_files"] = [str((source.parent / ref).resolve())
                                                      for ref in entry["spec"]["invocation_files"]]
        if environment is not None:
            path = self.folder / "environment.json"
            path.write_text(json.dumps(environment) + "\n", encoding="utf-8", newline="\n")
            document["environment_file"] = str(path)
        if strategy:
            strategy(document)
        path = self.folder / "strategy.json"
        path.write_text(json.dumps(document) + "\n", encoding="utf-8", newline="\n")
        return compile_unified_strategy_file(path)

    def test_three_adapters_accept_requests_only_after_safe_release(self):
        for name, result in self.full.items():
            run, session, _, _ = result
            with self.subTest(name=name):
                self.assertEqual(run.evaluation.status, "valid", run.compilation_error)
                self.assertEqual(run.termination_reason, "goal_reached")
                self.assertEqual([row.status for row in run.runtime_result.reclaim_records], ["accepted"])
                self.assertFalse(session.observe().pending_reclaim)
                self.assertFalse(session.observe().control_running)
                self.assertTrue(any(row.action.kind == "ACK_RECLAIM" for row in run.decisions))
                released = [row for row in run.decisions if row.observation.pending_reclaim and row.action.kind == "EVICT"]
                self.assertTrue(released)
                for row in released:
                    self.assertIn(row.action.target_id, row.observation.safe_weight_release_candidates)
                    self.assertEqual(row.action.location, "vram")
                for row in run.decisions:
                    if row.observation.pending_reclaim:
                        self.assertIn(row.action.kind, {"DECIDE", "ADVANCE", "WAIT_UNTIL", "EVICT", "ACK_RECLAIM", "DECLINE_RECLAIM"})
                    for memory in row.observation.memory:
                        if memory.capacity_bytes is not None:
                            self.assertLessEqual(memory.used_bytes, memory.capacity_bytes)

    def test_mid_control_checkpoint_reconstructs_same_absolute_timeline(self):
        for name, budget in (("m2", 21), ("global", 22), ("block", 15)):
            model = self.compiled[name]
            partial = run_unified_strategy(model, max_decisions=budget)
            self.assertIsNotNone(partial[1].observe().control_running)
            self.assertIsNotNone(partial[1].observe().pending_reclaim)
            resumed = run_unified_strategy(model, checkpoint=checkpoint_document(model, partial))
            self.assertEqual(_runtime_signature(resumed[1]), _runtime_signature(self.full[name][1]))
            self.assertEqual(resumed[2].audit_document(), self.full[name][2].audit_document())
            self.assertEqual(resumed[0].runtime_result.control_events, self.full[name][0].runtime_result.control_events)

    def test_different_private_recovery_preserves_public_prefix(self):
        environment = json.loads((EXAMPLES / "global.environment.json").read_text(encoding="utf-8"))
        for event in environment["events"]:
            if "vram_external_bytes" in event:
                event["t_ns"] = 8000
        variant = self.variant("global", environment=environment)
        other = run_unified_strategy(variant)
        def prefix(result):
            return [(row.observation_sha256, asdict(row.decision)) for row in result[0].decisions
                    if row.observation.t_ns < 6500]
        self.assertEqual(prefix(other), prefix(self.full["global"]))
        self.assertTrue(prefix(other))

    def test_cpu_restoration_cannot_rebind_prepared_gpu_projection(self):
        environment = json.loads((EXAMPLES / "environment.json").read_text(encoding="utf-8"))
        environment["events"] = [{"t_ns": 0, "cpu_available": False}, {"t_ns": 1000, "cpu_available": True}]
        variant = self.variant(strategy=lambda doc: doc["reconfiguration"].update(cpu_response="current_cost"),
                               environment=environment)
        run, session, policy, _ = run_unified_strategy(variant)
        self.assertEqual(run.evaluation.status, "valid", run.compilation_error)
        projection = next(op for op in session.observe().operations if op.kind == "projection")
        compute = next(row for row in run.decisions if row.action.kind == "COMPUTE" and row.action.target_id == projection.id)
        self.assertTrue(compute.observation.cpu_available)
        self.assertEqual(compute.action.device, "gpu")
        self.assertIn({"operation_id": projection.id, "device": "gpu"}, policy.audit_document()["operation_bindings"])

    def test_staged_moe_region_pauses_without_rebinding_any_stage(self):
        run, session, _, _ = self.moe_full
        self.assertEqual(run.evaluation.status, "valid", run.compilation_error)
        operations = {op.id: op for op in session.observe().operations}
        regions = {}
        for row in run.decisions:
            if row.action.kind != "COMPUTE":
                continue
            op = operations[row.action.target_id]
            if op.kind in {"expert_gate", "expert_up", "expert_swiglu", "expert_down"}:
                regions.setdefault((op.invocation_id, op.region_id), []).append((op.kind, row.action.device, row.observation.t_ns))
        for rows in regions.values():
            self.assertEqual({kind for kind, _, _ in rows}, {"expert_gate", "expert_up", "expert_swiglu", "expert_down"})
            self.assertEqual(len({device for _, device, _ in rows}), 1)
        spanning = [rows for rows in regions.values() if rows[0][1] == "cpu"
                    and rows[0][2] < 2008000 and rows[-1][2] > 2050000]
        self.assertTrue(spanning)
        self.assertTrue(any(not row.observation.cpu_available and any(
            running.resource == "cpu_compute" for running in row.observation.running) for row in run.decisions))

    def test_moe_private_cpu_recovery_cannot_change_earlier_decisions(self):
        environment = json.loads((EXAMPLES / "moe.environment.json").read_text(encoding="utf-8"))
        environment["events"][1]["t_ns"] = 2080000
        other = run_unified_strategy(self.variant("moe", environment=environment))
        prefix = lambda result: [(row.observation_sha256, asdict(row.decision)) for row in result[0].decisions
                                 if row.observation.t_ns < 2050000]
        self.assertEqual(prefix(other), prefix(self.moe_full))

    def test_decline_finishes_inference_but_never_counts_as_resource_success(self):
        model = self.variant(strategy=lambda doc: doc["reconfiguration"].update(reclaim_response="decline"))
        result = run_unified_strategy(model)
        self.assertEqual(result[0].evaluation.status, "valid")
        write_unified_result(self.folder / "declined", model, result)
        summary = json.loads((self.folder / "declined/summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["inference_goal_reached"])
        self.assertFalse(summary["resource_compliant_success"])
        self.assertFalse(summary["all_published_reclaim_requests_satisfied"])
        self.assertEqual(summary["unmet_reclaim_request_count"], 1)

    def test_cli_decline_exit_code_reports_resource_failure(self):
        model = self.variant(strategy=lambda doc: doc["reconfiguration"].update(reclaim_response="decline"))
        strategy = self.folder / "strategy.json"
        cli = subprocess.run([sys.executable, "-m", "tensor_mapping.unified_strategy", str(strategy),
                              "--out", str(self.folder / "cli-declined")], capture_output=True, text=True)
        self.assertEqual(cli.returncode, 4, cli.stderr)
        self.assertFalse(json.loads(cli.stdout)["resource_compliant_success"])

    def test_real_versioned_kv_is_not_a_clean_weight_release_candidate(self):
        run = self.full["global"][0]
        saw_kv = False
        for row in run.decisions:
            kvs = {tensor.id for tensor in row.observation.tensors if tensor.id.startswith("kv:")}
            saw_kv |= bool(kvs)
            self.assertFalse(kvs & set(row.observation.safe_weight_release_candidates))
            if row.observation.pending_reclaim and row.action.kind == "EVICT":
                self.assertNotIn(row.action.target_id, kvs)
        self.assertTrue(saw_kv)
        self.assertEqual(len(self.full["global"][1].persistent_state_document()["kv_ids"]), 8)

    def test_artifacts_keep_control_cost_separate_from_payload(self):
        model, result = self.compiled["global"], self.full["global"]
        write_unified_result(self.folder / "full", model, result)
        summary = json.loads((self.folder / "full/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["control_cost_ns"], 9)
        self.assertEqual(summary["control_timing_scope"], "dedicated_serial_control_resource")
        self.assertTrue(summary["resource_compliant_success"])
        actions = json.loads((self.folder / "full/execution.json").read_text(encoding="utf-8"))["actions"]
        self.assertFalse(any(row["kind"] in {"DECIDE", "ACK_RECLAIM", "DECLINE_RECLAIM", "WAIT_UNTIL"} for row in actions))
        self.assertTrue((self.folder / "full/reconfiguration.json").is_file())

    def test_compiler_rejects_guessed_request_and_tensor_lowering(self):
        observation = create_session(self.compiled["global"]).observe()
        for decision in (ControlDecision("ACK_RECLAIM", request_id="global-external-app"),
                         ControlDecision("DECIDE", request_id="future-request"),
                         ControlDecision("WAIT_UNTIL", until_ns=0),
                         Decision("ACK_RECLAIM", target_id="global-external-app")):
            with self.subTest(decision=decision), self.assertRaises(SpecError):
                compile_command(decision, observation)

    def test_strict_version_and_cost_contract(self):
        environment = json.loads((EXAMPLES / "m2.environment.json").read_text(encoding="utf-8"))
        for mutation in (
                lambda doc: doc.update(unrecognized=True),
                lambda doc: doc["control_costs"].update(ack_ns=None),
                lambda doc: doc["control_costs"].pop("decision_ns"),
                lambda doc: doc["requests"][0].update(domain="dram"),
                lambda doc: doc["requests"][0].update(deadline_ns=0)):
            altered = deepcopy(environment)
            mutation(altered)
            with self.subTest(mutation=mutation), self.assertRaises(SpecError):
                parse_environment_trace(altered)
        with self.assertRaises(SpecError):
            self.variant(strategy=lambda doc: doc.update(format_version="unified-strategy-0.2"))
        with self.assertRaises(SpecError):
            self.variant(strategy=lambda doc: doc.pop("reconfiguration"))
        with self.assertRaises(SpecError):
            parse_reconfiguration({"reclaim_response": "release_clean_weights", "release_order": "lru"})


if __name__ == "__main__":
    unittest.main()
