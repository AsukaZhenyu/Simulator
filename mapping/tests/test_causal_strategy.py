"""Integrated execution variants, causal rules, compiler provenance and CLI."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.causal import compile_decision
from tensor_mapping import causal_strategy

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/causal-strategy"


def load(name):
    return causal_strategy.compile_causal_strategy_file(EXAMPLES / f"{name}.strategy.json")


class CausalStrategyTests(unittest.TestCase):
    def test_rows_mixed_moves_compact_groups_and_reconstructs_complete_output(self):
        compiled = load("rows-mixed")
        result = causal_strategy.run_causal_strategy(compiled)
        self.assertEqual(result.evaluation.status, "valid")
        graph = result.session.execution_document()
        regions = graph["boundary"]["row_execution"]["regions"]
        tensors = {row["id"]: row for row in graph["workload"]["tensors"]}
        copies = {event.target_id: event.bytes_transferred for event in result.evaluation.events
                  if event.kind == "COPY" and event.src == "vram" and event.dst == "dram"}
        self.assertEqual([copies[region["grouped_input_tensor"]] for region in regions], [16, 32, 16])
        self.assertTrue(all(tensors[region["grouped_input_tensor"]]["storage_bytes"] == len(region["work_items"]) * 16
                            for region in regions))
        self.assertEqual(result.session.state_out().requests[0].next_position, 2)
        self.assertEqual(result.session.state_out().hidden[0].storage_bytes, 32)

    def test_staged_placements_are_faithfully_compiled_and_audited(self):
        compiled = load("staged-hybrid")
        result = causal_strategy.run_causal_strategy(compiled)
        self.assertEqual(result.evaluation.status, "valid")
        kinds = {task["operation_id"]: task["kind"] for task in result.session.execution_document()["tasks"]}
        expected = dict(compiled.policy.placements)
        kinds_seen = set()
        for record in result.run.decisions:
            self.assertTrue(record.accepted)
            self.assertEqual(record.action, compile_decision(record.decision, record.observation))
            if record.action.kind == "COMPUTE":
                kind = kinds[record.action.target_id]
                kinds_seen.add(kind)
                self.assertEqual(record.action.device, expected[kind])
        self.assertTrue({"expert_gate", "expert_up", "expert_swiglu", "expert_down"} <= kinds_seen)
        self.assertEqual(len(result.policy_executor.audit_document()["decisions"]), len(result.run.decisions))
        self.assertFalse(result.session.execution_document()["boundary"]["numeric_execution_verified"])

    def test_history_prefetch_miss_is_real_paid_work_from_declared_past(self):
        compiled = load("history")
        result = causal_strategy.run_causal_strategy(compiled)
        self.assertEqual(result.evaluation.status, "valid")
        audit = result.policy_executor.audit_document()
        prediction = next(row for row in audit["predictions"] if row["tensor_id"] == "w:moe-small:E3")
        self.assertEqual(prediction["outcome"], "miss")
        read = next(event for event in result.evaluation.events
                    if event.kind == "READ_SSD" and event.target_id == prediction["tensor_id"])
        self.assertEqual(read.bytes_transferred, 64)
        decision = result.run.decisions[read.action_index]
        self.assertEqual(decision.observation.route_history, ())
        self.assertGreaterEqual(result.evaluation.ssd_read_bytes, 256)
        self.assertEqual(len(result.policy_executor.history_document()["entries"]), 5)

    def test_cpu_pause_and_external_memory_are_observed_without_future_queue(self):
        result = causal_strategy.run_causal_strategy(load("fluctuating"))
        self.assertEqual(result.evaluation.status, "valid")
        observations = [record.observation for record in result.run.decisions]
        paused = [observation for observation in observations if not observation.cpu_available]
        self.assertTrue(paused)
        self.assertTrue(any(usage.external_bytes == 3000 for observation in paused for usage in observation.memory))
        for observation in observations:
            self.assertFalse(hasattr(observation, "environment_events"))
            self.assertFalse(hasattr(observation, "route_trace"))

    def test_full_cli_artifacts_preserve_sources_and_privileged_graph_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = causal_strategy.main([str(EXAMPLES / "staged-hybrid.strategy.json"), "--out", str(output)])
            self.assertEqual(code, 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["execution_variant"], "selected_rows")
            self.assertFalse(summary["physical_calibration_claimed"])
            self.assertFalse(summary["paper_strategy_reproduced"])
            self.assertEqual(summary["policy_prediction_runtime_overhead"], "not_profiled_host_control_logic")
            for source in summary["sources"]:
                self.assertEqual(hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(), source["sha256"])
            graph = json.loads((output / "compiled_graph.json").read_text())
            self.assertEqual(graph["visibility_scope"], "modeling_debug_complete_graph_not_policy_input")
            for name in ("policy.json", "strategy.json", "policy_audit.json", "history_out.json", "execution_spec.json"):
                self.assertTrue((output / name).is_file())
            for path in output.glob("*.json"):
                self.assertNotIn(b"\r", path.read_bytes(), path.name)

    def temporary_strategy(self, directory, **overrides):
        document = json.loads((EXAMPLES / "rows-cpu.strategy.json").read_text())
        for field in ("invocation_file", "environment_file", "execution_file", "policy_file"):
            document[field] = str((EXAMPLES / document[field]).resolve())
        document.update(overrides)
        path = Path(directory) / "strategy.json"
        path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
        return path

    def test_history_cannot_declare_this_invocations_future_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = json.loads((EXAMPLES / "past.history.json").read_text())
            document["entries"][0]["request_id"] = "r0"
            history = Path(temporary) / "history.json"
            history.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            strategy = self.temporary_strategy(temporary, history_file=str(history))
            with self.assertRaisesRegex(InvalidInput, "current/future"):
                causal_strategy.compile_causal_strategy_file(strategy)

    def test_policy_missing_stage_or_wrong_history_catalogue_refused_at_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = json.loads((EXAMPLES / "cpu.policy.json").read_text())
            policy["placements"].pop("gather_rows")
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "missing execution kinds"):
                causal_strategy.compile_causal_strategy_file(self.temporary_strategy(temporary, policy_file=str(path)))
            history = json.loads((EXAMPLES / "past.history.json").read_text())
            history["entries"][0]["expert_ids"] = [0, 99]
            path = Path(temporary) / "history.json"
            path.write_text(json.dumps(history), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "catalogue"):
                causal_strategy.compile_causal_strategy_file(self.temporary_strategy(temporary, history_file=str(path)))

    def test_fixed_placement_without_allowed_exact_device_cost_refused_at_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = (EXAMPLES / "../m2/moe-interface-synthetic-prefill-2.json").resolve()
            invocation = json.loads(source.read_text())
            for field in ("layer_file", "costs_file"):
                invocation[field] = str((source.parent / invocation[field]).resolve())
            invocation["devices"] = ["cpu"]
            invocation_path = Path(temporary) / "cpu-only.invocation.json"
            invocation_path.write_text(json.dumps(invocation), encoding="utf-8", newline="\n")
            policy = json.loads((EXAMPLES / "cpu.policy.json").read_text())
            policy["placements"]["gather_rows"] = "gpu"
            policy_path = Path(temporary) / "unsupported.policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
            strategy = self.temporary_strategy(temporary, invocation_file=str(invocation_path),
                                               policy_file=str(policy_path))
            with self.assertRaisesRegex(Unsupported, "no allowed exact execution cost"):
                causal_strategy.compile_causal_strategy_file(strategy)

    def test_last_use_release_lowers_rows_dram_peak_without_changing_execution_time(self):
        retained = causal_strategy.run_causal_strategy(load("rows-retain"))
        released = causal_strategy.run_causal_strategy(load("rows-cpu"))
        self.assertEqual(retained.evaluation.status, "valid")
        self.assertEqual(released.evaluation.status, "valid")
        self.assertLess(released.evaluation.peak_dram_bytes, retained.evaluation.peak_dram_bytes)
        self.assertEqual(released.evaluation.makespan_ns, retained.evaluation.makespan_ns)
        release_records = [record for record in released.run.decisions if record.action.kind == "EVICT"]
        self.assertTrue(release_records)
        self.assertTrue(all(record.observation.task_graph_complete for record in release_records))
        audit = released.policy_executor.audit_document()
        self.assertTrue(any(row["reason"] == "release_after_last_visible_use" for row in audit["decisions"]))
        for record in release_records:
            self.assertEqual(record.action, compile_decision(record.decision, record.observation))
            tensor = next(item for item in record.observation.tensors if item.id == record.action.target_id)
            self.assertEqual(tensor.role, "intermediate")
            self.assertFalse(tensor.initial_locations)
            self.assertIsNone(tensor.goal_location)

    def test_warm_cache_pressure_needs_last_use_and_evicts_inherited_expert_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            strategy = json.loads((EXAMPLES / "warm-cache-pressure.strategy.json").read_text())
            for field in ("invocation_file", "environment_file", "execution_file", "policy_file"):
                strategy[field] = str((EXAMPLES / strategy[field]).resolve())
            policy = json.loads((EXAMPLES / "warm-cache.policy.json").read_text())
            self.assertEqual(policy["release"]["rule"], "last_use")
            policy["release"]["rule"] = "none"
            policy_path = Path(temporary) / "retain.policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
            strategy["policy_file"] = str(policy_path)
            strategy_path = Path(temporary) / "retain.strategy.json"
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8", newline="\n")
            retained = causal_strategy.run_causal_strategy(causal_strategy.compile_causal_strategy_file(strategy_path))
            self.assertNotEqual(retained.evaluation.status, "valid")
        released = causal_strategy.run_causal_strategy(load("warm-cache-pressure"))
        self.assertEqual(released.evaluation.status, "valid")
        evicted = [record for record in released.run.decisions if record.action.kind == "EVICT"
                   and record.action.target_id == "w:moe-small:E3" and record.action.location == "vram"]
        self.assertTrue(evicted)
        for record in evicted:
            tensor = next(item for item in record.observation.tensors if item.id == record.action.target_id)
            self.assertEqual(set(tensor.initial_locations), {"ssd", "vram"})
            self.assertEqual(tensor.goal_location, "ssd")
            self.assertEqual(tensor.status_at("ssd"), "READY")
            self.assertEqual(record.action, compile_decision(record.decision, record.observation))
        audit = released.policy_executor.audit_document()
        self.assertTrue(any(row["reason"] == "demand_capacity_eviction" for row in audit["decisions"]))
        self.assertFalse(audit["claims"]["optimality"])

    def test_changed_execution_bytes_during_parse_refuse_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution.json"
            path.write_bytes((EXAMPLES / "rows.execution.json").read_bytes())
            strategy = self.temporary_strategy(temporary, execution_file=str(path))
            original = causal_strategy._read_object
            def parse_then_change(source, data):
                result = original(source, data)
                if source == path:
                    path.write_bytes(data + b"\n")
                return result
            with patch.object(causal_strategy, "_read_object", side_effect=parse_then_change):
                with self.assertRaisesRegex(InvalidInput, "input changed"):
                    causal_strategy.compile_causal_strategy_file(strategy)

    def test_budget_exhaustion_and_nonempty_output_do_not_become_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            strategy = self.temporary_strategy(temporary, max_decisions=1)
            compiled = causal_strategy.compile_causal_strategy_file(strategy)
            result = causal_strategy.run_causal_strategy(compiled)
            self.assertEqual(result.evaluation.status, "incomplete_mapping")
            output = Path(temporary) / "run"
            causal_strategy.write_strategy_result(output, compiled, result)
            self.assertFalse((output / "state_out.json").exists())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(causal_strategy.main([str(strategy), "--out", str(output)]), 2)

    def test_strategy_has_no_hidden_defaults(self):
        document = json.loads((EXAMPLES / "rows-cpu.strategy.json").read_text())
        for field in document:
            bad = deepcopy(document)
            bad.pop(field)
            with self.subTest(field=field), self.assertRaises(InvalidInput):
                causal_strategy.parse_causal_strategy(bad)
        for value in (True, 0, -1, None):
            bad = dict(document, max_decisions=value)
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                causal_strategy.parse_causal_strategy(bad)


if __name__ == "__main__":
    unittest.main()
