"""End-to-end causal artifacts, failure attribution and input evidence."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.tensor.spec import InvalidInput
from tensor_mapping import causal_cli
from tensor_mapping.causal import SerialDemandPolicy, run_causal


_ROOT = Path(__file__).resolve().parents[1]
_INVOCATION = _ROOT / "examples/m2/moe-interface-synthetic-prefill-2.json"
_ENVIRONMENTS = _ROOT / "examples/causal"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class CausalCliTests(unittest.TestCase):
    def call(self, output, environment="no-disturbance", policy="serial_cpu", budget=100):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return causal_cli.main([
                str(_INVOCATION), "--environment", str(_ENVIRONMENTS / f"{environment}.environment.json"),
                "--policy", policy, "--max-decisions", str(budget), "--out", str(output),
            ])

    def test_cpu_artifacts_trace_decisions_and_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(self.call(output), 0)
            summary = read(output / "summary.json")
            self.assertEqual((summary["status"], summary["makespan_ns"]), ("valid", 34_000_000))
            self.assertEqual(summary["information_mode"], "causal_observation")
            for flag in ("search_performed", "optimality_proven", "global_infeasibility_proven",
                         "physical_calibration_claimed", "paper_strategy_reproduced",
                         "future_routes_in_policy_api", "future_environment_events_in_policy_api"):
                self.assertFalse(summary[flag])
            for source in summary["sources"]:
                self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            decisions = read(output / "decisions.json")["decisions"]
            actions = read(output / "execution.json")["actions"]
            events = read(output / "events.json")["events"]
            self.assertEqual(len(decisions), summary["decision_count"])
            self.assertEqual(len(actions), len(events))
            for index, (decision, action, event) in enumerate(zip(decisions, actions, events)):
                self.assertTrue(decision["accepted"])
                self.assertEqual(action["decision_index"], index)
                self.assertEqual(event["action_index"], index)
                observation_bytes = json.dumps(decision["observation"], sort_keys=True,
                                              separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.assertEqual(decision["observation_sha256"], hashlib.sha256(observation_bytes).hexdigest())
            self.assertEqual(read(output / "state_out.json")["requests"][0]["next_position"], 2)
            self.assertEqual(len(read(output / "revealed_history.json")["entries"]), 2)
            self.assertEqual(read(output / "environments.json")["events"], [])
            for path in output.glob("*.json"):
                self.assertNotIn(b"\r", path.read_bytes(), path.name)

    def test_blackout_records_actual_completion_and_service_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(self.call(output, "cpu-blackout"), 0)
            self.assertEqual(read(output / "summary.json")["makespan_ns"], 36_000_000)
            projection = read(output / "events.json")["events"][0]
            self.assertEqual((projection["start_ns"], projection["nominal_end_ns"], projection["end_ns"]),
                             (0, 1_000_000, 3_000_000))
            service = [interval for interval in read(output / "service_intervals.json")["intervals"]
                       if interval["action_index"] == 0]
            self.assertEqual([(row["start_ns"], row["end_ns"]) for row in service],
                             [(0, 500_000), (2_500_000, 3_000_000)])
            environments = read(output / "environments.json")["events"]
            self.assertEqual([row["t_ns"] for row in environments], [500_000, 2_500_000])
            self.assertTrue(all(row["accepted"] for row in environments))
            self.assertEqual(environments[0]["memory"][0]["workspace_bytes"], 16)
            self.assertEqual(environments[0]["memory"][0]["external_bytes"], 128)

    def test_gpu_artifacts_return_kv_copies_without_cpu_service(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(self.call(output, policy="serial_gpu", budget=150), 0)
            summary = read(output / "summary.json")
            self.assertGreater(summary["h2d_bytes"], 0)
            self.assertGreater(summary["d2h_bytes"], 0)
            self.assertFalse(any(row["resource"] == "cpu_compute"
                                 for row in read(output / "service_intervals.json")["intervals"]))
            state = read(output / "state_out.json")
            self.assertTrue(all(block["locations"] == ["dram", "vram"]
                                for block in state["requests"][0]["blocks"]))

    def test_memory_conflict_has_bounded_action_index_and_uncommitted_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(self.call(output, "memory-conflict"), 3)
            summary = read(output / "summary.json")
            self.assertEqual((summary["error_code"], summary["error_t_ns"]),
                             ("external_memory_conflict", 500_000))
            self.assertEqual(summary["error_action_index"], 1)
            self.assertIsNone(summary["makespan_ns"])
            self.assertFalse((output / "state_out.json").exists())
            environment = read(output / "environments.json")["events"][0]
            self.assertFalse(environment["accepted"])
            self.assertEqual(environment["memory"][0]["used_bytes"], 4224)
            observation = read(output / "decisions.json")["decisions"][-1]["observation"]
            self.assertEqual(observation["memory"][0]["external_bytes"], 0)
            self.assertEqual(len(read(output / "execution.json")["actions"]), 2)
            self.assertIsNone(read(output / "events.json")["events"][0]["end_ns"])

    def test_budget_exhaustion_is_incomplete_and_running_event_has_no_actual_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(self.call(output, budget=1), 3)
            summary = read(output / "summary.json")
            self.assertEqual(summary["termination_reason"], "decision_budget_exhausted")
            self.assertEqual(summary["status"], "incomplete_mapping")
            self.assertFalse(summary["global_infeasibility_proven"])
            self.assertFalse((output / "state_out.json").exists())
            self.assertIsNone(read(output / "events.json")["events"][0]["end_ns"])

    def test_invalid_environment_declarations_are_loud(self):
        for document in ({}, {"format_version": "environment-trace-0.1", "events": [], "unknown": 0},
                         {"format_version": "environment-trace-0.1", "events": None}):
            with self.subTest(document=document), self.assertRaises(InvalidInput):
                causal_cli.parse_environment(document)
        for event in ({"t_ns": True, "cpu_available": True}, {"t_ns": 1},
                      {"t_ns": 1, "cpu_available": 0}, {"t_ns": 1, "cpu_available": None},
                      {"t_ns": 1, "dram_external_bytes": -1}, {"t_ns": 1, "unknown": 0}):
            with self.subTest(event=event), self.assertRaises(InvalidInput):
                causal_cli.parse_environment({"format_version": "environment-trace-0.1", "events": [event]})

    def test_same_time_duplicate_and_unordered_events_are_rejected_before_outputs(self):
        for events in ([{"t_ns": 1, "cpu_available": False}, {"t_ns": 1, "cpu_available": True}],
                       [{"t_ns": 2, "cpu_available": False}, {"t_ns": 1, "cpu_available": True}]):
            with tempfile.TemporaryDirectory() as temporary:
                environment = Path(temporary) / "env.json"
                environment.write_text(json.dumps({"format_version": "environment-trace-0.1", "events": events}),
                                       encoding="utf-8", newline="\n")
                with self.assertRaises(InvalidInput):
                    causal_cli.load_session(_INVOCATION, environment)

    def test_nonempty_output_preserved_and_invalid_budget_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"keep")
            self.assertEqual(self.call(output), 2)
            self.assertEqual(sentinel.read_bytes(), b"keep")
            fresh = Path(temporary) / "fresh"
            self.assertEqual(self.call(fresh, budget=0), 2)
            self.assertFalse(fresh.exists())

    def test_bytes_changed_while_parsing_environment_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "env.json"
            environment.write_bytes((_ENVIRONMENTS / "no-disturbance.environment.json").read_bytes())
            original = causal_cli._read_object
            def mutate_after_parse(path, data):
                document = original(path, data)
                if path == environment.resolve():
                    environment.write_bytes(data + b"\n")
                return document
            with patch.object(causal_cli, "_read_object", side_effect=mutate_after_parse):
                with self.assertRaisesRegex(InvalidInput, "input changed"):
                    causal_cli.load_session(_INVOCATION, environment)

    def test_bytes_changed_after_replay_refuse_to_publish_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "env.json"
            environment.write_bytes((_ENVIRONMENTS / "no-disturbance.environment.json").read_bytes())
            session, sources = causal_cli.load_session(_INVOCATION, environment)
            run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=100)
            environment.write_bytes(environment.read_bytes() + b"\n")
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(InvalidInput, "input changed"):
                causal_cli.write_causal_run(output, run, session, sources, policy_name="serial_cpu", max_decisions=100)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
