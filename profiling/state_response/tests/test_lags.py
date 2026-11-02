"""Owned-worker timing evidence tests; no foreground timing/hardware inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "modeling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from profiling.state_response import analysis as a
from profiling.state_response import lags


class LagEvidenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="worker-lag-test-", dir=ROOT / "build")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0

    def record(self, *, start_delay=5, final_end=65, sampling_end=100, unknown_worker=False,
               both_heldout=False):
        self.count += 1
        directory = self.root / str(self.count)
        directory.mkdir()
        go = dict(start_tick=0, end_tick=100, phases=[
            dict(phase_index=0, begin_tick=0, end_tick=20, cpu_replay=0, gpu_replay=0),
            dict(phase_index=1, begin_tick=20, end_tick=60, cpu_replay=1, gpu_replay=1),
            dict(phase_index=2, begin_tick=60, end_tick=100, cpu_replay=0, gpu_replay=0)])
        record = dict(split="heldout" if both_heldout else "profile", run_id=f"profile-{self.count}",
            process_id=100 + self.count, seed=self.count, program_sha256="a" * 64,
            action_inventory=[dict(kind="COMPUTE", target_id="x", device="gpu")],
            clock=dict(clock_id="synthetic-QPC", frequency_hz=1_000_000_000),
            context=dict(execution_boundary="synthetic-completed"), raw_bindings={}, source_files={})
        inputs = dict(events={"go": go})
        for resource in ("cpu", "gpu"):
            intervals = [dict(iteration=0, enabled_check_tick=22, begin_tick=20 + start_delay,
                              end_tick=40, elapsed_ns=20 - start_delay),
                         dict(iteration=1, enabled_check_tick=50, begin_tick=51,
                              end_tick=final_end, elapsed_ns=final_end - 51)]
            inputs["worker_" + resource] = dict(role=resource + "-worker", process_id=200 + self.count,
                program_sha256=record["program_sha256"], actions=record["action_inventory"], clock=record["clock"],
                go_schedule=go, execution_boundary="synthetic-completed", ready_tick=0,
                sampling_begin_tick=0, sampling_end_tick=sampling_end,
                blocks=[dict(iterations=intervals, iteration_count=2)])
        for name, value in inputs.items():
            path = directory / (name + ".json")
            path.write_text(json.dumps(value), encoding="utf-8")
            record["raw_bindings"][name] = str(path)
            record["source_files"][str(path)] = a.file_sha(path)
        if unknown_worker:
            record["raw_bindings"].pop("worker_cpu")
        record["record_sha256"] = a.digest(record)
        return record

    def test_identifies_first_on_and_final_old_request_completion(self):
        result = lags.extract_worker_lags(self.record(), "cpu")
        self.assertEqual([a.exact(edge["delay_ns"]) for edge in result["edges"]], [5, 5])
        self.assertTrue(all(edge["evidence_issue"] is None for edge in result["edges"]))

    def test_zero_stop_requires_complete_old_work_and_clock_coverage(self):
        result = lags.extract_worker_lags(self.record(final_end=55), "cpu")
        self.assertEqual(a.exact(result["edges"][1]["delay_ns"]), 0)
        self.assertIn("sampling clock covers", result["edges"][1]["proof"])

    def test_missing_clock_coverage_is_unknown_not_zero(self):
        result = lags.extract_worker_lags(self.record(final_end=55, sampling_end=55), "cpu")
        self.assertIsNone(result["edges"][1]["delay_ns"])
        self.assertIn("clock", result["edges"][1]["evidence_issue"])

    def test_missing_worker_binding_is_unknown_not_zero(self):
        result = lags.extract_worker_lags(self.record(unknown_worker=True), "cpu")
        self.assertTrue(all(edge["delay_ns"] is None for edge in result["edges"]))

    def test_worker_source_tamper_is_rejected(self):
        record = self.record()
        Path(record["raw_bindings"]["worker_cpu"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            lags.extract_worker_lags(record, "cpu")

    def test_heldout_workers_cannot_estimate_delays(self):
        with self.assertRaisesRegex(ValueError, "profile-only"):
            lags.extract_worker_lags(self.record(both_heldout=True), "cpu")

    def test_profile_process_medians_are_exact_and_equally_weighted(self):
        records = [self.record(start_delay=value) for value in (5, 6, 7)]
        parameters, evidence = lags.estimate_request_delays(records, config={"enabled": True})
        self.assertEqual(parameters.cpu_start_ns, 6)
        self.assertEqual(parameters.gpu_start_ns, 6)
        self.assertEqual(parameters.cpu_stop_ns, 5)
        self.assertTrue(parameters.qualified)
        self.assertFalse(evidence["foreground_durations_used"])
        self.assertFalse(evidence["heldout_worker_intervals_used"])

    def test_unstable_delay_retains_diagnostic_numeric_value(self):
        records = [self.record(start_delay=value) for value in (5, 6, 19)]
        parameters, evidence = lags.estimate_request_delays(records, config={"enabled": True})
        self.assertEqual(parameters.cpu_start_ns, 6)
        self.assertFalse(parameters.qualified)
        cell = evidence["cells"][0]
        self.assertIn("between_process_lag_median_cv_gate_failed_or_unknown", cell["evidence_issues"])

    def test_unknown_direction_remains_none(self):
        parameters, _ = lags.estimate_request_delays([self.record(unknown_worker=True) for _ in range(3)],
                                                     config={"enabled": True})
        self.assertIsNone(parameters.cpu_start_ns)
        self.assertIsNone(parameters.cpu_stop_ns)
        self.assertFalse(parameters.qualified)

    def test_terminal_go_end_request_to_idle_is_explicit(self):
        go = dict(start_tick=0, end_tick=100, phases=[dict(phase_index=0, begin_tick=0, end_tick=100,
                                                          cpu_replay=1, gpu_replay=0)])
        edges = lags.control_edges(go, "cpu")
        self.assertEqual([(edge["tick"], edge["new"]) for edge in edges], [(0, 1), (100, 0)])
        self.assertTrue(edges[-1]["terminal_scheduled_idle"])


if __name__ == "__main__":
    unittest.main()
