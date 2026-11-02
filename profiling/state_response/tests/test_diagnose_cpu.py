"""Offline fixtures for completed-call request-edge classification."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "diagnose_cpu.py"
SPEC = importlib.util.spec_from_file_location("state_cpu_diagnose", SOURCE)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class CpuAuditTests(unittest.TestCase):
    def fixture(self, rows):
        native = {"process_id": 1, "clock": {"frequency_hz": 1000}, "sampling_end_tick": 45,
                  "blocks": [{"iterations": rows}]}
        phases = [{"begin_tick": a, "end_tick": b, "cpu_replay": cpu, "gpu_replay": 0}
                  for a, b, cpu in ((0, 10, 0), (10, 20, 1), (20, 30, 0), (30, 40, 1))]
        return native, {"start_tick": 0, "end_tick": 40, "phases": phases}

    def test_on_acceptance_and_retained_off_tail_are_separate(self):
        native, go = self.fixture([{"enabled_check_tick": 12, "begin_tick": 13, "end_tick": 23},
                                   {"enabled_check_tick": 35, "begin_tick": 35, "end_tick": 45}])
        result = DIAG.worker_edges(native, go)
        on, off, on2, off2 = result["edges"]
        self.assertEqual(on["first_acceptance_delay_ns"], 2_000_000)
        self.assertEqual(on["first_call_begin_delay_ns"], 3_000_000)
        self.assertEqual(off["retained_busy_tail_after_off_ns"], 3_000_000)
        self.assertEqual(off2["retained_busy_tail_after_off_ns"], 5_000_000)
        self.assertIsNone(off["stop_response_latency_ns"])
        self.assertEqual(result["acceptance_outside_cpu_on_request"], [])

    def test_unaccepted_on_window_is_censored_not_zero_delay(self):
        native, go = self.fixture([{"enabled_check_tick": 35, "begin_tick": 35, "end_tick": 45}])
        edge = DIAG.worker_edges(native, go)["edges"][0]
        self.assertTrue(edge["acceptance_censored"])
        self.assertIsNone(edge["first_acceptance_delay_ns"])
        self.assertEqual(edge["observed_request_window_ns"], 10_000_000)

    def test_accepting_while_off_is_flagged_not_silently_dropped(self):
        bad = {"enabled_check_tick": 22, "begin_tick": 22, "end_tick": 25}
        native, go = self.fixture([bad])
        result = DIAG.worker_edges(native, go)
        self.assertEqual(result["acceptance_outside_cpu_on_request"], [bad])
        self.assertEqual(result["edges"][1]["accepted_calls_while_requested_off"], 1)

    def test_singleton_cv_unknown_and_exact_boundary_containment(self):
        self.assertIsNone(DIAG.summary([100])["population_cv"])
        segments = [{"begin_tick": 0, "end_tick": 10, "calibration_begin_tick": 3, "state": {}}]
        self.assertEqual(DIAG.classify(3, 10, segments)[1], "steady")
        self.assertEqual(DIAG.classify(2, 10, segments)[1], "within_state_before_settle")
        self.assertEqual(DIAG.classify(3, 11, segments)[1], "crossing_or_outside_known_segments")

    def test_call_wall_overlap_not_inferred_cpu_time(self):
        rows = [{"begin_tick": 0, "end_tick": 5}, {"begin_tick": 8, "end_tick": 12}]
        self.assertEqual(DIAG.busy_overlap(3, 10, rows), 4)
        self.assertEqual(DIAG.busy_overlap(5, 8, rows), 0)
        counters = DIAG.cpu_envelope({"cpu_diagnostics": {"sampling_wall_ns": 100,
                                                                          "main_thread_cpu_ns": 20,
                                                                          "process_cpu_ns": 40}})
        self.assertEqual(counters["non_main_cpu_ns"], 20)
        self.assertFalse(counters["per_cpu_action_cpu_time_available"])

    def test_measured_job_is_audited_and_failed_job_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs/profile-r0-m0"
            (job / "main").mkdir(parents=True)
            (job / "cpu-worker").mkdir()
            program = root / "program.json"
            def save(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            save(program, {"actions": [{"kind": "COMPUTE", "device": "cpu", "target_id": "projection"}],
                           "bindings": [{"operation_id": "projection", "kind": "projection"}]})
            worker, go = self.fixture([{"enabled_check_tick": 12, "begin_tick": 13, "end_tick": 23}])
            save(job / "execution.json", {"status": "measured"})
            save(job / "go.json", go)
            save(job / "events.json", {"known_segments": [
                {"begin_tick": 10, "end_tick": 20, "calibration_begin_tick": 11,
                 "state": {"cpu_replay": 1, "gpu_replay": 0, "host_memory_bytes": 0, "device_memory_bytes": 0}}]})
            save(job / "main/process.json", {"command": ["unused", str(program)], "affinity": {"observed_mask": 3}})
            save(job / "cpu-worker/process.json", {"affinity": {"observed_mask": 3}})
            save(job / "cpu-worker/result.json", worker)
            save(job / "main/result.json", {"process_id": 2, "instrumentation": "stamped", "blocks": [
                {"action_intervals": [{"iteration": 0, "action_index": 0, "begin_tick": 14,
                                       "end_tick": 16, "elapsed_ns": 2_000_000}]}]})
            bad = root / "jobs/profile-r1-m0"
            bad.mkdir()
            save(bad / "execution.json", {"status": "failed", "error_type": "TimeoutError", "error": "startup"})
            with patch.object(DIAG, "topology", return_value={"known": False}):
                result = DIAG.audit(root)
            self.assertEqual(len(result["successful_jobs"]), 1)
            self.assertEqual(len(result["failed_jobs"]), 1)
            self.assertEqual(result["steady_cpu_action_groups"][0]["minimum_steady_samples_per_pid"], 1)
            self.assertIsNone(result["steady_cpu_action_groups"][0]["between_pid_medians_ns"]["population_cv"])


if __name__ == "__main__":
    unittest.main()
