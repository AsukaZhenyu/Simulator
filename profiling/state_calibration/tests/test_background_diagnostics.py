"""Synthetic interval/evidence tests; no telemetry or native execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "profiling"))
from state_calibration import diagnose_background as diagnostic


class BackgroundDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "build/background-diagnostics-tests"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def test_clipping_uses_full_durations_for_cv_and_marks_censoring(self):
        report = diagnostic.bracket_coverage([(0, 5), (7, 20)], [(3, 10)], 1_000_000_000)
        self.assertEqual(report["covered_ticks"], 5)
        self.assertEqual(report["uncovered_ticks"], 2)
        self.assertEqual(report["overlapping_invocation_full_durations"]["median_ns"], 9)
        self.assertEqual(report["left_censored_invocations"], 1)
        self.assertEqual(report["right_censored_invocations"], 1)
        self.assertEqual(report["complement_gap_durations"]["maximum_ns"], 2)

    def test_edges_and_disjoint_windows_do_not_double_count(self):
        report = diagnostic.bracket_coverage([(2, 4), (6, 8)], [(0, 5), (5, 10)], 1_000_000_000)
        self.assertEqual(report["coverage_fraction_exact"], {"numerator": 2, "denominator": 5})
        self.assertEqual(report["unique_overlapping_invocations"], 2)
        self.assertEqual(report["positive_intersection_pairs"], 2)
        self.assertEqual(report["leading_edge_gaps"]["maximum_ns"], 2)
        self.assertEqual(report["trailing_edge_gaps"]["maximum_ns"], 2)

    def test_malformed_overlapping_or_reversed_intervals_are_rejected(self):
        for background, windows in (([(0, 3), (2, 4)], [(0, 5)]), ([(2, 1)], [(0, 5)]),
                                    ([(0, 1)], [(0, 3), (2, 4)]), ([(True, 3)], [(0, 5)])):
            with self.subTest(background=background, windows=windows), self.assertRaises(diagnostic.EvidenceError):
                diagnostic.bracket_coverage(background, windows, 1000)

    def fixture(self, directory):
        job = {"id": "profile-0-case-gpu-isolated", "split": "profile", "case_id": "case", "repeat": 0,
               "condition": "gpu_replay_load", "mode": "isolated", "instrumentation": "stamped"}
        directory = directory / job["id"]
        directory.mkdir()
        clock = {"clock_id": "synthetic_monotonic", "frequency_hz": 1_000_000_000}
        action = {"kind": "COMPUTE", "device": "gpu", "target_id": "pv"}

        def native(pid, begin, end, left, right):
            invocation = {"iteration": 0, "begin_tick": left, "end_tick": right, "elapsed_ns": right - left}
            return {"format_version": "state-calibration-native-v1", "clock": clock, "process_id": pid, "mode": "isolated",
                    "program_sha256": "a" * 64, "observed_runtime_flags": {"precision": "F32"}, "target_action_index": 0,
                    "actions": [action], "target": {"action": action},
                    "blocks": [{"block_index": 0, "begin_tick": begin, "end_tick": end, "wall_elapsed_ns": end - begin,
                                "completed_work_ns": right - left, "iteration_count": 1, "iterations": [invocation],
                                "action_intervals": [{**invocation, "action_index": 0}]}]}

        main = native(123, 10, 30, 12, 28)
        background = native(456, 0, 40, 5, 35)
        artifacts = {}
        for name, value in (("native.json", main), ("background-native.json", background)):
            content = (json.dumps(value) + "\n").encode()
            (directory / name).write_bytes(content)
            artifacts[name] = hashlib.sha256(content).hexdigest()
        row = {"job": job, "status": "measured", "returncode": 0, "process_id": 123, "native_clock": clock,
               "summary": {"native_numeric_passed": True, "condition_verified": True, "native_sha256": artifacts["native.json"]},
               "background": {"returncode": 0, "pid": 456}, "artifacts_sha256": artifacts, "raw_file_sha256": dict(artifacts)}
        (directory / "execution.json").write_text(json.dumps(row), encoding="utf-8")
        return directory, job, row

    def test_bound_evidence_covers_main_block_and_selected_gpu_action(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            directory, job, _ = self.fixture(Path(temporary))
            report = diagnostic.diagnose_job(directory, job)
            self.assertEqual(report["status"], "observed")
            self.assertEqual(report["all_main_blocks"]["coverage_fraction"], 1)
            self.assertEqual(report["selected_GPU_action_windows"]["coverage_fraction"], 1)
            self.assertEqual(set(report["inputs_sha256"]), {"execution.json", "native.json", "background-native.json"})
            self.assertNotIn(str(directory), json.dumps(report))

    def test_clock_and_hash_corruption_are_unknown_not_zero_duty(self):
        for corruption in ("clock", "hash"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
                directory, job, row = self.fixture(Path(temporary))
                path = directory / "background-native.json"
                native = json.loads(path.read_text())
                native["clock"]["frequency_hz"] = 123
                content = json.dumps(native).encode()
                path.write_bytes(content)
                if corruption == "clock":
                    digest = hashlib.sha256(content).hexdigest()
                    row["artifacts_sha256"]["background-native.json"] = digest
                    row["raw_file_sha256"]["background-native.json"] = digest
                    (directory / "execution.json").write_text(json.dumps(row), encoding="utf-8")
                report = diagnostic.diagnose_job(directory, job)
                self.assertEqual(report["status"], "unknown")
                self.assertNotIn("all_main_blocks", report)

    def test_missing_or_failed_background_remains_unknown_and_baseline_is_not_applicable(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
                directory, job, row = self.fixture(Path(temporary))
                if failed:
                    row["background"]["returncode"] = 2
                    (directory / "execution.json").write_text(json.dumps(row), encoding="utf-8")
                else:
                    (directory / "background-native.json").unlink()
                self.assertEqual(diagnostic.diagnose_job(directory, job)["status"], "unknown")
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            directory, job, row = self.fixture(Path(temporary))
            job["condition"] = "baseline"
            row["job"] = job
            row.pop("background")
            row["artifacts_sha256"].pop("background-native.json")
            row["raw_file_sha256"].pop("background-native.json")
            (directory / "execution.json").write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(diagnostic.diagnose_job(directory, job)["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
