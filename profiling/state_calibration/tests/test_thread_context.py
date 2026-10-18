"""Thread-control provenance and minimum-wall-time protocol tests, no hardware."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "profiling"))
from state_calibration import thread_context as diagnostic


class ThreadContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = ROOT / "build/thread-context-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        from highlevel_accuracy.cases import build_case
        from highlevel_accuracy.bridge import export_plan
        cls.original = export_plan(build_case(cls.directory / "case", devices=("cpu", "cpu", "cpu"), threads=2).compile(), seed=41)

    def test_thread_intervention_preserves_all_other_fields_and_reference(self):
        from highlevel_accuracy.bridge import reference_ports, validate_program
        reference = reference_ports(self.original)
        for threads in (1, 2):
            with self.subTest(threads=threads):
                variant = diagnostic.thread_variant(self.original, threads)
                validate_program(variant)
                self.assertEqual(variant["physical"]["threads"], threads)
                self.assertEqual(reference_ports(variant), reference)
                restored = deepcopy(variant)
                for key in ("program_sha256",):
                    restored[key] = self.original[key]
                restored["physical"]["threads"] = self.original["physical"]["threads"]
                restored["identity"]["physical"] = self.original["identity"]["physical"]
                self.assertEqual(restored, self.original)

    def test_unsupported_thread_counts_never_coerce_or_fallback(self):
        for value in (0, 3, -1, True, False, "1", 1.0, 2.0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.thread_variant(self.original, value)

    @staticmethod
    def native_fixture(counts=(1, 2)):
        from highlevel_accuracy.bridge import RUNTIME_FLAGS
        action = {"kind": "COMPUTE", "device": "gpu", "target_id": "pv"}
        program = {"program_sha256": "a" * 64, "physical": {"threads": 2}, "actions": [action],
                   "bindings": [{"operation_id": "pv", "kind": "attention_pv"}]}
        job = {"mode": "isolated", "threads": 2}
        protocol = {"blocks": len(counts), "warmup": 20, "minimum_block_ms": 200}
        clock = {"clock_id": "synthetic_monotonic", "frequency_hz": 1_000_000_000}
        blocks = []
        for index, count in enumerate(counts):
            begin = 1000 + index * 300000000
            cursor = begin + 1000
            iterations, actions = [], []
            for iteration in range(count):
                duration = 200000000 // count
                item = {"iteration": iteration, "begin_tick": cursor, "end_tick": cursor + duration, "elapsed_ns": duration}
                iterations.append(item)
                actions.append({**item, "action_index": 0})
                cursor += duration + 1000
            end = cursor + 1000
            blocks.append({"block_index": index, "begin_tick": begin, "end_tick": end, "wall_elapsed_ns": end - begin,
                           "minimum_elapsed_satisfied": True, "fixed_iteration_count_satisfied": None, "target_diagnostics": [],
                           "iterations": iterations, "iteration_count": count, "completed_work_ns": sum(item["elapsed_ns"] for item in iterations),
                           "action_intervals": actions})
        native = {"format_version": "state-calibration-native-v1", "program_sha256": program["program_sha256"], "process_id": 123,
                  "mode": "isolated", "target_action_index": 0, "actions": deepcopy(program["actions"]), "target": {"action": deepcopy(action)},
                  "observed_runtime_flags": dict(RUNTIME_FLAGS), "execution_boundary": "legacy_double_sync", "clock": dict(clock),
                  "iteration_policy": "minimum_wall_time", "requested_iterations_per_block": None, "full_fixed_count": None,
                  "graceful_stop_requested": False, "requested_blocks": len(counts), "warmup_iterations": 20, "minimum_block_ms": 200,
                  "instrumentation": "per_iteration_and_per_action_host_completed_work", "diagnostic_begin_tick": 0,
                  "warmup_begin_tick": 100, "warmup_end_tick": 200, "ready_tick": 300, "blocks": blocks,
                  "diagnostic_end_tick": blocks[-1]["end_tick"] + 1000}
        return native, program, job, protocol, clock

    def test_minimum_wall_verifier_allows_unequal_invocation_counts(self):
        diagnostic.verify_native(*self.native_fixture(), process_id=123)

    def test_minimum_wall_verifier_rejects_fixed_protocol_and_corrupt_evidence(self):
        mutations = [lambda x: x.update(iteration_policy="fixed_count"), lambda x: x.update(requested_iterations_per_block=256),
                     lambda x: x.update(process_id=999), lambda x: x["blocks"][0].update(wall_elapsed_ns=1),
                     lambda x: x["blocks"][0].update(completed_work_ns=1),
                     lambda x: x["blocks"][0]["action_intervals"][0].update(elapsed_ns=1),
                     lambda x: x["blocks"][0]["action_intervals"].clear(), lambda x: x.update(graceful_stop_requested=True)]
        for index, mutate in enumerate(mutations):
            native, *rest = self.native_fixture()
            mutate(native)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                diagnostic.verify_native(native, *rest, process_id=123)

    def test_summary_keeps_failed_process_and_never_claims_calibration(self):
        rows = [{"job": {"id": f"r{repeat}-{mode}-{threads}", "repeat": repeat, "mode": mode, "threads": threads},
                 "status": "measured", "summary": {"target_median_ns": threads * 100, "target_cv": 0.1}}
                for repeat in range(3) for mode in ("isolated", "mixed") for threads in (1, 2)]
        report = diagnostic.summarize(rows)
        self.assertTrue(report["all_twelve_successful"])
        self.assertTrue(all(pair["threads_2_over_1_target_median_ratio"] == 2 for pair in report["paired_comparisons"]))
        rows[0]["status"] = "failed"
        report = diagnostic.summarize(rows)
        group = next(item for item in report["groups"] if item["mode"] == "isolated" and item["threads"] == 1)
        self.assertIsNone(group["diagnostic_median_of_process_medians_ns"])
        self.assertEqual(group["failed_job_ids"], [rows[0]["job"]["id"]])
        self.assertFalse(report["all_twelve_successful"])
        self.assertFalse(report["cost_fitted"])
        self.assertFalse(report["performance_acceptance_implied"])

    def test_launch_contract_unsets_fixed_count_and_preserves_legacy_boundary(self):
        native, program, _, _, _ = self.native_fixture()
        manifest = {"programs": {"p": {"path": "synthetic-program", "reference": "synthetic-reference", "sha256": "b" * 64}},
                    "runtime_dirs": [], "binary": "synthetic-probe", "protocol": {"warmup": 20, "blocks": 3, "minimum_block_ms": 200,
                    "interval_ms": 50, "affinity_mask": 3, "process_timeout_seconds": 180}}
        job = {"id": "launch-contract", "program_key": "p", "mode": "isolated", "threads": 2}
        sampler = mock.Mock()
        sampler.start.return_value = sampler
        sampler.metadata.return_value = {"stopped": True}
        with mock.patch.object(diagnostic.run, "read", side_effect=[program, {}]), \
                mock.patch.object(diagnostic.run, "child_environment", return_value={"STATE_PROBE_FIXED_ITERATIONS": "256"}), \
                mock.patch.object(diagnostic, "DeviceTelemetrySampler", return_value=sampler), \
                mock.patch.object(diagnostic.run, "launch", side_effect=RuntimeError("mock launch only")) as launch:
            row = diagnostic._measure(self.directory / "mock-launch", manifest, job)
        self.assertEqual(row["status"], "failed")
        command, env, _, _, affinity = launch.call_args.args
        self.assertEqual(len(command), 9)
        self.assertEqual(command[2:8], ["isolated", "0", "20", "3", "200", "stamped"])
        self.assertNotIn("STATE_PROBE_FIXED_ITERATIONS", env)
        self.assertEqual(env["STATE_PROBE_EXECUTION_BOUNDARY"], "legacy_double_sync")
        self.assertEqual(affinity, 3)


if __name__ == "__main__":
    unittest.main()
