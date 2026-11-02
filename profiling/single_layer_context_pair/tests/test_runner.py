"""Synthetic paired-protocol guards; these tests never launch a native process."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from single_layer_context_pair import runner


class PairTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3] / "build/context-pair-runner-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.program = {"program_sha256": "b" * 64, "seed": 42,
                        "identity": {"graph": "same", "initial_values": "42"},
                        "actions": [{"kind": "COMPUTE", "target_id": "attn.pv", "device": "gpu"}, {"kind": "ADVANCE"}],
                        "bindings": [{"operation_id": "attn.pv", "kind": "attention_pv"}]}
        self.protocol = {"samples_per_block": 3, "warmup_per_block": 30}
        self.job = {"run_id": "synthetic-pair", "pair_kind": "context_vs_isolated", "order": "ABBA",
                    "target_action_index": 0, "seed": 42, "poll": "default"}
        self.oracle = patch.object(runner, "verify_native", return_value={"all_ports_passed": True})
        self.oracle.start()
        self.addCleanup(self.oracle.stop)

    def native(self, *, kind="context_vs_isolated", order="ABBA"):
        clock = {"clock_id": "synthetic-ns", "frequency_hz": 10**9, "cross_process_monotonic_clock": True}
        pool = {"requested_poll": "default", "effective_params": {"n_threads": 2, "poll": 50}}
        raw = {"format_version": "single-layer-context-pair-native-v2", "process_id": 123,
               "program_sha256": self.program["program_sha256"], "seed": 42,
               "target_action_index": 0, "target_action": deepcopy(self.program["actions"][0]),
               "pair_kind": kind, "block_order": order, "samples_per_block": 3, "warmup_per_block": 30,
               "A_label": runner.MODES[kind]["A"], "B_label": runner.MODES[kind]["B"],
               "same_PID_Executor_program_and_pool_for_all_blocks": True,
               "threadpool": pool, "native_clock": clock, "begin_tick": 0, "end_tick": 5000,
               "diagnostic_only": True, "usable_for_parameter_fit": False,
               "usable_for_heldout_acceptance": False, "blocks": []}
        for index, label in enumerate(order):
            start = index * 1000 + 100
            mode = runner.MODES[kind][label]
            native = {"format_version": "single-layer-calibration-native-v1", "process_id": 123,
                      "program_sha256": self.program["program_sha256"], "actions": deepcopy(self.program["actions"]),
                      "mode": mode, "samples": 3, "warmup": 30, "native_clock": deepcopy(clock),
                      "threadpool": deepcopy(pool), "components": [], "iterations": [], "total_samples_ns": [],
                      "execution_contract": "prebuilt_synchronized_GGML_actions;reset_excluded;preallocated_trace_buffer_v1",
                      "verification_native": {"observed_runtime_flags": {"NVIDIA_TF32_OVERRIDE": "0"}},
                      "diagnostic_only": True, "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}
            if mode == "isolated-components":
                native["component_subset"] = True
                native["components"] = [{"action_index": 0, "action": deepcopy(self.program["actions"][0]),
                    "samples_ns": [10] * 3, "workspace_bytes": 0,
                    "intervals": [{"sample": number, "begin_tick": start + 100 + number * 100,
                                   "end_tick": start + 110 + number * 100, "elapsed_ns": 10} for number in range(3)]}]
            else:
                for number in range(3):
                    begin = start + 100 + number * 100
                    intervals = [{"action_index": 0, "kind": "COMPUTE", "begin_tick": begin + 1,
                                  "end_tick": begin + 21, "begin_offset_ns": 1, "end_offset_ns": 21, "elapsed_ns": 20},
                                 {"action_index": 1, "kind": "ADVANCE", "begin_tick": begin + 22,
                                  "end_tick": begin + 24, "begin_offset_ns": 22, "end_offset_ns": 24, "elapsed_ns": 2}]
                    native["iterations"].append({"iteration": number, "begin_tick": begin,
                        "end_tick": begin + 25, "elapsed_ns": 25,
                        "action_intervals": intervals if mode == "context-trace" else []})
                    native["total_samples_ns"].append(25)
            raw["blocks"].append({"block_index": index, "label": label,
                                  "block_begin_tick": start, "block_end_tick": start + 900,
                                  "block_wall_ns": 900, "raw": native})
        return raw

    def check(self, raw=None, job=None):
        return runner.validate_native(self.program, raw or self.native(), job or self.job,
                                      self.protocol, process_id=123)

    def test_context_pair_checks_four_independent_numerical_outputs(self):
        with patch.object(runner, "verify_native", return_value={"all_ports_passed": True}) as oracle:
            result = self.check()
        self.assertEqual(oracle.call_count, 4)
        self.assertEqual(result["B_to_A_ratio"], 2)
        self.assertEqual(result["independently_verified_blocks"], 4)
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["usable_for_parameter_fit"])
        self.assertFalse(result["usable_for_heldout_acceptance"])

    def test_instrumentation_pair_compares_whole_invocation_only(self):
        job = {**self.job, "pair_kind": "instrumentation", "order": "BAAB"}
        result = self.check(self.native(kind="instrumentation", order="BAAB"), job)
        self.assertEqual(result["B_to_A_ratio"], 1)
        self.assertTrue(all(block["metric"] == "whole_invocation" for block in result["blocks"]))

    def test_wrong_launched_pid_or_seed_rejected(self):
        for key in ("process_id", "seed"):
            raw = self.native()
            raw[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "PID/program/seed"):
                self.check(raw)

    def test_block_relabel_or_reorder_rejected(self):
        raw = self.native()
        raw["blocks"][1]["label"] = "A"
        with self.assertRaisesRegex(ValueError, "index/label"):
            self.check(raw)

    def test_wrong_pair_kind_or_mode_rejected(self):
        raw = self.native()
        raw["pair_kind"] = "instrumentation"
        with self.assertRaisesRegex(ValueError, "context/order/target"):
            self.check(raw)
        raw = self.native()
        raw["blocks"][0]["raw"]["mode"] = "bare-replay"
        with self.assertRaisesRegex(ValueError, "label/native mode"):
            self.check(raw)

    def test_overlapping_block_or_outside_sample_rejected(self):
        raw = self.native()
        raw["blocks"][1]["block_begin_tick"] = 0
        with self.assertRaisesRegex(ValueError, "clocks overlap"):
            self.check(raw)
        raw = self.native()
        interval = raw["blocks"][0]["raw"]["components"][0]["intervals"][0]
        interval.update(begin_tick=0, end_tick=10)
        with self.assertRaisesRegex(ValueError, "outside its paired block|sample clocks overlap"):
            self.check(raw)

    def test_poll_and_block_pool_are_bound(self):
        raw = self.native()
        raw["threadpool"]["requested_poll"] = "0"
        with self.assertRaisesRegex(ValueError, "polling contract"):
            self.check(raw)
        raw = self.native()
        raw["blocks"][2]["raw"]["threadpool"]["effective_params"]["poll"] = 0
        with self.assertRaisesRegex(ValueError, "clock/pool"):
            self.check(raw)

    def test_block_numeric_failure_never_becomes_success(self):
        with patch.object(runner, "verify_native", side_effect=ValueError("independent numerical mismatch")):
            with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                self.check()

    def test_missing_target_or_count_rejected(self):
        raw = self.native()
        raw["blocks"][0]["raw"]["components"][0]["action_index"] = 1
        with self.assertRaisesRegex(ValueError, "target subset"):
            self.check(raw)
        raw = self.native()
        raw["blocks"].pop()
        with self.assertRaisesRegex(ValueError, "four counterbalanced"):
            self.check(raw)

    def test_any_attempt_to_relabel_as_calibration_is_rejected(self):
        raw = self.native()
        raw["usable_for_parameter_fit"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic boundary"):
            self.check(raw)

    def test_prepare_predeclares_twenty_four_processes_without_hardware(self):
        campaign = self.root / "old"
        campaign.mkdir()
        (campaign / "manifest.json").write_text("{}", encoding="utf-8")
        cases, programs = [], {}
        for case in ("case-a", "case-b"):
            keys = {}
            for repeat in range(3):
                key = f"{case}-profile-{repeat}"
                path = campaign / (key + ".json")
                path.write_text(json.dumps(self.program), encoding="utf-8")
                programs[key] = {"path": str(path), "sha256": runner.driver.sha(path), "seed": 42}
                keys[f"profile-{repeat}"] = key
            cases.append({"id": case, "program_keys": keys})
        base = {"protocol": {"independent_processes": 3}, "runtime_dirs": [], "cases": cases, "programs": programs}
        args = SimpleNamespace(out=self.root / "new", campaign=campaign, executable=self.root / "fake.exe", warmup=30, samples=101)
        with patch.object(runner.driver, "load_manifest", return_value=base), \
                patch.object(runner.driver, "verify_current"), patch.object(runner, "own_inventory", return_value={}), \
                patch.object(runner, "verify_sources"), patch.object(runner.subprocess, "Popen") as popen:
            result = runner.prepare(args)
        manifest = runner.load_manifest(args.out)
        self.assertEqual(result["jobs"], 24)
        self.assertFalse(result["hardware_executed"])
        self.assertEqual(len({row["run_id"] for row in manifest["jobs"]}), 24)
        self.assertEqual({row["pair_kind"] for row in manifest["jobs"]}, set(runner.KINDS))
        self.assertEqual({row["order"] for row in manifest["jobs"]}, {"ABBA", "BAAB"})
        popen.assert_not_called()

    def test_manifest_changed_or_diagnostic_flag_lost_is_rejected(self):
        manifest = {"schema": runner.SCHEMA, "jobs": [], "diagnostic_only": True,
                    "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}
        manifest["manifest_sha256"] = runner.driver.content_sha(manifest)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(runner.load_manifest(self.root)["jobs"], [])
        manifest["diagnostic_only"] = False
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            runner.load_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
