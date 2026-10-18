"""Hardware-free runner regression and evidence-corruption checks."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "profiling"))
from state_calibration import run


class RunnerTests(unittest.TestCase):
    def test_precision_flags_are_set_before_child_launch(self):
        with mock.patch.dict(run.os.environ, {'NVIDIA_TF32_OVERRIDE':'1',
                'GGML_CUDA_DISABLE_GRAPHS':'0','STATE_PROBE_READY_FILE':'unrelated',
                'STATE_PROBE_STOP_FILE':'unrelated'},clear=False):
            env=run.child_environment(['runtime'],256)
        self.assertEqual(env['NVIDIA_TF32_OVERRIDE'],'0')
        self.assertEqual(env['GGML_CUDA_DISABLE_GRAPHS'],'1')
        self.assertEqual(env['STATE_PROBE_FIXED_ITERATIONS'],'256')
        self.assertTrue(env['PATH'].startswith('runtime'+run.os.pathsep))
        self.assertNotIn('STATE_PROBE_READY_FILE',env)
        self.assertNotIn('STATE_PROBE_STOP_FILE',env)

    def setUp(self):
        scratch = ROOT / "build/state-run-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    @staticmethod
    def native():
        return {"initial_ports": {"a": [1.0, 0.0]}, "ports": {"a": [1.0, 0.0]},
                "copy_values_equal": True, "native_repeat_consistency": {"passed": True},
                "post_replay_consistency": {"passed": True}}

    def test_target_selection_requires_device_and_exact_semantic_kind(self):
        program = {"bindings": [{"operation_id": "wrong", "kind": "attention_pv"},
                                {"operation_id": "pv", "kind": "attention_pv"},
                                {"operation_id": "down", "kind": "ffn_down"}],
                   "actions": [{"kind": "COMPUTE", "device": "cpu", "target_id": "wrong"},
                               {"kind": "COPY", "target_id": "unused"},
                               {"kind": "COMPUTE", "device": "gpu", "target_id": "pv"},
                               {"kind": "COMPUTE", "device": "cpu", "target_id": "down"}]}
        self.assertEqual(run.choose_target(program), 2)
        self.assertEqual(run.choose_target(program, "cpu"), 3)
        program["actions"] = program["actions"][:2]
        with self.assertRaisesRegex(ValueError, "no exact supported target"):
            run.choose_target(program)

    def test_independent_oracle_checks_both_initial_and_final_ports(self):
        reference = {"a": [1.0, 0.0]}
        self.assertTrue(run.validate_numeric(self.native(), reference)["passed"])
        for field in ("initial_ports", "ports"):
            altered = self.native()
            altered[field]["a"][0] = 1.1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "independent numerical mismatch"):
                run.validate_numeric(altered, reference)

    def test_numeric_identity_shape_nonfinite_and_native_flags_rejected(self):
        reference = {"a": [1.0, 0.0]}
        mutations = []
        for field in ("initial_ports", "ports"):
            for replacement in ({}, {"a": [1.0]}, {"a": [1.0, float("nan")]}, {"a": [1.0, float("inf")]}, {"a": [[1.0, 0.0]]}):
                native = self.native()
                native[field] = replacement
                mutations.append(native)
        for flag in ("copy_values_equal", "native_repeat_consistency", "post_replay_consistency"):
            native = self.native()
            native[flag] = False if flag == "copy_values_equal" else {"passed": False}
            mutations.append(native)
        for index, native in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                run.validate_numeric(native, reference)

    def test_ready_wait_retries_partial_json_without_accepting_it(self):
        path = self.path / "ready.json"
        path.write_text("{}", encoding="utf-8")
        process = mock.Mock()
        process.poll.return_value = None
        complete = {"format_version": "state-calibration-ready-v1", "process_id": 123}
        partial = json.JSONDecodeError("partial writer", "{", 1)
        with mock.patch.object(run, "read", side_effect=[partial, complete]) as reader, mock.patch.object(run.time, "sleep"):
            self.assertEqual(run._wait_ready(process, path), complete)
        self.assertEqual(reader.call_count, 2)

    def test_ready_wait_retries_exclusive_writer_handle(self):
        path = self.path / "ready.json"
        path.write_text("{}", encoding="utf-8")
        process = mock.Mock()
        process.poll.return_value = None
        complete = {"format_version": "state-calibration-ready-v1", "process_id": 123}
        with mock.patch.object(run, "read", side_effect=[PermissionError("writer still open"), complete]), mock.patch.object(run.time, "sleep"):
            self.assertEqual(run._wait_ready(process, path), complete)

    def test_failed_popen_closes_both_owned_log_handles(self):
        stdout, stderr = mock.Mock(), mock.Mock()
        out_handle, err_handle = mock.Mock(), mock.Mock()
        stdout.open.return_value, stderr.open.return_value = out_handle, err_handle
        with mock.patch.object(run.subprocess, "Popen", side_effect=OSError("launch unavailable")), self.assertRaises(OSError):
            run.launch(["missing.exe"], {}, stdout, stderr, 3)
        out_handle.close.assert_called_once()
        err_handle.close.assert_called_once()

    def test_source_inventory_expands_native_build_bound_files(self):
        binary = self.path / "probe.exe"
        dependency = self.path / "pinned-ggml-helper.cpp"
        binary.write_bytes(b"binary")
        dependency.write_bytes(b"original helper")
        run.save(self.path / "build-manifest.json", {"returncode": 0, "binary_sha256": run.sha(binary), "bound_inputs_unchanged": True,
                                                   "bound_files": {str(dependency): run.sha(dependency)}})
        inventory = run.sources(binary, [])
        self.assertEqual(inventory[str(dependency.resolve())], run.sha(dependency))
        dependency.write_bytes(b"changed helper")
        with self.assertRaisesRegex(ValueError, "changed"):
            run.verify_sources({"sources": inventory})

    def test_source_inventory_rejects_stale_native_build_binding(self):
        binary = self.path / "probe.exe"
        dependency = self.path / "pinned-ggml-helper.cpp"
        binary.write_bytes(b"binary")
        dependency.write_bytes(b"changed helper")
        run.save(self.path / "build-manifest.json", {"returncode": 0, "binary_sha256": run.sha(binary), "bound_inputs_unchanged": True,
                                                   "bound_files": {str(dependency): "0" * 64}})
        with self.assertRaises(ValueError):
            run.sources(binary, [])

    def test_profile_collect_passes_flat_sha_bindings_and_writes_freeze_barrier(self):
        manifest = {"sources": {"source.py": "a" * 64}, "jobs": []}
        run.save(self.path / "manifest.json", manifest)

        def fitted(rows, source_bindings):
            self.assertEqual(rows, [])
            self.assertTrue(source_bindings)
            for name, digest in source_bindings.items():
                self.assertIsInstance(name, str)
                self.assertIsInstance(digest, str)
                self.assertEqual(len(digest), 64)
            return {"frozen": True}

        with mock.patch.object(run, "verify_sources"), mock.patch.object(run, "measurement_lock", return_value=nullcontext()), mock.patch("state_calibration.model.fit_model", side_effect=fitted):
            self.assertEqual(run.collect(self.path, "profile"), [])
        barrier = run.read(self.path / "profile-barrier.json")
        self.assertEqual(barrier["model_sha256"], run.sha(self.path / "frozen-model.json"))
        self.assertEqual(barrier["profile_rows_sha256"], run.sha(self.path / "profile-rows.json"))
        self.assertTrue(barrier["all_sources_finished"])

    @staticmethod
    def protocol_fixture(mode="isolated", instrumentation="stamped"):
        from highlevel_accuracy.bridge import RUNTIME_FLAGS
        action = {"kind": "COMPUTE", "device": "gpu", "target_id": "pv"}
        program = {"program_sha256": "b" * 64, "bindings": [{"operation_id": "pv", "kind": "attention_pv"}], "actions": [action]}
        job = {"mode": mode, "instrumentation": instrumentation}
        protocol = {"blocks": 1, "fixed_iterations_per_block": 2, "minimum_block_ms": 200}
        clock = {"clock_id": "synthetic_monotonic", "frequency_hz": 1000}
        iterations = [{"iteration": 0, "begin_tick": 1100, "end_tick": 1200, "elapsed_ns": 100000000},
                      {"iteration": 1, "begin_tick": 1300, "end_tick": 1400, "elapsed_ns": 100000000}]
        actions = [{**item, "action_index": 0} for item in iterations] if instrumentation == "stamped" else []
        native = {"format_version": "state-calibration-native-v1", "program_sha256": program["program_sha256"], "mode": mode,
                  "target_action_index": 0, "process_id": 123, "actions": [action], "target": {"action": action},
                  "observed_runtime_flags": dict(RUNTIME_FLAGS), "clock": clock, "graceful_stop_requested": False,
                  "execution_boundary":"legacy_double_sync",
                  "iteration_policy": "fixed_count", "requested_iterations_per_block": 2, "requested_blocks": 1, "full_fixed_count": True,
                  "blocks": [{"block_index": 0, "begin_tick": 1000, "end_tick": 1600, "wall_elapsed_ns": 600000000,
                              "completed_work_ns": 200000000, "iterations": iterations, "action_intervals": actions, "iteration_count": 2,
                              "target_diagnostics":[],
                              "minimum_elapsed_satisfied": True, "fixed_iteration_count_satisfied": True}]}
        return native, program, job, protocol, clock

    def test_native_protocol_accepts_each_declared_execution_instrumentation_mode(self):
        for mode in ("isolated", "mixed"):
            for instrumentation in ("stamped", "bare"):
                with self.subTest(mode=mode, instrumentation=instrumentation):
                    self.assertTrue(run.verify_native_protocol(*self.protocol_fixture(mode, instrumentation), process_id=123))

    def test_native_protocol_rejects_wrong_identity_missing_samples_and_forged_intervals(self):
        mutations = [lambda x: x.update(program_sha256="c" * 64), lambda x: x.update(mode="mixed"),
                     lambda x: x.update(process_id=999), lambda x: x.update(target_action_index=2),
                     lambda x: x["observed_runtime_flags"].update(matmul_precision="GGML_PREC_DEFAULT"),
                     lambda x: x["clock"].update(frequency_hz=999), lambda x: x.update(graceful_stop_requested=True),
                     lambda x: x["blocks"][0]["iterations"].pop(),
                     lambda x: x["blocks"][0]["iterations"][1].update(begin_tick=1000),
                     lambda x: x["blocks"][0]["iterations"][0].update(elapsed_ns=100),
                     lambda x: x["blocks"][0]["action_intervals"].pop(),
                     lambda x: x["blocks"][0]["action_intervals"][0].update(action_index=99),
                     lambda x: x["blocks"][0]["action_intervals"][0].update(end_tick=1500)]
        for index, mutate in enumerate(mutations):
            native, *rest = self.protocol_fixture()
            mutate(native)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                run.verify_native_protocol(native, *rest, process_id=123)

    def test_native_protocol_reconstructs_block_duration_and_completed_work(self):
        for field, wrong in (("wall_elapsed_ns", 1), ("completed_work_ns", 1)):
            native, *rest = self.protocol_fixture()
            native["blocks"][0][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.verify_native_protocol(native, *rest, process_id=123)


if __name__ == "__main__":
    unittest.main()
