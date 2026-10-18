"""Small synthetic protocol fixtures; no GPU, device query or native launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from profiling.state_calibration import model, run, telemetry, verify as v
from highlevel_accuracy.bridge import BOUNDARY, RUNTIME_FLAGS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def program(seed):
    actions = [{"kind": "COMPUTE", "device": "gpu", "target_id": "pv"},
               {"kind": "COMPUTE", "device": "cpu", "target_id": "down"}]
    bindings = [{"operation_id": "pv", "kind": "attention_pv"}, {"operation_id": "down", "kind": "ffn_down"}]
    return {"seed": seed, "program_sha256": f"synthetic-identity-{seed}", "actions": actions, "bindings": bindings,
            "graph": {"workload": {"operations": [{"id": "pv", "inputs": ["x"], "output": "y"}, {"id": "down", "inputs": ["x"], "output": "z"}],
                                     "tensors": [{"id": name, "storage_bytes": 4} for name in ("x", "y", "z")]}}}


REFERENCE = {"y": [1.0], "z": [2.0]}
CLOCK = {"clock_id": "windows_qpc", "frequency_hz": 1000}


def native(prog, pid):
    iterations = [{"iteration": index, "begin_tick": 1001 + index * 100, "end_tick": 1101 + index * 100, "elapsed_ns": 100_000_000} for index in range(3)]
    actions = [{"iteration": index, "action_index": 0, "begin_tick": 1002 + index * 100, "end_tick": 1052 + index * 100, "elapsed_ns": 50_000_000} for index in range(3)]
    return {"format_version": "state-calibration-native-v1", "program_sha256": prog["program_sha256"], "boundary_contract": BOUNDARY,
            "actions": prog["actions"], "target": {"action": prog["actions"][0], "binding": prog["bindings"][0]},
            "target_action_index": 0, "mode": "isolated", "process_id": pid,
            "observed_runtime_flags": RUNTIME_FLAGS, "clock": CLOCK, "graceful_stop_requested": False,
            "iteration_policy": "fixed_count", "requested_iterations_per_block": 3, "full_fixed_count": True,
            "warmup_iterations": 20, "warmup_begin_tick": 970, "warmup_end_tick": 990, "minimum_block_ms": 200,
            "instrumentation": "per_iteration_and_per_action_host_completed_work", "execution_boundary": "legacy_double_sync",
            "blocks": [{"block_index": 0, "begin_tick": 1000, "end_tick": 1302, "wall_elapsed_ns": 302_000_000,
                        "completed_work_ns": 300_000_000, "iteration_count": 3, "iterations": iterations,
                        "action_intervals": actions, "target_diagnostics": [], "minimum_elapsed_satisfied": True, "fixed_iteration_count_satisfied": True}],
            "target_input_checks": [{"tensor_id": "x", "location": "vram", "storage_bytes": 4, "exact_bytes_unchanged": True}],
            "copies": [], "copy_values_equal": True, "initial_ports": REFERENCE, "ports": REFERENCE,
            "native_repeat_consistency": {"passed": True}, "post_replay_consistency": {"passed": True},
            "kernel_execution_path_proven": False, "physical_device_capacity_claimed": False, "cuda_private_workspace_bytes": None}


def observations():
    return [{"sample_index": index, "deadline_tick": tick, "query_begin_tick": tick, "query_end_tick": tick + 1,
             "query_duration_ticks": 1, "lateness_ticks": 0, "missed_periods": 0,
             "gpu": {"status": "ok", "values": {"sm_clock_mhz": 1200}, "unknown": {},
                     "provider_query_begin_tick": tick, "provider_query_end_tick": tick + 1}}
            for index, tick in enumerate(range(900, 1450, 10))]


def fixture(root):
    source = root / "synthetic-source.py"
    source.write_text("# SYNTHETIC ONLY\n", encoding="utf-8", newline="\n")
    binary = root / "native/probe.exe"
    binary.parent.mkdir()
    binary.write_bytes(b"SYNTHETIC ONLY; NOT EXECUTABLE")
    build = binary.parent / "build-manifest.json"
    save(build, {"returncode": 0, "bound_inputs_unchanged": True, "binary_sha256": v._sha(binary), "bound_files": {str(source): v._sha(source)}})
    paths = [source, binary, build, Path(v.__file__), Path(run.__file__), Path(model.__file__), Path(telemetry.__file__)]
    manifest = {"schema": "state-calibration-campaign-0.1", "binary": str(binary), "runtime_dirs": [],
                "sources": {str(path.resolve()): v._sha(path) for path in paths},
                "protocol": {"blocks": 1, "fixed_iterations_per_block": 3, "minimum_block_ms": 200, "warmup": 20,
                             "interval_ms": 10, "within_process_cv_max": .15, "between_process_cv_max": .20, "error_max": .20},
                "programs": {"case-a": {}}, "jobs": []}
    for split, seed in (("profile", 1), ("heldout", 99)):
        prog_path, ref_path = root / f"programs/{split}.json", root / f"references/{split}.json"
        save(prog_path, program(seed)); save(ref_path, REFERENCE)
        manifest["programs"]["case-a"][split] = {"path": str(prog_path), "sha256": v._sha(prog_path), "reference": str(ref_path), "reference_sha256": v._sha(ref_path), "seed": seed}
        for repeat in range(3):
            manifest["jobs"].append({"id": f"{split}-{repeat}", "split": split, "case_id": "case-a", "seed": seed,
                                     "repeat": repeat, "condition": "baseline", "mode": "isolated", "instrumentation": "stamped",
                                     "telemetry": True, "program_key": split, "fresh_native_process": True})
    save(root / "manifest.json", manifest)
    collected = {"profile": [], "heldout": []}
    for job in manifest["jobs"]:
        directory = root / "jobs" / job["id"]
        directory.mkdir(parents=True)
        binding = manifest["programs"]["case-a"][job["program_key"]]
        prog = v._read(Path(binding["path"]))
        output = native(prog, 100 + len(collected[job["split"]]))
        save(directory / "native.json", output)
        save(directory / "numerical.json", run.validate_numeric(output, REFERENCE))
        records = observations()
        (directory / "telemetry.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8", newline="\n")
        metadata = {"schema": telemetry.SCHEMA, "clock": CLOCK, "interval_ms": 10, "period_ticks": 10,
                    "started": True, "stopped": True, "thread_error": None, "sample_count": len(records), "missed_periods": 0,
                    "source_sha256": v._sha(Path(telemetry.__file__)), "start_tick": 900, "end_tick": 1450}
        save(directory / "telemetry-metadata.json", metadata)
        save(directory / "alignment.json", telemetry.align_intervals(output["blocks"], records, CLOCK, 10, CLOCK))
        (directory / "native.stdout").write_bytes(b"")
        (directory / "native.stderr").write_bytes(b"")
        second = 1 + job["repeat"] if job["split"] == "profile" else 10 + job["repeat"]
        row = {"job": job, "status": "measured", "returncode": 0, "process_id": output["process_id"],
               "requested_runtime_flags": RUNTIME_FLAGS, "flags_set_before_process_launch": True,
               "started_utc": f"2026-10-18T00:00:{second:02d}+00:00", "completed_utc": f"2026-10-18T00:00:{second:02d}.900000+00:00",
               "launch_tick": 900, "exit_tick": 1400, "completed_tick": 1450, "native_clock": CLOCK,
               "program_sha256": binding["sha256"],
               "command": [manifest["binary"], binding["path"], "isolated", "0", "20", "1", "200", "stamped", str(directory / "native.json")]}
        row["summary"] = v._summary(directory, row, manifest, prog, REFERENCE)
        row["artifacts_sha256"] = {path.name: v._sha(path) for path in directory.iterdir() if path.is_file()}
        row["raw_file_sha256"] = dict(row["artifacts_sha256"])
        save(directory / "execution.json", row)
        collected[job["split"]].append(row)
    for split, rows in collected.items():
        save(root / f"{split}-rows.json", rows)
    fitted = model.fit_model(collected["profile"], {"manifest_sha256": v._sha(root / "manifest.json"), **manifest["sources"]}, "2026-10-18T00:00:05+00:00")
    save(root / "frozen-model.json", fitted)
    save(root / "profile-barrier.json", {"frozen_utc": "2026-10-18T00:00:06+00:00", "all_sources_finished": True,
                                        "model_sha256": v._sha(root / "frozen-model.json"), "profile_rows_sha256": v._sha(root / "profile-rows.json"),
                                        "manifest_sha256": v._sha(root / "manifest.json")})
    save(root / "report.json", model.evaluate(fitted, collected["heldout"]))
    return manifest, collected


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.root_parent = Path(__file__).parent.resolve()
        self.temporary = tempfile.TemporaryDirectory(dir=self.root_parent, prefix="synthetic-state-verifier-")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.root_parent))
        self.reference_patch = mock.patch.object(v, "_reference", return_value=REFERENCE)
        self.reference_patch.start()
        self.manifest, self.rows = fixture(self.root)

    def tearDown(self):
        self.reference_patch.stop()
        self.assertTrue(self.root.is_relative_to(self.root_parent))
        self.temporary.cleanup()

    def verify(self):
        return v.verify(self.root)

    def coherent_row_edit(self, change, split="profile", index=0, artifact=None):
        row = self.rows[split][index]
        directory = self.root / "jobs" / row["job"]["id"]
        change(row, directory)
        if artifact:
            row["artifacts_sha256"][artifact] = v._sha(directory / artifact)
            row["raw_file_sha256"] = dict(row["artifacts_sha256"])
        save(directory / "execution.json", row)
        save(self.root / f"{split}-rows.json", self.rows[split])

    def assertRejected(self, text=None):
        result = self.verify()
        self.assertFalse(result["verified"])
        if text:
            self.assertIn(text, str(result["errors"]))
        return result

    def test_complete_fixture_verifies_without_launching_any_process(self):
        with mock.patch.object(run.subprocess, "Popen", side_effect=AssertionError("must never launch")):
            result = self.verify()
        self.assertTrue(result["verified"], result["errors"])
        self.assertEqual(result["measured_jobs_verified"], 6)
        self.assertEqual(result["native_processes_launched"], 0)
        self.assertFalse(result["accuracy_acceptance_overridden"])
        self.assertIsNone(result["cuda_private_workspace_bytes"])

    def test_native_artifact_byte_mutation_rejected(self):
        path = self.root / "jobs/profile-0/native.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.assertRejected("SHA-256 mismatch")

    def test_source_mutation_rejected(self):
        (self.root / "synthetic-source.py").write_bytes(b"changed")
        self.assertRejected("SHA-256 mismatch")

    def test_missing_required_core_source_rejected(self):
        self.manifest["sources"].pop(str(Path(v.__file__).resolve()))
        save(self.root / "manifest.json", self.manifest)
        self.assertRejected("required verification/execution source not bound")

    def test_build_binding_omission_rejected(self):
        self.manifest["sources"].pop(str((self.root / "native/build-manifest.json").resolve()))
        save(self.root / "manifest.json", self.manifest)
        self.assertRejected("native binary/build manifest")

    def test_reference_wrong_but_rebound_hash_still_rejected(self):
        binding = self.manifest["programs"]["case-a"]["profile"]
        save(Path(binding["reference"]), {"y": [100.0], "z": [2.0]})
        binding["reference_sha256"] = v._sha(Path(binding["reference"]))
        save(self.root / "manifest.json", self.manifest)
        self.assertRejected("independent reference does not reconstruct")

    def test_declared_job_coverage_rejected(self):
        save(self.root / "profile-rows.json", self.rows["profile"][:-1])
        self.assertRejected("declared job coverage differs")

    def test_aggregate_individual_mismatch_rejected(self):
        self.rows["profile"][0]["summary"]["target_median_ns"] = 1
        save(self.root / "profile-rows.json", self.rows["profile"])
        self.assertRejected("individual execution record")

    def test_wrong_job_or_command_rejected(self):
        self.coherent_row_edit(lambda row, directory: row["command"].__setitem__(2, "mixed"))
        self.assertRejected("native command differs")

    def test_prelaunch_precision_flags_cannot_be_missing_or_false(self):
        for field, value in (("flags_set_before_process_launch", False), ("flags_set_before_process_launch", None),
                             ("requested_runtime_flags", {}), ("requested_runtime_flags", None)):
            original = copy.deepcopy(self.rows["profile"][0])
            self.coherent_row_edit(lambda row, directory: row.__setitem__(field, value))
            self.assertRejected("flags were not recorded as set before process launch")
            self.rows["profile"][0] = original
            save(self.root / "jobs/profile-0/execution.json", original)
            save(self.root / "profile-rows.json", self.rows["profile"])

    def test_summary_mutation_rejected_after_row_consistency_rebound(self):
        self.coherent_row_edit(lambda row, directory: row["summary"].__setitem__("target_median_ns", 1))
        self.assertRejected("summary does not reconstruct")

    def test_telemetry_alignment_mutation_rejected_even_when_file_rebound(self):
        def change(row, directory):
            value = v._read(directory / "alignment.json")
            value[0]["quality"]["model_ready"] = False
            save(directory / "alignment.json", value)
        self.coherent_row_edit(change, artifact="alignment.json")
        self.assertRejected("bounded alignment reconstruction differs")

    def test_missed_cadence_forgery_rejected(self):
        def change(row, directory):
            path = directory / "telemetry.jsonl"
            data = [json.loads(line) for line in path.read_text().splitlines()]
            data[0]["lateness_ticks"] = 123
            path.write_text("".join(json.dumps(item) + "\n" for item in data), encoding="utf-8", newline="\n")
        self.coherent_row_edit(change, artifact="telemetry.jsonl")
        self.assertRejected("query/cadence reconstruction differs")

    def test_native_input_immutability_forgery_rejected(self):
        def change(row, directory):
            value = v._read(directory / "native.json")
            value["target_input_checks"][0]["exact_bytes_unchanged"] = False
            save(directory / "native.json", value)
        self.coherent_row_edit(change, artifact="native.json")
        self.assertRejected("input identity/immutability differs")

    def test_native_process_identity_forgery_rejected(self):
        self.coherent_row_edit(lambda row, directory: row.__setitem__("process_id", 9999))
        self.assertRejected("launcher/native process identity differs")

    def test_model_change_rejected_by_original_barrier(self):
        value = v._read(self.root / "frozen-model.json")
        value["profile_seeds"] = [999]
        save(self.root / "frozen-model.json", value)
        self.assertRejected("freeze barrier bindings differ")

    def test_heldout_started_before_freeze_rejected(self):
        barrier = v._read(self.root / "profile-barrier.json")
        barrier["frozen_utc"] = "2026-10-18T00:00:15+00:00"
        save(self.root / "profile-barrier.json", barrier)
        self.assertRejected("freeze chronology violated")

    def test_evaluation_mutation_is_not_accepted_as_ground_truth(self):
        value = v._read(self.root / "report.json")
        value["families"]["conditional"]["accepted_count"] = 999
        save(self.root / "report.json", value)
        self.assertRejected("heldout evaluation does not reconstruct")

    def test_extra_artifact_not_silently_ignored(self):
        (self.root / "jobs/profile-0/extra.txt").write_text("new", encoding="utf-8")
        self.assertRejected("raw artifact inventory differs")

    def test_input_mutation_during_verification_is_reported(self):
        original = v._summary
        changed = False
        def wrapped(*arguments, **keywords):
            nonlocal changed
            result = original(*arguments, **keywords)
            if not changed:
                (self.root / "synthetic-source.py").write_bytes(b"changed after initial binding")
                changed = True
            return result
        with mock.patch.object(v, "_summary", side_effect=wrapped):
            self.assertRejected("ChangedInputs")

    def test_reference_regeneration_is_cached_per_program_sha(self):
        with mock.patch.object(v, "_reference", return_value=REFERENCE) as reference:
            result = self.verify()
        self.assertTrue(result["verified"], result["errors"])
        self.assertEqual(reference.call_count, 2)

    def background_fixture(self):
        row = copy.deepcopy(self.rows["profile"][0])
        row["job"]["condition"] = "cpu_replay_load"
        directory = self.root / "jobs/profile-0"
        prog = program(1)
        primary = v._read(directory / "native.json")
        background = native(prog, 999)
        background.update(mode="isolated", target_action_index=1, target={"action": prog["actions"][1], "binding": prog["bindings"][1]},
                          iteration_policy="minimum_wall_time", requested_iterations_per_block=None,
                          graceful_stop_requested=True, ready_tick=800, stop_observed_tick=1600,
                          instrumentation="per_iteration_host_completed_work_only")
        background["target_input_checks"][0]["location"] = "dram"
        background["blocks"] = [{"block_index": 0, "begin_tick": 840, "end_tick": 1460, "iteration_count": 1,
                                  "completed_work_ns": 600_000_000, "action_intervals": [], "target_diagnostics": [],
                                  "iterations": [{"iteration": 0, "begin_tick": 850, "end_tick": 1450, "elapsed_ns": 600_000_000}]}]
        ready = {"format_version": "state-calibration-ready-v1", "clock": CLOCK, "process_id": 999, "target_action_index": 1,
                 "mode": "isolated", "executing_work_proven_by_ready_file": False, "ready_tick": 800}
        save(directory / "background-native.json", background)
        save(directory / "background-ready.json", ready)
        save(directory / "background-stop.json", {"requested_tick": 1500})
        row["background"] = {"returncode": 0, "pid": 999, "ready": ready, "active_begin_tick": 850, "active_end_tick": 1450,
                              "condition_interval_covers_all_blocks": True,
                              "command": [row["command"][0], row["command"][1], "isolated", "1", "20", "1000", "200", "bare", str(directory / "background-native.json")]}
        return directory, row, prog, primary, background

    def test_background_active_envelope_and_numerical_checks(self):
        directory, row, prog, primary, _ = self.background_fixture()
        self.assertTrue(v._background(directory, row, prog, REFERENCE, primary))

    def test_background_clock_and_target_mismatch_rejected(self):
        directory, row, prog, primary, background = self.background_fixture()
        background["clock"] = {"clock_id": "windows_qpc", "frequency_hz": 999}
        save(directory / "background-native.json", background)
        with self.assertRaisesRegex(ValueError, "clock differs"):
            v._background(directory, row, prog, REFERENCE, primary)
        background["clock"] = CLOCK
        background["target_action_index"] = 0
        save(directory / "background-native.json", background)
        with self.assertRaisesRegex(ValueError, "process/mode/target differs"):
            v._background(directory, row, prog, REFERENCE, primary)

    def test_background_ready_does_not_substitute_for_active_work(self):
        directory, row, prog, primary, background = self.background_fixture()
        background["blocks"] = []
        save(directory / "background-native.json", background)
        with self.assertRaisesRegex(ValueError, "no recorded activity"):
            v._background(directory, row, prog, REFERENCE, primary)

    def test_background_incomplete_coverage_is_not_hidden(self):
        directory, row, prog, primary, _ = self.background_fixture()
        primary["blocks"][0]["end_tick"] = 1500
        row["background"]["condition_interval_covers_all_blocks"] = False
        self.assertFalse(v._background(directory, row, prog, REFERENCE, primary))

    def test_background_bad_numerical_values_rejected(self):
        directory, row, prog, primary, background = self.background_fixture()
        background["ports"] = {"y": [99.0], "z": [2.0]}
        save(directory / "background-native.json", background)
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            v._background(directory, row, prog, REFERENCE, primary)

    def test_background_timing_boundary_and_stamping_scope_rejected(self):
        for field, value, reason in (("execution_boundary", "async_events", "boundary/instrumentation differs"),
                                     ("instrumentation", "per_iteration_and_per_action_host_completed_work", "boundary/instrumentation differs"),
                                     ("target_diagnostics", [{"invented": True}], "unexpected target/event diagnostics")):
            directory, row, prog, primary, background = self.background_fixture()
            if field == "target_diagnostics":
                background["blocks"][0][field] = value
            else:
                background[field] = value
            save(directory / "background-native.json", background)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, reason):
                v._background(directory, row, prog, REFERENCE, primary)


if __name__ == "__main__":
    unittest.main()
