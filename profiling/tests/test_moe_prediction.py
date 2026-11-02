"""Prediction validator tests with fabricated native-protocol evidence.

These fixtures measure no hardware and prove no numerical kernel correctness.
Binary/runtime hashes are intentionally fake; live-source verification is disabled.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

SIMULATOR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMULATOR / "profiling"))
sys.path.insert(0, str(SIMULATOR / "modeling"))
import validate_moe_prediction as prediction
from run_moe_measurements import dump, load_measurements, sha256, summarize_native
from moe_contract import MEASUREMENT_SCHEMA, CONTRACT, contract_hash, physical_signature
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from llm_infer_model.tensor.engine import load_mapping, evaluate_mapping, MappingIdentityError

_fixture_spec = importlib.util.spec_from_file_location(
    "_prediction_synthetic_protocol_fixture", SIMULATOR / "mapping" / "tests" / "test_moe_measured.py")
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
protocol_fixture = _fixture_module.protocol_fixture
hardware_fixture = _fixture_module.hardware_fixture


def record(*, joint=1500, marker="holdout"):
    value = protocol_fixture()
    value["native"]["stages"]["joint"]["samples_ns"] = [joint] * 3
    value["raw_sha256"] = ("1" if marker == "profile" else "2" if marker == "holdout" else "3") * 64
    return value


def pair():
    return record(joint=999999, marker="profile"), record()


def write_evidence(directory, value):
    """Write the runner's protocol, summary, flags and raw bindings synthetically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw").mkdir()
    value = copy.deepcopy(value)
    raw = directory / "raw" / "000.json"
    dump(raw, value["native"])
    value.update(raw_output="raw/000.json", raw_sha256=sha256(raw))
    summary, flags = summarize_native(value["native"])
    value.update(summary=summary, quality_flags=flags, validation_only=bool(flags),
                 status="measured_validation_only" if flags else "measured")
    dump(directory / "measurements.json",
         dict(schema_version=MEASUREMENT_SCHEMA, contract=CONTRACT, records=[value]))
    dump(directory / "manifest.json",
         dict(schema_version="moe_measurement_manifest_v1",
              native_binary_sha256="a" * 64,
              runtime_libraries_sha256={"synthetic-not-live-ggml-cpu.dll": "b" * 64},
              native_binary="synthetic-not-live-probe.exe", suite="synthetic-not-live-suite.json",
              suite_sha256="c" * 64, machine="SYNTHETIC_PROTOCOL_MACHINE",
              platform="SYNTHETIC_PROTOCOL_PLATFORM", seed=value["native"]["seed"],
              warmup=value["native"]["warmup"], samples=value["native"]["samples"],
              record_count=1, fixture_is_synthetic=True))
    return value


class CompareCaseTests(unittest.TestCase):
    def compare(self, profile=None, holdouts=None, hardware=None, tolerance=0, **kwargs):
        a, b = pair()
        return prediction.compare_case(profile or a, holdouts or [b], hardware or hardware_fixture(),
                                       max_relative_error=tolerance, **kwargs)

    def test_correct_independent_joint_comparison_accepts_declared_local_scope(self):
        compiled, replay, result = self.compare()
        self.assertEqual(replay.makespan_ns, 1500)
        self.assertEqual(result["predicted_ns"], 1500)
        self.assertEqual(result["holdouts"][0]["absolute_relative_error"], 0)
        self.assertTrue(result["local_prediction_accepted"])
        for claim in ("performance_calibrated", "whole_decoder_layer_measured",
                      "original_model_weights", "GPU_compute_measured", "physical_capacity_verified"):
            self.assertFalse(result[claim])
        self.assertNotIn("joint", compiled.scenario_document["costs"]["compute"])

    def test_profile_joint_value_never_becomes_a_price(self):
        profile, holdout = pair()
        profile["native"]["stages"]["joint"]["samples_ns"] = [1, 1, 99999999]
        compiled, replay, result = self.compare(profile, [holdout])
        self.assertEqual(result["predicted_ns"], 1500)
        self.assertTrue(result["local_prediction_accepted"])
        self.assertNotIn("joint", compiled.bindings["stage_statistics"])

    def test_error_threshold_is_actual_gate_and_signed_error_is_preserved(self):
        profile, holdout = pair()
        holdout["native"]["stages"]["joint"]["samples_ns"] = [1600] * 3
        _, _, refused = self.compare(profile, [holdout], tolerance=0.01)
        self.assertFalse(refused["local_prediction_accepted"])
        self.assertAlmostEqual(refused["holdouts"][0]["signed_relative_error"], -0.0625)
        _, _, accepted = self.compare(profile, [holdout], tolerance=0.07)
        self.assertTrue(accepted["local_prediction_accepted"])

    def test_tolerance_must_be_explicit_finite_nonnegative(self):
        for value in (None, True, -1, float("nan"), float("inf"), "0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare(tolerance=value)

    def test_route_change_with_new_signature_is_not_comparable(self):
        profile, holdout = pair()
        holdout["case"]["routes"][0] = [1, 0]
        holdout["native"]["routes"][0] = [1, 0]
        holdout["physical_signature"] = physical_signature(holdout["case"], "a" * 64)
        holdout["physical_contract_sha256"] = contract_hash(holdout["physical_signature"])
        with self.assertRaisesRegex(ValueError, "signature differs"):
            self.compare(profile, [holdout])

    def test_seed_change_is_not_comparable(self):
        profile, holdout = pair()
        holdout["native"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed differs"):
            self.compare(profile, [holdout])

    def test_port_digest_change_is_not_comparable(self):
        profile, holdout = pair()
        holdout["native"]["stages"]["router"]["ports"]["router_weight"]["digest"] = "f" * 16
        with self.assertRaisesRegex(ValueError, "ports differ"):
            self.compare(profile, [holdout])

    def test_workspace_change_is_not_comparable(self):
        profile, holdout = pair()
        holdout["native"]["stages"]["router"]["workspace_bytes"] += 16
        with self.assertRaisesRegex(ValueError, "workspace_bytes differs"):
            self.compare(profile, [holdout])

    def test_raw_profile_reuse_is_rejected(self):
        profile, holdout = pair()
        holdout["raw_sha256"] = profile["raw_sha256"]
        with self.assertRaisesRegex(ValueError, "reused as a holdout"):
            self.compare(profile, [holdout])

    def test_duplicate_holdout_raw_is_rejected(self):
        profile, holdout = pair()
        with self.assertRaisesRegex(ValueError, "duplicate holdout"):
            self.compare(profile, [holdout, copy.deepcopy(holdout)])

    def test_empty_holdout_list_is_rejected(self):
        profile, _ = pair()
        with self.assertRaisesRegex(ValueError, "at least one"):
            prediction.compare_case(profile, [], hardware_fixture(), max_relative_error=0)

    def test_unknown_hardware_is_not_guessed(self):
        for key in hardware_fixture():
            hardware = hardware_fixture()
            del hardware[key]
            with self.subTest(key=key), self.assertRaises(InvalidInput):
                self.compare(hardware=hardware)

    def test_unstable_profile_is_refused_by_default(self):
        profile, holdout = pair()
        profile["native"]["stages"]["expert.1"]["samples_ns"] = [100, 100, 10000]
        with self.assertRaises(Unsupported):
            self.compare(profile, [holdout])

    def test_unstable_profile_opt_in_is_validation_only(self):
        profile, holdout = pair()
        profile["native"]["stages"]["expert.1"]["samples_ns"] = [100, 100, 10000]
        _, _, result = self.compare(profile, [holdout], tolerance=100, allow_unstable=True)
        self.assertTrue(result["validation_only"])
        self.assertFalse(result["local_prediction_accepted"])
        self.assertFalse(result["holdouts"][0]["local_prediction_accepted"])

    def test_unstable_joint_is_validation_only_even_with_loose_error_tolerance(self):
        profile, holdout = pair()
        holdout["native"]["stages"]["joint"]["samples_ns"] = [100, 100, 10000]
        _, _, result = self.compare(profile, [holdout], tolerance=100)
        self.assertTrue(result["validation_only"])
        self.assertFalse(result["local_prediction_accepted"])


class DirectoryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile_dir, self.holdout_dir = self.root / "profile", self.root / "holdout"
        a, b = pair()
        self.profile = write_evidence(self.profile_dir, a)
        self.holdout = write_evidence(self.holdout_dir, b)
        self.hardware = self.root / "hardware.json"
        dump(self.hardware, hardware_fixture())
        self.output = self.root / "validated"
        self.args = SimpleNamespace(profile=self.profile_dir, holdout=[self.holdout_dir],
                                    hardware=self.hardware, output=self.output,
                                    max_relative_error=0, allow_unstable=False, verify_live_sources=False)

    def test_full_export_reloads_mapping_and_verifies_fingerprints(self):
        result = prediction.validate_prediction(self.args)
        self.assertTrue(result["all_local_predictions_accepted"])
        self.assertEqual(result["price_source"], "profile_materialized_stage_medians_only")
        self.assertEqual(result["sampling_modes"], ["isolated_warm"])
        self.assertEqual(result["truth_source"], "independent_holdout_joint_medians")
        self.assertFalse(result["verify_live_sources"])
        folder = self.output / "000"
        from llm_infer_model.tensor.spec import build_scenario
        scenario = build_scenario(json.loads((folder / "workload.json").read_text()),
                                  json.loads((folder / "scenario.json").read_text()))
        loaded = load_mapping(folder / "mapping.json", scenario)
        self.assertTrue(loaded.fingerprint_verified)
        replay = evaluate_mapping(scenario, loaded.actions)
        self.assertEqual(replay.makespan_ns, result["cases"][0]["predicted_ns"])
        self.assertEqual(replay.status, "valid")
        self.assertNotIn(b"\r\n", (self.output / "report.json").read_bytes())

    def test_profile_directory_cannot_be_its_own_holdout(self):
        self.args.holdout = [self.profile_dir]
        with self.assertRaisesRegex(ValueError, "independent and distinct"):
            prediction.validate_prediction(self.args)

    def test_duplicate_holdout_directory_is_rejected(self):
        self.args.holdout = [self.holdout_dir, self.holdout_dir]
        with self.assertRaisesRegex(ValueError, "independent and distinct"):
            prediction.validate_prediction(self.args)

    def test_same_raw_bytes_in_another_directory_are_not_independent(self):
        with tempfile.TemporaryDirectory() as other:
            write_evidence(Path(other), self.profile)
            self.args.holdout = [Path(other)]
            with self.assertRaisesRegex(ValueError, "reused as a holdout"):
                prediction.validate_prediction(self.args)

    def test_raw_sha_change_before_validation_is_rejected(self):
        raw = self.holdout_dir / "raw" / "000.json"
        raw.write_bytes(raw.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "raw measurement source changed"):
            prediction.validate_prediction(self.args)

    def test_embedded_native_change_is_rejected(self):
        path = self.holdout_dir / "measurements.json"
        report = json.loads(path.read_text())
        report["records"][0]["native"]["seed"] += 1
        dump(path, report)
        with self.assertRaisesRegex(ValueError, "embedded native measurement changed"):
            prediction.validate_prediction(self.args)

    def test_forged_quality_summary_is_rejected(self):
        path = self.holdout_dir / "measurements.json"
        report = json.loads(path.read_text())
        report["records"][0]["summary"]["joint"]["median_ns"] = 123
        dump(path, report)
        with self.assertRaisesRegex(ValueError, "quality or statistics changed"):
            prediction.validate_prediction(self.args)

    def test_holdout_runtime_metadata_change_is_rejected(self):
        path = self.holdout_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["machine"] = "OTHER_SYNTHETIC_MACHINE"
        dump(path, manifest)
        with self.assertRaisesRegex(ValueError, "machine differs"):
            prediction.validate_prediction(self.args)

    def test_output_is_not_overwritten(self):
        self.output.mkdir()
        (self.output / "existing.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            prediction.validate_prediction(self.args)
        self.assertEqual((self.output / "existing.txt").read_text(), "keep")

    def test_hardware_change_during_export_is_rejected(self):
        original_dump = prediction.dump
        mutated = []
        def changing_dump(path, value):
            original_dump(path, value)
            if not mutated:
                changed = hardware_fixture()
                changed["dram_capacity_bytes"] += 1
                original_dump(self.hardware, changed)
                mutated.append(True)
        with patch.object(prediction, "dump", side_effect=changing_dump), self.assertRaisesRegex(ValueError, "changed during comparison"):
            prediction.validate_prediction(self.args)
        self.assertFalse((self.output / "report.json").exists())

    def test_evidence_is_rechecked_after_export(self):
        original_dump = prediction.dump
        original_snapshot = prediction.evidence_snapshot
        snapshots = []
        mutated = []
        def tracking_snapshot(path):
            snapshots.append(str(path))
            return original_snapshot(path)
        def changing_dump(path, value):
            original_dump(path, value)
            if not mutated:
                manifest_path = self.profile_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["fixture_mutated_during_export"] = True
                original_dump(manifest_path, manifest)
                mutated.append(True)
        with patch.object(prediction, "dump", side_effect=changing_dump), \
             patch.object(prediction, "evidence_snapshot", side_effect=tracking_snapshot), \
             self.assertRaisesRegex(ValueError, "changed during comparison"):
            prediction.validate_prediction(self.args)
        self.assertGreaterEqual(snapshots.count(str(self.profile_dir.resolve())), 2)
        self.assertFalse((self.output / "report.json").exists())

    def test_exported_fingerprint_tampering_is_detected(self):
        original_dump = prediction.dump
        def tamper_dump(path, value):
            if Path(path).name == "mapping.json":
                value = copy.deepcopy(value)
                value["scenario_fingerprint"] = "0" * 64
            original_dump(path, value)
        with patch.object(prediction, "dump", side_effect=tamper_dump), self.assertRaises(MappingIdentityError):
            prediction.validate_prediction(self.args)


if __name__ == "__main__":
    unittest.main()
