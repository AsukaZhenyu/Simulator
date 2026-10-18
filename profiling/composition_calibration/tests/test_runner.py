"""Pipeline invariants, independent of hardware and historical timing values."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("tested_composition_runner", HERE / "run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
vspec = importlib.util.spec_from_file_location("tested_composition_verifier", HERE / "verify.py")
v = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(v)


def fixtures():
    cases = []
    for index in range(4):
        cases.append(dict(id=f"geometry{index}", family="ffn", backend="hybrid", dtype="f32",
            hidden=32 * (index + 1), intermediate=64 * (index + 1), cpu_intermediate=16 * (index + 1),
            tokens=index+1, history=0, heads=0, kv_heads=0, head_dim=0))
    suite = dict(cases=cases, profile_seeds=[113], heldout_seeds=[127], repeats=3,
        quality=dict(max_within_process_cv=.15, max_between_process_cv=.2, max_heldout_relative_error=.2),
        bootstrap=dict(iterations=100, seed=13))
    rows = []
    for split, seed in (("profile", 113), ("heldout", 127)):
        for case in cases:
            for repeat in range(3):
                metrics = {key: [10000] * 3 for expression in r.batch.recipes(case).values() for key in r.a.terms(expression)}
                prices = {key: 10 for key in metrics}
                for strategy, expression in r.batch.recipes(case).items():
                    if "@" not in strategy:
                        predicted = r.a.predict(expression, prices)
                        metrics[strategy] = [round((predicted + 3) * 1000)] * 3
                rows.append(dict(id=f"{split}-{case['id']}-s{seed}-r{repeat}", split=split, seed=seed,
                    repeat=repeat, case=copy.deepcopy(case), status="measured", metrics=metrics,
                    raw_sha256="0"*64, raw_path="unused", batch_repetitions=16,
                    command=["probe", "--threads", "1"]))
    return suite, rows[:12], rows[12:]


class RunnerTests(unittest.TestCase):
    def test_profile_stream_never_deserializes_heldout_objects(self):
        profile = dict(id="profile-one", split="profile", metrics={"total": [1, 2, 3]})
        heldout = dict(id="heldout-one", split="heldout", metrics={"total": [987, 654, 321]})
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            path = Path(directory) / "rows.json"
            path.write_text(json.dumps([heldout, profile, heldout]), encoding="utf-8")
            seen = []
            original = r.object_document
            def decoder(text):
                seen.append(text)
                self.assertNotIn("heldout-one", text)
                return original(text)
            with patch.object(r, "object_document", decoder):
                self.assertEqual(list(r.profile_objects(path)), [profile])
            self.assertEqual(len(seen), 1)

    def test_fold_excludes_every_repeat_of_entire_geometry(self):
        suite, profile, _ = fixtures()
        folds = r.grouped_folds(profile, suite["cases"])
        self.assertEqual(len(folds), 4)
        for fold in folds:
            excluded = fold["excluded_case_id"]
            self.assertEqual(len(fold["training_rows"]), 9)
            self.assertEqual(len(fold["test_component_rows"]), 3)
            self.assertFalse(any(row["case"]["id"] == excluded for row in fold["training_rows"]))

    def test_component_prices_do_not_read_profile_target_values(self):
        suite, profile, _ = fixtures()
        expected = r.prices_from_components(profile, suite["cases"])
        changed = copy.deepcopy(profile)
        for row in changed:
            for strategy in r.overhead.primary_strategies(row["case"]):
                row["metrics"][strategy] = [999999999] * 3
        self.assertEqual(r.prices_from_components(changed, suite["cases"]), expected)

    def test_heldout_mutation_cannot_change_frozen_bundle(self):
        suite, profile, heldout = fixtures()
        bundle = r.build_bundle(profile, suite, "test")
        changed = copy.deepcopy(heldout)
        for row in changed:
            for strategy in r.overhead.primary_strategies(row["case"]):
                row["metrics"][strategy] = [900000000] * 3
        first = r.evaluate(profile, heldout, suite, bundle)
        second = r.evaluate(profile, changed, suite, bundle)
        self.assertEqual(r.build_bundle(profile, suite, "test"), bundle)
        self.assertEqual([x["predicted_ns"] for x in first["corrected_results"]],
                         [x["predicted_ns"] for x in second["corrected_results"]])
        self.assertNotEqual(first["corrected_summary"], second["corrected_summary"])

    def test_excluded_geometry_targets_cannot_change_its_fold_coefficients(self):
        suite, profile, _ = fixtures()
        first = r.build_bundle(profile, suite, "test")
        changed = copy.deepcopy(profile)
        excluded = suite["cases"][0]["id"]
        for row in changed:
            if row["case"]["id"] == excluded:
                for strategy in r.overhead.primary_strategies(row["case"]):
                    row["metrics"][strategy] = [900000000] * 3
        second = r.build_bundle(changed, suite, "test")
        self.assertEqual(first["folds"][0]["model"], second["folds"][0]["model"])
        self.assertEqual(first["component_prices_us"][excluded], second["component_prices_us"][excluded])

    def test_failed_cv_gate_stays_failed_after_prediction_correction(self):
        suite, profile, heldout = fixtures()
        heldout[0]["metrics"]["gpu/resident"] = [1000, 10000, 1000000]
        result = r.evaluate(profile, heldout, suite, r.build_bundle(profile, suite, "test"))
        old, new = result["baseline_results"][0], result["corrected_results"][0]
        self.assertFalse(old["gates"]["within_process_cv"])
        self.assertFalse(new["gates"]["within_process_cv"])
        for key in old["gates"]:
            if key != "heldout_relative_error":
                self.assertEqual(old["gates"][key], new["gates"][key])
        self.assertFalse(new["calibrated"])

    def test_failures_and_unknown_workspace_are_retained(self):
        suite, profile, heldout = fixtures()
        bundle = r.build_bundle(profile, suite, "test")
        heldout[0] = {key: value for key, value in heldout[0].items() if key != "metrics"}
        heldout[0].update(status="failed", error="retained failure")
        result = r.evaluate(profile, heldout, suite, bundle)
        self.assertIn(heldout[0]["id"], result["failed_native_rows"])
        self.assertFalse(result["numeric_execution_valid"])
        self.assertIsNone(result["gpu_private_workspace_bytes"])
        self.assertFalse(result["physical_capacity_validated"])
        self.assertFalse(result["corrected_results"][0]["calibrated"])
        bundle["component_prices_us"][suite["cases"][0]["id"]] = {}
        unknown = r.evaluate(profile, heldout, suite, bundle)["corrected_results"][0]
        self.assertIsNone(unknown["predicted_ns"])
        self.assertFalse(unknown["calibrated"])

    def test_frozen_bundle_records_failed_training_without_hiding_failure(self):
        suite, profile, _ = fixtures()
        profile[0].update(status="failed", error="retained failure")
        bundle = r.build_bundle(profile, suite, "test")
        self.assertEqual(bundle["failed_training_rows_retained"], [profile[0]["id"]])
        self.assertNotIn(profile[0]["id"], bundle["training_row_ids"])

    def test_staged_run_freezes_all_models_before_loading_heldout(self):
        suite, profile, heldout = fixtures()
        # Hardware verification is independently tested by the existing strict
        # verifier. This test spies on the orchestration's deserialization edge.
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            contexts = {}
            for label in ("initial", "confirmation"):
                experiment = root / label
                experiment.mkdir()
                (experiment / "measurements.json").write_text(json.dumps(profile+heldout), encoding="utf-8")
                contexts[experiment] = dict(experiment=experiment, suite=suite, original_files={}, manifest={}, changed_current_metadata=[])
            out = root / "out"
            original_document = r.base.document
            def document(path):
                path = Path(path)
                if path.name == "measurements.json":
                    for label in ("initial", "confirmation"):
                        self.assertTrue((out / (label + "-frozen-model.json")).is_file())
                    ledger = original_document(out / "freeze-ledger.json")
                    self.assertEqual(len(ledger["model_sha256"]), 2)
                    for label, digest in ledger["model_sha256"].items():
                        self.assertEqual(r.a.sha(out / (label + "-frozen-model.json")), digest)
                return original_document(path)
            # Fixture IDs exactly match base.jobs format; shuffle order is not
            # material to this stage-order test, so supply the same plan.
            with patch.object(r, "preflight", side_effect=lambda path: contexts[Path(path)]), \
                 patch.object(r, "verify_campaign", return_value={"verified": True}), \
                 patch.object(r, "compatibility", return_value={"compatible": False}), \
                 patch.object(r.base, "jobs", side_effect=lambda s, split: profile if split == "profile" else heldout), \
                 patch.object(r.base, "document", side_effect=document):
                self.assertEqual(r.run(SimpleNamespace(initial=root/"initial", confirmation=root/"confirmation", output=out)), 0)

    def test_verifier_rejects_frozen_source_or_report_tampering(self):
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            manifest = dict(schema=r.SCHEMA, fresh_blind_validation=False,
                joint_chain_totals_used_for_calibration=True, gpu_private_workspace_bytes=None,
                physical_capacity_validated=False, sources_unchanged=True, original_assets_unchanged=True,
                source_modules_before={"model.py": "old"}, source_modules_after={"model.py": "old"})
            r.a.dump(root / "manifest.json", manifest)
            r.a.dump(root / "freeze-ledger.json", {})
            with self.assertRaisesRegex(ValueError, "source changed"):
                v.verify(root)
            manifest.update(source_modules_before=v.runner.source_inventory(), source_modules_after=v.runner.source_inventory(),
                            original_assets_before={}, original_assets_after={}, report_sha256="not the report digest")
            r.a.dump(root / "manifest.json", manifest)
            r.a.dump(root / "report.json", {})
            with self.assertRaisesRegex(ValueError, "report hash mismatch"):
                v.verify(root)

    def test_verifier_rejects_frozen_model_tampering(self):
        suite, profile, _ = fixtures()
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            model = root / "initial-frozen-model.json"
            r.a.dump(model, {"coefficients": "original"})
            digest = r.a.sha(model)
            stages = [dict(stage="full_and_all_grouped_models_saved_and_hashed", campaign=label) for label in ("initial", "confirmation")]
            stages += [dict(stage="heldout_totals_deserialization_started", campaign=label) for label in ("initial", "confirmation")]
            r.a.dump(root / "report.json", {})
            manifest = dict(schema=r.SCHEMA, fresh_blind_validation=False,
                joint_chain_totals_used_for_calibration=True, gpu_private_workspace_bytes=None,
                physical_capacity_validated=False, sources_unchanged=True, original_assets_unchanged=True,
                source_modules_before=v.runner.source_inventory(), source_modules_after=v.runner.source_inventory(),
                original_assets_before={}, original_assets_after={}, report_sha256=r.a.sha(root / "report.json"),
                model_sha256={"initial": digest}, stages=stages, input_paths={"initial": str(root/"input")})
            r.a.dump(root / "manifest.json", manifest)
            r.a.dump(root / "freeze-ledger.json", dict(model_sha256=manifest["model_sha256"], stages=stages[:2]))
            r.a.dump(model, {"coefficients": "tampered"})
            with patch.object(v.runner, "preflight", return_value={"suite": suite}), \
                 patch.object(v.runner, "profile_objects", return_value=iter(profile)):
                with self.assertRaisesRegex(ValueError, "frozen model hash mismatch"):
                    v.verify(root)


if __name__ == "__main__":
    unittest.main()
