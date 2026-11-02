"""Evidence analysis tests use synthetic captured outputs; no hardware runs."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_dense_error as analyzer
import run_dense_layer_measurements as runner
from profile_contract import summarize_samples
from test_dense_layer import case, layer_document, native, suite


class DenseErrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dense-error-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.layer = self.root / "layer.json"
        self.write(self.layer, layer_document())
        self.binary = self.root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
        self.binary.write_bytes(b"synthetic not executable")
        self.binary_hash = hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")

    def evidence(self, name="run-1", *, shapes=None, timings=None):
        folder = self.root / name
        cases = []
        for index, shape in enumerate(shapes or [{}]):
            cases += [case(kind, id=f"shape-{index}-{kind}", **shape) for kind in ("projection", "attention", "dense_ffn", "block")]
        manifest = {"schema_version":"dense_layer_measurement_profile_v1", "contract":analyzer.load_layer(self.layer).physical.contract,
            "profile_id":name, "suite":suite(cases),
            "protocol":{"backend":"CPU", "batch":1, "threads":2, "warmup":2, "samples":3, "seed":17,
                "sequential_execution":True, "graphs_per_sample":1, "setup_compute_repetitions":1,
                "measured_graph_untimed_compute_repetitions":3,
                "upstream_setup_compute_repetitions_by_kind":{kind:0 if kind == "block" else 1 for kind in ("projection", "attention", "dense_ffn", "block")},
                "total_untimed_compute_repetitions_by_kind":{kind:3 if kind == "block" else 4 for kind in ("projection", "attention", "dense_ffn", "block")}},
            "software":{"runtime_sha256":{str(self.binary):self.binary_hash}}}
        rows = []
        for index, selected in enumerate(cases):
            observed = native(selected)
            if timings and selected["kind"] in timings:
                observed["wall_ns"] = timings[selected["kind"]]
            summary = summarize_samples(observed["wall_ns"])
            row = {"case_id":selected["id"], "case":selected, "status":"measured",
                "signature":runner.measurement_signature(selected, self.binary_hash),
                "native":{**observed, "expected_samples":3, "summary":summary}, "summary":summary,
                "quality_flags":["high_observed_variability_cv_gt_0.1"] if summary["cv"] > 0.1 else [],
                "raw_output":f"raw/{index+1:03d}.json"}
            rows.append(row)
            self.write_raw(folder, row)
        self.write(folder / "manifest.json", manifest)
        self.write(folder / "measurements.json", rows)
        return folder / "measurements.json", rows, manifest

    def write_raw(self, folder, row):
        payload = {key:value for key,value in row["native"].items() if key not in ("summary", "expected_samples")}
        self.write(folder / row["raw_output"], {"case":row["case"], "returncode":0,
            "command":runner.native_command(row["case"], self.binary.parent, SimpleNamespace(warmup=2, samples=3, seed=17)),
            "stdout":json.dumps(payload), "stderr":"synthetic evidence; never executed"})

    def analyze(self, *paths):
        return analyzer.analyze_runs(list(paths), [self.layer])

    def test_exact_error_and_every_stage_sample_are_reported(self):
        path, rows, manifest = self.evidence(timings={"block":[4000, 4001, 3999]})
        report = self.analyze(path)
        self.assertTrue(report["verification_passed"])
        group = report["runs"][0]["groups"][0]
        self.assertEqual(group["estimate"]["predicted_serial_stage_sum_ns"], 3000)
        self.assertEqual(group["estimate"]["observed_joint_block_median_ns"], 4000)
        self.assertEqual(group["estimate"]["signed_relative_error"], -0.25)
        self.assertEqual(group["estimate"]["absolute_relative_error"], 0.25)
        self.assertEqual(len(group["stages"]), 4)
        for row in rows:
            stage = group["stages"][row["case"]["kind"]]
            self.assertEqual(stage["wall_ns"], row["native"]["wall_ns"])
            self.assertEqual(stage["summary"], row["summary"])
            self.assertEqual(stage["raw_sha256"], hashlib.sha256((path.parent/row["raw_output"]).read_bytes()).hexdigest())
        self.assertFalse(report["analysis_policy"]["joint_is_oracle"])
        self.assertFalse(report["analysis_policy"]["calibration_claimed"])

    def test_high_cv_is_retained_with_stability_marked_false(self):
        path, rows, _ = self.evidence(timings={"projection":[100, 1000, 10000]})
        report = self.analyze(path)
        group = report["runs"][0]["groups"][0]
        self.assertTrue(report["verification_passed"])
        self.assertTrue(group["stages"]["projection"]["cv_gt_0_1"])
        self.assertEqual(group["stages"]["projection"]["wall_ns"], [100, 1000, 10000])
        self.assertFalse(group["suitable_as_stable_costs"])
        self.assertTrue(group["stages"]["projection"]["quality_flags"])

    def test_independent_runs_compare_estimates_without_pooling(self):
        first, _, _ = self.evidence("first")
        second, _, _ = self.evidence("second", timings={"projection":[2000, 2001, 1999], "block":[6000, 6001, 5999]})
        report = self.analyze(first, second)
        comparison = report["exact_shape_repeat_comparisons"][0]
        self.assertEqual(comparison["run_count"], 2)
        self.assertFalse(comparison["samples_pooled"])
        pair = comparison["pairwise_run_comparisons"][0]
        self.assertTrue(pair["same_protocol_runtime_repeat"])
        self.assertEqual(pair["differences"][0]["signed_change_ns"], 1000)
        self.assertEqual(pair["differences"][1]["signed_change_relative_to_left"], 1)
        self.assertEqual(pair["stage_median_changes_relative_to_left"]["projection"], 1)
        self.assertAlmostEqual(pair["signed_relative_error_change"], -1/3)

    def test_different_shapes_are_never_grouped_or_pooled(self):
        path, _, _ = self.evidence(shapes=[{}, {"phase":"decode", "tokens":1, "visible_history":8}])
        report = self.analyze(path)
        self.assertEqual(report["verified_group_count"], 2)
        self.assertEqual(len(report["exact_shape_repeat_comparisons"]), 2)
        self.assertEqual({row["shape"]["tokens"] for row in report["exact_shape_repeat_comparisons"]}, {1, 3})

    def test_missing_all_cases_at_one_manifest_shape_stays_visible(self):
        path, rows, _ = self.evidence(shapes=[{}, {"phase":"decode", "tokens":1, "visible_history":8}])
        self.write(path, rows[:4])
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(report["verified_group_count"], 1)
        self.assertEqual(report["rejected_group_count"], 1)
        missing = next(group for group in report["runs"][0]["groups"] if group["status"] == "rejected")
        self.assertEqual(len(missing["missing_case_ids"]), 4)
        self.assertEqual(missing["records"], [])
        self.assertFalse(missing["raw_samples_verified"])

    def test_failed_case_and_raw_hash_stay_in_rejected_group(self):
        path, rows, _ = self.evidence()
        rows[0].update(status="failed", error="probe failed deliberately")
        self.write(path, rows)
        report = self.analyze(path)
        group = report["runs"][0]["groups"][0]
        self.assertFalse(report["verification_passed"])
        self.assertEqual(len(group["records"]), 4)
        self.assertEqual(group["records"][0]["error"], "probe failed deliberately")
        self.assertIn("raw_sha256", group["records"][0])
        self.assertTrue(any("failed" in error for error in group["errors"]))

    def test_raw_summary_and_manifest_runtime_tampering_rejected(self):
        for change in ("raw", "summary", "runtime"):
            with self.subTest(change=change):
                path, rows, manifest = self.evidence(change)
                if change == "raw":
                    raw_path = path.parent / rows[0]["raw_output"]
                    raw = json.loads(raw_path.read_text()); raw["stdout"] = "{}"
                    self.write(raw_path, raw)
                elif change == "summary":
                    rows[0]["summary"]["median_ns"] = 99; self.write(path, rows)
                else:
                    manifest["software"]["runtime_sha256"][str(self.binary)] = "0"*64
                    self.write(path.parent / "manifest.json", manifest)
                report = self.analyze(path)
                self.assertFalse(report["verification_passed"])
                self.assertEqual(report["rejected_group_count"], 1)
                self.assertTrue(report["runs"][0]["groups"][0]["errors"])

    def test_protocol_changes_are_disclosed_as_nonmatching_repeat(self):
        first, _, _ = self.evidence("first")
        second, _, manifest = self.evidence("second")
        manifest["protocol"]["added_environment_note"] = "different configuration note"
        self.write(second.parent / "manifest.json", manifest)
        report = self.analyze(first, second)
        pair = report["exact_shape_repeat_comparisons"][0]["pairwise_run_comparisons"][0]
        self.assertFalse(pair["exact_protocol_equal"])
        self.assertFalse(pair["same_protocol_runtime_repeat"])
        self.assertTrue(pair["runtime_hash_inventory_equal"])

    def test_malformed_record_is_visible_and_plan_case_still_missing(self):
        path, rows, _ = self.evidence()
        rows[0]["case"].pop("config")
        self.write(path, rows)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(len(report["runs"][0]["unresolved_records"]), 1)
        self.assertEqual(len(report["runs"][0]["record_inventory"]), 4)
        self.assertEqual(report["runs"][0]["groups"][0]["missing_case_ids"], [rows[0]["case_id"]])

    def test_inputs_unchanged_and_source_file_hashes_bound(self):
        path, _, _ = self.evidence()
        before = {file:file.read_bytes() for file in self.root.rglob("*") if file.is_file()}
        report = self.analyze(path)
        self.assertEqual(before, {file:file.read_bytes() for file in self.root.rglob("*") if file.is_file()})
        source = report["runs"][0]["source"]
        self.assertEqual(source["measurements"]["sha256"], hashlib.sha256(before[path]).hexdigest())
        self.assertEqual(source["manifest"]["sha256"], hashlib.sha256(before[path.parent/"manifest.json"]).hexdigest())

    def test_unmatched_layer_never_fills_another_config(self):
        path, _, _ = self.evidence()
        changed = layer_document(); changed["physical"]["config"]["rope_base"] = 20000
        self.write(self.layer, changed)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("no layer exactly matches", report["runs"][0]["groups"][0]["errors"][0])

    def test_duplicate_runs_cannot_create_replicates(self):
        path, _, _ = self.evidence()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.analyze(path, path)

    def test_cli_writes_lf_report_and_does_not_overwrite(self):
        path, _, _ = self.evidence()
        output = self.root / "analysis.json"
        args = ["--measurements", str(path), "--layer", str(self.layer), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(analyzer.main(args), 0)
        original = output.read_bytes()
        self.assertNotIn(b"\r\n", original)
        with self.assertRaisesRegex(ValueError, "already exists"):
            analyzer.main(args)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_rejected_evidence_writes_report_and_returns_nonzero(self):
        path, rows, _ = self.evidence()
        self.write(path, rows[:-1])
        output = self.root / "analysis.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = analyzer.main(["--measurements", str(path), "--layer", str(self.layer), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.read_text())["verification_passed"])


if __name__ == "__main__":
    unittest.main()
