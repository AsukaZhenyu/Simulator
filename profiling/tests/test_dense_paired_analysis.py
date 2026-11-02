"""Paired analysis tests use synthetic captures; subprocesses never run."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_dense_paired as analyzer
import run_dense_paired_measurements as runner
from llm_infer_model.m2.dense_contract import resident_accounting
from test_dense_paired import fixture


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dense-paired-analysis-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "synthetic-probe.exe"
        self.binary.write_bytes(b"never executed")
        self.binary_hash = hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")

    def native(self, selected, mode, timing=None):
        _, payload, _ = fixture(mode)
        payload = copy.deepcopy(payload)
        payload.update(config=selected["config"], q=selected["tokens"], history=selected["visible_history"])
        inventory = resident_accounting(selected["config"], "block", selected["tokens"], selected["visible_history"])
        payload.update(inventory)
        shared = payload["shared_memory_inventory"]
        delta = inventory["resident_logical_bytes"] - shared["resident_logical_bytes"]
        shared["resident_logical_bytes"] += delta
        shared["resident_allocated_bytes"] += delta
        shared["physical_tensor_and_work_allocated_bytes"] += delta
        if timing:
            payload["stage_wall_ns"] = timing.get("stages", payload["stage_wall_ns"])
            payload["stage_sum_wall_ns"] = [sum(payload["stage_wall_ns"][stage][i] for stage in runner.STAGES) for i in range(4)]
            payload["serial_wall_ns"] = timing.get("serial", [value+10 for value in payload["stage_sum_wall_ns"]])
            payload["joint_wall_ns"] = timing.get("joint", payload["joint_wall_ns"])
        return payload

    def evidence(self, name="run", *, mode="graph", shapes=None, timings=None, repeats=2, split=None,
                 min_warmup_ms=None, cpu_mask=None):
        selected, _, args = fixture(mode)
        selected = copy.deepcopy(selected)
        selected["metadata"] = {"split":"development"}
        cases = []
        for index, changes in enumerate(shapes or [{}]):
            case = copy.deepcopy(selected); case.update(changes); case["id"] = f"case-{index}"
            cases.append(case)
        protocol = {"threads":2, "warmup":10, "samples":4, "seed":17, "repeats":repeats, "compute_mode":mode}
        if min_warmup_ms is not None:
            protocol.update(min_warmup_ms=min_warmup_ms, cpu_mask=cpu_mask)
        args = SimpleNamespace(**protocol)
        plan = [case for case in cases if split is None or case["metadata"]["split"] == split]
        random.Random(17).shuffle(plan)
        jobs = [(repeat, case) for repeat in range(repeats) for case in (plan if repeat % 2 == 0 else list(reversed(plan)))]
        folder = self.root / name
        rows = []
        for index, (repeat, case) in enumerate(jobs):
            payload = self.native(case, mode, (timings or {}).get(repeat))
            if min_warmup_ms is not None:
                affinity = {"requested_logical_ids":cpu_mask, "applied_logical_ids":cpu_mask,
                    "strict_cpu":cpu_mask is not None, "verified":True if cpu_mask is not None else None,
                    "observed_threads":[{"thread_id":100+i, "group":0, "mask_hex":hex(1<<cpu), "logical_id":cpu}
                        for i,cpu in enumerate(cpu_mask or [])], "verification_source":"synthetic_fixture_never_executed"}
                payload.update(min_warmup_ms=min_warmup_ms, warmup_rounds_requested=10,
                    warmup_rounds_effective=20, actual_warmup_elapsed_ns=min_warmup_ms*1000000+1, cpu_affinity=affinity)
            summary = runner.summarize_paired(payload)
            row = {"case_id":case["id"], "case":case, "repeat":repeat, "compute_mode":mode,
                "native_binary_sha256":self.binary_hash, "raw_output":f"raw/{index+1:03d}.json",
                "status":"measured", "native":payload, "summary":summary,
                "quality_flags":runner.paired_quality_flags(summary)}
            rows.append(row)
            self.write_raw(folder, row, args)
        manifest = {"schema_version":runner.SCHEMA, "scope":analyzer.SCOPE, "split":split,
            "protocol":protocol, "suite":{"schema_version":"dense_layer_measurement_suite_v1", "cases":cases},
            "suite_sha256":"a"*64, "model_contract_sha256":"b"*64,
            "benchmark_source_sha256":{"run_dense_paired_measurements.py":"c"*64, "native\\dense_layer_probe.cpp":"d"*64},
            "runtime_sha256":{str(self.binary):self.binary_hash}, "planned_jobs":len(jobs),
            "case_order":[{"repeat":repeat, "case_id":case["id"]} for repeat,case in jobs],
            "completed_jobs":len(rows), "measured_jobs":len(rows)}
        self.write(folder / "manifest.json", manifest)
        self.write(folder / "measurements.json", rows)
        return folder / "measurements.json", rows, manifest

    def write_raw(self, folder, row, args=None):
        args = args or SimpleNamespace(compute_mode=row["compute_mode"], warmup=10, samples=4, seed=17)
        raw = {"case":row["case"], "repeat":row["repeat"], "command":runner.paired_command(row["case"], self.binary, args),
            "returncode":0, "stdout":json.dumps(row["native"]), "stderr":"synthetic capture never executed"}
        self.write(folder / row["raw_output"], raw)

    def analyze(self, *paths):
        return analyzer.analyze_runs(list(paths))

    def test_independent_repeat_error_is_not_same_sample_bookkeeping(self):
        timing = {0:{"stages":{stage:[100]*4 for stage in runner.STAGES}, "serial":[310]*4, "joint":[250]*4},
                  1:{"stages":{stage:[200]*4 for stage in runner.STAGES}, "serial":[610]*4, "joint":[500]*4}}
        path, _, _ = self.evidence(timings=timing)
        report = self.analyze(path)
        self.assertTrue(report["verification_passed"])
        validation = report["groups"][0]["independent_repeat_validation"][0]
        self.assertEqual(validation["predicted_stage_medians_sum_ns"], 300)
        self.assertEqual(validation["observed_independent_serial_median_ns"], 610)
        self.assertAlmostEqual(validation["signed_relative_error"], -310/610)
        first = report["runs"][0]["records"][0]
        self.assertAlmostEqual(first["same_repeat_clock_bookkeeping"]["signed_ratio"], -10/310)
        self.assertNotEqual(validation["signed_relative_error"], first["same_repeat_clock_bookkeeping"]["signed_ratio"])
        self.assertFalse(validation["same_batch_bookkeeping_used_as_accuracy"])
        self.assertFalse(first["segmented_vs_joint_execution"]["model_accuracy_claimed"])

    def test_ab_ba_subseries_are_explicit_and_all_samples_survive(self):
        stages = {stage:[10, 20, 12, 22] for stage in runner.STAGES}
        path, rows, _ = self.evidence(timings={0:{"stages":stages}})
        report = self.analyze(path)
        observed = report["runs"][0]["records"][0]
        self.assertEqual(observed["samples"]["stage_wall_ns"], stages)
        self.assertEqual(observed["order_strata"]["AB"]["indices"], [0, 2])
        self.assertEqual(observed["order_strata"]["BA"]["indices"], [1, 3])
        self.assertEqual(observed["order_strata"]["AB"]["samples"]["stage_wall_ns"]["projection"], [10, 12])
        self.assertEqual(observed["order_strata"]["BA"]["summary"]["stages"]["projection"]["median_ns"], 21)

    def test_stage_variability_is_retained_even_under_stable_sum(self):
        stages = {"projection":[10,190,10,190], "attention":[190,10,190,10], "dense_ffn":[100]*4}
        path, _, _ = self.evidence(timings={0:{"stages":stages, "serial":[310]*4, "joint":[250]*4}})
        report = self.analyze(path)
        first = report["runs"][0]["records"][0]
        self.assertTrue(report["verification_passed"])
        self.assertEqual(first["quality_flags"], ["stage:projection", "stage:attention"])
        self.assertFalse(first["all_reported_timing_cvs_le_0_1"])
        stat = report["independent_repeat_error_statistics_by_split_and_mode"][0]
        self.assertEqual(stat["all_verified_errors"]["count"], 1)
        self.assertEqual(stat["stable_evidence_errors"]["count"], 0)
        self.assertEqual(len(stat["flagged_observations_ineligible_for_error_acceptance"]), 1)
        self.assertEqual(first["samples"]["stage_wall_ns"], stages)

    def test_development_and_holdout_statistics_are_separate(self):
        path, _, _ = self.evidence(shapes=[{}, {"phase":"decode", "tokens":1, "visible_history":3,
                                                "metadata":{"split":"held_out"}}])
        report = self.analyze(path)
        self.assertEqual(len(report["groups"]), 2)
        stats = {(row["split"],row["compute_mode"]):row for row in report["independent_repeat_error_statistics_by_split_and_mode"]}
        self.assertEqual(stats[("development","graph")]["all_verified_errors"]["count"], 1)
        self.assertEqual(stats[("held_out","graph")]["all_verified_errors"]["count"], 1)
        self.assertEqual(stats[("held_out","plan")]["all_verified_errors"]["count"], 0)
        for row in stats.values():
            if row["all_verified_errors"]["count"]:
                self.assertIsNotNone(row["all_verified_errors"]["worst_observation"])

    def test_matched_graph_plan_estimates_are_compared_without_pooling(self):
        graph, _, _ = self.evidence("graph")
        plan, _, _ = self.evidence("plan", mode="plan", timings={0:{"serial":[400]*4},1:{"serial":[410]*4}})
        report = self.analyze(graph, plan)
        self.assertEqual(len(report["matched_graph_plan_comparisons"]), 2)
        comparison = report["matched_graph_plan_comparisons"][0]
        self.assertTrue(comparison["protocol_equal_except_compute_mode"])
        self.assertTrue(comparison["native_binary_hash_equal"])
        self.assertFalse(comparison["samples_pooled"])
        self.assertFalse(comparison["model_accuracy_claimed"])
        self.assertEqual(comparison["quantities"]["serial_wall_ns"]["right_median_ns"], 400)

    def test_failed_repeat_remains_visible_without_validation_substitution(self):
        path, rows, manifest = self.evidence()
        rows[0].update(status="failed", error="synthetic failure")
        manifest["measured_jobs"] = 1
        self.write(path, rows); self.write(path.parent/"manifest.json", manifest)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(report["rejected_record_count"], 1)
        first = report["runs"][0]["records"][0]
        self.assertEqual(first["captured_status"], "failed")
        self.assertEqual(first["error"], "synthetic failure")
        self.assertIn("raw_sha256", first)
        validation = report["groups"][0]["independent_repeat_validation"][0]
        self.assertEqual(validation["status"], "unavailable")
        self.assertEqual(validation["available_verified_repeats"], [1])

    def test_missing_repeat_jobs_are_reported(self):
        path, rows, manifest = self.evidence()
        rows.pop(); manifest.update(completed_jobs=1, measured_jobs=1)
        self.write(path, rows); self.write(path.parent/"manifest.json", manifest)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(len(report["runs"][0]["missing_jobs"]), 1)
        self.assertEqual(report["runs"][0]["missing_jobs"][0]["repeat"], 1)
        self.assertEqual(report["groups"][0]["independent_repeat_validation"][0]["status"], "unavailable")

    def test_raw_summary_command_binary_repeat_and_mode_tampering_rejects(self):
        for kind in ("raw", "summary", "command", "binary", "repeat", "mode"):
            with self.subTest(kind=kind):
                path, rows, _ = self.evidence(kind)
                if kind in ("raw", "command"):
                    raw_path = path.parent/rows[0]["raw_output"]
                    raw = json.loads(raw_path.read_text())
                    if kind == "raw": raw["stdout"] = "{}"
                    else: raw["command"][-1] = "plan"
                    self.write(raw_path, raw)
                elif kind == "summary": rows[0]["summary"]["serial_wall_ns"]["median_ns"] = 1
                elif kind == "binary": rows[0]["native_binary_sha256"] = "0"*64
                elif kind == "repeat": rows[0]["repeat"] = 1
                else: rows[0]["compute_mode"] = "plan"
                self.write(path, rows)
                report = self.analyze(path)
                self.assertFalse(report["verification_passed"])
                self.assertFalse(report["runs"][0]["records"][0]["raw_samples_verified"])
                self.assertTrue(report["runs"][0]["records"][0]["errors"])

    def test_native_binding_failure_is_not_hidden_by_matching_raw(self):
        path, rows, _ = self.evidence()
        rows[0]["native"]["paired_validation"]["joint"]["stage_port_digests"]["q"] = "f"*16
        self.write_raw(path.parent, rows[0]); self.write(path, rows)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(report["rejected_record_count"], 1)

    def test_source_hash_inventory_and_manifest_count_tampering_rejects(self):
        for kind in ("source", "contract", "count", "order", "seed"):
            with self.subTest(kind=kind):
                path, _, manifest = self.evidence(kind)
                if kind == "source": manifest["benchmark_source_sha256"]["native\\dense_layer_probe.cpp"] = "unknown"
                elif kind == "contract": manifest["model_contract_sha256"] = None
                elif kind == "count": manifest["completed_jobs"] = 3
                elif kind == "order": manifest["case_order"].reverse()
                else: manifest["protocol"]["seed"] = 18
                self.write(path.parent/"manifest.json", manifest)
                report = self.analyze(path)
                self.assertFalse(report["verification_passed"])
                self.assertEqual(len(report["runs"][0]["record_inventory"]), 2)
                if kind != "seed":
                    self.assertEqual(report["groups"], [])

    def test_unrecorded_quality_flags_cannot_hide_stage_cv(self):
        stages = {stage:[100,1000,100,1000] for stage in runner.STAGES}
        path, rows, _ = self.evidence(timings={0:{"stages":stages}})
        rows[0]["quality_flags"] = []
        self.write(path, rows)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("quality flags", report["runs"][0]["records"][0]["errors"][0])

    def test_split_must_be_predeclared_in_manifest_suite(self):
        path, _, manifest = self.evidence()
        manifest["suite"]["cases"][0]["metadata"] = {}
        self.write(path.parent/"manifest.json", manifest)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("predeclare", report["runs"][0]["errors"][0])

    def test_explicit_split_selection_lists_unmeasured_cases(self):
        path, _, _ = self.evidence(shapes=[{}, {"phase":"decode", "tokens":1, "visible_history":3,
                                                "metadata":{"split":"held_out"}}], split="held_out")
        report = self.analyze(path)
        self.assertTrue(report["verification_passed"])
        self.assertEqual(report["runs"][0]["excluded_by_explicit_split_case_ids"], ["case-0"])
        self.assertEqual(report["groups"][0]["shape"]["split"], "held_out")

    def test_duplicate_exact_shapes_are_rejected_instead_of_overwriting_repeats(self):
        path, _, _ = self.evidence(shapes=[{}, {}])
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("duplicate exact", report["runs"][0]["errors"][0])

    def test_input_files_are_unchanged_and_bound_to_hashes(self):
        path, _, _ = self.evidence()
        before = {file:file.read_bytes() for file in self.root.rglob("*") if file.is_file()}
        report = self.analyze(path)
        self.assertEqual(before, {file:file.read_bytes() for file in self.root.rglob("*") if file.is_file()})
        self.assertEqual(report["runs"][0]["source"]["measurements"]["sha256"], hashlib.sha256(before[path]).hexdigest())
        self.assertTrue(report["runs"][0]["records"][0]["source_hash_binding"]["native_binary_matches_manifest"])

    def test_cli_new_lf_report_and_failure_exit_status(self):
        path, _, _ = self.evidence()
        output = self.root/"analysis.json"
        args = ["--measurements",str(path),"--output",str(output)]
        with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(analyzer.main(args), 0)
        self.assertNotIn(b"\r\n", output.read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"): analyzer.main(args)
        failed, rows, manifest = self.evidence("failed")
        rows[0].update(status="failed", error="failure"); manifest["measured_jobs"] = 1
        self.write(failed, rows); self.write(failed.parent/"manifest.json", manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(analyzer.main(["--measurements",str(failed),"--output",str(self.root/"failed-analysis.json")]), 1)

    def test_duplicate_runs_cannot_masquerade_as_repeats(self):
        path, _, _ = self.evidence()
        with self.assertRaisesRegex(ValueError, "distinct"): self.analyze(path, path)

    def test_explicit_duration_and_observed_affinity_are_preserved(self):
        path, rows, _ = self.evidence(min_warmup_ms=250, cpu_mask=[0,2])
        report = self.analyze(path)
        self.assertTrue(report["verification_passed"])
        shape = report["groups"][0]["shape"]
        self.assertEqual(shape["min_warmup_ms"], 250)
        self.assertEqual(shape["cpu_mask"], [0,2])
        self.assertTrue(shape["duration_warmup_affinity_protocol_declared"])
        evidence = report["runs"][0]["records"][0]["warmup_affinity_evidence"]
        self.assertEqual(evidence["warmup_rounds_effective"], 20)
        self.assertEqual(evidence["actual_warmup_elapsed_ns"], 250000001)
        self.assertEqual(evidence["cpu_affinity"], rows[0]["native"]["cpu_affinity"])

    def test_old_protocol_unknown_fields_do_not_become_zero_or_claimed_binding(self):
        path, _, _ = self.evidence()
        report = self.analyze(path)
        shape = report["groups"][0]["shape"]
        self.assertIsNone(shape["min_warmup_ms"])
        self.assertIsNone(shape["cpu_mask"])
        self.assertFalse(shape["duration_warmup_affinity_protocol_declared"])
        self.assertIsNone(report["runs"][0]["records"][0]["warmup_affinity_evidence"]["cpu_affinity"])

    def test_declared_unbound_affinity_stays_unasserted(self):
        path, _, _ = self.evidence(min_warmup_ms=250)
        report = self.analyze(path)
        self.assertTrue(report["verification_passed"])
        affinity = report["runs"][0]["records"][0]["warmup_affinity_evidence"]["cpu_affinity"]
        self.assertIsNone(affinity["requested_logical_ids"])
        self.assertIsNone(affinity["applied_logical_ids"])
        self.assertIsNone(affinity["verified"])

    def test_warmup_and_affinity_failures_are_not_repaired_by_analyzer(self):
        for kind in ("elapsed", "rounds", "mask", "thread"):
            with self.subTest(kind=kind):
                path, rows, manifest = self.evidence(kind, min_warmup_ms=250, cpu_mask=[0,2])
                payload = rows[0]["native"]
                if kind == "elapsed": payload["actual_warmup_elapsed_ns"] = 1000
                elif kind == "rounds": payload["warmup_rounds_effective"] = 1
                elif kind == "mask": payload["cpu_affinity"]["observed_threads"][0]["mask_hex"] = "0x3"
                else: payload["cpu_affinity"]["observed_threads"][1]["thread_id"] = 100
                self.write_raw(path.parent, rows[0], SimpleNamespace(**manifest["protocol"]))
                self.write(path, rows)
                report = self.analyze(path)
                self.assertFalse(report["verification_passed"])
                self.assertEqual(report["rejected_record_count"], 1)

    def test_different_warmup_or_masks_are_separate_exact_protocol_shapes(self):
        first, _, _ = self.evidence("first", min_warmup_ms=250, cpu_mask=[0,2])
        second, _, _ = self.evidence("second", min_warmup_ms=500, cpu_mask=[0,2])
        third, _, _ = self.evidence("third", mode="plan", min_warmup_ms=250, cpu_mask=[2,4])
        report = self.analyze(first, second, third)
        self.assertTrue(report["verification_passed"])
        self.assertEqual(len(report["groups"]), 3)
        self.assertEqual(report["matched_graph_plan_comparisons"], [])

    def test_partial_new_protocol_is_rejected(self):
        path, _, manifest = self.evidence()
        manifest["protocol"]["min_warmup_ms"] = 250
        self.write(path.parent/"manifest.json", manifest)
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("exactly", report["runs"][0]["errors"][0])

    def test_same_physical_shape_cannot_be_in_both_splits_in_one_suite(self):
        path, _, _ = self.evidence(shapes=[{}, {"metadata":{"split":"held_out"}}])
        report = self.analyze(path)
        self.assertFalse(report["verification_passed"])
        self.assertIn("development/held_out overlap", report["runs"][0]["errors"][0])
        self.assertFalse(report["runs"][0]["split_isolation_valid"])
        self.assertEqual(len(report["development_held_out_split_conflicts"]), 1)
        self.assertEqual(report["unprocessed_record_count"], 4)
        self.assertEqual(len(report["runs"][0]["record_inventory"]), 4)
        self.assertEqual(report["groups"], [])

    def test_cross_run_split_conflict_rejects_scoring_and_preserves_samples(self):
        development, _, _ = self.evidence("development")
        held_out, rows, _ = self.evidence("held-out", mode="plan", shapes=[{"metadata":{"split":"held_out"}}])
        report = self.analyze(development, held_out)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(len(report["development_held_out_split_conflicts"]), 1)
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["rejected_record_count"], 4)
        self.assertEqual(report["matched_graph_plan_comparisons"], [])
        for run in report["runs"]:
            self.assertFalse(run["split_isolation_valid"])
            for row in run["records"]:
                self.assertEqual(row["status"], "rejected")
                self.assertTrue(row["raw_samples_verified"])
                self.assertTrue(row["raw_evidence_verification_passed"])
                self.assertIn("split conflict", row["errors"][0])
                self.assertIn("summary", row)
                self.assertIn("samples", row)
        self.assertEqual(report["runs"][1]["records"][0]["samples"]["stage_wall_ns"], rows[0]["native"]["stage_wall_ns"])

    def test_mode_warmup_mask_and_phase_cannot_disguise_split_leakage(self):
        development, _, _ = self.evidence("development", min_warmup_ms=250, cpu_mask=[0,2],
            shapes=[{"phase":"prefill", "tokens":1, "visible_history":3}])
        held_out, _, _ = self.evidence("held-out", mode="plan", min_warmup_ms=500, cpu_mask=[2,4],
            shapes=[{"phase":"decode", "tokens":1, "visible_history":3, "metadata":{"split":"held_out"}}])
        report = self.analyze(development, held_out)
        self.assertFalse(report["verification_passed"])
        conflict = report["development_held_out_split_conflicts"][0]
        self.assertEqual(set(conflict["physical_identity"]), {"config", "threads", "batch", "tokens", "visible_history"})
        self.assertEqual(report["groups"], [])

    def test_conflict_does_not_silently_delete_other_verified_geometries(self):
        development, _, _ = self.evidence("development", shapes=[{}, {"phase":"decode", "tokens":1, "visible_history":8}])
        held_out, _, _ = self.evidence("held-out", shapes=[{"metadata":{"split":"held_out"}},
            {"phase":"decode", "tokens":1, "visible_history":3, "metadata":{"split":"held_out"}}])
        report = self.analyze(development, held_out)
        self.assertFalse(report["verification_passed"])
        self.assertEqual(report["verified_record_count"], 4)
        self.assertEqual(report["rejected_record_count"], 4)
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual({group["shape"]["visible_history"] for group in report["groups"]}, {3,8})

    def test_same_split_across_modes_remains_valid(self):
        graph, _, _ = self.evidence("graph")
        plan, _, _ = self.evidence("plan", mode="plan")
        report = self.analyze(graph, plan)
        self.assertTrue(report["verification_passed"])
        self.assertEqual(report["development_held_out_split_conflicts"], [])
        self.assertEqual(len(report["matched_graph_plan_comparisons"]), 2)


if __name__ == "__main__":
    unittest.main()
