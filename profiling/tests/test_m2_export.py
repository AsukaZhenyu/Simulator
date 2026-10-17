"""Exact export boundaries; native fixtures are synthetic and never run hardware."""
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
import export_m2_macros as exporter
from macro_contract import macro_signature, physical_weight_dtypes, implementation_variant, validate_macro_case
from profile_contract import summarize_samples
from run_macro_measurements import included_operations, native_command, _EXCLUDED
from llm_infer_model.m2.costs import MacroShape, parse_cost_table
from llm_infer_model.m2.spec import parse_layer_document
from llm_infer_model.tensor.spec import Unsupported


def layer_document():
    return {"format_version": "m2-layer-0.2", "id": "synthetic-export-fixture",
        "provenance": "synthetic_layer", "hidden_size": 16, "hidden_bytes_per_token": 64,
        "kv_bytes_per_token": 64, "attention": "full_causal", "ffn": "dense",
        "physical": {"contract": "ggml_cpu_partial_dense_macros_v1", "scope": "partial_decoder_block",
            "input_boundary": "post_positional_qkv", "port_layout": "materialized_contiguous",
            "intermediate": 32, "heads": 4, "kv_heads": 2, "head_dim": 4,
            "attention_gate": "sigmoid", "threads": 2, "ffn_weight_policy": "uniform_declared_weight_dtype",
            "weight_groups": {"attention": {"output": "W_o"}, "dense_ffn": {"gate": "W_gate", "up": "W_up", "down": "W_down"}},
            "activation_dtype": "f32", "kv_dtype": "f32", "kv_allocation": "logical_blocks",
            "excluded_operations": ["qkv_projection", "normalization", "RoPE", "residual_add"]},
        "weights": [{"id": name, "dtype": "f32", "shape": shape,
                     "storage_bytes": size, "initial_location": "dram"}
                    for name, shape, size in (("W_o", [16,16],1024), ("W_gate",[16,32],2048),
                                             ("W_up",[16,32],2048), ("W_down",[32,16],2048))],
        "tasks": [{"id": "attn", "kind": "attention", "depends_on": []},
                  {"id": "ffn", "kind": "dense_ffn", "depends_on": ["attn"]}]}


def case_for(kind, *, phase="prefill", tokens=3, history=0):
    case = {"id": kind + "_" + phase, "kind": kind, "phase": phase, "batch": 1,
            "tokens": tokens, "visible_history": history, "hidden": 16,
            "weight_dtype": "f32", "threads": 2}
    if kind == "attention":
        case.update(heads=4, kv_heads=2, head_dim=4, attention_gate="sigmoid")
    else:
        case["intermediate"] = 32
    return validate_macro_case(case)


def native_for(case):
    attention = case["kind"] == "attention"
    q, history = case["tokens"], case["visible_history"]
    inputs = q*192 if attention else q*64
    output = q*64
    weight = 1024 if attention else 6144
    cache = (history+q)*64 if attention else 0
    return {"schema": "ggml_macro_probe/1", "kind": "macro", "macro": case["kind"],
        "backend": "CPU", "batch": 1, "hidden": 16, "intermediate": case["intermediate"],
        "heads": case["heads"], "kv_heads": case["kv_heads"], "head_dim": case["head_dim"],
        "q": q, "history": history, "dtype": "f32", "weight_dtypes": physical_weight_dtypes(case),
        "activation_dtype": "f32", "kv_dtype": "f32" if attention else "not_applicable",
        "attention_gate": case["attention_gate"], "implementation_variant": implementation_variant(case),
        "history_semantics": "existing_prefix_length_in_causal_attention" if attention else "not_used_by_dense_ffn",
        "threads": 2, "threads_requested": 2, "threads_applied": 2,
        "warmup": 2, "samples": 3, "seed": 17, "setup_compute_repetitions": 1,
        "timing": "steady_clock_graph_compute_including_sync", "scope": "resident_repeated_partial_block_macro",
        "included": included_operations(case["kind"]), "excluded": _EXCLUDED,
        "weight_bytes": weight, "input_bytes": inputs, "output_bytes": output,
        "kv_history_bytes": history*64 if attention else 0, "kv_new_bytes": q*64 if attention else 0,
        "kv_cache_bytes": cache, "internal_tensor_bytes_upper_bound": 64,
        "cpu_cplan_work_size_bytes": 32, "workspace_bytes": 128,
        "workspace_source": "ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
        "memory_accounting": {"resident_logical_bytes": weight+inputs+output+cache,
            "resident_allocated_bytes": weight+inputs+output+cache, "internal_allocated_buffer_bytes": 96,
            "internal_logical_tensor_bytes_sum": 64, "cpu_cplan_work_size_bytes": 32, "workspace_bytes": 128,
            "workspace_scope": "internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads",
            "workspace_bound": "conservative_allocated_capacity",
            "kv_new_bytes_semantics": "appended_slots_in_cache_distinct_from_new_kv_input_tensors"},
        "all_outputs_finite": True, "reference_check": {"passed": True},
        "kv_append_check": {"passed": True, "append_exact": True, "history_unchanged": True} if attention else None,
        "wall_ns": [1000, 1001, 999]}


class ExactExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="m2-export-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "raw").mkdir()
        self.layer_doc = layer_document()
        self.layer = parse_layer_document(self.layer_doc)
        self.binary = self.root / "bin" / ("ggml-macro-probe.exe" if os.name == "nt" else "ggml-macro-probe")
        self.binary_hash = hashlib.sha256(b"synthetic fixture; not executed").hexdigest()
        self.cases = [case_for("attention"), case_for("dense_ffn")]
        self.manifest = {"schema_version": "macro_measurement_profile_v1", "contract": "ggml_cpu_partial_dense_macros_v1",
            "profile_id": "synthetic-export-tests", "suite": {"schema_version": "macro_measurement_suite_v1", "cases": self.cases},
            "protocol": {"backend": "CPU", "batch": 1, "threads": 2, "warmup": 2, "samples": 3, "seed": 17,
                "sequential_execution": True, "graphs_per_sample": 1, "setup_compute_repetitions": 1, "total_untimed_compute_repetitions": 3},
            "software": {"runtime_sha256": {str(self.binary): self.binary_hash}}}
        self.rows = []
        for i, case in enumerate(self.cases):
            native = native_for(case)
            summary = summarize_samples(native["wall_ns"])
            row = {"case_id": case["id"], "case": case, "status": "measured",
                   "signature": macro_signature(case, native_binary_sha256=self.binary_hash),
                   "raw_output": f"raw/{i+1:03d}.json", "native": {**native, "expected_samples": 3, "summary": summary},
                   "summary": summary, "quality_flags": [],
                   "eligibility": {"automatic_M2_export": False, "M2_memory_accounting_alignment_verified": False, "complete_layer": False}}
            self.rows.append(row)
            self.write_raw(i, native)

    def write_raw(self, index, native):
        case = self.rows[index]["case"]
        raw = {"case": case, "command": native_command(case, self.binary.parent, SimpleNamespace(warmup=2,samples=3,seed=17)),
               "returncode": 0, "stdout": json.dumps(native), "stderr": "synthetic fixture; native not executed"}
        self.path(index).write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    def path(self, index=0):
        return self.root / self.rows[index]["raw_output"]

    def update_native(self, index, changes, *, summaries=True):
        native = {k:v for k,v in self.rows[index]["native"].items() if k not in {"expected_samples", "summary"}}
        native.update(changes)
        self.rows[index]["native"].update(changes)
        if summaries:
            summary = summarize_samples(native["wall_ns"])
            self.rows[index]["native"]["summary"] = summary
            self.rows[index]["summary"] = summary
        self.write_raw(index, native)

    def export(self, **changes):
        return exporter.export_cost_table(self.layer, self.rows, raw_directory=self.root, manifest=self.manifest, **changes)

    def test_exact_rows_export_with_source_and_memory_boundaries(self):
        costs, report = self.export()
        self.assertEqual(len(costs["entries"]), 2)
        self.assertFalse(report["calibration_claimed"])
        self.assertFalse(report["validation_only"])
        self.assertFalse(costs["validation_only"])
        self.assertIn(report["records_canonical_sha256"][:16], costs["source"])
        self.assertIn(report["manifest_canonical_sha256"][:16], costs["source"])
        self.assertFalse(report["complete_layer"])
        self.assertEqual(report["entries"][0]["memory_boundary_check"]["input_ports_bytes"], {"q":192,"k":96,"v":96,"gate":192})
        self.assertEqual(report["entries"][0]["memory_boundary_check"]["resident_logical_bytes"], 1984)
        self.assertEqual(report["entries"][1]["memory_boundary_check"]["resident_logical_bytes"], 6528)
        self.assertEqual(report["entries"][0]["raw_sha256"], hashlib.sha256(self.path().read_bytes()).hexdigest())
        self.assertEqual(costs["entries"][0]["workspace_bytes"], 128)
        self.assertEqual(costs["entries"][0]["accounting"], "per_request")

    def test_source_identity_changes_when_observed_samples_change(self):
        first_costs, first_report = self.export()
        self.update_native(0, {"wall_ns": [1001,1002,1000]})
        second_costs, second_report = self.export()
        self.assertNotEqual(first_costs["source"], second_costs["source"])
        self.assertNotEqual(first_report["records_canonical_sha256"], second_report["records_canonical_sha256"])
        self.assertEqual(first_report["manifest_canonical_sha256"], second_report["manifest_canonical_sha256"])

    def test_export_does_not_mutate_records(self):
        original = copy.deepcopy((self.rows, self.manifest, self.layer_doc))
        self.export()
        self.assertEqual((self.rows, self.manifest, self.layer_doc), original)

    def test_missing_history_remains_missing_even_for_ffn(self):
        costs, _ = self.export()
        table = parse_cost_table(costs)
        with self.assertRaises(Unsupported):
            table.lookup("dense_ffn", "cpu", MacroShape("prefill",1,3,1))

    def test_unstable_samples_require_an_explicit_validation_waiver(self):
        self.update_native(0, {"wall_ns": [1000,10000,1000]})
        self.rows[0]["quality_flags"] = ["high_observed_variability_cv_gt_0.1"]
        with self.assertRaisesRegex(ValueError, "unstable"):
            self.export()
        costs, report = self.export(allow_unstable=True)
        self.assertTrue(report["validation_only"])
        self.assertTrue(costs["validation_only"])
        self.assertEqual(report["entries"][0]["quality_flags"], self.rows[0]["quality_flags"])
        self.assertIn("validation_only=true", costs["comment"])

    def test_unstable_cv_is_detected_even_if_flag_is_missing(self):
        self.update_native(0, {"wall_ns": [1000,10000,1000]})
        with self.assertRaisesRegex(ValueError, "unstable"):
            self.export()

    def test_unknown_quality_flags_cannot_be_waived(self):
        self.rows[0]["quality_flags"] = ["reference_suspect"]
        with self.assertRaisesRegex(ValueError, "quality flags"):
            self.export(allow_unstable=True)

    def test_failed_record_cannot_be_waived(self):
        self.rows[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "failed"):
            self.export(allow_unstable=True)

    def test_unknown_workspace_cannot_become_zero(self):
        self.update_native(0, {"workspace_bytes": None})
        with self.assertRaisesRegex(ValueError, "workspace is unknown"):
            self.export()

    def test_workspace_allocator_plus_scratch_must_agree(self):
        self.update_native(0, {"workspace_bytes": 127})
        with self.assertRaisesRegex(ValueError, "workspace"):
            self.export()

    def test_logical_resident_inventory_must_match_model_ports(self):
        for field in ("input_bytes", "weight_bytes", "output_bytes", "kv_cache_bytes", "kv_new_bytes"):
            with self.subTest(field=field):
                original = copy.deepcopy(self.rows)
                self.update_native(0, {field: self.rows[0]["native"][field]+1})
                with self.assertRaisesRegex(ValueError, field):
                    self.export()
                self.rows = original
                self.write_raw(0, {k:v for k,v in self.rows[0]["native"].items() if k not in {"expected_samples","summary"}})

    def test_old_signature_and_tampered_hash_are_refused(self):
        for change in ("missing_field", "digest"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.rows[0]["signature"])
                if change == "missing_field":
                    del self.rows[0]["signature"]["setup_compute_repetitions"]
                else:
                    self.rows[0]["signature"]["physical_contract_sha256"] = "0"*64
                with self.assertRaisesRegex(ValueError, "signature"):
                    self.export()
                self.rows[0]["signature"] = original

    def test_model_dtype_and_variant_mismatch_is_refused(self):
        for mismatch in ("dtype", "variant", "gate", "threads"):
            with self.subTest(mismatch=mismatch):
                document = copy.deepcopy(self.layer_doc)
                if mismatch == "dtype":
                    document["weights"][0].update(dtype="f16", storage_bytes=512)
                elif mismatch == "variant":
                    document["physical"]["ffn_weight_policy"] = "explicit_per_matrix_dtypes"
                elif mismatch == "gate":
                    document["physical"]["attention_gate"] = "none"
                else:
                    document["physical"]["threads"] = 4
                self.layer = parse_layer_document(document)
                with self.assertRaisesRegex(ValueError, "model physical signature"):
                    self.export()

    def test_native_binding_geometry_must_match_even_if_raw_and_row_agree(self):
        self.update_native(0, {"q":4})
        with self.assertRaisesRegex(ValueError, "native q"):
            self.export()

    def test_modified_native_stdout_is_refused(self):
        raw = json.loads(self.path().read_text())
        native = json.loads(raw["stdout"])
        native["wall_ns"][0] += 1
        raw["stdout"] = json.dumps(native)
        self.path().write_text(json.dumps(raw), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "raw stdout"):
            self.export()

    def test_modified_summary_is_refused(self):
        self.rows[0]["summary"] = {**self.rows[0]["summary"], "median_ns": 1}
        with self.assertRaisesRegex(ValueError, "summary"):
            self.export()

    def test_failed_native_exit_is_refused(self):
        raw = json.loads(self.path().read_text())
        raw["returncode"] = 1
        self.path().write_text(json.dumps(raw), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "did not succeed"):
            self.export()

    def test_native_binary_hash_must_be_bound_to_manifest(self):
        self.manifest["software"]["runtime_sha256"][str(self.binary)] = "0"*64
        with self.assertRaisesRegex(ValueError, "binary signature hash"):
            self.export()

    def test_raw_path_cannot_escape_run_directory(self):
        self.rows[0]["raw_output"] = "../other-run.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.export()

    def test_duplicate_shapes_are_not_silently_overwritten(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export()

    def test_raw_command_must_match_requested_protocol(self):
        raw = json.loads(self.path().read_text())
        raw["command"][raw["command"].index("--samples")+1] = "9"
        self.path().write_text(json.dumps(raw), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "raw native command"):
            self.export()

    def test_resident_accounting_cannot_understate_the_inventory(self):
        accounting = copy.deepcopy(self.rows[0]["native"]["memory_accounting"])
        accounting["resident_logical_bytes"] -= 1
        self.update_native(0, {"memory_accounting": accounting})
        with self.assertRaisesRegex(ValueError, "resident logical"):
            self.export()

    def test_explicit_phase_selection_records_omitted_unstable_cases(self):
        case = case_for("dense_ffn", phase="decode", tokens=1, history=3)
        self.cases.append(case)
        native = native_for(case)
        native["wall_ns"] = [1000,10000,1000]
        summary = summarize_samples(native["wall_ns"])
        self.rows.append({"case_id": case["id"], "case": case, "status": "measured",
            "signature": macro_signature(case, native_binary_sha256=self.binary_hash),
            "raw_output": "raw/003.json", "native": {**native, "expected_samples":3, "summary":summary},
            "summary": summary, "quality_flags": ["high_observed_variability_cv_gt_0.1"]})
        self.write_raw(2, native)
        for name, document in (("measurements.json",self.rows),("manifest.json",self.manifest),("layer.json",self.layer_doc)):
            (self.root/name).write_text(json.dumps(document),encoding="utf-8",newline="\n")
        output = self.root/"prefill-export"
        argv = ["--measurements",str(self.root/"measurements.json"),"--layer",str(self.root/"layer.json"),
                "--output",str(output),"--phase","prefill"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(exporter.main(argv), 0)
        report = json.loads((output/"export_report.json").read_text())
        self.assertEqual(report["record_selection"]["omitted_case_ids"], [case["id"]])
        self.assertEqual(report["record_selection"]["omitted_records"][0]["quality_flags"], self.rows[2]["quality_flags"])
        self.assertFalse(report["allow_unstable"])

    def test_cli_writes_lf_and_refuses_nonempty_output(self):
        measurements, manifest, layer = self.root/"measurements.json", self.root/"manifest.json", self.root/"layer.json"
        for path, document in ((measurements,self.rows),(manifest,self.manifest),(layer,self.layer_doc)):
            path.write_text(json.dumps(document),encoding="utf-8",newline="\n")
        output = self.root/"export"
        argv = ["--measurements",str(measurements),"--layer",str(layer),"--output",str(output),"--phase","prefill"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(exporter.main(argv), 0)
        report = json.loads((output/"export_report.json").read_text())
        self.assertEqual(report["record_selection"]["selected_case_ids"], [row["case_id"] for row in self.rows])
        self.assertFalse(report["record_selection"]["silent_quality_filtering"])
        for path in output.iterdir():
            self.assertNotIn(b"\r\n", path.read_bytes())
        with self.assertRaisesRegex(ValueError, "empty"):
            exporter.main(argv)


if __name__ == "__main__":
    unittest.main()
