"""Macro contracts and runner failure evidence; fake probes never run hardware."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macro_contract import build_macro_plan, implementation_variant, macro_signature, physical_weight_dtypes, validate_macro_case
import run_macro_measurements as runner


def attention_case(**changes):
    case = {"id": "attn", "kind": "attention", "phase": "prefill", "batch": 1,
        "tokens": 3, "visible_history": 0, "hidden": 16,
        "heads": 4, "kv_heads": 2, "head_dim": 4, "weight_dtype": "f32", "attention_gate": "none"}
    case.update(changes)
    return case


def ffn_case(**changes):
    case = {"id": "ffn", "kind": "dense_ffn", "phase": "decode", "batch": 1,
        "tokens": 1, "visible_history": 3, "hidden": 16, "intermediate": 32, "weight_dtype": "f32"}
    case.update(changes)
    return case


def native_payload(case, **changes):
    case = validate_macro_case(case)
    result = {"schema": "ggml_macro_probe/1", "kind": "macro", "macro": case["kind"],
        "backend": "CPU", "batch": 1, "hidden": case["hidden"], "intermediate": case["intermediate"],
        "heads": case["heads"], "kv_heads": case["kv_heads"], "head_dim": case["head_dim"],
        "q": case["tokens"], "history": case["visible_history"], "dtype": case.get("weight_dtype", "mixed"),
        "weight_dtypes": physical_weight_dtypes(case), "implementation_variant": implementation_variant(case),
        "activation_dtype": "f32", "kv_dtype": "f32" if case["kind"] == "attention" else "not_applicable",
        "timing": "steady_clock_graph_compute_including_sync", "scope": "resident_repeated_partial_block_macro",
        "included": runner.included_operations(case["kind"]), "excluded": runner._EXCLUDED,
        "setup_compute_repetitions": 1,
        "attention_gate": case["attention_gate"], "threads": 2, "threads_requested": 2, "threads_applied": 2,
        "warmup": 5, "samples": 3, "seed": 17,
        "all_outputs_finite": True, "reference_check": {"passed": True},
        "wall_ns": [1000, 3000, 2000], "workspace_bytes": None,
        "workspace_source": "unknown_test_fixture", "memory_accounting": {"source": "mock_only"}}
    if case["kind"] == "attention":
        result["kv_append_check"] = {"passed": True, "append_exact": True, "history_unchanged": True}
    else:
        result["history_semantics"] = "not_used_by_dense_ffn"
    result.update(changes)
    return result


class MacroContractTests(unittest.TestCase):
    def test_batch_and_decode_semantics_are_explicit(self):
        for case in (attention_case(batch=2), attention_case(batch=True),
                     attention_case(phase="decode", tokens=3), attention_case(tokens=0),
                     attention_case(visible_history=-1)):
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_macro_case(case)

    def test_missing_geometry_cannot_be_recovered_from_hidden_or_a_model_name(self):
        for key in ("heads", "kv_heads", "head_dim", "attention_gate", "weight_dtype", "batch", "visible_history"):
            case = attention_case(metadata={"model": "qwen_example"})
            del case[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_macro_case(case)
        case = ffn_case()
        del case["intermediate"]
        with self.assertRaises(ValueError):
            validate_macro_case(case)

    def test_gqa_requires_integer_grouping_but_hidden_need_not_equal_head_output(self):
        case = validate_macro_case(attention_case(hidden=20))
        self.assertEqual(case["hidden"], 20)
        self.assertEqual(case["heads"] * case["head_dim"], 16)
        with self.assertRaises(ValueError):
            validate_macro_case(attention_case(heads=3, kv_heads=2))

    def test_quantized_constraints_apply_to_each_weight_reduction_axis(self):
        for case in (ffn_case(weight_dtype="q4_k", hidden=256, intermediate=32),
                     ffn_case(weight_dtype="q6_k", hidden=16, intermediate=256),
                     attention_case(weight_dtype="q4_k")):
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_macro_case(case)
        self.assertEqual(validate_macro_case(ffn_case(weight_dtype="q6_k", hidden=256, intermediate=256))["weight_dtype"], "q6_k")
        self.assertEqual(validate_macro_case(attention_case(weight_dtype="q4_k", hidden=20, heads=4, head_dim=64))["hidden"], 20)

    def test_unsupported_backend_cannot_fallback_to_cpu(self):
        for backend in ("CUDA0", "gpu", "cpu"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                validate_macro_case(ffn_case(backend=backend))

    def test_unknown_execution_fields_and_typos_are_rejected(self):
        for key, value in (("kv_dtype", "f16"), ("activation_dtype", "f16"), ("rope", True),
                           ("visibile_history", 3), ("implementation_variant", "flash")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_macro_case(attention_case(**{key: value}))
        for case in (ffn_case(heads=False), attention_case(intermediate=False), attention_case(metadata="not an object")):
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_macro_case(case)

    def test_per_matrix_ffn_precision_is_explicit_and_checks_each_reduction_axis(self):
        case = ffn_case(hidden=256, intermediate=256)
        del case["weight_dtype"]
        case["weight_dtypes"] = {"gate": "q4_k", "up": "q4_k", "down": "q6_k"}
        validated = validate_macro_case(case)
        self.assertEqual(physical_weight_dtypes(validated), case["weight_dtypes"])
        self.assertEqual(validate_macro_case(validated), validated)
        with self.assertRaises(ValueError):
            validate_macro_case({**case, "weight_dtype": "f32"})
        with self.assertRaises(ValueError):
            validate_macro_case({**case, "weight_dtypes": {"gate": "q4_k", "down": "q6_k"}})
        with self.assertRaises(ValueError):
            validate_macro_case({**case, "intermediate": 32})
        mixed_axes = {**case, "intermediate": 32, "weight_dtypes": {"gate": "q4_k", "up": "f32", "down": "f16"}}
        self.assertEqual(validate_macro_case(mixed_axes)["intermediate"], 32)

    def test_duplicate_case_ids_and_conflicting_threads_are_rejected(self):
        for cases in ([attention_case(), attention_case()], [ffn_case(threads=4)]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                build_macro_plan({"schema_version": "macro_measurement_suite_v1", "cases": cases}, 2)

    def test_signature_binds_history_geometry_dtype_gate_and_threads(self):
        base = attention_case(threads=2)
        initial = macro_signature(base)
        for changes in ({"visible_history": 3}, {"hidden": 20}, {"head_dim": 8},
                        {"weight_dtype": "f16"}, {"attention_gate": "sigmoid"}, {"threads": 4}):
            with self.subTest(changes=changes):
                changed = macro_signature({**base, **changes})
                self.assertNotEqual(initial["physical_contract_sha256"], changed["physical_contract_sha256"])
        self.assertEqual(initial["backend"], "CPU")
        self.assertEqual(initial["excluded_operations"], ["normalization", "RoPE", "residual_add"])
        self.assertIsNone(initial["native_binary_sha256"])

    def test_contract_validation_does_not_mutate_user_cases(self):
        case = ffn_case(metadata={"model": "example"})
        original = copy.deepcopy(case)
        validate_macro_case(case)
        self.assertEqual(case, original)


class MacroRunnerTests(unittest.TestCase):
    def run_fake(self, payload, *, case=None, failure=None, exitcode=0):
        case = case or attention_case()
        with tempfile.TemporaryDirectory(prefix="macro-runner-test-") as temporary:
            root = Path(temporary)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            executable = binary_dir / ("ggml-macro-probe.exe" if os.name == "nt" else "ggml-macro-probe")
            executable.write_bytes(b"fake_probe_not_executed")
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps({"schema_version": "macro_measurement_suite_v1", "cases": [case]}), encoding="utf-8", newline="\n")
            output = root / "output"
            arguments = ["--suite", str(suite_path), "--native-bin", str(binary_dir), "--output", str(output),
                "--threads", "2", "--samples", "3", "--warmup", "5", "--seed", "17"]
            stdout = payload if isinstance(payload, str) else json.dumps(payload)
            completed = subprocess.CompletedProcess([], exitcode, stdout, "probe diagnostic")
            with (mock.patch.object(runner, "ROOT", root),
                  mock.patch.object(runner, "environment", return_value={"mocked": True}),
                  mock.patch.object(runner.subprocess, "run", return_value=completed, side_effect=failure) as child,
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
                code = runner.main(arguments)
            self.assertEqual(child.call_count, 1)
            self.assertFalse((root / ".measurement.lock").exists())
            return {"code": code,
                "rows": json.loads((output / "measurements.json").read_text(encoding="utf-8")),
                "raw": json.loads((output / "raw" / "001.json").read_text(encoding="utf-8")),
                "manifest": json.loads((output / "manifest.json").read_text(encoding="utf-8")),
                "bytes": {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}}

    def test_exact_macro_run_preserves_raw_samples_and_unknown_workspace(self):
        payload = native_payload(attention_case())
        result = self.run_fake(payload)
        self.assertEqual(result["code"], 0)
        row = result["rows"][0]
        self.assertEqual(row["summary"]["median_ns"], 2000)
        self.assertEqual(json.loads(result["raw"]["stdout"])["wall_ns"], [1000, 3000, 2000])
        self.assertIsNone(row["native"]["workspace_bytes"])
        self.assertFalse(row["eligibility"]["workspace_known"])
        self.assertFalse(row["eligibility"]["automatic_M2_export"])
        self.assertFalse(row["eligibility"]["complete_layer"])
        self.assertTrue(row["signature"]["native_binary_sha256"])

    def test_ffn_records_requested_history_without_claiming_it_is_read(self):
        case = ffn_case()
        result = self.run_fake(native_payload(case), case=case)
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["rows"][0]["signature"]["visible_history"], 3)
        self.assertIn("not_used_by_dense_ffn", result["rows"][0]["signature"]["history_semantics"])
        result = self.run_fake(native_payload(case, history_semantics="unknown"), case=case)
        self.assertEqual(result["code"], 1)

    def test_native_signature_mismatch_cannot_inherit_requested_configuration(self):
        for field, value in (("schema", "ggml_compute_probe/1"), ("macro", "dense_ffn"), ("backend", "CUDA0"),
                             ("q", 1), ("history", 3), ("hidden", 32), ("heads", 8), ("kv_heads", 1),
                             ("dtype", "f16"), ("attention_gate", "sigmoid"), ("threads_applied", 4),
                             ("implementation_variant", "flash_attention"), ("kv_dtype", "f16"),
                             ("timing", "device_event_ns"), ("scope", "whole_layer"), ("setup_compute_repetitions", 0),
                             ("warmup", 4), ("seed", 18), ("batch", True)):
            with self.subTest(field=field):
                result = self.run_fake(native_payload(attention_case(), **{field: value}))
                self.assertEqual(result["code"], 1)
                self.assertEqual(result["rows"][0]["status"], "failed")
                self.assertEqual(json.loads(result["raw"]["stdout"])[field], value)

    def test_mixed_ffn_native_precision_binds_each_matrix(self):
        case = ffn_case(hidden=256, intermediate=256)
        del case["weight_dtype"]
        case["weight_dtypes"] = {"gate": "q4_k", "up": "q4_k", "down": "q6_k"}
        payload = native_payload(case)
        result = self.run_fake(payload, case=case)
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["rows"][0]["signature"]["weight_dtypes"], case["weight_dtypes"])
        command = result["raw"]["command"]
        self.assertNotIn("--dtype", command)
        self.assertEqual(command[command.index("--down-dtype") + 1], "q6_k")
        payload["weight_dtypes"]["down"] = "q4_k"
        result = self.run_fake(payload, case=case)
        self.assertEqual(result["code"], 1)

    def test_attention_missing_or_failed_kv_checks_are_rejected(self):
        for checks in (None, {"passed": True}, {"passed": True, "append_exact": True, "history_unchanged": False}):
            with self.subTest(checks=checks):
                result = self.run_fake(native_payload(attention_case(), kv_append_check=checks))
                self.assertEqual(result["code"], 1)
        result = self.run_fake(native_payload(attention_case(), kv_validation={"append_exact": False, "history_unchanged": True}))
        self.assertEqual(result["code"], 1)

    def test_failed_numerical_check_and_missing_wall_sample_are_rejected(self):
        for changes in ({"reference_check": {"passed": False}}, {"all_outputs_finite": False}, {"wall_ns": [1000, 2000]}):
            with self.subTest(changes=changes):
                result = self.run_fake(native_payload(attention_case(), **changes))
                self.assertEqual(result["code"], 1)

    def test_unknown_source_cannot_be_converted_to_zero_workspace(self):
        result = self.run_fake(native_payload(attention_case(), workspace_bytes=0))
        self.assertEqual(result["code"], 1)

    def test_known_workspace_must_match_allocator_and_scratch_evidence(self):
        accounting = {"workspace_bytes": 1024, "internal_allocated_buffer_bytes": 768,
            "cpu_cplan_work_size_bytes": 256, "workspace_bound": "conservative_allocated_capacity",
            "workspace_scope": "internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads"}
        payload = native_payload(attention_case(), workspace_bytes=1024,
            workspace_source="ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
            cpu_cplan_work_size_bytes=256, memory_accounting=accounting)
        result = self.run_fake(payload)
        self.assertEqual(result["code"], 0)
        self.assertTrue(result["rows"][0]["eligibility"]["workspace_known"])
        self.assertFalse(result["rows"][0]["eligibility"]["M2_memory_accounting_alignment_verified"])
        for key, wrong in (("workspace_bytes", 1000), ("internal_allocated_buffer_bytes", 700),
                           ("cpu_cplan_work_size_bytes", 0), ("workspace_bound", "unknown"),
                           ("workspace_scope", "scratch_only")):
            with self.subTest(key=key):
                altered = copy.deepcopy(payload)
                altered["memory_accounting"][key] = wrong
                result = self.run_fake(altered)
                self.assertEqual(result["code"], 1)

    def test_every_generated_json_and_csv_uses_lf(self):
        result = self.run_fake(native_payload(attention_case()))
        self.assertEqual(set(result["bytes"]), {"manifest.json", "plan.json", "measurements.json", "summary.csv", str(Path("raw") / "001.json")})
        for name, contents in result["bytes"].items():
            with self.subTest(name=name):
                self.assertNotIn(b"\r\n", contents)
                self.assertTrue(contents.endswith(b"\n"))

    def test_invalid_json_and_native_exit_failure_preserve_evidence(self):
        for stdout in ("[]", "null", "not JSON"):
            with self.subTest(stdout=stdout):
                result = self.run_fake(stdout)
                self.assertEqual(result["code"], 1)
                self.assertEqual(result["raw"]["stdout"], stdout)
        result = self.run_fake("partial", exitcode=7)
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["raw"]["returncode"], 7)
        self.assertEqual(result["raw"]["stderr"], "probe diagnostic")

    def test_timeout_preserves_partial_output(self):
        failure = subprocess.TimeoutExpired(["mock"], 1, output=b"partial stdout", stderr=b"partial stderr")
        result = self.run_fake({}, failure=failure)
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["raw"]["stdout"], "partial stdout")
        self.assertEqual(result["raw"]["stderr"], "partial stderr")
        self.assertTrue(result["raw"]["timed_out"])

    def test_shared_lock_prevents_hardware_and_is_not_removed_by_rejected_run(self):
        with tempfile.TemporaryDirectory(prefix="macro-lock-test-") as temporary:
            root = Path(temporary)
            binary = root / ("ggml-macro-probe.exe" if os.name == "nt" else "ggml-macro-probe")
            binary.write_bytes(b"mock")
            suite = root / "suite.json"
            suite.write_text(json.dumps({"schema_version": "macro_measurement_suite_v1", "cases": [ffn_case()]}), encoding="utf-8")
            lock = root / ".measurement.lock"
            lock.write_text("another_process", encoding="utf-8")
            with (mock.patch.object(runner, "ROOT", root), mock.patch.object(runner.subprocess, "run") as child,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit):
                    runner.main(["--suite", str(suite), "--native-bin", str(root)])
                child.assert_not_called()
            self.assertEqual(lock.read_text(encoding="utf-8"), "another_process")

    def test_nonfinite_timeout_is_rejected_before_output(self):
        for timeout in ("nan", "inf"):
            with self.subTest(timeout=timeout), contextlib.redirect_stderr(io.StringIO()), mock.patch.object(runner.subprocess, "run") as child:
                with self.assertRaises(SystemExit):
                    runner.main(["--timeout-s", timeout, "--dry-run"])
                child.assert_not_called()

    def test_dry_run_does_not_require_native_binary_or_write_measurements(self):
        with tempfile.TemporaryDirectory(prefix="macro-dryrun-test-") as temporary:
            root = Path(temporary)
            suite = root / "suite.json"
            suite.write_text(json.dumps({"schema_version": "macro_measurement_suite_v1", "cases": [ffn_case()]}), encoding="utf-8")
            buffer = io.StringIO()
            with mock.patch.object(runner, "ROOT", root), mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(buffer):
                code = runner.main(["--suite", str(suite), "--dry-run"])
                child.assert_not_called()
            self.assertEqual(code, 0)
            plan = json.loads(buffer.getvalue())
            self.assertEqual(plan["supported_backend"], "CPU")
            self.assertEqual(plan["cases"][0]["visible_history"], 3)
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
