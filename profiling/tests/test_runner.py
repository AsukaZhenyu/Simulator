"""Runner protocol tests using fake probes; no executable or hardware is run."""

from __future__ import annotations

import contextlib
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
import run_measurements as runner


def compute_payload(**changes):
    result = {
        "schema": "ggml_compute_probe/1", "backend": "CPU", "dtype": "f32",
        "activation_dtype": "f32", "m": 32, "k": 32, "n": 1, "seed": 17,
        "threads": 2, "threads_applied": 2, "warmup": 5, "samples": 3,
        "wall_ns": [1000, 3000, 2000], "workspace_bytes": 0,
        "workspace_source": "ggml_graph_plan", "all_outputs_finite": True,
        "reference_check": {"passed": True},
    }
    result.update(changes)
    return result


def transfer_payload(**changes):
    result = {
        "kind": "transfer", "status": "measured", "direction": "H2D",
        "host_memory": "pinned", "size_bytes": 4096, "device_index": 1,
        "warmup": 5, "samples": 3, "wall_ns": [1000, 3000, 2000],
        "device_ns": [900, 2700, 1800], "validation": "all_bytes_exact",
    }
    result.update(changes)
    return result


class RunnerProtocolTests(unittest.TestCase):
    def run_fake(self, payload, *, kind="compute", backend="CPU", failure=None, exitcode=0):
        """Return parsed artifacts after a run with one mocked child process."""
        with tempfile.TemporaryDirectory(prefix="profiling-runner-test-") as temporary:
            root = Path(temporary)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            executable = "ggml-compute-probe" if kind == "compute" else "cuda-transfer-probe"
            executable += ".exe" if os.name == "nt" else ""
            (binary_dir / executable).write_bytes(b"fake-probe-never-executed")
            suite = {
                "schema_version": "measurement_suite_v1",
                "compute_cases": [{"id": "projection", "m": 32, "k": 32, "n": 1, "dtype": "f32"}],
                "transfer_cases": [{"id": "weight-copy", "direction": "H2D", "host_memory": "pinned", "size_bytes": 4096}],
            }
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            output = root / "output"
            arguments = [
                "--suite", str(suite_path), "--native-bin", str(binary_dir),
                "--output", str(output), "--kinds", kind, "--backends", backend,
                "--threads", "2", "--samples", "3", "--warmup", "5", "--seed", "17",
            ]
            if kind == "transfer":
                arguments += ["--device", "1"]
            stdout = payload if isinstance(payload, str) else json.dumps(payload)
            completed = subprocess.CompletedProcess([], exitcode, stdout, "probe diagnostic")
            with (
                mock.patch.object(runner, "ROOT", root),
                mock.patch.object(runner, "environment", return_value={"mocked": True}),
                mock.patch.object(runner, "gpu_snapshot", return_value={"mocked": True}) as snapshot,
                mock.patch.object(runner.subprocess, "run", return_value=completed, side_effect=failure) as native,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = runner.main(arguments)
            self.assertEqual(native.call_count, 1)
            artifacts = {
                "code": code,
                "rows": json.loads((output / "measurements.json").read_text(encoding="utf-8")),
                "raw": json.loads((output / "raw" / "001.json").read_text(encoding="utf-8")),
                "manifest": json.loads((output / "manifest.json").read_text(encoding="utf-8")),
                "gpu_calls": snapshot.call_count,
                "artifact_bytes": {str(path.relative_to(output)): path.read_bytes()
                                   for path in output.rglob("*") if path.is_file()},
            }
            self.assertFalse((root / ".measurement.lock").exists())
            return artifacts

    def test_cpu_only_preserves_raw_samples_and_does_not_require_transfer_probe(self):
        payload = compute_payload()
        result = self.run_fake(payload)
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["gpu_calls"], 0)
        self.assertEqual(result["rows"][0]["status"], "measured")
        self.assertEqual(result["rows"][0]["summary"]["median_ns"], 2000)
        self.assertEqual(json.loads(result["raw"]["stdout"])["wall_ns"], payload["wall_ns"])
        self.assertEqual(result["raw"]["stderr"], "probe diagnostic")
        self.assertEqual(result["manifest"]["completed_cases"], 1)

    def test_saved_json_and_csv_use_lf_even_on_windows(self):
        result = self.run_fake(compute_payload())
        self.assertEqual(set(result["artifact_bytes"]),
                         {"manifest.json", "plan.json", "measurements.json", "summary.csv",
                          str(Path("raw") / "001.json")})
        for name, contents in result["artifact_bytes"].items():
            with self.subTest(artifact=name):
                self.assertNotIn(b"\r\n", contents)
                self.assertTrue(contents.endswith(b"\n"))

    def test_requested_and_applied_threads_remain_distinct_in_signatures(self):
        for backend, applied in (("CPU", 2), ("CUDA0", None)):
            with self.subTest(backend=backend):
                result = self.run_fake(compute_payload(backend=backend, threads_requested=2,
                    threads_applied=applied, workspace_bytes=None,
                    workspace_source="unknown_backend_private_allocations"), backend=backend)
                self.assertEqual(result["code"], 0)
                signature = result["rows"][0]["signature"]
                self.assertEqual(signature["threads_requested"], 2)
                self.assertEqual(signature["threads_applied"], applied)
                self.assertEqual(signature["threads"], applied)

    def test_new_thread_metadata_cannot_claim_wrong_requested_or_applied_counts(self):
        for backend, changes in (("CPU", {"threads_requested": 4, "threads_applied": 2}),
                                 ("CPU", {"threads_requested": 2, "threads_applied": 4}),
                                 ("CPU", {"threads_requested": 2, "threads_applied": None}),
                                 ("CUDA0", {"threads_requested": 4, "threads_applied": None}),
                                 ("CUDA0", {"threads_requested": 2, "threads_applied": 2})):
            with self.subTest(backend=backend, changes=changes):
                result = self.run_fake(compute_payload(backend=backend, workspace_bytes=None,
                    workspace_source="unknown_backend_private_allocations", **changes), backend=backend)
                self.assertEqual(result["code"], 1)
                self.assertEqual(result["rows"][0]["status"], "failed")

    def test_legacy_probe_without_requested_or_applied_metadata_remains_readable(self):
        payload = compute_payload()
        payload.pop("threads_applied")
        result = self.run_fake(payload)
        self.assertEqual(result["code"], 0)
        self.assertNotIn("threads_requested", result["rows"][0]["native"])
        self.assertNotIn("threads_applied", result["rows"][0]["native"])

    def test_cuda_unknown_workspace_survives_runner_and_is_not_exportable(self):
        result = self.run_fake(compute_payload(backend="CUDA0", threads_applied=None, workspace_bytes=None,
                                             workspace_source="unknown_backend_private_allocations"), backend="CUDA0")
        self.assertEqual(result["code"], 0)
        row = result["rows"][0]
        self.assertIsNone(row["native"]["workspace_bytes"])
        self.assertFalse(row["eligibility"]["workspace_known"])
        self.assertFalse(row["eligibility"]["automatic_M1_export"])

    def test_cuda_unknown_allocation_source_cannot_claim_zero_workspace(self):
        result = self.run_fake(compute_payload(backend="CUDA0", threads_applied=None, workspace_bytes=0,
                                             workspace_source="unknown_backend_private_allocations"), backend="CUDA0")
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["rows"][0]["status"], "failed")

    def test_wrong_compute_configuration_cannot_inherit_the_planned_signature(self):
        for field, wrong in (("backend", "CUDA0"), ("dtype", "f16"), ("m", 64),
                             ("k", 64), ("n", 2), ("threads", 4), ("seed", 18),
                             ("warmup", 4), ("kind", "transfer")):
            with self.subTest(field=field):
                result = self.run_fake(compute_payload(**{field: wrong}))
                self.assertEqual(result["code"], 1, f"mismatched {field} was accepted as measured")
                self.assertEqual(result["rows"][0]["status"], "failed")
                self.assertEqual(json.loads(result["raw"]["stdout"])[field], wrong)

    def test_wrong_transfer_configuration_or_device_cannot_be_measured(self):
        for field, wrong in (("direction", "D2H"), ("host_memory", "pageable"),
                             ("size_bytes", 8192), ("device_index", 0)):
            with self.subTest(field=field):
                result = self.run_fake(transfer_payload(**{field: wrong}), kind="transfer")
                self.assertEqual(result["code"], 1, f"mismatched {field} was accepted as measured")
                self.assertEqual(result["rows"][0]["status"], "failed")

    def test_nonobject_json_and_invalid_json_are_saved_as_failed_cases(self):
        for stdout in ("[]", "null", "42", "not valid JSON"):
            with self.subTest(stdout=stdout):
                result = self.run_fake(stdout)
                self.assertEqual(result["code"], 1)
                self.assertEqual(result["raw"]["stdout"], stdout)
                self.assertEqual(result["rows"][0]["status"], "failed")

    def test_native_failure_keeps_stdout_and_stderr(self):
        result = self.run_fake("partial output", exitcode=7)
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["raw"]["returncode"], 7)
        self.assertEqual(result["raw"]["stdout"], "partial output")
        self.assertEqual(result["raw"]["stderr"], "probe diagnostic")

    def test_timeout_preserves_partial_process_output(self):
        failure = subprocess.TimeoutExpired(["fake-probe"], 1, output=b"partial stdout", stderr=b"partial stderr")
        result = self.run_fake({}, failure=failure)
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["raw"].get("stdout"), "partial stdout")
        self.assertEqual(result["raw"].get("stderr"), "partial stderr")
        self.assertIn("error", result["raw"])

    def test_missing_wall_sample_is_not_silently_averaged(self):
        result = self.run_fake(compute_payload(wall_ns=[1000, 2000]))
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["rows"][0]["status"], "failed")
        self.assertEqual(json.loads(result["raw"]["stdout"])["wall_ns"], [1000, 2000])

    def test_optional_device_samples_cannot_have_a_different_sample_count(self):
        result = self.run_fake(transfer_payload(device_ns=[900]), kind="transfer")
        self.assertEqual(result["code"], 1)
        self.assertEqual(result["rows"][0]["status"], "failed")

    def test_nonfinite_timeout_is_rejected_before_hardware_or_output(self):
        with tempfile.TemporaryDirectory(prefix="profiling-invalid-args-") as temporary:
            suite = Path(temporary) / "suite.json"
            suite.write_text(json.dumps({
                "schema_version": "measurement_suite_v1",
                "compute_cases": [{"id": "projection", "m": 32, "k": 32, "n": 1, "dtype": "f32"}],
            }), encoding="utf-8")
            for timeout in ("nan", "inf"):
                with self.subTest(timeout=timeout):
                    with (
                        mock.patch.object(runner.subprocess, "run") as child,
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            runner.main(["--suite", str(suite), "--timeout-s", timeout, "--dry-run"])
                        self.assertEqual(raised.exception.code, 2)
                        child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
