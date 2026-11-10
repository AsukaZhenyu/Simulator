"""The reproducibility receipt must be derived from this build, not copied.

`report_e2a` needs `native-reproducibility.json` in the campaign folder. The
archived campaign had one assembled by hand, which a second machine cannot
reproduce. These tests pin the three properties that make the derived version
worth having: it refuses a binary whose build manifest names a different file, it
carries the compiler and hash the build actually recorded, and it says null
rather than guessing the fields the build manifest does not establish.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from profiling.dac_validation import native_reproducibility as subject


def build_manifest(binary: Path, sha: str) -> dict:
    return dict(format_version="steady-costs-native-build-1", returncode=0,
                binary=str(binary), binary_sha256=sha,
                compiler=r"C:\MSVC\cl.exe", compiler_version="19.44.35217.0",
                linked_at_utc="2026-11-07T02:35:41.596476+00:00",
                bound_files={"a.h": "0" * 64}, bound_inputs_unchanged=True,
                runtime_loader_resolution_proven=False,
                primary_execution_boundary="unchanged Executor.action",
                hardware_executed=False)


class NativeReproducibility(unittest.TestCase):
    def cohort(self, root: Path, manifest_binary: Path | None = None, sha="a" * 64):
        binary = root / "Release" / "probe.exe"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"probe")
        (root / "Release" / "build-manifest.json").write_text(
            json.dumps(build_manifest(manifest_binary or binary, sha)), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(dict(
            binary=str(binary), runtime_dirs=[str(root / "runtime")],
            context=dict(binary_sha256=sha, runtime_dlls={}))), encoding="utf-8")
        return binary

    def run_subject(self, root: Path) -> int:
        return subject.main(["--out", str(root)])

    def test_writes_the_hash_the_build_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.cohort(root)
            self.assertEqual(self.run_subject(root), 0)
            written = json.loads((root / "native-reproducibility.json").read_text(encoding="utf-8"))
            self.assertEqual(written["native_binary_sha256"], "a" * 64)
            self.assertEqual(written["compiler_version"], "19.44.35217.0")
            self.assertEqual(written["bound_input_files"], 1)
            self.assertFalse(written["runtime_loader_resolution_proven"])

    def test_unestablished_fields_are_null_rather_than_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.cohort(root)
            self.run_subject(root)
            written = json.loads((root / "native-reproducibility.json").read_text(encoding="utf-8"))
            for field in ("configuration", "optimization", "runtime_library", "NDEBUG",
                          "language", "llama_cpp_exact_git_revision", "CUDA_SDK"):
                self.assertIsNone(written[field], field)
            self.assertTrue(written["configuration_note"])

    def test_a_binary_the_build_manifest_does_not_name_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            other = root / "elsewhere" / "probe.exe"
            other.parent.mkdir()
            other.write_bytes(b"other")
            self.cohort(root, manifest_binary=other)
            with self.assertRaises(SystemExit):
                self.run_subject(root)
            self.assertFalse((root / "native-reproducibility.json").exists())


if __name__ == "__main__":
    unittest.main()
