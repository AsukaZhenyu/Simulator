"""The E2A column's preparation, on a machine that has never run it.

The adapter exists so a second device can prepare the fixed E2A matrix from
explicit binaries. What it has to get right is small and load-bearing: a binary
without the build manifest the run re-checks is not a source-bound instrument,
and a runtime list that holds no library of this platform is a cohort whose
every process exits before it measures anything.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling import host
from profiling import runtime_libraries
from profiling.dac_validation import prepare_portable as subject


# The probe loads the accelerator this host builds for: `ggml-cuda` on a CUDA
# box, `ggml-metal` on a Mac, and no third library on a CPU-only host. A
# fixture that spelled the CUDA name would build a directory this host's
# probe does not link, and would pass for the wrong reason.
COMPONENTS = runtime_libraries.components()


def library(directory: Path, *components: str) -> list[Path]:
    """The libraries the probe loads, written with this host's own names."""
    written = []
    for component in components or COMPONENTS:
        name = {"*.dll": f"{component}.dll", "*.dylib": f"lib{component}.dylib",
                "*.so*": f"lib{component}.so"}[host.shared_library_pattern()]
        path = directory / name
        path.write_bytes(b"")
        written.append(path)
    return written


class PrepareTests(unittest.TestCase):
    def run_main(self, *arguments: str) -> None:
        with mock.patch.object(sys, "argv", ["prepare_portable", *arguments]), \
             mock.patch("builtins.print"):
            subject.main()

    def test_a_binary_without_its_build_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with self.assertRaises(ValueError) as raised:
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(runtime))
        self.assertIn("build-manifest.json", str(raised.exception))

    def test_a_runtime_list_holding_nothing_this_host_can_load_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            (root / "build-manifest.json").write_text("{}", encoding="utf-8")
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "notes.txt").write_text("no libraries", encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(runtime))
        self.assertIn(host.shared_library_pattern(), str(raised.exception))

    def test_a_runtime_holding_one_of_the_three_libraries_is_refused(self):
        # A strict-F32 build that stopped part way leaves `libggml-base.so` and
        # nothing else. It passes "holds a library of this platform", the cohort
        # is prepared, and then every process exits on an unresolved symbol.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            (root / "build-manifest.json").write_text("{}", encoding="utf-8")
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime, "ggml-base")
            with self.assertRaises(ValueError) as raised:
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(runtime))
        self.assertIn("ggml-cpu", str(raised.exception))
        self.assertIn("ggml-cuda", str(raised.exception))

    def test_the_three_libraries_may_be_split_across_directories(self):
        # The campaign passes the runtime, the toolkit's bin and CUPTI's lib as
        # three separate directories: only their union has to be complete.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            (root / "build-manifest.json").write_text("{}", encoding="utf-8")
            runtime, toolkit = root / "runtime", root / "toolkit"
            runtime.mkdir()
            toolkit.mkdir()
            library(runtime, "ggml-base", "ggml-cpu")
            library(toolkit, "ggml-cuda")
            with mock.patch.object(subject.e2a, "prepare", return_value={}) as prepare, \
                 mock.patch.object(subject.e2a, "write", side_effect=lambda path, value: None), \
                 mock.patch.object(subject.e2a, "receipt", side_effect=lambda path: str(path)):
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(runtime), "--runtime-dir", str(toolkit))
            prepare.assert_called_once_with(root / "out")

    def test_a_missing_binary_is_named(self):
        # "explicit source-bound native build required" was the whole message
        # before; on a rented machine the useful half is which path was typed.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with self.assertRaises(ValueError) as raised:
                self.run_main("--out", str(root / "out"),
                              "--binary", str(root / "steady-costs-probe"),
                              "--runtime-dir", str(runtime))
        self.assertIn("steady-costs-probe", str(raised.exception))

    def test_a_missing_runtime_directory_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            (root / "build-manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(root / "absent"))
        self.assertIn("absent", str(raised.exception))

    def test_a_complete_pair_reaches_the_historical_prepare(self):
        # The adapter's whole job is to hand `e2a.prepare` a descriptor; it must
        # not read a prior profile or frozen result on the way.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "steady-costs-probe"
            binary.write_bytes(b"")
            (root / "build-manifest.json").write_text("{}", encoding="utf-8")
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with mock.patch.object(subject.e2a, "prepare", return_value={}) as prepare, \
                 mock.patch.object(subject.e2a, "write", side_effect=lambda path, value: None), \
                 mock.patch.object(subject.e2a, "receipt", side_effect=lambda path: str(path)):
                self.run_main("--out", str(root / "out"), "--binary", str(binary),
                              "--runtime-dir", str(runtime))
            prepare.assert_called_once_with(root / "out")


if __name__ == "__main__":
    unittest.main()
