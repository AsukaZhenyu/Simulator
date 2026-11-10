"""The dense column's preparation, on a device that has never run it.

`prepare_portable` exists so a second machine can prepare the same cohort as the
published one. The two things that decide whether it can are checked here: where
the GGML libraries come from (the probe links them out of the pinned tree but
loads them from that tree's `bin/`, which is a different directory) and the
refusal that turns "the probe would load nothing" into an error at preparation
time rather than 36 processes later.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling import host
from profiling import runtime_libraries
from profiling.depth_validation import prepare_portable as subject


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


class RuntimeDirectoryTests(unittest.TestCase):
    def test_the_ggml_libraries_are_derived_from_the_build_the_probe_names(self):
        # The probe's own CMakeCache.txt names its GGML build; the runtime
        # directory is that build's `bin/`, not the probe's own directory.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "ggml"
            (build / "bin").mkdir(parents=True)
            library(build / "bin")
            probe = root / "depth-probe" / "depth-probe"
            probe.parent.mkdir(parents=True)
            probe.write_bytes(b"")
            (probe.parent / "CMakeCache.txt").write_text(
                "GGML_BUILD_DIR:PATH=%s\n" % build, encoding="utf-8")
            self.assertEqual(subject.ggml_runtime_dirs(probe), [build / "bin"])

    def test_a_probe_built_outside_cmake_derives_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder) / "depth-probe"
            probe.write_bytes(b"")
            self.assertEqual(subject.ggml_runtime_dirs(probe), [])

    def test_derived_directories_are_appended_once_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "ggml"
            (build / "bin").mkdir(parents=True)
            library(build / "bin")
            supplied = root / "supplied"
            supplied.mkdir()
            library(supplied)
            probe = root / "depth-probe" / "depth-probe"
            probe.parent.mkdir(parents=True)
            probe.write_bytes(b"")
            (probe.parent / "CMakeCache.txt").write_text(
                "GGML_BUILD_DIR:PATH=%s\n" % build, encoding="utf-8")
            with mock.patch.object(subject, "hardware_identity", return_value={}):
                record = subject.descriptor(probe, [supplied])
            self.assertEqual(record["runtime_dirs"],
                             [str(supplied.resolve()), str((build / "bin").resolve())])
            self.assertEqual(record["derived_runtime_dirs"], [str((build / "bin").resolve())])

    def test_a_runtime_list_holding_nothing_this_host_can_load_is_refused(self):
        # The classic silent failure: preparation succeeds, then every process
        # exits before it measures anything.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = root / "runtime"
            empty.mkdir()
            probe = root / "depth-probe"
            probe.write_bytes(b"")
            with mock.patch.object(subject, "hardware_identity", return_value={}):
                with self.assertRaises(ValueError) as raised:
                    subject.descriptor(probe, [empty])
        self.assertIn(host.shared_library_pattern(), str(raised.exception))

    def test_a_runtime_holding_one_of_the_three_libraries_is_refused(self):
        # A runtime that stopped part way holds `libggml-base.so` alone. It
        # answers "holds a library of this platform" and then the probe exits on
        # an unresolved symbol, one phase into a prepared cohort.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial = root / "runtime"
            partial.mkdir()
            library(partial, "ggml-base")
            probe = root / "depth-probe"
            probe.write_bytes(b"")
            with mock.patch.object(subject, "hardware_identity", return_value={}):
                with self.assertRaises(ValueError) as raised:
                    subject.descriptor(probe, [partial])
        self.assertIn("ggml-cpu", str(raised.exception))
        self.assertIn("ggml-cuda", str(raised.exception))

    def test_a_missing_probe_is_refused_by_path(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "depth-probe"
            with self.assertRaises(ValueError) as raised:
                subject.descriptor(missing, [Path(folder)])
        self.assertIn("depth-probe", str(raised.exception))

    def test_the_protocol_is_carried_over_and_the_machine_fields_are_asked(self):
        # The published protocol constants describe the probe, not the machine;
        # the affinity mask and CPU count do describe the machine.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            probe = root / "depth-probe"
            probe.write_bytes(b"")
            with mock.patch.object(subject, "hardware_identity", return_value={"os": "test"}), \
                 mock.patch.object(host, "process_affinity_mask", return_value=3), \
                 mock.patch.object(host, "main_thread_mask", return_value=1), \
                 mock.patch.object(host, "online_cpus", return_value=2):
                context = subject.descriptor(probe, [runtime])["baseline_context"]
        self.assertEqual(context["threads"], 1)
        self.assertEqual(context["dtype"], "F32")
        self.assertEqual(context["process_affinity_mask"], 3)
        self.assertEqual(context["main_thread_mask"], 1)
        self.assertEqual(context["online_cpus"], 2)


if __name__ == "__main__":
    unittest.main()
