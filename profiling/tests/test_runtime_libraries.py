"""The one question both V-A native columns ask about a runtime directory.

A probe that links ggml loads its libraries by name at launch, so "the directory
holds a library of this platform" is not the check that matters: a build that
stopped part way holds the first one and dies at the first launch, after a
cohort has already been prepared. The union across directories is what matters,
because the campaign hands the probe the runtime, the toolkit's `bin` and
CUPTI's `lib` as three separate `--runtime-dir` arguments.

Every case names its backend explicitly rather than asking this host. Which
libraries a runtime needs is the one thing here that differs between a CUDA box
and a Mac, and a test that could only be written on one of them would leave the
other's answer unchecked -- including the one that used to be wrong: the third
library is `ggml-metal` on a Mac, so demanding `ggml-cuda` there refuses a
complete runtime.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from profiling import host
from profiling import runtime_libraries as subject

NAMES = {"*.dll": "{}.dll", "*.dylib": "lib{}.dylib", "*.so*": "lib{}.so"}


def write(directory: Path, *components: str) -> None:
    for component in components:
        (directory / NAMES[host.shared_library_pattern()].format(component)).write_bytes(b"")


class ComponentTests(unittest.TestCase):
    def test_the_three_components_are_named_per_backend(self):
        self.assertEqual(subject.components("cuda"), ("ggml-base", "ggml-cpu", "ggml-cuda"))
        self.assertEqual(subject.components("metal"), ("ggml-base", "ggml-cpu", "ggml-metal"))
        # A CPU-only host links no accelerator kernels, so asking it for one
        # would refuse a runtime that is complete for the probe it can build.
        self.assertEqual(subject.components(None), ("ggml-base", "ggml-cpu"))

    def test_a_cuda_runtime_does_not_satisfy_a_metal_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            write(Path(folder), "ggml-base", "ggml-cpu", "ggml-cuda")
            self.assertEqual(subject.missing([Path(folder)], backend="metal"), ["ggml-metal"])

    def test_a_complete_directory_answers_nothing_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            write(Path(folder), *subject.components("cuda"))
            self.assertEqual(subject.missing([Path(folder)], backend="cuda"), [])

    def test_the_first_library_alone_is_named_as_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            write(Path(folder), "ggml-base")
            missing = subject.missing([Path(folder)], backend="cuda")
        self.assertIn("ggml-cpu", missing)
        self.assertIn("ggml-cuda", missing)
        self.assertNotIn("ggml-base", missing)

    def test_the_three_may_be_split_across_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runtime").mkdir()
            (root / "toolkit").mkdir()
            write(root / "runtime", "ggml-base", "ggml-cpu")
            write(root / "toolkit", "ggml-cuda")
            self.assertEqual(
                subject.missing([root / "runtime", root / "toolkit"], backend="cuda"), [])

    def test_a_soname_answers_for_its_component(self):
        # A Linux shared build writes `libggml-cuda.so -> libggml-cuda.so.0`, and
        # only one of the three names may survive a copy.
        suffix = {"*.dll": ".dll", "*.dylib": ".dylib", "*.so*": ".so.0"}
        with tempfile.TemporaryDirectory() as folder:
            write(Path(folder), "ggml-base", "ggml-cpu")
            (Path(folder) / ("libggml-cuda" + suffix[host.shared_library_pattern()])
             ).write_bytes(b"")
            self.assertEqual(subject.missing([Path(folder)], backend="cuda"), [])

    def test_another_platforms_libraries_never_answer(self):
        # The glob is the host's own, so a directory holding only the other
        # platform's names is incomplete here even if the list looks right.
        other = ".dll" if host.shared_library_pattern() != "*.dll" else ".dylib"
        with tempfile.TemporaryDirectory() as folder:
            for component in subject.components("cuda"):
                (Path(folder) / ("lib" + component + other)).write_bytes(b"")
            self.assertEqual(len(subject.missing([Path(folder)], backend="cuda")), 3)

    def test_the_refusal_names_the_libraries_and_the_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            write(Path(folder), "ggml-base")
            with self.assertRaises(ValueError) as raised:
                subject.require([Path(folder)], backend="cuda")
        message = str(raised.exception)
        self.assertIn("ggml-cpu", message)
        self.assertIn("ggml-cuda", message)
        self.assertIn(str(Path(folder).resolve()), message)

    def test_the_default_is_this_hosts_backend(self):
        # The callers that matter -- the campaign preflight and the two prepare
        # adapters -- pass no backend, so the default has to be the host's own
        # rather than a fixed CUDA answer.
        self.assertEqual(subject.components(), subject.components(host.backend()))


class HostBackendTests(unittest.TestCase):
    def test_a_named_backend_has_a_library_name(self):
        self.assertEqual(host.BACKEND_LIBRARY["cuda"], "ggml-cuda")
        self.assertEqual(host.BACKEND_LIBRARY["metal"], "ggml-metal")

    def test_the_device_name_follows_the_backend(self):
        # The probe used to spell this in one place as "CUDA0"; a Mac's ggml
        # device is "Metal0" and asking for the other name finds nothing.
        if host.backend() == "cuda":
            self.assertEqual(host.accelerator_device_name(), "CUDA0")
        elif host.backend() == "metal":
            self.assertEqual(host.accelerator_device_name(), "Metal0")
        else:
            with self.assertRaises(host.UnsupportedHost):
                host.accelerator_device_name()

    def test_device_memory_is_shared_only_where_the_host_says_so(self):
        # `unknown` is a legitimate answer on a CUDA host whose architecture does not
        # decide the memory topology (an aarch64 host may carry a discrete card).
        self.assertIn(host.device_memory(), ("separate", "shared", "unknown"))
        if host.is_macos():
            self.assertEqual(host.device_memory(), "shared")

    def test_the_arch_is_one_of_the_two_the_toolchains_name(self):
        self.assertIn(host.arch(), ("x86_64", "aarch64"))


if __name__ == "__main__":
    unittest.main()
