"""The dense probe's build driver, checked without compiling anything.

`profiling/dac_validation/test_build_probe.py` covers the two columns' shared
generator conventions. What is specific to this column is checked here: the
platform it can run on at all, and the fact that a binary an experiment has
already cited is never replaced by a rebuild over it.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling import host
from profiling.depth_validation import build_probe as subject


class DepthDriverTests(unittest.TestCase):
    def test_a_mac_builds_for_metal_and_is_not_asked_for_a_toolkit(self):
        # The Mac answer is a build, not a reason: the pinned fragments' CUDA
        # event vocabulary is answered by native/backend/metal/include, which the
        # project puts first on the include path and uses as CUDA_ROOT's default,
        # so neither --cuda-root nor --cupti-root is needed or passed through.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stream = io.StringIO()
            with mock.patch.object(host, "backend", return_value="metal"), \
                 mock.patch.object(subject, "generator",
                                   return_value=(["-G", "Ninja"], False)), \
                 contextlib.redirect_stdout(stream):
                code = subject.main(["--build-dir", str(root / "build"),
                                     "--llama-root", str(root / "llama"),
                                     "--ggml-build-dir", str(root / "ggml"),
                                     "--print"])
        self.assertEqual(code, 0)
        printed = stream.getvalue()
        self.assertIn("-DPROBE_BACKEND=metal", printed)
        self.assertNotIn("-DCUDA_ROOT", printed)
        self.assertNotIn("-DCUPTI_ROOT", printed)

    def test_a_cuda_build_without_its_toolkit_is_refused_by_name(self):
        # Without this the failure is CMake's, three layers down, and it names
        # neither the argument nor the toolkit.
        with mock.patch.object(host, "backend", return_value="cuda"):
            with self.assertRaises(SystemExit) as raised:
                subject.main(["--build-dir", "b", "--llama-root", "l",
                              "--ggml-build-dir", "g", "--print"])
        self.assertIn("--cuda-root", str(raised.exception))
        self.assertIn("--cupti-root", str(raised.exception))

    def test_an_existing_binary_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            build = Path(folder) / "build"
            binary = subject.binary_path(build, "Release", False)
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"cited by a campaign")
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(subject, "generator", return_value=(["-G", "Ninja"], False)):
                with self.assertRaises(SystemExit) as raised:
                    subject.main(["--build-dir", str(build), "--llama-root", "l",
                                  "--ggml-build-dir", "g", "--cuda-root", "c",
                                  "--cupti-root", "p", "--print"])
            self.assertEqual(binary.read_bytes(), b"cited by a campaign")
        self.assertIn("already exists", str(raised.exception))

    def test_the_print_line_names_the_three_inputs_the_build_is_bound_to(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(host, "is_macos", return_value=False):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    code = subject.main(["--build-dir", str(root / "build"),
                                         "--llama-root", str(root / "llama"),
                                         "--ggml-build-dir", str(root / "ggml"),
                                         "--cuda-root", str(root / "cuda"),
                                         "--cupti-root", str(root / "cupti"),
                                         "--print"])
        self.assertEqual(code, 0)
        printed = stream.getvalue()
        self.assertIn(f"-DLLAMA_ROOT={(root / 'llama').resolve()}", printed)
        self.assertIn(f"-DGGML_BUILD_DIR={(root / 'ggml').resolve()}", printed)
        self.assertIn(f"-DCUPTI_ROOT={(root / 'cupti').resolve()}", printed)
        self.assertNotIn("--target", printed.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
