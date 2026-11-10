"""The ggml-build script refuses before it wastes a build.

`tools/build-ggml.sh` is the Linux half of "the probe needs four libraries from a
tree this repository does not ship". Nothing here runs CMake: what is checked is
that the script is valid `sh`, that its refusals fire on the inputs a rented
machine actually produces, and that it keeps the one line that matters -- the
toolkit's `bin` on PATH, without which CMake cannot find `nvcc` at all.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build-ggml.sh"


@unittest.skipIf(sys.platform == "win32", "a POSIX shell script")
class GgmlBuildScriptTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", str(SCRIPT), *arguments],
                              capture_output=True, text=True, encoding="utf-8")

    def test_the_script_is_valid_posix_shell(self):
        done = subprocess.run(["sh", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_requires_the_two_arguments_that_have_no_default(self):
        done = self.run_script()
        self.assertEqual(done.returncode, 2)
        self.assertIn("--llama-root", done.stderr)

    def test_a_directory_that_is_not_a_llama_checkout_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            done = self.run_script("--llama-root", folder, "--build-dir", folder + "/b")
        self.assertEqual(done.returncode, 2)
        self.assertIn("ggml/include/ggml.h", done.stderr)

    def test_a_toolkit_without_nvcc_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder) / "tree"
            (tree / "ggml" / "include").mkdir(parents=True)
            (tree / "ggml" / "include" / "ggml.h").write_text("", encoding="utf-8")
            toolkit = Path(folder) / "cuda"
            toolkit.mkdir()
            done = self.run_script("--llama-root", str(tree), "--build-dir",
                                   str(Path(folder) / "build"),
                                   "--cuda-root", str(toolkit))
        self.assertEqual(done.returncode, 2)
        self.assertIn("bin/nvcc", done.stderr)

    def test_the_toolkit_is_put_on_path_and_named_as_cudacxx(self):
        # The one line that is the difference between a configure and
        # "No CMAKE_CUDA_COMPILER could be found."
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('PATH="$CUDA_ROOT/bin:$PATH"', text)
        self.assertIn('CUDACXX="$CUDA_ROOT/bin/nvcc"', text)
        self.assertIn("export PATH CUDACXX", text)

    def test_all_four_targets_are_named(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--target ggml ggml-base ggml-cpu ggml-cuda", text)


if __name__ == "__main__":
    unittest.main()
