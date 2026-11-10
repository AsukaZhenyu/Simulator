"""The strict-F32 runtime on a POSIX host: same contract, different route.

The published runtime was built by recompiling one patched translation unit and
relinking the objects a Windows build already had. That route needs MSBuild, a
link record and `.lib` files, so on a POSIX host the whole tree is compiled
instead. Two things must not drift: the patch itself (the same pinned bytes and
the same one-line replacement), and the honesty of the manifest, which has to
say that everything except the router is a clean build rather than a preserved
binary.

The build itself takes a quarter of an hour and needs a toolkit, so the tests
here stop it at `run` and check what is written around it. Everything that needs
the pinned tree skips when it is absent, naming the variable that points at a
copy.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling import host
from profiling.native.highlevel_accuracy import build_f32_runtime_linux as subject

PINNED = Path(os.environ.get(
    "LLAMA_B8705_ROOT",
    "C:/Users/89664/Desktop/master degree/LLM-Infer/llama.cpp-implement/"
    "release-b8705/llama.cpp-b8705"))
PINNED_MISSING = ("the pinned llama.cpp b8705 checkout is not here: set "
                  "LLAMA_B8705_ROOT to a llama.cpp-b8705 tree")


class RefusalTests(unittest.TestCase):
    def test_windows_is_sent_to_the_route_that_preserves_the_build(self):
        with mock.patch.object(host, "is_windows", return_value=True):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError) as raised:
                    subject.build(Path(folder), Path(folder) / "out", None)
        self.assertIn("POSIX only", str(raised.exception))
        self.assertIn("build_f32_runtime", str(raised.exception))

    def test_a_mac_is_told_which_kernel_would_have_to_change(self):
        # A Mac is POSIX, so the Windows refusal does not stop it: without this it would
        # configure a CUDA build and fail inside CMake with a toolkit error, which reads
        # as "install CUDA".  It has no CUDA tree to patch, and the exactness it would
        # buy needs a different kernel there.
        with mock.patch.object(host, "is_windows", return_value=False), \
             mock.patch.object(host, "is_macos", return_value=True):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError) as raised:
                    subject.build(Path(folder), Path(folder) / "out", None)
        message = str(raised.exception)
        self.assertIn("ggml-cuda.cu", message)
        self.assertIn("kernel_mul_mm_f32_f32", message)

    def test_a_tree_without_the_router_is_refused_before_output_exists(self):
        with mock.patch.object(host, "is_windows", return_value=False), \
             tempfile.TemporaryDirectory() as folder:
            tree = Path(folder) / "tree"
            tree.mkdir()
            out = Path(folder) / "out"
            with self.assertRaises(ValueError) as raised:
                subject.build(tree, out, None)
            self.assertIn("ggml-cuda.cu", str(raised.exception))
            self.assertFalse(out.exists())

    def test_an_existing_output_is_not_built_over(self):
        with mock.patch.object(host, "is_windows", return_value=False), \
             tempfile.TemporaryDirectory() as folder:
            tree = Path(folder) / "tree"
            (tree / "src/ggml-cuda").mkdir(parents=True)
            out = Path(folder) / "out"
            out.mkdir()
            (out / "already").write_text("x", encoding="utf-8")
            with mock.patch.object(subject, "patch_router", return_value=b"patched"):
                with self.assertRaises(ValueError) as raised:
                    subject.build(tree, out, None)
            self.assertIn("NEW directory", str(raised.exception))


class CopyTests(unittest.TestCase):
    def test_the_private_copy_leaves_outputs_and_git_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "build").mkdir()
            (root / "build-old").mkdir()
            (root / "cmake").mkdir()
            (root / "cmake" / "build-info.cmake").write_text("", encoding="utf-8")
            (root / "models").mkdir()
            (root / "models" / "ggml-vocab-gpt-2.gguf").write_bytes(b"")
            names = ["build", "build-old", "cmake", ".git", "models", "src"]
            skipped = subject.ignore(str(root), names)
        # `build-info.cmake` lives in `cmake/` and the tree's CMakeLists includes
        # it: a name-based "build*" filter that also drops the directory holding
        # it makes the copy fail at configure as if the checkout were broken.
        self.assertEqual(sorted(skipped), sorted(["build", "build-old", ".git", "models"]))

    def test_the_architecture_sources_under_src_are_not_taken_for_fixtures(self):
        # A first version skipped every directory named `models` at any depth.
        # `src/models/afmoe.cpp` then vanished from the copy and CMake stopped at
        # generate with "No SOURCES given to target: llama" -- a port that reads
        # as a broken checkout rather than an over-eager filter.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src" / "models").mkdir(parents=True)
            (root / "src" / "models" / "afmoe.cpp").write_text("", encoding="utf-8")
            nested = subject.ignore(str(root / "src"), ["models"])
        self.assertEqual(nested, [])

    def test_a_models_directory_holding_weights_is_a_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "models" / "7B").mkdir()
            (root / "models" / "7B" / "ggml-model-q4_0.gguf").write_bytes(b"")
            skipped = subject.ignore(str(root), ["models"])
        self.assertEqual(skipped, ["models"])


@unittest.skipUnless(PINNED.is_dir(), PINNED_MISSING)
class ManifestTests(unittest.TestCase):
    """What the receipt says about a build, with the compile step stubbed out."""

    def run_build(self, folder: str) -> tuple[dict, Path, Path]:
        root = Path(folder)
        out = root / "runtime"
        library_names = ["libllama.so", "libggml-base.so", "libggml-cpu.so", "libggml-cuda.so"]

        def fake_run(command, output, name, environment):
            # The real step writes the shared libraries; this one writes files
            # with those names so everything after the compile can be checked.
            if name == "cmake-build.json":
                binaries = output / "build" / "bin"
                binaries.mkdir(parents=True, exist_ok=True)
                for library in library_names:
                    (binaries / library).write_bytes(b"\x7fELF")
            subject.write(output / name, dict(returncode=0, command=command))
            return dict(returncode=0)

        with mock.patch.object(host, "is_windows", return_value=False), \
             mock.patch.object(subject, "run", side_effect=fake_run), \
             mock.patch.object(subject.linux_build, "build_environment",
                               return_value=dict(os.environ)), \
             mock.patch.object(subject.linux_build, "resolved_architectures",
                               return_value=(["-DCMAKE_CUDA_ARCHITECTURES=89"],
                                             "named by the caller")), \
             mock.patch.object(subject, "compiler_identity", return_value="g++ 13.3.0"):
            receipt = subject.build(PINNED, out, Path("/usr/local/cuda-12.4"), jobs=4)
        return receipt, out, out / "build" / "bin"

    def test_the_receipt_says_what_this_route_can_and_cannot_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt, out, binaries = self.run_build(folder)
            manifest = json.loads((binaries / "strict-f32-runtime.json")
                                  .read_text(encoding="utf-8"))
            patch = (out / "strict-f32-router.patch").read_text(encoding="utf-8")
            tree_router = (out / "source" / "ggml" / subject.ROUTER).read_bytes()
            original_router = (PINNED / "ggml" / subject.ROUTER).read_bytes()
        self.assertEqual(manifest["format_version"], "ggml-strict-f32-runtime-posix-v1")
        # The one claim that separates this route from the Windows one.
        self.assertIn("complete CMake build", manifest["build_scope"])
        self.assertIn("rather than preserved", manifest["build_scope"])
        self.assertIn("GGML_PREC_F32", manifest["precision_scope"])
        self.assertEqual(manifest["original_router_sha256"], subject.PINNED_ROUTER_SHA256)
        self.assertEqual(manifest["patch_hunks"], 1)
        self.assertEqual(manifest["cuda_architecture_source"], "named by the caller")
        # The private copy carries the patched router; the pinned tree does not.
        self.assertNotEqual(tree_router, original_router)
        self.assertIn("GGML_PREC_F32", patch)
        self.assertIn("+", patch)
        self.assertEqual(receipt["runtime_dir"], str(binaries))
        self.assertEqual(sorted(receipt["libraries"]),
                         sorted(["libggml-base.so", "libggml-cpu.so",
                                 "libggml-cuda.so", "libllama.so"]))

    def test_the_pinned_tree_is_not_modified(self):
        router = PINNED / "ggml" / subject.ROUTER
        before = subject.digest(router)
        with tempfile.TemporaryDirectory() as folder:
            self.run_build(folder)
        self.assertEqual(before, subject.digest(router))


if __name__ == "__main__":
    unittest.main()
