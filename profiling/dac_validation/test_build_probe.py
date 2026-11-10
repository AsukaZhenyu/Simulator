"""The E2A probe's build driver, checked without compiling anything.

The V-B column cannot start until `prepare_portable` finds a binary *and* the
`build-manifest.json` its POST_BUILD step writes beside it. So the driver has to
get three things right on a device it has never seen: the generator's config
convention (Visual Studio writes `Release/`, Ninja writes the build root), the
runtime directory the campaign will later hand the probe, and the refusals that
turn a silently unloadable instrument into an error with a reason.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling import host
from profiling import runtime_libraries
from profiling.dac_validation import build_probe as subject
from profiling.depth_validation import build_probe as depth_build


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


class DriverTests(unittest.TestCase):
    def run_driver(self, *arguments: str) -> tuple[int, str]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = subject.main([*arguments])
        return code, stream.getvalue()

    def test_the_configure_line_names_the_runtime_the_campaign_will_use(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with mock.patch.object(host, "is_macos", return_value=False):
                code, printed = self.run_driver(
                    "--build-dir", str(root / "build"), "--llama-root", str(root / "llama"),
                    "--ggml-build-dir", str(root / "ggml"), "--runtime-dir", str(runtime),
                    "--cuda-root", str(root / "cuda"), "--print")
        self.assertEqual(code, 0)
        self.assertIn(f"-DRUNTIME_DIR={runtime.resolve()}", printed)
        self.assertIn(f"-DCUDA_ROOT={(root / 'cuda').resolve()}", printed)

    def test_a_multi_config_generator_takes_its_config_at_build_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(depth_build, "generator",
                                   return_value=(["-G", "Visual Studio 17 2022"], True)):
                _, printed = self.run_driver(
                    "--build-dir", str(root / "build"), "--llama-root", str(root / "llama"),
                    "--ggml-build-dir", str(root / "ggml"), "--runtime-dir", str(runtime),
                    "--cuda-root", str(root / "cuda"), "--print")
        configure, compile_ = printed.strip().splitlines()
        self.assertNotIn("CMAKE_BUILD_TYPE", configure)
        self.assertTrue(compile_.endswith("--target steady-costs-probe --config Release"),
                        compile_)

    def test_a_single_config_generator_sets_the_build_type_at_configure_time(self):
        # Unset on Ninja means an unoptimised measurement, not a failed build.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(depth_build, "generator", return_value=(["-G", "Ninja"], False)):
                _, printed = self.run_driver(
                    "--build-dir", str(root / "build"), "--llama-root", str(root / "llama"),
                    "--ggml-build-dir", str(root / "ggml"), "--runtime-dir", str(runtime),
                    "--cuda-root", str(root / "cuda"), "--print")
        configure, compile_ = printed.strip().splitlines()
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", configure)
        self.assertNotIn("--config", compile_)

    def test_a_runtime_directory_the_loader_cannot_use_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "notes.txt").write_text("no libraries here", encoding="utf-8")
            with mock.patch.object(host, "is_macos", return_value=False):
                with self.assertRaises(SystemExit) as raised:
                    subject.main(["--build-dir", str(root / "build"),
                                  "--llama-root", str(root / "llama"),
                                  "--ggml-build-dir", str(root / "ggml"),
                                  "--runtime-dir", str(runtime),
                                  "--cuda-root", str(root / "cuda"), "--print"])
        self.assertIn(host.shared_library_pattern(), str(raised.exception))

    def test_a_runtime_holding_one_of_the_three_libraries_is_refused(self):
        # The build would succeed -- `find_library` looks in the ggml build dir,
        # not here -- and the binary would then exit at its first launch.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime, "ggml-base")
            with mock.patch.object(host, "is_macos", return_value=False):
                with self.assertRaises(SystemExit) as raised:
                    subject.main(["--build-dir", str(root / "build"),
                                  "--llama-root", str(root / "llama"),
                                  "--ggml-build-dir", str(root / "ggml"),
                                  "--runtime-dir", str(runtime),
                                  "--cuda-root", str(root / "cuda"), "--print"])
        self.assertIn("ggml-cpu", str(raised.exception))
        self.assertIn("ggml-cuda", str(raised.exception))

    def test_an_existing_binary_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime)
            binary = root / "build" / host.executable_name(subject.STEM)
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"cited by a campaign")
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(depth_build, "generator", return_value=(["-G", "Ninja"], False)):
                with self.assertRaises(SystemExit) as raised:
                    subject.main(["--build-dir", str(root / "build"),
                                  "--llama-root", str(root / "llama"),
                                  "--ggml-build-dir", str(root / "ggml"),
                                  "--runtime-dir", str(runtime),
                                  "--cuda-root", str(root / "cuda"), "--print"])
            self.assertEqual(binary.read_bytes(), b"cited by a campaign")
        self.assertIn("already exists", str(raised.exception))

    def test_a_mac_builds_for_metal_and_is_not_asked_for_a_toolkit(self):
        # A Mac is a build host now: the same CMake project with
        # PROBE_BACKEND=metal answers the pinned fragments' CUDA event vocabulary
        # from profiling/native/backend/metal/include, so no toolkit, no CUDART
        # and no `--cuda-root` are involved. The runtime it names holds
        # ggml-metal in place of ggml-cuda.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            library(runtime, "ggml-base", "ggml-cpu", "ggml-metal")
            stream = io.StringIO()
            with mock.patch.object(host, "backend", return_value="metal"), \
                 mock.patch.object(depth_build, "generator",
                                   return_value=(["-G", "Ninja"], False)), \
                 contextlib.redirect_stdout(stream):
                code = subject.main(["--build-dir", str(root / "build"),
                                     "--llama-root", str(root / "llama"),
                                     "--ggml-build-dir", str(root / "ggml"),
                                     "--runtime-dir", str(runtime), "--print"])
        self.assertEqual(code, 0)
        printed = stream.getvalue()
        self.assertIn("-DPROBE_BACKEND=metal", printed)
        self.assertNotIn("-DCUDA_ROOT", printed)

    def test_a_cuda_build_without_a_toolkit_is_refused_by_name(self):
        with mock.patch.object(host, "backend", return_value="cuda"):
            with self.assertRaises(SystemExit) as raised:
                subject.main(["--build-dir", "b", "--llama-root", "l",
                              "--ggml-build-dir", "g", "--runtime-dir", "r", "--print"])
        self.assertIn("--cuda-root", str(raised.exception))

    def test_the_binary_lands_where_the_campaign_default_looks(self):
        with mock.patch.object(host, "is_windows", return_value=True):
            self.assertEqual(Path("Release") / "steady-costs-probe.exe",
                             depth_build.binary_path(Path("."), "Release", True,
                                                     subject.STEM))
        with mock.patch.object(host, "is_windows", return_value=False):
            self.assertEqual(Path("steady-costs-probe"),
                             depth_build.binary_path(Path("."), "Release", False,
                                                     subject.STEM))


if __name__ == "__main__":
    unittest.main()
