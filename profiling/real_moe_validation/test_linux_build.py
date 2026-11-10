"""The model column's POSIX build path, checked without a compiler or a GPU.

The Windows chain rebuilds one translation unit at a time using MSVC's `.tlog`
command files; nothing about that survives a change of host. What does survive is
the set of anchor-checked source transforms, and this module's job is to reuse
them and replace only the plumbing. So the tests fall into three groups:

  * the transforms still find their anchors in the frozen sources, and refuse
    when a source has drifted -- an anchor that silently misses would produce a
    probe that runs and measures nothing;
  * the derived sources carry none of the constructs the POSIX host cannot
    compile: no `<windows.h>`, no `<bcrypt.h>`, no `__declspec(dllexport)`;
  * the build receipts are in the schema `runtime_window_campaign.prepare`
    reads, because that is the only thing standing between a successful build
    and a campaign that starts on the wrong files.

The tests that read the pinned llama.cpp checkout skip when it is absent, so the
suite still runs on a device that only has the one model it is measuring.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from profiling import host
from profiling.real_moe_validation import linux_build as subject
from profiling.real_moe_validation.tap_v4 import sha

PINNED = Path(os.environ.get(
    "LLAMA_B8705_ROOT",
    "C:/Users/89664/Desktop/master degree/LLM-Infer/llama.cpp-implement/"
    "release-b8705/llama.cpp-b8705"))
# The default is where the tree sits on the machine that produced the published
# build; it is a convenience, not a claim. Every other host -- a Linux box, a
# container, a second Windows machine -- names its copy with the environment
# variable, which is why the skip below has to say so rather than only that the
# tree is missing.
PINNED_MISSING = ("the pinned llama.cpp b8705 checkout is not here: set "
                  "LLAMA_B8705_ROOT to a llama.cpp-b8705 tree")

FROZEN_PROBE = subject.NATIVE / "real_moe_tap_probe.cpp"
FROZEN_POD = subject.NATIVE / "route_tap_pod.hpp"
FROZEN_API = subject.NATIVE / "route_tap_api.h"
GPU_API = subject.NATIVE_V5 / "gpu_tap_api.h"


class TransformTests(unittest.TestCase):
    """`replace_once` is the only place an anchor can be missed."""

    def test_an_anchor_that_is_absent_is_a_refusal_not_a_no_op(self):
        with self.assertRaises(ValueError) as raised:
            subject.replace_once("nothing here", "#include <windows.h>", "x", "the shim")
        self.assertIn("found 0", str(raised.exception))

    def test_an_anchor_that_appears_twice_is_a_refusal_not_a_guess(self):
        # Two occurrences mean a source this transform was not written for; the
        # first would be edited and the second left Windows-only.
        with self.assertRaises(ValueError) as raised:
            subject.replace_once("a b a b", "a", "x", "twice")
        self.assertIn("found 2", str(raised.exception))

    def test_the_probe_include_becomes_the_shim_and_nothing_else(self):
        text = '#include <windows.h>\nint main(){}\n'
        self.assertEqual(subject.portable_probe(text),
                         '#include "platform.hpp"\nint main(){}\n')

    def test_the_clock_the_receipt_names_is_the_one_the_host_ran_on(self):
        # `windows_qpc` in a Linux row would be a false provenance line, and the
        # macro resolves to exactly that string on Windows.
        text = '{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",f()}}}'
        self.assertEqual(subject.portable_clock(text),
                         '{"clock",json{{"clock_id",PORTABLE_CLOCK_ID},{"frequency_hz",f()}}}')
        platform = (subject.PORTABLE / "platform.hpp").read_text(encoding="utf-8")
        self.assertIn('#define PORTABLE_CLOCK_ID "windows_qpc"', platform)
        self.assertIn('#define PORTABLE_CLOCK_ID "posix_clock_monotonic"', platform)


class FrozenSourceTests(unittest.TestCase):
    def test_the_frozen_probe_survives_derivation_byte_for_byte(self):
        # The published campaign's `native_source_sha256` names these bytes; the
        # platform edits belong in the derived copy, never in the evidence.
        before = sha(FROZEN_PROBE)
        derived = subject.native_source()
        self.assertEqual(sha(FROZEN_PROBE), before)
        self.assertNotEqual(derived, FROZEN_PROBE.read_text(encoding="utf-8"))

    def test_the_derived_probe_has_no_windows_only_construct(self):
        text = subject.native_source()
        self.assertNotIn("#include <windows.h>", text)
        self.assertEqual(text.count('#include "platform.hpp"'), 1)
        # The v5 and runtime-window transforms really ran: both hops are present.
        self.assertIn('#include "runtime_windows.hpp"', text)
        self.assertIn("runtime_windows->mark(", text)
        self.assertEqual(text.count("__declspec"), 0)
        self.assertEqual(text.count("windows_qpc"), 0)
        self.assertIn("PORTABLE_CLOCK_ID", text)

    def test_the_derived_pod_hashes_with_the_in_tree_sha256(self):
        text = subject.support_headers()["route_tap_pod.hpp"]
        self.assertNotIn("<bcrypt.h>", text)
        self.assertIn('#include "sha256.hpp"', text)
        self.assertIn("portable_sha256::digest_hex", text)
        for windows_only in ("BCryptOpenAlgorithmProvider", "BCRYPT_ALG_HANDLE",
                             "BCryptFinishHash"):
            self.assertNotIn(windows_only, text)

    def test_both_tap_headers_keep_one_export_statement_each(self):
        headers = subject.support_headers()
        for name in ("route_tap_api.h", "gpu_tap_api.h"):
            text = headers[name]
            self.assertNotIn("__declspec(dllexport)", text, name)
            self.assertNotIn("__declspec(dllimport)", text, name)
            self.assertEqual(text.count('__attribute__((visibility("default")))'), 1, name)

    def test_a_tap_header_without_its_export_guard_is_refused(self):
        with self.assertRaises(ValueError):
            subject.portable_exports("#pragma once\nvoid f(void);\n")

    def test_every_portability_edit_names_a_file_the_build_writes(self):
        targets = {edit["file"] for edit in subject.portability_edits()}
        self.assertLessEqual(targets, {"source/real_moe_tap_probe.cpp",
                                       "source/route_tap_pod.hpp",
                                       "source/route_tap_api.h",
                                       "source/gpu_tap_api.h",
                                       "ggml/src/ggml-backend.cpp",
                                       "ggml/src/ggml-cpu/ggml-cpu.c",
                                       "common/debug.cpp"})
        # The Windows-only include in ggml-cpu.c is inside its own `_WIN32` block,
        # so that file is edited for the export only -- never for the shim.
        cpu_edits = [e for e in subject.portability_edits()
                     if e["file"] == "ggml/src/ggml-cpu/ggml-cpu.c"]
        self.assertEqual([e["anchor"] for e in cpu_edits], [subject.EXPORT])


@unittest.skipUnless(PINNED.is_dir(), PINNED_MISSING)
class PinnedTreeTests(unittest.TestCase):
    """The transforms against the tree the published table was measured on."""

    def test_the_five_transformed_sources_lose_their_windows_constructs(self):
        sources = subject.patched_sources(PINNED)
        self.assertEqual(set(sources), {"ggml/src/ggml-backend.cpp",
                                        "ggml/src/ggml-cpu/ggml-cpu.c",
                                        "ggml/src/ggml-cuda/ggml-cuda.cu",
                                        "src/llama-context.h",
                                        "common/debug.cpp"})
        for name, text in sources.items():
            self.assertEqual(text.count("__declspec(dllexport)"), 0, name)

    def test_the_moe_diagnostic_gets_the_include_msvc_lends_it(self):
        # `common/debug.cpp` is the repository's own `ffn_moe_topk` diagnostic and
        # calls `strstr`/`memcpy`; libstdc++ does not put `<cstring>` in scope.
        text = subject.patched_sources(PINNED)["common/debug.cpp"]
        self.assertIn("#include <cstring>", text)
        self.assertIn("#include <string>", text)
        self.assertEqual(text.count("#include <cstring>"), 1)
        # Nothing else about the file changes: it is not an observer.
        original = (PINNED / "common/debug.cpp").read_text(encoding="utf-8")
        if "#include <cstring>" not in original:
            self.assertEqual(text.replace("#include <cstring>\n#include <string>",
                                          "#include <string>"), original)

    def test_a_source_that_already_includes_cstring_is_left_alone(self):
        text = "#include <cstring>\n#include <string>\nint f();\n"
        self.assertEqual(subject.portable_cstring(text), text)

    def test_the_cpu_translation_unit_keeps_its_own_windows_block(self):
        # It is C, and its own `<windows.h>` sits inside `#if defined(_WIN32)`;
        # rewriting it would put a C++ header into a C file for no platform gain.
        text = subject.patched_sources(PINNED)["ggml/src/ggml-cpu/ggml-cpu.c"]
        self.assertIn("#include <windows.h>", text)
        self.assertNotIn('#include "platform.hpp"', text)
        # Both taps are present: the consumer tap this repository ships and the
        # runtime window layered over it.
        self.assertEqual(text.count("simulator_moe_tap_set"), 1)
        self.assertEqual(text.count("simulator_runtime_cpu_window_set"), 1)

    def test_the_backend_and_cuda_hooks_are_the_ones_the_probe_binds(self):
        sources = subject.patched_sources(PINNED)
        self.assertEqual(sources["ggml/src/ggml-backend.cpp"].count(
            "simulator_runtime_window_set"), 1)
        self.assertEqual(sources["ggml/src/ggml-cuda/ggml-cuda.cu"].count(
            "simulator_gpu_tap_set_v5"), 1)
        self.assertIn("simulator_tap_graph_result", sources["src/llama-context.h"])

    def test_a_tree_that_has_drifted_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder)
            for name in ("ggml/src/ggml-backend.cpp", "ggml/src/ggml-cpu/ggml-cpu.c",
                         "ggml/src/ggml-cuda/ggml-cuda.cu", "src/llama-context.h",
                         "common/debug.cpp"):
                path = tree / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("a source the anchors were not written for\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                subject.patched_sources(tree)


class SymbolTests(unittest.TestCase):
    """The build checks the symbols the probe really asks for, and no others."""

    def test_the_checked_symbols_are_the_ones_the_probe_binds(self):
        sources = [subject.native_source()] + list(subject.support_headers().values())
        bound = set()
        for text in sources:
            bound.update(re.findall(r'GetProcAddress\([^,()]+,\s*"([A-Za-z0-9_]+)"\)', text))
        checked = {symbol for symbols in subject.REQUIRED_SYMBOLS.values()
                   for symbol in symbols}
        # Both directions: a symbol the build does not check would fail at run
        # time with a null observer, and one it checks that nothing binds would
        # refuse a build that is fine.
        self.assertEqual(checked, bound)

    def test_each_library_is_named_with_the_suffix_the_host_links(self):
        self.assertTrue(all(name.endswith(".so") for name in subject.REQUIRED_SYMBOLS))


class CopyTests(unittest.TestCase):
    """What the private copy keeps, because the tree's own configure needs it."""

    def test_a_directory_named_like_a_build_is_left_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("build", "build-cuda", "build-xcframework"):
                (root / name).mkdir()
            names = sorted(p.name for p in root.iterdir())
            self.assertEqual(subject.copy_ignore(folder, names), names)

    def test_a_weights_directory_is_skipped_and_a_source_one_is_not(self):
        # `models/` holds the architecture sources in the pinned tree and GGUFs
        # in a checkout that downloads them; only the second is not a build
        # input, and dropping the first fails configure with "Cannot find
        # source file: models/afmoe.cpp".
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "models" / "afmoe.cpp").write_text("// model\n", encoding="utf-8")
            self.assertEqual(subject.copy_ignore(folder, ["models"]), [])
            (root / "models" / "afmoe.cpp").unlink()
            (root / "models" / "OLMoE.gguf").write_bytes(b"")
            self.assertEqual(subject.copy_ignore(folder, ["models"]), ["models"])
            # A directory this rule says nothing about is copied.
            (root / "benches").mkdir()
            self.assertEqual(subject.copy_ignore(folder, ["benches"]), [])

    def test_a_file_whose_name_starts_with_build_is_kept(self):
        # `cmake/build-info.cmake` is included by the tree's `CMakeLists.txt`;
        # `build-xcframework.sh` is a script. A name pattern that ignored
        # `build-*` dropped both, and the build then failed at configure.
        files = ["build-info.cmake", "build-xcframework.sh", "common.cmake"]
        self.assertEqual(subject.copy_ignore("/somewhere", files), [])

    def test_the_digest_that_lives_under_a_build_prefix_is_really_copied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cmake").mkdir()
            (root / "cmake" / "build-info.cmake").write_text("# info\n", encoding="utf-8")
            (root / "build").mkdir()
            (root / "build" / "left-over.txt").write_text("x\n", encoding="utf-8")
            import shutil
            shutil.copytree(root, root / "copy", ignore=subject.copy_ignore)
            self.assertTrue((root / "copy" / "cmake" / "build-info.cmake").is_file())
            self.assertFalse((root / "copy" / "build").exists())


class EnvironmentTests(unittest.TestCase):
    """A toolkit passed by path is not on `PATH`, and CMake looks there."""

    def test_the_named_toolkit_is_put_where_the_cuda_compiler_is_looked_for(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "bin").mkdir()
            (Path(folder) / "bin" / "nvcc").write_bytes(b"")
            environment = subject.build_environment(Path(folder), {"PATH": "/usr/bin"})
            self.assertEqual(environment["PATH"].split(os.pathsep)[0],
                             str(Path(folder).resolve() / "bin"))
            self.assertEqual(environment["CUDACXX"],
                             str(Path(folder).resolve() / "bin" / "nvcc"))
            self.assertTrue(environment["PATH"].endswith("/usr/bin"))

    def test_no_toolkit_leaves_the_environment_as_it_was(self):
        # DGX Spark images carry their own toolkit on `PATH`; inventing a
        # `CUDACXX` there would override the one the image was built with.
        base = {"PATH": "/opt/bin"}
        self.assertEqual(subject.build_environment(None, base), base)

    def test_a_toolkit_without_nvcc_does_not_claim_a_compiler(self):
        with tempfile.TemporaryDirectory() as folder:
            environment = subject.build_environment(Path(folder), {"PATH": "/usr/bin"})
            self.assertNotIn("CUDACXX", environment)


class CommandTests(unittest.TestCase):
    def out(self) -> Path:
        return Path("build") / "runtime-linux"

    def test_configure_asks_for_an_optimised_cuda_build(self):
        line = subject.configure_command(self.out(), Path("/usr/local/cuda"), [])
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", line)
        self.assertIn("-DGGML_CUDA=ON", line)
        self.assertIn("-DBUILD_SHARED_LIBS=ON", line)
        self.assertIn("-DCUDAToolkit_ROOT=%s" % Path("/usr/local/cuda").resolve(), line)

    def test_a_toolkit_is_optional_so_DGX_Spark_keeps_its_own(self):
        line = subject.configure_command(self.out(), None, [])
        self.assertFalse(any(argument.startswith("-DCUDAToolkit_ROOT") for argument in line))

    def test_extra_arguments_are_passed_through(self):
        line = subject.configure_command(self.out(), None, ["-DCMAKE_CUDA_ARCHITECTURES=121"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121", line)

    def test_the_probe_can_find_the_toolkit_it_was_built_against(self):
        # `libggml-cuda.so` needs `libcudart.so.12`; a toolkit outside the
        # loader's default path leaves the probe unable to start after a
        # successful build.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "lib64").mkdir()
            (root / "targets" / "aarch64-linux" / "lib").mkdir(parents=True)
            line = subject.probe_command(self.out(), "g++", "0" * 64, root)
            self.assertIn("-Wl,-rpath," + str((root / "lib64").resolve()), line)
            self.assertIn("-Wl,-rpath,"
                          + str((root / "targets/aarch64-linux/lib").resolve()), line)
            # A directory that does not exist is not put on the link line.
            self.assertNotIn("-Wl,-rpath," + str((root / "lib").resolve()), line)

    def test_no_toolkit_adds_no_rpath_of_its_own(self):
        line = subject.probe_command(self.out(), "g++", "0" * 64, None)
        self.assertEqual(len([x for x in line if x.startswith("-Wl,-rpath,")]), 1)

    def test_the_probe_links_what_the_shim_needs(self):
        line = subject.probe_command(self.out(), "g++", "0" * 64)
        # dlopen/dlsym for GetModuleHandle/GetProcAddress, and rpath so the
        # process finds the libraries it was built against.
        self.assertIn("-ldl", line)
        self.assertIn("-lpthread", line)
        self.assertTrue(any(argument.startswith("-Wl,-rpath,") for argument in line))
        self.assertIn("-std=c++17", line)

    def test_a_named_architecture_is_never_overridden_by_detection(self):
        # An operator asking for `89;121` wants one binary for two machines.
        named = ["-DCMAKE_CUDA_ARCHITECTURES=89;121"]
        with mock.patch.object(subject, "compute_capability") as asked:
            arguments, source = subject.resolved_architectures(named)
        self.assertEqual(arguments, named)
        self.assertIn("named by the caller", source)
        asked.assert_not_called()

    def test_the_device_answers_the_architecture_question_when_none_is_named(self):
        completed = mock.Mock(returncode=0, stdout="12.1\n", stderr="")
        with mock.patch.object(subject.subprocess, "run", return_value=completed):
            arguments, source = subject.resolved_architectures([])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121", arguments)
        self.assertIn("nvidia-smi", source)

    def test_a_host_that_cannot_answer_keeps_the_toolkits_default(self):
        # No nvidia-smi, no GPU, or two GPUs of different capabilities: a guess
        # here would produce a binary for one device and name it as both.
        answers = [(dict(side_effect=OSError("no nvidia-smi")), "absent"),
                   (dict(return_value=mock.Mock(returncode=1, stdout="", stderr="no devices")),
                    "no device"),
                   (dict(return_value=mock.Mock(returncode=0, stdout="8.9\n12.1\n", stderr="")),
                    "two capabilities")]
        for answer, why in answers:
            with self.subTest(why), mock.patch.object(subject.subprocess, "run", **answer):
                arguments, source = subject.resolved_architectures(["-DOTHER=1"])
            self.assertEqual(arguments, ["-DOTHER=1"])
            self.assertIn("cmake default", source)

    def test_the_architecture_the_cuda_objects_target_is_recorded(self):
        # An sm_89 binary on an sm_121 device launches no kernel and reports
        # "no kernel image is available"; the receipt has to say which one it is.
        self.assertEqual(
            subject.cuda_architectures(["-DCMAKE_CUDA_ARCHITECTURES=121"]), "121")
        self.assertEqual(
            subject.cuda_architectures(["-DOTHER=1", "-DCMAKE_CUDA_ARCHITECTURES=89;121"]),
            "89;121")
        default = subject.cuda_architectures(["-DOTHER=1"])
        self.assertIn("cmake default", default)

    def test_a_missing_nvcc_is_recorded_as_missing_not_as_a_version(self):
        # Windows resolves the executable against the *parent's* `PATH` even when
        # the child is given another environment, so this branch is exercised
        # with the call itself faked rather than with a host-shaped environment.
        missing = mock.Mock(returncode=1, stdout="", stderr="nvcc: not found")
        with mock.patch.object(subject.subprocess, "run", return_value=missing):
            self.assertIn("not found", subject.nvcc_version({"PATH": "/nonexistent"}))

    @unittest.skipIf(host.is_windows(), "a shell script is not an executable here")
    def test_nvcc_is_asked_through_the_toolkit_the_build_named(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "nvcc"
            fake.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n"
                            "echo 'Build cuda_12.4.r12.4/compiler.34899100_0'\n",
                            encoding="utf-8", newline="\n")
            fake.chmod(0o755)
            described = subject.nvcc_version({"CUDACXX": str(fake), "PATH": "/nonexistent"})
        self.assertIn("12.4", described)
        self.assertIn("compiler", described)


class PrepareTests(unittest.TestCase):
    def test_an_existing_build_root_is_never_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "build"
            out.mkdir()
            with self.assertRaises(ValueError) as raised:
                subject.prepare(out, PINNED if PINNED.is_dir() else Path(folder))
            self.assertIn("new private build root required", str(raised.exception))

    def test_a_tree_without_its_cuda_source_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as raised:
                subject.prepare(Path(folder) / "build", Path(folder) / "tree")
            self.assertIn("ggml-cuda.cu", str(raised.exception))

    def test_prepare_without_a_tree_says_which_flag_is_missing(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "is_windows", return_value=False):
            with self.assertRaises(SystemExit) as raised:
                subject.main(["prepare", "--out", "nowhere"])
        self.assertIn("--llama-root", str(raised.exception))

    def test_macos_is_refused_before_a_cuda_toolkit_is_looked_for(self):
        with mock.patch.object(host, "is_macos", return_value=True):
            with self.assertRaises(SystemExit) as raised:
                subject.main(["build", "--out", "nowhere"])
        self.assertIn("CUDA", str(raised.exception))

    def test_the_windows_chain_keeps_its_own_build_path(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "is_windows", return_value=True):
            with self.assertRaises(SystemExit) as raised:
                subject.main(["prepare", "--out", "nowhere", "--llama-root", "x"])
        self.assertIn("MSVC", str(raised.exception))


    def test_a_failed_attempt_is_kept_and_a_retry_does_not_crash(self):
        # A configure that failed on a toolkit path is fixed and run again. The
        # record of the failure must survive, and the second attempt must not
        # die on "file exists" instead of reporting what the build did.
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            failed = mock.Mock(returncode=1, stdout="", stderr="boom")
            with mock.patch.object(subject.subprocess, "run", return_value=failed):
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        subject._run(out, "CONFIGURE", ["cmake"], {})
            self.assertTrue((out / "CONFIGURE.json").is_file())
            superseded = list(out.glob("CONFIGURE.superseded-*.json"))
            self.assertEqual(len(superseded), 1)

    def test_a_rebuild_keeps_the_receipt_it_replaces(self):
        # `build` writes its symbol and completion receipts at the end. A second
        # `build` over the same directory must be able to replace them, and must
        # not lose the first run's record: the file it overwrote is renamed.
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "libggml-base.so.exports.json"
            subject.write_retained(target, dict(attempt=1))
            subject.write_retained(target, dict(attempt=2))
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             dict(attempt=2))
            kept = list(Path(folder).glob("libggml-base.so.exports.superseded-*.json"))
            self.assertEqual(len(kept), 1)
            self.assertEqual(json.loads(kept[0].read_text(encoding="utf-8")),
                             dict(attempt=1))


@unittest.skipUnless(PINNED.is_dir(), PINNED_MISSING)
class ReceiptTests(unittest.TestCase):
    def test_prepare_writes_the_receipts_the_campaign_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "runtime-linux"
            manifest = subject.prepare(out, PINNED)
            written = json.loads((out / "PREPARED.json").read_text(encoding="utf-8"))
            self.assertEqual(written, json.loads(json.dumps(manifest)))
            self.assertEqual(written["schema"],
                             "real-moe-runtime-window-linux-prepared-1")
            self.assertEqual(written["native_source_sha256"], sha(
                out / "source/real_moe_tap_probe.cpp"))
            # Not claimed: nothing here executed a model.
            self.assertFalse(written["hardware_executed"])
            self.assertEqual(written["extra_GPU_syncs"], 0)
            self.assertEqual(written["CPU_barriers_added"], 0)
            # The copies the tree's build will compile are in place.
            self.assertTrue((out / "tree/ggml/src/ggml-cpu/route_tap_api.h").is_file())
            self.assertTrue((out / "tree/ggml/src/ggml-cuda/gpu_tap_api.h").is_file())
            self.assertTrue((out / "source/private_llama_context.h").is_file())
            # The copy has to be configurable by the tree's own CMakeLists.txt:
            # it includes `cmake/build-info.cmake`, and it must not find the
            # pinned tree's 900 MB of build outputs beside it.
            self.assertTrue((out / "tree/cmake/build-info.cmake").is_file())
            self.assertFalse((out / "tree/build").exists())

    def test_build_refuses_a_prepared_source_that_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "runtime-linux"
            subject.prepare(out, PINNED)
            (out / "source/route_tap_pod.hpp").write_text("edited\n", encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                subject.build(out, None, 1, [])
            self.assertIn("changed since `prepare`", str(raised.exception))

    def test_build_refuses_an_unprepared_root(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as raised:
                subject.build(Path(folder), None, 1, [])
            self.assertIn("prepare this build first", str(raised.exception))


class PrintTests(unittest.TestCase):
    def test_print_shows_the_three_steps_without_touching_the_host(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "is_windows", return_value=False):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = subject.main(["print", "--out", "b", "--cuda-root", "/usr/local/cuda"])
        self.assertEqual(code, 0)
        steps = stream.getvalue().strip().splitlines()
        self.assertEqual(len(steps), 3)
        self.assertTrue(steps[0].startswith("cmake -S"))
        self.assertTrue(steps[1].startswith("cmake --build"))
        self.assertTrue(steps[2].startswith("g++"))


if __name__ == "__main__":
    unittest.main()
