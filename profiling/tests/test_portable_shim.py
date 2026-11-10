"""The depth probe's Win32 surface, checked without a GPU, a compiler or CUDA.

The probe is one compile unit: a small hand-written entry point plus the pinned
generated Executor fragments. Porting it to Linux and macOS was not a rewrite --
the fragments already branch on `_WIN32` and carry real POSIX bodies -- but the
hand-written files call Win32 directly, and `profiling/native/portable/platform.hpp`
supplies those names on POSIX. That division is what these tests hold in place:

  * a call outside a `_WIN32` branch is either a project type or a name the shim
    defines -- so a new `GetSomething()` added without a branch fails here
    instead of on a rented device;
  * the shim defines every name the probes use unguarded;
  * the generated fragments still carry their POSIX bodies (they are byte-locked
    by `generate_sources.py`, so this is about noticing, not fixing);
  * where a C++ compiler exists, the shim itself is compiled and its checks run.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from profiling import host

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "profiling" / "native"
PLATFORM = NATIVE / "portable" / "platform.hpp"
SELFTEST = NATIVE / "portable" / "selftest.cpp"
FRAGMENTS = ROOT / "profiling" / "depth_validation" / "generated"

# Everything the probe compiles: the hand-written half and the pinned fragments.
COMPILE_UNIT = [ROOT / "profiling" / "depth_validation" / "native" / "depth_probe.cpp",
                ROOT / "profiling" / "depth_validation" / "native" / "initial_assets.hpp",
                NATIVE / "steady_costs" / "probe.cpp",
                NATIVE / "steady_costs" / "components.hpp",
                NATIVE / "physical_stages" / "cupti_capture.hpp"] + sorted(FRAGMENTS.glob("*.inc"))

# Constructors of the probe's own types: capitalised because they are C++, not
# because they are Win32. Anything else capitalised outside a `_WIN32` branch has
# to be defined by the shim.
PROJECT_TYPES = frozenset({"ArenaRegistration", "Capture", "Component", "Executor", "Graph",
                           "OwnThreadAffinity", "PrivateWait", "Request", "TimingEvents",
                           # The model column's derived probe and its two observer pods.
                           "BatchOwner", "ContextOwner", "ModelOwner", "SamplerOwner",
                           "ThreadpoolOwner", "TapBinding", "GPUBinding", "GPUCapture",
                           "RuntimeWindows"})

CALL = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\(")
C_PUTENV = re.compile(r"\b_putenv_s\s*\(")


def inside_windows_branch(text: str) -> list[bool]:
    """For every line, whether it sits inside a `_WIN32`-only preprocessor branch.

    Only the shape these files actually use is handled -- `#if defined(_WIN32)`,
    `#ifndef _WIN32`, their `#else`/`#elif`, and nesting -- because that is what
    makes the answer checkable by reading it. A condition that does not mention
    `_WIN32` is neither branch, so lines under it are reported as unguarded and
    have to be accounted for by the shim.
    """
    frames: list[list[bool]] = []          # [is_windows_when_true, in_else]
    result: list[bool] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#if"):
            if stripped.startswith("#ifndef _WIN32") or "!defined(_WIN32)" in stripped:
                frames.append([False, False])      # the true branch is POSIX
            elif "_WIN32" in stripped:
                frames.append([True, False])
            else:
                frames.append([False, False])
            result.append(any(windows != in_else for windows, in_else in frames))
            continue
        if stripped.startswith(("#else", "#elif")):
            if frames:
                frames[-1][1] = not frames[-1][1]
            result.append(any(windows != in_else for windows, in_else in frames))
            continue
        if stripped.startswith("#endif"):
            if frames:
                frames.pop()
            result.append(any(windows != in_else for windows, in_else in frames))
            continue
        result.append(any(windows != in_else for windows, in_else in frames))
    return result


def calls_outside_windows(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    guarded = inside_windows_branch(text)
    lines = text.splitlines()
    assert len(guarded) == len(lines), path
    found: dict[str, int] = {}
    for number, (line, is_guarded) in enumerate(zip(lines, guarded), 1):
        if is_guarded:
            continue
        names = CALL.findall(line)
        if C_PUTENV.search(line):
            names.append("_putenv_s")
        for name in names:
            found.setdefault(name, number)
    return found


def posix_shim_names() -> set[str]:
    """The names `platform.hpp` defines on a POSIX host."""
    text = PLATFORM.read_text(encoding="utf-8")
    match = re.search(r"^#if defined\(_WIN32\).*?^#else$", text, re.S | re.MULTILINE)
    assert match, "platform.hpp no longer has a Windows/POSIX split"
    posix = text[match.end():].split("\n#endif  // _WIN32")[0]
    return (set(CALL.findall(posix))
            | set(re.findall(r"\busing\s+(\w+)\s*=", posix))
            | set(re.findall(r"\bstruct\s+(\w+)", posix))
            | set(re.findall(r"\bconstexpr\s+\w+\s+(\w+)", posix))
            # The one Win32 name that does not start with a capital letter.
            | set(re.findall(r"\b(_putenv_s)\s*\(", posix)))


class BranchParserTests(unittest.TestCase):
    """The checks below are only as good as this parser, so it is tested directly."""

    def test_a_windows_branch_is_recognized_and_its_else_is_not(self):
        text = "#if defined(_WIN32)\nGetFoo();\n#else\nposix();\n#endif\nBare();\n"
        self.assertEqual(inside_windows_branch(text), [True, True, False, False, False, False])

    def test_an_inverted_condition_puts_windows_in_the_else_branch(self):
        text = "#if !defined(_WIN32)\nposix();\n#else\nGetFoo();\n#endif\n"
        self.assertEqual(inside_windows_branch(text), [False, False, True, True, False])

    def test_nesting_keeps_the_outer_branch(self):
        text = "#if defined(_WIN32)\n#if X\nGetFoo();\n#endif\n#endif\nBare();\n"
        self.assertEqual(inside_windows_branch(text), [True, True, True, True, False, False])


class ShimCoverageTests(unittest.TestCase):
    def test_no_win32_call_is_left_unguarded(self):
        supplied = posix_shim_names()
        problems = []
        for path in COMPILE_UNIT:
            for name, number in sorted(calls_outside_windows(path).items()):
                if name in PROJECT_TYPES or name in supplied:
                    continue
                problems.append(f"{path.relative_to(ROOT).as_posix()}:{number} calls {name}")
        self.assertEqual(problems, [],
                         "these calls sit outside `#if defined(_WIN32)` and platform.hpp "
                         "does not define them:\n  " + "\n  ".join(problems))

    def test_the_shim_defines_every_name_the_probes_use(self):
        # The complement of the test above: there, a shimmed name is accepted;
        # here, each name the probes call has to be a real POSIX definition.
        supplied = posix_shim_names()
        for path in COMPILE_UNIT:
            if path.suffix == ".inc":
                continue                      # generated; covered by the test above
            for name, number in calls_outside_windows(path).items():
                if name in PROJECT_TYPES:
                    continue
                self.assertIn(name, supplied, f"{path.name}:{number}")

    def test_the_affinity_refusal_is_a_named_capability_not_a_guess(self):
        text = PLATFORM.read_text(encoding="utf-8")
        self.assertIn("#define PORTABLE_AFFINITY 0", text)
        self.assertIn("sched_setaffinity", text)
        # macOS is the reason the capability exists: it has no enforced affinity.
        self.assertIn("PORTABLE_AFFINITY!=0", (NATIVE / "steady_costs" / "probe.cpp").read_text(
            encoding="utf-8"))
        self.assertIn("PORTABLE_AFFINITY!=0", (ROOT / "profiling" / "depth_validation" / "native"
                                               / "depth_probe.cpp").read_text(encoding="utf-8"))

    def test_the_shim_never_reports_a_fabricated_cycle_count(self):
        text = PLATFORM.read_text(encoding="utf-8")
        body = re.search(r"inline BOOL QueryThreadCycleTime.*?\n}", text, re.S)
        self.assertIsNotNone(body)
        self.assertIn("return FALSE", body.group(0))


class ModelColumnCoverageTests(unittest.TestCase):
    """The model column's probe is derived, not written: check the derived text.

    `linux_build` rewrites the frozen probe and its headers for a POSIX host, so
    the file this column compiles is not the one in the repository. Analysing the
    frozen file would report BCrypt and `<windows.h>` that the derivation has
    already removed; analysing the derivation is also what makes a *new* Win32
    call fail here instead of on a rented device.
    """

    def derived(self) -> dict[str, str]:
        from profiling.real_moe_validation import linux_build
        files = dict(linux_build.support_headers())
        files["real_moe_tap_probe.cpp"] = linux_build.native_source()
        return files

    def test_every_win32_name_the_derived_probe_calls_is_shimmed(self):
        supplied = posix_shim_names()
        problems = []
        with tempfile.TemporaryDirectory(prefix="model-column-") as folder:
            for name, text in self.derived().items():
                path = Path(folder) / name
                path.write_text(text, encoding="utf-8")
                for call, number in sorted(calls_outside_windows(path).items()):
                    if call in PROJECT_TYPES or call in supplied:
                        continue
                    problems.append(f"{name}:{number} calls {call}")
        self.assertEqual(problems, [],
                         "these calls sit outside `#if defined(_WIN32)` and platform.hpp "
                         "does not define them:\n  " + "\n  ".join(problems))

    def test_the_derived_probe_and_pods_keep_no_windows_only_construct(self):
        for name, text in self.derived().items():
            if name == "platform.hpp":
                continue      # the shim *is* the Windows surface, by design
            self.assertNotIn("__declspec", text, name)
            self.assertNotIn("#include <windows.h>", text, name)
            self.assertNotIn("<bcrypt.h>", text, name)

    def test_the_shim_looks_modules_up_the_way_the_probe_binds_its_hooks(self):
        # GetModuleHandle/GetProcAddress are how the pods reach the library they
        # are running inside; a lookup that loaded a second copy would observe
        # nothing, so the POSIX body must refuse to load one.
        text = PLATFORM.read_text(encoding="utf-8")
        self.assertIn("RTLD_NOLOAD", text)
        self.assertIn("dlsym", text)
        self.assertIn('"lib" + text + ".so"', text)
        self.assertIn('"lib" + text + ".dylib"', text)


class GeneratedFragmentTests(unittest.TestCase):
    def test_the_platform_dependent_bodies_survive_generation(self):
        # `numeric_helpers.inc` is arithmetic and has no platform branch at all;
        # the clock lives in the single-layer fragment and the wait and the
        # exclusive output creation in the state-response one. They are
        # byte-locked, so this is a check that a regeneration kept them, not an
        # invitation to edit them.
        text = "".join(path.read_text(encoding="utf-8") for path in sorted(FRAGMENTS.glob("*.inc")))
        for body in ("clock_gettime(CLOCK_MONOTONIC", "sleep_for", "O_CREAT|O_EXCL", "setenv("):
            self.assertIn(body, text)


@unittest.skipIf(host.is_windows(),
                 "on Windows the shim is <windows.h> itself; this proves the POSIX half")
class CompilerProofTests(unittest.TestCase):
    def test_the_shim_compiles_and_its_checks_pass_on_this_host(self):
        compiler = next((found for found in
                         (shutil.which("c++"), shutil.which("g++"), shutil.which("clang++")) if found),
                        None)
        if compiler is None:
            self.skipTest("no C++ compiler on PATH")
        with tempfile.TemporaryDirectory(prefix="portable-selftest-") as folder:
            binary = Path(folder) / "portable-selftest"
            build = subprocess.run([compiler, "-std=c++17", "-O2", "-o", str(binary), str(SELFTEST)],
                                   capture_output=True, text=True, timeout=600)
            self.assertEqual(build.returncode, 0, build.stderr[-4000:])
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=180)
            self.assertEqual(run.returncode, 0,
                             "the shim's own checks failed:\n" + run.stdout[-4000:])
            self.assertIn("0 failure(s)", run.stdout)


if __name__ == "__main__":
    unittest.main()
