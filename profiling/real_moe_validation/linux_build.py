"""Build the real-MoE runtime-window probe on Linux from a pinned llama.cpp tree.

The Windows chain (`tap_v4` -> `tap_v5` -> `numerical_v6` -> `mmid_runtime_fix` ->
`runtime_windows`) recompiles one translation unit at a time inside an existing
MSVC build and reconstructs each compiler command from the project's `.tlog`
files. None of that survives a change of host: the tlogs, the `.vcxproj`
contracts, `/DLL /MACHINE:X64` and `dumpbin` are all MSVC's.

What *does* survive is the part that matters -- the four source transforms that
insert the observer hooks, each of which refuses to apply unless its anchor is
byte-exactly where the published build found it:

  * `ggml/src/ggml-backend.cpp`   the graph/synchronize/copy window
  * `ggml/src/ggml-cpu/ggml-cpu.c` the CPU consumer tap
  * `ggml/src/ggml-cuda/ggml-cuda.cu` the same-stream GPU consumer tap
  * `src/llama-context.h`          two read-only accessors
  * `real_moe_tap_probe.cpp`       the v4 -> v5 -> runtime-window native edits

So this module reuses those transforms unchanged and replaces only the build
plumbing: a private copy of the pinned tree is patched, configured with CMake and
rebuilt whole. Rebuilding is acceptable on a Linux host -- the Windows chain
exists because a full MSVC rebuild on the laptop that produced the published
table was too slow, and that constraint does not bind here.

    python -m profiling.real_moe_validation.linux_build prepare \\
        --out build/runtime-linux --llama-root <llama.cpp b8705>
    python -m profiling.real_moe_validation.linux_build build \\
        --out build/runtime-linux --cuda-root /usr/local/cuda

`prepare` writes the patched copy and `PREPARED.json`; `build` configures and
compiles, checks the two observer symbols really are exported, copies the probe
and the libraries it loads into `bin/`, and writes `BUILD_COMPLETE.json` in the
same schema the Windows chain writes, because `runtime_window_campaign.prepare`
reads exactly that.

The portability edits applied on top of the Windows transforms are the ones a
POSIX host needs, each on the files that need it and no others:

  * the probe's `<windows.h>` becomes `profiling/native/portable/platform.hpp`
    (the same shim the depth and steady-cost probes use: QPC, affinity, process
    id and the exclusive CREATE_NEW output write all come from it);
  * `route_tap_pod.hpp`'s `<bcrypt.h>` becomes the in-tree `sha256.hpp` -- one
    SHA-256 on every host, proven against NIST's vectors by
    `portable/selftest.cpp`;
  * `__declspec(dllexport)` is dropped from the two injected hooks: they are
    `extern "C"` and ggml does not compile with hidden visibility, so they are
    exported anyway;
  * `common/debug.cpp` -- this repository's own `ffn_moe_topk` diagnostic, not an
    observer -- gets `#include <cstring>`, which MSVC's headers lend it and
    libstdc++'s do not. Without it the whole tree fails to build at 93 %.

ggml's own C sources already branch on `_WIN32`, so their Windows includes and
`__declspec(align(...))` are left exactly as the tree has them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from profiling import host

from . import runtime_windows, tap_v4, tap_v5
from .tap_v4 import receipt, sha, write

HERE = Path(__file__).resolve().parent
NATIVE = HERE / "tap_native"
NATIVE_V5 = HERE / "tap_native_v5"
PORTABLE = HERE.parent / "native" / "portable"
PROBE = "real-moe-runtime-window-probe"

LIBRARY_NAMES = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                 "libggml-cuda.so")

# Every symbol the probe binds by name, in the library that has to export it.
# A missing one is not a warning: the probe would find a null observer and
# measure nothing, which is the failure mode this table exists to prevent. The
# names are the ones `GetProcAddress` is called with in the derived sources, and
# `test_linux_build` checks that the two sets still agree.
REQUIRED_SYMBOLS = {
    "libggml-base.so": ("simulator_runtime_window_set",),
    "libggml-cpu.so": ("simulator_moe_tap_set", "simulator_runtime_cpu_window_set"),
    "libggml-cuda.so": ("simulator_gpu_tap_set_v5", "simulator_gpu_tap_host_alloc_v5",
                        "simulator_gpu_tap_host_free_v5"),
}

# The portability edits, each applied to the files that need it and no others:
# ggml's own C sources already branch on `_WIN32`, so rewriting their Windows
# includes would be an edit with no effect on the host and a silent change on
# the other one.
EXPORT = "__declspec(dllexport) "
PROBE_INCLUDE = "#include <windows.h>"
PROBE_INCLUDE_REPLACEMENT = '#include "platform.hpp"'
BCRYPT_INCLUDE = "#include <bcrypt.h>"
BCRYPT_INCLUDE_REPLACEMENT = '#include "sha256.hpp"'
CLOCK_LITERAL = '{"clock_id","windows_qpc"}'
CLOCK_REPLACEMENT = '{"clock_id",PORTABLE_CLOCK_ID}'
STRING_INCLUDE = "#include <string>"
CSTRING_INCLUDE = "#include <cstring>\n#include <string>"

SHA_FUNCTION = re.compile(
    r"static std::string tap_sha256\(const void \* data, size_t bytes\) \{.*?\n\}", re.S)
SHA_REPLACEMENT = '''static std::string tap_sha256(const void * data, size_t bytes) {
 return portable_sha256::digest_hex(static_cast<const char *>(data), bytes);
}'''


def replace_once(text: str, anchor: str, replacement: str, why: str) -> str:
    """Exactly one occurrence, or a refusal naming what the anchor was for."""
    if text.count(anchor) != 1:
        raise ValueError("expected one %s (%s), found %d" % (anchor, why, text.count(anchor)))
    return text.replace(anchor, replacement, 1)


def remove_export(text: str) -> str:
    """The injected hook exports `extern "C"`, which POSIX exports without a keyword."""
    return replace_once(text, EXPORT, "",
                        "observer export in the injected hook")


def portable_probe(text: str) -> str:
    """The probe's one Win32 include, replaced by the shared shim."""
    return replace_once(text, PROBE_INCLUDE, PROBE_INCLUDE_REPLACEMENT,
                        "Win32 timing, affinity, process id and exclusive file creation")


def portable_clock(text: str) -> str:
    """The clock the probe's receipt names, in the host's spelling.

    The frozen source says `windows_qpc` because that is the clock the published
    cohort ran on. On a POSIX host the same `QueryPerformanceCounter` is
    CLOCK_MONOTONIC at a declared 1 ns tick, so a receipt that still said
    `windows_qpc` would be a false provenance line in a V-A row. `platform.hpp`
    defines both strings, and on Windows `PORTABLE_CLOCK_ID` is literally
    `windows_qpc` -- so the published receipts do not change.
    """
    return replace_once(text, CLOCK_LITERAL, CLOCK_REPLACEMENT,
                        "the clock the probe stamped with")


def portable_pod(text: str) -> str:
    """`route_tap_pod.hpp` with the in-tree SHA-256 instead of BCrypt."""
    text = replace_once(text, BCRYPT_INCLUDE, BCRYPT_INCLUDE_REPLACEMENT,
                        "SHA-256 from the in-tree implementation")
    if len(SHA_FUNCTION.findall(text)) != 1:
        raise ValueError("the pinned SHA-256 helper is not where the build expects it")
    return SHA_FUNCTION.sub(SHA_REPLACEMENT, text)


def portability_edits() -> list[dict]:
    """What was changed, per file, and why -- the manifest says which edits ran."""
    return [
        dict(file="source/real_moe_tap_probe.cpp", anchor=PROBE_INCLUDE,
             replacement=PROBE_INCLUDE_REPLACEMENT,
             makes_portable="Win32 timing, affinity, process id and exclusive file creation"),
        dict(file="source/route_tap_pod.hpp", anchor=BCRYPT_INCLUDE,
             replacement=BCRYPT_INCLUDE_REPLACEMENT,
             makes_portable="SHA-256 from the in-tree implementation"),
        dict(file="source/real_moe_tap_probe.cpp", anchor=CLOCK_LITERAL,
             replacement=CLOCK_REPLACEMENT,
             makes_portable="the receipt names the clock the probe actually stamped with"),
        dict(file="source/route_tap_api.h", anchor="__declspec(dllexport)",
             replacement='__attribute__((visibility("default")))',
             makes_portable="the same export promise in the host's spelling"),
        dict(file="source/gpu_tap_api.h", anchor="__declspec(dllexport)",
             replacement='__attribute__((visibility("default")))',
             makes_portable="the same export promise in the host's spelling"),
        dict(file="ggml/src/ggml-backend.cpp", anchor=EXPORT, replacement="",
             makes_portable='the injected hook is extern "C" and is exported anyway'),
        dict(file="ggml/src/ggml-cpu/ggml-cpu.c", anchor=EXPORT, replacement="",
             makes_portable='the injected hook is extern "C" and is exported anyway'),
        dict(file="common/debug.cpp", anchor=STRING_INCLUDE, replacement=CSTRING_INCLUDE,
             makes_portable="the C string functions this file calls, which MSVC lends it"),
    ]


def copy_ignore(directory: str, names: list[str]) -> list[str]:
    """What the private copy leaves behind: another build's outputs, and the weights.

    Three rules, each learned from a configure that failed without them.

    * `build` and `build-*` are *directories*, never names: the same pattern
      applied to names also matches `cmake/build-info.cmake`, which the tree's
      own `CMakeLists.txt` includes, and a copy without it fails at configure
      with "include could not find requested file" -- a port that reads as a
      broken checkout rather than an over-eager filter.
    * `.git` is never a build input.
    * `models/` is a GGUFS directory in a checkout that downloads weights into
      it and the home of the architecture sources (`models/afmoe.cpp`, ...) in
      this one. It is skipped only when it holds no C++ source the build would
      compile, which is also the test for which of the two it is.
    """
    ignored = []
    for name in names:
        path = os.path.join(directory, name)
        if name == ".git":
            ignored.append(name)
        elif os.path.isdir(path) and (name == "build" or name.startswith("build-")):
            ignored.append(name)
        elif os.path.isdir(path) and name == "models" and not any(
                child.endswith((".cpp", ".h", ".hpp", ".c")) for child in os.listdir(path)):
            ignored.append(name)
    return ignored


def build_environment(cuda_root: Path | None, base: dict | None = None) -> dict:
    """The environment the configure and compile steps run under.

    A toolkit named by `--cuda-root` is not necessarily on `PATH` -- a host
    that installed it under `/usr/local/cuda-12.4` and never edited a profile
    has no `nvcc` on the search path -- and CMake looks for the CUDA compiler
    there. Naming it directly is the difference between "No
    CMAKE_CUDA_COMPILER could be found" and a build.
    """
    environment = dict(os.environ if base is None else base)
    if cuda_root is None:
        return environment
    binaries = Path(cuda_root).resolve() / "bin"
    environment["PATH"] = os.pathsep.join(
        [str(binaries), environment.get("PATH", "")])
    nvcc = binaries / "nvcc"
    if nvcc.is_file():
        environment["CUDACXX"] = str(nvcc)
    return environment


def portable_cstring(text: str) -> str:
    """`common/debug.cpp` asks for `<cstring>` instead of inheriting it.

    That file carries this repository's MoE routing diagnostic (`ffn_moe_topk`)
    and calls `strstr`/`memcpy` with only `<cmath>` and `<string>` included.
    MSVC's standard library brings `<cstring>` in transitively, so the published
    build never needed the line; libstdc++ does not, and the whole tree fails to
    build with "'strstr' was not declared in this scope". Adding the include to
    the private copy is the whole fix -- the pinned tree stays as measured.
    """
    if "#include <cstring>" in text:
        return text
    return replace_once(text, STRING_INCLUDE, CSTRING_INCLUDE,
                        "the C string functions `common/debug.cpp` calls")


def patched_sources(llama_root: Path) -> dict[str, str]:
    """Every file the observer touches, as it has to be written into the copy."""
    root = Path(llama_root)
    backend = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    cpu = (root / "ggml/src/ggml-cpu/ggml-cpu.c").read_text(encoding="utf-8")
    cuda = (root / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    context = (root / "src/llama-context.h").read_text(encoding="utf-8")
    debug = (root / "common/debug.cpp").read_text(encoding="utf-8")
    return {
        "ggml/src/ggml-backend.cpp": remove_export(runtime_windows.patched_base(backend)),
        # Two taps in one file: the consumer-only tap this repository already
        # ships, then the runtime window that observes around it.
        "ggml/src/ggml-cpu/ggml-cpu.c": remove_export(
            runtime_windows.patched_cpu(tap_v4.patched_cpu(cpu))),
        "ggml/src/ggml-cuda/ggml-cuda.cu": tap_v5.patched_cuda(cuda),
        "src/llama-context.h": tap_v4.private_context_header(context),
        # Not an observer: the tree's own MoE diagnostic, which MSVC compiles
        # only because its headers leak `<cstring>` into every translation unit.
        "common/debug.cpp": portable_cstring(debug),
    }


def native_source() -> str:
    """The probe, after the three chained transforms the Windows chain applies.

    The base file is read, never edited: `tap_native/real_moe_tap_probe.cpp` is
    the frozen source the published campaign's `native_source_sha256` names, and
    the platform edits belong in the derived copy.
    """
    base = (NATIVE / "real_moe_tap_probe.cpp").read_text(encoding="utf-8")
    return portable_clock(portable_probe(
        runtime_windows.patched_native(tap_v5.native_v5(base))))


def portable_exports(text: str) -> str:
    """A tap header's export promise in the host's spelling.

    Both API headers guard the same two macros -- `..._API` is `dllexport` inside
    the observer build and `dllimport` for its consumers. On POSIX the producer
    side is the visibility attribute and the consumer side is nothing at all.
    """
    if text.count("__declspec(dllexport)") != 1 or text.count("__declspec(dllimport)") != 1:
        raise ValueError("a tap API header no longer spells its exports the MSVC way")
    return text.replace("__declspec(dllexport)",
                        '__attribute__((visibility("default")))').replace(
        "__declspec(dllimport)", "")


def support_headers() -> dict[str, str]:
    """The private headers the probe includes, made portable where they are not."""
    files = {
        "route_tap_pod.hpp": (NATIVE / "route_tap_pod.hpp").read_text(encoding="utf-8"),
        "route_tap_api.h": (NATIVE / "route_tap_api.h").read_text(encoding="utf-8"),
        "route_context_inspect.hpp": (NATIVE / "route_context_inspect.hpp").read_text(encoding="utf-8"),
        "gpu_tap_api.h": (NATIVE_V5 / "gpu_tap_api.h").read_text(encoding="utf-8"),
        "gpu_tap_pod.hpp": (NATIVE_V5 / "gpu_tap_pod.hpp").read_text(encoding="utf-8"),
        "runtime_windows.hpp": (HERE / "runtime_windows_native/runtime_windows.hpp").read_text(
            encoding="utf-8"),
        "platform.hpp": (PORTABLE / "platform.hpp").read_text(encoding="utf-8"),
        "sha256.hpp": (PORTABLE / "sha256.hpp").read_text(encoding="utf-8"),
    }
    # The in-tree headers are left alone; these are the derived copies.
    files["route_tap_api.h"] = portable_exports(files["route_tap_api.h"])
    files["gpu_tap_api.h"] = portable_exports(files["gpu_tap_api.h"])
    files["route_tap_pod.hpp"] = portable_pod(files["route_tap_pod.hpp"])
    return files


def prepare(out: Path, llama_root: Path) -> dict:
    out, llama_root = Path(out).resolve(), Path(llama_root).resolve()
    if out.exists():
        raise ValueError("new private build root required")
    if not (llama_root / "ggml/src/ggml-cuda/ggml-cuda.cu").is_file():
        raise ValueError("the pinned tree needs ggml/src/ggml-cuda/ggml-cuda.cu: %s" % llama_root)
    sources = patched_sources(llama_root)
    probe = native_source()
    headers = support_headers()
    # The observer TUs are compiled by the tree's own build, so they have to live
    # in a copy of it: the pinned checkout is evidence of what was measured and is
    # never edited. The copy keeps the tree's paths, which is what lets the
    # `#include "gpu_tap_api.h"` the CUDA transform inserts resolve.
    tree = out / "tree"
    shutil.copytree(llama_root, tree, symlinks=True, ignore=copy_ignore)
    for name in ("source", "objects", "bin", "lib"):
        (out / name).mkdir(parents=True, exist_ok=True)
    for name, text in sources.items():
        (tree / name).write_text(text, encoding="utf-8", newline="\n")
    for directory in (tree / "ggml/src/ggml-cpu", tree / "ggml/src/ggml-cuda"):
        (directory / "route_tap_api.h").write_text(headers["route_tap_api.h"],
                                                   encoding="utf-8", newline="\n")
    (tree / "ggml/src/ggml-cuda/gpu_tap_api.h").write_text(headers["gpu_tap_api.h"],
                                                           encoding="utf-8", newline="\n")
    for name, text in headers.items():
        (out / "source" / name).write_text(text, encoding="utf-8", newline="\n")
    (out / "source/private_llama_context.h").write_text(
        sources["src/llama-context.h"], encoding="utf-8", newline="\n")
    (out / "source/real_moe_tap_probe.cpp").write_text(probe, encoding="utf-8", newline="\n")
    manifest = dict(
        schema="real-moe-runtime-window-linux-prepared-1",
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        llama_root=str(llama_root),
        patched=[receipt(tree / name) for name in sorted(sources)],
        sources=[receipt(path) for path in sorted((out / "source").iterdir())],
        native_source_sha256=sha(out / "source/real_moe_tap_probe.cpp"),
        original_base_source=receipt(llama_root / "ggml/src/ggml-backend.cpp"),
        original_cpu_source=receipt(llama_root / "ggml/src/ggml-cpu/ggml-cpu.c"),
        original_cuda_source=receipt(llama_root / "ggml/src/ggml-cuda/ggml-cuda.cu"),
        original_context_header=receipt(llama_root / "src/llama-context.h"),
        original_native_source=receipt(NATIVE / "real_moe_tap_probe.cpp"),
        portability_edits=portability_edits(),
        mathematical_graph_changed=False, extra_GPU_syncs=0, CPU_barriers_added=0,
        CUDA_objects_rebuilt=False, whole_model_price_supported=False, hardware_executed=False)
    write(out / "PREPARED.json", manifest)
    return manifest


def _run(out: Path, name: str, argv: list[str], env: dict) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    done = subprocess.run(argv, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    record = dict(command=argv, started_at_utc=started,
                  ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  returncode=done.returncode, stdout=done.stdout[-20000:],
                  stderr=done.stderr[-20000:], hardware_executed=False)
    target = out / (name + ".json")
    if target.exists():
        # A retry after fixing a toolkit path must not destroy the record of the
        # attempt that failed -- and must not fail with "file exists", which says
        # nothing about the build. Every attempt is kept; the canonical name
        # holds the latest one.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target.rename(out / ("%s.superseded-%s.json" % (name, stamp)))
    write(target, record)
    if done.returncode:
        raise RuntimeError(name + " failed; the execution receipt is retained")
    return record


def write_retained(target: Path, record: dict) -> Path:
    """Write `target`, keeping whatever was there under a superseded name.

    `write()` opens with mode 'x', which is the right rule for a receipt that
    must never be silently replaced. It is the wrong rule for the receipts a
    rebuild writes over its own output: re-running `build` in place would stop
    with "file exists", a message that says nothing about the build. The earlier
    file is renamed rather than removed, exactly as `_run` keeps earlier tries,
    so the canonical name holds the latest attempt and the history survives.
    """
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target.rename(target.with_name("%s.superseded-%s%s" % (target.stem, stamp,
                                                               target.suffix)))
    write(target, record)
    return target


def configure_command(out: Path, cuda_root: Path | None, extra: list[str]) -> list[str]:
    command = ["cmake", "-S", str(out / "tree"), "-B", str(out / "tree/build")]
    command += ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DBUILD_SHARED_LIBS=ON",
                "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF"]
    if cuda_root is not None:
        command.append("-DCUDAToolkit_ROOT=%s" % Path(cuda_root).resolve())
    return command + list(extra)


def cuda_runtime_dirs(cuda_root: Path | None) -> list[Path]:
    """The toolkit directories that hold `libcudart.so` on this layout.

    `libggml-cuda.so` records `libcudart.so.12` as a needed library, so the probe
    will not even start if the loader cannot find the toolkit. A host that keeps
    its toolkit outside the loader's search path (a tarball install, `/opt/cuda`,
    a DGX image with `/usr/local/cuda-13`) would otherwise fail at launch with
    "while loading shared libraries: libcudart.so.12" -- after a successful
    build. Naming the directory in the probe's rpath is the fix; the campaign's
    runtime directories stay the authority at run time, because `LD_LIBRARY_PATH`
    is searched before `DT_RUNPATH`.
    """
    if cuda_root is None:
        return []
    root = Path(cuda_root).resolve()
    candidates = [root / "lib64", root / "targets/aarch64-linux/lib",
                  root / "targets/x86_64-linux/lib", root / "lib"]
    return [directory for directory in candidates if directory.is_dir()]


def probe_command(out: Path, compiler: str, source_sha256: str,
                  cuda_root: Path | None = None) -> list[str]:
    """The probe's own compile-and-link line.

    `source_sha256` is a parameter rather than a read: it is the value `prepare`
    recorded and `build` re-verified, and a line printed before either has run
    must not claim a digest it has not checked.
    """
    tree = out / "tree"
    include = [out / "source", tree / "ggml/include", tree / "ggml/src",
               tree / "src", tree / "include", tree / "vendor"]
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-o", str(out / "bin" / PROBE),
               str(out / "source/real_moe_tap_probe.cpp")]
    command += ["-I" + str(path) for path in include]
    command += ["-DGGML_SHARED", "-DGGML_BACKEND_SHARED",
                '-DREAL_MOE_SOURCE_SHA256="%s"' % source_sha256]
    libraries = out / "tree/build/bin"
    command += ["-L" + str(libraries), "-lllama", "-lggml", "-lggml-base", "-lggml-cpu",
                "-lggml-cuda"]
    for directory in [libraries] + cuda_runtime_dirs(cuda_root):
        command.append("-Wl,-rpath," + str(directory))
    # `dlopen`/`dlsym` (the shim's GetModuleHandle/GetProcAddress) and threads.
    command += ["-ldl", "-lpthread"]
    return command


def exported_symbols(binary: Path, symbol: str) -> bool:
    """Whether a shared library really exports `symbol`."""
    done = subprocess.run(["nm", "-D", "--defined-only", str(binary)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return done.returncode == 0 and symbol in done.stdout


def cuda_architectures(extra_cmake: list[str]) -> str:
    """What the CUDA objects were compiled for, as the receipt has to say it.

    A binary built for `sm_89` refuses to launch kernels on a `sm_121` device
    with "no kernel image is available", which reads like a broken probe rather
    than a build for another GPU, and a group of architectures that includes the
    device is portable but slow to compile. Neither is a default the receipt may
    leave unstated: when the caller names one it is recorded, and when it does
    not the receipt says so in words rather than pretending a value.
    """
    for argument in extra_cmake:
        if argument.startswith("-DCMAKE_CUDA_ARCHITECTURES="):
            return argument.split("=", 1)[1]
    return "cmake default (the toolkit's own list; no -DCMAKE_CUDA_ARCHITECTURES given)"


def nvcc_version(environment: dict) -> str:
    """The toolkit the CUDA half of the build was compiled by, as nvcc says it."""
    candidates = [environment["CUDACXX"]] if environment.get("CUDACXX") else []
    candidates.append("nvcc")                 # `build_environment` put the toolkit first
    for candidate in candidates:
        done = subprocess.run([candidate, "--version"], env=environment, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if done.returncode == 0:
            lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
            return " | ".join(lines[-2:])
    return "nvcc not found on PATH or in the named toolkit"


def compute_capability() -> str | None:
    """The compute capability of the GPU this host would measure on.

    CMake's default architecture list belongs to the toolkit, not to the device:
    a build whose list stops at `sm_90` runs on an `sm_121` DGX Spark and answers
    "no kernel image is available for execution on the device", which names
    neither the toolkit nor the flag that would fix it. Asking the driver is the
    question the build actually has. `None` -- no `nvidia-smi`, several GPUs
    with different capabilities, or an unparsable answer -- is recorded as "not
    asked" rather than replaced by a guess.
    """
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    values = {line.strip() for line in done.stdout.splitlines() if line.strip()}
    if len(values) != 1:                     # no GPU, or a heterogeneous pair
        return None
    digits = values.pop().replace(".", "")
    return digits if digits.isdigit() else None


def resolved_architectures(extra_cmake: list[str]) -> tuple[list[str], str]:
    """The configure arguments, with the device's own architecture unless one was named."""
    for argument in extra_cmake:
        if argument.startswith("-DCMAKE_CUDA_ARCHITECTURES="):
            return list(extra_cmake), "named by the caller"
    detected = compute_capability()
    if detected is None:
        return list(extra_cmake), "cmake default (the toolkit's own list)"
    return list(extra_cmake) + ["-DCMAKE_CUDA_ARCHITECTURES=" + detected], \
        "detected from nvidia-smi compute_cap"


def build(out: Path, cuda_root: Path | None, jobs: int, extra_cmake: list[str],
          compiler: str = "g++") -> dict:
    out = Path(out).resolve()
    if not (out / "PREPARED.json").is_file():
        raise ValueError("prepare this build first")
    manifest = json.loads((out / "PREPARED.json").read_text(encoding="utf-8"))
    for entry in manifest["sources"] + manifest["patched"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("a prepared source changed since `prepare`: " + entry["path"])
    env = build_environment(cuda_root)
    extra_cmake, architecture_source = resolved_architectures(extra_cmake)
    _run(out, "CONFIGURE", configure_command(out, cuda_root, extra_cmake), env)
    _run(out, "COMPILE", ["cmake", "--build", str(out / "tree/build"), "--config", "Release",
                          "-j", str(jobs)], env)
    _run(out, "NATIVE_COMPILE",
         probe_command(out, compiler, manifest["native_source_sha256"], cuda_root),
         env)
    libraries = out / "tree/build/bin"
    for name in LIBRARY_NAMES:
        source = libraries / name
        if not source.is_file():
            raise ValueError("the build did not produce %s" % source)
        shutil.copy2(source, out / "bin" / name)
    # Every symbol the probe binds by name, in the library that has to export it.
    for library, symbols in REQUIRED_SYMBOLS.items():
        found = {symbol: exported_symbols(libraries / library, symbol) for symbol in symbols}
        write_retained(out / ("%s.exports.json" % library),
                       dict(library=str(libraries / library), symbols=found,
                            hardware_executed=False))
        missing = sorted(symbol for symbol, present in found.items() if not present)
        if missing:
            raise ValueError("%s does not export %s" % (library, ", ".join(missing)))
    result = dict(schema="real-moe-runtime-window-build-complete-1",
                  prepared=receipt(out / "PREPARED.json"),
                  binaries=[receipt(path) for path in sorted((out / "bin").iterdir())],
                  compiled=True, mathematical_graph_changed=False, extra_GPU_syncs=0,
                  CPU_barriers_added=0, CUDA_objects_rebuilt=False, hardware_executed=False,
                  build_host=dict(platform=sys.platform,
                                  library_pattern=host.shared_library_pattern(),
                                  compiler=compiler, jobs=jobs,
                                  cmake_arguments=extra_cmake,
                                  cuda_root=str(Path(cuda_root).resolve()) if cuda_root else None,
                                  cuda_architectures=cuda_architectures(extra_cmake),
                                  cuda_architecture_source=architecture_source,
                                  nvcc=nvcc_version(env)))
    write_retained(out / "BUILD_COMPLETE.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "build", "print"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path,
                        help="the pinned llama.cpp b8705 checkout (prepare)")
    parser.add_argument("--cuda-root", type=Path, help="CUDA toolkit root (build)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cmake-argument", action="append", default=[],
                        help="extra -D argument, e.g. -DCMAKE_CUDA_ARCHITECTURES=121")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args(argv)
    if host.is_macos():
        raise SystemExit("the model column's observer is a hook inside the llama.cpp CUDA "
                         "backend, and no macOS release has CUDA: this column runs on a "
                         "CUDA host. A Mac measures the dense column instead")
    if args.command != "print" and host.is_windows():
        raise SystemExit("this is the POSIX build path: on Windows the published "
                         "tap chain (tap_v4 -> tap_v5 -> runtime_windows) drives MSVC")
    if args.command == "prepare":
        if args.llama_root is None:
            raise SystemExit("prepare needs --llama-root: the pinned llama.cpp b8705 checkout")
        result = prepare(args.out, args.llama_root)
    elif args.command == "print":
        out = Path(args.out).resolve()
        # A printed line may be run before `prepare` has; in that case say where
        # the digest comes from rather than reading a file that is not there.
        prepared = out / "PREPARED.json"
        source_sha256 = (json.loads(prepared.read_text(encoding="utf-8"))["native_source_sha256"]
                         if prepared.is_file() else "<native_source_sha256 from PREPARED.json>")
        printed, _ = resolved_architectures(args.cmake_argument)
        print(" ".join(configure_command(out, args.cuda_root, printed)))
        print(" ".join(["cmake", "--build", str(out / "tree/build"), "--config", "Release",
                        "-j", str(args.jobs)]))
        print(" ".join(probe_command(out, args.compiler, source_sha256, args.cuda_root)))
        return 0
    else:
        result = build(args.out, args.cuda_root, args.jobs, args.cmake_argument, args.compiler)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
