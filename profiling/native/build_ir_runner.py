#!/usr/bin/env python3
"""Build ``ggml-ir-runner`` on a machine that has never seen this project.

The probes in this directory link against an existing ggml build; they never
build ggml themselves.  On the machine the campaigns were measured on, that
build was a llama.cpp checkout at tag ``b8705`` configured into
``llama.cpp-implement/release-b8705/llama.cpp-b8705/build``.  Nothing recorded
how it got there, so a fresh clone could compile the C++ but had no ggml to
link against.  This script supplies that missing step and nothing else:

    python profiling/native/build_ir_runner.py --fetch        # clone + build ggml, then us
    python profiling/native/build_ir_runner.py --llama PATH   # reuse a llama.cpp checkout
    python profiling/native/build_ir_runner.py --print        # show the commands only

What it does, in order:

1. **ggml**: reuse the checkout given by ``--llama``, or (``--fetch``) clone
   llama.cpp at the pinned tag into ``profiling/native/_llama`` and build
   ``ggml-base``/``ggml-cpu`` (+ ``ggml-cuda`` when CUDA is found) there.  The
   tag is pinned because a different ggml is a different measurement, not a
   different binary.
2. **the runner**: configure ``profiling/native`` with ``GGML_SOURCE_DIR`` /
   ``GGML_BUILD_DIR`` pointing at that build and build the ``ggml-ir-runner``
   target into ``build-ir``.

The binary is then selected at run time through ``IR_RUNNER``; nothing in this
repository hard-codes a path to it.

CUDA is optional.  ``GGML_CUDA=ON`` makes ggml configure a CUDA language, so it
cannot be left on for a machine that has no toolkit: CUDA is used when ``nvcc``
is on PATH, ``--no-cuda`` turns it off, and ``--cuda`` refuses to continue if the
toolkit is missing rather than silently building a CPU-only executor.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build-ir"
FETCH_DIR = HERE / "_llama"
PINNED_TAG = "b8705"
# Tags move; this one did not.  The check is not ceremony: the executor links against a
# particular ggml, and a silently different one would change the measurements the
# repository's price tables were made from.
PINNED_COMMIT = "09343c01986667b26d414733950c9d38e91c3065"
PINNED_URL = "https://github.com/ggml-org/llama.cpp"

# llama.cpp targets that carry the libraries the CMakeLists finds by name
# (ggml-base, ggml-cpu, ggml-cuda).  Building only these keeps the fetch short.
GGML_TARGETS = ["ggml-base", "ggml-cpu"]
GGML_CUDA_TARGETS = ["ggml-cuda"]


def run(call: list[str], *, print_only: bool) -> int:
    print(" ".join(call), flush=True)
    if print_only:
        return 0
    done = subprocess.run(call)
    if done.returncode != 0:
        print(f"FAILED ({done.returncode}): {' '.join(call)}", file=sys.stderr)
    return done.returncode


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def checkout_commit(dest: Path) -> str | None:
    done = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def fetch_llama(dest: Path, *, print_only: bool) -> Path:
    """Clone the pinned llama.cpp and build its ggml.  Returns the checkout root."""
    if not dest.exists():
        if not have("git"):
            raise SystemExit("--fetch needs git on PATH, or pass --llama PATH")
        rc = run(["git", "clone", "--depth", "1", "--branch", PINNED_TAG, PINNED_URL, str(dest)],
                 print_only=print_only)
        if rc:
            raise SystemExit(rc)
    if print_only:
        return dest
    commit = checkout_commit(dest)
    if commit != PINNED_COMMIT:
        raise SystemExit(f"{dest} is at {commit}, but the pinned tag {PINNED_TAG} is "
                         f"{PINNED_COMMIT}; the executor's prices were measured against "
                         f"that ggml.  Remove the directory to re-fetch, or pass "
                         f"--llama if you mean a different one.")
    return dest


def cmake_ggml(llama: Path, *, cuda: bool, print_only: bool) -> int:
    build = llama / "build"
    configure = ["cmake", "-S", str(llama), "-B", str(build),
                 "-DBUILD_SHARED_LIBS=ON",
                 "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                 "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_CURL=OFF",
                 f"-DGGML_CUDA={'ON' if cuda else 'OFF'}"]
    rc = run(configure, print_only=print_only)
    if rc:
        return rc
    targets = GGML_TARGETS + (GGML_CUDA_TARGETS if cuda else [])
    return run(["cmake", "--build", str(build), "--config", "Release",
                "--target", *targets], print_only=print_only)


def find_runner(out: Path) -> Path | None:
    """Multi-config generators put the binary in Release/, single-config at the root."""
    for candidate in (out / "Release" / "ggml-ir-runner.exe", out / "Release" / "ggml-ir-runner",
                      out / "ggml-ir-runner.exe", out / "ggml-ir-runner"):
        if candidate.exists():
            return candidate
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama", type=Path,
                        help="a llama.cpp checkout that already has (or can get) a ggml build")
    parser.add_argument("--fetch", action="store_true",
                        help=f"clone {PINNED_URL} at tag {PINNED_TAG} into {FETCH_DIR.name}/ first")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="CMake tree for the runner (default: profiling/native/build-ir)")
    parser.add_argument("--cuda", action="store_true",
                        help="require the CUDA toolkit; refuse to build without it "
                             "(default: on iff nvcc is on PATH)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="configure CUDA off; the runner then refuses GPU nodes")
    parser.add_argument("--clean", action="store_true", help="delete the runner's build tree first")
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="print the commands without running them")
    args = parser.parse_args(argv)

    if args.cuda and args.no_cuda:
        raise SystemExit("pass either --cuda or --no-cuda, not both")
    cuda = args.cuda or (not args.no_cuda and have("nvcc"))
    if args.cuda and not have("nvcc"):
        raise SystemExit("--cuda was asked for, but nvcc is not on PATH; the configure "
                         "step would fail in ggml's CUDA project.  Drop --cuda to build "
                         "a CPU-only executor.")
    print(f"CUDA: {'on (nvcc found)' if cuda else 'off'}"
          + ("" if cuda or args.no_cuda else " (no nvcc on PATH)"), flush=True)
    if args.fetch and args.llama:
        raise SystemExit("pass either --fetch or --llama, not both")
    if args.fetch:
        llama = fetch_llama(FETCH_DIR, print_only=args.print_only)
    elif args.llama:
        llama = args.llama.resolve()
        if not (llama / "ggml" / "include" / "ggml.h").exists():
            raise SystemExit(f"{llama} does not look like a llama.cpp checkout "
                             f"(no ggml/include/ggml.h)")
        # A different ggml is a different measurement, so say which one this is.  Not a
        # refusal: a release snapshot has no .git, and re-measuring on another version is
        # a legitimate thing to want -- it just must not pass silently as the pinned one.
        if not args.print_only:
            commit = checkout_commit(llama)
            print(f"ggml source: {llama}"
                  + (f" at {commit}" if commit else " (no git metadata)")
                  + ("" if commit == PINNED_COMMIT
                     else f"\n  note: the pinned build is {PINNED_TAG} ({PINNED_COMMIT}); "
                          f"prices measured on this ggml are not comparable row for row"))
    else:
        raise SystemExit("pass --fetch (clone the pinned tag) or --llama PATH")

    if args.clean and args.out.exists() and not args.print_only:
        shutil.rmtree(args.out)
        print(f"removed {args.out}")

    ggml_build = llama / "build"
    if not (ggml_build / "CMakeCache.txt").exists():
        rc = cmake_ggml(llama, cuda=cuda, print_only=args.print_only)
        if rc:
            return rc

    configure = ["cmake", "-S", str(HERE), "-B", str(args.out),
                 f"-DGGML_SOURCE_DIR={llama / 'ggml'}",
                 f"-DGGML_BUILD_DIR={ggml_build}",
                 "-DMEASURE_GGML_SHARED=ON",
                 f"-DMEASURE_ENABLE_CUDA={'ON' if cuda else 'OFF'}"]
    rc = run(configure, print_only=args.print_only)
    if rc:
        return rc
    rc = run(["cmake", "--build", str(args.out), "--config", "Release",
              "--target", "ggml-ir-runner"], print_only=args.print_only)
    if rc:
        return rc

    if args.print_only:
        return 0
    runner = find_runner(args.out)
    if runner is None:
        print(f"build reported success but no ggml-ir-runner below {args.out}", file=sys.stderr)
        return 1
    blob = runner.read_bytes()
    has_cuda = b"--ssd-dir" in blob  # present in every build; kept as a size sanity check
    lib_dir = ggml_build / "bin" / "Release"
    if not lib_dir.is_dir():
        lib_dir = ggml_build / "bin"
    print(json.dumps({"runner": str(runner), "bytes": runner.stat().st_size,
                      "cuda_configured": cuda, "binary_readable": has_cuda,
                      "lib_dir": str(lib_dir)}, indent=1))
    # run.py already prefers these locations, so the variables are only needed when the
    # binary or the libraries were built somewhere else.
    print("\nrun.py finds these by default. To override:\n"
          f"    IR_RUNNER={runner}\n    IR_DLL_DIR={lib_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
