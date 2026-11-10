#!/usr/bin/env python3
"""Build the edge executor into a *separate* CMake tree: `build-ir-edge`.

Why this script exists rather than a hand-typed cmake line: `build-ir/Release/
ggml-ir-runner.exe` is the binary the pinned v3 campaign is measuring with.  Rebuilding it
would silently change the instrument between tranches, so the v4 executor changes are
built into their own tree and selected at run time through `IR_RUNNER` (which `run.py`
honours).  This script never touches `build-ir`.

    python build_ir_edge.py            # configure (if needed) + build Release
    python build_ir_edge.py --clean    # delete build-ir-edge first
    python build_ir_edge.py --print    # only print the commands it would run

The GGML it links against is the existing b8705 build, taken from `build-ir`'s own cache
so the two trees cannot drift onto different ggml libraries -- a different ggml would be a
different measurement, not a different binary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = HERE / "build-ir"
EDGE = HERE / "build-ir-edge"
RUNNER = EDGE / "Release" / "ggml-ir-runner.exe"


def cached(name: str) -> str | None:
    """One value out of the pinned tree's cache, so the two trees share a ggml."""
    cache = PINNED / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{name}:"):
            return line.split("=", 1)[1]
    return None


def commands() -> list[list[str]]:
    source = cached("GGML_SOURCE_DIR")
    build = cached("GGML_BUILD_DIR")
    if not source or not build:
        raise SystemExit(f"{PINNED}/CMakeCache.txt does not pin GGML_SOURCE_DIR / "
                         f"GGML_BUILD_DIR; build the pinned tree first so the edge tree can "
                         f"reuse its ggml instead of guessing")
    configure = ["cmake", "-S", str(HERE), "-B", str(EDGE), "-G", "Visual Studio 17 2022",
                 "-A", "x64", f"-DGGML_SOURCE_DIR={source}", f"-DGGML_BUILD_DIR={build}",
                 "-DMEASURE_GGML_SHARED=ON", "-DMEASURE_ENABLE_CUDA=ON"]
    build_call = ["cmake", "--build", str(EDGE), "--config", "Release",
                  "--target", "ggml-ir-runner"]
    return [configure, build_call]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args(argv)

    if args.clean and EDGE.exists():
        shutil.rmtree(EDGE)
        print(f"removed {EDGE}")
    plan = commands()
    for call in plan:
        print(" ".join(call))
        if args.print_only:
            continue
        done = subprocess.run(call)
        if done.returncode != 0:
            print(f"FAILED: {' '.join(call)}", file=sys.stderr)
            return done.returncode
    if args.print_only:
        return 0
    if not RUNNER.exists():
        print(f"build reported success but {RUNNER} is missing", file=sys.stderr)
        return 1
    blob = RUNNER.read_bytes()
    missing = [flag for flag in (b"--ssd-dir",) if flag not in blob]
    if missing:
        print(f"{RUNNER} does not contain {missing} -- it is not the edge build",
              file=sys.stderr)
        return 1
    print(json.dumps({"runner": str(RUNNER), "bytes": RUNNER.stat().st_size}, indent=1))
    print("select it with:  set IR_RUNNER=" + str(RUNNER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
