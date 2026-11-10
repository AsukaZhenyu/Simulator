"""Build the strict-F32 CUDA runtime from the pinned tree on a POSIX host.

`build_f32_runtime.py` beside this file is the Windows route: it recompiles one
patched translation unit and relinks the object files the supplied build already
had, so the runtime keeps every byte it can. None of that machinery exists here
-- there is no MSBuild project to read the flags from, no `.lib` to relink
against, no link record to prove the object set -- so this route compiles the
whole tree instead, with the same one-line router patch:

    python -m profiling.native.highlevel_accuracy.build_f32_runtime_linux \\
        --source <llama.cpp b8705 tree> --output <new directory> \\
        --cuda-root <CUDA toolkit>

What both routes claim is the same: `GGML_OP_MUL_MAT` whose op parameters carry
`GGML_PREC_F32` takes the F32 path instead of the TF32 MMA instructions the
custom mat-mul kernel would otherwise select for an F32 src0/src1/dst. What only
the Windows route claims is that the rest of the binary is the one that was
measured. This one says plainly that the rest is a clean build of the pinned
tree, and records the compiler, the toolkit, the architecture and every bound
source, so a reader can tell the two apart from the manifest alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from profiling import host
from profiling.native.highlevel_accuracy.build_f32_runtime import (
    METAL_STRICT_F32_KERNELS,
    PINNED_ROUTER_SHA256,
    patch_router,
)
from profiling.real_moe_validation import linux_build

ROUTER = "src/ggml-cuda/ggml-cuda.cu"
TARGETS = ("llama", "ggml", "ggml-base", "ggml-cpu", "ggml-cuda")


def ggml_source(source: Path) -> Path:
    """The ggml directory whose router is patched.

    `--source` is the llama.cpp tree, not its `ggml/` directory: this route
    builds the `llama` target as well, because the probe links against it. A
    caller who passes the ggml directory gets told which one to pass instead.
    """
    candidate = source / "ggml"
    if (candidate / ROUTER).is_file():
        return candidate
    if (source / ROUTER).is_file():
        raise ValueError("--source is the ggml directory; pass the llama.cpp tree "
                         "that holds it (" + str(source.parent) + "), because this "
                         "route builds the llama target too")
    raise ValueError("neither " + str(source) + " nor " + str(source / "ggml")
                     + " holds " + ROUTER + ": --source must be a llama.cpp b8705 tree")


def digest(path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def ignore(directory: str, names: list[str]) -> list[str]:
    """What the private copy leaves behind, by the rule the model column proved.

    `linux_build.copy_ignore` is the same decision about the same tree: git
    objects and build outputs are never inputs, and `models/` is the vocabulary
    fixtures at the top of a checkout but the architecture sources under `src/`,
    so it is skipped only when it holds no C++ the build would compile. One rule
    is the point: a second, simpler one here dropped `src/models/afmoe.cpp`, and
    CMake stopped at generate with "No SOURCES given to target: llama".
    """
    return linux_build.copy_ignore(directory, names)


def compiler_identity(compiler: str) -> str:
    try:
        done = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "not asked"
    return (done.stdout.splitlines() or [compiler])[0].strip()


def run(command: list[str], output: Path, name: str, environment: dict) -> dict:
    """Run one build step and write what it printed, whether or not it passed."""
    started = datetime.now(timezone.utc).isoformat()
    try:
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=5400, env=environment)
        record = dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        record = dict(returncode=None, stdout="", stderr=repr(error))
    record.update(command=command, started_at=started,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write(output / name, record)
    if record["returncode"] != 0:
        raise RuntimeError("the strict-F32 runtime build failed; preserved "
                           f"{output / name}: {record['stderr'][-3000:]}")
    return record


def build(source: Path, output: Path, cuda_root: Path | None, jobs: int = os.cpu_count() or 1,
          extra_cmake: list[str] | None = None, compiler: str = "c++") -> dict:
    if host.is_windows():
        raise ValueError("this route is POSIX only; on Windows use "
                         "profiling.native.highlevel_accuracy.build_f32_runtime")
    if host.is_macos():
        raise ValueError("this route patches ggml-cuda.cu, and a Mac has no CUDA tree to "
                         "patch: " + METAL_STRICT_F32_KERNELS + ". Making the batched F32 "
                         "matmul exact on Metal is a new fp32-fragment kernel and a "
                         "dispatch switch, not this patch")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a NEW directory: " + str(output))
    ggml = ggml_source(source)
    router = ggml / ROUTER
    # Fail before the output directory exists: a refused patch must not leave a
    # half-runtime that a later run could mistake for a finished one.
    patched = patch_router(router.read_bytes())

    tree = output / "source"
    shutil.copytree(source, tree, ignore=ignore)
    private_router = tree / ggml.relative_to(source) / ROUTER
    private_router.write_bytes(patched)
    (output / "strict-f32-router.patch").write_text(
        "".join(difflib.unified_diff(router.read_text(encoding="utf-8").splitlines(True),
                                     patched.decode("utf-8").splitlines(True),
                                     fromfile="pinned/ggml-cuda.cu",
                                     tofile="private/ggml-cuda.cu")),
        encoding="utf-8", newline="\n")

    environment = linux_build.build_environment(cuda_root)
    arguments, architecture_source = linux_build.resolved_architectures(list(extra_cmake or []))
    build_dir = output / "build"
    configure = ["cmake", "-S", str(tree), "-B", str(build_dir),
                 "-DCMAKE_BUILD_TYPE=Release",
                 "-DGGML_CUDA=ON",
                 "-DBUILD_SHARED_LIBS=ON",
                 "-DLLAMA_CURL=OFF",
                 "-DLLAMA_BUILD_TESTS=OFF",
                 "-DLLAMA_BUILD_EXAMPLES=OFF",
                 "-DLLAMA_BUILD_SERVER=OFF"] + arguments
    run(configure, output, "cmake-configure.json", environment)
    run(["cmake", "--build", str(build_dir), "--config", "Release",
         "--target", *TARGETS, "-j", str(jobs)], output, "cmake-build.json", environment)

    libraries = build_dir / "bin"
    pattern = host.shared_library_pattern()
    found = sorted(path for path in libraries.glob(pattern))
    if not found:
        raise RuntimeError("the build wrote no " + pattern + " libraries to " + str(libraries))
    names = {path.name for path in found}
    for required in ("libllama", "libggml-base", "libggml-cuda"):
        if not any(name.startswith(required) for name in names):
            raise RuntimeError(f"the build is incomplete: no {required}* among {sorted(names)}")
    receipt = dict(
        format_version="ggml-strict-f32-runtime-posix-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        build_scope=("a complete CMake build of the private tree with ONE patched router "
                     "translation unit; unlike the Windows route, the other libraries are "
                     "built here rather than preserved from a supplied build"),
        precision_scope=("regular GGML_OP_MUL_MAT with requested GGML_PREC_F32 bypasses "
                         "custom MMF TF32; MUL_MAT_ID unchanged"),
        pinned_source=str(source),
        ggml_source=str(ggml),
        original_router_sha256=PINNED_ROUTER_SHA256,
        patched_router_sha256=digest(private_router),
        patch_hunks=1,
        compiler=compiler,
        compiler_version=compiler_identity(compiler),
        cuda_root=str(cuda_root) if cuda_root is not None else "not named",
        cuda_architecture_source=architecture_source,
        cmake_arguments=arguments,
        jobs=jobs,
        libraries={path.name: digest(path) for path in found},
        evidence={name: digest(output / name)
                  for name in ("strict-f32-router.patch",
                               "cmake-configure.json", "cmake-build.json")},
    )
    write(libraries / "strict-f32-runtime.json", receipt)
    return dict(runtime_dir=str(libraries),
                manifest=str(libraries / "strict-f32-runtime.json"),
                manifest_sha256=digest(libraries / "strict-f32-runtime.json"),
                libraries=sorted(receipt["libraries"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="llama.cpp b8705 tree the router is pinned to")
    parser.add_argument("--output", required=True, type=Path,
                        help="a new directory; the runtime lands in its build/bin")
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cmake-argument", action="append", default=[])
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.output, args.cuda_root, args.jobs,
                           args.cmake_argument, args.compiler)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
