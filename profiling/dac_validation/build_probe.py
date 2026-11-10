"""Build `steady-costs-probe` for this machine and record what it was built from.

The E2A/V-B column measures with one binary per machine: a price measured here
says nothing about a different CPU, GPU, driver or toolkit. So the build is part
of the measurement, and this script is the whole of it --

    python -m profiling.dac_validation.build_probe \\
        --build-dir build/steady-costs-probe --llama-root <llama.cpp b8705> \\
        --ggml-build-dir <its build dir> --runtime-dir <that build>/bin \\
        --cuda-root <CUDA toolkit>

It refuses to overwrite an existing build (a binary a campaign has already cited
must never change underneath it), refuses a runtime directory that holds none of
this platform's shared libraries, compiles, and afterwards checks the
`build-manifest.json` that the CMake target writes beside the binary: the one
`prepare_portable` requires as proof that sources, event ABI and runtime were
bound at compile time.

The same generator selection as the depth column is reused
(`profiling.depth_validation.build_probe`): Visual Studio is multi-config and
writes `Release/`, Ninja and Unix Makefiles are single-config and write the
binary at the build root, where an unset `CMAKE_BUILD_TYPE` would compile an
unoptimised measurement instead of failing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from profiling import host
from profiling.dac_validation import e2a
from profiling.depth_validation import build_probe as depth_build

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "profiling" / "native" / "steady_costs"
STEM = "steady-costs-probe"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_sources() -> dict[str, str]:
    files = [HERE / "CMakeLists.txt", HERE / "probe.cpp", HERE / "components.hpp",
             HERE / "generate_probe.py", HERE / "record_build.py",
             ROOT / "profiling/native/portable/platform.hpp",
             ROOT / "profiling/native/portable/sha256.hpp",
             Path(__file__).resolve()]
    return {str(path.resolve()): sha(path) for path in files}


def runtime_libraries(directory: Path) -> list[Path]:
    """The shared libraries a probe started against this directory will load."""
    return sorted(path for path in directory.glob(host.shared_library_pattern()) if path.is_file())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True,
                        help="llama.cpp b8705 checkout the ggml libraries were built from")
    parser.add_argument("--ggml-build-dir", type=Path, required=True,
                        help="that checkout's build directory (holds ggml-*.lib and "
                             "ggml/src/<config> on Windows, libggml-*.so on Linux)")
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True,
                        help="a directory the probe loads GGML from at launch; pass the "
                             "same list the campaign will pass. Repeatable.")
    parser.add_argument("--cuda-root", type=Path,
                        help="CUDA toolkit (required when this host's backend is cuda; a "
                             "metal build needs none)")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--print", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    backend = host.backend()
    if backend is None:
        raise SystemExit("this host has no accelerator ggml can compute on, so the e2a "
                         "column has no GPU placement to compare with the CPU one; see "
                         "`python -m tools.device_check`")
    if backend == "cuda" and args.cuda_root is None:
        raise SystemExit("a cuda build needs --cuda-root: the probe links the toolkit "
                         "and observes CUDA events")
    build = args.build_dir.resolve()
    flags, multi_config = depth_build.generator()
    binary = depth_build.binary_path(build, args.config, multi_config, STEM)
    if binary.exists():
        raise SystemExit(f"{binary} already exists; the campaign it belongs to cites its "
                         "hash, so build into a new directory instead")
    # A runtime directory that holds no library of this platform is the classic
    # silent failure: the build succeeds and every measurement process exits with
    # an unresolved import before it measures anything. Refuse it here instead.
    runtime_dirs = [Path(directory).resolve() for directory in args.runtime_dir]
    empty = [str(directory) for directory in runtime_dirs
             if not runtime_libraries(directory)]
    if empty:
        raise SystemExit("these runtime directories hold no %s library: %s"
                         % (host.shared_library_pattern(), ", ".join(empty)))
    # All three of the probe's components, not just one: a half-built runtime
    # links here and fails at launch, on the machine that was supposed to measure.
    absent = e2a.missing_components(runtime_dirs)
    if absent:
        raise SystemExit("these runtime directories hold no %s: %s"
                         % (", ".join(absent), ", ".join(str(d) for d in runtime_dirs)))

    configure = ["cmake", "-S", str(HERE), "-B", str(build), *flags,
                 f"-DPROBE_BACKEND={backend}",
                 f"-DLLAMA_ROOT={args.llama_root.resolve()}",
                 f"-DGGML_BUILD_DIR={args.ggml_build_dir.resolve()}",
                 f"-DRUNTIME_DIR={runtime_dirs[0]}"]
    if args.cuda_root is not None:
        configure.append(f"-DCUDA_ROOT={args.cuda_root.resolve()}")
    if not multi_config:
        configure.append(f"-DCMAKE_BUILD_TYPE={args.config}")
    compile_ = ["cmake", "--build", str(build), "--target", STEM]
    if multi_config:
        compile_ += ["--config", args.config]
    if args.print:
        print(" ".join(configure))
        print(" ".join(compile_))
        return 0

    bound = bound_sources()
    started = datetime.now(timezone.utc).isoformat()
    for command in (configure, compile_):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-4000:])
            print(done.stderr[-4000:])
            raise SystemExit(f"build failed: {' '.join(command)}")
    if not binary.is_file():
        raise SystemExit(f"the build reported success but {binary} is missing")
    changed = [path for path, digest in bound.items() if sha(Path(path)) != digest]
    if changed:
        raise SystemExit("a bound input changed during the build: " + ", ".join(changed))
    manifest = binary.parent / "build-manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} is missing; the probe's POST_BUILD receipt did not run, "
                         "so nothing proves what this binary was built from")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    if not recorded.get("bound_inputs_unchanged"):
        raise SystemExit("the probe's own build receipt reports a changed input")
    receipt = dict(format_version="steady-costs-build-driver-1", started_utc=started,
                   finished_utc=datetime.now(timezone.utc).isoformat(), returncode=0,
                   binary=str(binary), binary_sha256=sha(binary), bound_files=bound,
                   build_manifest=str(manifest), build_manifest_sha256=sha(manifest),
                   commands=[configure, compile_], config=args.config,
                   generator=flags[1], multi_config=multi_config,
                   runtime_dirs=[str(directory) for directory in runtime_dirs],
                   runtime_libraries={str(directory): [path.name for path in
                                                       runtime_libraries(directory)]
                                      for directory in runtime_dirs},
                   toolchain=dict(llama_root=str(args.llama_root.resolve()),
                                  ggml_build_dir=str(args.ggml_build_dir.resolve()),
                                  backend=backend,
                                  cuda_root=(str(args.cuda_root.resolve())
                                             if args.cuda_root is not None else None),
                                  device_memory=host.device_memory()))
    path = binary.parent / "build-driver-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(binary=str(binary), build_manifest=str(manifest),
                          driver_receipt=str(path)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
