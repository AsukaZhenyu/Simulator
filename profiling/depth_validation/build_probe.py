"""Build `depth-probe` for this machine and record what it was built from.

The depth campaign measures with one binary per machine: a table measured here
says nothing about a different CPU, GPU or driver. So the build itself is part of
the measurement, and this script is the whole of it --

    python -m profiling.depth_validation.build_probe \\
        --build-dir build/depth-probe --llama-root <llama.cpp b8705> \\
        --ggml-build-dir <its build dir> --cuda-root <CUDA toolkit> \\
        --cupti-root <CUDA>/extras/CUPTI

It refuses to overwrite an existing build (a binary an experiment has already
cited must never change underneath it), verifies the five generated Executor
fragments against their recorded hashes before compiling, and afterwards writes
`build-manifest.json` next to the binary with the binary's hash, the compiler,
the fragment hashes and the hashes of every bound source.

The fragments matter more than they look: this probe needs the historical
highlevel executor, not today's, because only it carries the binary
initial-asset transport the depth programs are written against. See
`generate_sources.py` for why, and `--fragments` for choosing a directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import generate_sources
from profiling import host

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "native"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_sources(fragments: Path) -> dict[str, str]:
    files = [HERE / "CMakeLists.txt", HERE / "depth_probe.cpp", HERE / "initial_assets.hpp",
             ROOT / "profiling/native/steady_costs/probe.cpp",
             ROOT / "profiling/native/steady_costs/components.hpp",
             ROOT / "profiling/native/physical_stages/cupti_capture.hpp",
             ROOT / "profiling/native/portable/platform.hpp",
             ROOT / "profiling/native/portable/sha256.hpp",
             Path(__file__).resolve(), Path(generate_sources.__file__).resolve()]
    files += sorted(fragments.glob("*.inc"))
    return {str(path.resolve()): sha(path) for path in files}


def generator() -> tuple[list[str], bool]:
    """The configure flags this platform needs, and whether output is per-config.

    Visual Studio is a multi-config generator: it writes `Release/depth-probe.exe`
    and takes `--config` at build time. Ninja and Unix Makefiles are
    single-config: they write `depth-probe` at the build root and read the build
    type at configure time -- leaving `CMAKE_BUILD_TYPE` unset there compiles a
    release measurement with no optimisation at all, which is a silently wrong
    instrument rather than a failed build.

    It is shared with `profiling.dac_validation.build_probe`, because both
    native probes are one CMake project per column and a second device should
    not have to learn a second set of flags.
    """
    if host.is_windows():
        return ["-G", "Visual Studio 17 2022", "-A", "x64"], True
    if shutil.which("ninja"):
        return ["-G", "Ninja"], False
    return ["-G", "Unix Makefiles"], False


def binary_path(build: Path, config: str, multi_config: bool, stem: str = "depth-probe") -> Path:
    name = host.executable_name(stem)
    return build / config / name if multi_config else build / name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True,
                        help="llama.cpp b8705 checkout the ggml libraries were built from")
    parser.add_argument("--ggml-build-dir", type=Path, required=True,
                        help="that checkout's build directory (holds ggml-*.lib on Windows, "
                             "libggml-*.so on Linux)")
    parser.add_argument("--cuda-root", type=Path,
                        help="CUDA toolkit (required when this host's backend is cuda; a "
                             "metal build takes the CUDA-vocabulary headers from "
                             "profiling/native/backend/metal and needs none)")
    parser.add_argument("--cupti-root", type=Path,
                        help="<CUDA>/extras/CUPTI, the one holding include/cupti.h (cuda only)")
    parser.add_argument("--fragments", type=Path,
                        help="generated Executor fragments; defaults to the shipped set")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--print", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    backend = host.backend()
    if backend is None:
        raise SystemExit("this host has no accelerator ggml can compute on, so the dense "
                         "column has no GPU placement to compare with the CPU one; see "
                         "`python -m tools.device_check`")
    if backend == "cuda" and (args.cuda_root is None or args.cupti_root is None):
        raise SystemExit("a cuda build needs --cuda-root and --cupti-root: the probe links "
                         "the toolkit and CUPTI (see `python -m tools.device_check`)")
    fragments = (args.fragments or generate_sources.GENERATED).resolve()
    if generate_sources.check() != 0 and fragments == generate_sources.GENERATED.resolve():
        raise SystemExit("the shipped Executor fragments do not hash to their record; "
                         "refusing to build an instrument that is not the published one")
    build = args.build_dir.resolve()
    flags, multi_config = generator()
    binary = binary_path(build, args.config, multi_config)
    if binary.exists():
        raise SystemExit(f"{binary} already exists; the campaign it belongs to cites its "
                         "hash, so build into a new directory instead")

    configure = ["cmake", "-S", str(HERE), "-B", str(build), *flags,
                 f"-DPROBE_BACKEND={backend}",
                 f"-DLLAMA_ROOT={args.llama_root.resolve()}",
                 f"-DGGML_BUILD_DIR={args.ggml_build_dir.resolve()}",
                 f"-DSTEADY_GENERATED_DIR={fragments}"]
    if args.cuda_root is not None:
        configure.append(f"-DCUDA_ROOT={args.cuda_root.resolve()}")
    if args.cupti_root is not None:
        configure.append(f"-DCUPTI_ROOT={args.cupti_root.resolve()}")
    if not multi_config:
        configure.append(f"-DCMAKE_BUILD_TYPE={args.config}")
    compile_ = ["cmake", "--build", str(build), "--target", "depth-probe"]
    if multi_config:
        compile_ += ["--config", args.config]
    if args.print:
        print(" ".join(configure))
        print(" ".join(compile_))
        return 0

    bound = bound_sources(fragments)
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
    record = dict(format_version="depth-native-build-1", started_utc=started,
                  finished_utc=datetime.now(timezone.utc).isoformat(), returncode=0,
                  binary=str(binary), binary_sha256=sha(binary), bound_files=bound,
                  bound_inputs_unchanged=True, fragments_dir=str(fragments),
                  fragment_hashes={p.name: sha(p) for p in sorted(fragments.glob("*.inc"))},
                  commands=[configure, compile_], config=args.config,
                  generator=flags[1], multi_config=multi_config,
                  toolchain=dict(llama_root=str(args.llama_root.resolve()),
                                 ggml_build_dir=str(args.ggml_build_dir.resolve()),
                                 backend=backend,
                                 cuda_root=(str(args.cuda_root.resolve())
                                            if args.cuda_root is not None else None),
                                 cupti_root=(str(args.cupti_root.resolve())
                                             if args.cupti_root is not None else None),
                                 device_memory=host.device_memory()),
                  execution_boundary="unchanged completed Executor.action; reset, prime, "
                                     "verification and serialization excluded",
                  hardware_executed=False)
    path = binary.parent / "build-manifest.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(binary=str(binary), build_manifest=str(path),
                          fragments=str(fragments)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
