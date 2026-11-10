"""Bind the native binary a campaign measured to the toolchain that produced it.

`report_e2a` requires `native-reproducibility.json` in the campaign folder, and
the archived campaign had one that was assembled by hand. A hand-assembled file
cannot be reproduced on a second machine, so this derives the same statement from
receipts that are already written by the build and by preparation:

  * the probe's own `build-manifest.json`, which records the binary hash, the
    compiler path and its version, every bound input file, and whether the
    runtime loader's resolution was proven;
  * the campaign `manifest.json`, which records the binary hash and the hashes of
    every runtime DLL the process actually loaded;
  * the CUDA toolkit's `version.json` and the pinned llama.cpp tree's
    `build-info.cpp`, when they are present.

Fields this cannot establish are written as null with a stated reason rather
than guessed. In particular a directory named `llama.cpp-b8705` is a label, not a
verified git revision, and the archived campaign said so too.

    python -m profiling.dac_validation.native_reproducibility --out <campaign>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .e2a import read, receipt, sha, write


def find_build_manifest(binary: Path) -> Path | None:
    for parent in [binary.parent, *binary.parents]:
        candidate = parent / "build-manifest.json"
        if candidate.is_file():
            return candidate
    return None


def find_build_info(runtime_dir: Path | None) -> Path | None:
    """The pinned llama.cpp tree's generated build-info.cpp, if it is reachable."""
    for start in filter(None, (runtime_dir,)):
        for parent in [start, *start.parents]:
            candidate = parent / "build" / "common" / "build-info.cpp"
            if candidate.is_file():
                return candidate
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    manifest = read(out / "manifest.json")
    binary = Path(manifest["binary"])
    context = manifest.get("context", {})
    build_path = find_build_manifest(binary) if binary.is_file() else None
    if build_path is None:
        raise SystemExit(f"no build-manifest.json above {binary}; the binary is unbound")

    build = read(build_path)
    if build.get("binary_sha256") != context.get("binary_sha256"):
        raise SystemExit("the build manifest and the campaign manifest name different binaries")
    if Path(build.get("binary", "")).resolve() != binary.resolve():
        raise SystemExit(f"the build manifest was written for {build.get('binary')}, not {binary}")

    runtime_dirs = [Path(p) for p in manifest.get("runtime_dirs", [])]
    build_info = find_build_info(next((p for p in runtime_dirs if p.is_dir()), None))
    cuda_version = None
    if args.cuda_root is not None:
        candidate = Path(args.cuda_root) / "version.json"
        if candidate.is_file():
            cuda_version = candidate
    else:
        for directory in runtime_dirs:
            candidate = directory.parent / "version.json"
            if candidate.is_file():
                cuda_version = candidate
                break

    sources = [receipt(build_path)]
    if cuda_version is not None:
        sources.append(receipt(cuda_version))
    if build_info is not None:
        sources.append(receipt(build_info))

    result = dict(
        schema="dac-e2a-native-reproducibility-1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        campaign_manifest=receipt(out / "manifest.json"),
        native_binary=receipt(binary),
        native_binary_sha256=build["binary_sha256"],
        compiled_at_utc=build.get("linked_at_utc"),
        compiler=build.get("compiler"),
        compiler_version=build.get("compiler_version"),
        # The build manifest binds the files the compiler read; it does not state
        # the flags. These stay null rather than being copied from another build.
        configuration=None, optimization=None, runtime_library=None, NDEBUG=None,
        language=None,
        configuration_note="not recorded by the build manifest; read it from the "
                           "vcxproj/CMakeCache of this build if a report needs it",
        CUDA_SDK=(json.loads(cuda_version.read_text(encoding="utf-8")).get("cuda", {}).get("version")
                  if cuda_version is not None else None),
        CUDA_SDK_receipt=str(cuda_version.resolve()) if cuda_version is not None else None,
        llama_cpp_exact_git_revision=None,
        llama_cpp_revision_note="a directory labelled b8705 is not a verified git revision; "
                                "no build-info or git metadata was found that proves one",
        llama_build_info_receipt=str(build_info.resolve()) if build_info is not None else None,
        runtime_dlls_loaded=sorted(context.get("runtime_dlls", {})),
        runtime_loader_resolution_proven=build.get("runtime_loader_resolution_proven"),
        bound_input_files=len(build.get("bound_files", {})),
        bound_inputs_unchanged=build.get("bound_inputs_unchanged"),
        timing_boundary=build.get("primary_execution_boundary"),
        hardware_executed_by_this_read=False,
        sources=sources,
    )
    write(out / "native-reproducibility.json", result)
    print(json.dumps(dict(out=str(out), binary=result["native_binary_sha256"],
                          compiler=result["compiler_version"],
                          bound_inputs=result["bound_input_files"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
