"""Bind an independently linked NVTX diagnostic; never run hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_record(binary, source_manifest, compiler, version, cache, runtime, nvtx):
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if source.get("format_version") != "single-layer-wait-diagnosis-generated-v1":
        raise ValueError("generated source manifest schema differs")
    bound = {**source["bound_files"], **source["generated_files"]}
    if any(sha(path) != value for path, value in bound.items()):
        raise ValueError("bound input changed during compilation")
    for path in (Path(source_manifest), Path(compiler), Path(cache)):
        bound[str(path.resolve())] = sha(path)
    return dict(format_version="single-layer-wait-diagnosis-native-build-v1", returncode=0,
        binary=str(Path(binary).resolve()), binary_sha256=sha(binary), bound_files=bound, bound_inputs_unchanged=True,
        immutable_executors_sha256=source["immutable_executors_sha256"],
        allowed_generated_changes=source["allowed_changes"], compiler=str(Path(compiler).resolve()), compiler_version=version,
        runtime_dir=str(Path(runtime).resolve()), nvtx_include_dir=str(Path(nvtx).resolve()),
        nvtx_header_only=True, runtime_loader_resolution_proven=False,
        diagnostic_only=True, usable_for_parameter_fit=False, usable_for_heldout_acceptance=False,
        linked_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "source-manifest", "compiler", "cmake-cache", "runtime-dir", "nvtx-include-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compiler-version", required=True)
    args = parser.parse_args()
    record = build_record(args.binary, args.source_manifest, args.compiler, args.compiler_version,
                          args.cmake_cache, args.runtime_dir, args.nvtx_include_dir)
    (args.binary.parent / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
