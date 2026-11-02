"""Record a successfully linked diagnostic build without running the probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_record(binary, source_manifest, compiler, compiler_version, cache, runtime_dir):
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if source.get("format_version") != "single-layer-context-pair-generated-v1":
        raise ValueError("generated source manifest schema differs")
    bound = {**source["bound_files"], **source["generated_files"]}
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError("bound build input changed during compile")
    for path in (Path(source_manifest), Path(compiler), Path(cache)):
        bound[str(path.resolve())] = sha(path)
    return dict(format_version="single-layer-context-pair-native-build-v2", returncode=0,
        binary=str(Path(binary).resolve()), binary_sha256=sha(binary), bound_files=bound, bound_inputs_unchanged=True,
        immutable_executors_sha256=source["immutable_executors_sha256"],
        compiler=str(Path(compiler).resolve()), compiler_version=compiler_version,
        runtime_dir=str(Path(runtime_dir).resolve()), runtime_loader_resolution_proven=False,
        diagnostic_only=True, usable_for_parameter_fit=False, usable_for_heldout_acceptance=False,
        supported_pair_kinds=["context_vs_isolated", "instrumentation"],
        linked_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "source-manifest", "compiler", "cmake-cache", "runtime-dir"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--compiler-version", required=True)
    args = parser.parse_args()
    record = build_record(args.binary, args.source_manifest, args.compiler, args.compiler_version, args.cmake_cache, args.runtime_dir)
    (args.binary.parent / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
