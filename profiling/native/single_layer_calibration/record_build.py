"""Bind a successful independent build to all explicitly supplied input bytes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(binary, source_manifest, compiler, compiler_version, cmake_cache, runtime_dir):
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if source.get("format_version") != "single-layer-generated-source-v1":
        raise ValueError("generated source manifest schema differs")
    bound = dict(source["bound_files"])
    bound[source["generated_file"]] = source["generated_sha256"]
    unchanged = all(sha(path) == digest for path, digest in bound.items())
    if not unchanged:
        raise ValueError("bound build input changed during compile")
    for path in (Path(source_manifest), Path(compiler), Path(cmake_cache)):
        bound[str(path.resolve())] = sha(path)
    return dict(format_version="single-layer-calibration-native-build-v1", returncode=0,
                binary=str(Path(binary).resolve()), binary_sha256=sha(binary), bound_files=bound,
                bound_inputs_unchanged=True, immutable_helper_sha256=source["immutable_helper_sha256"],
                compiler=str(Path(compiler).resolve()), compiler_version=compiler_version,
                compiler_flags_scope="CMakeCache plus CMake generated compile/link rules; no claim to historical DLL compiler flags",
                runtime_dir=str(Path(runtime_dir).resolve()), runtime_loader_resolution_proven=False,
                supported_execution_boundary="legacy_double_sync", supported_pool_poll=["default", "0"],
                linked_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "source-manifest", "compiler", "cmake-cache", "runtime-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compiler-version", required=True)
    args = parser.parse_args()
    record = build_record(args.binary, args.source_manifest, args.compiler, args.compiler_version,
                          args.cmake_cache, args.runtime_dir)
    (args.binary.parent / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
