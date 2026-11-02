"""Record CMake target identity and place source metadata beside its executable.

This helper never compiles or executes a probe. Its default invocation is
metadata-only. The CMake POST_BUILD hook separately declares that CMake reached
that hook after a successful target build; it is not compiler tracing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = "execution-transform-batch-probe"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_file(left, right):
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def copy_identical(source, destination):
    """Keep exact bytes; avoid truncation when source and destination alias."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    content = source.read_bytes()
    same_file = _same_file(source, destination)
    if not same_file:
        destination.write_bytes(content)
    if destination.read_bytes() != content:
        raise ValueError(f"binary-adjacent copy differs from its source: {destination}")
    return dict(source=str(source), destination=str(destination),
                sha256=hashlib.sha256(content).hexdigest(), same_file_skipped=bool(same_file))


def record_build(*, binary, generated_source, source_manifest,
                 cmake_source_dir, cmake_build_dir, config,
                 cmake_command="cmake", cmake_generator="unspecified",
                 compiler=None, compiler_id="unspecified", compiler_version="unspecified",
                 cmake_post_build=False):
    binary, generated_source, source_manifest = map(Path, (binary, generated_source, source_manifest))
    binary, generated_source, source_manifest = binary.resolve(), generated_source.resolve(), source_manifest.resolve()
    cmake_source_dir, cmake_build_dir = Path(cmake_source_dir).resolve(), Path(cmake_build_dir).resolve()
    if binary.name not in (TARGET, TARGET + ".exe") or not binary.is_file():
        raise ValueError("an existing new batch target executable is required")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "execution-transform-batch-source-0.1":
        raise ValueError("batch source manifest schema mismatch")
    source_paths = {"generator": HERE / "generate_probe.py",
                    "context": Path(manifest["context_source"]),
                    "helper": Path(manifest["helper_source"]),
                    "generated": generated_source}
    identities = {}
    for label, path in source_paths.items():
        actual = sha(path)
        if actual != manifest.get(label + "_sha256"):
            raise ValueError(f"source manifest identity differs for {label}")
        identities[label] = dict(path=str(path.resolve()), sha256=actual)
    cmake_file = cmake_source_dir / "CMakeLists.txt"
    if not cmake_file.is_file():
        raise ValueError("CMake source directory must contain CMakeLists.txt")
    identities["cmake_lists"] = dict(path=str(cmake_file), sha256=sha(cmake_file))
    identities["metadata_helper"] = dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))
    adjacent_cpp = binary.parent / "compute_batch.cpp"
    adjacent_manifest = binary.parent / "source-manifest.json"
    if any(path.resolve() != adjacent_cpp.resolve() for path in binary.parent.glob("*.cpp")):
        raise ValueError("target directory has ambiguous extra C++ sources; choose a dedicated build directory")
    copied_source = copy_identical(generated_source, adjacent_cpp)
    copied_manifest = copy_identical(source_manifest, adjacent_manifest)
    command = [str(cmake_command), "--build", str(cmake_build_dir)]
    if config:
        command.extend(["--config", str(config)])
    command.extend(["--target", TARGET])
    compiler_metadata = dict(path=None if compiler is None else str(Path(compiler).resolve()),
                             id=compiler_id, version=compiler_version,
                             sha256=sha(compiler) if compiler is not None and Path(compiler).is_file() else None)
    record = dict(schema="execution-transform-batch-build-0.1",
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        build_method="cmake_post_build" if cmake_post_build else "manual_metadata_only",
        compile_verification_performed=False,
        compile_success_reported_by="cmake_post_build_hook" if cmake_post_build else None,
        invocation_context_is_caller_declared=True,
        command=command, command_scope="CMake target reproduction command; not a captured compiler/linker command",
        cmake_source_dir=str(cmake_source_dir), cmake_build_dir=str(cmake_build_dir),
        cmake_generator=cmake_generator, configuration=str(config), compiler=compiler_metadata,
        binary_path=str(binary), binary_sha256=sha(binary), source_manifest=manifest,
        source_manifest_sha256=sha(adjacent_manifest), source_identities=identities,
        generated_source_copy=copied_source, source_manifest_copy=copied_manifest,
        generated_source_sha256=sha(adjacent_cpp),
        helper_runs_probe=False, hardware_measurement_performed=False,
        limitations=["The metadata helper neither compiles nor verifies compiler behavior.",
            "CMake POST_BUILD context is declared by its caller, not cryptographic proof of compilation.",
            "Manual invocation records file identities only and must not be presented as a verified build."])
    destination = binary.parent / "compute-batch-build.json"
    destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    return destination, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--generated-source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cmake-source-dir", type=Path, required=True)
    parser.add_argument("--cmake-build-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cmake-command", default="cmake")
    parser.add_argument("--cmake-generator", default="unspecified")
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--compiler-id", default="unspecified")
    parser.add_argument("--compiler-version", default="unspecified")
    parser.add_argument("--cmake-post-build", action="store_true")
    destination, record = record_build(**vars(parser.parse_args(argv)))
    print(json.dumps(dict(build_record=str(destination), binary_sha256=record["binary_sha256"],
                          build_method=record["build_method"], compile_verification_performed=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
