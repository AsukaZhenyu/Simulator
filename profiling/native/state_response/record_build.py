"""Bind compiled state-response inputs without executing the measurement binary."""
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
    if source.get("format_version") != "state-response-generated-v1":
        raise ValueError("generated source manifest schema differs")
    bound = {**source["bound_files"], **source["generated_files"]}
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError("bound build input changed during compile")
    for path in (Path(source_manifest), Path(compiler), Path(cache)):
        bound[str(path.resolve())] = sha(path)
    return dict(format_version="state-response-native-build-v1", returncode=0,
        binary=str(Path(binary).resolve()), binary_sha256=sha(binary), bound_files=bound,
        bound_inputs_unchanged=True, immutable_executors_sha256=source["immutable_executors_sha256"],
        event_abi_sha256=source["event_abi_sha256"], compiler=str(Path(compiler).resolve()),
        compiler_version=compiler_version, runtime_dir=str(Path(runtime_dir).resolve()),
        runtime_loader_resolution_proven=False, primary_execution_boundary="legacy_double_sync",
        diagnostic_execution_boundary="explicit_async_double_sync_diagnostic",
        supported_roles=["main", "cpu-worker", "gpu-worker"],
        supported_instrumentation=["bare", "stamped", "diagnostic", "diagnostic-control", "thread-diagnostic"],
        thread_diagnostic_execution_boundary="unchanged legacy Executor.action with CPU query brackets; diagnostic only",
        main_thread_id_provider="Windows GetCurrentThreadId; unknown on other platforms",
        cuda_schedule_observation="actual cudaGetDevice/cudaGetDeviceFlags after warmup and after STOP, outside DUT timers",
        cuda_schedule_override_environment="SIMULATOR_CUDA_SCHEDULE",
        cuda_schedule_override_values=["auto", "spin", "yield", "blocking"],
        cuda_schedule_unset_behavior="no override; actual flags queried, never inferred",
        cuda_schedule_requested_mode_must_be_observed=True,
        cuda_auto_wait_behavior_inferred=False,
        hardware_executed=False, linked_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "source-manifest", "compiler", "cmake-cache", "runtime-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compiler-version", required=True)
    args = parser.parse_args()
    result = build_record(args.binary, args.source_manifest, args.compiler, args.compiler_version, args.cmake_cache, args.runtime_dir)
    (args.binary.parent / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
