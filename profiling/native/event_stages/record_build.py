"""Record the explicit linked native inputs without executing hardware."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_record(binary, source_manifest, compiler, compiler_version, cache, runtime_dir, cupti_root):
    parent = Path(__file__).resolve().parents[1] / "state_response" / "record_build.py"
    spec = importlib.util.spec_from_file_location("physical_stage_historical_build_record", parent)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = module.build_record(binary, source_manifest, compiler, compiler_version, cache, runtime_dir)
    record.update(format_version="event-stage-native-build-1",
        primary_execution_boundary="unchanged historical Executor.action; event GO/STOP parking; optional raw cycle diagnostics",
        supported_modes=["full", "target"], supported_instrumentation=["bare", "cupti"],
        native_clock="windows_qpc_floor_ns; absolute_qpc_zero",
        synchronization_transport="parent-owned manual-reset Windows GO/STOP events; files are receipts",
        required_environment=["SIMULATOR_PHYSICAL_GO_EVENT", "SIMULATOR_PHYSICAL_STOP_EVENT"],
        optional_cycle_environment="SIMULATOR_PHYSICAL_CYCLES=0|1; absent disables",
        completed_reference_periodic_wakeup=False,
        cupti_timestamp_callback_before_activity_enable=True,
        cupti_root=str(Path(cupti_root).resolve()),
        available_resource_rate_inferred=False, hardware_accuracy_qualified=False)
    record.pop("supported_roles", None)
    record.pop("diagnostic_execution_boundary", None)
    record["bound_files"][str(parent)] = sha(parent)
    record["bound_files"][str(Path(__file__).resolve())] = sha(__file__)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "source-manifest", "compiler", "cmake-cache", "runtime-dir", "cupti-root"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--compiler-version", required=True)
    args = parser.parse_args()
    value = build_record(args.binary, args.source_manifest, args.compiler, args.compiler_version,
                         args.cmake_cache, args.runtime_dir, args.cupti_root)
    with (args.binary.parent / "build-manifest.json").open("w", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
