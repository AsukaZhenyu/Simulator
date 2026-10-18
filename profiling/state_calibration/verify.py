"""Read-only reconstruction of a completed state-calibration campaign.

No native process is launched. Verification establishes retained evidence
consistency, not acceptance of its accuracy gates or a causal slowdown model.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from . import model as cost_model
from . import run as runner
from .telemetry import align_intervals

SCHEMA = "state-calibration-verification-0.1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "UTC chronology requires timezone-aware timestamps")
    return parsed


def _same_clock(left: dict, right: dict) -> bool:
    return (left.get("clock_id"), left.get("frequency_hz")) == (right.get("clock_id"), right.get("frequency_hz"))


def _reference(program: dict) -> dict:
    from highlevel_accuracy.bridge import reference_ports
    return reference_ports(program)  # also validates embedded program identity


def _reference_equal(actual: dict, expected: dict) -> bool:
    import numpy as np
    if set(actual) != set(expected):
        return False
    for key, value in expected.items():
        left, right = np.asarray(actual[key], dtype=float), np.asarray(value, dtype=float)
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.allclose(left, right, atol=1e-12, rtol=1e-12):
            return False
    return True


def _source_inventory(manifest: dict) -> None:
    """Check declared build/runtime inventories without hashing assets twice."""
    sources = {str(Path(name).resolve()): digest for name, digest in manifest["sources"].items()}
    for module in (Path(__file__), Path(runner.__file__), Path(cost_model.__file__), Path(__file__).parent / "telemetry.py"):
        _require(str(module.resolve()) in sources, f"required verification/execution source not bound: {module.name}")
    binary = Path(manifest["binary"]).resolve()
    build_file = binary.parent / "build-manifest.json"
    _require(str(binary) in sources and str(build_file) in sources, "native binary/build manifest not source-bound")
    build = _read(build_file)
    _require(build.get("returncode") == 0 and build.get("bound_inputs_unchanged") is True
             and build.get("binary_sha256") == sources[str(binary)], "native successful build binding differs")
    for name, digest in build["bound_files"].items():
        _require(sources.get(str(Path(name).resolve())) == digest, "native build input missing from flat source inventory")
    for folder in (Path(name).resolve() for name in manifest["runtime_dirs"]):
        for library in folder.iterdir():
            if library.is_file() and library.suffix.lower() == ".dll":
                _require(str(library.resolve()) in sources, "runtime DLL not in flat source inventory")
        private = folder / "strict-f32-runtime.json"
        if private.is_file():
            _require(str(private) in sources, "private F32 runtime manifest not source-bound")
            runtime = _read(private)
            for field in ("original_source_sha256", "inputs_sha256", "output_sha256", "evidence_sha256"):
                for name, digest in runtime[field].items():
                    _require(sources.get(str(Path(name).resolve())) == digest, "private F32 runtime asset missing from flat source inventory")


def _native_identity(native: dict, program: dict, target: int) -> None:
    """Additional operand/copy checks shared by primary and background audits."""
    from highlevel_accuracy.bridge import BOUNDARY, RUNTIME_FLAGS
    _require(native.get("format_version") == "state-calibration-native-v1", "native format mismatch")
    _require(native.get("program_sha256") == program["program_sha256"], "native program SHA mismatch")
    _require(native.get("boundary_contract") == BOUNDARY, "native physical boundary mismatch")
    _require(native.get("actions") == program["actions"], "native full action identity mismatch")
    _require(native.get("observed_runtime_flags") == RUNTIME_FLAGS, "native runtime precision flags mismatch")
    action = program["actions"][target]
    binding = next(item for item in program["bindings"] if item["operation_id"] == action["target_id"])
    _require(native["target"]["action"] == action and native["target"]["binding"] == binding, "native target action/binding differs")
    operations = {item["id"]: item for item in program["graph"]["workload"]["operations"]}
    tensors = {item["id"]: item for item in program["graph"]["workload"]["tensors"]}
    inputs = operations[action["target_id"]]["inputs"]
    location = "vram" if action["device"] == "gpu" else "dram"
    expected_inputs = [{"tensor_id": identifier, "location": location,
                        "storage_bytes": tensors[identifier]["storage_bytes"], "exact_bytes_unchanged": True}
                       for identifier in inputs]
    _require(native.get("target_input_checks") == expected_inputs, "native target input identity/immutability differs")
    expected_copies = [{"index": index, "tensor_id": item["target_id"], "src": item["src"], "dst": item["dst"],
                        "storage_bytes": tensors[item["target_id"]]["storage_bytes"]}
                       for index, item in enumerate(program["actions"]) if item["kind"] == "COPY"]
    # Executor.validate appends copies once before and once after timed replay.
    _require(native.get("copies") == expected_copies * 2, "native copy route/bytes/two-validation inventory differs")
    _require(native.get("copy_values_equal") is True, "native copy values differ")
    _require(native.get("kernel_execution_path_proven") is False and native.get("physical_device_capacity_claimed") is False
             and native.get("cuda_private_workspace_bytes") is None, "native unsupported claims changed")


def _background(directory: Path, row: dict, program: dict, reference: dict, primary: dict) -> bool:
    job = row["job"]
    if job["condition"] == "baseline":
        _require("background" not in row, "baseline unexpectedly records a background process")
        return True
    _require(job["condition"] in ("cpu_replay_load", "gpu_replay_load"), "unknown controlled condition")
    record = row["background"]
    _require(record.get("returncode") == 0 and not row.get("background_forced_termination"), "owned background failed or was forcibly terminated")
    native = _read(directory / "background-native.json")
    ready = _read(directory / "background-ready.json")
    stop = _read(directory / "background-stop.json")
    target = runner.choose_target(program, "gpu" if job["condition"] == "gpu_replay_load" else "cpu")
    # The main command pins both executable and immutable program path.
    command = [row["command"][0], row["command"][1], "isolated", str(target), "20", "1000", "200", "bare", str(directory / "background-native.json")]
    _require(record.get("command") == command, "owned background command differs")
    _native_identity(native, program, target)
    _require(native.get("process_id") == record["pid"] and native.get("mode") == "isolated"
             and native.get("target_action_index") == target, "owned background process/mode/target differs")
    _require(native.get("execution_boundary") == "legacy_double_sync"
             and native.get("instrumentation") == "per_iteration_host_completed_work_only",
             "owned background execution boundary/instrumentation differs")
    _require(_same_clock(native["clock"], primary["clock"]) and _same_clock(ready["clock"], primary["clock"]), "owned background clock differs")
    _require(ready == record["ready"] and ready.get("process_id") == record["pid"]
             and ready.get("target_action_index") == target and ready.get("mode") == "isolated"
             and ready.get("executing_work_proven_by_ready_file") is False, "owned background readiness identity differs")
    _require(ready.get("format_version") == "state-calibration-ready-v1", "owned background readiness schema differs")
    _require(native.get("iteration_policy") == "minimum_wall_time" and native.get("requested_iterations_per_block") is None
             and native.get("graceful_stop_requested") is True, "owned background finite stop policy differs")
    _require(native.get("stop_observed_tick") is not None and native["stop_observed_tick"] >= stop["requested_tick"], "owned background stop chronology differs")
    intervals = []
    previous = -1
    for index, block in enumerate(native["blocks"]):
        _require(block["block_index"] == index and previous <= block["begin_tick"] < block["end_tick"], "owned background block chronology differs")
        _require(block["action_intervals"] == [] and block["iteration_count"] == len(block["iterations"]), "owned background bare invocation coverage differs")
        _require(block.get("target_diagnostics") == [], "owned background contains unexpected target/event diagnostics")
        _require(block["completed_work_ns"] == sum(item["elapsed_ns"] for item in block["iterations"]), "owned background completed duration differs")
        last = block["begin_tick"]
        for invocation, item in enumerate(block["iterations"]):
            _require(item["iteration"] == invocation and last <= item["begin_tick"] < item["end_tick"] <= block["end_tick"], "owned background invocation chronology differs")
            expected = (item["end_tick"] - item["begin_tick"]) * 1e9 / native["clock"]["frequency_hz"]
            _require(abs(expected - item["elapsed_ns"]) <= 1, "owned background duration/clock differs")
            intervals.append((item["begin_tick"], item["end_tick"]))
            last = item["end_tick"]
        previous = block["end_tick"]
    _require(bool(intervals), "owned background has no recorded activity")
    first, last = intervals[0][0], intervals[-1][1]
    _require(native["ready_tick"] == ready["ready_tick"] <= first, "owned background readiness chronology differs")
    coverage = all(first <= item["begin_tick"] and item["end_tick"] <= last for item in primary["blocks"])
    _require(record["active_begin_tick"] == first and record["active_end_tick"] == last
             and record["condition_interval_covers_all_blocks"] == coverage, "owned background active envelope reconstruction differs")
    runner.validate_numeric(native, reference)
    return coverage


def _summary(directory: Path, row: dict, manifest: dict, program: dict, reference: dict) -> dict:
    """Reconstruct from raw native and telemetry data without trusting summaries."""
    job, protocol = row["job"], manifest["protocol"]
    native = _read(directory / "native.json")
    _require(row.get("returncode") == 0, "measured native process did not exit successfully")
    _require(row.get("process_id") == native.get("process_id"), "launcher/native process identity differs")
    runner.verify_native_protocol(native, program, job, protocol, row["native_clock"], row["process_id"])
    expected_instrumentation = "per_iteration_and_per_action_host_completed_work" if job["instrumentation"] == "stamped" else "per_iteration_host_completed_work_only"
    _require(native.get("instrumentation") == expected_instrumentation, "native instrumentation label differs")
    target = runner.choose_target(program)
    _native_identity(native, program, target)
    _require(native["warmup_iterations"] == protocol["warmup"] and native["minimum_block_ms"] == protocol["minimum_block_ms"], "native warmup/block configuration differs")
    _require(row["launch_tick"] <= native["warmup_begin_tick"] <= native["warmup_end_tick"] <= native["blocks"][0]["begin_tick"]
             <= native["blocks"][-1]["end_tick"] <= row["exit_tick"] <= row["completed_tick"], "launcher/native chronology differs")
    numerical = runner.validate_numeric(native, reference)
    _require(_read(directory / "numerical.json") == numerical, "independent numerical evidence reconstruction differs")
    iterations = [item["elapsed_ns"] for block in native["blocks"] for item in block["iterations"]]
    targets = [item["elapsed_ns"] for block in native["blocks"] for item in block["action_intervals"] if item["action_index"] == target]
    if job["mode"] == "isolated" and not targets:
        targets = iterations
    condition_verified = _background(directory, row, program, reference, native)
    alignments = []
    if job["telemetry"]:
        metadata = _read(directory / "telemetry-metadata.json")
        records = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        _require(metadata.get("schema") == "device-state-telemetry-0.1" and metadata.get("started") is True
                 and metadata.get("stopped") is True and metadata.get("thread_error") is None, "telemetry collection lifecycle failed")
        _require(metadata["interval_ms"] == protocol["interval_ms"] and metadata["sample_count"] == len(records), "telemetry cadence/count metadata differs")
        expected_source = manifest["sources"].get(str((Path(__file__).parent / "telemetry.py").resolve()))
        _require(expected_source is not None and metadata["source_sha256"] == expected_source, "telemetry source byte binding differs")
        _require(metadata["missed_periods"] == sum(item["missed_periods"] for item in records), "telemetry missed cadence total differs")
        period = max(1, round(protocol["interval_ms"] * metadata["clock"]["frequency_hz"] / 1000))
        _require(metadata["period_ticks"] == period, "telemetry period clock conversion differs")
        deadline = metadata["start_tick"]
        for index, sample in enumerate(records):
            _require(sample["sample_index"] == index and sample["deadline_tick"] == deadline, "telemetry sample/deadline identity differs")
            begin, end = sample["query_begin_tick"], sample["query_end_tick"]
            _require(metadata["start_tick"] <= begin <= end <= metadata["end_tick"], "telemetry query chronology differs")
            _require(sample["query_duration_ticks"] == end - begin and sample["lateness_ticks"] == max(0, begin - deadline)
                     and sample["missed_periods"] == max(0, (end - deadline) // period), "telemetry query/cadence reconstruction differs")
            for provider in (value for value in sample.values() if isinstance(value, dict) and "values" in value):
                _require(begin <= provider["provider_query_begin_tick"] <= provider["provider_query_end_tick"] <= end, "telemetry provider query bounds differ")
            deadline += (sample["missed_periods"] + 1) * period
        alignments = align_intervals(native["blocks"], records, metadata["clock"], interval_ms=protocol["interval_ms"], action_clock=native["clock"])
        _require(_read(directory / "alignment.json") == alignments, "telemetry bounded alignment reconstruction differs")
    else:
        _require(not any((directory / name).exists() for name in ("telemetry.jsonl", "telemetry-metadata.json", "alignment.json")), "sampler-off control contains unexpected telemetry")
    ready = bool(alignments) and all(item["quality"]["model_ready"] for item in alignments)
    fixed = len(native["blocks"]) == protocol["blocks"] and all(block["iteration_count"] == protocol["fixed_iterations_per_block"] for block in native["blocks"])
    minimum = all(block["wall_elapsed_ns"] >= protocol["minimum_block_ms"] * 1_000_000 for block in native["blocks"])
    state = {key: statistics.median([item["observed_fields"][key]["median"] for item in alignments if key in item["observed_fields"]])
             for key in sorted({key for item in alignments for key in item["observed_fields"]})}
    return {"process_id": native["process_id"], "target_median_ns": statistics.median(targets) if targets else None,
            "iteration_median_ns": statistics.median(iterations), "target_cv": runner.cv(targets) if targets else None,
            "iteration_cv": runner.cv(iterations), "target_samples_ns": targets, "iteration_samples_ns": iterations,
            "native_numeric_passed": True, "condition_verified": condition_verified,
            "alignment_model_ready": ready and fixed and minimum, "device_state": state,
            "fixed_count_verified": fixed, "minimum_block_elapsed_verified": minimum,
            "native_sha256": _sha(directory / "native.json"),
            "target_boundary": "selected completed GGML action" if job["instrumentation"] == "stamped" else "outer isolated invocation bracket; no per-action stamping"}


def verify(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    errors, inputs, rows, measured, failed = [], {}, [], 0, 0

    def bind(path: Path, expected: str | None = None) -> None:
        path = path.resolve()
        digest = _sha(path)
        _require(expected is None or digest == expected, f"input SHA-256 mismatch: {path}")
        if str(path) in inputs:
            _require(inputs[str(path)] == digest, f"input changed while verifying: {path}")
        inputs[str(path)] = digest

    manifest, report = None, None
    try:
        bind(root / "manifest.json")
        manifest = _read(root / "manifest.json")
        _require(manifest.get("schema") == "state-calibration-campaign-0.1", "unsupported campaign schema")
        protocol = manifest["protocol"]
        _require((protocol["within_process_cv_max"], protocol["between_process_cv_max"], protocol["error_max"]) == (.15, .20, .20), "campaign acceptance gates changed")
        for name, digest in manifest["sources"].items():
            bind(Path(name), digest)
        _source_inventory(manifest)
        declared = {job["id"]: job for job in manifest["jobs"]}
        _require(len(declared) == len(manifest["jobs"]), "duplicate declared job ID")
        programs, references = {}, {}
        for case, bindings in manifest["programs"].items():
            for key, binding in bindings.items():
                bind(Path(binding["path"]), binding["sha256"])
                bind(Path(binding["reference"]), binding["reference_sha256"])
                if binding["sha256"] not in programs:
                    program = _read(Path(binding["path"]))
                    expected = _reference(program)
                    _require(program["seed"] == binding["seed"], "program numerical seed differs from manifest")
                    programs[binding["sha256"]] = program
                    references[binding["sha256"]] = expected
                _require(_reference_equal(_read(Path(binding["reference"])), references[binding["sha256"]]), "stored independent reference does not reconstruct")
        for split in ("profile", "heldout", "diagnostic"):
            path = root / f"{split}-rows.json"
            expected = {identifier for identifier, job in declared.items() if job["split"] == split}
            if split == "diagnostic" and not path.exists():
                # Recorded diagnostic executions are audited even without an
                # optional aggregate; entirely absent controls stay unknown.
                data = [_read(root / "jobs" / identifier / "execution.json") for identifier in sorted(expected)
                        if (root / "jobs" / identifier / "execution.json").exists()]
            else:
                bind(path)
                data = _read(path)
                identifiers = [row["job"]["id"] for row in data]
                _require(len(identifiers) == len(set(identifiers)) and set(identifiers) == expected, f"{split} declared job coverage differs")
            rows.extend(data)
        for row in rows:
            identifier = row["job"]["id"]
            _require(identifier in declared and row["job"] == declared[identifier], "row differs from immutable declared job")
            job = declared[identifier]
            _require(job.get("fresh_native_process") is True, "fresh native process not declared")
            from highlevel_accuracy.bridge import RUNTIME_FLAGS
            _require(row.get("requested_runtime_flags") == RUNTIME_FLAGS and row.get("flags_set_before_process_launch") is True,
                     "required precision/runtime flags were not recorded as set before process launch")
            job_root = (root / "jobs" / identifier).resolve()
            _require(job_root.is_relative_to((root / "jobs").resolve()), "job directory escapes campaign")
            bind(job_root / "execution.json")
            _require(_read(job_root / "execution.json") == row, "aggregate row differs from individual execution record")
            _require(_timestamp(row["started_utc"]) <= _timestamp(row["completed_utc"]), "job UTC chronology differs")
            artifacts = row["artifacts_sha256"]
            _require(row.get("raw_file_sha256") == artifacts, "raw artifact binding aliases differ")
            _require(set(artifacts) == {path.name for path in job_root.iterdir() if path.is_file() and path.name != "execution.json"}, "raw artifact inventory differs")
            for name, digest in artifacts.items():
                _require(len(Path(name).parts) == 1 and not Path(name).is_absolute() and name not in (".", ".."), "raw artifact path escapes job directory")
                bind(job_root / name, digest)
            binding = manifest["programs"][job["case_id"]][job["program_key"]]
            program = programs[binding["sha256"]]
            _require(row["program_sha256"] == binding["sha256"] and job["seed"] == binding["seed"], "row program bytes/numerical seed differs")
            target = runner.choose_target(program)
            command = [manifest["binary"], binding["path"], job["mode"], str(target), str(protocol["warmup"]), str(protocol["blocks"]), str(protocol["minimum_block_ms"]), job["instrumentation"], str(job_root / "native.json")]
            _require(row["command"] == command, "native command differs from declared job")
            if row.get("status") == "measured":
                _require(row["summary"] == _summary(job_root, row, manifest, program, references[binding["sha256"]]), "raw native/telemetry summary does not reconstruct")
                measured += 1
            else:
                _require(row.get("status") == "failed" and row.get("error") and row["summary"].get("native_numeric_passed") is False
                         and row["summary"].get("condition_verified") is False and row["summary"].get("alignment_model_ready") is False,
                         "failed job not explicitly retained as unknown")
                failed += 1
        for name in ("frozen-model.json", "profile-barrier.json", "report.json"):
            bind(root / name)
        frozen, barrier, report = (_read(root / name) for name in ("frozen-model.json", "profile-barrier.json", "report.json"))
        _require(barrier.get("all_sources_finished") is True and barrier["model_sha256"] == _sha(root / "frozen-model.json")
                 and barrier["profile_rows_sha256"] == _sha(root / "profile-rows.json"), "profile freeze barrier bindings differ")
        if "manifest_sha256" in barrier:
            _require(barrier["manifest_sha256"] == _sha(root / "manifest.json"), "profile barrier manifest binding differs")
        profile = [row for row in rows if row["job"]["split"] == "profile"]
        heldout = [row for row in rows if row["job"]["split"] == "heldout"]
        _require(profile and heldout, "both source and heldout process records are required")
        freeze, boundary = _timestamp(frozen["frozen_utc"]), _timestamp(barrier["frozen_utc"])
        _require(max(_timestamp(row["completed_utc"]) for row in profile) <= freeze <= boundary
                 <= min(_timestamp(row["started_utc"]) for row in heldout), "source/model/heldout freeze chronology violated")
        rebuilt = cost_model.fit_model(profile, {"manifest_sha256": _sha(root / "manifest.json"), **manifest["sources"]}, frozen["frozen_utc"])
        _require(rebuilt == frozen, "profile-only frozen model does not reconstruct")
        evaluated = cost_model.evaluate(frozen, heldout)
        _require(evaluated == report, "frozen heldout evaluation does not reconstruct")
    except Exception as error:
        errors.append({"error_type": type(error).__name__, "error": str(error)})
    changed = []
    for name, digest in inputs.items():
        try:
            if _sha(Path(name)) != digest:
                changed.append(name)
        except OSError:
            changed.append(name)
    if changed:
        errors.append({"error_type": "ChangedInputs", "error": "Bound inputs changed during verification", "paths": changed})
    report_path = root / "report.json"
    if report is None and report_path.exists():
        try:
            report = _read(report_path)
        except (OSError, ValueError) as error:
            errors.append({"error_type": type(error).__name__, "error": f"unreadable retained report: {error}"})
    acceptance = None if report is None else {name: {key: family.get(key) for key in ("cell_count", "accepted_count", "known_error_count", "unknown_error_count", "all_cells_accepted")}
                                              for name, family in report.get("families", {}).items()}
    optional = [job["id"] for job in manifest["jobs"] if job["split"] == "diagnostic"
                and job["id"] not in {row["job"]["id"] for row in rows}] if manifest else []
    return {"schema": SCHEMA, "directory": str(root), "verified": not errors, "errors": errors,
            "input_count": len(inputs), "inputs_unchanged": not changed,
            "input_bindings_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "declared_jobs": len(manifest["jobs"]) if manifest else None, "retained_jobs": len(rows),
            "measured_jobs_verified": measured, "failed_jobs_retained": failed,
            "unknown_failed_job_scopes": failed, "unknown_optional_diagnostic_jobs": optional,
            "accuracy_acceptance": acceptance,
            "accuracy_acceptance_overridden": False, "native_processes_launched": 0,
            "causal_slowdown_explanation_proven": False, "resolved_kernel_path_proven": False,
            "physical_capacity_proven": False, "cuda_private_workspace_bytes": None,
            "scope": "Read-only evidence reconstruction, source/heldout separation and fixed-gate evaluation; failure records and unknown accuracy scopes remain unchanged."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
