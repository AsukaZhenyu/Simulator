"""Independently reconstruct a saved high-level accuracy experiment, offline.

No native process, hardware lock, measurement, source edit or artifact overwrite
is performed. Verification is STRICTLY against current bytes at the recorded
source/binary/library paths. This does not implement archived-software replay:
an unavailable or changed source/runtime is a rejection, not historical proof.
Recorded timestamps and distinct execution records establish the declared
chronology/process protocol; they are not externally signed attestations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]


class VerificationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise VerificationError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_text(text, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def constant(value):
        raise VerificationError(f"non-finite JSON in {label}: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read(path):
    path = Path(path)
    return json_text(path.read_text(encoding="utf-8"), str(path))


def same(actual, expected, label):
    require(canonical(actual) == canonical(expected), f"{label} differs from independent reconstruction")


def seal(document, key, label):
    unsigned = deepcopy(document)
    claimed = unsigned.pop(key, None)
    require(isinstance(claimed, str) and claimed == digest(unsigned), f"{label} SHA256 changed")


def timestamp(value, label):
    require(isinstance(value, str), f"missing {label} timestamp")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, f"{label} timestamp must be timezone aware")
    return parsed


def path_same(actual, expected, label):
    require(isinstance(actual, str) and Path(actual).resolve() == Path(expected).resolve(), f"{label} path differs")


def sha_files(inventory, label):
    require(isinstance(inventory, dict) and inventory, f"missing {label} inventory")
    for name, claimed in inventory.items():
        require(isinstance(name, str) and isinstance(claimed, str) and re.fullmatch("[0-9a-f]{64}", claimed), f"invalid {label} hash entry")
        path = Path(name)
        require(path.is_file(), f"{label} current file unavailable: {path}")
        require(file_sha(path) == claimed, f"{label} current bytes changed: {path}")


def verify_software(record):
    """Validate immutable bytes BEFORE importing any frozen bridge helper."""
    sha_files(record["source_sha256"], "bridge source")
    sha_files(record["runtime_sha256"], "native/runtime")
    build = record["successful_build_record"]
    require(build.get("format_version") == "highlevel-native-successful-build-v1", "unknown successful-link build record")
    sha_files(build["source_sha256"], "compiled native source")
    sha_files(build["linked_library_sha256"], "linked import/static library")
    executable = Path(build["binary"]).resolve()
    require(file_sha(executable) == record["binary_sha256"] == build["binary_sha256"], "native executable/build linkage differs")
    require(record["runtime_sha256"].get(str(executable)) == record["binary_sha256"], "executable absent from runtime snapshot")
    build_file = executable.parent / "build-manifest.json"
    require(file_sha(build_file) == record["successful_build_record_sha256"], "saved successful-link manifest bytes changed")
    same(read(build_file), build, "successful-link manifest")
    for directory in record["runtime_search_dirs"]:
        require(Path(directory).is_dir(), f"current runtime search directory unavailable: {directory}")
    # Rebuild the exact inventory, including newly added/removed source or DLL
    # files and the simulator's Python runtime fingerprint. No hardware runs.
    for directory in (ROOT / "modeling", ROOT / "mapping", ROOT / "profiling"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from highlevel_accuracy.run import software
    same(software(executable, [Path(p) for p in record["runtime_search_dirs"]]), record, "full current software/runtime inventory")
    return executable


def raw_evidence(directory):
    """Use canonical local raw paths; never follow evidence outside its process."""
    directory = Path(directory).resolve()
    native_file, execution_file = directory / "native.json", directory / "execution.json"
    evidence = dict(raw_native_file=str(native_file), raw_native_sha256=file_sha(native_file),
                    execution_file=str(execution_file), execution_sha256=file_sha(execution_file))
    return read(native_file), read(execution_file), evidence


def verify_execution(execution, native, *, executable, program_file, native_file, mode, samples=None):
    require(execution.get("returncode") == 0 and type(execution.get("returncode")) is int,
            "successful native result has failed execution status")
    require(execution.get("timed_out") is False and not execution.get("launch_failed"), "successful native result came from timeout/launch failure")
    require(isinstance(execution.get("stdout"), str) and isinstance(execution.get("stderr"), str), "raw stdout/stderr missing")
    require(execution.get("pid_scope") == "one_fresh_native_process", "independent process execution contract missing")
    from highlevel_accuracy.bridge import RUNTIME_FLAGS
    same(execution.get("requested_runtime_flags"), RUNTIME_FLAGS, "requested runtime flags")
    same(native.get("observed_runtime_flags"), RUNTIME_FLAGS, "observed runtime flags")
    require(native.get("mode") == mode, "native profile/heldout mode changed")
    require(type(native.get("samples")) is int and native["samples"] >= 3, "each process requires >=3 samples")
    require(type(native.get("warmup")) is int and native["warmup"] >= 0, "warmup count missing")
    if samples is not None:
        require(native["samples"] == samples, "source samples differ from frozen campaign protocol")
    command = execution.get("command")
    require(isinstance(command, list) and len(command) == 6, "native command arity differs")
    path_same(command[0], executable, "native executable")
    path_same(command[1], program_file, "native program")
    path_same(command[5], native_file, "native raw output")
    same(command[2:5], [mode, str(native["warmup"]), str(native["samples"])], "native command protocol")
    stdout = json_text(execution["stdout"], "raw native stdout")
    same(stdout, dict(status="ok", mode=mode, ports=len(native["ports"]), program_sha256=native["program_sha256"]), "native stdout status binding")
    started = timestamp(execution["started_at"], "process start")
    ended = timestamp(execution["completed_at"], "process completion")
    require(started <= ended, "process completion precedes launch")
    return started, ended


def merge_profiles(program, case, count, samples, executable, *, created_at, frozen_at):
    """Rebuild the exact merged object from every independent raw process."""
    from highlevel_accuracy.bridge import verify_native
    merged = None
    evidence, rows, intervals, warmups = [], [], [], []
    for process in range(count):
        directory = case / f"profile-process-{process}"
        native, execution, raw = raw_evidence(directory)
        interval = verify_execution(execution, native, executable=executable, program_file=case / "program.json",
                                    native_file=directory / "native.json", mode="components", samples=samples)
        require(created_at <= interval[0] <= interval[1] <= frozen_at,
                "source collection must occur after preparation and before ledger freeze")
        verify_native(program, native)
        require(not native.get("independent_processes"), "raw native profile must describe one process")
        evidence.append(raw)
        rows.append(dict(process_index=process, components=deepcopy(native["components"]), raw_evidence=raw))
        intervals.append(interval)
        warmups.append(native["warmup"])
        if merged is None:
            merged = deepcopy(native)
        else:
            same([r["action"] for r in merged["components"]], [r["action"] for r in native["components"]], "independent process component order")
            # Fields other than observed samples/outputs must agree too. In
            # particular, a workspace observation cannot silently be discarded.
            for left, right in zip(merged["components"], native["components"]):
                require(left["action_index"] == right["action_index"] and left["workspace_bytes"] == right["workspace_bytes"], "independent process component boundary/workspace differs")
                left["samples_ns"].extend(right["samples_ns"])
            merged["samples"] += native["samples"]
    require(len(set(warmups)) == 1, "profile processes used inconsistent warmup counts")
    for previous, current in zip(intervals, intervals[1:]):
        require(previous[1] <= current[0], "supposed serial source processes overlap")
    merged["independent_processes"] = rows
    return merged, evidence, intervals


def group_result(case_id, local, ledger, expected_processes, gates):
    """Small independent aggregation; numerical/model math is shared and pure."""
    if len(local) != expected_processes:
        return dict(case_id=case_id, status="incomplete", accepted=False, coverage="unknown_failed_not_passing")
    medians = [row["observed_whole_program_median_ns"] for row in local]
    pooled = statistics.median([sample for row in local for sample in row["observed_samples_ns"]])
    predictions = {row["baseline"]["makespan_ns"] for row in local}
    require(len(predictions) == 1, "one strategy has inconsistent frozen predictions")
    prediction = next(iter(predictions))
    error = abs(prediction - pooled) / pooled
    within = all(row["heldout_within_process_cv_passed"] for row in local)
    between_cv = statistics.pstdev(medians) / statistics.mean(medians)
    between = between_cv <= gates["between_process_median_cv_max"]
    accepted = ledger["source_quality_passed"] and within and between and error <= gates["pooled_median_absolute_relative_error_max"]
    return dict(case_id=case_id, status="complete", independent_processes=len(local), source_quality_passed=ledger["source_quality_passed"],
                heldout_within_process_cv_passed=within, heldout_process_medians_ns=medians, heldout_between_process_median_cv=between_cv,
                heldout_between_process_median_cv_passed=between, pooled_heldout_median_ns=pooled, predicted_ns=prediction,
                pooled_median_absolute_relative_error=error, fixed_quality_gates=gates, accepted=accepted)


def verify_experiment(experiment):
    experiment = Path(experiment).resolve()
    require(experiment.is_dir(), "experiment directory unavailable")
    campaign = read(experiment / "campaign.json")
    seal(campaign, "campaign_sha256", "campaign")
    require(campaign.get("format_version") == "highlevel-accuracy-campaign-v1" and campaign.get("hardware_executed") is False, "unknown campaign preparation schema")
    barrier = read(experiment / "all_profiles_frozen.json")
    seal(barrier, "barrier_sha256", "all-profile barrier")
    require(barrier.get("format_version") == "highlevel-all-profiles-frozen-v1" and barrier["campaign_sha256"] == campaign["campaign_sha256"], "barrier/campaign identity differs")
    report = read(experiment / "report.json")
    require(report.get("format_version") == "highlevel-accuracy-campaign-report-v1", "unknown full report schema")
    executable = verify_software(barrier["software"])
    from highlevel_accuracy.bridge import export_plan, verify_native
    from highlevel_accuracy.profiles import GATES, compare_heldout, freeze_profile, run_plan_with_profile, validate_ledger
    from highlevel_accuracy.run import case_specs, load_case
    same(GATES, dict(within_process_cv_max=.15, between_process_median_cv_max=.20,
                     pooled_median_absolute_relative_error_max=.20, independent_processes_min=3), "fixed quality gates")
    same(barrier["fixed_quality_gates"], GATES, "barrier quality gates")
    declared = campaign["cases"]
    require(isinstance(declared, list) and declared, "empty campaign")
    for item in declared:
        require(isinstance(item["id"], str) and re.fullmatch(r"[A-Za-z0-9_.-]+", item["id"]), "unsafe case identity")
    require(len({row["id"] for row in declared}) == len(declared), "duplicate declared case")
    same([{k: row[k] for k in ("id", "kwargs")} for row in declared], case_specs(campaign["suite"]), "complete suite/strategy inventory")
    same([row["case_id"] for row in barrier["ledgers"]], [row["id"] for row in declared], "ordered all-profile ledger coverage")
    source_count = barrier["profile_processes_per_case"]
    target_count = report["independent_heldout_processes_per_case"]
    source_samples = barrier["samples_per_process"]
    require(type(source_count) is int and source_count >= 3 and type(target_count) is int and target_count >= 3, ">=3 independent processes required for both splits")
    require(type(source_samples) is int and source_samples >= 3, ">=3 source samples required")
    created = timestamp(campaign["created_at"], "campaign preparation")
    linked = timestamp(barrier["software"]["successful_build_record"]["linked_at"], "successful native link")
    global_freeze = timestamp(barrier["frozen_at"], "global profile freeze")
    require(created <= global_freeze, "global freeze precedes preparation")
    all_source_intervals, all_target_intervals = [], []
    rebuilt_results, rebuilt_groups, rebuilt_failures, phase_rows = [], [], [], []
    ledger_source_quality = []
    explicit_tolerances = []
    checked_programs, checked_ports, checked_sources = 0, 0, 0
    for item, ledger_row in zip(declared, barrier["ledgers"]):
        case, plan, program = load_case(experiment, item)
        checked_programs += 1
        require(program["seed"] == item["profile_seed"], "source numerical seed differs from prepared declaration")
        # Bind every parser input used by the actual model/compiler, not merely
        # the high-level adapter's own Python source inventory.
        for source in plan._compiled.sources:
            require(file_sha(source["path"]) == source["sha256"], "model/compiler source bytes changed: " + source["path"])
            checked_sources += 1
        ledger = read(case / "frozen_profile.json")
        validate_ledger(program, ledger)
        frozen = timestamp(ledger["frozen_at"], "case profile freeze")
        require(created <= frozen <= global_freeze, "case freeze lies outside the preparation/global-freeze interval")
        same(ledger_row, dict(case_id=item["id"], ledger_sha256=ledger["ledger_sha256"]), "barrier ledger identity")
        merged, evidence, intervals = merge_profiles(program, case, source_count, source_samples, executable,
                                                     created_at=max(created, linked), frozen_at=frozen)
        all_source_intervals.extend(intervals)
        rebuilt_ledger = freeze_profile(program, merged, executable_sha256=barrier["software"]["binary_sha256"],
            evidence=dict(processes=evidence, software=barrier["software"], independent_process_count=source_count), frozen_at=ledger["frozen_at"])
        same(ledger, rebuilt_ledger, "frozen ledger reconstructed from raw independent processes")
        prediction = run_plan_with_profile(plan, program, rebuilt_ledger)
        same(read(case / "modelplan_measured_prediction.json"), prediction, "actual measured ModelPlan prediction")
        phase_rows.append(dict(case_id=item["id"], status="frozen", source_quality_passed=ledger["source_quality_passed"], ledger_sha256=ledger["ledger_sha256"]))
        ledger_source_quality.append(ledger["source_quality_passed"])
        expected_profile_dirs = {f"profile-process-{i}" for i in range(source_count)}
        same(sorted(p.name for p in case.glob("profile-process-*")), sorted(expected_profile_dirs), "raw source process directory coverage")
        expected_target_dirs = {f"heldout-process-{i}" for i in range(target_count)}
        # A missing target execution must be explicitly represented as a
        # failed process in the report; it cannot silently reduce coverage.
        require({p.name for p in case.glob("heldout-process-*")} <= expected_target_dirs, "undeclared heldout process directory")
        local = []
        for process in range(target_count):
            failure_file = case / f"failure-{process}.json"
            comparison_file = case / f"comparison-{process}.json"
            require(not (failure_file.exists() and comparison_file.exists()), "one heldout process is both failed and compared")
            require(comparison_file.exists() or failure_file.exists(), "missing declared heldout process coverage")
            if failure_file.exists():
                # Failures remain unknown/non-passing. The saved exception is
                # rederived below where raw evidence permits it.
                failure = read(failure_file)
                same({k: failure[k] for k in ("case_id", "process_index", "status", "coverage")},
                     dict(case_id=item["id"], process_index=process, status="failed", coverage="unknown_failed_not_passing"), "heldout failure identity")
                require(isinstance(failure.get("error_type"), str) and isinstance(failure.get("error"), str), "missing heldout failure cause")
                execution_file = case / f"heldout-process-{process}" / "execution.json"
                require(execution_file.exists(), "failed heldout has no raw process evidence")
                execution = read(execution_file)
                require(isinstance(execution.get("stdout"), str) and isinstance(execution.get("stderr"), str), "failed process stdout/stderr missing")
                start, end = timestamp(execution["started_at"], "failed process start"), timestamp(execution["completed_at"], "failed process completion")
                require(global_freeze <= start <= end, "failed heldout violates global freeze chronology")
                heldout_file=case/f"heldout-program-{process}.json"
                heldout=read(heldout_file)
                require(heldout["seed"]==100_000_000+item["profile_seed"]*10+process,"failed heldout seed/process identity changed")
                same(heldout,export_plan(plan,seed=heldout["seed"]),"failed heldout compiled program")
                command=execution.get("command")
                require(isinstance(command,list) and len(command)==6,"failed process native command missing")
                path_same(command[0],executable,"failed native executable")
                path_same(command[1],heldout_file,"failed native program")
                path_same(command[5],execution_file.parent/"native.json","failed native output")
                require(command[2]=="replay" and int(command[3])>=0 and int(command[4])>=3,"failed native command sampling protocol differs")
                from highlevel_accuracy.bridge import RUNTIME_FLAGS, BridgeError
                same(execution.get("requested_runtime_flags"),RUNTIME_FLAGS,"failed process runtime flags")
                require(execution.get("pid_scope")=="one_fresh_native_process","failed process independence contract missing")
                if execution.get("returncode")!=0 or execution.get("timed_out") is True or execution.get("launch_failed") is True:
                    expected_error=BridgeError("native execution failed; raw evidence preserved: "+str(execution_file)+" "+execution["stderr"][-3000:])
                else:
                    native,_,_=raw_evidence(execution_file.parent)
                    verify_execution(execution,native,executable=executable,program_file=heldout_file,
                                     native_file=execution_file.parent/"native.json",mode="replay")
                    # Reproduce the real numerical/model failure. Successful
                    # data cannot be assigned an invented failure description.
                    expected_error=None
                    try:
                        compare_heldout(heldout,rebuilt_ledger,native,heldout_started_at=execution["started_at"],
                            executable_sha256=barrier["software"]["binary_sha256"])
                        run_plan_with_profile(plan,heldout,rebuilt_ledger)
                    except Exception as error:expected_error=error
                    require(expected_error is not None,"declared numerical/model failure cannot be reproduced")
                same(failure,dict(case_id=item["id"],process_index=process,status="failed",error_type=type(expected_error).__name__,
                    error=str(expected_error),coverage="unknown_failed_not_passing"),"reconstructed heldout failure")
                all_target_intervals.append((start, end))
                rebuilt_failures.append(failure)
                continue
            saved = read(comparison_file)
            heldout_file = case / f"heldout-program-{process}.json"
            heldout = read(heldout_file)
            require(heldout["seed"] == 100_000_000 + item["profile_seed"] * 10 + process, "heldout seed/process identity changed")
            same(heldout, export_plan(plan, seed=heldout["seed"]), "heldout actual compiled ModelPlan graph/operands/data")
            checked_programs += 1
            directory = case / f"heldout-process-{process}"
            native, execution, raw = raw_evidence(directory)
            interval = verify_execution(execution, native, executable=executable, program_file=heldout_file,
                                        native_file=directory / "native.json", mode="replay")
            require(global_freeze <= interval[0] <= interval[1], "heldout launched before ALL profiles were frozen")
            all_target_intervals.append(interval)
            verification = verify_native(heldout, native)
            checked_ports += verification["independently_checked_ports"]
            tolerance = saved.get("explicit_max_relative_error")
            explicit_tolerances.append(tolerance)
            rebuilt = compare_heldout(heldout, rebuilt_ledger, native, heldout_started_at=execution["started_at"],
                executable_sha256=barrier["software"]["binary_sha256"], max_relative_error=tolerance)
            rebuilt.update(case_id=item["id"], process_index=process, raw_evidence=raw,
                           actual_modelplan_prediction=run_plan_with_profile(plan, heldout, rebuilt_ledger))
            same(saved, rebuilt, "heldout comparison/raw evidence/prediction")
            rebuilt_results.append(rebuilt)
            local.append(rebuilt)
        rebuilt_groups.append(group_result(item["id"], local, rebuilt_ledger, target_count, GATES))
    expected_phase = dict(format_version="highlevel-profile-phase-status-v1", campaign_sha256=campaign["campaign_sha256"],
        cases=phase_rows, failures=[], all_cases_frozen=True, source_quality_passed=all(ledger_source_quality), fixed_quality_gates=GATES)
    same(read(experiment / "profile_phase_status.json"), expected_phase, "profile phase aggregate report")
    require(barrier["source_quality_passed"] == all(ledger_source_quality), "barrier source quality verdict differs")
    require(all_source_intervals and all_target_intervals, "both source and heldout process evidence required")
    require(max(end for _, end in all_source_intervals) <= global_freeze <= min(start for start, _ in all_target_intervals),
            "all-source / global-freeze / first-heldout chronology differs")
    for intervals, label in ((all_source_intervals, "source"), (all_target_intervals, "heldout")):
        ordered = sorted(intervals)
        require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), f"serialized {label} process records overlap")
    require(not explicit_tolerances or all(value == explicit_tolerances[0] for value in explicit_tolerances), "per-process tolerance changed inside one heldout campaign")
    tolerance = explicit_tolerances[0] if explicit_tolerances else None
    rebuilt_report = dict(format_version="highlevel-accuracy-campaign-report-v1", campaign_sha256=campaign["campaign_sha256"],
        barrier_sha256=barrier["barrier_sha256"], all_profiles_frozen_before_any_heldout=True,
        independent_heldout_processes_per_case=target_count, results=rebuilt_results, case_groups=rebuilt_groups, failures=rebuilt_failures,
        fixed_quality_gates=GATES, all_cases_accepted=not rebuilt_failures and all(row["accepted"] for row in rebuilt_groups),
        max_absolute_relative_error=max((row["absolute_relative_error"] for row in rebuilt_results), default=None),
        within_explicit_tolerance=None if tolerance is None else not rebuilt_failures and all(row["within_explicit_tolerance"] for row in rebuilt_results))
    same(report, rebuilt_report, "complete campaign report and fixed acceptance gates")
    # Detect mid-verification source/library edits too. This still certifies
    # current bytes, not the existence of an archived runtime snapshot.
    verify_software(barrier["software"])
    return dict(format_version="offline-highlevel-verification-v1", status="verified", experiment=str(experiment),
        hardware_executed=False, campaign_sha256=campaign["campaign_sha256"], barrier_sha256=barrier["barrier_sha256"],
        report_file_sha256=file_sha(experiment/"report.json"),software_snapshot_sha256=digest(barrier["software"]),
        cases=len(declared), source_processes=len(all_source_intervals), heldout_processes=len(all_target_intervals),
        checked_programs=checked_programs, checked_compiler_model_sources=checked_sources, checked_heldout_materialized_ports=checked_ports,
        all_cases_accepted=rebuilt_report["all_cases_accepted"], failed_or_unknown_processes=len(rebuilt_failures),
        source_verification_mode="strict_current_bytes_at_recorded_paths; archived_software_replay_not_supported",
        chronology_scope="recorded_timezone_aware_timestamps_and_distinct_process_execution_records; no_external_attestation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_experiment(args.experiment)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps(dict(status="rejected", hardware_executed=False, error_type=type(error).__name__, error=str(error)), allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
