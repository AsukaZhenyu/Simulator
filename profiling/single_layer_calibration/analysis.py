"""Source-bound statistics for independent single-layer latency calibration.

Measurements and parameter estimation live here. Every duration prediction is
delegated to the model-owned serial-action evaluator. Whole-program times are
diagnostic/validation observations and are never fitted into a parameter.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

from llm_infer_model.performance import Parameter, ParameterBundle
from llm_infer_model.performance.serial_actions import (
    SerialActionTiming, SerialTimingProgram, evaluate_serial_actions,
)
from highlevel_accuracy.bridge import sha, verify_native
from highlevel_accuracy.profiles import contract_identity

MODES = ("isolated-components", "context-trace", "bare-replay")
MODEL_NAMES = ("isolated_components", "context_components", "context_actions_and_gaps")
GATES = {"within_process_cv_max": 0.15, "between_process_median_cv_max": 0.20,
         "pooled_median_absolute_relative_error_max": 0.20,
         "independent_processes_min": 3}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"{label} must be integer >= {minimum}")
    return value


def _utc(value):
    _require(isinstance(value, str), "UTC timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid UTC timestamp") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             "timestamp must carry the UTC offset")
    return parsed.astimezone(timezone.utc)


def _hash(value):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), "SHA256 required")
    return value


def _fraction(value):
    if isinstance(value, dict):
        _require(set(value) == {"numerator", "denominator"}, "invalid exact fraction")
        return Fraction(_integer(value["numerator"], "numerator"),
                        _integer(value["denominator"], "denominator", 1))
    return Fraction(value)


def _encoded(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def median(values):
    values = sorted(Fraction(value) for value in values)
    _require(bool(values), "median needs observations")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def cv(values):
    values = [float(value) for value in values]
    _require(bool(values) and all(math.isfinite(value) and value >= 0 for value in values),
             "CV needs finite nonnegative observations")
    average = statistics.mean(values)
    return 0.0 if average == 0 else statistics.pstdev(values) / average


def _samples(values, label, *, allow_zero=False):
    _require(isinstance(values, list) and len(values) >= 3,
             f"{label} needs >=3 retained invocation samples")
    for value in values:
        _integer(value, label, 0 if allow_zero else 1)
    return values


def _verify_source(record):
    path = Path(record["source_path"])
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"raw source unavailable: {path}") from error
    _require(hashlib.sha256(content).hexdigest() == _hash(record["source_sha256"]),
             "raw source bytes changed")
    _require(json.loads(content) == record["native"], "record does not match hashed raw JSON")


def _execution_scope(native):
    pool = native["threadpool"]
    _require(pool.get("requested_poll") in {"default", "0"}, "threadpool polling contract differs")
    _require(isinstance(pool.get("effective_params"), dict) and bool(pool["effective_params"]),
             "effective threadpool parameters required")
    clock = native["native_clock"]
    return {"threadpool_requested_poll": pool["requested_poll"],
            "threadpool_effective_params": deepcopy(pool["effective_params"]),
            "clock_id": clock["clock_id"], "clock_frequency_hz": clock["frequency_hz"],
            "execution_contract": native["execution_contract"],
            "runtime_flags": deepcopy(native["verification_native"]["observed_runtime_flags"]),
            "completed_action_scope": "inclusive_host_completed_compute_copy_or_control_in_exact_serial_sequence"}


def _validate_native(program, native):
    _require(native.get("format_version") == "single-layer-calibration-native-v1", "native schema differs")
    _require(native.get("program_sha256") == program["program_sha256"], "native program differs")
    _require(native.get("actions") == program["actions"], "native action inventory differs")
    mode = native.get("mode")
    _require(mode in MODES, "unknown native measurement mode")
    _integer(native.get("process_id"), "native process ID", 1)
    count = _integer(native.get("samples"), "native samples", 3)
    _integer(native.get("warmup"), "native warmup")
    actions = program["actions"]
    frequency = _integer(native["native_clock"]["frequency_hz"], "clock frequency", 1)
    _require(native["native_clock"].get("cross_process_monotonic_clock") is True,
             "native clock contract differs")
    if mode == "isolated-components":
        _require(native.get("total_samples_ns") == [] and native.get("iterations") == [],
                 "isolated components cannot contain whole-program samples")
        expected = {i for i, action in enumerate(actions) if action["kind"] in {"COMPUTE", "COPY"}}
        seen = set()
        for item in native.get("components", []):
            index = item["action_index"]
            _require(type(index) is int and index in expected and index not in seen,
                     "isolated component inventory differs")
            seen.add(index)
            _require(item["action"] == actions[index], "component action differs")
            samples = _samples(item["samples_ns"], "component durations")
            _require(len(samples) == count, "component sample count differs")
            _integer(item["workspace_bytes"], "component workspace")
            intervals = item["intervals"]
            _require(len(intervals) == count, "component interval count differs")
            for number, interval in enumerate(intervals):
                _require(interval["sample"] == number and interval["elapsed_ns"] == samples[number],
                         "component interval/sample identity differs")
                _validate_interval(interval, frequency)
        _require(seen == expected, "isolated source misses a selected component")
    else:
        _require(native.get("components") == [], "whole replay cannot masquerade as components")
        totals = _samples(native.get("total_samples_ns"), "whole-program observations")
        iterations = native.get("iterations", [])
        _require(len(totals) == count and len(iterations) == count, "whole-program sample count differs")
        last_end = None
        for number, iteration in enumerate(iterations):
            _require(iteration["iteration"] == number and iteration["elapsed_ns"] == totals[number],
                     "iteration/sample identity differs")
            _validate_interval(iteration, frequency)
            _require(last_end is None or iteration["begin_tick"] >= last_end, "iterations overlap")
            last_end = iteration["end_tick"]
            intervals = iteration["action_intervals"]
            if mode == "bare-replay":
                _require(intervals == [], "bare heldout contains action instrumentation")
                continue
            _require(len(intervals) == len(actions), "context trace misses actions")
            previous = 0
            for index, interval in enumerate(intervals):
                _require(interval["action_index"] == index and interval["kind"] == actions[index]["kind"],
                         "context action index/kind differs")
                _validate_interval(interval, frequency)
                begin = _integer(interval["begin_offset_ns"], "action begin offset")
                end = _integer(interval["end_offset_ns"], "action end offset")
                _require(previous <= begin <= end <= totals[number], "action offsets overlap/outside iteration")
                _require(interval["begin_tick"] >= iteration["begin_tick"] and
                         interval["end_tick"] <= iteration["end_tick"], "action ticks outside iteration")
                _require(abs(Fraction(interval["begin_tick"] - iteration["begin_tick"]) * 10**9 / frequency - begin) <= 1 and
                         abs(Fraction(interval["end_tick"] - iteration["begin_tick"]) * 10**9 / frequency - end) <= 1,
                         "action offset does not match native clock")
                previous = end
    return mode


def _validate_interval(interval, frequency):
    begin = _integer(interval["begin_tick"], "begin tick")
    end = _integer(interval["end_tick"], "end tick")
    elapsed = _integer(interval["elapsed_ns"], "elapsed ns")
    _require(end >= begin and abs(Fraction(end - begin) * 10**9 / frequency - elapsed) <= 1,
             "native clock interval differs from duration")


def make_run_record(program, native, *, run_id, split, seed, started_at_utc,
                    completed_at_utc, source_path, source_sha256, software_sha256,
                    numerical_verification=None):
    """Verify one raw process; the driver separately verifies software inventory.

    ``software_sha256`` binds the driver's verified source/binary/runtime
    inventory. It is not an independent observation of the OS loader paths.
    Numerical results are recomputed through the independent bridge oracle;
    an optional caller result is compared rather than trusted.
    """
    _require(isinstance(run_id, str) and bool(run_id), "run ID required")
    _require(split in {"profile", "heldout", "diagnostic"}, "unknown split")
    _require(type(seed) is int and seed == program["seed"], "run seed differs from program")
    _require(_utc(started_at_utc) <= _utc(completed_at_utc), "process chronology reversed")
    mode = _validate_native(program, native)
    verification = verify_native(program, native["verification_native"])
    if numerical_verification is not None:
        _require(numerical_verification == verification, "caller numerical verification differs")
    record = {"format_version": "single-layer-calibration-run-v1", "run_id": run_id,
              "split": split, "seed": seed, "process_id": native["process_id"], "mode": mode,
              "started_at_utc": started_at_utc, "completed_at_utc": completed_at_utc,
              "source_path": str(Path(source_path).resolve()), "source_sha256": _hash(source_sha256),
              "software_sha256": _hash(software_sha256), "program_sha256": program["program_sha256"],
              "contract_identity": contract_identity(program), "verification": verification,
              "execution_scope": _execution_scope(native),
              "numerical_passed": True, "source_verified": True, "native": deepcopy(native)}
    _verify_source(record)
    return record


def _validate_records(records, program, split, *, verify_sources=True):
    records = list(records)
    _require(bool(records), f"{split} observations required")
    ids = set()
    for record in records:
        _require(record.get("format_version") == "single-layer-calibration-run-v1" and
                 record.get("split") == split, f"only verified {split} records allowed")
        _require(record["run_id"] not in ids, "duplicate run ID")
        ids.add(record["run_id"])
        _require(record["contract_identity"] == contract_identity(program), "run exact program contract differs")
        _require(record["execution_scope"] == _execution_scope(record["native"]), "record native execution scope differs")
        _require(record["numerical_passed"] is True and record["source_verified"] is True,
                 "unverified source or numerical results")
        _require(_utc(record["started_at_utc"]) <= _utc(record["completed_at_utc"]), "run chronology reversed")
        _require(record["mode"] == record["native"]["mode"] and
                 record["process_id"] == record["native"]["process_id"] and
                 record["program_sha256"] == record["native"]["program_sha256"], "run metadata/native identity differs")
        _hash(record["software_sha256"])
        native_program = {**program, "program_sha256": record["program_sha256"]}
        _validate_native(native_program, record["native"])
        if verify_sources:
            _verify_source(record)
    _require(len({record["software_sha256"] for record in records}) == 1, "software inventory changed")
    _require(all(record["execution_scope"] == records[0]["execution_scope"] for record in records),
             "threadpool/clock/action execution contracts cannot be mixed")
    # A renamed record from the same native process cannot become independent.
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            if first["process_id"] == second["process_id"]:
                overlap = max(_utc(first["started_at_utc"]), _utc(second["started_at_utc"])) <= min(
                    _utc(first["completed_at_utc"]), _utc(second["completed_at_utc"]))
                _require(not overlap, "native PID/run intervals are not independent")
            _require(first["source_path"] != second["source_path"] and
                     first["source_sha256"] != second["source_sha256"], "raw process evidence reused")
    return records


def _series(record, index, part="duration"):
    native = record["native"]
    if record["mode"] == "isolated-components":
        return next(item["samples_ns"] for item in native["components"] if item["action_index"] == index)
    result = []
    for iteration in native["iterations"]:
        intervals = iteration["action_intervals"]
        if part == "tail":
            value = iteration["elapsed_ns"] - (intervals[-1]["end_offset_ns"] if intervals else 0)
        else:
            interval = intervals[index]
            value = (interval["end_offset_ns"] - interval["begin_offset_ns"] if part == "duration" else
                     interval["begin_offset_ns"] - (intervals[index - 1]["end_offset_ns"] if index else 0))
        result.append(value)
    return result


def _statistics(records, index, part="duration"):
    processes = []
    for record in records:
        values = _series(record, index, part)
        value = median(values)
        processes.append({"run_id": record["run_id"], "process_id": record["process_id"],
                          "samples_ns": values, "median_ns": _encoded(value), "cv": cv(values),
                          "within_process_cv_passed": cv(values) <= GATES["within_process_cv_max"]})
    medians = [_fraction(row["median_ns"]) for row in processes]
    between = cv(medians)
    return {"value_ns": _encoded(median(medians)), "processes": processes,
            "estimator": "exact_median_of_independent_process_medians_equal_process_weight",
            "between_process_median_cv": between,
            "within_process_cv_passed": all(row["within_process_cv_passed"] for row in processes),
            "between_process_median_cv_passed": between <= GATES["between_process_median_cv_max"]}


def _total_statistics(records):
    medians = [median(record["native"]["total_samples_ns"]) for record in records]
    return {"process_medians_ns": [_encoded(value) for value in medians],
            "median_of_process_medians_ns": _encoded(median(medians)),
            "pooled_median_ns": _encoded(median([value for record in records for value in record["native"]["total_samples_ns"]])),
            "within_process_cv": [cv(record["native"]["total_samples_ns"]) for record in records],
            "between_process_median_cv": cv(medians)}


def _predict(model):
    program = SerialTimingProgram.from_dict(model["program"])
    bundle = ParameterBundle.from_dict(model["parameters"])
    return evaluate_serial_actions(program, bundle).to_dict()


def freeze_profile(program, runs, *, frozen_at_utc, instrumentation_relative_error_max=0.05):
    """Freeze three bounded model alternatives using component observations only."""
    _require(type(instrumentation_relative_error_max) in {int, float} and
             math.isfinite(instrumentation_relative_error_max) and instrumentation_relative_error_max >= 0,
             "instrumentation tolerance must be finite and nonnegative")
    records = _validate_records(runs, program, "profile")
    frozen_at = _utc(frozen_at_utc)
    _require(all(_utc(row["completed_at_utc"]) <= frozen_at for row in records), "profile completed after freeze")
    grouped = {mode: [row for row in records if row["mode"] == mode] for mode in MODES}
    _require(all(len(rows) >= GATES["independent_processes_min"] for rows in grouped.values()),
             "each measurement mode requires >=3 independent profile processes")
    counts = {len(row["native"]["total_samples_ns"]) for row in grouped["bare-replay"] + grouped["context-trace"]}
    _require(len(counts) == 1, "context/bare instrumentation comparison needs equal sample counts")
    actions = program["actions"]
    cc = {index for index, action in enumerate(actions) if action["kind"] in {"COMPUTE", "COPY"}}
    isolated = {index: _statistics(grouped["isolated-components"], index) for index in cc}
    context = {index: _statistics(grouped["context-trace"], index) for index in range(len(actions))}
    gaps = {index: _statistics(grouped["context-trace"], index, "gap") for index in range(len(actions))}
    tail = _statistics(grouped["context-trace"], None, "tail")
    models = {}
    for name in MODEL_NAMES:
        whole_context = name == "context_actions_and_gaps"
        selected = isolated if name == "isolated_components" else context
        references = tuple(f'{row["source_path"]}#sha256={row["source_sha256"]}' for row in
                           grouped["isolated-components" if name == "isolated_components" else "context-trace"])
        parameters, steps = [], []
        for index, action in enumerate(actions):
            duration_name, gap_name = f"action:{index}:duration", f"action:{index}:gap_before"
            measured = whole_context or index in cc
            duration = _fraction(selected[index]["value_ns"]) if measured else 0
            gap = _fraction(gaps[index]["value_ns"]) if whole_context else 0
            scope = "exact_program_completed_host_action_in_" + ("isolated_repetition" if name == "isolated_components" else "full_sequence")
            parameters.append(Parameter(duration_name, duration, "ns", "profiled" if measured else "declared",
                                        references if measured else (), scope))
            parameters.append(Parameter(gap_name, gap, "ns", "profiled" if whole_context else "declared",
                                        references if whole_context else (), "before_action_host_dispatch_gap"))
            steps.append(SerialActionTiming(index, action["kind"], duration_name, gap_name))
        parameters.append(Parameter("tail", _fraction(tail["value_ns"]) if whole_context else 0,
                                    "ns", "profiled" if whole_context else "declared", references if whole_context else (),
                                    "after_last_action_host_dispatch_gap"))
        model_program = SerialTimingProgram(tuple(steps), "tail")
        model_program.validate_action_inventory(tuple(action["kind"] for action in actions))
        relevant = [selected[index] for index in cc]
        models[name] = {"program": model_program.to_dict(), "parameters": ParameterBundle(tuple(parameters)).to_dict(),
                        "source_compute_copy_quality_passed": all(row["within_process_cv_passed"] and
                          row["between_process_median_cv_passed"] for row in relevant),
                        "whole_program_duration_used_as_parameter": False,
                        "explicit_ignored_action_kinds": [] if whole_context else sorted({action["kind"] for i, action in enumerate(actions) if i not in cc})}
        models[name]["prediction"] = _predict(models[name])
    context_totals, bare_totals = (_total_statistics(grouped[mode]) for mode in ("context-trace", "bare-replay"))
    context_total, bare_total = (_fraction(stats["median_of_process_medians_ns"]) for stats in (context_totals, bare_totals))
    instrumentation_error = abs(context_total - bare_total) / bare_total
    differences = []
    for index in sorted(cc):
        old, new = _fraction(isolated[index]["value_ns"]), _fraction(context[index]["value_ns"])
        differences.append({"action_index": index, "action": actions[index], "isolated_ns": _encoded(old),
                            "context_ns": _encoded(new), "context_to_isolated_ratio": float(new / old),
                            "signed_context_change_ns": {"numerator": (new-old).numerator, "denominator": (new-old).denominator},
                            "cause": "unresolved; measured_context_and_process_cohorts_differ"})
    gap_totals = [sum(_series(record, i, "gap")[j] for i in range(len(actions))) + _series(record, None, "tail")[j]
                  for record in grouped["context-trace"] for j in range(record["native"]["samples"])]
    observed_context_total = median([value for record in grouped["context-trace"] for value in record["native"]["total_samples_ns"]])
    abnormal = []
    for context_name, stats in (("isolated", isolated), ("context", context)):
        for index, row in stats.items():
            if not row["within_process_cv_passed"] or not row["between_process_median_cv_passed"]:
                abnormal.append({"context": context_name, "action_index": index, "action": actions[index],
                                 "median_ns": row["value_ns"], "within_process_cv_passed": row["within_process_cv_passed"],
                                 "between_process_median_cv": row["between_process_median_cv"], "cause": "unresolved"})
    abnormal.sort(key=lambda row: _fraction(row["median_ns"]), reverse=True)
    frozen = {"format_version": "single-layer-calibration-frozen-v1", "frozen_at_utc": frozen_at_utc,
              "contract_identity": contract_identity(program), "actions": deepcopy(actions),
              "execution_scope": deepcopy(records[0]["execution_scope"]),
              "software_sha256": records[0]["software_sha256"], "profile_run_ids": [row["run_id"] for row in records],
              "profile_seeds": sorted({row["seed"] for row in records}), "profile_runs": deepcopy(records),
              "models": models, "fixed_quality_gates": dict(GATES),
              "action_statistics": {"isolated": {str(k): v for k, v in isolated.items()}, "context": {str(k): v for k, v in context.items()}},
              "gap_statistics": {str(k): {**v, "acceptance_scope": "diagnostic_not_original_source_criterion"} for k, v in gaps.items()},
              "tail_statistics": {**tail, "acceptance_scope": "diagnostic_not_original_source_criterion"},
              "instrumentation_control": {"context": context_totals, "bare": bare_totals,
                  "relative_difference": float(instrumentation_error), "maximum_relative_difference": instrumentation_relative_error_max,
                  "passed": instrumentation_error <= instrumentation_relative_error_max,
                  "scope": "independent_process_cohort_comparison_not_pure_causal_instrumentation_cost",
                  "used_as_performance_parameter": False},
              "context_isolated_differences": differences, "dominant_unstable_sources": abnormal,
              "observed_gap_median_ns": _encoded(median(gap_totals)),
              "observed_gap_to_context_total_median_ratio": float(median(gap_totals) / observed_context_total),
              "residual_cause": "unresolved", "heldout_used": False,
              "software_scope": "driver_verified_inventory_fingerprint_not_resolved_loader_paths",
              "all_samples_retained": True}
    frozen["frozen_sha256"] = sha(frozen)
    return frozen


def evaluate_heldout(program, frozen, runs, *, verify_sources=True):
    """Evaluate fresh bare whole-program runs without updating any parameter."""
    copied = deepcopy(frozen)
    _require(copied.pop("frozen_sha256", None) == sha(copied), "frozen parameter artifact changed")
    _require(frozen.get("format_version") == "single-layer-calibration-frozen-v1" and frozen.get("heldout_used") is False,
             "invalid frozen profile")
    _require(frozen["contract_identity"] == contract_identity(program) and frozen["actions"] == program["actions"],
             "heldout program/action contract differs")
    _require(frozen["fixed_quality_gates"] == GATES, "original acceptance gates changed")
    profile = _validate_records(frozen["profile_runs"], program, "profile", verify_sources=verify_sources)
    heldout = _validate_records(runs, program, "heldout", verify_sources=verify_sources)
    _require(all(row["mode"] == "bare-replay" for row in heldout), "heldout must be bare whole-program replay")
    _require(len(heldout) >= GATES["independent_processes_min"], ">=3 independent heldout processes required")
    _require(all(row["run_id"] not in frozen["profile_run_ids"] and row["seed"] not in frozen["profile_seeds"] for row in heldout),
             "heldout run/seed reused from profiling")
    _require(all(_utc(row["started_at_utc"]) >= _utc(frozen["frozen_at_utc"]) for row in heldout),
             "heldout began before parameter freeze")
    _require(all(row["software_sha256"] == frozen["software_sha256"] for row in heldout),
             "heldout software inventory differs")
    _require(all(row["execution_scope"] == frozen["execution_scope"] for row in heldout),
             "heldout threadpool/clock/action execution scope differs")
    _require(all(row["source_sha256"] not in {old["source_sha256"] for old in profile} for row in heldout),
             "heldout raw evidence reused")
    totals = _total_statistics(heldout)
    observed = _fraction(totals["pooled_median_ns"])
    within = all(value <= GATES["within_process_cv_max"] for value in totals["within_process_cv"])
    between = totals["between_process_median_cv"] <= GATES["between_process_median_cv_max"]
    models = {}
    for name in MODEL_NAMES:
        model = frozen["models"][name]
        prediction = _predict(model)
        _require(prediction == model["prediction"], "model implementation/prediction changed after freeze")
        duration = prediction["duration_ns"]
        signed = (duration - observed) / observed
        source_quality = model["source_compute_copy_quality_passed"]
        accepted = (verify_sources and source_quality and within and between and
                    abs(signed) <= GATES["pooled_median_absolute_relative_error_max"] and frozen["instrumentation_control"]["passed"])
        models[name] = {"prediction": prediction, "predicted_ns": duration,
                        "signed_relative_error": float(signed), "absolute_relative_error": float(abs(signed)),
                        "per_process_absolute_relative_errors": [float(abs(Fraction(duration) - _fraction(value)) / _fraction(value)) for value in totals["process_medians_ns"]],
                        "source_compute_copy_quality_passed": source_quality,
                        "heldout_within_process_cv_passed": within, "heldout_between_process_median_cv_passed": between,
                        "instrumentation_control_passed": frozen["instrumentation_control"]["passed"],
                        "accepted": accepted, "residual_cause": "unresolved",
                        "gap_quality_claimed": False,
                        "scope": "exact_supported_serial_F32_dense_program_not_general_model_accuracy"}
    return {"format_version": "single-layer-calibration-evaluation-v1", "frozen_sha256": frozen["frozen_sha256"],
            "contract_identity": frozen["contract_identity"], "models": models, "heldout_statistics": totals,
            "heldout_runs": deepcopy(heldout), "fixed_quality_gates": dict(GATES),
            "heldout_used_for_fitting": False, "parameter_frozen_before_heldout": True,
            "raw_sources_verified": verify_sources, "all_samples_retained": True,
            "primary_model": "context_actions_and_gaps", "accepted": models["context_actions_and_gaps"]["accepted"],
            "gap_cv_scope": "reported_separately_not_original_source_criterion",
            "native_memory_capacity_accuracy_claimed": False}
