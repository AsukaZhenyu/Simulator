"""Read-only post-hoc overlap of owned replay brackets with measured windows.

This tool never samples devices, runs native code, changes a model, fits heldout
data, or interprets host invocation brackets as CPU/GPU active occupancy.
"""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

SCHEMA = "owned-background-bracket-diagnostics-0.1"
LIMITS = {
    "measurement": "Completed host invocation brackets, including submission, possible descheduling, driver calls and synchronization; not CPU/GPU active occupancy.",
    "coverage": "Intersection of recorded background invocation brackets with exact main windows. Inter-invocation gaps remain uncovered.",
    "causality": "Descriptive post-hoc association only. Waiting caused by the main process can lengthen background brackets; coverage is not an exogenous load measurement.",
    "prediction": "No fitting, heldout-derived predictor, interpolation, outlier filtering, automatic state classification or acceptance-gate changes.",
    "censoring": "Full durations of every overlapping background invocation enter duration statistics; clipped segments enter coverage only. Edge censoring is reported.",
    "baseline": "No owned replay is expected for baseline. This does not establish that the machine was idle.",
}


class EvidenceError(ValueError):
    """An artifact cannot support the declared observation."""


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read(path: Path) -> tuple[Any, str]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"missing_or_unreadable_{path.name}") from error
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid_json_{path.name}") from error
    return value, _digest(content)


def _integer(value: Any, name: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        raise EvidenceError(f"invalid_{name}")
    return value


def _clock(value: dict[str, Any]) -> tuple[str, int]:
    if not isinstance(value, dict) or not isinstance(value.get("clock_id"), str) or not value["clock_id"]:
        raise EvidenceError("missing_clock_identity")
    return value["clock_id"], _integer(value.get("frequency_hz"), "clock_frequency", True)


def _ns(ticks: int, frequency: int) -> int:
    # Positive durations use the native C++ llround half-up rule.
    value = Fraction(ticks * 1_000_000_000, frequency)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum_ns": None, "median_ns": None, "p90_ns": None,
                "p99_ns": None, "maximum_ns": None, "population_cv": None}
    ordered = sorted(values)
    mean = statistics.mean(values)
    return {"count": len(values), "minimum_ns": ordered[0], "median_ns": statistics.median(ordered),
            "p90_ns": ordered[math.ceil(0.90 * len(values)) - 1],
            "p99_ns": ordered[math.ceil(0.99 * len(values)) - 1], "maximum_ns": ordered[-1],
            "population_cv": statistics.pstdev(values) / mean if mean > 0 else None}


def _ordered(intervals: list[tuple[int, int]], name: str) -> None:
    previous = -1
    for begin, end in intervals:
        _integer(begin, name + "_begin")
        _integer(end, name + "_end")
        if end <= begin or begin < previous:
            raise EvidenceError(name + "_reversed_or_overlapping_intervals")
        previous = end


def _native_intervals(native: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if native.get("format_version") != "state-calibration-native-v1":
        raise EvidenceError("unsupported_native_schema")
    _, frequency = _clock(native.get("clock"))
    blocks, intervals = [], []
    for index, block in enumerate(native.get("blocks", [])):
        if block.get("block_index") != index:
            raise EvidenceError("native_block_identity_mismatch")
        begin = _integer(block.get("begin_tick"), "block_begin")
        end = _integer(block.get("end_tick"), "block_end")
        blocks.append((begin, end))
        samples = block.get("iterations")
        if not isinstance(samples, list) or not samples or block.get("iteration_count") != len(samples):
            raise EvidenceError("native_invocation_count_mismatch")
        total = 0
        for iteration, item in enumerate(samples):
            if item.get("iteration") != iteration:
                raise EvidenceError("native_invocation_identity_mismatch")
            left = _integer(item.get("begin_tick"), "invocation_begin")
            right = _integer(item.get("end_tick"), "invocation_end")
            if not begin <= left < right <= end:
                raise EvidenceError("native_invocation_outside_block")
            duration = _integer(item.get("elapsed_ns"), "invocation_duration", True)
            if abs(duration - _ns(right - left, frequency)) > 1:
                raise EvidenceError("native_invocation_duration_clock_mismatch")
            intervals.append((left, right))
            total += duration
        if abs(_integer(block.get("wall_elapsed_ns"), "block_wall", True) - _ns(end - begin, frequency)) > 1:
            raise EvidenceError("native_block_duration_clock_mismatch")
        if block.get("completed_work_ns") != total:
            raise EvidenceError("native_completed_work_sum_mismatch")
    if not blocks:
        raise EvidenceError("native_has_no_measured_blocks")
    _ordered(blocks, "native_block")
    _ordered(intervals, "native_invocation")
    return blocks, intervals


def bracket_coverage(background: list[tuple[int, int]], windows: list[tuple[int, int]], frequency: int) -> dict[str, Any]:
    """Return exact clipped union coverage; malformed overlaps are rejected."""
    _integer(frequency, "frequency", True)
    _ordered(background, "background")
    _ordered(windows, "window")
    if not windows:
        raise EvidenceError("coverage_has_no_windows")
    ends = [end for _, end in background]
    covered = 0
    window_ticks = sum(end - begin for begin, end in windows)
    gaps, leading, trailing = [], [], []
    selected: set[int] = set()
    left_censored: set[int] = set()
    right_censored: set[int] = set()
    intersecting_pairs = 0
    for begin, end in windows:
        cursor = begin
        index = bisect.bisect_right(ends, begin)
        first, final = None, None
        while index < len(background) and background[index][0] < end:
            left, right = background[index]
            clipped_left, clipped_right = max(left, begin), min(right, end)
            if clipped_left < clipped_right:
                if clipped_left > cursor:
                    gaps.append(_ns(clipped_left - cursor, frequency))
                cursor = max(cursor, clipped_right)
                covered += clipped_right - clipped_left
                selected.add(index)
                intersecting_pairs += 1
                if left < begin:
                    left_censored.add(index)
                if right > end:
                    right_censored.add(index)
                first = clipped_left if first is None else first
                final = clipped_right
            index += 1
        if cursor < end:
            gaps.append(_ns(end - cursor, frequency))
        leading.append(_ns((first - begin) if first is not None else end - begin, frequency))
        trailing.append(_ns((end - final) if final is not None else end - begin, frequency))
    # Windows and background intervals are each non-overlapping. Therefore the
    # clipped segment sum is their union intersection, without double counting.
    durations = [_ns(background[index][1] - background[index][0], frequency) for index in sorted(selected)]
    ratio = Fraction(covered, window_ticks)
    return {"window_count": len(windows), "window_ticks": window_ticks, "covered_ticks": covered,
            "uncovered_ticks": window_ticks - covered,
            "coverage_fraction": float(ratio), "coverage_fraction_exact": {"numerator": ratio.numerator, "denominator": ratio.denominator},
            "unique_overlapping_invocations": len(selected), "positive_intersection_pairs": intersecting_pairs,
            "left_censored_invocations": len(left_censored), "right_censored_invocations": len(right_censored),
            "overlapping_invocation_full_durations": _stats(durations), "complement_gap_durations": _stats(gaps),
            "leading_edge_gaps": _stats(leading), "trailing_edge_gaps": _stats(trailing),
            "quantile_estimator": "nearest_rank; median uses the arithmetic mean of the two middle observations",
            "duration_censoring_policy": "Full overlapping invocation durations retained; clipping only contributes to coverage."}


def _target_windows(native: dict[str, Any], job: dict[str, Any]) -> list[tuple[int, int]] | None:
    if job.get("instrumentation") != "stamped":
        return None
    target = _integer(native.get("target_action_index"), "target_action_index")
    actions = native.get("actions", [])
    if target >= len(actions) or actions[target].get("kind") != "COMPUTE" or actions[target].get("device") != "gpu":
        raise EvidenceError("selected_main_target_is_not_GPU_compute")
    windows = []
    for block in native["blocks"]:
        selected = [item for item in block.get("action_intervals", []) if item.get("action_index") == target]
        if len(selected) != block["iteration_count"]:
            raise EvidenceError("selected_GPU_action_coverage_mismatch")
        for index, (action, invocation) in enumerate(zip(selected, block["iterations"])):
            left = _integer(action.get("begin_tick"), "action_begin")
            right = _integer(action.get("end_tick"), "action_end")
            if action.get("iteration") != index or not invocation["begin_tick"] <= left < right <= invocation["end_tick"]:
                raise EvidenceError("selected_GPU_action_chronology_mismatch")
            windows.append((left, right))
    _ordered(windows, "selected_GPU_action")
    return windows


def diagnose_job(directory: Path, expected_job: dict[str, Any]) -> dict[str, Any]:
    """Read one completed job; retain failure and missing evidence as unknown."""
    directory = Path(directory)
    job_id = expected_job.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id) or directory.name != job_id:
        raise EvidenceError("unsafe_or_mismatched_job_identity")
    base = {key: expected_job.get(key) for key in ("id", "split", "case_id", "seed", "repeat", "condition", "mode", "instrumentation", "telemetry", "execution_boundary")}
    result = {"job": base, "status": "unknown", "reason": None, "inputs_sha256": {}}

    def bound(name: str, artifacts: dict[str, str] | None = None):
        value, digest = _read(directory / name)
        result["inputs_sha256"][name] = digest
        if artifacts is not None and artifacts.get(name) != digest:
            raise EvidenceError("artifact_sha256_mismatch_" + name)
        return value

    try:
        row = bound("execution.json")
        if row.get("job") != expected_job:
            raise EvidenceError("execution_job_does_not_match_manifest")
        if row.get("status") != "measured" or row.get("returncode") != 0 or row.get("summary", {}).get("native_numeric_passed") is not True:
            result["reason"] = "job_not_successfully_measured_and_numerically_verified"
            return result
        artifacts = row.get("artifacts_sha256")
        if not isinstance(artifacts, dict):
            raise EvidenceError("missing_raw_artifact_digest_inventory")
        duplicate = row.get("raw_file_sha256")
        if duplicate is not None and duplicate != artifacts:
            raise EvidenceError("raw_artifact_digest_inventories_disagree")
        main = bound("native.json", artifacts)
        if row.get("summary", {}).get("native_sha256") != result["inputs_sha256"]["native.json"]:
            raise EvidenceError("main_native_summary_digest_mismatch")
        clock = _clock(row.get("native_clock"))
        if _clock(main.get("clock")) != clock:
            raise EvidenceError("main_execution_clock_mismatch")
        if main.get("process_id") != row.get("process_id") or main.get("mode") != expected_job.get("mode"):
            raise EvidenceError("main_native_process_or_mode_identity_mismatch")
        if main.get("execution_boundary", "legacy_double_sync") != expected_job.get("execution_boundary", "legacy_double_sync"):
            raise EvidenceError("main_native_execution_boundary_mismatch")
        main_blocks, _ = _native_intervals(main)
        if expected_job.get("condition") == "baseline":
            if row.get("background") is not None or "background-native.json" in artifacts:
                raise EvidenceError("baseline_contains_unexpected_owned_background")
            result.update(status="not_applicable", reason="baseline_has_no_owned_background_replay")
            return result
        if expected_job.get("condition") not in ("cpu_replay_load", "gpu_replay_load"):
            raise EvidenceError("unknown_declared_background_condition")
        background_row = row.get("background")
        if not isinstance(background_row, dict) or background_row.get("returncode") != 0 or row.get("background_forced_termination"):
            raise EvidenceError("background_missing_failed_or_forced_termination")
        background = bound("background-native.json", artifacts)
        if _clock(background.get("clock")) != clock:
            raise EvidenceError("main_background_execution_clock_mismatch")
        if background.get("process_id") != background_row.get("pid") or background.get("mode") != "isolated":
            raise EvidenceError("background_native_process_or_mode_identity_mismatch")
        if not main.get("program_sha256") or main.get("program_sha256") != background.get("program_sha256"):
            raise EvidenceError("main_background_program_identity_mismatch")
        if main.get("observed_runtime_flags") != background.get("observed_runtime_flags"):
            raise EvidenceError("main_background_runtime_flags_differ")
        expected_device = "cpu" if expected_job["condition"] == "cpu_replay_load" else "gpu"
        if background.get("target", {}).get("action", {}).get("device") != expected_device:
            raise EvidenceError("background_target_device_disagrees_with_condition")
        _, background_intervals = _native_intervals(background)
        frequency = clock[1]
        per_block = [{"block_index": index, **bracket_coverage(background_intervals, [window], frequency)}
                     for index, window in enumerate(main_blocks)]
        selected = _target_windows(main, expected_job)
        all_durations = [_ns(end - begin, frequency) for begin, end in background_intervals]
        all_gaps = [_ns(background_intervals[index][0] - background_intervals[index - 1][1], frequency)
                    for index in range(1, len(background_intervals))]
        result.update(status="observed", reason=None, clock={"clock_id": clock[0], "frequency_hz": frequency},
                      recorded_condition_envelope_verified=row["summary"].get("condition_verified"),
                      main_blocks=per_block, all_main_blocks=bracket_coverage(background_intervals, main_blocks, frequency),
                      selected_GPU_action_windows=bracket_coverage(background_intervals, selected, frequency) if selected is not None else None,
                      selected_GPU_action_unknown_reason=None if selected is not None else "bare_instrumentation_has_no_selected_action_brackets",
                      all_background_invocation_durations=_stats(all_durations), all_background_inter_invocation_gaps=_stats(all_gaps),
                      background_invocation_count=len(background_intervals))
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        # Never expose workstation paths or command strings in public reports.
        result.update(status="unknown", reason=str(error) if isinstance(error, EvidenceError) else "malformed_evidence_" + type(error).__name__)
    return result


def diagnose_campaign(campaign: Path) -> dict[str, Any]:
    campaign = Path(campaign)
    manifest, digest = _read(campaign / "manifest.json")
    inputs = {"manifest.json": digest}
    splits: dict[str, list[dict[str, Any]]] = {"profile": [], "heldout": [], "diagnostic": []}
    identifiers = set()
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict) or job.get("id") in identifiers or job.get("split") not in splits:
            raise EvidenceError("duplicate_or_invalid_manifest_job")
        identifiers.add(job["id"])
        result = diagnose_job(campaign / "jobs" / job["id"], job)
        for name, value in result["inputs_sha256"].items():
            inputs[f"jobs/{job['id']}/{name}"] = value
        splits[job["split"]].append(result)
    for name in ("profile-rows.json", "heldout-rows.json", "diagnostic-rows.json", "frozen-model.json", "profile-barrier.json", "report.json"):
        path = campaign / name
        if path.is_file():
            inputs[name] = _digest(path.read_bytes())
    unchanged = all(_digest((campaign / name).read_bytes()) == digest for name, digest in inputs.items())
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "tool_sha256": _digest(Path(__file__).read_bytes()), "inputs_sha256": inputs, "inputs_unchanged": unchanged,
            "limits": LIMITS, "splits": splits,
            "counts": {split: {status: sum(row["status"] == status for row in rows) for status in ("observed", "unknown", "not_applicable")}
                       for split, rows in splits.items()},
            "heldout_used_for_fit": False, "performance_acceptance_implied": False,
            "CPU_GPU_active_occupancy_measured": False, "causal_effect_proven": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise EvidenceError("new_output_path_required")
    report = diagnose_campaign(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"schema": report["schema"], "counts": report["counts"], "inputs_unchanged": report["inputs_unchanged"]}))
    return 0 if report["inputs_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
