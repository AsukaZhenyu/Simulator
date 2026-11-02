"""Verify same-resident Dense paired evidence and independent-repeat errors.

Repeat 0 stage medians predict repeat 1 serial execution at the same exact shape.
Nested stage sums versus their own serial clock are only bookkeeping checks.
Segmented versus joint graphs are separate valid execution modes, not an accuracy
test. Every raw observation and quality flag remains visible; no costs are fitted
or exported, and no samples are pooled across repeats, shapes, modes or AB/BA.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
from typing import Any

from run_dense_paired_measurements import (SCHEMA as PROFILE_SCHEMA, STAGES, paired_command,
    validate_paired_binding, summarize_paired, paired_quality_flags)
from profile_contract import summarize_samples
from llm_infer_model.m2.dense_contract import build_plan, validate_case
from llm_infer_model.tensor.spec import SpecError


SCHEMA = "dense_paired_error_analysis_v1"
SCOPE = "same-resident graph-partition diagnostic; not independent-resident M2 costs"


class SplitConflict(ValueError):
    """A valid geometry declaration violates development/held-out isolation."""
    def __init__(self, identity: dict, declarations: list[dict]):
        super().__init__("development/held_out overlap at the same physical paired shape")
        self.conflict = {"physical_identity": identity, "declarations": declarations,
                         "error": str(self)}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _same(actual: Any, expected: Any, label: str) -> None:
    if _json(actual) != _json(expected):
        raise ValueError(f"{label} does not match its captured evidence or declared protocol")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _load(path: Path) -> Any:
    def nonfinite(value: str) -> None:
        raise ValueError(f"{path}: nonfinite JSON value {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=nonfinite)


def _source(path: Path) -> dict:
    return {"path": str(path), **({"sha256": _sha(path)} if path.is_file() else {"error": "file is missing"})}


def _raw_path(row: dict, directory: Path) -> Path:
    name = row.get("raw_output")
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("raw_output must be a relative run-directory path")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("raw_output escapes the measurement run directory")
    return path


def _inventory(row: Any, directory: Path) -> dict:
    if not isinstance(row, dict):
        return {"status": "invalid_record", "error": "record must be an object", "record": row}
    result = {key: row.get(key) for key in ("case_id", "repeat", "compute_mode", "status", "error", "quality_flags", "raw_output")}
    result["record_canonical_sha256"] = hashlib.sha256(_json(row).encode()).hexdigest()
    try:
        path = _raw_path(row, directory)
        result["raw_sha256"] = _sha(path)
    except (ValueError, OSError) as exc:
        result["raw_evidence_error"] = str(exc)
    return result


def _shape(case: dict, mode: str, protocol: dict | None = None) -> dict:
    protocol = protocol or {}
    return {"config": case["config"], "threads": case["threads"], "phase": case["phase"],
        "batch": case["batch"], "tokens": case["tokens"], "visible_history": case["visible_history"],
        "compute_mode": mode, "split": case["metadata"]["split"],
        "warmup_rounds_requested": protocol.get("warmup"),
        "min_warmup_ms": protocol.get("min_warmup_ms"),
        "cpu_mask": protocol.get("cpu_mask"),
        "duration_warmup_affinity_protocol_declared": "min_warmup_ms" in protocol and "cpu_mask" in protocol}


def _physical_identity(case: dict) -> dict:
    """Execution labels and warmup/affinity cannot manufacture held-out geometry.

    Paired native graphs depend on q/history, not the prefill/decode label. A
    physical geometry therefore has one split even if its execution mode,
    warmup, CPU placement, phase label or case id changes between runs.
    """
    return {name: case[name] for name in ("config", "threads", "batch", "tokens", "visible_history")}


def _summary(native: dict) -> dict:
    """Preserve every paired observation and stratify only by observed AB/BA."""
    summary = summarize_paired(native)
    strata = {}
    for order in ("AB", "BA"):
        indices = [i for i, selected in enumerate(native["paired_order"]) if selected == order]
        raw = {key: [native[key][i] for i in indices] for key in ("stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns")}
        raw["stage_wall_ns"] = {stage: [native["stage_wall_ns"][stage][i] for i in indices] for stage in STAGES}
        # This is a descriptive subseries, not a new measurement or an estimate
        # formed by discarding the other experimental order.
        strata[order] = {"indices": indices, "samples": raw, "summary": summarize_paired(raw)}
    median_stage_sum = sum(summary["stages"][stage]["median_ns"] for stage in STAGES)
    serial = summary["serial_wall_ns"]["median_ns"]
    joint = summary["joint_wall_ns"]["median_ns"]
    return {"summary": summary, "order_strata": strata,
        "stage_medians_sum_ns": median_stage_sum,
        "same_repeat_clock_bookkeeping": {
            "stage_medians_sum_minus_serial_median_ns": median_stage_sum - serial,
            "signed_ratio": (median_stage_sum-serial)/serial,
            "model_accuracy_claimed": False,
            "scope": "nested timer accounting; sum of medians can differ from median of sums"},
        "segmented_vs_joint_execution": {
            "stage_medians_sum_ns": median_stage_sum, "serial_median_ns": serial, "joint_median_ns": joint,
            "stage_medians_sum_minus_joint_signed_ratio": (median_stage_sum-joint)/joint,
            "serial_minus_joint_signed_ratio": (serial-joint)/joint,
            "model_accuracy_claimed": False,
            "scope": "different legal graph execution partitions with shared resident addresses"},
        "derived_stage_quality_flags": ["stages."+stage for stage in STAGES if summary["stages"][stage]["cv"] > 0.1]}


def _protocol(manifest: dict) -> tuple[SimpleNamespace, list[tuple[int, dict]]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("manifest must declare the paired measurement profile schema")
    _same(manifest.get("scope"), SCOPE, "manifest scope")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("manifest protocol must be an object")
    names = ("threads", "warmup", "samples", "seed", "repeats", "compute_mode")
    if set(protocol) not in (set(names), set(names) | {"min_warmup_ms", "cpu_mask"}):
        raise ValueError("manifest protocol must contain exactly the declared paired axes")
    args = SimpleNamespace(**protocol)
    args.min_warmup_ms = protocol.get("min_warmup_ms")
    args.cpu_mask = protocol.get("cpu_mask")
    _int(args.threads, "threads", 1); _int(args.warmup, "warmup")
    _int(args.samples, "samples", 4); _int(args.seed, "seed"); _int(args.repeats, "repeats", 1)
    if args.samples > 10000 or args.samples % 2 or args.seed > 0xFFFFFFFF or args.compute_mode not in ("graph", "plan"):
        raise ValueError("invalid paired samples/seed/compute mode")
    if "min_warmup_ms" in protocol:
        _int(args.min_warmup_ms, "min_warmup_ms")
        if args.cpu_mask is not None and (not isinstance(args.cpu_mask, list) or not args.cpu_mask
                or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in args.cpu_mask)
                or len(set(args.cpu_mask)) != len(args.cpu_mask)):
            raise ValueError("cpu_mask must be null or a nonempty unique list of nonnegative logical CPU ids")
    split = manifest.get("split")
    if split not in (None, "development", "held_out"):
        raise ValueError("manifest split must be development, held_out or explicit null")
    if "split" not in manifest:
        raise ValueError("manifest must explicitly declare its split selection")
    complete = build_plan(manifest.get("suite"), args.threads)
    seen_shapes = {}
    for case in complete:
        if case["kind"] != "block":
            raise ValueError("paired cases must declare complete block geometry")
        metadata = case.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("split") not in ("development", "held_out"):
            raise ValueError("every shape must predeclare metadata.split as development or held_out")
        shape_key = _json(_physical_identity(case))
        if shape_key in seen_shapes:
            if seen_shapes[shape_key]["split"] != metadata["split"]:
                raise SplitConflict(_physical_identity(case), [seen_shapes[shape_key],
                    {"case_id": case["id"], "split": metadata["split"]}])
            raise ValueError("suite contains duplicate exact physical paired shape entries")
        seen_shapes[shape_key] = {"case_id": case["id"], "split": metadata["split"]}
    selected = [case for case in complete if split is None or case["metadata"]["split"] == split]
    if not selected:
        raise ValueError("manifest split selects no cases")
    random.Random(args.seed).shuffle(selected)
    jobs = [(repeat, case) for repeat in range(args.repeats)
            for case in (selected if repeat % 2 == 0 else list(reversed(selected)))]
    _same(manifest.get("case_order"), [{"repeat": repeat, "case_id": case["id"]} for repeat, case in jobs], "manifest case order")
    _same(manifest.get("planned_jobs"), len(jobs), "manifest planned jobs")
    _digest(manifest.get("suite_sha256"), "suite_sha256")
    _digest(manifest.get("model_contract_sha256"), "model_contract_sha256")
    sources = manifest.get("benchmark_source_sha256")
    source_names = {name.replace("\\", "/") for name in sources if isinstance(name, str)} if isinstance(sources, dict) else set()
    if not isinstance(sources, dict) or not {"run_dense_paired_measurements.py", "native/dense_layer_probe.cpp"}.issubset(source_names):
        raise ValueError("manifest lacks runner/native source hash inventory")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("benchmark source names must be nonempty strings")
        _digest(digest, "benchmark source " + name)
    runtime = manifest.get("runtime_sha256")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("manifest runtime hash inventory must be nonempty")
    for name, digest in runtime.items():
        if not isinstance(name, str) or not name:
            raise ValueError("runtime names must be nonempty strings")
        _digest(digest, "runtime " + name)
    return args, jobs


def _verify_record(row: dict, directory: Path, manifest: dict, args: SimpleNamespace,
                   expected_case: dict, expected_repeat: int) -> dict:
    if row.get("status") != "measured":
        raise ValueError("failed or incomplete record cannot support an error estimate")
    case = validate_case(row.get("case"))
    _same(case, expected_case, "record case and manifest plan")
    _same(row.get("case_id"), case["id"], "record case id")
    _same(row.get("repeat"), expected_repeat, "record repeat")
    _same(row.get("compute_mode"), args.compute_mode, "record compute mode")
    raw_path = _raw_path(row, directory)
    raw = _load(raw_path)
    if not isinstance(raw, dict) or isinstance(raw.get("returncode"), bool) or raw.get("returncode") != 0 or "error" in raw or raw.get("timed_out"):
        raise ValueError("raw execution was not successful")
    _same(raw.get("case"), case, "raw case")
    _same(raw.get("repeat"), expected_repeat, "raw repeat")
    command = raw.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise ValueError("raw command must be an explicit argument list")
    _same(command, paired_command(case, Path(command[0]), args), "raw native command")
    binary_hash = _digest(row.get("native_binary_sha256"), "record native binary hash")
    _same(manifest["runtime_sha256"].get(command[0]), binary_hash, "manifest native binary hash")
    if not isinstance(raw.get("stdout"), str) or not isinstance(raw.get("stderr"), str):
        raise ValueError("raw stdout and stderr must be captured strings")
    native = json.loads(raw["stdout"], parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite native JSON value " + value)))
    _same(row.get("native"), native, "raw stdout and recorded native payload")
    validate_paired_binding(case, native, args)
    summary = summarize_paired(native)
    _same(row.get("summary"), summary, "record summary and raw samples")
    expected_flags = paired_quality_flags(summary)
    _same(row.get("quality_flags"), expected_flags, "quality flags and raw variability")
    observations = _summary(native)
    stable = not expected_flags and not observations["derived_stage_quality_flags"]
    return {"case_id": case["id"], "case": case, "repeat": expected_repeat,
        "shape": _shape(case, args.compute_mode, manifest["protocol"]), "status": "verified", "errors": [],
        "raw_samples_verified": True, "raw_sha256": _sha(raw_path), "native_binary_sha256": binary_hash,
        "record_canonical_sha256": hashlib.sha256(_json(row).encode()).hexdigest(),
        "quality_flags": row["quality_flags"], "all_reported_timing_cvs_le_0_1": stable,
        "warmup_affinity_evidence": {"explicit_duration_affinity_contract": args.min_warmup_ms is not None,
            **{name: native.get(name) for name in ("min_warmup_ms", "warmup_rounds_requested",
                "warmup_rounds_effective", "actual_warmup_elapsed_ns", "cpu_affinity")}},
        "samples": {key: native[key] for key in ("paired_order", "stage_wall_ns", "stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns")},
        "source_hash_binding": {"native_binary_matches_manifest": True,
            "benchmark_source_sha256": manifest["benchmark_source_sha256"],
            "model_contract_sha256": manifest["model_contract_sha256"],
            "source_runtime_inventory_scope": "captured declarations bound to hashed manifest; loader resolution and historical source contents are not independently attested"},
        **observations}


def _groups(runs: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for run in runs:
        if not run.get("manifest_binding_valid", False):
            continue
        for row in run["records"]:
            if row["status"] != "verified":
                continue
            key = _json(row["shape"])
            group = groups.setdefault(key, {"shape": row["shape"], "observations": [],
                "independent_repeat_validation": [], "samples_pooled": False})
            group["observations"].append({"run": run["source"]["measurements"]["path"], "repeat": row["repeat"],
                "summary": row["summary"], "quality_flags": row["quality_flags"],
                "derived_stage_quality_flags": row["derived_stage_quality_flags"],
                "all_reported_timing_cvs_le_0_1": row["all_reported_timing_cvs_le_0_1"],
                "stage_medians_sum_ns": row["stage_medians_sum_ns"],
                "same_repeat_clock_bookkeeping": row["same_repeat_clock_bookkeeping"],
                "segmented_vs_joint_execution": row["segmented_vs_joint_execution"]})
        grouped = {}
        for row in run["records"]:
            if row["status"] == "verified":
                grouped.setdefault(_json(row["shape"]), {})[row["repeat"]] = row
        for key, repeats in grouped.items():
            validation = {"run": run["source"]["measurements"]["path"], "prediction_repeat": 0, "verification_repeat": 1,
                "available_verified_repeats": sorted(repeats), "status": "unavailable",
                "scope": "same exact resident partition variant; independent process repeat, not unmeasured-shape generalization",
                "same_batch_bookkeeping_used_as_accuracy": False}
            if 0 in repeats and 1 in repeats:
                prediction = repeats[0]["stage_medians_sum_ns"]
                observed = repeats[1]["summary"]["serial_wall_ns"]["median_ns"]
                error = (prediction-observed)/observed
                validation.update(status="observed", predicted_stage_medians_sum_ns=prediction,
                    observed_independent_serial_median_ns=observed, signed_relative_error=error,
                    absolute_relative_error=abs(error),
                    all_reported_timing_cvs_le_0_1=repeats[0]["all_reported_timing_cvs_le_0_1"] and repeats[1]["all_reported_timing_cvs_le_0_1"],
                    eligible_for_error_acceptance=repeats[0]["all_reported_timing_cvs_le_0_1"] and repeats[1]["all_reported_timing_cvs_le_0_1"],
                    explicit_error_acceptance_tolerance=None,
                    prediction_quality_flags=repeats[0]["quality_flags"],
                    verification_quality_flags=repeats[1]["quality_flags"])
            groups[key]["independent_repeat_validation"].append(validation)
    return [groups[key] for key in sorted(groups)]


def _mode_comparisons(runs: list[dict]) -> list[dict]:
    matched = {}
    for run in runs:
        if not run.get("manifest_binding_valid", False):
            continue
        for row in run["records"]:
            if row["status"] != "verified":
                continue
            shape = {key: value for key, value in row["shape"].items() if key != "compute_mode"}
            key = _json({"shape": shape, "repeat": row["repeat"]})
            matched.setdefault(key, []).append((run, row))
    results = []
    for key, observations in sorted(matched.items()):
        for (left_run, left), (right_run, right) in itertools.combinations(observations, 2):
            if left["shape"]["compute_mode"] == right["shape"]["compute_mode"]:
                continue
            quantities = {}
            for name in ("stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns"):
                a, b = left["summary"][name]["median_ns"], right["summary"][name]["median_ns"]
                quantities[name] = {"left_median_ns": a, "right_median_ns": b,
                    "signed_change_ns": b-a, "signed_change_relative_to_left": (b-a)/a}
            left_protocol = {name: value for name, value in left_run["protocol"].items() if name != "compute_mode"}
            right_protocol = {name: value for name, value in right_run["protocol"].items() if name != "compute_mode"}
            results.append({**json.loads(key), "left_run": left_run["source"]["measurements"]["path"],
                "right_run": right_run["source"]["measurements"]["path"],
                "left_compute_mode": left["shape"]["compute_mode"], "right_compute_mode": right["shape"]["compute_mode"],
                "protocol_equal_except_compute_mode": _json(left_protocol) == _json(right_protocol),
                "native_binary_hash_equal": left["native_binary_sha256"] == right["native_binary_sha256"],
                "benchmark_source_hashes_equal": _json(left_run["benchmark_source_sha256"]) == _json(right_run["benchmark_source_sha256"]),
                "model_contract_hash_equal": left_run["model_contract_sha256"] == right_run["model_contract_sha256"],
                "runtime_hash_inventory_equal": _json(left_run["runtime_sha256"]) == _json(right_run["runtime_sha256"]),
                "left_quality_flags": left["quality_flags"], "right_quality_flags": right["quality_flags"],
                "left_derived_stage_quality_flags": left["derived_stage_quality_flags"],
                "right_derived_stage_quality_flags": right["derived_stage_quality_flags"],
                "quantities": quantities, "samples_pooled": False, "model_accuracy_claimed": False,
                "scope": "matched shape and repeat estimates under graph versus prebuilt-plan execution"})
    return results


def _error_stats(observations: list[dict]) -> dict:
    if not observations:
        return {"count": 0, "mean_absolute_relative_error": None,
                "median_absolute_relative_error": None, "max_absolute_relative_error": None,
                "worst_observation": None}
    errors = [row["absolute_relative_error"] for row in observations]
    worst = max(observations, key=lambda row: row["absolute_relative_error"])
    return {"count": len(observations), "mean_absolute_relative_error": statistics.mean(errors),
        "median_absolute_relative_error": statistics.median(errors),
        "max_absolute_relative_error": max(errors), "worst_observation": worst}


def _split_error_statistics(groups: list[dict]) -> list[dict]:
    """Describe error estimates, exposing every unstable exclusion explicitly."""
    result = []
    for split in ("development", "held_out"):
        for mode in ("graph", "plan"):
            observations = [{"shape": group["shape"], **validation}
                for group in groups if group["shape"]["split"] == split and group["shape"]["compute_mode"] == mode
                for validation in group["independent_repeat_validation"] if validation["status"] == "observed"]
            stable = [row for row in observations if row["eligible_for_error_acceptance"]]
            flagged = [row for row in observations if not row["eligible_for_error_acceptance"]]
            result.append({"split": split, "compute_mode": mode,
                "all_verified_observations": observations, "all_verified_errors": _error_stats(observations),
                "stable_evidence_errors": _error_stats(stable), "stable_evidence_observations": stable,
                "flagged_observations_ineligible_for_error_acceptance": flagged,
                "statistics_scope": "unweighted per-shape/run independent-repeat error estimates; raw samples are not pooled",
                "explicit_error_acceptance_tolerance": None,
                "quality_exclusions_silent": False})
    return result


def _enforce_split_isolation(runs: list[dict]) -> list[dict]:
    """Check declarations across all supplied runs before computing any score."""
    ownership = {}
    for index, run in enumerate(runs):
        for declaration in run.get("declared_physical_shape_splits", []):
            key = _json(declaration["physical_identity"])
            ownership.setdefault(key, []).append({"run_index": index,
                "measurements": run["source"]["measurements"]["path"],
                "case_id": declaration["case_id"], "split": declaration["split"]})
    conflicts = [{**conflict, "measurements": run["source"]["measurements"]["path"]}
        for run in runs for conflict in run.get("local_split_conflicts", [])]
    for key, declarations in sorted(ownership.items()):
        if len({item["split"] for item in declarations}) < 2:
            continue
        conflicts.append({"physical_identity": json.loads(key), "declarations": declarations,
            "error": "physical geometry belongs to both development and held_out across supplied runs"})
        for index in {item["run_index"] for item in declarations}:
            run = runs[index]
            run["status"] = "rejected"
            run["errors"].append("development/held_out split conflict across supplied runs at one physical shape")
            run["split_isolation_valid"] = False
            for row in run["records"]:
                if row["status"] == "verified" and _json(_physical_identity(row["case"])) == key:
                    # Raw timing evidence can remain numerically valid while
                    # the declared experimental split is invalid. Preserve the
                    # samples and summaries, but reject this scoring record.
                    row.update(status="rejected", split_isolation_valid=False,
                        raw_evidence_verification_passed=True, eligible_for_error_acceptance=False)
                    row["errors"].append("development/held_out split conflict at this physical shape")
    return conflicts


def analyze_runs(measurements: list[str | Path]) -> dict:
    """Analyze captured runs only. Invalid evidence is preserved and rejected."""
    paths = [Path(path).resolve() for path in measurements]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("measurements must contain one or more distinct run paths")
    runs = []
    for path in paths:
        manifest_path = path.parent / "manifest.json"
        run = {"run_id": path.parent.name, "status": "rejected", "errors": [], "records": [],
            "record_inventory": [], "missing_jobs": [], "manifest_binding_valid": False,
            "split_isolation_valid": True,
            "source": {"measurements": _source(path), "manifest": _source(manifest_path)}}
        runs.append(run)
        try:
            records = _load(path)
            if not isinstance(records, list):
                raise ValueError("measurements must be a JSON array")
            run["record_inventory"] = [_inventory(row, path.parent) for row in records]
            manifest = _load(manifest_path)
            args, jobs = _protocol(manifest)
            complete = build_plan(manifest["suite"], args.threads)
            run.update(protocol=manifest["protocol"], split=manifest["split"],
                benchmark_source_sha256=manifest["benchmark_source_sha256"],
                model_contract_sha256=manifest["model_contract_sha256"],
                runtime_sha256=manifest["runtime_sha256"],
                manifest_canonical_sha256=hashlib.sha256(_json(manifest).encode()).hexdigest(),
                records_canonical_sha256=hashlib.sha256(_json(records).encode()).hexdigest(),
                planned_jobs=len(jobs), completed_records=len(records),
                excluded_by_explicit_split_case_ids=[case["id"] for case in complete
                    if manifest["split"] is not None and case["metadata"]["split"] != manifest["split"]],
                declared_physical_shape_splits=[{"case_id": case["id"], "physical_identity": _physical_identity(case),
                    "split": case["metadata"]["split"]} for case in complete])
            for index, row in enumerate(records):
                try:
                    if index >= len(jobs):
                        raise ValueError("record has no declared manifest job")
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    repeat, selected = jobs[index]
                    run["records"].append(_verify_record(row, path.parent, manifest, args, selected, repeat))
                except (ValueError, OSError, SpecError, TypeError, KeyError) as exc:
                    run["records"].append({**_inventory(row, path.parent), "status": "rejected", "errors": [str(exc)],
                        "raw_samples_verified": False, "record_index": index,
                        "captured_status": row.get("status") if isinstance(row, dict) else None})
            run["missing_jobs"] = [{"repeat": repeat, "case_id": selected["id"], "shape": _shape(selected, args.compute_mode, manifest["protocol"])}
                for repeat, selected in jobs[len(records):]]
            _same(manifest.get("completed_jobs"), len(records), "manifest completed jobs")
            _same(manifest.get("measured_jobs"), sum(isinstance(row, dict) and row.get("status") == "measured" for row in records), "manifest measured jobs")
            run["manifest_binding_valid"] = True
            if run["missing_jobs"]:
                run["errors"].append("declared manifest jobs are missing from measurements")
            if any(row["status"] != "verified" for row in run["records"]):
                run["errors"].append("one or more records failed evidence verification")
            run["status"] = "verified" if len(records) == len(jobs) and not run["errors"] and all(row["status"] == "verified" for row in run["records"]) else "rejected"
        except (ValueError, OSError, SpecError, TypeError, KeyError) as exc:
            run["errors"].append(str(exc))
            if isinstance(exc, SplitConflict):
                run["split_isolation_valid"] = False
                run["local_split_conflicts"] = [exc.conflict]
    split_conflicts = _enforce_split_isolation(runs)
    groups = _groups(runs)
    return {"schema_version": SCHEMA, "runs": runs, "groups": groups,
        "verification_passed": all(run["status"] == "verified" for run in runs),
        "verified_record_count": sum(row["status"] == "verified" for run in runs for row in run["records"]),
        "rejected_record_count": sum(row["status"] == "rejected" for run in runs for row in run["records"]),
        "unprocessed_record_count": sum(len(run["record_inventory"])-len(run["records"]) for run in runs),
        "matched_graph_plan_comparisons": _mode_comparisons(runs),
        "development_held_out_split_conflicts": split_conflicts,
        "independent_repeat_error_statistics_by_split_and_mode": _split_error_statistics(groups),
        "analysis_policy": {"quality_filtering": False, "samples_pooled": False, "fitted_coefficients": False,
            "same_batch_stage_sum_vs_serial_is_accuracy": False, "joint_is_accuracy_oracle": False,
            "cost_table_exported": False, "independent_resident_cost_calibration_claimed": False,
            "split_identity_axes": ["config", "threads", "batch", "tokens", "visible_history"],
            "phase_mode_warmup_affinity_cannot_create_held_out_geometry": True,
            "prediction_rule": "repeat 0 sum of individual stage medians; validation against repeat 1 serial wall median",
            "split_scope": "predeclared shape metadata bound to captured suite; chronological implementation freeze is a separate experiment provenance requirement"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON report; never overwrite")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    report = analyze_runs(args.measurements)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    print(f"Verified {report['verified_record_count']} records; rejected {report['rejected_record_count']}; report: {args.output}")
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, SpecError) as exc:
        print(f"analyze_dense_paired: {exc}", file=sys.stderr)
        raise SystemExit(1)
