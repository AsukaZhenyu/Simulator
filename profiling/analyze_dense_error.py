"""Read verified Dense runs and report additive timing error without fitting costs.

Each exact shape is checked with the existing exporter. High-CV observations are
retained for analysis; failed, missing or invalid evidence stays visible as a
rejected group. Samples from different runs or shapes are never pooled. Joint
execution is a comparison measurement, not an oracle for a deployment workload.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from export_dense_layer import export_cost_table, validate_model_layer
from llm_infer_model.m2.dense_contract import STAGES, build_plan, validate_case
from llm_infer_model.m2.spec import load_layer
from llm_infer_model.tensor.spec import SpecError


SCHEMA = "dense_error_analysis_v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"{path}: nonfinite JSON number {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def _source(path: Path) -> dict:
    result = {"path": str(path)}
    if path.is_file():
        result["sha256"] = _sha(path)
    else:
        result["error"] = "file is missing"
    return result


def _shape(case: dict, threads: int) -> dict:
    return {"config": case["config"], "threads": threads, "phase": case["phase"],
            "batch": case["batch"], "tokens": case["tokens"], "visible_history": case["visible_history"]}


def _raw_inventory(row: Any, directory: Path) -> dict:
    """Retain evidence references, including failed cases, without trusting them."""
    if not isinstance(row, dict):
        return {"status": "invalid_record", "error": "record must be an object", "record": row}
    result = {key: row.get(key) for key in ("case_id", "status", "error", "quality_flags", "raw_output")}
    result["record_sha256"] = hashlib.sha256(_json(row).encode("utf-8")).hexdigest()
    raw_name = row.get("raw_output")
    if not isinstance(raw_name, str) or not raw_name or Path(raw_name).is_absolute():
        result["raw_evidence_error"] = "raw_output must be a relative run-directory path"
        return result
    raw = (directory / raw_name).resolve()
    if not raw.is_relative_to(directory):
        result["raw_evidence_error"] = "raw_output escapes the measurement run directory"
    elif not raw.is_file():
        result["raw_evidence_error"] = "raw output is missing"
    else:
        result["raw_sha256"] = _sha(raw)
    return result


def _estimate(values: list[float]) -> dict:
    mean = statistics.mean(values)
    median = statistics.median(values)
    return {"count": len(values), "values": values, "min": min(values), "max": max(values),
            "median": median, "mean": mean,
            "population_cv_of_run_estimates": statistics.pstdev(values) / mean if mean else None,
            "range_over_median": (max(values) - min(values)) / median if median else None}


def _repeat_report(runs: list[dict]) -> list[dict]:
    by_shape: dict[str, list[dict]] = {}
    for run in runs:
        for group in run["groups"]:
            if group["status"] != "verified":
                continue
            by_shape.setdefault(_json(group["shape"]), []).append({
                "run_id": run["run_id"], "measurements": run["source"]["measurements"]["path"],
                "estimate": group["estimate"], "stages": group["stages"],
                "protocol": run["protocol"], "runtime_sha256": run["runtime_sha256"]})
    result = []
    for key, observations in sorted(by_shape.items()):
        pairs = []
        for left, right in itertools.combinations(observations, 2):
            differences = []
            for label in ("predicted_serial_stage_sum_ns", "observed_joint_block_median_ns"):
                a, b = left["estimate"][label], right["estimate"][label]
                differences.append({"quantity": label, "left_ns": a, "right_ns": b,
                                    "signed_change_ns": b-a, "signed_change_relative_to_left": (b-a)/a})
            same_protocol = _json(left["protocol"]) == _json(right["protocol"])
            same_runtime = _json(left["runtime_sha256"]) == _json(right["runtime_sha256"])
            pairs.append({"left_run": left["measurements"], "right_run": right["measurements"],
                "exact_protocol_equal": same_protocol, "runtime_hash_inventory_equal": same_runtime,
                "same_protocol_runtime_repeat": same_protocol and same_runtime,
                "differences": differences,
                "signed_relative_error_change": right["estimate"]["signed_relative_error"] - left["estimate"]["signed_relative_error"],
                "stage_median_changes_relative_to_left": {kind:
                    (right["stages"][kind]["summary"]["median_ns"] - left["stages"][kind]["summary"]["median_ns"])
                    / left["stages"][kind]["summary"]["median_ns"] for kind in STAGES}})
        result.append({"shape": json.loads(key), "run_count": len(observations),
            "run_observations": [{name: row[name] for name in ("run_id", "measurements", "estimate")}
                                 for row in observations],
            "descriptive_stage_sum_run_estimates": _estimate([row["estimate"]["predicted_serial_stage_sum_ns"] for row in observations]),
            "descriptive_joint_run_estimates": _estimate([row["estimate"]["observed_joint_block_median_ns"] for row in observations]),
            "pairwise_run_comparisons": pairs, "samples_pooled": False,
            "spread_scope": "per-run estimates at one exact shape; protocol/runtime differences are disclosed per pair"})
    return result


def analyze_runs(measurements: list[str | Path], layers: list[str | Path]) -> dict:
    """Return a complete analysis document; never mutate inputs or run hardware.

    Layer files select the physical contract only. No cost table is written. An
    analytical allow_unstable=True verifies the raw evidence while deliberately
    retaining high-CV records, and does not certify their suitability as costs.
    """
    paths = [Path(path).resolve() for path in measurements]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("measurements must name one or more distinct run files")
    layer_paths = [Path(path).resolve() for path in layers]
    if not layer_paths or len(set(layer_paths)) != len(layer_paths):
        raise ValueError("layers must name one or more distinct physical layer files")
    contracts: dict[str, Any] = {}
    layer_sources = []
    for path in layer_paths:
        layer = load_layer(path)
        validate_model_layer(layer)
        key = _json({"config": layer.physical.config, "threads": layer.physical.threads})
        if key in contracts:
            raise ValueError("multiple layer files define the same config and threads")
        contracts[key] = layer
        layer_sources.append(_source(path))
    runs = []
    for path in paths:
        manifest_path = path.parent / "manifest.json"
        run = {"run_id": path.parent.name, "status": "rejected", "source": {
            "measurements": _source(path), "manifest": _source(manifest_path)},
            "groups": [], "unresolved_records": [], "record_inventory": [], "errors": []}
        runs.append(run)
        try:
            records = _load(path)
            if not isinstance(records, list):
                raise ValueError("measurements must be a JSON array")
            run["record_inventory"] = [_raw_inventory(row, path.parent) for row in records]
            manifest = _load(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            run["protocol"] = manifest.get("protocol", {})
            if not isinstance(run["protocol"], dict) or not isinstance(manifest.get("software"), dict):
                raise ValueError("manifest protocol and software must be objects")
            run["runtime_sha256"] = manifest["software"].get("runtime_sha256", {})
            threads = run["protocol"].get("threads")
            plan = build_plan(manifest.get("suite"), threads)
            run["planned_case_count"] = len(plan)
            run["record_count"] = len(records)
            groups: dict[str, dict] = {}
            for case in plan:
                shape = _shape(case, threads)
                groups.setdefault(_json(shape), {"shape": shape, "rows": [], "expected_case_ids": []})["expected_case_ids"].append(case["id"])
            for index, row in enumerate(records):
                try:
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    case = validate_case(row.get("case"))
                    shape = _shape(case, case.get("threads", threads))
                    groups.setdefault(_json(shape), {"shape": shape, "rows": [], "expected_case_ids": []})["rows"].append(row)
                except (ValueError, SpecError) as exc:
                    run["unresolved_records"].append({"record_index": index, "case_id": row.get("case_id") if isinstance(row, dict) else None,
                                                       "error": str(exc)})
            for key, selected in sorted(groups.items()):
                rows = selected["rows"]
                group = {"shape": selected["shape"], "status": "rejected", "errors": [],
                    "expected_case_ids": selected["expected_case_ids"],
                    "records": [_raw_inventory(row, path.parent) for row in rows],
                    "missing_case_ids": sorted(set(selected["expected_case_ids"]) - {row.get("case_id") for row in rows}),
                    "raw_samples_verified": False}
                run["groups"].append(group)
                try:
                    if group["missing_case_ids"]:
                        raise ValueError(f"manifest cases missing from records: {group['missing_case_ids']}")
                    layer_key = _json({name: selected["shape"][name] for name in ("config", "threads")})
                    if layer_key not in contracts:
                        raise ValueError("no layer exactly matches config and threads")
                    _, evidence = export_cost_table(contracts[layer_key], rows, raw_directory=path.parent,
                        manifest=manifest, allow_unstable=True,
                        source_evidence={"measurements": run["source"]["measurements"], "manifest": run["source"]["manifest"]})
                    if len(evidence["composition"]) != 1:
                        raise ValueError("exact shape group must produce exactly one composition comparison")
                    group.update(status="verified", raw_samples_verified=True, estimate=evidence["composition"][0],
                        records_canonical_sha256=evidence["records_canonical_sha256"],
                        manifest_canonical_sha256=evidence["manifest_canonical_sha256"],
                        stages={row["case"]["kind"]: {"case_id": row["case_id"], "summary": row["summary"],
                            "wall_ns": row["native"]["wall_ns"], "quality_flags": row["quality_flags"],
                            "cv_gt_0_1": row["summary"]["cv"] > 0.1,
                            "raw_sha256": next(item["raw_sha256"] for item in evidence["entries"] if item["case_id"] == row["case_id"])}
                            for row in rows},
                        suitable_as_stable_costs=all(not row["quality_flags"] and row["summary"]["cv"] <= 0.1 for row in rows))
                except (ValueError, OSError, SpecError, TypeError, KeyError) as exc:
                    group["errors"].append(str(exc))
            run["status"] = "verified" if run["groups"] and not run["unresolved_records"] and all(group["status"] == "verified" for group in run["groups"]) else "rejected"
        except (ValueError, OSError, SpecError, TypeError, KeyError) as exc:
            run["errors"].append(str(exc))
    verified = sum(group["status"] == "verified" for run in runs for group in run["groups"])
    rejected = sum(group["status"] == "rejected" for run in runs for group in run["groups"])
    return {"schema_version": SCHEMA, "source_layers": layer_sources, "runs": runs,
        "verification_passed": all(run["status"] == "verified" for run in runs),
        "verified_group_count": verified, "rejected_group_count": rejected,
        "exact_shape_repeat_comparisons": _repeat_report(runs),
        "analysis_policy": {"quality_filtering": False, "high_cv_records_retained": True,
            "exporter_allow_unstable_for_evidence_verification": True, "samples_pooled": False,
            "fitted_coefficients": False, "joint_is_oracle": False, "calibration_claimed": False,
            "comparison_scope": "isolated stage medians versus separately measured declared joint graph",
            "statistical_claim": "descriptive observations only; no inference from one run to unmeasured shapes"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, action="append", required=True,
                        help="run measurements.json; repeat for independent runs")
    parser.add_argument("--layer", type=Path, action="append", required=True,
                        help="full physical layer; repeat for distinct config/threads")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    document = analyze_runs(args.measurements, args.layer)
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    print(f"Verified {document['verified_group_count']} exact groups; rejected {document['rejected_group_count']}; report: {args.output}")
    return 0 if document["verification_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, SpecError) as exc:
        print(f"analyze_dense_error: {exc}", file=sys.stderr)
        raise SystemExit(1)
