"""Replay raw provenance, frozen costs and diagnostics without running hardware."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import run as r
from diagnostics import summarize
from trace_evaluate import evaluate as evaluate_trace


def verify(out):
    out = Path(out).resolve()
    manifest = r.legacy.document(out / "manifest.json")
    suite = r.validate_suite(r.legacy.document(out / "suite.json"))
    plan = r.legacy.document(out / "plan.json")
    rows = r.legacy.document(out / "measurements.json")
    for name, key in (("plan.json", "plan_sha256_before_collection"),
                      ("suite.json", "suite_sha256_before_collection"),
                      ("recipes.json", "recipes_sha256_before_collection"),
                      ("prediction-model.json", "model_frozen_before_heldout_sha256"),
                      ("trace-model.json", "trace_model_frozen_before_heldout_sha256"),
                      ("shape-models.json", "shape_models_frozen_before_heldout_sha256")):
        if r.a.sha(out / name) != manifest[key]:
            raise ValueError("frozen asset changed: " + name)
    for original, snap in manifest["source_snapshots"].items():
        if r.a.sha(out / snap["path"]) != snap["sha256"] or manifest["inventory_before"][original] != snap["sha256"]:
            raise ValueError("source snapshot mismatch: " + original)
    if manifest.get("assets_unchanged") is not True or manifest.get("inventory_after") != manifest["inventory_before"]:
        raise ValueError("campaign asset preservation failed")
    r.runtime_manifest.verify_recorded(manifest.get("private_runtime_builds", []), manifest["inventory_before"])
    declared = {case["id"]: r.recipes(case) for case in suite["cases"]}
    if r.legacy.document(out / "recipes.json") != declared:
        raise ValueError("frozen recipes differ from suite protocol")
    binary = Path(manifest["bound_binary"])
    if not binary.is_absolute() or str(binary) not in manifest["inventory_before"]:
        raise ValueError("bound executable absent from inventory")
    expected = [dict(job, command=r.legacy.command(binary, job["case"], suite["protocol"], job["seed"]))
                for split in ("profile", "heldout") for job in r.legacy.jobs(suite, split)]
    if expected != plan or [x["id"] for x in rows] != [x["id"] for x in plan]:
        raise ValueError("declared jobs/process coverage differs")
    dispatch = r.legacy.dispatch_cpu_id(manifest["observed_child_flags"], suite["protocol"]["cpu_id"])
    by_id = {job["id"]: job for job in expected}
    for row in rows:
        job = by_id[row["id"]]
        if any(row.get(key) != value for key, value in job.items()):
            raise ValueError("row immutable planned split/case/seed/repeat/command binding differs")
        if r.a.sha(row["raw_path"]) != row["raw_sha256"]:
            raise ValueError("raw record changed: " + row["id"])
        raw = r.legacy.document(row["raw_path"])
        if raw["command"] != job["command"] or raw["observed_child_flags"] != manifest["observed_child_flags"]:
            raise ValueError("raw command/environment mismatch")
        runtime_hashes = r.runtime_manifest.hash_bindings(manifest.get("private_runtime_builds", []))
        if raw.get("private_runtime_manifest_sha256_at_start", {}) != runtime_hashes or row.get("private_runtime_manifest_sha256_at_start", {}) != runtime_hashes:
            raise ValueError("per-process private strict-F32 runtime manifest binding differs")
        if row["split"] == "heldout":
            for key, mkey in (("frozen_model_sha256_at_start", "model_frozen_before_heldout_sha256"),
                              ("frozen_trace_model_sha256_at_start", "trace_model_frozen_before_heldout_sha256"),
                              ("frozen_shape_models_sha256_at_start", "shape_models_frozen_before_heldout_sha256")):
                if row[key] != manifest[mkey] or raw[key] != manifest[mkey]:
                    raise ValueError("heldout model freeze binding mismatch")
            if raw["started_utc"] < max(manifest["model_frozen_utc"], manifest["trace_model_frozen_utc"], manifest["shape_models_frozen_utc"]):
                raise ValueError("heldout started before all models froze")
        if row["status"] == "measured":
            if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("timed_out") or raw.get("validation_error"):
                raise ValueError("measured row has unsuccessful raw process evidence")
            payload = json.loads(raw["stdout"])
            r.validate_native(row, payload, suite, dispatch)
            if payload["wall_ns"] != row["metrics"] or payload["chain_traces"] != row["chain_traces"] or payload["checks"] != row["checks"]:
                raise ValueError("retained row differs from native raw source")
        elif not raw.get("validation_error"):
            raise ValueError("failed row lacks retained reason")
    model = r.a.freeze_model(rows, declared)
    if model != r.legacy.document(out / "prediction-model.json"):
        raise ValueError("independent model does not replay from profile-only sources")
    profile = [x for x in rows if x["split"] == "profile"]
    try:
        trace = r.trace_model.freeze(profile)
    except (ValueError, TypeError, KeyError) as exc:
        trace = {"schema": "chain-profile-trace-model-unavailable-0.1", "status": "unavailable", "error": str(exc)}
    if trace != r.legacy.document(out / "trace-model.json"):
        raise ValueError("in-chain model does not replay from profile-only sources")
    try:
        shape = r.shape_models.freeze(profile)
    except (ValueError, TypeError, KeyError) as exc:
        shape = {"schema": "chain-shape-models-unavailable-0.1", "status": "unavailable", "error": str(exc)}
    if shape != r.legacy.document(out / "shape-models.json"):
        raise ValueError("full/LOCO models do not replay from profile-only sources")
    shape_report = ({"schema": "chain-shape-models-unavailable-0.1", "status": "unavailable", "error": shape["error"]}
                    if shape.get("status") == "unavailable" else r.shape_models.evaluate(shape, rows, frozen_before_heldout=True))
    if shape_report != r.legacy.document(out / "shape-report.json"):
        raise ValueError("full/LOCO predictions do not replay from frozen profile coefficients")
    if evaluate_trace(rows, trace, suite, r.trace_model) != r.legacy.document(out / "trace-report.json"):
        raise ValueError("in-chain prediction/gate report differs")
    if summarize(rows) != r.legacy.document(out / "diagnostic-summary.json"):
        raise ValueError("raw diagnostic summary differs")
    if r.a.sha(out / "analysis" / "report.json") != manifest["report_sha256"]:
        raise ValueError("independent report hash differs")
    report = r.legacy.document(out / "analysis" / "report.json")
    groups = {case["id"]: {strategy: ("three_weights_reloaded_output_dram" if strategy == "hybrid/preload"
                                      else "resident_inputs_weights_output_dram")
                           for strategy in r.recipes(case)} for case in suite["cases"]}
    expected_report = r.a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
                                  scope=report["scope"], frozen_before_heldout=True, comparison_groups=groups)
    for key in ("results", "rankings", "numeric_execution_valid", "failed_native_rows", "all_declared_strategies_calibrated"):
        if expected_report[key] != report[key]:
            raise ValueError("independent predictions/CV/error/ranking gates do not replay: " + key)
    return {"schema": "chain-calibration-verification-0.1", "verified": True,
            "rows": len(rows), "failed_rows": sum(x["status"] != "measured" for x in rows),
            "heldout_used_for_fitting": False, "timestamp_overhead_subtracted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.experiment), indent=2))
