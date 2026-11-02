"""Verify a completed transform calibration without invoking hardware."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as a
import run_native as r


def verify(experiment):
    experiment = Path(experiment).resolve()
    suite = r.validate_suite(r.document(experiment / "suite.json"))
    manifest = r.document(experiment / "manifest.json")
    rows = r.document(experiment / "measurements.json")
    expected = {row["id"]: row for split in ("profile", "heldout") for row in r.jobs(suite, split)}
    if not isinstance(rows, list) or len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("measurement row coverage differs from the declared independent-process plan")
    if not manifest.get("assets_unchanged") or manifest["inventory_before"] != manifest["inventory_after"]:
        raise ValueError("source/binary/runtime inventory changed during experiment")
    for original, snapshot in manifest["source_snapshots"].items():
        path = (experiment / snapshot["path"]).resolve()
        if not path.is_relative_to(experiment / "source") or a.sha(path) != snapshot["sha256"] or snapshot["sha256"] != manifest["inventory_before"][original]:
            raise ValueError("executed source snapshot differs from its pre-run identity")
    for source in (Path(a.__file__).resolve(), Path(r.__file__).resolve()):
        if a.sha(source) != manifest["inventory_before"].get(str(source)):
            raise ValueError("current statistical/contract code differs from the exact executed implementation")
    model_path = experiment / "prediction-model.json"
    frozen_sha = manifest["model_frozen_before_heldout_sha256"]
    if a.sha(model_path) != frozen_sha:
        raise ValueError("frozen profile model hash mismatch")
    freeze_time = datetime.fromisoformat(manifest["model_frozen_utc"])
    rebuilt = []
    for saved in rows:
        plan = expected[saved["id"]]
        if any(saved[key] != plan[key] for key in ("case", "split", "seed", "repeat")):
            raise ValueError("saved measurement declaration mismatch")
        raw_path = experiment / "raw" / (saved["id"] + ".json")
        if str(raw_path) != saved["raw_path"] or a.sha(raw_path) != saved["raw_sha256"]:
            raise ValueError("raw evidence path/hash mismatch")
        raw = r.document(raw_path)
        command = r.command(saved["command"][0], plan["case"], suite["protocol"], plan["seed"])
        if raw["command"] != command or saved["command"] != command:
            raise ValueError("native command differs from the declared geometry/protocol")
        if raw["observed_child_flags"] != manifest["observed_child_flags"]:
            raise ValueError("child runtime environment flags differ")
        if saved["split"] == "heldout":
            if raw.get("frozen_model_sha256_at_start") != frozen_sha or saved.get("frozen_model_sha256_at_start") != frozen_sha:
                raise ValueError("heldout child did not bind the frozen profile model")
            if datetime.fromisoformat(raw["started_utc"]) < freeze_time:
                raise ValueError("heldout started before the model was frozen")
        if saved["status"] == "measured":
            if raw.get("returncode") != 0 or raw.get("timed_out") or raw.get("validation_error"):
                raise ValueError("failed native process was saved as a measured price")
            payload = json.loads(raw["stdout"])
            r.validate_native(plan, payload, suite)
            for key, native_key in (("metrics", "wall_ns"), ("checks", "checks"), ("workspaces", "workspaces"), ("ports", "ports"), ("boundary", "boundary")):
                if saved[key] != payload[native_key]:
                    raise ValueError("saved evidence differs from raw native output: " + key)
            if not isinstance(raw.get("child_pid"), int) or saved["child_pid"] != raw["child_pid"]:
                raise ValueError("independent native child PID evidence missing")
        elif saved["status"] != "failed" or not raw.get("validation_error"):
            raise ValueError("unknown or unsupported saved status")
        rebuilt.append(saved)
    declared = {case["id"]: r.calibration_recipes(case) for case in suite["cases"]}
    aliases = {case["id"]: r.observed_aliases(case) for case in suite["cases"]}
    model = a.freeze_model(rebuilt, declared, aliases)
    if model != r.document(model_path):
        raise ValueError("profile-only component model cannot be reconstructed")
    comparisons = {case["id"]: {strategy: ("weights_resident_inputs_resident_output_dram" if strategy.split("@", 1)[0].endswith("/resident") else "three_gpu_shard_weights_reloaded_then_output_dram")
                                        + (";dispatch_model" if "@" in strategy else ";ideal_model")
                               for strategy in r.calibration_recipes(case)} for case in suite["cases"]}
    report = a.evaluate(rebuilt, model, suite["quality"], suite["bootstrap"],
                        scope="真实 F32 FFN 中间维 CPU/GPU 分片并行与 CPU 汇合，以及 post-RoPE Attention QK/softmax/PV 异构串行链。",
                        frozen_before_heldout=True, comparison_groups=comparisons)
    report_path = experiment / "analysis" / "report.json"
    saved_report = r.document(report_path)
    for key, value in report.items():
        if key == "unknown_or_uncovered":
            if saved_report[key][:len(value)] != value:
                raise ValueError("baseline unknown/coverage boundary statements were changed")
        elif saved_report[key] != value:
            raise ValueError("reconstructed statistical report differs: " + key)
    if a.sha(report_path) != manifest["report_sha256"]:
        raise ValueError("completed report hash mismatch")
    return {"schema": "execution-transform-verification-0.1", "verified": True,
            "measurement_rows": len(rows), "failed_rows": len(report["failed_native_rows"]),
            "profile_only_model_reconstructed": True, "heldout_frozen_hash_verified": True,
            "component_predictions": len(report["results"]),
            "accepted_predictions": sum(item["calibrated"] for item in report["results"]),
            "numeric_execution_valid": report["numeric_execution_valid"],
            "gpu_private_workspace_known": False, "physical_capacity_validated": False,
            "report_sha256": a.sha(report_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("verification output must be a new file")
        result = verify(args.experiment)
        a.dump(args.output, result)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("execution calibration verification: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
