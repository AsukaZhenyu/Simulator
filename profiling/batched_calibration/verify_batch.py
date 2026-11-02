"""Offline verification of raw completed-work batch evidence and frozen costs.

This verifier never executes hardware. Source, build and raw-process identities
are checked before reconstructing normalization, the profile-only model and
the held-out report. Repeated-invocation means never become isolated latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent
sys.path[:0] = [str(PROFILING / "execution_calibration"), str(PROFILING)]
import analysis as a
import run_dual as base

_spec = importlib.util.spec_from_file_location("verified_batch_runner", HERE / "run.py")
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    return value


def verify(experiment):
    experiment = Path(experiment).resolve()
    suite = batch.validate_suite(_object(base.document(experiment / "suite.json"), "suite"))
    manifest = _object(base.document(experiment / "manifest.json"), "manifest")
    if (manifest.get("schema") != "execution-transform-batch-manifest-0.1"
            or manifest.get("timing_unit") != "completed_work_batch_total"
            or manifest.get("derived_metric_unit") != "rounded_mean_completed_invocation_within_batch_ns"
            or manifest.get("normalization") != "round(batch_total_ns / batch_repetitions); raw totals retained"
            or manifest.get("sample_filtering") != "none"):
        raise ValueError("batch manifest timing/normalization contract mismatch")
    if not manifest.get("assets_unchanged") or manifest["inventory_before"] != manifest["inventory_after"]:
        raise ValueError("source/binary/runtime inventory changed during collection")
    if manifest.get("gpu_capacity_proved") is not False:
        raise ValueError("unknown CUDA workspace cannot prove capacity")
    inventory = manifest["inventory_before"]
    snapshots = manifest["source_snapshots"]
    for original, record in snapshots.items():
        archived = (experiment / record["path"]).resolve()
        if (not archived.is_relative_to(experiment / "source") or a.sha(archived) != record["sha256"]
                or inventory.get(original) != record["sha256"]):
            raise ValueError("source snapshot hash/path differs from the executed inventory")
    # Current analysis/validation code must equal the exact executed protocol.
    for module, expected in ((a, PROFILING / "execution_calibration/analysis.py"),
                              (base, PROFILING / "execution_calibration/run_dual.py"),
                              (batch, HERE / "run.py")):
        source = Path(module.__file__).resolve()
        if source != expected.resolve() or a.sha(source) != inventory.get(str(source)):
            raise ValueError("current imported protocol differs from the executed source")
    for original, digest in inventory.items():
        if not Path(original).is_file() or a.sha(original) != digest:
            raise ValueError("bound source/binary/runtime asset missing or changed: " + original)
    rows = base.document(experiment / "measurements.json")
    expected = {row["id"]: row for split in ("profile", "heldout") for row in base.jobs(suite, split)}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)
            or {row["id"] for row in rows} != set(expected)):
        raise ValueError("batch independent-process row coverage mismatch")
    binaries = {row["command"][0] for row in rows}
    if len(binaries) != 1 or next(iter(binaries)) not in inventory:
        raise ValueError("native binary is not uniquely bound to the inventory")
    binary = next(iter(binaries))
    def snapshot_named(name):
        matches = [record for original, record in snapshots.items() if Path(original).name == name]
        if len(matches) != 1:
            raise ValueError("missing/ambiguous native build snapshot: " + name)
        return base.document(experiment / matches[0]["path"])
    build = snapshot_named("compute-batch-build.json")
    generated_manifest = snapshot_named("source-manifest.json")
    if build.get("binary_sha256") != inventory[binary] or build.get("source_manifest") != generated_manifest:
        raise ValueError("native executable is not bound to the archived build/source manifest")
    for field, filename in (("generator_sha256", "generate_probe.py"),
                            ("context_sha256", "compute_context.cpp"),
                            ("helper_sha256", "compute.cpp")):
        matches = [original for original in snapshots if Path(original).name == filename]
        if len(matches) != 1 or generated_manifest.get(field) != inventory[matches[0]]:
            raise ValueError("generated probe semantic baseline/generator hash mismatch: " + field)
    generated = [original for original in snapshots if Path(original).suffix == ".cpp"
                 and Path(original).parent == Path(binary).parent]
    if len(generated) != 1 or generated_manifest.get("generated_sha256") != inventory[generated[0]]:
        raise ValueError("generated native batch source hash mismatch")
    frozen_path = experiment / "prediction-model.json"
    frozen_sha = manifest["model_frozen_before_heldout_sha256"]
    if a.sha(frozen_path) != frozen_sha:
        raise ValueError("frozen component model hash mismatch")
    freeze_time = datetime.fromisoformat(manifest["model_frozen_utc"])
    flags = manifest["observed_child_flags"]
    if flags.get("GGML_CALIBRATION_BATCH_REPETITIONS") != str(suite["batch_repetitions"]):
        raise ValueError("batch repetition environment differs from suite")
    dispatch = base.dispatch_cpu_id(flags, suite["protocol"]["cpu_id"])
    raw_evidence = []
    for row in rows:
        declared = expected[row["id"]]
        if any(row[key] != declared[key] for key in ("case", "split", "seed", "repeat")):
            raise ValueError("row geometry/split/seed/repeat differs from the declared plan")
        path = experiment / "raw" / (row["id"] + ".json")
        if str(path) != row["raw_path"] or a.sha(path) != row["raw_sha256"]:
            raise ValueError("raw process evidence path/hash mismatch")
        record = _object(base.document(path), "raw process")
        command = base.command(binary, row["case"], suite["protocol"], row["seed"])
        if row["command"] != command or record["command"] != command or record["observed_child_flags"] != flags:
            raise ValueError("native command or observed environment binding mismatch")
        if record.get("child_pid") != row.get("child_pid"):
            raise ValueError("independent native PID evidence missing")
        if row["split"] == "heldout":
            if (row.get("frozen_model_sha256_at_start") != frozen_sha
                    or record.get("frozen_model_sha256_at_start") != frozen_sha
                    or datetime.fromisoformat(record["started_utc"]) < freeze_time):
                raise ValueError("heldout child did not bind a previously frozen profile-only model")
        if row["status"] == "measured":
            if type(record.get("child_pid")) is not int:
                raise ValueError("successful native child PID evidence missing")
            if record.get("returncode") != 0 or record.get("timed_out") or record.get("validation_error"):
                raise ValueError("failed native child was saved as measured")
            payload = _object(json.loads(record["stdout"]), "native payload")
            normalized = batch.normalized_payload(payload, row, suite, dispatch)
            if row.get("metrics") != normalized or row.get("raw_batch_metrics") != payload["batch_wall_ns"]:
                raise ValueError("raw batch totals or per-invocation normalization changed")
            if row.get("raw_batch_statistics") != {key: a.stats(values) for key, values in payload["batch_wall_ns"].items()}:
                raise ValueError("raw batch statistics changed")
            if row.get("batch_repetitions") != suite["batch_repetitions"]:
                raise ValueError("measurement row unit/repetitions mismatch")
            for key in ("checks", "workspaces", "ports", "boundary"):
                if row[key] != payload[key]:
                    raise ValueError("saved physical evidence differs from raw payload: " + key)
        elif row["status"] != "failed" or not record.get("validation_error"):
            raise ValueError("unknown row status or missing failure evidence")
        raw_evidence.append(dict(row_id=row["id"], split=row["split"], raw_sha256=row["raw_sha256"],
                                 child_pid=row["child_pid"], status=row["status"]))
    declared = {case["id"]: batch.recipes(case) for case in suite["cases"]}
    aliases = {case["id"]: base.observed_aliases(case) for case in suite["cases"]}
    model = a.freeze_model(rows, declared, aliases)
    if model != base.document(frozen_path):
        raise ValueError("profile-only batch model cannot be reconstructed")
    groups = {case["id"]: {name: ("resident" if name.split("@", 1)[0].endswith("resident") else "preload")
        + (";dispatch_model" if "@" in name else ";ideal_model") for name in batch.recipes(case)} for case in suite["cases"]}
    rebuilt = a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
        scope="F32 compact FFN and post-RoPE Attention; repeated completed-work batch means, explicit CPU/GPU worker contexts.",
        frozen_before_heldout=True, comparison_groups=groups)
    report_path = experiment / "analysis/report.json"
    saved = base.document(report_path)
    if a.sha(report_path) != manifest["report_sha256"]:
        raise ValueError("completed batch report hash mismatch")
    for key, value in rebuilt.items():
        if key == "unknown_or_uncovered":
            if saved[key][:len(value)] != value:
                raise ValueError("baseline unknown-workspace/coverage limitations changed")
        elif saved[key] != value:
            raise ValueError("held-out batch analysis does not reconstruct: " + key)
    if (saved.get("batch_repetitions") != suite["batch_repetitions"]
            or saved.get("raw_timing_unit") != "completed_work_batch_total"
            or saved.get("older_isolated_protocol_replaced") is not False
            or saved.get("gpu_capacity_proved") is not False):
        raise ValueError("batch report unit or acceptance scope changed")
    primary = [result for result in rebuilt["results"] if "@" not in result["strategy"]]
    return dict(schema="execution-batch-verification-0.1", verified=True,
        measurement_rows=len(rows), failed_rows=sum(row["status"] != "measured" for row in rows),
        batch_repetitions=suite["batch_repetitions"], timing_unit=manifest["derived_metric_unit"],
        raw_timing_unit=manifest["timing_unit"], normalization=manifest["normalization"],
        profile_only_model_reconstructed=True, heldout_frozen_hash_verified=True,
        raw_batch_totals_and_normalization_verified=True, source_build_binary_hashes_verified=True,
        numeric_execution_valid=rebuilt["numeric_execution_valid"],
        primary_predictions=len(primary), accepted_primary_predictions=sum(result["calibrated"] for result in primary),
        gpu_private_workspace_known=False, physical_capacity_validated=False,
        report_sha256=a.sha(report_path), verifier_source_sha256=a.sha(__file__), raw_evidence=raw_evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("verification output must be a new file")
        result = verify(args.experiment)
        a.dump(args.output, result)
        print(json.dumps({key: result[key] for key in ("verified", "measurement_rows", "accepted_primary_predictions", "physical_capacity_validated")}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print("batch verification: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
