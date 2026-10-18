"""Freeze and evaluate one numeric resource-response model, without hardware.

Each calibration row is one run/steady-phase median. Raw invocation samples,
transition coverage and stability remain separate evidence. The model never
uses target-inclusive device telemetry as an explanatory variable.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "profiling", ROOT / "mapping", ROOT / "modeling", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from .functional import (CalibrationRow, FitConfig, Hardware, Operation,
                         ResourceFunctional, ResourceState, evaluate_heldout, fit_model)
from .schedule import phase_for_interval
from state_calibration import run
from state_calibration.telemetry import NativeClock

SCHEMA = "unified-resource-fit-0.1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tick_ns(tick, clock):
    if type(tick) is not int or tick < 0 or type(clock.get("frequency_hz")) is not int or clock["frequency_hz"] <= 0:
        raise ValueError("invalid monotonic timestamp/frequency")
    return tick * 1_000_000_000 // clock["frequency_hz"]


def median_exact(values):
    values = sorted(Fraction(value) for value in values)
    if not values:
        raise ValueError("cannot summarize an empty phase")
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def _jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verified_artifacts(directory, record):
    artifacts = record.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("missing raw artifact bindings")
    root = directory.resolve()
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or run.sha(path) != expected:
            raise ValueError("raw artifact is outside its job or changed: " + name)


def _capacity(records, field):
    found = set(records)
    if len(found) != 1 or any(type(value) is not int or value <= 0 for value in found):
        raise ValueError("missing/inconsistent observed " + field)
    return next(iter(found))


def calibration_rows(campaign, split):
    """Reconstruct all declared runs, numerical results and phase assignments.

    No failed run, empty phase, or altered raw artifact is silently excluded.
    Transition samples remain in coverage receipts but have no fabricated state.
    """
    from .collect import SCHEMA as CAMPAIGN_SCHEMA, extract_observations, inventory_digest, verify_native_timeline
    root = Path(campaign).resolve()
    manifest = run.read(root / "manifest.json")
    if manifest.get("schema") != CAMPAIGN_SCHEMA or inventory_digest(manifest["sources"]) != manifest["source_inventory_sha256"]:
        raise ValueError("invalid campaign/source inventory")
    run.verify_sources(manifest)
    jobs = [job for job in manifest["jobs"] if job["split"] == split]
    records = run.read(root / (split + "-rows.json"))
    if [row["job"] for row in records] != jobs or not jobs:
        raise ValueError("declared split has missing, duplicated, or reordered runs")
    result, summaries, host_totals, gpu_totals = [], [], [], []
    for record in records:
        job = record["job"]
        directory = root / "jobs" / job["id"]
        if record != run.read(directory / "execution.json") or record.get("status") != "measured" or record.get("cleanup_verified") is not True:
            raise ValueError("trajectory failed, changed or lacks cleanup evidence")
        if record.get("source_inventory_sha256") != manifest["source_inventory_sha256"]:
            raise ValueError("trajectory source binding mismatch")
        _verified_artifacts(directory, record)
        binding = manifest["programs"][job["program_key"]]
        program = run.read(binding["path"])
        native = run.read(directory / "main" / "result.json")
        clock = native["clock"]
        target = run.choose_target(program)
        verify_native_timeline(native, program, process_id=record["process_id"], target=target,
                               mode=job["mode"], clock=clock)
        run.validate_numeric(native, run.read(binding["reference"]))
        timeline = run.read(directory / "timeline.json")
        observations = run.read(directory / "observations.json")
        if timeline["schedule"] != job["schedule"] or observations != extract_observations(native, timeline["events"], job=job):
            raise ValueError("phase assignments differ from independently reconstructed samples")
        gpu_totals.extend(sample["gpu"]["values"]["memory_total_bytes"] for sample in _jsonl(directory / "telemetry.jsonl")
                          if "memory_total_bytes" in sample.get("gpu", {}).get("values", {}))
        host_totals.extend(sample["snapshots"]["host"]["host_memory"]["values"]["total_physical_bytes"]
                           for sample in _jsonl(directory / "counters.jsonl")
                           if "total_physical_bytes" in sample.get("snapshots", {}).get("host", {}).get("host_memory", {}).get("values", {}))
        op = Operation("attention_pv", "gpu", 1_000_000,
                       constants=(("physical", json.dumps(program["physical"], sort_keys=True)),
                                  ("graph", digest(program["graph"])), ("actions", digest(program["actions"])),
                                  ("mode", job["mode"]), ("boundary", manifest["protocol"]["execution_boundary"]),
                                  ("native_binary_sha256", manifest["sources"][manifest["binary"]]),
                                  ("reference_unit", "one_million_normalized_reference_ns_not_a_measured_baseline")))
        targets = [item for item in observations if item["metric"] == "target"]
        for event in timeline["events"]:
            items = [item for item in targets if item["phase"]["kind"] == "steady" and item["phase"]["phase_index"] == event["phase_index"]]
            if not items:
                raise ValueError("declared phase has no complete target invocation")
            # Assign the vector from exogenous resource commands, never telemetry.
            control = event["requested_resources"]
            state = ResourceState(int(control["cpu_replay"]), int(control["gpu_replay"]),
                                  control["host_memory_bytes"], control["device_memory_bytes"])
            elapsed = median_exact([item["elapsed_ns"] for item in items])
            start = tick_ns(event["steady_begin_tick"], clock)
            end = tick_ns(event["steady_end_tick"], clock)
            row_id = f"{job['id']}-phase-{event['phase_index']}"
            result.append(CalibrationRow(row_id, split, job["id"], job["seed"], start, elapsed, end,
                                         op, state, manifest["source_inventory_sha256"], manifest["protocol_id"],
                                         clock["clock_id"], "offered_load_controlled", True))
            samples = [item["elapsed_ns"] for item in items]
            summaries.append({"row_id": row_id, "run_id": job["id"], "state": state.to_dict(),
                              "sample_count": len(samples), "median_ns": str(elapsed), "within_process_cv": cv(samples),
                              "within_process_cv_passed": cv(samples) <= .15})
    hardware = Hardware(manifest["source_inventory_sha256"], _capacity(host_totals, "DRAM total"), _capacity(gpu_totals, "VRAM total"))
    coverage = {"run_count": len(records), "phase_count": len(summaries), "rows": summaries,
                "target_samples": sum(row["summary"]["observations"] // 2 for row in records),
                "steady_target_samples": sum(row["sample_count"] for row in summaries),
                "phase_aggregation": "equal weight per run/phase; exact pooled invocation median",
                "transition_samples_retained_but_unpriced": True,
                "total_telemetry_used_as_predictor": False,
                "physical_capacity_claimed": False, "cross_transition_hardware_accuracy_claimed": False}
    coverage["unpriced_target_samples"] = coverage["target_samples"] - coverage["steady_target_samples"]
    return tuple(result), coverage, hardware


def freeze(campaign, output):
    root = Path(campaign).resolve()
    if (root / "heldout-rows.json").exists() or (root / "heldout-barrier.json").exists():
        raise ValueError("cannot fit/refit after heldout collection has begun")
    manifest = run.read(root / "manifest.json")
    rows, coverage, hardware = calibration_rows(root, "train")
    config = FitConfig.from_dict(manifest["protocol"]["fit_config"])
    clock = NativeClock()
    frozen_at_ns = tick_ns(clock.now(), clock.metadata())
    model = fit_model(rows, hardware, frozen_at_ns, config)
    model = replace(model, frozen_at_ns=tick_ns(clock.now(), clock.metadata()))
    artifact = {"schema": SCHEMA, "model": model.to_dict(), "model_sha256": model.model_sha256,
                "bindings": {"manifest_sha256": run.sha(root / "manifest.json"),
                             "source_inventory_sha256": manifest["source_inventory_sha256"],
                             "train_rows_sha256": run.sha(root / "train-rows.json"),
                             "calibration_rows_sha256": digest([row.to_dict() for row in rows])},
                "calibration_rows": [row.to_dict() for row in rows], "training_coverage": coverage,
                "fit_config_declared_before_collection": True, "heldout_used_for_fit": False,
                "hardware_accuracy_accepted": False}
    run.save(output, artifact)
    return artifact


def load_frozen(campaign, path, verify_training=True):
    """Verify a real frozen training-only artifact, not just an existing file."""
    root = Path(campaign).resolve()
    artifact = run.read(path)
    if artifact.get("schema") != SCHEMA or artifact.get("heldout_used_for_fit") is not False:
        raise ValueError("not a frozen unified-resource training artifact")
    model = ResourceFunctional.from_dict(artifact["model"])
    manifest = run.read(root / "manifest.json")
    bindings = artifact["bindings"]
    if (artifact["model_sha256"] != model.model_sha256
            or bindings["manifest_sha256"] != run.sha(root / "manifest.json")
            or bindings["source_inventory_sha256"] != manifest["source_inventory_sha256"]
            or bindings["train_rows_sha256"] != run.sha(root / "train-rows.json")
            or model.source_sha256 != manifest["source_inventory_sha256"]
            or model.config.to_dict() != manifest["protocol"]["fit_config"]):
        raise ValueError("frozen model bindings differ from the training campaign")
    rows = tuple(CalibrationRow.from_dict(row) for row in artifact["calibration_rows"])
    if digest([row.to_dict() for row in rows]) != bindings["calibration_rows_sha256"]:
        raise ValueError("altered calibration rows")
    if verify_training:
        rebuilt_rows, coverage, hardware = calibration_rows(root, "train")
        if rows != rebuilt_rows or coverage != artifact["training_coverage"] or hardware != model.hardware:
            raise ValueError("model source rows differ from reconstructed training evidence")
    # Refit from frozen training inputs only as an integrity check; no heldout input.
    rebuilt = fit_model(rows, model.hardware, model.frozen_at_ns, model.config)
    if rebuilt != model:
        raise ValueError("frozen coefficients or metadata differ from their training-only reconstruction")
    return artifact, model


def evaluate(campaign, model_path, output):
    root = Path(campaign).resolve()
    artifact, model = load_frozen(root, model_path)
    barrier = run.read(root / "heldout-barrier.json")
    if barrier["model_sha256"] != run.sha(model_path) or barrier["train_rows_sha256"] != artifact["bindings"]["train_rows_sha256"]:
        raise ValueError("heldout barrier does not bind this frozen training model")
    rows, coverage, hardware = calibration_rows(root, "heldout")
    if hardware != model.hardware:
        raise ValueError("heldout hardware capacities differ from training")
    report = evaluate_heldout(model, rows)
    errors = [float(Fraction(row["relative_error"])) for row in report["rows"] if row["status"] == "predicted"]
    process_groups = {}
    for row in rows:
        process_groups.setdefault(json.dumps(row.state.to_dict(), sort_keys=True), []).append(float(row.elapsed_ns))
    stability = [{"state": json.loads(key), "process_count": len(values), "between_process_cv": cv(values),
                  "passed": len(values) >= 3 and cv(values) <= .20} for key, values in sorted(process_groups.items())]
    report.update(schema="unified-resource-evaluation-0.1", coverage=coverage,
                  training_coverage=artifact["training_coverage"], heldout_process_stability=stability,
                  median_relative_error_float=statistics.median(errors) if errors else None,
                  worst_relative_error_float=max(errors) if errors else None,
                  error_only_20_percent_count=sum(error <= .20 for error in errors),
                  all_quality_gates_accepted=False,
                  acceptance_scope="Diagnostic response surface; original run-count/stability and observation gates are not waived",
                  cross_transition_hardware_accuracy_claimed=False, physical_capacity_claimed=False,
                  raw_manifest_sha256=run.sha(root / "manifest.json"), frozen_artifact_sha256=run.sha(model_path))
    run.save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate", "verify"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        artifact = freeze(args.campaign, args.model)
        print(json.dumps({"model_sha256": artifact["model_sha256"], "training_phase_rows": len(artifact["calibration_rows"]),
                          "hardware_accuracy_accepted": False}), flush=True)
    elif args.command == "verify":
        artifact, model = load_frozen(args.campaign, args.model)
        print(json.dumps({"verified": True, "model_sha256": model.model_sha256, "native_launches": 0}), flush=True)
    else:
        if args.out is None:
            parser.error("evaluate requires --out")
        report = evaluate(args.campaign, args.model, args.out)
        print(json.dumps({key: report[key] for key in ("row_count", "predicted_count", "median_relative_error_float",
                          "worst_relative_error_float", "all_quality_gates_accepted")}), flush=True)


if __name__ == "__main__":
    main()
