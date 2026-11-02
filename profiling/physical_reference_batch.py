"""Write-once, offline strict reference adaptation for every effective job."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

from profiling.available_service.calibration import digest, require, utc
from profiling.available_service.reference_supply import validate_reference_supply
from profiling.physical_posttarget_collection import effective_posttarget_jobs
from profiling.physical_reference_adapter import build_job_reference_supply, CPU_RESOURCE, GPU_RESOURCE
from profiling.state_calibration import run


def _receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return _receipt(path)


def export_campaign(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = run.read(root / "manifest.json")
    jobs = effective_posttarget_jobs(root, manifest)
    sources = [_receipt(path) for path in (
        Path(__file__), Path(__file__).with_name("physical_reference_adapter.py"),
        Path(__file__).with_name("physical_posttarget_collection.py"),
        Path(__file__).with_name("physical_job_index.py"),
        Path(__file__).parent / "available_service" / "physical_records.py",
        Path(__file__).parent / "available_service" / "reference_supply.py")]
    rows, failures = [], []
    counts = Counter()
    reason_counts = Counter()
    for number, job in enumerate(jobs, 1):
        execution_path = root / "jobs" / job["id"] / "execution.json"
        evidence_path = output / (job["id"] + "-evidence.json")
        supply_path = output / (job["id"] + "-supply.json")
        try:
            supply, evidence = build_job_reference_supply(execution_path)
            execution = run.read(execution_path)
            require(execution.get("job") == job, "batch/evidence effective job differs")
            require(evidence["job"] == job, "adapter evidence effective job differs")
            actor_names = set(evidence["actors"])
            require(actor_names == set(execution["numerical"]), "actor/numerical receipt inventories differ")
            checked_ports = 0
            for name in sorted(actor_names):
                numerical = execution["numerical"][name]
                require(all(numerical.get(flag) is True for flag in ("passed", "full_graph_equal",
                    "action_trace_equal", "full_port_inventory_equal", "copy_routes_bytes_and_values_equal")),
                    "independent numerical receipt not passed: " + name)
                checked_ports += numerical["independently_checked_ports"]
            sample_rows = [] if supply is None else supply["sample_estimates"]
            if supply is not None:
                validate_reference_supply(supply)
                require(supply["after_reference_used_for_forward_supply"] is False and
                    supply["target_timing_used_to_compute_rates"] is False, "causal supply gate changed")
                counts["jobs_with_supply"] += 1
            else:
                require(job["references"] is False, "declared reference job unexpectedly has no supply")
                counts["jobs_without_references"] += 1
            job_counts = Counter()
            for sample in sample_rows:
                kind = "cpu" if sample["resource"] == CPU_RESOURCE else "gpu" if sample["resource"] == GPU_RESOURCE else "other"
                key = kind + ("_known_samples" if sample["rate"] is not None else "_unknown_samples")
                counts[key] += 1
                job_counts[key] += 1
                reason_counts.update(sample["evidence_issues"])
                if kind == "cpu":
                    require(sample["rate"] is None, "unverified CPU physical service was incorrectly qualified")
            counts["actor_identity_program_clock_and_raw_hash_crosschecks"] += len(actor_names)
            counts["bound_independent_numerical_receipts"] += len(actor_names)
            counts["bound_independently_checked_ports"] += checked_ports
            counts["bound_program_and_oracle_file_hash_crosschecks"] += 1
            counts["posttarget_unloaded_after_jobs"] += execution.get("posttarget_cleanup_applied") is True
            counts["main_capture_qualified"] += evidence["main_capture_qualified"] is True
            counts["main_bare_without_capture"] += job["instrumentation"] == "bare"
            row = dict(job_id=job["id"], split=job["split"], condition=job["condition"],
                status="adapted", actor_count=len(actor_names), independently_checked_ports=checked_ports,
                sample_counts=dict(job_counts), main_capture_qualified=evidence["main_capture_qualified"],
                main_instrumentation=job["instrumentation"],
                after_reference_state_scope=evidence["after_reference_state_scope"],
                evidence=_write(evidence_path, evidence), supply=_write(supply_path, supply),
                reference_supply_present=supply is not None)
            counts["adapted_jobs"] += 1
        except Exception as error:
            failure = dict(schema="physical-reference-adaptation-error-1", job=job,
                execution=_receipt(execution_path) if execution_path.exists() else dict(path=str(execution_path), sha256=None),
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                source_receipts=sources, unknown_not_zero=True, skipped=False)
            failures.append(dict(job_id=job["id"], error_type=type(error).__name__, error=str(error)))
            # Failure evidence is distinct from successful evidence; never overwrite a partial output.
            error_file = _write(output / (job["id"] + "-error.json"), failure)
            row = dict(job_id=job["id"], split=job["split"], status="error", error=error_file)
            counts["error_jobs"] += 1
        rows.append(row)
        if number % 10 == 0 or number == len(jobs):
            print(json.dumps(dict(processed=number, planned=len(jobs), errors=len(failures))), flush=True)
    for receipt in sources:
        require(_receipt(receipt["path"]) == receipt, "adapter source changed during batch export")
    index = dict(schema="physical-reference-adaptation-batch-1", campaign_root=str(root),
        manifest=_receipt(root / "manifest.json"), effective_job_inventory_sha256=digest(jobs),
        posttarget_index=_receipt(root / "posttarget-job-index.json"), source_receipts=sources,
        planned_jobs=len(jobs), counts=dict(counts), unknown_reason_counts=dict(reason_counts),
        errors=failures, jobs=rows, status="completed" if not failures else "completed_with_errors",
        all_jobs_attempted=True, silent_skips=False, native_numeric_recomputed_here=False,
        numerical_claim="saved independent oracle/port receipts bound to unchanged native/program/oracle hashes",
        physical_CPU_interval_verified=False, universal_available_capacity_identified=False,
        reference_to_target_transfer_qualified=False, hardware_accuracy_qualified=False)
    index["artifact_sha256"] = digest(index)
    receipt = _write(output / "index.json", index)
    return dict(index=receipt, status=index["status"], planned_jobs=len(jobs), counts=dict(counts), errors=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_campaign(args.campaign, args.output)
    print(json.dumps(result), flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
