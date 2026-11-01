"""Declared shared-pool acceptance matrix, not a hardware accuracy experiment.

Runs the same execution/storage choices as the separate-memory campaign on
five small Dense/MoE workloads. Every case retains states and allocation refs.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hardware_sensitivity import stable_hash, write_json
from experiments.strategy_comparison import frozen_evidence
from experiments.strategy_comparison_campaign import run_case, execution_source_manifest
from experiments.strategy_comparison_scenarios import ScenarioSpec, candidate_ids


def acceptance_scenarios():
    return (
        ScenarioSpec("shared-dense-prefill", "dense", 16, 0, None),
        ScenarioSpec("shared-dense-decode", "dense", 4, 1, None),
        *(ScenarioSpec(f"shared-moe-{route}", "moe", 4, 1, route)
          for route in ("hot", "rotating", "phase_shift")),
    )


def source_manifest():
    manifest = execution_source_manifest()
    manifest["experiments/shared_storage_campaign.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    specifications = acceptance_scenarios()
    jobs = [(spec, candidate) for spec in specifications for candidate in candidate_ids(spec.model_kind)]
    override = dict(memory_mode="shared", shared_capacity_bytes=16 * 1024**2,
                    transfer_bandwidth_bytes_per_second=None)
    before = source_manifest()
    evidence = frozen_evidence()
    protocol = dict(format_version="shared-storage-acceptance-0.1", cases=len(jobs),
        scenarios=[asdict(spec) for spec in specifications], hardware_override=override,
        warning="Uncalibrated analytical costs. No measured UMA hardware speedup is claimed.",
        memory_semantics="one physical pool; explicit zero-payload SHARE; real COPY retains positive pricing",
        cache_budget_semantics="192 KiB logical GPU weight-view quota, not extra physical storage",
        execution_sources_sha256=stable_hash(before), frozen_accuracy_evidence=evidence,
        workers=args.workers)
    write_json(output / "protocol.json", protocol)
    write_json(output / "source-manifest-before.json", before)
    records = []
    started = time.perf_counter()
    with (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_case, asdict(spec), candidate, str(output), True,
                                   hardware_override=override): (spec, candidate)
                       for spec, candidate in jobs}
            for future in as_completed(pending):
                row = future.result()
                records.append(row)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(f"{len(records)}/{len(jobs)} {row['scenario_id']} {row['candidate_id']} "
                      f"success={row['success']} T={row.get('completed_makespan_ns')}ns "
                      f"wall={time.perf_counter()-started:.1f}s", flush=True)
    order = {(spec.id, candidate): index for index, (spec, candidate) in enumerate(jobs)}
    records.sort(key=lambda row: order[(row["scenario_id"], row["candidate_id"])] )
    write_json(output / "records.json", records)
    after = source_manifest()
    write_json(output / "source-manifest-after.json", after)
    errors = []
    for row in records:
        if not row["success"]:
            errors.append(f"{row['scenario_id']}/{row['candidate_id']}: execution did not finish")
            continue
        if (row.get("memory_mode") != "shared" or row["h2d_bytes"] != 0 or row["d2h_bytes"] != 0 or
                row.get("copy_event_span_ns") != 0 or row["event_counts"].get("COPY", 0) != 0):
            errors.append(f"{row['scenario_id']}/{row['candidate_id']}: phantom transfer or mode mismatch")
        if row.get("physical_memory_peaks_bytes") != {"shared": row.get("peak_shared_bytes")}:
            errors.append(f"{row['scenario_id']}/{row['candidate_id']}: duplicated physical pool")
        if row["peak_shared_bytes"] > override["shared_capacity_bytes"]:
            errors.append(f"{row['scenario_id']}/{row['candidate_id']}: capacity exceeded")
    if before != after:
        errors.append("execution sources changed during campaign")
    if evidence != frozen_evidence():
        errors.append("frozen accuracy evidence changed")
    completion = dict(expected_cases=len(jobs), completed_cases=len(records),
        successful_cases=sum(row["success"] for row in records), errors=errors,
        passed=not errors and len(records) == len(jobs),
        unchanged_execution_sources=before == after, frozen_accuracy_evidence_unchanged=evidence == frozen_evidence(),
        total_share_events=sum(row["event_counts"].get("SHARE", 0) for row in records),
        wall_seconds=time.perf_counter()-started, records_sha256=stable_hash(records))
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    return 0 if completion["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
