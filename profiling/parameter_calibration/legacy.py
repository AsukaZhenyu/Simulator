"""Read-only recovery of historical total timings, never hardware parameters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .observations import ParameterObservation, canonical, digest, utc
from .parameters import cv, median


@dataclass(frozen=True)
class ImportedReferences:
    observations: tuple[ParameterObservation, ...]
    receipt: dict


def import_legacy_resource_campaign(campaign: str | Path, *, split: str = "profile") -> ImportedReferences:
    """Import existing offered-load PV samples without executing the old fit.

    All bound raw job files are rehashed. Historical source inventories are
    preserved as version identities; they are not asserted to match today's
    working tree. Only fully contained steady-phase intervals receive a state;
    transition samples remain counted in the receipt, never fitted or erased.
    """
    if split not in {"profile", "heldout"}:
        raise ValueError("legacy import supports profile or heldout only")
    root = Path(campaign).resolve()
    old_split = "train" if split == "profile" else "heldout"
    manifest_path, records_path = root / "manifest.json", root / f"{old_split}-rows.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "resource-trajectory-campaign-0.1":
        raise ValueError("unsupported historical resource campaign")
    if digest(manifest["sources"]) != manifest["source_inventory_sha256"]:
        raise ValueError("historical source inventory identity is inconsistent")
    jobs = [job for job in manifest["jobs"] if job["split"] == old_split]
    if not jobs or [record["job"] for record in records] != jobs:
        raise ValueError("historical split is missing, duplicated, or reordered")
    observations, excluded, counts, verified_files, states = [], 0, 0, 0, set()
    for record in records:
        job = record["job"]
        directory = (root / "jobs" / job["id"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("historical job escapes campaign directory")
        if record != json.loads((directory / "execution.json").read_text(encoding="utf-8")):
            raise ValueError("historical execution record changed")
        if record.get("status") != "measured" or record.get("cleanup_verified") is not True:
            raise ValueError("historical job failed or lacks cleanup evidence")
        bindings = record.get("artifacts_sha256")
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("historical raw artifact bindings are missing")
        required_raw = {"observations.json", "numerical.json", "timeline.json", "main/result.json"}
        if not required_raw <= bindings.keys():
            raise ValueError("a consumed historical raw file lacks a hash binding")
        for name, expected in bindings.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"historical bound artifact changed: {name}")
            verified_files += 1
        numerical = json.loads((directory / "numerical.json").read_text(encoding="utf-8"))
        if numerical.get("passed") is not True or record["summary"].get("native_numeric_passed") is not True:
            raise ValueError("historical numerical validation failed")
        raw_path = directory / "observations.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        timeline = json.loads((directory / "timeline.json").read_text(encoding="utf-8"))
        native = json.loads((directory / "main" / "result.json").read_text(encoding="utf-8"))
        frequency = timeline["clock"]["frequency_hz"]
        if (type(frequency) is not int or frequency <= 0 or native["clock"]["frequency_hz"] != frequency
                or native["clock"]["clock_id"] != timeline["clock"]["clock_id"]):
            raise ValueError("historical clocks disagree")
        if record["process_id"] != native["process_id"]:
            raise ValueError("historical foreground process identity disagrees")
        intervals = timeline["events"]
        for sample in raw:
            if sample["metric"] != "target":
                continue
            counts += 1
            if sample["run_id"] != job["id"] or sample["split"] != old_split or sample["seed"] != job["seed"]:
                raise ValueError("historical sample process/split identity disagrees")
            begin, end = sample["begin_tick"], sample["end_tick"]
            if type(begin) is not int or type(end) is not int or end <= begin:
                raise ValueError("historical target interval is invalid")
            elapsed = Fraction((end - begin) * 1_000_000_000, frequency)
            if elapsed.denominator != 1 or sample["elapsed_ns"] != elapsed:
                raise ValueError("historical elapsed time does not reconstruct")
            matches = [event for event in intervals
                       if event["steady_begin_tick"] <= begin and end <= event["steady_end_tick"]]
            if not matches:
                if sample["phase"]["kind"] != "transition_or_outside":
                    raise ValueError("historical state assignment does not reconstruct")
                excluded += 1
                continue
            if len(matches) != 1:
                raise ValueError("historical steady intervals overlap")
            event = matches[0]
            phase = sample["phase"]
            if phase["kind"] != "steady" or phase["phase_index"] != event["phase_index"] or phase["offered_resources"] != event["requested_resources"]:
                raise ValueError("historical steady state label disagrees with interval evidence")
            resources = event["requested_resources"]
            context = dict(protocol_id=manifest["protocol_id"], mode=job["mode"],
                action="gpu_attention_pv", execution_boundary=native["execution_boundary"],
                state_semantics="offered_load_controlled", offered_resources=resources)
            states.add(canonical(resources).decode())
            # Wall-time fields cover the source process because QPC/UTC offset
            # was not calibrated closely enough for sub-action UTC timestamps.
            observations.append(ParameterObservation(
                observation_id=sample["id"], parameter_name="gpu_attention_pv_reference_ns",
                kind="reference_duration", value=elapsed, unit="ns",
                timing_scope="whole_action_completed", context_json=canonical(context).decode(),
                run_id=job["id"], process_id=record["process_id"], seed=job["seed"], split=split,
                started_at_utc=record["started_utc"], ended_at_utc=record["completed_utc"],
                source_path=str(raw_path), source_sha256=bindings["observations.json"],
                evidence=(f"sample:{sample['id']}", "timing includes host submission, GPU work, and completed synchronization",
                          f"historical_source_inventory_sha256:{manifest['source_inventory_sha256']}"),
                source_verified=True, numerical_passed=True))
    by_state = {}
    for context_json in sorted({row.context_json for row in observations}):
        rows = [row for row in observations if row.context_json == context_json]
        values = [row.value for row in rows]
        by_state[context_json] = dict(sample_count=len(values),
            median_ns_exact={"numerator": median(values).numerator, "denominator": median(values).denominator},
            cv=cv(values), independent_runs=len({row.run_id for row in rows}),
            evidence_issues=["fewer_than_three_independent_runs"] if len({row.run_id for row in rows}) < 3 else [])
    receipt = dict(schema="historical-resource-reference-import-1", split=split,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        split_records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
        historical_source_inventory_sha256=manifest["source_inventory_sha256"],
        current_working_tree_source_equivalence_claimed=False,
        bound_raw_files_verified=verified_files, target_sample_count=counts,
        steady_reference_samples=len(observations), transition_or_outside_samples=excluded,
        transition_samples_discarded_as_outliers=False, resource_state_count=len(states),
        state_summaries=by_state, physical_parameters_estimated=False,
        modeling_prediction_executed=False, legacy_ridge_fit_imported=False,
        missing_physical_parameters=["gpu_effective_flops_per_second",
            "gpu_memory_bandwidth_bytes_per_second", "host_fixed_overhead_ns"],
        missing_reason="Whole completed PV intervals cannot separate device-active work, payload traffic, and exclusive host cost.",
        accuracy_accepted=False)
    return ImportedReferences(tuple(observations), receipt)
