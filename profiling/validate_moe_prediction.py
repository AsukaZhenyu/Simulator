"""Compare stage-priced tensor replay with independently sampled joint MoE FFN.

No joint timing is used as a simulator price. This validates only the declared
resident CPU F32 subgraph with prescribed routing and reproducible random weights.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from llm_infer_model.m2.moe_ffn_contract import contract_hash, statistics_ns, STABILITY_CV_LIMIT
from llm_infer_model.m2.moe_measured import compile_measured_ffn, plan_measured_ffn, replay_measured_ffn
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import workload_fingerprint, scenario_fingerprint, InvalidInput, Unsupported
from run_moe_measurements import dump, load_measurements, sha256


def evidence_snapshot(directory):
    directory = Path(directory).resolve()
    report = load_measurements(directory)
    paths = [directory / "manifest.json", directory / "measurements.json"]
    paths += [directory / record["raw_output"] for record in report["records"]]
    return {str(path): sha256(path) for path in paths}


def check_pair(profile, holdout):
    """Demand identical physical work, weights, materialized ports and runtime."""
    if profile["case_id"] != holdout["case_id"]:
        raise ValueError("holdout case id differs")
    if contract_hash(profile["physical_signature"]) != contract_hash(holdout["physical_signature"]):
        raise ValueError("holdout physical case/signature differs")
    if profile["raw_sha256"] == holdout["raw_sha256"]:
        raise ValueError("profile raw evidence reused as a holdout")
    a, b = profile["native"], holdout["native"]
    for key in ("seed", "threads_requested", "threads_applied", "timing_scope"):
        if a[key] != b[key]:
            raise ValueError(f"holdout {key} differs")
    for stage in a["stages"]:
        if a["stages"][stage]["ports"] != b["stages"][stage]["ports"]:
            raise ValueError(f"holdout materialized ports differ at {stage}")
        for key in ("workspace_bytes", "workspace_source", "memory_accounting"):
            if a["stages"][stage][key] != b["stages"][stage][key]:
                raise ValueError(f"holdout {stage}.{key} differs")


def compare_case(profile, holdouts, hardware, *, max_relative_error, allow_unstable=False):
    if type(max_relative_error) not in (int, float) or not math.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError("max_relative_error must be an explicit finite nonnegative number")
    if not holdouts:
        raise ValueError("at least one independent holdout is required")
    compiled = compile_measured_ffn(profile, hardware, allow_unstable=allow_unstable)
    replay = replay_measured_ffn(compiled)
    if replay.status != "valid":
        raise ValueError(f"tensor replay failed: {replay.reason}")
    seen = {profile["raw_sha256"]}
    rows = []
    for holdout in holdouts:
        check_pair(profile, holdout)
        if holdout["raw_sha256"] in seen:
            raise ValueError("duplicate holdout raw evidence")
        seen.add(holdout["raw_sha256"])
        joint = statistics_ns(holdout["native"]["stages"]["joint"]["samples_ns"])
        error = (replay.makespan_ns - joint["median_ns"]) / joint["median_ns"]
        quality = joint["cv"] <= STABILITY_CV_LIMIT
        within = abs(error) <= max_relative_error
        rows.append(dict(observed_joint_statistics=joint, predicted_ns=replay.makespan_ns,
            signed_relative_error=error, absolute_relative_error=abs(error), within_requested_tolerance=within,
            joint_stability_passed=quality, local_prediction_accepted=within and quality and not compiled.validation_only,
            holdout_raw_sha256=holdout["raw_sha256"], holdout_quality_flags=holdout["quality_flags"]))
    result = dict(case_id=profile["case_id"], physical_contract_sha256=profile["physical_contract_sha256"],
        profile_stage_statistics=compiled.bindings["stage_statistics"], profile_quality_flags=profile["quality_flags"],
        predicted_ns=replay.makespan_ns, max_relative_error=max_relative_error, holdouts=rows,
        validation_only=compiled.validation_only or any(not row["joint_stability_passed"] for row in rows),
        sampling_mode=compiled.bindings["sampling_mode"],
        context_sensitive_prices=compiled.bindings["context_sensitive_prices"],
        local_prediction_accepted=all(row["local_prediction_accepted"] for row in rows),
        performance_calibrated=False, whole_decoder_layer_measured=False, original_model_weights=False,
        original_model_kernel_reproduced=False, GPU_compute_measured=False, offloading_joint_measured=False,
        physical_capacity_verified=False, scope="resident_cpu_f32_moe_ffn_prescribed_trace_replay")
    return compiled, replay, result


def validate_prediction(args):
    profile_dir = args.profile.resolve()
    holdout_dirs = [path.resolve() for path in args.holdout]
    if profile_dir in holdout_dirs or len(set(holdout_dirs)) != len(holdout_dirs):
        raise ValueError("profile and holdout directories must be independent and distinct")
    directories = [profile_dir, *holdout_dirs]
    snapshots = {str(path): evidence_snapshot(path) for path in directories}
    reports = [load_measurements(path, verify_live_sources=args.verify_live_sources) for path in directories]
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in directories]
    for manifest in manifests[1:]:
        for key in ("native_binary_sha256", "runtime_libraries_sha256", "seed", "machine", "platform"):
            if manifest[key] != manifests[0][key]:
                raise ValueError(f"holdout runtime manifest {key} differs")
    indices = [{record["case_id"]: record for record in report["records"]} for report in reports]
    if any(set(index) != set(indices[0]) for index in indices[1:]):
        raise ValueError("holdout case coverage differs")
    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    hardware_hash = sha256(args.hardware)
    compared = []
    for case_id, profile in indices[0].items():
        compiled, replay, row = compare_case(profile, [index[case_id] for index in indices[1:]], hardware,
            max_relative_error=args.max_relative_error, allow_unstable=args.allow_unstable)
        compared.append((compiled, replay, row))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (compiled, replay, row) in enumerate(compared):
        folder = args.output / f"{index:03d}"
        folder.mkdir()
        dump(folder / "workload.json", compiled.workload_document)
        dump(folder / "scenario.json", compiled.scenario_document)
        actions = [dict(kind=action.kind, **({"operation_id": action.target_id, "device": action.device}
                   if action.kind == "COMPUTE" else {})) for action in plan_measured_ffn(compiled)]
        dump(folder / "mapping.json", dict(schema_version="0.2", scenario_id=compiled.scenario.id,
            workload_fingerprint=workload_fingerprint(compiled.scenario.workload),
            scenario_fingerprint=scenario_fingerprint(compiled.scenario), actions=actions))
        loaded = load_mapping(folder / "mapping.json", compiled.scenario)
        second = evaluate_mapping(compiled.scenario, loaded.actions)
        if not loaded.fingerprint_verified or second != replay:
            raise ValueError("exported mapping replay changed or lacks verified fingerprints")
        dump(folder / "bindings.json", compiled.bindings)
        dump(folder / "replay.json", asdict(replay))
        dump(folder / "comparison.json", row)
    if sha256(args.hardware) != hardware_hash or any(evidence_snapshot(path) != snapshots[str(path)] for path in directories):
        raise ValueError("evidence or declared hardware changed during comparison")
    report = dict(schema_version="moe_prediction_validation_v1", cases=[row for _, _, row in compared],
        evidence_sources=snapshots, hardware_sha256=hardware_hash, hardware=hardware,
        verify_live_sources=args.verify_live_sources, allow_unstable=args.allow_unstable,
        price_source="profile_materialized_stage_medians_only", truth_source="independent_holdout_joint_medians",
        sampling_modes=sorted({row["sampling_mode"] for _, _, row in compared}),
        tolerance_source="explicit_command_argument", max_relative_error=args.max_relative_error,
        all_local_predictions_accepted=all(row["local_prediction_accepted"] for _, _, row in compared),
        performance_calibrated=False, declared_scope="resident_cpu_f32_moe_ffn_subgraph")
    dump(args.output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-error", type=float, required=True)
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--verify-live-sources", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_prediction(args)
        print(json.dumps({"case_count": len(report["cases"]),
            "all_local_predictions_accepted": report["all_local_predictions_accepted"], "output": str(args.output)}))
        return 0
    except (ValueError, OSError, InvalidInput, Unsupported) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
