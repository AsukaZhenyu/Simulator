"""Opt-in complete-chain latency accounting for an exact measured IR binding.

The empirical residual has no identified position on a resource timeline.
Consequently this adapter preserves the independently replayed component
makespan and exposes a separate corrected bundle latency. It creates no fake
tensor work, controller command, event timing, or capacity profile.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as overhead

SCHEMA = "exact-ir-composition-latency-overlay-0.1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _sources(items, *, bound_components=False):
    if not isinstance(items, list) or not items:
        raise ValueError("explicit nonempty profile provenance is required")
    result = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("invalid profile source")
        if bound_components and item.get("split") != "profile":
            raise ValueError("a component cost contains a nonprofile source")
        row_id, sha = item.get("row_id"), item.get("raw_sha256")
        if (not isinstance(row_id, str) or not row_id or not isinstance(sha, str)
                or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("profile row ID and raw SHA256 are required")
        if (row_id, sha) in result:
            raise ValueError("duplicate profile source")
        result.add((row_id, sha))
    return result


def apply(binding, model, *, applicability_signature):
    """Account for a calibrated complete chain without changing its replay.

    Only cases and raw profile records actually used to fit this model are
    eligible. New hardware, shapes, batch protocols, prices and transformed
    graphs fail closed; predict() alone is a lower-level extrapolation API.
    applicability_signature must be independently reconstructed from verified
    original measurement records, not copied from an unverified binding. The
    CLI verifies the calibration campaign and reconstructs both this signature
    and the fixed IR replay before applying any correction.
    """
    if (not isinstance(binding, Mapping)
            or binding.get("schema") != "exact-measured-fixed-ir-binding-0.1"
            or binding.get("timing_only") is not True
            or binding.get("physical_capacity_validated") is not False
            or binding.get("capacity_fields_are_nonphysical") is not True
            or binding.get("unknown_cuda_workspace_preserved") is not True
            or binding.get("high_level_ModelPlan_transform_lowering_exercised") is not False
            or binding.get("full_Dense_MoE_inference_validated") is not False):
        raise ValueError("an explicit timing-only exact measured fixed-IR binding is required")
    if (not isinstance(model, Mapping) or model.get("schema") != overhead.SCHEMA
            or model.get("trained_on") != "profile_only"
            or model.get("heldout_used_for_calibration") is not False
            or model.get("profile_complete_strategy_totals_used_for_calibration") is not True
            or model.get("accepted_capacity_costs") is not False
            or model.get("gpu_capacity_proved") is not False
            or model.get("timing_unit") != "microseconds_per_completed_invocation_within_batch"
            or model.get("feature_units") != overhead.FEATURE_UNITS
            or model.get("coefficient_units") != {
                "composition_boundaries": "us/boundary",
                "hybrid_ffn_critical_branch_us": "dimensionless",
                "preload_event": "us/preload_event"}):
        raise ValueError("an explicit profile-only effective composition model is required")
    signature = binding.get("measured_physical_signature")
    if not isinstance(signature, Mapping) or canonical_hash(signature) != binding.get("physical_signature_sha256"):
        raise ValueError("physical signature hash mismatch")
    if not isinstance(applicability_signature, Mapping) or signature != applicability_signature:
        raise ValueError("binding differs from the independently verified applicability signature")
    cid, strategy = binding.get("case_id"), binding.get("strategy")
    case = model.get("profile_cases", {}).get(cid)
    if not case or case != signature.get("case"):
        raise ValueError("the exact geometry is absent from the fitted profile model")
    protocol = signature.get("protocol", {})
    measurement = protocol.get("measurement_contract", {})
    repetitions = model.get("batch_repetitions")
    if (type(repetitions) is not int or repetitions < 1
            or measurement.get("kind") != "completed_batch_mean"
            or measurement.get("batch_repetitions") != repetitions
            or measurement.get("isolated_latency_validated") is not False):
        raise ValueError("measurement batch protocol mismatch")
    training = [row for row in model.get("training", {}).get("rows", []) if row.get("case_id") == cid]
    expected_sources = _sources(training)
    if _sources(signature.get("profile_evidence")) != expected_sources:
        raise ValueError("hardware/protocol source records differ from model training")
    expected_metrics = overhead.component_metrics(case)
    prices = model.get("profile_component_prices_us", {}).get(cid, {})
    if set(prices) != set(expected_metrics):
        raise ValueError("frozen exact-case component inventory mismatch")
    # The fixed IR bridge rounds each independently measured component upward
    # to integer ns. Preserve that baseline exactly instead of replacing it
    # with a floating-point sum from a separate accounting layer.
    expression = overhead.recipe(case, strategy)
    def terms(node):
        return {node} if isinstance(node, str) else set().union(*(terms(v) for v in next(iter(node.values()))))
    required, runtime_prices = terms(expression), {}
    entries = binding.get("component_bindings")
    if not isinstance(entries, list) or len(entries) != len(required):
        raise ValueError("one exact component binding per used metric is required")
    for entry in entries:
        metric = entry.get("metric")
        if metric not in required or metric in runtime_prices:
            raise ValueError("duplicate or undeclared bound metric")
        if (_sources(entry.get("sources"), bound_components=True) != expected_sources
                or any(s.get("metric") != metric for s in entry["sources"])):
            raise ValueError("component cost provenance mismatch")
        measured, duration = entry.get("measured_median_ns"), entry.get("runtime_duration_ns")
        if (isinstance(measured, bool) or not isinstance(measured, (int, float))
                or not math.isfinite(measured) or measured <= 0
                or not math.isclose(measured, prices[metric] * 1000, rel_tol=0, abs_tol=1e-6)
                or type(duration) is not int or duration != math.ceil(measured)):
            raise ValueError("component price or integer runtime rounding mismatch")
        runtime_prices[metric] = duration / 1000.0
    prediction = overhead.predict(case, strategy, runtime_prices, model)
    if any(not math.isfinite(prediction[k]) for k in ("baseline_ns", "correction_ns", "predicted_ns")):
        raise ValueError("complete-chain prediction is not finite in nanoseconds")
    replayed = binding.get("independently_replayed_makespan_ns")
    if (type(replayed) is not int or replayed <= 0
            or replayed != binding.get("formula_with_integer_prices_ns")
            or not math.isclose(prediction["baseline_ns"], replayed, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("replayed makespan differs from exact component formula")
    return dict(schema=SCHEMA, case_id=cid, strategy=strategy,
        model_sha256=canonical_hash(model), physical_signature_sha256=binding["physical_signature_sha256"],
        binding_sha256=canonical_hash(binding),
        independently_replayed_makespan_ns=replayed,
        effective_composition_correction_ns=prediction["correction_ns"],
        corrected_complete_chain_latency_ns=replayed + prediction["correction_ns"],
        correction_contributions_us=prediction["correction_contributions_us"],
        features=prediction["features"], feature_units=prediction["feature_units"],
        timeline_modified=False, per_event_correction_identified=False,
        timing_only=True, physical_capacity_validated=False, unknown_cuda_workspace_preserved=True,
        latency_quality_acceptance="not_evaluated_by_overlay; consult calibration report and source-stability gates",
        evaluation_design="retrospective; no fresh blind validation",
        profile_complete_strategy_totals_used_for_calibration=True, heldout_used_for_calibration=False,
        application_scope="exact retained geometry, raw profile provenance, batch protocol and fixed tensor component recipe",
        limitation="Effective complete-chain correction has no identified resource or position in the timeline; cannot be composed arbitrarily with overlapping bundles.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="frozen bundle saved by composition_calibration/run.py")
    parser.add_argument("--experiment", type=Path, required=True, help="original verified batch campaign used by this frozen bundle")
    parser.add_argument("--bindings", type=Path, required=True, help="verified exact IR replay directory containing physical-binding.json files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be a new directory")
    application_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    calibration_bytes = args.calibration.read_bytes()
    parsed_calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
    bundle = json.loads(calibration_bytes.decode("utf-8"))
    if bundle.get("schema") != "retrospective-composition-calibration-0.1":
        raise ValueError("unsupported frozen calibration bundle")
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    verifier = load("composition_application_verifier", HERE / "verify.py")
    audit = verifier.verify(args.calibration.parent)
    runner = verifier.runner
    manifest = json.loads((args.calibration.parent / "manifest.json").read_text(encoding="utf-8"))
    label = bundle.get("label")
    if (args.calibration.resolve() != (args.calibration.parent / (str(label) + "-frozen-model.json")).resolve()
            or args.experiment.resolve() != Path(manifest["input_paths"].get(label, "")).resolve()
            or parsed_calibration_sha256 != manifest["model_sha256"].get(label)):
        raise ValueError("calibration filename/campaign differs from the verified manifest")
    replay = load("composition_application_replay", HERE.parent / "execution_calibration/replay_ir.py")
    context = runner.preflight(args.experiment)
    rows = runner.base.document(args.experiment / "measurements.json")
    original_model = runner.base.document(args.experiment / "prediction-model.json")
    original_manifest, suite = context["manifest"], context["suite"]
    measurement_contract = dict(kind="completed_batch_mean", batch_repetitions=suite["batch_repetitions"],
        raw_timing_unit=original_manifest["timing_unit"],
        timing_unit=original_manifest["derived_metric_unit"], normalization=original_manifest["normalization"],
        isolated_latency_validated=False)
    signatures = {}
    for case in suite["cases"]:
        selected = [row for row in rows if row["case"]["id"] == case["id"]]
        signatures[case["id"]] = replay.physical_signature(case, selected, dict(**suite["protocol"],
            measurement_contract=measurement_contract,
            observed_child_flags=original_manifest["observed_child_flags"],
            environment_before=original_manifest["environment_before"], environment_after=original_manifest.get("environment_after"),
            executed_assets_sha256=original_manifest["inventory_before"], executed_context_source_sha256=None))
    sources = [args.calibration, *sorted(args.bindings.rglob("physical-binding.json"))]
    if len(sources) < 2:
        raise ValueError("no exact measured IR bindings found")
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results = []
    for path in sources[1:]:
        binding = json.loads(path.read_text(encoding="utf-8"))
        cid = binding.get("case_id")
        if cid not in signatures:
            raise ValueError("binding case absent from original verified campaign")
        frozen = next(item for item in original_model["cases"] if item["case_id"] == cid)
        _, _, reconstructed, _, _ = replay.bind_resident(signatures[cid]["case"], binding["strategy"], frozen, signatures[cid])
        if binding != reconstructed:
            raise ValueError("saved binding differs from freshly reconstructed fixed IR replay")
        results.append(apply(binding, bundle["model"], applicability_signature=signatures[cid]))
    keys = [(r["case_id"], r["strategy"]) for r in results]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case/strategy bindings")
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != before[str(p.resolve())] for p in sources):
        raise ValueError("an input artifact changed during accounting")
    for assets in manifest["original_assets_before"].values():
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in assets.items()):
            raise ValueError("an original campaign artifact changed after verification")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != application_source_sha256:
        raise ValueError("application source changed during accounting")
    args.output.mkdir(parents=True)
    report = dict(schema=SCHEMA, source_sha256=before, source_artifacts_unchanged=True,
                  exact_bindings_accounted=len(results), results=results,
                  calibration_campaign_verification=audit, fixed_ir_replayed_again=True,
                  application_source_sha256=application_source_sha256,
                  timeline_modified=False, physical_capacity_validated=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Exact IR complete-chain calibration accounting", "",
        "The independently replayed component makespan and timeline remain unchanged. Effective correction is a separate opt-in complete-chain estimate, with no identified event placement or capacity claim.", "",
        "| Case | Strategy | Replayed component us | Correction us | Complete-chain us |",
        "|---|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['case_id']} | {r['strategy']} | {r['independently_replayed_makespan_ns']/1000:.3f} | {r['effective_composition_correction_ns']/1000:.3f} | {r['corrected_complete_chain_latency_ns']/1000:.3f} |")
    lines += ["", "These are retrospective timing estimates at retained exact shapes. Acceptance remains subject to the calibration report, including all original variability gates and new profile-target stability gates. Arbitrary overlapping bundles cannot simply add these residuals.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(dict(output=str(args.output.resolve()), exact_bindings_accounted=len(results))))


if __name__ == "__main__":
    main()
