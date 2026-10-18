"""Rebuild offline composition models and retrospective reports; never run hardware."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as runner


def verify(output):
    output = Path(output).resolve()
    manifest = runner.base.document(output / "manifest.json")
    ledger = runner.base.document(output / "freeze-ledger.json")
    if (manifest.get("schema") != runner.SCHEMA or manifest.get("fresh_blind_validation") is not False
            or manifest.get("joint_chain_totals_used_for_calibration") is not True
            or manifest.get("gpu_private_workspace_bytes", "missing") is not None
            or manifest.get("physical_capacity_validated") is not False
            or manifest.get("sources_unchanged") is not True or manifest.get("original_assets_unchanged") is not True):
        raise ValueError("retrospective/target-use/unknown-workspace contract changed")
    if (manifest["source_modules_before"] != manifest["source_modules_after"]
            or manifest["source_modules_after"] != runner.source_inventory()):
        raise ValueError("bound analysis/model source changed")
    if manifest["original_assets_before"] != manifest["original_assets_after"]:
        raise ValueError("original assets changed during evaluation")
    for assets in manifest["original_assets_before"].values():
        for path, digest in assets.items():
            if runner.a.sha(path) != digest:
                raise ValueError("original asset changed: " + path)
    if runner.a.sha(output / "report.json") != manifest["report_sha256"]:
        raise ValueError("report hash mismatch")
    stages = manifest["stages"]
    freezes = [i for i, item in enumerate(stages) if item["stage"] == "full_and_all_grouped_models_saved_and_hashed"]
    reads = [i for i, item in enumerate(stages) if item["stage"] == "heldout_totals_deserialization_started"]
    if len(freezes) != 2 or len(reads) != 2 or max(freezes) >= min(reads):
        raise ValueError("all models must be frozen before any heldout deserialization")
    if ledger.get("model_sha256") != manifest["model_sha256"] or ledger.get("stages") != stages[:min(reads)]:
        raise ValueError("pre-evaluation freeze ledger mismatch")
    contexts, bundles, profiles, heldout, audits, campaigns = {}, {}, {}, {}, {}, {}
    for label, path in manifest["input_paths"].items():
        context = runner.preflight(path)
        contexts[label] = context
        profile = list(runner.profile_objects(Path(path) / "measurements.json"))
        profiles[label] = profile
        bundle_path = output / (label + "-frozen-model.json")
        if runner.a.sha(bundle_path) != manifest["model_sha256"][label]:
            raise ValueError("frozen model hash mismatch: " + label)
        saved = runner.base.document(bundle_path)
        if saved != runner.build_bundle(profile, context["suite"], label):
            raise ValueError("profile-only full/grouped model does not reconstruct: " + label)
        bundles[label] = saved
        for fold in saved["folds"]:
            excluded_ids = {r["id"] for r in profile if r["case"]["id"] == fold["excluded_case_id"]}
            if excluded_ids & set(fold["training_row_ids"]):
                raise ValueError("case group leaked into coefficient fitting")
    for label, context in contexts.items():
        rows = runner.base.document(context["experiment"] / "measurements.json")
        audits[label] = runner.verify_campaign(context, rows)
        heldout[label] = [r for r in rows if r["split"] == "heldout"]
        campaigns[label] = {"full": runner.evaluate(profiles[label], heldout[label], context["suite"], bundles[label]),
            "grouped_leave_one_case_out": runner.evaluate(profiles[label], heldout[label], context["suite"], bundles[label], fold_models=True)}
    transfer = dict(compatibility=runner.compatibility(contexts["initial"], contexts["confirmation"]))
    if transfer["compatibility"]["compatible"]:
        transfer["evaluation"] = runner.evaluate(profiles["initial"], heldout["confirmation"], contexts["confirmation"]["suite"], bundles["initial"])
        transfer["evaluation"]["transfer_mode"] = "zero refit: both initial profile component prices and initial composition coefficients frozen"
    report = runner.base.document(output / "report.json")
    if (report["campaigns"] != campaigns or report["cross_campaign_transfer"] != transfer
            or report["input_verification"] != audits):
        raise ValueError("retrospective report does not reconstruct")
    if (output / "REPORT.md").read_text(encoding="utf-8") != runner.report_markdown(report):
        raise ValueError("human-readable report does not reconstruct")
    return dict(schema=runner.SCHEMA, verified=True, full_and_grouped_models_reconstructed=True,
        profile_only_training_reconstructed=True, joint_chain_totals_used_for_calibration=True,
        models_saved_and_hashed_before_offline_heldout_deserialization=True,
        fresh_blind_validation=False, original_assets_unchanged=True,
        gpu_private_workspace_bytes=None, physical_capacity_validated=False,
        source_modules=runner.source_inventory(), report_sha256=runner.a.sha(output / "report.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError("verification output must be a new file")
        result = verify(args.experiment)
        if args.output:
            runner.a.dump(args.output, result)
        print("verified retrospective composition evaluation; physical capacity remains false")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("composition verification: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
