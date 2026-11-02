"""Offline, retrospective composition calibration of retained batch campaigns.

Only profile process objects are deserialized before every model is written and
hashed. Heldout data have been inspected historically: the ordering here makes
the reusable pipeline auditable, and does not create a fresh blind experiment.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent
sys.path[:0] = [str(HERE), str(PROFILING / "execution_calibration"), str(PROFILING)]
import analysis as a
import run_dual as base
import model as overhead

_spec = importlib.util.spec_from_file_location("composition_batch", PROFILING / "batched_calibration/run.py")
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)
_verify_spec = importlib.util.spec_from_file_location("composition_strict_batch_verifier", PROFILING / "batched_calibration/verify_batch.py")
strict_batch = importlib.util.module_from_spec(_verify_spec)
_verify_spec.loader.exec_module(strict_batch)
SCHEMA = "retrospective-composition-calibration-0.1"
SCOPE = "F32 compact FFN and post-RoPE Attention; repeated completed-work batch means, explicit CPU/GPU worker contexts."


def now():
    return datetime.now(timezone.utc).isoformat()


def object_document(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def profile_objects(path):
    """Stream array objects, decoding only a profile-labelled object prefix.

    JSON values belonging to heldout rows are never deserialized in this stage.
    Byte hashing and lexical scanning are provenance operations, not evaluation.
    The complete JSON and exact coverage are verified after model freezing.
    """
    import re
    with Path(path).open(encoding="utf-8") as stream:
        depth, quoted, escaped, current = 0, False, False, []
        started = False
        for chunk in iter(lambda: stream.read(65536), ""):
            for char in chunk:
                if not started:
                    if char.isspace():
                        continue
                    if char != "[":
                        raise ValueError("measurements must be an array")
                    started = True
                    continue
                if depth == 0:
                    if char.isspace() or char in ",]":
                        continue
                    if char != "{":
                        raise ValueError("measurement array must contain objects")
                current.append(char)
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        raw = "".join(current)
                        # Declared collection schema places split before metrics.
                        prefix = raw.split('"metrics"', 1)[0]
                        matches = re.findall(r'"split"\s*:\s*"([^"\\]*)"', prefix)
                        if len(matches) != 1 or matches[0] not in ("profile", "heldout"):
                            raise ValueError("unambiguous split before metrics required")
                        if matches[0] == "profile":
                            value = object_document(raw)
                            if value["split"] != "profile":
                                raise ValueError("profile selection mismatch")
                            yield value
                        current = []
        if depth or quoted or not started:
            raise ValueError("truncated measurement array")


def function_signature(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
    if set(selected) != set(names):
        raise ValueError("missing physical protocol function")
    return selected


def preflight(experiment):
    """Bind archived evidence without deserializing any heldout timing row."""
    experiment = Path(experiment).resolve()
    suite = batch.validate_suite(base.document(experiment / "suite.json"))
    manifest = base.document(experiment / "manifest.json")
    if (manifest.get("schema") != "execution-transform-batch-manifest-0.1"
            or manifest.get("timing_unit") != "completed_work_batch_total"
            or manifest.get("derived_metric_unit") != "rounded_mean_completed_invocation_within_batch_ns"
            or manifest.get("normalization") != "round(batch_total_ns / batch_repetitions); raw totals retained"
            or manifest.get("sample_filtering") != "none"
            or manifest.get("gpu_capacity_proved") is not False
            or manifest.get("assets_unchanged") is not True
            or manifest["inventory_before"] != manifest["inventory_after"]):
        raise ValueError("original timing, immutability or capacity contract changed")
    inventory, snapshots = manifest["inventory_before"], manifest["source_snapshots"]
    archived = {}
    for original, record in snapshots.items():
        path = (experiment / record["path"]).resolve()
        if (not path.is_relative_to(experiment / "source") or a.sha(path) != record["sha256"]
                or inventory.get(original) != record["sha256"]):
            raise ValueError("archived source path/hash mismatch")
        archived[original] = path
    expected_modules = ((a, PROFILING / "execution_calibration/analysis.py"),
                        (base, PROFILING / "execution_calibration/run_dual.py"),
                        (batch, PROFILING / "batched_calibration/run.py"),
                        (overhead, HERE / "model.py"))
    for module, expected in expected_modules:
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError("module imported outside bound checkout: " + module.__name__)
    changes = []
    for original, digest in inventory.items():
        path = Path(original)
        current = a.sha(path) if path.is_file() else None
        if current == digest:
            continue
        if original not in archived:
            raise ValueError("unarchived bound asset changed: " + original)
        if path.resolve() == (PROFILING / "batched_calibration/run.py").resolve():
            names = ("validate_suite", "normalized_payload", "recipes")
            if function_signature(path, names) != function_signature(archived[original], names):
                raise ValueError("archived/current physical protocol functions differ")
            reason = "collection/build metadata revision; suite, normalization and recipe ASTs identical"
        elif path.resolve() == (PROFILING / "native/execution_transforms_batch/CMakeLists.txt").resolve():
            reason = "CMake metadata revision; archived source retained, compiled binary/build/generated-source identities remain required"
        else:
            raise ValueError("bound physical/analysis asset changed: " + original)
        changes.append(dict(path=original, collected_sha256=digest, current_sha256=current, reason=reason))
    local_files = [path for path in experiment.rglob("*") if path.is_file()]
    return dict(experiment=experiment, suite=suite, manifest=manifest,
                original_files={str(path): a.sha(path) for path in sorted(local_files)},
                changed_current_metadata=changes)


def verify_campaign(context, rows):
    """Archived evidence reconstruction; deliberately distinct from strict current-source verification."""
    experiment, suite, manifest = context["experiment"], context["suite"], context["manifest"]
    expected = {r["id"]: r for split in ("profile", "heldout") for r in base.jobs(suite, split)}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(expected)):
        raise ValueError("independent-process row coverage mismatch")
    inventory, snapshots = manifest["inventory_before"], manifest["source_snapshots"]
    binaries = {r["command"][0] for r in rows}
    if len(binaries) != 1 or next(iter(binaries)) not in inventory:
        raise ValueError("native binary not uniquely bound")
    binary = next(iter(binaries))
    def snapshot(name):
        matches = [record for original, record in snapshots.items() if Path(original).name == name]
        if len(matches) != 1:
            raise ValueError("missing/ambiguous archived build source: " + name)
        return base.document(experiment / matches[0]["path"])
    build, generated = snapshot("compute-batch-build.json"), snapshot("source-manifest.json")
    if build.get("binary_sha256") != inventory[binary] or build.get("source_manifest") != generated:
        raise ValueError("archived binary/build/source manifest mismatch")
    for field, filename in (("generator_sha256", "generate_probe.py"), ("context_sha256", "compute_context.cpp"),
                            ("helper_sha256", "compute.cpp")):
        matches = [p for p in snapshots if Path(p).name == filename]
        if len(matches) != 1 or generated.get(field) != inventory[matches[0]]:
            raise ValueError("generated source dependency mismatch: " + field)
    cpp = [p for p in snapshots if Path(p).suffix == ".cpp" and Path(p).parent == Path(binary).parent]
    if len(cpp) != 1 or generated.get("generated_sha256") != inventory[cpp[0]]:
        raise ValueError("generated native source mismatch")
    frozen_sha = manifest["model_frozen_before_heldout_sha256"]
    if a.sha(experiment / "prediction-model.json") != frozen_sha:
        raise ValueError("original frozen component model changed")
    flags = manifest["observed_child_flags"]
    if flags.get("GGML_CALIBRATION_BATCH_REPETITIONS") != str(suite["batch_repetitions"]):
        raise ValueError("batch repetition environment mismatch")
    dispatch = base.dispatch_cpu_id(flags, suite["protocol"]["cpu_id"])
    for row in rows:
        declared = expected[row["id"]]
        if any(row[k] != declared[k] for k in ("case", "split", "seed", "repeat")):
            raise ValueError("row geometry/split/seed/repeat mismatch")
        path = experiment / "raw" / (row["id"] + ".json")
        if str(path) != row["raw_path"] or a.sha(path) != row["raw_sha256"]:
            raise ValueError("raw process path/hash mismatch")
        record = base.document(path)
        command = base.command(binary, row["case"], suite["protocol"], row["seed"])
        if row["command"] != command or record["command"] != command or record["observed_child_flags"] != flags:
            raise ValueError("native command/environment mismatch")
        if row.get("child_pid") != record.get("child_pid"):
            raise ValueError("native PID mismatch")
        if row["split"] == "heldout" and (row.get("frozen_model_sha256_at_start") != frozen_sha
                or record.get("frozen_model_sha256_at_start") != frozen_sha
                or datetime.fromisoformat(record["started_utc"]) < datetime.fromisoformat(manifest["model_frozen_utc"])):
            raise ValueError("original heldout process not bound to prior component freeze")
        if row["status"] == "measured":
            if type(record.get("child_pid")) is not int or record.get("returncode") != 0 or record.get("timed_out") or record.get("validation_error"):
                raise ValueError("failed native process saved as measured")
            payload = object_document(record["stdout"])
            normalized = batch.normalized_payload(payload, row, suite, dispatch)
            if row.get("metrics") != normalized or row.get("raw_batch_metrics") != payload["batch_wall_ns"]:
                raise ValueError("raw totals/normalization mismatch")
            if row.get("raw_batch_statistics") != {k: a.stats(v) for k, v in payload["batch_wall_ns"].items()}:
                raise ValueError("raw batch statistics mismatch")
            if row.get("batch_repetitions") != suite["batch_repetitions"]:
                raise ValueError("row batch unit mismatch")
            if any(row[k] != payload[k] for k in ("checks", "workspaces", "ports", "boundary")):
                raise ValueError("raw physical evidence mismatch")
        elif row["status"] != "failed" or not record.get("validation_error"):
            raise ValueError("unknown status/missing failure evidence")
    recipes = {case["id"]: batch.recipes(case) for case in suite["cases"]}
    aliases = {case["id"]: base.observed_aliases(case) for case in suite["cases"]}
    original_model = a.freeze_model(rows, recipes, aliases)
    if original_model != base.document(experiment / "prediction-model.json"):
        raise ValueError("original component model cannot reconstruct")
    baseline = original_report(rows, original_model, suite)
    saved_path = experiment / "analysis/report.json"
    saved = base.document(saved_path)
    if a.sha(saved_path) != manifest["report_sha256"]:
        raise ValueError("original report hash mismatch")
    for key, value in baseline.items():
        if (saved[key][:len(value)] != value if key == "unknown_or_uncovered" else saved[key] != value):
            raise ValueError("original report cannot reconstruct: " + key)
    if saved.get("gpu_capacity_proved") is not False or saved.get("older_isolated_protocol_replaced") is not False:
        raise ValueError("original capacity/scope flag changed")
    strict_result = strict_batch.verify(experiment) if not context["changed_current_metadata"] else None
    return dict(verified=True, mode="archived-evidence-reconstruction", strict_current_source_verified=strict_result is not None and strict_result["verified"],
                strict_current_source_verifier_sha256=a.sha(strict_batch.__file__),
                strict_current_source_qualification="strict verifier executed successfully" if strict_result else "strict current-source identity differs; separate archived reconstruction used without changing strict verifier",
                changed_current_metadata=context["changed_current_metadata"], measurement_rows=len(rows),
                failed_rows=[r["id"] for r in rows if r["status"] != "measured"],
                raw_totals_normalization_numeric_and_original_report_verified=True,
                original_report_sha256=a.sha(saved_path), physical_capacity_validated=False)


def original_report(rows, component_model, suite):
    groups = {case["id"]: {name: ("resident" if name.split("@", 1)[0].endswith("resident") else "preload")
        + (";dispatch_model" if "@" in name else ";ideal_model") for name in batch.recipes(case)} for case in suite["cases"]}
    return a.evaluate(rows, component_model, suite["quality"], suite["bootstrap"], scope=SCOPE,
                      frozen_before_heldout=True, comparison_groups=groups)


def prices_from_components(profile, cases):
    """Independent profile terms only; never inspect strategy target values."""
    recipes = {case["id"]: {k: v for k, v in batch.recipes(case).items() if "@" not in k} for case in cases}
    frozen = a.freeze_model(profile, recipes)
    return {c["case_id"]: {k: p["median_ns"] / 1000 for k, p in c["prices"].items()} for c in frozen["cases"]}


def grouped_folds(profile, cases):
    """Leave an entire geometry and every repeat/seed out of coefficient fitting."""
    result = []
    for case in cases:
        excluded = case["id"]
        training = [r for r in profile if r["case"]["id"] != excluded]
        held = [r for r in profile if r["case"]["id"] == excluded]
        result.append(dict(excluded_case_id=excluded, training_rows=training, test_component_rows=held))
    return result


def build_bundle(profile, suite, label):
    if not profile or any(r["split"] != "profile" for r in profile):
        raise ValueError("coefficient fitting accepts profile rows only")
    measured = [r for r in profile if r["status"] == "measured"]
    model = overhead.fit(measured)
    bundle = dict(schema=SCHEMA, label=label, evaluation_design="retrospective; historically inspected heldout data",
                  fresh_blind_validation=False, joint_chain_totals_used_for_calibration=True,
                  training_row_ids=[r["id"] for r in measured],
                  failed_training_rows_retained=[r["id"] for r in profile if r["status"] != "measured"],
                  model=model, component_prices_us=prices_from_components(profile, suite["cases"]), folds=[])
    for fold in grouped_folds(profile, suite["cases"]):
        train = [r for r in fold["training_rows"] if r["status"] == "measured"]
        bundle["folds"].append(dict(excluded_case_id=fold["excluded_case_id"],
            training_row_ids=[r["id"] for r in train], model=overhead.fit(train),
            test_component_row_ids=[r["id"] for r in fold["test_component_rows"]],
            excluded_shape_component_prices_calibrated=True,
            excluded_shape_joint_totals_used_for_coefficient_fit=False))
    return bundle


def freeze_bundle(profile, suite, label, output):
    bundle = build_bundle(profile, suite, label)
    a.dump(output, bundle)
    return bundle, a.sha(output)


def compatibility(left, right):
    """Transfer requires identical executed binary, physical protocol and geometry."""
    ls, rs = left["suite"], right["suite"]
    checks = {"same_geometry": ls["cases"] == rs["cases"], "same_protocol": ls["protocol"] == rs["protocol"],
              "same_batch_repetitions": ls["batch_repetitions"] == rs["batch_repetitions"],
              "same_timing_unit": left["manifest"]["timing_unit"] == right["manifest"]["timing_unit"],
              "same_normalization": left["manifest"]["normalization"] == right["manifest"]["normalization"],
              "same_child_flags": left["manifest"]["observed_child_flags"] == right["manifest"]["observed_child_flags"],
              "transfer_seeds_disjoint": not (set(ls["profile_seeds"]) & set(rs["heldout_seeds"])),
              "all_campaign_seeds_disjoint": not (set(ls["profile_seeds"] + ls["heldout_seeds"]) & set(rs["profile_seeds"] + rs["heldout_seeds"]))}
    def physical(context):
        return {str(Path(path).name): digest for path, digest in context["manifest"]["inventory_before"].items()
                if Path(path).suffix.lower() in (".exe", ".dll", ".so", ".dylib", ".cpp") or Path(path).name == "generate_probe.py"}
    checks["same_physical_binary_source_runtime_signatures"] = physical(left) == physical(right)
    return dict(compatible=all(checks.values()), checks=checks,
                source_profile_seeds=ls["profile_seeds"], target_heldout_seeds=rs["heldout_seeds"],
                source_metadata_differences=left["changed_current_metadata"],
                qualification="Exact executed physical identities; CMake/collection metadata differences do not establish a new physical protocol.")


def error_summary(results, cases):
    families = {case["id"]: case["family"] for case in cases}
    def summarize(values):
        known = [r for r in values if r.get("observed_ns") is not None and r.get("predicted_ns") is not None]
        return dict(declared=len(values), evaluated=len(known), unknown=len(values)-len(known),
                    accepted=sum(r["calibrated"] for r in values),
                    median_absolute_relative_error=statistics.median([r["absolute_relative_error"] for r in known]) if known else None,
                    mean_signed_bias_us=statistics.mean([r["signed_error_ns"] / 1000 for r in known]) if known else None,
                    mean_signed_relative_bias=statistics.mean([r["signed_error_ns"] / r["observed_ns"] for r in known]) if known else None)
    return dict(overall=summarize(results), by_family={family: summarize([r for r in results if families[r["case_id"]] == family])
                for family in sorted(set(families.values()))})


def target_quality(profile, model, quality):
    """Expose variability of joint totals newly used in coefficient fitting.

    This is additional calibration provenance. It never changes the original
    component/heldout gates or their numerical thresholds.
    """
    ids = {source["row_id"] for source in model["training"]["rows"]}
    sources, groups = [], {}
    for row in profile:
        if row["id"] not in ids:
            continue
        for strategy in overhead.primary_strategies(row["case"]):
            source = dict(row_id=row["id"], case_id=row["case"]["id"], strategy=strategy,
                          raw_sha256=row["raw_sha256"], **a.stats(row["metrics"][strategy]))
            sources.append(source)
            groups.setdefault((source["case_id"], strategy), []).append(source)
    summaries = []
    for (cid, strategy), values in sorted(groups.items()):
        medians = [value["median_ns"] for value in values]
        cv = statistics.pstdev(medians)/statistics.mean(medians) if len(medians) >= 2 else None
        summaries.append(dict(case_id=cid, strategy=strategy, process_count=len(values),
            max_within_process_cv=max(value["cv"] for value in values), between_process_cv=cv,
            within_process_cv_pass=all(value["cv"] <= quality["max_within_process_cv"] for value in values),
            between_process_cv_pass=cv is not None and cv <= quality["max_between_process_cv"]))
    return dict(thresholds=copy.deepcopy(quality), sources=sources, paths=summaries,
        stability_accepted=bool(summaries) and all(path["within_process_cv_pass"] and path["between_process_cv_pass"] for path in summaries),
        original_quality_gate_definitions_unchanged=True)


def rankings(results):
    """Resident paths share a boundary; singleton preload paths cannot rank."""
    grouped = {}
    for row in results:
        if row["strategy"].endswith("resident"):
            grouped.setdefault(row["case_id"], []).append(row)
    output = []
    for cid, declared in sorted(grouped.items()):
        known = [r for r in declared if r.get("observed_ns") is not None and r.get("predicted_ns") is not None]
        if len(known) < 2:
            output.append(dict(case_id=cid, unknown_reason="missing comparable resident predictions", latency_quality_accepted=False))
            continue
        po = sorted(known, key=lambda r: (r["predicted_ns"], r["strategy"]))
        oo = sorted(known, key=lambda r: (r["observed_ns"], r["strategy"]))
        left, right = known[:2]
        li, ri = left["heldout_median_95pct_interval_ns"], right["heldout_median_95pct_interval_ns"]
        decisive = li[1] < ri[0] or ri[1] < li[0]
        output.append(dict(case_id=cid, boundary_contract="resident;ideal_model", predicted_order=[r["strategy"] for r in po],
            observed_order=[r["strategy"] for r in oo], median_order_correct=[r["strategy"] for r in po] == [r["strategy"] for r in oo],
            observed_95pct_intervals_disjoint=decisive, selection_regret_ns=po[0]["observed_ns"]-oo[0]["observed_ns"],
            selection_regret_relative=(po[0]["observed_ns"]-oo[0]["observed_ns"])/oo[0]["observed_ns"],
            latency_quality_accepted=len(known)==len(declared) and all(r["calibrated"] for r in known)))
    return output


def evaluate(profile, heldout, suite, bundle, *, fold_models=False):
    rows = profile + heldout
    component_model = a.freeze_model(profile, {c["id"]: batch.recipes(c) for c in suite["cases"]},
                                    {c["id"]: base.observed_aliases(c) for c in suite["cases"]})
    baseline = original_report(rows, component_model, suite)
    original = [r for r in baseline["results"] if "@" not in r["strategy"]]
    corrected = []
    cases = {case["id"]: case for case in suite["cases"]}
    folds = {fold["excluded_case_id"]: fold for fold in bundle["folds"]}
    full_target_quality = target_quality(profile, bundle["model"], suite["quality"])
    fold_target_quality = {cid: target_quality(profile, fold["model"], suite["quality"]) for cid, fold in folds.items()} if fold_models else {}
    for old in original:
        row = copy.deepcopy(old)
        model = folds[row["case_id"]]["model"] if fold_models else bundle["model"]
        try:
            prediction = overhead.predict(cases[row["case_id"]], row["strategy"], bundle["component_prices_us"].get(row["case_id"], {}), model)
            row["composition_prediction"] = prediction
            row["predicted_ns"] = prediction["total_us"] * 1000
            if row.get("observed_ns") is not None:
                observed, pred = row["observed_ns"], row["predicted_ns"]
                row.update(signed_error_ns=pred-observed, absolute_error_ns=abs(pred-observed), absolute_relative_error=abs(pred-observed)/observed)
                row["gates"]["heldout_relative_error"] = row["absolute_relative_error"] <= suite["quality"]["max_heldout_relative_error"]
                row["calibrated"] = all(row["gates"].values())
        except (ValueError, KeyError, TypeError) as exc:
            row.update(predicted_ns=None, calibrated=False, unknown_reason="composition model unavailable: " + str(exc))
        target_stability = fold_target_quality[row["case_id"]] if fold_models else full_target_quality
        row["calibration_target_stability_accepted"] = target_stability["stability_accepted"]
        row["composition_calibration_quality_accepted"] = row["calibrated"] and target_stability["stability_accepted"]
        row["original_freeze_gate_scope"] = "original component model frozen before collection; correction model frozen only before this retrospective offline scoring"
        corrected.append(row)
    return dict(evaluation_design="retrospective", fresh_blind_validation=False,
        joint_chain_totals_used_for_calibration=True, profile_target_totals_used_for_coefficients=True,
        heldout_target_totals_used_for_coefficients=False, grouped_leave_one_case_out=fold_models,
        quality=copy.deepcopy(suite["quality"]), quality_gate_definitions="unchanged original within-process CV, between-process CV, numeric/process coverage, original frozen-before-collection and heldout error; only prediction/error changes",
        model_frozen_before_this_offline_evaluation=True, fresh_model_frozen_before_original_heldout_collection=False,
        original_collection_component_model_was_frozen_before_heldout=True,
        calibration_target_quality=full_target_quality, grouped_calibration_target_quality=fold_target_quality,
        baseline_results=original, corrected_results=corrected,
        baseline_summary=error_summary(original, suite["cases"]), corrected_summary=error_summary(corrected, suite["cases"]),
        baseline_rankings=rankings(original), corrected_rankings=rankings(corrected),
        failed_native_rows=baseline["failed_native_rows"], numeric_execution_valid=baseline["numeric_execution_valid"],
        gpu_private_workspace_bytes=None, gpu_workspace_known=False, physical_capacity_validated=False,
        unknown_or_uncovered=baseline["unknown_or_uncovered"] + [
            "Only three shared coefficients and four retained geometries; no broad shape generalization claim.",
            "Grouped folds exclude all joint totals of one geometry from coefficient fitting, but retain independent component calibration at that exact geometry.",
            "Historically inspected heldout rows make this retrospective evaluation; a new seed/process campaign is needed for fresh validation."])


def report_markdown(report):
    def percent(value):
        return "unknown" if value is None else f"{value:.2%}"
    def number(value):
        return "unknown" if value is None else f"{value:.3f}"
    lines = ["# Retrospective composition calibration", "", "This is an offline retrospective evaluation of historically inspected batch data. It is not fresh blind validation.", "",
        "Profile split joint chain totals now train three shared composition coefficients. Heldout totals enter evaluation only after every full and grouped-fold model is saved and hashed. All original assets stay immutable.", "",
        "Original CV, process coverage and error gates are retained. A smaller prediction error does not repair a failed CV gate. GPU private workspace remains null and physical capacity remains unvalidated.", ""]
    sections = dict(report["campaigns"])
    if "evaluation" in report["cross_campaign_transfer"]:
        sections["initial profile → confirmation heldout"] = {"zero_refit": report["cross_campaign_transfer"]["evaluation"]}
    for label, campaign in sections.items():
        for kind, result in campaign.items():
            old, new = result["baseline_summary"]["overall"], result["corrected_summary"]["overall"]
            lines += [f"## {label}: {kind}", "", f"Primary paths: {old['declared']}. Median relative error: {percent(old['median_absolute_relative_error'])} → {percent(new['median_absolute_relative_error'])}. Accepted: {old['accepted']} → {new['accepted']}.", "",
                "| Case | Strategy | Old us | Correction us | Corrected us | Observed us | Old error | New error | Accepted | Failed gates |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
            for before, after in zip(result["baseline_results"], result["corrected_results"]):
                if after.get("predicted_ns") is None or after.get("observed_ns") is None:
                    lines.append(f"| {after['case_id']} | {after['strategy']} | | | | | | | false | {after.get('unknown_reason')} |")
                    continue
                failed = ", ".join(k for k, v in after["gates"].items() if not v)
                lines.append(f"| {after['case_id']} | {after['strategy']} | {before['predicted_ns']/1000:.3f} | {after['composition_prediction']['correction_us']:.3f} | {after['predicted_ns']/1000:.3f} | {after['observed_ns']/1000:.3f} | {before['absolute_relative_error']:.2%} | {after['absolute_relative_error']:.2%} | {after['calibrated']} | {failed} |")
            lines += ["", "Per-family errors and signed bias:", "", "| Family | Old median error | New median error | Old mean signed bias us | New mean signed bias us |", "|---|---:|---:|---:|---:|"]
            for family, oldfam in result["baseline_summary"]["by_family"].items():
                newfam = result["corrected_summary"]["by_family"][family]
                lines.append(f"| {family} | {percent(oldfam['median_absolute_relative_error'])} | {percent(newfam['median_absolute_relative_error'])} | {number(oldfam['mean_signed_bias_us'])} | {number(newfam['mean_signed_bias_us'])} |")
            target_quality = result["calibration_target_quality"]
            failed_targets = sum(not path["within_process_cv_pass"] or not path["between_process_cv_pass"] for path in target_quality["paths"])
            lines += ["", f"Profile chain-total stability: {len(target_quality['paths'])-failed_targets}/{len(target_quality['paths'])} paths pass the existing CV thresholds. Newly fitted target statistics are retained separately from original quality gates."]
            lines += ["", "Resident comparisons (each case has a common input/output placement boundary):", ""]
            for oldrank, rank in zip(result["baseline_rankings"], result["corrected_rankings"]):
                lines.append(f"- {rank['case_id']}: old order correct {oldrank.get('median_order_correct')}; corrected {rank.get('median_order_correct')}; regret {percent(oldrank.get('selection_regret_relative'))} → {percent(rank.get('selection_regret_relative'))}; CV/error quality accepted {rank['latency_quality_accepted']}.")
            lines.append("")
    transfer = report["cross_campaign_transfer"]
    lines += ["## Initial profile → confirmation heldout", "", f"Physical protocol compatible: {transfer['compatibility']['compatible']}."]
    if "evaluation" in transfer:
        old, new = transfer["evaluation"]["baseline_summary"]["overall"], transfer["evaluation"]["corrected_summary"]["overall"]
        lines += [f"Median relative error {percent(old['median_absolute_relative_error'])} → {percent(new['median_absolute_relative_error'])}; accepted {old['accepted']} → {new['accepted']}. Both initial component prices and initial coefficients stay frozen; no refitting."]
    lines += ["", "Only four geometries and three shared coefficients are available. Leave-one-case-out uses independent component probes at the excluded geometry; it tests coefficient transfer, not unprofiled-shape prediction. Batch means do not certify isolated-request jitter, general CUDA overlap, complete inference, or capacity.", ""]
    return "\n".join(lines)


def source_inventory():
    sources = [Path(__file__), HERE / "model.py", HERE / "verify.py", Path(a.__file__), Path(base.__file__), Path(batch.__file__), PROFILING / "batched_calibration/verify_batch.py"]
    return {str(p.resolve()): a.sha(p) for p in sources if p.is_file()}


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output must be a new directory")
    contexts = {"initial": preflight(args.initial), "confirmation": preflight(args.confirmation)}
    before = source_inventory()
    output.mkdir(parents=True)
    stages, bundles, profiles, hashes = [], {}, {}, {}
    stages.append(dict(stage="preflight_without_heldout_deserialization", utc=now()))
    for label, context in contexts.items():
        profiles[label] = list(profile_objects(context["experiment"] / "measurements.json"))
        expected = {r["id"] for r in base.jobs(context["suite"], "profile")}
        if len(profiles[label]) != len(expected) or {r["id"] for r in profiles[label]} != expected:
            raise ValueError("profile process coverage mismatch")
        path = output / (label + "-frozen-model.json")
        bundles[label], hashes[label] = freeze_bundle(profiles[label], context["suite"], label, path)
        stages.append(dict(stage="full_and_all_grouped_models_saved_and_hashed", campaign=label, path=str(path), sha256=hashes[label], utc=now()))
    a.dump(output / "freeze-ledger.json", dict(schema=SCHEMA, model_sha256=hashes, stages=stages,
        historically_inspected_data=True, fresh_blind_validation=False))
    reports, heldouts, audits = {}, {}, {}
    for label, context in contexts.items():
        if a.sha(output / (label + "-frozen-model.json")) != hashes[label]:
            raise ValueError("frozen model changed before heldout read")
        stages.append(dict(stage="heldout_totals_deserialization_started", campaign=label, utc=now()))
        rows = base.document(context["experiment"] / "measurements.json")
        if [r for r in rows if r["split"] == "profile"] != profiles[label]:
            raise ValueError("profile rows changed between stages")
        audits[label] = verify_campaign(context, rows)
        heldouts[label] = [r for r in rows if r["split"] == "heldout"]
        reports[label] = {"full": evaluate(profiles[label], heldouts[label], context["suite"], bundles[label]),
                          "grouped_leave_one_case_out": evaluate(profiles[label], heldouts[label], context["suite"], bundles[label], fold_models=True)}
    compatible = compatibility(contexts["initial"], contexts["confirmation"])
    transfer = dict(compatibility=compatible)
    if compatible["compatible"]:
        transfer["evaluation"] = evaluate(profiles["initial"], heldouts["confirmation"], contexts["confirmation"]["suite"], bundles["initial"])
        transfer["evaluation"]["transfer_mode"] = "zero refit: both initial profile component prices and initial composition coefficients frozen"
    report = dict(schema=SCHEMA, evaluation_design="retrospective", fresh_blind_validation=False,
                  joint_chain_totals_used_for_calibration=True, heldout_joint_totals_used_for_fitting=False,
                  campaigns=reports, cross_campaign_transfer=transfer, input_verification=audits,
                  gpu_private_workspace_bytes=None, physical_capacity_validated=False)
    a.dump(output / "report.json", report)
    (output / "REPORT.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    after = source_inventory()
    original_after = {label: {path: a.sha(path) for path in context["original_files"]} for label, context in contexts.items()}
    originals = {label: context["original_files"] for label, context in contexts.items()}
    if before != after or originals != original_after:
        raise ValueError("source or original campaign asset changed during offline evaluation")
    for label in contexts:
        if a.sha(output / (label + "-frozen-model.json")) != hashes[label]:
            raise ValueError("frozen model changed during evaluation")
    manifest = dict(schema=SCHEMA, source_modules_before=before, source_modules_after=after, sources_unchanged=True,
        input_paths={label: str(context["experiment"]) for label, context in contexts.items()},
        original_assets_before=originals, original_assets_after=original_after, original_assets_unchanged=True,
        model_sha256=hashes, report_sha256=a.sha(output / "report.json"), stages=stages,
        joint_chain_totals_used_for_calibration=True, fresh_blind_validation=False,
        gpu_private_workspace_bytes=None, physical_capacity_validated=False, completed_utc=now())
    a.dump(output / "manifest.json", manifest)
    print(json.dumps({label: result["full"]["corrected_summary"]["overall"] for label, result in reports.items()}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        return run(parser.parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("composition calibration: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
