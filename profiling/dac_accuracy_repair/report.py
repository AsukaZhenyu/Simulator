"""Read-only evidence audit and Chinese report for the frozen repair campaign.

No native executable, parameter fitter or acquisition command is called here.
The builder refuses to run before a final summary exists and creates new report
files only. It recomputes statistics from retained raw intervals; it never
changes the campaign, frozen prices, old results, or acceptance thresholds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
DEFAULT_BUILD = ROOT / "build/dac-accuracy-repair-20261026"
DEFAULT_REPORT = PROJECT / "paper-write/第八教学周/experiment-reports/dac-accuracy-repair-20261026"
SPLITS = ("transfer-profile", "profile", "transfer-heldout", "dense")
MODE_LABELS = {"pageable": "普通可分页内存", "registered": "显式注册内存"}
PROVIDER_LABELS = {"hot_ns": "主结果：独立 hot 动作价格", "transfer_secondary_ns": "次结果：替换 D2H 价格（有假设）"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def csv_new(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values)


def cv(values):
    return statistics.stdev(values) / mean(values)


def higher_quantile(values, probability=.9):
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * probability)]


def transfer_value(parameters, size):
    """Independent direct evaluation of the already frozen transparent family."""
    split = parameters["breakpoint_bytes"]
    small = size if split is None else min(size, split)
    large = 0 if split is None else max(0, size - split)
    return parameters["fixed_ns"] + parameters["small_ns_per_byte"] * small + parameters["large_ns_per_byte"] * large


def audit(campaign, summary):
    """Independently recompute raw metrics and freeze/source boundaries.

    The action-coordinate serializer is reused to identify an action, but no
    production evaluator or fitter is used to recompute numerical gates.
    """
    from llm_infer_model.performance.reusable_action_costs import coordinate, digest, semantic_action
    import numpy as np

    campaign = Path(campaign).resolve()
    manifest = read(campaign / "manifest.json")
    barrier = read(campaign / "frozen-barrier.json")
    checks, sources, cached = [], {}, {}

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def bound(item):
        path = Path(item["path"])
        exists = path.is_file()
        actual = sha(path) if exists else None
        key = str(path.resolve())
        sources[key] = dict(path=key, sha256=actual, expected_sha256=item["sha256"])
        return exists and actual == item["sha256"]

    def load(item):
        path = str(Path(item["path"]).resolve())
        if path not in cached:
            cached[path] = read(path)
        return cached[path]

    def equal(name, actual, expected):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
            passed = math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-7)
        else:
            passed = actual == expected
        check(name, passed, actual=actual, expected=expected)

    def at(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    for item in [receipt(campaign / "manifest.json"), receipt(campaign / "summary.json"), receipt(campaign / "frozen-barrier.json")]:
        bound(item)
    implementation_ok = all([bound(item["source"]) for item in manifest["sources"]] + [bound(item["archived"]) for item in manifest["sources"]])
    implementation_ok = bound(manifest["binary"]) and implementation_ok
    implementation_ok = all([bound(item) for item in manifest["runtime"]]) and implementation_ok
    check("frozen_implementation_runtime_hashes", implementation_ok)
    check("mathematical_program_and_oracle_receipts", all([bound(binding[field]) for binding in manifest["programs"].values() for field in ("program", "reference")]))
    check("frozen_parameter_receipts", all([bound(dict(path=str(campaign / name), sha256=value)) for name, value in barrier["files"].items()]))
    check("no_target_fitting_declared", summary["target_time_used_for_fit"] is False and manifest["protocol"]["target_time_used_for_fit"] is False and barrier["heldout_read"] is False)
    check("old_results_preserved_declared", summary["old_results_modified"] is False and manifest["protocol"]["original_results_modified"] is False)
    check("quality_thresholds_unchanged", manifest["protocol"]["quality"] == dict(max_process_mean_ape=.1, max_trial_average_p90_ape=.2, max_process_trial_cv=.1, max_observer_relative_mean_difference=.05))
    train = manifest["protocol"]["transfer_train_sizes"]
    held = manifest["protocol"]["transfer_heldout_sizes"]
    check("transfer_size_split_disjoint", not (set(train) & set(held)) and len(train) == len(held) == 8)

    executions = []
    by_split = {}
    for split in SPLITS:
        file = campaign / f"{split}-executions.json"
        bound(receipt(file))
        by_split[split] = read(file)
        executions.extend(by_split[split])
    equal("planned_execution_inventory", sorted(e["job"]["id"] for e in executions), sorted(j["id"] for j in manifest["jobs"]))
    planned_jobs = {job["id"]: job for job in manifest["jobs"]}
    check("complete_frozen_job_request_bindings", all(e["job"] == planned_jobs.get(e["job"]["id"]) for e in executions))
    equal("declared_count", summary["counts"]["declared"], len(manifest["jobs"]))
    equal("completed_count", summary["counts"]["completed"], sum(e["status"] == "completed" for e in executions))
    equal("failed_count", summary["counts"]["failed"], sum(e["status"] != "completed" for e in executions))
    check("all_owned_children_exited", all(e.get("process_exited_verified") for e in executions))
    pids = [e["process_id"] for e in executions if e.get("process_id")]
    # PID reuse after exit is legal and does not indicate process reuse. Distinct
    # per-job outputs plus nonoverlapping lifetimes support the launch ledger.
    native_paths = [e["native"]["path"] for e in executions if "native" in e]
    check("distinct_per_job_raw_outputs", len(native_paths) == len(set(native_paths)))
    lifetime_ok = all(at(e["started_at_utc"]) <= at(e["completed_at_utc"]) for e in executions)
    for pid in set(pids):
        same_pid = sorted((e for e in executions if e.get("process_id") == pid), key=lambda e: e["started_at_utc"])
        lifetime_ok = lifetime_ok and all(at(a["completed_at_utc"]) <= at(b["started_at_utc"]) for a, b in zip(same_pid, same_pid[1:]))
    check("PID_generation_lifetime_ledger", lifetime_ok, reused_pid_count=len(pids) - len(set(pids)))
    frozen_at = at(barrier["frozen_at_utc"])
    check("profile_finished_before_freeze", all(at(e["completed_at_utc"]) <= frozen_at for split in ("profile", "transfer-profile") for e in by_split[split]))
    check("target_started_after_freeze", all(at(e["started_at_utc"]) >= frozen_at for split in ("transfer-heldout", "dense") for e in by_split[split]))
    for filename in barrier["files"]:
        item = read(campaign / filename)
        if isinstance(item, dict) and "frozen_at_utc" in item:
            check("parameter_frozen_before_barrier:" + filename, at(item["frozen_at_utc"]) <= frozen_at)

    raw_by_id, transfer_raw, dense_raw, action_samples = {}, {}, {}, {}
    numerical_ok = clock_ok = state_ok = native_identity_ok = affinity_ok = memory_ok = prep_ok = runtime_ok = True
    for execution in executions:
        if execution["status"] != "completed":
            continue
        native_identity_ok = bound(execution["native"]) and native_identity_ok
        native = load(execution["native"])
        job = execution["job"]
        raw_by_id[job["id"]] = native
        binding = manifest["programs"][job["program_key"]]
        program, oracle = load(binding["program"]), load(binding["reference"])
        request = job["request"]
        runtime_ok = runtime_ok and native["observed_runtime_flags"] == dict(GGML_CUDA_DISABLE_GRAPHS="1", NVIDIA_TF32_OVERRIDE="0", matmul_precision="GGML_PREC_F32") and program.get("threads", program.get("physical", {}).get("threads")) == 1
        native_identity_ok = native_identity_ok and native["request"] == request and native["process_id"] == execution["process_id"] and native["program_sha256"] == program["program_sha256"] and native["source_sha256"] == manifest["native_source_sha256"] and native["caller_source_sha256_match"]
        process = native["process_affinity"]
        thread = native["main_thread_affinity"]
        affinity_ok = affinity_ok and process["requested_mask"] == process["observed_mask"] == process["final_mask"] == manifest["process_affinity_mask"] and process["applied_before_backend_initialization"] and thread["requested_mask"] == thread["observed_mask"] == thread["final_mask"] == 4 and thread["restored"]
        groups = [native["trials"], *[g["trials"] for g in native.get("groups", [])]]
        frequency = native["clock"]["frequency_hz"]
        for rows in groups:
            for row in rows:
                clock_ok = clock_ok and row["end_tick"] > row["begin_tick"] and row["repeats"] == request["sampling"]["repeats"] and abs((row["end_tick"] - row["begin_tick"]) * 1e9 / frequency - row["wall_ns"]) <= 1
                state = row["preblock_state"]
                state_ok = state_ok and state["sample_tick"] <= row["begin_tick"] and state["derived_from_target_duration"] is False and state["outside_timed_interval"]
        if job["split"].startswith("transfer"):
            visit = native["visits"][0]
            expected = "registered_host" if job["memory_mode"] == "registered" else "unregistered_pageable"
            memory_ok = memory_ok and visit["host_pointer"].get(expected) is True and native["host_registration_restored"] and all(r["host_pointer"].get(expected) is True for r in native["trials"])
            proof = visit["full_byte_verification"]
            numerical_ok = numerical_ok and proof["full_byte_equality"] and proof["checked_bytes"] == oracle["bytes"] and proof["source_sha256"] == proof["output_sha256"] == oracle["sha256"]
            bare = [r["wall_ns"] / r["repeats"] for r in native["trials"] if not r["actual_stamped"]]
            stamped = [r["wall_ns"] / r["repeats"] for r in native["trials"] if r["actual_stamped"]]
            transfer_raw[job["id"]] = dict(bytes=binding["bytes"], memory_mode=job["memory_mode"], mean_ns=mean(bare), cv=cv(bare), observer_effect=mean(stamped) / mean(bare) - 1, trial_ns=bare, raw=execution["native"])
        else:
            proof = native["verification_native"]
            numerical_ok = numerical_ok and proof["copy_values_equal"] and proof["actions"] == program["actions"] and native["native_repeat_consistency"]["passed"] and set(proof["ports"]) == set(oracle)
            for key, wanted in oracle.items():
                observed = np.asarray(proof["ports"][key])
                wanted = np.asarray(wanted)
                numerical_ok = numerical_ok and observed.shape == wanted.shape and bool(np.all(np.isfinite(observed))) and bool(np.allclose(observed, wanted, atol=5e-5, rtol=5e-5))
            if job["split"] == "profile":
                for group in native["groups"]:
                    key = digest(coordinate(program, group["action_index"], manifest["context"]))
                    action_samples.setdefault(key, []).extend(Fraction(str(r["wall_ns"])) / r["repeats"] for r in group["trials"])
            else:
                expected_prep = dict(reset_before_batch=True, untimed_complete_prime_passes=1,
                    timed_replays_per_batch=8, fixed_input_replay=True,
                    warmup_iterations_min=request["sampling"]["warmup"], warmup_ms_min=request["sampling"]["warmup_min_ms"])
                prep_ok = prep_ok and native["preparation_contract"] == expected_prep and all(r["preparation_contract"] == expected_prep for r in native["trials"])
                bare = [r["wall_ns"] / r["repeats"] for r in native["trials"] if not r["actual_stamped"]]
                row = dict(mean_ns=mean(bare), cv=cv(bare), trial_ns=bare, preparation_contract=native["preparation_contract"], raw=execution["native"])
                if job["mode"] == "dense-observer":
                    stamped = [r["wall_ns"] / r["repeats"] for r in native["trials"] if r["actual_stamped"]]
                    row["observer_effect"] = mean(stamped) / mean(bare) - 1
                dense_raw[job["id"]] = row
    for name, passed in (("native_process_request_source_identity", native_identity_ok), ("QPC_intervals_and_repeats", clock_ok), ("preblock_observations_outside_future_target", state_ok), ("actual_process_and_main_thread_affinity", affinity_ok), ("actual_host_memory_and_registration_restore", memory_ok), ("independent_numpy_or_full_byte_oracle", numerical_ok), ("primary_observer_reset_prime_eight_replay_contract", prep_ok), ("one_compute_thread_and_explicit_precision_flags", runtime_ok)):
        check(name, passed)

    action_profile = read(campaign / "action-profile-frozen.json")
    entries = {e["key"]: e for e in action_profile["entries"]}
    equal("action_coordinate_inventory", sorted(entries), sorted(action_samples))
    for key, samples in action_samples.items():
        check("raw_action_price:" + key, Fraction(entries[key]["mean_ns"]) == sum(samples, Fraction()) / len(samples))
    frozen_predictions = read(campaign / "predictions.json")
    models = {mode: read(campaign / f"transfer-{mode}-frozen.json") for mode in ("pageable", "registered")}
    for mode, model in models.items():
        check("transfer_train_only:" + mode, model["heldout_rows_used"] is False and model["complete_strategy_times_used"] is False and all(r["split"] == "calibration" and r["bytes"] in train and r["memory_mode"] == mode for r in model["calibration_rows"]))
        for i, row in enumerate(model["calibration_rows"]):
            matches = [r for r in transfer_raw.values() if r["raw"]["path"] == row["raw"]["path"]]
            check(f"transfer_calibration_raw_binding:{mode}:{i}", len(matches) == 1 and math.isclose(matches[0]["mean_ns"], row["mean_ns"], rel_tol=1e-12))
        expected = summary["D2H"][mode]
        local = [transfer_raw[e["job"]["id"]] for e in by_split["transfer-heldout"] if e["job"]["memory_mode"] == mode and e["status"] == "completed"]
        equal("heldout_process_inventory:" + mode, len(local), 24)
        raw_by_path = {row["raw"]["path"]: row for row in local}
        equal("summary_transfer_process_paths:" + mode, sorted(row["raw"]["path"] for row in expected["processes"]), sorted(raw_by_path))
        for row in expected["processes"]:
            actual = raw_by_path[row["raw"]["path"]]
            for field in ("bytes", "memory_mode", "mean_ns", "cv", "observer_effect", "trial_ns"):
                equal(f"summary_transfer_process:{mode}:{Path(row['raw']['path']).parent.name}:{field}", row[field], actual[field])
            prediction = transfer_value(model["parameters"], actual["bytes"])
            equal("summary_transfer_process_price:" + row["raw"]["path"], row["predicted_ns"], prediction)
            equal("summary_transfer_process_ape:" + row["raw"]["path"], row["ape"], abs(prediction - actual["mean_ns"]) / actual["mean_ns"])
        process_apes, errors = [], []
        size_apes = []
        for size in held:
            group = [r for r in local if r["bytes"] == size]
            prediction = transfer_value(model["parameters"], size)
            observed = mean(r["mean_ns"] for r in group)
            size_apes.append(abs(prediction - observed) / observed)
            record = next(r for r in expected["sizes"] if r["bytes"] == size)
            equal(f"size_observed:{mode}:{size}", record["observed_ns"], observed)
            equal(f"size_predicted:{mode}:{size}", record["predicted_ns"], prediction)
            equal(f"size_between_process_cv:{mode}:{size}", record["between_process_cv"], cv([r["mean_ns"] for r in group]))
            equal(f"same_input_feasible:{mode}:{size}", record["same_input_ten_percent_feasible"], max(.9 * r["mean_ns"] for r in group) <= min(1.1 * r["mean_ns"] for r in group))
            for row in group:
                process_apes.append(abs(prediction - row["mean_ns"]) / row["mean_ns"])
                errors.extend(abs(prediction - value) / value for value in row["trial_ns"])
        max_size, max_process, p90 = max(size_apes), max(process_apes), higher_quantile(errors)
        cv_passes = sum(r["cv"] <= .1 for r in local)
        observer_passes = sum(abs(r["observer_effect"]) <= .05 for r in local)
        original = max_size <= .1 and cv_passes == observer_passes == 24
        strict = original and max_process <= .1 and p90 <= .2
        for field, actual in (("maximum_size_mean_ape", max_size), ("maximum_process_ape", max_process), ("trial_p90_ape", p90), ("cv_passes", cv_passes), ("observer_passes", observer_passes), ("qualified_original_size_mean_gate", original), ("qualified_strict_process_mean_gate", strict)):
            equal(f"transfer_gate:{mode}:{field}", expected[field], actual)

    program_results = {}
    for prediction in frozen_predictions:
        key = prediction["program_key"]
        program = load(manifest["programs"][key]["program"])
        exact = [Fraction(entries[digest(coordinate(program, i, manifest["context"]))]["mean_ns"]) for i in range(len(program["actions"]))]
        secondary = []
        for i, value in enumerate(exact):
            action = semantic_action(program, i)
            if action["kind"] == "COPY" and action["src"] == "vram" and action["dst"] == "dram":
                value = Fraction(str(transfer_value(models["pageable"]["parameters"], action["tensor"]["storage_bytes"])))
            secondary.append(value)
        check("frozen_primary_serial_sum:" + key, Fraction(prediction["hot_ns"]) == sum(exact, Fraction()))
        check("frozen_secondary_serial_sum:" + key, Fraction(prediction["transfer_secondary_ns"]) == sum(secondary, Fraction()))
        expected = next(r for r in summary["programs"] if r["program_key"] == key)
        primary = [dense_raw[e["job"]["id"]] for e in by_split["dense"] if e["job"]["program_key"] == key and e["job"]["mode"] == "dense-primary" and e["status"] == "completed"]
        observer = [dense_raw[e["job"]["id"]] for e in by_split["dense"] if e["job"]["program_key"] == key and e["job"]["mode"] == "dense-observer" and e["status"] == "completed"]
        for field, raw_rows in (("primary", primary), ("observers", observer)):
            raw_by_path = {row["raw"]["path"]: row for row in raw_rows}
            equal(f"summary_dense_process_paths:{key}:{field}", sorted(row["raw"]["path"] for row in expected[field]), sorted(raw_by_path))
            for row in expected[field]:
                actual = raw_by_path[row["raw"]["path"]]
                for metric in ("mean_ns", "cv", "trial_ns", "preparation_contract", *(["observer_effect"] if field == "observers" else [])):
                    equal(f"summary_dense_process:{key}:{field}:{Path(row['raw']['path']).parent.name}:{metric}", row[metric], actual[metric])
        prepare_match = len(primary) == len(observer) == 4 and all(a["preparation_contract"] == b["preparation_contract"] for a, b in zip(primary, observer))
        check("dense_process_inventory:" + key, len(primary) == len(observer) == 4, primary=len(primary), observer=len(observer))
        equal("dense_preparation_match:" + key, expected["preparation_match"], prepare_match)
        if not primary:
            check("dense_program_measured:" + key, False)
            continue
        equal("dense_measured_mean:" + key, expected["measured_mean_ns"], mean(r["mean_ns"] for r in primary))
        outcomes = {}
        for field in PROVIDER_LABELS:
            price = float(Fraction(prediction[field]))
            apes = [abs(price - r["mean_ns"]) / r["mean_ns"] for r in primary]
            p90 = higher_quantile([abs(price - v) / v for r in primary for v in r["trial_ns"]])
            qualified = prepare_match and all(a <= .1 for a in apes) and all(r["cv"] <= .1 for r in primary) and p90 <= .2 and all(abs(r["observer_effect"]) <= .05 for r in observer)
            outcomes[field] = qualified
            for metric, actual in (("predicted_ns", price), ("maximum_process_ape", max(apes)), ("mean_process_ape", mean(apes)), ("trial_p90_ape", p90), ("fully_qualified", qualified)):
                equal(f"dense_gate:{key}:{field}:{metric}", expected["modes"][field][metric], actual)
        program_results[key] = dict(layers=prediction["layers"], strategy=prediction["strategy"], measured=mean(r["mean_ns"] for r in primary), qualified=outcomes, prices={field: float(Fraction(prediction[field])) for field in PROVIDER_LABELS})
    equal("prepared_program_count", summary["counts"]["preparation_matched_programs"], sum(r["preparation_match"] for r in summary["programs"]))
    for field, count in (("hot_ns", "hot_qualified_programs"), ("transfer_secondary_ns", "transfer_secondary_qualified_programs")):
        equal(count, summary["counts"][count], sum(r["qualified"][field] for r in program_results.values()))
    for group in summary["groups"]:
        if "modes" not in group:
            check("complete_selection_group:" + str(group["layers"]), False)
            continue
        rows = [r for r in program_results.values() if r["layers"] == group["layers"]]
        winner = min(rows, key=lambda r: r["measured"])
        for field in PROVIDER_LABELS:
            selected = min(rows, key=lambda r: r["prices"][field])
            actual = group["modes"][field]
            equal(f"group_winner:{group['layers']}:{field}", actual["observed_mean_winner"], winner["strategy"])
            equal(f"group_selected:{group['layers']}:{field}", actual["selected"], selected["strategy"])
            equal(f"group_point_regret:{group['layers']}:{field}", actual["regret"], selected["measured"] / winner["measured"] - 1)
            equal(f"group_all_candidates_quality:{group['layers']}:{field}", actual["all_candidates_qualified"], all(r["qualified"][field] for r in rows))
    return dict(schema="dac-accuracy-repair-independent-audit-1", created_at_utc=datetime.now(timezone.utc).isoformat(), passed=all(c["passed"] for c in checks), checks=checks, check_count=len(checks), failed_checks=[c for c in checks if not c["passed"]], sources=list(sources.values()), raw_native_processes_read=len(raw_by_id), independent_statistics=True, parameter_fitting=False, native_launches=0, statistical_selection_upper_bound_computed=False)


def audit_diagnostic(diagnostic, summary):
    """Recompute the retained six-process intervention without reusing its stats."""
    diagnostic = Path(diagnostic)
    manifest, executions = read(diagnostic / "manifest.json"), read(diagnostic / "executions.json")
    checks, sources, processes = [], [], []

    def check(name, actual, expected=True):
        passed = math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10) if type(actual) in (int, float) and type(expected) in (int, float) else actual == expected
        checks.append(dict(name=name, passed=passed, actual=actual, expected=expected))

    def bound(item):
        actual = receipt(item["path"])
        sources.append(actual)
        check("receipt:" + item["path"], actual["sha256"], item["sha256"])

    for item in [receipt(diagnostic / "summary.json"), receipt(diagnostic / "manifest.json"), manifest["binary"], *[s["archived"] for s in manifest["sources"]]]:
        bound(item)
    check("six_declared_processes", len(executions), 6)
    check("diagnostic_not_fit", summary["independent_parameter_fit"], False)
    for execution in executions:
        if execution["status"] != "completed":
            check("completed:" + execution["job"]["id"], False)
            continue
        bound(execution["raw"])
        native = read(execution["raw"]["path"])
        check("source_identity:" + execution["job"]["id"], native["source_sha256"], manifest["source_sha256"])
        check("process_mask_four:" + execution["job"]["id"], execution["affinity"]["observed_mask"], 4)
        check("registered_restored:" + execution["job"]["id"], native["host_registration_restored"])
        for mode in ("pageable", "registered"):
            rows = [r for r in native["trials"] if r["memory_mode"] == mode]
            expected_kind = "registered_host" if mode == "registered" else "unregistered_pageable"
            check("actual_memory:" + execution["job"]["id"] + ":" + mode, all(r["host_pointer"].get(expected_kind) is True for r in rows))
            bare = [r["wall_ns"] / r["repeats"] for r in rows if not r["actual_stamped"]]
            stamped = [r["wall_ns"] / r["repeats"] for r in rows if r["actual_stamped"]]
            paired_trials = [mean(r["wall_ns"] / r["repeats"] for r in rows if r["visit"] == visit and r["trial"] == trial and not r["actual_stamped"]) for visit in sorted({r["visit"] for r in rows}) for trial in range(8)]
            processes.append(dict(job=execution["job"]["id"], bytes=execution["job"]["bytes"], mode=mode, mean=mean(bare), cv=statistics.pstdev(paired_trials) / mean(paired_trials), observer=mean(stamped) / mean(bare) - 1))
    for group in summary["sizes"]:
        for mode in ("pageable", "registered"):
            rows = [p for p in processes if p["bytes"] == group["bytes"] and p["mode"] == mode]
            means = [p["mean"] for p in rows]
            expected = group["modes"][mode]
            for index, value in enumerate(means):
                check(f"diagnostic_mean:{group['bytes']}:{mode}:{index}", value, expected["means_ns"][index])
            for field, value in (("between_process_sample_cv", cv(means)), ("maximum_minimum_ratio", max(means) / min(means)), ("ten_percent_same_input_feasible", max(means) / min(means) <= 1.1 / .9), ("trial_cv_passes", sum(p["cv"] <= .1 for p in rows)), ("observer_passes", sum(abs(p["observer"]) <= .05 for p in rows))):
                check(f"diagnostic_gate:{group['bytes']}:{mode}:{field}", value, expected[field])
        for pair in group["within_process_pairs"]:
            values = {p["mode"]: p["mean"] for p in processes if p["job"] == pair["job"]}
            check("paired_memory_ratio:" + pair["job"], values["registered"] / values["pageable"], pair["registered_over_pageable"])
    return dict(passed=all(c["passed"] for c in checks), checks=checks, sources=sources, independent_statistics=True, parameter_fitting=False, native_launches=0, current_source_paths_not_required_unchanged=True)


def h(value):
    return html.escape(str(value))


def percent(value, digits=2):
    return "未知" if value is None else f"{100 * value:.{digits}f}%"


def ms(value):
    return "未知" if value is None else f"{value / 1e6:.4f}"


def us(value):
    return "未知" if value is None else f"{value / 1e3:.3f}"


def status(passed):
    return '<span class="pass">通过</span>' if passed else '<span class="fail">未通过</span>'


def link(path, label, report_dir):
    path = Path(path).resolve()
    relative = os.path.relpath(path, Path(report_dir).resolve()).replace("\\", "/")
    return f'<a href="{h(quote(relative, safe="/:"))}">{h(label)}</a>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h(item)}</th>' for item in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{item}</td>' for item in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def build(campaign=DEFAULT_BUILD / "campaign-v1", diagnostic=DEFAULT_BUILD / "paired-diagnostic-v1", out=DEFAULT_REPORT):
    campaign, diagnostic, out = Path(campaign).resolve(), Path(diagnostic).resolve(), Path(out).resolve()
    summary_path = campaign / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("最终 summary.json 尚不存在；报告不生成占位结果：" + str(summary_path))
    summary, manifest = read(summary_path), read(campaign / "manifest.json")
    if summary.get("schema") != "dac-accuracy-repair-result-1":
        raise ValueError("unsupported final campaign result schema")
    diagnostic_summary = read(diagnostic / "summary.json")
    independent = audit(campaign, summary)
    independent["paired_diagnostic_audit"] = audit_diagnostic(diagnostic, diagnostic_summary)
    independent["passed"] = independent["passed"] and independent["paired_diagnostic_audit"]["passed"]
    independent["checks"].extend(independent["paired_diagnostic_audit"]["checks"])
    independent["sources"].extend(independent["paired_diagnostic_audit"]["sources"])
    independent["check_count"] = len(independent["checks"])
    independent["failed_checks"] = [check for check in independent["checks"] if not check["passed"]]
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "independent-audit.json", independent)
    transfer_processes, transfer_sizes, dense_programs, dense_processes, group_rows = [], [], [], [], []
    for mode, data in summary["D2H"].items():
        for row in data["processes"]:
            transfer_processes.append(dict(memory_mode=mode, bytes=row["bytes"], observed_ns=row["mean_ns"], predicted_ns=row["predicted_ns"], process_mean_ape=row["ape"], trial_cv=row["cv"], observer_effect=row["observer_effect"], raw_path=row["raw"]["path"], raw_sha256=row["raw"]["sha256"]))
        for row in data["sizes"]:
            transfer_sizes.append(dict(memory_mode=mode, **row))
    for program in summary["programs"]:
        for field, data in program["modes"].items():
            dense_programs.append(dict(program=program["program_key"], layers=program["layers"], strategy=program["strategy"], provider=field, measured_mean_ns=program["measured_mean_ns"], predicted_ns=data["predicted_ns"], maximum_process_ape=data["maximum_process_ape"], mean_process_ape=data["mean_process_ape"], trial_p90_ape=data["trial_p90_ape"], preparation_match=program["preparation_match"], fully_qualified=data["fully_qualified"], assumptions=json.dumps(data["assumption"], ensure_ascii=False)))
        for kind, rows in (("primary", program["primary"]), ("observer", program["observers"])):
            for index, row in enumerate(rows):
                dense_processes.append(dict(program=program["program_key"], role=kind, process_index=index, mean_ns=row["mean_ns"], trial_cv=row["cv"], observer_effect=row.get("observer_effect"), raw_path=row["raw"]["path"], raw_sha256=row["raw"]["sha256"]))
    for group in summary["groups"]:
        for field, row in group.get("modes", {}).items():
            group_rows.append(dict(layers=group["layers"], provider=field, **row))
    for filename, rows in (("transfer-processes.csv", transfer_processes), ("transfer-sizes.csv", transfer_sizes), ("dense-programs.csv", dense_programs), ("dense-processes.csv", dense_processes), ("strategy-selection.csv", group_rows)):
        csv_new(out / filename, rows)
    sources = [receipt(summary_path), receipt(campaign / "manifest.json"), receipt(campaign / "frozen-barrier.json"), receipt(diagnostic / "summary.json"), receipt(Path(__file__))]
    write_new(out / "report-sources.json", dict(schema="dac-accuracy-repair-report-sources-1", sources=sources, old_evidence_modified=False, native_launches=0, heldout_refitting=False))

    counts = summary["counts"]
    sections = []
    sections.append('<h1>D2H 与完整 Dense 策略：独立准确性修正报告</h1><p class="meta">来自已完成的新采集；所有失败与原阈值保留。报告生成程序不测量、不重新拟合。</p>')
    sections.append('<div class="notice">这是一套新的执行环境验收：进程可用 CPU 位图为 <code>' + h(hex(manifest["process_affinity_mask"])) + '</code>，主线程仍固定 <code>0x4</code>，GGML 计算线程数仍为 1。此前六进程内存诊断和旧实验限制整个进程在 <code>0x4</code>。两者同时改变了环境，不能据此认定“放开辅助线程”是改善或恶化的唯一原因，也不能追认旧实验通过。</div>')
    sections.append('<h2>1. 先看结果</h2>')
    sections.append(table(["检查", "实际结果"], [
        ["新进程完成", h(f"{counts['completed']}/{counts['declared']}；失败 {counts['failed']}")],
        ["另行保留的旧 mask 配对诊断", h(f"{diagnostic_summary['completed']}/{diagnostic_summary['expected_processes']}；失败 {diagnostic_summary['failed']}")],
        ["完整 Dense：准备过程匹配", h(f"{counts['preparation_matched_programs']}/{len(summary['programs'])} 程序")],
        ["完整 Dense：主 provider 全部数值与稳定性门", h(f"{counts['hot_qualified_programs']}/{len(summary['programs'])} 程序")],
        ["完整 Dense：D2H 替换次 provider", h(f"{counts['transfer_secondary_qualified_programs']}/{len(summary['programs'])} 程序（仍有迁移假设）")],
        ["独立原始证据复算", status(independent["passed"]) + h(f"；{independent['check_count']} 条检查，{len(independent['failed_checks'])} 条不一致")],
    ]))
    transfer_overview = []
    for mode in ("pageable", "registered"):
        data = summary["D2H"][mode]
        transfer_overview.append([h(MODE_LABELS[mode]), percent(data["maximum_size_mean_ape"]), percent(data["maximum_process_ape"]), percent(data["trial_p90_ape"]), h(f"{data['cv_passes']}/{data['process_count']}"), h(f"{data['observer_passes']}/{data['process_count']}"), status(data["qualified_original_size_mean_gate"]), status(data["qualified_strict_process_mean_gate"])])
    sections.append(table(["D2H 宿主内存", "最大尺寸均值误差", "最大进程均值误差", "试次 P90 误差", "CV 通过", "观察器通过", "原尺寸均值门", "严格进程门"], transfer_overview))
    sections.append('<p>“原尺寸均值门”要求八个留出尺寸的三进程平均误差均不超过 10%，且 24 个进程各自 CV≤10%、观察器差≤5%。“严格进程门”另外要求每个独立进程的均值误差≤10%，并要求所有 bare block 的 P90 误差≤20%。两种口径独立保留，不通过平均数掩盖坏进程。</p>')
    sections.append('<p>参数冻结文件中的 <code>qualified=False</code> 保持不变；本报告的通过/未通过来自后续独立留出验收，而非翻转冻结标志。运行时启用新传输价格需要另附与严格验收和契约绑定的资格收据。本报告不会自动升级默认成本表，也不把 Dense 次结果的数值过门当作单次 COPY 的通用资格。</p>')

    sections.append('<h2>2. 测量什么，哪些量进入预测</h2>')
    sections.append(table(["采集阶段", "预先声明进程数"], [[h(split), h(manifest["protocol"]["counts"][split])] for split in SPLITS]))
    sections.append('<ul><li>传输：8 个训练尺寸和 8 个互不重叠的留出尺寸，每个内存类别、每个尺寸使用 3 个新进程。每个进程 8 轮 ABBA/BAAB，每块连续完成 256 次 COPY；先做一遍未计时的 COPY prime。</li><li>完整策略：H512/P64 的一层、两层，分别使用 S0（全部 CPU）、S1（全部 GPU）、S2（GPU 投影/FFN＋CPU Attention）。各程序 4 个 primary 进程和 4 个相邻 observer 进程。</li><li>primary 与 observer 的每个 A/B 块都使用同样的 reset→一次完整 prime→计时 8 次固定输入重放。修复的是可核对的准备过程，未模拟操作系统的任务调度。</li><li>动作参数来自独立 hot-action 进程；传输模型的参数和候选选择只读训练尺寸。参数、预测和源码哈希先冻结，之后才采集留出传输与完整目标。</li><li>构图、分配、reset、prime、状态查询、数值验证、注册/解注册均不计入 steady target。它们并非系统中没有成本；本报告只验收所声明的计时边界。</li></ul>')
    sections.append('<h3>可解释的同一类传输模型</h3><div class="formula">g(B; m, c) = a + b₁·min(B,k) + b₂·max(0,B−k)</div><p>B 是字节数，m 是已验证的 host 内存类别，c 是 API、完成语义、准备过程及硬件/运行时的契约；a、b₁、b₂ 非负。无断点时退化为 a+bB。profiling 用训练尺寸的留一尺寸误差选择原有仿射/单折点族；modeling 只计算冻结后的公式并检查契约，不拟合目标耗时。pageable 与 registered 是同一公式的不同参数契约，不能互相静默替换。</p>')
    parameter_rows = []
    for mode in ("pageable", "registered"):
        profile = read(campaign / f"transfer-{mode}-frozen.json")
        parameters = profile["parameters"]
        parameter_rows.append([h(MODE_LABELS[mode]), us(parameters["fixed_ns"]), h(f"{parameters['small_ns_per_byte']:.6g}"), h(f"{parameters['large_ns_per_byte']:.6g}"), h("无断点" if parameters["breakpoint_bytes"] is None else str(parameters["breakpoint_bytes"])), h(f"{profile['minimum_bytes']}–{profile['maximum_bytes']} B")])
    sections.append(table(["内存类别", "固定项 µs", "小尺寸 ns/B", "大尺寸 ns/B", "断点 B", "训练包络"], parameter_rows))
    sections.append('<p class="notice">这里的 g 预测 GGML COPY API 加 GPU/CPU 完成等待的 inclusive host response。API 内可能有 staging、内部等待与同步。分项时间只作诊断，不能与 inclusive 价格再次相加，也没有把拟合斜率认定为物理 PCIe 带宽。</p>')

    sections.append('<h2>3. D2H：逐个留出尺寸</h2>')
    rows = []
    for mode in ("pageable", "registered"):
        for row in summary["D2H"][mode]["sizes"]:
            rows.append([h(MODE_LABELS[mode]), h(row["bytes"]), us(row["observed_ns"]), us(row["predicted_ns"]), percent(row["ape"]), percent(row["maximum_process_ape"]), percent(row["between_process_cv"]), status(row["same_input_ten_percent_feasible"])])
    sections.append(table(["内存", "字节数", "实测均值 µs", "预测 µs", "尺寸均值误差", "最差进程误差", "跨进程 CV", "三进程共用±10%预测可行"], rows))
    sections.append('<p>“共用预测可行”仅检查三个实测均值的 ±10% 区间是否存在交集；不代表当前预测已落在交集内。跨进程 CV 与进程内试次 CV 的分母和含义不同。</p>')

    sections.append('<h2>4. 完整 Dense：逐个程序验收</h2>')
    rows = []
    for program in summary["programs"]:
        primary_cv = max((r["cv"] for r in program["primary"]), default=None)
        obs_effect = max((abs(r["observer_effect"]) for r in program["observers"]), default=None)
        for field in PROVIDER_LABELS:
            data = program["modes"][field]
            rows.append([h(f"{program['layers']}L S{program['strategy']}"), h("主 hot" if field == "hot_ns" else "次 D2H 替换"), ms(program["measured_mean_ns"]), ms(data["predicted_ns"]), percent(data["maximum_process_ape"]), percent(data["trial_p90_ape"]), percent(primary_cv), percent(obs_effect), status(program["preparation_match"]), status(data["fully_qualified"])])
    sections.append(table(["程序", "价格来源", "实测 ms", "预测 ms", "最差进程误差", "试次 P90", "最差进程内 CV", "最大观察器差", "prep 一致", "全门通过"], rows))
    sections.append('<p class="notice">主结果预先选定为独立 hot 动作表。D2H 替换只是预先声明的次结果：把 256 次连续传输的单次平均用于完整策略的一次 COPY，另外假定独立传输的内存类别与准备条件可迁移到 Dense arena。它没有验证 Dense arena 注册、前驱上下文或单次采样等价性；即使数值过门，也不能消除这些假设。本报告不在看完每个目标之后挑更准确的 provider。</p>')
    sections.append('<h3>策略选择结果</h3>')
    rows = []
    for group in group_rows:
        rows.append([h(f"{group['layers']} 层"), h("主 hot" if group["provider"] == "hot_ns" else "次 D2H 替换"), h(f"S{group['selected']}"), h(f"S{group['observed_mean_winner']}"), percent(group["regret"]), status(group["all_candidates_qualified"])])
    sections.append(table(["工作负载", "provider", "预测选择", "实测均值赢家", "point regret", "全部三候选质量合格"], rows))
    sections.append('<p>point regret=(所选策略实测均值/实测最佳均值)−1。本轮没有计算同时后悔值置信上界，因此不将 point regret 或两组新结果写成原六组完整策略选择验收已通过。</p>')

    sections.append('<h2>5. 最初六进程配对诊断：独立保留</h2>')
    sections.append('<p>同一个拥有的 destination buffer 按 ABBA/BAAB 顺序切换 pageable/registered。两个尺寸各 3 个新进程；这是旧进程 mask0x4 下的内存注册干预诊断，没有拟合参数，也不替代完整 campaign 的留出结果。</p>')
    rows = []
    for size in diagnostic_summary["sizes"]:
        for mode in ("pageable", "registered"):
            data = size["modes"][mode]
            rows.append([h(size["bytes"]), h(MODE_LABELS[mode]), h(data["completed_processes"]), h(', '.join(us(value) for value in data["means_ns"])), percent(data["between_process_sample_cv"]), h(f"{data['trial_cv_passes']}/{data['completed_processes']}"), h(f"{data['observer_passes']}/{data['completed_processes']}"), status(data["ten_percent_same_input_feasible"] is True)])
    sections.append(table(["字节数", "内存", "进程", "三个进程均值 µs", "跨进程 CV", "试次 CV 门", "观察器门", "共用±10%预测可行"], rows))
    paired = []
    for size in diagnostic_summary["sizes"]:
        for pair in size["within_process_pairs"]:
            paired.append([h(size["bytes"]), h(pair["job"]), h(f"{pair['registered_over_pageable']:.4f}×")])
    sections.append(table(["字节数", "配对进程", "registered/pageable 均值比"], paired))
    sections.append('<p>比值小于 1 表示该配对进程中 registered 更快。注册与解注册在计时外执行，并检查真实 CUDA pointer attributes 和恢复；这只说明所测注册干预下的差异，不唯一识别内部 staging、时钟或操作系统等待原因。</p>')
    sections.append('<p>这份已保留的六进程诊断先对每个 visit/trial 的两个 bare 块取均值，再以 16 个均值的总体标准差计算进程内 CV；主 campaign 的传输 CV 则以 16 个 bare 块的样本标准差计算。报告按各自预设口径重算，不将两组 CV 合格率互换。</p>')

    sections.append('<h2>6. 能说明什么，边界在哪里</h2><ul><li>新数据回答的是新执行环境、固定输入 warm serial 重放的准确性。旧环境、原 CPU/GPU 价格和历史完整验收均独立保留，不会被新数据覆盖或追认。</li><li>可验证主张只落在本次已采集的尺寸、memory/API/preparation 契约及六个 Dense 程序。完整任务的非线性状态、动态后台负载、多轮生成、真实 MoE 缓存/预取和物理分配器行为没有因此被验证。</li><li>状态采样发生在目标 timer 前，CPU/GPU 利用率和 GPU 时钟未知时保留未知；这些状态与完整目标耗时不输入拟合，也不把包含本任务负载的未来利用率反向喂给预测。</li><li>所有训练阶段早于留出阶段，尺寸/候选顺序轮换不能单独排除跨阶段温度、频率或系统负载漂移。本轮不把前后均值差归因到唯一物理机制。</li><li>若仍有大误差，分别看 raw 进程均值、试次 CV、observer effect、prep 匹配和 transfer-secondary 假设；未通过项原样保留，不以调均值或修改阈值换取通过。</li></ul>')
    sections.append('<h2>7. 原始证据与复算</h2>')
    items = [(campaign / "summary.json", "完整 campaign summary"), (campaign / "manifest.json", "采集前冻结的矩阵、进程 mask 与源码"), (campaign / "frozen-barrier.json", "参数/预测冻结屏障"), (diagnostic / "summary.json", "六进程配对诊断"), (out / "independent-audit.json", "独立 raw 复算与哈希/冻结检查"), (out / "transfer-processes.csv", "传输逐进程 CSV"), (out / "transfer-sizes.csv", "传输逐尺寸 CSV"), (out / "dense-programs.csv", "Dense 逐程序/provider CSV"), (out / "dense-processes.csv", "Dense 逐进程 CSV"), (out / "strategy-selection.csv", "策略选择 CSV"), (out / "report-sources.json", "报告数据来源与 SHA256")]
    sections.append('<ul>' + ''.join('<li>' + link(path, label, out) + '</li>' for path, label in items) + '</ul>')
    if not independent["passed"]:
        sections.append('<div class="notice fail">独立复算发现不一致；以上 summary 数字不能视为已审计通过。详见 independent-audit.json 的 failed_checks。</div>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>D2H 与完整 Dense 准确性修正</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#20314b;line-height:1.65;background:#fff}h1{font-size:30px;color:#203864}h2{margin-top:38px;border-bottom:2px solid #2f6f9f;padding-bottom:8px}h3{margin-top:24px}.meta{color:#58657a}.notice{background:#edf3f8;border-left:4px solid #2f6f9f;padding:15px 20px;margin:20px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{border:1px solid #d7dfe7;padding:9px 11px;text-align:left;vertical-align:top}th{background:#2f6f9f;color:white}tr:nth-child(even){background:#f4f7fa}.pass{color:#236547;font-weight:700}.fail{color:#b62828;font-weight:700}.formula{font-family:Cambria,serif;font-size:24px;background:#f4f7fa;padding:15px;text-align:center}code{background:#f1f3f6;padding:2px 5px}a{color:#1e6194}li{margin:7px 0}</style><body>' + ''.join(sections) + '</body></html>\n'
    with (out / "REPORT_zh.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    return dict(report=str(out / "REPORT_zh.html"), audit_passed=independent["passed"], audit_checks=independent["check_count"], counts=counts, new_native_processes_launched=0, fitting_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_BUILD / "campaign-v1")
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_BUILD / "paired-diagnostic-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(build(args.campaign, args.diagnostic, args.out), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
