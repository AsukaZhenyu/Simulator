"""Independently recompute follow-up statistics from retained native samples.

This audit does not import the collector, fitter or response model. It never
launches a benchmark, changes a frozen parameter or drops a failed process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from fractions import Fraction


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-8), name)


def response(model, size):
    split = model["breakpoint_bytes"]
    short = size if split is None else min(size, split)
    long = 0 if split is None else max(0, size - split)
    return model["fixed_ns"] + short * model["small_ns_per_byte"] + long * model["large_ns_per_byte"]


def audit_task1(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    frozen = read(directory / "transfer-model-frozen.json")
    barrier = read(directory / "frozen-barrier.json")
    summary = read(directory / "summary.json")
    require(sha(directory / "transfer-model-frozen.json") == barrier["frozen_sha256"], "Frozen model changed")
    require(barrier["heldout_read"] is False, "Calibration read heldout data")
    require(set(manifest["protocol"]["calibration_sizes"]).isdisjoint(manifest["protocol"]["heldout_sizes"]),
            "Calibration and heldout byte sizes overlap")
    frozen_time = datetime.fromisoformat(frozen["frozen_at_utc"])
    started_time = datetime.fromisoformat(read(directory / "transfer-heldout-started.json")["started_at_utc"])
    require(frozen_time <= started_time, "Heldout started before parameter freeze")
    sources = [dict(path=str(p), sha256=sha(p)) for p in
               (directory / "manifest.json", directory / "transfer-model-frozen.json", directory / "summary.json")]
    job_inventory = {row["id"]: row for row in manifest["jobs"]}
    samples, observer, failures, executions = [], [], [], []
    for split in ("calibration", "transfer-heldout", "observer"):
        expected = [row for row in manifest["jobs"] if row["split"] == split]
        file = directory / (split + "-executions.json")
        rows = read(file) if file.exists() else []
        require(len(rows) == len(expected), "Missing process in " + split)
        require({row["job"]["id"] for row in rows} == {row["id"] for row in expected}, "Job coverage differs")
        for row in rows:
            executions.append(row)
            job = row["job"]
            require(job == job_inventory[job["id"]], "Changed requested job")
            if row["status"] != "completed":
                failures.append(job["id"])
                continue
            require(row["returncode"] == 0 and row["numerical"]["passed"], "Completed process failed numerical contract")
            artifact = row["native"]
            require(sha(artifact["path"]) == artifact["sha256"], "Changed raw native output")
            native = read(artifact["path"])
            require(native["request"] == job["request"] and native["process_id"] == row["process_id"], "Native request/PID differs")
            frequency = native["clock"]["frequency_hz"]
            for trial in native["trials"]:
                require(abs((trial["end_tick"] - trial["begin_tick"]) * 1e9 / frequency - trial["wall_ns"]) <= 1,
                        "Native QPC duration differs")
            if split == "observer":
                a, b = [], []
                for index in range(8):
                    group = [trial for trial in native["trials"] if trial["trial"] == index]
                    require(len(group) == 4 and Counter(x["assigned_label"] for x in group) == {"A": 2, "B": 2},
                            "Unbalanced observer contrast")
                    a.append(statistics.mean(x["wall_ns"] / x["repeats"] for x in group if x["assigned_label"] == "A"))
                    b.append(statistics.mean(x["wall_ns"] / x["repeats"] for x in group if x["assigned_label"] == "B"))
                effect = statistics.mean(b) / statistics.mean(a) - 1
                observer.append(dict(job=job["id"], effect=effect, passed=abs(effect) <= .05,
                                     sham=native["sham_bare_bare"], program=job["program_key"]))
            else:
                values = [trial["wall_ns"] / trial["repeats"] for trial in native["trials"]]
                mean = statistics.mean(values)
                samples.append(dict(job=job["id"], route=row["program"]["route"], split=split,
                                    size=row["program"]["bytes"], mean=mean,
                                    cv=statistics.stdev(values) / mean))
    require(summary["all_native_processes"] == len(executions), "Process denominator differs")
    calibration = {(row["route"], row["size"], row["job"]): row for row in samples if row["split"] == "calibration"}
    require(len(frozen["training_observations"]) == len(calibration), "Training observation denominator differs")
    expected_training_means = Counter((row["route"], row["size"], row["mean"]) for row in calibration.values())
    frozen_training_means = Counter((row["route"], row["bytes"], row["mean_ns"]) for row in frozen["training_observations"])
    require(expected_training_means == frozen_training_means, "Frozen parameters refer to changed training observations")
    for route, result in summary["transfer"].items():
        model = frozen["models"][route]["selected"]
        require(result["model"] == model, "Result changed frozen parameters")
        held = [x for x in samples if x["route"] == route and x["split"] == "transfer-heldout"]
        grouped = defaultdict(list)
        for sample in held:
            grouped[sample["size"]].append(sample["mean"])
        errors = []
        for reported in result["heldout_size_means"]:
            mean = statistics.mean(grouped[reported["bytes"]])
            prediction = response(model, reported["bytes"])
            error = abs(prediction - mean) / mean
            close(mean, reported["observed_ns"], "Heldout size mean differs")
            close(prediction, reported["predicted_ns"], "Prediction differs")
            close(error, reported["ape"], "Heldout APE differs")
            errors.append(error)
        close(max(errors), result["maximum_heldout_size_mean_ape"], "Maximum heldout APE differs")
        qualified = max(errors) <= .10 and all(x["cv"] <= .10 for x in held)
        require(result["qualified"] == qualified, "Transfer qualification differs")
    by_job = {row["job"]: row for row in summary["observer"]["rows"]}
    for row in observer:
        close(row["effect"], by_job[row["job"]]["relative_difference"], "Observer effect differs")
        require(row["passed"] == by_job[row["job"]]["passed"], "Observer gate differs")
    real = [row for row in observer if not row["sham"]]
    sham = [row for row in observer if row["sham"]]
    require(summary["observer"]["passed_processes"] == sum(row["passed"] for row in real), "Observer count differs")
    require(summary["observer"]["sham_passes"] == sum(row["passed"] for row in sham), "Sham count differs")
    return dict(schema="dac-followup-independent-audit-1", passed=True, directory=str(directory),
                raw_native_processes_checked=len(executions), failed_processes=failures,
                transfer_process_means_recomputed=len(samples), observer_contrasts_recomputed=len(observer),
                source_receipts=sources, frozen_before_heldout=True,
                old_results_requalified=False, fitter_and_model_not_imported=True)


def audit_multilayer(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    barrier = read(directory / "frozen-barrier.json")
    summary = read(directory / "summary.json")
    predictions = read(directory / "predictions.json")
    executions = read(directory / "heldout-executions.json")
    require(sha(directory / "profile-frozen.json") == barrier["frozen_sha256"], "Multilayer profile changed")
    require(sha(directory / "predictions.json") == barrier["predictions_sha256"], "Multilayer predictions changed")
    old_profile = manifest["frozen_independent_profile"]
    require(sha(old_profile["path"]) == old_profile["sha256"] == barrier["frozen_sha256"], "Not the pre-existing independent profile")
    require(len(predictions) == 12 and len(executions) == len(manifest["jobs"]) == 48, "Multilayer denominator differs")
    prices = {}
    for row in predictions:
        total = sum(map(Fraction, row["action_ns"]), Fraction())
        require(total == Fraction(row["predicted_ns"]), "Multilayer action prices do not sum")
        require(row["exact_coordinate_coverage"] and row["target_time_read"] is False, "Unexpected price substitution")
        prices[row["program_key"]] = float(total)
    processes, failures = [], []
    reported = {row["job"]: row for row in summary["processes"]}
    for execution in executions:
        job = execution["job"]
        require(datetime.fromisoformat(execution["started_at_utc"]) >= datetime.fromisoformat(barrier["frozen_at_utc"]),
                "Multilayer target predates prediction freeze")
        if execution["status"] != "completed":
            failures.append(job["id"])
            continue
        artifact = execution["native"]
        require(sha(artifact["path"]) == artifact["sha256"], "Multilayer raw output changed")
        native = read(artifact["path"])
        require(native["process_id"] == execution["process_id"] and native["request"] == job["request"], "Multilayer PID/request differs")
        values = [row["wall_ns"] / row["repeats"] for row in native["trials"]]
        mean, price = statistics.mean(values), prices[job["program_key"]]
        cv, error = statistics.stdev(values) / mean, abs(price - mean) / mean
        other = reported[job["id"]]
        for name, value in (("real_ns", mean), ("cv", cv), ("ape", error), ("predicted_ns", price)):
            close(value, other[name], "Multilayer process statistic differs: " + name)
        require(execution["numerical"]["passed"] and other["numerical_pass"], "Multilayer numerical contract failed")
        processes.append(dict(key=job["program_key"], mean=mean, error=error, cv=cv))
    require(len(processes) == summary["completed_processes"], "Multilayer completed denominator differs")
    selected = 0
    for group in summary["groups"]:
        for candidate in group["candidates"]:
            key = group["workload"] + f"-s{candidate['strategy']}"
            rows = [row for row in processes if row["key"] == key]
            if len(rows) == 4:
                close(statistics.mean(row["mean"] for row in rows), candidate["measured_mean_ns"], "Multilayer aggregate mean differs")
        if group["complete"]:
            actual = min(group["candidates"], key=lambda row: (row["measured_mean_ns"], row["strategy"]))
            chosen = min(group["candidates"], key=lambda row: (row["predicted_ns"], row["strategy"]))
            require(actual["strategy"] == group["real_winner"] and chosen["strategy"] == group["predicted_winner"], "Multilayer winner differs")
            close(chosen["measured_mean_ns"] / actual["measured_mean_ns"] - 1, group["point_regret"], "Multilayer regret differs")
            selected += actual["strategy"] == chosen["strategy"]
    require(selected == summary["groups_selected_observed_winner"], "Multilayer selection count differs")
    return dict(schema="dac-followup-multilayer-independent-audit-1", passed=True, directory=str(directory),
                raw_native_processes_checked=len(executions), failed_processes=failures,
                independently_recomputed_process_statistics=len(processes), frozen_before_every_target=True,
                pre_existing_profile_unchanged=True, observed_winner_groups=selected,
                cost_sum_checked=True, semantic_cost_key_validation_scope="Original exact-key provider and focused bridge tests; not independently reimplemented here",
                original_E2_quality_not_requalified=True, source_summary_sha256=sha(directory / "summary.json"))


def audit_affinity(directory):
    directory = Path(directory).resolve()
    manifest, summary = read(directory / "manifest.json"), read(directory / "summary.json")
    executions = read(directory / "calibration-executions.json")
    require(len(executions) == len(manifest["jobs"]) == 32, "Affinity intervention denominator differs")
    means = defaultdict(list)
    for execution in executions:
        require(execution["status"] == "completed" and execution["numerical"]["passed"], "Affinity process incomplete")
        native_path = execution["native"]["path"]
        require(sha(native_path) == execution["native"]["sha256"], "Affinity raw data changed")
        native = read(native_path)
        require(native["process_id"] == execution["process_id"] and native["request"] == execution["job"]["request"], "Affinity PID/request differs")
        mask = execution["job"]["process_mask"]
        require(execution["affinity"]["requested_mask"] == execution["affinity"]["observed_mask"] == mask,
                "Parent process-affinity readback differs")
        affinity = native["main_thread_affinity"]
        require(affinity["process_mask"] == mask and affinity["requested_mask"] == affinity["observed_mask"] == affinity["final_mask"] == 4,
                "Main-thread affinity changed with process budget")
        values = [row["wall_ns"] / row["repeats"] for row in native["trials"]]
        key = (execution["program"]["route"], execution["program"]["bytes"], str(mask))
        means[key].append(statistics.mean(values))
    for condition in summary["conditions"]:
        route, size = condition["route"], condition["bytes"]
        for mask in ("4", "20"):
            values = means[route, size, mask]
            require(len(values) == 4, "Missing independent affinity round")
            close(statistics.mean(values), condition["masks"][mask]["mean_ns"], "Affinity aggregate mean differs")
            close(statistics.stdev(values) / statistics.mean(values), condition["masks"][mask]["between_process_cv"],
                  "Affinity between-process CV differs")
        effect = statistics.mean(means[route, size, "20"]) / statistics.mean(means[route, size, "4"]) - 1
        close(effect, condition["aggregate_relative_effect"], "Affinity intervention effect differs")
    require(summary["specific_helper_thread_mechanism_identified"] is False, "Unexpected driver-specific causal claim")
    return dict(schema="dac-followup-affinity-independent-audit-1", passed=True,
                raw_processes_checked=32, intervention_conditions_recomputed=4,
                native_and_parent_affinity_readbacks_verified=True, fixed_main_thread_mask=4,
                failed_processes=[], source_summary_sha256=sha(directory / "summary.json"),
                specific_driver_thread_causation_claimed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--task1", type=Path)
    choice.add_argument("--multilayer", type=Path)
    choice.add_argument("--affinity", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = (audit_task1(args.task1) if args.task1 else audit_multilayer(args.multilayer)
              if args.multilayer else audit_affinity(args.affinity))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
