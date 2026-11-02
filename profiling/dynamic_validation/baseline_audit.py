"""Audit whether the measured zero-idle suffix supports one fixed price.

This is a calibration support gate, not a fitter or a causal diagnosis. Every
trial remains in the output. A heldout audit is explicitly diagnostic and can
never supply parameters or an independent-validation claim after repair.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an explicit integer")
    return value


def audit_native(native, *, tail_first_ordinal, split_trial_count, relative_limit):
    """Compare chronological halves of the *same* zero-idle copy suffix.

    The suffix is declared by the caller; it is not assumed to be physically
    steady. The relative change is (late mean / early mean - 1). Population CV
    describes the trial means and is not substituted for prediction accuracy.
    """
    first = _integer(tail_first_ordinal, "suffix first ordinal", 1)
    split = _integer(split_trial_count, "trial split", 1)
    require(type(relative_limit) in (int, float) and math.isfinite(relative_limit)
            and 0 < relative_limit < 1, "explicit relative support limit required")
    request = native["request"]
    sampling = request["sampling"]
    trials = _integer(sampling["trials"], "trials", 2)
    repeats = _integer(sampling["repeats"], "repeats", 2)
    require(first < repeats and split < trials, "suffix/split outside declared inventory")
    grid = request["idle_ms"]
    require(len(set(grid)) == len(grid) and grid.count(0) == 1, "unique zero-idle control required")
    sequences = native["sequences"]
    expected = {(trial, idle) for trial in range(trials) for idle in grid}
    seen = set()
    zero = {}
    for sequence in sequences:
        trial = _integer(sequence["trial"], "trial index")
        idle = sequence["requested_idle_ms"]
        key = (trial, idle)
        require(key in expected and key not in seen, "missing/duplicate/unknown condition inventory")
        seen.add(key)
        copies = sequence["copies"]
        require(len(copies) == repeats and [c["ordinal"] for c in copies] == list(range(repeats)),
                "copy ordinal inventory differs")
        values = [_integer(c["wall_ns"], "copy response ns", 1) for c in copies]
        if idle == 0:
            zero[trial] = dict(trial=trial, first_copy_ns=values[0],
                               suffix_response_ns=values[first:],
                               suffix_mean_ns=statistics.mean(values[first:]),
                               actual_idle_ns=sequence["actual_idle_ns"])
    require(seen == expected, "incomplete declared condition inventory; no silent omission")
    rows = [zero[i] for i in range(trials)]
    early = statistics.mean(r["suffix_mean_ns"] for r in rows[:split])
    late = statistics.mean(r["suffix_mean_ns"] for r in rows[split:])
    values = [r["suffix_mean_ns"] for r in rows]
    mean = statistics.mean(values)
    drift = late / early - 1
    return dict(memory_mode=request["memory_mode"], tail_first_ordinal=first,
                suffix_is_proven_steady=False, declared_trials=trials, observed_trials=len(rows),
                split_trial_count=split, zero_idle_trials=rows,
                early_suffix_mean_ns=early, late_suffix_mean_ns=late,
                all_suffix_mean_ns=mean, late_over_early_minus_one=drift,
                trial_mean_cv_population=statistics.pstdev(values) / mean,
                fixed_baseline_support=abs(drift) <= relative_limit,
                relative_limit=relative_limit, samples_removed=0,
                physical_reason_identified=False, performance_parameters_fitted=False)


def inspect_campaign(campaign, *, split, role, tail_first_ordinal, split_trial_count, relative_limit):
    require(split in {"profile", "heldout"}, "explicit split required")
    require(role in {"calibration_support", "diagnostic_replay"}, "explicit audit role required")
    require(split == "profile" or role == "diagnostic_replay", "heldout cannot be calibration input")
    campaign = Path(campaign).resolve()
    manifest_path = campaign / "manifest.json"
    manifest = read(manifest_path)
    if role == "calibration_support":
        require(not (campaign / "heldout-started.json").exists()
                and not any((campaign / "jobs" / j["id"]).exists()
                            for j in manifest["jobs"] if j["split"] == "heldout"),
                "calibration audit after heldout acquisition rejected; use diagnostic_replay")
    execution_path = campaign / (split + "-executions.json")
    executions = read(execution_path)
    jobs = [j for j in manifest["jobs"] if j["split"] == split]
    require({e["job"]["id"] for e in executions} == {j["id"] for j in jobs}
            and len(executions) == len(jobs), "execution denominator differs from declared campaign")
    evidence = [receipt(manifest_path), receipt(execution_path)]
    records = []
    for execution in executions:
        row = dict(job_id=execution["job"]["id"], status=execution["status"], fixed_baseline_support=False)
        if execution["status"] == "completed":
            bound = execution["native"]
            actual = receipt(bound["path"])
            require(actual["sha256"] == bound["sha256"], "native measurement receipt changed")
            native = read(bound["path"])
            require(native["request"] == execution["job"]["request"], "native request differs from executed request")
            row.update(audit_native(native, tail_first_ordinal=tail_first_ordinal,
                                   split_trial_count=split_trial_count, relative_limit=relative_limit))
            row["native"] = actual
            evidence.append(actual)
        else:
            row["error"] = execution.get("error", "measurement did not complete")
        records.append(row)
    groups = []
    memories = sorted({j["memory_mode"] for j in jobs})
    for memory in memories:
        ids = {j["id"] for j in jobs if j["memory_mode"] == memory}
        rs = [r for r in records if r["job_id"] in ids]
        complete = [r for r in rs if r["status"] == "completed"]
        values = [r["all_suffix_mean_ns"] for r in complete]
        spread = max(values) / min(values) - 1 if values else None
        groups.append(dict(memory_mode=memory, declared_processes=len(rs), completed_processes=len(complete),
                           zero_idle_suffix_process_means_ns=values,
                           process_range_over_min_minus_one=spread,
                           within_process_support=all(r["fixed_baseline_support"] for r in rs),
                           fixed_baseline_support=len(complete) == len(rs) and bool(rs)
                           and all(r["fixed_baseline_support"] for r in rs)
                           and spread <= relative_limit))
    return dict(schema="zero-idle-baseline-support-audit-1", audited_at_utc=datetime.now(timezone.utc).isoformat(),
                campaign=str(campaign), split=split, role=role, declared_processes=len(jobs),
                completed_processes=sum(r["status"] == "completed" for r in records),
                records=records, groups=groups,
                supports_one_fixed_baseline=bool(groups) and all(g["fixed_baseline_support"] for g in groups),
                gate=dict(tail_first_ordinal=tail_first_ordinal, split_trial_count=split_trial_count,
                          relative_limit=relative_limit, process_spread_denominator="minimum process mean"),
                evidence=evidence, auditor=receipt(__file__), samples_removed=0,
                parameter_estimation_performed=False, target_total_used=False,
                independent_accuracy_claimed=False, physical_reason_identified=False,
                next_step="identify/condition observable state or retain unsupported baseline; never discard slow trials")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--split", choices=("profile", "heldout"), required=True)
    p.add_argument("--role", choices=("calibration_support", "diagnostic_replay"), required=True)
    p.add_argument("--tail-first-ordinal", type=int, required=True)
    p.add_argument("--split-trial-count", type=int, required=True)
    p.add_argument("--relative-limit", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = inspect_campaign(a.campaign, split=a.split, role=a.role,
                              tail_first_ordinal=a.tail_first_ordinal,
                              split_trial_count=a.split_trial_count, relative_limit=a.relative_limit)
    target = Path(a.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(report=str(target), supports_one_fixed_baseline=result["supports_one_fixed_baseline"],
                          completed=result["completed_processes"], declared=result["declared_processes"])))


if __name__ == "__main__":
    main()
