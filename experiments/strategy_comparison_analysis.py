"""Pure paired analysis for a predeclared, finite analytical strategy catalogue.

No latency is inferred for missing or failed runs.  Offline selection here is
an experiment statistic, not a strategy-search implementation or global optimum.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import re


REFERENCE = "e0-s0"
METRICS = (
    "completed_makespan_ns", "h2d_bytes", "d2h_bytes",
    "peak_vram_bytes", "peak_dram_bytes", "immutable_weight_reload_bytes",
    "immutable_weight_reload_count", "cpu_event_span_ns", "gpu_event_span_ns",
    "copy_event_span_ns", "cpu_gpu_event_overlap_ns", "copy_compute_event_overlap_ns",
)
EXECUTION_LABELS = {
    "e0": "All GPU", "e1": "All CPU", "e2": "FFN on CPU",
    "e3": "Attention on CPU", "e4": "Resident-first experts",
    "e5": "Current-cost experts",
}
STORAGE_LABELS = {
    "s0": "LRU / no prefetch", "s1": "LRU / prefetch",
    "s2": "History-frequency / no prefetch", "s3": "History-frequency / prefetch",
}


def _axes(kind):
    return [f"e{i}" for i in range(4 if kind == "dense" else 6)], [f"s{i}" for i in range(4)]


def _candidate_key(candidate):
    execution, storage = candidate.split("-")
    return int(execution[1:]), int(storage[1:])


def _complete(row):
    return row is not None and row.get("success") is True


def _time(row):
    return row["completed_makespan_ns"] if _complete(row) else None


def _snapshot(row, candidate):
    if row is None:
        return {"candidate_id": candidate, "present": False, "success": False,
                "status": "missing", "completed_makespan_ns": None}
    result = {"candidate_id": candidate, "present": True,
              "success": row.get("success") is True,
              "status": row.get("status"), "error_code": row.get("error_code")}
    for metric in METRICS:
        result[metric] = row.get(metric) if _complete(row) else None
    result["copy_bytes_by_direction_and_role"] = (
        row.get("copy_bytes_by_direction_and_role") if _complete(row) else None)
    return result


def _known_difference(left, right, key):
    """After minus before; absent measurements remain unknown, not zero."""
    a, b = left.get(key), right.get(key)
    if a is None or b is None:
        return None
    return b - a


def _copy_difference(before, after):
    a = before.get("copy_bytes_by_direction_and_role")
    b = after.get("copy_bytes_by_direction_and_role")
    if a is None or b is None:
        return None
    return {
        direction: {
            role: b.get(direction, {}).get(role, 0) - a.get(direction, {}).get(role, 0)
            for role in sorted(set(a.get(direction, {})) | set(b.get(direction, {})))
        }
        for direction in sorted(set(a) | set(b))
    }


def _pair(index, before_id, after_id, **identity):
    before, after = index.get(before_id), index.get(after_id)
    complete = _complete(before) and _complete(after)
    row = dict(identity, before_candidate=before_id, after_candidate=after_id,
               complete_pair=complete,
               before=_snapshot(before, before_id), after=_snapshot(after, after_id),
               latency_delta_ns=None, latency_saving_ns=None,
               latency_saving_fraction=None, before_over_after_ratio=None,
               metric_deltas={key: None for key in METRICS},
               copy_bytes_by_direction_and_role_delta=None)
    if complete:
        a, b = _time(before), _time(after)
        row.update(latency_delta_ns=b-a, latency_saving_ns=a-b,
                   latency_saving_fraction=(a-b)/a, before_over_after_ratio=a/b,
                   metric_deltas={key: _known_difference(before, after, key) for key in METRICS},
                   copy_bytes_by_direction_and_role_delta=_copy_difference(before, after))
    return row


def _one_shot(index, executions, storages, best_ids, best_time, **identity):
    execution_cells = [f"{e}-s0" for e in executions if _complete(index.get(f"{e}-s0"))]
    storage_cells = [f"e0-{s}" for s in storages if _complete(index.get(f"e0-{s}"))]

    def pick(cells):
        if not cells:
            return []
        minimum = min(_time(index[c]) for c in cells)
        return sorted([c for c in cells if _time(index[c]) == minimum], key=_candidate_key)

    e_ties, s_ties = pick(execution_cells), pick(storage_cells)
    composed = f"{e_ties[0].split('-')[0]}-{s_ties[0].split('-')[1]}" if e_ties and s_ties else None
    composed_row = index.get(composed) if composed else None
    status = ("complete" if _complete(composed_row) else
              "composed_candidate_missing" if composed and composed_row is None else
              "composed_candidate_failed" if composed else "selectors_unavailable")
    time = _time(composed_row)
    return dict(identity, status=status,
                execution_selection_ties_at_s0=e_ties,
                storage_selection_ties_at_e0=s_ties,
                deterministic_tie_rule="smallest execution number, then smallest storage number",
                composed_candidate=composed,
                composed=_snapshot(composed_row, composed) if composed else None,
                completed_makespan_ns=time,
                offline_catalogue_best_candidates=best_ids,
                offline_catalogue_best_makespan_ns=best_time,
                finite_catalogue_regret_fraction=(time-best_time)/best_time if time is not None and best_time is not None else None,
                composed_over_offline_best_ratio=time/best_time if time is not None and best_time is not None else None,
                execution_selector_complete_cells=len(execution_cells),
                execution_selector_expected_cells=len(executions),
                storage_selector_complete_cells=len(storage_cells),
                storage_selector_expected_cells=len(storages))


def analyze(records):
    """Return JSON-safe paired results without modifying records.

    Times must be exact positive integer nanoseconds for successful runs.  A
    failed run may carry diagnostic partial times in a separate field, but no
    completed time.  Duplicate scenario/candidate pairs are rejected.  Every
    scenario is assessed against its declared Dense 4x4 or MoE 6x4 catalogue.
    """
    groups = {}
    for row in records:
        scenario = row.get("scenario_id")
        kind = row.get("model_kind")
        candidate = row.get("candidate_id")
        if not isinstance(scenario, str) or not scenario or kind not in ("dense", "moe"):
            raise ValueError("each row requires a scenario_id and dense/moe model_kind")
        executions, storages = _axes(kind)
        catalogue = {f"{e}-{s}" for e in executions for s in storages}
        if candidate not in catalogue:
            raise ValueError(f"candidate {candidate!r} is outside the {kind} catalogue")
        if not isinstance(row.get("success"), bool):
            raise ValueError("success must be an explicit boolean")
        time = row.get("completed_makespan_ns")
        if _complete(row):
            if isinstance(time, bool) or not isinstance(time, int) or time <= 0:
                raise ValueError("successful rows require positive integer completed_makespan_ns")
        elif time is not None:
            raise ValueError("failed rows must not carry a completed_makespan_ns")
        if row.get("calibrated", False) is not False:
            raise ValueError("this analytical campaign does not accept calibrated=True rows")
        if scenario not in groups:
            groups[scenario] = {"model_kind": kind, "rows": {}, "metadata": {
                key: row.get(key) for key in ("prefill_tokens", "decode_steps", "route_pattern")}}
        group = groups[scenario]
        if group["model_kind"] != kind:
            raise ValueError("a scenario_id cannot mix model kinds")
        if any(row.get(key) != value for key, value in group["metadata"].items()):
            raise ValueError("scenario metadata differs between candidates")
        if candidate in group["rows"]:
            raise ValueError(f"duplicate scenario/candidate: {scenario}/{candidate}")
        group["rows"][candidate] = row

    summaries, comparisons, prefetch, cache, reversals, interactions, selections = [], [], [], [], [], [], []
    rank_missing = 0
    for scenario, group in sorted(groups.items()):
        kind, index = group["model_kind"], group["rows"]
        executions, storages = _axes(kind)
        candidates = [f"{e}-{s}" for e in executions for s in storages]
        identity = dict(scenario_id=scenario, model_kind=kind, **group["metadata"])
        complete = [c for c in candidates if _complete(index.get(c))]
        best_time = min((_time(index[c]) for c in complete), default=None)
        best_ids = [c for c in complete if _time(index[c]) == best_time]
        reference = index.get(REFERENCE)
        summary = dict(identity, expected_count=len(candidates), present_count=len(index),
                       complete_count=len(complete), failure_count=len(index)-len(complete),
                       missing_count=len(candidates)-len(index),
                       failed_status_counts=dict(sorted(Counter(str(r.get("status", "unspecified"))
                           for r in index.values() if not _complete(r)).items())),
                       best_candidate=best_ids[0] if best_ids else None,
                       best_candidates=best_ids, best_makespan_ns=best_time,
                       reference=_snapshot(reference, REFERENCE),
                       best_reference_speedup=_time(reference)/best_time
                           if _complete(reference) and best_time is not None else None)
        summaries.append(summary)
        for candidate in candidates:
            row = index.get(candidate)
            comparisons.append(dict(identity, **_snapshot(row, candidate),
                reference_candidate=REFERENCE,
                reference_speedup=_time(reference)/_time(row)
                    if _complete(reference) and _complete(row) else None))
        for execution in executions:
            for before, after, family in (("s0", "s1", "lru"), ("s2", "s3", "history_frequency")):
                prefetch.append(_pair(index, f"{execution}-{before}", f"{execution}-{after}",
                                      **identity, execution=execution, cache_family=family))
            for before, after, enabled in (("s0", "s2", False), ("s1", "s3", True)):
                cache.append(_pair(index, f"{execution}-{before}", f"{execution}-{after}",
                                   **identity, execution=execution, prefetch_enabled=enabled))
        for ea, eb in combinations(executions, 2):
            for sa, sb in combinations(storages, 2):
                ids = [f"{ea}-{sa}", f"{eb}-{sa}", f"{ea}-{sb}", f"{eb}-{sb}"]
                if not all(_complete(index.get(c)) for c in ids):
                    rank_missing += 1
                    continue
                times = [_time(index[c]) for c in ids]
                da, db = times[0]-times[1], times[2]-times[3]
                reversals.append(dict(identity, execution_a=ea, execution_b=eb,
                    storage_a=sa, storage_b=sb, candidates=ids, times_ns=times,
                    execution_a_minus_b_at_storage_a_ns=da,
                    execution_a_minus_b_at_storage_b_ns=db,
                    absolute_gap_at_storage_a_ns=abs(da),
                    absolute_gap_at_storage_b_ns=abs(db),
                    relative_gap_at_storage_a=abs(da)/min(times[0], times[1]),
                    relative_gap_at_storage_b=abs(db)/min(times[2], times[3]),
                    strict_reversal=da*db < 0,
                    tie_at_either_storage=da == 0 or db == 0))
        for execution in executions[1:]:
            for storage in storages[1:]:
                ids = [REFERENCE, f"{execution}-s0", f"e0-{storage}", f"{execution}-{storage}"]
                complete_four = all(_complete(index.get(c)) for c in ids)
                cell = dict(identity, execution=execution, storage=storage, candidates=ids,
                    cells=[_snapshot(index.get(c), c) for c in ids], complete_four_cells=complete_four,
                    joint_gain_over_better_single=None, nonadditive_time_saving_ns=None,
                    strict_execution_order_reversal=None)
                if complete_four:
                    t00, t10, t01, t11 = [_time(index[c]) for c in ids]
                    single = min(t10, t01)
                    cell.update(joint_gain_over_better_single=(single-t11)/single,
                        nonadditive_time_saving_ns=t10+t01-t00-t11,
                        strict_execution_order_reversal=(t00-t10)*(t01-t11) < 0)
                interactions.append(cell)
        selection = _one_shot(index, executions, storages, best_ids, best_time, **identity)
        selections.append(selection)
        summary["one_shot_selection"] = selection

    result = {
        "protocol": {
            "format_version": "finite-strategy-analysis-0.1", "calibrated": False,
            "reference_candidate": REFERENCE,
            "candidate_catalogue": {kind: [f"{e}-{s}" for e in _axes(kind)[0] for s in _axes(kind)[1]]
                                    for kind in ("dense", "moe")},
            "time_unit": "integer nanoseconds", "exact_ties": True,
            "paired_delta_sign": "after minus before; positive latency/traffic delta is an increase",
            "latency_saving_sign": "before minus after; positive is faster",
            "nonadditive_saving_formula": "T(e,s0)+T(e0,s)-T(e0,s0)-T(e,s)",
            "joint_gain_formula": "(min(T(e,s0),T(e0,s))-T(e,s))/min(T(e,s0),T(e0,s))",
            "rank_relative_gap_formula": "absolute latency gap / faster of the two execution times at that storage point",
            "one_shot_selection": "Independently select execution at s0 and storage at e0, then evaluate their composed cell; numeric tie-breaking.",
            "scope": "Offline results from a finite predeclared analytical catalogue; not measurements, a search algorithm, or a global optimum.",
            "failed_or_missing": "No completed latency, speedup, or zero substitution; four-cell tests require four completed cells.",
        },
        "totals": {
            "scenario_count": len(summaries), "present_count": sum(s["present_count"] for s in summaries),
            "expected_count": sum(s["expected_count"] for s in summaries),
            "complete_count": sum(s["complete_count"] for s in summaries),
            "failure_count": sum(s["failure_count"] for s in summaries),
            "missing_count": sum(s["missing_count"] for s in summaries),
            "complete_prefetch_pairs": sum(p["complete_pair"] for p in prefetch),
            "complete_cache_pairs": sum(p["complete_pair"] for p in cache),
            "complete_interactions": sum(p["complete_four_cells"] for p in interactions),
            "rank_four_cells_complete": len(reversals), "rank_four_cells_unavailable": rank_missing,
            "strict_execution_rank_reversals": sum(p["strict_reversal"] for p in reversals),
            "scenarios_with_strict_execution_rank_reversal": len({p["scenario_id"] for p in reversals if p["strict_reversal"]}),
        },
        "scenarios": summaries, "comparisons": comparisons,
        "prefetch_pairs": prefetch, "cache_pairs": cache,
        "execution_rank_reversals": reversals, "interactions": interactions,
        "one_shot_selections": selections,
    }
    return result


def plots(records, analysis, outputdir):
    """Write analytical latency PNG/SVG figures; gray FAIL/MISS cells are not zero.

    Return relative filenames so callers can create an artifact manifest.  Plot
    imports are delayed so pure analysis does not require a plotting runtime.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def save(figure, stem):
        for extension in ("png", "svg"):
            name = f"{stem}.{extension}"
            figure.savefig(outputdir/name, dpi=160, facecolor="white")
            artifacts.append(name)
        plt.close(figure)

    indexed = {(r["scenario_id"], r["candidate_id"]): r for r in records}
    scenario_ids = [s["scenario_id"] for s in analysis["scenarios"]]
    for summary in analysis["scenarios"]:
        scenario, kind = summary["scenario_id"], summary["model_kind"]
        executions, storages = _axes(kind)
        values = np.full((len(executions), len(storages)), np.nan)
        for i, execution in enumerate(executions):
            for j, storage in enumerate(storages):
                row = indexed.get((scenario, f"{execution}-{storage}"))
                if _complete(row):
                    values[i, j] = _time(row)/1e6
        figure, axis = plt.subplots(figsize=(10, 5.6 if kind == "dense" else 6.6), layout="constrained")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#dedede")
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(storages)), [f"{s}\n{STORAGE_LABELS[s]}" for s in storages], fontsize=9)
        axis.set_yticks(range(len(executions)), [f"{e}: {EXECUTION_LABELS[e]}" for e in executions])
        finite = values[np.isfinite(values)]
        mid = (finite.min()+finite.max())/2 if finite.size else None
        for i, execution in enumerate(executions):
            for j, storage in enumerate(storages):
                row = indexed.get((scenario, f"{execution}-{storage}"))
                value = values[i, j]
                label = f"{value:.4g}" if math.isfinite(value) else ("MISS" if row is None else "FAIL")
                axis.text(j, i, label, ha="center", va="center",
                          color="white" if math.isfinite(value) and value < mid else "black", fontsize=10)
        figure.colorbar(image, ax=axis, label="Simulated completion latency (ms)")
        axis.set_title(f"{scenario}: {kind.upper()}, Prefill {summary['prefill_tokens']}, Decode {summary['decode_steps']}\n"
                       "Uncalibrated analytical simulation; not hardware measurements", fontsize=11)
        stem = "latency_"+re.sub(r"[^a-zA-Z0-9_-]+", "_", scenario)
        save(figure, stem)

    figure, axis = plt.subplots(figsize=(max(10, len(scenario_ids)*0.72), 5.5), layout="constrained")
    ratios = [s["best_reference_speedup"] for s in analysis["scenarios"]]
    axis.bar(range(len(ratios)), [r if r is not None else float("nan") for r in ratios], color="#356ca5")
    for i, ratio in enumerate(ratios):
        axis.text(i, ratio if ratio is not None else 0, f"{ratio:.3g}x" if ratio is not None else "unavailable",
                  ha="center", va="bottom", fontsize=8)
    axis.axhline(1, color="#666666", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(scenario_ids)), scenario_ids, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("Reference time / offline catalogue-best time")
    axis.set_title("Best of declared finite candidates versus e0-s0\nUncalibrated analytical simulation; no search or real-device speedup claim")
    save(figure, "summary_speedup")

    figure, axis = plt.subplots(figsize=(max(10, len(scenario_ids)*0.72), 5.5), layout="constrained")
    colours = {"lru": "#356ca5", "history_frequency": "#b86525"}
    for family, colour in colours.items():
        xs, ys = [], []
        for pair in analysis["prefetch_pairs"]:
            if pair["cache_family"] != family or not pair["complete_pair"]:
                continue
            execution_number = int(pair["execution"][1:])
            jitter = (execution_number-2.5)*0.047 + (-0.16 if family == "lru" else 0.16)
            xs.append(scenario_ids.index(pair["scenario_id"])+jitter)
            ys.append(100*pair["latency_saving_fraction"])
        axis.scatter(xs, ys, color=colour, label=family.replace("_", " "), s=23, alpha=0.8)
    axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(scenario_ids)), scenario_ids, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("Prefetch latency saving (%)\npositive = faster; negative = slower")
    axis.set_title("Same-execution, same-cache-budget prefetch pairs; one dot per execution\nUncalibrated analytical simulation; not repeated statistical samples")
    axis.legend()
    save(figure, "summary_prefetch_effect")
    return artifacts
