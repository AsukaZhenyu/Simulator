"""Evaluate frozen profile in-chain model against uninstrumented heldout totals.

All profile cost CVs (including small stage gaps) are reported and gated. The
original independent-component analysis stays separate and unchanged.
"""
from collections import defaultdict
import statistics


def cv(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def evaluate(rows, model, suite, trace_model):
    result = {"schema": "chain-profile-trace-report-0.1", "quality": suite["quality"],
              "independent_component_baseline_replaced": False,
              "timestamp_overhead_subtracted": False,
              "cv_source_unit": "each profile process and stage/gap/launch cost; every modeled cost retained",
              "heldout_target_unit": "separate uninstrumented completed-work path invocation",
              "physical_context_switch_attribution_proved": False, "results": []}
    if model.get("status") == "unavailable":
        result.update(model_available=False, error=model["error"])
        return result
    profiles = [r for r in rows if r["split"] == "profile" and r["status"] == "measured"]
    per_process = {row["id"]: trace_model.freeze([row]) for row in profiles}
    quality = suite["quality"]
    for path in model["paths"]:
        cid, strategy = path["case_id"], path["strategy"]
        costs = defaultdict(list)
        profile_sources = []
        for row in profiles:
            if row["case"]["id"] != cid:
                continue
            selected = next(x for x in per_process[row["id"]]["paths"] if x["strategy"] == strategy)
            profile_sources.append({"row_id": row["id"], "raw_sha256": row["raw_sha256"], "costs": selected["costs"]})
            for name, values in selected["costs"].items():
                costs[name].append(values)
        heldout = [r for r in rows if r["split"] == "heldout" and r["status"] == "measured" and r["case"]["id"] == cid]
        targets = [r["metrics"][strategy] for r in heldout]
        prediction = trace_model.predict(model, cid, strategy)
        source_cv = {name: {"max_within_process_cv": max(v["cv"] for v in values),
                            "between_process_medians_cv": cv([v["median_ns"] for v in values]),
                            "pooled_cv": path["costs"][name]["cv"]} for name, values in costs.items()}
        observed = statistics.median([v for group in targets for v in group]) if targets else None
        error = abs(prediction["predicted_ns"] - observed) / observed if observed else None
        declared_count = len(suite["heldout_seeds"]) * suite["repeats"]
        gates = {"numeric_and_process_coverage": len(heldout) == declared_count
                 and len(profile_sources) == len(suite["profile_seeds"]) * suite["repeats"]
                 and not any(r["status"] != "measured" for r in rows if r["case"]["id"] == cid),
                 "independent_processes_at_least_three_each_split": len(profile_sources) >= 3 and len(heldout) >= 3,
                 "within_process_cv": bool(targets) and all(x["max_within_process_cv"] <= quality["max_within_process_cv"] for x in source_cv.values())
                 and all(cv(group) <= quality["max_within_process_cv"] for group in targets),
                 "between_process_cv": bool(targets) and all(x["between_process_medians_cv"] <= quality["max_between_process_cv"] for x in source_cv.values())
                 and cv([statistics.median(group) for group in targets]) <= quality["max_between_process_cv"],
                 "heldout_relative_error": error is not None and error <= quality["max_heldout_relative_error"],
                 "model_frozen_before_heldout": all(r.get("frozen_trace_model_sha256_at_start") for r in heldout) and bool(heldout)}
        result["results"].append({"case_id": cid, "strategy": strategy, **prediction,
                                   "observed_uninstrumented_ns": observed, "absolute_relative_error": error,
                                   "profile_cost_cv": source_cv, "profile_sources": profile_sources,
                                   "heldout_within_process_cv": [cv(group) for group in targets],
                                   "heldout_between_process_medians_cv": cv([statistics.median(group) for group in targets]) if targets else None,
                                   "gates": gates, "calibrated": all(gates.values())})
    result.update(model_available=True, all_declared_strategies_calibrated=bool(result["results"]) and all(x["calibrated"] for x in result["results"]))
    return result
