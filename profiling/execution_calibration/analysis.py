"""Profile-only component models and independent-process strategy validation.

This module never executes hardware and never fits a joint strategy duration.
Expression trees are protocol declarations, not learned scheduling algorithms.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    if not isinstance(values, list) or len(values) < 3 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("timings require at least three positive integer nanoseconds")
    mean = statistics.mean(values)
    return {"sample_count": len(values), "median_ns": statistics.median(values),
            "mean_ns": mean, "cv": statistics.pstdev(values) / mean,
            "min_ns": min(values), "max_ns": max(values)}


def terms(expression):
    """Compatibility; expression validation is model-owned."""
    from llm_infer_model.performance.legacy_expressions import terms as validate
    return validate(expression)


def predict(expression, prices):
    """Compatibility; all declared execution arithmetic is model-owned."""
    from llm_infer_model.performance.legacy_expressions import predict as calculate
    return calculate(expression, prices)


def expression_text(expression):
    """Compatibility formatting for the model-owned expression structure."""
    from llm_infer_model.performance.legacy_expressions import expression_text as format_expression
    return format_expression(expression)


def _rows(rows, cid, split, metric):
    return [r for r in rows if r["case"]["id"] == cid and r["split"] == split
            and r["status"] == "measured" and metric in r["metrics"]]


def _median(rows, metric):
    return statistics.median([value for row in rows for value in row["metrics"][metric]])


def _source(row, metric):
    return {"row_id": row["id"], "metric": metric, "split": row["split"],
            "seed": row.get("seed"), "repeat": row.get("repeat"),
            "raw_path": row.get("raw_path"), "raw_sha256": row.get("raw_sha256"),
            **stats(row["metrics"][metric])}


def freeze_model(rows, recipes, observed_metrics=None):
    """Only profile samples of declared independent terms enter the prices."""
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("explicit strategy expressions are required")
    model = {"schema": "execution-component-model-0.1", "trained_on": "profile_only",
             "interpolation": "none", "joint_strategy_times_used_as_costs": False,
             "cases": []}
    for cid, strategies in recipes.items():
        if not isinstance(strategies, dict) or not strategies:
            raise ValueError("case requires explicit strategy expressions")
        used = set().union(*(terms(expression) for expression in strategies.values()))
        if used & set(strategies):
            raise ValueError("a measured strategy total cannot be its own component price")
        prices = {}
        missing = []
        for metric in sorted(used):
            selected = _rows(rows, cid, "profile", metric)
            if not selected:
                missing.append(metric)
                continue
            prices[metric] = {"median_ns": _median(selected, metric),
                              "sources": [_source(row, metric) for row in selected]}
        aliases = (observed_metrics or {}).get(cid, {})
        if set(aliases) - set(strategies) or any(not isinstance(v, str) or not v for v in aliases.values()):
            raise ValueError("observed metric aliases must match declared strategies")
        if used & {aliases.get(strategy, strategy) for strategy in strategies}:
            raise ValueError("an observed strategy total cannot enter its component prices")
        model["cases"].append({"case_id": cid, "strategies": strategies, "observed_metrics": aliases,
                               "prices": prices, "missing_component_metrics": missing})
    return model


def _between(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) >= 2 else None


def median_interval(groups, bootstrap):
    """Hierarchical percentile bootstrap: process resampling, then sample resampling."""
    rng = random.Random(bootstrap["seed"])
    medians = []
    for _ in range(bootstrap["iterations"]):
        processes = [groups[rng.randrange(len(groups))] for _ in groups]
        draws = [group[rng.randrange(len(group))] for group in processes for _ in group]
        medians.append(statistics.median(draws))
    medians.sort()
    def percentile(p):
        index = (len(medians) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return medians[lo] * (hi - index) + medians[hi] * (index - lo) if hi != lo else medians[lo]
    return [percentile(.025), percentile(.975)]


def evaluate(rows, model, quality, bootstrap, *, scope, frozen_before_heldout, comparison_groups=None):
    failed = [r["id"] for r in rows if r["status"] != "measured"]
    report = {"schema": "execution-calibration-report-0.1", "scope": scope,
              "quality": quality, "bootstrap": bootstrap, "frozen_before_heldout": frozen_before_heldout,
              "failed_native_rows": failed, "numeric_execution_valid": bool(rows) and not failed,
              "joint_strategy_times_used_as_costs": False,
              "results": [], "rankings": [], "all_declared_strategies_calibrated": False,
              "gpu_workspace_known": False, "physical_capacity_validated": False,
              "unknown_or_uncovered": [
                  "GPU private scratch/workspace remains unknown; no physical capacity acceptance",
                  "No quantized or pretrained-weight accuracy claim",
                  "No complete multi-layer Dense/MoE Prefill-to-Decode runtime validation",
                  "No general resource-contention or arbitrary copy/compute-overlap law",
                  "No future-route knowledge and no model-quality claim",
                  "Only exact declared shapes, kernels, devices, thread count and timing boundary are evaluated"]}
    for case in model["cases"]:
        cid, prices = case["case_id"], case["prices"]
        for strategy, expression in case["strategies"].items():
            target = case.get("observed_metrics", {}).get(strategy, strategy)
            used = terms(expression)
            missing = sorted(used - set(prices))
            heldout = _rows(rows, cid, "heldout", target)
            result = {"case_id": cid, "strategy": strategy, "observed_metric": target, "expression": expression,
                      "expression_text": expression_text(expression), "calibrated": False,
                      "missing_component_metrics": missing, "predicted_ns": None,
                      "observed_ns": None, "source_evidence": [], "heldout_evidence": []}
            if missing or not heldout:
                result["unknown_reason"] = "missing profile components" if missing else "missing heldout strategy measurements"
                report["results"].append(result)
                continue
            prediction = predict(expression, {m: prices[m]["median_ns"] for m in used})
            observed = _median(heldout, target)
            sources = [source for metric in sorted(used) for source in prices[metric]["sources"]]
            truth = [_source(row, target) for row in heldout]
            between_profile = {metric: _between([s["median_ns"] for s in prices[metric]["sources"]]) for metric in sorted(used)}
            between_heldout = _between([s["median_ns"] for s in truth])
            incomplete = any(r["case"]["id"] == cid and r["status"] != "measured" for r in rows)
            gates = {
                "numeric_and_process_coverage": not incomplete,
                "model_frozen_before_heldout": frozen_before_heldout is True,
                "within_process_cv": all(s["cv"] <= quality["max_within_process_cv"] for s in sources + truth),
                "between_process_cv": all(v is not None and v <= quality["max_between_process_cv"]
                                          for v in [*between_profile.values(), between_heldout]),
                "heldout_relative_error": abs(prediction - observed) / observed <= quality["max_heldout_relative_error"]}
            result.update(predicted_ns=prediction, observed_ns=observed,
                          signed_error_ns=prediction - observed, absolute_error_ns=abs(prediction - observed),
                          absolute_relative_error=abs(prediction - observed) / observed,
                          heldout_median_95pct_interval_ns=median_interval([r["metrics"][target] for r in heldout], bootstrap),
                          between_process_cv={"profile_components": between_profile, "heldout_strategy": between_heldout},
                          source_evidence=sources, heldout_evidence=truth, gates=gates,
                          calibrated=all(gates.values()))
            report["results"].append(result)
    grouped = {}
    for r in report["results"]:
        contract = (comparison_groups or {}).get(r["case_id"], {}).get(r["strategy"], "declared_common_boundary")
        grouped.setdefault((r["case_id"], contract), []).append(r)
    for (cid, contract), declared in sorted(grouped.items()):
        results = [r for r in declared if r["observed_ns"] is not None]
        if len(results) < 2:
            continue
        predicted_order = sorted(results, key=lambda r: (r["predicted_ns"], r["strategy"]))
        observed_order = sorted(results, key=lambda r: (r["observed_ns"], r["strategy"]))
        pairs = []
        for index, left in enumerate(results):
            for right in results[index + 1:]:
                li, ri = left["heldout_median_95pct_interval_ns"], right["heldout_median_95pct_interval_ns"]
                decisive = li[1] < ri[0] or ri[1] < li[0]
                psign = (left["predicted_ns"] > right["predicted_ns"]) - (left["predicted_ns"] < right["predicted_ns"])
                osign = (left["observed_ns"] > right["observed_ns"]) - (left["observed_ns"] < right["observed_ns"])
                pairs.append({"left": left["strategy"], "right": right["strategy"],
                              "median_order_correct": psign == osign,
                              "observed_95pct_intervals_disjoint": decisive,
                              "latency_quality_accepted": left["calibrated"] and right["calibrated"]})
        chosen, best = predicted_order[0], observed_order[0]
        report["rankings"].append({"case_id": cid, "boundary_contract": contract,
            "predicted_order": [r["strategy"] for r in predicted_order],
            "observed_order": [r["strategy"] for r in observed_order],
            "pairs": pairs, "correct_pairs": sum(p["median_order_correct"] for p in pairs),
            "pair_count": len(pairs), "decisive_pairs": sum(p["observed_95pct_intervals_disjoint"] for p in pairs),
            "correct_decisive_pairs": sum(p["median_order_correct"] and p["observed_95pct_intervals_disjoint"] for p in pairs),
            "selection_regret_ns": chosen["observed_ns"] - best["observed_ns"],
            "selection_regret_relative": (chosen["observed_ns"] - best["observed_ns"]) / best["observed_ns"],
            "latency_quality_accepted": len(results) == len(declared) and all(r["calibrated"] for r in results),
            "comparison_scope": "same case, geometry, input/output placement contract and native processes; interval overlap is inconclusive"})
    report["all_declared_strategies_calibrated"] = bool(report["results"]) and not failed and all(r["calibrated"] for r in report["results"])
    return report


def _csv(path, headers, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, report, model, rows):
    output = Path(output)
    if output.exists():
        raise ValueError("report output must be a new directory")
    output.mkdir(parents=True)
    dump(output / "component-model.json", model)
    dump(output / "report.json", report)
    errors = []
    for r in report["results"]:
        errors.append({k: r.get(k) for k in ("case_id", "strategy", "predicted_ns", "observed_ns", "signed_error_ns", "absolute_error_ns", "absolute_relative_error", "calibrated", "unknown_reason")})
    _csv(output / "prediction-errors.csv", list(errors[0]) if errors else ["case_id", "strategy"], errors)
    samples, processes, checks = [], [], []
    for row in rows:
        if row["status"] != "measured":
            continue
        base = {"row_id": row["id"], "case_id": row["case"]["id"], "split": row["split"],
                "seed": row.get("seed"), "repeat": row.get("repeat"),
                "raw_path": row.get("raw_path"), "raw_sha256": row.get("raw_sha256")}
        for metric, values in row["metrics"].items():
            processes.append({**base, "metric": metric, **stats(values)})
            samples.extend({**base, "metric": metric, "sample_index": index, "duration_ns": value}
                           for index, value in enumerate(values))
        checks.append({**base, "checks": row.get("checks"), "workspaces": row.get("workspaces"),
                       "fixture_hashes": row.get("fixture_hashes"), "ports": row.get("ports")})
    for filename, data in (("raw-timing-samples.csv", samples), ("process-statistics.csv", processes)):
        _csv(output / filename, list(data[0]) if data else ["row_id"], data)
    dump(output / "numerical-and-workspace-evidence.json", checks)
    lines = ["# 执行变换性能校准报告", "", report["scope"], "",
             "预测只使用 profile 独立阶段的中位数；heldout 是新的进程与输入 seed。真实策略总时间不作为该策略的预测成本。所有样本保留，不删离群点。", "",
             f"- 数值与运行检查：{'通过' if report['numeric_execution_valid'] else '存在失败'}。",
             f"- 策略验收：{sum(r['calibrated'] for r in report['results'])}/{len(report['results'])} 通过全部预声明门。",
             f"- 全部策略校准通过：{report['all_declared_strategies_calibrated']}。", "",
             "质量门同时要求：进程内 CV、独立进程间 CV、holdout 相对误差、数值/覆盖检查以及预测模型在 holdout 前冻结。误差小但波动超门仍不接受。", "",
             "## 预测与真实执行", "", "| 形状案例 | 策略 | 预测 μs | 实测 μs | 绝对误差 μs | 相对误差 | 验收 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in report["results"]:
        if r["observed_ns"] is None:
            lines.append(f"| {r['case_id']} | {r['strategy']} | unknown | unknown | unknown | unknown | {r['unknown_reason']} |")
        else:
            failed_gates = ", ".join(k for k, v in r["gates"].items() if not v)
            lines.append(f"| {r['case_id']} | {r['strategy']} | {r['predicted_ns']/1000:.3f} | {r['observed_ns']/1000:.3f} | {r['absolute_error_ns']/1000:.3f} | {r['absolute_relative_error']:.1%} | {'通过' if r['calibrated'] else failed_gates} |")
    lines += ["", "## 策略排序", "", "排序用于检查模拟器能否选择较好的实际方案。区间重叠的微小差异不解释为稳定优势；质量门失败的排序仅为诊断。", ""]
    for ranking in report["rankings"]:
        lines += [f"- **{ranking['case_id']}**：预测 `{' < '.join(ranking['predicted_order'])}`；实测 `{' < '.join(ranking['observed_order'])}`。",
                  f"  中位数排序 {ranking['correct_pairs']}/{ranking['pair_count']} 对；区间不重叠的排序 {ranking['correct_decisive_pairs']}/{ranking['decisive_pairs']} 对；按预测选择的额外实测耗时 {ranking['selection_regret_relative']:.1%}；延迟质量验收 {ranking['latency_quality_accepted']}（该项不是排序正确性门）。"]
    lines += ["", "## 预测公式与证据", ""]
    for r in report["results"]:
        lines.append(f"- `{r['case_id']}/{r['strategy']}` = `{r['expression_text']}`。")
    lines += ["", "逐个组件的 profile 来源、heldout 策略来源和 raw SHA256 保存在 `report.json`；每一条原始计时在 `raw-timing-samples.csv`，数值/工作区声明在 `numerical-and-workspace-evidence.json`。", "",
              "## 未知与尚未验证", ""]
    lines.extend("- " + text for text in report["unknown_or_uncovered"])
    if report["failed_native_rows"]:
        lines += ["", "失败进程：" + ", ".join(report["failed_native_rows"])]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _write_svg(output / "prediction-vs-measurement.svg", report["results"])


def _write_svg(path, results):
    valid = [r for r in results if r["observed_ns"] is not None]
    height = max(160, 80 + len(valid) * 44)
    width, left, scale_width = 1200, 460, 580
    maximum = max([r[k] for r in valid for k in ("predicted_ns", "observed_ns")] or [1])
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="Arial, sans-serif" font-size="12">',
               '<text x="20" y="24">Exact-shape component prediction vs independent-process measurement</text>',
               '<rect x="470" y="38" width="16" height="9" fill="#2457a5"/><text x="492" y="47">prediction</text>',
               '<rect x="590" y="38" width="16" height="9" fill="#e88c30"/><text x="612" y="47">measurement (microseconds)</text>']
    for index, r in enumerate(valid):
        y = 78 + index * 44
        label = html.escape(r["case_id"] + " / " + r["strategy"])
        content.append(f'<text x="20" y="{y+10}">{label}</text>')
        for offset, key, color in ((0, "predicted_ns", "#2457a5"), (15, "observed_ns", "#e88c30")):
            bar = r[key] / maximum * scale_width
            content.append(f'<rect x="{left}" y="{y+offset}" width="{bar:.3f}" height="10" fill="{color}"/>')
            content.append(f'<text x="{left+bar+7:.3f}" y="{y+offset+9}">{r[key]/1000:.2f}</text>')
        content.append(f'<text x="1090" y="{y+12}" fill="{"#22703b" if r["calibrated"] else "#b12828"}">{r["absolute_relative_error"]:.1%} {"PASS" if r["calibrated"] else "FAIL"}</text>')
    content += ['</g>', '</svg>']
    Path(path).write_text("\n".join(content) + "\n", encoding="utf-8", newline="\n")
