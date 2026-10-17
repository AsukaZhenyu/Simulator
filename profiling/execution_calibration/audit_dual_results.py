"""Dual-affinity post-experiment audit; never refits frozen models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as a
import run_dual as r
from verify_dual import verify


def error_summary(results):
    measured = [row for row in results if row["observed_ns"] is not None]
    values = [row["absolute_relative_error"] for row in measured]
    failures = Counter(key for row in results for key, passed in row.get("gates", {}).items() if not passed)
    return {"prediction_count": len(results), "known_prediction_count": len(measured),
            "unknown_prediction_count": len(results) - len(measured),
            "accepted_count": sum(row["calibrated"] for row in results),
            "median_absolute_relative_error": statistics.median(values) if values else None,
            "mean_absolute_relative_error": statistics.mean(values) if values else None,
            "maximum_absolute_relative_error": max(values) if values else None,
            "error_at_most_20pct_count": sum(value <= .2 for value in values),
            "failed_gate_counts": dict(failures)}


def _median(rows, metric):
    values = [value for row in rows for value in row["metrics"].get(metric, [])]
    return statistics.median(values) if values else None


def audit(experiment, output_json, output_md):
    experiment = Path(experiment).resolve()
    if output_json.exists() or output_md.exists():
        raise ValueError("audit outputs must be new files")
    verification = verify(experiment)
    suite = r.document(experiment / "suite.json")
    rows = r.document(experiment / "measurements.json")
    report = r.document(experiment / "analysis" / "report.json")
    manifest = r.document(experiment / "manifest.json")
    primary = [row for row in report["results"] if "@" not in row["strategy"]]
    dispatch = [row for row in report["results"] if "@dispatch_model" in row["strategy"]]
    summary = {"schema": "execution-dual-calibration-summary-audit-0.1", "source_experiment": str(experiment),
               "verification": verification, "original_report_sha256": a.sha(experiment / "analysis" / "report.json"),
               "frozen_model_sha256": manifest["model_frozen_before_heldout_sha256"],
               "audit_source_sha256": a.sha(__file__),
               "primary_19_unique_execution_paths": error_summary(primary),
               "dispatch_15_model_counterparts_not_new_execution_paths": error_summary(dispatch),
               "ranking_groups": report["rankings"], "ffn_parallel_diagnostics": [],
               "host_bracket_evidence": [],
               "diagnostic_predictions_accepted_as_costs": False,
               "diagnostic_policy": "Post-experiment decomposition only; no refit, sample removal, gate change or frozen-cost replacement"}
    for case in suite["cases"]:
        if case["family"] != "ffn":
            continue
        selected = [row for row in rows if row["case"]["id"] == case["id"] and row["split"] == "profile" and row["status"] == "measured"]
        cpu_stages = {stage: _median(selected, "cpu/shard_" + stage) for stage in ("gate", "up", "activation", "down")}
        gpu_stages = {stage: _median(selected, "gpu/shard_" + stage) for stage in ("gate", "up", "activation", "down")}
        parallel = _median(selected, "hybrid/parallel_branches")
        handshake = _median(selected, "hybrid/dispatch_barrier")
        if any(value is None for value in [*cpu_stages.values(), *gpu_stages.values(), parallel, handshake]):
            summary["ffn_parallel_diagnostics"].append({"case_id": case["id"], "status": "unknown_missing_profile_component"})
            continue
        cpu, gpu = sum(cpu_stages.values()), sum(gpu_stages.values())
        ideal, serial = max(cpu, gpu), cpu + gpu
        summary["ffn_parallel_diagnostics"].append({
            "case_id": case["id"], "split": "profile_only_diagnostic",
            "cpu_isolated_stage_medians_ns": cpu_stages, "gpu_isolated_stage_medians_ns": gpu_stages,
            "cpu_stage_sum_ns": cpu, "gpu_stage_sum_ns": gpu, "ideal_max_ns": ideal,
            "serial_component_sum_ns": serial, "measured_parallel_branches_ns": parallel,
            "separately_measured_empty_worker_handshake_ns": handshake,
            "parallel_minus_ideal_max_ns": parallel - ideal,
            "parallel_minus_ideal_and_handshake_ns": parallel - ideal - handshake,
            "parallel_minus_serial_component_sum_ns": parallel - serial,
            "parallel_over_ideal_max": parallel / ideal, "parallel_over_serial_component_sum": parallel / serial,
            "profile_sources": [{"row_id": row["id"], "raw_path": row["raw_path"], "raw_sha256": row["raw_sha256"]} for row in selected],
            "interpretation": "Differences mix dispatch/synchronization, OS scheduling/resource competition, grouped measurement order and median non-additivity; not an independently identified overhead"})
    for row in rows:
        if row["case"]["family"] != "ffn":
            continue
        raw_path = experiment / "raw" / (row["id"] + ".json")
        raw = r.document(raw_path)
        if row["status"] != "measured":
            continue
        payload = json.loads(raw["stdout"])
        bracket = payload.get("diagnostic_completed_work_brackets")
        summary["host_bracket_evidence"].append({"row_id": row["id"], "split": row["split"],
            "raw_path": str(raw_path), "raw_sha256": row["raw_sha256"], "diagnostic": bracket, "observed_affinity": payload.get("observed_affinity"), "dispatch_cpu_id": payload.get("dispatch_cpu_id")})
    brackets = summary["host_bracket_evidence"]
    observed = [row["diagnostic"] for row in brackets if row["diagnostic"] is not None]
    overlaps = [item.get("times_ns", {}).get("completed_work_bracket_overlap_ns") for item in observed]
    summary["host_bracket_summary"] = {"expected_ffn_process_count": sum(row["case"]["family"] == "ffn" for row in rows),
        "observed_diagnostic_count": len(observed), "zero_host_overlap_count": sum(value == 0 for value in overlaps),
        "positive_host_overlap_count": sum(value is not None and value > 0 for value in overlaps),
        "kernel_overlap_proved_count": sum(item.get("kernel_overlap_proved") is True for item in observed),
        "interpretation": "Host completed-call brackets are not GPU kernel events. Actual dual main/dispatch affinity is read back and verified; bracket overlap alone does not establish a per-kernel overlap model."}
    a.dump(output_json, summary)
    p, d = summary["primary_19_unique_execution_paths"], summary["dispatch_15_model_counterparts_not_new_execution_paths"]
    lines = ["# 双逻辑处理器校准结果独立审计", "", "该文件只分解已完成实验，不修改正式预测公式、成本、样本或验收门。", "",
             f"- 原始核验通过：{verification['verified']}；原始进程 {verification['measurement_rows']}；数值检查 {verification['numeric_execution_valid']}。",
             f"- 主要路径：{p['prediction_count']} 个，误差中位数 {p['median_absolute_relative_error']:.1%}，最大 {p['maximum_absolute_relative_error']:.1%}，误差≤20% 为 {p['error_at_most_20pct_count']} 个，全部质量门通过 {p['accepted_count']} 个。",
             f"- dispatch 模型对照：{d['prediction_count']} 条预测，误差中位数 {d['median_absolute_relative_error']:.1%}，最大 {d['maximum_absolute_relative_error']:.1%}，误差≤20% 为 {d['error_at_most_20pct_count']} 个，全部质量门通过 {d['accepted_count']} 个。其中重复的 GPU 预测不是新的执行策略。", "",
             "主要路径失败门计数：`" + json.dumps(p["failed_gate_counts"]) + "`。", "",
             "dispatch 对照失败门计数：`" + json.dumps(d["failed_gate_counts"]) + "`。", "",
             "## 同一边界下的 resident 策略选择", "", "| 案例 | 模型 | 预测最快 | 实测最快 | 选择额外耗时 | 区间不重叠 | 延迟质量接受 |", "|---|---|---|---|---:|---|---|"]
    for ranking in report["rankings"]:
        lines.append(f"| {ranking['case_id']} | {'dispatch' if ';dispatch_model' in ranking['boundary_contract'] else 'ideal'} | {ranking['predicted_order'][0]} | {ranking['observed_order'][0]} | {ranking['selection_regret_relative']:.1%} | {ranking['decisive_pairs']}/{ranking['pair_count']} | {ranking['latency_quality_accepted']} |")
    lines += ["", "## FFN 并行分支的诊断分解", "",
              "下面只使用 profile 进程，比较实测分支总时间与孤立阶段相加的 max/sum。它用于定位公式哪里失效，不作为新的校准成本。空 worker 握手之外的残差没有被归因成单独的派发开销。", "",
              "| 案例 | CPU sum μs | GPU sum μs | max μs | C+G μs | 实测分支 μs | 空握手 μs | 超出 max μs | 超出 max+握手 μs | 超出 C+G μs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["ffn_parallel_diagnostics"]:
        if "status" in row:
            lines.append(f"| {row['case_id']} | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |")
            continue
        keys = ("cpu_stage_sum_ns", "gpu_stage_sum_ns", "ideal_max_ns", "serial_component_sum_ns", "measured_parallel_branches_ns", "separately_measured_empty_worker_handshake_ns", "parallel_minus_ideal_max_ns", "parallel_minus_ideal_and_handshake_ns", "parallel_minus_serial_component_sum_ns")
        lines.append("| " + row["case_id"] + " | " + " | ".join(f"{row[key]/1000:.3f}" for key in keys) + " |")
    b = summary["host_bracket_summary"]
    lines += ["", "## 实际并发是否得到证明", "",
              f"{b['expected_ffn_process_count']} 个 FFN 进程中，观察到 {b['observed_diagnostic_count']} 份 host bracket 诊断；{b['zero_host_overlap_count']} 份 overlap=0，{b['positive_host_overlap_count']} 份>0；kernel_overlap_proved=true 为 {b['kernel_overlap_proved_count']} 份。", "",
              "这些是一次未计时诊断调用的 CPU/GPU host 完成区间，不是每个计时样本的 GPU kernel trace。本轮主线程与 GPU dispatch worker 分别显式绑定两个逻辑处理器，实际掩码已由 native 读回并核验。host bracket 重叠不等于逐 kernel 的重叠时间线，仍须独立检查 max 模型的 heldout 误差；不能仅凭该诊断把调度、同步与设备竞争的贡献严格分离。", "",
              "完整逐条 bracket、原始 SHA256、模型预测来源与质量门保存在 JSON 审计文件。GPU 私有 workspace 仍未知，本实验没有物理容量验收。"]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = audit(args.experiment, args.output_json, args.output_md)
        print(json.dumps({"primary": summary["primary_19_unique_execution_paths"],
                          "dispatch": summary["dispatch_15_model_counterparts_not_new_execution_paths"],
                          "brackets": summary["host_bracket_summary"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("calibration summary audit: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
