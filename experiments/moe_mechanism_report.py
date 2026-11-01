"""Render mechanism-constrained calibration evidence, preserving old failures."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

from experiments.h2d_moe_calibration_report import aggregate_moe, number, percentage, receipt, table


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    campaign, reference, out = Path(a.campaign), Path(a.reference), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = read(campaign / "INDEPENDENT_EVALUATION.json")
    conditions = aggregate_moe(evaluation)
    catalog = read(campaign / "WINDOW_PRICES.json")
    forecasts = read(campaign / "PREDICTIONS.json")["forecasts"]
    forecast_windows = sum(row.get("window_count", 0) for row in forecasts)
    extrapolated_windows = sum(row.get("extrapolated_windows", 0) for row in forecasts)
    prices = catalog["prices"]
    assert all(r["feature_policy"] == "mechanism_v1" for r in prices)
    prohibited = [r["key"] for r in prices if number(r["per_attention_pair_ns"]) and "query_tokens_times_KV" not in r["work_spec"]["allowed_features"]]
    assert not prohibited, "a fitted price violates its declared mechanism"
    kinds = Counter(r["work_spec"]["kind"] for r in prices)
    posthoc = read(campaign / "OLD_CASE_POSTHOC.json")
    assert posthoc["independent_validation"] is False and posthoc["parameters_refitted"] is False
    assert posthoc["price_freeze"] == receipt(campaign / "PRICE_FREEZE.json")
    posthoc_rows = []
    compared = [row for row in posthoc["rows"] if "old_APE" in row and "proposed_APE" in row]
    decode_improved = sum(row["proposed_APE"]["decode_with_sampling"] < row["old_APE"]["decode_with_sampling"] for row in compared)
    decode_worse = sum(row["proposed_APE"]["decode_with_sampling"] > row["old_APE"]["decode_with_sampling"] for row in compared)
    posthoc_exceptions = "；".join(escape(row["placement"] + " / " + row["request_id"]) + " 的 Decode：" + percentage(row["old_APE"]["decode_with_sampling"]) + " → " + percentage(row["proposed_APE"]["decode_with_sampling"]) for row in compared if max(row["old_APE"]["decode_with_sampling"], row["proposed_APE"]["decode_with_sampling"]) > .1)
    for row in posthoc["rows"]:
        if "old_APE" in row and "proposed_APE" in row:
            posthoc_rows.append([row["placement"], row["request_id"], percentage(row["old_APE"]["total"]), percentage(row["proposed_APE"]["total"]), percentage(row["old_APE"]["decode_with_sampling"]), percentage(row["proposed_APE"]["decode_with_sampling"])])
        else:
            posthoc_rows.append([row["placement"], row["request_id"], "原数值失败，不评分", "—", "—", "—"])
    mechanism_rows = []
    for kind, count in sorted(kinds.items()):
        rows = [r for r in prices if r["work_spec"]["kind"] == kind]
        mechanism_rows.append([kind, count, sum("query_tokens_times_KV" in r["work_spec"]["allowed_features"] for r in rows), sum(number(r["per_attention_pair_ns"]) != 0 for r in rows)])
    sources = {name: receipt(campaign / name) for name in ("manifest.json", "PRICE_FREEZE.json", "PREDICTION_FREEZE.json", "WINDOW_PRICES.json", "INDEPENDENT_EVALUATION.json")}
    for name in ("OLD_CASE_POSTHOC.json", "MECHANISM_AUDIT.json", "FORECAST_SUPPORT_AUDIT.json"):
        if (campaign / name).is_file():
            sources[name] = receipt(campaign / name)
    sources["original_numeric_failure"] = receipt(reference / "NUMERICAL_ENDPOINT_DIAGNOSTIC.json")
    protocol = Path(__file__).resolve().parents[1] / "build/moe-mechanism-calibration-20261029"
    verification = read(protocol / "FINAL_VERIFICATION.json")
    assert verification["protected_files_unchanged"] and verification["compiler_unchanged"]
    sources["scope_and_regression_verification"] = receipt(protocol / "FINAL_VERIFICATION.json")
    audit = read(protocol / "INDEPENDENT_AUDIT.json")
    assert audit["passed"] is True
    sources["independent_provenance_audit"] = receipt(protocol / "INDEPENDENT_AUDIT.json")
    summary = dict(schema="moe-mechanism-correction-report-1", generated_at_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                   independent_conditions=conditions, independent_runs={k:evaluation[k] for k in ("declared", "evaluated", "passed", "mean_APE", "worst_APE")},
                   mechanism_prices=dict(count=len(prices), kinds=dict(kinds), prohibited_KV_coefficients=len(prohibited)),
                   forecast_support=dict(windows=forecast_windows, extrapolated_windows=extrapolated_windows, explicit_extrapolation=True),
                   original_case_posthoc=posthoc,
                   numerical_issue_4_modified=False, original_results_preserved=True, parameters_fit_to_heldout_times=False)
    (out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    result_rows = []
    for row in conditions["rows"]:
        if row["status"] != "evaluated":
            result_rows.append([row["placement"], row["request_id"], "不评分，保留失败", "—", "—", "—"])
        else:
            result_rows.append([row["placement"], row["request_id"], f"{row['measured_mean_ns']['total']/1e6:.2f}", percentage(row["APE"]["total"]), percentage(row["APE"]["prefill"]), percentage(row["APE"]["decode_with_sampling"])])
    links = "".join(f'<li><a href="{Path(s["path"]).as_uri()}">{escape(name)}</a><br><code>{s["sha256"]}</code></li>' for name,s in sources.items())
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MoE 成本变量约束修正</title>
<style>body{{font:17px/1.7 'Microsoft YaHei',sans-serif;color:#171717;background:white;max-width:1120px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:34px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:16px 0}}th,td{{border:1px solid #bbb;padding:8px;text-align:left}}th{{background:#f4f4f4}}code{{font-size:12px;overflow-wrap:anywhere}}.box{{border:1px solid #777;padding:14px}}a{{color:#153d70}}</style>
<h1>MoE 成本模型：按实际工作约束变量</h1>
<p>本轮仅修改价格模型的变量依赖。H2D、CPU 外部资源模型、原生执行器、DLL、编译器与数值一致性判据均保持原样。第 4 项数值异常未修复、未改判；旧结果和失败证据保留。</p>
<h2>一、修正了什么</h2>
<p>原来所有窗口都可拟合固定项、token 项、q×KV 项和字节项。现在先由 modeling 根据实际 op、张量与权重形状声明允许的变量，再由 profiling 仅从校准集测量值估计这些变量的系数；非法的非零系数会被模型拒绝。</p>
{table(['工作类别','允许的依赖及边界'],[['FFN／普通矩阵','实际局部行数；固定矩阵、专家维度和 dtype 属于价格结构。禁止 KV 项。'],['已识别 norm／SwiGLU','使用真实局部 token 维度，末层裁剪为 1 时按 1 计价。禁止 KV 项。'],['Attention 核心／含 Attention 的混合窗口','以实际算子与 q、K 几何确认后保留 q×KV；保留固定 head 几何。'],['传输响应','只有记录到实际复制调用时才允许字节项；不额外叠加已包含的设备耗时。'],['无法识别内部工作的 HOST／其他节点','固定条件下的响应价格，显式标记未知，保留尺寸范围守卫。未声称任意尺寸下均为常量。']])}
<p>当前事件不能独立证明主机上的 mask 构造工作，因此没有擅自给普通 HOST 增加 KV 项。API 只对有明确阶段证据的 attention_mask_prepare 允许该项。缓存、专家复用、OS 状态没有用猜测值补齐；调用次数也没有被伪装成已独立识别的固定开销。</p>
{table(['机制类别','价格数量','允许 KV 项','实际非零 KV 项'],mechanism_rows)}
<p>共 {len(prices)} 项价格，非法非零 KV 系数为 {len(prohibited)}。旧价表仍能读取，并标为 legacy_affine_v1，不冒充新的机制约束模型。</p>
<p>相关软件回归 {verification['tests_passed']} 项通过。审计确认 17 个保护文件、44 个编译器文件及本轮原生程序均未改变，价格与预测冻结顺序正确。</p>
<h2>二、如何验收</h2>
<p>校准继续使用原来的分项校准记录，不读取旧验证耗时重新估计参数。原验证案例只作事后对照；新的验证请求按同一冻结数据集每个“领域×长短”类别的 ID 排序选第 2 条，与校准集及旧验证请求均不重叠。选择规则在新计时之前固定。</p>
<p>先冻结价格，采集新请求的算子、张量形状和搬运工作清单，移除耗时后计算并冻结预测，再运行 CPU 和混合模式各两个新的无观测进程。本轮工作清单没有记录逐 token 的专家 ID／复用轨迹，未据此虚构缓存命中状态。每个请求取两次实测均值评分。数值判据保持输入 token、生成 token 和完整最终 logits 摘要一致；不一致仍保留在分母并单独列出。</p>
<p>{forecast_windows} 个预测窗口中，{extrapolated_windows} 个显式外推；没有将它们冒充为校准域内价格。未知或数值不一致条件均未静默替换。</p>
<div class="box">{conditions['evaluated']}/{conditions['declared']} 个条件可评分；总时间／Prefill／Decode 三项误差同时 ≤10%：{conditions['passed']}/{conditions['declared']}。可比条件的 MAPE 分别为 {percentage(conditions['mean_APE']['total'])}、{percentage(conditions['mean_APE']['prefill'])}、{percentage(conditions['mean_APE']['decode_with_sampling'])}。</div>
{table(['模式','新请求','实测总时间/ms','总时间误差','Prefill误差','Decode误差'],result_rows)}
<p>逐次运行的原始判定也保留：{evaluation['evaluated']}/{evaluation['declared']} 次可评分，三项同时合格 {evaluation['passed']}/{evaluation['declared']}。这里的均值口径不会覆盖逐次标志。</p>
<h2>三、结果应如何解释</h2>
<p>本轮验证的是按工作机制限制变量后，组件价格能否预测未用于本次修正的请求。旧案例上的变化属于诊断对照，不能代替本表的独立结果，也不能把新旧两组不同请求的 MAPE 差值直接称为精度提升。</p>
<p>以下完整保留旧案例对照。{len(compared)} 个可比旧条件中，Decode 误差下降 {decode_improved} 项、上升 {decode_worse} 项；约束不合理变量不保证每条已知样本都改善，没有据此继续调参或改判。涉及 10% 标准的旧条件：{posthoc_exceptions}。</p>
{table(['旧模式','旧请求','旧总时间误差','新总时间误差','旧Decode误差','新Decode误差'],posthoc_rows)}
<p>这仍是固定模型、后端和运行生命周期下的完成响应模型。未分类工作、资源状态与缓存历史可能造成剩余偏差；没有实现操作系统调度模拟。数值失败条件不用于宣称性能准确或策略排序正确，第 4 项仍等待用户后续决定。</p>
<h2>四、可复现证据</h2><ul>{links}</ul><p><a href="SUMMARY.json">机器可读汇总</a></p></html>"""
    (out / "REPORT_zh.html").write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps(dict(report=str((out / "REPORT_zh.html").resolve()), comparable_conditions=conditions["evaluated"], condition_mean_APE=conditions["mean_APE"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
