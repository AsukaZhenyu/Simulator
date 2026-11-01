"""Render the frozen 2026-10-29 H2D / MoE evidence; never change prices.

The prose diagnosis is specific to the bound v8 evaluation snapshot, rather
than a generic conclusion to attach to an arbitrary future experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from html import escape
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return float(Fraction(value["numerator"], value["denominator"]))
    return float(value)


def receipt(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def percentage(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{escape(str(v))}</th>" for v in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def aggregate_moe(data):
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[row["placement"], row["request_id"]].append(row)
    result = []
    for (placement, request), rows in groups.items():
        if any(r["status"] != "evaluated" for r in rows):
            result.append(dict(placement=placement, request_id=request, status="not_evaluated", records=rows))
            continue
        assert len(rows) == 2, "two predeclared independent bare executions required"
        phases = ("total", "prefill", "decode_with_sampling")
        predicted = {p: number(rows[0]["predicted_ns"][p]) for p in phases}
        assert all(number(r["predicted_ns"][p]) == predicted[p] for r in rows for p in phases)
        measured = {p: sum(r["measured_ns"][p] for r in rows) / len(rows) for p in phases}
        ape = {p: abs(predicted[p] / measured[p] - 1) for p in phases}
        result.append(dict(placement=placement, request_id=request, status="evaluated", independent_runs=len(rows), predicted_ns=predicted, measured_mean_ns=measured, APE=ape, all_phases_under_10pct=all(v <= .1 for v in ape.values())))
    scored = [r for r in result if r["status"] == "evaluated"]
    return dict(rows=result, declared=len(groups), evaluated=len(scored), passed=sum(r["all_phases_under_10pct"] for r in scored), mean_APE={p: sum(r["APE"][p] for r in scored) / len(scored) if scored else None for p in ("total", "prefill", "decode_with_sampling")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2d-out", required=True)
    parser.add_argument("--moe-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    hpath = Path(args.h2d_out) / "INDEPENDENT_EVALUATION.json"
    mpath = Path(args.moe_out) / "INDEPENDENT_EVALUATION.json"
    assert receipt(mpath)["sha256"] == "021546a565eb0f8c25fe2c2987a4555933a49f6f7d2cf641fc46a823cf6593f9", "diagnostic prose is bound to the original v8 snapshot"
    h, m = read(hpath), read(mpath)
    assert h["parameters_refitted"] is False
    assert m["parameters_refitted"] is False and m["whole_model_multiplier_fitted"] is False
    verification_path = Path(__file__).resolve().parents[1] / "build/h2d-moe-calibration-20261028/COMPILER_VERIFICATION.json"
    verification = read(verification_path)
    assert verification["unchanged"] and not verification["changed"]
    forecasts = read(Path(args.moe_out) / "PREDICTIONS.json")["forecasts"]
    extrapolated = [r for r in forecasts if r.get("extrapolated_windows", 0) > 0]
    tests_path = verification_path.with_name("FINAL_TESTS_V2.json")
    tests = read(tests_path)
    assert tests["returncode"] == 0
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grouped = aggregate_moe(m)
    summary = dict(schema="h2d-real-moe-calibration-report-1", generated_at_utc=datetime.now(timezone.utc).isoformat(), sources=dict(h2d=receipt(hpath), moe=receipt(mpath), compiler_verification=receipt(verification_path), software_tests=receipt(tests_path)), H2D={k: v for k, v in h.items() if not isinstance(v, (dict, list))}, MoE_independent_run_metrics={k: m[k] for k in ("declared", "evaluated", "passed", "mean_APE", "worst_APE", "parameters_refitted", "whole_model_multiplier_fitted")}, MoE_mean_latency_by_request=grouped, MoE_forecast_domain=dict(forecasts=len(forecasts),forecasts_with_extrapolated_windows=len(extrapolated),all_prices_in_calibration_domain=sum(r.get("status")=="predicted" and r.get("extrapolated_windows")==0 for r in forecasts)), compiler_changes=False, paper_updated=False)
    for name in ("NUMERICAL_ENDPOINT_DIAGNOSTIC.json", "DECODE_FAILURE_DIAGNOSIS.json", "DECODE_COMPONENT_DIAGNOSTIC.json", "CALIBRATION_AUDIT.json", "FORECAST_SUPPORT_AUDIT.json", "PRICE_FREEZE.json", "PREDICTION_FREEZE.json"):
        summary["sources"][name] = receipt(Path(args.moe_out) / name)
    summary["sources"]["scoped_H2D_catalog"] = receipt(Path(args.h2d_out) / "scoped-lifecycle-catalog.json")
    summary["all_acceptance_checks_passed"] = False
    summary["hardware_measurements_completed"] = True
    (out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    hrows = [
        ["首次 COPY", "42", percentage(h["first_MAPE"]), percentage(h["first_worst_APE"]), f"{sum(r['passed'] for r in h['first_rows'])}/42"],
        ["连续 COPY 窗口", "84", percentage(h["window_MAPE"]), percentage(h["window_worst_APE"]), f"{sum(r['passed'] for r in h['window_rows'])}/84"],
    ]
    failed = [[r["memory_mode"], f"{r['bytes']/1024:g} KiB", f"{r['idle_ms']:g} ms", percentage(r["APE"])] for r in h["first_rows"] if not r["passed"]]
    moe_rows = []
    for r in grouped["rows"]:
        if r["status"] != "evaluated":
            moe_rows.append([r["placement"], r["request_id"], "未完成数值/价格域验收", "—", "—", "—"])
        else:
            moe_rows.append([r["placement"], r["request_id"], f"{r['measured_mean_ns']['total']/1e6:.2f}", percentage(r["APE"]["total"]), percentage(r["APE"]["prefill"]), percentage(r["APE"]["decode_with_sampling"])])
    source_links = "".join(f'<li><a href="{Path(v["path"]).as_uri()}">{escape(k)} 证据 JSON</a>；SHA256 <code>{v["sha256"]}</code></li>' for k, v in summary["sources"].items())
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H2D 与真实 MoE 性能校准</title>
<style>body{{font:17px/1.7 'Microsoft YaHei',sans-serif;color:#171717;background:white;max-width:1100px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:36px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:16px 0}}th,td{{border:1px solid #bbb;padding:9px;text-align:left}}th{{background:#f4f4f4}}code{{overflow-wrap:anywhere;font-size:13px}}.box{{border:1px solid #777;padding:16px}}.formula{{font:18px 'Times New Roman',serif;border-left:3px solid #333;padding:12px}}a{{color:#153d70}}</style>
<h1>H2D 与真实 MoE 性能校准</h1>
<p>本轮只处理传输与真实模型的性能计价；编译器及论文文件保持本轮开始时的内容。指标是冻结预测与独立运行的耗时误差，CV 保留作诊断，不作为均值预测的否决条件。</p>
<h2>一、H2D：区分首次响应与连续搬运</h2>
<p>原来的单一恢复曲线不能稳定解释首次搬运。本轮沿用实际执行器的 COPY、原有完整计算图准备过程及独立缓冲区，分别测量连续响应和闲置后的首次响应；没有额外插入“改善成绩”的预热搬运。</p>
<p class="formula">T(N,b,i) = b<sub>m</sub>(b) + p<sub>m</sub>(b,i) + (N−1)b<sub>m</sub>(b)</p>
<p>b 是搬运字节数，i 是事先声明的闲置时间，m 区分 registered/pageable。b<sub>m</sub> 是连续搬运响应，p<sub>m</sub> 是首次响应与连续响应的差值。第一次搬运的预测不使用未来窗口长度 N。</p>
{table(['指标','条件数','MAPE','最差条件误差','≤10% 条件数'],hrows)}
<p>校准与验证各六个新进程，覆盖七种大小（4 KiB–8 MiB）、0/2/10 ms 三种闲置状态和两种主机内存模式；每个首次条件的独立验证均值来自 216 次记录，每个连续窗口来自 72 次记录。验证使用相同有限坐标与独立进程，不声称覆盖任意新大小、闲置时间、后台负载或任意运行时。</p>
<p>完整首 COPY 网格仍保留 first_qualified=false，以下三个坐标未达到逐条件 10% 标准；可复用资格表明确拒绝它们，原始记录没有删除。</p>
{table(['主机内存','搬运大小','闲置时间','均值误差'],failed)}
<h2>二、MoE：把真实执行拆成不重叠的计价窗口</h2>
<p>真实模型为本机固定的 OLMoE-1B-7B-0924-Instruct Q4_K_M，包含 16 层、每层 64 个专家、top-8 路由。比较 CPU 与 CPU/GPU 混合执行；实际输入、输出 token、算子形状和专家工作由真实运行采集。</p>
<p>profiling 负责采集 CPU 的实际节点、自然 GPU 完成边界、搬运字节及主机工作窗口，并从校准集估计参数。modeling 负责按照不重叠窗口和依赖关系计算时间；不会把异步提交的返回时间当成 GPU 完成时间，也不会在已包含等待的窗口上再次叠加 GPU 或传输成本。</p>
<p class="formula">t<sub>w</sub> = a<sub>w</sub> + b<sub>w</sub>q + c<sub>w</sub>qK + d<sub>w</sub>B；T = Σ t<sub>w</sub></p>
<p>q 是该窗口实际处理的 token 数，K 是真实的 padded KV 长度，B 是实际请求的 H2D 字节数。系数非负，缺少价格或越出支持域时明确标记。校准表同时保留每项价格的样本数及设计矩阵秩；有限尺寸下的共线系数不能分别解释为已识别的物理常数。此处的 runtime bridge 包含具体实现的自然等待和设备执行，不能解释为独立 GPU 核心的物理成本。</p>
<p>参数仅使用校准集分项计时。验证集的算子、路由和形状清单去除耗时后才用于预测，预测冻结后才运行新的无观测版本。结果必须先通过输入 token、生成 token 与最终完整 logits 摘要一致性检查；这不等于验证了每一步中间 logits 或所有内部路由逐项相同。</p>
<p>在 {len(forecasts)} 个冻结预测中，{len(extrapolated)} 个含超出校准坐标范围的窗口。外推是在独立验证前明确启用的，每窗标志保留；不能把这些预测称为全部价格都在校准范围内。</p>
<div class="box">两次独立无观测运行取每个请求的耗时均值：{grouped['evaluated']}/{grouped['declared']} 个条件完成评分，三阶段误差同时 ≤10% 的条件为 {grouped['passed']}/{grouped['declared']}。总时间 MAPE {percentage(grouped['mean_APE']['total'])}；Prefill {percentage(grouped['mean_APE']['prefill'])}；Decode（含采样）{percentage(grouped['mean_APE']['decode_with_sampling'])}。</div>
{table(['执行模式','请求','实测平均总时长/ms','总时间误差','Prefill误差','Decode误差'],moe_rows)}
<p>完整逐次运行结果仍保留在原始验收 JSON：{m['evaluated']}/{m['declared']} 次运行完成评分，三阶段同时合格 {m['passed']}/{m['declared']}。均值验收不会改写逐次标志。请求预热、模型加载、分词和重置不计入推理目标时间；自然衔接、必要等待和采样计入。</p>
<h2>三、没有通过的条件及原因</h2>
<p>可比的 11 个条件，其总耗时与 Prefill 均值误差均低于 10%。剩余问题集中在以下三项，未按全局平均值将它们改为通过：</p>
<ul><li><b>CPU 短代码请求的 Decode：</b>预测 268.908 ms，两个新进程实测均值 302.558 ms，低估 11.12%。此前有观测工作清单的对应阶段是 268.228 ms，模型与它只差约 0.25%；新裸跑的核心 forward 变慢，衔接段只有约 2 ms，现有证据不能把约 34 ms 差额解释为漏掉的衔接。具体可用算力、缓存及历史条件仍未分离证明。</li>
<li><b>混合长数学请求的 Decode：</b>预测 226.783 ms，独立实测均值 182.806 ms，高估 24.06%。有观测诊断阶段本身约 179.885 ms，故偏差在价格组合时已经存在。诊断定位到 CPU 节点约 +20.47 ms、自然 bridge 约 +18.07 ms、HOST 约 +8.36 ms；是对新工作/历史状态的价格泛化不足，不能称作不可控随机波动。</li>
<li><b>混合长代码请求的数值端点：</b>两个裸跑的完整最终 logits 摘要彼此相同，却都不同于有观测清单；生成 token 相同。最大 logit 从 26.83497 变为 26.45177，已有 top-8 差异表明不能只按末位浮点差异处理。缺少完整向量对照和机制定位，保留数值门失败；这两次运行不进入性能准确性均值。</li></ul>
<p>上述均为结果冻结后的只读诊断，没有用诊断阶段耗时重新拟合价格。现阶段能声明有限范围的总耗时均值准确性，不能声明全条件、全阶段、全部数值端点均已验收。</p>
<h2>四、结论的边界</h2>
<p>H2D 结果证明的是本机、固定生命周期和有限参数域内的均值预测。MoE 结果证明的是固定真实模型及 CPU/混合实现上的分项价格组合能力；真实工作清单属于 modeling 的评估输入，未交给 mapping 作为未来专家信息。上述结果不能直接推广到新的 CUDA 实现、任意 token 长度、统一内存设备或未测量硬件。</p>
<p>本轮没有用完整验证请求的时间拟合整程序乘数，也没有按验证结果删除样本。若有未通过或域外条件，应按原结果报告，不把它们改名为已完成。</p>
<h2>五、证据与复用</h2><p>最终相关软件回归 105 项通过；编译器 44 个文件哈希均未改变。完整原始数据、失败版本、解析修订和预测冻结记录都保留在磁盘。</p><ul>{source_links}</ul>
<p><a href="SUMMARY.json">本报告机器可读汇总</a>。参数表和冻结记录保存在对应实验目录，报告生成器只读取结果，不修改模型与价格。</p></html>"""
    (out / "REPORT_zh.html").write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps(dict(report=str(out / "REPORT_zh.html"), summary=str(out / "SUMMARY.json"), moe_grouped=grouped["mean_APE"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
