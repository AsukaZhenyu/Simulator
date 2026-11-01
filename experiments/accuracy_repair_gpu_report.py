"""Report a versioned GPU route-observation repair without rewriting v4 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

from experiments.accuracy_repair_report import ROOT, read, receipt, table, write


def build(out, summary):
    out, summary = Path(out).resolve(), Path(summary).resolve()
    if out.exists():
        raise FileExistsError("New report directory required")
    old_path = ROOT / "build/real-moe-validation-20261027/tap-smoke-v4b-memory-recovered-v1/SMOKE_ANALYSIS.json"
    callbacks_path = ROOT / "build/dynamic-diagnosis-20261027/external-v6/DIAGNOSTIC_DIGEST.json"
    prior = ROOT.parent / "paper-write/第八教学周/experiment-reports/accuracy-repair-20261027/REPORT_MANIFEST.json"
    current, old, callbacks = read(summary), read(old_path), read(callbacks_path)
    if current["conclusion"]["whole_model_performance_accuracy_evaluated"] is not False:
        raise ValueError("This report only covers route observation, not whole-model accuracy")
    before = {p["placement"]: p for p in old["placements"]}
    rows = []
    for p in current["placements"]:
        routes, previous = p["routes"], before[p["placement"]]["routes"]
        rows.append((p["placement"], p["normal_DLL_equivalence"]["status"],
                     f'{p["capture_full_logit_hash_matches"]}/{p["capture_full_logit_hash_denominator"]}',
                     f'{p["capture_graph_fingerprint_matches"]}/{p["capture_graph_forward_denominator"]}',
                     f'{previous["observed_by_phase"]["prefill"]}/{previous["expected_by_phase"]["prefill"]} → '
                     f'{routes["observed_by_phase"]["prefill"]}/{routes["expected_by_phase"]["prefill"]}',
                     f'{routes["observed_by_phase"]["decode"]}/{routes["expected_by_phase"]["decode"]}',
                     f'{routes["observed_three_stage_events"]}/{routes["expected_three_stage_events"]}',
                     p["strict_capture_pair_status"]))
    callback_rows = [("同核 CPU2" if p["cpu_actor_mask"] == 4 else "异物理核 CPU4",
                      "无回调" if p["role"] == "control" else "有回调",
                      f'{p["target_wall_ns"] / 1e9:.3f}',
                      f'{p["longest_attention"]["host_wall_ns"] / 1e6:.2f}')
                     for p in callbacks["processes"]]
    evidence = {"current_moe": receipt(summary), "previous_moe": receipt(old_path),
                "external_callbacks": receipt(callbacks_path), "prior_report_manifest": receipt(prior)}
    dataset = dict(schema="gpu-route-repair-report-1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   evidence=evidence, current_moe=current,
                   previous_moe_status=old["status"], external_callbacks=callbacks,
                   historical_results_changed=False, whole_model_accuracy_claimed=False)
    lead = (f'新采集 {current["jobs_completed"]}/{current["jobs_declared"]} 个进程完成，'
            f'{current["jobs_failed"]} 个执行失败，{current["jobs_not_run"]} 个未执行。'
            "下表分别报告数值、计算图和路由覆盖，不能将采集完成数当作性能准确性通过数。")
    capacity = current["capacity"]
    parts = ["<h1>准确性修复更新：GPU Prefill 专家路由</h1>", "<p class='lead'>" + escape(lead) + "</p>",
             "<h2>1. 采集点为什么需要修改</h2>",
             "<p>真实 OLMoE 在混合模式下，并不总在 CPU 执行专家。原 CPU 采集点已经完整捕获 Decode，"
             "但 Prefill 有 75 个层调用、225 个专家算子实际走 CUDA 量化 MMQ 路径，绕过 CPU 消费点。"
             "旧结果的数值与计算图一致，失败是观察覆盖不完整；旧失败记录完整保留。</p>",
             "<p>这里的 hybrid 是原运行配置的名称，不能自动理解成“所有阶段的专家算子都在 CPU”。"
             "后续建模和成本查表必须使用实际执行设备；GPU Prefill 的时间不能填进 CPU 算子成本格。"
             "本轮保留原运行时卸载行为来检查采集机制，严格 CPU 专家策略属于另一个明确的配置。</p>",
             "<p>这个具体案例也说明，策略描述需要分别记录“权重驻留在哪”和“算子实际在哪执行”，"
             "再由搬运与依赖连接二者。只用 CPU/GPU 权重比例给整个策略命名，会丢失这项行为。</p>",
             "<h2>2. 新版怎样采集</h2>",
             "<p>新增 GPU 消费点观察，沿真实 CUDA stream 将 IDs 异步保存到预分配的 host 槽位，"
             "在原有执行同步完成后解析。保留原数学图、后端选择和默认 CUDA graph 模式；"
             "不能观察到的 graph replay 或不支持的布局必须明确拒绝，不能复用上一次 IDs。</p>",
             "<p>观察新增的 copy 命令会占用资源，因此这些计时始终是带观察的诊断时间，"
             "不作为裸跑耗时或性能成本。配对的数值和图分区检查负责验证采集运行的数学过程。</p>",
             "<h2>3. 实际验收结果</h2>",
             table(["放置", "原/私有 DLL 正常执行", "完整 logits hash", "图分区", "Prefill 路由：原 → 新", "Decode 路由", "up/gate/down 阶段", "严格配对"], rows),
             "<p>本轮混合模式仍有严格数值失败：一条短请求的私有运行库 normal 执行出现不同的完整 logits hash，"
             "而原运行库 normal、私有 replay 与 capture 的相应结果一致。全部生成 token 和图分区仍一致。"
             "这不能直接证明异步采集 copy 引起差异；也不能因 token 相同就把严格失败改成通过。"
             "未保存完整浮点向量，因此全词表数值误差仍未知；已保存 top-logits 的差异仅是局部诊断。"
             "没有替换主参考、删去样本或放宽门槛。</p>",
             "<p>范围仍为真实 16 层 OLMoE、两条旧校准请求（短/长各一条）、每条最多 8 个输出 token。"
             "这不是旧 12 条 heldout 的完整重复，也没有证明更长生成长度、所有模型和所有 CUDA graph 路径均受支持。</p>",
             f'<p>原容量门槛保留；本轮监测到的最大 peak working set 为 '
             f'{capacity["maximum_observed_peak_wset_bytes"] / 2**30:.2f} GiB，'
             f'最小系统可用内存为 {capacity["minimum_observed_system_available_bytes"] / 2**30:.2f} GiB。</p>',
             "<details><summary>完整采集摘要、CUDA graph 与流覆盖证据</summary><pre>"
             + escape(json.dumps(current, ensure_ascii=False, indent=2)) + "</pre></details>",
             "<h2>4. 设备波动诊断保持原结论</h2>",
             table(["背景 CPU 任务", "观察模式", "整个目标 s", "最长 Attention ms"], callback_rows),
             "<p>核位置干预提示，应考察提交线程所在资源的可用性和核重叠；"
             "本轮没有控制或证明两条件全局利用率相同。同核首控制有潜在统计分析重叠，已保留标注；"
             "异核对仅通过本轮有限的观测器检查，不等于性能模型已通过独立验收。</p>",
             "<p>H2D 中 PCIe 代际有解释价值，但相同 Gen4×8 下仍存在较大差异。"
             "三参数候选仅在校准集内部做留一进程诊断，未冻结或导出，也没有运行该候选的独立 heldout。</p>",
             "<h2>5. 哪些事情仍未被本轮证明</h2>",
             "<p>本轮修复和验收的是路由采集机制。完整真实 MoE 的算子成本覆盖与端到端时间误差、"
             "设备竞争下提交/等待阶段的服务函数、H2D 相同可观测状态下的历史漂移，仍需各自的校准和独立数据。"
             "旧 Dense 准确性结果、旧严格验收失败分母均未被改写。</p>",
             "<p>详细 H2D/小图/旧采集诊断见 <a href='../REPORT_zh.html'>本轮基础诊断报告</a>。"
             "全部新输入收据见 <a href='REPORT_DATA.json'>REPORT_DATA.json</a>，"
             "输出完整性见 <a href='REPORT_MANIFEST.json'>REPORT_MANIFEST.json</a>。</p>"]
    style = "body{max-width:1100px;margin:32px auto;padding:0 24px;font:17px/1.7 'Times New Roman','Microsoft YaHei',serif;color:#14253e;background:#fafbfd}h1{font-size:30px}h2{font-size:23px;border-bottom:2px solid #28538c;margin-top:30px}table{width:100%;border-collapse:collapse;background:white;font-size:15px}th,td{padding:9px;border:1px solid #ccd5e0;vertical-align:top;text-align:left}th{background:#eaf0f8}.lead{border:2px solid #28538c;padding:15px;background:white}details{background:white;padding:12px}pre{white-space:pre-wrap;word-break:break-word;max-height:500px;overflow:auto;font-size:12px}a{color:#28538c}"
    out.mkdir(parents=True)
    write(out / "REPORT_DATA.json", json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write(out / "REPORT_zh.html", "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
          + "<title>GPU Route Observation Repair</title><style>" + style + "</style><body>" + "\n".join(parts) + "</body></html>")
    write(out / "REPORT_MANIFEST.json", json.dumps(dict(schema="gpu-route-repair-report-manifest-1", generator=receipt(__file__),
          helper=receipt(ROOT / "experiments/accuracy_repair_report.py"), inputs=evidence,
          outputs=[receipt(out / n) for n in ("REPORT_DATA.json", "REPORT_zh.html")]), ensure_ascii=False, indent=2) + "\n")
    return dict(report=str(out / "REPORT_zh.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.summary), ensure_ascii=False))
