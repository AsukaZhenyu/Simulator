"""Evidence-bound repair report; never fits parameters or executes hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def table(headers, rows):
    return ("<table><thead><tr>" + "".join("<th>" + escape(str(x)) + "</th>" for x in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row)
                                             + "</tr>" for row in rows) + "</tbody></table>")


def write(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def build(out, moe_summary):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("New report directory required; historical evidence is not overwritten")
    paths = {
        "baseline": ROOT / "build/accuracy-repair-20261027/BASELINE_AUDIT_profile.json",
        "h2d_stages": ROOT / "build/dynamic-diagnosis-20261027/h2d-v3-round1/stage-analysis.json",
        "h2d_states": ROOT / "build/dynamic-diagnosis-20261027/h2d-v3-round1/state-stage-diagnosis.json",
        "h2d_crossover": ROOT / "build/dynamic-diagnosis-20261027/h2d-v4-crossover/crossover-analysis.json",
        "link_profile": ROOT / "build/link-calibration-20261027/campaign-v1/profile-summary.json",
        "link_diagnosis": ROOT / "build/link-calibration-20261027/campaign-v1/profile-only-diagnosis-v1/DIAGNOSIS.json",
        "external_activity": ROOT / "build/dynamic-diagnosis-20261027/external-v5/DIAGNOSTIC_DIGEST.json",
        "external_callbacks": ROOT / "build/dynamic-diagnosis-20261027/external-v6/DIAGNOSTIC_DIGEST.json",
        "tiny_moe": ROOT / "build/real-moe-validation-20261027/consumer-contract-v1/RESULT.json",
        "real_moe": Path(moe_summary).resolve(),
        "memory": ROOT / "build/accuracy-repair-20261027/memory-reclaim/MEMORY_NOW.json",
        "office": ROOT / "build/accuracy-repair-20261027/memory-reclaim/office-residual-close.json",
        "background": ROOT / "build/accuracy-repair-20261027/memory-reclaim/authorized-background-close.json",
    }
    data = {key: read(path) for key, path in paths.items()}
    data.update(schema="accuracy-repair-report-1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                evidence={key: receipt(path) for key, path in paths.items()},
                report_scope="measurement contract repair and diagnostic evidence, not new whole-model accuracy qualification",
                historical_results_changed=False, new_whole_model_accuracy_claim=False)
    activity_rows = []
    for process in data["external_activity"]["processes"]:
        if "first_joint_physical" not in process:
            continue
        p = process["first_joint_physical"]
        activity_rows.append((process["init_policy"],
                              f'{process["first_joint_longest"]["host_wall_ns"] / 1e9:.4f} s',
                              f'{p["kernel_memcpy_combined_union_ns"] / 1e6:.3f} ms',
                              f'{p["host_extra_sync_ns"] / 1e3:.1f} µs',
                              "205 kernels + 129 D2D copies"))
    baseline_rows = [(p["job_id"], f'{p["early_suffix_mean_ns"] / 1e3:.2f}',
                      f'{p["late_suffix_mean_ns"] / 1e3:.2f}',
                      f'{p["late_over_early_minus_one"] * 100:.1f}%',
                      "支持" if p["fixed_baseline_support"] else "不支持")
                     for p in data["baseline"]["records"]]
    coverage_rows = [(str(cell["state_key"]), f'{cell["bytes"] / 1024:g} KiB',
                      f'{cell["independent_contexts"]}/3')
                     for cell in data["link_profile"]["exact_state_coverage"]["cells"]]
    callback_rows = []
    for process in data["external_callbacks"]["processes"]:
        longest = process["longest_attention"]
        callback_rows.append(("同核 CPU2" if process["cpu_actor_mask"] == 4 else "异物理核 CPU4",
                              "无回调控制" if process["role"] == "control" else "回调观察",
                              f'{process["target_wall_ns"] / 1e9:.3f}',
                              f'{longest["host_wall_ns"] / 1e6:.2f}',
                              "未采集" if longest["selected_API_union_ns"] is None
                              else f'{longest["selected_API_union_ns"] / 1e6:.2f}'))
    neutrality_rows = [("同核 CPU2" if p["cpu_actor_mask"] == 4 else "异物理核 CPU4",
                        f'{p["total_wall_relative_difference"] * 100:.2f}%',
                        f'{p["complete_off_state_attention_mean_relative_difference"] * 100:.2f}%',
                        "仅本轮观测对通过" if p["qualified"] else "未通过，保留数据")
                       for p in data["external_callbacks"]["observer_pairs"]]
    moe_rows = []
    for p in data["real_moe"]["placements"]:
        routes = p["routes"]
        moe_rows.append((p["placement"], p["normal_DLL_equivalence"]["status"],
                        f'{p["capture_full_logit_hash_matches"]}/{p["capture_full_logit_hash_denominator"]}',
                        f'{p["capture_graph_fingerprint_matches"]}/{p["capture_graph_forward_denominator"]}',
                        f'{routes["observed_by_phase"]["prefill"]}/{routes["expected_by_phase"]["prefill"]}',
                        f'{routes["observed_by_phase"]["decode"]}/{routes["expected_by_phase"]["decode"]}',
                        p["strict_capture_pair_status"]))
    content = ["<h1>准确性修复：测量边界、H2D 状态与真实 MoE 路由</h1>",
        "<p class='lead'>本轮将问题从“平均误差/CV”继续拆到具体执行阶段。已经修复了测量契约和不受支持的参数拟合入口；"
        "新的诊断数据没有被直接冒充为已通过独立验收的成本参数。历史准确性结论与失败分母保持不变。</p>",
        "<h2>1. 修复了哪些代码</h2>",
        table(["位置", "修改", "意义"], [
            ("profiling / baseline_audit + idle_fit", "先检查同一基线是否成立，再允许拟合；保留全部慢样本", "避免把持续漂移的基线压成一个常数"),
            ("profiling / consumer tap v4", "在 CPU 实际消费 MUL_MAT_ID 的位置读专家 IDs，不新增图节点、同步或复制", "消除旧路由观察可能改变执行图的干扰"),
            ("profiling / H2D v3/v4 + link_calibration", "分离 API 与设备活动、记录实测 PCIe 状态，增加多尺寸诊断", "区分传输、提交、同步与状态缺失"),
            ("profiling / external v5/v6", "将 Attention 拆到具体 runtime API；新回调在目标外写 JSON", "避免把秒级主机响应误判为 GPU 算术变慢"),
            ("modeling / pcie_response", "按实测代际/宽度选择既有 TransferResponse 参数，检查因果性与适用范围", "保持原计算公式；未知状态拒绝预测，不套理论速率"),
        ]),
        "<p>CPU/GPU 的统一服务率框架仍在 modeling；测量、分项统计与候选参数估计在 profiling。"
        "API enter-to-exit 是含等待的响应时间，不是 CPU 真正运行时间；它也不能与内部 GPU 耗时重复相加。</p>",
        "<p>计算过程仍是：依赖满足后在时间 s 启动作业，不断累计服务量，直到 "
        "<code>∫[s,e] r(t) dt ≥ W</code>。其中 <code>r(t) = g(当前可用资源、竞争、历史)</code>。"
        "本轮修正重点是确认 g 缺少哪些条件，以及 W/速率参数的测量边界；"
        "不是给完整 LLM 总时间再乘一个拟合系数。对 PCIe 的新增机制是实测状态选参，"
        "尚未识别的自然状态演化与 burst 内变化仍报告未知。</p>",
        "<h2>2. H2D：为什么不能只改带宽或尾部同步</h2>",
        f'<p>v3 共 {data["h2d_stages"]["completed_processes"]}/{data["h2d_stages"]["declared_processes"]} 个进程完成。'
        "相关联的设备 copy 与主机 API 显示，变慢落在 copy/内部等待阶段；额外的尾部同步很短。"
        "宽亲和性初始化没有消除漂移。观测器与裸跑的差异未满足价格迁移门槛，所以这些活动时长只用于定位。</p>",
        "<p>v4 的两个进程虽均通过数值检查，但 1,536 次目标 copy 没有获得直接关联记录；"
        "这些设备活动时长保持 unsupported，没有按时间邻近关系补配。</p>",
        "<p>原校准数据在相同零空闲条件下也存在随运行历史改变的基线：</p>",
        table(["校准进程", "前半段 µs", "后半段 µs", "变化", "单一基线"], baseline_rows),
        "<p>多尺寸新数据中，即使前后读数同为 Gen4×8，2 MiB 的后 15 次 copy 平均值仍出现 "
        "170.73–549.36 µs 的差异。因此 PCIe 状态有解释价值，但当前可观测状态还不足。"
        "前后状态相同也不能证明 burst 内从未切换。查询引入约百微秒的实际间隙，不能与无查询裸跑视为同一零空闲条件。</p>",
        table(["实际状态 key (current/max generation, current/max width)", "尺寸", "独立进程覆盖"], coverage_rows),
        "<p>三参数 PCIe 候选的留一校准进程诊断 MAPE 为 28.72%，P90 APE 为 48.39%；"
        "它不是独立 heldout 验收结果。Gen3/128 KiB 覆盖不足也保留为失败，未追加采样凑通过，未冻结或导出该候选。</p>",
        "<h2>3. 外部竞争：秒级 Attention 响应主要花在哪里</h2>",
        table(["初始化条件", "首个长 Attention 主机响应", "kernel + D2D 活动 union", "额外同步", "逐项关联"], activity_rows),
        "<p>本协议显式关联的 kernel/D2D 活动 union 很短，秒级主机响应主要落在大量 launch/copy API 返回及间隙。"
        "不能把这段墙钟整体等同于 GPU 执行时长，也不能仅靠 API 墙钟区分线程 Ready、阻塞等待和驱动内部行为。"
        "活动观测器改变了时间线，因此新 v6 改用目标线程回调，并把记录写出放到目标和背景任务结束之后。</p>",
        table(["CPU 背景任务位置", "测量方式", "整个目标 s", "最长 Attention ms", "该调用所选 API union ms"], callback_rows),
        "<p>将相同 CPU 背景任务从提交线程所在 CPU2 移到另一物理核 CPU4 后，秒级等待在本轮数据中消失，"
        "最长 Attention 变为约 22–23 ms。这一干预提示，全局平均利用率可能丢失核重叠信息；"
        "本轮没有控制或证明两个条件的全局利用率相等。接下来服务函数需要考察提交线程所在资源的可用性，以及背景任务是否与它共享核。"
        "现有命名资源框架可承载该输入，这不等于相关参数已经校准完成，也不需要在模拟器内重做操作系统任务调度。</p>",
        table(["观测对", "整个目标差异（门槛 5%）", "完整无 actor 状态 Attention 均值差异（门槛 5%）", "结果"], neutrality_rows),
        "<p>同核首个控制可能与一次诊断统计重叠，其准确起止时间和 BLAS 线程数未知；已明确标注，"
        "全部原始数据保留，未重跑或移除。异核对的有限观测器检查通过也不自动授予可复用价格资格。</p>",
        "<details><summary>v6 完整诊断摘要（含观测器资格与可能的背景分析重叠）</summary><pre>"
        + escape(json.dumps(data["external_callbacks"], ensure_ascii=False, indent=2)) + "</pre></details>",
        "<h2>4. 真实 MoE：先保证测到的是同一个执行过程</h2>",
        f'<p>小图契约 {data["tiny_moe"]["native_cases_completed"]}/4 完成；'
        "16 层、真实 ggml MUL_MAT_ID、带步长的专家 IDs，在 hook 开/关时输出保持一致。"
        "这验证采集机制，不代替真实量化模型验收。</p>",
        f'<p>真实 OLMoE 新采集 {data["real_moe"]["jobs_completed"]}/{data["real_moe"]["jobs_declared"]} 个进程完成，'
        f'{data["real_moe"]["jobs_failed"]} 个执行失败，{data["real_moe"]["jobs_not_run"]} 个未执行。'
        "本次是两条旧校准请求（短/长各一条）、每条最多 8 个输出 token 的 smoke；"
        "不是旧 12 条 heldout 的完整重复，也没有覆盖旧采集中更后面的分歧位置。</p>",
        table(["放置", "原/私有 DLL 正常执行", "完整 logits hash", "图分区指纹", "Prefill 路由层调用", "Decode 路由层调用", "严格配对"], moe_rows),
        "<p>两个放置的数值与图分区一致。CPU 路由完整；混合模式 Decode 也完整，"
        "但 75 个 Prefill 层调用、合计 225 个专家算子实际位于 CUDA0，超出 CPU 消费点观察范围。"
        "这项失败属于采集覆盖，不是已观测到的数值不一致；没有拿 CPU 上的权重驻留位置推断算子的实际执行设备。</p>",
        "<details><summary>真实 OLMoE 完整采集与逐层证据</summary><pre>"
        + escape(json.dumps(data["real_moe"], ensure_ascii=False, indent=2)) + "</pre></details>",
        "<p>真实路由、正常/固定 token 执行、图分区与完整 logits 的一致性，和完整模型性能误差是不同的验收。"
        "只有观察契约通过后才能补齐真实算子成本；不使用 F32 小图成本冒充 Q4/Q6 价格，也不把带观察的时间当作裸跑真值。</p>",
        "<h2>5. 本轮结论与仍缺的证据</h2>",
        table(["问题", "当前结论", "下一步证据"], [
            ("旧 H2D 单基线恢复模型", "校准数据不支持；入口已加拒绝门槛", "能解释同状态下历史漂移的新状态/分项参数与独立数据"),
            ("仅按 PCIe 代际修正", "有帮助但不足，覆盖和诊断误差未过", "保持测量契约一致的更完整状态，不能用理论比值补格"),
            ("外部竞争下整块 Attention 定价", "原宏任务跨越多个外部状态，不足以识别联合状态价格", "可靠的提交/等待阶段响应及跨状态组合验收"),
            ("MoE 路由观察", "以真实量化模型新配对验收为准，见上表", "观察通过后补真实 Attention、专家与模型其余开销"),
            ("CV 来源", "不能下定论为不可控随机性", "区分真实执行波动、观测扰动与预测残差"),
        ]),
        "<p>本报告没有新增“外部状态准确性已通过”或“完整真实 MoE 性能已通过”的声明。"
        "测量完成、观测器中性、状态覆盖、模型准确性四个结果分别报告。</p>",
        "<h2>6. 内存释放与可复核证据</h2>",
        f'<p>用户授权关闭无关应用并确认 Office 无未保存文档。Office 全部退出；随后一次内存观察为 '
        f'{data["memory"]["available_GiB"]:.2f} GiB，原 MoE 门槛约 5.66 GiB。'
        "Codex、任务进程、网络与系统服务保留。实际每次 MoE 启动仍重新检查容量，不降低门槛。</p>",
        "<p>全部输入收据见 <a href='REPORT_DATA.json'>REPORT_DATA.json</a>；文件完整性见 "
        "<a href='REPORT_MANIFEST.json'>REPORT_MANIFEST.json</a>。</p>"]
    out.mkdir(parents=True)
    write(out / "REPORT_DATA.json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    style = """body{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.7 'Times New Roman','Microsoft YaHei',serif;color:#14253e;background:#fafbfd}h1{font-size:30px}h2{font-size:23px;margin-top:32px;border-bottom:2px solid #28538c}table{border-collapse:collapse;width:100%;font-size:15px;background:white}th,td{border:1px solid #ccd5e0;padding:9px;text-align:left;vertical-align:top}th{background:#eaf0f8}.lead{padding:16px;border:2px solid #28538c;background:white}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:520px;overflow:auto}details{margin:20px 0;background:white;padding:12px}a{color:#28538c}"""
    write(out / "REPORT_zh.html", "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
          + "<title>Accuracy Repair — 27 October 2026</title><style>" + style + "</style><body>" + "\n".join(content) + "</body></html>")
    write(out / "REPORT_MANIFEST.json", json.dumps(dict(schema="accuracy-repair-report-manifest-1", generator=receipt(__file__),
          inputs=data["evidence"], outputs=[receipt(out / n) for n in ("REPORT_DATA.json", "REPORT_zh.html")]), ensure_ascii=False, indent=2) + "\n")
    return dict(report=str(out / "REPORT_zh.html"), manifest=str(out / "REPORT_MANIFEST.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--moe-summary", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.moe_summary), ensure_ascii=False))
