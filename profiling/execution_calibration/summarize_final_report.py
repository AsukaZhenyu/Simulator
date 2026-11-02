"""Create the Chinese experiment report from completed, separately verified runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary(rows):
    values = [row["absolute_relative_error"] for row in rows if row["observed_ns"] is not None]
    return dict(predictions=len(rows), available=len(values), median_relative_error=statistics.median(values),
        max_relative_error=max(values), errors_at_most_20pct=sum(value <= .2 for value in values),
        accepted=sum(row["calibrated"] for row in rows),
        gate_failures=dict(Counter(key for row in rows for key, passed in row.get("gates", {}).items() if not passed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed experiment-day directory")
    parser.add_argument("--variant", default="transforms-context-01")
    args = parser.parse_args()
    folder = args.root.resolve()
    native = read(folder / args.variant / "analysis/report.json")
    original = read(folder / "transforms-01/analysis/report.json")
    baseline = read(folder / "baseline-component-analysis-final/report.json")
    baseline_raw = read(folder / "baseline-01/report.json")
    primary = [row for row in native["results"] if "@" not in row["strategy"]]
    dispatch = [row for row in native["results"] if "@" in row["strategy"]]
    verification = read(folder / args.variant / "verification.json")
    assert verification["verified"]
    primary_stats, dispatch_stats = summary(primary), summary(dispatch)
    baseline_stats = summary(baseline["results"])
    replay = read(folder / "baseline-modeling-replay/report.json")
    experiments=tuple(dict.fromkeys(("baseline-01", "transforms-01", "transforms-dual-01", args.variant)))
    manifests = [read(folder / name / "manifest.json") for name in experiments]
    assert all(manifest["assets_unchanged"] for manifest in manifests)
    rows = [read(folder / name / "measurements.json") for name in experiments]
    sample_counts = [sum(len(values) for row in dataset if row["status"] == "measured" for values in row["metrics"].values()) for dataset in rows]
    brackets=[]
    for row in rows[-1]:
        if row["case"]["family"]=="ffn" and row["status"]=="measured":
            payload=json.loads(read(row["raw_path"])["stdout"])
            brackets.append(payload.get("diagnostic_completed_work_brackets",{}).get("times_ns",{}))
    overlapping=sum(item.get("completed_work_bracket_overlap_ns",0)>0 for item in brackets)
    total_processes = sum(len(dataset) for dataset in rows)
    ideal_rankings = [r for r in native["rankings"] if r["boundary_contract"].endswith(";ideal_model")]
    counts = dict(native_processes=total_processes, formal_sample_count=sum(sample_counts),
        baseline_processes=len(rows[0]), transform_processes=sum(len(dataset) for dataset in rows[1:]), primary=primary_stats,
        dispatch_model=dispatch_stats, baseline=baseline_stats,
        primary_ranking_pairs=sum(r["pair_count"] for r in ideal_rankings),
        primary_ranking_correct_pairs=sum(r["correct_pairs"] for r in ideal_rankings),
        primary_ranking_decisive_pairs=sum(r["decisive_pairs"] for r in ideal_rankings),
        primary_ranking_correct_decisive_pairs=sum(r["correct_decisive_pairs"] for r in ideal_rankings),
        accepted_rankings=sum(r["latency_quality_accepted"] for r in ideal_rankings),
        ffn_host_diagnostics=len(brackets), ffn_overlapping_host_brackets=overlapping,
        gpu_workspace_known=False, full_M2_compiler_calibrated=False)
    lines = ["# 执行变换性能校准实验报告", "", "日期：2026-10-17。设备：AMD Ryzen 9 7940HX、RTX 4070 Laptop GPU（8188 MiB）、Windows 11、CUDA 12.6 / NVIDIA 581.57。", "",
        "## 先看结论", "",
        f"本轮完成 **{total_processes} 个正式原生测量进程、{sum(sample_counts):,} 条原始计时样本**，另保留数值预检和失败记录。新 FFN 分片及 Attention 异构执行已经有真实测量入口、独立组件预测和新进程验证；**当前仍不能宣布整个模拟器完成性能校准**。", "",
        f"- 新执行路径有 **{len(primary)} 个主要预测**，中位相对误差 **{primary_stats['median_relative_error']:.2%}**，最大 **{primary_stats['max_relative_error']:.2%}**；{primary_stats['errors_at_most_20pct']}/{len(primary)} 个点误差不超过 20%。",
        f"- 同时要求数值、独立进程覆盖、进程内/间稳定性和误差全部通过时，主要预测 **{primary_stats['accepted']}/{len(primary)}** 通过。点误差与正式接受分开解释。",
        f"- 同条件 resident 策略的中位数排序 **{counts['primary_ranking_correct_pairs']}/{counts['primary_ranking_pairs']}** 对；95% 区间不重叠的排序 **{counts['primary_ranking_correct_decisive_pairs']}/{counts['primary_ranking_decisive_pairs']}** 对。通过全部延迟质量门的比较组为 {counts['accepted_rankings']}/{len(ideal_rankings)}，其余排序用于诊断。",
        "- 数值预检发现 F32 存储与默认计算精度不能混同。明确 F32/TF32 合同后，大 token case 的数值检查在原容差下恢复通过，详见精度诊断。", "",
        "## 1. 实验回答什么问题", "",
        "问题一：孤立阶段的实测成本，能否预测完整执行路径？问题二：CPU/GPU 分片能否按理想 max 重叠计算？问题三：单独计费的权重加载能否与计算成本组合？问题四：预测能否正确选择同一初始/最终条件下的方案？", "",
        "本轮执行边界是 F32 compact FFN core（gate/up/SwiGLU/down、分片和 SUM）以及 post-RoPE Attention core/output。使用生成的输入和权重，尺寸为小型机制实验，未加载预训练模型，也未覆盖量化权重。计时来自真实硬件执行，不是合成耗时。Dense norm/FFN residual、MoE router/gather/scatter/combine、完整模型、多层 P/D 持续实机链路属于后续单独校准项。", "",
        "## 2. 方法与防止自证", "",
        "- 旧基线与单逻辑处理器诊断使用 profile seed=17、heldout seed=29；双逻辑处理器对照使用 profile seed=41、heldout seed=53；最终相同 worker 复测使用 profile seed=67、heldout seed=79。每种形状、每个 split 各启动两次独立原生进程。随机样本不是独立进程的替代品。",
        "- 下文主要完整路径预测仅使用 profile 的独立组件中位数；完整路径和 parallel_branches 的实测总时间不用于拟合自身成本。heldout 启动前将模型和 SHA256 冻结。",
        "- 新入口每指标 51 样本、预热至少 20 次且不少于 200 ms；旧基线每计算指标 51 样本、至少 500 ms，搬运 101 样本。",
        "- 预声明质量门：进程内 CV≤15%、跨进程中位数 CV≤20%、heldout 相对误差≤20%。所有原始样本保留，失败不移除，不根据结果改门。",
        "- CPU 计算单线程绑定逻辑处理器0。第一轮GPU派发worker也只能使用0；补充对照为其显式绑定逻辑处理器2，并回读实际mask。CUDA graphs关闭。桌面应用、功耗和时钟策略保持原样，环境与子进程flags被记录。",
        "- 新入口固定 GGML_PREC_F32 并设置 NVIDIA_TF32_OVERRIDE=0；旧基线使用旧精度合同。因此两批的时间差不解释为策略加速。",
        "- resident 比较均以不可变输入在 CPU/GPU 已驻留、输出最终交付 DRAM 为条件。preload 每次实际重载 GPU shard 三份权重，单独分组，不与免费驻留混排。", "",
        "## 3. 测试范围", "",
        "| 分组 | 几何与路径 | 独立进程 |", "|---|---|---:|",
        "| 旧基线 | 7 个 FFN/Attention case；17 个 CE/SM/duplex/control case | 96 |",
        "| 单逻辑CPU诊断 | 同下方7个新case；发现双线程分支未出现host完成区间重叠 | 28 |",
        "| 双逻辑CPU对照 | 新 FFN 5case、Attention 2case；GPU孤立stage在main0、分片执行在worker2 | 28 |",
        *(["| 相同worker采样复测 | GPU shard孤立stage改在worker2，与实际分片相同；新profile67/heldout79 | 28 |"] if "context" in args.variant else []), "",
        "新 FFN：H256/I512、q1/q3、CPU切点128/256；H512/I1024、q32、切点256。新 Attention：H256、8 Q heads/2 KV heads、head_dim32，q32/hist0与q1/hist128。",
        "FFN 实测 GPU 完整链、双线程提交的两分片链及显式预载链。Attention 实测 GPU 全链和 CPU QK/softmax→概率 H2D→GPU PV/output 链。下方主要结果采用当前最后一轮，前面各轮诊断保持独立证据。", "",
        "## 4. 主要预测结果", "",
        "| case | 路径 | 预测 μs | heldout μs | 相对误差 | 未通过的门 |", "|---|---|---:|---:|---:|---|"]
    for row in primary:
        failed = ", ".join(k for k, v in row.get("gates", {}).items() if not v)
        if row["observed_ns"] is None:
            lines.append(f"| {row['case_id']} | {row['strategy']} | unknown | unknown | unknown | {row['unknown_reason']} |")
        else:
            lines.append(f"| {row['case_id']} | {row['strategy']} | {row['predicted_ns']/1000:.2f} | {row['observed_ns']/1000:.2f} | {row['absolute_relative_error']:.2%} | {'通过' if row['calibrated'] else failed} |")
    lines += ["", "![预测与独立实测](figures/prediction_error.png)", "",
        "## 5. 机制模型与误差定位", "",
        "FFN 主模型为 `max(CPU 分片四阶段之和, GPU 分片四阶段之和) + 输出D2H + CPU SUM`。另在实验前声明 `主模型 + 单独测得的空 worker 握手`，评估派发/同步是否是缺失项。握手并非从验证误差拟合的截距，也不能自动代表全部软件开销。", "",
        "| FFN case | 理想max误差 | 加独立握手误差 | 两者的实际heldout μs |", "|---|---:|---:|---:|"]
    extra = {(row["case_id"], row["strategy"]): row for row in dispatch}
    for row in primary:
        if row["strategy"] == "hybrid/resident" and row["case_id"].startswith("ffn-"):
            other = extra[row["case_id"], "hybrid/resident@dispatch_model"]
            lines.append(f"| {row['case_id']} | {row['absolute_relative_error']:.2%} | {other['absolute_relative_error']:.2%} | {row['observed_ns']/1000:.2f} |")
    original_stats=summary([row for row in original["results"] if "@" not in row["strategy"]])
    dual=read(folder / "transforms-dual-01/analysis/report.json")
    dual_stats=summary([row for row in dual["results"] if "@" not in row["strategy"]])
    def ranking_count(report):
        values = [r for r in report["rankings"] if r["boundary_contract"].endswith(";ideal_model")]
        return f"{sum(r['correct_pairs'] for r in values)}/{sum(r['pair_count'] for r in values)}"
    lines += ["", "![预声明派发模型](figures/dispatch_model.png)", "",
        "| 轮次 | 主模型中位误差 | 最大误差 | resident 中位数排序正确 | 全质量通过 |",
        "|---|---:|---:|---:|---:|",
        f"| 单逻辑处理器 | {original_stats['median_relative_error']:.2%} | {original_stats['max_relative_error']:.2%} | {ranking_count(original)} | {original_stats['accepted']}/19 |",
        f"| 主线程0、派发线程2 | {dual_stats['median_relative_error']:.2%} | {dual_stats['max_relative_error']:.2%} | {ranking_count(dual)} | {dual_stats['accepted']}/19 |",
        f"| 相同 worker 采样（最终） | {primary_stats['median_relative_error']:.2%} | {primary_stats['max_relative_error']:.2%} | {ranking_count(native)} | {primary_stats['accepted']}/19 |", "",
        "三轮的采样种子和运行时间不同，并非配对的单因素消融。因此最终轮排序正确，只说明这一组实测中的排序结果；不能将跨轮差异全部归因于绑核或采样上下文，也不能宣称准确度随修改单调提高。", "",
        f"第一轮单逻辑CPU诊断：主模型中位误差{original_stats['median_relative_error']:.2%}，最大{original_stats['max_relative_error']:.2%}，{original_stats['errors_at_most_20pct']}/19点误差≤20%；20次FFN未计时诊断均未观察到host完成区间重叠。它反映当前probe提交配置，不能泛称硬件无法并行或模拟器并发模型必然错误。因此保留原结果，并以新profile/heldout seed补充主线程0、GPU派发线程2的对照。",
        f"双逻辑CPU轮的主模型中位误差{dual_stats['median_relative_error']:.2%}，最大{dual_stats['max_relative_error']:.2%}。其GPU独立stage在main0测量、实际GPU分片在worker2执行，两者wall时间的提交上下文不同。" + ("当前最后一轮把GPU shard的独立计时移至同一worker，外部唤醒/等待不纳入stage样本，另保留独立握手模型；使用新profile/heldout种子验证。" if "context" in args.variant else "这一上下文错配需要单独复测，不能靠事后比例拟合掩盖。"),
        f"最终轮中，{overlapping}/{len(brackets)} 次FFN未计时诊断观察到host完成区间重叠。host完成区间不是CUDA kernel trace，不能据此计算kernel重叠率或声称任意硬件都能完美重叠。误差可能同时包含阶段分组、缓存、线程唤醒、派发/等待与共享资源影响；本轮没有将某个来源确定为唯一原因。", "",
        "当前更集中的模型问题是混合 FFN 仍被低估：四个小 FFN 的 resident 路径误差为31.65%～44.60%，较大 q32 路径为20.63%。对齐 worker 后仍存在该残差，不能继续将已修复的线程上下文错配作为当前解释；增加一个空握手成本也未充分解决。", "",
        f"主要预测的质量失败计数（同一项可重叠）：`{json.dumps(primary_stats['gate_failures'], ensure_ascii=False)}`。",
        "`within_process_cv` 表示同一进程的逐次耗时波动；`between_process_cv` 表示不同进程的中位耗时漂移；`heldout_relative_error` 表示预测与独立实测中位数的偏差。CV 为标准差除以均值。全部19项稳定性失败，说明这些成本暂时不能作为稳定的正式价格；它不等于19项点预测都错误，也不能据此认定误差全由后台应用造成。每个 split 仅两个独立进程，bootstrap 区间和跨进程漂移仍受重复数量限制。",
        "![完整样本稳定性](figures/measurement_stability.png)", "",
        "## 6. 旧基线与 Modeling 回放", "",
        f"旧基线 96/96 进程数值/字节检查通过，99 个不同时间指标中 {sum(row['calibrated'] for row in baseline_raw['results'])}/99 通过全部门。真正用独立组件预测组合路径时：19 个预测的中位误差 {baseline_stats['median_relative_error']:.2%}，最大 {baseline_stats['max_relative_error']:.2%}，{baseline_stats['errors_at_most_20pct']}/19 个点误差≤20%，但严格质量门 {baseline_stats['accepted']}/19 通过。旧路径初始驻留和输出边界不同，最终分析禁用跨路径公平排名。", "",
        "旧实测 CPU 端口侧表实际经过 Modeling 的动作、依赖和工作区账本：5 个 CPU 图 valid；5 个 GPU 图因私有 workspace 未知返回 unsupported。CPU 图仍为诊断用途，其成本的稳定性门未通过。新异构实验目前是原生执行＋组件模型验证，尚未完成新 Python→IR→Compiler→Modeling 全链的实测绑定。", "",
        "## 7. 数值精度修复", "",
        "H512/I1024/q32/cut256、seed101 的第一次预检中，GPU gate/up 有4项检查失败，最大绝对误差约 3.7516e-5。新 probe 同时显式设置 GGML_PREC_F32 与关闭 TF32 后，同 case 15项检查全部通过，最大绝对误差约 2.6077e-8；数值容差始终为 atol=1e-5、rtol=1e-4。", "",
        "这说明 profile 签名还必须包含实际计算精度，不能只记录存储 dtype。两项设置同时改变且程序重编译，未做2×2消融，因此不能证明 TF32 为唯一原因。旧预检二进制哈希未保存、TF32 flag未记录的局限已写入[精度诊断](precision-diagnostic.md)，失败 raw 保留。", "",
        "## 8. 当前能用到什么程度", "",
        "| 层次 | 本轮结论 |", "|---|---|",
        "| 原生执行和数值 | 新 FFN/Attention 具体实现已真实执行；各端口/输出/KV检查有原始证据 |",
        f"| 独立组件预测 | 19 个主要预测中 {primary_stats['accepted']} 个通过全部预声明门，其余保持诊断标记 |",
        "| 单设备 Modeling 侧表回放 | 旧 CPU 端口图5例 valid，质量仍未接受；GPU 容量明确拒绝 |",
        "| 新执行变换的实测 Compiler 绑定 | 待补 measured provenance、精确lowering签名与未知workspace状态 |",
        "| 完整Dense/MoE、多层P/D、预取/计算并发 | 本轮未验收 |",
        "| GPU实际容量 | 私有workspace未知，未验收 |", "",
        "实测数据作为独立侧表保存；原合成夹具继续承担逻辑测试用途。不能把未知 workspace 填0后声称容量正确，也不能把质量失败的数据发布成正式成本表。", "",
        "## 9. 接下来的具体工作", "",
        "1. 将实际计算精度、lowering、输入/输出驻留、CPU线程和GPU派发方式纳入性能签名；先对齐一个Dense单层的真实执行合同。",
        "2. 分开采集GPU device-event时间与host派发/等待，验证软件控制开销模型；对很短的操作增加明确的多次重复计时口径，再用独立进程验证，保留逐次与批次口径区别。",
        "3. 补Dense projection/norm/residual及新IR实测侧表绑定；继续保持GPU未知workspace可表达，延迟与容量验收分离。",
        "4. 再验证完整Prefill→Decode、真实MoE路由外围和预取/计算并发，最后开展策略收益比较。", "",
        "## 10. 证据与复用", "",
        f"- [新完整分析]({args.variant}/analysis/REPORT.md)、[JSON]({args.variant}/analysis/report.json)、[逐条样本CSV]({args.variant}/analysis/raw-timing-samples.csv)",
        f"- [新实验离线核验]({args.variant}/verification.json)、[第一轮诊断](transforms-01/analysis/REPORT.md)、[旧组件对照](baseline-component-analysis-final/REPORT.md)、[Modeling回放](baseline-modeling-replay/report.json)",
        "- [最终轮独立摘要](summary-context-audit.md)、[双逻辑处理器轮摘要](summary-dual-audit.md)、[首轮摘要](summary-audit.md)",
        "- 43项非硬件工具/契约测试通过：[38项测试](tool-tests-final.log)及[5项context核验测试](context-wrapper-tests.log)。它们验证分析与证据合同，不代替性能验收。",
        "- 构建记录：[首轮](execution_transforms-build.json)、[双逻辑处理器](execution_transforms_dual-build.json)、[同worker采样](execution_transforms_context-build.json)。",
        "- [精度诊断](precision-diagnostic.md)、[测量程序与命令](../../execution_calibration/README.md)、[严格合同](../../EXECUTION_CALIBRATION_CONTRACT.md)",
        "- raw命令/stdout/stderr、子进程PID、源快照、程序与运行库前后SHA、冻结模型均位于各实验目录。旧实验与失败记录保留。", ""]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (folder / "report-summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
