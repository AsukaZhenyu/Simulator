"""Assemble a readable report from completed follow-up summaries and audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return "未获得" if value is None else f"{value * 100:.2f}%"


def build(base, external, figures, out, wsl=None, backend=None):
    base, external = base.resolve(), external.resolve()
    paths = dict(short=base / "task1-v1/summary.json", long=base / "task1-v2/summary.json",
                 multi=base / "multilayer-v1/campaign-v1/summary.json",
                 affinity=base / "task1-affinity-v1/summary.json", external=external / "summary.json")
    data = {name: read(path) for name, path in paths.items()}
    sources = []
    def bind(path, label):
        path = Path(path).resolve()
        sources.append(dict(path=str(path), sha256=sha(path), label=label))
    for name, path in paths.items():
        bind(path, name + " 原始汇总")
    for name in ("baseline.json", "followup-protocol.json", "audit-task1-v1.json", "audit-task1-v2.json", "audit-multilayer.json", "audit-affinity.json"):
        path = base / name
        if not read(path).get("passed", name in ("baseline.json", "followup-protocol.json")):
            raise ValueError("Failed required audit: " + str(path))
        bind(path, name)
    for name in ("independent-audit.json", "tests-receipt.json", "REPORT.md"):
        bind(external / name, "外部对照 " + name)
    baseline = read(base / "baseline.json")
    for row in baseline["old_frozen_outputs"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Historical frozen evidence changed")
    shift_path = base / "task1-window-shift-v1/summary.json"
    shift = read(shift_path)
    bind(shift_path, "短窗口与长窗口的描述性均值变化")
    bind(base / "task1-postcampaign-hardware.json", "实验后静态硬件身份核对")
    bind(base / "task1-postcampaign-scope-patch.json", "面向未来采集的上下文补丁")
    bind(base / "integration/profiling-tests-workspace-tmp.json", "593项profiling软件回归")
    v1, v2, multi, affinity, ext = (data[key] for key in ("short", "long", "multi", "affinity", "external"))
    transfer_rows = []
    for route, label in (("dram->vram", "H2D"), ("vram->dram", "D2H")):
        for result, window in ((v1, "8次重放"), (v2, "256次重放")):
            row = result["transfer"][route]
            transfer_rows.append([label, window, pct(row["maximum_heldout_size_mean_ape"]),
                                  f"{row['process_cv_passes']}/{row['processes']}", "通过" if row["qualified"] else "未通过"])
    observer = v1["observer"]
    real = [row for row in observer["rows"] if not row.get("sham") and row["status"] == "completed"]
    failures = [row for row in real if not row["passed"]]
    observer_fail = "；".join(row["program_key"] + "，差异" + pct(row["relative_difference"]) for row in failures) or "无超限进程"
    qualified = sum(row["qualified"] for row in v2["transfer"].values())
    conditions = []
    for row in affinity["conditions"]:
        small, large = row["masks"]["4"], row["masks"]["20"]
        conditions.append([row["route"], f"{row['bytes']:,}", f"{small['mean_ns']/1000:.3f}",
                           f"{large['mean_ns']/1000:.3f}", pct(row["aggregate_relative_effect"]),
                           f"{small['trial_cv_passes']}/4 → {large['trial_cv_passes']}/4"])
    stage1 = dict(title="传输响应与观测一致性", conclusion=f"新增可解释传输模型与独立测量流程；长窗口的两个方向有{qualified}/2通过。新同进程观测对照{observer['passed_processes']}/{observer['processes']}通过。",
        paragraphs=[
            "计算机制放在modeling：固定响应项，加一段或两段非负字节服务项。profiling负责采样、在训练尺寸上选参数与独立尺寸验收；候选包括简单线性和一个连续分段模型，不强加更复杂形式。固定项尚未独立拆成具体派发、等待或同步机制，没有用完整策略耗时反过来修正参数。",
            "每轮使用8个训练尺寸与8个不同验收尺寸，每个尺寸、每个方向各3个新进程。第一轮每批8次COPY；第二轮仅将每批重放改为256次，保留尺寸、模型候选、阈值及失败数据。更长连续传输也可能改变运行状态，不能将其视为唯一不变的降噪操作。",
            f"两种窗口的最大描述性均值变化为{pct(shift['maximum_absolute_mean_shift'])}，因此不能只说方差下降。跨轮时间和状态也可能参与其中；新H2D模型即使通过，也只适用于声明的新采样条件。",
            "表中均值误差要求最大不超过10%；CV要求每个验收进程不超过10%。两个条件同时满足才通过。第一轮小COPY波动大，因此增加拟合复杂度不足以说明已经修好。",
            f"观测诊断使用同进程相邻ABBA/BAAB，96次bare/stamped对照有{observer['passed_processes']}次通过，24个策略中{observer['qualified_programs']}个四进程全部通过；16次bare/bare空对照有{observer['sham_passes']}次通过。超限项：{observer_fail}。",
            "32次CPU资源干预没有显示一致改善，因此不能认定整进程affinity是原波动的主因。该假设及其负结果也被保留。静态硬件身份是在全部测量后补采并与旧E2核对一致，不冒充测量前的动态状态轨迹；未来prepare已补齐这些上下文记录。",
            "整进程affinity对照保持主计算线程mask=4、GGML线程数=1，只把允许进程使用的逻辑CPU从mask=4改为mask=20。下表负值表示第二种配置均值更低；它是该干预的总体效应，未定位到具体驱动线程，也没有假定物理核拓扑。"],
        tables=[dict(headers=["方向", "采样窗口", "最大验收尺寸均值误差", "CV通过进程", "完整传输验收"], rows=transfer_rows),
                dict(headers=["方向", "字节", "mask4均值μs", "mask20均值μs", "均值变化", "CV通过进程"], rows=conditions)],
        limits="原E2的冻结价格与0/6完整资格结果没有重写。新配对诊断使用不同harness和前处理，不是对历史两种采样方法的严格因果A/B。尺寸响应拟合通过也不等于物理PCIe带宽效应已识别；主机page-lock状态仍未知。", figures=[])
    group_rows = []
    for group in multi["groups"]:
        chosen = next(row for row in group["candidates"] if row["strategy"] == group["predicted_winner"])
        group_rows.append([group["workload"], "S" + str(group["predicted_winner"]), "S" + str(group["real_winner"]),
                           f"{chosen['predicted_ns']/1e6:.3f}", f"{chosen['measured_mean_ns']/1e6:.3f}", pct(group["point_regret"])])
    stage2 = dict(title="本机真实连接的多层Dense", conclusion=f"48个进程全部完成数值与动作验证；4组均选中实测均值最快方案。预测误差中位数{pct(multi['median_process_mean_ape'])}，最大{pct(multi['maximum_process_mean_ape'])}。",
        paragraphs=[
            "新增的是实际连接的2层/4层图：前层输出就是后层输入，各层拥有自己的权重与KV。原生程序执行完整图；预测复用此前冻结的137个独立动作价格，未拟合任何多层完整耗时。",
            "这里使用合成尺寸的Dense模板与F32数据，不是生产规模的量化LLM；“真实连接”描述数据依赖和实机执行方式。",
            f"矩阵为4组×全CPU/全GPU/CPU Attention混合3种方案×4个新进程。旧均值误差门槛通过{multi['process_mean_gate_passes']}/48，旧CV门槛通过{multi['process_cv_gate_passes']}/48；数值最大绝对误差{multi['max_absolute_port_error']:.6g}，满足冻结的atol/rtol=5e-5。",
            "4次均值误差失败集中于2L/H512/P64的混合方案：预测9.547ms，实测均值10.985ms，稳定低估约13.09%。该项CV通过，不能简单归为测量抖动；尚未分项辨别参数漂移与跨动作组合机制遗漏。",
            "该矩阵中，真实全图预测相对“独立单层预测×层数”的差异不足1%，没有导致赢家变化。因此这轮证明连接执行和成本复用可工作，不构成全图联合优化显著胜出的证据。"],
        tables=[dict(headers=["工作负载", "预测最快", "实测最快", "选中方案预测ms", "选中方案实测ms", "选错导致的额外耗时"], rows=group_rows)],
        limits="Decode是从给定旧KV开始的一次多层前向，不是自回归生成循环。本轮不包含MoE真实专家缓存/预取、UMA、并发TP或物理allocator/eviction验证；未重跑原E2完整观测矩阵。", figures=[])
    serving = ext["servingsim"]
    max_round = max(abs(row["rounding_difference_ns"]) for row in ext["compass"]["resident_whole_block_controls"])
    stage3 = dict(title="外部模拟器与同任务方法对照", conclusion="对照已从组件检查推进到完整块及完整请求轨迹；共成本等价与真实精度/性能优越性分开报告。",
        paragraphs=[
            f"LLMCompass固定版本运行8个完整原生Transformer块，导入相同LayerNorm/GeLU/MHA数学边界，再走我们的真实StrategyIR、compiler和tensor engine。8个全驻留整块成本对齐，最大差{max_round:.6f}ns，处于3个宏任务整数化误差界内。",
            "另外32种导入图的CPU/GPU放置组合通过独立搬运字节与时序账本检查，并从磁盘重载IR回放。CPU响应×4及COPY=5μs+字节/16GBps为明确的反事实输入；没有宣称它们是本机实测，也没有宣称超过原论文。",
            f"LLMServingSim固定版本完成{serving['whole_frontend_traces']}条完整多层请求轨迹、{serving['native_rows']}行原生trace，以及{len(serving['additional_pd_byte_trace_witnesses']) if isinstance(serving['additional_pd_byte_trace_witnesses'],list) else serving['additional_pd_byte_trace_witnesses']}项额外P/D字节检查。性能profile是显式构造的fixture，用于检查结构、放置和交接，不能作为真实时间。",
            "外部源码文件保持不变。PyTorch/Python导入兼容与TP=1未初始化诊断变量使用了明确记录的内存适配，并保留原始失败；没有修改roofline成本公式。"],
        tables=[dict(headers=["对象", "本轮实际完成", "能证明", "不能证明"], rows=[
            ["LLMCompass", "8完整块共成本对齐；32个IR组合检查", "同任务成本与协调语义可对齐", "独立实测精度、CPU/GPU方案速度胜出"],
            ["LLMServingSim", f"{serving['whole_frontend_traces']}完整trace与P/D检查", "前端层结构、字节与放置字段", "后台完整耗时；真实vLLM性能"],
        ])], limits="对已发表卸载系统的完整实机复现仍未完成。本轮属于模拟框架与受控策略机制对照，不是FlexGen/NEO/HybriMoE等完整系统的端到端性能排名。宏内部临时存储及workspace没有被共成本比较认证。", figures=[])
    if wsl:
        bind(wsl, "WSL ASTRA环境与执行尝试")
        stage3["paragraphs"].append("ASTRA后台的进一步本机尝试及结论见附带WSL执行收据；失败、缺依赖和未执行工作负载均保留，前端trace通过不替代后台计时。")
    if backend:
        full = read(backend)
        if full["status"] != "pass" or not full["common_cost_semantics_comparison"]:
            raise ValueError("Native backend common-cost comparison did not pass")
        parent = backend.parent
        backend_audit = read(parent / full["audit_file"])
        if backend_audit.get("passed", backend_audit.get("status") == "pass") is not True or sha(parent / full["audit_file"]) != full["audit_sha256"]:
            raise ValueError("Native backend audit changed")
        if sha(parent / full["manifest_file"]) != full["manifest_sha256"]:
            raise ValueError("Native backend artifact manifest changed")
        for path, label in ((backend, "新增4个原生后台完整请求"), (parent / full["audit_file"], "原生后台与事件图独立审计"),
                            (parent / full["manifest_file"], "原生后台产物哈希清单"), (parent / "BACKEND_REPORT.md", "后台完整对照报告")):
            bind(path, label)
        stage3["conclusion"] = "完整块、完整请求轨迹及4个实际后台完整请求均已完成共成本语义对照；实机性能精度比较仍是另一项工作。"
        stage3["paragraphs"].extend([
            f"随后补齐了{full['actual_complete_native_cli_jobs']}个完整原生CLI请求：调度器→原生trace→Chakra→ASTRA C++后台→请求完成。使用H128、FP16、2/4层与P16/P64，profile仍为明确的合成fixture。",
            "这4个Serving ET导入走的是我们的modeling.performance.Graph/evaluate事件图，并未经过该ET的StrategyIR/compiler路径。编译器路径的对照证据是前面的LLMCompass导入实验，二者不能合并为同一种表达覆盖证明。",
            "显式到达事件来自原前端max(1, arrival)约定，本组arrival=0因而是1ns；将该ET节点纳入后，全请求与目标区间均对齐。原生成器的sampler输出为prompt_count×4字节（64或256B），也按实际ET保留，没有假定成4B或用补常数拟合。该检查不验证真实LLM采样算法。",
            "早先24条前端轨迹的后台计数仍保留为0；新增4个后台请求另存目录。依赖仅在workspace私有prefix中解包，未作全局软件安装。"
        ])
        stage3["tables"][0]["rows"][1] = ["LLMServingSim", "24完整trace、P/D检查；新增4个完整后台请求",
            "原生后台与事件图共成本语义一致", "实测硬件精度、真实vLLM性能、全部策略IR覆盖"]
        stage3["tables"].append(dict(headers=["层数", "Prefill", "ASTRA目标ns", "我们的事件图目标ns", "ASTRA全请求ns", "全请求差异ns"], rows=[
            [row["layers"], row["prompt"], row["target_interval_ns"], row["ours_modeling_event_dag_ns"],
             row["native_total_clock_ns"], row["whole_protocol_difference_ns"]] for row in full["cases"]]))
    figure_specs = ((stage1, "transfer-validation.png", "两种采样窗口分别使用新训练和新验收进程；通过条件不变。"),
                    (stage1, "observer-consistency.png", "新同进程观测与空对照的全部差异；5%为预设门槛。"),
                    (stage2, "multilayer-validation.png", "冻结独立动作预测与48个新多层实机进程；点表示新进程均值。"))
    for stage, filename, caption in figure_specs:
        path = figures / filename
        stage["figures"].append(dict(path=str(path.resolve()), sha256=sha(path), caption=caption))
    total = v1["all_native_processes"] + v2["all_native_processes"] + multi["completed_processes"] + affinity["native_completed"]
    evidence = dict(schema="dac-followup-report-1", sources=sources,
        overview=f"本轮新增{total}个完成的本机原生测量进程，分开验证传输响应、观测一致性、多层组合与辅助线程资源条件；另完成外部模拟器的有界同任务对照。结果同时保留通过项、未通过项和未测范围。",
        historical_policy="此前E2/E3的冻结参数、原始结果与失败门槛均保持原样；本轮结果另建目录，没有用新数据覆盖旧失败。",
        sections=[stage1, stage2, stage3],
        reproduction="代码位于Simulator/profiling/dac_followup、profiling/highlevel_accuracy及experiments/dac_followup_*。每个campaign保存请求、原始输出、进程身份、数据/源码/二进制哈希与冻结屏障。profiling软件回归共593项，592通过、1项可选原生单测跳过。首次回归因sandbox临时目录权限报错；改用workspace TMP后重跑全部通过，原日志保留。下列JSON和报告可追溯数字与代码。")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return dict(evidence=str(out), completed_native_processes=total, long_window_routes_qualified=qualified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "external", "figures", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--wsl", type=Path)
    parser.add_argument("--backend", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.external, args.figures, args.out, args.wsl, args.backend)))
