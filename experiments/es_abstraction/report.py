"""Render retained E/S evidence without changing or rerunning an experiment.

The report deliberately separates semantic correctness, canonical-program
diversity, measured utility, and novelty.  Its inputs are immutable experiment
outputs; missing final native evidence is an error, never a placeholder result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import statistics


COLORS = {"main": "#3274A1", "control": "#D48A35", "geometry": "#34836D"}
GROUP_NAMES = {"main": "Boundary", "control": "q = 33", "geometry": "Larger matrices"}
PATTERN_NAMES = {
    "fixed_heavy_light_pairs": "固定重/轻专家",
    "swap_heavy_light_pairs": "重/轻专家互换",
    "halfway_heavy_light_swap": "中途重/轻专家互换",
    "rotating_disjoint_pairs": "轮换专家对",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def esc(value):
    return html.escape(str(value))


def number(value, digits=2):
    return f"{float(value):,.{digits}f}"


def pct(value):
    return number(value) + "%"


def interval(values):
    return f"[{number(values[0])}, {number(values[1])}]"


def link(path, out, label=None):
    relative = Path(os.path.relpath(Path(path).resolve(), Path(out).resolve())).as_posix()
    return f'<a href="{esc(relative)}">{esc(label or Path(path).name)}</a>'


def table(headers, rows, css=""):
    return (f'<div class="scroll"><table class="{css}"><thead><tr>'
            + "".join(f"<th>{esc(x)}</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def validate(semantics, audit, manifest, analysis):
    if semantics.get("schema") != "es-semantics-result-1":
        raise ValueError("unrecognized semantic evidence")
    if analysis.get("schema") != "es-abstraction-native-analysis-1":
        raise ValueError("unrecognized final native analysis")
    main = [x for x in manifest["labels"] if x["main"]]
    if len(main) != analysis["nominal_labels"]:
        raise ValueError("analysis and frozen labels do not match")
    ids = {x["program_id"] for x in main}
    if ids != {x["id"] for x in analysis["rows"]}:
        raise ValueError("analysis does not cover the exact frozen main programs")
    if {x["id"] for x in manifest["workloads"]} != {x["workload_id"] for x in analysis["selection"]}:
        raise ValueError("final analysis lacks a frozen workload")
    if any(x["rounds"] not in (5, 10) for x in analysis["selection"]):
        raise ValueError("expected five or ten independent process rounds")
    if audit["cases"] != semantics["nominal_cases"]:
        raise ValueError("semantic audit belongs to a different case matrix")
    return main


def build_summary(semantics, audit, witness, g0, manifest, analysis, inputs,
                  profile_costs, hardware, pilot, accuracy_note):
    main = validate(semantics, audit, manifest, analysis)
    selections = analysis["selection"]
    strong_ids = {x["program_id"] for x in main if "B_STRONG" in x["baseline_families"]}
    full_ids = {x["program_id"] for x in main}
    additional = full_ids - strong_ids
    if len(additional) != analysis["additional_joint_programs"]:
        raise ValueError("strong-envelope program count mismatch")
    groups = []
    for name in ("main", "control", "geometry", "transfer"):
        ls = [x for x in main if x["group"] == name]
        if not ls:
            continue
        ids = {x["program_id"] for x in ls}
        rows = [x for x in selections if x["group"] == name]
        groups.append(dict(group=name, workloads=len(rows), nominal_labels=len(ls),
                           unique_programs=len(ids), beyond_strong=len(ids - strong_ids),
                           median_regret_percent=statistics.median(x["selection_regret_percent"] for x in rows),
                           worst_regret_percent=max(x["selection_regret_percent"] for x in rows),
                           min_gain_percent=min(x["native_gain_vs_strong_percent"] for x in rows),
                           max_gain_percent=max(x["native_gain_vs_strong_percent"] for x in rows)))
    witnesses = []
    for item in witness:
        diagnostic = item["diagnostic"]
        witnesses.append(dict(case_id=item["case"]["id"], preparation=item["case"]["preparation"],
                              weight_bytes=sum(x["bytes"] for x in diagnostic["loads"]),
                              loads=len(diagnostic["loads"]), unused_bytes=diagnostic["unused_weight_bytes"],
                              unused_loads=diagnostic["unused_weight_loads"],
                              complete_parent_weight_bytes=item["case"]["parent_group_bytes"]))
    target = analysis["median_regret_percent"] <= 5 and analysis["p90_regret_percent"] <= 10
    # A good numerical result is not a novelty verdict. This audit explicitly
    # leaves the key source-checked witness unresolved, regardless of speedup.
    evidence = [
        dict(id="A1", condition="重要且经过最近工作源代码核查的表达/组合差异", status="not_established",
             reason="G0仅发现有待验证的接口差异，尚无具体可执行的差异见证；广义E/S联合、缓存和负载感知已有先例。"),
        dict(id="A2", condition="明确组合规则与独立语义保持证据", status="supported_with_limits",
             reason=f"{semantics['nominal_cases']}项限定物理契约中无可行性误判；FFN最大绝对误差{semantics['max_ffn_absolute_error']:.3g}。{audit['cases_with_unused_weight_loads']}项仍有未消费的搬运，不能声称编译器没有优化空间。"),
        dict(id="A3", condition="强对照下可重复的真实收益，或有真实证据的关键可行性/排序纠正",
             status="not_established" if not analysis["positive_gain_workloads"] else "requires_mechanism_and_replication_review",
             reason=f"超出强对照的独立程序为{len(additional)}；{analysis['positive_gain_workloads']}个负载满足预先定义的点收益与名义区间筛选。A的逻辑容量反例未构成同一机制的真实设备收益证据。"),
        dict(id="A4", condition="真实专家矩阵尺寸迁移、准确范围与可复用成本", status="supported_with_limits",
             reason="包含H2048/I1024矩阵；权重、专家数E8、top-2与路由为合成。记录编译、模拟、实测成本；不等于真实完整OLMoE模型或全流程服务验证。"),
        dict(id="SA1", condition="未见条件迁移、超越已知耦合的设计结论、独立第二机制", status="not_established",
             reason="可选32次调用迁移没有因增加工作量自动获得意义。先建立新物理组合及其后果，再按冻结规则验证迁移；自然路由与第二机制证据不在本次范围。"),
    ]
    return dict(schema="es-abstraction-evidence-summary-1", created_utc=datetime.now(timezone.utc).isoformat(),
                input_files={str(p): digest(p) for p in inputs},
                semantic={k: semantics[k] for k in ("nominal_cases", "status_counts", "false_feasible", "false_infeasible",
                           "independent_feasible", "independent_infeasible", "simplified_false_feasible",
                           "max_ffn_absolute_error", "numerical_requests", "source_integrity_verified")},
                lowering_audit={k: audit[k] for k in ("cases_with_unused_weight_loads", "unused_weight_loads",
                                                    "slice_bounds_checked", "ffn_output_shapes_checked")},
                unused_load_witnesses=witnesses, q_b=manifest["q_b"],
                space=dict(nominal_labels=len(main), unique_programs=len(full_ids),
                           collapsed_labels=len(main)-len(full_ids), strong_unique_programs=len(strong_ids),
                           beyond_strong_programs=len(additional), groups=groups),
                hardware=hardware,
                profiling={**profile_costs, "peak_host_ram_bytes": None,
                           "peak_host_ram_status": "not_measured"},
                pilot={**pilot, "fresh_processes": 1, "included_in_validation": False},
                accuracy_diagnostic_five_round_snapshot=accuracy_note,
                native={k: analysis[k] for k in ("workloads", "independent_processes", "rounds_by_workload",
                        "all_numerical_and_physical_checks_passed", "max_abs_error", "mean_ape_percent", "max_ape_percent",
                        "median_regret_percent", "p90_regret_percent", "worst_regret_percent", "positive_gain_workloads",
                        "compile_seconds", "evaluate_seconds", "native_process_wall_seconds")},
                engineering_regret_target_met=target, g0_status=g0["status"],
                evidence_conditions=evidence, accept_evidence_complete=False, strong_accept_evidence_complete=False,
                paper_action="retain_v5_no_new_contribution_claim", selection=selections,
                claim_boundary="Project evidence targets are not DAC acceptance criteria or guarantees. Counts prove scope, not novelty.")


def make_figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9, "text.color": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "lines.linewidth": 1.1,
                         "savefig.facecolor": "white", "figure.facecolor": "white",
                         "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    specs = []
    def store(fig, name, caption):
        # Never use bbox_inches='tight': preserve the declared 3.4-inch width.
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{suffix}", dpi=300)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = []
        for item in fig.findobj(matplotlib.text.Text):
            if item.get_visible() and item.get_text():
                bb = item.get_window_extent(renderer)
                if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
                    overflow.append(item.get_text())
        if overflow:
            raise ValueError(f"figure text exceeds physical canvas {name}: {overflow}")
        specs.append(dict(stem=name, width_inches=float(fig.get_figwidth()),
                          height_inches=float(fig.get_figheight()), text_pt=9,
                          font="Times New Roman", font_path=font, caption=caption,
                          english_only=True, overflow_text=overflow))
        plt.close(fig)

    witnesses = summary["unused_load_witnesses"]
    byprep = {w["preparation"]: w for w in witnesses}
    if set(byprep) < {"demand", "spare"}:
        raise ValueError("need demand and spare concrete lowering witnesses")
    baseline = byprep["demand"]["complete_parent_weight_bytes"]
    values = [baseline, byprep["demand"]["weight_bytes"], byprep["spare"]["weight_bytes"]]
    unused = [0, byprep["demand"]["unused_bytes"], byprep["spare"]["unused_bytes"]]
    used = np.array(values) - np.array(unused)
    fig, ax = plt.subplots(figsize=(3.4, 2.30))
    fig.subplots_adjust(left=.20, right=.98, bottom=.23, top=.79)
    x = np.arange(3)
    ax.bar(x, used, width=.57, color=COLORS["main"], label="Consumed")
    ax.bar(x, unused, bottom=used, width=.57, color=COLORS["control"], label="Unused")
    ax.set_xticks(x, ["Three weights", "Demand", "Spare"])
    ax.set_ylabel("Weight H2D bytes")
    ax.set_ylim(0, max(values)*1.20)
    ax.set_yticks([0, 1024, 2048])
    for i, v in enumerate(values):
        ax.text(i, v+65, str(v), ha="center", va="bottom")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, handlelength=1.1,
              columnspacing=1.0, borderaxespad=0)
    store(fig, "fig1_lowering_witness", "One analytical shared-parent FFN case. Correct outputs coexist with avoidable weight loads; the three-weight reference is a byte accounting bound, not measured latency.")

    groups = summary["space"]["groups"]
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    fig.subplots_adjust(left=.17, right=.98, bottom=.24, top=.76)
    x = np.arange(len(groups)); width=.24
    definitions = [("nominal_labels", "Labels", "#C2CCD5"),
                   ("unique_programs", "Unique", COLORS["main"]),
                   ("beyond_strong", "Beyond strong", COLORS["control"])]
    for offset, (field, name, color) in zip((-width, 0, width), definitions):
        vals = [g[field] for g in groups]
        ax.bar(x+offset, vals, width=width*.94, label=name, color=color)
        for pos, val in zip(x+offset, vals):
            ax.text(pos, val+3, str(val), ha="center", va="bottom")
    ax.set_xticks(x, [GROUP_NAMES.get(g["group"], g["group"]) for g in groups])
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(g["nominal_labels"] for g in groups)*1.20)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=.8, borderaxespad=0)
    store(fig, "fig2_program_diversity", "Exact full-program deduplication within each workload. Beyond strong counts unique programs absent from the union of the four prespecified restricted baseline families; it is not a novelty score.")

    rows = summary["selection"]
    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    fig.subplots_adjust(left=.20, right=.98, bottom=.23, top=.76)
    for group in groups:
        positions = [i+1 for i, r in enumerate(rows) if r["group"] == group["group"]]
        selected = [r for r in rows if r["group"] == group["group"]]
        color = COLORS.get(group["group"], "#775EA8")
        vals = [r["selection_regret_percent"] for r in selected]
        cis = np.array([r["regret_95ci_percent"] for r in selected])
        # Intervals need not contain the point after bootstrap minima;
        # draw endpoints directly rather than impose nonnegative yerr.
        ax.vlines(positions, cis[:, 0], cis[:, 1], color=color, linewidth=.7, alpha=.7)
        ax.scatter(positions, vals, s=13, color=color, label=GROUP_NAMES.get(group["group"], group["group"]), zorder=3)
    ax.axhline(0, color="black", linewidth=.6)
    ax.set_xlim(.3, len(rows)+.7)
    ax.set_ylabel("Selection regret (%)")
    ax.set_xlabel("Frozen workload index")
    ticks = [1, 4, 8, 12, 16, 20, 24] if len(rows) == 24 else list(range(1, len(rows)+1, 4))
    ax.set_xticks(ticks)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2,
              frameon=False, handletextpad=.3, columnspacing=.6, borderaxespad=0)
    identical_pool = summary["space"]["beyond_strong_programs"] == 0
    relationship = (" Joint and strong-baseline program pools are identical, so native gain equals negative regret; it is not a second independent result."
                    if identical_pool else " The reference is the finite measured joint pool.")
    store(fig, "fig3_native_decisions", "Frozen simulation-selected policies evaluated on fresh-process means, with ordinary nominal 95% bootstrap intervals." + relationship + " Small-sample coverage is not guaranteed.")
    return specs


def render_html(out, summary, semantics, audit, g0, analysis, paths, figures):
    space, native = summary["space"], summary["native"]
    profile, hardware, pilot = summary["profiling"], summary["hardware"], summary["pilot"]
    accuracy_note = summary["accuracy_diagnostic_five_round_snapshot"]
    round_counts = Counter(native["rounds_by_workload"].values())
    round_text = "；".join(f"{count}个负载各{nr}轮" for nr, count in sorted(round_counts.items()))
    unique_message = ("联合集合没有增加强对照之外的实际执行程序，不能把更多策略标签写成新的策略能力。"
                      if space["beyond_strong_programs"] == 0 else
                      f"联合集合增加了{space['beyond_strong_programs']}个强对照之外的执行程序；不同程序仍须单独证明意义和文献差异。")
    outcome = ("当前证据补强了正确性与可检验性，但尚未建立足以宣称新抽象贡献的完整证据链。"
               "保持v5，暂不生成声称新增贡献的v6。")
    sections = []
    sections.append(f'<section class="lead"><p class="eyebrow">E/S ABSTRACTION · RETAINED EVIDENCE</p><h1>E/S抽象价值实验报告</h1>'
                    f'<p>{outcome}</p><p>{unique_message}</p><div class="metrics">'
                    + "".join(f'<div><b>{esc(v)}</b><span>{esc(t)}</span></div>' for v,t in [
                        (f"{semantics['nominal_cases']}", "声明语义组合"),
                        (f"{space['nominal_labels']} → {space['unique_programs']}", "标签 → 独立程序"),
                        (str(space['beyond_strong_programs']), "强对照之外的程序"),
                        (pct(native['median_regret_percent']), "真实选择遗憾中位数")])
                    + '</div><p class="note">以上数量用于说明覆盖范围。它们不是新颖性得分，也不是DAC录用标准。</p></section>')
    sections.append('<section><h2>1. 实验到底比较什么</h2><p>策略表达分为执行选择E和存储调度S；编译器把它们落实为实际动作，性能器计算同一动作的成本。'
                    '本次分别验证：A，编译和物理存储语义是否正确；B，增加组合是否产生了已有强对照不能产生的实际程序；C，模拟器选择的策略在本机是否接近候选集合中的实测最优。'
                    'G0文献/接口审查单独判断新颖性，不由A/B/C的测试数量代替。</p>'
                    + table(["部分", "输入与比较", "可以支持的结论"], [
                        ["A：物理契约", "Dense FFN拆分、物化分片/共享父分配、按需/空闲准备、1/2请求、3档容量", "限定契约内的可行性、依赖、字节数和FFN数值保持"],
                        ["B：表达空间", "5种E × 3种S，在24负载上编译；完整程序精确去重", "名义组合与实际执行多样性之间的关系"],
                        ["C：设备验证", "冻结成本和预测；相同程序、输入、路由、执行边界的CPU/GPU实测", "有限候选池的选择遗憾和强对照收益"],
                        ["G0：最近工作", "版本化源码接口与原始论文", "差异是否超出API改名或已知计算/缓存耦合"]]) + '</section>')
    arows = [
        ["独立判定物理可行/不可行", f"{semantics['independent_feasible']} / {semantics['independent_infeasible']}", "允许逐矩阵加载和驱逐，不人为要求三个矩阵同时驻留"],
        ["完整语义误判可行 / 误判不可行", f"{semantics['false_feasible']} / {semantics['false_infeasible']}", "依赖独立物理账本及明确容量契约"],
        ["仅按逻辑切片计容量的对照误判", str(semantics['simplified_false_feasible']), "人工简化消融；不代表LLMServingSim或LLMCompass有此错误"],
        ["FFN数值最大绝对误差", f"{semantics['max_ffn_absolute_error']:.3g}", f"{semantics['numerical_requests']}个请求的FFN边界；其他Dense算子仅做结构核查"],
        ["未被消费的权重载入", f"{audit['cases_with_unused_weight_loads']}个组合，{audit['unused_weight_loads']}次", "结果正确，但编译/运行时准备顺序仍有可消除的搬运"],
    ]
    sections.append('<section><h2>2. A：物理语义正确，同时暴露了编译器冗余</h2>'
                    '<p>逻辑上只计算一个切片，不意味着物理上只分配这个切片。共享父分配的视图必须承担父矩阵的实际驻留和传输；真正物化的独立分片则按分片计费。'
                    '这一差异会改变容量是否可行。本次没有发现完整物理账本的可行性误判。</p>' + table(["观测", "结果", "解释/边界"], arows)
                    + '<p>简化对照的4个误判来自同一物理机制的4种配置，不是4个独立的新颖性发现。</p>'
                    '<p><strong>正确性不等于编译器已经没有优化空间。</strong>后置审计找到载入后未经读取就被驱逐的权重。'
                    '一个共享父矩阵见证中，三个权重合计1536 B，按需执行搬运2048 B、空闲准备搬运2560 B。'
                    '这是模拟动作记录中的真实冗余，不是机器计时波动；它指出下一步应先修复准备/驱逐的协调规则。</p>'
                    + figure_html(figures[0]) + '<p class="note">A是小型解析语义实验，没有GPU性能测量；图中的字节差不能直接写成端到端速度提升。'
                    '后置审计对被拒绝图做的代数形状核查，也不表示这些图可以执行。</p></section>')
    erows = [
        ["E_CPU / E_GPU", "全部专家在固定设备执行"],
        ["E_SHAPE", "决策只比较该形状的CPU/GPU计算；决策时忽略权重LOAD，物理计时仍收费"],
        ["E_RESIDENT", "命中优先GPU；缺失比较CPU与加载+GPU完整路径"],
        ["E_PATH", "命中和缺失均比较完整可行路径"],
        ["S_RELEASE / S_LRU / S_FREQ", "用后释放 / LRU保留 / 依据已公开路由历史的频率保留"],
    ]
    baseline_rows = [[key, val] for key,val in [
        ("B_E", "固定用后释放，枚举全部E"), ("B_S", "固定纯CPU或纯GPU，枚举全部S"),
        ("B_FIXED_HYBRID", "E_SHAPE × 全部S；形状感知混合对照，不是固定专家ID分割"),
        ("B_RESIDENT", "E_RESIDENT × 全部S"), ("B_STRONG", "上述四个集合的并集；实测结果取最好的程序作为强对照"),
        ("B_JOINT", "全部15种组合；枚举器、顺序、物理成本和初始状态相同")]]
    group_rows = [[esc(GROUP_NAMES.get(g['group'],g['group'])), str(g['workloads']), str(g['nominal_labels']),
                   str(g['unique_programs']), str(g['beyond_strong'])] for g in space['groups']]
    sections.append('<section><h2>3. B：策略标签增加了，实际程序增加了吗</h2>'
                    + table(["规则", "含义"], erows) + table(["对照", "定义"], baseline_rows)
                    + f'<p>完整输入、路由、种子、配置和动作序列一致的标签只测量一次，避免把同一程序当成多条独立证据。'
                    f'{space["nominal_labels"]}个标签去重为{space["unique_programs"]}个实际程序，折叠{space["collapsed_labels"]}个别名。{unique_message}</p>'
                    + table(["工作负载组", "负载数", "名义标签", "实际程序", "超出B_STRONG"], group_rows)
                    + figure_html(figures[1])
                    + ('<p><strong>本次还有一个不依赖测量噪声的结论：</strong>联合集合的全部实际程序都在强对照集合中，'
                       '而强对照分母取同一份实测数据中的最小值。因此任何冻结选择相对这个强对照的收益都不可能大于0；'
                       '并且强对照收益恒等于选择遗憾的负值，因此图3只画遗憾，避免把镜像指标当成两条证据。'
                       '不能把不同标签的计时噪声包装成新策略收益。真实测试仍有价值：它衡量模拟选择是否接近这个有限池中的最好方案。</p>'
                       if space['beyond_strong_programs'] == 0 else '')
                    + '<p>命中时如果GPU本来就比CPU快，E_RESIDENT与E_PATH会选同一路径；缺失时两者又比较相同的完整路径。'
                    '因此相同缓存策略下可能逐步产生完全相同的程序。这是策略规则与负载条件的共同结果，不是靠给策略换名字就能扩大的空间。</p>'
                    '<p class="note">这些对照是已声明机制的同边界投影，不是完整Fiddler、HybriMoE或Apex系统复现。有限集合的模拟最小值更低，本身也不能证明研究贡献。</p></section>')
    sections.append('<section><h2>4. C：本机真实选择结果</h2>'
                    + table(["本机平台", "配置"], [
                        ["CPU", esc(hardware["cpu"].strip()) + f'；{hardware["logical_processors"]}个逻辑处理器'],
                        ["GPU", esc(hardware["gpu"]) + f'；驱动报告总显存{hardware["gpu_total_MiB"]} MiB'],
                        ["操作系统", esc(hardware["os"])],
                    ])
                    + f'<p>独立单位是新进程的均值，不是一个进程内的10个样本。共{native["independent_processes"]}个独立进程，{round_text}。'
                    '前5轮采用分块随机顺序；若主对比的普通区间宽度超过10个百分点，对该负载的所有程序追加5轮，最多10轮，追加规则不依赖收益正负。</p>'
                    '<div class="formula">选择遗憾 = 实测时间(模拟选中的策略) / 实测候选池最短时间 − 1<br>'
                    '强对照收益 = 1 − 实测时间(模拟选中的策略) / 实测强对照集合最短时间</div>'
                    + table(["指标", "结果", "如何理解"], [
                        ["选择遗憾：中位数 / P90 / 最大", f'{pct(native["median_regret_percent"])} / {pct(native["p90_regret_percent"])} / {pct(native["worst_regret_percent"])}', "越接近0越好；相对于同一有限候选池，不是所有可能策略"],
                        ["平均 / 最大绝对百分比误差", f'{pct(native["mean_ape_percent"])} / {pct(native["max_ape_percent"])}', "时间预测误差，与选对策略是不同指标"],
                        ["正收益筛选", str(native["positive_gain_workloads"]), "点收益≥5%且名义多重比较区间下界>0；仍需重复机制解释，不是录用票数"],
                        ["数值、动作和容量核查", "全部通过" if native['all_numerical_and_physical_checks_passed'] else "存在失败", "存在失败的对应性能结论不得成立"],
                        ["最大数值绝对误差", f'{native["max_abs_error"]:.3g}', "原生F32专家FFN输出的独立参考核查"],
                    ]) + figure_html(figures[2])
                    + '<p class="note">区间通过共同重采样匹配进程轮次获得，每次重算强对照最小值；模拟选中的策略保持冻结。'
                    '收益区间做Bonferroni调整，Holm精确符号翻转p值另外列作诊断。仅5/10轮时，bootstrap尾部覆盖率没有有限样本保证；5轮的p值分辨率也不足以通过24项严格校正。'
                    '不把这些名义区间写成确定性证明，也不以调用级CV设通过门槛。</p>'
                    '<p>大尺寸组采用H2048/I1024、E8/top-2的合成F32专家矩阵；这验证矩阵尺寸迁移。它不是完整OLMoE模型、真实路由数据集、全LLM多层服务或异步重叠执行验证。'
                    f'边界行数q<sub>b</sub>={summary["q_b"]}仅由独立校准集选择，所有物理价格保持冻结。策略只看当前公开路由和历史；离线选择仍是给定轨迹条件下的设计选择，不是未知未来请求上的全局最优。</p></section>')
    cpu_pair = accuracy_note['pure_cpu_k_comparison']['representative_pair']
    cpu_rows = [[str(k), number(cpu_pair[f'k{k}']['predicted_ns']/1e6), number(cpu_pair[f'k{k}']['mean_ns']/1e6)]
                for k in (2, 4)]
    q32 = next(x for x in accuracy_note['calibration_context_diagnostics']
               if x['geometry']=='small' and x['k']==4 and x['category']=='EXPERT:gpu:q=32')
    q32_rows = [[esc(context), number(v['mean_ns']/1e3)]
                for context,v in q32['by_calibration_light_q'].items()]
    sections.append('<section><h2>4.1 绝对时延误差：先查校准上下文，不先归咎操作系统</h2>'
                    '<p class="note"><strong>本节仅引用冻结的5轮诊断快照：</strong>133个程序、665个进程。'
                    '不包含后来对部分负载增加的5轮，因此这里的分组与示例不是最终补轮统计；第4节主指标和图3使用最终结果。</p>'
                    '<p>该快照的大误差主要是高估，集中在K=4条件，不能描述成系统性少算衔接时间。'
                    '133个程序的动作价目求和均与冻结预测一致，排除了这批结果的查表求和算术错误，但没有排除校准条件不匹配或未表示的状态影响。</p>'
                    '<p>一个直接见证是大矩阵纯CPU路径：K=2与K=4使用相同的动作价目坐标和次数，均不分配GPU权重，实测接近；'
                    '预测却随着分表显著改变。K本身不是这条纯CPU路径的GPU驻留开销，说明K分表携带了校准上下文差异。</p>'
                    + table(["槽配置K", "冻结预测(ms)", "5轮实测均值(ms)"], cpu_rows)
                    + '<p>另一个见证是small/K4的同一个gpu:q32成本单元：在不同轻专家行数的校准上下文中，分项均值差异很大。'
                    f'冻结价格把这些上下文汇总为{number(q32["frozen_price_ns"]/1e3)} μs。</p>'
                    + table(["校准轻专家上下文", "gpu:q32分项均值(μs)"], q32_rows)
                    + '<p>这证明冻结价目的跨上下文可迁移性不足，但<strong>不能直接确定原因是OS、H2D、热状态、缓存或slot复用</strong>。'
                    '验收为8次调用的bare总时间，校准为4次调用的stamped分项；缺少同程序对照，不能把插桩影响直接减成常数，也不能把总时间差归到某个算子。</p>'
                    '<p>若后续继续精度修正，应先用独立数据匹配q、调用长度、bare/stamped模式和上下文，检查本应一致的成本坐标是否一致，'
                    '再区分参数漂移与modeling缺少的状态。不能拿当前验收集重新拟合后称为独立通过。'
                    '<strong>重测均值不会解决本轮的新颖性缺口；选择接近最优与全范围时延预测准确也不能混为一谈。</strong></p></section>')
    native_rows = []
    for i,r in enumerate(summary['selection'],1):
        native_rows.append([f'W{i:02d}', esc(GROUP_NAMES.get(r['group'],r['group'])), str(r['k']),
                            esc(PATTERN_NAMES.get(r['pattern'],r['pattern'])), str(r['seed']), str(r['rounds']),
                            esc(r['joint_choice']), pct(r['selection_regret_percent']),
                            pct(r['native_gain_vs_strong_percent']), interval(r['gain_familywise_95ci_percent']),
                            number(r['holm_adjusted_p'],3)])
    sections.append('<section><h2>5. 全部负载明细</h2><p>图3的横轴序号与下表一致。所有零收益和负收益保留。</p>'
                    + table(["编号", "组", "槽K", "路由", "种子", "进程轮", "冻结选择", "遗憾", "强对照收益", "名义调整区间(%)", "Holm p诊断"], native_rows, "compact") + '</section>')
    evidence_rows = [[esc(x['id']), esc(x['condition']), esc({"not_established":"证据不足", "supported_with_limits":"限定范围支持",
                      "requires_mechanism_and_replication_review":"仍需机制与重复审查"}[x['status']]), esc(x['reason'])] for x in summary['evidence_conditions']]
    sections.append('<section><h2>6. 距离Accept / Strong Accept证据目标</h2>'
                    '<p>下面是项目自己的研究证据目标，不是DAC官方打分表。即使某个数值目标达成，审稿人仍需要判断差异、意义、相关工作和写作。'
                    'G0指出，联合执行/存储、按token负载选择CPU/GPU、缓存/预取协同并非首次提出。A的物理契约证据与B/C的整专家调度证据属于不同机制，不能拼接成一个未经证明的新颖性结论。</p>'
                    + table(["条件", "需要证明", "当前判断", "证据与缺口"], evidence_rows)
                    + '<p><strong>本轮不生成v6贡献升级稿。</strong>应先完成下面的具体差异见证和现有lowering修正，再决定论文主张是否能增强。</p></section>')
    sections.append('<section><h2>7. 下一步只围绕已发现的缺口</h2>'
                    '<ol><li><strong>修复无用搬运。</strong>从保留的demand/spare最小见证追踪“准备→预留→驱逐→首次读取”。'
                    '修正同一物理父分配的准备去重、即将消费对象的保留或过早预取规则；不允许用后验实测时间调整均值掩盖问题。'
                    '保留失败版本，要求独立数值/容量审计不退化，具体见证中未消费载入被消除；再用新版本、独立输入核查收益与代价。</li>'
                    '<li><strong>构造一个具体的物理组合差异见证。</strong>保持同一分片计算，明确物化独立分片或共享父矩阵、多个读者寿命、容量约束和准备时机。'
                    '对照版本化的最近模拟器接口：直接可表达就承认一致；需要扩展就标明缺哪个语义字段和由此可能发生的可行性/排序差异。'
                    '给独立账本和FFN数值参考，然后为同一个机制补一组同边界原生执行证据。不能把本次人工逻辑字节消融扣到已有模拟器上。</li>'
                    '<li><strong>只为检验可辨识性设计A/B/C负载。</strong>若后续研究仍需区分E_PATH、E_RESIDENT和E_SHAPE，可分别覆盖C&gt;G+L、G&lt;C&lt;G+L、C&lt;G，'
                    '并安排实际产生缓存命中的路由历史。必须独立登记，不修改本轮冻结验收集。三个区间能区分程序，并不意味着策略思想新颖；已知驻留专家仍可在CPU执行的先例必须保留。</li>'
                    '<li><strong>最后才启动迁移。</strong>只有具体差异和有效机制确立后，才用原计划的未见种子/32次调用做冻结选择迁移。'
                    '不为凑数字增加多GPU、操作系统调度、强化学习或另一个执行器。</li></ol></section>')
    refs = ''.join(f'<li>{link(p,out,label)}</li>' for label,p in paths.items())
    sources = ''.join(f'<li><a href="{esc(x["url"])}">{esc(x["title"])} — {esc(x["version"])}</a></li>' for x in g0['sources'])
    sections.append('<section><h2>8. 成本与复核入口</h2>'
                    + table(["采集阶段", "独立进程数", "累计记录耗时", "用途/计时边界"], [
                        ["Profiling校准", str(profile["calibration_processes"]),
                         f'{number(profile["native_subprocess_seconds"])} s', "原生子进程累计墙钟，包括进程生命周期；不等于纯CPU/GPU运算时间"],
                        ["Pilot", str(pilot["fresh_processes"]), f'{number(pilot["wall_seconds"])} s',
                         "独立种子33030，检查正确性和运行耗时；不用于拟合，也不并入验收"],
                        ["独立验收", str(native["independent_processes"]),
                         f'{number(native["native_process_wall_seconds"])} s', "一次invoke及随后verify的累计墙钟，不含gzip与全部准备；10个进程内样本不当成10个独立进程"],
                    ])
                    + f'<p>新增{profile["new_exact_cells"]}个精确成本单元，复用旧单元{profile["reused_exact_cells"]}个。'
                    '本轮采用混合重/轻专家的完整生命周期口径，旧数据口径不匹配，因此重新测量，未在验收结果上拟合。'
                    '<strong>峰值主存占用：未知，未采集。</strong>不以估算值填补。</p>'
                    + table(["观测成本", "数值", "范围"], [
                        ["全部策略编译", f'{number(native["compile_seconds"])} s', "各候选单次内存编译累计；不含因果前缀重复检查和证据JSON写盘"],
                        ["全部动作模拟", f'{number(native["evaluate_seconds"])} s', "共享modeling调用的累计耗时，不含初始profiling"],
                        ["语义验证墙钟", f'{number(semantics["elapsed_seconds"])} s', "解析验证，不含后置审计"],
                    ]) + '<p class="note">端到端候选生成墙钟与峰值主存未采集，均为未知。以上阶段的采集边界不同，验收累计墙钟不是整个实验的总墙钟；'
                    '不能用这些分项构造严谨的整体加速比，不把累计墙钟笼统称为“原生计算时间”，也不据此宣称已证明整体探索成本优势。'
                    'PDF/SVG图保留3.4英寸原宽、Times New Roman 9 pt、白底黑字；可直接按单栏原尺寸用于论文。</p>'
                    + f'<ul>{refs}</ul><h3>G0引用的原始工作</h3><ul>{sources}</ul></section>')
    css = """
    :root{color-scheme:light;--ink:#162331;--line:#d9e1e8;--blue:#245e85}
    *{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:var(--ink);font:16px/1.8 Arial,'Microsoft YaHei',sans-serif}
    main{max-width:1160px;margin:36px auto 64px;padding:0 24px}section{background:white;border:1px solid var(--line);border-radius:12px;padding:26px 32px;margin:20px 0}
    h1{font-size:31px;line-height:1.3;margin:8px 0 18px}h2{font-size:23px;margin:0 0 15px}h3{font-size:18px}p{margin:12px 0}.eyebrow{letter-spacing:.12em;color:var(--blue);font-size:12px;font-weight:bold}
    .lead{border-top:5px solid var(--blue)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px}.metrics div{border-left:3px solid #8da9bd;padding-left:14px}.metrics b{font:700 25px Georgia,serif;display:block}.metrics span{font-size:13px;color:#4c6072}
    .note{font-size:14px;color:#506171}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.65;margin:16px 0}th,td{border-bottom:1px solid var(--line);text-align:left;padding:10px;vertical-align:top}th{background:#eef3f7;font-weight:600}td:first-child{font-weight:600}table.compact{font-size:12px;white-space:nowrap}a{color:#1d5f91;text-decoration:underline;text-underline-offset:2px}
    figure{margin:24px auto;max-width:570px;text-align:center}figure img{width:510px;max-width:100%;height:auto;background:white}figcaption{font-size:13px;text-align:left;color:#465766;line-height:1.7}.figurelinks{font-size:12px;margin-top:6px}.formula{background:#f5f8fa;border-left:3px solid #688ba4;padding:12px 18px;overflow:auto}li{margin:10px 0}footer{font-size:12px;text-align:center;color:#687987}
    @media(max-width:720px){main{padding:0 10px}section{padding:20px}.metrics{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}h2{font-size:20px}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0}section{break-inside:avoid;border:0;padding:15px 0}.scroll{overflow:visible}table.compact{white-space:normal}figure{break-inside:avoid}}
    """
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E/S抽象价值实验报告</title><style>'+css+'</style></head><body><main>'+''.join(sections)+f'<footer>Generated from retained outputs · {esc(summary["created_utc"])} · No native experiment was executed by this report generator.</footer></main></body></html>'


def figure_html(spec):
    stem = esc(spec["stem"])
    return (f'<figure><img src="{stem}.png" alt="{esc(spec["caption"])}">'
            f'<figcaption>{esc(spec["caption"])}</figcaption><div class="figurelinks">'
            f'<a href="{stem}.pdf">PDF · 3.4 in</a> · <a href="{stem}.svg">SVG</a></div></figure>')


def run(run_root, out):
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    paths = {
        "A：原始语义结果": run_root / "semantics/REPORT.json",
        "A：后置物理/形状审计": run_root / "semantics/POST_AUDIT.json",
        "A：无用载入具体见证": run_root / "semantics/UNUSED_LOAD_WITNESS.json",
        "G0：相关工作与接口审查": run_root / "g0-review.json",
        "B：冻结候选与预测": run_root / "candidates/FROZEN_PREDICTIONS.json",
        "C：最终原生结果分析": run_root / "candidates/ANALYSIS_FINAL.json",
        "Profiling：成本与采集口径": run_root / "PROFILE_COSTS.json",
        "本机硬件记录": run_root / "HARDWARE.json",
        "独立Pilot记录": run_root / "candidates/pilot/PILOT.json",
        "准确度诊断：冻结5轮快照，非最终补轮统计": run_root / "candidates/ACCURACY_SCOPE_NOTE.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Final evidence is required; report not produced: " + ", ".join(missing))
    semantics, audit, witness, g0, manifest, analysis, profile_costs, hardware, pilot, accuracy_note = (read(p) for p in paths.values())
    before = {str(p): digest(p) for p in paths.values()}
    summary = build_summary(semantics, audit, witness, g0, manifest, analysis, list(paths.values()),
                            profile_costs, hardware, pilot, accuracy_note)
    out.mkdir(parents=True, exist_ok=False)
    figures = make_figures(out, summary)
    summary["figures"] = figures
    save_json(out / "SUMMARY.json", summary)
    with (out / "REPORT_zh.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_html(out, summary, semantics, audit, g0, analysis, paths, figures))
        stream.write("\n")
    after = {str(p): digest(p) for p in paths.values()}
    if before != after:
        raise RuntimeError("Input evidence changed while rendering report")
    save_json(out / "REPORT_PROVENANCE.json", dict(schema="es-report-provenance-1", inputs=after,
               generator_sha256=digest(__file__), inputs_unchanged=True, figures=figures,
               outputs={p.name: digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(report=str(out/"REPORT_zh.html"), summary=str(out/"SUMMARY.json"),
                          unique_programs=summary["space"]["unique_programs"],
                          beyond_strong=summary["space"]["beyond_strong_programs"],
                          median_regret_percent=summary["native"]["median_regret_percent"],
                          accept_evidence_complete=False), ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_root, args.out)


if __name__ == "__main__":
    main()
