"""Render the completed three-study campaign without inventing missing evidence.

Report generation is deliberately separate from the runtime and study runners.
Only completed, source-stable receipts are accepted. All performance plots from
the design study are explicitly analytical, and historical timing cohorts never
merge into the current calibrated-model accuracy number.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STUDIES = REPO / "build/dac-evidence-20261023"
DEFAULT_OUT = REPO.parent / "paper-write/第七教学周/experiment-reports/dac-three-studies-20261023"
WORKLOADS = {"dense": "Dense", "moe_hot": "MoE 热点专家", "moe_rotating": "MoE 轮换专家"}
EXECUTION = {"e0": "全部 GPU", "e1": "全部 CPU", "e2": "FFN/专家 CPU", "e3": "Attention CPU", "e4": "因果专家选设备"}
EXPR_NAMES = {
    "R01": "FFN 不等分片、CPU/GPU 执行及结果合并",
    "R02": "多请求 CPU Attention/GPU FFN 流水与跨层预取",
    "R03": "因果专家选设备、共享缓存与预取",
    "R04": "GPU Prefill → CPU Decode，全层 KV 交接与预取",
    "X01": "Attention 细分 + FFN 分片 + 初始化 + CPU 恢复",
    "X02": "多请求异步流水 + 预取 + GPU 恢复",
    "X03": "MoE P/D 交接 + 历史排序 + 私有未来路由 + CPU 恢复",
    "X04": "不同请求采用相反 FFN 分片，共享完整父权重与物理内存池",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(data) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_design(directory: Path, expected_cases: int):
    receipt = load(directory / "completion.json")
    require(receipt.get("complete") is True, f"Incomplete design study: {directory}")
    require(receipt.get("sources_unchanged") is True, "Design execution sources changed")
    require(receipt.get("accuracy_evidence_unchanged") is True, "Frozen accuracy evidence changed")
    require(receipt.get("audit", {}).get("pass_") is True, "Design artifact audit did not pass")
    records = load(directory / "records.json")
    require(receipt["records_sha256"] == canonical_sha(records), "Design records hash mismatch")
    require(receipt["protocol_sha256"] == canonical_sha(load(directory / "protocol.json")), "Design protocol hash mismatch")
    require(len(records) == receipt["cases"] == expected_cases, "Design denominator mismatch")
    require(sum(r.get("success") is True for r in records) == receipt["success"], "Design success count mismatch")
    require(dict(Counter(r["status"] for r in records)) == receipt["statuses"], "Design statuses mismatch")
    require(all(r.get("calibrated") is False and r.get("hardware_measurement") is False for r in records), "Unexpected measured data in analytical campaign")
    groups = load(directory / "analysis.json")["groups"]
    verify_analysis(records, groups)
    return receipt, records, groups


def verify_analysis(records, groups):
    """Bind each unsealed analysis value back to receipt-bound raw records."""
    fields = ("workload", "decode_steps", "initial_state", "vram_gib", "link_gbs")
    by_group = {}
    for row in records:
        key = f"{row['workload']}-d{row['decode_steps']}-{row['initial_state']}-v{row['vram_gib']}-b{row['link_gbs']}"
        by_group.setdefault(key, []).append(row)
    require(len(groups) == len(by_group), "Analysis group count mismatch")
    require(len({g['group'] for g in groups}) == len(groups), "Duplicate analysis group")
    for group in groups:
        require(group['group'] in by_group, "Unknown analysis group")
        rows = by_group[group['group']]
        valid = [r for r in rows if r['success']]
        require(group['feasible'] == bool(valid), "Analysis feasibility mismatch")
        if not valid:
            require(group['statuses'] == dict(Counter(r['status'] for r in rows)), "Analysis failed statuses mismatch")
            continue
        require(all(group[f] == rows[0][f] for f in fields), "Analysis identity mismatch")
        finite_best = min(r['completed_makespan_ns'] for r in valid)
        exec_choice = group['choices']['execution_only_s0']['candidate'].split('-')[0]
        eligible = {
            'joint': rows,
            'execution_only_s0': [r for r in rows if r['candidate'].endswith('-s0')],
            'storage_only_e0': [r for r in rows if r['candidate'].startswith('e0-')],
            'stagewise_execution_then_storage': [r for r in rows if r['candidate'].startswith(exec_choice+'-')],
        }
        for method, candidates in eligible.items():
            choice = group['choices'][method]
            available = [r for r in candidates if r['success']]
            require(available and choice is not None, "Missing feasible analysis choice")
            selected = next((r for r in available if r['candidate'] == choice['candidate']), None)
            require(selected is not None, "Analysis chooses ineligible candidate")
            reference = min(available, key=lambda r:(r['completed_makespan_ns'],r['candidate']))
            require(selected['candidate'] == reference['candidate'], "Analysis winner mismatch")
            require(choice['latency_ns'] == selected['completed_makespan_ns'], "Analysis latency mismatch")
            expected_regret = selected['completed_makespan_ns'] / finite_best - 1
            require(abs(choice['regret_vs_finite_best'] - expected_regret) < 1e-12, "Analysis regret mismatch")
            evaluated = {r['candidate'] for r in candidates}
            if method == 'stagewise_execution_then_storage':
                evaluated |= {r['candidate'] for r in eligible['execution_only_s0']}
            require(choice['evaluations'] == len(evaluated), "Analysis evaluation count mismatch")
        require(set(group['tied_best']) == {r['candidate'] for r in valid if r['completed_makespan_ns'] == finite_best}, "Analysis tie set mismatch")
        center_rows = [r for r in records if all(r[f] == rows[0][f] for f in fields[:3]) and r['vram_gib'] == 12 and r['link_gbs'] == 8 and r['success']]
        if center_rows:
            center = min(center_rows, key=lambda r:(r['completed_makespan_ns'],r['candidate']))
            require(group['center_frozen_candidate'] == center['candidate'], "Frozen center identity mismatch")
            target = next((r for r in rows if r['candidate'] == center['candidate']), None)
            if target and target['success']:
                require(group['center_frozen_status'] == 'success', "Frozen center status mismatch")
                require(abs(group['center_frozen_regret'] - (target['completed_makespan_ns']/finite_best - 1)) < 1e-12, "Frozen center regret mismatch")
            else:
                require(group['center_frozen_status'] == 'candidate_infeasible' and group['center_frozen_regret'] is None, "Frozen center missing candidate mismatch")
        else:
            require(group['center_frozen_status'] == 'center_infeasible_or_absent' and group['center_frozen_candidate'] is None and group['center_frozen_regret'] is None, "Frozen center absent mismatch")


def read_campaign(expression: Path, selection: Path, main: Path, long: Path):
    expr = load(expression / "summary.json")
    require(expr.get("complete") is True and expr.get("source_unchanged") is True, "Expression study incomplete or changed")
    require(len(expr["cases"]) == 8, "Expression denominator changed")
    require(expr["manifest_sha256"] == sha(expression / "manifest.json"), "Expression manifest hash mismatch")
    sel = load(selection / "results.json")
    receipt = load(selection / "completion.json")
    require(sel.get("execution_completed") is True and receipt.get("execution_completed") is True, "Selection study incomplete")
    require(receipt.get("input_hashes_unchanged") is True, "Selection evidence changed")
    require(receipt["result_sha256"] == sha(selection / "results.json"), "Selection results hash mismatch")
    require(sel["protocol_sha256"] == sha(selection / "protocol.json"), "Selection protocol hash mismatch")
    main_r, main_rows, main_g = read_design(main, 504)
    long_r, long_rows, long_g = read_design(long, 112)
    keys = [(r["workload"], r["decode_steps"], r["initial_state"], r["vram_gib"], r["link_gbs"], r["candidate"]) for r in main_rows + long_rows]
    require(len(keys) == len(set(keys)) == 616, "Duplicate or missing design cases")
    require({r["decode_steps"] for r in main_rows} == {1}, "Primary matrix must be D1")
    require({r["decode_steps"] for r in long_rows} == {8, 32}, "Long matrix must contain D8 and D32")
    require(all(r["vram_gib"] == 12 and r["link_gbs"] == 8 for r in long_rows), "Long matrix hardware changed")
    return expr, sel, main_r, long_r, main_rows + long_rows, main_g + long_g


def pct(value):
    return f"{100 * value:.3f}%"


def paired_prefetch(records):
    """Same execution and boundary; vary only the prefetch switch."""
    inventory = {}
    for row in records:
        execution, storage = row['candidate'].split('-')
        key = (row['workload'],row['decode_steps'],row['initial_state'],row['vram_gib'],row['link_gbs'],execution)
        pair=inventory.setdefault(key,{})
        require(storage not in pair,'Duplicate prefetch coordinate')
        pair[storage]=row
    comparisons=[]
    for key,pair in sorted(inventory.items()):
        require(set(pair)=={'s0','s1'},'Incomplete prefetch pair')
        base, prefetch = pair['s0'],pair['s1']
        require(base['shared_input_sha256']==prefetch['shared_input_sha256'],'Prefetch pair workload/hardware mismatch')
        result=dict(zip(('workload','decode_steps','initial_state','vram_gib','link_gbs','execution'),key))
        result['s0_status']=base['status'];result['s1_status']=prefetch['status']
        if base['success'] and prefetch['success']:
            a,b=base['completed_makespan_ns'],prefetch['completed_makespan_ns']
            result.update(status='faster' if b<a else 'slower' if b>a else 'tie',
                s0_ns=a,s1_ns=b,delta_ns=b-a,relative_delta=b/a-1,
                s0_h2d_bytes=base['h2d_bytes'],s1_h2d_bytes=prefetch['h2d_bytes'],
                h2d_delta_bytes=prefetch['h2d_bytes']-base['h2d_bytes'])
        elif base['status']==prefetch['status']=='initial_capacity_infeasible':
            result['status']='initial_capacity_infeasible'
        else:
            result['status']='not_comparable'
        comparisons.append(result)
    return comparisons


def csv_file(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


class Document:
    def __init__(self):
        self.html, self.md = [], []

    def title(self, text, level=2):
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")
        self.md.append(f"{'#' * level} {text}\n")

    def para(self, text, kind=""):
        self.html.append(f'<p class="{kind}">{html.escape(text)}</p>')
        self.md.append(text + "\n")

    def table(self, headers, rows):
        self.html.append("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows) + "</tbody></table></div>")
        self.md.append("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(str(x).replace("|", " / ") for x in r) + " |" for r in rows) + "\n")

    def figure(self, filename, caption):
        self.html.append(f'<figure><a href="{filename}.svg"><img src="{filename}.png" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}（点击查看 SVG）</figcaption></figure>')
        self.md.append(f"![{caption}]({filename}.png)\n\n{caption}\n")

    def link(self, label, href):
        self.html.append(f'<p><a href="{html.escape(href)}">{html.escape(label)}</a></p>')
        self.md.append(f"[{label}]({href})\n")


def setup_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"})
    return plt


def plots(out: Path, sel, groups, records, *, main_only_preview=False):
    plt = setup_plots()
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    steady = sel["steady_oct22"]
    cases = steady["cases"]
    fig, ax = plt.subplots(figsize=(12, 6.7))
    for i, case in enumerate(cases):
        color = "#167565" if case["all_gates"] else "#B94835"
        values = [100 * x for x in case["process_mean_apes"]]
        ax.scatter(values, [i] * len(values), c=color, s=38, alpha=.8)
        ax.plot([min(values), max(values)], [i, i], color=color, alpha=.35)
    ax.axvline(10, color="#8c8c8c", linestyle="--", lw=1, label="单进程均值误差门槛 10%")
    ax.set_yticks(range(len(cases)), [c["case_id"] + ("  [全门槛通过]" if c["all_gates"] else "  [仍有未通过门槛]") for c in cases])
    ax.invert_yaxis()
    ax.set_xlabel("绝对相对误差（%；每点 = 一个独立 heldout 进程的均值）")
    ax.set_title("10 月 22 日当前校准：11 个用例 × 3 个独立进程\n均值误差低，不代表稳定性与策略选择均已通过", loc="left", pad=16)
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False)
    save(fig, "01-current-accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = ["链式执行\n23 组", "串行 / 重叠\n36 组"]
    cohorts = [sel["chain_oct18"], sel["overlap_oct18"]]
    x = np.arange(2)
    axes[0].bar(x - .17, [c["selection"]["winner_matches"] for c in cohorts], .33, label="选中观测最快策略", color="#355C88")
    axes[0].bar(x + .17, [c["selection"]["original_all_gates_passed"] for c in cohorts], .33, label="全部候选满足原有门槛", color="#27816D")
    for i, c in enumerate(cohorts):
        for shift, key in [(-.17, "winner_matches"), (.17, "original_all_gates_passed")]:
            value = c["selection"][key]
            axes[0].text(i + shift, value + .6, str(value), ha="center")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("比较组数")
    axes[0].set_ylim(0, 40)
    axes[0].legend(loc="upper left")
    for i, c in enumerate(cohorts):
        values = sorted(100 * g["selection_regret"] for g in c["groups"])
        axes[1].scatter([i + (j - len(values)/2) * .008 for j in range(len(values))], values, c=["#355C88", "#27816D"][i], alpha=.7, s=35)
        axes[1].text(i + .14, max(values), f"最坏 {max(values):.2f}%", va="center")
    axes[1].set_xticks(x, names)
    axes[1].set_xlim(-.45, 1.8)
    axes[1].set_ylabel("选择损失：所选实测时间 / 实测最快时间 − 1（%）")
    axes[1].grid(axis="y", alpha=.18)
    fig.suptitle("10 月 18 日历史数据的策略选择复核：不代表当前模型的选择准确率", x=.05, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, "02-historical-selection")

    color_list = ["#E1EDF7", "#B4D8BC", "#F8D7A8", "#C3BFE0", "#88CFCD", "#EEEEEE"]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5,5.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    for i, initial in enumerate(["cold", "gpu_resident"]):
        for j, workload in enumerate(WORKLOADS):
            ax = axes[i, j]
            grid = np.full((3, 3), 5.)
            for row, capacity in enumerate([2, 4, 12]):
                for col, bandwidth in enumerate([2, 8, 24]):
                    group = next(g for g in groups if g.get("group") == f"{workload}-d1-{initial}-v{capacity}-b{bandwidth}")
                    if group["feasible"]:
                        best = group["choices"]["joint"]
                        grid[row, col] = int(best["candidate"][1])
                        text = f"{best['candidate']}\n{best['latency_ns']/1e6:.1f} ms"
                    else:
                        text = "初始放置\n容量不足"
                    ax.text(col, row, text, ha="center", va="center", fontsize=9)
            ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            ax.set_xticks(range(3), [2, 8, 24])
            ax.set_yticks(range(3), [2, 4, 12])
            ax.set_xlabel("传输带宽 GB/s")
            ax.set_ylabel("VRAM GiB")
            ax.set_title(WORKLOADS[workload] + (" · DRAM 初始" if initial == "cold" else " · 仅 GPU 初始驻留"))
    fig.suptitle("分析假设下的 P8D1 策略选择地图（同一格内比较相同初始状态）", x=.05, ha="left", y=1.01)
    fig.text(.05, -.01, "e0 全 GPU｜e1 全 CPU｜e2 FFN/专家 CPU｜e3 Attention CPU｜e4 因果专家选设备；s0 无预取 / s1 预取。并列时按编号取一个。", fontsize=9)
    fig.tight_layout()
    save(fig, "03-hardware-map")
    if main_only_preview:
        return

    colors = ["#355C88", "#27816D", "#B2762F", "#8C6CAD", "#C74B4B"]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11))
    for i, workload in enumerate(WORKLOADS):
        for j, initial in enumerate(["cold", "gpu_resident"]):
            ax = axes[i,j]
            n = 4 if workload == "dense" else 5
            for e in range(n):
                subset = sorted([r for r in records if r["workload"] == workload and r["initial_state"] == initial and r["vram_gib"] == 12 and r["link_gbs"] == 8 and r["candidate"] == f"e{e}-s0"], key=lambda r:r["decode_steps"])
                require(len(subset) == 3 and all(r["success"] for r in subset), "Length plot is incomplete")
                ax.plot([r["decode_steps"] for r in subset], [r["makespan_ms"] for r in subset], marker="o", color=colors[e], label=f"e{e}: {EXECUTION[f'e{e}']}")
            best_groups = sorted([g for g in groups if g.get("feasible") and g.get("workload") == workload and g.get("initial_state") == initial and g.get("vram_gib") == 12 and g.get("link_gbs") == 8], key=lambda g:g["decode_steps"])
            ax.plot([g["decode_steps"] for g in best_groups], [g["choices"]["joint"]["latency_ns"]/1e6 for g in best_groups], "k--", lw=2.2, label="完整候选集最佳（含预取）")
            ax.set_title(WORKLOADS[workload] + (" · DRAM 初始" if initial == "cold" else " · 仅 GPU 初始驻留"))
            ax.set_xticks([1,8,32])
            ax.set_xlabel("Decode 轮数 D（Prefill 固定 8 tokens）")
            ax.set_ylabel("有限执行总耗时 ms（对数轴）")
            ax.set_yscale("log")
            ax.grid(alpha=.18)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("12 GiB / 8 GB/s：一次初始放置与后续复用共同决定选择（分析假设）", x=.05, ha="left", y=1.005)
    fig.tight_layout()
    save(fig, "04-generation-length")

    feasible = [g for g in groups if g["feasible"]]
    methods = [("execution_only_s0", "只选执行\n固定无预取"), ("storage_only_e0", "只选存储\n固定全 GPU"), ("stagewise_execution_then_storage", "先执行\n再存储")]
    fig, axes = plt.subplots(1,2,figsize=(12.5, 5))
    max_regret=max(g['choices'][key]['regret_vs_finite_best']*100 for g in feasible for key,_ in methods)
    for i,(key,label) in enumerate(methods):
        vals = sorted(g["choices"][key]["regret_vs_finite_best"] * 100 for g in feasible)
        axes[0].scatter([i + (k-len(vals)/2)*.006 for k in range(len(vals))], vals, s=20, alpha=.55, color=colors[i])
        axes[0].text(i, max(vals)+max(1,max_regret*.035), f"{sum(v>1e-8 for v in vals)}/{len(vals)} 组有损失\n最大 {max(vals):.3f}%", ha="center", fontsize=9)
    axes[0].set_xticks(range(3), [m[1] for m in methods])
    axes[0].set_ylabel("相对完整候选集最佳的耗时增幅（%）")
    axes[0].set_ylim(-max(1,max_regret*.05),max(5,max_regret*1.25))
    axes[0].grid(axis="y",alpha=.18)
    transfer = [g for g in feasible if g.get("decode_steps") == 1 and g.get("center_frozen_status") == "success"]
    for i,w in enumerate(WORKLOADS):
        gs=[g for g in transfer if g["workload"]==w]
        vals=[g["center_frozen_regret"]*100 for g in gs]
        axes[1].scatter([i+(k-len(vals)/2)*.018 for k in range(len(vals))],vals,s=28,color=colors[i],alpha=.65)
        axes[1].text(i,max(vals)+1.2,f"最坏 {max(vals):.2f}%",ha="center",fontsize=9)
    axes[1].set_xticks(range(3), list(WORKLOADS.values()))
    axes[1].set_ylabel("沿用 12 GiB / 8 GB/s 所选策略的损失（%）")
    axes[1].set_ylim(bottom=-2)
    axes[1].grid(axis="y",alpha=.18)
    axes[0].set_title("限制候选维度的代价")
    axes[1].set_title("硬件改变后是否应重新选择")
    fig.suptitle("有限枚举的设计分析；完整集合天然不劣于子集，不能据此声称搜索算法创新",x=.04,ha="left",y=1.02)
    fig.tight_layout()
    save(fig,"05-selection-boundaries")


def prefetch_waterfall(out: Path, records, consumer_audit, sources):
    """Post-run visualization of actual saved intervals, with conservative labels."""
    plt=setup_plots()
    from matplotlib.patches import Patch
    worst=max((p for p in paired_prefetch(records) if p['status']=='slower'),key=lambda p:p['relative_delta'])
    fields=('workload','decode_steps','initial_state','vram_gib','link_gbs')
    rows=[r for r in records if all(r[k]==worst[k] for k in fields) and r['candidate'].startswith(worst['execution']+'-')]
    require(len(rows)==2,'Waterfall pair missing')
    lookup={tuple(r[k] for k in (*fields,'candidate')):r for r in consumer_audit['records']}
    fig,axes=plt.subplots(2,1,figsize=(12.5,6.6),sharex=True)
    entries=[]
    max_time=max(r['completed_makespan_ns'] for r in rows)/1e6
    for ax,row in zip(axes,sorted(rows,key=lambda r:r['candidate'])):
        directory=sources['design-main' if row['decode_steps']==1 else 'design-long']
        event_path=directory/row['artifact_directory']/'events.json'
        events=load(event_path)
        evidence=lookup[tuple(row[k] for k in (*fields,'candidate'))]
        unused=set(evidence['bytes_by_tensor_without_gpu_consumer'])
        entries.append({'candidate':row['candidate'],'events_path':str(event_path.resolve()),'events_sha256':sha(event_path),'events':events,'tensors_without_gpu_consumer':sorted(unused),'audit_scope':consumer_audit['scope']})
        for event in events:
            if event['kind'] not in {'COMPUTE','COPY'}:
                continue
            start,end=event['start_ns']/1e6,event['end_ns']/1e6
            if event['kind']=='COMPUTE':
                lane=0 if event['device']=='cpu' else 1
                color='#355C88' if lane==0 else '#27816D'
            else:
                lane=2
                color='#BD4B3D' if event['dst']=='vram' and event['target_id'] in unused else '#8A98A8'
            ax.broken_barh([(start,end-start)],(lane-.32,.64),facecolors=color,edgecolors='none')
        total=row['completed_makespan_ns']/1e6
        ax.axvline(total,color='#35475B',linestyle='--',lw=1)
        ax.text(total+max_time*.01,-.65,f'{total:.3f} ms',ha='left',fontsize=10)
        ax.set_yticks([0,1,2],['CPU 计算','GPU 计算','CPU/GPU 搬运'])
        ax.set_ylim(2.6,-.9)
        ax.set_title(row['candidate']+(' · 不预取' if row['candidate'].endswith('s0') else ' · 开启预取'),loc='left')
        ax.grid(axis='x',alpha=.15)
    axes[-1].set_xlim(0,max_time*1.14)
    axes[-1].set_xlabel('模拟时间 ms；两幅图使用相同横轴')
    fig.suptitle(f"预取与 CPU 专家放置的冲突：{WORKLOADS[worst['workload']]} P8D{worst['decode_steps']}，{worst['vram_gib']} GiB / {worst['link_gbs']} GB/s\nF32 分析假设；红色张量在整个有限轨迹内没有 GPU 计算消费者",x=.06,ha='left',y=1.04)
    fig.legend(handles=[Patch(color='#355C88',label='CPU 计算'),Patch(color='#27816D',label='GPU 计算'),Patch(color='#8A98A8',label='其他传输'),Patch(color='#BD4B3D',label='对应张量没有 GPU 计算消费者')],loc='lower center',ncol=4,bbox_to_anchor=(.52,-.055),frameon=False,fontsize=9)
    fig.tight_layout()
    for suffix in ['png','svg']:
        fig.savefig(out/f'06-prefetch-waterfall.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    dump(out/'evidence/waterfall-source.json',{'schema':'paper-prefetch-waterfall-1','pair':worst,'inputs':entries,'interpretation':'Post-run diagnostic; not an online future oracle, not proof all latency increase is solely due to extra bytes.'})


def build_document(expr, sel, mr, lr, records, groups, sources, consumer_audit=None):
    doc = Document()
    feasible = [g for g in groups if g["feasible"]]
    passed = sum(c["status"] == "pass" for c in expr["cases"])
    steady = sel["steady_oct22"]
    status = Counter(r["status"] for r in records)
    doc.title("三组实验：表达是否可信、选择是否可靠、硬件变化是否值得重选",1)
    doc.para("报告日期：2026-10-24。执行模型与校准参数保持冻结；本轮增加实验与证据复核。", "meta")
    doc.para("基础设施足以完成这三组实验，但实验完成与论文主张成立是两件事：组合表达已经得到有限验证；当前完整单层的策略选择仍缺匹配实测；硬件与策略分析已经得到可解释的模拟结果。", "lead")
    doc.table(["问题", "本轮完成", "可以说什么", "还不能说什么"],[
        ["1. 表达与编译保真",f"{passed}/8：4 个参考 + 4 个新组合", "同一套 IR/编译/状态账本能组合这些已支持机制", "覆盖任意策略，或比其他工具更易用"],
        ["2. 选择可信度", "复核 33 个当前进程 + 59 组历史策略比较", "当前有限形状的时延校准；7 组历史重叠选择通过全部门槛", "当前完整单层已经能可靠选优"],
        ["3. 设计空间",f"616 个声明案例：{status.get('valid',0)} 可执行，{status.get('initial_capacity_infeasible',0)} 初始容量不可行", "给定假设与有限候选，展示选择边界和复用影响", "真实硬件性能排名，或新的最优搜索算法"],
    ])
    doc.title("我们在抽象什么，这一轮具体验证什么")
    doc.para("把两件事放到同一个可执行语义里：计算怎样拆分、放在哪个设备执行；同一份物理张量的副本怎样驻留、搬运、预取和释放。编译后，它们共同服从依赖、物理容量与设备资源约束。MoE 的未来路由由 workload 揭示，策略只能使用已知信息。")
    doc.para("例如，把 FFN 的计算分成两部分，并不必然把权重搬运量减半。若两部分是完整父权重的视图，仍要承认完整父权重的物理占用；若是真实物化的子权重，则需要另一套存储身份与成本。X04 专门检查这个区别。")
    doc.para("这是一项候选方法学主张。运行通过证明有限语义与组合可行；要证明相对已有工作的独特性，还需要公平的外部工具对照。")
    doc.title("实验一：八个组合能否通过同一条编译路径")
    doc.table(["案例", "组合", "结果", "recipe/helper 摘录行数 / AST 语句"],[[c["id"],EXPR_NAMES[c["id"]],"通过" if c["status"]=="pass" else c["status"],f"{c['authoring']['nonblank_noncomment_lines']} / {c['authoring']['ast_statements']}"] for c in expr["cases"]])
    doc.para("检查的不是只有程序有没有跑完：手工几何与操作数约束检查输入张量、分片边界、合并依赖和实际搬运；事件检查实际容量与计算顺序；两条未来路由不同的 MoE 轨迹检查公共前缀一致。")
    x3=next(c for c in expr["cases"] if c["id"]=="X03")["evidence"]
    x4=next(c for c in expr["cases"] if c["id"]=="X04")["evidence"]
    doc.para(f"具体证据：X03 在私有未来路由分叉前有 {x3['identical_prefix_decisions']} 个相同公共决策。X04 检查 {x4['exact_operand_contracts']} 项精确操作数契约；独立回放共享物理池，确认切片没有凭空变成较小权重，也没有虚构跨设备 COPY。")
    doc.para(f"{expr['source_file_count']} 份冻结源文件未改变；这八个例子无需修改后端或手工改低层图。行数是选定 recipe/helper 的摘录，包含工作负载、价格和检查，但不递归统计全部依赖。它不是纯用户代码量或完整实现工作量；没有人工用时或公平低层基线，因而不报告压缩倍数。")
    doc.para("首轮 expression-v1 保留 7/8：X02 在实验脚本中使用了不存在的 API 参数。改成既有的服务率 0/1 表达后，同一八项要求在 v2 全部通过；运行时没有为此改变。这些新组合由作者设计，不是盲测新算法。")
    doc.title("实验二：误差不大，是否就能选对策略")
    doc.para("不一定。预测一个方案的耗时，与在相同工作负载上从多个方案里选优，需要不同的数据。选优要求候选的形状、输入、精度、初始驻留和最终输出边界一致，不能把三个不同尺寸当成三种策略。")
    doc.table(["数据", "误差单位 / 数量", "中位数", "P90", "最坏", "全部门槛通过"],[
        ["10 月 22 日当前校准", "独立进程均值 / 33", pct(steady['process_mean_ape']['median']),pct(steady['process_mean_ape']['p90']),pct(steady['process_mean_ape']['worst']),f"{steady['all_gates_passed']}/11 用例"],
        *[[label, f"路径合并中位数 / {sel[key]['summary']['results']}", *[pct(sel[key]['summary']['pooled_median_ape'][k]) for k in ['median','p90','worst']],f"{sel[key]['summary']['original_all_gates_passed']}/{sel[key]['summary']['results']} 路径"] for key,label in [('chain_oct18','10 月 18 日历史链式'),('overlap_oct18','10 月 18 日历史重叠')]],
    ])
    doc.para("三行的误差单位、冻结预测器与工作负载范围不同，不能合并成一个准确率。10 月 18 日的大误差或选择损失不表示 10 月 22 日模型退化，也不能用后者的低误差替旧策略选择补合格证明。当前 9/11 用例通过误差门槛，7/11 通过包括稳定性与观测开销在内的完整门槛；三个完整 Dense 单层分别为 2/3 与 1/3。每点是批量重复后的进程均值，不是单次调用尾延迟。")
    doc.figure("01-current-accuracy","当前校准的 33 个进程误差；连线只是三次点值跨度，不是置信区间。颜色表示完整门槛，虚线仅表示均值误差门槛")
    doc.para("当前完整单层选择：不可评估。三个不同形状各只有一种策略，没有同形状的多策略候选。这个缺口不会因为平均误差较小而自动消失。", "warning")
    doc.table(["历史匹配比较", "选中观测最快", "损失中位数 / P90 / 最坏", "所有候选通过原门槛"],[[label,f"{sel[k]['selection']['winner_matches']}/{sel[k]['selection']['groups']}"," / ".join(pct(sel[k]['selection']['selection_regret'][x]) for x in ['median','p90','worst']),f"{sel[k]['selection']['original_all_gates_passed']}/{sel[k]['selection']['groups']}"] for k,label in [('chain_oct18','驻留链式执行'),('overlap_oct18','串行 / 计算搬运重叠')]])
    doc.figure("02-historical-selection","历史选择结果单独展示；高命中数不能覆盖失败的稳定性门槛")
    doc.para("选择损失 = 所选策略的实测时间 / 所有同条件候选的实测最短时间 − 1。链式最坏案例把 283.2 μs 的 CPU 方案换成 552.5 μs 的 hybrid，损失 95.09%；这是历史旧预测器的失败，不是当前模型的测量。重叠最坏 53.86%，三个进程的粗略配对重采样区间跨零，不能据此断言重叠普遍更差。")
    doc.para("保留所有不稳定候选，不筛掉失败者再重新排名。7 组重叠比较的所有候选均通过原门槛，且粗略过程簇区间区分了赢家；按四条路径都通过计，只有 3/18 个 overlap 用例通过。未新增测量、未拟合参数、未放宽门槛。")
    doc.title("实验三：容量、带宽、初始放置和复用长度如何改变选择")
    doc.para("本组全部使用明确声明的分析成本：两层 F32，H6144/I16384，24 个 query heads、6 个 KV heads；MoE 为 4 专家/top-2。CPU 为 0.6 TFLOP/s、50 GB/s，GPU 为 8 TFLOP/s、300 GB/s；额外 workspace 假设为 0。它们不是某台机器的实测参数，结果不用于宣传 GPU、NPU 或厂商产品性能。", "warning")
    doc.table(["变量", "取值"],[["容量 × 传输带宽", "VRAM 2/4/12 GiB × 2/8/24 GB/s，DRAM 32 GiB"],["工作负载", "Dense、MoE 热点、MoE 轮换；Prefill 8 tokens"],["初始状态", "所有权重仅在 DRAM；或所有权重仅在 VRAM。KV 初始为空"],["执行候选", "e0 全 GPU；e1 全 CPU；e2 FFN/专家 CPU；e3 Attention CPU；MoE 加 e4 因果专家选设备"],["存储候选", "s0 LRU 无预取；s1 同预算 LRU + 下一层最多两个权重预取"],["长度", "主矩阵 D1；参考硬件 12 GiB/8 GB/s 补 D8、D32"],["统一结果边界", "有限 P8+D 执行与必要最终 hidden/KV 到 DRAM；不是 TTFT/TPOT"]])
    doc.para(f"主矩阵 504 条，长度扩展 112 条，共 616 条；{status.get('valid',0)} 条有效执行，{status.get('initial_capacity_infeasible',0)} 条初始容量不可行。初始不可行被保留，不当成策略执行失败。Dense 权重约 2.95 GiB，MoE 约 9.70 GiB，因此部分 GPU 初始驻留组合从一开始就放不下。")
    doc.para("GPU 初始驻留只表示权重事先放在 VRAM，CPU 需要时仍要拿回 DRAM。没有免费的双份副本，也没有从冷启动总时间里减掉搬运来制造暖态。跨初始状态的差异只能解释给定起点的结果，不能称为一般稳态吞吐。")
    doc.table(["边界", "实际约定及影响"], [["输入与 Decode", "Decode 是给定输入轨迹，不是实际采样。CPU 定价的 trace_next_token 消费前一轮最终 hidden，因此轮次交接需要 DRAM hidden。"],["输出与 KV", "全流程结束时要求最终 hidden 与所有 KV block 的 DRAM 视图；没有逐层 hidden 的 DRAM 屏障。一直只把 KV 留在 GPU 的终止策略不在本实验边界内。"],["权重终态", "MoE 显式保留不可变权重原始位置的 backing；Dense 没有同样的显式 weight goal_location。两族分别在自己的固定约定内比较，不将结果解释成无约束全局最优。"]])
    doc.figure("03-hardware-map","主矩阵中有限候选的最佳选择；灰格代表同一组所有候选都无法满足初始容量")
    doc.figure("04-generation-length","在固定参考硬件上增加 Decode 轮数；彩色线固定为无预取 s0，黑色虚线允许在完整候选中选择。累计耗时含初始放置后的必要数据搬运")
    comparisons=[]
    for key,label in [('execution_only_s0','只选执行，固定 s0'),('storage_only_e0','只选存储，固定 e0'),('stagewise_execution_then_storage','先执行，再存储')]:
        vals=sorted(g['choices'][key]['regret_vs_finite_best'] for g in feasible)
        comparisons.append([label,f"{sum(x>1e-10 for x in vals)}/{len(vals)}",pct(statistics.median(vals)),pct(max(vals)),"4/5；2；5/6 次（见对应行）"])
    comparisons[0][-1]='Dense 4 / MoE 5'
    comparisons[1][-1]='2'
    comparisons[2][-1]='Dense 5 / MoE 6（复用已算 s0）'
    doc.table(["限制方式", "比完整候选集慢的组数", "损失中位数", "最大损失", "枚举次数"],comparisons)
    doc.para("完整候选集枚举 Dense 8 / MoE 10 次。完整集合的最优值天然不劣于子集；上表衡量候选限制的代价，不证明新搜索算法。若联合与分步相同，也应接受这个负结果，不能凭完整枚举强行宣称协同收益。")
    doc.figure("05-selection-boundaries","候选限制与硬件迁移的损失；每个点来自一个确定性、同条件的比较组")
    pairs=paired_prefetch(records)
    pc=Counter(p['status'] for p in pairs)
    doc.para(f"把执行方式固定，只改变是否预取：共 {len(pairs)} 对，其中 {pc['faster']} 对更快、{pc['tie']} 对相同、{pc['slower']} 对更慢；{pc['initial_capacity_infeasible']} 对因初始容量不足而不比较，另外 {pc['not_comparable']} 对没有可比结果。这比仅看最佳方案更直接地揭示存储动作与执行放置的关系。")
    doc.table(["工作负载", "预取更快", "相同", "更慢", "初始容量不足"],[[WORKLOADS[w],*[sum(p['workload']==w and p['status']==s for p in pairs) for s in ['faster','tie','slower','initial_capacity_infeasible']]] for w in WORKLOADS])
    slower=[p for p in pairs if p['status']=='slower']
    if slower:
        worst_pf=max(slower,key=lambda p:p['relative_delta'])
        doc.para(f"预取最差的一对是 {WORKLOADS[worst_pf['workload']]}，D{worst_pf['decode_steps']}、{worst_pf['initial_state']}、{worst_pf['vram_gib']} GiB / {worst_pf['link_gbs']} GB/s、{worst_pf['execution']}：不开预取 {worst_pf['s0_ns']/1e6:.3f} ms，开启后 {worst_pf['s1_ns']/1e6:.3f} ms，多花 {pct(worst_pf['relative_delta'])}（{worst_pf['delta_ns']/1e6:.3f} ms）；H2D 从 {worst_pf['s0_h2d_bytes']:,} 增至 {worst_pf['s1_h2d_bytes']:,} 字节。")
    doc.para("这里的预取规则按下一层权重工作，不能自动等同于“恰好预取未来将在 GPU 使用的数据”。固定 CPU 执行的部分也可能触发预取。配对结果能确认额外搬运量及净耗时变化；仅凭总量不能把这些字节全部称为“从未使用的数据”，需要进一步逐张量检查消费者。这提示后续设计应联合考虑预取对象与执行放置，但本轮没有因此得到联合枚举优于先后选择的普遍证据。")
    if consumer_audit is not None and slower:
        coord=('workload','decode_steps','initial_state','vram_gib','link_gbs','execution')
        detail=next(p for p in consumer_audit['storage_pairs'] if all(p[k]==worst_pf[k] for k in coord))
        require(detail['evaluated'] and detail['extra_h2d_bytes']==worst_pf['h2d_delta_bytes'],'Consumer audit paired identity mismatch')
        doc.para(f"对最差配对进一步做事后检查：增加的 {detail['extra_h2d_bytes']:,} 字节 H2D 中，{detail['extra_bytes_without_gpu_consumer']:,} 字节来自在整个有限轨迹里没有任何 GPU COMPUTE 消费者的不可变权重张量。这是保守的逐张量诊断，不能识别所有重复拷贝或未来窗口之外的复用价值；完整未来轨迹只给事后评估器，未传给策略。时间差还包含依赖与重叠变化，不将全部延迟归因于字节数。")
        doc.figure('06-prefetch-waterfall','相同执行放置下预取开关的实际事件轨迹；红色依据事后消费者检查，表示有限窗口内对应张量没有 GPU 计算消费者')
        doc.link('瀑布图逐事件数据与来源哈希','evidence/waterfall-source.json')
    dynamic=[r for r in records if r.get('success') and r['candidate']=='e4-s0']
    fixed=[r for r in records if r.get('success') and r['candidate'] in {'e0-s0','e1-s0','e2-s0','e3-s0'}]
    doc.para(f"异构协作也有事件证据：MoE 的 e4-s0 有 {sum(r['cpu_gpu_event_overlap_ns']>0 for r in dynamic)}/{len(dynamic)} 条有效案例在模拟事件记录中出现 CPU/GPU 计算区间重叠，累计重叠区间最多 {max(r['cpu_gpu_event_overlap_ns'] for r in dynamic)/1e6:.3f} ms；固定 e0–e3 的 s0 共有 {len(fixed)} 条有效案例，其中 {sum(r['cpu_gpu_event_overlap_ns']>0 for r in fixed)} 条出现重叠。e4 同时改变专家设备选择与搬运需求，不能将全部性能收益单独归因于重叠。")
    transfer=[g for g in feasible if g.get('decode_steps')==1 and g.get('center_frozen_status')=='success']
    worst=max(transfer,key=lambda g:g['center_frozen_regret'])
    doc.para(f"可以明确解释的硬件结论：把参考点（12 GiB、8 GB/s）选出的策略直接搬到其他硬件点，最坏多花 {pct(worst['center_frozen_regret'])} 时间，发生于 {worst['group']}。这是给定成本假设与候选的策略迁移损失；它提示要随硬件重新评估，而不是照搬固定方案。")
    doc.para("跨硬件冻结的是候选 recipe 的规则，不是参考硬件上的完整动作轨迹。若选中 e4，它的 current_cost 因果设备选择仍依据目标硬件的已知成本运行；这里衡量的是沿用同一规则的代价。")
    center=[g for g in feasible if g.get('vram_gib')==12 and g.get('link_gbs')==8]
    doc.table(["工作负载 / 初始", "D1 最佳", "D8 最佳", "D32 最佳"],[[WORKLOADS[w]+(' / DRAM' if init=='cold' else ' / 仅 VRAM'), *[next(g['choices']['joint']['candidate']+' · '+f"{g['choices']['joint']['latency_ns']/1e6:.2f} ms" for g in center if g['workload']==w and g['initial_state']==init and g['decode_steps']==d) for d in [1,8,32]]] for w in WORKLOADS for init in ['cold','gpu_resident']])
    transitions=[]
    for w in ['dense','moe_hot']:
        chosen=[next(g['choices']['joint']['candidate'] for g in center if g['workload']==w and g['initial_state']=='cold' and g['decode_steps']==d) for d in [1,8,32]]
        transitions.append(WORKLOADS[w]+'：'+' → '.join(chosen))
    doc.para('固定参考硬件、DRAM 起点，随 D1 → D8 → D32 的选择变化为：'+'；'.join(transitions)+'。较长窗口可多次复用搬入 GPU 的权重，初始搬运成本的相对影响下降。因此应同时看执行窗口和驻留方式，短窗口最佳不能直接外推到长窗口。')
    hot32={r['candidate']:r for r in records if r['workload']=='moe_hot' and r['decode_steps']==32 and r['initial_state']=='cold' and r['vram_gib']==12 and r['link_gbs']==8 and r['success']}
    if 'e0-s1' in hot32 and 'e4-s0' in hot32:
        doc.para(f"例如 MoE 热点 D32：全 GPU + 预取为 {hot32['e0-s1']['makespan_ms']:.2f} ms，当前成本专家选设备为 {hot32['e4-s0']['makespan_ms']:.2f} ms，后者仍有 {hot32['e4-s0']['cpu_gpu_event_overlap_ns']/1e6:.2f} ms 计算重叠。这说明重叠量本身不是最终目标；基于当前已知成本的因果选择，也不保证整个窗口最佳。该结论仍限于此分析假设和候选集。")
    doc.title("运行这些实验花了多久")
    costs = [r['harness_wall_seconds'] for r in records if r.get('success') and r.get('harness_wall_seconds') is not None]
    runtime_rows=[["504 条主矩阵",f"{mr['elapsed_wall_seconds']:.1f} 秒 wall time"],["112 条长度扩展",f"{lr['elapsed_wall_seconds']:.1f} 秒 wall time"],[f"有效执行 {len(costs)} 条", f"单案例中位数 {statistics.median(costs):.3f} 秒；最慢 {max(costs):.3f} 秒"]]
    for d in [1,8,32]:
        values=[r['harness_wall_seconds'] for r in records if r.get('success') and r['decode_steps']==d]
        runtime_rows.append([f'D{d} 有效执行 {len(values)} 条',f'单案例中位数 {statistics.median(values):.3f} 秒；最慢 {max(values):.3f} 秒'])
    runtime_rows.append([f"初始容量不可行 {status.get('initial_capacity_infeasible',0)} 条",'单独统计；不并入有效执行的时长中位数'])
    doc.table(["范围", "运行成本"],runtime_rows)
    doc.para("以上是模拟器在本机执行实验的墙钟时间，包含编译、运行与产物导出；不是被模拟的 LLM 延迟，也不是纯事件引擎速度或标准化速度基准。批量实验使用两个 worker，同期分析任务也可能影响本机墙钟。长序列增加事件和导出成本，这一轮不能证明快速的大规模搜索或更大层数的可扩展性。")
    doc.title("对论文意味着什么，以及最小的下一步")
    doc.table(["可立即写入", "仍缺的关键证据", "下一步怎样做"],[["有限策略组合、因果性与物理资产语义", "相对已有工具的表达代价/增益", "固定版本的 LLMCompass / LLMServingSim，在共同小子集记录配置、适配器、核心修改；未测不能写不支持"],["现有校准范围、稳定性边界与历史选择结果", "当前完整单层同条件多策略选优", "固定 H512/q64/hist0，先比较全 CPU、全 GPU、CPU Attention/GPU FFN 三个完整宏放置方案；再加入明确标为串行异构 FFN 分片的第四候选"],["确定性设计空间、选择边界、初始容量与复用影响", "分析假设下结论的现实代表性", "先在有限同形状候选完成可信选择，再用于一个真实校准硬件点；其他点保持敏感性分析口径"],["已有高层接口无需改核心即可组合案例", "用户效率或代码压缩优势", "建立匹配低层对照，完整统计辅助代码；没有用户实验就不宣称人力节省"]])
    doc.para("下一组候选所需的原生内核已有对应分支，但仍需新的小型实验编排：共同输入与父权重、相同驻留和最终 hidden/KV 边界、跨方案数值等价、profile 与独立验收分离。现有 executor 串行 launch 并同步动作，FFN 分片依次执行；它不能验证并发 TP、计算搬运重叠或多请求流水的性能收益。新版 attention_core 等变换也不在当前 native 覆盖内。")
    doc.para("还有一个必须公开的前提：steady 的 action_key 绑定完整 program signature。最小可行验证需要每个候选各自采集并冻结 profile，再独立重复执行；它验证的是已 profile 候选的预测与选择，不能宣称从通用算子价格泛化到未测策略。四个候选只跑三轮也不能完全平衡顺序，需要明确采用部分轮转或增加轮次。这些均是下一步计划，本轮没有执行新的测量。")
    doc.para("目前最值得推进的是把“同一抽象下的多个可比策略”与“可信的选择结果”接起来。先补一个小而完整的单层多策略实测闭环，比继续扩大硬件名单或泛化覆盖口号更直接。三组实验完成不等于 DAC 贡献已经充分；外部对照与当前策略选择是仍需补强的两处证据。")
    doc.para("外部对照尚未运行。官方 LLMCompass 已有硬件/软件建模与设计空间探索；LLMServingSim 已有运行时驱动的 serving 模拟和相关策略支持，不能把它们描述成没有策略层。本轮没有建立任何具体组合在这些工具中无法表达的证据。")
    doc.link("LLMCompass 官方仓库", "https://github.com/PrincetonUniversity/LLMCompass")
    doc.link("LLMServingSim 官方仓库", "https://github.com/casys-kaist/LLMServingSim")
    doc.title("复现、数据与证据")
    doc.para("原始案例保留声明、编译 IR、事件、成本与哈希。报告复制便携摘要、CSV 与瀑布图所用的两条轨迹，其余大体积原始轨迹保留在实验目录。验收条件保持原样；程序错误的旧版本保留。最终 report-manifest.json 记录本报告文件的 SHA-256 哈希。")
    for label,href in [("616 行设计记录 CSV","evidence/design-records.csv"),("设计比较组 CSV","evidence/design-groups.csv"),("固定执行的预取配对 CSV","evidence/paired-prefetch.csv"),("历史 59 组策略选择 CSV","evidence/historical-selection.csv"),("33 个当前独立进程误差 CSV","evidence/current-process-errors.csv"),("八项表达案例 CSV","evidence/expression-cases.csv"),("汇总与边界 JSON","evidence/report-summary.json"),("输入文件 SHA-256","evidence/input-hashes.json"),("总协议（English）","evidence/DAC_THREE_STUDIES_PROTOCOL.md")]:
        doc.link(label,href)
    for label,path in sources.items():
        doc.link(label + "：完整原始目录",path.resolve().as_uri())
    return doc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression',type=Path,default=STUDIES/'expression-v2')
    parser.add_argument('--selection',type=Path,default=STUDIES/'selection-v3')
    parser.add_argument('--main',type=Path,default=STUDIES/'design-main-v1')
    parser.add_argument('--long',type=Path,default=STUDIES/'design-long-v1')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--review-evidence',type=Path,action='append',default=[],help='Additional completed audit/test receipt or log; copied and hashed, without altering results')
    args=parser.parse_args(argv)
    expr,sel,mr,lr,records,groups=read_campaign(args.expression,args.selection,args.main,args.long)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    sources={'expression':args.expression,'selection':args.selection,'design-main':args.main,'design-long':args.long}
    inputs=[]
    for name,path in sources.items():
        names={'expression':['summary.json','manifest.json','source-before.json','source-after.json'], 'selection':['results.json','completion.json','protocol.json','software-validation.json'], 'design-main':['completion.json','analysis.json','records.json','protocol.json','independent-artifact-audit.json','source-manifest-before.json','source-manifest-after.json'], 'design-long':['completion.json','analysis.json','records.json','protocol.json','independent-artifact-audit.json','source-manifest-before.json','source-manifest-after.json']}[name]
        for filename in names:
            src=path/filename
            if src.exists():
                dst=evidence/f'{name}-{filename}'
                shutil.copyfile(src,dst)
                inputs.append({'path':str(src.resolve()),'sha256':sha(src),'report_copy':str(dst.relative_to(out))})
    for name in ['DAC_THREE_STUDIES_PROTOCOL','PAPER_EXPRESSION_STUDY','PAPER_SELECTION_STUDY','PAPER_DESIGN_STUDY']:
        src=REPO/'docs'/f'{name}.md';shutil.copyfile(src,evidence/src.name)
        inputs.append({'path':str(src),'sha256':sha(src),'report_copy':str((evidence/src.name).relative_to(out))})
    for index,src in enumerate(args.review_evidence,1):
        require(src.is_file(),f'Missing review evidence: {src}')
        dst=evidence/f'review-{index:02d}-{src.name}'
        shutil.copyfile(src,dst)
        inputs.append({'path':str(src.resolve()),'sha256':sha(src),'report_copy':str(dst.relative_to(out))})
    consumer_audit=None
    for src in args.review_evidence:
        if src.suffix=='.json':
            candidate=load(src)
            if candidate.get('schema')=='paper-prefetch-consumer-audit-1':
                require(candidate.get('completed') is True and candidate['cases']==len(records),'Consumer audit incomplete')
                require(candidate.get('future_information_only_in_post_run_evaluator') is True,'Consumer audit must remain post-run')
                fields=('workload','decode_steps','initial_state','vram_gib','link_gbs','candidate')
                audited={tuple(r[f] for f in fields):r for r in candidate['records']}
                require(len(audited)==len(records),'Consumer audit denominator mismatch')
                for row in records:
                    audit_row=audited[tuple(row[f] for f in fields)]
                    require(audit_row['status']==row['status'] and audit_row['success']==row['success'],'Consumer audit status mismatch')
                    if row['success']:
                        require(audit_row['completed_makespan_ns']==row['completed_makespan_ns'] and audit_row['h2d_bytes']==row['h2d_bytes'],'Consumer audit measurements mismatch')
                consumer_audit=candidate
    csv_file(evidence/'design-records.csv',records)
    flat=[]
    for g in groups:
        row={k:v for k,v in g.items() if k!='choices'}
        for method,value in g.get('choices',{}).items():
            row.update({f'{method}_{k}':v for k,v in value.items()})
        flat.append(row)
    csv_file(evidence/'design-groups.csv',flat)
    pairs=paired_prefetch(records)
    csv_file(evidence/'paired-prefetch.csv',pairs)
    csv_file(evidence/'historical-selection.csv',[{'cohort':cohort,**g} for cohort in ['chain_oct18','overlap_oct18'] for g in sel[cohort]['groups']])
    csv_file(evidence/'current-process-errors.csv',[{'case_id':c['case_id'],'process_index':i,'ape':v,'accuracy_gate':c['accuracy_gate'],'all_gates':c['all_gates']} for c in sel['steady_oct22']['cases'] for i,v in enumerate(c['process_mean_apes'],1)])
    csv_file(evidence/'expression-cases.csv',[{'id':c['id'],'name':c['name'],'group':c['group'],'status':c['status'],**c['authoring']} for c in expr['cases']])
    dump(evidence/'input-hashes.json',inputs)
    dump(evidence/'report-summary.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'expression_passed':sum(c['status']=='pass' for c in expr['cases']),'expression_total':8,'current_accuracy':sel['steady_oct22']['process_mean_ape'],'current_whole_layer_selection':'not_evaluable','historical_selection':{k:sel[k]['selection'] for k in ['chain_oct18','overlap_oct18']},'design_cases':len(records),'design_statuses':dict(Counter(r['status'] for r in records)),'paired_prefetch_statuses':dict(Counter(p['status'] for p in pairs)),'design_calibrated':False,'runtime_sources_unchanged':expr['source_unchanged'] and mr['sources_unchanged'] and lr['sources_unchanged'],'design_complete_receipts':[mr,lr],'external_tool_comparison_executed':False,'matched_authoring_baseline_executed':False})
    plots(out,sel,groups,records)
    if consumer_audit is not None:
        prefetch_waterfall(out,records,consumer_audit,sources)
    doc=build_document(expr,sel,mr,lr,records,groups,sources,consumer_audit)
    review_names={p.name for p in args.review_evidence}
    if {'REVIEW.md','FINAL_REVIEW.md','final-review.json'}.issubset(review_names):
        doc.para('附件 review.json / REVIEW.md 形成于长序列实验运行期间，其中“待完成”是当时的状态记录。现已由 FINAL_REVIEW.md、final-review.json 与长序列独立审计补齐；旧记录保留，以最终完成审计判断本轮完整性。')
    if 'SCALING_NOTE.md' in review_names:
        doc.para('SCALING_NOTE.md 与 SCALING_EVIDENCE.json 保留在长实验 92/112 时采集的扩展性快照，其中 Dense 与 MoE 热点的 36 个 D8/D32 配对已经完整。它不代表全量扩展性验收；正文运行成本表使用最终全部 112 条长实验记录。')
    for index,src in enumerate(args.review_evidence,1):
        doc.link(f'独立复核 / 软件检查：{src.name}',f'evidence/review-{index:02d}-{src.name}')
        if src.name == 'all-study-tests.log':
            log = src.read_text(encoding='utf-8-sig')
            match = re.search(r'Ran (\d+) tests in',log)
            if match and log.rstrip().endswith('OK'):
                doc.para(f'合并软件检查：{match.group(1)}/{match.group(1)} 项通过，包含研究脚本、独立事件审计与报告输入的防误用检查。这个数字是软件检查项数，不是实测性能通过数。')
        elif src.suffix == '.json' and src.name.startswith('independent-design-audit-'):
            audit=load(src)
            require(audit.get('passed') is True,'Attached independent design audit failed')
            counts=audit['counts']
            doc.para(f"独立设计审计 {src.name}：{counts['declared_cases']} 条声明案例，回放 {counts['events']:,} 个事件，检查 {counts['macro_operations_checked']:,} 项宏操作与 {counts['dependency_edges_checked']:,} 条依赖。")
    style="""body{font:16px/1.8 'Microsoft YaHei',sans-serif;color:#24344a;background:#f2f5f8;margin:0}main{max-width:1240px;margin:30px auto;background:white;padding:42px 50px;border-top:7px solid #244c78}h1{font-size:30px;color:#153955;line-height:1.45}h2{font-size:23px;margin-top:42px;padding-top:15px;border-top:1px solid #ccd6e0;color:#153955}.lead{font-size:19px;padding:18px 22px;background:#eaf3f3;border-left:5px solid #237562}.warning{padding:15px 20px;background:#fff4e8;border-left:5px solid #bb7b2d}.meta{color:#63788e;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#e8eff7;color:#254b70;text-align:left}td,th{border:1px solid #d8e1ea;padding:10px 12px;vertical-align:top}tr:nth-child(even){background:#f8fafc}figure{margin:25px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#5f7285}a{color:#1b6398} @media(max-width:800px){main{margin:0;padding:20px}h1{font-size:25px}table{min-width:640px}}"""
    output='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>三组实验报告</title><style>'+style+'</style><main>'+''.join(doc.html)+'</main></html>'
    (out/'REPORT_zh.html').write_text(output,encoding='utf-8',newline='\n')
    (out/'REPORT_zh.md').write_text('\n'.join(doc.md),encoding='utf-8',newline='\n')
    shutil.copyfile(Path(__file__),evidence/'paper_evidence_report.py')
    dump(out/'report-manifest.json',{'schema':'paper-evidence-report-1','generated_utc':datetime.now(timezone.utc).isoformat(),'source_hash':sha(Path(__file__)),'files':{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report-manifest.json'}})
    print(json.dumps({'report':str(out/'REPORT_zh.html'),'cases':len(records),'figures':len(list(out.glob('*.png'))),'complete':True},ensure_ascii=False))


if __name__=='__main__':
    main()
