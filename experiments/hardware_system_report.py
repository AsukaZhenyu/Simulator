"""Render an evidence-bound Chinese hardware/strategy report; never simulate.

Incomplete data yields an explicit pending report and exit status 1. All figures
read the completed matrix; failed cases are missing latency, never zero time.
The reporter is intentionally excluded from execution-source freezes.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "paper-write/第七教学周/experiment-reports/hardware-systems-20261023"
GIB = 1024 ** 3
WORKLOADS = ("dense-prefill", "dense-decode", "moe-decode-hot", "moe-decode-rotating")
CANDIDATES = tuple(f"e{e}-s{s}" for e in range(4) for s in range(2))
WORKLOAD_ZH = {"dense-prefill": "Dense，32 token Prefill", "dense-decode": "Dense，8 token 输入＋2 步 Decode",
               "moe-decode-hot": "MoE，固定热点专家", "moe-decode-rotating": "MoE，轮换专家"}
EXEC_ZH = {"e0": "LLM 宏任务在加速器", "e1": "LLM 宏任务在 CPU",
           "e2": "FFN／专家在 CPU，其余在加速器", "e3": "Attention 在 CPU，其余在加速器"}
STORE_ZH = {"s0": "LRU，不预取", "s1": "相同 LRU 预算，跨层预取"}
SYSTEM_ZH = {
    "rtx4060_laptop": "典型独立显存笔记本。显卡 8 GB 来自 NVIDIA；主机 32 GiB、CPU 型号占位和预留量由本轮指定。CPU 可以保留主存中的权重，GPU 执行则需要准备显存副本。",
    "rtx5090_desktop": "大显存独立 GPU 台式机。显卡 32 GB 有厂商依据；主机内存与 CPU 占位是假设。即使图可以放进显存，冷启动的权重传输仍会进入短序列总耗时。",
    "m4pro_mini": "Apple M4 Pro 的 64 GB 可选配置，CPU/GPU 共享一个物理内存池；厂商标称内存带宽 273 GB/s。Neural Engine 的存在不等于这些 LLM 任务已能真实部署。",
    "dgx_spark": "选定 GB10 的 128 GB 一致性统一内存配置，厂商标称 273 GB/s。只描述一台单芯片系统，不模拟联网多节点，也不把 FP4 稀疏峰值换算成 F32 价格。",
    "ryzen_ai_max395": "选定 128 GB 共享内存配置，厂商描述 256 GB/s，包含 Radeon 8060S 与 XDNA 2 NPU。图形专用内存划分仍来自同一物理内存，不是额外容量；本轮没有强制真实驱动的显存 carve-out 限制。",
    "core_ultra258v": "32 GB 共享内存、Arc 140V GPU 与 Intel AI Boost NPU 的移动平台。只声明可用预算和任务价格，未从 INT8 TOPS 反推 F32 性能。",
    "snapdragon_x_elite": "Oryon CPU、Adreno GPU、Hexagon NPU，厂商标称内存带宽 135 GB/s。32 GiB 是本轮选定的 OEM 配置占位，没有声称具体笔记本 SKU 已测。",
    "jetson_agx_orin": "选定 AGX Orin 64 GB 模块，CPU/GPU 使用共享内存，厂商标称 204.8 GB/s；额外假定接入 NVMe。两个 NVDLA v2 的 LLM 算子支持未建立，因此没有当作通用 NPU 评测。",
    "snapdragon8elite_phone": "选定 OnePlus 13 的 16 GB 内存／512 GB UFS 4.0 配置。CPU、GPU 与 Hexagon NPU 的可兼容共享缓冲区是建模假设；6 GiB 系统预留也是假设。存储价格借用 SSD 字段，不表示手机装有 NVMe。",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def fmt(value, digits=3):
    return "未知／不适用" if value is None else f"{value:,.{digits}f}"


def key(row):
    return row["system_id"], row["variant_id"], row["workload_id"], row["candidate_id"]


def duration(row):
    value = row.get("completed_makespan_ns")
    return value if row.get("success") and type(value) is int and value > 0 else None


def failure(row):
    return str(row.get("error_message") or row.get("error_code") or
               row.get("termination_reason") or row.get("status") or "原因未知")


def load_complete(campaign, audits):
    required = ("records.json", "protocol.json", "completion.json", "controls.json",
                "source-manifest-before.json", "source-manifest-after.json")
    missing = [name for name in required if not (campaign / name).is_file()]
    if missing:
        raise ValueError("缺少完成证据：" + ", ".join(missing))
    records, protocol, completion, controls = (read_json(campaign / name) for name in required[:4])
    if (completion.get("complete") is not True or completion.get("completed_attempts") != 480
            or protocol.get("declared_attempts") != 480 or len(records) != 480):
        raise ValueError("尚未完整完成预声明的 480 组，不能生成完成报告")
    expected = {(j["system_id"], "gpu" if j["accelerator"] == "gpu" else
                 "npu-proxy-a" + format(j["npu_time_multiplier"], "g").replace(".", "p"),
                 j["workload_id"], j["candidate_id"]) for j in protocol["jobs"]}
    actual = [key(r) for r in records]
    if len(set(actual)) != 480 or set(actual) != expected:
        raise ValueError("480 组记录与预声明组合不一致或包含重复项")
    if (completion["records_sha256"] != canonical(records)
            or completion["protocol_sha256"] != canonical(protocol)
            or not completion.get("source_unchanged") or not completion.get("accuracy_evidence_unchanged")
            or read_json(campaign / required[4]) != read_json(campaign / required[5])
            or completion["source_manifest_sha256"] != canonical(read_json(campaign / required[5]))):
        raise ValueError("记录、协议或冻结证据校验失败")
    if not completion.get("controls_passed") or not all(c.get("equivalent") for c in controls):
        raise ValueError("等价性对照未全部通过")
    if not audits:
        raise ValueError("独立审计尚未提供；实验完成与独立审计通过必须分开")
    audit_documents = []
    for path in audits:
        if not path.is_file():
            raise ValueError("独立审计文件尚未生成：" + str(path))
        audit = read_json(path)
        if audit.get("passed") is not True:
            raise ValueError("独立审计未通过：" + str(path))
        for field in ("records_sha256",):
            if field in audit and audit[field] not in (canonical(records),sha(campaign / "records.json")):
                raise ValueError("独立审计不属于当前记录：" + str(path))
        if "completed_campaign_sha256" in audit and audit["completed_campaign_sha256"] != sha(campaign / "completion.json"):
            raise ValueError("独立审计不属于当前完成文件：" + str(path))
        audit_documents.append(audit)
    for row in records:
        if row.get("calibrated") is not False or row.get("hardware_measurement") is not False:
            raise ValueError("本报告只接受明确标记的未校准分析数据")
        if row.get("success") and duration(row) is None:
            raise ValueError("成功记录缺少正值的完整耗时")
    return records, protocol, completion, controls, audit_documents


def summarize(records, protocol):
    indexed = {key(r): r for r in records}
    systems = protocol["system_catalogue"]
    native_best = []
    for system in systems:
        for workload in WORKLOADS:
            rows = [indexed[(system["id"], "gpu", workload, c)] for c in CANDIDATES]
            valid = [r for r in rows if duration(r) is not None]
            best = min(map(duration, valid)) if valid else None
            base = duration(rows[0])
            native_best.append(dict(system_id=system["id"], workload_id=workload,
                feasible_count=len(valid), attempts=len(rows), baseline_ns=base, best_ns=best,
                winners=[r["candidate_id"] for r in valid if duration(r) == best],
                improvement_fraction=(1-best/base) if best is not None and base else None,
                best_speedup=base/best if best and base else None,
                failures=[dict(candidate_id=r["candidate_id"], reason=failure(r)) for r in rows if duration(r) is None]))
    storage_pairs = []
    for system in systems:
        for workload in WORKLOADS:
            for e in range(4):
                a = indexed[(system["id"], "gpu", workload, f"e{e}-s0")]
                b = indexed[(system["id"], "gpu", workload, f"e{e}-s1")]
                ta, tb = duration(a), duration(b)
                storage_pairs.append(dict(system_id=system["id"], workload_id=workload, execution=f"e{e}",
                    no_prefetch_ns=ta, prefetch_ns=tb,
                    improvement_fraction=1-tb/ta if ta and tb else None,
                    outcome=("unpaired_infeasible" if not ta or not tb else
                             "faster" if tb < ta else "slower" if tb > ta else "tie")))
    proxy_best = []
    for system_id in dict.fromkeys(r["system_id"] for r in records if r["accelerator"] == "npu_proxy"):
        for workload in WORKLOADS:
            gpu = [indexed[(system_id, "gpu", workload, c)] for c in CANDIDATES]
            gpu_best = min((duration(r) for r in gpu if duration(r)), default=None)
            for alpha in (.5, 1., 2.):
                variant = "npu-proxy-a" + format(alpha, "g").replace(".", "p")
                rows = [indexed[(system_id, variant, workload, c)] for c in CANDIDATES]
                valid = [r for r in rows if duration(r)]
                best = min(map(duration, valid)) if valid else None
                proxy_best.append(dict(system_id=system_id, workload_id=workload, alpha=alpha,
                    feasible_count=len(valid), best_ns=best,
                    winners=[r["candidate_id"] for r in valid if duration(r) == best],
                    relative_to_best_gpu=best/gpu_best if best and gpu_best else None))
    return dict(native_best=native_best, storage_pairs=storage_pairs, proxy_best=proxy_best,
        native_runs=sum(r["accelerator"] == "gpu" for r in records),
        proxy_runs=sum(r["accelerator"] == "npu_proxy" for r in records),
        success_count=sum(bool(r["success"]) for r in records), failed_count=sum(not r["success"] for r in records),
        storage_pair_outcomes=dict(Counter(p["outcome"] for p in storage_pairs)))


class Document:
    def __init__(self, title):
        self.title, self.md, self.html = title, [], []
        self.h(1, title)

    def h(self, level, text):
        self.md.append("#" * level + " " + text + "\n")
        self.html.append(f"<h{level}>{escape(text)}</h{level}>")

    def p(self, text):
        self.md.append(text + "\n")
        self.html.append("<p>" + escape(text) + "</p>")

    def table(self, headers, rows):
        def cell(t):
            return str(t).replace("|", "\\|").replace("\n", " / ")
        self.md.extend(["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        self.md.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
        self.md.append("")
        self.html.append('<div class="table-wrap"><table><thead><tr>' + "".join("<th>"+escape(str(v))+"</th>" for v in headers)
            + "</tr></thead><tbody>" + "".join("<tr>"+"".join("<td>"+escape(str(v))+"</td>" for v in row)+"</tr>" for row in rows)
            + "</tbody></table></div>")

    def links(self, links):
        self.md.extend(f"- [{text}]({url})" for text, url in links)
        self.md.append("")
        self.html.append("<ul>"+"".join(f'<li><a href="{escape(url, quote=True)}">{escape(text)}</a></li>' for text, url in links)+"</ul>")

    def image(self, path, caption):
        self.md.append(f"![{caption}](figures/{path.name})\n\n{caption}\n")
        uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        self.html.append(f'<figure><img src="{uri}" alt="{escape(caption, quote=True)}"><figcaption>{escape(caption)}</figcaption></figure>')

    def write(self, output):
        (output / "REPORT_zh.md").write_text("\n".join(self.md), encoding="utf-8", newline="\n")
        css = ('body{font-family:"Microsoft YaHei",Arial,sans-serif;line-height:1.8;color:#182b43;max-width:1250px;margin:36px auto;padding:0 28px}'
            'h1{color:#203864;border-bottom:3px solid #326b9f;padding-bottom:14px}h2{margin-top:42px;color:#203864}p,li{font-size:16px}'
            'table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{border:1px solid #ccd8e3;padding:9px 11px;vertical-align:top;text-align:left}'
            'th{background:#e7f0f8}tr:nth-child(even){background:#f7f9fc}.table-wrap{overflow-x:auto}img{max-width:100%;display:block;margin:auto}'
            'figure{margin:30px 0}figcaption{font-size:14px;color:#536a7e}a{color:#215f96}@media print{figure,table{break-inside:avoid}h2{break-after:avoid}}')
        page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(self.title)+'</title><style>'+css+'</style></head><body>'+"".join(self.html)+'</body></html>'
        (output / "REPORT_zh.html").write_text(page, encoding="utf-8", newline="\n")


def plots(records, protocol, analysis, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    systems = protocol["system_catalogue"]
    names = [s["id"].replace("_", " ") for s in systems]
    indexed = {key(r): r for r in records}
    saved = []

    def save(fig, stem):
        fig.savefig(folder / (stem + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(folder / (stem + ".svg"), bbox_inches="tight")
        plt.close(fig)
        saved.append(folder / (stem + ".png"))

    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#dddddd")
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    for ax, workload in zip(axes.flat, WORKLOADS):
        data = np.full((len(systems), len(CANDIDATES)), np.nan)
        texts = []
        for i, s in enumerate(systems):
            base = duration(indexed[(s["id"], "gpu", workload, "e0-s0")])
            line = []
            for j, c in enumerate(CANDIDATES):
                value = duration(indexed[(s["id"], "gpu", workload, c)])
                ratio = value/base if value and base else None
                if ratio:
                    data[i, j] = math.log2(ratio)
                line.append("FAIL" if value is None else "N/A" if base is None else f"{ratio:.2f}")
            texts.append(line)
        image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        for i, line in enumerate(texts):
            for j, text in enumerate(line):
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        ax.set(xticks=range(8), xticklabels=CANDIDATES, yticks=range(len(systems)), yticklabels=names,
               title=workload + " | latency / same-system e0-s0")
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(image, ax=axes, label="log2 ratio: <0 faster, >0 slower; clipped color scale", shrink=.65)
    fig.suptitle("Hypothetical task costs: compare policies within each hardware row", fontsize=16)
    save(fig, "01_native_policy_ratios")

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    best_index = {(r["system_id"], r["workload_id"]): r for r in analysis["native_best"]}
    matrix = np.full((len(systems), 4), np.nan)
    for i, system in enumerate(systems):
        for j, workload in enumerate(WORKLOADS):
            item = best_index[(system["id"], workload)]
            value = item["improvement_fraction"]
            if value is not None:
                matrix[i, j] = value * 100
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for i, system in enumerate(systems):
        for j, workload in enumerate(WORKLOADS):
            item = best_index[(system["id"], workload)]
            winners = "/".join(item["winners"]) or "none"
            text = winners + f"\n{item['feasible_count']}/8 feasible"
            gain=item["improvement_fraction"]
            text += "\n" + ("baseline unavailable" if gain is None else "same as baseline" if gain == 0 else f"{gain*100:.1f}% less time")
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if matrix[i,j] > 65 else "#10253a")
    ax.set(xticks=range(4), xticklabels=WORKLOADS, yticks=range(len(systems)), yticklabels=names,
           title="Best tested native policy, ties retained | gain over same-system e0-s0")
    fig.colorbar(image, ax=ax, label="Latency reduction (%)", shrink=.7)
    save(fig, "02_native_winners")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for ax, workload in zip(axes.flat, WORKLOADS):
        rows = [indexed[(s["id"], "gpu", workload, "e0-s0")] for s in systems]
        physical = [(r.get("h2d_bytes", 0)+r.get("d2h_bytes", 0))/GIB if duration(r) else np.nan for r in rows]
        reload = [r.get("immutable_weight_reload_bytes", 0)/GIB if duration(r) else np.nan for r in rows]
        x = np.arange(len(systems))
        ax.bar(x-.18, physical, .36, label="Physical H2D + D2H", color="#427fb0")
        ax.bar(x+.18, reload, .36, label="Repeated immutable-weight loads", color="#da9d48")
        for i, r in enumerate(rows):
            if not duration(r):
                ax.text(i, 0, "FAIL", ha="center", va="bottom", color="#a12d37", fontsize=8)
        ax.set(xticks=x, xticklabels=names, ylabel="GiB", title=workload + " | e0-s0 only")
        ax.tick_params(axis="x", rotation=65)
        ax.grid(axis="y", alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Traffic mechanism under declared topology, not measured PCIe bandwidth", fontsize=15)
    save(fig, "03_native_physical_traffic")

    pools = [(s, p) for s in systems for p in (("shared",) if s["topology"] == "shared" else ("dram", "vram"))]
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    fill = np.full((len(pools), 4), np.nan)
    labels = []
    texts = []
    for i, (system, pool) in enumerate(pools):
        h = system["hardware"]
        capacity = h["shared_capacity_bytes"] if pool == "shared" else h[pool+"_capacity_bytes"]
        reserve = h.get("shared_runtime_reserved_bytes", 0) if pool == "shared" else 0
        labels.append(system["id"].replace("_", " ") + " / " + pool)
        line = []
        for j, workload in enumerate(WORKLOADS):
            row = indexed[(system["id"], "gpu", workload, "e0-s0")]
            peak = row.get("peak_"+pool+"_bytes") if duration(row) else None
            if peak is None:
                line.append("FAIL / unknown")
            else:
                fill[i,j] = max(0, (peak-reserve)/(capacity-reserve))*100
                line.append(f"{(capacity-peak)/GIB:.2f} GiB free\n{fill[i,j]:.1f}% budget used")
        texts.append(line)
    image = ax.imshow(np.ma.masked_invalid(fill), cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    for i, line in enumerate(texts):
        for j, text in enumerate(line):
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if fill[i,j] > 70 else "#10253a")
    ax.set(xticks=range(4), xticklabels=WORKLOADS, yticks=range(len(labels)), yticklabels=labels,
        title="e0-s0 physical-pool headroom | shared reserve excluded from workload budget use")
    fig.colorbar(image, ax=ax, label="Workload peak / declared usable budget (%)", shrink=.7)
    save(fig, "04_native_memory_headroom")

    proxy_systems = list(dict.fromkeys(r["system_id"] for r in records if r["accelerator"] == "npu_proxy"))
    if len(proxy_systems) != 2:
        raise ValueError("final report requires the declared two-platform NPU sensitivity matrix")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for i, identity in enumerate(proxy_systems):
        for j, workload in enumerate(WORKLOADS):
            ax = axes[i,j]
            for k, candidate in enumerate(CANDIDATES):
                base = duration(indexed[(identity, "gpu", workload, candidate)])
                ys = []
                for alpha in (.5, 1., 2.):
                    variant = "npu-proxy-a"+format(alpha, "g").replace(".", "p")
                    value = duration(indexed[(identity, variant, workload, candidate)])
                    ys.append(value/base if value and base else np.nan)
                ax.plot([.5,1,2], ys, marker="o", linewidth=1.4, color=colors(k), label=candidate)
            ax.axhline(1, color="#777777", linestyle=":", linewidth=1)
            ax.set(title=identity.replace("_", " ")+"\n"+workload, xticks=[.5,1,2], xlabel="Assumed task-time multiplier alpha")
            if j == 0:
                ax.set_ylabel("Proxy latency / paired same-policy GPU latency")
            ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8)
    fig.suptitle("CPU + one accelerator slot: hypothetical NPU substitution, not a hardware benchmark", fontsize=15)
    save(fig, "05_npu_proxy_sensitivity")
    return saved


def report(records, protocol, completion, controls, audits, analysis, figures, output, campaign, unused_copy=None, tests=None):
    d = Document("端侧异构硬件与推理策略对比实验报告")
    d.p("2026-10-23。本报告比较同一套执行与存储策略在九类声明硬件配置下的表现。所有耗时均由分析模型计算，未运行真实 LLM，也未采集这些设备的性能。硬件型号与容量选项有厂商来源，任务有效速率、可用预算和启动开销是明确给定的假设。")
    d.h(2, "先看结论与证据范围")
    d.p(f"预声明 480 组已完成：CPU＋GPU {analysis['native_runs']} 组；CPU＋假设 NPU {analysis['proxy_runs']} 组。成功 {analysis['success_count']} 组，未完成／不可行 {analysis['failed_count']} 组。后者保留在统计中，未当作零耗时。{len(controls)} 项等价性对照全部通过，独立审计 {len(audits)} 份通过。")
    valid_best = [r for r in analysis["native_best"] if r["best_speedup"] is not None]
    if valid_best:
        maximum = max(valid_best, key=lambda r:r["best_speedup"])
        d.p(f"在固定的八个候选中，相对同设备 e0-s0，最大模拟加速为 {maximum['best_speedup']:.3f} 倍，出现在 {maximum['system_id']} / {maximum['workload_id']}，候选为 {'、'.join(maximum['winners'])}。这是指定参数下的策略收益，不是该型号的真实性能结论。")
    outcomes = analysis["storage_pair_outcomes"]
    d.p("单独打开预取的 144 对比较结果为："+"；".join(f"{name} {outcomes.get(tag,0)} 对" for tag,name in (("faster","更快"),("tie","相同"),("slower","更慢"),("unpaired_infeasible","至少一侧不可行")))+"。因此是否有收益应按实际组合判断，不预设预取总是更好。")
    separate_ids={s["id"] for s in protocol["system_catalogue"] if s["topology"]=="separate"}
    separate_best=[r for r in analysis["native_best"] if r["system_id"] in separate_ids]
    cpu_wins=sum(any(c.startswith("e1-") for c in r["winners"]) for r in separate_best)
    d.p(f"两类独立显存配置的 {len(separate_best)} 个硬件／负载组合中，有 {cpu_wins} 个由全 CPU 候选获得或并列获得最优。必须结合冷态、两层和最多两步 Decode 的短窗口理解：CPU 直接使用主存权重，GPU 首次准备全部权重的传输容易主导总耗时。这不是 CPU 长期推理速度高于 GPU 的结论。")
    negative=[p for p in analysis["storage_pairs"] if p["improvement_fraction"] is not None and p["improvement_fraction"] < 0]
    if negative:
        worst=min(negative,key=lambda p:p["improvement_fraction"])
        d.p(f"预取退化最大的已完成配对为 {worst['system_id']} / {worst['workload_id']} / {worst['execution']}：不预取 {worst['no_prefetch_ns']/1e6:.3f} ms，预取 {worst['prefetch_ns']/1e6:.3f} ms，增加 {(worst['prefetch_ns']/worst['no_prefetch_ns']-1)*100:.2f}%。这揭示当前固定广泛预取规则与执行放置的组合问题，不意味着预取普遍无效，也不代表某篇论文的完整算法。")
    shared = [r for r in records if r["accelerator"] == "gpu" and r["memory_mode"] == "shared" and duration(r)]
    zero = sum(not r.get("h2d_bytes",0) and not r.get("d2h_bytes",0) for r in shared)
    all_shared=[r for r in records if r["memory_mode"]=="shared" and duration(r)]
    all_shared_zero=sum(not r.get("h2d_bytes",0) and not r.get("d2h_bytes",0) for r in all_shared)
    d.p(f"CPU＋GPU 原生模式下，共享存储成功案例 {len(shared)} 组，其中 {zero} 组物理 H2D/D2H 字节均为零；加上 NPU 代理，全部共享案例为 {len(all_shared)} 组，零载荷交接为 {all_shared_zero} 组。这验证了相同缓冲区交接无需复制的抽象；没有验证实际设备的同步、格式转换或并发共享带宽开销。")
    d.h(2, "一、这些硬件系统具体描述了什么")
    rows=[]
    for s in protocol["system_catalogue"]:
        total = f"主存 {s['host_memory_gib']} ＋显存 {s['device_memory_gib']}" if s["topology"]=="separate" else f"共享 {s['shared_memory_gib']}"
        usable = f"主存 {s['available_host_gib']} ＋显存 {s['available_device_gib']}" if s["topology"]=="separate" else f"共享 {s['available_shared_gib']}"
        rows.append((s["name"],s["topology"],total,usable,s["accelerator_name"] or "无单列 NPU"))
    d.table(["所选配置","存储形式","配置容量（GiB 约定）","声明可用预算（GiB）","额外加速器身份"],rows)
    d.p("厂商 GB 选项在本轮统一按 GiB 表示；带宽 GB/s 使用十进制字节。独立存储为两份物理池；共享存储的 CPU/GPU 视图只有一份物理容量。表中可用预算扣除了作者声明的预留量，不是测得的操作系统占用。")
    for s in protocol["system_catalogue"]:
        d.h(3,s["name"])
        d.p(SYSTEM_ZH[s["id"]])
        d.links([("厂商资料："+source["title"],source["url"]) for source in s["sources"]])
    d.table(["配置 ID","CPU TFLOP/s／GB/s","GPU TFLOP/s／GB/s","真实复制 GB/s","存储读／写 GB/s"],
        [(s["id"],f"{s['cpu_tflops']} / {s['cpu_gbps']}",f"{s['gpu_tflops']} / {s['gpu_gbps']}",
          s["link_gbps"] if s["link_gbps"] is not None else "共享视图不复制",f"{s['ssd_read_gbps']} / {s['ssd_write_gbps']}") for s in protocol["system_catalogue"]])
    d.p("上表全部是任务有效速率假设。CPU/GPU 每个任务的启动开销分别统一为 500 ns / 5,000 ns；独立存储复制固定延迟 1,000 ns。没有从厂商的 INT8 TOPS、FP4 稀疏性能推算 F32 运算速度。厂商标称内存带宽只保存在元数据中，没有重复乘进计算公式。")
    d.p("计算公式沿用现有 modeling：任务时间=max(1 ns, ceil(10⁹×max(FLOPs/有效算力, 字节数/有效带宽))＋启动开销)。真实复制按载荷除链路速率加固定延迟计时；共享同一缓冲区使用 SHARE，零载荷。依赖与资源占用决定任务何时启动，再递推完整时间轴。")
    d.h(2,"二、公平比较：相同工作负载、相同八个策略")
    d.p("采用两层 LLM 形状计算图，hidden=6144、intermediate=16384、24 个 Attention head、6 个 KV head，所有逻辑张量为 F32。MoE 每层 4 个专家、每 token 激活 2 个。只构造工作量、依赖与容量，不分配真实大矩阵，也没有验证模型数值或语言质量。")
    d.table(["场景","输入与推理","专家路由"],[(WORKLOAD_ZH[w["spec"]["id"]],f"Prefill {w['spec']['prefill_tokens']} token，Decode {w['spec']['decode_steps']} 步",w["spec"].get("route_pattern") or "不适用") for w in protocol["workloads"]])
    d.table(["执行维度","含义"],EXEC_ZH.items())
    d.table(["存储维度","含义"],STORE_ZH.items())
    d.p("八个候选为 e0-s0 到 e3-s1：4 种执行放置乘以 2 种预取设置，是 IR 空间中预先选定的八个点。LRU 预算直接等于该硬件声明可用的加速器物理容量：独立存储取可用 VRAM，共享存储取总容量减预留。同一硬件所有策略的预算相同，s0/s1 都显式传入正值，满足预取接口检查；没有为了制造效果加入更小的任意公共配额。预取是一层前瞻、每次最多两个权重、包含 Dense 权重。这里不是八篇完整论文实现，也没有进行全空间搜索或证明全局最优。个别状态维护或路由动作有自己的固定设备约束，不能把 e0 理解为所有底层动作都必须在加速器。")
    budgets=protocol.get("cache_budget_by_system",{})
    if budgets:
        d.table(["硬件","所有候选相同的 LRU 预算 GiB"],[(identity,fmt(value/GIB)) for identity,value in budgets.items()])
    d.p("每次从冷态开始：权重和输入在 DRAM，KV 为空，策略拥有全新运行时；结束要求相同的 hidden/KV DRAM 可用目标。耗时包含首次权重准备和必要的结束交接。它是两层有限推理序列总耗时，不能直接称为稳态 TPOT、TTFT 或完整生产模型吞吐。未建模 embedding、LM head、采样等全部服务端过程。")
    d.p("两层 Dense 权重约 2.953 GiB，MoE 权重约 9.703 GiB。前者可放入声明的 6 GiB RTX 4060 可用显存，后者会给该显存造成容量压力；手机声明 10 GiB 共享预算，MoE 全权重仅剩约 0.297 GiB 余量，再容纳激活与 KV。两步 Decode 不能代表长上下文或长期稳态。workspace=0 是本轮显式简化假设，不是测得没有额外空间。")
    d.h(2,"三、CPU＋GPU：策略在不同声明硬件下怎样变化")
    captions=["每个硬件行内部，以该行 e0-s0 为 1。小于 1 更快，大于 1 更慢；FAIL 表示没有完整执行结果，不能与耗时混算。颜色是 log2 比值，标注保留实际比值。",
        "每个硬件／负载单元格给出八候选中的最优者、并列候选和可行数。收益相对同设备 e0-s0；这不是九台真实机器的性能排名。",
        "e0-s0 的真实主存—显存搬运与重复权重装载量。共享视图的 SHARE 没有载荷复制；零值反映声明语义。重复装载量是总搬运中的一部分，不能再相加。",
        "e0-s0 各物理池剩余空间。共享池只画一次；共享预留先从峰值和预算中扣除，再计算工作负载占可用预算的比例。失败案例没有伪造峰值。",
        "两类选定平台上，将同平台 GPU 任务价格整体乘 α 后替代为假设 NPU；CPU 和搬运价格保持不变。每条线比较同一策略的代理耗时与原 GPU 耗时，包含启动价格。"]
    for image, caption in zip(figures[:4],captions[:4]):
        d.image(image,caption)
    d.table(["硬件","场景","可行数","最优候选（含并列）","最优总耗时 ms","相对 e0-s0 加速"],
        [(r["system_id"],WORKLOAD_ZH[r["workload_id"]],f"{r['feasible_count']}/8","、".join(r["winners"]) or "无",fmt(r["best_ns"]/1e6 if r["best_ns"] else None),fmt(r["best_speedup"])) for r in analysis["native_best"]])
    failed=[r for r in records if not r["success"]]
    if failed:
        d.h(3,"不可行／失败组合保留在评估中")
        d.table(["硬件","加速器配置","场景","策略","原因"],[(r["system_id"],r["variant_id"],r["workload_id"],r["candidate_id"],failure(r)) for r in failed])
    if all(not r.get("ssd_read_bytes",0) and not r.get("ssd_write_bytes",0) for r in records if duration(r)):
        d.p("本批所有成功组合的 SSD／UFS 读写载荷均为零，因此没有验证超出内存后的存储溢出策略、SSD 预取收益或写回代价。表中存储速率只是声明输入，并未在本批形成性能结论。")
    if all(not r.get("cpu_accelerator_event_overlap_ns",0) for r in records if duration(r)):
        d.p("本批 CPU 与加速器的 COMPUTE 事件重叠均为零，因此没有用这些案例验证异构计算流水线的收益。部分案例可以存在复制与计算重叠，但这与 CPU／加速器同时计算不同。")
    d.h(3,"冷态与轮次诊断：不要把总耗时当作稳定 Decode 速度")
    d.p("下面只展示独立显存系统、e0-s0 与 e1-s0 的若干轮次。时间戳从整次运行起点计算；首计算到末计算的跨度可能包含中间搬运或等待，但排除了第一项计算开始前的准备及最后计算完成后的目标交接。各设备计算工作量是事件时长之和。它们是诊断边界，不是 TTFT、TPOT 或完整轮次耗时，不能简单相加还原因果开销。")
    round_rows=[]
    for r in records:
        if (r["accelerator"] == "gpu" and r["memory_mode"] == "separate"
                and r["candidate_id"] in ("e0-s0","e1-s0")
                and r["workload_id"] in ("dense-decode","moe-decode-rotating") and duration(r)):
            for boundary in r.get("round_compute_boundaries",[]):
                start=boundary["first_compute_start_ns"];end=boundary["last_compute_completion_ns"]
                round_rows.append((r["system_id"],r["workload_id"],r["candidate_id"],boundary["round_id"],
                    fmt(start/1e6),fmt(end/1e6),fmt((end-start)/1e6),
                    fmt(boundary["cpu_compute_duration_sum_ns"]/1e6),fmt(boundary["accelerator_compute_duration_sum_ns"]/1e6)))
    d.table(["硬件","场景","策略","轮次","首计算开始 ms","末计算完成 ms","跨度 ms","CPU 计算工作量 ms","加速器计算工作量 ms"],round_rows)
    indexed={key(r):r for r in records}
    example_gpu=indexed.get(("rtx5090_desktop","gpu","dense-decode","e0-s0"))
    example_cpu=indexed.get(("rtx5090_desktop","gpu","dense-decode","e1-s0"))
    if example_gpu and example_cpu and duration(example_gpu) and duration(example_cpu):
        by_round_gpu={r["round_id"]:r for r in example_gpu.get("round_compute_boundaries",[])}
        by_round_cpu={r["round_id"]:r for r in example_cpu.get("round_compute_boundaries",[])}
        if "decode0" in by_round_gpu and "decode0" in by_round_cpu:
            g=by_round_gpu["decode0"];c=by_round_cpu["decode0"]
            gs=g["last_compute_completion_ns"]-g["first_compute_start_ns"]
            cs=c["last_compute_completion_ns"]-c["first_compute_start_ns"]
            d.p(f"一个直观例子：RTX 5090 声明配置的 Dense 短序列，总耗时 GPU 候选为 {duration(example_gpu)/1e6:.3f} ms，全 CPU 为 {duration(example_cpu)/1e6:.3f} ms；GPU 路径需要先准备 {example_gpu['h2d_bytes']/GIB:.3f} GiB 的 H2D 载荷。但 decode0 的首计算到末计算跨度，GPU 仅 {gs/1e6:.3f} ms，CPU 为 {cs/1e6:.3f} ms，相差 {cs/gs:.2f} 倍。整段冷态赢家与准备权重后的计算区段赢家可以不同；这正是需要统一时间轴的原因。该跨度仍排除了完整输出交接，不能称为 TPOT。")
    if unused_copy is not None:
        extra=[p for p in unused_copy["storage_pairs"] if p["variant_id"]=="gpu" and p["extra_unused_weight_copy_bytes"]>0]
        d.h(3,"已确认的策略组合缺口：预取了没有加速器消费者的权重")
        d.p("对执行结束后的完整事件做只读检查：找出被真实复制到加速器、但在整个有限执行里从未被加速器计算消费的不可变权重。这是可证明未使用流量的保守子集。事后使用未来信息只用于诊断，策略运行时没有获得未来路由或消费者信息。它不能统计所有无效预取，也不能证明这些字节独自解释了全部耗时差。")
        d.table(["硬件","场景","执行","不预取 ms","预取 ms","新增零消费者搬运 GiB"],
            [(p["system_id"],p["workload_id"],p["execution"],fmt(p["no_prefetch_completed_makespan_ns"]/1e6 if p["no_prefetch_completed_makespan_ns"] else None),
              fmt(p["prefetch_completed_makespan_ns"]/1e6 if p["prefetch_completed_makespan_ns"] else None),fmt(p["extra_unused_weight_copy_bytes"]/GIB)) for p in extra])
        d.p("这些例子说明，固定的跨层广泛预取不能脱离任务的实际执行位置。一个直接的改进方向是按已知执行放置筛选预取目标；本轮只暴露并解释缺口，尚未更改被冻结的候选或据此宣称新算法收益。")
    d.h(2,"四、CPU＋假设 NPU：先做参数敏感性，不冒充真机数据")
    proxy_systems=list(dict.fromkeys(r["system_id"] for r in records if r["accelerator"]=="npu_proxy"))
    d.p("本轮只在 "+"、".join(proxy_systems)+" 做配对代理实验。公式 t_NPU=ceil(α×同平台 GPU 完整任务时间)，α∈{0.5,1,2}，对应任务时间减半、相同、加倍。我们替换现有加速器槽，并没有让 CPU、GPU、NPU 三者同时并发。")
    d.p("这是一份 F32 等价工作量下、全部任务假设可执行、布局兼容且 workspace 相同的理想价格表。它不表示真实 NPU 支持 F32、这些算子或这些动态形状。未测 CPU 回退、编译、转换、设备同步或 NPU 可用内存限制。结论只能是：若任务价格满足这些假设，策略选择如何变化。")
    d.image(figures[4],captions[4])
    d.table(["平台","场景","α","可行数","代理最优候选","代理最优 ms","相对 GPU 最优时间"],[(r["system_id"],r["workload_id"],r["alpha"],f"{r['feasible_count']}/8","、".join(r["winners"]) or "无",fmt(r["best_ns"]/1e6 if r["best_ns"] else None),fmt(r["relative_to_best_gpu"])) for r in analysis["proxy_best"]])
    d.p("α=1 的代理与 GPU 采用相同价格，应得到相同事件序列；全 CPU 策略不依赖代理价格，应在所有 α 下保持一致。本轮通过的等价性对照直接检查了这些性质，因此变化来自声明的价格替换，而非换了一份输入或策略。")
    d.h(2,"五、当前可以使用的结论与下一步")
    d.p("现在可以使用同一策略表达与编译流程，在独立显存和共享内存两种拓扑上比较可行性、总耗时、物理内存与搬运代价；可以验证硬件预算与计算／搬运比例改变后，最佳候选是否变化。现有性能公式没有为某台产品另开一套。")
    d.p("共享存储没有自动模拟 CPU/GPU 争抢同一 DRAM 带宽；零载荷 SHARE 只针对同一个已兼容缓冲区，不代表 UMA 的所有通信都零开销。真实驱动的共享 buffer 支持、缓存一致性、格式转换、启动和并发开销，需要后续 profile。固定任务价格不保证真实设备上的并发性能。即使某批全部偏向 GPU，也不能据此否定 CPU 混合计算在其他容量、竞争、算子支持或有效性能条件下的价值。")
    d.p("本轮八个点只改变宏任务执行位置与权重预取，不覆盖张量并行拆分、P/D 分离、多请求流水或 CPU／加速器并行计算，也未触发 SSD 溢出。报告不能据此展示这些机制的收益；已有 IR 是否支持它们，与本批是否实际验证它们，是两件不同的事。")
    d.p("能耗、稳态 TPOT、真实 NPU 算子支持率、设备动态波动、完整 LLM 数值精度和真机策略收益仍未知。本轮不重新打开已经冻结的准确性实验，也没有用这些分析结果证明真实硬件误差。下一步先对计算速率、链路速率与容量做受控扫描，找到策略交叉点，再优先替换会改变名次的关键任务与交接价格，并补真实共享存储／NPU 测量。")
    d.h(2,"六、复核与完整数据")
    d.table(["证据","结果"],[("预声明组合",f"{completion['completed_attempts']} / {completion['declared_attempts']}"),("等价性对照",f"{len(controls)} / {len(controls)}"),("执行源代码未变化",str(completion['source_unchanged'])),("冻结准确性依据未变化",str(completion['accuracy_evidence_unchanged'])),("独立审计",f"{len(audits)} 份 passed=true"),("记录规范化 SHA-256",completion['records_sha256']),("协议规范化 SHA-256",completion['protocol_sha256'])])
    if tests:
        test_groups=[child for test in tests for child in
            ([test["focused"],test["independent_audit"]] if "focused" in test and "independent_audit" in test else [test])]
        d.table(["专项检查命令","测试数","结果"],[(t.get("command","见原始凭据"),t.get("tests_run",t.get("test_executions","见原始凭据")),"通过") for t in test_groups])
    d.links([("全部 480 组原始记录 JSON","data/records.json"),("全部 480 组 CSV","data/runs.csv"),("冻结协议与硬件来源","data/protocol.json"),("完成证明","data/completion.json"),("派生分析","data/analysis.json"),("交付文件哈希清单","delivery-manifest.json"),("历史运行时源代码哈希基线","data/legacy-runtime-source-manifest.json")]+[("独立审计 "+str(i+1),f"data/audit-{i+1}.json") for i in range(len(audits))]+([("未使用权重搬运的事后审计","data/unused-weight-copy-audit.json")] if unused_copy else [])+[("专项测试凭据 "+str(i+1),f"data/tests-{i+1}.json") for i in range(len(tests or []))])
    if tests:
        d.links([("测试日志："+p.name,"data/"+p.name) for p in sorted((output/"data").glob("tests-*.log"))])
    d.p("每组完整输入、编译 IR、成本表、事件、张量、算子、策略日志与文件哈希保存在实验原始目录："+str(campaign)+"。本目录复制摘要、协议、分析和图表；HTML 内嵌全部图，便于独立阅读。")
    d.h(3,"复现命令")
    d.p("在 Simulator 目录运行以下命令。输出目录必须为新的空目录。库和实验无需测量文件；报告绘图需另外安装 Matplotlib（同时安装 NumPy）。历史不变性审计还需要本报告保存的旧版本源代码哈希基线，--legacy-manifest 必须显式指向它；干净克隆若没有此基线，不能直接声称历史回归审计通过。审计不会静默跳过这些检查。")
    commands=("python -m pip install -e .\n"
        "python -m pip install matplotlib\n"
        "python experiments/hardware_system_campaign.py --out build/hardware-systems-20261023/new-run --proxy-systems core_ultra258v snapdragon8elite_phone --workers 4\n"
        'python experiments/hardware_system_audit.py build/hardware-systems-20261023/new-run build/hardware-systems-20261023/new-run-audit.json --legacy-manifest "../paper-write/第七教学周/experiment-reports/hardware-systems-20261023/data/legacy-runtime-source-manifest.json"\n'
        "python experiments/hardware_system_unused_copy_audit.py --root build/hardware-systems-20261023/new-run --out build/hardware-systems-20261023/new-run-unused-weight-copy-audit.json\n"
        'python experiments/hardware_system_report.py --campaign build/hardware-systems-20261023/new-run --audit build/hardware-systems-20261023/new-run-audit.json --unused-copy-audit build/hardware-systems-20261023/new-run-unused-weight-copy-audit.json --legacy-manifest "../paper-write/第七教学周/experiment-reports/hardware-systems-20261023/data/legacy-runtime-source-manifest.json" --out "../paper-write/第七教学周/experiment-reports/hardware-systems-new-run"')
    d.md.append("```powershell\n"+commands+"\n```\n")
    d.html.append("<pre style=\"overflow-x:auto;background:#eef3f8;padding:16px\"><code>"+escape(commands)+"</code></pre>")
    d.write(output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,default=ROOT/"build/hardware-systems-20261023/final-v2")
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--audit",type=Path,action="append",default=[])
    parser.add_argument("--unused-copy-audit",type=Path)
    parser.add_argument("--test-results",type=Path,action="append",default=[])
    parser.add_argument("--legacy-manifest",type=Path)
    args=parser.parse_args(argv)
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        records,protocol,completion,controls,audits=load_complete(args.campaign.resolve(),args.audit)
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as error:
        d=Document("端侧硬件策略实验：等待完整证据")
        d.p("当前状态：未知／未完成。"+str(error))
        d.p("未输出性能结果与图表，未将试跑、缺失组合或部分完成的数据作为正式结论。")
        d.write(output)
        write_json(output/"delivery-manifest.json",dict(complete=False,reason=str(error),
            warning="Existing data/figures, if any, are not a completed delivery for these inputs."))
        print(json.dumps(dict(complete=False,reason=str(error)),ensure_ascii=False))
        return 1
    data=output/"data";figures=output/"figures"
    data.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    for name in ("records.json","protocol.json","completion.json","controls.json","source-manifest-before.json","source-manifest-after.json"):
        shutil.copyfile(args.campaign/name,data/name)
    for index,path in enumerate(args.audit,1):
        shutil.copyfile(path,data/f"audit-{index}.json")
    legacy_manifest=args.legacy_manifest or ROOT/"build/shared-storage-20261023/legacy-regression-v3/source-manifest-after.json"
    if not legacy_manifest.is_file():
        raise ValueError("archived legacy source baseline is unavailable")
    for audit in audits:
        if audit.get("legacy_runtime_manifest_sha256") and audit["legacy_runtime_manifest_sha256"] != sha(legacy_manifest):
            raise ValueError("legacy manifest differs from independent audit binding")
    if legacy_manifest.resolve() != (data/"legacy-runtime-source-manifest.json").resolve():
        shutil.copyfile(legacy_manifest,data/"legacy-runtime-source-manifest.json")
    unused_copy=None
    if args.unused_copy_audit:
        unused_copy=read_json(args.unused_copy_audit)
        if unused_copy.get("campaign_complete") is not True or unused_copy.get("cases") != 480:
            raise ValueError("unused-copy diagnostic must cover this complete 480-case campaign")
        indexed={key(r):r for r in records}
        for pair in unused_copy["storage_pairs"]:
            for suffix,field in (("s0","no_prefetch_completed_makespan_ns"),("s1","prefetch_completed_makespan_ns")):
                row=indexed[(pair["system_id"],pair["variant_id"],pair["workload_id"],pair["execution"]+"-"+suffix)]
                if pair[field] != duration(row):
                    raise ValueError("unused-copy diagnostic is not bound to current durations")
        shutil.copyfile(args.unused_copy_audit,data/"unused-weight-copy-audit.json")
    tests=[]
    for index,path in enumerate(args.test_results,1):
        test=read_json(path)
        if test.get("passed") is not True or test.get("exit_code",0) != 0:
            raise ValueError("test receipt is not a passing result")
        groups=[("focused",test["focused"]),("independent_audit",test["independent_audit"])] if "focused" in test and "independent_audit" in test else [("run",test)]
        for name,group in groups:
            if group.get("passed") is not True or group.get("exit_code",0) != 0:
                raise ValueError("test sub-receipt is not a passing result")
            if group.get("log_file") and group.get("log_sha256"):
                logfile=path.parent/group["log_file"]
                if sha(logfile) != group["log_sha256"]:
                    raise ValueError("test log hash differs from receipt")
                shutil.copyfile(logfile,data/f"tests-{index}-{name}.log")
        tests.append(test)
        shutil.copyfile(path,data/f"tests-{index}.json")
    fields=list(dict.fromkeys(field for row in records for field in row))
    with (data/"runs.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fields,lineterminator="\n");writer.writeheader()
        for row in records:
            writer.writerow({name:json.dumps(value,ensure_ascii=False,allow_nan=False) if isinstance(value,(dict,list)) else value for name,value in row.items()})
    analysis=summarize(records,protocol)
    write_json(data/"analysis.json",analysis)
    images=plots(records,protocol,analysis,figures)
    report(records,protocol,completion,controls,audits,analysis,images,output,args.campaign.resolve(),unused_copy,tests)
    manifest=dict(format_version="hardware-system-report-delivery-0.1",complete=True,
        input_campaign=str(args.campaign.resolve()),records_sha256=canonical(records),protocol_sha256=canonical(protocol),
        generator_sha256=sha(__file__),figure_count=len(images),attempts=len(records),
        files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="delivery-manifest.json"})
    write_json(output/"delivery-manifest.json",manifest)
    print(json.dumps(dict(complete=True,attempts=len(records),figures=len(images),output=str(output),delivery_manifest_sha256=sha(output/"delivery-manifest.json")),ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
