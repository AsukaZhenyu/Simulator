#!/usr/bin/env python3
"""把 ``python -m tensor_mapping`` 的结果目录画成瀑布图与显存曲线。

用法::

    python tools/plot_results.py outputs/chain-search            # 文本视图（只用标准库）
    python tools/plot_results.py outputs/chain-search --png      # 另出两张 PNG
    python tools/plot_results.py outputs/chain-search --png --out fig --dpi 160

本脚本**只读** ``stats.json`` / ``events.json`` / ``states.json``，不写任何产物、不改
任何语义，所以它永远不会和 ``DESIGN.md`` §8 的产物契约打架。

两条纪律（都不是随手定的）：

1. **文本视图只用标准库，matplotlib 按需导入。** ``mapping/pyproject.toml`` 的
   ``dependencies`` 只有本仓库的 ``llm-infer-model``、``dev`` 是空列表，两个包都
   **不引入第三方运行库**。所以本脚本刻意放在 ``tensor_mapping/`` **包外**、也不被
   ``tests/`` 导入——它是一把手工具，不是包的一部分。（同目录的
   ``demo_window_vs_search.py`` 是同一类工具：包外、只读、被手动执行。）
2. **不假设事件字段非空。** 实测 ``events.json`` 里 ``resource`` 与 ``start_ns`` 都
   可以是 ``null``：只有 ``COMPUTE``（``resource="gpu_compute"``）和 ``COPY_H2D``
   （``resource="h2d_copy"``）有真正的 resource 与 start/end；``EVICT`` 是
   ``resource=null, start_ns=end_ns=t_ns``；``ADVANCE`` 是 ``resource=null,
   start_ns=null, end_ns=t_ns``。容量上限则**不在** ``states.json``（它的 ``limits``
   是 ``null``），在 ``stats.json`` 的 ``limits.vram_capacity_bytes``。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# 读产物
# --------------------------------------------------------------------------

STATS, EVENTS, STATES = "stats.json", "events.json", "states.json"


@dataclass(frozen=True)
class Run:
    directory: Path
    stats: dict[str, Any]
    events: list[dict[str, Any]]
    states: list[dict[str, Any]]

    @property
    def label(self) -> str:
        return f"{self.stats.get('scenario_id')} [{self.stats.get('mode')}]"

    @property
    def makespan_ns(self) -> int | None:
        return self.stats.get("makespan_ns")

    @property
    def capacity_bytes(self) -> int | None:
        limits = self.stats.get("limits") or {}
        return limits.get("vram_capacity_bytes")

    @property
    def peak_bytes(self) -> int | None:
        if self.stats.get("peak_vram_bytes") is not None:
            return self.stats["peak_vram_bytes"]
        if not self.states:
            return None
        return max(state["used_vram_bytes"] for state in self.states)


def load_run(directory: Path) -> Run:
    """读一个结果目录。缺文件时直接抛，因为半套产物没法判读。"""
    payload: dict[str, Any] = {}
    for name in (STATS, EVENTS, STATES):
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"读不到 {path}——请先跑 python -m tensor_mapping，"
                             f"或用 --out 指向正确的目录")
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        # 四份产物都是「顶层带身份键 + 一个列表键」的壳，但 events/states 在为空时
        # 写的是裸 []，所以两种形状都要接。
        if name == STATS:
            payload["stats"] = document
        else:
            key = "events" if name == EVENTS else "states"
            payload[key] = document[key] if isinstance(document, dict) else document
    return Run(directory, payload["stats"], payload["events"], payload["states"])


# --------------------------------------------------------------------------
# 事件时间轴：把可空字段统一成半开区间
# --------------------------------------------------------------------------

#: 事件种类 → 画图用的字符 / 颜色 / 中文名
KIND_STYLE = {
    "COPY_H2D": ("=", "#3b7dd8", "H2D 搬运"),
    "COMPUTE": ("#", "#d8723b", "GPU 计算"),
    "EVICT": ("o", "#5aa469", "释放"),
    "ADVANCE": (".", "#999999", "推进"),
}

#: resource 字段 → 泳道名（按事件的 resource 值分道，null 归到「释放/推进」）
LANE_TITLES = {
    "h2d_copy": "H2D 搬运",
    "gpu_compute": "GPU 计算",
    None: "释放 / 推进",
}


def span_ns(event: dict[str, Any]) -> tuple[int, int]:
    """返回 ``(start_ns, end_ns)``；``ADVANCE`` 的 ``start_ns`` 是 null，退化成点。"""
    moment = event.get("t_ns") or 0
    start = event.get("start_ns")
    end = event.get("end_ns")
    return (moment if start is None else start, moment if end is None else end)


def lane_of(event: dict[str, Any]) -> str | None:
    return event.get("resource")


def lane_order(events: list[dict[str, Any]]) -> list[str | None]:
    order: list[str | None] = []
    for event in events:
        lane = lane_of(event)
        if lane not in order:
            order.append(lane)
    # 让「释放 / 推进」永远在最后一条道，读图时它是一层注记而不是一条资源
    if None in order:
        order.remove(None)
        order.append(None)
    return order


def lane_name(lane: str | None) -> str:
    return LANE_TITLES.get(lane, lane or "?")


def target_label(event: dict[str, Any]) -> str:
    return str(event.get("target_id") or "")


def display_width(text: str) -> int:
    """终端列宽。汉字占两列，``len()`` 只算 1，直接拿它做对齐会让带中文的泳道名错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


# --------------------------------------------------------------------------
# 文本视图（无依赖，可 diff、可贴报告）
# --------------------------------------------------------------------------


def render_text(run: Run, width: int = 80, stream: Any = None) -> None:
    out = stream or sys.stdout
    makespan = run.makespan_ns
    cap = run.capacity_bytes
    peak = run.peak_bytes

    print("=" * (width + 26), file=out)
    print(f"{run.label}   status={run.stats.get('status')}   "
          f"makespan={(makespan or 0)/1e6:.1f} ms   peak={peak} B   cap={cap} B", file=out)
    print("=" * (width + 26), file=out)

    if not run.events:
        note = run.stats.get("reason") or run.stats.get("status")
        print(f"\n【瀑布图】events 为空（{note}），本次运行没有动作可画。", file=out)
    else:
        _render_text_waterfall(run, width, out)

    _render_text_vram(run, width, out)


def _render_text_waterfall(run: Run, width: int, out: Any) -> None:
    makespan = run.makespan_ns or 1
    lanes = lane_order(run.events)
    label_width = max([display_width(lane_name(lane)) for lane in lanes] + [6])

    used = [kind for kind in KIND_STYLE
            if any(event["kind"] == kind for event in run.events)]
    legend = "  ".join(f"{KIND_STYLE[k][0]} = {k}" for k in used)
    print(f"\n【瀑布图】1 格 ≈ {makespan/width/1e6:.3f} ms      {legend}", file=out)
    print(f"  {'':<{label_width}} +{'-'*width}+", file=out)

    for lane in lanes:
        row = [" "] * width
        spans = [event for event in run.events if lane_of(event) == lane]
        for event in spans:
            start, end = span_ns(event)
            first = int(start / makespan * width)
            last = max(first + 1, round(end / makespan * width))
            for cell in range(first, min(last, width)):
                row[cell] = KIND_STYLE.get(event["kind"], ("?", ""))[0]
        busy = sum(end - start for start, end in map(span_ns, spans) if end > start)
        suffix = f"{len(spans)} 段" + (f", 忙 {busy/1e6:.1f} ms" if busy else "")
        print(f"  {pad(lane_name(lane), label_width)} |{''.join(row)}|  {suffix}", file=out)

    blank = " " * label_width
    print(f"  {blank} +{'-'*width}+", file=out)
    print(f"  {blank}  " + "".join("|" if i % 16 == 0 else " "
                                  for i in range(width)), file=out)
    ticks = "".join(f"{i*makespan/width/1e6:<16.1f}" if i % 16 == 0 else ""
                    for i in range(width))
    print(f"  {blank}  {ticks.strip()} ms", file=out)

    slow_lanes = [lane for lane in lanes if lane is not None]
    if len(slow_lanes) >= 2:
        print("\n  【重叠判定】两条资源道是否并行（同行同时刻占用）:", file=out)
        for i, a in enumerate(slow_lanes):
            for b in slow_lanes[i + 1:]:
                spans_a = [span_ns(e) for e in run.events if lane_of(e) == a]
                spans_b = [span_ns(e) for e in run.events if lane_of(e) == b]
                overlap = sum(min(y, v) - max(x, u)
                              for x, y in spans_a for u, v in spans_b if x < v and u < y)
                verdict = "← 并行" if overlap else "← 串行 (零重叠)"
                print(f"     {pad(lane_name(a), 10)} × {pad(lane_name(b), 10)} "
                      f"{overlap/1e6:>5.1f} ms  {verdict}", file=out)


def _render_text_vram(run: Run, width: int, out: Any) -> None:
    states = run.states
    cap = run.capacity_bytes
    peak = run.peak_bytes
    print(f"\n【显存曲线】{len(states)} 个状态，上限 {cap} B", file=out)
    if not states:
        print("  states 为空，没有曲线可画。", file=out)
        return

    top = max(cap or 0, peak or 0, 1)
    rows = 11
    for level in range(rows, -1, -1):
        threshold = top * level / rows
        line = "".join("#" if s["used_vram_bytes"] >= threshold else "."
                       for s in states)
        mark = "  ← 容量上限" if cap and abs(threshold - cap) < top / (2 * rows) else ""
        print(f"  {threshold:>5.0f} B |{line}|{mark}", file=out)
    print(f"  {'':>7}  +{'-'*len(states)}+  x = 状态序号（0 = 初始态，"
          f"i+1 = 施加 actions[i] 之后）", file=out)

    if cap is None:
        print(f"  peak={peak} B   cap 未知（stats.json 的 limits 缺失）", file=out)
    else:
        verdict = "OK 全程未超容量" if (peak or 0) <= cap else "!! 超出容量"
        print(f"  peak={peak} B   cap={cap} B   余量={cap-(peak or 0)} B   {verdict}",
              file=out)

    print("  逐状态：action_index |     t_ns    | used_vram |    Δ", file=out)
    previous = None
    for state in states:
        used = state["used_vram_bytes"]
        delta = "  —" if previous is None else f"{used - previous:+d}"
        print(f"    {str(state['action_index']):>4} | {state['t_ns']/1e6:>7.1f} ms | "
              f"{used:>6} B | {delta:>5}", file=out)
        previous = used


# --------------------------------------------------------------------------
# 一致性检查（「展示清楚」的另一半：顺手把可核对的量核一遍）
# --------------------------------------------------------------------------


def check(run: Run) -> list[tuple[bool, str]]:
    """返回 ``(是否通过, 说明)`` 列表。只查产物自身的自洽性，不重算语义。"""
    results: list[tuple[bool, str]] = []
    cap = run.capacity_bytes
    peak = run.peak_bytes

    if cap is not None and peak is not None:
        results.append((peak <= cap, f"peak_vram_bytes ({peak} B) ≤ 容量上限 ({cap} B)"))
    if cap is not None and run.states:
        worst = max(s["used_vram_bytes"] for s in run.states)
        results.append((worst <= cap,
                        f"每个状态的 used_vram ≤ 容量上限（最大 {worst} B）"))

    if run.states:
        times = [s["t_ns"] for s in run.states]
        results.append((all(b >= a for a, b in zip(times, times[1:])),
                        "states 的 t_ns 单调不减"))
    if run.events:
        indices = [e["action_index"] for e in run.events]
        results.append((indices == sorted(indices),
                        "events 的 action_index 递增（同刻动作按序号保留）"))
        results.append((indices == list(range(len(indices))),
                        f"events 的 action_index 连续 0..{len(indices)-1}"))
    if run.events and run.states:
        results.append((len(run.states) == len(run.events) + 1,
                        f"len(states) == len(events) + 1 "
                        f"（{len(run.states)} == {len(run.events)} + 1）"))
    makespan = run.makespan_ns
    if makespan is not None and run.events:
        latest = max(span_ns(e)[1] for e in run.events)
        results.append((latest <= makespan,
                        f"最晚事件结束 ({latest/1e6:.1f} ms) ≤ makespan "
                        f"({makespan/1e6:.1f} ms)"))
    return results


def render_check(run: Run, stream: Any = None) -> bool:
    out = stream or sys.stdout
    print("\n【一致性检查】", file=out)
    results = check(run)
    if not results:
        print("  产物为空，没有可检查的项。", file=out)
        return True
    for ok, message in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}", file=out)
    return all(ok for ok, _ in results)


# --------------------------------------------------------------------------
# PNG（唯一需要 matplotlib 的地方，按需导入）
# --------------------------------------------------------------------------


def _configure_cjk_font() -> None:
    """中文字形。Windows 上用雅黑；缺字体装的是方框，所以给一条真正的回退链。"""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                      "Source Han Sans SC", "PingFang SC", "Arial Unicode MS"):
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def render_png(run: Run, outdir: Path, dpi: int = 140) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")           # 无窗口——本脚本不弹任何 UI
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _configure_cjk_font()
    outdir.mkdir(parents=True, exist_ok=True)
    makespan = run.makespan_ns or 1
    cap = run.capacity_bytes
    written: list[Path] = []

    # ---- 图一：瀑布图（Gantt） ----
    lanes = lane_order(run.events)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * max(len(lanes), 1) + 1.5), dpi=dpi)
    ypos = {lane: i for i, lane in enumerate(reversed(lanes))}
    for lane in lanes:
        for event in run.events:
            if lane_of(event) != lane:
                continue
            start, end = span_ns(event)
            _, color, _ = KIND_STYLE.get(event["kind"], ("?", "#cccccc", event["kind"]))
            y = ypos[lane]
            if end > start:
                ax.broken_barh([(start / 1e6, (end - start) / 1e6)], (y - 0.32, 0.64),
                               facecolors=color, edgecolors="white", linewidth=0.6)
                if (end - start) > makespan * 0.05:
                    ax.text((start + end) / 2e6, y, target_label(event),
                            ha="center", va="center", fontsize=7, color="white")
            else:   # 零宽事件（EVICT / ADVANCE）画成竖线注记
                ax.plot([start / 1e6] * 2, [y - 0.32, y + 0.32],
                        color=color, linewidth=1.4, solid_capstyle="butt")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([lane_name(l) for l in reversed(lanes)])
    ax.set_ylim(-0.7, len(lanes) - 0.3)
    ax.set_xlim(0, makespan / 1e6)
    ax.set_xlabel("时间 (ms)")
    ax.set_title(f"瀑布图 — {run.label}   makespan {makespan/1e6:.1f} ms", fontsize=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    present = [k for k in KIND_STYLE if any(e["kind"] == k for e in run.events)]
    ax.legend(handles=[Patch(facecolor=KIND_STYLE[k][1], label=KIND_STYLE[k][2])
                       for k in present],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(present),
              frameon=False, fontsize=8)
    figure_path = outdir / "waterfall.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    written.append(figure_path)

    # ---- 图二：显存曲线（对时间，与瀑布图共用 x 轴） ----
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=dpi)
    if run.states:
        times = [s["t_ns"] / 1e6 for s in run.states]
        used = [s["used_vram_bytes"] for s in run.states]
        ax.step(times, used, where="post", color="#3b7dd8", linewidth=1.8,
                label="已用显存")
        ax.fill_between(times, used, step="post", color="#3b7dd8", alpha=0.16)
        if cap is not None:
            ax.axhline(cap, color="#c0392b", linestyle="--", linewidth=1.4,
                       label=f"容量上限 {cap} B")
            ax.fill_between([0, makespan / 1e6], cap, max(cap, max(used)) * 1.15,
                            color="#c0392b", alpha=0.07)
    if cap is not None:
        ax.set_ylim(0, max(cap, (run.peak_bytes or 0)) * 1.15)
    ax.set_xlim(0, makespan / 1e6)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("已用显存 (B)")
    ax.set_title(f"显存曲线 — {run.label}   peak {run.peak_bytes} / cap {cap} B",
                 fontsize=10)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    figure_path = outdir / "vram.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    written.append(figure_path)
    return written


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_results",
        description="把 tensor_mapping 的结果目录画成瀑布图与显存曲线（只读产物）",
    )
    parser.add_argument("results", type=Path,
                        help="结果目录，即 python -m tensor_mapping --out 指向的那个")
    parser.add_argument("--png", action="store_true",
                        help="额外写出 waterfall.png 与 vram.png（需要 matplotlib）")
    parser.add_argument("--out", type=Path, default=None,
                        help="PNG 的输出目录，默认与结果目录相同")
    parser.add_argument("--dpi", type=int, default=140, help="PNG 分辨率，默认 140")
    parser.add_argument("--quiet", action="store_true", help="不打印文本视图")
    args = parser.parse_args(argv)

    run = load_run(args.results)
    if not args.quiet:
        render_text(run)
    ok = render_check(run)

    if args.png:
        outdir = args.out or args.results
        try:
            written = render_png(run, outdir, args.dpi)
        except ImportError:
            print("\n需要 matplotlib 才能出 PNG：pip install matplotlib\n"
                  "（文本视图不需要任何依赖，上面那份就够判读了）", file=sys.stderr)
            return 3
        print("\n【PNG】")
        for path in written:
            print(f"  {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
