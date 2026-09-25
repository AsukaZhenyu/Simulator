#!/usr/bin/env python3
"""三份结果并排看：旧窗口模型、新窗口策略、搜索。

用法::

    python tools/demo_window_vs_search.py
    python tools/demo_window_vs_search.py --scenario examples/three_layer_chain-cap1024.json
    python tools/demo_window_vs_search.py --windows 1 2 3

``ALIGNMENT_IMPLEMENTATION.md`` §6-B 要求「旧窗口策略接进新内核」这件事**可以被看到**，
§6-C 要求搜索产生的动作交给新内核回放后数字一致。本脚本就把这两句话跑成一张表：左边
是 ``llm_infer_model.simulator.simulate_decode``（v0.8 的窗口规则），右边是
``tensor_mapping.policies.window_mapping`` + ``llm_infer_model.tensor.evaluate_mapping``
（新内核），最后一段是 ``tensor_mapping.mapper.search`` 的通用解。

它**只读**示例、不写任何产物、不改任何语义，也不被 ``tests/`` 导入——和
``tools/plot_results.py`` 一样是包外的一把手工具。任何一项对不上就以非零码退出，所以它
也可以直接当验收命令跑。

四处需要说明的对照口径：

* **成本走公式，不走实测覆盖**。示例里写死的 ``synthetic_fixed`` 值必须与「4×4 权重
  作用于 4 维向量 = 16 次乘加 = 32 FLOP」这条闭式算出来的值逐项相等。给定
  ``gpu_effective_flops=16000``、``h2d_latency_s=1e-3``、``h2d_bandwidth=32 kB/s``，
  公式正好给出 2 ms 与 3 ms——和示例写死的数一样。于是这张表同时证明了「成本来自旧
  模型」和「换算是无损的」。
* **权重峰值**：旧模型的 ``peak_streamed_weight_bytes`` 只数权重；新内核的
  ``peak_vram_bytes`` 是**总**占用（含激活与 workspace）。两者本来就不该相等，所以
  这里从 ``states`` 单独数一遍链上权重再比，总占用另列一栏。
* **K = 层数**：旧模型在 ``window_size >= layer_count`` 时走一条专门的「全部常驻」分支
  （``simulator.py:90``），一次搬运都不记。新内核没有这条捷径，等价的场景是把三份权重
  的 ``initial_locations`` 改成 ``["vram"]``——常驻是 **scenario 的属性**，不是窗口策略
  给自己开的后门。
* **拒绝的类型要分清**：K 小于初始驻留份数时抛 ``WindowPolicyFailed``（策略走不下去），
  残差/分叉抛 ``Unsupported``（图形状不支持）。两者都**不是** ``infeasible``。
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # 允许直接 `python tools/demo_window_vs_search.py`
    sys.path.insert(0, str(REPO_ROOT))

try:
    from llm_infer_model.model import HardwareSpec, LayerSpec, ModelConfig, PolicySpec
    from llm_infer_model.simulator import simulate_decode
    from llm_infer_model.tensor import (
        COPY_ABSENT,
        LOC_VRAM,
        evaluate_mapping,
        load_and_validate,
    )
    from llm_infer_model.tensor.layer_costs import costs_from_model_config
except ModuleNotFoundError as exc:  # pragma: no cover - 环境问题不是逻辑问题
    raise SystemExit(
        "需要先安装两个本地包（必须在同一次 pip 调用里给出，否则 pip 会去 PyPI 找 "
        "llm-infer-model）：\n"
        "    python -m pip install -e ./modeling -e ./mapping\n"
        f"原始错误：{exc}"
    ) from exc

from tensor_mapping.mapper import search
from tensor_mapping.policies import (
    WindowPolicyFailed,
    chain_weight_ids,
    derive_chain,
    window_mapping,
)
from tensor_mapping.spec import SpecError, Unsupported

EXAMPLES = REPO_ROOT / "examples"

# 示例 scenario 里三层各自的 FLOPs：4×4 的权重乘 4 维向量 = 16 次乘加 = 32 FLOP。
FLOPS_PER_LAYER = 32.0
WEIGHT_BYTES = 64

# 一组能复现示例那两行成本的硬件参数。带宽 32 kB/s 是刻意取小的：64 B / 32 kB/s 恰好
# 是 1 ms，加上 1 ms 的固定延迟就是示例里的 3 ms。
_HARDWARE = HardwareSpec(
    gpu_effective_flops=16_000.0,
    h2d_bandwidth_bytes_per_s=32_000.0,
    h2d_latency_s=1e-3,
    vram_capacity_bytes=1024,
)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def display_width(text: str) -> int:
    """终端列宽。汉字占两列，直接拿 ``len()`` 对齐会让带中文的表头错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def ms(nanoseconds: int | None) -> str:
    return "—" if nanoseconds is None else f"{nanoseconds / 1e6:g} ms"


def hardware_for(scenario) -> HardwareSpec:
    return replace(_HARDWARE, vram_capacity_bytes=scenario.architecture.vram_capacity_bytes)


def layers_by_operation(scenario) -> dict[str, LayerSpec]:
    """算子 ID → 该层的 :class:`LayerSpec`，走公式而非实测覆盖。"""
    return {
        operation_id: LayerSpec(
            name=f"blk.{index}.attn_q",
            weight_bytes=WEIGHT_BYTES,
            flops=FLOPS_PER_LAYER,
        )
        for index, operation_id in enumerate(sorted(scenario.workload.operation_by_id))
    }


def old_config(scenario, window_size: int, layers=None) -> ModelConfig:
    """与新 scenario 等价的旧配置：同硬件、同层数、同公式。

    刻意不填任何 ``measured_*``：这样 ``_assert_no_extra_services`` 一项都不会被点亮，
    旧模型也就只剩下「窗口换入换出 + 计算」——正是新内核已经接管的那件事。填了实测值
    反而会让这张表退化成「两边都照抄同一个常数」。
    """
    return ModelConfig(
        name=scenario.id,
        layers=tuple(layers.values()) if layers is not None else tuple(
            layers_by_operation(scenario).values()
        ),
        hardware=hardware_for(scenario),
        policy=PolicySpec(window_size=window_size),
    )


def weight_peak_bytes(scenario, evaluation) -> int:
    """链上权重在整段回放里的最大占用（新内核只报总占用，这里单独数权重）。"""
    weights = set(chain_weight_ids(scenario))
    peak = 0
    for snapshot in evaluation.states:
        resident = sum(
            scenario.size_alloc(tensor_id)
            for tensor_id, status in snapshot.copy_status
            if tensor_id in weights and status != COPY_ABSENT
        )
        peak = max(peak, resident)
    return peak


def make_resident(scenario):
    """把链上权重改成「一开始就在显存里」，对上旧模型的常驻分支。"""
    weights = set(chain_weight_ids(scenario))
    tensors = tuple(
        replace(tensor, initial_locations=(LOC_VRAM,)) if tensor.id in weights else tensor
        for tensor in scenario.workload.tensors
    )
    return replace(
        scenario,
        id=f"{scenario.id}-resident",
        workload=replace(scenario.workload, tensors=tensors),
    )


# --------------------------------------------------------------------------
# 五个对照
# --------------------------------------------------------------------------


def check_cost_bridge(scenario, failures: list[str]) -> None:
    """示例写死的成本，必须等于经旧层模型求值再换算出来的成本。"""
    layers = layers_by_operation(scenario)
    bridge = costs_from_model_config(old_config(scenario, 1, layers), scenario.workload, layers)

    print("① 成本桥接：示例里的 synthetic_fixed vs LayerSpec 闭式求值后换算")
    print(f"   source: {scenario.costs.source} → {bridge.source}")
    agrees = True
    for operation_id in sorted(scenario.costs.compute):
        fixture = scenario.costs.compute[operation_id].duration_ns
        derived = bridge.compute[operation_id].duration_ns
        agrees &= fixture == derived
        print(f"   compute {operation_id}: {fixture:>9} ns  {_mark(fixture == derived)} {derived} ns")
    for tensor_id in sorted(scenario.costs.h2d):
        fixture = scenario.costs.h2d[tensor_id].duration_ns
        derived = bridge.h2d[tensor_id].duration_ns
        agrees &= fixture == derived
        print(f"   h2d     {tensor_id}: {fixture:>9} ns  {_mark(fixture == derived)} {derived} ns")
    print(f"   → {'一致' if agrees else '不一致'}")
    if not agrees:
        failures.append("成本桥接与示例不一致")


def _mark(ok: bool) -> str:
    return "==" if ok else "!="


def check_window_table(scenario, window_sizes, failures: list[str]) -> None:
    """每个 K 一行：旧模型 vs 新窗口回放。"""
    chain_length = len(chain_weight_ids(scenario))
    rows = [(k, scenario, False) for k in window_sizes if k < chain_length]
    rows.append((chain_length, make_resident(scenario), True))

    print()
    print("② 窗口 K：旧规则 simulate_decode vs 新策略 window_mapping + evaluate_mapping")
    print(
        "   " + pad("K", 4) + pad("形态", 8)
        + pad("旧 makespan", 12) + pad("旧 搬运", 9) + pad("旧 权重峰", 10)
        + "| " + pad("新 makespan", 12) + pad("新 搬运", 9) + pad("新 权重峰", 10)
        + pad("新 总峰", 9) + "判定"
    )
    for window_size, target, is_resident in rows:
        old = simulate_decode(old_config(target, window_size))
        actions = window_mapping(target, window_size)
        new = evaluate_mapping(target, actions)
        new_weight_peak = weight_peak_bytes(target, new)

        old_makespan = round(old.makespan_seconds * 1e9)
        agrees = (
            old_makespan == new.makespan_ns
            and old.bytes_transferred == new.h2d_bytes
            and old.peak_streamed_weight_bytes == new_weight_peak
        )
        print(
            "   " + pad(str(window_size), 4) + pad("常驻" if is_resident else "流式", 8)
            + pad(ms(old_makespan), 12) + pad(str(old.bytes_transferred), 9)
            + pad(str(old.peak_streamed_weight_bytes), 10)
            + "| " + pad(ms(new.makespan_ns), 12) + pad(str(new.h2d_bytes), 9)
            + pad(str(new_weight_peak), 10) + pad(str(new.peak_vram_bytes), 9)
            + ("一致" if agrees else "不一致")
        )
        if not agrees:
            failures.append(
                f"K={window_size}（{'常驻' if is_resident else '流式'}）两侧数字不一致："
                f"旧 {old_makespan} ns / {old.bytes_transferred} B / "
                f"{old.peak_streamed_weight_bytes} B，新 {new.makespan_ns} ns / "
                f"{new.h2d_bytes} B / {new_weight_peak} B"
            )


def check_k_precondition(scenario, failures: list[str]) -> None:
    """K 小于初始驻留份数时，必须由策略自己拒绝，而不是让内核报无解。"""
    resident = make_resident(scenario)
    chain_length = len(chain_weight_ids(scenario))

    print()
    print("③ 窗口前提：初始驻留份数 > K")
    for window_size in range(1, chain_length):
        try:
            window_mapping(resident, window_size)
        except WindowPolicyFailed as error:
            print(f"   K={window_size}: WindowPolicyFailed（策略走不下去，不是 infeasible）")
            print(f"         {error}")
        except SpecError as error:
            failures.append(
                f"K={window_size} 抛了 {type(error).__name__} 而不是 WindowPolicyFailed"
            )
            print(f"   K={window_size}: 期望 WindowPolicyFailed，实得 {type(error).__name__}: {error}")
        else:
            failures.append(f"K={window_size} 在 3 份权重常驻的场景上竟然产出了计划")
            print(f"   K={window_size}: 期望拒绝，实际产出了计划")


def check_unsupported(failures: list[str]) -> None:
    """残差/分叉这样的结构要明确拒绝，而不是给一条错计划。"""
    print()
    print("④ 窗口策略能读哪些形状（通用搜索不受这些限制）")
    for path in sorted(EXAMPLES.glob("*-cap160.json")):
        try:
            other = load_and_validate(path)
        except SpecError as error:
            failures.append(f"{path.name} 没通过校验：{error}")
            print(f"   {path.name}: 载入失败 — {error}")
            continue
        try:
            chain = derive_chain(other.workload)
        except Unsupported as error:
            print(f"   {path.name}: Unsupported — {error}")
        else:
            weights = " → ".join(link.weight_id for link in chain)
            print(f"   {path.name}: 可读，{len(chain)} 环（{weights}）")


def check_search(scenario, failures: list[str]) -> None:
    """搜索的动作交给独立评估器回放，数字必须与搜索自己报的一致。"""
    print()
    print("⑤ 搜索：通用解，以及它的独立回放")
    result = search(scenario)
    print(
        f"   status={result.status}  reason={result.termination_reason}  "
        f"optimality_proven={result.optimality_proven}  "
        f"expanded={result.expanded_states}  visited={result.visited_states}"
    )
    if result.evaluation is None:
        failures.append(f"搜索没有产出可回放的计划：{result.status}/{result.termination_reason}")
        print("   没有可回放的计划")
        return

    replay = evaluate_mapping(scenario, result.actions)
    agrees = (
        replay.makespan_ns == result.makespan_ns
        and replay.peak_vram_bytes == result.peak_vram_bytes
        and replay.h2d_bytes == result.h2d_bytes
        and replay.status == result.evaluation.status
        and replay.events == result.evaluation.events
    )
    print(
        f"   search 自身评估：makespan={ms(result.makespan_ns)}  峰={result.peak_vram_bytes} B  "
        f"搬运={result.h2d_bytes} B  动作={len(result.actions)}"
    )
    print(
        f"   独立回放：      makespan={ms(replay.makespan_ns)}  峰={replay.peak_vram_bytes} B  "
        f"搬运={replay.h2d_bytes} B  动作={len(replay.actions)}"
    )
    print(f"   → {'逐字段一致（含 events）' if agrees else '不一致'}")
    if not agrees:
        failures.append("搜索结果与独立回放不一致")

    # 搜索给最优解，窗口策略给一条可行解，两者可以不同——列出来是为了让人看见差异。
    window = evaluate_mapping(scenario, window_mapping(scenario, 2))
    print(
        f"   对照：窗口策略 K=2 的可行解 {ms(window.makespan_ns)}/峰 {window.peak_vram_bytes} B；"
        f"搜索最优 {ms(result.makespan_ns)}/峰 {result.peak_vram_bytes} B"
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="旧窗口模型 / 新窗口策略 / 搜索 三份结果并排对照")
    parser.add_argument(
        "--scenario", type=Path,
        default=EXAMPLES / "three_layer_chain-cap1024.json",
        help="三链示例 scenario，默认 examples/three_layer_chain-cap1024.json",
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[1, 2],
        help="要对照的窗口大小，默认 1 2；常驻那行由链长决定，会自动追加",
    )
    args = parser.parse_args(argv)

    scenario = load_and_validate(args.scenario)
    chain = derive_chain(scenario.workload)

    print(f"示例：{args.scenario}")
    print(
        f"链：{len(chain)} 环，每份权重 {WEIGHT_BYTES} B，"
        f"计算 {FLOPS_PER_LAYER} FLOP，显存 {scenario.architecture.vram_capacity_bytes} B"
    )

    failures: list[str] = []
    check_cost_bridge(scenario, failures)
    check_window_table(scenario, args.windows, failures)
    check_k_precondition(scenario, failures)
    check_unsupported(failures)
    check_search(scenario, failures)

    print()
    if failures:
        print(f"✗ {len(failures)} 项对不上：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("✓ 全部对照一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
