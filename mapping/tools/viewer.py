#!/usr/bin/env python3
"""轻量查看器：同屏看 mapping 的搜索与 modeling 的执行。

用法::

    cd mapping
    python tools/viewer.py --scenario examples/three_layer_chain-cap1024.json
    python tools/viewer.py --scenario examples/chain-cap160.json \\
        --config ../modeling/configs/toy_dense_decode.json
    python tools/viewer.py --self-check          # 无头自检，退出 0/1

**最省事的是双击 ``mapping/viewer.cmd``**（参数一个都不用给，scenario 与 config 都有
默认值），它会自己找到 venv 的解释器并打开浏览器。这里用哪个 python 起都行：解释器里
没有那两个本地包时，本文件会**自己换到** ``modeling/.venv`` 重跑一遍（只多打一行说明）。

起一个只绑 ``127.0.0.1`` 的本地服务，用浏览器看五块屏：计算图、动作时间线、
搜索树、成本桥（modeling 的逐层预测怎么变成一个整数纳秒的动作）、参数（改旋钮重跑
搜索并与改动前对比）。页面上有「导出快照」按钮，存出一个自包含的单文件 HTML——
双击就能看、能发给人，不需要 Python。

为什么在这里（而不是包内）
--------------------------
本文件与 ``plot_results.py`` / ``demo_window_vs_search.py`` 同一条纪律：**包外、只读、
不被 ``tests/`` 导入**。落实「不被导入」的办法就是**不可导入**——``tools/`` 下至今没有
``__init__.py``，往后再加也不能加：一旦有，``import viewer`` 就合法了，零依赖纪律迟早
从那里漏进包里。（所以本文件要 import 同目录的 ``viewer_payload`` 得自己插 ``sys.path``，
见下面那段。）

覆盖了哪条旧口径
----------------
``ALIGNMENT_IMPLEMENTATION.md:167``、``mapping/README.md`` 与 ``DESIGN.md:45`` 都写着
「不开发 GUI」。``DESIGN.md`` 禁的是**依赖**（求解器 / 网络服务 / GPU 运行依赖），
本查看器不新增任何依赖——只用标准库，两个 ``pyproject.toml`` 都没动。而「不做 GUI」
那条的理由是「文本输出是否已足够理解执行过程」，回答的是**验证引擎**；「看见并操作
搜索空间」是另一个问题，判据换成「文本表格够不够**操作**搜索空间」——不够。

三块屏的数据从哪来
------------------
``viewer_payload`` 负责装配，本文件只负责跑起来、路由、以及**自检**。自检是这个工具
唯一的测试（``tools/`` 不进 ``tests/``），按 ``demo_window_vs_search.py`` 的 ``✓/✗``
格式，退出 0/1，不需要浏览器也不需要网络。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# 允许直接 `python tools/viewer.py`：``tools/`` 不是包（刻意没有 ``__init__.py``），所以
# 同目录的 ``viewer_payload`` 只能靠把本目录挂上 sys.path 才 import 得到。范式照抄
# ``demo_window_vs_search.py:46-48``。
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENV_PYTHON = REPO_ROOT.parent / "modeling" / ".venv" / "Scripts" / "python.exe"
RELAUNCH_FLAG = "LLM_INFER_VIEWER_RELAUNCHED"


def _relaunch_into_venv() -> int | None:
    """在 venv 的解释器里重跑自己。返回子进程的退出码；换不了返回 ``None``。

    本机有两个 python：conda base（12 个环境、求解器已坏，**不该动**）和
    ``modeling/.venv``（两个包 editable 装在那里）。用错那个是这个工具最常见的失败，
    而且报错本身帮不上忙——用户会去给 base 装包，那正是不能做的事。与其教人查该用哪个，
    不如自己换过去：``python tools/viewer.py`` 于是在任何解释器下都能跑，双击也行。

    为什么不用 ``os.execv``：Windows 上它把 argv 用**空格拼成命令行且不加引号**，于是路径
    里的空格把参数拆散——实测踩到，报 ``can't open file
    '...\\mapping\\degree\\LLM-Infer\\...'``（本仓库路径里正好有 ``master degree``）。
    ``subprocess`` 会正确加引号，代价是多一层进程，Ctrl+C 两边都收得到。

    ``RELAUNCH_FLAG`` 是防死循环的哨兵：换过去之后 import 仍然失败时，必须**报错**而不是
    再换一次。
    """
    if os.environ.get(RELAUNCH_FLAG) or not VENV_PYTHON.exists():
        return None
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return None
    print(
        f"（当前解释器 {sys.executable} 里没有这两个包，改用 {VENV_PYTHON} 重跑）",
        flush=True,
    )
    os.environ[RELAUNCH_FLAG] = "1"
    try:
        return subprocess.call(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    except OSError:
        return None
    except KeyboardInterrupt:  # 父进程跟着 Ctrl+C 一起收到，别打一堆 traceback
        return 130


try:
    from llm_infer_model.model import load_config
    from llm_infer_model.tensor import load_and_validate
    from llm_infer_model.tensor.spec import SpecError
except ModuleNotFoundError as exc:  # pragma: no cover - 环境问题不是逻辑问题
    status = _relaunch_into_venv()
    if status is not None:  # 已经用 venv 跑过一遍了，把它的退出码原样带出去
        raise SystemExit(status)
    raise SystemExit(
        f"当前解释器 {sys.executable} 里没有这两个本地包，而且没法自动换到 venv。\n\n"
        f"在 Simulator 根目录建一个 venv 并装上（不要装进 conda base）：\n"
        f"    python -m venv modeling/.venv\n"
        f"    modeling/.venv/Scripts/python -m pip install -e ./modeling -e ./mapping\n\n"
        "两个 -e 必须在同一次 pip 调用里给出，否则 pip 会去 PyPI 找 llm-infer-model。\n\n"
        f"原始错误：{exc}"
    ) from exc

from tensor_mapping.mapper import search

import viewer_payload as vp

EXAMPLES = REPO_ROOT / "examples"
CONFIGS = REPO_ROOT.parent / "modeling" / "configs"
STATIC = HERE / "viewer_static"

DEFAULT_SCENARIO = EXAMPLES / "chain-cap160.json"
# 曾经是 ``--config`` 的默认值。现在默认是 ``None``（见 ``_effective_config``：不给就现造
# 一份「除硬件外一切服务都关着」的配置，时长来源才默认是 modeling 推导，而不是一上来就
# 展示门 1 的拒绝）。这份配置只剩一个用途：自检里当**触发门 1 的样本**。
DEFAULT_CONFIG = CONFIGS / "toy_dense_decode.json"
# 双击即用的启动器，就在 mapping/ 下、和本工具平级。它坏起来**没有任何症状**：自检照样
# 全绿，只是双击之后窗口闪一下就没了——所以下面 check_launcher() 专门钉它三条硬约束。
LAUNCHER = REPO_ROOT / "viewer.cmd"

# 快照内联时**不做通用转义器，改做断言**：与其写一个「大概能转义」的函数，不如把
# 「这几份文件里不许出现这些串」变成自检里的一条。断言过了，拼接就是安全的；哪天有人
# 往 app.js 里写了一句 `"</script>"`，自检会当场红，而不是产出一个静默坏掉的 HTML。
SNAPSHOT_FORBIDDEN = {"app.js": ("</script",), "style.css": ("</style", "<!--")}

# ``index.html`` 里要被换成内联内容的两个**字面**标签。必须与那个文件里写的一模一样，
# 所以自检 D 组要断言它们在 index.html 里各出现恰好一次——少一次是拼不上（快照掉样式
# 或掉脚本），多一次是替换了不该替换的地方。改 index.html 时这两个常量要跟着改。
LINK_TAG = '<link rel="stylesheet" href="style.css">'
SCRIPT_TAG = '<script src="app.js"></script>'


# ---------------------------------------------------------------------------
# 参数编辑：全程在内存里
# ---------------------------------------------------------------------------


def edit_scenario(scenario: Any, params: dict[str, Any]) -> Any:
    """按表单把 scenario 改出新的一份，**不写任何临时文件**。

    没有公开的 dict → Scenario（``spec._parse_*`` 是私有的，``load_and_validate`` 只吃
    路径），但 ``Scenario``/``Architecture``/``Costs``/``Mapspace`` 全是 frozen dataclass，
    所以 ``dataclasses.replace`` 够用。``replace`` 会重跑 ``__post_init__``：手填的非法
    参数（比如容量 0）会当场抛 ``InvalidInput``——这不是要吞掉的麻烦，它本身就是要显示
    的一类拒绝（门 3）。
    """
    architecture = scenario.architecture
    if "vram_capacity_bytes" in params:
        architecture = replace(
            architecture, vram_capacity_bytes=int(params["vram_capacity_bytes"])
        )
    if "runtime_reserved_bytes" in params:
        architecture = replace(
            architecture, runtime_reserved_bytes=int(params["runtime_reserved_bytes"])
        )

    mapspace = scenario.mapspace
    for key in ("allow_copy_compute_overlap", "allow_eviction"):
        if key in params:
            mapspace = replace(mapspace, **{key: bool(params[key])})

    costs = scenario.costs
    overrides = params.get("compute_ns") or {}
    if overrides:
        compute = dict(costs.compute)
        for operation_id, nanoseconds in overrides.items():
            if operation_id not in compute:
                continue
            compute[operation_id] = replace(compute[operation_id], duration_ns=int(nanoseconds))
        costs = replace(costs, compute=compute)

    return replace(scenario, architecture=architecture, mapspace=mapspace, costs=costs)


def budget_from(params: dict[str, Any], fallback_states: int, fallback_time: float) -> tuple[int, float]:
    states = params.get("max_expanded_states")
    seconds = params.get("wall_time_limit_s")
    return (
        fallback_states if states is None else int(states),
        fallback_time if seconds is None else float(seconds),
    )


# 同一个 scenario 的另外两种**形态**，不是另一张图：``*.workload.json`` 是喂给 modeling 的
# 那份工作负载，``*.mapping.json`` 是 mapping 自己的产物。把它们列进「换计算图」的名单，点下
# 去只会得到一次「这份不是 scenario」的拒绝，白占一格。
SCENARIO_ARTIFACT_SUFFIXES = (".workload.json", ".mapping.json")


def scenario_catalog(scenario_path: str | None) -> list[dict[str, str]]:
    """``scenario_path`` 所在目录里**可以互相切换**的那些 scenario。

    只列同一目录下的兄弟文件，不做全仓扫描：``--scenario examples/chain-cap160.json`` 已经
    说清了「这一批」是哪些，换成别的目录里的东西要么得另拼路径、要么根本不是同一类东西。

    **只列名字，不预先载入。** 载不进来的那份（手改坏了 JSON / 不是 Scenario）要留在名单
    里——它明明白白躺在那个目录里，从屏上抹掉它才是骗人；点下去会得到一次带原因的 422，
    横幅就落在那份文件上。所以这里不做「启动时把它们都验一遍」，那只会把一次可读的拒绝
    变成一次静默的消失。
    """
    if not scenario_path:
        return []
    directory = os.path.dirname(scenario_path)
    folder = Path(directory or ".")
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.glob("*.json")):
        if path.name.endswith(SCENARIO_ARTIFACT_SUFFIXES):
            continue
        # 拼显示路径时**沿用调用者给的写法**（``examples/x.json`` 与 ``examples\x.json``
        # 是同一个文件，但顶栏上突然换一种斜杠会像是换了目录）。
        items.append({
            "name": path.stem,
            "path": f"{directory}/{path.name}" if directory else path.name,
        })
    return items


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


class Checks:
    """``✓/✗`` 行收集器。任何一条 ✗ 都让进程退出 1。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"label": label, "ok": bool(ok), "detail": detail})
        mark = "✓" if ok else "✗"
        line = f"  {mark} {label}"
        if detail and not ok:
            line += f"\n      {detail}"
        elif detail:
            line += f"  （{detail}）"
        print(line)
        return bool(ok)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if not row["ok"]]


class FakeClock:
    """永远「已经超时」的时钟，用来确定性地走一遍 ``time_budget_exhausted``。

    真时钟在 Windows 上做不到这件事：``time.monotonic()`` 实测是 ``GetTickCount64()``，
    分辨率 15.625 ms，所以 ``wall_time_limit_s=1e-9`` 这种极小预算第一次检查时
    ``elapsed_s()`` 仍然返回 0.0，**永远不触发**（``mapping/README.md`` 记了这条）。
    """

    def elapsed_s(self) -> float:
        return 1e9

    def expired(self) -> bool:
        return True


def _load(name: str) -> Any:
    return load_and_validate(EXAMPLES / f"{name}.json")


def check_walk(checks: Checks) -> None:
    """A 组：走查与 ``mapper.search()`` 逐字段对账。

    这是搜索树那一屏唯一的回归网。走查与内核共用 ``transition``，所以分叉只可能是这段
    重走写错了；对不上就要**禁用那一屏**，不能画一棵可能是错的树。
    """
    print("A · 搜索走查 == mapper.search()")
    cases: list[tuple[str, str, dict[str, Any]]] = [
        ("matvec-cap160", "matvec-cap160", {}),
        ("chain-cap160", "chain-cap160", {}),
        ("fork-cap160", "fork-cap160", {}),
        ("residual-cap160", "residual-cap160", {}),
        ("three_layer_chain-cap1024", "three_layer_chain-cap1024", {}),
        ("chain-cap160 k=28", "chain-cap160", {"max_expanded_states": 28}),
        ("chain-cap160 k=30（恰好够）", "chain-cap160", {"max_expanded_states": 30}),
        ("chain-cap160 k=8", "chain-cap160", {"max_expanded_states": 8}),
        ("three_layer_chain k=100", "three_layer_chain-cap1024", {"max_expanded_states": 100}),
    ]
    for label, fixture, kwargs in cases:
        block = vp.explore(_load(fixture), **kwargs)
        verdict = block["verdict"]
        reconcile = block["reconcile"]
        checks.check(
            f"{label}：{verdict['status']} / {verdict['termination_reason']}"
            f" / exp={verdict['expanded_states']} / {block['solution']['makespan_ns']} ns",
            reconcile["agrees"] is True,
            "；".join(reconcile["differences"]) or f"与 search 一致（{len(block['nodes'])} 节点）",
        )

    # 起点就装不下：展开 0 个**不是**「没跑」，而是一条可证的结论。
    small = replace(
        _load("chain-cap160"),
        architecture=replace(_load("chain-cap160").architecture, vram_capacity_bytes=1),
    )
    block = vp.explore(small)
    verdict = block["verdict"]
    checks.check(
        "chain cap=1 → infeasible / initial_capacity_exceeded / 可证 / 零节点",
        verdict["status"] == "infeasible"
        and verdict["termination_reason"] == "initial_capacity_exceeded"
        and verdict["optimality_proven"] is True
        and verdict["expanded_states"] == 0
        and block["nodes"] == []
        and block["reconcile"]["agrees"] is True,
        f"{verdict['status']} / {verdict['termination_reason']} / {len(block['nodes'])} 节点",
    )

    tight = replace(
        _load("chain-cap160"),
        architecture=replace(_load("chain-cap160").architecture, vram_capacity_bytes=80),
    )
    block = vp.explore(tight)
    verdict = block["verdict"]
    checks.check(
        "chain cap=80 → infeasible / search_space_exhausted / exp=5",
        verdict["status"] == "infeasible"
        and verdict["termination_reason"] == "search_space_exhausted"
        and verdict["expanded_states"] == 5
        and block["reconcile"]["agrees"] is True,
        f"exp={verdict['expanded_states']}",
    )

    # 时间预算：走的是假时钟，因为真时钟在这台机器上不触发（见 FakeClock）。
    scenario = _load("chain-cap160")
    clock = FakeClock()
    result = search(scenario, clock=clock)
    block = vp.explore(scenario, clock=FakeClock(), result=result)
    verdict = block["verdict"]
    checks.check(
        "时间预算耗尽 → unknown / time_budget_exhausted",
        verdict["status"] == "unknown"
        and verdict["termination_reason"] == "time_budget_exhausted"
        and block["reconcile"]["agrees"] is True,
        f"{verdict['status']} / {verdict['termination_reason']}",
    )

    # feasible 绝不能被读成最优。
    block = vp.explore(_load("chain-cap160"), max_expanded_states=28)
    checks.check(
        "feasible 不带最优性（k=28 有解、但没证明）",
        block["verdict"]["status"] == "feasible"
        and block["verdict"]["optimality_proven"] is False
        and block["solution"]["makespan_ns"] == 8_000_000
        and block["solution"]["action_count"] == 9,
        f"{block['solution']['action_count']} 个动作 / {block['solution']['makespan_ns']} ns",
    )


def check_bridge(checks: Checks) -> None:
    """B 组：成本桥。三类拒绝 + 换算等价 + E5 的字节推导。"""
    print("B · 成本桥")
    config = load_config(DEFAULT_CONFIG)
    chain = _load("three_layer_chain-cap1024")

    bridge = vp.bridge_table(chain, config=config)
    enabled = [row for row in bridge["toggles"] if row["enabled"]]
    checks.check(
        "门 1：随包配置开箱即被拒绝，六条子句齐全",
        bridge["kind"] == "unsupported" and len(enabled) == 6,
        f"{len(enabled)} 条：{'、'.join(r['label'] for r in enabled)}",
    )
    checks.check(
        "门 1：消息原样带出（三个字段路径都在）",
        all(
            needle in (bridge["message"] or "")
            for needle in (
                "hardware.cpu_effective_flops",
                "hardware.global_bytes",
                "policy.static_gpu_layers",
            )
        ),
        "",
    )

    current = config
    removed = 0
    for toggle in vp.BRIDGE_TOGGLES:
        before = len([r for r in vp.bridge_table(chain, config=current)["toggles"] if r["enabled"]])
        current = vp.apply_toggle(current, toggle["block"], toggle["field"], toggle["off"])
        after = len([r for r in vp.bridge_table(chain, config=current)["toggles"] if r["enabled"]])
        if before and after == before - 1:
            removed += 1
    checks.check(
        "门 1：每个开关恰好关掉一条子句",
        removed == 6,
        f"六条开着的里关掉 {removed} 条",
    )

    bridge = vp.bridge_table(chain, config=current)
    checks.check(
        "门 2：清单清空后撞上「分辨率」这道门（与迁移无关）",
        bridge["kind"] == "invalid" and "1.6e-11" in (bridge["message"] or ""),
        (bridge["message"] or "")[:80],
    )

    # 门 3：手填的非法参数由 frozen dataclass 自己的不变式拒绝。触发点是屏上那个
    # flops_per_op 旋钮——``LayerSpec(flops=0)`` 当场 ValueError。
    # （``PolicySpec.static_gpu_layers`` 只要求 >= 0，拿它当门 3 的探针是试不出来的。）
    zero_flops = vp.bridge_table(chain, config=current, flops_per_op=0.0)
    checks.check(
        "门 3：手填 flops=0 被 LayerSpec 自己的不变式拒绝（不是被吞掉）",
        zero_flops["kind"] == "value" and "flops" in (zero_flops["message"] or ""),
        (zero_flops["message"] or "")[:70],
    )

    for name in ("chain-cap160", "three_layer_chain-cap1024"):
        bridge = vp.bridge_table(_load(name), config=current, reference=True)
        checks.check(
            f"参考硬件复现 {name} 声明的全部 ns",
            bridge["kind"] == "ok" and bridge["all_agree"] is True,
            f"{len(bridge['table'])} 个 compute + {len(bridge['h2d'])} 个 h2d",
        )

    # E5：weight_bytes 必须从 scenario 推。抄 demo 的 64 会让这一行静默算错。
    matvec = vp.bridge_table(_load("matvec-cap160"), config=current, reference=True)
    weight_bytes = [layer["weight_bytes"] for layer in matvec["layers"]]
    checks.check(
        "E5：matvec 的 weight_bytes 从 scenario 推得 48（不是抄来的 64）",
        weight_bytes == [48],
        f"推出 {weight_bytes}",
    )
    checks.check(
        "E5：48 字节的 h2d 是 2.5 ms，与 fixture 写死的 3 ms 不同——这一行必须照实报",
        matvec["kind"] == "ok"
        and matvec["all_agree"] is False
        and [row for row in matvec["h2d"] if not row["agrees"]][0]["model_ns"] == 2_500_000,
        "model=2500000 declared=3000000",
    )

    # 第三道门是结构性的：0 份权重的算子（分叉 / 残差汇合）没有对应的层。
    for name in ("fork-cap160", "residual-cap160"):
        bridge = vp.bridge_table(_load(name), config=current, reference=True)
        checks.check(
            f"{name}：0 份权重的算子按图形状拒绝（不是按迁移拒绝）",
            bridge["kind"] == "unsupported"
            and bridge["stage"] == "graph ↔ layers 的配对"
            and bool(bridge["structural_problems"]),
            f"{len(bridge['structural_problems'])} 个算子没有唯一权重",
        )

    # ``config=None``：``--config`` 指的文件读不出来时 main() 选择「继续跑」。没这一支的话
    # ``costs_from_model_config(None, …)`` 会抛 AttributeError，不在三个 except 里，于是连
    # 时间线那一屏一起死在一条 traceback 上。``reference=True`` 也要一并查——它的
    # ``replace(config, …)`` 是**另一条**会碰 config 的路。
    for reference in (False, True):
        bridge = vp.bridge_table(_load("chain-cap160"), config=None,
                                 config_error="读不出来", reference=reference)
        checks.check(
            f"config=None（reference={reference}）回一条拒绝，而不是抛 AttributeError",
            bridge["kind"] == "config" and bridge["ok"] is False
            and bridge["toggles"] == [] and "读不出来" in bridge["message"],
            f"kind={bridge['kind']}，开关 {len(bridge['toggles'])} 行",
        )


def check_source(checks: Checks) -> None:
    """时长来源：modeling 推出来的 ``Costs`` **真的**换掉了内核吃的那一份。

    这一组是「联动」唯一的证伪网。只断言「面板上写着 modeling 推导」是不够的——那种断言
    在一份把 ``sourced_scenario`` 写成恒等函数的实现上**照样全绿**，屏上会一边写着「时长
    来自 modeling」一边画着照抄声明值的时间线，正是这次改造要消灭的那种假联动。所以每一条
    都落在**内核算出来的数**上：makespan、以及 disagreements 里那一对 ns。
    """
    print("· 时长来源（modeling → 内核）")

    def built(name: str, **params: Any) -> dict[str, Any]:
        viewer = Viewer(_load(name), scenario_path=None, config=None)
        viewer.last_params = params
        return viewer.build()

    def compute_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [row for row in payload["cost_source"]["rows"] if row["kind"] == "compute"]

    chain = built("chain-cap160")
    source = chain["cost_source"]
    checks.check(
        "没给 --config 时默认来源就是 modeling 推导（不是「配置没载入」）",
        source["kind"] == "model" and source["swapped"] is True,
        f"kind={source['kind']}，swapped={source['swapped']}",
    )
    checks.check(
        "chain：推导值与声明值逐项相等，makespan 仍是冻结记录里的 8000000",
        not source["disagreements"] and chain["solution"]["makespan_ns"] == 8_000_000,
        f"makespan={chain['solution']['makespan_ns']}，不一致 {len(source['disagreements'])} 项",
    )

    three = built("three_layer_chain-cap1024")
    checks.check(
        "three_layer_chain：同上，makespan 仍是 11000000",
        not three["cost_source"]["disagreements"]
        and three["solution"]["makespan_ns"] == 11_000_000,
        f"makespan={three['solution']['makespan_ns']}",
    )

    # matvec 是唯一一个「换了来源数字就动」的示例：它的 W 是 48 B，而声明值是按 64 B 写的。
    # 这一条同时钉住两件事——来源确实换了，以及**示例自己不自洽**（M0_VERIFICATION.md:138
    # 记的是 5 ms，而它记的那份是 64 B 的算式）。
    matvec = built("matvec-cap160")
    checks.check(
        "matvec：推导 2.5 ms ≠ 声明 3 ms，makespan 因此从 5 ms 变成 4.5 ms",
        matvec["cost_source"]["swapped"] is True
        and [
            (d["kind"], d["id"], d["model_ns"], d["declared_ns"])
            for d in matvec["cost_source"]["disagreements"]
        ] == [("h2d", "W", 2_500_000, 3_000_000)]
        and matvec["solution"]["makespan_ns"] == 4_500_000,
        f"makespan={matvec['solution']['makespan_ns']}（存档记的是 5000000）",
    )

    for name in ("fork-cap160", "residual-cap160"):
        payload = built(name)
        checks.check(
            f"{name}：图形状推不出层 → 退回声明值并带上原因（不是静默回退）",
            payload["cost_source"]["kind"] == "declared"
            and payload["cost_source"]["swapped"] is False
            and bool(payload["cost_source"]["message"]),
            f"reason_class={payload['cost_source']['reason_class']}，"
            f"makespan={payload['solution']['makespan_ns']}",
        )

    # 手改过算子时长时必须退回声明值：那些改动手写在 ``scenario.costs`` 上，而推导出来的
    # ``Costs`` 会把它们整份换掉——不退回的话「改了参数」与「改了没反应」在屏上长得一样。
    edited = built("chain-cap160", compute_ns={"c1": 5_000_000})
    checks.check(
        "手改过 compute_ns → 退回声明值并说明理由",
        edited["cost_source"]["kind"] == "declared"
        and edited["cost_source"]["reason_class"] == "edited"
        and edited["params_echo"]["compute_ns"]["c1"] == 5_000_000,
        f"c1={edited['params_echo']['compute_ns']['c1']}，"
        f"makespan={edited['solution']['makespan_ns']}",
    )

    # **这一条是整组的证伪网。** 参考硬件那组恰好复现示例写死的值，所以光看 chain 分不出
    # 「接上了 modeling」与「照抄声明值」。把 FLOP/s 减半，算出来的时长必须翻倍、makespan
    # 必须跟着变——这就是屏上那三个旋钮存在的全部理由。
    fast = built("chain-cap160")
    slow = built("chain-cap160", hardware={"gpu_effective_flops": 8_000.0})
    checks.check(
        "改硬件参数（FLOP/s 减半）→ 推导的 ns 翻倍，makespan 跟着变",
        bool(compute_rows(slow))
        and compute_rows(slow)[0]["model_ns"] == 2 * compute_rows(fast)[0]["model_ns"]
        and slow["solution"]["makespan_ns"] != fast["solution"]["makespan_ns"],
        f"compute {compute_rows(fast)[0]['model_ns']} → {compute_rows(slow)[0]['model_ns']}，"
        f"makespan {fast['solution']['makespan_ns']} → {slow['solution']['makespan_ns']}",
    )


def check_payload(checks: Checks) -> None:
    """C 组：载荷自洽。最有价值的是最后一条——独立重算那本唯一的内存账。"""
    print("C · 载荷自洽")
    scenario = _load("three_layer_chain-cap1024")
    payload = vp.payload(scenario, scenario_path="examples/three_layer_chain-cap1024.json")

    states = payload["log"]["states"]
    events = payload["log"]["events"]
    checks.check(
        "len(states) == len(events) + 1",
        len(states) == len(events) + 1,
        f"{len(states)} / {len(events)}",
    )
    times = [state["t_ns"] for state in states]
    checks.check("states[].t_ns 单调不减", all(b >= a for a, b in zip(times, times[1:])), "")
    limits = payload["source"]["limits"]
    checks.check(
        "peak_vram_bytes <= vram_capacity_bytes",
        payload["solution"]["peak_vram_bytes"] <= limits["vram_capacity_bytes"],
        f"{payload['solution']['peak_vram_bytes']} <= {limits['vram_capacity_bytes']}",
    )

    nodes = payload["explore"]["nodes"]
    start = payload["explore"]["start_id"]
    checks.check("每个节点都能沿父链回溯到起点", all(_reaches_start(nodes, n, start) for n in nodes), "")

    goal_id = payload["explore"]["goal_id"]
    checks.check(
        "nodes[goal_id].g_ns == solution.makespan_ns",
        goal_id is not None
        and nodes[goal_id]["g_ns"] == payload["solution"]["makespan_ns"],
        f"{nodes[goal_id]['g_ns'] if goal_id is not None else None}",
    )
    checks.check(
        "解路径上的节点都不含被剪的父边",
        all(
            nodes[node["parent"]]["children"] is not None
            for node in nodes
            if node["on_solution_path"] and node["parent"] is not None
        ),
        "",
    )
    checks.check(
        "t_ns == g_ns（t 就是路径代价——State.key() 敢排除 t 就是因为这条）",
        all(node["t_ns"] == node["g_ns"] for node in nodes),
        f"{len(nodes)} 个节点",
    )
    # 每个被展开的节点，它的每一条合法动作要么有出边、要么是一条被剪边，
    # 而且被剪边的目标**也在 nodes 里**——这是离线单步能完备的全部依据。
    dangling = [
        child["to"]
        for node in nodes
        for child in node["children"]
        if child["to"] is None or child["to"] >= len(nodes)
    ]
    checks.check("每条边的目标都在 nodes 里（离线单步因此完备）", not dangling, f"{len(dangling)} 条悬空")

    # 本组最有价值的一条：独立重算 engine.py 那个唯一的内存账本。
    architecture = scenario.architecture
    mismatched = []
    for node in nodes:
        expected = architecture.runtime_reserved_bytes
        for tensor_id, status in node["copy_status"].items():
            if status != "ABSENT":
                expected += scenario.size_alloc(tensor_id)
        for task in node["running"]:
            if task["resource"] == "gpu_compute":
                expected += scenario.workspace_alloc(task["target_id"])
        if expected != node["used_vram_bytes"]:
            mismatched.append((node["id"], expected, node["used_vram_bytes"]))
    checks.check(
        "used_vram_bytes 独立重算一致（runtime_reserved + Σ alloc + Σ workspace）",
        not mismatched,
        f"{len(mismatched)} 个节点对不上",
    )

    # 顶栏选择器的名单。``vp.payload()`` **不**产出这一块（它是 ``Viewer`` 贴上去的，
    # 因为要扫目录），所以这里直接查那个函数和 ``Viewer``——屏上那个下拉框里能选什么，
    # 就由这两处决定。
    items = scenario_catalog(str(EXAMPLES / "chain-cap160.json"))
    names = [item["name"] for item in items]
    checks.check(
        "名单 = examples/ 下的那 5 个 scenario，且不含派生产物",
        names == ["chain-cap160", "fork-cap160", "matvec-cap160", "residual-cap160",
                  "three_layer_chain-cap1024"],
        "、".join(names),
    )
    # 逐份试载。**不要写成 ``all(load_and_validate(...))``**：目录里躺着一份不是 scenario 的
    # json（手记、别人导出的产物）时那个生成器会当场抛，把整组自检带走——一次可读的 ✗ 变
    # 成一个 traceback。名单是「目录里有什么」，本来就该容得下这种东西。
    unloadable = []
    for item in items:
        try:
            load_and_validate(Path(item["path"]))
        except (SpecError, OSError, ValueError) as error:
            unloadable.append(f"{item['name']}：{error}")
    checks.check(
        "名单里的每一项都真的能载入（点了不会白点）",
        not unloadable,
        "；".join(unloadable) or f"{len(items)} 份都载得进来",
    )
    checks.check(
        "派生产物被排除在外（.workload.json / .mapping.json 不是另一张图）",
        not any(name in ("chain.workload", "chain.mapping") for name in names)
        and (EXAMPLES / "chain.workload.json").exists()
        and (EXAMPLES / "chain.mapping.json").exists(),
        "两个文件都在 examples/ 里，但都没进名单",
    )
    checks.check(
        "没有路径（内存里构造的那份）时名单是空的，不是「全仓扫描」",
        scenario_catalog(None) == [] and scenario_catalog("不存在/x.json") == [],
        "",
    )
    viewer = Viewer(_load("chain-cap160"), scenario_path=str(EXAMPLES / "fork-cap160.json"),
                    config=None)
    block = viewer.catalog_block()
    checks.check(
        "当前那一份一定在名单里（<select> 的 value 不在选项里时它会去显示第一个）",
        block["current"] == "fork-cap160"
        and any(item["name"] == "fork-cap160" for item in block["items"]),
        f"current={block['current']}",
    )
    # 名单里没有当前这一份时**必须补进去**：文件被删、后缀不是 .json 都会撞到这里，而
    # 补不补的区别就是「顶栏写着另一张图的名字」和「顶栏写着真话」。
    orphan = Viewer(_load("chain-cap160"), scenario_path=str(EXAMPLES / "ggml" / "nope.json"),
                    config=None)
    checks.check(
        "当前那份不在名单里时，补在最前面（顶栏不许显示成别的图）",
        orphan.catalog_block()["items"][0]["name"] == "nope"
        and orphan.catalog_block()["current"] == "nope",
        "、".join(item["name"] for item in orphan.catalog_block()["items"]),
    )

    # 目录里躺着**不是 scenario 的 json** 时的行为。真实场景：手记、别处导出的产物、
    # 手改坏了一半的文件。三条都必须在屏上是一次**可读的拒绝**（点下去弹原因），
    # 而不是「名单里看不见」或者「点了白屏/500」。
    with tempfile.TemporaryDirectory(prefix="viewer-catalog-") as tmp:
        folder = Path(tmp)
        (folder / "chain-cap160.json").write_text(
            (EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n")
        (folder / "bogus.json").write_text('{"kind": "不是 scenario"}', encoding="utf-8",
                                          newline="\n")
        (folder / "broken.json").write_text("{ 这压根不是 JSON", encoding="utf-8", newline="\n")
        viewer = Viewer(_load("chain-cap160"), scenario_path=str(folder / "chain-cap160.json"),
                        config=None)
        checks.check(
            "名单照列目录里非产物的 json（连载不进来的也留着，不静默抹掉）",
            [item["name"] for item in viewer.catalog] == ["bogus", "broken", "chain-cap160"],
            "、".join(item["name"] for item in viewer.catalog),
        )
        # 服务端把这几类都转成 422：``Handler.do_POST`` 收的就是下面这四种（外加 ``SpecError``
        # 之外的 JSON 解析错，那是 ValueError）。收到别的东西就等于一个不透明的 500。
        handled = (SpecError, TypeError, ValueError, KeyError)
        surprises = []
        for name in ("bogus", "broken", "并没有这张图"):
            try:
                viewer.load_scenario(name)
                surprises.append(f"{name} 竟然被接受了")
            except handled as error:
                if not str(error):
                    surprises.append(f"{name} 的拒绝没有原因（横幅上会是一句空话）")
            except Exception as error:  # noqa: BLE001 —— 这里就是要抓**没预料到**的那类
                surprises.append(f"{name} 抛的是 {type(error).__name__}（前端只能看到 500）")
        checks.check("坏文件 / 错名字都是可读的拒绝（422 那一类，不是 500）",
                     not surprises, "；".join(surprises))
        checks.check("连着被拒三次，这一侧还停在原来那份上（没有换到一半）",
                     viewer.scenario_name == "chain-cap160"
                     and Path(viewer.scenario_path).name == "chain-cap160.json",
                     f"{viewer.scenario_name}")


def _reaches_start(nodes: list[dict[str, Any]], node: dict[str, Any], start: int) -> bool:
    seen = set()
    current = node
    while current is not None:
        if current["id"] == start:
            return True
        if current["id"] in seen:
            return False  # 父链成环：走查写错了
        seen.add(current["id"])
        parent = current["parent"]
        current = None if parent is None else nodes[parent]
    return False


def check_snapshot(checks: Checks) -> None:
    """D 组：快照完整性。转义不做通用转义器，改做断言。"""
    print("D · 快照完整性")
    for name, needles in SNAPSHOT_FORBIDDEN.items():
        text = (STATIC / name).read_text(encoding="utf-8")
        found = [needle for needle in needles if needle in text]
        checks.check(f"{name} 不含 {needles}", not found, f"出现 {found}")

    # 替换靠字面标签，所以先确认那两个标签确实在、且只出现一次。少了 → 快照掉样式或掉
    # 脚本；多了 → 替换到了不该替换的地方。这条断言是 render_snapshot 的前提。
    template = (STATIC / "index.html").read_text(encoding="utf-8")
    checks.check(
        "index.html 里外链标签各恰好一次",
        template.count(LINK_TAG) == 1 and template.count(SCRIPT_TAG) == 1,
        f"link={template.count(LINK_TAG)} script={template.count(SCRIPT_TAG)}",
    )
    # 服务模式把 index.html 原样发出去：那两个标签必须能被浏览器直接解析（即没被写成
    # 占位符注释），否则浏览器里会是一个没有样式、没有脚本的空壳。
    checks.check(
        "index.html 原样发给浏览器时可用（无占位符）",
        "/*__" not in template and "__PAYLOAD__" not in template,
        "",
    )

    # 每个标签都得有一行副标题说明它属于哪一侧（mapping / modeling / 合作）。一排光秃秃的
    # 名词看不出这一点——用户就是照着那个栏问「显得意义不明」的。将来加第六屏时，这条会
    # 拦住「只写个名字就上线」，而那正是原来那个样子。
    tabs = re.findall(r'data-tab="([^"]+)"(.*?)</button>', template, re.S)
    scopeless = [name for name, body in tabs if 'class="tab-scope"' not in body]
    scopes = re.findall(r'class="tab-scope">([^<]*)<', template)
    checks.check(
        "每个标签都带副标题（说明属于哪一侧）",
        len(tabs) == 5 and not scopeless and len(scopes) == 5 and all(s.strip() for s in scopes),
        f"{len(tabs)} 个标签，缺副标题的 {scopeless}，副标题 {scopes}",
    )

    # 屏序：**搜索树 → 计算图 → 时间线 → 时长来源 → 参数**。这不是排版偏好，是「先看
    # 什么」：先看策略长什么样（树），再看这条策略把算子与张量排成什么形状（图），再看它
    # 花掉多少时间（线），最后才问「这些时长是哪来的」（来源）。倒过来的话，第一次打开
    # 看到的是「2 000 000 ns」而没有任何东西解释它。
    tab_names = [name for name, _ in tabs]
    pane_names = re.findall(r'data-pane="([^"]+)"', template)
    checks.check(
        "五块屏的顺序是 搜索树 → 计算图 → 时间线 → 时长来源 → 参数",
        pane_names == ["tree", "graph", "timeline", "source", "params"],
        "、".join(pane_names),
    )
    # 跳转条上的每一项都得落在一块真实存在的屏上。落空的那一项点下去**什么都不发生**，
    # 而且它看起来完全正常——所以这条要卡住。多一块没被指向的屏同理（用户永远滚不到它）。
    checks.check(
        "跳转条与屏一一对应（点每一项都落得下去）",
        sorted(set(tab_names)) == sorted(set(pane_names)) and len(set(tab_names)) == len(tab_names),
        f"tab={tab_names} pane={pane_names}",
    )

    scenario = _load("chain-cap160")
    html = render_snapshot(vp.payload(scenario, mode="snapshot"))
    checks.check("快照 < 2 MB", len(html.encode("utf-8")) < 2 * 1024 * 1024, f"{len(html) / 1024:.0f} KB")
    baked = _extract_baked(html)
    checks.check(
        "快照里的载荷能解析回来且 viewer_schema 一致",
        baked is not None and baked.get("viewer_schema") == vp.VIEWER_SCHEMA,
        "",
    )
    checks.check(
        "快照无外链、无 fetch 回退（双击即离线可用）",
        LINK_TAG not in html and SCRIPT_TAG not in html and "<style>" in html,
        "app.js 末尾那一处分叉决定走 __BAKED__ 还是 fetch",
    )
    # 内联后的形状：样式块恰好一段、脚本恰好两段（载荷 + app.js）。数目对不上说明拼接多
    # 插或漏插了一段——「掉样式」和「插了两遍」都只会在浏览器里显形，所以在这里锁住。
    checks.check(
        "快照里样式块恰好 1 段、脚本恰好 2 段",
        html.count("<style>") == 1 and html.count("<script>") == 2,
        f"style={html.count('<style>')} script={html.count('<script>')}",
    )
    # 载荷被内联进 <script>，所以它自己不能带能提前关标签的序列。``json.dumps`` 出来的
    # 那段里所有 ``</`` 都该已经变成 ``<\/``，于是段内不该再出现任何字面的 ``</``。
    segment = html.split("window.__BAKED__=")[-1].split(";</script>")[0]
    checks.check(
        "内联载荷段里没有字面的 </",
        "</" not in segment,
        f"段长 {len(segment)} 字节",
    )


def _extract_baked(html: str) -> dict[str, Any] | None:
    """从产出的 HTML 里把内联载荷解析回来。

    段尾那个 ``;`` 必须剥掉：``render_snapshot`` 写的是 ``window.__BAKED__={…};``，
    而这里读到 ``</script>`` 为止，所以末尾会带一个语句分号——不剥就是
    ``JSONDecodeError: Extra data``，于是这条检查会**永远**红。这是修过的一个真 bug，
    别再把它当格式细节。
    """
    marker = "window.__BAKED__="
    start = html.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = html.find("</script>", start)
    if end < 0:
        return None
    text = html[start:end].strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    try:
        return json.loads(text.replace("<\\/", "</"))
    except json.JSONDecodeError:
        return None


def check_house_rules(checks: Checks) -> None:
    """E 组：房子规矩——不可导入、不写仓库、零依赖。"""
    print("E · 房子规矩")
    checks.check("viewer_static/ 里没有 __init__.py", not (STATIC / "__init__.py").exists(), "")
    checks.check("tools/ 里没有 __init__.py", not (HERE / "__init__.py").exists(), "")
    outputs = REPO_ROOT / "outputs"
    before = sorted(p.name for p in outputs.iterdir()) if outputs.exists() else None
    vp.payload(_load("matvec-cap160"))
    after = sorted(p.name for p in outputs.iterdir()) if outputs.exists() else None
    checks.check("工具不往仓库里写任何东西", before == after, "")
    third_party = _third_party_imports()
    checks.check("两个模块只 import 标准库与本地包", not third_party, f"多出 {third_party}")


def _third_party_imports() -> list[str]:
    """粗查一遍两个模块的顶层 import 有没有第三方。纯文本扫，不 import 它们。"""
    import ast

    allowed = {
        "__future__", "argparse", "ast", "collections", "dataclasses", "hashlib", "heapq",
        "http", "json", "os", "pathlib", "re", "shutil", "subprocess", "sys", "tempfile",
        "threading", "time", "typing",
        "unicodedata", "urllib", "webbrowser", "llm_infer_model", "tensor_mapping",
        "viewer_payload",
    }
    found: list[str] = []
    for name in ("viewer.py", "viewer_payload.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root and root not in allowed:
                    found.append(f"{name}: {module}")
    return found


def check_launcher(checks: Checks) -> None:
    """双击启动器（计划外新增的一组）。

    这条路坏起来是**无声**的：自检全绿、命令行照常能用，只是双击之后窗口闪一下就没了，
    用户看到的是「这东西打不开」。三条约束都是查过资料确认过的 cmd.exe 行为，不是风格
    偏好——正文必须是 ASCII（``.cmd`` 按 OEM 码页读，UTF-8 中文注释会碎成标点混进命令
    行）、行尾必须是 CRLF、以及不能出现绝对路径（写进去了，任何别的检出都静默失效）。
    """
    print("· 双击启动器 viewer.cmd")
    if not LAUNCHER.exists():
        checks.check("viewer.cmd 在（双击即用）", False, "没找到")
        return
    raw = LAUNCHER.read_bytes()
    tidy = raw.replace(b"\r\n", b"")
    checks.check(
        "正文只有 ASCII、行尾只有 CRLF",
        raw.isascii() and b"\n" not in tidy and b"\r" not in tidy,
        f"非 ASCII {sum(1 for b in raw if b > 0x7F)} 字节；"
        f"裸 LF {raw.count(b'\\n') - raw.count(b'\\r\\n')} 处、裸 CR {raw.count(b'\\r') - raw.count(b'\\r\\n')} 处",
    )
    text = raw.decode("ascii", errors="replace")
    absolute = ":\\" in text
    checks.check(
        "不含绝对路径，且指向 venv 与 tools\\viewer.py",
        not absolute and ".venv" in text and "tools\\viewer.py" in text,
        f"绝对路径={'有' if absolute else '无'}",
    )


# 非法参数的四条真实路径，覆盖三道门。每一条都必须**只**抛出它自己的理由。
BAD_PARAMS: list[tuple[str, dict[str, Any]]] = [
    ("门 3 · 容量下界", {"vram_capacity_bytes": 0}),
    ("门 3 · 预留超容量", {"runtime_reserved_bytes": 10**9}),
    ("非数字 · 旋钮", {"vram_capacity_bytes": "abc"}),
    ("非数字 · 算子时长", {"compute_ns": {"c1": "x"}}),
]
CAUGHT_BY_HANDLER = (SpecError, TypeError, ValueError, KeyError)


def check_params(checks: Checks) -> None:
    """参数提交的两条不变式（计划外新增的一组）。

    两条都对应**实测踩到过**的 bug，不是设想的边角：一次手滑之后查看器要么静默丢掉用户
    的旋钮修改，要么被永久毒住只能重启。这两条最容易在后续改动里悄悄退化——都是「正常
    路径照常工作、只有某个组合才坏」的那种——所以锁进自检。
    """
    print("· 参数提交（合并 / 回滚）")
    viewer = Viewer(_load("chain-cap160"), scenario_path=None, config=load_config(DEFAULT_CONFIG))

    # 1) 合并而不是替换。桥的开关提交里只有 bridge_toggle 一个键；替换会让用户在参数页
    #    改好的 VRAM 被一次「关掉某个未迁移项」的点击悄悄冲掉。
    viewer.commit({"vram_capacity_bytes": 96})
    toggles = viewer.build()["bridge"]["toggles"]
    closed = [row for row in toggles if row["enabled"]]
    if closed:
        row = closed[0]
        viewer.commit(
            {"bridge_toggle": {"block": row["block"], "field": row["field"],
                               "value": row["off_value"]}}
        )
    after = viewer.build()
    checks.check(
        "参数提交是**合并**不是替换（点桥开关不冲掉场景旋钮）",
        viewer.last_params.get("vram_capacity_bytes") == 96
        and after["params_echo"]["vram_capacity_bytes"] == 96,
        f"last_params={viewer.last_params}，载荷里 {after['params_echo']['vram_capacity_bytes']}",
    )
    checks.check(
        "同一次提交里桥开关也真的落了地（启用子句恰好少一条）",
        bool(closed)
        and len([r for r in after["bridge"]["toggles"] if r["enabled"]]) == len(closed) - 1,
        f"{len(closed)} → {len([r for r in after['bridge']['toggles'] if r['enabled']])} 条",
    )

    # 2) 回滚。``apply`` 先改状态、``build`` 后验证，非法值必须在抛出的同时被撤回；否则
    #    last_params 里留着那个非法值，此后每个请求都在 before=build() 上抛同一个错，
    #    整页死在一个与当前操作无关的错误上，只能重启进程。
    good = viewer.build()
    raised: list[tuple[str, str]] = []
    for label, bad in BAD_PARAMS:
        try:
            viewer.commit(bad)
        except CAUGHT_BY_HANDLER as error:
            raised.append((label, str(error)))
        else:
            raised.append((label, "**没有拒绝**"))
    recovered = viewer.build()
    checks.check(
        "四种非法输入都被拒绝，且各报各的原因（毒住时会全报同一个）",
        len(set(message for _, message in raised)) == len(BAD_PARAMS),
        "；".join(f"{label} → {message[:60]}" for label, message in raised),
    )
    checks.check(
        "被拒绝后**回滚**：查看器还活着，配置也没被改动",
        recovered["params_echo"] == good["params_echo"]
        and recovered["solution"]["makespan_ns"] == good["solution"]["makespan_ns"],
        f"恢复后 VRAM={recovered['params_echo']['vram_capacity_bytes']}、"
        f"makespan={recovered['solution']['makespan_ns']}",
    )
    viewer.commit({"vram_capacity_bytes": 160})
    checks.check(
        "拒绝之后再提交合法参数仍然生效（毒住时这一步会抛旧错）",
        viewer.build()["params_echo"]["vram_capacity_bytes"] == 160,
        "改回 160 后载荷里就是 160",
    )


def check_http(checks: Checks) -> None:
    """真起一个服务，把路由与两种 POST 打一遍（计划外新增的一组）。

    前面几组验的都是「装配出来的东西对不对」，路由那一层一直没人管——它是 ``Handler`` 里
    二十行 dispatch，但那二十行决定了用户点下去是看到横幅还是白屏。所以这里**真的**绑一个
    套接字（``port=0``，让内核挑一个空闲端口，不会和别的东西撞）、真的发 HTTP。

    绑定失败就**跳过**并说明，不计入 ✓ 也不计入 ✗：某些沙箱环境不给套接字，那是环境问题
    不是逻辑问题（和文件头那段 ``ModuleNotFoundError`` 的处理同一个态度）。但跳过要**看得
    见**——把跳过写成绿就是撒谎。
    """
    print("· 真实 HTTP 路由（loopback，port=0）")
    import urllib.error
    import urllib.request

    viewer = Viewer(_load("chain-cap160"), scenario_path=str(EXAMPLES / "chain-cap160.json"),
                    config=load_config(DEFAULT_CONFIG))
    handler = type("BoundHandler", (Handler,), {"viewer": viewer})
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError as error:
        print(f"  · 跳过：本机不允许绑套接字（{error}）——这是环境问题，不是逻辑问题")
        return
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def get(path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(base + path, timeout=20) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def post(body: Any) -> tuple[int, Any]:
        request = urllib.request.Request(
            base + "/api/params",
            data=(body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"null")

    try:
        statuses = {path: get(path)[0] for path in ("/", "/style.css", "/app.js", "/api/payload")}
        checks.check(
            "四个 GET 路由都回 200（index / 样式 / 脚本 / 载荷）",
            all(code == 200 for code in statuses.values()),
            " ".join(f"{path}={code}" for path, code in statuses.items()),
        )
        # ``<title>`` 在 index.html 里；CSS/JS 各挑一个只在正确文件里出现的串，防止路由
        # 接错了文件却都回 200 这种「看起来全绿」。
        checks.check(
            "每条件回的是**它自己**那份内容（不是都回了 index.html）",
            b"<title>" in get("/")[1]
            and b":root" in get("/style.css")[1]
            and b"__BAKED__" in get("/app.js")[1],
            "",
        )
        code, body = get("/api/payload")
        checks.check(
            "载荷路由能被 JSON 解析且 viewer_schema 一致",
            code == 200 and json.loads(body)["viewer_schema"] == vp.VIEWER_SCHEMA,
            "",
        )
        checks.check("未知道路回 404，不是 500", get("/nope")[0] == 404, "")

        code, pair = post({"vram_capacity_bytes": 96})
        checks.check(
            "POST 合法参数 → 200，且**同时**给出 before 与 after",
            code == 200 and isinstance(pair, dict)
            and pair["after"]["params_echo"]["vram_capacity_bytes"] == 96
            and pair["before"]["params_echo"]["vram_capacity_bytes"] != 96,
            f"before={pair['before']['params_echo']['vram_capacity_bytes']} "
            f"after={pair['after']['params_echo']['vram_capacity_bytes']}" if code == 200 else str(pair),
        )
        code, error = post({"vram_capacity_bytes": 0})
        checks.check(
            "POST 非法参数 → 422 + 异常类名（不是不透明的 500）",
            code == 422 and error.get("error_class") == "InvalidInput"
            and ">= 1" in error.get("error", ""),
            f"{code} {error}",
        )
        checks.check("非 JSON 的请求体 → 400", post(b"{ not json")[0] == 400, "")

        # 前端点开关时发的是**载荷里那一行的 off_value**。逐个点掉所有启用行都必须被接受——
        # 只要有一行的 off 值服务端不收，那一行就成了点了就白屏的按钮。
        code, payload = get("/api/payload")
        enabled = [row for row in json.loads(payload)["bridge"]["toggles"] if row["enabled"]]
        rejected: list[str] = []
        for row in enabled:
            status, body = post({"bridge_toggle": {
                "block": row["block"], "field": row["field"], "value": row["off_value"]}})
            if status != 200:
                rejected.append(f"{row['label']} → {status} {body}")
        checks.check(
            f"载荷里每一个启用行的 off_value 服务端都收（{len(enabled)} 行）",
            not rejected,
            "；".join(rejected),
        )
        code, payload = get("/api/payload")
        bridge = json.loads(payload)["bridge"]
        checks.check(
            "全部关掉后确实推进到第二道门（InvalidInput），而不是失败",
            code == 200 and bridge["kind"] == "invalid" and "1.6e-11" in bridge["message"],
            f"kind={bridge['kind']}",
        )
        checks.check(
            "被拒绝的那次请求没有毒住服务（后面每一次都还是 200）",
            get("/api/payload")[0] == 200,
            "",
        )

        # ---- 顶栏「换计算图」。这是这一版唯一一个**换掉整份载荷**的入口，所以它要过的是
        # 一道比参数提交更严的关：换过去、换回来、换一个不存在的名字，每一步都得是 200/422
        # 而不是白屏，而且**失败之后那一侧不能停在半路**（屏上是旧图、顶栏写着新名字）。
        code, payload = get("/api/payload")
        catalog = json.loads(payload)["scenarios"]
        checks.check(
            "载荷里有选择器要的名单，且当前那份在里面",
            code == 200 and catalog["current"] == "chain-cap160"
            and len(catalog["items"]) == 5
            and any(item["name"] == catalog["current"] for item in catalog["items"]),
            f"current={catalog.get('current')} n={len(catalog.get('items', []))}",
        )
        # 换图前先拧一个旋钮：下一步要看它有没有跟着走到新图上（这是「同一套旋钮换一张图」
        # 这句话的全部内容）。这一步本身不另设断言，上面已经验过参数提交了。
        post({"vram_capacity_bytes": 96})
        code, pair = post({"scenario": "fork-cap160"})
        checks.check(
            "POST scenario → 200，after 换成了那张图（before 还是旧的）",
            code == 200 and pair["after"]["scenario_id"] == "fork-cap160"
            and pair["before"]["scenario_id"] == "chain-cap160"
            and Path(pair["after"]["scenario_path"]).name == "fork-cap160.json",
            f"{code} before={pair.get('before', {}).get('scenario_id')} "
            f"after={pair.get('after', {}).get('scenario_id')}" if code == 200 else str(pair),
        )
        checks.check(
            "换图之后 before/after 是**两张真不一样的图**（不是换了个标签）",
            code == 200
            and pair["after"]["scenarios"]["current"] == "fork-cap160"
            and pair["after"]["graph"]["operations"] != pair["before"]["graph"]["operations"],
            "",
        )
        checks.check(
            "换图不改旋钮：刚提交的 VRAM 跟着走到新图上",
            code == 200 and pair["after"]["params_echo"]["vram_capacity_bytes"] == 96,
            f"after={pair.get('after', {}).get('params_echo', {}).get('vram_capacity_bytes')}"
            if code == 200 else str(pair),
        )
        code, pair = post({"scenario": "chain-cap160"})
        checks.check(
            "换回去也是 200，而且这一侧的**状态真的变了**（before 是刚换过去那张）",
            code == 200 and pair["before"]["scenario_id"] == "fork-cap160"
            and pair["after"]["scenario_id"] == "chain-cap160",
            f"{code}",
        )
        code, error = post({"scenario": "并没有这张图"})
        checks.check(
            "POST 不存在的名字 → 422，且**把这一批有哪些列出来**（不然只能猜）",
            code == 422 and "fork-cap160" in error.get("error", "") and "并没有这张图" in error.get("error", ""),
            f"{code} {error}",
        )
        code, payload = get("/api/payload")
        checks.check(
            "被拒的那次没有换到一半：图仍是 chain-cap160，服务照常（回滚生效）",
            code == 200 and json.loads(payload)["scenario_id"] == "chain-cap160",
            f"{code}",
        )
        # 名单里**没有**的东西一律拒（包括想借这个键读文件的情况）：这个键是从 HTTP 进来的，
        # 拼路径就等于把一个本地玩具变成任意读文件的接口。
        code, error = post({"scenario": "../../../etc/passwd"})
        checks.check(
            "名字走白名单：路径样子的名字也被同一道门挡下",
            code == 422 and error.get("error_class") in ("ValueError", "SpecError", "InvalidInput"),
            f"{code} {error.get('error_class')}",
        )
        checks.check(
            "被拒两次之后服务还是好的（每次都是 200）",
            get("/api/payload")[0] == 200,
            "",
        )
        # 回滚**含 scenario** 的那一条。上面那次拒绝是在改状态**之前**抛的（名字不在名单里），
        # 所以它证明不了回滚；这一条才把「已经换过去了、随后 build 抛了」摆出来：一次提交里
        # 同时换图和填非法容量，``apply`` 顺序是「先换图（改了 base_scenario）再落参数」，
        # 于是 ``build`` 是在**已经换成 fork 的**那一侧上抛的。三元组的旧回滚在这里会留下
        # 「屏上是 fork、报的是 chain 的参数错」——最刺眼的那种半截状态。
        code, error = post({"scenario": "fork-cap160", "vram_capacity_bytes": 0})
        code2, payload = get("/api/payload")
        after_body = json.loads(payload)
        checks.check(
            "换图 + 非法参数同一次提交 → 422，且**图也退回去了**（回滚含 scenario）",
            code == 422 and error.get("error_class") == "InvalidInput"
            and after_body["scenario_id"] == "chain-cap160"
            and after_body["params_echo"]["vram_capacity_bytes"] == 96,
            f"{code} 之后图={after_body['scenario_id']} "
            f"VRAM={after_body['params_echo']['vram_capacity_bytes']}",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def check_js(checks: Checks) -> str | None:
    """· 组：前端 JS 的**落地**烟测——在 node 里把 ``app.js`` 真跑一遍。

    前面每一组看得到的都是**静态**的东西：载荷的字段、``index.html`` 里有几个标签。
    可「计算图一条边都没有」那次，载荷是对的、``index.html`` 是对的、``render()`` 也没抛
    异常——错的是 JS 算出来的每个坐标都是 ``NaN``，而 **SVG 对无效属性是静默的**：
    ``<rect x="NaN">`` 的 ``x`` 被忽略（所有框挤到 x=0）、``<path d="…NaN…">`` 整条元素
    被丢弃（一条边都画不出来）。grep 级别的东西在这一类 bug 面前是全绿的。

    所以这一组把 ``app.js`` 放进一个 ~300 行的 DOM 垫片里跑，换到三件静态查不到的事：
    五个渲染器都真的跑了（空 ``<svg>`` 与空 ``#params-body`` 一个都过不了）、整棵文档树
    扫不到一个 ``NaN``、计数**卡相等**（边数 == ``graph.links.length``）。

    ``node`` 不在就算了——它是**开发工具**，不是运行依赖（``DESIGN.md`` 禁的是依赖）。
    少了这一组不代表工具坏了，代表这台机器上「前端真的跑过一遍」这句话没人验证过；
    返回的字符串会挂在总结那一行上，别让它静默地少一组。
    """
    print("· JS 落地烟测（node）")
    node = shutil.which("node")
    if node is None:
        print("  · 跳过：这台机器上没有 node。")
        print("    「五个渲染器都真的跑过一遍」这句话因此没有被验证——D 组只看得到静态的 HTML。")
        return "跳过 JS 烟测：没有 node"

    cases = []
    for name in ("chain-cap160", "matvec-cap160", "fork-cap160", "residual-cap160",
                 "three_layer_chain-cap1024"):
        # ``scenario_path`` 一律给**绝对**路径：名单是按它的目录扫出来的，写成
        # ``examples/x.json`` 就跟着启动时的 cwd 走了（从仓库根跑 self-check 时那个
        # ``examples/`` 不存在，名单空掉，顶栏选择器那几条断言会**静默地空转**成「0 个
        # 选项也匹配 0 个选项」）。绝对路径下这一批恒为那 5 个，与在哪儿敲命令无关。
        viewer = Viewer(_load(name), scenario_path=str(EXAMPLES / f"{name}.json"), config=None)
        cases.append({"name": name, "payload": viewer.build()})

    # 顶栏「换计算图」那一下的**真跑**：换个 scenario 提交，前端拿到新载荷之后三块屏都得跟着
    # 换。垫片里没有服务端，所以只给这一个用例配一个 fetch 桩，返回的是服务端**真装配**出来
    # 的那份载荷（就在下面现 build）——桩里现编一份载荷等于自己给自己出题。
    switch_viewer = Viewer(_load("residual-cap160"),
                           scenario_path=str(EXAMPLES / "residual-cap160.json"), config=None)
    cases[0]["switch_to"] = {"name": "residual-cap160", "payload": switch_viewer.build()}

    # 只读那一支：``mode != 'live'`` 时选择器与整张参数表都必须是禁用的。原先七份用例
    # 全是 live，于是「快照里所有按钮都点不动」这句话一次都没被验过（D 组只看得到静态
    # 的 HTML，而禁用是在 JS 里做的）。
    snapshot_viewer = Viewer(_load("chain-cap160"),
                             scenario_path=str(EXAMPLES / "chain-cap160.json"), config=None)
    cases.append({"name": "chain-cap160·导出快照", "payload": snapshot_viewer.build(mode="snapshot")})

    # 一份**没有邻居**的载荷（内存里构造的，``scenario_path=None``）。它覆盖的是「名单空着」
    # 那条分支：选择器整格收起来、选中值置空。这条分支在真实使用里几乎走不到（``main()``
    # 总带着路径），但**打开一份更早导出的快照**（载荷里没有 ``scenarios`` 这一块）走的正是
    # 它——那时候前端不能报错，只是少一个控件。
    cases.append({
        "name": "chain-cap160·没有邻居（载荷里没有名单）",
        "payload": Viewer(_load("chain-cap160"), scenario_path=None, config=None).build(),
    })

    # 两份**没有解**的载荷。上面五个用例全都有解，于是都走「先后取动作表步号」那条分支；
    # 而 ``byExecution = actions.length > 0`` 为假时会退回拓扑秩，**那条分支一个用例都没覆盖**，
    # 在浏览器里却只要把 VRAM 调小就进得去。容量 80 搜得到 5 个节点但一步都排不出来；
    # 容量 1 是零节点。两份的 ``log.events`` / ``log.states`` 都是 0。
    #
    # 这两份**不是**用来放宽「每块屏都要有画出来的东西」那条断言的——曾经按推理改成
    # 「从载荷推哪几屏该是空的」，量出来是**反的**：``#vram-svg`` 的刻度与容量上限线照画，
    # 0 个 state 时它仍有 14 个子节点。所以照旧一律要求非空，教训写在 ``viewer_smoke.mjs`` 里。
    # 它们真正的用处是**覆盖无解那条分支**（没有动作表，先后退回拓扑秩：横纵两轴的排布都要能在
    # 没有「第几步」的情况下算出来，说明那句「这一步没有解」也要出现），
    # 顺带把无解路径上的两个真 bug（``undefined.t_ns``、共用 ``try`` 的连坐）钉住。
    for cap in (80, 1):
        viewer = Viewer(_load("chain-cap160"),
                        scenario_path=str(EXAMPLES / "chain-cap160.json"), config=None)
        cases.append({
            "name": f"chain-cap160·容量{cap}（无解）",
            "payload": viewer.build({"vram_capacity_bytes": cap}),
        })

    # 用例只在**临时目录**里过一手：E 组断言工具不往仓库里写任何东西，这条不能破。
    with tempfile.TemporaryDirectory(prefix="viewer-smoke-") as tmp:
        case_path = Path(tmp) / "cases.json"
        case_path.write_text(
            json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        script = HERE / "viewer_smoke.mjs"
        common = [
            "--app", str(STATIC / "app.js"),
            "--html", str(STATIC / "index.html"),
            "--payload", str(case_path),
        ]

        def run(extra: list[str]) -> tuple[list[dict[str, Any]], str]:
            done = subprocess.run(
                [node, str(script), *common, *extra],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO_ROOT),
            )
            rows: list[dict[str, Any]] = []
            for line in done.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    return [], f"读不懂 node 的输出：{line[:200]}"
            tail = (done.stderr or "").strip().splitlines()
            note = "　".join(tail[-3:]) if tail else f"退出码 {done.returncode}"
            return rows, "" if rows else note

        rows, trouble = run([])
        if trouble:
            checks.check("烟测脚本跑得起来", False, trouble)
            return None
        skipped: list[str] = []
        for row in rows:
            if row.get("done"):
                continue
            # **跳过的不记成通过项。** 无解的那两份载荷没有终点圈，交互那一项在那里不适用；
            # 把它算成「通过」就是在数一个没跑过的东西。它进 `skipped`，回头挂在总结行上。
            if row.get("skip"):
                skipped.append(row["label"])
                continue
            checks.check(f"JS · {row['label']}", row["ok"], row.get("detail", ""))

        # **证伪**：往一个坐标里注入 NaN，这张通用网必须变红。
        # 不做这一步，「扫 NaN」就没被证明过——它可能从来没红过，那它到底在扫什么就没人知道。
        # 上一版「标题里有中文」的断言正是因为漏了这一步而空转过一次。
        falsified, trouble = run(["--falsify"])
        if trouble:
            checks.check("证伪：注入 NaN 后这张网变红", False, trouble)
        else:
            for row in falsified:
                if row.get("done"):
                    continue
                checks.check(row["label"], row["ok"], row.get("detail", ""))
    # 跳过的那几项挂在总结行上，不静默。返回 ``None`` 表示「这一组没有需要另行说明的」。
    if skipped:
        return f"前端交互一项跳过 {len(skipped)} 处（{skipped[0].split(' · ')[-1]}）"
    return None


def run_self_check() -> int:
    checks = Checks()
    print("查看器自检（无头、不写任何文件；只绑一次 loopback 套接字）")
    print()
    # A–E 是计划里那五组，字母连着读；后面两组是实施中新加的（参数提交的不变式、真实 HTTP
    # 路由），按「先装配后传输」排在最后，不去打乱那五个字母的顺序。
    check_walk(checks)
    print()
    check_bridge(checks)
    print()
    check_source(checks)
    print()
    check_payload(checks)
    print()
    check_snapshot(checks)
    print()
    # 放在 D/E 之后：前面把载荷与页面都查完了，这一组才把那份页面真的跑一遍。
    js_note = check_js(checks)
    print()
    check_house_rules(checks)
    check_launcher(checks)
    print()
    check_params(checks)
    print()
    check_http(checks)
    print()
    if checks.failures:
        print(f"✗ {len(checks.failures)}/{len(checks.rows)} 项不通过")
        return 1
    print(f"✓ 全部 {len(checks.rows)} 项通过" + (f"（{js_note}）" if js_note else ""))
    return 0


# ---------------------------------------------------------------------------
# 服务与快照
# ---------------------------------------------------------------------------


def render_snapshot(payload: dict[str, Any]) -> str:
    """把三个前端文件 + 载荷拼成一个自包含的 HTML。

    替换的是 ``index.html`` 里**字面**的两个标签，不是占位符注释：

    ``<link rel="stylesheet" href="style.css">`` → ``<style>…</style>``
    ``<script src="app.js"></script>``           → 载荷 + ``<script>…</script>``

    为什么不用 ``/*__STYLE__*/`` 这类占位符：服务模式是把 ``index.html`` **原样**
    发出去的（``Handler.do_GET`` 直接读文件），占位符只会在导出时被替换，于是浏览
    器模式下拿到的是一个满是占位符、没有 CSS 也没有 JS 的空壳。用真标签就两边都对：
    服务模式靠浏览器自己去取那两件，快照模式把它们内联进来。前端零改动、不分叉。

    载荷里 ``</`` 一律转成 ``<\\/``：JSON 里唯一能提前关掉 ``<script>`` 的序列就是
    ``</script``，而 ``<\\/`` 在 JSON 字符串里与 ``</`` 等价。真正的保险是自检 D 组
    对那三份静态文件的断言 + 「快照里不含外链」那条。
    """
    style = (STATIC / "style.css").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    body = (STATIC / "index.html").read_text(encoding="utf-8")
    baked = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return body.replace(
        LINK_TAG, "<style>\n" + style + "\n</style>"
    ).replace(
        SCRIPT_TAG, "<script>window.__BAKED__=" + baked + ";</script>\n<script>\n" + script + "\n</script>"
    )


class Viewer:
    """一份 scenario + 当前参数，服务一个浏览器页。"""

    def __init__(self, scenario: Any, *, scenario_path: str | None, config: Any,
                 config_error: str | None = None) -> None:
        self.base_scenario = scenario
        self.scenario_path = scenario_path
        self.config = config
        # ``config is None`` 时那句话要原样带上屏：读不出来的原因（路径不存在 / 不是合法
        # JSON / 字段不认得）只有 ``load_config`` 知道，屏上再编一句就等于把诊断信息丢了。
        self.config_error = config_error
        self.reference = False
        # 可切换的邻居**只在构造时扫一次**：名单由目录决定，一次会话里不会变，而每帧重扫
        # 目录就等于每帧多一次磁盘往返。换图之后 ``self.catalog`` 不动（还是同一批），
        # 变的只有 ``scenario_name``。
        self.catalog = scenario_catalog(scenario_path)
        self.scenario_name = Path(scenario_path).stem if scenario_path else None
        # 上一次提交留下的参数。``build()`` 不带参数时就用它——「before/after」这对载荷
        # 必须是**同一条代码路径**产出的，否则「改了个参数」和「换了份配置」就分不清。
        self.last_params: dict[str, Any] = {}
        self.lock = threading.Lock()

    def catalog_block(self) -> dict[str, Any]:
        """顶栏那个选择器要的全部东西：当前是哪一个 + 这一批有哪些。"""
        items = list(self.catalog)
        if self.scenario_name and not any(item["name"] == self.scenario_name for item in items):
            # 名单里没有当前这一份（文件被删了 / 后缀不是 ``.json``，于是没被扫到）。
            # 补进去而不是留着不管：``<select>`` 的 value 不在选项里时会**显示第一个选项**，
            # 于是顶栏会理直气壮地写着另一张图的名字——那是这一屏唯一不能容忍的一类错。
            items.insert(0, {"name": self.scenario_name, "path": self.scenario_path or ""})
        return {"current": self.scenario_name, "items": items}

    def load_scenario(self, name: str) -> None:
        """按名字换一份 scenario（名字来自 ``catalog_block()['items']``）。

        **名字走白名单，不拼路径**：``scenario`` 这个键是从 ``POST /api/params`` 进来的，
        谁都能发；拼路径就等于把这个本地服务变成一个任意读文件的接口。拒绝信息里带上这一批
        有哪些，手打错名字的人一眼能看出该写哪个。
        """
        entry = next((item for item in self.catalog if item["name"] == name), None)
        if entry is None:
            known = "、".join(item["name"] for item in self.catalog) or "（这一份没有可切换的邻居）"
            raise ValueError(f"没有名为 {name!r} 的 scenario；这一批是：{known}")
        try:
            scenario = load_and_validate(Path(entry["path"]))
        except OSError as error:
            # 读不出来（文件被删了 / 权限）在 ``except`` 那几类里没有对口的分支，会一路冒成
            # 500。这不是服务器故障，是「点了一个坏文件」——包成 ValueError 归到可读的 422。
            raise ValueError(f"{entry['path']} 读不出来：{error}") from error
        self.base_scenario = scenario
        self.scenario_path = entry["path"]
        self.scenario_name = name

    def build(self, params: dict[str, Any] | None = None, *, mode: str = "live") -> dict[str, Any]:
        params = self.last_params if params is None else params
        scenario = edit_scenario(self.base_scenario, params)
        states, seconds = budget_from(
            params,
            scenario.mapper.max_expanded_states,
            scenario.mapper.wall_time_limit_s,
        )
        bridge = vp.bridge_table(
            scenario,
            config=self.config,
            config_error=self.config_error,
            flops_per_op=float(params.get("flops_per_op", vp.FLOPS_PER_OP)),
            reference=bool(params.get("reference", self.reference)),
            hardware=params.get("hardware"),
        )
        # 来源必须在 ``payload()`` **之前**定下来：``sourced_scenario`` 换的是喂给内核的
        # 那份 ``Costs``，而 makespan / 峰值 / 事件 / 每一根柱子的宽度全从它推出来。放在
        # 后面就等于先按声明值算完一遍、再改一个标签——数字不会动，而屏上会写着「来自
        # modeling」。那正是这次改造要消灭的那种假联动。
        sourced, source_block = vp.sourced_scenario(
            scenario, bridge, edited=bool(params.get("compute_ns")),
        )
        payload = vp.payload(
            sourced,
            scenario_path=self.scenario_path,
            mode=mode,
            max_expanded_states=states,
            wall_time_limit_s=seconds,
        )
        payload["bridge"] = bridge
        payload["cost_source"] = source_block
        payload["scenarios"] = self.catalog_block()
        return payload

    def apply(self, params: dict[str, Any]) -> None:
        """把一次表单提交**落到**这一侧（改桥的开关、切参考硬件、换图、记住场景旋钮）。"""
        params = dict(params)  # 不要 pop 调用方的字典
        toggle = params.pop("bridge_toggle", None)
        if toggle is not None:
            self.config = vp.apply_toggle(
                self.config, toggle["block"], toggle.get("field"), toggle["value"]
            )
        if "reference" in params:
            self.reference = bool(params.pop("reference"))
        scenario = params.pop("scenario", None)
        if scenario is not None:
            # 换图**不动** last_params：VRAM、预算、桥的开关都跟着走。「同一套旋钮换一张图
            # 看看」正是这个按钮存在的理由，换一次图就把用户刚调好的容量清掉是反的。
            # ``compute_ns`` 也照样留着——它按算子 id 给值，而 ``edit_scenario`` 对这张图上
            # 不存在的 id 是**跳过**（不是报错），同名的算子（``c1`` 这类）则如用户所愿继续生效。
            self.load_scenario(str(scenario))
        # **合并**，不是替换。桥的开关提交里只有 ``bridge_toggle`` 一个键，替换会把用户
        # 刚在参数页改好的 VRAM 悄悄丢掉；而参数页提交的是全套旋钮，两者只有合并才对得住
        # 「上一次提交留下的参数」这个说法。``flops_per_op`` 也留在这里：build() 从 params
        # 里读它，``edit_scenario`` 读场景旋钮，各取所需，不认识的键会被忽略。
        self.last_params.update(params)

    def commit(self, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """提交一次参数，成功返回 ``(before, after)``；失败则**回滚**并把异常抛出去。

        回滚是必需的，不是讲究：``apply`` 会改 ``last_params`` / ``config`` /
        ``reference``，而 ``build`` 可能因为新参数非法而抛。不还原的话，一次手滑就把查看器
        **永久**毒住了——``last_params`` 里留着那个非法值，此后每个请求都在 ``before =
        self.build()`` 上抛同一个错，整页死在一个和当前操作无关的错误上，只能重启进程。
        （实测踩到过：连着四次不同的非法输入，报的都是第一次那个 ``got 0``。）
        换图那一支同理，而且更刺眼：换到一半失败而没退回，屏上留着旧图、顶栏写着新名字。

        这里不做「先验证再提交」的两段式，因为验证就是 ``build`` 本身——想验就得跑一遍，
        跑完了不如直接用它。所以是「先做后撤」，撤回点就是这六个字段。
        """
        saved = (dict(self.last_params), self.config, self.reference,
                 self.base_scenario, self.scenario_path, self.scenario_name)
        before = self.build()
        try:
            self.apply(params)
            after = self.build()
        except BaseException:
            (self.last_params, self.config, self.reference,
             self.base_scenario, self.scenario_path, self.scenario_name) = saved
            raise
        return before, after


class Handler(BaseHTTPRequestHandler):
    viewer: Viewer  # 由 serve() 赋值

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # 本地玩具，不要把访问日志刷到用户的终端上

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/style.css":
            self._send(200, (STATIC / "style.css").read_bytes(), "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send(200, (STATIC / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/api/payload":
            with self.viewer.lock:
                payload = self.viewer.build()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/params":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            self._send(400, json.dumps({"error": str(error)}).encode("utf-8"), "application/json")
            return
        try:
            with self.viewer.lock:
                before, after = self.viewer.commit(params)
        except SpecError as error:
            # 门 3：手填的参数被 frozen dataclass 自己的不变式拒绝。这不是崩溃，是**要
            # 显示的一类拒绝**——原样报回前端，让它渲染成横幅。
            self._send_rejection(type(error).__name__, str(error))
            return
        except (TypeError, ValueError, KeyError) as error:
            # ``int("abc")`` / 缺字段也会撞到这里。表单是 type=number，浏览器基本不会发
            # 这种值，但**这不是 500 的理由**：用户输入被拒绝就该是一次可读的拒绝，而不是
            # 一个不透明的服务器错误。归到门 3 那类横幅里显示。
            self._send_rejection(type(error).__name__, str(error) or "参数无法解析")
            return
        self._send(
            200,
            json.dumps({"before": before, "after": after}, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_rejection(self, error_class: str, message: str) -> None:
        """一次可读的拒绝：422 + 异常类名，前端渲染成横幅而不是白屏。"""
        body = json.dumps(
            {"error": message, "error_class": error_class}, ensure_ascii=False
        ).encode("utf-8")
        self._send(422, body, "application/json; charset=utf-8")


def serve(viewer: Viewer, *, port: int, open_browser: bool) -> int:
    handler = type("BoundHandler", (Handler,), {"viewer": viewer})
    # 只绑 127.0.0.1，**绝不是 0.0.0.0**：本地玩具，仓库里没有鉴权故事，也就不该把它
    # 暴露到局域网上。
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    host, actual = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{actual}/"
    # flush=True：双击启动器把 stdout 接到控制台时本来就不缓冲，但一旦有人
    # `viewer.cmd > log.txt`，不 flush 就什么都看不到——实测踩过，看着像没起来。
    print(f"查看器已启动：{url}", flush=True)
    print(f"  scenario: {viewer.scenario_path or '（内存里构造的）'}", flush=True)
    if len(viewer.catalog) > 1:
        # 顶栏那个选择器能换的那几个，先在终端里报一遍：用户是在这个终端里敲命令的，
        # 想看的图叫什么名字，不该只能到浏览器里去找。
        print("  可换成：" + "、".join(item["name"] for item in viewer.catalog), flush=True)
    print("  Ctrl+C 退出", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="轻量查看器：看 mapping 的搜索与 modeling 的执行",
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO,
                        help="要看的 scenario，默认 examples/chain-cap160.json")
    parser.add_argument("--config", type=Path, default=None,
                        help="可选的 modeling 配置，用来演示门 1（未迁移的服务）那一类拒绝，"
                             "例如 modeling/configs/toy_dense_decode.json。不给就用参考硬件"
                             "现造一份，时长来源默认能是 modeling 推导")
    parser.add_argument("--port", type=int, default=0,
                        help="监听端口，默认 0 = 让系统挑一个空闲端口")
    parser.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    parser.add_argument("--export", type=Path, default=None,
                        help="不起服务，直接把当前视图导出成一个自包含 HTML")
    parser.add_argument("--self-check", action="store_true",
                        help="无头自检（走查对账 / 成本桥 / 时长来源 / 载荷自洽 / 参数提交 / "
                             "HTTP 路由 / 快照 / 房子规矩；装了 node 还会真跑一遍前端 JS），退出 0/1")
    args = parser.parse_args(argv)

    if args.self_check:
        return run_self_check()

    try:
        scenario = load_and_validate(args.scenario)
    except SpecError as error:
        print(f"✗ 载入 scenario 失败：{error}")
        return 2
    config = None
    config_error = None
    if args.config is not None:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as error:
            # 不退出：scenario 一侧是好的，图 / 时间线 / 搜索树照常能看。失败原因带进去
            # 在成本桥那一屏原样显示——只 `print` 的话，`--export` 和双击启动器都把
            # stdout 丢了，用户看到的是「这一屏是空的」，而原因滚过去了。
            config_error = f"载入 {args.config} 失败：{error}"
            print(f"✗ {config_error}（成本桥那一屏会显示这条；其余照常）")

    viewer = Viewer(scenario, scenario_path=str(args.scenario), config=config,
                    config_error=config_error)

    if args.export is not None:
        html = render_snapshot(viewer.build(mode="snapshot"))
        args.export.write_text(html, encoding="utf-8", newline="\n")
        print(f"已导出快照：{args.export}（{len(html.encode('utf-8')) / 1024:.0f} KB，双击即可打开）")
        return 0

    return serve(viewer, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
