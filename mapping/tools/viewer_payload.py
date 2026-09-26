"""查看器的纯数据装配层：把 scenario 变成一页能画出来的载荷。

这个模块**不做任何决策**。它只做三件事：把内核已有的数字搬进一个 JSON 友好的
形状、把搜索的**过程**重走一遍并记下来、把两处会「拒绝」的边界原样报出来。合法性、
时间推进、最优性一律还是问内核——和 ``artifacts.py`` 同一条纪律。

为什么要把搜索重走一遍（``explore()``）
----------------------------------------
``mapper.search()`` 只回一个 :class:`SearchResult`：终局、数字、动作序列。它**不回
搜索树**，也没有回调钩子（``best_cost``/``parents`` 都是函数局部变量），所以想看
「搜索长什么样」只能在包外按同样的规则再走一遍。这份重走是按 ``mapper.py`` 逐条对着
写的，六条容易写错的地方都写在 ``explore()`` 的注释里。

重走有三条硬规则，都是实测踩出来的：

1. **两条终局路径的数字都必须来自 :func:`evaluate_mapping`**，不能从堆里的 cost 合成。
   ``mapper._budget_cut`` 就是这么做的：它把待定目标的动作重放一遍再取 makespan。
   自己从堆里取 cost 会在 ``feasible`` 那条路上得到 ``None``（而 ``search`` 报 8000000），
   因为「搜到过目标」和「这条路能走通」是两件事。``optimal`` 那条路还要镜像
   ``mapper._replay`` 的断言：重放的 makespan 与弹出的 cost 不等就是内核有 bug，宁可炸。
2. **堆元素带递增序号平局裁决**。``State`` 是 frozen 但**不可排序**（没有 ``order=True``），
   两个同 ``g`` 的条目一进堆就 ``TypeError``。这是输入相关的：只在这类图上炸。
3. **预算检查在目标检查之后、``expanded += 1`` 之前**。两行颠倒会静默地把一个可证的
   ``optimal`` 变成 ``unknown``，而且只在 ``k == expanded_states`` 那一点显形。

走完还要与 ``mapper.search()`` **逐字段对账**（``reconcile``）：status、
termination_reason、expanded_states、动作序列（tuple 相等，不是长度相等）、makespan。
对不上就在屏上打红条并禁用搜索树屏——宁可说「我不知道」，不画一棵可能是错的树。
这条对账是这份重走唯一的回归网。

节点与边的形状
--------------
``nodes`` 覆盖**每一个创建过的状态**：起点、被展开过的、进过堆但没被展开的（前沿）、
以及**只由被剪边到达**的。后两类都带 ``expand_order == -1``。每个被展开的节点带一份
``children``——它在 ``legal_actions`` 里的**全部**合法动作，每个或者有出边、或者是一条
``cut``（被支配）。因此「从某个已展开的节点出发，所有合法动作都能点」在离线也成立，
不需要 Python 侧再算一次。**没被展开的节点没有 ``children``**（前沿就是前沿），屏上
要照实说，不能装成「这个状态没有合法动作」。

``t_ns`` 与 ``g_ns`` 恒等（``t`` 就是路径代价，这也是 ``State.key()`` 敢把 ``t`` 排除在外的
原因）。两个都发是为了让自检能独立验证这条不变式；只发一个就验不了了。

单位与边界
----------
本模块只处理内核的整数纳秒，不碰 ``modeling`` 的浮点秒——那条换算在
:mod:`llm_infer_model.tensor.layer_costs` 里，``bridge_table()`` 只调用它、不重算。
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Sequence

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY_H2D,
    ACTION_EVICT,
    Action,
    EvaluationResult,
    MappingError,
    State,
    WallClock,
    evaluate_mapping,
    initial_state,
    is_goal,
    legal_actions,
    transition,
    used_vram_bytes,
)
from llm_infer_model.tensor.spec import (
    ROLE_WEIGHT,
    InitialCapacityExceeded,
    Scenario,
)
from tensor_mapping.artifacts import (
    MODE_SEARCH,
    action_to_dict,
    assumptions,
    events_document,
    limits_block,
    search_block,
    states_document,
)
from tensor_mapping.mapper import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNKNOWN,
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_POPPED,
    TERMINATION_INITIAL_CAPACITY,
    TERMINATION_STATE_BUDGET,
    TERMINATION_TIME_BUDGET,
    SearchResult,
    search,
)

__all__ = [
    "DEFAULT_MAX_NODES",
    "VIEWER_SCHEMA",
    "explore",
    "graph_block",
    "payload",
]

# 查看器自己的载荷版本，**不是** ``spec.SCHEMA_VERSION``。两者会各自演化：产物格式是
# 冻结契约，载荷是这一页的私有形状。共用一个号会让「产物没变但载荷变了」看起来像
# 「产物变了」。
VIEWER_SCHEMA = 1

# 载荷里最多记多少个状态节点。防御性的：真到这一步说明图上了一个量级，此时用户要的是
# 「搜索给了什么结论」而不是「把几百 MB 的树画出来」。超了就**停止记录、把搜索跑完**，
# 终局照常正确，另挂一个「图已截断」徽章。
DEFAULT_MAX_NODES = 5000

# 被剪边只有一种原因，而且只有一个（``mapper.py`` 的更新条件）。写成常量是为了让
# 前端的字符串有一个来源，而不是散在各处的字面量。
CUT_DOMINATED = "dominated"


# ---------------------------------------------------------------------------
# 计算图
# ---------------------------------------------------------------------------


def _ranks(scenario: Scenario) -> dict[str, int]:
    """拓扑秩：张量的秩是「生产者算子的秩 + 1」，算子的秩是「最晚输入的秩 + 1」。

    这是**最长的那个**前驱路径长度（不是最短），所以同一秩里的节点之间不可能有边，
    分层布局因此不会把边画回头。图是 DAG（``spec._has_cycle`` 在校验时就拒了环），
    所以这里可以直接记忆化递归，不需要另写拓扑排序。
    """
    workload = scenario.workload
    ranks: dict[str, int] = {}

    def tensor_rank(tensor_id: str) -> int:
        cached = ranks.get(tensor_id)
        if cached is not None:
            return cached
        producer = workload.producer_of.get(tensor_id)
        # 叶张量（输入与权重）在最左边；只有它们没有生产者。
        value = 0 if producer is None else operation_rank(producer) + 1
        ranks[tensor_id] = value
        return value

    def operation_rank(operation_id: str) -> int:
        cached = ranks.get(operation_id)
        if cached is not None:
            return cached
        inputs = workload.operation_by_id[operation_id].inputs
        value = max((tensor_rank(t) for t in inputs), default=-1) + 1
        ranks[operation_id] = value
        return value

    for tensor in workload.tensors:
        tensor_rank(tensor.id)
    for operation in workload.operations:
        operation_rank(operation.id)
    return ranks


def graph_block(scenario: Scenario) -> dict[str, Any]:
    """计算图本身：张量、算子、连线、布局秩。

    一切从 ``workload`` 推，**没有任何常量**。依赖关系用 ``producer_of`` /
    ``consumers_of``——内核的合法性检查（``engine._check_compute`` 的 INPUT_NOT_READY、
    ``_check_evict`` 的 LIVE_VALUE_LOSS）就是从这两张表推的，所以画出来的图不可能与
    模型不一致。前端只负责算坐标。
    """
    workload = scenario.workload
    ranks = _ranks(scenario)
    outputs = set(workload.outputs)
    copyable = set(scenario.mapspace.copy_tensor_ids)

    tensors = [
        {
            "id": tensor.id,
            "name": tensor.name,
            "role": tensor.role,
            "dtype": tensor.dtype,
            "storage_bytes": tensor.storage_bytes,
            # 占用是**对齐后**的；``storage_bytes`` 是搬运量。两者在 alignment > 1 时
            # 不同，混淆它们正是 ``ACCEPTANCE.md`` §5 点名的那类错误。
            "alloc_bytes": scenario.size_alloc(tensor.id),
            "initial_locations": list(tensor.initial_locations),
            "producer": workload.producer_of.get(tensor.id),
            "consumers": list(workload.consumers_of.get(tensor.id, ())),
            "is_leaf": tensor.id in workload.leaf_tensors,
            "is_output": tensor.id in outputs,
            "is_weight": tensor.role == ROLE_WEIGHT,
            "copyable": tensor.id in copyable,
            "rank": ranks[tensor.id],
        }
        for tensor in sorted(workload.tensors, key=lambda t: t.id)
    ]
    operations = [
        {
            "id": operation.id,
            "semantic_op": operation.semantic_op,
            "ggml_op": operation.ggml_op,
            "unary_op": operation.unary_op,
            "inputs": list(operation.inputs),
            "output": operation.output,
            "compute_ns": scenario.costs.compute[operation.id].duration_ns,
            "workspace_bytes": scenario.costs.compute[operation.id].workspace_bytes,
            "output_alloc_bytes": scenario.size_alloc(operation.output),
            "rank": ranks[operation.id],
        }
        for operation in sorted(workload.operations, key=lambda o: o.id)
    ]
    # 连线在 Python 侧算好，前端只画。``from``/``to`` 是张量或算子的 id，靠两条表
    # （``tensor_by_id`` / ``operation_by_id``）区分，不需要单独的节点类型字段。
    links: list[dict[str, Any]] = []
    for operation in sorted(workload.operations, key=lambda o: o.id):
        for index, tensor_id in enumerate(operation.inputs):
            links.append(
                {
                    "from": tensor_id,
                    "to": operation.id,
                    "kind": "input",
                    "slot": index,
                    "weight": workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT,
                }
            )
        links.append({"from": operation.id, "to": operation.output, "kind": "output"})

    return {
        "workload_id": workload.id,
        "origin": {
            "kind": workload.origin.kind,
            "ggml_version": workload.origin.ggml_version,
            "sample": workload.origin.sample,
        },
        "outputs": list(workload.outputs),
        "tensors": tensors,
        "operations": operations,
        "links": links,
        # 张量画在偶数列、算子画在奇数列（所以算子的列号是 ``2*秩 - 1``）。秩越大越靠右；
        # 算子的输入张量秩 ≤ 秩-1、输出张量秩 = 秩+1，于是**每条边都从左指向右**，不会画
        # 回头。（注意：「同秩」的算子与张量里，算子在左——不是反过来。）
        "columns": {
            "tensors": {t["id"]: 2 * t["rank"] for t in tensors},
            "operations": {o["id"]: 2 * o["rank"] - 1 for o in operations},
        },
    }


# ---------------------------------------------------------------------------
# 搜索走查
# ---------------------------------------------------------------------------


def _reconstruct(
    parents: dict[tuple[Any, ...], tuple[tuple[Any, ...], Action]],
    goal_key: tuple[Any, ...],
    start_key: tuple[Any, ...],
) -> tuple[Action, ...]:
    """沿父链走回起点再反转。与 ``mapper._reconstruct`` 同一份逻辑。"""
    actions: list[Action] = []
    key = goal_key
    while key != start_key:
        parent_key, action = parents[key]
        actions.append(action)
        key = parent_key
    actions.reverse()
    return tuple(actions)


def _replay(scenario: Scenario, actions: Sequence[Action], claimed_ns: int) -> EvaluationResult:
    """重放并镜像 ``mapper._replay`` 的断言。

    走查与内核共用同一个 ``transition``，所以两者对不上只可能是这段重走写错了。宁可
    当场炸掉，也不要在屏上画一个内核不认的 makespan。
    """
    evaluation = evaluate_mapping(scenario, actions)
    if evaluation.status != "valid" or evaluation.makespan_ns != claimed_ns:
        raise RuntimeError(
            f"走查在 {claimed_ns} ns 到达目标，但重放这份映射得到 "
            f"status={evaluation.status!r}, makespan={evaluation.makespan_ns!r} "
            f"({evaluation.reason})；走查与 mapper.py 已经分叉"
        )
    return evaluation


def _reconcile(
    block: dict[str, Any], result: SearchResult | None, expected_actions: tuple[Action, ...]
) -> dict[str, Any]:
    """把走查的终局与 ``mapper.search()`` 逐字段比一遍。

    比的是**动作序列本身**（tuple 相等）而不是长度——两条不同的最优映射完全可能一样长，
    而「一样长」什么都不证明。``makespan_ns`` 在两条路上都来自 ``evaluate_mapping``，
    所以相等是应当的，不是巧合。
    """
    if result is None:
        return {"checked": False, "agrees": None, "differences": []}

    differences: list[str] = []
    verdict = block["verdict"]
    if verdict["status"] != result.status:
        differences.append(f"status: 走查 {verdict['status']!r} vs search {result.status!r}")
    if verdict["termination_reason"] != result.termination_reason:
        differences.append(
            f"termination_reason: 走查 {verdict['termination_reason']!r} vs "
            f"search {result.termination_reason!r}"
        )
    if verdict["optimality_proven"] != result.optimality_proven:
        differences.append(
            f"optimality_proven: 走查 {verdict['optimality_proven']} vs "
            f"search {result.optimality_proven}"
        )
    if verdict["expanded_states"] != result.expanded_states:
        differences.append(
            f"expanded_states: 走查 {verdict['expanded_states']} vs search {result.expanded_states}"
        )
    if expected_actions != result.actions:
        differences.append(
            f"动作序列不同：走查 {len(expected_actions)} 条 vs search {len(result.actions)} 条"
        )
    if block["solution"]["makespan_ns"] != result.makespan_ns:
        differences.append(
            f"makespan_ns: 走查 {block['solution']['makespan_ns']} vs search {result.makespan_ns}"
        )
    return {"checked": True, "agrees": not differences, "differences": differences}


def explore(
    scenario: Scenario,
    *,
    max_expanded_states: int | None = None,
    wall_time_limit_s: float | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    result: SearchResult | None = None,
    clock: Any = None,
) -> dict[str, Any]:
    """重走一次 Dijkstra，把整棵状态图记下来，并与 ``search()`` 对账。

    ``result`` 是调用方已经算好的 :class:`SearchResult`（传 None 就自己调一次
    ``search``）。传进来而不是在里面调，是为了让调用方决定要不要多花那一次搜索。

    ``clock`` 只在自检里用：Windows 上 ``time.monotonic()`` 实测是 ``GetTickCount64()``
    （分辨率 15.625 ms），所以 ``wall_time_limit_s=1e-9`` 这种极小预算**永远不触发**
    ——第一次 ``elapsed_s()`` 就返回 0.0。想确定性地走一遍 ``time_budget_exhausted``
    就得注入一个假时钟，和 ``mapper.search`` 的测试同一个做法。
    """
    state_limit = (
        scenario.mapper.max_expanded_states
        if max_expanded_states is None
        else max_expanded_states
    )
    time_limit = (
        scenario.mapper.wall_time_limit_s
        if wall_time_limit_s is None
        else wall_time_limit_s
    )
    if clock is None:
        clock = WallClock(time_limit)

    expanded = 0
    best_cost: dict[tuple[Any, ...], int] = {}
    parents: dict[tuple[Any, ...], tuple[tuple[Any, ...], Action]] = {}
    nodes: list[dict[str, Any]] = []
    # 与 ``nodes`` 同序的 State，只为走完之后补算「每个已展开节点有哪些动作不合法」。
    # 放这里而不是塞进节点字典，是因为 State 不是 JSON，而且节点要精简。
    states: list[State] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    cut_counts: dict[str, int] = {}
    truncated = False

    def register(
        key: tuple[Any, ...],
        state: State,
        cost: int,
        parent_id: int | None,
        action: Action | None,
    ) -> int | None:
        """把一个状态登记成节点；已经登记过的只在**更省**时更新父与代价。

        返回节点 id，超出 ``max_nodes`` 时返回 ``None``（此时**不**建节点，但搜索
        照常往下跑）。
        """
        nonlocal truncated
        existing = index_by_key.get(key)
        if existing is not None:
            node = nodes[existing]
            if cost < node["g_ns"]:
                # 只有严格更省才换父。等值保持第一条路径——这正是零成本动作不会
                # 被按各种排列各走一遍的原因（``mapper.py`` 的更新条件）。
                node["g_ns"] = cost
                node["t_ns"] = state.t
                node["parent"] = parent_id
                node["action"] = action_to_dict(action) if action is not None else None
            return existing
        if len(nodes) >= max_nodes:
            truncated = True
            return None
        node_id = len(nodes)
        index_by_key[key] = node_id
        nodes.append(
            {
                "id": node_id,
                "t_ns": state.t,
                "g_ns": cost,
                "used_vram_bytes": used_vram_bytes(scenario, state),
                "op_status": dict(state.op_status),
                "copy_status": dict(state.copy_status),
                "running": [
                    {
                        "resource": task.resource,
                        "target_id": task.target_id,
                        "remaining_ns": task.remaining_ns,
                    }
                    for task in state.running
                ],
                "is_goal": is_goal(scenario, state),
                "parent": parent_id,
                "action": action_to_dict(action) if action is not None else None,
                "expand_order": -1,
                "role": "cut",
                "depth": 0,
                "on_solution_path": False,
                "children": [],
            }
        )
        states.append(state)
        return node_id

    verdict: dict[str, Any] = {
        "status": None,
        "termination_reason": None,
        "optimality_proven": False,
        "expanded_states": 0,
        "visited_states": 0,
        "wall_time_s": 0.0,
    }
    solution: dict[str, Any] = {"actions": [], "makespan_ns": None, "peak_vram_bytes": None,
                               "h2d_bytes": None, "evaluation": None}

    started = time.monotonic()
    try:
        start = initial_state(scenario)
    except InitialCapacityExceeded:
        # 起点就装不下：这是关于 scenario 的结论，不是输入错误，而且**是可证的**
        # （没有任何映射能绕过起点）。别把「展开 0 个」读成「没跑」。
        verdict.update(
            status=STATUS_INFEASIBLE,
            termination_reason=TERMINATION_INITIAL_CAPACITY,
            optimality_proven=True,
            wall_time_s=time.monotonic() - started,
        )
        block = {
            "verdict": verdict,
            "start_id": None,
            "goal_id": None,
            "nodes": [],
            "frontier": [],
            "cut_counts": {},
            "budget": {"max_expanded_states": state_limit, "wall_time_limit_s": time_limit},
            "truncated": False,
            "solution": solution,
            # 起点都没活下来，没有节点，也就没有候选可言。
            "candidates": {"messages": [], "nodes": []},
            "reconcile": None,
        }
        if result is None:
            result = search(
                scenario,
                max_expanded_states=max_expanded_states,
                wall_time_limit_s=wall_time_limit_s,
                clock=clock,
            )
        block["reconcile"] = _reconcile(block, result, ())
        return block

    start_key = start.key()
    best_cost[start_key] = 0
    start_id = register(start_key, start, 0, None, None)
    assert start_id == 0

    # (累计时间, 递增序号, 状态)。序号让同 g 的条目有确定的先后，也让堆**永远不需要
    # 比较两个 State**——它本来也做不到，State 没有 order=True。
    queue: list[tuple[int, int, State]] = [(0, 0, start)]
    sequence = 1
    # 见过但还没弹出的最省目标。只有预算切断时才会被用到：正是它让搜索能答
    # `feasible` 而不是 `unknown`。
    pending_goal: tuple[int, tuple[Any, ...]] | None = None

    goal_key: tuple[Any, ...] | None = None
    claimed_ns: int | None = None
    cut_reason: str | None = None

    while queue:
        cost, _, state = heapq.heappop(queue)
        key = state.key()
        if cost > best_cost[key]:
            continue  # 被更省的路径取代的陈旧条目

        if is_goal(scenario, state):
            goal_key, claimed_ns = key, cost
            break

        # 在展开**之前**检查，而不是之后：已经弹出的目标因此总是被确认，而不是被
        # 丢掉。顺序颠倒会静默地把可证的 optimal 变成 unknown。
        if state_limit and expanded >= state_limit:
            cut_reason = TERMINATION_STATE_BUDGET
            break
        if clock.expired():
            cut_reason = TERMINATION_TIME_BUDGET
            break

        expanded += 1
        node_id = index_by_key.get(key)
        if node_id is not None:
            nodes[node_id]["expand_order"] = expanded

        children: list[dict[str, Any]] = []
        for action in legal_actions(scenario, state):
            successor = transition(scenario, state, action)
            successor_key = successor.key()
            # 只有 ADVANCE 有代价，另外三个不动时钟。
            successor_cost = cost + (successor.t - state.t)
            previous = best_cost.get(successor_key)
            if previous is not None and successor_cost >= previous:
                # 被支配：已经有一条不更差的路径。**只记录，不展开**——这就是
                # 「被剪边」在屏上的全部含义。
                to_id = index_by_key.get(successor_key)
                cut_counts[CUT_DOMINATED] = cut_counts.get(CUT_DOMINATED, 0) + 1
                if to_id is not None:
                    children.append(
                        {
                            "action": action_to_dict(action),
                            "to": to_id,
                            "cut": {
                                "reason": CUT_DOMINATED,
                                "g_new_ns": successor_cost,
                                "g_existing_ns": previous,
                            },
                        }
                    )
                continue
            best_cost[successor_key] = successor_cost
            parents[successor_key] = (key, action)
            successor_id = register(successor_key, successor, successor_cost, node_id, action)
            heapq.heappush(queue, (successor_cost, sequence, successor))
            sequence += 1
            if successor_id is not None:
                children.append(
                    {"action": action_to_dict(action), "to": successor_id, "cut": None}
                )
            if is_goal(scenario, successor):
                if pending_goal is None or successor_cost < pending_goal[0]:
                    pending_goal = (successor_cost, successor_key)

        if node_id is not None:
            nodes[node_id]["children"] = children

    # --- 终局：五种，照抄 ``mapper.search``，不自创 ---
    actions: tuple[Action, ...] = ()
    evaluation: EvaluationResult | None = None
    if goal_key is not None:
        verdict.update(
            status=STATUS_OPTIMAL,
            termination_reason=TERMINATION_GOAL_POPPED,
            optimality_proven=True,
        )
        actions = _reconstruct(parents, goal_key, start_key)
        assert claimed_ns is not None
        evaluation = _replay(scenario, actions, claimed_ns)
    elif cut_reason is not None:
        verdict.update(termination_reason=cut_reason, optimality_proven=False)
        if pending_goal is None:
            verdict.update(status=STATUS_UNKNOWN)
        else:
            # 目标见过但没弹出：这是一份真能重放的映射，所以报 feasible——**绝不能**
            # 报 optimal，搜索没有证明不存在更省的。
            verdict.update(status=STATUS_FEASIBLE)
            actions = _reconstruct(parents, pending_goal[1], start_key)
            evaluation = evaluate_mapping(scenario, actions)
    else:
        verdict.update(
            status=STATUS_INFEASIBLE,
            termination_reason=TERMINATION_EXHAUSTED,
            optimality_proven=True,
        )

    verdict["expanded_states"] = expanded
    verdict["visited_states"] = len(best_cost)
    verdict["wall_time_s"] = time.monotonic() - started

    # 解路径：沿父链把节点标出来，前端的树据此高亮。
    terminal_key = goal_key if goal_key is not None else (
        pending_goal[1] if (cut_reason is not None and pending_goal is not None) else None
    )
    if terminal_key is not None:
        key = terminal_key
        while True:
            node_id = index_by_key.get(key)
            if node_id is None:
                break
            nodes[node_id]["on_solution_path"] = True
            if key == start_key:
                break
            key = parents[key][0]

    # 深度与前沿：深度从父链算（父在更新时会换），前沿是「进过堆但从未展开」的那些。
    # 只由被剪边到达的节点既没进过堆也不会被展开，是第三类，屏上要分开说。
    pushed = {index_by_key[key] for key in best_cost if key in index_by_key}
    frontier: list[int] = []
    for node in nodes:
        if node["expand_order"] > 0:
            node["role"] = "start" if node["id"] == start_id else "expanded"
        elif node["id"] in pushed:
            node["role"] = "frontier"
            frontier.append(node["id"])
        else:
            # 只由被剪边到达：从来没有进过堆，所以也不会被展开。它**不是**前沿，
            # 屏上不能混为一谈。
            node["role"] = "cut"
        parent_id = node["parent"]
        if parent_id is not None:
            node["depth"] = nodes[parent_id]["depth"] + 1
    frontier.sort()

    solution = {
        "actions": [action_to_dict(action) for action in actions],
        "action_count": len(actions),
        "makespan_ns": evaluation.makespan_ns if evaluation else None,
        "peak_vram_bytes": evaluation.peak_vram_bytes if evaluation else None,
        "h2d_bytes": evaluation.h2d_bytes if evaluation else None,
        "evaluation_status": evaluation.status if evaluation else None,
        "evaluation_reason": evaluation.reason if evaluation else None,
        "evaluation": evaluation,
    }

    block = {
        "verdict": verdict,
        "start_id": start_id,
        "goal_id": index_by_key.get(goal_key) if goal_key is not None else None,
        "nodes": nodes,
        "frontier": frontier,
        "cut_counts": cut_counts,
        "budget": {"max_expanded_states": state_limit, "wall_time_limit_s": time_limit},
        "truncated": truncated,
        "solution": solution,
        # 每个已展开节点的**非法**候选（合法的在 children 里，不存两遍）。去重后的理由
        # 表放在 candidates.messages。
        "candidates": _intern_candidates(scenario, nodes, states),
        "reconcile": None,
    }

    if result is None:
        result = search(
            scenario,
            max_expanded_states=max_expanded_states,
            wall_time_limit_s=wall_time_limit_s,
            clock=clock,
        )
    block["reconcile"] = _reconcile(block, result, actions)
    return block


# ---------------------------------------------------------------------------
# 载荷装配
# ---------------------------------------------------------------------------


def ordered_candidates(scenario: Scenario) -> list[Action]:
    """**全部**候选动作，固定顺序。

    契约顺序（``engine.legal_actions``）：COPY_H2D 按 ``mapspace.copy_tensor_ids`` 排序
    → COMPUTE 按算子 id 排序 → EVICT 按张量 id 排序 → ADVANCE 最后。照抄而不是自己排，
    是为了让候选表不随实现漂移——``legal_actions`` 只给合法集，这份是全集的顺序来源，
    两者必须用同一套次序，否则屏上「为什么这个不行」会指着一个邻居说。
    """
    ordered: list[Action] = []
    for tensor_id in sorted(scenario.mapspace.copy_tensor_ids):
        ordered.append(Action(ACTION_COPY_H2D, tensor_id))
    for operation_id in sorted(scenario.workload.operation_by_id):
        ordered.append(Action(ACTION_COMPUTE, operation_id))
    for tensor_id in sorted(scenario.workload.tensor_by_id):
        ordered.append(Action(ACTION_EVICT, tensor_id))
    ordered.append(Action(ACTION_ADVANCE))
    return ordered


def illegal_candidates(scenario: Scenario, state: State) -> list[dict[str, Any]]:
    """**不合法**的候选动作，带内核自己给的理由。

    理由靠 ``transition`` 自己抛：:class:`MappingError` 带公开的 ``code`` / ``message``，
    所以这里读的是契约，不是内部实现。合法的那些**不在这里**——它们在节点的
    ``children`` 里，存两遍就是两个真相。
    """
    rows: list[dict[str, Any]] = []
    for action in ordered_candidates(scenario):
        try:
            transition(scenario, state, action)
        except MappingError as error:
            rows.append(
                {
                    "action": action_to_dict(action),
                    "code": error.code,
                    "message": error.message,
                }
            )
    return rows


def _intern_candidates(
    scenario: Scenario, nodes: list[dict[str, Any]], states: list[State]
) -> dict[str, Any]:
    """把每个已展开节点的非法候选收成一张**去重字符串表**。

    实测（``three_layer_chain-cap1024``）：2206 条非法候选的原文是 463 KB，但只有
    **45 条不同的理由**，去重后 64 KB。少了七倍，所以不搞按需取、不搞懒加载，
    直接烤进载荷——快照模式下没有 Python 侧可以问。
    """
    messages: list[list[str | None]] = []
    seen: dict[tuple[str | None, str], int] = {}
    rows: list[list[Any]] = []
    for node, state in zip(nodes, states):
        if node["expand_order"] <= 0:
            continue  # 没被展开过就没有「为什么不行」可言
        entries: list[list[Any]] = []
        for item in illegal_candidates(scenario, state):
            key = (item["code"], item["message"])
            index = seen.get(key)
            if index is None:
                index = len(messages)
                seen[key] = index
                messages.append([item["code"], item["message"]])
            action = item["action"]
            entries.append(
                [
                    action["kind"],
                    action.get("operation_id") or action.get("tensor_id"),
                    index,
                ]
            )
        rows.append([node["id"], entries])
    return {"messages": messages, "nodes": rows}


def _log_block(
    scenario: Scenario, evaluation: EvaluationResult | None, note: str | None
) -> dict[str, Any]:
    """事件与状态，**直接用 artifacts 的公开序列化器**。

    自己手写事件/状态字典等于给「查看器与磁盘上的 ``events.json`` 早晚对不上」埋雷。
    复用之后载荷就是产物再加坐标：``_state_to_dict`` 会补上 ``StateSnapshot`` 刻意没有
    的 ``action_index``，并把 ``t`` 改名成 ``t_ns``。
    """
    events = events_document(mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=note)
    states = states_document(mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=note)
    return {"events": events["events"], "states": states["states"]}


def payload(
    scenario: Scenario,
    *,
    scenario_path: str | None = None,
    mode: str = "live",
    search_result: SearchResult | None = None,
    max_expanded_states: int | None = None,
    wall_time_limit_s: float | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """把一个 scenario 装成整页载荷。

    ``mode`` 是 ``"live"``（本地服务）或 ``"snapshot"``（导出的自包含 HTML）。快照是
    活页面的**死拷贝**：没有 Python 侧可以对话，所以参数编辑与重跑在那一侧必须禁用，
    屏上也要说清楚原因，而不是给一个点了没反应的按钮。
    """
    result = search_result
    if result is None:
        result = search(
            scenario,
            max_expanded_states=max_expanded_states,
            wall_time_limit_s=wall_time_limit_s,
        )

    block = explore(
        scenario,
        max_expanded_states=max_expanded_states,
        wall_time_limit_s=wall_time_limit_s,
        max_nodes=max_nodes,
        result=result,
    )
    evaluation = block["solution"]["evaluation"]
    block["solution"].pop("evaluation", None)

    limits = limits_block(
        scenario,
        max_expanded_states=result.max_expanded_states,
        wall_time_limit_s=result.wall_time_limit_s,
    )
    return {
        "viewer_schema": VIEWER_SCHEMA,
        "generated_by": "mapping/tools/viewer.py",
        "mode": mode,
        "scenario_path": scenario_path,
        "scenario_id": scenario.id,
        "scenario_fingerprint": result.scenario_fingerprint,
        "workload_fingerprint": result.workload_fingerprint,
        "solution": {
            **block["solution"],
            "status": result.status,
            "termination_reason": result.termination_reason,
            "optimality_proven": result.optimality_proven,
            "search": search_block(result),
        },
        "explore": block,
        "graph": graph_block(scenario),
        "log": _log_block(scenario, evaluation, None),
        "source": {"limits": limits, "assumptions": assumptions(scenario)},
        "params_echo": params_echo(scenario, result),
        "checks": [],
    }


# ---------------------------------------------------------------------------
# 成本桥：modeling 的逐层预测 → 内核的整数纳秒
# ---------------------------------------------------------------------------

# 示例图里每个算子的 FLOPs：4×4 的权重乘 4 维向量 = 16 次乘加 = 32 FLOP
# （``demo_window_vs_search.py`` 的注释与示例里写死的成本都是这么来的）。
FLOPS_PER_OP = 32.0

# 一组能复现示例那几行成本的硬件参数。带宽 32 kB/s 是刻意取小的：64 B / 32 kB/s 恰好
# 是 1 ms，加上 1 ms 的固定延迟就是示例里的 3 ms。
#
# **这份常量不 import demo**：让一个工具 import 另一个会破坏「包外一把手工具」的纪律，
# 也会把 demo 的数字变成查看器的承重件。重复一份是安全的，因为**它是被检查的**——
# ``--self-check`` 的 B 组断言这组参数能复现 fixture 声明的 ns。两份被检查的常量没问题，
# 两份没人查的才有问题。
REFERENCE_HARDWARE = {
    "gpu_effective_flops": 16_000.0,
    "h2d_bandwidth_bytes_per_s": 32_000.0,
    "h2d_latency_s": 1e-3,
}

# ``layer_costs._assert_no_extra_services`` 逐项点名的那些开关。``needle`` 是那条**原样
# 消息**里的字段路径片段——消息永远是权威：未来 guard 加了字段而这张表里没有，消息照常
# 渲染成散文，只是勾选框不全，是降级不是矛盾。
BRIDGE_TOGGLES: tuple[dict[str, Any], ...] = (
    {"block": "hardware", "field": "cpu_effective_flops", "off": None,
     "needle": "hardware.cpu_effective_flops", "label": "CPU 数学"},
    {"block": "hardware", "field": "global_bytes", "off": 0,
     "needle": "hardware.global_bytes", "label": "全局暂存 global_bytes"},
    {"block": "hardware", "field": "kv_bytes", "off": 0,
     "needle": "hardware.kv_bytes", "label": "KV 常驻 kv_bytes"},
    {"block": "hardware", "field": "workspace_bytes", "off": 0,
     "needle": "hardware.workspace_bytes", "label": "工作区 workspace_bytes"},
    {"block": "hardware", "field": "activation_transfer_bytes", "off": 0,
     "needle": "hardware.activation_transfer_bytes", "label": "激活搬运"},
    {"block": "hardware", "field": "host_staging_bandwidth_bytes_per_s", "off": None,
     "needle": "host_staging_bandwidth_bytes_per_s", "label": "host staging"},
    {"block": "hardware", "field": "state_transfer_bandwidth_bytes_per_s", "off": None,
     "needle": "state_transfer_bandwidth_bytes_per_s", "label": "状态往返搬运"},
    {"block": "hardware", "field": "kv_storage_cached_bandwidth_bytes_per_s", "off": None,
     "needle": "kv_storage_cached_bandwidth_bytes_per_s", "label": "KV 分层：缓存带宽"},
    {"block": "hardware", "field": "kv_storage_uncached_bandwidth_bytes_per_s", "off": None,
     "needle": "kv_storage_uncached_bandwidth_bytes_per_s", "label": "KV 分层：未缓存带宽"},
    {"block": "hardware", "field": "kv_storage_cache_bytes", "off": 0,
     "needle": "kv_storage_cache_bytes", "label": "KV 分层：缓存容量"},
    {"block": "state", "field": None, "off": None,
     "needle": "state（每层 KV", "label": "每层 KV / SSM 状态"},
    {"block": "policy", "field": "static_gpu_layers", "off": 0,
     "needle": "policy.static_gpu_layers", "label": "静态常驻层"},
    {"block": "policy", "field": "token_pipeline_overhead_s", "off": 0.0,
     "needle": "policy.token_pipeline_overhead_s", "label": "每 token 控制开销"},
    {"block": "policy", "field": "layer_scheduler_overhead_s", "off": 0.0,
     "needle": "policy.layer_scheduler_overhead_s", "label": "层调度开销"},
)

# 三类拒绝的补救方向**不一样**，屏上必须分开说。只讲「未迁移」那一类，用户关掉所有开关
# 之后撞上第二道门时会以为查看器坏了。
REFUSAL_REMEDY = {
    "unsupported": "补救：把这些服务**逐项关掉**（下面每一行都能点）。它们在本轮的动作空间里没有对应动作。",
    "invalid": "补救：**改数字或改图**。这一道门与迁移无关——它是「浮点秒 → 整数纳秒」的分辨率损失本身。",
    "value": "补救：**改参数**。这是 frozen dataclass 自己的不变式拒绝了手填的值。",
    # 第四类不是门：三道门都是 ``costs_from_model_config`` 在拒绝，这一类是**还没走到门口**
    # ——配置根本没读进来。所以措辞上不说「门 4」，也不给「换成参考硬件」（那也要先有个配置）。
    "config": "补救：**检查 ``--config`` 指的文件**。这是三道门之前的一步——配置没读进来，"
              "所以开关清单是空的，那不是坏了。",
}

REFUSAL_CLASS_LABEL = {
    "unsupported": "Unsupported · 未迁移的服务",
    "invalid": "InvalidInput · 单位分辨率 / 图形状",
    "value": "ValueError · 参数不变式",
    "config": "配置未载入 · 路径或内容",
}


def _sole_weight(workload: Any, operation_id: str) -> tuple[str | None, int]:
    """这个算子的唯一权重输入。返回 ``(权重 id | None, 权重份数)``。

    份数一起返回是因为「0 份」与「2 份」都要报出来，而它们都不是 ``None`` 能表达的。
    """
    operation = workload.operation_by_id[operation_id]
    weights = [
        tensor_id
        for tensor_id in operation.inputs
        if workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT
    ]
    return (weights[0] if len(weights) == 1 else None), len(weights)


def derive_layers(scenario: Scenario, *, flops_per_op: float = FLOPS_PER_OP) -> tuple[
    dict[str, Any], list[dict[str, Any]]
]:
    """算子 → :class:`LayerSpec`，**weight_bytes 从 scenario 推**。

    ``demo_window_vs_search.py:81`` 写死了 ``WEIGHT_BYTES = 64``，但 ``matvec`` 的权重是
    **48 字节**——照抄常量会让五个示例里的一个静默算错。从张量的 ``storage_bytes`` 推，
    ``costs_for_chain`` 的字节一致性检查就此恒真，还消掉一整类伪拒绝。

    返回 ``(layers, problems)``。``problems`` 非空表示图的**形状**不在「一个计算节点恰好
    一份权重」这个范围内（``fork``/``residual`` 的 0 权重 ADD 算子）。这一条
    ``costs_for_chain`` 自己也会查（它的条件 2），但那样报出来的是「键集合缺少 ['add']」，
    指向的是配对而不是图形状。所以这里先查一遍，好把话说准。
    """
    from llm_infer_model.model import LayerSpec

    workload = scenario.workload
    layers: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    for index, operation_id in enumerate(sorted(workload.operation_by_id)):
        weight_id, weight_count = _sole_weight(workload, operation_id)
        if weight_id is None:
            problems.append(
                {
                    "operation_id": operation_id,
                    "weight_count": weight_count,
                    "inputs": list(workload.operation_by_id[operation_id].inputs),
                }
            )
            continue
        layers[operation_id] = LayerSpec(
            name=f"blk.{index}",
            weight_bytes=workload.tensor_by_id[weight_id].storage_bytes,
            flops=flops_per_op,
        )
    return layers, problems


def apply_toggle(config: Any, block: str, field: str | None, value: Any) -> Any:
    """把一个开关换成 off 值，两层 ``dataclasses.replace``。

    ``replace`` 会重跑 ``__post_init__``，所以手填的非法参数会当场 ``ValueError``——
    这不是要绕开的麻烦，它本身就是要显示的一类拒绝（门 3）。
    """
    from dataclasses import replace

    if block == "state":
        return replace(config, state=None)
    section = getattr(config, block)
    return replace(config, **{block: replace(section, **{field: value})})


def _toggle_rows(config: Any, message: str) -> list[dict[str, Any]]:
    """每一行开关当前是 on 还是 off。

    判据是**那条原样消息里有没有这个词**（消息是权威），另附上配置里**现在**的值——
    用户在屏上点掉一行之后，看得到值确实变了。两者不一致时两栏并排，一眼能看出。
    """
    rows: list[dict[str, Any]] = []
    for toggle in BRIDGE_TOGGLES:
        block, field = toggle["block"], toggle["field"]
        current = getattr(config, block) if field is None else getattr(getattr(config, block), field)
        rows.append(
            {
                "block": block,
                "field": field,
                "label": toggle["label"],
                "off_value": toggle["off"],
                "off_is_null": toggle["off"] is None,
                "enabled": toggle["needle"] in message,
                "value_now": None if current is None else (
                    current if isinstance(current, (int, float, str)) else str(current)
                ),
                "value_is_null": current is None,
            }
        )
    return rows


def _refusal(
    scenario: Scenario,
    config: Any,
    kind: str,
    *,
    code: str | None,
    message: str,
    stage: str,
    layers: dict[str, Any] | None = None,
    problems: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "ok": False,
        "refusal_class": kind,
        "refusal_label": REFUSAL_CLASS_LABEL[kind],
        "code": code,
        "message": message,
        "stage": stage,
        "remedy": REFUSAL_REMEDY[kind],
        "toggles": _toggle_rows(config, message) if config is not None else [],
        "structural_problems": problems or [],
        "layers": [
            {
                "operation_id": operation_id,
                "name": layer.name,
                "weight_bytes": layer.weight_bytes,
                "flops": layer.flops,
                "weight_tensor_id": _sole_weight(scenario.workload, operation_id)[0],
            }
            for operation_id, layer in sorted((layers or {}).items())
        ],
        "table": [],
        "h2d": [],
        "all_agree": None,
        # 见 ``bridge_table`` 的 docstring：装的是推导出来的 ``Costs``，由
        # ``sourced_scenario`` pop 掉。拒绝这一路上没有东西可装，但键要在，
        # 好让调用方无条件 pop。
        "_costs": None,
    }


def bridge_table(
    scenario: Scenario,
    *,
    config: Any,
    config_error: str | None = None,
    flops_per_op: float = FLOPS_PER_OP,
    reference: bool = False,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 modeling 的逐层预测换算成内核的整数纳秒动作，并把**被拒绝的那些项**也报出来。

    这是「时长来源」那一屏的全部数据。它只用公开 API：``costs_from_model_config``、
    ``Unsupported``/``InvalidInput``/``ValueError`` 三种异常、以及 scenario 自己的成本表。
    **永不** import 私有的 ``_assert_no_extra_services``——那条 guard 的消息就是给用户看的
    文本，读它、渲染它，但不复制它的判断。

    ``reference=True`` 时把硬件换成 :data:`REFERENCE_HARDWARE`（VRAM 仍取 scenario 的），
    仍然走 ``costs_from_model_config``，所以门 1 还在链路上——参考硬件不是绕过 guard 的
    后门，它只是换一组能让数字落到纳秒分辨率之上的参数。

    ``hardware`` 在生效硬件之上**逐字段**覆盖那三个数（FLOP/s、带宽、延迟）。这是屏上那组
    旋钮的入口，也是「联动」唯一看得见的地方：参考硬件那组恰好复现示例写死的值，不动它，
    时间线上的宽度一个像素都不会变。

    成功的返回值里带一个 ``_costs`` 键，装的是**推导出来的那个** :class:`Costs` 对象。
    它不是 JSON 能序列化的，所以 :func:`sourced_scenario` 会把它 pop 掉——留着就会在
    ``render_snapshot`` 的 ``json.dumps`` 上炸。之所以夹带而不是让调用方再推一次，是因为
    这里已经把层、硬件、异常路径都摆好了，重推一遍等于把同一条链路走两次，将来只会漂。
    """
    from llm_infer_model.tensor.layer_costs import costs_from_model_config
    from llm_infer_model.tensor.spec import InvalidInput, Unsupported

    if config is None and config_error is not None:
        # ``--config`` 指的文件没读进来（``main()`` 的选择是「打印失败、继续跑」，因为
        # scenario 那边还活着，图和时间线照常能看）。
        #
        # 不加这一支的后果实测过：``costs_from_model_config(None, …)`` 会在
        # ``config.hardware`` 上抛 AttributeError——不在下面三个 except 里，于是**整个
        # 查看器**（连和时长来源无关的屏一起）死在一条 traceback 上，而上面刚打印的那句
        # 「✗ 载入 config 失败」反倒成了噪音。``_refusal`` 本来就容得下 ``config=None``
        # （开关清单为空），缺的只是别往下走。
        #
        # 注意这一支只接「给了但读不出来」。**没给** ``--config`` 是另一回事：那不是失败，
        # 是没要，下面照样用参考硬件推——否则默认打开查看器时时长来源永远是空的。
        return _refusal(
            scenario, config, "config",
            code=None,
            message=config_error,
            stage="--config 指的文件",
        )
    if config is None:
        reference = True

    effective: Any = config
    try:
        # 层的构造也在 try 里：``LayerSpec(flops=0)`` 会当场抛 ValueError，那正是门 3
        # 可被触发的一条真实路径（屏上那个 flops_per_op 旋钮）。
        layers, problems = derive_layers(scenario, flops_per_op=flops_per_op)
        if problems:
            detail = "；".join(
                f"算子 {p['operation_id']} 有 {p['weight_count']} 份权重输入"
                for p in problems
            )
            return _refusal(
                scenario,
                effective,
                "unsupported",
                code=None,
                message=(
                    f"{detail}。成本桥接只支持「一个计算节点恰好一份权重」，不做任何平均摊派，"
                    "所以这张图（分叉 / 残差汇合）没有对应的层。通用搜索照常支持这类图——"
                    "受限的是成本桥，不是内核。"
                ),
                stage="graph ↔ layers 的配对",
                layers=layers,
                problems=problems,
            )
        effective = _effective_config(
            scenario, config, layers, reference=reference, hardware=hardware
        )
        derived = costs_from_model_config(effective, scenario.workload, layers)
    except Unsupported as error:  # 门 1/3：未迁移的服务
        return _refusal(
            scenario, effective, "unsupported",
            code=getattr(error, "code", None), message=str(error),
            stage="costs_from_model_config", layers=None,
        )
    except InvalidInput as error:  # 门 2/3：单位分辨率 / 图形状
        return _refusal(
            scenario, effective, "invalid",
            code=getattr(error, "code", None), message=str(error),
            stage="costs_from_model_config", layers=None,
        )
    except ValueError as error:  # 门 3/3：frozen dataclass 自己的不变式
        return _refusal(
            scenario, effective, "value",
            code=None, message=f"{error}（{type(error).__name__}）",
            stage="LayerSpec / dataclasses.replace", layers=None,
        )

    table = [
        {
            "operation_id": operation_id,
            "model_ns": derived.compute[operation_id].duration_ns,
            "declared_ns": scenario.costs.compute[operation_id].duration_ns,
            "agrees": derived.compute[operation_id].duration_ns
            == scenario.costs.compute[operation_id].duration_ns,
        }
        for operation_id in sorted(derived.compute)
    ]
    h2d = [
        {
            "tensor_id": tensor_id,
            # 字节数带上屏，是因为搬运时长那一行的算式是
            # ``延迟 + 字节 ÷ 带宽``——不给字节，那一行就只能看不能验。
            # 从张量自己推（**不是**抄常量）：``matvec`` 的权重是 48 字节，
            # 而 ``demo_window_vs_search.py:81`` 写死的 64 只对另外四个示例成立。
            "bytes": scenario.workload.tensor_by_id[tensor_id].storage_bytes,
            "model_ns": derived.h2d[tensor_id].duration_ns,
            "declared_ns": scenario.costs.h2d[tensor_id].duration_ns,
            "agrees": derived.h2d[tensor_id].duration_ns == scenario.costs.h2d[tensor_id].duration_ns,
        }
        for tensor_id in sorted(derived.h2d)
    ]
    return {
        "kind": "ok",
        "ok": True,
        "refusal_class": None,
        "refusal_label": None,
        "code": None,
        "message": None,
        "stage": "costs_from_model_config",
        "remedy": None,
        "toggles": _toggle_rows(effective, ""),
        "structural_problems": [],
        "layers": [
            {
                "operation_id": operation_id,
                "name": layer.name,
                "weight_bytes": layer.weight_bytes,
                "flops": layer.flops,
                "weight_tensor_id": _sole_weight(scenario.workload, operation_id)[0],
            }
            for operation_id, layer in sorted(layers.items())
        ],
        "table": table,
        "h2d": h2d,
        "all_agree": all(row["agrees"] for row in (*table, *h2d)),
        "hardware": {
            "gpu_effective_flops": effective.hardware.gpu_effective_flops,
            "h2d_bandwidth_bytes_per_s": effective.hardware.h2d_bandwidth_bytes_per_s,
            "h2d_latency_s": effective.hardware.h2d_latency_s,
            "vram_capacity_bytes": effective.hardware.vram_capacity_bytes,
        },
        "reference_used": reference,
        "_costs": derived,
    }


def _effective_config(
    scenario: Scenario,
    config: Any,
    layers: dict[str, Any],
    *,
    reference: bool,
    hardware: dict[str, Any] | None,
) -> Any:
    """这次求值实际用的 :class:`ModelConfig`。

    三条路，**分开**是关键：``config is None`` 不是「坏了」，是「没要」——现造一份除硬件外
    一切服务都关着的配置，时长来源才默认是 modeling。写成 ``replace(config, …)`` 会在
    ``None`` 上当场 ``TypeError``（实测踩到），而那个异常不在 ``bridge_table`` 的三个
    ``except`` 里，整页会死在一条 traceback 上。

    ``hardware`` 是**逐字段**覆盖，不是整份换掉：``HardwareSpec`` 还有十几个字段
    （``kv_bytes``、state 带宽…），屏上那三个旋钮只说得了三个；整份换会把其余的静默清零，
    而「清零」正好是门 1 的判据——等于用一次界面操作把拒绝清单伪造掉。
    """
    from dataclasses import replace

    from llm_infer_model.model import HardwareSpec, ModelConfig, PolicySpec

    override = {**REFERENCE_HARDWARE, **(hardware or {})}
    if config is None:
        # ``PolicySpec()`` 的默认值必须全都是「关」：guard 逐项点名的那几个字段一旦非零
        # 就会把这条默认路径也拒掉（参考硬件**不**豁免它们——实测：拿 toy 配置配参考硬件
        # 时仍然被 ``policy.static_gpu_layers`` 拒，因为换硬件换不掉策略）。
        # ``window_size`` 是唯一没有默认值的字段，而 ``costs_from_model_config`` **不读它**
        # （``layer_costs.py`` 的注释明写：窗口尺寸与逐层成本无关），所以填 1 就行。
        return ModelConfig(
            name=scenario.id,
            layers=tuple(layers.values()),
            hardware=HardwareSpec(
                vram_capacity_bytes=scenario.architecture.vram_capacity_bytes, **override
            ),
            policy=PolicySpec(window_size=1),
        )
    if reference:
        return replace(
            config,
            hardware=HardwareSpec(
                vram_capacity_bytes=scenario.architecture.vram_capacity_bytes, **override
            ),
        )
    if hardware:
        return replace(
            config,
            hardware=replace(
                config.hardware, **{key: float(value) for key, value in hardware.items()}
            ),
        )
    return config


def sourced_scenario(
    scenario: Scenario,
    bridge: dict[str, Any],
    *,
    edited: bool = False,
) -> tuple[Scenario, dict[str, Any]]:
    """按「时长来源」决定喂给内核的那一份 scenario。

    **这一步就是「联动」本身。** 内核算出来的 makespan、峰值、事件、以及时间线上每一根
    柱子的宽度，全部是从这份 ``Costs`` 推出来的；把它换成 :func:`bridge_table` 推出来的
    那一份，屏上的一切就跟着 modeling 走。``examples/*.json`` 一个字节都不动——换的只是
    内存里的这一份，所以 ``M0_VERIFICATION.md`` 里那些逐字节记录仍然对得上它自己记的
    scenario。

    两条**必须**退回声明值的路，都不是保守，是正确性：

    * ``edited``——用户在参数页手改过某个算子的 ``duration_ns``。那些改动手写在
      ``scenario.costs`` 上，而推导出来的 ``Costs`` 会把它整份换掉。不退回的话，「改了参数」
      和「改了没反应」在屏上长得一模一样，而表单还会照常回显用户填的数。
    * 推导失败（结构性 / 门 1 / 门 2 / 门 3 / 配置没载入）——没有东西可换。

    两条都带上原样原因，屏上要写出来：一个静默回退的来源面板，比一个空来源面板更坏。
    """
    from dataclasses import replace

    derived = bridge.pop("_costs", None)
    rows = [
        {"kind": "compute", "id": row["operation_id"], **row} for row in bridge.get("table", [])
    ] + [{"kind": "h2d", "id": row["tensor_id"], **row} for row in bridge.get("h2d", [])]
    disagreements = [
        {"kind": row["kind"], "id": row["id"],
         "model_ns": row["model_ns"], "declared_ns": row["declared_ns"]}
        for row in rows
        if not row["agrees"]
    ]

    block: dict[str, Any] = {
        "rows": rows,
        "disagreements": disagreements,
        "source": getattr(derived, "source", None) if derived is not None else None,
    }

    if edited:
        # 「手改过」在最前面：这一条与推导成功与否无关。推导失败时退回声明值本来就对，
        # 但退回的**理由**是「你改过」，不是「推不出来」——两者的补救方向完全相反。
        block.update(
            kind="declared", swapped=False, reason_class="edited",
            label="scenario 声明值（你在参数页手改过）",
            message="参数页里改过的算子时长写在 scenario 的成本表上，所以这一份用的是你的值，"
                    "不是 modeling 推的。想回到推导值，把那个框改回原数（或刷新页面）。",
            remedy="补救：把参数页里手改过的 ``duration_ns`` 改回去，来源会自动回到推导。",
        )
        return scenario, block

    if derived is None:
        block.update(
            kind="declared", swapped=False,
            reason_class=bridge.get("kind"),
            label=f"scenario 声明值（modeling 推不出来：{bridge.get('refusal_label') or bridge.get('kind')}）",
            message=bridge.get("message"),
            remedy=bridge.get("remedy"),
        )
        return scenario, block

    block.update(
        kind="model", swapped=True, reason_class=None,
        label="modeling 推导（llm_infer_model.layer_costs）",
        message=None,
        remedy=None,
    )
    return replace(scenario, costs=derived), block


def params_echo(scenario: Scenario, result: SearchResult) -> dict[str, Any]:
    """实际生效的旋钮值，原样回显。

    屏上的表单是**从这里**生成的，不是前端写死的一份。否则改了一个旋钮、表单却还显示
    旧值，用户会以为没生效。
    """
    return {
        "vram_capacity_bytes": scenario.architecture.vram_capacity_bytes,
        "runtime_reserved_bytes": scenario.architecture.runtime_reserved_bytes,
        "allocation_alignment_bytes": scenario.architecture.allocation_alignment_bytes,
        "allow_copy_compute_overlap": scenario.mapspace.allow_copy_compute_overlap,
        "allow_eviction": scenario.mapspace.allow_eviction,
        "copy_tensor_ids": list(scenario.mapspace.copy_tensor_ids),
        "max_expanded_states": result.max_expanded_states,
        "wall_time_limit_s": result.wall_time_limit_s,
        "compute_ns": {
            operation_id: cost.duration_ns
            for operation_id, cost in sorted(scenario.costs.compute.items())
        },
        "h2d_ns": {
            tensor_id: cost.duration_ns for tensor_id, cost in sorted(scenario.costs.h2d.items())
        },
    }
