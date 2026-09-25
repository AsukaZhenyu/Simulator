"""窗口策略：把旧 v0.8 的「窗口 K」规则接到张量内核上。

这是 ``modeling/llm_infer_model`` 那台张量模拟器接入旧窗口策略的第一步，**不**声称
复制了 v0.8 的全部 KV、控制和 staging 语义（``ALIGNMENT_IMPLEMENTATION.md`` §4）。

策略只决定**动作顺序**，不解释结果：合法性与时间推进一律问公共内核
（``legal_actions`` / ``transition``），makespan、峰值显存、搬运量一律由
``llm_infer_model.tensor.evaluate_mapping`` 产生。所以这里没有第二套规则——本模块
唯一新增的判断是「先做哪一件」，那是策略的本分，不是语义。

.. code-block:: python

    actions = window_mapping(scenario, window_size=2)
    result = evaluate_mapping(scenario, actions)

首版只支持**显式的线性链**：每个计算节点使用一份独有权重与前一个节点的输出，
首节点使用输入 ``x``。残差、分叉、共享权重这些结构明确拒绝（:class:`Unsupported`），
通用搜索与评估仍然照旧支持它们。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY_H2D,
    ACTION_EVICT,
    OP_DONE,
    OP_NOT_STARTED,
    Action,
    State,
    initial_state,
    is_goal,
    legal_actions,
    transition,
    used_vram_bytes,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT,
    COPY_READY,
    LOC_VRAM,
    ROLE_WEIGHT,
    InvalidInput,
    Scenario,
    Unsupported,
    Workload,
)

__all__ = [
    "ChainLink",
    "WindowPolicyFailed",
    "derive_chain",
    "window_mapping",
]

# 每个链环大约需要：1 次装载 + 1 次计算 + 至多 2 次释放 + 少量等待。给足余量后
# 仍然超限，说明策略陷入了某种没被 seen 抓到的发散，宁可报错也不要静默跑飞。
_ACTIONS_PER_LINK = 16


class WindowPolicyFailed(RuntimeError):
    """窗口策略在给定场景下走不下去。

    **这不等于「场景无解」。** 它只说明这条策略的限制（窗口 K，或它只会按链序推进）
    挡住了它；搜索可能仍然找得到可行计划，所以调用方不得把它当成 ``infeasible``
    （``ALIGNMENT_IMPLEMENTATION.md`` §4 最后一段）。
    """


@dataclass(frozen=True)
class ChainLink:
    """线性链上的一环：一个算子、它唯一的权重、以及进出的激活。"""

    operation_id: str
    weight_id: str
    activation_in: str
    activation_out: str


def derive_chain(workload: Workload) -> tuple[ChainLink, ...]:
    """从依赖关系读出链序，读不出就明确拒绝。

    顺序由「谁吃谁的输出」决定，**不**把 ``workload.operations`` 的数组序当层序——
    数组序是文件里的书写顺序，不是语义上的先后（``ALIGNMENT_IMPLEMENTATION.md`` §4）。
    """
    operations = workload.operation_by_id
    if not operations:
        raise Unsupported("workload 里没有任何算子，窗口策略无从下手")

    links: dict[str, ChainLink] = {}
    weight_owner: dict[str, str] = {}
    for operation_id in sorted(operations):
        operation = operations[operation_id]
        weights = [
            tensor_id
            for tensor_id in operation.inputs
            if workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT
        ]
        others = [
            tensor_id
            for tensor_id in operation.inputs
            if workload.tensor_by_id[tensor_id].role != ROLE_WEIGHT
        ]
        if len(weights) != 1 or len(others) != 1:
            raise Unsupported(
                f"算子 {operation_id} 的输入是 {list(operation.inputs)}：窗口策略要求每个"
                "计算节点恰好一份权重和一个激活输入。残差、纯逐元素算子（如 ADD）"
                "都不在这个形状里，请改用 search 求通用解"
            )
        weight_id = weights[0]
        if weight_id in weight_owner:
            raise Unsupported(
                f"权重 {weight_id} 同时被 {weight_owner[weight_id]} 与 {operation_id} 使用"
                "（共享权重）；共享权重在窗口里没有唯一的位置，本轮不支持"
            )
        weight_owner[weight_id] = operation_id
        links[operation_id] = ChainLink(
            operation_id=operation_id,
            weight_id=weight_id,
            activation_in=others[0],
            activation_out=operation.output,
        )

    for tensor_id, consumers in workload.consumers_of.items():
        if len(consumers) > 1:
            raise Unsupported(
                f"张量 {tensor_id} 被 {len(consumers)} 个算子读取（分叉）；窗口策略只处理"
                "一条链，请改用 search 求通用解"
            )

    produced_by = {link.activation_out: link.operation_id for link in links.values()}
    heads = [link for link in links.values() if link.activation_in not in produced_by]
    if len(heads) != 1:
        raise Unsupported(
            f"链上有 {len(heads)} 个起点，应当恰好 1 个（起点 = 激活输入不由任何算子产出）："
            f"{sorted(link.operation_id for link in heads)}"
        )

    order: list[ChainLink] = []
    current = heads[0]
    while current is not None and len(order) <= len(links):
        order.append(current)
        following = [
            link for link in links.values() if link.activation_in == current.activation_out
        ]
        if not following:
            current = None
        elif len(following) == 1:
            current = following[0]
        else:
            raise Unsupported(
                f"{current.activation_out} 被多个算子读取（分叉）："
                f"{sorted(link.operation_id for link in following)}"
            )

    on_chain = {link.operation_id for link in order}
    if on_chain != set(links):
        raise Unsupported(
            f"图不是一条链：链上只有 {sorted(on_chain)}，还有 "
            f"{sorted(set(links) - on_chain)} 不在任何一条从起点出发的路径上"
        )

    tail = order[-1]
    if tail.activation_out not in workload.outputs:
        raise Unsupported(
            f"链的末端输出是 {tail.activation_out!r}，但 workload.outputs 是 "
            f"{list(workload.outputs)}；窗口策略要求链的末端就是被请求的输出"
        )
    for tensor_id in workload.outputs:
        if tensor_id == tail.activation_out:
            continue
        tensor = workload.tensor_by_id[tensor_id]
        if LOC_VRAM not in tensor.initial_locations:
            raise Unsupported(
                f"输出 {tensor_id!r} 不是链的末端，初始也不在显存里；窗口策略不会为它"
                "单独安排装载，请改用 search"
            )

    return tuple(order)


def window_mapping(scenario: Scenario, window_size: int) -> tuple[Action, ...]:
    """按窗口 K 生成一条线性链的动作序列。

    ``window_size`` 限制**同时占用窗口的权重份数**：在途复制、已就绪、以及正在被
    计算读取的权重都计数（内核里的权重只要不是 ``ABSENT`` 就占着显存，所以这里就是
    数它）。激活与 workspace 不占窗口，但仍由内核计入显存容量。

    返回的动作序列可以直接交给 ``evaluate_mapping``；本函数不计算 makespan。

    抛出的异常分三类，调用方需要区别对待：

    * :class:`Unsupported` —— 这张图不在窗口策略的支持范围内（残差/分叉/共享权重）。
      这是**图的形状**问题，通用搜索不受影响。
    * :class:`InvalidInput` —— ``window_size`` 本身不合法。
    * :class:`WindowPolicyFailed` —— 图形状没问题、K 也合法，但这条策略走不下去。
      **不是**「场景无解」。
    """
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise InvalidInput(f"window_size 必须是正整数，得到 {window_size!r}")
    if window_size <= 0:
        raise InvalidInput(f"window_size 必须为正，得到 {window_size}")

    chain = derive_chain(scenario.workload)
    weight_ids = [link.weight_id for link in chain]

    state = initial_state(scenario)
    resident = sum(1 for tensor_id in weight_ids if state.copy(tensor_id) != COPY_ABSENT)
    if resident > window_size:
        raise WindowPolicyFailed(
            f"初始就有 {resident} 份权重在显存里，超过窗口 K={window_size}。"
            "窗口策略不因为 K 小而偷偷把 DRAM 权重算成已就绪，也不先把它们扔掉再重装——"
            "初始驻留由 scenario 决定；请调大 K，或改用 search"
        )

    actions: list[Action] = []
    seen: set[tuple[object, ...]] = set()
    limit = _ACTIONS_PER_LINK * len(chain) + _ACTIONS_PER_LINK

    while not is_goal(scenario, state):
        if len(actions) >= limit:
            raise WindowPolicyFailed(
                f"窗口策略用了 {len(actions)} 个动作仍未到达目标（上限 {limit}），判定为发散"
            )
        action = _choose(scenario, state, chain, window_size)
        marker = (state.key(), action.kind, action.target_id)
        if marker in seen:
            raise WindowPolicyFailed(
                f"窗口策略在 t={state.t} 重复了同一个动作 {action}；它绕回了已经走过的状态，"
                "不会自己走出去"
            )
        seen.add(marker)
        state = transition(scenario, state, action)
        actions.append(action)

    return tuple(actions)


def _choose(
    scenario: Scenario,
    state: State,
    chain: tuple[ChainLink, ...],
    window_size: int,
) -> Action:
    """在合法动作里挑一个：先释放、再计算、再装载、最后等待。

    只在 ``legal_actions`` 给的集合里挑，所以永远不会绕过场景的 overlap/eviction
    限制，也不会自己发明一条内核不认的规则。
    """
    legal = legal_actions(scenario, state)
    if not legal:
        raise WindowPolicyFailed(
            f"公共内核在 t={state.t} 说没有任何合法动作，而目标还没达到"
        )

    for action in _preferred_actions(scenario, state, chain, window_size):
        if action in legal:
            return action

    waiting = Action(ACTION_ADVANCE)
    if waiting in legal:
        return waiting
    # 走到这里说明「连等待都不合法」：没有在途事件，也没人能开工。最常见的原因是显存
    # 不够——该装的装不下，该算的还在等它。把占用与容量一起报出来，调用方才能一眼看出
    # 是容量问题而不是图形状问题；本函数仍然**不**替内核断定「无解」。
    raise WindowPolicyFailed(
        f"窗口策略在 t={state.t} 无事可做：没有合法动作，也没有在途事件可等。"
        f"当前显存占用 {used_vram_bytes(scenario, state)} B / 容量 "
        f"{scenario.architecture.vram_capacity_bytes} B（窗口 K={window_size}）。"
        "这仍然是这条策略走不下去，不是「场景无解」——请改用 search 拿结论"
    )


def _preferred_actions(
    scenario: Scenario,
    state: State,
    chain: tuple[ChainLink, ...],
    window_size: int,
) -> Iterator[Action]:
    """按优先级产出候选动作；真正的合法性仍由 ``legal_actions`` 判定。

    顺序体现三条规则：释放腾出的额度要能被**同一轮**的装载用上，所以释放排在装载
    前面；计算排在装载前面，因为计算不占 H2D 槽，先发出去就能与后面的搬运重叠。
    """
    yield from _release_candidates(state, chain)
    yield from _compute_candidates(state, chain)
    yield from _load_candidates(state, chain, window_size)


def _release_candidates(state: State, chain: tuple[ChainLink, ...]) -> Iterator[Action]:
    """已完成的环所对应的权重与中间值，按链序释放。

    只产出「算完了」的那些；「还有消费者」与「是目标输出」两类由内核拦下，所以这里
    不需要重复那条判断。
    """
    for link in chain:
        if state.op(link.operation_id) == OP_DONE:
            yield Action(ACTION_EVICT, link.weight_id)
    for link in chain:
        if state.op(link.operation_id) == OP_DONE:
            yield Action(ACTION_EVICT, link.activation_out)


def _compute_candidates(state: State, chain: tuple[ChainLink, ...]) -> Iterator[Action]:
    """链上最早那个还没开始的算子。

    链序是唯一的执行序（每环都要吃上一环的输出），所以不需要考虑别的候选。
    """
    for link in chain:
        if state.op(link.operation_id) == OP_NOT_STARTED:
            yield Action(ACTION_COMPUTE, link.operation_id)
            return


def _load_candidates(
    state: State, chain: tuple[ChainLink, ...], window_size: int
) -> Iterator[Action]:
    """下一项装载：先补最早那一环缺的激活，再考虑最靠前的、还没搬的权重。

    权重受 K 约束；激活不受（K 数的是权重份数）。窗口不允许时**不产出**这个候选，
    于是 ``_choose`` 会退到 ``ADVANCE`` 去等一个完成事件——等待正是窗口策略让出
    额度的手段。
    """
    pending = [link for link in chain if state.op(link.operation_id) == OP_NOT_STARTED]
    if not pending:
        return

    head = pending[0]
    if state.copy(head.activation_in) == COPY_ABSENT:
        yield Action(ACTION_COPY_H2D, head.activation_in)

    occupying = sum(
        1 for link in chain if state.copy(link.weight_id) != COPY_ABSENT
    )
    for link in pending:
        if state.copy(link.weight_id) == COPY_ABSENT:
            if occupying + 1 <= window_size:
                yield Action(ACTION_COPY_H2D, link.weight_id)
            return


def chain_weight_ids(scenario: Scenario) -> tuple[str, ...]:
    """链上权重张量的 id，按链序。给演示脚本核对窗口占用用。"""
    return tuple(link.weight_id for link in derive_chain(scenario.workload))


def chain_is_fully_resident(scenario: Scenario) -> bool:
    """三份权重是否一开始就都在显存里。演示脚本用它挑对照条件。"""
    state = initial_state(scenario)
    return all(state.copy(tensor_id) == COPY_READY for tensor_id in chain_weight_ids(scenario))
