"""跑一次展开好的 layer 调用，并把内核状态读回成 M2 的 ``state_out``。

这一层不做模拟也不做校验：回放交给 ``tensor/engine.evaluate_mapping``，校验交给
``m2.expand``。它只负责最后一件事——把内核的**三本账**（DRAM / VRAM / SSD）读成
M2 的状态记录。这件事不能省：块现在在哪个域、hidden 落在哪，只有快照知道，而
「下次调用要接住的正是这些副本」是验收 §2 最后一句。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..tensor.engine import (
    STATUS_VALID,
    Action,
    EvaluationResult,
    StateSnapshot,
    evaluate_mapping,
)
from ..tensor.spec import COPY_READY, LOC_DRAM, LOC_SSD, LOC_VRAM, InvalidInput
from .expand import CompiledInvocation, plan_for
from .spec import HiddenOutRef, KvBlockRef, RequestState, StateOut, WeightStateRef, weight_id

__all__ = [
    "LayerInvocationResult",
    "run_invocation",
    "state_out_from_result",
]

# 读副本位置的顺序。它不是优先级，是一条**确定性**的书写顺序：同一组域永远写成
# 同一个序列，产物与指纹才不会随实现顺序漂移。
_LEDGER_ORDER = (LOC_DRAM, LOC_VRAM, LOC_SSD)


@dataclass(frozen=True)
class LayerInvocationResult:
    """一次调用的回放结果，连同它读出来的持久状态。

    ``state_out`` 在回放没走到目标时是 ``None``：一份**没完成**的调用的产物不是
    「状态没变」，而是「没有产物」——把它读成前者会让下游接着用一个不存在的状态。
    """

    compiled: CompiledInvocation
    evaluation: EvaluationResult
    state_out: StateOut | None

    @property
    def status(self) -> str:
        return self.evaluation.status

    @property
    def makespan_ns(self) -> int | None:
        return self.evaluation.makespan_ns

    @property
    def cost_provenance(self) -> str:
        """这张成本表在内核里的来源标签（如 ``m2:synthetic_layer``）。"""
        return self.compiled.costs_source


def _ready_locations(snapshot: StateSnapshot, tensor_id: str) -> tuple[str, ...]:
    """``tensor_id`` 在快照里哪些驻留域有一份 READY 的副本。

    按 :data:`_LEDGER_ORDER` 的顺序读三本账。只认 READY：``RESERVED_OUTPUT`` 与
    在飞的 ``RESERVED_COPY`` 都不是「这块数据现在在这里」。
    """
    return tuple(
        location
        for location in _LEDGER_ORDER
        if dict(snapshot.statuses(location)).get(tensor_id) == COPY_READY
    )


def state_out_from_result(
    compiled: CompiledInvocation, evaluation: EvaluationResult
) -> StateOut:
    """把回放的最终快照读成 ``state_out``。

    每条序列的块 = 这次调用开始时已有的块 **加上** 这次追加的块，位置从快照现读；
    hidden 用交付张量自己的名字。块的身份（id / 首位置 / token 数 / 字节数）来自
    ``compiled.new_blocks`` 与 ``state_in``，不重新推导——推导出来的身份会在
    「名字是对的、数字是错的」时把错误藏起来。
    """
    if evaluation.status != STATUS_VALID:
        raise InvalidInput(
            f"回放没有走到目标（status={evaluation.status}: {evaluation.reason}），"
            "没有可读的 state_out"
        )
    if not evaluation.states:
        raise InvalidInput("回放结果是 valid 却没有状态快照；内核不允许出现这种结果")
    snapshot = evaluation.states[-1]

    requests: list[RequestState] = []
    for request in compiled.invocation.requests:
        request_id = request.request_id
        state_in = compiled.state_in.for_request(request_id)
        blocks = [*state_in.blocks, *compiled.new_blocks[request_id]]

        recorded: list[KvBlockRef] = []
        for block in blocks:
            locations = _ready_locations(snapshot, block.id)
            if not locations:
                raise InvalidInput(
                    f"回放结束但块 {block.id!r} 在 dram/vram/ssd 里都不是 READY；"
                    "状态记录要求每个块至少有一个驻留域"
                )
            recorded.append(
                KvBlockRef(
                    id=block.id,
                    first_position=block.first_position,
                    tokens=block.tokens,
                    storage_bytes=block.storage_bytes,
                    locations=locations,
                )
            )
        if not compiled.new_blocks[request_id]:
            raise InvalidInput(
                f"request {request_id!r} 这次调用没有追加任何块；"
                "一次调用至少要物化一个块，否则 next_position 会静默不变"
            )
        requests.append(
            RequestState(
                request_id=request_id,
                next_position=state_in.next_position + request.tokens,
                blocks=tuple(recorded),
                # 版本号是这条序列的 KV 状态被扩展过几次：这次调用追加了块，所以
                # 加一。没有变化的序列不该换版本号（SWA 淘汰掉整块时会是那种情形）。
                version=state_in.version + 1,
                state_location=state_in.state_location,
            )
        )

    hidden: list[HiddenOutRef] = []
    for request in compiled.invocation.requests:
        tensor_id_value = compiled.hidden_out_ids[request.request_id]
        locations = _ready_locations(snapshot, tensor_id_value)
        if not locations:
            raise InvalidInput(
                f"回放结束但交付张量 {tensor_id_value!r} 在 dram/vram/ssd 里都不是 "
                "READY"
            )
        hidden.append(
            HiddenOutRef(
                request_id=request.request_id,
                first_position=request.position_start,
                tokens=request.tokens,
                storage_bytes=compiled.scenario.workload.tensor_by_id[
                    tensor_id_value
                ].storage_bytes,
                locations=locations,
            )
        )

    weights = []
    if compiled.layer.format_version == "m2-layer-0.6":
        for previous in compiled.state_in.weight_copies:
            locations = _ready_locations(snapshot, weight_id(compiled.layer.id, previous.id))
            if not locations:
                raise InvalidInput(f"weight {previous.id!r} has no persistent READY copy")
            weights.append(WeightStateRef(id=previous.id, version=previous.version,
                storage_bytes=previous.storage_bytes, locations=locations))
    return StateOut(
        layer=compiled.layer.id,
        version=max(state.version for state in requests),
        requests=tuple(requests),
        hidden=tuple(hidden),
        weight_copies=tuple(weights),
    )


def run_invocation(
    compiled: CompiledInvocation, *, actions: Sequence[Action] | None = None,
    placements: Mapping[str, str] | None = None,
) -> LayerInvocationResult:
    """Replay a baseline placement or an explicit search-produced action sequence.

    The kernel checks the same dependencies, capacities and goals for both. A
    failed mapping has no state_out. Supplying two different plan sources is an
    input error rather than an implicit precedence rule.
    """
    if actions is not None and placements is not None:
        raise InvalidInput("supply actions or placements, not both")
    plan = plan_for(compiled, placements=placements) if actions is None else tuple(actions)
    evaluation = evaluate_mapping(compiled.scenario, plan)
    return LayerInvocationResult(
        compiled=compiled,
        evaluation=evaluation,
        state_out=(
            state_out_from_result(compiled, evaluation)
            if evaluation.status == STATUS_VALID
            else None
        ),
    )
