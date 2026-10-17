"""注意力族的语义：读集、保留状态、执行峰值。

``m2/spec.py`` 定义词汇表（``full_causal`` / ``sliding_window`` /
``recurrent_linear``）并把每种类型的参数**解析**好；本模块定义这些参数
**意味着什么**。一个类一个族，族内差异全部走构造参数。

返回的四个量，正好是 ``M2_LAYER_DESIGN.md`` §4 要求分开的那几件事：

``read_positions``
    本次调用读到的位置数（含本次新增里被自己读到的部分）。位置型注意力才有；
    递归状态没有「位置」可言，恒为 0。
``kept_positions``
    调用结束后该 request 保留的位置数。
``state_bytes``
    调用结束后的持久状态字节（KV 块或递归状态）。
``peak_state_bytes``
    调用**期间**的峰值。它与 ``state_bytes`` 不相等是常态，而不是异常：SWA 在
    prefill 时旧块与新 chunk 同时占内存（§4 第 2 条明说不许拿最终
    ``W × KV/token`` 冒充执行峰值），递归状态在更新期间旧版与新版并存
    （§4 第 3 条）。把这两个数分开报，是因为合并成一个数就等于伪造峰值。

``read_positions`` 是本次所有 query 可见位置的并集，不能当作每个 query
的读长度或 attention 的乘加次数。SWA 的多 token prefill 会读到随后被窗口淘汰
的位置，因此该并集可以大于 ``kept_positions``；后者只描述调用结束后的保留状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

from ..m2.spec import (
    ATTENTION_FULL_CAUSAL,
    ATTENTION_RECURRENT_LINEAR,
    ATTENTION_SLIDING_WINDOW,
)
from ..tensor.spec import InvalidInput, Unsupported
from ._require import (
    reject_extra_keys,
    require_int,
    require_mapping,
    require_non_negative,
    require_str,
)

__all__ = [
    "ATTENTION_MODELS",
    "AttentionState",
    "FullCausal",
    "RecurrentLinear",
    "SlidingWindow",
    "attention_model",
]


@dataclass(frozen=True)
class AttentionState:
    """一次注意力调用**涉及的状态**：读多少、留多少、什么时候最高。

    这是「算这次 attention 需要同时在场的字节」，不是「缓存里放了什么」——见模块
    docstring 的四个量。
    """

    read_positions: int
    kept_positions: int
    state_bytes: int
    peak_state_bytes: int

    def __post_init__(self) -> None:
        for name in ("read_positions", "kept_positions", "state_bytes", "peak_state_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidInput(f"AttentionState.{name} 必须是非负整数，得到 {value!r}")
        if self.peak_state_bytes < self.state_bytes:
            raise InvalidInput(
                f"peak_state_bytes ({self.peak_state_bytes}) 不能小于 "
                f"state_bytes ({self.state_bytes})：峰值至少包含最终状态"
            )


def _span(history: int, tokens: int) -> int:
    """调用结束后该 request 的位置总数。"""
    return require_non_negative(history, "history") + require_non_negative(tokens, "tokens")


@dataclass(frozen=True)
class FullCausal:
    """全局因果注意力：位置 ``p`` 只能读该 request 中 ``<= p`` 的 KV（§4）。"""

    kv_bytes_per_token: int
    kind: ClassVar[str] = ATTENTION_FULL_CAUSAL

    def __post_init__(self) -> None:
        require_int(self.kv_bytes_per_token, "kv_bytes_per_token", minimum=1)

    def state(self, *, history: int, tokens: int) -> AttentionState:
        total = _span(history, tokens)
        used = self.kv_bytes_per_token * total
        # 旧块不因 append 整体复制（§4 第 1 条），所以执行期间与结束后同量。
        return AttentionState(total, total, used, used)


@dataclass(frozen=True)
class SlidingWindow:
    """SWA：窗口 ``W`` **包含当前 token**（§4 第 2 条）。

    位置 ``p`` 可读 ``[max(0, p-W+1), p]``；调用后只保留最后
    ``min(W, past+q)`` 个位置。
    """

    kv_bytes_per_token: int
    window: int
    kind: ClassVar[str] = ATTENTION_SLIDING_WINDOW

    def __post_init__(self) -> None:
        require_int(self.kv_bytes_per_token, "kv_bytes_per_token", minimum=1)
        require_int(self.window, "window", minimum=1)

    def state(self, *, history: int, tokens: int) -> AttentionState:
        total = _span(history, tokens)
        kept = min(self.window, total)
        # 本 chunk 的第一条 query 在 history，最后一条在 total-1。窗口并集
        # 是 [max(0, history-window+1), total-1]；空调用没有 query，也没有读取。
        read = total - max(0, history - self.window + 1) if tokens else 0
        # 执行期间调用前的常驻（至多 window 个）与本 chunk 的 token 都在场——
        # §4 第 2 条：「不能直接用最终 W × KV/token 冒充执行峰值」。
        peak_positions = min(self.window, history) + tokens
        return AttentionState(
            read_positions=read,
            kept_positions=kept,
            state_bytes=self.kv_bytes_per_token * kept,
            peak_state_bytes=self.kv_bytes_per_token * peak_positions,
        )


@dataclass(frozen=True)
class RecurrentLinear:
    """递归式线性注意力：固定大小状态，**不**生成随上下文增长的 KV（§4 第 3 条）。

    状态分量由变体**显式声明**（``name`` / ``bytes``），而不是由某个通用公式推
    出来——§4 第 3 条要求「特定变体无法给出状态大小时标为 unsupported，不能用
    通用 KV 公式代替」，显式声明正是这条的落法：声明了就有数，没声明就根本构不
    出这个对象。
    """

    variant: str
    components: tuple[tuple[str, int], ...]
    kind: ClassVar[str] = ATTENTION_RECURRENT_LINEAR

    def __post_init__(self) -> None:
        require_str(self.variant, "variant")
        if not self.components:
            raise InvalidInput("components 不能为空：递归状态必须至少声明一个分量")
        seen: set[str] = set()
        for name, size in self.components:
            require_str(name, "components[].name")
            if name in seen:
                raise InvalidInput(f"components[].name 重复：{name!r}")
            seen.add(name)
            require_int(size, f"components[{name!r}].bytes", minimum=1)

    @property
    def state_bytes(self) -> int:
        return sum(size for _, size in self.components)

    def state(self, *, history: int, tokens: int) -> AttentionState:
        # history/tokens 只被校验，不进结果：这正是这个族与另外两族的分界。
        _span(history, tokens)
        size = self.state_bytes
        # §4 第 3 条：版本化的旧/新状态在更新期间同时记内存，完成后释放旧版。
        return AttentionState(0, 0, size, 2 * size)


def _components_from_raw(raw: Any, what: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise InvalidInput(f"{what} 必须是数组，得到 {raw!r}")
    if not raw:
        raise InvalidInput(f"{what} 不能为空")
    parsed: list[tuple[str, int]] = []
    for i, entry in enumerate(raw):
        label = f"{what}[{i}]"
        if not isinstance(entry, Mapping):
            raise InvalidInput(f"{label} 必须是对象，得到 {entry!r}")
        reject_extra_keys(entry, ("name", "bytes"), label)
        parsed.append(
            (
                require_str(entry.get("name"), f"{label}.name"),
                require_int(entry.get("bytes"), f"{label}.bytes", minimum=1),
            )
        )
    return tuple(parsed)


def _build_full_causal(kv_bytes_per_token: int, raw: Mapping[str, Any]) -> FullCausal:
    reject_extra_keys(raw, (), f"{ATTENTION_FULL_CAUSAL} 的 attention_params")
    return FullCausal(kv_bytes_per_token=kv_bytes_per_token)


def _build_sliding_window(kv_bytes_per_token: int, raw: Mapping[str, Any]) -> SlidingWindow:
    reject_extra_keys(raw, ("window",), f"{ATTENTION_SLIDING_WINDOW} 的 attention_params")
    return SlidingWindow(
        kv_bytes_per_token=kv_bytes_per_token,
        window=require_int(raw.get("window"), "window", minimum=1),
    )


def _build_recurrent_linear(kv_bytes_per_token: int, raw: Mapping[str, Any]) -> RecurrentLinear:
    # 递归状态不用 kv_bytes_per_token：它没有「每 token 多少 KV」这个概念。
    reject_extra_keys(raw, ("variant", "components"), f"{ATTENTION_RECURRENT_LINEAR} 的 attention_params")
    return RecurrentLinear(
        variant=require_str(raw.get("variant"), "variant"),
        components=_components_from_raw(raw.get("components"), "components"),
    )


# 一张表同时管两件事：每种类型对应哪个类，以及怎么从参数造出来。``ATTENTION_MODELS``
# 是**从它推出来的**，不是并列的第二份名单——两份名单会各自漂移，而漂移的表现是
# 「注册表里有、工厂却造不出来」这种只在某个 kind 上出现的空洞。
_ATTENTION_BUILDERS: dict[str, tuple[type, Callable[[int, Mapping[str, Any]], Any]]] = {
    ATTENTION_FULL_CAUSAL: (FullCausal, _build_full_causal),
    ATTENTION_SLIDING_WINDOW: (SlidingWindow, _build_sliding_window),
    ATTENTION_RECURRENT_LINEAR: (RecurrentLinear, _build_recurrent_linear),
}

ATTENTION_MODELS: dict[str, type] = {
    kind: model for kind, (model, _) in _ATTENTION_BUILDERS.items()
}


def attention_model(
    kind: str, *, kv_bytes_per_token: int, params: Mapping[str, Any] | None = None
) -> FullCausal | SlidingWindow | RecurrentLinear:
    """按 ``kind`` 与已解析的 ``params`` 造出对应的注意力族。

    未注册的 ``kind`` 一律 ``Unsupported``——**绝不静默退回** ``full_causal``。
    静默退回会让一个 SWA 层被当成全局因果算完，而所有断言照样通过。
    """
    raw = {} if params is None else require_mapping(params, "attention_params")
    entry = _ATTENTION_BUILDERS.get(kind)
    if entry is None:
        raise Unsupported(
            f"未注册的注意力类型 {kind!r}；已注册 {sorted(_ATTENTION_BUILDERS)}"
            "（M2_LAYER_DESIGN.md §4 的三节各对应一个类）"
        )
    return entry[1](kv_bytes_per_token, raw)
