"""FFN 族的语义：本次计算要用到哪些权重。

``m2/spec.py`` 定义词汇表（``dense`` / ``moe``）；本模块定义这两个词意味着什么。

两族只有一个轴：**这次计算需要哪些权重参与**。稠密 FFN 是 ``experts = 1``、
``top_k = 1`` 的退化情形，所以 ``DenseFFN`` 与 ``MoEFFN(experts=1, top_k=1, …)``
必须给出同样的账——``tests/test_components.py`` 把这条当不变量钉住。

**这里只回答「计算涉及什么」，不回答「那些字节放在哪」。** 权重是否常驻 VRAM、
什么时候预取、用完保留还是释放，是 M3 的搜索变量——``M3_GLOBAL_DESIGN.md:63``
原文字面列着「权重是否尝试**常驻** VRAM、已知下一层权重的预取距离 ``{0,1}``、
使用后保留或释放」。本模块给出的 ``total`` 与 ``active`` 之差，正是那些策略能动的
那块地方：**它是策略的定义域，不是策略本身**。说「这次只需要 3 份」是架构加
router 的事实；决定那 125 份在哪，是策略的事。

MoE 那一族的难点不在算，在**什么时候允许知道路由结果**：``MoEFFN.plan`` 拿不到
本轮每个专家分到的 token 数时直接 ``Unsupported``（``M2_LAYER_DESIGN.md`` §5：
「当前调用的 top-k 结果在 router 完成后才**对因果回放策略可见**」）。这不是防御性
编程，而是把揭示门写成结构：没有路由就没有专家参与计算的账，硬凑一个数等于假装
router 已经跑完。路由那份答案从哪来不归这里管——见
:mod:`~llm_infer_model.components.routing`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

from ..m2.spec import FFN_DENSE, FFN_MOE
from ..tensor.spec import InvalidInput, Unsupported
from ._require import reject_extra_keys, require_int, require_mapping

__all__ = [
    "FFN_MODELS",
    "DenseFFN",
    "FFNState",
    "MoEFFN",
    "ffn_model",
]


@dataclass(frozen=True)
class FFNState:
    """一次 FFN 调用**涉及的计算**：多少权重、哪些权重组。

    ``active_weight_bytes``
        本次计算真正要用到的权重字节。MoE 下没被选中的专家**这次根本不参与
        乘加**——不是「暂时没放进显存」，是这次不碰。
    ``total_weight_bytes``
        这一层 FFN 的全部权重。两者相等表示没有稀疏性：每个字节这次都要参与。
        不等表示有一部分这次根本不参与。
    ``active_weight_groups``
        那些权重组的**身份**，不是个数。策略要决定「哪些专家留在 VRAM、哪些赶
        出去」（``M3_GLOBAL_DESIGN.md:63`` ：「对命名的算子/权重组选择 CPU 或
        GPU 计算」），要的是名字，不是算术。MoE 下这些就是被选中的专家编号；
        稠密是 1 选 1 的退化情形，那一组就是 ``0``。

    升序是**规范形式**，不是风格：``{0:3, 5:1}`` 与 ``{5:1, 0:3}`` 是同一份路由，
    必须给出同一个状态对象，否则相等性会随字典插入顺序变化。
    """

    active_weight_bytes: int
    total_weight_bytes: int
    active_weight_groups: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("active_weight_bytes", "total_weight_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidInput(f"FFNState.{name} 必须是非负整数，得到 {value!r}")
        if self.active_weight_bytes > self.total_weight_bytes:
            raise InvalidInput(
                f"active_weight_bytes ({self.active_weight_bytes}) 不能超过 "
                f"total_weight_bytes ({self.total_weight_bytes})"
            )
        if not isinstance(self.active_weight_groups, tuple):
            raise InvalidInput(
                f"active_weight_groups 必须是元组，得到 {self.active_weight_groups!r}"
            )
        for group in self.active_weight_groups:
            require_int(group, "active_weight_groups 的权重组编号", minimum=0)
        if len(set(self.active_weight_groups)) != len(self.active_weight_groups):
            raise InvalidInput(
                f"active_weight_groups 有重复 {self.active_weight_groups}；"
                "同一份权重只算一次"
            )
        if tuple(sorted(self.active_weight_groups)) != self.active_weight_groups:
            raise InvalidInput(
                f"active_weight_groups 必须升序，得到 {self.active_weight_groups}；"
                "同一组权重集合只能有一个表示"
            )


@dataclass(frozen=True)
class DenseFFN:
    """稠密 FFN：每个 token 走同一组权重，没有可省的部分。"""

    weight_bytes: int
    kind: ClassVar[str] = FFN_DENSE

    def __post_init__(self) -> None:
        require_int(self.weight_bytes, "weight_bytes", minimum=1)

    def plan(self, *, tokens: int) -> FFNState:
        require_int(tokens, "tokens", minimum=1)
        # 稠密是 1 选 1：只有一个权重组，就是 0 号，全部字节这次都参与。
        return FFNState(self.weight_bytes, self.weight_bytes, (0,))


@dataclass(frozen=True)
class MoEFFN:
    """MoE FFN：只展开/运行被选中的 routed experts；shared experts 总是运行（§5）。

    参数与 ``m2/spec.py`` 的 ``_FFN_PARAM_KEYS[moe]`` 一一对应，没有多余字段。
    """

    experts: int
    top_k: int
    expert_weight_bytes: int
    shared_expert_bytes: int = 0
    kind: ClassVar[str] = FFN_MOE

    def __post_init__(self) -> None:
        require_int(self.experts, "experts", minimum=1)
        require_int(self.top_k, "top_k", minimum=1)
        if self.top_k > self.experts:
            raise InvalidInput(f"top_k ({self.top_k}) 不能超过 experts ({self.experts})")
        require_int(self.expert_weight_bytes, "expert_weight_bytes", minimum=1)
        require_int(self.shared_expert_bytes, "shared_expert_bytes", minimum=0)

    @property
    def total_weight_bytes(self) -> int:
        return self.experts * self.expert_weight_bytes + self.shared_expert_bytes

    def plan(self, *, tokens: int, expert_tokens: Mapping[int, int] | None = None) -> FFNState:
        """``expert_tokens``：本轮每个被选中的专家分到的 token 数。

        缺省是 ``None``，也就是「路由结果还没揭示」——那时拒绝出账（§5 的揭示门）。
        """
        require_int(tokens, "tokens", minimum=1)
        if expert_tokens is None:
            raise Unsupported(
                f"MoE 层（{self.experts} 选 {self.top_k}）没有本轮路由结果；"
                "哪些专家参与这次计算、各自分担多少 token，都取决于 router 的输出，"
                "而它在 router 跑完之前不可见（M2_LAYER_DESIGN.md §5）。请传入"
                "expert_tokens（MoEFFN 自己的 :mod:`~llm_infer_model.components."
                "routing` 或外部 trace 都行）。**不要**用均匀分布或 top_k×tokens 顶替。"
            )
        routed = require_mapping(expert_tokens, "expert_tokens")
        if not routed:
            raise InvalidInput("expert_tokens 不能为空：至少有一个专家被选中")
        seen: set[int] = set()
        total_tokens = 0
        for expert_id, count in routed.items():
            require_int(expert_id, "expert_tokens 的专家编号", minimum=0)
            if expert_id >= self.experts:
                raise InvalidInput(
                    f"expert_tokens 的专家编号 {expert_id} 超出范围 [0, {self.experts})"
                )
            if expert_id in seen:
                raise InvalidInput(f"expert_tokens 的专家编号重复：{expert_id}")
            seen.add(expert_id)
            count = require_int(count, f"expert_tokens[{expert_id}]", minimum=1)
            if count > tokens:
                raise InvalidInput(
                    f"expert_tokens[{expert_id}] = {count} 超过本轮 token 总数 {tokens}"
                )
            total_tokens += count
        # 本组件没有 token dropping：每个 token 恰好选择 top_k 个不同专家。
        # 只检查上界会让丢失的分派悄悄减少计算与活跃权重。
        expected = tokens * self.top_k
        if total_tokens != expected:
            raise InvalidInput(
                f"分派总量 {total_tokens} 必须等于 tokens×top_k = {expected}；"
                "每个 token 恰好选 top_k 个专家，当前模型不支持 token dropping"
            )
        # §5：「同一专家服务多个 token 时，权重副本只占一次空间」。
        return FFNState(
            active_weight_bytes=len(seen) * self.expert_weight_bytes + self.shared_expert_bytes,
            total_weight_bytes=self.total_weight_bytes,
            active_weight_groups=tuple(sorted(seen)),
        )


def _build_dense(weight_bytes: int | None, raw: Mapping[str, Any]) -> DenseFFN:
    reject_extra_keys(raw, (), f"{FFN_DENSE} 的 ffn_params")
    if weight_bytes is None:
        raise InvalidInput(
            "dense FFN 需要 layer 给出的 weight_bytes（本层 FFN 全部权重字节）；"
            "「没给」和「是 0」不是一回事，所以这里不设默认值"
        )
    return DenseFFN(weight_bytes=require_int(weight_bytes, "weight_bytes", minimum=1))


def _build_moe(weight_bytes: int | None, raw: Mapping[str, Any]) -> MoEFFN:
    # MoE 不看 weight_bytes：它的权重由 experts × expert_weight_bytes + shared 推出来，
    # 和递归注意力不看 kv_bytes_per_token 是同一回事。
    allowed = ("experts", "top_k", "expert_weight_bytes", "shared_expert_bytes")
    reject_extra_keys(raw, allowed, f"{FFN_MOE} 的 ffn_params")
    return MoEFFN(
        experts=require_int(raw.get("experts"), "experts", minimum=1),
        top_k=require_int(raw.get("top_k"), "top_k", minimum=1),
        expert_weight_bytes=require_int(
            raw.get("expert_weight_bytes"), "expert_weight_bytes", minimum=1
        ),
        shared_expert_bytes=require_int(
            raw.get("shared_expert_bytes", 0), "shared_expert_bytes", minimum=0
        ),
    )


# 与 attention.py 同一张表的写法：类别与构造器绑在一处，``FFN_MODELS`` 从它推出来。
# 两份并列的名单会各自漂移，而漂移只在某个 kind 上表现为空洞——注册表里有名字，
# 工厂却造不出来。
_FFN_BUILDERS: dict[str, tuple[type, Callable[[int | None, Mapping[str, Any]], Any]]] = {
    FFN_DENSE: (DenseFFN, _build_dense),
    FFN_MOE: (MoEFFN, _build_moe),
}

FFN_MODELS: dict[str, type] = {kind: model for kind, (model, _) in _FFN_BUILDERS.items()}


def ffn_model(
    kind: str, *, weight_bytes: int | None = None, params: Mapping[str, Any] | None = None
) -> DenseFFN | MoEFFN:
    """按 ``kind`` 与已解析的 ``params`` 造出对应的 FFN 族。

    ``weight_bytes`` 与 ``attention_model`` 的 ``kv_bytes_per_token`` 同一个位置：
    它来自 layer 自己的清单（FFN 的 weights），**不属于** ``ffn_params``——格式里
    ``_FFN_PARAM_KEYS[dense]`` 是空集，稠密 FFN 一个字都不用写。稠密族用它，
    MoE 族不用。
    """
    raw = {} if params is None else require_mapping(params, "ffn_params")
    entry = _FFN_BUILDERS.get(kind)
    if entry is None:
        raise Unsupported(
            f"未注册的 FFN 类型 {kind!r}；已注册 {sorted(_FFN_BUILDERS)}"
            "（M2_LAYER_DESIGN.md §5 的稠密/MoE 两节）"
        )
    return entry[1](weight_bytes, raw)
