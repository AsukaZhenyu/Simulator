"""注意力与 FFN 的**族语义**：一个族一个类，族内只调参数。

``m2/spec.py`` 已经把五种类型（``full_causal`` / ``sliding_window`` /
``recurrent_linear`` / ``dense`` / ``moe``）的词汇表和参数面定完了——参数名、类型、
上下界都在那里，多写一个键会被拒。本包补的是那些参数**算出来是什么**：

:mod:`~llm_infer_model.components.attention`
    读集、保留状态、执行峰值（``M2_LAYER_DESIGN.md`` §4 的三节）。
:mod:`~llm_infer_model.components.ffn`
    本次计算要用到哪些权重；MoE 缺路由结果时拒绝出账（§5 的揭示门）。
:mod:`~llm_infer_model.components.routing`
    那份路由结果由谁给——目前是确定性的随机 top-k。

**为什么值得单独一层**：这三种注意力与两种 FFN 的差别不在算子，在*状态表示*——
KV 每 token 增长、SWA 有界、递归状态是常数；稠密 FFN 每个字节都要参与，MoE 只有
本轮命中的专家参与。把这些差别收进一个类，``expand.py`` 就只需要问「这层是什么」，
而不是在展开里长出一串 ``if kind == ...``。

**这一层只说「计算涉及什么」，不说「那些字节放在哪」。** 权重是否常驻 VRAM、
什么时候预取、用完保留还是释放，属于 M3 的策略空间（``M3_GLOBAL_DESIGN.md:63``）；
内核按容量与策略决定它们从哪来。本层给出的账是**策略的定义域**，不是策略本身。

**依赖方向是单向的**：``components`` → ``m2.spec``（取词汇表），反向的
``m2`` → ``components`` 只能在**函数内部**做。模块级反向 import 会让
「先 import 谁」决定成败，``tests/test_components.py`` 里有一条子进程测试钉住
这一条。

典型用法::

    from llm_infer_model.components import attention_model, ffn_model

    attn = attention_model("full_causal", kv_bytes_per_token=4096)
    attn.state(history=3, tokens=1).state_bytes        # 16384

    ffn = ffn_model("dense", weight_bytes=45_895_680)
    ffn.plan(tokens=3).active_weight_bytes             # 45895680
"""

from __future__ import annotations

from .attention import (
    ATTENTION_MODELS,
    AttentionState,
    FullCausal,
    RecurrentLinear,
    SlidingWindow,
    attention_model,
)
from .ffn import (
    FFN_MODELS,
    DenseFFN,
    FFNState,
    MoEFFN,
    ffn_model,
)
from .routing import RandomTopK

__all__ = [
    "ATTENTION_MODELS",
    "FFN_MODELS",
    "AttentionState",
    "DenseFFN",
    "FFNState",
    "FullCausal",
    "MoEFFN",
    "RandomTopK",
    "RecurrentLinear",
    "SlidingWindow",
    "attention_model",
    "ffn_model",
]
