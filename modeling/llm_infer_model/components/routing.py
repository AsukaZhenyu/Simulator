"""路由来源：这一轮里，每个 token 选中了哪些专家。

``MoEFFN.plan`` 要一份 ``{专家编号: token 数}`` 才肯出账（揭示门，见 ``ffn.py``）；
本模块负责把那份**答案**造出来。两者分开，是因为它们回答的是不同的问题：
router 决定选谁，FFN 的族语义决定「选了这些意味着多少权重参与这次计算」。

今天只有一个来源 :class:`RandomTopK`。以后接真实 trace 或数据集，是**再加一个
同样形状的类**（都提供 ``route(*, tokens, salt)``），而不是改这里——``salt`` 这个
参数存在的唯一理由，就是让「同一个调用」在任何来源下都有稳定的身份。

两条纪律，和成本表那边一致：

**确定性。** 展开器要求「同一个输入永远展开出同一个负载」（``m2/expand.py`` 开头），
真随机会让成本表、回放和断言全部对不上。所以 :class:`RandomTopK` 用自己那支
``random.Random``，不碰全局 ``random``；种子由 ``zlib.crc32`` 从 ``salt`` 推出来，
**不能用 ``hash()``**——Python 对字符串的 ``hash`` 每个进程都按 ``PYTHONHASHSEED``
加盐，同一个 salt 换个进程就是另一份路由，而这件事在单进程测试里看不出来。
``tests/test_components.py`` 用两个不同 ``PYTHONHASHSEED`` 的子进程钉住了这一条。

**来源可辨。** 随机路由是合成的，绝不能报成实测。``source`` 就是给这件事留的位置，
和 ``costs`` 的 ``provenance`` 同一个用意：合成的东西走到哪都要带着这个标记。
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import ClassVar

from ..tensor.spec import InvalidInput
from ._require import require_int, require_str

__all__ = [
    "RandomTopK",
]


def _seed(seed: int, salt: str) -> int:
    """跨进程稳定的 32 位种子。

    ``hash(salt)`` 在这里是错的：CPython 对 str 的 ``hash`` 每次启动都加随机盐，
    所以同一个 salt 在两个进程里会给出两个种子，路由也就跟着变了。``crc32`` 没有
    这个问题，且不引入依赖。
    """
    return (zlib.crc32(salt.encode("utf-8")) ^ (seed & 0xFFFFFFFF)) & 0xFFFFFFFF


@dataclass(frozen=True)
class RandomTopK:
    """目前的路由来源：**确定性的**随机 top-k。

    ``salt`` 由调用方给出，应当是这个调用的稳定身份（例如
    ``f"{layer_id}:{request_id}:{position_start}"``），这样同一份负载每次编译都
    路由到同一份结果。换 ``salt`` 就是换一次抽样，换 ``seed`` 就是换一条 trace。
    """

    experts: int
    top_k: int
    seed: int = 0
    source: ClassVar[str] = "synthetic_random"

    def __post_init__(self) -> None:
        require_int(self.experts, "experts", minimum=1)
        require_int(self.top_k, "top_k", minimum=1)
        if self.top_k > self.experts:
            raise InvalidInput(f"top_k ({self.top_k}) 不能超过 experts ({self.experts})")
        require_int(self.seed, "seed", minimum=0)

    def route(self, *, tokens: int, salt: str) -> dict[int, int]:
        """本轮每个被选中的专家分到多少 token。

        每个 token 选 ``top_k`` 个**互不相同**的专家（同一次前向里不会把同一个专家
        算两遍）；不同 token 之间会重复选中——那正是 ``MoEFFN.plan`` 里「同一专家
        服务多个 token 时权重只占一次」那条规则的来源。
        """
        require_int(tokens, "tokens", minimum=1)
        require_str(salt, "salt")
        rng = random.Random(_seed(self.seed, salt))
        counts: dict[int, int] = {}
        for _ in range(tokens):
            for expert in rng.sample(range(self.experts), self.top_k):
                counts[expert] = counts.get(expert, 0) + 1
        return counts
