"""组件层共用的入参校验。

三个模块都要做同一件事：**把「文档里没写」和「文档里写错了」分开**。写错一个
参数名或类型是文档错（``InvalidInput``）；名字合法、但本版本没有实现才是
``Unsupported``。这与 ``m2/spec.py`` 的 ``_require_one_of`` 是同一条界线——
两者的后续动作完全不同：前者要改文档，后者要等实现。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..tensor.spec import InvalidInput


def require_int(value: Any, what: str, *, minimum: int | None = None) -> int:
    """``bool`` 是 ``int`` 的子类，所以必须显式挡掉——``True`` 不是宽度 1。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{what} 必须是整数，得到 {value!r}")
    if minimum is not None and value < minimum:
        raise InvalidInput(f"{what} 必须 >= {minimum}，得到 {value}")
    return value


def require_non_negative(value: Any, what: str) -> int:
    return require_int(value, what, minimum=0)


def require_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidInput(f"{what} 必须是非空字符串，得到 {value!r}")
    return value


def require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidInput(f"{what} 必须是对象，得到 {value!r}")
    return value


def reject_extra_keys(mapping: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    """每种类型只认自己那几个参数名。

    多写一个键是**笔误**，不是「被忽略的额外信息」：``full_causal`` 上写
    ``window`` 会被这里拒掉，而不是安静地当成没有窗口。
    """
    extra = sorted(set(mapping) - set(allowed))
    if extra:
        raise InvalidInput(f"{what} 有本类型不认的键 {extra}；允许 {sorted(allowed)}")
