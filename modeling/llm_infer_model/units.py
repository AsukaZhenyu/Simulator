"""Shared unit conversion, independent of the legacy layer-cost bridge."""

from __future__ import annotations

import math

from .tensor.spec import InvalidInput

__all__ = ["seconds_to_ns"]

_NS_PER_SECOND = 1_000_000_000


def seconds_to_ns(seconds: float, what: str) -> int:
    """把秒换成整数纳秒，拒绝让转换**悄悄地**产出 0 或负数。

    ``round`` 的静默是这里唯一的风险：一个 4e-10 秒的时长会变成 0 ns，而
    0 ns 的动作在核心里是合法的（不推进时间），于是它会变成一个「这个动作不花
    时间」的假事实，而不是一个错误。所以取整后非正也要拒绝。
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        raise InvalidInput(f"{what}: 不是数值（{seconds!r}）") from None
    if not math.isfinite(value):
        raise InvalidInput(f"{what}: 必须是有限值，得到 {value!r}")
    if value <= 0:
        raise InvalidInput(f"{what}: 必须为正，得到 {value!r} 秒")
    nanoseconds = round(value * _NS_PER_SECOND)
    if nanoseconds <= 0:
        raise InvalidInput(
            f"{what}: {value!r} 秒取整到整数纳秒后为 {nanoseconds}，无法表达；"
            "内核的时间单位是整数纳秒，不接受非正时长"
        )
    return nanoseconds
