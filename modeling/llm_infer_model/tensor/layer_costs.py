"""把旧的逐层成本方法桥接成张量内核要的 :class:`Costs`。

``modeling`` 既有的成本模型是**闭式逐层**的（``model.py``）：

.. code-block:: text

    compute  = flops / gpu_effective_flops
    transfer = h2d_latency_s + weight_bytes / h2d_bandwidth_bytes_per_s

两者都允许被逐层的实测值 ``measured_compute_seconds`` /
``measured_transfer_seconds`` 覆盖。本模块只做一件事：把这个已经是闭式的公式
**逐项求值**，再按 ``round(seconds * 1_000_000_000)`` 换成张量内核的整数纳秒。

它**不是** ``ModelConfig`` → GGML 图的转换器。真实的一层里含多个算子，把整层的
成本摊给其中一个小矩阵乘（或平均摊给所有算子）会得到看起来合理、但没有物理含义
的数。所以对应关系必须由调用方**显式**给出：哪个算子是这个计算节点、哪份权重就是
这份权重张量。``costs_for_chain`` 会核对这份对应关系（ID、覆盖范围、权重字节数），
但不猜。

同理，本轮**不**迁移 KV、SSM 状态、每 token 控制开销、host staging、KV 分层存储
这些旧模型独有的服务。演示配置把它们全部关闭；``costs_from_model_config`` 这个便利
入口遇到非零值会**明确拒绝**，而不是静默丢弃或折成一个任意总开销——静默丢弃会让
算出来的时延偏小，而偏小的数字没有任何提示。

单位边界：旧 API 一律浮点**秒**，内核一律整数**纳秒**，转换只发生在本模块里。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..model import HardwareSpec, LayerSpec, ModelConfig
from .spec import (
    ROLE_WEIGHT,
    ComputeCost,
    Costs,
    H2DCost,
    InvalidInput,
    Scenario,
    Unsupported,
    Workload,
)

__all__ = [
    "SOURCE_LAYER_COSTS",
    "assert_weights_copyable",
    "costs_for_chain",
    "costs_from_layers",
    "costs_from_model_config",
    "seconds_to_ns",
]

# 写进 ``Costs.source``。M0 只把它当自由字符串，但下游（产物、绘图脚本）靠它
# 区分「合成固定值」与「来自层级成本模型」，所以这里用模块路径做稳定标识。
SOURCE_LAYER_COSTS = "llm_infer_model.layer_costs"

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


def costs_from_layers(
    hardware: HardwareSpec,
    *,
    compute_layers: Mapping[str, LayerSpec],
    weight_layers: Mapping[str, LayerSpec],
    workspace_bytes: int = 0,
    source: str = SOURCE_LAYER_COSTS,
) -> Costs:
    """按显式的一一对应关系求值，返回内核用的 :class:`Costs`。

    ``compute_layers`` 把**算子 ID** 映到它的 :class:`LayerSpec`；
    ``weight_layers`` 把**权重张量 ID** 映到同一个 :class:`LayerSpec`。同一层出现
    在两个字典里是刻意的——它是「这个算子的权重就是这份张量」这句话的载体。

    这里只调用旧 API 的两个方法，不重算公式：

    * ``layer.compute_seconds(hardware)``
    * ``layer.transfer_seconds(hardware)``

    因此「实测覆盖值优先、否则用 FLOPs/算力与字节数/带宽+延迟」这条既有规则
    原样生效，没有被复制成第二份实现。

    ``workspace_bytes`` 是每个算子的瞬时 workspace，由新场景明确提供（演示为 0）。
    """
    if not compute_layers:
        raise InvalidInput("compute_layers 为空：至少要有一个算子")
    if not weight_layers:
        raise InvalidInput("weight_layers 为空：至少要有一份权重")
    if workspace_bytes < 0:
        raise InvalidInput(f"workspace_bytes 不能为负，得到 {workspace_bytes}")

    compute: dict[str, ComputeCost] = {}
    for operation_id, layer in compute_layers.items():
        compute[operation_id] = ComputeCost(
            duration_ns=seconds_to_ns(
                layer.compute_seconds(hardware), f"算子 {operation_id}（{layer.name}）的计算时长"
            ),
            workspace_bytes=workspace_bytes,
        )

    h2d: dict[str, H2DCost] = {}
    for tensor_id, layer in weight_layers.items():
        h2d[tensor_id] = H2DCost(
            duration_ns=seconds_to_ns(
                layer.transfer_seconds(hardware), f"权重 {tensor_id}（{layer.name}）的搬运时长"
            )
        )

    return Costs(source=source, compute=compute, h2d=h2d)


def costs_for_chain(
    workload: Workload,
    hardware: HardwareSpec,
    layers_by_operation: Mapping[str, LayerSpec],
    *,
    workspace_bytes: int = 0,
    source: str = SOURCE_LAYER_COSTS,
) -> Costs:
    """从「一个计算节点 = 一层、它的权重 = 同一层」推出成本表，并核对这份对应关系。

    ``layers_by_operation`` 由调用方给出（算子 ID → :class:`LayerSpec`）；权重那一
    侧由本函数从算子的输入里读出来。核对四件事，任何一件对不上都直接报错而不是
    猜一个：

    1. **覆盖范围**：算子的键集合必须与 workload 完全一致，多一个少一个都拒绝。
    2. **权重归属**：每个算子恰好有一份 ``role == "weight"`` 的输入。零份（如 ``ADD``）
       或多份都拒绝。
    3. **不共享**：同一个权重张量不能出现在两个算子里。
    4. **字节数**：``LayerSpec.weight_bytes`` 必须等于该权重张量的
       ``storage_bytes``——这是防止「层填错了」最有效的一条，因为层与张量的字节数
       在玩具图上也很少巧合相等。

    第 2、3 条同时也是线性链的形状要求，所以本函数与
    :func:`tensor_mapping.policies.window_mapping` 接受同一类图。
    """
    known = set(workload.operation_by_id)
    given = set(layers_by_operation)
    if known != given:
        missing = sorted(known - given)
        extra = sorted(given - known)
        raise InvalidInput(
            "layers_by_operation 与 workload 的算子集合不一致："
            f"缺少 {missing or '（无）'}，多出 {extra or '（无）'}"
        )

    compute_layers: dict[str, LayerSpec] = {}
    weight_layers: dict[str, LayerSpec] = {}
    for operation_id in sorted(known):
        layer = layers_by_operation[operation_id]
        operation = workload.operation_by_id[operation_id]
        weights = [
            tensor_id
            for tensor_id in operation.inputs
            if workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT
        ]
        if len(weights) != 1:
            raise Unsupported(
                f"算子 {operation_id} 有 {len(weights)} 份权重输入；成本桥接本轮只支持"
                "「一个计算节点恰好一份权重」，不做任何平均摊派"
            )
        weight_id = weights[0]
        if weight_id in weight_layers:
            raise Unsupported(
                f"权重 {weight_id} 同时被 {operation_id} 与另一个算子使用（共享权重）；"
                "共享权重在窗口策略里没有唯一的位置，本轮不支持"
            )
        tensor = workload.tensor_by_id[weight_id]
        if layer.weight_bytes != tensor.storage_bytes:
            raise InvalidInput(
                f"{operation_id} → 层 {layer.name!r} 的 weight_bytes={layer.weight_bytes}，"
                f"但权重张量 {weight_id!r} 的 storage_bytes={tensor.storage_bytes}；"
                "这份对应关系对不上，不按数组下标猜"
            )
        compute_layers[operation_id] = layer
        weight_layers[weight_id] = layer

    weight_tensors = sorted(
        tensor.id for tensor in workload.tensors if tensor.role == ROLE_WEIGHT
    )
    if weight_tensors != sorted(weight_layers):
        raise InvalidInput(
            "workload 里还有没被任何算子使用的权重张量："
            f"{sorted(set(weight_tensors) - set(weight_layers)) or '（无）'}"
        )

    return costs_from_layers(
        hardware,
        compute_layers=compute_layers,
        weight_layers=weight_layers,
        workspace_bytes=workspace_bytes,
        source=source,
    )


def assert_weights_copyable(scenario: Scenario) -> None:
    """核对 scenario 的 mapspace 认得成本表里的每一份权重。

    ``Costs.h2d`` 里写了时延、``mapspace.copy_tensor_ids`` 里却没有的权重，是
    **永远不会被装载**的：算子在等一份搬不过来的权重，而报错会出现在很久之后、
    以一个与成本表无关的形状出现。这条检查让它在场景装配处就失败。
    """
    copyable = set(scenario.mapspace.copy_tensor_ids)
    missing = sorted(set(scenario.costs.h2d) - copyable)
    if missing:
        raise InvalidInput(
            f"成本表里有搬运时延、但 mapspace.copy_tensor_ids 里没有的权重：{missing}；"
            "它们永远装载不了"
        )


def costs_from_model_config(
    config: ModelConfig,
    workload: Workload,
    layers_by_operation: Mapping[str, LayerSpec],
    *,
    workspace_bytes: int = 0,
    source: str = SOURCE_LAYER_COSTS,
) -> Costs:
    """便利入口：用一份完整的旧配置，但要求额外服务全部关闭。

    **只取 ``config.hardware`` 与「额外服务已关闭」这个事实**；``config.layers``
    不参与映射——层的对应关系仍然必须由 ``layers_by_operation`` 显式给出（见模块
    docstring 里为什么不能自动推）。

    ``config.policy.window_size`` 有意**不**在这里读取：窗口大小属于
    :func:`tensor_mapping.policies.window_mapping` 的入参，成本表里没有它的位置。
    调用方（演示脚本）显式把它传过去，这样「用的是哪个 K」在调用点上看得见。
    """
    _assert_no_extra_services(config)
    return costs_for_chain(
        workload,
        config.hardware,
        layers_by_operation,
        workspace_bytes=workspace_bytes,
        source=source,
    )


def _assert_no_extra_services(config: ModelConfig) -> None:
    """把旧模型里本轮未迁移的服务逐项点名，一次性全部报出来。

    一次性而不是遇到第一个就抛，是因为这份清单是「还没搬过来」的待办，不是「你写错
    了」——把全部缺口一次说清楚，省得修一个跑一次。
    """
    hardware = config.hardware
    policy = config.policy
    state = config.state

    enabled: list[str] = []
    if hardware.cpu_effective_flops is not None:
        enabled.append("hardware.cpu_effective_flops（CPU 数学）")
    if hardware.global_bytes:
        enabled.append(f"hardware.global_bytes（{hardware.global_bytes} B）")
    if hardware.kv_bytes:
        enabled.append(f"hardware.kv_bytes（{hardware.kv_bytes} B）")
    if hardware.workspace_bytes:
        enabled.append(f"hardware.workspace_bytes（{hardware.workspace_bytes} B）")
    if hardware.activation_transfer_bytes:
        enabled.append(f"hardware.activation_transfer_bytes（{hardware.activation_transfer_bytes} B）")
    if hardware.host_staging_bandwidth_bytes_per_s is not None:
        enabled.append("host staging（host_staging_bandwidth_bytes_per_s）")
    if hardware.state_transfer_bandwidth_bytes_per_s is not None:
        enabled.append("状态往返搬运（state_transfer_bandwidth_bytes_per_s）")
    # 逐项点名而不是写 "kv_storage_*"：用户拿到的是一条能直接拿去搜索/改配置的
    # 字段路径，而不是一个还得自己展开的通配符。同组的其它条目也是这个写法。
    kv_storage = [
        name
        for name, value in (
            ("kv_storage_cached_bandwidth_bytes_per_s", hardware.kv_storage_cached_bandwidth_bytes_per_s),
            ("kv_storage_uncached_bandwidth_bytes_per_s", hardware.kv_storage_uncached_bandwidth_bytes_per_s),
            ("kv_storage_cache_bytes", hardware.kv_storage_cache_bytes),
        )
        if value
    ]
    if kv_storage:
        enabled.append("KV 分层存储（" + "、".join(kv_storage) + "）")
    if state is not None and (
        state.attention_kv_bytes_per_layer or state.recurrent_state_bytes_per_layer
    ):
        enabled.append(
            "state（每层 KV "
            f"{state.attention_kv_bytes_per_layer} B / SSM "
            f"{state.recurrent_state_bytes_per_layer} B）"
        )
    if policy.static_gpu_layers:
        enabled.append(f"policy.static_gpu_layers（{policy.static_gpu_layers} 层常驻）")
    if policy.effective_token_pipeline_overhead_s:
        enabled.append(
            "每 token 控制开销（policy.token_pipeline_overhead_s = "
            f"{policy.effective_token_pipeline_overhead_s!r} 秒）"
        )
    if policy.layer_scheduler_overhead_s:
        enabled.append(f"policy.layer_scheduler_overhead_s（{policy.layer_scheduler_overhead_s!r} 秒）")

    if enabled:
        raise Unsupported(
            "这份 ModelConfig 启用了张量内核本轮未迁移的服务："
            + "；".join(enabled)
            + "。这些项在 M0 的动作空间里没有对应动作，静默丢弃会让时延偏小且不留痕迹，"
            "因此这里明确拒绝。请先把它们关掉（演示配置见 "
            "mapping/examples/three_layer_chain.*），或等张量内核逐项接管后再用。"
        )
