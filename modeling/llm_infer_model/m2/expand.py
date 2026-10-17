"""把一次 layer 调用**确定性地**展开成内核能回放的 (workload, scenario)。

内核只认扁平的张量与算子（``tensor/spec.py`` 的 ``Workload``）：它没有 request、
batch、phase、token 位置这些概念，也不该有——``M2_LAYER_DESIGN.md`` §6 第 1 条说
清楚了，时间推进、资源互斥、容量与动作回放仍归 ``tensor/engine.py``，M2 只在它
旁边加**必需**的新语义。所以「一次 layer 调用」到这里被抹平成一张小图：request
之间靠命名区分，而不是靠内核理解 request。

展开规则（每一条都是为了「同一个输入永远展开出同一个负载」）：

* 每条序列一条**独立的链**，链上的任务按 layer 描述符的拓扑序排；
* 每条序列每个任务一个算子 ``op:{layer}:{request}:{task}``，产出张量用任务 id
  当阶段名 ``t:{layer}:{request}:{task}``；
* 本次调用追加的 KV 块按**绝对位置**命名 ``kv:{layer}:{request}:{position}``，
  并且是 **attention 任务的多余输出**——这正是内核要支持 ``extra_outputs`` 的
  唯一原因：验收 §2 里「读历史并追加新块」是**一个** 3 ms 任务，不是两个；
* attention 的输入里**列出这次要读的每一块历史 KV**。少列一块，makespan、字节数
  与 ``next_position`` 全都不变——这正是验收 :16 那条要求没有可观测后果的地方，
  所以这条依赖必须写在负载里，才能被断言抓住；
* 上一次调用留下的块在这一次是**叶子**：带 ``initial_locations``、没有生产者。
  于是「不许把 48 B 整份重写」不是一个纪律要求，而是**结构上做不到**——
  ``_validate_workload`` 禁止产出张量带 ``initial_locations``，
  ``_m1_check_compute`` 又拒绝给已经有副本的地址再写一次
  （``M2_LAYER_ACCEPTANCE.md`` §2 最后一句要求的正是这件事）；
* 权重按 ``w:{layer}:{weight}`` 命名，一次调用里只出现一次，被同层所有 request
  共享（一份字节，一份占用）。

命名是契约的一部分：块名里带绝对位置，所以第一次调用的输出和第二次调用的输入
天然对齐，``state_in_from_state_out`` 不需要做任何映射。

成本表提供逐任务 scratch，内核按正在运行的任务计账。旧层的非零
``workspace_bytes`` 校验串行 scratch 最大值；新版物理层可显式给出
``workspace_limit_bytes`` 上限。二者都不额外增加一份常驻分配。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..tensor.engine import ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, Action
from ..tensor.spec import (
    ALGORITHM_UNIFORM_COST,
    DTYPE_F32,
    DTYPE_I32,
    ROLE_INTERMEDIATE,
    ROLE_INPUT,
    ROLE_OUTPUT,
    ROLE_WEIGHT,
    SCHEMA_VERSION_M1,
    InvalidInput,
    Scenario,
    Unsupported,
    build_scenario,
)
from .costs import M2CostTable, MacroShape, expand_costs_document, expand_transfer_costs, parse_cost_table
from .physical import DTYPES, signature_for
from .full_physical import FullDensePhysical, signature_for as full_signature_for
from .cuda_physical import CudaDensePhysical, signature_for as cuda_signature_for
from .hybrid_physical import HybridDensePhysical, signature_for as hybrid_signature_for

from .spec import (
    STAGE_IN,
    Invocation,
    KvBlockRef,
    LayerDescriptor,
    RequestInput,
    RequestState,
    StateIn,
    _read_json,
    block_id,
    hidden_bytes,
    load_invocation,
    load_layer,
    op_id,
    tensor_id,
    validate_invocation,
    weight_id,
)

_COMPLETE_PHYSICAL = (FullDensePhysical, CudaDensePhysical, HybridDensePhysical)

__all__ = [
    "CompiledInvocation",
    "CompiledTask",
    "compile_invocation",
    "load_and_compile",
    "plan_for",
]

# 展开出来的负载 id 前缀。内核只把它当标签，但产物里要一眼看出「这是一次 M2
# layer 调用展开出来的」。
_M2_ID_PREFIX = "m2"

# attention 的宏任务名。这里写死一份常量，与 ``m2.spec.TASK_KINDS`` 里那一项同源。
_ATTENTION = "attention"


@dataclass(frozen=True)
class CompiledTask:
    """展开后的一条可调度任务，连同它查成本表要用的形状。"""

    request_id: str
    task_id: str
    kind: str
    operation_id: str
    device: str
    shape: MacroShape
    output_tensor: str
    weight_id: str | None
    depends_on: tuple[str, ...]
    expert_id: int | None = None
    work_items: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CompiledInvocation:
    """一次调用的完整展开：两份文档、已校验的 :class:`Scenario` 与任务序。

    ``scenario`` 是 ``build_scenario`` 的产物——它跑过文件入口的每一个解析器，
    所以这份展开不是「一份还没校验的草稿」。
    """

    layer: LayerDescriptor
    invocation: Invocation
    state_in: StateIn
    cost_table: M2CostTable
    scenario: Scenario
    costs_source: str
    workload_document: dict[str, Any]
    scenario_document: dict[str, Any]
    tasks: tuple[CompiledTask, ...]
    appended_positions: dict[str, tuple[int, ...]]
    new_blocks: dict[str, tuple[KvBlockRef, ...]]
    hidden_out_ids: dict[str, str]
    physical_boundary: dict[str, Any] | None = None
    expert_work_items: dict[str, tuple[tuple[str, int], ...]] | None = None

    @property
    def compute_ids(self) -> tuple[str, ...]:
        return tuple(task.operation_id for task in self.tasks)

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.id for blocks in self.new_blocks.values() for block in blocks)

    def tasks_of_request(self, request_id: str) -> tuple[CompiledTask, ...]:
        return tuple(task for task in self.tasks if task.request_id == request_id)

    def attention_task(self, request_id: str) -> CompiledTask:
        for task in self.tasks_of_request(request_id):
            if task.kind == _ATTENTION:
                return task
        raise InvalidInput(f"request {request_id!r} 的链上没有 attention 任务")


def _f32_layout(nbytes: int, what: str) -> tuple[list[int], list[int]]:
    """一个连续 F32 一维张量的 ``ne``/``nb``。

    内核只收连续 F32（``_require_f32_nbytes``），并且核对 ``storage_bytes`` 与
    ``ne``/``nb`` 推出来的一致——所以字节数不是 4 的倍数时要在这里拒，而不是让
    内核报一句「ne/nb 与 storage_bytes 不符」。
    """
    if nbytes % 4 != 0:
        raise Unsupported(
            f"{what}: {nbytes} B 不是 4 的倍数，表达不成连续的 F32 张量；"
            "M2 的形状必须是整数个 f32"
        )
    elements = nbytes // 4
    return [elements, 1, 1, 1], [4, nbytes, nbytes, nbytes]


def _tensor(
    tensor_id_value: str,
    role: str,
    nbytes: int,
    locations: tuple[str, ...] = (),
) -> dict[str, Any]:
    ne, nb = _f32_layout(nbytes, f"tensor {tensor_id_value!r}")
    entry: dict[str, Any] = {
        "id": tensor_id_value,
        "name": tensor_id_value,
        "role": role,
        "dtype": DTYPE_F32,
        "ne": ne,
        "nb": nb,
        "storage_bytes": nbytes,
    }
    if locations:
        entry["initial_locations"] = list(locations)
    return entry


def _sink_task(layer: LayerDescriptor) -> str:
    """任务图里唯一的末端任务——hidden_out 的产出者。"""
    depended_on = {dependency for task in layer.tasks for dependency in task.depends_on}
    sinks = [task.id for task in layer.tasks if task.id not in depended_on]
    if len(sinks) != 1:
        raise InvalidInput(
            f"layer {layer.id!r} 的任务图有 {len(sinks)} 个末端任务 "
            f"({', '.join(sorted(sinks)) or '一个都没有'})；交付的 hidden_out 是哪一个"
            "必须唯一，否则「这次调用的产出」有两个说法"
        )
    return sinks[0]


def _layer_tasks(
    layer: LayerDescriptor, invocation: Invocation, history: StateIn, device: str
) -> tuple[list[CompiledTask], dict[str, tuple[int, ...]]]:
    """每条序列的宏任务序列，按 layer 的拓扑序；顺带算出这次追加的位置。"""
    tasks: list[CompiledTask] = []
    appended: dict[str, tuple[int, ...]] = {}
    for request in invocation.requests:
        state = history.for_request(request.request_id)
        shape = MacroShape(
            phase=invocation.phase,
            batch=invocation.batch,
            tokens=request.tokens,
            # 这次调用开始时这条序列可读的历史长度。对每个宏任务用同一个定义，
            # 不按任务类型换含义（见 costs.MacroShape 的说明）。
            visible_history=state.next_position,
        )
        positions = tuple(
            range(request.position_start, request.position_start + request.tokens)
        )
        appended[request.request_id] = positions
        for task in layer.tasks:
            tasks.append(
                CompiledTask(
                    request_id=request.request_id,
                    task_id=task.id,
                    kind=task.kind,
                    operation_id=op_id(layer.id, request.request_id, task.id),
                    device=device,
                    shape=shape,
                    output_tensor=tensor_id(layer.id, request.request_id,
                        f"{task.id}.q" if isinstance(layer.physical, _COMPLETE_PHYSICAL) and task.kind == "projection"
                        else task.id),
                    weight_id=(
                        weight_id(layer.id, task.weight)
                        if task.weight is not None
                        else None
                    ),
                    depends_on=task.depends_on,
                )
            )
    return tasks, appended


def _expand_request(
    layer: LayerDescriptor,
    request: RequestInput,
    state: RequestState,
    request_tasks: tuple[CompiledTask, ...],
    appended: tuple[int, ...],
    sink: str,
    tensors: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], tuple[KvBlockRef, ...], str]:
    """一条序列的链。

    返回 ``(outputs, goal_locations, new_blocks, hidden_out)``；``goal_locations``
    只覆盖这条序列产出的东西，由调用方合并。
    """
    if isinstance(layer.physical, _COMPLETE_PHYSICAL):
        return _expand_full_physical_request(layer, request, state, request_tasks, appended, sink, tensors, operations)
    if layer.physical is not None:
        return _expand_physical_request(layer, request, state, request_tasks, appended, sink, tensors, operations)
    by_task = {task.task_id: task for task in request_tasks}
    hidden_in = tensor_id(layer.id, request.request_id, STAGE_IN)

    # 上一次调用留下的块：叶子，带 initial_locations、没有生产者。
    for block in state.blocks:
        tensors.append(_tensor(block.id, ROLE_INPUT, block.storage_bytes, block.locations))
    tensors.append(
        _tensor(
            hidden_in,
            ROLE_INPUT,
            hidden_bytes(layer, request.tokens),
            (request.input_location,),
        )
    )

    new_blocks: list[KvBlockRef] = []
    block_ids: list[str] = []
    goal_locations: dict[str, str] = {}

    for task in request_tasks:
        spec = layer.task_by_id(task.task_id)
        inputs: list[str] = []
        if task.weight_id is not None:
            inputs.append(task.weight_id)
        if spec.depends_on:
            inputs.extend(
                by_task[dependency].output_tensor for dependency in spec.depends_on
            )
        else:
            # 根任务的实参是本次调用的 hidden activation。
            inputs.append(hidden_in)
        if spec.kind == _ATTENTION:
            # 这次要读的每一块历史 KV 都在依赖里。
            inputs.extend(block.id for block in state.blocks)

        operation: dict[str, Any] = {
            "id": task.operation_id,
            "semantic_op": spec.kind,
            "ggml_op": f"M2_{spec.kind.upper()}",
            "unary_op": None,
            "op_params": [],
            "inputs": inputs,
            "output": task.output_tensor,
        }
        if spec.kind == _ATTENTION:
            # 「读历史并追加新块」是**一个**任务，所以三条事件区间 0-2/2-5/5-10
            # 里第二条同时包含读与写；拆成两个任务第三条区间就不成立了。
            names = [
                block_id(layer.id, request.request_id, position) for position in appended
            ]
            operation["extra_outputs"] = names
            for name, position in zip(names, appended):
                tensors.append(_tensor(name, ROLE_OUTPUT, layer.kv_bytes_per_token))
                new_blocks.append(
                    KvBlockRef(
                        id=name,
                        first_position=position,
                        tokens=1,
                        storage_bytes=layer.kv_bytes_per_token,
                        locations=(state.state_location,),
                    )
                )
                goal_locations[name] = state.state_location
            block_ids.extend(names)
        operations.append(operation)

        tensors.append(
            _tensor(
                task.output_tensor,
                ROLE_OUTPUT if task.task_id == sink else ROLE_INTERMEDIATE,
                hidden_bytes(layer, request.tokens),
            )
        )

    hidden_out = by_task[sink].output_tensor
    goal_locations[hidden_out] = request.output_location
    # hidden_out 排在这条序列的块前面：交付物在前，持久状态在后。
    return [hidden_out, *block_ids], goal_locations, tuple(new_blocks), hidden_out


def _physical_tensor(
    name: str, role: str, dtype: str, ne: tuple[int, ...], locations: tuple[str, ...] = (),
) -> dict[str, Any]:
    from ..tensor.spec import ggml_contiguous_layout
    nb, size = ggml_contiguous_layout(dtype, ne)
    entry = {"id": name, "name": name, "role": role, "dtype": dtype,
             "ne": list(ne), "nb": list(nb), "storage_bytes": size}
    if locations:
        entry["initial_locations"] = list(locations)
    return entry


def _expand_physical_request(
    layer: LayerDescriptor, request: RequestInput, state: RequestState,
    request_tasks: tuple[CompiledTask, ...], appended: tuple[int, ...], sink: str,
    tensors: list[dict[str, Any]], operations: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], tuple[KvBlockRef, ...], str]:
    """Materialized QKV inputs, two macro boundaries, logical immutable KV blocks.

    New K/V input tensors and appended cache slots are separate allocations.
    Cache blocks model logical dynamic capacity; no original padded cache or
    Q/gate aliasing is inferred. Runtime scratch stays in the per-op cost.
    """
    physical = layer.physical
    assert physical is not None
    for block in state.blocks:
        tensors.append(_tensor(block.id, ROLE_INPUT, block.storage_bytes, block.locations))
    input_ids: list[str] = []
    for port, ne in physical.input_ports(request.tokens).items():
        name = tensor_id(layer.id, request.request_id, f"in.{port}")
        input_ids.append(name)
        tensors.append(_physical_tensor(name, ROLE_INPUT, DTYPE_F32, ne, (request.input_location,)))
    by_kind = {task.kind: task for task in request_tasks}
    attention = by_kind[_ATTENTION]
    new_blocks: list[KvBlockRef] = []
    goals: dict[str, str] = {}
    for position in appended:
        name = block_id(layer.id, request.request_id, position)
        tensors.append(_tensor(name, ROLE_OUTPUT, layer.kv_bytes_per_token))
        new_blocks.append(KvBlockRef(name, position, 1, layer.kv_bytes_per_token, (state.state_location,)))
        goals[name] = state.state_location
    for task in request_tasks:
        weights = [weight_id(layer.id, name) for name in physical.weight_groups[task.kind].values()]
        inputs = ([*input_ids, *(block.id for block in state.blocks)] if task.kind == _ATTENTION
                  else [attention.output_tensor])
        operation = {"id": task.operation_id, "semantic_op": task.kind,
                     "ggml_op": f"M2_{task.kind.upper()}", "unary_op": None,
                     "op_params": [], "inputs": [*weights, *inputs], "output": task.output_tensor}
        if task.kind == _ATTENTION:
            operation["extra_outputs"] = [block.id for block in new_blocks]
        operations.append(operation)
        tensors.append(_physical_tensor(task.output_tensor,
            ROLE_OUTPUT if task.task_id == sink else ROLE_INTERMEDIATE,
            DTYPE_F32, (layer.hidden_size, request.tokens, 1, 1)))
    hidden_out = by_kind["dense_ffn"].output_tensor
    goals[hidden_out] = request.output_location
    return [hidden_out, *(block.id for block in new_blocks)], goals, tuple(new_blocks), hidden_out


def _expand_full_physical_request(
    layer: LayerDescriptor, request: RequestInput, state: RequestState,
    request_tasks: tuple[CompiledTask, ...], appended: tuple[int, ...], sink: str,
    tensors: list[dict[str, Any]], operations: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], tuple[KvBlockRef, ...], str]:
    """Hidden -> materialized post-RoPE QKV -> attention residual -> FFN residual.

    Position values are the absolute request positions, represented by an I32
    allocation. Projection's outputs and appended KV slots are separate F32
    allocations; the original hidden must remain available for attention's
    residual. No aliases, original cache padding or fused model kernels are
    silently inferred. The selected backend timing model composes three isolated macros.
    """
    physical = layer.physical
    assert isinstance(physical, _COMPLETE_PHYSICAL)
    for block in state.blocks:
        tensors.append(_tensor(block.id, ROLE_INPUT, block.storage_bytes, block.locations))
    input_ids: dict[str, str] = {}
    for port, ne in physical.input_ports(request.tokens).items():
        name = tensor_id(layer.id, request.request_id, f"in.{port}")
        input_ids[port] = name
        tensors.append(_physical_tensor(name, ROLE_INPUT,
            DTYPE_I32 if port == "positions" else DTYPE_F32, ne, (request.input_location,)))
    by_kind = {task.kind: task for task in request_tasks}
    projection = by_kind["projection"]
    attention = by_kind["attention"]
    projected: dict[str, str] = {}
    for port, ne in physical.projection_ports(request.tokens).items():
        name = tensor_id(layer.id, request.request_id, f"{projection.task_id}.{port}")
        projected[port] = name
        tensors.append(_physical_tensor(name, ROLE_INTERMEDIATE, DTYPE_F32, ne))
    new_blocks: list[KvBlockRef] = []
    goals: dict[str, str] = {}
    for position in appended:
        name = block_id(layer.id, request.request_id, position)
        tensors.append(_tensor(name, ROLE_OUTPUT, layer.kv_bytes_per_token))
        new_blocks.append(KvBlockRef(name, position, 1, layer.kv_bytes_per_token, (state.state_location,)))
        goals[name] = state.state_location
    for task in request_tasks:
        weights = [weight_id(layer.id, name) for name in physical.weight_groups[task.kind].values()]
        if task.kind == "projection":
            inputs = list(input_ids.values())
        elif task.kind == _ATTENTION:
            inputs = [*projected.values(), input_ids["hidden"], *(block.id for block in state.blocks)]
        else:
            inputs = [attention.output_tensor]
        operation = {"id": task.operation_id, "semantic_op": task.kind,
                     "ggml_op": f"M2_{task.kind.upper()}", "unary_op": None,
                     "op_params": [], "inputs": [*weights, *inputs], "output": task.output_tensor}
        if task.kind == "projection":
            operation["extra_outputs"] = [name for name in projected.values() if name != task.output_tensor]
        elif task.kind == _ATTENTION:
            operation["extra_outputs"] = [block.id for block in new_blocks]
        operations.append(operation)
        if task.kind != "projection":
            tensors.append(_physical_tensor(task.output_tensor,
                ROLE_OUTPUT if task.task_id == sink else ROLE_INTERMEDIATE,
                DTYPE_F32, (layer.hidden_size, request.tokens, 1, 1)))
    hidden_out = by_kind["dense_ffn"].output_tensor
    goals[hidden_out] = request.output_location
    return [hidden_out, *(block.id for block in new_blocks)], goals, tuple(new_blocks), hidden_out


def _check_derived_name_collisions(layer: LayerDescriptor) -> None:
    """任务的阶段名不能撞上 ``in``，否则 hidden 输入与某个任务的产出同名。"""
    for task in layer.tasks:
        reserved = {STAGE_IN}
        if layer.physical is not None:
            reserved.update(f"in.{port}" for port in layer.physical.input_ports(1))
        if task.id in reserved:
            raise InvalidInput(
                f"layer {layer.id!r}: 任务 id {STAGE_IN!r} 与输入 hidden 的阶段名同名；"
                "展开后它们是同一个张量名，必须换一个任务 id"
            )

    if isinstance(layer.physical, _COMPLETE_PHYSICAL):
        inputs = {f"in.{port}" for port in layer.physical.input_ports(1)}
        stages: list[str] = []
        for task in layer.tasks:
            if task.kind == "projection":
                stages.extend(f"{task.id}.{port}" for port in layer.physical.projection_ports(1))
            else:
                stages.append(task.id)
        if len(stages) != len(set(stages)) or set(stages) & inputs:
            raise InvalidInput("full physical task ids collide with generated stage port names")


def _check_workspace_agreement(
    layer: LayerDescriptor, tasks: list[CompiledTask], cost_table: M2CostTable,
    physical_signatures: dict[Any, dict[str, Any]],
    *, devices: tuple[str, ...] | None = None,
) -> None:
    """Validate a scratch declaration, never add it as a resident allocation.

    The supported single-device invocation runs one macro at a time. Its reusable scratch requirement is
    the maximum, not the sum. This is not the whole-layer memory peak: resident
    inputs, matrices, cache and live outputs are separately counted by engine.
    Legacy nonzero workspace_bytes declares that exact scratch maximum. New
    physical layers optionally set an upper bound via workspace_limit_bytes.
    """
    per_request: dict[str, int] = {}
    for task in tasks:
        for device in devices if devices is not None else (task.device,):
            cost = cost_table.lookup(task.kind, device, task.shape,
                physical_signature=physical_signatures.get((task.operation_id, device),
                                                          physical_signatures.get(task.operation_id)))
            assert cost.workspace_bytes is not None
            per_request[task.request_id] = max(per_request.get(task.request_id, 0), cost.workspace_bytes)
    for request_id, measured in sorted(per_request.items()):
        if layer.workspace_bytes and measured != layer.workspace_bytes:
            raise InvalidInput(
                f"layer {layer.id!r} 声明串行 scratch 最大值 {layer.workspace_bytes} B，"
                f"但 request {request_id!r} 的任务最大值是 {measured} B；顺序 scratch"
                "相加会重复计费，整个层的峰值由运行时存活张量另外计算"
            )
        if layer.workspace_limit_bytes is not None and measured > layer.workspace_limit_bytes:
            raise InvalidInput(f"layer {layer.id!r}: scratch {measured} B exceeds workspace_limit_bytes={layer.workspace_limit_bytes}")


def compile_invocation(
    layer: LayerDescriptor,
    invocation: Invocation,
    cost_table: M2CostTable,
    *,
    state_in: StateIn | None = None,
    where: str = "<memory>",
) -> CompiledInvocation:
    """把 (layer, invocation, 历史状态) 编译成内核能回放的负载与场景。

    全程在内存里完成，但走的是 ``build_scenario``：它会跑和文件入口一样的
    ``_parse_costs_m1`` / ``_parse_mapspace_m1`` / ``_parse_goal``，所以手工构造
    dataclass 会静默跳过的双向一致性与可达性检查在这里一条都不会少。
    """
    if layer.format_version == "m2-layer-0.6":
        from .moe import compile_moe_invocation
        return compile_moe_invocation(layer, invocation, cost_table, state_in=state_in, where=where)
    history = invocation.state_in if state_in is None else state_in
    validate_invocation(layer, invocation, history)
    if cost_table.validation_only and not invocation.allow_validation_only_costs:
        raise Unsupported("成本表 validation_only=true；仅显式 allow_validation_only_costs=true 的验证调用可执行，不能用于性能结论")
    _check_derived_name_collisions(layer)

    device = invocation.devices[0]
    tasks, appended = _layer_tasks(layer, invocation, history, device)
    physical_signature_for = (cuda_signature_for if isinstance(layer.physical, CudaDensePhysical)
                              else full_signature_for if isinstance(layer.physical, FullDensePhysical) else signature_for)
    hybrid = isinstance(layer.physical, HybridDensePhysical)
    physical_signatures = ({(task.operation_id, dev): hybrid_signature_for(layer, task.kind, task.shape, dev)
                            for task in tasks for dev in invocation.devices} if hybrid else
                           {task.operation_id: physical_signature_for(layer, task.kind, task.shape) for task in tasks}
                           if layer.physical is not None else {})
    _check_workspace_agreement(layer, tasks, cost_table, physical_signatures,
                               devices=invocation.devices if hybrid else None)
    sink = _sink_task(layer)

    tensors: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    outputs: list[str] = []
    goal_locations: dict[str, str] = {}
    new_blocks: dict[str, tuple[KvBlockRef, ...]] = {}
    hidden_out_ids: dict[str, str] = {}

    # 权重：一次调用里只出现一次，被所有 request 共享。
    for ref in layer.weights:
        if layer.physical is not None:
            tensors.append(_physical_tensor(weight_id(layer.id, ref.id), ROLE_WEIGHT,
                           DTYPES[ref.dtype], (*ref.shape, *(1 for _ in range(4-len(ref.shape)))), (ref.initial_location,)))
            continue
        tensors.append(
            _tensor(
                weight_id(layer.id, ref.id),
                ROLE_WEIGHT,
                ref.storage_bytes,
                (ref.initial_location,),
            )
        )

    for request in invocation.requests:
        request_id = request.request_id
        request_outputs, request_goals, blocks, hidden_out = _expand_request(
            layer,
            request,
            history.for_request(request_id),
            tuple(task for task in tasks if task.request_id == request_id),
            appended[request_id],
            sink,
            tensors,
            operations,
        )
        # Full causal state_out needs both retained history and this call's new
        # blocks. A historical leaf can also be a delivery goal: this adds no
        # producer or second allocation, but prevents a memory-saving mapping
        # from discarding history after attention and passing the final goal.
        for retained in history.for_request(request_id).blocks:
            request_outputs.append(retained.id)
            request_goals[retained.id] = history.for_request(request_id).state_location
        outputs.extend(request_outputs)
        goal_locations.update(request_goals)
        new_blocks[request_id] = blocks
        hidden_out_ids[request_id] = hidden_out

    if set(goal_locations) != set(outputs) or len(set(outputs)) != len(outputs):
        # 编译器自身的一致性：每个交付物恰好一个目标位置，每个张量只有一个生产者。
        raise InvalidInput(
            f"M2 展开内部不一致：{len(outputs)} 个交付物、"
            f"{len(goal_locations)} 个目标位置；重复或漏掉都会让内核的 goal 检查失败"
        )

    allowed_devices = {task.operation_id: list(invocation.devices) if hybrid else [task.device]
                       for task in tasks}
    allowed_transfers = {"dram_to_vram": [], "vram_to_dram": [],
                         "ssd_to_dram": [], "dram_to_ssd": []}
    transfer_provenances, transfer_document = frozenset(), {}
    if hybrid:
        inventory = {tensor["id"]: tensor["storage_bytes"] for tensor in tensors}
        for route in ("dram_to_vram", "vram_to_dram"):
            allowed_transfers[route] = list(inventory)
        transfer_provenances, transfer_document = expand_transfer_costs(cost_table, inventory, allowed_transfers)
    costs_source, costs_document = expand_costs_document(
        cost_table,
        [(task.operation_id, task.kind, dev, task.shape) for task in tasks
         for dev in allowed_devices[task.operation_id]],
        physical_signatures=physical_signatures, transfer_provenances=transfer_provenances,
    )
    costs_document["transfer"] = transfer_document

    workload_id = f"{_M2_ID_PREFIX}:{layer.id}:{invocation.id}"
    boundary = None
    if layer.physical is not None:
        boundary = {
            "scope": layer.physical.scope,
            "input_boundary": ("hidden_activation" if isinstance(layer.physical, _COMPLETE_PHYSICAL) else "post_positional_qkv"),
            "port_layout": layer.physical.port_layout,
            "kv_allocation": layer.physical.kv_allocation,
            "excluded_operations": list(layer.physical.excluded_operations),
            "capacity_semantics": "independent_resident_allocations_with_scenario_alignment_plus_running_macro_scratch",
            "original_model_kernel_reproduced": False,
            "complete_decoder_layer": isinstance(layer.physical, _COMPLETE_PHYSICAL),
            "cost_table_source": cost_table.source,
            "validation_only_costs": cost_table.validation_only,
            "cost_contract_hashes": ({task.operation_id: {dev: cost_table.lookup(task.kind, dev, task.shape,
                physical_signature=physical_signatures[(task.operation_id, dev)]).contract_hash
                for dev in invocation.devices} for task in tasks} if hybrid else
                {task.operation_id: cost_table.lookup(task.kind, task.device, task.shape,
                 physical_signature=physical_signatures[task.operation_id]).contract_hash for task in tasks}),
            "physical_signatures": ({task.operation_id: {dev: physical_signatures[(task.operation_id, dev)]
                for dev in invocation.devices} for task in tasks} if hybrid else physical_signatures),
        }
    if boundary is not None and isinstance(layer.physical, _COMPLETE_PHYSICAL):
        boundary.update({
            "complete_declared_decoder_block": True,
            "timing_composition": "three_isolated_macro_costs; joint_block_measurement_is_validation_only",
            "position_dtype": DTYPE_I32,
            "absolute_positions": {request.request_id: list(appended[request.request_id])
                                   for request in invocation.requests},
        })
    if boundary is not None and isinstance(layer.physical, CudaDensePhysical):
        boundary.update({
            "device": "gpu", "backend": f"CUDA{layer.physical.cuda_device}",
            "cuda_device": layer.physical.cuda_device,
            "cuda_graphs": layer.physical.cuda_graphs,
            "execution_scope": "single_cuda_device_resident_vram",
            "capacity_verification": "requires_known_macro_workspace; unknown_backend_scratch_is_rejected",
        })
    if boundary is not None and hybrid:
        boundary.update({
            "devices": list(invocation.devices), "cpu_threads": layer.physical.threads,
            "cuda_device": layer.physical.cuda_device, "cuda_graphs": layer.physical.cuda_graphs,
            "execution_scope": "explicit_dram_vram_copies_and_macro_device_selection",
            "transfer_cost_semantics": "exact_route_and_payload_bytes; synthetic_transfer",
            "capacity_verification": "declared_budget_and_known_macro_workspace_only",
        })
    workload_document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_M1,
        "id": workload_id,
        "origin": {
            "kind": layer.provenance,
            "ggml_version": None,
            "sample": f"{invocation.id} ({invocation.phase})" + (("; complete_declared_decoder_block; hidden_activation; logical_KV" if isinstance(layer.physical, _COMPLETE_PHYSICAL)
                    else "; partial_decoder_block; external_post_positional_QKV; logical_KV") if boundary else ""),
        },
        "comment": (
            f"M2 单层一次调用的展开。来源标签是 {layer.provenance!r}：宏任务语义与 "
            "extra_outputs 只在这个来源下被内核接受（tensor/spec.py 的 M2 来源门）。"
        ) + (f" Physical scope={boundary['scope']}; excluded={boundary['excluded_operations']}; capacity={boundary['capacity_semantics']}." if boundary else ""),
        "tensors": tensors,
        "operations": operations,
        "outputs": outputs,
    }

    scenario_document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_M1,
        "id": workload_id,
        "architecture": {
            "dram_capacity_bytes": invocation.hardware.dram_capacity_bytes,
            "vram_capacity_bytes": invocation.hardware.vram_capacity_bytes,
            "dram_runtime_reserved_bytes": invocation.hardware.dram_runtime_reserved_bytes,
            "vram_runtime_reserved_bytes": invocation.hardware.vram_runtime_reserved_bytes,
            "dram_alignment_bytes": invocation.hardware.dram_alignment_bytes,
            "vram_alignment_bytes": invocation.hardware.vram_alignment_bytes,
            "cpu_compute_slots": 1,
            "gpu_compute_slots": 1,
            "pcie_copy_slots": 1,
            "ssd_io_slots": 1,
        },
        "goal": {"output_locations": goal_locations},
        "costs": costs_document,
        "mapspace": {
            "allowed_devices": allowed_devices,
            "allowed_transfers": allowed_transfers,
            # 驱逐开关要留着：旧块在 attention 运行期间不可驱逐，这条读锁规则是
            # M2_LAYER_DESIGN.md §4「过期块只有在本次任务不再读取时才可释放」的
            # 同一条规则，关掉开关就看不见它了。
            "allow_eviction": True,
            "allow_copy_compute_overlap": True,
            "allow_cpu_gpu_overlap": True,
            "allow_recomputation": False,
        },
        "mapper": dict(
            invocation.mapper
            if invocation.mapper is not None
            else {
                "algorithm": ALGORITHM_UNIFORM_COST,
                "max_expanded_states": 0,
                "wall_time_limit_s": 0,
            }
        ),
        "comment": "M2 展开出来的场景：成本来自随包成本表，来源标签见 costs.source。",
    }

    scenario = build_scenario(workload_document, scenario_document, where=where)

    return CompiledInvocation(
        layer=layer,
        invocation=invocation,
        state_in=history,
        cost_table=cost_table,
        scenario=scenario,
        costs_source=costs_source,
        workload_document=workload_document,
        scenario_document=scenario_document,
        tasks=tuple(tasks),
        appended_positions=appended,
        new_blocks=new_blocks,
        hidden_out_ids=hidden_out_ids,
        physical_boundary=boundary,
    )


def plan_for(
    compiled: CompiledInvocation, *, placements: Mapping[str, str] | None = None,
) -> tuple[Action, ...]:
    """Build a deterministic serial plan; hybrid copies are explicit actions.

    A placement must name every macro. This baseline retains resident copies,
    serializes transfers and computation, and does not optimize eviction or
    prefetch. Capacity validity is decided by the kernel, never by this planner.
    Without a placement, the declared first device is the explicit baseline.
    """
    if compiled.layer.format_version == "m2-layer-0.6":
        from .moe import plan_moe
        return plan_moe(compiled, placements=placements)
    if not isinstance(compiled.layer.physical, HybridDensePhysical):
        if placements is not None:
            raise Unsupported("placements require m2-layer-0.5; existing physical versions remain fixed-device")
        return tuple(action for task in compiled.tasks for action in
                     (Action(ACTION_COMPUTE, task.operation_id, device=task.device), Action(ACTION_ADVANCE)))
    selected = ({task.operation_id: task.device for task in compiled.tasks}
                if placements is None else dict(placements))
    if set(selected) != set(compiled.compute_ids):
        raise InvalidInput("placements must name exactly every compiled operation_id")
    for operation_id, device in selected.items():
        if device not in compiled.scenario.mapspace.devices_for(operation_id):
            raise InvalidInput(f"placement {operation_id!r}: device {device!r} is not allowed")
    locations = {tensor.id: set(tensor.initial_locations)
                 for tensor in compiled.scenario.workload.tensors}
    actions: list[Action] = []

    def materialize(tensor_id: str, destination: str) -> None:
        if destination in locations[tensor_id]:
            return
        sources = [domain for domain in ("dram", "vram") if domain in locations[tensor_id]]
        if not sources:
            raise InvalidInput(f"serial plan has no produced or initial source for {tensor_id!r}")
        source = sources[0]
        actions.extend((Action(ACTION_COPY, tensor_id, src=source, dst=destination),
                        Action(ACTION_ADVANCE)))
        locations[tensor_id].add(destination)

    for task in compiled.tasks:
        device = selected[task.operation_id]
        domain = "dram" if device == "cpu" else "vram"
        operation = compiled.scenario.workload.operation_by_id[task.operation_id]
        for tensor_id in operation.inputs:
            materialize(tensor_id, domain)
        actions.extend((Action(ACTION_COMPUTE, task.operation_id, device=device), Action(ACTION_ADVANCE)))
        for tensor_id in (operation.output, *operation.extra_outputs):
            locations[tensor_id].add(domain)
    for tensor_id, destination in compiled.scenario_document["goal"]["output_locations"].items():
        materialize(tensor_id, destination)
    return tuple(actions)


def load_and_compile(
    invocation_path: str | Path, *, where: str | None = None
) -> CompiledInvocation:
    """从文件编译。``layer_file`` / ``costs_file`` 相对**调用文件**解析。"""
    target = Path(invocation_path)
    invocation = load_invocation(target)
    if invocation.layer_file is None:
        raise InvalidInput(
            f"{target.name}: 调用文件要给出 ``layer_file``；layer 描述符是可复用的"
            "独立文档，不内嵌在每次调用里"
        )
    layer = load_layer(target.parent / invocation.layer_file)
    if invocation.raw_cost_table is not None:
        table_raw: Mapping[str, Any] = invocation.raw_cost_table
        costs_where = f"{target.name}.costs"
    else:
        assert invocation.costs_file is not None  # 解析器保证恰好给了一处
        cost_path = target.parent / invocation.costs_file
        table_raw = _read_json(cost_path)
        costs_where = cost_path.name
    table = parse_cost_table(table_raw, where=costs_where)
    return compile_invocation(layer, invocation, table, where=where or target.name)
