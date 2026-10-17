"""M2 单层调用：创作格式、输入校验与持久状态文档。

M2 的单位是**一个 decoder layer 的一次调用**（``M2_LAYER_DESIGN.md`` §1）。这里
定义它的输入长什么样、哪些输入非法，以及下一次调用要接住的持久状态怎么表达。

本模块**不**模拟任何东西：时间推进、资源互斥、容量与动作回放全在
``tensor/engine.py``（``M2_LAYER_DESIGN.md`` §6 第 1 条）。它做的是把一份 layer
描述符、一次调用和一份历史状态编译成内核能回放的对象之前的那道门——文档里没写的
东西一律拒绝，**不猜默认值**。

三层文档，各有自己的 ``format_version``：

``layer``
    稳定的 layer 身份与形状：hidden 维度/每 token 字节数、注意力与 FFN 类型及参数、
    权重清单、可调度任务的依赖 DAG 与 workspace。一份 layer 描述符可以被多次调用
    复用。
``invocation``
    一次调用：``prefill`` 或 ``decode``、B 条互异序列、每条序列的 ``position_start``
    与 token 数、hidden 输入及其初始位置、硬件容量与（可选的）成本表与设备集合。
    它内嵌 ``state_in``。
``costs``
    逐形状的宏任务成本表。物理格式只在 :mod:`llm_infer_model.m2.costs` 的
    ``parse_cost_table`` 里出现一次。

为什么用独立的 ``format_version`` 而不是第三个 ``schema_version``：``0.2`` 小图
（``Workload`` + ``Scenario``）和「一次 layer 调用」是两种东西，共用版本号会让
``load_and_validate`` 的返回类型变成一个联合体，而「不要让旧 0.2 小图被悄悄解释成
layer」（``M2_LAYER_DESIGN.md`` §2）正是这条边界要挡住的事。

命名约定（展开时必须确定，跨调用必须对齐；见 :mod:`llm_infer_model.m2.expand`）::

    kv:{layer}:{request}:{position}      一个 KV 块（新块按绝对位置命名）
    op:{layer}:{request}:{task}          一个宏任务
    t:{layer}:{request}:{stage}          宏任务产出的张量
    w:{layer}:{weight}                   一份权重
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .physical import DensePhysical, parse_physical, validate_physical_layer
from .full_physical import (
    FULL_FORMAT_VERSION, FullDensePhysical, parse_full_physical, validate_full_physical_layer,
)
from .cuda_physical import (
    CUDA_FORMAT_VERSION, CudaDensePhysical, parse_cuda_physical, validate_cuda_physical_layer,
)
from .hybrid_physical import (
    HYBRID_FORMAT_VERSION, HybridDensePhysical, parse_hybrid_physical, validate_hybrid_physical_layer,
)
from .moe_spec import (MOE_FORMAT_VERSION, STATE_CACHE_FORMAT_VERSION, RouteTraceEntry,
                       INFORMATION_MODES, parse_route_trace, validate_moe_layer,
                       validate_moe_invocation)
from typing import Any, Mapping, Sequence

from ..tensor.spec import (
    DOMAIN_OF_DEVICE,
    DEVICE_CPU,
    DEVICE_GPU,
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    ORIGIN_MEASURED_MACRO,
    ORIGIN_SYNTHETIC_LAYER,
    SEMANTIC_ATTENTION,
    SEMANTIC_COMBINE,
    SEMANTIC_DENSE_FFN,
    SEMANTIC_EXPERT_FFN,
    SEMANTIC_PROJECTION,
    SEMANTIC_ROUTER,
    SEMANTIC_STATE_UPDATE,
    InvalidInput,
    Unsupported,
)

__all__ = [
    "ATTENTION_FULL_CAUSAL",
    "ATTENTION_KINDS",
    "ATTENTION_RECURRENT_LINEAR",
    "ATTENTION_SLIDING_WINDOW",
    "FFN_DENSE",
    "FFN_KINDS",
    "FFN_MOE",
    "FORMAT_VERSION",
    "PHYSICAL_FORMAT_VERSION",
    "FULL_FORMAT_VERSION",
    "CUDA_FORMAT_VERSION",
    "HYBRID_FORMAT_VERSION",
    "MOE_FORMAT_VERSION",
    "STATE_CACHE_FORMAT_VERSION",
    "RouteTraceEntry",
    "WeightStateRef",
    "PHASE_DECODE",
    "PHASE_PREFILL",
    "PHASES",
    "PROVENANCE_MEASURED",
    "PROVENANCE_SYNTHETIC",
    "PROVENANCE_KINDS",
    "SLICE_ATTENTION",
    "SLICE_DEVICE",
    "SLICE_FFN",
    "STAGE_IN",
    "SUPPORTED_FORMAT_VERSIONS",
    "TASK_KINDS",
    "Hardware",
    "HiddenOutRef",
    "Invocation",
    "KvBlockRef",
    "LayerDescriptor",
    "RequestInput",
    "RequestState",
    "StateDocument",
    "StateIn",
    "StateOut",
    "TaskSpec",
    "WeightRef",
    "block_id",
    "hidden_bytes",
    "invocation_hidden_bytes",
    "load_invocation",
    "load_layer",
    "load_state_document",
    "op_id",
    "parse_invocation_document",
    "parse_layer_document",
    "parse_state_document",
    "parse_state_in",
    "state_in_from_state_out",
    "tensor_id",
    "to_state_document",
    "to_state_in_document",
    "validate_invocation",
    "weight_id",
]

# ---------------------------------------------------------------------------
# 词汇表
# ---------------------------------------------------------------------------

FORMAT_VERSION = "m2-layer-0.1"
PHYSICAL_FORMAT_VERSION = "m2-layer-0.2"
SUPPORTED_FORMAT_VERSIONS = (FORMAT_VERSION, PHYSICAL_FORMAT_VERSION, FULL_FORMAT_VERSION, CUDA_FORMAT_VERSION, HYBRID_FORMAT_VERSION, MOE_FORMAT_VERSION)
_PHYSICAL_VERSIONS = (PHYSICAL_FORMAT_VERSION, FULL_FORMAT_VERSION, CUDA_FORMAT_VERSION, HYBRID_FORMAT_VERSION)
_COMPLETE_VERSIONS = (FULL_FORMAT_VERSION, CUDA_FORMAT_VERSION, HYBRID_FORMAT_VERSION)

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASES = (PHASE_PREFILL, PHASE_DECODE)

# 两个正交维度，不是四个互斥类（M2_LAYER_DESIGN.md §1）。
ATTENTION_FULL_CAUSAL = "full_causal"
ATTENTION_SLIDING_WINDOW = "sliding_window"
ATTENTION_RECURRENT_LINEAR = "recurrent_linear"
ATTENTION_KINDS = (
    ATTENTION_FULL_CAUSAL,
    ATTENTION_SLIDING_WINDOW,
    ATTENTION_RECURRENT_LINEAR,
)

FFN_DENSE = "dense"
FFN_MOE = "moe"
FFN_KINDS = (FFN_DENSE, FFN_MOE)

# 宏任务名与内核的 ``semantic_op`` 是同一批字符串：那边多一个名字、这边少一个，
# 展开出来的负载就会被内核拒收，所以只允许有一份定义（tensor/spec.py:177-183）。
TASK_KINDS = (
    SEMANTIC_PROJECTION,
    SEMANTIC_ATTENTION,
    SEMANTIC_STATE_UPDATE,
    SEMANTIC_ROUTER,
    SEMANTIC_EXPERT_FFN,
    SEMANTIC_COMBINE,
    SEMANTIC_DENSE_FFN,
)

# 「这个数字是哪来的」。和内核的 ``origin.kind`` 用同一套词：合成成本永远不会被
# 报成实测（M2_LAYER_DESIGN.md §3、§6 末段）。
PROVENANCE_SYNTHETIC = ORIGIN_SYNTHETIC_LAYER
PROVENANCE_MEASURED = ORIGIN_MEASURED_MACRO
PROVENANCE_KINDS = (PROVENANCE_SYNTHETIC, PROVENANCE_MEASURED)

# 展开时每个 request 的输入张量阶段名。**只有输入用固定阶段名**：任务的产出用
# 任务 id 当阶段名（``t:{layer}:{request}:{task}``），所以这里不需要 proj/attn/out
# 这类名字——多一份平行的名字表，就多一处能和任务图对不上的地方。
STAGE_IN = "in"

_DEVICES = (DEVICE_CPU, DEVICE_GPU)
_LOCATIONS = (LOC_DRAM, LOC_VRAM)


def block_id(layer_id: str, request_id: str, position: int) -> str:
    """新 KV 块的张量名。按**绝对位置**命名，所以跨调用天然对齐。"""
    return f"kv:{layer_id}:{request_id}:{position}"


def op_id(layer_id: str, request_id: str, task_id: str) -> str:
    return f"op:{layer_id}:{request_id}:{task_id}"


def tensor_id(layer_id: str, request_id: str, stage: str) -> str:
    return f"t:{layer_id}:{request_id}:{stage}"


def weight_id(layer_id: str, weight: str) -> str:
    return f"w:{layer_id}:{weight}"


# ---------------------------------------------------------------------------
# 小校验器。刻意不复用 ``tensor.spec`` 的私有 ``_require_*``：那是内核的解析器
# 细节，M2 的报错要说出 M2 的字段名，而跨包引用下划线名字会让两边悄悄长在一起。
# ---------------------------------------------------------------------------


def _expect(condition: bool, message: str, exc: type[Exception] = InvalidInput) -> None:
    if not condition:
        raise exc(message)


def _require_int(value: Any, what: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{what} 必须是整数，得到 {value!r}")
    if value < minimum:
        raise InvalidInput(f"{what} 必须 >= {minimum}，得到 {value}")
    return value


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise InvalidInput(f"{what} 必须是字符串，得到 {value!r}")
    if not value:
        raise InvalidInput(f"{what} 必须非空")
    return value


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidInput(f"{what} 必须是 JSON 对象，得到 {value!r}")
    return value


def _require_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidInput(f"{what} 必须是 JSON 数组，得到 {value!r}")
    return value


def _reject_unknown_keys(block: Mapping[str, Any], allowed: set[str], what: str) -> None:
    """多出来的键一律拒绝：静默丢键会把一个笔误变成一条被关掉的约束。"""
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise InvalidInput(f"{what}: 未知字段 {', '.join(unknown)}")


def _check_format_version(raw: Any, where: str) -> str:
    version = _require_str(raw, f"{where}.format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise Unsupported(
            f"{where}.format_version 是 {version!r}；本版本实现 "
            f"{', '.join(repr(v) for v in SUPPORTED_FORMAT_VERSIONS)}"
        )
    return version


def _require_one_of(value: Any, what: str, allowed: Sequence[str]) -> str:
    """字段值必须是枚举里的一员。

    ``InvalidInput`` 而不是 ``Unsupported``：写错一个类型名是**文档错**。真正
    ``Unsupported`` 的是「名字合法、但本版本没实现」，那由
    :func:`validate_invocation` 判——两者的后续动作完全不同。
    """
    text = _require_str(value, what)
    if text not in allowed:
        raise InvalidInput(f"{what} 是 {text!r}；只认 {list(allowed)}")
    return text


def _require_location(value: Any, what: str) -> str:
    return _require_one_of(value, what, _LOCATIONS)


def _require_provenance(value: Any, what: str) -> str:
    return _require_one_of(value, what, PROVENANCE_KINDS)


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidInput(f"读不到 {path}：{exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInput(f"{path} 不是合法 JSON：{exc}") from exc


# ---------------------------------------------------------------------------
# layer 描述符
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightRef:
    """一份常驻权重。``storage_bytes`` 与 ``initial_location`` 都要显式给。"""

    id: str
    name: str
    storage_bytes: int
    initial_location: str = LOC_DRAM
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TaskSpec:
    """一个可调度的宏任务。

    ``depends_on`` 是**任务 id**（层内局部名，如 ``"attention"``），不是张量名：
    依赖 DAG 是层的结构，张量名是展开时按 request 生成的。
    ``weight`` 指向 :class:`WeightRef` 的 id，只在该任务真的读权重时给出。
    """

    id: str
    kind: str
    depends_on: tuple[str, ...] = ()
    weight: str | None = None


@dataclass(frozen=True)
class LayerDescriptor:
    """一层 decoder 的稳定描述：形状、类型与任务 DAG。

    ``provenance`` 是整层的来源标签（``synthetic_layer`` / ``measured_macro``），
    会被带进内核负载的 ``origin.kind``；内核只对这两个值放行宏任务语义
    （``tensor/spec.py`` 的 M2 来源门）。
    """

    id: str
    hidden_size: int
    hidden_bytes_per_token: int
    kv_bytes_per_token: int
    attention: str
    ffn: str
    weights: tuple[WeightRef, ...]
    tasks: tuple[TaskSpec, ...]
    provenance: str = PROVENANCE_SYNTHETIC
    attention_params: dict[str, Any] | None = None
    ffn_params: dict[str, Any] | None = None
    workspace_bytes: int = 0
    physical: DensePhysical | FullDensePhysical | CudaDensePhysical | HybridDensePhysical | None = None
    workspace_limit_bytes: int | None = None
    format_version: str = FORMAT_VERSION

    def task_by_id(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise InvalidInput(f"layer {self.id!r}: 没有任务 {task_id!r}")

    def tasks_of_kind(self, kind: str) -> tuple[TaskSpec, ...]:
        return tuple(task for task in self.tasks if task.kind == kind)

    def weight_by_id(self, weight: str) -> WeightRef:
        for ref in self.weights:
            if ref.id == weight:
                return ref
        raise InvalidInput(f"layer {self.id!r}: 没有权重 {weight!r}")


_LAYER_KEYS = {
    "format_version",
    "id",
    "provenance",
    "hidden_size",
    "hidden_bytes_per_token",
    "kv_bytes_per_token",
    "attention",
    "attention_params",
    "ffn",
    "ffn_params",
    "weights",
    "tasks",
    "workspace_bytes",
    "workspace_limit_bytes",
    "physical",
    "comment",
}
_WEIGHT_KEYS = {"id", "name", "storage_bytes", "initial_location", "dtype", "shape"}
_TASK_KEYS = {"id", "kind", "depends_on", "weight"}

# 每种注意力类型必须给出的参数。参数只在**这一种**类型下有名，所以每种的允许键
# 是分开的：``full_causal`` 写 ``window`` 是笔误，不是「被忽略的额外信息」。
_ATTENTION_PARAM_KEYS: dict[str, set[str]] = {
    ATTENTION_FULL_CAUSAL: set(),
    ATTENTION_SLIDING_WINDOW: {"window"},
    ATTENTION_RECURRENT_LINEAR: {"variant", "components"},
}
_FFN_PARAM_KEYS: dict[str, set[str]] = {
    FFN_DENSE: set(),
    FFN_MOE: {"experts", "top_k", "expert_weight_bytes", "shared_expert_bytes"},
}


def _parse_attention(raw_kind: Any, raw_params: Any, where: str) -> tuple[str, dict[str, Any]]:
    kind = _require_one_of(raw_kind, f"{where}.attention", ATTENTION_KINDS)
    params = _require_mapping(raw_params if raw_params is not None else {}, f"{where}.attention_params")
    label = f"{where}.attention_params"
    _reject_unknown_keys(params, allowed=_ATTENTION_PARAM_KEYS[kind], what=label)
    parsed: dict[str, Any] = {}
    if kind == ATTENTION_SLIDING_WINDOW:
        # 窗口**包含当前 token**（M2_LAYER_DESIGN.md §4）。
        parsed["window"] = _require_int(params.get("window"), f"{label}.window", minimum=1)
    elif kind == ATTENTION_RECURRENT_LINEAR:
        parsed["variant"] = _require_str(params.get("variant"), f"{label}.variant")
        components_raw = _require_list(params.get("components"), f"{label}.components")
        _expect(bool(components_raw), f"{label}.components 不能为空")
        components: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for i, entry in enumerate(components_raw):
            item_label = f"{label}.components[{i}]"
            item = _require_mapping(entry, item_label)
            _reject_unknown_keys(item, {"name", "bytes"}, item_label)
            name = _require_str(item.get("name"), f"{item_label}.name")
            _expect(name not in seen_names, f"{item_label}.name 重复：{name!r}")
            seen_names.add(name)
            components.append(
                {
                    "name": name,
                    "bytes": _require_int(item.get("bytes"), f"{item_label}.bytes", minimum=1),
                }
            )
        parsed["components"] = components
    return kind, parsed


def _parse_ffn(raw_kind: Any, raw_params: Any, where: str, *, version: str = FORMAT_VERSION) -> tuple[str, dict[str, Any]]:
    kind = _require_one_of(raw_kind, f"{where}.ffn", FFN_KINDS)
    params = _require_mapping(raw_params if raw_params is not None else {}, f"{where}.ffn_params")
    label = f"{where}.ffn_params"
    allowed = _FFN_PARAM_KEYS[kind]
    if version == MOE_FORMAT_VERSION and kind == FFN_MOE:
        allowed = allowed | {"expert_weights", "shared_expert_weight"}
    _reject_unknown_keys(params, allowed=allowed, what=label)
    parsed: dict[str, Any] = {}
    if kind == FFN_MOE:
        experts = _require_int(params.get("experts"), f"{label}.experts", minimum=1)
        top_k = _require_int(params.get("top_k"), f"{label}.top_k", minimum=1)
        _expect(top_k <= experts, f"{label}.top_k ({top_k}) 不能超过 experts ({experts})")
        parsed["experts"] = experts
        parsed["top_k"] = top_k
        parsed["expert_weight_bytes"] = _require_int(
            params.get("expert_weight_bytes"), f"{label}.expert_weight_bytes", minimum=1
        )
        if params.get("shared_expert_bytes") is not None:
            parsed["shared_expert_bytes"] = _require_int(
                params["shared_expert_bytes"], f"{label}.shared_expert_bytes", minimum=1
            )
        if version == MOE_FORMAT_VERSION:
            ids = _require_list(params.get("expert_weights"), f"{label}.expert_weights")
            parsed["expert_weights"] = [_require_str(v, f"{label}.expert_weights[{i}]") for i, v in enumerate(ids)]
            if "shared_expert_weight" in params:
                parsed["shared_expert_weight"] = _require_str(params["shared_expert_weight"], f"{label}.shared_expert_weight")
    return kind, parsed


def _parse_layer_tasks(raw: Any, weights: tuple[WeightRef, ...], where: str) -> tuple[TaskSpec, ...]:
    """任务 DAG：id 唯一、依赖存在、无环，且至少要有一条能走完的链。"""
    entries = _require_list(raw, f"{where}.tasks")
    _expect(bool(entries), f"{where}.tasks 不能为空")
    label = f"{where}.tasks"
    seen: dict[str, TaskSpec] = {}
    order: list[str] = []
    for i, entry in enumerate(entries):
        item_label = f"{label}[{i}]"
        item = _require_mapping(entry, item_label)
        _reject_unknown_keys(item, _TASK_KEYS, item_label)
        task_id = _require_str(item.get("id"), f"{item_label}.id")
        _expect(task_id not in seen, f"{item_label}.id 重复：{task_id!r}")
        kind = _require_one_of(item.get("kind"), f"{item_label}.kind", TASK_KINDS)
        depends_raw = _require_list(item.get("depends_on", []), f"{item_label}.depends_on")
        depends: list[str] = []
        for j, dependency in enumerate(depends_raw):
            dependency = _require_str(dependency, f"{item_label}.depends_on[{j}]")
            _expect(
                dependency not in depends,
                f"{item_label}.depends_on 列了 {dependency!r} 两次",
            )
            _expect(
                dependency != task_id,
                f"{item_label}.depends_on 依赖自己",
            )
            depends.append(dependency)
        weight = item.get("weight")
        if weight is not None:
            weight = _require_str(weight, f"{item_label}.weight")
            _expect(
                any(ref.id == weight for ref in weights),
                f"{item_label}.weight 指向未知权重 {weight!r}",
            )
        seen[task_id] = TaskSpec(
            id=task_id, kind=kind, depends_on=tuple(depends), weight=weight
        )
        order.append(task_id)

    for task in seen.values():
        for dependency in task.depends_on:
            _expect(
                dependency in seen,
                f"{label}[{task.id!r}].depends_on 指向未知任务 {dependency!r}",
            )

    # Kahn 拓扑排序，同层按文档顺序取：展开出来的计划顺序必须是文档决定的，
    # 不能随 dict 的迭代顺序变。排不完就是有环。
    ordered: list[TaskSpec] = []
    remaining = list(order)
    while remaining:
        ready = [
            task_id
            for task_id in remaining
            if all(dep in {task.id for task in ordered} for dep in seen[task_id].depends_on)
        ]
        if not ready:
            raise InvalidInput(f"{label}: 依赖图有环，环上还剩 {', '.join(remaining)}")
        for task_id in ready:
            ordered.append(seen[task_id])
            remaining.remove(task_id)
    return tuple(ordered)


def parse_layer_document(raw: Any, *, where: str = "<layer>") -> LayerDescriptor:
    block = _require_mapping(raw, where)
    _reject_unknown_keys(block, _LAYER_KEYS, where)
    version = _check_format_version(block.get("format_version"), where)
    physical = None
    if version in _PHYSICAL_VERSIONS:
        if "workspace_bytes" in block:
            raise InvalidInput(f"{where}: physical layers use workspace_limit_bytes, not legacy workspace_bytes")
        physical = (parse_hybrid_physical(block.get("physical")) if version == HYBRID_FORMAT_VERSION
                    else parse_cuda_physical(block.get("physical")) if version == CUDA_FORMAT_VERSION
                    else parse_full_physical(block.get("physical")) if version == FULL_FORMAT_VERSION
                    else parse_physical(block.get("physical")))
    elif "physical" in block or "workspace_limit_bytes" in block:
        raise InvalidInput(f"{where}: physical and workspace_limit_bytes require {PHYSICAL_FORMAT_VERSION}")

    layer_id = _require_str(block.get("id"), f"{where}.id")
    provenance = _require_provenance(
        block.get("provenance", PROVENANCE_SYNTHETIC), f"{where}.provenance"
    )

    if version == MOE_FORMAT_VERSION and block.get("provenance") != PROVENANCE_SYNTHETIC:
        raise InvalidInput(f"{where}.provenance must explicitly be synthetic_layer for MoE 0.6")

    weights_raw = _require_list(block.get("weights", []), f"{where}.weights")
    weights: list[WeightRef] = []
    seen_weights: set[str] = set()
    for i, entry in enumerate(weights_raw):
        item_label = f"{where}.weights[{i}]"
        item = _require_mapping(entry, item_label)
        _reject_unknown_keys(item, _WEIGHT_KEYS, item_label)
        weight_name = _require_str(item.get("id"), f"{item_label}.id")
        _expect(weight_name not in seen_weights, f"{item_label}.id 重复：{weight_name!r}")
        seen_weights.add(weight_name)
        dtype = item.get("dtype")
        shape = None
        if physical is not None:
            dtype = _require_str(dtype, f"{item_label}.dtype")
            dimensions = _require_list(item.get("shape"), f"{item_label}.shape")
            if version in _COMPLETE_VERSIONS:
                _expect(len(dimensions) in (1, 2), f"{item_label}.shape requires a norm vector or reduction-first matrix")
            else:
                _expect(len(dimensions) == 2, f"{item_label}.shape requires reduction-first [input, output]")
            shape = tuple(_require_int(v, f"{item_label}.shape[{j}]", minimum=1) for j, v in enumerate(dimensions))
        elif "dtype" in item or "shape" in item:
            raise InvalidInput(f"{item_label}: physical matrix metadata requires {PHYSICAL_FORMAT_VERSION}")
        if version in (HYBRID_FORMAT_VERSION, MOE_FORMAT_VERSION) and "initial_location" not in item:
            raise InvalidInput(f"{item_label}.initial_location must be explicit for hybrid execution")
        weights.append(
            WeightRef(
                id=weight_name,
                name=_require_str(item.get("name", weight_name), f"{item_label}.name"),
                storage_bytes=_require_int(
                    item.get("storage_bytes"), f"{item_label}.storage_bytes", minimum=1
                ),
                dtype=dtype,
                shape=shape,
                initial_location=_require_one_of(
                    item.get("initial_location", LOC_DRAM),
                    f"{item_label}.initial_location",
                    (LOC_DRAM, LOC_VRAM, LOC_SSD) if version == MOE_FORMAT_VERSION else _LOCATIONS,
                ),
            )
        )

    attention, attention_params = _parse_attention(
        block.get("attention"), block.get("attention_params"), where
    )
    ffn, ffn_params = _parse_ffn(block.get("ffn"), block.get("ffn_params"), where, version=version)

    layer = LayerDescriptor(
        id=layer_id,
        hidden_size=_require_int(block.get("hidden_size"), f"{where}.hidden_size", minimum=1),
        hidden_bytes_per_token=_require_int(
            block.get("hidden_bytes_per_token"),
            f"{where}.hidden_bytes_per_token",
            minimum=1,
        ),
        kv_bytes_per_token=_require_int(
            block.get("kv_bytes_per_token"), f"{where}.kv_bytes_per_token", minimum=1
        ),
        attention=attention,
        ffn=ffn,
        weights=tuple(weights),
        tasks=_parse_layer_tasks(block.get("tasks"), tuple(weights), where),
        provenance=provenance,
        attention_params=attention_params,
        ffn_params=ffn_params,
        workspace_bytes=_require_int(
            block.get("workspace_bytes", 0), f"{where}.workspace_bytes"
        ),
        physical=physical,
        format_version=version,
        workspace_limit_bytes=(_require_int(block["workspace_limit_bytes"], f"{where}.workspace_limit_bytes")
                               if "workspace_limit_bytes" in block else None),
    )
    if physical is not None:
        (validate_hybrid_physical_layer if version == HYBRID_FORMAT_VERSION
         else validate_cuda_physical_layer if version == CUDA_FORMAT_VERSION
         else validate_full_physical_layer if version == FULL_FORMAT_VERSION else validate_physical_layer)(layer)
    if version == MOE_FORMAT_VERSION:
        validate_moe_layer(layer)
    return layer


def load_layer(path: str | Path) -> LayerDescriptor:
    target = Path(path)
    return parse_layer_document(_read_json(target), where=target.name)


# ---------------------------------------------------------------------------
# 状态：state_in 与 state_out 是同一种东西
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KvBlockRef:
    """一个不可变 KV 块（``M2_LAYER_DESIGN.md`` §4）。

    ``first_position`` 是**绝对** token 位置，不是块内偏移；``locations`` 是它当前
    的驻留副本（顺序即 ``dram``/``vram``/``ssd`` 的优先级，见
    :func:`llm_infer_model.m2.runner.state_out_from_result`）。
    """

    id: str
    first_position: int
    tokens: int
    storage_bytes: int
    locations: tuple[str, ...]

    @property
    def last_position(self) -> int:
        return self.first_position + self.tokens - 1


@dataclass(frozen=True)
class RequestState:
    """一条序列在一次调用**之前**（``state_in``）或**之后**（``state_out``）的状态。

    ``next_position`` 是这条序列累计处理过的 token 数，**不是**它当前保留的块数：
    SWA 只留窗口内的块，也不能把窗口长度当成已处理 token 数
    （``M2_LAYER_DESIGN.md`` §2）。
    """

    request_id: str
    next_position: int
    blocks: tuple[KvBlockRef, ...] = ()
    version: int = 1
    state_location: str = LOC_DRAM


@dataclass(frozen=True)
class WeightStateRef:
    """One immutable weight version and its surviving storage copies.

    IDs are the layer-local WeightRef.id; expansion adds the layer namespace.
    Current locations are authoritative and must not reset to initial_location.
    """

    id: str
    version: int
    storage_bytes: int
    locations: tuple[str, ...]


@dataclass(frozen=True)
class StateIn:
    """一次调用的历史状态：每条序列一条记录，缺一条就拒绝（不按零历史猜）。"""

    requests: tuple[RequestState, ...] = ()
    weight_copies: tuple[WeightStateRef, ...] = ()
    layer: str | None = None

    def for_request(self, request_id: str) -> RequestState:
        for state in self.requests:
            if state.request_id == request_id:
                return state
        raise InvalidInput(f"state_in 里没有 request {request_id!r} 的记录")


@dataclass(frozen=True)
class HiddenOutRef:
    """一次调用交付的 hidden activation。"""

    request_id: str
    first_position: int
    tokens: int
    storage_bytes: int
    locations: tuple[str, ...]


@dataclass(frozen=True)
class StateOut:
    """一次调用的持久产物：更新后的状态块 **加上** 交付的 hidden。

    ``requests`` 与 :class:`StateIn` 用的是同一个类型——「一份合法的
    ``state_out`` 就是下一次调用的 ``state_in``」这句话因此在类型上成立，而不是
    靠调用方自觉。这个等式只在 M2 层成立：内核里第一次调用的块是产出张量，第二次
    是带 ``initial_locations`` 的叶子，按构造就不同（见
    ``M2_LAYER_DESIGN.md`` §2 与展开器的说明）。
    """

    layer: str
    version: int
    requests: tuple[RequestState, ...]
    hidden: tuple[HiddenOutRef, ...] = ()
    weight_copies: tuple[WeightStateRef, ...] = ()

    def total_kv_bytes(self, request_id: str) -> int:
        for state in self.requests:
            if state.request_id == request_id:
                return sum(block.storage_bytes for block in state.blocks)
        raise InvalidInput(f"state_out 里没有 request {request_id!r} 的记录")


@dataclass(frozen=True)
class StateDocument:
    """磁盘上的一份状态文档：``state_in`` 或 ``state_out`` 都能写成它。"""

    layer: str
    requests: tuple[RequestState, ...]
    version: int = 1
    hidden: tuple[HiddenOutRef, ...] = ()
    weight_copies: tuple[WeightStateRef, ...] = ()

    def as_state_in(self) -> StateIn:
        return StateIn(requests=self.requests, weight_copies=self.weight_copies,
                       layer=self.layer if self.weight_copies else None)

    def as_state_out(self) -> StateOut:
        return StateOut(
            layer=self.layer, version=self.version, requests=self.requests, hidden=self.hidden,
            weight_copies=self.weight_copies
        )


_REQUEST_STATE_KEYS = {"request_id", "next_position", "version", "state_location", "blocks"}
_BLOCK_KEYS = {"id", "first_position", "tokens", "storage_bytes", "locations"}
_STATE_IN_KEYS = {"requests", "weight_copies", "layer"}
_STATE_DOCUMENT_KEYS = {"format_version", "layer", "version", "requests", "hidden", "weight_copies", "comment"}
_WEIGHT_STATE_KEYS = {"id", "version", "storage_bytes", "locations"}
_HIDDEN_KEYS = {"request_id", "first_position", "tokens", "storage_bytes", "locations"}


def _parse_locations(raw: Any, what: str, *, minimum: int = 0) -> tuple[str, ...]:
    entries = _require_list(raw if raw is not None else [], what)
    if len(entries) < minimum:
        raise InvalidInput(f"{what} 至少要有 {minimum} 个驻留域，得到 {len(entries)}")
    locations: list[str] = []
    for i, entry in enumerate(entries):
        location = _require_one_of(entry, f"{what}[{i}]", (LOC_DRAM, LOC_VRAM, LOC_SSD))
        _expect(location not in locations, f"{what} 列了 {location!r} 两次")
        locations.append(location)
    return tuple(locations)


def _parse_block(raw: Any, what: str) -> KvBlockRef:
    item = _require_mapping(raw, what)
    _reject_unknown_keys(item, _BLOCK_KEYS, what)
    first_position = _require_int(item.get("first_position"), f"{what}.first_position")
    tokens = _require_int(item.get("tokens"), f"{what}.tokens", minimum=1)
    return KvBlockRef(
        id=_require_str(item.get("id"), f"{what}.id"),
        first_position=first_position,
        tokens=tokens,
        storage_bytes=_require_int(
            item.get("storage_bytes"), f"{what}.storage_bytes", minimum=1
        ),
        locations=_parse_locations(item.get("locations"), f"{what}.locations", minimum=1),
    )


def _parse_request_state(raw: Any, what: str, *, allow_ssd: bool = False) -> RequestState:
    item = _require_mapping(raw, what)
    _reject_unknown_keys(item, _REQUEST_STATE_KEYS, what)
    blocks_raw = _require_list(item.get("blocks", []), f"{what}.blocks")
    blocks = tuple(
        _parse_block(entry, f"{what}.blocks[{i}]") for i, entry in enumerate(blocks_raw)
    )
    return RequestState(
        request_id=_require_str(item.get("request_id"), f"{what}.request_id"),
        next_position=_require_int(item.get("next_position"), f"{what}.next_position"),
        blocks=blocks,
        version=_require_int(item.get("version", 1), f"{what}.version", minimum=1),
        state_location=_require_one_of(
            item.get("state_location", LOC_DRAM), f"{what}.state_location",
            (LOC_DRAM, LOC_VRAM, LOC_SSD) if allow_ssd else _LOCATIONS,
        ),
    )


def _parse_requests_raw(raw: Any, what: str, *, allow_ssd: bool = False) -> tuple[RequestState, ...]:
    entries = _require_list(raw, what)
    states: list[RequestState] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        state = _parse_request_state(entry, f"{what}[{i}]", allow_ssd=allow_ssd)
        _expect(
            state.request_id not in seen,
            f"{what}[{i}].request_id 重复：{state.request_id!r}",
        )
        seen.add(state.request_id)
        states.append(state)
    return tuple(states)


def _parse_weight_copies(raw: Any, what: str) -> tuple[WeightStateRef, ...]:
    copies: list[WeightStateRef] = []
    seen: set[str] = set()
    for i, entry in enumerate(_require_list(raw, what)):
        label = f"{what}[{i}]"
        item = _require_mapping(entry, label)
        _reject_unknown_keys(item, _WEIGHT_STATE_KEYS, label)
        identity = _require_str(item.get("id"), f"{label}.id")
        if identity in seen:
            raise InvalidInput(f"{what}: duplicate weight id {identity!r}")
        seen.add(identity)
        copies.append(WeightStateRef(
            id=identity, version=_require_int(item.get("version"), f"{label}.version", minimum=1),
            storage_bytes=_require_int(item.get("storage_bytes"), f"{label}.storage_bytes", minimum=1),
            locations=_parse_locations(item.get("locations"), f"{label}.locations", minimum=1),
        ))
    return tuple(copies)


def parse_state_in(raw: Any, *, where: str = "<state_in>", allow_ssd: bool = False) -> StateIn:
    block = _require_mapping(raw, where)
    _reject_unknown_keys(block, _STATE_IN_KEYS, where)
    return StateIn(requests=_parse_requests_raw(block.get("requests"), f"{where}.requests", allow_ssd=allow_ssd),
                   weight_copies=_parse_weight_copies(block.get("weight_copies", []), f"{where}.weight_copies"),
                   layer=_require_str(block["layer"], f"{where}.layer") if "layer" in block else None)


def parse_state_document(raw: Any, *, where: str = "<state>") -> StateDocument:
    """读一份独立的状态文件（可以是 ``state_in``，也可以是 ``state_out``）。"""
    block = _require_mapping(raw, where)
    _reject_unknown_keys(block, _STATE_DOCUMENT_KEYS, where)
    state_format = _require_str(block.get("format_version"), f"{where}.format_version")
    if state_format != STATE_CACHE_FORMAT_VERSION:
        _check_format_version(state_format, where)
        if "weight_copies" in block:
            raise InvalidInput(f"{where}.weight_copies requires {STATE_CACHE_FORMAT_VERSION}; legacy state must not drop cache copies")
    elif "weight_copies" not in block:
        raise InvalidInput(f"{where}.weight_copies must be explicit in {STATE_CACHE_FORMAT_VERSION}")
    hidden_raw = _require_list(block.get("hidden", []), f"{where}.hidden")
    hidden: list[HiddenOutRef] = []
    for i, entry in enumerate(hidden_raw):
        item_label = f"{where}.hidden[{i}]"
        item = _require_mapping(entry, item_label)
        _reject_unknown_keys(item, _HIDDEN_KEYS, item_label)
        hidden.append(
            HiddenOutRef(
                request_id=_require_str(item.get("request_id"), f"{item_label}.request_id"),
                first_position=_require_int(
                    item.get("first_position"), f"{item_label}.first_position"
                ),
                tokens=_require_int(item.get("tokens"), f"{item_label}.tokens", minimum=1),
                storage_bytes=_require_int(
                    item.get("storage_bytes"), f"{item_label}.storage_bytes", minimum=1
                ),
                locations=_parse_locations(item.get("locations"), f"{item_label}.locations"),
            )
        )
    return StateDocument(
        layer=_require_str(block.get("layer"), f"{where}.layer"),
        version=_require_int(block.get("version", 1), f"{where}.version", minimum=1),
        requests=_parse_requests_raw(block.get("requests"), f"{where}.requests", allow_ssd=state_format == STATE_CACHE_FORMAT_VERSION),
        hidden=tuple(hidden),
        weight_copies=_parse_weight_copies(block.get("weight_copies", []), f"{where}.weight_copies"),
    )


def load_state_document(path: str | Path) -> StateDocument:
    target = Path(path)
    return parse_state_document(_read_json(target), where=target.name)


def state_in_from_state_out(state_out: StateOut) -> StateIn:
    """把一次调用的产物接成下一次调用的输入。

    这正是 ``M2_LAYER_ACCEPTANCE.md`` §2 最后一句要检查的动作：旧块的副本位置从
    第一次输出传进第二次输入。没有这一步，"第二次调用还得重新搬一遍历史"就会是
    一个没人看得见的假设。
    """
    return StateIn(requests=state_out.requests, weight_copies=state_out.weight_copies,
                   layer=state_out.layer if state_out.weight_copies else None)


def to_state_document(state: StateIn | StateOut, *, layer: str | None = None) -> dict[str, Any]:
    """把状态写成 JSON 文档。``state_in`` 与 ``state_out`` 走同一个函数。"""
    if isinstance(state, StateIn):
        if state.weight_copies and state.layer is not None and layer is not None and layer != state.layer:
            raise InvalidInput("to_state_document cannot relabel a cache snapshot from another layer")
        resolved_layer = layer if layer is not None else state.layer or ""
        version = 1
        hidden: tuple[HiddenOutRef, ...] = ()
    else:
        resolved_layer = state.layer
        version = state.version
        hidden = state.hidden
    if not resolved_layer:
        raise InvalidInput("to_state_document 需要 layer（state_in 上不带层名）")
    document = {
        "format_version": STATE_CACHE_FORMAT_VERSION if state.weight_copies else FORMAT_VERSION,
        "layer": resolved_layer,
        "version": version,
        "requests": [
            {
                "request_id": request.request_id,
                "next_position": request.next_position,
                "version": request.version,
                "state_location": request.state_location,
                "blocks": [
                    {
                        "id": block.id,
                        "first_position": block.first_position,
                        "tokens": block.tokens,
                        "storage_bytes": block.storage_bytes,
                        "locations": list(block.locations),
                    }
                    for block in request.blocks
                ],
            }
            for request in state.requests
        ],
        "hidden": [
            {
                "request_id": ref.request_id,
                "first_position": ref.first_position,
                "tokens": ref.tokens,
                "storage_bytes": ref.storage_bytes,
                "locations": list(ref.locations),
            }
            for ref in hidden
        ],
    }
    if state.weight_copies:
        document["weight_copies"] = [
            {"id": ref.id, "version": ref.version, "storage_bytes": ref.storage_bytes,
             "locations": list(ref.locations)} for ref in state.weight_copies
        ]
    return document


def to_state_in_document(state_in: StateIn, *, layer: str) -> dict[str, Any]:
    """``state_in`` 内嵌进 invocation 文档时的那一小块（不带 ``format_version``）。"""
    document = to_state_document(state_in, layer=layer)
    result = {"requests": document["requests"]}
    if "weight_copies" in document:
        result["weight_copies"] = document["weight_copies"]
        result["layer"] = document["layer"]
    return result


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """这次调用跑在哪台机器上。容量与对齐都要显式给，没有默认值可用。"""

    dram_capacity_bytes: int
    vram_capacity_bytes: int
    dram_alignment_bytes: int = 1
    vram_alignment_bytes: int = 1
    dram_runtime_reserved_bytes: int = 0
    vram_runtime_reserved_bytes: int = 0


@dataclass(frozen=True)
class RequestInput:
    """一次调用里的一条序列。

    ``position_start`` 必须等于该序列 ``state_in.next_position``——两者不一致意味着
    「这条序列已经处理到哪」有两个说法，而它们在 M2 里必须是同一件事。
    """

    request_id: str
    position_start: int
    tokens: int
    input_location: str = LOC_DRAM
    output_location: str = LOC_DRAM


@dataclass(frozen=True)
class Invocation:
    """一次 layer 调用（``prefill`` 或 ``decode``）。"""

    id: str
    layer: str
    phase: str
    requests: tuple[RequestInput, ...]
    hardware: Hardware
    state_in: StateIn
    layer_file: str | None = None
    costs_file: str | None = None
    raw_cost_table: Mapping[str, Any] | None = None
    devices: tuple[str, ...] = (DEVICE_CPU,)
    mapper: dict[str, Any] | None = None
    format_version: str = FORMAT_VERSION
    allow_validation_only_costs: bool = False
    route_trace: tuple[RouteTraceEntry, ...] = ()
    route_source: str | None = None
    information_mode: str | None = None

    @property
    def batch(self) -> int:
        return len(self.requests)


_INVOCATION_KEYS = {
    "format_version",
    "id",
    "layer",
    "layer_file",
    "costs_file",
    "costs",
    "phase",
    "hardware",
    "requests",
    "state_in",
    "devices",
    "mapper",
    "allow_validation_only_costs",
    "route_trace",
    "route_source",
    "information_mode",
    "comment",
}
_HARDWARE_KEYS = {
    "dram_capacity_bytes",
    "vram_capacity_bytes",
    "dram_alignment_bytes",
    "vram_alignment_bytes",
    "dram_runtime_reserved_bytes",
    "vram_runtime_reserved_bytes",
}
_REQUEST_IN_KEYS = {"request_id", "position_start", "tokens", "input_location", "output_location"}


def _require_explicit_hybrid_locations(raw: Any, where: str, fields: tuple[str, ...], *, allow_ssd: bool = False) -> None:
    """The hybrid parser must not turn missing placement into a DRAM default."""
    entry = _require_mapping(raw, where)
    for field in fields:
        if field not in entry:
            raise InvalidInput(f"{where}.{field} must be explicit for hybrid execution")
        _require_one_of(entry[field], f"{where}.{field}", (LOC_DRAM, LOC_VRAM, LOC_SSD) if allow_ssd else _LOCATIONS)


def _parse_hardware(raw: Any, where: str) -> Hardware:
    block = _require_mapping(raw, f"{where}.hardware")
    label = f"{where}.hardware"
    _reject_unknown_keys(block, _HARDWARE_KEYS, label)
    return Hardware(
        dram_capacity_bytes=_require_int(
            block.get("dram_capacity_bytes"), f"{label}.dram_capacity_bytes", minimum=1
        ),
        vram_capacity_bytes=_require_int(
            block.get("vram_capacity_bytes"), f"{label}.vram_capacity_bytes", minimum=1
        ),
        dram_alignment_bytes=_require_int(
            block.get("dram_alignment_bytes", 1), f"{label}.dram_alignment_bytes", minimum=1
        ),
        vram_alignment_bytes=_require_int(
            block.get("vram_alignment_bytes", 1), f"{label}.vram_alignment_bytes", minimum=1
        ),
        dram_runtime_reserved_bytes=_require_int(
            block.get("dram_runtime_reserved_bytes", 0),
            f"{label}.dram_runtime_reserved_bytes",
        ),
        vram_runtime_reserved_bytes=_require_int(
            block.get("vram_runtime_reserved_bytes", 0),
            f"{label}.vram_runtime_reserved_bytes",
        ),
    )


def _parse_request_input(raw: Any, where: str, *, allow_ssd: bool = False) -> RequestInput:
    item = _require_mapping(raw, where)
    _reject_unknown_keys(item, _REQUEST_IN_KEYS, where)
    return RequestInput(
        request_id=_require_str(item.get("request_id"), f"{where}.request_id"),
        position_start=_require_int(item.get("position_start"), f"{where}.position_start"),
        tokens=_require_int(item.get("tokens"), f"{where}.tokens", minimum=1),
        input_location=_require_one_of(
            item.get("input_location", LOC_DRAM), f"{where}.input_location",
            (LOC_DRAM, LOC_VRAM, LOC_SSD) if allow_ssd else _LOCATIONS,
        ),
        output_location=_require_one_of(
            item.get("output_location", LOC_DRAM), f"{where}.output_location",
            (LOC_DRAM, LOC_VRAM, LOC_SSD) if allow_ssd else _LOCATIONS,
        ),
    )


def parse_invocation_document(raw: Any, *, where: str = "<invocation>") -> Invocation:
    """解析调用文档。``layer``/``costs`` 的**文件**引用原样带出，由调用方解析。"""
    block = _require_mapping(raw, where)
    _reject_unknown_keys(block, _INVOCATION_KEYS, where)
    version = _check_format_version(block.get("format_version"), where)
    allow_validation = block.get("allow_validation_only_costs", False)
    if not isinstance(allow_validation, bool):
        raise InvalidInput(f"{where}.allow_validation_only_costs must be boolean")
    if "allow_validation_only_costs" in block and version not in _PHYSICAL_VERSIONS:
        raise InvalidInput(f"{where}.allow_validation_only_costs requires an explicit physical 0.2, 0.3, 0.4 or 0.5 boundary")

    if version != MOE_FORMAT_VERSION and any(key in block for key in ("route_trace", "route_source", "information_mode")):
        raise InvalidInput(f"{where}: route declarations require {MOE_FORMAT_VERSION}")
    if version in (HYBRID_FORMAT_VERSION, MOE_FORMAT_VERSION) and "devices" not in block:
        raise InvalidInput(f"{where}.devices must be explicit for hybrid execution")
    requests_raw = _require_list(block.get("requests"), f"{where}.requests")
    _expect(bool(requests_raw), f"{where}.requests 不能为空（B 至少为 1）")
    requests: list[RequestInput] = []
    seen: set[str] = set()
    for i, entry in enumerate(requests_raw):
        if version in (HYBRID_FORMAT_VERSION, MOE_FORMAT_VERSION):
            _require_explicit_hybrid_locations(entry, f"{where}.requests[{i}]", ("input_location", "output_location"), allow_ssd=version == MOE_FORMAT_VERSION)
        request = _parse_request_input(entry, f"{where}.requests[{i}]", allow_ssd=version == MOE_FORMAT_VERSION)
        _expect(
            request.request_id not in seen,
            f"{where}.requests[{i}].request_id 重复：{request.request_id!r}",
        )
        seen.add(request.request_id)
        requests.append(request)

    devices_raw = _require_list(block.get("devices", [DEVICE_CPU]), f"{where}.devices")
    _expect(bool(devices_raw), f"{where}.devices 不能为空")
    devices: list[str] = []
    for i, entry in enumerate(devices_raw):
        device = _require_one_of(entry, f"{where}.devices[{i}]", _DEVICES)
        _expect(device not in devices, f"{where}.devices 列了 {device!r} 两次")
        devices.append(device)

    costs_file = block.get("costs_file")
    raw_cost_table = block.get("costs")
    _expect(
        (costs_file is None) != (raw_cost_table is None),
        f"{where}: 成本表要恰好给一处——``costs_file`` 指向文件，或者内嵌 ``costs``",
    )
    if costs_file is not None:
        costs_file = _require_str(costs_file, f"{where}.costs_file")
    else:
        raw_cost_table = _require_mapping(raw_cost_table, f"{where}.costs")

    mapper = block.get("mapper")
    if mapper is not None:
        mapper = dict(_require_mapping(mapper, f"{where}.mapper"))

    state_in_raw = block.get("state_in")
    if state_in_raw is None:
        raise InvalidInput(
            f"{where}.state_in 必须显式给出：没有历史的序列写 "
            '{"requests": [{"request_id": ..., "next_position": 0, "blocks": []}]}，'
            "省略会让「这次调用有没有历史」变成猜的"
        )

    if version in (HYBRID_FORMAT_VERSION, MOE_FORMAT_VERSION):
        state_document = _require_mapping(state_in_raw, f"{where}.state_in")
        for i, entry in enumerate(_require_list(state_document.get("requests"), f"{where}.state_in.requests")):
            _require_explicit_hybrid_locations(entry, f"{where}.state_in.requests[{i}]", ("state_location",), allow_ssd=version == MOE_FORMAT_VERSION)

    if version == MOE_FORMAT_VERSION:
        if "weight_copies" not in _require_mapping(state_in_raw, f"{where}.state_in"):
            raise InvalidInput(f"{where}.state_in.weight_copies must be explicit for MoE; no implicit weight reset")
        route_source = _require_str(block.get("route_source"), f"{where}.route_source")
        information_mode = _require_one_of(block.get("information_mode"), f"{where}.information_mode", INFORMATION_MODES)
        route_trace = parse_route_trace(block.get("route_trace"), where=f"{where}.route_trace")
    else:
        if "weight_copies" in state_in_raw or "layer" in state_in_raw:
            raise InvalidInput(f"{where}.state_in cache/source declarations require the MoE cache-aware boundary")
        route_source, information_mode, route_trace = None, None, ()

    parsed_history = parse_state_in(state_in_raw, where=f"{where}.state_in", allow_ssd=version == MOE_FORMAT_VERSION)
    if version == MOE_FORMAT_VERSION and parsed_history.layer is None:
        parsed_history = replace(parsed_history, layer=_require_str(block.get("layer"), f"{where}.layer"))

    return Invocation(
        id=_require_str(block.get("id"), f"{where}.id"),
        layer=_require_str(block.get("layer"), f"{where}.layer"),
        phase=_require_one_of(block.get("phase"), f"{where}.phase", PHASES),
        requests=tuple(requests),
        hardware=_parse_hardware(block.get("hardware"), where),
        state_in=parsed_history,
        layer_file=(
            _require_str(block.get("layer_file"), f"{where}.layer_file")
            if block.get("layer_file") is not None
            else None
        ),
        costs_file=costs_file,
        raw_cost_table=raw_cost_table,
        devices=tuple(devices),
        mapper=mapper,
        format_version=version,
        allow_validation_only_costs=allow_validation,
        route_source=route_source,
        information_mode=information_mode,
        route_trace=route_trace,
    )


def load_invocation(path: str | Path) -> Invocation:
    """读一份调用文件。``layer_file`` / ``costs_file`` 相对**该文件**解析。"""
    target = Path(path)
    return parse_invocation_document(_read_json(target), where=target.name)


# ---------------------------------------------------------------------------
# 跨文档校验
# ---------------------------------------------------------------------------

# 本刀支持的组合（M2_LAYER_DESIGN.md §6 第 2 条：先打通 full_causal + dense）。
SLICE_ATTENTION = ATTENTION_FULL_CAUSAL
SLICE_FFN = FFN_DENSE
SLICE_DEVICE = DEVICE_CPU

_REQUIRED_TASK_KINDS = (SEMANTIC_PROJECTION, SEMANTIC_ATTENTION, SEMANTIC_DENSE_FFN)


def validate_invocation(
    layer: LayerDescriptor, invocation: Invocation, state_in: StateIn | None = None
) -> None:
    """层 × 调用 × 历史状态一起看才成立的规则。

    每一条拒绝都对应一个**会安静地给出错误答案**的默认值，所以这里一条都不给：
    ``position_start`` 与 ``next_position`` 不一致时，无论挑哪一个当真都是错的；
    历史状态缺一条记录时，「按零历史猜」会把一次带历史的 decode 变成一次没有历史
    的 decode。
    """
    history = invocation.state_in if state_in is None else state_in
    if not isinstance(invocation.allow_validation_only_costs, bool):
        raise InvalidInput("invocation.allow_validation_only_costs must be boolean")

    # State objects can also arrive directly from Python, without the JSON
    # parser. Reject ambiguous identity and invalid versions before Dense/MoE
    # expansion can select only the first matching record or advance a bad one.
    request_ids: set[str] = set()
    for request in invocation.requests:
        identity = _require_str(request.request_id, "invocation.request_id")
        if identity in request_ids:
            raise InvalidInput(f"invocation.request_id 重复：{identity!r}")
        request_ids.add(identity)
    state_ids: set[str] = set()
    for state in history.requests:
        identity = _require_str(state.request_id, "state_in.request_id")
        if identity in state_ids:
            raise InvalidInput(f"state_in.request_id 重复：{identity!r}")
        state_ids.add(identity)
        _require_int(state.version, f"state_in[{identity!r}].version", minimum=1)

    if invocation.layer != layer.id:
        raise InvalidInput(
            f"invocation {invocation.id!r} 说的是 layer {invocation.layer!r}，"
            f"而拿到的是 {layer.id!r} 的描述符"
        )

    if invocation.format_version != layer.format_version:
        raise InvalidInput(
            f"invocation/layer format_version mismatch: {invocation.format_version!r} vs {layer.format_version!r}; "
            "hidden inputs must not silently become external Q/K/V/gate ports"
        )
    if (layer.physical is not None) != (layer.format_version in _PHYSICAL_VERSIONS):
        raise InvalidInput("physical boundary requires an explicit supported physical layer and invocation")

    if layer.format_version == MOE_FORMAT_VERSION:
        validate_moe_invocation(layer, invocation, history)
        return
    if history.weight_copies:
        raise Unsupported("weight cache snapshots require the MoE cache-aware boundary; legacy execution must not discard them")
    if invocation.route_trace or invocation.route_source is not None or invocation.information_mode is not None:
        raise InvalidInput("route declarations require m2-layer-0.6")

    if layer.attention != SLICE_ATTENTION:
        raise Unsupported(
            f"layer {layer.id!r} 的注意力类型是 {layer.attention!r}；本版本只实现 "
            f"{SLICE_ATTENTION!r}（M2_LAYER_DESIGN.md §6 第 2 条；SWA 与递归线性状态"
            "分别见 §4 的另外两节，尚未实现）"
        )
    if layer.ffn != SLICE_FFN:
        raise Unsupported(
            f"layer {layer.id!r} 的 FFN 类型是 {layer.ffn!r}；本版本只实现 "
            f"{SLICE_FFN!r}（MoE 路由与揭示门见 M2_LAYER_DESIGN.md §5，尚未实现）"
        )
    hybrid = isinstance(layer.physical, HybridDensePhysical)
    execution_device = DEVICE_GPU if isinstance(layer.physical, CudaDensePhysical) else SLICE_DEVICE
    if hybrid:
        if not invocation.devices or any(device not in _DEVICES for device in invocation.devices):
            raise Unsupported("hybrid devices must be a nonempty subset of cpu/gpu")
        if len(set(invocation.devices)) != len(invocation.devices):
            raise InvalidInput("hybrid devices must not contain duplicates")
    else:
        if isinstance(layer.physical, CudaDensePhysical) and invocation.devices != (DEVICE_GPU,):
            raise Unsupported("CUDA full block requires devices=['gpu']; hybrid execution is not implemented")
        for device in invocation.devices:
            if device != execution_device:
                raise Unsupported(
                    f"invocation {invocation.id!r} allows {device!r}; this physical boundary "
                    f"requires {execution_device!r} single-device execution; no implicit transfer"
                )

    if layer.physical is not None:
        (validate_hybrid_physical_layer if layer.format_version == HYBRID_FORMAT_VERSION
         else validate_cuda_physical_layer if layer.format_version == CUDA_FORMAT_VERSION
         else validate_full_physical_layer if layer.format_version == FULL_FORMAT_VERSION else validate_physical_layer)(layer)
        if invocation.batch != 1:
            raise Unsupported("physical macro execution currently requires batch=1; per-request graphs are not whole-batch kernels")
    required_kinds = (_REQUIRED_TASK_KINDS if layer.physical is None or layer.format_version in _COMPLETE_VERSIONS
                      else (SEMANTIC_ATTENTION, SEMANTIC_DENSE_FFN))
    kinds = {task.kind for task in layer.tasks}
    missing = [kind for kind in required_kinds if kind not in kinds]
    if missing:
        raise InvalidInput(
            f"layer {layer.id!r} 的任务图缺少 {', '.join(missing)}；"
            "full_causal + dense 的可调度链是 projection -> attention -> dense_ffn"
        )
    for kind in required_kinds:
        chosen = layer.tasks_of_kind(kind)
        if len(chosen) != 1:
            raise Unsupported(
                f"layer {layer.id!r} 有 {len(chosen)} 个 {kind!r} 任务；本版本每个"
                "层里这几种各只支持一个"
            )

    # 历史状态：调用里每条序列都要有记录，且不能多出调用里没有的序列。
    known = {request.request_id for request in invocation.requests}
    for state in history.requests:
        if state.request_id not in known:
            raise InvalidInput(
                f"state_in 里有 request {state.request_id!r}，但这次调用没有它；"
                "把不同 request 的状态合并或错配都会让状态延续悄悄错位"
            )
    seen_blocks: set[str] = set()
    for request in invocation.requests:
        state = _state_for(history, request, invocation)
        if request.position_start != state.next_position:
            raise InvalidInput(
                f"request {request.request_id!r}: position_start="
                f"{request.position_start} 与 state_in.next_position="
                f"{state.next_position} 不一致；两者必须是同一件事"
            )
        if invocation.phase == PHASE_DECODE and request.tokens != 1:
            raise InvalidInput(
                f"request {request.request_id!r}: decode 的 q_i 必须是 1，得到 "
                f"{request.tokens}"
            )
        for block in state.blocks:
            if block.id in seen_blocks:
                raise InvalidInput(f"state_in 里块 id {block.id!r} 出现了两次")
            seen_blocks.add(block.id)

        _validate_full_causal_history(layer, request, state)
        if hybrid:
            _require_location(request.input_location, f"request {request.request_id!r}.input_location")
            _require_location(request.output_location, f"request {request.request_id!r}.output_location")
            _require_location(state.state_location, f"request {request.request_id!r}.state_location")
            for retained in state.blocks:
                if not retained.locations or any(location not in _LOCATIONS for location in retained.locations):
                    raise Unsupported("hybrid history KV requires nonempty dram/vram locations; SSD is not in this slice")
                if len(set(retained.locations)) != len(retained.locations):
                    raise InvalidInput("hybrid history KV locations must not contain duplicates")
            continue
        if isinstance(layer.physical, CudaDensePhysical) and any(block.locations != (LOC_VRAM,) for block in state.blocks):
            raise Unsupported("CUDA full block requires history KV blocks exclusively in vram; no implicit copy")
        if request.output_location != DOMAIN_OF_DEVICE[execution_device]:
            raise Unsupported(
                f"request {request.request_id!r}: 本版本在 {execution_device!r} 上执行，"
                f"产出的驻留域只能是 {DOMAIN_OF_DEVICE[execution_device]!r}，得到 "
                f"{request.output_location!r}"
            )
        if state.state_location != DOMAIN_OF_DEVICE[execution_device]:
            raise Unsupported(
                f"request {request.request_id!r}: 状态驻留域只能是 "
                f"{DOMAIN_OF_DEVICE[execution_device]!r}，得到 {state.state_location!r}"
            )
        if request.input_location != DOMAIN_OF_DEVICE[execution_device]:
            raise Unsupported(
                f"request {request.request_id!r}: 输入驻留域只能是 "
                f"{DOMAIN_OF_DEVICE[execution_device]!r}，得到 {request.input_location!r}"
            )


def _state_for(history: StateIn, request: RequestInput, invocation: Invocation) -> RequestState:
    for state in history.requests:
        if state.request_id == request.request_id:
            return state
    raise InvalidInput(
        f"state_in 里没有 request {request.request_id!r} 的记录；没有历史的序列要"
        "显式写一条零 KV 块、next_position=0 的记录，不能省略（省略会让"
        "「这次调用有没有历史」变成猜的）"
    )


def _validate_full_causal_history(
    layer: LayerDescriptor, request: RequestInput, state: RequestState
) -> None:
    """全局因果注意力：位置 ``p`` 只能读 ``<= p`` 的 KV，且历史必须连续。"""
    covered = 0
    expected_start = 0
    for block in sorted(state.blocks, key=lambda b: b.first_position):
        if block.first_position != expected_start:
            raise InvalidInput(
                f"request {request.request_id!r}: {block.id!r} 从位置 "
                f"{block.first_position} 开始，但前面只覆盖到 {expected_start - 1}；"
                "全局因果注意力不接受不连续的历史（M2_LAYER_DESIGN.md §4）"
            )
        expected_size = block.tokens * layer.kv_bytes_per_token
        if block.storage_bytes != expected_size:
            raise InvalidInput(
                f"request {request.request_id!r}: {block.id!r} 是 {block.tokens} 个"
                f" token，按每 token {layer.kv_bytes_per_token} B 应为 {expected_size} B，"
                f"但文档写的是 {block.storage_bytes} B"
            )
        expected_start = block.first_position + block.tokens
        covered += block.tokens
    if covered != state.next_position:
        raise InvalidInput(
            f"request {request.request_id!r}: 块合计覆盖 {covered} 个位置，但 "
            f"next_position={state.next_position}；两者不一致时无法判断历史到哪"
        )


def hidden_bytes(layer: LayerDescriptor, tokens: int) -> int:
    """``tokens`` 个 token 的 hidden 字节数（验收 §1 的 F32/连续/独立分配）。"""
    return tokens * layer.hidden_bytes_per_token


def invocation_hidden_bytes(layer: LayerDescriptor, invocation: Invocation, request_id: str) -> int:
    for request in invocation.requests:
        if request.request_id == request_id:
            return hidden_bytes(layer, request.tokens)
    raise InvalidInput(f"unknown request {request_id!r}")
