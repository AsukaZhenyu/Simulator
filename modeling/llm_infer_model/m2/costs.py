"""M2 逐形状的宏任务成本表，以及它到内核 ``costs`` 文档的展开。

这个模块是**唯一**接触成本表物理格式的地方：``parse_cost_table`` 读它，
``expand_costs_document`` 把它变成内核认的 ``{"source", "compute", "transfer"}``。
将来 codex 测出来的真实表换了字段名或换成了浮点秒，改动只发生在这两个函数里。

为什么不能直接调 :mod:`llm_infer_model.tensor.layer_costs` 的现成入口
（``costs_for_chain`` / ``costs_from_layers``）：那两条路都要求每个计算节点**恰好**
一份 ``role="weight"`` 的输入。M2 的宏任务两头都不满足——``attention`` /
``state_update`` / ``combine`` 一份权重都不读，``expert_ffn`` 要读 E 份；它们还
把 KV、workspace、staging 这些成本之外的占用**静默丢掉**。所以这里走加法式入口：
逐个宏任务按形状查表，查不到就拒绝，``workspace_bytes`` 逐条带进来。
实测条目还须与展开器的物理签名完全一致，不能把孤立算子或不同边界的数字
装入同一个宏任务。未知 workspace 可以如实保留为 ``None``，执行时必须拒绝。

**来源标签进 ``costs.source``**：内核的 ``Costs.source`` 是一个自由字符串，下游
（产物、绘图、结论）靠它区分合成值与实测值。这里由每条成本自己的 ``provenance``
拼出来，所以一张全是合成值的表永远不会被报成实测
（``M2_LAYER_DESIGN.md`` §3 末段、§6 末段）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from ..units import seconds_to_ns
from ..tensor.spec import InvalidInput, Unsupported, ROUTES

__all__ = [
    "COST_SOURCE_PREFIX",
    "M2CostTable",
    "MacroCost",
    "MacroShape",
    "TransferEntry",
    "canonical_contract_hash",
    "expand_costs_document",
    "expand_transfer_costs",
    "parse_cost_table",
]

# 内核 ``costs.source`` 的前缀。后面接的是各条成本的来源标签，如
# ``m2:synthetic_layer``。前缀的作用是让「这张表来自 M2 的宏任务展开」在产物里
# 一眼可辨，而不是和手写的 ``synthetic_fixed`` 混在一起。
COST_SOURCE_PREFIX = "m2"

# Each expanded operation belongs to one request. B is invocation context, never
# permission to charge a measured whole-batch time once for each request.
ACCOUNTING_PER_REQUEST = "per_request"
ACCOUNTING_PER_EXPERT_GROUP = "per_expert_group"
_EXPERT_GROUP_KINDS = frozenset({"expert_ffn", "shared_expert_ffn"})
# Macro-owned reusable allocator capacity (including alignment) plus backend
# scratch; resident input/output ports, weights, KV and runtime overheads are
# excluded. The physical envelope must bind these external/internal boundaries.
WORKSPACE_SCOPE = "macro_internal_scratch"
MEASURED_PROVENANCE = "measured_macro"

_ENTRY_KEYS = {
    "kind",
    "device",
    "phase",
    "batch",
    "tokens",
    "visible_history",
    "duration_ns",
    "duration_s",
    "workspace_bytes",
    "provenance",
    "accounting",
    "physical_signature",
    "contract_hash",
    "workspace_scope",
    "workspace_source",
}
_TABLE_KEYS = {
    "format_version", "source", "provenance", "entries", "comment",
    "validation_only", "transfer_entries",
}
_HYBRID_VERSION = "m2-layer-0.5"
_MOE_VERSION = "m2-layer-0.6"
_TRANSFER_ROUTES = frozenset({"dram_to_vram", "vram_to_dram"})
_MOE_TRANSFER_ROUTES = _TRANSFER_ROUTES | frozenset({"dram_to_ssd", "ssd_to_dram"})
_TRANSFER_KEYS = {"route", "storage_bytes", "duration_ns", "provenance"}
_TRANSFER_PROVENANCE = "synthetic_transfer"


@dataclass(frozen=True, order=True)
class MacroShape:
    """一次宏任务调用的形状。

    ``visible_history`` 是这次调用里该序列**可读**的历史 token 数（本次调用开始时
    的 ``next_position``）。它对每个宏任务都用同一个定义——不按任务类型换含义。
    ``batch`` 是本次调用的请求数；当前展开器逐请求执行。因此旧合成表的 B>1
    条目仍然是「处于该 batch 上下文中的单请求成本」，不代表整批执行时间。
    实测在联合 batch 执行与计费实现之前只接受 B=1。

    .6 专家组是明确例外：tokens 是整个 batch 路由到同一专家的 token 总数，
    visible_history=0 表示历史不适用，不代表每个请求的历史长度为零；
    accounting=per_expert_group 将整个组只计费一次。它目前仅接受合成成本。

    一个随类型变的字段会让「查不到就拒绝」变成「查不到是因为我理解错了这个字段」，
    而那正是插值规则会溜进来的地方（``M2_LAYER_DESIGN.md`` §3）。
    """

    phase: str
    batch: int
    tokens: int
    visible_history: int

    def key(self) -> str:
        return f"{self.phase}/B{self.batch}/q{self.tokens}/hist{self.visible_history}"


@dataclass(frozen=True)
class MacroCost:
    """一个宏任务在一个形状上的成本。

    ``provenance`` 没有默认值：一个不说自己从哪来的数字，没法区分「实测」与
    「合成」，而这两者在结论里的分量完全不同。
    ``workspace_bytes=None`` 表示未知，不是零；它能存入测量表但不能执行。
    ``physical_signature`` 与 ``contract_hash`` 绑定该宏任务的物理边界。
    """

    duration_ns: int
    provenance: str
    workspace_bytes: int | None = 0
    accounting: str = ACCOUNTING_PER_REQUEST
    physical_signature: dict[str, Any] | None = None
    contract_hash: str | None = None
    workspace_scope: str | None = None
    workspace_source: str | None = None


@dataclass(frozen=True)
class TransferEntry:
    """One exact route/byte-count cost; M2 .5 currently accepts synthetic data.

    The byte count is tensor storage size, before hardware capacity alignment.
    Copying a tensor allocates its destination separately in the core engine;
    this entry specifies duration only and does not hide a staging allocation.
    """

    duration_ns: int
    provenance: str


@dataclass(frozen=True)
class M2CostTable:
    """``(kind, device, shape) -> MacroCost``。

    ``source`` 是作者给这张表起的名字，**不会**进内核：内核看到的是由
    ``provenance`` 拼出来的那个标签。一个作者写 ``source="measured"``、条目却全是
    ``synthetic_layer`` 的表，在内核产物里仍然是合成的。
    """

    source: str
    entries: dict[tuple[str, str, MacroShape], MacroCost]
    validation_only: bool = False
    transfers: dict[tuple[str, int], TransferEntry] = field(default_factory=dict)
    format_version: str | None = None

    @property
    def allowed_transfer_routes(self) -> frozenset[str]:
        return _MOE_TRANSFER_ROUTES if self.format_version == _MOE_VERSION else _TRANSFER_ROUTES

    def lookup(
        self, kind: str, device: str, shape: MacroShape,
        *, physical_signature: dict[str, Any] | None = None,
    ) -> MacroCost:
        cost = self.entries.get((kind, device, shape))
        if cost is None:
            available = sorted(
                f"{k}/{d}/{s.key()}"
                for (k, d, s) in self.entries
                if k == kind and d == device
            )
            raise Unsupported(
                f"成本表里没有 {kind!r} 在 {device!r}、形状 {shape.key()} 上的条目；"
                f"{kind!r}/{device!r} 已有的形状是 "
                f"{', '.join(available) if available else '（一个都没有）'}。"
                "缺形状时拒绝或用**明确标注、经验证**的插值规则，不能拿整层测量时间"
                "填给每个宏任务（M2_LAYER_DESIGN.md §3）"
            )
        _validate_usage(cost, shape, physical_signature, kind=kind, format_version=self.format_version)
        return cost

    @property
    def provenances(self) -> frozenset[str]:
        return frozenset(
            [cost.provenance for cost in self.entries.values()]
            + [cost.provenance for cost in self.transfers.values()]
        )

    def lookup_transfer(self, route: str, storage_bytes: int) -> TransferEntry:
        """Reject missing sizes/directions instead of interpolating or using zero."""
        if not isinstance(route, str) or route not in self.allowed_transfer_routes:
            raise Unsupported(f"M2 {self.format_version or 'legacy'} 不支持搬运路线 {route!r}")
        _transfer_positive_int(storage_bytes, "storage_bytes")
        cost = self.transfers.get((route, storage_bytes))
        if cost is None:
            available = sorted(size for direction, size in self.transfers if direction == route)
            raise Unsupported(
                f"成本表里没有搬运 {route!r}/storage_bytes={storage_bytes} 的精确条目；"
                f"已有字节数为 {available}，不插值、不按另一方向复用、不默认为 0"
            )
        _transfer_positive_int(cost.duration_ns, "transfer.duration_ns")
        if cost.provenance != _TRANSFER_PROVENANCE:
            raise Unsupported("M2 .5 搬运成本目前只接受 provenance='synthetic_transfer'")
        return cost


def _transfer_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput(f"{label} 必须是正整数，得到 {value!r}")
    return value


def _parse_transfer_entry(
    raw: Any, index: int, where: str, *, format_version: str,
) -> tuple[tuple[str, int], TransferEntry]:
    label = f"{where}.transfer_entries[{index}]"
    if not isinstance(raw, dict):
        raise InvalidInput(f"{label} 必须是 JSON 对象")
    if set(raw) != _TRANSFER_KEYS:
        raise InvalidInput(
            f"{label} 必须恰好包含 {sorted(_TRANSFER_KEYS)}；"
            f"缺失 {sorted(_TRANSFER_KEYS - set(raw))}，未知 {sorted(set(raw) - _TRANSFER_KEYS)}"
        )
    route = raw["route"]
    if not isinstance(route, str) or not route:
        raise InvalidInput(f"{label}.route 必须是非空字符串")
    allowed = _MOE_TRANSFER_ROUTES if format_version == _MOE_VERSION else _TRANSFER_ROUTES
    if route not in allowed:
        raise Unsupported(f"{label}: {format_version} 不支持搬运路线 {route!r}")
    storage_bytes = _transfer_positive_int(raw["storage_bytes"], f"{label}.storage_bytes")
    duration_ns = _transfer_positive_int(raw["duration_ns"], f"{label}.duration_ns")
    provenance = raw["provenance"]
    if not isinstance(provenance, str) or not provenance.strip():
        raise InvalidInput(f"{label}.provenance 必须是非空字符串")
    if provenance != _TRANSFER_PROVENANCE:
        raise Unsupported(f"{label}: 目前只接受 provenance='synthetic_transfer'")
    return (route, storage_bytes), TransferEntry(duration_ns, provenance)



def canonical_contract_hash(signature: dict[str, Any]) -> str:
    """Hash an exact, stable physical envelope, excluding external evidence IDs.

    This checks JSON representation only. The producer of the envelope validates
    geometry and boundary semantics; lookup binds its entire selected envelope.
    """
    if not isinstance(signature, dict) or not signature:
        raise InvalidInput("physical_signature 必须是非空 JSON 对象")

    def check(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise InvalidInput(f"{path}: JSON 对象键必须是字符串")
            for key, item in value.items():
                check(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check(item, f"{path}[{index}]")
        elif value is None or isinstance(value, (str, bool, int)):
            return
        elif isinstance(value, float) and math.isfinite(value):
            return
        else:
            raise InvalidInput(f"{path}: 必须是有限的 JSON 值，得到 {value!r}")

    check(signature, "physical_signature")
    encoded = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_signature_pair(
    signature: dict[str, Any] | None, contract_hash: str | None, label: str,
) -> None:
    if signature is None and contract_hash is None:
        return
    digest = canonical_contract_hash(signature)
    if (
        not isinstance(contract_hash, str) or len(contract_hash) != 64
        or any(char not in "0123456789abcdef" for char in contract_hash)
    ):
        raise InvalidInput(f"{label}.contract_hash 必须是小写 SHA256")
    if digest != contract_hash:
        raise InvalidInput(f"{label}.contract_hash 与 physical_signature 不一致")


def _validate_usage(
    cost: MacroCost, shape: MacroShape,
    physical_signature: dict[str, Any] | None,
    *, kind: str, format_version: str | None,
) -> None:
    expert_group = format_version == _MOE_VERSION and kind in _EXPERT_GROUP_KINDS
    expected_accounting = ACCOUNTING_PER_EXPERT_GROUP if expert_group else ACCOUNTING_PER_REQUEST
    if cost.accounting != expected_accounting:
        raise Unsupported(f"{kind}: accounting 必须是 {expected_accounting!r}")
    if expert_group and shape.visible_history != 0:
        raise Unsupported(".6 expert group visible_history 必须为 0（历史不适用于专家 FFN）")
    if format_version == _MOE_VERSION:
        if cost.provenance != "synthetic_layer":
            raise Unsupported(".6 仅接受 provenance='synthetic_layer'，不能用于实测声明")
        if cost.physical_signature is not None or cost.contract_hash is not None or physical_signature is not None:
            raise Unsupported(".6 synthetic MoE 不接受 physical_signature/contract_hash")
    measured = cost.provenance == MEASURED_PROVENANCE
    if measured and shape.batch != 1:
        raise Unsupported("当前 measured_macro 仅支持 batch=1；不能重复计入整批测量时长")
    if cost.workspace_bytes is None:
        raise Unsupported("workspace_bytes 未知，不能作为 0 参与容量验证")
    if (
        isinstance(cost.workspace_bytes, bool)
        or not isinstance(cost.workspace_bytes, int) or cost.workspace_bytes < 0
    ):
        raise InvalidInput("workspace_bytes 必须是非负整数，或显式未知 None")
    if measured:
        if cost.workspace_scope != WORKSPACE_SCOPE or not cost.workspace_source:
            raise InvalidInput("measured_macro 必须声明 workspace_scope 和 workspace_source")
        if physical_signature is None:
            raise Unsupported("measured_macro 需要展开器提供 physical_signature 验证宏任务边界")
    _check_signature_pair(cost.physical_signature, cost.contract_hash, "cost")
    if physical_signature is not None:
        expected_hash = canonical_contract_hash(physical_signature)
        if cost.physical_signature is None:
            raise Unsupported("成本没有 physical_signature，不能用于物理宏任务")
        # Comparing canonical bytes via their hash also avoids Python's True == 1
        # equality treating distinct JSON signatures as interchangeable.
        if cost.contract_hash != expected_hash:
            raise Unsupported("成本 physical_signature 与展开宏任务的物理边界/形状不匹配")

def _parse_entry(
    raw: Any, index: int, table_provenance: str | None, where: str,
    *, format_version: str | None,
) -> tuple[tuple[str, str, MacroShape], MacroCost]:
    label = f"{where}.entries[{index}]"
    if not isinstance(raw, dict):
        raise InvalidInput(f"{label} 必须是 JSON 对象，得到 {raw!r}")
    unknown = sorted(set(raw) - _ENTRY_KEYS)
    if unknown:
        raise InvalidInput(f"{label}: 未知字段 {', '.join(unknown)}")

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise InvalidInput(f"{label}.kind 必须是字符串，得到 {kind!r}")
    # 循环导入不值得为一句报错付：宏任务名由调用方给的 requirements 校验，
    # 这里的名字最终一定会进内核的 ``semantic_op`` 解析，那时会拒。
    device = raw.get("device")
    if not isinstance(device, str) or not device:
        raise InvalidInput(f"{label}.device 必须是字符串，得到 {device!r}")

    def _int(field: str, minimum: int = 0) -> int:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidInput(f"{label}.{field} 必须是整数，得到 {value!r}")
        if value < minimum:
            raise InvalidInput(f"{label}.{field} 必须 >= {minimum}，得到 {value}")
        return value

    shape = MacroShape(
        phase=raw.get("phase") if isinstance(raw.get("phase"), str) else "",
        batch=_int("batch", minimum=1),
        tokens=_int("tokens", minimum=1),
        visible_history=_int("visible_history"),
    )
    if not shape.phase:
        raise InvalidInput(f"{label}.phase 必须是非空字符串，得到 {raw.get('phase')!r}")

    # 时长只有两种拼法，且必须恰好给一种：都写会让「以哪个为准」变成猜的。
    has_ns = "duration_ns" in raw
    has_s = "duration_s" in raw
    if has_ns == has_s:
        raise InvalidInput(
            f"{label}: 时长要恰好给一处——``duration_ns``（整数纳秒）或 "
            "``duration_s``（浮点秒，经 seconds_to_ns 转换）"
        )
    if has_ns:
        duration_ns = _int("duration_ns", minimum=1)
    else:
        # 浮点秒 → 整数纳秒的唯一转换点。取整到 0 会被 seconds_to_ns 拒掉：
        # 一个 4e-10 秒的时长变成 0 ns 之后，会变成一个「这个动作不花时间」的
        # 假事实，而不是一个错误。
        duration_ns = seconds_to_ns(raw["duration_s"], f"{label}.duration_s")

    provenance = raw.get("provenance", table_provenance)
    if provenance is None:
        raise InvalidInput(
            f"{label}: 没有 provenance，表级也没有；每个数字都要说明它从哪来"
        )
    if not isinstance(provenance, str) or not provenance:
        raise InvalidInput(f"{label}.provenance 必须是非空字符串，得到 {provenance!r}")

    if format_version == _MOE_VERSION:
        if provenance != "synthetic_layer":
            raise Unsupported(f"{label}: .6 仅接受 provenance='synthetic_layer'")
        if "physical_signature" in raw or "contract_hash" in raw:
            raise Unsupported(f"{label}: .6 synthetic MoE 不接受 physical_signature/contract_hash 字段")

    if "workspace_bytes" not in raw:
        # 不是「不写就是 0」：不写是没说过，而 0 是一句「这个任务不额外占草稿空间」
        # 的声明。两者混起来，漏掉的那条会变成一个偏小的内存账。
        raise InvalidInput(
            f"{label}.workspace_bytes 必须给出（没有额外草稿空间就写 0）"
        )

    measured = provenance == MEASURED_PROVENANCE
    accounting = raw.get("accounting", ACCOUNTING_PER_REQUEST)
    if measured and "accounting" not in raw:
        raise InvalidInput(f"{label}.accounting 必须明确声明 per_request")
    expert_group = format_version == _MOE_VERSION and kind in _EXPERT_GROUP_KINDS
    expected_accounting = ACCOUNTING_PER_EXPERT_GROUP if expert_group else ACCOUNTING_PER_REQUEST
    if expert_group and "accounting" not in raw:
        raise InvalidInput(f"{label}.accounting 必须明确声明 per_expert_group")
    if accounting != expected_accounting:
        raise Unsupported(f"{label}: accounting 必须是 {expected_accounting!r}")
    if expert_group and shape.visible_history != 0:
        raise Unsupported(f"{label}: expert group visible_history 必须为 0（不适用历史）")
    if measured and shape.batch != 1:
        raise Unsupported(f"{label}: measured_macro 仅支持 batch=1，拒绝整批成本重复计费")

    workspace_bytes = (
        None if measured and raw["workspace_bytes"] is None
        else _int("workspace_bytes")
    )
    workspace_scope = raw.get("workspace_scope")
    workspace_source = raw.get("workspace_source")
    if workspace_scope is not None and workspace_scope != WORKSPACE_SCOPE:
        raise Unsupported(f"{label}.workspace_scope 只支持 {WORKSPACE_SCOPE!r}")
    if workspace_source is not None and (
        not isinstance(workspace_source, str) or not workspace_source.strip()
    ):
        raise InvalidInput(f"{label}.workspace_source 必须是非空字符串")
    if measured:
        if workspace_scope != WORKSPACE_SCOPE:
            raise InvalidInput(f"{label}.workspace_scope 必须声明 {WORKSPACE_SCOPE!r}")
        if workspace_source is None:
            raise InvalidInput(f"{label}.workspace_source 必须给出测量证据来源")

    signature = raw.get("physical_signature")
    contract_hash = raw.get("contract_hash")
    _check_signature_pair(signature, contract_hash, label)
    if measured and signature is None:
        raise InvalidInput(f"{label}.physical_signature 必须给出 measured_macro 的物理边界")

    key = (kind, device, shape)
    cost = MacroCost(
        duration_ns=duration_ns,
        provenance=provenance,
        workspace_bytes=workspace_bytes,
        accounting=accounting,
        physical_signature=signature,
        contract_hash=contract_hash,
        workspace_scope=workspace_scope,
        workspace_source=workspace_source,
    )
    return key, cost


def parse_cost_table(raw: Any, *, where: str = "<costs>") -> M2CostTable:
    """读一张成本表。物理格式只在这里出现一次。"""
    if not isinstance(raw, dict):
        raise InvalidInput(f"{where} 必须是 JSON 对象，得到 {raw!r}")
    unknown = sorted(set(raw) - _TABLE_KEYS)
    if unknown:
        raise InvalidInput(f"{where}: 未知字段 {', '.join(unknown)}")
    version = raw.get("format_version")
    if "format_version" in raw:
        version = raw["format_version"]
        if not isinstance(version, str) or not version:
            raise InvalidInput(f"{where}.format_version 必须是非空字符串")
        if version not in {"m2-layer-0.1", "m2-layer-0.2", "m2-layer-0.3", "m2-layer-0.4", _HYBRID_VERSION, _MOE_VERSION}:
            raise Unsupported(f"{where}.format_version 不支持 {version!r}")
    if version not in {_HYBRID_VERSION, _MOE_VERSION} and "transfer_entries" in raw:
        raise Unsupported(f"{where}.transfer_entries 仅用于 {_HYBRID_VERSION}/{_MOE_VERSION}")
    if version in {_HYBRID_VERSION, _MOE_VERSION} and not isinstance(raw.get("transfer_entries"), list):
        raise InvalidInput(f"{where}.transfer_entries 必须显式给出 JSON 数组（可为空）")
    validation_only = raw.get("validation_only", False)
    if not isinstance(validation_only, bool):
        raise InvalidInput(f"{where}.validation_only 必须是 bool")

    source = raw.get("source")
    if not isinstance(source, str) or not source:
        raise InvalidInput(f"{where}.source 必须是非空字符串，得到 {source!r}")
    table_provenance = raw.get("provenance")
    if table_provenance is not None and not isinstance(table_provenance, str):
        raise InvalidInput(
            f"{where}.provenance 必须是字符串，得到 {table_provenance!r}"
        )
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise InvalidInput(f"{where}.entries 必须是非空 JSON 数组")

    entries: dict[tuple[str, str, MacroShape], MacroCost] = {}
    for index, entry in enumerate(entries_raw):
        key, cost = _parse_entry(entry, index, table_provenance, where, format_version=version)
        if key in entries:
            kind, device, shape = key
            raise InvalidInput(
                f"{where}.entries[{index}]: {kind!r}/{device!r}/{shape.key()} 重复"
            )
        entries[key] = cost
    if any(cost.provenance == MEASURED_PROVENANCE for cost in entries.values()) and "validation_only" not in raw:
        raise InvalidInput(f"{where}.validation_only must be explicitly declared for measured macro tables")
    transfers: dict[tuple[str, int], TransferEntry] = {}
    for index, entry in enumerate(raw.get("transfer_entries", [])):
        key, cost = _parse_transfer_entry(entry, index, where, format_version=version)
        if key in transfers:
            raise InvalidInput(f"{where}.transfer_entries[{index}]: {key!r} 重复")
        transfers[key] = cost
    return M2CostTable(
        source=source, entries=entries, validation_only=validation_only,
        transfers=transfers, format_version=version,
    )


def expand_costs_document(
    table: M2CostTable,
    requirements: Sequence[tuple[str, str, str, MacroShape]],
    *, physical_signatures: Mapping[str | tuple[str, str], dict[str, Any]] | None = None,
    transfer_provenances: Iterable[str] = (),
) -> tuple[str, dict[str, Any]]:
    """把这张表按 ``requirements`` 展开成内核的 ``costs`` 文档。

    ``requirements`` 是 ``(operation_id, kind, device, shape)`` 的序列，由展开器
    给出。返回 ``(source, costs_document)``，其中 ``source`` 是由各条成本的
    ``provenance`` 拼出来的标签——例如全是合成值时是 ``m2:synthetic_layer``。
    ``physical_signatures`` 可按 (operation_id, device) 提供，也兼容旧 operation_id
    键；前者优先，避免同一宏任务的 CPU/GPU 签名互相覆盖。实测条目必须有完全
    匹配的签名；合成物理层也要绑定签名，以免跨形状误用成本。
    ``transfer`` 由 ``expand_transfer_costs`` 单独生成。它返回的实际使用来源通过
    ``transfer_provenances`` 合并进 source，未使用的表项不会改变来源标签。
    """
    compute: dict[str, dict[str, dict[str, int]]] = {}
    provenances: set[str] = set(transfer_provenances)
    for operation_id, kind, device, shape in requirements:
        signatures = physical_signatures or {}
        selected_signature = (
            signatures[(operation_id, device)]
            if (operation_id, device) in signatures
            else signatures.get(operation_id)
        )
        cost = table.lookup(
            kind, device, shape,
            physical_signature=selected_signature,
        )
        provenances.add(cost.provenance)
        per_device = compute.setdefault(operation_id, {})
        if device in per_device:
            raise InvalidInput(
                f"展开要求里 {operation_id!r} 在 {device!r} 上有两条成本；"
                "同一 (算子, 设备) 只能有一条"
            )
        per_device[device] = {
            "duration_ns": cost.duration_ns,
            "workspace_bytes": cost.workspace_bytes,
        }
    source = f"{COST_SOURCE_PREFIX}:{'+'.join(sorted(provenances))}"
    return source, {"source": source, "compute": compute, "transfer": {}}


def expand_transfer_costs(
    table: M2CostTable,
    tensor_sizes: Mapping[str, int],
    allowed_routes: Mapping[str, Sequence[str]],
) -> tuple[set[str], dict[str, dict[str, dict[str, int]]]]:
    """Bind allowed tensor copies to exact storage-byte costs for the core.

    Returns ``(used_provenances, transfer_document)``. M2 .5 accepts only
    DRAM↔VRAM. M2 .6 additionally accepts DRAM↔SSD, always by exact size.
    Destination allocation and capacity are handled by explicit COPY actions,
    never by rounding bytes to capacity alignment or by a free transfer default.
    """
    if not isinstance(tensor_sizes, Mapping) or not isinstance(allowed_routes, Mapping):
        raise InvalidInput("tensor_sizes 与 allowed_routes 必须是映射")
    transfer: dict[str, dict[str, dict[str, int]]] = {}
    provenances: set[str] = set()
    for route, tensor_ids in allowed_routes.items():
        if not isinstance(route, str) or route not in ROUTES:
            raise Unsupported(f"M2 .5 未知搬运路线 {route!r}")
        if isinstance(tensor_ids, (str, bytes)) or not isinstance(tensor_ids, Sequence):
            raise InvalidInput(f"allowed_routes[{route!r}] 必须是张量 id 序列")
        if route not in table.allowed_transfer_routes and tensor_ids:
            raise Unsupported(f"M2 {table.format_version or 'legacy'} 尚未开放该 SSD 搬运路线")
        transfer[route] = {}
        for tensor_id in tensor_ids:
            if not isinstance(tensor_id, str) or not tensor_id:
                raise InvalidInput(f"allowed_routes[{route!r}] 的张量 id 必须是非空字符串")
            if tensor_id in transfer[route]:
                raise InvalidInput(f"allowed_routes[{route!r}] 重复张量 {tensor_id!r}")
            if tensor_id not in tensor_sizes:
                raise InvalidInput(f"搬运张量 {tensor_id!r} 缺少 storage_bytes")
            cost = table.lookup_transfer(route, tensor_sizes[tensor_id])
            transfer[route][tensor_id] = {"duration_ns": cost.duration_ns}
            provenances.add(cost.provenance)
    return provenances, transfer


def shapes_of(table: M2CostTable, kind: str, device: str) -> tuple[MacroShape, ...]:
    """诊断用：某类任务在某设备上已有的形状。"""
    return tuple(
        sorted(shape for (k, d, shape) in table.entries if k == kind and d == device)
    )
