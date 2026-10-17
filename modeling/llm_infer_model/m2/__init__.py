"""M2：一个 decoder layer 的一次调用。

M2 的单位不是「一个模型」也不是「一次请求」，而是**一层的一次调用**
（``M2_LAYER_DESIGN.md`` §1）。两个正交维度——注意力
（``full_causal`` / ``sliding_window`` / ``recurrent_linear``）与 FFN
（``dense`` / ``moe``）——给出六种组合；本版本实现 ``full_causal + dense``
CPU、CUDA 或显式 CPU/CUDA 混合，另有合成 full_causal + MoE 路由闭环；其余在 ``validate_invocation`` 里以 ``Unsupported`` 拒绝，而不是当成
没写过的字段忽略掉。

四个模块，各管一件事：

:mod:`~llm_infer_model.m2.spec`
    三份创作文档（layer / invocation / costs 的持有者）与它们的校验。**只认文档
    里写了的东西**，缺一项就拒绝——这一层的每个默认值都会安静地改变答案。
:mod:`~llm_infer_model.m2.costs`
    逐形状的宏任务成本表，以及它到内核 ``costs`` 文档的展开。物理格式只在这里
    出现一次，因为将来换的是它。
:mod:`~llm_infer_model.m2.expand`
    确定性展开成内核能回放的 (workload, scenario)。
:mod:`~llm_infer_model.m2.runner`
    回放一次，并把内核的三本账读回成 ``state_out``。

典型用法::

    from llm_infer_model.m2 import load_and_compile, run_invocation, plan_for

    compiled = load_and_compile("examples/m2/fcd-prefill-3.json")
    result = run_invocation(compiled)
    assert result.makespan_ns == 10_000_000
    state_out = result.state_out          # 下一次调用的 state_in
"""

from __future__ import annotations

from .cuda_physical import CUDA_FORMAT_VERSION, CudaDensePhysical
from .hybrid_physical import HYBRID_FORMAT_VERSION, HybridDensePhysical
from .moe_spec import MOE_FORMAT_VERSION, STATE_CACHE_FORMAT_VERSION, RouteTraceEntry
from .moe_measured import MeasuredMoEFFN, compile_measured_ffn, plan_measured_ffn, replay_measured_ffn
from .costs import (
    COST_SOURCE_PREFIX,
    M2CostTable,
    MacroCost,
    MacroShape,
    expand_costs_document,
    expand_transfer_costs,
    parse_cost_table,
    shapes_of,
)
from .expand import (
    CompiledInvocation,
    CompiledTask,
    compile_invocation,
    load_and_compile,
    plan_for,
)
from .runner import LayerInvocationResult, run_invocation, state_out_from_result
from .spec import (
    ATTENTION_FULL_CAUSAL,
    ATTENTION_KINDS,
    ATTENTION_RECURRENT_LINEAR,
    ATTENTION_SLIDING_WINDOW,
    FFN_DENSE,
    FFN_KINDS,
    FFN_MOE,
    FORMAT_VERSION,
    PHASE_DECODE,
    PHASE_PREFILL,
    PHASES,
    PROVENANCE_KINDS,
    PROVENANCE_MEASURED,
    PROVENANCE_SYNTHETIC,
    SLICE_ATTENTION,
    SLICE_DEVICE,
    SLICE_FFN,
    STAGE_IN,
    TASK_KINDS,
    Hardware,
    HiddenOutRef,
    Invocation,
    KvBlockRef,
    LayerDescriptor,
    RequestInput,
    RequestState,
    StateDocument,
    StateIn,
    StateOut,
    TaskSpec,
    WeightRef,
    WeightStateRef,
    block_id,
    hidden_bytes,
    invocation_hidden_bytes,
    load_invocation,
    load_layer,
    load_state_document,
    op_id,
    parse_invocation_document,
    parse_layer_document,
    parse_state_document,
    parse_state_in,
    state_in_from_state_out,
    tensor_id,
    to_state_document,
    to_state_in_document,
    validate_invocation,
    weight_id,
)

__all__ = [
    "ATTENTION_FULL_CAUSAL",
    "ATTENTION_KINDS",
    "ATTENTION_RECURRENT_LINEAR",
    "ATTENTION_SLIDING_WINDOW",
    "COST_SOURCE_PREFIX",
    "CUDA_FORMAT_VERSION",
    "CudaDensePhysical",
    "HYBRID_FORMAT_VERSION",
    "HybridDensePhysical",
    "MOE_FORMAT_VERSION",
    "STATE_CACHE_FORMAT_VERSION",
    "RouteTraceEntry",
    "MeasuredMoEFFN",
    "compile_measured_ffn",
    "plan_measured_ffn",
    "replay_measured_ffn",
    "CompiledInvocation",
    "CompiledTask",
    "FFN_DENSE",
    "FFN_KINDS",
    "FFN_MOE",
    "FORMAT_VERSION",
    "Hardware",
    "HiddenOutRef",
    "Invocation",
    "KvBlockRef",
    "LayerDescriptor",
    "LayerInvocationResult",
    "M2CostTable",
    "MacroCost",
    "MacroShape",
    "PHASE_DECODE",
    "PHASE_PREFILL",
    "PHASES",
    "PROVENANCE_KINDS",
    "PROVENANCE_MEASURED",
    "PROVENANCE_SYNTHETIC",
    "RequestInput",
    "RequestState",
    "SLICE_ATTENTION",
    "SLICE_DEVICE",
    "SLICE_FFN",
    "STAGE_IN",
    "StateDocument",
    "StateIn",
    "StateOut",
    "TASK_KINDS",
    "TaskSpec",
    "WeightRef",
    "WeightStateRef",
    "block_id",
    "compile_invocation",
    "expand_costs_document",
    "expand_transfer_costs",
    "hidden_bytes",
    "invocation_hidden_bytes",
    "load_and_compile",
    "load_invocation",
    "load_layer",
    "load_state_document",
    "op_id",
    "parse_cost_table",
    "parse_invocation_document",
    "parse_layer_document",
    "parse_state_document",
    "parse_state_in",
    "plan_for",
    "run_invocation",
    "shapes_of",
    "state_in_from_state_out",
    "state_out_from_result",
    "tensor_id",
    "to_state_document",
    "to_state_in_document",
    "validate_invocation",
    "weight_id",
]
