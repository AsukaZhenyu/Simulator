"""Model-owned measured MoE carrier and replay, with legacy ingestion shims.

New raw-evidence conversion lives in llm_infer_profile.moe_measurements.
Imports remain cheap: profiling is loaded only when a caller explicitly asks
for legacy measured-data ingestion. Ordinary simulation needs no profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tensor.engine import Action, ACTION_ADVANCE, ACTION_COMPUTE, evaluate_mapping
from .moe_ffn_contract import stage_names


@dataclass(frozen=True)
class MeasuredMoEFFN:
    case: dict[str, Any]
    record: dict[str, Any]
    scenario: Any
    workload_document: dict[str, Any]
    scenario_document: dict[str, Any]
    bindings: dict[str, Any]
    validation_only: bool


def _physical_tensor(name, port, *, initial):
    """Compatibility for the former private raw-port conversion helper."""
    from llm_infer_profile.moe_measurements import _physical_tensor as convert
    return convert(name, port, initial=initial)


def compile_measured_ffn(record: dict, hardware: dict, *, allow_unstable=False) -> MeasuredMoEFFN:
    """Compatibility entry point; profiling owns raw samples and median prices."""
    from llm_infer_profile.moe_measurements import compile_measured_ffn as convert
    return convert(record, hardware, allow_unstable=allow_unstable)


def plan_measured_ffn(compiled: MeasuredMoEFFN):
    return tuple(action for stage in stage_names(compiled.case)[:-1]
                 for action in (Action(ACTION_COMPUTE,stage,device="cpu"),Action(ACTION_ADVANCE)))


def replay_measured_ffn(compiled: MeasuredMoEFFN):
    return evaluate_mapping(compiled.scenario,plan_measured_ffn(compiled))
