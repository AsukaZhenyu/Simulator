"""Composable expert priorities over public observations, never private graphs.

These functions rank already revealed work and declared paths. They submit no
commands, reserve no memory, predict no future router rows and create no runtime.
The caller retains dependency/capacity checks and commits a region's device only
after returning its first real preparation or compute command.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from llm_infer_model.m2.causal import (
    CausalObservation, RevealedRoute, compute_service_available,
    current_compute_duration_ns,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .policy_ir import EXPERT_COMPUTE_KINDS, EXPERT_ORDER_RULES

DEVICE_RULES = frozenset({"fixed", "resident_first", "current_cost", "fiddler_paths", "hybrimoe_queues"})


@dataclass(frozen=True)
class ExpertSchedulingSpec:
    order_rule: str = "graph_order"
    device_rule: str = "fixed"
    selection_unit: str = "expert_region"

    def __post_init__(self):
        if not isinstance(self.order_rule, str) or self.order_rule not in EXPERT_ORDER_RULES:
            raise Unsupported(f"expert order_rule must be one of {sorted(EXPERT_ORDER_RULES)}")
        if not isinstance(self.device_rule, str) or self.device_rule not in DEVICE_RULES:
            raise Unsupported(f"expert device_rule must be one of {sorted(DEVICE_RULES)}")
        if self.selection_unit != "expert_region":
            raise Unsupported("expert scheduling selection_unit must be expert_region")


def _is_expert(operation):
    return (operation.kind in EXPERT_COMPUTE_KINDS and
            (type(operation.expert_id) is int or operation.expert_id == "shared"))


def _revealed_rows(observation, history):
    if not isinstance(history, tuple) or any(not isinstance(row, RevealedRoute) for row in history):
        raise InvalidInput("expert scheduling history must be an immutable tuple of revealed routes")
    rows = {}
    for row in history + observation.route_history:
        if not isinstance(row, RevealedRoute):
            raise InvalidInput("expert scheduling accepts only revealed route observations")
        key = (row.request_id, row.layer_id, row.absolute_position)
        previous = rows.get(key)
        if previous is not None and previous != row:
            raise InvalidInput("expert scheduling history contradicts a revealed route")
        rows[key] = row
    return rows


def order_expert_operations(observation, spec, *, history=()):
    """Return a priority order, preserving nonexpert slots and regional stages.

    This is not a rewritten dependency graph. Runtime readiness still gates each
    command. graph_order is an exact compatibility identity. Other orders require
    each candidate's precise rows to have been revealed by a router. History is
    an explicitly supplied known past, deduplicated with the current snapshot.
    """
    if not isinstance(observation, CausalObservation) or not isinstance(spec, ExpertSchedulingSpec):
        raise InvalidInput("expert ordering requires a public observation and ExpertSchedulingSpec")
    if spec.order_rule == "graph_order":
        return observation.operations
    rows = _revealed_rows(observation, history)
    groups = {}
    for operation in observation.operations:
        if not _is_expert(operation):
            continue
        if not operation.region_id and operation.kind not in {"expert_ffn", "shared_expert_ffn"}:
            raise Unsupported("expert priority needs complete revealed region identities for staged work")
        if (type(operation.token_count) is not int or operation.token_count < 1 or
                len(operation.work_items) != operation.token_count or
                len(set(operation.work_items)) != len(operation.work_items)):
            raise Unsupported("expert priority requires exact, distinct revealed token rows")
        for request, position in operation.work_items:
            row = rows.get((request, operation.layer_id, position))
            if row is None or (operation.expert_id != "shared" and operation.expert_id not in row.expert_ids):
                raise InvalidInput("expert priority cannot use an unrevealed or unselected router row")
        key = (operation.invocation_id, operation.region_id or operation.id)
        group = groups.setdefault(key, [])
        if group and (group[0].expert_id != operation.expert_id or group[0].layer_id != operation.layer_id or
                      group[0].work_items != operation.work_items):
            raise InvalidInput("expert region priority requires one consistent expert and token-row scope")
        group.append(operation)
    frequency = Counter((row.layer_id, expert) for row in rows.values() for expert in row.expert_ids)
    original = {key: index for index, key in enumerate(groups)}
    def rank(item):
        key, operations = item
        first = operations[0]
        score = (first.token_count if spec.order_rule == "token_count_desc" else
                 frequency[(first.layer_id, first.expert_id)])
        return (-score, original[key])
    ranked = iter(operation for _, operations in sorted(groups.items(), key=rank) for operation in operations)
    return tuple(next(ranked) if _is_expert(operation) else operation for operation in observation.operations)


def expert_device_ranking(observation, spec, paths, *, preferred_device, cached_gpu=False):
    """Return feasible CPU/GPU candidates using current public rates only.

    ``paths`` are the existing complete-region estimate dictionaries, derived
    from public stage prices and explicitly selected isolated COPY prices. Each
    path must state physical/budget feasibility. No unseen state, measured
    future busy interval or forecast service recovery is accepted. Busy compute
    slots are not forecasts: _demand decides whether preparation/compute may
    proceed. The returned order is tentative until a real command is issued.
    """
    if not isinstance(observation, CausalObservation) or not isinstance(spec, ExpertSchedulingSpec):
        raise InvalidInput("expert device ranking requires a public observation and ExpertSchedulingSpec")
    if preferred_device not in {"cpu", "gpu"} or not isinstance(cached_gpu, bool):
        raise InvalidInput("expert device ranking requires an explicit preferred cpu/gpu and residency boolean")
    if spec.device_rule in {"fiddler_paths", "hybrimoe_queues"}:
        raise Unsupported("paper-specific expert queues retain their existing explicit dispatcher")
    candidates, seen = [], set()
    for path in paths:
        if not isinstance(path, dict):
            raise InvalidInput("expert paths must contain explicit public estimate dictionaries")
        device = path.get("device")
        if device not in {"cpu", "gpu"} or device in seen:
            raise InvalidInput("expert paths require distinct cpu/gpu devices")
        seen.add(device)
        for flag in ("budget_feasible", "physical_feasible"):
            if type(path.get(flag)) is not bool:
                raise InvalidInput(f"expert path requires explicit {flag}")
        for field in ("compute_ns", "weight_transfer_ns", "selected_input_transfer_ns", "output_return_ns"):
            if type(path.get(field)) is not int or path[field] < 0:
                raise InvalidInput(f"expert path requires nonnegative integer {field}; unknown is not zero")
        if not path["budget_feasible"] or not path["physical_feasible"] or not compute_service_available(observation, device):
            continue
        if spec.device_rule == "fixed" and device != preferred_device:
            continue
        current = current_compute_duration_ns(path["compute_ns"], observation, device)
        total = current + path["weight_transfer_ns"] + path["selected_input_transfer_ns"] + path["output_return_ns"]
        resident_priority = 0 if spec.device_rule == "resident_first" and cached_gpu and device == "gpu" else 1
        candidates.append((resident_priority, total, device != preferred_device, device))
    return tuple(entry[-1] for entry in sorted(candidates))


# Alias keeps the integration call site readable in a policy's _devices method.
rank_expert_devices = expert_device_ranking
