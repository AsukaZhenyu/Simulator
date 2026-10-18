"""Incremental causal policy execution, independent of the offline fixed IR.

The driver may call a modeling session. A policy callback is handed only a
frozen observation, never the session, Scenario, CompiledInvocation or trace.
This is a callback interface, not yet a declarative policy language or search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Any

from llm_infer_model.m2.causal import CausalObservation
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT,
    ACTION_READ_SSD, ACTION_WRITE_SSD, Action,
)
from llm_infer_model.tensor.spec import COPY_READY, OP_NOT_STARTED, InvalidInput, Unsupported
from llm_infer_model.tensor.runtime import ExecutionCommand, dispatch_command


@dataclass(frozen=True)
class Decision:
    kind: str
    target_id: str | None = None
    device: str | None = None
    src: str | None = None
    dst: str | None = None
    location: str | None = None
    transfer_engine: str | None = None


@dataclass(frozen=True)
class ControlDecision:
    kind: str
    request_id: str | None = None
    until_ns: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    index: int
    observation_sha256: str
    observation: CausalObservation
    decision: Decision | ControlDecision | None
    action: Any | None
    accepted: bool
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class CausalRun:
    runtime_result: Any
    decisions: tuple[DecisionRecord, ...]
    termination_reason: str
    information_mode: str = "causal_observation"
    compilation_error: str | None = None

    @property
    def evaluation(self):
        return self.runtime_result.evaluation


def observation_document(observation: CausalObservation) -> dict:
    return asdict(observation)


def observation_hash(observation: CausalObservation) -> str:
    raw = json.dumps(observation_document(observation), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compile_action(decision: Decision, observation: CausalObservation) -> Action:
    """Lower one action using public objects only; runtime verifies legality."""
    if not isinstance(decision, Decision):
        raise InvalidInput("causal policy must return Decision or None")
    if not isinstance(observation, CausalObservation):
        raise InvalidInput("compile_decision requires an immutable CausalObservation")
    permitted = {
        ACTION_ADVANCE: (), ACTION_COMPUTE: ("target_id", "device"),
        ACTION_COPY: ("target_id", "src", "dst"),
        ACTION_READ_SSD: ("target_id",), ACTION_WRITE_SSD: ("target_id",),
        ACTION_EVICT: ("target_id", "location"),
    }
    if not isinstance(decision.kind, str) or not decision.kind:
        raise InvalidInput("causal decision kind must be a nonempty string")
    if decision.kind not in permitted:
        raise Unsupported(f"unsupported causal decision {decision.kind!r}")
    for name in ("target_id", "device", "src", "dst", "location"):
        value = getattr(decision, name)
        if name in permitted[decision.kind]:
            if not isinstance(value, str) or not value:
                raise InvalidInput(f"{decision.kind} requires explicit {name}")
        elif value is not None:
            raise InvalidInput(f"{decision.kind} does not use {name}")
    action = Action(**{key: value for key, value in asdict(decision).items() if key != "transfer_engine"})
    if decision.kind == ACTION_ADVANCE:
        return action
    if decision.kind == ACTION_COMPUTE:
        operation = next((op for op in observation.operations if op.id == decision.target_id), None)
        if operation is None:
            raise Unsupported(f"operation {decision.target_id!r} has not been revealed")
        if decision.device not in tuple(price.device for price in operation.prices):
            raise Unsupported(f"operation {decision.target_id!r} has no public exact {decision.device!r} cost")
    else:
        tensor = next((item for item in observation.tensors if item.id == decision.target_id), None)
        if tensor is None:
            raise Unsupported(f"tensor {decision.target_id!r} has not been revealed")
        if decision.kind == ACTION_EVICT:
            if decision.location not in ("dram", "vram"):
                raise Unsupported("release permits explicit dram or vram only")
        elif action.route not in tuple(price.route for price in tensor.transfers):
            raise Unsupported(f"tensor {decision.target_id!r} has no public exact route {action.route!r}")
    return action


def compile_decision(decision: Decision, observation: CausalObservation) -> Action:
    """Legacy Action-only lowering; engine metadata cannot be silently lost."""
    action = _compile_action(decision, observation)
    if decision.transfer_engine is not None:
        raise Unsupported("engine-bearing decisions require compile_command")
    if observation.services is not None and action.kind == ACTION_COPY:
        raise Unsupported("services COPY requires compile_command with an explicit engine")
    return action


def compile_command(decision: Decision, observation: CausalObservation) -> ExecutionCommand:
    """Lower through public facts, preserving an explicit PCIe engine choice."""
    if isinstance(decision, ControlDecision):
        from llm_infer_model.tensor.runtime import ControlCommand
        if not isinstance(observation, CausalObservation):
            raise InvalidInput("compile_command requires immutable CausalObservation")
        permitted = {"ACK_RECLAIM": {"request_id"}, "DECLINE_RECLAIM": {"request_id", "reason"},
                     "WAIT_UNTIL": {"until_ns", "request_id", "reason"}, "DECIDE": {"request_id", "reason"}}
        if not isinstance(decision.kind, str) or not decision.kind:
            raise InvalidInput("control decision kind must be a nonempty string")
        if decision.kind not in permitted:
            raise Unsupported(f"unsupported control decision {decision.kind!r}")
        for name in ("request_id", "until_ns", "reason"):
            value = getattr(decision, name)
            if value is not None and name not in permitted[decision.kind]:
                raise InvalidInput(f"{decision.kind} does not use {name}")
            if value is not None and name != "until_ns" and (not isinstance(value, str) or not value):
                raise InvalidInput(f"{decision.kind}.{name} must be nonempty")
        pending = observation.pending_reclaim
        if decision.kind in {"ACK_RECLAIM", "DECLINE_RECLAIM"} and (
                pending is None or decision.request_id != pending.id):
            raise Unsupported("control response requires the current public pending request")
        if decision.request_id is not None and (pending is None or decision.request_id != pending.id):
            raise Unsupported("control request_id has not been publicly revealed as pending")
        if decision.kind == "WAIT_UNTIL" and (type(decision.until_ns) is not int or decision.until_ns <= observation.t_ns):
            raise InvalidInput("WAIT_UNTIL requires an explicit public timer strictly after the current clock")
        if decision.kind == "DECLINE_RECLAIM" and decision.reason is None:
            raise InvalidInput("DECLINE_RECLAIM requires an explicit reason")
        if observation.control_costs is None:
            raise Unsupported("control decisions require an explicit control-cost contract")
        return ExecutionCommand(ControlCommand(**asdict(decision)))
    action = _compile_action(decision, observation)
    command = ExecutionCommand(action, decision.transfer_engine)
    if action.kind == ACTION_COPY and observation.services is not None:
        if command.transfer_engine is None:
            raise InvalidInput("services COPY requires an explicit transfer_engine")
        tensor = next(item for item in observation.tensors if item.id == action.target_id)
        price = next(item for item in tensor.transfers if item.route == action.route)
        if command.transfer_engine not in dict(price.engine_durations_ns):
            raise Unsupported("COPY has no public engine-specific service price")
    elif command.transfer_engine is not None:
        raise Unsupported("transfer_engine requires a services-enabled observation")
    return command


def run_causal(policy: Callable[[CausalObservation], Decision | None], session, *, max_decisions: int) -> CausalRun:
    if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions < 1:
        raise InvalidInput("max_decisions must be an explicit positive integer")
    records = []
    reason = "decision_budget_exhausted"
    compilation_error = None
    for index in range(max_decisions):
        observation = session.observe()
        if observation.error_code is not None:
            reason = "runtime_failure"
            break
        if observation.complete:
            reason = "goal_reached"
            break
        decision = policy(observation)
        if decision is None:
            reason = "policy_stopped"
            break
        try:
            command = compile_command(decision, observation)
            action = command.action
        except (InvalidInput, Unsupported) as error:
            compilation_error = str(error)
            code = "invalid_input" if isinstance(error, InvalidInput) else "unsupported_decision"
            records.append(DecisionRecord(index, observation_hash(observation), observation,
                                          decision if isinstance(decision, Decision) else None,
                                          None, False, code, str(error)))
            reason = "decision_rejected"
            break
        step = dispatch_command(session, command, source_index=index)
        records.append(DecisionRecord(index, observation_hash(observation), observation,
                                      decision, action, step.ok, step.error_code, step.error_message))
        if not step.ok:
            reason = "runtime_failure"
            break
    else:
        after = session.observe()
        if after.complete:
            reason = "goal_reached"
        elif after.error_code is not None:
            reason = "runtime_failure"
    return CausalRun(session.result(), tuple(records), reason, compilation_error=compilation_error)


class SerialDemandPolicy:
    """Simple fixed-device baseline using current demand only, without search."""

    def __init__(self, device: str):
        if device not in ("cpu", "gpu"):
            raise InvalidInput("SerialDemandPolicy requires explicit cpu or gpu")
        self.device = device

    @staticmethod
    def _load(tensor, destination):
        if tensor.status_at(destination) == COPY_READY:
            return None
        if destination == "vram":
            if tensor.status_at("dram") == COPY_READY:
                return Decision(ACTION_COPY, tensor.id, src="dram", dst="vram")
            if tensor.status_at("ssd") == COPY_READY:
                return Decision(ACTION_READ_SSD, tensor.id)
        elif destination == "dram":
            if tensor.status_at("vram") == COPY_READY:
                return Decision(ACTION_COPY, tensor.id, src="vram", dst="dram")
            if tensor.status_at("ssd") == COPY_READY:
                return Decision(ACTION_READ_SSD, tensor.id)
        elif destination == "ssd":
            if tensor.status_at("dram") == COPY_READY:
                return Decision(ACTION_WRITE_SSD, tensor.id)
            if tensor.status_at("vram") == COPY_READY:
                return Decision(ACTION_COPY, tensor.id, src="vram", dst="dram")
        return None

    def __call__(self, observation: CausalObservation) -> Decision | None:
        if observation.complete or observation.error_code is not None:
            return None
        if observation.running:
            return Decision(ACTION_ADVANCE)
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        destination = "dram" if self.device == "cpu" else "vram"
        for operation in observation.operations:
            if operation.status != OP_NOT_STARTED:
                continue
            if self.device not in tuple(price.device for price in operation.prices):
                raise Unsupported(f"baseline device {self.device} not allowed for {operation.id}")
            # Do not try to prepare an unproduced dependency. The visible graph
            # may list other independent branches before their inputs exist.
            if any(not any(status == COPY_READY for _, status in tensors[tensor].copies)
                   for tensor in operation.inputs):
                continue
            for tensor_id in operation.inputs:
                tensor = tensors[tensor_id]
                if tensor.status_at(destination) != COPY_READY:
                    decision = self._load(tensor, destination)
                    if decision is None:
                        raise Unsupported(f"no ready source for current input {tensor_id!r}")
                    return decision
            from llm_infer_model.m2.causal import compute_service_available
            if not compute_service_available(observation, self.device):
                return Decision(ACTION_ADVANCE)
            return Decision(ACTION_COMPUTE, operation.id, device=self.device)
        for tensor_id, location in observation.deliveries:
            if tensors[tensor_id].status_at(location) != COPY_READY:
                decision = self._load(tensors[tensor_id], location)
                if decision is not None:
                    return decision
        # ADVANCE lets modeling reveal a forthcoming environment event without
        # disclosing its time. No event/work left is a loud runtime failure.
        return Decision(ACTION_ADVANCE)
