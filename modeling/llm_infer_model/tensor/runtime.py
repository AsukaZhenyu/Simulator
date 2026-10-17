"""Shared execution envelope for fixed programs and causal policy decisions.

An engine is execution metadata, not a new tensor action. The modeling session
still owns capacity, dependencies, copy locks and shared-link arbitration.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import ACTION_ADVANCE, ACTION_COPY, Action
from .spec import InvalidInput, Unsupported


@dataclass(frozen=True)
class ControlCommand:
    """A serial controller operation, independent of tensor actions.

    WAIT_UNTIL is a public timer upper bound; earlier runtime events interrupt
    it so the caller can observe again. It is never a private-trace lookup.
    """
    kind: str
    request_id: str | None = None
    until_ns: int | None = None
    reason: str | None = None

    def __post_init__(self):
        if self.kind not in ("ACK_RECLAIM", "DECLINE_RECLAIM", "WAIT_UNTIL", "DECIDE"):
            raise Unsupported("unsupported control command")
        if self.request_id is not None and (not isinstance(self.request_id, str) or not self.request_id.strip()):
            raise InvalidInput("control request_id must be a nonempty string")
        if self.kind in ("ACK_RECLAIM", "DECLINE_RECLAIM") and self.request_id is None:
            raise InvalidInput("reclaim responses require request_id")
        if self.kind == "WAIT_UNTIL":
            if type(self.until_ns) is not int or self.until_ns < 0:
                raise InvalidInput("WAIT_UNTIL requires a non-negative integer until_ns")
        elif self.until_ns is not None:
            raise InvalidInput("until_ns applies only to WAIT_UNTIL")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise InvalidInput("control reason must be a nonempty string")
        if self.kind == "DECLINE_RECLAIM" and self.reason is None:
            raise InvalidInput("DECLINE_RECLAIM requires an explicit reason")


@dataclass(frozen=True)
class ExecutionCommand:
    action: Action | ControlCommand
    transfer_engine: str | None = None

    def __post_init__(self):
        if not isinstance(self.action, (Action, ControlCommand)):
            raise InvalidInput("ExecutionCommand requires Action or ControlCommand")
        if self.transfer_engine is not None:
            if not isinstance(self.transfer_engine, str) or self.transfer_engine not in ("ce", "sm"):
                raise Unsupported("transfer_engine must be explicit ce or sm")
            if self.action.kind != ACTION_COPY:
                raise InvalidInput("transfer_engine applies only to PCIe COPY")


def dispatch_command(session, command: ExecutionCommand, *, source_index: int | None = None):
    """Execute one explicit command without inferring engines or legality."""
    if not isinstance(command, ExecutionCommand):
        raise InvalidInput("dispatch_command requires ExecutionCommand")
    if isinstance(command.action, ControlCommand):
        return session.submit_control(command.action, source_index=source_index)
    if command.action.kind == ACTION_ADVANCE:
        return session.advance(source_index=source_index)
    if command.transfer_engine is not None:
        submit_transfer = getattr(session, "submit_transfer", None)
        if submit_transfer is None:
            raise Unsupported("transfer engine requires a services-enabled runtime")
        return submit_transfer(command.action, engine=command.transfer_engine,
                               source_index=source_index)
    return session.submit(command.action, source_index=source_index)


def create_runtime(scenario, *, services=None, environment_events=(), visibility=None,
                   reclaim_requests=(), control_costs=None):
    """Model-owned factory shared by single-layer and global causal adapters.

    Only the resulting observation is passed to a policy. This factory never
    makes a private graph, visibility projector or environment queue public.
    """
    from .environment import SimulationSession
    if services is None:
        return SimulationSession(scenario, environment_events=environment_events,
                                 visibility=visibility, reclaim_requests=reclaim_requests,
                                 control_costs=control_costs)
    from .services import ServiceSession
    return ServiceSession(scenario, services, environment_events=environment_events,
                          visibility=visibility, reclaim_requests=reclaim_requests,
                          control_costs=control_costs)
