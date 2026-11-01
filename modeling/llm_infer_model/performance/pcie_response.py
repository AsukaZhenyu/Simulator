"""Measured PCIe-state selection for the existing inclusive transfer law.

Generation and width select measured parameters; neither is converted into
theoretical bandwidth. An intercept is a dispatch/completion residual, not
identified CPU work. A byte slope is an effective completed-response slope,
not automatically a physical DMA supply. Therefore a copy that crosses a link
state change is unknown until a separately validated physical-stage model is
provided. This module measures/fits nothing and never reads future telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .parameters import InvalidModel, UnknownParameter, integer
from .transfer_response import TransferResponse


@dataclass(frozen=True)
class PCIeState:
    generation: int
    width: int

    def __post_init__(self):
        integer(self.generation, "observed PCIe generation", 1)
        integer(self.width, "observed PCIe width", 1)


@dataclass(frozen=True)
class PCIeResponseCell:
    state: PCIeState
    response: TransferResponse

    def __post_init__(self):
        if not isinstance(self.state, PCIeState) or not isinstance(self.response, TransferResponse):
            raise InvalidModel("explicit PCIe state and measured transfer response required")


@dataclass(frozen=True)
class PCIeStatePoint:
    at_ns: int
    state: PCIeState | None

    def __post_init__(self):
        integer(self.at_ns, "PCIe-state point")
        if self.state is not None and not isinstance(self.state, PCIeState):
            raise InvalidModel("unknown link state is None, never a guessed generation")


@dataclass(frozen=True)
class PCIeStatePlan:
    points: tuple[PCIeStatePoint, ...]
    valid_until_ns: int
    source_kind: str
    evidence: tuple[str, ...]

    def __post_init__(self):
        integer(self.valid_until_ns, "link-state horizon", 1)
        if (type(self.points) is not tuple or not self.points
                or any(not isinstance(p, PCIeStatePoint) for p in self.points)
                or self.points[0].at_ns != 0
                or any(a.at_ns >= b.at_ns for a, b in zip(self.points, self.points[1:]))
                or self.points[-1].at_ns > self.valid_until_ns):
            raise InvalidModel("ordered finite link-state plan starting at zero required")
        if self.source_kind not in {"declared_scenario", "diagnostic_replay"}:
            raise InvalidModel("future measured target state cannot masquerade as a declared plan")
        if type(self.evidence) is not tuple or not self.evidence or any(not isinstance(s, str) or not s for s in self.evidence):
            raise InvalidModel("link-state provenance required")

    def state_at(self, at_ns):
        integer(at_ns, "link-state query")
        if at_ns > self.valid_until_ns:
            raise UnknownParameter("link-state query outside declared horizon")
        state = None
        for point in self.points:
            if point.at_ns > at_ns:
                break
            state = point.state
        return state


@dataclass(frozen=True)
class PCIeResponseTable:
    context_sha256: str
    cells: tuple[PCIeResponseCell, ...]

    def __post_init__(self):
        if (not isinstance(self.context_sha256, str) or len(self.context_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.context_sha256)):
            raise InvalidModel("explicit hardware/runtime/preparation context SHA256 required")
        if type(self.cells) is not tuple or not self.cells or any(not isinstance(c, PCIeResponseCell) for c in self.cells):
            raise InvalidModel("nonempty explicit measured link-state cells required")
        if len({c.state for c in self.cells}) != len(self.cells):
            raise InvalidModel("duplicate link-state response")
        scopes = {(c.response.route, c.response.host_memory_contract, c.response.boundary) for c in self.cells}
        if len(scopes) != 1:
            raise InvalidModel("link-state cells cannot mix routes, memory contracts or action boundaries")

    def _cell(self, state, context_sha256):
        if context_sha256 != self.context_sha256:
            raise UnknownParameter("unmeasured transfer context; no runtime/preparation fallback")
        if not isinstance(state, PCIeState):
            raise UnknownParameter("PCIe generation/width unknown")
        for cell in self.cells:
            if cell.state == state:
                return cell
        raise UnknownParameter("unmeasured PCIe generation/width; no theoretical-rate fallback")

    def prediction(self, size_bytes, state, *, context_sha256, require_qualified=True):
        if type(require_qualified) is not bool:
            raise InvalidModel("qualification requirement must be an explicit boolean")
        cell = self._cell(state, context_sha256)
        # Reuse the existing transfer formula exactly. No generation ratio,
        # isolated GPU charge, response-to-throughput conversion or fitting.
        duration = cell.response.predict_ns(size_bytes, require_qualified=require_qualified)
        return dict(duration_ns=math.ceil(duration), unrounded_duration_ns=duration,
                    components=cell.response.components(size_bytes),
                    generation=state.generation, width=state.width,
                    boundary=cell.response.boundary, evidence=list(cell.response.evidence),
                    qualified=cell.response.qualified,
                    physical_DMA_service_identified=False,
                    intercept_is_active_CPU_work=False,
                    theoretical_generation_scaling_used=False,
                    device_activity_added_to_completed_response=False,
                    conditional_on_constant_state=True)

    def predict_at(self, size_bytes, plan: PCIeStatePlan, *, start_ns, context_sha256,
                   require_qualified=True, allow_diagnostic_replay=False):
        if not isinstance(plan, PCIeStatePlan):
            raise InvalidModel("explicit link-state plan required")
        if type(allow_diagnostic_replay) is not bool:
            raise InvalidModel("diagnostic replay permission must be an explicit boolean")
        integer(start_ns, "transfer start")
        if plan.source_kind == "diagnostic_replay" and not allow_diagnostic_replay:
            raise InvalidModel("future realized link state is diagnostic, not an advance forecast")
        state = plan.state_at(start_ns)
        result = self.prediction(size_bytes, state, context_sha256=context_sha256,
                                 require_qualified=require_qualified)
        end = start_ns + result["duration_ns"]
        if end > plan.valid_until_ns:
            raise UnknownParameter("predicted copy extends beyond link-state horizon")
        if any(start_ns < p.at_ns < end and p.state != state for p in plan.points):
            raise UnknownParameter("copy crosses link-state change; physical-stage parameters not established")
        return dict(result, start_ns=start_ns, end_ns=end,
                    input_role=plan.source_kind, diagnostic_replay=plan.source_kind == "diagnostic_replay",
                    whole_ahead_natural_state_evolution_predicted=False)

    def predict_from_observation(self, size_bytes, state, *, observed_at_ns, known_at_ns,
                                 start_ns, maximum_age_ns, context_sha256,
                                 assume_constant_for_invocation, require_qualified=True):
        for value, name in ((observed_at_ns, "state observation"), (known_at_ns, "state knowledge"),
                            (start_ns, "transfer start"), (maximum_age_ns, "maximum state age")):
            integer(value, name)
        if not observed_at_ns <= known_at_ns <= start_ns:
            raise InvalidModel("state was not known before the transfer; future observation rejected")
        if start_ns - observed_at_ns > maximum_age_ns:
            raise UnknownParameter("link-state observation too old for declared forecast scope")
        if assume_constant_for_invocation is not True:
            raise UnknownParameter("future link-state evolution unknown; explicit one-copy continuity assumption required")
        result = self.prediction(size_bytes, state, context_sha256=context_sha256,
                                 require_qualified=require_qualified)
        return dict(result, start_ns=start_ns, end_ns=start_ns + result["duration_ns"],
                    input_role="known_prefix_conditional_forecast", state_age_ns=start_ns - observed_at_ns,
                    future_target_observation_used=False,
                    whole_ahead_natural_state_evolution_predicted=False)
