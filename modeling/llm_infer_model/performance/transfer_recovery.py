"""Size-aware, preparation-conditioned reuse of the existing recovery law.

A completed COPY response has a measured byte-dependent baseline and bounded
readiness impact. A known preceding preparation supplies an explicit initial
impact; it does not imply zero debt. Idle fills the remaining impact capacity,
and predicted completed copies recover it. These are empirical response states,
not identified cache entries, power states, CPU work or physical DMA service.
All estimation stays in profiling. Historical fixed-payload providers remain
unchanged; this optional adapter reuses their exact evolution equations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .idle_recovery import (IdleRecoveryScope, IdleRecoveryParameters,
                            IdleRecoveryResponse, ReadinessDebt)
from .parameters import InvalidModel, UnknownParameter, integer
from .prepared_transfer import COMPLETION_SEMANTIC
from .transfer_response import TransferResponse


def _sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(x not in "0123456789abcdef" for x in value)):
        raise InvalidModel(label + " requires explicit SHA256 provenance")


def _number(value, label, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise InvalidModel(label + " requires an explicit finite nonnegative value")


def _evidence(value):
    if type(value) is not tuple or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise InvalidModel("nonempty explicit measurement evidence required")


@dataclass(frozen=True)
class AffineResponseImpact:
    fixed_ns: float
    ns_per_byte: float
    evidence: tuple[str, ...]

    def __post_init__(self):
        _number(self.fixed_ns, "fixed completed-response impact")
        _number(self.ns_per_byte, "byte-dependent completed-response impact")
        _evidence(self.evidence)

    def at(self, size_bytes):
        integer(size_bytes, "transfer bytes", 1)
        value = self.fixed_ns + self.ns_per_byte * size_bytes
        _number(value, "completed-response impact at the requested size")
        return value


@dataclass(frozen=True)
class TransferRecoveryScope:
    context_sha256: str
    preparation_sha256: str
    route: str
    memory_mode: str
    maximum_idle_ns: int

    def __post_init__(self):
        _sha(self.context_sha256, "runtime/response boundary")
        _sha(self.preparation_sha256, "semantic preparation contract")
        if self.route not in {"dram->vram", "vram->dram"} or self.memory_mode not in {"registered", "pageable"}:
            raise InvalidModel("explicit measured route and host-memory category required")
        integer(self.maximum_idle_ns, "declared idle envelope", 1)


@dataclass(frozen=True)
class SizeAwareRecoveryResponse:
    scope: TransferRecoveryScope
    baseline: TransferResponse
    maximum_impact: AffineResponseImpact
    prepared_impact: AffineResponseImpact
    idle_fill_tau_ns: float
    active_recovery_tau_ns: float
    evidence: tuple[str, ...]
    qualified: bool = False

    def __post_init__(self):
        if not isinstance(self.scope, TransferRecoveryScope) or not isinstance(self.baseline, TransferResponse):
            raise InvalidModel("explicit recovery scope and completed-transfer baseline required")
        if not isinstance(self.maximum_impact, AffineResponseImpact) or not isinstance(self.prepared_impact, AffineResponseImpact):
            raise InvalidModel("explicit maximum and prepared response-impact curves required")
        if self.baseline.route != self.scope.route or self.baseline.host_memory_contract != self.scope.memory_mode:
            raise InvalidModel("baseline route/host-memory contract differs")
        if self.baseline.boundary != COMPLETION_SEMANTIC:
            raise InvalidModel("baseline must price the inclusive host-completed transfer boundary; device-event prices cannot substitute")
        for name in ("idle_fill_tau_ns", "active_recovery_tau_ns"):
            _number(getattr(self, name), name, positive=True)
        # Affine differences have their extrema at the envelope endpoints.
        for size in (self.baseline.minimum_bytes, self.baseline.maximum_bytes):
            maximum, initial = self.maximum_impact.at(size), self.prepared_impact.at(size)
            if maximum <= 0 or initial > maximum:
                raise InvalidModel("prepared response impact must lie within the explicit positive capacity at every accepted size")
            _number(self.baseline.predict_ns(size, require_qualified=False),
                    "completed-transfer baseline at accepted size", positive=True)
        _evidence(self.evidence)
        if type(self.qualified) is not bool:
            raise InvalidModel("independent qualification flag must be explicit")

    def _response(self, size_bytes, *, require_qualified):
        if type(require_qualified) is not bool:
            raise InvalidModel("explicit qualification requirement required")
        if require_qualified and not self.qualified:
            raise UnknownParameter("size/history response has not passed independent qualification")
        integer(size_bytes, "transfer bytes", 1)
        if not self.baseline.minimum_bytes <= size_bytes <= self.baseline.maximum_bytes:
            raise UnknownParameter("transfer size outside the measured interpolation envelope")
        # Qualification belongs to the composed response and its exact scope.
        # A baseline component need not be separately usable as a complete law.
        steady = self.baseline.predict_ns(size_bytes, require_qualified=False)
        scope = IdleRecoveryScope(self.scope.context_sha256, self.scope.route,
                                  size_bytes, self.scope.memory_mode, self.scope.maximum_idle_ns)
        parameters = IdleRecoveryParameters(steady, self.maximum_impact.at(size_bytes),
            self.idle_fill_tau_ns, self.active_recovery_tau_ns,
            self.evidence + self.baseline.evidence + self.maximum_impact.evidence + self.prepared_impact.evidence)
        return IdleRecoveryResponse(scope, parameters)

    def _coordinate(self, *, context_sha256, preparation_sha256):
        if context_sha256 != self.scope.context_sha256:
            raise UnknownParameter("runtime/response boundary is unmeasured; no cross-context fallback")
        if preparation_sha256 != self.scope.preparation_sha256:
            raise UnknownParameter("preceding preparation is unmeasured; no automatic warm/zero initialization")

    def prepared_state(self, size_bytes, *, context_sha256, preparation_sha256,
                       require_qualified=True):
        self._coordinate(context_sha256=context_sha256, preparation_sha256=preparation_sha256)
        response = self._response(size_bytes, require_qualified=require_qualified)
        return ReadinessDebt(self.prepared_impact.at(size_bytes) / response.parameters.maximum_extra_response_ns)

    def predict_burst(self, size_bytes, requested_idle_ns, copies, *, context_sha256, preparation_sha256,
                      require_qualified=True):
        self._coordinate(context_sha256=context_sha256, preparation_sha256=preparation_sha256)
        response = self._response(size_bytes, require_qualified=require_qualified)
        initial = self.prepared_state(size_bytes, context_sha256=context_sha256,
                                      preparation_sha256=preparation_sha256,
                                      require_qualified=require_qualified)
        result = response.predict_burst(requested_idle_ns, copies, initial_state=initial)
        return dict(result, payload_bytes=size_bytes, initial_prepared_impact_ns=self.prepared_impact.at(size_bytes),
                    maximum_response_impact_ns=self.maximum_impact.at(size_bytes),
                    baseline_completed_response_ns=response.parameters.steady_response_ns,
                    context_sha256=context_sha256,
                    preparation_sha256=preparation_sha256,
                    initialization="profiled nonzero-or-zero preparation impact, never an inferred physical reset",
                    profile_independently_qualified=self.qualified,
                    event_or_device_work_added=False, physical_CPU_or_DMA_time_identified=False,
                    switching_and_idle_impacts_added_independently=False)

    def continuous_burst(self, size_bytes, requested_idle_ns, copies, *, context_sha256,
                         preparation_sha256):
        """Smooth form for profiling estimators, using the same model equation."""
        integer(copies, "burst copies", 1)
        if copies > 100000:
            raise InvalidModel("burst exceeds declared safety envelope")
        self._coordinate(context_sha256=context_sha256, preparation_sha256=preparation_sha256)
        response = self._response(size_bytes, require_qualified=False)
        initial = self.prepared_state(size_bytes, context_sha256=context_sha256,
            preparation_sha256=preparation_sha256, require_qualified=False)
        return response.continuous_burst(requested_idle_ns, copies, initial_state=initial)
