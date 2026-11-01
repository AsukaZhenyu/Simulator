"""Causal composition of inclusive runtime API and intervening host responses.

Each measured stage is one completed response, never active CPU time. Its
internal GPU work/wait must not be added again. The existing event engine and
the existing g law perform all time integration. Profiling alone supplies
stage coordinates, prices, state support and any recovery parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .parameters import InvalidModel, UnknownParameter, integer
from .planned_response import PlannedCompletedResponse, STATES


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise InvalidModel(name + " requires an explicit SHA256")


def _evidence(value):
    if (type(value) is not tuple or not value
            or any(not isinstance(x, str) or not x.strip() for x in value)):
        raise InvalidModel("nonempty response evidence tuple required")


@dataclass(frozen=True)
class HostPlacement:
    target_cpu_mask: int
    cpu_actor_mask: int
    target_physical_cores: tuple[int, ...]
    actor_physical_cores: tuple[int, ...]
    topology_sha256: str

    def __post_init__(self):
        for name in ("target_cpu_mask", "cpu_actor_mask"):
            integer(getattr(self, name), name, 1)
        for name in ("target_physical_cores", "actor_physical_cores"):
            value = getattr(self, name)
            if (type(value) is not tuple or not value
                    or any(type(x) is not int or x < 0 for x in value)
                    or len(set(value)) != len(value)):
                raise InvalidModel("verified nonempty physical-core mapping required")
        _sha(self.topology_sha256, "topology provenance")

    @property
    def physical_core_overlap(self):
        # The mapping is supplied from verified topology, not inferred from
        # logical CPU bit numbers. This relation never becomes a service rate.
        return bool(set(self.target_physical_cores) & set(self.actor_physical_cores))


@dataclass(frozen=True)
class RuntimeScope:
    context_sha256: str
    placement: HostPlacement
    boundary: str = "inclusive_host_response_partition"

    def __post_init__(self):
        _sha(self.context_sha256, "hardware/runtime/template context")
        if not isinstance(self.placement, HostPlacement):
            raise InvalidModel("explicit host placement required")
        if self.boundary != "inclusive_host_response_partition":
            raise InvalidModel("runtime responses cannot be relabelled physical CPU/GPU work")


@dataclass(frozen=True)
class RuntimeStage:
    id: str
    coordinate_sha256: str
    role: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidModel("stage occurrence id required")
        _sha(self.coordinate_sha256, "runtime semantic coordinate")
        if not isinstance(self.role, str) or self.role not in {"api", "gap"}:
            raise InvalidModel("only inclusive API responses and disjoint host gaps are accepted")


@dataclass(frozen=True)
class RuntimeCell:
    coordinate_sha256: str
    role: str
    state: tuple[int, int]
    mean_ns: int | Fraction | None
    evidence: tuple[str, ...]

    def __post_init__(self):
        RuntimeStage("cell", self.coordinate_sha256, self.role)
        if (type(self.state) is not tuple or len(self.state) != 2
                or any(type(x) is not int or x not in (0, 1) for x in self.state)):
            raise InvalidModel("explicit offered CPU/GPU 0/1 state required")
        if self.mean_ns is not None:
            if type(self.mean_ns) not in (int, Fraction) or self.mean_ns < 0:
                raise InvalidModel("nonnegative exact response nanoseconds or unknown required")
            object.__setattr__(self, "mean_ns", Fraction(self.mean_ns))
        _evidence(self.evidence)


@dataclass(frozen=True)
class RuntimeProfile:
    scope: RuntimeScope
    cells: tuple[RuntimeCell, ...]
    evidence: tuple[str, ...]
    qualified: bool = False

    def __post_init__(self):
        if not isinstance(self.scope, RuntimeScope):
            raise InvalidModel("explicit runtime profile scope required")
        if type(self.cells) is not tuple or not self.cells or any(not isinstance(c, RuntimeCell) for c in self.cells):
            raise InvalidModel("nonempty explicit runtime cells required")
        keys = [(c.coordinate_sha256, c.role, c.state) for c in self.cells]
        if len(keys) != len(set(keys)):
            raise InvalidModel("duplicate runtime state cell")
        _evidence(self.evidence)
        if type(self.qualified) is not bool:
            raise InvalidModel("explicit independent-qualification flag required")


@dataclass(frozen=True)
class RuntimeProgram:
    scope: RuntimeScope
    stages: tuple[RuntimeStage, ...]
    evidence: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.scope, RuntimeScope):
            raise InvalidModel("runtime program scope required")
        if type(self.stages) is not tuple or not self.stages or any(not isinstance(s, RuntimeStage) for s in self.stages):
            raise InvalidModel("ordered nonempty API/gap partition required")
        if len({s.id for s in self.stages}) != len(self.stages):
            raise InvalidModel("duplicate runtime stage occurrence")
        if (len(self.stages) < 3 or len(self.stages) % 2 != 1
                or any(s.role != ("gap" if i % 2 == 0 else "api")
                       for i, s in enumerate(self.stages))):
            raise InvalidModel("complete prefix/API/intercall/suffix partition required; explicit zero gaps if established")
        roles = {}
        for stage in self.stages:
            if stage.coordinate_sha256 in roles and roles[stage.coordinate_sha256] != stage.role:
                raise InvalidModel("one semantic coordinate cannot mix API and gap boundaries")
            roles[stage.coordinate_sha256] = stage.role
        _evidence(self.evidence)

    def predict(self, profile: RuntimeProfile, *, cpu_intervals_ns, gpu_intervals_ns,
                provider, origin_ns, horizon_ns, replays, block_replays, start_ns,
                transition_ns=None, require_qualified=True):
        if not isinstance(profile, RuntimeProfile) or profile.scope != self.scope:
            raise UnknownParameter("runtime/boundary/host affinity profile mismatch")
        if type(require_qualified) is not bool:
            raise InvalidModel("explicit qualification requirement required")
        if require_qualified and not profile.qualified:
            raise UnknownParameter("runtime stage profile lacks independent qualification")
        cells = {(c.coordinate_sha256, c.role, c.state): c for c in profile.cells}
        prices = {}
        for stage in self.stages:
            if stage.coordinate_sha256 in prices:
                continue
            values = {}
            for state in STATES:
                cell = cells.get((stage.coordinate_sha256, stage.role, state))
                if cell is None or cell.mean_ns is None:
                    raise UnknownParameter("missing exact runtime role/state response: " + stage.coordinate_sha256 + str(state))
                values[state] = cell.mean_ns
            prices[stage.coordinate_sha256] = dict(kind="RUNTIME_RESPONSE", states=values)
        # No API->CPU conversion, GPU charge, whole-target multiplier or fitted
        # parameter is introduced by this wrapper. Every completion advances
        # the next stage on the predictor's own clock.
        law = PlannedCompletedResponse(self.scope.context_sha256, prices,
            tuple(s.coordinate_sha256 for s in self.stages), cpu_intervals_ns, gpu_intervals_ns,
            provider=provider, evidence=profile.evidence + self.evidence, tau_ns=transition_ns,
            origin_ns=origin_ns, horizon_ns=horizon_ns)
        result = law.predict_replays(replays=replays, block_replays=block_replays, start_ns=start_ns)
        return dict(result, stage_occurrences=len(self.stages),
                    host_physical_core_overlap=self.scope.placement.physical_core_overlap,
                    profile_independently_qualified=profile.qualified,
                    service_semantics="inclusive runtime API/gap response progress",
                    internal_GPU_or_DMA_time_added=False, active_CPU_time_inferred=False,
                    measured_target_boundaries_used=False, OS_scheduler_simulated=False)
