"""Explicit fixed slots and two-engine shared-link payload service.

The existing tensor transition rules own every copy, dependency, read lock and
completion. This sidecar only changes storage allocation and physical service.
Time is integer ns; Fraction byte progress is exact within each event interval.
Rounding completion up to a ns may leave unused link service, never extra bytes.
No copy launch overhead, pageable/pinned conversion or SM occupancy fraction is
inferred. SM copy exclusively occupies the GPU execution stream; CE may overlap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from typing import Iterable

from .engine import (ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT,
    Action, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION, CODE_RESOURCE_BUSY,
    MappingError, State, _accumulate_transfer_bytes, _event_for, advance_elapsed, transition)
from .environment import (CODE_ENVIRONMENT_DEADLOCK, CODE_INFORMATION_NOT_KNOWN,
    CODE_RECLAIM_ADMISSION, CODE_CONTROL_BUSY, EnvironmentEvent, ServiceInterval, SessionStep, SimulationSession)
from .spec import (COPY_ABSENT, COPY_READY, DOMAINS, DOMAIN_OF_DEVICE,
    LOC_DRAM, LOC_SSD, LOC_VRAM, OP_DONE, RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE,
    RESOURCE_PCIE_COPY, ROUTE_ENDS, Scenario, InvalidInput, Unsupported)

FORMAT_VERSION = "tensor-services-0.1"
RESOURCE_CE = "pcie_ce"
RESOURCE_SM = "pcie_sm"
RESOURCE_OF_ENGINE = {"ce": RESOURCE_CE, "sm": RESOURCE_SM}
CODE_SLOT_OCCUPIED = "SLOT_OCCUPIED"


@dataclass(frozen=True)
class FixedSlot:
    id: str
    domain: str
    capacity_bytes: int
    members: tuple[str, ...]


@dataclass(frozen=True)
class ServiceContract:
    bandwidth_bytes_per_second: int
    ce_bytes_per_second: int
    sm_bytes_per_second: int
    slots: tuple[FixedSlot, ...]
    arbitration: str = "max_min_fair"
    timing_scope: str = "payload_only_no_launch_latency"
    format_version: str = FORMAT_VERSION


@dataclass(frozen=True)
class PublicFixedSlot:
    id: str
    domain: str
    capacity_bytes: int
    members: tuple[str, ...]
    owner: str | None


@dataclass(frozen=True)
class PublicLinkFlow:
    engine: str
    tensor_id: str
    remaining_bytes_numerator: int
    remaining_bytes_denominator: int
    rate_bytes_per_ns_numerator: int
    rate_bytes_per_ns_denominator: int


@dataclass(frozen=True)
class PublicServices:
    bandwidth_bytes_per_second: int
    ce_bytes_per_second: int
    sm_bytes_per_second: int
    arbitration: str
    timing_scope: str
    slots: tuple[PublicFixedSlot, ...]
    flows: tuple[PublicLinkFlow, ...]
    sm_gpu_rule: str = "exclusive_stream_no_partial_sm_occupancy"


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise InvalidInput(f"{name} requires exactly {sorted(keys)}")
    return value


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise InvalidInput(f"{name} must be an explicit positive integer")
    return value


def parse_services(document) -> ServiceContract:
    obj = _object(document, {"format_version", "pcie", "slots"}, "services")
    if obj["format_version"] != FORMAT_VERSION:
        raise Unsupported(f"services requires {FORMAT_VERSION}")
    link = _object(obj["pcie"], {"bandwidth_bytes_per_second", "ce_bytes_per_second",
        "sm_bytes_per_second", "arbitration", "timing_scope"}, "services.pcie")
    for field in ("bandwidth_bytes_per_second", "ce_bytes_per_second", "sm_bytes_per_second"):
        _positive(link[field], f"services.pcie.{field}")
    if link["arbitration"] != "max_min_fair":
        raise Unsupported("services.pcie requires explicit max_min_fair arbitration")
    if link["timing_scope"] != "payload_only_no_launch_latency":
        raise Unsupported("services.pcie must acknowledge payload_only_no_launch_latency")
    if not isinstance(obj["slots"], list):
        raise InvalidInput("services.slots must be an explicit list")
    slots, ids, assignments = [], set(), set()
    for raw in obj["slots"]:
        slot = _object(raw, {"id", "domain", "capacity_bytes", "members"}, "services slot")
        if not isinstance(slot["id"], str) or not slot["id"].strip() or slot["id"] in ids:
            raise InvalidInput("slot ids must be unique nonempty strings")
        if slot["domain"] not in DOMAINS:
            raise Unsupported("fixed slots support explicit dram/vram only")
        _positive(slot["capacity_bytes"], "slot.capacity_bytes")
        if not isinstance(slot["members"], list) or not slot["members"] or any(
                not isinstance(tid, str) or not tid for tid in slot["members"]):
            raise InvalidInput("slot.members must list nonempty tensor ids")
        if len(set(slot["members"])) != len(slot["members"]):
            raise InvalidInput("a slot cannot repeat a member")
        for tid in slot["members"]:
            key = (slot["domain"], tid)
            if key in assignments:
                raise InvalidInput("a tensor/domain can belong to only one slot")
            assignments.add(key)
        ids.add(slot["id"])
        slots.append(FixedSlot(slot["id"], slot["domain"], slot["capacity_bytes"], tuple(slot["members"])))
    return ServiceContract(link["bandwidth_bytes_per_second"], link["ce_bytes_per_second"],
                           link["sm_bytes_per_second"], tuple(slots))


def services_document(contract: ServiceContract) -> dict:
    if not isinstance(contract, ServiceContract):
        raise InvalidInput("services requires a parsed ServiceContract")
    if not isinstance(contract.slots, tuple) or any(not isinstance(slot, FixedSlot)
            or not isinstance(slot.members, tuple) for slot in contract.slots):
        raise InvalidInput("ServiceContract slots/members must be frozen typed tuples")
    return {"format_version": contract.format_version, "pcie": {
        name: getattr(contract, name) for name in ("bandwidth_bytes_per_second", "ce_bytes_per_second",
        "sm_bytes_per_second", "arbitration", "timing_scope")},
        "slots": [{**asdict(slot), "members": list(slot.members)} for slot in contract.slots]}


@dataclass(frozen=True)
class AllocatedScenario(Scenario):
    """Fixed slot reserve is charged once; a bound member's payload is intact."""
    fixed_slots: tuple[FixedSlot, ...] = ()

    def size_alloc(self, tensor_id: str, domain: str = LOC_VRAM) -> int:
        if any(slot.domain == domain and tensor_id in slot.members for slot in self.fixed_slots):
            return 0
        return super().size_alloc(tensor_id, domain)


def allocated_scenario(scenario: Scenario, contract: ServiceContract) -> AllocatedScenario:
    if not isinstance(scenario, Scenario) or not scenario.is_m1:
        raise Unsupported("services requires a validated M1 Scenario")
    if isinstance(scenario, AllocatedScenario):
        raise InvalidInput("scenario already has service slot allocations")
    from .program import validate_device_state
    validate_device_state(scenario, scenario)
    contract = parse_services(services_document(contract))
    reserves = {domain: scenario.architecture.reserved_of(domain) for domain in DOMAINS}
    for slot in contract.slots:
        if slot.capacity_bytes % scenario.architecture.alignment_of(slot.domain):
            raise InvalidInput("fixed slot capacity must satisfy its domain allocation alignment")
        resident = []
        for tid in slot.members:
            tensor = scenario.workload.tensor_by_id.get(tid)
            if tensor is None:
                raise InvalidInput(f"slot references unknown tensor {tid!r}")
            if scenario.size_alloc(tid, slot.domain) > slot.capacity_bytes:
                raise InvalidInput(f"{tid!r} does not fit its fixed slot capacity")
            if slot.domain in tensor.initial_locations:
                resident.append(tid)
        if len(resident) > 1:
            raise InvalidInput("fixed slot cannot have multiple initial owners")
        reserves[slot.domain] += slot.capacity_bytes
    architecture = replace(scenario.architecture, dram_runtime_reserved_bytes=reserves[LOC_DRAM],
                           vram_runtime_reserved_bytes=reserves[LOC_VRAM])
    values = {field.name: getattr(scenario, field.name) for field in fields(Scenario)}
    values["architecture"] = architecture
    return AllocatedScenario(**values, fixed_slots=contract.slots)


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _fraction(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rates(contract: ServiceContract, resources: Iterable[str]) -> dict[str, Fraction]:
    """Exact max-min fairness subject to engine caps and one shared link."""
    caps = {RESOURCE_CE: Fraction(contract.ce_bytes_per_second, 10**9),
            RESOURCE_SM: Fraction(contract.sm_bytes_per_second, 10**9)}
    pending = set(resources)
    budget = Fraction(contract.bandwidth_bytes_per_second, 10**9)
    result = {}
    while pending:
        share = budget / len(pending)
        saturated = {resource for resource in pending if caps[resource] <= share}
        if not saturated:
            result.update({resource: share for resource in sorted(pending)})
            break
        for resource in sorted(saturated):
            result[resource] = caps[resource]
            budget -= caps[resource]
        pending -= saturated
    return result


class ServiceSession(SimulationSession):
    """One slot/copy/dependency ledger; service sidecar stores only byte progress."""
    def __init__(self, scenario: Scenario, services: ServiceContract, *,
                 environment_events: Iterable[EnvironmentEvent] = (), visibility=None,
                 reclaim_requests=(), control_costs=None):
        self.services = parse_services(services_document(services))
        self.base_scenario = scenario
        self._flows: dict[tuple[str, str], Fraction] = {}
        self._engine_choices: list[dict] = []
        self._link_intervals: list[dict] = []
        super().__init__(allocated_scenario(scenario, self.services),
                         environment_events=environment_events, visibility=visibility,
                         reclaim_requests=reclaim_requests, control_costs=control_costs)

    def _slot_conflict(self, action: Action) -> str | None:
        destinations = []
        if action.kind == ACTION_COMPUTE and action.target_id in self.scenario.workload.operation_by_id \
                and action.device in DOMAIN_OF_DEVICE:
            destinations = [(tid, DOMAIN_OF_DEVICE[action.device])
                            for tid in self.scenario.workload.operation_by_id[action.target_id].outputs]
        elif action.route is not None and action.target_id is not None:
            destinations = [(action.target_id, ROUTE_ENDS[action.route][1])]
        for slot in self.services.slots:
            targets = [tid for tid, domain in destinations if domain == slot.domain and tid in slot.members]
            if len(targets) > 1:
                return f"action would produce several simultaneous owners of fixed slot {slot.id!r}"
            for tid in targets:
                others = [member for member in slot.members if member != tid
                          and self.state.copy_at(member, slot.domain) != COPY_ABSENT]
                if others:
                    return (f"fixed slot {slot.id!r} is owned by {others}; explicit EVICT after last reader "
                            f"is required before assigning {tid!r}")
        return None

    def _reject(self, action, source_index, code, message):
        self._validate_source(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        index = len(self._actions)
        self._actions.append(action); self._sources.append(source_index)
        return self._fail(code, message, action_index=index)

    def submit(self, action: Action, source_index: int | None = None) -> SessionStep:
        if not isinstance(action, Action):
            raise InvalidInput("ServiceSession.submit requires Action")
        self._validate_source(source_index)
        if action.kind == ACTION_ADVANCE:
            return self.advance(source_index)
        # Preserve the trusted session's error/goal/visibility gates before
        # consulting private slot membership or an undisclosed operation's outputs.
        if self._error_code or self.finished or (self._visibility
                and not self._visibility.allow_action(action, self.state)):
            return super().submit(action, source_index)
        if action.kind == ACTION_COPY:
            return self._reject(action, source_index, CODE_ILLEGAL_ACTION,
                                "PCIe COPY requires submit_transfer with explicit ce/sm engine")
        conflict = self._slot_conflict(action)
        if conflict:
            return self._reject(action, source_index, CODE_SLOT_OCCUPIED, conflict)
        if action.kind == ACTION_COMPUTE:
            if action.device == "gpu" and self.state.task_on(RESOURCE_SM):
                return self._reject(action, source_index, CODE_RESOURCE_BUSY, "SM copy occupies GPU execution stream")
            if not self.scenario.mapspace.allow_copy_compute_overlap and self._flows:
                return self._reject(action, source_index, CODE_RESOURCE_BUSY, "copy/compute overlap is disabled")
        return super().submit(action, source_index)

    def submit_transfer(self, action: Action, *, engine: str, source_index: int | None = None) -> SessionStep:
        if not isinstance(action, Action):
            raise InvalidInput("submit_transfer requires Action")
        if not isinstance(engine, str) or engine not in RESOURCE_OF_ENGINE:
            raise Unsupported("transfer engine must be explicit ce or sm")
        if action.kind != ACTION_COPY:
            raise InvalidInput("submit_transfer engines apply only to H2D/D2H COPY")
        self._validate_source(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        index = len(self._actions)
        self._actions.append(action); self._sources.append(source_index)
        if self.finished:
            return self._fail("actions_remain_after_goal", "actions follow the completed goal", action_index=index)
        if self._visibility and not self._visibility.allow_action(action, self.state):
            return self._fail(CODE_INFORMATION_NOT_KNOWN, "action references information not yet revealed", action_index=index)
        resource = RESOURCE_OF_ENGINE[engine]
        if self.state.task_on(resource):
            return self._fail(CODE_RESOURCE_BUSY, f"{engine} copy engine is busy", action_index=index)
        if engine == "sm" and self.state.task_on(RESOURCE_GPU_COMPUTE):
            return self._fail(CODE_RESOURCE_BUSY, "GPU execution stream is busy; SM copy must wait", action_index=index)
        conflict = self._slot_conflict(action)
        if conflict:
            return self._fail(CODE_SLOT_OCCUPIED, conflict, action_index=index)
        try:
            candidate = transition(self.scenario, self.state, action)
        except MappingError as error:
            return self._fail(error.code, str(error), action_index=index)
        admission = self._admission_message(action, candidate)
        if admission:
            return self._fail(CODE_RECLAIM_ADMISSION if self._pending_reclaim else CODE_CONTROL_BUSY,
                              admission, action_index=index)
        conflict = self._capacity_message(self._memory(candidate))
        if conflict:
            return self._fail(CODE_CAPACITY_EXCEEDED, conflict, action_index=index)
        payload = self.scenario.workload.tensor_by_id[action.target_id].storage_bytes
        isolated_rate = _rates(self.services, [resource])[resource]
        duration = _ceil(Fraction(payload) / isolated_rate)
        candidate = replace(candidate, running=tuple(sorted(
            replace(task, resource=resource, remaining_ns=duration)
            if task.resource == RESOURCE_PCIE_COPY and task.target_id == action.target_id else task
            for task in candidate.running)))
        self._state = candidate
        self._flows[(resource, action.target_id)] = Fraction(payload)
        event = replace(_event_for(self.scenario, self.state, len(self._events), action), resource=resource)
        self._nominal_ends.append(self.state.t + duration)
        self._launches[(resource, action.target_id)] = len(self._events)
        self._engine_choices.append({"action_index": index, "event_index": len(self._events),
                                    "source_index": source_index, "engine": engine,
                                    "tensor_id": action.target_id, "route": action.route,
                                    "isolated_rate_bytes_per_ns": _fraction(isolated_rate)})
        self._events.append(replace(event, end_ns=None))
        self._amounts = _accumulate_transfer_bytes(event, *self._amounts)
        self._project_rates()
        self._update_peaks(); self._snapshots.append(self._snapshot(len(self._events)))
        return SessionStep(True, self.state.t, action=action)

    def _project_rates(self):
        rates = _rates(self.services, [resource for resource, _ in self._flows])
        self._state = replace(self.state, running=tuple(sorted(
            replace(task, remaining_ns=_ceil(self._flows[(task.resource, task.target_id)] / rates[task.resource]))
            if (task.resource, task.target_id) in self._flows else task for task in self.state.running)))
        return rates

    def advance(self, source_index: int | None = None) -> SessionStep:
        self._validate_source(source_index)
        if self._error_code:
            return self._fail(self._error_code, self._error_message or "session failed")
        was_finished = self.finished
        self._seal_deadline()
        if not was_finished and self.finished:
            return SessionStep(True, self.state.t)
        action = Action(ACTION_ADVANCE); index = len(self._actions)
        self._actions.append(action); self._sources.append(source_index)
        if self.finished:
            return self._fail("actions_remain_after_goal", "advance follows the completed goal", action_index=index)
        rates = self._project_rates()
        paused = () if self._cpu_available else (RESOURCE_CPU_COMPUTE,)
        active = [task for task in self.state.running if task.resource not in paused]
        times = [self.state.t + task.remaining_ns for task in active]
        times.extend(self._extra_boundary_times())
        if self._queue_index < len(self._queue):
            times.append(self._queue[self._queue_index].t_ns)
        if not times:
            code = CODE_ENVIRONMENT_DEADLOCK if self.state.running else CODE_ILLEGAL_ACTION
            return self._fail(code, "no running service or future environment event can make progress", action_index=index)
        destination = min(times); before = self.state; elapsed = destination - before.t
        amounts = {}
        for key, remaining in list(self._flows.items()):
            served = min(remaining, rates[key[0]] * elapsed)
            amounts[key] = served; self._flows[key] -= served
        for task in active:
            self._service.append(ServiceInterval(self._launches[(task.resource, task.target_id)],
                                                task.resource, task.target_id, before.t, destination))
        if amounts:
            budget = Fraction(self.services.bandwidth_bytes_per_second, 10**9) * elapsed
            assert sum(amounts.values()) <= budget
            self._link_intervals.append({"start_ns": before.t, "end_ns": destination,
                "link_budget_bytes": _fraction(budget), "unused_budget_bytes": _fraction(budget - sum(amounts.values())),
                "flows": [{"resource": key[0], "tensor_id": key[1],
                           "rate_bytes_per_ns": _fraction(rates[key[0]]), "served_bytes": _fraction(value)}
                          for key, value in sorted(amounts.items())]})
        self._state = advance_elapsed(self.scenario, before, elapsed, paused)
        after_keys = {(task.resource, task.target_id) for task in self.state.running}
        for task in before.running:
            key = (task.resource, task.target_id)
            if key not in after_keys:
                event_index = self._launches.pop(key)
                self._events[event_index] = replace(self._events[event_index], end_ns=destination)
                if key in self._flows:
                    assert self._flows[key] == 0
                    del self._flows[key]
        self._project_rates()
        completed_ops = tuple(oid for oid, status in self.state.op_status
                              if status == OP_DONE and before.op(oid) != OP_DONE)
        completed_copies = tuple((tid, domain) for domain in (LOC_DRAM, LOC_VRAM, LOC_SSD)
                                for tid, status in self.state.statuses(domain)
                                if status == COPY_READY and before.copy_at(tid, domain) != COPY_READY)
        event = _event_for(self.scenario, self.state, len(self._events), action)
        self._events.append(event); self._nominal_ends.append(event.end_ns)
        self._after_time_boundary(action_index=index)
        self._update_peaks(); self._snapshots.append(self._snapshot(len(self._events)))
        return SessionStep(not self._error_code, self.state.t, action, completed_ops, completed_copies,
                           self._error_code, self._error_message)

    def service_document(self) -> dict:
        contract = services_document(self.services)
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return deepcopy({"format_version": "tensor-service-execution-0.1", "contract": contract,
            "contract_sha256": digest, "timing_resolution": "completion_ceil_to_integer_ns_without_extra_service",
            "transfer_engines": list(self._engine_choices), "link_service_intervals": list(self._link_intervals),
            "slots": [{"id": slot.id, "domain": slot.domain, "reserved_bytes": slot.capacity_bytes,
                "owner": next((tid for tid in slot.members if self.state.copy_at(tid, slot.domain) != COPY_ABSENT), None)}
                      for slot in self.services.slots],
            "running_byte_progress": [{"resource": resource, "tensor_id": tid, "remaining_bytes": _fraction(remaining)}
                                      for (resource, tid), remaining in sorted(self._flows.items())],
            "physical_calibration_claimed": False, "startup_latency_included": False,
            "pinned_or_pageable_conversion_modeled": False,
            "gpu_execution_semantics": "SM_copy_exclusive_with_GPU_compute; CE_copy_may_overlap"})

    def public_services(self, visible_tensor_ids: Iterable[str]) -> PublicServices:
        """Immutable current service state; undisclosed tensor IDs stay private.

        Slot capacities and physical link parameters are public hardware state.
        Membership/ownership and byte progress are projected through the same
        model-owned tensor visibility used for causal compute and COPY actions.
        Future environment events and future dynamic slot members are absent.
        """
        visible = frozenset(visible_tensor_ids)
        slots = tuple(PublicFixedSlot(slot.id, slot.domain, slot.capacity_bytes,
            tuple(tid for tid in slot.members if tid in visible),
            next((tid for tid in slot.members if tid in visible and
                  self.state.copy_at(tid, slot.domain) != COPY_ABSENT), None))
            for slot in self.services.slots)
        rates = _rates(self.services, [resource for resource, _ in self._flows])
        engines = {resource: engine for engine, resource in RESOURCE_OF_ENGINE.items()}
        flows = tuple(PublicLinkFlow(engines[resource], tid,
            remaining.numerator, remaining.denominator,
            rates[resource].numerator, rates[resource].denominator)
            for (resource, tid), remaining in sorted(self._flows.items()) if tid in visible)
        return PublicServices(self.services.bandwidth_bytes_per_second,
            self.services.ce_bytes_per_second, self.services.sm_bytes_per_second,
            self.services.arbitration, self.services.timing_scope, slots, flows)

    def isolated_transfer_duration(self, tensor_id: str, *, engine: str) -> int:
        """Declared payload-only isolated duration, never a contention prediction."""
        if engine not in RESOURCE_OF_ENGINE:
            raise Unsupported("transfer engine must be explicit ce or sm")
        tensor = self.scenario.workload.tensor_by_id.get(tensor_id)
        if tensor is None:
            raise InvalidInput("isolated duration references an unknown tensor")
        return _ceil(Fraction(tensor.storage_bytes) /
                     _rates(self.services, [RESOURCE_OF_ENGINE[engine]])[RESOURCE_OF_ENGINE[engine]])
