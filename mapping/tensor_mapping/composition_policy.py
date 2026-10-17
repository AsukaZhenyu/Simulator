"""Composable, observation-only scheduling mechanisms; no search algorithm.

The public catalogue contains model structure, never future routing decisions.
Dynamic expert selectors choose a complete revealed region. Their estimates
sum declared stage prices and boundary transfers, and do not claim fusion or
predict the overlap that the runtime will actually arbitrate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from llm_infer_model.m2.causal import CausalObservation
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT, COPY_READY, OP_DONE, OP_NOT_STARTED, InvalidInput, Unsupported,
)

from .causal import Decision, ControlDecision
from .policy_executor import DeclarativePolicy
from .policy_ir import EXPERT_COMPUTE_KINDS, parse_policy
from .unified_policy import UnifiedPolicy
from .reconfiguration import ReconfigurationSpec, reconfiguration_document


@dataclass(frozen=True)
class TransportRule:
    role: str
    stage: str | None
    src: str
    dst: str
    engine: str | None


@dataclass(frozen=True)
class CompositionSpec:
    policy: object
    cache_unit: str
    cache_budget_bytes: int | None
    prefetch_scope: str
    lookahead_layers: int
    prefetch_ranking: str
    include_dense_weights: bool
    transport_rules: tuple[TransportRule, ...]
    slot_reuse: str
    boundary_release: str
    execution_bound_kinds: tuple[str, ...] = ()
    execution_bound_requests: tuple[tuple[str, str], ...] = ()
    reconfiguration: ReconfigurationSpec | None = None
    scoped_bound_kinds: tuple[str, ...] = ()


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise InvalidInput(f"{label} requires exactly {sorted(fields)}")
    return value


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise InvalidInput(f"{label} must be a nonnegative integer")
    return value


def _choice(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise Unsupported(f"{label} must be one of {sorted(choices)}")
    return value


def parse_composition_policy(document) -> CompositionSpec:
    """Parse the seven inline mechanism objects of unified-strategy-0.2."""
    _object(document, {"placements", "request_placements", "experts", "release",
                       "cache", "prefetch", "transport"}, "composition")
    experts = _object(document["experts"], {"device_rule", "selection_unit"}, "experts")
    _choice(experts["selection_unit"], {"expert_region"}, "experts.selection_unit")
    cache = _object(document["cache"], {"rule", "unit", "budget_bytes", "alpha", "top_p"}, "cache")
    _choice(cache["unit"], {"tensor", "expert_group"}, "cache.unit")
    budget = cache["budget_bytes"]
    if budget is not None:
        _integer(budget, "cache.budget_bytes")
    if cache["rule"] != "mrs" and (cache["alpha"] is not None or cache["top_p"] is not None):
        raise InvalidInput("non-MRS cache requires explicit null alpha/top_p")
    legacy_cache = {"rule": cache["rule"]}
    if cache["rule"] == "mrs":
        legacy_cache.update(alpha=cache["alpha"], top_p=cache["top_p"])
    prefetch = _object(document["prefetch"], {"enabled", "scope", "lookahead_layers",
                      "max_count", "ranking", "include_dense_weights"}, "prefetch")
    if type(prefetch["enabled"]) is not bool or type(prefetch["include_dense_weights"]) is not bool:
        raise InvalidInput("prefetch.enabled and include_dense_weights must be boolean")
    scope = _choice(prefetch["scope"], {"none", "next_layer"}, "prefetch.scope")
    lookahead = _integer(prefetch["lookahead_layers"], "prefetch.lookahead_layers")
    count = _integer(prefetch["max_count"], "prefetch.max_count")
    ranking = _choice(prefetch["ranking"], {"history_frequency", "mrs"}, "prefetch.ranking")
    if prefetch["enabled"]:
        if scope != "next_layer" or not lookahead or not count or budget is None or not budget:
            raise InvalidInput("enabled cross-layer prefetch requires next_layer, positive lookahead/count/budget")
        if ranking == "mrs" and cache["rule"] != "mrs":
            raise InvalidInput("MRS prefetch ranking requires the revealed-score MRS cache")
    elif scope != "none" or lookahead or count or prefetch["include_dense_weights"]:
        raise InvalidInput("disabled prefetch requires none scope, zero lookahead/count and no dense weights")
    transport = _object(document["transport"], {"rules", "slot_reuse", "boundary_release"}, "transport")
    _choice(transport["slot_reuse"], {"disabled", "release_unlocked_backed_copy"}, "transport.slot_reuse")
    _choice(transport["boundary_release"], {"disabled", "declared_dead_intermediates"}, "transport.boundary_release")
    if not isinstance(transport["rules"], list) or not transport["rules"]:
        raise InvalidInput("transport.rules must be a nonempty explicit list")
    rules, seen = [], set()
    for row in transport["rules"]:
        _object(row, {"role", "stage", "src", "dst", "engine"}, "transport rule")
        _choice(row["role"], {"weight", "input", "intermediate", "output"}, "transport role")
        if row["stage"] is not None and (not isinstance(row["stage"], str) or not row["stage"]):
            raise InvalidInput("transport.stage must be explicit null or a nonempty string")
        if (row["src"], row["dst"]) not in (("dram", "vram"), ("vram", "dram")):
            raise Unsupported("transport engine rules bind PCIe dram/vram directions only")
        if row["engine"] is not None and row["engine"] not in ("ce", "sm"):
            raise Unsupported("transport engine is explicit null, ce or sm")
        key = (row["role"], row["stage"], row["src"], row["dst"])
        if key in seen:
            raise InvalidInput("duplicate transport rule of equal specificity")
        seen.add(key)
        rules.append(TransportRule(**row))
    placements = deepcopy(document["placements"])
    overrides = deepcopy(document["request_placements"])
    bound_kinds, bound_requests, scoped_kinds = [], [], []
    if not isinstance(placements, dict) or not isinstance(overrides, dict):
        raise InvalidInput("placements and request_placements must be explicit objects")
    for kind, device in placements.items():
        if device == "scoped_binding":
            placements[kind] = "cpu"  # Unreachable parser token; frontend resolves it.
            scoped_kinds.append(kind)
            continue
        if device == "execution_binding":
            from llm_infer_model.m2.execution_pipeline import TRANSFORM_KINDS
            if kind != "attention_core" and kind not in TRANSFORM_KINDS:
                raise Unsupported("execution_binding requires an explicitly bound execution stage")
            placements[kind] = "cpu"  # A parser placeholder, never a fallback.
            bound_kinds.append(kind)
    for request, bindings in overrides.items():
        if not isinstance(bindings, dict):
            raise InvalidInput("request placements must contain explicit binding objects")
        for kind, device in bindings.items():
            if device == "execution_binding":
                from llm_infer_model.m2.execution_pipeline import TRANSFORM_KINDS
                if kind != "attention_core" and kind not in TRANSFORM_KINDS:
                    raise Unsupported("execution_binding requires an explicitly bound execution stage")
                bindings[kind] = "cpu"
                bound_requests.append((request, kind))
    legacy = parse_policy({"format_version": "causal-policy-0.2",
        "placements": placements, "request_placements": overrides,
        "experts": {"device_rule": experts["device_rule"]}, "cache": legacy_cache,
        "release": document["release"], "prefetch": {"enabled": prefetch["enabled"],
             "max_count": count, "cache_budget_bytes": budget if prefetch["enabled"] else 0}})
    return CompositionSpec(legacy, cache["unit"], budget, scope, lookahead, ranking,
        prefetch["include_dense_weights"], tuple(rules), transport["slot_reuse"], transport["boundary_release"],
        tuple(sorted(bound_kinds)), tuple(sorted(bound_requests)), scoped_bound_kinds=tuple(sorted(scoped_kinds)))


def composition_policy_document(spec: CompositionSpec) -> dict:
    if not isinstance(spec, CompositionSpec):
        raise InvalidInput("composition_policy_document requires CompositionSpec")
    policy = spec.policy
    result = {"placements": dict(policy.placements),
        "request_placements": {request: dict(bindings) for request, bindings in policy.request_placements},
        "experts": {"device_rule": policy.expert_device_rule, "selection_unit": "expert_region"},
        "release": {"rule": policy.release_rule},
        "cache": {"rule": policy.cache_rule, "unit": spec.cache_unit, "budget_bytes": spec.cache_budget_bytes,
                  "alpha": policy.mrs_alpha, "top_p": policy.mrs_top_p},
        "prefetch": {"enabled": policy.prefetch_enabled, "scope": spec.prefetch_scope,
                    "lookahead_layers": spec.lookahead_layers, "max_count": policy.prefetch_max_count,
                    "ranking": spec.prefetch_ranking, "include_dense_weights": spec.include_dense_weights},
        "transport": {"rules": [rule.__dict__.copy() for rule in spec.transport_rules],
                      "slot_reuse": spec.slot_reuse, "boundary_release": spec.boundary_release}}
    for kind in spec.execution_bound_kinds:
        result["placements"][kind] = "execution_binding"
    for kind in spec.scoped_bound_kinds:
        result["placements"][kind] = "scoped_binding"
    for request, kind in spec.execution_bound_requests:
        result["request_placements"][request][kind] = "execution_binding"
    return result


class CompositionPolicy(UnifiedPolicy):
    """Greedy, explainable composition of declared rules over public snapshots.

    The driver submits each returned decision before asking again. The fixed
    slot guard can replace a proposed action, in which case *all* tentative
    placement, eviction and prefetch state is rolled back.
    """
    def __init__(self, spec: CompositionSpec, *, history=()):
        if not isinstance(spec, CompositionSpec):
            raise InvalidInput("CompositionPolicy requires parsed CompositionSpec")
        spec = replace(parse_composition_policy(composition_policy_document(spec)), reconfiguration=spec.reconfiguration)
        # The role-only binding exists only to reuse the lifetime/slot guard.
        # All estimation/loading below uses the explicit direction/stage rules.
        super().__init__(spec.policy, history=history, transport={
            "transfer_engines": {role: None for role in ("weight", "input", "intermediate", "output")},
            "slot_reuse": spec.slot_reuse, "boundary_release": spec.boundary_release,
            "fragment_cache_unit": "tensor"})
        self.spec = spec
        self._catalogue = {}
        self._groups = {}
        self._tensor_groups = {}
        self._regions = {}
        self._region_devices = {}
        self._region_estimates = {}
        self._pending_eviction_group = None
        self._prefetch_units = set()
        self._composition_records = []
        self._tensor_stages = {}
        self._operation_devices = {}
        self._reclaim_decisions = set()
        self._observed_cpu = None

    def _placement(self, operation):
        override = self._request_placements.get(operation.request_id, {})
        bound = ((operation.request_id, operation.kind) in self.spec.execution_bound_requests or
                 (operation.kind not in override and operation.kind in self.spec.execution_bound_kinds))
        if bound:
            if operation.kind == "attention_core" and (operation.execution_variant != "pure_attention" or len(operation.prices) != 1):
                raise Unsupported("execution_binding needs pure_attention attention_core with one declared exact device")
            if len(operation.prices) != 1:
                raise Unsupported("execution_binding needs one declared exact device for each stage")
            return operation.prices[0].device
        if operation.kind in self.spec.scoped_bound_kinds and operation.kind not in override:
            raise Unsupported("scoped_binding requires the Python frontend's explicit selector resolver")
        return super()._placement(operation)

    def _public_catalogue(self, observation):
        rows = getattr(observation, "weight_catalogue", ())
        catalogue = {}
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 4:
                raise InvalidInput("public weight catalogue rows must be frozen (tensor,layer,stage,expert) tuples")
            tid, layer, stage, expert = row
            if not isinstance(tid, str) or not isinstance(layer, str) or not isinstance(stage, str):
                raise InvalidInput("public weight catalogue requires string identities and stage")
            if tid in catalogue:
                raise InvalidInput("duplicate public weight catalogue tensor")
            catalogue[tid] = (layer, stage, expert)
        for expert in observation.experts:
            children = expert.weight_tensor_ids or ((expert.weight_tensor_id,) if expert.weight_tensor_id else ())
            for tid in children:
                catalogue.setdefault(tid, (expert.layer_id, "", expert.expert_id))
        self._catalogue = catalogue
        self._tensor_stages = {tid: row[1] for tid, row in catalogue.items()}
        for operation in observation.operations:
            for tid in operation.outputs:
                self._tensor_stages[tid] = operation.execution_stage or operation.kind
        groups = {}
        for tensor in observation.tensors:
            if tensor.role != "weight":
                continue
            layer, _, expert = catalogue.get(tensor.id, ("", "", None))
            key = ("expert", layer, expert) if expert is not None else ("tensor", tensor.id)
            groups.setdefault(key, []).append(tensor.id)
        self._groups = {key: tuple(sorted(tids)) for key, tids in groups.items()}
        self._tensor_groups = {tid: key for key, tids in self._groups.items() for tid in tids}
        self._regions = {}
        for operation in observation.operations:
            if operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int):
                if not operation.region_id and self.policy.expert_device_rule != "fixed":
                    raise Unsupported("dynamic expert selection requires a revealed expert region")
                key = self._region_key(operation)
                self._regions.setdefault(key, []).append(operation)
        self._regions = {key: tuple(operations) for key, operations in self._regions.items()}

    @staticmethod
    def _region_key(operation):
        return (operation.invocation_id, operation.region_id or operation.id)

    def _engine(self, tensor, src, dst):
        stage = self._tensor_stages.get(tensor.id, "")
        matches = [rule for rule in self.spec.transport_rules if rule.role == tensor.role
                   and rule.src == src and rule.dst == dst and rule.stage in (None, stage)]
        if not matches:
            raise Unsupported(f"no explicit engine rule for {tensor.id}/{tensor.role}/{stage}/{src}->{dst}")
        return max(matches, key=lambda rule: rule.stage is not None).engine

    def _estimate_observation(self, observation):
        self._public_catalogue(observation)
        self._services_enabled = observation.services is not None
        tensors = []
        for tensor in observation.tensors:
            prices = []
            for price in tensor.transfers:
                if {price.src, price.dst} == {"dram", "vram"}:
                    engine = self._engine(tensor, price.src, price.dst)
                    if self._services_enabled:
                        if engine is None:
                            raise InvalidInput("services COPY requires an explicitly declared ce/sm engine")
                        duration = dict(price.engine_durations_ns).get(engine)
                        if duration is None:
                            raise Unsupported(f"no isolated {engine} service estimate for {tensor.id}/{price.route}")
                        prices.append(replace(price, duration_ns=duration))
                    else:
                        if engine is not None:
                            raise Unsupported("ce/sm engine rules require services-enabled modeling")
                        if price.duration_ns is None:
                            raise Unsupported("legacy transfer has no exact public duration")
                        prices.append(price)
                else:
                    if price.duration_ns is None:
                        raise Unsupported("SSD transfer needs an exact public duration")
                    prices.append(price)
            tensors.append(replace(tensor, transfers=tuple(prices)))
        return replace(observation, tensors=tuple(tensors))

    def _load_step(self, tensor, destination):
        result = DeclarativePolicy._load_step(tensor, destination)
        if result is None or result[0].kind != ACTION_COPY:
            return result
        decision, price, resource = result
        if self._services_enabled:
            engine = self._engine(tensor, price.src, price.dst)
            return replace(decision, transfer_engine=engine), price, f"pcie_{engine}"
        return result

    def observe_final(self, observation):
        if not isinstance(observation, CausalObservation):
            raise InvalidInput("CompositionPolicy requires public CausalObservation")
        self._public_catalogue(observation)
        DeclarativePolicy.observe_final(self, self._fragment_catalogue(observation))

    def _region_parts(self, operation, observation, tensors):
        operations = self._regions[self._region_key(operation)]
        kinds = {op.kind for op in operations}
        if kinds not in ({"expert_ffn"}, {"expert_gate", "expert_up", "expert_swiglu", "expert_down"}):
            raise Unsupported("complete-region selector requires one opaque FFN or all four explicit SwiGLU stages")
        if len(operations) != len(kinds):
            raise InvalidInput("complete expert region contains duplicate stage kinds")
        if any(op.token_count < 1 or len(op.work_items) != op.token_count or
               op.work_items != operation.work_items for op in operations):
            raise Unsupported("complete expert region needs consistent exact revealed token rows")
        internal = {tid for op in operations for tid in op.outputs}
        inputs = tuple(sorted({tid for op in operations for tid in op.inputs if tid not in internal}))
        weights = tuple(tid for tid in inputs if tensors[tid].role == "weight")
        activation = tuple(tid for tid in inputs if tensors[tid].role != "weight")
        outputs = tuple(sorted(tid for tid in internal if not any(tid in op.inputs for op in operations)))
        producers = [op for op in observation.operations if set(op.outputs) & set(activation)]
        consumers = [op for op in observation.operations if set(op.inputs) & set(outputs)]
        if not producers or any(op.kind != "gather_rows" or self._placement(op) != "gpu" for op in producers):
            raise Unsupported("complete expert path requires an explicit GPU gather boundary")
        if not consumers or any(op.kind != "scatter_rows" or self._placement(op) != "gpu" for op in consumers):
            raise Unsupported("complete expert path requires an explicit GPU scatter boundary")
        return operations, weights, activation, outputs

    def _region_paths(self, operation, observation, tensors):
        operations, weights, activation, outputs = self._region_parts(operation, observation, tensors)
        paths = []
        for device in ("cpu", "gpu"):
            location = "dram" if device == "cpu" else "vram"
            stages = [next((price for price in op.prices if price.device == device), None) for op in operations]
            if any(price is None for price in stages):
                raise Unsupported(f"complete expert region has no exact {device} stage prices")
            movements = [self._transfer_estimate(tensors[tid], location) for tid in weights + activation]
            if any(value is None for value in movements):
                # An in-flight first boundary copy is not a new decision point.
                # Skip this path rather than pretending its remaining time is 0.
                continue
            returns = []
            for tid in outputs:
                if device == "gpu":
                    returns.append(0)
                else:
                    price = next((price for price in tensors[tid].transfers
                                  if price.src == "dram" and price.dst == "vram"), None)
                    if price is None:
                        raise Unsupported("CPU expert output has no exact GPU return boundary price")
                    returns.append(price.duration_ns)
            footprint = self._budget_for_additions(tensors, weights) if device == "gpu" else 0
            feasible = self.spec.cache_budget_bytes is None or footprint <= self.spec.cache_budget_bytes
            if any(price.allocated_workspace_bytes is None for price in stages):
                raise Unsupported(f"complete expert region has unknown aligned {device} workspace")
            all_ports = {tid for op in operations for tid in op.inputs + op.outputs}
            physical_bytes = sum(self._allocation(tensors[tid], location) for tid in all_ports)
            physical_bytes += max(price.allocated_workspace_bytes for price in stages)
            memory = next((entry for entry in observation.memory if entry.domain == location), None)
            if memory is None:
                raise Unsupported(f"complete expert path has no public {location} capacity ledger")
            physical_feasible = memory.capacity_bytes is None or physical_bytes <= (
                memory.capacity_bytes - memory.runtime_reserved_bytes - memory.external_bytes)
            paths.append({"device": device, "weight_transfer_ns": sum(movements[:len(weights)]),
                "selected_input_transfer_ns": sum(movements[len(weights):]),
                "compute_ns": sum(price.duration_ns for price in stages), "output_return_ns": sum(returns),
                "total_ns": sum(movements) + sum(price.duration_ns for price in stages) + sum(returns),
                "stage_operations": [op.id for op in operations], "unique_weights": list(weights),
                "price_scope": "sum_of_declared_stages_and_isolated_boundary_transfers",
                "weight_budget_footprint_bytes": footprint, "budget_feasible": feasible,
                "conservative_region_allocation_bytes": physical_bytes,
                "physical_feasible": physical_feasible})
        return paths

    def _devices(self, operation, observation, tensors):
        from llm_infer_model.m2.execution_pipeline import TRANSFORM_KINDS
        if operation.kind in TRANSFORM_KINDS and len(operation.prices)==1:
            device=self._placement(operation)
            return () if device=="cpu" and not observation.cpu_available else (device,)
        if self.spec.reconfiguration is not None and self.spec.reconfiguration.cpu_response == "current_cost" and operation.kind != "state_update":
            return self._current_cost_devices(operation, observation, tensors)
        if operation.kind not in EXPERT_COMPUTE_KINDS or not isinstance(operation.expert_id, int):
            return super()._devices(operation, observation, tensors)
        rule = self.policy.expert_device_rule
        if rule == "fixed":
            return super()._devices(operation, observation, tensors)
        key = self._region_key(operation)
        if key in self._region_devices:
            device = self._region_devices[key]
            return () if device == "cpu" and not observation.cpu_available else (device,)
        paths = self._region_paths(operation, observation, tensors)
        available = [path for path in paths if path["budget_feasible"] and path["physical_feasible"] and
                     (path["device"] != "cpu" or observation.cpu_available)]
        if not available:
            return ()
        _, weights, _, _ = self._region_parts(operation, observation, tensors)
        cached = bool(weights) and all(tensors[tid].status_at("vram") == COPY_READY for tid in weights)
        preferred = self._placement(operation)
        if rule in ("resident_first", "fiddler_paths") and cached:
            chosen = next((path for path in available if path["device"] == "gpu"), None)
        else:
            chosen = None
        if chosen is None:
            chosen = min(available, key=lambda path: (path["total_ns"], path["device"] != preferred, path["device"]))
        self._region_devices[key] = chosen["device"]
        self._region_estimates[key] = paths
        for op in self._regions[key]:
            self._fiddler_devices[op.id] = chosen["device"]
            self._fiddler_paths[op.id] = paths
        return (chosen["device"],)

    def _current_cost_devices(self, operation, observation, tensors):
        """Choose an unsubmitted task/region using current exact prices only.

        This is isolated path ranking, not a cross-layer benefit estimate. A
        binding is committed by _demand only when a preparation/compute action
        is actually returned; subsequent blackout never rebinds that work.
        """
        is_region = operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int)
        key = self._region_key(operation)
        bound = self._region_devices.get(key) if is_region else self._operation_devices.get(operation.id)
        if bound is not None:
            return () if bound == "cpu" and not observation.cpu_available else (bound,)
        preferred = self._placement(operation)
        if is_region:
            paths = self._region_paths(operation, observation, tensors)
            available = [path for path in paths if path["budget_feasible"] and path["physical_feasible"]
                         and (path["device"] != "cpu" or observation.cpu_available)]
            self._region_estimates[key] = paths
            return (min(available, key=lambda path: (path["total_ns"], path["device"] != preferred, path["device"]))["device"],) if available else ()
        candidates = []
        for price in operation.prices:
            if price.device == "cpu" and not observation.cpu_available:
                continue
            if price.allocated_workspace_bytes is None:
                raise Unsupported("current_cost requires exact allocated workspace; unknown is not zero")
            location = "dram" if price.device == "cpu" else "vram"
            movement = [self._transfer_estimate(tensors[tid], location) for tid in operation.inputs]
            if any(cost is None for cost in movement):
                continue
            memory = next(entry for entry in observation.memory if entry.domain == location)
            minimum = sum(self._allocation(tensors[tid], location)
                          for tid in set(operation.inputs + operation.outputs)) + price.allocated_workspace_bytes
            if memory.capacity_bytes is not None and minimum > memory.capacity_bytes - memory.runtime_reserved_bytes - memory.external_bytes:
                continue
            candidates.append((price.duration_ns + sum(movement), price.device != preferred, price.device))
        return (min(candidates)[2],) if candidates else ()

    def _reconfiguration_step(self, observation):
        spec = self.spec.reconfiguration
        if spec is None:
            return None
        if observation.control_running is not None:
            return self._record(observation, Decision(ACTION_ADVANCE), "wait_for_committed_control")
        pending = observation.pending_reclaim
        if pending is not None:
            if pending.id not in self._reclaim_decisions:
                self._reclaim_decisions.add(pending.id)
                return self._record(observation, ControlDecision("DECIDE", request_id=pending.id,
                    reason="observed_reclaim_request"), "reclaim_decision_cost", request_id=pending.id)
            if spec.reclaim_response == "decline":
                return self._record(observation, ControlDecision("DECLINE_RECLAIM", request_id=pending.id,
                    reason="explicit_strategy_decline"), "decline_public_reclaim", request_id=pending.id)
            memory = next(entry for entry in observation.memory if entry.domain == "vram")
            if memory.capacity_bytes is None:
                raise Unsupported("reclaim requires a finite public VRAM capacity")
            shortfall = memory.used_bytes - memory.external_bytes + pending.target_external_bytes - memory.capacity_bytes
            if shortfall <= 0:
                return self._record(observation, ControlDecision("ACK_RECLAIM", request_id=pending.id),
                    "accept_safely_reclaimed_budget", request_id=pending.id, physical_shortfall_bytes=shortfall)
            tensors = {tensor.id: tensor for tensor in observation.tensors}
            candidates = [tid for tid in observation.safe_weight_release_candidates
                          if tid in tensors and tensors[tid].role == "weight"]
            if candidates:
                if spec.release_order == "largest_first":
                    rank = lambda tid: (-self._allocation(tensors[tid], "vram"), tid)
                elif spec.release_order == "history_frequency":
                    frequency = {}
                    for entry in observation.route_history:
                        for expert in entry.expert_ids:
                            owner = (entry.layer_id, expert)
                            frequency[owner] = frequency.get(owner, 0) + 1
                    rank = lambda tid: (frequency.get((self._catalogue.get(tid, ("", "", None))[0],
                                                      self._catalogue.get(tid, ("", "", None))[2]), 0),
                                        self._last_used.get(tid, -1), tid)
                else:
                    rank = lambda tid: (self._last_used.get(tid, -1), tid)
                chosen = min(candidates, key=rank)
                return self._record(observation, Decision(ACTION_EVICT, chosen, location="vram"),
                    "reclaim_model_certified_clean_weight", request_id=pending.id, tensor=chosen,
                    release_order=spec.release_order, physical_shortfall_bytes=shortfall)
            if observation.t_ns < pending.deadline_ns:
                return self._record(observation, ControlDecision("WAIT_UNTIL", request_id=pending.id,
                    until_ns=pending.deadline_ns, reason="wait_for_safe_release_or_public_deadline"),
                    "wait_for_reclaim_capacity", request_id=pending.id, physical_shortfall_bytes=shortfall)
            return self._record(observation, ControlDecision("DECLINE_RECLAIM", request_id=pending.id,
                reason="no_safe_clean_weight_capacity_at_deadline"), "unmet_public_reclaim", request_id=pending.id)
        if self._observed_cpu is None:
            self._observed_cpu = observation.cpu_available
        elif self._observed_cpu != observation.cpu_available:
            self._observed_cpu = observation.cpu_available
            return self._record(observation, ControlDecision("DECIDE", reason="observed_cpu_availability_change"),
                "cpu_change_decision_cost", cpu_response=spec.cpu_response)
        return None

    def _protected(self, observation, tensors, current_inputs=()):
        protected = super()._protected(observation, tensors, current_inputs)
        # Cache groups remain intact throughout a committed expert execution.
        # Stage-local tensor mode can evict later, but never a current reader.
        if self.spec.cache_unit == "expert_group":
            for key, operations in self._regions.items():
                if key not in self._region_devices or all(op.status == OP_DONE for op in operations):
                    continue
                protected.update(tid for op in operations for tid in op.inputs if tensors[tid].role == "weight")
        return protected

    def _declared_release(self, observation, tensors):
        decision = super()._declared_release(observation, tensors)
        if decision is not None or self.spec.boundary_release == "disabled" or not observation.allow_eviction:
            return decision
        protected = self._protected(observation, tensors)
        # A ready backup makes a copy recoverable, but it does not make that
        # copy idle: pending public consumers may have just loaded it. Releasing
        # it here would produce a load/release loop before compute can start.
        protected.update(tid for op in observation.operations if op.status == OP_NOT_STARTED for tid in op.inputs)
        for tid, location in getattr(observation, "recoverable_release_candidates", ()):
            tensor = tensors.get(tid)
            if (tensor is None or tid in protected or tensor.goal_location == location or
                    tensor.status_at(location) != COPY_READY or
                    any(status not in (COPY_ABSENT, COPY_READY) for _, status in tensor.copies)):
                continue
            routes = {(price.src, price.dst) for price in tensor.transfers}
            backups = {other for other, status in tensor.copies if other != location and status == COPY_READY}
            if not any((source, location) in routes or (source == "ssd" and location == "vram" and
                    ("ssd", "dram") in routes and ("dram", "vram") in routes) for source in backups):
                continue
            self._composition_records.append({"t_ns": observation.t_ns,
                "reason": "release_recoverable_copy_keep_home", "tensor_id": tid, "location": location,
                "goal_location_preserved": tensor.goal_location, "ready_backups": sorted(backups)})
            return Decision(ACTION_EVICT, tid, location=location)
        return None

    def _cache_units(self):
        if self.spec.cache_unit == "expert_group":
            return self._groups
        return {("tensor", tid): (tid,) for tid in self._tensor_groups}

    def _budget_for_additions(self, tensors, additions=()):
        additions = set(additions)
        return sum(sum(tensors[tid].storage_bytes for tid in tids) for tids in self._cache_units().values()
                   if any(tid in additions or tensors[tid].status_at("vram") != COPY_ABSENT for tid in tids))

    def _eviction(self, observation, tensors, location, protected):
        if self.policy.cache_rule == "none" or not observation.allow_eviction:
            return None
        units = self._cache_units()
        frequency = self._frequency()
        candidates = []
        for key, tids in units.items():
            resident = [tid for tid in tids if tensors[tid].status_at(location) != COPY_ABSENT]
            if not resident or any(tid in protected for tid in tids):
                continue
            valid = True
            for tid in resident:
                tensor = tensors[tid]
                routes = {(price.src, price.dst) for price in tensor.transfers}
                ready = {other for other, status in tensor.copies if other != location and status == COPY_READY}
                recoverable = any((source, location) in routes or (source == "ssd" and location == "vram"
                    and ("ssd", "dram") in routes and ("dram", "vram") in routes) for source in ready)
                if tensor.status_at(location) != COPY_READY or tensor.goal_location == location or not recoverable:
                    valid = False
                    break
            if not valid:
                continue
            layer, _, expert = self._catalogue.get(tids[0], ("", "", None))
            last = max(self._last_used.get(tid, -1) for tid in tids)
            if self.policy.cache_rule == "mrs":
                # Non-expert weights have no router scores. They can still be
                # managed deterministically by age under this mixed catalogue.
                score = self._mrs_scores.get((layer, expert), 0.0)
            elif self.policy.cache_rule == "history_frequency":
                score = frequency[(layer, expert)] if expert is not None else 0
            else:
                score = 0
            # Finish an already started group eviction before choosing another
            # victim: partial groups keep their full budget reservation.
            pending = key == self._pending_eviction_group
            candidates.append(((not pending, score, last, str(key)), key, min(resident)))
        if not candidates:
            return None
        _, key, tid = min(candidates)
        self._pending_eviction_group = key if self.spec.cache_unit == "expert_group" else None
        self._composition_records.append({"t_ns": observation.t_ns, "reason": "cache_unit_eviction",
            "unit": self.spec.cache_unit, "members": list(units[key]), "next_tensor": tid,
            "group_budget_retained_until_last_member_absent": self.spec.cache_unit == "expert_group"})
        return Decision(ACTION_EVICT, tid, location=location)

    def _demand(self, operation, observation, tensors, device, *, queue=None):
        weights = tuple(tid for tid in operation.inputs if tensors[tid].role == "weight")
        if device == "gpu" and self.spec.cache_budget_bytes is not None and weights:
            footprint = self._budget_for_additions(tensors, weights)
            if footprint > self.spec.cache_budget_bytes:
                protected = self._protected(observation, tensors, operation.inputs)
                eviction = self._eviction(observation, tensors, "vram", protected)
                if eviction:
                    return self._record(observation, eviction, "demand_weight_budget_eviction",
                                        operation=operation.id, payload_budget_bytes=self.spec.cache_budget_bytes,
                                        prospective_payload_bytes=footprint)
                unit_minimum = sum(sum(tensors[tid].storage_bytes for tid in tids)
                    for tids in self._cache_units().values() if set(tids) & set(weights))
                if unit_minimum > self.spec.cache_budget_bytes:
                    raise Unsupported(f"declared GPU cache unit for {operation.id} exceeds its explicit weight budget")
                return None
        decision = super()._demand(operation, observation, tensors, device, queue=queue)
        if decision is not None and decision.kind == ACTION_COMPUTE and self.spec.cache_unit == "expert_group":
            for tid in weights:
                for member in self._groups[self._tensor_groups[tid]]:
                    self._last_used[member] = self._clock
        if (decision is not None and decision.kind in (ACTION_COPY, "READ_SSD", ACTION_COMPUTE)
                and self.spec.reconfiguration is not None and self.spec.reconfiguration.cpu_response == "current_cost"):
            if operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int):
                self._region_devices[self._region_key(operation)] = device
            else:
                self._operation_devices[operation.id] = device
        return decision

    @staticmethod
    def _ready(operation, tensors):
        return operation.status == OP_NOT_STARTED and all(
            any(status == COPY_READY for _, status in tensors[tid].copies) for tid in operation.inputs)

    def _hybrimoe_regions(self, observation, tensors):
        committed, fresh = [], []
        for key, operations in self._regions.items():
            ready = [op for op in operations if self._ready(op, tensors)]
            if not ready:
                continue
            first = ready[0]
            if key in self._region_devices:
                committed.append((key, first))
            else:
                self._region_parts(first, observation, tensors)
                fresh.append((key, first))
        for key, operation in committed:
            decision = self._demand(operation, observation, tensors, self._region_devices[key],
                                    queue={"rule": "continue_committed_expert_region", "region": list(key)})
            if decision:
                return decision
        def cached(entry):
            _, weights, _, _ = self._region_parts(entry[1], observation, tensors)
            return bool(weights) and all(tensors[tid].status_at("vram") == COPY_READY for tid in weights)
        gpu = sorted((entry for entry in fresh if cached(entry)), key=lambda entry: (-entry[1].token_count, entry[0]))
        uncached = [entry for entry in fresh if not cached(entry)]
        cpu = sorted(uncached, key=lambda entry: (entry[1].token_count, entry[0]))
        transfer = sorted(uncached, key=lambda entry: (-entry[1].token_count, entry[0]))
        queues = {"gpu_cached_desc": [list(key) for key, _ in gpu],
                  "cpu_uncached_asc": [list(key) for key, _ in cpu],
                  "transfer_uncached_desc": [list(key) for key, _ in transfer]}
        for device, entries, rule in (("gpu", gpu, "gpu_cached_desc"),
                                      ("cpu", cpu if observation.cpu_available else [], "cpu_uncached_asc")):
            for key, operation in entries:
                paths = self._region_paths(operation, observation, tensors)
                if not any(path["device"] == device and path["budget_feasible"] and path["physical_feasible"] for path in paths):
                    continue
                decision = self._demand(operation, observation, tensors, device,
                                        queue={"rule": rule, "queues": queues})
                if decision:
                    # Capacity evictions are preparatory, not a compute/transfer
                    # claim. Only an actual region input load or compute commits.
                    if decision.kind != ACTION_EVICT:
                        self._region_devices[key] = device
                        self._region_estimates[key] = paths
                    return decision
        if observation.cpu_available and not cpu:
            for key, operation in sorted(gpu, key=lambda entry: (entry[1].token_count, entry[0])):
                paths = self._region_paths(operation, observation, tensors)
                if not any(path["device"] == "cpu" and path["physical_feasible"] for path in paths):
                    continue
                decision = self._demand(operation, observation, tensors, "cpu",
                    queue={"rule": "cpu_cached_asc_when_uncached_queue_empty", "queues": queues})
                if decision:
                    if decision.kind != ACTION_EVICT:
                        self._region_devices[key] = "cpu"
                        self._region_estimates[key] = paths
                    return decision
        for key, operation in transfer:
            _, weights, _, _ = self._region_parts(operation, observation, tensors)
            paths = self._region_paths(operation, observation, tensors)
            if not any(path["device"] == "gpu" and path["budget_feasible"] and path["physical_feasible"] for path in paths):
                continue
            protected = self._protected(observation, tensors, operation.inputs)
            for tid in weights:
                loading = self._load_step(tensors[tid], "vram")
                if loading is None:
                    continue
                decision, price, resource = loading
                if not self._resource_available(observation, resource):
                    continue
                fits, eviction = self._fit_or_evict(observation, tensors, price.dst,
                                                    self._allocation(tensors[tid], price.dst), protected)
                if eviction:
                    return self._record(observation, eviction, "region_transfer_capacity_eviction")
                if fits:
                    self._region_devices[key] = "gpu"
                    self._region_estimates[key] = paths
                    return self._record(observation, decision, "promote_complete_expert_region",
                        region=list(key), tensor=tid, queue_priority={"rule": "transfer_uncached_desc", "queues": queues},
                        public_duration_ns=price.duration_ns)
        return None

    def _prefetch(self, observation, tensors):
        if not self.policy.prefetch_enabled:
            return None
        order = getattr(observation, "layer_order", ())
        active = getattr(observation, "active_invocations", ())
        if not isinstance(order, tuple) or not order or len(set(order)) != len(order):
            raise Unsupported("cross-layer prefetch needs an explicit immutable static layer order")
        targets, target_scopes = set(), {}
        for invocation, layer in active:
            if layer not in order:
                raise InvalidInput("active invocation layer is absent from public static order")
            start = order.index(layer)
            following = order[start + 1:start + 1 + self.spec.lookahead_layers]
            targets.update(following)
            for target in following:
                target_scopes.setdefault(target, invocation)
        if not targets:
            return None
        frequency = self._frequency()
        units = self._cache_units()
        candidates = []
        for key, tids in units.items():
            layer, _, expert = self._catalogue.get(tids[0], ("", "", None))
            if layer not in targets:
                continue
            if expert is None:
                if not self.spec.include_dense_weights:
                    continue
                score = 0
            else:
                score = (self._mrs_scores.get((layer, expert), 0.0) if self.spec.prefetch_ranking == "mrs"
                         else frequency[(layer, expert)])
                if score <= 0:
                    continue
            candidates.append(((-score, order.index(layer), str(key)), key, tids, score))
        for _, key, tids, score in sorted(candidates):
            source_scope = target_scopes[self._catalogue[tids[0]][0]]
            scoped_key = (source_scope, key)
            counted = sum(scope == source_scope for scope, _ in self._prefetch_units)
            if scoped_key not in self._prefetch_units and counted >= self.policy.prefetch_max_count:
                continue
            for tid in tids:
                tensor = tensors[tid]
                loading = self._load_step(tensor, "vram")
                if loading is None:
                    continue
                decision, price, resource = loading
                if not self._resource_available(observation, resource):
                    continue
                hop = (source_scope, tid, price.route)
                if hop in self._prefetch_hops:
                    continue
                protected = self._protected(observation, tensors, tids)
                for op in observation.operations:
                    if op.status == OP_NOT_STARTED:
                        protected.update(op.inputs)
                footprint = self._budget_for_additions(tensors, tids)
                if footprint > self.spec.cache_budget_bytes:
                    eviction = self._eviction(observation, tensors, "vram", protected)
                    if eviction:
                        return self._record(observation, eviction, "cross_layer_prefetch_budget_eviction")
                    continue
                fits, eviction = self._fit_or_evict(observation, tensors, price.dst,
                                                   self._allocation(tensor, price.dst), protected)
                if eviction:
                    return self._record(observation, eviction, "cross_layer_prefetch_capacity_eviction")
                if not fits:
                    continue
                self._prefetch_units.add(scoped_key)
                self._predicted.add(tid)
                self._prefetch_hops.add(hop)
                self._last_used[tid] = self._clock
                return self._record(observation, decision, "cross_layer_public_history_prefetch",
                    tensor=tid, layer_id=self._catalogue[tid][0], expert_id=self._catalogue[tid][2],
                    ranking=self.spec.prefetch_ranking, public_priority=score, cache_unit=self.spec.cache_unit,
                    source_invocation_id=source_scope, max_count_scope="per_active_source_invocation",
                    unit_members=list(tids), prospective_payload_bytes=footprint,
                    information_scope="static_next_layer_catalogue_and_revealed_history_only",
                    public_duration_ns=price.duration_ns)
        return None

    def _propose(self, observation):
        DeclarativePolicy.observe_final(self, self._fragment_catalogue(observation))
        self._clock += 1
        if observation.complete or observation.error_code is not None:
            return None
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        dynamic_cpu = self.spec.reconfiguration is not None and self.spec.reconfiguration.cpu_response == "current_cost"
        use_hybri_queue = self.policy.expert_device_rule == "hybrimoe_queues" and not dynamic_cpu
        resident = self._budget_for_additions(tensors)
        if self.spec.cache_budget_bytes is not None and resident > self.spec.cache_budget_bytes:
            eviction = self._eviction(observation, tensors, "vram", self._protected(observation, tensors))
            if eviction:
                return self._record(observation, eviction, "existing_weight_budget_eviction",
                                    payload_budget_bytes=self.spec.cache_budget_bytes, resident_payload_bytes=resident)
            raise Unsupported("existing GPU weight cache exceeds the declared budget and cannot be safely evicted")
        for operation in observation.operations:
            self._placement(operation)
        release = self._declared_release(observation, tensors) or self._release_last_use(observation, tensors)
        if release:
            return release
        for operation in observation.operations:
            if not self._ready(operation, tensors):
                continue
            if use_hybri_queue and operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int):
                continue
            for device in self._devices(operation, observation, tensors):
                decision = self._demand(operation, observation, tensors, device)
                if decision:
                    return decision
        if use_hybri_queue:
            decision = self._hybrimoe_regions(observation, tensors)
            if decision:
                return decision
        for tid, location in observation.deliveries:
            tensor = tensors[tid]
            loading = self._load_step(tensor, location)
            if loading is None:
                continue
            decision, price, resource = loading
            if not self._resource_available(observation, resource):
                continue
            fits, eviction = self._fit_or_evict(observation, tensors, price.dst,
                self._allocation(tensor, price.dst), self._protected(observation, tensors, (tid,)))
            if eviction:
                return self._record(observation, eviction, "delivery_capacity_eviction")
            if fits:
                return self._record(observation, decision, "deliver_known_output", tensor=tid, route=price.route)
        prefetch = self._prefetch(observation, tensors)
        if prefetch:
            return prefetch
        return self._record(observation, Decision(ACTION_ADVANCE), "wait_for_public_progress",
                            running_resources=[running.resource for running in observation.running])

    def __call__(self, observation):
        if not isinstance(observation, CausalObservation):
            raise InvalidInput("CompositionPolicy requires public CausalObservation")
        self._public_catalogue(observation)
        control = self._reconfiguration_step(observation)
        if control is not None:
            return control
        estimated = self._estimate_observation(observation)
        before = self._intent_snapshot()
        decision = self._propose(estimated)
        guarded = self._slot_guard(observation, {tensor.id: tensor for tensor in observation.tensors}, decision)
        if guarded != decision:
            self._restore_intent(before)
        return guarded

    def _intent_snapshot(self):
        """Snapshot only tentative intent; immutable observations need no copy.

        Audit rows and path descriptions are append-only. Copying an entire
        growing audit for every decision makes a multi-round run quadratic and
        needlessly duplicates every region estimate. Mutable dict/set intent
        is copied, while existing immutable values and path lists are shared.
        """
        dictionary_fields = ("_history", "_last_used", "_fiddler_devices", "_fiddler_paths",
                             "_mrs_scores", "_region_devices", "_region_estimates", "_operation_devices")
        set_fields = ("_predicted", "_prefetch_hops", "_revealed_demand", "_hybri_transfer_claims",
                      "_mrs_seen", "_prefetch_units")
        scalar_fields = ("_complete", "_clock", "_pending_eviction_group")
        append_fields = ("_records", "_mrs_updates", "_composition_records")
        return ({name: dict(getattr(self, name)) for name in dictionary_fields} |
                {name: set(getattr(self, name)) for name in set_fields} |
                {name: getattr(self, name) for name in scalar_fields},
                {name: len(getattr(self, name)) for name in append_fields})

    def _restore_intent(self, snapshot):
        fields, lengths = snapshot
        self.__dict__.update(fields)
        for name, length in lengths.items():
            del getattr(self, name)[length:]

    def audit_document(self):
        result = DeclarativePolicy.audit_document(self)
        # Legacy parsing uses a CPU placeholder for execution_binding. It is
        # never an execution choice and must not appear as a second strategy.
        result.pop("policy", None)
        result.update(format_version="composition-policy-audit-0.1",
            authoritative_strategy_field="composition", composition=composition_policy_document(self.spec),
            transport_decisions=list(self._transport_records), cache_decisions=list(self._composition_records),
            cache_unit=self.spec.cache_unit, cache_budget_scope="materialized_payload_bytes",
            group_partial_residency_reserves_full_group_budget=self.spec.cache_unit == "expert_group",
            region_bindings=[{"invocation_id": key[0], "region_id": key[1], "device": device,
                              "paths": self._region_estimates.get(key, [])}
                             for key, device in sorted(self._region_devices.items())],
            prefetch_information_scope="public_static_architecture_and_revealed_history_only",
            prefetch_max_count_scope="per_active_source_invocation",
            engine_path_estimates_are_contention_free=True,
            claims={"search": False, "optimality": False, "paper_strategy_reproduced": False,
                    "physical_calibration": False})
        if self.spec.reconfiguration is not None:
            result.update(reconfiguration=reconfiguration_document(self.spec.reconfiguration),
                operation_bindings=[{"operation_id": oid, "device": device}
                                    for oid, device in sorted(self._operation_devices.items())],
                dynamic_device_estimate_scope="current_declared_compute_and_isolated_input_transfers",
                decision_cost_scope="one_DECIDE_per_observed_reclaim_or_cpu_change",
                cpu_task_rebinding="forbidden_after_first_demand_preparation_or_compute_command")
        return result
