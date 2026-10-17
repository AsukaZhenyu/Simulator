"""Explainable policies that use only frozen, currently public observations.

The executor makes every device, load and eviction choice explicitly. It is a
deterministic heuristic, not an optimizer and not a paper reproduction.
"""
from __future__ import annotations

from collections import Counter

from llm_infer_model.m2.causal import CausalObservation, RevealedRoute
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT,
    ACTION_READ_SSD, ACTION_WRITE_SSD,
)
from llm_infer_model.tensor.spec import (
    COPY_ABSENT, COPY_READY, OP_NOT_STARTED, OP_DONE,
    RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE, RESOURCE_PCIE_COPY, RESOURCE_SSD_IO,
    InvalidInput, Unsupported,
)

from .causal import Decision
from .policy_ir import (
    EXPERT_COMPUTE_KINDS, PolicyIR, history_document, parse_history, parse_policy,
    policy_document,
)


class DeclarativePolicy:
    """A bounded driver repeatedly calls this object with a public snapshot.

    Fixed placement uses every explicit per-kind binding. Expert resident/cost
    rules apply only to expert compute stages; the binding breaks ties and is
    the residency rule's fallback. Gather/scatter/combine retain their binding.
    History is deduplicated by (request, layer, position), never by task names.
    """

    def __init__(self, policy: PolicyIR, *, history: tuple[RevealedRoute, ...] = ()):
        if not isinstance(policy, PolicyIR):
            raise InvalidInput("DeclarativePolicy requires parsed PolicyIR")
        # Validate directly constructed frozen objects as strictly as JSON.
        self.policy = parse_policy(policy_document(policy))
        validated = parse_history(history_document(tuple(history)))
        self._history = {(entry.request_id, entry.layer_id, entry.absolute_position): entry
                         for entry in validated}
        self._placements = dict(policy.placements)
        self._request_placements = {request: dict(bindings) for request, bindings in self.policy.request_placements}
        self._records: list[dict] = []
        self._last_used: dict[str, int] = {}
        self._predicted: set[str] = set()
        self._prefetch_hops: set[tuple[str, str]] = set()
        self._revealed_demand: set[str] = set()
        self._complete = False
        self._clock = 0
        self._fiddler_devices: dict[str, str] = {}
        self._fiddler_paths: dict[str, list[dict]] = {}
        self._hybri_transfer_claims: set[str] = set()
        self._mrs_scores: dict[tuple[str, int], float] = {}
        self._mrs_seen: set[tuple[str, str, int]] = set()
        self._mrs_updates: list[dict] = []

    def _placement(self, operation):
        bindings = self._request_placements.get(operation.request_id, {})
        if operation.kind in bindings:
            return bindings[operation.kind]
        if operation.kind not in self._placements:
            raise Unsupported(f"policy has no explicit placement for {operation.kind}")
        return self._placements[operation.kind]

    def history_document(self) -> dict:
        rows = tuple(self._history.values())
        entries = (rows if any(getattr(entry, "router_scores_scope", "not_reported") != "not_reported"
                               for entry in rows) else
                   tuple(sorted(rows, key=lambda entry: (entry.request_id, entry.layer_id, entry.absolute_position))))
        return history_document(entries)

    def audit_document(self) -> dict:
        return {"format_version": "causal-policy-audit-0.1",
                "policy": policy_document(self.policy), "decisions": list(self._records),
                "history": self.history_document(),
                "paper_mechanisms": {"expert_rule": self.policy.expert_device_rule,
                                     "cache_rule": self.policy.cache_rule,
                                     "hybrimoe_allocation_optimizer": False,
                                     "cross_layer_gate_predictor": False},
                "mrs_updates": list(self._mrs_updates),
                "mrs_priorities": [{"layer_id": layer, "expert_id": expert, "priority": priority}
                                   for (layer, expert), priority in sorted(self._mrs_scores.items())],
                "predictions": [{"tensor_id": tensor,
                                 "outcome": ("revealed_demand" if tensor in self._revealed_demand
                                             else "miss" if self._complete else "not_yet_demanded")}
                                for tensor in sorted(self._predicted)],
                "claims": {"search": False, "optimality": False,
                           "paper_strategy_reproduced": False}}

    def _record(self, observation, decision, reason, **evidence):
        operation = evidence.get("operation")
        if operation in self._fiddler_paths:
            evidence["fiddler_complete_paths"] = self._fiddler_paths[operation]
            evidence["fiddler_chosen_device"] = self._fiddler_devices[operation]
        evidence["memory_available_bytes"] = {entry.domain: entry.available_bytes
                                             for entry in observation.memory}
        evidence["cpu_available"] = observation.cpu_available
        self._records.append({"index": len(self._records), "t_ns": observation.t_ns,
                              "reason": reason, "evidence": evidence})
        return decision

    def _update_history(self, observation):
        for entry in observation.route_history:
            key = (entry.request_id, entry.layer_id, entry.absolute_position)
            previous = self._history.get(key)
            if previous is not None and previous != entry:
                raise InvalidInput("revealed history contradicts an already recorded route")
            self._history[key] = entry

    def observe_final(self, observation: CausalObservation) -> None:
        """Record a terminal/budget-stop public snapshot without choosing work."""
        if not isinstance(observation, CausalObservation):
            raise InvalidInput("policy requires public CausalObservation")
        self._update_history(observation)
        self._update_mrs(observation)
        self._complete = observation.complete
        expert_weights = {expert.weight_tensor_id for expert in observation.experts}
        self._revealed_demand.update(tid for operation in observation.operations
                                     for tid in operation.inputs if tid in expert_weights)

    def _frequency(self):
        return Counter((entry.layer_id, expert) for entry in self._history.values()
                       for expert in entry.expert_ids)

    def _update_mrs(self, observation):
        if self.policy.cache_rule != "mrs":
            return
        catalog = {}
        for expert in observation.experts:
            if isinstance(expert.expert_id, int):
                catalog.setdefault(expert.layer_id, set()).add(expert.expert_id)
        # Replay a declared past in its explicit input order; newly revealed
        # router rows are ordered deterministically within a single snapshot.
        for entry in self._history.values():
            key = (entry.request_id, entry.layer_id, entry.absolute_position)
            if key in self._mrs_seen:
                continue
            if entry.layer_id not in catalog:
                continue
            scope = getattr(entry, "router_scores_scope", "not_reported")
            scores = dict(getattr(entry, "router_scores", ()))
            if scope != "all_experts" or set(scores) != catalog[entry.layer_id]:
                raise Unsupported("MRS requires revealed all-expert router scores; selected top-k is insufficient")
            if self.policy.mrs_top_p > len(scores):
                raise InvalidInput("MRS top_p exceeds the public expert catalogue")
            chosen = set(sorted(scores, key=lambda expert: (-scores[expert], expert))[:self.policy.mrs_top_p])
            alpha = self.policy.mrs_alpha
            before, after = {}, {}
            for expert in sorted(scores):
                previous = self._mrs_scores.get((entry.layer_id, expert), 0.0)
                sample = scores[expert] if expert in chosen else 0.0
                value = alpha * sample + (1 - alpha) * previous
                before[str(expert)] = previous
                after[str(expert)] = value
                self._mrs_scores[(entry.layer_id, expert)] = value
            self._mrs_seen.add(key)
            self._mrs_updates.append({"request_id": entry.request_id, "layer_id": entry.layer_id,
                                      "absolute_position": entry.absolute_position, "alpha": alpha,
                                      "top_p": self.policy.mrs_top_p, "top_p_experts": sorted(chosen),
                                      "router_scores_scope": scope, "before": before, "after": after})

    @staticmethod
    def _allocation(tensor, location):
        # SSD IO payload is charged by Modeling, but SSD capacity is explicitly
        # outside this runtime's DRAM/VRAM capacity ledger.
        if location == "ssd":
            return 0
        allocations = dict(getattr(tensor, "allocation_bytes", ()))
        if location not in allocations:
            raise Unsupported(f"public aligned allocation missing for {tensor.id}/{location}")
        return allocations[location]

    @staticmethod
    def _free(observation, location):
        if location == "ssd":
            return None
        memory = next((entry for entry in observation.memory if entry.domain == location), None)
        if memory is None:
            raise Unsupported(f"public memory ledger missing for {location}")
        return memory.available_bytes

    @staticmethod
    def _resource_available(observation, resource, compute=False):
        busy = {entry.resource for entry in observation.running}
        if resource in busy:
            return False
        compute_busy = bool(busy & {RESOURCE_CPU_COMPUTE, RESOURCE_GPU_COMPUTE})
        transfer_busy = bool(busy & {RESOURCE_PCIE_COPY, RESOURCE_SSD_IO})
        if not observation.allow_copy_compute_overlap and (
                (compute and transfer_busy) or (not compute and compute_busy)):
            return False
        if compute and not observation.allow_cpu_gpu_overlap and compute_busy:
            return False
        return True

    @staticmethod
    def _load_step(tensor, destination):
        """Public shortest available route (SSD->VRAM is two explicit actions)."""
        if tensor.status_at(destination) != COPY_ABSENT:
            return None
        paths = {"vram": (("dram", "vram"), ("ssd", "dram")),
                 "dram": (("vram", "dram"), ("ssd", "dram")),
                 "ssd": (("dram", "ssd"), ("vram", "dram"))}
        candidates = []
        prices = {(price.src, price.dst): price for price in tensor.transfers}
        for src, dst in paths[destination]:
            if tensor.status_at(src) != COPY_READY or tensor.status_at(dst) != COPY_ABSENT:
                continue
            price = prices.get((src, dst))
            if price is None:
                continue
            final = prices.get((dst, destination)) if dst != destination else None
            if dst != destination and final is None:
                continue
            total = price.duration_ns + (final.duration_ns if final is not None else 0)
            if src == "ssd":
                item = (Decision(ACTION_READ_SSD, tensor.id), price, RESOURCE_SSD_IO)
            elif dst == "ssd":
                item = (Decision(ACTION_WRITE_SSD, tensor.id), price, RESOURCE_SSD_IO)
            else:
                item = (Decision(ACTION_COPY, tensor.id, src=src, dst=dst), price, RESOURCE_PCIE_COPY)
            candidates.append((total, src, dst, item))
        return min(candidates, key=lambda item: item[:3])[3] if candidates else None

    @staticmethod
    def _transfer_estimate(tensor, destination):
        if tensor.status_at(destination) == COPY_READY:
            return 0
        routes = {(price.src, price.dst): price.duration_ns for price in tensor.transfers}
        choices = []
        for src, status in tensor.copies:
            if status != COPY_READY:
                continue
            if (src, destination) in routes:
                choices.append(routes[(src, destination)])
            elif (src, destination) == ("ssd", "vram") and (
                    ("ssd", "dram") in routes and ("dram", "vram") in routes):
                choices.append(routes[("ssd", "dram")] + routes[("dram", "vram")])
        return min(choices) if choices else None

    def _devices(self, operation, observation, tensors):
        preferred = self._placement(operation)
        available = [price for price in operation.prices
                     if price.device != "cpu" or observation.cpu_available]
        dynamic = operation.kind in EXPERT_COMPUTE_KINDS and self.policy.expert_device_rule != "fixed"
        if self.policy.expert_device_rule in ("fiddler_paths", "hybrimoe_queues"):
            if operation.kind == "expert_ffn" and isinstance(operation.expert_id, int):
                self._check_paper_expert(operation)
                if self.policy.expert_device_rule == "fiddler_paths":
                    return self._fiddler_device(operation, observation, tensors)
            dynamic = False
        if not dynamic:
            if preferred not in tuple(price.device for price in operation.prices):
                raise Unsupported(f"explicit placement {preferred} has no exact public cost for {operation.id}")
            return tuple(price.device for price in available if price.device == preferred)
        weights = [tensors[tid] for tid in operation.inputs if tensors[tid].role == "weight"]
        def score(price):
            location = "dram" if price.device == "cpu" else "vram"
            movement = [self._transfer_estimate(tensors[tid], location) for tid in operation.inputs]
            if any(value is None for value in movement):
                return (2, 0, price.device != preferred, price.device)
            cost = price.duration_ns + sum(movement)
            resident = bool(weights) and all(weight.status_at(location) == COPY_READY for weight in weights)
            if self.policy.expert_device_rule == "resident_first":
                return (0 if resident else 1, price.device != preferred, cost, price.device)
            return (0, cost, price.device != preferred, price.device)
        return tuple(price.device for price in sorted(available, key=score))

    @staticmethod
    def _check_paper_expert(operation):
        if operation.execution_variant != "opaque" or operation.kind != "expert_ffn" or not operation.region_id:
            raise Unsupported("paper expert rules require selected_rows with opaque expert_ffn")
        if operation.token_count < 1 or len(operation.work_items) != operation.token_count:
            raise Unsupported("paper expert rules require revealed exact token rows/count")

    def _fiddler_device(self, operation, observation, tensors):
        if operation.id in self._fiddler_devices:
            chosen = self._fiddler_devices[operation.id]
            return () if chosen == "cpu" and not observation.cpu_available else (chosen,)
        consumers = [consumer for consumer in observation.operations
                     if set(consumer.inputs) & set(operation.outputs)]
        if not consumers or any(consumer.kind != "scatter_rows" or self._placement(consumer) != "gpu"
                                for consumer in consumers):
            raise Unsupported("Fiddler path comparison requires explicit GPU scatter boundary")
        group_inputs = [tid for tid in operation.inputs if tensors[tid].role != "weight"]
        producers = [producer for producer in observation.operations if set(producer.outputs) & set(group_inputs)]
        if not producers or any(producer.kind != "gather_rows" or self._placement(producer) != "gpu"
                                for producer in producers):
            raise Unsupported("Fiddler path comparison requires explicit GPU selected-row gather boundary")
        weights = [tensors[tid] for tid in operation.inputs if tensors[tid].role == "weight"]
        paths = []
        for device in ("cpu", "gpu"):
            price = next((price for price in operation.prices if price.device == device), None)
            if price is None:
                raise Unsupported(f"Fiddler needs exact {device} expert compute cost")
            location = "dram" if device == "cpu" else "vram"
            weight_ns = [self._transfer_estimate(weight, location) for weight in weights]
            input_ns = [self._transfer_estimate(tensors[tid], location) for tid in group_inputs]
            output_ns = 0
            for tid in operation.outputs:
                if location != "vram":
                    route = next((route for route in tensors[tid].transfers
                                  if route.src == location and route.dst == "vram"), None)
                    if route is None:
                        raise Unsupported("Fiddler CPU output return has no exact H2D cost")
                    output_ns += route.duration_ns
            if any(value is None for value in weight_ns + input_ns):
                raise Unsupported("Fiddler complete expert path has an unavailable public transfer route")
            paths.append({"device": device, "weight_transfer_ns": sum(weight_ns),
                          "selected_input_transfer_ns": sum(input_ns), "compute_ns": price.duration_ns,
                          "output_return_ns": output_ns,
                          "total_ns": sum(weight_ns) + sum(input_ns) + price.duration_ns + output_ns})
        cached = bool(weights) and all(weight.status_at("vram") == COPY_READY for weight in weights)
        chosen = "gpu" if cached or not observation.cpu_available else min(
            paths, key=lambda path: (path["total_ns"], path["device"] != self._placement(operation)))["device"]
        self._fiddler_devices[operation.id] = chosen
        self._fiddler_paths[operation.id] = paths
        return (chosen,)

    def _device_evidence(self, operation, tensors):
        evidence = []
        weights = [tensors[tid] for tid in operation.inputs if tensors[tid].role == "weight"]
        for price in operation.prices:
            location = "dram" if price.device == "cpu" else "vram"
            transfers = [self._transfer_estimate(tensors[tid], location) for tid in operation.inputs]
            movement = None if any(value is None for value in transfers) else sum(transfers)
            evidence.append({"device": price.device, "compute_ns": price.duration_ns,
                             "missing_input_transfer_ns": movement,
                             "local_estimate_ns": None if movement is None else price.duration_ns + movement,
                             "expert_weights_ready": bool(weights) and all(
                                 weight.status_at(location) == COPY_READY for weight in weights)})
        return evidence

    def _protected(self, observation, tensors, current_inputs=()):
        protected = set(current_inputs)
        operations = {operation.id: operation for operation in observation.operations}
        for running in observation.running:
            if running.target_id in operations:
                protected.update(operations[running.target_id].inputs)
            elif running.target_id in tensors:
                protected.add(running.target_id)
        return protected

    def _eviction(self, observation, tensors, location, protected):
        if self.policy.cache_rule == "none" or not observation.allow_eviction:
            return None
        expert_ids = {entry.weight_tensor_id: (entry.layer_id, entry.expert_id)
                      for entry in observation.experts}
        candidates = []
        frequency = self._frequency()
        for tensor_id, expert_id in expert_ids.items():
            tensor = tensors[tensor_id]
            if self.policy.cache_rule == "mrs" and not isinstance(expert_id[1], int):
                # Shared experts have no routed-gate score and are outside MRS.
                continue
            if tensor_id in protected or tensor.status_at(location) != COPY_READY:
                continue
            # A cache inherited from a prior call remains evictable. Preserve
            # the declared goal/source and another recoverable READY copy.
            if location == tensor.goal_location:
                continue
            if not any(other != location and status == COPY_READY for other, status in tensor.copies):
                continue
            # Another ready source must be able to replenish this exact copy.
            routes = {(price.src, price.dst) for price in tensor.transfers}
            ready = {other for other, status in tensor.copies if other != location and status == COPY_READY}
            if not any((source, location) in routes or (source == "ssd" and location == "vram"
                       and ("ssd", "dram") in routes and ("dram", "vram") in routes) for source in ready):
                continue
            if self.policy.cache_rule == "mrs":
                key = (self._mrs_scores.get(expert_id, 0.0), self._last_used.get(tensor_id, -1), tensor_id)
            elif self.policy.cache_rule == "history_frequency":
                key = (frequency[expert_id], self._last_used.get(tensor_id, -1), tensor_id)
            else:
                key = (self._last_used.get(tensor_id, -1), tensor_id)
            candidates.append((key, tensor_id))
        if not candidates:
            return None
        return Decision(ACTION_EVICT, min(candidates)[1], location=location)

    def _fit_or_evict(self, observation, tensors, location, needed, protected):
        free = self._free(observation, location)
        if free is None or needed <= free:
            return True, None
        return False, self._eviction(observation, tensors, location, protected)

    def _release_last_use(self, observation, tensors):
        if (self.policy.release_rule != "last_use" or not observation.allow_eviction
                or not observation.task_graph_complete):
            return None
        protected = self._protected(observation, tensors)
        consumers = {tensor_id: [] for tensor_id in tensors}
        for operation in observation.operations:
            for tensor_id in operation.inputs:
                consumers[tensor_id].append(operation)
        for tensor_id in sorted(tensors):
            tensor = tensors[tensor_id]
            if (tensor.role != "intermediate" or tensor.initial_locations
                    or tensor.goal_location is not None or tensor_id in protected):
                continue
            if any(status not in (COPY_ABSENT, COPY_READY) for _, status in tensor.copies):
                continue
            if any(operation.status != OP_DONE for operation in consumers[tensor_id]):
                continue
            for location in ("dram", "vram"):
                if tensor.status_at(location) == COPY_READY:
                    return self._record(observation, Decision(ACTION_EVICT, tensor_id, location=location),
                                        "release_after_last_visible_use", tensor=tensor_id, location=location,
                                        completed_consumers=[operation.id for operation in consumers[tensor_id]],
                                        task_graph_complete=True)
        return None

    def _demand(self, operation, observation, tensors, device, *, queue=None):
        location = "dram" if device == "cpu" else "vram"
        price = next((price for price in operation.prices if price.device == device), None)
        if price is None:
            raise Unsupported(f"no exact public {device} cost for {operation.id}")
        if device == "cpu" and not observation.cpu_available:
            return None
        workspace = price.allocated_workspace_bytes
        if workspace is None:
            raise Unsupported(f"public aligned workspace missing for {operation.id}/{device}")
        protected = self._protected(observation, tensors, operation.inputs)
        outputs = sum(self._allocation(tensors[tid], location) for tid in operation.outputs)
        missing = [tensors[tid] for tid in operation.inputs if tensors[tid].status_at(location) != COPY_READY]
        extra = {"queue_priority": queue} if queue is not None else {}
        if not missing:
            resource = RESOURCE_CPU_COMPUTE if device == "cpu" else RESOURCE_GPU_COMPUTE
            if not self._resource_available(observation, resource, compute=True):
                return None
            fits, eviction = self._fit_or_evict(observation, tensors, location, outputs + workspace, protected)
            if eviction is not None:
                return self._record(observation, eviction, "demand_capacity_eviction", operation=operation.id,
                                    needed_bytes=outputs + workspace, location=location, **extra)
            if fits:
                for tid in operation.inputs:
                    if tensors[tid].role == "weight":
                        self._last_used[tid] = self._clock
                return self._record(observation, Decision(ACTION_COMPUTE, operation.id, device=device),
                                    "execute_current_demand", operation=operation.id, device=device,
                                    expert_device_rule=self.policy.expert_device_rule,
                                    explicit_kind_placement=self._placement(operation),
                                    device_candidates=self._device_evidence(operation, tensors),
                                    public_duration_ns=price.duration_ns, **extra)
            return None
        for tensor in missing:
            loading = self._load_step(tensor, location)
            if loading is None:
                continue
            decision, transfer, resource = loading
            if not self._resource_available(observation, resource):
                continue
            needed = self._allocation(tensor, transfer.dst)
            if transfer.dst == location:
                needed += outputs + workspace + sum(
                    self._allocation(other, location) for other in missing
                    if other.id != tensor.id and other.status_at(location) == COPY_ABSENT)
            fits, eviction = self._fit_or_evict(observation, tensors, transfer.dst, needed, protected)
            if eviction is not None:
                return self._record(observation, eviction, "demand_capacity_eviction",
                                    operation=operation.id, needed_bytes=needed, location=transfer.dst, **extra)
            if fits:
                return self._record(observation, decision, "load_current_demand", operation=operation.id,
                                    tensor=tensor.id, route=transfer.route, chosen_device=device,
                                    device_candidates=self._device_evidence(operation, tensors),
                                    public_duration_ns=transfer.duration_ns, **extra)
        return None

    def _hybrimoe(self, observation, tensors):
        pending = [operation for operation in observation.operations if operation.kind == "expert_ffn"
                   and isinstance(operation.expert_id, int) and operation.status == OP_NOT_STARTED
                   and all(any(status == COPY_READY for _, status in tensors[tid].copies)
                           for tid in operation.inputs)]
        def cached(operation):
            weights = [tensors[tid] for tid in operation.inputs if tensors[tid].role == "weight"]
            return bool(weights) and all(weight.status_at("vram") == COPY_READY for weight in weights)
        gpu = sorted((op for op in pending if cached(op)), key=lambda op: (-op.token_count, op.id))
        uncached = [op for op in pending if not cached(op)]
        cpu = sorted((op for op in uncached if op.id not in self._hybri_transfer_claims),
                     key=lambda op: (op.token_count, op.id))
        transfer = sorted(uncached, key=lambda op: (-op.token_count, op.id))
        queues = {"gpu_cached_desc": [{"operation": op.id, "tokens": op.token_count} for op in gpu],
                  "cpu_uncached_asc": [{"operation": op.id, "tokens": op.token_count} for op in cpu],
                  "transfer_uncached_desc": [{"operation": op.id, "tokens": op.token_count} for op in transfer]}
        for operation in gpu:
            decision = self._demand(operation, observation, tensors, "gpu",
                                    queue={"rule": "gpu_cached_desc", "queues": queues})
            if decision is not None:
                return decision
        if observation.cpu_available:
            for operation in cpu:
                decision = self._demand(operation, observation, tensors, "cpu",
                                        queue={"rule": "cpu_uncached_asc", "queues": queues})
                if decision is not None:
                    return decision
            if not cpu:
                for operation in sorted(gpu, key=lambda op: (op.token_count, op.id)):
                    decision = self._demand(operation, observation, tensors, "cpu",
                                            queue={"rule": "cpu_cached_asc_when_uncached_queue_empty", "queues": queues})
                    if decision is not None:
                        return decision
        for operation in transfer:
            weights = [tensors[tid] for tid in operation.inputs if tensors[tid].role == "weight"]
            if not weights:
                continue
            gpu_price = next((price for price in operation.prices if price.device == "gpu"), None)
            if gpu_price is None or gpu_price.allocated_workspace_bytes is None:
                raise Unsupported("HybriMoE transfer queue requires exact GPU cost/workspace")
            memory = next(entry for entry in observation.memory if entry.domain == "vram")
            footprint = sum(self._allocation(tensors[tid], "vram")
                            for tid in operation.inputs + operation.outputs) + gpu_price.allocated_workspace_bytes
            if memory.capacity_bytes is not None and footprint > (
                    memory.capacity_bytes - memory.runtime_reserved_bytes - memory.external_bytes):
                continue
            for weight in weights:
                loading = self._load_step(weight, "vram")
                if loading is None:
                    continue
                decision, price, resource = loading
                if not self._resource_available(observation, resource):
                    continue
                protected = self._protected(observation, tensors, operation.inputs)
                needed = self._allocation(weight, price.dst)
                if price.dst == "vram":
                    needed += sum(self._allocation(tensors[tid], "vram")
                                  for tid in operation.outputs) + gpu_price.allocated_workspace_bytes
                    needed += sum(self._allocation(tensors[tid], "vram") for tid in operation.inputs
                                  if tid != weight.id and tensors[tid].status_at("vram") == COPY_ABSENT)
                fits, eviction = self._fit_or_evict(observation, tensors, price.dst, needed, protected)
                evidence = {"operation": operation.id, "queue_priority": {"rule": "transfer_uncached_desc", "queues": queues},
                            "revealed_token_count": operation.token_count}
                if eviction is not None:
                    return self._record(observation, eviction, "demand_capacity_eviction", **evidence)
                if fits:
                    # Remove a promoted expert from CPU's pending queue as soon
                    # as its first explicit loading hop starts; no duplicate
                    # CPU computation while its weight transfer is in flight.
                    self._hybri_transfer_claims.add(operation.id)
                    return self._record(observation, decision, "hybrimoe_promote_high_load_expert",
                                        tensor=weight.id, route=price.route, public_duration_ns=price.duration_ns, **evidence)
        return None

    def __call__(self, observation: CausalObservation) -> Decision | None:
        if not isinstance(observation, CausalObservation):
            raise InvalidInput("policy requires public CausalObservation")
        self.observe_final(observation)
        self._clock += 1
        if observation.complete or observation.error_code is not None:
            return None
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        for operation in observation.operations:
            self._placement(operation)
            if self.policy.expert_device_rule in ("fiddler_paths", "hybrimoe_queues"):
                if operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int):
                    self._check_paper_expert(operation)
        release = self._release_last_use(observation, tensors)
        if release is not None:
            return release
        # Prepare current demand first. Independent CPU/GPU/transfer resources
        # can run concurrently only when public architecture flags permit it.
        for operation in observation.operations:
            if operation.status != OP_NOT_STARTED:
                continue
            if any(not any(status == COPY_READY for _, status in tensors[tid].copies)
                   for tid in operation.inputs):
                continue
            if (self.policy.expert_device_rule == "hybrimoe_queues" and operation.kind == "expert_ffn"
                    and isinstance(operation.expert_id, int)):
                continue
            for device in self._devices(operation, observation, tensors):
                decision = self._demand(operation, observation, tensors, device)
                if decision is not None:
                    return decision
        if self.policy.expert_device_rule == "hybrimoe_queues":
            decision = self._hybrimoe(observation, tensors)
            if decision is not None:
                return decision
        for tensor_id, location in observation.deliveries:
            tensor = tensors[tensor_id]
            if tensor.status_at(location) == COPY_READY:
                continue
            loading = self._load_step(tensor, location)
            if loading is None:
                continue
            decision, transfer, resource = loading
            if self._resource_available(observation, resource):
                fits, eviction = self._fit_or_evict(observation, tensors, transfer.dst,
                                                   self._allocation(tensor, transfer.dst),
                                                   self._protected(observation, tensors, (tensor_id,)))
                if eviction is not None:
                    return self._record(observation, eviction, "delivery_capacity_eviction", tensor=tensor_id)
                if fits:
                    return self._record(observation, decision, "deliver_known_output", tensor=tensor_id,
                                        route=transfer.route)
        prediction = self._prefetch(observation, tensors)
        if prediction is not None:
            return prediction
        return self._record(observation, Decision(ACTION_ADVANCE), "wait_for_public_progress",
                            cpu_available=observation.cpu_available,
                            running_resources=[running.resource for running in observation.running])

    def _prefetch(self, observation, tensors):
        if not self.policy.prefetch_enabled:
            return None
        frequency = self._frequency()
        if not frequency:
            return None
        experts = {(expert.layer_id, expert.expert_id): expert.weight_tensor_id
                   for expert in observation.experts}
        catalog = set(experts.values())
        resident_bytes = sum(self._allocation(tensors[tid], "vram") for tid in catalog
                             if tensors[tid].status_at("vram") != COPY_ABSENT)
        for expert in sorted((expert for expert in experts if frequency[expert]),
                             key=lambda expert: (-frequency[expert], str(expert))):
            tid = experts[expert]
            tensor = tensors[tid]
            if tensor.status_at("vram") == COPY_READY:
                continue
            if tid not in self._predicted and len(self._predicted) >= self.policy.prefetch_max_count:
                continue
            loading = self._load_step(tensor, "vram")
            if loading is None:
                continue
            decision, transfer, resource = loading
            if (tid, transfer.route) in self._prefetch_hops:
                continue
            if not self._resource_available(observation, resource):
                continue
            protected = self._protected(observation, tensors, (tid,))
            for op in observation.operations:
                if op.status == OP_NOT_STARTED:
                    protected.update(op.inputs)
            prospective = resident_bytes + (self._allocation(tensor, "vram")
                                             if tensor.status_at("vram") == COPY_ABSENT else 0)
            if prospective > self.policy.cache_budget_bytes:
                # Prefetch never evicts an expert demanded by a revealed task.
                eviction = self._eviction(observation, tensors, "vram", protected)
                if eviction is not None:
                    return self._record(observation, eviction, "prefetch_cache_budget_eviction",
                                        budget_bytes=self.policy.cache_budget_bytes)
                continue
            fits, eviction = self._fit_or_evict(observation, tensors, transfer.dst,
                                               self._allocation(tensor, transfer.dst), protected)
            if eviction is not None:
                return self._record(observation, eviction, "prefetch_capacity_eviction", expert=expert)
            if not fits:
                continue
            self._predicted.add(tid)
            self._prefetch_hops.add((tid, transfer.route))
            self._last_used[tid] = self._clock
            return self._record(observation, decision, "history_prefetch", expert=expert, tensor=tid,
                                historical_activations=frequency[expert], route=transfer.route,
                                budget_bytes=self.policy.cache_budget_bytes,
                                public_duration_ns=transfer.duration_ns)
        return None
