"""Explicit transport and lifetime rules around an observation-only policy.

Engine rates are isolated payload estimates for choosing a path. The runtime,
not this adapter, arbitrates shared bandwidth and verifies slot/copy locks.
"""
from dataclasses import replace
from copy import deepcopy

from llm_infer_model.tensor.engine import ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT
from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, InvalidInput, Unsupported

from .causal import Decision
from .policy_executor import DeclarativePolicy


TRANSPORT_FIELDS = {"transfer_engines", "slot_reuse", "boundary_release", "fragment_cache_unit"}


def parse_transport(document):
    if not isinstance(document, dict) or set(document) != TRANSPORT_FIELDS:
        raise InvalidInput("transport requires explicit engines, slot reuse, boundary release and fragment cache unit")
    engines = document["transfer_engines"]
    if not isinstance(engines, dict) or set(engines) != {"weight", "input", "intermediate", "output"}:
        raise InvalidInput("transfer_engines must cover all four tensor roles")
    if any(engine not in ("ce", "sm", None) for engine in engines.values()):
        raise Unsupported("transfer engine must be ce, sm or explicit null for legacy transport")
    if document["slot_reuse"] not in ("disabled", "release_unlocked_backed_copy"):
        raise Unsupported("unsupported slot reuse rule")
    if document["boundary_release"] not in ("disabled", "declared_dead_intermediates"):
        raise Unsupported("unsupported boundary release rule")
    if document["fragment_cache_unit"] != "tensor":
        raise Unsupported("the fragment cache explicitly budgets individual materialized tensors")
    return {**document, "transfer_engines": dict(engines)}


class UnifiedPolicy(DeclarativePolicy):
    def __init__(self, policy, *, transport, history=()):
        super().__init__(policy, history=history)
        self.transport = parse_transport(transport)
        self._services_enabled = False
        self._transport_records = []

    def _placement(self, operation):
        if operation.kind == "state_update" and operation.execution_stage == "trace_supplied_next_token":
            if len(operation.prices) != 1:
                raise Unsupported("trace input gates require one explicitly bound device")
            return operation.prices[0].device
        return super()._placement(operation)

    def _estimate_observation(self, observation):
        self._services_enabled = observation.services is not None
        tensors = []
        for tensor in observation.tensors:
            engine = self.transport["transfer_engines"][tensor.role]
            if self._services_enabled and engine is None:
                raise InvalidInput("services-enabled policies must explicitly bind every role to ce or sm")
            if not self._services_enabled and engine is not None:
                raise Unsupported("engine bindings require a services-enabled model")
            prices = []
            for price in tensor.transfers:
                if price.engine_durations_ns:
                    duration = dict(price.engine_durations_ns).get(engine)
                    if duration is None:
                        raise Unsupported(f"no isolated {engine} payload estimate for {tensor.id}/{price.route}")
                    prices.append(replace(price, duration_ns=duration))
                else:
                    if price.duration_ns is None:
                        raise Unsupported("transfer has neither an exact duration nor a declared engine estimate")
                    prices.append(price)
            tensors.append(replace(tensor, transfers=tuple(prices)))
        return replace(observation, tensors=tuple(tensors))

    def _record(self, observation, decision, reason, **evidence):
        if self._services_enabled:
            evidence["transfer_estimate_scope"] = "isolated_payload_service_subject_to_runtime_contention"
            if "public_duration_ns" in evidence and decision.kind == ACTION_COPY:
                evidence["isolated_engine_service_estimate_ns"] = evidence.pop("public_duration_ns")
        return super()._record(observation, decision, reason, **evidence)

    def _load_step(self, tensor, destination):
        result = super()._load_step(tensor, destination)
        if result is None or result[0].kind != ACTION_COPY or not self._services_enabled:
            return result
        decision, price, _ = result
        engine = self.transport["transfer_engines"][tensor.role]
        return replace(decision, transfer_engine=engine), price, f"pcie_{engine}"

    @staticmethod
    def _resource_available(observation, resource, compute=False):
        from llm_infer_model.m2.causal import compute_service_available
        if resource in ("cpu_compute", "gpu_compute") and not compute_service_available(
                observation, "cpu" if resource == "cpu_compute" else "gpu"):
            return False
        busy = {entry.resource for entry in observation.running}
        if resource in busy:
            return False
        if (resource == "gpu_compute" and "pcie_sm" in busy) or (resource == "pcie_sm" and "gpu_compute" in busy):
            return False
        computing = bool(busy & {"cpu_compute", "gpu_compute"})
        moving = bool(busy & {"pcie_copy", "pcie_ce", "pcie_sm", "ssd_io"})
        if not observation.allow_copy_compute_overlap and ((compute and moving) or (not compute and computing)):
            return False
        if compute and not observation.allow_cpu_gpu_overlap and computing:
            return False
        return True

    @staticmethod
    def _fragment_catalogue(observation):
        experts = tuple(replace(expert, weight_tensor_id=tid, weight_tensor_ids=(tid,))
                        for expert in observation.experts
                        for tid in (expert.weight_tensor_ids or ((expert.weight_tensor_id,) if expert.weight_tensor_id else ())))
        return replace(observation, experts=experts)

    def observe_final(self, observation):
        super().observe_final(self._fragment_catalogue(observation))

    def _eviction(self, observation, tensors, location, protected):
        return super()._eviction(self._fragment_catalogue(observation), tensors, location, protected)

    def _prefetch(self, observation, tensors):
        if not self.policy.prefetch_enabled:
            return None
        frequency = self._frequency()
        catalogue = {(expert.layer_id, expert.expert_id): expert.weight_tensor_ids or
                     ((expert.weight_tensor_id,) if expert.weight_tensor_id else ()) for expert in observation.experts}
        all_weights = {tid for tids in catalogue.values() for tid in tids}
        # Cache budgets charge materialized payload even when a fixed slot has
        # already reserved the physical capacity (incremental allocation = 0).
        resident = sum(tensors[tid].storage_bytes for tid in all_weights
                       if tensors[tid].status_at("vram") != COPY_ABSENT)
        for key in sorted((key for key in catalogue if frequency[key]), key=lambda key: (-frequency[key], str(key))):
            for tid in catalogue[key]:
                tensor = tensors[tid]
                if tensor.status_at("vram") == COPY_READY:
                    continue
                if tid not in self._predicted and len(self._predicted) >= self.policy.prefetch_max_count:
                    continue
                loading = self._load_step(tensor, "vram")
                if loading is None:
                    continue
                decision, price, resource = loading
                if (tid, price.route) in self._prefetch_hops or not self._resource_available(observation, resource):
                    continue
                protected = self._protected(observation, tensors, (tid,))
                for op in observation.operations:
                    if op.status == "NOT_STARTED":
                        protected.update(op.inputs)
                prospective = resident + (tensor.storage_bytes if tensor.status_at("vram") == COPY_ABSENT else 0)
                if prospective > self.policy.cache_budget_bytes:
                    eviction = self._eviction(observation, tensors, "vram", protected)
                    if eviction:
                        return self._record(observation, eviction, "fragment_prefetch_budget_eviction")
                    continue
                fits, eviction = self._fit_or_evict(observation, tensors, price.dst,
                                                    self._allocation(tensor, price.dst), protected)
                if eviction:
                    return self._record(observation, eviction, "fragment_prefetch_capacity_eviction")
                if not fits:
                    continue
                self._predicted.add(tid)
                self._prefetch_hops.add((tid, price.route))
                self._last_used[tid] = self._clock
                return self._record(observation, decision, "history_fragment_prefetch", expert=key, tensor=tid,
                                    historical_activations=frequency[key], cache_unit="tensor",
                                    public_duration_ns=price.duration_ns)
        return None

    def _declared_release(self, observation, tensors):
        if self.transport["boundary_release"] == "disabled" or not observation.allow_eviction:
            return None
        protected = self._protected(observation, tensors)
        for tid, location in getattr(observation, "release_candidates", ()):
            tensor = tensors.get(tid)
            if tensor is None or tid in protected or tensor.goal_location is not None:
                continue
            if tensor.status_at(location) == COPY_READY:
                return Decision(ACTION_EVICT, tid, location=location)
        return None

    def _slot_guard(self, observation, tensors, decision):
        if observation.services is None or decision is None:
            return decision
        targets = []
        if decision.kind == ACTION_COPY:
            targets.append((decision.target_id, decision.dst))
        elif decision.kind == ACTION_COMPUTE:
            op = next(op for op in observation.operations if op.id == decision.target_id)
            targets.extend((tid, "dram" if decision.device == "cpu" else "vram") for tid in op.outputs)
        elif decision.kind == "READ_SSD":
            targets.append((decision.target_id, "dram"))
        protected = self._protected(observation, tensors)
        if decision.kind == ACTION_COMPUTE:
            protected.update(op.inputs)
        for tid, location in targets:
            for slot in observation.services.slots:
                if slot.domain != location or tid not in slot.members or slot.owner in (None, tid):
                    continue
                owner = tensors.get(slot.owner)
                dead = (slot.owner, location) in getattr(observation, "release_candidates", ())
                backed = owner is not None and any(other != location and status == COPY_READY for other, status in owner.copies)
                releasable = (owner is not None and owner.status_at(location) == COPY_READY and
                              slot.owner not in protected and owner.goal_location != location and
                              all(status in (COPY_ABSENT, COPY_READY) for _, status in owner.copies))
                if (self.transport["slot_reuse"] == "release_unlocked_backed_copy" and
                        observation.allow_eviction and releasable and (backed or dead)):
                    self._transport_records.append({"t_ns": observation.t_ns, "reason": "explicit_slot_owner_release",
                                                    "slot": slot.id, "owner": slot.owner, "next_tensor": tid,
                                                    "recoverable_backup": backed, "declared_dead": dead})
                    return Decision(ACTION_EVICT, slot.owner, location=location)
                self._transport_records.append({"t_ns": observation.t_ns, "reason": "wait_for_locked_or_live_slot",
                                                "slot": slot.id, "owner": slot.owner, "next_tensor": tid})
                return Decision(ACTION_ADVANCE)
        return decision

    def __call__(self, observation):
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        release = self._declared_release(observation, tensors)
        if release is not None:
            self._transport_records.append({"t_ns": observation.t_ns, "reason": "declared_closed_scope_release",
                                            "tensor_id": release.target_id, "location": release.location})
            return release
        estimated = self._estimate_observation(observation)
        before = deepcopy(self.__dict__)
        decision = super().__call__(estimated)
        guarded = self._slot_guard(observation, tensors, decision)
        if guarded != decision:
            # A replaced proposal was never submitted. In particular, do not
            # consume its prefetch hop or claim a CPU/GPU queue promotion.
            transport_records = self._transport_records
            self.__dict__.update(before)
            self._transport_records = transport_records
        return guarded

    def audit_document(self):
        result = super().audit_document()
        result.update(transport=self.transport, transport_decisions=list(self._transport_records),
                      fragment_cache_unit="tensor", cache_budget_scope="materialized_payload_bytes",
                      engine_path_estimates_are_contention_free=True)
        return result
