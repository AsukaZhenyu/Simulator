"""Dense and routed-layer adapters for the model-neutral tensor scheduler.

Legacy Dense state documents carry KV, not weight residency. A separate,
versioned boundary document therefore carries every immutable weight's actual
READY locations. Its inventory and layer hashes bind those copies to the exact
same layer; it is never a replacement for the runner's KV state_out.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation, compile_invocation
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import StateIn, state_in_from_state_out, to_state_in_document, weight_id
from llm_infer_model.tensor.engine import ACTION_COMPUTE, State
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported, build_scenario, scenario_fingerprint

from .schedule import ScheduleFacts, SchedulePolicy, make_schedule_context, replay_schedule_plan

_BOUNDARY_VERSION = "m2-reusable-boundary-0.1"
_LEDGER_ORDER = ("dram", "vram", "ssd")
_LAYER_VERSIONS = {"m2-layer-0.3", "m2-layer-0.4", "m2-layer-0.5", "m2-layer-0.6"}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def compile_schedule_invocation(template: CompiledInvocation, state_in: StateIn) -> CompiledInvocation:
    """Legacy rebind: retain only the two explicit engine concurrency switches."""
    compiled = compile_invocation(template.layer, template.invocation, template.cost_table,
                                  state_in=state_in, where="schedule_invocation")
    switches = {name: getattr(template.scenario.mapspace, name) for name in (
        "allow_copy_compute_overlap", "allow_cpu_gpu_overlap")}
    if any(not isinstance(value, bool) for value in switches.values()):
        raise InvalidInput("engine overlap switches must be explicit booleans")
    if all(getattr(compiled.scenario.mapspace, name) == value for name, value in switches.items()):
        return compiled
    document = {**compiled.scenario_document,
                "mapspace": {**compiled.scenario_document["mapspace"], **switches}}
    scenario = build_scenario(compiled.workload_document, document, where="schedule_invocation")
    return replace(compiled, scenario=scenario, scenario_document=document)


@dataclass(frozen=True)
class M2BoundInvocation:
    compiled: CompiledInvocation
    boundary_state_in: Mapping

    @property
    def scenario(self):
        return self.compiled.scenario

    @property
    def layer(self):
        return self.compiled.layer

    @property
    def compute_ids(self):
        return self.compiled.compute_ids


class M2ScheduleAdapter:
    """Workload facts and actual KV/weight boundaries, without search semantics.

    ``all_weights`` is the new reusable-tensor abstraction. ``experts`` is an
    explicit compatibility scope for the historical routed-layer formats.
    Dense physical cost tables retain their original provenance and declared
    CPU/GPU availability; this adapter never supplies missing device prices.
    The first call preserves its supplied scenario. Later calls are recompiled
    from the original layer/invocation/cost-table metadata and two explicit
    overlap flags. Other programmatic scenario overrides are refused at a
    boundary, including overrides written into the accompanying documents.
    The historical ``experts`` scope also preserves its existing convention
    that a future invocation's hardware metadata replaces an older compiled
    architecture. Modern ``all_weights`` templates require that explicit
    hardware updates be recompiled before use.
    """

    def __init__(self, reusable_scope: str = "all_weights"):
        if reusable_scope not in {"all_weights", "experts"}:
            raise InvalidInput("reusable_scope must be all_weights or experts")
        self.reusable_scope = reusable_scope

    def validate_templates(self, templates: Sequence[CompiledInvocation]):
        result = tuple(templates)
        if not result or any(not isinstance(item, CompiledInvocation) for item in result):
            raise InvalidInput("M2 schedule requires a nonempty sequence of CompiledInvocation templates")
        for item in result:
            if item.layer.format_version not in _LAYER_VERSIONS:
                raise Unsupported("M2 scheduling supports Dense layer .3/.4/.5 and routed layer .6")
            if self.reusable_scope == "experts" and (item.layer.format_version != "m2-layer-0.6" or item.layer.ffn != "moe"):
                raise Unsupported("expert compatibility scope requires synthetic m2-layer-0.6 MoE")
        if any(item.layer != result[0].layer for item in result):
            raise InvalidInput("schedule sequence must preserve the exact same immutable layer definition")
        if len({item.invocation.id for item in result}) != len(result):
            raise InvalidInput("schedule invocation IDs must be unique within an episode")
        return result

    def _inventory(self, compiled, sources=None):
        versions = {entry.id: entry.version for entry in compiled.state_in.weight_copies}
        source_map = {} if sources is None else dict(sources)
        result = []
        for weight in compiled.layer.weights:
            tensor = weight_id(compiled.layer.id, weight.id)
            graph_tensor = compiled.scenario.workload.tensor_by_id[tensor]
            if tensor not in source_map:
                if compiled.layer.format_version == "m2-layer-0.6":
                    source_map[tensor] = compiled.scenario.goal_location(tensor)
                else:
                    source_map[tensor] = weight.initial_location
            source = source_map[tensor]
            if source not in graph_tensor.initial_locations:
                raise InvalidInput(f"weight {weight.id!r} has no READY authoritative source {source!r}")
            result.append({"id": weight.id, "tensor_id": tensor,
                           "storage_bytes": graph_tensor.storage_bytes,
                           "version": versions.get(weight.id),
                           "version_semantics": "explicit_snapshot" if weight.id in versions else "immutable_layer_definition",
                           "authoritative_source": source})
        return tuple(result)

    def _boundary(self, compiled, *, inventory=None, state=None):
        inventory = self._inventory(compiled) if inventory is None else tuple(inventory)
        weights = []
        for entry in inventory:
            tensor = entry["tensor_id"]
            locations = tuple(domain for domain in _LEDGER_ORDER if (
                state.copy_at(tensor, domain) == COPY_READY if state is not None
                else domain in compiled.scenario.workload.tensor_by_id[tensor].initial_locations))
            if not locations or entry["authoritative_source"] not in locations:
                raise InvalidInput(f"weight {entry['id']!r} lost its authoritative READY source")
            weights.append({**entry, "locations": list(locations)})
        return {"format_version": _BOUNDARY_VERSION, "layer_id": compiled.layer.id,
                "layer_sha256": _sha(asdict(compiled.layer)),
                "inventory_sha256": _sha(inventory), "weights": weights}

    def _validate_boundary(self, compiled, boundary, *, expected_inventory=None):
        if not isinstance(boundary, Mapping) or set(boundary) != {
                "format_version", "layer_id", "layer_sha256", "inventory_sha256", "weights"}:
            raise InvalidInput("reusable boundary requires an explicit complete versioned document")
        if boundary["format_version"] != _BOUNDARY_VERSION or boundary["layer_id"] != compiled.layer.id or boundary["layer_sha256"] != _sha(asdict(compiled.layer)):
            raise InvalidInput("reusable boundary belongs to another immutable layer definition")
        entries = boundary["weights"]
        if not isinstance(entries, (list, tuple)):
            raise InvalidInput("boundary weights must be an explicit list")
        expected_ids = {weight.id for weight in compiled.layer.weights}
        if len(entries) != len(expected_ids) or {entry.get("id") for entry in entries if isinstance(entry, Mapping)} != expected_ids:
            raise InvalidInput("reusable boundary must retain exactly the entire weight inventory")
        inventory = []
        by_id = {weight.id: weight for weight in compiled.layer.weights}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"id", "tensor_id", "storage_bytes", "version", "version_semantics", "authoritative_source", "locations"}:
                raise InvalidInput("reusable boundary weight fields are incomplete or unknown")
            weight = by_id[entry["id"]]
            tensor = weight_id(compiled.layer.id, weight.id)
            if entry["tensor_id"] != tensor or entry["storage_bytes"] != compiled.scenario.workload.tensor_by_id[tensor].storage_bytes:
                raise InvalidInput("reusable boundary tensor identity/bytes do not match the layer")
            locations = entry["locations"]
            if not isinstance(locations, (tuple, list)) or not locations or len(set(locations)) != len(locations) or any(domain not in _LEDGER_ORDER for domain in locations):
                raise InvalidInput("reusable boundary locations must be nonempty unique READY domains")
            if entry["authoritative_source"] not in locations:
                raise InvalidInput("reusable boundary lost an authoritative source")
            version = entry["version"]
            if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 0):
                raise InvalidInput("reusable boundary explicit versions must be nonnegative integers")
            semantics = "immutable_layer_definition" if version is None else "explicit_snapshot"
            if entry["version_semantics"] != semantics:
                raise InvalidInput("reusable boundary version semantics do not match its explicit value")
            native_versions = {ref.id: ref.version for ref in compiled.state_in.weight_copies}
            if version != native_versions.get(weight.id):
                raise InvalidInput("reusable boundary version disagrees with the immutable/native inventory")
            source = (compiled.scenario.goal_location(tensor)
                      if compiled.layer.format_version == "m2-layer-0.6" else weight.initial_location)
            if entry["authoritative_source"] != source:
                raise InvalidInput("reusable boundary changed its authoritative source")
            inventory.append({key: entry[key] for key in entry if key != "locations"})
        inventory = tuple(inventory)
        if boundary["inventory_sha256"] != _sha(inventory):
            raise InvalidInput("reusable boundary inventory hash mismatch")
        if expected_inventory is not None and _sha(inventory) != _sha(tuple(expected_inventory)):
            raise InvalidInput("reusable boundary changed weight versions, sources or inventory")
        return inventory

    def boundary_from_compiled(self, compiled):
        """Explicit first-call residency, useful for historical seed conversion."""
        return self._boundary(compiled)

    def boundary_from_result(self, compiled, result, *, inventory=None):
        """Read READY copies from the verified final ledger, never a sidecar claim."""
        if result.state_out is None or not result.evaluation.states:
            raise InvalidInput("an incomplete result has no reusable boundary")
        inventory = self._inventory(compiled) if inventory is None else tuple(inventory)
        snapshot = result.evaluation.states[-1]
        boundary = self._boundary(compiled, inventory=inventory)
        for entry in boundary["weights"]:
            entry["locations"] = [domain for domain in _LEDGER_ORDER
                if dict(snapshot.statuses(domain)).get(entry["tensor_id"]) == COPY_READY]
            if not entry["locations"] or entry["authoritative_source"] not in entry["locations"]:
                raise InvalidInput("final replay lost an authoritative READY weight copy")
        return boundary

    def initial(self, template):
        self.validate_templates((template,))
        # The first context uses the actual supplied scenario, including explicit
        # programmatic cost/device/resource restrictions; no recompilation reset.
        return M2BoundInvocation(template, self._boundary(template))

    def bind(self, template, previous_call):
        if previous_call.result is None or previous_call.result.state_out is None:
            raise InvalidInput("an incomplete invocation has no reusable/KV boundary to bind")
        previous = previous_call.compiled
        if isinstance(previous, M2BoundInvocation):
            previous = previous.compiled
        if previous.layer != template.layer:
            raise InvalidInput("cannot bind a reusable boundary across different layers")
        boundary = previous_call.boundary_state_out
        if boundary is None or previous_call.boundary_state_in is None:
            raise InvalidInput("M2 generic schedule requires both explicit reusable boundary documents")
        expected = self._validate_boundary(previous, previous_call.boundary_state_in)
        inventory = self._validate_boundary(previous, boundary, expected_inventory=expected)
        actual_previous = self.boundary_from_result(previous, previous_call.result, inventory=expected)
        if boundary != actual_previous:
            raise InvalidInput("reusable boundary locations disagree with the actual final READY ledger")
        canonical = compile_schedule_invocation(template, template.state_in)
        declared_scope = template.scenario
        if self.reusable_scope == "experts":
            declared_scope = replace(declared_scope, architecture=canonical.scenario.architecture)
        if declared_scope != canonical.scenario:
            raise Unsupported("cross-call binding cannot reproduce programmatic scenario overrides beyond the two explicit overlap flags; declare restrictions in the M2 layer/invocation/cost-table metadata")
        history = state_in_from_state_out(previous_call.result.state_out)
        compiled = compile_schedule_invocation(template, history)
        document = deepcopy(compiled.workload_document)
        by_tensor = {entry["tensor_id"]: entry for entry in boundary["weights"]}
        for tensor in document["tensors"]:
            if tensor["id"] in by_tensor:
                tensor["initial_locations"] = list(by_tensor[tensor["id"]]["locations"])
        scenario = build_scenario(document, compiled.scenario_document, where="schedule_boundary")
        compiled = replace(compiled, workload_document=document, scenario=scenario)
        # MoE has a native weight-state contract too: both representations must
        # identify exactly the same actual copies and versions.
        if compiled.layer.format_version == "m2-layer-0.6":
            native = {entry.id: entry for entry in history.weight_copies}
            for entry in boundary["weights"]:
                if entry["id"] not in native or native[entry["id"]].version != entry["version"] or set(native[entry["id"]].locations) != set(entry["locations"]):
                    raise InvalidInput("native weight state disagrees with reusable boundary")
        actual = self._boundary(compiled, inventory=inventory)
        if actual != boundary:
            raise InvalidInput("compiled reusable state did not exactly preserve the boundary")
        return M2BoundInvocation(compiled, actual)

    def make_context(self, bound, policy: SchedulePolicy, placements=None):
        if isinstance(bound, CompiledInvocation):
            bound = self.initial(bound)
        compiled = bound.compiled
        inventory = self._validate_boundary(compiled, bound.boundary_state_in)
        if bound.boundary_state_in != self._boundary(compiled, inventory=inventory):
            raise InvalidInput("reusable input boundary locations disagree with the actual initial READY ledger")
        ids = frozenset(entry["tensor_id"] for entry in inventory)
        if self.reusable_scope == "experts":
            # Classification is an adapter-only compatibility declaration.
            from .m2_cache import expert_weight_ids
            ids = expert_weight_ids(compiled)
        sources = tuple((entry["tensor_id"], entry["authoritative_source"]) for entry in inventory if entry["tensor_id"] in ids)
        facts = ScheduleFacts(compiled.compute_ids, ids, sources,
                              frozenset(compiled.scenario.workload.tensor_by_id) - ids,
                              compiled.layer.id)
        return make_schedule_context(compiled.scenario, facts, policy, placements, payload=bound)

    def identity(self, bound):
        inventory = self._validate_boundary(bound.compiled, bound.boundary_state_in)
        if bound.boundary_state_in != self._boundary(bound.compiled, inventory=inventory):
            raise InvalidInput("reusable input boundary differs from the actual initial ledger")
        return (scenario_fingerprint(bound.scenario),
                _canonical(to_state_in_document(bound.compiled.state_in, layer=bound.layer.id)),
                _canonical(bound.boundary_state_in))

    def finish_call(self, context, actions, state: State, *, require_complete: bool = True):
        from .schedule_search import ScheduleInvocationResult, verify_replayed_state
        plan = replay_schedule_plan(context, actions, require_complete=require_complete)
        bound = context.payload
        result = run_invocation(bound.compiled, actions=actions)
        verify_replayed_state(result.evaluation, state)
        placements = tuple((action.target_id, action.device) for action in actions if action.kind == ACTION_COMPUTE)
        boundary_out = None
        if result.state_out is not None:
            inventory = self._validate_boundary(bound.compiled, bound.boundary_state_in)
            boundary_out = self.boundary_from_result(bound.compiled, result, inventory=inventory)
        return ScheduleInvocationResult(bound.compiled, placements, plan, result,
            boundary_state_in=bound.boundary_state_in, boundary_state_out=boundary_out)

    def validate_seed_call(self, bound, seed_call):
        compiled = seed_call.compiled
        if isinstance(compiled, M2BoundInvocation):
            compiled = compiled.compiled
        if (compiled.layer != bound.layer or scenario_fingerprint(compiled.scenario) != scenario_fingerprint(bound.scenario)
                or compiled.invocation != bound.compiled.invocation
                or compiled.state_in != bound.compiled.state_in or seed_call.boundary_state_in != bound.boundary_state_in):
            raise InvalidInput("seed invocation does not match the actual scoped M2 boundary/scenario")
        if seed_call.boundary_state_out is not None:
            inventory = self._validate_boundary(bound.compiled, bound.boundary_state_in)
            if seed_call.result is None or seed_call.boundary_state_out != self.boundary_from_result(bound.compiled, seed_call.result, inventory=inventory):
                raise InvalidInput("seed reusable boundary does not match its exact replay")
