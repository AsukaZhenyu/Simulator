"""Finite block traversal through the common observation/command interface.

The old block document supplies a synthetic graph, traversal and exact GPU
prices. Its generated fixed program is not executed here. The enclosing
composed IR must explicitly replace its weight-release rule with its own cache
and lifetime rules. Traversal gates constrain compute order; transfers may
overlap and prefetch immutable weights across layers.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from llm_infer_model.m2.causal import (CausalObservation, DevicePrice, current_compute_duration_ns,
    PublicOperation, PublicTensor, TransferPrice)
from llm_infer_model.tensor.engine import (Action, ACTION_ADVANCE,
    ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD, ACTION_WRITE_SSD)
from llm_infer_model.tensor.runtime import ExecutionCommand, create_runtime, dispatch_command
from llm_infer_model.tensor.services import ServiceSession
from llm_infer_model.tensor.spec import (COPY_READY, InvalidInput, OP_DONE,
    ROLE_WEIGHT, ROUTE_ENDS, Unsupported, build_scenario)

from .block_strategy import compile_blocks
from .strategy_compiler import _read_object


@dataclass(frozen=True)
class BlockTask:
    request_id: str
    kind: str
    operation_id: str
    layer_id: str
    invocation_id: str
    generation: int
    traversal_index: int


@dataclass(frozen=True)
class CompiledBlockModel:
    document: dict
    scenario: object
    workload_document: dict
    scenario_document: dict
    tasks: tuple[BlockTask, ...]
    layer_order: tuple[str, ...]
    sources: tuple[dict, ...]
    operation_metadata: tuple[dict, ...]
    policy_binding: str = "composed_ir_replaces_legacy_fixed_program_and_weight_release"


@dataclass(frozen=True)
class BlockObservation(CausalObservation):
    layer_order: tuple[str, ...] = ()
    active_invocations: tuple[tuple[str, str], ...] = ()
    weight_catalogue: tuple[tuple[str, str, str, int | str | None], ...] = ()
    release_candidates: tuple[tuple[str, str], ...] = ()
    recoverable_release_candidates: tuple[tuple[str, str], ...] = ()
    completed_invocations: tuple[str, ...] = ()


def verify_block_sources(model):
    for source in model.sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise InvalidInput(f"block source changed after compile: {source['path']}")


def load_block_model_file(path) -> CompiledBlockModel:
    path = Path(path).resolve()
    data = path.read_bytes()
    source = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    raw = _read_object(path, data)
    legacy = compile_blocks(raw, sources=(source,))
    # Use the validated graph only. No old offline program or decisions enter
    # the common runtime; the new IR's policy sees the same tensor state.
    workload = deepcopy(legacy.workload)
    scenario_document = deepcopy(legacy.scenario)
    scenario = build_scenario(workload, scenario_document, where="composition_block")
    tasks = []
    layers = raw["layers"]
    batches = raw["microbatches"]
    for generation in range(raw["generations"]):
        order = (((layer, batch) for layer in layers for batch in batches)
                 if raw["order"] == "layer_major" else
                 ((layer, batch) for batch in batches for layer in layers))
        for layer, batch in order:
            oid = f"op:g{generation}:{layer['id']}:{batch['id']}"
            tasks.append(BlockTask(batch["id"], "dense_ffn", oid, layer["id"],
                                   oid, generation, len(tasks)))
    result = CompiledBlockModel(deepcopy(raw), scenario, workload, scenario_document,
        tuple(tasks), tuple(layer["id"] for layer in layers), (source,),
        tuple(asdict(task) for task in tasks))
    verify_block_sources(result)
    return result


class _BlockVisibility:
    """Traversal order is model input; weight prediction has no oracle data."""
    def __init__(self, model):
        self._model = model
        self._tensors = tuple(sorted(model.scenario.workload.tensor_by_id))

    def current_index(self, state):
        return next((index for index, task in enumerate(self._model.tasks)
                     if state.op(task.operation_id) != OP_DONE), len(self._model.tasks))

    def visible_operation_ids(self, state):
        count = min(len(self._model.tasks), self.current_index(state) + 1)
        return tuple(task.operation_id for task in self._model.tasks[:count])

    def visible_tensor_ids(self, state):
        # This is an entirely static finite graph, without future expert routes
        # or dynamic requests. All identities/shapes are public model inputs.
        return self._tensors

    def allow_action(self, action, state):
        if action.kind == ACTION_ADVANCE:
            return True
        if action.kind == ACTION_COMPUTE:
            return action.target_id in self.visible_operation_ids(state)
        if action.kind in (ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD, ACTION_EVICT):
            return action.target_id in self._tensors
        return False


class CausalBlockSession:
    """One runtime owns all finite blocks, versions, resources and time."""
    def __init__(self, model: CompiledBlockModel, *, services=None, environment_events=(),
                 reclaim_requests=(), control_costs=None):
        if not isinstance(model, CompiledBlockModel):
            raise InvalidInput("CausalBlockSession requires CompiledBlockModel")
        verify_block_sources(model)
        self.model = model
        self._visibility = _BlockVisibility(model)
        self._runtime = create_runtime(model.scenario, services=services,
            environment_events=environment_events, visibility=self._visibility,
            reclaim_requests=reclaim_requests, control_costs=control_costs)
        self._task_by_id = {task.operation_id: task for task in model.tasks}
        self._consumers = {tensor.id: tuple(operation.id for operation in model.scenario.workload.operations
                                          if tensor.id in operation.inputs)
                           for tensor in model.scenario.workload.tensors}

    def observe(self):
        base = self._runtime.observe()
        scenario, state = self._runtime.scenario, self._runtime.state
        operations = []
        for oid in self._visibility.visible_operation_ids(state):
            task = self._task_by_id[oid]
            op = scenario.workload.operation_by_id[oid]
            prices = tuple(DevicePrice(device, cost.duration_ns, cost.workspace_bytes,
                                       scenario.workspace_alloc(oid, device),
                                       current_compute_duration_ns(cost.duration_ns, base, device))
                for device in scenario.mapspace.devices_for(oid)
                for cost in (scenario.costs.compute_cost(oid, device),) if cost is not None)
            operations.append(PublicOperation(id=oid, kind=task.kind, status=state.op(oid),
                inputs=op.inputs, outputs=op.outputs, prices=prices, request_id=task.request_id,
                phase="synthetic_generation", region_id=oid, execution_stage="synthetic_layer_block",
                execution_variant="finite_block_traversal", layer_id=task.layer_id,
                invocation_id=task.invocation_id))
        tensors = []
        visible = self._visibility.visible_tensor_ids(state)
        for tid in visible:
            tensor = scenario.workload.tensor_by_id[tid]
            transfers = []
            for route, (src, dst) in ROUTE_ENDS.items():
                if not scenario.mapspace.allows(route, tid):
                    continue
                price = scenario.costs.transfer_cost(route, tid)
                if price is None:
                    raise Unsupported(f"missing exact block transfer cost {tid}/{route}")
                if isinstance(self._runtime, ServiceSession) and route in ("dram_to_vram", "vram_to_dram"):
                    transfers.append(TransferPrice(route, src, dst, None,
                        tuple((engine, self._runtime.isolated_transfer_duration(tid, engine=engine))
                              for engine in ("ce", "sm")),
                        "isolated_payload_service_subject_to_contention"))
                else:
                    transfers.append(TransferPrice(route, src, dst, price.duration_ns))
            tensors.append(PublicTensor(tid, tensor.role, tensor.storage_bytes,
                tuple((location, state.copy_at(tid, location)) for location in ("dram", "vram", "ssd")),
                tuple(transfers), allocation_bytes=tuple((domain, scenario.size_alloc(tid, domain))
                    for domain in ("dram", "vram")), initial_locations=tensor.initial_locations,
                goal_location=scenario.goal.output_locations.get(tid)))
        closed = tuple(task.operation_id for task in self.model.tasks if state.op(task.operation_id) == OP_DONE)
        index = self._visibility.current_index(state)
        active = self.model.tasks[index] if index < len(self.model.tasks) else None
        active_inputs = set(scenario.workload.operation_by_id[active.operation_id].inputs) if active else set()
        candidates = []
        recoverable = []
        for tensor in tensors:
            # KV goals are retained at their declared home. A backed VRAM copy
            # is a replaceable cache copy, not a declaration that KV is dead.
            # Keeping DRAM staging copies after SSD write matches the original
            # template's documented limitation.
            home = tensor.goal_location or self.model.document["kv_home"]
            if (tensor.id.startswith("kv:") and home in ("dram", "ssd")
                    and tensor.id not in active_inputs
                    and tensor.status_at(home) == COPY_READY
                    and tensor.status_at("vram") == COPY_READY):
                recoverable.append((tensor.id, "vram"))
            if tensor.role == ROLE_WEIGHT or tensor.goal_location is not None:
                continue
            consumers = self._consumers[tensor.id]
            if not consumers or any(state.op(oid) != OP_DONE for oid in consumers):
                continue
            # A finite static traversal proves the final use. The policy still
            # decides explicit release and the runtime verifies live locks.
            candidates.extend((tensor.id, location) for location in ("dram", "vram")
                if tensor.status_at(location) == COPY_READY and location not in tensor.initial_locations)
        return BlockObservation(t_ns=base.t_ns, operations=tuple(operations), tensors=tuple(tensors),
            running=base.running, memory=base.memory, cpu_available=base.cpu_available, route_history=(),
            deliveries=tuple(sorted(scenario.goal.output_locations.items())), complete=base.finished,
            error_code=base.error_code, allow_copy_compute_overlap=scenario.mapspace.allow_copy_compute_overlap,
            allow_cpu_gpu_overlap=scenario.mapspace.allow_cpu_gpu_overlap,
            allow_eviction=scenario.mapspace.allow_eviction, task_graph_complete=index >= len(self.model.tasks)-1,
            services=self._runtime.public_services(visible) if isinstance(self._runtime, ServiceSession) else None,
            pending_reclaim=base.pending_reclaim,
            safe_weight_release_candidates=tuple(tid for tid in base.safe_weight_release_candidates if tid in visible),
            reclaim_records=base.reclaim_records, control_running=base.control_running, control_costs=base.control_costs,
            **{f"{device}_service_rate_{part}": getattr(base, f"{device}_service_rate_{part}")
               for device in ("cpu", "gpu") for part in ("numerator", "denominator")},
            layer_order=self.model.layer_order,
            active_invocations=((active.invocation_id, active.layer_id),) if active else (),
            weight_catalogue=tuple(("w:" + layer, layer, "dense_ffn", None) for layer in self.model.layer_order),
            release_candidates=tuple(sorted(candidates)),
            recoverable_release_candidates=tuple(sorted(recoverable)), completed_invocations=closed)

    def submit(self, action: Action, *, source_index=None):
        return self._runtime.submit(action, source_index=source_index)

    def submit_transfer(self, action: Action, *, engine, source_index=None):
        if not isinstance(self._runtime, ServiceSession):
            raise Unsupported("transfer engine requires services-enabled block session")
        return self._runtime.submit_transfer(action, engine=engine, source_index=source_index)

    def submit_command(self, command: ExecutionCommand, *, source_index=None):
        return dispatch_command(self._runtime, command, source_index=source_index)

    def submit_control(self, command, *, source_index=None):
        return self._runtime.submit_control(command, source_index=source_index)

    def advance(self, *, source_index=None):
        return self._runtime.advance(source_index=source_index)

    def result(self):
        return self._runtime.result()

    def service_document(self):
        if not isinstance(self._runtime, ServiceSession):
            raise Unsupported("no service contract supplied to block session")
        return self._runtime.service_document()

    def persistent_state_document(self):
        state = self._runtime.state
        return dict(format_version="block-persistent-state-0.1", t_ns=state.t,
            status=self.result().evaluation.status,
            state_scope="finite_block_inventory_and_in_flight_services",
            kv_identity_contract="immutable_absolute_position_tensor_ids_no_in_place_mutation",
            tensor_inventory=[dict(tensor_id=t.id, role=t.role, storage_bytes=t.storage_bytes,
                copies={location:state.copy_at(t.id, location) for location in ("dram", "vram", "ssd")})
                for t in self._runtime.scenario.workload.tensors],
            running=[asdict(task) for task in state.running],
            compute_progress=[asdict(row) for row in self._runtime.compute_progress_document()],
            completed_blocks=[task.operation_id for task in self.model.tasks if state.op(task.operation_id) == OP_DONE],
            resumable_checkpoint=False, scope="runtime_inventory_not_a_checkpoint")

    def execution_document(self):
        return deepcopy(dict(workload=self.model.workload_document, scenario=self.model.scenario_document,
            tasks=[asdict(task) for task in self.model.tasks], execution_variant="finite_block_traversal",
            expert_variant="not_applicable", execution_spec=None,
            boundary=dict(block_execution=dict(order=self.model.document["order"],
                layer_order=list(self.model.layer_order), generations=self.model.document["generations"],
                policy_binding=self.model.policy_binding,
                legacy_weight_release=self.model.document["weight_release"],
                compute_traversal_is_enforced=True, legacy_fixed_program_executed=False,
                stage_homes="initial_and_goal_locations_only_intermediate_swaps_are_policy_decisions",
                provenance="synthetic_block_template", real_llm_frontend=False,
                ssd_capacity_modeled=False, dram_copy_retained_after_ssd_write=True,
                numeric_sampling=False, numeric_embedding=False, global_performance_calibrated=False))))


def create_session(model, *, services=None, environment=None, environment_events=(), reclaim_requests=(), control_costs=None):
    if environment is not None:
        if environment_events:
            raise InvalidInput("provide environment or environment_events, not both")
        environment_events = environment
    return CausalBlockSession(model, services=services, environment_events=environment_events,
                             reclaim_requests=reclaim_requests, control_costs=control_costs)
