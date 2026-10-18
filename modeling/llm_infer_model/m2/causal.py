"""Model-owned MoE revelations for a genuinely causal policy boundary.

The replay graph and route trace remain private to this adapter. A policy sees
immutable values only. Expert/combine graph details are withheld until all
routers finish, because the existing M2 graph batches experts across requests.
This conservative release rule is not token-streaming MoE execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from ..tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT,
    ACTION_READ_SSD, ACTION_WRITE_SSD, Action, State,
)
from ..tensor.spec import (
    COPY_ABSENT, COPY_READY, OP_DONE, ROLE_WEIGHT, ROUTE_ENDS,
    InvalidInput, Unsupported, build_scenario,
)
from .costs import expand_transfer_costs
from .expand import CompiledInvocation, load_and_compile
from .spec import op_id, weight_id
from ..tensor.runtime import ExecutionCommand, create_runtime, dispatch_command
from ..tensor.services import PublicServices, ServiceContract, ServiceSession


@dataclass(frozen=True)
class DevicePrice:
    device: str
    duration_ns: int
    workspace_bytes: int
    allocated_workspace_bytes: int | None = None
    # Reference duration above never changes. This separate estimate assumes
    # the currently observed rate remains constant for the whole operation.
    # Model-owned observations use None when the effective service rate is 0.
    current_duration_ns: int | None = None


def compute_service_available(observation, device: str) -> bool:
    """Current public availability only; no prediction about future recovery."""
    if device not in ("cpu", "gpu"):
        raise InvalidInput("compute availability requires cpu or gpu")
    return ((device != "cpu" or observation.cpu_available)
            and getattr(observation, f"{device}_service_rate_numerator", 1) > 0)


def current_compute_duration_ns(reference_duration_ns: int, observation, device: str) -> int | None:
    """Whole-operation estimate at the current explicit reference-service rate."""
    if not compute_service_available(observation, device):
        return None
    numerator = getattr(observation, f"{device}_service_rate_numerator", 1)
    denominator = getattr(observation, f"{device}_service_rate_denominator", 1)
    return (reference_duration_ns * denominator + numerator - 1) // numerator


@dataclass(frozen=True)
class TransferPrice:
    route: str
    src: str
    dst: str
    duration_ns: int | None
    engine_durations_ns: tuple[tuple[str, int], ...] = ()
    timing_scope: str = "exact_isolated_duration"


@dataclass(frozen=True)
class PublicOperation:
    id: str
    kind: str
    status: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    prices: tuple[DevicePrice, ...]
    expert_id: int | str | None = None
    work_items: tuple[tuple[str, int], ...] = ()
    token_count: int = 0
    phase: str = ""
    request_id: str = ""
    region_id: str = ""
    execution_stage: str = ""
    execution_variant: str = ""
    layer_id: str = ""
    invocation_id: str = ""


@dataclass(frozen=True)
class PublicTensor:
    id: str
    role: str
    storage_bytes: int
    copies: tuple[tuple[str, str], ...]
    transfers: tuple[TransferPrice, ...]
    row_scope: tuple[tuple[str, int], ...] = ()
    allocation_bytes: tuple[tuple[str, int], ...] = ()
    initial_locations: tuple[str, ...] = ()
    goal_location: str | None = None

    def status_at(self, location: str) -> str:
        return dict(self.copies).get(location, COPY_ABSENT)


@dataclass(frozen=True)
class RevealedRoute:
    request_id: str
    layer_id: str
    absolute_position: int
    expert_ids: tuple[int, ...]
    scores: tuple[float, ...] | None
    router_scores: tuple[tuple[int,float], ...] = ()
    router_scores_scope: str = "not_reported"


@dataclass(frozen=True)
class PublicExpert:
    expert_id: int | str
    weight_tensor_id: str | None
    layer_id: str = ""
    weight_tensor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CausalObservation:
    t_ns: int
    operations: tuple[PublicOperation, ...]
    tensors: tuple[PublicTensor, ...]
    running: tuple[Any, ...]
    memory: tuple[Any, ...]
    cpu_available: bool
    route_history: tuple[RevealedRoute, ...]
    deliveries: tuple[tuple[str, str], ...]
    complete: bool
    error_code: str | None
    information_mode: str = "causal_observation"
    experts: tuple[PublicExpert, ...] = ()
    allow_copy_compute_overlap: bool = False
    allow_cpu_gpu_overlap: bool = False
    allow_eviction: bool = False
    # This completeness is scoped to this invocation only. It cannot establish
    # liveness across subsequent layers/invocations or unrevealed future routes.
    task_graph_complete: bool = False
    services: PublicServices | None = None
    pending_reclaim: Any | None = None
    safe_weight_release_candidates: tuple[str, ...] = ()
    reclaim_records: tuple[Any, ...] = ()
    control_running: Any | None = None
    control_costs: Any | None = None
    cpu_service_rate_numerator: int = 1
    cpu_service_rate_denominator: int = 1
    gpu_service_rate_numerator: int = 1
    gpu_service_rate_denominator: int = 1


class _MoEVisibility:
    """Privileged model-side projection; never supplied to the policy."""

    def __init__(self, compiled: CompiledInvocation):
        self._compiled = compiled
        self._routers = tuple(task.operation_id for task in compiled.tasks if task.kind == "router")
        self._static_ops = tuple(task.operation_id for task in compiled.tasks
                                 if task.kind in ("projection", "attention", "router", "attention_core", "attention_output",
                                                  "attention_projection", "attention_qk", "attention_softmax", "attention_pv"))
        self._dynamic_ops = tuple(task.operation_id for task in compiled.tasks
                                  if task.operation_id not in self._static_ops)
        tensors = compiled.scenario.workload.tensor_by_id
        public = {tensor.id for tensor in tensors.values() if tensor.initial_locations}
        for operation_id in self._static_ops:
            operation = compiled.scenario.workload.operation_by_id[operation_id]
            public.update(operation.inputs)
            public.update(operation.outputs)
        self._static_tensors = tuple(sorted(public))

    def visible_operation_ids(self, state: State) -> tuple[str, ...]:
        if all(state.op(router) == OP_DONE for router in self._routers):
            return self._static_ops + self._dynamic_ops
        return self._static_ops

    def task_graph_complete(self, state: State) -> bool:
        return all(state.op(router) == OP_DONE for router in self._routers)

    def visible_tensor_ids(self, state: State) -> tuple[str, ...]:
        if all(state.op(router) == OP_DONE for router in self._routers):
            return tuple(sorted(self._compiled.scenario.workload.tensor_by_id))
        return self._static_tensors

    def allow_action(self, action: Action, state: State) -> bool:
        if action.kind == ACTION_ADVANCE:
            return True
        if action.kind == ACTION_COMPUTE:
            return action.target_id in self.visible_operation_ids(state)
        if action.kind in (ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD, ACTION_EVICT):
            return action.target_id in self.visible_tensor_ids(state)
        return False

    def revealed_routes(self, state: State) -> tuple[RevealedRoute, ...]:
        template = next(task for task in self._compiled.layer.tasks if task.kind == "router")
        entries = (entry for entry in self._compiled.invocation.route_trace
                   if state.op(op_id(self._compiled.layer.id, entry.request_id, template.id)) == OP_DONE)
        return tuple(RevealedRoute(entry.request_id, entry.layer_id, entry.absolute_position,
                                   entry.expert_ids, entry.scores)
                     for entry in sorted(entries, key=lambda row: (row.request_id, row.absolute_position)))


class _DenseVisibility:
    """A deterministic Dense invocation has no hidden router-generated tasks.

    Completeness means this declared invocation, never future request arrivals,
    later layers or subsequent decode iterations.
    """
    def __init__(self, compiled):
        self._operations=tuple(task.operation_id for task in compiled.tasks)
        self._tensors=tuple(sorted(compiled.scenario.workload.tensor_by_id))

    def visible_operation_ids(self,state):return self._operations
    def visible_tensor_ids(self,state):return self._tensors
    def task_graph_complete(self,state):return True
    def revealed_routes(self,state):return ()
    def allow_action(self,action,state):
        if action.kind==ACTION_ADVANCE:return True
        if action.kind==ACTION_COMPUTE:return action.target_id in self._operations
        if action.kind in (ACTION_COPY,ACTION_READ_SSD,ACTION_WRITE_SSD,ACTION_EVICT):return action.target_id in self._tensors
        return False


def _causal_scenario(compiled: CompiledInvocation):
    """Restore static catalogue transfers without guessing an inactive cost."""
    if compiled.layer.ffn=="dense" and compiled.layer.format_version=="m2-layer-0.5":
        document=deepcopy(compiled.scenario_document)
        return build_scenario(compiled.workload_document,document,where="causal_dense"),document
    if compiled.layer.format_version != "m2-layer-0.6" or compiled.layer.ffn != "moe":
        raise Unsupported("causal M2 adapter currently requires synthetic MoE m2-layer-0.6")
    if compiled.invocation.information_mode != "causal_replay":
        raise Unsupported("causal M2 adapter refuses offline_oracle invocations")
    document = deepcopy(compiled.scenario_document)
    mapspace = document["mapspace"]
    weights = tuple(t.id for t in compiled.scenario.workload.tensors if t.role == ROLE_WEIGHT)
    for route in ROUTE_ENDS:
        existing = mapspace["allowed_transfers"].get(route, [])
        mapspace["allowed_transfers"][route] = sorted(set(existing) | set(weights))
    # Loading known immutable weights is legal before a route is revealed.
    # Computational dependency/revelation, rather than load prohibition, makes
    # this execution causal. Predictions that miss still incur time and bytes.
    mapspace["transfer_dependencies"] = {}
    _, transfers = expand_transfer_costs(
        compiled.cost_table,
        {t.id: t.storage_bytes for t in compiled.scenario.workload.tensors},
        mapspace["allowed_transfers"],
    )
    document["costs"]["transfer"] = transfers
    return build_scenario(compiled.workload_document, document, where="causal_moe"), document


class CausalM2Session:
    """Modeling's privileged session; ``policy`` receives only ``observe()``.

    Python callbacks are trusted code, not an adversarial security sandbox.
    The causal guarantee is an API/data boundary, verified by prefix tests.
    """

    def __init__(self, compiled: CompiledInvocation, *, environment_events: Iterable[Any] = (),
                 execution_spec=None, services: ServiceContract | None = None,
                 reclaim_requests=(), control_costs=None):
        if execution_spec is not None:
            from .execution_pipeline import ExecutionPipelineSpec, apply_execution_transform
            from .ffn_execution import FFNExecutionSpec
            from .attention_stages import AttentionStagesSpec
            from .attention_execution import DenseAttentionExecutionSpec, attention_execution_document, transform_dense_attention
            if isinstance(execution_spec,(ExecutionPipelineSpec,FFNExecutionSpec,AttentionStagesSpec)):
                compiled=apply_execution_transform(compiled,execution_spec)
            elif isinstance(execution_spec,DenseAttentionExecutionSpec):
                existing=(compiled.physical_boundary or {}).get("attention_execution")
                if existing is None:
                    compiled=transform_dense_attention(compiled,execution_spec)
                elif existing["execution_spec"]!=attention_execution_document(execution_spec):
                    raise InvalidInput("already materialized Dense graph differs from execution_spec")
            else:
                from .row_execution import transform_moe_execution
                compiled = transform_moe_execution(compiled, execution_spec)
        elif ((compiled.physical_boundary or {}).get("attention_execution") and
              not (compiled.physical_boundary or {}).get("execution_pipeline")):
            from .attention_execution import parse_attention_execution
            execution_spec=parse_attention_execution(compiled.physical_boundary["attention_execution"]["execution_spec"])
        self._execution_spec = execution_spec
        self._compiled = compiled
        if compiled.layer.ffn=="dense":
            self._visibility=_DenseVisibility(compiled)
            self._experts=()
        else:
            self._visibility = _MoEVisibility(compiled)
            from .execution_pipeline import materialized_weight_members
            available={tensor.id for tensor in compiled.scenario.workload.tensors if tensor.role==ROLE_WEIGHT}
            def expert_catalog(index, weight):
                parent = weight_id(compiled.layer.id, weight)
                members = materialized_weight_members(compiled.physical_boundary,parent,available)
                if any(member not in compiled.scenario.workload.tensor_by_id for member in members):
                    raise InvalidInput("expert catalogue references an unmaterialized weight tensor")
                return PublicExpert(index, parent if parent in compiled.scenario.workload.tensor_by_id else None,
                                    compiled.layer.id, members)
            self._experts = tuple(expert_catalog(index, weight)
                                  for index, weight in enumerate(compiled.layer.ffn_params["expert_weights"]))
            shared = compiled.layer.ffn_params.get("shared_expert_weight")
            if shared is not None:
                self._experts += (expert_catalog("shared", shared),)
        causal_scenario, self._scenario_document = _causal_scenario(compiled)
        self._runtime = create_runtime(causal_scenario, services=services,
                                       environment_events=tuple(environment_events), visibility=self._visibility,
                                       reclaim_requests=reclaim_requests, control_costs=control_costs)

    @classmethod
    def from_file(cls, path: str | Path, *, environment_events: Iterable[Any] = (), execution_spec=None,
                  services: ServiceContract | None = None, reclaim_requests=(), control_costs=None):
        return cls(load_and_compile(path), environment_events=environment_events,
                   execution_spec=execution_spec, services=services,
                   reclaim_requests=reclaim_requests, control_costs=control_costs)

    def execution_document(self):
        """Privileged driver/debug export; never supply this to a policy."""
        from dataclasses import asdict
        from .execution_pipeline import transform_document
        spec_document=(transform_document(self._execution_spec) if self._execution_spec is not None else
                       (self._compiled.physical_boundary or {}).get("execution_pipeline"))
        return deepcopy(dict(workload=self._compiled.workload_document,
                             scenario=self._scenario_document,
                             tasks=[asdict(task) for task in self._compiled.tasks],
                             boundary=self._compiled.physical_boundary,
                             execution_variant=("execution_pipeline" if spec_document and spec_document.get("format_version")=="execution-pipeline-0.1" else
                                                "whole_request" if self._execution_spec is None else
                                                self._execution_spec.variant),
                             expert_variant=("opaque" if self._execution_spec is None else
                                             self._execution_spec.expert_variant),
                             execution_spec=spec_document))

    def observe(self) -> CausalObservation:
        base = self._runtime.observe()
        state, scenario = self._runtime.state, self._runtime.scenario
        operations = []
        tasks = {task.operation_id: task for task in self._compiled.tasks}
        boundary=self._compiled.physical_boundary or {}
        from .execution_pipeline import execution_metadata
        stage_metadata,row_scopes=execution_metadata(boundary)
        for operation_id in self._visibility.visible_operation_ids(state):
            operation = scenario.workload.operation_by_id[operation_id]
            task = tasks[operation_id]
            prices = tuple(DevicePrice(device, price.duration_ns, price.workspace_bytes,
                                       scenario.workspace_alloc(operation_id, device),
                                       current_compute_duration_ns(price.duration_ns, base, device))
                           for device in scenario.mapspace.devices_for(operation_id)
                           for price in (scenario.costs.compute_cost(operation_id, device),)
                           if price is not None)
            expert = "shared" if task.kind == "shared_expert_ffn" else task.expert_id
            metadata = stage_metadata.get(operation_id, {})
            expert = metadata.get("expert_id", expert)
            operations.append(PublicOperation(operation_id, task.kind,
                                              state.op(operation_id), operation.inputs,
                                              operation.outputs, prices, expert, task.work_items,
                                              task.shape.tokens, task.shape.phase, task.request_id,
                                              metadata.get("region_id", ""), metadata.get("execution_stage", ""),
                                              metadata.get("execution_variant", ""),
                                              self._compiled.layer.id, self._compiled.invocation.id))
        visible = self._visibility.visible_tensor_ids(state)
        tensors = []
        for tensor_id in visible:
            tensor = scenario.workload.tensor_by_id[tensor_id]
            transfers = []
            for route, (src, dst) in ROUTE_ENDS.items():
                if scenario.mapspace.allows(route, tensor_id):
                    price = scenario.costs.transfer_cost(route, tensor_id)
                    if price is None:
                        raise Unsupported(f"public transfer {tensor_id!r}/{route} has no exact cost")
                    if isinstance(self._runtime, ServiceSession) and route in ("dram_to_vram", "vram_to_dram"):
                        transfers.append(TransferPrice(route, src, dst, None,
                            tuple((engine, self._runtime.isolated_transfer_duration(tensor_id, engine=engine))
                                  for engine in ("ce", "sm")),
                            "isolated_payload_service_subject_to_contention"))
                    else:
                        transfers.append(TransferPrice(route, src, dst, price.duration_ns))
            tensors.append(PublicTensor(tensor_id, tensor.role, tensor.storage_bytes,
                                        tuple((location, state.copy_at(tensor_id, location))
                                              for location in ("dram", "vram", "ssd")), tuple(transfers),
                                        tuple((item["request_id"], item["absolute_position"])
                                              for item in row_scopes.get(tensor_id, ())),
                                        tuple((domain, scenario.size_alloc(tensor_id, domain))
                                              for domain in ("dram", "vram")),
                                        tensor.initial_locations, scenario.goal.output_locations.get(tensor_id)))
        # Deliveries are only for currently public produced values. Weight goals
        # and hidden future expert outputs are not passed to a policy.
        deliveries = tuple(sorted((tensor, location)
                                  for tensor, location in scenario.goal.output_locations.items()
                                  if tensor in visible and scenario.workload.tensor_by_id[tensor].role != ROLE_WEIGHT))
        return CausalObservation(base.t_ns, tuple(operations), tuple(tensors),
                                 base.running, base.memory, base.cpu_available,
                                 self._visibility.revealed_routes(state), deliveries,
                                 base.finished, base.error_code, experts=self._experts,
                                 allow_copy_compute_overlap=scenario.mapspace.allow_copy_compute_overlap,
                                 allow_cpu_gpu_overlap=scenario.mapspace.allow_cpu_gpu_overlap,
                                 allow_eviction=scenario.mapspace.allow_eviction,
                                 task_graph_complete=self._visibility.task_graph_complete(state),
                                 services=(self._runtime.public_services(visible)
                                           if isinstance(self._runtime, ServiceSession) else None),
                                 pending_reclaim=base.pending_reclaim,
                                 safe_weight_release_candidates=tuple(tid for tid in base.safe_weight_release_candidates
                                                                      if tid in visible),
                                 reclaim_records=base.reclaim_records, control_running=base.control_running,
                                 control_costs=base.control_costs,
                                 **{f"{device}_service_rate_{part}": getattr(base, f"{device}_service_rate_{part}")
                                    for device in ("cpu", "gpu") for part in ("numerator", "denominator")})

    def submit(self, action: Action, *, source_index: int | None = None):
        return self._runtime.submit(action, source_index=source_index)

    def submit_transfer(self, action: Action, *, engine: str, source_index: int | None = None):
        if not isinstance(self._runtime, ServiceSession):
            raise Unsupported("transfer engine requires a services-enabled causal session")
        return self._runtime.submit_transfer(action, engine=engine, source_index=source_index)

    def submit_command(self, command: ExecutionCommand, *, source_index: int | None = None):
        return dispatch_command(self._runtime, command, source_index=source_index)

    def submit_control(self, command, *, source_index=None):
        return self._runtime.submit_control(command, source_index=source_index)

    def service_document(self):
        if not isinstance(self._runtime, ServiceSession):
            raise Unsupported("no service contract was supplied to this causal session")
        return self._runtime.service_document()

    def advance(self, *, source_index: int | None = None):
        return self._runtime.advance(source_index=source_index)

    def result(self):
        return self._runtime.result()

    def state_out(self):
        from .runner import state_out_from_result
        if ((self._compiled.physical_boundary or {}).get("row_execution", {}).get("subweights") or
                (self._compiled.physical_boundary or {}).get("ffn_execution")):
            raise Unsupported("materialized subweights require persistent_state_document; "
                              "legacy state_out cannot represent independently resident fragments")
        return state_out_from_result(self._compiled, self._runtime.result().evaluation)

    def persistent_state_document(self):
        """Export actual fragment residency without inventing a parent copy.

        The fragment envelope is not a legacy M2 state_in. Its KV/hidden section
        alone cannot resume the layer; a driver must also retain the explicitly
        versioned fragment inventory. Multi-layer drivers own their checkpoint.
        """
        from dataclasses import asdict, replace
        from .runner import state_out_from_result
        from .spec import to_state_document
        from ..tensor.engine import STATUS_VALID
        rows = (self._compiled.physical_boundary or {}).get("row_execution", {}).get("subweights", ())
        partitions=(self._compiled.physical_boundary or {}).get("ffn_execution",{}).get("weight_partitions",())
        evaluation = self._runtime.result().evaluation
        if not rows and not partitions and evaluation.status == STATUS_VALID:
            return to_state_document(self.state_out())
        kv_state = None
        if evaluation.status == STATUS_VALID:
            kv_compiled = replace(self._compiled, state_in=replace(self._compiled.state_in, weight_copies=()))
            kv_state = to_state_document(state_out_from_result(kv_compiled, evaluation))
        fragments = []
        replaced={row["source_tensor_id"] for row in partitions}
        for row in rows:
            for matrix in row["matrices"]:
                tid = matrix["tensor_id"]
                if tid in replaced:continue
                locations = [location for location in ("dram", "vram", "ssd")
                             if self._runtime.state.copy_at(tid, location) == COPY_READY]
                if not locations:
                    raise InvalidInput(f"fragment {tid!r} has no persistent READY copy")
                fragments.append(dict(tensor_id=tid, parent_weight_id=row["parent_weight_id"],
                    stage=matrix["stage"], version=matrix["version"],
                    storage_bytes=matrix["byte_length"], locations=locations))
        versions={matrix["tensor_id"]:matrix["version"] for row in rows for matrix in row["matrices"]}
        tensors=self._runtime.scenario.workload.tensor_by_id
        for part in partitions:
            tid=part["tensor_id"]
            locations=[location for location in ("dram","vram","ssd") if self._runtime.state.copy_at(tid,location)==COPY_READY]
            if not locations:raise InvalidInput(f"FFN partition {tid!r} has no persistent READY copy")
            fragments.append(dict(**part,version=versions.get(part["source_tensor_id"]),
                                  storage_bytes=tensors[tid].storage_bytes,locations=locations))
        fragment_ids = {row["tensor_id"] for row in fragments}
        previous_weights = {weight_id(self._compiled.layer.id, row.id): row
                            for row in self._compiled.state_in.weight_copies}
        whole_weights = []
        for tensor in self._runtime.scenario.workload.tensors:
            if tensor.role != ROLE_WEIGHT or tensor.id in fragment_ids:
                continue
            previous = previous_weights.get(tensor.id)
            locations = [location for location in ("dram", "vram", "ssd")
                         if self._runtime.state.copy_at(tensor.id, location) == COPY_READY]
            if not locations:
                raise InvalidInput(f"whole weight {tensor.id!r} has no persistent READY copy")
            whole_weights.append(dict(tensor_id=tensor.id, version=previous.version if previous else None,
                                      storage_bytes=tensor.storage_bytes, locations=locations))
        return dict(format_version="causal-persistent-state-0.1",
            state_scope="KV_hidden_and_materialized_weight_fragments_not_legacy_state_in",
            status=evaluation.status, t_ns=self._runtime.state.t,
            kv_state=kv_state, weight_fragments=fragments, whole_weights=whole_weights,
            tensor_inventory=[dict(tensor_id=tensor.id, role=tensor.role, storage_bytes=tensor.storage_bytes,
                copies={location:self._runtime.state.copy_at(tensor.id, location)
                        for location in ("dram", "vram", "ssd")})
                for tensor in self._runtime.scenario.workload.tensors],
            running=[asdict(task) for task in self._runtime.state.running],
            compute_progress=[asdict(row) for row in self._runtime.compute_progress_document()],
            resumable_checkpoint=False, scope="runtime_inventory_not_a_checkpoint")
