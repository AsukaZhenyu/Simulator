"""Architecture-only context over an immutable, causal runtime observation."""
from dataclasses import dataclass, fields

from llm_infer_model.m2.causal import CausalObservation
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, OP_DONE, OP_NOT_STARTED, ROLE_WEIGHT


@dataclass(frozen=True)
class CompositionObservation(CausalObservation):
    layer_order: tuple[str, ...] = ()
    active_invocations: tuple[tuple[str, str], ...] = ()
    weight_catalogue: tuple[tuple[str, str, str, int | str | None], ...] = ()
    release_candidates: tuple[tuple[str, str], ...] = ()
    recoverable_release_candidates: tuple[tuple[str, str], ...] = ()
    completed_invocations: tuple[str, ...] = ()


class CompositionSession:
    """Never gives a policy a model, session, future task, or route sidecar.

    The added catalogue consists of immutable model weights and declared layer
    order. Active invocations are derived only from already public operations.
    The driver retains the underlying session to dispatch runtime commands.
    """
    def __init__(self, session, model, kind):
        self._session = session
        self._kind = kind
        self._dependency_visibility = (kind == "global_m2" and
            getattr(model, "visibility_mode", "scope_barriers") == "data_dependencies")
        self._incoming_hidden = ({scope.id: scope.incoming_hidden for scope in model.invocation_scopes}
                                 if self._dependency_visibility else {})
        if kind == "block":
            self._layers = model.layer_order
            self._catalogue = ()  # adapter already supplies the same public fields
            return
        binding = model.aggregate if kind == "global_m2" else getattr(session,"_compiled",model)
        self._layers = (tuple(dict.fromkeys(scope.layer_id for scope in model.invocation_scopes))
                        if kind == "global_m2" else (model.layer.id,))
        observation = session.observe()
        expert_members = {}
        for expert in observation.experts:
            members = expert.weight_tensor_ids or ((expert.weight_tensor_id,) if expert.weight_tensor_id else ())
            for member in members:
                expert_members[member] = (expert.layer_id, expert.expert_id)
        stages = {}
        boundary = binding.physical_boundary or {}
        for part in boundary.get("row_execution", {}).get("subweights", ()):
            for matrix in part["matrices"]:
                stages[matrix["tensor_id"]] = matrix["stage"]
        for part in boundary.get("ffn_execution",{}).get("weight_partitions",()):
            stages[part["tensor_id"]]=part["stage"]
        catalogue = []
        for tensor in binding.scenario.workload.tensors:
            if tensor.role != ROLE_WEIGHT:
                continue
            owner = expert_members.get(tensor.id)
            if owner is None:
                # Exact ownership is model architecture, not inferred from
                # names or private future operation inputs. Layer IDs may
                # themselves contain colons.
                layer = (boundary.get("global_execution", {}).get("weight_owners", {}).get(tensor.id)
                         if kind == "global_m2" else model.layer.id)
                if layer not in self._layers:
                    raise InvalidInput(f"weight {tensor.id!r} has no unique declared layer owner")
                owner = (layer, None)
            catalogue.append((tensor.id, owner[0], stages.get(tensor.id, ""), owner[1]))
        self._catalogue = tuple(sorted(catalogue, key=lambda row: row[0]))

    def observe(self):
        observation = self._session.observe()
        values = {field.name: getattr(observation, field.name) for field in fields(CausalObservation)}
        if self._kind == "block":
            active = observation.active_invocations
            catalogue = observation.weight_catalogue
        else:
            active = tuple(dict.fromkeys((operation.invocation_id, operation.layer_id)
                           for operation in observation.operations
                           if operation.status != OP_DONE and operation.kind != "state_update"))
            if self._dependency_visibility:
                # Public visibility permits planning, not activation. Merely
                # prefetching a future layer's weights must not make that
                # layer a source for another wave of lookahead prefetch.
                started = {operation.invocation_id for operation in observation.operations
                           if operation.kind != "state_update" and operation.status != OP_NOT_STARTED}
                ready = {tensor.id for tensor in observation.tensors
                         if any(status == COPY_READY for _, status in tensor.copies)}
                active = tuple((scope, layer) for scope, layer in active
                               if scope in started or ready.intersection(self._incoming_hidden[scope]))
            catalogue = self._catalogue
        # A launched speculative transfer consumes service, capacity and time
        # even if inference has already reached its output goal. This envelope
        # explicitly drains committed work before reporting completion; the
        # goal latency therefore includes that tail, rather than granting free
        # unfinished speculation. No further speculative target is opened.
        if observation.complete and observation.running:
            values["complete"] = False
            active = ()
        return CompositionObservation(**values, layer_order=self._layers,
            active_invocations=active, weight_catalogue=catalogue,
            release_candidates=getattr(observation, "release_candidates", ()),
            recoverable_release_candidates=getattr(observation, "recoverable_release_candidates", ()),
            completed_invocations=getattr(observation, "completed_invocations", ()))

    def __getattr__(self, name):
        return getattr(self._session, name)
