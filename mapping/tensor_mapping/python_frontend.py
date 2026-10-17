"""Hierarchical Python strategy plans over the existing causal tensor runtime.

Modules are selector scopes and provenance, not executable barriers. Static
bindings and exact refinements lower through the composition compiler. Runtime
Python receives public observations only and returns the existing typed commands.
This frontend does not trace arbitrary Python or invent kernels, costs or devices.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Mapping

from llm_infer_model.tensor.spec import InvalidInput, Unsupported, ROLE_WEIGHT, COPY_READY

from .causal import Decision, ControlDecision, compile_command
from .composition_ir import compile_composition_document
from .composition_policy import CompositionPolicy
from .policy_ir import OPERATION_KINDS, EXPERT_COMPUTE_KINDS
from .reconfiguration import ReconfigurationSpec
from .phase_scheduling import DataPreparation


FORMAT_VERSION = "hierarchical-strategy-0.2"


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise InvalidInput(f"strategy parameters must be finite JSON values: {error}") from error


def _identity(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
        raise InvalidInput(f"{label} must be a nonempty single-line string")
    return value


@dataclass(frozen=True)
class Select:
    """Conjunction of stable model labels; never selects by future routes."""
    layer_id: str | None = None
    invocation_id: str | None = None
    request_id: str | None = None
    kind: str | None = None
    phase: str | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is not None:
                _identity(value, f"selector.{key}")
        if self.kind is not None and self.kind not in OPERATION_KINDS:
            raise Unsupported(f"unsupported task kind {self.kind!r}")
        if self.phase is not None and self.phase not in {"prefill", "decode", "synthetic_generation"}:
            raise Unsupported("selector.phase must be prefill, decode or synthetic_generation")

    def matches(self, operation):
        return all(value is None or getattr(operation, key) == value
                   for key, value in asdict(self).items())

    def intersect(self, other):
        if not isinstance(other, Select):
            raise InvalidInput("module selectors must be Select objects")
        values = {}
        for key, value in asdict(self).items():
            added = getattr(other, key)
            if value is not None and added is not None and value != added:
                raise InvalidInput(f"contradictory nested selector {key}: {value!r}/{added!r}")
            values[key] = value if added is None else added
        return Select(**values)

    def specializes(self, other):
        own, parent = asdict(self), asdict(other)
        return all(value is None or own[key] == value for key, value in parent.items())


@dataclass(frozen=True)
class Cache:
    rule: str = "none"
    unit: str = "tensor"
    budget_bytes: int | None = None
    alpha: float | None = None
    top_p: int | None = None


@dataclass(frozen=True)
class Prefetch:
    next_layers: int = 0
    max_count: int = 0
    ranking: str = "history_frequency"
    include_dense_weights: bool = False

    def document(self):
        return dict(enabled=self.next_layers != 0,
                    scope="next_layer" if self.next_layers != 0 else "none",
                    lookahead_layers=self.next_layers, max_count=self.max_count,
                    ranking=self.ranking, include_dense_weights=self.include_dense_weights)


@dataclass(frozen=True)
class Transport:
    """An engine must match the supplied service contract; None means no services."""
    engine: str | None = None
    stage_rules: tuple = ()
    slot_reuse: str = "release_unlocked_backed_copy"
    boundary_release: str = "declared_dead_intermediates"

    def document(self):
        rules = [dict(role=role, stage=None, src=src, dst=dst, engine=self.engine)
                 for role in ("weight", "input", "intermediate", "output")
                 for src, dst in (("dram", "vram"), ("vram", "dram"))]
        rules.extend(deepcopy(list(self.stage_rules)))
        return dict(rules=rules, slot_reuse=self.slot_reuse, boundary_release=self.boundary_release)


@dataclass(frozen=True)
class Scheduling:
    """One global scheduling/storage contract shared by all module scopes."""
    placements: Mapping[str, str] = field(default_factory=dict)
    request_placements: Mapping = field(default_factory=dict)
    expert_rule: str = "fixed"
    release: str = "none"
    cache: Cache = field(default_factory=Cache)
    prefetch: Prefetch = field(default_factory=Prefetch)
    transport: Transport = field(default_factory=Transport)
    reconfiguration: ReconfigurationSpec | None = None
    default_device: str | None = None

    def sections(self):
        if not isinstance(self.cache, Cache) or not isinstance(self.prefetch, Prefetch) or not isinstance(self.transport, Transport):
            raise InvalidInput("Scheduling requires Cache, Prefetch and Transport objects")
        result = dict(policy=dict(placements=dict(self.placements),
                                  request_placements=deepcopy(dict(self.request_placements)),
                                  experts=dict(device_rule=self.expert_rule, selection_unit="expert_region"),
                                  release=dict(rule=self.release)),
                      cache=asdict(self.cache), prefetch=self.prefetch.document(),
                      transport=self.transport.document())
        if self.reconfiguration is not None:
            if not isinstance(self.reconfiguration, ReconfigurationSpec):
                raise InvalidInput("reconfiguration must be ReconfigurationSpec")
            result["reconfiguration"] = asdict(self.reconfiguration)
        return result


@dataclass(frozen=True)
class ModuleIR:
    path: tuple[str, ...]
    selector: Select


@dataclass(frozen=True)
class PlacementIR:
    selector: Select
    device: str
    priority: int = 0
    module_path: tuple[str, ...] = ()


class PlanModule:
    """Reusable Python plan component. build() records strategy choices only."""
    def build(self, plan):
        raise NotImplementedError

    def __call__(self, plan, *, name=None, select=None):
        if not isinstance(plan, ModelPlan):
            raise InvalidInput("PlanModule requires a ModelPlan")
        with plan.module(name or type(self).__name__, select=select):
            self.build(plan)
        return plan


class _Defer:
    pass


DEFER = _Defer()


class RuntimePolicy:
    """Observation-only Python extension. DEFER uses the declared base rules.

    None stops execution. Instances are recreated for each run and replay.
    Any stochastic input or external dependency must be explicitly supplied
    and source-bound by the author; Python is trusted code, not a sandbox.
    """
    def decide(self, observation):
        return DEFER

    def state_document(self):
        return {}


@dataclass(frozen=True)
class _RuntimeSpec:
    factory: type
    parameters_json: str
    source_files: tuple[str, ...]
    allow_device_override: bool = False
    allow_transport_override: bool = False
    allow_control_override: bool = False

    def document(self):
        return dict(class_name=f"{self.factory.__module__}.{self.factory.__qualname__}",
                    parameters=json.loads(self.parameters_json), source_files=list(self.source_files),
                    allow_device_override=self.allow_device_override,
                    allow_transport_override=self.allow_transport_override,
                    allow_control_override=self.allow_control_override)


def _choose_rule(rules, operation):
    matches = [rule for rule in rules if rule.selector.matches(operation)]
    if not matches:
        return None
    priority = max(rule.priority for rule in matches)
    matches = [rule for rule in matches if rule.priority == priority]
    most_specific = [rule for rule in matches if not any(
        other.selector != rule.selector and other.selector.specializes(rule.selector)
        for other in matches)]
    devices = {rule.device for rule in most_specific}
    if len(devices) != 1:
        raise InvalidInput(f"ambiguous scoped placements for {operation.kind}; use explicit priority")
    return most_specific[0]


def _rebase_execution_sources(document, directory):
    document=deepcopy(document)
    if document.get("format_version")=="execution-pipeline-0.1":
        document["transforms"]=[_rebase_execution_sources(row,directory) for row in document["transforms"]]
    elif document.get("format_version")=="dense-attention-execution-0.1":
        document["invocation_files"]=[str((Path(directory)/path).resolve()) for path in document["invocation_files"]]
    return document


class ModelPlan:
    """Mutable Python builder; compile() freezes a concrete strategy instance."""
    def __init__(self, *, id, model_file, model_kind, environment_file, scheduling,
                 base_dir=".", services_file=None, history_file=None, routing_scores_file=None,
                 max_decisions=10000, visibility_mode="data_dependencies"):
        if not isinstance(scheduling, Scheduling):
            raise InvalidInput("ModelPlan requires Scheduling")
        sections = scheduling.sections()
        document = dict(format_version="unified-strategy-0.3" if scheduling.reconfiguration else "unified-strategy-0.2",
                        id=_identity(id, "id"), model=dict(kind=model_kind, file=str(model_file)),
                        execution=[], services_file=str(services_file) if services_file is not None else None,
                        environment_file=str(environment_file),
                        history_file=str(history_file) if history_file is not None else None,
                        routing_scores_file=str(routing_scores_file) if routing_scores_file is not None else None,
                        max_decisions=max_decisions, **sections)
        if model_kind == "block":
            # Explicitly constructing a plan opts into replacing the block's
            # legacy example program, as required by the existing compiler.
            document["model"]["override_model_policy"] = True
        self._initialize(document, base_dir, visibility_mode)
        if scheduling.default_device is not None and scheduling.default_device not in {"cpu","gpu"}:
            raise Unsupported("default_device must be explicit cpu, gpu or None")
        self._default_device=scheduling.default_device

    def _initialize(self, document, base_dir, visibility_mode):
        if visibility_mode not in {"scope_barriers", "data_dependencies"}:
            raise Unsupported("visibility_mode must be scope_barriers or data_dependencies")
        self._document = deepcopy(document)
        self._base_dir = Path(base_dir).resolve()
        self._visibility_mode = visibility_mode
        self._modules = []
        self._bindings = []
        self._stack = []
        self._sources = set()
        self._source_snapshots = {}
        self._runtime_spec = None
        self._preparations = []
        self._default_device = None

    @classmethod
    def from_strategy(cls, path, *, visibility_mode="scope_barriers"):
        """Migration helper; new plans can be authored entirely through Python."""
        from .strategy_compiler import _read_object
        path = Path(path).resolve()
        raw = path.read_bytes()
        document = _read_object(path, raw)
        if document.get("format_version") not in {"unified-strategy-0.2", "unified-strategy-0.3"}:
            raise Unsupported("Python frontend imports unified-strategy-0.2 or 0.3")
        plan = cls.__new__(cls)
        plan._initialize(document, path.parent, visibility_mode)
        plan._sources.add(path)
        plan._source_snapshots[path] = hashlib.sha256(raw).hexdigest()
        return plan

    @contextmanager
    def module(self, name, *, select=None, **labels):
        name = _identity(name, "module name")
        if "/" in name:
            raise InvalidInput("module name must not contain '/'")
        if select is not None and labels:
            raise InvalidInput("use Select or keyword labels, not both")
        selected = select if select is not None else Select(**labels)
        parent = self._stack[-1] if self._stack else ModuleIR((), Select())
        row = ModuleIR((*parent.path, name), parent.selector.intersect(selected))
        if any(module.path == row.path for module in self._modules):
            raise InvalidInput(f"duplicate module path {'/'.join(row.path)}")
        self._modules.append(row)
        self._stack.append(row)
        try:
            yield self
        finally:
            self._stack.pop()

    def place(self, kind, device, *, select=None, priority=0):
        if not isinstance(kind, str) or kind not in OPERATION_KINDS:
            raise Unsupported("place requires an explicit supported task kind")
        if not isinstance(device, str) or device not in {"cpu", "gpu"}:
            raise Unsupported("this backend supports cpu and gpu placement only")
        if type(priority) is not int:
            raise InvalidInput("placement priority must be an integer")
        current = self._stack[-1] if self._stack else ModuleIR((), Select())
        selector = current.selector.intersect(Select() if select is None else select).intersect(Select(kind=kind))
        self._bindings.append(PlacementIR(selector, device, priority, current.path))
        return self

    def refine(self, invocation_id, spec, *, replace_existing=False):
        """Bind an existing exact Attention/MoE refinement, never guessed costs."""
        _identity(invocation_id, "refinement invocation")
        rows = self._document["execution"]
        previous = [row for row in rows if row["scope_id"] == invocation_id]
        if previous and not replace_existing:
            raise InvalidInput("refinement already bound; use replace_existing=True explicitly")
        self._document["execution"] = [row for row in rows if row["scope_id"] != invocation_id]
        self._document["execution"].append(dict(scope_id=invocation_id, spec=deepcopy(spec)))
        return self

    def _read_bound_source(self, path):
        from .strategy_compiler import _read_object
        path=(self._base_dir/Path(path)).resolve()
        raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        previous=self._source_snapshots.get(path)
        if previous is not None and previous!=digest:
            raise InvalidInput(f"strategy source changed since first binding: {path}")
        self._sources.add(path);self._source_snapshots[path]=digest
        return _read_object(path,raw),path

    def transform(self, invocation_id, spec):
        """Append a semantic pass, retaining prior inline/model transformations."""
        from llm_infer_model.m2.execution_pipeline import parse_transform, transform_document, ExecutionPipelineSpec
        _identity(invocation_id,"transform invocation")
        raw=deepcopy(spec) if isinstance(spec,dict) else transform_document(spec)
        normalized=parse_transform(raw)
        raw=transform_document(normalized)
        rows=self._document["execution"]
        existing=next((row for row in rows if row["scope_id"]==invocation_id),None)
        previous=[]
        if existing is not None:
            previous=(existing["spec"]["transforms"] if existing["spec"].get("format_version")=="execution-pipeline-0.1"
                      else [existing["spec"]])
        elif self._document["model"]["kind"]=="global_m2":
            model,model_path=self._read_bound_source(self._document["model"]["file"])
            entry=next((entry for round_ in model["rounds"] for entry in round_["invocations"]
                        if entry["id"]==invocation_id),None)
            if entry is not None and entry["execution_file"] is not None:
                inherited,file=self._read_bound_source(model_path.parent/entry["execution_file"])
                inherited=_rebase_execution_sources(inherited,file.parent)
                previous=inherited["transforms"] if inherited.get("format_version")=="execution-pipeline-0.1" else [inherited]
        added=raw["transforms"] if isinstance(normalized,ExecutionPipelineSpec) else [raw]
        pipeline=dict(format_version="execution-pipeline-0.1",transforms=[*previous,*added])
        parse_transform(pipeline)
        self._document["execution"]=[row for row in rows if row["scope_id"]!=invocation_id]
        self._document["execution"].append(dict(scope_id=invocation_id,spec=pipeline))
        return self

    def transform_file(self, invocation_id, path):
        raw,file=self._read_bound_source(path)
        return self.transform(invocation_id,_rebase_execution_sources(raw,file.parent))

    def prepare(self, rule):
        if not isinstance(rule,DataPreparation):
            raise InvalidInput("prepare requires a DataPreparation rule")
        # Revalidation also catches directly constructed malformed tuples.
        rule=DataPreparation.from_document(rule.document())
        self._preparations.append(rule)
        return self

    def runtime_policy(self, factory, *, parameters=None, source_files=(),
                       allow_device_override=False, allow_transport_override=False,
                       allow_control_override=False):
        if not isinstance(factory, type) or not issubclass(factory, RuntimePolicy):
            raise InvalidInput("runtime policy must be a RuntimePolicy subclass")
        parameters = {} if parameters is None else parameters
        if not isinstance(parameters, dict):
            raise InvalidInput("runtime policy parameters must be an explicit dictionary")
        if any(type(flag) is not bool for flag in (allow_device_override, allow_transport_override, allow_control_override)):
            raise InvalidInput("runtime policy delegation flags must be explicit booleans")
        source = inspect.getsourcefile(factory)
        if source is None or not Path(source).is_file():
            raise Unsupported("runtime policy needs a saved Python source file for reproducible evaluation")
        files = tuple(sorted({str(Path(source).resolve()), *(str(Path(p).resolve()) for p in source_files)}))
        self._runtime_spec = _RuntimeSpec(factory, _json(parameters), files,
                                          allow_device_override, allow_transport_override, allow_control_override)
        self._sources.update(Path(p) for p in files)
        self._source_snapshots.update({Path(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
        return self

    def compile(self, *, source_files=()):
        if self._stack:
            raise InvalidInput("close all module contexts before compiling")
        sources = self._sources | {Path(p).resolve() for p in source_files}
        for source, digest in self._source_snapshots.items():
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise InvalidInput(f"Python frontend source changed since import/binding: {source}")
        if self._document.get("model", {}).get("kind") == "block" and self._visibility_mode != "scope_barriers":
            raise Unsupported("block traversal has an explicit enforced order; data_dependencies is supported for global_m2")
        # Constructor documents and imported documents share all existing
        # model/profile/environment/source validation.
        compiled = compile_composition_document(self._document, base_dir=self._base_dir,
            source_files=tuple(sorted(sources)), visibility_mode=self._visibility_mode,
            derive_placements=True, default_device=self._default_device,
            placement_resolver=lambda operation: (rule.device if (rule:=_choose_rule(self._bindings,operation)) else None))
        for source in compiled.sources:
            expected = self._source_snapshots.get(Path(source["path"]))
            if expected is not None and source["sha256"] != expected:
                raise InvalidInput(f"Python frontend source changed during compile: {source['path']}")
        operations = _validation_operations(compiled)
        for binding in self._bindings:
            selected = [operation for operation in operations if binding.selector.matches(operation)]
            if not selected:
                raise InvalidInput(f"placement selector matches no task in this workload: {asdict(binding.selector)}")
            for operation in selected:
                if binding.device not in {price.device for price in operation.prices}:
                    raise Unsupported(f"scoped placement has no exact {binding.device} cost for {operation.kind}")
        for operation in operations:
            _choose_rule(self._bindings, operation)
        if self._bindings and compiled.composition.policy.expert_device_rule != "fixed":
            if any(binding.selector.kind in {"expert_ffn", "shared_expert_ffn", "expert_gate", "expert_up",
                                             "expert_swiglu", "expert_down"} for binding in self._bindings):
                raise Unsupported("fixed scoped expert bindings conflict with adaptive expert_rule; choose one explicitly")
        if self._bindings and compiled.composition.reconfiguration is not None and compiled.composition.reconfiguration.cpu_response != "fixed":
            raise Unsupported("fixed scoped placements require fixed cpu_response; use a runtime policy for adaptation")
        from .unified_strategy import create_session
        if (self._runtime_spec is not None and self._runtime_spec.allow_control_override and
                any(rule.trigger == "startup" for rule in self._preparations)):
            raise Unsupported("startup preparation requires the built-in resource controller; "
                              "custom control override can be combined with spare preparation only")
        initial=create_session(compiled).observe()
        for rule in self._preparations:
            rule.matching_tensor_ids(initial)
        if initial.services is not None:
            for slot in initial.services.slots:
                targets={tid for rule in self._preparations if rule.trigger=="startup" and rule.destination==slot.domain
                         for tid in rule.matching_tensor_ids(initial)}
                if len(targets.intersection(slot.members))>1:
                    raise Unsupported("startup preload cannot retain multiple tensors in the same fixed slot")
        ir = dict(format_version=FORMAT_VERSION, backend=compiled.document,
                  visibility_mode=self._visibility_mode,
                  modules=[asdict(row) for row in self._modules],
                  placements=[asdict(row) for row in self._bindings],
                  runtime_policy=self._runtime_spec.document() if self._runtime_spec else None,
                  data_preparations=[rule.document() for rule in self._preparations],
                  default_device=self._default_device,
                  contracts=dict(module_boundary="transparent_selector_scope",
                                 resource_ledger="single_modeling_runtime",
                                 python_control="construction_or_causal_runtime_callback",
                                 cross_round="trace_gate_preserved", moe_revelation="scope_router_barrier_preserved"))
        frozen = _json(ir)
        digest = hashlib.sha256(frozen.encode("utf-8")).hexdigest()
        compiled = replace(compiled, frontend_sha256=digest)
        return CompiledPlan(compiled, frozen, tuple(self._bindings), self._runtime_spec,tuple(self._preparations))


def _validation_operations(compiled):
    """Privileged compile validation; these records never enter policy objects.

    Route-dependent tasks may be checked for exact costs, as in the existing
    compiler. No task counts, resolved task lists or future routes are exported
    as selector metadata to the runtime policy.
    """
    from llm_infer_model.m2.causal import PublicOperation, DevicePrice
    from .unified_strategy import create_session
    graph = create_session(compiled).execution_document()
    boundary = graph.get("boundary") or {}
    # Execution documents retain stage metadata in the compiled model, whose
    # format is shared with the model-owned observation projection.
    model = compiled.model.aggregate if compiled.global_model else compiled.model
    model_boundary = boundary or getattr(model, "physical_boundary", None) or {}
    from llm_infer_model.m2.execution_pipeline import execution_metadata
    stages,_ = execution_metadata(model_boundary)
    result = []
    scope_lookup = {scope.id: scope for scope in getattr(compiled.model, "invocation_scopes", ())}
    for task in graph["tasks"]:
        oid = task["operation_id"]
        metadata = stages.get(oid, {})
        invocation = metadata.get("invocation_id", getattr(getattr(model, "invocation", None), "id", "single"))
        scope = scope_lookup.get(invocation)
        shape = task.get("shape", {})
        if compiled.document["model"]["kind"] == "block":
            # BlockTask has its own synthetic metadata contract, no MacroShape.
            metadata = task
            invocation = task["invocation_id"]
        prices = tuple(DevicePrice(device, row["duration_ns"], row["workspace_bytes"])
                       for device, row in graph["scenario"]["costs"]["compute"][oid].items())
        result.append(PublicOperation(oid, task["kind"], "NOT_STARTED", (), (), prices,
            request_id=task["request_id"], layer_id=metadata.get("layer_id", getattr(getattr(model, "layer", None), "id", "")),
            invocation_id=invocation, phase=("synthetic_generation" if compiled.document["model"]["kind"] == "block"
                                           else scope.phase if scope else shape["phase"])))
    return tuple(result)


class _PlanPolicy(CompositionPolicy):
    def __init__(self, compiled, bindings, runtime_spec, preparations=()):
        super().__init__(compiled.composition, history=compiled.history)
        self._scoped_bindings = bindings
        self._runtime_spec = runtime_spec
        self._python_policy = runtime_spec.factory(**json.loads(runtime_spec.parameters_json)) if runtime_spec else None
        self._custom_decisions = []
        self._preparations=preparations
        self._preparation_records=[]
        self._preparation_targets={}
        self._prepared=set()
        if self._python_policy is not None and not isinstance(self._python_policy, RuntimePolicy):
            raise InvalidInput("runtime policy factory returned an invalid instance")

    def _placement(self, operation):
        rule = _choose_rule(self._scoped_bindings, operation)
        return rule.device if rule is not None else super()._placement(operation)

    def __call__(self, observation):
        if any(rule.trigger=="startup" for rule in self._preparations) and not observation.complete and not observation.error_code:
            self._public_catalogue(observation)
            control=self._reconfiguration_step(observation)
            if control is not None:return control
            startup=self._preparation_step(self._estimate_observation(observation),"startup")
            if startup is not None:return startup
        if self._python_policy is None:
            return super().__call__(observation)
        self.observe_final(observation)
        if not self._runtime_spec.allow_control_override:
            control = self._reconfiguration_step(observation)
            if control is not None:
                return control
        decision = self._python_policy.decide(observation)
        if decision is DEFER:
            return super().__call__(observation)
        if decision is None:
            return None
        if not isinstance(decision, (Decision, ControlDecision)):
            raise InvalidInput("RuntimePolicy.decide must return Decision, ControlDecision, DEFER or None")
        if isinstance(decision, ControlDecision) and not self._runtime_spec.allow_control_override:
            raise Unsupported("runtime control override requires allow_control_override=True")
        compile_command(decision, observation)
        tensors = {tensor.id: tensor for tensor in observation.tensors}
        if isinstance(decision, Decision):
            if (self.spec.cache_budget_bytes is not None and decision.kind != "EVICT" and
                    self._budget_for_additions(tensors) > self.spec.cache_budget_bytes):
                raise Unsupported("runtime proposal observes an over-budget shared weight cache; release before other work")
            if decision.kind == "COMPUTE":
                operation = next(op for op in observation.operations if op.id == decision.target_id)
                committed = self._region_devices.get(self._region_key(operation),
                                                      self._operation_devices.get(operation.id))
                if committed is not None and decision.device != committed:
                    raise Unsupported("runtime decision conflicts with an already committed task/region device")
                rule = _choose_rule(self._scoped_bindings, operation)
                if rule is not None and decision.device != rule.device:
                    raise Unsupported("runtime decision conflicts with a fixed scoped execution binding")
                if not self._runtime_spec.allow_device_override and decision.device != self._placement(operation):
                    raise Unsupported("runtime device override requires allow_device_override=True")
            if decision.kind == "COPY" and not self._runtime_spec.allow_transport_override:
                expected = self._engine(tensors[decision.target_id], decision.src, decision.dst)
                if decision.transfer_engine != expected:
                    raise Unsupported("runtime transport override requires allow_transport_override=True")
            if decision.kind == "COPY" and decision.dst == "vram" and tensors[decision.target_id].role == ROLE_WEIGHT:
                budget = self.spec.cache_budget_bytes
                if budget is not None and self._budget_for_additions(tensors, (decision.target_id,)) > budget:
                    raise Unsupported("runtime proposal exceeds the shared weight cache budget")
            # The common driver dispatches this proposal immediately and stops
            # on rejection. Keep accepted custom work and later DEFER work in
            # the same controller ledger; a custom first stage cannot leave
            # the rest of an adaptive expert region accidentally unbound.
            self._clock += 1
            if decision.kind == "COMPUTE":
                dynamic_cpu = self.spec.reconfiguration is not None and self.spec.reconfiguration.cpu_response == "current_cost"
                if (operation.kind in EXPERT_COMPUTE_KINDS and isinstance(operation.expert_id, int) and
                        (self.policy.expert_device_rule != "fixed" or dynamic_cpu)):
                    self._region_devices[self._region_key(operation)] = decision.device
                elif dynamic_cpu:
                    self._operation_devices[operation.id] = decision.device
                for tid in operation.inputs:
                    if tensors[tid].role == ROLE_WEIGHT:
                        self._last_used[tid] = self._clock
            elif decision.kind == "COPY" and tensors[decision.target_id].role == ROLE_WEIGHT:
                self._last_used[decision.target_id] = self._clock
        self._custom_decisions.append(dict(t_ns=observation.t_ns, decision=asdict(decision)))
        return decision

    def _preparation_step(self, observation, trigger):
        if observation.complete or observation.error_code:return None
        tensors={tensor.id:tensor for tensor in observation.tensors}
        for index,rule in enumerate(self._preparations):
            targets=self._preparation_targets.setdefault(index,rule.matching_tensor_ids(observation))
            for tid in targets:
                if tensors[tid].status_at(rule.destination)==COPY_READY:self._prepared.add((index,tid))
        protected_targets={tid for index,rule in enumerate(self._preparations) if rule.trigger=="startup"
                           for tid in self._preparation_targets[index]}
        pending=False
        for index,rule in enumerate(self._preparations):
            if rule.trigger!=trigger:continue
            for tid in self._preparation_targets[index]:
                if (index,tid) in self._prepared:continue
                pending=True
                tensor=tensors[tid]
                loading=self._load_step(tensor,rule.destination)
                if loading is None:
                    if observation.running:continue
                    raise Unsupported(f"preparation has no exact available route to {rule.destination} for {tid}")
                decision,price,resource=loading
                if not self._resource_available(observation,resource):continue
                budget=self.spec.cache_budget_bytes
                if rule.destination=="vram" and budget is not None and self._budget_for_additions(tensors,(tid,))>budget:
                    if trigger=="spare":continue
                    eviction=self._eviction(observation,tensors,"vram",self._protected(observation,tensors,protected_targets))
                    if eviction:return self._record(observation,eviction,"startup_preload_budget_release",tensor=tid)
                    raise Unsupported("startup preload exceeds the shared cache budget")
                needed=self._allocation(tensor,price.dst)
                if needed>self._free(observation,price.dst):
                    if trigger=="spare":continue
                    fits,eviction=self._fit_or_evict(observation,tensors,price.dst,needed,
                                                   self._protected(observation,tensors,protected_targets))
                    if eviction:return self._record(observation,eviction,"startup_preload_capacity_release",tensor=tid)
                    if not fits:
                        if observation.running:continue
                        raise Unsupported("startup preload does not fit the current physical capacity")
                if trigger=="spare" and observation.services is not None and any(
                        slot.domain==price.dst and tid in slot.members and slot.owner not in (None,tid)
                        for slot in observation.services.slots):continue
                guarded=self._slot_guard(observation,tensors,decision)
                if trigger=="spare" and guarded!=decision:continue
                self._clock+=1;self._last_used[tid]=self._clock
                self._preparation_records.append(dict(t_ns=observation.t_ns,rule_index=index,tensor_id=tid,
                    destination=rule.destination,trigger=trigger,action=asdict(guarded)))
                return self._record(observation,guarded,"explicit_"+trigger+"_preparation",tensor=tid,
                    rule_index=index,destination=rule.destination,information_scope="public_static_weight_catalogue")
        if trigger=="startup" and pending:
            return self._record(observation,Decision("ADVANCE"),"wait_for_startup_preparation")
        return None

    def _prefetch(self, observation, tensors):
        prepared=self._preparation_step(observation,"spare") if self._preparations else None
        return prepared if prepared is not None else super()._prefetch(observation,tensors)

    def audit_document(self):
        result = super().audit_document()
        state = self._python_policy.state_document() if self._python_policy else None
        _json(state)
        result.update(scoped_placements=[asdict(row) for row in self._scoped_bindings],
                      python_decisions=list(self._custom_decisions), python_state=state,
                      explicit_data_preparations=list(self._preparation_records),
                      arbitrary_python_static_compilation=False)
        return result


@dataclass(frozen=True)
class CompiledPlan:
    _compiled: object
    _ir_json: str
    _bindings: tuple[PlacementIR, ...]
    _runtime_spec: _RuntimeSpec | None
    _preparations: tuple = ()

    def ir_document(self):
        return json.loads(self._ir_json)

    def run(self, *, checkpoint=None, max_decisions=None):
        from .unified_strategy import run_unified_strategy
        return run_unified_strategy(self._compiled, checkpoint=checkpoint, max_decisions=max_decisions,
            policy_factory=lambda compiled: _PlanPolicy(compiled, self._bindings, self._runtime_spec,self._preparations))

    def checkpoint(self, result):
        from .unified_strategy import checkpoint_document
        return checkpoint_document(self._compiled, result)

    def write_result(self, output, result):
        from .artifacts import write_json
        from .unified_strategy import write_unified_result
        write_unified_result(output, self._compiled, result)
        write_json(Path(output) / "plan.json", self.ir_document())
        path = Path(output) / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary.update(python_frontend=FORMAT_VERSION, frontend_sha256=self._compiled.frontend_sha256,
                       module_boundary="transparent_selector_scope", module_count=len(self.ir_document()["modules"]))
        write_json(path, summary)
