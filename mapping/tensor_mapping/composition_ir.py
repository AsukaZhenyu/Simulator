"""One strategy document for execution decomposition and scheduling mechanisms.

Model/profile/trace files remain reusable inputs. All mapping choices are
inline and source-bound; a legacy model program is never silently overridden.
"""
import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from llm_infer_model.m2.attention_execution import (
    DenseAttentionExecutionSpec, parse_attention_execution, transform_dense_attention)
from llm_infer_model.m2.row_execution import parse_execution_spec
from llm_infer_model.m2.routing_scores import parse_routing_scores
from llm_infer_model.tensor.services import parse_services
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .causal_cli import parse_environment, verify_sources
from .composition_policy import CompositionPolicy, composition_policy_document, parse_composition_policy
from .policy_ir import parse_history
from .strategy_compiler import _read_object, _resolve_scenario
from .reconfiguration import parse_environment_trace, parse_reconfiguration
from llm_infer_model.m2.execution_pipeline import (
    parse_transform, transform_document, ExecutionPipelineSpec, apply_execution_transform, execution_metadata,
)


FIELDS = {"format_version", "id", "model", "execution", "policy", "cache", "prefetch", "transport",
          "services_file", "environment_file", "history_file", "routing_scores_file", "max_decisions"}


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")
    return value


def _reference(value, label, *, nullable=False):
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InvalidInput(f"{label} must be a nonempty path" + (" or explicit null" if nullable else ""))
    return value


def compile_composition_file(path):
    """Compile the existing file format without changing its legacy semantics."""
    return _compile_composition(path=Path(path).resolve())


def compile_composition_document(document, *, base_dir, source_files=(),
                                 visibility_mode="scope_barriers", derive_placements=False,
                                 placement_resolver=None, default_device=None):
    """Compile an in-memory Python frontend through the same strict compiler.

    Model, environment and profile paths are resolved against ``base_dir``.
    Additional Python sources are fingerprinted as bytes, never parsed as JSON.
    No temporary strategy file or second evaluator is involved.
    """
    if not isinstance(document, dict):
        raise InvalidInput("composition document must be an object")
    return _compile_composition(document=deepcopy(document), base_dir=base_dir,
                                source_files=source_files, visibility_mode=visibility_mode,
                                derive_placements=derive_placements, placement_resolver=placement_resolver,
                                default_device=default_device)


def _compile_composition(*, path=None, document=None, base_dir=None,
                         source_files=(), visibility_mode="scope_barriers", derive_placements=False,
                         placement_resolver=None, default_device=None):
    from .unified_strategy import CompiledUnifiedStrategy, create_session
    # A synthetic path is only an anchor for relative references, not an input
    # that is written to disk or included in the source manifest.
    path = Path(path).resolve() if path is not None else Path(base_dir).resolve() / "python-strategy.json"
    snapshots, documents = {}, {}

    def read(source):
        source = Path(source).resolve()
        if source not in snapshots:
            snapshots[source] = source.read_bytes()
            documents[source] = _read_object(source, snapshots[source])
        return documents[source]

    def include(sources):
        for source in sources:
            file = Path(source["path"]).resolve()
            read(file)
            if hashlib.sha256(snapshots[file]).hexdigest() != source["sha256"]:
                raise InvalidInput(f"model source changed during composition compile: {file}")

    document = read(path) if document is None else document
    if visibility_mode not in {"scope_barriers", "data_dependencies"}:
        raise Unsupported("visibility_mode must be scope_barriers or data_dependencies")
    version = document.get("format_version")
    if version not in {"unified-strategy-0.2", "unified-strategy-0.3"}:
        raise Unsupported("composition requires unified-strategy-0.2 or unified-strategy-0.3")
    _exact(document, FIELDS | ({"reconfiguration"} if version == "unified-strategy-0.3" else set()), version)
    _reference(document["id"], "id")
    if type(document["max_decisions"]) is not int or document["max_decisions"] < 1:
        raise InvalidInput("max_decisions must be positive")
    model_descriptor = document["model"]
    if not isinstance(model_descriptor, dict):
        raise InvalidInput("model must be an object")
    kind = model_descriptor.get("kind")
    if kind not in ("m2", "global_m2", "block"):
        raise Unsupported("model.kind must be m2, global_m2 or block")
    if kind == "block" and visibility_mode != "scope_barriers":
        raise Unsupported("block traversal has an enforced order; data_dependencies requires a global_m2 model")
    _exact(model_descriptor, {"kind", "file", "override_model_policy"} if kind == "block" else {"kind", "file"}, "model")
    if kind == "block" and model_descriptor["override_model_policy"] is not True:
        raise InvalidInput("block model requires explicit override_model_policy:true to replace its legacy fixed program")
    model_path = (path.parent / _reference(model_descriptor["file"], "model.file")).resolve()
    raw_model = read(model_path)
    _exact(document["policy"], {"placements", "request_placements", "experts", "release"}, "policy")
    if not isinstance(document["execution"], list):
        raise InvalidInput("execution must be an array of inline scope bindings")
    selections, seen = [], set()
    scope_files = {}
    if kind == "global_m2":
        if raw_model.get("format_version") != "global-execution-0.1":
            raise InvalidInput("global_m2 requires a global-execution-0.1 model")
        _exact(raw_model, {"format_version", "id", "rounds"}, "global model")
        if not isinstance(raw_model["rounds"], list) or not raw_model["rounds"]:
            raise InvalidInput("global model rounds must be a nonempty array")
        # Validate the reference structure before resolving overrides. Calling
        # the legacy compiler first would read files that the IR replaces.
        for round_ in raw_model["rounds"]:
            _exact(round_, {"id", "input_mode", "gate_cost", "invocations"}, "global round")
            if not isinstance(round_["invocations"], list) or not round_["invocations"]:
                raise InvalidInput("global round invocations must be a nonempty array")
            for entry in round_["invocations"]:
                _exact(entry, {"id", "invocation_file", "execution_file"}, "global invocation")
                scope = _reference(entry["id"], "global invocation.id")
                if scope in scope_files:
                    raise InvalidInput("global invocation scopes must be unique")
                reference = _reference(entry["invocation_file"], "global invocation.invocation_file")
                scope_files[scope] = (model_path.parent / reference).resolve()
    elif kind == "m2":
        scope_files = {"single": model_path}
    for row in document["execution"]:
        _exact(row, {"scope_id", "spec"}, "execution binding")
        scope = row["scope_id"]
        if not isinstance(scope, str) or scope not in scope_files or scope in seen:
            raise InvalidInput("execution scope must be present, unique and use 'single' for an m2 model")
        seen.add(scope)
        raw = row["spec"]
        if not isinstance(raw, dict):
            raise InvalidInput("execution.spec must be an inline object")
        selection = parse_transform(raw)
        components=selection.transforms if isinstance(selection,ExecutionPipelineSpec) else (selection,)
        for component in components:
            if isinstance(component, DenseAttentionExecutionSpec):
                if len(component.invocation_files) != 1 or (
                        path.parent / component.invocation_files[0]).resolve() != scope_files[scope]:
                    raise InvalidInput("inline attention invocation_files must identify its exact bound model scope, relative to the IR")
        selections.append((scope, selection))
    execution = None
    if kind == "global_m2":
        from llm_infer_model.m2.global_execution import compile_global_execution_file
        model = compile_global_execution_file(model_path, execution_overrides=tuple(selections),
                                              visibility_mode=visibility_mode)
        include(model.sources)
        positions = {}
        for entry in raw_model["rounds"][0]["invocations"]:
            raw = read(model_path.parent / entry["invocation_file"])
            positions.update({(raw["layer"], row["request_id"]): row["position_start"] for row in raw["requests"]})
    elif kind == "m2":
        _, model = _resolve_scenario(model_path, read)
        if model is None:
            raise Unsupported("m2 model must be an actual layer invocation")
        if selections:
            execution = selections[0][1]
            from llm_infer_model.m2.row_execution import MoEExecutionSpec
            if not isinstance(execution,MoEExecutionSpec):
                model = apply_execution_transform(model, execution)
                execution = None
        positions = {(model.layer.id, row.request_id): row.position_start for row in model.invocation.requests}
    else:
        from .composition_block import load_block_model_file
        if selections:
            raise Unsupported("block models declare finite synthetic tasks; they cannot bind M2 execution decomposition")
        model = load_block_model_file(model_path)
        include(model.sources)
        positions = {}
    validation_model = model.aggregate if kind=="global_m2" else model
    if execution is not None:
        validation_model=apply_execution_transform(validation_model,execution)
    if derive_placements:
        _derive_placements(document,validation_model,model,placement_resolver,default_device)
    spec = parse_composition_policy(dict(document["policy"], cache=document["cache"],
                                       prefetch=document["prefetch"], transport=document["transport"]))
    if version == "unified-strategy-0.3":
        spec = replace(spec, reconfiguration=parse_reconfiguration(document["reconfiguration"]))
    resolved = {}
    for name, parser, empty in (("services_file", parse_services, None), ("environment_file",
                                parse_environment_trace if version == "unified-strategy-0.3" else parse_environment, ()),
                                ("history_file", parse_history, ()), ("routing_scores_file", parse_routing_scores, ())):
        reference = _reference(document[name], name, nullable=name != "environment_file")
        resolved[name] = parser(read(path.parent / reference)) if reference is not None else empty
    if kind == "block" and (resolved["history_file"] or resolved["routing_scores_file"]):
        raise Unsupported("synthetic block models have no MoE routing catalogue")
    for source in source_files:
        source = Path(source).resolve()
        data = source.read_bytes()
        if source in snapshots and snapshots[source] != data:
            raise InvalidInput(f"source changed during composition compile: {source}")
        snapshots[source] = data
    sources = tuple({"path": str(file), "sha256": hashlib.sha256(data).hexdigest()}
                    for file, data in sorted(snapshots.items(), key=lambda row: str(row[0])))
    verify_sources(sources)
    compiled = CompiledUnifiedStrategy(document, model, kind == "global_m2", execution, spec.policy,
        resolved["services_file"], (resolved["environment_file"].events if version == "unified-strategy-0.3"
                                    else resolved["environment_file"]), resolved["history_file"],
        resolved["routing_scores_file"], sources, spec,
        resolved["environment_file"].requests if version == "unified-strategy-0.3" else (),
        resolved["environment_file"].control_costs if version == "unified-strategy-0.3" else None,
        visibility_mode=visibility_mode)
    if version == "unified-strategy-0.3" and compiled.control_costs is None:
        raise InvalidInput("unified-strategy-0.3 requires environment-trace-0.2 with explicit control costs")
    session = create_session(compiled)
    catalogue = {}
    for expert in session.observe().experts:
        catalogue.setdefault(expert.layer_id, set()).add(expert.expert_id)
    for row in compiled.history:
        key = (row.layer_id, row.request_id)
        if key not in positions or row.absolute_position >= positions[key]:
            raise InvalidInput("imported history must precede the first invocation for its layer/request")
        if not set(row.expert_ids) <= catalogue.get(row.layer_id, set()):
            raise InvalidInput("imported history references an unknown layer/expert")
        if row.router_scores_scope == "all_experts" and {expert for expert, _ in row.router_scores} != {
                expert for expert in catalogue.get(row.layer_id, set()) if isinstance(expert, int)}:
            raise InvalidInput("historical full scores must cover the exact public expert catalogue")
    if spec.policy.cache_rule == "mrs" and not compiled.routing_scores:
        raise Unsupported("MRS requires full router-score evidence; top-k is insufficient")
    graph = session.execution_document()
    if spec.policy.expert_device_rule == "hybrimoe_queues" and any(
            task["kind"].startswith("ffn_") and isinstance(task.get("expert_id"), int)
            for task in graph["tasks"]):
        raise Unsupported("hybrimoe_queues requires a whole-expert device choice; "
                          "partitioned FFN stages have explicit per-shard devices")
    original = composition_policy_document(spec)
    placements, overrides = original["placements"], original["request_placements"]
    requests_and_kinds = {}
    for task in graph["tasks"]:
        kind_, request, oid = task["kind"], task["request_id"], task["operation_id"]
        requests_and_kinds.setdefault(request, set()).add(kind_)
        allowed = graph["scenario"]["mapspace"]["allowed_devices"][oid]
        if kind_ == "state_update":
            if len(allowed) != 1:
                raise InvalidInput("trace gate must have one explicit device")
            continue
        context=_task_context(task,validation_model,model)
        resolved=placement_resolver(context) if placement_resolver is not None else None
        device = resolved if resolved is not None else overrides.get(request, {}).get(kind_, placements.get(kind_))
        if device == "execution_binding":
            if len(allowed) != 1:
                raise Unsupported("execution_binding requires one exact device per transformed stage")
        elif device is None or device not in allowed:
            raise Unsupported(f"explicit placement for {request}/{kind_} is absent or has no allowed exact cost")
    if any(request not in requests_and_kinds or set(bindings) - requests_and_kinds[request]
           for request, bindings in overrides.items()):
        raise InvalidInput("request placement references an absent request or kind")
    # This first public projection checks service engines without exposing the
    # privately validated graph or future route inputs to the policy object.
    CompositionPolicy(spec, history=compiled.history)._estimate_observation(session.observe())
    return compiled


def _task_context(task, model, owner):
    from types import SimpleNamespace
    stages,_=execution_metadata(getattr(model,"physical_boundary",None))
    oid=task["operation_id"] if isinstance(task,dict) else task.operation_id
    kind=task["kind"] if isinstance(task,dict) else task.kind
    request=task["request_id"] if isinstance(task,dict) else task.request_id
    metadata=stages.get(oid,{})
    if isinstance(task,dict):
        layer=task.get("layer_id",getattr(getattr(model,"layer",None),"id",""))
        invocation=task.get("invocation_id",getattr(getattr(model,"invocation",None),"id","single"))
        phase=task.get("shape",{}).get("phase","synthetic_generation")
    else:
        layer=getattr(task,"layer_id",getattr(getattr(model,"layer",None),"id",""))
        invocation=getattr(task,"invocation_id",getattr(getattr(model,"invocation",None),"id","single"))
        phase=getattr(getattr(task,"shape",None),"phase","synthetic_generation")
    layer=metadata.get("layer_id",layer);invocation=metadata.get("invocation_id",invocation)
    scopes={scope.id:scope for scope in getattr(owner,"invocation_scopes",())}
    if invocation in scopes:phase=scopes[invocation].phase
    return SimpleNamespace(id=oid,kind=kind,request_id=request,layer_id=layer,invocation_id=invocation,phase=phase)


def _derive_placements(document, model, owner, resolver, default_device):
    """Complete only declared preferences or fully bound stages/selectors.

    Hardware costs are never inferred. ``scoped_binding`` is an unreachable
    parser token: the frontend must resolve it for every affected operation.
    """
    from llm_infer_model.m2.execution_pipeline import TRANSFORM_KINDS
    placements=document["policy"]["placements"]
    if not isinstance(placements,dict):raise InvalidInput("placements must be an object")
    groups={}
    for task in model.tasks:
        if task.kind!="state_update":groups.setdefault(task.kind,[]).append(task)
    for kind,tasks in groups.items():
        if kind in placements:continue
        allowed=[model.scenario.mapspace.devices_for(task.operation_id) for task in tasks]
        contexts=[_task_context(task,model,owner) for task in tasks]
        resolved=[resolver(context) if resolver else None for context in contexts]
        if all(len(devices)==1 for devices in allowed) and (kind in TRANSFORM_KINDS or kind=="attention_core"):
            placements[kind]="execution_binding"
        elif all(device is not None for device in resolved):
            placements[kind]="scoped_binding"
        else:
            if kind.startswith("attention_"):
                parent="projection" if kind=="attention_projection" else "attention"
            elif kind.startswith("ffn_"):
                parent="expert_ffn" if any(getattr(task,"expert_id",None) is not None for task in tasks) else "dense_ffn"
            elif kind in {"gather_rows","scatter_rows","combine_rows"}:
                parent="combine"
            elif kind in {"expert_gate","expert_up","expert_swiglu","expert_down"}:
                parent="expert_ffn"
            else:parent=kind
            preferred=placements.get(parent,default_device)
            if preferred not in {"cpu","gpu"}:
                raise Unsupported(f"no explicit placement or complete scoped binding for {kind}")
            if any(choice is None and preferred not in devices for choice,devices in zip(resolved,allowed)):
                raise Unsupported(f"preferred {preferred} has no exact cost for an unbound {kind} task")
            placements[kind]=preferred
