"""Ordered, composable execution transforms with shared model/state contracts.

This module owns lowering only. It never selects a mapping, estimates missing
prices, or runs a second simulator. Individual passes preserve hidden/KV ports
and contribute metadata to the same dependency graph.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

from ..tensor.spec import InvalidInput, Unsupported

FORMAT_VERSION = "execution-pipeline-0.1"
REGION_FORMAT_VERSION = "execution-region-0.1"
SECTIONS = ("row_execution", "attention_execution", "attention_stages", "ffn_execution")
STATIC_TASK_KINDS = frozenset({"projection", "attention", "router", "attention_core", "attention_output",
    "attention_projection", "attention_qk", "attention_softmax", "attention_pv"})
TRANSFORM_KINDS = frozenset({"ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down", "ffn_sum", "ffn_residual",
    "attention_qk", "attention_softmax", "attention_pv", "attention_projection", "attention_output"})


@dataclass(frozen=True)
class ExecutionPipelineSpec:
    transforms: tuple
    format_version: str = FORMAT_VERSION

    @property
    def variant(self):
        return "execution_pipeline"

    @property
    def expert_variant(self):
        return next((getattr(spec, "expert_variant", None) for spec in reversed(self.transforms)
                     if getattr(spec, "expert_variant", None)), None)


@dataclass(frozen=True)
class ExecutionRegionSpec:
    """One transformation over explicit, public request identities.

    A region does not create a runtime or storage pool. Its transformation must
    preserve every unselected task and all hidden/KV ports. Physical weight
    representation remains an explicit part of the inner transform contract.
    """
    request_ids: tuple[str, ...]
    transform: object
    format_version: str = REGION_FORMAT_VERSION

    @property
    def variant(self):
        return "execution_region"

    @property
    def expert_variant(self):
        return getattr(self.transform, "expert_variant", None)


def parse_execution_region(document):
    if not isinstance(document, dict) or set(document) != {"format_version", "request_ids", "transform"}:
        raise InvalidInput("execution region requires exactly format_version, request_ids and transform")
    if document["format_version"] != REGION_FORMAT_VERSION:
        raise Unsupported("unsupported execution region version")
    selected = document["request_ids"]
    if not isinstance(selected, list) or not selected or any(
            not isinstance(item, str) or not item or any(c in item for c in ":\r\n") for item in selected):
        raise InvalidInput("execution region requires nonempty public request identities")
    if len(selected) != len(set(selected)):
        raise InvalidInput("execution region request identities must be unique")
    inner = document["transform"]
    if not isinstance(inner, dict) or inner.get("format_version") not in {"attention-stages-0.1", "ffn-execution-0.1"}:
        raise Unsupported("request regions support Attention stages and FFN partitions; no nested region/pipeline or routed-row rewrite")
    return ExecutionRegionSpec(tuple(selected), parse_transform(inner))


def parse_transform(document):
    if not isinstance(document, dict):
        raise InvalidInput("execution transform must be an object")
    version = document.get("format_version")
    if not isinstance(version, str):
        raise InvalidInput("execution transform format_version must be a string")
    if version == FORMAT_VERSION:
        return parse_execution_pipeline(document)
    if version == REGION_FORMAT_VERSION:
        return parse_execution_region(document)
    if version == "dense-attention-execution-0.1":
        from .attention_execution import parse_attention_execution
        return parse_attention_execution(document)
    if version in {"moe-execution-0.1", "moe-execution-0.2"}:
        from .row_execution import parse_execution_spec
        return parse_execution_spec(document)
    if version == "attention-stages-0.1":
        from .attention_stages import parse_attention_stages
        return parse_attention_stages(document)
    if version == "ffn-execution-0.1":
        from .ffn_execution import parse_ffn_execution
        return parse_ffn_execution(document)
    raise Unsupported(f"unsupported execution transform {version!r}")


def transform_document(spec):
    from .attention_execution import DenseAttentionExecutionSpec, attention_execution_document
    from .row_execution import MoEExecutionSpec, execution_spec_to_document
    from .attention_stages import AttentionStagesSpec, attention_stages_document
    from .ffn_execution import FFNExecutionSpec, ffn_execution_document
    if isinstance(spec, ExecutionPipelineSpec):
        return dict(format_version=spec.format_version, transforms=[transform_document(row) for row in spec.transforms])
    if isinstance(spec, ExecutionRegionSpec):
        if not isinstance(spec.request_ids, tuple):
            raise InvalidInput("typed execution region request_ids must be an immutable tuple")
        return dict(format_version=spec.format_version, request_ids=list(spec.request_ids),
                    transform=transform_document(spec.transform))
    for cls, writer in ((DenseAttentionExecutionSpec, attention_execution_document),
                        (MoEExecutionSpec, execution_spec_to_document),
                        (AttentionStagesSpec, attention_stages_document), (FFNExecutionSpec, ffn_execution_document)):
        if isinstance(spec, cls):
            return writer(spec)
    raise InvalidInput("execution requires a parsed supported transform")


def parse_execution_pipeline(document):
    if not isinstance(document, dict) or set(document) != {"format_version", "transforms"}:
        raise InvalidInput("execution pipeline requires exactly format_version and transforms")
    if document["format_version"] != FORMAT_VERSION:
        raise Unsupported("unsupported execution pipeline version")
    rows = document["transforms"]
    if not isinstance(rows, list) or not rows:
        raise InvalidInput("execution pipeline requires a nonempty ordered transform list")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("format_version") == FORMAT_VERSION:
            raise InvalidInput("execution pipelines must be flat, not nested")
        if row.get("format_version") == "dense-attention-execution-0.1" and index:
            raise Unsupported("legacy pure_attention must precede other transforms; use attention-stages for composition")
        result.append(parse_transform(row))
    families = []
    for row in result:
        inner = row.transform if isinstance(row, ExecutionRegionSpec) else row
        version = getattr(inner, "format_version", "")
        family = "moe-execution" if version.startswith("moe-execution-") else version
        selected = set(row.request_ids) if isinstance(row, ExecutionRegionSpec) else None
        for previous_family, previous_selected in families:
            if family == previous_family and (selected is None or previous_selected is None or selected & previous_selected):
                raise InvalidInput("duplicate transform family on overlapping regions in one invocation pipeline")
        families.append((family, selected))
    return ExecutionPipelineSpec(tuple(result))


def execution_metadata(boundary):
    """Merge refinement metadata; later sections describe replaced regions."""
    boundary = boundary or {}
    operations = deepcopy(boundary.get("operation_stages", {}))
    tensors = deepcopy(boundary.get("tensor_row_scopes", {}))
    for name in SECTIONS:
        section = boundary.get(name, {})
        operations.update(deepcopy(section.get("operation_stages", {})))
        tensors.update(deepcopy(section.get("tensor_row_scopes", {})))
    return operations, tensors


def weight_children(boundary):
    result = {row["parent_weight_id"]: tuple(matrix["tensor_id"] for matrix in row["matrices"])
              for row in (boundary or {}).get("row_execution", {}).get("subweights", ())}
    result.update({parent: tuple(children) for parent, children in
                   (boundary or {}).get("ffn_execution", {}).get("weight_children", {}).items()})
    return result


def materialized_weight_members(boundary, parent, available):
    children = weight_children(boundary)
    def visit(tid, seen):
        if tid in available:
            return (tid,)
        if tid in seen or tid not in children:
            raise InvalidInput(f"weight representation has no materialized leaf for {tid!r}")
        return tuple(leaf for child in children[tid] for leaf in visit(child, seen | {tid}))
    result = visit(parent, set())
    if len(result) != len(set(result)):
        raise InvalidInput("weight representation duplicates a materialized leaf")
    return result


def apply_execution_transform(compiled, spec):
    from .attention_execution import DenseAttentionExecutionSpec, transform_dense_attention
    from .row_execution import MoEExecutionSpec, transform_moe_execution
    from .attention_stages import AttentionStagesSpec, transform_attention_stages
    from .ffn_execution import FFNExecutionSpec, transform_ffn_execution
    if isinstance(spec, ExecutionPipelineSpec):
        normalized = parse_execution_pipeline(transform_document(spec))
        original_hidden = dict(compiled.hidden_out_ids)
        for component in normalized.transforms:
            compiled = apply_execution_transform(compiled, component)
            if dict(compiled.hidden_out_ids) != original_hidden:
                raise InvalidInput("execution transform changed the model's hidden output ports")
        boundary = deepcopy(compiled.physical_boundary or {})
        boundary["execution_pipeline"] = transform_document(normalized)
        return replace(compiled, physical_boundary=boundary)
    if isinstance(spec, ExecutionRegionSpec):
        spec = parse_execution_region(transform_document(spec))
        available = {request.request_id for request in compiled.invocation.requests}
        if not set(spec.request_ids).issubset(available):
            raise InvalidInput("execution region references an absent request")
        selected = set(spec.request_ids)
        protected_ids = {task.operation_id for task in compiled.tasks if task.request_id not in selected}
        protected_tasks = {task.operation_id: task for task in compiled.tasks if task.operation_id in protected_ids}
        before = {op["id"]: op for op in compiled.workload_document["operations"] if op["id"] in protected_ids}
        protected_tensors = {tid for op in before.values()
                             for tid in [*op["inputs"], op["output"], *op.get("extra_outputs", ())]}
        tensor_before = {row["id"]: row for row in compiled.workload_document["tensors"]
                         if row["id"] in protected_tensors}
        inner = spec.transform
        if isinstance(inner, AttentionStagesSpec):
            result = transform_attention_stages(compiled, inner, request_ids=spec.request_ids)
        elif isinstance(inner, FFNExecutionSpec):
            result = transform_ffn_execution(compiled, inner, request_ids=spec.request_ids)
        else:
            raise Unsupported("unsupported region transformation")
        after = {op["id"]: op for op in result.workload_document["operations"]}
        after_tasks = {task.operation_id: task for task in result.tasks}
        for oid, operation in before.items():
            if (after.get(oid) != operation or after_tasks.get(oid) != protected_tasks[oid] or
                    result.scenario_document["costs"]["compute"].get(oid) != compiled.scenario_document["costs"]["compute"].get(oid) or
                    result.scenario_document["mapspace"]["allowed_devices"].get(oid) != compiled.scenario_document["mapspace"]["allowed_devices"].get(oid)):
                raise InvalidInput("execution region changed an unselected task or its exact cost")
        tensor_after = {row["id"]: row for row in result.workload_document["tensors"]}
        if any(tensor_after.get(tid) != tensor for tid, tensor in tensor_before.items()):
            raise InvalidInput("execution region changed an unselected tensor or shared physical storage")
        if result.hidden_out_ids != compiled.hidden_out_ids or result.new_blocks != compiled.new_blocks:
            raise InvalidInput("execution region changed hidden or KV output identities")
        boundary = deepcopy(result.physical_boundary or {})
        boundary.setdefault("execution_regions", []).append(transform_document(spec))
        return replace(result, physical_boundary=boundary)
    # Revalidate typed objects through the same strict serializer/parser.
    spec = parse_transform(transform_document(spec))
    for cls, transform in ((DenseAttentionExecutionSpec, transform_dense_attention),
                           (MoEExecutionSpec, transform_moe_execution),
                           (AttentionStagesSpec, transform_attention_stages), (FFNExecutionSpec, transform_ffn_execution)):
        if isinstance(spec, cls):
            return transform(compiled, spec)
    raise InvalidInput("unsupported typed execution transform")
