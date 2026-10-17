"""Ordered, composable execution transforms with shared model/state contracts.

This module owns lowering only. It never selects a mapping, estimates missing
prices, or runs a second simulator. Individual passes preserve hidden/KV ports
and contribute metadata to the same dependency graph.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

from ..tensor.spec import InvalidInput, Unsupported

FORMAT_VERSION = "execution-pipeline-0.1"
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


def parse_transform(document):
    if not isinstance(document, dict):
        raise InvalidInput("execution transform must be an object")
    version = document.get("format_version")
    if not isinstance(version, str):
        raise InvalidInput("execution transform format_version must be a string")
    if version == FORMAT_VERSION:
        return parse_execution_pipeline(document)
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
    versions = [("moe-execution" if getattr(row,"format_version","").startswith("moe-execution-") else
                 getattr(row,"format_version",None)) for row in result]
    if len(versions) != len(set(versions)):
        raise InvalidInput("duplicate transform family in one invocation pipeline")
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
    # Revalidate typed objects through the same strict serializer/parser.
    spec = parse_transform(transform_document(spec))
    for cls, transform in ((DenseAttentionExecutionSpec, transform_dense_attention),
                           (MoEExecutionSpec, transform_moe_execution),
                           (AttentionStagesSpec, transform_attention_stages), (FFNExecutionSpec, transform_ffn_execution)):
        if isinstance(spec, cls):
            return transform(compiled, spec)
    raise InvalidInput("unsupported typed execution transform")
