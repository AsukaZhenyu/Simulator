"""Explicit immutable F32 matrix partitions, not aliases or quantized slices.

The manifest replaces a synthetic packed parent allocation with its complete
gate/up/down partition at the same initial locations. Its digest proves the
layout declaration, not the existence or numerical contents of real weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from ..tensor.spec import InvalidInput, Unsupported, ROLE_WEIGHT
from .expand import _tensor


@dataclass(frozen=True)
class SubweightMatrix:
    stage: str
    tensor_id: str
    dtype: str
    physical_shape: tuple[int, int]
    byte_offset: int
    byte_length: int
    version: int


@dataclass(frozen=True)
class ExpertPartition:
    parent_weight_id: str
    parent_version: int
    matrices: tuple[SubweightMatrix, ...]
    evidence_id: str
    evidence_sha256: str

    def for_stage(self, stage: str) -> SubweightMatrix:
        return next(item for item in self.matrices if item.stage == stage)


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise InvalidInput(f"{label} must have exactly {sorted(keys)}")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def partition_digest(parent_weight_id: str, parent_version: int, matrices: list[dict]) -> str:
    """Digest of declarations only; canonical order is gate, up, down."""
    by_stage = {item["stage"]: item for item in matrices}
    payload = dict(parent_weight_id=parent_weight_id, parent_version=parent_version,
                   matrices=[by_stage[stage] for stage in ("gate", "up", "down")])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def parse_subweights(value: Any) -> tuple[ExpertPartition, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidInput("subweights must be a nonempty explicit list")
    partitions, parents, identities = [], set(), set()
    for index, raw in enumerate(value):
        label = f"subweights[{index}]"
        obj = _exact(raw, {"parent_weight_id", "parent_version", "matrices", "source_evidence"}, label)
        parent = obj["parent_weight_id"]
        if not isinstance(parent, str) or not parent.startswith("w:") or parent in parents:
            raise InvalidInput(f"{label}.parent_weight_id must be a unique full weight identity")
        parents.add(parent)
        version = _integer(obj["parent_version"], f"{label}.parent_version", 1)
        if not isinstance(obj["matrices"], list) or len(obj["matrices"]) != 3:
            raise InvalidInput(f"{label}.matrices requires exactly gate, up, down")
        matrices, stages = [], set()
        for position, entry in enumerate(obj["matrices"]):
            ml = f"{label}.matrices[{position}]"
            item = _exact(entry, {"stage", "tensor_id", "dtype", "physical_shape", "byte_offset", "byte_length", "version"}, ml)
            stage = item["stage"]
            if not isinstance(stage, str) or stage not in {"gate", "up", "down"} or stage in stages:
                raise InvalidInput(f"{ml}.stage must be a unique gate/up/down stage")
            stages.add(stage)
            tid = item["tensor_id"]
            expected = f"{parent}:v{version}:{stage}"
            if tid != expected or tid in identities:
                raise InvalidInput(f"{ml}.tensor_id must be {expected!r} and globally unique")
            identities.add(tid)
            if item["dtype"] != "F32":
                raise Unsupported("subweights support contiguous F32 only; no quantized proportional slicing")
            shape = item["physical_shape"]
            if not isinstance(shape, list) or len(shape) != 2:
                raise InvalidInput(f"{ml}.physical_shape must be [input_dimension, output_dimension]")
            shape = tuple(_integer(x, f"{ml}.physical_shape", 1) for x in shape)
            offset = _integer(item["byte_offset"], f"{ml}.byte_offset")
            length = _integer(item["byte_length"], f"{ml}.byte_length", 4)
            if offset % 4 or length != 4 * shape[0] * shape[1]:
                raise InvalidInput(f"{ml}: F32 range must be aligned and match physical_shape")
            if _integer(item["version"], f"{ml}.version", 1) != version:
                raise InvalidInput(f"{ml}.version must equal immutable parent_version")
            matrices.append(SubweightMatrix(stage, tid, "F32", shape, offset, length, version))
        ordered = sorted(matrices, key=lambda entry: entry.byte_offset)
        end = 0
        for matrix in ordered:
            if matrix.byte_offset != end:
                raise InvalidInput(f"{label}: ranges must partition the parent without gaps or overlap")
            end += matrix.byte_length
        evidence = _exact(obj["source_evidence"], {"kind", "id", "sha256"}, f"{label}.source_evidence")
        if evidence["kind"] != "synthetic_layout_manifest":
            raise Unsupported("recorded weights require file-backed evidence verification; this version accepts synthetic_layout_manifest only")
        if not isinstance(evidence["id"], str) or not evidence["id"].strip():
            raise InvalidInput(f"{label}.source_evidence.id must be nonempty")
        digest = partition_digest(parent, version, obj["matrices"])
        if evidence["sha256"] != digest:
            raise InvalidInput(f"{label}.source_evidence.sha256 does not match the declared layout")
        partitions.append(ExpertPartition(parent, version,
                          tuple(sorted(matrices, key=lambda entry: ("gate", "up", "down").index(entry.stage))),
                          evidence["id"], digest))
    return tuple(partitions)


def subweights_to_document(partitions: tuple[ExpertPartition, ...]) -> list[dict]:
    if not isinstance(partitions, tuple) or any(not isinstance(part, ExpertPartition) for part in partitions):
        raise InvalidInput("subweights must be a tuple of parsed ExpertPartition values")
    result = []
    for part in partitions:
        if not isinstance(part.matrices, tuple) or any(not isinstance(matrix, SubweightMatrix) for matrix in part.matrices):
            raise InvalidInput("subweight matrices must be a tuple of SubweightMatrix values")
        matrices = [asdict(matrix) for matrix in part.matrices]
        for item in matrices:
            item["physical_shape"] = list(item["physical_shape"])
        result.append(dict(parent_weight_id=part.parent_weight_id, parent_version=part.parent_version,
                           matrices=matrices, source_evidence=dict(kind="synthetic_layout_manifest",
                           id=part.evidence_id, sha256=part.evidence_sha256)))
    return result


def materialize_subweights(compiled, partitions: tuple[ExpertPartition, ...], intermediate_bytes: int):
    """Return independent tensors and parent->stage mapping; remove parents.

    Parent placements are converted to partition placements at load time. This
    is a mutually exclusive initial representation, not a zero-cost runtime
    transformation of a previously loaded parent. Alignment is charged per child.
    """
    from .spec import weight_id
    expected = {weight_id(compiled.layer.id, name) for name in compiled.layer.ffn_params["expert_weights"]}
    shared = compiled.layer.ffn_params.get("shared_expert_weight")
    if shared is not None:
        expected.add(weight_id(compiled.layer.id, shared))
    by_parent = {part.parent_weight_id: part for part in partitions}
    if set(by_parent) != expected:
        raise InvalidInput("subweights must exactly cover every routed/shared expert, including inactive experts")
    sources = compiled.scenario.workload.tensor_by_id
    versions = {weight_id(compiled.layer.id, item.id): item.version for item in compiled.state_in.weight_copies}
    hidden, middle = compiled.layer.hidden_size, intermediate_bytes // 4
    if compiled.layer.hidden_bytes_per_token != hidden * 4:
        raise Unsupported("subweight F32 execution requires hidden_bytes_per_token == 4 * hidden_size")
    child_tensors, stage_ids = [], {}
    for parent, part in by_parent.items():
        source = sources[parent]
        if part.parent_version != versions[parent]:
            raise InvalidInput(f"subweight parent_version differs from state_in for {parent!r}")
        if sum(matrix.byte_length for matrix in part.matrices) != source.storage_bytes:
            raise InvalidInput(f"subweight ranges exceed or do not fully cover parent {parent!r}")
        stage_ids[parent] = {}
        for matrix in part.matrices:
            expected_shape = (middle, hidden) if matrix.stage == "down" else (hidden, middle)
            if matrix.physical_shape != expected_shape:
                raise InvalidInput(f"{matrix.tensor_id}: shape does not match layer hidden and declared intermediate width")
            if matrix.tensor_id in sources:
                raise InvalidInput(f"subweight identity collides with existing tensor {matrix.tensor_id!r}")
            entry = _tensor(matrix.tensor_id, ROLE_WEIGHT, matrix.byte_length, source.initial_locations)
            # GGML contiguous order: the first physical dimension has stride 4.
            first, second = matrix.physical_shape
            entry["ne"] = [first, second, 1, 1]
            entry["nb"] = [4, first * 4, matrix.byte_length, matrix.byte_length]
            child_tensors.append(entry)
            stage_ids[parent][matrix.stage] = matrix.tensor_id
    return child_tensors, stage_ids
