"""Typed preparation selectors and phase placement for the Python frontend.

These are strategy declarations, not another executor. The frontend lowers
them through the existing shared storage/resource ledger. A phase placement
uses the same cpu/gpu resources; it does not create independent P/D servers,
request queues, extra compute slots, or a network fabric.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .policy_ir import OPERATION_KINDS


_PREPARATION_FIELDS = frozenset(("layer_id", "stage", "tensor_ids", "expert_id",
                                  "destination", "trigger", "max_count"))


def _name(value, label, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
        raise InvalidInput(f"{label} must be a nonempty single-line string")
    return value


@dataclass(frozen=True)
class DataPreparation:
    """Prepare public model weights, selected by a conjunction of labels.

    ``startup`` is a blocking initialization phase, whose copies still charge
    time, services, and capacity. ``spare`` is opportunistic: it must not evict
    for speculation. Both use an exact public route and the normal global
    ledger. ``max_count`` bounds selected weight tensors, not inferred experts.

    Expert identifiers refer only to the static architecture catalogue. A
    declaration never predicts which experts a future router will select.
    """
    layer_id: str | None = None
    stage: str | None = None
    tensor_ids: tuple[str, ...] = ()
    expert_id: int | str | None = None
    destination: str = "vram"
    trigger: str = "spare"
    max_count: int | None = None

    def __post_init__(self):
        _name(self.layer_id, "preparation.layer_id", optional=True)
        _name(self.stage, "preparation.stage", optional=True)
        if not isinstance(self.tensor_ids, tuple):
            raise InvalidInput("preparation.tensor_ids must be an immutable tuple")
        for tensor_id in self.tensor_ids:
            _name(tensor_id, "preparation.tensor_ids entry")
        if len(set(self.tensor_ids)) != len(self.tensor_ids):
            raise InvalidInput("preparation.tensor_ids must be unique")
        if self.expert_id is not None and not (
                type(self.expert_id) is int and self.expert_id >= 0 or self.expert_id == "shared"):
            raise InvalidInput("preparation.expert_id must be a nonnegative integer, 'shared', or None")
        if self.layer_id is None and self.stage is None and not self.tensor_ids and self.expert_id is None:
            raise InvalidInput("preparation requires an explicit layer, stage, tensor, or expert selector")
        if not isinstance(self.destination, str) or self.destination not in {"dram", "vram"}:
            raise Unsupported("preparation.destination must be dram or vram")
        if not isinstance(self.trigger, str) or self.trigger not in {"startup", "spare"}:
            raise Unsupported("preparation.trigger must be startup or spare")
        if self.max_count is not None and (type(self.max_count) is not int or self.max_count < 1):
            raise InvalidInput("preparation.max_count must be a positive integer or None")

    def document(self):
        return dict(layer_id=self.layer_id, stage=self.stage, tensor_ids=list(self.tensor_ids),
                    expert_id=self.expert_id, destination=self.destination, trigger=self.trigger,
                    max_count=self.max_count)

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, dict) or set(document) != _PREPARATION_FIELDS:
            raise InvalidInput(f"data preparation requires exactly {sorted(_PREPARATION_FIELDS)}")
        if not isinstance(document["tensor_ids"], list):
            raise InvalidInput("serialized preparation.tensor_ids must be an array")
        return cls(**(document | {"tensor_ids": tuple(document["tensor_ids"])}))

    def matches_catalogue(self, row):
        """Match only a public (tensor_id, layer_id, stage, expert_id) row."""
        if not isinstance(row, tuple) or len(row) != 4:
            raise InvalidInput("preparation catalogue row must be a frozen four-tuple")
        tensor_id, layer_id, stage, expert_id = row
        for value, label in ((tensor_id, "tensor"), (layer_id, "layer")):
            _name(value, f"catalogue.{label}")
        if not isinstance(stage, str):
            raise InvalidInput("catalogue.stage must be a string, empty when not refined")
        return (self.layer_id in (None, layer_id) and self.stage in (None, stage) and
                (not self.tensor_ids or tensor_id in self.tensor_ids) and
                self.expert_id in (None, expert_id))

    def matching_tensor_ids(self, observation):
        """Resolve static public weight metadata without touching route history."""
        catalogue = getattr(observation, "weight_catalogue", None)
        if not isinstance(catalogue, tuple):
            raise Unsupported("data preparation requires an immutable public weight catalogue")
        identifiers = [row[0] for row in catalogue if self.matches_catalogue(row)]
        if len(identifiers) != len(set(identifiers)):
            raise InvalidInput("public preparation catalogue contains duplicate weight identities")
        if self.tensor_ids and set(self.tensor_ids) - set(identifiers):
            raise InvalidInput("preparation tensor selector is absent or contradicts public catalogue labels")
        if not identifiers:
            raise InvalidInput("preparation selector matches no public model weight")
        identifiers.sort()
        return tuple(identifiers if self.max_count is None else identifiers[:self.max_count])


def _placements(value, label):
    if not isinstance(value, Mapping) or not value:
        raise InvalidInput(f"{label} requires a nonempty task-kind/device mapping")
    copied = dict(value)
    for kind, device in copied.items():
        if not isinstance(kind, str) or kind not in OPERATION_KINDS:
            raise Unsupported(f"{label} contains unsupported task kind {kind!r}")
        if not isinstance(device, str) or device not in {"cpu", "gpu"}:
            raise Unsupported(f"{label} supports explicit cpu/gpu devices only")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class PhasePlacement:
    """Different P/D execution bindings on existing shared cpu/gpu resources.

    The original model owns KV identities, versions, and token dependencies.
    When a phase consumes KV in another domain, normal demand preparation
    issues charged copies. This helper never resets, clones, or fabricates KV.
    """
    prefill: Mapping[str, str]
    decode: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "prefill", _placements(self.prefill, "prefill"))
        object.__setattr__(self, "decode", _placements(self.decode, "decode"))

    def document(self):
        return dict(prefill=dict(self.prefill), decode=dict(self.decode),
                    resource_scope="shared_cpu_gpu_phase_placement")

    def apply(self, plan, *, priority=0):
        from .python_frontend import ModelPlan, Select
        if not isinstance(plan, ModelPlan):
            raise InvalidInput("PhasePlacement.apply requires a ModelPlan")
        if type(priority) is not int:
            raise InvalidInput("phase placement priority must be an integer")
        for phase, bindings in (("prefill", self.prefill), ("decode", self.decode)):
            for kind, device in bindings.items():
                plan.place(kind, device, select=Select(phase=phase), priority=priority)
        return plan
