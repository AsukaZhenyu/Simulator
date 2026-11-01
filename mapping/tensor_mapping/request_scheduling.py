"""Finite single-CPU/GPU request scheduling choices, not another executor.

Request arrivals, token dependencies and KV ownership belong to modeling.
These declarations select a public ready-work order and an explicit P/D
handoff destination on the existing shared memory/resource ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from llm_infer_model.tensor.spec import InvalidInput, Unsupported


@dataclass(frozen=True)
class RequestPipeline:
    ready_rule: str = "graph_order"
    batch_accounting: str = "independent_request_kernels"

    def __post_init__(self):
        if not isinstance(self.ready_rule,str) or self.ready_rule not in {"graph_order", "decode_first", "round_robin"}:
            raise Unsupported("request ready_rule must be graph_order, decode_first or round_robin")
        if self.batch_accounting != "independent_request_kernels":
            raise Unsupported("request pipeline supports explicitly priced independent request kernels; fused batch execution is unsupported")

    def document(self):
        return dict(format_version="request-pipeline-0.1", ready_rule=self.ready_rule,
                    batch_accounting=self.batch_accounting,
                    resource_scope="shared_cpu_gpu")

    @classmethod
    def from_document(cls, document):
        fields={"format_version", "ready_rule", "batch_accounting", "resource_scope"}
        if not isinstance(document, dict) or set(document)!=fields:
            raise InvalidInput("request pipeline requires its exact versioned fields")
        if document["format_version"]!="request-pipeline-0.1" or document["resource_scope"]!="shared_cpu_gpu":
            raise Unsupported("unsupported request pipeline format or resource scope")
        return cls(document["ready_rule"], document["batch_accounting"])


@dataclass(frozen=True)
class PhaseHandoff:
    """Join final Prefill hidden state and all layers' KV before first Decode.

    destination is a storage domain, independent of an executor's identity.
    Empty request_ids explicitly selects all requests in this finite model.
    PhasePlacement separately chooses computation devices; the compiler checks
    that the first Decode Attention reads from this declared destination.
    """
    destination: str
    request_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.destination,str) or self.destination not in {"dram", "vram"}:
            raise Unsupported("phase handoff destination must be dram or vram")
        if not isinstance(self.request_ids, tuple):
            raise InvalidInput("handoff request_ids must be an immutable tuple")
        if any(not isinstance(r,str) or not r or any(c in r for c in ":\r\n") for r in self.request_ids):
            raise InvalidInput("handoff request identities must be nonempty without colon/newline")
        if len(set(self.request_ids))!=len(self.request_ids):
            raise InvalidInput("handoff request_ids must be unique")

    def document(self):
        return dict(format_version="phase-handoff-0.1", destination=self.destination,
                    request_ids=list(self.request_ids),
                    boundary="prefill_final_hidden_and_all_layer_kv")

    @classmethod
    def from_document(cls, document):
        fields={"format_version", "destination", "request_ids", "boundary"}
        if not isinstance(document,dict) or set(document)!=fields:
            raise InvalidInput("phase handoff requires its exact versioned fields")
        if document["format_version"]!="phase-handoff-0.1" or document["boundary"]!="prefill_final_hidden_and_all_layer_kv":
            raise Unsupported("unsupported phase handoff format or boundary")
        if not isinstance(document["request_ids"],list):
            raise InvalidInput("serialized handoff request_ids must be a list")
        return cls(document["destination"],tuple(document["request_ids"]))


def normalize_handoffs(rules, request_ids):
    """Expand static selectors, rejecting unknown or multiply owned requests."""
    if not isinstance(rules,tuple) or any(not isinstance(rule,PhaseHandoff) for rule in rules):
        raise InvalidInput("phase_handoffs must be an immutable tuple of PhaseHandoff")
    result={}
    for rule in rules:
        rule.__post_init__()
        selected=rule.request_ids or tuple(request_ids)
        for request in selected:
            if request not in request_ids:
                raise InvalidInput(f"handoff references unknown request {request!r}")
            if request in result:
                raise InvalidInput("overlapping phase handoffs for one request")
            result[request]=rule.destination
    return tuple(sorted(result.items()))


def order_request_operations(operations, spec, *, last_request=None, priority=None):
    """Reorder only already public work; no private arrivals or graph lookup.

    `priority` is a public critical-path rank computed by the policy. It only
    breaks ties inside a class the declared rule already selected: a ready
    rule fixes which requests/phases go first, not the order of independent
    public work inside one class. Equal ranks keep the previous stable order,
    so work the declared rule did not distinguish is never reordered.
    """
    if spec is None:
        return tuple(operations)
    if not isinstance(spec,RequestPipeline):
        raise InvalidInput("request ordering requires RequestPipeline")
    if priority is not None and (not isinstance(priority,Mapping)
            or any(type(value) is not int or value<0 for value in priority.values())):
        raise InvalidInput("public request priority must be nonnegative integer ranks per operation id")
    if spec.ready_rule=="graph_order":
        return tuple(operations)
    position={operation.id:index for index,operation in enumerate(operations)}
    def rank(op):
        return (-priority.get(op.id,0),position[op.id]) if priority is not None else (0,"")
    if spec.ready_rule=="decode_first":
        # A ready gate with no inputs costs one declared unit and unlocks a
        # whole request, so it is served inside its class before longer work.
        return tuple(sorted(operations,key=lambda op:(op.phase!="decode",bool(op.inputs),*rank(op))))
    requests=sorted({op.request_id for op in operations if op.request_id})
    if last_request in requests:
        pivot=requests.index(last_request)+1
        requests=requests[pivot:]+requests[:pivot]
    rank={request:index for index,request in enumerate(requests)}
    return tuple(sorted(operations,key=lambda op:rank.get(op.request_id,len(rank))))
