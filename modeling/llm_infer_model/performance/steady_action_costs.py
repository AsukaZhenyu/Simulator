"""Exact scoped steady completed-action prices, evaluated by the common Graph.

Profiling supplies the mean of a declared complete action. This module never
fits parameters, subtracts host wall from device time, or adds overlapping
device/API diagnostics to an inclusive price. These are response estimates,
not identified CPU/GPU physical work. A supplied invocation-rate timeline is
an explicit response scenario, not a utilization-to-throughput conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

from .evaluator import evaluate
from .graph import Graph, ResourceTimeline, Task, WorkDemand
from .parameters import (
    INSTANTANEOUS_RATE, InvalidModel, Parameter, ParameterBundle,
    UnknownParameter, encode_fraction, integer, nonempty, rational,
)


def _evidence(values, label):
    if type(values) is not tuple or not values:
        raise InvalidModel(label+" requires a nonempty tuple of source references")
    for value in values:
        nonempty(value,label)


def _optional_duration(value,label):
    return None if value is None else rational(value,label)


@dataclass(frozen=True)
class SteadyActionKey:
    action_signature: str
    backend: str
    shape_signature: str
    execution_scope: str

    def __post_init__(self):
        for field in ("action_signature","backend","shape_signature","execution_scope"):
            nonempty(getattr(self,field),field)

    def to_dict(self):
        return dict(action_signature=self.action_signature,backend=self.backend,
                    shape_signature=self.shape_signature,execution_scope=self.execution_scope)

    @property
    def parameter_name(self):
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        return "steady_completed_response:"+hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SteadyActionCost:
    key: SteadyActionKey
    mean_ns: int | Fraction | None
    evidence: tuple[str,...]
    sample_count: int
    process_ids: tuple[int,...]
    device_interval_ns: int | Fraction | None = None
    zero_duration_approximation: bool = False
    device_interval_scope: str | None = None
    device_interval_evidence: tuple[str,...] = ()

    def __post_init__(self):
        if not isinstance(self.key,SteadyActionKey):
            raise InvalidModel("explicit SteadyActionKey required")
        object.__setattr__(self,"mean_ns",_optional_duration(self.mean_ns,"complete action mean"))
        object.__setattr__(self,"device_interval_ns",_optional_duration(self.device_interval_ns,"device diagnostic interval"))
        _evidence(self.evidence,"complete action evidence")
        integer(self.sample_count,"profile sample count",1)
        if type(self.process_ids) is not tuple or not self.process_ids or len(set(self.process_ids))!=len(self.process_ids):
            raise InvalidModel("explicit distinct profile process identities required")
        for pid in self.process_ids:
            integer(pid,"profile process identity",1)
        if self.sample_count<len(self.process_ids):
            raise InvalidModel("profile samples cannot be fewer than source processes")
        if type(self.zero_duration_approximation) is not bool:
            raise InvalidModel("zero-duration approximation flag must be boolean")
        if self.mean_ns==0 and not self.zero_duration_approximation:
            raise InvalidModel("measured zero needs an explicit resolution-limited approximation")
        if self.zero_duration_approximation and self.mean_ns!=0:
            raise InvalidModel("zero-duration approximation requires an explicit measured zero")
        if self.device_interval_ns is not None:
            nonempty(self.device_interval_scope,"device diagnostic scope")
            _evidence(self.device_interval_evidence,"device diagnostic evidence")
        elif self.device_interval_scope is not None or self.device_interval_evidence:
            raise InvalidModel("device diagnostic metadata cannot establish an unknown interval")

    def parameter(self):
        rate=(None if self.mean_ns is None else INSTANTANEOUS_RATE if self.mean_ns==0
              else Fraction(1_000_000_000)/self.mean_ns)
        return Parameter(self.key.parameter_name,rate,"invocation/s","profiled",self.evidence,
            "inclusive complete steady action; "+self.key.execution_scope)

    def to_dict(self):
        return dict(key=self.key.to_dict(),mean_ns=None if self.mean_ns is None else encode_fraction(self.mean_ns),
            evidence=list(self.evidence),sample_count=self.sample_count,process_ids=list(self.process_ids),
            device_interval_ns=None if self.device_interval_ns is None else encode_fraction(self.device_interval_ns),
            device_interval_scope=self.device_interval_scope,device_interval_evidence=list(self.device_interval_evidence),
            device_interval_is_additional_latency=False,physical_work_identified=False,
            zero_duration_approximation=self.zero_duration_approximation)


@dataclass(frozen=True)
class SteadyActionTable:
    entries: tuple[SteadyActionCost,...]

    def __post_init__(self):
        if type(self.entries) is not tuple or not self.entries or any(not isinstance(c,SteadyActionCost) for c in self.entries):
            raise InvalidModel("nonempty SteadyActionCost tuple required")
        if len({c.key for c in self.entries})!=len(self.entries):
            raise InvalidModel("duplicate exact steady-action key")

    def lookup(self,key):
        if not isinstance(key,SteadyActionKey):
            raise InvalidModel("exact SteadyActionKey required; no fallback or interpolation")
        for cost in self.entries:
            if cost.key==key:
                return cost
        raise UnknownParameter("no exact complete-action profile for "+repr(key))

    def to_dict(self):
        return dict(schema_version="steady-completed-action-table-1",entries=[c.to_dict() for c in self.entries],
            unknown_interpolation=False,physical_components_added=False,hardware_accuracy_qualified=False)


@dataclass(frozen=True)
class CompiledSteadyActions:
    graph: Graph
    parameters: ParameterBundle
    keys: tuple[SteadyActionKey,...]

    def evaluate(self,timeline: ResourceTimeline | None = None):
        return evaluate(self.graph,self.parameters,timeline)

    def to_dict(self):
        return dict(schema_version="compiled-steady-completed-actions-1",graph=self.graph.to_dict(),
            parameters=self.parameters.to_dict(),keys=[k.to_dict() for k in self.keys],
            cost_boundary="inclusive steady completed response",device_diagnostics_added=False,
            dynamic_available_resource_function_inferred=False,hardware_accuracy_qualified=False)


def compile_steady_actions(keys: tuple[SteadyActionKey,...],table: SteadyActionTable):
    """Compile a declared serial composition; costs never learn from its result.

    Repeated keys reuse frozen parameters, but remain distinct invocations.
    The Graph alone rounds individual completion events to integer nanoseconds.
    Mean batch latency and per-call variation are independent validation metrics
    supplied by profiling; this model does not fit either heldout statistic.
    """
    if type(keys) is not tuple or not keys or any(not isinstance(k,SteadyActionKey) for k in keys):
        raise InvalidModel("explicit nonempty sequence of exact action keys required")
    if not isinstance(table,SteadyActionTable):
        raise InvalidModel("SteadyActionTable required")
    parameters={};tasks=[];previous=()
    for index,key in enumerate(keys):
        cost=table.lookup(key)
        parameter=cost.parameter()
        parameters[parameter.name]=parameter
        task_id="steady_action:"+str(index)
        tasks.append(Task(task_id,"steady_response:"+key.backend,
            (WorkDemand(parameter.name,1,"invocation"),),previous))
        previous=(task_id,)
    return CompiledSteadyActions(Graph(tuple(tasks)),ParameterBundle(tuple(parameters.values())),keys)
