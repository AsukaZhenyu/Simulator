"""Explicit diagnostic component response to independent reference workloads.

Reference-kernel activity rate and completed CPU reference response are workload
measurements, not neutral available hardware capacity. In particular a host-wall
residual is never relabeled as active CPU work. This candidate supplies an
explainable response hypothesis and delegates execution to the common engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .evaluator import evaluate
from .graph import Graph, Task
from .parameters import InvalidModel, Parameter, ParameterBundle, UnknownParameter, rational


@dataclass(frozen=True)
class ReferenceComponent:
    name: str
    base_duration_ns: int | Fraction | None
    reference: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidModel("component name required")
        if self.reference not in {"static", "cpu_reference", "gpu_reference"}:
            raise InvalidModel("unknown diagnostic reference kind")
        if self.base_duration_ns is not None:
            object.__setattr__(self, "base_duration_ns", rational(self.base_duration_ns, "component duration"))


def evaluate_reference_components(components, profile_references, current_references, *, evidence, scope):
    """Compute base_ns * profile_reference/current_reference, without clipping.

    Ratios can exceed one. They are not physical efficiencies. Missing rates
    leave the candidate unknown. Static terms need no reference. All components
    form a serial accounting graph; their synthetic order does not reconstruct
    the observed device/host interleaving or establish a physical DAG lowering.
    """
    if type(components) is not tuple or any(not isinstance(c, ReferenceComponent) for c in components):
        raise InvalidModel("tuple of ReferenceComponent required")
    if len({c.name for c in components}) != len(components):
        raise InvalidModel("duplicate reference component")
    if type(evidence) is not tuple or not evidence or not isinstance(scope, str) or not scope:
        raise InvalidModel("explicit diagnostic evidence and scope required")
    parameters, tasks, previous, rows = [], [], (), []
    for component in components:
        if component.base_duration_ns is None:
            raise UnknownParameter("unknown profile component: " + component.name)
        ratio = Fraction(1)
        if component.reference != "static":
            before = profile_references.get(component.reference)
            current = current_references.get(component.reference)
            if before is None or current is None:
                raise UnknownParameter("unknown reference measurement: " + component.reference)
            before = rational(before, "profile reference rate")
            current = rational(current, "current reference rate")
            if before <= 0 or current <= 0:
                raise UnknownParameter("nonpositive reference measurement: " + component.reference)
            ratio = before / current
        value = component.base_duration_ns * ratio
        parameters.append(Parameter(component.name, value, "ns", "profiled", evidence, scope))
        tasks.append(Task(component.name, "serial_component_accounting", depends_on=previous,
                          fixed_parameter=component.name))
        previous = (component.name,)
        rows.append(dict(name=component.name, reference=component.reference,
                         base_duration_ns=component.base_duration_ns, ratio=ratio, response_duration_ns=value))
    graph, bundle = Graph(tuple(tasks)), ParameterBundle(tuple(parameters))
    result = evaluate(graph, bundle)
    return dict(execution=result, graph=graph, parameters=bundle, components=rows,
        hardware_accuracy_qualified=False, physical_available_capacity_claimed=False,
        host_wall_residual_is_active_cpu_work=False,
        scope="Reference-conditioned inclusive component response hypothesis; diagnostic only")
