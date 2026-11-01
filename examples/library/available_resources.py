"""External available service drives one CPU/GPU/copy execution time axis.

Run after installing the library. Every value is a declared teaching scenario;
no profiles, native programs, GPU or operating-system simulator are required.
"""
from fractions import Fraction
import json

from llm_simulator.performance import (
    AvailableResources, Graph, MemoryDomain, MemoryReservation, Parameter,
    ParameterBundle, ResourcePoint, ResourceTimeline, ServiceBinding, Task,
    WorkDemand, evaluate_available_resources,
)


def main():
    scope = "declared teaching scenario, not measured hardware"
    evidence = (scope,)

    def parameter(name, value, unit):
        return Parameter(name, value, unit, "declared", evidence, scope)

    supply = ParameterBundle((
        parameter("host_available", 1_000_000_000, "thread_cpu_ns/s"),
        parameter("gpu_available", 1_000_000_000, "flop/s"),
        parameter("link_available", 1_000_000_000, "byte/s"),
        parameter("vram_budget", 8 << 20, "byte"),
    ))
    trajectory = ResourceTimeline((
        ResourcePoint(1_000_000, rates={"host_available": 500_000_000}),
        ResourcePoint(3_000_000, rates={"gpu_available": 0}),
        ResourcePoint(4_000_000, rates={"host_available": 1_000_000_000}),
        ResourcePoint(6_000_000, capacities={"vram": 4 << 20}),
        ResourcePoint(7_000_000, rates={"gpu_available": 1_000_000_000}),
        ResourcePoint(14_000_000, rates={"link_available": 500_000_000}),
        ResourcePoint(16_000_000, capacities={"vram": 8 << 20}),
    ))
    available = AvailableResources(supply, trajectory, 30_000_000,
        "scenario_ns", "scenario_start", "declared_scenario", evidence, scope)
    bindings = tuple(ServiceBinding(target, source, Fraction(1), "declared", evidence, scope)
        for target, source in (("host_work", "host_available"),
                               ("gpu_work", "gpu_available"),
                               ("copy_work", "link_available")))
    graph = Graph((
        Task("submit", "host", (WorkDemand("host_work", 2_000_000, "thread_cpu_ns"),)),
        Task("gpu_compute", "gpu", (WorkDemand("gpu_work", 6_000_000, "flop"),),
             ("submit",), memory=(MemoryReservation("vram", 2 << 20),)),
        Task("host_overlap", "host", (WorkDemand("host_work", 3_000_000, "thread_cpu_ns"),),
             ("submit",)),
        Task("wait_device", "blocked_host", depends_on=("submit",), waits_for=("gpu_compute",)),
        Task("copy", "link", (WorkDemand("copy_work", 3_000_000, "byte"),), ("wait_device",)),
        Task("resume", "host", (WorkDemand("host_work", 500_000, "thread_cpu_ns"),),
             ("copy", "host_overlap")),
    ), (MemoryDomain("vram", "vram_budget", 1 << 20, 0),))
    result = evaluate_available_resources(graph, ParameterBundle(()), available, bindings)
    constant = AvailableResources(supply, ResourceTimeline(), 30_000_000,
        "scenario_ns", "scenario_start", "declared_scenario", evidence, scope)
    control = evaluate_available_resources(graph, ParameterBundle(()), constant, bindings)
    print(json.dumps(dict(
        scope=scope, os_scheduler_simulated=False, hardware_accuracy_claimed=False,
        constant_duration_ns=control.duration_ns, dynamic_duration_ns=result.duration_ns,
        tasks=[t.to_dict() for t in result.tasks],
        timeline_origin_ns=result.start_ns,
        available_memory_semantics="budget already excludes background allocations",
        prediction_provider="llm_infer_model.performance.evaluate",
        external_background_applied_again=False,
    ), indent=2))


if __name__ == "__main__":
    main()
