"""A declared scenario, runnable without profiles, native code or a GPU.

Ten ms of CPU work is interrupted by the union [3,9) ms, so it finishes at
16 ms. The independent GPU task still finishes at 12 ms. This demonstrates
model semantics, not a measured device or a prediction of future interrupts.
"""
import json

from llm_simulator.performance import (
    CpuInterruption, CpuInterruptionTimebase, CpuInterruptionTrace,
    Graph, Parameter, ParameterBundle, Task, ThreadCpuBinding, WorkDemand,
    compose_cpu_interruptions, evaluate,
)


def main():
    parameters = ParameterBundle((
        Parameter("host_service", 1_000_000_000, "thread_cpu_ns/s", "declared", (), "one simulated thread"),
        Parameter("gpu_service", 1_000_000_000, "flop/s", "declared", (), "independent declared GPU service"),
    ))
    timebase = CpuInterruptionTimebase("scenario_ns", "example-start", 1_000_000_000,
                                       "simulated-host:processor-group-0", (0,))
    interruptions = CpuInterruptionTrace(timebase, (
        CpuInterruption(0, 3_000_000, 7_000_000, "dpc", "predeclared_scenario", ("example:DPC",)),
        CpuInterruption(0, 5_000_000, 9_000_000, "isr", "predeclared_scenario", ("example:ISR",)),
    ))
    supply = compose_cpu_interruptions(parameters, (ThreadCpuBinding("host_service", 0),),
                                       interruptions, baseline_timebase=timebase)
    graph = Graph((
        Task("host-work", "host", (WorkDemand("host_service", 10_000_000, "thread_cpu_ns"),)),
        Task("device-work", "gpu", (WorkDemand("gpu_service", 12_000_000, "flop"),)),
    ))
    result = evaluate(graph, parameters, supply.timeline)
    print(json.dumps({"scenario": "declared; no hardware calibration",
                      "cpu_end_ms": next(t.end_ns for t in result.tasks if t.task_id == "host-work") / 1e6,
                      "gpu_end_ms": next(t.end_ns for t in result.tasks if t.task_id == "device-work") / 1e6,
                      "accuracy_qualified": supply.accuracy_qualified}, indent=2))


if __name__ == "__main__":
    main()
