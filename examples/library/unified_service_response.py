"""One declared g for CPU/GPU competition and residual recovery.

Runs with the installed library, without optional profiling or native programs.
All numbers describe an illustrative scenario, not measured hardware. The
normalized reservoir represents a declared interference proxy, not a GPU FIFO.
"""
from fractions import Fraction
import json

from llm_simulator.performance import (
    BackgroundReservoir, Graph, Parameter, ParameterBundle,
    ServiceInterference, ServiceRequestPoint, ServiceRequestTrajectory,
    ServiceResponseModel, ServiceResponseParameter, Task, WorkDemand,
    build_service_response,
)


def main():
    evidence = ("declared teaching scenario; no measured parameters",)
    parameters, competitors, links = [], [], []

    def setting(name, value, unit):
        parameters.append(ServiceResponseParameter(name, value, unit, "declared", evidence,
                                                   "illustrative normalized service response"))
        return name

    for competitor in ("cpu_background", "gpu_background"):
        competitors.append(BackgroundReservoir(competitor,
            setting(competitor + ":capacity", 1, "normalized_work"),
            setting(competitor + ":arrival", Fraction(1, 1000000), "normalized_work/ns"),
            setting(competitor + ":drain", Fraction(1, 2000000), "normalized_work/ns"),
            setting(competitor + ":activity", 1, "dimensionless"),
            setting(competitor + ":backlog", 1, "dimensionless")))
    for target, own in (("cpu_service", "cpu_background"), ("gpu_service", "gpu_background")):
        for competitor in ("cpu_background", "gpu_background"):
            coefficient = setting(target + ":" + competitor,
                                  Fraction(1, 2) if competitor == own else Fraction(1, 4),
                                  "dimensionless")
            links.append(ServiceInterference(target, competitor, coefficient))
    model = ServiceResponseModel("0" * 64, tuple(parameters), tuple(competitors), tuple(links),
        ("normalized residual interference is an assumed scenario", "same law for CPU and GPU",
         "cross-resource effects are explicitly declared, not inferred"))
    base = ParameterBundle((
        Parameter("cpu_service", 1000000000, "thread_cpu_ns/s", "declared", evidence, "one simulated thread"),
        Parameter("gpu_service", 1000000000, "flop/s", "declared", evidence, "simulated device"),
        Parameter("copy_service", 1000000000, "byte/s", "declared", evidence, "independent declared link"),
    ))
    requests = ServiceRequestTrajectory(
        {"cpu_background": 0, "gpu_background": 0},
        {"cpu_background": 0, "gpu_background": 0},
        (ServiceRequestPoint(2000000, {"cpu_background": 1}),
         ServiceRequestPoint(5000000, {"gpu_background": 1}),
         ServiceRequestPoint(8000000, {"cpu_background": 0}),
         ServiceRequestPoint(12000000, {"gpu_background": 0})),
        "predeclared_scenario", evidence, "scenario_ns", "example_start")
    service = build_service_response(model, base, requests, end_ns=40000000,
                                     max_step_ns=10000, max_points=20000)
    graph = Graph((
        Task("host_compute", "cpu", (WorkDemand("cpu_service", 3000000, "thread_cpu_ns"),)),
        Task("device_compute", "gpu", (WorkDemand("gpu_service", 6000000, "flop"),), ("host_compute",)),
        Task("copy_result", "copy", (WorkDemand("copy_service", 1000000, "byte"),), ("device_compute",)),
    ))
    result = service.evaluate(graph)
    recovery = [dict(at_ms=s.at_ns / 1e6, requests=dict(s.requests),
                     residual_work={k: float(v) for k, v in s.backlog.items()},
                     pressure={k: float(v) for k, v in s.pressure.items()})
                for s in service.samples if s.at_ns in {8000000, 9000000, 10000000, 12000000, 13000000, 14000000}]
    print(json.dumps(dict(scope="declared scenario; no hardware accuracy claim",
        duration_ms=result.duration_ns / 1e6,
        tasks=[dict(name=t.task_id, start_ms=t.start_ns / 1e6, end_ms=t.end_ns / 1e6) for t in result.tasks],
        recovery=recovery, accuracy_qualified=service.accuracy_qualified), indent=2))


if __name__ == "__main__":
    main()
