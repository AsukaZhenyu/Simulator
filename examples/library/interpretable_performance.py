"""Declared-parameter example: modeling computes times, without any profile.

These tiny integers are hand-computable demonstration inputs, not measurements.
Run from an installed library or a checkout with its runtime packages available.
"""
from fractions import Fraction
import json

from llm_simulator.performance import (
    Graph, Parameter, ParameterBundle, ResourcePoint, ResourceTimeline,
    Task, WorkDemand, evaluate,
)


def main():
    bundle = ParameterBundle(tuple(
        Parameter(name, value, unit, "declared", (), "synthetic hand calculation")
        for name, value, unit in (
            ("host.submit", 2, "ns"),
            ("pcie.bandwidth", 1_000_000_000, "byte/s"),
            ("gpu.compute", 1_000_000_000, "flop/s"),
            ("gpu.memory", 1_000_000_000, "byte/s"),
            ("cpu.compute", 1_000_000_000, "flop/s"),
        )
    ))
    graph = Graph((
        Task("submit", "host", fixed_parameter="host.submit"),
        Task("copy", "pcie", (WorkDemand("pcie.bandwidth", 4, "byte"),), ("submit",)),
        Task("gpu", "gpu", (WorkDemand("gpu.compute", 10, "flop"),
                             WorkDemand("gpu.memory", 6, "byte")), ("copy",)),
        Task("cpu", "cpu", (WorkDemand("cpu.compute", 5, "flop"),)),
    ))
    constant = evaluate(graph, bundle)
    dynamic = evaluate(graph, bundle, ResourceTimeline((
        ResourcePoint(9, rates={"gpu.compute": Fraction(500_000_000)}),
    )))
    assert constant.duration_ns == 16
    assert dynamic.duration_ns == 23
    print(json.dumps({
        "parameter_source": "declared_synthetic_not_hardware_calibrated",
        "prediction_provider": "llm_infer_model.performance.evaluate",
        "constant": constant.to_dict(), "dynamic": dynamic.to_dict(),
        "physical_capacity_claimed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
