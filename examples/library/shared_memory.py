"""Run a declared shared-pool CPU/GPU model without saved profiles.

This example is an uncalibrated abstraction, not an Apple/Jetson/Spark profile.
Compute memory traffic still costs time. Same-buffer SHARE moves zero bytes;
real synchronization and concurrent shared-bandwidth competition are not
automatically inferred. An explicit COPY still needs an actual transfer quote.
"""
import json

from llm_simulator import AnalyticalCosts, Cache, Device, Hardware, Model, Scheduling, Simulator, Workload


def main():
    hardware = Hardware.shared_memory(
        cpu=Device(100e9, 20e9, launch_overhead_ns=500),
        gpu=Device(1e12, 100e9, launch_overhead_ns=2000),
        capacity_bytes=64 * 1024 * 1024,
    )
    simulator = Simulator(model=Model.dense(layers=2, hidden_size=16,
        intermediate_size=32, heads=4, kv_heads=2), hardware=hardware,
        workload=Workload(prefill_tokens=3, decode_steps=2),
        cost_model=AnalyticalCosts(workspace_bytes=0))
    scheduling = Scheduling(default_device="gpu", release="last_use",
                            cache=Cache(rule="lru", budget_bytes=1024 * 1024))
    with simulator.compile(scheduling=scheduling) as compiled:
        result = compiled.run()
        print(json.dumps(result.summary(), indent=2))
        if not result.success:
            raise RuntimeError("shared-memory example did not reach its goal")


if __name__ == "__main__":
    main()
