"""Small, fictional hardware examples; never identifies the local computer."""
from .analytical import AnalyticalCosts
from .config import Device, Hardware, Model, Workload
from .simulation import Simulator
from llm_infer_model.m2.moe_spec import RouteTraceEntry
from tensor_mapping.python_frontend import Cache, Scheduling


def make_demo(kind="dense", *, device="gpu"):
    if kind not in {"dense", "moe"}:
        raise ValueError("demo kind must be dense or moe")
    if device not in {"cpu", "gpu"}:
        raise ValueError("demo device must be cpu or gpu")
    # Explicit example numbers, not a built-in profile of any real hardware.
    hardware = Hardware(
        cpu=Device(compute_flops_per_second=100e9, memory_bandwidth_bytes_per_second=20e9,
                   launch_overhead_ns=500),
        gpu=Device(compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9,
                   launch_overhead_ns=2000),
        dram_capacity_bytes=64 * 1024 * 1024, vram_capacity_bytes=8 * 1024 * 1024,
        transfer_bandwidth_bytes_per_second=8e9,
        ssd_read_bandwidth_bytes_per_second=1e9,
        ssd_write_bandwidth_bytes_per_second=500e6,
        transfer_latency_ns=1000,
    )
    kwargs = dict(layers=2, hidden_size=16, intermediate_size=32, heads=4, kv_heads=2)
    model = Model.dense(**kwargs) if kind == "dense" else Model.moe(**kwargs, experts=4, top_k=2)
    routes = () if kind == "dense" else tuple(
        RouteTraceEntry("request0", f"layer{layer}", position,
                        (position % 4, (position + 1) % 4))
        for layer in range(2) for position in range(5)
    )
    workload = Workload(prefill_tokens=3, decode_steps=2, routes=routes,
                        route_source="explicit_demo_round_robin_not_measured" if routes else None)
    simulator = Simulator(model=model, hardware=hardware, workload=workload,
                          cost_model=AnalyticalCosts(workspace_bytes=0))
    scheduling = Scheduling(default_device=device, release="last_use",
                            cache=Cache(rule="lru", budget_bytes=1024 * 1024))
    return simulator, scheduling
