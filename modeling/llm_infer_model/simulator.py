from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass

from .analytical import analyze_decode, state_bytes_by_layer
from .model import ModelConfig


@dataclass(frozen=True)
class TraceEvent:
    task: str
    layer: int
    resource: str
    start_s: float
    end_s: float
    layer_name: str = ""
    layer_kind: str = "unknown"
    bytes: int = 0
    flops: float = 0.0
    scheduler_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    name: str
    window_size: int
    makespan_seconds: float
    throughput_tokens_per_s: float
    compute_service_seconds: float
    scheduler_service_seconds: float
    transfer_service_seconds: float
    state_transfer_service_seconds: float
    recurrent_state_transfer_service_seconds: float
    attention_kv_transfer_service_seconds: float
    kv_storage_service_seconds: float
    host_staging_service_seconds: float
    exposed_host_staging_seconds: float
    overlap_seconds: float
    gpu_utilization: float
    h2d_utilization: float
    peak_streamed_weight_bytes: int
    peak_runtime_state_bytes: int
    bytes_transferred: int
    state_roundtrip_bytes: int
    kv_storage_bytes: int
    events: tuple[TraceEvent, ...]

    def to_dict(self, include_events: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_events:
            result.pop("events")
        return result


def simulate_decode(config: ModelConfig) -> SimulationResult:
    analytical = analyze_decode(config)
    if not analytical.capacity_feasible:
        raise ValueError(
            "window does not fit available VRAM: "
            f"required={analytical.required_window_bytes}, "
            f"available={analytical.available_weight_bytes}"
        )

    layers = config.layers
    hardware = config.hardware
    layer_count = len(layers)
    window_size = min(config.policy.window_size, layer_count)
    state_by_layer = state_bytes_by_layer(config)
    state_transfer_service = analytical.state_transfer_seconds
    recurrent_state_transfer_service = (
        analytical.recurrent_state_transfer_seconds
    )
    attention_kv_transfer_service = analytical.attention_kv_transfer_seconds
    kv_storage_service = analytical.attention_kv_storage_io_seconds
    token_control_service = analytical.token_pipeline_control_seconds
    token_pipeline_overhead = (
        token_control_service + recurrent_state_transfer_service
    )
    scheduler_service = (
        token_pipeline_overhead
        + layer_count * config.policy.layer_scheduler_overhead_s
    )
    compute_service = sum(layer.compute_seconds(hardware) for layer in layers)
    compute_service += scheduler_service

    if window_size >= layer_count:
        events: list[TraceEvent] = []
        now = 0.0
        if state_transfer_service > 0:
            events.append(
                TraceEvent(
                    "STATE_ROUNDTRIP",
                    -1,
                    "CONTROL",
                    now,
                    now + state_transfer_service,
                    bytes=analytical.state_roundtrip_bytes_per_token,
                    scheduler_seconds=state_transfer_service,
                )
            )
            now += state_transfer_service
        if token_control_service > 0:
            events.append(
                TraceEvent(
                    "PIPELINE_CONTROL",
                    -1,
                    "GPU",
                    now,
                    now + token_control_service,
                    scheduler_seconds=token_control_service,
                )
            )
            now += token_control_service
        for layer_index, layer in enumerate(layers):
            duration = (
                layer.compute_seconds(hardware)
                + config.policy.layer_scheduler_overhead_s
            )
            events.append(
                TraceEvent(
                    "COMPUTE",
                    layer_index,
                    "GPU",
                    now,
                    now + duration,
                    layer_name=layer.name,
                    layer_kind=layer.kind,
                    flops=layer.flops,
                    scheduler_seconds=config.policy.layer_scheduler_overhead_s,
                )
            )
            now += duration
        return SimulationResult(
            name=config.name,
            window_size=window_size,
            makespan_seconds=now,
            throughput_tokens_per_s=1.0 / now,
            compute_service_seconds=compute_service,
            scheduler_service_seconds=scheduler_service,
            transfer_service_seconds=0.0,
            state_transfer_service_seconds=state_transfer_service,
            recurrent_state_transfer_service_seconds=(
                recurrent_state_transfer_service
            ),
            attention_kv_transfer_service_seconds=attention_kv_transfer_service,
            kv_storage_service_seconds=kv_storage_service,
            host_staging_service_seconds=0.0,
            exposed_host_staging_seconds=0.0,
            overlap_seconds=0.0,
            gpu_utilization=1.0,
            h2d_utilization=0.0,
            peak_streamed_weight_bytes=sum(layer.weight_bytes for layer in layers),
            peak_runtime_state_bytes=analytical.total_runtime_state_bytes,
            bytes_transferred=analytical.state_roundtrip_bytes_per_token,
            state_roundtrip_bytes=analytical.state_roundtrip_bytes_per_token,
            kv_storage_bytes=analytical.attention_kv_storage_bytes_per_token,
            events=tuple(events),
        )

    queue: list[tuple[float, int, str, int, float]] = []
    sequence = 0
    now = 0.0
    next_load = 0
    next_compute = 0
    h2d_busy = False
    gpu_busy = False
    loaded: set[int] = set()
    occupying: dict[int, tuple[int, int]] = {}
    current_weight_bytes = 0
    current_state_bytes = 0
    peak_weight_bytes = 0
    peak_state_bytes = analytical.resident_state_vram_bytes
    trace: list[TraceEvent] = []
    resident_state_bytes = analytical.resident_state_vram_bytes
    available_window_bytes = hardware.available_weight_bytes - resident_state_bytes

    def push_event(end_s: float, kind: str, layer_index: int, start_s: float) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(queue, (end_s, sequence, kind, layer_index, start_s))

    def try_launch() -> None:
        nonlocal next_load, h2d_busy, gpu_busy, current_weight_bytes
        nonlocal current_state_bytes, peak_weight_bytes, peak_state_bytes

        if not gpu_busy and next_compute < layer_count and next_compute in loaded:
            layer = layers[next_compute]
            duration = (
                layer.compute_seconds(hardware)
                + config.policy.layer_scheduler_overhead_s
            )
            gpu_busy = True
            trace.append(
                TraceEvent(
                    "COMPUTE",
                    next_compute,
                    "GPU",
                    now,
                    now + duration,
                    layer_name=layer.name,
                    layer_kind=layer.kind,
                    flops=layer.flops,
                    scheduler_seconds=config.policy.layer_scheduler_overhead_s,
                )
            )
            push_event(now + duration, "COMPUTE_DONE", next_compute, now)

        if not h2d_busy and next_load < layer_count and len(occupying) < window_size:
            layer = layers[next_load]
            layer_state_bytes = (
                state_by_layer[next_load]
                if config.policy.state_placement == "roundtrip"
                else 0
            )
            if (
                current_weight_bytes
                + current_state_bytes
                + layer.weight_bytes
                + layer_state_bytes
                <= available_window_bytes
            ):
                layer_index = next_load
                next_load += 1
                h2d_busy = True
                occupying[layer_index] = (layer.weight_bytes, layer_state_bytes)
                current_weight_bytes += layer.weight_bytes
                current_state_bytes += layer_state_bytes
                peak_weight_bytes = max(peak_weight_bytes, current_weight_bytes)
                peak_state_bytes = max(
                    peak_state_bytes,
                    resident_state_bytes + current_state_bytes,
                )
                duration = layer.transfer_seconds(hardware)
                trace.append(
                    TraceEvent(
                        "LOAD_WEIGHT",
                        layer_index,
                        "H2D",
                        now,
                        now + duration,
                        layer_name=layer.name,
                        layer_kind=layer.kind,
                        bytes=layer.weight_bytes,
                    )
                )
                push_event(now + duration, "LOAD_DONE", layer_index, now)

    control_cursor = 0.0
    if recurrent_state_transfer_service > 0:
        trace.append(
            TraceEvent(
                "STATE_ROUNDTRIP",
                -1,
                "CONTROL",
                control_cursor,
                control_cursor + recurrent_state_transfer_service,
                bytes=analytical.recurrent_state_roundtrip_bytes_per_token,
                scheduler_seconds=recurrent_state_transfer_service,
            )
        )
        control_cursor += recurrent_state_transfer_service
    if token_control_service > 0:
        # Nsight shows that the fixed pipeline path is interleaved with streamed
        # weights rather than paid as a setup barrier. Represent it as a parallel
        # control lane and take the maximum service time after the DES finishes.
        trace.append(
            TraceEvent(
                "PIPELINE_CONTROL",
                -1,
                "CONTROL",
                control_cursor,
                control_cursor + token_control_service,
                scheduler_seconds=token_control_service,
            )
        )

    try_launch()
    while next_compute < layer_count or queue:
        if not queue:
            raise RuntimeError("simulation deadlock: no runnable event")
        now, _, kind, layer_index, _ = heapq.heappop(queue)
        if kind == "LOAD_DONE":
            h2d_busy = False
            loaded.add(layer_index)
        elif kind == "COMPUTE_DONE":
            gpu_busy = False
            loaded.remove(layer_index)
            layer_weight_bytes, layer_state_bytes = occupying.pop(layer_index)
            current_weight_bytes -= layer_weight_bytes
            current_state_bytes -= layer_state_bytes
            if layer_index != next_compute:
                raise RuntimeError("out-of-order GPU completion")
            next_compute += 1
        else:
            raise RuntimeError(f"unknown event kind: {kind}")
        try_launch()

    weight_transfer_service = sum(
        event.end_s - event.start_s for event in trace if event.resource == "H2D"
    )
    exposed_host_staging = analytical.exposed_host_staging_seconds
    if exposed_host_staging > 0:
        trace.append(
            TraceEvent(
                "HOST_STAGING_EXPOSED",
                -1,
                "HOST",
                now,
                now + exposed_host_staging,
                bytes=analytical.bytes_transferred_per_token,
            )
        )
    exposed_cursor = now + exposed_host_staging
    if attention_kv_transfer_service > 0:
        trace.append(
            TraceEvent(
                "ATTENTION_KV_PCIE_EXPOSED",
                -1,
                "STATE_IO",
                exposed_cursor,
                exposed_cursor + attention_kv_transfer_service,
                bytes=analytical.attention_kv_roundtrip_bytes_per_token,
            )
        )
        exposed_cursor += attention_kv_transfer_service
    if kv_storage_service > 0:
        trace.append(
            TraceEvent(
                "KV_STORAGE_IO_EXPOSED",
                -1,
                "STORAGE",
                exposed_cursor,
                exposed_cursor + kv_storage_service,
                bytes=analytical.attention_kv_storage_bytes_per_token,
            )
        )
        exposed_cursor += kv_storage_service
    transfer_service = (
        weight_transfer_service
        + attention_kv_transfer_service
        + kv_storage_service
    )
    makespan = max(exposed_cursor, compute_service)
    overlap = (
        compute_service
        + transfer_service
        + exposed_host_staging
        - makespan
    )
    return SimulationResult(
        name=config.name,
        window_size=window_size,
        makespan_seconds=makespan,
        throughput_tokens_per_s=1.0 / makespan,
        compute_service_seconds=compute_service,
        scheduler_service_seconds=scheduler_service,
        transfer_service_seconds=transfer_service,
        state_transfer_service_seconds=state_transfer_service,
        recurrent_state_transfer_service_seconds=recurrent_state_transfer_service,
        attention_kv_transfer_service_seconds=attention_kv_transfer_service,
        kv_storage_service_seconds=kv_storage_service,
        host_staging_service_seconds=analytical.host_staging_service_seconds,
        exposed_host_staging_seconds=exposed_host_staging,
        overlap_seconds=max(0.0, overlap),
        gpu_utilization=compute_service / makespan,
        h2d_utilization=(
            weight_transfer_service + attention_kv_transfer_service
        ) / makespan,
        peak_streamed_weight_bytes=peak_weight_bytes,
        peak_runtime_state_bytes=peak_state_bytes,
        bytes_transferred=(
            sum(layer.weight_bytes for layer in layers)
            + analytical.state_roundtrip_bytes_per_token
        ),
        state_roundtrip_bytes=analytical.state_roundtrip_bytes_per_token,
        kv_storage_bytes=analytical.attention_kv_storage_bytes_per_token,
        events=tuple(trace),
    )
