from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

from .model import ModelConfig


def max_window_weight_bytes(config: ModelConfig, window_size: int | None = None) -> int:
    weights = [layer.weight_bytes for layer in config.layers]
    k = min(window_size or config.policy.window_size, len(weights))
    return max(sum(weights[start : start + k]) for start in range(len(weights) - k + 1))


def state_bytes_by_layer(config: ModelConfig) -> tuple[int, ...]:
    if config.state is None:
        return tuple(0 for _ in config.layers)
    return tuple(
        config.state.bytes_for_layer_kind(layer.kind) for layer in config.layers
    )


def max_window_state_bytes(config: ModelConfig, window_size: int | None = None) -> int:
    states = state_bytes_by_layer(config)
    k = min(window_size or config.policy.window_size, len(states))
    return max(sum(states[start : start + k]) for start in range(len(states) - k + 1))


def required_dynamic_vram_bytes(
    config: ModelConfig, window_size: int | None = None
) -> int:
    k = min(window_size or config.policy.window_size, len(config.layers))
    weights = [layer.weight_bytes for layer in config.layers]
    states = state_bytes_by_layer(config)
    if config.policy.state_placement == "resident":
        return max_window_weight_bytes(config, k) + sum(states)
    footprints = [weight + state for weight, state in zip(weights, states)]
    return max(
        sum(footprints[start : start + k])
        for start in range(len(footprints) - k + 1)
    )


def kv_bytes_per_context_token(config: ModelConfig) -> int:
    if config.state is None:
        return 0
    attention_layers = sum(layer.kind == "attention" for layer in config.layers)
    return (
        attention_layers
        * config.state.sequence_count
        * config.state.attention_kv_head_count
        * (config.state.attention_key_length + config.state.attention_value_length)
        * config.state.kv_element_bytes
    )


def kv_storage_io_seconds(
    config: ModelConfig,
    attention_kv_bytes: int | None = None,
    window_size: int | None = None,
) -> float:
    """Return synchronous file-cache/storage service for one KV round trip.

    The patched scheduler writes an evicted attention layer's KV to its backing
    file and reads it again before the layer is reused. ``kv_storage_cache_bytes``
    is the one-way KV working-set threshold: bytes up to that threshold use the
    cached bandwidth and overflow uses the uncached bandwidth. A full layer
    window bypasses eviction and therefore has no KV storage I/O.
    """
    if config.policy.state_placement != "roundtrip":
        return 0.0
    k = min(window_size or config.policy.window_size, len(config.layers))
    if k >= len(config.layers):
        return 0.0
    if attention_kv_bytes is None:
        attention_kv_bytes = sum(
            state_bytes
            for layer, state_bytes in zip(
                config.layers, state_bytes_by_layer(config)
            )
            if layer.kind == "attention"
        )
    if attention_kv_bytes <= 0:
        return 0.0

    hardware = config.hardware
    cached_bandwidth = hardware.kv_storage_cached_bandwidth_bytes_per_s
    uncached_bandwidth = hardware.kv_storage_uncached_bandwidth_bytes_per_s
    if cached_bandwidth is None and uncached_bandwidth is None:
        return 0.0
    cached_bandwidth = cached_bandwidth or uncached_bandwidth
    uncached_bandwidth = uncached_bandwidth or cached_bandwidth
    assert cached_bandwidth is not None and uncached_bandwidth is not None

    cache_bytes = hardware.kv_storage_cache_bytes
    if cache_bytes <= 0:
        cached_bytes = 0
        uncached_bytes = attention_kv_bytes
    else:
        cached_bytes = min(attention_kv_bytes, cache_bytes)
        uncached_bytes = max(0, attention_kv_bytes - cache_bytes)
    # One write plus one read per decoded token.
    return 2.0 * (
        cached_bytes / cached_bandwidth
        + uncached_bytes / uncached_bandwidth
    )


@dataclass(frozen=True)
class AnalyticalResult:
    name: str
    layer_count: int
    window_size: int
    capacity_feasible: bool
    available_weight_bytes: int
    required_window_bytes: int
    required_weight_window_bytes: int
    required_state_window_bytes: int
    state_placement: str
    context_tokens: int
    total_recurrent_state_bytes: int
    total_attention_kv_bytes: int
    total_runtime_state_bytes: int
    kv_bytes_per_context_token: int
    resident_state_vram_bytes: int
    resident_attention_kv_bytes: int
    attention_kv_window_resident: bool
    recurrent_state_roundtrip_bytes_per_token: int
    attention_kv_roundtrip_bytes_per_token: int
    state_roundtrip_bytes_per_token: int
    attention_kv_storage_bytes_per_token: int
    state_latency_break_even_context_tokens: int | None
    max_resident_state_context_tokens: int | None
    total_layer_weight_bytes: int
    bytes_transferred_per_token: int
    total_flops_per_token: float
    layer_type_counts: dict[str, int]
    measured_compute_layer_count: int
    measured_transfer_layer_count: int
    arithmetic_intensity_flops_per_byte: float | None
    compute_seconds: float
    baseline_token_pipeline_overhead_seconds: float
    token_embedding_fallback_seconds: float
    baseline_token_state_roundtrip_seconds: float
    embedding_fallback_scale: float
    token_pipeline_control_seconds: float
    recurrent_state_transfer_seconds: float
    attention_kv_transfer_seconds: float
    state_transfer_seconds: float
    attention_kv_storage_io_seconds: float
    token_pipeline_overhead_seconds: float
    residual_token_pipeline_overhead_seconds: float
    layer_scheduler_overhead_seconds: float
    scheduler_overhead_seconds: float
    weight_stream_seconds: float
    host_staging_service_seconds: float
    staging_exposure_fraction: float
    exposed_host_staging_seconds: float
    transfer_seconds: float
    ideal_overlap_lower_bound_seconds: float
    no_overlap_upper_bound_seconds: float
    ideal_throughput_tokens_per_s: float
    no_overlap_throughput_tokens_per_s: float
    bottleneck: str
    static_offload_seconds: float | None
    static_offload_throughput_tokens_per_s: float | None
    bandwidth_to_match_static_bytes_per_s: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _static_offload_time(config: ModelConfig) -> float | None:
    hardware = config.hardware
    if hardware.cpu_effective_flops is None:
        return None
    gpu_layers = config.policy.static_gpu_layers
    split = len(config.layers) - gpu_layers
    cpu_flops = sum(layer.flops for layer in config.layers[:split])
    time_s = cpu_flops / hardware.cpu_effective_flops
    time_s += sum(layer.compute_seconds(hardware) for layer in config.layers[split:])
    if 0 < gpu_layers < len(config.layers) and hardware.activation_transfer_bytes > 0:
        time_s += hardware.h2d_latency_s
        time_s += hardware.activation_transfer_bytes / hardware.h2d_bandwidth_bytes_per_s
    return time_s


def staging_exposure_fraction(config: ModelConfig, window_size: int) -> float:
    """Fraction of pageable-to-pinned staging exposed by a short layer window.

    The implementation keeps a layer for one extra transition, so useful
    prefetch distance is K-2. `staging_overlap_window` is the empirically
    sufficient window; K=2 exposes all host staging, while larger windows
    linearly approach full overlap. The default threshold of two disables the
    correction for uncalibrated configs.
    """
    if config.hardware.host_staging_bandwidth_bytes_per_s is None:
        return 0.0
    threshold = config.policy.staging_overlap_window
    if threshold <= 2 or window_size >= threshold:
        return 0.0
    return max(0.0, min(1.0, (threshold - window_size) / (threshold - 2)))


def analyze_decode(config: ModelConfig) -> AnalyticalResult:
    hardware = config.hardware
    layer_count = len(config.layers)
    window_size = min(config.policy.window_size, layer_count)
    total_weight = sum(layer.weight_bytes for layer in config.layers)
    total_flops = sum(layer.flops for layer in config.layers)
    fully_resident = window_size >= layer_count
    required_window = max_window_weight_bytes(config, window_size)
    required_weight_window = required_window
    required_state_window = max_window_state_bytes(config, window_size)
    required_window = required_dynamic_vram_bytes(config, window_size)
    feasible = required_window <= hardware.available_weight_bytes

    states = state_bytes_by_layer(config)
    total_state = sum(states)
    total_recurrent_state = sum(
        state_bytes
        for layer, state_bytes in zip(config.layers, states)
        if layer.kind == "ssm"
    )
    total_attention_kv = sum(
        state_bytes
        for layer, state_bytes in zip(config.layers, states)
        if layer.kind == "attention"
    )
    kv_bytes_per_token = kv_bytes_per_context_token(config)
    context_tokens = config.state.context_tokens if config.state is not None else 0
    roundtrip_state = config.policy.state_placement == "roundtrip"
    recurrent_roundtrip_bytes = 2 * total_recurrent_state if roundtrip_state else 0
    attention_window_resident = not roundtrip_state or fully_resident
    attention_roundtrip_bytes = (
        0 if attention_window_resident else 2 * total_attention_kv
    )
    state_roundtrip_bytes = recurrent_roundtrip_bytes + attention_roundtrip_bytes
    state_bandwidth = (
        hardware.state_transfer_bandwidth_bytes_per_s
        or hardware.h2d_bandwidth_bytes_per_s
    )
    recurrent_state_transfer_s = recurrent_roundtrip_bytes / state_bandwidth
    attention_kv_transfer_s = attention_roundtrip_bytes / state_bandwidth
    state_transfer_s = recurrent_state_transfer_s + attention_kv_transfer_s
    attention_kv_storage_bytes = attention_roundtrip_bytes
    attention_kv_storage_s = kv_storage_io_seconds(
        config,
        total_attention_kv,
        window_size,
    )

    # Steady-state assumption: if every layer fits, weights remain resident between tokens.
    # Otherwise, a sequential sliding window reloads every layer once per decoded token.
    transferred_bytes = 0 if fully_resident else total_weight
    load_count = 0 if fully_resident else layer_count
    baseline_token_pipeline_overhead_s = config.policy.token_pipeline_overhead_s
    token_embedding_fallback_s = config.policy.token_embedding_fallback_s
    baseline_token_state_roundtrip_s = config.policy.token_state_roundtrip_s
    token_pipeline_control_s = (
        config.policy.effective_token_control_overhead_s
    )
    residual_token_pipeline_overhead_s = (
        baseline_token_pipeline_overhead_s
        - token_embedding_fallback_s
        - baseline_token_state_roundtrip_s
    )
    # Recurrent state is part of the fixed token-control path. Attention KV
    # eviction is instead serialized with streamed layer transitions below.
    token_pipeline_overhead_s = (
        token_pipeline_control_s + recurrent_state_transfer_s
    )
    layer_scheduler_overhead_s = (
        layer_count * config.policy.layer_scheduler_overhead_s
    )
    scheduler_overhead_s = (
        token_pipeline_overhead_s + layer_scheduler_overhead_s
    )
    compute_s = sum(layer.compute_seconds(hardware) for layer in config.layers)
    compute_s += scheduler_overhead_s
    weight_stream_s = (
        sum(layer.transfer_seconds(hardware) for layer in config.layers)
        if transferred_bytes
        else 0.0
    )
    host_staging_s = (
        transferred_bytes / hardware.host_staging_bandwidth_bytes_per_s
        if transferred_bytes and hardware.host_staging_bandwidth_bytes_per_s
        else 0.0
    )
    staging_exposure = (
        staging_exposure_fraction(config, window_size)
        if transferred_bytes
        else 0.0
    )
    exposed_host_staging_s = host_staging_s * staging_exposure
    transfer_s = (
        weight_stream_s
        + exposed_host_staging_s
        + attention_kv_transfer_s
        + attention_kv_storage_s
    )
    base_control_s = (
        sum(layer.compute_seconds(hardware) for layer in config.layers)
        + token_pipeline_control_s
        + layer_scheduler_overhead_s
    )
    state_latency_break_even = None
    max_resident_state_context = None
    if config.state is not None and kv_bytes_per_token > 0:
        if roundtrip_state and not fully_resident:
            target_state_service = max(
                0.0,
                weight_stream_s + exposed_host_staging_s - base_control_s,
            )

            def state_service_at(context: int) -> float:
                kv_bytes = context * kv_bytes_per_token
                return (
                    recurrent_state_transfer_s
                    + 2 * kv_bytes / state_bandwidth
                    + kv_storage_io_seconds(config, kv_bytes, window_size)
                )

            low = 0
            high = 1
            while state_service_at(high) < target_state_service and high < 2**31:
                high *= 2
            while low < high:
                middle = (low + high) // 2
                if state_service_at(middle) >= target_state_service:
                    high = middle
                else:
                    low = middle + 1
            state_latency_break_even = low
        resident_state_budget = (
            hardware.available_weight_bytes - required_weight_window
        )
        max_resident_state_context = math.floor(
            (resident_state_budget - total_recurrent_state) / kv_bytes_per_token
        )
    lower = max(compute_s, transfer_s)
    upper = compute_s + transfer_s
    if transfer_s > compute_s * 1.05:
        bottleneck = "transfer"
    elif compute_s > transfer_s * 1.05:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    static_s = _static_offload_time(config)
    bandwidth_to_match = None
    has_measured_transfer = any(
        layer.measured_transfer_seconds is not None for layer in config.layers
    )
    if (
        static_s is not None
        and transferred_bytes > 0
        and compute_s <= static_s
        and not has_measured_transfer
    ):
        latency_budget = (
            static_s
            - load_count * hardware.h2d_latency_s
            - exposed_host_staging_s
        )
        if latency_budget > 0:
            bandwidth_to_match = transferred_bytes / latency_budget

    return AnalyticalResult(
        name=config.name,
        layer_count=layer_count,
        window_size=window_size,
        capacity_feasible=feasible,
        available_weight_bytes=hardware.available_weight_bytes,
        required_window_bytes=required_window,
        required_weight_window_bytes=required_weight_window,
        required_state_window_bytes=required_state_window,
        state_placement=config.policy.state_placement,
        context_tokens=context_tokens,
        total_recurrent_state_bytes=total_recurrent_state,
        total_attention_kv_bytes=total_attention_kv,
        total_runtime_state_bytes=total_state,
        kv_bytes_per_context_token=kv_bytes_per_token,
        resident_state_vram_bytes=(
            total_state
            if config.policy.state_placement == "resident"
            else (total_attention_kv if fully_resident else 0)
        ),
        resident_attention_kv_bytes=(
            total_attention_kv if attention_window_resident else 0
        ),
        attention_kv_window_resident=attention_window_resident,
        recurrent_state_roundtrip_bytes_per_token=recurrent_roundtrip_bytes,
        attention_kv_roundtrip_bytes_per_token=attention_roundtrip_bytes,
        state_roundtrip_bytes_per_token=state_roundtrip_bytes,
        attention_kv_storage_bytes_per_token=attention_kv_storage_bytes,
        state_latency_break_even_context_tokens=state_latency_break_even,
        max_resident_state_context_tokens=max_resident_state_context,
        total_layer_weight_bytes=total_weight,
        bytes_transferred_per_token=transferred_bytes,
        total_flops_per_token=total_flops,
        layer_type_counts=dict(Counter(layer.kind for layer in config.layers)),
        measured_compute_layer_count=sum(
            layer.measured_compute_seconds is not None for layer in config.layers
        ),
        measured_transfer_layer_count=sum(
            layer.measured_transfer_seconds is not None for layer in config.layers
        ),
        arithmetic_intensity_flops_per_byte=(
            total_flops / transferred_bytes if transferred_bytes else None
        ),
        compute_seconds=compute_s,
        baseline_token_pipeline_overhead_seconds=baseline_token_pipeline_overhead_s,
        token_embedding_fallback_seconds=token_embedding_fallback_s,
        baseline_token_state_roundtrip_seconds=baseline_token_state_roundtrip_s,
        embedding_fallback_scale=config.policy.embedding_fallback_scale,
        token_pipeline_control_seconds=token_pipeline_control_s,
        recurrent_state_transfer_seconds=recurrent_state_transfer_s,
        attention_kv_transfer_seconds=attention_kv_transfer_s,
        state_transfer_seconds=state_transfer_s,
        attention_kv_storage_io_seconds=attention_kv_storage_s,
        token_pipeline_overhead_seconds=token_pipeline_overhead_s,
        residual_token_pipeline_overhead_seconds=residual_token_pipeline_overhead_s,
        layer_scheduler_overhead_seconds=layer_scheduler_overhead_s,
        scheduler_overhead_seconds=scheduler_overhead_s,
        weight_stream_seconds=weight_stream_s,
        host_staging_service_seconds=host_staging_s,
        staging_exposure_fraction=staging_exposure,
        exposed_host_staging_seconds=exposed_host_staging_s,
        transfer_seconds=transfer_s,
        ideal_overlap_lower_bound_seconds=lower,
        no_overlap_upper_bound_seconds=upper,
        ideal_throughput_tokens_per_s=1.0 / lower,
        no_overlap_throughput_tokens_per_s=1.0 / upper,
        bottleneck=bottleneck,
        static_offload_seconds=static_s,
        static_offload_throughput_tokens_per_s=(1.0 / static_s if static_s else None),
        bandwidth_to_match_static_bytes_per_s=bandwidth_to_match,
    )
