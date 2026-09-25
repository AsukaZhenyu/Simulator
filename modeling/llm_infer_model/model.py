from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerSpec:
    name: str
    weight_bytes: int
    flops: float
    kind: str = "unknown"
    measured_compute_seconds: float | None = None
    measured_transfer_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.weight_bytes <= 0:
            raise ValueError(f"{self.name}: weight_bytes must be positive")
        if self.flops <= 0:
            raise ValueError(f"{self.name}: flops must be positive")
        if self.measured_compute_seconds is not None and self.measured_compute_seconds <= 0:
            raise ValueError(f"{self.name}: measured_compute_seconds must be positive")
        if self.measured_transfer_seconds is not None and self.measured_transfer_seconds <= 0:
            raise ValueError(f"{self.name}: measured_transfer_seconds must be positive")

    def compute_seconds(self, hardware: "HardwareSpec") -> float:
        if self.measured_compute_seconds is not None:
            return self.measured_compute_seconds
        return self.flops / hardware.gpu_effective_flops

    def transfer_seconds(self, hardware: "HardwareSpec") -> float:
        if self.measured_transfer_seconds is not None:
            return self.measured_transfer_seconds
        return hardware.h2d_latency_s + self.weight_bytes / hardware.h2d_bandwidth_bytes_per_s


@dataclass(frozen=True)
class HardwareSpec:
    gpu_effective_flops: float
    h2d_bandwidth_bytes_per_s: float
    vram_capacity_bytes: int
    h2d_latency_s: float = 0.0
    cpu_effective_flops: float | None = None
    global_bytes: int = 0
    kv_bytes: int = 0
    workspace_bytes: int = 0
    activation_transfer_bytes: int = 0
    host_staging_bandwidth_bytes_per_s: float | None = None
    state_transfer_bandwidth_bytes_per_s: float | None = None
    kv_storage_cached_bandwidth_bytes_per_s: float | None = None
    kv_storage_uncached_bandwidth_bytes_per_s: float | None = None
    kv_storage_cache_bytes: int = 0

    def __post_init__(self) -> None:
        if self.gpu_effective_flops <= 0:
            raise ValueError("gpu_effective_flops must be positive")
        if self.h2d_bandwidth_bytes_per_s <= 0:
            raise ValueError("h2d_bandwidth_bytes_per_s must be positive")
        if self.vram_capacity_bytes <= 0:
            raise ValueError("vram_capacity_bytes must be positive")
        if self.h2d_latency_s < 0:
            raise ValueError("h2d_latency_s cannot be negative")
        if self.cpu_effective_flops is not None and self.cpu_effective_flops <= 0:
            raise ValueError("cpu_effective_flops must be positive when specified")
        if (
            self.host_staging_bandwidth_bytes_per_s is not None
            and self.host_staging_bandwidth_bytes_per_s <= 0
        ):
            raise ValueError(
                "host_staging_bandwidth_bytes_per_s must be positive when specified"
            )
        if (
            self.state_transfer_bandwidth_bytes_per_s is not None
            and self.state_transfer_bandwidth_bytes_per_s <= 0
        ):
            raise ValueError(
                "state_transfer_bandwidth_bytes_per_s must be positive when specified"
            )
        for field_name in (
            "kv_storage_cached_bandwidth_bytes_per_s",
            "kv_storage_uncached_bandwidth_bytes_per_s",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when specified")
        if self.kv_storage_cache_bytes < 0:
            raise ValueError("kv_storage_cache_bytes cannot be negative")
        for field_name in (
            "global_bytes",
            "kv_bytes",
            "workspace_bytes",
            "activation_transfer_bytes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")

    @property
    def permanent_vram_bytes(self) -> int:
        return self.global_bytes + self.kv_bytes + self.workspace_bytes

    @property
    def available_weight_bytes(self) -> int:
        return self.vram_capacity_bytes - self.permanent_vram_bytes


@dataclass(frozen=True)
class PolicySpec:
    window_size: int
    static_gpu_layers: int = 0
    token_pipeline_overhead_s: float = 0.0
    token_embedding_fallback_s: float = 0.0
    embedding_fallback_scale: float = 1.0
    staging_overlap_window: int = 2
    token_state_roundtrip_s: float = 0.0
    state_placement: str = "roundtrip"
    layer_scheduler_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.static_gpu_layers < 0:
            raise ValueError("static_gpu_layers cannot be negative")
        if self.token_pipeline_overhead_s < 0:
            raise ValueError("token_pipeline_overhead_s cannot be negative")
        if self.token_embedding_fallback_s < 0:
            raise ValueError("token_embedding_fallback_s cannot be negative")
        if self.token_embedding_fallback_s > self.token_pipeline_overhead_s:
            raise ValueError(
                "token_embedding_fallback_s cannot exceed token_pipeline_overhead_s"
            )
        if self.token_state_roundtrip_s < 0:
            raise ValueError("token_state_roundtrip_s cannot be negative")
        if (
            self.token_embedding_fallback_s + self.token_state_roundtrip_s
            > self.token_pipeline_overhead_s
        ):
            raise ValueError(
                "embedding and state components cannot exceed token_pipeline_overhead_s"
            )
        if not 0.0 <= self.embedding_fallback_scale <= 1.0:
            raise ValueError("embedding_fallback_scale must be between zero and one")
        if self.staging_overlap_window < 2:
            raise ValueError("staging_overlap_window must be at least two")
        if self.state_placement not in {"roundtrip", "resident"}:
            raise ValueError("state_placement must be roundtrip or resident")
        if self.layer_scheduler_overhead_s < 0:
            raise ValueError("layer_scheduler_overhead_s cannot be negative")

    @property
    def effective_token_control_overhead_s(self) -> float:
        residual = (
            self.token_pipeline_overhead_s
            - self.token_embedding_fallback_s
            - self.token_state_roundtrip_s
        )
        return residual + self.embedding_fallback_scale * self.token_embedding_fallback_s

    @property
    def effective_token_pipeline_overhead_s(self) -> float:
        state = (
            self.token_state_roundtrip_s
            if self.state_placement == "roundtrip"
            else 0.0
        )
        return self.effective_token_control_overhead_s + state


@dataclass(frozen=True)
class StateSpec:
    context_tokens: int = 0
    sequence_count: int = 1
    attention_kv_head_count: int = 0
    attention_key_length: int = 0
    attention_value_length: int = 0
    kv_element_bytes: int = 2
    ssm_conv_kernel: int = 0
    ssm_state_size: int = 0
    ssm_group_count: int = 0
    ssm_inner_size: int = 0
    recurrent_element_bytes: int = 4

    def __post_init__(self) -> None:
        if self.context_tokens < 0:
            raise ValueError("context_tokens cannot be negative")
        if self.sequence_count <= 0:
            raise ValueError("sequence_count must be positive")
        for field_name in (
            "attention_kv_head_count",
            "attention_key_length",
            "attention_value_length",
            "ssm_conv_kernel",
            "ssm_state_size",
            "ssm_group_count",
            "ssm_inner_size",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.kv_element_bytes <= 0 or self.recurrent_element_bytes <= 0:
            raise ValueError("state element sizes must be positive")

    @property
    def attention_kv_bytes_per_layer(self) -> int:
        elements_per_token = self.attention_kv_head_count * (
            self.attention_key_length + self.attention_value_length
        )
        return (
            self.context_tokens
            * self.sequence_count
            * elements_per_token
            * self.kv_element_bytes
        )

    @property
    def recurrent_state_bytes_per_layer(self) -> int:
        conv_elements = max(0, self.ssm_conv_kernel - 1) * (
            self.ssm_inner_size
            + 2 * self.ssm_group_count * self.ssm_state_size
        )
        matrix_elements = self.ssm_state_size * self.ssm_inner_size
        return (
            self.sequence_count
            * (conv_elements + matrix_elements)
            * self.recurrent_element_bytes
        )

    def bytes_for_layer_kind(self, kind: str) -> int:
        if kind == "attention":
            return self.attention_kv_bytes_per_layer
        if kind == "ssm":
            return self.recurrent_state_bytes_per_layer
        return 0


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: tuple[LayerSpec, ...]
    hardware: HardwareSpec
    policy: PolicySpec
    state: StateSpec | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("at least one layer is required")
        if self.policy.static_gpu_layers > len(self.layers):
            raise ValueError("static_gpu_layers cannot exceed layer count")

    def with_overrides(
        self,
        *,
        window_size: int | None = None,
        h2d_bandwidth_bytes_per_s: float | None = None,
        embedding_fallback_scale: float | None = None,
        context_tokens: int | None = None,
        state_placement: str | None = None,
    ) -> "ModelConfig":
        policy = self.policy
        hardware = self.hardware
        state = self.state
        if window_size is not None:
            policy = replace(policy, window_size=window_size)
        if embedding_fallback_scale is not None:
            policy = replace(
                policy, embedding_fallback_scale=embedding_fallback_scale
            )
        if state_placement is not None:
            policy = replace(policy, state_placement=state_placement)
        if context_tokens is not None:
            if state is None:
                raise ValueError("context_tokens override requires an explicit state model")
            state = replace(state, context_tokens=context_tokens)
        if h2d_bandwidth_bytes_per_s is not None:
            hardware = replace(hardware, h2d_bandwidth_bytes_per_s=h2d_bandwidth_bytes_per_s)
        return replace(self, policy=policy, hardware=hardware, state=state)


def _parse_layers(raw: Any) -> tuple[LayerSpec, ...]:
    if isinstance(raw, list):
        return tuple(
            LayerSpec(
                name=str(item.get("name", f"layer_{index}")),
                weight_bytes=int(item["weight_bytes"]),
                flops=float(item["flops"]),
                kind=str(item.get("kind", "unknown")),
                measured_compute_seconds=(
                    float(item["measured_compute_seconds"])
                    if item.get("measured_compute_seconds") is not None
                    else None
                ),
                measured_transfer_seconds=(
                    float(item["measured_transfer_seconds"])
                    if item.get("measured_transfer_seconds") is not None
                    else None
                ),
            )
            for index, item in enumerate(raw)
        )
    if isinstance(raw, dict):
        count = int(raw["count"])
        weight_bytes = int(raw["uniform_weight_bytes"])
        flops = float(raw["uniform_flops"])
        kind = str(raw.get("uniform_kind", "unknown"))
        measured_compute_seconds = (
            float(raw["uniform_measured_compute_seconds"])
            if raw.get("uniform_measured_compute_seconds") is not None
            else None
        )
        measured_transfer_seconds = (
            float(raw["uniform_measured_transfer_seconds"])
            if raw.get("uniform_measured_transfer_seconds") is not None
            else None
        )
        if count <= 0:
            raise ValueError("layers.count must be positive")
        return tuple(
            LayerSpec(
                name=f"layer_{index}",
                weight_bytes=weight_bytes,
                flops=flops,
                kind=kind,
                measured_compute_seconds=measured_compute_seconds,
                measured_transfer_seconds=measured_transfer_seconds,
            )
            for index in range(count)
        )
    raise ValueError("layers must be a list or a uniform-layer object")


def config_from_dict(raw: dict[str, Any]) -> ModelConfig:
    hardware_raw = raw["hardware"]
    policy_raw = raw["policy"]
    state_raw = raw.get("state")
    return ModelConfig(
        name=str(raw.get("name", "unnamed")),
        layers=_parse_layers(raw["layers"]),
        hardware=HardwareSpec(
            gpu_effective_flops=float(hardware_raw["gpu_effective_flops"]),
            cpu_effective_flops=(
                float(hardware_raw["cpu_effective_flops"])
                if hardware_raw.get("cpu_effective_flops") is not None
                else None
            ),
            h2d_bandwidth_bytes_per_s=float(hardware_raw["h2d_bandwidth_bytes_per_s"]),
            h2d_latency_s=float(hardware_raw.get("h2d_latency_s", 0.0)),
            vram_capacity_bytes=int(hardware_raw["vram_capacity_bytes"]),
            global_bytes=int(hardware_raw.get("global_bytes", 0)),
            kv_bytes=int(hardware_raw.get("kv_bytes", 0)),
            workspace_bytes=int(hardware_raw.get("workspace_bytes", 0)),
            activation_transfer_bytes=int(hardware_raw.get("activation_transfer_bytes", 0)),
            host_staging_bandwidth_bytes_per_s=(
                float(hardware_raw["host_staging_bandwidth_bytes_per_s"])
                if hardware_raw.get("host_staging_bandwidth_bytes_per_s") is not None
                else None
            ),
            state_transfer_bandwidth_bytes_per_s=(
                float(hardware_raw["state_transfer_bandwidth_bytes_per_s"])
                if hardware_raw.get("state_transfer_bandwidth_bytes_per_s") is not None
                else None
            ),
            kv_storage_cached_bandwidth_bytes_per_s=(
                float(hardware_raw["kv_storage_cached_bandwidth_bytes_per_s"])
                if hardware_raw.get("kv_storage_cached_bandwidth_bytes_per_s")
                is not None
                else None
            ),
            kv_storage_uncached_bandwidth_bytes_per_s=(
                float(hardware_raw["kv_storage_uncached_bandwidth_bytes_per_s"])
                if hardware_raw.get("kv_storage_uncached_bandwidth_bytes_per_s")
                is not None
                else None
            ),
            kv_storage_cache_bytes=int(
                hardware_raw.get("kv_storage_cache_bytes", 0)
            ),
        ),
        policy=PolicySpec(
            window_size=int(policy_raw["window_size"]),
            static_gpu_layers=int(policy_raw.get("static_gpu_layers", 0)),
            token_pipeline_overhead_s=float(
                policy_raw.get("token_pipeline_overhead_s", 0.0)
            ),
            token_embedding_fallback_s=float(
                policy_raw.get("token_embedding_fallback_s", 0.0)
            ),
            embedding_fallback_scale=float(
                policy_raw.get("embedding_fallback_scale", 1.0)
            ),
            staging_overlap_window=int(
                policy_raw.get("staging_overlap_window", 2)
            ),
            token_state_roundtrip_s=float(
                policy_raw.get("token_state_roundtrip_s", 0.0)
            ),
            state_placement=str(policy_raw.get("state_placement", "roundtrip")),
            layer_scheduler_overhead_s=float(
                policy_raw.get("layer_scheduler_overhead_s", 0.0)
            ),
        ),
        state=(
            StateSpec(
                context_tokens=int(state_raw.get("context_tokens", 0)),
                sequence_count=int(state_raw.get("sequence_count", 1)),
                attention_kv_head_count=int(
                    state_raw.get("attention_kv_head_count", 0)
                ),
                attention_key_length=int(state_raw.get("attention_key_length", 0)),
                attention_value_length=int(
                    state_raw.get("attention_value_length", 0)
                ),
                kv_element_bytes=int(state_raw.get("kv_element_bytes", 2)),
                ssm_conv_kernel=int(state_raw.get("ssm_conv_kernel", 0)),
                ssm_state_size=int(state_raw.get("ssm_state_size", 0)),
                ssm_group_count=int(state_raw.get("ssm_group_count", 0)),
                ssm_inner_size=int(state_raw.get("ssm_inner_size", 0)),
                recurrent_element_bytes=int(
                    state_raw.get("recurrent_element_bytes", 4)
                ),
            )
            if isinstance(state_raw, dict)
            else None
        ),
        notes=str(raw.get("notes", "")),
    )


def load_config(path: str | Path) -> ModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))
