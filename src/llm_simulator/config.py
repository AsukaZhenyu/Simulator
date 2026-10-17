"""Immutable, explicitly declared inputs for the public simulation API.

These are hypothetical hardware/model declarations, not hardware detection.
No installed GPU, model weights, measurements or network access are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from llm_infer_model.m2.moe_spec import RouteTraceEntry


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value, name, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        valid = math.isfinite(value) and value > 0
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class Model:
    """Full causal GQA with a Dense or trace-routed SwiGLU FFN.

    Small defaults are declared example geometry, never an inferred real model.
    All weights, hidden activations and KV state in this builder are F32.
    Normal adjacent-pair RoPE requires an even head dimension of at least two.
    """

    kind: str = "dense"
    layers: int = 2
    hidden_size: int = 16
    intermediate_size: int = 32
    heads: int = 4
    kv_heads: int = 2
    experts: int | None = None
    top_k: int | None = None

    def __post_init__(self):
        if self.kind not in ("dense", "moe"):
            raise ValueError("Model.kind must be 'dense' or 'moe'")
        for name in ("layers", "hidden_size", "intermediate_size", "heads", "kv_heads"):
            _integer(getattr(self, name), f"Model.{name}")
        if self.hidden_size % self.heads:
            raise ValueError("Model.hidden_size must be divisible by heads")
        if self.heads % self.kv_heads:
            raise ValueError("Model.heads must be divisible by kv_heads")
        if self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("Model.head_dim must be even and >= 2 for normal RoPE")
        if self.kind == "dense":
            if self.experts is not None or self.top_k is not None:
                raise ValueError("Dense Model must not declare experts or top_k")
        else:
            _integer(self.experts, "Model.experts")
            _integer(self.top_k, "Model.top_k")
            if self.top_k > self.experts:
                raise ValueError("Model.top_k must not exceed experts")

    @property
    def head_dim(self):
        return self.hidden_size // self.heads

    @property
    def hidden_bytes_per_token(self):
        return 4 * self.hidden_size

    @property
    def kv_bytes_per_token(self):
        return 8 * self.kv_heads * self.head_dim

    @classmethod
    def dense(cls, **parameters):
        """Declare a Dense model; parameters are the Model field names."""
        if "kind" in parameters:
            raise ValueError("Model.dense() already selects kind='dense'")
        return cls(kind="dense", **parameters)

    @classmethod
    def moe(cls, *, experts, top_k, **parameters):
        """Declare MoE geometry. Workload must separately supply token routes."""
        if "kind" in parameters:
            raise ValueError("Model.moe() already selects kind='moe'")
        return cls(kind="moe", experts=experts, top_k=top_k, **parameters)


@dataclass(frozen=True)
class Device:
    """User-declared effective compute and memory rates for one compute slot.

    These values are not peak specifications fetched from a device database.
    launch_overhead_ns is added once per completed modeled compute task.
    """

    compute_flops_per_second: float
    memory_bandwidth_bytes_per_second: float
    launch_overhead_ns: int = 0

    def __post_init__(self):
        _rate(self.compute_flops_per_second, "Device.compute_flops_per_second")
        _rate(self.memory_bandwidth_bytes_per_second, "Device.memory_bandwidth_bytes_per_second")
        _integer(self.launch_overhead_ns, "Device.launch_overhead_ns", 0)


@dataclass(frozen=True)
class Hardware:
    """Explicit CPU/GPU capacity and optional interconnect declarations.

    cpu=None or gpu=None disables computation on that device. The existing
    runtime retains both storage domains, so each capacity must be >= 1 even
    when that domain has no compute device. GPU execution requires an explicit
    DRAM/VRAM transfer bandwidth. The current MoE backend additionally requires
    both SSD rates because it declares all four possible transfer directions.
    SSD models average sequential payload bandwidth; there is no SSD capacity,
    page, filesystem or queue-depth model in this API.
    """

    cpu: Device | None
    gpu: Device | None
    dram_capacity_bytes: int
    vram_capacity_bytes: int
    transfer_bandwidth_bytes_per_second: float | None = None
    ssd_read_bandwidth_bytes_per_second: float | None = None
    ssd_write_bandwidth_bytes_per_second: float | None = None
    transfer_latency_ns: int = 0

    def __post_init__(self):
        for name in ("cpu", "gpu"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Device):
                raise ValueError(f"Hardware.{name} must be Device or None")
        if self.cpu is None and self.gpu is None:
            raise ValueError("Hardware must declare at least one compute device")
        for name in ("dram_capacity_bytes", "vram_capacity_bytes"):
            _integer(getattr(self, name), f"Hardware.{name}")
        for name in ("transfer_bandwidth_bytes_per_second", "ssd_read_bandwidth_bytes_per_second",
                     "ssd_write_bandwidth_bytes_per_second"):
            _rate(getattr(self, name), f"Hardware.{name}", optional=True)
        _integer(self.transfer_latency_ns, "Hardware.transfer_latency_ns", 0)

    @property
    def devices(self):
        return tuple(name for name in ("cpu", "gpu") if getattr(self, name) is not None)


@dataclass(frozen=True)
class Workload:
    """One prefill followed by a finite number of trace-supplied decode rounds.

    All requests use the same declared length. Their IDs are request0, ...;
    layer IDs are layer0, ... . MoE requires a RouteTraceEntry for every
    layer/request/absolute token position, including prefill. No random router
    or generated dataset is substituted when routes are absent.
    """

    prefill_tokens: int
    decode_steps: int = 0
    batch_size: int = 1
    routes: tuple[RouteTraceEntry, ...] = ()
    route_source: str | None = None

    def __post_init__(self):
        _integer(self.prefill_tokens, "Workload.prefill_tokens")
        _integer(self.decode_steps, "Workload.decode_steps", 0)
        _integer(self.batch_size, "Workload.batch_size")
        if self.prefill_tokens + self.decode_steps > 0x7fffffff:
            raise ValueError("Workload absolute token positions must fit I32")
        if not isinstance(self.routes, tuple) or any(not isinstance(row, RouteTraceEntry) for row in self.routes):
            raise ValueError("Workload.routes must be a tuple of RouteTraceEntry objects")
        for row in self.routes:
            if not isinstance(row.expert_ids, tuple) or (
                    row.scores is not None and not isinstance(row.scores, tuple)):
                raise ValueError("RouteTraceEntry expert_ids and optional scores must be immutable tuples")
        if self.route_source is not None and (
                not isinstance(self.route_source, str) or not self.route_source.strip()):
            raise ValueError("Workload.route_source must be a nonempty source description or None")
        if self.routes and self.route_source is None:
            raise ValueError("Workload.routes requires an explicit route_source")


__all__ = ["Model", "Device", "Hardware", "Workload", "RouteTraceEntry"]
