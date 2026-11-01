"""Interpretable inclusive transfer responses with an explicit calibration scope.

The mechanism is a nonnegative fixed dispatch/completion term plus one or two
byte-service slopes. A change to payload bandwidth scales byte-service terms,
not the fixed term. Qualification belongs to an independent profiling receipt;
this module never fits observations or substitutes a hardware rate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def inclusive_response_ns(model, size_bytes):
    """Unqualified mathematical response used only by explicit fitting code."""
    split = model["breakpoint_bytes"]
    short = size_bytes if split is None else min(size_bytes, split)
    long = 0 if split is None else max(0, size_bytes - split)
    return model["fixed_ns"] + model["small_ns_per_byte"] * short + model["large_ns_per_byte"] * long


@dataclass(frozen=True)
class TransferResponse:
    route: str
    fixed_ns: float
    small_ns_per_byte: float
    large_ns_per_byte: float
    breakpoint_bytes: int | None
    minimum_bytes: int
    maximum_bytes: int
    host_memory_contract: str
    boundary: str
    evidence: tuple[str, ...]
    qualified: bool = False
    payload_scaling_qualified: bool = False

    def __post_init__(self):
        if self.route not in {"dram->vram", "vram->dram"}:
            raise ValueError("explicit supported transfer direction required")
        for value in (self.fixed_ns, self.small_ns_per_byte, self.large_ns_per_byte):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative transfer response terms required")
        if not (type(self.minimum_bytes) is int and type(self.maximum_bytes) is int
                and 0 < self.minimum_bytes <= self.maximum_bytes):
            raise ValueError("positive measured size envelope required")
        if self.breakpoint_bytes is not None and not (
                type(self.breakpoint_bytes) is int
                and self.minimum_bytes < self.breakpoint_bytes < self.maximum_bytes):
            raise ValueError("breakpoint must lie strictly inside measured envelope")
        if not self.host_memory_contract or not self.boundary or not self.evidence:
            raise ValueError("host memory, timing boundary and measurement evidence required")
        if self.breakpoint_bytes is None and self.small_ns_per_byte != self.large_ns_per_byte:
            raise ValueError("one-regime response requires equal slopes")

    def components(self, size_bytes: int):
        if type(size_bytes) is not int or not self.minimum_bytes <= size_bytes <= self.maximum_bytes:
            raise ValueError("transfer size outside independently tested envelope")
        short = size_bytes if self.breakpoint_bytes is None else min(size_bytes, self.breakpoint_bytes)
        long = 0 if self.breakpoint_bytes is None else max(0, size_bytes - self.breakpoint_bytes)
        return dict(fixed_ns=self.fixed_ns,
                    small_payload_ns=self.small_ns_per_byte * short,
                    large_payload_ns=self.large_ns_per_byte * long)

    def predict_ns(self, size_bytes: int, *, payload_bandwidth_factor: float = 1.0,
                   require_qualified: bool = True):
        if require_qualified and not self.qualified:
            raise ValueError("transfer response lacks independent qualification")
        if isinstance(payload_bandwidth_factor, bool) or not math.isfinite(payload_bandwidth_factor) or payload_bandwidth_factor <= 0:
            raise ValueError("finite positive bandwidth factor required")
        terms = self.components(size_bytes)
        if payload_bandwidth_factor != 1 and not self.payload_scaling_qualified:
            raise ValueError("byte interpolation does not qualify hardware bandwidth scaling")
        return terms["fixed_ns"] + (terms["small_payload_ns"] + terms["large_payload_ns"]) / payload_bandwidth_factor

    def payload_ns(self, size_bytes: int):
        if not self.qualified:
            raise ValueError("payload response lacks independent qualification")
        terms = self.components(size_bytes)
        return terms["small_payload_ns"] + terms["large_payload_ns"]

    def scenario_ns(self, anchor_mean_ns: float, size_bytes: int, factor: float, *,
                    assume_effective_payload_scaling: bool = False):
        """Preserve an exact measured anchor; change only qualified byte service.

        A caller may explicitly assume effective byte-service scaling for a
        counterfactual. Such an assumption does not identify PCIe bandwidth.
        """
        if isinstance(anchor_mean_ns, bool) or not math.isfinite(anchor_mean_ns) or anchor_mean_ns < 0:
            raise ValueError("nonnegative finite exact anchor required")
        if isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("finite positive byte-service factor required")
        payload = self.payload_ns(size_bytes)
        residual = anchor_mean_ns - payload
        if residual < 0:
            raise ValueError("byte-service model exceeds exact anchor; fixed residual unknown")
        if factor != 1 and not (self.payload_scaling_qualified or assume_effective_payload_scaling):
            raise ValueError("bandwidth scaling requires qualification or explicit counterfactual assumption")
        return residual + payload / factor

