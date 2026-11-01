"""Explicit preparation contracts for inclusive CPU/GPU transfer responses.

This optional provider preserves the existing default. It reuses the existing
nonnegative affine/continuous-hinge response family and never fits observations.
An inclusive host-completed price is the sole timing contribution. CUDA event
intervals and copy-call aliases are diagnostics, not additive costs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from .transfer_response import inclusive_response_ns


MEASUREMENT_API = "ggml_tensor_copy_then_gpu_cpu_synchronize"
COMPLETION_SEMANTIC = "host_response_including_copy_api_and_gpu_cpu_completion"
CONTRACT_FIELDS = frozenset({"route", "memory_mode", "measurement_api",
                            "completion_semantic", "preparation_contract", "scope_sha256"})
PREPARATION_FIELDS = frozenset({"reset_before_batch", "untimed_transfer_prime_passes",
    "timed_replays_per_batch", "warmup_iterations_min", "warmup_ms_min",
    "fixed_input_replay", "equal_warmup_per_memory_mode_visit"})
PARAMETER_FIELDS = ("fixed_ns", "small_ns_per_byte", "large_ns_per_byte", "breakpoint_bytes")


def _snapshot(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_contract(contract):
    """Require explicit supported API, host-memory and preparation semantics."""
    value = _snapshot(contract)
    if not isinstance(value, dict) or set(value) != CONTRACT_FIELDS:
        raise ValueError("explicit transfer contract fields required; unknown fields rejected")
    if value["route"] not in {"dram->vram", "vram->dram"}:
        raise ValueError("unsupported transfer route")
    if value["memory_mode"] not in {"pageable", "registered"}:
        raise ValueError("observed pageable or registered host memory required")
    if value["measurement_api"] != MEASUREMENT_API or value["completion_semantic"] != COMPLETION_SEMANTIC:
        raise ValueError("unsupported transfer API or completion semantics")
    if not isinstance(value["scope_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["scope_sha256"]):
        raise ValueError("explicit hardware/binary/runtime scope digest required")
    prep = value["preparation_contract"]
    if not isinstance(prep, dict) or set(prep) != PREPARATION_FIELDS:
        raise ValueError("complete measured transfer preparation required")
    for name in ("reset_before_batch", "fixed_input_replay", "equal_warmup_per_memory_mode_visit"):
        if type(prep[name]) is not bool:
            raise ValueError(name + " must be explicit bool")
    if not prep["fixed_input_replay"] or not prep["equal_warmup_per_memory_mode_visit"]:
        raise ValueError("only measured fixed-input, equally warmed contexts are supported")
    for name in ("untimed_transfer_prime_passes", "timed_replays_per_batch", "warmup_iterations_min", "warmup_ms_min"):
        minimum = 1 if name == "timed_replays_per_batch" else 0
        if type(prep[name]) is not int or prep[name] < minimum:
            raise ValueError(name + " must be an explicit valid count")
    return value


def validate_parameters(parameters, minimum_bytes, maximum_bytes):
    if type(minimum_bytes) is not int or type(maximum_bytes) is not int or not 0 < minimum_bytes <= maximum_bytes:
        raise ValueError("positive calibration-size envelope required")
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_FIELDS):
        raise ValueError("exact nonnegative affine/continuous-hinge parameters required")
    for key in PARAMETER_FIELDS[:-1]:
        number = parameters[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("finite nonnegative response coefficients required")
    split = parameters["breakpoint_bytes"]
    if split is None:
        if parameters["small_ns_per_byte"] != parameters["large_ns_per_byte"]:
            raise ValueError("affine response must have equal slopes")
    elif type(split) is not int or not minimum_bytes < split < maximum_bytes:
        raise ValueError("continuous hinge must lie inside the calibration envelope")


def transfer_coordinate(size_bytes, *, contract):
    """A public byte/preparation coordinate, without program or process identity."""
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("positive integer transfer bytes required")
    return dict(schema="prepared-transfer-coordinate-1", bytes=size_bytes,
                contract=validate_contract(contract))


@dataclass(frozen=True)
class PreparedTransferModel:
    """A frozen response with exact contract matching and explicit reuse scope."""
    _profile_json: str

    def __post_init__(self):
        profile = json.loads(self._profile_json)
        if profile.get("schema") != "prepared-transfer-response-profile-1":
            raise ValueError("prepared transfer profile schema differs")
        validate_contract(profile["contract"])
        validate_parameters(profile["parameters"], profile["minimum_bytes"], profile["maximum_bytes"])
        if type(profile.get("qualified")) is not bool:
            raise ValueError("response qualification must be explicit")
        if not profile.get("evidence"):
            raise ValueError("independent calibration evidence required")
        if profile["qualified"] and not profile.get("qualification_receipt"):
            raise ValueError("qualification receipt required")

    @classmethod
    def from_profile(cls, profile):
        return cls(json.dumps(_snapshot(profile), sort_keys=True, allow_nan=False))

    @property
    def profile(self):
        """Return a copy so callers cannot silently mutate the frozen model."""
        return json.loads(self._profile_json)

    def prediction(self, size_bytes, *, contract, allow_unqualified=False,
                   allow_batch_mean_as_single_call=False):
        profile = self.profile
        target = validate_contract(contract)
        trained = profile["contract"]
        assumptions = []
        if target != trained:
            one_call = _snapshot(trained)
            one_call["preparation_contract"]["timed_replays_per_batch"] = 1
            if (allow_batch_mean_as_single_call and target == one_call
                    and trained["preparation_contract"]["timed_replays_per_batch"] > 1):
                assumptions.append("calibration batch mean reused as a single completed-call response; sampling equivalence is not qualified")
            else:
                raise ValueError("target transfer preparation, memory, API or runtime contract differs")
        if type(size_bytes) is not int or not profile["minimum_bytes"] <= size_bytes <= profile["maximum_bytes"]:
            raise ValueError("transfer size outside calibration envelope")
        if not allow_unqualified and (not profile["qualified"] or assumptions):
            raise ValueError("independent qualification for this exact transfer contract required")
        duration = inclusive_response_ns(profile["parameters"], size_bytes)
        return dict(duration_ns=duration, scope_assumptions=assumptions,
            exact_contract_match=not assumptions,
            qualified_for_requested_contract=profile["qualified"] and not assumptions,
            price_boundary=COMPLETION_SEMANTIC,
            diagnostic_event_or_phase_cost_added=False,
            physical_payload_bandwidth_identified=False,
            size_bytes=size_bytes, coordinate=transfer_coordinate(size_bytes, contract=target))

