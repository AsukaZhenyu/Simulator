"""Predeclared finite strategy/workload matrix for analytical comparisons.

This catalogue fixes inputs before simulation. Synthetic routing belongs to the
model input; it does not supply future routes to a scheduling policy. All runs
use the public compiler/runtime causal revelation mechanism without alteration.
No measured device rates, numerical LLM execution or global optimum is claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from llm_simulator import (
    Cache, Device, Hardware, Model, Prefetch, Route, Scheduling, Workload,
)


FORMAT_VERSION = "strategy-comparison-catalogue-0.1"
CACHE_BYTES = 192 * 1024
REFERENCE = "e0-s0"
ROUTE_PATTERNS = ("hot", "rotating", "phase_shift")
WORKLOAD_SHAPES = (
    ("short_prefill", 8, 0),
    ("long_prefill", 128, 0),
    ("decode_heavy", 8, 8),
    ("mixed", 128, 8),
)


@dataclass(frozen=True)
class ScenarioSpec:
    """One complete, immutable model-owned workload declaration."""

    id: str
    model_kind: str
    prefill_tokens: int
    decode_steps: int
    route_pattern: str | None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or any(c in self.id for c in "\r\n/\\"):
            raise ValueError("ScenarioSpec.id must be a nonempty single path component")
        if self.model_kind not in ("dense", "moe"):
            raise ValueError("ScenarioSpec.model_kind must be dense or moe")
        if type(self.prefill_tokens) is not int or self.prefill_tokens < 1:
            raise ValueError("ScenarioSpec.prefill_tokens must be a positive integer")
        if type(self.decode_steps) is not int or self.decode_steps < 0:
            raise ValueError("ScenarioSpec.decode_steps must be a nonnegative integer")
        if self.model_kind == "dense" and self.route_pattern is not None:
            raise ValueError("Dense scenarios must not declare a route pattern")
        if self.model_kind == "moe" and self.route_pattern not in ROUTE_PATTERNS:
            raise ValueError("MoE scenarios require an explicitly declared route pattern")


_SCENARIOS = tuple(
    ScenarioSpec(f"dense-{name}", "dense", prefill, decode, None)
    for name, prefill, decode in WORKLOAD_SHAPES
) + tuple(
    ScenarioSpec(f"moe-{name}-{pattern}", "moe", prefill, decode, pattern)
    for name, prefill, decode in WORKLOAD_SHAPES for pattern in ROUTE_PATTERNS
)


def scenarios() -> tuple[ScenarioSpec, ...]:
    """Return the fixed 4 Dense + 12 MoE cases, with no result-based selection."""
    return _SCENARIOS


def candidate_ids(kind: str) -> tuple[str, ...]:
    """Four or six execution choices crossed with four storage choices."""
    if kind not in ("dense", "moe"):
        raise ValueError("candidate model kind must be dense or moe")
    return tuple(f"e{execution}-s{storage}"
                 for execution in range(4 if kind == "dense" else 6)
                 for storage in range(4))


def candidate_spec(kind: str, candidate_id: str) -> Scheduling:
    """Declare placement/adaptive execution and same-budget cache/prefetch."""
    if candidate_id not in candidate_ids(kind):
        raise ValueError(f"undeclared {kind} candidate {candidate_id!r}")
    execution, storage = (int(part[1:]) for part in candidate_id.split("-"))
    ffn_kind = "dense_ffn" if kind == "dense" else "expert_ffn"
    placements = {2: {ffn_kind: "cpu"}, 3: {"attention": "cpu"}}.get(execution, {})
    expert_rule = {4: "resident_first", 5: "current_cost"}.get(execution, "fixed")
    return Scheduling(
        default_device="cpu" if execution == 1 else "gpu",
        placements=placements,
        expert_rule=expert_rule,
        expert_order="graph_order",
        release="last_use",
        cache=Cache(rule="lru" if storage < 2 else "history_frequency",
                    budget_bytes=CACHE_BYTES),
        prefetch=(Prefetch(next_layers=1, max_count=2, include_dense_weights=True)
                  if storage % 2 else Prefetch()),
    )


def _expert_ids(spec: ScenarioSpec, layer: int, position: int) -> tuple[int, int]:
    if spec.route_pattern == "hot":
        return (0, 1)
    if spec.route_pattern == "rotating":
        return ((position + layer) % 4, (position + layer + 1) % 4)
    if spec.route_pattern == "phase_shift":
        return (0, 1) if position < spec.prefill_tokens else (2, 3)
    raise ValueError("expert IDs require a declared MoE route pattern")


def make_inputs(spec: ScenarioSpec) -> tuple[Model, Hardware, Workload]:
    """Build identical hypothetical hardware and the declared finite workload."""
    if not isinstance(spec, ScenarioSpec):
        raise TypeError("make_inputs requires ScenarioSpec")
    geometry = dict(layers=2, hidden_size=64, intermediate_size=192, heads=4, kv_heads=2)
    model = (Model.dense(**geometry) if spec.model_kind == "dense"
             else Model.moe(**geometry, experts=4, top_k=2))
    hardware = Hardware(
        cpu=Device(100e9, 20e9, 500),
        gpu=Device(1e12, 100e9, 2000),
        dram_capacity_bytes=128 * 1024**2,
        vram_capacity_bytes=512 * 1024,
        transfer_bandwidth_bytes_per_second=8e9,
        transfer_latency_ns=1000,
        ssd_read_bandwidth_bytes_per_second=1e9,
        ssd_write_bandwidth_bytes_per_second=500e6,
    )
    routes = (() if spec.model_kind == "dense" else tuple(
        Route("request0", f"layer{layer}", position, _expert_ids(spec, layer, position))
        for layer in range(model.layers)
        for position in range(spec.prefill_tokens + spec.decode_steps)
    ))
    workload = Workload(
        prefill_tokens=spec.prefill_tokens,
        decode_steps=spec.decode_steps,
        batch_size=1,
        routes=routes,
        route_source=(f"declared_synthetic_{spec.route_pattern}_not_measured" if routes else None),
    )
    return model, hardware, workload


def catalogue_document() -> dict:
    """Serializable protocol, including every actual route and strategy input."""
    return dict(
        format_version=FORMAT_VERSION,
        calibrated=False,
        hardware_measurement=False,
        numerical_llm_execution=False,
        cost_model="analytical_roofline",
        dtype="F32",
        workspace_bytes=0,
        reference=REFERENCE,
        scenario_count=len(_SCENARIOS),
        distinct_workload_count=14,
        equivalence_controls={
            "moe-short_prefill-phase_shift": "moe-short_prefill-hot",
            "moe-long_prefill-phase_shift": "moe-long_prefill-hot",
        },
        equivalence_control_note=("With no Decode, phase_shift equals hot exactly; these two cases "
                                  "are intentional equivalence controls, not additional route-shift evidence."),
        candidate_count_by_model={kind: len(candidate_ids(kind)) for kind in ("dense", "moe")},
        declared_run_count=sum(len(candidate_ids(spec.model_kind)) for spec in _SCENARIOS),
        initial_state="cold weights in DRAM; empty KV; fresh runtime per candidate",
        final_goals="mandatory hidden/KV in DRAM; retained optional weight copies may differ",
        route_information="model-owned synthetic trace; public routes revealed only after router completion",
        shared_transfer_link="single DRAM/VRAM transfer link at the declared payload rate and latency",
        storage_choices={
            "s0": "LRU; no prefetch",
            "s1": "LRU; next-layer prefetch (lookahead 1, max_count 2, include dense weights)",
            "s2": "history_frequency; no prefetch",
            "s3": "history_frequency; identical next-layer prefetch",
        },
        execution_choices={
            "e0": "all GPU",
            "e1": "all CPU",
            "e2": "Dense FFN or MoE experts on CPU; remaining tasks on GPU",
            "e3": "Attention on CPU; remaining tasks on GPU",
            "e4": "MoE revealed expert regions use resident_first; remaining tasks on GPU",
            "e5": "MoE revealed expert regions use current_cost; remaining tasks on GPU",
        },
        strategies={kind: {candidate: asdict(candidate_spec(kind, candidate))
                           for candidate in candidate_ids(kind)} for kind in ("dense", "moe")},
        scenarios=[dict(spec=asdict(spec), model=asdict(model), hardware=asdict(hardware),
                        workload=asdict(workload))
                   for spec in _SCENARIOS for model, hardware, workload in (make_inputs(spec),)],
    )


__all__ = ["ScenarioSpec", "scenarios", "candidate_ids", "candidate_spec", "make_inputs",
           "catalogue_document", "CACHE_BYTES", "REFERENCE", "FORMAT_VERSION"]
