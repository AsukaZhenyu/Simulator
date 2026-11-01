"""Small expert-offload recipes for the existing strategy compiler.

These are executable *core mechanisms*, not reproductions of complete systems.
No recipe receives a workload, future router rows, a measured execution trace,
or local-machine defaults. The common runtime reveals selected experts after
their router completes; the common compiler derives preparation and release
actions under the declared memory and performance contracts.
"""
from __future__ import annotations

from tensor_mapping.python_frontend import Cache, Prefetch, Scheduling


EXPERT_RECIPE_NAMES = (
    "gpu_demand_lru",
    "cpu_experts_lru",
    "resident_paths_lru",
    "current_cost_lru",
    "hybrid_queues_lru",
)

_RULES = {
    "gpu_demand_lru": "fixed",
    "cpu_experts_lru": "fixed",
    "resident_paths_lru": "fiddler_paths",
    "current_cost_lru": "current_cost",
    "hybrid_queues_lru": "hybrimoe_queues",
}

_SCOPE = {
    "gpu_demand_lru": (
        "GPU executes revealed experts; missing weights are loaded on demand; "
        "LRU evicts recoverable weight groups when the declared budget requires it."
    ),
    "cpu_experts_lru": (
        "CPU executes routed experts; GPU executes the other modules. This is "
        "the fixed placement idea used by CPU/GPU hybrid inference, without "
        "KTransformers kernels, quantization, or expert deferral."
    ),
    "resident_paths_lru": (
        "Fiddler-inspired placement: use a feasible resident GPU expert; otherwise "
        "compare complete CPU and GPU paths including activation transfers. "
        "LRU is an explicit experimental addition, not Fiddler's offline hot set."
    ),
    "current_cost_lru": (
        "Compare currently feasible CPU and GPU paths even for resident experts, "
        "using only public prices and current resource service. No future "
        "router values or future device state is available."
    ),
    "hybrid_queues_lru": (
        "HybriMoE-inspired queue priorities: GPU prefers cached high-load experts, "
        "CPU prefers uncached low-load experts, and transfers prefer uncached "
        "high-load experts. Uses explicit LRU instead of MRS; does not implement "
        "the paper's full allocation optimizer or predictive prefetcher."
    ),
}


def expert_recipe(name: str, *, gpu_weight_budget_bytes: int) -> Scheduling:
    """Return one ordinary Scheduling declaration, usable by ModelPlan too.

    The explicit budget counts GPU *weight payload*, including nonexpert weights.
    It is neither an expert-only cache capacity nor the total VRAM capacity;
    hardware capacity independently includes activations, KV and workspace.
    All recipes use the same cache budget, expert-group eviction, and no prefetch.
    """
    if not isinstance(name, str) or name not in EXPERT_RECIPE_NAMES:
        raise ValueError(f"unknown expert recipe; choose one of {EXPERT_RECIPE_NAMES}")
    if type(gpu_weight_budget_bytes) is not int or gpu_weight_budget_bytes < 0:
        raise ValueError("gpu_weight_budget_bytes must be an explicit nonnegative integer")
    return Scheduling(
        default_device="gpu",
        placements={"expert_ffn": "cpu"} if name == "cpu_experts_lru" else {},
        expert_rule=_RULES[name],
        expert_order="graph_order",
        release="last_use",
        cache=Cache(rule="lru", unit="expert_group", budget_bytes=gpu_weight_budget_bytes),
        prefetch=Prefetch(),
    )


def expert_recipe_scope(name: str) -> dict:
    """Return machine-readable claim limits alongside an experiment result."""
    if not isinstance(name, str) or name not in EXPERT_RECIPE_NAMES:
        raise ValueError(f"unknown expert recipe; choose one of {EXPERT_RECIPE_NAMES}")
    return {
        "id": name,
        "core_mechanism": _SCOPE[name],
        "full_paper_reproduction": False,
        "future_route_visibility": False,
        "prefetch_enabled": False,
        "cache_rule": "lru",
        "cache_unit": "expert_group",
        "budget_scope": "all_gpu_weight_payload_not_total_vram",
        "numerical_execution_provided_by_recipe": False,
        "cost_source": "must_be_supplied_by_caller",
    }
