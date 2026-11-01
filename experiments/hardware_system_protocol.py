"""Frozen workloads and honest GPU-derived NPU substitution experiments.

This is an experiment adapter, not an additional simulator device backend.
An NPU proxy substitutes for the existing ``gpu`` execution slot. It cannot
execute alongside a real GPU in the same run. No NPU measurement, compiler,
operator support claim or hardware-specific F32 performance is implied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path

from llm_simulator import (
    AnalyticalCosts, Cache, Hardware, Model, Prefetch, Route, Scheduling, Workload,
)


FORMAT_VERSION = "hardware-system-protocol-0.1"
CACHE_BUDGET_BYTES = None
GEOMETRY = dict(layers=2, hidden_size=6144, intermediate_size=16384,
                heads=24, kv_heads=6)
PROXY_ALPHAS = (0.5, 1.0, 2.0)
REFERENCE = "e0-s0"


def _positive_alpha(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("NPU proxy alpha must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("NPU proxy alpha must be a finite positive number")
    return Fraction(str(value))


def proxy_duration_ns(reference_gpu_duration_ns: int, alpha: float) -> int:
    """Scale a complete integer task price, with exact rational ceiling.

    The reference price already contains its declared launch cost. Scaling it
    therefore assumes *both* useful execution and launch overhead scale by
    alpha; it is deliberately not a TOPS-to-FLOPS conversion.
    """
    if type(reference_gpu_duration_ns) is not int or reference_gpu_duration_ns < 1:
        raise ValueError("reference GPU duration must be a positive integer")
    factor = _positive_alpha(alpha)
    numerator = reference_gpu_duration_ns * factor.numerator
    return (numerator + factor.denominator - 1) // factor.denominator


@dataclass(frozen=True)
class NPUProxyCosts(AnalyticalCosts):
    """CPU plus an estimated NPU in the existing GPU execution slot.

    t_proxy = ceil(alpha * t_reference_gpu), including reference launch time.
    CPU prices, physical storage, transfers, dependencies and declared scratch
    are unchanged. F32 availability and all scheduled task support are explicit
    assumptions. Saved boundary signatures remain synthetic GPU-compatible
    envelopes; they are not real NPU execution contracts.
    """

    alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        _positive_alpha(self.alpha)

    def _cost_document(self, model, hardware, workload, layer, phase, tokens, history, routes):
        document, details = super()._cost_document(
            model, hardware, workload, layer, phase, tokens, history, routes)
        for row in document["entries"]:
            if row["device"] == "gpu":
                row["duration_ns"] = proxy_duration_ns(row["duration_ns"], self.alpha)
        for row in details:
            if row["device"] == "gpu":
                reference = row["duration_ns"]
                row.update(reference_gpu_duration_ns=reference,
                           duration_ns=proxy_duration_ns(reference, self.alpha),
                           npu_proxy_alpha=self.alpha,
                           actual_npu_measurement=False)
        if self.alpha != 1:
            document["source"] = (
                f"synthetic NPU proxy in gpu slot; ceil({self.alpha} * complete GPU task price); uncalibrated")
        return document, details

    def build_inputs(self, directory, model: Model, hardware: Hardware, workload: Workload):
        if not isinstance(hardware, Hardware) or hardware.cpu is None or hardware.gpu is None:
            raise ValueError("NPU substitution requires declared CPU and reference GPU slots")
        if not isinstance(workload, Workload):
            raise ValueError("this NPU proxy experiment supports finite Workload inputs only")
        result = super().build_inputs(directory, model, hardware, workload)
        metadata = result["metadata"]
        reference_formula = metadata["compute_formula"]
        metadata.update(
            cost_model="gpu_derived_npu_proxy",
            compute_formula=("cpu: reference analytical formula; gpu slot substitutes NPU: "
                             "ceil(alpha * complete integer reference GPU task duration_ns)"),
            npu_proxy=dict(
                estimated=True, measured=False, alpha=self.alpha,
                proxy_device="npu", execution_slot="gpu",
                simultaneous_gpu_and_npu=False,
                reference_gpu_device=asdict(hardware.gpu),
                reference_gpu_formula=reference_formula,
                scaling_boundary="complete task duration including reference GPU launch overhead",
                cpu_costs_scaled=False, transfer_costs_scaled=False,
                workspace_bytes_per_task=self.workspace_bytes,
                workspace_provenance="same explicit synthetic scratch assumption as GPU reference",
                precision="F32 logical tensors; assumed task support, not a real NPU F32 claim",
                support_assumption="every scheduled GPU-envelope task can execute on the hypothetical NPU",
                physical_signatures=("synthetic GPU-compatible structural envelopes only; "
                                     "not NPU ABI, compiled kernels, measurements or support evidence"),
                transfer_assumption=("same declared memory topology and transfer prices as paired GPU run; "
                                     "no actual NPU copy/conversion/synchronization measurement")),
        )
        metadata["limitations"].extend([
            "NPU proxy substitutes the gpu slot; true CPU+GPU+NPU concurrency is not implemented",
            "NPU duration scaling applies uniformly to all task shapes, phases and launch costs; actual NPU speedups may be nonuniform",
            "NPU F32 task availability, layout compatibility and unchanged scratch are assumptions; unsupported operators and CPU fallback are not modeled",
            "NPU proxy is a sensitivity experiment; no named NPU performance ranking is established",
        ])
        path = Path(directory) / "analytical-metadata.json"
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
        result["cost_model"] = "gpu_derived_npu_proxy"
        return result


@dataclass(frozen=True)
class WorkloadSpec:
    id: str
    model_kind: str
    prefill_tokens: int
    decode_steps: int
    route_pattern: str | None = None


_WORKLOADS = (
    WorkloadSpec("dense-prefill", "dense", 32, 0),
    WorkloadSpec("dense-decode", "dense", 8, 2),
    WorkloadSpec("moe-decode-hot", "moe", 8, 2, "hot"),
    WorkloadSpec("moe-decode-rotating", "moe", 8, 2, "rotating"),
)


def workload_specs() -> tuple[WorkloadSpec, ...]:
    return _WORKLOADS


def make_workload(spec: WorkloadSpec) -> tuple[Model, Workload]:
    """Same two-layer F32 graph for every hardware/strategy comparison.

    Shape magnitudes affect symbolic task work and physical capacities, not
    actual array allocation in the measurement host. These are LLM-shaped
    finite subgraphs rather than a claim about complete model quality.
    """
    if spec not in _WORKLOADS:
        raise ValueError("workload must be one of the predeclared hardware-system cases")
    model = (Model.dense(**GEOMETRY) if spec.model_kind == "dense" else
             Model.moe(**GEOMETRY, experts=4, top_k=2))
    routes = ()
    if spec.model_kind == "moe":
        routes = tuple(
            Route("request0", f"layer{layer}", position,
                  (0, 1) if spec.route_pattern == "hot" else
                  ((position + layer) % 4, (position + layer + 1) % 4))
            for layer in range(model.layers)
            for position in range(spec.prefill_tokens + spec.decode_steps)
        )
    return model, Workload(
        prefill_tokens=spec.prefill_tokens, decode_steps=spec.decode_steps,
        routes=routes,
        route_source=(f"declared_synthetic_{spec.route_pattern}_not_measured" if routes else None),
    )


def candidate_ids(kind: str) -> tuple[str, ...]:
    if kind not in ("dense", "moe"):
        raise ValueError("candidate model kind must be dense or moe")
    return tuple(f"e{execution}-s{storage}" for execution in range(4) for storage in range(2))


def candidate_spec(kind: str, candidate_id: str, *, cache_budget_bytes: int | None = CACHE_BUDGET_BYTES) -> Scheduling:
    if candidate_id not in candidate_ids(kind):
        raise ValueError("candidate must be one of the eight predeclared strategies")
    if cache_budget_bytes is not None and (type(cache_budget_bytes) is not int or cache_budget_bytes < 1):
        raise ValueError("cache budget must be None or an explicit positive integer")
    execution, storage = (int(part[1:]) for part in candidate_id.split("-"))
    ffn = "dense_ffn" if kind == "dense" else "expert_ffn"
    return Scheduling(
        default_device="cpu" if execution == 1 else "gpu",
        placements={2: {ffn: "cpu"}, 3: {"attention": "cpu"}}.get(execution, {}),
        expert_rule="fixed", expert_order="graph_order", release="last_use",
        cache=Cache(rule="lru", budget_bytes=cache_budget_bytes),
        prefetch=(Prefetch(next_layers=1, max_count=2, include_dense_weights=True)
                  if storage else Prefetch()),
    )


def protocol_document() -> dict:
    return dict(
        format_version=FORMAT_VERSION,
        calibrated=False, hardware_measurements_used=False,
        precision="F32", workspace_bytes_per_task=0,
        reference=REFERENCE, geometry=dict(GEOMETRY),
        graph_scope="two finite LLM layers; not full production model numerical execution",
        initial_state="cold immutable weights in DRAM; empty KV; fresh runtime per candidate",
        final_goals="same mandatory hidden and KV DRAM views; retained optional weights may differ",
        device_scope="single CPU plus one accelerator slot; GPU or hypothetical NPU substitution",
        cache_budget_bytes=CACHE_BUDGET_BYTES,
        cache_budget_semantics=("no artificial software quota; LRU retention is constrained by declared physical capacity, "
                                "including workspace, activations and shared allocations"),
        capacity_semantics="installed or declared usable capacity is explicit per hardware; no hidden rescaling to graph size",
        candidate_count_per_workload=8,
        candidate_descriptions=dict(
            e0="all modeled compute tasks on accelerator",
            e1="all modeled compute tasks on CPU",
            e2="Dense FFN or MoE experts on CPU, other tasks on accelerator",
            e3="attention on CPU, other tasks on accelerator",
            s0="LRU weight quota, no prefetch",
            s1="same LRU quota plus one-layer lookahead, max two weights, including Dense weights"),
        npu_proxy_alphas=PROXY_ALPHAS,
        npu_proxy_equation="t_proxy_ns = ceil(alpha * t_gpu_reference_ns), including reference launch cost",
        npu_proxy_claim="hypothetical sensitivity only; no measured or complete three-device NPU support",
        workloads=[dict(spec=asdict(spec), model=asdict(model), workload=asdict(workload))
                   for spec in _WORKLOADS for model, workload in (make_workload(spec),)],
        candidates={kind: {candidate: asdict(candidate_spec(kind, candidate))
                           for candidate in candidate_ids(kind)} for kind in ("dense", "moe")},
    )


__all__ = ["NPUProxyCosts", "proxy_duration_ns", "WorkloadSpec", "workload_specs",
           "make_workload", "candidate_ids", "candidate_spec", "protocol_document",
           "CACHE_BUDGET_BYTES", "GEOMETRY", "PROXY_ALPHAS", "REFERENCE", "FORMAT_VERSION"]
