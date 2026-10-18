"""Profile-independent input construction with an explicit analytical roofline.

This module does not read packaged fixtures or previous measurements. It emits
ordinary backend inputs and verifies them using the existing M2 compiler.
Every emitted timing remains synthetic; physical signatures describe matching
graph/port boundaries, never proof that this analytical timing was measured.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from llm_infer_model.m2.costs import (
    M2CostTable, MacroShape, TransferEntry, canonical_contract_hash, parse_cost_table,
)
from llm_infer_model.m2.dense_contract import (
    resident_accounting, stage_weight_roles, validate_config, weight_shapes,
)
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.full_physical import signature_for as cpu_signature_for
from llm_infer_model.m2.hybrid_physical import signature_for as hybrid_signature_for
from llm_infer_model.m2.moe_spec import parse_route_trace
from llm_infer_model.m2.spec import parse_invocation_document, parse_layer_document
from llm_infer_model.tensor.spec import Unsupported
from llm_infer_model.performance import (
    analytical_compute_duration_ns, analytical_transfer_duration_ns,
)

from .config import Device, Hardware, Model, Workload, _integer


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _route_document(route):
    result = dict(request_id=route.request_id, layer_id=route.layer_id,
                  absolute_position=route.absolute_position, expert_ids=list(route.expert_ids))
    if route.scores is not None:
        result["scores"] = list(route.scores)
    return result


@dataclass(frozen=True)
class AnalyticalCosts:
    """Explicitly choose uncalibrated max(compute time, memory time) costs.

    workspace_bytes=0 deliberately declares a simplified model with no extra
    internal scratch capacity. It does not mean that real kernels use zero
    scratch. A positive value charges that same declared scratch per macro.
    Macro byte traffic is an explanatory estimate, not a cache simulation.
    """

    workspace_bytes: int = 0

    def __post_init__(self):
        _integer(self.workspace_bytes, "AnalyticalCosts.workspace_bytes", 0)

    def compute_duration_ns(self, device: Device, *, flops: int, traffic_bytes: int):
        if not isinstance(device, Device):
            raise ValueError("analytical compute requires a declared Device")
        _integer(flops, "flops", 0)
        _integer(traffic_bytes, "traffic_bytes", 0)
        return analytical_compute_duration_ns(
            flops=flops, traffic_bytes=traffic_bytes,
            flops_per_second=device.compute_flops_per_second,
            memory_bytes_per_second=device.memory_bandwidth_bytes_per_second,
            launch_overhead_ns=device.launch_overhead_ns,
        )

    def transfer_duration_ns(self, hardware: Hardware, *, route: str, storage_bytes: int):
        if not isinstance(hardware, Hardware):
            raise ValueError("analytical transfer requires a declared Hardware")
        _integer(storage_bytes, "storage_bytes")
        if not isinstance(route, str):
            raise ValueError("analytical transfer route must be a string")
        name = {"dram_to_vram": "transfer_bandwidth_bytes_per_second",
                "vram_to_dram": "transfer_bandwidth_bytes_per_second",
                "ssd_to_dram": "ssd_read_bandwidth_bytes_per_second",
                "dram_to_ssd": "ssd_write_bandwidth_bytes_per_second"}.get(route)
        if name is None:
            raise ValueError(f"unsupported analytical transfer route {route!r}")
        rate = getattr(hardware, name)
        if rate is None:
            raise Unsupported(f"analytical route {route!r} needs explicit Hardware.{name}; unknown is not zero")
        return analytical_transfer_duration_ns(
            storage_bytes=storage_bytes, bytes_per_second=rate,
            latency_ns=hardware.transfer_latency_ns,
        )

    def build_inputs(self, directory, model: Model, hardware: Hardware, workload: Workload):
        """Create a self-contained finite model in a new or empty directory.

        The returned files are reusable ordinary JSON inputs. Keep the directory
        while compiling, running, exporting or resuming the backend plan, whose
        existing reproducibility checks hash these input files.
        """
        if not isinstance(model, Model) or not isinstance(hardware, Hardware) or not isinstance(workload, Workload):
            raise ValueError("build_inputs requires Model, Hardware and Workload objects")
        if model.kind == "dense" and workload.batch_size != 1:
            raise Unsupported("the complete Dense backend currently supports batch_size=1 only; no whole-batch timing is duplicated")
        if model.kind == "dense" and (workload.routes or workload.route_source is not None):
            raise ValueError("Dense workloads must not declare MoE routing data")
        if hardware.gpu is not None and hardware.transfer_bandwidth_bytes_per_second is None:
            raise Unsupported("GPU execution requires explicit Hardware.transfer_bandwidth_bytes_per_second")
        if model.kind == "moe":
            for route in ("dram_to_vram", "ssd_to_dram", "dram_to_ssd"):
                self.transfer_duration_ns(hardware, route=route, storage_bytes=4)
            _validate_routes(model, workload)

        directory = Path(directory).resolve()
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("input directory must be new or empty; existing files are never replaced")

        documents, rounds, evidence = {}, [], []
        transfer_records = {}
        version = "m2-layer-0.6" if model.kind == "moe" else (
            "m2-layer-0.5" if hardware.gpu is not None else "m2-layer-0.3")
        parsed_layers = {}
        for layer_index in range(model.layers):
            layer_id = f"layer{layer_index}"
            layer_document = _layer_document(model, layer_id, version)
            documents[f"{layer_id}.layer.json"] = layer_document
            parsed_layers[layer_id] = parse_layer_document(layer_document)

        for round_index in range(workload.decode_steps + 1):
            phase = "prefill" if round_index == 0 else "decode"
            round_id = "prefill" if round_index == 0 else f"decode{round_index - 1}"
            position = 0 if round_index == 0 else workload.prefill_tokens + round_index - 1
            tokens = workload.prefill_tokens if round_index == 0 else 1
            round_entries = []
            for layer_index in range(model.layers):
                layer_id = f"layer{layer_index}"
                scope = f"{layer_id}-{round_id}"
                layer = parsed_layers[layer_id]
                routes = tuple(route for route in workload.routes if route.layer_id == layer_id
                               and position <= route.absolute_position < position + tokens)
                invocation = _invocation_document(model, hardware, workload, layer, scope, phase,
                                                  position, tokens, round_index + 1, routes)
                cost_document, details = self._cost_document(model, hardware, workload, layer,
                                                             phase, tokens, position, routes)
                parsed_costs = parse_cost_table(cost_document)
                recorder = _RecordingCostTable(parsed_costs, hardware, self)
                compile_invocation(layer, parse_invocation_document(invocation), recorder)
                if layer.format_version in ("m2-layer-0.5", "m2-layer-0.6"):
                    cost_document["transfer_entries"] = [
                        dict(route=route, storage_bytes=size, duration_ns=cost.duration_ns,
                             provenance="synthetic_transfer")
                        for (route, size), cost in sorted(recorder.recorded.items())]
                # Validate again through the public exact-table parser. No
                # implicit lookup adapter is involved in the actual saved run.
                compile_invocation(layer, parse_invocation_document(invocation), parse_cost_table(cost_document))
                documents[f"{scope}.json"] = invocation
                documents[f"{scope}.costs.json"] = cost_document
                evidence.extend(dict(layer_id=layer_id, invocation_id=scope, **row) for row in details)
                for key, value in recorder.recorded.items():
                    transfer_records[key] = value.duration_ns
                round_entries.append(dict(id=scope, invocation_file=f"{scope}.json", execution_file=None))
            gate_device = "cpu" if hardware.cpu is not None else "gpu"
            gate_cost = None if round_index == 0 else dict(
                device=gate_device,
                duration_ns=self.compute_duration_ns(getattr(hardware, gate_device),
                    flops=0, traffic_bytes=model.hidden_bytes_per_token),
                workspace_bytes=self.workspace_bytes)
            rounds.append(dict(id=round_id,
                input_mode="initial_supplied_hidden" if round_index == 0 else "trace_supplied_next_token",
                gate_cost=gate_cost, invocations=round_entries))

        model_document = dict(format_version="global-execution-0.1", id="analytical-model", rounds=rounds)
        environment = dict(format_version="environment-trace-0.1", events=[])
        metadata = dict(
            format_version="analytical-inputs-0.1", cost_model="analytical_roofline",
            calibrated=False, hardware_measurements_used=False, generated_profile_required=False,
            numerical_llm_execution=False, provenance="synthetic_layer",
            model=asdict(model), hardware=asdict(hardware),
            workload=dict(prefill_tokens=workload.prefill_tokens, decode_steps=workload.decode_steps,
                          batch_size=workload.batch_size, route_source=workload.route_source,
                          route_count=len(workload.routes)),
            compute_formula="max(1, ceil(1e9 * max(flops / declared_flops_per_second, traffic_bytes / declared_memory_bytes_per_second)) + declared_launch_overhead_ns)",
            transfer_formula="max(1, ceil(1e9 * payload_storage_bytes / declared_route_bandwidth_bytes_per_second) + declared_transfer_latency_ns)",
            flop_convention="multiply-add counts as two FLOPs; elementwise arithmetic coefficients are declared approximations, not instrumented kernel operation counts",
            attention_arithmetic="materialized rectangular QK/PV over q*(history+q) positions; causal masking does not imply free skipping of masked GEMM work",
            workspace_bytes_per_macro=self.workspace_bytes,
            workspace_semantics="explicit simplified internal scratch declaration; zero is not a measurement or a claim about backend allocator use",
            storage="F32 immutable weights and logical per-token KV blocks; weights/input/KV initially DRAM; no numerical weight values loaded",
            macro_contract=version,
            macro_boundaries=("complete declared Dense projection -> materialized attention -> Dense SwiGLU FFN; input and post-attention RMSNorm, normal RoPE, residuals; no QK norm or attention gate"
                              if model.kind == "dense" else
                              "synthetic opaque projection -> attention -> router -> grouped expert FFN -> combine; grouped experts charged once; not a measured complete decoder implementation"),
            trace_gate="per-request trace-supplied next-hidden marker, charged one declared hidden-vector memory write; sampling and embedding are not executed",
            limitations=[
                "uncalibrated analytical estimate, not a hardware prediction validation",
                "declared effective throughput and byte traffic; no cache or operator fusion model",
                "one compute slot per device, one shared PCIe copy slot and one SSD I/O slot",
                "fixed F32 model; no numerical weight loading, quantization or output-token generation",
                "Dense complete-contract batch_size is currently restricted to one",
                "CPU physical envelope threads=1 is a signature identifier, not measured OS scheduling",
                "prefill/decode rounds share one runtime; decode inputs are trace supplied",
                "device availability controls computation; runtime retains both explicit storage-domain capacities",
                "MoE routes are model-owned traces revealed to policy only after router completion",
                "MoE opaque QKV/O pack is storage abstraction and does not expose numerical QKV ports",
                "MoE opaque projection owns the entire packed weight allocation; O-projection arithmetic is estimated in attention but has no independent weight-residency action in this abstraction",
                "MoE backend requires four route bandwidth declarations even when SSD is unused",
                "split/fused execution transforms require independently supplied stage cost contracts",
            ],
            compute_estimates=evidence,
            transfer_estimates=[dict(route=route, storage_bytes=size, duration_ns=duration)
                                for (route, size), duration in sorted(transfer_records.items())],
        )
        documents["model.global.json"] = model_document
        documents["environment.json"] = environment
        documents["analytical-metadata.json"] = metadata
        directory.mkdir(parents=True, exist_ok=True)
        for name, document in documents.items():
            _write(directory / name, document)
        return dict(model_file=directory / "model.global.json", environment_file=directory / "environment.json",
                    model_kind="global_m2", cost_model="analytical_roofline", metadata=metadata)

    def _cost_document(self, model, hardware, workload, layer, phase, tokens, history, routes):
        entries, details = [], []
        kinds = ("projection", "attention", "dense_ffn") if model.kind == "dense" else (
            "projection", "attention", "router", "expert_ffn", "combine")
        group_counts = Counter(expert for route in routes for expert in route.expert_ids)
        for kind in kinds:
            counts = sorted(set(group_counts.values())) if kind == "expert_ffn" else [tokens]
            for count in counts:
                shape = MacroShape(phase, workload.batch_size, count, 0 if kind == "expert_ffn" else history)
                flops, traffic = _macro_work(model, layer, kind, count, shape.visible_history)
                for device_name in hardware.devices:
                    duration = self.compute_duration_ns(getattr(hardware, device_name), flops=flops, traffic_bytes=traffic)
                    entry = dict(kind=kind, device=device_name, phase=phase, batch=workload.batch_size,
                                 tokens=count, visible_history=shape.visible_history, duration_ns=duration,
                                 workspace_bytes=self.workspace_bytes, provenance="synthetic_layer",
                                 accounting="per_expert_group" if kind == "expert_ffn" else "per_request")
                    if model.kind == "dense":
                        signature = (hybrid_signature_for(layer, kind, shape, device_name)
                                     if layer.format_version == "m2-layer-0.5" else cpu_signature_for(layer, kind, shape))
                        entry.update(physical_signature=signature, contract_hash=canonical_contract_hash(signature))
                    entries.append(entry)
                    details.append(dict(kind=kind, device=device_name, phase=phase, batch=workload.batch_size,
                                        tokens=count, visible_history=shape.visible_history, flops=flops,
                                        traffic_bytes=traffic, duration_ns=duration,
                                        workspace_bytes=self.workspace_bytes, calibrated=False))
        document = dict(format_version=layer.format_version, source="explicit analytical roofline; uncalibrated",
                        entries=entries)
        if layer.format_version in ("m2-layer-0.5", "m2-layer-0.6"):
            document["transfer_entries"] = []
        return document, details


class _RecordingCostTable(M2CostTable):
    """Enumerate graph-required payloads once, then save ordinary exact prices."""
    def __init__(self, table, hardware, estimator):
        super().__init__(source=table.source, entries=table.entries, validation_only=table.validation_only,
                         transfers=table.transfers, format_version=table.format_version)
        object.__setattr__(self, "hardware", hardware)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "recorded", {})

    def lookup_transfer(self, route, storage_bytes):
        cost = TransferEntry(self.estimator.transfer_duration_ns(self.hardware, route=route,
                             storage_bytes=storage_bytes), "synthetic_transfer")
        self.recorded[(route, storage_bytes)] = cost
        return cost


def _validate_routes(model, workload):
    if not workload.routes or workload.route_source is None:
        raise ValueError("MoE requires explicit routes and route_source; no synthetic router is substituted")
    parsed = parse_route_trace([_route_document(row) for row in workload.routes], where="Workload.routes")
    positions = workload.prefill_tokens + workload.decode_steps
    expected_count = model.layers * workload.batch_size * positions

    def matches_id(value, prefix, count):
        if not value.startswith(prefix):
            return False
        try:
            index = int(value[len(prefix):])
        except ValueError:
            return False
        return 0 <= index < count and value == f"{prefix}{index}"

    seen = set()
    for row in parsed:
        key = (row.layer_id, row.request_id, row.absolute_position)
        if key in seen:
            raise ValueError(f"duplicate MoE route for {key}")
        seen.add(key)
        if (not matches_id(row.layer_id, "layer", model.layers)
                or not matches_id(row.request_id, "request", workload.batch_size)
                or not 0 <= row.absolute_position < positions):
            raise ValueError(f"MoE route {key} is outside the declared layer/request/token range")
        if len(row.expert_ids) != model.top_k or len(set(row.expert_ids)) != model.top_k:
            raise ValueError(f"MoE route {key} must declare exactly top_k distinct expert IDs")
        if any(type(expert) is not int or not 0 <= expert < model.experts for expert in row.expert_ids):
            raise ValueError(f"MoE route {key} references an expert outside the declared model")
    if len(seen) != expected_count:
        raise ValueError(f"MoE routes must exactly cover every layer/request/token; missing={expected_count-len(seen)}")


def _layer_document(model, layer_id, version):
    base = dict(format_version=version, id=layer_id, provenance="synthetic_layer",
                hidden_size=model.hidden_size, hidden_bytes_per_token=model.hidden_bytes_per_token,
                kv_bytes_per_token=model.kv_bytes_per_token, attention="full_causal", ffn=model.kind)
    if model.kind == "moe":
        hidden, kv = model.hidden_size, model.kv_heads * model.head_dim
        pack = 4 * hidden * (2 * hidden + 2 * kv)
        expert_bytes = 12 * hidden * model.intermediate_size
        weights = [dict(id="W_proj", storage_bytes=pack, initial_location="dram"),
                   dict(id="W_router", storage_bytes=4 * hidden * model.experts, initial_location="dram")]
        weights.extend(dict(id=f"E{expert}", storage_bytes=expert_bytes, initial_location="dram")
                       for expert in range(model.experts))
        kinds = ("projection", "attention", "router", "expert_ffn", "combine")
        ids = ("proj", "attn", "router", "expert", "combine")
        tasks = []
        for index, (kind, name) in enumerate(zip(kinds, ids)):
            task = dict(id=name, kind=kind, depends_on=[] if index == 0 else [ids[index - 1]])
            if kind in ("projection", "router"):
                task["weight"] = "W_proj" if kind == "projection" else "W_router"
            tasks.append(task)
        base.update(weights=weights, tasks=tasks, ffn_params=dict(experts=model.experts, top_k=model.top_k,
            expert_weight_bytes=expert_bytes, expert_weights=[f"E{expert}" for expert in range(model.experts)]))
        return base

    roles = ("input_norm", "q", "k", "v", "output", "post_attention_norm", "gate", "up", "down")
    config = validate_config(dict(hidden=model.hidden_size, intermediate=model.intermediate_size,
        heads=model.heads, kv_heads=model.kv_heads, head_dim=model.head_dim,
        attention_gate="none", qk_norm="none", rope_mode="normal", rope_dims=model.head_dim,
        rope_base=10000.0, norm_epsilon=1e-6, weight_dtypes={role: "f32" for role in roles}))
    weights = [dict(id=f"W_{role}", dtype="f32", shape=list(shape), storage_bytes=4 * math.prod(shape),
                    initial_location="dram") for role, shape in weight_shapes(config).items()]
    tasks = [dict(id="proj", kind="projection", depends_on=[]),
             dict(id="attn", kind="attention", depends_on=["proj"]),
             dict(id="ffn", kind="dense_ffn", depends_on=["attn"])]
    physical = dict(contract="ggml_hybrid_dense_materialized_macros_v1" if version == "m2-layer-0.5" else "ggml_cpu_dense_decoder_block_v1",
        scope="complete_declared_decoder_block", input_boundary="hidden_activation", port_layout="materialized_stage_ports",
        kv_allocation="logical_blocks", activation_dtype="f32", kv_dtype="f32", config=config, threads=1,
        weight_groups={stage: {role: f"W_{role}" for role in stage_weight_roles(config, stage)}
                       for stage in ("projection", "attention", "dense_ffn")}, excluded_operations=[])
    if version == "m2-layer-0.5":
        physical.update(cuda_device=0, cuda_graphs="disabled")
    base.update(weights=weights, tasks=tasks, physical=physical)
    return base


def _invocation_document(model, hardware, workload, layer, scope, phase, position, tokens, version, routes):
    requests, states = [], []
    for index in range(workload.batch_size):
        request = f"request{index}"
        requests.append(dict(request_id=request, position_start=position, tokens=tokens,
                             input_location="dram", output_location="dram"))
        blocks = [dict(id=f"kv:{layer.id}:{request}:{p}", first_position=p, tokens=1,
                       storage_bytes=model.kv_bytes_per_token, locations=["dram"]) for p in range(position)]
        states.append(dict(request_id=request, next_position=position, version=version,
                           state_location="dram", blocks=blocks))
    state = dict(requests=states)
    result = dict(format_version=layer.format_version, id=scope, layer=layer.id,
        layer_file=f"{layer.id}.layer.json", costs_file=f"{scope}.costs.json", phase=phase,
        devices=list(hardware.devices), hardware=dict(dram_capacity_bytes=hardware.dram_capacity_bytes,
            vram_capacity_bytes=hardware.vram_capacity_bytes, dram_alignment_bytes=1, vram_alignment_bytes=1,
            dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0), requests=requests, state_in=state)
    if model.kind == "moe":
        state["weight_copies"] = [dict(id=weight.id, version=1, storage_bytes=weight.storage_bytes,
                                        locations=["dram"]) for weight in layer.weights]
        result.update(route_trace=[_route_document(row) for row in routes], route_source=workload.route_source,
                      information_mode="causal_replay")
    return result


def _macro_work(model, layer, kind, tokens, history):
    """Declared FLOP/byte estimates; never extracted from performance results."""
    h, i, k = model.hidden_size, model.intermediate_size, model.kv_heads * model.head_dim
    q = tokens
    # The materialized QK/PV contract computes rectangular matrices, then
    # applies a causal mask. Do not credit triangular/FlashAttention savings
    # that this graph/kernel contract does not declare.
    pairs = q * (history + q)
    if model.kind == "dense":
        accounting = resident_accounting(layer.physical.config, kind, q, history)
        traffic = (accounting["weight_bytes"] + accounting["input_bytes"] + accounting["output_bytes"]
                   + accounting["kv_history_bytes"] + accounting["kv_new_bytes"])
        if kind == "projection":
            flops = 2 * h * q * (h + 2 * k) + 6 * h * q + 4 * (h + k) * q
        elif kind == "attention":
            flops = 4 * h * pairs + 5 * model.heads * pairs + 2 * h * h * q + h * q
            traffic += 8 * model.heads * q * (history + q)
        else:
            flops = 6 * h * i * q + 8 * i * q + 7 * h * q
            traffic += 28 * i * q
        return flops, traffic
    if kind == "projection":
        return 2 * h * q * (h + 2 * k), layer.weight_by_id("W_proj").storage_bytes + 8 * h * q
    if kind == "attention":
        flops = 4 * h * pairs + 5 * model.heads * pairs + 2 * h * h * q
        traffic = 8 * h * q + model.kv_bytes_per_token * (history + q) + 8 * model.heads * q * (history + q)
        return flops, traffic
    if kind == "router":
        return 2 * h * model.experts * q, 4 * h * model.experts + 4 * h * q + 4 * model.top_k * q
    if kind == "expert_ffn":
        return 6 * h * i * q + 8 * i * q, 12 * h * i + 8 * h * q + 28 * i * q
    if kind == "combine":
        return 2 * model.top_k * h * q, 4 * h * q * (model.top_k + 2) + 4 * model.top_k * q
    raise ValueError(f"unknown analytical macro kind {kind!r}")


__all__ = ["AnalyticalCosts"]
