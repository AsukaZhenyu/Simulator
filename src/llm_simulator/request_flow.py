"""Profile-independent, finite request streams on one CPU/GPU runtime.

Each request kernel has its own exact B1 price. Concurrent requests are not
silently reinterpreted as a fused batch, and token inputs are trace supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from llm_infer_model.m2.costs import parse_cost_table
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.moe_spec import RouteTraceEntry, parse_route_trace
from llm_infer_model.m2.spec import parse_invocation_document, parse_layer_document
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .analytical import _RecordingCostTable, _invocation_document, _layer_document, _memory_metadata, _write
from .config import Hardware, Model, Workload, _integer


@dataclass(frozen=True)
class Request:
    request_id: str
    prefill_tokens: int
    decode_steps: int = 0
    arrival_ns: int = 0

    def __post_init__(self):
        if not isinstance(self.request_id,str) or not self.request_id or any(c in self.request_id for c in ":\r\n/\\"):
            raise InvalidInput("Request.request_id must be nonempty without colon, newline or path separators")
        _integer(self.prefill_tokens,"Request.prefill_tokens")
        _integer(self.decode_steps,"Request.decode_steps",0)
        _integer(self.arrival_ns,"Request.arrival_ns",0)
        if self.prefill_tokens+self.decode_steps>0x7fffffff:
            raise InvalidInput("request absolute token positions must fit I32")


@dataclass(frozen=True)
class RequestFlow:
    """Model-owned finite arrivals and per-request sequence lengths.

    The model keeps future route rows private. The policy receives only arrived
    requests and router-revealed routes. This workload declares independent
    request kernels, not online fused batching, token sampling or prefix sharing.
    """
    requests: tuple[Request, ...]
    routes: tuple[RouteTraceEntry, ...] = ()
    route_source: str | None = None

    def __post_init__(self):
        if not isinstance(self.requests,tuple) or not self.requests or any(not isinstance(r,Request) for r in self.requests):
            raise InvalidInput("RequestFlow.requests must be a nonempty immutable tuple of Request")
        if len({r.request_id for r in self.requests})!=len(self.requests):
            raise InvalidInput("RequestFlow request identities must be unique")
        if not isinstance(self.routes,tuple) or any(not isinstance(r,RouteTraceEntry) for r in self.routes):
            raise InvalidInput("RequestFlow.routes must be an immutable tuple of Route")
        if any(not isinstance(r.expert_ids,tuple) or r.scores is not None and not isinstance(r.scores,tuple) for r in self.routes):
            raise InvalidInput("route expert_ids and optional scores must be immutable tuples")
        if self.route_source is not None and (not isinstance(self.route_source,str) or not self.route_source.strip()):
            raise InvalidInput("RequestFlow.route_source must be nonempty or None")
        if self.routes and self.route_source is None:
            raise InvalidInput("RequestFlow routes require an explicit route_source")


def build_request_flow_inputs(costs, directory, model, hardware, workload):
    """Emit ordinary global-execution-0.2 inputs for the existing compiler."""
    if not isinstance(model,Model) or not isinstance(hardware,Hardware) or not isinstance(workload,RequestFlow):
        raise InvalidInput("request flow needs Model, Hardware and RequestFlow")
    if hardware.devices!=("cpu","gpu"):
        raise Unsupported("request flow currently requires one explicitly declared CPU and one GPU")
    if hardware.memory_mode == "separate" and hardware.transfer_bandwidth_bytes_per_second is None:
        raise Unsupported("request flow requires explicit DRAM/VRAM transfer bandwidth")
    if model.kind=="dense" and (workload.routes or workload.route_source is not None):
        raise InvalidInput("Dense request flow must not declare MoE routes")
    if model.kind=="moe":
        required={(r.request_id,f"layer{l}",p) for r in workload.requests for l in range(model.layers)
                  for p in range(r.prefill_tokens+r.decode_steps)}
        actual={(r.request_id,r.layer_id,r.absolute_position) for r in workload.routes}
        if len(actual)!=len(workload.routes) or actual!=required:
            raise InvalidInput("MoE request flow requires exactly one route per layer/request/token position")
        # Reuse the strict model route validator for expert IDs/top-k/scores.
        parse_route_trace([dict(request_id=r.request_id,layer_id=r.layer_id,
            absolute_position=r.absolute_position,expert_ids=list(r.expert_ids),
            **({"scores":list(r.scores)} if r.scores is not None else {})) for r in workload.routes],
            where="RequestFlow.routes")
    directory=Path(directory).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise InvalidInput("request flow input directory must be new or empty")
    version="m2-layer-0.6" if model.kind=="moe" else "m2-layer-0.5"
    documents={};layers={};streams=[];estimates=[]
    for index in range(model.layers):
        layer_id=f"layer{index}"
        doc=_layer_document(model,layer_id,version)
        documents[f"{layer_id}.layer.json"]=doc
        layers[layer_id]=parse_layer_document(doc)
    for request in workload.requests:
        steps=[]
        local=Workload(prefill_tokens=request.prefill_tokens,decode_steps=request.decode_steps,
            batch_size=1, routes=workload.routes,route_source=workload.route_source)
        for step_index in range(request.decode_steps+1):
            phase="prefill" if step_index==0 else "decode"
            step_id="prefill" if step_index==0 else f"decode{step_index-1}"
            position=0 if step_index==0 else request.prefill_tokens+step_index-1
            tokens=request.prefill_tokens if step_index==0 else 1
            entries=[]
            for layer_index in range(model.layers):
                layer_id=f"layer{layer_index}";layer=layers[layer_id]
                scope=f"{request.request_id}-{step_id}-{layer_id}"
                routes=tuple(r for r in workload.routes if r.request_id==request.request_id and
                    r.layer_id==layer_id and position<=r.absolute_position<position+tokens)
                invocation=_invocation_document(model,hardware,local,layer,scope,phase,position,tokens,step_index+1,routes)
                invocation["requests"][0]["request_id"]=request.request_id
                state=invocation["state_in"]["requests"][0]
                state["request_id"]=request.request_id
                for block in state["blocks"]:
                    block["id"]=f"kv:{layer_id}:{request.request_id}:{block['first_position']}"
                price_doc,details=costs._cost_document(model,hardware,local,layer,phase,tokens,position,routes)
                recorder=_RecordingCostTable(parse_cost_table(price_doc),hardware,costs)
                compile_invocation(layer,parse_invocation_document(invocation),recorder)
                price_doc["transfer_entries"]=[dict(route=route,storage_bytes=size,duration_ns=price.duration_ns,
                    provenance="synthetic_transfer") for (route,size),price in sorted(recorder.recorded.items())]
                compile_invocation(layer,parse_invocation_document(invocation),parse_cost_table(price_doc))
                documents[f"{scope}.json"]=invocation
                documents[f"{scope}.costs.json"]=price_doc
                estimates.extend(dict(request_id=request.request_id,invocation_id=scope,layer_id=layer_id,**row) for row in details)
                entries.append(dict(id=scope,invocation_file=f"{scope}.json",execution_file=None))
            gate_costs={device:dict(duration_ns=costs.compute_duration_ns(getattr(hardware,device),
                flops=0,traffic_bytes=(model.hidden_bytes_per_token+4*model.layers)*tokens),workspace_bytes=costs.workspace_bytes)
                for device in hardware.devices}
            steps.append(dict(id=f"{request.request_id}-{step_id}",
                input_mode="initial_supplied_hidden" if step_index==0 else "trace_supplied_next_token",
                gate_costs=gate_costs,invocations=entries))
        streams.append(dict(request_id=request.request_id,arrival_ns=request.arrival_ns,steps=steps))
    documents["model.global.json"]=dict(format_version="global-execution-0.2",id="analytical-request-flow",requests=streams)
    documents["environment.json"]=dict(format_version="environment-trace-0.1",events=[])
    metadata=dict(format_version="request-flow-inputs-0.1",provenance="synthetic_layer",
        cost_model="analytical_roofline",calibrated=False,hardware_measurements_used=False,
        generated_profile_required=False,numerical_llm_execution=False,model=asdict(model),hardware=asdict(hardware),
        workload=dict(requests=[asdict(r) for r in workload.requests],route_source=workload.route_source,
                      route_count=len(workload.routes)),macro_contract=version,
        batch_accounting="independent_request_kernels_exact_B1_no_fused_batch",
        token_inputs="trace_supplied_hidden_and_per_layer_I32_positions_with_explicit_materialization_costs_no_sampling_or_embedding",
        compute_estimates=estimates,
        limitations=["finite declared arrival trace, not a stochastic workload generator",
                     "independent B1 kernels, no fused dynamic-batch speedup",
                     "one compute slot per device and one shared transport/memory ledger",
                     "no multi-GPU, token sampling, prefix sharing or speculative decoding",
                     "uncalibrated analytical prices; not hardware performance validation"])
    if hardware.memory_mode == "shared":
        metadata["memory_contract"] = _memory_metadata(hardware)
        metadata["limitations"].append("same-buffer SHARE is a declared zero-time abstraction, not measured synchronization; no automatic shared-bandwidth contention correction")
    documents["analytical-metadata.json"]=metadata
    directory.mkdir(parents=True,exist_ok=True)
    for name,document in documents.items():
        _write(directory/name,document)
    return dict(model_file=directory/"model.global.json",environment_file=directory/"environment.json",
                model_kind="global_m2",cost_model="analytical_roofline",metadata=metadata)


__all__=["Request","RequestFlow"]
