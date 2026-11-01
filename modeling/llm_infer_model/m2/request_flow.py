"""Finite independent request lifecycles over the common global tensor runtime.

This is graph construction, not a serving scheduler or another simulator.
Requests are trace supplied, kernels retain exact B=1 prices, and later token
gates depend on that request alone. Optional mapping-owned P/D handoffs add
all prefill KV tensors as real first-decode gate operands.

This format defines input materialization and ordinary trace-token gates on
the CPU, requiring explicit CPU prices. A first-Decode handoff gate instead
uses its explicitly selected destination device and price. JSON key order is
never an execution-device choice.
"""
from copy import deepcopy
from dataclasses import replace

from ..tensor.spec import InvalidInput, Unsupported, ROLE_WEIGHT, ROUTE_ENDS, build_scenario
from ..tensor.environment import EnvironmentEvent

FORMAT_VERSION = "global-execution-0.2"


class RequestArrivalBoundary(EnvironmentEvent):
    """A model-owned clock boundary; it changes no physical resource state.

    Ordinary EnvironmentEvent deliberately rejects a no-op. This explicit
    private subtype has its own narrower invariant and enters the same event
    queue, so in-flight compute is interrupted at an arrival without fabricating
    a CPU availability transition or publishing a future timer to a policy.
    """
    def __post_init__(self):
        if type(self.t_ns) is not int or self.t_ns < 0:
            raise InvalidInput("request arrival boundary requires a nonnegative integer time")
        if any(getattr(self, field) is not None for field in self.__dataclass_fields__ if field != "t_ns"):
            raise InvalidInput("request arrival boundary cannot modify physical environment fields")


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")


def _identity(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in ":\r\n"):
        raise InvalidInput(f"{label} requires a nonempty identity without colon/newline")
    return value


def parse_request_flow(document):
    """Validate declarations without opening model, cost or routing files."""
    _exact(document, {"format_version", "id", "requests"}, "request flow")
    if document["format_version"] != FORMAT_VERSION:
        raise Unsupported("unsupported request-flow version")
    _identity(document["id"], "request flow id")
    if not isinstance(document["requests"], list) or not document["requests"]:
        raise InvalidInput("request flow requires a nonempty finite request list")
    requests, scopes = set(), set()
    for index, request in enumerate(document["requests"]):
        _exact(request, {"request_id", "arrival_ns", "steps"}, f"requests[{index}]")
        rid = _identity(request["request_id"], "request_id")
        if rid in requests:
            raise InvalidInput("duplicate request-flow request identity")
        requests.add(rid)
        if type(request["arrival_ns"]) is not int or request["arrival_ns"] < 0:
            raise InvalidInput("arrival_ns must be a nonnegative integer")
        if not isinstance(request["steps"], list) or not request["steps"]:
            raise InvalidInput("each request requires a nonempty finite step list")
        steps = set()
        for number, step in enumerate(request["steps"]):
            _exact(step, {"id", "input_mode", "gate_costs", "invocations"}, "request step")
            sid = _identity(step["id"], "step id")
            if sid in steps:
                raise InvalidInput("duplicate step identity within request")
            steps.add(sid)
            expected = "initial_supplied_hidden" if number == 0 else "trace_supplied_next_token"
            if step["input_mode"] != expected:
                raise Unsupported("request flow supports one complete prefill then decode; chunked prefill is not implemented")
            costs = step["gate_costs"]
            if not isinstance(costs, dict) or not costs or set(costs) - {"cpu", "gpu"}:
                raise InvalidInput("gate_costs requires explicit cpu/gpu exact prices, including initial input materialization")
            for device, price in costs.items():
                _exact(price, {"duration_ns", "workspace_bytes"}, f"gate_costs.{device}")
                if (type(price["duration_ns"]) is not int or price["duration_ns"] < 1 or
                        type(price["workspace_bytes"]) is not int or price["workspace_bytes"] < 0):
                    raise InvalidInput("gate duration must be positive and workspace nonnegative integers")
            if not isinstance(step["invocations"], list) or not step["invocations"]:
                raise InvalidInput("request step invocations must be nonempty")
            for entry in step["invocations"]:
                _exact(entry, {"id", "invocation_file", "execution_file"}, "request invocation")
                identity = _identity(entry["id"], "invocation scope id")
                if identity in scopes:
                    raise InvalidInput("request-flow invocation scope IDs must be globally unique")
                scopes.add(identity)
                if not isinstance(entry["invocation_file"], str) or not entry["invocation_file"]:
                    raise InvalidInput("invocation_file requires an explicit path")
                if entry["execution_file"] is not None and (not isinstance(entry["execution_file"], str) or not entry["execution_file"]):
                    raise InvalidInput("execution_file must be null or an explicit path")
    return deepcopy(document)


def normalize_handoffs(handoffs, request_ids):
    if not isinstance(handoffs, tuple):
        raise InvalidInput("handoffs must be an explicit tuple of (request_id, destination) pairs")
    result = {}
    for row in handoffs:
        if not isinstance(row, tuple) or len(row) != 2:
            raise InvalidInput("handoff requires a request/destination pair")
        request, destination = row
        if request not in request_ids or request in result:
            raise InvalidInput("handoff request must exist and be unique")
        if destination not in ("dram", "vram"):
            raise Unsupported("P/D handoff destination must be dram or vram in the single CPU/GPU runtime")
        result[request] = destination
    return result


def _gate_device(number, request_id, handoffs):
    if number == 1 and request_id in handoffs:
        return "cpu" if handoffs[request_id] == "dram" else "gpu"
    return "cpu"


def _require_gate_price(prices, device):
    if device not in prices:
        raise Unsupported(f"missing exact {device} token/input gate price; default gates are explicitly CPU in global-execution-0.2")
    return prices[device]


def compile_request_flow(document, source_path, sources, overrides, *, handoffs=(), visibility_mode):
    """Compose validated request subgraphs, then create one shared graph.

    _combine is reused strictly as a compiler helper; no per-request session,
    runtime, clock, inventory or capacity ledger is instantiated.
    """
    from .global_execution import _compile_leaf, _combine
    parsed = parse_request_flow(document)
    requested_handoffs = normalize_handoffs(handoffs, {row["request_id"] for row in parsed["requests"]})
    streams, all_scopes, layers = [], set(), {}
    common_layer_order = None
    for request in parsed["requests"]:
        rid, rounds = request["request_id"], []
        for number, step in enumerate(request["steps"]):
            leaves = []
            for entry in step["invocations"]:
                leaf = _compile_leaf(entry, source_path, sources, overrides, all_scopes)
                if tuple(row.request_id for row in leaf.invocation.requests) != (rid,):
                    raise InvalidInput("each request-flow leaf must contain exactly its owning request, using B=1 exact kernel prices")
                if (number == 0 and (leaf.invocation.phase != "prefill" or
                        leaf.invocation.requests[0].position_start != 0)):
                    raise Unsupported("request-flow starts with complete prefill at position zero; chunked prefill/initial external history is not supported")
                old_layer = layers.setdefault(leaf.layer.id, leaf.layer)
                if old_layer != leaf.layer:
                    raise InvalidInput("shared layer descriptor changed across request streams")
                leaves.append((entry["id"], leaf))
            layer_order = tuple(leaf.layer.id for _, leaf in leaves)
            if common_layer_order is None:
                common_layer_order = layer_order
            if layer_order != common_layer_order:
                raise InvalidInput("request flows must retain the same complete layer order")
            price = None
            if number:
                device = _gate_device(number, rid, requested_handoffs)
                selected_price = _require_gate_price(step["gate_costs"], device)
                if device not in leaves[0][1].invocation.devices:
                    raise Unsupported("token gate device is outside the declared hardware devices")
                price = dict(device=device, **selected_price)
            rounds.append((dict(id=f"{rid}-{step['id']}", input_mode=step["input_mode"],
                                gate_cost=price, invocations=step["invocations"]), leaves))
        normalized = dict(format_version="global-execution-0.1", id=f"{parsed['id']}-{rid}",
                          rounds=[raw for raw, _ in rounds])
        stream = _combine(normalized, rounds, sources, visibility_mode=visibility_mode)
        # Bind explicit source prices. The old compiler helper takes one gate
        # price; mapping's handoff picks its destination-specific alternative.
        wd, sd = deepcopy(stream.aggregate.workload_document), deepcopy(stream.aggregate.scenario_document)
        metadata = deepcopy(stream.aggregate.physical_boundary)
        from .expand import CompiledTask
        from .costs import MacroShape
        from ..tensor.spec import ROLE_INTERMEDIATE
        arrival_id = f"g:{rid}:request_arrival"
        first_scope = stream.invocation_scopes[0]
        hidden_input = first_scope.incoming_hidden[0]
        initial_prices = deepcopy(request["steps"][0]["gate_costs"])
        if set(initial_prices) - set(rounds[0][1][0][1].invocation.devices):
            raise Unsupported("request input costs include a device outside declared hardware")
        initial_device = "cpu"
        _require_gate_price(initial_prices, initial_device)
        def step_supplied_ports(step):
            prefixes = tuple(f"g:{entry['id']}:" for entry in step["invocations"])
            ports = []
            for tensor in wd["tensors"]:
                if (tensor.get("initial_locations") and tensor["role"] != ROLE_WEIGHT
                        and tensor["id"].startswith(prefixes)):
                    tensor.pop("initial_locations")
                    tensor["role"] = ROLE_INTERMEDIATE
                    ports.append(tensor["id"])
            return ports
        initial_ports = step_supplied_ports(request["steps"][0])
        for tensor in wd["tensors"]:
            if tensor["id"] == hidden_input:
                tensor.pop("initial_locations", None)
                tensor["role"] = ROLE_INTERMEDIATE
        wd["operations"].insert(0, dict(id=arrival_id, semantic_op="state_update",
            ggml_op="TRACE_SUPPLIED_REQUEST_INPUT", unary_op=None, op_params=[], inputs=[], output=hidden_input))
        initial_extras = [tid for tid in initial_ports if tid != hidden_input]
        if initial_extras:
            wd["operations"][0]["extra_outputs"] = initial_extras
        sd["costs"]["compute"][arrival_id] = {initial_device: initial_prices[initial_device]}
        sd["mapspace"]["allowed_devices"][arrival_id] = [initial_device]
        # Input was originally a supplied leaf, so give its produced port the
        # same explicit transfer prices from that exact leaf declaration.
        first_request = rounds[0][1][0][1].invocation.requests[0]
        initial_task = CompiledTask(rid, "request_input", "state_update", arrival_id, initial_device,
            MacroShape("prefill", 1, first_request.tokens, 0), hidden_input, None, ())
        metadata["row_execution"]["operation_stages"][arrival_id] = dict(
            layer_id=first_scope.layer_id, invocation_id=first_scope.id, execution_stage="request_input_materialization")
        handoff_records = []
        for number, step in enumerate(request["steps"]):
            if not number:
                continue
            oid = f"g:{rid}-{step['id']}:trace_next:{rid}"
            supplied = step_supplied_ports(step)
            gate_operation = next(row for row in wd["operations"] if row["id"] == oid)
            if supplied:
                gate_operation["extra_outputs"] = supplied
            prices = deepcopy(step["gate_costs"])
            if set(prices) - set(rounds[number][1][0][1].invocation.devices):
                raise Unsupported("token gate costs include a device outside declared hardware")
            default_device = _gate_device(number, rid, requested_handoffs)
            _require_gate_price(prices, default_device)
            sd["costs"]["compute"][oid] = {default_device: prices[default_device]}
            sd["mapspace"]["allowed_devices"][oid] = [default_device]
            if number == 1 and rid in requested_handoffs:
                destination = requested_handoffs[rid]
                device = "cpu" if destination == "dram" else "gpu"
                if device not in prices:
                    raise Unsupported(f"missing exact {device} token/handoff gate price")
                op = next(row for row in wd["operations"] if row["id"] == oid)
                prefill_kv = sorted(block.id for _, leaf in rounds[0][1]
                                    for block in leaf.new_blocks[rid])
                op["inputs"] = list(dict.fromkeys([*op["inputs"], *prefill_kv]))
                sd["costs"]["compute"][oid] = {device: prices[device]}
                sd["mapspace"]["allowed_devices"][oid] = [device]
                metadata["row_execution"]["operation_stages"][oid]["execution_stage"] = "prefill_decode_handoff"
                decode_prefixes = tuple(f"g:{entry['id']}:" for entry in step["invocations"])
                readers = [operation["id"] for operation in wd["operations"]
                    if operation["id"].startswith(decode_prefixes) and set(operation["inputs"]) & set(prefill_kv)]
                for reader in readers:
                    if sd["architecture"].get("memory_mode", "separate") == "shared":
                        # The gate consumes completed KV views at the declared
                        # handoff location. Readers may then use either device
                        # through an explicit SHARE of that same allocation.
                        continue
                    applicable = sd["costs"]["compute"][reader]
                    if device not in sd["mapspace"]["allowed_devices"][reader] or device not in applicable:
                        raise Unsupported("first Decode KV reader has no exact permitted price at the declared handoff destination")
                    # A handoff constrains actual first-Decode KV readers, not
                    # merely a policy default. Even custom device overrides
                    # must obey the resulting graph's allowed-device contract.
                    sd["mapspace"]["allowed_devices"][reader] = [device]
                    sd["costs"]["compute"][reader] = {device: applicable[device]}
                handoff_records.append(dict(request_id=rid, operation_id=oid, destination=destination,
                    hidden_tensor_id=op["inputs"][0], kv_tensor_ids=prefill_kv,
                    first_decode_scope_ids=[entry["id"] for entry in step["invocations"]],
                    kv_reader_operation_ids=readers,
                    semantics="all_prefill_hidden_and_all_layer_KV_ready_at_destination_before_trace_token_gate"))
        if rid in requested_handoffs and len(request["steps"]) < 2:
            raise InvalidInput("handoff requires at least one declared decode step")
        # Gate task metadata must agree with the fixed handoff/default device.
        # Native MoE group tasks use a batch pseudo-identity. Here every leaf
        # was proved B=1 and owned by this stream, so public scheduling labels
        # must identify its actual request. IDs, prices and routed work rows
        # remain unchanged; legacy batched invocation semantics are untouched.
        tasks = (initial_task, *tuple(replace(task, request_id=rid,
            device=next(iter(sd["costs"]["compute"][task.operation_id]))
                   if task.kind == "state_update" else task.device)
            for task in stream.aggregate.tasks))
        aggregate = replace(stream.aggregate, workload_document=wd, scenario_document=sd,
            physical_boundary=metadata, tasks=tasks, scenario=build_scenario(wd, sd, where="request_flow_stream"))
        streams.append((request, replace(stream, aggregate=aggregate,
            invocation_scopes=tuple(replace(scope, dependency_operations=(arrival_id, *scope.dependency_operations))
                                    for scope in stream.invocation_scopes),
            gate_operations=((arrival_id, ()), *stream.gate_operations)), handoff_records))
    if set(overrides) - all_scopes:
        raise InvalidInput("execution override references an absent request-flow invocation")
    return _merge_streams(parsed, streams, sources, visibility_mode)


def _merge_streams(document, streams, sources, visibility_mode):
    from .global_execution import CompiledGlobalExecution, verify_global_sources
    first = streams[0][1].aggregate
    tensors, operations, compute = {}, [], {}
    transfer, allowed = {route: {} for route in ROUTE_ENDS}, {route: set() for route in ROUTE_ENDS}
    goals, tasks, scopes, gates = {}, [], [], []
    row_scopes, stages, subweights, children, partitions, views, pipelines = {}, {}, {}, {}, {}, [], []
    global_fields = {key: {} for key in ("weight_owners", "tensor_owners", "hidden_port_successors", "declared_parent_versions")}
    kv, weights, hidden, experts, request_records, handoffs = set(), set(), set(), {}, [], []
    architecture = first.scenario_document["architecture"]
    for request, stream, records in streams:
        aggregate = stream.aggregate
        if aggregate.scenario_document["architecture"] != architecture:
            raise InvalidInput("all request flows must share one hardware contract")
        for tensor in aggregate.workload_document["tensors"]:
            previous = tensors.setdefault(tensor["id"], deepcopy(tensor))
            if previous != tensor:
                raise InvalidInput("shared request-flow tensor definition/initial placement changed")
        operations.extend(deepcopy(aggregate.workload_document["operations"]))
        compute.update(deepcopy(aggregate.scenario_document["costs"]["compute"]))
        for route, rows in aggregate.scenario_document["costs"]["transfer"].items():
            for tid, price in rows.items():
                old = transfer[route].setdefault(tid, deepcopy(price))
                if old != price:
                    raise InvalidInput("shared request-flow tensor has conflicting exact transfer prices")
                allowed[route].add(tid)
        if architecture.get("memory_mode", "separate") == "shared":
            for route in ("dram_to_vram", "vram_to_dram"):
                allowed[route].update(aggregate.scenario_document["mapspace"]["allowed_transfers"].get(route, ()))
        local_kv = set(aggregate.physical_boundary["global_execution"]["kv_ids"])
        for tid, location in aggregate.scenario_document["goal"]["output_locations"].items():
            if tid in local_kv:
                continue  # Retired request KV is releasable, not a final delivery.
            old = goals.setdefault(tid, location)
            if old != location:
                raise InvalidInput("shared request-flow persistent goal changed")
        boundary = aggregate.physical_boundary
        row = boundary["row_execution"]
        row_scopes.update(deepcopy(row["tensor_row_scopes"]))
        stages.update(deepcopy(row["operation_stages"]))
        for part in row["subweights"]:
            old = subweights.setdefault(part["parent_weight_id"], deepcopy(part))
            if old != part:
                raise InvalidInput("shared request-flow subweight geometry changed")
        ffn = boundary["ffn_execution"]
        for parent, values in ffn["weight_children"].items():
            old = children.setdefault(parent, deepcopy(values))
            if old != values:
                raise InvalidInput("shared request-flow weight representation changed")
        for part in ffn["weight_partitions"]:
            old = partitions.setdefault(part["tensor_id"], deepcopy(part))
            if old != part:
                raise InvalidInput("shared request-flow weight partition changed")
        views.extend(deepcopy(ffn.get("weight_views", [])))
        pipelines.extend(deepcopy(boundary["execution_pipelines"]))
        global_ = boundary["global_execution"]
        for key, values in global_fields.items():
            for identity, value in global_[key].items():
                old = values.setdefault(identity, deepcopy(value))
                if old != value:
                    raise InvalidInput(f"shared request-flow {key} changed")
        kv.update(global_["kv_ids"]); weights.update(global_["weight_ids"]); hidden.update(global_["hidden_ports"])
        for expert in global_["experts"]:
            key = (expert["layer_id"], expert["expert_id"])
            old = experts.setdefault(key, deepcopy(expert))
            if old != expert:
                raise InvalidInput("shared request-flow expert catalogue changed")
        tasks.extend(aggregate.tasks); scopes.extend(stream.invocation_scopes); gates.extend(stream.gate_operations)
        handoffs.extend(records)
        request_records.append(dict(request_id=request["request_id"], arrival_ns=request["arrival_ns"],
            operation_ids=[task.operation_id for task in aggregate.tasks],
            scope_ids=[scope.id for scope in stream.invocation_scopes],
            kv_tensor_ids=sorted(local_kv), input_operation_id=f"g:{request['request_id']}:request_arrival",
            final_hidden_ids=sorted(set(aggregate.scenario_document["goal"]["output_locations"]) - local_kv - weights)))
    wd = dict(schema_version=first.workload_document["schema_version"], id=document["id"],
        origin=dict(kind="synthetic_layer", ggml_version=None,
                    sample="finite independent request flow; charged arrival input materialization; no numeric token generation"),
        tensors=list(tensors.values()), operations=operations, outputs=list(goals))
    sd = deepcopy(first.scenario_document)
    sd.update(id=document["id"], goal=dict(output_locations=goals),
        costs=dict(source="request-flow exact leaf and token-gate prices", compute=compute,
                   transfer={route: prices for route, prices in transfer.items() if prices}))
    sd["mapspace"].update(allowed_devices={oid: list(prices) for oid, prices in compute.items()},
        allowed_transfers={route: sorted(ids) for route, ids in allowed.items() if ids}, transfer_dependencies={})
    boundary = dict(row_execution=dict(tensor_row_scopes=row_scopes, operation_stages=stages, subweights=list(subweights.values())),
        ffn_execution=dict(weight_children=children, weight_partitions=list(partitions.values()), weight_views=views),
        execution_pipelines=pipelines,
        request_flow=dict(format_version=FORMAT_VERSION, requests=request_records, handoffs=handoffs,
            input_memory_semantics="charged_trace_input_producer_allocates_on_compute_after_arrival; no_future_input_reservation",
            kernel_accounting="independent_exact_B1_request_kernels; no_fused_batch",
            gate_device_contract="input_and_ordinary_trace_token_gates_explicit_CPU; first_decode_handoff_gate_uses_declared_destination_device",
            chunked_prefill=False, speculative_decoding=False),
        global_execution=dict(format_version=FORMAT_VERSION, invocation_scope_ids=[scope.id for scope in scopes],
            execution_bindings=[dict(invocation_id=scope.id, variant=scope.execution_variant) for scope in scopes],
            kv_ids=sorted(kv), weight_ids=sorted(weights), hidden_ports=sorted(hidden), experts=list(experts.values()),
            **global_fields, numeric_sampling=False, numeric_embedding=False, global_performance_calibrated=False,
            visibility_mode=visibility_mode, continuation="one_runtime_independent_request_tokens_no_batch_round_barrier"))
    aggregate = replace(first, workload_document=wd, scenario_document=sd, scenario=build_scenario(wd, sd, where="request_flow"),
                        tasks=tuple(tasks), physical_boundary=boundary, costs_source=sd["costs"]["source"])
    result = CompiledGlobalExecution(deepcopy(document), aggregate, tuple(scopes), tuple(sources.values()), tuple(gates), visibility_mode)
    verify_global_sources(result)
    return result
