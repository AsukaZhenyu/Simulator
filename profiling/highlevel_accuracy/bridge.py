"""Export the actual compiled Python strategy, including full operands and actions.

This adapter supports complete F32 Dense invocations and homogeneous layers in
one inference round. Both the
original three macros and the public AttentionStages / FFNPartition transforms
are executable. Other transforms, overlapping launches, services, disturbances,
and opaque/non-F32 models are rejected, rather than assigned surrogate kernels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

FORMAT = "highlevel-accuracy-program-v1"
BOUNDARY = "independent_contiguous_f32_ports_full_dense_norm_rope_residual_v1"
RUNTIME_FLAGS = {"GGML_CUDA_DISABLE_GRAPHS":"1", "NVIDIA_TF32_OVERRIDE":"0",
                 "matmul_precision":"GGML_PREC_F32"}
KINDS = {"projection", "attention", "dense_ffn", "attention_qk", "attention_softmax",
         "attention_pv", "attention_output", "ffn_norm", "ffn_gate", "ffn_up",
         "ffn_swiglu", "ffn_down", "ffn_sum", "ffn_residual"}

# Explicit policy authored in the high-level frontend. Export never silently
# serializes an overlapping strategy after compilation.
from tensor_mapping.python_frontend import RuntimePolicy, DEFER
from tensor_mapping.causal import Decision


class SerialExecutionPolicy(RuntimePolicy):
    def decide(self, observation):
        return Decision("ADVANCE") if observation.running else DEFER


class BridgeError(ValueError):
    pass


def canonical(value):
    def array_identity(item):
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            _require(array.dtype in (np.dtype("float32"), np.dtype("int32")),
                     "numeric initializer arrays require F32 or I32")
            _require(np.isfinite(array).all(), "numeric initializer arrays must be finite")
            return {"__numeric_array__": dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest())}
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=array_identity)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def frontend_sha(value):
    """Hash a frontend document exactly as ``tensor_mapping.python_frontend``.

    The compiler hashes its own IR with ``_json``, which keeps non-ASCII
    characters unescaped.  Hashing the same document here with the default
    (escaped) form changes the bytes on one side only, so the compile-identity
    check below reported a changed document for any case whose path or label is
    not ASCII.  The escaped and unescaped forms of ASCII content are identical,
    so this returns the same digest for every document recorded so far; it is
    used only for that one check, because program digests are taken over
    documents that may contain non-ASCII text.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _require(ok, message):
    if not ok:
        raise BridgeError(message)


def _values(tensor, seed, gamma=False):
    # Tensor identity, rather than inventory order, controls input contents.
    local = int(hashlib.sha256(f"{seed}:{tensor['id']}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(local)
    values = rng.uniform(-0.5, 0.5, math.prod(tensor["ne"])).astype(np.float32)
    return ((1 + 0.4 * values) if gamma else values / math.sqrt(tensor["ne"][0])
            if tensor["role"] == "weight" else values).astype(np.float32)


def _invocation_contexts(compiled):
    """Read exact source-bound leaf metadata without inventing global seams.

    The native executor has one physical configuration. Multiple invocations are
    consequently admitted only for one round of homogeneous Dense layers. Trace
    token gates still require a separate numeric token-input implementation.
    """
    from llm_infer_model.m2.expand import load_and_compile
    from llm_infer_model.m2.global_execution import verify_global_sources
    scopes = compiled.model.invocation_scopes
    if len(scopes) == 1:
        return {scopes[0].id: compiled.model.aggregate}
    verify_global_sources(compiled.model)
    doc = compiled.model.document
    _require(doc.get("format_version") == "global-execution-0.1" and len(doc.get("rounds", [])) == 1,
             "multi-round trace token gates have no numeric native implementation")
    _require(not compiled.model.gate_operations,
             "multi-round trace token gates have no numeric native implementation")
    root = Path(compiled.model.sources[0]["path"]).parent
    rows = doc["rounds"][0]["invocations"]
    _require([r["id"] for r in rows] == [s.id for s in scopes], "global invocation order changed")
    contexts = {r["id"]: load_and_compile(root / r["invocation_file"]) for r in rows}
    from llm_infer_model.m2.hybrid_physical import HybridDensePhysical
    from llm_infer_model.m2.full_physical import FullDensePhysical
    _require(all(isinstance(leaf.layer.physical, (HybridDensePhysical, FullDensePhysical))
                 for leaf in contexts.values()), "a complete Dense physical boundary is required")
    base = next(iter(contexts.values())).layer.physical
    for scope in scopes:
        leaf = contexts[scope.id]
        _require(leaf.layer.id == scope.layer_id, "scope layer differs from source-bound invocation")
        _require(type(leaf.layer.physical) is type(base) and leaf.layer.physical.config == base.config
                 and leaf.layer.physical.threads == base.threads
                 and getattr(leaf.layer.physical, "cuda_device", None) == getattr(base, "cuda_device", None),
                 "native multilayer boundary requires homogeneous physical configuration, threads and CUDA device")
    return contexts


class _InitialDiagnosticSnapshotSink:
    """Keep one initial diagnostic, releasing each computed later snapshot.

    SimulationSession computes ``_snapshot(...)`` before calling ``append``.
    Its state transitions, event log and peak counters therefore still execute
    unchanged. Only the historical ``EvaluationResult.states`` output shrinks.
    """
    __slots__ = ("initial", "append_count")

    def __init__(self, initial):
        from llm_infer_model.tensor.engine import StateSnapshot
        _require(type(initial) is StateSnapshot, "explicit initial StateSnapshot required")
        self.initial = initial
        self.append_count = 0

    def append(self, snapshot):
        _require(type(snapshot) is type(self.initial), "diagnostic snapshot computation changed type")
        self.append_count += 1
        # The supplied snapshot and its tensor ledgers are intentionally not retained.

    def __iter__(self):
        return iter((self.initial,))

    def __len__(self):
        return 1


def _install_profiling_snapshot_sink(session):
    """Omit diagnostic history only for an unstarted ordinary causal runtime.

    This private profiling helper does not change Modeling's default recorder.
    Known adapters are unwrapped explicitly; service subclasses are rejected.
    """
    from tensor_mapping.composition_observation import CompositionSession
    from llm_infer_model.m2.causal import CausalM2Session
    from llm_infer_model.tensor.environment import SimulationSession

    while isinstance(session, CompositionSession):
        session = session._session
    _require(isinstance(session, CausalM2Session), "known causal profiling adapter required")
    runtime = session._runtime
    _require(type(runtime) is SimulationSession,
             "profiling diagnostic sink requires ordinary SimulationSession without services")
    snapshots = runtime._snapshots
    _require(not runtime._actions and not runtime._events
             and isinstance(snapshots, list) and len(snapshots) == 1,
             "install profiling diagnostic sink before any action with one initial snapshot")
    sink = _InitialDiagnosticSnapshotSink(snapshots[0])
    runtime._snapshots = sink
    return sink


def _lightweight_actions(plan):
    """Run the identical causal driver without retaining full observations.

    This profiling-only export does not implement a new policy or scheduler.
    Observation objects still reach the original policy and command validator;
    only their debug hashes, long-lived DecisionRecords and historical runtime
    diagnostics are omitted. The original snapshot computation still runs.
    """
    from tensor_mapping.causal import compile_command
    from tensor_mapping.causal_cli import verify_sources
    from tensor_mapping.python_frontend import _PlanPolicy
    from tensor_mapping.unified_strategy import create_session
    from llm_infer_model.tensor.runtime import dispatch_command

    compiled = plan._compiled
    verify_sources(compiled.sources)
    session = create_session(compiled)
    _install_profiling_snapshot_sink(session)
    policy = _PlanPolicy(compiled, plan._bindings, plan._runtime_spec, plan._preparations)
    budget = compiled.document["max_decisions"]
    _require(type(budget) is int and budget >= 1, "positive causal decision budget required")
    actions = []
    for index in range(budget):
        observation = session.observe()
        if observation.complete or observation.error_code:
            break
        decision = policy(observation)
        _require(decision is not None, "causal policy stopped before completing the graph")
        command = compile_command(decision, observation)
        action = asdict(command.action)
        _require(action["kind"] in {"COMPUTE", "COPY", "EVICT", "ADVANCE"}, "unsupported native action")
        _require(not (action["kind"] in {"COMPUTE", "COPY"} and observation.running),
                 "overlapping launch is unsupported; author a serial RuntimePolicy explicitly")
        step = dispatch_command(session, command, source_index=index)
        _require(step.ok, "causal command rejected: " + str(step.error_message))
        actions.append(action)
    final = session.observe()
    policy.observe_final(final)
    evaluation = session.result().evaluation
    _require(final.complete and not final.error_code and evaluation.status == "valid",
             "compiled ModelPlan must complete successfully before export")
    return evaluation, actions


def export_plan(plan, *, seed=1234, numeric_arrays=False, element_limit=None, lightweight_actions=False):
    """Freeze a CompiledPlan's real graph and accepted serial action trace.

    The original simulation costs remain labelled as their original source.
    They are NOT baseline measurements. Use freeze_profile / simulate_frozen for
    measured duration prediction, with a separate subsequently collected replay.
    """
    from tensor_mapping.python_frontend import CompiledPlan
    from tensor_mapping.unified_strategy import create_session, runtime_fingerprint
    from llm_infer_model.m2.hybrid_physical import HybridDensePhysical
    from llm_infer_model.m2.full_physical import FullDensePhysical

    _require(isinstance(plan, CompiledPlan), "export requires the public compiled ModelPlan")
    _require(plan._runtime_spec is None or plan._runtime_spec.factory is SerialExecutionPolicy,
             "unsupported runtime policy: this bridge requires static choices plus explicit serial execution")
    _require(type(seed) is int and 0 <= seed < 2**32, "seed must be uint32")
    _require(type(numeric_arrays) is bool, "numeric_arrays must be a boolean")
    _require(type(lightweight_actions) is bool, "lightweight_actions must be a boolean")
    _require(element_limit is None or (type(element_limit) is int
        and 1 <= element_limit <= 128_000_000), "explicit tensor element limit must be within 1..128000000")
    compiled = plan._compiled
    scopes = getattr(compiled.model, "invocation_scopes", ())
    _require(compiled.global_model and len(scopes) >= 1,
             "bridge requires explicit global_m2 invocation scopes")
    contexts = _invocation_contexts(compiled)
    _require(not compiled.environment and compiled.services is None and not compiled.reclaim_requests,
             "disturbances, transport services and reclamation are unsupported")
    aggregate = compiled.model.aggregate
    physical = aggregate.layer.physical
    _require(isinstance(physical, (HybridDensePhysical, FullDensePhysical)),
             "a complete Dense physical boundary is required")
    _require(all(v == "f32" for v in physical.config["weight_dtypes"].values()),
             "this numerical bridge requires F32 weights")
    graph = create_session(compiled).execution_document()
    _require(all(task["kind"] in KINDS for task in graph["tasks"]), "unsupported execution transform")
    pipeline = (graph.get("boundary") or {}).get("execution_pipelines", [])
    for row in pipeline:
        for transform in row["pipeline"]["transforms"]:
            _require(transform["format_version"] in {"attention-stages-0.1", "ffn-execution-0.1"},
                     "unsupported transform: no surrogate execution is permitted")
    if lightweight_actions:
        evaluation, actions = _lightweight_actions(plan)
    else:
        result = plan.run()
        run = result[0]
        _require(run.evaluation.status == "valid" and run.termination_reason == "goal_reached",
                 "compiled ModelPlan must complete successfully before export")
        evaluation = run.evaluation
        actions = []
        for record in run.decisions:
            _require(record.accepted and record.action is not None, "rejected/control decisions cannot be exported")
            action = asdict(record.action)
            _require(action["kind"] in {"COMPUTE", "COPY", "EVICT", "ADVANCE"}, "unsupported native action")
            _require(not (action["kind"] in {"COMPUTE", "COPY"} and record.observation.running),
                     "overlapping launch is unsupported; author a serial RuntimePolicy explicitly")
            actions.append(action)
    tensors = {t["id"]: t for t in graph["workload"]["tensors"]}
    allowed_elements = element_limit if element_limit is not None else (4_000_000 if len(scopes) == 1 else 16_000_000)
    _require(sum(math.prod(t["ne"]) for t in tensors.values()) <= allowed_elements,
             "bridge tensor envelope exceeds declared element limit")
    scope_for_operation = {oid: scope.id for scope in scopes for oid in scope.operation_ids}
    _require(len(scope_for_operation) == len(graph["tasks"]), "native scope inventory must cover exact tasks")
    weights_by_scope = {sid: {role: f"w:{leaf.layer.id}:{name}"
                       for group in leaf.layer.physical.weight_groups.values() for role, name in group.items()}
                       for sid, leaf in contexts.items()}
    tasks = {t["operation_id"]: t for t in graph["tasks"]}
    ops = {op["id"]: op for op in graph["workload"]["operations"]}
    requests = {(sid, r.request_id): r for sid, leaf in contexts.items() for r in leaf.invocation.requests}
    states = {(sid, r.request_id): r for sid, leaf in contexts.items() for r in leaf.state_in.requests}
    projection = {(scope_for_operation[t["operation_id"]], t["request_id"]): ops[t["operation_id"]]
                  for t in graph["tasks"] if t["kind"] == "projection"}
    position_inputs = {}
    for key, op in projection.items():
        positions = [tid for tid in op["inputs"] if tensors[tid]["dtype"] == "GGML_TYPE_I32"]
        _require(len(positions) == 1, "projection requires one explicit position input")
        request = requests[key]
        position_inputs[positions[0]] = list(range(request.position_start, request.position_start + request.tokens))
    norm_weights = {weights.get(r) for weights in weights_by_scope.values()
                    for r in ("input_norm", "post_attention_norm", "q_norm", "k_norm")}
    partition = {p["tensor_id"]: p for p in (graph.get("boundary") or {}).get("ffn_execution", {}).get("weight_partitions", [])}
    bindings = []
    initial_values = {}
    for tid, tensor in tensors.items():
        if not tensor.get("initial_locations"):
            continue
        if tensor["dtype"] == "GGML_TYPE_I32":
            _require(tid in position_inputs, "unrecognized position input")
            initial_values[tid] = np.asarray(position_inputs[tid], dtype=np.int32) if numeric_arrays else position_inputs[tid]
        elif tid in partition:
            continue
        else:
            gamma = tid in norm_weights
            values = _values(tensor, seed, gamma)
            initial_values[tid] = values if numeric_arrays else values.tolist()
    # Immutable partition leaves are exact slices of the SAME full parent weights.
    parents = {}
    for tid, part in partition.items():
        parent_id = part["source_tensor_id"]
        role = part["stage"]
        if parent_id not in parents:
            ne = [physical.config["intermediate"], physical.config["hidden"], 1, 1] if role == "down" else [physical.config["hidden"], physical.config["intermediate"], 1, 1]
            parents[parent_id] = _values(dict(id=parent_id, ne=ne, role="weight"), seed).reshape(ne[1], ne[0])
        matrix = parents[parent_id]
        leaf = matrix[:, part["start"]:part["end"]] if role == "down" else matrix[part["start"]:part["end"], :]
        initial_values[tid] = leaf.ravel() if numeric_arrays else leaf.ravel().tolist()
    for oid, op in ops.items():
        task = tasks[oid]
        sid = scope_for_operation[oid]
        leaf = contexts[sid]
        weights = weights_by_scope[sid]
        request_key = (sid, task["request_id"])
        request = requests[request_key]
        state = states[request_key]
        _require(tuple(b.first_position for b in state.blocks) == tuple(sum(b.tokens for b in state.blocks[:i]) for i in range(len(state.blocks)))
                 and sum(b.tokens for b in state.blocks) == request.position_start,
                 "only complete contiguous history beginning at absolute position zero is supported")
        qkv = [projection[request_key]["output"], *projection[request_key].get("extra_outputs", [])]
        p = dict(zip(("q", "k", "v", "gate"), qkv))
        history = [asdict(b) for b in state.blocks]
        new = [asdict(b) for b in leaf.new_blocks[task["request_id"]]]
        kind = task["kind"]
        operands, outputs = {}, {"result": op["output"]}
        if kind == "projection":
            operands = {r: weights[r] for r in physical.weight_groups["projection"]}
            values = [tid for tid in op["inputs"] if tensors[tid]["role"] != "weight"]
            operands.update(hidden=values[0], positions=values[1])
            outputs = {r: p[r] for r in p}
        elif kind == "dense_ffn":
            operands = {r: weights[r] for r in physical.weight_groups["dense_ffn"]}
            operands["hidden"] = next(tid for tid in op["inputs"] if tensors[tid]["role"] != "weight")
        elif kind == "attention":
            operands = {r: p[r] for r in p}
            hidden_input = next(tid for tid in projection[request_key]["inputs"]
                                if tensors[tid]["role"] != "weight" and tensors[tid]["dtype"] == "GGML_TYPE_F32")
            operands.update(output=weights["output"], hidden=hidden_input)
        elif kind == "attention_qk":
            operands = dict(q=op["inputs"][0], k=op["inputs"][1])
        elif kind == "attention_softmax":
            operands = dict(scores=op["inputs"][0])
        elif kind == "attention_pv":
            operands = dict(probabilities=op["inputs"][0], k=op["inputs"][1], v=op["inputs"][2])
        elif kind == "attention_output":
            operands = dict(context=op["inputs"][0], output=weights["output"], hidden=op["inputs"][-1])
            if "gate" in p:
                operands["gate"] = p["gate"]
        else:
            operands = {f"x{i}": tid for i, tid in enumerate(op["inputs"])}
        uses_history = kind in {"attention", "attention_qk", "attention_pv"}
        append_kv = kind in {"attention", "attention_pv"}
        _require(set(operands.values()) | ({b["id"] for b in history} if uses_history else set()) == set(op["inputs"]),
                 f"exact operand binding failed for {oid}")
        _require(set(outputs.values()) | ({b["id"] for b in new} if append_kv else set()) == {op["output"], *op.get("extra_outputs", [])},
                 f"exact output binding failed for {oid}")
        bindings.append(dict(operation_id=oid, kind=kind, operands=operands, outputs=outputs,
            shape=task["shape"], history=history if uses_history else [], new_kv=new if append_kv else [],
            position_start=request.position_start, op_params=op["op_params"]))
    program = dict(format_version=FORMAT, boundary_contract=BOUNDARY,
        frontend_ir=plan.ir_document(), frontend_sha256=compiled.frontend_sha256,
        simulator_runtime_sha256=runtime_fingerprint(),
        graph=graph, physical=asdict(physical), layer_id=aggregate.layer.id,
        actions=actions, bindings=bindings, initial_values=initial_values,
        required_runtime_flags=RUNTIME_FLAGS,
        seed=seed, original_simulation=dict(makespan_ns=evaluation.makespan_ns,
            cost_source=graph["scenario"]["costs"]["source"], measured_accuracy_claimed=False),
        limitations=["one_invocation" if len(scopes) == 1 else "one_round_homogeneous_dense_layers", "serial_launches_only", "F32_only", "no_services_or_disturbances",
                    "native_backing_preallocated_eviction_is_logical_not_memory_measurement"])
    if len(scopes) > 1:
        program["invocation_scopes"] = [asdict(scope) for scope in scopes]
        program["layer_weight_bindings"] = weights_by_scope
        program["source_bound_leaf_files"] = list(compiled.model.sources)
    if lightweight_actions:
        program["export_trace_recording"] = "profiling_actions_only_same_causal_driver"
    program["identity"] = {key: sha(program[key]) for key in ("frontend_ir", "graph", "physical", "actions", "bindings", "initial_values", "simulator_runtime_sha256")}
    if len(scopes) > 1:
        for key in ("invocation_scopes", "layer_weight_bindings", "source_bound_leaf_files"):
            program["identity"][key] = sha(program[key])
    program["program_sha256"] = sha(program)
    validate_program(program)
    return program


def validate_program(program):
    _require(program.get("format_version") == FORMAT and program.get("boundary_contract") == BOUNDARY,
             "unsupported program/boundary contract")
    _require(program.get("required_runtime_flags") == RUNTIME_FLAGS, "unsupported runtime/precision contract")
    copy = deepcopy(program)
    digest = copy.pop("program_sha256", None)
    _require(digest == sha(copy), "program identity changed")
    for key, value in program["identity"].items():
        _require(sha(program[key]) == value, f"{key} identity changed")
    _require(program["frontend_sha256"] == frontend_sha(program["frontend_ir"]), "frontend compile identity changed")
    ops = {op["id"]: op for op in program["graph"]["workload"]["operations"]}
    _require(len(ops) == len(program["bindings"]), "every compiled graph operation must have one native binding")
    seen = set()
    for binding in program["bindings"]:
        oid = binding["operation_id"]
        _require(oid in ops and oid not in seen and binding["kind"] in KINDS, "unsupported/duplicate binding")
        seen.add(oid)
        op = ops[oid]
        _require(binding["kind"] == op["semantic_op"] and binding["op_params"] == op["op_params"], "binding operation differs from compiled graph")
        _require(op["ggml_op"] == "M2_"+binding["kind"].upper() and op["unary_op"] is None,
                 "unsupported lowered operation contract")
        task = next(t for t in program["graph"]["tasks"] if t["operation_id"] == oid)
        _require(binding["shape"] == task["shape"], "binding physical shape differs from compiled task")
        _require(set(binding["operands"].values()) | {b["id"] for b in binding["history"]} == set(op["inputs"]), "binding operands differ from compiled graph")
        _require(set(binding["outputs"].values()) | {b["id"] for b in binding["new_kv"]} == {op["output"], *op.get("extra_outputs", [])}, "binding ports differ from compiled graph")
    compute = [a["target_id"] for a in program["actions"] if a["kind"] == "COMPUTE"]
    _require(len(compute) == len(ops) and set(compute) == set(ops), "action trace must execute the exact complete graph")
    tensors = {t["id"]: t for t in program["graph"]["workload"]["tensors"]}
    _require(set(program["initial_values"]) == {tid for tid, t in tensors.items() if t.get("initial_locations")},
             "initial data must match exactly the compiler's initial tensor inventory")
    for t in tensors.values():
        _require(t["dtype"] in {"GGML_TYPE_F32", "GGML_TYPE_I32"}, "unsupported tensor dtype")
        stride = 4
        for dim, actual in zip(t["ne"], t["nb"]):
            _require(type(dim) is int and dim > 0 and stride == actual, "tensor must be exactly contiguous")
            stride *= dim
        _require(stride == t["storage_bytes"], "physical tensor storage boundary changed")
        _require(set(t.get("initial_locations", [])) <= {"dram", "vram"}, "SSD tensors unsupported")
        if t.get("initial_locations"):
            _require(len(program["initial_values"].get(t["id"], [])) == math.prod(t["ne"]), "initial data shape mismatch")
    from llm_infer_model.m2.dense_contract import validate_config, weight_shapes
    config = validate_config(program["physical"]["config"])
    _require(all(dtype == "f32" for dtype in config["weight_dtypes"].values()), "unsupported physical weight dtype")
    def expect(tid, shape):
        _require(tid in tensors and tuple(tensors[tid]["ne"]) == (*shape, *(1 for _ in range(4-len(shape)))),
                 "tensor shape differs from exact physical operation boundary: "+tid)
    for b in program["bindings"]:
        kind, x, out = b["kind"], b["operands"], b["outputs"]
        n, h = b["shape"]["tokens"], config["hidden"]
        _require(b["shape"]["batch"] == 1 and n > 0, "native physical boundary requires batch one")
        if kind == "projection":
            expect(x["hidden"],(h,n));expect(x["positions"],(n,))
            for role in set(x)-{"hidden","positions"}:expect(x[role],weight_shapes(config)[role])
            for role in ("q","k","v"):expect(out[role],(config["head_dim"],n,config["heads"] if role=="q" else config["kv_heads"]))
            if "gate" in out:expect(out["gate"],(config["heads"]*config["head_dim"],n))
        elif kind in {"ffn_gate","ffn_up","ffn_down"}:
            w, activation = tensors[x["x0"]]["ne"], tensors[x["x1"]]["ne"]
            interval = _ffn_view(b, tensors, config["intermediate"])
            reduction, outputs = w[:2]
            if interval:
                start,end=interval
                boundary=(program["graph"].get("boundary") or {}).get("ffn_execution") or {}
                views=boundary.get("weight_views",[])
                _require(any(
                    v["source_tensor_id"]==x["x0"] and v["stage"]==kind.removeprefix("ffn_")
                    and (v["start"],v["end"])==interval and v.get("allocation") is False
                    and v.get("storage_bytes")==0 and v.get("copy_contract")=="complete_parent_tensor" for v in views),
                    "full parent slice missing explicit physical view contract")
                if kind=="ffn_down":reduction=end-start
                else:outputs=end-start
            _require(reduction == activation[0] and w[2:] == [1,1], "FFN matrix reduction dimension mismatch")
            expect(out["result"],(outputs,n))
        elif kind == "dense_ffn":
            expect(x["hidden"],(h,n));expect(out["result"],(h,n))
            for role in ("post_attention_norm","gate","up","down"):expect(x[role],weight_shapes(config)[role])
        elif kind == "ffn_norm":expect(x["x0"],(h,));expect(x["x1"],(h,n));expect(out["result"],(h,n))
    scopes = program.get("invocation_scopes", [])
    declared = ((program["graph"].get("boundary") or {}).get("global_execution") or {}).get("invocation_scope_ids", [])
    if len(declared) > 1:
        _require([s["id"] for s in scopes] == declared, "multilayer scope order/inventory differs from compiler")
        _require(len({s["round_id"] for s in scopes}) == 1, "multi-round native gates unsupported")
        _require(len({s["layer_id"] for s in scopes}) == len(scopes), "each native round needs distinct layer identities")
        all_ids = [oid for s in scopes for oid in s["operation_ids"]]
        _require(len(all_ids) == len(ops) and set(all_ids) == set(ops), "multilayer operation scope inventory differs")
        weight_bindings = program.get("layer_weight_bindings", {})
        _require(set(weight_bindings) == set(declared), "multilayer weight binding scope inventory differs")
        bindings = {b["operation_id"]: b for b in program["bindings"]}
        for index, scope in enumerate(scopes):
            scoped = [bindings[oid] for oid in scope["operation_ids"]]
            hidden_inputs = {b["operands"]["hidden"] for b in scoped if b["kind"] == "projection"}
            _require(hidden_inputs == set(scope["incoming_hidden"]), "scope hidden inputs differ from projection operands")
            if index:
                _require(hidden_inputs == set(scopes[index-1]["hidden_outputs"]), "inter-layer hidden value seam changed")
                _require(all(not tensors[tid].get("initial_locations") for tid in hidden_inputs),
                         "inter-layer hidden inputs must be produced, not independently initialized")
            else:
                _require(all(tensors[tid].get("initial_locations") for tid in hidden_inputs),
                         "first-layer hidden input requires explicit supplied initial data")
            layer_prefix = f"w:{scope['layer_id']}:"
            children = ((program["graph"].get("boundary") or {}).get("ffn_execution") or {}).get("weight_children", {})
            def owned_weight(tid):
                if not tid.startswith(layer_prefix):
                    return False
                physical_ids = [tid] if tid in tensors else children.get(tid, [])
                return bool(physical_ids) and all(leaf in tensors and leaf.startswith(layer_prefix)
                    and tensors[leaf]["role"] == "weight" for leaf in physical_ids)
            _require(all(owned_weight(tid) for tid in weight_bindings[scope["id"]].values()),
                     "layer weight ownership mismatch")
            for b in scoped:
                _require(all(block["id"].startswith(f"kv:{scope['layer_id']}:") for block in b["history"] + b["new_kv"]),
                         "KV state cannot be borrowed from another layer")
    return program


def _array(values, ne):
    return np.asarray(values, dtype=np.float64).reshape(tuple(reversed(ne)))


def _ffn_view(binding, tensors, intermediate):
    """Recognize a full-parent slice without inventing a smaller allocation.

    Materialized leaves already have the narrowed physical shape. A full
    parent must have an explicit interval and the exact full matrix shape.
    """
    kind = binding["kind"]
    if kind not in {"ffn_gate", "ffn_up", "ffn_down"}:
        return None
    params = binding["op_params"]
    _require(len(params) == 2 and all(type(v) is int for v in params), "FFN slice needs explicit integer interval")
    start, end = params
    _require(0 <= start < end <= intermediate, "FFN slice interval outside parent")
    weight = tensors[binding["operands"]["x0"]]["ne"]
    axis = 0 if kind == "ffn_down" else 1
    width = end-start
    _require(weight[axis] in (width, intermediate), "FFN weight is neither exact parent nor materialized shard")
    return (start, end) if weight[axis] == intermediate and width != intermediate else None


def reference_ports(program):
    """Independent NumPy full Dense evaluation, with every materialized port."""
    validate_program(program)
    tensors = {t["id"]: t for t in program["graph"]["workload"]["tensors"]}
    values = {tid: _array(v, tensors[tid]["ne"]) for tid, v in program["initial_values"].items()}
    c = program["physical"]["config"]
    h, heads, kvh, d = (c[k] for k in ("hidden", "heads", "kv_heads", "head_dim"))
    bindings = {b["operation_id"]: b for b in program["bindings"]}
    produced = {}
    eps = c["norm_epsilon"]
    def matrix(v): return v.reshape(-1, v.shape[-1])
    def norm(v, gamma): return v / np.sqrt(np.mean(v*v, axis=-1, keepdims=True) + eps) * gamma.ravel()
    def mm(w, x): return matrix(x) @ matrix(w).T
    def attention(b, x, want):
        q = x.get("q")
        key = x["k"][0]
        val = x.get("v", x["k"])[0]
        n, past = b["shape"]["tokens"], b["shape"]["visible_history"]
        k = np.zeros((kvh, past+n, d)); v = np.zeros_like(k)
        for block in b["history"]:
            z = values[block["id"]].ravel().reshape(block["tokens"], 2, kvh, d)
            start = block["first_position"]
            k[:, start:start+block["tokens"]] = z[:, 0].transpose(1, 0, 2)
            v[:, start:start+block["tokens"]] = z[:, 1].transpose(1, 0, 2)
        k[:, past:] = key
        v[:, past:] = val
        if want == "qk":
            return np.stack([q[0, i] @ k[i//(heads//kvh)].T / math.sqrt(d) for i in range(heads)])[None]
        probabilities = x["probabilities"][0]
        context = np.stack([probabilities[i] @ v[i//(heads//kvh)] for i in range(heads)])
        context = context.transpose(1, 0, 2).reshape(n, heads*d)
        kv = {}
        for block in b["new_kv"]:
            start = block["first_position"] - past
            kv[block["id"]] = np.stack((key[:, start:start+block["tokens"]], val[:, start:start+block["tokens"]]), axis=0).transpose(2, 0, 1, 3).ravel()
        return context, kv
    for action in program["actions"]:
        if action["kind"] != "COMPUTE": continue
        b = bindings[action["target_id"]]
        x = {k: values[tid] for k, tid in b["operands"].items()}
        kind, n = b["kind"], b["shape"]["tokens"]
        out = {}
        if kind == "projection":
            normalized = norm(matrix(x["hidden"]), x["input_norm"])
            for role, count in (("q", heads), ("k", kvh), ("v", kvh)):
                z = mm(x[role], normalized).reshape(n, count, d)
                if role != "v":
                    if c["qk_norm"] == "rms": z = norm(z, x[role+"_norm"])
                    for token, position in enumerate(x["positions"].ravel()):
                        for pair in range(c["rope_dims"]//2):
                            angle = position / c["rope_base"]**(2*pair/c["rope_dims"])
                            u, v = z[token, :, 2*pair].copy(), z[token, :, 2*pair+1].copy()
                            z[token, :, 2*pair] = u*math.cos(angle)-v*math.sin(angle)
                            z[token, :, 2*pair+1] = u*math.sin(angle)+v*math.cos(angle)
                out[b["outputs"][role]] = z.transpose(1, 0, 2)
            if "gate_proj" in x: out[b["outputs"]["gate"]] = mm(x["gate_proj"], normalized)
        elif kind in {"attention", "attention_qk"}:
            scores = attention(b, x, "qk")
            if kind == "attention_qk": out[b["outputs"]["result"]] = scores
            else:
                for token in range(n): scores[:, :, token, b["position_start"]+token+1:] = -np.inf
                prob = np.exp(scores - np.max(scores, axis=-1, keepdims=True)); prob /= prob.sum(axis=-1, keepdims=True)
                context, extra = attention(b, {**x, "probabilities": prob}, "pv")
                if "gate" in x: context *= 1/(1+np.exp(-matrix(x["gate"])))
                out = {b["outputs"]["result"]: mm(x["output"], context) + matrix(x["hidden"]), **extra}
        elif kind == "attention_softmax":
            scores = x["scores"].copy()
            for token in range(n): scores[:, :, token, b["position_start"]+token+1:] = -np.inf
            z = np.exp(scores-np.max(scores, axis=-1, keepdims=True)); z /= z.sum(axis=-1, keepdims=True)
            out[b["outputs"]["result"]] = z
        elif kind == "attention_pv":
            context, extra = attention(b, x, "pv"); out = {b["outputs"]["result"]: context, **extra}
        elif kind == "attention_output":
            z = matrix(x["context"])
            if "gate" in x: z = z/(1+np.exp(-matrix(x["gate"])))
            out[b["outputs"]["result"]] = mm(x["output"], z) + matrix(x["hidden"])
        elif kind == "dense_ffn":
            z = norm(matrix(x["hidden"]), x["post_attention_norm"])
            gate, up = mm(x["gate"], z), mm(x["up"], z)
            out[b["outputs"]["result"]] = mm(x["down"], gate/(1+np.exp(-gate))*up) + matrix(x["hidden"])
        elif kind == "ffn_norm": out[b["outputs"]["result"]] = norm(matrix(x["x1"]), x["x0"])
        elif kind in {"ffn_gate", "ffn_up", "ffn_down"}:
            w=matrix(x["x0"])
            interval=_ffn_view(b,tensors,c["intermediate"])
            if interval:
                start,end=interval
                w=w[:,start:end] if kind=="ffn_down" else w[start:end,:]
            out[b["outputs"]["result"]] = mm(w, x["x1"])
        elif kind == "ffn_swiglu": out[b["outputs"]["result"]] = x["x0"]/(1+np.exp(-x["x0"]))*x["x1"]
        elif kind in {"ffn_sum", "ffn_residual"}: out[b["outputs"]["result"]] = sum(x.values())
        else: raise BridgeError(f"no independent reference for {kind}")
        for tid, value in out.items():
            values[tid] = np.asarray(value, dtype=np.float32).astype(np.float64).reshape(tuple(reversed(tensors[tid]["ne"])))
            produced[tid] = values[tid].ravel().tolist()
    return produced


def verify_native(program, native, *, atol=5e-5, rtol=5e-5):
    validate_program(program)
    _require(native.get("format_version") == "highlevel-accuracy-native-v1", "native schema mismatch")
    _require(native.get("program_sha256") == program["program_sha256"], "native program identity mismatch")
    _require(native.get("actions") == program["actions"], "native action trace differs")
    _require(native.get("boundary_contract") == BOUNDARY, "native physical boundary differs")
    _require(native.get("observed_runtime_flags") == program["required_runtime_flags"], "native runtime/precision flags differ")
    _require(native.get("final_execution_ports_captured_outside_timing") is True, "final execution output ports were not checked")
    expected = reference_ports(program)
    actual = native.get("ports", {})
    _require(set(expected) == set(actual), "native port inventory differs from full compiled graph")
    worst = 0.0
    for tid, wanted in expected.items():
        got = np.asarray(actual[tid], dtype=np.float64)
        wanted = np.asarray(wanted, dtype=np.float64)
        _require(got.shape == wanted.shape and np.all(np.isfinite(got)), f"invalid native port {tid}")
        worst = max(worst, float(np.max(np.abs(got-wanted))))
        _require(np.allclose(got, wanted, atol=atol, rtol=rtol), f"independent numerical mismatch: {tid}")
    copies = [dict(index=i, tensor_id=a["target_id"], src=a["src"], dst=a["dst"],
                   storage_bytes=next(t["storage_bytes"] for t in program["graph"]["workload"]["tensors"] if t["id"] == a["target_id"]))
              for i, a in enumerate(program["actions"]) if a["kind"] == "COPY"]
    _require(native.get("copies") == copies, "native copy identity/route/byte boundary differs")
    _require(native.get("copy_values_equal") is True, "native copy data not independently checked")
    return dict(full_graph_equal=True, action_trace_equal=True, full_port_inventory_equal=True,
                copy_routes_bytes_and_values_equal=True, independently_checked_ports=len(expected),
                max_absolute_error=worst, atol=atol, rtol=rtol)
