"""Export the actual compiled Python strategy, including full operands and actions.

This adapter deliberately supports one complete F32 Dense invocation. Both the
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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


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


def export_plan(plan, *, seed=1234):
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
    compiled = plan._compiled
    scopes = getattr(compiled.model, "invocation_scopes", ())
    _require(compiled.global_model and len(scopes) == 1,
             "bridge supports exactly one global_m2 invocation; multi-round/trace gates rejected")
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
    result = plan.run()
    run = result[0]
    _require(run.evaluation.status == "valid" and run.termination_reason == "goal_reached",
             "compiled ModelPlan must complete successfully before export")
    actions = []
    for record in run.decisions:
        _require(record.accepted and record.action is not None, "rejected/control decisions cannot be exported")
        action = asdict(record.action)
        _require(action["kind"] in {"COMPUTE", "COPY", "EVICT", "ADVANCE"}, "unsupported native action")
        _require(not (action["kind"] in {"COMPUTE", "COPY"} and record.observation.running),
                 "overlapping launch is unsupported; author a serial RuntimePolicy explicitly")
        actions.append(action)
    tensors = {t["id"]: t for t in graph["workload"]["tensors"]}
    _require(sum(math.prod(t["ne"]) for t in tensors.values()) <= 4_000_000,
             "JSON bridge tensor envelope exceeds 4 million elements")
    weights = {role: f"w:{aggregate.layer.id}:{name}" for group in physical.weight_groups.values()
               for role, name in group.items()}
    tasks = {t["operation_id"]: t for t in graph["tasks"]}
    ops = {op["id"]: op for op in graph["workload"]["operations"]}
    requests = {r.request_id: r for r in aggregate.invocation.requests}
    states = {r.request_id: r for r in aggregate.state_in.requests}
    projection = {t["request_id"]: ops[t["operation_id"]] for t in graph["tasks"] if t["kind"] == "projection"}
    partition = {p["tensor_id"]: p for p in (graph.get("boundary") or {}).get("ffn_execution", {}).get("weight_partitions", [])}
    bindings = []
    initial_values = {}
    for tid, tensor in tensors.items():
        if not tensor.get("initial_locations"):
            continue
        if tensor["dtype"] == "GGML_TYPE_I32":
            matches = [r for rid, r in requests.items() if tid.endswith(f":{rid}:in.positions")]
            _require(len(matches) == 1, "unrecognized position input")
            request = matches[0]
            initial_values[tid] = list(range(request.position_start, request.position_start + request.tokens))
        elif tid in partition:
            continue
        else:
            gamma = tid in {weights.get(r) for r in ("input_norm", "post_attention_norm", "q_norm", "k_norm")}
            initial_values[tid] = _values(tensor, seed, gamma).tolist()
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
        initial_values[tid] = leaf.ravel().tolist()
    for oid, op in ops.items():
        task = tasks[oid]
        request = requests[task["request_id"]]
        state = states[task["request_id"]]
        _require(tuple(b.first_position for b in state.blocks) == tuple(sum(b.tokens for b in state.blocks[:i]) for i in range(len(state.blocks)))
                 and sum(b.tokens for b in state.blocks) == request.position_start,
                 "only complete contiguous history beginning at absolute position zero is supported")
        qkv = [projection[task["request_id"]]["output"], *projection[task["request_id"]].get("extra_outputs", [])]
        p = dict(zip(("q", "k", "v", "gate"), qkv))
        history = [asdict(b) for b in state.blocks]
        new = [asdict(b) for b in aggregate.new_blocks[task["request_id"]]]
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
            operands.update(output=weights["output"], hidden=next(tid for tid in op["inputs"] if tid.endswith(":in.hidden")))
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
        seed=seed, original_simulation=dict(makespan_ns=run.evaluation.makespan_ns,
            cost_source=graph["scenario"]["costs"]["source"], measured_accuracy_claimed=False),
        limitations=["one_invocation", "serial_launches_only", "F32_only", "no_services_or_disturbances",
                    "native_backing_preallocated_eviction_is_logical_not_memory_measurement"])
    program["identity"] = {key: sha(program[key]) for key in ("frontend_ir", "graph", "physical", "actions", "bindings", "initial_values", "simulator_runtime_sha256")}
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
    _require(program["frontend_sha256"] == sha(program["frontend_ir"]), "frontend compile identity changed")
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
            _require(w[0] == activation[0] and w[2:] == [1,1], "FFN matrix reduction dimension mismatch")
            expect(out["result"],(w[1],n))
        elif kind == "dense_ffn":
            expect(x["hidden"],(h,n));expect(out["result"],(h,n))
            for role in ("post_attention_norm","gate","up","down"):expect(x[role],weight_shapes(config)[role])
        elif kind == "ffn_norm":expect(x["x0"],(h,));expect(x["x1"],(h,n));expect(out["result"],(h,n))
    return program


def _array(values, ne):
    return np.asarray(values, dtype=np.float64).reshape(tuple(reversed(ne)))


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
        elif kind in {"ffn_gate", "ffn_up", "ffn_down"}: out[b["outputs"]["result"]] = mm(x["x0"], x["x1"])
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
