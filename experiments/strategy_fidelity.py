"""Small, hand-derived semantic references for Python -> IR -> runtime fidelity.

No profile, model weights, GPU, network, or pre-generated example is required.
All durations are explicitly synthetic. Expected geometry and algebra below
are specified before compilation; compiler output is only the observed side.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    for directory in (ROOT / "src", ROOT / "mapping", ROOT / "modeling"):
        sys.path.insert(0, str(directory))

from llm_simulator import (
    AnalyticalCosts, Cache, DataPreparation, Decision, Device, EnvironmentEvent,
    Hardware, InvalidInput, Model, PhasePlacement, Prefetch, Route, Scheduling,
    Select, Simulator, Unsupported, Workload,
)
from llm_infer_model.m2.ffn_execution import parse_ffn_execution
from tensor_mapping.causal import compile_command
from tensor_mapping.unified_strategy import create_session
from llm_infer_model.tensor.runtime import dispatch_command


SOURCE_ROOTS = {"llm_simulator": ROOT / "src/llm_simulator",
                "llm_infer_model": ROOT / "modeling/llm_infer_model",
                "tensor_mapping": ROOT / "mapping/tensor_mapping"}


def source_snapshot():
    """Freeze all potentially lazy-imported Python bytes before the cases."""
    files = {Path(__file__).resolve()}
    for directory in SOURCE_ROOTS.values():
        files.update(path.resolve() for path in directory.rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def imported_sources():
    """Reject installed/foreign modules masquerading as the tested checkout."""
    result = []
    for name, module in tuple(sys.modules.items()):
        prefix = name.split(".", 1)[0]
        if prefix not in SOURCE_ROOTS:
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            raise ValueError(f"tested package module has no inspectable source: {name}")
        path = Path(file).resolve()
        if not path.is_relative_to(SOURCE_ROOTS[prefix].resolve()) or path.suffix != ".py":
            raise ValueError(f"tested module is not the expected checkout source: {name}: {path}")
        result.append(dict(module=name, path=str(path),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return sorted(result, key=lambda row: row["module"])


def verify_source_integrity(before, after, modules):
    if before != after:
        raise ValueError("source bytes changed during the semantic evidence run")
    for row in modules:
        relative = Path(row["path"]).resolve().relative_to(ROOT).as_posix()
        if before.get(relative) != row["sha256"]:
            raise ValueError(f"imported module is not bound to frozen source bytes: {row['module']}")


def tensor_id(suffix):
    # This fixture's public identity contract is handwritten, never obtained
    # by asking the compiler to produce its own expected operands.
    return "g:layer0-prefill:t:layer0:request0:" + suffix


def attention_wiring():
    t = tensor_id
    return {
        "projection": dict(inputs=["w:layer0:W_input_norm", "w:layer0:W_q", "w:layer0:W_k",
                                    "w:layer0:W_v", t("in.hidden"), t("in.positions")],
                           outputs=[t("proj.q"), t("proj.k"), t("proj.v")]),
        "attention_qk": dict(inputs=[t("proj.q"), t("proj.k")], outputs=[t("attn.scores")]),
        "attention_softmax": dict(inputs=[t("attn.scores")], outputs=[t("attn.probabilities")]),
        "attention_pv": dict(inputs=[t("attn.probabilities"), t("proj.k"), t("proj.v")],
                             outputs=[t("attn.context"), "kv:layer0:request0:0"]),
        "attention_output": dict(inputs=[t("attn.context"), "w:layer0:W_output", t("in.hidden")],
                                 outputs=[t("attn")]),
    }


def ffn_wiring():
    t = tensor_id
    rows = {
        "ffn_norm": dict(inputs=["w:layer0:W_post_attention_norm", t("attn")],
                         outputs=[t("ffn.ffn.norm")]),
        "ffn_sum": dict(inputs=[t("ffn.ffn.host.partial"), t("ffn.ffn.accelerator.partial")],
                        outputs=[t("ffn.ffn.sum")]),
        "ffn_residual": dict(inputs=[t("ffn.ffn.sum"), t("attn")], outputs=[t("ffn")]),
    }
    for shard in ("host", "accelerator"):
        prefix = "ffn.ffn." + shard + "."
        for stage in ("gate", "up"):
            rows[prefix + stage] = dict(kind="ffn_" + stage,
                inputs=[f"w:layer0:W_{stage}:ffn:{shard}", t("ffn.ffn.norm")], outputs=[t(prefix + stage)])
        rows[prefix + "swiglu"] = dict(kind="ffn_swiglu",
            inputs=[t(prefix + "gate"), t(prefix + "up")], outputs=[t(prefix + "swiglu")])
        rows[prefix + "down"] = dict(kind="ffn_down",
            inputs=[f"w:layer0:W_down:ffn:{shard}", t(prefix + "swiglu")], outputs=[t(prefix + "partial")])
    return rows


def check_expected_wiring(operations, expected):
    for label, contract in expected.items():
        kind = contract.get("kind", label)
        # Repeated shard kinds are disambiguated by their independent expected
        # result identity; inputs are then compared exactly, including order.
        matches = [op for op in operations.values() if op.kind == kind and
                   tuple(op.outputs) == tuple(contract["outputs"])]
        equal(len(matches), 1, f"{label} exact output identity")
        equal(tuple(matches[0].inputs), tuple(contract["inputs"]), f"{label} exact operand identities")
    return len(expected)


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, label):
    require(actual == expected, f"{label}: expected {expected!r}, observed {actual!r}")


def hardware():
    # One byte/ns link, no fixed latency. These are declared hypothetical rates.
    return Hardware(Device(1e9, 1e9), Device(2e9, 2e9), 1 << 20, 1 << 20,
                    1e9, 1e9, 1e9)


def dense(*, layers=1, prefill=1, decode=0, declared_hardware=None):
    return Simulator(model=Model.dense(layers=layers, hidden_size=8,
        intermediate_size=16, heads=2, kv_heads=1),
        hardware=declared_hardware or hardware(),
        workload=Workload(prefill, decode_steps=decode), cost_model=AnalyticalCosts())


def transfers():
    # Explicit exact prices for the hand-enumerated F32 shapes, not interpolation.
    return [dict(route=route, storage_bytes=size, duration_ns=size)
            for route in ("dram_to_vram", "vram_to_dram")
            for size in (4, 8, 16, 32, 48, 64, 128, 384, 512)]


def attention_spec():
    return dict(format_version="attention-stages-0.1", variant="qk_softmax_pv",
        provenance="synthetic_execution",
        geometry=dict(hidden=8, heads=2, kv_heads=1, head_dim=4, attention_gate="none"),
        frontend=dict(contract="existing_materialized_post_rope_qkv"),
        entries=[dict(kind=kind, device=device, phase="prefill", batch=1,
            tokens=1, visible_history=0, duration_ns=duration, workspace_bytes=4)
            for kind, duration in (("attention_qk", 11), ("attention_softmax", 13),
                                   ("attention_pv", 17), ("attention_output", 19))
            for device in ("cpu", "gpu")], transfer_entries=transfers())


def ffn_spec():
    return dict(format_version="ffn-execution-0.1", target_kinds=["dense_ffn"],
        intermediate_size=16,
        shards=[dict(id="host", start=0, end=4, device="cpu"),
                dict(id="accelerator", start=4, end=16, device="gpu")],
        merge_input_count=2, preprocess_device="gpu", merge_device="gpu",
        provenance="synthetic_execution",
        weight_storage_contract="materialized_partition_replaces_parent",
        entries=[dict(kind=kind, device=device, phase="prefill", batch=1, tokens=1,
            visible_history=0, hidden_size=8, intermediate_size=width,
            duration_ns=duration, workspace_bytes=4)
            for kind, duration in (("ffn_norm", 7), ("ffn_gate", 11), ("ffn_up", 13),
                ("ffn_swiglu", 17), ("ffn_down", 19), ("ffn_sum", 23), ("ffn_residual", 29))
            for device in ("cpu", "gpu") for width in (4, 12, 16)],
        transfer_entries=transfers())


def split_plan(directory, *, attention=True, ffn=True, startup=False):
    def configure(plan):
        if attention:
            plan.transform("layer0-prefill", attention_spec())
            plan.place("attention_qk", "cpu")
            plan.place("attention_softmax", "cpu")
        if ffn:
            plan.transform("layer0-prefill", ffn_spec())
        if startup:
            plan.prepare(DataPreparation(tensor_ids=tuple(
                f"w:layer0:W_{stage}:ffn:accelerator" for stage in ("gate", "up", "down")),
                trigger="startup"))
    return dense().compile(directory=Path(directory) / "inputs", configure=configure,
        scheduling=Scheduling(default_device="gpu", release="last_use",
                              cache=Cache(rule="lru", budget_bytes=4096)))


def completed(compiled, directory):
    result = compiled.run()
    require(result.success, f"execution failed: {result.summary()}")
    result.save(Path(directory) / "execution")
    return result


def observed(result):
    observation = result.raw[1].observe()
    return ({row.id: row for row in observation.operations},
            {row.id: row for row in observation.tensors},
            {row.target_id: row for row in result.events if row.kind == "COMPUTE"})


def one_kind(operations, kind):
    matches = [row for row in operations.values() if row.kind == kind]
    equal(len(matches), 1, f"{kind} instance count")
    return matches[0]


def one_suffix(tensors, suffix):
    matches = [row for row in tensors.values() if row.id.endswith(suffix)]
    equal(len(matches), 1, f"tensor suffix {suffix}")
    return matches[0]


def check_copy(result, tensor_id, source, destination, size):
    events = [row for row in result.events if row.kind == "COPY" and
              (row.target_id, row.src, row.dst) == (tensor_id, source, destination)]
    equal(len(events), 1, f"charged {source}->{destination} copy of {tensor_id}")
    equal(events[0].bytes_transferred, size, "copy bytes")
    equal(events[0].end_ns - events[0].start_ns, size, "one-byte/ns exact copy time")
    return events[0]


def check_dependency_lifetimes(result):
    operations, _, computation = observed(result)
    producers = {tid: operation.id for operation in operations.values() for tid in operation.outputs}
    checked = 0
    for operation in operations.values():
        for tid in operation.inputs:
            if tid in producers:
                require(computation[operation.id].start_ns >= computation[producers[tid]].end_ns,
                        f"consumer started before producer completed: {tid}")
                checked += 1
    for event in result.events:
        if event.kind == "EVICT":
            for op_id, compute in computation.items():
                if compute.start_ns <= event.start_ns < compute.end_ns:
                    domain = "dram" if compute.device == "cpu" else "vram"
                    require(event.location != domain or event.target_id not in operations[op_id].inputs,
                            "an in-flight compute input was evicted")
    return checked


def attention_boundary(directory):
    expected = dict(geometry="F32 H=8, heads=2, KV_heads=1, D=4, q=1, history=0",
        q_bytes=32, k_bytes=16, v_bytes=16, scores_bytes=8, probabilities_bytes=8,
        context_bytes=32, appended_kv_bytes=32,
        sequence=["projection", "attention_qk", "attention_softmax", "attention_pv", "attention_output"],
        stage_durations_ns=dict(attention_qk=11, attention_softmax=13,
                                attention_pv=17, attention_output=19), operand_wiring=attention_wiring())
    write_json(Path(directory) / "expected.json", expected)
    with split_plan(directory, ffn=False) as compiled:
        result = completed(compiled, directory)
        ops, tensors, events = observed(result)
        wiring = check_expected_wiring(ops, expected["operand_wiring"])
        for suffix, size in (("proj.q", 32), ("proj.k", 16), ("proj.v", 16),
                             ("attn.scores", 8), ("attn.probabilities", 8), ("attn.context", 32)):
            equal(one_suffix(tensors, suffix).storage_bytes, size, suffix)
        qk, softmax, pv, output = [one_kind(ops, kind) for kind in expected["sequence"][1:]]
        equal(softmax.inputs, qk.outputs, "softmax consumes scores")
        require(softmax.outputs[0] in pv.inputs, "PV must consume probabilities")
        require(pv.outputs[0] in output.inputs, "output projection must consume context")
        equal(pv.outputs[1], "kv:layer0:request0:0", "new KV is owned by PV completion")
        equal(tensors[pv.outputs[1]].storage_bytes, 32, "appended K+V bytes")
        for kind, duration in expected["stage_durations_ns"].items():
            op = one_kind(ops, kind)
            equal(events[op.id].end_ns - events[op.id].start_ns, duration, f"exact {kind} time")
        check_copy(result, one_suffix(tensors, "proj.q").id, "vram", "dram", 32)
        check_copy(result, one_suffix(tensors, "proj.k").id, "vram", "dram", 16)
        probabilities = check_copy(result, softmax.outputs[0], "dram", "vram", 8)
        require(events[pv.id].start_ns >= probabilities.end_ns, "PV did not wait for probability transfer")
        return dict(expected=expected, exact_operand_contracts_checked=wiring,
                    dependency_edges_checked=check_dependency_lifetimes(result),
                    observed=result.summary())


def ffn_partition(directory):
    # Algebra: y = sum_s[(SiLU(x Wg_s) * x Wu_s) Wd_s]; add residual once.
    expected = dict(hidden=8, intermediate=16, cuts=[0, 4, 16],
        per_matrix_leaf_bytes=[128, 384], total_partitioned_matrix_bytes=1536,
        shard_activation_bytes=[16, 48], partial_output_bytes=32,
        norm_count=1, sum_count=1, sum_arity=2, residual_count=1,
        cpu_partial_return_bytes=32, operand_wiring=ffn_wiring())
    write_json(Path(directory) / "expected.json", expected)
    with split_plan(directory, attention=False) as compiled:
        result = completed(compiled, directory)
        ops, tensors, events = observed(result)
        wiring = check_expected_wiring(ops, expected["operand_wiring"])
        total = 0
        for stage in ("gate", "up", "down"):
            require(f"w:layer0:W_{stage}" not in tensors, "unsplit parent matrix still allocates capacity")
            for shard, size in (("host", 128), ("accelerator", 384)):
                equal(tensors[f"w:layer0:W_{stage}:ffn:{shard}"].storage_bytes, size, "matrix leaf bytes")
                total += size
        equal(total, 1536, "complete partition conserves three HxI F32 matrices")
        for shard, width in (("host", 4), ("accelerator", 12)):
            for stage in ("gate", "up", "swiglu"):
                equal(one_suffix(tensors, f"ffn.{shard}.{stage}").storage_bytes, 4 * width,
                      f"{shard} intermediate bytes")
            equal(one_suffix(tensors, f"ffn.{shard}.partial").storage_bytes, 32, "down partial hidden width")
        norm, merge, residual = [one_kind(ops, kind) for kind in ("ffn_norm", "ffn_sum", "ffn_residual")]
        partials = tuple(one_suffix(tensors, f"ffn.{shard}.partial").id for shard in ("host", "accelerator"))
        equal(merge.inputs, partials, "SUM consumes both independent partials")
        equal(len(residual.inputs), 2, "one residual input plus merged output")
        require(merge.outputs[0] in residual.inputs, "residual must follow SUM")
        returned = check_copy(result, partials[0], "dram", "vram", 32)
        require(events[merge.id].start_ns >= returned.end_ns, "SUM started before CPU partial arrived")
        require(events[residual.id].start_ns >= events[merge.id].end_ns, "residual started before SUM")
        equal(events[norm.id].end_ns - events[norm.id].start_ns, 7, "independently supplied norm cost")
        equal(events[merge.id].end_ns - events[merge.id].start_ns, 23, "independently supplied SUM cost")
        return dict(expected=expected, exact_operand_contracts_checked=wiring,
                    dependency_edges_checked=check_dependency_lifetimes(result), observed=result.summary())


def composition(directory):
    expected = dict(startup_tensors=3, bytes_per_startup_tensor=384,
                    startup_total_bytes=1152, weight_cache_budget_bytes=4096,
                    attention_stages=4, ffn_stages=11)
    write_json(Path(directory) / "expected.json", expected)
    with split_plan(directory, startup=True) as compiled:
        result = completed(compiled, directory)
        ops, _, _ = observed(result)
        check_expected_wiring(ops, attention_wiring())
        check_expected_wiring(ops, ffn_wiring())
        equal(sum(op.kind.startswith("attention_") for op in ops.values()), 4, "attention transforms survive composition")
        equal(sum(op.kind.startswith("ffn_") for op in ops.values()), 11, "FFN transforms survive composition")
        first_compute = min(event.start_ns for event in result.events if event.kind == "COMPUTE")
        for stage in ("gate", "up", "down"):
            event = check_copy(result, f"w:layer0:W_{stage}:ffn:accelerator", "dram", "vram", 384)
            require(event.end_ns <= first_compute, "startup preparation did not precede computation")
        for record in result.raw[0].decisions:
            weight_bytes = sum(row.storage_bytes for row in record.observation.tensors
                if row.role == "weight" and row.status_at("vram") != "ABSENT")
            require(weight_bytes <= 4096, "cache exceeded declared weight budget")
        audit = result.raw[2].audit_document()["explicit_data_preparations"]
        equal(len(audit), 3, "startup preparation catalogue count")
        return dict(expected=expected, dependency_edges_checked=check_dependency_lifetimes(result), observed=result.summary())


def phase_kv_persistence(directory):
    expected = dict(layers=2, prefill_tokens=2, decode_steps=2,
                    attention_histories=[0, 2, 3], kv_blocks=8,
                    kv_bytes_per_block=32, final_persistent_kv_bytes=256,
                    completed_invocations=6,
                    phase_devices=dict(prefill="gpu", decode="cpu"))
    write_json(Path(directory) / "expected.json", expected)
    def configure(plan):
        PhasePlacement(prefill={k: "gpu" for k in ("projection", "attention", "dense_ffn")},
                       decode={k: "cpu" for k in ("projection", "attention", "dense_ffn")}).apply(plan)
    with dense(layers=2, prefill=2, decode=2).compile(directory=Path(directory) / "inputs",
            scheduling=Scheduling(default_device="gpu", release="last_use"), configure=configure) as compiled:
        result = completed(compiled, directory)
        ops, tensors, events = observed(result)
        for op in ops.values():
            if op.kind in ("projection", "attention", "dense_ffn"):
                equal(events[op.id].device, expected["phase_devices"][op.phase], "phase-specific execution")
        state = result.raw[1].persistent_state_document()
        equal(len(state["completed_invocations"]), 6, "global invocation count")
        kv_ids = {f"kv:layer{layer}:request0:{position}" for layer in range(2) for position in range(4)}
        equal(set(state["kv_ids"]), kv_ids, "stable KV identities across phases and layers")
        equal(sum(tensors[tid].storage_bytes for tid in kv_ids), 256, "all persistent KV bytes")
        tasks = result.raw[1].execution_document()["tasks"]
        for layer in ("layer0", "layer1"):
            histories = [task["shape"]["visible_history"] for task in tasks
                         if task["kind"] == "attention" and f":op:{layer}:" in task["operation_id"]]
            equal(histories, [0, 2, 3], "history grows across rounds")
        # GPU prefill produces two blocks per layer. CPU decode can only use
        # those same blocks after four charged 32-byte D2H copies.
        for layer in range(2):
            for position in range(2):
                check_copy(result, f"kv:layer{layer}:request0:{position}", "vram", "dram", 32)
        return dict(expected=expected, dependency_edges_checked=check_dependency_lifetimes(result), observed=result.summary())


def prefetch_cache_placement(directory):
    expected = dict(prefetch_layer="layer1", prefetch_lookahead=1,
        cache_budget_bytes=4096, every_copy_duration="one ns per payload byte",
        execution="layer0 FFN on CPU; layer1 FFN on GPU; attention/projection on GPU")
    write_json(Path(directory) / "expected.json", expected)
    def configure(plan):
        plan.place("dense_ffn", "cpu", select=Select(layer_id="layer0"))
    with dense(layers=2, decode=1).compile(directory=Path(directory) / "inputs", configure=configure,
            scheduling=Scheduling(default_device="gpu", release="last_use",
                cache=Cache(rule="lru", budget_bytes=4096),
                prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True))) as compiled:
        result = completed(compiled, directory)
        ops, _, events = observed(result)
        for op in ops.values():
            if op.kind == "dense_ffn":
                equal(events[op.id].device, "cpu" if op.layer_id == "layer0" else "gpu", "scoped placement")
        speculative = [row for row in result.raw[2].audit_document()["decisions"]
                       if row["reason"] == "cross_layer_public_history_prefetch"]
        require(speculative, "declared cross-layer prefetch never produced a charged action")
        for record in result.raw[0].decisions:
            weight_bytes = sum(row.storage_bytes for row in record.observation.tensors
                if row.role == "weight" and row.status_at("vram") != "ABSENT")
            require(weight_bytes <= 4096, "prefetch/cache composition exceeded budget")
        for event in result.events:
            if event.kind == "COPY":
                equal(event.end_ns - event.start_ns, event.bytes_transferred, "no free data movement")
        return dict(expected=expected, charged_prefetch_decisions=len(speculative),
                    dependency_edges_checked=check_dependency_lifetimes(result), observed=result.summary())


def moe_simulator(second_expert):
    return Simulator(model=Model.moe(layers=1, hidden_size=8, intermediate_size=16,
        heads=2, kv_heads=1, experts=3, top_k=1), hardware=hardware(),
        workload=Workload(1, decode_steps=1, routes=(Route("request0", "layer0", 0, (0,)),
            Route("request0", "layer0", 1, (second_expert,))), route_source="hand-declared-causal-pair"),
        cost_model=AnalyticalCosts())


def moe_causality(directory):
    expected = dict(past_route=[0], alternative_unrevealed_futures=[[1], [2]],
                    routes_before_first_router=0, final_route_count=2,
                    future_expert_command="rejected before reveal",
                    prefix_requirement="identical public observations and commands through second router launch")
    write_json(Path(directory) / "expected.json", expected)
    prefixes = []
    for future in (1, 2):
        sub = Path(directory) / f"future-{future}"
        with moe_simulator(future).compile(directory=sub / "inputs",
                scheduling=Scheduling(default_device="cpu", release="last_use")) as compiled:
            initial = create_session(compiled.plan._compiled).observe()
            equal(initial.route_history, (), "initial routing history")
            require(not any(op.kind == "expert_ffn" for op in initial.operations), "future expert task leaked")
            try:
                compile_command(Decision("COMPUTE", "g:layer0-decode0:op:layer0:__moe_batch__:expert.2", device="cpu"), initial)
            except Unsupported as error:
                require("not been revealed" in str(error), "wrong reason for rejecting hidden expert")
            else:
                raise AssertionError("compiler accepted an unrevealed future expert")
            result = completed(compiled, sub)
            prefix = []
            for record in result.raw[0].decisions:
                if any(route.absolute_position == 1 for route in record.observation.route_history):
                    break
                prefix.append(dict(observation=asdict(record.observation), action=asdict(record.action)))
                for op in record.observation.operations:
                    if op.kind == "expert_ffn" and op.invocation_id == "layer0-decode0":
                        raise AssertionError("decode expert became public before its route")
                if record.action.kind == "COMPUTE":
                    op = next(row for row in record.observation.operations if row.id == record.action.target_id)
                    if op.kind == "expert_ffn":
                        require(any(route.absolute_position == 0 and op.expert_id in route.expert_ids
                                    for route in record.observation.route_history), "expert lacks a public route")
            prefixes.append(prefix)
            equal(len(result.raw[1].observe().route_history), 2, "all final routes revealed")
    equal(prefixes[0], prefixes[1], "changing private future must not alter causal prefix")
    return dict(expected=expected, identical_prefix_decisions=len(prefixes[0]),
                public_prefix_sha256=hashlib.sha256(json.dumps(prefixes[0], sort_keys=True).encode()).hexdigest())


def transform_rejections(directory):
    expected = dict(rejections=["partition gap", "partition overlap", "incomplete partition",
        "changed SUM arity", "missing exact shard price", "missing exact attention price",
        "missing exact probability transfer", "unmatched selector", "unsupported device"])
    write_json(Path(directory) / "expected.json", expected)
    errors = {}
    for label, mutate in (("partition gap", lambda s: s["shards"][1].update(start=5)),
                          ("partition overlap", lambda s: s["shards"][1].update(start=3)),
                          ("incomplete partition", lambda s: s["shards"][1].update(end=15)),
                          ("changed SUM arity", lambda s: s.update(merge_input_count=3))):
        spec = ffn_spec(); mutate(spec)
        try:
            parse_ffn_execution(spec)
        except InvalidInput as error:
            errors[label] = str(error)
        else:
            raise AssertionError(f"invalid transform accepted: {label}")
    for index, label in enumerate(expected["rejections"][4:]):
        def configure(plan):
            if label == "missing exact shard price":
                spec = ffn_spec()
                spec["entries"] = [row for row in spec["entries"] if not
                    (row["kind"] == "ffn_gate" and row["device"] == "cpu" and row["intermediate_size"] == 4)]
                plan.transform("layer0-prefill", spec)
            elif label in ("missing exact attention price", "missing exact probability transfer"):
                spec = attention_spec()
                if label == "missing exact attention price":
                    spec["entries"] = [row for row in spec["entries"] if row["kind"] != "attention_pv"]
                else:
                    spec["transfer_entries"] = [row for row in spec["transfer_entries"] if row["storage_bytes"] != 8]
                plan.transform("layer0-prefill", spec)
            elif label == "unmatched selector":
                plan.place("dense_ffn", "cpu", select=Select(layer_id="absent"))
            else:
                plan.place("dense_ffn", "gpu1")
        try:
            compiled = dense().compile(directory=Path(directory) / f"invalid-{index}",
                scheduling=Scheduling(default_device="gpu"), configure=configure)
        except (InvalidInput, Unsupported) as error:
            errors[label] = str(error)
        else:
            compiled.close()
            raise AssertionError(f"unsupported transform silently accepted: {label}")
    equal(set(errors), set(expected["rejections"]), "rejection coverage")
    return dict(expected=expected, observed_rejections=errors)


def unavailable_capacity(directory):
    expected = dict(vram_capacity_bytes=1, gpu_execution="rejected or infeasible",
                    implicit_cpu_fallback=False, direct_weight_copy_bytes=32,
                    direct_copy_error_code="CAPACITY_EXCEEDED",
                    high_level_failure_kind="policy_dead_end", high_level_error_code="no_progress",
                    global_infeasibility_proven=False)
    write_json(Path(directory) / "expected.json", expected)
    simulator = dense(declared_hardware=replace(hardware(), vram_capacity_bytes=1))
    with simulator.compile(directory=Path(directory) / "inputs",
            scheduling=Scheduling(default_device="gpu", release="last_use")) as compiled:
        session = create_session(compiled.plan._compiled)
        observation = session.observe()
        weight = next(tensor for tensor in observation.tensors if tensor.id == "w:layer0:W_input_norm")
        equal(weight.storage_bytes, 32, "hand-derived input norm weight exceeds one-byte VRAM")
        command = compile_command(Decision("COPY", weight.id, src="dram", dst="vram"), observation)
        step = dispatch_command(session, command)
        require(not step.ok, "oversize direct copy was admitted")
        equal(step.error_code, "CAPACITY_EXCEEDED", "runtime capacity admission diagnostic")
        result = compiled.run()
        result.save(Path(directory) / "execution")
        require(not result.success, "impossible one-byte VRAM was accepted")
        equal(result.raw[0].termination_reason, "policy_dead_end", "high-level dead-end classification")
        equal(result.evaluation.error_code, "no_progress", "idle clock has an explicit no-progress diagnostic")
        require(not any(event.kind == "COMPUTE" and event.device == "cpu" for event in result.events),
                "infeasible GPU strategy silently fell back to CPU")
        return dict(expected=expected, direct_copy_rejection=asdict(step), observed=result.summary(),
            high_level_policy_failure_reason=result.evaluation.reason,
            high_level_policy_dead_end_diagnostic_validated=True, global_infeasibility_proven=False)


def cpu_service_blackout(directory):
    expected = dict(initial_cpu_unavailable_ns=10000, additional_elapsed_ns=10000,
                    compute_slot="one CPU slot; no OS thread simulation")
    write_json(Path(directory) / "expected.json", expected)
    summaries = []
    for delayed in (False, True):
        sub = Path(directory) / ("blackout" if delayed else "baseline")
        with dense(decode=1, declared_hardware=replace(hardware(), gpu=None)).compile(
                directory=sub / "inputs", scheduling=Scheduling(default_device="cpu", release="last_use"),
                environment_events=((EnvironmentEvent(t_ns=0, cpu_available=False),
                    EnvironmentEvent(t_ns=10000, cpu_available=True)) if delayed else ())) as compiled:
            result = completed(compiled, sub)
            if delayed:
                require(min(event.start_ns for event in result.events if event.kind == "COMPUTE") >= 10000,
                        "CPU ran while its service was unavailable")
            summaries.append(result.summary())
    equal(summaries[1]["makespan_ns"] - summaries[0]["makespan_ns"], 10000, "hand-derived blackout delay")
    return dict(expected=expected, observed=summaries)


CASES = {
    "attention_boundary": attention_boundary,
    "ffn_partition": ffn_partition,
    "composition": composition,
    "phase_kv_persistence": phase_kv_persistence,
    "prefetch_cache_placement": prefetch_cache_placement,
    "moe_causality": moe_causality,
    "transform_rejections": transform_rejections,
    "unavailable_capacity": unavailable_capacity,
    "cpu_service_blackout": cpu_service_blackout,
}


def run_suite(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    source_before = source_snapshot()
    imports_before = imported_sources()
    write_json(output / "source-manifest-before.json", source_before)
    write_json(output / "imported-modules-before.json", imports_before)
    rows = []
    for name, case in CASES.items():
        folder = output / name
        folder.mkdir()
        try:
            evidence = case(folder)
            row = dict(case=name, status="passed", evidence=evidence)
        except Exception as error:
            row = dict(case=name, status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(folder / "result.json", row)
        rows.append(row)
    source_after = source_snapshot()
    imports_after = imported_sources()
    verify_source_integrity(source_before, source_after, imports_before + imports_after)
    source_manifest = [dict(path=path, sha256=digest) for path, digest in source_before.items()]
    write_json(output / "source-manifest.json", source_manifest)
    write_json(output / "imported-modules-after.json", imports_after)
    report = dict(format_version="strategy-fidelity-evidence-0.1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((output / "source-manifest.json").read_bytes()).hexdigest(),
        source_integrity_verified=True, imported_source_paths_verified=True,
        imported_module_count=len(imports_after),
        python=sys.version, platform=platform.platform(),
        evidence_kind="hand_derived_semantic_reference", hardware_measurement=False,
        numerical_tensor_execution=False, calibrated=False,
        passed=sum(row["status"] == "passed" for row in rows), total=len(rows), cases=rows,
        limitations=["Finite F32 materialized reference cases; not exhaustive strategy-space verification.",
            "Reference algebra/geometry and byte assertions are handwritten; tensor numerical values are not executed.",
            "All compute/transfer durations are synthetic contracts, not validated hardware predictions.",
            "Phase placement uses shared CPU/GPU resources, not independent disaggregated serving instances.",
            "MoE uses an explicit two-token trace; no router accuracy or full-paper reproduction claim.",
            "A high-level policy_dead_end/no_progress result is an unsuccessful schedule, not proof that all mappings on the hardware are infeasible; direct oversized COPY admission is independently rejected with CAPACITY_EXCEEDED.",
            "No unified-memory, multi-GPU collectives, operating-system scheduler, or global-optimum claim."])
    write_json(output / "report.json", report)
    text = ["# Strategy fidelity evidence", "", f"Passed: {report['passed']}/{report['total']} cases.", "",
            "These are semantic reference checks with synthetic timings, not hardware accuracy results.", "",
            "| Case | Result |", "|---|---|"]
    text.extend(f"| {row['case']} | {row['status']} |" for row in rows)
    text.extend(["", "## Limits", ""] + [f"- {line}" for line in report["limitations"]])
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_suite(args.out)
    print(json.dumps(dict(passed=report["passed"], total=report["total"], report=str(args.out / "report.json"))))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
