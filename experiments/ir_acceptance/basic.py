"""Frozen B01--B12 adapters; synthetic semantic evidence, never timing accuracy.

Handwritten expected contracts are persisted before compiling each case. Existing
independent semantic references are reused with their origin recorded explicitly.
No production module or measured profile is changed by this campaign.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path
import traceback

from experiments import strategy_fidelity as reference
from llm_simulator import (
    AnalyticalCosts, Cache, Device, Hardware, Model, ModelPlan, Prefetch,
    ReconfigurationSpec, Scheduling, Select, Simulator, Transport, Workload,
)
from llm_simulator.simulation import CompiledSimulation


def _write(path, value):
    reference.write_json(path, value)


def _execute(compiled, directory):
    _write(Path(directory) / "compiled-ir.json", compiled.ir_document())
    result = compiled.run()
    result.save(Path(directory) / "execution")
    _write(Path(directory) / "observed-summary.json", result.summary())
    reference.require(result.success, f"policy did not complete: {result.summary()}")
    return result


def _cache_bytes(result):
    observations = [row.observation for row in result.raw[0].decisions]
    observations.append(result.raw[1].observe())
    return max(sum(t.storage_bytes for t in obs.tensors
                   if t.role == "weight" and t.status_at("vram") != "ABSENT")
               for obs in observations)


def _placement(directory):
    expected = dict(layers=2, phases=["prefill"], ffns={"layer0": "cpu", "layer1": "gpu"},
                    other_compute="gpu", layer0_attention_to_cpu_ffn_bytes=32,
                    layer0_ffn_to_layer1_projection_bytes=32)
    _write(directory / "expected.json", expected)

    def configure(plan):
        with plan.module("first-layer", layer_id="layer0"):
            plan.place("dense_ffn", "cpu")

    with reference.dense(layers=2).compile(directory=directory / "inputs", configure=configure,
            scheduling=Scheduling(default_device="gpu", release="last_use")) as compiled:
        result = _execute(compiled, directory)
        ops, _, events = reference.observed(result)
        for op in ops.values():
            if op.kind != "state_update":
                device = expected["ffns"][op.layer_id] if op.kind == "dense_ffn" else "gpu"
                reference.equal(events[op.id].device, device, "exact module binding")
        first = next(op for op in ops.values() if op.layer_id == "layer0" and op.kind == "dense_ffn")
        attention = next(op for op in ops.values() if op.layer_id == "layer0" and op.kind == "attention")
        reference.check_copy(result, attention.outputs[0], "vram", "dram", 32)
        reference.check_copy(result, first.outputs[0], "dram", "vram", 32)
        return dict(observed=result.summary(), edges=reference.check_dependency_lifetimes(result))


def _rounds(directory):
    expected = dict(completed_invocations=[f"layer{layer}-{phase}" for phase in
        ("prefill", "decode0", "decode1") for layer in range(2)],
        compute_instances={"projection": 6, "attention": 6, "dense_ffn": 6},
        kv_ids=[f"kv:layer{layer}:request0:{position}" for layer in range(2) for position in range(4)],
        final_kv_bytes=256, identical_flat_and_module_events=True,
        requirement="modules add labels, not barriers; all producer dependencies retained")
    _write(directory / "expected.json", expected)
    signatures, summaries = [], []
    for scoped in (False, True):
        sub = directory / ("modules" if scoped else "flat")

        def configure(plan):
            if scoped:
                with plan.module("decoder-stack"):
                    for layer in range(2):
                        with plan.module(f"layer{layer}", layer_id=f"layer{layer}"):
                            plan.place("dense_ffn", "gpu")
            else:
                for layer in range(2):
                    plan.place("dense_ffn", "gpu", select=Select(layer_id=f"layer{layer}"))

        with reference.dense(layers=2, prefill=2, decode=2).compile(directory=sub / "inputs",
                configure=configure, scheduling=Scheduling(default_device="gpu", release="last_use")) as compiled:
            result = _execute(compiled, sub)
            ops, tensors, _ = reference.observed(result)
            for kind, count in expected["compute_instances"].items():
                selected = [op for op in ops.values() if op.kind == kind]
                reference.equal(len(selected), count, "hand-counted core instances")
                for op in selected:
                    reference.equal(sum(e.kind == "COMPUTE" and e.target_id == op.id for e in result.events),
                                    1, "each intended compute completes exactly once")
            state = result.raw[1].persistent_state_document()
            reference.equal(state["completed_invocations"], expected["completed_invocations"], "exact layer/round completion order")
            reference.equal(set(state["kv_ids"]), set(expected["kv_ids"]), "stable persistent KV identities")
            reference.equal(sum(tensors[tid].storage_bytes for tid in expected["kv_ids"]), 256, "history payload conserved")
            reference.check_dependency_lifetimes(result)
            signatures.append([asdict(event) for event in result.events])
            summaries.append(result.summary())
    reference.equal(signatures[0], signatures[1], "module labels must not invent scheduling barriers")
    return dict(observed=summaries, identical_flat_and_module_events=True,
                exact_completed_invocations=expected["completed_invocations"])


def _pipeline(directory):
    """The frozen B06 requirement now uses the common Python/IR/runtime path."""
    return _pipeline_public_attempt(directory)


def _pipeline_public_attempt(directory, *, layers=1, prefetch=False):
    expected = dict(subbatches=["request0", "request1"], microbatch_tokens=1,
        layers=layers, accounting_batch=2, cpu_attention_duration_ns=50_000,
        placements={"projection": "gpu", "attention": "cpu", "dense_ffn": "gpu"},
        required="positive-duration CPU attention / GPU FFN overlap across different subbatches",
        dependency_contract="each subbatch attention precedes its own FFN",
        weight_cache_budget_bytes=8192,
        cost_provenance="explicit hypothetical CPU attention prices; no fitted or measured batch timing")
    _write(directory / "expected.json", expected)
    simulator = Simulator(model=Model.dense(layers=layers, hidden_size=8, intermediate_size=16,
        heads=2, kv_heads=1), hardware=reference.hardware(), workload=Workload(1, batch_size=2),
        cost_model=AnalyticalCosts())

    def configure(plan):
        for request in expected["subbatches"]:
            with plan.module("subbatch-" + request, request_id=request):
                plan.place("attention", "cpu")
                plan.place("dense_ffn", "gpu")

    # Author all inputs before binding source hashes. A batch=2 entry prices
    # each independent request kernel under the declared accounting batch;
    # it is not a measured fused-batch result copied to both requests.
    root = directory / "inputs"
    generated = simulator.cost_model.build_inputs(root, simulator.model, simulator.hardware, simulator.workload)
    for path in sorted(root.glob("*.costs.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for entry in document["entries"]:
            if entry["kind"] == "attention" and entry["device"] == "cpu":
                entry["duration_ns"] = 50_000
                entry["provenance"] = "synthetic_layer"
        _write(path, document)
    plan = ModelPlan(id="acceptance-subbatch-pipeline", model_kind=generated["model_kind"],
        model_file=generated["model_file"], environment_file=generated["environment_file"],
        base_dir=root, visibility_mode="data_dependencies", max_decisions=100000,
        scheduling=Scheduling(default_device="gpu", release="last_use",
            cache=Cache(rule="lru", budget_bytes=8192),
            prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True) if prefetch else Prefetch()))
    configure(plan)
    compiled = CompiledSimulation(plan.compile(), root,
        dict(generated.get("metadata", {}), cost_model=generated["cost_model"]), None)
    with compiled:
        result = _execute(compiled, directory)
        ops, _, events = reference.observed(result)
        reference.equal({op.request_id for op in ops.values()}, {"request0", "request1"}, "two requests share one runtime")
        reference.equal(len([op for op in ops.values() if op.kind == "attention"]), 2 * layers, "one attention per request/layer")
        for op in ops.values():
            reference.equal(op.status, "DONE", "all requests finish")
            if op.kind == "attention":
                reference.equal(events[op.id].device, "cpu", "declared CPU attention")
                reference.equal(events[op.id].end_ns - events[op.id].start_ns, 50_000, "declared synthetic CPU attention cost")
            elif op.kind == "dense_ffn":
                reference.equal(events[op.id].device, "gpu", "declared GPU FFN")
        witnesses = []
        for cpu in ops.values():
            if cpu.kind != "attention":
                continue
            for gpu in ops.values():
                if gpu.kind != "dense_ffn" or cpu.request_id == gpu.request_id:
                    continue
                a, b = events[cpu.id], events[gpu.id]
                overlap = min(a.end_ns, b.end_ns) - max(a.start_ns, b.start_ns)
                if overlap > 0 and a.device == "cpu" and b.device == "gpu":
                    witnesses.append(dict(cpu=cpu.id, gpu=gpu.id, overlap_ns=overlap))
        reference.require(witnesses, "two scoped subbatches were expressible but no required cross-subbatch compute overlap occurred")
        evidence = dict(observed=result.summary(), overlap_witnesses=witnesses,
            edges=reference.check_dependency_lifetimes(result), entry_path="python_model_plan",
            expression_path="hierarchical-strategy common composition IR",
            public_builder_support=True, unified_frontend_supported=True,
            peak_weight_bytes=_cache_bytes(result), single_runtime=True,
            completed_operations=len(ops))
        reference.require(evidence["peak_weight_bytes"] <= 8192, "shared pipeline cache exceeded")
        if prefetch:
            # The helper is imported only on this optional path to avoid a
            # module-level basic/combinations dependency cycle.
            from experiments.ir_acceptance.combinations import _early_prefetch
            evidence["prefetch"] = _early_prefetch(result)
        return evidence


WEIGHTS = {"W_input_norm": 32, "W_q": 256, "W_k": 128, "W_v": 128,
           "W_output": 256, "W_post_attention_norm": 32,
           "W_gate": 512, "W_up": 512, "W_down": 512}


def _retention(directory):
    expected = dict(layers=1, invocations=3, per_layer_weight_bytes=2368,
                    weight_bytes=WEIGHTS, budget_bytes=4096,
                    required_h2d_copies_per_weight=1)
    _write(directory / "expected.json", expected)
    with reference.dense(decode=2).compile(directory=directory / "inputs",
            scheduling=Scheduling(default_device="gpu", release="last_use",
                                  cache=Cache(rule="lru", budget_bytes=4096))) as compiled:
        result = _execute(compiled, directory)
        _, tensors, _ = reference.observed(result)
        counts = {}
        for name, size in WEIGHTS.items():
            tid = "w:layer0:" + name
            reference.equal(tensors[tid].storage_bytes, size, "independent physical weight size")
            copies = [e for e in result.events if e.kind == "COPY" and e.target_id == tid
                      and e.src == "dram" and e.dst == "vram"]
            reference.equal(len(copies), 1, "retained weight must not reload")
            counts[tid] = len(copies)
        reference.require(_cache_bytes(result) <= 4096, "shared cache capacity exceeded")
        return dict(observed=result.summary(), copy_counts=counts,
                    peak_reserved_weight_payload_bytes=_cache_bytes(result),
                    edges=reference.check_dependency_lifetimes(result))


def _eviction(directory):
    expected = dict(layers=2, per_layer_weight_bytes=2368, total_weight_bytes=4736,
                    shared_cache_budget_bytes=2368,
                    required="safe weight EVICT and completed execution, reservations count")
    _write(directory / "expected.json", expected)
    with reference.dense(layers=2).compile(directory=directory / "inputs",
            scheduling=Scheduling(default_device="gpu", release="last_use",
                                  cache=Cache(rule="lru", budget_bytes=2368))) as compiled:
        result = _execute(compiled, directory)
        _, tensors, _ = reference.observed(result)
        evictions = [e for e in result.events if e.kind == "EVICT"
                     and tensors[e.target_id].role == "weight"]
        reference.require(evictions, "undersized shared cache produced no weight eviction")
        peak = _cache_bytes(result)
        reference.require(peak <= 2368, "cache reservations exceed budget")
        return dict(observed=result.summary(), weight_evictions=[asdict(e) for e in evictions],
                    peak_reserved_weight_payload_bytes=peak,
                    edges=reference.check_dependency_lifetimes(result))


def _prefetch(directory):
    expected = dict(layers=2, layer1_weight_prefix="w:layer1:", cache_budget_bytes=4096,
                    required="a charged next-layer COPY starts during positive-duration layer0 computation")
    _write(directory / "expected.json", expected)
    with reference.dense(layers=2).compile(directory=directory / "inputs",
            scheduling=Scheduling(default_device="gpu", release="last_use",
                cache=Cache(rule="lru", budget_bytes=4096),
                prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True))) as compiled:
        result = _execute(compiled, directory)
        ops, _, events = reference.observed(result)
        first_events = [events[o.id] for o in ops.values() if o.layer_id == "layer0"]
        witnesses = [asdict(e) for e in result.events if e.kind == "COPY"
                     and e.target_id.startswith("w:layer1:") and e.bytes_transferred > 0
                     and any(c.start_ns <= e.start_ns < c.end_ns for c in first_events)]
        reference.require(witnesses, "next-layer weight transfer never began during current-layer computation")
        for event in result.events:
            if event.kind == "COPY":
                reference.equal(event.end_ns - event.start_ns, event.bytes_transferred, "charged declared payload")
        return dict(observed=result.summary(), overlap_witnesses=witnesses,
                    peak_reserved_weight_payload_bytes=_cache_bytes(result),
                    edges=reference.check_dependency_lifetimes(result))


def _transport(directory):
    expected = dict(weight_engines={"layer0": "ce", "layer1": "sm"},
        cache_budget_bytes=8192, shared_bandwidth_bytes_per_second=1_000_000_000,
        ce_bytes_per_second=1_000_000_000, sm_bytes_per_second=500_000_000,
        timing_scope="payload_only_no_launch_latency", physical_links=1)
    _write(directory / "expected.json", expected)
    service_file = directory / "services.json"
    _write(service_file, dict(format_version="tensor-services-0.1", pcie=dict(
        bandwidth_bytes_per_second=1_000_000_000, ce_bytes_per_second=1_000_000_000,
        sm_bytes_per_second=500_000_000, arbitration="max_min_fair",
        timing_scope="payload_only_no_launch_latency"), slots=[]))

    def configure(plan):
        for layer, engine in expected["weight_engines"].items():
            with plan.module(layer, layer_id=layer):
                plan.transfer(engine=engine)

    hw = replace(reference.hardware(), transfer_latency_ns=0)
    with reference.dense(layers=2, declared_hardware=hw).compile(directory=directory / "inputs",
            services_file=service_file, configure=configure,
            scheduling=Scheduling(default_device="gpu", release="last_use",
                cache=Cache(rule="lru", budget_bytes=8192), transport=Transport(engine="ce"))) as compiled:
        result = _execute(compiled, directory)
        _, tensors, _ = reference.observed(result)
        engines = []
        service = result.raw[1].service_document()
        for transfer in service["transfer_engines"]:
            if transfer["route"] == "dram_to_vram" and tensors[transfer["tensor_id"]].role == "weight":
                layer = transfer["tensor_id"].split(":")[1]
                reference.equal(transfer["engine"], expected["weight_engines"][layer], "engine by physical owner")
                # The observed service ledger binds an actual charged COPY,
                # not just a controller's proposed engine choice.
                event = result.events[transfer["event_index"]]
                reference.equal(event.kind, "COPY", "engine ledger points to COPY")
                reference.equal(event.target_id, transfer["tensor_id"], "engine ledger tensor")
                reference.require(event.bytes_transferred > 0, "engine attached to free transfer")
                engines.append((layer, transfer["engine"]))
        reference.equal(set(engines), {("layer0", "ce"), ("layer1", "sm")}, "both scoped engines occur")
        peak = _cache_bytes(result)
        reference.require(peak <= 8192, "per-module scope duplicated cache budget")
        reference.equal(service["contract"]["pcie"], dict(
            bandwidth_bytes_per_second=1_000_000_000, ce_bytes_per_second=1_000_000_000,
            sm_bytes_per_second=500_000_000, arbitration="max_min_fair",
            timing_scope="payload_only_no_launch_latency"), "single physical service contract")
        _write(directory / "observed-services.json", service)
        return dict(observed=result.summary(), weight_copy_engines=engines,
                    peak_reserved_weight_payload_bytes=peak, shared_service_document=service,
                    edges=reference.check_dependency_lifetimes(result))


def _resource_response(directory):
    expected = dict(scenarios=["cpu-ready", "cpu-unavailable-recover-1000000",
                               "cpu-unavailable-recover-2000000"],
        initial_choice={"cpu-ready": "cpu", "cpu-unavailable-recover-1000000": "gpu",
                        "cpu-unavailable-recover-2000000": "gpu"},
        controller="current_cost over public current prices, memory and availability",
        requirement="different current CPU availability changes placement; private recovery does not alter prefix",
        decision_cost_ns=3)
    _write(directory / "expected.json", expected)
    choices, prefixes = {}, []
    for available, restoration in ((True, None), (False, 1_000_000), (False, 2_000_000)):
        name = "cpu-ready" if available else "cpu-unavailable-recover-" + str(restoration)
        sub = directory / name
        sub.mkdir(parents=True, exist_ok=True)
        simulator = reference.dense()
        generated = simulator.cost_model.build_inputs(sub / "inputs", simulator.model,
                                                      simulator.hardware, simulator.workload)
        events = [] if available else [dict(t_ns=0, cpu_available=False),
                                       dict(t_ns=restoration, cpu_available=True)]
        environment = sub / "environment.json"
        _write(environment, dict(format_version="environment-trace-0.2", events=events,
            requests=[], control_costs=dict(ack_ns=2, decline_ns=2, decision_ns=3,
                source="explicit synthetic B12", timing_scope="dedicated_serial_control_resource")))
        plan = ModelPlan(id="acceptance-resource-response", model_kind=generated["model_kind"],
            model_file=generated["model_file"], base_dir=sub / "inputs", environment_file=environment,
            scheduling=Scheduling(default_device="cpu", release="last_use",
                reconfiguration=ReconfigurationSpec("release_clean_weights", "lru", "current_cost")))
        frozen = plan.compile()
        metadata = dict(generated.get("metadata", {}), cost_model=generated["cost_model"])
        with CompiledSimulation(frozen, sub / "inputs", metadata, None) as compiled:
            result = _execute(compiled, sub)
            computations = [event for event in result.events if event.kind == "COMPUTE"]
            first = min(computations, key=lambda e: e.start_ns)
            reference.equal(first.device, expected["initial_choice"][name], "causal observed-state response")
            choices[name] = dict(first_device=first.device, first_start_ns=first.start_ns,
                                 summary=result.summary())
            if not available:
                prefix = [dict(observation=asdict(row.observation), action=asdict(row.action))
                          for row in result.raw[0].decisions if row.observation.t_ns < 1_000_000]
                prefixes.append(prefix)
            reference.check_dependency_lifetimes(result)
    reference.equal(prefixes[0], prefixes[1], "unknown future restoration cannot change early response")
    reference.require(prefixes[0], "empty causal prefix")
    return dict(choices=choices, identical_prefix_decisions=len(prefixes[0]))


CASES = {
    "B01": _placement,
    "B02": reference.phase_kv_persistence,
    "B03": reference.attention_boundary,
    "B04": reference.ffn_partition,
    "B05": _rounds,
    "B06": _pipeline,
    "B07": _retention,
    "B08": _eviction,
    "B09": _prefetch,
    "B10": _transport,
    "B11": reference.moe_causality,
    "B12": _resource_response,
}


def run_group(output: Path) -> list[dict]:
    output = Path(output).resolve()
    rows = []
    for case_id, function in CASES.items():
        directory = output / case_id
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(id=case_id, status="unverified", expression_status="unverified",
                   fidelity_status="not_tested", reason="", evidence={}, artifacts=[])
        try:
            evidence = function(directory)
            row.update(status="pass", expression_status="complete", fidelity_status="pass",
                       reason="Handwritten pre-execution semantic contract satisfied.", evidence=evidence)
            if case_id == "B06":
                row["reason"] = ("Actual subbatch CPU Attention/GPU FFN overlap completes through the shared "
                                 "Python ModelPlan/composition compiler and single modeling runtime.")
        except Exception as error:
            _write(directory / "error.json", dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
            # B06 is a required higher-level schedule. Placement scopes alone
            # do not establish this capability if the runtime serializes it.
            if case_id == "B06" and type(error).__name__ == "Unsupported" and "batch_size=1" in str(error):
                row.update(status="unsupported", expression_status="partial", fidelity_status="not_tested",
                           reason="Request-scoped placements exist, but the tested public Dense input builder rejects two subbatches; an explicit executable stage/subbatch schedule is not established.",
                           evidence=dict(error=str(error),
                               source="src/llm_simulator/analytical.py:104-105",
                               limitation="Do not confuse execution-pipeline-0.1 transform ordering with subbatch execution overlap."))
            elif case_id == "B06" and isinstance(error, AssertionError) and "overlap" in str(error):
                row.update(status="partial", expression_status="partial", fidelity_status="fail",
                           reason=str(error), evidence=dict(
                               expressed="request-scoped stage placements", missing="required positive cross-subbatch compute overlap"))
            elif type(error).__name__ == "Unsupported":
                row.update(status="unsupported", expression_status="partial", fidelity_status="not_tested",
                           reason=f"Unsupported: {error}",
                           evidence=dict(classification="Explicit tested-path limitation; no claim that all alternative entry paths are impossible."))
            elif isinstance(error, AssertionError) and (
                    (directory / "compiled-ir.json").is_file()
                    or (directory / "execution").is_dir()
                    or (directory / "advanced-execution").is_dir()
                    or (directory / "flat/execution").is_dir()):
                row.update(status="fail", expression_status="complete", fidelity_status="fail",
                           reason=f"Compiled declaration violated its independent semantic oracle: {error}")
            else:
                row.update(status="unverified", expression_status="unverified", fidelity_status="not_tested",
                           reason=f"Adapter/input failure cannot establish expression or fidelity: {type(error).__name__}: {error}")
        row["artifacts"] = sorted(str(path.resolve()) for path in directory.rglob("*") if path.is_file())
        _write(directory / "result.json", row)
        rows.append(row)
    return rows
