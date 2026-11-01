"""Paired strategy-authoring tasks, with independent semantic assertions.

No production code is changed, no hardware is measured, and no human task
completion time or code compression ratio is inferred from this campaign.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import io
import json
from pathlib import Path
import tokenize
import traceback
from types import SimpleNamespace

from llm_simulator import ModelPlan
from experiments import strategy_fidelity as sf


ROOT = Path(__file__).resolve().parents[2]


def _snapshot():
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (ROOT / "src/llm_simulator", ROOT / "mapping/tensor_mapping",
                              ROOT / "modeling/llm_infer_model")
            for path in directory.rglob("*.py")}


def _lines(source):
    # Count actual source lines containing non-comment Python tokens. Multiline
    # expressions count each occupied physical line; blank/comments do not.
    occupied = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                              tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            occupied.update(range(token.start[0], token.end[0] + 1))
    return len(occupied)


def _write(path, value):
    sf.write_json(path, value)


def _snippet(directory, name, source):
    path = directory / (name + "_strategy.py")
    path.write_text(source, encoding="utf-8", newline="\n")
    namespace = {"__name__": "ir_acceptance_authoring_" + name}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace, dict(path=str(path.resolve()),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          authored_nonblank_noncomment_lines=_lines(source))


def _inputs(directory, *, layers=2, decode=0, environment=None):
    simulator = sf.dense(layers=layers, decode=decode)
    paths = simulator.cost_model.build_inputs(directory / "inputs", simulator.model,
                                              simulator.hardware, simulator.workload)
    if environment is not None:
        _write(Path(paths["environment_file"]), environment)
    return paths


def _execute(directory, paths, namespace, *, services=None):
    directory.mkdir(parents=True)
    plan = ModelPlan(id="authoring-pair", model_kind=paths["model_kind"],
                     model_file=paths["model_file"], environment_file=paths["environment_file"],
                     scheduling=namespace["scheduling"](), services_file=services,
                     visibility_mode="data_dependencies")
    if "configure" in namespace:
        namespace["configure"](plan)
    _write(directory / "declaration.json", plan.declaration_document())
    compiled = plan.compile()
    _write(directory / "ir.json", compiled.ir_document())
    raw = compiled.run()
    compiled.write_result(directory / "execution", raw)
    sf.equal(raw[0].termination_reason, "goal_reached", "paired strategy must complete")
    sf.equal(raw[0].evaluation.status, "valid", "paired runtime validity")
    result = SimpleNamespace(raw=raw, events=raw[0].evaluation.events)
    sf.check_dependency_lifetimes(result)
    return result


def _pair(directory, paths, baseline, changed, *, services=None):
    results, metrics = [], {}
    for name, source in (("baseline", baseline), ("changed", changed)):
        namespace, measure = _snippet(directory, name, source)
        results.append(_execute(directory / name, paths, namespace, services=services))
        metrics[name] = measure
    return results, metrics


DEFAULT = '''from llm_simulator import Cache, Scheduling
def scheduling():
    return Scheduling(default_device="gpu", release="last_use",
                      cache=Cache(rule="lru", budget_bytes=8192))
'''


def _module_placement(directory):
    _write(directory / "expected.json", dict(
        same_input=True, baseline_layer_ffn_devices={"layer0": "gpu", "layer1": "gpu"},
        changed_layer_ffn_devices={"layer0": "cpu", "layer1": "gpu"},
        changed_layer0_attention_to_ffn_d2h_bytes=32,
        changed_layer0_ffn_to_layer1_h2d_bytes=32))
    paths = _inputs(directory)
    changed = DEFAULT + '''def configure(plan):
    with plan.module("first_layer", layer_id="layer0"):
        plan.place("dense_ffn", "cpu")
'''
    results, metrics = _pair(directory, paths, DEFAULT, changed)
    bindings = []
    for result, expected in zip(results, ({"layer0": "gpu", "layer1": "gpu"},
                                          {"layer0": "cpu", "layer1": "gpu"})):
        ops, _, events = sf.observed(result)
        actual = {op.layer_id: events[op.id].device for op in ops.values() if op.kind == "dense_ffn"}
        sf.equal(actual, expected, "per-layer FFN devices")
        for op in ops.values():
            if op.kind in ("projection", "attention"):
                sf.equal(events[op.id].device, "gpu", "unmodified stages stay on GPU")
        bindings.append(actual)
    sf.check_copy(results[1], "g:layer0-prefill:t:layer0:request0:attn", "vram", "dram", 32)
    sf.check_copy(results[1], "g:layer0-prefill:t:layer0:request0:ffn", "dram", "vram", 32)
    sf.require(not any(event.kind == "COPY" and event.src == "vram" and event.dst == "dram"
                       and event.target_id == "g:layer0-prefill:t:layer0:request0:attn"
                       for event in results[0].events), "baseline unexpectedly copies the selected FFN input back to CPU")
    return dict(observed_bindings=bindings, changed_boundary_bytes=64), metrics


def _partition_ratio(directory):
    _write(directory / "expected.json", dict(
        same_input=True, hidden=8, intermediate=16, dtype_bytes=4,
        baseline_widths={"host": 4, "accelerator": 12},
        changed_widths={"host": 12, "accelerator": 4},
        per_run_total_partitioned_matrix_bytes=1536,
        cpu_partial_return_bytes=32, sum_arity=2,
        cost_source="Existing handwritten synthetic exact table covers widths 4, 12 and 16; unchanged across both strategies."))
    paths = _inputs(directory, layers=1)
    # This helper constructs the transform, so it is authoring code even when
    # reused from an older reference. Embed and count it instead of hiding its
    # complexity behind a one-line import. Exact-price literals are therefore
    # conservatively included too; model-input generation remains separate.
    template = (DEFAULT + inspect.getsource(sf.transfers) + "\n" +
                inspect.getsource(sf.ffn_spec) + "\n" + '''def configure(plan):
    spec = ffn_spec()
    spec["shards"][0]["end"] = CPU_WIDTH
    spec["shards"][1]["start"] = CPU_WIDTH
    plan.transform("layer0-prefill", spec)
''')
    results, metrics = _pair(directory, paths, template.replace("CPU_WIDTH", "4"),
                              template.replace("CPU_WIDTH", "12"))
    counts = []
    for result, cpu_width in zip(results, (4, 12)):
        ops, tensors, _ = sf.observed(result)
        sf.check_expected_wiring(ops, sf.ffn_wiring())
        total = 0
        sizes = {}
        for shard, width in (("host", cpu_width), ("accelerator", 16 - cpu_width)):
            sizes[shard] = 8 * width * 4
            for stage in ("gate", "up", "down"):
                sf.equal(tensors[f"w:layer0:W_{stage}:ffn:{shard}"].storage_bytes,
                         sizes[shard], "hand-derived partitioned matrix payload")
                total += sizes[shard]
            for stage in ("gate", "up", "swiglu"):
                sf.equal(sf.one_suffix(tensors, f"ffn.{shard}.{stage}").storage_bytes,
                         width * 4, "intermediate activation payload")
        sf.equal(total, 1536, "partition conserves total matrix payload")
        sf.check_copy(result, sf.tensor_id("ffn.ffn.host.partial"), "dram", "vram", 32)
        counts.append(sizes)
    sf.require(counts[0] != counts[1], "ratio change had no physical effect")
    return dict(matrix_leaf_bytes=counts, total_matrix_bytes_each=1536), metrics


def _cache_budget(directory):
    _write(directory / "expected.json", dict(
        same_input=True, baseline_cache_budget_bytes=8192, changed_cache_budget_bytes=4096,
        requirement="Both runs finish. The smaller budget actually causes safe weight eviction; reservations count in the one budget."))
    paths = _inputs(directory, decode=1)
    results, metrics = _pair(directory, paths, DEFAULT, DEFAULT.replace("8192", "4096"))
    counts = []
    for result, budget in zip(results, (8192, 4096)):
        owners = {row[0] for row in result.raw[0].decisions[0].observation.weight_catalogue}
        evictions = [event for event in result.events if event.kind == "EVICT" and
                     event.target_id in owners and event.location == "vram"]
        peak = 0
        for record in result.raw[0].decisions:
            used = sum(row.storage_bytes for row in record.observation.tensors
                       if row.role == "weight" and row.status_at("vram") != "ABSENT")
            sf.require(used <= budget, "shared cache including reservations exceeded budget")
            peak = max(peak, used)
        counts.append(dict(weight_evictions=len(evictions), peak_cached_payload_bytes=peak))
    sf.equal(counts[0]["weight_evictions"], 0, "large cache retains all reusable weights")
    sf.require(counts[1]["weight_evictions"] > 0, "smaller cache never exercised eviction")
    return dict(observed=counts), metrics


def _prefetch_toggle(directory):
    _write(directory / "expected.json", dict(
        same_input=True, baseline_prefetch=False, changed_prefetch=True,
        requirement="Changed plan has a charged layer1 weight COPY overlapping layer0 COMPUTE, tagged cross_layer_public_history_prefetch; baseline has no such tag."))
    paths = _inputs(directory)
    template = '''from llm_simulator import Cache, Prefetch, Scheduling
def scheduling():
    return Scheduling(default_device="gpu", release="last_use",
        cache=Cache(rule="lru", budget_bytes=8192),
        prefetch=PREFETCH)
'''
    results, metrics = _pair(directory, paths, template.replace("PREFETCH", "Prefetch()"),
                              template.replace("PREFETCH", "Prefetch(next_layers=1, max_count=2, include_dense_weights=True)"))
    observed = []
    for result in results:
        ops, _, _ = sf.observed(result)
        copies = [row for row in result.raw[2].audit_document()["decisions"]
                  if row["reason"] == "cross_layer_public_history_prefetch"]
        overlap = []
        for row in copies:
            tid = row["evidence"]["tensor"]
            for event in result.events:
                if event.kind == "COPY" and event.target_id == tid and tid.startswith("w:layer1:"):
                    sf.equal(event.end_ns - event.start_ns, event.bytes_transferred, "prefetch is charged")
                    if any(compute.kind == "COMPUTE" and ops[compute.target_id].layer_id == "layer0"
                           and compute.start_ns <= event.start_ns < compute.end_ns for compute in result.events):
                        overlap.append(asdict(event))
        observed.append(dict(prefetch_decisions=len(copies), layer1_overlapping_copies=overlap))
    sf.equal(observed[0]["prefetch_decisions"], 0, "disabled prefetch audit")
    sf.require(observed[1]["layer1_overlapping_copies"], "enabled prefetch never starts during layer0 work")
    return dict(observed=observed), metrics


def _transfer_engine(directory):
    _write(directory / "expected.json", dict(
        same_input=True, baseline_layer_engines={"layer0": "ce", "layer1": "ce"},
        changed_layer_engines={"layer0": "ce", "layer1": "sm"},
        shared_link_bytes_per_second=1_000_000_000,
        engine_rates_bytes_per_second={"ce": 1_000_000_000, "sm": 500_000_000}))
    paths = _inputs(directory)
    services = directory / "services.json"
    _write(services, dict(format_version="tensor-services-0.1", pcie=dict(
        bandwidth_bytes_per_second=1_000_000_000, ce_bytes_per_second=1_000_000_000,
        sm_bytes_per_second=500_000_000, arbitration="max_min_fair",
        timing_scope="payload_only_no_launch_latency"), slots=[]))
    template = '''from llm_simulator import Cache, Scheduling, Transport
def scheduling():
    return Scheduling(default_device="gpu", release="last_use",
        cache=Cache(rule="lru", budget_bytes=8192), transport=Transport(engine="ce"))
def configure(plan):
    with plan.module("second_layer", layer_id="layer1"):
        plan.transfer(engine="ENGINE")
'''
    results, metrics = _pair(directory, paths, template.replace("ENGINE", "ce"),
                              template.replace("ENGINE", "sm"), services=services)
    engines = []
    for result, expected in zip(results, ({"layer0": "ce", "layer1": "ce"},
                                          {"layer0": "ce", "layer1": "sm"})):
        owners = {row[0]: row[1] for row in result.raw[0].decisions[0].observation.weight_catalogue}
        seen = {"layer0": set(), "layer1": set()}
        for event in result.events:
            if event.kind == "COPY" and event.target_id in owners and event.dst == "vram":
                owner = owners[event.target_id]
                sf.equal(event.resource, "pcie_" + expected[owner], "actual owner-scoped copy engine")
                seen[owner].add(expected[owner])
        sf.equal({layer: sorted(values) for layer, values in seen.items()},
                 {layer: [engine] for layer, engine in expected.items()}, "both layers exercised")
        engines.append({layer: sorted(values) for layer, values in seen.items()})
    return dict(observed_layer_engines=engines, shared_services_sha256=hashlib.sha256(services.read_bytes()).hexdigest()), metrics


def _resource_response(directory):
    _write(directory / "expected.json", dict(
        same_input=True, same_cpu_and_memory_scenario=True,
        cpu_unavailable_interval_ns=[0, 10000],
        memory_scenario="No external occupancy change; identical finite DRAM/VRAM in both runs.",
        baseline_response="fixed CPU placement waits for public CPU restoration",
        changed_response="current_cost chooses GPU while current CPU service is unavailable",
        first_projection_devices={"baseline": "cpu", "changed": "gpu"},
        private_restoration_causality="A third execution moves restoration from 10000 to 20000; changed policy decisions before 10000 must remain identical."))
    environment = dict(format_version="environment-trace-0.2", events=[
        dict(t_ns=0, cpu_available=False), dict(t_ns=10000, cpu_available=True)], requests=[],
        control_costs=dict(ack_ns=3, decline_ns=1, decision_ns=2,
                          source="synthetic_not_hardware_measured", timing_scope="dedicated_serial_control_resource"))
    paths = _inputs(directory, layers=1, environment=environment)
    template = '''from llm_simulator import Cache, Scheduling
from tensor_mapping.reconfiguration import ReconfigurationSpec
def scheduling():
    return Scheduling(default_device="cpu", release="last_use",
        cache=Cache(rule="lru", budget_bytes=8192),
        reconfiguration=ReconfigurationSpec("release_clean_weights", "lru", "RESPONSE"))
'''
    results, metrics = _pair(directory, paths, template.replace("RESPONSE", "fixed"),
                              template.replace("RESPONSE", "current_cost"))
    first = []
    for result, device in zip(results, ("cpu", "gpu")):
        ops, _, events = sf.observed(result)
        projection = next(op for op in ops.values() if op.kind == "projection")
        event = events[projection.id]
        sf.equal(event.device, device, "resource rule must change actual placement")
        if device == "cpu":
            sf.require(event.start_ns >= 10000, "fixed CPU ran during unavailable service")
        else:
            sf.require(event.start_ns < 10000, "adaptive GPU projection missed unavailable CPU interval")
            decision = next(row for row in result.raw[0].decisions if row.action.kind == "COMPUTE" and row.action.target_id == projection.id)
            sf.equal(decision.observation.cpu_available, False, "adaptive placement uses current public CPU state")
        first.append(dict(device=event.device, start_ns=event.start_ns))
    # The independent future variant is not included in authoring line counts:
    # it changes the test input to audit causality, not the user's strategy.
    later = json.loads(json.dumps(environment))
    later["events"][1]["t_ns"] = 20000
    future = directory / "private_future_variant"
    future.mkdir()
    variant_paths = _inputs(future, layers=1, environment=later)
    namespace, _ = _snippet(future, "changed", template.replace("RESPONSE", "current_cost"))
    other = _execute(future / "changed", variant_paths, namespace)
    def prefix(result):
        return [dict(observation=asdict(row.observation), action=asdict(row.action))
                for row in result.raw[0].decisions if row.observation.t_ns < 10000]
    sf.equal(prefix(results[1]), prefix(other), "private future restoration cannot change earlier decisions")
    sf.require(prefix(results[1]), "causality prefix must be nonempty")
    return dict(first_projection=first, private_future_prefix_decisions=len(prefix(other))), metrics


CASES = {"U01": _module_placement, "U02": _partition_ratio, "U03": _cache_budget,
         "U04": _prefetch_toggle, "U05": _transfer_engine, "U06": _resource_response}


def run_group(output: Path) -> list[dict]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("authoring output must be new or empty")
    before = _snapshot()
    _write(output / "production-before.json", before)
    rows = []
    for identifier, case in CASES.items():
        folder = output / identifier
        folder.mkdir()
        try:
            evidence, metrics = case(folder)
            row = dict(id=identifier, status="pass", reason="Both authored strategies completed and independent semantic difference checks passed.",
                       evidence=evidence, authoring=dict(production_files_changed=0,
                           authored_nonblank_noncomment_lines={name: value["authored_nonblank_noncomment_lines"] for name, value in metrics.items()},
                           snippets=metrics, code_compression_ratio=None, human_completion_time=None,
                           input_generation=dict(source="experiments/strategy_fidelity.py", functions=["dense", "hardware"],
                               accounting="Existing input/geometry/exact-price fixtures, separately identified; not strategy task helpers."),
                           checker=dict(source=str(Path(__file__).resolve()), function=case.__name__,
                                        accounting="Independent assertions and evidence writing; excluded from strategy code counts.")),
                       artifacts=[str(path.resolve()) for path in sorted(folder.rglob("*")) if path.is_file()])
        except Exception as error:
            row = dict(id=identifier, status="fail", reason=str(error), evidence=dict(traceback=traceback.format_exc()),
                       authoring=dict(production_files_changed=0, code_compression_ratio=None, human_completion_time=None),
                       artifacts=[str(path.resolve()) for path in sorted(folder.rglob("*")) if path.is_file()])
        _write(folder / "result.json", row)
        rows.append(row)
    after = _snapshot()
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    _write(output / "production-after.json", after)
    for row in rows:
        row["authoring"]["production_files_changed"] = len(changed)
        if changed:
            row.update(status="unverified", reason="Production sources changed during campaign; attribution is not stable.")
        _write(output / row["id"] / "result.json", row)
    _write(output / "results.json", rows)
    return rows
