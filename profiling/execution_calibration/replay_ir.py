"""Offline exact-shape measured components through the fixed tensor IR/runtime.

This is a timing-only bridge, not a ModelPlan semantic-transform calibration.
Unknown CUDA workspace remains null in the binding contract. The mandatory
integer field in the runtime receives only its recorded known allocator lower
bound, with deliberately nonphysical, nonbinding capacities. No exported
scenario is an accepted capacity profile. No heldout duration enters a cost.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "modeling"), str(ROOT / "mapping")]
import analysis as a
import run_dual as protocol
from verify_context import verify
from llm_infer_model.tensor.spec import SpecError, build_scenario
from tensor_mapping.strategy import write_run
from tensor_mapping.strategy_compiler import compile_strategy, run_strategy
from tensor_mapping.strategy_ir import parse_strategy_ir
from tensor_mapping.unified_strategy import runtime_fingerprint


WEIGHT_H2D = {stage: "hybrid/preload_" + stage + "_weight_h2d" for stage in ("gate", "up", "down")}


def declared_recipe(case, strategy, signature):
    if strategy == "hybrid/preload":
        if signature["protocol"].get("measurement_contract", {}).get("kind") != "completed_batch_mean":
            raise ValueError("only resident paths have exact per-action transfer prices in the isolated protocol")
        if case["family"] != "ffn":
            raise ValueError("per-weight preload requires an FFN case")
        return {"sum": [{"sum": sorted(WEIGHT_H2D.values())}, protocol.recipes(case)["hybrid/resident"]]}
    if strategy not in ("gpu/resident", "hybrid/resident"):
        raise ValueError("unsupported primary measured path")
    return protocol.recipes(case)[strategy]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def physical_signature(case, rows, protocol_fields):
    """Require same physical contract across every independent process.

    Input values/seed digests may differ, but every port shape/byte count,
    materialization, workspace declaration and execution protocol must match.
    Caller must additionally verify original raw evidence before replay.
    """
    if not rows or any(row["case"] != case or row["status"] != "measured" for row in rows):
        raise ValueError("exact case geometry and successful process coverage are required")
    expected = protocol.expected_port_bytes(case)
    signatures = []
    for row in rows:
        if set(row["ports"]) != set(expected) or any(
                row["ports"][name]["bytes"] != count for name, count in expected.items()):
            raise ValueError("physical port byte signature mismatch")
        boundary = row["boundary"]
        required = dict(
            input_location="all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu",
            output_location="dram", merge_device="cpu", weights="generated_f32_not_model_weights",
            ffn_contract="compact_selected_rows_gate_up_swiglu_down_no_norm_or_residual",
            ffn_partition="gate_up_output_columns_and_down_input_rows_materialized_before_timing",
            attention_contract="post_RoPE_QKV_to_causal_GQA_QK_softmax_PV_sigmoid_gate_O_residual",
            kv_materialization="QK_private_K_scratch_then_PV_KV_append_outputs; old_cache_inputs_immutable")
        if any(boundary.get(key) != value for key, value in required.items()):
            raise ValueError("native operation/materialization boundary mismatch")
        workspaces = row["workspaces"]
        for name, entry in workspaces.items():
            known = entry.get("known_gallocr_plus_cplan_bytes")
            if type(known) is not int or known < 0:
                raise ValueError("missing explicit allocator lower bound")
            if name.startswith("gpu/"):
                if entry.get("workspace_bytes") is not None or entry.get("source") != "unknown_private_cuda_scratch":
                    raise ValueError("CUDA workspace must remain unknown, never rewritten to zero")
            elif name.startswith("cpu/"):
                if entry.get("workspace_bytes") != known or entry.get("source") != "gallocr_plus_explicit_cplan":
                    raise ValueError("CPU allocator workspace contract mismatch")
            else:
                raise ValueError("unsupported workspace device")
        signatures.append(dict(case=case, protocol=protocol_fields,
                               port_bytes=expected, boundary=boundary, workspaces=workspaces))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("physical contract differs between profile/heldout processes")
    signature = signatures[0]
    signature["profile_evidence"] = [dict(row_id=row["id"], raw_sha256=row["raw_sha256"],
                                           seed=row["seed"], repeat=row["repeat"])
                                      for row in rows if row["split"] == "profile"]
    return signature


class _Graph:
    def __init__(self, case, strategy, frozen_case, signature):
        if frozen_case["case_id"] != case["id"]:
            raise ValueError("frozen cost case mismatch")
        expected = declared_recipe(case, strategy, signature)
        if frozen_case["strategies"].get(strategy) != expected:
            raise ValueError("frozen strategy recipe differs from the declared native path")
        self.case, self.strategy, self.frozen, self.signature = case, strategy, frozen_case, signature
        self.tensors, self.ops, self.costs, self.placements = {}, [], {}, {}
        self.program, self.routes, self.transfers, self.cost_binding = [], {}, {}, []
        self.rounded_prices = {}
        self.workspace_assumptions = []

    def price(self, metric):
        if metric not in a.terms(self.frozen["strategies"][self.strategy]):
            raise ValueError("undeclared component, total duration or validation target cannot be a price")
        value = self.frozen["prices"].get(metric)
        if not value or not value.get("sources") or any(source.get("split") != "profile" for source in value["sources"]):
            raise ValueError("missing exact profile-only component cost: " + metric)
        sources = {(source["row_id"], source["raw_sha256"]) for source in value["sources"]}
        expected = {(source["row_id"], source["raw_sha256"]) for source in self.signature["profile_evidence"]}
        if sources != expected or any(source.get("metric") != metric for source in value["sources"]):
            raise ValueError("component sources do not match the exact physical profile evidence")
        number = value["median_ns"]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 1:
            raise ValueError("invalid exact component duration")
        rounded = math.ceil(number)
        self.rounded_prices[metric] = rounded
        self.cost_binding.append(dict(metric=metric, measured_median_ns=number,
                                      runtime_duration_ns=rounded, sources=value["sources"]))
        return rounded

    def tensor(self, name, shape, *, role="intermediate", device=None):
        ne = list(shape) + [1] * (4 - len(shape))
        if len(ne) != 4 or any(type(n) is not int or n < 1 for n in ne):
            raise ValueError("invalid explicit materialized F32 shape")
        nb = [4]
        for n in ne[:-1]:
            nb.append(nb[-1] * n)
        row = dict(id=name, name=name, role=role, dtype="GGML_TYPE_F32", ne=ne, nb=nb,
                   storage_bytes=4 * math.prod(ne),
                   initial_locations=[] if device is None else ["dram" if device == "cpu" else "vram"])
        if name in self.tensors:
            raise ValueError("duplicate physical tensor: " + name)
        self.tensors[name] = row
        return name

    def op(self, name, kind, device, inputs, output, metric, workspace_key, *, extras=()):
        if any(tid not in self.tensors for tid in (*inputs, output, *extras)):
            raise ValueError("operation port absent from explicit tensor inventory")
        workspace = self.signature["workspaces"].get(workspace_key)
        if workspace is None:
            raise ValueError("missing exact operation workspace signature: " + workspace_key)
        # GPU total is unknown. This lower bound is only a runtime scheduling
        # accommodation and is never presented as a measured total or capacity.
        lower = workspace["known_gallocr_plus_cplan_bytes"]
        self.workspace_assumptions.append(dict(operation_id=name, metric=metric,
            measured_workspace_bytes=workspace["workspace_bytes"],
            runtime_workspace_bytes=lower, capacity_acceptance=False,
            interpretation="recorded_known_allocator_lower_bound_only" if device == "gpu" else "recorded_CPU_allocator_bytes"))
        self.ops.append(dict(id=name, semantic_op=kind, ggml_op="EXPLICIT_NATIVE_COMPONENT",
                             unary_op=None, op_params=[], inputs=list(inputs), output=output,
                             extra_outputs=list(extras)))
        self.costs[name] = {device: dict(duration_ns=self.price(metric), workspace_bytes=lower)}
        self.placements[name] = device
        return name

    def compute(self, name):
        self.program.append(dict(kind="compute", operation_id=name))

    def wait(self, *names):
        self.program.append(dict(kind="wait", operations=list(names), copies=[]))

    def serial(self, names):
        for name in names:
            self.compute(name)
            self.wait(name)

    def parallel_chains(self, chains):
        """Launch independent CPU/GPU stage chains at their event boundaries.

        Ordering is derived solely from frozen integer component prices. The
        actual compiler/runtime still checks readiness, resources and transfers.
        """
        active, cursors = [], [0] * len(chains)
        for index, chain in enumerate(chains):
            self.compute(chain[0])
            active.append((self.costs[chain[0]][self.placements[chain[0]]]["duration_ns"], index, chain[0]))
        while active:
            end, index, name = min(active)
            active.remove((end, index, name))
            self.wait(name)
            cursors[index] += 1
            if cursors[index] < len(chains[index]):
                following = chains[index][cursors[index]]
                self.compute(following)
                active.append((end + self.costs[following][self.placements[following]]["duration_ns"], index, following))

    def copy(self, tensor, src, dst, metric):
        route = src + "_to_" + dst
        self.routes.setdefault(route, []).append(tensor)
        self.transfers.setdefault(route, {})[tensor] = dict(duration_ns=self.price(metric))
        self.program.extend([dict(kind="transfer", tensor_id=tensor, src=src, dst=dst),
                             dict(kind="wait", operations=[], copies=[dict(tensor_id=tensor, location=dst)])])

    def finish(self, output):
        self.tensors[output]["role"] = "output"
        workload = dict(schema_version="0.2", id=self.case["id"] + "-" + self.strategy.replace("/", "-"),
                        origin=dict(kind="synthetic_layer"), tensors=list(self.tensors.values()),
                        operations=self.ops, outputs=[output])
        routes = ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")
        # These nonbinding limits are a scheduling experiment parameter, not
        # the measured device's memory capacity. Peaks are not physical claims.
        scene = dict(schema_version="0.2", id=workload["id"], architecture=dict(
            dram_capacity_bytes=1 << 60, vram_capacity_bytes=1 << 60,
            dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
            dram_alignment_bytes=1, vram_alignment_bytes=1,
            cpu_compute_slots=1, gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
            costs=dict(source="profile_only_exact_components_timing_only_NOT_capacity_profile",
                       compute=self.costs, transfer={route: self.transfers.get(route, {}) for route in routes}),
            mapspace=dict(allowed_devices={name: [device] for name, device in self.placements.items()},
                          allowed_transfers={route: self.routes.get(route, []) for route in routes},
                          allow_eviction=False, allow_copy_compute_overlap=True, allow_cpu_gpu_overlap=True),
            goal=dict(output_locations={output: "dram"}), mapper=dict(algorithm="uniform_cost"))
        ir = dict(format_version="strategy-ir-0.1", id=workload["id"], scenario_file="scenario.json",
                  information_mode="offline_fixed", regions=[], placements=self.placements,
                  program=self.program, device_states={})
        return workload, scene, ir


def _ffn(graph):
    case = graph.case
    h, i, q, cut = (case[key] for key in ("hidden", "intermediate", "tokens", "cpu_intermediate"))
    def branch(prefix, width, device):
        x = graph.tensor(prefix + "/input", (h, q), role="input", device=device)
        home = "cpu" if graph.strategy == "hybrid/preload" and device == "gpu" else device
        gate_w = graph.tensor(prefix + "/gate_weight", (h, width), role="weight", device=home)
        up_w = graph.tensor(prefix + "/up_weight", (h, width), role="weight", device=home)
        down_w = graph.tensor(prefix + "/down_weight", (width, h), role="weight", device=home)
        gate, up, activated = [graph.tensor(prefix + "/" + name, (width, q)) for name in ("gate", "up", "activated")]
        out = graph.tensor(prefix + "/output", (h, q))
        metrics = prefix.rsplit("/", 1)
        names = []
        for stage, kind, inputs, output in (
                ("gate", "ffn_gate", (x, gate_w), gate),
                ("up", "ffn_up", (x, up_w), up),
                ("activation", "ffn_swiglu", (gate, up), activated),
                ("down", "ffn_down", (activated, down_w), out)):
            metric = metrics[0] + "/" + metrics[1] + "_" + stage
            names.append(graph.op(prefix + "/" + stage, kind, device, inputs, output, metric, prefix + "/" + stage))
        return names, out
    if graph.strategy == "gpu/resident":
        names, out = branch("gpu/full", i, "gpu")
        graph.serial(names)
        graph.copy(out, "vram", "dram", "gpu/output_d2h")
        return graph.finish(out)
    cpu, cpu_out = branch("cpu/shard", cut, "cpu")
    gpu, gpu_out = branch("gpu/shard", i - cut, "gpu")
    if graph.strategy == "hybrid/preload":
        # The immutable runtime needs an absent *logical* GPU version to
        # require a copy. Native destination buffers already exist and are
        # overwritten on every invocation. This is an explicit timing-only
        # lifecycle difference, never a physical allocation/capacity claim.
        for stage in ("gate", "up", "down"):
            graph.copy("gpu/shard/" + stage + "_weight", "dram", "vram", WEIGHT_H2D[stage])
    graph.parallel_chains((cpu, gpu))
    graph.copy(gpu_out, "vram", "dram", "hybrid/output_d2h")
    out = graph.tensor("cpu/sum_output", (h, q))
    summed = graph.op("cpu/sum", "ffn_sum", "cpu", (cpu_out, gpu_out), out, "cpu/sum", "cpu/sum")
    graph.serial((summed,))
    return graph.finish(out)


def _attention(graph):
    case = graph.case
    h, q, history, heads, kvh, d = (case[key] for key in ("hidden", "tokens", "history", "heads", "kv_heads", "head_dim"))
    length = history + q
    first = "cpu" if graph.strategy == "hybrid/resident" else "gpu"
    def input_(prefix, name, shape, role="input"):
        return graph.tensor(prefix + "/" + name, shape, role=role, device=prefix)
    query = input_(first, "query", (d, q, heads))
    k, newk = input_(first, "key_cache", (d, length, kvh)), input_(first, "new_key", (d, q, kvh))
    scores = graph.tensor(first + "/scores", (length, q, heads))
    probabilities = graph.tensor(first + "/probabilities", (length, q, heads))
    qk = graph.op(first + "/qk", "attention_qk", first, (query, k, newk), scores, first + "/qk", first + "/qk")
    softmax = graph.op(first + "/softmax", "attention_softmax", first, (scores,), probabilities,
                       first + "/softmax", first + "/softmax")
    graph.serial((qk, softmax))
    if first == "cpu":
        graph.copy(probabilities, "dram", "vram", "hybrid/probabilities_h2d")
        k, newk = input_("gpu", "key_cache", (d, length, kvh)), input_("gpu", "new_key", (d, q, kvh))
    v, newv = input_("gpu", "value_cache", (d, length, kvh)), input_("gpu", "new_value", (d, q, kvh))
    context = graph.tensor("gpu/context", (h, q))
    append_k = graph.tensor("gpu/key_appended", (d, length, kvh))
    append_v = graph.tensor("gpu/value_appended", (d, length, kvh))
    pv = graph.op("gpu/pv", "attention_pv", "gpu", (probabilities, k, newk, v, newv), context,
                  "gpu/pv", "gpu/pv", extras=(append_k, append_v))
    gate = input_("gpu", "attention_gate", (h, q))
    hidden = input_("gpu", "hidden", (h, q))
    weight = input_("gpu", "output_weight", (h, h), "weight")
    out = graph.tensor("gpu/output", (h, q))
    output = graph.op("gpu/output", "attention_output", "gpu", (context, gate, hidden, weight), out,
                      "gpu/output", "gpu/attention_output")
    graph.serial((pv, output))
    graph.copy(out, "vram", "dram", "gpu/output_d2h")
    return graph.finish(out)


def bind_resident(case, strategy, frozen_case, signature):
    if signature["case"] != case:
        raise ValueError("physical signature case mismatch")
    graph = _Graph(case, strategy, frozen_case, signature)
    workload, scene, ir = (_ffn if case["family"] == "ffn" else _attention)(graph)
    used = a.terms(frozen_case["strategies"][strategy])
    if set(graph.rounded_prices) != used:
        raise ValueError("not every declared component was faithfully bound")
    compiled = compile_strategy(parse_strategy_ir(ir), build_scenario(workload, scene))
    replay = run_strategy(compiled)
    if replay.evaluation.status != "valid":
        raise ValueError("compiled runtime rejected the measured path: " + str(replay.evaluation.reason))
    reference = a.predict(frozen_case["strategies"][strategy], graph.rounded_prices)
    if replay.evaluation.makespan_ns != reference:
        raise ValueError("compiler/runtime differs from predeclared component timing semantics")
    binding = dict(schema="exact-measured-fixed-ir-binding-0.1", case_id=case["id"], strategy=strategy,
                   physical_signature_sha256=canonical_hash(signature), measured_physical_signature=signature,
                   timing_only=True, physical_capacity_validated=False, capacity_fields_are_nonphysical=True,
                   unknown_cuda_workspace_preserved=True, workspace_assumptions=graph.workspace_assumptions,
                   component_bindings=graph.cost_binding, rounding="ceil each positive median to integer nanoseconds",
                   independently_replayed_makespan_ns=replay.evaluation.makespan_ns,
                   formula_with_integer_prices_ns=reference,
                   high_level_ModelPlan_transform_lowering_exercised=False,
                   frontend_scope="Python-authored fixed tensor StrategyIR; common compiler and Modeling runtime",
                   full_Dense_MoE_inference_validated=False)
    if strategy == "hybrid/preload":
        binding["preload_buffer_lifecycle_equivalence_validated"] = False
        binding["preload_representation_difference"] = (
            "Native repeats H2D into already allocated GPU buffers. The immutable runtime represents a fresh required "
            "logical weight version whose GPU copy starts absent. Exact H2D timing/ports/order are replayed; "
            "allocation, overwrite-lifecycle and physical capacity equivalence are not validated.")
    return compiled, replay, binding, workload, scene


def run(experiment, output, *, batch_mode=False, host_interference_note=None):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory")
    preserved = [experiment / name for name in ("suite.json", "manifest.json", "measurements.json",
                                                "prediction-model.json", "analysis/report.json")]
    preserved.extend(sorted((experiment / "raw").glob("*.json")))
    source_before = {str(path): a.sha(path) for path in preserved}
    runtime_before = runtime_fingerprint()
    batch_verifier = None
    if batch_mode:
        verifier_path = ROOT / "profiling/batched_calibration/verify_batch.py"
        spec = importlib.util.spec_from_file_location("binding_batch_verifier", verifier_path)
        batch_verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch_verifier)
        audited = batch_verifier.verify(experiment)
    else:
        audited = verify(experiment)
    suite = protocol.document(experiment / "suite.json")
    manifest = protocol.document(experiment / "manifest.json")
    rows = protocol.document(experiment / "measurements.json")
    model = protocol.document(experiment / "prediction-model.json")
    saved = protocol.document(experiment / "analysis/report.json")
    output.mkdir(parents=True)
    a.dump(output / "source-verification.json", audited)
    results, rejected = [], []
    for case in suite["cases"]:
        selected = [row for row in rows if row["case"]["id"] == case["id"]]
        measurement_contract = (dict(kind="completed_batch_mean", batch_repetitions=suite["batch_repetitions"],
            raw_timing_unit=audited["raw_timing_unit"], timing_unit=audited["timing_unit"],
            normalization=audited["normalization"], isolated_latency_validated=False) if batch_mode else
            dict(kind="isolated_completed_invocation", timing_unit="steady_clock_completed_work_ns"))
        signature = physical_signature(case, selected, dict(**suite["protocol"],
            measurement_contract=measurement_contract,
            observed_child_flags=manifest["observed_child_flags"],
            environment_before=manifest["environment_before"], environment_after=manifest.get("environment_after"),
            executed_assets_sha256=manifest["inventory_before"],
            executed_context_source_sha256=(None if batch_mode else audited["context_native_source_sha256"])))
        frozen = next(item for item in model["cases"] if item["case_id"] == case["id"])
        for strategy in protocol.recipes(case):
            if strategy == "hybrid/preload" and not batch_mode:
                rejected.append(dict(case_id=case["id"], strategy=strategy, status="unsupported",
                    reason="One aggregate three-weight H2D timing cannot be divided into three exact per-tensor COPY costs. Collect each weight H2D separately, or add an explicit validated ordered-copy bundle action."))
                continue
            compiled, replay, binding, workload, scene = bind_resident(case, strategy, frozen, signature)
            target = output / case["id"] / strategy.replace("/", "-")
            write_run(target, compiled, replay)
            a.dump(target / "physical-binding.json", binding)
            a.dump(target / "workload.json", workload)
            a.dump(target / "scenario.TIMING-ONLY.json", scene)
            previous = next(item for item in saved["results"] if item["case_id"] == case["id"] and item["strategy"] == strategy)
            heldout = [row for row in selected if row["split"] == "heldout"]
            observed = statistics.median([value for row in heldout for value in row["metrics"][strategy]])
            predicted = replay.evaluation.makespan_ns
            gates = deepcopy(previous["gates"])
            gates["heldout_relative_error"] = abs(predicted - observed) / observed <= suite["quality"]["max_heldout_relative_error"]
            results.append(dict(case_id=case["id"], strategy=strategy, status="replayed",
                predicted_ns=predicted, frozen_formula_prediction_ns=previous["predicted_ns"],
                observed_ns=observed, absolute_relative_error=abs(predicted - observed) / observed,
                gates=gates, latency_quality_accepted=all(gates.values()), physical_capacity_validated=False,
                binding_path=str((target / "physical-binding.json").relative_to(output)),
                physical_signature_sha256=binding["physical_signature_sha256"]))
    source_after = {str(path): a.sha(path) for path in preserved}
    if source_before != source_after or runtime_before != runtime_fingerprint():
        raise ValueError("source evidence or simulator implementation changed during replay")
    report = dict(schema="measured-fixed-ir-replay-report-0.2", experiment=str(experiment),
        frozen_component_model_sha256=a.sha(experiment / "prediction-model.json"),
        audit_source_sha256=a.sha(__file__), runtime_fingerprint=runtime_before,
        preserved_source_sha256=source_before,
        source_verification=audited, exact_paths_replayed=len(results),
        exact_resident_paths_replayed=sum(row["strategy"].endswith("resident") for row in results),
        latency_accepted=sum(item["latency_quality_accepted"] for item in results),
        high_level_ModelPlan_transform_lowering_exercised=False, physical_capacity_validated=False,
        no_heldout_duration_used_as_cost=True, source_measurements_unchanged=True,
        measurement_contract=("completed_batch_mean" if batch_mode else "isolated_completed_invocation"),
        host_interference_note=host_interference_note or "Host activity was not controlled or fully observed; no host-idle claim.",
        results=results, unsupported_paths=rejected,
        remaining_gaps=["High-level ModelPlan AttentionStages/FFNPartition semantic lowering has different physical boundaries and is not calibrated by this replay.",
            "Full Dense/MoE/multilayer/multiround graph calibration is absent.",
            "CUDA private workspace is unknown; replay scenarios use a documented allocator lower bound and artificial nonbinding capacities.",
            "Exact source-process quality failures remain failures after correct IR/runtime replay."])
    if batch_mode:
        report["remaining_gaps"].extend([
            "Repeated-invocation batch means do not establish isolated-request latency variability.",
            "Preload uses an absent logical GPU weight version to require H2D; native buffers are already allocated and overwritten. Physical overwrite/allocation lifecycle is not validated."])
    a.dump(output / "report.json", report)
    lines = ["# Measured component binding to fixed tensor IR", "",
        "Python constructs the measured task/port graph and fixed StrategyIR, the existing compiler emits actions, and the Modeling runtime replays them. This is **not** evidence for the higher-level ModelPlan semantic transforms or full LLM accuracy.", "",
        f"- Exact component-cost paths replayed: {len(results)} (resident: {report['exact_resident_paths_replayed']}).",
        f"- Latency quality accepted: {report['latency_accepted']}/{len(results)}.",
        f"- Unsupported preload paths: {len(rejected)}.",
        "- Timing contract: " + report["measurement_contract"] + ".",
        "- Host condition: " + report["host_interference_note"],
        "- Physical capacity accepted: false. CUDA total workspace remains null; only recorded allocator lower bounds are inserted into a deliberately nonbinding timing ledger.",
        "- Every duration comes from the pre-heldout frozen independent component model. Validation totals are read only for comparison. No fitting or interpolation is performed.", "",
        "| Case | Strategy | Runtime prediction (us) | Heldout total (us) | Relative error | Accepted |",
        "|---|---|---:|---:|---:|---|"]
    for row in results:
        lines.append(f"| {row['case_id']} | {row['strategy']} | {row['predicted_ns']/1000:.3f} | {row['observed_ns']/1000:.3f} | {row['absolute_relative_error']:.2%} | {row['latency_quality_accepted']} |")
    lines.extend(["", "## Remaining gaps", "", *("- " + text for text in report["remaining_gaps"]), "",
        "Per-path physical-binding.json records the source stage metrics, profile-only raw evidence, full physical signature, measured unknown workspace and explicit lower-bound scheduling assumptions. strategy.json, compiled.json, execution.json and events.json permit independent lowering/action inspection. scenario.TIMING-ONLY.json is diagnostic evidence and must never be installed as an accepted capacity profile."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", action="store_true", help="Verify/replay explicit completed-work batch means, including independent per-weight H2D.")
    parser.add_argument("--host-interference-note", help="Recorded collection condition; never implies the host was idle.")
    args = parser.parse_args(argv)
    try:
        report = run(args.experiment, args.output, batch_mode=args.batch,
                     host_interference_note=args.host_interference_note)
        print(json.dumps({key: report[key] for key in ("exact_paths_replayed", "latency_accepted", "physical_capacity_validated")}))
        return 0  # Successful audit; quality remains a separate report result.
    except (OSError, ValueError, KeyError, TypeError, SpecError) as exc:
        print("measured IR replay: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
