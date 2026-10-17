"""Bind exact stage-port evidence to the tensor performance engine.

CPU FFN and Attention graphs execute through Modeling with real known cplan
workspace. Quality failures produce a diagnostic replay, never accepted costs.
GPU capacity replay is explicitly refused while private scratch is unknown.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "modeling")]
import experiment as e
from export_profiles import stage_profiles, PORTS
from llm_infer_model.tensor.spec import build_scenario
from llm_infer_model.tensor.engine import Action, ACTION_COMPUTE
from llm_infer_model.tensor.program import ActionStep, WaitStep, execute_program


def build_cpu_binding(case, entries, *, diagnostic):
    order = ("gate", "up", "activation", "down") if case["family"] == "ffn" else ("attention_core", "attention_output")
    selected = {x["stage"]: x for x in entries if x["case_id"] == case["id"] and x["device"] == "cpu"}
    if set(selected) != set(order):
        raise ValueError("exact CPU stage coverage is missing")
    if not diagnostic and any(not x["latency_quality_accepted"] for x in selected.values()):
        raise ValueError("quality-rejected stage cannot enter an accepted cost table")
    if any(x["workspace_bytes"] is None for x in selected.values()):
        raise ValueError("unknown workspace cannot be replaced by zero")
    roles = {}; ports = {}
    outputs = set(); initial = set()
    operations = []
    semantics = {"gate": "expert_gate", "up": "expert_up", "activation": "expert_swiglu", "down": "expert_down",
                 "attention_core": "attention_core", "attention_output": "attention_output"}
    for stage in order:
        entry = selected[stage]
        inputs, outs = PORTS[stage]
        for name, port in {**entry["inputs"], **entry["outputs"]}.items():
            if name in ports and ports[name] != port:
                raise ValueError("stage boundary port identity mismatch")
            ports[name] = port
        initial.update(n for n in inputs if n not in outputs)
        outputs.update(outs)
        operations.append({"id": stage, "semantic_op": semantics[stage], "ggml_op": "SYNTHETIC_MACRO",
                           "unary_op": None, "op_params": [], "inputs": inputs, "output": outs[0], "extra_outputs": outs[1:]})
    tensors = []
    for name, port in ports.items():
        ne = [*port["ne"], 1]
        nb = [4, ne[0] * 4, ne[0] * ne[1] * 4, ne[0] * ne[1] * ne[2] * 4]
        role = "weight" if name.endswith("weight") else "input" if name in initial else "output" if name == "output" else "intermediate"
        tensors.append({"id": name, "name": name, "role": role, "dtype": "GGML_TYPE_F32", "ne": ne, "nb": nb,
                        "storage_bytes": port["bytes"], "initial_locations": ["dram"] if name in initial else []})
    # The runtime's row labels intentionally require a synthetic graph origin.
    # Physical semantics/values are bound externally by the measured sidecar;
    # this is not a measured moe-execution frontend or a relaxed provenance gate.
    workload = {"schema_version": "0.2", "id": case["id"] + "-measured-stages", "origin": {"kind": "synthetic_layer"},
                "tensors": tensors, "operations": operations, "outputs": ["output"]}
    # This is an explicit logical replay budget, not a claim to have calibrated
    # allocator alignment or complete physical process memory.
    logical_budget = sum(t["storage_bytes"] for t in tensors) + max(x["workspace_bytes"] for x in selected.values())
    scenario_doc = {"schema_version": "0.2", "id": workload["id"],
        "architecture": {"dram_capacity_bytes": max(1, logical_budget), "vram_capacity_bytes": 1,
                         "dram_alignment_bytes": 1, "vram_alignment_bytes": 1,
                         "dram_runtime_reserved_bytes": 0, "vram_runtime_reserved_bytes": 0,
                         "cpu_compute_slots": 1, "gpu_compute_slots": 1, "pcie_copy_slots": 1, "ssd_io_slots": 1},
        "costs": {"source": "quality_rejected_measured_diagnostic" if diagnostic else "accepted_matched_stage_profile",
                  "compute": {stage: {"cpu": {"duration_ns": math.ceil(selected[stage]["duration_ns"]),
                                                 "workspace_bytes": selected[stage]["workspace_bytes"]}} for stage in order},
                  "transfer": {r: {} for r in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")}},
        "mapspace": {"allowed_devices": {stage: ["cpu"] for stage in order},
                     "allowed_transfers": {r: [] for r in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")},
                     "allow_eviction": False, "allow_copy_compute_overlap": False, "allow_cpu_gpu_overlap": False},
        "goal": {"output_locations": {"output": "dram"}}, "mapper": {"algorithm": "uniform_cost"}}
    scenario = build_scenario(workload, scenario_doc)
    steps = []
    for n, stage in enumerate(order):
        steps += [ActionStep(Action(ACTION_COMPUTE, target_id=stage, device="cpu"), source_index=2*n),
                  WaitStep(operations=(stage,), source_index=2*n+1)]
    result = execute_program(scenario, steps)
    if result.evaluation.status != "valid":
        raise ValueError("declared matched CPU graph replay failed: " + str(result.evaluation.reason))
    return workload, scenario_doc, result


def replay(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError("replay output must be new")
    profiles = stage_profiles(directory)
    suite = e.document(directory / "suite.json")
    native = e.document(directory / "report.json")
    lookup = {(r["case_id"], r["metric"]): r for r in native["results"]}
    runtime = HERE.parents[1] / "modeling" / "llm_infer_model" / "tensor"
    executed_sources = e.inventory([Path(__file__), HERE / "export_profiles.py", HERE / "verify.py",
                                    HERE / "binding.py", *[runtime / n for n in ("engine.py", "program.py", "spec.py")]])
    output.mkdir(parents=True)
    e.dump(output / "stage-port-profiles.json", profiles)
    results = []
    for case in suite["cases"]:
        if case["backend"] in ("cpu", "hybrid"):
            entries = [x for x in profiles["entries"] if x["case_id"] == case["id"] and x["device"] == "cpu"]
            rejected = [x["stage"] for x in entries if not x["latency_quality_accepted"]]
            workload, scenario_doc, run = build_cpu_binding(case, entries, diagnostic=bool(rejected))
            predicted = run.evaluation.makespan_ns
            truth = lookup[(case["id"], "cpu/serial")]
            error = abs(predicted - truth["observed_ns"]) / truth["observed_ns"]
            calibration_accepted = (not rejected and truth["calibrated"] and error <= suite["quality"]["max_heldout_relative_error"])
            e.dump(output / (case["id"] + ".workload.json"), workload)
            e.dump(output / (case["id"] + ".scenario.json"), scenario_doc)
            results.append({"case_id": case["id"], "device": "cpu", "modeled_execution_status": run.evaluation.status,
                "predicted_ns": predicted, "heldout_serial_ns": truth["observed_ns"], "heldout_relative_error": error,
                "quality_rejected_stages": rejected, "calibration_accepted": calibration_accepted,
                "purpose": "accepted_latency_validation" if calibration_accepted else "diagnostic_only_not_a_cost_table",
                "workspace_unknown_replaced_by_zero": False, "physical_capacity_validation": False,
                "graph_provenance": "synthetic_declared_port_graph_with_external_real_measurement_binding",
                "profile_model_sha256": profiles["prediction_model_sha256"],
                "actions": [{"kind": a.kind, "target_id": a.target_id, "device": a.device} for a in run.evaluation.actions]})
        if case["backend"] in ("gpu", "hybrid"):
            results.append({"case_id": case["id"], "device": "gpu", "modeled_execution_status": "unsupported",
                "reason": "private GPU workspace unknown; no capacity replay or invented zero",
                "calibration_accepted": False, "latency_only_heldout_diagnostic": lookup[(case["id"], "gpu/serial")]})
    report = {"schema": "matched-modeling-serial-replay-0.1", "source_stage_profiles_sha256": e.sha(output / "stage-port-profiles.json"),
              "search_performed": False, "old_m2_fixtures_modified": False, "results": results,
              "executed_source_sha256": executed_sources,
              "accepted_cpu_graphs": sum(x["calibration_accepted"] for x in results if x["device"] == "cpu")}
    if e.inventory(executed_sources) != executed_sources:
        raise ValueError("Modeling/replay source changed during validation")
    e.dump(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--experiment", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        report = replay(args.experiment, args.output)
        print(e.canonical({"output": str(args.output), "accepted_cpu_graphs": report["accepted_cpu_graphs"], "results": len(report["results"])})); return 0
    except (OSError, ValueError, KeyError) as exc:
        print("matched serial replay: " + str(exc), file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
