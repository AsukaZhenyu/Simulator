"""Export measured stage-port profiles without turning unknown memory into zero.

This sidecar is an evidence interface, not an automatic M2 cost-table conversion.
Consumers must match the exact operation/port/materialization contract and may
only use quality-accepted latency. GPU workspace remains unavailable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as e
from verify import verify

PORTS = {
    "gate": (["input", "gate_weight"], ["gate"]),
    "up": (["input", "up_weight"], ["up"]),
    "activation": (["gate", "up"], ["activated"]),
    "down": (["activated", "down_weight"], ["output"]),
    "attention_core": (["query", "new_key", "new_value", "key_cache", "value_cache"],
                       ["context", "key_appended", "value_appended"]),
    "attention_output": (["context", "attention_gate", "output_weight", "hidden"], ["output"]),
}


def stage_profiles(directory):
    directory = Path(directory).resolve()
    verification = verify(directory)
    model = e.document(directory / "prediction-model.json")
    report = e.document(directory / "report.json")
    measured = e.document(directory / "measurements.json")
    manifest = e.document(directory / "manifest.json")
    lookup = {(r["case_id"], r["metric"]): r for r in report["results"]}
    by_id = {r["id"]: r for r in measured}
    entries = []
    for item in model["entries"]:
        metric = item["metric"]
        parts = metric.split("/")
        if len(parts) != 2 or parts[0] not in ("cpu", "gpu") or parts[1] not in PORTS:
            continue
        device, stage = parts
        rows = [by_id[rid] for rid in item["profile_rows"]]
        inputs, outputs = PORTS[stage]
        workspace = item["workspace"]
        accepted = lookup[item["case_id"], metric]["calibrated"]
        recomputed = {"gate", "up", "activated", "output"} if item["case"]["family"] == "ffn" else {"context", "output", "key_appended", "value_appended"}
        role = lambda n: "initial_oracle_reference_recomputed_during_execution" if n in recomputed else "immutable_initial_native_input"
        entries.append({"case_id": item["case_id"], "stage": stage, "device": device,
            "dtype": "f32", "batch": 1, "geometry": {k: item["case"][k] for k in
                ("hidden", "intermediate", "tokens", "history", "heads", "kv_heads", "head_dim")},
            "history_axis": "not_applicable_no_KV_input" if item["case"]["family"] == "ffn" else "exact_visible_history",
            "duration_ns": item["median_ns"], "latency_quality_accepted": accepted,
            "workspace_bytes": workspace["workspace_bytes"], "workspace_source": workspace["source"],
            "workspace_known": device == "cpu", "capacity_validation_performed": False,
            "inputs": {n: {**item["ports"][n], "fixture_identity_scope": role(n)} for n in inputs},
            "outputs": {n: {**item["ports"][n], "fixture_identity_scope": role(n)} for n in outputs},
            "source_fixtures": {row["id"]: {n: {**row["fixture_hashes"][n], "identity_scope": role(n)} for n in inputs} for row in rows},
            "timed_intermediate_bit_identity_captured": False,
            "semantics": "plain expert FFN without router/gather/scatter/norm/residual" if item["case"]["family"] == "ffn"
                else "post-RoPE pure GQA core materializes the complete appended KV; output includes sigmoid gate, O projection, residual",
            "source_metrics": {"profile": item["profile_rows"], "heldout": [r["row_id"] for r in lookup[item["case_id"], metric]["processes"] if r["split"] == "heldout"]}})
    return {"schema": "matched-stage-port-profiles-0.1", "experiment": str(directory),
            "verification": verification, "prediction_model_sha256": e.sha(directory / "prediction-model.json"),
            "experiment_inventory": manifest["inventory_before"], "interpolation": "none",
            "automatic_m2_conversion": False, "entries": entries,
            "unmeasured": ["quantized stages", "multi-thread resource utilization law", "general shared-link fairness law",
                           "GPU private workspace", "arbitrary SM copy implementations", "CUDA graphs enabled"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.output.exists(): raise ValueError("output must be new")
        profiles = stage_profiles(args.experiment)
        e.dump(args.output, profiles)
        print(e.canonical({"entries": len(profiles["entries"]), "accepted_latency_entries": sum(x["latency_quality_accepted"] for x in profiles["entries"]), "output": str(args.output)}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print("matched profile export: " + str(exc), file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
