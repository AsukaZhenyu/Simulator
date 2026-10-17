"""Measure a reusable F32 FFN resident/reload subset and CUDA allocator APIs.

The existing matched-mechanism native probe is reused without modification.
Allocator API timings do not identify a mapping policy's reconfiguration cost.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from profiling.matched_mechanisms import experiment as matched
from profiling.run_measurements import environment

ROOT = Path(__file__).resolve().parent
VERSION = "reconfiguration-measurement-suite-0.1"
ALLOC_FIELDS = {"id", "bytes", "device", "warmup", "samples", "seed"}
ALLOC_METRICS = {"cuda_malloc", "cuda_free", "pinned_h2d_completed", "idle_stream_synchronize"}


def validate_suite(original):
    if not isinstance(original, dict) or set(original) != {
        "format_version", "ffn_suite", "allocator_cases", "allocator_repeats", "run_context"
    } or original["format_version"] != VERSION:
        raise ValueError("all exact reconfiguration suite fields required")
    matched.validate_suite(original["ffn_suite"])
    if any(c["family"] != "ffn" or c["backend"] != "hybrid" for c in original["ffn_suite"]["cases"]):
        raise ValueError("resident/reload comparison requires explicit hybrid FFN cases")
    if original["ffn_suite"]["transfers"]:
        raise ValueError("this subset uses explicit matched FFN path copies, not separate transfer cases")
    matched.integer(original["allocator_repeats"], "allocator_repeats", 2, 8)
    if not isinstance(original["run_context"], str) or not original["run_context"].strip():
        raise ValueError("run_context must explicitly describe expected interference")
    cases = original["allocator_cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("explicit nonempty allocator cases required")
    ids = set()
    for c in cases:
        if not isinstance(c, dict) or set(c) != ALLOC_FIELDS:
            raise ValueError("allocator case fields must be exact")
        if not isinstance(c["id"], str) or not c["id"] or any(
            x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for x in c["id"]
        ) or c["id"] in ids:
            raise ValueError("allocator id must be unique and safe")
        ids.add(c["id"])
        for key, lo, hi in (("bytes", 1024, 67108864), ("device", 0, 63), ("warmup", 0, 1000),
                            ("samples", 3, 1000), ("seed", 0, 1000000)):
            matched.integer(c[key], key, lo, hi)
        if c["device"] != original["ffn_suite"]["compute_protocol"]["device"]:
            raise ValueError("allocator and compute logical device must match")
    return original


def allocator_command(binary, case):
    return [str(binary), *[value for key in ("bytes", "device", "warmup", "samples", "seed")
                          for value in ("--" + key, str(case[key]))]]


def validate_allocator(case, payload):
    if not isinstance(payload, dict) or payload.get("schema") != "reconfiguration-allocator-native-0.1":
        raise ValueError("native allocator schema mismatch")
    for key in ALLOC_FIELDS - {"id"}:
        matched.integer(payload.get(key), key, 0, 67108864)
        if payload.get(key) != case[key]:
            raise ValueError("native allocator case binding mismatch: " + key)
    for key, value in {
        "host_memory": "cudaMallocHost_pinned",
        "device_allocator": "direct_cudaMalloc_cudaFree_not_GGML_or_async_pool",
        "timer": "steady_clock_completed_api_wall", "cuda_graphs": "not_used",
        "validation": "all_bytes_exact_each_iteration", "workspace_bytes": None,
        "logical_evict_cost_ns": None, "mapper_control_cost_ns": None,
        "allocator_release_guaranteed_by_observed_free_delta": False,
    }.items():
        if key not in payload or payload[key] != value or (value is False and payload[key] is not False):
            raise ValueError("native allocator physical semantics mismatch: " + key)
    if not isinstance(payload.get("device_name"), str) or not payload["device_name"]:
        raise ValueError("native device provenance missing")
    for key in ("cuda_driver_version", "cuda_runtime_version"):
        matched.integer(payload.get(key), key, 1, 999999)
    if set(payload.get("wall_ns", {})) != ALLOC_METRICS:
        raise ValueError("native allocator metric coverage mismatch")
    for values in payload["wall_ns"].values():
        matched.stats(values)
        if len(values) != case["samples"]:
            raise ValueError("allocator sample count mismatch")
    checkpoints = payload.get("memory_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != case["samples"]:
        raise ValueError("allocator checkpoints must cover every sample")
    fields = {"before_free_bytes", "allocated_free_bytes", "after_free_bytes", "total_bytes"}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != fields:
            raise ValueError("memory checkpoint fields mismatch")
        for key, value in checkpoint.items():
            matched.integer(value, key, 0, 1 << 64)
        if not checkpoint["total_bytes"] or any(checkpoint[k] > checkpoint["total_bytes"] for k in fields - {"total_bytes"}):
            raise ValueError("impossible native memory checkpoint")
    return payload


def nondegenerate(payload):
    results = {}
    for name in ("input", "gate_weight", "up_weight", "down_weight"):
        values = payload["source_values"][name]
        coarse = sum(v * 2048 == round(v * 2048) for v in values) / len(values)
        passed = len(set(values)) >= min(16, len(values)) and coarse < .05
        results[name] = {"elements": len(values), "distinct_values": len(set(values)),
                         "exact_integer_over_2048_fraction": coarse, "passed": passed}
    return results


def summarize(directory):
    directory = Path(directory)
    suite = validate_suite(matched.document(directory / "suite.json"))
    rows = matched.document(directory / "ffn" / "measurements.json")
    ffn_report = matched.document(directory / "ffn" / "report.json")
    lookup = {(r["case_id"], r["metric"]): r for r in ffn_report["results"]}
    comparisons, numerical_inputs = [], {}
    for row in rows:
        if row["status"] == "measured":
            raw = matched.document(directory / "ffn" / "raw" / (row["id"] + ".json"))
            numerical_inputs[row["id"]] = nondegenerate(json.loads(raw["stdout"]))
    for case in suite["ffn_suite"]["cases"]:
        pairs = []
        for row in rows:
            if row["case"]["id"] == case["id"] and row["status"] == "measured":
                resident = statistics.median(row["metrics"]["gpu/serial"])
                reload = statistics.median(row["metrics"]["hybrid/gpu_weights"])
                pairs.append({"row_id": row["id"], "split": row["split"],
                              "resident_median_ns": resident, "reload_path_median_ns": reload,
                              "reload_over_resident_ratio": reload / resident})
        a = lookup.get((case["id"], "gpu/serial"))
        b = lookup.get((case["id"], "hybrid/gpu_weights"))
        comparisons.append({"case_id": case["id"], "shape": case,
            "weight_bytes": 3 * case["hidden"] * case["intermediate"] * 4,
            "same_process_same_fixture_pairs_grouped_not_interleaved": pairs,
            "resident_quality": a, "reload_path_quality": b,
            "both_latency_gates_passed": bool(a and b and a["calibrated"] and b["calibrated"]),
            "workspace_bytes": None,
            "interpretation": "same four-stage F32 GGML FFN; reload overwrites preallocated weights and includes pinned staging memcpy; allocation outside FFN timing"})
    alloc_rows = matched.document(directory / "allocator-measurements.json")
    alloc_metrics = []
    for case in suite["allocator_cases"]:
        selected = [r for r in alloc_rows if r["case"] == case and r["status"] == "measured"]
        for metric in sorted(ALLOC_METRICS):
            process = [{"row_id": r["id"], **matched.stats(r["payload"]["wall_ns"][metric])} for r in selected]
            medians = [p["median_ns"] for p in process]
            between = statistics.pstdev(medians) / statistics.mean(medians) if len(medians) > 1 else None
            quality = suite["ffn_suite"]["quality"]
            gates = {"within_process": bool(process) and all(p["cv"] <= quality["max_within_process_cv"] for p in process),
                     "between_process": between is not None and between <= quality["max_between_process_cv"],
                     "all_declared_repeats_measured": len(selected) == suite["allocator_repeats"]}
            alloc_metrics.append({"case_id": case["id"], "metric": metric, "processes": process,
                                  "between_process_cv": between, "gates": gates,
                                  "stable_api_observation": all(gates.values()),
                                  "heldout_prediction_validated": False})
    return {"schema": "reconfiguration-measurement-report-0.1", "comparisons": comparisons,
            "input_nondegeneracy": numerical_inputs,
            "all_input_nondegeneracy_checks_passed": bool(numerical_inputs) and all(c["passed"] for row in numerical_inputs.values() for c in row.values()),
            "ffn_numeric_execution_valid": ffn_report["numeric_execution_valid"],
            "ffn_all_declared_metrics_calibrated": ffn_report["all_declared_metrics_calibrated"],
            "ffn_metrics_calibrated": sum(r["calibrated"] for r in ffn_report["results"]),
            "ffn_metrics_total": len(ffn_report["results"]), "allocator_api_metrics": alloc_metrics,
            "allocator_numeric_execution_valid": bool(alloc_rows) and all(r["status"] == "measured" for r in alloc_rows),
            "gpu_capacity_proved": False, "workspace_bytes": None, "mapper_control_cost_ns": None,
            "logical_evict_cost_ns": None, "real_llm_or_paper_reproduction": False,
            "does_not_calibrate": ["full LLM", "GPU private workspace", "CPU load to utilization law", "cooperative reclaim controller", "GGML allocator return", "CUDA graph execution", "arbitrary overlap"]}


def run(args):
    suite_path = args.suite.resolve()
    suite = validate_suite(matched.document(suite_path))
    if args.dry_run:
        print(matched.canonical({"suite": suite, "compute_binary": str(args.compute_binary),
                                 "allocator_binary": str(args.allocator_binary), "hardware_called": False}))
        return 0
    if args.output is None or args.compute_binary is None or args.allocator_binary is None or not args.runtime_dir:
        raise ValueError("actual run requires explicit output, compute-binary, allocator-binary, runtime-dir")
    if args.timeout_s < 1:
        raise ValueError("timeout-s must be positive")
    for path in (args.compute_binary, args.allocator_binary):
        if not path.is_file():
            raise ValueError("requested native binary does not exist; no fallback")
    out = args.output.resolve()
    if out.exists():
        raise ValueError("output must be a new directory")
    files = [suite_path, *ROOT.glob("*.py"), *(ROOT / "native").glob("*.cu"),
             ROOT / "native" / "CMakeLists.txt", args.allocator_binary.resolve()]
    before = matched.inventory(files)
    out.mkdir(parents=True)
    (out / "raw").mkdir()
    (out / "source").mkdir()
    for path in files:
        if path.parent == ROOT or path.parent == ROOT / "native":
            (out / "source" / path.name).write_bytes(path.read_bytes())
    matched.dump(out / "suite.json", suite)
    matched.dump(out / "ffn-suite.json", suite["ffn_suite"])
    manifest = {"schema": "reconfiguration-measurement-manifest-0.1",
                "started_utc": datetime.now(timezone.utc).isoformat(), "inventory_before": before,
                "run_context": suite["run_context"], "environment_before": environment(),
                "source_paths_are_actual_executed_paths": True, "sample_filtering": "none",
                "system_settings_changed": False, "status": "running"}
    matched.dump(out / "manifest.json", manifest)
    ffn_args = argparse.Namespace(suite=out / "ffn-suite.json", output=out / "ffn",
                                 compute_binary=args.compute_binary, transfer_binary=None,
                                 runtime_dir=args.runtime_dir, timeout_s=args.timeout_s)
    matched.run(ffn_args)
    alloc_rows = []
    lock = REPO / "profiling" / ".measurement.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        child_env, flags = matched.child_context(args.runtime_dir)
        for case in suite["allocator_cases"]:
            for repeat in range(suite["allocator_repeats"]):
                row = {"id": f"{case['id']}-r{repeat}", "case": case, "repeat": repeat,
                       "command": allocator_command(args.allocator_binary.resolve(), case)}
                try:
                    proc = subprocess.run(row["command"], env=child_env, capture_output=True, text=True,
                                          timeout=args.timeout_s)
                    raw = {"command": row["command"], "returncode": proc.returncode, "stdout": proc.stdout,
                           "stderr": proc.stderr, "child_flags": flags}
                    matched.dump(out / "raw" / (row["id"] + ".json"), raw)
                    if proc.returncode:
                        raise ValueError("native allocator failed: " + str(proc.returncode))
                    row["payload"] = validate_allocator(case, json.loads(proc.stdout))
                    row["status"] = "measured"
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    row["status"], row["error"] = "failed", str(exc)
                    if not (out / "raw" / (row["id"] + ".json")).exists():
                        matched.dump(out / "raw" / (row["id"] + ".json"), {"command": row["command"], "error": str(exc)})
                row["raw_sha256"] = matched.sha(out / "raw" / (row["id"] + ".json"))
                alloc_rows.append(row)
                matched.dump(out / "allocator-measurements.json", alloc_rows)
                print(row["id"] + ": " + row["status"], flush=True)
    finally:
        lock.unlink(missing_ok=True)
    report = summarize(out)
    matched.dump(out / "report.json", report)
    after = matched.inventory(before)
    manifest.update(inventory_after=after, assets_unchanged=before == after,
                    environment_after=environment(), completed_utc=datetime.now(timezone.utc).isoformat(),
                    status="completed")
    matched.dump(out / "manifest.json", manifest)
    if before != after:
        raise ValueError("executed source or native binary changed during measurement")
    print(matched.canonical({"output": str(out), "ffn_quality_pairs": sum(c["both_latency_gates_passed"] for c in report["comparisons"]),
                            "ffn_case_count": len(report["comparisons"]), "gpu_workspace_known": False}))
    return 0 if report["ffn_numeric_execution_valid"] and report["allocator_numeric_execution_valid"] and report["all_input_nondegeneracy_checks_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--compute-binary", type=Path)
    parser.add_argument("--allocator-binary", type=Path)
    parser.add_argument("--runtime-dir", type=Path, action="append")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print("reconfiguration measurement: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
