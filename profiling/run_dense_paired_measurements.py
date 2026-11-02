"""Same-resident CPU stage-chain diagnostics; never exports M2 costs.

The three partition graphs share the joint graph's allocations. These timings
must not silently replace independent-resident macro costs or workspace.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

MODELING = Path(__file__).resolve().parents[1] / "modeling"
sys.path.insert(0, str(MODELING))
from llm_infer_model.m2.dense_contract import build_plan, native_arguments, resident_accounting, validate_config, ports
from llm_infer_model.tensor.spec import SpecError
from profile_contract import summarize_samples
from run_measurements import dump, environment, sha256

ROOT = Path(__file__).resolve().parent
SCHEMA = "dense_paired_measurement_profile_v1"
NATIVE_SCHEMA = "ggml_dense_layer_paired_probe/1"
STAGES = ("projection", "attention", "dense_ffn")


def _same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{label} differs from the declared paired experiment")


def samples(value, count, label):
    if not isinstance(value, list) or len(value) != count or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError(f"{label} requires {count} positive integral nanosecond observations")
    return value


def paired_command(case, binary, args):
    arguments = native_arguments(case, args.warmup, args.samples, args.seed)
    arguments[arguments.index("--stage") + 1] = "paired"
    command = [str(binary), *arguments, "--compute-mode", args.compute_mode]
    if getattr(args, "min_warmup_ms", None) is not None:
        command += ["--min-warmup-ms", str(args.min_warmup_ms)]
        if args.cpu_mask is not None:
            command += ["--cpu-mask", ",".join(str(value) for value in args.cpu_mask)]
    return command


def validate_paired_binding(case, native, args):
    if not isinstance(native, dict):
        raise ValueError("native paired payload must be one object")
    expected = {"schema": NATIVE_SCHEMA, "stage": "paired", "compute_mode": args.compute_mode,
                "backend": "CPU", "batch": 1, "q": case["tokens"], "history": case["visible_history"],
                "threads": case["threads"], "warmup": args.warmup, "samples": args.samples, "seed": args.seed,
                "implementation_variant": f"ggml_cpu_dense_materialized_paired_{args.compute_mode}_v1",
                "timing": "steady_clock_paired_same_resident_graph_partitions_including_sync"}
    for key, value in expected.items():
        _same(native.get(key), value, "native " + key)
    if getattr(args, "min_warmup_ms", None) is not None:
        _same(native.get("min_warmup_ms"), args.min_warmup_ms, "minimum warmup duration")
        _same(native.get("warmup_rounds_requested"), args.warmup, "requested warmup rounds")
        for key, minimum in (("warmup_rounds_effective", args.warmup),
                             ("actual_warmup_elapsed_ns", args.min_warmup_ms*1000000)):
            value = native.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("actual warmup cannot understate its requested rounds or duration")
        affinity = native.get("cpu_affinity")
        if not isinstance(affinity, dict):
            raise ValueError("CPU affinity must be recorded, including unknown/unbound states")
        _same(affinity.get("requested_logical_ids"), args.cpu_mask, "requested CPU mask")
        if args.cpu_mask is None:
            if affinity.get("applied_logical_ids") is not None or affinity.get("verified") is not None:
                raise ValueError("unspecified affinity must not become an asserted binding")
        else:
            applied = affinity.get("applied_logical_ids")
            if affinity.get("strict_cpu") is not True or affinity.get("verified") is not True or not isinstance(applied, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in applied) or sorted(applied) != sorted(args.cpu_mask):
                raise ValueError("CPU mask must be strictly applied and observed")
            observed = affinity.get("observed_threads")
            if not isinstance(observed, list) or len(observed) != case["threads"]:
                raise ValueError("CPU binding needs observations of every compute thread")
            ids = []
            thread_ids = []
            for thread in observed:
                if not isinstance(thread, dict) or isinstance(thread.get("logical_id"), bool) or not isinstance(thread.get("logical_id"), int) or thread.get("logical_id") not in args.cpu_mask or isinstance(thread.get("group"), bool) or thread.get("group") != 0:
                    raise ValueError("CPU binding observations must match the requested group0 processors")
                try:
                    mask = int(thread["mask_hex"], 16)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("CPU binding requires an observed hexadecimal mask") from error
                if mask != 1 << thread["logical_id"]:
                    raise ValueError("each compute thread must have a single observed processor")
                if isinstance(thread.get("thread_id"), bool) or not isinstance(thread.get("thread_id"), int) or thread["thread_id"] <= 0:
                    raise ValueError("CPU binding needs observed positive OS thread identities")
                ids.append(thread["logical_id"])
                thread_ids.append(thread.get("thread_id"))
            if sorted(ids) != sorted(args.cpu_mask) or len(set(thread_ids)) != case["threads"]:
                raise ValueError("CPU binding must cover distinct threads and processors")
    _same(validate_config(native.get("config")), case["config"], "native config")
    _same(native.get("paired_order"), ["AB" if i % 2 == 0 else "BA" for i in range(args.samples)], "paired order")
    stages = native.get("stage_wall_ns")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ValueError("stage timings must contain exactly three partitions")
    for stage in STAGES:
        samples(stages[stage], args.samples, stage)
    for field in ("stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns"):
        samples(native.get(field), args.samples, field)
    for i, total in enumerate(native["stage_sum_wall_ns"]):
        if total != sum(stages[stage][i] for stage in STAGES):
            raise ValueError("stage sum disagrees with its raw partition samples")
        if native["serial_wall_ns"][i] < total:
            raise ValueError("outer serial clock cannot be smaller than its nested clocks")
    ranges = native.get("stage_node_ranges")
    if not isinstance(ranges, dict) or set(ranges) != set(STAGES):
        raise ValueError("partition node ranges are incomplete")
    previous = 0
    for stage in STAGES:
        part = ranges[stage]
        if not isinstance(part, dict) or part.get("begin") != previous or isinstance(part.get("end"), bool) or not isinstance(part.get("end"), int) or part["end"] <= previous:
            raise ValueError("partition ranges must be contiguous nonempty half-open ranges")
        if not isinstance(part.get("ops"), list) or len(part["ops"]) != part["end"] - previous or any(not isinstance(op, str) for op in part["ops"]):
            raise ValueError("partition operation inventory disagrees with its range")
        previous = part["end"]
    reference = native.get("reference_check")
    if native.get("all_outputs_finite") is not True or not isinstance(reference, dict) or reference.get("passed") is not True:
        raise ValueError("paired graph numerical checks failed")
    validation = native.get("paired_validation")
    if not isinstance(validation, dict) or any(validation.get(key) is not True for key in ("ports_exact", "kv_caches_exact", "input_ports_unchanged", "checked_after_timing")):
        raise ValueError("paired paths require untimed complete port and KV equivalence checks")
    _same(native.get("port_digest_algorithm"), "fnv1a64_native_tensor_bytes", "port digest algorithm")
    inputs, outputs = ports(case["config"], "block", case["tokens"])
    for path in ("serial", "joint"):
        checks = validation.get(path)
        if not isinstance(checks, dict) or checks.get("all_outputs_finite") is not True:
            raise ValueError("both paired paths must check every output")
        kv = checks.get("kv_append_check")
        if not isinstance(kv, dict) or any(kv.get(key) is not True for key in ("passed", "append_exact", "history_unchanged")):
            raise ValueError("both paired paths must validate complete KV content")
        for field, keys in (("input_port_digests", set(inputs)), ("stage_port_digests", set(outputs)),
                            ("kv_cache_digests", {"key_cache", "value_cache"})):
            digests = checks.get(field)
            if not isinstance(digests, dict) or set(digests) != keys or any(not isinstance(value, str) or len(value) != 16 or any(c not in "0123456789abcdef" for c in value) for value in digests.values()):
                raise ValueError(f"{path}.{field} must bind all expected ports with FNV1a64")
    for field in ("input_port_digests", "stage_port_digests", "kv_cache_digests"):
        _same(validation["serial"][field], validation["joint"][field], "paired " + field)
    inventory = resident_accounting(case["config"], "block", case["tokens"], case["visible_history"])
    for key, value in inventory.items():
        _same(native.get(key), value, "native resident " + key)
    memory = native.get("shared_memory_inventory")
    if not isinstance(memory, dict) or memory.get("timed_paths_share_tensor_addresses") is not True:
        raise ValueError("paired graphs must explicitly share their resident/internal tensors")
    _same(memory.get("resident_logical_bytes"), inventory["resident_logical_bytes"], "shared resident inventory")
    for key in ("resident_allocated_bytes", "internal_allocated_buffer_bytes", "backend_shared_cplan_capacity_bytes",
                "cached_plans_total_work_bytes", "physical_tensor_and_work_allocated_bytes"):
        value = memory.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("shared memory capacity must be explicitly nonnegative")
    if memory["resident_allocated_bytes"] < inventory["resident_logical_bytes"]:
        raise ValueError("allocated resident bytes cannot understate the logical inventory")
    work = memory.get("per_graph_cplan_work_size_bytes")
    if not isinstance(work, dict) or set(work) != {*STAGES, "joint"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in work.values()):
        raise ValueError("each graph must declare its CPU cplan scratch")
    _same(memory["backend_shared_cplan_capacity_bytes"], max(work.values()) if args.compute_mode == "graph" else 0, "shared backend work capacity")
    _same(memory["cached_plans_total_work_bytes"], sum(work.values()) if args.compute_mode == "plan" else 0, "cached plans work allocation")
    _same(memory["physical_tensor_and_work_allocated_bytes"], sum(memory[key] for key in
          ("resident_allocated_bytes", "internal_allocated_buffer_bytes", "backend_shared_cplan_capacity_bytes", "cached_plans_total_work_bytes")), "physical shared allocation inventory")
    return native


def summarize_paired(native):
    result = {key: summarize_samples(native[key]) for key in ("stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns")}
    result["stages"] = {key: summarize_samples(values) for key, values in native["stage_wall_ns"].items()}
    bookkeeping = [a-b for a, b in zip(native["serial_wall_ns"], native["stage_sum_wall_ns"])]
    mean = statistics.mean(bookkeeping)
    result["clock_bookkeeping_ns"] = {"samples_ns": bookkeeping, "count": len(bookkeeping),
                                      "median_ns": statistics.median(bookkeeping), "mean_ns": mean,
                                      "min_ns": min(bookkeeping), "max_ns": max(bookkeeping),
                                      "cv": statistics.pstdev(bookkeeping)/mean if mean else None}
    # These paired differences measure two legitimate execution plans, not model accuracy.
    result["paired_serial_minus_joint_ns"] = [a-b for a, b in zip(native["serial_wall_ns"], native["joint_wall_ns"])]
    result["median_stage_sum_minus_joint_ratio"] = (result["stage_sum_wall_ns"]["median_ns"]-result["joint_wall_ns"]["median_ns"])/result["joint_wall_ns"]["median_ns"]
    result["model_accuracy_claimed"] = False
    return result


def paired_quality_flags(summary):
    flags = [key for key in ("stage_sum_wall_ns", "serial_wall_ns", "joint_wall_ns") if summary[key]["cv"] > 0.1]
    return flags + ["stage:"+key for key in STAGES if summary["stages"][key]["cv"] > 0.1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True, help="explicit executable path")
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--compute-mode", choices=("graph", "plan"), required=True)
    parser.add_argument("--min-warmup-ms", type=int, required=True)
    parser.add_argument("--cpu-mask", help="explicit comma-separated logical CPUs, never inferred")
    for name in ("threads", "warmup", "samples", "seed", "repeats"):
        parser.add_argument("--"+name, type=int, required=True)
    parser.add_argument("--split", choices=("development", "held_out"))
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_mask is not None:
        try:
            args.cpu_mask = [int(value) for value in args.cpu_mask.split(",")]
        except ValueError:
            parser.error("cpu-mask must be comma-separated integral logical CPU IDs")
        if len(args.cpu_mask) != args.threads or len(set(args.cpu_mask)) != len(args.cpu_mask) or any(value < 0 or value >= 512 for value in args.cpu_mask):
            parser.error("cpu-mask must explicitly select one distinct CPU for each requested thread")
    if args.min_warmup_ms < 0:
        parser.error("minimum warmup duration must be nonnegative")
    if args.threads < 1 or args.warmup < 0 or not 4 <= args.samples <= 10000 or args.samples % 2 or not 0 <= args.seed <= 0xFFFFFFFF or args.repeats < 1 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid protocol; paired samples require an even count >=4")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    complete = build_plan(suite, args.threads)
    if any(case["kind"] != "block" for case in complete):
        parser.error("paired suite cases must describe block geometry")
    physical_splits = {}
    for case in complete:
        split = case.get("metadata", {}).get("split")
        if split not in ("development", "held_out"):
            parser.error("every shape must predeclare development or held_out membership")
        identity = json.dumps({key: case[key] for key in ("config", "threads", "batch", "tokens", "visible_history")}, sort_keys=True)
        if identity in physical_splits:
            parser.error("duplicate physical geometry cannot be relabeled as a different phase or split")
        physical_splits[identity] = split
    plan = [case for case in complete if args.split is None or case.get("metadata", {}).get("split") == args.split]
    if not plan:
        parser.error("explicit split selects no cases")
    random.Random(args.seed).shuffle(plan)
    binary = args.native_bin.resolve()
    jobs = [(repeat, case) for repeat in range(args.repeats) for case in (plan if repeat % 2 == 0 else list(reversed(plan)))]
    if args.dry_run:
        print(json.dumps({"compute_mode": args.compute_mode, "jobs": [{"repeat": repeat, "case": case, "command": paired_command(case, binary, args)} for repeat, case in jobs]}, indent=2))
        return 0
    if not binary.is_file():
        parser.error("missing paired executable")
    directories = [path.resolve() for path in args.runtime_dir]
    if any(not path.is_dir() for path in directories):
        parser.error("runtime directories must exist")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output directory must be empty")
    lock_path = ROOT / ".measurement.lock"
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(lock, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid())+"\n")
    try:
        (output / "raw").mkdir(parents=True)
        child_env = os.environ.copy()
        child_env["PATH"] = os.pathsep.join([str(path) for path in directories]+[child_env.get("PATH", "")])
        if os.name != "nt":
            child_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(path) for path in directories]+[child_env.get("LD_LIBRARY_PATH", "")])
        child_env.pop("LLAMA_VIZ_TIMING", None)
        manifest = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "suite": suite, "suite_sha256": sha256(args.suite), "split": args.split,
                    "protocol": {key: getattr(args, key) for key in ("threads", "warmup", "samples", "seed", "repeats", "compute_mode", "min_warmup_ms", "cpu_mask")},
                    "planned_jobs": len(jobs), "case_order": [{"repeat": repeat, "case_id": case["id"]} for repeat, case in jobs],
                    "runtime_sha256": {str(binary): sha256(binary)}, "environment": environment(),
                    "benchmark_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT/"native/dense_layer_probe.cpp"]},
                    "model_contract_sha256": sha256(MODELING/"llm_infer_model/m2/dense_contract.py"),
                    "scope": "same-resident graph-partition diagnostic; not independent-resident M2 costs"}
        for directory in directories:
            for path in directory.glob("*ggml*"):
                if path.suffix in (".dll", ".so", ".dylib") or ".so." in path.name:
                    manifest["runtime_sha256"][str(path)] = sha256(path)
        dump(output/"manifest.json", manifest)
        rows = []
        for i, (repeat, case) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {case['id']} repeat={repeat} mode={args.compute_mode}", flush=True)
            command = paired_command(case, binary, args)
            raw = {"command": command, "case": case, "repeat": repeat}
            row = {"case_id": case["id"], "case": case, "repeat": repeat, "compute_mode": args.compute_mode,
                   "raw_output": f"raw/{i:03d}.json", "native_binary_sha256": manifest["runtime_sha256"][str(binary)]}
            try:
                process = subprocess.run(command, capture_output=True, text=True, errors="replace", env=child_env, timeout=args.timeout_s)
                raw.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
                if process.returncode:
                    raise ValueError(f"paired executable returned {process.returncode}: {process.stderr[-1000:]}")
                native = validate_paired_binding(case, json.loads(process.stdout), args)
                summary = summarize_paired(native)
                flags = paired_quality_flags(summary)
                row.update(status="measured", native=native, summary=summary, quality_flags=flags)
            except (OSError, ValueError, SpecError, subprocess.TimeoutExpired) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    raw.update(timed_out=True, stdout=error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout,
                               stderr=error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr)
                raw["error"] = str(error)
                row.update(status="failed", error=str(error))
                print(str(error), file=sys.stderr)
            dump(output/row["raw_output"], raw)
            rows.append(row)
            dump(output/"measurements.json", rows)
        manifest.update(completed_jobs=len(rows), measured_jobs=sum(row["status"] == "measured" for row in rows), finished_at_utc=datetime.now(timezone.utc).isoformat())
        dump(output/"manifest.json", manifest)
        return 0 if manifest["measured_jobs"] == len(jobs) else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SpecError) as error:
        print(f"run_dense_paired_measurements: {error}", file=sys.stderr)
        raise SystemExit(1)
