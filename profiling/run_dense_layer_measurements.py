"""Measure explicitly configured complete declared Dense decoder stages on CPU, batch 1.

This runner is independent of the isolated matmul runner and does not produce
M2 cost tables. The shared lock serializes all profiling hardware execution.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

MODELING = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))
from llm_infer_model.m2.dense_contract import (
    CONTRACT, build_plan, validate_case, validate_config, signature, native_arguments, resident_accounting,
    VARIANTS, operations, EXCLUDED, ports
)
from llm_infer_model.tensor.spec import SpecError
from profile_contract import summarize_samples, validate_measurement
from run_measurements import dump, environment, sha256

ROOT = Path(__file__).resolve().parent


def _same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{label} does not match the exact requested contract")


def measurement_signature(case: dict, binary_hash: str | None = None) -> dict:
    result = signature(case["config"], case["kind"], case["phase"], case["tokens"],
                       case["visible_history"], case["threads"], batch=case["batch"])
    return {"physical_signature": result, "physical_contract_sha256": sha_signature(result),
            "native_binary_sha256": binary_hash}


def sha_signature(value: dict) -> str:
    import hashlib
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_memory_accounting(native: dict) -> None:
    """Known workspace must match its explicit allocator/scratch accounting."""
    workspace = native.get("workspace_bytes")
    if workspace is None:
        return  # Keeps analysis valid; does not certify M2 capacity feasibility.
    accounting = native.get("memory_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("numeric macro workspace requires a memory_accounting object")
    if native.get("workspace_source") != "ggml_gallocr_internal_buffer_plus_cpu_graph_plan":
        raise ValueError("numeric macro workspace source does not match allocator-plus-scratch evidence")
    for container, keys in ((native, ("workspace_bytes", "cpu_cplan_work_size_bytes")),
                            (accounting, ("workspace_bytes", "internal_allocated_buffer_bytes", "cpu_cplan_work_size_bytes"))):
        for key in keys:
            value = container.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"memory accounting {key} must be a nonnegative integer")
    if workspace != accounting["workspace_bytes"] or native["cpu_cplan_work_size_bytes"] != accounting["cpu_cplan_work_size_bytes"]:
        raise ValueError("top-level macro workspace/scratch disagrees with memory accounting")
    if workspace != accounting["internal_allocated_buffer_bytes"] + accounting["cpu_cplan_work_size_bytes"]:
        raise ValueError("macro workspace must equal internal allocated capacity plus CPU scratch")
    if accounting.get("workspace_bound") != "conservative_allocated_capacity":
        raise ValueError("macro workspace must describe its conservative allocated-capacity scope")
    if accounting.get("workspace_scope") != "internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads":
        raise ValueError("macro workspace scope does not match this probe contract")


def native_command(case: dict, binary_dir: Path, args: argparse.Namespace) -> list[str]:
    suffix = ".exe" if os.name == "nt" else ""
    return [str(binary_dir / ("ggml-dense-layer-probe" + suffix)),
            *native_arguments(case, args.warmup, args.samples, args.seed)]


def validate_native_binding(case: dict, native: dict, args: argparse.Namespace) -> None:
    """Bind native geometry, arithmetic, materialized ports and timing protocol."""
    if not isinstance(native, dict):
        raise ValueError("native output must be one JSON object")
    expected = {"schema": "ggml_dense_layer_probe/1", "kind": "dense_layer",
        "contract": CONTRACT, "stage": case["kind"], "backend": "CPU", "batch": 1,
        "config": case["config"], "q": case["tokens"], "history": case["visible_history"],
        "activation_dtype": "f32", "kv_dtype": "f32",
        "implementation_variant": VARIANTS[case["kind"]],
        "included": operations(case["config"], case["kind"]), "excluded": list(EXCLUDED),
        "timing": "steady_clock_graph_compute_including_sync",
        "scope": "resident_repeated_complete_declared_decoder_block",
        "setup_compute_repetitions": 1,
        "upstream_setup_compute_repetitions": 0 if case["kind"] == "block" else 1,
        "thread_management": "persistent_ggml_threadpool",
        "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
        "position_dtype": "i32", "threads": case["threads"],
        "threads_requested": case["threads"], "threads_applied": case["threads"],
        "warmup": args.warmup, "samples": args.samples, "seed": args.seed}
    for key, value in expected.items():
        actual = validate_config(native.get(key)) if key == "config" else native.get(key)
        _same(actual, value, f"native {key}")
    validate_memory_accounting(native)
    reference = native.get("reference_check")
    if native.get("all_outputs_finite") is not True or not isinstance(reference, dict) or reference.get("passed") is not True:
        raise ValueError("native Dense stage did not pass its numerical checks")
    if case["kind"] in ("attention", "block"):
        append = native.get("kv_append_check")
        if not isinstance(append, dict) or any(append.get(field) is not True for field in ("passed", "append_exact", "history_unchanged")):
            raise ValueError("attention/block requires successful KV append and history-preservation checks")
    expected_memory = resident_accounting(case["config"], case["kind"], case["tokens"], case["visible_history"])
    for key, value in expected_memory.items():
        _same(native.get(key), value, f"native resident {key}")
    accounting = native.get("memory_accounting")
    if native.get("workspace_bytes") is not None:
        resident = sum(expected_memory[key] for key in ("weight_bytes", "input_bytes", "output_bytes", "kv_cache_bytes"))
        _same(accounting.get("resident_logical_bytes"), resident, "resident logical inventory")
        allocated = accounting.get("resident_allocated_bytes")
        if isinstance(allocated, bool) or not isinstance(allocated, int) or allocated < resident:
            raise ValueError("resident allocated bytes cannot be smaller than logical resident bytes")
    _same(native.get("port_digest_algorithm"), "fnv1a64_native_tensor_bytes", "native stage digest algorithm")
    inputs, outputs = ports(case["config"], case["kind"], case["tokens"])
    expected_port_accounting = {}
    for category, declared in (("input", inputs), ("output", outputs)):
        for name, port in declared.items():
            expected_port_accounting[name] = {"dtype": port["dtype"], "shape": list(port["ne"][:3]), "ne": list(port["ne"]),
                "bytes": math.prod(port["ne"])*4, "category": category}
    if case["kind"] in ("attention", "block"):
        config = case["config"]
        for name in ("key_cache", "value_cache"):
            expected_port_accounting[name] = {"dtype": "f32", "shape": [config["head_dim"],
                case["visible_history"]+case["tokens"], config["kv_heads"]],
                "bytes": expected_memory["kv_cache_bytes"]//2, "category": "kv_cache",
                "ne": [config["head_dim"], case["visible_history"]+case["tokens"], config["kv_heads"], 1]}
    _same(native.get("port_accounting"), expected_port_accounting, "native port shape/dtype/category accounting")

    for field, expected_ports in (("input_port_digests", inputs), ("stage_port_digests", outputs)):
        digests = native.get(field)
        if not isinstance(digests, dict) or set(digests) != set(expected_ports):
            raise ValueError(f"native {field} must bind exactly the declared materialized ports")
        for port, digest in digests.items():
            if not isinstance(digest, str) or len(digest) != 16 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"native {field}.{port} must be a lowercase 16-character FNV1a64 digest")


def write_summary(output: Path, rows: list[dict]) -> None:
    fields = ["case_id", "kind", "phase", "tokens", "visible_history", "status", "median_ns", "cv", "workspace_bytes", "raw_output"]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"case_id": row["case_id"],
                **{key: row["case"].get(key, "") for key in ("kind", "phase", "tokens", "visible_history")},
                "status": row["status"], "median_ns": row.get("summary", {}).get("median_ns", ""),
                "cv": row.get("summary", {}).get("cv", ""),
                "workspace_bytes": row.get("native", {}).get("workspace_bytes"), "raw_output": row["raw_output"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "cases" / "dense_layer_cpu_smoke.json")
    parser.add_argument("--native-bin", type=Path, default=ROOT / "native" / "build-cpu" / ("Release" if os.name == "nt" else ""))
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=1, help="CPU threads, recorded as requested and applied")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend-revision", help="Declared source revision; actual executable/library hashes recorded separately")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warmup < 0 or not 3 <= args.samples <= 10000 or not 0 <= args.seed <= 0xFFFFFFFF or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid threads, warmup, samples, seed or timeout")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    plan = build_plan(suite, args.threads)
    random.Random(args.seed).shuffle(plan)
    binary_dir = args.native_bin.resolve()
    commands = [native_command(case, binary_dir, args) for case in plan]
    if args.dry_run:
        print(json.dumps({"contract": CONTRACT, "supported_backend": "CPU", "supported_batch": 1,
            "suite": str(args.suite.resolve()), "cases": plan, "commands": commands}, ensure_ascii=False, indent=2))
        return 0
    for command in commands:
        if not Path(command[0]).is_file():
            parser.error(f"missing Dense decoder probe: {command[0]}; build profiling/native first")
    runtime_dirs = [folder.resolve() for folder in args.runtime_dir]
    for folder in runtime_dirs:
        if not folder.is_dir():
            parser.error(f"runtime directory does not exist: {folder}")
    lock_path = ROOT / ".measurement.lock"
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"another measurement run holds {lock_path}; verify its PID before removing a stale lock")
    with os.fdopen(lock, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        output = (args.output or ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S-dense-layer")).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be empty: {output}")
        (output / "raw").mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env["PATH"] = os.pathsep.join([str(folder) for folder in runtime_dirs] + [child_env.get("PATH", "")])
        if os.name != "nt":
            child_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(folder) for folder in runtime_dirs] + [child_env.get("LD_LIBRARY_PATH", "")])
        child_env.pop("LLAMA_VIZ_TIMING", None)
        binary = Path(commands[0][0])
        binary_hash = sha256(binary)
        source_paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "native").glob("*.cpp")) + [ROOT / "native" / "CMakeLists.txt"]
        runtime_files = {binary}
        for folder in [binary_dir, *runtime_dirs]:
            runtime_files.update(path for path in folder.glob("*ggml*") if path.suffix in {".dll", ".so", ".dylib"} or ".so." in path.name)
        manifest = {"schema_version": "dense_layer_measurement_profile_v1", "contract": CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "profile_id": output.name,
            "environment": environment(), "suite": suite, "suite_sha256": sha256(args.suite),
            "case_order": [case["id"] for case in plan],
            "protocol": {"backend": "CPU", "batch": 1, "threads": args.threads,
                "warmup": args.warmup, "samples": args.samples, "seed": args.seed,
                "sequential_execution": True, "graphs_per_sample": 1,
                "setup_compute_repetitions": 1, "measured_graph_untimed_compute_repetitions": 1 + args.warmup,
                "upstream_setup_compute_repetitions_by_kind": {kind: 0 if kind == "block" else 1 for kind in ("projection", "attention", "dense_ffn", "block")},
                "total_untimed_compute_repetitions_by_kind": {kind: args.warmup + (1 if kind == "block" else 2) for kind in ("projection", "attention", "dense_ffn", "block")},
                "timing": "one completed resident CPU Dense stage graph; setup and explicit external copies excluded",
                "inputs": "deterministic generated tensors; not pretrained model weight contents",
                "layer_coverage": "complete declared pre-norm Dense decoder block; not original Qwen kernel"},
            "software": {"backend_revision_declared": args.backend_revision,
                "benchmark_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths if path.is_file()},
                "model_contract_source_sha256": {"m2/dense_contract.py": sha256(MODELING / "llm_infer_model" / "m2" / "dense_contract.py")},
                "runtime_sha256": {str(path): sha256(path) for path in sorted(runtime_files) if path.is_file()},
                "runtime_search_dirs": [str(folder) for folder in runtime_dirs],
                "runtime_hash_scope": "native executable and supplied GGML libraries; actual resolved loader paths not observed",
                "child_flags": {key: child_env.get(key) for key in ("OMP_NUM_THREADS", "GGML_CUDA_DISABLE_GRAPHS", "CUDA_VISIBLE_DEVICES")}}}
        dump(output / "manifest.json", manifest)
        dump(output / "plan.json", plan)
        rows = []
        for index, (case, command) in enumerate(zip(plan, commands), 1):
            print(f"[{index}/{len(plan)}] {case['id']}", flush=True)
            raw_name = f"raw/{index:03d}.json"
            raw = {"case": case, "command": command}
            row = {"case_id": case["id"], "case": case,
                "signature": measurement_signature(case, binary_hash), "raw_output": raw_name}
            try:
                result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=child_env, timeout=args.timeout_s)
                raw.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode != 0:
                    raise RuntimeError(f"Dense decoder probe exited {result.returncode}: {result.stderr[-1000:]}")
                native = json.loads(result.stdout)
                validate_native_binding(case, native, args)
                native["expected_samples"] = args.samples
                native = validate_measurement(native)
                row.update(status="measured", native=native, summary=summarize_samples(native["wall_ns"]))
                row["quality_flags"] = ["high_observed_variability_cv_gt_0.1"] if row["summary"]["cv"] > 0.1 else []
                row["eligibility"] = {"isolated_latency": True, "concurrent_latency": False,
                    "workspace_known": native["workspace_bytes"] is not None,
                    "exact_signature_required": True, "automatic_M2_export": False,
                    "complete_declared_block_variant": True, "M2_memory_accounting_alignment_verified": False}
            except (OSError, ValueError, RuntimeError, SpecError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    for key, captured in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                        raw[key] = captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured
                    raw["timed_out"] = True
                raw["error"] = str(exc)
                row.update(status="failed", error=str(exc))
            dump(output / raw_name, raw)
            rows.append(row)
            dump(output / "measurements.json", rows)
            write_summary(output, rows)
            if row["status"] == "failed":
                print(row["error"], file=sys.stderr)
                if not args.keep_going:
                    break
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), completed_cases=len(rows),
            measured_cases=sum(row["status"] == "measured" for row in rows), planned_cases=len(plan),
            environment_after=environment())
        dump(output / "manifest.json", manifest)
        print(f"Measured {manifest['measured_cases']}/{len(plan)} CPU Dense stage cases: {output}")
        return 0 if len(rows) == len(plan) and all(row["status"] == "measured" for row in rows) else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, SpecError) as exc:
        print(f"run_dense_layer_measurements: {exc}", file=sys.stderr)
        raise SystemExit(1)
