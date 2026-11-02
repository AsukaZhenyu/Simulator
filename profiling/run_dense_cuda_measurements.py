"""Measure complete declared Dense decoder stages on one CUDA device, batch 1.

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
from llm_infer_model.m2.dense_cuda_contract import (
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
                       case["visible_history"], case["cuda_device"], batch=case["batch"], cuda_graphs=case["cuda_graphs"])
    return {"physical_signature": result, "physical_contract_sha256": sha_signature(result),
            "native_binary_sha256": binary_hash}


def sha_signature(value: dict) -> str:
    import hashlib
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_memory_accounting(native: dict) -> None:
    """Known tensor capacities are not the opaque total CUDA workspace."""
    accounting = native.get("memory_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("CUDA memory accounting must explicitly distinguish known and unknown bytes")
    for container in (native, accounting):
        if "workspace_bytes" not in container or container["workspace_bytes"] is not None:
            raise ValueError("unobserved CUDA total workspace must remain explicit null")
    _same(native.get("workspace_source"), "unavailable_cuda_backend_private_scratch_and_runtime_overheads", "workspace source")
    _same(accounting.get("workspace_bound"), "unknown", "workspace bound")
    _same(accounting.get("workspace_scope"), "known_cuda_gallocr_capacity_excludes_private_backend_scratch_and_runtime_overheads", "workspace scope")
    for key in ("backend_scratch_bytes", "runtime_overhead_bytes"):
        if key not in accounting or accounting[key] is not None:
            raise ValueError(f"{key} must remain explicit unknown, not zero")
    for key in ("resident_logical_bytes", "resident_allocated_bytes", "internal_allocated_buffer_bytes", "internal_logical_tensor_bytes_sum"):
        _integer(accounting.get(key), "memory accounting " + key)
    _same(_integer(accounting.get("known_tensor_allocation_bytes"), "known tensor allocation"),
          accounting["resident_allocated_bytes"] + accounting["internal_allocated_buffer_bytes"],
          "known tensor allocation inventory")
    _same(_integer(native.get("internal_tensor_bytes_upper_bound"), "internal tensor component upper bound"),
          max(accounting["internal_allocated_buffer_bytes"], accounting["internal_logical_tensor_bytes_sum"]),
          "internal tensor component upper bound")
    if accounting["resident_allocated_bytes"] < accounting["resident_logical_bytes"]:
        raise ValueError("allocated resident bytes cannot be smaller than logical resident bytes")


def validate_device_execution(case: dict, native: dict) -> None:
    backend = f"CUDA{case['cuda_device']}"
    checks = native.get("backend_execution_check")
    if not isinstance(checks, dict):
        raise ValueError("CUDA execution must include no-fallback and device-buffer evidence")
    for key, expected in {"all_nodes_supported": True, "all_tensors_on_requested_cuda_buffer": True, "cpu_fallback_used": False}.items():
        _same(checks.get(key), expected, "backend execution " + key)
    nodes = native.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("CUDA node inventory must be nonempty")
    _same(_integer(checks.get("node_count"), "node_count", 1), len(nodes), "checked node count")
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("node inventory entries must be objects")
        _same(node.get("supported_by_requested_backend"), True, "CUDA node support")
        _same(node.get("buffer_type"), backend, "CUDA node buffer")
    device = native.get("device_info")
    if not isinstance(device, dict):
        raise ValueError("CUDA device identity must be queried, independently of NVML")
    _same(_integer(device.get("index"), "device index"), case["cuda_device"], "actual CUDA device")
    for key in ("name", "uuid"):
        if not isinstance(device.get(key), str) or not device[key]:
            raise ValueError(f"CUDA device {key} must be observed")
    for key in ("total_memory_bytes", "compute_capability_major"):
        _integer(device.get(key), "device " + key, 1)
    _integer(device.get("compute_capability_minor"), "device compute_capability_minor")
    for key in ("threads", "threads_requested", "threads_applied", "cpu_cplan_work_size_bytes"):
        if key in native:
            raise ValueError("CUDA native evidence must not carry a CPU thread/scratch contract")


def native_command(case: dict, binary_dir: Path, args: argparse.Namespace) -> list[str]:
    suffix = ".exe" if os.name == "nt" else ""
    return [str(binary_dir / ("ggml-dense-layer-cuda-probe" + suffix)),
            *native_arguments(case, args.warmup, args.samples, args.seed)]


def validate_native_binding(case: dict, native: dict, args: argparse.Namespace) -> None:
    """Bind native geometry, arithmetic, materialized ports and timing protocol."""
    if not isinstance(native, dict):
        raise ValueError("native output must be one JSON object")
    expected = {"schema": "ggml_dense_layer_cuda_probe/1", "kind": "dense_layer",
        "contract": CONTRACT, "stage": case["kind"], "backend": f"CUDA{case["cuda_device"]}", "batch": 1,
        "cuda_device": case["cuda_device"], "cuda_graphs": case["cuda_graphs"],
        "cuda_graphs_requested": "disabled", "cuda_graphs_applied": "disabled",
        "cuda_graph_disable_environment": "1",
        "matmul_precision": "ggml_cuda_backend_default",
        "config": case["config"], "q": case["tokens"], "history": case["visible_history"],
        "activation_dtype": "f32", "kv_dtype": "f32",
        "implementation_variant": VARIANTS[case["kind"]],
        "included": operations(case["config"], case["kind"]), "excluded": list(EXCLUDED),
        "timing": "steady_clock_cuda_graph_compute_including_sync",
        "scope": "resident_repeated_complete_declared_decoder_block",
        "setup_compute_repetitions": 1,
        "upstream_setup_compute_repetitions": 0 if case["kind"] == "block" else 1,
        "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
        "position_dtype": "i32",
        "warmup": args.warmup, "samples": args.samples, "seed": args.seed}
    for key, value in expected.items():
        actual = validate_config(native.get(key)) if key == "config" else native.get(key)
        _same(actual, value, f"native {key}")
    validate_memory_accounting(native)
    validate_device_execution(case, native)
    reference = native.get("reference_check")
    if native.get("all_outputs_finite") is not True or not isinstance(reference, dict) or reference.get("passed") is not True:
        raise ValueError("native Dense stage did not pass its numerical checks")
    if case["kind"] in ("attention", "block"):
        append = native.get("kv_append_check")
        if not isinstance(append, dict) or any(append.get(field) is not True for field in ("passed", "append_exact", "history_unchanged")):
            raise ValueError("attention/block requires successful KV append and history-preservation checks")
    validate_reference_ports(case, native)
    expected_memory = resident_accounting(case["config"], case["kind"], case["tokens"], case["visible_history"])
    for key, value in expected_memory.items():
        _same(native.get(key), value, f"native resident {key}")
    accounting = native.get("memory_accounting")
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


def _nonnegative_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be nonnegative finite numeric evidence")
    return value


def validate_reference_ports(case: dict, native: dict) -> None:
    """Verify declared per-port numerical evidence, including coverage and gates."""
    _, declared = ports(case["config"], case["kind"], case["tokens"])
    reference_names = {"q": "query", "k": "new_key", "v": "new_value", "gate": "attention_gate", "hidden_out": "hidden_output"}
    outputs = {reference_names.get(name, name): port for name, port in declared.items()}
    checks = native.get("port_reference_checks")
    if not isinstance(checks, dict) or set(checks) != set(outputs):
        raise ValueError("port_reference_checks must include exactly the declared stage output ports")
    quantized = any(dtype in ("q4_k", "q6_k") for dtype in case["config"]["weight_dtypes"].values())
    atol, rtol = (0.003, 0.035) if quantized else (0.0002, 0.002)
    totals = {"samples": 0, "elements": 0, "squared": 0.0, "reference_squared": 0.0,
              "max_abs_error": 0.0, "max_relative_error": 0.0}

    def check_one(check, elements, samples, label):
        if not isinstance(check, dict) or check.get("passed") is not True:
            raise ValueError(f"{label} did not pass its numerical check")
        for key, expected in {"output_elements": elements, "samples": samples,
            "coverage": "all_output_elements" if samples == elements else "deterministic_output_element_sample",
            "reference": "independent_double_declared_block_with_dequantized_stored_weights_and_f32_operator_boundaries",
            "criterion": "each_port_normalized_rmse_le_rtol_or_rmse_le_atol_and_exact_kv_validation",
            "atol": atol, "rtol": rtol}.items():
            _same(check.get(key), expected, label + " " + key)
        for key in ("max_abs_error", "max_relative_error", "rmse", "reference_rms", "normalized_rmse"):
            _nonnegative_finite(check.get(key), label + " " + key)
        expected_nrmse = check["rmse"] / max(check["reference_rms"], 1e-12)
        if not math.isclose(check["normalized_rmse"], expected_nrmse, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError(f"{label} normalized RMSE is inconsistent")
        return check

    for name, port in outputs.items():
        elements = math.prod(port["ne"])
        nt = min(3, case["tokens"])
        width = elements // case["tokens"]
        samples = elements if elements <= 512 else nt * min(512 // nt, width)
        check = check_one(checks[name], elements, samples, "port reference " + name)
        if not (check["normalized_rmse"] <= rtol or check["rmse"] <= atol):
            raise ValueError(f"port reference {name} fails its declared numerical tolerance")
        totals["samples"] += samples
        totals["elements"] += elements
        totals["squared"] += samples * check["rmse"] ** 2
        totals["reference_squared"] += samples * check["reference_rms"] ** 2
        for key in ("max_abs_error", "max_relative_error"):
            totals[key] = max(totals[key], check[key])
    aggregate = check_one(native.get("reference_check"), totals["elements"], totals["samples"], "aggregate reference")
    expected_metrics = {"rmse": math.sqrt(totals["squared"] / totals["samples"]),
                        "reference_rms": math.sqrt(totals["reference_squared"] / totals["samples"]),
                        **{key: totals[key] for key in ("max_abs_error", "max_relative_error")}}
    for key, expected in expected_metrics.items():
        if not math.isclose(aggregate[key], expected, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError("aggregate reference disagrees with its per-port " + key)


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
    parser.add_argument("--suite", type=Path, default=ROOT / "cases" / "dense_layer_cuda_smoke.json")
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--cuda-graphs", choices=("disabled",), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend-revision", help="Declared source revision; actual executable/library hashes recorded separately")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.cuda_device < 0 or args.warmup < 0 or not 3 <= args.samples <= 10000 or not 0 <= args.seed <= 0xFFFFFFFF or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid CUDA device, warmup, samples, seed or timeout")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    plan = build_plan(suite, args.cuda_device, args.cuda_graphs)
    random.Random(args.seed).shuffle(plan)
    binary_dir = args.native_bin.resolve()
    commands = [native_command(case, binary_dir, args) for case in plan]
    if args.dry_run:
        print(json.dumps({"contract": CONTRACT, "supported_backend": f"CUDA{args.cuda_device}", "supported_batch": 1,
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
        child_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        binary = Path(commands[0][0])
        binary_hash = sha256(binary)
        source_paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "native").glob("*.cpp")) + [ROOT / "native" / "CMakeLists.txt"]
        runtime_files = {binary}
        for folder in [binary_dir, *runtime_dirs]:
            runtime_files.update(path for path in folder.glob("*ggml*") if path.suffix in {".dll", ".so", ".dylib"} or ".so." in path.name)
        manifest = {"schema_version": "dense_cuda_measurement_profile_v1", "contract": CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "profile_id": output.name,
            "environment": environment(), "suite": suite, "suite_sha256": sha256(args.suite),
            "case_order": [case["id"] for case in plan],
            "protocol": {"backend": f"CUDA{args.cuda_device}", "batch": 1,
                "cuda_device": args.cuda_device, "cuda_graphs": args.cuda_graphs,
                "warmup": args.warmup, "samples": args.samples, "seed": args.seed,
                "sequential_execution": True, "graphs_per_sample": 1,
                "setup_compute_repetitions": 1, "measured_graph_untimed_compute_repetitions": 1 + args.warmup,
                "upstream_setup_compute_repetitions_by_kind": {kind: 0 if kind == "block" else 1 for kind in ("projection", "attention", "dense_ffn", "block")},
                "total_untimed_compute_repetitions_by_kind": {kind: args.warmup + (1 if kind == "block" else 2) for kind in ("projection", "attention", "dense_ffn", "block")},
                "timing": "one completed resident CUDA Dense stage graph; setup and explicit external copies excluded",
                "inputs": "deterministic generated tensors; not pretrained model weight contents",
                "layer_coverage": "complete declared pre-norm Dense decoder block; not original Qwen kernel"},
            "software": {"backend_revision_declared": args.backend_revision,
                "benchmark_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths if path.is_file()},
                "model_contract_source_sha256": {name: sha256(MODELING / "llm_infer_model" / "m2" / name) for name in ("dense_contract.py", "dense_cuda_contract.py")},
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
                    "complete_declared_block_variant": True, "M2_memory_accounting_alignment_verified": False, "capacity_verified": False,
                    "CPU_fallback_permitted": False}
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
        print(f"Measured {manifest['measured_cases']}/{len(plan)} CUDA Dense stage cases: {output}")
        return 0 if len(rows) == len(plan) and all(row["status"] == "measured" for row in rows) else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, SpecError) as exc:
        print(f"run_dense_cuda_measurements: {exc}", file=sys.stderr)
        raise SystemExit(1)
