"""Run configurable, sequential GGML/CUDA microbenchmarks using only stdlib."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from profile_contract import summarize_samples, validate_case, validate_measurement

ROOT = Path(__file__).resolve().parent


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str], timeout: float = 10) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def gpu_snapshot() -> dict:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"status": "unavailable"}
    query = "index,uuid,pci.bus_id,name,memory.total,memory.used,driver_version,pstate,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem"
    return capture([binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"])


def environment() -> dict:
    cpu = platform.processor() or None
    ram = None
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.total_phys
        except (OSError, AttributeError):
            pass
    else:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            if Path("/proc/cpuinfo").exists():
                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except (OSError, ValueError):
            pass
    return {"os": platform.platform(), "architecture": platform.machine(), "cpu_model": cpu,
            "logical_cpus": os.cpu_count(), "ram_bytes": ram, "python": sys.version,
            "gpu_snapshot": gpu_snapshot(), "power_mode": "not_modified_or_verified",
            "clock_policy": "not_modified_or_locked"}


def build_plan(suite: dict, backends: list[str], kinds: set[str], threads: int) -> list[dict]:
    if suite.get("schema_version") != "measurement_suite_v1":
        raise ValueError("suite schema_version must be measurement_suite_v1")
    cases = []
    if "compute" in kinds:
        for original in suite.get("compute_cases", []):
            for backend in backends:
                case = dict(original, kind="compute", backend=backend, threads=threads)
                case["id"] = f"{original['id']}__{backend.lower()}"
                cases.append(validate_case(case))
    if "transfer" in kinds:
        for original in suite.get("transfer_cases", []):
            cases.append(validate_case(dict(original, kind="transfer")))
    if not cases:
        raise ValueError("the selected suite/kinds contain no cases")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique after backend expansion")
    return cases


def native_command(case: dict, binary_dir: Path, args: argparse.Namespace) -> list[str]:
    suffix = ".exe" if os.name == "nt" else ""
    if case["kind"] == "compute":
        return [str(binary_dir / ("ggml-compute-probe" + suffix)), "--backend", case["backend"],
                "--m", str(case["m"]), "--k", str(case["k"]), "--n", str(case["n"]),
                "--dtype", case["dtype"], "--threads", str(case["threads"]), "--seed", str(args.seed),
                "--warmup", str(args.warmup), "--samples", str(args.samples)]
    return [str(binary_dir / ("cuda-transfer-probe" + suffix)), "--bytes", str(case["size_bytes"]),
            "--direction", case["direction"].upper(), "--host-memory", case["host_memory"],
            "--device", str(args.device), "--warmup", str(args.warmup), "--samples", str(args.samples)]


def signature(case: dict) -> dict:
    if case["kind"] == "compute":
        return {"op": "MUL_MAT", "scope": "operator", "weight_ne": [case["k"], case["m"]],
                "input_ne": [case["k"], case["n"]], "output_ne": [case["m"], case["n"]],
                "weight_dtype": case["dtype"], "activation_dtype": "f32", "output_dtype": "f32",
                "total_tokens": case["n"], "backend": case["backend"],
                "threads": case["threads"] if case["backend"] == "CPU" else None,
                "threads_requested": case["threads"],
                "threads_applied": case["threads"] if case["backend"] == "CPU" else None}
    return {"scope": "copy", "direction": case["direction"].upper(), "bytes": case["size_bytes"],
            "host_memory": case["host_memory"], "copy_api": "cudaMemcpyAsync"}


def validate_native_binding(case: dict, native: dict, args: argparse.Namespace) -> None:
    """Reject valid-looking output from the wrong benchmark configuration."""
    if not isinstance(native, dict):
        raise ValueError("native output must be one JSON object")
    if "kind" in native and native["kind"] != case["kind"]:
        raise ValueError("native kind does not match the requested case")
    for key, expected in (("warmup", args.warmup), ("samples", args.samples)):
        if native.get(key) != expected:
            raise ValueError(f"native {key}={native.get(key)!r}, requested {expected!r}")
    keys = ("backend", "dtype", "m", "k", "n") if case["kind"] == "compute" else ("direction", "host_memory", "size_bytes")
    for key in keys:
        actual = native.get(key)
        expected = case[key]
        if key == "direction" and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected:
            raise ValueError(f"native {key}={actual!r} does not match requested {expected!r}")
    if case["kind"] == "compute":
        if native.get("seed") != args.seed:
            raise ValueError("native generation seed does not match the case")
        if case["backend"] == "CPU" and native.get("threads") != case["threads"]:
            raise ValueError("native CPU thread count does not match the case")
        # Historical probe/1 records lack threads_requested. New records state
        # requested and applied separately; CUDA does not apply CPU pool threads.
        if "threads_requested" in native:
            if native["threads_requested"] != case["threads"] or native.get("threads") != case["threads"]:
                raise ValueError("native requested thread count does not match the case")
            applied = case["threads"] if case["backend"] == "CPU" else None
            if "threads_applied" not in native or native["threads_applied"] != applied:
                raise ValueError("native applied thread count does not match the backend")
        reference = native.get("reference_check")
        if native.get("all_outputs_finite") is not True or not isinstance(reference, dict) or reference.get("passed") is not True:
            raise ValueError("native compute output did not pass its numerical checks")
    elif native.get("device_index") != args.device or native.get("validation") != "all_bytes_exact":
        raise ValueError("native transfer device or byte validation does not match the case")


def write_summary(output: Path, rows: list[dict]) -> None:
    fields = ["case_id", "kind", "backend_or_route", "dtype", "m", "k", "n", "size_bytes", "host_memory",
              "status", "samples", "median_ns", "p10_ns", "p90_ns", "cv", "workspace_bytes", "raw_output"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            case = row["case"]
            stats = row.get("summary", {})
            native = row.get("native", {})
            writer.writerow({"case_id": case["id"], "kind": case["kind"],
                "backend_or_route": case.get("backend", case.get("direction", "")),
                **{key: case.get(key, "") for key in ("dtype", "m", "k", "n", "size_bytes", "host_memory")},
                "status": row["status"], "samples": stats.get("count", ""),
                **{key: stats.get(key, "") for key in ("median_ns", "p10_ns", "p90_ns", "cv")},
                "workspace_bytes": native.get("workspace_bytes"), "raw_output": row["raw_output"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "cases" / "dense_smoke.json")
    parser.add_argument("--native-bin", type=Path, default=ROOT / "native" / "build" / ("Release" if os.name == "nt" else ""))
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[], help="Directory containing GGML shared libraries; repeatable")
    parser.add_argument("--backends", default="CPU", help="Comma separated CPU,CUDA0; applies to compute cases")
    parser.add_argument("--kinds", default="compute", help="compute,transfer or both")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--device", type=int, default=0, help="CUDA device for transfer cases; compute currently uses CUDA0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend-revision", help="Optional declared source revision; binary hashes are recorded independently")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without running hardware")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed case")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 3 or args.samples > 10000 or args.threads < 1 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid warmup, samples, threads or timeout")
    kinds = {item.strip().lower() for item in args.kinds.split(",")}
    if not kinds or not kinds <= {"compute", "transfer"}:
        parser.error("--kinds must select compute and/or transfer")
    backends = [item.strip().upper() for item in args.backends.split(",")]
    if not backends or any(item not in {"CPU", "CUDA0"} for item in backends) or len(set(backends)) != len(backends):
        parser.error("--backends must contain unique CPU and/or CUDA0")
    if args.device != 0 and "compute" in kinds and "CUDA0" in backends:
        parser.error("compute currently targets CUDA0; --device only changes transfer device")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    plan = build_plan(suite, backends, kinds, args.threads)
    random.Random(args.seed).shuffle(plan)
    binary_dir = args.native_bin.resolve()
    commands = [native_command(case, binary_dir, args) for case in plan]
    if args.dry_run:
        print(json.dumps({"suite": str(args.suite.resolve()), "cases": plan, "commands": commands}, ensure_ascii=False, indent=2))
        return 0
    for command in commands:
        if not Path(command[0]).is_file():
            parser.error(f"missing native executable: {command[0]}; build profiling/native first")
    lock_path = ROOT / ".measurement.lock"
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"another run holds {lock_path}; if it crashed, verify its PID before removing the stale lock")
    with os.fdopen(lock, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        output = args.output or ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
        output = output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"output directory must be empty: {output}")
        (output / "raw").mkdir(parents=True, exist_ok=True)
        runtime_dirs = [item.resolve() for item in args.runtime_dir]
        child_env = os.environ.copy()
        child_env["PATH"] = os.pathsep.join([str(item) for item in runtime_dirs] + [child_env.get("PATH", "")])
        if os.name != "nt":
            child_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(item) for item in runtime_dirs] + [child_env.get("LD_LIBRARY_PATH", "")])
        child_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        # Timing instrumentation from the unrelated private fork must not append to old data.
        child_env.pop("LLAMA_VIZ_TIMING", None)
        source_paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "native").glob("*.cpp")) + [ROOT / "native" / "CMakeLists.txt"]
        runtime_files = {Path(command[0]) for command in commands}
        for folder in [binary_dir, *runtime_dirs]:
            runtime_files.update(path for path in folder.glob("*ggml*") if path.suffix in {".dll", ".so", ".dylib"} or ".so." in path.name)
        manifest = {"schema_version": "measurement_profile_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_id": output.name, "environment": environment(), "suite": suite,
            "suite_sha256": sha256(args.suite), "case_order": [case["id"] for case in plan],
            "protocol": {"warmup": args.warmup, "samples": args.samples, "threads": args.threads,
                "seed": args.seed, "sequential_execution": True, "ops_per_sample": 1,
                "cuda_graphs": "disabled", "timing": "single isolated task wall latency; not concurrent prediction",
                "inputs": "deterministic generated tensors; not model weight contents"},
            "software": {"backend_revision_declared": args.backend_revision,
                "benchmark_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths if path.is_file()},
                "runtime_sha256": {str(path): sha256(path) for path in sorted(runtime_files) if path.is_file()},
                "runtime_search_dirs": [str(path) for path in runtime_dirs],
                "runtime_hash_scope": "native executables and provided GGML library files; resolved loader paths are not observed",
                "child_flags": {key: child_env.get(key) for key in ("GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}}}
        dump(output / "manifest.json", manifest)
        dump(output / "plan.json", plan)
        rows = []
        for index, (case, command) in enumerate(zip(plan, commands), 1):
            print(f"[{index}/{len(plan)}] {case['id']}", flush=True)
            before = gpu_snapshot() if case["kind"] == "transfer" or case.get("backend") == "CUDA0" else None
            raw_name = f"raw/{index:03d}.json"
            raw = {"case": case, "command": command, "gpu_before": before}
            row = {"case_id": case["id"], "case": case, "signature": signature(case), "raw_output": raw_name}
            try:
                result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=child_env, timeout=args.timeout_s)
                raw.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode != 0:
                    raise RuntimeError(f"native probe exited {result.returncode}: {result.stderr[-1000:]}")
                native = json.loads(result.stdout)
                validate_native_binding(case, native, args)
                native["kind"] = case["kind"]
                native["expected_samples"] = args.samples
                native = validate_measurement(native)
                row.update(status="measured", native=native, summary=summarize_samples(native["wall_ns"]))
                row["quality_flags"] = ["high_observed_variability_cv_gt_0.1"] if row["summary"]["cv"] > 0.1 else []
                if "device_ns" in native:
                    if not isinstance(native["device_ns"], list) or len(native["device_ns"]) != len(native["wall_ns"]):
                        raise ValueError("device_ns and wall_ns must contain the same number of samples")
                    row["device_summary"] = summarize_samples(native["device_ns"])
                row["eligibility"] = {"isolated_latency": True, "concurrent_latency": False,
                    "workspace_known": native.get("workspace_bytes") is not None if case["kind"] == "compute" else None,
                    "exact_signature_required": True, "automatic_M1_export": False}
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    for key, captured in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                        raw[key] = captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured
                    raw["timed_out"] = True
                raw["error"] = str(exc)
                row.update(status="failed", error=str(exc))
            raw["gpu_after"] = gpu_snapshot() if before is not None else None
            dump(output / raw_name, raw)
            rows.append(row)
            dump(output / "measurements.json", rows)
            write_summary(output, rows)
            if row["status"] == "failed":
                print(row["error"], file=sys.stderr)
                if not args.keep_going:
                    break
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_cases"] = len(rows)
        manifest["measured_cases"] = sum(row["status"] == "measured" for row in rows)
        manifest["planned_cases"] = len(plan)
        manifest["environment_after"] = environment()
        dump(output / "manifest.json", manifest)
        print(f"Measured {manifest['measured_cases']}/{len(plan)} cases: {output}")
        return 0 if len(rows) == len(plan) and all(row["status"] == "measured" for row in rows) else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"run_measurements: {exc}", file=sys.stderr)
        raise SystemExit(1)
