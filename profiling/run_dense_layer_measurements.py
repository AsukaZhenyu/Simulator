"""Measure explicitly configured complete declared decoder stages on CPU, batch 1.

The suite's own schema decides whether the declared block is a dense one or a
sparse (mixture-of-experts) one; every declaration below comes from that
contract, so a sparse case is never measured with a dense price.

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
import threading

MODELING = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))
from llm_infer_model.m2 import dense_contract
from llm_infer_model.m2.dense_contract import KV_DTYPE, EXCLUDED
import contracts
from llm_infer_model.tensor.spec import SpecError
from profile_contract import summarize_samples, validate_measurement
from run_measurements import dump, environment, sha256

ROOT = Path(__file__).resolve().parent


def _same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{label} does not match the exact requested contract")


def measurement_signature(case: dict, binary_hash: str | None = None, contract=None,
                          residency: str | None = None,
                          process_scope: str | None = None,
                          cpu_ids: list[int] | None = None) -> dict:
    contract = contract or dense_contract
    result = contract.signature(case["config"], case["kind"], case["phase"], case["tokens"],
                                case["visible_history"], case["threads"], batch=case["batch"],
                                residency=residency, process_scope=process_scope,
                                cpu_ids=cpu_ids)
    return {"physical_signature": result, "physical_contract_sha256": sha_signature(result),
            "native_binary_sha256": binary_hash}


def declared_process_scope(args) -> str | None:
    """The process scope this invocation declares, or None for the default.

    None is the historical protocol (one probe process per case) and is what
    keeps every previously measured signature byte-identical; "per_suite" is
    emitted only when the run really did measure the whole table inside one
    process, so the two protocols can never be confused for each other.
    """
    scope = getattr(args, "process_scope", None) or "per_case"
    return dense_contract.signature_process_scope(scope) if scope else None


def declared_residency(args) -> str | None:
    """The residency regime this invocation declares, or None for the default.

    None is the historical resident protocol and is what keeps every previously
    measured signature byte-identical; "streaming" is emitted only when the run
    actually evicts before each sample, so a price can never be silently reused
    across the two protocols.
    """
    return "streaming" if getattr(args, "evict_bytes", 0) else None


def declared_placement(args) -> str | None:
    """The manifest label for the placement this invocation declared, or None.

    The label always describes the command, but only a pinned run makes the mask
    part of the physical signature -- see :func:`declared_cpu_ids`.
    """
    cpu_ids = getattr(args, "cpu_ids", None)
    return dense_contract.THREAD_PLACEMENTS["pinned"] if cpu_ids else None


def declared_cpu_ids(args) -> list[int] | None:
    """The pinned logical CPUs this invocation put in its signature, or None.

    None is the historical protocol: the OS chose the workers' logical CPUs and
    no mask entered the signature, which keeps every table measured before the
    dimension existed byte-identical. A pinned run declares the mask, because
    the same threads on two SMT siblings price the same block differently from
    two threads on two separate cores, and a price must only answer to the
    placement that produced it.
    """
    cpu_ids = getattr(args, "cpu_ids", None)
    return list(cpu_ids) if cpu_ids else None


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


def parse_cpu_ids(text: str | None, threads: int) -> list[int] | None:
    """Parse an explicit logical-CPU list, or None to leave placement to the OS.

    Without this the probe still creates a persistent threadpool, but the OS
    chooses which logical CPUs the workers land on, and it may choose the two
    SMT siblings of one physical core or two cores across a CCD boundary.  A
    campaign that compares shapes then mixes that placement effect into every
    price, which is what the probe's strict mask exists to prevent.
    """
    if text is None:
        return None
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"cpu id {part!r} is not a nonnegative integer")
        ids.append(int(part))
    if len(set(ids)) != len(ids):
        raise ValueError(f"cpu ids {ids} repeat a logical CPU")
    if len(ids) != threads:
        raise ValueError(f"{len(ids)} cpu ids for {threads} threads: every worker needs exactly one")
    if any(cpu >= 64 for cpu in ids):
        raise ValueError(f"cpu ids {ids} exceed the single processor group the probe supports")
    return sorted(ids)


def native_command(case: dict, binary_dir: Path, args: argparse.Namespace, contract=None) -> list[str]:
    contract = contract or dense_contract
    suffix = ".exe" if os.name == "nt" else ""
    command = [str(binary_dir / ("ggml-dense-layer-probe" + suffix)),
               *contract.native_arguments(case, args.warmup, args.samples, args.seed,
                                          getattr(args, "evict_bytes", 0))]
    cpu_ids = getattr(args, "cpu_ids", None)
    if cpu_ids:
        command += ["--cpu-mask", ",".join(str(cpu) for cpu in cpu_ids)]
    return command


def _captured(value) -> str:
    """Subprocess output as text, whether the call decoded it or not."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def serve_records(commands: list[list[str]], child_env: dict, stderr_path: Path,
                  timeout_s: float):
    """Drive one probe process case by case, yielding each case's own record.

    The probe runs ``--cases-from-stdin``: one tab-separated argument list per
    line, so every case is the very command line a standalone invocation would
    have used.  Only the process is shared, which is the whole point -- whatever
    a process starts out with is a property of the process, and measuring a
    table inside one process keeps that property out of the ratios between its
    entries.

    Input is written one line at a time and only when its record is about to be
    read.  That keeps the write side from ever filling the pipe while the reader
    waits on a measurement, and it means a case that hangs or kills the process
    leaves every earlier record already consumed rather than lost in a buffer.

    Yields ``(index, command, record_text, code, note)``.  ``note`` is not None
    when the case produced no record at all (the process timed out, ended, or
    was never reached); ``record_text`` is then None.
    """
    executable = commands[0][0]
    stderr_file = stderr_path.open("w", encoding="utf-8", newline="\n")
    proc = subprocess.Popen([executable, "--cases-from-stdin"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=stderr_file, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=child_env)
    expired = []

    def expire():
        expired.append(True)
        proc.kill()

    timer = threading.Timer(timeout_s, expire)
    timer.start()
    try:
        for index, command in enumerate(commands):
            proc.stdin.write("\t".join(command[1:]) + "\n")
            proc.stdin.flush()
            record, code = [], None
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                if line.startswith("#case "):
                    code = int(line.split()[2])
                    break
                record.append(line.rstrip("\r\n"))
            if code is None:
                note = "probe process timed out" if expired else "probe process ended early"
                yield index, command, None, None, note
                for rest in range(index + 1, len(commands)):
                    yield rest, commands[rest], None, None, "not run: " + note
                return
            yield index, command, "\n".join(record), code, None
    finally:
        timer.cancel()
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()
        stderr_file.close()


def validate_native_binding(case: dict, native: dict, args: argparse.Namespace, contract=None) -> None:
    """Bind native geometry, arithmetic, materialized ports and timing protocol."""
    contract = contract or dense_contract
    if not isinstance(native, dict):
        raise ValueError("native output must be one JSON object")
    expected = {"schema": "ggml_dense_layer_probe/1", "kind": "dense_layer",
        "contract": contract.CONTRACT, "stage": case["kind"], "backend": "CPU", "batch": 1,
        "config": case["config"], "q": case["tokens"], "history": case["visible_history"],
        "activation_dtype": "f32", "kv_dtype": KV_DTYPE,
        "implementation_variant": contract.VARIANTS[case["kind"]],
        "included": contract.operations(case["config"], case["kind"]), "excluded": list(EXCLUDED),
        "evict_bytes": getattr(args, "evict_bytes", 0),
        "residency_regime": contract.RESIDENCY_REGIMES[declared_residency(args) or "resident"],
        "timing": ("steady_clock_graph_compute_including_sync_after_declared_cache_eviction"
                   if getattr(args, "evict_bytes", 0)
                   else "steady_clock_graph_compute_including_sync"),
        "scope": contract.SCOPES.get(case["kind"], "resident_repeated_complete_declared_decoder_block"),
        "setup_compute_repetitions": 1,
        "upstream_setup_compute_repetitions": 1 if case["kind"] in contract.STAGES[:3] else 0,
        "thread_management": "persistent_ggml_threadpool",
        "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
        "position_dtype": "i32", "threads": case["threads"],
        "threads_requested": case["threads"], "threads_applied": case["threads"],
        "warmup": args.warmup, "samples": args.samples, "seed": args.seed}
    for key, value in expected.items():
        actual = contract.validate_config(native.get(key)) if key == "config" else native.get(key)
        _same(actual, value, f"native {key}")
    cpu_ids = getattr(args, "cpu_ids", None)
    if cpu_ids:
        affinity = native.get("cpu_affinity")
        if not isinstance(affinity, dict):
            raise ValueError("native output lacks cpu affinity evidence for the requested mask")
        if affinity.get("verified") is not True or affinity.get("verified_before_and_after_timing") is not True:
            raise ValueError("native cpu affinity was not verified before and after timing")
        if affinity.get("strict_cpu") is not True:
            raise ValueError("native cpu affinity is not strict for an explicitly requested mask")
    validate_memory_accounting(native)
    reference = native.get("reference_check")
    if native.get("all_outputs_finite") is not True or not isinstance(reference, dict) or reference.get("passed") is not True:
        raise ValueError("native Dense stage did not pass its numerical checks")
    if case["kind"] in ("attention", "block"):
        append = native.get("kv_append_check")
        if not isinstance(append, dict) or any(append.get(field) is not True for field in ("passed", "append_exact", "history_unchanged")):
            raise ValueError("attention/block requires successful KV append and history-preservation checks")
    expected_memory = contract.resident_accounting(case["config"], case["kind"], case["tokens"], case["visible_history"])
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
    inputs, outputs = contract.ports(case["config"], case["kind"], case["tokens"])
    expected_port_accounting = {}
    for category, declared in (("input", inputs), ("output", outputs)):
        for name, port in declared.items():
            expected_port_accounting[name] = {"dtype": port["dtype"], "shape": list(port["ne"][:3]), "ne": list(port["ne"]),
                "bytes": math.prod(port["ne"])*4, "category": category}
    if case["kind"] in ("attention", "block"):
        config = case["config"]
        length = case["visible_history"] + case["tokens"]
        # K is row-major [head, position, kv head]. V is stored transposed as
        # [position, head, kv head] so the PV product consumes the cache
        # directly, which is the layout llama.cpp reads.
        for name, shape in (("key_cache", [config["head_dim"], length, config["kv_heads"]]),
                            ("value_cache", [length, config["head_dim"], config["kv_heads"]])):
            expected_port_accounting[name] = {"dtype": KV_DTYPE, "shape": shape,
                "bytes": expected_memory["kv_cache_bytes"]//2, "category": "kv_cache",
                "ne": shape + [1]}
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
    parser.add_argument("--cpu-mask", help="explicit logical CPUs for the workers, e.g. 0,2; omitted leaves placement to the OS scheduler")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evict-bytes", type=int, default=0,
                        help="declared cold residency regime: stream this many bytes of scratch outside the "
                             "timed region before each warmup iteration and each sample (0 = the resident "
                             "protocol where consecutive samples share caches); must exceed the last-level "
                             "cache, and the buffer stays allocated for the whole run")
    parser.add_argument("--process-scope", choices=sorted(dense_contract.PROCESS_SCOPES), default="per_case",
                        help="which process a price is measured in: 'per-case' (default, historical) gives every "
                             "case its own probe process; 'per-suite' measures the whole table inside one probe "
                             "process in the shuffled case order, so a machine state common to every entry cancels "
                             "out of the ratios between entries. The scope is part of the physical signature")
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend-revision", help="Declared source revision; actual executable/library hashes recorded separately")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warmup < 0 or not 3 <= args.samples <= 10000 or not 0 <= args.seed <= 0xFFFFFFFF or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid threads, warmup, samples, seed or timeout")
    if args.evict_bytes < 0 or args.evict_bytes > (1 << 34):
        parser.error("--evict-bytes must be 0..17179869184; the probe allocates it once per invocation")
    try:
        args.cpu_ids = parse_cpu_ids(args.cpu_mask, args.threads)
    except ValueError as error:
        parser.error(str(error))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    contract = contracts.for_suite(suite)
    plan = contract.build_plan(suite, args.threads)
    random.Random(args.seed).shuffle(plan)
    kinds = sorted({case["kind"] for case in plan})
    binary_dir = args.native_bin.resolve()
    commands = [native_command(case, binary_dir, args, contract) for case in plan]
    if args.dry_run:
        print(json.dumps({"contract": contract.CONTRACT, "supported_backend": "CPU", "supported_batch": 1,
            "suite": str(args.suite.resolve()), "cases": plan, "commands": commands}, ensure_ascii=False, indent=2))
        return 0
    for command in commands:
        if not Path(command[0]).is_file():
            parser.error(f"missing decoder probe: {command[0]}; build profiling/native first")
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
        manifest = {"schema_version": contracts.PROFILE_SCHEMAS[contract.CONTRACT],
            "contract": contract.CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "profile_id": output.name,
            "environment": environment(), "suite": suite, "suite_sha256": sha256(args.suite),
            "case_order": [case["id"] for case in plan],
            "protocol": {"backend": "CPU", "batch": 1, "threads": args.threads,
                "cpu_mask": args.cpu_ids,
                "thread_placement": (declared_placement(args)
                                     or dense_contract.THREAD_PLACEMENTS["unpinned"]),
                # A mask that was merely passed on the command line is a detail
                # of this run; a mask that entered the physical signature is an
                # identity. Written only in the latter case, so a manifest and
                # a record always agree on which of the two they describe.
                **({"cpu_mask_in_physical_signature": True} if declared_cpu_ids(args) else {}),
                "warmup": args.warmup, "samples": args.samples, "seed": args.seed,
                "evict_bytes": args.evict_bytes,
                "residency_regime": contract.RESIDENCY_REGIMES[declared_residency(args) or "resident"],
                "process_scope": contract.PROCESS_SCOPES[declared_process_scope(args) or "per_case"],
                "process_groups": ([{"index": 1, "case_ids": [case["id"] for case in plan],
                                     "probe": str(binary),
                                     "stdin_framing": "one tab-separated argument list per line; each case's "
                                                      "record is followed by a '#case <index> <code>' sentinel",
                                     "stderr_file": "raw/suite-001.stderr.txt"}]
                                   if declared_process_scope(args) else []),
                "per_process_timeout_s": (args.timeout_s * len(plan) if declared_process_scope(args)
                                          else args.timeout_s),
                "eviction_buffer_scope": ("one scratch allocation streamed outside the timed region before each "
                                          "warmup iteration and each sample; never inside a sample"),
                "sequential_execution": True, "graphs_per_sample": 1,
                "setup_compute_repetitions": 1, "measured_graph_untimed_compute_repetitions": 1 + args.warmup,
                "upstream_setup_compute_repetitions_by_kind": {kind: (1 if kind in contract.STAGES[:3] else 0) for kind in kinds},
                "total_untimed_compute_repetitions_by_kind": {kind: args.warmup + (2 if kind in contract.STAGES[:3] else 1) for kind in kinds},
                "timing": "one completed resident CPU declared stage graph; setup and explicit external copies excluded",
                "inputs": "deterministic generated tensors; not pretrained model weight contents",
                "layer_coverage": "complete declared pre-norm decoder block; not the original model kernel"},
            "software": {"backend_revision_declared": args.backend_revision,
                "benchmark_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths if path.is_file()},
                "model_contract_source_sha256": {f"m2/{contract.__name__.rsplit('.', 1)[-1]}.py":
                    sha256(MODELING / "llm_infer_model" / "m2" / (contract.__name__.rsplit(".", 1)[-1] + ".py"))},
                "runtime_sha256": {str(path): sha256(path) for path in sorted(runtime_files) if path.is_file()},
                "runtime_search_dirs": [str(folder) for folder in runtime_dirs],
                "runtime_hash_scope": "native executable and supplied GGML libraries; actual resolved loader paths not observed",
                "child_flags": {key: child_env.get(key) for key in ("OMP_NUM_THREADS", "GGML_CUDA_DISABLE_GRAPHS", "CUDA_VISIBLE_DEVICES")}}}
        dump(output / "manifest.json", manifest)
        dump(output / "plan.json", plan)
        rows = []

        def finalize(case, command, raw_name, raw, stdout_text, stderr_text, returncode, failure=None):
            """Validate one case's record, file it, and append its row."""
            row = {"case_id": case["id"], "case": case,
                "signature": measurement_signature(case, binary_hash, contract,
                                                   declared_residency(args),
                                                   declared_process_scope(args),
                                                   declared_cpu_ids(args)),
                "raw_output": raw_name}
            raw.update(returncode=returncode, stdout=stdout_text, stderr=stderr_text)
            try:
                if failure is not None:
                    raise RuntimeError(failure)
                if returncode is None:
                    raise RuntimeError(f"decoder probe did not finish within {args.timeout_s}s")
                if returncode != 0:
                    raise RuntimeError(f"decoder probe exited {returncode}: {(stderr_text or '')[-1000:]}")
                native = json.loads(stdout_text)
                validate_native_binding(case, native, args, contract)
                native["expected_samples"] = args.samples
                native = validate_measurement(native)
                row.update(status="measured", native=native, summary=summarize_samples(native["wall_ns"]))
                row["quality_flags"] = ["high_observed_variability_cv_gt_0.1"] if row["summary"]["cv"] > 0.1 else []
                row["eligibility"] = {"isolated_latency": True, "concurrent_latency": False,
                    "workspace_known": native["workspace_bytes"] is not None,
                    "exact_signature_required": True, "automatic_M2_export": False,
                    "complete_declared_block_variant": True, "M2_memory_accounting_alignment_verified": False}
            except (OSError, ValueError, RuntimeError, SpecError) as exc:
                raw["error"] = str(exc)
                row.update(status="failed", error=str(exc))
            dump(output / raw_name, raw)
            rows.append(row)
            dump(output / "measurements.json", rows)
            write_summary(output, rows)
            return row

        def stop(row):
            if row["status"] == "failed":
                print(row["error"], file=sys.stderr)
                return not args.keep_going
            return False

        scope = declared_process_scope(args)
        if scope is None:
            for index, (case, command) in enumerate(zip(plan, commands), 1):
                print(f"[{index}/{len(plan)}] {case['id']}", flush=True)
                raw_name = f"raw/{index:03d}.json"
                raw = {"case": case, "command": command}
                try:
                    result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                            env=child_env, timeout=args.timeout_s)
                    row = finalize(case, command, raw_name, raw, result.stdout, result.stderr, result.returncode)
                except subprocess.TimeoutExpired as exc:
                    raw["timed_out"] = True
                    row = finalize(case, command, raw_name, raw,
                                   _captured(exc.stdout), _captured(exc.stderr), None)
                except OSError as exc:
                    row = finalize(case, command, raw_name, raw, "", str(exc), None, failure=str(exc))
                if stop(row):
                    break
        else:
            stderr_name = "raw/suite-001.stderr.txt"
            stderr_path = output / stderr_name
            stream = serve_records(commands, child_env, stderr_path, args.timeout_s * len(plan))
            for index, command, record, code, note in stream:
                case = plan[index]
                print(f"[{index + 1}/{len(plan)}] {case['id']} (one shared process)", flush=True)
                raw_name = f"raw/{index + 1:03d}.json"
                raw = {"case": case, "command": command, "process_group": 1,
                       "process_scope": contract.PROCESS_SCOPES[scope], "stderr_file": stderr_name}
                stderr_text = ""
                if note is None and code != 0:
                    try:
                        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    except OSError:
                        stderr_text = ""
                row = finalize(case, command, raw_name, raw, record, stderr_text, code, failure=note)
                if stop(row):
                    stream.close()
                    break
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), completed_cases=len(rows),
            measured_cases=sum(row["status"] == "measured" for row in rows), planned_cases=len(plan),
            environment_after=environment())
        dump(output / "manifest.json", manifest)
        print(f"Measured {manifest['measured_cases']}/{len(plan)} CPU {contract.CONTRACT} cases: {output}")
        return 0 if len(rows) == len(plan) and all(row["status"] == "measured" for row in rows) else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, SpecError) as exc:
        print(f"run_dense_layer_measurements: {exc}", file=sys.stderr)
        raise SystemExit(1)
