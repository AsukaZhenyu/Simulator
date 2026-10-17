"""Measure concrete execution transforms with independent-process holdout.

No joint strategy duration enters the prediction model. CUDA workspace remains
unknown and this runner does not export an accepted capacity cost table.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent
sys.path[:0] = [str(HERE), str(PROFILING)]
import analysis as a
from run_measurements import environment

VERSION = "execution-transform-suite-0.1"
CASE_FIELDS = {"id", "family", "backend", "dtype", "hidden", "intermediate", "cpu_intermediate",
               "tokens", "history", "heads", "kv_heads", "head_dim"}
PROTOCOL_FIELDS = {"threads", "cpu_id", "device", "warmup", "samples", "min_warmup_ms", "cuda_graphs"}
ENV_NAMES = {"GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
             "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "OMP_PROC_BIND",
             "OMP_PLACES", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NVIDIA_TF32_OVERRIDE"}


def document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def _int(value, label, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{label} must be an explicit integer in {lower}..{upper}")


def validate_suite(original):
    suite = copy.deepcopy(original)
    fields = {"format_version", "cases", "protocol", "profile_seeds", "heldout_seeds", "repeats", "quality", "bootstrap"}
    if not isinstance(suite, dict) or set(suite) != fields or suite["format_version"] != VERSION:
        raise ValueError("suite requires all explicit documented fields")
    _int(suite["repeats"], "repeats", 2, 8)
    seen = set()
    for split in ("profile_seeds", "heldout_seeds"):
        seeds = suite[split]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(split + " requires nonempty seeds")
        for seed in seeds:
            _int(seed, "seed", 0, 1000000)
            if seed in seen:
                raise ValueError("profile/heldout seeds must be disjoint and unique")
            seen.add(seed)
    protocol = suite["protocol"]
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_FIELDS:
        raise ValueError("all protocol fields must be explicit")
    for key, lo, hi in (("threads", 1, 1), ("cpu_id", 0, 63), ("device", 0, 64),
                        ("warmup", 0, 10000), ("samples", 3, 10000), ("min_warmup_ms", 0, 60000)):
        _int(protocol[key], key, lo, hi)
    if protocol["cuda_graphs"] != "disabled":
        raise ValueError("the measured boundary requires CUDA graphs disabled")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("explicit nonempty cases are required")
    ids = set()
    for case in suite["cases"]:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError("case fields must be exact")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) or case["id"] in ids:
            raise ValueError("case ID must be safe, unique and nonempty")
        ids.add(case["id"])
        if case["family"] not in ("ffn", "attention") or case["backend"] != "hybrid" or case["dtype"] != "f32":
            raise ValueError("only the explicit F32 hybrid transform boundary is supported")
        for key, lo, hi in (("hidden", 1, 1024), ("intermediate", 0, 4096), ("cpu_intermediate", 0, 4096),
                           ("tokens", 1, 128), ("history", 0, 4096), ("heads", 0, 1024),
                           ("kv_heads", 0, 1024), ("head_dim", 0, 1024)):
            _int(case[key], key, lo, hi)
        if case["family"] == "ffn":
            if not 0 < case["cpu_intermediate"] < case["intermediate"] or any(case[k] for k in ("history", "heads", "kv_heads", "head_dim")):
                raise ValueError("FFN requires explicit nonempty CPU/GPU intermediate partitions and no attention geometry")
        elif (case["intermediate"] or case["cpu_intermediate"] or not case["heads"] or not case["kv_heads"]
              or not case["head_dim"] or case["heads"] * case["head_dim"] != case["hidden"] or case["heads"] % case["kv_heads"]):
            raise ValueError("invalid explicit GQA geometry")
    quality = suite["quality"]
    if not isinstance(quality, dict) or set(quality) != {"max_within_process_cv", "max_between_process_cv", "max_heldout_relative_error"}:
        raise ValueError("all quality thresholds must be declared before measurement")
    for key, value in quality.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("invalid quality threshold: " + key)
    boot = suite["bootstrap"]
    if not isinstance(boot, dict) or set(boot) != {"iterations", "seed"}:
        raise ValueError("bootstrap protocol must be explicit")
    _int(boot["iterations"], "bootstrap iterations", 100, 10000)
    _int(boot["seed"], "bootstrap seed", 0, 1000000)
    return suite


def recipes(case):
    if case["family"] == "ffn":
        cpu = {"sum": ["cpu/shard_" + x for x in ("gate", "up", "activation", "down")]}
        gpu = {"sum": ["gpu/shard_" + x for x in ("gate", "up", "activation", "down")]}
        mixed = {"sum": [{"max": [cpu, gpu]}, "hybrid/output_d2h", "cpu/sum"]}
        return {"gpu/resident": {"sum": [*["gpu/full_" + x for x in ("gate", "up", "activation", "down")], "gpu/output_d2h"]},
                "hybrid/resident": mixed,
                "hybrid/preload": {"sum": ["hybrid/preload_weights", mixed]}}
    return {"gpu/resident": {"sum": ["gpu/qk", "gpu/softmax", "gpu/pv", "gpu/output", "gpu/output_d2h"]},
            "hybrid/resident": {"sum": ["cpu/qk", "cpu/softmax", "hybrid/probabilities_h2d", "gpu/pv", "gpu/output", "gpu/output_d2h"]}}


def calibration_recipes(case):
    """Two predeclared FFN models; neither fits the measured parallel total."""
    original = recipes(case)
    if case["family"] != "ffn":
        return original
    result = copy.deepcopy(original)
    result["gpu/resident@dispatch_model"] = copy.deepcopy(original["gpu/resident"])
    result["hybrid/resident@dispatch_model"] = {"sum": ["hybrid/dispatch_barrier", copy.deepcopy(original["hybrid/resident"])]}
    result["hybrid/preload@dispatch_model"] = {"sum": ["hybrid/preload_weights", copy.deepcopy(result["hybrid/resident@dispatch_model"])]}
    return result


def observed_aliases(case):
    return {strategy: strategy.split("@", 1)[0] for strategy in calibration_recipes(case) if "@" in strategy}


def expected_metrics(case):
    result = set(recipes(case)) | set().union(*(a.terms(expression) for expression in recipes(case).values()))
    if case["family"] == "ffn":
        result |= {"hybrid/dispatch_barrier", "hybrid/parallel_branches"}
    else:
        result.add("cpu/pv")
    return result


def command(binary, case, protocol, seed):
    argv = [str(binary)]
    for key in ("family", "backend", "hidden", "intermediate", "cpu_intermediate", "tokens", "history", "heads", "kv_heads", "head_dim", "dtype"):
        argv += ["--" + key.replace("_", "-"), str(case[key])]
    for key in ("threads", "cpu_id", "device", "warmup", "samples", "min_warmup_ms", "cuda_graphs"):
        argv += ["--" + key.replace("_", "-"), str(protocol[key])]
    return argv + ["--seed", str(seed)]


def expected_port_bytes(case):
    h, q, i, cut = (case[k] for k in ("hidden", "tokens", "intermediate", "cpu_intermediate"))
    if case["family"] == "ffn":
        result = {}
        for prefix, width in (("cpu/shard/", cut), ("gpu/shard/", i - cut), ("gpu/full/", i)):
            result.update({prefix + name: count * 4 for name, count in {
                "input": h*q, "gate_weight": h*width, "up_weight": h*width,
                "down_weight": width*h, "gate": width*q, "up": width*q,
                "activated": width*q, "output": h*q}.items()})
        return result
    d, heads, kvh, length = case["head_dim"], case["heads"], case["kv_heads"], case["history"] + q
    sizes = {"query": d*q*heads, "new_key": d*q*kvh, "new_value": d*q*kvh,
             "key_cache": d*length*kvh, "value_cache": d*length*kvh,
             "key_appended": d*length*kvh, "value_appended": d*length*kvh,
             "context": h*q, "hidden": h*q, "attention_gate": h*q, "output": h*q,
             "output_weight": h*h}
    return {prefix + name: count * 4 for prefix in ("cpu/", "gpu/") for name, count in sizes.items()}


def validate_native(row, payload, suite):
    case, protocol = row["case"], suite["protocol"]
    if payload.get("schema") != "execution-transform-native-0.1":
        raise ValueError("unexpected native schema")
    if payload.get("matmul_precision") != "GGML_PREC_F32_explicit":
        raise ValueError("native explicit F32 matmul precision contract missing")
    for key in CASE_FIELDS - {"id"}:
        if payload.get(key) != case[key]:
            raise ValueError("native geometry binding mismatch: " + key)
    for key in PROTOCOL_FIELDS:
        if payload.get(key) != protocol[key]:
            raise ValueError("native protocol binding mismatch: " + key)
    if payload.get("seed") != row["seed"] or payload.get("timer") != "steady_clock_completed_work":
        raise ValueError("native seed/timer mismatch")
    metrics = payload.get("wall_ns")
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics(case):
        raise ValueError("native metric coverage mismatch")
    for values in metrics.values():
        a.stats(values)
        if len(values) != protocol["samples"]:
            raise ValueError("native sample count mismatch")
    boundary = payload.get("boundary", {})
    if (boundary.get("output_location") != "dram" or boundary.get("merge_device") != "cpu"
            or boundary.get("input_location") != "all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu"
            or boundary.get("weights") != "generated_f32_not_model_weights"):
        raise ValueError("native input/output boundary mismatch")
    checks = payload.get("checks")
    if payload.get("all_checks_passed") is not True or not isinstance(checks, dict) or not checks:
        raise ValueError("native numerical evidence missing")
    required = {"gpu/resident/output", "hybrid/resident/output"}
    if case["family"] == "ffn":
        required.add("hybrid/preload/output")
        required |= {device + "/shard/" + stage for device in ("cpu", "gpu") for stage in ("gate", "up", "activated", "output")}
    else:
        required |= {"gpu/pv/context", "gpu/pv/key_appended", "gpu/pv/value_appended", "cpu/cache_unchanged", "gpu/cache_unchanged", "cpu/pv/context"}
    if not required <= set(checks):
        raise ValueError("native numerical endpoint coverage missing")
    for check in checks.values():
        if check.get("passed") is not True:
            raise ValueError("native numerical check failed")
        for key in ("max_abs_error", "max_relative_error", "atol", "rtol"):
            value = check.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("native numerical error evidence invalid")
        _int(check.get("elements"), "numerical elements", 1, 1 << 40)
        if check["atol"] != .00001 or check["rtol"] != .0001:
            raise ValueError("native numerical tolerance differs from the declared fixed oracle")
    ports = payload.get("ports")
    expected = expected_port_bytes(case)
    if not isinstance(ports, dict) or set(ports) != set(expected):
        raise ValueError("native port inventory mismatch")
    for name, size in expected.items():
        if ports[name].get("bytes") != size or not re.fullmatch("[0-9a-f]{16}", ports[name].get("input_fixture_fnv1a64", "")):
            raise ValueError("native port bytes/digest mismatch: " + name)
    workspaces = payload.get("workspaces")
    if not isinstance(workspaces, dict) or not workspaces or not any(k.startswith("cpu/") for k in workspaces) or not any(k.startswith("gpu/") for k in workspaces):
        raise ValueError("native workspace inventory missing")
    for name, workspace in workspaces.items():
        _int(workspace.get("known_gallocr_plus_cplan_bytes"), "known allocator bytes", 0, 1 << 50)
        if name.startswith("gpu/"):
            if workspace.get("workspace_bytes") is not None or workspace.get("source") != "unknown_private_cuda_scratch":
                raise ValueError("private GPU workspace must remain unknown")
        elif name.startswith("cpu/"):
            if workspace.get("workspace_bytes") != workspace["known_gallocr_plus_cplan_bytes"] or workspace.get("source") != "gallocr_plus_explicit_cplan":
                raise ValueError("CPU workspace evidence mismatch")
        else:
            raise ValueError("unknown workspace device")
    if payload.get("gpu_capacity_proved") is not False or payload.get("control_overhead_separately_identified") is not False:
        raise ValueError("unsupported capacity/control-overhead claim")


def inventory(paths):
    return {str(Path(path).resolve()): a.sha(path) for path in sorted(set(map(str, paths)))}


def child_context(runtime_dirs):
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(p).resolve()) for p in runtime_dirs] + [env.get("PATH", "")])
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(Path(p).resolve()) for p in runtime_dirs] + [env.get("LD_LIBRARY_PATH", "")])
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    names = ENV_NAMES | {name for name in env if name.startswith(("GGML_", "CUDA_", "OMP_", "CUBLAS_"))}
    return env, {name: env.get(name) for name in sorted(names)}


def jobs(suite, split):
    result = []
    for seed in suite[split + "_seeds"]:
        for repeat in range(suite["repeats"]):
            ordered = list(suite["cases"])
            random.Random(seed + repeat).shuffle(ordered)
            for case in ordered:
                result.append({"id": f"{split}-{case['id']}-s{seed}-r{repeat}", "case": case,
                               "split": split, "seed": seed, "repeat": repeat})
    return result


def _execute(row, args, suite, out, env, flags):
    row = copy.deepcopy(row)
    row["command"] = command(args.binary.resolve(), row["case"], suite["protocol"], row["seed"])
    row["raw_path"] = str(out / "raw" / (row["id"] + ".json"))
    record = {"command": row["command"], "observed_child_flags": flags,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    if row["split"] == "heldout":
        record["frozen_model_sha256_at_start"] = row["frozen_model_sha256_at_start"]
    start = time.perf_counter_ns()
    try:
        proc = subprocess.Popen(row["command"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record["child_pid"] = proc.pid
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            record["timed_out"] = True
        record.update(returncode=proc.returncode, stdout=stdout.decode("utf-8", errors="replace"), stderr=stderr.decode("utf-8", errors="replace"))
        if record.get("timed_out") or proc.returncode:
            raise ValueError("native timeout or nonzero exit: " + str(proc.returncode))
        payload = json.loads(record["stdout"])
        validate_native(row, payload, suite)
        row.update(status="measured", metrics=payload["wall_ns"], checks=payload["checks"],
                   workspaces=payload["workspaces"], ports=payload["ports"], boundary=payload["boundary"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        row.update(status="failed", error=str(exc))
        record["validation_error"] = str(exc)
    record.update(completed_utc=datetime.now(timezone.utc).isoformat(), process_wall_ns=time.perf_counter_ns() - start)
    a.dump(row["raw_path"], record)
    row["raw_sha256"] = a.sha(row["raw_path"])
    row["child_pid"] = record.get("child_pid")
    print(row["id"] + ": " + row["status"], flush=True)
    return row


def run(args):
    suite = validate_suite(document(args.suite))
    if args.timeout_s < 1:
        raise ValueError("timeout must be an explicit positive integer")
    if not args.binary.is_file() or any(not path.is_dir() for path in args.runtime_dir):
        raise ValueError("binary/runtime paths must exist")
    out = args.output.resolve()
    if out.exists():
        raise ValueError("output must be a new directory")
    plan = [dict(row, command=command(args.binary.resolve(), row["case"], suite["protocol"], row["seed"]))
            for split in ("profile", "heldout") for row in jobs(suite, split)]
    if args.dry_run:
        print(json.dumps({"schema": "execution-transform-dry-run-0.1", "jobs": plan}, indent=2))
        return 0
    sources = [*HERE.glob("*.py"), *PROFILING.joinpath("native", "execution_transforms").glob("*.cpp"),
               *PROFILING.joinpath("native", "execution_transforms").glob("CMakeLists.txt"),
               PROFILING / "native" / "matched_mechanisms" / "compute.cpp", PROFILING / "run_measurements.py"]
    runtime = [p for directory in args.runtime_dir for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".dll", ".so", ".dylib")]
    before = inventory([args.suite.resolve(), args.binary.resolve(), *sources, *runtime, *args.preserve_file])
    lock = PROFILING / ".measurement.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        out.mkdir(parents=True)
        (out / "raw").mkdir()
        (out / "source").mkdir()
        snapshots = {}
        for index, source in enumerate(sources):
            destination = out / "source" / (f"{index:03d}-" + source.name)
            destination.write_bytes(source.read_bytes())
            snapshots[str(source.resolve())] = {"path": str(destination.relative_to(out)), "sha256": a.sha(destination)}
        a.dump(out / "suite.json", suite)
        a.dump(out / "plan.json", plan)
        env, flags = child_context(args.runtime_dir)
        manifest = {"schema": "execution-transform-manifest-0.1", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "inventory_before": before, "source_snapshots": snapshots,
                    "environment_before": environment(), "observed_child_flags": flags,
                    "sample_filtering": "none", "user_applications_changed": False,
                    "holdout_unit": "fresh native process with disjoint input seed; same declared shape"}
        a.dump(out / "manifest.json", manifest)
        rows = []
        declared = {case["id"]: calibration_recipes(case) for case in suite["cases"]}
        aliases = {case["id"]: observed_aliases(case) for case in suite["cases"]}
        for split in ("profile", "heldout"):
            for job in jobs(suite, split):
                if inventory(before) != before:
                    raise ValueError("source/binary/runtime/protocol changed during experiment")
                if child_context(args.runtime_dir)[1] != flags:
                    raise ValueError("child runtime flags changed during experiment")
                if split == "heldout" and a.sha(out / "prediction-model.json") != manifest["model_frozen_before_heldout_sha256"]:
                    raise ValueError("profile model changed during heldout")
                if split == "heldout":
                    job["frozen_model_sha256_at_start"] = manifest["model_frozen_before_heldout_sha256"]
                rows.append(_execute(job, args, suite, out, env, flags))
                a.dump(out / "measurements.json", rows)
            if split == "profile":
                model = a.freeze_model(rows, declared, aliases)
                a.dump(out / "prediction-model.json", model)
                manifest["model_frozen_before_heldout_sha256"] = a.sha(out / "prediction-model.json")
                manifest["model_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                a.dump(out / "manifest.json", manifest)
        if a.freeze_model(rows, declared, aliases) != model or a.sha(out / "prediction-model.json") != manifest["model_frozen_before_heldout_sha256"]:
            raise ValueError("profile-only frozen model changed")
        comparison_groups = {case["id"]: {strategy: ("weights_resident_inputs_resident_output_dram" if strategy.split("@", 1)[0].endswith("/resident") else "three_gpu_shard_weights_reloaded_then_output_dram")
                                                    + (";dispatch_model" if "@" in strategy else ";ideal_model")
                                                 for strategy in calibration_recipes(case)} for case in suite["cases"]}
        report = a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
                            scope="真实 F32 FFN 中间维 CPU/GPU 分片并行与 CPU 汇合，以及 post-RoPE Attention QK/softmax/PV 异构串行链。",
                            frozen_before_heldout=True, comparison_groups=comparison_groups)
        report["unknown_or_uncovered"].extend([
            "FFN 不包含 Dense 的 norm/residual 或 MoE 的 router/gather/scatter/combine；仅紧凑 selected-rows FFN core",
            "Attention 的具体 KV 物化与 GGML lowering 已声明；尚未证明等同于新 IR 所有 attention variants",
            "preload 测量包含每次三份权重 H2D，未模拟冷缓存，也未验证预取/计算重叠",
            "CPU 与 GPU dispatch worker 继承同一逻辑处理器 affinity；不是多核操作系统竞争模型",
            "hybrid/parallel_branches 只作诊断，没有被回填进独立组件预测",
            "预声明两种 FFN 模型：ideal=max(CPU,GPU)+copy+sum；dispatch_model 再加独立空worker握手时间。后者是待验证假设，非事后拟合或证明不存在重叠/双计成本"])
        report["source_experiment"] = str(out)
        report["source_frozen_model_sha256"] = manifest["model_frozen_before_heldout_sha256"]
        a.write_outputs(out / "analysis", report, model, rows)
        after = inventory(before)
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), inventory_after=after,
                        assets_unchanged=after == before, environment_after=environment(),
                        report_sha256=a.sha(out / "analysis" / "report.json"))
        a.dump(out / "manifest.json", manifest)
        if after != before:
            raise ValueError("source/binary/runtime assets changed during experiment")
        print(json.dumps({"output": str(out), "native_rows": len(rows), "strategies": len(report["results"]),
                          "calibrated": sum(r["calibrated"] for r in report["results"]), "numeric_execution_valid": report["numeric_execution_valid"]}))
        return 0 if report["numeric_execution_valid"] else 2
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--preserve-file", type=Path, action="append", default=[])
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("execution transform calibration: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
