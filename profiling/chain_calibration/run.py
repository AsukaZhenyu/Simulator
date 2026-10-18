"""Collect independent components, real paths and separate in-chain diagnostics.

No hardware is executed on import. All profile and heldout processes and failures
are retained. Profile component prices freeze before the first heldout launch.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent
sys.path[:0] = [str(PROFILING / "execution_calibration"), str(PROFILING)]
import analysis as a
import run_dual as legacy
from run_measurements import environment
sys.path.insert(0, str(PROFILING / "accuracy_suite"))
import trace_model
import shape_models
import runtime_manifest

SCHEMA = "chain-calibration-suite-0.1"
GATES = {"max_within_process_cv": .15, "max_between_process_cv": .20,
         "max_heldout_relative_error": .20}
CONTEXT_SHA = "37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4"
MATCHED_SHA = "2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327"


def validate_suite(original):
    converted = copy.deepcopy(original)
    if converted.get("format_version") != SCHEMA:
        raise ValueError("unexpected chain suite schema")
    converted["format_version"] = legacy.VERSION
    # New explicit size budget, with the historical validator still checking
    # every other field/protocol restriction against bounded placeholders.
    for case in converted.get("cases", []):
        if not isinstance(case, dict) or set(case) != legacy.CASE_FIELDS:
            continue
        legacy._int(case["hidden"], "hidden", 1, 2048)
        legacy._int(case["intermediate"], "intermediate", 0, 8192)
        legacy._int(case["cpu_intermediate"], "cpu_intermediate", 0, 8192)
        if case["family"] == "ffn":
            if not 0 < case["cpu_intermediate"] < case["intermediate"]:
                raise ValueError("FFN requires actual nonempty intermediate shards")
            case["hidden"] = min(case["hidden"], 1024)
            case["intermediate"] = min(case["intermediate"], 4096)
            case["cpu_intermediate"] = 1
        elif case["family"] == "attention" and case["hidden"] > 1024:
            legacy._int(case["head_dim"], "head_dim", 1, 2048)
            legacy._int(case["heads"], "heads", 1, 1024)
            if case["heads"] * case["head_dim"] != case["hidden"]:
                raise ValueError("actual attention geometry differs")
            case["hidden"], case["head_dim"] = case["heads"], 1
    legacy.validate_suite(converted)
    if original["quality"] != GATES:
        raise ValueError("original CV/error gates must remain unchanged")
    for split in ("profile", "heldout"):
        if len(original[split + "_seeds"]) * original["repeats"] < 3:
            raise ValueError("at least three independent processes per case and split required")
    return copy.deepcopy(original)


def recipes(case):
    if case["family"] == "ffn":
        stages = ("gate", "up", "activation", "down")
        summed = lambda prefix: {"sum": [prefix + x for x in stages]}
        hybrid = {"sum": [{"max": [summed("cpu/shard_"), summed("gpu/shard_")]},
                            "hybrid/output_d2h", "cpu/sum"]}
        return {"cpu/resident": summed("cpu/full_"),
                "gpu/resident": {"sum": [summed("gpu/full_"), "gpu/output_d2h"]},
                "hybrid/resident": hybrid,
                "hybrid/preload": {"sum": ["hybrid/preload_weights", hybrid]},
                "cpu/joint_resident": "cpu/full_joint",
                "gpu/joint_resident": {"sum": ["gpu/full_joint", "gpu/output_d2h"]},
                "hybrid/joint_resident": {"sum": [{"max": ["cpu/shard_joint", "gpu/shard_joint"]},
                                                    "hybrid/output_d2h", "cpu/sum"]}}
    return {"cpu/resident": {"sum": ["cpu/qk", "cpu/softmax", "cpu/pv", "cpu/output"]},
            "gpu/resident": {"sum": ["gpu/qk", "gpu/softmax", "gpu/pv", "gpu/output", "gpu/output_d2h"]},
            "hybrid/resident": {"sum": ["cpu/qk", "cpu/softmax", "hybrid/probabilities_h2d",
                                        "gpu/pv", "gpu/output", "gpu/output_d2h"]},
            "cpu/joint_resident": "cpu/joint",
            "gpu/joint_resident": {"sum": ["gpu/joint", "gpu/output_d2h"]}}


def expected_metrics(case):
    expressions = recipes(case)
    result = set(expressions) | set().union(*(a.terms(x) for x in expressions.values()))
    if case["family"] == "ffn":
        result.add("hybrid/dispatch_barrier")
        result.update("hybrid/preload_" + x + "_weight_h2d" for x in ("gate", "up", "down"))
    return result


def trace_inventory(case, strategy):
    """Exact ordered same-thread spans, fixed before collection."""
    if case["family"] == "ffn":
        stages = ["gate", "up", "activation", "down"]
        joint = strategy.endswith("joint_resident")
        if strategy.startswith("hybrid/"):
            main = (["preload_gate_weight_h2d", "preload_up_weight_h2d", "preload_down_weight_h2d"]
                    if strategy == "hybrid/preload" else [])
            return {"main": main + ["dispatch"] + (["cpu_joint"] if joint else stages) + ["join", "output_d2h", "sum"],
                    "dispatch": ["gpu_joint"] if joint else stages}
        dev = strategy.split("/")[0]
        return {"main": ([dev + "_full_joint"] if joint else stages) + (["output_d2h"] if dev == "gpu" else []), "dispatch": []}
    if strategy == "hybrid/resident":
        return {"main": ["cpu_qk", "cpu_softmax", "probabilities_h2d", "gpu_pv", "gpu_attention_output", "output_d2h"], "dispatch": []}
    return {"main": (["joint"] if strategy.endswith("joint_resident") else ["qk", "softmax", "pv", "attention_output"])
            + (["output_d2h"] if strategy.startswith("gpu/") else []), "dispatch": []}


def validate_trace(trace, *, hybrid_ffn, inventory=None):
    if not isinstance(trace, dict) or trace.get("begin_ns") != 0:
        raise ValueError("trace requires relative zero origin")
    end = trace.get("end_ns")
    if isinstance(end, bool) or not isinstance(end, int) or end < 1:
        raise ValueError("trace end must be positive integer nanoseconds")
    stages = trace.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("trace requires real stage spans")
    names = set()
    for span in stages:
        if set(span) != {"name", "thread", "begin_ns", "end_ns"} or not isinstance(span["name"], str):
            raise ValueError("invalid trace span")
        key = (span["thread"], span["name"])
        if key in names or span["thread"] not in ("main", "dispatch"):
            raise ValueError("duplicate or unknown stage/thread")
        names.add(key)
        b, e = span["begin_ns"], span["end_ns"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (b, e)) or not 0 <= b <= e <= end:
            raise ValueError("stage span outside trace")
    for thread in ("main", "dispatch"):
        ordered = sorted((x for x in stages if x["thread"] == thread), key=lambda x: x["begin_ns"])
        if inventory is not None and [x["name"] for x in ordered] != inventory[thread]:
            raise ValueError("exact ordered strategy stage coverage differs: " + thread)
        if any(left["end_ns"] > right["begin_ns"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("same-thread stage spans overlap")
    values = [trace.get(k) for k in ("branch_start_skew_ns", "join_wait_ns", "completed_work_bracket_overlap_ns")]
    if hybrid_ffn:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values) or any(v < 0 for v in values[1:]):
            raise ValueError("hybrid FFN requires signed skew and nonnegative wait/overlap")
        if ("main", "join") not in names or not any(x[0] == "dispatch" for x in names):
            raise ValueError("hybrid FFN requires dispatch branch and join")
        spans = {(x["thread"], x["name"]): x for x in stages}
        dispatch_stage = spans[("main", "dispatch")]
        join = spans[("main", "join")]
        gpu_stages = [x for x in stages if x["thread"] == "dispatch"]
        cpu_stages = [x for x in stages if x["thread"] == "main" and x["name"] in ("gate", "up", "activation", "down", "cpu_joint")]
        if (min(x["begin_ns"] for x in gpu_stages) < dispatch_stage["begin_ns"]
                or min(x["begin_ns"] for x in cpu_stages) < dispatch_stage["end_ns"]
                or max(x["end_ns"] for x in gpu_stages) > join["end_ns"]
                or max(x["end_ns"] for x in cpu_stages) > join["begin_ns"]
                or join["end_ns"] > spans[("main", "output_d2h")]["begin_ns"]
                or spans[("main", "output_d2h")]["end_ns"] > spans[("main", "sum")]["begin_ns"]):
            raise ValueError("hybrid dispatch/join/copy/merge dependency violation")
        brackets = trace.get("branch_brackets")
        if not isinstance(brackets, dict) or set(brackets) != {"cpu", "gpu_host", "join"}:
            raise ValueError("hybrid branch brackets missing")
        for key, branch_stages in (("cpu", cpu_stages), ("gpu_host", gpu_stages), ("join", [join])):
            bracket = brackets[key]
            if (set(bracket) != {"begin_ns", "end_ns"}
                    or not 0 <= bracket["begin_ns"] <= bracket["end_ns"] <= end
                    or min(x["begin_ns"] for x in branch_stages) < bracket["begin_ns"]
                    or max(x["end_ns"] for x in branch_stages) > bracket["end_ns"]):
                raise ValueError("branch bracket does not contain completed-work spans")
        cpu_b, gpu_b, join_b = (brackets[k] for k in ("cpu", "gpu_host", "join"))
        if (values[0] != gpu_b["begin_ns"] - cpu_b["begin_ns"]
                or values[1] != join_b["end_ns"] - join_b["begin_ns"]
                or values[2] != max(0, min(cpu_b["end_ns"], gpu_b["end_ns"]) - max(cpu_b["begin_ns"], gpu_b["begin_ns"]))):
            raise ValueError("derived branch diagnostics differ from raw brackets")
    elif values != [None, None, None]:
        raise ValueError("serial path cannot claim branch evidence")
    elif trace.get("branch_brackets") is not None:
        raise ValueError("serial path cannot claim branch brackets")


def validate_native(row, payload, suite, dispatch):
    case, protocol = row["case"], suite["protocol"]
    if payload.get("schema") != "chain-calibration-native-0.1":
        raise ValueError("unexpected native protocol")
    for key in legacy.CASE_FIELDS - {"id"}:
        if payload.get(key) != case[key]:
            raise ValueError("native geometry mismatch: " + key)
    for key in legacy.PROTOCOL_FIELDS:
        if payload.get(key) != protocol[key]:
            raise ValueError("native protocol mismatch: " + key)
    if (payload.get("seed") != row["seed"] or payload.get("dispatch_cpu_id") != dispatch
            or payload.get("timer") != "steady_clock_completed_work"
            or payload.get("matmul_precision") != "GGML_PREC_F32_explicit"
            or payload.get("nvidia_tf32_override") != "0"):
        raise ValueError("native seed/timer/precision/dispatch contract mismatch")
    if payload.get("source_binding") != {"context_sha256": CONTEXT_SHA, "matched_sha256": MATCHED_SHA,
                                         "probe_sha256": a.sha(PROFILING / "native" / "chain_calibration" / "compute.cpp")}:
        raise ValueError("native historical source binding mismatch")
    metrics = payload.get("wall_ns")
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics(case):
        raise ValueError("native metric coverage mismatch")
    for values in metrics.values():
        a.stats(values)
        if len(values) != protocol["samples"]:
            raise ValueError("native completed-work sample count mismatch")
    strategies = recipes(case)
    traces = payload.get("chain_traces")
    if not isinstance(traces, dict) or set(traces) != set(strategies):
        raise ValueError("trace strategy coverage mismatch")
    for strategy, values in traces.items():
        if len(values) != protocol["samples"]:
            raise ValueError("trace invocation count mismatch")
        for trace in values:
            validate_trace(trace, hybrid_ffn=case["family"] == "ffn" and strategy.startswith("hybrid/"),
                           inventory=trace_inventory(case, strategy))
    costs = payload.get("timestamp_overhead_ns")
    if not isinstance(costs, dict) or set(costs) != {"clock_pair", "noop_stage_pair", *("noop_trace/" + s for s in strategies)}:
        raise ValueError("separate timestamp overhead coverage missing")
    for values in costs.values():
        a.stats(values)
        if len(values) != protocol["samples"]:
            raise ValueError("overhead invocation count mismatch")
    boundary = payload.get("boundary", {})
    if (boundary.get("output_location") != "dram" or boundary.get("overhead_subtracted") is not False
            or boundary.get("input_location") != "all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu"):
        raise ValueError("boundary contract mismatch")
    for claim in ("kernel_overlap_proved", "gpu_capacity_proved", "context_switch_attribution_proved"):
        if payload.get(claim) is not False:
            raise ValueError("unsupported physical attribution: " + claim)
    checks = payload.get("checks")
    if payload.get("all_checks_passed") is not True or not isinstance(checks, dict) or not checks:
        raise ValueError("numerical checks missing or failed")
    if not {s + "/output" for s in strategies} <= set(checks):
        raise ValueError("numerical strategy output coverage missing")
    if case["family"] == "ffn":
        required_checks = {device + "/" + scope + "/" + stage for device in ("cpu", "gpu")
                           for scope in ("shard", "full") for stage in ("gate", "up", "activated", "output")}
    else:
        required_checks = {device + "/" + stage for device in ("cpu", "gpu")
                           for stage in ("context", "key_appended", "value_appended", "key_cache_unchanged", "value_cache_unchanged", "qk/scores", "softmax/probabilities")}
    if not required_checks <= set(checks):
        raise ValueError("numerical intermediate/cache coverage missing")
    for check in checks.values():
        if check.get("passed") is not True or check.get("elements", 0) < 1:
            raise ValueError("numerical port failed")
        for key in ("max_abs_error", "max_relative_error", "atol", "rtol"):
            value = check.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid numerical error evidence")
    aff = payload.get("observed_affinity", {})
    expected = {"process_mask": (1 << protocol["cpu_id"]) | (1 << dispatch),
                "main_thread_mask": 1 << protocol["cpu_id"], "main_thread_group": 0,
                "dispatch_thread_mask": (1 << dispatch) if case["family"] == "ffn" else 0,
                "dispatch_thread_group": 0}
    if aff != expected:
        raise ValueError("native affinity readback mismatch")
    if (payload.get("profile_gpu_shard_thread") != ("persistent_dispatch_worker_same_as_parallel_path" if case["family"] == "ffn" else "not_applicable")
            or payload.get("profile_gpu_shard_affinity_mask") != ((1 << dispatch) if case["family"] == "ffn" else 0)
            or payload.get("profile_gpu_shard_affinity_group") != 0):
        raise ValueError("isolated GPU shard and real chain worker contexts differ")
    expected_ports = legacy.expected_port_bytes(case)
    if case["family"] == "ffn":
        expected_ports.update({k.replace("gpu/full/", "cpu/full/"): v for k, v in list(expected_ports.items()) if k.startswith("gpu/full/")})
    ports = payload.get("ports", {})
    if set(ports) != set(expected_ports) or any(ports[k].get("bytes") != value for k, value in expected_ports.items()):
        raise ValueError("port geometry mismatch")
    if any(not re.fullmatch("[0-9a-f]{16}", str(value.get("input_fixture_fnv1a64"))) for value in ports.values()):
        raise ValueError("port fixture digest missing")


def execute(job, args, suite, out, env, flags):
    row = copy.deepcopy(job)
    raw_path = out / "raw" / (row["id"] + ".json")
    record = {"started_utc": utc(), "command": row["command"], "observed_child_flags": flags,
              "frozen_model_sha256_at_start": row.get("frozen_model_sha256_at_start"),
              "frozen_trace_model_sha256_at_start": row.get("frozen_trace_model_sha256_at_start"),
              "frozen_shape_models_sha256_at_start": row.get("frozen_shape_models_sha256_at_start")}
    record["private_runtime_manifest_sha256_at_start"] = row.get("private_runtime_manifest_sha256_at_start", {})
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
        record.update(returncode=proc.returncode, stdout=stdout.decode("utf-8", errors="replace"),
                      stderr=stderr.decode("utf-8", errors="replace"))
        if record.get("timed_out") or proc.returncode:
            raise ValueError("native timeout/nonzero exit: " + str(proc.returncode))
        payload = json.loads(record["stdout"])
        validate_native(row, payload, suite, legacy.dispatch_cpu_id(flags, suite["protocol"]["cpu_id"]))
        row.update(status="measured", metrics=payload["wall_ns"], checks=payload["checks"], chain_traces=payload["chain_traces"],
                   ports=payload["ports"], boundary=payload["boundary"], child_pid=proc.pid)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="failed", error=str(exc))
        record["validation_error"] = str(exc)
    record.update(completed_utc=utc(), process_wall_ns=time.perf_counter_ns() - start)
    a.dump(raw_path, record)
    row.update(raw_path=str(raw_path), raw_sha256=a.sha(raw_path))
    print(row["id"] + ": " + row["status"], flush=True)
    return row


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_paths():
    return [*HERE.glob("*.py"), *HERE.glob("*.json"),
            *PROFILING.joinpath("native", "chain_calibration").glob("*.cpp"),
            PROFILING / "native" / "chain_calibration" / "CMakeLists.txt",
            PROFILING / "native" / "execution_transforms_dual" / "compute_context.cpp",
            PROFILING / "native" / "matched_mechanisms" / "compute.cpp",
            PROFILING / "execution_calibration" / "analysis.py",
            PROFILING / "execution_calibration" / "run_dual.py", PROFILING / "run_measurements.py"]


def run(args):
    suite = validate_suite(legacy.document(args.suite))
    if args.timeout_s < 1 or not args.binary.is_file() or any(not p.is_dir() for p in args.runtime_dir):
        raise ValueError("explicit existing binary/runtime and positive timeout required")
    out = args.output.resolve()
    if out.exists():
        raise ValueError("output must be a new directory; prior failures are never replaced")
    plan = [dict(job, command=legacy.command(args.binary.resolve(), job["case"], suite["protocol"], job["seed"]))
            for split in ("profile", "heldout") for job in legacy.jobs(suite, split)]
    if args.dry_run:
        print(json.dumps({"schema": "chain-calibration-plan-0.1", "jobs": plan}, indent=2))
        return 0
    sources = [*source_paths(), PROFILING / "accuracy_suite" / "trace_model.py",
               PROFILING / "accuracy_suite" / "shape_models.py", PROFILING / "composition_calibration" / "model.py",
               PROFILING / "native" / "chain_calibration" / "build.py"]
    runtime = [p for directory in args.runtime_dir for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".dll", ".so", ".dylib")]
    bindings = [p for p in (args.binary.parent / "source-binding.json", args.binary.parent.parent / "source-binding.json") if p.is_file()]
    if len(bindings) != 1:
        raise ValueError("native build source-binding.json must exist beside binary or its parent")
    build_binding = legacy.document(bindings[0])
    if build_binding != {"context_sha256": CONTEXT_SHA, "matched_sha256": MATCHED_SHA,
                         "probe_sha256": a.sha(PROFILING / "native" / "chain_calibration" / "compute.cpp"),
                         "cmake_sha256": a.sha(PROFILING / "native" / "chain_calibration" / "CMakeLists.txt"),
                         "generated_include_sha256": a.sha(bindings[0].parent / "context_generated.inc"),
                         "transformation": "rename_terminal_main_only"}:
        raise ValueError("native build historical source binding differs")
    private_runtimes, private_assets = runtime_manifest.discover(args.runtime_dir)
    before = legacy.inventory([args.suite, args.binary, *sources, *runtime, *bindings, *args.preserve_file, *private_assets])
    env, flags = legacy.child_context(args.runtime_dir)
    legacy.dispatch_cpu_id(flags, suite["protocol"]["cpu_id"])
    lock = PROFILING / ".measurement.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "output": str(out), "started_utc": utc()}))
    manifest = None
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
        declared = {case["id"]: recipes(case) for case in suite["cases"]}
        a.dump(out / "recipes.json", declared)
        manifest = {"schema": "chain-calibration-manifest-0.1", "started_utc": utc(),
                    "bound_binary": str(args.binary.resolve()), "private_runtime_builds": private_runtimes,
                    "inventory_before": before, "source_snapshots": snapshots,
                    "plan_sha256_before_collection": a.sha(out / "plan.json"),
                    "suite_sha256_before_collection": a.sha(out / "suite.json"),
                    "recipes_sha256_before_collection": a.sha(out / "recipes.json"),
                    "environment_before": environment(), "observed_child_flags": flags,
                    "sample_filtering": "none", "holdout_unit": "new process, disjoint seed, declared same geometry",
                    "timestamp_overhead_subtracted": False}
        a.dump(out / "manifest.json", manifest)
        rows = []
        for split in ("profile", "heldout"):
            for job in (x for x in plan if x["split"] == split):
                job = dict(job, private_runtime_manifest_sha256_at_start=runtime_manifest.hash_bindings(private_runtimes))
                if legacy.inventory(before) != before or legacy.child_context(args.runtime_dir)[1] != flags:
                    raise ValueError("sources/binary/runtime/environment changed during campaign")
                if a.sha(out / "plan.json") != manifest["plan_sha256_before_collection"]:
                    raise ValueError("frozen collection plan changed")
                if split == "heldout":
                    frozen = manifest["model_frozen_before_heldout_sha256"]
                    if a.sha(out / "prediction-model.json") != frozen:
                        raise ValueError("profile model changed during heldout")
                    frozen_trace = manifest["trace_model_frozen_before_heldout_sha256"]
                    if a.sha(out / "trace-model.json") != frozen_trace:
                        raise ValueError("profile in-chain model changed during heldout")
                    frozen_shape = manifest["shape_models_frozen_before_heldout_sha256"]
                    if a.sha(out / "shape-models.json") != frozen_shape:
                        raise ValueError("profile full/LOCO shared-coefficient models changed during heldout")
                    job = dict(job, frozen_model_sha256_at_start=frozen,
                               frozen_trace_model_sha256_at_start=frozen_trace,
                               frozen_shape_models_sha256_at_start=frozen_shape)
                rows.append(execute(job, args, suite, out, env, flags))
                a.dump(out / "measurements.json", rows)
            if split == "profile":
                model = a.freeze_model(rows, declared)
                a.dump(out / "prediction-model.json", model)
                profile = [r for r in rows if r["split"] == "profile"]
                try:
                    inchain_model = trace_model.freeze(profile)
                except (ValueError, TypeError, KeyError) as exc:
                    inchain_model = {"schema": "chain-profile-trace-model-unavailable-0.1", "status": "unavailable", "error": str(exc)}
                a.dump(out / "trace-model.json", inchain_model)
                try:
                    shape_bundle = shape_models.freeze(profile)
                except (ValueError, TypeError, KeyError) as exc:
                    shape_bundle = {"schema": "chain-shape-models-unavailable-0.1", "status": "unavailable", "error": str(exc)}
                a.dump(out / "shape-models.json", shape_bundle)
                manifest.update(model_frozen_before_heldout_sha256=a.sha(out / "prediction-model.json"), model_frozen_utc=utc())
                manifest.update(trace_model_frozen_before_heldout_sha256=a.sha(out / "trace-model.json"), trace_model_frozen_utc=utc())
                manifest.update(shape_models_frozen_before_heldout_sha256=a.sha(out / "shape-models.json"), shape_models_frozen_utc=utc())
                a.dump(out / "manifest.json", manifest)
        if a.freeze_model(rows, declared) != model:
            raise ValueError("heldout values affected profile-only model")
        for filename, key in (("prediction-model.json", "model_frozen_before_heldout_sha256"),
                              ("trace-model.json", "trace_model_frozen_before_heldout_sha256"),
                              ("shape-models.json", "shape_models_frozen_before_heldout_sha256")):
            if a.sha(out / filename) != manifest[key]:
                raise ValueError("frozen model changed during final heldout process: " + filename)
        comparison_groups = {case["id"]: {strategy: ("three_weights_reloaded_output_dram" if strategy == "hybrid/preload"
                                                          else "resident_inputs_weights_output_dram")
                                          for strategy in recipes(case)} for case in suite["cases"]}
        report = a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
                            scope="F32 compact FFN and post-RoPE attention; completed-work CPU/GPU serial/joint/parallel paths",
                            frozen_before_heldout=True, comparison_groups=comparison_groups)
        report["unknown_or_uncovered"].extend([
            "Timestamped stage spans are separate diagnostics; no timestamp cost is subtracted from uninstrumented totals.",
            "Branch overlap is completed-work host bracket overlap; CUDA kernel overlap and physical context-switch attribution remain unproved.",
            "FFN excludes norm, residual, router, gather, scatter and combine. Attention starts at post-RoPE QKV.",
            "CUDA private workspace/capacity is unknown. Generated F32 fixtures are not full model weights."])
        report["source_experiment"] = str(out)
        by_path = {(x["case_id"], x["strategy"]): x for x in report["results"]}
        report["near_crossing_selection"] = {"threshold_profile_prediction_ratio": 1.25,
            "selection_uses": "profile-only independent-component predictions; no heldout tuning", "pairs": []}
        for ranking in report["rankings"]:
            for pair in ranking["pairs"]:
                left = by_path[(ranking["case_id"], pair["left"])]["predicted_ns"]
                right = by_path[(ranking["case_id"], pair["right"])]["predicted_ns"]
                ratio = max(left, right) / min(left, right)
                if ratio <= 1.25:
                    report["near_crossing_selection"]["pairs"].append({"case_id": ranking["case_id"],
                        "boundary_contract": ranking["boundary_contract"], "profile_prediction_ratio": ratio, **pair})
        a.write_outputs(out / "analysis", report, model, rows)
        from trace_evaluate import evaluate as evaluate_trace
        from diagnostics import summarize
        a.dump(out / "trace-report.json", evaluate_trace(rows, inchain_model, suite, trace_model))
        shape_report = ({"schema": "chain-shape-models-unavailable-0.1", "status": "unavailable", "error": shape_bundle["error"]}
                        if shape_bundle.get("status") == "unavailable" else shape_models.evaluate(shape_bundle, rows, frozen_before_heldout=True))
        a.dump(out / "shape-report.json", shape_report)
        a.dump(out / "diagnostic-summary.json", summarize(rows))
        manifest.update(completed_utc=utc(), inventory_after=legacy.inventory(before), report_sha256=a.sha(out / "analysis" / "report.json"))
        manifest["assets_unchanged"] = manifest["inventory_after"] == before
        a.dump(out / "manifest.json", manifest)
        print(json.dumps({"output": str(out), "rows": len(rows), "failures": sum(r["status"] != "measured" for r in rows),
                          "calibrated": sum(x["calibrated"] for x in report["results"])}))
        return 0 if report["numeric_execution_valid"] else 2
    except BaseException as exc:
        if manifest is not None:
            manifest.update(aborted_utc=utc(), error=str(exc))
            a.dump(out / "manifest.json", manifest)
        raise
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
        print("chain calibration: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
