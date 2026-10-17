"""Explicit completed-work batch timing, independent of historical protocols.

Raw samples are batch totals. Derived per-invocation means are used only for
the explicitly repeated steady-execution prediction problem, never for the
single-invocation quality claim in an older campaign.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent
sys.path[:0] = [str(PROFILING / "execution_calibration"), str(PROFILING)]
import analysis as a
import run_dual as base
import run_measurements
from run_measurements import environment

SCHEMA = "execution-transform-batched-suite-0.1"
WEIGHT_METRICS = {"hybrid/preload_" + name + "_weight_h2d" for name in ("gate", "up", "down")}


def validate_suite(value):
    value = copy.deepcopy(value)
    if value.get("format_version") != SCHEMA:
        raise ValueError("unsupported batched suite")
    batch = value.pop("batch_repetitions", None)
    base._int(batch, "batch_repetitions", 1, 1024)
    value["format_version"] = base.VERSION
    result = base.validate_suite(value)
    result["format_version"] = SCHEMA
    result["batch_repetitions"] = batch
    return result


def normalized_payload(payload, row, suite, dispatch_cpu):
    """Validate untouched native batch evidence before deriving mean durations."""
    if not isinstance(payload, dict):
        raise ValueError("native payload must be a JSON object")
    if (isinstance(payload.get("batch_repetitions"), bool)
            or not isinstance(payload.get("batch_repetitions"), int)
            or payload.get("schema") != "execution-transform-native-batch-0.1"
            or payload.get("timing_unit") != "completed_work_batch_total"
            or payload.get("batch_repetitions") != suite["batch_repetitions"]
            or "wall_ns" in payload):
        raise ValueError("batch schema, unit or repetition mismatch")
    raw = payload.get("batch_wall_ns")
    extras = WEIGHT_METRICS if row["case"]["family"] == "ffn" else set()
    if not isinstance(raw, dict) or set(raw) != base.expected_metrics(row["case"]) | extras:
        raise ValueError("exact raw batch metric coverage required")
    for samples in raw.values():
        a.stats(samples)
        if len(samples) != suite["protocol"]["samples"]:
            raise ValueError("raw batch sample count mismatch")
    factor = suite["batch_repetitions"]
    normalized = {key: [round(x / factor) for x in values] for key, values in raw.items()}
    if any(x < 1 for values in normalized.values() for x in values):
        raise ValueError("batch mean is not representable as positive integer nanoseconds")
    compat = copy.deepcopy(payload)
    compat["schema"] = "execution-transform-native-0.1"
    compat["wall_ns"] = {k: v for k, v in normalized.items() if k not in extras}
    base.validate_native(row, compat, suite, expected_dispatch_cpu_id=dispatch_cpu)
    ffn = row["case"]["family"] == "ffn"
    if payload.get("profile_gpu_shard_thread") != (
            "dispatch_worker_same_as_parallel_strategy" if ffn else "not_applicable_attention_has_no_shard"):
        raise ValueError("component/strategy GPU dispatch context mismatch")
    if (payload.get("profile_gpu_shard_affinity_mask") != ((1 << dispatch_cpu) if ffn else None)
            or payload.get("profile_gpu_shard_affinity_group") != (0 if ffn else None)):
        raise ValueError("component dispatch affinity readback mismatch")
    if payload.get("nvidia_tf32_override") != "0":
        raise ValueError("explicit F32/TF32 environment contract missing")
    return normalized


def recipes(case):
    result = base.calibration_recipes(case)
    if case["family"] == "ffn":
        independent = {"sum": sorted(WEIGHT_METRICS)}
        result["hybrid/preload"] = {"sum": [independent, copy.deepcopy(result["hybrid/resident"])]}
        result["hybrid/preload@dispatch_model"] = {"sum": [independent, copy.deepcopy(result["hybrid/resident@dispatch_model"])]}
    return result


def write_english_report(out, report, suite, rows):
    primary = [x for x in report["results"] if "@" not in x["strategy"]]
    errors = [x["absolute_relative_error"] for x in primary if x["observed_ns"] is not None]
    lines = ["# Repeated completed-work timing experiment", "",
        f"Batch repetitions: {suite['batch_repetitions']}. Every raw timing sample is a batch total; the analysis derives its per-invocation mean.", "",
        "Profile component samples are frozen before fresh held-out processes. Complete strategy times are used only for validation. All samples and failed processes remain recorded.", "",
        f"Native processes completed: {sum(r['status'] == 'measured' for r in rows)}/{len(rows)}.",
        f"Primary predictions accepted under this repeated-execution contract: {sum(x['calibrated'] for x in primary)}/{len(primary)}.",
        f"Primary median relative error: {statistics.median(errors):.2%}." if errors else "No complete primary prediction.", "",
        "These results do not replace historical isolated-invocation results, establish CUDA kernel overlap, validate full LLM inference, or establish GPU capacity. GPU private workspace remains unknown.", "",
        "| Case | Strategy | Predicted us/invocation | Observed us/invocation | Error | Accepted | Failed gates |",
        "|---|---|---:|---:|---:|---|---|"]
    for r in primary:
        if r["observed_ns"] is None:
            lines.append(f"| {r['case_id']} | {r['strategy']} | | | | false | {r['unknown_reason']} |")
        else:
            failures = ", ".join(k for k, passed in r["gates"].items() if not passed)
            lines.append(f"| {r['case_id']} | {r['strategy']} | {r['predicted_ns']/1000:.3f} | {r['observed_ns']/1000:.3f} | {r['absolute_relative_error']:.2%} | {r['calibrated']} | {failures} |")
    lines += ["", "## Interpretation and limits", "",
        "The repeated operation reuses the same generated tensors and completes each invocation before proceeding. Batch averaging changes the statistical unit and can change cache/power behavior. A lower CV of batch means does not prove that individual requests have low latency variability.", "",
        "Independent gate/up/down H2D probes permit an additive transfer hypothesis without splitting a previously measured aggregate. Their sum is still tested against the actual complete preload path. CPU/GPU branch overlap remains an ideal max hypothesis with a separately declared empty-worker alternative.", "",
        "No timings are exported as accepted physical capacity costs. No user applications, power policies or clocks are changed. No samples are removed, and no gate is relaxed after observing held-out data.", "",
        "The report retains alternative dispatch-model predictions, process statistics, raw batch totals, normalization rule, source/binary/runtime hashes, numerical checks and workspace status in machine-readable artifacts."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(args):
    suite = validate_suite(base.document(args.suite))
    if args.timeout_s < 1 or not args.binary.is_file() or any(not p.is_dir() for p in args.runtime_dir):
        raise ValueError("explicit existing binary/runtime paths and positive timeout required")
    out = args.output.resolve()
    if out.exists():
        raise ValueError("output must be a new directory")
    plan = [dict(job, command=base.command(args.binary.resolve(), job["case"], suite["protocol"], job["seed"]))
            for split in ("profile", "heldout") for job in base.jobs(suite, split)]
    env, flags = base.child_context(args.runtime_dir)
    dispatch_cpu = base.dispatch_cpu_id(flags, suite["protocol"]["cpu_id"])
    env["GGML_CALIBRATION_BATCH_REPETITIONS"] = str(suite["batch_repetitions"])
    flags["GGML_CALIBRATION_BATCH_REPETITIONS"] = env["GGML_CALIBRATION_BATCH_REPETITIONS"]
    if args.dry_run:
        print(json.dumps({"suite": suite, "plan": plan, "child_flags": flags}, indent=2))
        return 0
    native = PROFILING / "native"
    expected_modules = {a: PROFILING / "execution_calibration/analysis.py",
        base: PROFILING / "execution_calibration/run_dual.py", run_measurements: PROFILING / "run_measurements.py"}
    for module, expected in expected_modules.items():
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError("imported module is not the bound checkout source: " + module.__name__)
    sources = [Path(__file__), PROFILING / "execution_calibration/analysis.py",
        PROFILING / "execution_calibration/run_dual.py", PROFILING / "run_measurements.py",
        native / "execution_transforms_dual/compute_context.cpp", native / "matched_mechanisms/compute.cpp",
        native / "execution_transforms_batch/generate_probe.py", native / "execution_transforms_batch/CMakeLists.txt",
        native / "execution_transforms_batch/record_build.py",
        *args.binary.parent.glob("*.cpp"), *args.binary.parent.glob("*manifest*.json")]
    sources = list(dict.fromkeys(p.resolve() for p in sources))
    build_record = args.binary.parent / "compute-batch-build.json"
    source_manifest = args.binary.parent / "source-manifest.json"
    if not build_record.is_file() or not source_manifest.is_file():
        raise ValueError("native build record and source manifest are required")
    build = base.document(build_record)
    bound = base.document(source_manifest)
    if build.get("binary_sha256") != a.sha(args.binary) or build.get("source_manifest") != bound:
        raise ValueError("native build does not bind current executable/source manifest")
    checks = {"generator_sha256": native / "execution_transforms_batch/generate_probe.py",
        "context_sha256": native / "execution_transforms_dual/compute_context.cpp",
        "helper_sha256": native / "matched_mechanisms/compute.cpp"}
    for key, path in checks.items():
        if bound.get(key) != a.sha(path):
            raise ValueError("native build source mismatch: " + key)
    generated = list(args.binary.parent.glob("*.cpp"))
    if len(generated) != 1 or bound.get("generated_sha256") != a.sha(generated[0]):
        raise ValueError("generated probe source is missing, ambiguous or changed")
    sources.append(build_record.resolve())
    runtime = [p for d in args.runtime_dir for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".dll", ".so", ".dylib")]
    before = base.inventory([args.suite.resolve(), args.binary.resolve(), *sources, *runtime])
    lock = PROFILING / ".measurement.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        out.mkdir(parents=True)
        (out / "raw").mkdir()
        (out / "source").mkdir()
        snapshots = {}
        for i, source in enumerate(sources):
            dest = out / "source" / (f"{i:02d}-" + source.name)
            dest.write_bytes(source.read_bytes())
            snapshots[str(source)] = {"path": str(dest.relative_to(out)), "sha256": a.sha(dest)}
        a.dump(out / "suite.json", suite)
        a.dump(out / "plan.json", plan)
        manifest = {"schema": "execution-transform-batch-manifest-0.1", "started_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_before": before, "source_snapshots": snapshots, "observed_child_flags": flags,
            "environment_before": environment(), "timing_unit": "completed_work_batch_total",
            "derived_metric_unit": "rounded_mean_completed_invocation_within_batch_ns",
            "normalization": "round(batch_total_ns / batch_repetitions); raw totals retained",
            "sample_filtering": "none", "user_applications_changed": False, "gpu_capacity_proved": False}
        a.dump(out / "manifest.json", manifest)
        declared = {c["id"]: recipes(c) for c in suite["cases"]}
        aliases = {c["id"]: base.observed_aliases(c) for c in suite["cases"]}
        rows = []
        for split in ("profile", "heldout"):
            if base.inventory(before) != before:
                raise ValueError("source/binary/runtime changed")
            for job in base.jobs(suite, split):
                row = copy.deepcopy(job)
                command = base.command(args.binary.resolve(), row["case"], suite["protocol"], row["seed"])
                row["command"] = command
                row["raw_path"] = str(out / "raw" / (row["id"] + ".json"))
                if split == "heldout":
                    if a.sha(out / "prediction-model.json") != manifest["model_frozen_before_heldout_sha256"]:
                        raise ValueError("frozen model changed")
                    row["frozen_model_sha256_at_start"] = manifest["model_frozen_before_heldout_sha256"]
                record = {"command": command, "observed_child_flags": flags, "started_utc": datetime.now(timezone.utc).isoformat()}
                if split == "heldout":
                    record["frozen_model_sha256_at_start"] = row["frozen_model_sha256_at_start"]
                start = time.perf_counter_ns()
                try:
                    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    record["child_pid"] = proc.pid
                    try:
                        stdout, stderr = proc.communicate(timeout=args.timeout_s)
                    except subprocess.TimeoutExpired:
                        proc.kill(); stdout, stderr = proc.communicate()
                        record["timed_out"] = True
                    record.update(returncode=proc.returncode, stdout=stdout.decode("utf-8", errors="replace"), stderr=stderr.decode("utf-8", errors="replace"))
                    if record.get("timed_out") or proc.returncode:
                        raise ValueError("native timeout/nonzero exit: " + str(proc.returncode))
                    payload = json.loads(record["stdout"])
                    normalized = normalized_payload(payload, row, suite, dispatch_cpu)
                    row.update(status="measured", metrics=normalized, raw_batch_metrics=payload["batch_wall_ns"],
                        raw_batch_statistics={k: a.stats(v) for k, v in payload["batch_wall_ns"].items()},
                        checks=payload["checks"], workspaces=payload["workspaces"], ports=payload["ports"], boundary=payload["boundary"],
                        batch_repetitions=suite["batch_repetitions"])
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    row.update(status="failed", error=str(exc)); record["validation_error"] = str(exc)
                record.update(completed_utc=datetime.now(timezone.utc).isoformat(), process_wall_ns=time.perf_counter_ns() - start)
                a.dump(row["raw_path"], record)
                row.update(raw_sha256=a.sha(row["raw_path"]), child_pid=record.get("child_pid"))
                rows.append(row)
                a.dump(out / "measurements.json", rows)
                print(row["id"] + ": " + row["status"], flush=True)
            if split == "profile":
                model = a.freeze_model(rows, declared, aliases)
                a.dump(out / "prediction-model.json", model)
                manifest.update(model_frozen_before_heldout_sha256=a.sha(out / "prediction-model.json"), model_frozen_utc=datetime.now(timezone.utc).isoformat())
                a.dump(out / "manifest.json", manifest)
        if a.freeze_model(rows, declared, aliases) != model:
            raise ValueError("profile-only model not reproducible")
        groups = {c["id"]: {name: ("resident" if name.split("@", 1)[0].endswith("resident") else "preload")
            + (";dispatch_model" if "@" in name else ";ideal_model") for name in recipes(c)} for c in suite["cases"]}
        report = a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
            scope="F32 compact FFN and post-RoPE Attention; repeated completed-work batch means, explicit CPU/GPU worker contexts.",
            frozen_before_heldout=True, comparison_groups=groups)
        report.update(batch_repetitions=suite["batch_repetitions"], raw_timing_unit="completed_work_batch_total",
            gpu_capacity_proved=False, older_isolated_protocol_replaced=False)
        report["unknown_or_uncovered"].extend([
            "Batch means cannot certify isolated request jitter or full token generation.",
            "GPU private workspace unknown; no physical capacity acceptance.",
            "Dense norm/residual, projection and MoE routing overhead remain outside compact FFN scope.",
            "No CUDA device-event/kernel overlap evidence; completed host wall time only.",
            "Repeated warm tensors do not model arbitrary dynamic token-state changes."])
        a.write_outputs(out / "analysis", report, model, rows)
        write_english_report(out / "analysis", report, suite, rows)
        after = base.inventory(before)
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), inventory_after=after,
            assets_unchanged=after == before, report_sha256=a.sha(out / "analysis/report.json"))
        a.dump(out / "manifest.json", manifest)
        if before != after:
            raise ValueError("source/binary/runtime changed during experiment")
        return 0
    finally:
        if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, action="append", type=Path)
    parser.add_argument("--timeout-s", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
