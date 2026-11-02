"""Reusable F32 measurement / heldout calibration, with no synthetic fallback.

All timings retain native samples and source bindings. The component-sum model
is frozen before heldout processes. GPU workspace remains unknown. CE+SM probes
exercise a concrete UVA copy kernel, not a general contention or bandwidth law.
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
import statistics
import struct
import subprocess
import sys

PROFILING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILING))
from run_measurements import environment

VERSION = "matched-mechanisms-suite-0.1"
CASE_FIELDS = {"id", "family", "backend", "hidden", "intermediate", "tokens", "history",
               "heads", "kv_heads", "head_dim", "dtype"}
PROTO_FIELDS = {"threads", "cpu_id", "device", "warmup", "samples", "min_warmup_ms", "cuda_graphs"}
TRANSFER_FIELDS = {"id", "mode", "bytes", "device", "blocks", "threads_per_block", "warmup", "samples"}
ENV_KEYS = {"GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "OMP_PROC_BIND"}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def document(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key: " + k)
            result[k] = v
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def integer(value, label, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{label} requires an explicit integer in {lo}..{hi}")


def validate_suite(original):
    suite = copy.deepcopy(original)
    fields = {"format_version", "cases", "compute_protocol", "transfers", "profile_seeds",
              "heldout_seeds", "repeats", "quality", "bootstrap"}
    if not isinstance(suite, dict) or set(suite) != fields or suite["format_version"] != VERSION:
        raise ValueError("suite must contain exactly the documented explicit fields")
    integer(suite["repeats"], "repeats", 2, 8)
    seen_seeds = set()
    for part in ("profile_seeds", "heldout_seeds"):
        seeds = suite[part]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(part + " requires explicit nonempty seeds")
        for seed in seeds:
            integer(seed, part, 0, 1000000)
            if seed in seen_seeds:
                raise ValueError("profile/heldout seeds must be unique and disjoint")
            seen_seeds.add(seed)
    p = suite["compute_protocol"]
    if not isinstance(p, dict) or set(p) != PROTO_FIELDS:
        raise ValueError("explicit compute protocol required")
    for k, lo, hi in (("threads", 1, 1), ("cpu_id", 0, 63), ("device", 0, 64),
                      ("warmup", 0, 10000), ("samples", 3, 10000), ("min_warmup_ms", 0, 60000)):
        integer(p[k], k, lo, hi)
    if p["cuda_graphs"] != "disabled":
        raise ValueError("only explicit disabled CUDA graphs supported")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("explicit nonempty cases required")
    ids = set()
    for c in suite["cases"]:
        if not isinstance(c, dict) or set(c) != CASE_FIELDS:
            raise ValueError("case fields must be exact")
        if not isinstance(c["id"], str) or not c["id"] or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for x in c["id"]):
            raise ValueError("case id must be a safe nonempty identifier")
        if c["id"] in ids:
            raise ValueError("duplicate case id")
        ids.add(c["id"])
        if c["family"] not in ("ffn", "attention") or c["backend"] not in ("cpu", "gpu", "hybrid") or c["dtype"] != "f32":
            raise ValueError("only explicit supported F32 family/backend accepted")
        for k, lo, hi in (("hidden", 1, 1024), ("intermediate", 0, 4096), ("tokens", 1, 128),
                          ("history", 0, 4096), ("heads", 0, 1024), ("kv_heads", 0, 1024), ("head_dim", 0, 1024)):
            integer(c[k], k, lo, hi)
        if c["family"] == "ffn":
            if not c["intermediate"] or any(c[k] for k in ("history", "heads", "kv_heads", "head_dim")):
                raise ValueError("FFN requires separate gate/up/down and no attention geometry")
        elif (c["backend"] == "hybrid" or c["intermediate"] or not c["heads"] or not c["kv_heads"]
              or not c["head_dim"] or c["heads"] % c["kv_heads"]
              or c["heads"] * c["head_dim"] != c["hidden"]):
            raise ValueError("unsupported attention geometry / hybrid")
    if not isinstance(suite["transfers"], list):
        raise ValueError("transfers must be an explicit list")
    modes = {"ce_h2d", "ce_d2h", "sm_h2d", "sm_d2h", "ce_sm_duplex", "empty_kernel", "empty_events"}
    for t in suite["transfers"]:
        if not isinstance(t, dict) or set(t) != TRANSFER_FIELDS or t["mode"] not in modes:
            raise ValueError("explicit supported transfer fields required")
        if not isinstance(t["id"], str) or not t["id"] or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for x in t["id"]) or t["id"] in ids:
            raise ValueError("unique safe transfer id required")
        ids.add(t["id"])
        for k, lo, hi in (("bytes", 1, 67108864), ("device", 0, 64), ("blocks", 1, 1024),
                          ("threads_per_block", 32, 1024), ("warmup", 0, 10000), ("samples", 3, 10000)):
            integer(t[k], k, lo, hi)
    q = suite["quality"]
    if not isinstance(q, dict) or set(q) != {"max_within_process_cv", "max_between_process_cv", "max_heldout_relative_error"}:
        raise ValueError("all quality thresholds must be predeclared")
    for k, v in q.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1:
            raise ValueError(k + " requires an explicit finite threshold in (0,1]")
    b = suite["bootstrap"]
    if not isinstance(b, dict) or set(b) != {"iterations", "seed"}:
        raise ValueError("bootstrap protocol must be explicit")
    integer(b["iterations"], "bootstrap iterations", 100, 10000)
    integer(b["seed"], "bootstrap seed", 0, 1000000)
    return suite


def stats(samples):
    if not isinstance(samples, list) or len(samples) < 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in samples):
        raise ValueError("native wall samples require >=3 positive integer nanoseconds")
    mean = statistics.mean(samples)
    return {"median_ns": statistics.median(samples), "mean_ns": mean,
            "cv": statistics.pstdev(samples) / mean, "min_ns": min(samples), "max_ns": max(samples),
            "sample_count": len(samples)}


def compute_command(binary, case, protocol, seed):
    c = [str(binary)]
    keys = {"family": "family", "backend": "backend", "hidden": "hidden", "intermediate": "intermediate",
            "tokens": "tokens", "history": "history", "heads": "heads", "kv_heads": "kv-heads",
            "head_dim": "head-dim", "dtype": "dtype"}
    for key, flag in keys.items():
        c += ["--" + flag, str(case[key])]
    for key in ("threads", "cpu_id", "device", "warmup", "samples", "min_warmup_ms", "cuda_graphs"):
        c += ["--" + key.replace("_", "-"), str(protocol[key])]
    return c + ["--seed", str(seed)]


def transfer_command(binary, case):
    c = [str(binary)]
    for key in ("mode", "bytes", "device", "blocks", "threads_per_block", "warmup", "samples"):
        c += ["--" + key.replace("_", "-"), str(case[key])]
    return c


def _float_bytes(values):
    if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("source values must be finite explicit nonempty arrays")
    return struct.pack("<" + "f" * len(values), *values)


def fixture_hashes(payload):
    result = {}
    for name, values in payload["source_values"].items():
        raw = _float_bytes(values)
        result[name] = {"sha256_f32_le": hashlib.sha256(raw).hexdigest(), "storage_bytes": len(raw)}
    return result


def validate_native(row, payload, suite):
    c, p = row["case"], suite["compute_protocol"]
    if row["kind"] == "compute":
        if payload.get("schema") != "matched-compute-native-0.1":
            raise ValueError("unexpected native compute schema")
        for key in CASE_FIELDS - {"id"}:
            if payload.get(key) != c[key]:
                raise ValueError("native case binding mismatch: " + key)
        for key in ("threads", "cpu_id", "samples", "cuda_graphs"):
            if payload.get(key) != p[key]:
                raise ValueError("native protocol binding mismatch: " + key)
        if payload.get("seed") != row["seed"] or payload.get("all_checks_passed") is not True:
            raise ValueError("native seed or numerical check mismatch")
        expected = set()
        stages = ("gate", "up", "activation", "down", "joint", "serial") if c["family"] == "ffn" else ("attention_core", "attention_output", "joint", "serial")
        devices = ("cpu", "gpu") if c["backend"] == "hybrid" else (c["backend"],)
        for device in devices:
            expected |= {device + "/" + stage for stage in stages}
        if c["backend"] == "hybrid":
            expected |= {"hybrid/gpu_weights", "hybrid/cpu_rows", "hybrid/split_cpu_to_gpu", "hybrid/d2h/input"}
            expected |= {"hybrid/h2d/" + x for x in ("input", "activated", "output", "gate_weight", "up_weight", "down_weight")}
        if set(payload.get("wall_ns", {})) != expected:
            raise ValueError("native metric coverage mismatch")
        for metric, values in payload["wall_ns"].items():
            stats(values)
            if len(values) != p["samples"]:
                raise ValueError("native sample count mismatch")
        if not payload.get("checks") or any(check.get("passed") is not True for check in payload["checks"].values()):
            raise ValueError("missing or failed numerical checks")
        hashes = fixture_hashes(payload)
        for name, port in payload["ports"].items():
            if port["bytes"] != hashes[name]["storage_bytes"] or math.prod(port["ne"]) * 4 != port["bytes"]:
                raise ValueError("native port layout/size mismatch")
        for name, w in payload["workspaces"].items():
            if name.startswith("gpu/") and (w["workspace_bytes"] is not None or w["source"] != "unknown_private_cuda_scratch"):
                raise ValueError("unknown GPU workspace must remain unknown")
            if name.startswith("cpu/") and (isinstance(w["workspace_bytes"], bool) or not isinstance(w["workspace_bytes"], int) or w["workspace_bytes"] < 0):
                raise ValueError("CPU workspace must be measured, not guessed")
    else:
        if payload.get("schema") != "matched-transfer-native-0.1":
            raise ValueError("unexpected native transfer schema")
        for src, dst in (("mode", "mode"), ("bytes", "bytes"), ("samples", "samples"),
                         ("device", "device_index"), ("blocks", "kernel_blocks"), ("threads_per_block", "kernel_threads")):
            if payload.get(dst) != c[src]:
                raise ValueError("native transfer binding mismatch: " + src)
        if payload.get("validation") != "all_bytes_exact" or payload.get("sm_api") != "actual_cuda_kernel_copy_mapped_host_memory":
            raise ValueError("transfer semantics or numerical validation mismatch")
        if len(payload.get("wall_ns", [])) != c["samples"]:
            raise ValueError("transfer sample count mismatch")
        stats(payload["wall_ns"])


def components(metric, family):
    stages = ("gate", "up", "activation", "down") if family == "ffn" else ("attention_core", "attention_output")
    if metric in ("cpu/serial", "gpu/serial"):
        device = metric.split("/")[0]
        return [device + "/" + x for x in stages]
    if metric == "hybrid/gpu_weights":
        return ["hybrid/h2d/" + x for x in ("gate_weight", "up_weight", "down_weight")] + ["gpu/" + x for x in stages]
    if metric == "hybrid/cpu_rows":
        return ["hybrid/d2h/input", *["cpu/" + x for x in stages], "hybrid/h2d/output"]
    if metric == "hybrid/split_cpu_to_gpu":
        return ["cpu/gate", "cpu/up", "cpu/activation", "hybrid/h2d/activated", "gpu/down"]
    return []


def fit(rows):
    model = {"schema": "matched-mechanisms-profile-0.1", "trained_on": "profile_only",
             "interpolation": "none", "control_overhead_separately_identified": False,
             "gpu_workspace_known": False, "entries": []}
    ids = sorted({row["case"]["id"] for row in rows if row["split"] == "profile" and row["status"] == "measured"})
    for cid in ids:
        selected = [r for r in rows if r["case"]["id"] == cid and r["split"] == "profile" and r["status"] == "measured"]
        metrics = selected[0]["metrics"]
        medians = {m: statistics.median([v for r in selected for v in r["metrics"][m]]) for m in metrics}
        for metric, median in medians.items():
            parts = components(metric, selected[0]["case"].get("family"))
            model["entries"].append({"case_id": cid, "case": selected[0]["case"], "metric": metric,
                "median_ns": median, "profile_rows": [r["id"] for r in selected],
                "ports": selected[0].get("ports"),
                "workspace": selected[0].get("workspaces", {}).get(metric),
                "profile_fixture_hashes": {r["id"]: r.get("fixture_hashes") for r in selected},
                "component_metrics": parts, "component_sum_prediction_ns": sum(medians[p] for p in parts) if parts else None,
                "serial_minus_component_sum_ns": median - sum(medians[p] for p in parts) if parts else None})
    return model


def percentile(sorted_values, fraction):
    index = (len(sorted_values) - 1) * fraction
    lo = int(math.floor(index)); hi = int(math.ceil(index))
    return sorted_values[lo] * (hi - index) + sorted_values[hi] * (index - lo) if hi != lo else sorted_values[lo]


def median_interval(groups, boot):
    rng = random.Random(boot["seed"])
    values = []
    for _ in range(boot["iterations"]):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        draws = [g[rng.randrange(len(g))] for g in sampled for _ in g]
        values.append(statistics.median(draws))
    values.sort()
    return [percentile(values, .025), percentile(values, .975)]


def evaluate(rows, model, suite):
    report = {"schema": "matched-mechanisms-report-0.1", "quality": suite["quality"], "results": [],
              "failed_native_rows": [r["id"] for r in rows if r["status"] != "measured"],
              "numeric_execution_valid": all(r["status"] == "measured" for r in rows),
              "general_link_arbitration_calibrated": False, "gpu_capacity_proved": False,
              "scope": "exact declared shape/binary/device/thread/variant only; no model or paper reproduction"}
    for entry in model["entries"]:
        cid, metric = entry["case_id"], entry["metric"]
        selected = [r for r in rows if r["case"]["id"] == cid and r["status"] == "measured"]
        heldout = [r for r in selected if r["split"] == "heldout"]
        if not heldout:
            report["results"].append({"case_id": cid, "metric": metric, "calibrated": False, "reason": "missing heldout"})
            continue
        groups = [r["metrics"][metric] for r in heldout]
        observed = statistics.median([v for group in groups for v in group])
        predicted = entry["median_ns"]
        process = [{"row_id": r["id"], "split": r["split"], **stats(r["metrics"][metric])} for r in selected]
        by_split = {s: [p["median_ns"] for p in process if p["split"] == s] for s in ("profile", "heldout")}
        between = {s: statistics.pstdev(v) / statistics.mean(v) if len(v) > 1 else None for s, v in by_split.items()}
        error = abs(predicted - observed) / observed
        gates = {"within_process": all(p["cv"] <= suite["quality"]["max_within_process_cv"] for p in process),
                 "between_process": all(v is not None and v <= suite["quality"]["max_between_process_cv"] for v in between.values()),
                 "heldout_error": error <= suite["quality"]["max_heldout_relative_error"]}
        result = {"case_id": cid, "metric": metric, "predicted_ns": predicted, "observed_ns": observed,
                  "heldout_relative_error": error, "between_process_cv": between, "processes": process,
                  "heldout_median_95pct_interval_ns": median_interval(groups, suite["bootstrap"]),
                  "gates": gates, "calibrated": all(gates.values())}
        if entry["component_sum_prediction_ns"] is not None:
            pred = entry["component_sum_prediction_ns"]
            result["component_sum_prediction_ns"] = pred
            result["component_sum_relative_error"] = abs(pred - observed) / observed
            result["component_sum_calibrated"] = (result["calibrated"] and result["component_sum_relative_error"] <= suite["quality"]["max_heldout_relative_error"])
        report["results"].append(result)
    report["all_declared_metrics_calibrated"] = (not report["failed_native_rows"] and bool(report["results"]) and all(r["calibrated"] for r in report["results"]))
    return report


def inventory(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(set(map(str, paths)))}


def child_context(runtime_dirs):
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(p) for p in runtime_dirs] + [env.get("PATH", "")])
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(p) for p in runtime_dirs] + [env.get("LD_LIBRARY_PATH", "")])
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    keys = ENV_KEYS | {k for k in env if k.startswith(("GGML_", "CUDA_", "OMP_", "CUBLAS_"))}
    return env, {k: env.get(k) for k in sorted(keys)}


def run(args):
    suite_path, out = args.suite.resolve(), args.output.resolve()
    suite = validate_suite(document(suite_path))
    runtime_dirs = [p.resolve() for p in args.runtime_dir]
    if suite["transfers"] and args.transfer_binary is None:
        raise ValueError("transfer binary is required by the suite")
    binaries = [args.compute_binary.resolve()] + ([args.transfer_binary.resolve()] if suite["transfers"] else [])
    if any(not p.is_file() for p in binaries) or any(not p.is_dir() for p in runtime_dirs):
        raise ValueError("explicit binary/runtime paths must exist")
    # Protect all provided runtime libraries and all old native probe source/binaries.
    old_files = [p for p in (PROFILING / "native").rglob("*") if p.is_file() and "matched_mechanisms" not in p.parts and p.suffix in {".cpp", ".h", ".exe"}]
    source_files = list(Path(__file__).parent.glob("*.py")) + list((PROFILING / "native" / "matched_mechanisms").glob("*.cpp")) + list((PROFILING / "native" / "matched_mechanisms").glob("*.cu"))
    runtime_files = [p for d in runtime_dirs for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".dll", ".so", ".dylib"}]
    before = inventory([suite_path, *binaries, *source_files, *runtime_files, *old_files])
    if out.exists():
        raise ValueError("output must be a new directory; old evidence is never overwritten")
    out.mkdir(parents=True); (out / "raw").mkdir(); (out / "fixtures").mkdir(); (out / "source").mkdir()
    for p in source_files:
        (out / "source" / p.name).write_bytes(p.read_bytes())
    dump(out / "suite.json", suite)
    env, flags = child_context(runtime_dirs)
    manifest = {"schema": "matched-mechanisms-manifest-0.1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment_before": environment(), "observed_child_flags": flags, "inventory_before": before,
                "threads_affinity": "Windows explicit single CPU-id affinity; other systems affinity not applied by native",
                "user_applications_changed": False, "sample_filtering": "none", "timing_scope": "complete work; grouped warm native calls"}
    dump(out / "manifest.json", manifest)
    rows = []
    lock = PROFILING / ".measurement.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        for split in ("profile", "heldout"):
            for c in suite["cases"]:
                for seed in suite[split + "_seeds"]:
                    for repeat in range(suite["repeats"]):
                        row = {"id": f"{split}-{c['id']}-s{seed}-r{repeat}", "kind": "compute", "case": c,
                               "split": split, "seed": seed, "repeat": repeat}
                        row["command"] = compute_command(args.compute_binary.resolve(), c, suite["compute_protocol"], seed)
                        rows.append(_execute(row, args, out, env, flags, suite))
            for c in suite["transfers"]:
                for repeat in range(suite["repeats"]):
                    row = {"id": f"{split}-{c['id']}-r{repeat}", "kind": "transfer", "case": c,
                           "split": split, "seed": None, "repeat": repeat,
                           "command": transfer_command(args.transfer_binary.resolve(), c)}
                    rows.append(_execute(row, args, out, env, flags, suite))
            if split == "profile":
                model = fit(rows)
                dump(out / "prediction-model.json", model)
                manifest["model_frozen_before_heldout_sha256"] = sha(out / "prediction-model.json")
                dump(out / "manifest.json", manifest)
        if sha(out / "prediction-model.json") != manifest["model_frozen_before_heldout_sha256"]:
            raise ValueError("prediction model changed during heldout")
        report = evaluate(rows, model, suite)
        dump(out / "measurements.json", rows); dump(out / "report.json", report)
        after = inventory(before)
        manifest["inventory_after"] = after
        manifest["assets_unchanged"] = after == before
        manifest["environment_after"] = environment()
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        dump(out / "manifest.json", manifest)
        if after != before:
            raise ValueError("source/binary/runtime/frozen assets changed during execution")
        print(canonical({"output": str(out), "native_rows": len(rows), "numeric_execution_valid": report["numeric_execution_valid"],
                         "all_declared_metrics_calibrated": report["all_declared_metrics_calibrated"],
                         "calibrated_metrics": sum(r["calibrated"] for r in report["results"]), "total_metrics": len(report["results"])}))
        return 0 if report["numeric_execution_valid"] else 2
    finally:
        lock.unlink(missing_ok=True)


def _execute(row, args, out, env, flags, suite):
    _, current_flags = child_context(args.runtime_dir)
    if current_flags != flags:
        raise ValueError("inherited child flags changed during the experiment")
    try:
        proc = subprocess.run(row["command"], env=env, capture_output=True, text=True, timeout=args.timeout_s)
        record = {"command": row["command"], "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                  "observed_child_flags": flags}
        dump(out / "raw" / (row["id"] + ".json"), record)
        row["raw_sha256"] = sha(out / "raw" / (row["id"] + ".json"))
        if proc.returncode:
            raise ValueError("native failed with exit code " + str(proc.returncode))
        payload = json.loads(proc.stdout)
        validate_native(row, payload, suite)
        row["status"] = "measured"
        if row["kind"] == "compute":
            row["metrics"] = payload["wall_ns"]
            hashes = fixture_hashes(payload)
            row["fixture_hashes"] = hashes
            row["ports"] = payload["ports"]; row["workspaces"] = payload["workspaces"]; row["checks"] = payload["checks"]
            # Concrete F32 little-endian values give a reproducible physical weight identity.
            for name, values in payload["source_values"].items():
                path = out / "fixtures" / (row["id"] + "-" + name + ".f32le")
                path.write_bytes(_float_bytes(values))
        else:
            row["metrics"] = {"wall": payload["wall_ns"]}
            row["device_intervals"] = {"ce_interval_ns": payload["ce_interval_ns"], "sm_interval_ns": payload["sm_interval_ns"]}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        row["status"] = "failed"; row["error"] = str(exc)
        if not (out / "raw" / (row["id"] + ".json")).exists():
            dump(out / "raw" / (row["id"] + ".json"), {"command": row["command"], "error": str(exc)})
            row["raw_sha256"] = sha(out / "raw" / (row["id"] + ".json"))
    dump(out / "measurements.partial.json", row)
    print(row["id"] + ": " + row["status"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--compute-binary", type=Path, required=True)
    parser.add_argument("--transfer-binary", type=Path)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_s < 1:
        parser.error("timeout-s must be explicit positive")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print("matched experiment: " + str(exc), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
