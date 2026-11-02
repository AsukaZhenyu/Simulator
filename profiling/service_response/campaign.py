"""Source-bound collection for the common stateful service-response model.

This module schedules owned native actors; it contains no service equation.
It reuses the numerical/lifetime/affinity checks of state_response.collect,
but stores a small service-model freeze instead of duplicating all normalized
records inside a frozen model. Every raw observation remains on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src", ROOT / "modeling", ROOT / "mapping", ROOT / "profiling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from profiling.state_response import collect as common
from profiling.state_response.schedule import SCHEMA, state, validate_schedule
from profiling.state_calibration import run


def declared_protocol(*, settle_ms=1000, max_step_ns=250000):
    """Explicit experimental structure; these choices are not measurements."""
    return dict(
        schema="service-response-protocol-1",
        model_name="signed_inclusive_response_reservoir_v1",
        competitors=[dict(name="cpu", request_key="cpu_replay", actor_role="cpu-worker"),
                     dict(name="gpu", request_key="gpu_replay", actor_role="gpu-worker")],
        capacity=dict(numerator=1, denominator=1),
        activity_weight=dict(numerator=1, denominator=1),
        backlog_weight=dict(numerator=1, denominator=1),
        coefficient_domain="signed_inclusive_response",
        profile_settle_ns=settle_ms * 1000000,
        minimum_independent_runs=3, minimum_cell_samples_per_process=10,
        within_process_cv_max=.15, between_process_cv_max=.20,
        relative_error_max=.20, max_step_ns=max_step_ns, max_points=100000,
        initial_work={name: dict(numerator=0, denominator=1) for name in ("cpu", "gpu")},
        initial_history_evidence=[
            "controlled persistent workers complete warmup before READY; no scheduled target replay before GO; "
            "normalized reservoir initial_work=0 is a declared model assumption, not physical quiescence"],
        control_overhead_policy="baseline_fixed_ns_unqualified",
        proxy_mapping="arrival=drain+capacity/fill_median; drain=capacity/off_tail_median",
        combined_state_role="validation_only",
        model_selection="predeclared_before_profile_and_heldout", heldout_refit=False,
    )


def recovery_schedule(*, split, seed, settle_ms):
    """Fresh orders, with isolated calibration and independently varied pulses."""
    if split not in {"profile", "heldout"}:
        raise ValueError("unknown split")
    rng = random.Random(seed)
    pattern = [(0, 0, 1400 if split == "profile" else 419)]
    loads = [(1, 0), (0, 1), (1, 1)]
    if split == "profile":
        short_order = list(loads)
        rng.shuffle(short_order)
        for cpu, gpu in short_order:
            pattern.extend(((cpu, gpu, 100), (0, 0, 500)))
    for cycle in range(2):
        order = list(loads)
        rng.shuffle(order)
        for cpu, gpu in order:
            hold = 2000 if split == "profile" else rng.choice((47, 83, 149, 227, 419))
            recovery = 1400 if split == "profile" else rng.choice((61, 193, 419))
            pattern.extend(((cpu, gpu, hold), (0, 0, recovery)))
    if split == "heldout":
        pattern.append((0, 0, 1200))
    phases, elapsed = [], 0
    for index, (cpu, gpu, hold_ms) in enumerate(pattern):
        end = elapsed + hold_ms * 1000000
        calibration_begin = (min(end - 1, elapsed + settle_ms * 1000000)
                             if split == "profile" else elapsed)
        phases.append(dict(index=index, begin_ns=elapsed, end_ns=end,
            calibration_begin_ns=calibration_begin,
            state=state(cpu, gpu, 0, 0)))
        elapsed = end
    result = dict(schema=SCHEMA, seed=seed, state_semantics="offered_load_controlled",
                  memory_fixed_for_trajectory=True, duration_ns=elapsed, phases=phases)
    validate_schedule(result)
    return result


def snapshot_sources(out, manifest):
    """Retain exact mutable sources before any timed run; never rewrite history."""
    destination = Path(out) / "recorded-sources"
    destination.mkdir(exist_ok=False)
    entries = []
    for name, expected in manifest["sources"].items():
        path = Path(name).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if any(part in {"build", "results", "__pycache__"} or part.startswith("build-")
               for part in relative.parts[:-1]):
            continue
        if run.sha(path) != expected:
            raise ValueError("source changed before snapshot: " + name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        entries.append(dict(original_path=str(path), archive_path=str(target), sha256=expected))
    run.save(destination / "archive.json", dict(schema="service-response-source-snapshot-1",
        created_at_utc=run.utc(), manifest_sha256=run.sha(Path(out) / "manifest.json"), entries=entries))


def prepare(args):
    if args.settle_ms >= 1400 or args.settle_ms < 0:
        raise ValueError("settle-ms must be in [0,1400)")
    if type(args.max_step_ns) is not int or not 0 < args.max_step_ns <= 1000000000:
        raise ValueError("max-step-ns must be an explicit positive interval <=1s")
    setup = argparse.Namespace(out=args.out, binary=args.binary, runtime_dir=args.runtime_dir,
        cuda_runtime=args.cuda_runtime, seed=args.seed, repeats=3, memory_bytes=64 << 20,
        memory_index=0, profile_hold_ms=2000, profile_state_holds_ms=None,
        settle_ms=args.settle_ms, heldout_cycles=2,
        main_affinity_mask=args.main_affinity_mask,
        main_thread_affinity_mask=args.main_thread_affinity_mask,
        cpu_worker_affinity_mask=args.cpu_worker_affinity_mask,
        gpu_worker_affinity_mask=args.gpu_worker_affinity_mask)
    common.prepare(setup)
    out = args.out.resolve()
    manifest = run.read(out / "manifest.json")
    (out / "manifest.json").rename(out / "collector-base-manifest.json")
    for job in manifest["jobs"]:
        job["schedule"] = recovery_schedule(split=job["split"],
            seed=job["schedule"]["seed"], settle_ms=args.settle_ms)
    protocol = declared_protocol(settle_ms=args.settle_ms, max_step_ns=args.max_step_ns)
    manifest["protocol"]["service_response"] = protocol
    manifest["protocol"]["collection_record_format"] = "raw_native_evidence_bindings_v1"
    manifest["protocol"]["short_profile_pulses"] = dict(
        on_ms=100, recovery_ms=500, plateau_prices_exclude_short_pulses=True,
        raw_transitions_and_worker_intervals_retained=True)
    manifest["protocol"]["primary_prediction_model"] = protocol["model_name"]
    manifest["protocol"]["comparison_models"] = [protocol["model_name"]]
    manifest["context"]["service_response_scope"] = (
        "exact serial inclusive completed actions; normalized residual-interference proxy; "
        "not actual GPU hardware queues or arbitrarily transferable physical service")
    for path in common.python_sources(Path(__file__).parent):
        manifest["sources"][str(path.resolve())] = run.sha(path)
    manifest["source_inventory_sha256"] = common.digest(manifest["sources"])
    manifest["context"]["software_sha256"] = manifest["source_inventory_sha256"]
    common.verify_sources(manifest)
    run.save(out / "manifest.json", manifest)
    snapshot_sources(out, manifest)
    return dict(status="prepared", jobs=len(manifest["jobs"]),
                service_model=protocol["model_name"], hardware_executed=False)


def collect(args):
    from .analysis import extract_job, make_barrier

    out = args.out.resolve()
    manifest = common.load_manifest(out)
    protocol = manifest["protocol"]["service_response"]
    if protocol["model_name"] != "signed_inclusive_response_reservoir_v1":
        raise ValueError("unsupported predeclared service model")
    rows_path = out / (args.split + "-processes.json")
    started_path = out / (args.split + "-collection-started.json")
    if rows_path.exists() or (out / (args.split + "-records.json")).exists() or started_path.exists():
        raise ValueError("split already attempted; preserve it and use a fresh campaign")
    if args.split == "profile":
        if (out / "heldout-barrier.json").exists():
            raise ValueError("cannot collect profile after heldout barrier")
    else:
        if args.frozen is None:
            raise ValueError("heldout requires an already frozen service model")
        from .calibration import validate_frozen
        frozen = validate_frozen(run.read(args.frozen), verify_sources=True)
        if frozen["protocol"] != protocol or frozen["context"] != manifest["context"]:
            raise ValueError("frozen and predeclared service protocol/context differ")
        barrier = make_barrier(frozen, created_at_utc=run.utc(),
            frozen_file_sha256=run.sha(args.frozen), manifest_sha256=run.sha(out / "manifest.json"))
        run.save(out / "heldout-barrier.json", barrier)
    run.save(started_path, dict(created_at_utc=run.utc(), manifest_sha256=run.sha(out / "manifest.json"),
                               all_attempts_retained=True))
    rows, records = [], []
    with run.measurement_lock():
        for job in (item for item in manifest["jobs"] if item["split"] == args.split):
            for attempt in range(manifest["protocol"]["maximum_pre_go_startup_retries"] + 1):
                common.verify_sources(manifest)
                print(json.dumps(dict(event="starting", job=job["id"], attempt=attempt)), flush=True)
                try:
                    row = common.execute_job(out, manifest, job, attempt=attempt, native_evidence_only=True)
                except Exception:
                    name = job["id"] + (f"-startup-retry{attempt}" if attempt else "")
                    path = out / "jobs" / name / "execution.json"
                    row = run.read(path) if path.exists() else dict(status="failed", attempt_id=name)
                    rows.append(row)
                    run.save(out / (args.split + f"-processes-checkpoint-{len(rows)}.json"), rows)
                    run.save(out / (args.split + f"-retained-failure-{len(rows)}.json"), row)
                    if attempt < manifest["protocol"]["maximum_pre_go_startup_retries"] and common.startup_retry_eligible(row):
                        print(json.dumps(dict(event="retained_pre_GO_failure", evidence=str(path))), flush=True)
                        continue
                    raise
                rows.append(row)
                run.save(out / (args.split + f"-processes-checkpoint-{len(rows)}.json"), rows)
                break
            record = extract_job(out / "manifest.json", row,
                competitors=protocol["competitors"], verify_sources=True)
            records.append(record)
            run.save(out / (args.split + f"-records-checkpoint-{len(records)}.json"), records)
            common.verify_sources(manifest)
            print(json.dumps(dict(event="completed", job=job["id"],
                                  process_id=row["process_id"])), flush=True)
    run.save(rows_path, rows)
    run.save(out / (args.split + "-records.json"), records)
    return dict(status="collected", split=args.split, independent_processes=len(records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--out", type=Path, required=True)
    setup.add_argument("--binary", type=Path, required=True)
    setup.add_argument("--runtime-dir", type=Path, action="append", required=True)
    setup.add_argument("--cuda-runtime", type=Path, required=True)
    setup.add_argument("--seed", type=int, required=True)
    setup.add_argument("--main-affinity-mask", type=int, required=True)
    setup.add_argument("--main-thread-affinity-mask", type=int, required=True)
    setup.add_argument("--cpu-worker-affinity-mask", type=int, required=True)
    setup.add_argument("--gpu-worker-affinity-mask", type=int, required=True)
    setup.add_argument("--settle-ms", type=int, default=1000)
    setup.add_argument("--max-step-ns", type=int, default=250000)
    record = sub.add_parser("collect")
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--split", choices=("profile", "heldout"), required=True)
    record.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args) if args.command == "prepare" else collect(args)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
