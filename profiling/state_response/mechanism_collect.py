"""Matched actor-presence, thread-count and post-load recovery diagnosis.

No duration is fitted or predicted. Diagnostic instrumentation is never a
formal cost source. Each process covers every declared predecessor state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src", ROOT / "modeling", ROOT / "mapping", ROOT / "profiling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from . import collect as common
from .schedule import SCHEMA as SCHEDULE_SCHEMA, state, validate_schedule
from profiling.state_calibration import run

DEFAULT_CONDITIONS = ("solo", "idle-actors", "recovery", "solo-one-thread", "solo-bare", "recovery-gpu-phases", "solo-gpu-phases", "solo-bare-observer-off")
PINNED_CONDITIONS = {
    "solo-main-cpu0": (1, "stamped"), "solo-main-cpu2": (4, "stamped"),
    "solo-main-cpu0-bare": (1, "bare"), "solo-main-cpu2-bare": (4, "bare"),
    "recovery-main-cpu0": (1, "diagnostic"), "recovery-main-cpu2": (4, "diagnostic"),
}
CONDITIONS = DEFAULT_CONDITIONS + ("solo-gpu-auto", "solo-gpu-spin", "solo-gpu-blocking", "solo-gpu-yield", "solo-stamped-observer-off") + tuple(PINNED_CONDITIONS)


def make_jobs(*, repeats=3, seed=20261022, conditions=DEFAULT_CONDITIONS, disable_thread_observer=False):
    if type(repeats) is not int or not 1 <= repeats <= 10 or type(seed) is not int or seed < 0:
        raise ValueError("explicit bounded repeats and nonnegative seed required")
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS):
        raise ValueError("unknown, duplicate or empty diagnosis conditions")
    if type(disable_thread_observer) is not bool:
        raise ValueError("disable_thread_observer must be an explicit boolean")
    jobs = []
    for repeat in range(repeats):
        rng = random.Random(seed + repeat * 101)
        order = list(conditions)
        rng.shuffle(order)
        loads = [(1, 0), (0, 1), (1, 1)]
        rng.shuffle(loads)
        for condition in order:
            recovery = condition.startswith("recovery")
            pinned = PINNED_CONDITIONS.get(condition)
            phases, elapsed = [], 0
            pattern = [(0, 0, 2000)] + [item for cpu, gpu in loads
                for item in ((cpu, gpu, 1000), (0, 0, 2000))] if recovery else [(0, 0, 5000)]
            for index, (cpu, gpu, hold_ms) in enumerate(pattern):
                end = elapsed + hold_ms * 1_000_000
                phases.append(dict(index=index, begin_ns=elapsed, end_ns=end,
                    calibration_begin_ns=elapsed, state=state(cpu, gpu, 0, 0)))
                elapsed = end
            schedule = dict(schema=SCHEDULE_SCHEMA, seed=seed + repeat * 101,
                state_semantics="offered_load_controlled", memory_fixed_for_trajectory=True,
                duration_ns=elapsed, phases=phases)
            validate_schedule(schedule)
            jobs.append(dict(id=f"diagnostic-r{repeat}-{condition}", split="diagnostic", repeat=repeat,
                condition=condition, numeric_seed=seed + repeat * 101, schedule=schedule,
                cpu_threads=1 if condition == "solo-one-thread" else 2,
                main_thread_affinity_mask=None if pinned is None else pinned[0],
                instrumentation=pinned[1] if pinned is not None else
                    "stamped" if condition == "solo-stamped-observer-off" else
                    "bare" if condition in ("solo-bare", "solo-bare-observer-off") else
                    "diagnostic" if condition in ("recovery-gpu-phases", "solo-gpu-phases", "solo-gpu-auto", "solo-gpu-spin",
                        "solo-gpu-blocking", "solo-gpu-yield") else "thread-diagnostic",
                thread_observer=not disable_thread_observer and condition not in
                    ("solo-bare-observer-off", "solo-stamped-observer-off"),
                cuda_schedule=condition.removeprefix("solo-gpu-") if condition in
                    ("solo-gpu-auto", "solo-gpu-spin", "solo-gpu-blocking", "solo-gpu-yield") else None,
                owned_roles=["cpu-worker", "gpu-worker", "main"] if recovery or
                    condition == "idle-actors" else ["main"]))
    return jobs


def prepare(args):
    from profiling.highlevel_accuracy.bridge import export_plan, reference_ports, RUNTIME_FLAGS
    from profiling.highlevel_accuracy.cases import build_case
    from profiling.highlevel_accuracy.run import case_specs
    from .thread_activity import helper_sources
    from .worker_activity import capture_openmp_environment

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(repeats=args.repeats, seed=args.seed, conditions=args.conditions,
                     disable_thread_observer=getattr(args, "disable_thread_observer", False))
    base = next(item["kwargs"] for item in case_specs("representative")
                if item["id"] == "h128-q16-history0-attention-ffn-cut128")
    compiled, contracts, programs, sources = {}, {}, {}, {}
    for threads in sorted({job["cpu_threads"] for job in jobs}):
        compiled[threads] = build_case(out / f"inputs-t{threads}", **dict(base, threads=threads)).compile()
        for record in compiled[threads]._compiled.sources:
            sources[str(Path(record["path"]).resolve())] = record["sha256"]
    for job in jobs:
        program = export_plan(compiled[job["cpu_threads"]], seed=job["numeric_seed"])
        path = out / "programs" / (job["id"] + ".json")
        ref = out / "references" / (job["id"] + ".json")
        run.save(path, program)
        run.save(ref, reference_ports(program))
        contract = common.digest({key: program[key] for key in ("graph", "actions", "physical", "bindings")})
        old = contracts.setdefault(str(job["cpu_threads"]), contract)
        common._require(old == contract, "seed changed a thread-specific program contract")
        programs[job["id"]] = dict(path=str(path), sha256=run.sha(path), reference=str(ref),
            reference_sha256=run.sha(ref), program_sha256=program["program_sha256"])
        sources[str(path)] = run.sha(path)
        sources[str(ref)] = run.sha(ref)
    sources.update(common.software_inventory(args.binary, args.runtime_dir))
    sources.update(helper_sources(args.thread_helper))
    sources[str(args.cuda_runtime.resolve())] = run.sha(args.cuda_runtime)
    protocol = dict(warmup=30, ready_timeout_seconds=10, child_stop_timeout_seconds=20,
        process_timeout_seconds=120, native_go_wait_timeout_seconds=30, go_lead_ms=500,
        main_affinity_mask=5, cpu_worker_affinity_mask=5, gpu_worker_affinity_mask=48,
        post_schedule_known_request_grace_ms=1000, telemetry_interval_ms=50,
        owned_process_counter_interval_ms=10, profiling_instrumentation="stamped", heldout_instrumentation="bare",
        maximum_pre_go_startup_retries=1, all_raw_samples_retained=True,
        diagnostic_only=True, usable_for_parameter_fit=False, actual_service_inferred_from_requests=False,
        thread_sampler=dict(binary=str(args.thread_helper.resolve()), interval_ms=2,
            affinity_mask=1024, max_duration_ms=15000, state_provider="nt_system_process_information_x64"),
        observation_scope="thread state snapshots and inclusive CPU brackets; no exact state durations or automatic attribution",
        recovery_off_age_bins_ms=[0, 50, 150, 500, 1000, 2000],
        diagnostic_conditions=list(args.conditions))
    trace_config = getattr(args, "scheduler_trace_config", None)
    if trace_config is not None:
        config = run.read(trace_config)
        common._require(len(jobs) == 1 and jobs[0]["condition"] == "solo-stamped-observer-off",
                        "one bounded stamped/observer-off job is required for scheduler recording")
        common._require(config.get("schema") == "llm-sim-etw-controller-config-1" and
                        config.get("maximum_trace_seconds") == 30, "bounded scheduler trace config required")
        for path, expected in config["bound_files"].items():
            common._require(run.sha(path) == expected, "scheduler tool/source changed: " + path)
        sources.update(config["bound_files"])
        sources[str(Path(trace_config).resolve())] = run.sha(trace_config)
        protocol["scheduler_trace"] = config
    manifest = dict(schema=common.SCHEMA, created_at_utc=run.utc(), binary=str(args.binary.resolve()),
        runtime_dirs=[str(path.resolve()) for path in args.runtime_dir], cuda_runtime=str(args.cuda_runtime.resolve()),
        sources=sources, source_inventory_sha256=common.digest(sources), programs=programs, jobs=jobs, protocol=protocol,
        context=dict(state_semantics="offered_load_controlled", time_origin="planned_go_start_v1",
            execution_boundary="legacy_double_sync", numeric_dtype="F32", hardware_identity=common.hardware_identity(),
            program_contracts_by_threads=contracts, cpu_threads="explicit job/program.physical.threads",
            backend_flags=RUNTIME_FLAGS, diagnostic_protocol="owned-thread-mechanisms-1",
            openmp_launch_environment=capture_openmp_environment(run.child_environment(args.runtime_dir),
                source="Popen child launch environment")))
    common.verify_sources(manifest)
    run.save(out / "manifest.json", manifest)
    return dict(status="prepared", jobs=len(jobs), out=str(out), hardware_executed=False)


def collect(args):
    out = args.out.resolve()
    manifest = common.load_manifest(out)
    common._require(manifest["protocol"].get("diagnostic_only") is True, "diagnostic manifest required")
    common._require(not (out / "diagnostic-rows.json").exists(), "diagnosis already collected")
    rows = []
    with run.measurement_lock():
        for job in manifest["jobs"]:
            for attempt in range(manifest["protocol"]["maximum_pre_go_startup_retries"] + 1):
                common.verify_sources(manifest)
                try:
                    row = common.execute_job(out, manifest, job, attempt=attempt)
                except BaseException:
                    suffix = f"-startup-retry{attempt}" if attempt else ""
                    receipt = out / "jobs" / (job["id"] + suffix) / "execution.json"
                    row = run.read(receipt) if receipt.is_file() else dict(status="failed", job=job)
                    rows.append(row)
                    if attempt == 0 and common.startup_retry_eligible(row):
                        continue
                    run.save(out / "diagnostic-failure.json", dict(rows=rows, failed_job=job["id"]))
                    raise
                rows.append(row)
                print(json.dumps(dict(job=job["id"], status=row["status"], process_id=row["process_id"])), flush=True)
                break
    run.save(out / "diagnostic-rows.json", rows)
    return dict(status="collected", diagnostic_only=True, successful=sum(row["status"] == "measured" for row in rows),
                retained_failed_attempts=sum(row["status"] != "measured" for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    for name in ("out", "binary", "cuda-runtime", "thread-helper"):
        setup.add_argument("--" + name, type=Path, required=True)
    setup.add_argument("--runtime-dir", type=Path, action="append", required=True)
    setup.add_argument("--seed", type=int, default=20261022)
    setup.add_argument("--repeats", type=int, default=3)
    setup.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(DEFAULT_CONDITIONS))
    setup.add_argument("--scheduler-trace-config", type=Path)
    setup.add_argument("--disable-thread-observer", action="store_true",
                       help="omit the perturbing state-query helper in every condition; retain native counters/telemetry")
    record = sub.add_parser("collect")
    record.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args) if args.command == "prepare" else collect(args)), flush=True)


if __name__ == "__main__":
    main()
