"""Serial, source-bound measurement phases for complete compiled Dense layers.

No existing campaign is overwritten. Diagnostic/profile data and independent
heldout data are collected in separate commands. Every prediction is supplied
by modeling through analysis.py; this driver only orchestrates measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "modeling", ROOT / "mapping", ROOT / "src", ROOT / "profiling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from highlevel_accuracy.bridge import RUNTIME_FLAGS, export_plan, verify_native
from highlevel_accuracy.cases import build_case
from highlevel_accuracy.run import case_specs, private_runtime_builds
from state_calibration.telemetry import DeviceTelemetrySampler

MODES = ("isolated-components", "context-trace", "bare-replay")
SCHEMA = "single-layer-context-campaign-2"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_paths():
    folders = (ROOT / "modeling/llm_infer_model", ROOT / "mapping/tensor_mapping",
               ROOT / "src/llm_simulator")
    paths = {p.resolve() for folder in folders for p in folder.rglob("*.py")
             if "results" not in p.parts and "__pycache__" not in p.parts}
    for folder in (ROOT / "profiling/highlevel_accuracy", ROOT / "profiling/single_layer_calibration"):
        paths.update(p.resolve() for p in folder.glob("*.py"))
    paths.add((ROOT / "profiling/state_calibration/telemetry.py").resolve())
    return sorted(paths)


def shared_libraries(runtime_dirs):
    return sorted({p.resolve() for folder in runtime_dirs for p in Path(folder).iterdir()
                   if p.is_file() and (p.suffix.lower() in {".dll", ".so", ".dylib"}
                                       or ".so." in p.name)})


def software(executable, runtime_dirs):
    executable = Path(executable).resolve()
    build_path = executable.parent / "build-manifest.json"
    build = read(build_path)
    if build["binary_sha256"] != sha(executable):
        raise ValueError("binary differs from its successful build binding")
    if build.get("format_version") != "single-layer-calibration-native-build-v1" or build.get("returncode") != 0:
        raise ValueError("successful single-layer native build record required")
    bound_build_files = build["bound_files"]
    native_sources = {path: digest for path, digest in bound_build_files.items()
                      if Path(path).suffix.lower() in {".py", ".cpp", ".h", ".hpp", ".inc"}}
    for path, digest in bound_build_files.items():
        if sha(path) != digest:
            raise ValueError("native source differs from successful build binding: " + path)
    libraries = {executable, build_path}
    libraries.update(shared_libraries(runtime_dirs))
    paths = {**{str(p): sha(p) for p in source_paths()}, **native_sources}
    return dict(binary=str(executable), source_sha256=paths,
                python_version=sys.version, python_executable=sys.executable,
                python_executable_sha256=sha(sys.executable),
                build_bound_sha256=bound_build_files,
                runtime_sha256={str(p): sha(p) for p in sorted(libraries)},
                shared_library_inventory=[str(p) for p in shared_libraries(runtime_dirs)],
                successful_build_record=build, private_runtime_builds=private_runtime_builds(runtime_dirs),
                runtime_search_dirs=list(map(str, runtime_dirs)),
                loader_scope="Supplied search directories and their shared-library bytes; resolved loader paths are not observed.")


def file_stats(inventory):
    return {path: [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
            for field in ("source_sha256", "runtime_sha256", "build_bound_sha256") for path in inventory[field]}


def verify_current(manifest, *, full=False):
    inventory = manifest["software"]
    for path, digest in inventory["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("software source changed after protocol freeze: " + path)
    if [str(p) for p in source_paths()] != manifest["python_source_inventory"]:
        raise ValueError("Python source inventory changed after protocol freeze")
    if [str(p) for p in shared_libraries(manifest["runtime_dirs"])] != inventory["shared_library_inventory"]:
        raise ValueError("runtime shared-library inventory changed after protocol freeze")
    if file_stats(inventory) != manifest["source_runtime_file_stats"]:
        raise ValueError("bound source/runtime metadata changed after protocol freeze")
    if full:
        for path, digest in {**inventory["runtime_sha256"], **inventory["build_bound_sha256"]}.items():
            if sha(path) != digest:
                raise ValueError("runtime library or binary bytes changed: " + path)
        if private_runtime_builds(manifest["runtime_dirs"]) != inventory["private_runtime_builds"]:
            raise ValueError("private F32 runtime binding changed")
    elif sha(manifest["executable"]) != inventory["runtime_sha256"][manifest["executable"]]:
        raise ValueError("native executable changed")


@contextmanager
def hardware_lock(directory):
    path = ROOT / "profiling/.measurement.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), campaign=str(directory), started_at=utc()), stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def selected_cases(suite):
    cases = case_specs("representative")
    if suite == "representative":
        return cases
    if suite == "diagnostic":
        wanted = {"h256-q32-history0-attention-ffn-cut252",
                  "h128-q16-history0-attention-ffn-cut128"}
        return [item for item in cases if item["id"] in wanted]
    if suite == "worst":
        return [item for item in cases if item["id"] == "h256-q32-history0-attention-ffn-cut252"]
    raise ValueError("unknown suite")


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve()
    runtime_dirs = [p.resolve() for p in args.runtime_dir]
    inventory = software(executable, runtime_dirs)
    programs, cases = {}, []
    for index, item in enumerate(selected_cases(args.suite)):
        case = out / "cases" / item["id"]
        compiled = build_case(case / "inputs", **item["kwargs"]).compile()
        bindings = {}
        geometry = {name: item["kwargs"].get(name) for name in
                    ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "tokens", "history")}
        geometry_seed = int(content_sha(geometry)[:5], 16) * 10
        for split in ("profile", "heldout"):
            for repeat in range(args.processes):
                seed = numerical_seed(geometry_seed, split, repeat, args.seed_offset)
                program = export_plan(compiled, seed=seed)
                key = f"{item['id']}--{split}-{repeat}"
                path = out / "programs" / (key + ".json")
                save(path, program)
                programs[key] = dict(path=str(path), sha256=sha(path), seed=seed,
                                     program_sha256=program["program_sha256"])
                bindings[f"{split}-{repeat}"] = key
        cases.append(dict(**item, program_keys=bindings))
    protocol = dict(warmup=args.warmup, samples=args.samples, independent_processes=args.processes,
                    numerical_seed_offset=args.seed_offset,
                    distribution_trials=args.distribution_trials, distribution_seed=args.distribution_seed,
                    primary_model="context_actions_and_gaps",
                    distribution_model_scope="predeclared_secondary_conditional_independence_candidate; primary_gates_unchanged",
                    polls=args.poll, telemetry_interval_ms=50, native_timeout_seconds=300,
                    inherited_affinity_unchanged=True, numerical_atol=5e-5, numerical_rtol=5e-5,
                    instrumentation_relative_error_max=.05,
                    within_process_cv_max=.15, between_process_cv_max=.20, relative_error_max=.20,
                    source_stat_check_before_each_job=True, source_hash_check_before_each_job=True,
                    runtime_full_byte_hash_before_after_each_phase=True,
                    runtime_metadata_checked_before_each_job=True,
                    separate_process_per_mode=True, no_outlier_removal=True,
                    no_instrumentation_cost_subtraction=True)
    manifest = dict(schema=SCHEMA, created_at=utc(), suite=args.suite, executable=str(executable),
                    runtime_dirs=list(map(str, runtime_dirs)), software=inventory,
                    software_sha256=content_sha(inventory), source_runtime_file_stats=file_stats(inventory),
                    python_source_inventory=[str(p) for p in source_paths()],
                    programs=programs, cases=cases, protocol=protocol,
                    scope="Actual public-compiled single F32 Dense invocation, serial CPU/GPU/COPY; no numerical token generation or physical-memory acceptance.")
    manifest["manifest_sha256"] = content_sha(manifest)
    verify_current(manifest, full=True)
    save(out / "manifest.json", manifest)
    return dict(status="prepared", cases=len(cases), polls=args.poll, out=str(out))


def numerical_seed(geometry_seed, split, repeat, offset):
    if split not in {"profile", "heldout"} or any(type(x) is not int or x < 0 for x in (geometry_seed, repeat, offset)):
        raise ValueError("known split and nonnegative integer seed components required")
    seed = (1_000_000 if split == "profile" else 100_000_000) + geometry_seed + repeat + offset
    if seed >= 2**32:
        raise ValueError("numerical seed exceeds uint32; choose a smaller campaign offset")
    return seed


def affinity_readback(process):
    if os.name != "nt":
        return dict(changed=False, inherited_mask=None, observation="not implemented on this OS")
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = ctypes.c_int
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(int(process._handle), ctypes.byref(mask), ctypes.byref(system)):
        raise OSError("cannot read owned child's inherited affinity")
    return dict(changed=False, inherited_mask=mask.value, system_mask=system.value)


def launch_job(out, manifest, case, repeat, poll, mode, split):
    from single_layer_calibration.analysis import make_run_record
    key = case["program_keys"][f"{split}-{repeat}"]
    binding = manifest["programs"][key]
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("prepared numerical program changed")
    program = read(binding["path"])
    run_id = f"{case['id']}--poll-{poll}--{split}-{repeat}--{mode}"
    directory = out / "jobs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    native_file = directory / "native.json"
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(manifest["runtime_dirs"] + [env.get("PATH", "")])
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(manifest["runtime_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update({k: v for k, v in RUNTIME_FLAGS.items() if k != "matmul_precision"})
    command = [manifest["executable"], binding["path"], mode,
               str(manifest["protocol"]["warmup"]), str(manifest["protocol"]["samples"]), poll, str(native_file)]
    started = utc()
    sampler, process, result = None, None, None
    record = dict(run_id=run_id, split=split, repeat=repeat, mode=mode, poll=poll,
                  seed=binding["seed"], case_id=case["id"], started_at=started,
                  command=command, software_sha256=manifest["software_sha256"])
    try:
        sampler = DeviceTelemetrySampler(interval_ms=manifest["protocol"]["telemetry_interval_ms"]).start()
        with (directory / "stdout.txt").open("x", encoding="utf-8") as stdout, (directory / "stderr.txt").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            record.update(pid=process.pid, affinity=affinity_readback(process))
            record["returncode"] = process.wait(timeout=manifest["protocol"]["native_timeout_seconds"])
        if record["returncode"] != 0:
            raise ValueError("native execution failed; stdout/stderr retained")
        native = read(native_file)
        if native.get("process_id") != process.pid:
            raise ValueError("native PID differs from the launched child")
        numeric = verify_native(program, native["verification_native"])
        save(directory / "numerical.json", numeric)
        result = make_run_record(program, native, run_id=run_id, split=split, seed=binding["seed"],
            started_at_utc=started, completed_at_utc=utc(), source_path=str(native_file),
            source_sha256=sha(native_file), software_sha256=manifest["software_sha256"],
            numerical_verification=numeric)
        record["status"] = "measured"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            record["owned_process_terminated_after_failure"] = True
        if sampler is not None:
            sampler.stop()
            sampler.write_jsonl(directory / "telemetry.jsonl")
            save(directory / "telemetry-metadata.json", sampler.metadata())
        record["completed_at"] = utc()
        record["raw_file_sha256"] = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
        save(directory / "execution.json", record)
    if result is not None:
        save(directory / "record.json", result)
    print(json.dumps(dict(run_id=run_id, status=record["status"], error=record.get("error"))), flush=True)
    return record


def load_manifest(out):
    manifest = read(out / "manifest.json")
    unsigned = deepcopy(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    if manifest.get("schema") != SCHEMA or digest != content_sha(unsigned):
        raise ValueError("campaign manifest changed")
    return manifest


def expected_runs(manifest, split):
    modes = ("bare-replay",) if split == "heldout" else MODES
    return {f"{case['id']}--poll-{poll}--{split}-{repeat}--{mode}"
            for case in manifest["cases"] for poll in manifest["protocol"]["polls"]
            for repeat in range(manifest["protocol"]["independent_processes"]) for mode in modes}


def verified_phase(out, manifest, split):
    phase = read(out / (split + "-phase.json"))
    if phase.get("manifest_sha256") != manifest["manifest_sha256"] or phase.get("split") != split:
        raise ValueError("measurement phase does not bind this manifest/split")
    rows = phase["records"]
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_runs(manifest, split):
        raise ValueError("measurement phase omits or duplicates predeclared runs")
    if phase["failed"] != 0 or phase["successful"] != len(ids) or any(row["status"] != "measured" for row in rows):
        raise ValueError("failed processes cannot be hidden by evaluating a survivor subset")
    if phase.get("sources_and_runtime_verified_at_phase_end") is not True:
        raise ValueError("measurement phase lacks final source/runtime check")
    for row in rows:
        record_path = out / "jobs" / row["run_id"] / "record.json"
        record = read(record_path)
        if record["run_id"] != row["run_id"] or record["process_id"] != row["pid"] or record["seed"] != row["seed"]:
            raise ValueError("verified run differs from the launched process inventory")
        if record["software_sha256"] != manifest["software_sha256"]:
            raise ValueError("run software fingerprint differs")
    return phase


def verified_barrier(out, manifest):
    barrier = read(out / "all-models-frozen.json")
    unsigned = deepcopy(barrier)
    digest = unsigned.pop("barrier_sha256", None)
    if (barrier.get("schema") != "single-layer-all-models-frozen-1" or digest != content_sha(unsigned)
            or barrier["manifest_sha256"] != manifest["manifest_sha256"]
            or barrier["software_sha256"] != manifest["software_sha256"]
            or barrier.get("heldout_used_for_estimation") is not False):
        raise ValueError("global frozen model barrier differs")
    expected = {(case["id"], poll) for case in manifest["cases"] for poll in manifest["protocol"]["polls"]}
    actual = [(item["case_id"], item["poll"]) for item in barrier["models"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("global barrier omits or duplicates a predeclared case/poll")
    for item in barrier["models"]:
        expected_path = out / "models" / (item["case_id"] + "--poll-" + item["poll"] + ".json")
        if Path(item["path"]).resolve() != expected_path.resolve() or sha(expected_path) != item["sha256"]:
            raise ValueError("frozen model path or bytes differ")
    return barrier


def collect(args, *, heldout=False):
    out = args.out.resolve()
    manifest = load_manifest(out)
    verify_current(manifest, full=True)
    split = "heldout" if heldout else "profile"
    barrier = verified_barrier(out, manifest) if heldout else None
    barrier_file_sha = sha(out / "all-models-frozen.json") if heldout else None
    if heldout:
        verified_phase(out, manifest, "profile")
    else:
        if (out / "all-models-frozen.json").exists():
            raise ValueError("profiles already frozen; start a new campaign")
    records = []
    with hardware_lock(out):
        count = manifest["protocol"]["independent_processes"]
        # Alternate case/mode/poll order across independent repetitions. The
        # heldout command measures only the unchanged uninstrumented total.
        for repeat in range(count):
            cases = manifest["cases"] if repeat % 2 == 0 else list(reversed(manifest["cases"]))
            for index, case in enumerate(cases):
                polls = manifest["protocol"]["polls"]
                if (repeat + index) % 2:
                    polls = list(reversed(polls))
                for poll in polls:
                    modes = ["bare-replay"] if heldout else list(MODES[repeat % 3:] + MODES[:repeat % 3])
                    for mode in modes:
                        verify_current(manifest)
                        if heldout:
                            verified_barrier(out, manifest)
                            if sha(out / "all-models-frozen.json") != barrier_file_sha:
                                raise ValueError("global barrier file changed during heldout")
                        row = launch_job(out, manifest, case, repeat, poll, mode, split)
                        records.append(row)
        verify_current(manifest, full=True)
        if heldout:
            verified_barrier(out, manifest)
            if sha(out / "all-models-frozen.json") != barrier_file_sha:
                raise ValueError("global barrier file changed during heldout")
    status = dict(schema="single-layer-measurement-phase-1", split=split, completed_at=utc(),
                  records=records, successful=sum(r["status"] == "measured" for r in records),
                  failed=sum(r["status"] != "measured" for r in records),
                  manifest_sha256=manifest["manifest_sha256"], sources_and_runtime_verified_at_phase_end=True)
    save(out / (split + "-phase.json"), status)
    return dict(split=split, successful=status["successful"], failed=status["failed"])


def records_for(out, case, poll, split, manifest):
    prefix = f"{case['id']}--poll-{poll}--{split}-"
    phase = read(out / (split + "-phase.json"))
    expected = {row["run_id"]: row for row in phase["records"] if row["run_id"].startswith(prefix)}
    paths = {path.parent.name: path for path in (out / "jobs").glob(prefix + "*/record.json")}
    if not expected or set(paths) != set(expected):
        raise ValueError("consumed record set differs from the declared case/poll/split phase subset")
    # The caller verified the whole phase once; recheck only the consumed subset.
    rows = []
    for run_id in sorted(expected):
        row, record = expected[run_id], read(paths[run_id])
        if (record["run_id"], record["process_id"], record["seed"], record["software_sha256"]) != (
                run_id, row["pid"], row["seed"], manifest["software_sha256"]):
            raise ValueError("verified run differs from the launched process inventory or software fingerprint")
        rows.append(record)
    return rows


def freeze(args):
    from single_layer_calibration.analysis import freeze_profile
    from single_layer_calibration.distribution_profiles import freeze_distributions
    out = args.out.resolve()
    manifest = load_manifest(out)
    verify_current(manifest, full=True)
    verified_phase(out, manifest, "profile")
    models = []
    for case in manifest["cases"]:
        program = read(manifest["programs"][case["program_keys"]["profile-0"]]["path"])
        for poll in manifest["protocol"]["polls"]:
            frozen = freeze_profile(program, records_for(out, case, poll, "profile", manifest), frozen_at_utc=utc(),
                instrumentation_relative_error_max=manifest["protocol"]["instrumentation_relative_error_max"])
            distribution = freeze_distributions(frozen,
                trials=manifest["protocol"]["distribution_trials"], seed=manifest["protocol"]["distribution_seed"])
            combined = dict(format_version="single-layer-combined-frozen-v2", base=frozen,
                            distribution=distribution, frozen_at_utc=utc(),
                            heldout_used_for_estimation=False, primary_model="context_actions_and_gaps")
            combined["combined_sha256"] = content_sha(combined)
            path = out / "models" / (case["id"] + "--poll-" + poll + ".json")
            save(path, combined)
            models.append(dict(case_id=case["id"], poll=poll, path=str(path), sha256=sha(path)))
    barrier = dict(schema="single-layer-all-models-frozen-1", frozen_at_utc=utc(),
                   manifest_sha256=manifest["manifest_sha256"], software_sha256=manifest["software_sha256"],
                   models=models, heldout_used_for_estimation=False)
    barrier["barrier_sha256"] = content_sha(barrier)
    save(out / "all-models-frozen.json", barrier)
    return dict(status="all_models_frozen", models=len(models), barrier=str(out / "all-models-frozen.json"))


def evaluate(args):
    from single_layer_calibration.analysis import evaluate_heldout
    from single_layer_calibration.distribution_profiles import evaluate_distribution
    out = args.out.resolve()
    manifest = load_manifest(out)
    verify_current(manifest, full=True)
    barrier = verified_barrier(out, manifest)
    verified_phase(out, manifest, "profile")
    phase = verified_phase(out, manifest, "heldout")
    if any(datetime.fromisoformat(row["started_at"]) < datetime.fromisoformat(barrier["frozen_at_utc"])
           for row in phase["records"]):
        raise ValueError("heldout run began before the global model barrier")
    results = []
    for item in barrier["models"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("frozen model changed")
        case = next(x for x in manifest["cases"] if x["id"] == item["case_id"])
        program = read(manifest["programs"][case["program_keys"]["profile-0"]]["path"])
        combined = read(item["path"])
        unsigned = deepcopy(combined)
        signature = unsigned.pop("combined_sha256", None)
        if (combined.get("format_version") != "single-layer-combined-frozen-v2" or
                signature != content_sha(unsigned) or combined.get("heldout_used_for_estimation") is not False or
                combined.get("primary_model") != "context_actions_and_gaps"):
            raise ValueError("combined frozen model differs")
        report = evaluate_heldout(program, combined["base"], records_for(out, case, item["poll"], "heldout", manifest))
        report["distribution_candidate"] = evaluate_distribution(combined["distribution"], report)
        results.append(dict(case_id=item["case_id"], poll=item["poll"], report=report))
    path = out / "evaluation.json"
    save(path, dict(schema="single-layer-context-evaluation-2", evaluated_at=utc(),
                    manifest_sha256=manifest["manifest_sha256"], barrier_sha256=barrier["barrier_sha256"],
                    results=results, hardware_scope=manifest["scope"],
                    heldout_used_for_estimation=False, outliers_removed=0))
    return dict(status="evaluated", cases=len(results), output=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--suite", choices=("worst", "diagnostic", "representative"), default="diagnostic")
    prepare_parser.add_argument("--executable", type=Path, required=True)
    prepare_parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    prepare_parser.add_argument("--poll", choices=("default", "0"), nargs="+", default=["default", "0"])
    prepare_parser.add_argument("--processes", type=int, default=3)
    prepare_parser.add_argument("--warmup", type=int, default=30)
    prepare_parser.add_argument("--samples", type=int, default=101)
    prepare_parser.add_argument("--seed-offset", type=int, required=True,
                                help="Explicit numerical-data cohort offset; use a new value for fresh post-development validation")
    prepare_parser.add_argument("--distribution-trials", type=int, default=20000)
    prepare_parser.add_argument("--distribution-seed", type=int, default=20261018)
    for name in ("profile", "freeze", "heldout", "evaluate"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if (args.processes < 3 or not 3 <= args.samples <= 10000 or not 0 <= args.warmup <= 10000 or
                len(set(args.poll)) != len(args.poll) or not 0 <= args.seed_offset <= 4_000_000_000 or
                not 1000 <= args.distribution_trials <= 1_000_000 or args.distribution_seed < 0):
            raise ValueError("require >=3 processes, >=3 samples, bounded warmup and unique polling controls")
        result = prepare(args)
    elif args.command == "profile":
        result = collect(args)
    elif args.command == "heldout":
        result = collect(args, heldout=True)
    elif args.command == "freeze":
        result = freeze(args)
    else:
        result = evaluate(args)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
