"""Source-bound collection of continuous execution under planned offered work.

There is no performance equation or estimator here.  Native actors are owned,
finite processes.  Planned requests share QPC; actual worker intervals remain
separate evidence, never renamed utilization.  Historical campaigns are read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src", ROOT / "modeling", ROOT / "mapping", ROOT / "profiling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from profiling.state_calibration import run
from profiling.state_calibration.telemetry import DeviceTelemetrySampler, NativeClock
from profiling.resource_calibration.collect import OwnedChild, start_memory_child
from profiling.state_response.schedule import make_jobs, timed_segments

SCHEMA = "state-response-campaign-1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def hardware_identity():
    from profiling.state_calibration.telemetry import NVMLProvider
    provider = NVMLProvider(0)
    try:
        gpu = provider.metadata()
    finally:
        provider.close()
    from profiling.state_response.diagnose_cpu import topology
    return {"os": platform.platform(), "machine_architecture": platform.machine(),
            "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or None,
            "logical_processors": os.cpu_count(), "gpu_nvml_identity": gpu,
            "logical_to_physical_processor_topology": topology(),
            "unknown_identity_fields_are_not_inferred": True}


def software_inventory(binary, runtime_dirs):
    binary = Path(binary).resolve()
    build_path = binary.parent / "build-manifest.json"
    build = run.read(build_path)
    _require(build.get("returncode") == 0 and build.get("binary_sha256") == run.sha(binary),
             "a successful source-bound native build is required")
    files = {str(binary): run.sha(binary), str(build_path): run.sha(build_path)}
    for name, expected in build["bound_files"].items():
        _require(run.sha(name) == expected, "native build source changed: " + name)
        files[name] = expected
    for folder in (ROOT / "modeling/llm_infer_model", ROOT / "mapping/tensor_mapping",
                   ROOT / "src/llm_simulator", ROOT / "profiling/state_response",
                   ROOT / "profiling/state_calibration", ROOT / "profiling/resource_calibration",
                   ROOT / "profiling/highlevel_accuracy"):
        for path in python_sources(folder):
            files[str(path.resolve())] = run.sha(path)
    # Only explicitly supplied runtime directories are consumed.  No host CUDA
    # autodetection or unrecorded profile fallback enters this collector.
    for folder in runtime_dirs:
        for path in Path(folder).glob("*.dll"):
            files[str(path.resolve())] = run.sha(path)
        identity = Path(folder) / "strict-f32-runtime.json"
        if identity.is_file():
            files[str(identity.resolve())] = run.sha(identity)
            data = run.read(identity)
            for field in ("original_source_sha256", "inputs_sha256", "output_sha256", "evidence_sha256"):
                for name, expected in data[field].items():
                    _require(run.sha(name) == expected, "strict F32 source changed: " + name)
                    files[name] = expected
    return files


def python_sources(folder):
    """Bind executable sources, not archived experiment copies or build trees."""
    for path in Path(folder).rglob("*.py"):
        relative = path.relative_to(folder)
        if any(part in {"results", "__pycache__", "build", ".venv"} or part.startswith("build-")
               for part in relative.parts[:-1]):
            continue
        yield path


def bind_observed_runtime_files(observations):
    """Hash observed relevant paths after sampling; never infer loader resolution."""
    from profiling.state_response.worker_activity import read_file_version
    relevant = ("ggml", "llama", "cudart", "cublas", "nvcuda", "nvfatbinary", "vcomp", "libomp", "libiomp")
    paths = {item["path"] for observation in observations.values() for item in observation.get("modules", [])
             if item.get("path") and Path(item["path"]).name.lower().startswith(relevant)}
    files, rows = {}, []
    for name in sorted(paths):
        try:
            path = str(Path(name).resolve())
            value = run.sha(path)
            files[path] = value
            rows.append({"path": path, "on_disk_sha256": value, "file_version": read_file_version(path),
                         "loaded_memory_bytes_verified": False})
        except (OSError, ValueError) as error:
            rows.append({"path": name, "on_disk_sha256": None, "unknown": repr(error)})
    return files, {"modules": rows, "hash_phase": "after_sampling_before_record_normalization",
                   "loaded_memory_bytes_verified": False,
                   "scope": "disk files at actually observed relevant loaded paths; not PATH guesses or mapped memory hashes"}


def verify_sources(manifest):
    _require(manifest.get("schema") == SCHEMA, "wrong campaign schema")
    _require(digest(manifest["sources"]) == manifest["source_inventory_sha256"], "source inventory changed")
    for name, expected in manifest["sources"].items():
        _require(run.sha(name) == expected, "campaign source changed: " + name)


def prepare(args):
    from profiling.highlevel_accuracy.bridge import export_plan, reference_ports, RUNTIME_FLAGS
    from profiling.highlevel_accuracy.cases import build_case
    from profiling.highlevel_accuracy.run import case_specs
    from profiling.state_response.worker_activity import capture_openmp_environment

    for name in ("main_affinity_mask", "cpu_worker_affinity_mask", "gpu_worker_affinity_mask"):
        _require(type(getattr(args, name)) is int and 0 < getattr(args, name) < (1 << os.cpu_count()),
                 "affinity must explicitly name available logical processors: " + name)
    main_thread_mask = getattr(args, "main_thread_affinity_mask", None)
    if main_thread_mask is not None:
        _require(type(main_thread_mask) is int and main_thread_mask > 0 and
                 main_thread_mask & (main_thread_mask - 1) == 0 and
                 main_thread_mask & args.main_affinity_mask == main_thread_mask,
                 "main-thread affinity must select one CPU inside the main process mask")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(repeats=args.repeats, seed=args.seed, memory_bytes=args.memory_bytes,
                     profile_hold_ms=args.profile_hold_ms, settle_ms=args.settle_ms,
                     heldout_cycles=args.heldout_cycles,
                     profile_state_holds_ms=args.profile_state_holds_ms)
    if args.memory_index is not None:
        jobs = [job for job in jobs if job["id"].endswith(f"-m{args.memory_index}")]
    if main_thread_mask is not None:
        for job in jobs:
            job["main_thread_affinity_mask"] = main_thread_mask
    spec = next(item["kwargs"] for item in case_specs("representative")
                if item["id"] == "h128-q16-history0-attention-ffn-cut128")
    compiled = build_case(out / "inputs", **spec).compile()
    programs = {}
    contract = None
    for job in jobs:
        program = export_plan(compiled, seed=job["numeric_seed"])
        path, reference_path = out / "programs" / (job["id"] + ".json"), out / "references" / (job["id"] + ".json")
        run.save(path, program)
        run.save(reference_path, reference_ports(program))
        semantic = {key: program[key] for key in ("graph", "actions", "physical", "bindings")}
        current_contract = digest(semantic)
        _require(contract in (None, current_contract), "numeric seed changed the declared program contract")
        contract = current_contract
        programs[job["id"]] = {"path": str(path), "sha256": run.sha(path),
                               "reference": str(reference_path), "reference_sha256": run.sha(reference_path),
                               "program_sha256": program["program_sha256"]}
    sources = software_inventory(args.binary, args.runtime_dir)
    for record in compiled._compiled.sources:
        sources[str(Path(record["path"]).resolve())] = record["sha256"]
    for entry in programs.values():
        sources[entry["path"]] = entry["sha256"]
        sources[entry["reference"]] = entry["reference_sha256"]
    sources[str(args.cuda_runtime.resolve())] = run.sha(args.cuda_runtime)
    hardware = hardware_identity()
    manifest = {"schema": SCHEMA, "created_at_utc": run.utc(), "binary": str(args.binary.resolve()),
                "runtime_dirs": [str(p.resolve()) for p in args.runtime_dir],
                "cuda_runtime": str(args.cuda_runtime.resolve()), "sources": sources,
                "source_inventory_sha256": digest(sources), "programs": programs, "jobs": jobs,
                "context": {"program_contract_sha256": contract,
                            "state_semantics": "offered_load_controlled",
                            "time_origin": "planned_go_start_v1",
                            "initial_effective_cpu_gpu_requests": [0, 0],
                            "execution_boundary": "legacy_double_sync",
                            "software_sha256": digest(sources), "numeric_dtype": "F32",
                            "hardware_identity": hardware,
                            "backend_flags": RUNTIME_FLAGS, "cpu_threads": 2,
                            "worker_contract": "persistent_preinitialized_target_replay"},
                "protocol": {"warmup": 30, "ready_timeout_seconds": 10,
                             "native_go_wait_timeout_seconds": 30,
                             "child_stop_timeout_seconds": 20, "process_timeout_seconds": 120,
                             "go_lead_ms": 500, "main_affinity_mask": args.main_affinity_mask,
                             "profile_settle_ms": args.settle_ms,
                             "minimum_cell_samples_per_process": 10,
                             "maximum_pre_go_startup_retries": 1,
                             "startup_retry_policy": "only_no_GO_and_verified_owned_process_exit; preserve_all_attempts",
                             "request_response_lags": {
                                 "enabled": True, "minimum_edges_per_process": 1,
                                 "minimum_independent_processes": 3, "within_process_cv_max": .15,
                                 "between_process_cv_max": .20, "rounding": "ceil_exact_median_ns"},
                             "post_schedule_known_request_grace_ms": 1000,
                             "telemetry_interval_ms": 50,
                             "owned_process_counter_interval_ms": 10,
                             "owned_process_counters_used_as_predictor": False,
                             "loaded_runtime_observation": "actual_owned_handle_paths_before_GO; disk_hashes_after_sampling",
                             "total_telemetry_used_as_predictor": False,
                             "cpu_worker_affinity_mask": args.cpu_worker_affinity_mask,
                             "gpu_worker_affinity_mask": args.gpu_worker_affinity_mask,
                             "profiling_instrumentation": "stamped", "heldout_instrumentation": "bare",
                             "min_independent_processes": 3, "within_process_cv_max": .15,
                             "between_process_cv_max": .20, "median_relative_error_max": .20,
                             "dynamic_metrics_predeclared": ["steady", "crossing", "all_known", "unknown"],
                             "comparison_models": ["lagged_dynamic_state_integration",
                                                   "dynamic_state_integration", "no_owned_cpu_gpu_load",
                                                   "entry_state_constant"],
                             "primary_prediction_model": "lagged_dynamic_state_integration",
                             "no_heldout_refit": True, "all_raw_samples_retained": True,
                             "memory_fixed_within_each_trajectory": True,
                             "predeclared_memory_cohort": args.memory_index,
                             "external_percent_occupancy_claimed": False,
                             "memory_transition_hardware_accuracy_claimed": False,
                             "physical_capacity_claimed": False,
                             "modeling_owns_all_predictions": True}}
    manifest["context"]["openmp_launch_environment"] = capture_openmp_environment(
        run.child_environment(args.runtime_dir), source="Popen child launch environment")
    if main_thread_mask is not None:
        manifest["protocol"]["main_thread_affinity_mask"] = main_thread_mask
        manifest["context"]["main_thread_affinity_mask"] = main_thread_mask
        manifest["context"]["process_affinity_masks"] = dict(main=args.main_affinity_mask,
            cpu_worker=args.cpu_worker_affinity_mask, gpu_worker=args.gpu_worker_affinity_mask)
    verify_sources(manifest)
    run.save(out / "manifest.json", manifest)
    return {"status": "prepared", "jobs": len(jobs), "out": str(out), "hardware_executed": False}


def load_manifest(out):
    manifest = run.read(Path(out) / "manifest.json")
    verify_sources(manifest)
    _require(len({job["id"] for job in manifest["jobs"]}) == len(manifest["jobs"]), "duplicate jobs")
    return manifest


def _wait_sampling(children, timeout):
    deadline = time.monotonic() + timeout
    pending = list(children)
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError("owned native trajectory exceeded its finite deadline")
        for child in tuple(pending):
            path = child.directory / "sampling-complete.json"
            if path.exists():
                try:
                    receipt = run.read(path)
                except (OSError, json.JSONDecodeError):
                    continue
                _require(receipt.get("process_id") == child.process.pid and receipt.get("role") == child.name,
                         "sampling-complete receipt has foreign process/role")
                child.record["sampling_complete"] = receipt
                pending.remove(child)
            elif child.process.poll() is not None:
                raise RuntimeError("owned actor exited before its sampling-complete handshake: " + child.name)
        time.sleep(.02)


def make_go(schedule, clock, start_tick):
    segments = timed_segments(schedule, start_tick=start_tick, frequency_hz=clock.frequency_hz)
    return {"schema": "state-response-go-v1", "clock": clock.metadata(),
            "start_tick": segments[0]["begin_tick"], "end_tick": segments[-1]["end_tick"],
            "phases": [{"phase_index": item["phase_index"], "begin_tick": item["begin_tick"],
                        "end_tick": item["end_tick"], "cpu_replay": item["state"]["cpu_replay"],
                        "gpu_replay": item["state"]["gpu_replay"]} for item in segments],
            "memory": {"dram_hold_bytes": segments[0]["state"]["host_memory_bytes"],
                       "vram_hold_bytes": segments[0]["state"]["device_memory_bytes"]}}, segments


def validate_result(native, *, program, child, role, go, clock, instrumentation):
    _require(native.get("format_version") == "state-response-native-v1", "wrong native result schema")
    _require(native.get("process_id") == child.process.pid and native.get("role") == role,
             "native process/role identity differs")
    _require(native.get("program_sha256") == program["program_sha256"], "native program identity differs")
    for key in ("clock_id", "frequency_hz"):
        _require(native["clock"][key] == clock.metadata()[key], "native clock differs")
    _require(native.get("go_schedule") == go, "native consumed GO differs from declared external schedule")
    _require(native.get("actions") == program["actions"], "native action inventory differs")
    _require(native.get("observed_runtime_flags") ==
             {"GGML_CUDA_DISABLE_GRAPHS": "1", "NVIDIA_TF32_OVERRIDE": "0", "matmul_precision": "GGML_PREC_F32"},
             "native execution flags differ")
    _require(child.record["ready"]["ready_tick"] < go["start_tick"], "actor not ready before GO")
    blocks = native.get("blocks", [])
    _require(bool(blocks), "native returned no completed work")
    _require(sum(len(block["iterations"]) for block in blocks) > 0, "native has no samples")
    if role == "main":
        _require(instrumentation in ("bare", "stamped", "diagnostic", "diagnostic-control", "thread-diagnostic"), "unsupported instrumentation")
        for block in blocks:
            _require(len(block["action_intervals"]) ==
                     (0 if instrumentation == "bare" else len(block["iterations"]) * len(program["actions"])),
                     "native full-action trace coverage differs")


def selected_roles(job):
    """Explicit actor-presence controls; absent actors are never inferred idle."""
    roles = job.get("owned_roles", ["cpu-worker", "gpu-worker", "main"])
    _require(isinstance(roles, list) and all(isinstance(role, str) for role in roles)
             and len(roles) == len(set(roles)) and "main" in roles
             and set(roles) <= {"cpu-worker", "gpu-worker", "main"}, "invalid owned actor inventory")
    if roles != ["cpu-worker", "gpu-worker", "main"]:
        _require(job["split"] == "diagnostic", "actor-presence controls require a separate diagnostic protocol")
    for role, channel in (("cpu-worker", "cpu_replay"), ("gpu-worker", "gpu_replay")):
        if role not in roles:
            _require(all(phase["state"][channel] == 0 for phase in job["schedule"]["phases"]),
                     "offered replay cannot be requested without its owned actor: " + role)
    return [role for role in ("cpu-worker", "gpu-worker", "main") if role in roles]


def main_instrumentation(job, protocol, *, diagnostic=False):
    mode = job.get("instrumentation")
    if mode is not None:
        _require(job["split"] == "diagnostic" and mode in
                 {"bare", "stamped", "diagnostic", "diagnostic-control", "thread-diagnostic"},
                 "explicit instrumentation requires a diagnostic job")
        return mode
    return ("diagnostic" if diagnostic else protocol["profiling_instrumentation"]
            if job["split"] == "profile" else protocol["heldout_instrumentation"])


def scheduler_ready(config, clock):
    """Require this bounded recorder before launching actors; never start one."""
    root = Path(config["out_dir"])
    ready = run.read(root / "capture-ready.json")
    _require(ready.get("schema") == "llm-sim-etw-ready-1" and ready.get("instance_name") == config["instance_name"]
             and ready.get("qpc_frequency") == clock.frequency_hz and ready.get("actors_may_now_be_launched") is True,
             "scheduler recorder identity/clock/readiness differs")
    _require(ready["qpc_deadline"] - clock.now() >= 20 * clock.frequency_hz,
             "bounded recorder has insufficient time remaining; no actor launch allowed")
    return ready


def stop_scheduler(config, job_id, *, timeout=60):
    """Request only the recorder named in this owned config, never a global stop."""
    root = Path(config["out_dir"])
    stop = root / "stop.request.json"
    if not stop.exists():
        run.save(stop, dict(schema="llm-sim-etw-stop-request-1", instance_name=config["instance_name"], owned_job_id=job_id))
    deadline = time.monotonic() + timeout
    while not (root / "broker-final.json").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("owned scheduler recorder stop receipt deadline exceeded; no foreign recorder is touched")
        time.sleep(.05)
    final = run.read(root / "broker-final.json")
    _require(final.get("instance_name") == config["instance_name"] and final.get("recording_stop_success") is True
             and final.get("failure") is None and final.get("reason") == "matching_controller_request",
             "owned scheduler recorder did not complete the requested trace; retain partial evidence")
    return run.read(root / "capture-stopped.json"), {str(path.resolve()): run.sha(path)
             for path in root.rglob("*") if path.is_file()}


def native_evidence_bindings(*, program, context, run_id, split, seed, process_id,
                             source_files, numerical_verification, raw_bindings):
    """Small source-only receipt; action extraction stays in its scoped adapter.

    This does not estimate costs, normalize intervals or claim predictive
    accuracy. Actual native result/numerical/lifetime checks precede this call.
    """
    _require(numerical_verification.get("passed") is True, "numerical evidence must pass")
    _require(set(raw_bindings) >= {"program", "native", "events", "numerical"}, "raw source bindings incomplete")
    _require(all(name in source_files for name in raw_bindings.values()), "unhashed raw binding")
    return dict(schema="raw-native-evidence-bindings-1", program_sha256=program["program_sha256"],
                context=context, run_id=run_id, split=split, seed=seed, process_id=process_id,
                numerical_verification=numerical_verification, source_files=source_files,
                raw_bindings=raw_bindings, action_intervals_expanded=False,
                timing_parameters_estimated=False, accuracy_qualified=False)


def execute_job(out, manifest, job, *, diagnostic=False, attempt=0, native_evidence_only=False):
    from profiling.state_response.analysis import make_run_record
    from profiling.state_response.worker_activity import (
        OwnedActor, capture_openmp_environment, observe_loaded_module_paths, make_counter_sampler,
    )
    _require(type(native_evidence_only) is bool, "native_evidence_only must be explicit bool")
    if native_evidence_only:
        _require(manifest["protocol"].get("collection_record_format") == "raw_native_evidence_bindings_v1",
                 "compact evidence collection must be predeclared")
    attempt_id = job["id"] + (f"-startup-retry{attempt}" if attempt else "")
    directory = Path(out) / "jobs" / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    binding = manifest["programs"][job["id"]]
    program = run.read(binding["path"])
    clock, protocol = NativeClock(), manifest["protocol"]
    env = run.child_environment(manifest["runtime_dirs"])
    env.pop("SIMULATOR_CUDA_SCHEDULE", None)
    for key in ("STATE_PROBE_FIXED_ITERATIONS", "STATE_PROBE_READY_FILE", "STATE_PROBE_STOP_FILE"):
        env.pop(key, None)
    _require(capture_openmp_environment(env, source="Popen child launch environment") ==
             manifest["context"]["openmp_launch_environment"], "OpenMP launch environment changed since preparation")
    children, native_children, holders = [], [], []
    sampler = activity = thread_sampler = main_thread_pin = None
    thread_runtime_files = {}
    trace_config = manifest["protocol"].get("scheduler_trace")
    trace_files, trace_stopped = {}, False
    started = run.utc()
    row = {"schema": SCHEMA, "job": job, "attempt": attempt, "attempt_id": attempt_id,
           "started_at_utc": started,
           "source_inventory_sha256": manifest["source_inventory_sha256"], "status": "running"}
    try:
        if trace_config is not None:
            row["scheduler_recording_ready"] = scheduler_ready(trace_config, clock)
        first = job["schedule"]["phases"][0]["state"]
        for device, key in (("cpu", "host_memory_bytes"), ("gpu", "device_memory_bytes")):
            if first[key]:
                holder = start_memory_child(directory / (device + "-memory"), device=device,
                                            size=first[key], manifest=manifest, clock=clock)
                holders.append(holder)
                children.append(holder)
        go_path = directory / "go.json"
        roles = selected_roles(job)
        for role, mask_key in (("cpu-worker", "cpu_worker_affinity_mask"),
                               ("gpu-worker", "gpu_worker_affinity_mask"), ("main", "main_affinity_mask")):
            if role not in roles:
                continue
            target = run.choose_target(program, "cpu" if role == "cpu-worker" else "gpu")
            role_directory = directory / role
            instrument = main_instrumentation(job, protocol, diagnostic=diagnostic) if role == "main" else "bare"
            target_argument = "none" if instrument == "thread-diagnostic" else str(target)
            child_env = dict(env)
            if role == "main" and job.get("cuda_schedule") is not None:
                _require(job["split"] == "diagnostic" and job["cuda_schedule"] in {"auto", "spin", "yield", "blocking"},
                         "CUDA scheduling controls require an explicit diagnostic job")
                child_env["SIMULATOR_CUDA_SCHEDULE"] = job["cuda_schedule"]
            command = [manifest["binary"], binding["path"], role, instrument, target_argument, str(protocol["warmup"]),
                       str(role_directory / "ready.json"), str(go_path), str(role_directory / "stop.json"),
                       str(role_directory / "result.json")]
            child = OwnedChild(role, role_directory, command, child_env, clock,
                               affinity_mask=job.get(mask_key, protocol[mask_key]), ready_timeout=protocol["ready_timeout_seconds"],
                               stop_timeout=protocol["child_stop_timeout_seconds"])
            child.instrumentation = instrument
            children.append(child)
            native_children.append(child)
        actors = [OwnedActor.from_popen(child.name, child.process) for child in native_children]
        pin_mask = job.get("main_thread_affinity_mask")
        if pin_mask is not None:
            _require(job["split"] == "diagnostic" or
                     (pin_mask == protocol.get("main_thread_affinity_mask") ==
                      manifest["context"].get("main_thread_affinity_mask")),
                     "formal main-thread pinning requires matching frozen protocol/context scope")
            from profiling.state_response.thread_affinity import pin_owned_main_thread
            main_actor = next(actor for actor in actors if actor.actor_id == "main")
            main_child = next(child for child in native_children if child.name == "main")
            try:
                main_thread_pin = pin_owned_main_thread(main_actor, main_child.record["ready"]["main_thread_id"],
                                                       pin_mask, clock=clock)
            except Exception as error:
                if hasattr(error, "receipt"):
                    run.save(directory / "main-thread-affinity-failure.json", error.receipt)
                raise
            run.save(directory / "main-thread-affinity-before-go.json", main_thread_pin.receipt)
        runtime_observations = {actor.actor_id: observe_loaded_module_paths(actor, clock=clock) for actor in actors}
        run.save(directory / "loaded-runtime-paths.json", runtime_observations)
        activity = make_counter_sampler(actors, interval_ms=protocol["owned_process_counter_interval_ms"], clock=clock).start()
        sampler = DeviceTelemetrySampler(interval_ms=protocol["telemetry_interval_ms"], clock=clock).start()
        if protocol.get("thread_sampler") is not None and job.get("thread_observer", True):
            _require(job["split"] == "diagnostic", "thread snapshot observation is diagnostic only")
            from profiling.state_response.thread_activity import OwnedThreadSampler
            config = protocol["thread_sampler"]
            thread_sampler = OwnedThreadSampler(actors, directory / "thread-sampler",
                helper_binary=config["binary"], interval_ms=config["interval_ms"],
                affinity_mask=config["affinity_mask"], max_duration_ms=config["max_duration_ms"],
                state_provider=config["state_provider"], clock=clock).start()
        _require(all(child.process.poll() is None for child in children),
                 "an owned actor exited before GO; no timed foreground launch is allowed")
        start_tick = clock.now() + protocol["go_lead_ms"] * clock.frequency_hz // 1000
        go, segments = make_go(job["schedule"], clock, start_tick)
        run.save(go_path, go)
        recovery_end = go["end_tick"] + protocol["post_schedule_known_request_grace_ms"] * clock.frequency_hz // 1000
        recovery_state = deepcopy(segments[-1]["state"])
        recovery_state.update(cpu_replay=0, gpu_replay=0)
        # A last complete invocation may straddle the end request.  The offered
        # workers receive no more work after end_tick; their already-running
        # final calls are retained.  Allocations stay held through this whole
        # predeclared support window, rather than dropping the final sample.
        known_segments = segments + [{"phase_index": len(segments), "begin_tick": go["end_tick"],
                                      "end_tick": recovery_end, "calibration_begin_tick": recovery_end,
                                      "state": recovery_state}]
        events = {"known_segments": known_segments, "events": [
            {"phase_index": item["phase_index"], "requested_resources": {
                "cpu_replay": bool(item["state"]["cpu_replay"]), "gpu_replay": bool(item["state"]["gpu_replay"]),
                "host_memory_bytes": item["state"]["host_memory_bytes"],
                "device_memory_bytes": item["state"]["device_memory_bytes"]},
             "steady_begin_tick": item["calibration_begin_tick"], "steady_end_tick": item["end_tick"]}
            for item in segments], "schedule": job["schedule"], "go": go,
            "input_semantics": "planned_owned_work_requests_not_measured_utilization",
            "worker_response_is_included_in_empirical_response_not_assumed_instantaneous": True}
        run.save(directory / "events.json", events)
        _wait_sampling(native_children, protocol["process_timeout_seconds"])
        while clock.now() < recovery_end:
            time.sleep(min(.02, (recovery_end - clock.now()) / clock.frequency_hz))
        if main_thread_pin is not None:
            run.save(directory / "main-thread-affinity-after-sampling.json", main_thread_pin.observe())
        if trace_config is not None:
            receipt, trace_files = stop_scheduler(trace_config, job["id"])
            trace_stopped = True
            run.save(directory / "scheduler-trace.json", dict(config=trace_config, stop_receipt=receipt,
                trace_files=trace_files, exact_scheduler_intervals_validated=False))
        if thread_sampler is not None:
            thread_document = thread_sampler.stop()
            run.save(directory / "thread-activity.json", thread_document)
            thread_sampler = None
            assembly_receipts = []
            for key, value in thread_document.get("runtime", {}).items():
                if not key.endswith("_assembly"):
                    continue
                receipt = dict(runtime_field=key, path=value, on_disk_sha256=None,
                    mapped_image_memory_verified=False)
                try:
                    path = Path(value)
                    _require(path.is_absolute() and path.is_file(), "observed assembly path is not an absolute file")
                    receipt["on_disk_sha256"] = run.sha(path)
                    thread_runtime_files[str(path.resolve())] = receipt["on_disk_sha256"]
                except (OSError, ValueError) as error:
                    receipt["unknown"] = repr(error)
                assembly_receipts.append(receipt)
            run.save(directory / "thread-runtime-disk-files.json", dict(modules=assembly_receipts,
                hash_phase="after_thread_sampling_before_native_STOP", mapped_image_memory_verified=False,
                provenance="actual helper Assembly.Location receipts; not compiler reference path guesses"))
        activity.stop()
        run.save(directory / "worker-activity.json", {"metadata": activity.metadata(), "records": activity.records,
                 "actors": {actor.actor_id: {"process_id": actor.process_id} for actor in actors},
                 "used_as_predictor": False, "sampling_stopped_before_validation_or_teardown": True,
                 "zero_CPU_increment_proves_quiescence": False})
        activity = None
        samples = sampler.stop()
        run.save(directory / "telemetry.json", {"metadata": sampler.metadata(), "samples": samples,
                 "used_as_predictor": False,
                 "scope": "rolling total counters and clocks; not instantaneous external utilization"})
        sampler = None
        for child in native_children:
            child.stop()
            _require(child.record["returncode"] == 0 and child.record["forced_termination"] is False,
                     "owned native actor failed or required forced termination")
        numerical = {}
        for child in native_children:
            native = run.read(child.directory / "result.json")
            validate_result(native, program=program, child=child, role=child.name, go=go, clock=clock,
                            instrumentation=child.instrumentation)
            numerical[child.name] = run.validate_numeric(native, run.read(binding["reference"]))
        main = next(child for child in native_children if child.name == "main")
        for holder in holders:
            holder.stop()
            record = run.read(holder.directory / "result.json")
            _require(record.get("released") is True and record.get("status") == "released", "memory holder failed release")
            _require(record["allocation_end_tick"] <= start_tick and
                     recovery_end <= record["release_begin_tick"], "memory hold did not cover the known trajectory support")
        numerical["passed"] = all(numerical[name].get("passed", numerical[name].get("numerical_passed", False))
                                  for name in roles)
        run.save(directory / "numerical.json", numerical)
        _require(numerical["passed"], "independent numerical checks failed")
        ended = run.utc()
        raw_bindings = {"program": binding["path"], "native": str(main.directory / "result.json"),
                        "events": str(directory / "events.json"), "numerical": str(directory / "numerical.json"),
                        "worker_activity": str(directory / "worker-activity.json")}
        for role, key in (("cpu-worker", "worker_cpu"), ("gpu-worker", "worker_gpu")):
            if role in roles:
                raw_bindings[key] = str(directory / role / "result.json")
        source_files = dict(manifest["sources"])
        source_files.update(thread_runtime_files)
        source_files.update(trace_files)
        observed_files, runtime_receipt = bind_observed_runtime_files(runtime_observations)
        run.save(directory / "loaded-runtime-disk-files.json", runtime_receipt)
        source_files.update(observed_files)
        for path in directory.rglob("*"):
            if path.is_file():
                source_files[str(path.resolve())] = run.sha(path)
        if native_evidence_only:
            record = native_evidence_bindings(program=program, context=manifest["context"],
                run_id=attempt_id, split=job["split"], seed=job["numeric_seed"], process_id=main.process.pid,
                source_files=source_files, numerical_verification=numerical, raw_bindings=raw_bindings)
            record_path = directory / "native-evidence-bindings.json"
        else:
            record = make_run_record(program, run.read(raw_bindings["native"]), events,
                                     context=manifest["context"], run_id=attempt_id, split=job["split"],
                                     seed=job["numeric_seed"], process_id=main.process.pid,
                                     started_at_utc=started, ended_at_utc=ended,
                                     source_files=source_files, numerical_verification=numerical,
                                     raw_bindings=raw_bindings)
            record_path = directory / "normalized.json"
        run.save(record_path, record)
        row.update(status="measured", ended_at_utc=ended, normalized_path=str(record_path),
                   normalized_sha256=run.sha(record_path), process_id=main.process.pid,
                   children=[child.record for child in children], cleanup_verified=True)
    except BaseException as error:
        row.update(status="failed", ended_at_utc=run.utc(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup = []
        if main_thread_pin is not None:
            try:
                main_thread_pin.close()
            except Exception as error:
                cleanup.append({"child": "owned-main-thread-affinity-handle", "error": repr(error)})
        if trace_config is not None and not trace_stopped and (Path(trace_config["out_dir"]) / "capture-ready.json").exists():
            try:
                stop_scheduler(trace_config, job["id"])
            except Exception as error:
                cleanup.append({"child": "owned-scheduler-recorder", "error": repr(error)})
        if thread_sampler is not None:
            try:
                thread_sampler.stop()
            except Exception as error:
                cleanup.append({"child": "owned-thread-snapshot-sampler", "error": repr(error)})
        if activity is not None:
            try:
                activity.stop()
            except Exception as error:
                cleanup.append({"child": "owned-process-counter-sampler", "error": repr(error)})
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as error:
                cleanup.append({"child": "telemetry", "error": repr(error)})
        for child in reversed(children):
            if not child.record.get("process_exited_verified"):
                try:
                    child.stop(cleanup=True)
                except Exception as error:
                    cleanup.append({"child": child.name, "error": repr(error)})
        row["cleanup_errors"] = cleanup
        row["go_issued"] = (directory / "go.json").is_file()
        process_receipts = [run.read(path) for path in directory.glob("*/process.json")]
        row["all_created_processes_exited"] = bool(process_receipts) and all(
            item.get("process_exited_verified") is True for item in process_receipts)
        run.save(directory / "execution.json", row)
    return row


def startup_retry_eligible(row):
    """A failed timed observation can never be retried or omitted as startup."""
    return (row.get("status") == "failed" and row.get("go_issued") is False
            and row.get("all_created_processes_exited") is True
            and row.get("cleanup_errors") == [])


def collect(args):
    out = args.out.resolve()
    manifest = load_manifest(out)
    jobs = [job for job in manifest["jobs"] if job["split"] == args.split]
    _require(not (out / (args.split + "-records.json")).exists(), "split is already collected")
    if args.split == "profile":
        _require(not (out / "heldout-barrier.json").exists(), "cannot collect profile after heldout barrier")
    else:
        _require(args.frozen is not None, "heldout requires a frozen model")
        from profiling.state_response.analysis import load_frozen
        program = run.read(manifest["programs"][jobs[0]["id"]]["path"])
        frozen = load_frozen(program, args.frozen)
        _require(frozen["context"] == manifest["context"], "frozen model and campaign context differ")
        profile = run.read(out / "profile-records.json")
        barrier = {"schema": "state-response-heldout-barrier-1", "created_at_utc": run.utc(),
                   "frozen_artifact_sha256": frozen["artifact_sha256"], "frozen_file_sha256": run.sha(args.frozen),
                   "profile_record_sha256s": sorted(item["record_sha256"] for item in profile),
                   "manifest_sha256": run.sha(out / "manifest.json")}
        run.save(out / "heldout-barrier.json", barrier)
    rows, records = [], []
    with run.measurement_lock():
        for job in jobs:
            for attempt in range(manifest["protocol"]["maximum_pre_go_startup_retries"] + 1):
                verify_sources(manifest)
                print(json.dumps({"event": "starting", "job": job["id"], "attempt": attempt}), flush=True)
                try:
                    row = execute_job(out, manifest, job, attempt=attempt)
                except Exception:
                    attempt_id = job["id"] + (f"-startup-retry{attempt}" if attempt else "")
                    path = out / "jobs" / attempt_id / "execution.json"
                    row = run.read(path) if path.exists() else {"status": "failed", "attempt_id": attempt_id}
                    rows.append(row)
                    run.save(out / (args.split + f"-retained-failure-{len(rows)}.json"), row)
                    if attempt < manifest["protocol"]["maximum_pre_go_startup_retries"] and startup_retry_eligible(row):
                        print(json.dumps({"event": "retained_startup_failure_retry", "job": job["id"],
                                          "attempt": attempt, "evidence": str(path)}), flush=True)
                        continue
                    raise
                rows.append(row)
                break
            records.append(run.read(row["normalized_path"]))
            print(json.dumps({"event": "completed", "job": job["id"], "status": row["status"]}), flush=True)
            verify_sources(manifest)
    run.save(out / (args.split + "-processes.json"), rows)
    run.save(out / (args.split + "-records.json"), records)
    return {"status": "collected", "split": args.split, "processes": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--out", type=Path, required=True)
    setup.add_argument("--binary", type=Path, required=True)
    setup.add_argument("--runtime-dir", type=Path, action="append", required=True)
    setup.add_argument("--cuda-runtime", type=Path, required=True)
    setup.add_argument("--seed", type=int, default=20261021)
    setup.add_argument("--repeats", type=int, default=3)
    setup.add_argument("--memory-bytes", type=int, default=64 << 20)
    setup.add_argument("--memory-index", type=int, choices=range(4),
                       help="Optional independently frozen allocation cohort; limits peak offline analysis memory")
    setup.add_argument("--profile-hold-ms", type=int, default=2000)
    setup.add_argument("--profile-state-holds-ms", type=int, nargs=4,
                       help="Optional holds for CPU/GPU states 00,01,10,11; all raw samples retained")
    setup.add_argument("--settle-ms", type=int, default=150)
    setup.add_argument("--heldout-cycles", type=int, default=4)
    setup.add_argument("--main-affinity-mask", type=int, default=5)
    setup.add_argument("--main-thread-affinity-mask", type=int,
                       help="optional explicit single CPU within the main process mask; recorded in frozen context")
    setup.add_argument("--cpu-worker-affinity-mask", type=int, default=5)
    setup.add_argument("--gpu-worker-affinity-mask", type=int, default=48)
    record = sub.add_parser("collect")
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--split", choices=("profile", "heldout"), required=True)
    record.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args) if args.command == "prepare" else collect(args)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
