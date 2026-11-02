"""Explicit H2D primitive diagnosis; no automatic fitting or hardware launch.

The original GGML action is retained as a control. A separately named CUDA
primitive exposes copy/inner-sync/extra-sync responses. CUDA events are a
third observer boundary, not exact DMA timestamps or a second additive cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess

from profiling.dynamic_validation import idle_campaign as base

ROOT = base.ROOT
read, write, sha, receipt, require = base.read, base.write, base.sha, base.receipt, base.require
MODES = ("ggml_bare", "primitive_stamped", "primitive_events")
SIZES = {"profile": (131072, 524288, 2097152), "heldout": (262144, 1048576)}
IDLES = {"profile": (0, 20), "heldout": (2, 10)}


def balanced_orders(width, reverse=False):
    require(width in (2, 3, 6) and type(reverse) is bool, "explicit balanced width/order required")
    rows = [[(trial + slot) % width for slot in range(width)] for trial in range(12)]
    if reverse:
        rows = [list(reversed(row)) for row in reversed(rows)]
    require(all(Counter(row[slot] for row in rows) == Counter({v: 12 // width for v in range(width)})
                for slot in range(width)), "position balance failed")
    return rows


def protocol():
    return dict(
        primary_goal="Identify stage-local parameters, or refuse them; never compensate a target total with its residual",
        profile_processes=4, heldout_processes=4, fresh_contexts_per_memory_per_split=2,
        memory_modes=["registered", "pageable"], split_sizes_bytes=SIZES, split_idle_ms=IDLES,
        modes=list(MODES), trials=12, copies_per_burst=16, first_copy_separate=True,
        suffix_ordinals=list(range(1, 16)), early_trials=list(range(6)), late_trials=list(range(6, 12)),
        independent_heldout_program_seed=True, complete_original_prime_before_every_burst=True,
        state_queries="before full prime and after the completed burst; none after prime or within burst",
        query_effect_still_possible=True, within_burst_link_state_known=False,
        prior_link_snapshot_is_available_bandwidth=False, event_span_is_pure_DMA=False,
        event_clock_aligned_to_QPC=False, GPU_span_plus_host_wait_is_additive=False,
        original_GGML_control_is_replaced=False, fit_uses_target_total=False,
        acquisition_authorization_required_for_each_split=True, retries=0, missing_state_is_zero=False,
        acceptance=dict(
            all_numeric_and_copy_range_oracles=True, all_source_runtime_and_clock_receipts=True,
            host_QPC_partition_max_error_ns=1,
            observer_relative_difference_limit=.05,
            observer_comparisons=["primitive_stamped vs ggml_bare", "primitive_events vs primitive_stamped"],
            observer_metrics=["first_mean_ns", "suffix15_mean_ns", "burst_per_copy_mean_ns"],
            observer_scope="each exact size/idle/memory/fresh-process group; no selective averaging of failed cells",
            profile_early_late_suffix_relative_difference_limit=.10,
            profile_fresh_process_suffix_range_relative_limit=.10,
            phase_parameter_identifiability="require passed observer and base/state support before exporting any phase price",
            heldout_mean_relative_error_limit=.20, heldout_measured_population_cv_limit=.10,
            heldout_first_and_burst_reported_separately=True,
            unknown_or_failed_gates_stop_without_extra_cells=True),
        next_model_contract=dict(
            owner="modeling existing transfer_response/submission_graph/common service framework",
            profiling_outputs="individual completed API/sync response labels; optional qualified adjacent-event span; measured state and query age",
            host_response_is_CPU_active=False, ready_wait_durations_known=False,
            event_D2H_or_H2D_service_migration="only after observer neutrality; otherwise diagnostic only",
            fitting="root-selected interpretable mechanism; no layer/strategy IDs, DNN or total-target residual coefficient",
            independent_acceptance="freeze model coefficients and complete heldout prediction records before heldout collection"))


def original_stream_contract(llama_root):
    """Source evidence is not claimed to be dynamic stream observation."""
    lr = Path(llama_root).resolve()
    cuda = lr / "ggml/src/ggml-cuda/ggml-cuda.cu"
    backend = lr / "ggml/src/ggml-backend.cpp"
    text, generic = cuda.read_text(encoding="utf-8"), backend.read_text(encoding="utf-8")
    start = text.find("static void ggml_backend_cuda_buffer_set_tensor(")
    end = text.find("static void ggml_backend_cuda_buffer_get_tensor(", start)
    require(0 <= start < end, "pinned CUDA buffer-set function missing")
    function = text[start:end]
    copy = "cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread)"
    sync = "cudaStreamSynchronize(cudaStreamPerThread)"
    require(function.count(copy) == 1 and function.count(sync) == 1 and function.index(copy) < function.index(sync),
            "original H2D stream/inner synchronization changed")
    host_path = "if (ggml_backend_buffer_is_host(src->buffer)) {\n        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));"
    require(host_path in generic, "original host-buffer tensor_copy dispatch changed")
    return dict(cuda_source=receipt(cuda), backend_source=receipt(backend),
                selected_primitive_stream="cudaStreamPerThread", bare_internal_stream_dynamically_observed=False,
                scope="source-bound explicit replay of two inner APIs plus retained extra sync calls; not GGML action identity")


def prepare(out, binary, *, llama_root, nvml_dll):
    out, binary, dll = Path(out).resolve(), Path(binary).resolve(), Path(nvml_dll).resolve()
    require(not out.exists() and binary.is_file(), "new directory and explicit compiled v7 binary required")
    stream = original_stream_contract(llama_root)
    parent_path = ROOT / "build/dynamic-validation-20261026/idle-v2b/manifest.json"
    parent = read(parent_path)
    programs = {s: parent["programs"][s] for s in SIZES}
    require(programs["profile"]["program_sha256"] != programs["heldout"]["program_sha256"], "unseen heldout buffer seed required")
    for b in programs.values():
        for key in ("program", "reference"):
            require(sha(b[key]["path"]) == b[key]["sha256"], "retained mathematical input changed")
    here = Path(__file__).parent
    sources = [Path(__file__), here / "h2d_v7_analysis.py", here / "native/v7/h2d_primitive.cpp",
               here / "native/v7/CMakeLists.txt", here / "native/v3/common.hpp", here / "native/owned_timer.hpp",
               ROOT / "profiling/link_calibration/native/pcie_nvml.hpp",
               ROOT / "profiling/native/steady_costs/probe.cpp", ROOT / "profiling/native/steady_costs/components.hpp",
               ROOT / "profiling/native/physical_stages/cupti_capture.hpp", here / "idle_campaign.py",
               ROOT / "profiling/state_calibration/run.py", ROOT / "profiling/dac_validation/e2a.py"]
    sources += list((ROOT / "profiling/native/steady_costs/build-release-v1").glob("*.inc"))
    out.mkdir(parents=True); archived = []
    for path in sources:
        target = out / "executed_sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
        archived.append(dict(source=receipt(path), archived=receipt(target)))
    target = out / "executed_binary" / binary.name;target.parent.mkdir();shutil.copy2(binary, target)
    jobs = []
    for split in SIZES:
        for memory in ("registered", "pageable"):
            for round_id in range(2):
                sampling = base.e2a.request("action", trials=12, target=programs[split]["target_action_index"])
                sampling["repeats"] = 16
                r = dict(schema="dynamic-h2d-primitive-request-7", source_sha256=sha(here / "native/v7/h2d_primitive.cpp"),
                         original_CUDA_source_sha256=stream["cuda_source"]["sha256"], original_backend_source_sha256=stream["backend_source"]["sha256"],
                         split=split, sampling=sampling, memory_mode=memory, expected_process_mask=4294967295,
                         sizes_bytes=list(SIZES[split]), idle_ms=list(IDLES[split]), modes=list(MODES),
                         size_orders=balanced_orders(len(SIZES[split]), bool(round_id)), condition_orders=balanced_orders(6, bool(round_id)),
                         prior_query_placement="before_full_prime", post_queries_are_predictor_input=False, nvml_dll_path=str(dll))
                jobs.append(dict(id=f"{split}-{memory}-r{round_id}", split=split, memory_mode=memory, round=round_id, request=r))
    m = dict(schema="dynamic-h2d-primitive-campaign-7", prepared_at_utc=base.e2a.utc(), parent_manifest=receipt(parent_path),
             binary=receipt(target), sources=archived, original_stream_contract=stream,
             nvml_module=receipt(dll) if dll.is_file() else dict(path=str(dll), sha256=None, status="unknown_unavailable"),
             runtime=parent["runtime"], runtime_dirs=parent["runtime_dirs"], programs=programs, jobs=jobs, protocol=protocol())
    write(out / "manifest.json", m)
    return dict(manifest=receipt(out / "manifest.json"), profile_processes=4, heldout_processes=4, hardware_executed=False)


def authorization(out, split):
    require(split in SIZES, "unknown split")
    m = read(out / "manifest.json"); auth = read(out / f"{split}-authorization.json")
    require(auth.get("collection_authorized") is True and auth.get("manifest_sha256") == sha(out / "manifest.json")
            and auth.get("split") == split, "root must authorize this exact split and manifest")
    if split == "heldout":
        barrier = read(out / "frozen-barrier.json")
        require(auth.get("frozen_barrier_sha256") == sha(out / "frozen-barrier.json"), "heldout authorization not bound to freeze")
        require(barrier.get("schema") == "dynamic-h2d-phase-freeze-7" and barrier.get("manifest_sha256") == sha(out / "manifest.json"), "phase-model freeze missing")
        require(barrier.get("profile_gate_passed") is True and barrier.get("parameters_frozen") is True
                and barrier.get("heldout_predictions_frozen") is True and barrier.get("fit_uses_whole_target_total") is False,
                "identifiable profile phases and prior frozen predictions required")
        require(isinstance(barrier.get("files"), dict) and bool(barrier["files"]), "explicit frozen model/prediction files required")
        for name, value in barrier["files"].items():
            path = Path(name); require(not path.is_absolute() and ".." not in path.parts, "freeze files must stay in campaign")
            require(sha(out / path) == value, "frozen model/prediction file changed")
    return m


def verify(native, request, binding, pid):
    from .h2d_v7_analysis import validate_native
    require(native["request"] == request and native["process_id"] == pid, "native request/process differs")
    result = validate_native(native)
    require(native["program_sha256"] == binding["program_sha256"], "mathematical program differs")
    import numpy as np
    oracle, ports = read(binding["reference"]["path"]), native["verification_native"]["ports"]
    require(set(oracle) == set(ports), "independent port inventory differs")
    for name in oracle:
        left, right = np.asarray(ports[name]), np.asarray(oracle[name])
        require(left.shape == right.shape and np.all(np.isfinite(left)) and np.allclose(left, right, rtol=5e-5, atol=5e-5), "independent numerical mismatch " + name)
    return dict(result, independent_numerical_ports=True, performance_parameters_fitted=False)


def collect(out, split):
    out = Path(out).resolve();m = authorization(out, split)
    for source in m["sources"]:
        for key in ("source", "archived"):
            require(sha(source[key]["path"]) == source[key]["sha256"], "bound source changed")
    resources = [m["binary"], *m["runtime"], m["original_stream_contract"]["cuda_source"], m["original_stream_contract"]["backend_source"]]
    if m["nvml_module"].get("sha256"):
        resources.append(m["nvml_module"])
    for item in resources:
        require(sha(item["path"]) == item["sha256"], "bound input/runtime changed")
    for binding in m["programs"].values():
        for key in ("program", "reference"):
            require(sha(binding[key]["path"]) == binding[key]["sha256"], "bound program/reference changed")
    marker = out / f"{split}-started.json";require(not marker.exists(), "no retries or repeated split acquisition")
    write(marker, dict(started_at_utc=base.e2a.utc(), split=split));executions = []
    with base.run.measurement_lock():
        for job in (j for j in m["jobs"] if j["split"] == split):
            folder = out / "jobs" / job["id"];folder.mkdir(parents=True, exist_ok=False)
            write(folder / "request.json", job["request"]);binding = m["programs"][split]
            command = [m["binary"]["path"], binding["program"]["path"], str(folder / "request.json"), str(folder / "result.json")]
            record = dict(job_id=job["id"], split=split, status="failed", command=command, model_fitted=False,
                          in_burst_telemetry=False, declared_whole_split_jobs=4)
            process, handles = None, []
            try:
                env = base.run.child_environment(m["runtime_dirs"]);env["OPENBLAS_NUM_THREADS"] = "1"
                process, stdout, stderr, affinity = base.run.launch(command, env, folder / "stdout.txt", folder / "stderr.txt", 4294967295)
                handles = [stdout, stderr];record.update(process_id=process.pid, process_affinity=affinity)
                code = process.wait(timeout=180);record.update(returncode=code, process_exited_verified=True)
                require(code == 0, "native failed; no retry")
                record["verification"] = verify(read(folder / "result.json"), job["request"], binding, process.pid)
                record.update(status="completed", native=receipt(folder / "result.json"))
            except BaseException as error:
                record["error"] = repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
                if process is not None:
                    record.update(returncode=process.returncode, process_exited_verified=process.poll() is not None)
            finally:
                for handle in handles: handle.close()
                write(folder / "execution.json", record)
            executions.append(record)
            print(json.dumps(dict(job_id=job["id"], status=record["status"], error=record.get("error"))), flush=True)
            if record["status"] != "completed": break
    write(out / f"{split}-executions.json", executions)
    return dict(declared=4, acquired=len(executions), completed=sum(x["status"] == "completed" for x in executions),
                not_run=4-len(executions), performance_parameters_fitted=False)


def main():
    p = argparse.ArgumentParser();p.add_argument("command", choices=("prepare", "collect", "protocol"));p.add_argument("--out")
    p.add_argument("--binary");p.add_argument("--llama-root");p.add_argument("--nvml-dll");p.add_argument("--split", choices=tuple(SIZES))
    a = p.parse_args()
    if a.command == "protocol": result = protocol()
    elif a.command == "prepare": result = prepare(a.out, a.binary, llama_root=a.llama_root, nvml_dll=a.nvml_dll)
    else: result = collect(a.out, a.split)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
