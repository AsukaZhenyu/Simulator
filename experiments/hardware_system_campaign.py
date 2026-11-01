"""Predeclared hardware-system strategy comparisons with hypothetical costs.

Product identities/capacity choices have source links in the catalogue. All
effective execution rates remain author-selected analytical assumptions. A
paired NPU proxy substitutes costs in the existing GPU slot; it is not an NPU
backend, F32 operator-support validation, or CPU+GPU+NPU concurrency experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hardware_sensitivity import stable_hash, write_csv, write_json
from experiments.hardware_system_catalogue import (
    SYSTEMS, NPU_TIME_MULTIPLIERS, hardware_for, system_rows, systems_by_id,
)
from experiments.strategy_comparison import frozen_evidence, traffic_metrics
from experiments.strategy_comparison_campaign import verify_row_metrics
from experiments.strategy_comparison_metrics import timeline_metrics
from llm_simulator import AnalyticalCosts, InvalidInput, Simulator, Unsupported

WARNING = ("Uncalibrated analytical comparisons and hypothetical GPU-derived NPU "
           "costs; not measured hardware performance or native NPU support.")
FORMAT_VERSION = "hardware-system-campaign-0.1"
REFERENCE = "e0-s0"
_CONTROL_FIELDS = (
    "success", "status", "error_code", "termination_reason",
    "completed_makespan_ns", "h2d_bytes", "d2h_bytes", "ssd_read_bytes",
    "ssd_write_bytes", "peak_dram_bytes", "peak_vram_bytes", "peak_shared_bytes",
    "event_sha256", "event_counts",
)


def _protocol_module():
    from experiments import hardware_system_protocol
    return hardware_system_protocol


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_source_manifest():
    """Freeze executed library sources and every imported campaign helper."""
    paths = set()
    for directory in ("src/llm_simulator", "modeling/llm_infer_model", "mapping/tensor_mapping"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update(path for path in (ROOT / "src/llm_simulator/data").rglob("*") if path.is_file())
    for name in ("hardware_sensitivity.py", "strategy_comparison.py",
                 "strategy_comparison_metrics.py", "strategy_comparison_campaign.py",
                 "strategy_comparison_scenarios.py", "strategy_comparison_analysis.py",
                 "hardware_system_catalogue.py", "hardware_system_protocol.py",
                 "hardware_system_campaign.py"):
        paths.add(ROOT / "experiments" / name)
    return {path.relative_to(ROOT).as_posix(): sha_file(path) for path in sorted(paths)}


@dataclass(frozen=True)
class CaseSpec:
    system_id: str
    workload_id: str
    candidate_id: str
    accelerator: str = "gpu"
    npu_time_multiplier: float | None = None

    def __post_init__(self):
        for name in ("system_id", "workload_id", "candidate_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(char in value for char in "/\\\r\n"):
                raise ValueError(f"{name} must be a nonempty path component")
        if self.accelerator not in ("gpu", "npu_proxy"):
            raise ValueError("accelerator must be gpu or npu_proxy")
        if self.accelerator == "gpu" and self.npu_time_multiplier is not None:
            raise ValueError("GPU baseline must not have a proxy multiplier")
        if self.accelerator == "npu_proxy" and (
                isinstance(self.npu_time_multiplier, bool)
                or self.npu_time_multiplier not in NPU_TIME_MULTIPLIERS):
            raise ValueError("proxy multiplier must be a declared sensitivity point")

    @property
    def variant_id(self):
        return ("gpu" if self.accelerator == "gpu" else
                "npu-proxy-a" + format(self.npu_time_multiplier, "g").replace(".", "p"))

    @property
    def key(self):
        return self.system_id, self.variant_id, self.workload_id, self.candidate_id


def _workloads():
    return {spec.id: spec for spec in _protocol_module().workload_specs()}


def physical_cache_budget(hardware):
    """Bind the compiler's required positive quota to physical availability.

    Enabled prefetch requires an explicit positive budget in the current IR.
    Using the full available accelerator pool for BOTH storage candidates adds
    no smaller software quota. Live activations, workspaces and UMA allocations
    remain constrained by the ordinary runtime physical-capacity checks.
    """
    value = (hardware.vram_capacity_bytes if hardware.memory_mode == "separate" else
             hardware.shared_capacity_bytes - hardware.shared_runtime_reserved_bytes)
    if type(value) is not int or value < 1:
        raise ValueError("experiment needs positive available accelerator memory")
    return value


def declared_jobs(*, system_ids=None, proxy_system_ids=(), workload_ids=None, candidates=None):
    """Enumerate the full factorial before any result is observed."""
    protocol = _protocol_module()
    systems = systems_by_id()
    system_ids = tuple(systems) if system_ids is None else tuple(system_ids)
    proxy_system_ids = tuple(proxy_system_ids)
    specs = _workloads()
    workload_ids = tuple(specs) if workload_ids is None else tuple(workload_ids)
    for values, available, name in ((system_ids, systems, "system"),
                                    (workload_ids, specs, "workload")):
        if not values or len(set(values)) != len(values) or set(values) - set(available):
            raise ValueError(f"unknown, empty or repeated {name} selection")
    if (len(set(proxy_system_ids)) != len(proxy_system_ids)
            or set(proxy_system_ids) - set(system_ids)):
        raise ValueError("proxy systems must be unique selected GPU baseline systems")
    if any(not systems[identity].npu_proxy_eligible for identity in proxy_system_ids):
        raise ValueError("proxy experiment requires a catalogue NPU-eligible system")
    if candidates is not None and (not candidates or len(set(candidates)) != len(candidates)):
        raise ValueError("candidate selection must be nonempty and unique")
    jobs = []
    for identity in system_ids:
        variants = (("gpu", None),) + (tuple(("npu_proxy", alpha) for alpha in NPU_TIME_MULTIPLIERS)
                                            if identity in proxy_system_ids else ())
        for accelerator, alpha in variants:
            for workload_id in workload_ids:
                spec = specs[workload_id]
                available = protocol.candidate_ids(spec.model_kind)
                selected = available if candidates is None else tuple(candidates)
                if set(selected) - set(available):
                    raise ValueError(f"undeclared candidate for {spec.model_kind}")
                jobs.extend(CaseSpec(identity, workload_id, candidate, accelerator, alpha)
                            for candidate in selected)
    return tuple(jobs)


def _materialize(case):
    protocol = _protocol_module()
    spec = _workloads()[case.workload_id]
    model, workload = protocol.make_workload(spec)
    hardware = hardware_for(case.system_id)
    scheduling = protocol.candidate_spec(model.kind, case.candidate_id,
                                        cache_budget_bytes=physical_cache_budget(hardware))
    cost_model = (AnalyticalCosts(workspace_bytes=0) if case.accelerator == "gpu" else
                  protocol.NPUProxyCosts(alpha=case.npu_time_multiplier, workspace_bytes=0))
    return model, hardware, workload, scheduling, cost_model


def _physical_peaks(row, hardware):
    if hardware.memory_mode == "shared":
        return {"shared": row.get("peak_shared_bytes")}
    return {"dram": row.get("peak_dram_bytes"), "vram": row.get("peak_vram_bytes")}


def round_compute_boundaries(events, operations, global_document):
    """Read declared invocation ownership; never infer token output latency.

    These are first/last compute timestamps and compute duration sums per
    declared round. They exclude final copies, token sampling and any unmodeled
    output processing, and therefore are not TTFT/TPOT or full round latencies.
    """
    invocation_round = {}
    by_round = {entry["id"]: [] for entry in global_document["rounds"]}
    for entry in global_document["rounds"]:
        for invocation in entry["invocations"]:
            if invocation["id"] in invocation_round:
                raise ValueError("duplicate invocation ownership in round catalogue")
            invocation_round[invocation["id"]] = entry["id"]
    operation_round = {operation.id: invocation_round.get(operation.invocation_id)
                       for operation in operations}
    for event in events:
        if event.kind != "COMPUTE":
            continue
        owner = operation_round.get(event.target_id)
        if owner is None:
            raise ValueError("compute event has no declared invocation/round owner")
        by_round[owner].append(event)
    return [dict(round_id=round_id, compute_count=len(group),
        first_compute_start_ns=min(event.start_ns for event in group),
        last_compute_completion_ns=max(event.end_ns for event in group),
        cpu_compute_duration_sum_ns=sum(event.end_ns-event.start_ns for event in group if event.device == "cpu"),
        accelerator_compute_duration_sum_ns=sum(event.end_ns-event.start_ns for event in group if event.device == "gpu"),
        boundary="compute timestamps only; excludes copy/output completion; not TTFT or TPOT")
        for round_id, group in by_round.items() if group]


def verify_case(row):
    verify_row_metrics(row)
    if not row.get("success"):
        return
    if row["memory_mode"] == "shared" and (row["h2d_bytes"] or row["d2h_bytes"]):
        raise ValueError("shared visibility unexpectedly generated a physical host/device copy")
    if row.get("share_count", 0) and row["memory_mode"] != "shared":
        raise ValueError("separate-memory run unexpectedly contains SHARE")
    for key in ("cpu_event_fraction", "accelerator_event_fraction",
                "cpu_accelerator_overlap_fraction", "copy_compute_overlap_fraction"):
        if not 0 <= row[key] <= 1:
            raise ValueError("invalid event occupancy fraction")


def run_case(case, output, retain_traces=False):
    """Run one fresh runtime and retain exact declared inputs and causal events."""
    case = CaseSpec(**case) if isinstance(case, dict) else case
    folder = Path(output) / "cases" / Path(*case.key)
    folder.mkdir(parents=True, exist_ok=False)
    model, hardware, workload, scheduling, costs = _materialize(case)
    row = dict(**asdict(case), variant_id=case.variant_id, model_kind=model.kind,
               memory_mode=hardware.memory_mode, prefill_tokens=workload.prefill_tokens,
               decode_steps=workload.decode_steps, calibrated=False,
               hardware_measurement=False, numerical_llm_execution=False,
               native_npu_backend=False, npu_operator_support_validated=False,
               gpu_npu_concurrency=False, success=False, completed_makespan_ns=None,
               accelerator_runtime_slot="gpu", reference_candidate_id=REFERENCE,
               cache_budget_bytes=physical_cache_budget(hardware),
               cache_budget_policy="full available accelerator physical capacity; same bound for s0/s1",
               latency_boundary="finite cold execution, including first weight preparation and mandatory goal copies",
               effective_cost_provenance="author_selected_analytical_assumptions",
               artifact_directory=folder.relative_to(output).as_posix(),
               unavailable_metrics={"energy": "not modeled", "ttft": "no output boundary extraction",
                   "tpot": "no output boundary extraction", "cache_hit_rate": "no demand-residency audit",
                   "causal_wait_breakdown": "event spans are not causal wait attribution"})
    declarations = dict(model=asdict(model), hardware=asdict(hardware), workload=asdict(workload),
                        scheduling=asdict(scheduling), costs=asdict(costs), experiment=asdict(case))
    row["workload_hardware_sha256"] = stable_hash({key: declarations[key]
                                                for key in ("model", "hardware", "workload")})
    row["declared_inputs_sha256"] = stable_hash(declarations)
    row["strategy_sha256"] = stable_hash(asdict(scheduling))
    write_json(folder / "declarations.json", declarations)
    started = time.perf_counter()
    try:
        simulator = Simulator(model=model, hardware=hardware, workload=workload, cost_model=costs)
        with simulator.compile(scheduling=scheduling, directory=folder / "inputs") as compiled:
            ir = compiled.ir_document()
            write_json(folder / "compiled-ir.json", ir)
            row["compiled_ir_sha256"] = stable_hash(ir)
            result = compiled.run()
            row.update(result.summary())
            row.pop("input_directory", None)
            events = [asdict(event) for event in result.events]
            write_json(folder / "events.json", events)
            row["event_sha256"] = stable_hash(events)
            row["event_counts"] = dict(Counter(event.kind for event in result.events))
            row["share_count"] = row["event_counts"].get("SHARE", 0)
            row["physical_memory_peaks_bytes"] = _physical_peaks(row, hardware)
            observation = result.raw[1].observe()
            write_json(folder / "operation-catalogue.json", [asdict(op) for op in observation.operations])
            write_json(folder / "tensor-catalogue.json", [asdict(tensor) for tensor in observation.tensors])
            write_json(folder / "policy-audit.json", result.raw[2].audit_document())
            write_json(folder / "decision-log.json", [dict(index=decision.index,
                t_ns=decision.observation.t_ns, observation_sha256=decision.observation_sha256,
                decision=asdict(decision.decision) if decision.decision is not None else None,
                accepted=decision.accepted, error_code=decision.error_code,
                error_message=decision.error_message) for decision in result.raw[0].decisions])
            if result.success:
                global_document = json.loads((compiled.directory / "model.global.json").read_text(encoding="utf-8"))
                boundaries = round_compute_boundaries(result.events, observation.operations, global_document)
                write_json(folder / "round-compute-boundaries.json", boundaries)
                row["round_compute_boundaries"] = boundaries
                row["completed_makespan_ns"] = result.makespan_ns
                row.update(timeline_metrics(result.events, result.makespan_ns))
                row.update(traffic_metrics(result))
                row.update(accelerator_event_span_ns=row["gpu_event_span_ns"],
                           cpu_accelerator_event_overlap_ns=row["cpu_gpu_event_overlap_ns"])
                for key, numerator in (("cpu_event_fraction", "cpu_event_span_ns"),
                        ("accelerator_event_fraction", "gpu_event_span_ns"),
                        ("cpu_accelerator_overlap_fraction", "cpu_gpu_event_overlap_ns"),
                        ("copy_compute_overlap_fraction", "copy_compute_event_overlap_ns")):
                    row[key] = row[numerator] / result.makespan_ns
            else:
                row["diagnostic_partial_makespan_ns"] = result.makespan_ns
            if retain_traces:
                result.save(folder / "full-audit")
    except (Unsupported, InvalidInput) as error:
        # Expected inability to compile/fit is a result, never a timed success.
        # Unexpected Python/runtime errors propagate out of the worker.
        row.update(status="unsupported_or_invalid_input", success=False,
                   completed_makespan_ns=None, error_type=type(error).__name__, error_message=str(error))
    row["harness_wall_seconds"] = time.perf_counter() - started
    verify_case(row)
    write_json(folder / "summary.json", row)
    write_json(folder / "artifacts.json", {path.relative_to(folder).as_posix(): sha_file(path)
        for path in sorted(folder.rglob("*")) if path.is_file()})
    return row


def verify_controls(records):
    """Equal-cost proxy and all-CPU controls compare actual event sequences."""
    indexed = {(row["system_id"], row["variant_id"], row["workload_id"], row["candidate_id"]): row
               for row in records}
    controls = []
    for row in records:
        if row["accelerator"] != "npu_proxy":
            continue
        equal_proxy = row["npu_time_multiplier"] == 1
        all_cpu = row["candidate_id"].startswith("e1-")
        if not equal_proxy and not all_cpu:
            continue
        baseline_key = (row["system_id"], "gpu", row["workload_id"], row["candidate_id"])
        reference = indexed.get(baseline_key)
        if reference is None:
            raise ValueError("proxy control is missing its GPU baseline")
        same = all(row.get(field) == reference.get(field) for field in _CONTROL_FIELDS)
        controls.append(dict(system_id=row["system_id"], workload_id=row["workload_id"],
            candidate_id=row["candidate_id"], variant_id=row["variant_id"],
            control="equal_cost_proxy" if equal_proxy else "all_cpu_independent_of_proxy",
            equivalent=same))
        if not same:
            raise ValueError(f"proxy control differs from baseline: {row.get('artifact_directory', row['variant_id'])}")
    return controls


def audit_coverage(records, jobs):
    expected = {case.key for case in jobs}
    actual = [(row["system_id"], row["variant_id"], row["workload_id"], row["candidate_id"])
              for row in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("case coverage differs from predeclared matrix")
    for row in records:
        verify_case(row)
    grouped = {}
    for row in records:
        grouped.setdefault((row["system_id"], row["workload_id"]), set()).add(row["workload_hardware_sha256"])
    if any(len(hashes) != 1 for hashes in grouped.values()):
        raise ValueError("strategies/proxy variants did not receive identical model and hardware inputs")


def catalogue_document(jobs):
    protocol = _protocol_module()
    return dict(format_version=FORMAT_VERSION, warning=WARNING,
        cost_model="analytical_roofline_with_explicit_proxy_sensitivity",
        calibrated=False, hardware_measurement=False, native_npu_backend=False,
        proxy_formula="ceil(alpha * generated GPU task duration), alpha in {0.5,1,2}; CPU/transfer unchanged",
        reference_candidate_id=REFERENCE, cache_budget_bytes=None,
        cache_budget_provenance=("compiler requires positive budget for prefetch; BOTH s0/s1 instantiate "
                                 "the same full available accelerator physical-capacity bound, not a smaller software quota"),
        cache_budget_by_system={system.id: physical_cache_budget(hardware_for(system)) for system in SYSTEMS},
        workspace_bytes=0, workspace_provenance="simplifying declaration, not a measured zero",
        initial_state="cold immutable F32 weights/input in DRAM; empty KV; new runtime per case",
        final_goals="mandatory hidden/KV accessible in DRAM; optional cached views may differ",
        shared_memory_assumption="ideal aliasable buffers; no automatic shared-bandwidth contention or format/sync cost",
        statistics="deterministic analytical cases, not repeated timing samples",
        event_metrics="recorded action-span unions/intersections, not measured hardware utilization or causal waits",
        capacity_unit_convention="catalogue vendor GB options represented as GiB; transfer GB/s decimal",
        system_catalogue=system_rows(),
        workloads=[dict(spec=asdict(spec), model=asdict(model), workload=asdict(workload))
                   for spec in protocol.workload_specs() for model, workload in (protocol.make_workload(spec),)],
        strategies_by_system={system.id: {kind: {candidate: asdict(protocol.candidate_spec(kind, candidate,
                      cache_budget_bytes=physical_cache_budget(hardware_for(system))))
                     for candidate in protocol.candidate_ids(kind)} for kind in ("dense", "moe")}
                     for system in SYSTEMS},
        jobs=[asdict(case) for case in jobs], declared_attempts=len(jobs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--systems", nargs="+")
    parser.add_argument("--proxy-systems", nargs="+", default=())
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--retain-traces", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    try:
        jobs = declared_jobs(system_ids=args.systems, proxy_system_ids=args.proxy_systems,
                             workload_ids=args.workloads, candidates=args.candidates)
    except ValueError as error:
        parser.error(str(error))
    output = args.out.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--out must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    sources_before = execution_source_manifest()
    evidence_before = frozen_evidence()
    protocol = catalogue_document(jobs)
    protocol.update(execution_sources_sha256=stable_hash(sources_before),
                    frozen_accuracy_evidence=evidence_before, workers=args.workers,
                    worker_wall_time_is_not_simulated_latency=True, plan_only=args.plan_only)
    write_json(output / "source-manifest-before.json", sources_before)
    write_json(output / "protocol.json", protocol)
    if args.plan_only:
        print(json.dumps(dict(declared_attempts=len(jobs), protocol_sha256=stable_hash(protocol))))
        return 0
    started = time.perf_counter()
    records = []
    with (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, asdict(case), str(output), args.retain_traces): case
                       for case in jobs}
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(f"{len(records)}/{len(jobs)} {'/'.join(futures[future].key)} "
                      f"success={row['success']} T={row.get('completed_makespan_ns')}ns "
                      f"wall={time.perf_counter()-started:.1f}s", flush=True)
    ordering = {case.key: index for index, case in enumerate(jobs)}
    records.sort(key=lambda row: ordering[(row["system_id"], row["variant_id"],
                                          row["workload_id"], row["candidate_id"])])
    audit_coverage(records, jobs)
    controls = verify_controls(records)
    write_json(output / "records.json", records)
    write_csv(output / "runs.csv", records)
    write_json(output / "controls.json", controls)
    sources_after, evidence_after = execution_source_manifest(), frozen_evidence()
    write_json(output / "source-manifest-after.json", sources_after)
    if sources_before != sources_after or evidence_before != evidence_after:
        raise RuntimeError("execution sources or frozen accuracy evidence changed during campaign")
    completion = dict(format_version=FORMAT_VERSION, complete=True,
        declared_attempts=len(jobs), completed_attempts=len(records),
        successful_runs=sum(row["success"] for row in records),
        failed_runs=sum(not row["success"] for row in records),
        controls=len(controls), controls_passed=all(control["equivalent"] for control in controls),
        source_unchanged=True, accuracy_evidence_unchanged=True,
        source_manifest_sha256=stable_hash(sources_after), protocol_sha256=stable_hash(protocol),
        records_sha256=stable_hash(records), elapsed_wall_seconds=time.perf_counter()-started,
        warning=WARNING)
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
