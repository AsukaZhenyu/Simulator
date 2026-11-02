"""Freeze matching component evidence before independent full-program replay."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import statistics

from .bridge import BridgeError, SerialExecutionPolicy, _require, sha, validate_program, verify_native

GATES = dict(within_process_cv_max=0.15, between_process_median_cv_max=0.20,
             pooled_median_absolute_relative_error_max=0.20, independent_processes_min=3)


def cv(values):
    return statistics.pstdev(values)/statistics.mean(values)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def binary_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def contract_identity(program):
    # Different numerical seeds are deliberately held out. Everything affecting
    # the selected graph, dimensions, operands, locations and actions stays exact.
    return {k:v for k,v in program["identity"].items() if k != "initial_values"}


def _samples(values):
    _require(isinstance(values, list) and values and all(type(v) is int and v > 0 for v in values),
             "component durations must be positive integer observations")
    return values


def freeze_profile(program, native, *, executable_sha256, evidence, frozen_at=None):
    verification = verify_native(program, native)
    _require(native.get("mode") == "components" and not native.get("total_samples_ns"),
             "a component profile must not use heldout whole-program durations")
    actions = program["actions"]
    expected = {i for i, a in enumerate(actions) if a["kind"] in {"COMPUTE", "COPY"}}
    rows, seen = [], set()
    for item in native["components"]:
        index = item["action_index"]
        _require(type(index) is int and index in expected and index not in seen, "component action inventory differs")
        seen.add(index)
        _require(item["action"] == actions[index], "component action binding changed")
        samples = _samples(item["samples_ns"])
        _require(len(samples) == native["samples"], "component sample count differs")
        _require(type(item["workspace_bytes"]) is int and item["workspace_bytes"] >= 0, "workspace must be observed nonnegative bytes")
        rows.append(dict(action_index=index, action=item["action"], duration_ns=math.ceil(statistics.median(samples)),
                         workspace_bytes=item["workspace_bytes"], samples_ns=samples,
                         sample_cv=statistics.pstdev(samples)/statistics.mean(samples)))
    _require(seen == expected, "missing exact component profiles; no synthetic fallback")
    processes=native.get("independent_processes",[])
    _require(len(processes)>=GATES["independent_processes_min"], "component profile requires >=3 independent native processes")
    process_statistics=[];process_medians={index:[] for index in expected};collected={index:[] for index in expected}
    process_ids=set()
    for process in processes:
        pid=process["process_index"]
        _require(type(pid) is int and pid not in process_ids,"duplicate independent process identity")
        process_ids.add(pid);stats=[];indices=set()
        for row in process["components"]:
            index=row["action_index"]
            _require(index in expected and index not in indices and row["action"]==actions[index],"independent process action inventory changed")
            indices.add(index);samples=_samples(row["samples_ns"])
            _require(len(samples)>=3,"each independent process needs >=3 samples")
            value=cv(samples);median=statistics.median(samples)
            stats.append(dict(action_index=index,samples_ns=samples,median_ns=median,cv=value,
                              within_process_cv_passed=value<=GATES["within_process_cv_max"]))
            process_medians[index].append(median);collected[index].extend(samples)
        _require(indices==expected,"independent process omits component")
        process_statistics.append(dict(process_index=pid,components=stats,
            all_within_process_cv_passed=all(row["within_process_cv_passed"] for row in stats)))
    for row in rows:_require(row["samples_ns"]==collected[row["action_index"]],"pooled samples do not exactly concatenate independent process observations")
    between=[dict(action_index=index,process_medians_ns=medians,cv=cv(medians),
                  between_process_median_cv_passed=cv(medians)<=GATES["between_process_median_cv_max"])
             for index,medians in sorted(process_medians.items())]
    source_quality=all(row["all_within_process_cv_passed"] for row in process_statistics) and all(row["between_process_median_cv_passed"] for row in between)
    _require(len(executable_sha256) == 64 and all(c in "0123456789abcdef" for c in executable_sha256), "binary SHA256 required")
    ledger = dict(format_version="highlevel-accuracy-frozen-profile-v1",
        program_sha256=program["program_sha256"], identities=contract_identity(program),
        profile_seed=program["seed"], profile_initial_values_sha256=program["identity"]["initial_values"],
        executable_sha256=executable_sha256, evidence=evidence, frozen_at=frozen_at or utc_now(),
        native_component_sha256=sha(native), verification=verification, components=rows,
        independent_process_statistics=process_statistics, between_process_component_statistics=between,
        fixed_quality_gates=GATES, source_quality_passed=source_quality,
        workspace_scope=native.get("workspace_scope"), native_complete_workspace_or_memory_claimed=False,
        timing_scope=native["timing_scope"], heldout_used=False,
        provenance="measured_matching_GGML_graph_components_not_original_synthetic_prices")
    ledger["ledger_sha256"] = sha(ledger)
    return ledger


def validate_ledger(program, ledger):
    validate_program(program)
    copy = deepcopy(ledger)
    digest = copy.pop("ledger_sha256", None)
    _require(digest == sha(copy), "frozen profile changed")
    _require(ledger.get("format_version") == "highlevel-accuracy-frozen-profile-v1" and ledger.get("heldout_used") is False,
             "invalid or heldout-contaminated baseline ledger")
    _require(ledger.get("identities") == contract_identity(program),
             "frozen component profile does not exactly match this high-level program")
    return ledger


def measured_scenario(program, ledger):
    """Replace EVERY selected leaf price using only the frozen native evidence."""
    from llm_infer_model.tensor.spec import build_scenario
    validate_ledger(program, ledger)
    document = deepcopy(program["graph"]["scenario"])
    compute, transfers = {}, {}
    grouped = {}
    for item in ledger["components"]:
        a = item["action"]
        key = ("compute", a["target_id"], a["device"]) if a["kind"] == "COMPUTE" else ("copy", a["target_id"], a["src"]+"_to_"+a["dst"])
        grouped.setdefault(key, []).append(item)
    for (kind, tid, device), items in grouped.items():
        samples = [value for item in items for value in item["samples_ns"]]
        duration = math.ceil(statistics.median(samples))
        if kind == "compute":
            compute.setdefault(tid, {})[device] = dict(duration_ns=duration,
                workspace_bytes=max(item["workspace_bytes"] for item in items))
        else:
            transfers.setdefault(device, {})[tid] = dict(duration_ns=duration)
    _require(set(compute) == {b["operation_id"] for b in program["bindings"]}, "profile omits compiled graph leaves")
    document["costs"] = dict(source="highlevel-frozen-profile:"+ledger["ledger_sha256"], compute=compute, transfer=transfers)
    document["mapspace"]["allowed_devices"] = {oid: sorted(rows) for oid, rows in compute.items()}
    document["mapspace"]["allowed_transfers"] = {route: sorted(transfers.get(route, {})) for route in document["mapspace"]["allowed_transfers"]}
    return build_scenario(program["graph"]["workload"], document), document


def simulate_frozen(program, ledger):
    from llm_infer_model.tensor.engine import Action
    from llm_infer_model.tensor.program import ActionStep, execute_program
    scenario, document = measured_scenario(program, ledger)
    steps = tuple(ActionStep(Action(**row), index) for index, row in enumerate(program["actions"]))
    result = execute_program(scenario, steps)
    _require(result.evaluation.status == "valid", "frozen measured component replay is infeasible: "+result.evaluation.reason)
    return dict(status=result.evaluation.status, makespan_ns=result.evaluation.makespan_ns,
                cost_source=document["costs"]["source"], ledger_sha256=ledger["ledger_sha256"],
                components_only=True, synthetic_costs_used=False,
                peak_vram_bytes=result.evaluation.peak_vram_bytes, h2d_bytes=result.evaluation.h2d_bytes,
                memory_prediction_scope="model_logical_residents_plus_known_component_workspace; CUDA_private_scratch_unknown",
                native_memory_capacity_accuracy_claimed=False)


def run_plan_with_profile(plan, program, ledger):
    """Rebind the SAME CompiledPlan's model to frozen leaf prices and rerun policy.

    No source fixture is edited and no transform cost is divided. The clone uses
    the existing unified model/session/policy runtime. The action sequence must
    remain exactly equal, or this bounded bridge rejects the new strategy path.
    """
    from tensor_mapping.unified_strategy import create_session
    validate_ledger(program, ledger)
    _require(plan._compiled.frontend_sha256 == program["frontend_sha256"], "different compiled ModelPlan")
    _require(sha(create_session(plan._compiled).execution_document()) == program["identity"]["graph"], "compiled plan graph changed")
    _require(plan._runtime_spec is None or plan._runtime_spec.factory is SerialExecutionPolicy,
             "measured regeneration supports static policy plus explicit serial policy only")
    scenario, document = measured_scenario(program, ledger)
    aggregate = replace(plan._compiled.model.aggregate, scenario=scenario, scenario_document=document,
                        costs_source=document["costs"]["source"])
    model = replace(plan._compiled.model, aggregate=aggregate)
    cloned = replace(plan, _compiled=replace(plan._compiled, model=model))
    run = cloned.run()[0]
    _require(run.evaluation.status == "valid" and run.termination_reason == "goal_reached", "measured high-level plan did not complete")
    actions = [asdict(r.action) for r in run.decisions if r.accepted and r.action is not None]
    _require(actions == program["actions"], "measured high-level policy changed action path; recollect exact profile")
    direct = simulate_frozen(program, ledger)
    _require(run.evaluation.makespan_ns == direct["makespan_ns"], "ModelPlan and exact action replay disagree")
    return {**direct, "actual_ModelPlan_policy_rerun": True, "frontend_sha256":program["frontend_sha256"],
            "action_trace_equal": True}


def compare_heldout(program, ledger, native, *, heldout_started_at, executable_sha256, max_relative_error=None):
    validate_ledger(program, ledger)
    _require(datetime.fromisoformat(heldout_started_at) >= datetime.fromisoformat(ledger["frozen_at"]),
             "heldout must begin after baseline ledger was frozen")
    _require(executable_sha256 == ledger["executable_sha256"], "heldout executable differs from frozen profile")
    _require(program["seed"] != ledger["profile_seed"] and program["identity"]["initial_values"] != ledger["profile_initial_values_sha256"],
             "heldout numerical seed/data must be disjoint from the component profile")
    _require(native.get("mode") == "replay" and not native.get("components"), "heldout must be independent whole-program replay")
    verification = verify_native(program, native)
    samples = _samples(native["total_samples_ns"])
    _require(len(samples) == native["samples"], "heldout sample count mismatch")
    simulated = simulate_frozen(program, ledger)
    observed = statistics.median(samples)
    signed = (simulated["makespan_ns"]-observed)/observed
    if max_relative_error is not None:
        _require(isinstance(max_relative_error,(int,float)) and not isinstance(max_relative_error,bool) and math.isfinite(max_relative_error) and max_relative_error>=0,
                 "explicit nonnegative tolerance required")
    return dict(format_version="highlevel-accuracy-comparison-v1", program_sha256=program["program_sha256"],
        ledger_sha256=ledger["ledger_sha256"], frontend_sha256=program["frontend_sha256"],
        baseline=simulated, observed_whole_program_median_ns=observed,
        observed_samples_ns=samples, signed_relative_error=signed, absolute_relative_error=abs(signed),
        heldout_within_process_cv=cv(samples),
        heldout_within_process_cv_passed=cv(samples)<=GATES["within_process_cv_max"],
        source_quality_passed=ledger["source_quality_passed"],fixed_quality_gates=GATES,
        explicit_max_relative_error=max_relative_error,
        within_explicit_tolerance=None if max_relative_error is None else abs(signed)<=max_relative_error,
        heldout_started_at=heldout_started_at, baseline_frozen_before_heldout=True,
        verification=verification, supported_scope="actual_compiled_single_invocation_serial_F32_Dense_with_exact_optional_AttentionStages_FFNPartition",
        native_memory_accuracy_claimed=False, original_model_kernel_reproduction_claimed=False)
