"""Frozen, causal evaluation of scoped event-parked response hypotheses.

The fit is PROFILE-only. BEFORE resource inputs contain QPC walls, never raw
heldout cycles. A call prediction is sealed before its completion is inspected;
completed past calls can update only the explicitly named history candidate.
None of these coarse response hypotheses identifies physical GPU active work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

from llm_infer_model.performance.cycle_response import (
    HOST_CYCLE_SERVICE, CycleResponseProgram, CycleResponseStage,
    compile_cycle_calendar, compile_cycle_response,
)
from llm_infer_model.performance.host_pause_observer import (
    CompletedHostAction, HostPauseObserver, HostPauseParameters, PauseEndAnchor,
)
from llm_infer_model.performance.parameters import InvalidModel, Parameter, ParameterBundle
from llm_infer_model.performance.service_calendar import PeriodicServiceCalendar
from profiling.event_stage_fit import (
    _ceil, _event, _fraction, _median, _pack, _read, _sha, _signature,
    before_wall_inputs, freeze_event_profiles,
)
from profiling.host_supply_calibration import digest, require
from profiling.event_stage_recovery import event_effective_jobs


SCHEMA = "event-stage-frozen-evaluation-1"
CANDIDATES = ("profile_scalar_median",) + tuple(
    f"cycle_{phase}_{mode}"
    for phase in ("after_host", "overlap_host")
    for mode in ("no_calendar", "before_only", "causal_history")
)
PRIOR_JITTER_SOURCE = "prior diagnostic sha256:3d50f9a7f63d161c28fe83fb468c6dd51c61255dc0e47bf37b9a0c03479f50b1"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_new(path, value):
    """Never overwrite an existing parameter, barrier or evaluation receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _exact(value):
    if isinstance(value, float):
        # Timestamp statistics use .5-valued medians. Decimal parsing avoids a
        # binary-float conversion and is an explicit quantization convention.
        return Fraction(str(value))
    return _fraction(value)


def _optional_ceil(value):
    return None if value is None else _ceil(_exact(value))


def freeze(campaign, output, barrier, *, jitter_ns, calendar_horizon_ns=1_000_000_000):
    """Freeze model, protocol, scalar baseline and PROFILE source bytes."""
    require(type(jitter_ns) is int and jitter_ns >= 0, "explicit nonnegative phase jitter required")
    require(type(calendar_horizon_ns) is int and calendar_horizon_ns > 0, "explicit positive calendar horizon required")
    require(not Path(output).exists() and not Path(barrier).exists(), "freeze outputs already exist")
    frozen = freeze_event_profiles(campaign, additional_source_files=(Path(__file__),))
    walls = []
    for profile in frozen["profile_jobs"]:
        raw = _read(profile["source_main"]["path"], profile["source_main"]["sha256"])
        walls.append(_median([i["end_ns"] - i["begin_ns"] for i in raw["iterations"]]))
    frozen["profile_scalar_wall_ns"] = _pack(_median(walls))
    frozen["evaluation_protocol"] = dict(
        candidates=list(CANDIDATES), jitter_ns=jitter_ns, jitter_source=PRIOR_JITTER_SOURCE,
        jitter_predeclaration="2 ms proposed from prior ~32 ms diagnostic pause scatter, before fresh split",
        calendar_horizon_ns=calendar_horizon_ns,
        head_tail="PROFILE minimum-bound median; coarse query-boundary response, no whole-layer residual fit",
        first_gap="explicit structural zero: first prefix assigned to the measured head envelope",
        work_fit="ceil(nonnegative PROFILE affine intercept+slope*BEFORE GPU reference relative slowdown)",
        completion_fit="exact PROFILE nonnegative additive residual affine; not measured GPU active work",
        overlap_scope="same additive fitted parameters reused as predeclared coarse sensitivity; no physical overlap refit",
        scalar_baseline="median of independent process full-call wall medians, PROFILE only",
        phase_rounding="ceil PROFILE period and width statistics; observer floors phase with explicit quantization record",
        phase_drift="extra jitter_ns per extrapolated cycle since pause-end anchor; diagnostic uncertainty, not a physical bound",
        phase_drift_unknown_rule="unknown if two-sided anchor+rounding+cycle-drift uncertainty spans an entire period",
        phase_anchor="last BEFORE long reference action pause-END window expanded by explicit jitter",
        calendar_scope="diagnostic periodic response hypothesis for declared affinity/runtime; no OS simulation or neutral capacity claim",
        unknown_phase="calendar candidate remains unknown; no-calendar control always reported separately",
        conflict_rule="record entry inside inferred blackout; do not adjust phase, remove call or select another variant",
        prediction_cutoff="only independent BEFORE walls, observed call entry and completed PAST calls",
        point_error_max=.20, point_median_error_max=.20, point_p90_error_max=.20,
        required_coverage=1.0, all_calls_retained=True, heldout_variant_selection=False,
    )
    frozen["frozen_at_utc"] = _utc()
    frozen.pop("artifact_sha256", None)
    frozen["artifact_sha256"] = digest(frozen)
    _write_new(output, frozen)
    receipt = dict(schema="event-stage-frozen-barrier-1", frozen_at_utc=frozen["frozen_at_utc"],
        event_protocol_sha256=frozen["event_protocol_sha256"], manifest_sha256=frozen["manifest_sha256"],
        frozen_file=str(Path(output).resolve()), frozen_file_sha256=_sha(output),
        source_receipts=frozen["source_receipts"], evaluation_protocol_sha256=digest(frozen["evaluation_protocol"]),
        heldout_read_before_freeze=False)
    _write_new(barrier, receipt)
    return receipt


def verify_freeze(campaign, frozen_path, barrier_path):
    root = Path(campaign).resolve()
    frozen = _read(frozen_path)
    barrier = _read(barrier_path)
    require(barrier.get("schema") == "event-stage-frozen-barrier-1", "freeze barrier schema differs")
    require(Path(barrier["frozen_file"]).resolve() == Path(frozen_path).resolve(), "barrier frozen path differs")
    require(_sha(frozen_path) == barrier["frozen_file_sha256"], "frozen parameter bytes changed")
    require(frozen["artifact_sha256"] == digest({k:v for k,v in frozen.items() if k != "artifact_sha256"}),
        "frozen semantic digest differs")
    require(frozen["campaign_root"] == str(root), "frozen campaign differs")
    require(_sha(root / "manifest.json") == frozen["manifest_sha256"] == barrier["manifest_sha256"], "manifest changed")
    require(_sha(root / "event-protocol.json") == frozen["event_protocol_sha256"] == barrier["event_protocol_sha256"],
        "event protocol changed")
    require(digest(frozen["evaluation_protocol"]) == barrier["evaluation_protocol_sha256"], "evaluation protocol changed")
    require(tuple(frozen["evaluation_protocol"]["candidates"]) == CANDIDATES, "candidate protocol differs")
    require(barrier["source_receipts"] == frozen["source_receipts"], "barrier source inventory differs")
    for source in frozen["source_receipts"]:
        require(_sha(source["path"]) == source["sha256"], "frozen source changed: " + source["path"])
    return frozen, barrier


def _affine(fit, feature):
    if fit is None or fit.get("identified") is not True or fit.get("intercept") is None or fit.get("slope") is None:
        return None
    value = _exact(fit["intercept"]) + _exact(fit["slope"]) * feature
    require(value >= 0, "frozen constrained response parameter became negative")
    return value


def build_response(frozen, before):
    """Pure cost instantiation: no current or past target timing arguments."""
    feature = max(Fraction(0), _exact(before["gpu_reference_fast_action_wall_ns"]) /
        _exact(frozen["baseline_gpu_reference_wall_ns"]) - 1)
    stages = []
    for index in range(len(frozen["actions"])):
        action = frozen["actions"][str(index)]
        work = _affine(action["cpu_work_affine"], feature)
        response = _affine(action["completion_response_affine"], feature)
        gap = 0 if index == 0 else frozen["gap_cycles"].get(str(index), {}).get("ceiling")
        stages.append(CycleResponseStage(index, action["kind"], None if work is None else _ceil(work), response, gap))
    program = CycleResponseProgram(tuple(stages),
        None if frozen["head_wall_bounds"]["minimum_statistics"]["median"] is None else
            _exact(frozen["head_wall_bounds"]["minimum_statistics"]["median"]),
        None if frozen["tail_wall_bounds"]["minimum_statistics"]["median"] is None else
            _exact(frozen["tail_wall_bounds"]["minimum_statistics"]["median"]),
        ("frozen PROFILE semantic digest:" + frozen["artifact_sha256"],),
        "scoped raw host cycles + local completion response; no GPU-active or submission-acceptance identification", "profiled")
    bundle = ParameterBundle((Parameter(HOST_CYCLE_SERVICE, _exact(before["cpu_cycle_rate_per_s"]),
        "cpu_cycle/s", "profiled", tuple(before["evidence"]["cpu"]),
        "frozen PROFILE work / independent BEFORE reference wall; includes observer/runtime response"),))
    compiled, errors = {}, {}
    for phase in ("after_host", "overlap_host"):
        try:
            compiled[phase] = compile_cycle_response(program, completion_phase=phase)
        except InvalidModel as error:
            errors[phase] = str(error)
    return compiled, bundle, dict(feature=_pack(feature), stages=[_pack(asdict(s)) for s in stages],
        head_ns=_pack(program.head_ns), tail_ns=_pack(program.tail_ns), compilation_errors=errors)


def make_observers(frozen, before, *, cpu_reference_cutoff_ns):
    hypothesis = frozen["CPU_blackout_hypothesis"]
    period = _optional_ceil(hypothesis["period_candidate_ns"])
    width = _optional_ceil(hypothesis["pause_width_candidate_ns"])
    jitter = frozen["evaluation_protocol"]["jitter_ns"]
    parameters = HostPauseParameters(period, width, frozen["declared_protocol"]["minimum_excess_ns"],
        {key:None if value["fast_wall_ns"] is None else _exact(value["fast_wall_ns"])
         for key,value in frozen["actions"].items()}, jitter,
        ("PROFILE pause hypothesis:" + hypothesis["artifact_sha256"],),
        "fixed recurrence/width; BEFORE or completed-response phase hypothesis, no neutral CPU capacity identified")
    phase = before.get("before_phase")
    anchor = None
    if phase and phase.get("known") is True:
        window = phase["pause_end_window_ns"]
        require(window["minimum"] <= window["maximum"] <= cpu_reference_cutoff_ns,
            "BEFORE pause-end observation extends beyond its available cutoff")
        anchor = PauseEndAnchor(window["minimum"] - jitter, window["maximum"] + jitter,
            cpu_reference_cutoff_ns, frozen["clock"]["clock_id"], frozen["clock"]["clock_origin"],
            tuple(before["evidence"]["cpu"]))
    return {mode:HostPauseObserver(parameters, initial_anchor=anchor,
        clock_id=frozen["clock"]["clock_id"], clock_origin=frozen["clock"]["clock_origin"], mode=mode)
        for mode in ("before_only", "causal_history")}


def phase_age_diagnostic(frozen, phase):
    """Do not claim a fixed initial jitter bounds arbitrarily old phases."""
    if not phase.known or phase.anchor_bounds_ns is None:
        return dict(known=False, reason=phase.reason, extrapolated_cycles=None,
            total_uncertainty_ns=None, physically_proven_bound=False)
    center = sum(phase.anchor_bounds_ns) / 2
    age = max(Fraction(0), Fraction(phase.at_ns) - center)
    cycles = _ceil(age / phase.period_ns)
    drift = cycles * frozen["evaluation_protocol"]["jitter_ns"]
    total = phase.uncertainty_ns + phase.phase_quantization_ns + drift
    known = 2 * total < phase.period_ns
    return _pack(dict(known=known, reason=None if known else "extrapolated_phase_uncertainty_spans_period",
        age_since_pause_end_center_ns=age, extrapolated_cycles=cycles, added_drift_uncertainty_ns=drift,
        total_uncertainty_ns=total, physically_proven_bound=False,
        scope="predeclared conservative phase drift hypothesis, no current/future target response"))


def continuous_predictions(compiled, bundle):
    """Per-process controls do not depend on entry phase or target outcomes."""
    rows = {}
    for assumption in ("after_host", "overlap_host"):
        model = compiled.get(assumption)
        if model is None:
            rows[assumption] = dict(known=False, predicted_ns=None, reason="frozen action component unidentified")
        else:
            try:
                rows[assumption] = dict(known=True, predicted_ns=model.evaluate(bundle).duration_ns,
                    source="common Graph, continuous BEFORE reference cycle response, no calendar")
            except InvalidModel as error:
                rows[assumption] = dict(known=False, predicted_ns=None, reason=str(error))
    return rows


def predict_call(frozen, compiled, bundle, phases, *, begin_ns, call_id, continuous=None):
    """Predict from entry and already sealed phase snapshots ONLY.

    There is deliberately no target end/action/cycle argument. The caller must
    seal this result before it extracts this call's measured outcome.
    """
    rows = {"profile_scalar_median":dict(known=frozen["profile_scalar_wall_ns"] is not None,
        predicted_ns=frozen["profile_scalar_wall_ns"], source="frozen PROFILE whole-call process median")}
    continuous = continuous_predictions(compiled, bundle) if continuous is None else continuous
    for assumption in ("after_host", "overlap_host"):
        model = compiled.get(assumption)
        no_calendar = f"cycle_{assumption}_no_calendar"
        rows[no_calendar] = dict(continuous[assumption])
        for mode in ("before_only", "causal_history"):
            name = f"cycle_{assumption}_{mode}"
            phase = phases[mode]
            age = phase_age_diagnostic(frozen, phase)
            row = dict(known=False, predicted_ns=None, phase=phase.to_dict(),
                phase_age_diagnostic=age,
                observation_cutoff_ns=phase.cutoff_ns, past_target_cutoff_ns=phase.history_cutoff_ns,
                observed_entry_ns=begin_ns, current_target_outcome_used=False,
                call_begin_inside_blackout_conflict=None)
            if model is None:
                row["reason"] = "frozen action component unidentified"
            elif not phase.known:
                row["reason"] = phase.reason
            elif not age["known"]:
                row["reason"] = age["reason"]
            else:
                require(phase.at_ns == begin_ns and phase.call_id == call_id, "phase snapshot belongs to another call")
                calendar = PeriodicServiceCalendar(HOST_CYCLE_SERVICE, phase.period_ns, phase.blackout_ns,
                    phase.relative_phase_ns, frozen["evaluation_protocol"]["calendar_horizon_ns"],
                    "observed_diagnostic", ("causal phase snapshot:" + digest(phase.to_dict()),),
                    "predeclared periodic external host-response hypothesis; fixed completion response continues",
                    phase.clock_id, "evaluation-relative entry=" + str(begin_ns) + "; original=" + phase.clock_origin)
                row["call_begin_inside_blackout_conflict"] = calendar.blackout_at(0)
                try:
                    result = compile_cycle_calendar(bundle, calendar, allow_observed_diagnostic=True).evaluate(model, bundle)
                    row.update(known=True, predicted_ns=result.duration_ns, reason=None,
                        source="common Graph under causal diagnostic host-response calendar")
                except InvalidModel as error:
                    row["reason"] = str(error)
            rows[name] = row
    return _pack(dict(call_id=call_id, begin_ns=begin_ns, predictions=rows,
        prediction_inputs="frozen PROFILE + BEFORE wall references + observed entry + PAST completed history only",
        current_target_end_read=False, current_target_actions_read=False, current_target_cycles_read=False))


def _load_heldout(root, parent, protocol, frozen, barrier, job):
    """Validate provenance without extracting current target timing/counters."""
    require(job["split"] == "heldout" and job["parking"] == "event" and job["cycles"] is False and job["samples"] == 96,
        "heldout declared protocol differs")
    execution = _read(root / "jobs" / job["id"] / "execution.json")
    require(execution.get("job") == job and execution.get("status") == "completed" and
        execution.get("cleanup_verified") is True, "heldout execution incomplete")
    require(datetime.fromisoformat(execution["started_at_utc"]) >= datetime.fromisoformat(barrier["frozen_at_utc"]),
        "heldout launch predates frozen parameters")
    binding = parent["programs"][job["program_key"]]
    require(execution.get("program") == binding, "heldout program binding differs")
    program = _read(binding["path"], binding["sha256"])
    _read(binding["oracle"], binding["oracle_sha256"])
    require(_signature(program) == frozen["signature"], "heldout workload signature differs")
    require(program["program_sha256"] == binding["program_sha256"] == digest({k:v for k,v in program.items() if k != "program_sha256"}),
        "heldout semantic program hash differs")
    require(binding["seed"] not in {p["seed"] for p in frozen["profile_identities"]}, "heldout numerical seed overlaps PROFILE")
    actors = {p["name"]:p for p in execution["processes"]}
    require(len(actors) == len(execution["processes"]) and set(actors) == set(execution["raw_files"]), "owned actor inventory differs")
    raw = {}
    for name, process in actors.items():
        require(process.get("returncode") == 0 and process.get("forced_termination") is False and
            process.get("process_exited_verified") is True, "heldout actor did not exit normally")
        numerical = execution["numerical"].get(name, {})
        require(all(numerical.get(k) is True for k in ("passed", "full_graph_equal", "action_trace_equal",
            "full_port_inventory_equal", "copy_routes_bytes_and_values_equal")), "numerical receipt incomplete")
        source = execution["raw_files"][name]
        require(_sha(source["path"]) == source["sha256"], "heldout raw source changed")
        if name not in {"main", "before-cpu", "before-gpu"}:
            continue
        native = _read(source["path"], source["sha256"])
        require(native.get("schema") == "physical-stage-native-1" and
            native.get("process_id") == process["pid"] == process["ready"]["process_id"], "heldout native identity differs")
        require(native.get("main_thread_id") == process["ready"]["main_thread_id"] and
            native["clock"] == process["ready"]["physical_clock"] == frozen["clock"], "heldout thread/common clock differs")
        require(native.get("program_sha256") == program["program_sha256"] and native.get("actions") == program["actions"],
            "heldout action inventory differs")
        require(Path(process["command"][0]).resolve() == Path(protocol["event_binary"]).resolve() and
            Path(process["command"][1]).resolve() == Path(binding["path"]).resolve(), "heldout binary/program differs")
        _event(native, process)
        count = job["samples"] if name == "main" else parent["protocol"]["reference_" + name.split("-")[-1] + "_samples"]
        require(native["samples_requested"] == native["samples_completed"] == len(native["iterations"]) == count,
            "heldout sample coverage differs")
        target = None if name == "main" else frozen["reference_targets"][name.split("-")[-1]]
        require(native.get("target_action_index") == target and native.get("mode") == ("full" if name == "main" else "target"),
            "heldout actor target differs")
        if name == "main":
            require(native.get("thread_cycles_enabled") is False and native.get("instrumentation") == "bare",
                "heldout target must use bare wall-only observer")
        raw[name] = native
    require(set(raw) == {"main", "before-cpu", "before-gpu"}, "heldout independent BEFORE references missing")
    return execution, raw


def _percentile(values, proportion):
    return None if not values else sorted(values)[max(0, math.ceil(len(values) * proportion) - 1)]


def metrics(rows, *, expected_count):
    result = {}
    for candidate in CANDIDATES:
        pairs = [(row["predictions"][candidate], row["observed_wall_ns"]) for row in rows]
        known = [(float(_exact(p["predicted_ns"])), y) for p,y in pairs if p["known"]]
        ape = [abs(p-y)/y for p,y in known]
        signed = [(p-y)/y for p,y in known]
        median = statistics.median(ape) if ape else None
        p90 = _percentile(ape, .90)
        complete = len(rows) == expected_count and len(known) == expected_count
        result[candidate] = dict(total_calls=len(rows), expected_calls=expected_count, known_calls=len(known),
            unknown_calls=len(rows)-len(known), coverage=len(known)/expected_count,
            known_subset_median_ape=median, known_subset_p90_ape=p90,
            known_subset_worst_ape=max(ape) if ape else None,
            known_subset_mean_signed_relative_error=statistics.mean(signed) if signed else None,
            within_20_percent=sum(e <= .20 for e in ape), complete_coverage=complete,
            point_accuracy_gate=bool(complete and median <= .20 and p90 <= .20),
            entry_blackout_conflicts=sum(p.get("call_begin_inside_blackout_conflict") is True for p,_ in pairs),
            conditional_metrics_do_not_establish_full_coverage=True)
    return result


def evaluate_campaign(campaign, frozen_path, barrier_path, output):
    require(not Path(output).exists(), "evaluation output already exists")
    root = Path(campaign).resolve()
    frozen, barrier = verify_freeze(root, frozen_path, barrier_path)
    parent, protocol = _read(root / "manifest.json"), _read(root / "event-protocol.json")
    heldout_started = _read(root / "heldout-started.json")
    require(heldout_started["barrier"] == barrier, "collection did not use this frozen barrier")
    jobs = [j for j in event_effective_jobs(root, protocol) if j["split"] == "heldout"]
    require(len(jobs) == 12, "declared heldout process count differs")
    rows, process_rows, legal_inputs = [], [], []
    for job in jobs:
        execution, raw = _load_heldout(root, parent, protocol, frozen, barrier, job)
        main, cpu, gpu = raw["main"], raw["before-cpu"], raw["before-gpu"]
        before = before_wall_inputs(frozen, cpu, gpu, cutoff_ns=main["sampling_begin_ns"],
            target_process_id=main["process_id"], cpu_evidence=[execution["raw_files"]["before-cpu"]["path"]],
            gpu_evidence=[execution["raw_files"]["before-gpu"]["path"]])
        compiled, bundle, costs = build_response(frozen, before)
        continuous = continuous_predictions(compiled, bundle)
        observers = make_observers(frozen, before, cpu_reference_cutoff_ns=cpu["sampling_end_ns"])
        legal_inputs.append(dict(job_id=job["id"], before=before, model_costs=costs,
            cycle_parameter_bundle=bundle.to_dict(), target_action_cycles_read=False,
            target_process_id=main["process_id"], source_receipts=execution["raw_files"]))
        own_rows = []
        for ordinal, invocation in enumerate(main["iterations"]):
            begin = invocation["begin_ns"]  # conditional entry is a declared legal input
            call_id = job["id"] + ":" + str(ordinal)
            phases = {mode:observer.predict(begin, call_id=call_id) for mode,observer in observers.items()}
            sealed = predict_call(frozen, compiled, bundle, phases, begin_ns=begin, call_id=call_id, continuous=continuous)
            # Causal fence: prediction bytes/digest are committed in the row
            # before any current end/action duration is extracted below.
            sealed_bytes = json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False)
            row = json.loads(sealed_bytes)
            row["sealed_prediction_sha256"] = digest(sealed)
            row["prediction_sealed_before_outcome_extraction"] = True
            end = invocation["end_ns"]
            require(end > begin, "heldout observed call duration must be positive")
            actions = invocation["actions"]
            require(len(actions) == len(frozen["actions"]) and
                all(a["action_index"] == i for i,a in enumerate(actions)), "heldout completed action trace differs")
            row.update(job_id=job["id"], condition=job["condition"], observed_wall_ns=end-begin,
                observed_end_ns=end, observed_after_prediction_seal=True)
            history = tuple(CompletedHostAction(str(a["action_index"]), a["begin_ns"], a["end_ns"]) for a in actions)
            row["history_update_after_prediction"] = {mode:asdict(observer.observe_completed(call_id, begin, end,
                actions=history, observed_at_ns=end, clock_id=frozen["clock"]["clock_id"],
                clock_origin=frozen["clock"]["clock_origin"])) for mode,observer in observers.items()}
            own_rows.append(row)
        rows.extend(own_rows)
        process_rows.append(dict(job_id=job["id"], condition=job["condition"], process_id=main["process_id"],
            metrics=metrics(own_rows, expected_count=job["samples"])))
    verify_freeze(root, frozen_path, barrier_path)
    result = dict(schema=SCHEMA, evaluated_at_utc=_utc(), campaign_root=str(root), frozen_file=str(Path(frozen_path).resolve()),
        frozen_file_sha256=_sha(frozen_path), frozen_artifact_sha256=frozen["artifact_sha256"],
        barrier_file_sha256=_sha(barrier_path), evaluation_protocol=frozen["evaluation_protocol"],
        evaluation_mode="offline sequential causal replay with per-call prediction serialization before outcome extraction; not a live precommit experiment",
        declared_candidates=list(CANDIDATES), selected_on_heldout=False, expected_calls=sum(j["samples"] for j in jobs),
        metrics=metrics(rows, expected_count=sum(j["samples"] for j in jobs)), processes=process_rows,
        by_condition={state:metrics([r for r in rows if r["condition"] == state],
            expected_count=sum(j["samples"] for j in jobs if j["condition"] == state))
            for state in sorted({j["condition"] for j in jobs})}, legal_inputs=legal_inputs, calls=rows,
        all_slow_calls_retained=True, target_heldout_cycles_read=False, before_heldout_cycles_read=False,
        heldout_parameters_fitted=False, target_current_outcome_used_for_prediction=False,
        completed_past_response_used_only_by_causal_history=True, whole_layer_offset_fitted=False,
        physical_stage_identification_qualified=False, hardware_accuracy_qualified=False,
        limitations=["CPU cycle work and completion response are coarse, underidentified PROFILE hypotheses.",
            "BEFORE probes characterize scoped response; they are not instantaneous physical CPU/GPU capacity.",
            "Fixed period/width calendar is a diagnostic hypothesis, and phase uncertainty is reported rather than fitted.",
            "Overlapping completion reuses additive fits as sensitivity, not a separately calibrated physical DAG.",
            "Point accuracy alone cannot establish observer noninterference, full workload generalization or causal mechanisms."])
    _write_new(output, _pack(result))
    return dict(output=str(Path(output).resolve()), metrics=result["metrics"], all_calls=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--barrier", type=Path, required=True)
    prepare.add_argument("--jitter-ns", type=int, required=True)
    prepare.add_argument("--calendar-horizon-ns", type=int, default=1_000_000_000)
    evaluate = commands.add_parser("evaluate")
    for name in ("campaign", "frozen", "barrier", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.campaign, args.output, args.barrier, jitter_ns=args.jitter_ns,
            calendar_horizon_ns=args.calendar_horizon_ns)
    else:
        result = evaluate_campaign(args.campaign, args.frozen, args.barrier, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
