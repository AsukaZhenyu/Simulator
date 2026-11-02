"""Profile-only periodic host-pause hypotheses, never OS scheduling models.

Inclusive host-action excess can locate a service-pause candidate. It cannot
identify active CPU time or prove that the pause was caused by the OS. The
period and width below are diagnostic hypotheses; no neutral availability or
phase is inferred. A separately supplied BEFORE-only phase observation is
required for an explicitly scoped forward scenario.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import statistics


SCHEMA = "host-periodic-pause-hypothesis-1"
OBSERVATION_SCHEMA = "host-pause-profile-observation-1"
ANCHOR_SCHEMA = "host-pause-before-anchor-1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def text(value, name):
    require(isinstance(value, str) and value.strip(), name + " must be nonempty text")
    return value


def _stats(values):
    if not values:
        return dict(n=0, median=None, mean=None, minimum=None, maximum=None, cv=None)
    mean = statistics.mean(values)
    return dict(n=len(values), median=statistics.median(values), mean=mean,
        minimum=min(values), maximum=max(values),
        cv=statistics.pstdev(values)/mean if mean else (0.0 if not any(values) else None))


def _evidence(value):
    require(isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value),
            "explicit nonempty evidence references required")
    return list(value)


def _observation(value, signature):
    require(isinstance(value, dict) and value.get("schema") == OBSERVATION_SCHEMA, "profile observation schema differs")
    require(value.get("split") == "profile", "only PROFILE observations can calibrate host pauses")
    require(value.get("signature") == signature, "profile execution signature differs")
    require(value.get("numerical_verified") is True, "profile numerical verification missing")
    competitor = value.get("competitor_context")
    require(isinstance(competitor, dict) and competitor and
            not set(competitor)&{"state", "condition", "label", "state_id", "cohort"},
            "explicit external-competitor context required; state labels are not features")
    actors=competitor.get("external_actors")
    require(isinstance(actors,list) and actors,"external actor resource/workload/placement identities required")
    for actor in actors:
        require(isinstance(actor,dict),"external actor descriptor required")
        text(actor.get("resource_id"),"external actor resource")
        workload=actor.get("workload_signature");placement=actor.get("placement")
        require((isinstance(workload,str) and workload.strip()) or (isinstance(workload,dict) and workload),
                "external actor workload signature required")
        require((isinstance(placement,str) and placement.strip()) or (isinstance(placement,dict) and placement),
                "external actor placement required")
    for key in ("observation_id", "run_id", "resource_id", "clock_id", "clock_origin"):
        text(value.get(key), key)
    for key in ("process_id", "seed"):
        integer(value.get(key), key, 1)
    begin, end = integer(value.get("begin_ns"), "observation begin"), integer(value.get("end_ns"), "observation end")
    require(begin < end, "observation interval must be positive")
    _evidence(value.get("evidence"))
    actions = value.get("actions")
    require(isinstance(actions, list) and actions, "complete ordered action intervals required")
    cursor = begin
    for ordinal, action in enumerate(actions):
        require(isinstance(action, dict) and action.get("action_index") == ordinal,
                "original action indices must be complete and contiguous")
        left, right = integer(action.get("begin_ns"), "action begin"), integer(action.get("end_ns"), "action end")
        require(cursor <= left <= right <= end, "action intervals overlap or escape the invocation")
        cursor = right
        cycles = action.get("active_cycles")
        if cycles is not None:
            integer(cycles, "raw active cycles")
            require(action.get("active_cycles_scope") == "main_thread_raw_cycles", "cycles need explicit raw main-thread scope")
    return value


def calibrate_blackout_hypothesis(profile_observations, signature, *, minimum_excess_ns,
        min_independent_runs=3, minimum_events_per_run=2, maximum_missing_cycles=4):
    """Freeze a periodic *hypothesis* from one exact profile resource context.

    The threshold and required coverage are caller-declared, never chosen from
    heldout errors. Per-action profile medians are reference fast wall values,
    not active CPU work parameters. At most one dominant-action pause candidate
    is retained per invocation, with its full action time window; the exact
    stall start is unknown. Run-local event spacings are measured recurrence,
    not a uniquely identified scheduler period. Raw cycle candidates, when
    supplied, remain cycles and are never converted using an assumed GHz rate.
    """
    require((isinstance(signature, dict) and signature) or (isinstance(signature, str) and signature.strip()),
            "nonempty explicit execution signature required")
    integer(minimum_excess_ns, "minimum excess", 1)
    integer(min_independent_runs, "minimum independent runs", 1)
    integer(minimum_events_per_run, "minimum events per run", 2)
    integer(maximum_missing_cycles, "maximum missing cycles", 1)
    require(isinstance(profile_observations, (list, tuple)) and profile_observations, "profile observations required")
    observations = [_observation(o, signature) for o in profile_observations]
    first = observations[0]
    context = {k: first[k] for k in ("resource_id", "competitor_context", "clock_id", "clock_origin")}
    require(all(all(o[k] == v for k,v in context.items()) for o in observations),
            "mixed resource/competitor/clock contexts; calibrate one explicit context at a time")
    require(len({o["observation_id"] for o in observations}) == len(observations), "duplicate observation id")
    runs = defaultdict(list)
    for observation in observations:
        runs[observation["run_id"]].append(observation)
    identities = []
    for run_id, members in runs.items():
        identities_here = {(m["process_id"], m["seed"]) for m in members}
        require(len(identities_here) == 1, "run id mixes native process/seed identities")
        pid, seed = next(iter(identities_here))
        require((pid,seed) not in [(i["process_id"],i["seed"]) for i in identities], "renamed duplicate process/seed run")
        identities.append(dict(run_id=run_id,process_id=pid,seed=seed))
        ordered = sorted(members,key=lambda o:o["begin_ns"])
        require(all(a["end_ns"]<=b["begin_ns"] for a,b in zip(ordered,ordered[1:])), "profile invocation windows overlap")
    require(len(runs)>=min_independent_runs,"insufficient independent profile runs")
    action_count = len(first["actions"])
    require(all(len(o["actions"])==action_count for o in observations), "action inventory length differs")
    # Medians are used only to locate excess and never advertised as a whole-call predictor.
    fast_wall = {index: statistics.median([o["actions"][index]["end_ns"]-o["actions"][index]["begin_ns"]
                 for o in observations]) for index in range(action_count)}
    events, by_run = [], {}
    for run_id, members in runs.items():
        selected = []
        for observation in sorted(members,key=lambda o:o["begin_ns"]):
            action = max(observation["actions"],key=lambda a:a["end_ns"]-a["begin_ns"]-fast_wall[a["action_index"]])
            excess = action["end_ns"]-action["begin_ns"]-fast_wall[action["action_index"]]
            if excess < minimum_excess_ns:
                continue
            event = dict(observation_id=observation["observation_id"],run_id=run_id,
                action_index=action["action_index"],action_begin_ns=action["begin_ns"],action_end_ns=action["end_ns"],
                excess_candidate_ns=excess,profile_fast_action_wall_ns=fast_wall[action["action_index"]],
                event_location_window=dict(begin_ns=action["begin_ns"],end_ns=action["end_ns"]),
                exact_pause_start_identified=False,excess_is_active_CPU_work=False,evidence=list(observation["evidence"]))
            events.append(event);selected.append(event)
        by_run[run_id]=selected
    coverage={run_id:dict(observations=len(runs[run_id]),events=len(found),
        minimum_events_pass=len(found)>=minimum_events_per_run) for run_id,found in by_run.items()}
    gaps=[dict(run_id=run_id,left_observation_id=a["observation_id"],right_observation_id=b["observation_id"],
              interval_ns=b["action_begin_ns"]-a["action_begin_ns"],
              timing_scope="dominant action entry recurrence; exact stall starts remain unknown")
          for run_id,found in by_run.items() for a,b in zip(found,found[1:])]
    raw_period=_stats([g["interval_ns"] for g in gaps])
    period=raw_period["median"]
    normalized=[]
    if period is not None and period>0:
        for gap in gaps:
            multiple=max(1,int(gap["interval_ns"]/period+.5))
            normalized.append(dict(gap,missing_cycle_multiple_candidate=multiple,
                allowed_multiple=multiple<=maximum_missing_cycles,
                normalized_interval_ns=gap["interval_ns"]/multiple if multiple<=maximum_missing_cycles else None))
    normalized_stats=_stats([r["normalized_interval_ns"] for r in normalized if r["normalized_interval_ns"] is not None])
    width=_stats([e["excess_candidate_ns"] for e in events])
    complete=all(c["minimum_events_pass"] for c in coverage.values()) and bool(gaps)
    cycle_candidates={}
    for index in range(action_count):
        values=[o["actions"][index].get("active_cycles") for o in observations]
        known=[v for v in values if v is not None]
        cycle_candidates[str(index)]=dict(unit="raw_main_thread_cycle",statistics=_stats(known),
            unknown_samples=len(values)-len(known),qualified_physical_work=False,
            cycle_to_time_conversion_performed=False)
    result=dict(schema=SCHEMA,signature=deepcopy(signature),signature_sha256=digest(signature),
        context=deepcopy(context),source_observations_sha256=digest(profile_observations),profile_identities=identities,
        profile_observations=len(observations),profile_event_candidates=len(events),run_coverage=coverage,
        declared_protocol=dict(minimum_excess_ns=minimum_excess_ns,min_independent_runs=min_independent_runs,
            minimum_events_per_run=minimum_events_per_run,maximum_missing_cycles=maximum_missing_cycles),
        fast_action_wall_reference_ns={str(k):v for k,v in fast_wall.items()},events=events,
        recurrence_intervals=gaps,normalized_recurrence_candidates=normalized,
        period_candidate_ns=normalized_stats["median"] if complete else None,
        pause_width_candidate_ns=width["median"] if complete else None,
        pause_width_statistics_ns=width,raw_recurrence_statistics_ns=raw_period,
        normalized_recurrence_statistics_ns=normalized_stats,
        coverage_qualified=complete,phase=None,phase_identified=False,
        period_uniquely_identified=False,phase_must_come_from_independent_BEFORE_observation=True,
        action_work_cycle_candidates=cycle_candidates,
        calibration_scope="profile-only dominant host-wall-excess periodic hypothesis; not OS scheduler or neutral capacity",
        heldout_observations_used_for_calibration=False,target_wall_used_to_derive_forward_supply=False,
        CPU_available_service_identified=False,pause_cause_identified=False,
        hardware_accuracy_qualified=False,
        limitations=["Action-entry recurrence is not an exact pause-start interval.",
            "Detected excess can contain runtime waits, host work and preemption; it is not proven CPU denial.",
            "A repeated recurrence can alias a shorter period or skip events; phase remains unknown.",
            "This hypothesis may be evaluated as an explicit diagnostic scenario only; it is not a measured neutral supply trajectory."])
    result["artifact_sha256"]=digest(result)
    return result


def bind_before_phase_anchor(frozen, anchor, *, prediction_begin_ns, target_process_id,
                             target_observation_ids=()):
    """Validate a BEFORE-only external phase declaration without fitting it.

    Source roles/identities and causal time bounds are checked. These declarations
    do not independently prove sensor neutrality or physical phase accuracy, so
    hardware qualification remains false. Target/AFTER anchors are rejected even
    if renamed or accompanied by a claimed independent source kind.
    """
    require(isinstance(frozen,dict) and frozen.get("schema")==SCHEMA,"frozen host hypothesis schema differs")
    require(frozen.get("artifact_sha256")==digest({k:v for k,v in frozen.items() if k!="artifact_sha256"}),
            "frozen hypothesis digest differs")
    require(frozen.get("period_candidate_ns") is not None and frozen.get("pause_width_candidate_ns") is not None,
            "period/width unknown; cannot bind a phase")
    require(isinstance(anchor,dict) and anchor.get("schema")==ANCHOR_SCHEMA,"BEFORE anchor schema differs")
    integer(prediction_begin_ns,"prediction begin")
    integer(target_process_id,"target PID",1)
    require(anchor.get("sample_role")=="before_target" and anchor.get("derived_from_target_timing") is False and
            anchor.get("uses_future_observations") is False,"phase must be independent BEFORE-only; target/AFTER leakage rejected")
    require(anchor.get("source_role") in {"external_competitor","independent_resource_probe","declared_scenario"},
            "target-derived phase source role rejected")
    require(anchor.get("signature_sha256")==frozen["signature_sha256"],"phase execution signature differs")
    for key in ("resource_id","clock_id","clock_origin"):
        require(anchor.get(key)==frozen["context"][key],"phase resource/clock context differs")
    require(anchor.get("competitor_context")==frozen["context"]["competitor_context"],"phase competitor context differs")
    begin=integer(anchor.get("sensor_begin_ns"),"anchor sensor begin")
    end=integer(anchor.get("sensor_end_ns"),"anchor sensor end")
    require(begin<end<=prediction_begin_ns,"phase sensor crosses target prediction cutoff")
    interval=anchor.get("phase_start_window_ns")
    require(isinstance(interval,dict) and set(interval)=={"minimum","maximum"},"explicit phase uncertainty bounds required")
    low=integer(interval["minimum"],"phase lower bound");high=integer(interval["maximum"],"phase upper bound")
    require(begin<=low<=high<=end,"phase bounds escape the BEFORE observation")
    ids=anchor.get("observed_process_ids")
    require(isinstance(ids,list) and ids and all(type(pid) is int and pid>0 for pid in ids),"observed process identities required")
    require(target_process_id not in ids,"target process counters/timing cannot supply its own phase")
    sources=anchor.get("source_observation_ids")
    require(isinstance(sources,list) and sources and all(isinstance(s,str) and s.strip() for s in sources),"anchor observation evidence required")
    require(not set(sources)&set(target_observation_ids),"target observations cannot be renamed into external phase evidence")
    _evidence(anchor.get("evidence"))
    return dict(schema="host-pause-bound-before-scenario-1",frozen_hypothesis_sha256=frozen["artifact_sha256"],
        signature_sha256=frozen["signature_sha256"],resource_context=deepcopy(frozen["context"]),
        period_candidate_ns=frozen["period_candidate_ns"],pause_width_candidate_ns=frozen["pause_width_candidate_ns"],
        phase_start_window_ns=deepcopy(interval),phase_anchor=deepcopy(anchor),prediction_begin_ns=prediction_begin_ns,
        origin="observed_diagnostic",phase_anchor_causality_checks_passed=True,
        source_neutrality_proven=False,physical_CPU_supply_identified=False,hardware_accuracy_qualified=False)
