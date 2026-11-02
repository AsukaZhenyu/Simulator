"""Neutrality, exact API partitions and non-fitted callback observation cells."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from profiling.dynamic_validation import external as plans
from profiling.dynamic_validation import external_v6 as campaign


def coordinate(metadata):
    """Exclude process-local stream identity from a reusable observation key."""
    return {k: v for k, v in metadata.items() if k != "stream_handle_opaque_process_local"}


def stable_state(request, begin_ns, end_ns):
    before, after = plans.state_at(request, begin_ns), plans.state_at(request, max(begin_ns, end_ns - 1))
    edges = plans.boundaries(request)
    last = max((edge for edge in edges if edge <= begin_ns), default=None)
    stable = before == after and not any(begin_ns < edge < end_ns for edge in edges) and (last is None or begin_ns - last >= 200_000_000)
    return before, stable


def analyze(out):
    out = Path(out).resolve()
    manifest = campaign.read(out / "manifest.json")
    campaign.base.check(manifest)
    executions = campaign.read(out / "executions.json")
    result = dict(schema="dynamic-external-callback-analysis-6", manifest=campaign.receipt(out / "manifest.json"),
                  executions=campaign.receipt(out / "executions.json"), processes=[], observer_pairs=[],
                  performance_parameters_fitted=False, whole_target_used_as_fit_label=False,
                  callback_response_is_not_pure_CPU_work_or_GPU_acceptance=True,
                  callback_response_is_not_additive_to_internal_GPU_service=True,
                  state_labels="predeclared owned actor ON/OFF at recorded API boundaries; actual actor receipts remain diagnostic",
                  stable_cell_rule="entire API interval in one planned state and >=200ms after the latest planned edge; all transition calls retained in raw",
                  excluded_for_CV=0, runtime_callback_scope="Attention only, all2048 phases")
    result["not_external_fluctuation_heldout_accuracy"] = True
    result["background_context"] = dict(
        source="parent-reported measurement coordination incident",
        first_control_possible_incidental_analysis_overlap=True,
        first_control_job="diagnostic-cpu-mask4-stamped",
        incidental_analysis_elapsed_seconds=5.828,
        analysis_begin_end_UTC=None, BLAS_threads=None,
        overlap_is_proven=False,
        interpretation="Cannot establish no overlap with first control target; all samples retained, no rerun. Parent reported analysis stopped before remaining3 jobs.")
    for execution in executions:
        job = execution["job"]
        process = dict(job_id=job["id"], cpu_actor_mask=job["cpu_actor_mask"], role=job["role"], status=execution["status"])
        if execution["status"] != "completed":
            process["error"] = execution.get("error")
            result["processes"].append(process)
            continue
        campaign.require(campaign.sha(execution["native"]["path"]) == execution["native"]["sha256"], "native raw changed")
        native = campaign.read(execution["native"]["path"])
        hz, epoch, request = native["clock"]["frequency_hz"], native["target_expected_origin_tick"], job["request"]
        phases = {action["callback_phase_id"]: action for block in native["blocks"] for action in block["actions"] if action["callback_phase_id"] is not None}
        ledger = campaign.callback_ledger(native["runtime_callbacks"], phases, hz)
        attention = []
        for phase_id, action in phases.items():
            start = (action["begin_tick"] - epoch) * 1_000_000_000 // hz
            end = (action["end_tick"] - epoch) * 1_000_000_000 // hz
            state, stable = stable_state(request, start, end)
            attention.append(dict(phase_id=phase_id, global_replay=action["global_replay"], begin_since_epoch_ns=start,
                                  end_since_epoch_ns=end, host_wall_ns=action["wall_ns"], planned_state=list(state),
                                  stable_in_one_state=stable, first_CPU_release_spanning=start < 4e9 < end,
                                  second_CPU_release_spanning=start < 14e9 < end))
        cells, sequence = {}, defaultdict(list)
        metadata = native["runtime_callbacks"]["metadata"]
        coordinates = [coordinate(m) for m in metadata]
        keys = [hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for c in coordinates]
        transition = 0
        for record in native["runtime_callbacks"]["callback_records"]:
            begin = (record["enter_tick"] - epoch) * 1_000_000_000 // hz
            end = (record["exit_tick"] - epoch) * 1_000_000_000 // hz
            index = record["metadata_index"]
            sequence[record["phase_id"]].append(keys[index])
            state, stable = stable_state(request, begin, end)
            if not stable:
                transition += 1
                continue
            key = (keys[index], *state)
            cell = cells.setdefault(key, dict(coordinate=coordinates[index], coordinate_sha256=keys[index],
                                              planned_CPU_actor=state[0], planned_GPU_actor=state[1], count=0,
                                              sum_ticks=0, sum_squared_ticks=0, minimum_ticks=None, maximum_ticks=None))
            duration = record["exit_tick"] - record["enter_tick"]
            cell["count"] += 1
            cell["sum_ticks"] += duration
            cell["sum_squared_ticks"] += duration * duration
            cell["minimum_ticks"] = duration if cell["minimum_ticks"] is None else min(cell["minimum_ticks"], duration)
            cell["maximum_ticks"] = duration if cell["maximum_ticks"] is None else max(cell["maximum_ticks"], duration)
        signatures = defaultdict(int)
        for values in sequence.values():
            signatures[hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()] += 1
        for cell in cells.values():
            cell["mean_inclusive_response_ns"] = cell["sum_ticks"] * 1_000_000_000 / (hz * cell["count"])
            cell["timing_resolution_limited"] = cell["mean_inclusive_response_ns"] < 1_000_000_000 / hz
            cell["parameter_qualified"] = False
        off = [a["host_wall_ns"] for a in attention if a["stable_in_one_state"] and a["planned_state"] == [0, 0]]
        process.update(native=execution["native"], target_wall_ns=native["target_wall_ns"],
                       complete_off_state_attention_count=len(off), complete_off_state_attention_mean_ns=statistics.mean(off) if off else None,
                       callback_POD_bytes=native["runtime_callbacks"]["allocated_POD_bytes"],
                       callback_materialization_ns_outside_target=native["callback_finalize_ns_outside_target"],
                       target_time_materialization_ns=0, API_partitions=ledger, attention_actions=attention,
                       coordinate_sequence_signatures=dict(signatures), observations=list(cells.values()),
                       transition_API_observations_retained_in_native=transition, actor_receipts=native["actors"])
        result["processes"].append(process)
    for mask in sorted({job["cpu_actor_mask"] for job in manifest["jobs"]}):
        controls = [p for p in result["processes"] if p["cpu_actor_mask"] == mask and p["role"] == "control"]
        observers = [p for p in result["processes"] if p["cpu_actor_mask"] == mask and p["role"] == "phase_profile"]
        pair = dict(cpu_actor_mask=mask, qualified=False)
        if len(controls) == len(observers) == 1 and controls[0]["status"] == observers[0]["status"] == "completed":
            control, observer = controls[0], observers[0]
            pair.update(control_job=control["job_id"], observer_job=observer["job_id"],
                        total_wall_relative_difference=abs(observer["target_wall_ns"] / control["target_wall_ns"] - 1),
                        same_POD_allocation=observer["callback_POD_bytes"] == control["callback_POD_bytes"])
            if control["complete_off_state_attention_mean_ns"] and observer["complete_off_state_attention_mean_ns"]:
                pair["complete_off_state_attention_mean_relative_difference"] = abs(observer["complete_off_state_attention_mean_ns"] / control["complete_off_state_attention_mean_ns"] - 1)
                quality = manifest["protocol"]["observer_quality"]
                pair["numeric_thresholds_passed"] = pair["same_POD_allocation"] and pair["total_wall_relative_difference"] <= quality["max_total_wall_relative_difference"] and pair["complete_off_state_attention_mean_relative_difference"] <= quality["max_complete_off_state_attention_mean_difference"]
                pair["possible_incidental_analysis_overlap_in_control"] = mask == 4
                pair["qualified"] = pair["numeric_thresholds_passed"] and mask != 4
                pair["qualification_scope"] = "observer neutrality in this one diagnostic pair only; not reusable price qualification"
                if mask == 4:
                    pair["qualification_refusal_reason"] = "First control possible incidental analysis overlap; timing endpoints and BLAS thread count unavailable"
            pair["phase_prices_automatically_transferred"] = False
            pair["one_pair_establishes_universal_observer_neutrality"] = False
        result["observer_pairs"].append(pair)
    result["generator"] = campaign.receipt(Path(__file__))
    result["supporting_sources"] = [campaign.receipt(Path(plans.__file__))]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.out)
    output = Path(args.out) / "callback-analysis.json"
    campaign.write(output, result)
    print(json.dumps(dict(output=campaign.receipt(output), observer_pairs=result["observer_pairs"], performance_parameters_fitted=False)))
