"""Correlated first-pulse Attention diagnosis; no fitting from target totals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from profiling.dynamic_validation import external_v5 as campaign
from profiling.dynamic_validation.h2d_v3_analysis import interval_union, physical_by_external, qpc_ns


def physical(stage, records, hz):
    begin, end = qpc_ns(stage["begin_tick"], hz), qpc_ns(stage["end_tick"], hz)
    api_begin, api_end = qpc_ns(stage["graph_compute_enter_tick"], hz), qpc_ns(stage["graph_compute_return_tick"], hz)
    kernels = [r for r in records if r["kind"] == "kernel" and r.get("timestamp_valid")]
    runtime = [r for r in records if r["kind"] == "runtime_api" and r.get("timestamp_valid")]
    clipped = lambda rows: interval_union((max(api_begin, r["begin_ns"]), min(api_end, r["end_ns"]))
                                         for r in rows if max(api_begin, r["begin_ns"]) < min(api_end, r["end_ns"]))
    reasons = []
    invalid = [r for r in records if "timestamp_valid" in r and not r["timestamp_valid"]]
    if any(r["kind"] == "kernel" for r in invalid):
        reasons.append("Some explicitly correlated kernel timestamps are unknown")
    if not kernels:
        reasons.append("No explicitly correlated valid kernel activity")
    if any(not begin <= r["begin_ns"] < r["end_ns"] <= end for r in kernels):
        reasons.append("Kernel activity not enclosed by measured Attention completion response")
    active = interval_union((r["begin_ns"], r["end_ns"]) for r in kernels) if kernels else None
    first = min((r["begin_ns"] for r in kernels), default=None)
    last = max((r["end_ns"] for r in kernels), default=None)
    return dict(supported=not reasons, unsupported_reasons=reasons, kernel_activity_records=len(kernels),
                kernel_names=dict(Counter(r.get("name", "unknown") for r in kernels)),
                kernel_activity_union_ns=active, device_kernel_envelope_ns=last - first if kernels else None,
                gaps_inside_device_envelope_ns=last - first - active if kernels else None,
                host_entry_to_first_kernel_ns=first - begin if kernels else None,
                last_kernel_to_host_completion_ns=end - last if kernels else None,
                runtime_API_union_in_graph_compute_ns=clipped(runtime),
                runtime_launch_API_union_ns=clipped([r for r in runtime if "Launch" in (r.get("name") or "")]),
                runtime_synchronize_union_in_graph_compute_ns=clipped([r for r in runtime if "Synchronize" in (r.get("name") or "")]),
                graph_compute_outside_recorded_runtime_union_ns=api_end - api_begin - clipped(runtime),
                invalid_timestamp_records=len(invalid), driver_API_intervals_added_to_runtime=False,
                runtime_records_present=bool(runtime),
                runtime_timestamps_valid=bool(runtime) and not any(r["kind"] == "runtime_api" for r in invalid),
                all_action_kernel_activity_identified=False,
                support_role="valid explicitly correlated observed kernel subset, not proof of complete kernel association",
                performance_parameter_qualified=False,
                device_activity_added_to_host_wait=False, device_activity_is_uncontended_FLOPS=False,
                raw_correlated_records=records)


def counter_difference(before, after):
    result = dict(scope="counter query/correlation bracket around observed action; no Ready/Blocked separation", cycles_converted_to_time=False)
    for field, name, multiplier in (("thread_user_100ns", "user_ns", 100), ("thread_kernel_100ns", "kernel_ns", 100), ("thread_cycles", "cycles", 1)):
        a, b = before.get(field), after.get(field)
        result[name] = (b - a) * multiplier if a is not None and b is not None and b >= a else None
    return result


def analyze(out):
    out = Path(out).resolve()
    manifest = campaign.read(out / "manifest.json")
    campaign.base.check(manifest)
    executions = campaign.read(out / "executions.json")
    result = dict(schema="dynamic-external-stage-analysis-5", manifest=campaign.receipt(out / "manifest.json"),
                  executions=campaign.receipt(out / "executions.json"), declared_processes=len(manifest["jobs"]), processes=[],
                  performance_parameters_fitted=False, whole_target_used_as_fit_label=False, device_times_added_to_host_wait=False,
                  scope="New diagnostic process/observer boundaries; no claim that app environment or CUPTI makes these timings equivalent to external-v1")
    for execution in executions:
        job = execution["job"]
        process = dict(job_id=job["id"], init_policy=job["request"]["init_policy"], instrumentation=job["request"]["instrumentation"], status=execution["status"])
        if execution["status"] != "completed":
            process["error"] = execution.get("error")
            result["processes"].append(process)
            continue
        native_path = Path(execution["native"]["path"])
        campaign.require(campaign.sha(native_path) == execution["native"]["sha256"], "bound native evidence changed")
        native = campaign.read(native_path)
        hz, epoch = native["clock"]["frequency_hz"], native["target_expected_origin_tick"]
        grouped, graph = ({}, None) if native["cupti"] is None else physical_by_external(native["cupti"])
        attention = []
        for block in native["blocks"]:
            for action in block["actions"]:
                if action["action_index"] != native["attention_action_index"]:
                    continue
                start = (action["begin_tick"] - epoch) * 1e9 / hz
                end = (action["end_tick"] - epoch) * 1e9 / hz
                row = dict(global_replay=action["global_replay"], action_index=action["action_index"], begin_tick=action["begin_tick"], end_tick=action["end_tick"],
                           begin_since_epoch_ns=start, end_since_epoch_ns=end, host_wall_ns=action["wall_ns"], external_id=action["external_id"],
                           first_joint_intersection_ns=max(0, min(end, 4e9) - max(start, 0)),
                           first_CPU_release_spanning=start < 4e9 < end,
                           distance_completion_from_planned_first_CPU_OFF_ns=end - 4e9,
                           attention_host_stages=action["attention_host_stages"])
                if action["attention_host_stages"] is not None:
                    campaign.validate_attention_stages(action["attention_host_stages"], hz)
                    row["physical"] = physical(action["attention_host_stages"], grouped.get(action["external_id"], []), hz)
                    row["thread_counter_difference"] = counter_difference(action["thread_counter_before"], action["thread_counter_after_diagnostic"])
                attention.append(row)
        joint = [x for x in attention if x["first_joint_intersection_ns"] > 0]
        process.update(native=execution["native"], actor_receipts=native["actors"],
                       target_wall_ns_is_diagnostic_only=native["target_wall_ns"], observer_flush_gap_ns=native["observer_flush_gap_in_target_wall_ns"],
                       threads_after_initialization=native["threads_after_initialization"], correlation_graph=graph,
                       cupti_integrity=native["cupti"]["all_requested_records_retained"] if native["cupti"] else None,
                       observed_attention_actions=sum(x["attention_host_stages"] is not None for x in attention),
                       identified_observed_attention=sum(x.get("physical", {}).get("supported", False) for x in attention),
                       first_joint_intersecting_actions=len(joint), first_CPU_release_spanning_actions=sum(x["first_CPU_release_spanning"] for x in joint),
                       first_joint_calls_above_100ms=sum(x["host_wall_ns"] > 100_000_000 for x in joint),
                       attention_actions=attention)
        result["processes"].append(process)
    result["generator"] = campaign.receipt(Path(__file__))
    result["supporting_sources"] = [campaign.receipt(Path(physical_by_external.__code__.co_filename)),
                                     campaign.receipt(campaign.ROOT / "profiling/h2d_history/analysis.py")]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.out)
    output = Path(args.out) / "attention-stage-analysis.json"
    campaign.write(output, result)
    print(json.dumps(dict(output=campaign.receipt(output), completed=sum(x["status"] == "completed" for x in result["processes"]), declared=result["declared_processes"], performance_parameters_fitted=False)))
