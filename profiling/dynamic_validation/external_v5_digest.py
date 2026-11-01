"""Small evidence-bound digest of an already completed v5 diagnosis."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from profiling.dynamic_validation import external_v5 as campaign
from profiling.dynamic_validation.h2d_v3_analysis import interval_union


def digest(out):
    out = Path(out)
    source = out / "attention-stage-analysis.json"
    data = campaign.read(source)
    result = dict(schema="dynamic-external-diagnostic-digest-5", analysis=campaign.receipt(source),
                  processes=[], performance_parameters_fitted=False,
                  raw_action_or_activity_samples_removed=False, OS_ready_blocked_cause_identified=False,
                  observer_total_minus_flush_is_not_a_bare_latency_estimate=True,
                  interpretation="Fixed external actor plans continue during observer flush; subtracting flush does not restore original actor exposure.")
    for process in data["processes"]:
        record = {k: process[k] for k in ("job_id", "init_policy", "instrumentation", "status")}
        if process["status"] != "completed":
            result["processes"].append(record)
            continue
        joint = [x for x in process["attention_actions"] if x["first_joint_intersection_ns"] > 0]
        longest = max(joint, key=lambda x: x["host_wall_ns"])
        record.update(native=process["native"], first_joint_longest=dict(
            replay=longest["global_replay"], start_since_epoch_ns=longest["begin_since_epoch_ns"],
            end_since_epoch_ns=longest["end_since_epoch_ns"], host_wall_ns=longest["host_wall_ns"],
            CPU_release_spanning=longest["first_CPU_release_spanning"]),
            target_wall_ns_diagnostic=process["target_wall_ns_is_diagnostic_only"],
            observer_flush_ns=process["observer_flush_gap_ns"])
        observed = longest.get("physical")
        if observed is not None:
            rows = observed["raw_correlated_records"]
            runtime = [r for r in rows if r["kind"] == "runtime_api" and r["timestamp_valid"]]
            kernels = [r for r in rows if r["kind"] == "kernel" and r["timestamp_valid"]]
            copies = [r for r in rows if r["kind"] == "memcpy" and r["timestamp_valid"]]
            launch = [r for r in runtime if (r.get("name") or "").startswith("cudaLaunchKernel")]
            copying = [r for r in runtime if (r.get("name") or "").startswith("cudaMemcpyAsync")]
            device = kernels + copies
            groups = {}
            for row in runtime:
                groups.setdefault(row.get("name") or "unknown", []).append(row)
            record["first_joint_physical"] = dict(
                kernel_records=len(kernels), memcpy_records=len(copies), memcpy_bytes=sum(r["bytes"] for r in copies),
                memcpy_kinds=dict(Counter(str(r["copy_kind"]) for r in copies)),
                observed_launch_API_count=len(launch), observed_memcpy_API_count=len(copying),
                all_observed_launch_correlation_IDs_match_kernel_IDs=Counter(r["correlation_id"] for r in launch) == Counter(r["correlation_id"] for r in kernels),
                all_observed_memcpy_correlation_IDs_match_device_copy_IDs=Counter(r["correlation_id"] for r in copying) == Counter(r["correlation_id"] for r in copies),
                kernel_memcpy_combined_union_ns=interval_union((r["begin_ns"], r["end_ns"]) for r in device),
                maximum_kernel_record_ns=max(r["end_ns"] - r["begin_ns"] for r in kernels),
                maximum_memcpy_record_ns=max(r["end_ns"] - r["begin_ns"] for r in copies),
                host_graph_compute_ns=longest["attention_host_stages"]["graph_compute_inclusive_ns"],
                host_extra_sync_ns=longest["attention_host_stages"]["extra_sync_ns"],
                host_entry_to_observed_first_kernel_ns=observed["host_entry_to_first_kernel_ns"],
                observed_last_kernel_to_host_completion_ns=observed["last_kernel_to_host_completion_ns"],
                runtime_API_unions=[dict(name=name, count=len(records), union_ns=interval_union((r["begin_ns"], r["end_ns"]) for r in records))
                                    for name, records in groups.items()],
                thread_counters=longest["thread_counter_difference"],
                physical_parameter_qualified=False, full_action_kernel_set_independently_proven=False)
        result["processes"].append(record)
    result["generator"] = campaign.receipt(Path(__file__))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = digest(args.out)
    output = Path(args.out) / "DIAGNOSTIC_DIGEST.json"
    campaign.write(output, result)
    print(json.dumps(campaign.receipt(output)))
