"""Compact diagnostic evidence; this is not a calibrated price catalog."""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
from pathlib import Path

from profiling.dynamic_validation import external_v6 as campaign


def digest(out):
    out = Path(out)
    source = out / "callback-analysis.json"
    analysis = campaign.read(source)
    executions_path = out / "executions.json"
    executions = campaign.read(executions_path)
    by_id = {entry["job"]["id"]: entry for entry in executions}
    result = dict(schema="dynamic-external-callback-digest-6", analysis=campaign.receipt(source),
                  executions=campaign.receipt(executions_path), declared_processes=4,
                  completed_processes=sum(entry["status"] == "completed" for entry in executions),
                  background_context=analysis["background_context"], observer_pairs=analysis["observer_pairs"],
                  processes=[], phase_prices_qualified=False, performance_parameters_fitted=False,
                  predictions_frozen=False, external_fluctuation_heldout_accuracy_evaluated=False,
                  host_API_wall_is_not_CPU_active_time=True, GPU_kernel_or_memcpy_activity_measured=False,
                  numerical_neutrality_is_not_response_model_accuracy=True)
    for process in analysis["processes"]:
        entry = by_id[process["job_id"]]
        record = {key: process[key] for key in ("job_id", "cpu_actor_mask", "role", "status")}
        if process["status"] != "completed":
            record["error"] = process.get("error")
            result["processes"].append(record)
            continue
        partitions = {p["phase_id"]: p for p in process["API_partitions"]["phase_partitions"]}
        def action(row):
            partition = partitions.get(row["phase_id"])
            return dict(phase_id=row["phase_id"], global_replay=row["global_replay"],
                        begin_since_epoch_ns=row["begin_since_epoch_ns"], end_since_epoch_ns=row["end_since_epoch_ns"],
                        host_wall_ns=row["host_wall_ns"], first_CPU_release_spanning=row["first_CPU_release_spanning"],
                        second_CPU_release_spanning=row["second_CPU_release_spanning"],
                        selected_API_union_ns=partition["selected_API_union_ns"] if partition else None,
                        outside_selected_API_union_ns=partition["outside_selected_API_union_ns"] if partition else None,
                        API_evidence_status="observed" if partition else "not_collected_in_control")
        record.update(native=process["native"], target_wall_ns=process["target_wall_ns"],
                      longest_attention=action(max(process["attention_actions"], key=lambda x: x["host_wall_ns"])),
                      CPU_release_spanning_attention=[action(x) for x in process["attention_actions"]
                                                     if x["first_CPU_release_spanning"] or x["second_CPU_release_spanning"]],
                      callback_phases=process["API_partitions"]["phases"], callback_records=process["API_partitions"]["records"],
                      non_fitted_observation_cells=len(process["observations"]),
                      all_coordinate_sequence_signatures=process["coordinate_sequence_signatures"],
                      complete_off_state_attention_count=process["complete_off_state_attention_count"],
                      complete_off_state_attention_mean_ns=process["complete_off_state_attention_mean_ns"],
                      callback_materialization_ns_outside_target=process["callback_materialization_ns_outside_target"],
                      target_time_materialization_ns=0, numerical_and_clock_oracle=entry["verification"],
                      possible_incidental_analysis_overlap=process["job_id"] == analysis["background_context"]["first_control_job"])
        if process["role"] == "phase_profile":
            native_path = Path(process["native"]["path"])
            campaign.require(campaign.sha(native_path) == process["native"]["sha256"], "native callback evidence changed")
            native = campaign.read(native_path)
            callbacks = native["runtime_callbacks"]
            rows, metadata = callbacks["callback_records"], callbacks["metadata"]
            hz, epoch = native["clock"]["frequency_hz"], native["target_expected_origin_tick"]
            def evidence(row):
                m = metadata[row["metadata_index"]]
                return dict(ordinal=row["ordinal"], phase_id=row["phase_id"], correlation_id=row["correlation_id"],
                            enter_tick=row["enter_tick"], exit_tick=row["exit_tick"],
                            enter_since_epoch_ns=(row["enter_tick"] - epoch) * 1e9 / hz,
                            exit_since_epoch_ns=(row["exit_tick"] - epoch) * 1e9 / hz,
                            inclusive_API_wall_ns=(row["exit_tick"] - row["enter_tick"]) * 1e9 / hz,
                            return_status=row["return_status"], metadata=m,
                            CPU_active_or_enqueue_acceptance_identified=False)
            record["longest_API_calls"] = [evidence(row) for row in heapq.nlargest(5, rows, key=lambda x: x["exit_tick"] - x["enter_tick"])]
            record["API_family_record_counts"] = dict(Counter(metadata[row["metadata_index"]]["name"] for row in rows))
            record["API_calls_above_10ms"] = sum((row["exit_tick"] - row["enter_tick"]) * 1000 > hz * 10 for row in rows)
            record["callback_integrity"] = {key: callbacks[key] for key in (
                "all_selected_calls_retained", "overflow_records", "pair_errors", "clock_errors", "unfinished_calls",
                "invalid_time_calls", "nonzero_or_unknown_return_status_calls")}
            del native, callbacks, rows, metadata
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
