"""Offline CPU sample/worker-edge audit. Never fits or launches native work.

Read native results, program descriptors and requested-state events only. A
worker call's wall interval is evidence of an in-flight call, not CPU activity.
Heldout predictions and frozen parameter tables are deliberately never read.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct


def summary(values):
    values = list(values)
    return {"count": len(values), "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "population_cv": (statistics.pstdev(values) / statistics.mean(values)
                              if len(values) >= 2 and statistics.mean(values) > 0 else None),
            "cv_estimable_from_at_least_two_samples": len(values) >= 2}


def ns(ticks, hz):
    return ticks * 1_000_000_000 / hz


def classify(begin, end, segments):
    """Require complete containment; touching a right boundary is not crossing."""
    if end < begin:
        raise ValueError("negative action interval")
    for segment in segments:
        if segment["begin_tick"] <= begin and end <= segment["end_tick"]:
            steady = begin >= segment.get("calibration_begin_tick", segment["begin_tick"])
            return segment, "steady" if steady else "within_state_before_settle"
    return None, "crossing_or_outside_known_segments"


def worker_rows(native):
    rows = [row for block in native["blocks"] for row in block["iterations"]]
    previous = None
    for row in rows:
        if not row["enabled_check_tick"] <= row["begin_tick"] < row["end_tick"]:
            raise ValueError("worker decision/work ordering differs")
        if previous is not None and row["begin_tick"] < previous:
            raise ValueError("worker invocation intervals overlap")
        previous = row["end_tick"]
    return rows


def busy_overlap(begin, end, rows):
    ends = [row["end_tick"] for row in rows]
    total = 0
    for row in rows[bisect_right(ends, begin):]:
        if row["begin_tick"] >= end:
            break
        total += max(0, min(end, row["end_tick"]) - max(begin, row["begin_tick"]))
    return total


def worker_edges(native, go):
    """Do not invent disabled-loop observations, which v3 did not record."""
    rows = worker_rows(native)
    hz = native["clock"]["frequency_hz"]
    phases = list(go["phases"])
    edges, previous = [], 0
    for phase in phases:
        requested = phase["cpu_replay"]
        if requested != previous:
            edges.append({"tick": phase["begin_tick"], "requested": requested})
        previous = requested
    if previous:
        edges.append({"tick": go["end_tick"], "requested": 0})
    output = []
    for index, edge in enumerate(edges):
        tick = edge["tick"]
        window_end = edges[index + 1]["tick"] if index + 1 < len(edges) else go["end_tick"]
        row = dict(edge_tick=tick, requested=edge["requested"], request_window_end_tick=window_end)
        if edge["requested"]:
            accepted = [sample for sample in rows if tick <= sample["enabled_check_tick"] < window_end]
            first = accepted[0] if accepted else None
            row.update(accepted_call_count=len(accepted),
                       first_acceptance_delay_ns=ns(first["enabled_check_tick"] - tick, hz) if first else None,
                       first_call_begin_delay_ns=ns(first["begin_tick"] - tick, hz) if first else None,
                       acceptance_censored=first is None,
                       censoring_scope=(None if first else "no acceptance before request changed/end; delay is at least the request-window duration"),
                       observed_request_window_ns=ns(window_end - tick, hz))
        else:
            inflight = [sample for sample in rows
                        if sample["enabled_check_tick"] < tick < sample["end_tick"]]
            tail_end = max((sample["end_tick"] for sample in inflight), default=tick)
            invalid = [sample for sample in rows if tick <= sample["enabled_check_tick"] < window_end]
            row.update(inflight_calls_at_off_request=len(inflight),
                       last_pre_off_accepted_call_end_tick=tail_end if inflight else None,
                       retained_busy_tail_after_off_ns=ns(tail_end - tick, hz),
                       accepted_calls_while_requested_off=len(invalid),
                       stop_response_latency_ns=None,
                       stop_response_unknown_reason="disabled-loop checks are not recorded; last in-flight call end is not an observed stop acknowledgement")
        output.append(row)
    invalid_decisions = []
    for sample in rows:
        phase = next((p for p in phases if p["begin_tick"] <= sample["enabled_check_tick"] < p["end_tick"]), None)
        if phase is None or phase["cpu_replay"] != 1:
            invalid_decisions.append(sample)
    return {"process_id": native["process_id"], "call_count": len(rows), "edges": output,
            "acceptance_outside_cpu_on_request": invalid_decisions,
            "sampling_end_tick": native["sampling_end_tick"],
            "interval_scope": "completed target-call host wall; neither pure CPU active time nor utilization"}


def topology():
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        return {"known": False, "reason": "64-bit Windows API unavailable"}
    try:
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
        api.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        api.restype = wintypes.BOOL
        length = wintypes.DWORD()
        api(0, None, ctypes.byref(length))
        if not length.value:
            raise OSError(ctypes.get_last_error(), "topology size unavailable")
        buffer = ctypes.create_string_buffer(length.value)
        if not api(0, buffer, ctypes.byref(length)):
            raise OSError(ctypes.get_last_error(), "topology query failed")
        raw, offset, cores = buffer.raw[:length.value], 0, []
        while offset < len(raw):
            relation, size = struct.unpack_from("<II", raw, offset)
            if relation != 0 or size < 48 or offset + size > len(raw):
                raise ValueError("unexpected processor-core record layout")
            flags, efficiency = struct.unpack_from("<BB", raw, offset + 8)
            groups = struct.unpack_from("<H", raw, offset + 30)[0]
            if 32 + groups * 16 > size:
                raise ValueError("group affinity exceeds core record")
            masks = []
            for index in range(groups):
                mask, group = struct.unpack_from("<QH", raw, offset + 32 + index * 16)
                masks.append({"group": group, "mask": mask,
                              "logical_processors": [bit for bit in range(64) if mask & (1 << bit)]})
            cores.append({"core_record": len(cores), "smt_flag": bool(flags & 1),
                          "efficiency_class": efficiency, "group_masks": masks})
            offset += size
        selected = [core for core in cores if any(m["group"] == 0 and m["mask"] & 3 for m in core["group_masks"])]
        mask_details = {}
        for name, mask in (("legacy_main_cpu", 3), ("candidate_main_cpu", 5), ("candidate_gpu_worker", 48)):
            matching = [core["core_record"] for core in cores
                        if any(m["group"] == 0 and m["mask"] & mask for m in core["group_masks"])]
            covered = 0
            for core in cores:
                for group in core["group_masks"]:
                    if group["group"] == 0:
                        covered |= group["mask"] & mask
            mask_details[name] = {"mask": mask, "processor_group": 0, "core_records": matching,
                                  "physical_core_count": len(matching), "all_requested_logicals_present": covered == mask}
        return {"known": True, "provider": "GetLogicalProcessorInformationEx(RelationProcessorCore)",
                "physical_core_record_count": len(cores), "affinity_mask": 3, "processor_group": 0,
                "all_core_records": cores, "affinity_masks": mask_details,
                "candidate_cpu_gpu_physical_cores_disjoint": not set(mask_details["candidate_main_cpu"]["core_records"]) & set(mask_details["candidate_gpu_worker"]["core_records"]),
                "matching_core_records": selected,
                "logical0_and1_same_physical_core": any(
                    any(m["group"] == 0 and m["mask"] & 3 == 3 for m in core["group_masks"]) for core in cores),
                "measurement_hardware_load_started": False}
    except (OSError, ValueError, struct.error) as error:
        return {"known": False, "reason": str(error)}


def cpu_envelope(native):
    raw = native.get("cpu_diagnostics", {})
    wall, process, main = (raw.get(key) for key in ("sampling_wall_ns", "process_cpu_ns", "main_thread_cpu_ns"))
    return {"sampling_wall_ns": wall, "process_cpu_ns": process, "main_thread_cpu_ns": main,
            "non_main_cpu_ns": process - main if process is not None and main is not None else None,
            "raw_main_thread_cycles": raw.get("main_thread_cycles"),
            "main_cpu_per_wall": main / wall if main is not None and wall else None,
            "process_cpu_per_wall": process / wall if process is not None and wall else None,
            "scope": "whole sampling envelope, includes auxiliary GGML threads and waits; cannot attribute CPU time to individual actions or request states",
            "per_cpu_action_cpu_time_available": False}


def audit(campaign, split="profile"):
    campaign = Path(campaign).resolve()
    sources, jobs, failures, grouped = {}, [], [], defaultdict(list)

    def read(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    for directory in sorted((campaign / "jobs").iterdir()):
        if not directory.is_dir() or split != "all" and not directory.name.startswith(split + "-"):
            continue
        execution_path = directory / "execution.json"
        if not execution_path.exists():
            continue
        execution = read(execution_path)
        if execution.get("status") != "measured":
            failures.append({"job_id": directory.name, "status": execution.get("status"),
                             "error_type": execution.get("error_type"), "error": execution.get("error"),
                             "timed_sample_included": False})
            continue
        main, worker = read(directory / "main/result.json"), read(directory / "cpu-worker/result.json")
        events, go = read(directory / "events.json"), read(directory / "go.json")
        process = read(directory / "main/process.json")
        program = read(process["command"][1])
        bindings = {item["operation_id"]: item for item in program["bindings"]}
        calls = worker_rows(worker)
        observations = defaultdict(list)
        classifications = defaultdict(int)
        sample_rows = []
        for block in main["blocks"]:
            for action in block["action_intervals"]:
                index = action["action_index"]
                descriptor = program["actions"][index]
                if descriptor["kind"] != "COMPUTE" or descriptor["device"] != "cpu":
                    continue
                begin, end = action["begin_tick"], action["end_tick"]
                segment, category = classify(begin, end, events["known_segments"])
                classifications[category] += 1
                state = segment["state"] if segment else None
                key = (index, json.dumps(state, sort_keys=True), category)
                observations[key].append(action["elapsed_ns"])
                sample_rows.append({"iteration": action["iteration"], "action_index": index,
                                    "begin_tick": begin, "end_tick": end, "elapsed_ns": action["elapsed_ns"],
                                    "classification": category, "offered_state": state,
                                    "worker_inflight_call_wall_overlap_fraction": busy_overlap(begin, end, calls) / (end - begin) if end > begin else None})
        per_action = []
        for (index, state_text, category), durations in sorted(observations.items()):
            descriptor = program["actions"][index]
            row = {"action_index": index, "target_id": descriptor["target_id"],
                   "semantic_kind": bindings[descriptor["target_id"]]["kind"],
                   "classification": category, "offered_state": json.loads(state_text),
                   "duration_ns": summary(durations)}
            per_action.append(row)
            if category == "steady":
                grouped[(index, state_text)].append({"job_id": directory.name, "process_id": main["process_id"],
                                                     "sample_count": len(durations), "median_ns": statistics.median(durations),
                                                     "within_process_cv": row["duration_ns"]["population_cv"],
                                                     "semantic_kind": row["semantic_kind"], "target_id": row["target_id"]})
        jobs.append({"job_id": directory.name, "process_id": main["process_id"],
                     "instrumentation": main["instrumentation"], "cpu_action_classifications": dict(classifications),
                     "per_action_state": per_action, "cpu_action_observations": sample_rows,
                     "cpu_worker_request_edges": worker_edges(worker, go),
                     "main_cpu_envelope": cpu_envelope(main), "cpu_worker_cpu_envelope": cpu_envelope(worker),
                     "main_affinity": process.get("affinity"),
                     "cpu_worker_affinity": read(directory / "cpu-worker/process.json").get("affinity"),
                     "cpu_action_timing_available": main["instrumentation"] != "bare"})
    aggregate = []
    for (index, state_text), rows in sorted(grouped.items()):
        aggregate.append({"action_index": index, "semantic_kind": rows[0]["semantic_kind"],
                          "target_id": rows[0]["target_id"], "offered_state": json.loads(state_text),
                          "independent_pid_count": len({row["process_id"] for row in rows}),
                          "per_pid_steady_samples": rows,
                          "between_pid_medians_ns": summary(row["median_ns"] for row in rows),
                          "minimum_steady_samples_per_pid": min(row["sample_count"] for row in rows),
                          "singleton_pids": sum(row["sample_count"] == 1 for row in rows)})
    return {"schema": "state-response-cpu-diagnosis-v1", "campaign": str(campaign), "split": split,
            "sources": sources, "topology": topology(), "successful_jobs": jobs, "failed_jobs": failures,
            "steady_cpu_action_groups": aggregate,
            "scope": {"new_native_work_started": False, "model_parameters_changed": False,
                      "heldout_predictions_read": False, "worker_wall_is_cpu_utilization": False,
                      "steady_means_request_steady_not_actual_cpu_occupancy": True,
                      "missing_action_samples_in_bare_are_unknown_not_zero": True,
                      "single_sample_cv_is_unknown_not_zero": True}}


def report_text(report):
    lines = ["# CPU state-response audit", "",
             f"Successful jobs: {len(report['successful_jobs'])}; failed jobs retained: {len(report['failed_jobs'])}.", "",
             "This report reads native wall intervals and requested schedules only. It does not fit parameters or read heldout predictions.", "",
             "## Affinity", "", json.dumps(report["topology"], ensure_ascii=False), "",
             "## Steady CPU action samples", "",
             "| Action | Kind | CPU/GPU requests | PIDs | Samples per PID | Between-PID CV |", "|---|---|---|---|---|---|"]
    for row in report["steady_cpu_action_groups"]:
        state = row["offered_state"]
        cv = row["between_pid_medians_ns"]["population_cv"]
        lines.append(f"| {row['action_index']} | {row['semantic_kind']} | {state['cpu_replay']}/{state['gpu_replay']} | {row['independent_pid_count']} | {', '.join(str(p['sample_count']) for p in row['per_pid_steady_samples'])} | {cv:.3f} |" if cv is not None else f"| {row['action_index']} | {row['semantic_kind']} | {state['cpu_replay']}/{state['gpu_replay']} | {row['independent_pid_count']} | {', '.join(str(p['sample_count']) for p in row['per_pid_steady_samples'])} | unknown |")
    lines.extend(["", "## Worker request edges", "",
                  "An enabled_check_tick establishes which request accepted a call. Completed calls are retained across off edges. Disabled-loop acknowledgements are unavailable; the end of an in-flight call is not a measured stop acknowledgement.", ""])
    for job in report["successful_jobs"]:
        lines.append(f"### {job['job_id']}")
        lines.append("")
        for edge in job["cpu_worker_request_edges"]["edges"]:
            if edge["requested"]:
                delay = edge["first_acceptance_delay_ns"]
                detail = f"first acceptance {delay / 1e6:.3f} ms after request" if delay is not None else "no acceptance observed before request changed/end (censored)"
                lines.append(f"- ON: {detail}; accepted calls {edge['accepted_call_count']}.")
            else:
                lines.append(f"- OFF: retained in-flight tail {edge['retained_busy_tail_after_off_ns'] / 1e6:.3f} ms; actual disabled acknowledgement unknown.")
        lines.append("")
    lines.extend(["## Limits", "", "All CPU counters are whole-sampling envelopes, not per-CPU-action measurements. GPU-target diagnostic counters cannot identify CPU projection/QK/FFN wall versus CPU time. Population CV is descriptive; a single sample has unknown CV. Steady request segments do not prove steady actual contention. Worker in-flight wall overlap is not CPU utilization.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("profile", "heldout", "all"), default="profile")
    args = parser.parse_args()
    report = audit(args.campaign, args.split)
    report["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cpu-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.output / "CPU_AUDIT.md").write_text(report_text(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "audited", "successful_jobs": len(report["successful_jobs"]),
                      "failed_jobs": len(report["failed_jobs"]), "output": str(args.output.resolve()),
                      "hardware_executed": False}))


if __name__ == "__main__":
    main()
