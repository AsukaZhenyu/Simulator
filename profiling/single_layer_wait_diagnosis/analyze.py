"""Attribute timed PV ranges to CUDA APIs and correlated GPU activity.

Uses only stdlib and read-only SQLite. Nsight timestamps are compared only with
other Nsight timestamps; native QPC clocks are never mapped to this timeline.
Profiler-instrumented observations cannot become timing parameters or accuracy
acceptance. API spans, GPU spans and GPU activity overlap and are not additive.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_length(intervals):
    values = sorted(intervals)
    require(all(type(begin) is int and type(end) is int and 0 <= begin <= end for begin, end in values),
            "invalid integer timestamp interval")
    if not values:
        return 0
    total, left, right = 0, *values[0]
    for begin, end in values[1:]:
        if begin > right:
            total += right - left
            left, right = begin, end
        else:
            right = max(right, end)
    return total + right - left


def stats(values):
    require(bool(values), "statistics need observations")
    ordered = sorted(values)
    quantile = lambda p: ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {"n": len(values), "min": ordered[0], "mean": statistics.mean(values),
            "p10": quantile(.10), "p50": quantile(.50), "p90": quantile(.90),
            "p99": quantile(.99), "max": ordered[-1]}


def rows(connection, query, arguments=()):
    return [dict(row) for row in connection.execute(query, arguments)]


def validate_tables(connection):
    required = {
        "StringIds": {"id", "value"},
        "NVTX_EVENTS": {"start", "end", "category", "text", "textId", "globalTid", "endGlobalTid", "eventType"},
        "CUPTI_ACTIVITY_KIND_RUNTIME": {"start", "end", "globalTid", "correlationId", "nameId", "returnValue"},
        "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "globalPid", "correlationId", "contextId", "streamId", "shortName"},
        "CUPTI_ACTIVITY_KIND_MEMCPY": {"start", "end", "globalPid", "correlationId", "contextId", "streamId", "bytes", "copyKind"},
        "ANALYSIS_FILE": {"globalPid", "content"},
        "TARGET_INFO_CUDA_CONTEXT_INFO": {"processId", "contextId"},
        "DIAGNOSTIC_EVENT": {"severity", "text"},
        "META_DATA_EXPORT": {"name", "value"},
    }
    for table, fields in required.items():
        present = {row[1] for row in connection.execute("PRAGMA table_info(" + table + ")")}
        require(fields <= present, "unsupported/missing Nsight table columns: " + table)


def correlate_range(host, apis, gpu_by_correlation, global_pid, contexts):
    """Compute attributed intervals within one completed host action range."""
    begin, end, thread = host["start"], host["end"], host["globalTid"]
    require(type(begin) is int and type(end) is int and end > begin, "invalid completed NVTX range")
    require(bool(apis), "target range has no CUDA APIs")
    require(all(begin <= api["start"] < api["end"] <= end and api["globalTid"] == thread and api["returnValue"] == 0 for api in apis),
            "CUDA API range/thread/return status differs")
    require(len({api["correlationId"] for api in apis}) == len(apis), "duplicate target API correlation")
    synchronization = [api for api in apis if "Synchronize" in api["name"]]
    require(len(synchronization) == 2 and all(api["name"].startswith("cudaStreamSynchronize_") for api in synchronization),
            "expected exactly two completed stream synchronization APIs")
    other = [api for api in apis if api not in synchronization]
    activities, post_return_delays, launch_to_start = [], [], []
    for api in apis:
        correlated = gpu_by_correlation.get(api["correlationId"], [])
        launches = api["name"].startswith("cudaLaunchKernel_")
        copies = api["name"].startswith("cudaMemcpyAsync_")
        if launches or copies:
            require(bool(correlated), "missing correlated GPU activity for " + api["name"])
        for activity in correlated:
            require(activity["globalPid"] == global_pid and activity["contextId"] in contexts,
                    "GPU activity process/context differs")
            require(begin <= activity["start"] < activity["end"] <= end,
                    "correlated GPU activity escapes completed host range")
            require((launches and activity["kind"] == "kernel") or (copies and activity["kind"] == "memcpy"),
                    "CUDA API/GPU activity kind correlation differs")
            # GPU work may begin before its host API returns. Keep signed
            # differences; do not convert API elapsed time into pure queueing.
            require(activity["start"] >= api["start"], "GPU activity predates correlated host submission")
            post_return_delays.append(activity["start"] - api["end"])
            launch_to_start.append(activity["start"] - api["start"])
            activities.append(activity)
    require(bool(activities), "missing target GPU work")
    active = lambda selected: union_length([(value["start"], value["end"]) for value in selected])
    kernels = [activity for activity in activities if activity["kind"] == "kernel"]
    copies = [activity for activity in activities if activity["kind"] == "memcpy"]
    span = max(activity["end"] for activity in activities) - min(activity["start"] for activity in activities)
    api_union = active(apis)
    host_ns, gpu_ns = end - begin, active(activities)
    sync = [api["end"] - api["start"] for api in synchronization]
    by_name = defaultdict(list)
    for api in apis:
        by_name[api["name"]].append(api["end"] - api["start"])
    maximum = max(apis, key=lambda api: api["end"] - api["start"])
    return {"host_nvtx_span_ns": host_ns, "runtime_api_union_ns": api_union,
            "host_outside_recorded_runtime_api_ns": host_ns - api_union,
            "non_sync_runtime_api_union_ns": active(other),
            "first_sync_api_ns": sync[0], "second_sync_api_ns": sync[1],
            "sync_api_union_ns": active(synchronization),
            "pre_first_sync_host_span_ns": synchronization[0]["start"] - begin,
            "correlated_gpu_activity_union_ns": gpu_ns,
            "correlated_kernel_union_ns": active(kernels), "correlated_memcpy_union_ns": active(copies),
            "correlated_gpu_activity_span_ns": span,
            "correlated_span_without_target_activity_ns": span - gpu_ns,
            "host_without_correlated_target_gpu_activity_ns": host_ns - gpu_ns,
            "gpu_activity_to_host_ratio": gpu_ns / host_ns,
            "sync_api_to_host_ratio": active(synchronization) / host_ns,
            "non_sync_api_to_host_ratio": active(other) / host_ns,
            "kernel_count": len(kernels), "memcpy_count": len(copies), "runtime_api_count": len(apis),
            "kernel_name_counts": dict(Counter(activity.get("kernel_name", "unknown") for activity in kernels)),
            "memcpy_kind_and_payload_counts": dict(Counter("kind_" + str(activity.get("copyKind", "unknown")) +
                "_bytes_" + str(activity.get("bytes", "unknown")) for activity in copies)),
            "post_api_return_to_gpu_start_ns": post_return_delays,
            "host_api_start_to_gpu_start_ns": launch_to_start,
            "post_return_delay_semantics": "signed_GPU_start_minus_host_API_return; negative_means_overlap; not_pure_queue_wait",
            "longest_api": {"name": maximum["name"], "duration_ns": maximum["end"] - maximum["start"], "correlation_id": maximum["correlationId"]},
            "api_by_name": {name: {"count": len(values), "inclusive_sum_ns": sum(values), "max_ns": max(values)} for name, values in by_name.items()},
            "sync_apis": [{"name": api["name"], "duration_ns": api["end"] - api["start"], "correlation_id": api["correlationId"]} for api in synchronization]}


def analyze_trace(connection, native):
    connection.row_factory = sqlite3.Row
    validate_tables(connection)
    require(native["format_version"] == "single-layer-context-pair-native-v2" and native["pair_kind"] == "context_vs_isolated", "unsupported native diagnostic")
    nvtx = native["nvtx_diagnostic"]
    warmup, count = native["warmup_per_block"], native["samples_per_block"]
    expected = 1 + warmup + count
    require(nvtx["enabled"] is True and nvtx["target_range_count_per_block"] == expected and
            nvtx["target_timed_range_count_per_block"] == count and nvtx["target_validation_range_count_per_block"] == 1 and
            nvtx["target_warmup_range_count_per_block"] == warmup and nvtx["usable_for_parameter_fit"] is False and
            nvtx["usable_for_heldout_acceptance"] is False, "NVTX target-selection contract differs")
    require(nvtx["categories"]["block"] == 1 and nvtx["categories"]["gpu_compute"] == 11,
            "NVTX category contract differs")
    require(native["block_order"] in {"ABBA", "BAAB"} and len(native["blocks"]) == 4 and
            all(block["block_index"] == index and block["label"] == native["block_order"][index] and
                block["raw"]["mode"] == ("isolated-components" if block["label"] == "A" else "context-trace")
                for index, block in enumerate(native["blocks"])), "native block mode/order inventory differs")
    require(native["diagnostic_only"] is True and native["usable_for_parameter_fit"] is False and
            native["usable_for_heldout_acceptance"] is False, "native cannot become primary timing evidence")
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    notes = rows(connection, "SELECT severity,text FROM DIAGNOSTIC_EVENT")
    require(not any(re.search(r"\b(dropped|lost)\s+(events|records|buffers)|data\s+loss", row["text"], re.I) for row in notes), "Nsight reports lost/dropped events")
    global_pids = set()
    for row in rows(connection, "SELECT globalPid,content FROM ANALYSIS_FILE"):
        for line in row["content"].splitlines():
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get("process_id") == native["process_id"] and value.get("status") == "ok":
                global_pids.add(row["globalPid"])
    require(len(global_pids) == 1, "cannot uniquely bind Nsight globalPid to native stdout PID")
    global_pid = next(iter(global_pids))
    contexts = {row["contextId"] for row in rows(connection, "SELECT processId,contextId FROM TARGET_INFO_CUDA_CONTEXT_INFO") if row["processId"] == native["process_id"]}
    require(bool(contexts), "CUDA context process ID differs from native")
    ranges = rows(connection, "SELECT * FROM NVTX_EVENTS")
    for row in ranges:
        row["message"] = row["text"] if row["text"] is not None else strings.get(row["textId"])
    parents = [row for row in ranges if row["category"] == 1 and row["message"] in nvtx["block_range_names"]]
    require(len(parents) == 4 and {row["message"] for row in parents} == {"pair_block_" + str(i) for i in range(4)}, "missing/extra NVTX parent blocks")
    parents.sort(key=lambda row: row["start"])
    thread_ids = {row["globalTid"] for row in parents}
    require(len(thread_ids) == 1, "parent ranges use different threads")
    thread = next(iter(thread_ids))
    require(thread - thread % (1 << 24) == global_pid, "NVTX thread belongs to another Nsight global process")
    target = native["target_action"]["target_id"]
    require(native["target_action"]["kind"] == "COMPUTE" and native["target_action"]["device"] == "gpu", "selected target is not GPU COMPUTE")
    targets = [row for row in ranges if row["category"] == 11 and row["message"] == target]
    require(len(targets) == 4 * expected, "missing/extra target NVTX ranges")
    apis = rows(connection, "SELECT start,end,globalTid,correlationId,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=? ORDER BY start", (thread,))
    require(len({api["correlationId"] for api in apis}) == len(apis), "ambiguous runtime correlation IDs")
    for api in apis:
        api["name"] = strings[api["nameId"]]
    starts = [api["start"] for api in apis]
    gpu = defaultdict(list)
    for table, kind, extra in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel", ",shortName"),
                               ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy", ",bytes,copyKind")):
        for activity in rows(connection, "SELECT start,end,globalPid,correlationId,contextId,streamId" + extra + " FROM " + table):
            if activity["globalPid"] == global_pid:
                activity["kind"] = kind
                if kind == "kernel":
                    activity["kernel_name"] = strings[activity["shortName"]]
                gpu[activity["correlationId"]].append(activity)
    blocks = []
    previous = -1
    for index, parent in enumerate(parents):
        require(parent["message"] == "pair_block_" + str(index) and parent["end"] is not None and
                previous <= parent["start"] < parent["end"] and parent["endGlobalTid"] in (None, thread) and
                parent["eventType"] == 59, "NVTX parent order/clock/thread differs")
        previous = parent["end"]
        selected = sorted([row for row in targets if parent["start"] <= row["start"] and row["end"] is not None and row["end"] <= parent["end"]], key=lambda row: row["start"])
        require(len(selected) == expected and all(row["globalTid"] == thread and row["endGlobalTid"] in (None, thread) and
                row["eventType"] == 59 for row in selected), "target range count/thread/parent mismatch")
        require(all(a["end"] <= b["start"] for a, b in zip(selected, selected[1:])), "target ranges overlap")
        invocations = []
        for sample, host in enumerate(selected[-count:]):
            first = bisect_left(starts, host["start"])
            last = bisect_left(starts, host["end"])
            require(first == 0 or apis[first - 1]["end"] <= host["start"], "CUDA API crosses NVTX start boundary")
            inside = apis[first:last]
            invocation = correlate_range(host, inside, gpu, global_pid, contexts)
            invocations.append({"sample_index": sample, **invocation})
        scalar_names = [name for name, value in invocations[0].items() if type(value) in (int, float) and name != "sample_index"]
        summary = {name: stats([row[name] for row in invocations]) for name in scalar_names}
        summary["post_api_return_to_gpu_start_ns"] = stats([value for row in invocations for value in row["post_api_return_to_gpu_start_ns"]])
        summary["host_api_start_to_gpu_start_ns"] = stats([value for row in invocations for value in row["host_api_start_to_gpu_start_ns"]])
        blocks.append({"block_index": index, "label": native["block_order"][index],
            "target_ranges_total": len(selected), "validation_ranges_excluded": 1,
            "warmup_ranges_excluded": warmup, "timed_ranges": count,
            "nsight_parent_start_ns": parent["start"], "nsight_parent_end_ns": parent["end"],
            "statistics": summary, "invocations": invocations})
    available_tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    capture = dict(connection.execute("SELECT name,value FROM META_DATA_CAPTURE")) if "META_DATA_CAPTURE" in available_tables else {}
    return {"native_process_id": native["process_id"], "nsight_global_pid": global_pid,
            "nsight_main_global_tid": thread, "block_order": native["block_order"],
            "target_action_index": native["target_action_index"], "target_operation_id": target,
            "blocks": blocks, "diagnostic_events": notes,
            "export_metadata": dict(connection.execute("SELECT name,value FROM META_DATA_EXPORT")),
            "selected_capture_options": {name: value for name, value in capture.items() if
                name.endswith(("COLLECT_CUDA_TRACE", "COLLECT_NVTX_TRACE", "CUDA_SKIP_SOME_API_CALLS"))},
            "complete_target_inventory_verified": True, "all_selected_launch_copy_correlations_verified": True,
            "kernel_count_distribution": dict(Counter(row["kernel_count"] for block in blocks for row in block["invocations"])),
            "memcpy_count_distribution": dict(Counter(row["memcpy_count"] for block in blocks for row in block["invocations"]))}


def analyze_campaign(campaign):
    campaign = Path(campaign).resolve()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    manifest, phase = read(campaign / "manifest.json"), read(campaign / "phase.json")
    require(manifest["schema"] == "single-layer-wait-nsys-campaign-v1" and
            manifest["diagnostic_only"] is True and manifest["usable_for_parameter_fit"] is False and
            manifest["usable_for_accuracy_acceptance"] is False, "unsupported profiler campaign")
    require(phase["successful"] == len(manifest["declared_orders"]) == 2 and phase["failed"] == 0 and
            phase["sources_verified_at_end"] is True and {row["order"] for row in phase["records"]} == {"ABBA", "BAAB"}, "incomplete profiler phase")
    require(sha(manifest["program_binding"]["path"]) == manifest["program_binding"]["sha256"], "program bytes changed")
    program = read(manifest["program_binding"]["path"])
    require(sha(manifest["executable"]) == manifest["binary_sha256"] and
            sha(manifest["nsys_path"]) == manifest["nsys_binary_sha256"], "profiler/executable bytes changed")
    build_path = Path(manifest["executable"]).parent / "build-manifest.json"
    require(sha(build_path) == manifest["build_record_sha256"] and read(build_path) == manifest["build_record"], "native build record changed")
    for path, expected_sha in manifest["build_record"]["bound_files"].items():
        require(sha(path) == expected_sha, "native build-bound source/library bytes changed")
    sources = [{"path": name, "sha256": sha(campaign / name)} for name in ("manifest.json", "phase.json")]
    reports = []
    for record in phase["records"]:
        order = record["order"]
        folder = campaign / order
        require(read(folder / "record.json") == record and record["status"] == "measured" and record["returncode"] == 0,
                "profiler process record differs")
        for name, expected_sha in record["retained_files_sha256"].items():
            require(Path(name).name == name and sha(folder / name) == expected_sha, "retained profiler source changed")
        native = read(folder / "native.json")
        require(native["process_id"] == record["native_process_id"] and native["program_sha256"] == program["program_sha256"] and
                native["seed"] == manifest["program_binding"]["seed"] and native["block_order"] == order and
                native["target_action_index"] == manifest["target_action_index"] and
                native["target_action"] == program["actions"][manifest["target_action_index"]] and
                native["warmup_per_block"] == manifest["warmup"] and native["samples_per_block"] == manifest["samples"], "native/source target identity differs")
        require(record["numerical_and_interval_verification"]["independently_verified_blocks"] == 4, "missing recorded numerical checks")
        path = folder / "trace.sqlite"
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            reports.append(analyze_trace(connection, native))
        sources.extend({"path": order + "/" + name, "sha256": sha(folder / name)} for name in ("record.json", "native.json", "trace.nsys-rep", "trace.sqlite"))
    return {"schema": "single-layer-cuda-wait-diagnosis-v1", "reports": reports,
            "source_files": sources, "program_sha256": manifest["program_binding"]["sha256"],
            "nsys_version": manifest["nsys_version"].strip(), "analyzer_sha256": sha(__file__),
            "quantile_method": "inverted_empirical_CDF_no_interpolation; all_invocations_retained",
            "timeline_scope": "Nsight_aligned_CPU_and_GPU_ns_only; native_QPC_not_mapped",
            "diagnostic_only": True, "usable_for_parameter_fit": False, "usable_for_accuracy_acceptance": False,
            "limits": ["NVTX_CUPTI_profiler_changes_timing", "APIs_and_GPU_activity_overlap_do_not_add",
                       "non_sync_API_time_can_include_internal_wait_or_host_descheduling_not_pure_CPU_work",
                       "correlated_GPU_span_gaps_are_not_devicewide_idleness", "post_API_return_delay_not_pure_queue_delay",
                       "retained_numerical_checks_bound_not_recomputed_by_stdlib_analyzer",
                       "CUDA_runtime_trace_may_select_a_subset_of_APIs_as_recorded_in_capture_options",
                       "SQLite_derived_export_bound_by_hash_not_independently_reexported_here",
                       "trace_kernel_names_not_proof_of_uninstrumented_execution"]}


def compact_report(report, full_evidence):
    """Retain all block quantiles/guards while binding the local full detail."""
    full_evidence = Path(full_evidence).resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        reference = full_evidence.relative_to(repository).as_posix()
    except ValueError:
        reference = full_evidence.name
    compact = {name: value for name, value in report.items() if name != "reports"}
    compact["schema"] = "single-layer-cuda-wait-diagnosis-compact-v1"
    compact["full_decomposition_evidence"] = {"path": reference, "sha256": sha(full_evidence),
        "scope": "local_full_per_invocation_details; not_embedded_in_portable_summary"}
    compact["reports"] = []
    for process in report["reports"]:
        result = {name: value for name, value in process.items() if name != "blocks"}
        result["blocks"] = []
        for block in process["blocks"]:
            reduced = {name: value for name, value in block.items() if name != "invocations"}
            names = {name for invocation in block["invocations"] for name in invocation["api_by_name"]}
            reduced["recorded_runtime_api_aggregates"] = {}
            for name in sorted(names):
                observations = [invocation["api_by_name"][name] for invocation in block["invocations"] if name in invocation["api_by_name"]]
                reduced["recorded_runtime_api_aggregates"][name] = {
                    "invocations_with_api": len(observations),
                    "recorded_api_calls": sum(row["count"] for row in observations),
                    "per_invocation_inclusive_wall_sum_ns": stats([row["inclusive_sum_ns"] for row in observations]),
                    "worst_single_api_wall_ns": max(row["max_ns"] for row in observations),
                    "semantics": "host_API_wall_intervals; may_include_internal_blocking_and_descheduling; overlap_GPU_work"}
            reduced["kernel_multiset_counts"] = dict(Counter(json.dumps(invocation["kernel_name_counts"], sort_keys=True)
                for invocation in block["invocations"]))
            reduced["memcpy_multiset_counts"] = dict(Counter(json.dumps(invocation["memcpy_kind_and_payload_counts"], sort_keys=True)
                for invocation in block["invocations"]))
            reduced["longest_api_name_counts"] = dict(Counter(invocation["longest_api"]["name"] for invocation in block["invocations"]))
            reduced["full_invocation_count"] = len(block["invocations"])
            result["blocks"].append(reduced)
        compact["reports"].append(result)
    compact["all_block_quantiles_retained"] = True
    compact["raw_details_omitted_only_from_portable_summary"] = True
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compact-out", type=Path, help="Optional new portable block/API aggregate JSON")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("new diagnostic output required")
    if args.compact_out is not None and args.compact_out.exists():
        raise FileExistsError("new compact diagnostic output required")
    if args.compact_out is not None:
        require(args.out.resolve() != args.compact_out.resolve(), "full and compact output files must differ")
    report = analyze_campaign(args.campaign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.compact_out is not None:
        compact = compact_report(report, args.out)
        args.compact_out.parent.mkdir(parents=True, exist_ok=True)
        with args.compact_out.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(compact, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(args.out), "reports": len(report["reports"]), "diagnostic_only": True}))


if __name__ == "__main__":
    main()
