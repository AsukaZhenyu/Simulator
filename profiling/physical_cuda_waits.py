"""Describe CUDA launch/return gaps without fitting or guessing causality.

Input is the exact-attribution output of ``physical_records``. Only matching
runtime launch/kernel correlation IDs within the same attributed action form
pairs. A synchronization API's return is a descriptive endpoint; the available
records do not identify its event/stream dependency or separate OS scheduling,
driver/runtime work, GPU dependency waits and device execution inside its span.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median


SCHEMA = "physical-cuda-wait-diagnostic-1"
SYNC_PREFIXES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


def _valid_time(row):
    begin, end = row.get("begin_ns"), row.get("end_ns")
    return (type(begin) is int and type(end) is int and 0 <= begin < end and
            row.get("timestamp_valid") is True)


def _correlation(row):
    value = row.get("correlation_id")
    return value if type(value) is int and value > 0 else None


def _union(rows):
    merged = []
    for begin, end in sorted((row["begin_ns"], row["end_ns"]) for row in rows if _valid_time(row)):
        if not merged or begin > merged[-1]["end_ns"]:
            merged.append(dict(begin_ns=begin, end_ns=end))
        else:
            merged[-1]["end_ns"] = max(merged[-1]["end_ns"], end)
    return merged


def _length(intervals):
    return sum(row["end_ns"]-row["begin_ns"] for row in intervals)


def _overlap(left, right):
    total, i, j = 0, 0, 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i]["end_ns"], right[j]["end_ns"])-max(left[i]["begin_ns"], right[j]["begin_ns"]))
        if left[i]["end_ns"] < right[j]["end_ns"]:
            i += 1
        else:
            j += 1
    return total


def _name(row):
    value = row.get("name")
    return value if isinstance(value, str) else ""


def _action(action, *, capture_qualified):
    kernels = action.get("kernel_records", [])
    copies = action.get("memcpy_records", [])
    runtime = action.get("runtime_api_records", [])
    driver = action.get("driver_api_records", [])
    launches = [row for row in runtime if _name(row).startswith("cudaLaunchKernel")]
    syncs = [row for row in runtime if _name(row).startswith(SYNC_PREFIXES)]
    other = [row for row in runtime if row not in launches and row not in syncs]
    by_correlation = defaultdict(list)
    for ordinal, launch in enumerate(launches):
        correlation = _correlation(launch)
        if correlation is not None:
            by_correlation[correlation].append((ordinal, launch))
    pairs, unpaired_kernels, paired_launches = [], [], set()
    for kernel in kernels:
        correlation = _correlation(kernel)
        matches = by_correlation.get(correlation, []) if correlation is not None else []
        reason = ("invalid_kernel_correlation_id" if correlation is None else
                  "no_exact_same_action_runtime_cudaLaunchKernel_correlation" if not matches else
                  "ambiguous_same_action_runtime_launch_correlation" if len(matches) != 1 else
                  "invalid_kernel_or_launch_timestamp" if not _valid_time(kernel) or not _valid_time(matches[0][1]) else None)
        if reason is not None:
            unpaired_kernels.append(dict(record=kernel, reason=reason, candidate_launch_records=[row for _, row in matches]))
            continue
        ordinal, launch = matches[0]
        paired_launches.add(ordinal)
        gap = kernel["begin_ns"]-launch["end_ns"]
        kernel_before_launch = kernel["begin_ns"] < launch["begin_ns"]
        pairs.append(dict(correlation_id=correlation, launch_api_record=launch, kernel_record=kernel,
            launch_api_span_ns=launch["end_ns"]-launch["begin_ns"],
            kernel_activity_envelope_ns=kernel["end_ns"]-kernel["begin_ns"],
            signed_kernel_begin_minus_launch_end_ns=gap,
            queue_after_launch_lower_bound_ns=max(0, gap),
            kernel_begins_before_launch_api_return=gap < 0,
            kernel_begins_before_launch_api_entry=kernel_before_launch,
            launch_api_and_kernel_overlap_ns=max(0, min(launch["end_ns"], kernel["end_ns"])-max(launch["begin_ns"], kernel["begin_ns"])),
            launch_return_value=launch.get("return_value"),
            exact_same_action_correlation_verified=True,
            causal_order_consistent=not kernel_before_launch,
            pair_metrology_qualified=capture_qualified and action.get("selected_action_attribution_qualified") is True and
                not kernel_before_launch and launch.get("return_value") == 0,
            queue_delay_cause_identified=False))
    unpaired_launches = []
    for ordinal, launch in enumerate(launches):
        if ordinal in paired_launches:
            continue
        correlation = _correlation(launch)
        reason = ("invalid_launch_correlation_id" if correlation is None else
                  "ambiguous_same_action_runtime_launch_correlation" if len(by_correlation[correlation]) != 1 else
                  "invalid_launch_timestamp" if not _valid_time(launch) else
                  "no_pairable_exact_same_action_kernel")
        unpaired_launches.append(dict(record=launch, reason=reason))

    kernel_union, copy_union = _union(kernels), _union(copies)
    device_union = _union([dict(row, timestamp_valid=True) for row in kernel_union+copy_union])
    runtime_union, driver_union = _union(runtime), _union(driver)
    launch_union, sync_union, other_union = _union(launches), _union(syncs), _union(other)
    overlap = _overlap(sync_union, device_union)
    valid_syncs = [row for row in syncs if _valid_time(row)]
    final = []
    if valid_syncs:
        final_return = max(row["end_ns"] for row in valid_syncs)
        final = [row for row in valid_syncs if row["end_ns"] == final_return]
    last_kernel = max((row["end_ns"] for row in kernel_union), default=None)
    last_device = max((row["end_ns"] for row in device_union), default=None)
    completion = dict(final_sync_api_records=final,
        selection="last return among synchronization APIs already attributed to this action by exact external correlation",
        final_sync_api_unique=len(final) == 1,
        last_kernel_end_ns=last_kernel, last_device_activity_end_ns=last_device,
        last_kernel_end_to_final_sync_return_signed_ns=None,
        last_device_end_to_final_sync_return_signed_ns=None,
        post_last_kernel_tail_ns=None, post_last_device_tail_ns=None,
        final_sync_returns_before_last_kernel=None,
        kernel_to_sync_dependency_proven=False, pure_scheduler_latency_identified=False,
        qualification="same-action endpoint comparison, not a kernel-to-stream/event dependency match; tail includes any host/runtime/driver work or waiting after the last attributed device endpoint")
    if len(final) == 1:
        end = final[0]["end_ns"]
        if last_kernel is not None:
            signed = end-last_kernel
            completion.update(last_kernel_end_to_final_sync_return_signed_ns=signed,
                post_last_kernel_tail_ns=signed if signed >= 0 else None,
                final_sync_returns_before_last_kernel=signed < 0)
        if last_device is not None:
            signed = end-last_device
            completion.update(last_device_end_to_final_sync_return_signed_ns=signed,
                post_last_device_tail_ns=signed if signed >= 0 else None)
    return dict(iteration=action.get("iteration"), action_index=action.get("action_index"),
        external_id=action.get("external_id"), kind=action.get("kind"), target_id=action.get("target_id"), device=action.get("device"),
        begin_ns=action.get("begin_ns"), end_ns=action.get("end_ns"),
        selected_action_attribution_qualified=action.get("selected_action_attribution_qualified") is True,
        launch_kernel_pairs=pairs, unpaired_kernel_records=unpaired_kernels, unpaired_launch_api_records=unpaired_launches,
        synchronization_api_records=syncs, other_runtime_api_records=other, driver_api_records=driver,
        runtime_api_records_not_pairable_as_launch=[dict(record=row, reason="synchronization_or_other_API_not_a_runtime_cudaLaunchKernel") for row in syncs+other],
        invalid_timestamp_records=[row for row in kernels+copies+runtime+driver if not _valid_time(row)],
        launch_api_union=launch_union, launch_api_union_ns=_length(launch_union),
        synchronization_api_union=sync_union, synchronization_api_union_ns=_length(sync_union),
        other_runtime_api_union=other_union, other_runtime_api_union_ns=_length(other_union),
        runtime_api_union=runtime_union, runtime_api_union_ns=_length(runtime_union),
        driver_api_union=driver_union, driver_api_union_ns=_length(driver_union),
        kernel_activity_union=kernel_union, kernel_activity_union_ns=_length(kernel_union),
        memcpy_activity_union=copy_union, memcpy_activity_union_ns=_length(copy_union),
        device_activity_union=device_union, device_activity_union_ns=_length(device_union),
        sync_API_time_overlapping_attributed_device_activity_ns=overlap,
        sync_API_time_without_attributed_device_activity_ns=_length(sync_union)-overlap,
        synchronization_completion=completion,
        timing_columns_additive=False, device_activity_is_SM_active_service=False)


def diagnose_cuda_waits(parsed):
    """Return descriptive intervals; never assign records using timestamps.

    ``queue_after_launch_lower_bound_ns`` is the nonnegative elapsed interval
    after launch API return and before the correlated kernel begins. Its cause
    is not identified: it can contain submission/dependency/scheduling delays.
    Negative signed gaps are retained when the kernel begins inside the API.
    """
    if not isinstance(parsed, dict) or parsed.get("schema") != "physical-stage-attribution-1":
        raise ValueError("exact physical-stage-attribution-1 input required")
    if parsed.get("timestamp_attribution_fallback_used") is not False:
        raise ValueError("input must explicitly disable timestamp attribution fallback")
    if not isinstance(parsed.get("actions"), list):
        raise ValueError("parsed action inventory required")
    actions = [_action(row, capture_qualified=parsed.get("capture_qualified") is True) for row in parsed["actions"]]
    pairs = [pair for row in actions for pair in row["launch_kernel_pairs"]]
    tails = [row["synchronization_completion"]["post_last_kernel_tail_ns"] for row in actions
             if row["synchronization_completion"]["post_last_kernel_tail_ns"] is not None]
    orphans = parsed.get("orphan_records", [])
    return dict(schema=SCHEMA, source_artifact_sha256=parsed.get("artifact_sha256"),
        source_native_sha256=parsed.get("native_semantic_sha256"), process_id=parsed.get("process_id"), clock=parsed.get("clock"),
        capture_qualified=parsed.get("capture_qualified") is True,
        qualification_scope="descriptive launch/return intervals from exact action and runtime-launch correlation only",
        actions=actions,
        counts=dict(actions=len(actions), exact_launch_kernel_pairs=len(pairs),
            kernel_begins_before_launch_return=sum(pair["kernel_begins_before_launch_api_return"] for pair in pairs),
            kernel_begins_before_launch_entry=sum(pair["kernel_begins_before_launch_api_entry"] for pair in pairs),
            unpaired_kernel_records=sum(len(row["unpaired_kernel_records"]) for row in actions),
            unpaired_launch_api_records=sum(len(row["unpaired_launch_api_records"]) for row in actions),
            orphan_records=len(orphans), invalid_time_records=len(parsed.get("invalid_time_records", []))),
        descriptive_statistics=dict(
            paired_kernel_queue_after_launch_lower_bound_median_ns=median([pair["queue_after_launch_lower_bound_ns"] for pair in pairs]) if pairs else None,
            action_post_last_kernel_tail_median_ns=median(tails) if tails else None,
            qualification="medians of retained descriptive records; no fit, no filtering by prediction error, no sum of overlapping intervals"),
        orphan_records=orphans, invalid_time_records=parsed.get("invalid_time_records", []),
        capture_issues=parsed.get("capture_issues", []),
        timestamp_attribution_fallback_used=False, synchronization_kernel_dependency_proven=False,
        runtime_driver_device_columns_additive=False, pure_scheduler_latency_identified=False,
        available_resource_rate_inferred=False, hardware_accuracy_qualified=False,
        limitations=["Launch API wall time and synchronization API wall time can overlap device work and nested driver APIs.",
            "Queue-after-launch is a lower-bound endpoint gap, not an identification of hardware queue, OS scheduling or GPU dependency causes.",
            "Synchronization may address a different stream/event; current records lack API arguments to establish that dependency.",
            "Time without attributed device activity can include host work, waits, other contexts, unobserved work or preemption; it is not pure OS/runtime overhead.",
            "Recorded kernel envelopes do not prove active SM service or absence of stalls/preemption."])
