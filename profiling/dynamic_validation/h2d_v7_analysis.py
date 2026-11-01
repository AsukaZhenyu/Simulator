"""Strict phase-record checks for v7; no coefficient estimation.

Host response segments form one QPC partition. The optional CUDA-event duration
has another clock and overlaps host synchronization; it is never added to that
partition, called pure DMA, or shifted to an invented QPC position.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from . import h2d_v7 as campaign

require = campaign.require


def host_partition(copy, hz):
    begin, end = copy["begin_tick"], copy["end_tick"]
    require(type(begin) is int and type(end) is int and begin < end, "invalid copy QPC interval")
    require(abs((end-begin)*1e9/hz-copy["wall_ns"]) <= 1, "copy duration differs from QPC")
    if copy["inner_response_parts_available"] is False:
        require("host_response_segments" not in copy and "device_event_span_ns" not in copy, "unmeasured GGML internals fabricated")
        return dict(host_wall_ns=copy["wall_ns"], parts_supported=False)
    require(copy["inner_response_parts_available"] is True, "explicit segment scope missing")
    rows, previous, covered = copy["host_response_segments"], begin, 0
    require(isinstance(rows, list) and len(rows) in (4, 6), "exact primitive stage count required")
    names = [s["name"] for s in rows]
    required = ["cudaMemcpyAsync_API_response", "inner_PTDS_sync_API_response", "extra_GGML_GPU_sync_response", "extra_GGML_CPU_sync_response"]
    require(names == required or names == ["event_start_record_API", required[0], "event_stop_record_API", *required[1:]], "unexpected primitive stage sequence")
    values = {}
    for row in rows:
        a, b = row["begin_tick"], row["end_tick"]
        require(type(a) is int and type(b) is int and previous <= a <= b <= end, "overlapping/reversed primitive API segments")
        require(abs((b-a)*1e9/hz-row["wall_ns"]) <= 1, "segment duration differs from QPC")
        values[row["name"]] = (b-a)*1e9/hz;covered += b-a;previous = b
    outside = end-begin-covered
    require(type(copy["outside_segments_ticks"]) is int and copy["outside_segments_ticks"] == outside and outside >= 0, "host interval partition does not close")
    # Work in ticks for closure; summing separately floored nanoseconds can differ.
    return dict(host_wall_ns=(end-begin)*1e9/hz, parts_supported=True,
                response_parts_ns=values, outside_segments_ns=outside*1e9/hz,
                host_partition_closes=True, device_event_span_ns=copy.get("device_event_span_ns"),
                device_event_overlaps_host_wait=True, GPU_event_QPC_position_known=False,
                host_API_is_CPU_active=False)


def validate_native(native):
    require(native["schema"] == "dynamic-h2d-primitive-native-7", "native schema differs")
    r = native["request"];require(r["schema"] == "dynamic-h2d-primitive-request-7", "request schema differs")
    require(native["source_sha256"] == r["source_sha256"] and native["original_CUDA_source_sha256"] == r["original_CUDA_source_sha256"]
            and native["original_backend_source_sha256"] == r["original_backend_source_sha256"], "source identity differs")
    require(r["sizes_bytes"] == list(campaign.SIZES[r["split"]]) and r["idle_ms"] == list(campaign.IDLES[r["split"]]), "split separation differs")
    require(r["modes"] == list(campaign.MODES) and r["prior_query_placement"] == "before_full_prime" and r["post_queries_are_predictor_input"] is False, "observation scope differs")
    registration = native["arena_registration"]
    field = "registered_host" if r["memory_mode"] == "registered" else "unregistered_pageable"
    require(registration["mode"] == r["memory_mode"] and registration["during"][field] is True
            and native["source_pointer_kind"][field] is True and registration["restored"] is True, "actual memory category/restoration differs")
    require(native["source_buffer_host"] is True and native["target_buffer_host"] is False and native["CUDA_device"] == 0, "actual host/device buffer contract missing")
    require(native["primitive_stream"]["kind"] == "cudaStreamPerThread" and native["primitive_stream"]["opaque_runtime_handle"] > 0
            and native["primitive_stream"]["GGML_bare_internal_stream_observed_dynamically"] is False, "explicit primitive stream evidence missing")
    for field in ("CUPTI_initialized", "foreground_queries_between_prime_and_first_copy", "primitive_is_GGML_action", "GPU_event_is_pure_DMA",
                  "GPU_event_QPC_aligned", "GPU_event_and_host_wait_additive", "host_API_response_is_CPU_active", "post_target_state_used_for_prediction", "performance_parameters_fitted"):
        require(native[field] is False, "diagnostic scope broadened: " + field)
    require(native["native_repeat_consistency"]["passed"] and native["verification_native"]["copy_values_equal"] and native["full_payload"]["full_byte_equality"]
            and native["timed_payload_checked_before_final_reset"] is True, "native numerical/copy oracle failed or timed result overwritten before check")
    require(native["main_thread_affinity"]["observed_mask"] == native["main_thread_affinity"]["final_mask"] == 4, "target affinity changed")
    expected = 12*len(r["sizes_bytes"])*6
    require(len(native["bursts"]) == expected, "declared burst inventory incomplete")
    oracles = [o for size in native["slice_copy_oracles"] for o in size]
    require(len(oracles) == 3*len(r["sizes_bytes"]) and {(o["mode"], o["bytes"]) for o in oracles}
            == {(m, b) for m in campaign.MODES for b in r["sizes_bytes"]}, "range qualification inventory differs")
    require(all(o["selected_range_initially_different"] and o["selected_bytes_equal_after_copy"] and o["outside_view_unchanged"] for o in oracles), "copy oracle vacuous or failed")
    hz = native["clock"]["frequency_hz"];require(type(hz) is int and hz > 0, "QPC frequency missing")
    count, previous = 0, 0
    for index, burst in enumerate(native["bursts"]):
        trial, offset = divmod(index, len(r["sizes_bytes"])*6);slot, cs = divmod(offset, 6)
        si, ci = r["size_orders"][trial][slot], r["condition_orders"][trial][cs]
        require((burst["trial"], burst["size_slot"], burst["condition_slot"], burst["size_index"], burst["condition_index"], burst["bytes"], burst["mode"], burst["requested_idle_ms"])
                == (trial, slot, cs, si, ci, r["sizes_bytes"][si], campaign.MODES[ci//2], r["idle_ms"][ci%2]), "balanced condition order differs")
        require(previous <= burst["prime_begin_tick"] <= burst["prime_end_tick"] <= burst["idle_begin_tick"] <= burst["idle_end_tick"] <= burst["burst_begin_tick"] < burst["burst_end_tick"]
                <= burst["event_read_begin_tick"] <= burst["event_read_end_tick"], "conditioning/completion clock differs")
        previous = burst["event_read_end_tick"]
        require(abs((burst["burst_end_tick"]-burst["burst_begin_tick"])*1e9/hz-burst["burst_wall_ns"])<=1, "burst duration differs from QPC")
        require(abs((burst["idle_end_tick"]-burst["idle_begin_tick"])*1e9/hz-burst["actual_idle_ns"])<=1, "actual idle differs from QPC")
        require(len(burst["copies"]) == 16, "all sixteen copy boundaries required")
        require(abs((burst["copies"][0]["begin_tick"]-burst["prime_end_tick"])*1e9/hz-burst["effective_prime_to_first_copy_ns"])<=1, "effective history gap differs from QPC")
        last = burst["burst_begin_tick"]
        for ordinal, copy in enumerate(burst["copies"]):
            require(copy["ordinal"] == ordinal and last <= copy["begin_tick"] < copy["end_tick"] <= burst["burst_end_tick"], "copy ordering differs")
            require(copy["inner_response_parts_available"] == (burst["mode"] != "ggml_bare"), "GGML versus primitive boundary changed")
            host_partition(copy, hz);last = copy["end_tick"];count += 1
            span = copy.get("device_event_span_ns")
            if burst["mode"] == "primitive_events":
                require(isinstance(span, (int, float)) and not isinstance(span, bool) and math.isfinite(span) and span >= 0, "event elapsed span missing")
            else:
                require(span is None, "event span fabricated for unobserved control")
    return dict(passed=True, copies=count, bursts=expected, host_partitions_closed=True,
                device_event_is_pure_DMA=False, response_parameters_qualified=False)


def describe(values):
    values = list(values);average = statistics.mean(values) if values else None
    return dict(count=len(values), mean=average, minimum=min(values, default=None), maximum=max(values, default=None),
                population_cv=statistics.pstdev(values)/average if len(values)>1 and average not in (None, 0) else None)


def process_summary(native):
    validation = validate_native(native);r = native["request"];hz = native["clock"]["frequency_hz"]
    groups = defaultdict(list)
    for burst in native["bursts"]:
        copies = burst["copies"]
        groups[(burst["bytes"], burst["requested_idle_ms"], burst["mode"])].append(dict(
            trial=burst["trial"], first=copies[0]["wall_ns"], suffix=statistics.mean(c["wall_ns"] for c in copies[1:]),
            burst=burst["burst_wall_ns"]/16, actual_idle_ns=burst["actual_idle_ns"],
            prime_to_first_ns=burst["effective_prime_to_first_copy_ns"], copies=[host_partition(c, hz) for c in copies]))
    summaries, differences = [], []
    for (size, idle, mode), rows in sorted(groups.items()):
        summaries.append(dict(bytes=size, requested_idle_ms=idle, mode=mode, trials=len(rows),
                              first_mean_ns=statistics.mean(x["first"] for x in rows), suffix15_mean_ns=statistics.mean(x["suffix"] for x in rows),
                              burst_per_copy_mean_ns=statistics.mean(x["burst"] for x in rows),
                              early_suffix15_mean_ns=statistics.mean(x["suffix"] for x in rows if x["trial"]<6),
                              late_suffix15_mean_ns=statistics.mean(x["suffix"] for x in rows if x["trial"]>=6),
                              first_distribution=describe(x["first"] for x in rows), suffix15_distribution=describe(x["suffix"] for x in rows),
                              actual_idle_ns=describe(x["actual_idle_ns"] for x in rows), prime_to_first_ns=describe(x["prime_to_first_ns"] for x in rows)))
    lookup = {(x["bytes"], x["requested_idle_ms"], x["mode"]):x for x in summaries}
    for size in r["sizes_bytes"]:
        for idle in r["idle_ms"]:
            for reference, observed in (("ggml_bare", "primitive_stamped"), ("primitive_stamped", "primitive_events")):
                left, right = lookup[size,idle,reference], lookup[size,idle,observed]
                delta = {k: abs(right[k]/left[k]-1) for k in ("first_mean_ns", "suffix15_mean_ns", "burst_per_copy_mean_ns")}
                differences.append(dict(bytes=size, requested_idle_ms=idle, reference=reference, observed=observed,
                                        relative_differences=delta, passed=all(v<=.05 for v in delta.values()), pure_observer_overhead_identified=False))
    baseline = []
    if r["split"] == "profile":
        for x in summaries:
            if x["requested_idle_ms"] == 0:
                delta = abs(x["late_suffix15_mean_ns"]/x["early_suffix15_mean_ns"]-1)
                baseline.append(dict(bytes=x["bytes"], mode=x["mode"], early_late_relative_difference=delta, passed=delta<=.10))
    return dict(validation=validation, memory_mode=r["memory_mode"], split=r["split"], groups=summaries,
                observer_comparisons=differences, all_observer_gates_passed=all(x["passed"] for x in differences),
                baseline_early_late=baseline, all_profile_baseline_gates_passed=all(x["passed"] for x in baseline) if baseline else None,
                actual_link_during_copy="unknown; pre-prime and post-burst snapshots cannot prove constancy",
                performance_parameters_fitted=False, mean_residual_compensation_used=False)


def main():
    p = argparse.ArgumentParser();p.add_argument("--native", required=True);p.add_argument("--out", required=True);a=p.parse_args()
    value = process_summary(campaign.read(a.native));value["native"] = campaign.receipt(Path(a.native))
    campaign.write(Path(a.out), value)
    print(json.dumps(dict(output=campaign.receipt(Path(a.out)), observer_passed=value["all_observer_gates_passed"], performance_parameters_fitted=False)))


if __name__ == "__main__": main()
