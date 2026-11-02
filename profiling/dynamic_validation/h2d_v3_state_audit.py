"""Read-only stage/state audit for a completed, bound H2D v3 diagnostic.

This does not fit a response law or make observed telemetry a forecast input.
The observed prefix / device activity / completion tail is a wall-clock
partition, whereas the runtime API and synchronization intervals overlap it.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import pathlib
import statistics


def receipt(path: pathlib.Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def distribution(values):
    return dict(collections.Counter(str(x) for x in values))


def preceding(samples, ends, tick):
    i = bisect.bisect_right(ends, tick) - 1
    return samples[i] if i >= 0 else None


def activity_partition(row: dict, frequency_hz: int) -> dict:
    physical = row["physical"]
    if not physical["supported"]:
        raise ValueError("A supported CUPTI association is required")
    copies = [x for x in physical["raw_correlated_records"]
              if x["kind"] == "memcpy" and x.get("timestamp_valid")]
    if len(copies) != 1:
        raise ValueError("Expected exactly one correlated 2 MiB memcpy")
    begin = row["begin_tick"] * 1_000_000_000 / frequency_hz
    end = row["end_tick"] * 1_000_000_000 / frequency_hz
    device_begin, device_end = copies[0]["begin_ns"], copies[0]["end_ns"]
    result = {
        "host_entry_to_device_begin_ns": device_begin - begin,
        "device_activity_span_ns": device_end - device_begin,
        "device_end_to_host_completion_ns": end - device_end,
    }
    if any(v < 0 for v in result.values()):
        raise ValueError("Device activity is not enclosed by the measured host call")
    if abs(sum(result.values()) - row["host_wall_ns"]) > 1:
        raise ValueError("Observed wall-clock partition does not close")
    return result


def audit(out: pathlib.Path) -> dict:
    path = out / "stage-analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "schema": "dynamic-h2d-v3-state-stage-audit-1",
        "stage_analysis": receipt(path),
        "role": "diagnosis_only_no_parameter_fitting_or_forecast_validation",
        "early_trials": list(range(8)), "late_trials": list(range(8, 16)),
        "copy_ordinals": list(range(1, 16)),
        "telemetry_alignment": "Last GPU provider query_end_tick <= individual copy begin_tick; no interpolation or future observations",
        "telemetry_age_is_preserved": True,
        "telemetry_is_target_inclusive": True,
        "telemetry_establishes_available_FLOPS_or_causality": False,
        "pcie_link_gen": {"status": "unknown", "reason": "Not collected by this frozen sampler"},
        "pcie_link_width": {"status": "unknown", "reason": "Not collected by this frozen sampler"},
        "processes": [],
        "observer_pairs": analysis["observer_pairs"],
    }
    for process in analysis["processes"]:
        job = out / "jobs" / process["job_id"]
        native_path = job / "result.json"
        telemetry_path = job / "telemetry.jsonl"
        native = json.loads(native_path.read_text(encoding="utf-8"))
        telemetry = [json.loads(x) for x in telemetry_path.read_text(encoding="utf-8").splitlines()]
        gpu = sorted((x["gpu"] for x in telemetry if x.get("gpu", {}).get("status") == "ok"),
                     key=lambda x: x["provider_query_end_tick"])
        ends = [x["provider_query_end_tick"] for x in gpu]
        frequency = native["clock"]["frequency_hz"]
        proc = {k: process[k] for k in ("job_id", "memory_mode", "init_policy", "instrumentation", "round")}
        proc.update(native=receipt(native_path), telemetry=receipt(telemetry_path), groups=[],
                    thread_inventories=process["thread_inventories"])
        for idle in (0, 20):
            for low, high in ((0, 8), (8, 16)):
                sequences = [x for x in native["sequences"]
                             if x["requested_idle_ms"] == idle and low <= x["trial"] < high]
                copies = [(s, c) for s in sequences for c in s["copies"] if c["ordinal"] > 0]
                observations = []
                for sequence, copy in copies:
                    obs = preceding(gpu, ends, copy["begin_tick"])
                    observations.append({
                        "trial": sequence["trial"], "ordinal": copy["ordinal"],
                        "copy_begin_tick": copy["begin_tick"],
                        "query_end_tick": obs["provider_query_end_tick"] if obs else None,
                        "observation_age_ns": ((copy["begin_tick"] - obs["provider_query_end_tick"])
                                               * 1_000_000_000 / frequency) if obs else None,
                        "values": obs["values"] if obs else {},
                    })
                group = {
                    "requested_idle_ms": idle, "trial_begin_inclusive": low,
                    "trial_end_exclusive": high, "trials": len(sequences),
                    "later_copy_samples": len(copies),
                    "later_copy_host_wall_mean_ns": mean(c["wall_ns"] for _, c in copies),
                    "prior_host_available_bytes_mean": mean(x["before_burst_state"]["host_available_bytes"] for x in sequences),
                    "full_prime_mean_ns": mean(x["prime_ns"] for x in sequences),
                    "state_query_between_idle_end_and_burst_begin_mean_ns": mean(
                        (x["burst_begin_tick"] - x["idle_end_tick"]) * 1_000_000_000 / frequency for x in sequences),
                    "observed_prior_gpu_values": {k: distribution(x["values"].get(k, "unknown") for x in observations)
                                                  for k in ("memory_clock_mhz", "graphics_clock_mhz", "pstate", "temperature_c")},
                    "per_copy_prior_gpu_observations": observations,
                }
                rows = [x for x in analysis["stage_rows"]
                        if x["job_id"] == process["job_id"] and x["requested_idle_ms"] == idle
                        and x["ordinal"] > 0 and low <= x["trial"] < high]
                if rows:
                    partitions = [activity_partition(x, frequency) for x in rows]
                    group["stage_means_ns"] = {
                        **{k: mean(x["physical"][k] for x in rows) for k in (
                            "copy_API_inclusive_ns", "runtime_memcpy_API_union_ns",
                            "inner_synchronization_runtime_union_ns", "device_memcpy_span_ns")},
                        "extra_GPU_sync_ns": mean(x["host_stages"]["extra_GPU_sync_ns"] for x in rows),
                        "CPU_sync_ns": mean(x["host_stages"]["CPU_sync_ns"] for x in rows),
                        **{k: mean(x[k] for x in partitions) for k in partitions[0]},
                    }
                    group["API_wait_and_device_are_not_additive"] = True
                proc["groups"].append(group)
        result["processes"].append(proc)
    result["interpretation_constraints"] = [
        "Full observer and bare are different process contexts; pair discrepancy is not pure observer overhead.",
        "No observed noncalling helper has CPU2-only affinity at the snapshots; transient affinity and scheduling remain unknown.",
        "Both registered init policies exhibit a later-copy slowdown in this new protocol; wide initialization is not an observed repair.",
        "The observed wall partition separates latency to device start, device activity and completion tail; its prefix is not pure host execution time.",
        "The diagnostic protocol has additional snapshots and telemetry versus v2 and is not an equivalent v2 calibration.",
        "Clock/Pstate associations are descriptive, not a causal explanation or a throughput scaling law.",
    ]
    result["performance_parameters_fitted"] = False
    result["generator"] = receipt(pathlib.Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = audit(args.out)
    target = args.output or args.out / "state-stage-diagnosis.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"output": receipt(target), "processes": len(result["processes"]),
                      "performance_parameters_fitted": False}))


if __name__ == "__main__":
    main()
