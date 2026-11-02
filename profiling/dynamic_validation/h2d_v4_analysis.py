"""No-fit analysis of the registered H2D low-intrusion crossover."""
from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path
import statistics

from profiling.dynamic_validation import h2d_v4 as campaign


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summarize_rows(rows):
    rows = list(rows)
    if not rows:
        return None
    return dict(count=len(rows), first_mean_ns=mean(x["first_ns"] for x in rows),
                later_mean_ns=mean(x["later_ns"] for x in rows),
                burst_per_copy_mean_ns=mean(x["burst_per_copy_ns"] for x in rows),
                early_later_mean_ns=mean(x["later_ns"] for x in rows if x["trial"] < 8),
                late_later_mean_ns=mean(x["later_ns"] for x in rows if x["trial"] >= 8),
                actual_idle_mean_ns=mean(x["actual_idle_ns"] for x in rows),
                prime_to_first_copy_mean_ns=mean(x["prime_to_first_copy_ns"] for x in rows))


def analyze(out):
    out = Path(out).resolve()
    manifest = campaign.read(out / "manifest.json")
    campaign.base.check(manifest)
    executions = campaign.read(out / "executions.json")
    result = dict(schema="dynamic-h2d-memcpy-analysis-4", manifest=campaign.receipt(out / "manifest.json"),
                  executions=campaign.receipt(out / "executions.json"), processes=[],
                  interpretation="diagnostic only; both order/time and persistent CUPTI lifecycle are retained; no prices or state response law fitted",
                  provider_alignment="last completed provider query before individual target copy; query age retained; no interpolation",
                  performance_parameters_fitted=False, observer_differences_subtracted=False, future_telemetry_used_for_forecast=False)
    state_fields = ("pcie_current_gen", "pcie_max_gen", "pcie_current_width", "pcie_max_width",
                    "memory_clock_mhz", "graphics_clock_mhz", "pstate", "temperature_c")
    for execution in executions:
        if execution["status"] != "completed":
            result["processes"].append(dict(job_id=execution["job"]["id"], status=execution["status"], error=execution.get("error")))
            continue
        native_path = Path(execution["native"]["path"])
        campaign.require(campaign.sha(native_path) == execution["native"]["sha256"], "bound native changed")
        native = campaign.read(native_path)
        telemetry_path = Path(execution["telemetry"]["path"])
        campaign.require(campaign.sha(telemetry_path) == execution["telemetry"]["sha256"], "telemetry changed")
        telemetry = [json.loads(x) for x in telemetry_path.read_text(encoding="utf-8").splitlines()]
        gpu = sorted((x["gpu"] for x in telemetry if x.get("gpu", {}).get("values")), key=lambda x: x["provider_query_end_tick"])
        ends = [x["provider_query_end_tick"] for x in gpu]
        hz = native["clock"]["frequency_hz"]
        proc = dict(job_id=execution["job"]["id"], status="completed", native=execution["native"], telemetry=execution["telemetry"],
                    source_pages_before=native["source_pages_before"], source_pages_after=native["source_pages_after"],
                    arena_registration=native["arena_registration"], blocks=[], observer_discrepancies=[])
        for block in native["blocks"]:
            mode = block["instrumentation"]
            focused = [copy for sequence in block["sequences"] for copy in sequence["copies"] if copy["external_id"] is not None]
            associations = campaign.direct_associations(block["cupti"], focused) if mode == "memcpy_only" else []
            by_id = {x["external_id"]: x for x in associations}
            rows, copies = [], []
            for sequence in block["sequences"]:
                if sequence["requested_idle_ms"] not in (0, 20):
                    continue
                samples = sequence["copies"]
                rows.append(dict(trial=sequence["trial"], requested_idle_ms=sequence["requested_idle_ms"],
                                 first_ns=samples[0]["wall_ns"], later_ns=mean(x["wall_ns"] for x in samples[1:]),
                                 burst_per_copy_ns=sequence["burst_wall_ns"] / 16, actual_idle_ns=sequence["actual_idle_ns"],
                                 prime_to_first_copy_ns=sequence["effective_prime_end_to_first_copy_ns"],
                                 prior_host_available_bytes=sequence["before_burst_state"]["host_available_bytes"]))
                for copy in samples:
                    before = bisect.bisect_right(ends, copy["begin_tick"]) - 1
                    observation = gpu[before] if before >= 0 else None
                    row = dict(trial=sequence["trial"], requested_idle_ms=sequence["requested_idle_ms"], ordinal=copy["ordinal"],
                               begin_tick=copy["begin_tick"], end_tick=copy["end_tick"], host_wall_ns=copy["wall_ns"], external_id=copy["external_id"],
                               prior_gpu_values=observation["values"] if observation else {},
                               prior_gpu_unknown=observation["unknown"] if observation else {"all": "no completed prior query"},
                               prior_gpu_query_end_tick=observation["provider_query_end_tick"] if observation else None,
                               prior_gpu_age_ns=(copy["begin_tick"] - observation["provider_query_end_tick"]) * 1e9 / hz if observation else None)
                    if copy["external_id"] is not None:
                        associated = by_id[copy["external_id"]]
                        row["physical_support"] = associated
                        if associated["supported"]:
                            event = associated["device_records"][0]
                            start = copy["begin_tick"] * 1_000_000_000 // hz
                            finish = copy["end_tick"] * 1_000_000_000 // hz
                            if start <= event["begin_ns"] < event["end_ns"] <= finish:
                                row["wall_partition_ns"] = dict(host_entry_to_device_begin=event["begin_ns"] - start,
                                                                 device_activity=event["end_ns"] - event["begin_ns"],
                                                                 device_end_to_host_completion=finish - event["end_ns"])
                                campaign.require(abs(sum(row["wall_partition_ns"].values()) - copy["wall_ns"]) <= 1, "observed wall partition does not close")
                            else:
                                row["physical_support"] = dict(associated, supported=False, reasons=["device interval is not enclosed by target host interval"])
                    copies.append(row)
            groups = []
            for idle in (0, 20):
                for low, high in ((0, 8), (8, 16)):
                    selected = [x for x in copies if x["requested_idle_ms"] == idle and x["ordinal"] > 0 and low <= x["trial"] < high]
                    supported = [x for x in selected if x.get("physical_support", {}).get("supported")]
                    complete = bool(selected) and len(supported) == len(selected)
                    groups.append(dict(requested_idle_ms=idle, trials=[low, high], copies=len(selected),
                                       host_wall_mean_ns=mean(x["host_wall_ns"] for x in selected),
                                       required_device_copies=len(selected) if mode == "memcpy_only" else 0,
                                       identified_device_copies=len(supported),
                                       wall_partition_means_ns={k: mean(x["wall_partition_ns"][k] for x in selected) for k in selected[0]["wall_partition_ns"]} if complete else None,
                                       prior_gpu_distributions={k: dict(collections.Counter(str(x["prior_gpu_values"].get(k, "unknown")) for x in selected)) for k in state_fields}))
            proc["blocks"].append(dict(block=block["block"], instrumentation=mode, trials=rows, copies=copies, groups=groups,
                                       focus={str(idle): summarize_rows(x for x in rows if x["requested_idle_ms"] == idle) for idle in (0, 20)},
                                       required_correlated_copies=len(associations), supported_correlated_copies=sum(x["supported"] for x in associations)))
        middle = proc["blocks"][1]
        for anchor in (proc["blocks"][0], proc["blocks"][2]):
            observer, bare = (middle, anchor) if middle["instrumentation"] == "memcpy_only" else (anchor, middle)
            for idle in ("0", "20"):
                comparisons = {metric: abs(observer["focus"][idle][metric] / bare["focus"][idle][metric] - 1)
                               for metric in ("first_mean_ns", "later_mean_ns", "burst_per_copy_mean_ns")}
                proc["observer_discrepancies"].append(dict(observer_block=observer["block"], bare_block=bare["block"], requested_idle_ms=int(idle),
                                                           relative_discrepancy=comparisons,
                                                           below_five_percent_all_metrics=all(x <= .05 for x in comparisons.values()),
                                                           pure_observer_overhead_identified=False))
        result["processes"].append(proc)
    result["generator"] = campaign.receipt(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.out)
    output = Path(args.out) / "crossover-analysis.json"
    campaign.write(output, result)
    print(json.dumps(dict(output=campaign.receipt(output), processes=len(result["processes"]), performance_parameters_fitted=False)))


if __name__ == "__main__":
    main()
