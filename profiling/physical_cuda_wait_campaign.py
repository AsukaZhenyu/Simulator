"""Offline, write-once CUDA interval diagnostics for selected physical jobs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from profiling.available_service.physical_records import load_physical_records
from profiling import physical_cuda_waits


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _stats(values):
    values = [value for value in values if value is not None]
    ordered = sorted(values)
    return dict(n=len(values), mean_ns=mean(values) if values else None, median_ns=median(values) if values else None,
        p90_ns=ordered[ceil(.90*len(ordered))-1] if ordered else None,
        min_ns=min(values) if values else None, max_ns=max(values) if values else None)


def _summarize(actions):
    pairs = [pair for row in actions for pair in row["launch_kernel_pairs"]]
    syncs = [api for row in actions for api in row["synchronization_api_records"]]
    return dict(action_invocations=len(actions), launch_kernel_pairs=len(pairs),
        pair_metrology_qualified=sum(pair["pair_metrology_qualified"] for pair in pairs),
        kernel_begins_before_launch_return=sum(pair["kernel_begins_before_launch_api_return"] for pair in pairs),
        kernel_begins_before_launch_entry=sum(pair["kernel_begins_before_launch_api_entry"] for pair in pairs),
        unpaired_kernels=sum(len(row["unpaired_kernel_records"]) for row in actions),
        unpaired_launch_APIs=sum(len(row["unpaired_launch_api_records"]) for row in actions),
        per_kernel_queue_after_launch_lower_bound=_stats([p["queue_after_launch_lower_bound_ns"] for p in pairs]),
        per_launch_pair_launch_API_span=_stats([p["launch_api_span_ns"] for p in pairs]),
        per_kernel_activity_envelope=_stats([p["kernel_activity_envelope_ns"] for p in pairs]),
        per_sync_API_span=_stats([row["end_ns"]-row["begin_ns"] for row in syncs if row.get("timestamp_valid") is True]),
        per_action_host_span=_stats([row["end_ns"]-row["begin_ns"] for row in actions]),
        per_action_launch_API_union=_stats([row["launch_api_union_ns"] for row in actions]),
        per_action_sync_API_union=_stats([row["synchronization_api_union_ns"] for row in actions]),
        per_action_other_runtime_API_union=_stats([row["other_runtime_api_union_ns"] for row in actions]),
        per_action_driver_API_union=_stats([row["driver_api_union_ns"] for row in actions]),
        per_action_device_activity_union=_stats([row["device_activity_union_ns"] for row in actions]),
        per_action_sync_API_time_overlapping_attributed_device=_stats([row["sync_API_time_overlapping_attributed_device_activity_ns"] for row in actions]),
        per_action_sync_API_time_without_attributed_device=_stats([row["sync_API_time_without_attributed_device_activity_ns"] for row in actions]),
        per_action_post_last_kernel_tail=_stats([row["synchronization_completion"]["post_last_kernel_tail_ns"] for row in actions]),
        per_action_post_last_device_tail=_stats([row["synchronization_completion"]["post_last_device_tail_ns"] for row in actions]),
        synchronization_dependency_identified=False, timing_columns_additive=False)


def diagnose_campaign(root):
    root = Path(root).resolve()
    audit_path = root/"campaign-audit.json"
    audit = _read(audit_path)
    jobs = [row for row in audit["jobs"] if row.get("instrumentation") == "cupti" and
            (row["job_id"].startswith("profile-0-") or row["split"] == "heldout")]
    out = root/"cuda-waits"
    out.mkdir(exist_ok=False)
    results = []
    for job in jobs:
        result_path = out/(job["job_id"]+".json")
        execution_path = root/"jobs"/job["job_id"]/"execution.json"
        execution = _read(execution_path)
        raw = execution["raw_files"]["main"]
        program = execution["program"]
        row = dict(job_id=job["job_id"], split=job["split"], condition=execution["job"]["condition"],
            diagnostic_campaign_admitted=job["diagnostic_included"],
            posttarget_cleanup_applied=execution.get("posttarget_cleanup_applied", False),
            raw=dict(raw), execution_sha256=_sha(execution_path))
        try:
            if _sha(program["path"]) != program["sha256"]:
                raise ValueError("bound program source differs")
            parsed = load_physical_records(raw["path"], raw["sha256"], program=_read(program["path"]))
            diagnostic = physical_cuda_waits.diagnose_cuda_waits(parsed)
            diagnostic["job"] = dict(execution["job"])
            diagnostic["source_path"] = raw["path"]
            diagnostic["source_sha256"] = raw["sha256"]
            _save(result_path, diagnostic)
            groups = defaultdict(list)
            for action in diagnostic["actions"]:
                groups[action["action_index"]].append(action)
            summaries = []
            for index, actions in sorted(groups.items()):
                first = actions[0]
                summaries.append(dict(action_index=index, kind=first["kind"], device=first["device"],
                    target_id=first["target_id"], metrics=_summarize(actions)))
            gpu_actions = [action for action in diagnostic["actions"] if action["device"] == "gpu" and action["kind"] == "COMPUTE"]
            row.update(status="described", process_id=parsed["process_id"],
                capture_qualified=parsed["capture_qualified"], capture_issues=parsed["capture_issues"],
                selected_action_issues=parsed["selected_action_issues"], orphan_records=len(parsed["orphan_records"]),
                invalid_time_records=len(parsed["invalid_time_records"]), counts=diagnostic["counts"],
                all_actions_metrics=_summarize(diagnostic["actions"]), gpu_COMPUTE_metrics=_summarize(gpu_actions),
                per_action=summaries, result=dict(path=str(result_path), sha256=_sha(result_path)))
        except (OSError, ValueError, TypeError, KeyError) as error:
            row.update(status="failed_to_describe", error=str(error), capture_qualified=False)
            if not result_path.exists():
                _save(result_path, row)
        results.append(row)
        print(json.dumps(dict(job=row["job_id"], status=row["status"], capture_qualified=row["capture_qualified"])), flush=True)
    source_paths = [Path(__file__), Path(physical_cuda_waits.__file__),
                    Path(__file__).parent/"available_service/physical_records.py"]
    summary = dict(schema="physical-cuda-wait-campaign-summary-1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        campaign_root=str(root), campaign_audit=dict(path=str(audit_path), sha256=_sha(audit_path)),
        source_receipts=[dict(path=str(path.resolve()), sha256=_sha(path)) for path in source_paths],
        counts=dict(selected_jobs=len(jobs), described_jobs=sum(row["status"] == "described" for row in results),
                    capture_qualified=sum(row["capture_qualified"] for row in results)),
        statistics_unit="nanoseconds; per-kernel/launch-pair, per-sync API call and per-action invocation are separately labeled; P90 is nearest-rank",
        timing_columns_additive=False, predictions_fitted=False, hardware_accuracy_qualified=False,
        limitations=["Sync API time without attributed device activity is not pure scheduling/runtime overhead.",
            "Kernel-to-synchronization dependencies are not established by available API records.",
            "CUPTI observer gate failed; these describe instrumented execution and cannot validate bare performance.",
            "Kernel envelopes do not prove SM-active or preemption-free execution.",
            "All kernel and action observations retained; different groups have different aggregation units."],
        jobs=results)
    _save(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = diagnose_campaign(args.root)
    print(json.dumps(summary["counts"]))
