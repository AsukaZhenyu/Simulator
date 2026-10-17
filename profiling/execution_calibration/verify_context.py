"""Offline contract audit for the same-worker GPU-shard profile variant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as a
import run_dual as r
from verify_dual import verify as verify_dual


def verify(experiment):
    experiment = Path(experiment).resolve()
    result = verify_dual(experiment)
    manifest = r.document(experiment / "manifest.json")
    suite = r.document(experiment / "suite.json")
    rows = r.document(experiment / "measurements.json")
    if manifest["observed_child_flags"].get("NVIDIA_TF32_OVERRIDE") != "0":
        raise ValueError("context protocol requires observed NVIDIA_TF32_OVERRIDE=0")
    dispatch = r.dispatch_cpu_id(manifest["observed_child_flags"], suite["protocol"]["cpu_id"])
    context_sources = [(original, snapshot) for original, snapshot in manifest["source_snapshots"].items()
                       if Path(original).name == "compute_context.cpp"]
    if len(context_sources) != 1:
        raise ValueError("exactly one executed compute_context.cpp source snapshot is required")
    original, snapshot = context_sources[0]
    if (not Path(original).is_file() or a.sha(original) != snapshot["sha256"]
            or snapshot["sha256"] != manifest["inventory_before"].get(original)
            or a.sha(experiment / snapshot["path"]) != snapshot["sha256"]):
        raise ValueError("context native source differs from the executed snapshot and inventory")
    evidence = []
    for row in rows:
        if row["status"] != "measured":
            continue
        raw_path = experiment / "raw" / (row["id"] + ".json")
        raw = r.document(raw_path)
        payload = json.loads(raw["stdout"])
        if (payload.get("nvidia_tf32_override") != "0"
                or raw["observed_child_flags"].get("NVIDIA_TF32_OVERRIDE") != "0"):
            raise ValueError("native precision environment differs from the observed child flag")
        if row["case"]["family"] == "ffn":
            if (payload.get("profile_gpu_shard_thread") != "dispatch_worker_same_as_parallel_strategy"
                    or payload.get("profile_gpu_shard_affinity_mask") != 1 << dispatch
                    or payload.get("profile_gpu_shard_affinity_group") != 0):
                raise ValueError("FFN isolated GPU-shard profile did not use the declared strategy worker")
        elif (payload.get("profile_gpu_shard_thread") != "not_applicable_attention_has_no_shard"
              or payload.get("profile_gpu_shard_affinity_mask") is not None
              or payload.get("profile_gpu_shard_affinity_group") is not None):
            raise ValueError("Attention must explicitly declare no GPU-shard profiling worker")
        evidence.append({"row_id": row["id"], "family": row["case"]["family"], "split": row["split"],
                         "raw_path": str(raw_path), "raw_sha256": row["raw_sha256"],
                         "profile_role": payload["profile_gpu_shard_thread"],
                         "profile_affinity_mask": payload["profile_gpu_shard_affinity_mask"],
                         "profile_affinity_group": payload["profile_gpu_shard_affinity_group"]})
    result.update(schema="execution-context-verification-0.1",
                  same_profile_execution_context_verified=len(evidence) == len(rows) and bool(rows),
                  precision_environment_verified=True,
                  ffn_context_processes=sum(item["family"] == "ffn" for item in evidence),
                  attention_not_applicable_processes=sum(item["family"] == "attention" for item in evidence),
                  context_native_source_sha256=snapshot["sha256"],
                  context_native_source_snapshot=snapshot["path"], context_evidence=evidence,
                  context_audit_source_sha256=a.sha(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("verification output must be a new file")
        result = verify(args.experiment)
        a.dump(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "context_evidence"}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("execution context verification: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
