"""Read-only conservative audit of immutable weight copies with no GPU consumer.

The GPU slot may represent a GPU or an explicitly hypothetical NPU substitute.
This post-run evaluation never feeds future consumer information to mapping.
It cannot attribute all prefetch waste or determine first-demand cache hits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_case(folder):
    summary = _read(folder / "summary.json")
    required = ("events.json", "operation-catalogue.json", "tensor-catalogue.json", "summary.json")
    row = {name: summary[name] for name in ("system_id", "variant_id", "workload_id", "candidate_id")}
    row.update(success=summary.get("success", False),
               completed_makespan_ns=summary.get("completed_makespan_ns"),
               source_artifact_hashes={name: _sha(folder / name) for name in required if (folder / name).exists()})
    if not all((folder / name).exists() for name in required):
        return dict(row, evaluated=False, reason="no complete event/catalogue evidence")
    events = _read(folder / "events.json")
    operations = {operation["id"]: operation for operation in _read(folder / "operation-catalogue.json")}
    weights = {tensor["id"] for tensor in _read(folder / "tensor-catalogue.json") if tensor["role"] == "weight"}
    consumed = set()
    for event in events:
        if event["kind"] == "COMPUTE" and event.get("device") == "gpu":
            consumed.update(identity for identity in operations[event["target_id"]]["inputs"] if identity in weights)
    copies = [event for event in events if event["kind"] == "COPY" and event.get("dst") == "vram"
              and event["target_id"] in weights]
    unused = [event for event in copies if event["target_id"] not in consumed]
    per_tensor = Counter()
    per_tensor_counts = Counter()
    for event in unused:
        per_tensor[event["target_id"]] += event["bytes_transferred"]
        per_tensor_counts[event["target_id"]] += 1
    row.update(evaluated=True,
        total_weight_copy_to_accelerator_bytes=sum(event["bytes_transferred"] for event in copies),
        total_weight_copy_to_accelerator_count=len(copies),
        weight_copy_bytes_with_no_accelerator_consumer=sum(event["bytes_transferred"] for event in unused),
        weight_copy_count_with_no_accelerator_consumer=len(unused),
        unused_weight_copy_bytes_by_tensor=dict(sorted(per_tensor.items())),
        unused_weight_copy_count_by_tensor=dict(sorted(per_tensor_counts.items())))
    return row


def audit(root):
    root = Path(root)
    rows = [audit_case(path.parent) for path in sorted((root / "cases").rglob("summary.json"))]
    indexed = {(row["system_id"], row["variant_id"], row["workload_id"], row["candidate_id"]): row for row in rows}
    pairs = []
    for row in rows:
        if not row["candidate_id"].endswith("-s1"):
            continue
        reference = indexed.get((row["system_id"], row["variant_id"], row["workload_id"],
                                 row["candidate_id"][:-1] + "0"))
        if not reference or not row.get("evaluated") or not reference.get("evaluated"):
            continue
        pairs.append(dict(system_id=row["system_id"], variant_id=row["variant_id"],
            workload_id=row["workload_id"], execution=row["candidate_id"].split("-")[0],
            no_prefetch_completed_makespan_ns=reference.get("completed_makespan_ns"),
            prefetch_completed_makespan_ns=row.get("completed_makespan_ns"),
            no_prefetch_unused_weight_copy_bytes=reference["weight_copy_bytes_with_no_accelerator_consumer"],
            prefetch_unused_weight_copy_bytes=row["weight_copy_bytes_with_no_accelerator_consumer"],
            extra_unused_weight_copy_bytes=(row["weight_copy_bytes_with_no_accelerator_consumer"]
                                           - reference["weight_copy_bytes_with_no_accelerator_consumer"])))
    return dict(format_version="hardware-unused-weight-copy-audit-0.1", cases=len(rows),
        campaign_complete=(root / "completion.json").exists(),
        scope=("immutable weight physical COPY to accelerator slot with no actual accelerator COMPUTE "
               "consumer anywhere in this finite run; conservative subset, not individual-copy usefulness"),
        causal_policy_unchanged=True, future_information_used_only_after_execution=True,
        not_claimed=["first-demand cache hit rate", "all speculative waste", "real hardware performance",
                     "causal proof that unused traffic alone explains total latency difference"],
        evaluated_cases=sum(row.get("evaluated", False) for row in rows),
        storage_pairs=pairs, records=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(dict(cases=result["cases"], evaluated=result["evaluated_cases"],
                          pairs=len(result["storage_pairs"]), campaign_complete=result["campaign_complete"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
