"""Post-run audit of immutable weight traffic in the finite design campaign.

Future consumer information is used only by this evaluator, never by mapping.
This conservative metric counts copies of weights with no GPU compute consumer
anywhere in the finite trace; it is not a per-copy cache-hit or usefulness metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

FIELDS = ("workload", "decode_steps", "initial_state", "vram_gib", "link_gbs")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unused_weight_copies(events, operations, tensors):
    ops = {op["id"]: op for op in operations}
    weights = {t["id"] for t in tensors if t["role"] == "weight"}
    consumed = set()
    for event in events:
        if event["kind"] == "COMPUTE" and event.get("device") == "gpu":
            consumed.update(set(ops[event["target_id"]]["inputs"]) & weights)
    copied = [e for e in events if e["kind"] == "COPY" and e.get("dst") == "vram"
              and e["target_id"] in weights]
    unused = [e for e in copied if e["target_id"] not in consumed]
    by_tensor = Counter()
    for event in unused:
        by_tensor[event["target_id"]] += event["bytes_transferred"]
    return {
        "physical_weight_copy_to_gpu_bytes": sum(e["bytes_transferred"] for e in copied),
        "bytes_without_gpu_consumer": sum(by_tensor.values()),
        "copies_without_gpu_consumer": len(unused),
        "bytes_by_tensor_without_gpu_consumer": dict(sorted(by_tensor.items())),
    }


def audit(campaigns):
    records, source_hashes = [], {}
    for directory in campaigns:
        directory = directory.resolve()
        receipt = read(directory / "completion.json")
        require(receipt.get("complete") is True, "campaign incomplete")
        rows = read(directory / "records.json")
        require(len(rows) == receipt["cases"], "campaign denominator changed")
        require(canonical(rows) == receipt["records_sha256"], "campaign records changed")
        source_hashes[str(directory / "records.json")] = sha(directory / "records.json")
        for row in rows:
            result = {k: row[k] for k in FIELDS + ("candidate", "status", "success")}
            result.update(completed_makespan_ns=row["completed_makespan_ns"],
                          h2d_bytes=row.get("h2d_bytes"), evaluated=False)
            if row["success"]:
                folder = (directory / row["artifact_directory"]).resolve()
                require(folder.is_relative_to(directory), "artifact path leaves campaign")
                manifest = read(folder / "artifacts.json")
                evidence = {}
                for name in ("events.json", "operation-catalogue.json", "tensor-catalogue.json"):
                    require(sha(folder / name) == manifest[name], "case evidence changed")
                    evidence[name] = read(folder / name)
                    source_hashes[str(folder / name)] = manifest[name]
                require(canonical(evidence["events.json"]) == row["event_sha256"], "event receipt changed")
                result.update(unused_weight_copies(evidence["events.json"],
                    evidence["operation-catalogue.json"], evidence["tensor-catalogue.json"]), evaluated=True)
            records.append(result)
    indexed = {tuple(r[k] for k in FIELDS) + (r["candidate"],): r for r in records}
    require(len(indexed) == len(records), "duplicate case")
    pairs = []
    for key, row in indexed.items():
        if not row["candidate"].endswith("-s1"):
            continue
        reference = indexed.get(key[:-1] + (row["candidate"][:-1] + "0",))
        require(reference is not None, "unmatched prefetch case")
        pair = {k: row[k] for k in FIELDS}
        pair.update(execution=row["candidate"].split("-")[0],
                    evaluated=row["evaluated"] and reference["evaluated"])
        if pair["evaluated"]:
            pair.update(prefetch_latency_ns=row["completed_makespan_ns"],
                no_prefetch_latency_ns=reference["completed_makespan_ns"],
                latency_increase=row["completed_makespan_ns"] / reference["completed_makespan_ns"] - 1,
                extra_h2d_bytes=row["h2d_bytes"] - reference["h2d_bytes"],
                prefetch_bytes_without_gpu_consumer=row["bytes_without_gpu_consumer"],
                no_prefetch_bytes_without_gpu_consumer=reference["bytes_without_gpu_consumer"],
                extra_bytes_without_gpu_consumer=row["bytes_without_gpu_consumer"] - reference["bytes_without_gpu_consumer"])
        pairs.append(pair)
    return {
        "schema": "paper-prefetch-consumer-audit-1", "completed": True,
        "source_sha256": sha(Path(__file__)), "source_artifact_hashes": source_hashes,
        "cases": len(records), "pairs": len(pairs),
        "evaluated_pairs": sum(p["evaluated"] for p in pairs),
        "future_information_only_in_post_run_evaluator": True,
        "scope": "Physical immutable-weight copies to GPU whose tensor has no GPU COMPUTE consumer anywhere in the finite trace.",
        "not_claimed": ["all speculative waste", "per-copy usefulness or cache hits", "physical hardware performance",
                        "latency change caused solely by unused bytes", "benefits beyond the finite terminal boundary"],
        "records": records, "storage_pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("completed", "cases", "pairs", "evaluated_pairs")}))


if __name__ == "__main__":
    main()
