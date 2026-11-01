"""Derive the E/S decision -> derived actions -> consequence chain from retained runs.

Read-only over already-completed evidence. Two sources are consumed:

* a finished ``rerun_ready_selection`` output directory (``runs/<id>/execution``
  holds ``events.json``, ``summary.json``, ``compiled_graph.json``) plus its
  ``RESULTS.json`` for the frozen certificates;
* the 72-case E/S semantics ``CASES.csv`` tables (old grid vs re-run grid).

Nothing is re-simulated and no T* is re-solved. The unused-load test reuses the
definition frozen in ``build/es-abstraction-20261030/semantics/UNUSED_LOAD_WITNESS.json``:
a weight COPY into VRAM is unused when no COMPUTE that reads the tensor starts in
``[copy_end, next VRAM eviction)``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unused_loads(events, graph):
    """Weight loads that complete but are never read before eviction."""
    graph = graph.get("graph", graph) if isinstance(graph, dict) else graph
    tensors = {t["id"]: t for t in graph["workload"]["tensors"]}
    readers = {}
    for op in graph["workload"]["operations"]:
        for tensor in op["inputs"]:
            readers.setdefault(tensor, []).append(op["id"])
    starts = {e["target_id"]: e["t_ns"] for e in events if e["kind"] == "COMPUTE"}
    evict_at = {}
    for e in events:
        if e["kind"] == "EVICT" and e.get("location") == "vram":
            evict_at.setdefault(e["target_id"], []).append(e["t_ns"])
    unused = []
    for e in events:
        if e["kind"] != "COPY" or e.get("dst") != "vram":
            continue
        tensor = e["target_id"]
        if (tensors.get(tensor) or {}).get("role") != "weight":
            continue
        stop = next((t for t in sorted(evict_at.get(tensor, [])) if t >= e["end_ns"]), float("inf"))
        hit = [r for r in readers.get(tensor, []) if e["end_ns"] <= starts.get(r, -1) < stop]
        if not hit:
            unused.append(dict(tensor=tensor, start_ns=e["t_ns"], end_ns=e["end_ns"],
                               bytes=e.get("bytes_transferred"),
                               eviction_ns=None if stop == float("inf") else stop))
    return unused


def case_stats(execution):
    events = read(execution / "events.json")
    if isinstance(events, dict):
        events = events["events"]
    summary = read(execution / "summary.json")
    graph = read(execution / "compiled_graph.json")
    graph = graph.get("graph", graph)
    role = {t["id"]: t.get("role") for t in graph["workload"]["tensors"]}
    copies, copy_bytes, evicts = Counter(), Counter(), Counter()
    for e in events:
        if e["kind"] == "COPY":
            key = f"{e['src']}>{e['dst']}"
            copies[key] += 1
            copy_bytes[key] += e.get("bytes_transferred") or 0
        elif e["kind"] == "EVICT":
            evicts[role.get(e["target_id"])] += 1
    return dict(copies=dict(copies), copy_bytes=dict(copy_bytes), evicts=dict(evicts),
                peak_vram_bytes=summary["peak_vram_bytes"],
                peak_dram_bytes=summary["peak_dram_bytes"],
                h2d_bytes=summary["h2d_bytes"], d2h_bytes=summary["d2h_bytes"],
                makespan_ns=summary["makespan_ns"],
                unused=unused_loads(events, graph))


def corpus(runs, results):
    rows = {r["id"]: r for r in read(results)["rows"] if r["audit_status"] == "pass"}
    stats = {pid: case_stats(runs / pid / "execution") for pid in rows}
    pairs = []
    for pid, row in rows.items():
        if not pid.endswith("-demand"):
            continue
        other = pid[: -len("-demand")] + "-spare"
        if other not in rows:
            continue
        d, s = stats[pid], stats[other]
        cd, cs = row["certificate"], rows[other]["certificate"]
        pairs.append(dict(
            base=pid[: -len("-demand")], scenario=row["scenario"], regime=row["regime"],
            makespan_demand_ns=d["makespan_ns"], makespan_spare_ns=s["makespan_ns"],
            h2d_demand_bytes=d["h2d_bytes"], h2d_spare_bytes=s["h2d_bytes"],
            peak_vram_demand_bytes=d["peak_vram_bytes"], peak_vram_spare_bytes=s["peak_vram_bytes"],
            copies_demand=d["copies"], copies_spare=s["copies"],
            evicts_demand=d["evicts"], evicts_spare=s["evicts"],
            T_star_demand_ns=cd.get("T_star_ns"), T_star_spare_ns=cs.get("T_star_ns"),
            g_demand=cd.get("g"), g_spare=cs.get("g"),
            unused_demand=len(d["unused"]), unused_spare=len(s["unused"])))
    pairs.sort(key=lambda p: (-abs(p["makespan_demand_ns"] - p["makespan_spare_ns"]), p["base"]))
    totals = Counter()
    unused_cases = []
    for pid, st in stats.items():
        totals["copies"] += sum(st["copies"].values())
        totals["copy_bytes"] += sum(st["copy_bytes"].values())
        totals["evicts"] += sum(st["evicts"].values())
        if st["unused"]:
            unused_cases.append(dict(id=pid, loads=st["unused"]))
    return dict(
        policies=len(rows),
        totals=dict(totals),
        unused_load_cases=len(unused_cases),
        unused_load_count=sum(len(c["loads"]) for c in unused_cases),
        unused_load_bytes=sum(x["bytes"] for c in unused_cases for x in c["loads"]),
        unused_detail=unused_cases,
        pairs=dict(
            groups=len(pairs),
            change_makespan=sum(1 for p in pairs if p["makespan_demand_ns"] != p["makespan_spare_ns"]),
            change_h2d_bytes=sum(1 for p in pairs if p["h2d_demand_bytes"] != p["h2d_spare_bytes"]),
            change_peak_vram=sum(1 for p in pairs if p["peak_vram_demand_bytes"] != p["peak_vram_spare_bytes"]),
            change_certification=sum(1 for p in pairs
                                     if (p["T_star_demand_ns"] is None) != (p["T_star_spare_ns"] is None)),
            zero_difference=sum(1 for p in pairs if p["makespan_demand_ns"] == p["makespan_spare_ns"]),
            rows=pairs))


def semantics(old_cases, new_cases):
    def table(path):
        with open(path, encoding="utf-8", newline="") as handle:
            return {r["case"]: r for r in csv.DictReader(handle)}

    a, b = table(old_cases), table(new_cases)
    if set(a) != set(b):
        raise SystemExit("case sets differ between grids")
    changed = []
    for name in sorted(a):
        diff = {col: [a[name][col], b[name][col]]
                for col in ("status", "weight_copy_bytes", "weight_copy_count",
                            "peak_vram_bytes", "duration_ns") if a[name][col] != b[name][col]}
        if diff:
            changed.append(dict(case=name, **diff))
    groups = {}
    for name, row in b.items():
        groups.setdefault((row["gpu_width"], row["requests"]), []).append(name)
    flips = []
    for key, ids in sorted(groups.items()):
        ok = [i for i in ids if a[i]["status"] == "valid" and b[i]["status"] == "valid"]
        wa = min(ok, key=lambda i: int(a[i]["duration_ns"]))
        wb = min(ok, key=lambda i: int(b[i]["duration_ns"]))
        if wa != wb:
            flips.append(dict(gpu_width=key[0], requests=key[1], before=wa, after=wb))
    return dict(cases=len(a), changed_cases=len(changed), changes=changed,
                groups=len(groups), winner_changes=flips)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--old-cases", type=Path, required=True)
    parser.add_argument("--new-cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = dict(schema="es-derived-chain-1",
                  utc=datetime.now(timezone.utc).isoformat(),
                  sources=dict(runs=str(args.runs), results=str(args.results),
                               results_sha256=digest(args.results),
                               old_cases=str(args.old_cases), old_cases_sha256=digest(args.old_cases),
                               new_cases=str(args.new_cases), new_cases_sha256=digest(args.new_cases)),
                  corpus=corpus(args.runs, args.results),
                  semantics=semantics(args.old_cases, args.new_cases))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("corpus", "semantics")}, indent=2))
    pairs = {k: v for k, v in result["corpus"]["pairs"].items() if k != "rows"}
    print("corpus:", {k: v for k, v in result["corpus"].items()
                      if k not in ("pairs", "unused_detail")})
    print("pairs:", pairs)
    print("semantics:", result["semantics"]["changed_cases"], "changed,",
          len(result["semantics"]["winner_changes"]), "winner flips")


if __name__ == "__main__":
    main()
