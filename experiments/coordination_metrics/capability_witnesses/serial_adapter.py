"""New, counted coordination mechanism for the two external frontends.

This is a serial, dependency-preserving executor of a supplied mathematical DAG.
It is not an upstream scheduler and does not implement concurrent prefetch,
P/D barriers, alias views, numerical kernels, or a physical timing model.
It never reads a production compiler's IR, event trace, or aggregate results.
"""
from collections import OrderedDict
import re


def run_declared_graph(graph, policy, native_check, capacity):
    tensors = graph["tensors"]
    locations = {tid: set(t.get("initial_locations", ())) for tid, t in tensors.items()}
    pending = dict(graph["operations"])
    events = []
    resident = OrderedDict()
    cache_bytes = 0
    peak_gpu = 0
    native_calls = []
    completed = set()
    goals = set(graph["terminal_hidden"] + graph["expected_all_kv"])

    def check_capacity():
        nonlocal peak_gpu
        used = sum(tensors[t]["storage_bytes"] for t, locs in locations.items() if "vram" in locs)
        peak_gpu = max(peak_gpu, used)
        if used > capacity:
            raise ValueError("declared physical VRAM capacity exceeded")

    def copy(tid, dst, operation):
        if dst in locations[tid]:
            return
        if not locations[tid]:
            raise ValueError("operand has no completed producer: " + tid)
        src = sorted(locations[tid])[0]
        locations[tid].add(dst)
        events.append(dict(kind="COPY", tensor=tid, src=src, dst=dst,
                           bytes=tensors[tid]["storage_bytes"], consumer=operation))
        check_capacity()

    def evict_group(key):
        nonlocal cache_bytes
        members = resident.pop(key)
        for tid in members:
            if "vram" in locations[tid]:
                locations[tid].remove("vram")
                events.append(dict(kind="EVICT", tensor=tid, group=key))
        cache_bytes -= sum(tensors[tid]["storage_bytes"] for tid in members)

    while pending:
        ready = [op for op in pending.values() if all(locations[t] for t in op["inputs"])]
        if not ready:
            raise ValueError("no progress in declared dependency graph")
        op = ready[0]
        oid = op["id"]
        kind = op["semantic_op"]
        device = "cpu" if kind == "state_update" else policy["placements"].get(kind, policy["default_device"])
        dst = "dram" if device == "cpu" else "vram"
        weights = tuple(t for t in op["inputs"] if tensors[t]["role"] == "weight")
        match = next((re.search(r"w:(layer\d+):E(\d+)", t) for t in weights if ":E" in t), None)
        key = None if match is None else ":".join(match.groups())
        if device == "gpu" and key is not None and policy["cache"] == "lru":
            if key in resident:
                resident.move_to_end(key)
            else:
                size = sum(tensors[t]["storage_bytes"] for t in weights)
                if size > policy["budget_bytes"]:
                    raise ValueError("expert exceeds declared cache budget")
                while cache_bytes + size > policy["budget_bytes"]:
                    evict_group(next(iter(resident)))
                resident[key] = weights
                cache_bytes += size
        for tid in op["inputs"]:
            copy(tid, dst, oid)
        native_calls.append(native_check(op, tensors, graph["contexts"].get(oid, {}), device))
        outputs = [op["output"], *op.get("extra_outputs", ())]
        for tid in outputs:
            locations[tid].add(dst)
        events.append(dict(kind="COMPUTE", operation=oid, semantic_op=kind, device=device,
                           inputs=list(op["inputs"]), outputs=outputs))
        check_capacity()
        del pending[oid]
        completed.add(oid)
        if policy["cache"] == "none":
            for tid in weights:
                if "vram" in locations[tid]:
                    locations[tid].remove("vram")
                    events.append(dict(kind="EVICT", tensor=tid, group=None))
        for tid in outputs:
            if tid in goals:
                copy(tid, "dram", oid)
        # This is lifetime safety in the evaluator, not an expert policy's
        # future-route oracle. Only LRU's past resident queue chooses victims.
        needed = {tid for p in pending.values() for tid in p["inputs"]} | goals
        for tid in tuple(locations):
            if tensors[tid]["role"] != "weight" and tid not in needed:
                locations[tid].clear()
    if not all("dram" in locations[t] for t in goals):
        raise ValueError("terminal DRAM goal not satisfied")
    return dict(events=events, completed_operations=sorted(completed), native_calls=native_calls,
                goals_in_dram=sorted(goals), peak_vram_bytes=peak_gpu,
                cache_resident_groups=list(resident), cache_budget_bytes=policy["budget_bytes"],
                mechanism="new_counted_serial_coordination_adapter")
