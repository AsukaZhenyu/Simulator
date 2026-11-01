"""Independent stdlib-only replay audit for fixed-state shared-storage traces.

No simulator modules are imported.  This checks emitted physical storage and
event semantics against the saved graph, prices, goals and every snapshot. It
does not establish physical device accuracy or validate a policy's optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


class AuditError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise AuditError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _classes(refs):
    """Canonical alias relation independent of the emitted allocation names."""
    classes = {}
    for tensor, location, identity in refs:
        _need((tensor, location) not in classes, "duplicate tensor/location allocation reference")
        classes[(tensor, location)] = identity
    names = {}
    return tuple((tensor, location, names.setdefault(identity, len(names)))
                 for (tensor, location), identity in sorted(classes.items()))


def audit_documents(graph, events, snapshots, summary):
    """Replay one fixed-state result; raise AuditError on any inconsistency."""
    workload, scenario = graph["workload"], graph["scenario"]
    arch = scenario["architecture"]
    _need(arch.get("memory_mode") == "shared", "audit requires explicit shared mode")
    capacity, reserve = arch["shared_capacity_bytes"], arch.get("shared_runtime_reserved_bytes", 0)
    alignment = arch.get("shared_alignment_bytes", 1)
    tensors = {tensor["id"]: tensor for tensor in workload["tensors"]}
    operations = {op["id"]: op for op in workload["operations"]}
    sizes = {tid: ((tensor["storage_bytes"] + alignment - 1) // alignment) * alignment
             for tid, tensor in tensors.items()}
    _need(len(snapshots) == len(events) + 1, "one state per event plus initial state is required")
    status = {(tid, loc): ("READY" if loc in tensor.get("initial_locations", []) else "ABSENT")
              for tid, tensor in tensors.items() for loc in ("dram", "vram", "ssd")}
    op_status = {op: "NOT_STARTED" for op in operations}
    refs, next_id = {}, 0
    for tid in sorted(tensors):
        locations = [loc for loc in ("dram", "vram") if status[(tid, loc)] == "READY"]
        if locations:
            for loc in locations:
                refs[(tid, loc)] = next_id
            next_id += 1
    running, now, peak = [], 0, 0
    amounts = Counter()
    kinds = Counter()
    zero_release = actual_release = compute_count = share_count = 0

    def finish_at(t):
        nonlocal running, now
        _need(type(t) is int and t >= now, "event time must be a monotone integer")
        kept = []
        for task in running:
            if task["end"] > t:
                kept.append(task)
                continue
            for destination in task["destinations"]:
                status[destination] = "READY"
            if task.get("op") is not None:
                op_status[task["op"]] = "DONE"
        running, now = kept, t

    def physical_usage():
        by_id = {}
        for (tid, loc), identity in refs.items():
            _need(status[(tid, loc)] != "ABSENT", "dangling reference to absent tensor view")
            _need(identity not in by_id or by_id[identity] == tid,
                  "one physical allocation aliases different tensor values")
            by_id[identity] = tid
        return reserve + sum(sizes[tid] for tid in by_id.values()) + sum(task["workspace"] for task in running)

    def check_state(snapshot, index):
        nonlocal peak
        usage = physical_usage()
        _need(usage <= capacity, f"snapshot {index}: shared physical capacity exceeded")
        peak = max(peak, usage)
        _need(snapshot["t"] == now, f"snapshot {index}: incorrect timestamp")
        _need(snapshot.get("used_shared_bytes") == usage, f"snapshot {index}: incorrect physical usage")
        for name in ("used_dram_bytes", "used_vram_bytes"):
            _need(snapshot[name] == usage, f"snapshot {index}: logical view is not the shared physical pool")
        for loc, field in (("dram", "dram_status"), ("vram", "copy_status"), ("ssd", "ssd_status")):
            _need(dict(snapshot[field]) == {tid: status[(tid, loc)] for tid in tensors},
                  f"snapshot {index}: {loc} readiness differs from independent replay")
        _need(dict(snapshot["op_status"]) == op_status, f"snapshot {index}: operation status mismatch")
        actual = snapshot.get("storage_allocations", [])
        _need(_classes(actual) == _classes((tid, loc, identity) for (tid, loc), identity in refs.items()),
              f"snapshot {index}: allocation aliases or reference coverage differ")
        serial = snapshot.get("next_allocation_id")
        _need(type(serial) is int and serial == next_id, f"snapshot {index}: allocation serial mismatch")
        for _, _, identity in actual:
            _need(isinstance(identity, str) and identity.startswith("a") and identity[1:].isdigit()
                  and int(identity[1:]) < serial, f"snapshot {index}: allocation identity can collide with next allocation")

    check_state(snapshots[0], 0)
    for index, event in enumerate(events):
        kind, target = event["kind"], event.get("target_id")
        _need(event.get("action_index") == index, f"event {index}: action index mismatch")
        finish_at(event["t_ns"])
        kinds[kind] += 1
        _need(event.get("bytes_transferred", 0) >= 0, f"event {index}: negative payload")
        _need(event.get("bytes_released", 0) >= 0, f"event {index}: negative release")
        if kind == "ADVANCE":
            _need(event.get("resource") is None and event.get("bytes_transferred", 0) == 0,
                  f"event {index}: ADVANCE performs work")
        elif kind == "SHARE":
            src, dst = event.get("src"), event.get("dst")
            _need({src, dst} == {"dram", "vram"}, f"event {index}: SHARE not inside shared pool")
            _need(event.get("resource") is None and event["start_ns"] == now == event["end_ns"]
                  and event.get("bytes_transferred", 0) == 0 and event.get("workspace_bytes", 0) == 0,
                  f"event {index}: SHARE must be zero-time zero-payload access")
            _need(status[(target, src)] == "READY", f"event {index}: SHARE producer/source not ready")
            _need(status[(target, dst)] == "ABSENT", f"event {index}: SHARE overwrites a reference")
            route = f"{src}_to_{dst}"
            _need(target in scenario["mapspace"]["allowed_transfers"].get(route, []),
                  f"event {index}: SHARE route not allowed")
            for predecessor in scenario["mapspace"].get("transfer_dependencies", {}).get(target, []):
                _need(op_status[predecessor] == "DONE", f"event {index}: causal transfer dependency not done")
            refs[(target, dst)] = refs[(target, src)]
            status[(target, dst)] = "READY"
            share_count += 1
        elif kind == "EVICT":
            loc = event.get("location")
            _need(loc in ("dram", "vram") and status[(target, loc)] == "READY",
                  f"event {index}: eviction does not target a READY allocation")
            identity = refs[(target, loc)]
            _need(not any(identity in task["reads"] for task in running),
                  f"event {index}: physical buffer evicted during a reader")
            del refs[(target, loc)]
            released = 0 if identity in refs.values() else sizes[target]
            _need(event.get("bytes_released", 0) == released, f"event {index}: incorrect physical release")
            status[(target, loc)] = "ABSENT"
            zero_release += released == 0
            actual_release += released
        elif kind == "COMPUTE":
            _need(target in operations and op_status[target] == "NOT_STARTED", f"event {index}: invalid compute producer")
            device = event.get("device")
            _need(device in ("cpu", "gpu"), f"event {index}: unknown device")
            loc = "dram" if device == "cpu" else "vram"
            op = operations[target]
            _need(all(status[(tid, loc)] == "READY" for tid in set(op["inputs"])),
                  f"event {index}: compute input/producer is not ready")
            _need(device in scenario["mapspace"]["allowed_devices"][target], f"event {index}: device is not allowed")
            destinations = [(op["output"], loc)] + [(tid, loc) for tid in op.get("extra_outputs", [])]
            _need(all(all(status[(tid, domain)] == "ABSENT" for domain in ("dram", "vram")) for tid, _ in destinations),
                  f"event {index}: output overwrites another allocation")
            price = scenario["costs"]["compute"][target][device]
            workspace = ((price.get("workspace_bytes", 0) + alignment - 1) // alignment) * alignment
            _need(event.get("workspace_bytes", 0) == workspace, f"event {index}: workspace does not match declared cost")
            _need(event["start_ns"] == now and event["end_ns"] - now == price["duration_ns"],
                  f"event {index}: compute duration differs from fixed declared price")
            resource = f"{device}_compute"
            _need(event.get("resource") == resource, f"event {index}: compute resource mislabeled")
            _need(not any(task["resource"] == resource for task in running), f"event {index}: resource overlap")
            reads = {refs[(tid, loc)] for tid in set(op["inputs"])}
            for tid, domain in destinations:
                refs[(tid, domain)] = next_id
                next_id += 1
                status[(tid, domain)] = "RESERVED_OUTPUT"
            running.append(dict(resource=resource, end=event["end_ns"], destinations=destinations,
                                workspace=workspace, reads=reads, op=target))
            op_status[target] = "RUNNING"
            compute_count += 1
        elif kind in ("COPY", "READ_SSD", "WRITE_SSD"):
            src, dst = event.get("src"), event.get("dst")
            _need((src, dst) in (("dram", "vram"), ("vram", "dram"), ("ssd", "dram"), ("dram", "ssd")),
                  f"event {index}: unsupported copy route")
            _need(status[(target, src)] == "READY" and status[(target, dst)] == "ABSENT",
                  f"event {index}: real copy source/destination readiness invalid")
            _need(event["start_ns"] == now and event["end_ns"] > now, f"event {index}: real COPY is not positive duration")
            _need(event.get("bytes_transferred") == tensors[target]["storage_bytes"], f"event {index}: incorrect copy payload")
            route = f"{src}_to_{dst}"
            _need(target in scenario["mapspace"]["allowed_transfers"].get(route, []), f"event {index}: copy route not allowed")
            price = scenario["costs"]["transfer"].get(route, {}).get(target)
            _need(price is not None and event["end_ns"] - now == price["duration_ns"], f"event {index}: real copy cost unavailable or incorrect")
            resource = "ssd_io" if "ssd" in (src, dst) else "pcie_copy"
            _need(event.get("resource") == resource and not any(task["resource"] == resource for task in running),
                  f"event {index}: transfer resource overlap or incorrect binding")
            reads = set() if src == "ssd" else {refs[(target, src)]}
            if dst != "ssd":
                refs[(target, dst)] = next_id
                next_id += 1
            status[(target, dst)] = "RESERVED_COPY"
            running.append(dict(resource=resource, end=event["end_ns"], destinations=[(target, dst)],
                                workspace=0, reads=reads, op=None))
            amounts[route] += event["bytes_transferred"]
        else:
            raise AuditError(f"event {index}: unsupported event kind {kind!r}")
        check_state(snapshots[index + 1], index + 1)

    for name in ("peak_shared_bytes", "peak_dram_bytes", "peak_vram_bytes"):
        _need(summary.get(name) == peak, f"summary {name} differs from independently replayed peak")
    for name, route in (("h2d_bytes", "dram_to_vram"), ("d2h_bytes", "vram_to_dram"),
                        ("ssd_read_bytes", "ssd_to_dram"), ("ssd_write_bytes", "dram_to_ssd")):
        _need(summary.get(name, 0) == amounts[route], f"summary {name} does not conserve payload bytes")
    goal = not running and all(value == "DONE" for value in op_status.values()) and all(
        status[(tid, loc)] == "READY" for tid, loc in scenario["goal"]["output_locations"].items())
    if summary.get("success"):
        _need(goal, "successful summary has incomplete operation, running task or undelivered goal")
        _need(summary.get("completed_makespan_ns") == now, "summary makespan does not match final clock")
    return dict(pass_=True, events=len(events), compute_events=compute_count, share_events=share_count,
                zero_release_events=zero_release, physical_released_bytes=actual_release,
                independent_peak_shared_bytes=peak, goal_reached=goal, event_counts=dict(kinds))


def audit_case(directory):
    directory = Path(directory)
    graph = _read(directory / "full-audit/compiled_graph.json")["graph"]
    events = _read(directory / "events.json")
    snapshots = _read(directory / "full-audit/states.json")["states"]
    summary = _read(directory / "summary.json")
    result = audit_documents(graph, events, snapshots, summary)
    return {"scenario_id": summary.get("scenario_id"), "candidate_id": summary.get("candidate_id"),
            **result, "artifact_sha256": {name: _sha(directory / name) for name in
                ("full-audit/compiled_graph.json", "events.json", "full-audit/states.json", "summary.json")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for directory in sorted((args.campaign / "cases").glob("*/*")):
        if not (directory / "summary.json").is_file():
            continue
        try:
            rows.append(audit_case(directory))
        except (AuditError, KeyError, ValueError, OSError) as error:
            rows.append(dict(directory=str(directory), pass_=False,
                             error_type=type(error).__name__, error=str(error)))
    protocol_path, completion_path = args.campaign / "protocol.json", args.campaign / "completion.json"
    protocol = _read(protocol_path) if protocol_path.is_file() else {}
    completion = _read(completion_path) if completion_path.is_file() else {}
    expected = protocol.get("cases")
    identities = [(row.get("scenario_id"), row.get("candidate_id")) for row in rows]
    complete = bool(rows) and type(expected) is int and expected == len(rows) and completion.get("passed") is True
    complete = complete and len(identities) == len(set(identities))
    result = dict(format_version="shared-storage-independent-audit-0.1", complete=complete,
        independent_audit_script_sha256=_sha(Path(__file__)),
        passed=complete and all(row["pass_"] for row in rows), case_count=len(rows), expected_cases=expected,
        passed_cases=sum(row["pass_"] for row in rows),
        completeness_note="Full acceptance requires the declared case count and successful campaign completion marker; partial audit never claims completion.",
        campaign_evidence_sha256={path.name: _sha(path) for path in (protocol_path, completion_path) if path.is_file()},
        independent_of_simulator_runtime=True, physical_accuracy_claimed=False,
        scope="fixed-state shared-memory event/snapshot semantics; no environment traces or fixed slots",
        totals={key: sum(row.get(key, 0) for row in rows) for key in
                ("events", "compute_events", "share_events", "zero_release_events", "physical_released_bytes")},
        cases=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
